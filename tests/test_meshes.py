"""The mesh path, on a model small enough to test in seconds.

Until this file existed the only mesh coverage was the 25-minute G1 export, so
a bug in mesh -> B-rep conversion could sit undetected through a whole green
test run.  The fixture deliberately includes a mesh made of two disconnected
pieces, which is what broke four G1 links.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sim2cad.meshes import connected_components, extract_mesh, mesh_to_shape
from sim2cad.stepread import read_step_assembly, solid_volume
from sim2cad.validate import validate_export


def _geom(lm, name):
    import mujoco

    return mujoco.mj_name2id(lm.model, mujoco.mjtObj.mjOBJ_GEOM, name)


def _load(path):
    from sim2cad.load import load_model, set_world_fix

    lm = load_model(mjcf=str(path))
    set_world_fix(lm)
    return lm


# -- component splitting ---------------------------------------------------


def test_connected_components_counts_pieces():
    # Two triangles sharing an edge are one piece; a third apart is another.
    faces = np.array([[0, 1, 2], [2, 1, 3], [4, 5, 6]])
    groups = connected_components(faces)
    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == [1, 2]

    one = connected_components(np.array([[0, 1, 2]]))
    assert len(one) == 1 and len(one[0]) == 1


def test_components_split_at_a_shared_vertex():
    """Sharing one vertex is not sharing an edge, and shells need edges.

    OpenCascade cannot represent two vertex-joined patches as a single shell:
    it splits them when writing STEP and duplicates the bridging faces, so the
    file comes back with more surface than it was handed.
    """
    # Two triangles meeting only at vertex 2.
    faces = np.array([[0, 1, 2], [2, 3, 4]])
    groups = connected_components(faces)
    assert len(groups) == 2, "a shared vertex must not merge two pieces"

    # Sharing a full edge does merge them.
    joined = connected_components(np.array([[0, 1, 2], [1, 2, 3]]))
    assert len(joined) == 1


def test_vertex_joined_mesh_round_trips_through_step(mesh_model, tmp_path):
    """The ANYmal thigh bug, reproduced on two boxes touching at one corner."""
    import cadquery as cq
    from cadquery.occ_impl.exporters.assembly import exportAssembly

    from sim2cad.occunits import cascade_unit
    from sim2cad.stepread import count_faces, read_step_assembly, surface_centroid

    lm = _load(mesh_model)
    mesh = extract_mesh(lm.model, _geom(lm, "g_corner"), scale=1000.0)
    assert mesh.n_components == 2, "the two boxes share a vertex, not an edge"

    build = mesh_to_shape(mesh)
    written_faces = count_faces(build.shape)
    _, written_area = surface_centroid(build.shape)

    assembly = cq.Assembly(name="t")
    assembly.add(cq.Shape.cast(build.shape), name="corner")
    path = tmp_path / "corner.step"
    with cascade_unit("MM"):
        exportAssembly(assembly, str(path), unit="MM")

    node, _ = next(read_step_assembly(str(path), unit="MM").leaves())
    _, read_area = surface_centroid(node.shape)
    assert count_faces(node.shape) == written_faces, "STEP gained or lost faces"
    assert np.isclose(read_area, written_area, rtol=1e-12), "STEP changed the surface"

    # Two 60 mm cubes: 12 faces each after merging, and twice the volume.
    assert np.isclose(build.volume_after_merge, 2 * 60**3, rtol=1e-6)


def test_single_component_mesh_is_one_exact_solid(mesh_model):
    lm = _load(mesh_model)
    mesh = extract_mesh(lm.model, _geom(lm, "g_single"), scale=1000.0)
    assert mesh.n_components == 1
    assert mesh.watertight

    build = mesh_to_shape(mesh)
    assert build.is_solid and build.valid
    # 100 x 120 x 140 mm.  MuJoCo stores mesh_vert as float32, so the exact
    # analytic answer is only reachable to single-precision relative error.
    assert np.isclose(build.volume_after_merge, 100 * 120 * 140, rtol=1e-6)
    # A box's six sides are 12 triangles, merged back into 6 planar faces.
    assert build.faces_before == 12
    assert build.faces_after == 6


def test_two_piece_mesh_becomes_two_solids_with_the_full_volume(mesh_model):
    """The G1 bug, reproduced in a second rather than 25 minutes.

    One mesh holding two disconnected boxes cannot be a single TopoDS_Solid:
    OpenCascade reads the second shell as a cavity, so the volume comes back
    short and BRepCheck rejects the result.  Each piece must be its own solid.
    """
    lm = _load(mesh_model)
    mesh = extract_mesh(lm.model, _geom(lm, "g_two"), scale=1000.0)
    assert mesh.n_components == 2, "fixture should hold two disconnected boxes"
    assert mesh.watertight

    build = mesh_to_shape(mesh)
    assert build.n_components == 2
    assert build.is_solid and build.valid

    expected = 80**3 + 40**3
    assert np.isclose(build.volume_after_merge, expected, rtol=1e-6), (
        f"got {build.volume_after_merge}, expected {expected}"
    )
    # The smaller box is 11% of the total: exactly the kind of shortfall that
    # treating the two as one solid would have produced.
    assert 40**3 / expected > 0.05
    assert any("disconnected" in n for n in build.notes)


def test_open_mesh_becomes_a_shell_and_says_so(mesh_model):
    """With repair off, an open surface must come out as a valid open shell."""
    lm = _load(mesh_model)
    mesh = extract_mesh(lm.model, _geom(lm, "g_open"), scale=1000.0, repair=False)
    assert not mesh.watertight, "fixture mesh should be missing its top face"

    build = mesh_to_shape(mesh)
    assert not build.is_solid
    assert build.valid, "an open shell should still be a valid shape"
    assert any("not watertight" in n for n in build.notes)
    assert np.isnan(build.volume_after_merge), "an open shell has no volume"


def test_hole_filling_closes_a_simple_hole(mesh_model):
    """And with repair on, trimesh caps it and we get the solid back."""
    lm = _load(mesh_model)
    mesh = extract_mesh(lm.model, _geom(lm, "g_open"), scale=1000.0, repair=True)
    assert mesh.repaired and mesh.watertight

    build = mesh_to_shape(mesh)
    assert build.is_solid and build.valid
    assert np.isclose(build.volume_after_merge, 60**3, rtol=1e-6)


def test_merge_coplanar_preserves_volume_exactly(mesh_model):
    lm = _load(mesh_model)
    for name in ("g_single", "g_two"):
        mesh = extract_mesh(lm.model, _geom(lm, name), scale=1000.0)
        merged = mesh_to_shape(mesh, merge_coplanar=True)
        plain = mesh_to_shape(mesh, merge_coplanar=False)
        assert np.isclose(
            merged.volume_after_merge, plain.volume_after_merge, rtol=1e-12
        ), name  # merging must not change the volume at all
        assert merged.faces_after < plain.faces_after, f"{name} should merge faces"


def test_mesh_volume_matches_trimesh(mesh_model):
    lm = _load(mesh_model)
    for name in ("g_single", "g_two"):
        mesh = extract_mesh(lm.model, _geom(lm, name), scale=1000.0)
        build = mesh_to_shape(mesh)
        assert np.isclose(build.volume_after_merge, mesh.volume, rtol=1e-9), name


# -- the whole pipeline on meshes -----------------------------------------


def test_mesh_export_passes_every_check(mesh_export):
    report = validate_export(mesh_export)
    failures = [f"{c.name}: {c.detail}" for c in report.checks if not (c.passed or c.skipped)]
    assert not failures, failures

    # The mesh checks must actually have run, not been skipped for want of
    # mesh parts -- that is the failure mode this file exists to prevent.
    for prefix in ("3a.", "3b.", "4a.", "4c."):
        check = next(c for c in report.checks if c.name.startswith(prefix))
        assert not check.skipped, f"{check.name} was skipped"


def test_exported_step_round_trips_the_full_volume(mesh_export):
    """Read the written file back and check the two-piece link is all there."""
    root = read_step_assembly(str(mesh_export.assembly_path), unit="MM")
    leaves = {n.name: (n, w) for n, w in root.leaves()}
    name = next(n for n in leaves if n.startswith("arm__"))
    node, world = leaves[name]
    placed = node.shape.Moved(world.to_location().wrapped)
    assert np.isclose(solid_volume(placed), 80**3 + 40**3, rtol=1e-6)


def test_report_records_the_component_count(mesh_export):
    report = validate_export(mesh_export)
    row = next(p for p in report.parts if p.body == "arm")
    assert row.n_components == 2
    assert row.geom_type == "mesh"

    single = next(p for p in report.parts if p.body == "base")
    assert single.n_components == 1


def test_recentering_applies_to_this_model_too(mesh_model):
    """The fixture is rotated and offset, so the compiler recenters it."""
    import mujoco

    from sim2cad.frames import Pose

    lm = _load(mesh_model)
    model = lm.model
    moved = 0
    for name in ("g_single", "g_two", "g_open"):
        geom = _geom(lm, name)
        mesh_id = int(model.geom_dataid[geom])
        recentre = Pose.from_mj(model.mesh_pos[mesh_id], model.mesh_quat[mesh_id])
        if np.linalg.norm(recentre.pos) > 1e-9:
            moved += 1
    assert moved, "expected the compiler to recenter at least one mesh"


# -- parallel vs serial ----------------------------------------------------


def test_parallel_build_matches_serial_exactly(mesh_model, tmp_path):
    """Workers hand shapes back as .brep; that must change nothing at all.

    If this ever diverges, the parallel path is silently producing different
    geometry from the one every other test checks.
    """
    from sim2cad.export import run_export
    from sim2cad.meshes import _volume

    serial = run_export(
        mjcf=str(mesh_model), out=str(tmp_path / "s"), jobs=1, write_render=False
    )
    parallel = run_export(
        mjcf=str(mesh_model), out=str(tmp_path / "p"), jobs=4, write_render=False
    )
    assert not any("parallel mesh build unavailable" in w for w in parallel.warnings), (
        parallel.warnings
    )

    by_name_s = {p.safe_name: p for p in serial.robot.parts}
    by_name_p = {p.safe_name: p for p in parallel.robot.parts}
    assert set(by_name_s) == set(by_name_p)

    for name, a in by_name_s.items():
        b = by_name_p[name]
        assert a.mesh_build.faces_after == b.mesh_build.faces_after, name
        assert a.mesh_build.n_components == b.mesh_build.n_components, name
        assert a.mesh_build.is_solid == b.mesh_build.is_solid, name
        assert a.mesh_build.valid == b.mesh_build.valid, name
        # .brep is a decimal text format, so the round trip is lossless to a
        # few ULP rather than bit-identical.  Anything larger means the
        # parallel path is building different geometry.
        assert np.isclose(_volume(a.shape), _volume(b.shape), rtol=1e-12, atol=0.0), name
        assert a.geom_from_body.allclose(b.geom_from_body, atol=0.0), name

    assert validate_export(parallel).passed


def test_jobs_one_is_the_serial_path(mesh_model, tmp_path):
    from sim2cad.export import run_export

    result = run_export(
        mjcf=str(mesh_model), out=str(tmp_path / "o"), jobs=1, write_render=False
    )
    assert result.options["jobs"] == 1
    assert not result.warnings or all("parallel" not in w for w in result.warnings)
