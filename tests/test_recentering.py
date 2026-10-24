"""§8.2 -- the mesh recentering pitfall, proved rather than asserted.

MuJoCo's compiler moves each mesh so its frame sits at the mesh's own centre,
and folds the inverse of that move into the geom's pos/quat.  These tests show
three things on the real G1:

1. the compiled ``mesh_vert`` really does live in the geom's frame;
2. the compiled geom pose is exactly ``authored_pose @ recentering``;
3. so placing the *original* STL with the *authored* pos/quat -- the obvious
   thing to do -- is wrong, by up to 429 mm on this robot.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
import pytest

from sim2cad.frames import Pose


def _mesh_geoms(model):
    return [g for g in range(model.ngeom) if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]


def _compiled_verts(model, geom_id):
    mesh = int(model.geom_dataid[geom_id])
    adr, num = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    return np.asarray(model.mesh_vert[adr : adr + num], dtype=float).reshape(-1, 3)


def test_compiled_mesh_vertices_live_in_the_geom_frame(g1_model):
    """``geom_aabb`` is MuJoCo's own bounds for the geom, in the geom frame.

    If our slice of ``mesh_vert`` agrees with it, the vertices are in the frame
    we think they are -- which is what makes
    ``geom_xpos + geom_xmat @ mesh_vert`` the correct world placement.
    """
    model = g1_model.model
    geoms = _mesh_geoms(model)
    assert geoms, "G1 should have mesh geoms"

    worst = 0.0
    for geom in geoms:
        verts = _compiled_verts(model, geom)
        lo, hi = verts.min(axis=0), verts.max(axis=0)
        centre, half = (lo + hi) / 2, (hi - lo) / 2
        aabb = np.asarray(model.geom_aabb[geom], dtype=float).reshape(6)
        worst = max(
            worst,
            float(np.abs(centre - aabb[:3]).max()),
            float(np.abs(half - aabb[3:]).max()),
        )
    assert worst < 1e-7, f"mesh_vert disagrees with geom_aabb by {worst}"


def test_compiled_geom_pose_is_authored_pose_times_recentering(g1_model):
    """The compiler's bookkeeping, stated exactly.

    ``mesh_pos``/``mesh_quat`` record how far the mesh was moved.  The compiled
    geom pose is the authored one composed with that move -- which is precisely
    why using the authored pose with un-recentered vertices double-counts
    nothing and under-counts the move.
    """
    lm = g1_model
    model = lm.model
    spec = mujoco.MjSpec.from_file(str(lm.xml_path))
    spec_geoms = list(spec.geoms)

    worst = 0.0
    for geom in _mesh_geoms(model):
        mesh = int(model.geom_dataid[geom])
        authored = Pose.from_mj(spec_geoms[geom].pos, spec_geoms[geom].quat)
        recentre = Pose.from_mj(model.mesh_pos[mesh], model.mesh_quat[mesh])
        compiled = Pose.from_mj(model.geom_pos[geom], model.geom_quat[geom])
        worst = max(
            worst, float(np.abs((authored @ recentre).as_matrix() - compiled.as_matrix()).max())
        )
    assert worst < 1e-12, f"recentering relation off by {worst}"


def test_naive_raw_stl_placement_is_wrong_and_by_how_much(g1_model):
    """The pitfall, measured on every mesh geom in the G1.

    The naive placement differs from the correct one by exactly the
    recentering transform, so the error a vertex at local position ``v`` picks
    up is ``recentre.apply(v) - v``.  Measuring it over each geom's bounding
    box corners catches rotation as well as translation -- one G1 mesh is
    recentered almost purely by rotation, and a position-only metric would miss
    it entirely while the part still lands 29 mm out.
    """
    lm = g1_model
    model, data = lm.model, lm.data
    spec = mujoco.MjSpec.from_file(str(lm.xml_path))
    spec_geoms = list(spec.geoms)

    displacements = []
    position_offsets = []
    for geom in _mesh_geoms(model):
        mesh = int(model.geom_dataid[geom])
        recentre = Pose.from_mj(model.mesh_pos[mesh], model.mesh_quat[mesh])

        aabb = np.asarray(model.geom_aabb[geom], dtype=float).reshape(6)
        centre, half = aabb[:3], aabb[3:]
        signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
        corners = centre + signs * half
        displacements.append(float(np.abs(recentre.apply(corners) - corners).max()) * 1000.0)

        body = int(model.geom_bodyid[geom])
        correct = Pose.from_xmat(data.geom_xpos[geom], data.geom_xmat[geom])
        naive = Pose.from_xmat(data.xpos[body], data.xmat[body]) @ Pose.from_mj(
            spec_geoms[geom].pos, spec_geoms[geom].quat
        )
        position_offsets.append(float(np.linalg.norm(correct.pos - naive.pos)) * 1000.0)

    displacements = np.array(displacements)
    position_offsets = np.array(position_offsets)

    # Every single mesh geom is misplaced, not just a handful.
    assert (displacements > 1.0).all(), (
        f"{(displacements <= 1.0).sum()} of {len(displacements)} mesh geoms were "
        "not visibly misplaced by the naive approach"
    )
    assert displacements.max() > 100.0, f"worst naive error only {displacements.max():.3f} mm"
    # And at least one is a near-pure rotation, which a position-only check
    # would wave through.
    assert position_offsets.min() < 1.0 < displacements.min()


def test_raw_stl_equals_compiled_vertices_moved_by_the_recentering(g1_model):
    """Load an actual STL off disk and show it is the compiled mesh, moved."""
    trimesh = pytest.importorskip("trimesh")
    lm = g1_model
    model = lm.model
    spec = mujoco.MjSpec.from_file(str(lm.xml_path))
    meshdir = spec.meshdir or ""
    base = os.path.join(os.path.dirname(str(lm.xml_path)), meshdir)

    by_name = {m.name: m for m in spec.meshes}
    name = "pelvis"
    if name not in by_name:  # pragma: no cover - model layout changed
        pytest.skip(f"{name} mesh not in this G1 revision")

    path = os.path.join(base, by_name[name].file)
    if not os.path.isfile(path):  # pragma: no cover - assets not unpacked
        pytest.skip(f"{path} not on disk")

    raw = trimesh.load(path, process=True)
    raw.merge_vertices()
    raw_verts = np.asarray(raw.vertices, dtype=float)

    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, name)
    adr, num = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    compiled = np.asarray(model.mesh_vert[adr : adr + num], dtype=float).reshape(-1, 3)

    recentre = Pose.from_mj(model.mesh_pos[mesh_id], model.mesh_quat[mesh_id])
    moved = recentre.apply(compiled)

    # Vertex ordering differs between the two, so compare as point sets via
    # their bounds -- enough to show the offset is exactly the recentering.
    assert np.allclose(moved.min(axis=0), raw_verts.min(axis=0), atol=1e-5)
    assert np.allclose(moved.max(axis=0), raw_verts.max(axis=0), atol=1e-5)

    # And the un-moved compiled vertices do *not* match the raw ones.
    gap = float(np.abs(compiled.min(axis=0) - raw_verts.min(axis=0)).max())
    assert gap > 1e-3, f"expected a real offset, got {gap} m"
