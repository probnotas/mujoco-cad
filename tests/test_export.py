"""§8.3-§8.6, §8.8 and §8.10 -- the export itself.

Most of the work is done by :func:`sim2cad.validate.validate_export`, which is
the same code path the CLI uses to write ``report.md``; the tests here assert
on its results so the report and the test suite can never disagree.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from sim2cad.export import run_export
from sim2cad.validate import validate_export


def _failures(report):
    return [f"{c.name}: {c.detail}" for c in report.checks if not (c.passed or c.skipped)]


def test_humanoid_passes_every_check(humanoid_export):
    report = validate_export(humanoid_export)
    assert not _failures(report), _failures(report)


def test_humanoid_primitives_are_exact(humanoid_export):
    """Primitives should land at machine precision, not merely within tolerance."""
    report = validate_export(humanoid_export)
    check = next(c for c in report.checks if c.name.startswith("3c."))
    assert not check.skipped
    assert check.value < 1e-6, check.detail

    volume = next(c for c in report.checks if c.name.startswith("4b."))
    assert volume.value < 1e-9, volume.detail


def test_capsule_volume_matches_the_closed_form(humanoid_export):
    """capsule = pi r^2 (2h) + 4/3 pi r^3, as the PRD spells out."""
    import mujoco

    lm = humanoid_export.lm
    model = lm.model
    report = validate_export(humanoid_export)
    capsules = [p for p in report.parts if p.geom_type == "capsule"]
    assert capsules, "humanoid is built from capsules"
    for part in capsules:
        geom = next(
            p for node in humanoid_export.robot.bodies_with_geometry()
            for p in node.parts if p.safe_name == part.part
        )
        r, h = (np.asarray(model.geom_size[geom.geom_id], dtype=float) * 1000.0)[:2]
        expected = np.pi * r**2 * 2 * h + 4.0 / 3.0 * np.pi * r**3
        assert abs(part.volume - expected) / expected < 1e-9


def test_hierarchy_and_names_survive_the_round_trip(humanoid_export):
    from sim2cad.stepread import read_step_assembly

    root = read_step_assembly(str(humanoid_export.assembly_path))
    names = set(root.names())
    for node in humanoid_export.robot.nodes.values():
        assert node.safe_name in names, node.name

    # Nesting, not a flat list: the hand must sit under the forearm.
    def find(node, target):
        if node.name == target:
            return node
        for child in node.children:
            got = find(child, target)
            if got is not None:
                return got
        return None

    forearm = find(root, "lower_arm_left")
    assert forearm is not None
    assert find(forearm, "hand_left") is not None


def test_geometryless_bodies_still_appear(tmp_path):
    xml = tmp_path / "gap.xml"
    xml.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base">
              <geom type="box" size="0.1 0.1 0.1"/>
              <body name="ghost" pos="0 0 0.3">
                <joint name="j" type="hinge" axis="0 1 0"/>
                <body name="tip" pos="0 0 0.3">
                  <geom type="sphere" size="0.05"/>
                </body>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    result = run_export(mjcf=str(xml), out=str(tmp_path / "out"))
    names = {n.name for n in result.robot.nodes.values()}
    assert {"base", "ghost", "tip"} <= names
    ghost = next(n for n in result.robot.nodes.values() if n.name == "ghost")
    assert not ghost.has_geometry

    from sim2cad.stepread import read_step_assembly

    assert "ghost" in set(read_step_assembly(str(result.assembly_path)).names())
    report = validate_export(result)
    assert not _failures(report), _failures(report)


def test_units_metres_scales_everything_once(tmp_path):
    mm = run_export(mjcf="tests/data/humanoid.xml", out=str(tmp_path / "mm"), units="mm")
    m = run_export(mjcf="tests/data/humanoid.xml", out=str(tmp_path / "m"), units="m")

    mm_doc = json.loads(mm.joints_path.read_text())
    m_doc = json.loads(m.joints_path.read_text())
    assert mm_doc["units"]["length"] == "mm"
    assert m_doc["units"]["length"] == "m"

    a = np.array(mm_doc["joints"][1]["anchor_world"])
    b = np.array(m_doc["joints"][1]["anchor_world"])
    assert np.allclose(a, b * 1000.0, atol=1e-9)

    # Volumes differ by 1000^3, which is the giveaway if a scale were applied
    # twice or in only one of the two places.
    rm = validate_export(mm)
    rmet = validate_export(m)
    va = next(p.volume for p in rm.parts if p.part == "head__head")
    vb = next(p.volume for p in rmet.parts if p.part == "head__head")
    assert abs(va - vb * 1e9) / va < 1e-9
    assert not _failures(rmet), _failures(rmet)

    text = m.assembly_path.read_text()
    assert "SI_UNIT($,.METRE.)" in text, "metre export must declare metres"


def test_keyframe_pose_differs_and_still_passes(tmp_path):
    zero = run_export(mjcf="tests/data/humanoid.xml", out=str(tmp_path / "a"))
    squat = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "b"), pose="keyframe:squat"
    )
    assert squat.lm.pose_label == "keyframe_squat"

    za = zero.robot.nodes
    sa = squat.robot.nodes
    moved = max(
        float(np.linalg.norm(za[i].world_pose.pos - sa[i].world_pose.pos)) for i in za
    )
    assert moved > 0.05, "the squat keyframe should move links by centimetres"

    report = validate_export(squat)
    assert not _failures(report), _failures(report)


def test_json_pose_is_applied(tmp_path):
    spec = tmp_path / "pose.json"
    spec.write_text(json.dumps({"knee_left": -1.2}))
    result = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "out"), pose=f"json:{spec}"
    )
    doc = json.loads(result.joints_path.read_text())
    knee = next(j for j in doc["joints"] if j["name"] == "knee_left")
    assert np.isclose(knee["qpos"][0], -1.2)


def test_ground_puts_the_lowest_point_at_zero(tmp_path):
    result = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "out"), ground=True
    )
    from sim2cad.geoms import geom_local_points

    lowest = min(
        float(result.lm.geom_pose(g).apply(geom_local_points(result.lm.model, g))[:, 2].min())
        for g in result.selection.geom_ids
    )
    assert abs(lowest) < 1e-9, f"lowest point at z={lowest}"


def test_keep_root_pose_leaves_the_base_where_the_sim_had_it(tmp_path):
    moved = run_export(mjcf="tests/data/humanoid.xml", out=str(tmp_path / "a"))
    kept = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "b"), keep_root_pose=True
    )
    root = moved.lm.root_body
    assert np.allclose(moved.lm.body_pose(root).pos, 0.0, atol=1e-12)
    assert np.allclose(kept.lm.body_pose(root).pos, kept.lm.raw_body_pose(root).pos)
    assert np.linalg.norm(kept.lm.body_pose(root).pos) > 0.1


def test_only_exports_a_subset_but_keeps_the_chain(tmp_path):
    result = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "out"), only="hand_left"
    )
    with_geom = {n.name for n in result.robot.bodies_with_geometry()}
    assert with_geom == {"hand_left"}
    # Ancestors are kept so the hand still lands in the right place.
    assert {"torso", "upper_arm_left", "lower_arm_left"} <= set(
        n.name for n in result.robot.nodes.values()
    )

    full = run_export(mjcf="tests/data/humanoid.xml", out=str(tmp_path / "full"))
    a = next(n for n in result.robot.nodes.values() if n.name == "hand_left")
    b = next(n for n in full.robot.nodes.values() if n.name == "hand_left")
    assert a.world_pose.allclose(b.world_pose, atol=1e-12)


def test_joint_markers_can_be_turned_off(tmp_path):
    on = run_export(mjcf="tests/data/humanoid.xml", out=str(tmp_path / "on"))
    off = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "off"), joint_markers=False
    )
    assert len(on.robot.joint_marker_names) == 21
    assert off.robot.joint_marker_names == []

    from sim2cad.stepread import read_step_assembly

    assert "_JOINTS" in set(read_step_assembly(str(on.assembly_path)).names())
    assert "_JOINTS" not in set(read_step_assembly(str(off.assembly_path)).names())


def test_collision_and_all_geom_modes(tmp_path):
    collision = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "c"), geoms="collision"
    )
    every = run_export(mjcf="tests/data/humanoid.xml", out=str(tmp_path / "a"), geoms="all")
    # humanoid.xml has no visual geoms, so collision and all agree,
    # and the floor plane is excluded from both.
    assert len(collision.robot.parts) == len(every.robot.parts) == 19
    assert not _failures(validate_export(collision))


def test_visual_mode_falls_back_and_says_so(humanoid_export):
    assert humanoid_export.selection.mode == "visual"
    assert len(humanoid_export.selection.fallbacks) == 16
    assert any("fell back" in w for w in humanoid_export.warnings)


def test_part_files_are_written_in_the_body_frame(humanoid_export):
    from sim2cad.stepread import read_step_assembly, shape_bbox

    path = humanoid_export.part_paths["head"]
    assert path.exists()
    root = read_step_assembly(str(path))
    leaves = list(root.leaves())
    assert len(leaves) == 1

    node, world = leaves[0]
    placed = node.shape.Moved(world.to_location().wrapped)
    lo, hi = shape_bbox(placed)
    centre = (lo + hi) / 2

    # The head sphere sits 190 mm above its own body origin in humanoid.xml.
    lm = humanoid_export.lm
    body = next(
        n for n in humanoid_export.robot.bodies_with_geometry() if n.name == "head"
    )
    expected = (
        lm.geom_pose(body.parts[0].geom_id)
        .relative_to(lm.body_pose(body.body_id))
        .pos
        * 1000.0
    )
    assert np.allclose(centre, expected, atol=1e-6)


@pytest.mark.slow
def test_g1_passes_every_check(g1_export):
    report = validate_export(g1_export)
    assert not _failures(report), _failures(report)


@pytest.mark.slow
def test_g1_completeness(g1_export):
    report = validate_export(g1_export)
    assert len(g1_export.robot.bodies_with_geometry()) == 30
    assert len(g1_export.robot.parts) == 35
    doc = json.loads(g1_export.joints_path.read_text())
    assert doc["counts"]["joints"] == 30
    assert doc["counts"]["articulated_dof"] == 29
