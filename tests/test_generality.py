"""§10 -- the exporter should not be G1-shaped.

The G1 and ``humanoid.xml`` are both free-base humanoids, so between them they
never exercise a fixed base, a four-way branch at the root, or a model whose
root link carries no joint at all.  These three do:

* ``franka_emika_panda`` -- fixed base, serial chain, has a keyframe;
* ``anybotics_anymal_c`` -- free base, four legs branching from one trunk;
* ``shadow_hand`` -- fixed base, 26 bodies, five branches of small links.

Marked slow: each is a real mesh export.  Run with ``--run-slow``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from sim2cad.export import run_export
from sim2cad.validate import validate_export

ROBOTS = ["franka_emika_panda", "anybotics_anymal_c", "shadow_hand"]


@pytest.fixture(scope="module")
def exported(request, tmp_path_factory):
    name = request.param
    try:
        return run_export(
            menagerie=name,
            out=str(tmp_path_factory.mktemp(name)),
            write_render=False,
        )
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"{name} unavailable: {exc}")


@pytest.mark.slow
@pytest.mark.parametrize("exported", ROBOTS, indirect=True)
def test_robot_passes_every_check(exported):
    report = validate_export(exported)
    failures = [f"{c.name}: {c.detail}" for c in report.checks if not (c.passed or c.skipped)]
    assert not failures, failures
    assert report.parts, "every one of these robots has geometry"


@pytest.mark.slow
@pytest.mark.parametrize("exported", ROBOTS, indirect=True)
def test_hierarchy_covers_the_whole_subtree(exported):
    """Every body under the root gets a node, branches and all."""
    lm = exported.lm
    assert set(exported.robot.nodes) == set(lm.subtree())

    for node in exported.robot.nodes.values():
        if node.body_id == exported.robot.root_id:
            continue
        assert node.parent_id in exported.robot.nodes, node.name


@pytest.mark.slow
@pytest.mark.parametrize("exported", ROBOTS, indirect=True)
def test_joint_anchors_replay_the_kinematics(exported):
    """The §8.7 check, on robots that are not the G1."""
    from sim2cad.frames import axis_angle_pose
    from sim2cad.joints import joints_document

    lm = exported.lm
    model, data = lm.model, lm.data
    doc = joints_document(lm, scale=1000.0, units="mm")
    by_name = {j["name"]: j for j in doc["joints"]}

    worst = 0.0
    tested = 0
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        child = int(model.jnt_bodyid[j])
        before = lm.body_pose(child)
        anchor = np.array(by_name[name]["anchor_world"]) / 1000.0
        axis = np.array(by_name[name]["axis_world"])

        adr = int(model.jnt_qposadr[j])
        saved = float(data.qpos[adr])
        data.qpos[adr] = saved + 0.5
        mujoco.mj_forward(model, data)
        actual = lm.body_pose(child)
        data.qpos[adr] = saved
        mujoco.mj_forward(model, data)

        predicted = axis_angle_pose(anchor, axis, 0.5) @ before
        worst = max(worst, float(np.linalg.norm(predicted.pos - actual.pos)))
        tested += 1

    assert tested, "expected some hinge joints"
    assert worst < 1e-6, f"worst replay error {worst} m over {tested} hinges"


@pytest.mark.slow
def test_fixed_base_robot_has_no_free_joint():
    """A fixed base is a different code path: nothing places the root for us."""
    from sim2cad.load import load_model, set_world_fix

    try:
        lm = load_model(menagerie="franka_emika_panda")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"panda unavailable: {exc}")

    free = [
        j for j in range(lm.model.njnt)
        if int(lm.model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    assert not free, "the panda should be bolted down"

    set_world_fix(lm)
    assert np.allclose(lm.body_pose(lm.root_body).pos, 0.0, atol=1e-12)

    from sim2cad.joints import joints_document

    doc = joints_document(lm, scale=1000.0, units="mm")
    # With no free joint, every DOF is articulated.
    assert doc["counts"]["articulated_dof"] == int(lm.model.nv)


@pytest.mark.slow
def test_panda_slide_joints_are_lengths_not_angles():
    """The panda's gripper fingers slide, so their range is a distance.

    A hinge range stays in radians under ``--units mm``; a slide range is a
    length and has to be scaled with everything else.  The panda is the first
    real robot that exercises the difference.
    """
    from sim2cad.joints import joints_document
    from sim2cad.load import apply_pose, load_model, set_world_fix

    try:
        lm = load_model(menagerie="franka_emika_panda")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"panda unavailable: {exc}")
    apply_pose(lm, "qpos0")
    set_world_fix(lm)

    mm_doc = joints_document(lm, scale=1000.0, units="mm")
    m_doc = joints_document(lm, scale=1.0, units="m")

    slides = [j for j in mm_doc["joints"] if j["type"] == "slide"]
    assert len(slides) == 2, "the panda has two sliding gripper fingers"
    for j in slides:
        assert "range_deg" not in j, "a sliding joint has no angular range"
        if j["limited"]:
            metres = next(k for k in m_doc["joints"] if k["name"] == j["name"])
            assert np.allclose(j["range"], np.array(metres["range"]) * 1000.0)
            # Finger travel is centimetres, so millimetres read in the tens.
            assert max(abs(v) for v in j["range"]) > 1.0

    hinges = [j for j in mm_doc["joints"] if j["type"] == "hinge" and j["limited"]]
    for j in hinges:
        metres = next(k for k in m_doc["joints"] if k["name"] == j["name"])
        # Angles are unit-free: identical in both exports.
        assert np.allclose(j["range"], metres["range"])


def test_multiple_root_bodies_are_reported(tmp_path):
    """A scene with a robot and a table exports the robot and says so."""
    xml = tmp_path / "scene.xml"
    xml.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="robot" pos="0 0 1">
              <geom type="box" size="0.1 0.1 0.1"/>
              <body name="arm" pos="0 0 0.2">
                <joint name="j" type="hinge" axis="0 1 0"/>
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
            <body name="table" pos="1 0 0">
              <geom type="box" size="0.3 0.3 0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    result = run_export(mjcf=str(xml), out=str(tmp_path / "out"), write_render=False)
    assert {n.name for n in result.robot.nodes.values()} == {"robot", "arm"}
    assert any("attached to the world" in w and "table" in w for w in result.warnings)
    assert validate_export(result).passed
