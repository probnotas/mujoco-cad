"""§8.5 and §8.7 -- joints.json is complete, and it is kinematically true.

The consistency test is the sharp one: take the anchor and axis sim2cad wrote
out at the reference pose, drive the real joint in MuJoCo, and check that
rotating the child link about *our* recorded anchor/axis lands it exactly where
MuJoCo put it.  If the anchor were off by a millimetre or the axis by a
milliradian, this would show up immediately.
"""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from sim2cad.frames import Pose, axis_angle_pose
from sim2cad.joints import joints_document
from sim2cad.load import apply_pose, body_names, load_model, set_world_fix

ANGLE = 0.5  # radians, as the PRD specifies


def _prepared(path=None, menagerie=None, pose="qpos0"):
    lm = load_model(mjcf=path, menagerie=menagerie)
    apply_pose(lm, pose)
    set_world_fix(lm)
    return lm


def _hinge_consistency(lm, scale=1000.0):
    """Max error, in metres, over every hinge joint in the model."""
    doc = joints_document(lm, scale=scale, units="mm")
    model, data = lm.model, lm.data
    bnames = body_names(model)
    by_name = {j["name"]: j for j in doc["joints"]}

    baseline = {b: lm.body_pose(b) for b in lm.subtree()}
    worst = 0.0
    tested = 0

    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        record = by_name[name]
        child = int(model.jnt_bodyid[j])

        anchor = np.array(record["anchor_world"]) / scale
        axis = np.array(record["axis_world"])

        adr = int(model.jnt_qposadr[j])
        saved = float(data.qpos[adr])
        data.qpos[adr] = saved + ANGLE
        mujoco.mj_forward(model, data)
        actual = lm.body_pose(child)
        data.qpos[adr] = saved
        mujoco.mj_forward(model, data)

        predicted = axis_angle_pose(anchor, axis, ANGLE) @ baseline[child]
        worst = max(
            worst,
            float(np.linalg.norm(predicted.pos - actual.pos)),
            float(np.abs(predicted.rot - actual.rot).max()),
        )
        tested += 1

    return worst, tested


def test_humanoid_hinges_are_consistent(humanoid_path):
    lm = _prepared(path=str(humanoid_path))
    worst, tested = _hinge_consistency(lm)
    assert tested == 21
    assert worst < 1e-9, f"worst hinge replay error {worst} m"


def test_g1_hinges_are_consistent(g1_model):
    lm = g1_model
    apply_pose(lm, "qpos0")
    set_world_fix(lm)
    worst, tested = _hinge_consistency(lm)
    assert tested == 29, f"expected 29 G1 hinges, found {tested}"
    assert worst < 1e-6, f"worst hinge replay error {worst} m"


def test_g1_named_joints_from_the_prd(g1_model):
    """The PRD asks specifically for a knee, an elbow and a waist joint."""
    lm = g1_model
    apply_pose(lm, "qpos0")
    set_world_fix(lm)
    doc = joints_document(lm, scale=1000.0, units="mm")
    by_name = {j["name"]: j for j in doc["joints"]}

    picks = [
        next(n for n in by_name if "knee" in n),
        next(n for n in by_name if "elbow" in n),
        next(n for n in by_name if "waist" in n),
    ]
    model, data = lm.model, lm.data
    for name in picks:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        child = int(model.jnt_bodyid[j])
        before = lm.body_pose(child)
        anchor = np.array(by_name[name]["anchor_world"]) / 1000.0
        axis = np.array(by_name[name]["axis_world"])

        adr = int(model.jnt_qposadr[j])
        saved = float(data.qpos[adr])
        data.qpos[adr] = saved + ANGLE
        mujoco.mj_forward(model, data)
        actual = lm.body_pose(child)
        data.qpos[adr] = saved
        mujoco.mj_forward(model, data)

        predicted = axis_angle_pose(anchor, axis, ANGLE) @ before
        assert np.linalg.norm(predicted.pos - actual.pos) < 1e-6, name
        assert np.abs(predicted.rot - actual.rot).max() < 1e-6, name


def test_joints_document_declares_units_and_counts(humanoid_path):
    lm = _prepared(path=str(humanoid_path))
    doc = joints_document(lm, scale=1000.0, units="mm")
    assert doc["units"]["length"] == "mm"
    assert "radians" in doc["units"]["angle"]
    assert doc["counts"]["articulated_dof"] == int(lm.model.nv) - 6
    assert len(doc["joints"]) == lm.model.njnt

    free = [j for j in doc["joints"] if j["type"] == "free"]
    assert len(free) == 1 and free[0]["ndof"] == 6

    hinges = [j for j in doc["joints"] if j["type"] == "hinge"]
    assert hinges, "humanoid should have hinges"
    for j in hinges:
        assert np.isclose(np.linalg.norm(j["axis_body_frame"]), 1.0)
        if j["limited"]:
            assert "range_deg" in j
            assert np.allclose(np.radians(j["range_deg"]), j["range"])


def test_slide_joint_range_is_converted_to_export_units(tmp_path):
    xml = tmp_path / "slider.xml"
    xml.write_text(
        """
        <mujoco>
          <worldbody>
            <body name="base" pos="0 0 1">
              <geom type="box" size="0.1 0.1 0.1"/>
              <body name="car" pos="0.5 0 0">
                <joint name="rail" type="slide" axis="1 0 0" range="-0.25 0.75" limited="true"/>
                <geom type="sphere" size="0.05"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    lm = _prepared(path=str(xml))
    doc = joints_document(lm, scale=1000.0, units="mm")
    rail = next(j for j in doc["joints"] if j["name"] == "rail")
    assert rail["type"] == "slide"
    assert np.allclose(rail["range"], [-250.0, 750.0])
    assert "range_deg" not in rail

    doc_m = joints_document(lm, scale=1.0, units="m")
    rail_m = next(j for j in doc_m["joints"] if j["name"] == "rail")
    assert np.allclose(rail_m["range"], [-0.25, 0.75])


def test_anchor_frames_agree_with_each_other(humanoid_path):
    """body-frame, parent-frame and world anchors must describe one point."""
    lm = _prepared(path=str(humanoid_path))
    doc = joints_document(lm, scale=1000.0, units="mm")
    model = lm.model
    for record in doc["joints"]:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, record["name"])
        child = int(model.jnt_bodyid[j])
        parent = int(model.body_parentid[child])

        anchor_child = np.array(record["anchor_body_frame"]) / 1000.0
        from_child = lm.body_pose(child).apply(anchor_child) * 1000.0
        assert np.allclose(from_child, record["anchor_world"], atol=1e-9)

        anchor_parent = np.array(record["anchor_parent_frame"]) / 1000.0
        from_parent = lm.body_pose(parent).apply(anchor_parent) * 1000.0
        assert np.allclose(from_parent, record["anchor_world"], atol=1e-9)
