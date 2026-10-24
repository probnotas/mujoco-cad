"""Joint extraction: the part STEP cannot carry, so it goes in ``joints.json``.

STEP has no reliable notion of a mate or a constraint, so the kinematics are
written out beside the geometry.  Each joint is reported in three frames:

* ``*_body_frame`` -- MuJoCo's native storage, in the **child** body's frame.
  This is the one to use when you open ``parts/<child>.step``, because that
  file's origin is the child body frame.
* ``*_parent_frame`` -- the same anchor and axis in the parent link's frame,
  which is what you want when designing a bracket that bolts to the parent.
* ``*_world`` -- in the exported assembly's world frame.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np

from .frames import Pose
from .load import LoadedModel, body_names, joint_names

__all__ = ["joint_records", "joints_document", "JOINT_TYPE_NAMES"]

JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}

#: Joint types whose position coordinate is a length, not an angle.
_LINEAR = {int(mujoco.mjtJoint.mjJNT_SLIDE)}


def _actuator_for_joint(model: mujoco.MjModel, joint_id: int) -> dict[str, Any] | None:
    for a in range(model.nu):
        if int(model.actuator_trntype[a]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(model.actuator_trnid[a, 0]) != joint_id:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"actuator{a}"
        rec: dict[str, Any] = {"name": name, "gear": float(model.actuator_gear[a, 0])}
        if bool(model.actuator_ctrllimited[a]):
            rec["ctrlrange"] = [float(v) for v in model.actuator_ctrlrange[a]]
        else:
            rec["ctrlrange"] = None
        rec["forcerange"] = (
            [float(v) for v in model.actuator_forcerange[a]]
            if bool(model.actuator_forcelimited[a])
            else None
        )
        return rec
    return None


def joint_records(lm: LoadedModel, scale: float = 1.0) -> list[dict[str, Any]]:
    """One record per joint belonging to a body in the exported subtree."""
    model, data = lm.model, lm.data
    bnames, jnames = body_names(model), joint_names(model)
    subtree = set(lm.subtree())
    records: list[dict[str, Any]] = []

    for j in range(model.njnt):
        child = int(model.jnt_bodyid[j])
        if child not in subtree:
            continue
        parent = int(model.body_parentid[child])
        jtype = int(model.jnt_type[j])
        type_name = JOINT_TYPE_NAMES.get(jtype, f"type{jtype}")

        anchor_child = np.asarray(model.jnt_pos[j], dtype=float) * scale
        axis_child = np.asarray(model.jnt_axis[j], dtype=float)
        norm = float(np.linalg.norm(axis_child))
        axis_child = axis_child / norm if norm > 0 else axis_child

        child_in_parent = lm.body_pose(child).relative_to(lm.body_pose(parent))
        anchor_parent = child_in_parent.apply(anchor_child / scale) * scale
        axis_parent = child_in_parent.rotate(axis_child)

        anchor_world = lm.world_fix.apply(np.asarray(data.xanchor[j], dtype=float)) * scale
        axis_world = lm.world_fix.rotate(np.asarray(data.xaxis[j], dtype=float))

        dof_adr = int(model.jnt_dofadr[j])
        ndof = {0: 6, 1: 3, 2: 1, 3: 1}[jtype]
        armature = [float(v) for v in model.dof_armature[dof_adr : dof_adr + ndof]]
        damping = [float(v) for v in model.dof_damping[dof_adr : dof_adr + ndof]]

        limited = bool(model.jnt_limited[j])
        rng = [float(v) for v in model.jnt_range[j]]
        if jtype in _LINEAR:
            rng = [v * scale for v in rng]

        record: dict[str, Any] = {
            "name": jnames[j],
            "type": type_name,
            "parent_body": bnames[parent],
            "child_body": bnames[child],
            "body_frame": bnames[child],
            "anchor_body_frame": anchor_child.tolist(),
            "axis_body_frame": axis_child.tolist(),
            "anchor_parent_frame": anchor_parent.tolist(),
            "axis_parent_frame": axis_parent.tolist(),
            "anchor_world": anchor_world.tolist(),
            "axis_world": axis_world.tolist(),
            "limited": limited,
            "range": rng if limited else None,
            "armature": armature,
            "damping": damping,
            "dof_index": dof_adr,
            "ndof": ndof,
            "qpos_index": int(model.jnt_qposadr[j]),
            "qpos": [
                float(v)
                for v in data.qpos[
                    int(model.jnt_qposadr[j]) : int(model.jnt_qposadr[j])
                    + {0: 7, 1: 4, 2: 1, 3: 1}[jtype]
                ]
            ],
            "actuator": _actuator_for_joint(model, j),
        }

        if type_name == "hinge" and limited:
            record["range_deg"] = [float(np.degrees(v)) for v in rng]
        if type_name in ("free", "ball"):
            # A ball joint has no single axis, and a free joint has no anchor
            # that means anything; keep the keys so consumers can rely on them.
            record["axis_body_frame"] = None if type_name == "ball" else None
            record["axis_parent_frame"] = None
            record["axis_world"] = None

        records.append(record)

    return records


def joints_document(lm: LoadedModel, scale: float, units: str) -> dict[str, Any]:
    """The full ``joints.json`` payload, units declared up front."""
    model = lm.model
    records = joint_records(lm, scale=scale)
    free_dofs = sum(
        6 for j in range(model.njnt) if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    return {
        "schema": "sim2cad/joints/1",
        "robot": lm.name,
        "source": str(lm.xml_path),
        "pose": lm.pose_label,
        "units": {
            "length": units,
            "angle": "radians (hinge ranges also given in degrees as range_deg)",
        },
        "frames": {
            "body_frame": "the joint's own child body frame, as MuJoCo stores it",
            "parent_frame": "the parent link's frame",
            "world": "the exported assembly's world frame, after root normalisation",
        },
        "counts": {
            "joints": len(records),
            "nv": int(model.nv),
            "actuated_dof": int(model.nu),
            "articulated_dof": int(model.nv) - free_dofs,
        },
        "joints": records,
    }
