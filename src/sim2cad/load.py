"""Loading, compiling and posing a MuJoCo model.

Everything downstream reads the *compiled* :class:`mujoco.MjModel`, never the
raw XML.  The compiler is what resolves ``<default>`` classes, ``childclass``,
``<compiler angle="degree">``, ``eulerseq``, ``fromto``, mesh ``scale`` and
``meshdir`` -- re-deriving any of that by hand is how exporters go subtly wrong.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .frames import Pose

__all__ = [
    "LoadedModel",
    "load_model",
    "resolve_source",
    "keyframe_names",
    "apply_pose",
    "body_names",
    "joint_names",
]


@dataclass
class LoadedModel:
    """A compiled model plus the kinematic state we exported it at."""

    model: mujoco.MjModel
    data: mujoco.MjData
    name: str
    xml_path: Path
    pose_label: str = "qpos0"
    #: Correction applied to every world-frame quantity on the way out, so the
    #: robot root lands at the origin (or on the ground).  Relative poses inside
    #: the tree are untouched by it.
    world_fix: Pose = field(default_factory=Pose.identity)
    notes: list[str] = field(default_factory=list)

    @property
    def root_body(self) -> int:
        """The first body parented to ``world``; the robot's base link."""
        for i in range(1, self.model.nbody):
            if self.model.body_parentid[i] == 0:
                return i
        raise ValueError("model has no body attached to the world body")

    def root_bodies(self) -> list[int]:
        """Every body parented to ``world``.

        A robot entry point normally has exactly one.  More than one means the
        file holds a scene, or several robots, and we only export the first --
        which is worth saying out loud rather than quietly dropping the rest.
        """
        return [i for i in range(1, self.model.nbody) if self.model.body_parentid[i] == 0]

    def subtree(self) -> list[int]:
        """Body ids of the robot root and all its descendants, parents first."""
        root = self.root_body
        out = [root]
        for i in range(root + 1, self.model.nbody):
            if self.model.body_parentid[i] in out:
                out.append(i)
        return out

    def body_pose(self, body_id: int) -> Pose:
        """World pose of a body, after :attr:`world_fix`."""
        return self.world_fix @ Pose.from_xmat(
            self.data.xpos[body_id], self.data.xmat[body_id]
        )

    def geom_pose(self, geom_id: int) -> Pose:
        """World pose of a geom, after :attr:`world_fix`."""
        return self.world_fix @ Pose.from_xmat(
            self.data.geom_xpos[geom_id], self.data.geom_xmat[geom_id]
        )

    def raw_body_pose(self, body_id: int) -> Pose:
        """World pose of a body as MuJoCo computed it, with no correction."""
        return Pose.from_xmat(self.data.xpos[body_id], self.data.xmat[body_id])

    def raw_geom_pose(self, geom_id: int) -> Pose:
        return Pose.from_xmat(self.data.geom_xpos[geom_id], self.data.geom_xmat[geom_id])

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)


def resolve_source(
    menagerie: str | None = None,
    mjcf: str | Path | None = None,
    entry: str | None = None,
) -> tuple[Path, str]:
    """Return ``(xml_path, model_name)`` for a Menagerie name or a local file."""
    if (menagerie is None) == (mjcf is None):
        raise ValueError("pass exactly one of menagerie= or mjcf=")

    if mjcf is not None:
        path = Path(mjcf).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"no such MJCF file: {path}")
        return path, path.stem

    return _resolve_menagerie(menagerie, entry)


def _resolve_menagerie(name: str, entry: str | None) -> tuple[Path, str]:
    """Fetch a Menagerie model, trying each backend that might be installed."""
    errors: list[str] = []
    try:
        import mujoco_menagerie as mm

        robot = mm.get(name)
        chosen = entry or robot.default_model
        if chosen.endswith(".xml"):
            chosen = chosen[: -len(".xml")]
        path = Path(robot.xml(chosen))
        return path, f"{name}_{chosen}" if chosen != name else name
    except ImportError as exc:  # pragma: no cover - depends on environment
        errors.append(f"mujoco_menagerie: {exc}")
    except Exception as exc:  # pragma: no cover - network / unknown robot
        errors.append(f"mujoco_menagerie: {type(exc).__name__}: {exc}")

    try:  # pragma: no cover - fallback path
        from robot_descriptions.loaders.mujoco import load_robot_description

        desc = name if name.endswith("_mj_description") else f"{name}_mj_description"
        model = load_robot_description(desc)
        del model
        import importlib

        mod = importlib.import_module(f"robot_descriptions.{desc}")
        return Path(mod.MJCF_PATH), name
    except Exception as exc:  # pragma: no cover
        errors.append(f"robot_descriptions: {type(exc).__name__}: {exc}")

    raise RuntimeError(
        f"could not resolve Menagerie model {name!r}.\n  " + "\n  ".join(errors)
    )


def load_model(
    menagerie: str | None = None,
    mjcf: str | Path | None = None,
    entry: str | None = None,
) -> LoadedModel:
    """Compile a model and run forward kinematics at ``qpos0``."""
    path, name = resolve_source(menagerie=menagerie, mjcf=mjcf, entry=entry)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return LoadedModel(model=model, data=data, name=name, xml_path=path)


# -- names -----------------------------------------------------------------


def _names(model: mujoco.MjModel, objtype, count: int) -> list[str]:
    out = []
    for i in range(count):
        n = mujoco.mj_id2name(model, objtype, i)
        out.append(n if n is not None else f"{objtype.name.lower()}_{i}")
    return out


def body_names(model: mujoco.MjModel) -> list[str]:
    return _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)


def joint_names(model: mujoco.MjModel) -> list[str]:
    return _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)


def geom_names(model: mujoco.MjModel) -> list[str]:
    return _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom)


def keyframe_names(model: mujoco.MjModel) -> list[str]:
    return _names(model, mujoco.mjtObj.mjOBJ_KEY, model.nkey)


# -- posing ----------------------------------------------------------------


def apply_pose(lm: LoadedModel, spec: str = "qpos0") -> None:
    """Set ``qpos`` from a pose spec and re-run kinematics.

    ``qpos0`` | ``keyframe:NAME`` | ``json:FILE``.
    """
    model, data = lm.model, lm.data
    mujoco.mj_resetData(model, data)

    if spec in ("qpos0", "", None):
        label = "qpos0"
    elif spec.startswith("keyframe:"):
        key = spec.split(":", 1)[1]
        names = keyframe_names(model)
        if key not in names:
            raise ValueError(
                f"no keyframe {key!r}; model has {names or 'none'}"
            )
        kid = names.index(key)
        data.qpos[:] = model.key_qpos[kid]
        label = f"keyframe_{key}"
    elif spec.startswith("json:"):
        path = Path(spec.split(":", 1)[1]).expanduser()
        values = json.loads(path.read_text())
        if not isinstance(values, dict):
            raise ValueError(f"{path} must contain a JSON object of joint -> value")
        _apply_joint_values(model, data, values)
        label = f"json_{path.stem}"
    else:
        raise ValueError(
            f"unknown pose spec {spec!r}; expected qpos0, keyframe:NAME or json:FILE"
        )

    mujoco.mj_forward(model, data)
    lm.pose_label = label


def _apply_joint_values(model, data, values: dict) -> None:
    names = joint_names(model)
    widths = {
        int(mujoco.mjtJoint.mjJNT_FREE): 7,
        int(mujoco.mjtJoint.mjJNT_BALL): 4,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    for jname, value in values.items():
        if jname not in names:
            raise ValueError(f"no joint named {jname!r} in the model")
        jid = names.index(jname)
        adr = int(model.jnt_qposadr[jid])
        width = widths[int(model.jnt_type[jid])]
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.size != width:
            raise ValueError(
                f"joint {jname!r} takes {width} value(s), got {arr.size}"
            )
        data.qpos[adr : adr + width] = arr


def set_world_fix(lm: LoadedModel, keep_root_pose: bool = False) -> None:
    """Choose the world correction: identity, or one that puts the root at origin."""
    if keep_root_pose:
        lm.world_fix = Pose.identity()
    else:
        lm.world_fix = lm.raw_body_pose(lm.root_body).inverse()


def ground_world_fix(lm: LoadedModel, lowest_z: float) -> None:
    """Shift the world correction so ``lowest_z`` (already corrected) sits at z=0."""
    lm.world_fix = Pose.from_translation([0.0, 0.0, -lowest_z]) @ lm.world_fix
