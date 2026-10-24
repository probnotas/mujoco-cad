"""Geom classification and primitive -> B-rep construction.

Two things to keep straight:

* MuJoCo ``geom_size`` holds **half**-sizes for boxes, cylinders and capsules.
  A box with ``size="0.1 0.2 0.3"`` is 0.2 x 0.4 x 0.6 metres.
* Every solid here is built in the geom's **own** frame, centred on the origin
  and aligned with the local z axis.  Placement is somebody else's job
  (:mod:`sim2cad.assembly`), which keeps the two concerns from contaminating
  each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import mujoco
import numpy as np
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.gp import gp_Ax2, gp_Dir, gp_GTrsf, gp_Mat, gp_Pnt, gp_XYZ
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopoDS import TopoDS_Shape

from .load import LoadedModel, geom_names

__all__ = [
    "GeomClass",
    "GeomSelection",
    "classify_geom",
    "select_geoms",
    "primitive_solid",
    "PRIMITIVE_TYPES",
    "analytic_volume",
    "geom_local_points",
]

#: Geom types we turn into exact B-rep primitives.
PRIMITIVE_TYPES = frozenset(
    {
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        int(mujoco.mjtGeom.mjGEOM_BOX),
    }
)

#: Geom types that describe the environment rather than the robot.
SKIPPED_TYPES = frozenset(
    {int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD)}
)

_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def geom_type_name(type_id: int) -> str:
    return _TYPE_NAMES.get(int(type_id), f"type{int(type_id)}")


class GeomClass(str, Enum):
    VISUAL = "visual"
    COLLISION = "collision"


def classify_geom(model: mujoco.MjModel, geom_id: int) -> GeomClass:
    """Visual geoms take part in no contacts at all."""
    if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
        return GeomClass.VISUAL
    return GeomClass.COLLISION


@dataclass
class GeomSelection:
    """Which geoms each body contributes, and how we decided."""

    by_body: dict[int, list[int]]
    mode: str
    fallbacks: list[str]
    skipped: list[str]

    @property
    def geom_ids(self) -> list[int]:
        return [g for ids in self.by_body.values() for g in ids]


def select_geoms(lm: LoadedModel, mode: str = "visual") -> GeomSelection:
    """Pick the geoms to export, per body.

    ``mode`` is ``visual``, ``collision`` or ``all``.  A body with no geom of
    the requested class falls back to whatever it does have, which is reported
    rather than hidden -- ``humanoid.xml``, for instance, has no visual geoms
    at all.
    """
    if mode not in ("visual", "collision", "all"):
        raise ValueError(f"unknown geom mode {mode!r}")

    model = lm.model
    gnames = geom_names(model)
    by_body: dict[int, list[int]] = {}
    fallbacks: list[str] = []
    skipped: list[str] = []
    bnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)]

    for body in lm.subtree():
        start = int(model.body_geomadr[body])
        count = int(model.body_geomnum[body])
        candidates = []
        for g in range(start, start + count):
            if int(model.geom_type[g]) in SKIPPED_TYPES:
                skipped.append(f"{gnames[g]} ({geom_type_name(model.geom_type[g])})")
                continue
            candidates.append(g)

        if mode == "all":
            chosen = candidates
        else:
            want = GeomClass(mode)
            chosen = [g for g in candidates if classify_geom(model, g) == want]
            if not chosen and candidates:
                chosen = candidates
                fallbacks.append(bnames[body] or f"body{body}")

        by_body[body] = chosen

    return GeomSelection(by_body=by_body, mode=mode, fallbacks=fallbacks, skipped=skipped)


# -- primitives ------------------------------------------------------------


def primitive_solid(model: mujoco.MjModel, geom_id: int, scale: float = 1.0) -> TopoDS_Shape:
    """An exact B-rep solid for a primitive geom, in the geom's own frame.

    ``scale`` converts metres to the export unit and is applied to the sizes
    here, once.  Nothing downstream scales the shape again.
    """
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float) * scale

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return BRepPrimAPI_MakeSphere(float(size[0])).Shape()

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        hx, hy, hz = size[:3]
        corner = gp_Pnt(-float(hx), -float(hy), -float(hz))
        return BRepPrimAPI_MakeBox(corner, 2 * float(hx), 2 * float(hy), 2 * float(hz)).Shape()

    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, half = float(size[0]), float(size[1])
        axis = gp_Ax2(gp_Pnt(0.0, 0.0, -half), gp_Dir(0.0, 0.0, 1.0))
        return BRepPrimAPI_MakeCylinder(axis, radius, 2 * half).Shape()

    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, half = float(size[0]), float(size[1])
        axis = gp_Ax2(gp_Pnt(0.0, 0.0, -half), gp_Dir(0.0, 0.0, 1.0))
        shape = BRepPrimAPI_MakeCylinder(axis, radius, 2 * half).Shape()
        for z in (-half, half):
            cap = BRepPrimAPI_MakeSphere(gp_Pnt(0.0, 0.0, z), radius).Shape()
            fuse = BRepAlgoAPI_Fuse(shape, cap)
            fuse.Build()
            if not fuse.IsDone():
                raise RuntimeError(f"capsule fuse failed for geom {geom_id}")
            shape = fuse.Shape()
        # The fuse leaves the sphere/cylinder tangency as a seam edge; merging
        # it keeps the capsule a clean three-face solid.
        return unify_same_domain(shape)

    if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        a, b, c = (float(v) for v in size[:3])
        sphere = BRepPrimAPI_MakeSphere(1.0).Shape()
        gtrsf = gp_GTrsf()
        gtrsf.SetVectorialPart(gp_Mat(a, 0, 0, 0, b, 0, 0, 0, c))
        gtrsf.SetTranslationPart(gp_XYZ(0.0, 0.0, 0.0))
        builder = BRepBuilderAPI_GTransform(sphere, gtrsf, True)
        builder.Build()
        if not builder.IsDone():
            raise RuntimeError(f"ellipsoid GTransform failed for geom {geom_id}")
        return builder.Shape()

    raise ValueError(f"geom {geom_id} is {geom_type_name(gtype)}, not a primitive")


def unify_same_domain(shape: TopoDS_Shape, edges: bool = True, faces: bool = True) -> TopoDS_Shape:
    """Merge adjacent faces that lie on the same surface (and likewise edges)."""
    upgrade = ShapeUpgrade_UnifySameDomain(shape, edges, faces, True)
    upgrade.Build()
    return upgrade.Shape()


def analytic_volume(model: mujoco.MjModel, geom_id: int, scale: float = 1.0) -> float | None:
    """Closed-form volume of a primitive geom, for the accuracy tests."""
    gtype = int(model.geom_type[geom_id])
    s = np.asarray(model.geom_size[geom_id], dtype=float) * scale
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return 4.0 / 3.0 * np.pi * s[0] ** 3
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        return 8.0 * s[0] * s[1] * s[2]
    if gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return np.pi * s[0] ** 2 * 2.0 * s[1]
    if gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return np.pi * s[0] ** 2 * 2.0 * s[1] + 4.0 / 3.0 * np.pi * s[0] ** 3
    if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        return 4.0 / 3.0 * np.pi * s[0] * s[1] * s[2]
    return None


def geom_local_points(model: mujoco.MjModel, geom_id: int, samples: int = 24) -> np.ndarray:
    """Reference points on a geom's surface, in the geom frame.

    Used by the placement checks: transform these to world via MuJoCo and via
    the exported CAD, and the two must agree.  For meshes these are the actual
    vertices; for primitives they are the eight bounding-box corners, which is
    enough to pin down position and orientation.
    """
    gtype = int(model.geom_type[geom_id])
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = int(model.geom_dataid[geom_id])
        adr = int(model.mesh_vertadr[mesh])
        num = int(model.mesh_vertnum[mesh])
        return np.asarray(model.mesh_vert[adr : adr + num], dtype=float).reshape(-1, 3)

    aabb = np.asarray(model.geom_aabb[geom_id], dtype=float).reshape(6)
    centre, half = aabb[:3], aabb[3:]
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    return centre + signs * half
