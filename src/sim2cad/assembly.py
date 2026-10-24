"""Build a nested CadQuery assembly that mirrors the kinematic tree.

Frames, and where each one is used:

* every solid is built in its **geom's** frame (:mod:`sim2cad.geoms`,
  :mod:`sim2cad.meshes`);
* each geom part is placed with ``geom_from_body``, so a body's contents sit in
  the **body** frame -- which is why ``parts/<body>.step`` has its origin at the
  link's own frame, right where you want it when designing a mount;
* each body node is placed with ``body_from_parent``, so dragging a
  sub-assembly in CAD drags its children with it;
* the robot root is placed in the world frame, after the root normalisation and
  ``--ground`` correction.

Metres become millimetres exactly once: the ``scale`` factor is applied to geom
sizes and mesh vertices when the solid is built, and to the translation part of
every pose.  Rotations are of course unscaled.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cadquery as cq
import mujoco
import numpy as np
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder
from OCP.gp import gp_Ax2, gp_Dir, gp_Pnt
from OCP.TopoDS import TopoDS_Shape

from .frames import Pose
from .geoms import (
    GeomSelection,
    PRIMITIVE_TYPES,
    classify_geom,
    geom_type_name,
    primitive_solid,
)
from .load import LoadedModel, body_names, geom_names
from .meshes import MeshBuildResult, MeshData, build_meshes, extract_mesh, mesh_to_shape
from .names import NameMap

__all__ = ["PartGeom", "BodyNode", "RobotAssembly", "build_assembly", "JOINTS_NODE"]

JOINTS_NODE = "_JOINTS"


@dataclass
class PartGeom:
    """One geom, as a solid in its own frame plus where it sits on the body."""

    geom_id: int
    name: str
    safe_name: str
    shape: TopoDS_Shape
    geom_from_body: Pose
    rgba: tuple[float, float, float, float]
    type_name: str
    geom_class: str
    mesh_build: MeshBuildResult | None = None
    mesh: MeshData | None = None

    @property
    def is_mesh(self) -> bool:
        return self.mesh is not None

    def shape_in_body(self) -> cq.Shape:
        """The solid moved into the body frame."""
        return cq.Shape.cast(self.shape).moved(self.geom_from_body.to_location())


@dataclass
class BodyNode:
    body_id: int
    name: str
    safe_name: str
    parent_id: int
    body_from_parent: Pose
    world_pose: Pose
    parts: list[PartGeom] = field(default_factory=list)

    @property
    def has_geometry(self) -> bool:
        return bool(self.parts)

    def compound_in_body(self) -> cq.Shape | None:
        """All of this body's geoms, in the body frame, as one compound."""
        shapes = [p.shape_in_body() for p in self.parts]
        if not shapes:
            return None
        if len(shapes) == 1:
            return shapes[0]
        return cq.Compound.makeCompound(shapes)


@dataclass
class RobotAssembly:
    assembly: cq.Assembly
    nodes: dict[int, BodyNode]
    order: list[int]
    root_id: int
    scale: float
    units: str
    name_map: NameMap
    joint_marker_names: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def parts(self) -> list[PartGeom]:
        return [p for i in self.order for p in self.nodes[i].parts]

    def bodies_with_geometry(self) -> list[BodyNode]:
        return [self.nodes[i] for i in self.order if self.nodes[i].has_geometry]


def _geom_rgba(model: mujoco.MjModel, geom_id: int) -> tuple[float, float, float, float]:
    """Colour for a geom: its material's, if it has one, else its own rgba."""
    matid = int(model.geom_matid[geom_id])
    if matid >= 0:
        rgba = np.asarray(model.mat_rgba[matid], dtype=float)
    else:
        rgba = np.asarray(model.geom_rgba[geom_id], dtype=float)
    return tuple(float(np.clip(v, 0.0, 1.0)) for v in rgba)  # type: ignore[return-value]


def _keep_bodies(lm: LoadedModel, only: list[str] | str | None) -> set[int] | None:
    """Bodies whose geometry we emit, or ``None`` for all of them."""
    if not only:
        return None
    if isinstance(only, str):
        only = [part.strip() for part in only.split(",") if part.strip()]
    names = body_names(lm.model)
    wanted: set[int] = set()
    for item in only:
        if item not in names:
            raise ValueError(f"no body named {item!r}; try `sim2cad info`")
        wanted.add(names.index(item))
    return wanted


def build_assembly(
    lm: LoadedModel,
    selection: GeomSelection,
    *,
    scale: float = 1000.0,
    units: str = "mm",
    merge_coplanar: bool = True,
    decimate: float | None = None,
    joint_markers: bool = True,
    only: list[str] | str | None = None,
    jobs: int = 1,
    progress=None,
) -> RobotAssembly:
    """Convert the selected geoms into a nested :class:`cadquery.Assembly`."""
    model = lm.model
    bnames, gnames = body_names(model), geom_names(model)
    name_map = NameMap()
    notes: list[str] = []

    wanted = _keep_bodies(lm, only)
    subtree = lm.subtree()

    # With --only, keep the chosen bodies and every ancestor, so the parts stay
    # correctly placed relative to one another.
    if wanted is not None:
        keep: set[int] = set()
        for b in wanted:
            cur = b
            while cur != 0:
                keep.add(cur)
                cur = int(model.body_parentid[cur])
        order = [b for b in subtree if b in keep]
    else:
        order = list(subtree)

    # Mesh geoms are the expensive part and are independent of one another,
    # so convert them all up front, optionally across several processes.
    mesh_geoms = [
        g
        for body in order
        if wanted is None or body in wanted
        for g in selection.by_body.get(body, [])
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)
    ]
    built_meshes, mesh_warnings = build_meshes(
        model,
        str(lm.xml_path),
        mesh_geoms,
        scale=scale,
        decimate=decimate,
        merge_coplanar=merge_coplanar,
        jobs=jobs,
        progress=progress,
    )
    notes.extend(mesh_warnings)

    nodes: dict[int, BodyNode] = {}
    for body in order:
        parent = int(model.body_parentid[body])
        world = lm.body_pose(body)
        parent_world = lm.body_pose(parent) if parent != 0 else Pose.identity()
        rel = world.relative_to(parent_world) if parent != 0 else world
        nodes[body] = BodyNode(
            body_id=body,
            name=bnames[body],
            safe_name=name_map.add(bnames[body]),
            parent_id=parent,
            body_from_parent=rel.scaled(scale),
            world_pose=world,
        )

    for body in order:
        if wanted is not None and body not in wanted:
            continue
        node = nodes[body]
        body_world = lm.body_pose(body)
        for geom in selection.by_body.get(body, []):
            gtype = int(model.geom_type[geom])
            geom_from_body = lm.geom_pose(geom).relative_to(body_world).scaled(scale)
            label = gnames[geom]
            if progress is not None:
                progress(f"{bnames[body]} / {label} ({geom_type_name(gtype)})")

            if gtype in PRIMITIVE_TYPES:
                shape = primitive_solid(model, geom, scale)
                part = PartGeom(
                    geom_id=geom,
                    name=label,
                    safe_name=name_map.add(f"{bnames[body]}__{label}"),
                    shape=shape,
                    geom_from_body=geom_from_body,
                    rgba=_geom_rgba(model, geom),
                    type_name=geom_type_name(gtype),
                    geom_class=classify_geom(model, geom).value,
                )
            elif gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh, build = built_meshes[geom]
                part = PartGeom(
                    geom_id=geom,
                    name=label,
                    safe_name=name_map.add(f"{bnames[body]}__{label}"),
                    shape=build.shape,
                    geom_from_body=geom_from_body,
                    rgba=_geom_rgba(model, geom),
                    type_name="mesh",
                    geom_class=classify_geom(model, geom).value,
                    mesh_build=build,
                    mesh=mesh,
                )
                notes.extend(f"{label}: {n}" for n in build.notes)
            else:
                notes.append(f"{label}: unsupported geom type {geom_type_name(gtype)}, skipped")
                continue

            node.parts.append(part)

    root_id = order[0]
    top = cq.Assembly(name=name_map.add(f"{lm.name}_assembly"))
    cq_nodes: dict[int, cq.Assembly] = {}

    for body in order:
        node = nodes[body]
        sub = cq.Assembly(name=node.safe_name)
        for part in node.parts:
            sub.add(
                cq.Shape.cast(part.shape),
                name=part.safe_name,
                loc=part.geom_from_body.to_location(),
                color=cq.Color(*part.rgba),
            )
        cq_nodes[body] = sub

    # CadQuery copies a sub-assembly when you add it, so a node must be complete
    # before it is attached to its parent: walk the tree leaves-first.
    for body in reversed(order):
        node = nodes[body]
        if body == root_id:
            continue
        cq_nodes[node.parent_id].add(
            cq_nodes[body], name=node.safe_name, loc=node.body_from_parent.to_location()
        )
    root = nodes[root_id]
    top.add(cq_nodes[root_id], name=root.safe_name, loc=root.body_from_parent.to_location())

    marker_names: list[str] = []
    if joint_markers:
        marker_names = _add_joint_markers(lm, top, scale, name_map)

    return RobotAssembly(
        assembly=top,
        nodes=nodes,
        order=order,
        root_id=root_id,
        scale=scale,
        units=units,
        name_map=name_map,
        joint_marker_names=marker_names,
        notes=notes,
    )


#: Marker colour, chosen to stand out against typical robot greys.
MARKER_RGBA = (1.0, 0.35, 0.0, 1.0)


def _add_joint_markers(
    lm: LoadedModel, top: cq.Assembly, scale: float, name_map: NameMap
) -> list[str]:
    """A thin shaft plus an arrowhead along each hinge/slide axis, in world coords."""
    model, data = lm.model, lm.data
    from .load import joint_names

    jnames = joint_names(model)
    subtree = set(lm.subtree())
    extent = float(model.stat.extent) * scale
    shaft_len = 0.09 * extent
    shaft_r = 0.0035 * extent
    head_len = 0.03 * extent
    head_r = 0.011 * extent

    markers = cq.Assembly(name=JOINTS_NODE)
    names: list[str] = []

    for j in range(model.njnt):
        body = int(model.jnt_bodyid[j])
        if body not in subtree:
            continue
        jtype = int(model.jnt_type[j])
        if jtype not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue

        anchor = lm.world_fix.apply(np.asarray(data.xanchor[j], float)) * scale
        axis = lm.world_fix.rotate(np.asarray(data.xaxis[j], float))
        norm = float(np.linalg.norm(axis))
        if norm == 0:
            continue
        axis = axis / norm

        start = anchor - axis * (shaft_len / 2.0)
        frame = gp_Ax2(gp_Pnt(*start), gp_Dir(*axis))
        shaft = BRepPrimAPI_MakeCylinder(frame, shaft_r, shaft_len).Shape()
        head_base = anchor + axis * (shaft_len / 2.0)
        head_frame = gp_Ax2(gp_Pnt(*head_base), gp_Dir(*axis))
        head = BRepPrimAPI_MakeCone(head_frame, head_r, 0.0, head_len).Shape()

        safe = name_map.add(f"joint_{jnames[j]}")
        markers.add(
            cq.Compound.makeCompound([cq.Shape.cast(shaft), cq.Shape.cast(head)]),
            name=safe,
            color=cq.Color(*MARKER_RGBA),
        )
        names.append(safe)

    if names:
        top.add(markers, name=JOINTS_NODE)
    return names
