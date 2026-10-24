"""Mesh extraction from the compiled model, and mesh -> B-rep conversion.

The recentering pitfall
-----------------------
MuJoCo's compiler moves every mesh so its frame sits at the mesh's own centre,
and compensates by adjusting the geom's pos/quat.  The compiled
``model.mesh_vert`` therefore lives in the **geom's** frame, and the correct
world geometry is::

    world_vertex = geom_xpos + geom_xmat @ mesh_vert

Loading the original STL from disk and applying the geom's *XML* pos/quat gives
a different, wrong answer -- for the G1 the recentering offsets reach 0.43 m.
So we never touch the source STL files.

Mesh -> solid
-------------
``BRepBuilderAPI_Sewing`` is the obvious way to join triangles into a shell, but
it searches geometrically for coincident points and goes quadratic: 20k
triangles take ~20 s, and the G1's torso alone has 51k.  We already know the
connectivity -- the mesh is indexed -- so instead we build the shell
topologically, creating each vertex once and each edge once and sharing them
between the faces that meet there.  That is linear, and the resulting shell is
sewn by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np
import trimesh
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeVertex,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.gp import gp_Dir, gp_Pln, gp_Pnt
from OCP.ShapeFix import ShapeFix_Shape, ShapeFix_Shell, ShapeFix_Solid
from OCP.TopoDS import TopoDS, TopoDS_Shape, TopoDS_Shell

from .geoms import unify_same_domain

__all__ = [
    "MeshData",
    "extract_mesh",
    "mesh_to_shape",
    "MeshBuildResult",
]


@dataclass
class MeshData:
    """A cleaned triangle mesh in its geom's frame."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    watertight: bool
    repaired: bool
    volume: float
    faces_raw: int
    decimated_to: int | None = None

    @property
    def nfaces(self) -> int:
        return int(len(self.faces))


def extract_mesh(
    model: mujoco.MjModel,
    geom_id: int,
    scale: float = 1.0,
    decimate: float | None = None,
    repair: bool = True,
) -> MeshData:
    """Pull a mesh geom's triangles out of the compiled model and clean them up."""
    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError(f"geom {geom_id} is not a mesh")

    mesh_id = int(model.geom_dataid[geom_id])
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or f"mesh{mesh_id}"

    vadr, vnum = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    fadr, fnum = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])

    # Compiled verts are already in the geom frame; scale once, here.
    verts = np.asarray(model.mesh_vert[vadr : vadr + vnum], dtype=float).reshape(-1, 3) * scale
    faces = np.asarray(model.mesh_face[fadr : fadr + fnum], dtype=np.int64).reshape(-1, 3)
    faces_raw = int(len(faces))

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    decimated_to = None
    if decimate is not None and 0.0 < decimate < 1.0:
        target = max(4, int(len(mesh.faces) * decimate))
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target)
            mesh.merge_vertices()
            mesh.update_faces(mesh.nondegenerate_faces())
            mesh.remove_unreferenced_vertices()
            mesh.fix_normals()
            decimated_to = int(len(mesh.faces))
        except Exception:  # pragma: no cover - depends on optional backend
            decimated_to = None

    repaired = False
    if repair and not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        trimesh.repair.fix_winding(mesh)
        mesh.fix_normals()
        repaired = True

    return MeshData(
        name=name,
        vertices=np.asarray(mesh.vertices, dtype=float),
        faces=np.asarray(mesh.faces, dtype=np.int64),
        watertight=bool(mesh.is_watertight),
        repaired=repaired,
        volume=float(mesh.volume) if mesh.is_watertight else float("nan"),
        faces_raw=faces_raw,
        decimated_to=decimated_to,
    )


@dataclass
class MeshBuildResult:
    """What came out of the B-rep conversion, for the report."""

    shape: TopoDS_Shape
    is_solid: bool
    valid: bool
    faces_before: int
    faces_after: int
    volume_before_merge: float = float("nan")
    volume_after_merge: float = float("nan")
    dropped_triangles: int = 0
    notes: list[str] = field(default_factory=list)


def _topological_shell(vertices: np.ndarray, faces: np.ndarray) -> tuple[TopoDS_Shell, int]:
    """Build a sewn shell by sharing vertices and edges between triangles."""
    builder = BRep_Builder()
    shell = TopoDS_Shell()
    builder.MakeShell(shell)

    occ_verts = [BRepBuilderAPI_MakeVertex(gp_Pnt(*v)).Vertex() for v in vertices]
    edges: dict[tuple[int, int], object] = {}

    def oriented_edge(i: int, j: int):
        key = (i, j) if i < j else (j, i)
        edge = edges.get(key)
        if edge is None:
            edge = BRepBuilderAPI_MakeEdge(occ_verts[key[0]], occ_verts[key[1]]).Edge()
            edges[key] = edge
        return edge if i < j else TopoDS.Edge_s(edge.Reversed())

    dropped = 0
    added = 0
    for tri in faces:
        i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
        p0, p1, p2 = vertices[i], vertices[j], vertices[k]
        normal = np.cross(p1 - p0, p2 - p0)
        length = float(np.linalg.norm(normal))
        if length < 1e-24:
            dropped += 1
            continue
        plane = gp_Pln(gp_Pnt(*p0), gp_Dir(*(normal / length)))
        wire = BRepBuilderAPI_MakeWire(
            oriented_edge(i, j), oriented_edge(j, k), oriented_edge(k, i)
        )
        if not wire.IsDone():
            dropped += 1
            continue
        face = BRepBuilderAPI_MakeFace(plane, wire.Wire())
        if not face.IsDone():
            dropped += 1
            continue
        builder.Add(shell, face.Face())
        added += 1

    if added == 0:
        raise RuntimeError("no usable triangles in mesh")
    return shell, dropped


def _volume(shape: TopoDS_Shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def _count_faces(shape: TopoDS_Shape) -> int:
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE

    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    count = 0
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def mesh_to_shape(mesh: MeshData, merge_coplanar: bool = True) -> MeshBuildResult:
    """Turn a cleaned mesh into a STEP-ready solid (or a shell, if it is open)."""
    shell, dropped = _topological_shell(mesh.vertices, mesh.faces)
    notes: list[str] = []
    if dropped:
        notes.append(f"dropped {dropped} degenerate triangle(s)")

    faces_before = _count_faces(shell)

    shape: TopoDS_Shape = shell
    is_solid = False
    if mesh.watertight:
        shell.Closed(True)
        fixer = ShapeFix_Shell(shell)
        fixer.Perform()
        fixed_shell = fixer.Shell() if fixer.NbShells() == 1 else shell
        maker = BRepBuilderAPI_MakeSolid(fixed_shell)
        if maker.IsDone():
            solid = maker.Solid()
            solid_fix = ShapeFix_Solid(solid)
            solid_fix.Perform()
            fixed = solid_fix.Solid()
            shape = fixed if not fixed.IsNull() else solid
            is_solid = True
        else:
            notes.append("MakeSolid failed; exported as an open shell")
    else:
        notes.append("mesh is not watertight; exported as an open shell")

    volume_before = _volume(shape) if is_solid else float("nan")

    if merge_coplanar:
        merged = unify_same_domain(shape)
        if not merged.IsNull():
            shape = merged

    volume_after = _volume(shape) if is_solid else float("nan")

    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    if not valid:
        # Some Menagerie visual meshes have non-manifold edges -- a few edges
        # shared by three or four triangles -- which OCC rejects.  ShapeFix
        # resolves them without dropping faces; only keep its answer if it
        # actually helped.
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        fixed = fixer.Shape()
        if not fixed.IsNull() and BRepCheck_Analyzer(fixed).IsValid():
            shape = fixed
            valid = True
            notes.append("ShapeFix repaired non-manifold edges")
    return MeshBuildResult(
        shape=shape,
        is_solid=is_solid,
        valid=valid,
        faces_before=faces_before,
        faces_after=_count_faces(shape),
        volume_before_merge=volume_before,
        volume_after_merge=volume_after,
        dropped_triangles=dropped,
        notes=notes,
    )
