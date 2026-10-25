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

One mesh is not always one solid
--------------------------------
Four of the G1's visual meshes are several disconnected pieces stored in a
single mesh -- ``left_hip_roll_link`` is two, ``right_knee_link`` is eight --
every edge perfectly manifold, just not all joined up.  A ``TopoDS_Solid`` is
defined as *one* outer shell plus optional inner voids, so handing it several
disjoint outer shells produces a solid that ``BRepCheck_Analyzer`` rejects and
whose volume STEP reads back short, because the extra shells get taken for
cavities.  Each connected component therefore becomes its own solid, and the
components are collected into a compound.
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
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopoDS import TopoDS, TopoDS_Compound, TopoDS_Shape, TopoDS_Shell

from .geoms import unify_same_domain

__all__ = [
    "MeshData",
    "MeshComponent",
    "extract_mesh",
    "mesh_to_shape",
    "MeshBuildResult",
    "connected_components",
]


@dataclass
class MeshComponent:
    """One connected piece of a mesh, reindexed to its own vertices."""

    vertices: np.ndarray
    faces: np.ndarray
    watertight: bool
    volume: float

    @property
    def nfaces(self) -> int:
        return int(len(self.faces))


@dataclass
class MeshData:
    """A cleaned triangle mesh in its geom's frame, split into components."""

    name: str
    vertices: np.ndarray
    faces: np.ndarray
    watertight: bool
    repaired: bool
    volume: float
    faces_raw: int
    components: list[MeshComponent] = field(default_factory=list)
    decimated_to: int | None = None

    @property
    def nfaces(self) -> int:
        return int(len(self.faces))

    @property
    def n_components(self) -> int:
        return len(self.components)


def connected_components(faces: np.ndarray) -> list[np.ndarray]:
    """Face-index arrays, one per connected piece, joined along shared **edges**.

    Vertex connectivity is the wrong notion here.  Two patches that meet at a
    single vertex and share no edge cannot be one ``TopoDS_Shell``: OpenCascade
    splits them when writing STEP and duplicates the faces that bridge them, so
    the file comes back with more surface than it was given.  The ANYmal thigh
    has exactly that shape -- a 14-triangle patch attached to a 1494-triangle
    body at vertices only -- and the round trip gained those 14 faces and
    3,320 mm^2 of area.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components as _cc

    n_faces = len(faces)
    if n_faces == 0:
        return []

    # Every triangle's three edges, each stored with its endpoints sorted so
    # the two faces either side of an edge produce identical keys.
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.sort(edges, axis=1)
    owners = np.tile(np.arange(n_faces), 3)

    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owners = edges[order], owners[order]
    shared = np.all(edges[1:] == edges[:-1], axis=1)
    rows, cols = owners[:-1][shared], owners[1:][shared]

    if len(rows) == 0:
        return [np.array([i]) for i in range(n_faces)]

    # Linking consecutive owners of equal edges is enough: an edge shared by
    # three or more faces still ends up in one component, transitively.
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_faces, n_faces)
    )
    count, labels = _cc(adjacency, directed=False)
    if count == 1:
        return [np.arange(n_faces)]
    return [np.nonzero(labels == i)[0] for i in range(count)]


def _split_components(mesh: trimesh.Trimesh) -> list[MeshComponent]:
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    out: list[MeshComponent] = []
    for group in connected_components(faces):
        sub_faces = faces[group]
        used = np.unique(sub_faces)
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        piece = trimesh.Trimesh(
            vertices=vertices[used], faces=remap[sub_faces], process=False
        )
        out.append(
            MeshComponent(
                vertices=np.asarray(piece.vertices, dtype=float),
                faces=np.asarray(piece.faces, dtype=np.int64),
                watertight=bool(piece.is_watertight),
                volume=float(piece.volume) if piece.is_watertight else float("nan"),
            )
        )
    return out


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
        components=_split_components(mesh),
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
    n_components: int = 1
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


def _area(shape: TopoDS_Shape) -> float:
    """Total surface area -- the only size an open shell has."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, props)
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


def _component_shape(component: MeshComponent) -> tuple[TopoDS_Shape, bool, int]:
    """One connected piece as a solid if it is closed, else as an open shell."""
    shell, dropped = _topological_shell(component.vertices, component.faces)
    if not component.watertight:
        return shell, False, dropped

    # The shell is sewn and manifold by construction, so ShapeFix_Shell and
    # ShapeFix_Solid have nothing to find -- and on a 45k-triangle mesh they
    # cost about 37 seconds between them.  Worse, ShapeFix_Solid silently
    # drops geometry when handed anything unusual.  Build the solid directly.
    shell.Closed(True)
    maker = BRepBuilderAPI_MakeSolid(shell)
    if not maker.IsDone():
        return shell, False, dropped

    solid = maker.Solid()
    if _volume(solid) < 0.0:
        # Inward-facing shell: flip it rather than run a repair pass.
        solid = TopoDS.Solid_s(solid.Reversed())
    return solid, True, dropped


def mesh_to_shape(mesh: MeshData, merge_coplanar: bool = True) -> MeshBuildResult:
    """Turn a cleaned mesh into STEP-ready solids (or shells, if it is open)."""
    notes: list[str] = []
    components = mesh.components or [
        MeshComponent(mesh.vertices, mesh.faces, mesh.watertight, mesh.volume)
    ]

    shapes: list[TopoDS_Shape] = []
    dropped = 0
    all_solid = True
    open_pieces = 0
    for component in components:
        shape, is_solid, lost = _component_shape(component)
        shapes.append(shape)
        dropped += lost
        if not is_solid:
            all_solid = False
            open_pieces += 1

    if dropped:
        notes.append(f"dropped {dropped} degenerate triangle(s)")
    if open_pieces:
        notes.append(
            f"{open_pieces} of {len(components)} piece(s) not watertight; "
            "exported as open shell(s)"
        )
    if len(components) > 1:
        notes.append(f"mesh is {len(components)} disconnected pieces; one solid each")

    if len(shapes) == 1:
        shape: TopoDS_Shape = shapes[0]
    else:
        builder = BRep_Builder()
        compound = TopoDS_Compound()
        builder.MakeCompound(compound)
        for piece in shapes:
            builder.Add(compound, piece)
        shape = compound

    faces_before = _count_faces(shape)
    volume_before = _volume(shape) if all_solid else float("nan")

    if merge_coplanar:
        merged = unify_same_domain(shape)
        if not merged.IsNull():
            shape = merged

    volume_after = _volume(shape) if all_solid else float("nan")

    valid = bool(BRepCheck_Analyzer(shape).IsValid())
    if not valid:
        # Some Menagerie visual meshes have non-manifold edges -- a few edges
        # shared by three or four triangles -- which OCC rejects.  ShapeFix can
        # resolve those, but it is also capable of quietly discarding geometry.
        # Volume is the sharper guard where there is one; an open shell has no
        # volume, so its surface area has to stand in.  Rejecting the repair
        # outright over a small area change would only swap one failure for
        # another -- the part would stay invalid -- so a change that is not
        # outright destructive is recorded and left for the placement check,
        # which measures the surface centroid and sees exactly this.
        area_before = _area(shape)
        fixer = ShapeFix_Shape(shape)
        fixer.Perform()
        fixed = fixer.Shape()
        if not fixed.IsNull() and BRepCheck_Analyzer(fixed).IsValid():
            fixed_volume = _volume(fixed) if all_solid else float("nan")
            volume_shift = (
                abs(fixed_volume - volume_after) / abs(volume_after)
                if all_solid and np.isfinite(volume_after) and volume_after != 0
                else 0.0
            )
            area_shift = (
                abs(_area(fixed) - area_before) / area_before if area_before > 0 else 0.0
            )
            if volume_shift > 1e-6:
                notes.append(
                    f"ShapeFix would have changed the volume by {volume_shift:.1%}; rejected"
                )
            else:
                shape = fixed
                valid = True
                volume_after = fixed_volume
                if area_shift > 1e-6:
                    notes.append(
                        f"ShapeFix repaired non-manifold edges and moved "
                        f"{area_shift:.2%} of the surface"
                    )
                else:
                    notes.append("ShapeFix repaired non-manifold edges")

    return MeshBuildResult(
        shape=shape,
        is_solid=all_solid,
        valid=valid,
        faces_before=faces_before,
        faces_after=_count_faces(shape),
        n_components=len(components),
        volume_before_merge=volume_before,
        volume_after_merge=volume_after,
        dropped_triangles=dropped,
        notes=notes,
    )


# -- parallel building -----------------------------------------------------
#
# The 35 mesh geoms of the G1 are independent of one another and took 522 s of
# the 25-minute export, on one of four cores.  TopoDS_Shape cannot be pickled,
# so workers hand their result back as a .brep file.  That round trip keeps the
# topology exactly (same face count, same component count) and the geometry to
# within a couple of ULP, since .brep writes coordinates as decimal text; a
# test asserts the parallel and serial paths agree to 1e-12 relative.


@dataclass
class _BuildMeta:
    """The picklable half of :class:`MeshBuildResult`."""

    is_solid: bool
    valid: bool
    faces_before: int
    faces_after: int
    n_components: int
    volume_before_merge: float
    volume_after_merge: float
    dropped_triangles: int
    notes: list[str]

    @staticmethod
    def of(result: "MeshBuildResult") -> "_BuildMeta":
        return _BuildMeta(
            result.is_solid,
            result.valid,
            result.faces_before,
            result.faces_after,
            result.n_components,
            result.volume_before_merge,
            result.volume_after_merge,
            result.dropped_triangles,
            list(result.notes),
        )

    def with_shape(self, shape: TopoDS_Shape) -> "MeshBuildResult":
        return MeshBuildResult(
            shape=shape,
            is_solid=self.is_solid,
            valid=self.valid,
            faces_before=self.faces_before,
            faces_after=self.faces_after,
            n_components=self.n_components,
            volume_before_merge=self.volume_before_merge,
            volume_after_merge=self.volume_after_merge,
            dropped_triangles=self.dropped_triangles,
            notes=list(self.notes),
        )


_WORKER: dict = {}


def _worker_init(xml_path: str) -> None:  # pragma: no cover - subprocess
    import mujoco

    _WORKER["model"] = mujoco.MjModel.from_xml_path(xml_path)


def _worker_build(args):  # pragma: no cover - subprocess
    import os
    import tempfile

    from OCP.BRepTools import BRepTools

    geom_id, scale, decimate, merge_coplanar = args
    mesh = extract_mesh(_WORKER["model"], geom_id, scale=scale, decimate=decimate)
    result = mesh_to_shape(mesh, merge_coplanar=merge_coplanar)
    handle, path = tempfile.mkstemp(suffix=".brep")
    os.close(handle)
    BRepTools.Write_s(result.shape, path)
    return geom_id, mesh, _BuildMeta.of(result), path


def _read_brep(path: str) -> TopoDS_Shape:
    from OCP.BRepTools import BRepTools

    shape = TopoDS_Shape()
    BRepTools.Read_s(shape, path, BRep_Builder())
    return shape


def build_meshes(
    model: mujoco.MjModel,
    xml_path: str,
    geom_ids: list[int],
    *,
    scale: float,
    decimate: float | None,
    merge_coplanar: bool,
    jobs: int = 1,
    progress=None,
) -> tuple[dict[int, tuple[MeshData, MeshBuildResult]], list[str]]:
    """Convert every mesh geom, in parallel when asked and when it is worth it."""
    out: dict[int, tuple[MeshData, MeshBuildResult]] = {}
    warnings: list[str] = []
    if not geom_ids:
        return out, warnings

    if jobs > 1 and len(geom_ids) > 1:
        try:
            return _build_meshes_parallel(
                xml_path,
                geom_ids,
                scale=scale,
                decimate=decimate,
                merge_coplanar=merge_coplanar,
                jobs=jobs,
                progress=progress,
            )
        except Exception as exc:
            # Falling back is about what this machine can do, not about
            # correctness: the serial path below produces identical shapes,
            # which the test suite checks.
            warnings.append(
                f"parallel mesh build unavailable ({type(exc).__name__}: {exc}); "
                "built serially"
            )

    for geom_id in geom_ids:
        if progress is not None:
            progress(f"mesh geom {geom_id}")
        mesh = extract_mesh(model, geom_id, scale=scale, decimate=decimate)
        out[geom_id] = (mesh, mesh_to_shape(mesh, merge_coplanar=merge_coplanar))
    return out, warnings


def _build_meshes_parallel(
    xml_path: str,
    geom_ids: list[int],
    *,
    scale: float,
    decimate: float | None,
    merge_coplanar: bool,
    jobs: int,
    progress=None,
):
    import multiprocessing as mp
    import os

    out: dict[int, tuple[MeshData, MeshBuildResult]] = {}
    context = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    tasks = [(g, scale, decimate, merge_coplanar) for g in geom_ids]
    done = 0
    with context.Pool(
        processes=min(jobs, len(geom_ids)), initializer=_worker_init, initargs=(xml_path,)
    ) as pool:
        for geom_id, mesh, meta, path in pool.imap_unordered(_worker_build, tasks):
            try:
                out[geom_id] = (mesh, meta.with_shape(_read_brep(path)))
            finally:
                try:
                    os.unlink(path)
                except OSError:  # pragma: no cover
                    pass
            done += 1
            if progress is not None:
                progress(f"meshes {done}/{len(geom_ids)}")
    return out, []
