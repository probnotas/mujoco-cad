"""``--printable``: one slicer-ready STL per link.

The STEP assembly is built for CAD, where an open shell is a perfectly
reasonable thing to hand someone.  A slicer is not so forgiving: it wants a
closed, manifold solid, and 12 of the G1's 35 parts are not one.  This module
takes each body's geometry, repairs it as gently as it can, and writes an STL
that a printer will accept -- saying clearly in the report what it had to do,
because some of the repairs change the shape.

What it does **not** do is make the robot buildable.  These are the outer
envelopes of the links, with no wall thickness, fasteners, bearing seats or
motor pockets, so printing the set gives you a model of a G1, not a G1.  See
the README.

Everything here is written in millimetres whatever ``--units`` says, because
that is what slicers assume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh

__all__ = ["PrintablePart", "body_mesh", "make_printable", "write_printable"]

#: How far a tessellated primitive may deviate from the true surface, as a
#: fraction of the part's own size.  Relative rather than absolute because
#: OpenCascade's tessellator is millimetre-native: the same absolute deflection
#: applied to a metre-scale shape produced 201k triangles where the millimetre
#: build produced 8k.  A relative tolerance gives identical output whatever
#: ``--units`` says, and 5e-4 is about 0.05 mm on a 100 mm link -- finer than
#: any nozzle.
TESSELLATION_TOLERANCE = 5e-4


@dataclass
class PrintablePart:
    body: str
    path: Path | None
    watertight: bool
    method: str
    lossy: bool
    pieces_in: int
    pieces_out: int
    faces: int
    volume_mm3: float
    volume_shift: float | None = None
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "body": self.body,
            "file": str(self.path) if self.path else None,
            "watertight": self.watertight,
            "method": self.method,
            "lossy": self.lossy,
            "pieces_in": self.pieces_in,
            "pieces_out": self.pieces_out,
            "faces": self.faces,
            "volume_mm3": self.volume_mm3,
            "volume_shift": self.volume_shift,
            "note": self.note,
        }


def _to_millimetres(shape, factor: float):
    """A uniformly scaled copy of a shape, so tessellation happens in mm."""
    if factor == 1.0:
        return shape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Pnt, gp_Trsf

    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), factor)
    builder = BRepBuilderAPI_Transform(shape, trsf, True)
    builder.Build()
    return builder.Shape() if builder.IsDone() else shape


def _tessellate(
    shape, to_mm: float, relative: float = TESSELLATION_TOLERANCE
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate a primitive, in millimetres, to a size-relative deflection.

    The shape is scaled to millimetres first.  A relative deflection alone is
    not enough: OpenCascade also clamps against its own absolute tolerances, so
    the same shape written in metres tessellated to 64k triangles where the
    millimetre build gave 8k.  Scaling into the kernel's native units first
    makes the output identical whatever ``--units`` says.
    """
    import cadquery as cq

    from .stepread import shape_bbox

    scaled = _to_millimetres(shape, to_mm)
    low, high = shape_bbox(scaled, optimal=False)
    diagonal = float(np.linalg.norm(high - low))
    tolerance = max(diagonal * relative, 1e-9)
    verts, tris = cq.Shape.cast(scaled).tessellate(tolerance)
    v = np.array([[p.x, p.y, p.z] for p in verts], dtype=float)
    f = np.array(tris, dtype=np.int64).reshape(-1, 3)
    return v, f


def body_mesh(node, scale: float) -> trimesh.Trimesh | None:
    """Every geom of one body, in the body frame, in millimetres.

    Mesh geoms reuse the cleaned triangles we already have.  Primitives are
    tessellated, since a sphere has to become triangles sooner or later and a
    slicer is the place for it.
    """
    to_mm = 1000.0 / scale
    chunks: list[trimesh.Trimesh] = []

    placement = None
    for part in node.parts:
        if part.mesh is not None:
            verts = part.mesh.vertices * to_mm
            faces = part.mesh.faces
        else:
            verts, faces = _tessellate(part.shape, to_mm)
        if len(faces) == 0:
            continue
        # Vertices are already in millimetres, so the pose's translation has
        # to be too; its rotation is unitless.
        placement = part.geom_from_body.scaled(to_mm)
        chunks.append(
            trimesh.Trimesh(vertices=placement.apply(verts), faces=faces, process=False)
        )

    if not chunks:
        return None
    mesh = trimesh.util.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return mesh


def _components(mesh: trimesh.Trimesh) -> list[trimesh.Trimesh]:
    try:
        pieces = mesh.split(only_watertight=False)
    except Exception:  # pragma: no cover - degenerate input
        return [mesh]
    return [p for p in pieces if len(p.faces)] or [mesh]


def _fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    patched = mesh.copy()
    trimesh.repair.fill_holes(patched)
    trimesh.repair.fix_winding(patched)
    patched.fix_normals()
    return patched


def _meshfix(mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """MeshFix: closes holes and strips non-manifold geometry."""
    try:
        from pymeshfix import MeshFix
    except Exception:  # pragma: no cover - optional extra
        return None
    try:
        fixer = MeshFix(np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int32))
        fixer.repair(verbose=False)
        if len(fixer.f) == 0:
            return None
        return trimesh.Trimesh(vertices=fixer.v, faces=fixer.f, process=True)
    except Exception:  # pragma: no cover - MeshFix gives up on some inputs
        return None


def _voxel_remesh(mesh: trimesh.Trimesh, divisions: int = 192) -> trimesh.Trimesh | None:
    """Last resort: re-surface the volume on a grid.

    Guaranteed closed, and visibly lossy -- sharp edges round off at the grid
    pitch -- so it is only reached when everything gentler has failed, and it
    is always reported.
    """
    try:
        diagonal = float(np.linalg.norm(mesh.extents))
        if diagonal <= 0:
            return None
        pitch = diagonal / divisions
        filled = mesh.voxelized(pitch=pitch).fill()
        out = filled.marching_cubes
        out.merge_vertices()
        out.fix_normals()
        return out if out.is_watertight else None
    except Exception:  # pragma: no cover - voxelisation can run out of memory
        return None


def make_printable(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, str, bool, float | None]:
    """Repair towards watertight, gently first.

    Returns the mesh, the method that worked, whether it changed the shape
    materially, and the relative volume shift where that can be measured.
    """
    if mesh.is_watertight:
        return mesh, "already watertight", False, 0.0

    reference = None
    if mesh.is_volume:  # pragma: no cover - rare for an open mesh
        reference = float(mesh.volume)

    patched = _fill_holes(mesh)
    if patched.is_watertight:
        shift = _shift(reference, patched)
        return patched, "holes filled", False, shift

    # Repair each piece on its own: MeshFix keeps only the largest shell when
    # it is handed several at once, which would quietly delete parts of a link.
    pieces = _components(mesh)
    repaired: list[trimesh.Trimesh] = []
    any_meshfix = False
    for piece in pieces:
        if piece.is_watertight:
            repaired.append(piece)
            continue
        filled = _fill_holes(piece)
        if filled.is_watertight:
            repaired.append(filled)
            continue
        fixed = _meshfix(piece)
        if fixed is not None and fixed.is_watertight:
            repaired.append(fixed)
            any_meshfix = True
            continue
        voxelised = _voxel_remesh(piece)
        if voxelised is not None:
            repaired.append(voxelised)
            any_meshfix = True
            continue
        repaired.append(piece)  # keep it; it will be reported as not watertight

    combined = trimesh.util.concatenate(repaired) if len(repaired) > 1 else repaired[0]
    combined.merge_vertices()
    combined.fix_normals()
    method = "meshfix per piece" if any_meshfix else "holes filled per piece"
    return combined, method, any_meshfix, _shift(reference, combined)


def _shift(reference: float | None, mesh: trimesh.Trimesh) -> float | None:
    if reference is None or reference == 0 or not mesh.is_volume:
        return None
    return abs(float(mesh.volume) - reference) / abs(reference)


def _union_overlapping(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, int]:
    """Fuse pieces that overlap, so a link prints as one object where it is one.

    Pieces that merely sit near each other are left alone: a slicer treats them
    as separate objects on the plate, which is what they are.
    """
    pieces = _components(mesh)
    if len(pieces) < 2:
        return mesh, len(pieces), len(pieces)
    if not all(p.is_watertight for p in pieces):
        return mesh, len(pieces), len(pieces)

    boxes = [p.bounds for p in pieces]
    overlaps = False
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            if np.all(boxes[i][0] <= boxes[j][1]) and np.all(boxes[j][0] <= boxes[i][1]):
                overlaps = True
                break
        if overlaps:
            break
    if not overlaps:
        return mesh, len(pieces), len(pieces)

    try:
        fused = trimesh.boolean.union(pieces)
    except Exception:  # pragma: no cover - backend missing
        return mesh, len(pieces), len(pieces)
    if fused is None or not fused.is_watertight:
        return mesh, len(pieces), len(pieces)
    return fused, len(pieces), len(_components(fused))


def write_printable(result, out_dir: Path | None = None) -> list[PrintablePart]:
    """Write ``printable/<body>.stl`` for every body with geometry."""
    out_dir = Path(out_dir or (result.out_dir / "printable"))
    out_dir.mkdir(parents=True, exist_ok=True)

    report: list[PrintablePart] = []
    for node in result.robot.bodies_with_geometry():
        mesh = body_mesh(node, result.scale)
        if mesh is None:
            continue

        pieces_in = len(_components(mesh))
        repaired, method, lossy, shift = make_printable(mesh)
        fused, _, pieces_out = _union_overlapping(repaired)
        if fused is not repaired:
            method += " + union"

        path = out_dir / f"{node.safe_name}.stl"
        fused.export(path)

        note = ""
        if not fused.is_watertight:
            note = "still open after repair; a slicer will have to guess"
        elif lossy:
            note = "geometry was rebuilt, not just patched; check the shape"

        report.append(
            PrintablePart(
                body=node.name,
                path=path,
                watertight=bool(fused.is_watertight),
                method=method,
                lossy=lossy,
                pieces_in=pieces_in,
                pieces_out=pieces_out,
                faces=int(len(fused.faces)),
                volume_mm3=float(fused.volume) if fused.is_volume else float("nan"),
                volume_shift=shift,
                note=note,
            )
        )
    return report
