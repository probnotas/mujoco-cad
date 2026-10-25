"""The export pipeline and the file writers."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import cadquery as cq
import numpy as np
from cadquery.occ_impl.exporters.assembly import exportAssembly

from .assembly import RobotAssembly, build_assembly
from .frames import Pose
from .geoms import GeomSelection, geom_local_points, select_geoms
from .joints import joints_document
from .load import LoadedModel, apply_pose, ground_world_fix, load_model, set_world_fix
from .occunits import cascade_unit

__all__ = ["ExportResult", "run_export", "UNIT_SCALE", "write_step_assembly"]

#: Metres -> export unit.  Applied once, when solids and poses are built.
UNIT_SCALE = {"mm": 1000.0, "m": 1.0}
_STEP_UNIT = {"mm": "MM", "m": "M"}


@dataclass
class ExportResult:
    out_dir: Path
    lm: LoadedModel
    selection: GeomSelection
    robot: RobotAssembly
    assembly_path: Path
    part_paths: dict[str, Path] = field(default_factory=dict)
    stl_paths: dict[str, Path] = field(default_factory=dict)
    joints_path: Path | None = None
    preview_path: Path | None = None
    render_paths: dict[str, Path] = field(default_factory=dict)
    printable: list = field(default_factory=list)
    render_skipped_because: str | None = None
    units: str = "mm"
    scale: float = 1000.0
    timings: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    #: The arguments this export was run with, so `sim2cad validate` can
    #: reproduce it exactly rather than guessing at the defaults.
    options: dict = field(default_factory=dict)


def _lowest_z(lm: LoadedModel, selection: GeomSelection) -> float:
    """Lowest point of the selected geometry, in corrected world metres."""
    lowest = np.inf
    for geom in selection.geom_ids:
        pts = geom_local_points(lm.model, geom)
        world = lm.geom_pose(geom).apply(pts)
        lowest = min(lowest, float(world[:, 2].min()))
    return 0.0 if not np.isfinite(lowest) else lowest


def write_step_assembly(robot: RobotAssembly, path: Path) -> None:
    """Write the nested assembly, declaring the unit the geometry is in."""
    unit = _STEP_UNIT[robot.units]
    with cascade_unit(unit):
        exportAssembly(robot.assembly, str(path), mode="default", unit=unit)


def _tessellate(shape: cq.Shape, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    verts, tris = shape.tessellate(tolerance)
    v = np.array([[p.x, p.y, p.z] for p in verts], dtype=float)
    f = np.array(tris, dtype=np.int64).reshape(-1, 3)
    return v, f


def write_preview_glb(result: ExportResult, path: Path, tolerance: float = 0.2) -> None:
    """A quick-look GLB, tessellated from the *exported CAD*, in metres.

    Tessellating the B-rep rather than re-dumping the MuJoCo triangles is what
    makes the visual check meaningful: if the CAD is wrong, the preview is wrong
    in the same way.  glTF is metre-based, so we convert on the way out.
    """
    import trimesh

    scene = trimesh.Scene()
    to_metres = 1.0 / UNIT_SCALE[result.units]
    for node in result.robot.bodies_with_geometry():
        world = node.world_pose.scaled(result.scale)
        for part in node.parts:
            placed = world @ part.geom_from_body
            try:
                v, f = _tessellate(cq.Shape.cast(part.shape), tolerance)
            except Exception:  # pragma: no cover - degenerate shapes
                continue
            if len(f) == 0:
                continue
            mesh = trimesh.Trimesh(vertices=placed.apply(v) * to_metres, faces=f, process=False)
            mesh.visual.face_colors = np.tile(
                (np.array(part.rgba) * 255).astype(np.uint8), (len(f), 1)
            )
            scene.add_geometry(mesh, node_name=part.safe_name, geom_name=part.safe_name)
    path.write_bytes(trimesh.exchange.gltf.export_glb(scene))


def write_part_files(
    result: ExportResult, parts_dir: Path, write_stl: bool = True, tolerance: float = 0.1
) -> None:
    """Per-link STEP (and STL), each with its origin at the link's own frame."""
    parts_dir.mkdir(parents=True, exist_ok=True)
    for node in result.robot.bodies_with_geometry():
        compound = node.compound_in_body()
        if compound is None:
            continue
        sub = cq.Assembly(name=node.safe_name)
        for part in node.parts:
            sub.add(
                cq.Shape.cast(part.shape),
                name=part.safe_name,
                loc=part.geom_from_body.to_location(),
                color=cq.Color(*part.rgba),
            )
        step_path = parts_dir / f"{node.safe_name}.step"
        unit = _STEP_UNIT[result.units]
        with cascade_unit(unit):
            exportAssembly(sub, str(step_path), mode="default", unit=unit)
        result.part_paths[node.name] = step_path

        if write_stl:
            stl_path = parts_dir / f"{node.safe_name}.stl"
            cq.exporters.export(
                cq.Workplane(obj=compound), str(stl_path), tolerance=tolerance
            )
            result.stl_paths[node.name] = stl_path


def run_export(
    *,
    menagerie: str | None = None,
    mjcf: str | Path | None = None,
    entry: str | None = None,
    geoms: str = "visual",
    pose: str = "qpos0",
    units: str = "mm",
    decimate: float | None = None,
    merge_coplanar: bool = True,
    joint_markers: bool = True,
    ground: bool = False,
    keep_root_pose: bool = False,
    only: list[str] | str | None = None,
    out: str | Path | None = None,
    write_parts: bool = True,
    write_preview: bool = True,
    write_render: bool = True,
    printable: bool = False,
    jobs: int = 0,
    progress=None,
) -> ExportResult:
    """Run the whole pipeline and write everything into ``out``."""
    if units not in UNIT_SCALE:
        raise ValueError(f"unknown unit {units!r}; expected mm or m")
    scale = UNIT_SCALE[units]
    if jobs <= 0:
        jobs = os.cpu_count() or 1
    timings: dict[str, float] = {}

    def step(label: str):
        if progress is not None:
            progress(label)
        return time.perf_counter()

    t0 = step("loading model")
    lm = load_model(menagerie=menagerie, mjcf=mjcf, entry=entry)
    apply_pose(lm, pose)
    set_world_fix(lm, keep_root_pose=keep_root_pose)
    timings["load"] = time.perf_counter() - t0

    t0 = step("selecting geoms")
    selection = select_geoms(lm, geoms)
    if ground:
        ground_world_fix(lm, _lowest_z(lm, selection))
    timings["select"] = time.perf_counter() - t0

    out_dir = Path(out) if out else Path("out") / f"{lm.name}_{lm.pose_label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = step("building solids")
    robot = build_assembly(
        lm,
        selection,
        scale=scale,
        units=units,
        merge_coplanar=merge_coplanar,
        decimate=decimate,
        joint_markers=joint_markers,
        only=only,
        jobs=jobs,
        progress=progress,
    )
    timings["build"] = time.perf_counter() - t0

    result = ExportResult(
        out_dir=out_dir,
        lm=lm,
        selection=selection,
        robot=robot,
        assembly_path=out_dir / "assembly.step",
        units=units,
        scale=scale,
        timings=timings,
        options={
            "menagerie": menagerie,
            "mjcf": str(mjcf) if mjcf else None,
            "entry": entry,
            "geoms": geoms,
            "pose": pose,
            "units": units,
            "decimate": decimate,
            "merge_coplanar": merge_coplanar,
            "joint_markers": joint_markers,
            "ground": ground,
            "keep_root_pose": keep_root_pose,
            "only": only,
            "printable": printable,
            "jobs": jobs,
        },
    )
    result.warnings.extend(robot.notes)
    roots = lm.root_bodies()
    if len(roots) > 1:
        from .load import body_names

        names = body_names(lm.model)
        result.warnings.append(
            f"model has {len(roots)} bodies attached to the world; exporting only "
            f"the subtree under {names[roots[0]]!r} and ignoring "
            f"{', '.join(names[b] for b in roots[1:][:5])}"
            + (" ..." if len(roots) > 6 else "")
        )
    if selection.fallbacks:
        result.warnings.append(
            f"{len(selection.fallbacks)} bod(y|ies) had no {geoms} geoms and fell back "
            f"to what they had: {', '.join(sorted(set(selection.fallbacks))[:8])}"
            + (" ..." if len(set(selection.fallbacks)) > 8 else "")
        )

    t0 = step("writing assembly.step")
    write_step_assembly(robot, result.assembly_path)
    timings["step"] = time.perf_counter() - t0

    if write_parts:
        t0 = step("writing per-part files")
        write_part_files(result, out_dir / "parts")
        timings["parts"] = time.perf_counter() - t0

    t0 = step("writing joints.json")
    doc = joints_document(lm, scale=scale, units=units)
    result.joints_path = out_dir / "joints.json"
    result.joints_path.write_text(json.dumps(doc, indent=2))
    timings["joints"] = time.perf_counter() - t0

    if write_preview:
        t0 = step("writing preview.glb")
        try:
            result.preview_path = out_dir / "preview.glb"
            write_preview_glb(result, result.preview_path)
        except Exception as exc:  # pragma: no cover - preview is best effort
            result.warnings.append(f"preview.glb failed: {type(exc).__name__}: {exc}")
            result.preview_path = None
        timings["preview"] = time.perf_counter() - t0

    if printable:
        t0 = step("writing printable STLs")
        from .printable import write_printable

        result.printable = write_printable(result)
        timings["printable"] = time.perf_counter() - t0
        open_after = [p for p in result.printable if not p.watertight]
        if open_after:
            result.warnings.append(
                f"{len(open_after)} printable part(s) could not be closed: "
                + ", ".join(p.body for p in open_after[:5])
            )
        lossy = [p for p in result.printable if p.lossy]
        if lossy:
            result.warnings.append(
                f"{len(lossy)} printable part(s) were rebuilt rather than patched, so "
                "their shape changed: " + ", ".join(p.body for p in lossy[:5])
            )

    if write_render and result.preview_path is not None:
        t0 = step("rendering comparison images")
        try:
            from .render import render_comparison

            images = render_comparison(result)
            if images is None:
                result.render_skipped_because = "no offscreen GL context available"
            else:
                result.render_paths = images
        except Exception as exc:  # pragma: no cover - rendering is best effort
            result.render_skipped_because = f"{type(exc).__name__}: {exc}"
        timings["render"] = time.perf_counter() - t0
    elif write_render:
        result.render_skipped_because = "no preview.glb to compare against"

    return result
