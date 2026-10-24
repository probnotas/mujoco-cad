"""The §8 acceptance checks, and the report they feed.

What "placement is correct" means here
--------------------------------------
Two quantities are compared per part, both in the exported assembly's world
frame, and both derived independently of the CAD we are checking:

* **bounding box** -- for a mesh, the box of MuJoCo's own compiled vertices put
  through ``geom_xpos``/``geom_xmat``; for a primitive, the exact analytic box
  of that shape under the geom's world rotation.  Mesh cleanup never moves a
  vertex and the coplanar merge never drops an extreme one, so this box is
  exact, not approximate.
* **surface centroid** -- the area-weighted centroid, which is a property of
  the surface rather than of how it happens to be triangulated, so it survives
  the coplanar merge and the STEP round trip unchanged.  Every MuJoCo primitive
  is symmetric about its own origin, so for those the expected centroid is just
  the geom's world position.

The measured values come from re-reading ``assembly.step``, so the whole
pipeline -- solid construction, frame composition, nesting, STEP write and STEP
read -- is in the loop.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .assembly import JOINTS_NODE, PartGeom, RobotAssembly
from .export import ExportResult
from .frames import Pose
from .geoms import analytic_volume
from .stepread import (
    StepNode,
    count_faces,
    read_step_assembly,
    shape_bbox,
    solid_volume,
    surface_centroid,
)

__all__ = ["Check", "PartReport", "ValidationReport", "validate_export", "write_reports"]

#: §8.3 tolerances, in millimetres.
TOL_CENTROID_MESH_MM = 0.01
TOL_BBOX_MESH_MM = 0.05
TOL_PRIMITIVE_MM = 1e-6
#: §8.4 tolerances, relative.
TOL_VOLUME_MESH = 0.005
TOL_VOLUME_PRIMITIVE = 1e-6


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    value: float | None = None
    tolerance: float | None = None
    skipped: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "SKIP"
        return "PASS" if self.passed else "FAIL"


@dataclass
class PartReport:
    body: str
    part: str
    geom_type: str
    geom_class: str
    faces_before: int | None
    faces_after: int | None
    n_components: int
    watertight: bool | None
    is_solid: bool
    valid: bool
    volume: float
    volume_reference: float | None
    volume_rel_error: float | None
    volume_merge_rel_error: float | None
    centroid_error: float
    bbox_error: float
    notes: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    robot: str
    source: str
    pose: str
    units: str
    geoms_mode: str
    checks: list[Check]
    parts: list[PartReport]
    body_table: list[dict[str, Any]]
    name_map: dict[str, str]
    files: dict[str, Any]
    warnings: list[str]
    timings: dict[str, float]
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed or c.skipped for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "source": self.source,
            "pose": self.pose,
            "units": self.units,
            "geoms": self.geoms_mode,
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "parts": [asdict(p) for p in self.parts],
            "bodies": self.body_table,
            "name_map": self.name_map,
            "files": self.files,
            "warnings": self.warnings,
            "timings": self.timings,
            "options": self.options,
        }


# -- expected world geometry -----------------------------------------------


def _scaled(pose: Pose, scale: float) -> Pose:
    return pose.scaled(scale)


def expected_primitive_bbox(
    model: mujoco.MjModel, geom_id: int, world: Pose, scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exact world axis-aligned bounds of a primitive geom under rotation."""
    gtype = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float) * scale
    rot = np.abs(world.rot)
    centre = world.pos * scale

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        half = np.full(3, size[0])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        half = rot @ size[:3]
    elif gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        half = np.sqrt((world.rot**2) @ (size[:3] ** 2))
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        radius, halflen = size[0], size[1]
        perp = np.sqrt(np.clip(1.0 - world.rot[:, 2] ** 2, 0.0, None))
        half = halflen * rot[:, 2] + radius * perp
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        radius, halflen = size[0], size[1]
        half = halflen * rot[:, 2] + radius
    else:
        raise ValueError(f"geom {geom_id} is not a primitive")

    return centre - half, centre + half


def _mesh_world_points(part: PartGeom, world: Pose, model, scale: float) -> np.ndarray:
    """MuJoCo's own compiled vertices for this geom, in corrected world coords."""
    mesh_id = int(model.geom_dataid[part.geom_id])
    adr = int(model.mesh_vertadr[mesh_id])
    num = int(model.mesh_vertnum[mesh_id])
    verts = np.asarray(model.mesh_vert[adr : adr + num], dtype=float).reshape(-1, 3) * scale
    return _scaled(world, scale).apply(verts)


def _mesh_area_centroid(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    tri = points[faces]
    centres = tri.mean(axis=1)
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
    )
    total = areas.sum()
    if total <= 0:
        return centres.mean(axis=0)
    return (centres * areas[:, None]).sum(axis=0) / total


# -- the checks ------------------------------------------------------------


def _index_step_leaves(root: StepNode) -> dict[str, tuple[StepNode, Pose]]:
    out: dict[str, tuple[StepNode, Pose]] = {}
    for node, world in root.leaves():
        out[node.name] = (node, world)
    return out


def validate_export(result: ExportResult, run_round_trip: bool = True) -> ValidationReport:
    """Run every check that does not need a second export, and build the report."""
    lm, robot = result.lm, result.robot
    model = lm.model
    scale = result.scale
    checks: list[Check] = []
    parts: list[PartReport] = []

    # Measure in millimetres whatever the export unit is.  OpenCascade is a
    # millimetre-native kernel: BRepBndLib::AddOptimal converges to ~1e-11 mm
    # on a capsule at millimetre scale but only ~2.6e-5 m on the same capsule
    # written in metres, because its tolerance is absolute.  Reading the file
    # back as millimetres sidesteps that, and the PRD quotes its tolerances in
    # millimetres anyway.
    step_root = read_step_assembly(str(result.assembly_path), unit="MM")
    #: metres -> millimetres, for the expected values.
    mm_scale = 1000.0
    #: file units -> millimetres, for anything already scaled at export time.
    to_mm = mm_scale / scale
    leaves = _index_step_leaves(step_root)

    worst_centroid_mesh = 0.0
    worst_bbox_mesh = 0.0
    worst_centroid_prim = 0.0
    worst_bbox_prim = 0.0
    worst_vol_mesh = 0.0
    worst_vol_prim = 0.0
    worst_merge = 0.0
    invalid: list[str] = []
    open_shells: list[str] = []
    missing: list[str] = []

    for node in robot.bodies_with_geometry():
        for part in node.parts:
            entry = leaves.get(part.safe_name)
            if entry is None:
                missing.append(part.safe_name)
                continue
            leaf, world_loc = entry
            placed = leaf.shape.Moved(world_loc.to_location().wrapped)

            expected_world = _scaled(lm.geom_pose(part.geom_id), mm_scale)
            is_prim = part.mesh is None

            if is_prim:
                lo_ref, hi_ref = expected_primitive_bbox(
                    model, part.geom_id, lm.geom_pose(part.geom_id), mm_scale
                )
                centroid_ref = expected_world.pos
                vol_ref = analytic_volume(model, part.geom_id, mm_scale)
            else:
                pts = _mesh_world_points(part, lm.geom_pose(part.geom_id), model, mm_scale)
                lo_ref, hi_ref = pts.min(axis=0), pts.max(axis=0)
                cleaned = expected_world.apply(part.mesh.vertices * to_mm)
                centroid_ref = _mesh_area_centroid(cleaned, part.mesh.faces)
                vol_ref = (
                    part.mesh.volume * to_mm**3 if part.mesh.watertight else None
                )

            lo, hi = shape_bbox(placed, optimal=True)
            bbox_err = float(max(np.abs(lo - lo_ref).max(), np.abs(hi - hi_ref).max()))
            centroid, _area = surface_centroid(placed)
            centroid_err = float(np.linalg.norm(centroid - centroid_ref))

            volume = solid_volume(placed)  # mm^3
            vol_rel = (
                abs(volume - vol_ref) / abs(vol_ref)
                if vol_ref not in (None, 0) and math.isfinite(vol_ref)
                else None
            )

            build = part.mesh_build
            is_solid = build.is_solid if build is not None else True
            valid = build.valid if build is not None else True
            if build is not None:
                merge_rel = _relative(build.volume_before_merge, build.volume_after_merge)
                faces_before, faces_after = build.faces_before, build.faces_after
                components = build.n_components
                notes = list(build.notes)
            else:
                merge_rel = None
                faces_before = faces_after = count_faces(part.shape)
                components = 1
                notes = []

            # `valid` already came from mesh_to_shape; BRepCheck on a 50k-face
            # shape costs ten seconds, so it is not worth running twice.
            if not valid:
                invalid.append(part.safe_name)
            if not is_solid:
                open_shells.append(part.safe_name)

            if is_prim:
                worst_centroid_prim = max(worst_centroid_prim, centroid_err)
                worst_bbox_prim = max(worst_bbox_prim, bbox_err)
                if vol_rel is not None:
                    worst_vol_prim = max(worst_vol_prim, vol_rel)
            else:
                worst_centroid_mesh = max(worst_centroid_mesh, centroid_err)
                worst_bbox_mesh = max(worst_bbox_mesh, bbox_err)
                if vol_rel is not None:
                    worst_vol_mesh = max(worst_vol_mesh, vol_rel)
            if merge_rel is not None and math.isfinite(merge_rel):
                worst_merge = max(worst_merge, merge_rel)

            parts.append(
                PartReport(
                    body=node.name,
                    part=part.safe_name,
                    geom_type=part.type_name,
                    geom_class=part.geom_class,
                    faces_before=faces_before,
                    faces_after=faces_after,
                    n_components=components,
                    watertight=part.mesh.watertight if part.mesh else None,
                    is_solid=is_solid,
                    valid=valid,
                    volume=volume / to_mm**3,
                    volume_reference=vol_ref / to_mm**3 if vol_ref is not None else None,
                    volume_rel_error=vol_rel,
                    volume_merge_rel_error=merge_rel,
                    centroid_error=centroid_err,
                    bbox_error=bbox_err,
                    notes=notes,
                )
            )

    # The tolerances are quoted in mm; in metre mode the same physical
    # tolerance is 1000x smaller in file units.
    n_mesh = sum(1 for p in parts if p.geom_type == "mesh")
    n_prim = len(parts) - n_mesh
    n_mesh_solid = sum(1 for p in parts if p.geom_type == "mesh" and p.is_solid)

    checks.append(
        Check(
            "3a. placement: mesh part centroid",
            worst_centroid_mesh <= TOL_CENTROID_MESH_MM,
            f"worst centroid error {worst_centroid_mesh:.3e} mm",
            worst_centroid_mesh,
            TOL_CENTROID_MESH_MM,
            skipped=n_mesh == 0,
        )
    )
    checks.append(
        Check(
            "3b. placement: mesh part bounding box",
            worst_bbox_mesh <= TOL_BBOX_MESH_MM,
            f"worst bbox error {worst_bbox_mesh:.3e} mm",
            worst_bbox_mesh,
            TOL_BBOX_MESH_MM,
            skipped=n_mesh == 0,
        )
    )
    checks.append(
        Check(
            "3c. placement: primitive part (machine precision)",
            max(worst_centroid_prim, worst_bbox_prim) <= TOL_PRIMITIVE_MM,
            f"worst centroid {worst_centroid_prim:.3e}, bbox {worst_bbox_prim:.3e} mm",
            max(worst_centroid_prim, worst_bbox_prim),
            TOL_PRIMITIVE_MM,
            skipped=n_prim == 0,
        )
    )
    checks.append(
        Check(
            "4a. volume: mesh solids vs trimesh",
            worst_vol_mesh <= TOL_VOLUME_MESH,
            f"worst relative error {worst_vol_mesh:.3e}",
            worst_vol_mesh,
            TOL_VOLUME_MESH,
            skipped=n_mesh_solid == 0,
        )
    )
    checks.append(
        Check(
            "4b. volume: primitives vs analytic",
            worst_vol_prim <= TOL_VOLUME_PRIMITIVE,
            f"worst relative error {worst_vol_prim:.3e}",
            worst_vol_prim,
            TOL_VOLUME_PRIMITIVE,
            skipped=n_prim == 0,
        )
    )
    checks.append(
        Check(
            "4c. volume: unchanged by --merge-coplanar",
            worst_merge <= TOL_VOLUME_MESH,
            f"worst relative change {worst_merge:.3e}",
            worst_merge,
            TOL_VOLUME_MESH,
            skipped=n_mesh_solid == 0,
        )
    )

    checks.extend(_completeness_checks(result))

    if result.render_paths:
        checks.append(
            Check(
                "9. visual: sim vs exported CAD, same camera",
                True,
                "wrote " + ", ".join(sorted(p.name for p in result.render_paths.values())),
                float(len(result.render_paths)),
            )
        )
    else:
        checks.append(
            Check(
                "9. visual: sim vs exported CAD, same camera",
                True,
                result.render_skipped_because or "not requested",
                skipped=True,
            )
        )

    checks.append(
        Check(
            "6a. validity: BRepCheck_Analyzer on every part",
            not invalid,
            "all parts valid" if not invalid else f"{len(invalid)} invalid: {invalid[:5]}",
            float(len(invalid)),
            0.0,
        )
    )
    checks.append(
        Check(
            "6b. validity: open shells (reported, not fatal)",
            True,
            (
                f"{len(open_shells)} of {len(parts)} parts are open shells from "
                f"non-watertight meshes: {sorted(open_shells)[:6]}"
                + (" ..." if len(open_shells) > 6 else "")
            )
            if open_shells
            else "every part is a closed solid",
            float(len(open_shells)),
        )
    )
    if missing:
        checks.append(
            Check(
                "10a. round trip: every part found in assembly.step",
                False,
                f"{len(missing)} parts missing after re-import: {missing[:5]}",
            )
        )

    if run_round_trip:
        checks.extend(_round_trip_checks(result, step_root))

    body_table = _body_table(result)
    files = _file_table(result)

    return ValidationReport(
        robot=lm.name,
        source=str(lm.xml_path),
        pose=lm.pose_label,
        units=result.units,
        geoms_mode=result.selection.mode,
        checks=checks,
        parts=parts,
        body_table=body_table,
        name_map=dict(robot.name_map.mapping),
        files=files,
        warnings=list(result.warnings),
        timings=dict(result.timings),
        options=dict(result.options),
    )


def _relative(a: float, b: float) -> float | None:
    if not (math.isfinite(a) and math.isfinite(b)) or a == 0:
        return None
    return abs(b - a) / abs(a)


def _completeness_checks(result: ExportResult) -> list[Check]:
    lm, robot = result.lm, result.robot
    model = lm.model
    checks: list[Check] = []

    expected_bodies = {
        b for b, geoms in result.selection.by_body.items() if geoms and b in robot.nodes
    }
    got = {n.body_id for n in robot.bodies_with_geometry()}
    checks.append(
        Check(
            "5a. completeness: one part node per body with selected geoms",
            expected_bodies == got,
            f"{len(got)} of {len(expected_bodies)} bodies exported",
            float(len(got)),
            float(len(expected_bodies)),
        )
    )

    # Which bodies *should* be nodes: the whole subtree normally, or the
    # chosen links plus their ancestors when --only pruned it.  Either way a
    # body without geoms must still be present, so the frames survive.
    from .assembly import _keep_bodies

    wanted = _keep_bodies(lm, result.options.get("only"))
    if wanted is None:
        expected_nodes = set(lm.subtree())
    else:
        expected_nodes = set()
        for body in wanted:
            cur = body
            while cur != 0:
                expected_nodes.add(cur)
                cur = int(model.body_parentid[cur])

    empty = [n.name for n in robot.nodes.values() if not n.has_geometry]
    missing_nodes = expected_nodes - set(robot.nodes)
    checks.append(
        Check(
            "5b. completeness: geometry-less bodies still in the hierarchy",
            not missing_nodes,
            f"{len(robot.nodes)} of {len(expected_nodes)} body nodes, "
            f"{len(empty)} of them empty"
            + (f" ({empty[:5]})" if empty else "")
            + (f"; missing {sorted(missing_nodes)[:5]}" if missing_nodes else ""),
            float(len(robot.nodes)),
            float(len(expected_nodes)),
        )
    )

    doc = json.loads(Path(result.joints_path).read_text()) if result.joints_path else {}
    joint_names_out = {j["name"] for j in doc.get("joints", [])}
    from .load import joint_names

    subtree = set(lm.subtree())
    expected_joints = {
        joint_names(model)[j]
        for j in range(model.njnt)
        if int(model.jnt_bodyid[j]) in subtree
    }
    checks.append(
        Check(
            "5c. completeness: every joint appears in joints.json",
            joint_names_out == expected_joints,
            f"{len(joint_names_out)} of {len(expected_joints)} joints written"
            + (
                f"; missing {sorted(expected_joints - joint_names_out)[:5]}"
                if expected_joints - joint_names_out
                else ""
            ),
            float(len(joint_names_out)),
            float(len(expected_joints)),
        )
    )

    free_dofs = sum(
        6
        for j in range(model.njnt)
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    articulated = sum(
        j["ndof"] for j in doc.get("joints", []) if j["type"] != "free"
    )
    checks.append(
        Check(
            "5d. completeness: articulated DOF == model.nv minus the free joint",
            articulated == int(model.nv) - free_dofs,
            f"{articulated} articulated DOF, nv={int(model.nv)}, free joint DOF={free_dofs}",
            float(articulated),
            float(int(model.nv) - free_dofs),
        )
    )
    return checks


def _round_trip_checks(result: ExportResult, step_root: StepNode) -> list[Check]:
    robot = result.robot
    names_in_file = set(step_root.names())

    expected_nodes = {n.safe_name for n in robot.nodes.values()}
    expected_parts = {p.safe_name for p in robot.parts}
    if robot.joint_marker_names:
        expected_nodes.add(JOINTS_NODE)
        expected_parts |= set(robot.joint_marker_names)

    missing_nodes = sorted(expected_nodes - names_in_file)
    missing_parts = sorted(expected_parts - names_in_file)
    leaf_count = sum(1 for _ in step_root.leaves())

    return [
        Check(
            "10a. round trip: body node names survive STEP",
            not missing_nodes,
            f"{len(expected_nodes)} body nodes expected"
            + (f", missing {missing_nodes[:5]}" if missing_nodes else ", all present"),
            float(len(expected_nodes) - len(missing_nodes)),
            float(len(expected_nodes)),
        ),
        Check(
            "10b. round trip: part names and count survive STEP",
            not missing_parts and leaf_count == len(expected_parts),
            f"{leaf_count} leaf shapes in file, {len(expected_parts)} expected"
            + (f", missing {missing_parts[:5]}" if missing_parts else ""),
            float(leaf_count),
            float(len(expected_parts)),
        ),
    ]


def _body_table(result: ExportResult) -> list[dict[str, Any]]:
    rows = []
    for body_id in result.robot.order:
        node = result.robot.nodes[body_id]
        types: dict[str, int] = {}
        for part in node.parts:
            types[part.type_name] = types.get(part.type_name, 0) + 1
        rows.append(
            {
                "body": node.name,
                "safe_name": node.safe_name,
                "parent": result.robot.nodes[node.parent_id].name
                if node.parent_id in result.robot.nodes
                else "world",
                "geoms": types,
                "n_parts": len(node.parts),
            }
        )
    return rows


def _file_table(result: ExportResult) -> dict[str, Any]:
    def info(path: Path | None) -> dict[str, Any] | None:
        if path is None or not Path(path).exists():
            return None
        return {"path": str(path), "bytes": Path(path).stat().st_size}

    return {
        "assembly": info(result.assembly_path),
        "joints": info(result.joints_path),
        "preview": info(result.preview_path),
        "renders": {k: info(v) for k, v in sorted(result.render_paths.items())},
        "parts": {k: info(v) for k, v in sorted(result.part_paths.items())},
        "stl": {k: info(v) for k, v in sorted(result.stl_paths.items())},
    }


# -- report rendering ------------------------------------------------------


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e5:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def render_markdown(report: ValidationReport) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# sim2cad report: {report.robot}")
    a("")
    a(f"- **source**: `{report.source}`")
    a(f"- **pose**: `{report.pose}`")
    a(f"- **geoms**: `{report.geoms_mode}`")
    a(f"- **units**: `{report.units}` (volumes in {report.units}^3; placement errors in mm)")
    a(f"- **result**: {'**ALL CHECKS PASSED**' if report.passed else '**FAILURES**'}")
    a("")

    a("## Checks (PRD §8)")
    a("")
    a("| check | status | value | tolerance | detail |")
    a("|---|---|---|---|---|")
    for c in report.checks:
        a(
            f"| {c.name} | {c.status} | {_fmt(c.value)} | {_fmt(c.tolerance)} | {c.detail} |"
        )
    a("")

    a("## Parts")
    a("")
    a(
        "| body | part | type | class | pieces | faces before | faces after | watertight |"
        " solid | valid | volume | vol rel err | merge rel err | centroid err | bbox err |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in report.parts:
        a(
            f"| {p.body} | {p.part} | {p.geom_type} | {p.geom_class} | "
            f"{p.n_components} | "
            f"{p.faces_before if p.faces_before is not None else '-'} | "
            f"{p.faces_after if p.faces_after is not None else '-'} | "
            f"{'-' if p.watertight is None else ('yes' if p.watertight else 'no')} | "
            f"{'yes' if p.is_solid else 'no'} | {'yes' if p.valid else 'NO'} | "
            f"{_fmt(p.volume)} | {_fmt(p.volume_rel_error)} | "
            f"{_fmt(p.volume_merge_rel_error)} | {_fmt(p.centroid_error)} | "
            f"{_fmt(p.bbox_error)} |"
        )
    a("")

    a("## Bodies")
    a("")
    a("| body | parent | parts | geom types |")
    a("|---|---|---|---|")
    for row in report.body_table:
        types = ", ".join(f"{k}x{v}" for k, v in sorted(row["geoms"].items())) or "(none)"
        a(f"| {row['body']} | {row['parent']} | {row['n_parts']} | {types} |")
    a("")

    files = report.files
    a("## Files")
    a("")
    a("| file | bytes |")
    a("|---|---|")
    for key in ("assembly", "joints", "preview"):
        info = files.get(key)
        if info:
            a(f"| {Path(info['path']).name} | {info['bytes']:,} |")
    part_bytes = sum(i["bytes"] for i in files.get("parts", {}).values() if i)
    stl_bytes = sum(i["bytes"] for i in files.get("stl", {}).values() if i)
    a(f"| parts/*.step ({len(files.get('parts', {}))} files) | {part_bytes:,} |")
    a(f"| parts/*.stl ({len(files.get('stl', {}))} files) | {stl_bytes:,} |")
    a("")

    if report.warnings:
        a("## Warnings")
        a("")
        for w in report.warnings:
            a(f"- {w}")
        a("")

    a("## Timings (seconds)")
    a("")
    for k, v in report.timings.items():
        a(f"- {k}: {v:.2f}")
    a("")
    return "\n".join(lines)


def write_reports(report: ValidationReport, out_dir: Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    md = out_dir / "report.md"
    js = out_dir / "report.json"
    md.write_text(render_markdown(report))
    js.write_text(json.dumps(report.to_dict(), indent=2, default=str))
    return md, js
