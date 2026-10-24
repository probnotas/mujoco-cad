"""The ``sim2cad`` command line."""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

app = typer.Typer(
    add_completion=False,
    help="Export MuJoCo robot models (MJCF) to STEP CAD assemblies.",
    no_args_is_help=True,
)
console = Console()


def _source_options(menagerie: str | None, mjcf: str | None) -> None:
    if (menagerie is None) == (mjcf is None):
        raise typer.BadParameter("pass exactly one of --menagerie or --mjcf")


@app.command()
def export(
    menagerie: Optional[str] = typer.Option(None, help="Menagerie robot name, e.g. unitree_g1."),
    mjcf: Optional[str] = typer.Option(None, help="Path to an MJCF file."),
    file: Optional[str] = typer.Option(
        None, "--file", help="Entry point inside a Menagerie model, e.g. g1 or g1_with_hands."
    ),
    geoms: str = typer.Option("visual", help="visual | collision | all."),
    pose: str = typer.Option("qpos0", help="qpos0 | keyframe:NAME | json:FILE."),
    units: str = typer.Option("mm", help="mm | m."),
    decimate: Optional[float] = typer.Option(
        None, min=0.0, max=1.0, help="Keep this fraction of mesh triangles (default: off)."
    ),
    merge_coplanar: bool = typer.Option(True, "--merge-coplanar/--no-merge-coplanar"),
    joint_markers: bool = typer.Option(True, "--joint-markers/--no-joint-markers"),
    ground: bool = typer.Option(False, "--ground", help="Drop the robot so its lowest point is z=0."),
    keep_root_pose: bool = typer.Option(
        False, "--keep-root-pose", help="Keep the sim's root pose instead of moving it to the origin."
    ),
    only: Optional[str] = typer.Option(None, help="Comma-separated body names to export."),
    out: Optional[str] = typer.Option(None, help="Output directory."),
    parts: bool = typer.Option(True, "--parts/--no-parts", help="Write per-link STEP and STL."),
    preview: bool = typer.Option(True, "--preview/--no-preview", help="Write preview.glb."),
    validate_after: bool = typer.Option(
        True, "--validate/--no-validate", help="Run the §8 checks and write the report."
    ),
) -> None:
    """Export a robot as a nested STEP assembly plus per-link files."""
    _source_options(menagerie, mjcf)
    from .export import run_export
    from .validate import validate_export, write_reports

    only_list = [s.strip() for s in only.split(",") if s.strip()] if only else None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("starting", total=None)

        def report(message: str) -> None:
            progress.update(task, description=message)

        result = run_export(
            menagerie=menagerie,
            mjcf=mjcf,
            entry=file,
            geoms=geoms,
            pose=pose,
            units=units,
            decimate=decimate,
            merge_coplanar=merge_coplanar,
            joint_markers=joint_markers,
            ground=ground,
            keep_root_pose=keep_root_pose,
            only=only_list,
            out=out,
            write_parts=parts,
            write_preview=preview,
            progress=report,
        )

        report_obj = None
        if validate_after:
            report("validating")
            report_obj = validate_export(result)
            write_reports(report_obj, result.out_dir)

    console.print(f"[bold green]wrote[/] {result.out_dir}")
    console.print(
        f"  assembly.step  {result.assembly_path.stat().st_size:,} bytes, "
        f"{len(result.robot.bodies_with_geometry())} links, {len(result.robot.parts)} parts"
    )
    if result.part_paths:
        console.print(f"  parts/         {len(result.part_paths)} STEP + {len(result.stl_paths)} STL")
    console.print(f"  joints.json    {len(result.robot.nodes)} bodies")
    for warning in result.warnings[:6]:
        console.print(f"  [yellow]note[/] {warning}")

    if report_obj is not None:
        _print_checks(report_obj)
        if not report_obj.passed:
            raise typer.Exit(code=1)


@app.command()
def info(
    menagerie: Optional[str] = typer.Option(None),
    mjcf: Optional[str] = typer.Option(None),
    file: Optional[str] = typer.Option(None, "--file"),
    list_keyframes: bool = typer.Option(False, "--list-keyframes", help="Only list keyframes."),
) -> None:
    """Print bodies, joints, geoms and keyframes for a model."""
    _source_options(menagerie, mjcf)
    import mujoco

    from .geoms import classify_geom, geom_type_name
    from .load import body_names, joint_names, keyframe_names, load_model
    from .joints import JOINT_TYPE_NAMES

    lm = load_model(menagerie=menagerie, mjcf=mjcf, entry=file)
    model = lm.model

    keys = keyframe_names(model)
    if list_keyframes:
        console.print("\n".join(keys) if keys else "(no keyframes)")
        return

    console.print(f"[bold]{lm.name}[/]  {lm.xml_path}")
    console.print(
        f"bodies={model.nbody}  joints={model.njnt}  geoms={model.ngeom}  "
        f"meshes={model.nmesh}  nq={model.nq}  nv={model.nv}  actuators={model.nu}"
    )
    console.print(f"keyframes: {', '.join(keys) if keys else '(none)'}")

    jt = collections.Counter(
        JOINT_TYPE_NAMES.get(int(t), str(t)) for t in model.jnt_type
    )
    console.print("joint types: " + ", ".join(f"{k}x{v}" for k, v in sorted(jt.items())))

    table = Table("geom type", "visual", "collision", title="geoms by type and class")
    by_type: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    groups: collections.Counter = collections.Counter()
    for g in range(model.ngeom):
        name = geom_type_name(model.geom_type[g])
        idx = 0 if classify_geom(model, g).value == "visual" else 1
        by_type[name][idx] += 1
        groups[(name, int(model.geom_group[g]), classify_geom(model, g).value)] += 1
    for name, (vis, col) in sorted(by_type.items()):
        table.add_row(name, str(vis), str(col))
    console.print(table)

    gtable = Table("geom group", "class", "count", title="geom groups (detected, not assumed)")
    for (name, group, cls), count in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        gtable.add_row(str(group), cls, f"{count}  ({name})")
    console.print(gtable)

    btable = Table("body", "parent", "geoms", "joints", title="bodies")
    bnames, jnames = body_names(model), joint_names(model)
    for b in lm.subtree():
        gcount = collections.Counter(
            geom_type_name(model.geom_type[g])
            for g in range(int(model.body_geomadr[b]), int(model.body_geomadr[b]) + int(model.body_geomnum[b]))
        )
        joints = [
            jnames[j] for j in range(model.njnt) if int(model.jnt_bodyid[j]) == b
        ]
        btable.add_row(
            bnames[b],
            bnames[int(model.body_parentid[b])],
            ", ".join(f"{k}x{v}" for k, v in sorted(gcount.items())) or "-",
            ", ".join(joints) or "-",
        )
    console.print(btable)


@app.command()
def validate(
    directory: str = typer.Argument(..., help="An existing export directory."),
) -> None:
    """Re-run the §8 checks against an existing export."""
    from .export import run_export
    from .validate import validate_export, write_reports

    path = Path(directory)
    report_path = path / "report.json"
    if not report_path.is_file():
        raise typer.BadParameter(f"{report_path} not found; re-run `sim2cad export` first")
    previous = json.loads(report_path.read_text())

    console.print(f"re-exporting [bold]{previous['robot']}[/] from {previous['source']} to compare")
    result = run_export(
        mjcf=previous["source"],
        geoms=previous["geoms"],
        pose=_pose_spec(previous["pose"]),
        units=previous["units"],
        out=str(path),
    )
    report_obj = validate_export(result)
    write_reports(report_obj, path)
    _print_checks(report_obj)
    if not report_obj.passed:
        raise typer.Exit(code=1)


def _pose_spec(label: str) -> str:
    if label.startswith("keyframe_"):
        return "keyframe:" + label[len("keyframe_") :]
    return "qpos0"


def _print_checks(report) -> None:
    table = Table("check", "status", "detail", title="PRD §8 checks")
    for c in report.checks:
        colour = {"PASS": "green", "FAIL": "red", "SKIP": "yellow"}[c.status]
        table.add_row(c.name, f"[{colour}]{c.status}[/]", c.detail)
    console.print(table)
    console.print(
        "[bold green]all checks passed[/]" if report.passed else "[bold red]checks failed[/]"
    )


if __name__ == "__main__":  # pragma: no cover
    app()
