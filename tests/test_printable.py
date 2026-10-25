"""``--printable``: STLs a slicer will actually accept.

The STEP assembly is allowed to contain open shells; a slicer is not. These
tests check that the repair ladder closes what it can, says so when it cannot,
and is honest about the repairs that change the shape.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from sim2cad.export import run_export
from sim2cad.printable import body_mesh, make_printable, write_printable
from sim2cad.validate import validate_export, write_reports


def test_every_primitive_link_is_already_closed(tmp_path):
    result = run_export(
        mjcf="tests/data/humanoid.xml",
        out=str(tmp_path / "out"),
        printable=True,
        write_render=False,
    )
    assert len(result.printable) == 16
    assert all(p.watertight for p in result.printable)
    assert not any(p.lossy for p in result.printable)

    for part in result.printable:
        assert part.path.exists()
        reloaded = trimesh.load(part.path)
        assert reloaded.is_watertight, part.body
        # STL stores vertices as float32, so a round trip is only good to
        # single precision.
        assert np.isclose(reloaded.volume, part.volume_mm3, rtol=1e-6)


def test_overlapping_pieces_are_fused(tmp_path):
    """The humanoid torso is two overlapping capsules; it should print as one."""
    result = run_export(
        mjcf="tests/data/humanoid.xml",
        out=str(tmp_path / "out"),
        printable=True,
        write_render=False,
    )
    torso = next(p for p in result.printable if p.body == "torso")
    assert torso.pieces_in == 2
    assert torso.pieces_out == 1
    assert "union" in torso.method

    # Fusing must not double-count the overlap: the union is smaller than the
    # two capsules added together.
    node = next(n for n in result.robot.bodies_with_geometry() if n.name == "torso")
    separate = sum(
        float(p.volume) for p in body_mesh(node, result.scale).split(only_watertight=True)
    )
    assert torso.volume_mm3 < separate


def test_open_mesh_is_repaired_to_watertight(mesh_model, tmp_path):
    """The fixture's open box has no top face; the STL must still be closed."""
    result = run_export(
        mjcf=str(mesh_model),
        out=str(tmp_path / "out"),
        printable=True,
        write_render=False,
    )
    assert result.printable
    for part in result.printable:
        assert part.watertight, f"{part.body} came out open ({part.method})"
        reloaded = trimesh.load(part.path)
        assert reloaded.is_watertight

    tip = next(p for p in result.printable if p.body == "tip")
    assert np.isclose(tip.volume_mm3, 60**3, rtol=1e-3)


def test_vertex_joined_pieces_stay_separate_when_they_do_not_touch(mesh_model, tmp_path):
    """Two boxes meeting at a corner do not overlap, so they stay two objects.

    A slicer will lay them out as two parts on the plate, which is what they
    are -- fusing them would invent material that is not in the model.
    """
    result = run_export(
        mjcf=str(mesh_model),
        out=str(tmp_path / "out"),
        printable=True,
        write_render=False,
    )
    corner = next(p for p in result.printable if p.body == "corner")
    assert corner.pieces_in == 2
    assert corner.watertight
    assert np.isclose(corner.volume_mm3, 2 * 60**3, rtol=1e-3)


def test_repair_ladder_reports_what_it_did():
    """An unpatchable surface must be reported, not silently emitted."""
    # A single triangle cannot be closed into a solid by any gentle means.
    sheet = trimesh.Trimesh(
        vertices=np.array([[0.0, 0, 0], [10, 0, 0], [0, 10, 0]]),
        faces=np.array([[0, 1, 2]]),
        process=False,
    )
    repaired, method, lossy, _shift = make_printable(sheet)
    assert not repaired.is_watertight
    assert isinstance(method, str) and method


def test_watertight_input_is_left_alone():
    box = trimesh.creation.box(extents=(10.0, 20.0, 30.0))
    repaired, method, lossy, shift = make_printable(box)
    assert method == "already watertight"
    assert not lossy
    assert shift == 0.0
    assert np.isclose(repaired.volume, 6000.0)


def test_printable_is_off_by_default(tmp_path):
    result = run_export(
        mjcf="tests/data/humanoid.xml", out=str(tmp_path / "out"), write_render=False
    )
    assert result.printable == []
    assert not (tmp_path / "out" / "printable").exists()


def test_stls_are_millimetres_whatever_the_units(tmp_path):
    """Slicers assume mm, so printable output ignores --units."""
    metres = run_export(
        mjcf="tests/data/humanoid.xml",
        out=str(tmp_path / "m"),
        units="m",
        printable=True,
        write_render=False,
    )
    millimetres = run_export(
        mjcf="tests/data/humanoid.xml",
        out=str(tmp_path / "mm"),
        units="mm",
        printable=True,
        write_render=False,
    )
    a = next(p for p in metres.printable if p.body == "head")
    b = next(p for p in millimetres.printable if p.body == "head")
    assert a.faces == b.faces, "tessellation must not depend on the export unit"
    assert np.isclose(a.volume_mm3, b.volume_mm3, rtol=1e-6)
    # A humanoid head is ~180 mm across, so ~3e6 mm^3 -- not 3e-3 m^3.
    assert a.volume_mm3 > 1e5


def test_printable_reaches_the_report(tmp_path):
    """Build and write the report too, not just the STLs.

    The first version of this module broke report construction outright and
    every test here still passed, because none of them went near
    validate_export.
    """
    import json

    out = tmp_path / "out"
    result = run_export(
        mjcf="tests/data/humanoid.xml",
        out=str(out),
        printable=True,
        write_render=False,
    )
    report = validate_export(result)
    assert report.passed
    assert len(report.printable) == len(result.printable)

    md, js = write_reports(report, out)
    text = md.read_text()
    assert "## Printable STLs" in text
    assert "sliceable" in text
    assert "not a working robot" in text

    payload = json.loads(js.read_text())
    assert payload["options"]["printable"] is True
    assert all(row["watertight"] for row in payload["printable"])
