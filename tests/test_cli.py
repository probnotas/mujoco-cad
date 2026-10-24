"""The command line surface."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sim2cad.cli import app

runner = CliRunner()


def test_info_lists_bodies_joints_and_keyframes():
    result = runner.invoke(app, ["info", "--mjcf", "tests/data/humanoid.xml"])
    assert result.exit_code == 0, result.output
    assert "bodies=17" in result.output
    assert "squat" in result.output
    assert "hingex21" in result.output


def test_list_keyframes():
    result = runner.invoke(
        app, ["info", "--mjcf", "tests/data/humanoid.xml", "--list-keyframes"]
    )
    assert result.exit_code == 0
    assert result.output.split() == ["squat", "stand_on_left_leg", "prone", "supine"]


def test_source_options_are_exclusive():
    assert runner.invoke(app, ["info"]).exit_code != 0
    assert (
        runner.invoke(
            app, ["info", "--mjcf", "tests/data/humanoid.xml", "--menagerie", "unitree_g1"]
        ).exit_code
        != 0
    )


def test_export_writes_everything_and_passes(tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "export",
            "--mjcf",
            "tests/data/humanoid.xml",
            "--out",
            str(out),
            "--no-render",
        ],
    )
    assert result.exit_code == 0, result.output
    for name in ("assembly.step", "joints.json", "report.md", "report.json", "preview.glb"):
        assert (out / name).exists(), name
    assert len(list((out / "parts").glob("*.step"))) == 16
    assert len(list((out / "parts").glob("*.stl"))) == 16

    report = json.loads((out / "report.json").read_text())
    assert report["passed"]
    assert report["options"]["pose"] == "qpos0"

    text = (out / "report.md").read_text()
    assert "ALL CHECKS PASSED" in text
    assert "| body | part | type |" in text


def test_export_honours_flags(tmp_path):
    out = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "export",
            "--mjcf",
            "tests/data/humanoid.xml",
            "--out",
            str(out),
            "--pose",
            "keyframe:prone",
            "--units",
            "m",
            "--no-joint-markers",
            "--no-preview",
            "--no-render",
            "--only",
            "head,torso",
        ],
    )
    assert result.exit_code == 0, result.output
    assert not (out / "preview.glb").exists()
    assert sorted(p.stem for p in (out / "parts").glob("*.step")) == ["head", "torso"]

    doc = json.loads((out / "joints.json").read_text())
    assert doc["pose"] == "keyframe_prone"
    assert doc["units"]["length"] == "m"


def test_validate_reruns_the_checks(tmp_path):
    out = tmp_path / "out"
    assert (
        runner.invoke(
            app, ["export", "--mjcf", "tests/data/humanoid.xml", "--out", str(out), "--no-render"]
        ).exit_code
        == 0
    )
    result = runner.invoke(app, ["validate", str(out)])
    assert result.exit_code == 0, result.output
    assert "all checks passed" in result.output


def test_validate_needs_an_export(tmp_path):
    result = runner.invoke(app, ["validate", str(tmp_path)])
    assert result.exit_code != 0
    assert "report.json" in result.output


def test_unknown_body_is_reported(tmp_path):
    result = runner.invoke(
        app,
        ["export", "--mjcf", "tests/data/humanoid.xml", "--out", str(tmp_path / "o"),
         "--only", "no_such_link", "--no-render"],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "no_such_link" in str(result.exception)


def test_unknown_keyframe_lists_the_real_ones(tmp_path):
    result = runner.invoke(
        app,
        ["export", "--mjcf", "tests/data/humanoid.xml", "--out", str(tmp_path / "o"),
         "--pose", "keyframe:nope", "--no-render"],
    )
    assert result.exit_code != 0
    assert "squat" in str(result.exception)
