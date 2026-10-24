"""Shared fixtures.

The G1 export is expensive (about 400k triangles through OpenCascade), so it is
built once per session and reused.  ``humanoid.xml`` is cheap and vendored in
``tests/data`` so the primitive tests run with no network at all.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"
HUMANOID = DATA / "humanoid.xml"


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run the full G1 mesh export tests (several minutes)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the full G1 export")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow") or os.environ.get("SIM2CAD_RUN_SLOW"):
        return
    skip = pytest.mark.skip(reason="needs --run-slow (G1 export takes minutes)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def humanoid_path() -> Path:
    return HUMANOID


@pytest.fixture(scope="session")
def g1_model():
    """The compiled G1 model, or a skip if Menagerie cannot be reached."""
    from sim2cad.load import load_model

    try:
        return load_model(menagerie="unitree_g1")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"unitree_g1 unavailable: {exc}")


@pytest.fixture(scope="session")
def humanoid_export(tmp_path_factory):
    from sim2cad.export import run_export

    out = tmp_path_factory.mktemp("humanoid_export")
    return run_export(mjcf=str(HUMANOID), out=str(out))


@pytest.fixture(scope="session")
def g1_export(tmp_path_factory):
    from sim2cad.export import run_export

    out = tmp_path_factory.mktemp("g1_export")
    try:
        return run_export(menagerie="unitree_g1", out=str(out))
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"unitree_g1 export unavailable: {exc}")
