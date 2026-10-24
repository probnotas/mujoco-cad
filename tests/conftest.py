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
def mesh_model(tmp_path_factory) -> Path:
    """A tiny mesh-based MJCF, built on the fly, with known exact volumes.

    The G1 is the only other mesh model we have and it takes 25 minutes, so
    every mesh-path bug used to be invisible to the fast suite.  This covers
    the three shapes that matter: one closed component, several disconnected
    components inside a single mesh (which is what a TopoDS_Solid cannot
    represent), and an open surface.
    """
    import numpy as np
    import trimesh

    out = tmp_path_factory.mktemp("mesh_model")
    assets = out / "assets"
    assets.mkdir()

    single = trimesh.creation.box(extents=(0.10, 0.12, 0.14))
    single.export(assets / "single.stl")

    # Two boxes far enough apart to share no vertex: one mesh, two solids.
    big = trimesh.creation.box(extents=(0.08, 0.08, 0.08))
    small = trimesh.creation.box(extents=(0.04, 0.04, 0.04))
    small.apply_translation([0.30, 0.0, 0.0])
    trimesh.util.concatenate([big, small]).export(assets / "twopiece.stl")

    # A box with its top face taken out, so it is open until something fills
    # it.  Select by centroid rather than by face index: trimesh does not
    # promise that faces 0 and 1 are the two halves of one side.
    opened = trimesh.creation.box(extents=(0.06, 0.06, 0.06))
    opened.update_faces(opened.triangles_center[:, 2] < 0.029)
    opened.remove_unreferenced_vertices()
    opened.export(assets / "open.stl")

    (out / "meshes.xml").write_text(
        """
<mujoco model="meshtest">
  <compiler meshdir="assets" angle="degree"/>
  <asset>
    <mesh name="single" file="single.stl"/>
    <mesh name="twopiece" file="twopiece.stl"/>
    <mesh name="opened" file="open.stl"/>
  </asset>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="g_single" type="mesh" mesh="single" contype="0" conaffinity="0" group="2"/>
      <body name="arm" pos="0.3 0 0" euler="0 30 0">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-90 90"/>
        <geom name="g_two" type="mesh" mesh="twopiece" contype="0" conaffinity="0" group="2"/>
        <body name="tip" pos="0 0 0.25" euler="20 0 45">
          <joint name="wrist" type="hinge" axis="1 0 0" range="-45 45"/>
          <geom name="g_open" type="mesh" mesh="opened" contype="0" conaffinity="0" group="2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    )
    return out / "meshes.xml"


@pytest.fixture(scope="session")
def mesh_export(mesh_model, tmp_path_factory):
    from sim2cad.export import run_export

    out = tmp_path_factory.mktemp("mesh_export")
    return run_export(mjcf=str(mesh_model), out=str(out), write_render=False)


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
