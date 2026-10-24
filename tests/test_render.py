"""§8.9 -- the side-by-side visual check.

Skips rather than fails when there is no offscreen GL context, as the PRD
asks.  On this container MuJoCo needs ``libosmesa6`` and ``MUJOCO_GL=osmesa``;
the CAD side is rasterised in numpy and always works.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim2cad.render import DEFAULT_VIEWS, CameraView, render_comparison


def test_camera_pose_matches_the_mujoco_convention():
    """Azimuth/elevation must place the eye the same way MuJoCo does."""
    lookat = np.array([0.0, 0.0, 1.0])
    view = CameraView("front", azimuth=90.0, elevation=0.0)
    eye = view.eye_from(lookat, 2.0)
    # azimuth 90 looks along +y, so the camera sits on the -y side.
    assert np.allclose(eye, [0.0, -2.0, 1.0], atol=1e-12)

    pose = view.pose_matrix(lookat, 2.0)
    assert np.allclose(pose[:3, 3], eye, atol=1e-12)
    # Column 2 is "backwards" in glTF convention: away from the subject.
    assert np.allclose(pose[:3, 2], [0.0, -1.0, 0.0], atol=1e-12)
    # The basis is right-handed and orthonormal.
    assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(pose[:3, :3]), 1.0)


def test_elevation_lifts_the_camera():
    lookat = np.zeros(3)
    low = CameraView("a", 0.0, 0.0).eye_from(lookat, 1.0)
    high = CameraView("b", 0.0, -30.0).eye_from(lookat, 1.0)
    # Negative elevation in MuJoCo means looking down from above.
    assert high[2] > low[2]


def test_rasteriser_draws_something_where_it_should(tmp_path):
    from sim2cad.render import _rasterize

    # A unit square in the z=0 plane, viewed from straight above.
    verts = np.array([[-1.0, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    img = _rasterize(
        [(verts, faces, (1.0, 0.0, 0.0, 1.0))],
        eye=np.array([0.0, 0.0, 5.0]),
        target=np.zeros(3),
        width=64,
        height=64,
        fovy=45.0,
    )
    assert img.shape == (64, 64, 3)
    assert img[32, 32, 0] > 0, "the square should cover the centre pixel"
    assert img[32, 32, 1] == 0 and img[32, 32, 2] == 0, "and it should be red"
    assert img[1, 1].sum() == 0, "the corners should be background"


def test_comparison_images(humanoid_export):
    images = render_comparison(humanoid_export, width=240, height=320)
    if images is None:
        pytest.skip("no offscreen GL context available for MuJoCo rendering")

    from PIL import Image

    assert set(images) == {v.name for v in DEFAULT_VIEWS}
    for path in images.values():
        assert path.exists()
        array = np.asarray(Image.open(path))
        assert array.shape == (320, 240 * 2 + 8, 3)
        left, right = array[:, :240], array[:, 248:]
        # Both halves must actually contain a robot, not be blank.
        assert (left.sum(axis=2) > 20).mean() > 0.05
        assert (right.sum(axis=2) > 20).mean() > 0.05

        # And the silhouettes must broadly agree: the robot should occupy the
        # same rows in both renders.  This is what catches a limb on the wrong
        # parent or a part mirrored about the body axis.
        lrows = (left.sum(axis=2) > 20).sum(axis=1).astype(float)
        rrows = (right.sum(axis=2) > 20).sum(axis=1).astype(float)
        occupied_left = lrows > 0
        occupied_right = rrows > 0
        overlap = (occupied_left & occupied_right).sum() / max(
            occupied_left.sum(), occupied_right.sum()
        )
        assert overlap > 0.85, f"silhouette rows only {overlap:.0%} in common"


def test_renders_a_model_that_declares_no_offscreen_buffer(mesh_export):
    """A model's offscreen buffer defaults to 640x480 and caps the renderer.

    humanoid.xml asks for 2560x1440 in its MJCF, so it hid this; the G1 says
    nothing and the default comparison size (480 wide, 640 high) exceeded the
    480-pixel default height.  The mesh fixture also declares nothing, so it
    reproduces the failure.
    """
    model = mesh_export.lm.model
    assert int(model.vis.global_.offheight) < 640, "fixture should use the default buffer"

    images = render_comparison(mesh_export, width=480, height=640)
    if images is None:
        pytest.skip("no offscreen GL context available for MuJoCo rendering")
    assert len(images) == len(DEFAULT_VIEWS)
    for path in images.values():
        assert path.exists()
