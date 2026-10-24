"""§8.1 -- frame maths, checked against hand-computed answers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from sim2cad.frames import (
    Pose,
    axis_angle_pose,
    matrix_to_mj_quat,
    mj_quat_to_matrix,
    mj_quat_to_scipy,
    scipy_quat_to_mj,
)


def test_quaternion_order_is_not_an_accident():
    # 90 degrees about z.  MuJoCo writes (w, x, y, z).
    mj = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert np.allclose(mj_quat_to_scipy(mj), [0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    rot = mj_quat_to_matrix(mj)
    assert np.allclose(rot @ [1, 0, 0], [0, 1, 0], atol=1e-15)

    # Feeding the MuJoCo order straight to scipy gives a different rotation --
    # this is the bug the conversion helpers exist to prevent.
    wrong = Rotation.from_quat(mj).as_matrix()
    assert not np.allclose(wrong, rot)


@pytest.mark.parametrize("seed", range(8))
def test_quaternion_round_trip(seed):
    rng = np.random.default_rng(seed)
    quat = scipy_quat_to_mj(Rotation.random(random_state=seed).as_quat())
    back = matrix_to_mj_quat(mj_quat_to_matrix(quat))
    if np.dot(back, quat) < 0:
        back = -back
    assert np.allclose(back, quat, atol=1e-14)
    assert np.allclose(mj_quat_to_scipy(scipy_quat_to_mj(rng.normal(size=4))).shape, (4,))


def test_compose_known_rotations():
    # 90 deg about z, then 90 deg about x (applied in the parent frame) sends
    # x->z, y->-x, z->-y, which is a 120 deg rotation about (1, -1, 1)/sqrt(3).
    rz = Pose(np.zeros(3), Rotation.from_euler("z", 90, degrees=True).as_matrix())
    rx = Pose(np.zeros(3), Rotation.from_euler("x", 90, degrees=True).as_matrix())
    combined = rx @ rz
    assert np.allclose(combined.rot @ [1, 0, 0], [0, 0, 1], atol=1e-12)
    assert np.allclose(combined.rot @ [0, 1, 0], [-1, 0, 0], atol=1e-12)
    assert np.allclose(combined.rot @ [0, 0, 1], [0, -1, 0], atol=1e-12)
    axis_angle = Rotation.from_matrix(combined.rot).as_rotvec()
    assert np.isclose(np.linalg.norm(axis_angle), np.deg2rad(120), atol=1e-12)
    assert np.allclose(
        axis_angle / np.linalg.norm(axis_angle),
        np.array([1.0, -1.0, 1.0]) / np.sqrt(3),
        atol=1e-12,
    )


def test_order_matters():
    rz = Pose(np.array([1.0, 0, 0]), Rotation.from_euler("z", 90, degrees=True).as_matrix())
    rx = Pose(np.array([0.0, 2, 0]), Rotation.from_euler("x", 90, degrees=True).as_matrix())
    assert not (rz @ rx).allclose(rx @ rz, atol=1e-9)


def test_inverse_and_relative_to():
    a = Pose.from_mj([1, 2, 3], [0.5, 0.5, 0.5, 0.5])
    b = Pose.from_mj([-1, 0.25, 4], [np.cos(0.3), 0, np.sin(0.3), 0])
    assert (a @ a.inverse()).allclose(Pose.identity(), atol=1e-14)
    assert (a @ b).relative_to(a).allclose(b, atol=1e-14)
    point = np.array([0.3, -0.2, 1.1])
    assert np.allclose(a.inverse().apply(a.apply(point)), point, atol=1e-14)


def test_three_link_chain_by_hand():
    """A planar 3-link arm with hand-computed tip positions.

    Link 1 runs 2 along x from the origin, then turns 90 deg about z; link 2
    runs 1 along its own x; link 3 turns -90 deg and runs 3 along its x.
    """
    rot90 = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    rotm90 = Rotation.from_euler("z", -90, degrees=True).as_matrix()

    l1 = Pose(np.array([2.0, 0.0, 0.0]), rot90)
    l2 = Pose(np.array([1.0, 0.0, 0.0]), rotm90)
    l3 = Pose(np.array([3.0, 0.0, 0.0]), np.eye(3))

    w1 = l1
    w2 = w1 @ l2
    w3 = w2 @ l3

    assert np.allclose(w1.pos, [2, 0, 0], atol=1e-14)
    # After the 90 deg turn, link 2's +x points along world +y.
    assert np.allclose(w2.pos, [2, 1, 0], atol=1e-14)
    assert np.allclose(w2.rot, np.eye(3), atol=1e-14)
    # Back to world axes, so link 3 runs along +x.
    assert np.allclose(w3.pos, [5, 1, 0], atol=1e-14)

    # And the chain composes the same way read as one product.
    assert (l1 @ l2 @ l3).allclose(w3, atol=1e-14)


def test_scaled_only_touches_translation():
    p = Pose.from_mj([0.1, 0.2, 0.3], [0.5, 0.5, 0.5, 0.5])
    s = p.scaled(1000.0)
    assert np.allclose(s.pos, [100, 200, 300], atol=1e-12)
    assert np.allclose(s.rot, p.rot, atol=0)


def test_axis_angle_pose_about_offset_line():
    # Rotating 180 deg about the line x=1 parallel to z maps (0,0,0) -> (2,0,0).
    pose = axis_angle_pose([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], np.pi)
    assert np.allclose(pose.apply([0.0, 0.0, 0.0]), [2.0, 0.0, 0.0], atol=1e-14)
    # Points on the axis do not move.
    assert np.allclose(pose.apply([1.0, 0.0, 5.0]), [1.0, 0.0, 5.0], atol=1e-14)


def test_to_location_round_trips_through_occ():
    pose = Pose.from_mj([0.1, -2.0, 3.5], [0.2, 0.4, -0.5, 0.74])
    loc = pose.to_location()
    trsf = loc.wrapped.Transformation()
    rot = np.array([[trsf.Value(i, j) for j in range(1, 4)] for i in range(1, 4)])
    pos = np.array([trsf.Value(i, 4) for i in range(1, 4)])
    assert np.allclose(rot, pose.rot, atol=1e-12)
    assert np.allclose(pos, pose.pos, atol=1e-12)
