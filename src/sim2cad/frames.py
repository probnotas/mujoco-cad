"""Rigid transforms, and the quaternion conventions that trip everyone up.

MuJoCo stores quaternions as ``(w, x, y, z)``.  ``scipy.spatial.transform``
stores them as ``(x, y, z, w)``.  Mixing the two silently produces a wrong but
plausible-looking rotation, so every conversion in sim2cad goes through
:func:`mj_quat_to_scipy` / :func:`scipy_quat_to_mj` rather than reordering
inline.

Rotation matrices here are the usual "columns are the frame's basis vectors
expressed in the parent frame" convention, which is what MuJoCo's ``xmat``
(row-major 3x3) gives us.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

__all__ = [
    "Pose",
    "mj_quat_to_scipy",
    "scipy_quat_to_mj",
    "mj_quat_to_matrix",
    "matrix_to_mj_quat",
]


def mj_quat_to_scipy(quat: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` -> ``(x, y, z, w)``."""
    q = np.asarray(quat, dtype=float).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]])


def scipy_quat_to_mj(quat: np.ndarray) -> np.ndarray:
    """``(x, y, z, w)`` -> ``(w, x, y, z)``."""
    q = np.asarray(quat, dtype=float).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]])


def mj_quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Rotation matrix from a MuJoCo ``(w, x, y, z)`` quaternion."""
    return Rotation.from_quat(mj_quat_to_scipy(quat)).as_matrix()


def matrix_to_mj_quat(mat: np.ndarray) -> np.ndarray:
    """MuJoCo ``(w, x, y, z)`` quaternion from a rotation matrix.

    The sign is normalised so ``w >= 0``; ``q`` and ``-q`` are the same
    rotation, but a stable choice makes round-trip tests readable.
    """
    q = scipy_quat_to_mj(Rotation.from_matrix(np.asarray(mat, float).reshape(3, 3)).as_quat())
    return -q if q[0] < 0 else q


@dataclass(frozen=True)
class Pose:
    """A rigid transform: a rotation then a translation.

    A ``Pose`` maps points from its own frame into its parent frame::

        p_parent = rot @ p_self + pos

    Composition uses ``@``: ``world_from_child = world_from_body @ body_from_child``.
    Read a chain right-to-left, the way the frames nest.
    """

    pos: np.ndarray
    rot: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", np.asarray(self.pos, dtype=float).reshape(3))
        object.__setattr__(self, "rot", np.asarray(self.rot, dtype=float).reshape(3, 3))

    # -- constructors ----------------------------------------------------

    @staticmethod
    def identity() -> "Pose":
        return Pose(np.zeros(3), np.eye(3))

    @staticmethod
    def from_mj(pos: np.ndarray, quat: np.ndarray) -> "Pose":
        """From a MuJoCo position and a ``(w, x, y, z)`` quaternion."""
        return Pose(np.asarray(pos, float), mj_quat_to_matrix(quat))

    @staticmethod
    def from_xmat(pos: np.ndarray, xmat: np.ndarray) -> "Pose":
        """From a MuJoCo position and a row-major flat ``xmat`` (9 floats)."""
        return Pose(np.asarray(pos, float), np.asarray(xmat, float).reshape(3, 3))

    @staticmethod
    def from_translation(pos: np.ndarray) -> "Pose":
        return Pose(np.asarray(pos, float), np.eye(3))

    # -- algebra ---------------------------------------------------------

    def __matmul__(self, other: "Pose") -> "Pose":
        if not isinstance(other, Pose):
            return NotImplemented
        return Pose(self.pos + self.rot @ other.pos, self.rot @ other.rot)

    def inverse(self) -> "Pose":
        rt = self.rot.T
        return Pose(-rt @ self.pos, rt)

    def relative_to(self, parent: "Pose") -> "Pose":
        """This pose expressed in ``parent``'s frame."""
        return parent.inverse() @ self

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Transform an ``(n, 3)`` array (or a single 3-vector) of points."""
        p = np.asarray(points, dtype=float)
        if p.ndim == 1:
            return self.rot @ p + self.pos
        return p @ self.rot.T + self.pos

    def rotate(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate direction vectors (no translation)."""
        v = np.asarray(vectors, dtype=float)
        if v.ndim == 1:
            return self.rot @ v
        return v @ self.rot.T

    def scaled(self, factor: float) -> "Pose":
        """Same rotation, translation multiplied by ``factor``.

        Used once, at export time, to go from metres to millimetres.
        """
        return Pose(self.pos * factor, self.rot)

    # -- interop ---------------------------------------------------------

    @property
    def quat(self) -> np.ndarray:
        """MuJoCo-order ``(w, x, y, z)`` quaternion."""
        return matrix_to_mj_quat(self.rot)

    def as_matrix(self) -> np.ndarray:
        """4x4 homogeneous matrix."""
        m = np.eye(4)
        m[:3, :3] = self.rot
        m[:3, 3] = self.pos
        return m

    @staticmethod
    def from_matrix(mat: np.ndarray) -> "Pose":
        m = np.asarray(mat, float).reshape(4, 4)
        return Pose(m[:3, 3], m[:3, :3])

    def to_location(self):
        """A CadQuery ``Location`` for this pose (imported lazily)."""
        import cadquery as cq
        from OCP.gp import gp_Trsf, gp_Mat, gp_XYZ

        trsf = gp_Trsf()
        r = self.rot
        trsf.SetValues(
            r[0, 0], r[0, 1], r[0, 2], self.pos[0],
            r[1, 0], r[1, 1], r[1, 2], self.pos[1],
            r[2, 0], r[2, 1], r[2, 2], self.pos[2],
        )
        return cq.Location(trsf)

    # -- comparison ------------------------------------------------------

    def allclose(self, other: "Pose", atol: float = 1e-12) -> bool:
        return bool(
            np.allclose(self.pos, other.pos, atol=atol, rtol=0)
            and np.allclose(self.rot, other.rot, atol=atol, rtol=0)
        )

    def __repr__(self) -> str:
        p = ", ".join(f"{v:.6g}" for v in self.pos)
        rpy = Rotation.from_matrix(self.rot).as_euler("xyz", degrees=True)
        r = ", ".join(f"{v:.3f}" for v in rpy)
        return f"Pose(pos=[{p}], rpy_deg=[{r}])"


def axis_angle_pose(anchor: np.ndarray, axis: np.ndarray, angle: float) -> Pose:
    """Rotation of ``angle`` radians about the line through ``anchor`` along ``axis``.

    This is how a hinge joint moves its child link, and it is what test 7 in the
    PRD replays from ``joints.json``.
    """
    axis = np.asarray(axis, float).reshape(3)
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("axis must be non-zero")
    rot = Rotation.from_rotvec(axis / norm * angle).as_matrix()
    anchor = np.asarray(anchor, float).reshape(3)
    return Pose(anchor - rot @ anchor, rot)
