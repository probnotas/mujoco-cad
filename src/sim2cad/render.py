"""§8.9 -- render the sim and the exported CAD from the same camera.

MuJoCo draws the simulation; pyrender draws ``preview.glb``, which was
tessellated from the exported B-rep.  Putting the two side by side is the check
that catches the errors numbers do not: a limb attached to the wrong parent, a
mirrored part, a link rotated by 180 degrees.

Both renderers need an offscreen GL context.  If there is not one,
:func:`render_comparison` returns ``None`` and the caller reports a skip rather
than a failure, as the PRD asks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["CameraView", "DEFAULT_VIEWS", "render_comparison", "RenderUnavailable"]


class RenderUnavailable(RuntimeError):
    """No usable offscreen GL context."""


@dataclass(frozen=True)
class CameraView:
    """A MuJoCo free-camera view, which we mirror exactly in pyrender."""

    name: str
    azimuth: float
    elevation: float

    def eye_from(self, lookat: np.ndarray, distance: float) -> np.ndarray:
        az, el = np.radians(self.azimuth), np.radians(self.elevation)
        forward = np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], dtype=float
        )
        return lookat - distance * forward

    def pose_matrix(self, lookat: np.ndarray, distance: float) -> np.ndarray:
        """glTF-style camera pose: +x right, +y up, -z forward."""
        eye = self.eye_from(lookat, distance)
        back = eye - lookat
        back /= np.linalg.norm(back)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(back @ world_up)) > 0.999:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(world_up, back)
        right /= np.linalg.norm(right)
        up = np.cross(back, right)
        mat = np.eye(4)
        mat[:3, 0], mat[:3, 1], mat[:3, 2], mat[:3, 3] = right, up, back, eye
        return mat


DEFAULT_VIEWS = (
    CameraView("front", azimuth=90.0, elevation=-10.0),
    CameraView("side", azimuth=0.0, elevation=-10.0),
    CameraView("three_quarter", azimuth=45.0, elevation=-20.0),
)


def _ensure_gl() -> None:
    """Pick a software GL backend if the environment has not chosen one."""
    if not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "osmesa"
    if not os.environ.get("PYOPENGL_PLATFORM"):
        os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]


def _render_mujoco(lm, views, width, height, lookat, distance, fovy, groups=None):
    """Render the sim with only the geoms we exported, and no scenery.

    The floor, shadows, skybox and haze have no counterpart in a CAD assembly,
    and they dominate the frame.  Switching them off is what turns the
    side-by-side into a real comparison rather than two unrelated pictures.
    """
    import mujoco

    # A model's offscreen buffer defaults to 640x480 unless its MJCF says
    # otherwise, and mujoco.Renderer refuses any size larger than that.
    # humanoid.xml happens to ask for 2560x1440, the G1 says nothing, so the
    # same request succeeded on one and failed on the other.  Raise the
    # buffer to fit what we are about to ask for.
    visual = lm.model.vis.global_
    visual.offwidth = max(int(visual.offwidth), int(width))
    visual.offheight = max(int(visual.offheight), int(height))

    try:
        renderer = mujoco.Renderer(lm.model, height, width)
    except Exception as exc:  # pragma: no cover - no GL
        raise RenderUnavailable(f"mujoco.Renderer unavailable: {exc}") from exc

    option = mujoco.MjvOption()
    # World-body geoms (the floor) are "static"; we never export those.
    option.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 0
    if groups is not None:
        for g in range(len(option.geomgroup)):
            option.geomgroup[g] = 1 if g in groups else 0

    images = {}
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = lookat
        camera.distance = distance
        for view in views:
            camera.azimuth = view.azimuth
            camera.elevation = view.elevation
            renderer.update_scene(lm.data, camera=camera, scene_option=option)
            for flag in (
                mujoco.mjtRndFlag.mjRND_SHADOW,
                mujoco.mjtRndFlag.mjRND_REFLECTION,
                mujoco.mjtRndFlag.mjRND_SKYBOX,
                mujoco.mjtRndFlag.mjRND_HAZE,
            ):
                renderer.scene.flags[flag] = 0
            images[view.name] = renderer.render().copy()
    finally:
        renderer.close()
    return images


def _rasterize(
    meshes, eye, target, width, height, fovy
) -> np.ndarray:
    """A small z-buffered software rasteriser.

    pyrender would do this, but it pins an old PyOpenGL whose OSMesa bindings
    are missing the entry point it needs.  Rasterising ourselves removes the
    dependency and, more usefully, means the CAD image uses exactly the camera
    maths we wrote down rather than a second implementation of it.
    """
    back = eye - target
    back = back / np.linalg.norm(back)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(back @ world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(world_up, back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    rot = np.stack([right, up, back])  # world -> camera

    focal = (height / 2.0) / np.tan(np.radians(fovy) / 2.0)
    colour = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.full((height, width), np.inf)

    for verts, faces, rgba in meshes:
        cam = (verts - eye) @ rot.T
        z = -cam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            sx = width / 2.0 + focal * cam[:, 0] / z
            sy = height / 2.0 - focal * cam[:, 1] / z
        base = np.array(rgba[:3], dtype=float)

        tri_z = z[faces]
        visible = (tri_z > 1e-6).all(axis=1)
        tri = faces[visible]
        if len(tri) == 0:
            continue

        p0 = np.stack([sx[tri[:, 0]], sy[tri[:, 0]]], axis=1)
        p1 = np.stack([sx[tri[:, 1]], sy[tri[:, 1]]], axis=1)
        p2 = np.stack([sx[tri[:, 2]], sy[tri[:, 2]]], axis=1)
        area = (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p1[:, 1] - p0[:, 1]) * (
            p2[:, 0] - p0[:, 0]
        )

        normals = np.cross(
            verts[tri[:, 1]] - verts[tri[:, 0]], verts[tri[:, 2]] - verts[tri[:, 0]]
        )
        lengths = np.linalg.norm(normals, axis=1)
        good = (np.abs(area) > 1e-9) & (lengths > 1e-20)
        # A headlight: brightness from how square-on the facet is to the camera.
        shade = np.zeros(len(tri))
        shade[good] = np.abs(normals[good] @ back) / lengths[good]
        shade = 0.25 + 0.75 * shade

        zs = tri_z[visible]
        for t in np.nonzero(good)[0]:
            a0, a1, a2 = p0[t], p1[t], p2[t]
            xmin = max(int(np.floor(min(a0[0], a1[0], a2[0]))), 0)
            xmax = min(int(np.ceil(max(a0[0], a1[0], a2[0]))) + 1, width)
            ymin = max(int(np.floor(min(a0[1], a1[1], a2[1]))), 0)
            ymax = min(int(np.ceil(max(a0[1], a1[1], a2[1]))) + 1, height)
            if xmin >= xmax or ymin >= ymax:
                continue
            xs = np.arange(xmin, xmax) + 0.5
            ys = np.arange(ymin, ymax) + 0.5
            gx, gy = np.meshgrid(xs, ys)
            inv = 1.0 / area[t]
            w0 = ((a1[0] - gx) * (a2[1] - gy) - (a1[1] - gy) * (a2[0] - gx)) * inv
            w1 = ((a2[0] - gx) * (a0[1] - gy) - (a2[1] - gy) * (a0[0] - gx)) * inv
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            zt = w0 * zs[t, 0] + w1 * zs[t, 1] + w2 * zs[t, 2]
            window = depth[ymin:ymax, xmin:xmax]
            hit = inside & (zt < window)
            if not hit.any():
                continue
            window[hit] = zt[hit]
            colour[ymin:ymax, xmin:xmax][hit] = base * shade[t]

    return (np.clip(colour, 0.0, 1.0) * 255).astype(np.uint8)


def _glb_meshes(path: Path):
    """World-space triangles and colours from the preview GLB (metres)."""
    import trimesh

    scene = trimesh.load(str(path), force="scene")
    out = []
    for name, geometry in scene.geometry.items():
        transform = scene.graph.get(scene.graph.geometry_nodes[name][0])[0]
        verts = trimesh.transform_points(np.asarray(geometry.vertices, dtype=float), transform)
        faces = np.asarray(geometry.faces, dtype=np.int64)
        colours = getattr(geometry.visual, "face_colors", None)
        rgba = (
            np.asarray(colours[0], dtype=float) / 255.0
            if colours is not None and len(colours)
            else np.array([0.7, 0.7, 0.72, 1.0])
        )
        out.append((verts, faces, rgba))
    return out


def _render_glb(path: Path, views, width, height, lookat, distance, fovy):
    meshes = _glb_meshes(path)
    images = {}
    for view in views:
        eye = view.eye_from(lookat, distance)
        images[view.name] = _rasterize(meshes, eye, lookat, width, height, fovy)
    return images


def render_comparison(
    result,
    out_dir: Path | None = None,
    width: int = 480,
    height: int = 640,
    views=DEFAULT_VIEWS,
) -> dict[str, Path] | None:
    """Write ``compare_<view>.png`` images, or return ``None`` if GL is absent."""
    _ensure_gl()
    from PIL import Image

    lm = result.lm
    if result.preview_path is None or not Path(result.preview_path).exists():
        return None

    out_dir = Path(out_dir or result.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = lm.model
    # The GLB is in metres and placed in the exported world frame, so the
    # camera must look at the same spot in that frame, not MuJoCo's.
    lookat_sim = np.array(model.stat.center, dtype=float)
    distance = 1.6 * float(model.stat.extent)
    fovy = float(model.vis.global_.fovy)
    lookat_cad = lm.world_fix.apply(lookat_sim)

    groups = {
        int(model.geom_group[g]) for g in result.selection.geom_ids
    } or None
    try:
        sim = _render_mujoco(
            lm, views, width, height, lookat_sim, distance, fovy, groups=groups
        )
        cad = _render_glb(Path(result.preview_path), views, width, height, lookat_cad, distance, fovy)
    except RenderUnavailable:
        return None

    written: dict[str, Path] = {}
    for view in views:
        left, right = sim[view.name], cad[view.name]
        canvas = np.zeros((height, width * 2 + 8, 3), dtype=np.uint8)
        canvas[:, :width] = left
        canvas[:, width + 8 :] = right
        path = out_dir / f"compare_{view.name}.png"
        Image.fromarray(canvas).save(path)
        written[view.name] = path
    return written
