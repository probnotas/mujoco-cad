# NOTES

Running log of decisions, measurements and places where reality differed from
the PRD. Numbers here were measured in this repo, not estimated.

## Environment

| thing | version | note |
|---|---|---|
| Python | 3.12 | pinned in `pyproject.toml` (`>=3.12,<3.13`); CadQuery has no 3.13 wheels yet |
| mujoco | 3.17.0 | |
| cadquery | 2.8.0 | OCP / OpenCascade underneath |
| trimesh | 5.2.0 | plus `networkx` (hole filling) and `fast-simplification` (`--decimate`) |
| mujoco_menagerie | 2026.10.4 | |

`uv sync` installs everything; `uv.lock` is committed.

## Menagerie API (§4 asked us to verify which one works)

`mujoco_menagerie` exists on PyPI and works, but **not** with the
`mm.load("unitree_g1")` signature the PRD guessed. What it actually offers:

```python
import mujoco_menagerie as mm
robot = mm.get("unitree_g1")      # metadata, no I/O
path  = robot.xml("g1")           # downloads on first use, returns a Path
```

`mm.load(name, entry)` returns a compiled `MjModel`, and `robot.path(...)` has
a signature that rejects a positional string. sim2cad uses `robot.xml(entry)`
and then `MjModel.from_xml_path`, so the compile is ours and the PRD's
"use the compiled model" rule holds. `robot_descriptions` is kept as a
fallback in `load.py`. Assets land in `~/.cache/mujoco_menagerie/`.

The default entry point is `g1`; `g1_with_hands`, `g1_mjx` and `g1_mjlab` are
also available via `--file`.

## Actual model counts

### unitree_g1 / g1.xml

| | |
|---|---|
| bodies | 31 (1 world + 30 robot links) |
| joints | 30 — 1 free + 29 hinge |
| `nq` / `nv` | 36 / 35, so 29 articulated DOF (matches the PRD's "~29 DOF") |
| actuators | 29 |
| geoms | 71 — 59 mesh, 8 sphere, 4 cylinder |
| visual / collision | 35 visual (group 2), 36 collision (group 3) |
| keyframes | one, named `stand` (the PRD guessed `home`/`stand`) |
| mesh data | 196,692 vertices, 393,270 triangles across the 35 visual meshes |
| bodies with no geoms | none, apart from `world` |

Group 2 = visual and group 3 = collision held here, but the code classifies by
`contype == conaffinity == 0` and only *reports* the groups, as the PRD asked.

### humanoid.xml (pipeline test model)

Not shipped with the `mujoco` wheel, so it is vendored at
`tests/data/humanoid.xml` from `google-deepmind/mujoco` (Apache 2.0) to keep
the fast tests offline.

17 bodies, 22 joints (1 free + 21 hinge), 20 geoms (16 capsule, 3 sphere,
1 plane), 4 keyframes (`squat`, `stand_on_left_leg`, `prone`, `supine`).

**Discrepancy worth knowing:** humanoid.xml has **zero** visual geoms — every
geom has contact enabled. With the default `--geoms visual` all 16 bodies fall
back to their collision geoms. That is the §5.2 fallback path, and it is
reported as a warning rather than hidden. The `floor` plane is skipped both
because it is a plane and because it belongs to `world`.

## Measurements that shaped the implementation

### The recentering pitfall is real, and large

`mesh_pos` reaches 0.43 m and `mesh_quat` departs from identity by up to 1.27.
Placing the raw STL with the authored XML pose misplaces **all 59** G1 mesh
geoms: median 107 mm, worst 433 mm (measured over each geom's bounding-box
corners, which catches rotation as well as translation).

One G1 mesh is recentered by almost pure rotation — its *position* error is
0.0001 mm while the part still lands 29 mm out. A position-only check would
wave it through, so the test measures corner displacement.

The exact relation `compiled_geom_pose == authored_geom_pose @ recentering`
holds to 0.0 on every G1 mesh geom.

### Sewing triangles with `BRepBuilderAPI_Sewing` does not scale

| triangles | build faces | `Sewing` | topological shell |
|---|---|---|---|
| 5,120 | 0.9 s | 1.8 s | 0.9 s total |
| 20,480 | 2.8 s | 20.0 s | 2.9 s total |
| 81,920 | — | (not attempted) | 11.1 s total |

`Sewing` searches geometrically for coincident points and goes superlinear. An
indexed mesh already carries the connectivity, so `meshes.py` builds the shell
topologically — one `TopoDS_Vertex` per vertex, one `TopoDS_Edge` per vertex
pair, shared between the triangles that meet there. Linear, and the volume
matches trimesh to full double precision.

### `ShapeFix_Shell` and `ShapeFix_Solid` were most of the runtime

Profiling the G1's `left_rubber_hand` (45,748 triangles):

| stage | seconds |
|---|---|
| extract + clean | 0.1 |
| topological shell | 6.5 |
| `ShapeFix_Shell` | 16.7 |
| `MakeSolid` | 0.0 |
| `ShapeFix_Solid` | 19.9 |
| `UnifySameDomain` | 3.8 |
| `BRepCheck_Analyzer` | 10.1 |

The shell is already sewn and manifold by construction, so both ShapeFix
passes had nothing to find. They are gone: `MakeSolid` runs directly, a
negative volume flips the shell (cheaper than a repair pass), and ShapeFix is
only reached if `BRepCheck` actually fails. `BRepCheck` also used to run twice
— once in `mesh_to_shape`, once in `validate` — and now runs once.

### 15 of 35 G1 visual meshes are not watertight

`trimesh.repair.fill_holes` rescues 6 of the 21 that start open; the other 15
have genuine holes or non-manifold edges in the source art and are exported as
open **shells**, flagged in `report.md`, exactly as §5.4 step 2 allows.

Separately, several meshes have edges shared by **three or four** triangles
(the torso has 11 and 23 of them). OpenCascade rejects those outright, so
`ShapeFix_Shape` runs as a fallback when `BRepCheck_Analyzer` fails; it repairs
them without dropping a single face.

## Bugs found while building

**CadQuery copies a sub-assembly when you add it.** Adding a parent to the
tree and *then* adding children to that parent silently loses the children —
the first humanoid export came out with only the torso. The tree is now built
leaves-first.

**Sub-assembly names do not survive STEP on the instance label.** Round-tripped
sub-assemblies came back named `29`, `34`, `57`. The name we wrote is on the
*referred* (prototype) label; leaf parts happen to carry it on both. The reader
prefers the referred label.

**OpenCascade's STEP unit setting is global and leaks.** `exportAssembly(...,
unit="M")` sets `xstep.cascade.unit` and never restores it, so a millimetre
file written or read later *in the same process* comes out scaled by 1000. This
showed up as a metre export and a millimetre export reporting identical
volumes. Every STEP read and write now goes through `occunits.cascade_unit`,
which saves and restores the setting.

**`BRepBndLib::AddOptimal` is millimetre-native.** On the same rotated capsule
it is exact to ~5e-11 mm at millimetre scale but off by 2.6e-5 m when the
geometry is written in metres — its convergence tolerance is absolute. Verified
against an independent analytic capsule bounding box, which agrees with our
formula to 1e-16 at both scales, so the kernel is the imprecise one. Validation
therefore reads every file back **as millimetres** regardless of `--units` and
compares in millimetres, which is also the unit the PRD quotes its tolerances
in. Also pass `useTriangulation=False`, or OCC measures whatever cached
triangulation happens to be on the shape.

## Rendering (§8.9)

No GPU and no EGL in this container. MuJoCo renders offscreen once `libosmesa6`
is installed (`MUJOCO_GL=osmesa`). `pyrender` cannot be used for the CAD side:
it hard-pins `pyopengl==3.1.0`, whose OSMesa bindings are missing
`OSMesaCreateContextAttribs`, and the pin makes a newer PyOpenGL unresolvable.
`render.py` therefore rasterises the preview GLB itself in numpy — a z-buffered
triangle rasteriser, about 80 lines — which removes the dependency and means
the CAD image uses exactly the camera maths we wrote down rather than a second
implementation of it. If no GL context exists at all, the check skips rather
than fails, as the PRD asks.

## Deliberate choices

- **Validation reads the written file.** Placement and volume are measured by
  re-opening `assembly.step` through XCAF, so the frame composition, the STEP
  write and the STEP read are all inside the loop being tested.
- **Surface centroid, not vertex mean.** The coplanar merge deletes interior
  vertices, so a vertex-mean centroid would move even when the surface is
  identical. The area-weighted centroid is a property of the surface.
- **Primitive bounding boxes are analytic.** Tessellating a sphere to compare
  bounding boxes would cap accuracy at the tessellation tolerance, far above
  the PRD's 1e-6 mm. Each primitive's exact world AABB is derived in closed
  form in `validate.expected_primitive_bbox`.
- **`--only` keeps ancestors.** Exporting one link alone would lose its place
  in the world, so ancestors are kept as empty nodes and the chosen link stays
  where the sim had it.
- **`--ground` and root normalisation are one world correction.** Rather than
  editing `qpos`, a single `world_fix` pose is applied to every world-frame
  quantity on the way out. Relative poses inside the tree are untouched, so it
  works for fixed-base robots too.
