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

### One mesh is not always one solid

Four G1 visual meshes are several disconnected pieces stored as a single mesh:

| mesh | pieces |
|---|---|
| `left_hip_roll_link` | 2 |
| `right_hip_roll_link` | 2 |
| `left_knee_link` | 8 |
| `right_knee_link` | 8 |

Every edge is perfectly manifold (`{2: 5763}` for the hip roll) -- the pieces
simply are not joined. A `TopoDS_Solid` is one outer shell plus optional inner
voids, so handing it several disjoint outer shells gives a solid that
`BRepCheck_Analyzer` rejects and whose volume STEP reads back short, the extra
shells having been taken for cavities. The shortfalls were 12.54%, 12.54%,
1.26% and 5.34% -- exactly the volume of the pieces being lost.

`ShapeFix_Solid` made it worse rather than better: handed the two-piece hip
roll it returned a *valid* solid with 12.5% less volume, having quietly
discarded a piece. Its repairs are now only accepted when the volume survives.

Building one solid per connected component and collecting them in a compound
fixes all four, to machine precision:

| mesh | pieces | volume vs trimesh |
|---|---|---|
| `left_hip_roll_link` | 2 | 4.6e-16 |
| `right_hip_roll_link` | 2 | 1.8e-15 |
| `left_knee_link` | 8 | 8.2e-16 |
| `right_knee_link` | 8 | 4.1e-15 |

The reason this was not caught earlier is worth recording: **no test under 25
minutes touched the mesh path at all.** `humanoid.xml` is entirely primitives,
so the only mesh coverage was the full G1 export. `tests/conftest.py` now
builds a small mesh MJCF on the fly -- one closed component, one mesh holding
two disconnected boxes, one open surface -- and `tests/test_meshes.py` runs
against it in about 3 seconds.

### `mesh_vert` is float32

`model.mesh_vert` is single precision, so a mesh volume can only match an exact
analytic answer to ~1e-7 relative. Comparisons against *trimesh* are exact,
because both sides start from the same float32 vertices; comparisons against a
closed form need `rtol=1e-6`.

### `--merge-coplanar` buys almost nothing on the G1

393,314 faces become 392,955: **0.09%**. The G1's visual meshes are organic,
curved surfaces with almost no coplanar neighbours to merge. On boxy geometry
it works exactly as intended -- the test fixture's 12-triangle box collapses to
6 planar faces -- but it is not a route to a smaller G1 file. Only
`--decimate` or `--geoms collision` move that. It is still worth leaving on:
it costs little and it makes flat faces selectable in CAD where they exist.

### Multi-component meshes are the rule, not the exception

16 of the G1's 35 visual parts are several disconnected pieces, the torso being
22. Only four showed up as *volume* errors, because the rest are also
non-watertight and were exported as shells, where treating several pieces as
one shell produces no measurable discrepancy -- it was still wrong, just
invisible.

### Where the time went

From `report.json`, before and after parallelising the mesh build:

| phase | before | after |
|---|---|---|
| build (mesh -> B-rep) | 522 s | **65 s** |
| write `assembly.step` | 222 s | 204 s |
| write `parts/` | 195 s | 158 s |
| `preview.glb` | 139 s | 144 s |
| whole export, wall | 25 m 10 s | **15 m 51 s** |

The build is 35 independent meshes and was running on one of four cores, so
`--jobs` now runs them in a process pool. `TopoDS_Shape` cannot be pickled, so
workers return a `.brep` file; that round trip preserves topology exactly and
geometry to within a couple of ULP (it is a decimal text format), and a test
asserts the parallel and serial paths agree to 1e-12 relative. The 8x speedup
beats the 4 cores available because the old figure also included ShapeFix
thrashing on the multi-component solids it could not represent.

Writing and reading STEP is now 2/3 of the export, and that is a property of
putting 393k triangles into a text format, not something more cores will fix.

## Bugs found while building

**CadQuery copies a sub-assembly when you add it.** Adding a parent to the
tree and *then* adding children to that parent silently loses the children —
the first humanoid export came out with only the torso. The tree is now built
leaves-first.

**Sub-assembly names do not survive STEP on the instance label.** Round-tripped
sub-assemblies came back named `29`, `34`, `57`. The name we wrote is on the
*referred* (prototype) label; leaf parts happen to carry it on both. The reader
prefers the referred label.

**A model's offscreen render buffer caps `mujoco.Renderer`.** It defaults to
640x480 unless the MJCF says otherwise, and the renderer refuses any larger
size. `humanoid.xml` asks for 2560x1440, so the visual check worked there and
silently skipped on the G1, which asks for nothing. `render.py` now raises the
buffer to fit the requested size. The mesh test fixture also declares nothing,
so it covers this.

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

## Known limitations

- **`sim2cad validate` re-exports rather than measuring the files in place.**
  It reloads the model and rebuilds the CAD so it can compare like with like,
  which on the G1 means paying the full export cost again (~20 minutes) to
  check files that already exist. The measurable checks (placement, volume,
  validity, round trip) only need the written STEP plus a cheap
  `extract_mesh` for the reference geometry -- about a minute -- while the
  build-time facts (face counts before the merge, the merge's effect on
  volume) would have to be carried forward from `report.json`. Worth doing;
  not done yet.
- **Writing and re-reading the STEP dominates a full-fidelity G1 export.**
  Parallelising the build helps the largest single phase, but roughly a
  gigabyte of STEP still has to be written once and read back once for
  validation. Only `--decimate` or `--geoms collision` move that.
- **Tendons and equality constraints are not exported.** The Panda has one of
  each and the Shadow Hand has four tendons; they couple joints together for
  control but are not part of the kinematic tree, so `joints.json` reports each
  joint independently. A reader should not assume two tendon-coupled joints can
  be driven separately on the real robot.
- **Only the first body attached to `world` is exported.** A file holding a
  robot *and* a table is a scene, not a robot; sim2cad takes the first subtree
  and now says in the report which bodies it ignored.
- **`parts/` and `preview.glb` are still built serially**, 18% and 13% of the
  export. Both are per-part and could use the same process pool as the mesh
  build.

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
