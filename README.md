# sim2cad

Turn a MuJoCo robot model into a STEP assembly you can open in Onshape,
Fusion or SolidWorks, measure, and design parts around.

Sim models know exactly where every link and joint is. CAD tools cannot read
MJCF. `sim2cad` bridges the two: one STEP part per link, nested the way the
kinematic tree nests, placed exactly where the simulation has it, in
millimetres, with the joint axes drawn in and the joint data written out
alongside.

## Quickstart

```bash
uv sync
uv run sim2cad info   --menagerie unitree_g1          # what is in the model
uv run sim2cad export --menagerie unitree_g1          # -> out/unitree_g1_qpos0/
```

That writes:

```
out/unitree_g1_qpos0/
  assembly.step        nested assembly, one node per link, joint axes included
  parts/<link>.step    per-link, origin at the link's own frame
  parts/<link>.stl
  joints.json          type, parent/child, anchor, axis, limits, actuator
  preview.glb          quick look
  report.md            every accuracy check, with numbers
  report.json
```

## What it is for

- *"Give me the G1 as a STEP assembly I can open in Onshape and measure."*
  → `assembly.step`
- *"Give me just the left forearm so I can design a mount around it."*
  → `parts/left_wrist_roll_link.step`, with its origin on the link's own frame
- *"Export it standing, not in the zero pose."*
  → `--pose keyframe:stand`

## What it is not

The output is a **dimensionally accurate envelope**, not a manufacturable
part. Sim geometry has no fasteners, wall thickness, bearings or motor
pockets. Mesh links arrive as triangulated B-rep solids, not as parametric
features — use them to measure and to design *around*, not to machine from.

## Usage

```
sim2cad export --menagerie unitree_g1 [--file g1_with_hands]
sim2cad export --mjcf path/to/model.xml

  --geoms visual|collision|all   default visual, falls back if a body has none
  --pose qpos0|keyframe:NAME|json:FILE
  --units mm|m                   default mm
  --decimate 0.0-1.0             keep this fraction of triangles (default off)
  --merge-coplanar / --no-merge-coplanar
  --joint-markers / --no-joint-markers
  --ground                       drop the robot so its lowest point is z=0
  --keep-root-pose               keep the sim's root pose
  --only BODY[,BODY...]          export a subset of links
  --printable                    also write slicer-ready STLs (always in mm)
  --jobs N / -j N                processes for mesh conversion (0 = one per CPU)
  --out DIR

sim2cad info     --menagerie unitree_g1   # bodies, joints, geoms, keyframes
sim2cad validate out/<dir>                # re-run the checks on an export
```

## Output size

Sim meshes are dense, and STEP stores triangles as individual B-rep faces. The
G1's 393,270 visual triangles come out as roughly a gigabyte of
`assembly.step`, which is accurate but awkward to open. The levers, in the
order worth trying:

| lever | effect |
|---|---|
| `--decimate 0.25` | keeps a quarter of the triangles; shrinks file, build and validation together |
| `--geoms collision` | the simplified collision set (219k triangles for the G1) rather than the detailed visual art |
| `--only LINK[,LINK]` | one link and its ancestors, when you only need somewhere to bolt to |

`--merge-coplanar` is on by default and is *not* one of these levers: the G1's
visual meshes are organic surfaces with almost no coplanar neighbours, so it
removes 0.09% of the faces. It earns its keep on boxy geometry, where it turns
a triangulated panel back into one flat face you can click on.

The default is full-fidelity `visual` geometry, because the point of this tool
is that the dimensions are right.

## Printing

`--printable` writes `printable/<link>.stl` alongside the CAD: each link
repaired to a closed, manifold solid a slicer will accept, always in
millimetres whatever `--units` says. Overlapping pieces of one link are fused;
pieces that merely sit near each other are left as separate objects, because
that is what they are.

**No repair is accepted if it moves the part's dimensions.** A repair must
keep the link's bounds within 0.1% of its own size or it is refused and the
link is left open, with the reason recorded. A closed shape that is the wrong
shape is worse than an open one, because nothing about the file looks wrong:
on the G1 the voxel fallback wanted to move three hip links' bounds by 117%,
117% and 585%, and all three were refused.

The repairs are reported per link, along with how far the bounds moved:

| method | what happened |
|---|---|
| already watertight | nothing; the mesh was closed to begin with |
| holes filled | boundary loops capped, geometry otherwise untouched |
| meshfix per piece | non-manifold geometry stripped and holes closed |
| voxel remesh per piece | surface rebuilt on a grid; lossy, and usually refused |
| ... + union | overlapping pieces of the link fused into one solid |

On the G1, 25 of 30 links come out closed, every one of them dimensionally
exact (bounds within 3e-8 of the CAD). The other five say why they could not
be closed. Fifteen stray single-triangle slivers are dropped and counted.

**This prints a model, not a robot.** The parts are the links' outer
envelopes: no wall thickness, no fasteners, no bearing seats, no motor
pockets, and no actuators — so there is nothing to bolt together. What it is
good for is a display model, a fit-check mock-up of a single link, or, most
usefully, the dimensions: take the link lengths and joint axes from
`joints.json` and design your own printable structure around them, and the
MuJoCo model stays a valid testbed for whatever you build.

## Accuracy

Every export runs the full check suite and writes the numbers into
`report.md`. On the G1 and on `humanoid.xml`:

- primitives land at machine precision — centroid and bounding box within
  **1e-10 mm** of the analytic answer, volumes within **1e-13** relative;
- mesh links land within **1e-9 mm** of MuJoCo's own world vertices;
- `joints.json` anchors and axes replay MuJoCo's kinematics to **< 1e-6 m**:
  drive a real joint 0.5 rad in the sim, rotate the child link about our
  recorded anchor and axis, and it lands in the same place;
- the coplanar merge changes no volume (**3.5e-14** relative on the G1).

A full G1 export takes about 11 minutes and produces roughly a gigabyte of
STEP; see **Output size** above. Mesh conversion runs one process per core by
default (`--jobs`).

The same checks pass on the Franka Panda (fixed base, serial chain, sliding
gripper), ANYmal C (free base, four legs) and the Shadow Hand (26 bodies, five
branches) -- see `tests/test_generality.py`, which runs under `--run-slow`.

Measurements are taken by re-opening the written `assembly.step`, so the frame
maths, the STEP write and the STEP read are all inside the loop being tested.

Run them yourself:

```bash
uv run pytest -q                 # fast: frame maths, recentering, humanoid
uv run pytest -q --run-slow      # adds the full G1 mesh export
```

## How it works

See [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md) — kinematic trees, frame
composition, the MuJoCo-vs-SciPy quaternion order, the mesh recentering trap
(which misplaces every G1 mesh by up to 433 mm if you get it wrong), what
sewing and coplanar merging do, and what each check proves.

[`NOTES.md`](NOTES.md) is the engineering log: measured model counts,
benchmarks, and the bugs found on the way.

## Licence

The vendored `tests/data/humanoid.xml` is from `google-deepmind/mujoco`,
Apache 2.0. Menagerie models carry their own licences (the G1 is BSD-3-Clause).
