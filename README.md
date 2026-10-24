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
  --out DIR

sim2cad info     --menagerie unitree_g1   # bodies, joints, geoms, keyframes
sim2cad validate out/<dir>                # re-run the checks on an export
```

## Accuracy

Every export runs the full check suite and writes the numbers into
`report.md`. On the G1 and on `humanoid.xml`:

- primitives land at machine precision — centroid and bounding box within
  **1e-10 mm** of the analytic answer, volumes within **1e-13** relative;
- mesh links land within **1e-9 mm** of MuJoCo's own world vertices;
- `joints.json` anchors and axes replay MuJoCo's kinematics to **< 1e-6 m**:
  drive a real joint 0.5 rad in the sim, rotate the child link about our
  recorded anchor and axis, and it lands in the same place;
- the coplanar merge changes no volume.

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
