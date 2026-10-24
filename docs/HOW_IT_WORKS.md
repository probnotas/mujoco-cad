# How sim2cad works

A MuJoCo model and a CAD assembly describe the same robot in two different
languages. This is the phrasebook.

## 1. A robot is a tree

MJCF nests bodies inside bodies. The G1's left hand is written inside the
forearm, which is inside the upper arm, inside the torso, inside the pelvis.
Each body stores its pose **relative to its parent**, not to the world:

```xml
<body name="upper_arm" pos="0 0.2 0.3">
  <body name="forearm" pos="0 0 -0.28"> ... </body>
</body>
```

The forearm's `pos` is `0 0 -0.28` no matter where the robot is standing. To
find out where it actually is, you multiply poses down the chain from the root.
That is what MuJoCo's `mj_kinematics` does, and sim2cad reads the answer out of
`data.xpos` / `data.xmat` rather than re-deriving it.

CAD assemblies work the same way, which is lucky: `assembly.step` reuses the
tree exactly, so grabbing the upper arm in Fusion drags the forearm and hand
with it.

## 2. Frames, and why order matters

A **pose** is a rotation plus a translation. Applied to a point:

> `p_parent = R · p_self + t`

Composing two of them, `A @ B`, means "do B, then A":

> `(A @ B).R = A.R · B.R`  `(A @ B).t = A.t + A.R · B.t`

**Worked example.** Pose `A` rotates 90° about z and sits at the origin. Pose
`B` is no rotation, translated `(1, 0, 0)`.

- `A @ B` — translate 1 along x, then rotate the whole thing 90°: the point
  lands at `(0, 1, 0)`.
- `B @ A` — rotate first, then translate along the *original* x: `(1, 0, 0)`.

Different answers, so the order is not a detail. In `frames.py` this is
`Pose.__matmul__`, and `tests/test_frames.py::test_order_matters` pins it down.

To get a child's pose *relative to* its parent — which is exactly what a CAD
sub-assembly needs — you invert the parent and compose:

> `child_in_parent = parent⁻¹ @ child`

## 3. Quaternions: (w, x, y, z) vs (x, y, z, w)

A quaternion packs a rotation into four numbers. MuJoCo writes them
**w first**. SciPy writes them **w last**. Nothing crashes if you mix them up —
you just get a different, wrong rotation.

**Worked example.** A 90° turn about z is `w = cos45° ≈ 0.7071`,
`z = sin45° ≈ 0.7071`:

| convention | numbers |
|---|---|
| MuJoCo `(w,x,y,z)` | `(0.7071, 0, 0, 0.7071)` |
| SciPy `(x,y,z,w)`  | `(0, 0, 0.7071, 0.7071)` |

Hand MuJoCo's ordering straight to SciPy and it reads `x = 0.7071`,
`w = 0.7071`: a 90° turn about **x** instead of z. Your robot's arm ends up
pointing out of its chest. Every conversion in sim2cad goes through
`mj_quat_to_scipy` / `scipy_quat_to_mj` so this can only be got wrong in one
place, and a test checks that place.

## 4. The recentering trap

This is the one that bites hardest.

When MuJoCo compiles a model it **moves every mesh** so the mesh's frame sits
at its own centre, then changes the geom's position and orientation by the
opposite amount so the robot still looks the same. The move is recorded in
`model.mesh_pos` / `model.mesh_quat`, and the compiled geom pose is exactly:

> `compiled_geom_pose = authored_geom_pose @ recentering`

(sim2cad verifies that equation holds to 1e-12 on every G1 mesh.)

So if you load `pelvis.STL` off disk and place it with the `pos` and `quat`
written in the XML, you have used the *authored* pose with *un-recentered*
vertices, and the part lands in the wrong place.

**Worked example.** The G1's pelvis mesh is recentered by
`mesh_pos = (0, -0.000001, -0.076377)` m. Place the raw STL with the authored
pose and the pelvis sits **76 mm** too high. Across the G1's 59 mesh geoms the
error runs from 29 mm to **433 mm** — every single one is wrong.

The fix is to never read the STL. The compiled `model.mesh_vert` is already in
the geom's frame, so:

> `world_vertex = geom_xpos + geom_xmat · mesh_vert`

`tests/test_recentering.py` proves all of this, including that the naive route
really is broken.

One more thing in the same family: MuJoCo stores **half**-sizes. A box with
`size="0.1 0.2 0.3"` is 200 × 400 × 600 mm.

## 5. Meshes, B-reps, sewing, and merging

A **mesh** is a bag of triangles. A **B-rep** ("boundary representation") is
what CAD uses: real faces bounded by real edges meeting at real vertices, so
the kernel knows that two faces share an edge rather than merely touching
there. STEP is a B-rep format.

**Sewing** is the act of working out which triangle edges are the same edge and
joining them into a watertight shell. OpenCascade's `BRepBuilderAPI_Sewing`
does it by hunting for coincident points, which is slow — 20,000 triangles take
about 20 seconds, and the G1's torso alone has 51,000. But an indexed mesh
*already says* which triangles share which corner. sim2cad therefore builds the
shell topologically: create each vertex once, each edge once, and hand the same
edge object to both triangles that use it. That is linear, and the shell comes
out sewn with no searching at all. 82,000 triangles take 11 seconds, and the
volume matches the triangle mesh to full double precision.

**Unify same domain** merges neighbouring faces that lie on the *same* surface.
A flat panel chopped into 400 triangles becomes one flat face you can click on
and measure in CAD, and the file shrinks. It must not change the shape, so
sim2cad records the volume before and after and the report shows the
difference, which is zero.

Not every mesh survives as a solid: 15 of the G1's 35 visual meshes have
genuine holes or non-manifold edges in the original art. Those are exported as
open **shells** instead, flagged in the report, and repaired enough that
OpenCascade still calls them valid.

## 6. What each check actually proves

| check | what would break if it failed |
|---|---|
| **1. frame maths** | composition, inversion or the quaternion order is wrong — everything downstream is then wrong too |
| **2. recentering** | the meshes are offset from where the sim has them |
| **3. placement** | a part is in the wrong place. Measured by re-reading `assembly.step`, flattening the nesting to world coordinates, and comparing the bounding box and the area-weighted surface centroid against MuJoCo's own numbers. The centroid is a property of the *surface*, not of how it is triangulated, so it survives the coplanar merge; the bounding box comes from MuJoCo's compiled vertices for meshes and from exact analytic formulas for primitives |
| **4. volume** | the shape is distorted, or a millimetre conversion got applied twice. Primitives are checked against closed forms (a capsule is `πr²·2h + 4/3·πr³`); meshes against the triangle mesh's own volume; and before vs after the coplanar merge |
| **5. completeness** | a link or a joint is missing. Part count, joint count, and articulated DOF against `model.nv` minus the free joint's 6 |
| **6. validity** | the file opens but the solid is corrupt. `BRepCheck_Analyzer` on every part |
| **7. joint consistency** | the anchor or axis in `joints.json` is wrong. Drive the real joint 0.5 rad in MuJoCo, then rotate the child link about *our* recorded anchor and axis and check it lands in the same place |
| **8. poses** | the keyframe path silently exports the zero pose |
| **9. visual** | something numbers cannot catch: a mirrored part, a limb on the wrong parent. Renders the sim and the exported CAD from the same camera, side by side |
| **10. round trip** | the hierarchy or the names did not survive STEP. Re-opens the file and compares |

Checks 3 and 4 are the sharp ones, and they are sharp because the thing being
measured comes back out of the written file, not out of the variable that was
used to write it.
