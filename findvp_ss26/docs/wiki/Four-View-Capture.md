# Four-View Capture

`sample/capture_four_views.py` captures one top and three side views from
analytic equal-solid-angle representatives of the upper hemisphere. It uses
the same IK candidate backends and PyBullet safety validators as Levels 6 and
7, records the exact simulated robot/camera state, and attempts to return the
robot to home on both success and failure.

![Four equal-solid-angle regions on the upper hemisphere around the object](../images/four-view-hemisphere-partition.png)

*Four-view partition used for capture. The blue top cap and the orange, green,
and purple side sectors each cover one quarter of the upper hemisphere's solid
angle. The four black camera markers show the region representatives; every
camera optical axis points toward the object center. The illustration is not
to scale.*

## What is sampled

`generate_four_equal_area_viewpoints()` returns exactly four deterministic
positions in this order:

1. `top`;
2. `side_1`;
3. `side_2`;
4. `side_3`.

The top representative is at 90 degrees elevation. Side representatives are
at `asin(3/8)`, approximately 22.0 degrees elevation, and at azimuths 60, 180,
and 300 degrees by default. They are separated by 120 degrees. All four
positions lie at one radius around the workpiece vertex centroid and point the
camera optical `+Z` axis at it. These nominal region representatives are always
tried first.

If a nominal position remains unreachable after the configured roll search,
the tool evaluates a fixed ordered set of alternative positions inside that
same region. Side alternatives stay strictly inside the original 120-degree
azimuth sector and the lower band of the upper hemisphere. Top alternatives
stay inside the top cap. They remain on the same framing radius, use the same
look-at target and safety validation, and prefer roll `0` before rotated
images. The selected strategy and fallback rank are saved with every capture. Pass
`--strict-region-representatives` to disable this search.

The radius starts at `0.40 m` and is increased when needed to frame the entire
workpiece bounding sphere with a 10% angular margin. A required radius above
`0.50 m` is rejected as outside the configured workspace.

## Reachability and motion sequence

For each run, the tool:

1. resolves `auto`, `pybullet`, or `curobo` exactly like the main Level 6 CLI;
2. checks all four positions with roll `0` and required line of sight;
3. retries only failed positions at rolls `90`, `-90`, and `180` degrees;
4. searches deterministic positions inside any still-missing region, first at
   roll `0` and then with roll redundancy;
5. stops before capture only if a region has no validated fallback endpoint;
6. plans from the current joints to each selected IK endpoint;
7. densely revalidates the returned trajectory in PyBullet at `0.005 m`
   clearance;
8. plays the path, captures RGB, and writes metadata;
9. returns home after the four captures.

CuRobo mode uses MotionGen. The PyBullet path uses joint-linear interpolation
with no joint changing by more than `0.02 rad` between samples. The word
"validated" refers to the simplified simulated geometry and discrete samples;
it is not a hardware or continuous-collision certificate.

If a direct move after the first capture fails, the tool returns home and
replans from the known home state. A direct final return is attempted first;
validated outward paths can be retraced as a fallback. The outer `finally`
block makes one more safe-return attempt and resets the simulation home if
that attempt fails.

## Run it

```bash
python3 sample/capture_four_views.py --headless
```

Automatic mode prefers the prepared V1 CuRobo runtime and falls back only for
typed runtime unavailability. Force CPU behavior with:

```bash
python3 sample/capture_four_views.py --headless --ik-backend pybullet
```

Require CuRobo without fallback with:

```bash
python3 sample/capture_four_views.py \
  --headless \
  --ik-backend curobo \
  --curobo-robot-config artifacts/curobo/ur5e_camera_curobo_v1.yml
```

See [CLI Reference](CLI-Reference) for every option.

## Output contract

The default directory is `output/pybullet/four_views`. Every image and JSON
sidecar has a UTC timestamp in its filename. PNG files also embed the capture
metadata as `capture_metadata_json`.

Each per-image record contains:

- region and image filename;
- UTC capture timestamp;
- six-joint order and positions in radians and degrees;
- nominal-representative or within-region-fallback selection strategy and rank;
- actual camera world position, XYZW orientation, and rotation matrix;
- requested radius, azimuth, elevation, roll, position, and orientation;
- validated endpoint residuals, clearance, seed, and solver provenance;
- executed planner identity, timing/status, waypoint count, PyBullet sample
  count, and minimum observed clearance.

The manifest is written before motion with `status: in_progress`, updated after
every image, and changed to `status: complete` only after the return home. This
allows the ranking notebook to ignore interrupted or partial runs. Its other
top-level fields include schema version `1`, start/completion timestamps,
object vertex centroid, shell radius, effective backend metadata, and the four
capture records.

## Important limitations

- The camera is simulated RGB only; there is no depth, distortion, exposure,
  rolling shutter, or sensor-noise model.
- A fallback camera roll rotates the resulting image even though position and
  viewing direction remain fixed.
- A within-region fallback preserves region membership but is not the nominal
  analytic representative shown by the black marker in the partition figure.
- The source gripper/cabling are omitted from collision geometry.
- Output metadata describes simulated state, not measured hardware state.
- Image ranking is a separate notebook step and never changes the capture run.

Next: [DINOv3 Matching and Ranking](DINOv3-Matching-and-Ranking),
[Level 7 CuRobo Motion Planning](Level-7-CuRobo-Motion-Planning), or
[Testing and Validation](Testing-and-Validation).
