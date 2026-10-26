# RGB-only initialization for viewpoint search

The optional RGB localization stage estimates an aiming point from multiple
calibrated camera observations before running the existing viewpoint search.
It replaces the simulator-derived object center used for search initialization.
The default remains `--object-localization simulator` for existing experiments.

An RGB camera alone does not provide metric depth from one image. This stage
matches features across translated camera observations and triangulates them
using the measured camera poses. Metric scale comes from robot forward
kinematics and calibrated hand-eye geometry, expressed in meters. It does not
run FoundationPose, fabricate depth, or infer a complete object pose in SE(3).

## Run in the simulator

```bash
python3 sample/active_perception_triangles.py \
  --object obj_000020 \
  --name Dog_rgb_localization \
  --target-image output/pybullet/target_views/dog.png \
  --object-localization rgb \
  --search-strategy geometric \
  --similarity-threshold 0.97 \
  --search-max-views 8 \
  --headless
```

RGB localization requires a target image. The `triangles`, `nbv`, and `geometric`
search strategies all reuse the estimated aiming point. `--object` still selects
the simulated scene asset; it is not a measured pose supplied to the localizer.

Use `--localization-viewpoints survey.json` to specify calibrated survey camera
poses. Survey poses describe workspace coverage, not an object center. Their
positions must provide overlapping target visibility and a lateral baseline.
For the live simulator, the file names the `pybullet_world` coordinate frame and
contains a `viewpoints` list with `position_xyz_m` and `orientation_xyzw` for each
camera pose. Quaternions use `(x, y, z, w)` ordering. Orientations describe the
optical camera frame; do not substitute a tool-link orientation.

The default survey uses three downward-looking optical camera poses at
`x = 0.36, 0.44, 0.52`, `y = -0.10`, `z = 1.30` meters in the simulator world.
These cover a fixed part of the table; they do not search an entire room. Supply
a workspace-specific survey if the target is outside their overlapping fields
of view. For example:

```json
{
  "coordinate_frame": "pybullet_world",
  "viewpoints": [
    {"position_xyz_m": [0.36, -0.10, 1.30], "orientation_xyzw": [1, 0, 0, 0]},
    {"position_xyz_m": [0.44, -0.10, 1.30], "orientation_xyzw": [1, 0, 0, 0]}
  ]
}
```

The integration acquires RGB and camera poses from the simulator and uses the
existing robot motion and reachability checks. It does not supply a real camera
driver, robot interface, or a perception-derived collision map. The collision
scene still uses simulator geometry, so this option is not a complete removal
of scene ground truth from the overall system.

## Process recorded RGB observations

Recorded observations let you test localization with a real RGB camera before
connecting the search loop to real robot motion:

```bash
python3 sample/rgb_object_localization.py \
  --observations observations.json \
  --target-image reference.png \
  --output-json localization.json
```

The manifest must name a common `coordinate_frame` and contain at least two
observations. Each observation supplies an RGB image, calibrated `intrinsics`,
and either a direct `world_from_camera` transform or both `world_from_tool` and
`tool_from_camera`. An optional `mask` supplies the target's binary foreground
mask for that observation; optional `distortion_coefficients` describe the lens.

Image and mask paths are relative to the manifest (absolute paths also work).
`intrinsics` is a 3×3 matrix, for example `[[500,0,320],[0,500,240],[0,0,1]]`
for a calibrated 640×480 image. A minimal manifest with illustrative calibration
values is below; replace them with your measured values:

```json
{
  "coordinate_frame": "robot_base",
  "observations": [
    {
      "image": "view_01.png",
      "intrinsics": [[500,0,320],[0,500,240],[0,0,1]],
      "world_from_camera": [[1,0,0,0.36],[0,-1,0,-0.10],[0,0,-1,0.55],[0,0,0,1]]
    },
    {
      "image": "view_02.png",
      "intrinsics": [[500,0,320],[0,500,240],[0,0,1]],
      "world_from_camera": [[1,0,0,0.44],[0,-1,0,-0.10],[0,0,-1,0.55],[0,0,0,1]]
    }
  ]
}
```

Optional `distortion_coefficients` use OpenCV's standard pinhole model ordering
(4, 5, 8, 12, or 14 coefficients), not a fisheye model. Omit them for already
rectified images and provide the corresponding rectified intrinsics.

Transforms are 4×4 homogeneous matrices acting on column vectors. In particular:

```text
world_from_camera = world_from_tool @ tool_from_camera
world_point = world_from_camera @ camera_point
```

The optical camera axes are **+X right, +Y down, +Z forward**. Translation units
are **meters**. Intrinsics refer to the actual saved image resolution, with
focal lengths and principal point in pixels. Use the camera pose measured at
image acquisition, not the commanded destination pose. Both observations must
use the same world coordinate frame and correct lens calibration.

## Target selection and measurement limits

DINOv3 compares reference object descriptors with candidate regions in each
survey image. Selection has no image-center preference. Low similarity or
similar-scoring distinct candidates causes localization to fail explicitly.
DINOv3 descriptors alone are not a general instance segmentation system.

The built-in RGB proposals assume separated objects against an approximately
uniform background visible at the image border. Textured tables, touching
objects, heavy shadows, occlusion, and objects resembling the background can
break that assumption. Offline supplied masks bypass that proposal stage;
produce those masks with an appropriate segmenter or annotate them for an
initial calibration test. Segmentation and DINO similarity thresholds must be
validated on the intended real scene.

Triangulation requires a static target with enough repeatable visual texture,
overlapping visibility, and camera translation that produces useful parallax.
Pure rotation, a very small baseline, featureless or repetitive surfaces,
moving objects, and inaccurate hand-eye calibration can prevent a reliable
estimate. A second view does not guarantee enough usable correspondences.

The output is a **`visible_surface_aim_point`**: a robust point derived from the
reconstructed visible target surface. It is not the object's geometric centroid,
bounding-box center, model origin, or full 6D pose. Partial visibility can bias
the point toward the observed surface; viewpoint search inherits that aiming
error. In particular, the geometric search strategy's orbit-center approximation
remains an approximation when initialized this way.

## Timing and evaluation

Measure localization and search separately, and include both in end-to-end
runtime. Initialization can include model loading, survey motion, image capture,
target selection, correspondence extraction, and triangulation. Cached model
startup and already-recorded observations do not represent a cold live run.

Compare the final similarity, localization error where ground truth is available
for evaluation, successful observations, and full wall time against the existing
simulator-center baseline. This feature does not establish a 10–15 second total
runtime or a real-world localization accuracy guarantee.

Live runs save `<name>_rgb_localization_observations.json` for replay and
`<name>_rgb_localization.json` with the estimated `center_world_m`, point cloud,
per-view selection scores, pairwise baseline/parallax/reprojection diagnostics,
and `bootstrap_seconds` including survey capture and model preparation.
`localization_seconds` measures detection, matching, and triangulation only;
the offline command additionally records `total_seconds` including model setup.
The search summary includes localization provenance and counts survey captures
in `total_images_captured`. Localization failure stops search without falling
back to the simulator center; captured observations remain available for replay.

## Validation performed

The tests were written before the corresponding implementations and exercised
through failing and passing runs. They cover off-center target selection and
distractors, ambiguity rejection, calibrated triangulation and invalid geometry,
lens distortion, hand-eye composition, measured rather than commanded camera
poses, failure cleanup, model reuse, and propagation through all three search
strategies without simulator-center/line-of-sight queries.

Two headless CuRobo smoke runs on the dog scene used the fixed default survey,
`--motion-seconds 0`, and a three-view search budget. They used cached model
weights; timings are local observations, not guarantees:

| Scene / search | Accepted pairwise points | RGB processing | Whole bootstrap | Best search score |
| --- | ---: | ---: | ---: | ---: |
| Default placement / Bayesian NBV | 105 | 0.53 s | 20.49 s | 0.660 |
| Object shifted +4 cm X, +4.5 cm Y / geometric | 122 | 0.49 s | 18.10 s | 0.692 |

The displaced scene was changed before the run, independently of the survey.
Both object-center getters and the object line-of-sight method were overridden
to raise if called during that run. Both searches completed their view budgets
and reported six captures (three survey plus three scored search images).
Neither reached the 0.97 similarity threshold within that budget. The two runs
used different search strategies and are smoke tests, not a controlled speed
or accuracy comparison. A 10–15 second end-to-end runtime remains unachieved.
