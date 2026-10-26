# CLI Reference

## Main entry point

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py [OPTIONS]
```

## Scene and camera options

| Option | Default | Meaning |
| --- | --- | --- |
| `--robot-urdf PATH` | `assets/steve_description/urdf/steve_ur5e.urdf` | Source robot URDF to reduce for PyBullet |
| `--object NAME` | `obj_000020` | PLY model name from `tasks/3d_models/`; the `.ply` suffix is optional |
| `--workpiece-ply PATH` | none | Explicit external PLY path; mutually exclusive with `--object` |
| `--workpiece-unit-scale FLOAT` | `0.001` | Source-coordinate to metre scale |
| `--workpiece-rotation-z DEG` | `0.0` | Z rotation baked into the runtime mesh |
| `--headless` | off | Use PyBullet DIRECT mode |
| `--show-camera-feed` | on | Open a live Matplotlib RGB preview in GUI mode |
| `--no-camera-feed` | off | Disable the live Matplotlib RGB preview |
| `--show-capability-map` | on | Load or generate and display the capability map in GUI mode |
| `--no-capability-map` | off | Disable the capability-map overlay |
| `--capability-map PATH` | `output/ur5e_capability_map.npz` | Capability-map cache to load or create |
| `--capability-samples N` | `100000` | Joint samples used only when the selected cache is missing |
| `--save-camera-path PATH` | `output/pybullet/wrist_camera_latest.png` | RGB capture path/prefix |
| `--camera-update-hz FLOAT` | `5.0` | GUI preview rate |

## Reachability and sampling options

| Option | Default | Meaning |
| --- | --- | --- |
| `--plan-viewpoints` | off | Run the Level 6 fixture and filter viewpoints |
| `--ik-backend {auto,pybullet,curobo}` | `auto` | Auto prefers pinned CuRobo V1 and uses PyBullet when its runtime is unavailable |
| `--curobo-robot-config PATH` | none | Auto uses `artifacts/curobo/ur5e_camera_curobo_v1.yml` when omitted; explicit CuRobo requires a path |
| `--curobo-seeds N` | `32` | GPU branches/seeds shared across roll goals |
| `--viewpoint-count N` | `100` | Number of deterministic fixture positions |
| `--view-radius-min M` | `0.30` | Requested minimum shell radius |
| `--view-radius-max M` | `0.50` | Maximum shell radius |
| `--view-elevation-min DEG` | `12.0` | Minimum fixture elevation |
| `--view-elevation-max DEG` | `82.0` | Maximum fixture elevation |
| `--ik-rolls LIST` | `0` | Absolute optical-axis rolls; nonzero alternatives increase reachability but rotate camera images |
| `--ik-seeds N` | `3` | PyBullet seeds per roll |
| `--ik-refinement-steps N` | `1` | PyBullet warm-start polishing passes |
| `--ik-position-tolerance M` | `0.001` | Maximum PyBullet FK camera-position error |
| `--ik-orientation-tolerance DEG` | `1.0` | Maximum PyBullet FK camera-angle error |
| `--collision-clearance M` | `0.005` | Required endpoint clearance |
| `--skip-line-of-sight-check` | off | Disable object-center ray rejection |
| `--path-validation-steps N` | `0` | Discrete straight-joint path diagnostic |
| `--motion-interpolation-dt SEC` | `0.02` | Dense CuRobo timestep for an interactive selected-point trajectory |
| `--motion-plan-attempts N` | `5` | Maximum CuRobo planning attempts per selected point |
| `--viewpoint-output PATH` | `output/pybullet/reachable_viewpoints.json` | JSON report path |
| `--visualize-viewpoints` | off | GUI debug points for accepted/rejected positions |

Backend selection is resolved before the reachability configuration is built.
When CuRobo is effective, `ReachabilityConfig.seed_count` comes from
`--curobo-seeds` and refinement is zero. When automatic selection falls back,
the effective PyBullet configuration uses `--ik-seeds` and
`--ik-refinement-steps`.

The default roll is only `0` so the camera's image vertical remains aligned
with world up. Pass `--ik-rolls 0,90,-90,180` only when the extra reachability
is worth allowing physically rotated camera frames.

### Backend policy and failure boundary

- `auto` performs the full pinned-V1 package, driver, Torch/CUDA, CUDA device,
  GPU properties, and tensor-execution preflight.
- A typed runtime-unavailability result prints its reason and selects PyBullet.
- If preflight succeeds, a missing config, wrong schema, construction error,
  solver error, malformed result, or validator error is a hard failure.
- `curobo` is strict, requires `--curobo-robot-config`, auto-detects V1/V2 from
  that YAML, and never falls back.
- `pybullet` does not probe the CuRobo runtime.

## Active-perception entry point

```bash
python3 sample/active_perception_triangles.py [OPTIONS]
```

Without `--target-image`, this command captures one coarse angular round only.
With a target image, it runs incumbent-preserving angular refinement followed
by closer/farther radial fine alignment. It shares the main environment's
`--object`/`--workpiece-ply`, scale, rotation, headless, robot URDF, and IK
backend conventions.

### Capture and angular-search options

| Option | Default | Meaning |
| --- | --- | --- |
| `--views {3,4,5}` | `3` | Equal-area spherical triangles evaluated per angular level |
| `--radius M` | `0.40` | Angular-search camera radius in metres |
| `--azimuth-offset DEG` | automatic | Force a partition rotation; automatic mode tests up to 12 offsets and keeps the most reachable |
| `--name TEXT` | `object` | Prefix for images and JSON/PLY reports |
| `--output-dir PATH` | `output/pybullet/triangle_views` | Active-perception output directory |
| `--method-name TEXT` | `Adaptive Spherical + Radial (DINOv3+SIFT)` | Method label stored in evaluation reports and used to group aggregate trials |
| `--motion-seconds SEC` | `1.5` | Playback duration for each planned path; `0.1` is useful for fast simulation runs |
| `--return-home-each-level` | off | Restore the legacy home traversal after each angular level; otherwise return home only during final cleanup |
| `--target-image PATH` | none | Enable target-guided angular ranking and radial fine alignment |
| `--target-pose-json PATH` | automatic sibling search | JSON containing a matching target-image record and camera pose; mutually exclusive with `--target-camera-position` |
| `--target-camera-position X Y Z` | none | Ground-truth target camera center in PyBullet-world metres; mutually exclusive with `--target-pose-json` |
| `--similarity-threshold FLOAT` | `0.90` | Stop angular search when the global best combined score reaches this value |
| `--max-refinement-depth N` | `3` | Maximum subdivisions after the initial level; level 0 is not counted as a subdivision |
| `--refinement-patience N` | `2` | Stop after this many consecutive levels lack a meaningful global-best gain; `0` disables this stop |
| `--min-score-improvement FLOAT` | `0.002` | Score gain required to reset angular patience |
| `--angular-lookahead-margin FLOAT` | `0.03` | Largest gap below the incumbent for which a non-improving child can receive look-ahead |
| `--angular-lookahead-levels N` | `1` | Maximum ambiguous non-improving levels explored before angular termination |
| `--feature-max-image-size PX` | `448` | Maximum DINOv3 image side; `640` restores the previous slower setting |
| `--ambiguity-margin FLOAT` | `0.03` | Score interval retained as plausible triangles in each ranking report |
| `--dino-weight FLOAT` | `0.75` | DINO contribution to the combined score |
| `--sift-weight FLOAT` | `0.25` | SIFT contribution to the combined score |
| `--debug` | off | Add spatial clustering and rendered diagnostics; it is not required for ranking or refinement |

For every rejected triangle centroid, the command tries nearby in-triangle
viewpoints before marking only that triangle as skipped. A round continues as
long as at least one candidate remains reachable.

### Radial fine-alignment options

| Option | Default | Meaning |
| --- | --- | --- |
| `--skip-radial-refinement` | off | End after angular search |
| `--radial-step M` | `0.02` | Signed radius increment along the angular winner's actual object-center ray |
| `--radial-probe-steps N` | `2` | Initial steps captured both inside (closer) and outside (farther) |
| `--radial-max-steps N` | `10` | Maximum absolute step followed in the selected direction |
| `--radial-patience N` | `2` | Stop after this many continuation steps without a meaningful gain |
| `--radial-min-improvement FLOAT` | `0.002` | Gain required to select/continue a radial direction |
| `--radial-min-radius M` | `0.20` | Minimum permitted object-to-camera radius |
| `--radial-max-radius M` | `0.60` | Maximum permitted object-to-camera radius |

The initial inside/outside candidates are reachability-filtered together, then
captured as separate directional sweeps. The camera orientation is recomputed
at every radius so its optical axis continues to point at the object center.

### Evaluation output

After a successful capture-only or target-guided run, the entry point writes:

| Output | Purpose |
| --- | --- |
| `<name>_evaluation_<UTC_RUN_ID>.json` | Immutable machine-readable report for one run |
| `<name>_evaluation_<UTC_RUN_ID>.txt` | Immutable readable report for the same run |
| `<name>_evaluation.json` / `.txt` | Stable latest-run copies for scripts and direct inspection |
| `<name>_evaluation_runs.csv` | Append-only rows used for per-method mean ± sample-standard-deviation aggregation |

The matrix fields are measured as follows:

- `n_views` is the number of successfully saved angular plus radial images;
- `d_target` is the Euclidean distance between the selected final camera center
  and the ground-truth target camera center, in metres. It is `N/A` when target
  pose ground truth cannot be resolved;
- `d_traveled` is the sum of camera-center displacement between executed motion
  endpoints through search termination, including the radial final-best revisit
  when performed and search-time recovery/home motions, but excluding final
  cleanup. JSON stores metres and centimetres; the text report and aggregate
  table show centimetres;
- `t_exec` starts after construction of the PyBullet environment and includes
  backend construction/warm-up, reachability, IK, planning, playback, capture,
  ranking, angular refinement, radial refinement, and final-best revisit. It
  excludes evaluation serialization, refinement-summary serialization, final
  return home, and disconnect;
- `PSNR` and `SSIM` compare the target and selected final full-color images with
  pixel range 255. A differently sized candidate is resized to the target size
  and that fact is recorded. Identical images store `"inf"` for PSNR.

`--target-camera-position` supplies the target center directly. With the
mutually exclusive `--target-pose-json`, or during the default sibling-directory
search, a nested JSON record must match the target by `image_file` or
`image_filename` and expose a supported camera pose. A minimal sidecar is:

```json
{
  "image_filename": "wrist_camera_latest_1788213774.png",
  "camera_pose_world": {
    "position_xyz_m": [0.45, -0.10, 1.20],
    "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]
  }
}
```

When both target and final orientations are available, the JSON/text reports
also include quaternion angular error. The JSON retains the camera and
end-effector motion-segment ledger and the current aggregate for the exact
`--method-name` label. Non-finite values such as identical-image PSNR are
omitted from aggregate means while remaining present in their per-run report.
Use the same `d_target`, travel, timing, and image-domain definitions as the
competing methods before placing values in one comparison table.

### Current Power Drill command

```bash
python3 sample/active_perception_triangles.py \
  --views 3 \
  --object "Power Drill" \
  --name Drill \
  --target-image output/pybullet/wrist_camera_latest_1788213774.png \
  --similarity-threshold 0.97 \
  --max-refinement-depth 7 \
  --motion-seconds 0.1 \
  --feature-max-image-size 448 \
  --angular-lookahead-margin 0.03 \
  --angular-lookahead-levels 1 \
  --radial-step 0.02 \
  --radial-probe-steps 2 \
  --radial-max-steps 10 \
  --radial-patience 2 \
  --radial-min-radius 0.20 \
  --radial-max-radius 0.60
```

## Capture-trajectory visualization

```bash
python3 sample/visualize_end_effector_capture_poses.py \
  --captures output/pybullet/triangle_views/object_end_effector_capture_poses.json
```

The default presentation follows the supplied PCL/C++ technique while making
dense view sets easier to read: every optical camera frustum has a distinct
high-contrast color, saved RGB pixels are projected onto each image plane, and
a purple polyline follows the ordered camera centers. When a straight
transition crosses the padded workpiece bounding sphere, the line switches to
a sampled great-circle surface arc and then returns to the destination. This is
an idealized view-space path, not the executed robot joint trajectory.

The visualizer automatically finds `<name>_refinement_summary.json` beside the
capture report, maps `final_best.best_image` back to its capture, and uses
`final_best.best_score` in the label. The winner receives a 15% larger frustum,
2.5x line width, bright-green marker sphere and object-pointing ray, and an
explicit `BEST VIEW | Picture N | score=...` label. The optional tool rectangle
is enlarged and recolored the same way.

| Option | Default | Meaning |
| --- | --- | --- |
| `--captures PATH` | required | Active-perception capture-pose JSON containing camera poses, joints, and image filenames |
| `--refinement-summary PATH` | sibling auto-discovery | Explicit summary JSON used to resolve the global-best image and score |
| `--best-picture N` | summary winner | Manually select the one-based capture number to highlight |
| `--frustum-depth M` | `0.08` | Distance from optical center to the wireframe image plane |
| `--image-depth M` | `0.079` | Distance to the projected RGB point plane; slightly in front of the wireframe plane |
| `--image-sample-step PX` | `4` | Pixel stride used to build each colored point cloud; larger values render fewer points |
| `--image-point-size FLOAT` | `2.0` | PyBullet debug-point size for projected pixels |
| `--path-step M` | `0.01` | Maximum sampling interval on straight and spherical path portions |
| `--path-min-height M` | `0.05` | Prefer the alternate sphere arc when the short arc drops below this world-Z height |
| `--obstacle-padding M` | `0.02` | Clearance added to the mesh bounding radius |
| `--obstacle-radius M` | automatic | Replace the derived avoidance radius |
| `--best-marker-radius M` | `0.012` | Radius of the bright-green sphere at the best camera center |
| `--label-size FLOAT` | `0.75` | PyBullet text size for ordinary view labels |
| `--best-label-size FLOAT` | `1.05` | PyBullet text size for the best-view label |
| `--best-view-scale FLOAT` | `1.15` | Size multiplier for the best frustum and optional tool rectangle |
| `--show-obstacle-sphere` | off | Draw the translucent avoidance envelope |
| `--flip-images` | off | Rotate image planes by 180 degrees, matching the explicit flip in the C++ example |
| `--no-frustums`, `--no-images`, `--no-path`, `--no-labels` | off | Suppress individual presentation layers |
| `--viewer-distance`, `--viewer-yaw`, `--viewer-pitch` | `1.4`, `0`, `-35` | Overview-camera framing around the object centroid |
| `--gif-output PATH` | off | Record a looping orbit of the actual X11 PyBullet viewport, save it as a GIF, and exit |
| `--gif-frames N`, `--gif-fps FLOAT` | `48`, `12` | Output frame count and playback rate |
| `--gif-orbit-degrees DEG` | `360` | Viewer-camera rotation covered by the GIF |
| `--gif-width PX` | `800` | Output width; height preserves the viewport aspect ratio |
| `--cycle-robot` | off | Replay the saved discrete joint states after drawing the static visualization |
| `--show-tool-rectangles` | off | Add the previous numbered `tool0` rectangle overlay |
| `--rectangle-width`, `--rectangle-height` | `0.045`, `0.034` | Dimensions in metres of each optional tool rectangle |
| `--rectangle-thickness M` | `0.001` | Thickness in metres of each optional tool rectangle |
| `--hold-seconds SEC` | `0` | GUI lifetime; zero waits until the window closes or Ctrl+C |

GIF export uses Pillow screen capture because PyBullet's off-screen camera does
not render user-debug frustums, point clouds, lines, or text. It therefore
requires an X11 desktop, the `xwininfo` utility, and an unobscured PyBullet
window. The control panels are hidden automatically. Combine `--gif-output`
with a positive `--hold-seconds` only when the GUI should remain open after the
GIF is written; otherwise the exporter exits immediately.

The image resolver first tries each saved `image_file`, then falls back to
`image_filename` beside the capture JSON. This makes reports produced under the
container's `/workspace` mount portable when the repository is viewed from the
host checkout. Use `--robot-urdf` and `--workpiece-ply` similarly when metadata
contains container-absolute asset paths.

## Argument validation

- Unit scale and camera update rate must be finite and positive.
- Capability-map sample count must be positive.
- Workpiece rotation must be finite.
- Viewpoint count must be positive.
- Radii must satisfy `0 < minimum <= maximum`.
- Elevations must satisfy `-90 < minimum <= maximum < 90 degrees`.
- Roll list must be non-empty and finite.
- Seed counts must be positive integers.
- Refinement and path sample counts must be non-negative integers.
- Motion planning attempts must be positive, and interpolation dt must be
  finite and positive.
- Position tolerance must be finite and positive.
- Orientation tolerance must be in `(0, 180] degrees`.
- Clearance must be finite and non-negative.
- Explicit CuRobo selection requires an existing, generation-detectable YAML.
- Automatic selection requires the generated V1 YAML only after V1 runtime
  preflight succeeds; a runtime fallback does not require that artifact.

The active-perception entry point additionally requires:

- similarity thresholds, ambiguity/look-ahead margins, and minimum score gains
  in `[0, 1]`;
- non-negative angular depth, patience, and look-ahead counts;
- a finite positive radial step and positive probe/patience counts;
- `--radial-max-steps >= --radial-probe-steps`;
- finite ordered radial bounds with `0 < minimum < maximum`;
- a positive feature-image size; and
- an existing target-image file whenever `--target-image` is supplied.

## Common commands

### GUI workcell

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py
```

### Select a bundled workpiece

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py --object "Christmas Bear"
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py --object "Power Drill"
```

The selection is independent of `--ik-backend`, so the same object is used by
automatic, PyBullet, and CuRobo IK. Quote names containing spaces.

### GUI without the default live camera preview

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py --no-camera-feed
```

### Capability-map visualization

The normal GUI command shows the map automatically. Workspace color represents
manipulability, workspace point size represents reachability, and magenta/pink
tabletop points represent inverse reachability/base placeability.

```bash
# Hide the overlay.
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py --no-capability-map

# Rebuild a denser custom cache on its first use.
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --capability-map output/ur5e_capability_map_250k.npz \
  --capability-samples 250000
```

See [PyBullet Environment](PyBullet-Environment#capability-map-overlay) for the
complete color, size, frame, and placeability interpretation.

### GUI viewpoint selection and CuRobo motion

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --plan-viewpoints \
  --ik-backend curobo \
  --curobo-robot-config artifacts/curobo/ur5e_camera_curobo_v2.yml
```

The GUI displays all Level 6 points, accepts viewport clicks, and adds **Move
to selected viewpoint (CuRobo)** to the Params panel. A collision-aware robot
config with reviewed spheres is required. Default V1 preparation maps the
pinned package's verified UR5e spheres; `--v1-ik-only` is not accepted for
motion planning.

### Headless camera snapshot

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --save-camera-path output/pybullet/wrist_camera_latest.png
```

### Default Level 6 fixture: prefer CuRobo V1

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints
```

This uses CuRobo V1 when its pinned runtime and generated V1 configuration are
usable. A typed runtime-unavailability result selects PyBullet and is recorded
in the report.

### Force the PyBullet backend

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --ik-backend pybullet
```

### Visualize fixture acceptance

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --plan-viewpoints \
  --visualize-viewpoints
```

### Add the straight-path diagnostic

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --path-validation-steps 50
```

This does not reject endpoint-reachable views, but it can change which valid
branch is selected for a viewpoint.

### Increase fixture search effort

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --ik-backend pybullet \
  --ik-rolls 0,45,-45,90,-90,135,-135,180 \
  --ik-seeds 6 \
  --view-radius-min 0.35 \
  --view-radius-max 0.55 \
  --viewpoint-output output/pybullet/reachable_viewpoints_custom.json
```

### Require CuRobo V1 without fallback

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --ik-backend curobo \
  --curobo-robot-config artifacts/curobo/ur5e_camera_curobo_v1.yml \
  --viewpoint-output output/pybullet/reachable_viewpoints_curobo_v1.json
```

## CuRobo preparation entry point

```bash
python3 sample/prepare_curobo_robot.py [OPTIONS]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--api-version {v1,v2}` | `v1` | Select mutually incompatible config generation |
| `--output-urdf PATH` | `artifacts/curobo/ur5e_camera_curobo.urdf` | Camera-aware URDF path |
| `--output-config PATH` | generation-specific | YAML path |
| `--run-builder` | off | V2 only: execute MorphIt RobotBuilder |
| `--visualize` | off | V2 only: inspect fitted collision spheres |
| `--v1-ik-only` | off | V1 only: retain the legacy sphere-free config; Level 7 rejects it |

V1 generation does not require importing PyBullet or executing a CUDA tensor,
but its default sphere mapping does require the pinned CuRobo v0.7.8 package
and bundled UR5e reference files. V2 with `--run-builder` requires a compatible
CuRobo V2/CUDA runtime.

## Four-view capture entry point

```bash
python3 sample/capture_four_views.py [OPTIONS]
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--headless` | off | Use PyBullet DIRECT mode; motion playback delays are skipped |
| `--output-dir PATH` | `output/pybullet/four_views` | Timestamped PNGs, per-image JSON, and run manifest directory |
| `--motion-seconds SEC` | `1.5` | GUI playback duration for each move; must be finite and non-negative |
| `--robot-urdf PATH` | normal workcell default | Source robot URDF |
| `--object NAME` | `obj_000020` | PLY model name from `tasks/3d_models/` |
| `--workpiece-ply PATH` | none | Explicit external PLY path; mutually exclusive with `--object` |
| `--workpiece-unit-scale FLOAT` | `0.001` | Source-coordinate to metre scale |
| `--workpiece-rotation-z DEG` | `0.0` | Workpiece Z rotation |
| `--strict-region-representatives` | off | Disable within-region position fallback and require the four nominal representatives |
| `--ik-backend {auto,pybullet,curobo}` | `auto` | Same backend selection policy as the main entry point |
| `--curobo-robot-config PATH` | none | Camera-aware YAML; required by strict CuRobo mode |
| `--curobo-seeds N` | `32` | Positive CuRobo seed count per viewpoint |

The tool chooses one radius that fits the workpiece bounding sphere in the
camera frustum, capped at `0.50 m`. It tests roll `0` first, then retries only
unreachable nominal positions with `90`, `-90`, and `180` degrees. If a region
is still missing, it searches a fixed, reproducible set of positions inside
that same top cap or side sector, preferring roll `0`. Pass
`--strict-region-representatives` to retain nominal-position-only behavior.

With CuRobo selected, MotionGen plans collision-aware motion. With PyBullet
selected or used by automatic fallback, a dense joint-linear path is produced.
Both paths must pass the same PyBullet trajectory validator before playback.
The tool returns home directly when possible and otherwise retraces already
validated paths; its `finally` block attempts a safe return even after errors.

Examples:

```bash
# Automatic backend selection, headless capture.
python3 sample/capture_four_views.py --headless

# Require CuRobo V1 and write a separate run directory.
python3 sample/capture_four_views.py \
  --headless \
  --ik-backend curobo \
  --curobo-robot-config artifacts/curobo/ur5e_camera_curobo_v1.yml \
  --output-dir output/pybullet/four_views_v1

# Force CPU/PyBullet reachability and validated joint-linear motion.
python3 sample/capture_four_views.py --headless --ik-backend pybullet
```

There is no standalone DINOv3 CLI. Use the Python API or run
`notebooks/dinov3_feature_matching.ipynb` as documented in
[DINOv3 Matching and Ranking](DINOv3-Matching-and-Ranking).

Next: [JSON Report Reference](JSON-Report-Reference) or
[Troubleshooting](Troubleshooting).
