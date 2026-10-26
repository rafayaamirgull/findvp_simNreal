# Testing and Validation

## Host-safe test suite

Run from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Current result: 130 tests passing with both the command above and `pytest` on
2026-09-20.

The suite is deliberately import-safe on a host without CuRobo or CUDA. The
automatic-selection tests mock runtime outcomes; they do not claim a real GPU
execution.

## Coverage by test module

### `test_viewpoint_reachability.py`

- exact analytic top/side representatives for four equal-solid-angle upper-
  hemisphere regions;
- deterministic fallback candidates remain inside their assigned top cap or
  side sector and preserve the framing radius and look-at direction;
- fixed radius, look-at orientation, 120-degree side spacing, azimuth offset,
  and invalid four-view input handling;
- exactly 100 unique deterministic default samples;
- requested radius and elevation bounds;
- every camera optical `+Z` axis points at the target;
- roll preserves position and forward direction;
- invalid centers, counts, radii, and elevation bounds are rejected.

### `test_pybullet_reachability_integration.py`

- camera target to `tool0` target round trip against PyBullet FK;
- default home configuration passes the clearance gate;
- a fake external backend still passes through the real PyBullet validator;
- finite, reproducible JSON schema and metadata;
- non-finite safety thresholds are rejected;
- camera collision proxy is anchored behind the optical center;
- joint commands require exactly six finite values and valid blend;
- simultaneous environment instances have isolated PyBullet clients;
- automatic mode prefers V1 and selects the matching search settings;
- typed V1 runtime unavailability falls back to PyBullet search settings;
- explicit PyBullet avoids the CuRobo probe and explicit CuRobo stays strict;
- backend-selection provenance is serialized under `backend_metadata.selection`;
- filter calls restore their start state after success and backend errors;
- the live camera feed is enabled by default and the opt-out CLI flag works;
- camera captures expose RGB only;
- dense trajectory validation restores the entry joints;
- viewpoint projection/selection and path playback are deterministic.

### `test_capture_four_views.py`

- CuRobo-returned trajectories are revalidated in PyBullet before playback;
- planner and PyBullet sample/clearance evidence is recorded in metadata;
- a PyBullet trajectory rejection stops motion.

### `test_active_perception_triangles.py`

- equal-area 3/4/5-view hemisphere partitions and recursive triangle paths;
- nearby in-triangle recovery followed by per-triangle skipping when all
  fallback poses remain unreachable;
- motion-planning failure isolation and capture-only empty-round behavior;
- incumbent/look-ahead, radial, feature-size, return-home, and debug CLI
  defaults;
- target images enable ranking/refinement without requiring debug output;
- fixed-ray radial sample geometry keeps the camera aimed at the object center;
- symmetric closer/farther probes select an improving direction and stop after
  the configured score plateau;
- evaluation tracking accumulates camera/end-effector endpoint motion, target
  poses are matched from sidecar JSON, and reports persist the requested
  view/distance/time/PSNR/SSIM fields as per-run/latest JSON/text plus CSV; and
- numbered JSON/PLY capture-pose output.

### `test_visualize_end_effector_capture_poses.py`

- capture records require finite camera and end-effector poses;
- optical frustum geometry follows the configured camera aspect and +Z axis;
- every view receives a distinct color, while the summary-selected global best
  gets an enlarged bright-green frame and score label;
- compact frustum, tool-rectangle, and label defaults remain independently
  adjustable from the CLI;
- orbit-GIF export selects the largest matching PyBullet X11 window and exposes
  deterministic frame-count, frame-rate, orbit, and output-size defaults;
- sibling refinement-summary image paths resolve back to capture numbers;
- unobstructed transitions remain straight while intersecting transitions use
  a sampled great-circle arc outside the spherical object envelope; and
- sampled RGB pixels retain their colors on the oriented 3D image plane.

### `test_dinov3_feature_matching.py`

- DINOv3/timm preprocessing, aspect-preserving patch-grid coordinates, and
  small-image upscaling;
- white-background, ROI, and mask filtering with full-image coordinate mapping;
- mutual-nearest matching, match limits, ambiguity filters, and argument checks;
- deterministic fundamental-matrix RANSAC outlier removal;
- ZNCC behavior, near-constant rejection, translated local patches, and
  different image dimensions;
- combined descriptor/geometry/local-ZNCC score diagnostics.

All model-dependent tests inject small fake features/models. They do not
download or validate the production SAM or DINOv3 checkpoints.

### `test_dinov3_spatial_clustering.py`

- isolated automatic-mask proposal selection and boundary rejection;
- deterministic PCA/K-Means clustering and explicit cluster overrides;
- complete-grid and image-dimension contracts;
- directional, reverse, and symmetric object similarity.

### `test_curobo_reachability.py`

- V1/V2 YAML schema detection;
- host/runtime compatibility diagnostics;
- exact V1 sine-half-angle rotation conversion;
- goal-set packing in robot-base coordinates and WXYZ order;
- malformed CuRobo result shape and joint-order normalization;
- all optimized successful branches are returned;
- generated URDF topology and exact camera extrinsic;
- V1 IK-only base-schema safety flags;
- guarded V1 reference-sphere mapping and geometry-mismatch rejection;
- V2 scene pose and RobotBuilder finalization invariants;
- PyBullet imports do not eagerly load CuRobo;
- CuRobo preparation imports do not load PyBullet;
- each backend performs and caches its own successful full runtime validation;
- runtime guards cover minimum compute capability and reject a detected GPU
  when its real CUDA tensor smoke test fails;
- the V1 container retains CUDA plus NVIDIA graphics/display capabilities and
  installs the X11 renderer-diagnostic utilities required by GUI workflows.

## Static checks used during cleanup

```bash
python3 -m compileall -q sample tests
pyflakes sample tests
git diff --check
```

All completed successfully in the last validation pass. `pyflakes` is a
development dependency and may not be installed in every runtime image.

## Dockerfile checks

```bash
docker build --check -f Dockerfile .
docker build --check -f Dockerfile.curobo-v1 .
```

Both Dockerfiles passed this syntax/build-policy check. The V1 image was also
built and exercised on the target GPU as described below.

The existing pinned V1 image was additionally started with the corrected GUI
runtime flags (`--ipc=host` and CUDA/graphics/display capabilities) on
2026-09-20. A real `p.GUI` connection reported the NVIDIA RTX 3060 renderer,
created a direct GLX context, and disconnected cleanly without an MIT-SHM
error.

## Explicit PyBullet baseline regression

The fixed PyBullet baseline can be run without overwriting the normal report:

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --ik-backend pybullet \
  --viewpoint-output /tmp/findvp_level6_default.json
```

Last observed with the world-up roll-0 default on 2026-07-23:

| Result | Value |
| --- | ---: |
| Accepted | 52/100 |
| Rejected | 48/100 |
| Best viewpoint | 23 |
| Best roll | 0 degrees |
| Best quality | 0.671198 |

The automated suite does not currently assert the complete 100-view accepted
set as a golden test. PyBullet itself is unpinned in the main Dockerfile, so a
full numerical regression should be rerun after dependency or model changes.

## CuRobo preparation check

Host-side V1 generation can be verified without importing CuRobo:

```bash
python3 sample/prepare_curobo_robot.py \
  --api-version v1 \
  --output-urdf /tmp/ur5e_camera.urdf \
  --output-config /tmp/ur5e_camera_v1.yml
```

This validates pure model/config generation. It does not execute GPU IK.

## CuRobo GPU status

Completed on the RTX 3060 Laptop GPU (environment on 2026-07-21 and upright
fixture on 2026-07-23):

- built `findvp-curobo:0.7.8` from the verified v0.7.8 commit;
- passed `check_env` with driver `535.261.03`, Torch CUDA `12.2`, compute `8.6`,
  and `5.80 GiB` visible VRAM;
- regenerated the V1 URDF/YAML inside the `/workspace` mount;
- completed the real 100-viewpoint sequential V1 fixture with no fallback
  using the world-up roll-0 default;
- report identity was `curobo_v1_legacy_goalset_ik` plus
  `pybullet_endpoint_validation`;
- 64/100 endpoints were accepted and 36 rejected in about `10.85 s` end to
  end;
- best viewpoint was index 23, roll 0 degrees, quality about `0.671`.

An earlier explicit four-roll run (`--ik-rolls 0,90,-90,180`) accepted 88/100
endpoints and made 100 cached `B=1` goal-set solves in about `6.61 s` of solver
time (`10.05 s` end to end). That run deliberately allowed rotated camera
frames and is not the current default.

The report was saved as
`output/pybullet/reachable_viewpoints_curobo_v1.json`. This result is execution
evidence, not a fixed regression target.
Still outstanding are a real V2 `solve_pose`, visual MorphIt sphere-quality
review, and cross-backend parity analysis.

At the 2026-07-31 documentation pass, the ignored
`reachable_viewpoints.json` reports `curobo_v1_legacy_goalset_ik`, 64 accepted,
and effective CuRobo selection with no automatic fallback. The older ignored
`reachable_viewpoints_curobo_v1.json` is misleadingly named: its contents
report `pybullet_multistart_ik`, four rolls, and 84 accepted. Report contents,
not filenames, are the source of truth.

## Known test gaps

- No GUI/OpenGL image regression.
- The RGB-only camera interface is covered for image shape and data type; there
  is no calibrated colour or photometric regression.
- No real camera noise/distortion model.
- No gripper or cable collision model.
- No exact continuous collision test between the Level 7 dense validation samples.
- No real CuRobo GPU Level 7 trajectory run in automated CI; adapter contracts,
  trajectory normalization, PyBullet revalidation, projection, and playback are
  covered host-side.
- No dynamics, control, torque, or hardware execution test.
- No full 100-point golden test in automated CI.
- No real CuRobo GPU execution in the host-safe automated suite; the V1 result
  above is a separate machine-specific integration run.
- No automated real-checkpoint SAM/DINOv3 download or GPU inference test.
- No golden mask, match set, or ranking for the repository RGB images.
- No automated end-to-end angular-plus-radial active-perception run with the
  real CuRobo GPU planner and DINOv3 checkpoint; search control and report
  contracts are covered with host-safe fakes.
- No end-to-end automated four-image capture run; analytic sampling and
  trajectory contracts are covered separately.
- No calibrated threshold proving that a combined DINO/ZNCC score corresponds
  to object identity or pose quality.

Next: [Troubleshooting](Troubleshooting) or
[JSON Report Reference](JSON-Report-Reference).
