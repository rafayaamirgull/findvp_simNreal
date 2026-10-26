# CuRobo Reachability Backends

## Supported paths

The repository has two deliberately separate CuRobo adapters behind the same
PyBullet endpoint validator:

| Path | Package and environment | CuRobo candidate generation | Final acceptance |
| --- | --- | --- | --- |
| **V1 (primary for this machine)** | exactly `nvidia-curobo==0.7.8`, PyTorch CUDA 12.2, NVIDIA driver `>=535.54.03` | persistent, sequential `B=1` `IKSolver.solve_goalset`; IK only | PyBullet |
| **V2 (secondary)** | `nvidia-curobo==0.8.x`, Python `>=3.10`, NVIDIA driver `>=580.65.06` | batched `InverseKinematics.solve_pose`; fitted robot spheres plus scene collisions | PyBullet revalidation |

The YAML schema selects the adapter automatically. A V1 file has a top-level
`robot_cfg` mapping; a V2 file has a top-level `kinematics` mapping. The runtime
then requires a matching supported CuRobo package. Do not mix a V1 YAML with a
V2 package or the other way around.

The CLI defaults to `--ik-backend auto`. This policy performs the complete V1
runtime preflight and prefers
`artifacts/curobo/ur5e_camera_curobo_v1.yml`. A typed
`CuRoboUnavailableError` caused by an unavailable or incompatible package,
driver, Torch/CUDA stack, CUDA device, GPU capability/memory check, or CUDA
tensor smoke test produces a visible warning and selects PyBullet multi-start
IK. It is the only automatic fallback path.

Fallback is not a general exception handler. Once V1 runtime preflight passes,
a missing or malformed YAML, a non-V1 schema in automatic mode, backend
construction failure, solver failure, malformed result, or PyBullet validation
error stops the command. Explicit `--ik-backend curobo` is strict and supports
the generation selected by its YAML; explicit `--ik-backend pybullet` performs
no CuRobo runtime probe. Torch and CuRobo imports remain lazy, so the PyBullet
path is import-safe on a machine without CuRobo or CUDA.

## License warning for v0.7.8

Read the [CuRobo v0.7.8 license](https://github.com/NVlabs/curobo/blob/v0.7.8/LICENSE)
before building the primary image. That release uses NVIDIA's license, whose
use limitation permits the work and derivative works only non-commercially,
defined there as research or evaluation. Redistribution also requires the
license and existing notices to be retained. Do not publish the resulting
image or use this Python package commercially without confirming that you have
the necessary rights or a separate license from NVIDIA. This paragraph is a
project warning, not legal advice.

The Dockerfile keeps the pinned source and license in `/opt/curobo` inside the
image. It clones tag `v0.7.8` and verifies commit
`d64c4b005459db10c5dd867d8b30a87d5bda9bdb` before installation.

## Primary v0.7.8 / R535 Docker workflow

The reported RTX 3060 Laptop GPU has compute capability 8.6 and 6 GiB VRAM, and
driver `535.261.03` is newer than the V1 minimum `535.54.03`. The provided
image is intentionally compiled for this class of GPU with
`TORCH_CUDA_ARCH_LIST=8.6+PTX`. Its `check_env` command requires:

- exactly `nvidia-curobo==0.7.8`
- a PyTorch CUDA 12.2 build
- driver `>=535.54.03`
- working container GPU passthrough
- compute capability `>=8.6` for this particular image
- at least 4 GiB GPU memory

The V1 adapter deliberately matches this image and rejects compute capability
below 8.6 before constructing a solver. Rebuild and validate both the image and
the adapter guard before targeting another GPU architecture. Docker, the NVIDIA
Container Toolkit, and network access to the NGC base image and the CuRobo
repository are required for the build.

From the repository root, build and start the pinned image:

```bash
docker build -f Dockerfile.curobo-v1 -t findvp-curobo:0.7.8 .

xhost +SI:localuser:root

docker run -it --rm \
  --name findvp-curobo \
  --gpus all \
  --ipc=host \
  --network=host \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/workspace \
  findvp-curobo:0.7.8
```

For headless-only use, the display, X11 socket, network, and host-IPC options
may be omitted.

This uses `nvcr.io/nvidia/pytorch:23.08-py3`, so the bare host's Python,
PyTorch, CUDA toolkit, and NVCC versions are not used. The host driver is passed
through to the container and is not changed.

Inside the container, verify the runtime before generating artifacts:

```bash
check_env
cd /workspace
python3 sample/prepare_curobo_robot.py --api-version v1
```

The preparation command writes exactly:

```text
artifacts/curobo/ur5e_camera_curobo.urdf
artifacts/curobo/ur5e_camera_curobo_v1.yml
```

The V1 YAML is generated directly; `--run-builder` and `--visualize` are V2-only
options. Both generated files contain absolute paths. Always regenerate them
inside the same `/workspace` mount after moving the checkout or changing its
container mount point. In particular, a YAML previously generated on the bare
host will contain host paths and should not be reused in the container.

Run the deterministic 100-position Level 6 fixture with the default automatic
V1-preferred policy:

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints
```

When the V1 runtime is usable, omitting `--curobo-robot-config` resolves to the
generated V1 path shown above. To require CuRobo and fail instead of falling
back when the runtime is unavailable, run:

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --ik-backend curobo \
  --curobo-robot-config artifacts/curobo/ur5e_camera_curobo_v1.yml
```

The default report path is:

```text
output/pybullet/reachable_viewpoints.json
```

The command overwrites that report. Use, for example,
`--viewpoint-output output/pybullet/reachable_viewpoints_curobo_v1.json` when a
separate GPU result should be preserved.

## What the V1 solver actually does

The 100-point Fibonacci shell is a deterministic Level 6 fixture, not the
production solver shape. A production caller may submit one externally
generated viewpoint. The V1 adapter uses the same path in both cases:

```text
one viewpoint position (B=1)
        x one goal set of G optical-axis rolls
        x K GPU seeds shared across the whole goal set
        -> solve_goalset
        -> up to K returned IK branches
        -> PyBullet endpoint validation
```

With defaults, `G=1` for roll `0` degrees and `K=32`, keeping camera frames
world-up. Passing `--ik-rolls 0,90,-90,180` restores the former four-goal
search, but a selected nonzero goal physically rotates the captured image.
`--curobo-seeds` changes both the optimized seed count and the maximum returned
branch count. The 32 seeds are shared across the goal set, not multiplied by
the roll count. More than one returned branch may select the same roll, and
only branches marked successful by CuRobo become candidates.

The V1 adapter creates a CUDA-graph solver for the fixed signature
`(G, K, position tolerance, rotation threshold)` and caches it on the backend
instance. It reuses that solver until the signature changes, serializes access
because CuRobo mutates cached graph/seed buffers, and resets the Halton seed
generator before every solve so identical sequential inputs remain
reproducible.

Even when the fixture submits all 100 `IKProblem` objects together, the adapter
loops over them and makes 100 sequential calls to `solve_goalset`, each with
`B=1`. It does not turn the fixture into a `B=100` V1 solve. One persistent
solver therefore exercises the production-shaped path while avoiding graph
initialization for every point. V1 result arrays are validated before use:
success and error fields must be `[1,K]`, joint solutions must be `[1,K,D]`,
and the V1 goal-set index must be `[1,K]`. The common result normalizer also
handles V2's trailing link dimension, `[B,K,1]`.

All successful optimized branches are returned for PyBullet revalidation. The
report records the selected `goalset_index` and `returned_branch_index`.
`seed_index` remains `null` because CuRobo's internal flattened/debug seed index
is not treated as a stable public identifier.

## Exact V1 orientation metric

CuRobo v0.7.8 does not express its IK rotation threshold or returned rotation
residual in radians. If `theta` is the full shortest quaternion angle in
`[0, pi]`, its non-negative residual is the norm of the relative quaternion's
vector part:

```text
m = sin(theta / 2)
```

The adapter therefore converts a requested tolerance in degrees with:

```text
m_threshold = sin(radians(theta_tolerance_deg) / 2)
```

For the default `--ik-orientation-tolerance 1.0`, the value passed to V1 is
`sin(0.5 degrees) = 0.008726535498373935`, not `radians(1.0)`. A returned V1
residual is exported in provenance both as the raw `sin_half_angle` metric and
as a full angle using:

```text
theta_deg = degrees(2 * asin(min(1, m)))
```

PyBullet independently recomputes the camera FK orientation error in degrees
before acceptance. Solver convergence alone never bypasses that shared
tolerance check.

## Frames and target packing

Repository, PyBullet, and JSON quaternions use `xyzw`; both CuRobo adapters
receive pose quaternions in `wxyz`. Component reordering happens only after the
pose has been transformed into the robot base frame:

```text
T_base_camera = inverse(T_world_base) @ T_world_camera
```

For this scene, `T_world_base` has identity rotation and translation
`[0, 0, 0.75]` metres. The generated robot has
`ur5e_camera_optical_frame` as its end-effector/tool frame, so the sampled
optical-camera pose is submitted directly. The adapter does not relabel a
`tool0` pose as a camera pose.

The generated CuRobo URDF has `ur5e_base_link` as its only root, retains the six
prefixed UR5e revolute joints, attaches the optical frame and a
`0.042 x 0.042 x 0.023 m` camera-housing box, removes fixed base collision
geometry at the allowed table mount, and omits the Robotiq/gripper assembly to
match the simplified PyBullet model.

## V1 safety boundary: Level 6 proposes, PyBullet decides

The default V1 preparation now maps the pinned CuRobo v0.7.8 UR5e collision
spheres onto the local prefixed links. Before mapping, it verifies the six arm
collision-mesh basenames and local origins plus the revolute-joint
parents/children, origins, and axes against CuRobo's bundled UR5e URDF. It adds
one conservative bounding sphere around the camera box. `--v1-ik-only` retains
the previous sphere-free artifact when motion planning is not needed.

Level 6 intentionally remains unchanged: its V1 `IKSolverConfig` receives no
world model and starts with self-collision checking and collision optimization
disabled. The mapped sphere model is used by Level 7 `MotionGen`, not to alter
endpoint acceptance.

Consequently, a V1 `success` means only that CuRobo proposed an IK branch within
its solver thresholds. Every proposed joint vector is passed through the
existing PyBullet validator, which authoritatively checks:

- expected joint order, finite values, and UR5e joint limits
- optical-camera FK position and orientation residuals
- arm/world, camera/world, camera/robot, and robot self-collision
- measured endpoint clearance, `0.005 m` by default
- camera-to-workpiece-center line of sight, unless explicitly disabled

The table top, four legs, workpiece mesh, simplified arm, and camera proxy are
part of those endpoint checks. The ground plane is intentionally excluded from
the reachability clearance query. The fixed robot base/table contact is an
allowed mounting contact.

This is not a hardware collision certificate. The robot model omits the
Robotiq/gripper and approximates the camera housing with a box; calibration,
unmodelled cabling/tools, controller limits, uncertainty, and continuous robot
motion remain outside this result. A validated endpoint also does not prove a
collision-free path from the current state. `--path-validation-steps N` merely
records a sampled straight joint-space diagnostic and does not change endpoint
acceptance or invoke a motion planner.

The interactive Level 7 GUI is a separate step after endpoint filtering. With
a reviewed collision-aware CuRobo config, clicking a point and pressing **Move
to selected viewpoint (CuRobo)** invokes V1 `MotionGen.plan_single_js` or V2
`MotionPlanner.plan_cspace`. PyBullet then revalidates every dense output
waypoint plus joint-space subdivisions before simulated playback. A config
created with `--v1-ik-only` is deliberately rejected for this workflow. See
[Level 7 CuRobo Motion Planning](wiki/Level-7-CuRobo-Motion-Planning).

A rejected record means this finite deterministic search found no branch that
passed every configured gate. It is reported as `no_valid_solution_found`, not
as a mathematical proof that the viewpoint is unreachable.

## Report contract

For the primary path, the JSON report should identify:

```text
backend:            curobo_v1_legacy_goalset_ik
validation_backend: pybullet_endpoint_validation
```

The CLI also writes backend-policy provenance under
`backend_metadata.selection`:

```text
requested_backend:       auto | curobo | pybullet
effective_backend:       curobo | pybullet
policy:                  prefer_curobo_v1_then_pybullet | forced_curobo | forced_pybullet
automatic_fallback_used: true | false
curobo_config_path:       absolute selected config path; auto always targets V1
fallback_reason:         present only when automatic fallback occurred
curobo_runtime:          API/version/driver details after CuRobo preflight
```

The top-level `backend` is always the candidate generator that actually ran.
For example, an automatic CPU fallback reports
`backend: pybullet_multistart_ik`, while its nested selection metadata explains
that `auto` requested V1 first and why fallback occurred.

`backend_metadata` records the package, Python, driver and Torch CUDA versions,
robot config, `solver_batch_size: 1`, number of sequential goal-set solves,
goal-set and seed counts, CUDA-graph shape, seed-reset behavior, elapsed solver
time, and explicit `curobo_self_collision_check: false` and
`curobo_world_collision_check: false` flags. Candidate provenance records the
selected roll index, returned branch index, position residual, raw rotation
metric and converted rotation error in degrees. Exported `clearance_m`, FK
residuals, acceptance status, and ranking values come from the common PyBullet
stage.

Do not infer that an existing `output/pybullet/reachable_viewpoints.json` is a
GPU result from its filename. Inspect its `backend` and `backend_metadata`.

## Secondary CuRoboV2 path

The V2 adapter is retained for a separate, newer environment. It accepts only a
`0.8.x` package and requires Python `>=3.10`, driver `>=580.65.06`, at least
compute capability 8.0, and at least 4 GiB VRAM. The V1 Docker image does not
install or support this API.

In a compatible V2 environment, generate and visually inspect its distinct
collision-aware artifact:

```bash
python3 sample/prepare_curobo_robot.py \
  --api-version v2 \
  --run-builder \
  --visualize
```

This writes:

```text
artifacts/curobo/ur5e_camera_curobo.urdf
artifacts/curobo/ur5e_camera_curobo_v2.yml
```

The script runs the official MorphIt RobotBuilder with sphere density `1.5`,
`4000` collision samples, fit metrics, and seed `123`. It then verifies the
base/tool/joint/camera-link contract, sets the fitted sphere buffer to
`0.005 m`, and sets the default joint position to the simulator home pose.
Sphere coverage, protrusion, and every self-collision ignore pair still require
visual review, especially around the camera and omitted gripper.

Run V2 through the same CLI with its own YAML:

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --ik-backend curobo \
  --curobo-robot-config artifacts/curobo/ur5e_camera_curobo_v2.yml
```

Unlike V1, V2 packs all submitted positions as batch `B`, keeps the rolls as
goal set `G`, and performs one `InverseKinematics.solve_pose` call through
`GoalToolPose`. It loads fitted robot collision spheres, enables self-collision,
and supplies the tabletop and four legs as cuboids plus the transformed
workpiece OBJ as a mesh. The ground plane and fixed base spheres remain omitted
for consistency with the endpoint query and allowed mount contact.

V2's optimizer collision activation distance is
`max(0.01 m, requested_clearance)`. That is an optimization cost distance, not
a measured-clearance certificate. V2 candidates still pass through the exact
same PyBullet endpoint validator, and the report's `clearance_m` still comes
from PyBullet.

## Verification status and limits

Run the host-safe suite from the repository root:

```bash
python3 -m unittest discover -s tests -v
```

Those tests cover deterministic viewpoint sampling, roll redundancy, pose and
quaternion packing, non-identity base transforms, the exact V1 rotation metric,
V1/V2 schema detection, compatibility diagnostics, result shape/joint-order
normalization, generated URDF topology and camera extrinsic, V2 scene poses and
config finalization, automatic V1 preference and typed fallback, strict
explicit backend modes, the CUDA tensor smoke-test failure path, and a fake
external candidate backend passing through the real PyBullet validator and
JSON writer. The integration suite also verifies that the filter restores its
entry joint state after successful evaluation and after a backend exception.

They do **not** install or import CuRobo, compile CUDA extensions, execute a real
V1/V2 solver, assess MorphIt sphere quality, or establish cross-backend parity.
Those remain separate integration checks.

The pinned V1 image and `check_env` passed on the RTX 3060 Laptop GPU with
driver `535.261.03`, Torch CUDA `12.2`, compute `8.6`, and `5.80 GiB` visible
VRAM. On 2026-07-23 the world-up default produced 64 validated endpoints plus
36 rejections in about `10.85 s`; viewpoint 23 ranked best. An earlier explicit
four-roll run produced 88 endpoints plus 12 rejections in about `10.05 s`, with
about `6.61 s` in the solver, but allowed rotated frames. These are execution
results, not promised accepted counts or continuous-motion/hardware-safety
guarantees. V2 and cross-backend parity remain unvalidated.
