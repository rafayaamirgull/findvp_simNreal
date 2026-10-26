# CuRobo Setup

## Supported project paths

The repository intentionally supports two incompatible CuRobo generations:

| Path | Runtime | Candidate generation | Final acceptance |
| --- | --- | --- | --- |
| V1, primary for this machine | exactly `nvidia-curobo==0.7.8`, PyTorch CUDA 12.2, driver `>=535.54.03` | sequential `B=1` `IKSolver.solve_goalset`, IK only | PyBullet |
| V2, secondary | `nvidia-curobo==0.8.x`, Python `>=3.10`, driver `>=580.65.06` | batched `InverseKinematics.solve_pose`, fitted spheres and scene optimization | PyBullet revalidation |

Schema detection is explicit:

- V1 YAML has top-level `robot_cfg`;
- V2 YAML has top-level `kinematics`;
- the installed package API must match the selected YAML.

The CLI defaults to `--ik-backend auto`: it prefers the pinned V1 path and
falls back to PyBullet only when full runtime preflight raises the typed
CuRobo-unavailable error. Explicit `--ik-backend curobo` remains strict and
never falls back; explicit `--ik-backend pybullet` does not probe CuRobo.

## Current RTX 3060 / R535 machine

The reported host is:

```text
GPU:    NVIDIA GeForce RTX 3060 Laptop GPU
Driver: 535.261.03
```

The RTX 3060 architecture is compute capability 8.6 and the reported laptop
has 6 GiB VRAM. The GPU architecture can run either project adapter, but the
R535 driver is below the V2 guard `580.65.06`. The reviewed path for this host
is therefore V1.

The previously tested CuRobo 0.8 environment failed because its Torch build
required a newer NVIDIA driver, so `torch.cuda.is_available()` returned false.
That is a software-stack mismatch, not evidence that the RTX 3060 hardware is
incapable of GPU IK.

## Build the V1 container

From the repository root:

```bash
docker build -f Dockerfile.curobo-v1 -t findvp-curobo:0.7.8 .
```

Start it with GPU passthrough and mount the checkout at `/workspace`:

```bash
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

The IPC setting is required for PyBullet's X11 MIT-SHM buffers. The driver
capability list exposes CUDA plus NVIDIA OpenGL/GLX and X11 display support;
`compute,utility` alone permits CuRobo but makes the GUI fall back to Mesa
`llvmpipe`. Omit the X11/IPC options only for commands using `--headless`.

The image uses:

| Component | Value |
| --- | --- |
| Base image | `nvcr.io/nvidia/pytorch:23.08-py3` |
| CuRobo tag | `v0.7.8` |
| Verified CuRobo commit | `d64c4b005459db10c5dd867d8b30a87d5bda9bdb` |
| Torch/CUDA target | CUDA 12.2 |
| Extension architecture | `TORCH_CUDA_ARCH_LIST=8.6+PTX` |
| NumPy | exactly `1.24.4` |
| numpy-quaternion | exactly `2023.0.4` |
| Warp | exactly `1.5.1` for the V1 MotionGen `warp.torch` API |
| timm | exactly `1.0.20` for DINOv3 ViT support |
| Transformers | exactly `4.44.2` for automatic SAM masks on the pinned PyTorch build |
| scikit-learn | exactly `1.3.2` for PCA/K-Means spatial clustering |
| Source and license | retained under `/opt/curobo` |

The NumPy pair is explicit because NGC 23.08's numba requires NumPy below 1.25,
while newer numpy-quaternion releases require NumPy 1.25 or newer. Warp is
pinned because newer releases removed the `warp.torch` module used by CuRobo
V1 MotionGen. timm and Transformers use the existing PyTorch installation so
pip does not replace NVIDIA's CUDA 12.2 build. The Docker build runs a small,
weight-free DINOv3 forward pass, imports the SAM pipeline, and runs a
PCA/K-Means smoke test. Full SAM and DINOv3 checkpoints remain runtime
downloads because access can require Hugging Face authentication. The build
and `check_env` both run `pip check`, and `check_env` verifies all pinned
runtime versions.

“Pinned” does not mean the image is bit-for-bit reproducible: other transitive
Python dependencies, PyBullet, and Pillow are not all version-pinned, and the
base image is referenced by tag rather than digest.

## Verify the container

Inside the container:

```bash
check_env
```

This strict V1 check requires:

- `nvidia-curobo==0.7.8`;
- `numpy==1.24.4` and `numpy-quaternion==2023.0.4` with a clean `pip check`;
- `timm==1.0.20` and an importable DINOv3 model definition;
- `transformers==4.44.2` and an importable automatic-mask pipeline;
- `scikit-learn==1.3.2` for deterministic PCA/K-Means clustering;
- Torch compiled for CUDA 12.2;
- driver `>=535.54.03`;
- a visible CUDA device;
- compute capability `>=8.6` for this image build;
- at least 4 GiB VRAM;
- successful imports of the legacy CuRobo IK API;
- a successful real CUDA tensor operation and synchronization.

The automatic selector performs equivalent runtime checks through the Python
adapter, including its own CUDA tensor smoke test. Detecting a device is not
enough if the first kernel cannot execute because of an ABI, container, or
architecture problem.

The ordinary Ubuntu `Dockerfile` has a different, permissive command also
called `check_env`; it does not install or validate CuRobo.

## Persist perception model weights

The V1 image installs the Python packages but does not bake restricted model
weights into the image. Mount the Hugging Face cache for disposable runs:

```bash
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
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  findvp-curobo:0.7.8
```

Accept the model license and use `hf auth login` if the hub requires it. See
[DINOv3 Matching and Ranking](DINOv3-Matching-and-Ranking) for the notebook
model IDs and memory-management sequence.

## Generate the V1 robot files

Always generate inside the same `/workspace` mount used to run IK:

```bash
cd /workspace
python3 sample/prepare_curobo_robot.py --api-version v1
```

This writes:

```text
artifacts/curobo/ur5e_camera_curobo.urdf
artifacts/curobo/ur5e_camera_curobo_v1.yml
```

The files contain absolute URDF and mesh paths. Regenerate them after moving
the checkout or changing the container mount. A config generated on the bare
host with `/home/rafay/...` paths is invalid inside `/workspace`.

V1 does not support `--run-builder` or `--visualize`; CuRobo v0.7.8 does not
provide the MorphIt RobotBuilder used by the V2 workflow.

## Generated CuRobo robot

The camera-aware URDF:

- has exactly one root, `ur5e_base_link`;
- retains the six prefixed UR5e revolute joints;
- uses absolute mesh paths;
- makes reviewed collision geometry visible to the V2 sphere fitter;
- removes fixed base collision geometry at the allowed table mount;
- adds `ur5e_camera_optical_frame` as the tool/EE frame;
- adds a `42 x 42 x 23 mm` camera box behind the optical center;
- omits the Robotiq/gripper assembly.

The default V1 YAML is collision-aware for Level 7:

- the preparation step loads the pinned v0.7.8 package's UR5e sphere model;
- it first verifies local/reference collision meshes, link origins, joint
  topology, joint origins, and axes;
- it maps the reference spheres onto the local prefixed links;
- it encloses the camera box in one conservative bounding sphere;
- it keeps PyBullet as the authoritative endpoint and returned-trajectory
  validator.

Level 6 still configures its V1 IK solver without CuRobo collision checking, so
the accepted endpoint set continues to be decided by PyBullet. Use
`--v1-ik-only` only when Level 7 is not needed; that option retains the legacy
sphere-free YAML and the motion planner rejects it.

## Run the V1 Level 6 fixture

The default command prefers the generated V1 artifact:

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --viewpoint-output output/pybullet/reachable_viewpoints_curobo_v1.json
```

If V1 preflight reports package, driver, Torch/CUDA, CUDA-device, GPU, or tensor
execution unavailability, this command prints the exact reason and runs
PyBullet instead. Inspect the report's actual `backend` before treating it as a
GPU result.

Use strict mode when a fallback result would be unacceptable:

```bash
python3 sample/pybullet_tabletop_ur5e_eye_in_hand.py \
  --headless \
  --plan-viewpoints \
  --ik-backend curobo \
  --curobo-robot-config artifacts/curobo/ur5e_camera_curobo_v1.yml \
  --viewpoint-output output/pybullet/reachable_viewpoints_curobo_v1.json
```

Use a distinct output filename so a GPU report cannot be confused with or
overwrite a PyBullet report.

## V1 solver behavior

With defaults, each viewpoint is solved as:

```text
B=1 camera position
  x G=1 absolute optical-axis roll goal by default (0 degrees, world-up)
  x K=32 seeds shared across the goal set
  -> solve_goalset
  -> up to K successful returned branches
  -> shared PyBullet endpoint validation
```

Important details:

- The fixture makes 100 sequential `solve_goalset` calls, not one `B=100` call.
- One solver is cached by `(G, K, position tolerance, rotation threshold)`.
- A lock serializes access to mutable CUDA-graph and seed buffers.
- `reset_seed()` runs before every viewpoint for deterministic internal samples.
- The call-entry joints are passed as `retract_config` for regularization.
- `use_nn_seed=False`, fixed samples, CUDA graphs, and seed `123` are used.
- `return_seeds == num_seeds`; every successful returned branch reaches PyBullet.
- Returned branch order is not treated as a stable original seed identifier, so
  report `seed_index` is `null` for CuRobo.
- Multiple returned branches may select the same roll goal.

Expected V1 result shapes are:

```text
success/error/goalset_index: [1, K]
joint solution:              [1, K, D]
```

The adapter validates shapes, normalizes the result joint order by name, and
rejects missing joints, invalid goal indices, or non-finite successful results.

### V1 orientation metric

CuRobo V1's raw orientation residual is not radians. For the full shortest
quaternion angle `theta`:

```text
m = sin(theta / 2)
```

The default 1-degree tolerance becomes:

```text
sin(0.5 degrees) = 0.008726535498373935
```

The adapter stores both the raw `sin_half_angle` value and its converted degree
value in solver provenance. PyBullet independently recomputes the final degree
error before acceptance.

## V2 workflow

V2 belongs in a separate environment with CuRobo 0.8.x and an R580+ driver.
Generate and inspect its distinct artifact with:

```bash
python3 sample/prepare_curobo_robot.py \
  --api-version v2 \
  --run-builder \
  --visualize
```

The builder uses:

```text
sphere density:       1.5
collision samples:    4000
compute metrics:      enabled
seed:                 123
```

After fitting, the script verifies the base, tool, six-joint, and camera-link
contracts; rejects unsafe camera ignore pairs; sets the collision-sphere buffer
to `0.005 m`; and sets the simulator home configuration.

The resulting spheres and every self-collision ignore pair still require
visual review, particularly around the camera and omitted gripper.

At solve time V2:

- packs all submitted viewpoints as batch `B` and rolls as goal set `G`;
- calls `solve_pose` once per submission;
- loads collision spheres and enables self-collision;
- supplies tabletop plus four legs as cuboids and the workpiece as a mesh;
- omits the ground and fixed mounting-base geometry;
- uses optimizer activation distance `max(0.01 m, requested_clearance)`;
- creates a fresh solver per submission;
- does not pass the current PyBullet joints into `solve_pose`.

The optimizer activation distance is not a measured-clearance certificate.
Every V2 branch still passes through PyBullet.

## Report identity

Expected backend names are:

```text
curobo_v1_legacy_goalset_ik
curobo_v2_batched_goalset_ik
```

Both use:

```text
validation_backend: pybullet_endpoint_validation
```

For CuRobo, `attempted_solutions` is `K`, not `G*K`. Candidate provenance can
include API generation, selected goal-set index, returned branch index, solver
position error, raw rotation error/units, and converted degree error.

Always inspect `backend` and `backend_metadata`; a filename alone does not prove
that a report came from GPU IK. CLI reports include
`backend_metadata.selection`, which records `requested_backend`,
`effective_backend`, policy, fallback status/reason, preferred V1 config, and
successful CuRobo preflight details when available. The nested key names retain
the V1-default terminology even when strict mode is given a V2 YAML.

Automatic fallback covers only typed runtime unavailability. If V1 preflight
succeeds, missing/malformed configuration, a non-V1 schema, backend
construction, solver, result-contract, or validation error stops the run.

## License warning

CuRobo v0.7.8 uses NVIDIA's non-commercial research/evaluation license. Review
the [v0.7.8 license](https://github.com/NVlabs/curobo/blob/v0.7.8/LICENSE)
before building, using, or redistributing the image. The image keeps the source
and license under `/opt/curobo`. This is a project warning, not legal advice.

## Current verification status

Completed:

- 81 host-safe tests across reachability, motion, capture, and perception;
- V1/V2 schema, packing, metric, generated-model, and result-normalization tests;
- automatic V1 selection, strict override, typed fallback, CUDA-smoke failure,
  and robot-state restoration tests;
- Dockerfile `docker build --check` validation;
- host-side V1 URDF/YAML generation;
- full V1 image build on 2026-07-21;
- V1 `check_env` on the RTX 3060 Laptop GPU with driver `535.261.03`, CUDA
  `12.2`, compute `8.6`, and `5.80 GiB` visible VRAM;
- the real 100-viewpoint V1 fixture with no fallback and the current world-up
  default: 64 accepted, 36 rejected, and about `10.85 s` end-to-end time,
  followed by PyBullet endpoint validation;
- the earlier explicit four-roll fixture: 88 accepted, 12 rejected, 100
  sequential cached `solve_goalset` calls, about `6.61 s` solver time and
  `10.05 s` end-to-end time. That configuration permits rotated frames.

Not yet completed in the current validation environment:

- a real V2 `solve_pose`;
- a MorphIt sphere-quality audit;
- fixed cross-backend parity.

These 100-view results are evidence that the V1 GPU path executes; they are not
golden accepted-count regressions.

The existing ignored `output/pybullet/reachable_viewpoints.json` is a custom
PyBullet run, not CuRobo evidence. See
[Testing and Validation](Testing-and-Validation).

Next: [IK Backends](IK-Backends),
[JSON Report Reference](JSON-Report-Reference), or
[Troubleshooting](Troubleshooting).
