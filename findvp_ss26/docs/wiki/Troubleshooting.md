# Troubleshooting

## The default command used PyBullet instead of CuRobo

`--ik-backend auto` is the default. It prefers pinned CuRobo V1, but catches a
typed runtime-unavailability diagnostic and prints:

```text
CuRobo V1 is unavailable; falling back to PyBullet multi-start IK.
Reason: ...
```

The reason can identify a missing/incompatible package, Python or driver
requirement, failed CuRobo import, wrong Torch CUDA build, unavailable CUDA
device, insufficient GPU memory/capability, or failed CUDA tensor smoke test.
The JSON preserves the same decision under `backend_metadata.selection`; the
top-level `backend` tells you which solver actually ran.

Use `--ik-backend curobo --curobo-robot-config PATH` when fallback is
unacceptable. Use `--ik-backend pybullet` to select CPU IK without probing the
GPU stack.

## CuRobo failed and auto did not fall back

This is intentional for failures outside runtime availability. Once V1
preflight succeeds, missing/malformed configuration, a wrong schema, backend
construction, solver execution, malformed solver results, and PyBullet
validation errors are hard failures. Retrying those errors through a different
solver would hide defects and make the run difficult to audit.

## CuRobo 0.8 reports that the R535 driver is too old

Symptom:

```text
CUDA available: False
RuntimeError: The NVIDIA driver on your system is too old
```

Cause: the installed Torch/CuRobo 0.8 stack requires an R580+ driver. The RTX
3060 hardware is not the blocker.

For the current R535 host, use the pinned V1 container:

```bash
docker build -f Dockerfile.curobo-v1 -t findvp-curobo:0.7.8 .
xhost +SI:localuser:root

docker run -it --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  -e DISPLAY="$DISPLAY" \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/workspace \
  findvp-curobo:0.7.8
```

Then run `check_env` inside it.

## `CUDA available: False` inside the V1 container

Check:

1. the container was started with `--gpus all`;
2. NVIDIA Container Toolkit is installed and configured on the host;
3. `nvidia-smi` works on the host;
4. `nvidia-smi` works inside the container;
5. the V1 `check_env` reports Torch CUDA 12.2 and driver `>=535.54.03`.

Do not install an arbitrary latest Torch wheel into the V1 image; that can
recreate the driver/runtime mismatch.

## PyBullet exits with `MIT-SHM BadValue` or reports `llvmpipe`

Typical output is:

```text
GL_VENDOR=Mesa
GL_RENDERER=llvmpipe (...)
X Error of failed request: BadValue
Major opcode of failed request: 130 (MIT-SHM)
Minor opcode of failed request: 3 (X_ShmPutImage)
```

This happens before active perception begins. CUDA compute and GUI rendering
are separate container capabilities: CuRobo may see the GPU while PyBullet
cannot access NVIDIA OpenGL. A default private Docker IPC namespace can also
prevent the host X server from reading the MIT-SHM image segment.

Rebuild the current V1 image and **recreate** the container; restarting the old
container does not change its driver capabilities or IPC namespace:

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
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  findvp-curobo:0.7.8
```

Use `--ipc=host` only for a trusted local container because it shares the
host's IPC namespace. Verify the renderer inside the new container:

```bash
glxinfo -B | grep -E 'OpenGL vendor|OpenGL renderer'
```

It should report NVIDIA, not `llvmpipe`. If no GUI is required, the immediate
alternative is to add `--headless`; that avoids X11 and MIT-SHM entirely while
preserving CuRobo IK, motion planning, image capture, and scoring.

## `module 'warp' has no attribute 'torch'`

CuRobo v0.7.8 MotionGen uses the older public `warp.torch` module. An unbounded
dependency installation can select Warp 1.11 or newer, where that API was
reorganized. The supported V1 image pins the tested `warp-lang==1.5.1`
release. Rebuild `Dockerfile.curobo-v1`, or repair an already-running
disposable container with:

```bash
python3 -m pip install --no-deps --force-reinstall warp-lang==1.5.1
check_env
```

## CuRobo robot config not found

Symptom:

```text
CuRobo robot config not found: artifacts/curobo/ur5e_camera_curobo.yml
```

V1 and V2 use explicit generation-specific names. For V1:

```bash
python3 sample/prepare_curobo_robot.py --api-version v1
```

Then pass:

```text
artifacts/curobo/ur5e_camera_curobo_v1.yml
```

Automatic mode uses that path when `--curobo-robot-config` is omitted. If V1
runtime preflight succeeds, a missing file is a hard configuration error; it
does not trigger CPU fallback. If preflight reports runtime unavailability,
auto selects PyBullet before the config is needed.

## `--run-builder` or `--visualize` fails for V1

Those options are V2-only. V1 generation is:

```bash
python3 sample/prepare_curobo_robot.py --api-version v1
```

## Generated CuRobo paths point to `/home/...` inside the container

The YAML and URDF mesh paths are absolute. Regenerate them after mounting the
repository at `/workspace`:

```bash
cd /workspace
python3 sample/prepare_curobo_robot.py --api-version v1
```

Do not copy a bare-host YAML into a differently mounted container.

## V1 YAML with CuRobo 0.8, or V2 YAML with 0.7.8

The schemas and APIs are intentionally incompatible. The adapter requires:

```text
V1 robot_cfg  <-> exactly nvidia-curobo 0.7.8
V2 kinematics <-> nvidia-curobo 0.8.x
```

Regenerate the correct artifact or switch to the matching environment.

## PyBullet prints missing inertial-data warnings

The reduced fixed/kinematic model retains several fixed-frame links without
inertial data. PyBullet assigns default inertial values and prints warnings.
They are expected for this kinematic viewpoint environment, but the environment
must not be reinterpreted as an accurate dynamics model.

## PyBullet prints a build timestamp

`pybullet build time: ...` is emitted by importing PyBullet. It is informational.
The standalone CuRobo preparation module should not emit it because preparation
no longer imports the PyBullet executable.

## Active perception pauses at `Creating new Mesh cache`

This message normally appears during the first CuRobo collision-world and CUDA
warm-up. It is not image capture, and `--motion-seconds` does not shorten it.
The same run also pays one-time DINO model/target-feature initialization after
its first successful capture.

The current active-perception path reduces repeated work by reusing the motion
planner and prepared target features, using a default DINO maximum image side
of 448 pixels, and keeping the robot at its current state between angular
levels. Keep `--debug` off unless cluster/match visualizations are required;
use `--feature-max-image-size 640` only when the extra descriptor resolution is
worth the slower ranking. `--return-home-each-level` deliberately restores the
slower legacy traversal.

## GUI does not open from Docker

On the host:

```bash
xhost +SI:localuser:root
```

Start the container with `DISPLAY` and the X11 socket mounted:

```bash
docker run -it --rm \
  --network=host \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v "$PWD":/workspace \
  findvp-pybullet-env:24.04
```

Use `--headless` when GUI access is unnecessary.

## `--visualize-viewpoints` shows nothing

Debug points are only drawn in GUI mode. Remove `--headless`. Accepted points
are green, rejected points red, and the object center blue.

## The minimum radius is automatically raised

This is expected when the requested near radius would clip the object's
bounding sphere. For the default mesh, `0.30 m` becomes approximately
`0.326624 m` with the configured camera and 10% margin.

If `--view-radius-max` is below the framing requirement, the CLI stops. Increase
the maximum radius, use a smaller/scaled object, or deliberately revise the
camera/framing policy.

## Every viewpoint is rejected

Inspect `failure_counts` and `furthest_failure_stage_observed` in the JSON:

- `ik_residual`: increase seeds/roll coverage before relaxing validated pose tolerances;
- `joint_limits`: add search diversity or inspect frame/target correctness;
- collision labels: inspect workcell transforms and proxy geometry;
- `occluded`: verify the target center and scene, or use
  `--skip-line-of-sight-check` only when that policy is intentional;
- `ik_solver_no_solution`: increase backend effort or inspect backend runtime/config.

Do not immediately loosen clearance or FK tolerances without understanding the
physical consequence.

## Active perception reports an unreachable triangle

An unreachable triangle no longer aborts the pipeline. The command first tries
the nominal centroid, then nearby points blended toward each of the triangle's
three vertices at two increasing offsets. If none passes the same IK,
collision/clearance, and line-of-sight validation, that triangle is recorded as
`unreachable_after_nearby_fallbacks` and omitted while reachable siblings
continue.

Inspect the level's `*_viewpoints.json` for `fallback_attempts`, per-attempt
failure counts, and `skipped`/`skip_reason`. If every triangle in a level is
skipped, angular search ends with `no_reachable_viewpoints` and preserves the
best observation from an earlier level. A radial continuation that becomes
unreachable similarly stops radial search and retains the best angular or
earlier radial observation.

## A rejected record says `endpoint_reachable: null`

This is deliberate. The finite search found no valid branch but cannot prove
that no endpoint exists. Use `rejection_reason`, failure counts, and search
configuration when interpreting the result.

## `linear_path_collision_free=false` but `reachable=true`

Endpoint acceptance and the optional straight-joint diagnostic are separate.
The endpoint passed; at least one discrete sample on that particular straight
interpolation did not. A curved planned path might exist, and the discrete
samples are not a continuous-collision certificate.

## Report results do not match the wiki snapshots

Check the report's:

- `backend`;
- `backend_metadata.selection.requested_backend`, `effective_backend`, and
  `automatic_fallback_used`;
- `reachability_config`;
- `sampling` requested and effective radii;
- model paths, scale, and rotation;
- PyBullet/package versions.

The default filename may contain a previous custom run. Use a unique
`--viewpoint-output` for every experiment.

The documented roll-0 observations are 52/100 for explicit PyBullet and
64/100 for the separately exercised CuRobo V1 path. They are execution
evidence, not golden counts.

## Workpiece PLY cannot be loaded

The built-in converter supports ASCII and binary little-/big-endian PLY files
whose first two elements are `vertex` and `face`. Confirm the header format,
vertex and face counts, positive unit scale, and file path. Use `--object NAME`
for a mesh in `tasks/3d_models/`, or `--workpiece-ply PATH` for an external mesh.

## Headless planning did not save a PNG

`--headless --plan-viewpoints` writes the reachability JSON and intentionally
does not also save the ordinary headless snapshot. Run `--headless` without
`--plan-viewpoints` for a single RGB capture.

## Four-view capture stops before saving all images

The capture command first tries each nominal equal-area region representative,
then roll redundancy, and finally a deterministic set of positions inside the
same region. If a region is still missing, inspect the backend diagnostics,
generated CuRobo config, workpiece scale/pose, and collision/line-of-sight
failures. `--strict-region-representatives` intentionally skips the position
fallback. A partial manifest stays `in_progress` and is ignored by the ranking
notebook.

## Four-view capture reports a required radius above 0.50 m

The workpiece bounding sphere does not fit the camera frustum within the
capture tool's configured workspace. Check `--workpiece-unit-scale`, the mesh,
and camera parameters. Do not increase the hard workspace limit without also
reviewing robot reachability and collision assumptions.

## The ranking notebook cannot find a complete capture

Run `sample/capture_four_views.py` first, or point `CAPTURE_DIR` at a directory
containing a completed run. The notebook requires a `status: complete`
manifest, exactly four regions, and four existing filenames. Interrupted runs
are intentionally skipped.

## DINOv3 or SAM download is unauthorized

Accept the checkpoint provider's license when required, then authenticate with:

```bash
hf auth login
```

Persist `/root/.cache/huggingface` when using disposable containers so models
are not downloaded again. A fully offline first run cannot construct the
pretrained models unless the cache is already populated.

## The notebook rejects the installed Transformers version

The pinned CuRobo V1 PyTorch build is paired with
`transformers==4.44.2`. Install the repository requirements, restart the
kernel, and rerun all cells:

```bash
python3 -m pip install -r requirements-dinov3.txt
```

Do not let pip replace the CUDA PyTorch build in the CuRobo image.

## SAM/DINO runs out of GPU memory

The notebook frees SAM before loading DINOv3 and temporarily moves DINOv3 to
CPU while segmenting all four robot captures. If a prior failed/interrupted
cell retained tensors, restart the kernel. You can also run on CPU, use the
smaller extractor default, or reduce `max_image_size`, with the tradeoff of
slower execution or a coarser patch grid.

## Automatic object selection chose the table or a partial object

The RGB-only proposal selector assumes one salient isolated object that does
not touch the image boundary. Inspect the proposal plots. For deterministic
PCA/K-Means discovery, inspect `format_cluster_report()` and override the
selection with `object_cluster_ids=(...)`. For the SAM notebook workflow,
revise the proposal/ROI after visual inspection rather than trusting the final
score from an incorrect mask.

## A view is rejected during image ranking

This is expected when it has fewer than 16 descriptor candidates, fewer than
12 RANSAC inliers, an inlier ratio below 30%, or no valid textured local ZNCC
patch. Rejected comparisons are not assigned a misleading numeric score.
Check the selected masks and full-image coordinate mapping before relaxing
matching or geometry thresholds.

Next: [CLI Reference](CLI-Reference),
[JSON Report Reference](JSON-Report-Reference), or
[Testing and Validation](Testing-and-Validation).
