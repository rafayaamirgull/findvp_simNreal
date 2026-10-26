# Using External Viewpoints

## Production-shaped integration

The 100-point shell belongs to the Level 6 fixture. An upstream planner can
submit one `ViewpointCandidate` at a time to:

```python
UR5EyeInHandPyBulletEnv.filter_reachable_viewpoints(...)
```

A one-item call executes the same roll expansion, IK generation, PyBullet
validation, branch selection, and diagnostics as a fixture sample.

## Convert an upstream 3D point to a camera pose

If the upstream pipeline supplies only an optical-center position and the
camera should look at the object center, create a candidate like this:

```python
import math

import numpy as np

from sample.viewpoint_reachability import (
    ViewpointCandidate,
    look_at_rotation_matrix,
    matrix_to_quaternion_xyzw,
)


def candidate_from_point(index, position_xyz, object_center_xyz):
    position = np.asarray(position_xyz, dtype=float)
    center = np.asarray(object_center_xyz, dtype=float)
    if position.shape != (3,) or center.shape != (3,):
        raise ValueError("Camera and target positions must contain three values")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(center)):
        raise ValueError("Camera and target positions must be finite")

    offset = position - center
    radius = float(np.linalg.norm(offset))
    if radius <= 0.0:
        raise ValueError("Camera position must be distinct from the target")

    rotation = look_at_rotation_matrix(position, center)
    azimuth = math.degrees(math.atan2(offset[1], offset[0])) % 360.0
    elevation = math.degrees(math.asin(offset[2] / radius))
    return ViewpointCandidate(
        index=index,
        position_xyz=tuple(float(value) for value in position),
        orientation_xyzw=matrix_to_quaternion_xyzw(rotation),
        radius_m=radius,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
    )
```

All positions must be world-frame metres. Repository quaternions are XYZW.

If the upstream pipeline already supplies a full optical-camera pose, use its
position and normalized XYZW orientation directly. Be aware that the default
roll expansion keeps the input forward axis but rebuilds absolute axial-roll
goals using the repository look-at convention; it does not preserve an
arbitrary incoming axial roll.

## Resolve one backend for a sequential stream

The CLI's automatic policy is not hidden inside
`filter_reachable_viewpoints()`. At the library boundary,
`candidate_backend=None` deliberately continues to mean built-in PyBullet IK.
External callers that want the same V1-preferred policy should resolve it once,
build one backend after constructing the environment, and reuse that backend
for the whole stream:

```python
from sample.pybullet_tabletop_ur5e_eye_in_hand import (
    DEFAULT_CUROBO_IK_SEED_COUNT,
    UR5EyeInHandPyBulletEnv,
    build_selected_candidate_backend,
    resolve_ik_backend,
)
from sample.viewpoint_reachability import (
    DEFAULT_IK_REFINEMENT_STEPS,
    DEFAULT_IK_SEED_COUNT,
    ReachabilityConfig,
)
from sample.workcell_config import DEFAULT_ROBOT_URDF, DEFAULT_WORKPIECE_PLY


# Preflight occurs before the PyBullet robot is created or moved. Passing None
# selects the default artifacts/curobo/ur5e_camera_curobo_v1.yml for auto mode.
selection = resolve_ik_backend("auto", None)

env = UR5EyeInHandPyBulletEnv(
    robot_urdf=DEFAULT_ROBOT_URDF.resolve(),
    workpiece_ply=DEFAULT_WORKPIECE_PLY.resolve(),
    workpiece_unit_scale=0.001,
    workpiece_rotation_z_deg=0.0,
    gui=False,
)

try:
    candidate_backend = build_selected_candidate_backend(
        env,
        selection,
        curobo_seeds=DEFAULT_CUROBO_IK_SEED_COUNT,
    )
    config = ReachabilityConfig(
        seed_count=(
            DEFAULT_CUROBO_IK_SEED_COUNT
            if selection.uses_curobo
            else DEFAULT_IK_SEED_COUNT
        ),
        refinement_steps=(
            0 if selection.uses_curobo else DEFAULT_IK_REFINEMENT_STEPS
        ),
    )
    object_center = env.get_workpiece_center()

    for upstream_id, world_camera_point in upstream_viewpoint_stream:
        viewpoint = candidate_from_point(
            upstream_id,
            world_camera_point,
            object_center,
        )
        report = env.filter_reachable_viewpoints(
            [viewpoint],
            object_center_xyz=object_center,
            config=config,
            candidate_backend=candidate_backend,
        )
        report.backend_metadata = dict(report.backend_metadata or {})
        report.backend_metadata["selection"] = selection.report_metadata()

        if report.reachable:
            selected = report.reachable[0]
            keep_viewpoint(upstream_id, selected.joint_positions_rad)
        else:
            rejection = report.rejected[0]
            remove_viewpoint(
                upstream_id,
                rejection.furthest_failure_stage,
                rejection.failure_counts,
            )
finally:
    env.disconnect()
```

Replace `upstream_viewpoint_stream`, `keep_viewpoint`, and `remove_viewpoint`
with the surrounding pipeline's interfaces. `resolve_ik_backend("pybullet",
None)` forces CPU IK without probing CuRobo. Strict CuRobo use passes
`"curobo"` plus an explicit V1 or V2 config path and propagates any
unavailability error.

Automatic fallback is limited to the typed package/driver/Torch/CUDA/GPU
runtime-unavailability result. If V1 preflight succeeds, configuration/schema,
backend construction, solver, malformed-result, and validation failures remain
hard errors and must not be converted into a CPU retry.

## Robot-state semantics

Every filter call:

1. captures the current joints as `q_start`;
2. evaluates candidate branches;
3. restores `q_start` in a `finally` block;
4. returns the selected joint vector without commanding it permanently.

The restoration also runs when candidate generation, result-contract checking,
or endpoint validation raises. The recommended automatic preflight happens
before environment construction, so a runtime-unavailability fallback has no
robot state to restore.

For fixed-reference filtering, leave the simulated robot at home between
calls. Quality and optional path diagnostics will then use home as the
reference every time.

For stateful sequential ranking, explicitly apply a chosen solution before the
next call:

```python
if report.reachable:
    selected = report.reachable[0]
    env.command_joint_positions(selected.joint_positions_rad, blend=1.0)
```

That changes the next call's travel score, path diagnostic, PyBullet seed 0,
and CuRobo V1 retract state. It still does not plan or execute a continuous
physical trajectory.

## Why the selected backend must be persistent

Call `resolve_ik_backend()` and `build_selected_candidate_backend()` once,
outside the viewpoint loop, and pass the same returned object on every call:

```python
report = env.filter_reachable_viewpoints(
    [viewpoint],
    object_center_xyz=object_center,
    config=config,
    candidate_backend=candidate_backend,
)
```

The builder returns `None` when the resolved backend is PyBullet and a
persistent `CuRoboIKCandidateBackend` when V1 is selected. Reusing the CuRobo
object is important because V1 caches its shape-specific CUDA-graph solver. The
backend serializes access to its mutable solver buffers and resets the seed
generator before each request, making repeated identical sequential inputs
reproducible. On its first CuRobo solve, the backend performs the complete
runtime validation itself, including CUDA device properties and a tensor smoke
test, then caches that successful validation on the backend instance. Later
viewpoints reuse the cached result.

V1 always executes one `B=1` goal-set solve per viewpoint, even if a caller
submits a larger list. See [CuRobo Setup](CuRobo-Setup).

## Batch submission

The common filter also accepts multiple externally generated candidates:

```python
report = env.filter_reachable_viewpoints(
    external_candidates,
    object_center_xyz=object_center,
    config=config,
    candidate_backend=candidate_backend,
)
```

Indices must be unique inside that list. PyBullet and CuRobo V1 still process
viewpoints sequentially; CuRobo V2 can pack them as batch dimension `B`.

## Saving direct-call reports

The CLI automatically records fixture sampling and backend-selection metadata.
For an external pipeline, set accurate provenance before calling
`save_reachability_report`; also attach `selection.report_metadata()` as shown
above when using the resolver directly:

```python
from pathlib import Path

from sample.pybullet_tabletop_ur5e_eye_in_hand import save_reachability_report

report.sampling_metadata = {
    "algorithm": "external_pipeline",
    "source": "name-or-version-of-upstream-stage",
    "count": len(report.sampled),
}

save_reachability_report(
    report,
    config,
    env,
    Path(f"output/pybullet/viewpoint_{upstream_id}.json"),
)
```

Without explicit metadata, the serializer's legacy fallback labels the input
as a Fibonacci shell. Also avoid reusing the default filename for multiple
backend runs, because it is overwritten.

## Integration checklist

- Confirm world-frame metres and XYZW quaternion order.
- Confirm the upstream pose represents the optical center, not `tool0`.
- Keep roll `0` for world-up images, or explicitly accept rotated frames by
  adding nonzero absolute roll goals.
- Keep viewpoint indices unique within a batch.
- Resolve once and reuse one environment and one selected backend.
- Build `ReachabilityConfig` from the effective backend, not merely the
  requested `auto` policy.
- Decide whether every call uses home or the previous selected state.
- Keep endpoint reachability separate from motion planning.
- Record upstream sampling provenance in the JSON report.

Next: [Frames and Transforms](Frames-and-Transforms),
[IK Backends](IK-Backends), or
[JSON Report Reference](JSON-Report-Reference).
