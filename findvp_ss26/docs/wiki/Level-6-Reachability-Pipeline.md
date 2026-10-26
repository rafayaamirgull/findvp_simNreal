# Level 6 Reachability Pipeline

## Purpose

Level 6 removes a camera viewpoint when the configured finite IK search does
not produce any branch that passes the shared PyBullet endpoint validator.

The filter accepts a list of `ViewpointCandidate` objects. The list may contain
the 100 deterministic fixture samples, a batch from another planner, or one
upstream viewpoint.

## Input pose contract

Each viewpoint describes the optical camera in the world frame:

```text
position_xyz:       metres
orientation_xyzw:   normalized quaternion [x, y, z, w]
optical frame:      +Z forward, +X right, -Y up
```

`index` must be unique within a submitted list. `radius_m`, `azimuth_deg`, and
`elevation_deg` are report metadata; the position and orientation define the
actual target.

Configured roll goals rebuild the camera orientation around the input pose's
forward axis using the repository look-at convention. They do not compose onto
an arbitrary upstream axial roll. If an upstream roll must remain exact,
provide the corresponding absolute roll goal or bypass roll expansion in an
application-specific adapter.

The CLI defaults to roll `0` only so accepted viewpoints produce world-up
camera frames. Adding nonzero `--ik-rolls` is an explicit tradeoff: it can
accept more positions, but the selected robot pose and captured image inherit
that optical-axis rotation.

## Shared data flow

```text
ViewpointCandidate
  -> IKProblem
       sampled_viewpoint
       IKGoal[G]
         rolled camera pose
         equivalent tool0 pose
  -> IKCandidateBackend.solve(...)
  -> IKProblemResult
       IKCandidate[K]
  -> PyBullet endpoint gates
  -> selected ReachableViewpoint or RejectedViewpoint
```

The structural backend interface is:

```python
def solve(
    problems: Sequence[IKProblem],
    start_joint_positions_rad: Sequence[float],
    reachability_config: ReachabilityConfig,
) -> list[IKProblemResult]:
    ...
```

## Complete algorithm

### 0. Resolve the CLI backend policy

The CLI default `auto` preflights pinned CuRobo V1 before constructing or
moving the PyBullet robot. A typed package/driver/Torch/CUDA/GPU runtime-
unavailability error selects PyBullet. A usable V1 runtime proceeds with the
generated V1 config; configuration/schema, construction, solver, result, and
validation errors remain hard failures. Explicit `curobo` is strict and
explicit `pybullet` skips this preflight.

This policy is outside the filter itself. Library calls with
`candidate_backend=None` continue to select built-in PyBullet IK; external
callers can use `resolve_ik_backend()` and
`build_selected_candidate_backend()` once before their sequential loop.

### 1. Capture the start state

The environment records the current six-joint configuration. Quality, the
optional straight-path diagnostic, and CuRobo's V1 retract state are relative
to this configuration.

### 2. Expand camera-roll redundancy

Each input position becomes one goal per configured optical-axis roll. The
defaults are:

```text
0, 90, -90, 180 degrees
```

The fixed camera extrinsic converts every desired camera pose to the equivalent
`tool0` pose:

```text
T_world_tool = T_world_camera * inverse(T_tool_camera)
```

See [Frames and Transforms](Frames-and-Transforms).

### 3. Generate IK branches

The selected backend proposes branches:

- PyBullet: deterministic multi-start damped least-squares IK for every
  roll/seed combination.
- CuRobo V1: one `B=1` goal-set solve per viewpoint.
- CuRobo V2: batched goal-set IK with CuRobo collision optimization.

Backend success never bypasses the following PyBullet gates.

### 4. Validate the backend result contract

The shared filter requires:

- exactly one result index for each submitted viewpoint index;
- no duplicate or missing result indices;
- a non-negative `attempted_solutions` value;
- every `goal_index` to address the matching roll goal;
- exactly six finite joints in `ARM_JOINT_ORDER`.

Malformed backend output raises an error instead of becoming an ordinary
unreachable result.

### 5. Apply endpoint acceptance gates

Each branch is checked in this order:

1. Canonicalize equivalent revolutions and check joint limits.
2. Apply the joints and recompute the optical-camera pose with PyBullet FK.
3. Require camera position error at or below `0.001 m` by default.
4. Require shortest-quaternion orientation error at or below `1.0 degree` by default.
5. Require robot, camera, environment, and self-collision clearance of `0.005 m` by default.
6. Require line of sight to the object center unless explicitly disabled.
7. Optionally sample a straight joint-space path as a diagnostic.
8. Compute quality metrics for valid endpoint branches.

See [Collision Safety and Limitations](Collision-Safety-and-Limitations) for
exact collision exclusions.

### 6. Select the best branch for one viewpoint

When the path diagnostic is disabled, the valid branch with the largest
quality score is selected. When it is enabled, the selection key is:

```text
(linear_path_collision_free is not False, quality_score)
```

A path-clear branch therefore wins over a path-colliding branch for the same
viewpoint. A path-colliding branch still makes the endpoint reachable.

### 7. Rank accepted viewpoints

After selecting at most one branch per viewpoint, accepted viewpoints are
sorted by descending quality score. No quality threshold is used for
acceptance.

### 8. Restore the start state

The filter restores the captured joint state before returning and also restores
it if backend candidate generation, backend-result contract checking, or
endpoint validation raises. Calling the filter does not command the selected
solution as the new robot state. Automatic runtime fallback occurs before the
filter and therefore cannot leave a partially evaluated robot configuration.

## Quality score

For joint `j`, define the normalized joint-limit margin:

```text
m_j = 2 * min(q_j - lower_j, upper_j - q_j) / (upper_j - lower_j)
joint_limit_margin = clamp(min_j(m_j), 0, 1)
```

Manipulability is the smallest singular value of the optical-center Jacobian,
with rotational rows scaled by a `0.20 m` characteristic length:

```text
manipulability = sigma_min([J_translation; 0.20 * J_rotation])
manipulability_score = manipulability / (manipulability + 0.04)
```

Normalized travel from the call's start state is:

```text
travel = norm((q - q_start) / (upper - lower)) / sqrt(6)
travel_score = clamp(1 - 2 * travel, 0, 1)
```

Clearance contributes:

```text
clearance_score = clamp(clearance_m / 0.05, 0, 1)
```

The final score is:

```text
quality = 0.30 * joint_limit_margin
        + 0.30 * manipulability_score
        + 0.20 * clearance_score
        + 0.20 * travel_score
```

## Failure diagnostics

Candidate failures use these labels:

| Label | Meaning |
| --- | --- |
| `joint_limits` | Canonicalized branch is outside a joint range |
| `ik_residual` | Recomputed camera FK exceeds a pose tolerance |
| `environment_collision` | Robot violates clearance to table, legs, or workpiece |
| `camera_collision` | Camera proxy violates environment clearance |
| `camera_robot_collision` | Camera proxy violates clearance to a non-allowed robot link |
| `self_collision` | Non-adjacent robot links violate clearance |
| `occluded` | Camera-to-object-center ray is blocked |
| `ik_solver_no_solution` | Backend reported no converged branch |
| `no_ik_solution` | No more specific failure stage was available |

If sibling branches fail at different stages, the rejected record's
`furthest_failure_stage_observed` follows this diagnostic priority:

```text
occluded
camera_robot_collision
camera_collision
environment_collision
self_collision
ik_residual
joint_limits
ik_solver_no_solution
```

The record-level status remains `no_valid_solution_found`; it is not a proof of
unreachability. Accepted records can also contain failure counts because other
roll/seed branches may have failed.

Next: [Using External Viewpoints](Using-External-Viewpoints),
[IK Backends](IK-Backends), or
[JSON Report Reference](JSON-Report-Reference).
