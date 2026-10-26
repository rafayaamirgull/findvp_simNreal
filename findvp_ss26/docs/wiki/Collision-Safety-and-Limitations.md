# Collision Safety and Limitations

## Authoritative endpoint validator

Every branch, including a CuRobo-successful branch, is applied to the PyBullet
model and checked against the same endpoint gates. The default required
clearance is `0.005 m`.

PyBullet calls `getClosestPoints` with a query radius of:

```text
max(0.10 m, required_clearance)
```

The exported minimum clearance therefore saturates at that query radius when
nothing is closer; it is not an unbounded exact distance to the nearest object.

## Robot against the environment

The robot is checked against:

- the tabletop;
- all four table legs;
- the workpiece mesh.

The fixed robot mounting contact is allowed for these links:

```text
ur5e_base_link
ur5e_base_link_inertia
ur5e_base
```

A non-allowed robot/environment distance below the configured clearance is
reported as `environment_collision`.

## Camera against the environment

The camera housing box is checked independently against the table, legs, and
workpiece. A distance below clearance is `camera_collision`.

The camera is also checked against the robot. Contact with its mounting parent
`ur5e_wrist_3_link` is allowed. A violation involving another robot link is
`camera_robot_collision`.

## Robot self-collision

PyBullet closest points are queried with the robot against itself. Identical
links and direct parent/child link pairs are excluded. Each remaining unordered
pair is evaluated once. A non-adjacent distance below clearance is
`self_collision`.

This adjacency exclusion avoids treating ordinary articulated neighboring-link
geometry as a collision, but it also means adjacent-link model errors are not
certified by this gate.

## Line of sight

When enabled, a ray starts `0.05 m` forward of the optical center and ends at
the supplied object center. It passes when the first hit is:

- the workpiece; or
- no body.

Any other first hit produces `occluded`.

This is a center-ray test. It does not prove that the complete object surface
or bounding box is visible, nor does it score coverage or expected information
gain.

## Straight joint-path diagnostic

`--path-validation-steps N` interpolates linearly from the call's start joints
to a valid endpoint and runs the same clearance checks at `N` discrete samples.

It does not check:

- the continuous segments between samples;
- velocities, accelerations, jerk, torque, or timing;
- controller or hardware limits beyond static joint position limits;
- alternate curved paths;
- dynamic obstacles.

A `False` diagnostic does not reject an otherwise valid endpoint. It only
prefers a path-clear branch when selecting between valid branches for the same
viewpoint. This is not a motion planner.

## Interactive CuRobo trajectory

The GUI Level 7 workflow is separate from `--path-validation-steps`. It sends
the current and selected endpoint joints to CuRobo's joint-space motion
planner, which can use graph-planner fallback when trajectory optimization
does not find a direct path.

Before playback, PyBullet rechecks every dense CuRobo waypoint and adds linear
subdivisions so each checked joint increment is at most `0.02 rad`. The same
joint-limit and collision-clearance gates used for endpoints are applied to
all these samples. A failed sample rejects the entire trajectory and leaves
the robot at its start state.

This remains a discrete revalidation. It is not exact continuous collision
detection between samples.

## What is included

- Simplified UR5e arm collision meshes.
- Tabletop and four legs.
- Workpiece collision mesh.
- Optical-camera housing box.
- Static endpoint joint limits and FK residuals.
- Optional object-center visibility ray.

## What is not included

- The original Robotiq/gripper assembly.
- Real camera mounting tolerances, cabling, hoses, or tools.
- The ground plane in reachability clearance queries.
- A full hardware cell, safety fencing, people, or unmodelled obstacles.
- Calibration uncertainty or state-estimation uncertainty.
- Exact continuous collision checking between trajectory validation samples.
- Controller dynamics, payload, torque, speed, acceleration, or jerk limits.
- RGB-camera distortion, sensor noise, exposure, or colour calibration.
- Grasping, contact dynamics, or physical execution.

## CuRobo safety boundary

### V1

For Level 6, CuRobo v0.7.8 proposes IK with its collision checks disabled; all
endpoint safety gates remain PyBullet responsibilities. The default preparation
also creates a Level 7 collision model by verifying the local UR5e geometry
against the pinned package model, mapping its spheres, and adding a conservative
camera sphere. MotionGen uses those spheres and the workcell scene. PyBullet
still revalidates every returned path before playback. `--v1-ik-only` omits the
spheres and is rejected by Level 7.

### V2

CuRobo V2 uses fitted collision spheres, self-collision, and the table/workpiece
scene during optimization. Its collision activation distance is an optimizer
cost distance, not a measured-clearance certificate. PyBullet still decides
endpoint acceptance, supplies exported `clearance_m`, and revalidates Level 7
trajectory samples before playback.

## Correct interpretation

Use these exact meanings:

| Result | Meaning |
| --- | --- |
| Accepted endpoint | At least one finite-search branch passed the configured simplified-model endpoint gates |
| Rejected record | No branch found by this finite search passed every gate |
| `linear_path_collision_free=true` | The configured discrete straight-joint samples passed |
| `linear_path_collision_free=false` | At least one configured straight-joint sample failed |
| Interactive trajectory accepted | CuRobo found a path and every dense PyBullet validation sample passed |
| Interactive trajectory rejected | CuRobo found no path, or PyBullet rejected at least one returned path sample |

Do not translate any of these into “safe to execute on hardware” without a
separate calibrated robot model, continuous collision validation,
controller-limit checks, and application-specific safety review.

Next: [Testing and Validation](Testing-and-Validation) or
[Troubleshooting](Troubleshooting).
