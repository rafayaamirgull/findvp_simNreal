# Frames and Transforms

## Notation

This wiki uses `T_A_B` for the homogeneous transform that expresses frame `B`
in frame `A` and maps coordinates from `B` into `A`.

The implementation names transforms as `A_from_B` with the same meaning.

## World and robot base

- World is Z-up.
- The table surface is at world `z=0.75 m`.
- The UR5e base is translated to `(0, 0, 0.75) m` with identity rotation.

Therefore:

```text
T_world_base = translation(0, 0, 0.75)
```

## Optical camera convention

A viewpoint is `T_world_camera`, where `camera` is the optical frame:

- `+Z` is the viewing direction;
- `+X` is image right;
- `-Y` is image up;
- repository and PyBullet quaternions are `[x, y, z, w]`.

The look-at rotation matrix stores `[right, down, forward]` as its columns.
PyBullet camera rendering uses `forward = +Z` and `up = -Y`.

## Camera extrinsic

The optical camera is attached to `ur5e_wrist_3_link`. Its transform is built
from three fixed transforms:

```text
T((0, 0, -0.03),       Rxyz(0, 0, 90))
T((0.08, 0, 0.05),     Rxyz(180, -101.46, 0))
T((0.01085, 0.009, 0.021), Rxyz(-90, 0, -90))
```

Fixed-axis XYZ Euler angles are composed as:

```text
R = Rz(yaw) * Ry(pitch) * Rx(roll)
```

The resulting `T_parent_camera` is approximately:

```text
[-1.000000000, -0.000000000,  0.000000000, 0.009000000]
[ 0.000000000, -0.980063651, -0.198683769, 0.098425618]
[ 0.000000000, -0.198683769,  0.980063651, 0.034806050]
[ 0.000000000,  0.000000000,  0.000000000, 1.000000000]
```

The current fixed `wrist_3 -> flange -> tool0` chain composes to identity, so
the preparation script can use this same numeric matrix as `T_tool_camera`.
Tests verify the camera-to-tool target round trip against PyBullet FK.

## Camera housing frame

The camera box center is offset from the optical center by:

```text
T_camera_body = translation(0, 0, -0.0115)
```

Thus the box extends behind the viewing origin instead of protruding equally in
front of it. The body and collision proxy use identical frames.

## Desired camera pose to PyBullet tool target

PyBullet IK targets `ur5e_tool0`, while the upstream target is the optical
camera. The conversion is:

```text
T_world_tool = T_world_camera * inverse(T_tool_camera)
```

The actual camera FK used for validation is:

```text
T_world_camera = T_world_parent * T_parent_camera
```

No target is accepted based only on tool FK; the optical camera pose is
recomputed and compared directly with the requested camera pose.

## CuRobo goal packing

CuRobo's generated URDF has `ur5e_base_link` as its root and
`ur5e_camera_optical_frame` as its tool frame. Camera goals are transformed to
robot-base coordinates:

```text
T_base_camera = inverse(T_world_base) * T_world_camera
```

Only then is quaternion storage changed from repository XYZW to CuRobo WXYZ:

```text
[x, y, z, w] -> [w, x, y, z]
```

Packed arrays are viewpoint-major and roll-minor:

```text
positions:   [B, G, 3]
quaternions: [B, G, 4]
```

The V1 adapter takes one `[1, G, ...]` slice per sequential solve. V2 consumes
the full batch.

## Rotation-error metrics

PyBullet validation uses the full shortest quaternion angle:

```text
theta = 2 * acos(abs(dot(q_actual, q_target)))
```

CuRobo V1 uses a different raw metric:

```text
m = sin(theta / 2)
```

The adapter converts the requested degree tolerance into `m` and converts
reported V1 residuals back to degrees for provenance. CuRobo V2 uses radians.
The final accepted orientation error is always independently recomputed by
PyBullet in degrees.

## Frame checklist for upstream pipelines

- Supply the optical center, not `tool0` or the camera housing center.
- Supply world-frame metres.
- Supply a normalized XYZW quaternion.
- Ensure the desired optical `+Z` points in the intended viewing direction.
- Do not pre-transform into the robot base for the common API.
- Let the CuRobo adapter perform XYZW-to-WXYZ reordering.

Next: [Using External Viewpoints](Using-External-Viewpoints) or
[Collision Safety and Limitations](Collision-Safety-and-Limitations).
