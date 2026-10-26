"""Generate and cache the UR5e capability map with the PyBullet model."""

from __future__ import annotations

if __package__:
    from .pybullet_tabletop_ur5e_eye_in_hand import (
        DEFAULT_CAPABILITY_SAMPLE_COUNT,
        UR5EyeInHandPyBulletEnv,
        generate_capability_map,
        save_capability_map,
    )
    from .workcell_config import (
        DEFAULT_CAPABILITY_MAP,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_PLY,
    )
else:
    from pybullet_tabletop_ur5e_eye_in_hand import (
        DEFAULT_CAPABILITY_SAMPLE_COUNT,
        UR5EyeInHandPyBulletEnv,
        generate_capability_map,
        save_capability_map,
    )
    from workcell_config import (
        DEFAULT_CAPABILITY_MAP,
        DEFAULT_ROBOT_URDF,
        DEFAULT_WORKPIECE_PLY,
    )


if __name__ == "__main__":
    env = UR5EyeInHandPyBulletEnv(
        robot_urdf=DEFAULT_ROBOT_URDF,
        workpiece_ply=DEFAULT_WORKPIECE_PLY,
        workpiece_unit_scale=0.001,
        workpiece_rotation_z_deg=0.0,
        gui=False,
    )
    try:
        capability_map = generate_capability_map(
            env, samples=DEFAULT_CAPABILITY_SAMPLE_COUNT
        )
        save_capability_map(DEFAULT_CAPABILITY_MAP, capability_map)
        print(
            f"Saved {len(capability_map.workspace_xyz_m)} workspace voxels "
            f"to {DEFAULT_CAPABILITY_MAP}"
        )
    finally:
        env.disconnect()
