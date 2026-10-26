"""CuRobo collision-aware joint-space motion planning.

The module stays import-safe on hosts without CuRobo.  Runtime imports happen
only when :meth:`CuRoboMotionPlanner.plan` is first called.  Both CuRobo v1
(``0.7.8``) and CuRoboV2 (``0.8.x``) are supported, but the supplied robot
configuration must contain collision spheres.  The repository's legacy V1
IK-only artifact is intentionally rejected because it cannot provide
collision-aware motion planning.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

if __package__:
    from .curobo_reachability import (
        CUROBO_API_V1,
        CUROBO_API_V2,
        CuRoboRuntimeProbe,
        detect_curobo_robot_config_api,
        require_curobo_runtime,
    )
else:  # Support direct execution/import from sample/.
    from curobo_reachability import (
        CUROBO_API_V1,
        CUROBO_API_V2,
        CuRoboRuntimeProbe,
        detect_curobo_robot_config_api,
        require_curobo_runtime,
    )


DEFAULT_MOTION_INTERPOLATION_DT_S = 0.02
DEFAULT_MOTION_MAX_ATTEMPTS = 5
DEFAULT_MOTION_IK_SEEDS = 32
DEFAULT_MOTION_GRAPH_SEEDS = 4
DEFAULT_MOTION_TRAJOPT_SEEDS = 12
SUPPORTED_CUROBO_V1_WARP_VERSION = "1.5.1"


class CuRoboMotionPlanningError(RuntimeError):
    """Raised when a collision-aware CuRobo motion cannot be produced."""


class CuRoboMotionInitializationError(CuRoboMotionPlanningError):
    """Raised when the configured CuRobo planner cannot be initialized."""


class CuRoboNoMotionPlanError(CuRoboMotionPlanningError):
    """Raised when initialized CuRobo cannot find a path for one request."""


@dataclass(frozen=True)
class CuRoboMotionPlan:
    """One dense, timed joint trajectory in the application's joint order."""

    joint_names: tuple[str, ...]
    joint_positions_rad: np.ndarray
    interpolation_dt_s: float
    planning_time_s: float
    used_graph: bool | None
    status: str
    initialization_time_s: float = 0.0

    @property
    def waypoint_count(self) -> int:
        return int(self.joint_positions_rad.shape[0])

    @property
    def duration_s(self) -> float:
        return max(0, self.waypoint_count - 1) * self.interpolation_dt_s


def _robot_kinematics_mapping(
    config_data: dict[str, object], api_generation: str
) -> dict[str, object]:
    if api_generation == CUROBO_API_V1:
        robot_cfg = config_data.get("robot_cfg")
        if not isinstance(robot_cfg, dict):
            raise CuRoboMotionPlanningError("CuRobo V1 config is missing 'robot_cfg'.")
        kinematics = robot_cfg.get("kinematics")
    elif api_generation == CUROBO_API_V2:
        kinematics = config_data.get("kinematics")
    else:  # pragma: no cover - guarded by schema detection
        raise CuRoboMotionPlanningError(
            f"Unsupported CuRobo API generation: {api_generation!r}."
        )
    if not isinstance(kinematics, dict):
        raise CuRoboMotionPlanningError("CuRobo config is missing a kinematics mapping.")
    return kinematics


def validate_collision_aware_robot_config(
    config_path: Path, api_generation: str | None = None
) -> dict[str, object]:
    """Load a CuRobo config and require usable robot collision geometry."""

    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"CuRobo robot config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CuRoboMotionPlanningError("CuRobo robot config must be a YAML mapping.")
    generation = api_generation or detect_curobo_robot_config_api(path)
    kinematics = _robot_kinematics_mapping(data, generation)
    collision_links = kinematics.get("collision_link_names")
    collision_spheres = kinematics.get("collision_spheres")
    if not isinstance(collision_links, list) or not collision_links or not collision_spheres:
        if generation == CUROBO_API_V1:
            raise CuRoboMotionPlanningError(
                "The selected CuRobo V1 robot config is IK-only and has no collision "
                "spheres. Generate or select a reviewed collision-aware CuRobo robot "
                "config before enabling robot motion."
            )
        raise CuRoboMotionPlanningError(
            "The selected CuRobo robot config has no collision links/spheres. "
            "Run sample/prepare_curobo_robot.py and review the fitted spheres first."
        )
    return data


def _to_numpy(value: object, field_name: str) -> np.ndarray:
    try:
        return np.asarray(value.detach().cpu().numpy())  # type: ignore[attr-defined]
    except Exception as exc:
        raise CuRoboMotionPlanningError(
            f"Malformed CuRobo motion result field {field_name!r}."
        ) from exc


def _successful(value: object) -> bool:
    array = _to_numpy(value, "success")
    return bool(np.any(array.astype(bool, copy=False)))


def _finite_scalar(value: object, field_name: str, default: float) -> float:
    if value is None:
        return default
    if hasattr(value, "detach"):
        array = _to_numpy(value, field_name).reshape(-1)
        result = float(array[0]) if array.size else default
    else:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = default
    return result if math.isfinite(result) else default


def _normalize_trajectory_positions(
    positions: object,
    result_joint_names: Sequence[str],
    expected_joint_names: Sequence[str],
) -> np.ndarray:
    """Normalize CuRobo trajectory shapes and reorder their joint axis."""

    array = _to_numpy(positions, "trajectory.position")
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[0] < 2:
        raise CuRoboMotionPlanningError(
            f"CuRobo returned an invalid trajectory position shape: {array.shape}."
        )
    names = tuple(str(name) for name in result_joint_names)
    expected = tuple(str(name) for name in expected_joint_names)
    if len(names) != array.shape[1] or len(set(names)) != len(names):
        raise CuRoboMotionPlanningError(
            "CuRobo trajectory joint names do not uniquely match its joint axis."
        )
    missing = [name for name in expected if name not in names]
    if missing:
        raise CuRoboMotionPlanningError(
            f"CuRobo trajectory is missing expected joints: {missing}."
        )
    ordered = np.asarray(array[:, [names.index(name) for name in expected]], dtype=float)
    if not np.all(np.isfinite(ordered)):
        raise CuRoboMotionPlanningError("CuRobo returned non-finite trajectory positions.")
    return ordered


class CuRoboMotionPlanner:
    """Lazy, reusable CuRobo joint-space planner for one reviewed workcell."""

    def __init__(
        self,
        *,
        robot_config_path: Path,
        scene_model: dict[str, object],
        expected_joint_names: Sequence[str],
        clearance_m: float,
        interpolation_dt_s: float = DEFAULT_MOTION_INTERPOLATION_DT_S,
        max_attempts: int = DEFAULT_MOTION_MAX_ATTEMPTS,
    ) -> None:
        self.robot_config_path = Path(robot_config_path).resolve()
        self.scene_model = scene_model
        self.expected_joint_names = tuple(str(name) for name in expected_joint_names)
        self.clearance_m = float(clearance_m)
        self.interpolation_dt_s = float(interpolation_dt_s)
        self.max_attempts = int(max_attempts)
        if not self.expected_joint_names or len(set(self.expected_joint_names)) != len(
            self.expected_joint_names
        ):
            raise ValueError("Expected motion-planner joint names must be non-empty and unique.")
        if not math.isfinite(self.clearance_m) or self.clearance_m < 0.0:
            raise ValueError("Motion-planner clearance must be finite and non-negative.")
        if not math.isfinite(self.interpolation_dt_s) or self.interpolation_dt_s <= 0.0:
            raise ValueError("Motion interpolation dt must be finite and positive.")
        if self.max_attempts <= 0:
            raise ValueError("Motion planning attempts must be positive.")
        if not isinstance(self.scene_model, dict) or not any(
            self.scene_model.get(kind) for kind in ("cuboid", "mesh")
        ):
            raise ValueError("CuRobo motion planning requires a non-empty collision scene.")

        self.api_generation = detect_curobo_robot_config_api(self.robot_config_path)
        self._config_data = validate_collision_aware_robot_config(
            self.robot_config_path, self.api_generation
        )
        self._runtime_probe: CuRoboRuntimeProbe | None = None
        self._planner: object | None = None
        self._planner_joint_names: tuple[str, ...] | None = None
        self._initialization_time_s = 0.0
        self._initialization_lock = threading.Lock()

    def _ensure_initialized(self) -> object:
        if self._planner is None:
            with self._initialization_lock:
                if self._planner is None:
                    started = time.monotonic()
                    print(
                        "Initializing CuRobo motion planner "
                        "(collision scene and CUDA warm-up) ...",
                        flush=True,
                    )
                    self._runtime_probe = require_curobo_runtime(
                        expected_api=self.api_generation
                    )
                    if self.api_generation == CUROBO_API_V1:
                        self._initialize_v1()
                    elif self.api_generation == CUROBO_API_V2:
                        self._initialize_v2()
                    self._initialization_time_s = time.monotonic() - started
                    print(
                        "CuRobo motion planner ready in "
                        f"{self._initialization_time_s:.1f} s.",
                        flush=True,
                    )
        if self._planner is None:  # pragma: no cover - defensive narrowing
            raise CuRoboMotionPlanningError("CuRobo motion planner initialization failed.")
        return self._planner

    def _initialize_v1(self) -> None:
        try:
            warp_version = version("warp-lang")
        except PackageNotFoundError as exc:
            raise CuRoboMotionInitializationError(
                "CuRobo V1 MotionGen requires warp-lang "
                f"{SUPPORTED_CUROBO_V1_WARP_VERSION}, but it is not installed."
            ) from exc
        if warp_version != SUPPORTED_CUROBO_V1_WARP_VERSION:
            raise CuRoboMotionInitializationError(
                "CuRobo V1 MotionGen requires the tested warp-lang "
                f"{SUPPORTED_CUROBO_V1_WARP_VERSION}, found {warp_version}. "
                "Rebuild Dockerfile.curobo-v1 or install the pinned version."
            )
        try:
            from curobo.types.base import TensorDeviceType
            from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

            tensor_args = TensorDeviceType()
            config = MotionGenConfig.load_from_robot_config(
                self._config_data,
                self.scene_model,
                tensor_args=tensor_args,
                num_ik_seeds=DEFAULT_MOTION_IK_SEEDS,
                num_graph_seeds=DEFAULT_MOTION_GRAPH_SEEDS,
                num_trajopt_seeds=DEFAULT_MOTION_TRAJOPT_SEEDS,
                cspace_threshold=0.01,
                interpolation_dt=self.interpolation_dt_s,
                self_collision_check=True,
                self_collision_opt=True,
                collision_activation_distance=max(0.01, self.clearance_m),
                evaluate_interpolated_trajectory=True,
                use_cuda_graph=True,
            )
            planner = MotionGen(config)
            self._warmup_v1_joint_space(planner)
            planner_joint_names = tuple(str(name) for name in planner.rollout_fn.joint_names)
        except Exception as exc:
            raise CuRoboMotionInitializationError(
                f"Could not initialize CuRobo V1 MotionGen: {type(exc).__name__}: {exc}"
            ) from exc
        self._validate_planner_joint_names(planner_joint_names)
        self._planner_joint_names = planner_joint_names
        self._planner = planner

    @staticmethod
    def _warmup_v1_joint_space(planner: object) -> None:
        """Warm only the V1 planning path used by this adapter.

        ``MotionGen.warmup`` also runs Cartesian pose planning and graph-search
        warm-ups.  This adapter exclusively calls ``plan_single_js`` and keeps
        graph search as an exceptional fallback, so eagerly compiling those
        unrelated paths adds substantial startup latency without helping the
        normal capture path.
        """

        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        joint_names = list(planner.rollout_fn.joint_names)  # type: ignore[attr-defined]
        retract = planner.rollout_fn.dynamics_model.retract_config  # type: ignore[attr-defined]
        start_state = JointState.from_position(
            retract.view(1, -1).clone(),
            joint_names=joint_names,
        )
        goal_state = start_state.clone()
        goal_state.position[..., 0] += 0.1
        warmup_config = MotionGenPlanConfig(
            max_attempts=1,
            enable_graph=False,
            enable_finetune_trajopt=True,
            check_start_validity=True,
        )
        # Match CuRobo's own joint-space warm-up count so CUDA graph/buffer
        # reuse is exercised while omitting its unused pose and graph passes.
        for _ in range(3):
            planner.plan_single_js(  # type: ignore[attr-defined]
                start_state.clone(),
                goal_state.clone(),
                warmup_config,
            )

    def _initialize_v2(self) -> None:
        try:
            import torch
            from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
            from curobo.types import DeviceCfg

            device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
            config = MotionPlannerCfg.create(
                robot=self.robot_config_path.as_posix(),
                scene_model=self.scene_model,
                collision_cache={"cuboid": 8, "mesh": 2},
                self_collision_check=True,
                device_cfg=device_cfg,
                num_ik_seeds=DEFAULT_MOTION_IK_SEEDS,
                num_trajopt_seeds=DEFAULT_MOTION_TRAJOPT_SEEDS,
                optimizer_collision_activation_distance=max(0.01, self.clearance_m),
                use_cuda_graph=True,
                max_batch_size=1,
                max_goalset=1,
            )
            planner = MotionPlanner(config)
            planner.warmup(enable_graph=True, num_warmup_iterations=3)
            planner_joint_names = tuple(str(name) for name in planner.joint_names)
        except Exception as exc:
            raise CuRoboMotionInitializationError(
                f"Could not initialize CuRoboV2 MotionPlanner: {type(exc).__name__}: {exc}"
            ) from exc
        self._validate_planner_joint_names(planner_joint_names)
        self._planner_joint_names = planner_joint_names
        self._planner = planner

    def _validate_planner_joint_names(self, planner_joint_names: Sequence[str]) -> None:
        names = tuple(planner_joint_names)
        missing = [name for name in self.expected_joint_names if name not in names]
        unexpected = [name for name in names if name not in self.expected_joint_names]
        if missing or unexpected:
            raise CuRoboMotionPlanningError(
                "CuRobo motion-planner joint mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )

    def _validated_endpoint(self, values: Sequence[float], label: str) -> tuple[float, ...]:
        endpoint = tuple(float(value) for value in values)
        if len(endpoint) != len(self.expected_joint_names):
            raise ValueError(
                f"{label} must contain {len(self.expected_joint_names)} joint positions."
            )
        if not all(math.isfinite(value) for value in endpoint):
            raise ValueError(f"{label} joint positions must be finite.")
        return endpoint

    def plan(
        self,
        start_joint_positions_rad: Sequence[float],
        goal_joint_positions_rad: Sequence[float],
    ) -> CuRoboMotionPlan:
        """Plan a dense collision-aware trajectory between two joint states."""

        start = self._validated_endpoint(start_joint_positions_rad, "Start state")
        goal = self._validated_endpoint(goal_joint_positions_rad, "Goal state")
        planner_was_initialized = self._planner is not None
        planner = self._ensure_initialized()
        planner_names = self._planner_joint_names
        if planner_names is None:  # pragma: no cover - defensive narrowing
            raise CuRoboMotionPlanningError("CuRobo planner joint order is unavailable.")
        start_by_name = dict(zip(self.expected_joint_names, start))
        goal_by_name = dict(zip(self.expected_joint_names, goal))
        ordered_start = [start_by_name[name] for name in planner_names]
        ordered_goal = [goal_by_name[name] for name in planner_names]
        started = time.monotonic()
        if self.api_generation == CUROBO_API_V1:
            result, trajectory, result_names, used_graph, status, reported_time = (
                self._plan_v1(planner, ordered_start, ordered_goal, planner_names)
            )
        else:
            result, trajectory, result_names, used_graph, status, reported_time = (
                self._plan_v2(planner, ordered_start, ordered_goal, planner_names)
            )
        if not _successful(result.success):
            raise CuRoboNoMotionPlanError(
                f"CuRobo could not find a collision-free path ({status}). "
                "Please select a different reachable viewpoint."
            )
        positions = _normalize_trajectory_positions(
            trajectory.position, result_names, self.expected_joint_names
        )
        dt = _finite_scalar(
            getattr(trajectory, "dt", None),
            "trajectory.dt",
            self.interpolation_dt_s,
        )
        if dt <= 0.0:
            dt = self.interpolation_dt_s
        elapsed = time.monotonic() - started
        return CuRoboMotionPlan(
            joint_names=self.expected_joint_names,
            joint_positions_rad=positions,
            interpolation_dt_s=dt,
            planning_time_s=max(elapsed, reported_time),
            initialization_time_s=(
                0.0 if planner_was_initialized else self._initialization_time_s
            ),
            used_graph=used_graph,
            status=status,
        )

    def _plan_v1(
        self,
        planner: object,
        start: Sequence[float],
        goal: Sequence[float],
        planner_names: tuple[str, ...],
    ) -> tuple[object, object, tuple[str, ...], bool | None, str, float]:
        import torch
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        device = planner.tensor_args.device  # type: ignore[attr-defined]
        dtype = planner.tensor_args.dtype  # type: ignore[attr-defined]
        start_state = JointState.from_position(
            torch.as_tensor(start, device=device, dtype=dtype).view(1, -1),
            joint_names=list(planner_names),
        )
        goal_state = JointState.from_position(
            torch.as_tensor(goal, device=device, dtype=dtype).view(1, -1),
            joint_names=list(planner_names),
        )
        plan_config = MotionGenPlanConfig(
            max_attempts=self.max_attempts,
            timeout=15.0,
            enable_graph=False,
            enable_graph_attempt=1,
            enable_finetune_trajopt=True,
            check_start_validity=True,
        )
        result = planner.plan_single_js(  # type: ignore[attr-defined]
            start_state, goal_state, plan_config
        )
        status = str(getattr(result, "status", "planning_failed"))
        if not _successful(result.success):
            return result, start_state, planner_names, None, status, 0.0
        trajectory = result.get_interpolated_plan()
        result_names = tuple(getattr(trajectory, "joint_names", None) or planner_names)
        return (
            result,
            trajectory,
            result_names,
            bool(getattr(result, "used_graph", False)),
            status,
            _finite_scalar(getattr(result, "total_time", None), "total_time", 0.0),
        )

    def _plan_v2(
        self,
        planner: object,
        start: Sequence[float],
        goal: Sequence[float],
        planner_names: tuple[str, ...],
    ) -> tuple[object, object, tuple[str, ...], bool | None, str, float]:
        import torch
        from curobo.types import JointState

        device_cfg = planner.device_cfg  # type: ignore[attr-defined]
        start_state = JointState.from_position(
            torch.as_tensor(start, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1),
            joint_names=list(planner_names),
        )
        goal_state = JointState.from_position(
            torch.as_tensor(goal, device=device_cfg.device, dtype=device_cfg.dtype).view(1, -1),
            joint_names=list(planner_names),
        )
        result = planner.plan_cspace(  # type: ignore[attr-defined]
            goal_state,
            start_state,
            max_attempts=self.max_attempts,
            enable_graph_attempt=1,
        )
        if result is None:
            raise CuRoboNoMotionPlanError(
                "CuRoboV2 returned no collision-free path. Please select a different "
                "reachable viewpoint."
            )
        default_status = "success" if _successful(result.success) else "failed"
        status = str(getattr(result, "status", default_status))
        if not _successful(result.success):
            return result, start_state, planner_names, None, status, 0.0
        trajectory = result.get_interpolated_plan()
        result_names = tuple(getattr(trajectory, "joint_names", None) or planner_names)
        return (
            result,
            trajectory,
            result_names,
            None,
            status,
            _finite_scalar(getattr(result, "total_time", None), "total_time", 0.0),
        )
