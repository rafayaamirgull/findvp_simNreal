# JSON Report Reference

## File and schema

The CLI writes schema version 1 to:

```text
output/pybullet/reachable_viewpoints.json
```

Use `--viewpoint-output` to keep separate reports. The default path is
overwritten without merging.

JSON serialization uses `allow_nan=False`, so NaN or Infinity cannot silently
enter a valid report.

## Top-level fields

| Field | Meaning |
| --- | --- |
| `schema_version` | Currently `1` |
| `backend` | Actual candidate generator identity |
| `validation_backend` | `pybullet_endpoint_validation` |
| `backend_metadata` | Backend-specific runtime and search provenance |
| `reachability_scope` | `collision_free_ik_endpoint` |
| `rejection_semantics` | Finite-search disclaimer |
| `collision_model_notes` | Simplified-model limitations |
| `pose_convention` | World optical-camera pose and XYZW convention |
| `object_center_xyz_m` | Validation target point |
| `robot` | Source URDF, base, joints, limits, and call-entry state |
| `camera` | Parent/tool transforms and collision proxy |
| `world` | Workpiece and table geometry |
| `sampled_count` | Submitted viewpoint count |
| `reachable_count` | Accepted endpoint count |
| `rejected_count` | No-valid-branch count |
| `elapsed_sec` | Filtering wall time |
| `sampling` | Fixture or caller-provided input provenance |
| `reachability_config` | Rolls, effort, tolerances, clearance, LOS, path samples |
| `viewpoints` | One record per submitted viewpoint in input order |
| `reachable_indices_ranked_by_quality` | Accepted IDs in descending quality order |
| `linear_path_collision_free_indices` | Accepted IDs whose optional path diagnostic passed |

## Robot, camera, and world metadata

The report is designed to make a result auditable without relying on console
output. It records:

- source robot URDF and fixed base position;
- exact active-joint order, limits, and start state;
- camera parent, tool-to-camera and parent-to-camera transforms;
- camera collision half-extents;
- workpiece source, scale, baked rotation, extent, base pose, and AABB;
- table size, center, top height, leg dimensions, and leg poses;
- requested and effective fixture bounds and FOV framing values.

Paths can be absolute and environment-specific.

## Fields present on every viewpoint

| Field | Meaning |
| --- | --- |
| `index` | Submitted stable identifier |
| `position_xyz_m` | Optical-center world position |
| `orientation_xyzw` | Optical-frame world orientation |
| `radius_m` | Caller/sampler radius metadata |
| `azimuth_deg` | Caller/sampler azimuth metadata |
| `elevation_deg` | Caller/sampler elevation metadata |
| `roll_deg` | Selected roll for accepted records; original roll for rejected records |
| `reachable` | Whether a validated endpoint branch was found |
| `reachability_status` | Explicit record status |
| `endpoint_reachable` | `true` for accepted; `null` for finite-search rejection |
| `linear_path_checked` | Whether a path diagnostic was run for this selected endpoint |
| `linear_path_collision_free` | `true`, `false`, or `null` |
| `attempted_solutions` | Backend finite-search attempt count |
| `failure_counts` | Aggregate sibling-branch failures by stage |

Records remain in original input order. Use
`reachable_indices_ranked_by_quality` when ranking is required.

## Accepted record additions

| Field | Meaning |
| --- | --- |
| `joint_positions_rad` | Selected six-joint endpoint in report joint order |
| `position_error_m` | PyBullet optical-center FK position residual |
| `orientation_error_deg` | PyBullet shortest-quaternion angle residual |
| `clearance_m` | PyBullet endpoint minimum within its closest-point query radius |
| `joint_limit_margin` | Minimum normalized distance from a joint hard limit |
| `manipulability` | Smallest singular value of scaled optical-center Jacobian |
| `quality_score` | Weighted ranking score |
| `seed_index` | Stable PyBullet seed ID or `null` when not meaningful |
| `solver_provenance` | Backend-specific selected-branch evidence |
| `successful_solution_count` | Number of sibling branches passing PyBullet |

An accepted record can have non-empty `failure_counts` because its other
roll/seed branches may have failed.

## Rejected record additions

| Field | Meaning |
| --- | --- |
| `rejection_reason` | Always `no_valid_solution_found` |
| `furthest_failure_stage_observed` | Highest-priority/deepest diagnostic seen |

A rejected record intentionally uses:

```json
{
  "reachable": false,
  "reachability_status": "no_valid_solution_found",
  "endpoint_reachable": null,
  "rejection_reason": "no_valid_solution_found"
}
```

`null` prevents the finite search from being misread as a proof that no endpoint
exists.

## Backend identities

Expected values include:

```text
pybullet_multistart_ik
curobo_v1_legacy_goalset_ik
curobo_v2_batched_goalset_ik
```

Never infer the backend from the filename. Inspect `backend` and
`backend_metadata`.

PyBullet metadata records roll count, seeds per goal, and refinement passes.
V1 metadata records package/Python/driver/Torch-CUDA versions, `B=1`, G/K,
sequential solve count, CUDA graph shape, seed resetting, solver reuse, timing,
and its disabled CuRobo collision checks. V2 records B/G/K, scene collision
types, self-collision, optimizer activation distance, and timing.

### CLI backend-selection provenance

CLI-generated reports add a nested `backend_metadata.selection` object. Its
fields are separate from solver metadata because requested policy and actual
candidate generator can differ:

| Field | Meaning |
| --- | --- |
| `requested_backend` | CLI request: `auto`, `curobo`, or `pybullet` |
| `effective_backend` | Resolved generator family: `curobo` or `pybullet` |
| `policy` | `prefer_curobo_v1_then_pybullet`, `forced_curobo`, or `forced_pybullet` |
| `automatic_fallback_used` | Whether auto caught a typed V1 runtime-unavailability error |
| `curobo_config_path` | Absolute config path carried by the selection object; auto always targets V1 |
| `fallback_reason` | Typed runtime diagnostic; present only after automatic fallback |
| `curobo_runtime` | API/package/Python/driver preflight details when CuRobo preflight succeeded |

For example, when a machine has no usable CUDA runtime:

```json
{
  "backend": "pybullet_multistart_ik",
  "backend_metadata": {
    "selection": {
      "requested_backend": "auto",
      "effective_backend": "pybullet",
      "policy": "prefer_curobo_v1_then_pybullet",
      "automatic_fallback_used": true,
      "fallback_reason": "CuRobo backend is unavailable ..."
    }
  }
}
```

Fallback is limited to package/driver/Torch/CUDA/GPU runtime unavailability.
Configuration/schema, solver, malformed-result, and validation errors stop the
run and therefore do not produce a misleading CPU-success report. Direct
library callers only get this nested selection object if they attach
`IKBackendSelection.report_metadata()` themselves; the CLI does so
automatically.

## Active-perception reports

`sample/active_perception_triangles.py` writes a family of reports under
`output/pybullet/triangle_views/` by default. Replace `object` below with the
value passed to `--name`.

| Path pattern | Contents |
| --- | --- |
| `object_viewpoints.json` | Level-0 spherical triangles, reachability/fallback evidence, executed pose, and capture metadata |
| `object_level_NN_viewpoints.json` | The three children evaluated at angular refinement level `NN` |
| `*_viewpoints_ranking.json` | Per-level DINO/SIFT ranking, plausible candidates, winning triangle, and winning saved viewpoint |
| `object_end_effector_capture_poses.json` | Append-only numbered angular and radial capture states |
| `object_end_effector_capture_poses.ply` | One colored oriented vertex per saved image for MeshLab |
| `object_radial_viewpoints.json` | Every closer/farther sample, including radius-bound, unreachable, or motion-failed samples |
| `object_radial_eval_NN.json` | Single-candidate ranking input/result for radial sample `NN` |
| `object_radial_summary.json` | Radial settings, selected direction, termination reason, samples, and best observation |
| `object_refinement_summary.json` | Complete angular/radial run configuration, results, timing, and global winner |
| `object_evaluation_<UTC_RUN_ID>.json` | Immutable per-run metrics, pose/motion evidence, definitions, and current aggregate |
| `object_evaluation_<UTC_RUN_ID>.txt` | Immutable readable matrix and aggregate for the same run |
| `object_evaluation.json` / `.txt` | Stable copies of the latest completed evaluation |
| `object_evaluation_runs.csv` | Append-only matrix rows across repeated runs |

### Angular level records

Each `levels` entry in the refinement summary contains:

| Field | Meaning |
| --- | --- |
| `stage` | `angular_refinement` |
| `level` | Zero-based angular level |
| `candidate_count` | Images successfully captured in this level |
| `triangle_count` | Candidate triangles submitted for the level |
| `skipped_triangle_paths` | Unreachable or motion-failed candidates omitted from ranking |
| `best_triangle_path` | Hierarchical path such as `[3, 2, 1]` |
| `best_score` | Best candidate score in this level, not necessarily the global best |
| `score_improvement_over_previous_global_best` | Signed difference from the incumbent before this level; `null` at level 0 |
| `accepted_as_global_best` | Whether this level strictly exceeded the previous incumbent |
| `meaningful_improvement` | Whether the gain met `min_score_improvement` |
| `global_best_score_after_level` | Incumbent score after evaluating this level |
| `score_gap_to_incumbent` | Non-negative gap used by angular look-ahead |
| `non_improving_levels` | Current consecutive non-meaningful-level count |
| `timing_seconds` | `capture_and_motion`, `ranking`, and `total` wall times |

An unreachable centroid records `fallback_attempts`. A successful alternative
uses `viewpoint_strategy: nearby_in_triangle_fallback`; a triangle exhausted at
both configured blend offsets uses
`skip_reason: unreachable_after_nearby_fallbacks`. These are per-triangle
outcomes, not a reason for the entire round to fail.

### Refinement summary

The top-level summary records all stopping/search parameters, including
angular patience and look-ahead, debug state, feature size, return-home policy,
and radial bounds. Important result fields are:

| Field | Meaning |
| --- | --- |
| `angular_termination_reason` | Why spherical-triangle search stopped |
| `levels_completed` | Number of ranked angular levels |
| `total_images_captured` | Successful angular plus radial captures |
| `levels` | Ordered angular records described above |
| `final_refinement` | Last angular level, which may differ from the global winner |
| `radial_refinement` | Radial result object, or `null` when disabled/not run |
| `evaluation` | Evaluation report embedded after the standalone files are written |
| `final_best` | Highest-scoring observation across angular levels and radial fine alignment |
| `best_view_with_target` | Stable debug comparison path, or `null` when diagnostics were disabled |
| `end_effector_capture_poses` | Absolute JSON capture-state path |
| `end_effector_capture_pose_mesh` | Absolute PLY capture-state path |

Expected angular termination values include
`similarity_threshold_reached`, `maximum_refinement_depth_reached`,
`refinement_stalled`, `angular_candidate_not_promising`,
`angular_lookahead_exhausted`, and `no_reachable_viewpoints`.

### Radial records

Radial `samples` use a signed `radial_step_index`: negative is `inside`
(closer), positive is `outside` (farther). `radial_offset_m` is the signed
distance from the actual angular winner radius. Captured records include score,
actual camera pose, selected joints, motion evidence, and ranking paths;
rejected records include `skipped`, `skip_reason`, and any available failure
counts.

The radial summary reports `base_angular_score`, `base_radius_m`, radius/step
settings, `selected_direction`, `candidate_count`,
`improved_over_angular_best`, `samples`, and `best_observation`. Expected radial
termination values include `no_initial_radial_improvement`,
`initial_probe_local_maximum`, `radial_score_plateau`,
`radial_radius_bound_reached`, `radial_viewpoint_unreachable`,
`radial_motion_planning_failed`, and `maximum_radial_steps_reached`.

### Evaluation metrics

The top-level evaluation object identifies the run through `run_id` and
`timestamp_utc`, stores the `method`, target/final image paths and final
similarity score, and groups the comparison-table values under `metrics`:

| Field | Unit | Definition |
| --- | --- | --- |
| top-level `method` | label | Exact `--method-name` value |
| `metrics.n_views` | count | Successfully saved angular plus radial images |
| `metrics.d_target_m` | metres | Selected final camera-center distance to target-camera ground truth |
| `metrics.d_target_cm` | centimetres | The same target distance in centimetres |
| `metrics.d_traveled_m` | metres | Camera-center endpoint displacement accumulated during the measured run |
| `metrics.d_traveled_cm` | centimetres | The same traveled distance; used by the text/aggregate matrix |
| `metrics.end_effector_distance_traveled_m` | metres | Parallel `ur5e_tool0` endpoint-distance accumulator |
| `metrics.t_exec_s` | seconds | Measured search wall time before evaluation/report/final cleanup work |
| `metrics.psnr_db` | decibels | Full-frame color PSNR with data range 255; `"inf"` for identical images |
| `metrics.ssim` | unitless | Full-frame color Gaussian-window SSIM |
| `metrics.camera_orientation_error_deg` | degrees | Absolute quaternion angular error when both poses include orientation |

`target_pose_ground_truth` is either `null` or contains `position_xyz_m`, an
optional `orientation_xyzw`, and `source`. `--target-camera-position X Y Z`
produces a position-only source. `--target-pose-json FILE` requires a nested
record matching the target by `image_file` or `image_filename`; without that
option, all JSON files beside the target image are searched in filename order.
Supported pose records use `camera_pose_world`, `actual_camera_xyz_m`,
`capture_viewpoint_xyz_m`, or `centroid_xyz_m` with their corresponding
orientation key. If no matching ground truth is found, target distance and
orientation error remain `null`.

`motion_details` stores the initial/latest camera and end-effector positions,
both accumulated distances, and `motion_segments`. Each segment has a label,
endpoint position, and displacement. The camera distance begins immediately
after environment construction and ends after the final-best revisit. It
includes search-time recovery/home endpoints and excludes the final cleanup
return home.

`t_exec_s` uses the same boundary: it starts after environment construction,
then includes backend setup/warm-up, reachability, IK, planning, playback,
capture, ranking, angular/radial search, and the final-best revisit. Evaluation
metric calculation/serialization, refinement-summary writing, final return
home, and disconnect are outside the measurement.

`image_quality_details` records mean squared error, whether the final image was
resized to the target dimensions, target HWC shape, implementation backend,
PSNR, its infinite flag, and SSIM. Metrics use the complete saved color frame,
not the DINO object mask. If both OpenCV and the Pillow/SciPy fallback fail, the
error is stored there and PSNR/SSIM remain `null`.

`aggregate_for_method` is computed after appending the current CSV row. For
`n_views`, `d_target_m`, `d_traveled_cm`, `t_exec_s`, `psnr_db`, and `ssim`, it
stores the finite-value count, mean, and sample standard deviation. A single
finite trial has standard deviation `0.0`; missing and non-finite values do not
contribute. `trial_count` still counts every CSV row with the exact method
label. `report_files` provides absolute paths for the immutable run files,
stable latest files, and CSV ledger.

## Read a report in Python

```python
import json
from pathlib import Path

report = json.loads(
    Path("output/pybullet/reachable_viewpoints.json").read_text()
)

print(report["backend"])
print(report["reachable_count"], report["rejected_count"])

by_index = {record["index"]: record for record in report["viewpoints"]}
for index in report["reachable_indices_ranked_by_quality"]:
    record = by_index[index]
    print(index, record["quality_score"], record["joint_positions_rad"])
```

## Distinguish fixture runs

At the 2026-07-31 documentation pass, the ignored default-named output in this
checkout reports `curobo_v1_legacy_goalset_ik`, roll `0`, and `64/100`
accepted. Its selection metadata confirms effective CuRobo execution. The
ignored file named `reachable_viewpoints_curobo_v1.json` is older and actually
reports `pybullet_multistart_ik`, four rolls, and `84/100`; the filename alone
is therefore not trustworthy.

The separately observed explicit-PyBullet roll-0 baseline is `52/100`. Always
identify a run from `backend`, `backend_metadata.selection`, and
`reachability_config`, not from its filename or an expected acceptance count.

Next: [Testing and Validation](Testing-and-Validation) or
[Troubleshooting](Troubleshooting).
