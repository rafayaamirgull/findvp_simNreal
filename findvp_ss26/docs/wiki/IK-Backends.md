# IK Backends

## Shared candidate-backend contract

Both backends return IK proposals through the same data model. PyBullet then
performs the final endpoint validation.

```python
class IKCandidateBackend(Protocol):
    backend_name: str
    metadata: dict[str, object]

    def solve(
        self,
        problems: Sequence[IKProblem],
        start_joint_positions_rad: Sequence[float],
        reachability_config: ReachabilityConfig,
    ) -> list[IKProblemResult]: ...
```

`IKProblem` contains one sampled camera position and all configured roll goals.
`IKCandidate` identifies a goal index, a six-joint branch, optional seed index,
and provenance. `IKProblemResult` provides all candidates, the finite attempt
count, and optional solver-stage failure counts.

The common filter rejects malformed backend output with an error. It does not
turn bad array shapes, missing indices, wrong joint order, or non-finite joints
into ordinary viewpoint rejections. Its `finally` block restores the call-entry
joint state after successful filtering and after backend or validation errors.

## CLI backend-selection policy

The CLI default is `--ik-backend auto`, not a fixed solver. Automatic selection:

1. preflights exactly CuRobo V1 against the pinned package, driver, Torch CUDA,
   visible device, GPU memory/capability, and a real CUDA tensor operation;
2. uses `artifacts/curobo/ur5e_camera_curobo_v1.yml` when no config is passed;
3. selects CuRobo V1 when preflight succeeds;
4. selects PyBullet only when preflight raises `CuRoboUnavailableError`.

Missing or malformed configuration after a usable preflight, a wrong schema,
backend construction, solver, result-contract, and validation errors are not
fallback conditions. Explicit `curobo` is strict; explicit `pybullet` skips the
CuRobo probe. The report records requested and effective choices under
`backend_metadata.selection`.

## PyBullet multi-start IK

The actual report identity for this backend is `pybullet_multistart_ik`. It is
used by explicit `pybullet` and by an automatic runtime-unavailability fallback.

### Deterministic seeds

Seed 0 is the joint state captured at filter entry. Additional seeds use a
six-dimensional Halton sequence with prime bases:

```text
(2, 3, 5, 7, 11, 13)
```

They are mapped into the central 84% of every joint range:

```text
q_seed = lower + (0.08 + 0.84 * halton_value) * (upper - lower)
```

This avoids starting exactly at hard limits. The default is three seeds per
roll.

### Solver settings

Every roll/seed combination calls PyBullet damped least-squares IK:

| Setting | Value |
| --- | ---: |
| Solver | `IK_DLS` |
| Maximum iterations | 200 |
| PyBullet residual threshold | `1e-5` |
| Joint damping | `0.01` per joint |
| Rest pose | Current deterministic seed |
| Default refinement passes | 1 |

The refinement pass warm-starts another solve from the previous result. It adds
solver work but does not create a separate candidate record.

With the upright roll-only default and three seeds, the finite PyBullet search
reports three attempted branches per viewpoint. An explicit four-roll search
reports 12.

### Equivalent revolutions

For each revolute joint, the code considers the returned value plus or minus up
to three full `2*pi` turns. It keeps an equivalent value inside the joint limits
that is nearest the call's start configuration. This canonicalization occurs
before joint-limit and FK validation.

## CuRobo adapters

`CuRoboIKCandidateBackend` auto-detects the generation from the YAML schema:

| YAML marker | Backend name | Supported package |
| --- | --- | --- |
| top-level `robot_cfg` | `curobo_v1_legacy_goalset_ik` | exactly `nvidia-curobo==0.7.8` |
| top-level `kinematics` | `curobo_v2_batched_goalset_ik` | `nvidia-curobo==0.8.x` |

The package and YAML generation must match. Explicit CuRobo selection fails on
any unavailable runtime or invalid configuration. Automatic mode is limited to
V1 and catches only its typed runtime-unavailability error; it does not catch
configuration, solver, or result errors.

### V1 execution shape

- One camera position is batch `B=1`.
- All configured rolls form one goal set `G`.
- The default 32 seeds are shared across the goal set, not multiplied by `G`.
- Every successful returned branch is forwarded to PyBullet.
- The solver is cached by `(G, K, position tolerance, rotation threshold)`.
- A lock serializes mutation of cached CUDA graph and seed buffers.
- The Halton seed generator is reset before each solve.
- A list of 100 fixture points becomes 100 sequential `B=1` calls.

V1 performs no CuRobo collision checking. It proposes IK branches only.

### V2 execution shape

- Submitted viewpoints are batch `B`.
- Rolls remain goal set `G`.
- `InverseKinematics.solve_pose` handles the batch.
- V2 loads fitted robot collision spheres, enables self-collision, and uses a
  scene containing table cuboids and the workpiece mesh.
- All successful V2 branches are still revalidated in PyBullet.

See [CuRobo Setup](CuRobo-Setup) for environments, generated models, and
commands.

## Common endpoint tolerances

Backend solver convergence is not enough. PyBullet independently applies:

| Gate | Default |
| --- | ---: |
| Optical-center position error | `<= 0.001 m` |
| Optical-frame orientation error | `<= 1.0 degree` |
| Collision clearance | `>= 0.005 m` |
| Object-center line of sight | required |

CuRobo solver residuals remain in `solver_provenance`; accepted report fields
such as `position_error_m`, `orientation_error_deg`, and `clearance_m` come from
the common PyBullet stage.

## Custom backends

A new backend should:

1. implement the protocol above;
2. preserve submitted viewpoint indices;
3. return branches in `ARM_JOINT_ORDER`, or explicitly normalize its own order;
4. return only finite joint values;
5. retain roll selection in `goal_index`;
6. describe solver-specific information in `metadata` and `solver_provenance`;
7. report failures honestly without claiming mathematical unreachability;
8. allow every branch to pass through the existing PyBullet validator.

The integration suite contains a fake external backend test that exercises this
shared path.

Next: [CuRobo Setup](CuRobo-Setup) or
[Level 6 Reachability Pipeline](Level-6-Reachability-Pipeline).
