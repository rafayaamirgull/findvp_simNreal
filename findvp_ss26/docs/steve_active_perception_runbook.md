# Steve: full FindVP active-perception search (real robot)

This is the real-robot version of

```bash
python3 sample/active_perception_triangles.py --views 3 --name object \
  --target-image /abs/target.png --similarity-threshold 0.95 --max-refinement-depth 4
```

The simulator script cannot drive Steve because it owns a PyBullet world.
[`real_robot/steve_active_perception.py`](../real_robot/steve_active_perception.py)
runs the same search algorithm with the same flags and defaults. It moves
the arm with the chain from the [three-view runbook](steve_three_view_runbook.md):
live `/joint_states` → CuRobo server → one `FollowJointTrajectory` goal, ×8
slower, projected-box masks, and a typed approval before every move.

| Stage | What happens | Stops when |
|---|---|---|
| level 0 | the 3 hemisphere triangles (60°/180°/300°, r 0.40 m) are captured and ranked with SIFT + DINOv3 | – |
| level 1…4 | the winning spherical triangle is split into 3 children, which are captured and ranked | score ≥ `--similarity-threshold`, depth = `--max-refinement-depth`, no gain ≥ 0.002 for `--refinement-patience` (2) levels, or the look-ahead rule |
| radial | ±1, ±2 steps of 2 cm along the best view's ray, then continues in the better direction | no gain / plateau for 2 steps / r ∉ [0.20, 0.60] m / unreachable |
| final | returns to the best image of the whole search and saves `final.png` | – |

Unreachable views are handled as in the simulator (nearby in-triangle points at
blend 0.20 / 0.40), with the three-view run's fallback radii (0.45, 0.50 m).
A triangle for which none of these plans is skipped.

**Expect many moves.** With depth 4 and radial refinement enabled: 3 + 4×3
triangle views, then 1–13 radial moves, plus the return: roughly **16–30
approved moves**. On real images the score is unlikely to reach 0.95 (the box
run `three_01` scored 0.81 at level 0). The search therefore normally ends by
depth or patience.

## 1. Prepare: identical to the three-view runbook

Do sections **1–4** of the [three-view runbook](steve_three_view_runbook.md):
terminals H/A/B/C, pendant, `check` (must pass and the overlay must sit on the
box), and `scene.json`/`locate`. Before section 5, set the active run directory
in terminal C:

```bash
export AP=/home/ws/findvp_ss26/real_robot/steve_active_perception.py
export RUN=/home/ws/findvp_runs/active_$(date +%Y%m%d_%H%M%S)
echo "$RUN"
```

Now do section **5** of the three-view runbook: `goto` a target view and run
`python3 "$FV" target --run-dir "$RUN"`. This writes `$RUN/target.png`,
`target_mask.png` and `target.json`.
To reuse a still-valid target from an earlier run instead, pass its path as
`--target-image`. The script uses the `<stem>_mask.png` and `<stem>.json`
files next to that image.

## 2. Preview level 0 (nothing moves)

```bash
python3 "$AP" --run-dir "$RUN" --plan-only
```

Every one of the three triangles must show a plan. Later levels cannot be
previewed because they depend on the scores. Their plans are shown one by one
during the run.

## 3. Run it

```bash
python3 "$AP" --run-dir "$RUN" \
  --views 3 --name object \
  --target-image "$RUN/target.png" \
  --similarity-threshold 0.95 \
  --max-refinement-depth 4
```

Before **each** move it prints the plan, draws the path in RViz and asks:

| You type | Effect |
|---|---|
| `go` | execute this move on the real arm |
| `skip` | do not move; this view counts as unreachable, and the search continues |
| anything else | stop the whole search; the arm stays, and results so far are saved |

A failed motion or capture also stops the search without any further motion.
After each level it prints the ranking with the angle of every view to the
target camera, and the level's decision (`refine`, `look-ahead`, or the stop
reason).

Useful options (all defaults as in the simulator):

* `--skip-radial-refinement`: angular search only
* `--radial-min-radius 0.30`: keep the camera farther from the box during the radial search
* `--refinement-patience 0`: never stop early; always go to the maximum depth
* `--until-convergence`: stop when the best score plateaus (no threshold/depth, no radial)
* `--debug`: also saves FindVP's DINOv3/SIFT match visualisations for every ranking
* `--feature-max-image-size 448`: uses the simulator's DINOv3 resolution; the
  default of 640 matches `steve_three_view.py`

## 4. Results: `$RUN/active_perception/`

| File | Content |
|---|---|
| `summary.png` | target + every captured image with its score (best in gold) |
| `views/object_LNN_T<path>.png` (+ `_mask`, `_overlay`) | triangle captures, e.g. `object_L02_T1-3-2.png` |
| `views/object_radial_{inside,outside}_NN.png` | radial captures |
| `level_NN_ranking.json`, `radial_eval_NN_ranking.json` | FindVP's full ranking reports |
| `result.json` | levels, decisions, radial search, best observation, stop reason, return-to-best |
| `final.png` | image after returning to the best view |

Quick check:

```bash
python3 -c 'import json,os; r=json.load(open(os.path.join(os.environ["RUN"],"active_perception/result.json"))); print("status:",r["status"],"| stop:",r["termination_reason"],"| radial:",r["radial_termination_reason"],"| images:",r["capture_count"],"| best:",r["best_observation"]["label"],round(r["best_observation"]["score"],3),"| return:",r["return_to_best"]["status"])'
```

Open `summary.png` and the `_overlay.png` of the winning view. The green
outline must lie on the box in every image that was scored. The same
limitations as in the three-view run apply. Scores compare these captures
with each other and are not probabilities. The box pose, calibration and TF
must stay unchanged from the moment the target is taken.
