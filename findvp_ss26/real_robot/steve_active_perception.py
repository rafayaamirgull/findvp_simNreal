#!/usr/bin/env python3
"""FindVP's full active-perception search on the real Steve robot.

Real-robot port of ``sample/active_perception_triangles.py`` (triangle strategy):

    level 0    the --views hemisphere triangles around the carton, SIFT + DINOv3 ranking
    level 1..  split the winning spherical triangle into 3 children, capture, rank again;
               stop on --similarity-threshold, --max-refinement-depth, --refinement-patience
               or the angular look-ahead rule, exactly as in the simulator
    radial     closer/farther steps along the best view's ray  (--skip-radial-refinement)
    final      return to the best observation of the whole search, save final.png

Unreachable triangle views are handled like the simulator (nearby in-triangle points at
blend 0.20 / 0.40), plus the fallback radii of scene.json that the three-view run uses.

Motion, masks and safety are those of steve_three_view.py: same scene.json, hand-eye and
CuRobo server, plans from the live joints, and every move waits for the operator:
    go   = execute on the real arm      skip = skip this view      anything else = stop
Prepare with steve_three_view.py check / locate / goto / target first
(docs/steve_active_perception_runbook.md).
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

import steve_three_view as stv  # noqa: E402  (also puts findvp_ss26 on sys.path)
from geometry import (  # noqa: E402
    CAMERA_FRAME,
    DEFAULT_HANDEYE,
    DEFAULT_SCENE,
    camera_to_tool0,
    direction_angle_deg,
    load_json,
    look_at_matrix,
    pose_to_matrix,
    scale_view,
    write_json,
)

# Same fallback as active_perception_triangles.py: points from the centroid towards each vertex.
TRIANGLE_FALLBACK_BLEND_FRACTIONS = (0.20, 0.40)


class StopSearch(Exception):
    """The operator stopped, or a motion/capture failed: the arm is not moved again."""


def path_label(path) -> str:
    return ".".join(str(int(v)) for v in path)


# =========================================================================== motion


def ask_operator(prompt: str) -> str:
    if not sys.stdin.isatty():
        print("! refusing: 'go' must be typed by the operator on a terminal (stdin is not a TTY)")
        return "stop"
    try:
        answer = input(prompt).strip()
    except EOFError:
        return "stop"
    return answer if answer in ("go", "skip") else "stop"


def move(ctx, robot, plan, start, label) -> bool:
    """True = moved, False = the operator skipped it. Raises StopSearch otherwise."""

    stv.review(plan, start, ctx.args.slow_factor, label)
    robot.publish_markers(ctx.scene, ctx.marker_views, target=ctx.marker_target, path=plan.get("tool0_path"))
    answer = ask_operator("  Watch RViz + the arm. 'go' = execute on the REAL arm, 'skip' = skip this view, "
                          "anything else = stop the search: ")
    if answer == "skip":
        print("  skipped by the operator; the arm was not moved.")
        return False
    if answer != "go":
        raise StopSearch("operator_stopped")
    ok, why = robot.execute(plan, ctx.args.slow_factor)
    print(("  done: " if ok else "  ! FAILED: ") + why)
    if not ok:
        raise StopSearch(f"motion_failed: {why}")
    return True


def go_to_joints(ctx, robot, joints, label) -> bool:
    q = robot.joints()
    if np.max(np.abs(q - np.asarray(joints, dtype=float))) < 0.01:
        print(f"  already at {label}")
        return True
    plan = ctx.planner.plan_joints(q, joints, ctx.world)
    if not plan.get("success"):
        print(f"  ! could not plan {label}: {plan.get('status')}")
        return False
    return move(ctx, robot, plan, q, label)


def capture(ctx, robot, views_dir: Path, stem: str, caption: str, T_planned) -> dict:
    try:
        shot = robot.capture(CAMERA_FRAME)
    except RuntimeError as exc:
        raise StopSearch(f"capture_failed: {exc}") from exc
    record = stv.save_capture(ctx, shot, views_dir, stem, caption)
    record["camera_position_error_mm"] = float(
        1000 * np.linalg.norm(shot["T_base_camera"][:3, 3] - np.asarray(T_planned)[:3, 3]))
    print(f"  captured {views_dir / record['image_filename']}  (camera {record['camera_position_error_mm']:.1f} mm "
          f"from the plan, r {record['radius_m']:.3f} m)")
    return record


# =========================================================================== triangle views


def nearby_in_triangle_views(triangle, center, radius_m, blend):
    """FindVP's fallback: three shell points from the centroid towards each vertex."""

    centroid = np.asarray(triangle["centroid_xyz_m"], dtype=float) - center
    centroid /= np.linalg.norm(centroid)
    views = []
    for vertex in np.asarray(triangle["vertices_xyz_m"], dtype=float):
        direction = (vertex - center) / np.linalg.norm(vertex - center)
        direction = (1.0 - blend) * centroid + blend * direction
        direction /= np.linalg.norm(direction)
        views.append(look_at_matrix(center + radius_m * direction, center))
    return views


def plan_triangle(ctx, start, triangle, radius, radii):
    """Plan to the triangle's centroid view, else to a nearby in-triangle view; each at every radius."""

    center = ctx.center
    options = [("triangle_centroid", None, look_at_matrix(triangle["centroid_xyz_m"], center))]
    for blend in TRIANGLE_FALLBACK_BLEND_FRACTIONS:
        options += [("nearby_in_triangle_fallback", blend, T)
                    for T in nearby_in_triangle_views(triangle, center, radius, blend)]
    last_status = None
    for strategy, blend, T_nominal in options:
        for r in radii:
            T = scale_view(T_nominal, center, r)
            plan = ctx.planner.plan_pose(start, camera_to_tool0(T, ctx.T_tool0_camera), ctx.world)
            if plan.get("success"):
                return plan, T, {
                    "viewpoint_strategy": strategy,
                    "fallback_blend_fraction": blend,
                    "fallback_angular_offset_deg": direction_angle_deg(T[:3, 3], triangle["centroid_xyz_m"], center),
                    "planned_radius_m": float(r),
                }
            last_status = plan.get("status")
    return None, None, {"skip_reason": "unreachable_after_nearby_fallbacks", "last_planner_status": last_status}


# =========================================================================== scoring


class Scorer:
    """DINOv3 + SIFT (FindVP's rank_triangle_views) with the target prepared once."""

    def __init__(self, args, target_path: Path, mask_path: Path):
        from PIL import Image

        from sample.dinov3_feature_matching import DINOv3FeatureExtractor
        from sample.target_guided_similarity import prepare_target_image

        self.args = args
        print(f"loading DINOv3 (CPU, max image side {args.feature_max_image_size}) ...", flush=True)
        self.extractor = DINOv3FeatureExtractor(max_image_size=args.feature_max_image_size,
                                                upscale_small_images=True)
        self.clusterer = None
        if args.debug:
            from sample.dinov3_spatial_clustering import DINOv3SpatialClusterer

            self.clusterer = DINOv3SpatialClusterer()
        with Image.open(mask_path) as m:
            mask = (np.asarray(m.convert("L")) > 0).astype(np.uint8) * 255
        self.target_path = target_path
        self.prepared = prepare_target_image(target_path, extractor=self.extractor, debug=args.debug,
                                             clusterer=self.clusterer, target_mask=mask)
        print(f"target ready: {target_path} ({len(self.prepared.object_features.descriptors)} DINOv3 object patches)")

    def rank(self, records, viewpoints_json: Path, output_json: Path) -> dict:
        from sample.target_guided_similarity import rank_triangle_views

        write_json(viewpoints_json, records)
        report = rank_triangle_views(
            self.target_path, viewpoints_json, output_path=output_json,
            ambiguity_margin=self.args.ambiguity_margin, dino_weight=self.args.dino_weight,
            sift_weight=self.args.sift_weight, debug=self.args.debug, extractor=self.extractor,
            clusterer=self.clusterer, prepared_target=self.prepared,
        )
        by_id = {r["triangle"]: r for r in records}
        for item in report["ranking"]:
            by_id[item["triangle"]].update({k: item[k] for k in ("score", "dino_score", "sift_score", "sift_inliers")})
        return report


def angle_to_target(record, target_record):
    if not target_record:
        return None
    return direction_angle_deg(record["camera_position_m"], target_record["camera_position_m"],
                               target_record["scene"]["object"]["center_xyz_m"])


def print_ranking(report, records, target_record):
    by_id = {r["triangle"]: r for r in records}
    print(f"\n{'rank':>4} {'view':>9} {'score':>7} {'DINOv3':>7} {'SIFT':>6} {'inliers':>7}   camera az/el/r"
          + ("     angle to target" if target_record else ""))
    for rank, item in enumerate(report["ranking"], 1):
        r = by_id[item["triangle"]]
        line = (f"{rank:>4} {r['label']:>9} {item['score']:7.3f} {item['dino_score']:7.3f} "
                f"{item['sift_score']:6.3f} {item['sift_inliers']:7d}   "
                f"{r['azimuth_deg']:4.0f} / {r['elevation_deg']:3.0f} / {r['radius_m']:.3f}")
        angle = angle_to_target(r, target_record)
        if angle is not None:
            line += f"   {angle:5.1f} deg"
        print(line)
    for failure in report["failures"]:
        print(f"   view {failure['triangle']}: could not be scored ({failure['error']})")


# =========================================================================== search


class SearchState:
    def __init__(self, out: Path, target_record):
        self.out = out
        self.views_dir = out / "views"
        self.target_record = target_record
        self.levels: list[dict] = []    # FindVP's level_results, one per scored level
        self.captures: list[dict] = []  # every saved image, in capture order
        self.radial: dict | None = None
        self.best: dict | None = None   # best scored capture so far
        self.termination = "not_started"

    def best_marker(self):
        if self.best is None:
            return []
        T = pose_to_matrix(self.best["camera_position_m"], self.best["camera_quaternion_xyzw"])
        return [{"label": f"best {self.best['label']}", "T_base_camera": T, "state": "best"}]


def capture_level(ctx, robot, st: SearchState, triangles, level, radius, radii):
    name = ctx.args.name
    records = []
    ctx.marker_views = st.best_marker() + [
        {"label": f"L{level} {path_label(t['triangle_path'])}",
         "T_base_camera": look_at_matrix(t["centroid_xyz_m"], ctx.center), "state": "planned"}
        for t in triangles]
    first = len(ctx.marker_views) - len(triangles)
    for i, triangle in enumerate(triangles):
        label = path_label(triangle["triangle_path"])
        marker = ctx.marker_views[first + i]
        stv.banner(f"LEVEL {level}  view {i + 1}/{len(triangles)}  ->  triangle {label}")
        q = robot.joints()
        plan, T, meta = plan_triangle(ctx, q, triangle, radius, radii)
        triangle.update(meta)
        if T is None:
            print(f"  ! triangle {label}: neither the centroid nor a nearby in-triangle view is reachable "
                  f"at radii {radii} ({meta['last_planner_status']}); skipping it")
            marker["state"] = "failed"
            continue
        if meta["viewpoint_strategy"] != "triangle_centroid":
            print(f"  centroid unreachable: using a nearby in-triangle view "
                  f"{meta['fallback_angular_offset_deg']:.1f} deg from it (blend {meta['fallback_blend_fraction']:.2f})")
        marker["T_base_camera"] = T
        if not move(ctx, robot, plan, q, f"L{level} triangle {label} at radius {meta['planned_radius_m']:.2f} m"):
            triangle["skip_reason"] = "operator_skipped"
            marker["state"] = "failed"
            continue
        record = capture(ctx, robot, st.views_dir, f"{name}_L{level:02d}_T{label.replace('.', '-')}",
                         f"level {level}  triangle {label}", T)
        record.update({"triangle": i + 1, "label": f"L{level} {label}", "stage": "angular_refinement",
                       "refinement_level": level, "triangle_path": triangle["triangle_path"],
                       "triangle_record": triangle, **meta})
        records.append(record)
        st.captures.append(record)
        marker["state"] = "captured"
    return records


def angular_search(ctx, robot, scorer: Scorer, st: SearchState, triangles, radius, radii):
    """The level loop of active_perception_triangles.run(), with the same stopping rules."""

    from findvp_ros.triangle_search import subdivide_spherical_triangle

    args = ctx.args
    level = 0
    best_score_seen = -math.inf
    non_improving_levels = 0
    lookahead_levels_used = 0
    while True:
        records = capture_level(ctx, robot, st, triangles, level, radius, radii)
        if not records:
            st.termination = "no_reachable_viewpoints"
            print("No triangle of this level could be captured; finishing with the best earlier observation.")
            return
        stv.banner(f"LEVEL {level}: scoring {len(records)} image(s) with SIFT + DINOv3")
        prefix = f"level_{level:02d}"
        started = time.perf_counter()
        report = scorer.rank(records, st.views_dir / f"{prefix}_viewpoints.json", st.out / f"{prefix}_ranking.json")
        ranking_seconds = time.perf_counter() - started
        print_ranking(report, records, st.target_record)
        best_record = next(r for r in records if r["triangle"] == report["best_triangle"])
        best_score = float(report["ranking"][0]["score"])

        previous = best_score_seen
        improvement = None if not math.isfinite(previous) else best_score - previous
        meaningful = not math.isfinite(previous) or best_score >= previous + args.min_score_improvement
        accepted = not math.isfinite(previous) or best_score > previous
        best_score_seen = max(best_score_seen, best_score)
        if meaningful:
            non_improving_levels = 0
            lookahead_levels_used = 0
        else:
            non_improving_levels += 1
        gap = max(0.0, best_score_seen - best_score)
        if accepted:
            st.best = best_record
        st.levels.append({
            "stage": "angular_refinement",
            "level": level,
            "candidate_count": len(records),
            "triangle_count": len(triangles),
            "skipped_triangle_paths": [t["triangle_path"] for t in triangles if "skip_reason" in t],
            "ranking_file": report["report_file"],
            "best_triangle_path": best_record["triangle_path"],
            "best_image": report["best_image"],
            "best_score": best_score,
            "plausible_triangles": report["plausible_triangles"],
            "score_improvement_over_previous_global_best": improvement,
            "accepted_as_global_best": accepted,
            "meaningful_improvement": meaningful,
            "global_best_score_after_level": best_score_seen,
            "score_gap_to_incumbent": gap,
            "non_improving_levels": non_improving_levels,
            "ranking_seconds": ranking_seconds,
            "best_record": best_record,
        })
        print(f"\nLevel {level} winner: triangle path {path_label(best_record['triangle_path'])}, "
              f"candidate={best_score:.3f}, global={best_score_seen:.3f}, "
              f"accepted={'yes' if accepted else 'no'} (ranking {ranking_seconds:.1f} s)")

        should_refine = True
        if args.until_convergence:
            if non_improving_levels >= args.convergence_patience:
                st.termination, should_refine = "similarity_converged", False
                print(f"Stopping: best similarity plateaued for {non_improving_levels} level(s); "
                      f"keeping score={best_score_seen:.3f}.")
        elif best_score_seen >= args.similarity_threshold:
            st.termination, should_refine = "similarity_threshold_reached", False
            print(f"Stopping: similarity {best_score_seen:.3f} >= threshold {args.similarity_threshold:.3f}.")
        elif level >= args.max_refinement_depth:
            st.termination, should_refine = "maximum_refinement_depth_reached", False
            print(f"Stopping: maximum refinement depth {args.max_refinement_depth} reached.")
        elif args.refinement_patience > 0 and non_improving_levels >= args.refinement_patience:
            st.termination, should_refine = "refinement_stalled", False
            print(f"Stopping refinement: no global-best improvement of at least {args.min_score_improvement:.4f} "
                  f"for {non_improving_levels} consecutive level(s).")
        elif not meaningful:
            if gap > args.angular_lookahead_margin:
                st.termination, should_refine = "angular_candidate_not_promising", False
                print(f"Stopping angular refinement: the current candidate is {gap:.3f} below the incumbent, "
                      f"beyond the {args.angular_lookahead_margin:.3f} look-ahead margin.")
            elif lookahead_levels_used >= args.angular_lookahead_levels:
                st.termination, should_refine = "angular_lookahead_exhausted", False
                print("Stopping angular refinement: the ambiguous-branch look-ahead budget is exhausted.")
            else:
                lookahead_levels_used += 1
                print(f"Candidate did not improve the incumbent, but remains within "
                      f"{args.angular_lookahead_margin:.3f}; allowing look-ahead "
                      f"{lookahead_levels_used}/{args.angular_lookahead_levels}.")
        if not should_refine:
            return
        triangles, _ = subdivide_spherical_triangle(best_record["triangle_record"], ctx.center, radius)
        level += 1
        print(f"\nRefining triangle {path_label(best_record['triangle_path'])} -> "
              + ", ".join(path_label(t["triangle_path"]) for t in triangles))


def radial_refinement(ctx, robot, scorer: Scorer, st: SearchState):
    """Closer/farther search along the best angular view's ray (FindVP's _run_radial_refinement)."""

    args = ctx.args
    angular_best = max(st.levels, key=lambda item: float(item["best_score"]))
    base = angular_best["best_record"]
    angular_score = float(angular_best["best_score"])
    center = ctx.center
    ray = np.asarray(base["camera_position_m"], dtype=float) - center
    base_radius = float(np.linalg.norm(ray))
    ray /= base_radius
    samples: list[dict] = []
    result = {
        "enabled": True, "base_angular_score": angular_score, "base_radius_m": base_radius,
        "base_label": base["label"], "step_size_m": args.radial_step, "minimum_radius_m": args.radial_min_radius,
        "maximum_radius_m": args.radial_max_radius, "initial_probe_steps_per_direction": args.radial_probe_steps,
        "selected_direction": None, "termination_reason": "initial_probes_complete", "samples": samples,
        "best_sample": None, "improved_over_angular_best": False,
    }
    st.radial = result

    stv.banner(f"RADIAL fine alignment from {base['label']} (score {angular_score:.3f}, r {base_radius:.3f} m)")
    ctx.marker_views = st.best_marker()
    if not go_to_joints(ctx, robot, base["joints_rad"], f"the angular best {base['label']}"):
        result["termination_reason"] = "could_not_return_to_angular_best"
        return result

    def visit(step, *, continuation):
        """Plan, move, capture and score one radial step. Returns the sample or None if not captured."""

        radius = base_radius + step * args.radial_step
        direction = "inside" if step < 0 else "outside"
        sample = {"triangle": len(samples) + 1, "label": f"R{step:+d}", "stage": "radial_refinement",
                  "radial_direction": direction, "radial_step_index": step,
                  "radial_offset_m": step * args.radial_step, "radius_target_m": radius}
        samples.append(sample)
        if not args.radial_min_radius <= radius <= args.radial_max_radius:
            sample["skip_reason"] = "outside_radial_radius_bounds"
            return None
        stv.banner(f"RADIAL {direction} step {abs(step)}{' (continuation)' if continuation else ''}: "
                   f"r {radius:.3f} m ({step * args.radial_step:+.3f} m)")
        T = look_at_matrix(center + radius * ray, center)
        ctx.marker_views = st.best_marker() + [{"label": sample["label"], "T_base_camera": T, "state": "planned"}]
        q = robot.joints()
        plan = ctx.planner.plan_pose(q, camera_to_tool0(T, ctx.T_tool0_camera), ctx.world)
        if not plan.get("success"):
            print(f"  ! unreachable: {plan.get('status')}")
            sample["skip_reason"] = f"planning_failed: {plan.get('status')}"
            return None
        if not move(ctx, robot, plan, q, f"radial {sample['label']} at radius {radius:.3f} m"):
            sample["skip_reason"] = "operator_skipped"
            return None
        record = capture(ctx, robot, st.views_dir, f"{args.name}_radial_{direction}_{abs(step):02d}",
                         f"radial {sample['label']}  r {radius:.3f} m", T)
        sample.update(record)
        st.captures.append(sample)
        scorer.rank([sample], st.views_dir / f"radial_eval_{sample['triangle']:02d}.json",
                    st.out / f"radial_eval_{sample['triangle']:02d}_ranking.json")
        print(f"  radial {sample['label']} score={sample['score']:.3f}  (angular best {angular_score:.3f})")
        return sample

    for step in (*range(-1, -args.radial_probe_steps - 1, -1), *range(1, args.radial_probe_steps + 1)):
        visit(step, continuation=False)

    scored = [s for s in samples if "score" in s]
    best_sample = max(scored, key=lambda s: float(s["score"])) if scored else None
    improving = [s for s in scored if float(s["score"]) >= angular_score + args.radial_min_improvement]
    seed = None
    if improving:
        highest = max(float(s["score"]) for s in improving)
        seed = min((s for s in improving if highest - float(s["score"]) <= args.radial_min_improvement),
                   key=lambda s: abs(int(s["radial_step_index"])))
    if seed is None:
        result["termination_reason"] = "no_initial_radial_improvement"
    else:
        direction = seed["radial_direction"]
        result["selected_direction"] = direction
        sign = -1 if direction == "inside" else 1
        if abs(int(seed["radial_step_index"])) < args.radial_probe_steps:
            # An already-tested following step did not win: the maximum lies inside the probes.
            result["termination_reason"] = "initial_probe_local_maximum"
        else:
            significant = float(seed["score"])
            plateau = 0
            step = int(seed["radial_step_index"]) + sign
            result["termination_reason"] = "maximum_radial_steps_reached"
            while abs(step) <= args.radial_max_steps:
                sample = visit(step, continuation=True)
                if sample is None:
                    reason = samples[-1]["skip_reason"]
                    result["termination_reason"] = ("radial_radius_bound_reached"
                                                    if reason == "outside_radial_radius_bounds"
                                                    else "radial_viewpoint_unreachable_or_skipped")
                    break
                if float(sample["score"]) >= significant + args.radial_min_improvement:
                    significant, plateau = float(sample["score"]), 0
                else:
                    plateau += 1
                if float(sample["score"]) > float(best_sample["score"]):
                    best_sample = sample
                if plateau >= args.radial_patience:
                    result["termination_reason"] = "radial_score_plateau"
                    break
                step += sign

    result["best_sample"] = best_sample
    result["improved_over_angular_best"] = best_sample is not None and float(best_sample["score"]) > angular_score
    print(f"\nRadial termination: {result['termination_reason']}; best radial "
          + (f"{best_sample['label']} score={best_sample['score']:.3f}" if best_sample else "none")
          + f" vs angular {angular_score:.3f}")
    return result


# =========================================================================== reporting


def save_contact_sheet(out: Path, target_path: Path, captures, best):
    from PIL import Image, ImageDraw

    overlay = target_path.with_name(f"{target_path.stem}_overlay.png")
    tiles = [("TARGET", overlay if overlay.exists() else target_path, (255, 0, 255))]
    for c in captures:
        text = f"{c['label']}  " + (f"{c['score']:.3f}" if "score" in c else "not scored")
        is_best = best is not None and c["image_filename"] == best["image_filename"]
        tiles.append((text + ("  BEST" if is_best else ""), out / "views" / c["overlay"],
                      (255, 200, 0) if is_best else (80, 80, 80)))
    columns, w = min(5, len(tiles)), 320
    with Image.open(tiles[0][1]) as first:
        h = int(first.height * w / first.width)
    rows = math.ceil(len(tiles) / columns)
    canvas = Image.new("RGB", (w * columns, (h + 26) * rows), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, (text, path, color) in enumerate(tiles):
        x, y = (i % columns) * w, (i // columns) * (h + 26)
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB").resize((w, h)), (x, y + 26))
        draw.rectangle([x, y + 26, x + w - 1, y + 26 + h - 1], outline=color, width=4)
        draw.text((x + 6, y + 6), text, fill=(255, 255, 255))
    canvas.save(out / "summary.png")
    return out / "summary.png"


def slim(record):
    """A capture record without the nested triangle (kept in the level files)."""

    return {k: v for k, v in record.items() if k != "triangle_record"} if record else None


# =========================================================================== commands


def resolve_target(args):
    run_dir = Path(args.run_dir)
    target_path = Path(args.target_image or run_dir / "target.png").expanduser().resolve()
    mask_path = Path(args.target_mask).expanduser() if args.target_mask else \
        target_path.with_name(f"{target_path.stem}_mask.png")
    target_json = target_path.with_name(f"{target_path.stem}.json")
    return target_path, mask_path, (load_json(target_json) if target_json.is_file() else None)


def plan_preview(ctx, triangles, radius, radii) -> int:
    robot = ctx.connect()
    q = robot.joints()
    ctx.marker_views = [{"label": f"L0 {t['triangle']}", "T_base_camera": look_at_matrix(t["centroid_xyz_m"],
                                                                                         ctx.center)}
                        for t in triangles]
    for i, triangle in enumerate(triangles):
        print(f"\nplan {i + 1}/{len(triangles)} -> level-0 triangle {triangle['triangle']} "
              f"(from {'the live pose' if i == 0 else 'the previous view'})")
        plan, T, meta = plan_triangle(ctx, q, triangle, radius, radii)
        if T is None:
            print(f"  ! unreachable, also with nearby in-triangle views ({meta['last_planner_status']})")
            ctx.marker_views[i]["state"] = "failed"
            continue
        stv.review(plan, q, ctx.args.slow_factor,
                   f"triangle {triangle['triangle']} ({meta['viewpoint_strategy']}, radius {meta['planned_radius_m']:.2f} m)")
        ctx.marker_views[i]["T_base_camera"] = T
        q = np.asarray(plan["positions"][-1])
        robot.publish_markers(ctx.scene, ctx.marker_views, target=ctx.marker_target, path=plan["tool0_path"])
        robot.spin_for(2.0)
    print("\nplan-only finished: nothing moved. Only level 0 can be previewed: the later levels depend on "
          "the images' scores.")
    return 0


def cmd_search(ctx) -> int:
    from findvp_ros.triangle_search import hemisphere_triangles

    args = ctx.args
    run_dir = Path(args.run_dir)
    out = run_dir / "active_perception"
    search = ctx.scene["search"]
    offset = args.azimuth_offset if args.azimuth_offset is not None else search.get("azimuth_offset_deg", 0.0)
    radius = args.radius if args.radius is not None else search.get("radius_m", 0.40)
    radii = [radius] + [r for r in search.get("fallback_radii_m", []) if r > radius]
    triangles, _ = hemisphere_triangles(ctx.center, radius, args.views, offset)
    target_path, mask_path, target_record = resolve_target(args)
    ctx.marker_target = (pose_to_matrix(target_record["camera_position_m"], target_record["camera_quaternion_xyzw"])
                         if target_record else None)

    stv.banner(f"ACTIVE PERCEPTION {'(PLAN ONLY - nothing moves)' if args.plan_only else '(REAL ARM)'}  ->  {out}")
    print(f"{args.views} level-0 triangles, radius {radius:.2f} m (fallback {radii[1:]}), azimuth offset {offset:g} deg")
    if args.until_convergence:
        print(f"stop: best score gains < {args.min_score_improvement} for {args.convergence_patience} level(s); "
              "no radial refinement")
    else:
        print(f"stop: score >= {args.similarity_threshold}, depth {args.max_refinement_depth}, "
              f"patience {args.refinement_patience}, look-ahead {args.angular_lookahead_levels} "
              f"(margin {args.angular_lookahead_margin}); radial: "
              + ("off" if args.skip_radial_refinement else
                 f"{args.radial_step:.3f} m steps, {args.radial_probe_steps} probes each way, "
                 f"r in [{args.radial_min_radius}, {args.radial_max_radius}] m"))
    if args.plan_only:
        return plan_preview(ctx, triangles, radius, radii)

    if out.exists():
        print(f"! {out} already exists; use a new --run-dir (pass --target-image to reuse a still-valid target)")
        return 1
    for path in (target_path, mask_path):
        if not path.is_file():
            print(f"! missing {path}; take the target with 'steve_three_view.py target' first")
            return 1
    if target_record is None:
        print("  (no target.json next to the target image: angles to the target will not be reported)")
    scorer = Scorer(args, target_path, mask_path)

    health = ctx.planner.health()
    print(f"planner: {ctx.planner.url} ({health['robot']})")
    robot = ctx.connect()
    ok, why = robot.motion_ready()
    print(f"robot: {why}")
    if not ok:
        print("! robot not ready for motion (see 'check'); nothing was moved")
        return 1
    check = ctx.planner.check(robot.joints())
    if not check["valid"]:
        print(f"! the CURRENT arm pose is in collision in the planning world ({check['status']}).")
        return 3

    st = SearchState(out, target_record)
    stopped = None
    started = time.perf_counter()
    try:
        angular_search(ctx, robot, scorer, st, triangles, radius, radii)
        if st.levels and not args.skip_radial_refinement and not args.until_convergence:
            radial_refinement(ctx, robot, scorer, st)
    except StopSearch as exc:
        stopped = str(exc)
        print(f"\n! search stopped ({stopped}); the arm stays where it is. Results so far are saved.")

    # ------------------------------------------------------------- best observation of the whole search
    candidates = [level["best_record"] for level in st.levels]
    if st.radial and st.radial.get("best_sample"):
        candidates.append(st.radial["best_sample"])
    best = max(candidates, key=lambda r: float(r["score"])) if candidates else None
    final = {"status": "not_moved"}
    if best is None:
        print("! no image was scored")
    else:
        stv.banner(f"BEST OBSERVATION: {best['label']}  score {best['score']:.3f}")
        angle = angle_to_target(best, target_record)
        print(f"image {st.views_dir / best['image_filename']}   camera az {best['azimuth_deg']:.0f} "
              f"el {best['elevation_deg']:.0f} r {best['radius_m']:.3f} m"
              + (f"   {angle:.1f} deg from the target camera" if angle is not None else ""))
        if stopped:
            print("  not returning to it: the search was stopped")
        else:
            ctx.marker_views = [{"label": f"best {best['label']}", "state": "best",
                                 "T_base_camera": pose_to_matrix(best["camera_position_m"],
                                                                 best["camera_quaternion_xyzw"])}]
            try:
                reached = go_to_joints(ctx, robot, best["joints_rad"], f"the best view {best['label']}")
            except StopSearch as exc:
                reached, stopped = False, str(exc)
            final = {"status": "reached" if reached else "not_reached"}
            if reached:
                try:
                    record = capture(ctx, robot, out, "final", f"FINAL at best view {best['label']}",
                                     pose_to_matrix(best["camera_position_m"], best["camera_quaternion_xyzw"]))
                except StopSearch as exc:
                    record, final["capture_error"] = None, str(exc)
                    print(f"  ! {exc}")
            if reached and record is not None:
                final["camera_position_error_mm"] = float(1000 * np.linalg.norm(
                    np.asarray(record["camera_position_m"]) - np.asarray(best["camera_position_m"])))
                final["final_image"] = str(out / "final.png")
                print(f"  final image {out / 'final.png'} (camera {final['camera_position_error_mm']:.1f} mm "
                      "from where the best image was taken)")
        robot.publish_markers(ctx.scene, ctx.marker_views, target=ctx.marker_target)

    termination = stopped or st.termination
    summary = save_contact_sheet(out, target_path, st.captures, best) if st.captures else None
    write_json(out / "result.json", {
        "status": "complete" if not stopped and final["status"] == "reached" else "incomplete",
        "method": "Adaptive Spherical + Radial (DINOv3+SIFT), real Steve",
        "termination_reason": termination,
        "radial_termination_reason": st.radial["termination_reason"] if st.radial else None,
        "best_observation": slim(best),
        "return_to_best": final,
        "levels": [{**level, "best_record": slim(level["best_record"])} for level in st.levels],
        "radial": ({**st.radial, "best_sample": slim(st.radial["best_sample"])} if st.radial else None),
        "captures": [slim(c) for c in st.captures],
        "capture_count": len(st.captures),
        "elapsed_s": time.perf_counter() - started,
        "target_image": str(target_path),
        "target": target_record,
        "settings": {k: v for k, v in vars(args).items() if k != "func"},
        "scene": ctx.scene,
    })

    stv.banner("ACTIVE PERCEPTION SUMMARY")
    for level in st.levels:
        print(f"  level {level['level']}: {level['candidate_count']}/{level['triangle_count']} captured, "
              f"best {path_label(level['best_triangle_path'])} = {level['best_score']:.3f}"
              f"{'  (new global best)' if level['accepted_as_global_best'] else ''}")
    if st.radial:
        n = sum(1 for s in st.radial["samples"] if "score" in s)
        print(f"  radial : {n} image(s), {st.radial['termination_reason']}")
    print(f"  stopped because: {termination}")
    if best is not None:
        print(f"  BEST   : {best['label']}  score {best['score']:.3f}  ({len(st.captures)} images in total)")
    if summary:
        print(f"  contact sheet: {summary}")
    print(f"  report: {out / 'result.json'}")
    return 0 if not stopped and final["status"] == "reached" else 1


# =========================================================================== CLI


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    robot = p.add_argument_group("real robot (as in steve_three_view.py)")
    robot.add_argument("--run-dir", required=True, help="outputs go to <run-dir>/active_perception/")
    robot.add_argument("--plan-only", action="store_true", help="plan and show the level-0 moves, move nothing")
    robot.add_argument("--target-mask", help="default <target-image stem>_mask.png next to the target image")
    robot.add_argument("--scene", default=str(DEFAULT_SCENE), help="scene JSON (carton pose, obstacles)")
    robot.add_argument("--handeye", default=str(DEFAULT_HANDEYE))
    robot.add_argument("--planner-url", default=stv.DEFAULT_PLANNER)
    robot.add_argument("--controller", default="/scaled_joint_trajectory_controller")
    robot.add_argument("--center", type=float, nargs=3, metavar=("X", "Y", "Z"),
                       help="override the carton centre (ur5ebase_link, m)")
    robot.add_argument("--yaw", type=float, help="override the carton yaw (deg)")
    robot.add_argument("--slow-factor", type=float, default=8.0, help="time-stretch every plan (default 8, >= 1)")

    fv = p.add_argument_group("FindVP search (same meaning and defaults as active_perception_triangles.py)")
    fv.add_argument("--views", type=int, choices=(3, 4, 5), default=3)
    fv.add_argument("--name", default="object", help="output filename prefix")
    fv.add_argument("--target-image", help="default <run-dir>/target.png")
    fv.add_argument("--radius", type=float, help="camera radius in m (default scene.json, 0.40)")
    fv.add_argument("--azimuth-offset", type=float, help="rotate the level-0 triangles (default scene.json, 0)")
    fv.add_argument("--similarity-threshold", type=float, default=0.90)
    fv.add_argument("--max-refinement-depth", type=int, default=3)
    fv.add_argument("--refinement-patience", type=int, default=2, help="0 disables")
    fv.add_argument("--min-score-improvement", type=float, default=0.002)
    fv.add_argument("--angular-lookahead-margin", type=float, default=0.03)
    fv.add_argument("--angular-lookahead-levels", type=int, default=1)
    fv.add_argument("--until-convergence", action="store_true",
                    help="3 views, stop when the best score plateaus; ignores threshold/depth; no radial step")
    fv.add_argument("--convergence-min-improvement", type=float, default=0.002)
    fv.add_argument("--convergence-patience", type=int, default=1)
    fv.add_argument("--skip-radial-refinement", action="store_true")
    fv.add_argument("--radial-step", type=float, default=0.02)
    fv.add_argument("--radial-probe-steps", type=int, default=2)
    fv.add_argument("--radial-max-steps", type=int, default=10)
    fv.add_argument("--radial-patience", type=int, default=2)
    fv.add_argument("--radial-min-improvement", type=float, default=0.002)
    fv.add_argument("--radial-min-radius", type=float, default=0.20)
    fv.add_argument("--radial-max-radius", type=float, default=0.60)
    fv.add_argument("--ambiguity-margin", type=float, default=0.03)
    fv.add_argument("--dino-weight", type=float, default=0.75)
    fv.add_argument("--sift-weight", type=float, default=0.25)
    fv.add_argument("--feature-max-image-size", type=int, default=640,
                    help="DINOv3 max image side (default 640 as in steve_three_view.py; the simulator uses 448)")
    fv.add_argument("--debug", action="store_true", help="also save FindVP's clustering/match visualizations")
    return p


def validate(args):
    def unit(name):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be in [0, 1]")

    if args.slow_factor < 1.0:
        raise SystemExit("--slow-factor must be >= 1 (it may only slow the motion down)")
    if args.until_convergence:
        if not 0.0 < args.convergence_min_improvement <= 1.0 or args.convergence_patience < 1:
            raise SystemExit("--convergence-min-improvement must be in (0, 1] and --convergence-patience >= 1")
        args.views = 3
        args.min_score_improvement = args.convergence_min_improvement
        args.skip_radial_refinement = True
    for name in ("similarity_threshold", "min_score_improvement", "angular_lookahead_margin",
                 "radial_min_improvement", "ambiguity_margin"):
        unit(name)
    if args.max_refinement_depth < 0 or args.refinement_patience < 0 or args.angular_lookahead_levels < 0:
        raise SystemExit("depth, patience and look-ahead levels must be non-negative")
    if args.radial_step <= 0 or args.radial_probe_steps <= 0 or args.radial_patience <= 0:
        raise SystemExit("--radial-step, --radial-probe-steps and --radial-patience must be positive")
    if args.radial_max_steps < args.radial_probe_steps:
        raise SystemExit("--radial-max-steps must be at least --radial-probe-steps")
    if not 0.0 < args.radial_min_radius < args.radial_max_radius:
        raise SystemExit("radial radius bounds must be positive and ordered")
    if args.dino_weight < 0 or args.sift_weight < 0 or args.dino_weight + args.sift_weight <= 0:
        raise SystemExit("similarity weights must be non-negative and not both zero")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    validate(args)
    ctx = stv.Context(args)
    ctx.marker_views, ctx.marker_target = [], None
    try:
        return cmd_search(ctx)
    except RuntimeError as exc:
        print(f"\n! {exc}")
        return 2
    finally:
        ctx.close()


if __name__ == "__main__":
    sys.exit(main())
