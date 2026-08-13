"""
Is the scene cache complete, well-formed, and good enough to ground on? No model needed.

    RUNS ON: CPU. Under a minute for all 141 val scenes. No GPU, no checkpoint -- it only
    reads .p files and does box arithmetic.

    python experiments/diagnostics/audit_scene_cache.py
    python experiments/diagnostics/audit_scene_cache.py --split train --limit 50

``validate_scene_cache.py`` is the strict test but needs a GPU, since ``lib/loss_helper.py``
calls ``.cuda()`` in 27 places. This answers a weaker question with no model at all:

1. **Integrity.** Every scene the split needs is present; every file carries the keys
   ``meta.json`` promises, with the recorded shapes and dtypes, and no NaN or Inf.

2. **Recall ceiling.** The fusion network can only return one of the 256 cached proposals,
   so ``max_i IoU(proposal_i, gt)`` bounds grounding accuracy for that annotation. Over the
   split that gives the best score the cache could support -- a ceiling below the reported
   accuracy means the cache is wrong.

The cache uses a deterministic per-scene point subsample while the original end-to-end
evaluation subsampled at random, so the two see slightly different point clouds and an
annotation's achieved IoU is not required to sit below its cached ceiling. How often it
does not comes out as a consistency statistic rather than pass/fail: a few percent is
expected from resampling, a large fraction means different weights or a different detector.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.analysis.common import (
    THRESHOLDS,
    ensure_dir,
    join,
    load_predictions,
    load_scanrefer,
    md_table,
    save_json,
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def aabb_iou(a, b):
    """3-D IoU of two axis-aligned boxes given as (8, 3) corner arrays.

    ScanRefer's boxes are axis aligned, so the extent alone defines them and the IoU is
    exact -- no convex-hull intersection needed.
    """
    a_low, a_high = a.min(axis=0), a.max(axis=0)
    b_low, b_high = b.min(axis=0), b.max(axis=0)
    overlap = np.minimum(a_high, b_high) - np.maximum(a_low, b_low)
    if np.any(overlap <= 0):
        return 0.0
    intersection = float(np.prod(overlap))
    volume_a = float(np.prod(a_high - a_low))
    volume_b = float(np.prod(b_high - b_low))
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0


def batch_max_iou(proposals, gt):
    """Best IoU between one gt box and all (N, 8, 3) proposals, vectorised."""
    p_low, p_high = proposals.min(axis=1), proposals.max(axis=1)      # (N, 3)
    g_low, g_high = gt.min(axis=0), gt.max(axis=0)                    # (3,)

    overlap = np.minimum(p_high, g_high) - np.maximum(p_low, g_low)
    overlap = np.clip(overlap, 0.0, None)
    intersection = overlap.prod(axis=1)

    volume_p = np.clip(p_high - p_low, 0.0, None).prod(axis=1)
    volume_g = float(np.clip(g_high - g_low, 0.0, None).prod())
    union = volume_p + volume_g - intersection

    ious = np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)
    return ious


def load_cached_scene(root, split, scene_id):
    import torch
    path = os.path.join(root, split, f"{scene_id}.p")
    if not os.path.isfile(path):
        return None, path
    return torch.load(path, map_location="cpu", weights_only=False), path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cached-scenes-root", dest="root", default="cached_scenes")
    parser.add_argument("--split", default="val")
    parser.add_argument("--predictions",
                        default="outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p",
                        help="used for the gt boxes and for the achieved-IoU comparison")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--limit", type=int, default=0, help="0 = every scene")
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/diagnostics")
    args = parser.parse_args()

    root = os.path.join(REPO, args.root)
    print("=" * 84)
    print(f"Scene-cache audit -- split '{args.split}', root '{args.root}'")
    print("=" * 84)

    # ---- meta ------------------------------------------------------------------------
    meta_path = os.path.join(root, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"\nno cache found: {os.path.relpath(meta_path, REPO)}")
        print("Build it first with experiments/ablation/scenes_cache.py (GPU).")
        return 1
    with open(meta_path) as handle:
        meta = json.load(handle)

    split_meta = (meta.get("splits") or {}).get(args.split, {})
    expected_keys = list(meta.get("cached_keys") or [])
    print(f"\n[meta] complete={meta.get('complete')}  "
          f"detector={meta.get('detector')}  num_points={meta.get('num_points')}  "
          f"num_proposals={meta.get('num_proposals')}  salt={meta.get('subsample_salt')!r}")
    print(f"[meta] checkpoint = {meta.get('checkpoint')}")
    print(f"[meta] {args.split}: {split_meta.get('num_scenes')} scenes, "
          f"complete={split_meta.get('complete')}")
    print(f"[meta] {len(expected_keys)} cached keys")

    # ---- integrity -------------------------------------------------------------------
    records = load_scanrefer(args.split, args.data_root)
    scene_ids = sorted({r["scene_id"] for r in records})
    if args.limit:
        scene_ids = scene_ids[:args.limit]

    print(f"\n[1] Integrity over {len(scene_ids)} scene(s)\n" + "-" * 84)
    missing_files, missing_keys, bad_values, shapes = [], [], [], {}
    cache = {}
    for scene_id in scene_ids:
        payload, path = load_cached_scene(root, args.split, scene_id)
        if payload is None:
            missing_files.append(scene_id)
            continue
        for key in expected_keys:
            if key not in payload:
                missing_keys.append((scene_id, key))
                continue
            value = np.asarray(payload[key])
            shapes.setdefault(key, (value.shape, str(value.dtype)))
            if value.dtype.kind == "f":
                as_float = value.astype(np.float64, copy=False)
                if not np.isfinite(as_float).all():
                    bad_values.append((scene_id, key))
        cache[scene_id] = np.asarray(payload["pred_bboxes"], dtype=np.float64)

    print(f"  scenes present            : {len(cache)}/{len(scene_ids)}")
    print(f"  files missing             : {len(missing_files)}")
    print(f"  key/scene pairs missing   : {len(missing_keys)}")
    print(f"  tensors with NaN or Inf   : {len(bad_values)}")
    for key in sorted(shapes):
        shape, dtype = shapes[key]
        print(f"    {key:32s} {str(shape):18s} {dtype}")
    for scene_id in missing_files[:5]:
        print(f"    MISSING FILE {scene_id}")
    for scene_id, key in missing_keys[:5]:
        print(f"    MISSING KEY  {scene_id}/{key}")
    for scene_id, key in bad_values[:5]:
        print(f"    NON-FINITE   {scene_id}/{key}")

    integrity_ok = not (missing_files or missing_keys or bad_values)

    # ---- recall ceiling ---------------------------------------------------------------
    predictions_path = os.path.join(REPO, args.predictions)
    if not os.path.isfile(predictions_path):
        print(f"\n[2] SKIPPED -- {args.predictions} not found, so there are no gt boxes "
              f"to measure against.")
        return 0 if integrity_ok else 1

    predictions = load_predictions(predictions_path)
    print(f"\n[2] Recall ceiling from the cached proposals\n" + "-" * 84)

    # load_predictions returns a flat dict keyed by (scene_id, object_id, ann_id); join()
    # is the one place that key convention is written down, so reuse it.
    joined, no_prediction = join(predictions, records)
    if no_prediction:
        print(f"  {no_prediction} annotation(s) have no prediction and are excluded")

    ceilings, achieved, rows = [], [], 0
    for row in joined:
        scene_id = row["scene_id"]
        if scene_id not in cache:
            continue
        gt = np.asarray(row["_prediction"]["gt_bbox"], dtype=np.float64)
        ceilings.append(float(batch_max_iou(cache[scene_id], gt).max()))
        achieved.append(float(row["iou"]))
        rows += 1

    if not rows:
        print("  no annotation could be matched to both a cached scene and a prediction")
        return 1

    ceilings = np.asarray(ceilings)
    achieved = np.asarray(achieved)

    table_rows = []
    for threshold in THRESHOLDS:
        table_rows.append([
            f"@{threshold}",
            f"{100 * (ceilings >= threshold).mean():.2f}",
            f"{100 * (achieved >= threshold).mean():.2f}",
            f"{100 * ((ceilings >= threshold).mean() - (achieved >= threshold).mean()):+.2f}",
        ])
    table = md_table(["IoU", "cache ceiling %", "achieved %", "headroom pp"], table_rows)
    print("\n" + table)
    print(f"\n  annotations compared     : {rows}")
    print(f"  mean best-proposal IoU   : {ceilings.mean():.4f}")
    print(f"  median best-proposal IoU : {np.median(ceilings):.4f}")

    violations = int((achieved > ceilings + 1e-6).sum())
    print(f"  achieved > ceiling       : {violations} ({100 * violations / rows:.2f}%)"
          f"   <- resampling noise; large values mean a different checkpoint")

    # ---- verdict ----------------------------------------------------------------------
    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    problems = []
    if not integrity_ok:
        problems.append("the cache is incomplete or malformed -- see section 1")
    for threshold in THRESHOLDS:
        ceiling = (ceilings >= threshold).mean()
        got = (achieved >= threshold).mean()
        if ceiling < got:
            problems.append(
                f"at IoU {threshold} the cached proposals top out at "
                f"{100 * ceiling:.2f}% but the run reports {100 * got:.2f}%. The fusion "
                f"network can only return one of these proposals, so the cache cannot "
                f"be the source of that number.")
    if violations / rows > 0.25:
        problems.append(
            f"{100 * violations / rows:.1f}% of annotations beat their own cached "
            f"ceiling, which is far more than resampling explains. Check that "
            f"meta.json's checkpoint matches the run that produced these predictions.")

    if problems:
        for index, item in enumerate(problems, 1):
            print(f"  {index}. {item}")
        print("\n  Do not run ablations on this cache.")
    else:
        print("  The cache is complete, finite, correctly shaped, and leaves "
              f"{100 * ((ceilings >= 0.25).mean() - (achieved >= 0.25).mean()):.1f} pp of "
              f"headroom above the reported Acc@0.25.")
        print("  Nothing here proves numerical equivalence -- that is "
              "validate_scene_cache.py's\n  job and it needs a GPU -- but every failure "
              "mode this can see is absent.")

    report = {
        "split": args.split, "root": args.root, "scenes": len(cache),
        "annotations": rows, "integrity_ok": integrity_ok,
        "missing_files": missing_files, "missing_keys": missing_keys[:50],
        "non_finite": bad_values[:50],
        "ceiling": {str(t): float((ceilings >= t).mean()) for t in THRESHOLDS},
        "achieved": {str(t): float((achieved >= t).mean()) for t in THRESHOLDS},
        "mean_best_iou": float(ceilings.mean()),
        "achieved_above_ceiling": violations,
        "problems": problems,
    }
    out_dir = os.path.join(REPO, args.out_dir)
    ensure_dir(out_dir)
    path = save_json(report, os.path.join(out_dir, f"scene_cache_audit_{args.split}.json"))
    print(f"\nwrote {os.path.relpath(path, REPO)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
