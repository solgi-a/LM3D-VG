
import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.ablation.parsers.eval_parser_target_accuracy import match_kinds
from experiments.analysis.common import (
    box_volume,
    ensure_dir,
    join,
    load_parse_cache,
    load_predictions,
    load_scanrefer,
    md_table,
    parse_for,
    parse_predictions_arg,
    save_json,
    write_bbox_ply,
)
from experiments.analysis.linguistic_complexity import spatial_hits

CAUSES = ("parse_target_wrong", "distractor_confusion", "small_object",
          "localization_drift", "complex_language", "unattributed")

CAUSE_TEXT = {
    "parse_target_wrong":
        "the parser extracted the wrong target class, so the grounding module was "
        "conditioned on the wrong object category",
    "distractor_confusion":
        "the correct category was identified but the wrong instance was selected; the "
        "scene contains several objects of that class",
    "small_object":
        "the referred object is among the smallest in the split, so the detector's "
        "proposals resolve it poorly",
    "localization_drift":
        "the correct region was found but the predicted box is too imprecise to pass "
        "the IoU threshold",
    "complex_language":
        "a long description with many spatial relations and no simpler failure cause",
    "unattributed":
        "none of the diagnosed causes applies",
}


def scene_class_counts(records):
    instances = defaultdict(lambda: defaultdict(set))
    for record in records:
        instances[record["scene_id"]][record["object_name"]].add(record["object_id"])
    return {scene: {name: len(ids) for name, ids in classes.items()}
            for scene, classes in instances.items()}


def attribute(row, signals, args):
    if not signals["parse_target_ok"]:
        return "parse_target_wrong"
    if row["iou"] < args.distractor_iou and signals["same_class_instances"] >= 2:
        return "distractor_confusion"
    if signals["gt_volume"] <= signals["small_threshold"]:
        return "small_object"
    if args.drift_low <= row["iou"] < 0.25:
        return "localization_drift"
    if (signals["num_tokens"] >= signals["long_threshold"]
            or signals["num_spatial"] >= args.many_spatial):
        return "complex_language"
    return "unattributed"


def select(cases, top, strategy):
    if strategy == "worst":
        return sorted(cases, key=lambda c: c["iou"])[:top]

    by_cause = defaultdict(list)
    for case in cases:
        by_cause[case["cause"]].append(case)
    for group in by_cause.values():
        group.sort(key=lambda c: c["iou"])

    order = [cause for cause, _ in Counter(c["cause"] for c in cases).most_common()]
    chosen, index = [], 0
    while len(chosen) < top and any(by_cause[c] for c in order):
        cause = order[index % len(order)]
        if by_cause[cause]:
            chosen.append(by_cause[cause].pop(0))
        index += 1
    return chosen


def dump_case(case, out_dir, args):
    directory = ensure_dir(os.path.join(
        out_dir, f"{case['rank']:02d}_{case['cause']}_"
                 f"{case['scene_id']}_obj{case['object_id']}_ann{case['ann_id']}"))

    write_bbox_ply(case["_pred_bbox"], os.path.join(directory, "pred.ply"),
                   color=(220, 30, 30), radius=args.box_radius)
    write_bbox_ply(case["_gt_bbox"], os.path.join(directory, "gt.ply"),
                   color=(30, 190, 60), radius=args.box_radius)

    with open(os.path.join(directory, "case.txt"), "w") as f:
        f.write(f"scene       : {case['scene_id']}\n")
        f.write(f"object      : {case['object_id']} ({case['object_name']})\n")
        f.write(f"annotation  : {case['ann_id']}\n")
        f.write(f"IoU         : {case['iou']:.4f}\n")
        f.write(f"cause       : {case['cause']}\n")
        f.write(f"              {CAUSE_TEXT[case['cause']]}\n\n")
        f.write(f"description : {case['description']}\n\n")
        f.write(f"parsed target     : {case['parsed_target']!r}"
                f"  (ground truth {case['object_name']!r})\n")
        f.write(f"parsed adjectives : {case['parsed_adjectives']!r}\n")
        f.write(f"parsed neighbors  : {case['parsed_neighbors']!r}\n\n")
        f.write("signals\n")
        for key, value in case["signals"].items():
            f.write(f"  {key:<22}: {value}\n")
        f.write("\nboxes: pred.ply (red) and gt.ply (green), wireframe PLY.\n")
        f.write("Full scene mesh:\n")
        f.write(f"  {case['visualize_command']}\n")
    return directory


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parse-folder", dest="parse_folder",
                        default="final_parsing_tokenized")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--criterion", default="fuzzy",
                        choices=["exact", "substring", "fuzzy"])
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="A failure is IoU below this.")
    parser.add_argument("--top", type=int, default=6,
                        help="How many cases to dump. The roadmap asks for at least 3.")
    parser.add_argument("--strategy", default="diverse", choices=["diverse", "worst"])
    parser.add_argument("--distractor-iou", dest="distractor_iou", type=float, default=0.05)
    parser.add_argument("--drift-low", dest="drift_low", type=float, default=0.10)
    parser.add_argument("--small-percentile", dest="small_percentile", type=float, default=10.0)
    parser.add_argument("--long-percentile", dest="long_percentile", type=float, default=75.0)
    parser.add_argument("--many-spatial", dest="many_spatial", type=int, default=3)
    parser.add_argument("--box-radius", dest="box_radius", type=float, default=0.02)
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/failure_cases")
    parser.add_argument("--no-ply", dest="write_ply", action="store_false")
    args = parser.parse_args()

    model_name, predictions_path = parse_predictions_arg(args.predictions)
    records = load_scanrefer(args.split, args.data_root)
    rows, missing = join(load_predictions(predictions_path), records)
    print(f"[{model_name}] {len(rows)} annotations joined" +
          (f"  ({missing} without a prediction)" if missing else ""))

    try:
        cache = load_parse_cache(args.parse_folder, args.split, args.parsing_root)
    except FileNotFoundError as error:
        print(f"WARNING: {error}\n         parse-based causes will be unavailable")
        cache = {}

    counts = scene_class_counts(records)
    volumes = np.array([box_volume(row["_prediction"]["gt_bbox"]) for row in rows])
    small_threshold = float(np.percentile(volumes, args.small_percentile))
    lengths = np.array([len(row.get("token") or []) for row in rows])
    long_threshold = float(np.percentile(lengths, args.long_percentile))
    print(f"[thresholds] small object <= {small_threshold:.3f} m^3 "
          f"(p{args.small_percentile:g}) | long description >= {long_threshold:.0f} tokens "
          f"(p{args.long_percentile:g})")

    failures = []
    for row in rows:
        if row["iou"] >= args.threshold:
            continue
        parsed = parse_for(cache, row["scene_id"], row["object_id"], row["ann_id"]) or {}
        predicted_target = " ".join(parsed.get("target") or [])
        target_ok = (match_kinds(predicted_target, row["object_name"])[args.criterion]
                     if parsed else True)

        signals = {
            "parse_target_ok": bool(target_ok),
            "same_class_instances": counts[row["scene_id"]].get(row["object_name"], 1),
            "gt_volume": round(box_volume(row["_prediction"]["gt_bbox"]), 4),
            "small_threshold": round(small_threshold, 4),
            "num_tokens": int(len(row.get("token") or [])),
            "long_threshold": int(long_threshold),
            "num_spatial": len(spatial_hits(row.get("token") or [])),
        }
        cause = attribute(row, signals, args)

        failures.append({
            "scene_id": row["scene_id"], "object_id": row["object_id"],
            "ann_id": row["ann_id"], "object_name": row["object_name"],
            "description": row.get("description", ""),
            "iou": row["iou"], "cause": cause, "signals": signals,
            "parsed_target": predicted_target or "(no parse)",
            "parsed_adjectives": " ".join(parsed.get("adjectives") or []) or "(no parse)",
            "parsed_neighbors": " ".join(parsed.get("neighbors") or []) or "(no parse)",
            "_pred_bbox": row["_prediction"]["pred_bbox"],
            "_gt_bbox": row["_prediction"]["gt_bbox"],
        })

    total = len(rows)
    print(f"\n[failures] {len(failures)} of {total} annotations "
          f"({100.0 * len(failures) / max(total, 1):.2f}%) fall below IoU {args.threshold}\n")

    tally = Counter(case["cause"] for case in failures)
    distribution = md_table(
        ["cause", "count", "% of failures", "% of all"],
        [[cause, tally[cause],
          f"{100.0 * tally[cause] / max(len(failures), 1):.1f}",
          f"{100.0 * tally[cause] / max(total, 1):.1f}"]
         for cause in CAUSES if tally[cause]])
    print(distribution)

    chosen = select(failures, args.top, args.strategy)
    ensure_dir(args.out_dir)

    lines = ["# Failure cases\n",
             f"model: `{model_name}` | split: `{args.split}` | "
             f"failure = IoU < {args.threshold} | parse: `{args.parse_folder}`\n",
             f"{len(failures)} of {total} annotations "
             f"({100.0 * len(failures) / max(total, 1):.2f}%) are failures.\n",
             "\n## Distribution of causes\n", distribution,
             "\n## Selected cases\n"]

    print(f"\n[selected] {len(chosen)} case(s), strategy={args.strategy}\n")
    for rank, case in enumerate(chosen, 1):
        case["rank"] = rank
        case["visualize_command"] = (
            f"python scripts/visualize.py --folder {os.path.basename(os.path.dirname(os.path.abspath(predictions_path)))} "
            f"--scene_id {case['scene_id']} --use_color --use_normal")

        print(f"[{rank}] {case['cause']:<22} IoU={case['iou']:.3f}  "
              f"{case['scene_id']} obj={case['object_id']} ({case['object_name']})")
        print(f"     {case['description'][:110]}")
        print(f"     parsed target={case['parsed_target']!r}  "
              f"same-class instances={case['signals']['same_class_instances']}  "
              f"tokens={case['signals']['num_tokens']}  "
              f"spatial={case['signals']['num_spatial']}")

        lines.append(f"\n### {rank}. {case['cause']} — IoU {case['iou']:.3f}\n")
        lines.append(f"- **scene / object**: `{case['scene_id']}` / "
                     f"`{case['object_id']}` ({case['object_name']}), ann `{case['ann_id']}`")
        lines.append(f"- **description**: {case['description']}")
        lines.append(f"- **parsed**: target=`{case['parsed_target']}`, "
                     f"adjectives=`{case['parsed_adjectives']}`, "
                     f"neighbors=`{case['parsed_neighbors']}`")
        lines.append(f"- **why it failed**: {CAUSE_TEXT[case['cause']]}")
        lines.append(f"- **signals**: same-class instances "
                     f"{case['signals']['same_class_instances']}, GT volume "
                     f"{case['signals']['gt_volume']} m³, {case['signals']['num_tokens']} "
                     f"tokens, {case['signals']['num_spatial']} spatial cues")
        lines.append(f"- **render**: `{case['visualize_command']}`")

        if args.write_ply:
            directory = dump_case(case, args.out_dir, args)
            lines.append(f"- **boxes**: `{directory}/pred.ply` (red), "
                         f"`{directory}/gt.ply` (green)")

    report_path = os.path.join(args.out_dir, "failure_cases.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    json_path = save_json({
        "model": model_name, "predictions": predictions_path, "split": args.split,
        "threshold": args.threshold, "criterion": args.criterion,
        "num_annotations": total, "num_failures": len(failures),
        "cause_distribution": dict(tally),
        "cause_explanations": CAUSE_TEXT,
        "selected": [{k: v for k, v in case.items() if not k.startswith("_")}
                     for case in chosen],
    }, os.path.join(args.out_dir, "failure_cases.json"))

    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")
    if args.write_ply:
        print(f"wrote {len(chosen)} case directories under {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
