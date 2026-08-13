
import argparse
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.ablation.parsers.eval_parser_target_accuracy import match_kinds
from experiments.analysis.common import (
    THRESHOLDS,
    accuracy,
    ensure_dir,
    join,
    load_parse_cache,
    load_predictions,
    load_scanrefer,
    md_table,
    parse_for,
    parse_predictions_arg,
    pct,
    save_json,
    two_proportion_z,
    wilson_ci,
)

CRITERIA = ("exact", "substring", "fuzzy")


def label_rows(rows, cache, criterion):
    correct, wrong, uncovered = [], [], 0
    for row in rows:
        parsed = parse_for(cache, row["scene_id"], row["object_id"], row["ann_id"])
        if not parsed:
            uncovered += 1
            continue
        predicted = " ".join(parsed.get("target") or [])
        kinds = match_kinds(predicted, row["object_name"])
        row = dict(row, _predicted_target=predicted)
        (correct if kinds[criterion] else wrong).append(row)
    return correct, wrong, uncovered


def subset_stats(rows, threshold):
    ious = [row["iou"] for row in rows]
    hits = int(sum(1 for i in ious if i >= threshold))
    low, high = wilson_ci(hits, len(ious))
    return {"n": len(ious), "hits": hits, "acc": accuracy(ious, threshold),
            "ci": [low, high]}


def controlled_difference(correct, wrong, threshold):
    by_class = defaultdict(lambda: {"correct": [], "wrong": []})
    for row in correct:
        by_class[row["object_name"]]["correct"].append(row["iou"])
    for row in wrong:
        by_class[row["object_name"]]["wrong"].append(row["iou"])

    total_weight, weighted_sum, used = 0.0, 0.0, 0
    observed, expected, variance = 0.0, 0.0, 0.0

    for _class_name, group in by_class.items():
        n1, n2 = len(group["correct"]), len(group["wrong"])
        if n1 == 0 or n2 == 0:
            continue
        weight = 2.0 * n1 * n2 / (n1 + n2)
        difference = (accuracy(group["correct"], threshold)
                      - accuracy(group["wrong"], threshold))
        weighted_sum += weight * difference
        total_weight += weight
        used += 1

        a = sum(1 for i in group["correct"] if i >= threshold)
        c = sum(1 for i in group["wrong"] if i >= threshold)
        total = n1 + n2
        hit_total = a + c
        miss_total = total - hit_total
        observed += a
        expected += n1 * hit_total / total
        if total > 1:
            variance += (n1 * n2 * hit_total * miss_total) / (total * total * (total - 1))

    if total_weight == 0:
        return {"difference": float("nan"), "classes_used": 0, "coverage": 0.0,
                "cmh_chi2": float("nan"), "cmh_p": float("nan")}

    if variance > 0:
        chi2 = (abs(observed - expected) - 0.5) ** 2 / variance
        cmh_p = math.erfc(math.sqrt(chi2 / 2.0))
    else:
        chi2, cmh_p = float("nan"), float("nan")

    covered = sum(len(g["correct"]) + len(g["wrong"]) for g in by_class.values()
                  if g["correct"] and g["wrong"])
    return {
        "difference": weighted_sum / total_weight,
        "classes_used": used,
        "classes_total": len(by_class),
        "coverage": covered / max(len(correct) + len(wrong), 1),
        "cmh_chi2": chi2,
        "cmh_p": cmh_p,
    }


def composition(rows, top_k=8):
    counts = Counter(row["object_name"] for row in rows)
    total = max(sum(counts.values()), 1)
    return [(name, count, 100.0 * count / total) for name, count in counts.most_common(top_k)]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True,
                        help="outputs/<run>/predictions.p, optionally NAME=PATH.")
    parser.add_argument("--parse", action="append", required=True, metavar="NAME=FOLDER",
                        help="Repeatable. Parse cache under data_parsing/ to score.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--criterion", default="fuzzy", choices=CRITERIA,
                        help="Which target-match criterion counts as a correct parse.")
    parser.add_argument("--threshold", type=float, default=0.25, choices=list(THRESHOLDS))
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/parse_quality_split")
    args = parser.parse_args()

    model_name, predictions_path = parse_predictions_arg(args.predictions)
    records = load_scanrefer(args.split, args.data_root)
    rows, missing = join(load_predictions(predictions_path), records)
    print(f"[{model_name}] {len(rows)} annotations joined from {predictions_path}" +
          (f"  ({missing} without a prediction)" if missing else ""))
    print(f"[criterion] a parse counts as correct when its target is a "
          f"{args.criterion!r} match to object_name\n")

    report, lines = {}, ["# Grounding accuracy vs parse correctness\n",
                         f"model: `{model_name}` | split: `{args.split}` | "
                         f"criterion: `{args.criterion}` | "
                         f"IoU threshold: {args.threshold}\n"]

    summary_rows = []
    for spec in args.parse:
        name, folder = (spec.split("=", 1) if "=" in spec else (spec, spec))
        name, folder = name.strip(), folder.strip()
        try:
            cache = load_parse_cache(folder, args.split, args.parsing_root)
        except FileNotFoundError as error:
            print(f"[{name}] SKIPPED: {error}")
            continue

        correct, wrong, uncovered = label_rows(rows, cache, args.criterion)
        good = subset_stats(correct, args.threshold)
        bad = subset_stats(wrong, args.threshold)
        difference, z, p = two_proportion_z(good["hits"], good["n"], bad["hits"], bad["n"])
        controlled = controlled_difference(correct, wrong, args.threshold)

        print(f"=== {name}  ({folder}) ===")
        if uncovered:
            print(f"  {uncovered} annotations absent from this parse cache; excluded")
        print(f"  parse correct : n={good['n']:>5}  Acc@{args.threshold}="
              f"{pct(good['acc'])}%  [{pct(good['ci'][0], 1)}-{pct(good['ci'][1], 1)}]")
        print(f"  parse wrong   : n={bad['n']:>5}  Acc@{args.threshold}="
              f"{pct(bad['acc'])}%  [{pct(bad['ci'][0], 1)}-{pct(bad['ci'][1], 1)}]")
        print(f"  raw difference          : {100 * difference:+.2f} pp "
              f"(z={z:.2f}, p={p:.3e})")
        print(f"  category-controlled     : {100 * controlled['difference']:+.2f} pp "
              f"(CMH chi2={controlled['cmh_chi2']:.2f}, p={controlled['cmh_p']:.3e}) "
              f"over {controlled['classes_used']}/{controlled.get('classes_total', 0)} "
              f"classes covering {100 * controlled['coverage']:.0f}% of annotations")

        if difference and controlled["difference"] == controlled["difference"]:
            ratio = controlled["difference"] / difference
            if ratio > 1.15:
                print(f"  -> controlling AMPLIFIES the gap ({ratio:.1f}x). Class "
                      f"composition was masking it: the misparsed subset happens to "
                      f"contain easier classes, so the raw number understates the "
                      f"association. Quote the controlled number.")
            elif ratio < 0.85:
                print(f"  -> controlling SHRINKS the gap to {100 * ratio:.0f}% of raw. "
                      f"Most of the raw difference was class composition, not the "
                      f"parse. Quote the controlled number and say so.")
            else:
                print("  -> controlling barely moves the gap; class composition is not "
                      "driving it.")

        print("  composition of each subset (top classes):")
        for label, subset in (("correct", correct), ("wrong", wrong)):
            top = ", ".join(f"{n} {c:.0f}%" for n, _k, c in composition(subset, 5))
            print(f"    {label:<8}: {top}")
        print()

        report[name] = {
            "folder": folder, "uncovered": uncovered,
            "correct": good, "wrong": bad,
            "raw_difference": difference, "z": z, "p_value": p,
            "category_controlled": controlled,
            "composition": {
                "correct": composition(correct), "wrong": composition(wrong)},
        }
        summary_rows.append([
            name, good["n"], f"{pct(good['acc'])}", bad["n"], f"{pct(bad['acc'])}",
            f"{100 * difference:+.2f}", f"{p:.2e}",
            f"{100 * controlled['difference']:+.2f}", f"{controlled['cmh_p']:.2e}",
        ])

    if not summary_rows:
        print("No parse cache could be loaded; nothing to report.")
        return 1

    table = md_table(
        ["parser", "n correct", f"Acc@{args.threshold} correct", "n wrong",
         f"Acc@{args.threshold} wrong", "raw diff (pp)", "raw p",
         "controlled diff (pp)", "CMH p"],
        summary_rows)
    print(table)

    lines.append(table)
    lines.append("\n**raw diff** is confounded: wrongly parsed descriptions skew towards "
                 "rare classes and unusual phrasing, which are harder to ground "
                 "regardless of the parse.\n")
    lines.append("**controlled diff** recomputes the same difference within each "
                 "`object_name` class and averages it with Cochran-Mantel-Haenszel "
                 "weights, so class composition cannot drive it.\n")
    lines.append("Neither is an intervention. Pair this with the corruption test "
                 "(`experiments/analysis/parse_error_propagation.py`), which holds the sample fixed "
                 "and changes only the parse.\n")

    ensure_dir(args.out_dir)
    json_path = save_json({
        "model": model_name, "predictions": predictions_path,
        "split": args.split, "criterion": args.criterion,
        "threshold": args.threshold, "parsers": report,
    }, os.path.join(args.out_dir, "parse_quality_split.json"))
    report_path = os.path.join(args.out_dir, "parse_quality_split.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nwrote {json_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
