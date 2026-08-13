"""
Main results table: Overall / Unique / Multiple, with a paired baseline test.

    RUNS ON: CPU. Seconds. No GPU, no model -- reads predictions.p.

    python experiments/analysis/results_table.py \
        --predictions ours=outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p \
        --predictions 3DVG-Trans=outputs/3DVG-TRANS-outputs/predictions.p

Every number is recomputed from a prediction file rather than copied from the published
tables.

The Unique/Multiple split follows ScanRefer: an annotation is *unique* when its scene
holds exactly one object of the referred object's NYU40 class. The class is the mapped
label, not the raw ``object_name`` -- mapping through ``scannetv2-labels.combined.tsv``
turns 3759/5749 into 1845/7663. ``common.unique_multiple_lookup`` ports
``lib/dataset.py:494-566``; its docstring covers why ``scores.p["masks"]`` is unusable
here.

Models share the same 9,508 annotations, so significance is tested with McNemar, which
conditions on the annotations where two models disagree. The bootstrap CI resamples
annotations and carries the same pairing.
"""

import argparse
import os
import sys

import numpy as np

# Resolve the repo root from this file, not the cwd, so the script works when
# invoked as `python experiments/analysis/<name>.py` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.analysis.common import (
    THRESHOLDS,
    bootstrap_ci,
    ensure_dir,
    join,
    load_predictions,
    load_scanrefer,
    mcnemar,
    md_table,
    parse_predictions_arg,
    pct,
    save_json,
    unique_multiple_lookup,
    wilson_ci,
)

SUBSETS = ("overall", "unique", "multiple")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", action="append", required=True,
                        metavar="NAME=PATH",
                        help="Repeatable. The first is the reference every other model is "
                             "compared against.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for the difference CIs; 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/analysis/results_table")
    parser.add_argument("--latex", action="store_true")
    args = parser.parse_args()

    records = load_scanrefer(args.split, args.data_root)
    lookup = unique_multiple_lookup(records)

    models, hits, ious = [], {}, {}
    common_keys = None
    for spec in args.predictions:
        name, path = parse_predictions_arg(spec)
        rows, missing = join(load_predictions(path), records)
        print(f"[{name}] {len(rows)} annotations from {path}" +
              (f"  ({missing} without a prediction)" if missing else ""))
        models.append(name)
        keys = {(r["scene_id"], r["object_id"], r["ann_id"]): r["iou"] for r in rows}
        ious[name] = keys
        common_keys = set(keys) if common_keys is None else (common_keys & set(keys))

    order = [k for k in ((r["scene_id"], r["object_id"], r["ann_id"]) for r in records)
             if k in common_keys]
    if len(order) != len(records):
        print(f"[join] {len(order)}/{len(records)} annotations are predicted by all "
              f"{len(models)} model(s); the rest are excluded so every column has the "
              f"same denominator")

    subset_mask = {
        "overall": np.ones(len(order), dtype=bool),
        "unique": np.array([lookup[k] == 0 for k in order]),
        "multiple": np.array([lookup[k] == 1 for k in order]),
    }
    print(f"[split] unique={subset_mask['unique'].sum()}  "
          f"multiple={subset_mask['multiple'].sum()}  total={len(order)}")

    for name in models:
        series = np.array([ious[name][k] for k in order], dtype=float)
        hits[name] = {t: (series >= t) for t in THRESHOLDS}

    reference = models[0]
    report = {"split": args.split, "models": models, "n": len(order),
              "subset_sizes": {s: int(subset_mask[s].sum()) for s in SUBSETS},
              "results": {}}

    rows = []
    for name in models:
        row = [name]
        report["results"][name] = {}
        for subset in SUBSETS:
            mask = subset_mask[subset]
            entry = {}
            for threshold in THRESHOLDS:
                selected = hits[name][threshold][mask]
                k, n = int(selected.sum()), int(mask.sum())
                low, high = wilson_ci(k, n)
                entry[f"acc_{threshold}"] = k / n if n else 0.0
                entry[f"acc_{threshold}_ci"] = [low, high]
                row.append(pct(k / n if n else 0.0))
            report["results"][name][subset] = entry
        rows.append(row)

    headers = ["model"]
    for subset in SUBSETS:
        for threshold in THRESHOLDS:
            headers.append(f"{subset} @{threshold}")
    table = md_table(headers, rows)
    print("\n" + table)

    # ---- paired comparison against the reference -------------------------------------
    comparison_rows = []
    for name in models[1:]:
        for subset in SUBSETS:
            mask = subset_mask[subset]
            for threshold in THRESHOLDS:
                a = hits[reference][threshold][mask]
                b = hits[name][threshold][mask]
                a_only, b_only, chi2, p = mcnemar(a, b)
                difference = float(a.mean() - b.mean())

                ci_text = "-"
                if args.bootstrap:
                    def statistic(idx, a=a, b=b):
                        return float(a[idx].mean() - b[idx].mean())

                    _point, low, high = bootstrap_ci(
                        statistic, int(mask.sum()), args.bootstrap, args.seed)
                    ci_text = f"[{100 * low:+.2f}, {100 * high:+.2f}]"

                comparison_rows.append([
                    f"{subset} @{threshold}", f"{100 * difference:+.2f}", ci_text,
                    f"{a_only}/{b_only}", f"{p:.2e}",
                    "yes" if (p == p and p < 0.05) else "no",
                ])
                report["results"][name].setdefault("vs_" + reference, {})[
                    f"{subset}_{threshold}"] = {
                    "diff_pp": 100 * difference, "mcnemar_chi2": chi2, "p_value": p,
                    "reference_only": a_only, "other_only": b_only,
                }

        comparison = md_table(
            ["subset", f"{reference} - {name} (pp)", "95% CI (pp)",
             f"{reference}-only/{name}-only", "McNemar p", "significant"],
            comparison_rows)
        print(f"\nPaired comparison: {reference} vs {name}\n")
        print(comparison)
        comparison_rows = []

    # ---- narrative ------------------------------------------------------------------
    lines = ["# Main results table\n",
             f"split: `{args.split}` | annotations: {len(order)} "
             f"(unique {int(subset_mask['unique'].sum())}, "
             f"multiple {int(subset_mask['multiple'].sum())})\n",
             "**All numbers below were computed on this machine from `predictions.p`, "
             "not quoted from the original papers.**\n",
             table]

    verdicts = []
    if len(models) > 1:
        for name in models[1:]:
            deltas = {}
            for subset in SUBSETS:
                mask = subset_mask[subset]
                deltas[subset] = float(hits[reference][0.25][mask].mean()
                                       - hits[name][0.25][mask].mean())
            if deltas["multiple"] > deltas["unique"] + 0.01:
                verdicts.append(
                    f"Against {name}, the advantage is concentrated in **multiple** "
                    f"({100 * deltas['multiple']:+.2f} pp) rather than unique "
                    f"({100 * deltas['unique']:+.2f} pp) at IoU 0.25. Ambiguous scenes are "
                    f"where adjacency reasoning should help, so this is the pattern the "
                    f"method predicts and it is worth stating explicitly.")
            elif deltas["unique"] > deltas["multiple"] + 0.01:
                verdicts.append(
                    f"Against {name}, the advantage is larger on **unique** "
                    f"({100 * deltas['unique']:+.2f} pp) than multiple "
                    f"({100 * deltas['multiple']:+.2f} pp). That is the opposite of what "
                    f"adjacency reasoning predicts -- report it and do not attribute the "
                    f"gain to disambiguation.")
            else:
                verdicts.append(
                    f"Against {name}, the advantage is roughly uniform across unique "
                    f"({100 * deltas['unique']:+.2f} pp) and multiple "
                    f"({100 * deltas['multiple']:+.2f} pp).")

    if verdicts:
        print("\nReading:")
        for verdict in verdicts:
            print(f"  {verdict}")
        lines.append("\n## Reading\n")
        lines += [f"- {v}" for v in verdicts]

    if args.latex:
        print("\nLaTeX rows:")
        latex = []
        for row in rows:
            latex.append("  " + row[0] + " & " + " & ".join(row[1:]) + r" \\")
        for line in latex:
            print(line)
        lines.append("\n## LaTeX\n\n```\n" + "\n".join(latex) + "\n```")

    ensure_dir(args.out_dir)
    report_path = os.path.join(args.out_dir, "results_table.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json(report, os.path.join(args.out_dir, "results_table.json"))
    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
