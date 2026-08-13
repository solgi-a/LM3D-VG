
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.analysis.common import (
    THRESHOLDS,
    accuracy,
    ensure_dir,
    load_predictions,
    md_table,
    pct,
    save_json,
    wilson_ci,
)

_RATE_IN_NAME = re.compile(r"_(\d{1,3})$")


def discover(run_dir):
    if not os.path.isdir(run_dir):
        raise SystemExit(
            f"run directory not found: {run_dir}\n"
            f"Produce it first (GPU):\n"
            f"    python experiments/ablation/runners/run_parse_corruption.py")

    levels = []
    for entry in sorted(os.listdir(run_dir)):
        directory = os.path.join(run_dir, entry)
        predictions = os.path.join(directory, "predictions.p")
        if not os.path.isfile(predictions):
            continue

        manifest_path = os.path.join(directory, "corruption_manifest.json")
        manifest = None
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)

        if manifest and manifest.get("requested_rate") is not None:
            rate = float(manifest["requested_rate"])
        elif entry.lower() in ("baseline", "clean", "rate_0", "00"):
            rate = 0.0
        else:
            match = _RATE_IN_NAME.search(entry)
            rate = int(match.group(1)) / 100.0 if match else float("nan")

        levels.append({"name": entry, "dir": directory, "rate": rate,
                       "predictions": predictions, "manifest": manifest})

    levels.sort(key=lambda level: (math.isnan(level["rate"]), level["rate"]))
    return levels


def corrupted_keys(manifest, split):
    if not manifest:
        return None
    entries = (manifest.get("corrupted") or {}).get(split)
    if entries is None:
        buckets = manifest.get("corrupted") or {}
        entries = next(iter(buckets.values()), []) if buckets else []
    return {(str(e["scene_id"]), str(e["object_id"]), str(e["ann_id"])) for e in entries}


def mcnemar(broke, fixed):
    n = broke + fixed
    if n == 0:
        return float("nan"), float("nan")
    chi2 = (abs(broke - fixed) - 1) ** 2 / n
    return chi2, math.erfc(math.sqrt(chi2 / 2.0))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", dest="run_dir", required=True,
                        help="Directory holding one subdirectory per corruption level.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.25, choices=list(THRESHOLDS))
    parser.add_argument("--baseline", default=None,
                        help="Explicit path to the uncorrupted predictions.p. Defaults to "
                             "the rate-0 level inside --run-dir.")
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/parse_error_propagation")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    levels = discover(args.run_dir)
    if not levels:
        raise SystemExit(f"no predictions.p found under {args.run_dir}")
    print(f"[levels] {len(levels)} found: " +
          ", ".join(f"{l['name']}({100 * l['rate']:.0f}%)" for l in levels))

    baseline_predictions = None
    if args.baseline:
        baseline_predictions = load_predictions(args.baseline)
    else:
        for level in levels:
            if level["rate"] == 0.0:
                baseline_predictions = load_predictions(level["predictions"])
                break
    if baseline_predictions is None:
        print("[baseline] no rate-0 level found; the paired flip analysis is unavailable. "
              "Add a 'baseline/' directory or pass --baseline.")

    results, table_rows = [], []
    for level in levels:
        predictions = load_predictions(level["predictions"])
        ious = [float(record["iou"]) for record in predictions.values()]
        hits = sum(1 for i in ious if i >= args.threshold)
        low, high = wilson_ci(hits, len(ious))

        entry = {
            "name": level["name"], "rate": level["rate"], "n": len(ious),
            "acc": accuracy(ious, args.threshold), "ci": [low, high],
            "acc_other": {str(t): accuracy(ious, t) for t in THRESHOLDS},
        }

        keys = corrupted_keys(level["manifest"], args.split)
        if keys:
            touched = [float(predictions[k]["iou"]) for k in keys if k in predictions]
            untouched = [float(record["iou"]) for key, record in predictions.items()
                         if key not in keys]
            entry["corrupted_subset"] = {
                "n": len(touched), "acc": accuracy(touched, args.threshold)}
            entry["untouched_subset"] = {
                "n": len(untouched), "acc": accuracy(untouched, args.threshold)}

            if baseline_predictions:
                broke = fixed = same = 0
                for key in keys:
                    if key not in predictions or key not in baseline_predictions:
                        continue
                    before = float(baseline_predictions[key]["iou"]) >= args.threshold
                    after = float(predictions[key]["iou"]) >= args.threshold
                    if before and not after:
                        broke += 1
                    elif after and not before:
                        fixed += 1
                    else:
                        same += 1
                chi2, p = mcnemar(broke, fixed)
                paired_n = broke + fixed + same
                entry["paired"] = {
                    "n": paired_n, "broke": broke, "fixed": fixed, "unchanged": same,
                    "net_broken": broke - fixed,
                    "net_rate": (broke - fixed) / paired_n if paired_n else 0.0,
                    "mcnemar_chi2": chi2, "mcnemar_p": p,
                }

        results.append(entry)

        row = [level["name"], f"{100 * level['rate']:.0f}", entry["n"],
               f"{pct(entry['acc'])} [{pct(low, 1)}-{pct(high, 1)}]"]
        if "corrupted_subset" in entry:
            row += [f"{entry['corrupted_subset']['n']}",
                    pct(entry["corrupted_subset"]["acc"]),
                    pct(entry["untouched_subset"]["acc"])]
        else:
            row += ["-", "-", "-"]
        if "paired" in entry:
            paired = entry["paired"]
            row += [f"{paired['broke']}/{paired['fixed']}",
                    f"{100 * paired['net_rate']:+.1f}", f"{paired['mcnemar_p']:.2e}"]
        else:
            row += ["-", "-", "-"]
        table_rows.append(row)

    table = md_table(
        ["level", "rate %", "n", f"global Acc@{args.threshold}",
         "n corrupted", "corrupted Acc", "untouched Acc",
         "broke/fixed", "net %", "McNemar p"],
        table_rows)
    print("\n" + table)

    lines = ["# Parse-error propagation\n",
             f"split: `{args.split}` | IoU threshold: {args.threshold} | "
             f"source: `{args.run_dir}`\n", table,
             "\n**global** dilutes the effect by (1 - rate): only the corrupted fraction "
             "was touched.\n",
             "**untouched Acc** is the control column — it should stay flat across "
             "levels.\n",
             "**broke/fixed** counts corrupted annotations that flipped against their own "
             "baseline outcome. `net %` is (broke - fixed) / n, and McNemar's test is the "
             "correct significance test for a paired binary flip.\n"]

    baseline_entry = next((r for r in results if r["rate"] == 0.0), None)
    verdicts = []
    if baseline_entry:
        worst = max((r for r in results if r["rate"] > 0),
                    key=lambda r: r["rate"], default=None)
        if worst:
            drop = baseline_entry["acc"] - worst["acc"]
            verdicts.append(
                f"At {100 * worst['rate']:.0f}% corruption the global accuracy falls "
                f"{100 * drop:.2f} pp (from {pct(baseline_entry['acc'])}% to "
                f"{pct(worst['acc'])}%).")
            if "paired" in worst:
                paired = worst["paired"]
                significant = paired["mcnemar_p"] == paired["mcnemar_p"] \
                    and paired["mcnemar_p"] < 0.05
                verdicts.append(
                    f"On the corrupted annotations themselves, {paired['broke']} flipped "
                    f"from correct to wrong and {paired['fixed']} the other way "
                    f"(net {100 * paired['net_rate']:+.1f}%, McNemar p="
                    f"{paired['mcnemar_p']:.2e})" +
                    (" — parse errors do propagate." if significant
                     else " — not statistically significant, so the model is fairly "
                          "robust to this corruption. Report that as the finding."))

    controls = [r["untouched_subset"]["acc"] for r in results if "untouched_subset" in r]
    if len(controls) > 1:
        spread = max(controls) - min(controls)
        verdicts.append(
            f"Control: accuracy on untouched annotations varies by {100 * spread:.2f} pp "
            f"across levels" +
            (" — flat, as it should be." if spread < 0.02 else
             " — this should be near zero; a large spread means the corruption affected "
             "annotations it was not supposed to, and the table cannot be trusted."))

    if verdicts:
        print("\nReading:")
        for verdict in verdicts:
            print(f"  {verdict}")
        lines.append("\n## Reading\n")
        lines += [f"- {v}" for v in verdicts]

    ensure_dir(args.out_dir)
    report_path = os.path.join(args.out_dir, "parse_error_propagation.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json({"run_dir": args.run_dir, "split": args.split,
                           "threshold": args.threshold, "levels": results,
                           "verdicts": verdicts},
                          os.path.join(args.out_dir, "parse_error_propagation.json"))
    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")

    if args.plot:
        path = make_plot(results, args.threshold,
                         os.path.join(args.out_dir, "parse_error_propagation.png"))
        if path:
            print(f"wrote {path}")
    return 0


def make_plot(results, threshold, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the plot")
        return None

    usable = [r for r in results if not math.isnan(r["rate"])]
    if len(usable) < 2:
        return None
    rates = [100 * r["rate"] for r in usable]

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(rates, [100 * r["acc"] for r in usable], marker="o", label="all annotations")
    if any("corrupted_subset" in r for r in usable):
        pairs = [(100 * r["rate"], 100 * r["corrupted_subset"]["acc"])
                 for r in usable if "corrupted_subset" in r]
        axis.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="s",
                  label="corrupted subset")
        pairs = [(100 * r["rate"], 100 * r["untouched_subset"]["acc"])
                 for r in usable if "untouched_subset" in r]
        axis.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="^",
                  linestyle="--", label="untouched subset (control)")

    axis.set_xlabel("parse corruption rate (%)")
    axis.set_ylabel(f"Acc@{threshold} (%)")
    axis.set_title("Grounding accuracy under deliberate parse corruption")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


if __name__ == "__main__":
    sys.exit(main())
