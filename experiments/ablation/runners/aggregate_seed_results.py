"""
Aggregate finished runs into mean +/- std.

    RUNS ON: CPU. Instant. Reads text files that training already wrote.

    # the seed sweep
    python experiments/ablation/runners/aggregate_seed_results.py --pattern 'ABL-SEED*'

    # any set of runs, e.g. every parser variant side by side
    python experiments/ablation/runners/aggregate_seed_results.py \
        --pattern '*ABL-PARSER-*' --group-by tag

``run_seeds.py`` launches the runs; this turns the resulting ``best.txt`` files into a
number and emits a LaTeX row alongside it.

What it reads
-------------
``outputs/<run>/best.txt`` -- written at the end of training, holding the best epoch's
metrics as ``[sco.] iou_rate_0.25: 0.49789, iou_rate_0.5: 0.3588`` and similar. Every
``key: value`` pair on a tagged line is collected, so metrics beyond the defaults are
available via ``--metrics``.

With ``--paired`` it also loads ``outputs/<run>/scores.p`` (written by ScanRefer_eval.py)
and, for exactly two groups, runs a paired per-annotation comparison: the same 9,508 val
items under both models, with McNemar's test. That is a sharper instrument than comparing
two means over three seeds when the difference at stake is under a point.

With one run the standard deviation is undefined and the script says so rather than
printing 0.00. With two it is reported but flagged as a weak estimate.
"""

import argparse
import fnmatch
import glob
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict

# experiments/ablation/runners/<this file> -> repo root is 4 levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

DEFAULT_METRICS = ("iou_rate_0.25", "iou_rate_0.5", "ref_acc", "lang_acc", "obj_acc")

#: `[sco.] iou_rate_0.25: 0.49789, iou_rate_0.5: 0.3588` -> both pairs.
_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(-?[0-9]*\.?[0-9]+)")
_TAGGED_LINE = re.compile(r"^\s*\[(best|loss|sco\.|train|val)\]")


def read_best(path):
    """Every key: value pair in a best.txt, as {key: float}."""
    values = {}
    with open(path) as f:
        for line in f:
            if not _TAGGED_LINE.match(line):
                continue
            for key, value in _PAIR.findall(line):
                try:
                    values[key] = float(value)
                except ValueError:
                    continue
    return values


def discover(output_root, pattern):
    """[(run_name, best.txt path)] for runs matching the glob pattern."""
    root = output_root if os.path.isabs(output_root) \
        else os.path.join(REPO_ROOT, output_root)
    found = []
    for path in sorted(glob.glob(os.path.join(root, "*", "best.txt"))):
        name = os.path.basename(os.path.dirname(path))
        if pattern and not fnmatch.fnmatch(name, pattern):
            continue
        found.append((name, path))
    return found


def group_name(run_name, mode):
    """How runs are pooled. 'all' pools everything; 'tag' pools by trailing tag."""
    if mode == "all":
        return "all runs"
    # outputs are named "<timestamp>_<TAG>"; the tag is what identifies a configuration,
    # and seeds of one configuration differ only by a trailing digit.
    tag = run_name.split("_", 2)[-1] if "_" in run_name else run_name
    if mode == "tag":
        return tag
    return re.sub(r"\d+$", "", tag) or tag          # mode == "config": strip the seed


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n == 1:
        return mean, None, 1
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)   # sample std
    return mean, math.sqrt(variance), n


def format_cell(mean, std, n, scale=100.0, digits=2):
    """Short enough to stay inside a table column; the n=1 caveat is printed separately."""
    if mean is None:
        return "n/a"
    if std is None:
        return f"{scale * mean:.{digits}f} (n=1)"
    return f"{scale * mean:.{digits}f}+/-{scale * std:.{digits}f}"


def mcnemar(a_hits, b_hits):
    """Paired binary comparison. Returns (a_only, b_only, chi2, p)."""
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, float("nan"), float("nan")
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    return a_only, b_only, chi2, math.erfc(math.sqrt(chi2 / 2.0))


def load_scores(run_dir, threshold):
    """Per-annotation hit indicators from scores.p, or None when unavailable."""
    path = os.path.join(run_dir, "scores.p")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        scores = pickle.load(f)
    ious = scores.get("ious")
    if not ious:
        return None
    return [1 if float(i) >= threshold else 0 for i in ious[0]]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", dest="output_root", default="outputs")
    parser.add_argument("--pattern", default="*ABL-SEED*",
                        help="Glob over run folder names under outputs/.")
    parser.add_argument("--group-by", dest="group_by", default="all",
                        choices=["all", "tag", "config"],
                        help="'all' pools every match into one mean; 'tag' keeps each "
                             "run folder's tag separate; 'config' strips a trailing "
                             "seed number so ABL-SEED1/2/3 pool together.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="IoU threshold for the paired comparison.")
    parser.add_argument("--paired", action="store_true",
                        help="Also run a paired per-annotation McNemar test between the "
                             "two groups (requires scores.p and exactly two groups).")
    parser.add_argument("--latex", action="store_true", help="Emit a LaTeX table row.")
    parser.add_argument("--out-dir", dest="out_dir",
                        default=os.path.join("outputs", "analysis", "seed_aggregate"),
                        help="Where to write the aggregate.")
    args = parser.parse_args()

    runs = discover(args.output_root, args.pattern)
    if not runs:
        print(f"No runs matched {args.pattern!r} under {args.output_root}/")
        print("Finished runs write outputs/<timestamp>_<TAG>/best.txt.")
        print("Launch the seed sweep with: python experiments/ablation/runners/run_seeds.py")
        return 1

    groups = defaultdict(list)
    for name, path in runs:
        values = read_best(path)
        if not values:
            print(f"[warn] {name}: best.txt has no parsable metrics; skipped")
            continue
        groups[group_name(name, args.group_by)].append((name, path, values))

    print(f"{len(runs)} run(s) matched {args.pattern!r}, "
          f"{len(groups)} group(s) by --group-by {args.group_by}\n")

    aggregated = {}
    for group, members in sorted(groups.items()):
        print(f"=== {group}  ({len(members)} run{'s' if len(members) != 1 else ''}) ===")
        width = max(len(name) for name, _p, _v in members)
        header = "  " + "run".ljust(width) + "  " + "  ".join(
            m.rjust(14) for m in args.metrics)
        print(header)
        for name, _path, values in sorted(members):
            cells = "  ".join(
                (f"{100 * values[m]:.2f}" if m in values else "-").rjust(14)
                for m in args.metrics)
            print("  " + name.ljust(width) + "  " + cells)

        stats = {}
        for metric in args.metrics:
            series = [values[metric] for _n, _p, values in members if metric in values]
            mean, std, n = mean_std(series)
            stats[metric] = {"mean": mean, "std": std, "n": n, "values": series}

        print("  " + "-" * (width + 2 + 16 * len(args.metrics)))
        print("  " + "mean +/- std".ljust(width) + "  " + "  ".join(
            format_cell(stats[m]["mean"], stats[m]["std"], stats[m]["n"]).rjust(14)
            for m in args.metrics))

        counts = {stats[m]["n"] for m in args.metrics}
        if counts == {1}:
            print("  NOTE: n=1. No variance can be estimated; run at least two more "
                  "seeds\n        (python experiments/ablation/runners/run_seeds.py).")
        elif counts == {2}:
            print("  NOTE: n=2. A standard deviation from two runs is a very weak "
                  "estimate;\n        three is materially better.")
        print()
        aggregated[group] = {"runs": [n for n, _p, _v in members], "stats": stats}

    latex_rows = []
    for group, data in sorted(aggregated.items()):
        cells = []
        for metric in args.metrics:
            entry = data["stats"][metric]
            if entry["mean"] is None:
                cells.append("--")
            elif entry["std"] is None:
                cells.append(f"{100 * entry['mean']:.2f}")
            else:
                cells.append(f"{100 * entry['mean']:.2f} $\\pm$ "
                             f"{100 * entry['std']:.2f}")
        latex_rows.append(f"  {group} & " + " & ".join(cells) + r" \\")

    if args.latex:
        print("LaTeX rows (values as percentages):")
        for row in latex_rows:
            print(row)
        print()

    paired_result = None
    if args.paired:
        if len(groups) != 2:
            print(f"--paired needs exactly two groups; got {len(groups)}. "
                  f"Use --group-by tag or narrow --pattern.")
        else:
            (name_a, members_a), (name_b, members_b) = sorted(groups.items())
            hits_a = load_scores(os.path.dirname(members_a[0][1]), args.threshold)
            hits_b = load_scores(os.path.dirname(members_b[0][1]), args.threshold)
            if hits_a is None or hits_b is None:
                print("--paired needs scores.p in both runs. Produce it with:")
                print("    python scripts/ScanRefer_eval.py --folder <run> --reference --force "
                      "--use_color --use_normal --lang_num_max 1")
            elif len(hits_a) != len(hits_b):
                print(f"--paired needs equal-length score vectors; got "
                      f"{len(hits_a)} vs {len(hits_b)}.")
            else:
                only_a, only_b, chi2, p = mcnemar(hits_a, hits_b)
                acc_a = sum(hits_a) / len(hits_a)
                acc_b = sum(hits_b) / len(hits_b)
                paired_result = {
                    "group_a": name_a, "group_b": name_b, "n": len(hits_a),
                    "threshold": args.threshold, "acc_a": acc_a, "acc_b": acc_b,
                    "difference": acc_a - acc_b, "only_a_correct": only_a,
                    "only_b_correct": only_b, "mcnemar_chi2": chi2, "p_value": p,
                    "significant": bool(p == p and p < 0.05),
                }
                print(f"Paired comparison at IoU {args.threshold} over {len(hits_a)} "
                      f"annotations:")
                print(f"  {name_a}: {100 * acc_a:.2f}%   {name_b}: {100 * acc_b:.2f}%   "
                      f"difference {100 * (acc_a - acc_b):+.2f} pp")
                print(f"  only {name_a} correct: {only_a}   only {name_b} correct: "
                      f"{only_b}")
                print(f"  McNemar chi2={chi2:.3f}, p={p:.3e}  -> " +
                      ("the difference is statistically significant"
                       if p == p and p < 0.05 else
                       "NOT significant; do not claim this difference"))

    # ---- persist ---------------------------------------------------------------------
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "pattern": args.pattern,
        "group_by": args.group_by,
        "metrics": list(args.metrics),
        "threshold": args.threshold,
        "num_runs": len(runs),
        "groups": aggregated,
        "latex_rows": latex_rows,
        "paired": paired_result,
    }
    json_path = os.path.join(out_dir, "seed_aggregate.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Seed aggregate", "",
             f"pattern `{args.pattern}` | {len(runs)} run(s) | grouped by `{args.group_by}`",
             ""]
    for group, data in sorted(aggregated.items()):
        stats = data["stats"]
        n = max((stats[m]["n"] or 0) for m in args.metrics)
        lines += [f"## {group}  (n = {n})", "",
                  "| metric | mean | std | n | values |", "|---|---|---|---|---|"]
        for m in args.metrics:
            e = stats[m]
            mean = f"{100 * e['mean']:.2f}" if e["mean"] is not None else "--"
            std = f"{100 * e['std']:.2f}" if e["std"] is not None else "--"
            vals = ", ".join(f"{100 * v:.2f}" for v in e["values"]) or "--"
            lines.append(f"| `{m}` | {mean} | {std} | {e['n']} | {vals} |")
        lines += ["", "runs: " + ", ".join(f"`{r}`" for r in data["runs"]), ""]
        if n == 1:
            lines += ["> **n=1.** No variance can be estimated from a single run.", ""]
        elif n == 2:
            lines += ["> **n=2.** A standard deviation from two runs is a very weak "
                      "estimate; three seeds is materially better.", ""]
    if latex_rows:
        lines += ["## LaTeX", "", "```", *latex_rows, "```", ""]
    if paired_result:
        pr = paired_result
        lines += ["## Paired comparison", "",
                  f"`{pr['group_a']}` vs `{pr['group_b']}` at IoU {pr['threshold']} "
                  f"over {pr['n']:,} annotations", "",
                  f"- {pr['group_a']}: {100 * pr['acc_a']:.2f}%  ·  "
                  f"{pr['group_b']}: {100 * pr['acc_b']:.2f}%  ·  "
                  f"difference {100 * pr['difference']:+.2f} pp",
                  f"- McNemar chi2 = {pr['mcnemar_chi2']:.3f}, p = {pr['p_value']:.3e} — "
                  + ("significant" if pr["significant"] else
                     "**not** significant; do not claim this difference"), ""]
    md_path = os.path.join(out_dir, "seed_aggregate.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
