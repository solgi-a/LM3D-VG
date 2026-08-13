
import argparse
import fnmatch
import glob
import math
import os
import pickle
import re
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO_ROOT)

DEFAULT_METRICS = ("iou_rate_0.25", "iou_rate_0.5", "ref_acc", "lang_acc", "obj_acc")

_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(-?[0-9]*\.?[0-9]+)")
_TAGGED_LINE = re.compile(r"^\s*\[(best|loss|sco\.|train|val)\]")


def read_best(path):
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
    if mode == "all":
        return "all runs"
    tag = run_name.split("_", 2)[-1] if "_" in run_name else run_name
    if mode == "tag":
        return tag
    return re.sub(r"\d+$", "", tag) or tag


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n == 1:
        return mean, None, 1
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance), n


def format_cell(mean, std, n, scale=100.0, digits=2):
    if mean is None:
        return "n/a"
    if std is None:
        return f"{scale * mean:.{digits}f} (n=1)"
    return f"{scale * mean:.{digits}f}+/-{scale * std:.{digits}f}"


def mcnemar(a_hits, b_hits):
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, float("nan"), float("nan")
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    return a_only, b_only, chi2, math.erfc(math.sqrt(chi2 / 2.0))


def load_scores(run_dir, threshold):
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
            print("  NOTE: n=1. No variance can be estimated. Reviewer #4 comment 8 asks "
                  "precisely for\n        this; run at least two more seeds "
                  "(python experiments/ablation/runners/run_seeds.py).")
        elif counts == {2}:
            print("  NOTE: n=2. A standard deviation from two runs is a very weak "
                  "estimate. The\n        roadmap permits two seeds but never one; three "
                  "is materially better.")
        print()
        aggregated[group] = {"runs": [n for n, _p, _v in members], "stats": stats}

    if args.latex:
        print("LaTeX rows (values as percentages):")
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
            print(f"  {group} & " + " & ".join(cells) + r" \\")
        print()

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
