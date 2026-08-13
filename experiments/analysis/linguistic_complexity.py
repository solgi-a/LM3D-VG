
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.analysis.common import (
    THRESHOLDS,
    accuracy,
    bootstrap_ci,
    ensure_dir,
    find_default_predictions,
    join,
    load_parse_cache,
    load_predictions,
    load_scanrefer,
    mcnemar,
    md_table,
    parse_for,
    parse_predictions_arg,
    pct,
    quantile_edges,
    save_json,
    spearman,
    wilson_ci,
)

METRICS = ("tokens", "depth", "neighbors", "spatial")


def metric_tokens(rows, _args):
    return [len(row.get("token") or []) for row in rows]


def metric_depth(rows, args):
    cache_path = args.depth_cache
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    def key_of(row):
        return f"{row['scene_id']}|{row['object_id']}|{row['ann_id']}"

    todo = [row for row in rows if key_of(row) not in cache]
    if todo:
        try:
            from experiments.ablation.parsers.spacy_parser import load_parser
        except ImportError as error:
            raise SystemExit(f"the 'depth' metric needs spaCy: {error}")
        print(f"  computing dependency depth for {len(todo)} descriptions "
              f"({len(cache)} cached)...")
        nlp = load_parser(args.spacy_model)
        texts = [(row.get("description") or "").lower() for row in todo]
        for row, doc in zip(todo, nlp.pipe(texts, batch_size=256)):
            cache[key_of(row)] = _max_depth(doc)
        if cache_path:
            ensure_dir(os.path.dirname(os.path.abspath(cache_path)))
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"  cached depths -> {cache_path}")

    return [cache[key_of(row)] for row in rows]


def _max_depth(doc):
    best = 0
    for token in doc:
        depth, node, guard = 0, token, 0
        while node.head.i != node.i and guard < len(doc) + 1:
            depth += 1
            node = node.head
            guard += 1
        best = max(best, depth)
    return best


def metric_neighbors(rows, args):
    cache = load_parse_cache(args.parse_folder, args.split, args.parsing_root)
    from experiments.ablation.parsers.tokenize_parse import NOT_MENTIONED_TOKENS

    values = []
    for row in rows:
        parsed = parse_for(cache, row["scene_id"], row["object_id"], row["ann_id"])
        if not parsed:
            values.append(0)
            continue
        tokens = parsed.get("neighbors") or []
        if not tokens or tokens == NOT_MENTIONED_TOKENS:
            values.append(0)
            continue
        values.append(tokens.count(",") + 1)
    return values


def _spatial_lexicon():
    from experiments.ablation.parsers.spacy_parser import SPATIAL_PHRASES, SPATIAL_PREPS

    phrases = sorted((p.split() for p in SPATIAL_PHRASES), key=len, reverse=True)
    return phrases, set(SPATIAL_PREPS)


def spatial_hits(tokens):
    phrases, preps = _spatial_lexicon()
    lowered = [str(t).lower() for t in tokens]
    hits, i = [], 0
    while i < len(lowered):
        matched = None
        for phrase in phrases:
            if lowered[i: i + len(phrase)] == phrase:
                matched = " ".join(phrase)
                i += len(phrase)
                break
        if matched is None:
            if lowered[i] in preps:
                matched = lowered[i]
            i += 1
        if matched:
            hits.append(matched)
    return hits


def metric_spatial(rows, _args):
    return [len(spatial_hits(row.get("token") or [])) for row in rows]


METRIC_FUNCTIONS = {
    "tokens": metric_tokens,
    "depth": metric_depth,
    "neighbors": metric_neighbors,
    "spatial": metric_spatial,
}

METRIC_LABELS = {
    "tokens": "description length (tokens)",
    "depth": "dependency-tree depth",
    "neighbors": "adjacent objects extracted by our parser",
    "spatial": "spatial-relation words (fixed lexicon)",
}

DISCRETE_METRICS = {"neighbors": 3, "spatial": 3}


def assign_bins(values, metric, num_bins):
    values = np.asarray(values)
    if metric in DISCRETE_METRICS:
        top = DISCRETE_METRICS[metric]
        indices = np.minimum(values, top).astype(int)
        labels = [str(i) for i in range(top)] + [f"{top}+"]
        return indices, labels

    edges = quantile_edges(values, num_bins)
    if not edges:
        return np.zeros(len(values), dtype=int), ["all"]

    indices = np.digitize(values, edges, right=True)
    labels, low = [], int(values.min())
    for edge in edges:
        labels.append(f"{low}-{int(edge)}")
        low = int(edge) + 1
    labels.append(f"{low}+")
    return indices.astype(int), labels


def summarise(rows, indices, labels, models, thresholds):
    bins = []
    for index, label in enumerate(labels):
        mask = indices == index
        selected = [row for row, keep in zip(rows, mask) if keep]
        entry = {"bin": label, "n": len(selected), "models": {}}
        for name in models:
            ious = [row["ious"][name] for row in selected if name in row["ious"]]
            per_model = {"n": len(ious)}
            for threshold in thresholds:
                key = f"acc_{threshold}"
                value = accuracy(ious, threshold)
                hits = int(sum(1 for i in ious if i >= threshold))
                low, high = wilson_ci(hits, len(ious))
                per_model[key] = value
                per_model[f"{key}_ci"] = [low, high]
            entry["models"][name] = per_model
        bins.append(entry)
    return bins


def render_metric(metric, labels, bins, models, threshold, rows, values,
                  bin_indices, bootstrap=2000, seed=42):
    reference = models[0]
    others = models[1:]

    headers = ["bin", "n"] + [f"{m} Acc@{threshold}" for m in models]
    headers += [f"gap {reference}-{m}" for m in others]

    table_rows = []
    for entry in bins:
        line = [entry["bin"], entry["n"]]
        for name in models:
            stats = entry["models"][name]
            low, high = stats[f"acc_{threshold}_ci"]
            line.append(f"{pct(stats[f'acc_{threshold}'])} "
                        f"[{pct(low, 1)}-{pct(high, 1)}]")
        for name in others:
            gap = (entry["models"][reference][f"acc_{threshold}"]
                   - entry["models"][name][f"acc_{threshold}"])
            line.append(f"{100 * gap:+.2f}")
        table_rows.append(line)

    findings = {"threshold": threshold, "trends": {}, "gaps": {},
                "test": "per-sample Spearman (complexity value vs hit indicator)"}

    def hits(name):
        return [1.0 if row["ious"].get(name, -1.0) >= threshold else 0.0 for row in rows]

    reference_hits = hits(reference)
    rho, p = spearman(list(values), reference_hits)
    findings["trends"][reference] = {
        "spearman_rho": rho, "p_value": p, "n": len(rows),
        "significant": bool(p == p and p < 0.05),
    }

    for name in others:
        other_hits = hits(name)
        advantage = [a - b for a, b in zip(reference_hits, other_hits)]
        rho_gap, p_gap = spearman(list(values), advantage)

        reference_array = np.asarray(reference_hits, dtype=bool)
        other_array = np.asarray(other_hits, dtype=bool)
        indices = np.asarray(bin_indices)

        per_bin = []
        for index in range(len(bins)):
            mask = indices == index
            if not mask.any():
                per_bin.append({"gap": 0.0, "n": 0, "p_value": float("nan")})
                continue
            a, b = reference_array[mask], other_array[mask]
            _a_only, _b_only, _chi2, p_bin = mcnemar(a, b)
            per_bin.append({"gap": float(a.mean() - b.mean()), "n": int(mask.sum()),
                            "p_value": p_bin})

        populated = [i for i, entry in enumerate(per_bin) if entry["n"] > 0]
        widening = {"available": False}
        if len(populated) >= 2:
            low_bin, high_bin = populated[0], populated[-1]

            def difference_of_gaps(idx, low=low_bin, high=high_bin):
                sub_indices = indices[idx]
                a, b = reference_array[idx], other_array[idx]
                low_mask, high_mask = sub_indices == low, sub_indices == high
                if not low_mask.any() or not high_mask.any():
                    return float("nan")
                return float((a[high_mask].mean() - b[high_mask].mean())
                             - (a[low_mask].mean() - b[low_mask].mean()))

            point, ci_low, ci_high = bootstrap_ci(
                difference_of_gaps, len(rows), num_resamples=bootstrap, seed=seed)
            widening = {
                "available": True,
                "low_bin": bins[low_bin]["bin"], "high_bin": bins[high_bin]["bin"],
                "difference": point, "ci": [ci_low, ci_high],
                "significant": bool(ci_low > 0),
                "directional": bool(point > 0),
                "resamples": bootstrap,
            }

        findings["gaps"][name] = {
            "per_bin": [entry["gap"] for entry in per_bin],
            "per_bin_detail": per_bin,
            "spearman_rho": rho_gap,
            "p_value": p_gap,
            "n": len(rows),
            "widening": widening,
            "widens_with_complexity": bool(widening.get("significant")),
        }

    return md_table(headers, table_rows), findings


def verdict_lines(metric, findings, models):
    reference = models[0]
    lines = []
    trend = findings["trends"][reference]
    rho, p, n = trend["spearman_rho"], trend["p_value"], trend["n"]
    direction = "falls" if rho < 0 else "rises"
    strength = "significantly" if trend["significant"] else "but NOT significantly"
    lines.append(f"  {reference}: accuracy {direction} as {METRIC_LABELS[metric]} grows, "
                 f"{strength} (per-sample Spearman rho={rho:+.4f}, p={p:.2e}, n={n}).")

    if not findings["gaps"]:
        lines.append("  No baseline supplied, so this says nothing about whether the "
                     "advantage over prior work grows with complexity -- which is the "
                     "actual claim the reviewer is testing. Add --predictions "
                     "NAME=PATH for a baseline.")
        return lines

    for name, gap in findings["gaps"].items():
        widening = gap.get("widening") or {}
        if not widening.get("available"):
            lines.append(f"  vs {name}: too few populated bins to test widening.")
            continue

        low, high = widening["ci"]
        summary = (f"gap in '{widening['high_bin']}' minus gap in "
                   f"'{widening['low_bin']}' = {100 * widening['difference']:+.2f} pp, "
                   f"95% CI [{100 * low:+.2f}, {100 * high:+.2f}]")

        if widening["significant"]:
            lines.append(f"  vs {name}: the advantage WIDENS with complexity -- "
                         f"{summary}. The CI excludes zero, so this supports the paper's "
                         f"motivation directly. Report it.")
        elif widening["directional"]:
            lines.append(
                f"  vs {name}: the advantage widens DIRECTIONALLY but not significantly "
                f"-- {summary}. The CI includes zero. State the trend and the CI, and "
                f"soften the claim to 'consistent with' rather than 'demonstrates'. Do "
                f"not report the point estimate alone.")
        else:
            lines.append(f"  vs {name}: the advantage does NOT widen with complexity -- "
                         f"{summary}. Report this honestly and drop the claim that the "
                         f"method's advantage is specific to complex language.")

        significant_bins = [entry for entry in gap["per_bin_detail"]
                            if entry["p_value"] == entry["p_value"]
                            and entry["p_value"] < 0.05]
        lines.append(f"      per-bin gaps (McNemar): " + "  ".join(
            f"{100 * entry['gap']:+.2f}{'*' if entry['p_value'] == entry['p_value'] and entry['p_value'] < 0.05 else ''}"
            for entry in gap["per_bin_detail"]) +
            f"   ({len(significant_bins)}/{len(gap['per_bin_detail'])} bins significant, "
            f"* = p<0.05)")
    return lines


def relation_type_table(rows, models, threshold, top_k):
    from collections import Counter

    counts = Counter()
    per_type = {}
    for row in rows:
        for relation in set(spatial_hits(row.get("token") or [])):
            counts[relation] += 1
            per_type.setdefault(relation, []).append(row)

    headers = ["relation", "n"] + [f"{m} Acc@{threshold}" for m in models]
    table_rows = []
    for relation, count in counts.most_common(top_k):
        line = [relation, count]
        for name in models:
            ious = [r["ious"][name] for r in per_type[relation] if name in r["ious"]]
            line.append(pct(accuracy(ious, threshold)))
        table_rows.append(line)
    return md_table(headers, table_rows), dict(counts.most_common(top_k))


def make_plot(results, models, threshold, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the plot")
        return None

    metrics = list(results)
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2),
                                squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        bins = results[metric]["bins"]
        labels = [entry["bin"] for entry in bins]
        x = range(len(labels))

        counts = [entry["n"] for entry in bins]
        twin = axis.twinx()
        twin.bar(x, counts, color="0.88", zorder=0)
        twin.set_ylabel("samples", color="0.5")
        twin.tick_params(axis="y", colors="0.5")

        for name in models:
            y = [100 * entry["models"][name][f"acc_{threshold}"] for entry in bins]
            axis.plot(x, y, marker="o", label=name, zorder=3)

        axis.set_zorder(twin.get_zorder() + 1)
        axis.patch.set_visible(False)
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels, rotation=0)
        axis.set_xlabel(METRIC_LABELS[metric])
        axis.set_ylabel(f"Acc@{threshold} (%)")
        axis.grid(alpha=0.3)
        axis.legend(loc="lower left", fontsize=8)

    figure.suptitle(f"Grounding accuracy by linguistic complexity (Acc@{threshold})")
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", action="append", default=None, metavar="NAME=PATH",
                        help="Repeatable. The first one is the reference model in every "
                             "gap column. Defaults to the newest outputs/*/predictions.p.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--num-bins", dest="num_bins", type=int, default=4,
                        help="Quantile bins for the continuous metrics (tokens, depth).")
    parser.add_argument("--threshold", type=float, default=0.25, choices=list(THRESHOLDS),
                        help="IoU threshold used for the tables, plot and verdicts.")
    parser.add_argument("--parse-folder", dest="parse_folder",
                        default="final_parsing_tokenized",
                        help="Parse cache backing the 'neighbors' metric.")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--spacy-model", dest="spacy_model", default="en_core_web_sm")
    parser.add_argument("--depth-cache", dest="depth_cache",
                        default="outputs/analysis/dep_depth_cache.json")
    parser.add_argument("--top-relations", dest="top_relations", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for the widening CI; 0 is not "
                             "recommended -- the CI is the headline statistic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/linguistic_complexity")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    specs = args.predictions
    if not specs:
        found = find_default_predictions()
        if not found:
            parser.error("no --predictions given and no outputs/*/predictions.p found")
        specs = [found[0]]
        print(f"[auto] using {found[0]}")

    records = load_scanrefer(args.split, args.data_root)

    models, base_rows = [], None
    for spec in specs:
        name, path = parse_predictions_arg(spec)
        rows, missing = join(load_predictions(path), records)
        print(f"[{name}] {len(rows)} joined from {path}" +
              (f"  ({missing} annotations had no prediction)" if missing else ""))
        models.append(name)
        if base_rows is None:
            base_rows = {r_key(row): dict(row, ious={}) for row in rows}
        for row in rows:
            entry = base_rows.get(r_key(row))
            if entry is not None:
                entry["ious"][name] = row["iou"]

    rows = list(base_rows.values())
    if len(models) > 1:
        complete = [row for row in rows if len(row["ious"]) == len(models)]
        if len(complete) != len(rows):
            print(f"[join] restricting to {len(complete)}/{len(rows)} annotations "
                  f"predicted by all {len(models)} models")
        rows = complete

    ensure_dir(args.out_dir)
    results, report_lines = {}, []
    report_lines.append("# Grounding accuracy by linguistic complexity\n")
    report_lines.append(f"split: `{args.split}` | annotations: {len(rows)} | "
                        f"models: {', '.join(models)} | IoU threshold: {args.threshold}\n")

    for metric in args.metrics:
        print(f"\n=== {metric}: {METRIC_LABELS[metric]} ===")
        values = METRIC_FUNCTIONS[metric](rows, args)
        indices, labels = assign_bins(values, metric, args.num_bins)
        bins = summarise(rows, indices, labels, models, THRESHOLDS)
        table, findings = render_metric(metric, labels, bins, models, args.threshold,
                                        rows, values, indices, args.bootstrap,
                                        args.seed)

        print(table)
        lines = verdict_lines(metric, findings, models)
        for line in lines:
            print(line)

        results[metric] = {
            "label": METRIC_LABELS[metric],
            "values_summary": {
                "min": int(np.min(values)), "max": int(np.max(values)),
                "mean": float(np.mean(values)), "median": float(np.median(values)),
            },
            "bins": bins,
            "findings": findings,
        }
        report_lines.append(f"\n## {metric} — {METRIC_LABELS[metric]}\n")
        report_lines.append(table)
        report_lines.append("\n**Reading:**\n")
        report_lines.extend(line.strip() + "\n" for line in lines)

    if "spatial" in args.metrics:
        print("\n=== accuracy per spatial-relation type ===")
        table, counts = relation_type_table(rows, models, args.threshold, args.top_relations)
        print(table)
        results["relation_types"] = {"counts": counts}
        report_lines.append("\n## Accuracy by spatial-relation type\n")
        report_lines.append("Reviewer #4 asked for the *number and type* of spatial "
                            "relations; the metric above covers number, this covers type.\n")
        report_lines.append(table)

    json_path = save_json({
        "split": args.split,
        "models": models,
        "num_annotations": len(rows),
        "threshold": args.threshold,
        "num_bins": args.num_bins,
        "results": results,
    }, os.path.join(args.out_dir, "linguistic_complexity.json"))

    report_path = os.path.join(args.out_dir, "linguistic_complexity.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nwrote {json_path}")
    print(f"wrote {report_path}")

    if args.plot:
        plot_metrics = {m: results[m] for m in args.metrics}
        plot_path = make_plot(plot_metrics, models, args.threshold,
                              os.path.join(args.out_dir, "linguistic_complexity.png"))
        if plot_path:
            print(f"wrote {plot_path}")

    if len(models) == 1:
        print("\nNOTE: only one model was supplied. The reviewer's question is whether "
              "the\n      advantage over a baseline grows with complexity, which needs a "
              "second\n      --predictions NAME=PATH. See the docstring for the required "
              "format.")
    return 0


def r_key(row):
    return (str(row["scene_id"]), str(row["object_id"]), str(row["ann_id"]))


if __name__ == "__main__":
    sys.exit(main())
