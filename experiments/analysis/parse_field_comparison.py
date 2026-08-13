"""
Adjectives and neighbors, scored across parsers -- the two fields nothing else measures.

    RUNS ON: CPU. Seconds. Reads only the parse caches and ScanRefer's JSON.

    python experiments/analysis/parse_field_comparison.py \
        --parse gpt4o-mini=final_parsing_tokenized \
        --parse llama=llama_parsing_tokenized_clipped \
        --parse spacy=spacy_parsing_tokenized

``eval_parser_target_accuracy.py`` scores the **target** field against ``object_name``. The
other two have no ground truth, yet they are what the fusion network consumes -- TAF reads
the attribute field and A2F the adjacency field. This scores them reference-free, three
ways:

    coverage      how often the parser declines the slot ("not mentioned").
    faithfulness  fraction of emitted tokens occurring in the source description.
                  Catches invention.
    agreement     mean per-annotation Jaccard against every other parser.

GPT-4o-mini and LLaMA-3 were built independently, so where they concur forms a reference
band and a parser far outside it is the outlier. That is weaker than ground truth.

Faithfulness alone is not a quality signal: a rule-based parser copies tokens verbatim and
scores near 100% by construction, and a parser emitting nothing scores perfectly. It only
means something read next to coverage.

Writes outputs/analysis/parse_field_comparison/{md,json,png}. Scope is parse quality; no
model is run and no prediction is read.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.analysis.common import (
    ensure_dir, load_parse_cache, load_scanrefer, md_table, parse_for, save_json,
    wilson_ci)
from experiments.ablation.parsers.tokenize_parse import (
    FIELD_MAX_TOKENS, NOT_MENTIONED_TOKENS)

#: The two fields this script exists for. `target` is deliberately absent -- it already
#: has a scorer with real ground truth, and repeating it here would invite quoting a
#: reference-free number when a referenced one exists.
FIELDS = ("adjectives", "neighbors")


def key_of(record):
    return (str(record["scene_id"]), str(record["object_id"]), str(record["ann_id"]))


def is_empty(tokens):
    """True when the parser declined the slot."""
    return not tokens or list(tokens) == list(NOT_MENTIONED_TOKENS)


def jaccard(a, b):
    """Token-set overlap. Two declined fields agree perfectly -- that is a real agreement."""
    set_a, set_b = set(a or []), set(b or [])
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def phrase_count(tokens):
    """Comma-separated segments, exactly as linguistic_complexity.metric_neighbors counts
    them, so the numbers here line up with that report."""
    if is_empty(tokens):
        return 0
    return list(tokens).count(",") + 1


# ======================================================================================

def collect(caches, records):
    """Per-parser per-field statistics over the annotations every parser covers.

    Restricting to the common key set is what makes the comparison paired: a parser that
    simply skipped the hard descriptions must not look good by having a smaller, easier
    denominator.
    """
    keys, parses = [], {name: {} for name in caches}
    for record in records:
        key = key_of(record)
        found = {name: parse_for(cache, *key) for name, cache in caches.items()}
        if any(parse is None for parse in found.values()):
            continue
        keys.append(key)
        for name, parse in found.items():
            parses[name][key] = parse
    return keys, parses


def field_stats(keys, parses, records_by_key, name, field):
    empty = faithful_hits = faithful_total = 0
    lengths, phrases, capped = [], [], 0
    for key in keys:
        tokens = parses[name][key].get(field) or []
        if is_empty(tokens):
            empty += 1
        else:
            source = {t.lower() for t in (records_by_key[key].get("token") or [])}
            faithful_hits += sum(1 for t in tokens if str(t).lower() in source)
            faithful_total += len(tokens)
            lengths.append(len(tokens))
        phrases.append(phrase_count(tokens))
        if len(tokens) >= FIELD_MAX_TOKENS[field]:
            capped += 1

    total = len(keys)
    low, high = wilson_ci(empty, total)
    return {
        "n": total,
        "empty": empty,
        "empty_rate": empty / total if total else 0.0,
        "empty_ci": [low, high],
        "capped": capped,
        "mean_tokens_when_present": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "mean_phrases": (sum(phrases) / total) if total else 0.0,
        "faithfulness": (faithful_hits / faithful_total) if faithful_total else float("nan"),
        "faithful_tokens": faithful_hits,
        "emitted_tokens": faithful_total,
    }


def agreement_matrix(keys, parses, names, field):
    """Mean per-annotation Jaccard for every parser pair."""
    matrix = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            score = sum(jaccard(parses[a][k].get(field), parses[b][k].get(field))
                        for k in keys) / max(len(keys), 1)
            matrix[f"{a} vs {b}"] = score
    return matrix


def consensus_agreement(keys, parses, names, field, reference_names):
    """Each parser's mean Jaccard against the reference parsers (excluding itself)."""
    scores = {}
    for name in names:
        others = [r for r in reference_names if r != name]
        if not others:
            scores[name] = float("nan")
            continue
        scores[name] = sum(
            sum(jaccard(parses[name][k].get(field), parses[other][k].get(field))
                for other in others) / len(others)
            for k in keys) / max(len(keys), 1)
    return scores


# ======================================================================================

def make_plot(results, names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the plot")
        return None

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    x = range(len(names))

    axis = axes[0][0]
    width = 0.38
    for offset, field in zip((-width / 2, width / 2), FIELDS):
        y = [100 * results[field]["parsers"][n]["empty_rate"] for n in names]
        axis.bar([i + offset for i in x], y, width, label=field)
    axis.set_ylabel("declined the slot (%)")
    axis.set_title("Coverage -- lower is better")

    axis = axes[0][1]
    for offset, field in zip((-width / 2, width / 2), FIELDS):
        y = [100 * results[field]["parsers"][n]["faithfulness"] for n in names]
        axis.bar([i + offset for i in x], y, width, label=field)
    axis.set_ylim(80, 101)
    axis.set_ylabel("tokens found in the description (%)")
    axis.set_title("Faithfulness -- near-tautological\nfor a copy-only parser")

    axis = axes[0][2]
    for offset, field in zip((-width / 2, width / 2), FIELDS):
        y = [results[field]["consensus"][n] for n in names]
        axis.bar([i + offset for i in x], y, width, label=field)
    axis.set_ylabel("mean Jaccard vs the other parsers")
    axis.set_title("Agreement -- higher is better")

    for axis in axes[0]:
        axis.set_xticks(list(x))
        axis.set_xticklabels(names, rotation=15, ha="right")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)

    figure.suptitle("Parser comparison on the two fields with no ground truth")
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parse", action="append", default=[], metavar="NAME=FOLDER",
                        help="Repeatable. Parse cache under data_parsing/.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--reference", action="append", default=None, metavar="NAME",
                        help="Parsers forming the consensus band. Default: every parser.")
    parser.add_argument("--out-dir", dest="out_dir",
                        default=os.path.join("outputs", "analysis", "parse_field_comparison"))
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    if not args.parse:
        parser.error("at least one --parse NAME=FOLDER is required")

    caches, folders = {}, {}
    for item in args.parse:
        name, _, folder = item.partition("=")
        if not folder:
            name, folder = os.path.basename(item), item
        try:
            caches[name] = load_parse_cache(folder, args.split, args.parsing_root)
            folders[name] = folder
        except FileNotFoundError as error:
            # Skipping rather than dying: a missing optional variant should not stop the
            # report on the ones that are present.
            print(f"  skipping {name}: {error}")
    if not caches:
        print("no parse cache could be loaded")
        return 1

    names = list(caches)
    references = [n for n in (args.reference or names) if n in caches] or names

    records = load_scanrefer(args.split, args.data_root)
    records_by_key = {key_of(r): r for r in records}
    keys, parses = collect(caches, records)
    if not keys:
        print("no annotation is covered by every parse cache")
        return 1

    print(f"split: {args.split} | annotations covered by all parsers: {len(keys)} "
          f"of {len(records)}")
    print(f"parsers: {', '.join(names)}")
    print(f"consensus reference: {', '.join(references)}\n")

    lines = ["# Parser comparison -- adjectives and neighbors", "",
             f"split: `{args.split}` | annotations: {len(keys)} (covered by every parser) "
             f"| parsers: {', '.join(names)}", "",
             "These two fields have **no ground truth** -- ScanRefer's `object_name` only "
             "grounds the `target` field. Every number below is reference-free.", ""]

    results = {}
    for field in FIELDS:
        stats = {n: field_stats(keys, parses, records_by_key, n, field) for n in names}
        pairs = agreement_matrix(keys, parses, names, field)
        consensus = consensus_agreement(keys, parses, names, field, references)
        results[field] = {"parsers": stats, "pairwise": pairs, "consensus": consensus}

        rows = []
        for name in names:
            s = stats[name]
            rows.append([
                name,
                f"{100 * s['empty_rate']:.1f}",
                f"[{100 * s['empty_ci'][0]:.1f}, {100 * s['empty_ci'][1]:.1f}]",
                f"{s['mean_tokens_when_present']:.2f}",
                f"{s['mean_phrases']:.2f}",
                f"{100 * s['faithfulness']:.1f}",
                f"{consensus[name]:.3f}",
            ])
        table = md_table(
            ["parser", "declined %", "95% CI", "tokens when present", "phrases",
             "faithful %", "agreement"], rows)

        pair_table = md_table(["pair", "mean Jaccard"],
                              [[k, f"{v:.3f}"] for k, v in pairs.items()])

        print(f"--- {field} ---")
        print(table)
        print()
        print(pair_table)
        print()

        lines += [f"## {field}", "", table, "", "**Pairwise agreement**", "",
                  pair_table, ""]

    # Verdicts, written from the numbers rather than asserted.
    lines += ["## Reading", ""]
    for field in FIELDS:
        stats = results[field]["parsers"]
        worst = max(names, key=lambda n: stats[n]["empty_rate"])
        best = min(names, key=lambda n: stats[n]["empty_rate"])
        ratio = (stats[worst]["empty_rate"] / stats[best]["empty_rate"]
                 if stats[best]["empty_rate"] else float("inf"))
        line = (f"- `{field}`: **{worst}** declines the slot most often "
                f"({100 * stats[worst]['empty_rate']:.1f}%), "
                f"{ratio:.1f}x as often as {best} "
                f"({100 * stats[best]['empty_rate']:.1f}%).")
        lines.append(line)
        print(line)

        consensus = results[field]["consensus"]
        odd = min(names, key=lambda n: consensus[n])
        top = max(names, key=lambda n: consensus[n])
        if len(names) > 2:
            line = (f"- `{field}`: **{odd}** agrees least with the others "
                    f"(mean Jaccard {consensus[odd]:.3f} against {consensus[top]:.3f} "
                    f"for {top}).")
            lines.append(line)
            print(line)

        faith = {n: stats[n]["faithfulness"] for n in names}
        most_faithful = max(names, key=lambda n: faith[n])
        if most_faithful == worst:
            line = (f"- `{field}`: {worst} scores highest on faithfulness "
                    f"({100 * faith[worst]:.1f}%) **while** declining the slot most often. "
                    f"That combination is the signature of a copy-only parser -- it cannot "
                    f"invent a token, so faithfulness is near-tautological for it. Read it "
                    f"beside coverage, never alone.")
            lines.append(line)
            print(line)

    lines += ["",
              "Scope: this measures parse quality, not grounding accuracy. Whether a "
              "worse parse *causes* a worse box is the corruption experiment "
              "(`run_parse_corruption.py` -> `parse_error_propagation.py`), not this.", ""]

    ensure_dir(args.out_dir)
    md_path = os.path.join(args.out_dir, "parse_field_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json({
        "split": args.split,
        "annotations": len(keys),
        "parsers": folders,
        "reference": references,
        "fields": results,
    }, os.path.join(args.out_dir, "parse_field_comparison.json"))

    print(f"\nwrote {md_path}")
    print(f"wrote {json_path}")
    if args.plot:
        png = make_plot(results, names,
                        os.path.join(args.out_dir, "parse_field_comparison.png"))
        if png:
            print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
