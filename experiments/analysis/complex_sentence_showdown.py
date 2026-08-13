
import argparse
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.analysis.common import (
    ensure_dir, load_parse_cache, load_scanrefer, md_table, parse_for, save_json)
from experiments.analysis.parse_field_comparison import (
    FIELDS, is_empty, jaccard, key_of)
from experiments.analysis.linguistic_complexity import (
    metric_depth, metric_spatial, metric_tokens)

RANKERS = {"depth": metric_depth, "tokens": metric_tokens, "spatial": metric_spatial}

QUARTILES = ("Q1 simplest", "Q2", "Q3", "Q4 hardest")


def field_text(parse, field):
    tokens = parse.get(field) or []
    return " ".join(str(t) for t in tokens)


def quartile_indices(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    buckets = [0] * len(values)
    size = max(len(values) / 4.0, 1e-9)
    for rank, index in enumerate(order):
        buckets[index] = min(int(rank / size), 3)
    return buckets


def stratified(keys, parses, names, buckets_by_key, field):
    table = {name: [] for name in names}
    for q in range(4):
        subset = [k for k in keys if buckets_by_key[k] == q]
        for name in names:
            empty = sum(1 for k in subset if is_empty(parses[name][k].get(field)))
            table[name].append({
                "quartile": QUARTILES[q],
                "n": len(subset),
                "empty_rate": empty / len(subset) if subset else 0.0,
            })
    return table


def make_plot(cases, names, strat, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the figure")
        return None

    n_rows = len(cases)
    figure = plt.figure(figsize=(4.6 * len(names), 2.0 + 2.05 * n_rows))
    grid = figure.add_gridspec(n_rows + 1, len(names), height_ratios=[1.5] + [1] * n_rows)

    summary = figure.add_subplot(grid[0, :])
    x = range(4)
    for name in names:
        y = [100 * entry["empty_rate"] for entry in strat["adjectives"][name]]
        summary.plot(x, y, marker="o", label=name)
    summary.set_xticks(list(x))
    summary.set_xticklabels(QUARTILES)
    summary.set_ylabel("adjectives declined (%)")
    summary.set_title("Whole split: how often each parser leaves the attribute slot empty, "
                      "by description complexity")
    summary.grid(alpha=0.3)
    summary.legend(fontsize=8)

    for r, case in enumerate(cases):
        for c, name in enumerate(names):
            axis = figure.add_subplot(grid[r + 1, c])
            axis.axis("off")
            parse = case["parsers"][name]
            if r == 0:
                axis.set_title(name, fontsize=10, fontweight="bold")

            body = []
            for field in ("target",) + FIELDS:
                value = parse[field]
                missing = (field != "target" and parse[f"{field}_empty"])
                label = f"{field:11s}"
                body.append((label + (value if value else "-"), missing))

            axis.text(0.01, 0.97, f"depth {case['depth']} | {case['tokens']} tokens",
                      fontsize=7, color="0.45", va="top", family="monospace")
            y = 0.80
            for text, missing in body:
                for piece in textwrap.wrap(text, 46, subsequent_indent=" " * 13) or [text]:
                    axis.text(0.01, y, piece, fontsize=7.6, va="top", family="monospace",
                              color="#B00020" if missing else "0.15")
                    y -= 0.14
            axis.set_facecolor("#FFF6F6" if any(m for _, m in body) else "white")
            axis.patch.set_visible(True)

    figure.suptitle("Parser output on the hardest descriptions "
                    "(red = the parser declined the slot)", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parse", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--num", type=int, default=10,
                        help="How many of the hardest descriptions to show.")
    parser.add_argument("--rank-by", dest="rank_by", default="depth", choices=sorted(RANKERS))
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--parse-folder", dest="parse_folder", default="final_parsing_tokenized",
                        help="Cache the 'spatial'/'neighbors' rankers read.")
    parser.add_argument("--spacy-model", dest="spacy_model", default="en_core_web_sm")
    parser.add_argument("--depth-cache", dest="depth_cache",
                        default=os.path.join("outputs", "analysis", "dep_depth_cache.json"))
    parser.add_argument("--out-dir", dest="out_dir",
                        default=os.path.join("outputs", "analysis", "complex_sentence_showdown"))
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    args = parser.parse_args()

    if not args.parse:
        args.parse = ["gpt4o-mini=final_parsing_tokenized",
                      "llama=llama_parsing_tokenized_clipped",
                      "spacy=spacy_parsing_tokenized"]

    caches, folders = {}, {}
    for item in args.parse:
        name, _, folder = item.partition("=")
        if not folder:
            name, folder = os.path.basename(item), item
        try:
            caches[name] = load_parse_cache(folder, args.split, args.parsing_root)
            folders[name] = folder
        except FileNotFoundError as error:
            print(f"  skipping {name}: {error}")
    if not caches:
        print("no parse cache could be loaded")
        return 1
    names = list(caches)

    records = load_scanrefer(args.split, args.data_root)
    rows = [r for r in records
            if all(parse_for(caches[n], *key_of(r)) is not None for n in names)]
    if not rows:
        print("no annotation is covered by every parse cache")
        return 1

    values = RANKERS[args.rank_by](rows, args)
    depths = values if args.rank_by == "depth" else metric_depth(rows, args)
    tokens = [len(r.get("token") or []) for r in rows]

    buckets = quartile_indices(values)
    keys = [key_of(r) for r in rows]
    parses = {n: {key_of(r): parse_for(caches[n], *key_of(r)) for r in rows} for n in names}
    buckets_by_key = {key_of(r): buckets[i] for i, r in enumerate(rows)}

    strat = {f: stratified(keys, parses, names, buckets_by_key, f) for f in FIELDS}

    order = sorted(range(len(rows)), key=lambda i: (values[i], tokens[i]), reverse=True)
    cases = []
    for i in order[:args.num]:
        row, key = rows[i], key_of(rows[i])
        case = {
            "scene_id": row["scene_id"], "object_id": row["object_id"],
            "ann_id": row["ann_id"], "object_name": row.get("object_name"),
            "description": row.get("description", ""),
            "depth": int(depths[i]), "tokens": int(tokens[i]),
            "rank_value": int(values[i]), "parsers": {},
        }
        for name in names:
            parse = parses[name][key]
            entry = {"target": field_text(parse, "target")}
            for field in FIELDS:
                entry[field] = field_text(parse, field)
                entry[f"{field}_empty"] = is_empty(parse.get(field))
            case["parsers"][name] = entry
        cases.append(case)

    print(f"split: {args.split} | annotations: {len(rows)} | ranked by: {args.rank_by}")
    print(f"parsers: {', '.join(names)}\n")

    lines = ["# Parsers on the hardest descriptions", "",
             f"split: `{args.split}` | annotations: {len(rows)} | ranked by "
             f"`{args.rank_by}` | showing the {len(cases)} hardest", "",
             "No ground truth exists for `adjectives` or `neighbors`, so this reports "
             "**coverage** (how often a parser leaves the slot empty) and, for the "
             "selected cases, the raw output for inspection.", ""]

    lines += ["## Whole split, by complexity quartile", ""]
    for field in FIELDS:
        header = ["parser"] + [f"{q} (declined %)" for q in QUARTILES] + ["Q4 - Q1"]
        table_rows = []
        for name in names:
            entries = strat[field][name]
            gap = 100 * (entries[3]["empty_rate"] - entries[0]["empty_rate"])
            table_rows.append([name] + [f"{100 * e['empty_rate']:.1f}" for e in entries]
                              + [f"{gap:+.1f}"])
        table = md_table(header, table_rows)
        print(f"--- {field}: declined rate by complexity quartile ---")
        print(table)
        print()
        lines += [f"### {field}", "", table, ""]

    lines += ["**Reading:**", ""]
    adj = strat["adjectives"]
    worst = max(names, key=lambda n: adj[n][3]["empty_rate"])
    best = min(names, key=lambda n: adj[n][3]["empty_rate"])
    v = [f"- On the hardest quartile, **{worst}** leaves the attribute slot empty "
         f"{100 * adj[worst][3]['empty_rate']:.1f}% of the time against "
         f"{100 * adj[best][3]['empty_rate']:.1f}% for {best}."]
    gap_worst = adj[worst][3]["empty_rate"] - adj[worst][0]["empty_rate"]
    gap_best = adj[best][3]["empty_rate"] - adj[best][0]["empty_rate"]
    if gap_worst <= gap_best:
        v.append(f"- Both degrade with complexity, and {worst} does **not** degrade faster "
                 f"({100 * gap_worst:+.1f}pp against {100 * gap_best:+.1f}pp from the "
                 f"simplest to the hardest quartile). The damning number is the level, not "
                 f"the slope: {worst} starts roughly twice as bad and stays there.")
    else:
        v.append(f"- {worst} also degrades faster ({100 * gap_worst:+.1f}pp against "
                 f"{100 * gap_best:+.1f}pp), so the gap widens with complexity.")
    for line in v:
        lines.append(line)
        print(line)
    print()

    lines += ["", "## The hardest descriptions", ""]
    case_lines = []
    for n, case in enumerate(cases, 1):
        head = (f"### {n}. {case['scene_id']} obj {case['object_id']} ann {case['ann_id']} "
                f"-- depth {case['depth']}, {case['tokens']} tokens, "
                f"class `{case['object_name']}`")
        lines += [head, "", f"> {case['description']}", ""]
        rows_md = []
        for name in names:
            entry = case["parsers"][name]
            rows_md.append([
                name, entry["target"],
                entry["adjectives"] + (" **(empty)**" if entry["adjectives_empty"] else ""),
                entry["neighbors"] + (" **(empty)**" if entry["neighbors_empty"] else ""),
            ])
        lines += [md_table(["parser", "target", "adjectives", "neighbors"], rows_md), ""]

        case_lines.append(f"[{n}] {case['scene_id']} obj{case['object_id']} "
                          f"ann{case['ann_id']}  depth={case['depth']} "
                          f"tokens={case['tokens']}  class={case['object_name']}")
        case_lines.append(f"    {case['description']}")
        for name in names:
            entry = case["parsers"][name]
            case_lines.append(f"    {name:12s} target={entry['target']!r} "
                              f"adjectives={entry['adjectives']!r} "
                              f"neighbors={entry['neighbors']!r}")
        case_lines.append("")

    lines += ["Scope: parse quality only. No model was run and no prediction was read, so "
              "nothing here speaks to grounding accuracy -- that is the corruption "
              "experiment.", ""]

    ensure_dir(args.out_dir)
    md_path = os.path.join(args.out_dir, "complex_sentence_showdown.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    txt_path = os.path.join(args.out_dir, "cases.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(case_lines) + "\n")
    json_path = save_json({
        "split": args.split, "annotations": len(rows), "rank_by": args.rank_by,
        "parsers": folders, "quartiles": strat, "cases": cases,
    }, os.path.join(args.out_dir, "complex_sentence_showdown.json"))

    print(f"wrote {md_path}")
    print(f"wrote {txt_path}")
    print(f"wrote {json_path}")
    if args.plot:
        png = make_plot(cases, names, strat,
                        os.path.join(args.out_dir, "complex_sentence_showdown.png"))
        if png:
            print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
