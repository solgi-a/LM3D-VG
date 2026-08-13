
import argparse
import csv
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.ablation.parsers.eval_parser_target_accuracy import match_kinds
from experiments.analysis.common import ensure_dir, load_parse_cache, load_scanrefer, parse_for, save_json

BLANK_COLUMNS = ("target_ok", "adjectives_ok", "neighbors_ok", "error_type", "notes")

FIXED_COLUMNS = ("sample_id", "parser", "scene_id", "object_id", "ann_id",
                 "object_name", "description",
                 "parsed_target", "parsed_adjectives", "parsed_neighbors",
                 "auto_target_match")

TAXONOMY = {
    "ok": "all three fields are acceptable",
    "wrong_target": "the target names the wrong object",
    "missed_attribute": "an attribute stated in the description is absent from the parse",
    "hallucinated_attribute": "an attribute appears that the description does not state",
    "missed_neighbor": "an adjacent object stated in the description is absent",
    "hallucinated_neighbor": "an adjacent object appears that the description does not state",
    "malformed_output": "the parse is structurally broken or empty where content exists",
}

INSTRUCTIONS = """# Manual parse annotation — instructions

Fill in the five blank columns of each `annotation_sheet_<parser>.csv`.

## Columns to fill

| column | values | meaning |
|---|---|---|
| `target_ok` | `1` / `0` | Does `parsed_target` name the object the description refers to? Judge against the description, not only against `object_name` — `object_name` is the annotated class and is sometimes coarser than the sentence. |
| `adjectives_ok` | `1` / `0` | Does `parsed_adjectives` capture the attributes the description gives for the target, without inventing any? `not mentioned` is **correct** when the description gives no attributes. |
| `neighbors_ok` | `1` / `0` | Does `parsed_neighbors` capture the spatial relations to other objects, without inventing any? `not mentioned` is **correct** when there are none. |
| `error_type` | see below | One or more codes, separated by `;`. Use `ok` when nothing is wrong. |
| `notes` | free text | Anything worth quoting in the paper. Optional. |

## Error codes

{taxonomy}

## Rules that keep the numbers honest

1. **Annotate every parser's sheet for the same `sample_id` before moving on.** The design
   is paired; breaking the pairing throws away most of its statistical power.
2. **Do not consult `auto_target_match`** when deciding `target_ok`. It is the automatic
   verdict, included only so disagreements can be found afterwards. Judging while looking
   at it makes the two measurements dependent and the comparison meaningless.
3. `not mentioned` is a legitimate, correct parse whenever the description genuinely does
   not supply that field. It is only an error when content exists and was dropped.
4. If two annotators are available, have both label the same sheet and save them
   separately. `experiments/analysis/error_taxonomy.py` computes Cohen's kappa when given two files,
   and an agreement number makes the manual layer far harder to dismiss.
5. The sample is **stratified, not uniform** — failures are deliberately oversampled. Do
   not quote raw percentages from it as population rates; `error_taxonomy.py` applies the
   weights in `sampling_manifest.json` and reports both.
"""


def build_pool(records, cache, criterion):
    correct, wrong = [], []
    for record in records:
        parsed = parse_for(cache, record["scene_id"], record["object_id"], record["ann_id"])
        if not parsed:
            continue
        predicted = " ".join(parsed.get("target") or [])
        kinds = match_kinds(predicted, record["object_name"])
        (correct if kinds[criterion] else wrong).append(record)
    return correct, wrong


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parse", action="append", required=True, metavar="NAME=FOLDER")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--wrong-fraction", dest="wrong_fraction", type=float, default=0.4,
                        help="Share of the sample drawn from automatically-detected "
                             "target errors. 0 gives a uniform random sample.")
    parser.add_argument("--criterion", default="fuzzy",
                        choices=["exact", "substring", "fuzzy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/analysis/annotation")
    args = parser.parse_args()

    parsers = []
    for spec in args.parse:
        name, folder = (spec.split("=", 1) if "=" in spec else (spec, spec))
        parsers.append((name.strip(), folder.strip()))

    records = load_scanrefer(args.split, args.data_root)
    caches = {}
    for name, folder in parsers:
        caches[name] = load_parse_cache(folder, args.split, args.parsing_root)

    anchor_name, anchor_folder = parsers[0]
    correct, wrong = build_pool(records, caches[anchor_name], args.criterion)
    print(f"[strata] anchored on {anchor_name} ({anchor_folder}): "
          f"{len(correct)} target-correct, {len(wrong)} target-wrong")

    rng = random.Random(args.seed)
    num_wrong = min(len(wrong), int(round(args.num * args.wrong_fraction)))
    num_correct = min(len(correct), args.num - num_wrong)
    sample = (rng.sample(wrong, num_wrong) + rng.sample(correct, num_correct))
    rng.shuffle(sample)
    print(f"[sample] {len(sample)} annotations: {num_wrong} from the target-wrong "
          f"stratum, {num_correct} from the target-correct stratum")

    ensure_dir(args.out_dir)
    written = []
    for name, folder in parsers:
        cache = caches[name]
        rows = []
        for index, record in enumerate(sample, 1):
            parsed = parse_for(cache, record["scene_id"], record["object_id"],
                               record["ann_id"]) or {}
            predicted = " ".join(parsed.get("target") or [])
            kinds = match_kinds(predicted, record["object_name"]) if parsed else None
            rows.append({
                "sample_id": f"S{index:03d}",
                "parser": name,
                "scene_id": record["scene_id"],
                "object_id": record["object_id"],
                "ann_id": record["ann_id"],
                "object_name": record["object_name"],
                "description": (record.get("description") or "").strip(),
                "parsed_target": predicted or "(absent from cache)",
                "parsed_adjectives": " ".join(parsed.get("adjectives") or []) or "(absent)",
                "parsed_neighbors": " ".join(parsed.get("neighbors") or []) or "(absent)",
                "auto_target_match": (args.criterion if kinds and kinds[args.criterion]
                                      else "MISS"),
                **{column: "" for column in BLANK_COLUMNS},
            })

        csv_path = os.path.join(args.out_dir, f"annotation_sheet_{name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(FIXED_COLUMNS) + list(BLANK_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

        txt_path = os.path.join(args.out_dir, f"annotation_sheet_{name}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Manual annotation sheet - parser: {name} ({folder})\n")
            f.write(f"split={args.split}  n={len(rows)}  seed={args.seed}\n")
            f.write("Fill target_ok / adjectives_ok / neighbors_ok / error_type in the CSV.\n")
            f.write("=" * 92 + "\n\n")
            for row in rows:
                f.write(f"[{row['sample_id']}] {row['scene_id']} "
                        f"obj={row['object_id']} ann={row['ann_id']}\n")
                f.write(f"  description : {row['description']}\n")
                f.write(f"  object_name : {row['object_name']}\n")
                f.write(f"  target      : {row['parsed_target']}\n")
                f.write(f"  adjectives  : {row['parsed_adjectives']}\n")
                f.write(f"  neighbors   : {row['parsed_neighbors']}\n")
                f.write("  target_ok=__  adjectives_ok=__  neighbors_ok=__  error_type=______\n\n")

        written += [csv_path, txt_path]
        print(f"[{name}] -> {csv_path}")
        print(f"[{name}] -> {txt_path}")

    instructions_path = os.path.join(args.out_dir, "instructions.md")
    taxonomy_table = "\n".join(f"- `{code}` — {text}" for code, text in TAXONOMY.items())
    with open(instructions_path, "w") as f:
        f.write(INSTRUCTIONS.format(taxonomy=taxonomy_table))

    manifest_path = save_json({
        "split": args.split, "seed": args.seed, "criterion": args.criterion,
        "anchor_parser": anchor_name,
        "requested": args.num, "sampled": len(sample),
        "strata": {
            "target_wrong": {
                "population": len(wrong), "sampled": num_wrong,
                "weight": (len(wrong) / num_wrong) if num_wrong else 0.0},
            "target_correct": {
                "population": len(correct), "sampled": num_correct,
                "weight": (len(correct) / num_correct) if num_correct else 0.0},
        },
        "note": "Failures are oversampled. Re-weight with the per-stratum weights before "
                "quoting any rate as a population rate; error_taxonomy.py does this.",
        "parsers": {name: folder for name, folder in parsers},
        "sample": [{"sample_id": f"S{i:03d}", "scene_id": r["scene_id"],
                    "object_id": r["object_id"], "ann_id": r["ann_id"]}
                   for i, r in enumerate(sample, 1)],
    }, os.path.join(args.out_dir, "sampling_manifest.json"))

    print(f"\nwrote {instructions_path}")
    print(f"wrote {manifest_path}")
    print("\nAnnotate the CSV files, then run:")
    print("    python experiments/analysis/error_taxonomy.py " +
          " ".join(f"--sheet {name}={os.path.join(args.out_dir, f'annotation_sheet_{name}.csv')}"
                   for name, _ in parsers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
