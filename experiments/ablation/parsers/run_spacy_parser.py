"""
Run the rule-based spaCy parser over ScanRefer and write the files lib/dataset.py reads.

    python experiments/ablation/parsers/run_spacy_parser.py --splits train val

Writes, mirroring the GPT-4o-mini layout exactly:

    data_parsing/spacy_parsing/parsed_result_{split}.json            (raw phrases)
    data_parsing/spacy_parsing_tokenized/tokenized_parsed_result_{split}.json

To run the ablation, set in experiments/ablation/ablation_config.py

    ABLATION.PARSING_FOLDER = "spacy_parsing_tokenized"

or pass ``--parsing_folder spacy_parsing_tokenized``. Nothing else changes -- the language
module and fusion network are untouched.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Resolve the repo root from this file, not the cwd, so the script works when
# invoked as `python experiments/ablation/parsers/<name>.py` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.ablation.parsers.spacy_parser import NOT_MENTIONED, load_parser, parse_doc
from experiments.ablation.parsers.tokenize_parse import FIELD_MAX_TOKENS, tokenize_record, validate_tokenized

FIELDS = ("target", "adjectives", "neighbors")


def load_split(split, data_root):
    """Load ScanRefer, tolerating the repo-root copy of the val file."""
    candidates = [
        os.path.join(data_root, f"ScanRefer_filtered_{split}.json"),
        os.path.join(os.getcwd(), f"ScanRefer_filtered_{split}.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f), path
    raise FileNotFoundError(
        f"ScanRefer_filtered_{split}.json not found. Looked in:\n  " + "\n  ".join(candidates)
    )


def _nested():
    return defaultdict(lambda: defaultdict(dict))


def _plain(nested):
    return {s: {o: dict(a) for o, a in objs.items()} for s, objs in nested.items()}


def run_split(split, args):
    records, source = load_split(split, args.data_root)
    if args.limit:
        records = records[: args.limit]
    print(f"[{split}] {len(records)} descriptions from {source}")

    nlp = load_parser(args.model)

    raw_out, tok_out = _nested(), _nested()
    stats = {"records": 0}
    for field in FIELDS:
        stats[f"not_mentioned_{field}"] = 0
        stats[f"capped_{field}"] = 0

    t0 = time.time()
    # nlp.pipe is markedly faster than calling nlp() per description.
    texts = [r["description"].lower() if r["description"] else "" for r in records]
    docs = nlp.pipe(texts, batch_size=256)

    for i, (record, doc) in enumerate(zip(records, docs)):
        parsed = parse_doc(doc)
        tokenized = tokenize_record(parsed)

        for field in FIELDS:
            if parsed[field] == NOT_MENTIONED:
                stats[f"not_mentioned_{field}"] += 1
            if len(tokenized[field]) == FIELD_MAX_TOKENS[field]:
                stats[f"capped_{field}"] += 1

        scene_id, object_id, ann_id = record["scene_id"], record["object_id"], record["ann_id"]
        raw_out[scene_id][object_id][ann_id] = parsed
        tok_out[scene_id][object_id][ann_id] = tokenized
        stats["records"] += 1

        if (i + 1) % 5000 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(records)}  ({rate:.0f} desc/s)")

    elapsed = time.time() - t0

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.raw_dir, f"parsed_result_{split}.json")
    tok_path = os.path.join(args.output_dir, f"tokenized_parsed_result_{split}.json")
    with open(raw_path, "w") as f:
        json.dump(_plain(raw_out), f)
    with open(tok_path, "w") as f:
        json.dump(_plain(tok_out), f)

    size_mb = os.path.getsize(tok_path) / (1024 ** 2)
    print(f"[{split}] parsed {stats['records']} in {elapsed:.1f}s "
          f"({stats['records'] / max(elapsed, 1e-9):.0f} desc/s)")
    print(f"[{split}] -> {raw_path}")
    print(f"[{split}] -> {tok_path} ({size_mb:.1f} MB)")
    print(f"[{split}] not mentioned: " + "  ".join(
        f"{f}={stats[f'not_mentioned_{f}']}" for f in FIELDS))
    print(f"[{split}] hit token cap: " + "  ".join(
        f"{f}={stats[f'capped_{f}']}" for f in FIELDS))

    stats["elapsed_sec"] = round(elapsed, 2)
    stats["desc_per_sec"] = round(stats["records"] / max(elapsed, 1e-9), 1)
    stats["raw_output"] = raw_path
    stats["tokenized_output"] = tok_path

    with open(tok_path) as f:
        ok, problems, checked = validate_tokenized(json.load(f))
    if ok:
        print(f"[{split}] schema check passed over {checked} entries "
              f"(keys, non-empty, within caps, all strings)")
    else:
        print(f"\n[{split}] SCHEMA CHECK FAILED ({len(problems)} problems):")
        for problem in problems[:20]:
            print("  -", problem)

    return stats, ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--raw-dir", dest="raw_dir",
                        default=os.path.join("data_parsing", "spacy_parsing"))
    parser.add_argument("--output-dir", dest="output_dir",
                        default=os.path.join("data_parsing", "spacy_parsing_tokenized"))
    parser.add_argument("--model", default="en_core_web_sm",
                        help="spaCy model: en_core_web_sm (fast) or en_core_web_trf "
                             "(more accurate, ~30-50x slower)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Parse only the first N descriptions (smoke test).")
    args = parser.parse_args()

    summary, ok = {}, True
    for split in args.splits:
        summary[split], split_ok = run_split(split, args)
        ok &= split_ok
        print()

    with open(os.path.join(args.output_dir, "parse_run_summary.json"), "w") as f:
        json.dump({"parser": "spacy", "model": args.model, "splits": summary}, f, indent=2)

    if not ok:
        return 1
    print("All splits parsed and schema-validated.")
    print("To use this parse in training, set in experiments/ablation/ablation_config.py:")
    print(f"    ABLATION.PARSING_FOLDER = {os.path.basename(args.output_dir)!r}")
    print("(the folder is resolved under data/scannet/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
