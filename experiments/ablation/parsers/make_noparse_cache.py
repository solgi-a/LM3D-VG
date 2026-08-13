"""
Parser ablation, variant D -- no parser at all.

    RUNS ON: CPU. Seconds. No model, no GPU, no network.

    python experiments/ablation/parsers/make_noparse_cache.py --splits train val

Writes ``data_parsing/noparse_tokenized/tokenized_parsed_result_{split}.json``, in which
every annotation carries the same content-free placeholder in all three fields. The
architecture is unchanged -- A2F and TAF still run -- so the difference against variants
A/B/C/E is attributable to the parser alone.

Why the placeholder is not zeros
--------------------------------
Two facts in the existing code decide the encoding:

1. ``__getitem__`` sets ``tgt_len = len(tokens)`` unclipped (lib/dataset.py:173-175) and
   the language module feeds that to ``pack_padded_sequence``, which raises on length 0.
   An empty list is therefore impossible.
2. ``_transform_parsed`` (lib/dataset.py:570-581) fills row *i* with ``glove[token]``, or
   ``glove["unk"]`` when out of vocabulary. No token maps to a zero row -- ``glove["pad"]``
   has sum|.| = 89.3, an ordinary trained vector. A zero-masked field is unreachable
   without editing lib/dataset.py.

So the field carries one token whose GloVe row is the model's own "no information" vector:

    ``--encoding unk``            -> ["unk"]              (default)
    ``--encoding not_mentioned``  -> ["not", "mentioned"]

``unk`` is the default because the codebase already writes exactly that for "no parse" when
``args.detection == True`` (lib/dataset.py:687-689), so variant D reaches that state
through the normal parse-cache path with no code changes. ``not_mentioned`` is the GPT
pipeline's convention for an unmentioned field and exists as a robustness check.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

# Resolve the repo root from this file, not the cwd, so the script works when
# invoked as `python experiments/ablation/parsers/<name>.py` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.ablation.parsers.run_spacy_parser import load_split
from experiments.ablation.parsers.tokenize_parse import (
    FIELD_MAX_TOKENS,
    NOT_MENTIONED_TOKENS,
    validate_tokenized,
)

FIELDS = ("target", "adjectives", "neighbors")

#: Placeholder token list per --encoding. Both are length >= 1 by construction, which is
#: the constraint pack_padded_sequence imposes.
ENCODINGS = {
    "unk": ["unk"],
    "not_mentioned": list(NOT_MENTIONED_TOKENS),
}

#: Where each encoding is written by default, so both can coexist on disk.
DEFAULT_DIRS = {
    "unk": os.path.join("data_parsing", "noparse_tokenized"),
    "not_mentioned": os.path.join("data_parsing", "noparse_notmentioned_tokenized"),
}


def build_split(split, placeholder, data_root, limit=None):
    """Return the nested {scene: {object: {ann: parsed}}} dict for one split."""
    records, source = load_split(split, data_root)
    if limit:
        records = records[:limit]

    out = defaultdict(lambda: defaultdict(dict))
    for record in records:
        # A fresh list per field per record: json.dump would happily share one object,
        # but a shared mutable would be a trap for anything that post-processes this.
        out[record["scene_id"]][record["object_id"]][record["ann_id"]] = {
            field: list(placeholder) for field in FIELDS
        }
    return {s: {o: dict(a) for o, a in objs.items()} for s, objs in out.items()}, \
        len(records), source


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--encoding", default="unk", choices=sorted(ENCODINGS),
                        help="Placeholder for every field. Default 'unk' matches the "
                             "no-parse encoding already in lib/dataset.py:687-689.")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Defaults to data_parsing/noparse_tokenized for --encoding "
                             "unk, data_parsing/noparse_notmentioned_tokenized otherwise.")
    parser.add_argument("--both", action="store_true",
                        help="Write both encodings, each to its own default directory.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N descriptions per split (smoke test).")
    args = parser.parse_args()

    encodings = sorted(ENCODINGS) if args.both else [args.encoding]
    if args.both and args.output_dir:
        parser.error("--both writes two directories; --output-dir cannot be used with it.")

    ok = True
    for encoding in encodings:
        placeholder = ENCODINGS[encoding]
        output_dir = args.output_dir or DEFAULT_DIRS[encoding]
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n=== variant D, encoding={encoding!r} -> {placeholder} ===")
        summary = {}
        for split in args.splits:
            data, n, source = build_split(split, placeholder, args.data_root, args.limit)
            path = os.path.join(output_dir, f"tokenized_parsed_result_{split}.json")
            with open(path, "w") as f:
                json.dump(data, f)

            size_mb = os.path.getsize(path) / (1024 ** 2)
            print(f"[{split}] {n} descriptions from {os.path.basename(source)}")
            print(f"[{split}] -> {path} ({size_mb:.1f} MB)")

            with open(path) as f:
                passed, problems, checked = validate_tokenized(json.load(f))
            if passed:
                print(f"[{split}] schema check passed over {checked} entries "
                      f"(keys, non-empty, within caps {dict(FIELD_MAX_TOKENS)}, all strings)")
            else:
                ok = False
                print(f"[{split}] SCHEMA CHECK FAILED ({len(problems)} problems):")
                for problem in problems[:20]:
                    print("  -", problem)

            summary[split] = {"records": n, "checked": checked, "schema_ok": passed,
                              "output": path}

        meta_path = os.path.join(output_dir, "parse_run_summary.json")
        with open(meta_path, "w") as f:
            json.dump({
                "parser": "none (variant D)",
                "encoding": encoding,
                "placeholder_tokens": placeholder,
                "rationale": "pack_padded_sequence rejects length 0 and no GloVe token "
                             "maps to a zero row, so the minimal content-free field is a "
                             "single placeholder token.",
                "splits": summary,
            }, f, indent=2)
        print(f"[meta] -> {meta_path}")

        print("\nTo train this variant:")
        print(f"    python scripts/ScanRefer_train.py --parsing_folder {os.path.basename(output_dir)} \\")
        print("        --use_cached_scenes --tag ABL-PARSER-NONE --use_color --use_normal")
        print("or:  python experiments/ablation/runners/run_parser_none.py")

    if not ok:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
