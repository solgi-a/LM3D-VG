
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.ablation.parsers.tokenize_parse import FIELD_MAX_TOKENS, clip_tokens, validate_tokenized


def clip_split(in_dir, out_dir, split):
    in_path = os.path.join(in_dir, f"tokenized_parsed_result_{split}.json")
    if not os.path.isfile(in_path):
        print(f"[{split}] skipped, no such file: {in_path}")
        return None

    with open(in_path) as f:
        data = json.load(f)

    clipped = {field: 0 for field in FIELD_MAX_TOKENS}
    filled = {field: 0 for field in FIELD_MAX_TOKENS}
    total = 0
    out = {}

    for scene_id, objects in data.items():
        out[scene_id] = {}
        for object_id, anns in objects.items():
            out[scene_id][object_id] = {}
            for ann_id, parsed in anns.items():
                total += 1
                record = {}
                for field in FIELD_MAX_TOKENS:
                    tokens = parsed.get(field) or []
                    if len(tokens) > FIELD_MAX_TOKENS[field]:
                        clipped[field] += 1
                    elif len(tokens) == 0:
                        filled[field] += 1
                    record[field] = clip_tokens(tokens, field)
                out[scene_id][object_id][ann_id] = record

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"tokenized_parsed_result_{split}.json")
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"[{split}] {total} annotations -> {out_path}")
    print(f"[{split}] clipped: " + "  ".join(f"{f}={clipped[f]}" for f in FIELD_MAX_TOKENS))
    if any(filled.values()):
        print(f"[{split}] empty fields filled with 'not mentioned': "
              + "  ".join(f"{f}={filled[f]}" for f in FIELD_MAX_TOKENS))

    ok, problems, checked = validate_tokenized(out, label=f"{split}:")
    if ok:
        print(f"[{split}] schema check passed over {checked} entries")
    else:
        print(f"[{split}] SCHEMA CHECK FAILED ({len(problems)} problems):")
        for problem in problems[:20]:
            print("  -", problem)
    return ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True,
                        help="Folder holding tokenized_parsed_result_{split}.json")
    parser.add_argument("--output", required=True,
                        help="Destination folder (must differ from --input)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()

    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("--output must differ from --input; this script never edits in place")

    ok = True
    for split in args.splits:
        result = clip_split(args.input, args.output, split)
        if result is not None:
            ok &= result
        print()

    if not ok:
        return 1
    print("Done. To use this cache set in experiments/ablation/ablation_config.py:")
    print(f"    ABLATION.PARSING_FOLDER = {os.path.basename(args.output.rstrip('/'))!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
