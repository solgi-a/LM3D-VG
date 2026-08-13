
import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.ablation.parsers.spacy_parser import NOT_MENTIONED
from experiments.ablation.parsers.run_spacy_parser import load_split
from experiments.ablation.parsers.tokenize_parse import (
    FIELD_MAX_TOKENS, tokenize_record, validate_tokenized)
from experiments.ablation.parsers.run_smalllm_parser import (
    MODELS, MODEL_ALIASES, SPLIT_SIZES, SYSTEM_PROMPT, ONE_SHOT_INPUT, ONE_SHOT_OUTPUT,
    check_requirements, cuda_is_usable, extract_fields, generate, list_models, load_model)

FIELDS = ("target", "adjectives", "neighbors")


def folder_slug(alias_or_id):
    if alias_or_id in MODELS:
        return alias_or_id
    for alias, model_id in MODEL_ALIASES.items():
        if alias_or_id == model_id and alias in MODELS:
            return alias
    slug = str(alias_or_id).lower()
    for bad in "/\\ ":
        slug = slug.replace(bad, "_")
    return "".join(c for c in slug if c.isalnum() or c in "._-")


def folders_for(alias_or_id):
    slug = folder_slug(alias_or_id)
    return (os.path.join("data_parsing", f"smalllm_{slug}_parsing"),
            os.path.join("data_parsing", f"smalllm_{slug}_parsing_tokenized"))


def _partial_path(output_dir, split):
    return os.path.join(output_dir, f".partial_{split}.json")


def _load_partial(path, model_id, split):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            blob = json.load(f)
    except (ValueError, OSError) as error:
        print(f"[{split}] ignoring unreadable checkpoint {path} ({error})")
        return {}

    if not isinstance(blob, dict) or "records" not in blob:
        print(f"[{split}] ignoring checkpoint without a model tag: {path}")
        return {}
    if blob.get("model") != model_id:
        print(f"[{split}] ignoring checkpoint from a different model "
              f"({blob.get('model')!r} != {model_id!r}): {path}")
        return {}

    done = blob.get("records") or {}
    print(f"[{split}] resuming: {len(done)} already parsed (from {path})")
    return done


def _save_partial(path, model_id, done):
    with open(path, "w") as f:
        json.dump({"model": model_id, "records": done}, f)


def run_split(split, args, tokenizer, model, kind, model_id):
    records, source = load_split(split, args.data_root)
    if args.limit:
        records = records[: args.limit]
    print(f"[{split}] {len(records)} descriptions from {source}")

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    partial = _partial_path(args.output_dir, split)
    done = _load_partial(partial, model_id, split) if args.resume else {}

    def key_of(record):
        return f"{record['scene_id']}|{record['object_id']}|{record['ann_id']}"

    todo = [r for r in records if key_of(r) not in done]
    stats = {"records": len(records), "generated": 0,
             "ok": 0, "repaired": 0, "malformed": 0}
    for field in FIELDS:
        stats[f"not_mentioned_{field}"] = 0
        stats[f"capped_{field}"] = 0

    t0 = time.time()
    if todo:
        texts = [(r["description"] or "").strip() for r in todo]
        stream = generate(texts, tokenizer, model, kind, args.device,
                          args.batch_size, args.max_new_tokens)
        for i, (record, raw) in enumerate(zip(todo, stream)):
            parsed, status = extract_fields(raw)
            stats[status] += 1
            stats["generated"] += 1
            done[key_of(record)] = parsed

            if (i + 1) % args.checkpoint_every == 0:
                _save_partial(partial, model_id, done)
                rate = (i + 1) / (time.time() - t0)
                remaining = (len(todo) - i - 1) / max(rate, 1e-9)
                print(f"  {i + 1}/{len(todo)}  ({rate:.1f} desc/s, "
                      f"~{remaining / 60:.0f} min left, "
                      f"malformed {stats['malformed']})")
        _save_partial(partial, model_id, done)

    elapsed = time.time() - t0

    raw_out = defaultdict(lambda: defaultdict(dict))
    tok_out = defaultdict(lambda: defaultdict(dict))
    for record in records:
        parsed = done[key_of(record)]
        tokenized = tokenize_record(parsed)
        for field in FIELDS:
            if parsed[field] == NOT_MENTIONED:
                stats[f"not_mentioned_{field}"] += 1
            if len(tokenized[field]) == FIELD_MAX_TOKENS[field]:
                stats[f"capped_{field}"] += 1
        scene_id, object_id, ann_id = record["scene_id"], record["object_id"], record["ann_id"]
        raw_out[scene_id][object_id][ann_id] = parsed
        tok_out[scene_id][object_id][ann_id] = tokenized

    def plain(nested):
        return {s: {o: dict(a) for o, a in objs.items()} for s, objs in nested.items()}

    raw_path = os.path.join(args.raw_dir, f"parsed_result_{split}.json")
    tok_path = os.path.join(args.output_dir, f"tokenized_parsed_result_{split}.json")
    with open(raw_path, "w") as f:
        json.dump(plain(raw_out), f)
    with open(tok_path, "w") as f:
        json.dump(plain(tok_out), f)

    generated = max(stats["generated"], 1)
    print(f"[{split}] generated {stats['generated']} in {elapsed:.1f}s "
          f"({stats['generated'] / max(elapsed, 1e-9):.2f} desc/s)")
    print(f"[{split}] output status: ok={stats['ok']}  repaired={stats['repaired']}  "
          f"malformed={stats['malformed']} "
          f"({100.0 * stats['malformed'] / generated:.2f}% of generated)")
    print(f"[{split}] -> {raw_path}")
    print(f"[{split}] -> {tok_path} ({os.path.getsize(tok_path) / (1024 ** 2):.1f} MB)")
    print(f"[{split}] not mentioned: " + "  ".join(
        f"{f}={stats[f'not_mentioned_{f}']}" for f in FIELDS))
    print(f"[{split}] hit token cap: " + "  ".join(
        f"{f}={stats[f'capped_{f}']}" for f in FIELDS))

    stats["elapsed_sec"] = round(elapsed, 2)
    stats["desc_per_sec"] = round(stats["generated"] / max(elapsed, 1e-9), 3)
    stats["malformed_pct"] = round(100.0 * stats["malformed"] / generated, 3)
    stats["raw_output"] = raw_path
    stats["tokenized_output"] = tok_path

    with open(tok_path) as f:
        written = json.load(f)
    ok, problems, checked = validate_tokenized(written, label=f"{split}:")
    if ok:
        print(f"[{split}] schema check passed over {checked} entries "
              f"(keys, non-empty, within caps, all strings)")
    else:
        print(f"\n[{split}] SCHEMA CHECK FAILED ({len(problems)} problems):")
        for problem in problems[:20]:
            print("  -", problem)

    if ok and not args.keep_partial and os.path.isfile(partial):
        os.remove(partial)

    return stats, ok


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--model", default="qwen2.5",
                        help="Alias from --list-models, or any HuggingFace model id. "
                             "The output folder is named after it.")
    parser.add_argument("--list-models", dest="list_models", action="store_true",
                        help="Show the supported models and exit. Downloads nothing.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=96)
    parser.add_argument("--raw-dir", dest="raw_dir", default=None,
                        help="Override the model-derived raw folder.")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Override the model-derived tokenized folder.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Parse only the first N descriptions (smoke test).")
    parser.add_argument("--checkpoint-every", dest="checkpoint_every", type=int, default=500)
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore any .partial_{split}.json and start over.")
    parser.add_argument("--keep-partial", dest="keep_partial", action="store_true",
                        help="Do not delete the resume file after a successful split.")
    args = parser.parse_args()

    if args.list_models:
        list_models()
        print("\nEach model writes its own cache:")
        for alias in MODELS:
            print(f"  --model {alias:10s} -> data_parsing/{os.path.basename(folders_for(alias)[1])}/")
        return 0

    if not check_requirements(args.model):
        return 1

    model_id = MODEL_ALIASES.get(args.model, args.model)
    default_raw, default_tok = folders_for(args.model)
    args.raw_dir = args.raw_dir or default_raw
    args.output_dir = args.output_dir or default_tok

    if args.device == "cuda":
        usable, detail = cuda_is_usable()
        if usable:
            print(f"[smalllm] GPU    : {detail}")
        else:
            print(f"[smalllm] ERROR: --device cuda requested but the GPU cannot run a "
                  f"CUDA kernel.\n           {detail}")
            print("           Either install a torch build matching this GPU's compute "
                  "capability,\n           or re-run with --device cpu.")
            return 1

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[smalllm] model  : {model_id}")
    print(f"[smalllm] device : {args.device} ({args.dtype})")
    print(f"[smalllm] raw    : {args.raw_dir}/")
    print(f"[smalllm] cache  : {args.output_dir}/")
    if args.device == "cpu" and not args.limit:
        total = sum(SPLIT_SIZES.get(s, 0) for s in args.splits) or None
        estimate = (f"; roughly {total / 2.0 / 3600:.1f}-{total / 1.0 / 3600:.1f} h for "
                    f"{total} descriptions at 1-2 desc/s" if total else "")
        print(f"[smalllm] WARNING: CPU generation over a full split takes hours{estimate}."
              f"\n           The run is resumable -- it checkpoints every "
              f"{args.checkpoint_every} records and\n           reloads automatically, so "
              f"interrupting it is safe. Use --limit for a smoke test.")

    t0 = time.time()
    tokenizer, model, kind = load_model(model_id, args.device, args.dtype)
    print(f"[smalllm] loaded {kind} model in {time.time() - t0:.1f}s\n")

    summary, ok = {}, True
    for split in args.splits:
        summary[split], split_ok = run_split(split, args, tokenizer, model, kind, model_id)
        ok &= split_ok
        print()

    with open(os.path.join(args.output_dir, "parse_run_summary.json"), "w") as f:
        json.dump({
            "parser": "small local LM (variant E)",
            "model": model_id,
            "model_arg": args.model,
            "kind": kind,
            "device": args.device,
            "dtype": args.dtype,
            "greedy": True,
            "prompt": SYSTEM_PROMPT,
            "one_shot": {"input": ONE_SHOT_INPUT, "output": ONE_SHOT_OUTPUT},
            "splits": summary,
        }, f, indent=2)

    if not ok:
        return 1

    folder = os.path.basename(args.output_dir)
    print("All splits parsed and schema-validated.")
    print("Report the malformed rate above in the manuscript (Reviewer #4 comment 3).")
    print("\nScore this parser against ScanRefer's object_name:")
    print(f"    python experiments/ablation/parsers/eval_parser_target_accuracy.py "
          f"--splits train \\\n        --parsed-dir {args.output_dir} --tag {folder_slug(args.model)}")
    print("\nTo train this variant:")
    print(f"    python scripts/ScanRefer_train.py --parsing_folder {folder} \\")
    print("        --use_cached_scenes --tag ABL-PARSER-SMALLLM --use_color --use_normal")
    return 0


if __name__ == "__main__":
    sys.exit(main())
