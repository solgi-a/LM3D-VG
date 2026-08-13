
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.ablation.parsers.spacy_parser import NOT_MENTIONED
from experiments.ablation.parsers.run_spacy_parser import load_split
from experiments.ablation.parsers.tokenize_parse import FIELD_MAX_TOKENS, tokenize_record, validate_tokenized

FIELDS = ("target", "adjectives", "neighbors")

SPLIT_SIZES = {"train": 36665, "val": 9508, "test": 0}

SYSTEM_PROMPT = (
    "You parse 3D scene referring expressions. Given one description, return a "
    "JSON object with exactly three string fields: \"target\" (the referred "
    "object), \"adjectives\" (its attributes as they appear in the sentence), "
    "and \"neighbors\" (the spatial-relation phrases describing nearby "
    "objects). Use the literal string \"not mentioned\" for any field the "
    "description does not supply. Return only the JSON object."
)

ONE_SHOT_INPUT = "there is a brown wooden chair. it is next to the table."
ONE_SHOT_OUTPUT = ('{"target": "chair", "adjectives": "brown wooden", '
                   '"neighbors": "next to the table"}')

MODELS = {
    "qwen2.5": {
        "id": "Qwen/Qwen2.5-0.5B-Instruct",
        "params": "0.49B", "download": "~1.0 GB", "kind": "causal",
        "requires": [],
        "notes": "best JSON adherence of the three; lowest malformed rate",
    },
    "qwen2": {
        "id": "Qwen/Qwen2-0.5B-Instruct",
        "params": "0.49B", "download": "~1.0 GB", "kind": "causal",
        "requires": [],
        "notes": "the model named in the revision roadmap",
    },
    "flan-t5": {
        "id": "google/flan-t5-base",
        "params": "0.25B", "download": "~1.0 GB", "kind": "seq2seq",
        "requires": ["sentencepiece"],
        "notes": "fastest on CPU (encoder-decoder); weakest at emitting JSON",
    },
    "smollm2": {
        "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": "0.36B", "download": "~0.7 GB", "kind": "causal",
        "requires": [],
        "notes": "smallest download; a reasonable floor for 'is a tiny LM enough?'",
    },
}

MODEL_ALIASES = {alias: spec["id"] for alias, spec in MODELS.items()}
MODEL_ALIASES["qwen"] = MODELS["qwen2"]["id"]
MODEL_ALIASES["flan"] = MODELS["flan-t5"]["id"]


def list_models():
    print("Small language models available for parser variant E\n")
    header = f"  {'alias':10s} {'model id':38s} {'params':7s} {'download':9s} {'needs':14s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for alias, spec in MODELS.items():
        needs = ", ".join(spec["requires"]) or "-"
        missing = [pkg for pkg in spec["requires"] if not _have(pkg)]
        mark = "  <-- MISSING: " + ", ".join(missing) if missing else ""
        print(f"  {alias:10s} {spec['id']:38s} {spec['params']:7s} "
              f"{spec['download']:9s} {needs:14s}{mark}")
    print()
    for alias, spec in MODELS.items():
        print(f"  {alias:10s} {spec['notes']}")
    print("\nPick one with --model <alias>, or pass any HuggingFace model id directly.")
    print("Nothing is downloaded until you run without --list-models.")


def _have(package):
    import importlib.util

    return importlib.util.find_spec(package) is not None


def check_requirements(alias_or_id):
    spec = MODELS.get(alias_or_id)
    if spec is None:
        lowered = str(alias_or_id).lower()
        needed = ["sentencepiece"] if ("t5" in lowered or "flan" in lowered) else []
    else:
        needed = spec["requires"]

    missing = [pkg for pkg in needed if not _have(pkg)]
    if missing:
        print(f"ERROR: {alias_or_id} needs {', '.join(missing)}, which is not installed.",
              file=sys.stderr)
        print(f"    pip install {' '.join(missing)}", file=sys.stderr)
        print("Nothing was downloaded. Run --list-models to see the alternatives "
              "that need no extra packages.", file=sys.stderr)
        return False
    return True


def dtype_kwargs(torch_dtype):
    try:
        import transformers

        major = int(str(transformers.__version__).split(".")[0])
    except Exception:
        major = 4
    return {"dtype": torch_dtype} if major >= 5 else {"torch_dtype": torch_dtype}


def cuda_is_usable():
    try:
        import torch

        if not torch.cuda.is_available():
            return False, "no CUDA device visible to torch"
        probe = torch.randn(8, 8, device="cuda")
        (probe @ probe).sum().item()
        return True, torch.cuda.get_device_name(0)
    except Exception as error:
        return False, f"{type(error).__name__}: {str(error).splitlines()[0][:120]}"

_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _repair(text):
    return re.sub(r",\s*}", "}", text.replace("'", '"'))


def extract_fields(raw):
    match = _JSON_BLOCK.search(raw or "")
    if not match:
        return {f: NOT_MENTIONED for f in FIELDS}, "malformed"

    block = match.group(0)
    status = "ok"
    try:
        obj = json.loads(block)
    except (ValueError, TypeError):
        try:
            obj = json.loads(_repair(block))
            status = "repaired"
        except (ValueError, TypeError):
            return {f: NOT_MENTIONED for f in FIELDS}, "malformed"

    if not isinstance(obj, dict):
        return {f: NOT_MENTIONED for f in FIELDS}, "malformed"

    parsed, missing = {}, 0
    for field in FIELDS:
        value = obj.get(field)
        if isinstance(value, list):
            value = " ".join(str(v) for v in value if v)
        if not isinstance(value, str) or not value.strip():
            value = NOT_MENTIONED
            missing += 1
        parsed[field] = value.strip()

    if missing == len(FIELDS):
        return parsed, "malformed"
    return parsed, status


def load_model(model_id, device, dtype):
    try:
        import torch
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        print("ERROR: this variant needs `transformers` (and `torch`), which are not "
              "installed in this environment.\n"
              "    pip install 'transformers>=4.40' accelerate\n"
              "Variants A/B/C/D need none of this and are unaffected.", file=sys.stderr)
        raise SystemExit(1)

    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

    config = AutoConfig.from_pretrained(model_id)
    kind = "seq2seq" if getattr(config, "is_encoder_decoder", False) else "causal"

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[dtype]
    if device == "cpu" and torch_dtype is not torch.float32:
        print(f"[smalllm] dtype {dtype} is not usable on CPU; falling back to float32")
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    loader = AutoModelForSeq2SeqLM if kind == "seq2seq" else AutoModelForCausalLM
    model = loader.from_pretrained(model_id, **dtype_kwargs(torch_dtype))
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if kind == "causal":
        tokenizer.padding_side = "left"

    return tokenizer, model, kind


def build_prompts(descriptions, tokenizer, kind):
    prompts = []
    for description in descriptions:
        if kind == "seq2seq":
            prompts.append(
                f"{SYSTEM_PROMPT}\n\nDescription: {ONE_SHOT_INPUT}\nJSON: {ONE_SHOT_OUTPUT}"
                f"\n\nDescription: {description}\nJSON:")
        elif getattr(tokenizer, "chat_template", None):
            prompts.append(tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": ONE_SHOT_INPUT},
                 {"role": "assistant", "content": ONE_SHOT_OUTPUT},
                 {"role": "user", "content": description}],
                tokenize=False, add_generation_prompt=True))
        else:
            prompts.append(
                f"{SYSTEM_PROMPT}\n\nDescription: {ONE_SHOT_INPUT}\nJSON: {ONE_SHOT_OUTPUT}"
                f"\n\nDescription: {description}\nJSON:")
    return prompts


def generate(descriptions, tokenizer, model, kind, device, batch_size, max_new_tokens):
    import torch

    prompts = build_prompts(descriptions, tokenizer, kind)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start: start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        if kind == "causal":
            output = output[:, encoded["input_ids"].shape[1]:]
        for row in output:
            yield tokenizer.decode(row, skip_special_tokens=True)


def _partial_path(output_dir, split):
    return os.path.join(output_dir, f".partial_{split}.json")


def run_split(split, args, tokenizer, model, kind):
    records, source = load_split(split, args.data_root)
    if args.limit:
        records = records[: args.limit]
    print(f"[{split}] {len(records)} descriptions from {source}")

    done = {}
    partial = _partial_path(args.output_dir, split)
    if args.resume and os.path.isfile(partial):
        with open(partial) as f:
            done = json.load(f)
        print(f"[{split}] resuming: {len(done)} already parsed (from {partial})")

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
                with open(partial, "w") as f:
                    json.dump(done, f)
                rate = (i + 1) / (time.time() - t0)
                remaining = (len(todo) - i - 1) / max(rate, 1e-9)
                print(f"  {i + 1}/{len(todo)}  ({rate:.1f} desc/s, "
                      f"~{remaining / 60:.0f} min left, "
                      f"malformed {stats['malformed']})")
        with open(partial, "w") as f:
            json.dump(done, f)

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

    os.makedirs(args.raw_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
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

    ok, problems, checked = validate_tokenized(plain(tok_out))
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
                        help="Alias from --list-models, or any HuggingFace model id.")
    parser.add_argument("--list-models", dest="list_models", action="store_true",
                        help="Show the supported models and exit. Downloads nothing.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=32)
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=96)
    parser.add_argument("--raw-dir", dest="raw_dir",
                        default=os.path.join("data_parsing", "smalllm_parsing"))
    parser.add_argument("--output-dir", dest="output_dir",
                        default=os.path.join("data_parsing", "smalllm_parsing_tokenized"))
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
        return 0

    if not check_requirements(args.model):
        return 1

    model_id = MODEL_ALIASES.get(args.model, args.model)

    device = args.device
    if device == "cuda":
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
    print(f"[smalllm] device : {device} ({args.dtype})")
    if device == "cpu" and not args.limit:
        total = sum(SPLIT_SIZES.get(s, 0) for s in args.splits) or None
        estimate = (f"; roughly {total / 2.0 / 3600:.1f}-{total / 1.0 / 3600:.1f} h for "
                    f"{total} descriptions at 1-2 desc/s" if total else "")
        print(f"[smalllm] WARNING: CPU generation over a full split takes hours{estimate}."
              f"\n           The run is resumable -- it checkpoints every "
              f"{args.checkpoint_every} records and\n           reloads automatically, so "
              f"interrupting it is safe. Use --limit for a smoke test.")

    args.device = device
    t0 = time.time()
    tokenizer, model, kind = load_model(model_id, args.device, args.dtype)
    print(f"[smalllm] loaded {kind} model in {time.time() - t0:.1f}s\n")

    summary, ok = {}, True
    for split in args.splits:
        summary[split], split_ok = run_split(split, args, tokenizer, model, kind)
        ok &= split_ok
        print()

    with open(os.path.join(args.output_dir, "parse_run_summary.json"), "w") as f:
        json.dump({
            "parser": "small local LM (variant E)",
            "model": model_id,
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
    print("All splits parsed and schema-validated.")
    print("Report the malformed rate above in the manuscript (Reviewer #4 comment 3).")
    print("\nTo train this variant:")
    print(f"    python scripts/ScanRefer_train.py --parsing_folder {os.path.basename(args.output_dir)} \\")
    print("        --use_cached_scenes --tag ABL-PARSER-SMALLLM --use_color --use_normal")
    print("or:  python experiments/ablation/runners/run_parser_smalllm.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
