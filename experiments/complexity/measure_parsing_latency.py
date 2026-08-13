
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lib.config import CONF  # noqa: E402

GPT_SYSTEM_PROMPT = (
    "You parse 3D scene referring expressions. Given one description, return a "
    "JSON object with exactly three string fields: \"target\" (the referred "
    "object), \"adjectives\" (its attributes as they appear in the sentence), "
    "and \"neighbors\" (the spatial-relation phrases describing nearby "
    "objects). Use the literal string \"not mentioned\" for any field the "
    "description does not supply. Return only the JSON object."
)

DEFAULT_PRICE_IN = 0.150
DEFAULT_PRICE_OUT = 0.600


def load_val_descriptions(n, seed=42):
    path = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")
    with open(path) as f:
        records = json.load(f)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
    out = [records[int(i)]["description"] for i in idx]
    lens = [len(d.split()) for d in out]
    return out, {"count": len(out), "mean_words": round(float(np.mean(lens)), 1),
                 "min_words": int(min(lens)), "max_words": int(max(lens))}


def stats_ms(times_s):
    t = np.asarray(times_s) * 1000.0
    return {
        "n": int(t.size),
        "mean_ms": round(float(t.mean()), 3),
        "std_ms": round(float(t.std()), 3),
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "min_ms": round(float(t.min()), 3),
        "max_ms": round(float(t.max()), 3),
    }


def offline_check():
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    findings = {}

    ds = open(os.path.join(repo, "lib", "dataset.py"), encoding="utf-8").read()
    findings["dataset_loads_precomputed_json"] = "tokenized_parsed_result_" in ds
    findings["parse_cache_read_at_construction"] = "_tranform_des" in ds

    parser_names = ("openai", "OpenAI", "ChatCompletion", "gpt-4o", "spacy.load")
    hits = []
    for rel in ("lib/dataset.py", "models/lang_module.py", "models/match_module.py",
                "models/refnet.py", "experiments/ablation/cached_refnet.py", "lib/solver.py"):
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            continue
        src = open(p, encoding="utf-8").read()
        for name in parser_names:
            if name in src:
                hits.append(f"{rel}: {name}")
    findings["parser_invocations_in_forward_path"] = hits or "none"

    caches = []
    pdir = os.path.join(repo, "data_parsing")
    if os.path.isdir(pdir):
        caches = sorted(d for d in os.listdir(pdir)
                        if os.path.isdir(os.path.join(pdir, d)))
    findings["available_parse_caches"] = caches

    findings["verdict"] = (
        "OFFLINE for all reported benchmarks. lib/dataset.py loads a precomputed "
        "tokenized_parsed_result_{split}.json once at dataset construction and "
        "indexes it by (scene_id, object_id, ann_id). No parser -- neither an LLM "
        "API call nor spaCy -- is invoked anywhere in the model forward path. "
        "Parsing therefore contributes ZERO to the per-query latency of every "
        "number reported on ScanRefer/ReferIt3D."
    ) if not hits else (
        "REVIEW NEEDED: a parser reference was found in the forward path -- see "
        "parser_invocations_in_forward_path."
    )
    findings["deployment_note"] = (
        "For a genuinely unseen sentence at deployment time, one parse call is "
        "required before grounding. Its cost is the 'gpt' or 'spacy' measurement "
        "in this report and is additive ONLY in that scenario. The two must be "
        "reported separately, never summed into a single latency figure."
    )
    return findings


def measure_spacy(descriptions, model="en_core_web_sm", warmup=5):
    from experiments.ablation.parsers.spacy_parser import load_parser, parse_with_spacy

    t0 = time.perf_counter()
    nlp = load_parser(model)
    load_s = time.perf_counter() - t0

    for d in descriptions[:warmup]:
        parse_with_spacy(d, nlp=nlp)

    times, samples = [], []
    for i, d in enumerate(descriptions):
        t0 = time.perf_counter()
        parsed = parse_with_spacy(d, nlp=nlp)
        times.append(time.perf_counter() - t0)
        if i < 3:
            samples.append({"description": d, "parsed": parsed})

    out = stats_ms(times)
    out.update({
        "model": model,
        "model_load_s": round(load_s, 2),
        "throughput_desc_per_s": round(1.0 / (float(np.mean(times)) or 1e-9), 1),
        "cost_usd_per_query": 0.0,
        "note": "Local CPU inference, per description, one at a time (the "
                "deployment scenario). Batched nlp.pipe() is faster and is what "
                "experiments/ablation/parsers/run_spacy_parser.py uses for bulk parsing.",
        "samples": samples,
    })
    return out


def measure_gpt(descriptions, model, price_in, price_out, warmup=2):
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed -- pip install openai"}
    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set in the environment"}

    client = OpenAI()

    def one(desc):
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": GPT_SYSTEM_PROMPT},
                      {"role": "user", "content": desc}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        dt = time.perf_counter() - t0
        return dt, r

    for d in descriptions[:warmup]:
        try:
            one(d)
        except Exception:
            pass

    times, tok_in, tok_out, errors, samples = [], [], [], 0, []
    for i, d in enumerate(descriptions):
        try:
            dt, r = one(d)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    [api error] {type(exc).__name__}: {exc}")
            continue
        times.append(dt)
        u = getattr(r, "usage", None)
        if u is not None:
            tok_in.append(u.prompt_tokens)
            tok_out.append(u.completion_tokens)
        if i < 3:
            samples.append({"description": d,
                            "response": r.choices[0].message.content})
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(descriptions)} calls, "
                  f"running mean {np.mean(times) * 1000:.0f} ms")

    if not times:
        return {"error": f"all {len(descriptions)} calls failed"}

    out = stats_ms(times)
    mean_in = float(np.mean(tok_in)) if tok_in else None
    mean_out = float(np.mean(tok_out)) if tok_out else None
    cost = None
    if mean_in is not None and mean_out is not None:
        cost = mean_in / 1e6 * price_in + mean_out / 1e6 * price_out
    out.update({
        "model": model,
        "errors": errors,
        "mean_prompt_tokens": round(mean_in, 1) if mean_in else None,
        "mean_completion_tokens": round(mean_out, 1) if mean_out else None,
        "price_usd_per_1m_input": price_in,
        "price_usd_per_1m_output": price_out,
        "cost_usd_per_query": round(cost, 8) if cost else None,
        "cost_usd_per_1k_queries": round(cost * 1000, 5) if cost else None,
        "cost_usd_full_scanrefer_46173": round(cost * 46173, 3) if cost else None,
        "note": "Real API calls over distinct val descriptions. Latency includes "
                "network round-trip and therefore depends on location and load. "
                "The prompt is RECONSTRUCTED from the observed output schema -- "
                "the original parsing script is not in this repository.",
        "samples": samples,
    })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["offline_check", "spacy", "gpt", "all"],
                   default="offline_check")
    p.add_argument("--num_samples", type=int, default=100,
                   help="Distinct val descriptions to parse (>=100 recommended).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--spacy_model", default="en_core_web_sm")
    p.add_argument("--gpt_model", default="gpt-4o-mini")
    p.add_argument("--price_in", type=float, default=DEFAULT_PRICE_IN,
                   help="USD per 1M input tokens.")
    p.add_argument("--price_out", type=float, default=DEFAULT_PRICE_OUT,
                   help="USD per 1M output tokens.")
    p.add_argument("--output", default=os.path.join("outputs", "complexity"))
    args = p.parse_args()

    print("=" * 78)
    print("  PARSING LATENCY & COST  (Reviewer #4 comment 5, Reviewer #2)")
    print("=" * 78)

    report = {"config": vars(args)}

    dep = offline_check()
    report["deployment"] = dep
    print("\n-- deployment analysis --")
    for k, v in dep.items():
        if k in ("verdict", "deployment_note"):
            continue
        print(f"  {k:<38s} {v}")
    print(f"\n  VERDICT: {dep['verdict']}\n")
    print(f"  {dep['deployment_note']}\n")

    need_desc = args.mode in ("spacy", "gpt", "all")
    if need_desc:
        descs, dstats = load_val_descriptions(args.num_samples, args.seed)
        report["description_sample"] = dstats
        print(f"-- sampled {dstats['count']} distinct val descriptions "
              f"(mean {dstats['mean_words']} words, "
              f"range {dstats['min_words']}-{dstats['max_words']}) --")

    if args.mode in ("spacy", "all"):
        print("\n-- spaCy (variant B), local --")
        r = measure_spacy(descs, args.spacy_model)
        report["spacy"] = r
        print(f"  mean {r['mean_ms']:.2f} ms   std {r['std_ms']:.2f}   "
              f"p50 {r['p50_ms']:.2f}   p95 {r['p95_ms']:.2f}   "
              f"({r['throughput_desc_per_s']:.0f} desc/s, $0 per query)")

    if args.mode in ("gpt", "all"):
        print(f"\n-- {args.gpt_model} (variant A), REAL API calls --")
        r = measure_gpt(descs, args.gpt_model, args.price_in, args.price_out)
        report["gpt"] = r
        if "error" in r:
            print(f"  SKIPPED: {r['error']}")
        else:
            print(f"  mean {r['mean_ms']:.1f} ms   std {r['std_ms']:.1f}   "
                  f"p50 {r['p50_ms']:.1f}   p95 {r['p95_ms']:.1f}   "
                  f"(errors: {r['errors']})")
            print(f"  tokens: {r['mean_prompt_tokens']} in / "
                  f"{r['mean_completion_tokens']} out per query")
            if r["cost_usd_per_query"]:
                print(f"  cost:   ${r['cost_usd_per_query']:.6f}/query   "
                      f"${r['cost_usd_per_1k_queries']:.4f}/1k   "
                      f"${r['cost_usd_full_scanrefer_46173']:.2f} for all 46,173")

    report["reporting_guidance"] = [
        "Report grounding-network latency and parsing latency as SEPARATE rows. "
        "Summing them describes a scenario the paper never benchmarks.",
        "For every ScanRefer/ReferIt3D number in the paper, parsing is offline "
        "and precomputed -- state this explicitly; it is the direct answer to "
        "Reviewer #4 comment 5.",
        "The online/deployment scenario (one unseen sentence) is the only case "
        "where parse latency is additive. Quote it as such.",
        "spaCy (variant B) removes the API dependency and its cost entirely, at "
        "a measured target-extraction accuracy penalty -- cross-reference the "
        "parser ablation table.",
    ]

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "parsing_latency_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
