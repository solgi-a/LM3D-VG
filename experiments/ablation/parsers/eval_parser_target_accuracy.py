
import argparse
import json
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.ablation.parsers.run_spacy_parser import load_split
from experiments.ablation.parsers.tokenize_parse import NOT_MENTIONED_TOKENS


def normalise(text):
    return " ".join(str(text).replace("_", " ").lower().split())


def levenshtein(a, b, cap=2):
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def match_kinds(predicted, truth):
    pred, gold = normalise(predicted), normalise(truth)
    if not pred or pred == " ".join(NOT_MENTIONED_TOKENS):
        return {"exact": False, "substring": False, "fuzzy": False}

    exact = pred == gold
    substring = exact or pred in gold or gold in pred

    pred_tokens, gold_tokens = set(pred.split()), set(gold.split())
    overlap = bool(pred_tokens & gold_tokens)
    head_close = levenshtein(pred.split()[-1], gold.split()[-1]) <= 1
    fuzzy = substring or overlap or head_close

    return {"exact": exact, "substring": substring, "fuzzy": fuzzy}


def _iter_parsed(records, preloaded, nlp):
    if preloaded is not None:
        for record in records:
            try:
                yield record, preloaded[record["scene_id"]][record["object_id"]][record["ann_id"]]
            except KeyError:
                continue
        return

    from experiments.ablation.parsers.spacy_parser import parse_doc
    from experiments.ablation.parsers.tokenize_parse import tokenize_record

    texts = [r["description"].lower() if r["description"] else "" for r in records]
    for record, doc in zip(records, nlp.pipe(texts, batch_size=256)):
        yield record, tokenize_record(parse_doc(doc))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parser", default="spacy", choices=["spacy"],
                        help="Parser to run when --parsed-dir is not given.")
    parser.add_argument("--model", default="en_core_web_sm")
    parser.add_argument("--parsed-dir", dest="parsed_dir", default=None,
                        help="Score an existing tokenized_parsed_result_*.json instead of "
                             "re-parsing. Use this for the GPT-4o-mini and LLaMA caches.")
    parser.add_argument("--tag", default=None,
                        help="Name for this parser in the report filenames.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=30,
                        help="How many parsed examples to dump for manual review.")
    parser.add_argument("--report-dir", dest="report_dir", default="outputs/parser_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tag = args.tag or (os.path.basename(args.parsed_dir.rstrip("/")) if args.parsed_dir
                       else args.parser)

    random.seed(args.seed)
    os.makedirs(args.report_dir, exist_ok=True)
    report = {"tag": tag, "parsed_dir": args.parsed_dir, "model": args.model, "splits": {}}

    nlp = None
    if args.parsed_dir is None:
        from experiments.ablation.parsers.spacy_parser import load_parser
        nlp = load_parser(args.model)

    for split in args.splits:
        records, source = load_split(split, args.data_root)
        if args.limit:
            records = records[: args.limit]

        preloaded = None
        if args.parsed_dir:
            path = os.path.join(args.parsed_dir, f"tokenized_parsed_result_{split}.json")
            with open(path) as f:
                preloaded = json.load(f)

        counts = Counter()
        total = 0
        rows, failures = [], []
        confusion = Counter()

        for record, parsed in _iter_parsed(records, preloaded, nlp):
            predicted = " ".join(parsed["target"])
            kinds = match_kinds(predicted, record["object_name"])

            total += 1
            for kind, hit in kinds.items():
                counts[kind] += int(hit)
            if parsed["target"] == NOT_MENTIONED_TOKENS:
                counts["no_target"] += 1

            row = {
                "scene_id": record["scene_id"],
                "object_id": record["object_id"],
                "ann_id": record["ann_id"],
                "description": record["description"],
                "object_name (ground truth)": record["object_name"],
                "parsed": parsed,
                "match": kinds,
            }
            rows.append(row)
            if not kinds["fuzzy"]:
                failures.append(row)
                confusion[(normalise(record["object_name"]), normalise(predicted))] += 1

        pct = {k: 100.0 * counts[k] / max(total, 1) for k in ("exact", "substring", "fuzzy")}
        print(f"\n=== {tag} | {split}  ({total} descriptions from {os.path.basename(source)}) ===")
        print(f"  exact match      : {counts['exact']:>6} / {total}  = {pct['exact']:.2f}%")
        print(f"  substring match  : {counts['substring']:>6} / {total}  = {pct['substring']:.2f}%")
        print(f"  fuzzy match      : {counts['fuzzy']:>6} / {total}  = {pct['fuzzy']:.2f}%")
        print(f"  no target found  : {counts['no_target']:>6} / {total}  "
              f"= {100.0 * counts['no_target'] / max(total, 1):.2f}%")

        if confusion:
            print("\n  most frequent target errors (ground truth -> predicted):")
            for (gold, pred), n in confusion.most_common(10):
                print(f"    {n:>5}x  {gold!r} -> {pred!r}")

        num_fail = min(len(failures), max(1, args.num_samples // 3)) if failures else 0
        num_rand = max(0, args.num_samples - num_fail)
        samples = random.sample(rows, min(num_rand, len(rows)))
        if num_fail:
            samples += random.sample(failures, num_fail)

        sample_path = os.path.join(args.report_dir, f"samples_{split}_{tag}.json")
        with open(sample_path, "w") as f:
            json.dump({
                "note": "Random draws followed by failure cases, for manual qualitative review.",
                "tag": tag,
                "num_random": len(samples) - num_fail,
                "num_failures": num_fail,
                "samples": samples,
            }, f, indent=2)

        txt_path = os.path.join(args.report_dir, f"samples_{split}_{tag}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Parser samples - {tag} - split={split}\n")
            f.write("=" * 88 + "\n\n")
            for i, row in enumerate(samples, 1):
                verdict = "OK " if row["match"]["fuzzy"] else "MISS"
                f.write(f"[{i:02d}] {verdict}  {row['scene_id']} "
                        f"obj={row['object_id']} ann={row['ann_id']}\n")
                f.write(f"  description : {row['description']}\n")
                f.write(f"  object_name : {row['object_name (ground truth)']}\n")
                f.write(f"  target      : {row['parsed']['target']}\n")
                f.write(f"  adjectives  : {row['parsed']['adjectives']}\n")
                f.write(f"  neighbors   : {row['parsed']['neighbors']}\n\n")

        print(f"\n  wrote {sample_path}")
        print(f"  wrote {txt_path}")

        report["splits"][split] = {
            "total": total,
            "exact": counts["exact"], "exact_pct": round(pct["exact"], 2),
            "substring": counts["substring"], "substring_pct": round(pct["substring"], 2),
            "fuzzy": counts["fuzzy"], "fuzzy_pct": round(pct["fuzzy"], 2),
            "no_target": counts["no_target"],
            "top_errors": [
                {"ground_truth": g, "predicted": p, "count": n}
                for (g, p), n in confusion.most_common(20)
            ],
        }

    report_path = os.path.join(args.report_dir, f"target_accuracy_{tag}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
