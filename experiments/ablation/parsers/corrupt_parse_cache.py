"""
Deliberately corrupt a parse cache, to measure whether parse errors propagate.

    RUNS ON: CPU. Seconds. No model, no GPU, no network.

    python experiments/ablation/parsers/corrupt_parse_cache.py --splits val --rates 0.10 0.25 0.50 --mode all

Needs no training -- only re-evaluation of an existing checkpoint.

What gets corrupted
-------------------
A fixed fraction of annotations is selected (seeded, reproducible) and rewritten:

``swap``      ``target`` is replaced with a different annotation's target -- the wrong
              object class, the most consequential parser failure.
``drop``      all three fields become ``["not", "mentioned"]`` -- a parser that produced
              nothing usable.
``shuffle``   ``neighbors`` is replaced with a different annotation's neighbors -- a
              hallucinated spatial context, the failure mode specific to the adjacency
              claim.
``all``       one of the three above per selected annotation, drawn uniformly. The
              default: a mixed error profile is closer to what a real parser does.

Output
------
One directory per (mode, rate), named ``<source>_corrupt_<mode>_<pct>``::

    data_parsing/final_parsing_tokenized_corrupt_all_25/
        tokenized_parsed_result_val.json
        corruption_manifest.json

The manifest records which (scene_id, object_id, ann_id) were corrupted and how, which lets
``parse_error_propagation.py`` compare the corrupted subset against the untouched one at
the same rate. The global accuracy curve mixes the two and dilutes the effect by (1 - rate).

Corrupted output goes through the same validator as every other cache
(``tokenize_parse.py``): keys present, no empty field, caps 7/17/75 respected. A donor is
redrawn until it differs from the original, so the reported rate is the effective one.

Next step (GPU)::

    python experiments/ablation/runners/run_parse_corruption.py
"""

import argparse
import json
import os
import random
import sys

# Resolve the repo root from this file, not the cwd, so the script works when
# invoked as `python experiments/ablation/parsers/<name>.py` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from experiments.ablation.parsers.tokenize_parse import (
    NOT_MENTIONED_TOKENS,
    clip_tokens,
    validate_tokenized,
)

FIELDS = ("target", "adjectives", "neighbors")
MODES = ("swap", "drop", "shuffle", "all")

#: How many times to redraw a donor before giving up and leaving the entry unchanged.
_MAX_DONOR_TRIES = 12


def flat_keys(data):
    """Sorted [(scene_id, object_id, ann_id)] so sampling is reproducible across runs."""
    keys = []
    for scene_id, objects in data.items():
        for object_id, anns in objects.items():
            for ann_id in anns:
                keys.append((scene_id, object_id, ann_id))
    keys.sort()
    return keys


def _get(data, key):
    scene_id, object_id, ann_id = key
    return data[scene_id][object_id][ann_id]


def _draw_donor(data, keys, key, field, rng):
    """A different annotation's value for ``field``, guaranteed to differ from key's."""
    original = _get(data, key)[field]
    for _ in range(_MAX_DONOR_TRIES):
        donor_key = keys[rng.randrange(len(keys))]
        if donor_key == key:
            continue
        candidate = _get(data, donor_key)[field]
        if candidate != original:
            return candidate
    return None


def corrupt(data, rate, mode, seed):
    """Return (corrupted_copy, manifest_entries, stats). ``data`` is not modified."""
    rng = random.Random(seed)
    keys = flat_keys(data)
    num_target = int(round(rate * len(keys)))
    selected = rng.sample(keys, num_target) if num_target else []

    # Deep-ish copy: the field lists are replaced wholesale, never mutated in place, so
    # copying one level below the annotation dict is sufficient and much cheaper.
    out = {
        scene_id: {
            object_id: {ann_id: {f: list(parsed[f]) for f in FIELDS}
                        for ann_id, parsed in anns.items()}
            for object_id, anns in objects.items()
        }
        for scene_id, objects in data.items()
    }

    manifest, stats = [], {"selected": len(selected), "applied": 0, "no_op": 0}
    for m in ("swap", "drop", "shuffle"):
        stats[f"applied_{m}"] = 0

    for key in sorted(selected):
        applied = mode if mode != "all" else rng.choice(("swap", "drop", "shuffle"))
        scene_id, object_id, ann_id = key
        entry = out[scene_id][object_id][ann_id]
        before = {f: list(entry[f]) for f in FIELDS}

        if applied == "drop":
            for field in FIELDS:
                entry[field] = list(NOT_MENTIONED_TOKENS)
        else:
            field = "target" if applied == "swap" else "neighbors"
            donor = _draw_donor(data, keys, key, field, rng)
            if donor is None:
                stats["no_op"] += 1
                continue
            entry[field] = clip_tokens(list(donor), field)

        if all(entry[f] == before[f] for f in FIELDS):
            # Can happen for `drop` when the parse was already all "not mentioned".
            stats["no_op"] += 1
            continue

        stats["applied"] += 1
        stats[f"applied_{applied}"] += 1
        manifest.append({
            "scene_id": scene_id, "object_id": object_id, "ann_id": ann_id,
            "applied": applied,
            "before": before,
            "after": {f: list(entry[f]) for f in FIELDS},
        })

    stats["total"] = len(keys)
    stats["effective_rate"] = round(stats["applied"] / max(len(keys), 1), 6)
    return out, manifest, stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--splits", nargs="+", default=["val"],
                        help="Only 'val' is needed for the evaluation-only experiment.")
    parser.add_argument("--source", default="final_parsing_tokenized",
                        help="Parse cache to corrupt, under data_parsing/.")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--rates", nargs="+", type=float, default=[0.10, 0.25, 0.50])
    parser.add_argument("--mode", default="all", choices=MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Keep only the first N annotations (smoke test).")
    parser.add_argument("--out-root", dest="out_root", default=None,
                        help="Defaults to --parsing-root.")
    args = parser.parse_args()

    out_root = args.out_root or args.parsing_root
    ok = True

    for rate in args.rates:
        if not 0.0 <= rate <= 1.0:
            parser.error(f"--rates must be in [0, 1]; got {rate}")
        pct = int(round(rate * 100))
        out_dir = os.path.join(out_root, f"{args.source}_corrupt_{args.mode}_{pct:02d}")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n=== corruption {pct}% | mode={args.mode} | seed={args.seed} ===")
        manifest_all, stats_all = {}, {}

        for split in args.splits:
            src = os.path.join(args.parsing_root, args.source,
                               f"tokenized_parsed_result_{split}.json")
            if not os.path.isfile(src):
                print(f"ERROR: source cache not found: {src}")
                return 1
            with open(src) as f:
                data = json.load(f)

            if args.limit:
                keep = set(flat_keys(data)[: args.limit])
                data = {
                    s: {o: {a: p for a, p in anns.items() if (s, o, a) in keep}
                        for o, anns in objs.items()}
                    for s, objs in data.items()
                }
                data = {s: {o: a for o, a in objs.items() if a}
                        for s, objs in data.items()}
                data = {s: o for s, o in data.items() if o}

            # Per-split seed offset, so val and train are not corrupted in lockstep.
            corrupted, manifest, stats = corrupt(
                data, rate, args.mode, args.seed + hash(split) % 1000)

            path = os.path.join(out_dir, f"tokenized_parsed_result_{split}.json")
            with open(path, "w") as f:
                json.dump(corrupted, f)

            passed, problems, checked = validate_tokenized(corrupted)
            status = "passed" if passed else f"FAILED ({len(problems)} problems)"
            print(f"[{split}] {stats['applied']}/{stats['total']} corrupted "
                  f"(effective {100 * stats['effective_rate']:.2f}%, "
                  f"no-op {stats['no_op']})")
            print(f"[{split}]   swap={stats['applied_swap']}  drop={stats['applied_drop']}"
                  f"  shuffle={stats['applied_shuffle']}")
            print(f"[{split}] -> {path}")
            print(f"[{split}] schema check {status} over {checked} entries")
            if not passed:
                ok = False
                for problem in problems[:10]:
                    print("  -", problem)

            manifest_all[split] = manifest
            stats_all[split] = stats

        manifest_path = os.path.join(out_dir, "corruption_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({
                "source": args.source,
                "mode": args.mode,
                "requested_rate": rate,
                "seed": args.seed,
                "stats": stats_all,
                "corrupted": manifest_all,
            }, f, indent=2)
        print(f"[manifest] -> {manifest_path}")

    print("\nEvaluate each level against the trained checkpoint (GPU):")
    print("    python experiments/ablation/runners/run_parse_corruption.py")
    print("then aggregate (CPU):")
    print("    python experiments/analysis/parse_error_propagation.py --run-dir outputs/<folder>/corruption")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
