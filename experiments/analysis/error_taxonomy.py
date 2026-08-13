"""
Tally the filled annotation sheets into the parse-error taxonomy table.

    RUNS ON: CPU. Instant. No GPU, no model.

    python experiments/analysis/error_taxonomy.py \
        --sheet gpt4o-mini=outputs/analysis/annotation/annotation_sheet_gpt4o-mini.csv \
        --sheet spacy=outputs/analysis/annotation/annotation_sheet_spacy.csv

    # two annotators on the same parser -> Cohen's kappa
    python experiments/analysis/error_taxonomy.py --sheet gpt4o-mini=a.csv --sheet gpt4o-mini=b.csv

``annotation_sheet.py`` exports the sample and a human fills it in; this turns it into
numbers.

Unknown error codes, out-of-range ``*_ok`` values and blank rows are reported as problems
rather than coerced, so a typo cannot become a category with n=1.

The sample oversamples failures (see ``sampling_manifest.json``), so its raw field accuracy
is not the dataset rate. Both are produced: ``sample`` for the annotated set, and
``population`` re-weighted by stratum.

Given two sheets for the same parser, raw agreement and Cohen's kappa per field are
reported as well.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict

# Resolve the repo root from this file, not the cwd, so the script works when
# invoked as `python experiments/analysis/<name>.py` from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.analysis.annotation_sheet import TAXONOMY
from experiments.analysis.common import ensure_dir, md_table, save_json, wilson_ci

FIELDS = ("target_ok", "adjectives_ok", "neighbors_ok")

_TRUE = {"1", "y", "yes", "true", "t", "ok"}
_FALSE = {"0", "n", "no", "false", "f"}


def to_bool(value):
    """Parse a tolerant yes/no cell. Returns True, False, or None when blank/unreadable."""
    text = (value or "").strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def read_sheet(path):
    """Return (rows, problems). Rows keep their raw cells; validation is reported."""
    if not os.path.isfile(path):
        raise SystemExit(f"annotation sheet not found: {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    problems = []
    for row in rows:
        sample_id = row.get("sample_id", "?")
        for field in FIELDS:
            if field not in row:
                problems.append(f"{sample_id}: column {field!r} missing from the sheet")
            elif to_bool(row[field]) is None:
                problems.append(f"{sample_id}: {field}={row[field]!r} is not yes/no")
        for code in split_codes(row.get("error_type")):
            if code not in TAXONOMY:
                problems.append(f"{sample_id}: unknown error code {code!r} "
                                f"(known: {', '.join(sorted(TAXONOMY))})")
    return rows, problems


def split_codes(value):
    return [c.strip().lower() for c in (value or "").split(";") if c.strip()]


def load_weights(manifest_path):
    """Per-stratum weights from the sampling manifest, plus the sample_id -> stratum map.

    The manifest does not record which stratum each sample came from directly, but
    ``auto_target_match`` in the sheet does: MISS means the target-wrong stratum.
    """
    if not manifest_path or not os.path.isfile(manifest_path):
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    strata = manifest.get("strata") or {}
    return {
        "target_wrong": strata.get("target_wrong", {}).get("weight", 1.0),
        "target_correct": strata.get("target_correct", {}).get("weight", 1.0),
        "population_total": (strata.get("target_wrong", {}).get("population", 0)
                             + strata.get("target_correct", {}).get("population", 0)),
    }


def stratum_of(row):
    return "target_wrong" if (row.get("auto_target_match", "").strip().upper() == "MISS") \
        else "target_correct"


def field_rates(rows, weights):
    """Sample and population-weighted accuracy per field."""
    result = {}
    for field in FIELDS:
        values = [(stratum_of(row), to_bool(row.get(field))) for row in rows]
        graded = [(s, v) for s, v in values if v is not None]
        n = len(graded)
        hits = sum(1 for _s, v in graded if v)
        low, high = wilson_ci(hits, n)
        entry = {"n": n, "hits": hits, "sample_rate": (hits / n) if n else 0.0,
                 "ci": [low, high], "ungraded": len(values) - n}

        if weights:
            numerator, denominator = 0.0, 0.0
            for stratum in ("target_wrong", "target_correct"):
                subset = [v for s, v in graded if s == stratum]
                if not subset:
                    continue
                weight = weights.get(stratum, 1.0)
                numerator += weight * sum(1 for v in subset if v)
                denominator += weight * len(subset)
            entry["population_rate"] = (numerator / denominator) if denominator else None
        result[field] = entry
    return result


def cohens_kappa(a, b):
    """Cohen's kappa for two aligned binary label lists, ignoring unpaired blanks."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    pa_true = sum(1 for x, _ in pairs if x) / n
    pb_true = sum(1 for _, y in pairs if y) / n
    expected = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if expected >= 1.0:
        return 1.0, n
    return (observed - expected) / (1 - expected), n


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheet", action="append", required=True, metavar="NAME=PATH",
                        help="Repeatable. Two sheets with the same NAME are treated as "
                             "two annotators and compared.")
    parser.add_argument("--manifest",
                        default="outputs/analysis/annotation/sampling_manifest.json",
                        help="Sampling manifest, used to re-weight to population rates.")
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/analysis/annotation")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any validation problem is found.")
    args = parser.parse_args()

    weights = load_weights(args.manifest)
    if weights:
        print(f"[weights] target_wrong x{weights['target_wrong']:.2f}, "
              f"target_correct x{weights['target_correct']:.2f} "
              f"(population {weights['population_total']})")
    else:
        print("[weights] no sampling manifest; population rates will not be reported")

    sheets = defaultdict(list)
    all_problems = []
    for spec in args.sheet:
        name, path = (spec.split("=", 1) if "=" in spec else (spec, spec))
        rows, problems = read_sheet(path.strip())
        sheets[name.strip()].append((path.strip(), rows))
        all_problems += [f"{name.strip()} [{os.path.basename(path)}] {p}" for p in problems]
        print(f"[{name.strip()}] {len(rows)} rows from {path.strip()}")

    if all_problems:
        print(f"\n{len(all_problems)} validation problem(s):")
        for problem in all_problems[:25]:
            print("  -", problem)
        if len(all_problems) > 25:
            print(f"  ... and {len(all_problems) - 25} more")
        print("  (rows with unreadable cells are excluded from that field's rate)")

    report, summary_rows, taxonomy_rows = {}, [], []
    for name, versions in sheets.items():
        path, rows = versions[0]
        rates = field_rates(rows, weights)

        codes = Counter()
        for row in rows:
            for code in split_codes(row.get("error_type")):
                codes[code] += 1
        graded_rows = sum(1 for row in rows if split_codes(row.get("error_type")))

        print(f"\n=== {name} ===")
        for field in FIELDS:
            entry = rates[field]
            population = entry.get("population_rate")
            population_text = (f"  population {100 * population:.1f}%"
                               if population is not None else "")
            print(f"  {field:<15}: sample {100 * entry['sample_rate']:.1f}% "
                  f"[{100 * entry['ci'][0]:.1f}-{100 * entry['ci'][1]:.1f}]  "
                  f"n={entry['n']}{population_text}"
                  + (f"  ({entry['ungraded']} ungraded)" if entry["ungraded"] else ""))

        if codes:
            print(f"  error codes over {graded_rows} labelled rows:")
            for code, count in codes.most_common():
                print(f"    {code:<24} {count:>4}  "
                      f"{100.0 * count / max(graded_rows, 1):.1f}%")
        else:
            print("  no error_type values filled in yet")

        agreement = {}
        if len(versions) > 1:
            other_path, other_rows = versions[1]
            index = {row.get("sample_id"): row for row in other_rows}
            print(f"  agreement with {os.path.basename(other_path)}:")
            for field in FIELDS:
                a = [to_bool(row.get(field)) for row in rows]
                b = [to_bool(index.get(row.get("sample_id"), {}).get(field)) for row in rows]
                kappa, n = cohens_kappa(a, b)
                raw = (sum(1 for x, y in zip(a, b)
                           if x is not None and y is not None and x == y) / n) if n else 0.0
                agreement[field] = {"kappa": kappa, "raw": raw, "n": n}
                kappa_text = f"{kappa:.3f}" if kappa is not None else "n/a"
                print(f"    {field:<15}: raw {100 * raw:.1f}%  kappa {kappa_text}  n={n}")

        report[name] = {"sheet": path, "n_rows": len(rows), "fields": rates,
                        "error_codes": dict(codes), "labelled_rows": graded_rows,
                        "agreement": agreement}

        row = [name, len(rows)]
        for field in FIELDS:
            entry = rates[field]
            population = entry.get("population_rate")
            row.append(f"{100 * entry['sample_rate']:.1f}" +
                       (f" / {100 * population:.1f}" if population is not None else ""))
        summary_rows.append(row)

        for code in TAXONOMY:
            if codes.get(code):
                taxonomy_rows.append([name, code, codes[code],
                                      f"{100.0 * codes[code] / max(graded_rows, 1):.1f}"])

    summary = md_table(
        ["parser", "n", "target_ok (sample/pop %)", "adjectives_ok (sample/pop %)",
         "neighbors_ok (sample/pop %)"], summary_rows)
    print("\n" + summary)

    lines = ["# Manual parse annotation — results\n",
             "Field accuracy, as `sample / population`. The sample oversamples target "
             "failures on purpose; the population column re-weights by stratum and is the "
             "one to quote as a dataset rate.\n", summary]
    if taxonomy_rows:
        taxonomy_table = md_table(["parser", "error code", "count", "% of labelled rows"],
                                  taxonomy_rows)
        print("\n" + taxonomy_table)
        lines += ["\n## Error taxonomy\n", taxonomy_table,
                  "\n" + "\n".join(f"- `{code}` — {text}" for code, text in TAXONOMY.items())]

    ensure_dir(args.out_dir)
    report_path = os.path.join(args.out_dir, "error_taxonomy.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json({"weights": weights, "problems": all_problems,
                           "parsers": report},
                          os.path.join(args.out_dir, "error_taxonomy.json"))

    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")

    if all_problems and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
