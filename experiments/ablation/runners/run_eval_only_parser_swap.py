"""
Evaluation-only parser ablation: swap the parser at test time on a trained checkpoint.

    RUNS ON: GPU. Evaluation only -- no training. One val pass per parser.

    python experiments/ablation/runners/run_eval_only_parser_swap.py

Takes the model trained on GPT-4o-mini parses and feeds it spaCy / LLaMA / no-parse output
at test time, measuring sensitivity to parse quality in a few GPU-minutes instead of
several GPU-hours.

The model was trained on one parser's output distribution and is evaluated on another's, so
the result confounds the intrinsic quality of the alternative parse with the train/test
distribution mismatch. The controlled comparison is the set of per-variant training runs
(``run_parser_{gpt,spacy,llama,none,smalllm}.py``); this is a supplement to them.

Result layout::

    outputs/<FOLDER>/parser_swap/
        gpt4o-mini/{predictions.p, scores.p, eval_stdout.txt}     <- train/test matched
        spacy/...
        llama/...
        none/...
        run_summary.json

The first entry is the parse the model was trained on, so the others are read against it.
Any pre-existing ``predictions.p`` / ``scores.p`` is backed up and restored, since
ScanRefer_eval.py always writes to the same filenames.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time

# ======================================================================================
# CONFIG -- edit here
# ======================================================================================

RUN = True                                     # master switch for this experiment
FOLDER = "2024-12-18_20-40-38_3DVG-FIXED"      # trained run under outputs/ to evaluate
#: Which MatchModule to build. FOLDER above was trained with the older fusion
#: head, so "original" is required -- with "current" the eval would leave 178
#: tensors randomly initialised and ablation_hooks now refuses outright.
FUSION_VARIANT = "original"

#: (label, parse folder under data_parsing/). The FIRST is the training-time parse.
PARSERS = [
    ("gpt4o-mini", "final_parsing_tokenized"),          # matched: what it was trained on
    ("spacy", "spacy_parsing_tokenized"),
    ("llama", "llama_parsing_tokenized_clipped"),
    ("none", "noparse_tokenized"),
    ("smalllm", "smalllm_parsing_tokenized"),
]

BATCH_SIZE = 32
LANG_NUM_MAX = 1
EXTRA_ARGS = ["--use_color", "--use_normal"]   # must match the trained configuration
SKIP_MISSING = True                            # skip a parser whose cache is absent

# ======================================================================================

# experiments/ablation/runners/<this file> -> repo root is 4 levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", FOLDER)
ARCHIVE_ROOT = os.path.join(OUTPUT_DIR, "parser_swap")
ARTIFACTS = ("predictions.p", "scores.p")

# ScanRefer_eval.py prints one line per (subset, split, metric), e.g.
#   overall | overall | acc@0.25iou: 0.4643458140513252
# The overall/overall row is the one to record; the rest are subset breakdowns.
_IOU_LINE = re.compile(
    r"overall\s*\|\s*overall\s*\|\s*acc@0\.25iou:\s*([0-9.]+)", re.IGNORECASE)
_IOU_LINE_50 = re.compile(
    r"overall\s*\|\s*overall\s*\|\s*acc@0\.5iou:\s*([0-9.]+)", re.IGNORECASE)


def has_cache(parsing_folder):
    return os.path.isfile(os.path.join(REPO_ROOT, "data_parsing", parsing_folder,
                                       "tokenized_parsed_result_val.json"))


def run_one(label, parsing_folder):
    cmd = [sys.executable, os.path.join("scripts", "ScanRefer_eval.py"),
           "--folder", FOLDER, "--reference", "--force",
           "--batch_size", str(BATCH_SIZE),
           "--lang_num_max", str(LANG_NUM_MAX),
           "--parsing_folder", parsing_folder,
           "--fusion_variant", FUSION_VARIANT] + EXTRA_ARGS

    print(f"\n{'=' * 78}\n[swap] parser {label}  ({parsing_folder})")
    print("[swap] " + " ".join(cmd) + "\n")

    started = time.time()
    process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    captured = []
    for line in process.stdout:
        sys.stdout.write(line)
        captured.append(line)
    code = process.wait()
    elapsed = time.time() - started

    destination = os.path.join(ARCHIVE_ROOT, label)
    os.makedirs(destination, exist_ok=True)
    moved = []
    for artifact in ARTIFACTS:
        source = os.path.join(OUTPUT_DIR, artifact)
        if os.path.isfile(source):
            shutil.move(source, os.path.join(destination, artifact))
            moved.append(artifact)
    with open(os.path.join(destination, "eval_stdout.txt"), "w") as f:
        f.writelines(captured)

    def _scrape(pattern):
        for line in reversed(captured):
            match = pattern.search(line)
            if match:
                return float(match.group(1))
        return None

    accuracy = _scrape(_IOU_LINE)
    accuracy_50 = _scrape(_IOU_LINE_50)
    if accuracy is None and code == 0:
        # A clean exit with no parseable accuracy means the eval output format moved.
        # Say so at the time rather than writing a null nobody notices.
        print(f"[swap] WARNING: {label} exited 0 but no 'overall | overall | acc@0.25iou' "
              f"line was found. Check {os.path.join(destination, 'eval_stdout.txt')}.")

    print(f"[swap] {label} finished in {elapsed / 60:.1f} min (exit {code}); "
          f"archived {moved}")
    return {"parser": label, "parsing_folder": parsing_folder, "exit_code": code,
            "elapsed_sec": round(elapsed, 1), "acc_25": accuracy,
            "acc_50": accuracy_50, "dir": destination}


def _cli_guard():
    """These runners are configured by the CONFIG block above, not by command-line flags.

    Without this, `--help` or a mistyped flag would be silently ignored and a long run
    would start anyway. Any argument prints the configuration and exits instead.
    """
    if len(sys.argv) <= 1:
        return False
    print(__doc__)
    print("This runner takes no command-line arguments. Edit the CONFIG block in")
    print(f"    {os.path.abspath(__file__)}")
    print("\nCurrent configuration:")
    for key, value in sorted(globals().items()):
        if key.isupper() and not key.startswith("_"):
            print(f"    {key} = {value!r}")
    return True


def main():
    if _cli_guard():
        return 0
    if not RUN:
        print("RUN = False in this file; nothing to do.")
        return 0

    if not os.path.isdir(OUTPUT_DIR):
        print(f"ERROR: trained run not found: {OUTPUT_DIR}")
        return 1

    selected = []
    for label, parsing_folder in PARSERS:
        if has_cache(parsing_folder):
            selected.append((label, parsing_folder))
        elif SKIP_MISSING:
            print(f"[swap] skipping {label}: no cache at "
                  f"data_parsing/{parsing_folder}/tokenized_parsed_result_val.json")
        else:
            print(f"ERROR: missing parse cache for {label} ({parsing_folder})")
            return 1
    if not selected:
        print("ERROR: no parse cache available; nothing to evaluate.")
        return 1

    print("\n" + "!" * 78)
    print("This is an evaluation-only swap. The model was TRAINED on "
          f"{PARSERS[0][1]!r}.")
    print("Results for the other parsers confound parse quality with train/test")
    print("distribution mismatch, and are NOT a fair parser comparison. State that")
    print("limitation explicitly wherever these numbers appear.")
    print("!" * 78)

    os.makedirs(ARCHIVE_ROOT, exist_ok=True)
    backups = []
    for artifact in ARTIFACTS:
        source = os.path.join(OUTPUT_DIR, artifact)
        if os.path.isfile(source):
            backup = source + ".bak"
            shutil.copy(source, backup)
            backups.append((source, backup))

    results, failures = [], []
    for label, parsing_folder in selected:
        result = run_one(label, parsing_folder)
        results.append(result)
        if result["exit_code"] != 0:
            failures.append(label)

    summary_path = os.path.join(ARCHIVE_ROOT, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"folder": FOLDER, "trained_on": PARSERS[0][1],
                   "caveat": "evaluation-only swap; confounds parse quality with "
                             "train/test distribution mismatch",
                   "results": results}, f, indent=2)

    for source, backup in backups:
        shutil.move(backup, source)

    print("\n[swap] summary:")
    reference = next((r for r in results if r["parser"] == PARSERS[0][0]), None)
    for result in results:
        accuracy = result["acc_25"]
        text = f"{100 * accuracy:.2f}%" if accuracy is not None else "see eval_stdout.txt"
        delta = ""
        if reference and reference["acc_25"] and accuracy is not None \
                and result is not reference:
            delta = f"  ({100 * (accuracy - reference['acc_25']):+.2f} pp vs trained parse)"
        print(f"  {result['parser']:<12} Acc@0.25 = {text}{delta}")
    print(f"\n[swap] -> {summary_path}")

    if failures:
        print(f"[swap] FAILED: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
