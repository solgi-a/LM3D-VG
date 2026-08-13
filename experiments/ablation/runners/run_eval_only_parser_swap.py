
import json
import os
import re
import shutil
import subprocess
import sys
import time


RUN = True
FOLDER = "2024-12-18_20-40-38_3DVG-FIXED"
FUSION_VARIANT = "original"

PARSERS = [
    ("gpt4o-mini", "final_parsing_tokenized"),
    ("spacy", "spacy_parsing_tokenized"),
    ("llama", "llama_parsing_tokenized_clipped"),
    ("none", "noparse_tokenized"),
]

BATCH_SIZE = 8
LANG_NUM_MAX = 1
EXTRA_ARGS = ["--use_color", "--use_normal"]
SKIP_MISSING = True


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", FOLDER)
ARCHIVE_ROOT = os.path.join(OUTPUT_DIR, "parser_swap")
ARTIFACTS = ("predictions.p", "scores.p")

_IOU_LINE = re.compile(r"iou rate 0\.25:\s*([0-9.]+)", re.IGNORECASE)


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

    accuracy = None
    for line in reversed(captured):
        match = _IOU_LINE.search(line)
        if match:
            accuracy = float(match.group(1))
            break

    print(f"[swap] {label} finished in {elapsed / 60:.1f} min (exit {code}); "
          f"archived {moved}")
    return {"parser": label, "parsing_folder": parsing_folder, "exit_code": code,
            "elapsed_sec": round(elapsed, 1), "acc_25": accuracy, "dir": destination}


def _cli_guard():
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
