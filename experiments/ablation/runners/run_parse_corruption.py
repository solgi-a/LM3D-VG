"""
Parse-corruption sweep: evaluate one trained checkpoint against deliberately broken parses.

    RUNS ON: GPU. Evaluation only -- no training. One val pass per level, so roughly 4x a
             single evaluation (baseline + three corruption levels).

    python experiments/ablation/runners/run_parse_corruption.py

The sample, model and weights are fixed and only the parse changes, so the effect is
causal. The runner:

1. Generates the corrupted parse caches if missing (CPU, seconds).
2. Runs ``ScanRefer_eval.py --force`` once per level, starting with the uncorrupted parse
   for a paired baseline.
3. Archives each level's output immediately -- ``ScanRefer_eval.py`` always writes to the
   same ``predictions.p`` / ``scores.p``, so without this each level would overwrite the
   previous one.

Result layout, which experiments/analysis/parse_error_propagation.py expects::

    outputs/<FOLDER>/corruption/
        baseline/{predictions.p, scores.p, eval_stdout.txt}
        all_10/{predictions.p, scores.p, corruption_manifest.json, eval_stdout.txt}
        all_25/...
        all_50/...
        run_summary.json

Any pre-existing ``predictions.p`` is backed up and restored at the end. Then aggregate
on CPU:

    python experiments/analysis/parse_error_propagation.py --run-dir outputs/<FOLDER>/corruption
"""

import json
import os
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
SOURCE_PARSING = "final_parsing_tokenized"     # the clean parse the model was trained on
RATES = [0.10, 0.25, 0.50]                     # corruption levels
MODE = "all"                                   # swap | drop | shuffle | all
SEED = 42
SPLIT = "val"
BATCH_SIZE = 8
LANG_NUM_MAX = 1                               # eval uses one description per sample
EXTRA_ARGS = ["--use_color", "--use_normal"]   # must match the trained configuration
RUN_BASELINE = True                            # evaluate the clean parse first (paired)
GENERATE_CACHES = True                         # build missing corrupted caches

# ======================================================================================

# experiments/ablation/runners/<this file> -> repo root is 4 levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", FOLDER)
ARCHIVE_ROOT = os.path.join(OUTPUT_DIR, "corruption")
ARTIFACTS = ("predictions.p", "scores.p")


def corrupt_folder(rate):
    return f"{SOURCE_PARSING}_corrupt_{MODE}_{int(round(rate * 100)):02d}"


def ensure_caches():
    missing = [r for r in RATES
               if not os.path.isfile(os.path.join(
                   REPO_ROOT, "data_parsing", corrupt_folder(r),
                   f"tokenized_parsed_result_{SPLIT}.json"))]
    if not missing:
        return True
    if not GENERATE_CACHES:
        print(f"ERROR: missing corrupted caches for rates {missing}. Generate them with:")
        print(f"    python experiments/ablation/parsers/corrupt_parse_cache.py --splits {SPLIT} "
              f"--rates {' '.join(str(r) for r in missing)} --mode {MODE}")
        return False

    cmd = [sys.executable, "experiments/ablation/parsers/corrupt_parse_cache.py",
           "--splits", SPLIT, "--source", SOURCE_PARSING, "--mode", MODE,
           "--seed", str(SEED), "--rates"] + [str(r) for r in missing]
    print("[corruption] generating missing caches")
    print("[corruption] " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, cwd=REPO_ROOT) == 0


def eval_command(parsing_folder):
    return [sys.executable, os.path.join("scripts", "ScanRefer_eval.py"),
            "--folder", FOLDER,
            "--reference",
            "--force",
            "--batch_size", str(BATCH_SIZE),
            "--lang_num_max", str(LANG_NUM_MAX),
            "--parsing_folder", parsing_folder,
            "--fusion_variant", FUSION_VARIANT] + EXTRA_ARGS


def archive(level_name, parsing_folder):
    """Move this level's eval output out of the way before the next level overwrites it."""
    destination = os.path.join(ARCHIVE_ROOT, level_name)
    os.makedirs(destination, exist_ok=True)

    moved = []
    for artifact in ARTIFACTS:
        source = os.path.join(OUTPUT_DIR, artifact)
        if os.path.isfile(source):
            shutil.move(source, os.path.join(destination, artifact))
            moved.append(artifact)

    manifest = os.path.join(REPO_ROOT, "data_parsing", parsing_folder,
                            "corruption_manifest.json")
    if os.path.isfile(manifest):
        shutil.copy(manifest, os.path.join(destination, "corruption_manifest.json"))

    if not moved:
        print(f"[corruption] WARNING: no {'/'.join(ARTIFACTS)} produced for {level_name}")
    return destination, moved


def run_level(level_name, parsing_folder):
    cmd = eval_command(parsing_folder)
    print(f"\n{'=' * 78}\n[corruption] level {level_name}  (parse: {parsing_folder})")
    print("[corruption] " + " ".join(cmd) + "\n")

    started = time.time()
    process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    captured = []
    for line in process.stdout:
        sys.stdout.write(line)
        captured.append(line)
    code = process.wait()
    elapsed = time.time() - started

    destination, moved = archive(level_name, parsing_folder)
    with open(os.path.join(destination, "eval_stdout.txt"), "w") as f:
        f.writelines(captured)

    print(f"[corruption] level {level_name} finished in {elapsed / 60:.1f} min "
          f"(exit {code}); archived {moved} -> {destination}")
    return {"level": level_name, "parsing_folder": parsing_folder, "exit_code": code,
            "elapsed_sec": round(elapsed, 1), "archived": moved, "dir": destination}


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
        print("Set FOLDER at the top of this file to a directory under outputs/.")
        return 1
    if not ensure_caches():
        return 1

    os.makedirs(ARCHIVE_ROOT, exist_ok=True)

    # Preserve any existing evaluation output; this sweep overwrites it repeatedly.
    backups = []
    for artifact in ARTIFACTS:
        source = os.path.join(OUTPUT_DIR, artifact)
        if os.path.isfile(source):
            backup = source + ".bak"
            shutil.copy(source, backup)
            backups.append((source, backup))
    if backups:
        print(f"[corruption] backed up {len(backups)} existing artifact(s) "
              f"(restored at the end)")

    levels = []
    if RUN_BASELINE:
        levels.append(("baseline", SOURCE_PARSING))
    levels += [(f"{MODE}_{int(round(r * 100)):02d}", corrupt_folder(r)) for r in RATES]

    results, failures = [], []
    for level_name, parsing_folder in levels:
        result = run_level(level_name, parsing_folder)
        results.append(result)
        if result["exit_code"] != 0:
            failures.append(level_name)

    summary_path = os.path.join(ARCHIVE_ROOT, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"folder": FOLDER, "source_parsing": SOURCE_PARSING,
                   "mode": MODE, "rates": RATES, "seed": SEED, "split": SPLIT,
                   "levels": results}, f, indent=2)

    for source, backup in backups:
        shutil.move(backup, source)
    if backups:
        print(f"\n[corruption] restored {len(backups)} original artifact(s)")

    print(f"\n[corruption] summary -> {summary_path}")
    if failures:
        print(f"[corruption] FAILED level(s): {', '.join(failures)}")
        return 1

    print("\n[corruption] all levels done. Aggregate on CPU:")
    print(f"    python experiments/analysis/parse_error_propagation.py --run-dir "
          f"outputs/{FOLDER}/corruption")
    return 0


if __name__ == "__main__":
    sys.exit(main())
