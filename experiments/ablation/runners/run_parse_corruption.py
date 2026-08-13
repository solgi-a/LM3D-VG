
import json
import os
import shutil
import subprocess
import sys
import time


RUN = True
FOLDER = "2024-12-18_20-40-38_3DVG-FIXED"
FUSION_VARIANT = "original"
SOURCE_PARSING = "final_parsing_tokenized"
RATES = [0.10, 0.25, 0.50]
MODE = "all"
SEED = 42
SPLIT = "val"
BATCH_SIZE = 8
LANG_NUM_MAX = 1
EXTRA_ARGS = ["--use_color", "--use_normal"]
RUN_BASELINE = True
GENERATE_CACHES = True


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
