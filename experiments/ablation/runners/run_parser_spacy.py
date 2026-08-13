
import os
import subprocess
import sys


RUN = True
TAG = "ABL-PARSER-SPACY"
PARSING_FOLDER = "spacy_parsing_tokenized"
USE_CACHED_SCENES = True
CACHED_SCENES_ROOT = 'cached_scenes'
SEED = 42
EPOCH = 50
BATCH_SIZE = 8
EXTRA_ARGS = ["--use_color", "--use_normal"]

WARM_START      = True
WARM_START_FROM = '2024-12-18_20-40-38_3DVG-FIXED'
FUSION_VARIANT  = 'original'

VAL_STEP  = 5000
VERBOSE   = 10
LR        = 0.002
COSLR     = True
LANG_NUM_MAX = 32

NUM_WORKERS     = None
PREFETCH_FACTOR = 3


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def check_parse_cache():
    folder = os.path.join(REPO_ROOT, "data_parsing", PARSING_FOLDER)
    missing = [s for s in ("train", "val")
               if not os.path.isfile(os.path.join(folder, f"tokenized_parsed_result_{s}.json"))]
    if missing:
        print(f"ERROR: no spaCy parse cache for split(s) {missing} in {folder}")
        print("Generate it first:")
        print("    python experiments/ablation/parsers/run_spacy_parser.py --splits train val")
        return False
    return True


def build_command():
    cmd = [sys.executable, os.path.join("scripts", "ScanRefer_train.py"),
           "--tag", TAG,
           "--seed", str(SEED),
           "--epoch", str(EPOCH),
           "--batch_size", str(BATCH_SIZE),
           "--parsing_folder", PARSING_FOLDER]
    if USE_CACHED_SCENES:
        cmd += ["--use_cached_scenes", "--cached_scenes_root", CACHED_SCENES_ROOT]
    if WARM_START:
        cmd += ["--use_checkpoint", WARM_START_FROM, "--keep_checkpoint",
                "--fusion_variant", FUSION_VARIANT]
    else:
        cmd += ["--no_warm_start"]
    cmd += ["--val_step", str(VAL_STEP),
            "--verbose", str(VERBOSE),
            "--lr", str(LR),
            "--lang_num_max", str(LANG_NUM_MAX)]
    if COSLR:
        cmd += ["--coslr"]
    if NUM_WORKERS is not None:
        cmd += ["--num_workers", str(NUM_WORKERS)]
    cmd += ["--prefetch_factor", str(PREFETCH_FACTOR)]
    return cmd + EXTRA_ARGS


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
    if not check_parse_cache():
        return 1
    cmd = build_command()
    print("[ablation] parser variant B (rule-based spaCy)")
    print("[ablation] " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
