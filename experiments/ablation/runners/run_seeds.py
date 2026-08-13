
import os
import subprocess
import sys


RUN = True
SEEDS = [1, 2]
TAG_PREFIX = "ABL-SEED"
PARSING_FOLDER = "final_parsing_tokenized"
USE_CACHED_SCENES = True
CACHED_SCENES_ROOT = 'cached_scenes'
EPOCH = 50
BATCH_SIZE = 8
STOP_ON_FAILURE = True
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


def build_command(seed):
    cmd = [sys.executable, os.path.join("scripts", "ScanRefer_train.py"),
           "--tag", f"{TAG_PREFIX}{seed}",
           "--seed", str(seed),
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

    failures = []
    for seed in SEEDS:
        cmd = build_command(seed)
        print(f"\n[ablation] seed {seed} of {SEEDS}")
        print("[ablation] " + " ".join(cmd) + "\n")
        code = subprocess.call(cmd, cwd=REPO_ROOT)
        if code != 0:
            failures.append((seed, code))
            print(f"[ablation] seed {seed} exited with code {code}")
            if STOP_ON_FAILURE:
                break

    if failures:
        print("\n[ablation] failed seeds: " + ", ".join(f"{s} (code {c})" for s, c in failures))
        return 1
    print(f"\n[ablation] all seeds finished: {SEEDS}")
    print("[ablation] collect iou_rate_0.25 / iou_rate_0.5 from each outputs/*_"
          f"{TAG_PREFIX}*/best.txt and report mean +/- std.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
