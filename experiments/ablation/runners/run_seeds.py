"""
Seed ablation -- repeat the main configuration under several random seeds.

    python experiments/ablation/runners/run_seeds.py

Runs the unchanged main configuration once per seed, each in its own output folder, so the
run-to-run spread can be measured.

Geometric point-cloud augmentation is off in cached mode, but copy-paste in MatchModule,
word masking / sentence reversal in LangModule, weight init and batch order all still vary
with the seed.
"""

import os
import subprocess
import sys

# ======================================================================================
# CONFIG -- edit here
# ======================================================================================

RUN = True                                    # master switch for this experiment
SEEDS = [1, 2]                                # two extra seeds beside the paper's 42
TAG_PREFIX = "ABL-SEED"                       # output folders: ABL-SEED1, ABL-SEED2, ...
PARSING_FOLDER = "final_parsing_tokenized"    # main configuration
USE_CACHED_SCENES = True                      # train the fusion net on cached proposals
CACHED_SCENES_ROOT = 'cached_scenes'
EPOCH = 20
BATCH_SIZE = 16
STOP_ON_FAILURE = True                        # abort the sweep if one seed fails
EXTRA_ARGS = ["--use_color", "--use_normal"]  # must match how the cache was built

# ---- warm start ----------------------------------------------------------------------
# Fine-tune from a trained run rather than random init. With the detector frozen this
# covers all 146 tensors, so it converges in far fewer epochs. All phase-A arms start
# from the same checkpoint.
WARM_START      = True
WARM_START_FROM = '2024-12-18_20-40-38_3DVG-FIXED'   # a folder under outputs/
FUSION_VARIANT  = 'original'    # the head WARM_START_FROM was trained with

# ---- training hyper-parameters --------------------------------------------------------
VAL_STEP  = 50   # validate every N *iterations* (not epochs). Lower = slower training.
VERBOSE   = 50     # print a training line every N iterations
LR        = 0.002  # initial learning rate
COSLR     = True   # cosine learning-rate schedule
LANG_NUM_MAX = 32  # language samples per scene per batch

# ---- data loading ----------------------------------------------------------------------
# __getitem__ costs ~20 ms, nearly all point-cloud work, so serial loading left the GPU
# idle ~12 min per epoch. Each worker holds its own dataset copy (~1.8 GB with the lazy
# language path).
NUM_WORKERS     = 4   # None = auto (cpu_count - 1, max 4); 0 = serial
PREFETCH_FACTOR = 1      # batches each worker keeps ready
# Setting this False only takes effect via --no_lazy_lang_data below.
LAZY_LANG_DATA  = True   # False materialises every GloVe embedding up front, ~36 GB RAM



# ======================================================================================

# experiments/ablation/runners/<this file> -> repo root is 4 levels up.
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
        # ablation_config.apply() clears --use_checkpoint in cached mode; --keep_checkpoint
        # opts back in. Without it the run silently trains from random init.
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
    if not LAZY_LANG_DATA:
        cmd += ["--no_lazy_lang_data"]
    return cmd + EXTRA_ARGS


def _cli_guard():
    """Print the config and exit if any argument is passed; flags are ignored otherwise."""
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
