"""
Parser ablation, variant E -- a small local language model (Qwen2-0.5B-Instruct).

    RUNS ON: GPU. One training run (~2-3 h under the cached-scene protocol), plus parse
             generation beforehand (also GPU; hours on CPU).

Build the parse cache first, then train:

    python experiments/ablation/parsers/run_smalllm_parser.py --splits train val --device cuda
    python experiments/ablation/runners/run_parser_smalllm.py

The parse step also reports its malformed-output rate.
"""

import os
import subprocess
import sys

# ======================================================================================
# CONFIG -- edit here
# ======================================================================================

RUN = True                                     # master switch for this experiment
TAG = "ABL-PARSER-SMALLLM"                     # output folder suffix under outputs/
PARSING_FOLDER = "smalllm_parsing_tokenized"   # variant E: Qwen2-0.5B-Instruct
USE_CACHED_SCENES = True                       # train the fusion net on cached proposals
CACHED_SCENES_ROOT = 'cached_scenes'
SEED = 42
EPOCH = 20
BATCH_SIZE = 16
EXTRA_ARGS = ["--use_color", "--use_normal"]   # must match how the cache was built

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


def check_parse_cache():
    folder = os.path.join(REPO_ROOT, "data_parsing", PARSING_FOLDER)
    missing = [s for s in ("train", "val")
               if not os.path.isfile(os.path.join(folder, f"tokenized_parsed_result_{s}.json"))]
    if missing:
        print(f"ERROR: no small-LM parse cache for split(s) {missing} in {folder}")
        print("Generate it first (GPU recommended):")
        print("    python experiments/ablation/parsers/run_smalllm_parser.py --splits train val --device cuda")
        return False

    summary = os.path.join(folder, "parse_run_summary.json")
    if os.path.isfile(summary):
        import json

        with open(summary) as f:
            data = json.load(f)
        print(f"[ablation] parse cache from {data.get('model')} ({data.get('kind')})")
        for split, stats in (data.get("splits") or {}).items():
            if "malformed_pct" in stats:
                print(f"[ablation]   {split}: malformed output "
                      f"{stats['malformed_pct']}% -- quote this in the paper")

    # The parse is a *pre*-processing step, deliberately: the LM ran once, offline, and
    # wrote token lists to disk. Training reads those JSON files through lib/dataset.py
    # and never constructs a language model, so no LM weights compete for GPU memory
    # with the fusion net and the run is reproducible from the cache alone.
    print(f"[ablation] offline: training reads {PARSING_FOLDER}/ only; "
          f"the LM is not loaded during training or evaluation")
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
    if not check_parse_cache():
        return 1
    cmd = build_command()
    print("[ablation] parser variant E (small local LM)")
    print("[ablation] " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, cwd=REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
