"""
Every training and evaluation arm of the ablation, driven from one config table.

    python experiments/ablation/train_arms.py list
    python experiments/ablation/train_arms.py run parser-gpt
    python experiments/ablation/train_arms.py eval-swap
    python experiments/ablation/train_arms.py corruption-sweep

All arms share COMMON and differ only in their row of ARMS, so a gap between two arms is
attributable to the one thing that row changes. Every COMMON entry can be overridden from
the environment as ABL_<NAME>, which is how the notebooks set it. See details.md §2, §3.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)


def _env(name, default):
    """COMMON value, overridable as ABL_<NAME>. Lists are comma-separated."""
    raw = os.environ.get(f"ABL_{name}")
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, list):
        return [x for x in (p.strip() for p in raw.split(",")) if x]
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


COMMON = dict(
    USE_CACHED_SCENES=_env("USE_CACHED_SCENES", True),
    CACHED_SCENES_ROOT=_env("CACHED_SCENES_ROOT", "cached_scenes"),
    SEED=_env("SEED", 42),
    EPOCH=_env("EPOCH", 20),
    BATCH_SIZE=_env("BATCH_SIZE", 16),
    EXTRA_ARGS=_env("EXTRA_ARGS", ["--use_color", "--use_normal"]),
    # Fine-tune from a trained run rather than random init. With the detector frozen the
    # checkpoint covers all 146 fusion tensors, so an arm converges in far fewer epochs;
    # every arm starts from the same one.
    WARM_START=_env("WARM_START", True),
    WARM_START_FROM=_env("WARM_START_FROM", "2024-12-18_20-40-38_3DVG-FIXED"),
    FUSION_VARIANT=_env("FUSION_VARIANT", "original"),
    VAL_STEP=_env("VAL_STEP", 50),
    VERBOSE=_env("VERBOSE", 50),
    LR=_env("LR", 0.002),
    COSLR=_env("COSLR", True),
    LANG_NUM_MAX=_env("LANG_NUM_MAX", 32),
    NUM_WORKERS=_env("NUM_WORKERS", 4),
    PREFETCH_FACTOR=_env("PREFETCH_FACTOR", 1),
    # False only takes effect via --no_lazy_lang_data; True is the ~1.8 GB path.
    LAZY_LANG_DATA=_env("LAZY_LANG_DATA", True),
)

_SPLITS = ("train", "val")

ARMS = {
    "parser-gpt": dict(
        TAG="ABL-PARSER-GPT", PARSING_FOLDER="final_parsing_tokenized",
        label="variant A (GPT-4o-mini)",
        note="the reference arm; its cache ships with the repo"),
    "parser-spacy": dict(
        TAG="ABL-PARSER-SPACY", PARSING_FOLDER="spacy_parsing_tokenized",
        label="variant B (rule-based spaCy)",
        needs_cache="no spaCy parse cache",
        fix="python experiments/ablation/parsing.py spacy --splits train val"),
    "parser-llama": dict(
        TAG="ABL-PARSER-LLAMA", PARSING_FOLDER="llama_parsing_tokenized_clipped",
        label="variant C (LLaMA, token caps enforced)",
        needs_cache="no clipped LLaMA parse cache",
        why="the raw LLaMA cache exceeds the 7/17/75 caps and would crash "
            "pack_padded_sequence mid-training",
        fix="python experiments/ablation/parsing.py clip "
            "--input data/scannet/final_parsing_tokenized_llama "
            "--output data_parsing/llama_parsing_tokenized_clipped"),
    "parser-none": dict(
        TAG="ABL-PARSER-NONE", PARSING_FOLDER="noparse_tokenized",
        label="variant D (no parser, fields = ['unk'])",
        needs_cache="no no-parse cache",
        fix="python experiments/ablation/parsing.py noparse --splits train val"),
    "parser-smalllm": dict(
        TAG="ABL-PARSER-SMALLLM", PARSING_FOLDER="smalllm_parsing_tokenized",
        label="variant E (small local LM)",
        needs_cache="no small-LM parse cache",
        fix="python experiments/ablation/parsing_smalllm.py --splits train val --device cuda",
        report_summary=True),
    "no-copypaste": dict(
        TAG="ABL-NO-COPYPASTE", PARSING_FOLDER="final_parsing_tokenized",
        label="proposal copy-paste DISABLED",
        extra_flags=["--disable_copy_paste"],
        note="compare against parser-gpt, which is the same settings with it on"),
    "seeds": dict(
        TAG_PREFIX="ABL-SEED", PARSING_FOLDER="final_parsing_tokenized",
        label="the main configuration under several seeds",
        SEEDS=[1, 2], STOP_ON_FAILURE=True),
}


def config_for(arm):
    cfg = dict(COMMON)
    cfg.update(ARMS[arm])
    return cfg


def build_command(cfg, seed=None, tag=None):
    """The scripts/ScanRefer_train.py argv for one arm."""
    cmd = [sys.executable, os.path.join("scripts", "ScanRefer_train.py"),
           "--tag", tag if tag is not None else cfg["TAG"],
           "--seed", str(cfg["SEED"] if seed is None else seed),
           "--epoch", str(cfg["EPOCH"]),
           "--batch_size", str(cfg["BATCH_SIZE"]),
           "--parsing_folder", cfg["PARSING_FOLDER"]]
    cmd += list(cfg.get("extra_flags", []))
    if cfg["USE_CACHED_SCENES"]:
        cmd += ["--use_cached_scenes", "--cached_scenes_root", cfg["CACHED_SCENES_ROOT"]]
    if cfg["WARM_START"]:
        # ablation_config.apply() clears --use_checkpoint in cached mode; --keep_checkpoint
        # opts back in. Without it the run silently trains from random init.
        cmd += ["--use_checkpoint", cfg["WARM_START_FROM"], "--keep_checkpoint",
                "--fusion_variant", cfg["FUSION_VARIANT"]]
    else:
        cmd += ["--no_warm_start"]
    cmd += ["--val_step", str(cfg["VAL_STEP"]),
            "--verbose", str(cfg["VERBOSE"]),
            "--lr", str(cfg["LR"]),
            "--lang_num_max", str(cfg["LANG_NUM_MAX"])]
    if cfg["COSLR"]:
        cmd += ["--coslr"]
    if cfg["NUM_WORKERS"] is not None:
        cmd += ["--num_workers", str(cfg["NUM_WORKERS"])]
    cmd += ["--prefetch_factor", str(cfg["PREFETCH_FACTOR"])]
    if not cfg["LAZY_LANG_DATA"]:
        cmd += ["--no_lazy_lang_data"]
    return cmd + list(cfg["EXTRA_ARGS"])


def check_parse_cache(cfg):
    """True when the arm's parse cache is on disk; otherwise print how to build it."""
    folder = os.path.join(REPO_ROOT, "data_parsing", cfg["PARSING_FOLDER"])
    missing = [s for s in _SPLITS
               if not os.path.isfile(os.path.join(folder, f"tokenized_parsed_result_{s}.json"))]
    if missing:
        print(f"ERROR: {cfg.get('needs_cache', 'no parse cache')} for split(s) {missing} "
              f"in {folder}")
        if cfg.get("why"):
            print(cfg["why"] + ".")
        print("Generate it first:")
        print(f"    {cfg['fix']}")
        return False

    if cfg.get("report_summary"):
        summary = os.path.join(folder, "parse_run_summary.json")
        if os.path.isfile(summary):
            with open(summary) as f:
                data = json.load(f)
            print(f"[ablation] parse cache from {data.get('model')} ({data.get('kind')})")
            for split, stats in (data.get("splits") or {}).items():
                if "malformed_pct" in stats:
                    print(f"[ablation]   {split}: malformed output {stats['malformed_pct']}%")

    # Parsing is a pre-processing step: the parser ran once, offline, and wrote token lists
    # to disk. No language model is constructed during training or evaluation.
    print(f"[ablation] offline: training reads {cfg['PARSING_FOLDER']}/ only")
    return True


def run_arm(args):
    arm = args.arm
    cfg = config_for(arm)
    print(f"[ablation] {arm}: {cfg['label']}")
    if cfg.get("note"):
        print(f"[ablation] {cfg['note']}")
    if cfg.get("fix") and not check_parse_cache(cfg):
        return 1

    if "SEEDS" in cfg:
        seeds = args.seeds if args.seeds else cfg["SEEDS"]
        failures = []
        for seed in seeds:
            cmd = build_command(cfg, seed=seed, tag=f"{cfg['TAG_PREFIX']}{seed}")
            print(f"\n[ablation] seed {seed} of {seeds}")
            print("[ablation] " + " ".join(cmd) + "\n")
            code = subprocess.call(cmd, cwd=REPO_ROOT)
            if code != 0:
                failures.append((seed, code))
                print(f"[ablation] seed {seed} exited with code {code}")
                if cfg.get("STOP_ON_FAILURE"):
                    break
        if failures:
            print("\n[ablation] failed seeds: "
                  + ", ".join(f"{s} (code {c})" for s, c in failures))
            return 1
        print(f"\n[ablation] all seeds finished: {seeds}")
        print(f"[ablation] aggregate with: python experiments/analysis.py seed-aggregate "
              f"--pattern '*{cfg['TAG_PREFIX']}*'")
        return 0

    cmd = build_command(cfg)
    print("[ablation] " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def run_list(args):
    width = max(len(k) for k in ARMS)
    print(f"{len(ARMS)} arms:\n")
    for key, row in ARMS.items():
        tag = row.get("TAG") or (row.get("TAG_PREFIX", "") + "*")
        print(f"  {key:<{width}}  {tag:<22} {row['label']}")
    print("\nshared config (override any of these as ABL_<NAME>):")
    for k, v in COMMON.items():
        print(f"  {k:<20} = {v!r}")
    return 0


_DOC_EVAL_SWAP = """
Evaluation-only parser ablation: swap the parser at test time on a trained checkpoint.

    RUNS ON: GPU. Evaluation only -- no training. One val pass per parser.

    python experiments/ablation/train_arms.py eval-swap

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

SWAP_FOLDER = "2024-12-18_20-40-38_3DVG-FIXED"      # trained run under outputs/ to evaluate
#: Which MatchModule to build. SWAP_FOLDER above was trained with the older fusion
#: head, so "original" is required -- with "current" the eval would leave 178
#: tensors randomly initialised and ablation_hooks now refuses outright.
SWAP_FUSION_VARIANT = "original"

#: (label, parse folder under data_parsing/). The FIRST is the training-time parse.
SWAP_PARSERS = [
    ("gpt4o-mini", "final_parsing_tokenized"),          # matched: what it was trained on
    ("spacy", "spacy_parsing_tokenized"),
    ("llama", "llama_parsing_tokenized_clipped"),
    ("none", "noparse_tokenized"),
    ("smalllm", "smalllm_parsing_tokenized"),
]

SWAP_BATCH_SIZE = 32
SWAP_LANG_NUM_MAX = 1
SWAP_EXTRA_ARGS = ["--use_color", "--use_normal"]   # must match the trained configuration
SWAP_SKIP_MISSING = True                            # skip a parser whose cache is absent

SWAP_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", SWAP_FOLDER)
SWAP_ARCHIVE_ROOT = os.path.join(SWAP_OUTPUT_DIR, "parser_swap")
SWAP_ARTIFACTS = ("predictions.p", "scores.p")

# ScanRefer_eval.py prints one line per (subset, split, metric), e.g.
#   overall | overall | acc@0.25iou: 0.4643458140513252
# The overall/overall row is the one to record; the rest are subset breakdowns.
_SWAP_IOU_LINE = re.compile(
    r"overall\s*\|\s*overall\s*\|\s*acc@0\.25iou:\s*([0-9.]+)", re.IGNORECASE)
_SWAP_IOU_LINE_50 = re.compile(
    r"overall\s*\|\s*overall\s*\|\s*acc@0\.5iou:\s*([0-9.]+)", re.IGNORECASE)


def _swap_has_cache(parsing_folder):
    return os.path.isfile(os.path.join(REPO_ROOT, "data_parsing", parsing_folder,
                                       "tokenized_parsed_result_val.json"))


def _swap_run_one(label, parsing_folder):
    cmd = [sys.executable, os.path.join("scripts", "ScanRefer_eval.py"),
           "--folder", SWAP_FOLDER, "--reference", "--force",
           "--batch_size", str(SWAP_BATCH_SIZE),
           "--lang_num_max", str(SWAP_LANG_NUM_MAX),
           "--parsing_folder", parsing_folder,
           "--fusion_variant", SWAP_FUSION_VARIANT] + SWAP_EXTRA_ARGS

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

    destination = os.path.join(SWAP_ARCHIVE_ROOT, label)
    os.makedirs(destination, exist_ok=True)
    moved = []
    for artifact in SWAP_ARTIFACTS:
        source = os.path.join(SWAP_OUTPUT_DIR, artifact)
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

    accuracy = _scrape(_SWAP_IOU_LINE)
    accuracy_50 = _scrape(_SWAP_IOU_LINE_50)
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




def run_eval_swap(args):

    if not os.path.isdir(SWAP_OUTPUT_DIR):
        print(f"ERROR: trained run not found: {SWAP_OUTPUT_DIR}")
        return 1

    selected = []
    for label, parsing_folder in SWAP_PARSERS:
        if _swap_has_cache(parsing_folder):
            selected.append((label, parsing_folder))
        elif SWAP_SKIP_MISSING:
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
          f"{SWAP_PARSERS[0][1]!r}.")
    print("Results for the other parsers confound parse quality with train/test")
    print("distribution mismatch, and are NOT a fair parser comparison. State that")
    print("limitation explicitly wherever these numbers appear.")
    print("!" * 78)

    os.makedirs(SWAP_ARCHIVE_ROOT, exist_ok=True)
    backups = []
    for artifact in SWAP_ARTIFACTS:
        source = os.path.join(SWAP_OUTPUT_DIR, artifact)
        if os.path.isfile(source):
            backup = source + ".bak"
            shutil.copy(source, backup)
            backups.append((source, backup))

    results, failures = [], []
    for label, parsing_folder in selected:
        result = _swap_run_one(label, parsing_folder)
        results.append(result)
        if result["exit_code"] != 0:
            failures.append(label)

    summary_path = os.path.join(SWAP_ARCHIVE_ROOT, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"folder": SWAP_FOLDER, "trained_on": SWAP_PARSERS[0][1],
                   "caveat": "evaluation-only swap; confounds parse quality with "
                             "train/test distribution mismatch",
                   "results": results}, f, indent=2)

    for source, backup in backups:
        shutil.move(backup, source)

    print("\n[swap] summary:")
    reference = next((r for r in results if r["parser"] == SWAP_PARSERS[0][0]), None)
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


_DOC_CORRUPTION_SWEEP = """
Parse-corruption sweep: evaluate one trained checkpoint against deliberately broken parses.

    RUNS ON: GPU. Evaluation only -- no training. One val pass per level, so roughly 4x a
             single evaluation (baseline + three corruption levels).

    python experiments/ablation/train_arms.py corruption-sweep

The sample, model and weights are fixed and only the parse changes, so the effect is
causal. The runner:

1. Generates the corrupted parse caches if missing (CPU, seconds).
2. Runs ``ScanRefer_eval.py --force`` once per level, starting with the uncorrupted parse
   for a paired baseline.
3. Archives each level's output immediately -- ``ScanRefer_eval.py`` always writes to the
   same ``predictions.p`` / ``scores.p``, so without this each level would overwrite the
   previous one.

Result layout, which experiments/analysis.py error-propagation expects::

    outputs/<FOLDER>/corruption/
        baseline/{predictions.p, scores.p, eval_stdout.txt}
        all_10/{predictions.p, scores.p, corruption_manifest.json, eval_stdout.txt}
        all_25/...
        all_50/...
        run_summary.json

Any pre-existing ``predictions.p`` is backed up and restored at the end. Then aggregate
on CPU:

    python experiments/analysis.py error-propagation --run-dir outputs/<FOLDER>/corruption
"""

CORR_FOLDER = "2024-12-18_20-40-38_3DVG-FIXED"      # trained run under outputs/ to evaluate
#: CORR_FOLDER was trained with the older fusion head, so "original" is required.
CORR_FUSION_VARIANT = "original"
CORR_SOURCE = "final_parsing_tokenized"     # the clean parse the model was trained on
CORR_RATES = [0.10, 0.25, 0.50]                     # corruption levels
CORR_MODE = "all"                                   # swap | drop | shuffle | all
CORR_SEED = 42
CORR_SPLIT = "val"
CORR_BATCH_SIZE = 8
CORR_LANG_NUM_MAX = 1                               # eval uses one description per sample
CORR_EXTRA_ARGS = ["--use_color", "--use_normal"]   # must match the trained configuration
CORR_RUN_BASELINE = True                            # evaluate the clean parse first (paired)
CORR_GENERATE_CACHES = True                         # build missing corrupted caches

CORR_OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", CORR_FOLDER)
CORR_ARCHIVE_ROOT = os.path.join(CORR_OUTPUT_DIR, "corruption")
CORR_ARTIFACTS = ("predictions.p", "scores.p")


def _corr_folder(rate):
    return f"{CORR_SOURCE}_corrupt_{CORR_MODE}_{int(round(rate * 100)):02d}"


def _corr_ensure_caches():
    missing = [r for r in CORR_RATES
               if not os.path.isfile(os.path.join(
                   REPO_ROOT, "data_parsing", _corr_folder(r),
                   f"tokenized_parsed_result_{CORR_SPLIT}.json"))]
    if not missing:
        return True
    if not CORR_GENERATE_CACHES:
        print(f"ERROR: missing corrupted caches for rates {missing}. Generate them with:")
        print(f"    python experiments/ablation/parsing.py corrupt --splits {CORR_SPLIT} "
              f"--rates {' '.join(str(r) for r in missing)} --mode {CORR_MODE}")
        return False

    cmd = [sys.executable, "experiments/ablation/parsing.py", "corrupt",
           "--splits", CORR_SPLIT, "--source", CORR_SOURCE, "--mode", CORR_MODE,
           "--seed", str(CORR_SEED), "--rates"] + [str(r) for r in missing]
    print("[corruption] generating missing caches")
    print("[corruption] " + " ".join(cmd) + "\n")
    return subprocess.call(cmd, cwd=REPO_ROOT) == 0


def _corr_eval_command(parsing_folder):
    return [sys.executable, os.path.join("scripts", "ScanRefer_eval.py"),
            "--folder", CORR_FOLDER,
            "--reference",
            "--force",
            "--batch_size", str(CORR_BATCH_SIZE),
            "--lang_num_max", str(CORR_LANG_NUM_MAX),
            "--parsing_folder", parsing_folder,
            "--fusion_variant", CORR_FUSION_VARIANT] + CORR_EXTRA_ARGS


def _corr_archive(level_name, parsing_folder):
    """Move this level's eval output out of the way before the next level overwrites it."""
    destination = os.path.join(CORR_ARCHIVE_ROOT, level_name)
    os.makedirs(destination, exist_ok=True)

    moved = []
    for artifact in CORR_ARTIFACTS:
        source = os.path.join(CORR_OUTPUT_DIR, artifact)
        if os.path.isfile(source):
            shutil.move(source, os.path.join(destination, artifact))
            moved.append(artifact)

    manifest = os.path.join(REPO_ROOT, "data_parsing", parsing_folder,
                            "corruption_manifest.json")
    if os.path.isfile(manifest):
        shutil.copy(manifest, os.path.join(destination, "corruption_manifest.json"))

    if not moved:
        print(f"[corruption] WARNING: no {'/'.join(CORR_ARTIFACTS)} produced for {level_name}")
    return destination, moved


def _corr_run_level(level_name, parsing_folder):
    cmd = _corr_eval_command(parsing_folder)
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

    destination, moved = _corr_archive(level_name, parsing_folder)
    with open(os.path.join(destination, "eval_stdout.txt"), "w") as f:
        f.writelines(captured)

    print(f"[corruption] level {level_name} finished in {elapsed / 60:.1f} min "
          f"(exit {code}); archived {moved} -> {destination}")
    return {"level": level_name, "parsing_folder": parsing_folder, "exit_code": code,
            "elapsed_sec": round(elapsed, 1), "archived": moved, "dir": destination}




def run_corruption_sweep(args):

    if not os.path.isdir(CORR_OUTPUT_DIR):
        print(f"ERROR: trained run not found: {CORR_OUTPUT_DIR}")
        print("Set CORR_FOLDER at the top of this file to a directory under outputs/.")
        return 1
    if not _corr_ensure_caches():
        return 1

    os.makedirs(CORR_ARCHIVE_ROOT, exist_ok=True)

    # Preserve any existing evaluation output; this sweep overwrites it repeatedly.
    backups = []
    for artifact in CORR_ARTIFACTS:
        source = os.path.join(CORR_OUTPUT_DIR, artifact)
        if os.path.isfile(source):
            backup = source + ".bak"
            shutil.copy(source, backup)
            backups.append((source, backup))
    if backups:
        print(f"[corruption] backed up {len(backups)} existing artifact(s) "
              f"(restored at the end)")

    levels = []
    if CORR_RUN_BASELINE:
        levels.append(("baseline", CORR_SOURCE))
    levels += [(f"{CORR_MODE}_{int(round(r * 100)):02d}", _corr_folder(r)) for r in CORR_RATES]

    results, failures = [], []
    for level_name, parsing_folder in levels:
        result = _corr_run_level(level_name, parsing_folder)
        results.append(result)
        if result["exit_code"] != 0:
            failures.append(level_name)

    summary_path = os.path.join(CORR_ARCHIVE_ROOT, "run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"folder": CORR_FOLDER, "source_parsing": CORR_SOURCE,
                   "mode": CORR_MODE, "rates": CORR_RATES, "seed": CORR_SEED, "split": CORR_SPLIT,
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
    print(f"    python experiments/analysis.py error-propagation --run-dir "
          f"outputs/{CORR_FOLDER}/corruption")
    return 0



def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python experiments/ablation/train_arms.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    p = sub.add_parser("list", help="print the arms and the shared config")
    p.set_defaults(func=run_list)

    p = sub.add_parser("run", help="train one arm",
                       description=__doc__,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("arm", choices=sorted(ARMS), help="which arm to train")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override the seed list of the 'seeds' arm.")
    p.set_defaults(func=run_arm)

    p = sub.add_parser("eval-swap", help=_DOC_EVAL_SWAP.strip().splitlines()[0],
                       description=_DOC_EVAL_SWAP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(func=run_eval_swap)

    p = sub.add_parser("corruption-sweep", help=_DOC_CORRUPTION_SWEEP.strip().splitlines()[0],
                       description=_DOC_CORRUPTION_SWEEP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.set_defaults(func=run_corruption_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
