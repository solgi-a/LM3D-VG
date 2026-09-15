"""
Switchboard for the ablation experiments. Design notes: details.md §2.

Every flag is off by default; with all of them off the hooks in ScanRefer_train.py /
ScanRefer_eval.py / predict.py / visualize.py fall through to ScannetReferenceDataset and
RefNet. Set a flag here or override it from the CLI; no other file needs editing.

    ABLATION.USE_CACHED_SCENES = True                      # train on cached detector output
    ABLATION.PARSING_FOLDER = "spacy_parsing_tokenized"    # parser ablation, variant B
    ABLATION.DISABLE_COPY_PASTE = True                     # copy-paste ablation
"""

import os


class _Config(dict):
    """dict with attribute access, so flags read as ABLATION.USE_CACHED_SCENES."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    __setattr__ = dict.__setitem__


ABLATION = _Config()

# Read the detection branch's output from disk instead of running it. Build the cache
# first with experiments/ablation/cache.py build.
ABLATION.USE_CACHED_SCENES = False
ABLATION.CACHED_SCENES_ROOT = "cached_scenes"

# Seed the 40k-point subsample from the scene id. Implied by USE_CACHED_SCENES; set it
# alone for an end-to-end run that must reproduce the cache. The salt must match meta.json.
ABLATION.DETERMINISTIC_SUBSAMPLE = False
ABLATION.SUBSAMPLE_SALT = "v1"

# Host-RAM control, not an ablation: eager precomputation needs ~36 GB and is SIGKILLed on
# smaller machines, while the lazy path gives identical tensors in ~1.8 GB.
ABLATION.LAZY_LANG_DATA = True

# Which parse cache lib/dataset.py reads, relative to data/scannet/. The schema is the
# same for every variant, so the language module and fusion net are untouched between arms.
ABLATION.PARSING_FOLDER = "final_parsing_tokenized"

# True disables the two copy-paste blocks in models/match_module.py.
ABLATION.DISABLE_COPY_PASTE = False

# "current" is models/match_module.py; "original" is the frozen earlier variant the
# 2024-12-18 checkpoint was trained with. ablation_hooks swaps the class at build time.
ABLATION.FUSION_VARIANT = "current"

# Refuse a checkpoint that would leave part of the model randomly initialised. Eval loads
# with strict=False, so a half-loaded fusion head is otherwise silent.
ABLATION.STRICT_CHECKPOINT = True

# None walks pointnet/__init__.py's three-tier chain; a tier name pins it and raises if it
# cannot be imported. Read at model import time, so there is no CLI flag.
ABLATION.POINTNET_IMPL = None

_DEFAULT_PARSING_FOLDER = "final_parsing_tokenized"

#: ABLATION name -> argparse destination
_CLI_FLAGS = {
    "USE_CACHED_SCENES": "use_cached_scenes",
    "CACHED_SCENES_ROOT": "cached_scenes_root",
    "DETERMINISTIC_SUBSAMPLE": "deterministic_subsample",
    "SUBSAMPLE_SALT": "subsample_salt",
    "PARSING_FOLDER": "parsing_folder",
    "DISABLE_COPY_PASTE": "disable_copy_paste",
    "FUSION_VARIANT": "fusion_variant",
}

FUSION_VARIANTS = ("current", "original")


def enabled():
    """True when any ablation is active."""
    return bool(
        ABLATION.USE_CACHED_SCENES
        or ABLATION.DETERMINISTIC_SUBSAMPLE
        or ABLATION.DISABLE_COPY_PASTE
        or ABLATION.PARSING_FOLDER != _DEFAULT_PARSING_FOLDER
        or ABLATION.FUSION_VARIANT != "current"
    )


def add_arguments(parser):
    """Optional CLI overrides, so a sweep does not need to edit this file per run."""
    group = parser.add_argument_group("ablations (experiments/ablation/ablation_config.py)")
    group.add_argument("--use_cached_scenes", action="store_true", default=None,
                       help="Train/eval on cached detector output (see experiments/ablation/cache.py).")
    group.add_argument("--cached_scenes_root", type=str, default=None,
                       help="Directory holding the scene cache (train/, val/, meta.json).")
    group.add_argument("--deterministic_subsample", action="store_true", default=None,
                       help="Pin the 40k-point subsample per scene. Implied by --use_cached_scenes.")
    group.add_argument("--subsample_salt", type=str, default=None,
                       help="Salt for the deterministic subsample seed; must match meta.json.")
    group.add_argument("--parsing_folder", type=str, default=None,
                       help="Parse cache under data/scannet/ (parser ablation).")
    group.add_argument("--disable_copy_paste", action="store_true", default=None,
                       help="Turn off the proposal copy-paste augmentation in MatchModule.")
    group.add_argument("--keep_checkpoint", action="store_true", default=False,
                       help="Keep --use_checkpoint in cached mode (it is cleared by "
                            "default so ablation runs start from a fresh fusion net).")
    group.add_argument("--fusion_variant", type=str, default=None,
                       choices=list(FUSION_VARIANTS),
                       help="Which MatchModule to build. 'original' uses "
                            "'models/match_module original.py', which is the architecture "
                            "the 2024-12-18 checkpoint was trained with.")
    group.add_argument("--no_strict_checkpoint", action="store_true", default=False,
                       help="Allow a checkpoint to leave part of the model randomly "
                            "initialised. Off by default: half-loaded weights produce a "
                            "plausible-looking but meaningless accuracy.")
    group.add_argument("--no_lazy_lang_data", action="store_true", default=False,
                       help="Precompute every annotation's GloVe embeddings up front, as "
                            "lib/dataset.py does. Needs ~36 GB of RAM for train+val; the "
                            "lazy default produces identical tensors in ~1.8 GB.")
    return parser


def apply(args):
    """Fold CLI overrides into ABLATION and mirror them back onto ``args``."""
    for key, dest in _CLI_FLAGS.items():
        value = getattr(args, dest, None)
        # store_true defaults to None, so "not passed" differs from "passed as false".
        if value is not None and value is not False:
            ABLATION[key] = value

    # These two default to True, so their switches are the negative form.
    if getattr(args, "no_strict_checkpoint", False):
        ABLATION.STRICT_CHECKPOINT = False

    if getattr(args, "no_lazy_lang_data", False):
        ABLATION.LAZY_LANG_DATA = False

    if ABLATION.FUSION_VARIANT not in FUSION_VARIANTS:
        raise ValueError(
            f"FUSION_VARIANT must be one of {FUSION_VARIANTS}, "
            f"got {ABLATION.FUSION_VARIANT!r}")

    if ABLATION.USE_CACHED_SCENES:
        # No detection branch to train or supervise.
        args.no_detection = True
        args.detection = False

        # comp_weight() would copy the whole model here, so every seed would share one
        # initialisation. The warm-starting arms opt back in with --keep_checkpoint.
        if getattr(args, "use_checkpoint", "") and not getattr(args, "keep_checkpoint", False):
            print(
                f"\n[ABLATION] ignoring --use_checkpoint {args.use_checkpoint!r}: in cached "
                f"mode it would preload the fusion weights and every run would share one\n"
                f"           initialisation. Pass --keep_checkpoint to override.\n"
            )
            args.use_checkpoint = ""

    for key, dest in _CLI_FLAGS.items():
        setattr(args, dest, ABLATION[key])

    return args


def describe():
    lines = ["ablation config (experiments/ablation/ablation_config.py):"]
    for key in sorted(ABLATION):
        lines.append(f"  {key} = {ABLATION[key]!r}")
    if not enabled():
        lines.append("  -> all experiments OFF; behaviour identical to the original code")
    return "\n".join(lines)
