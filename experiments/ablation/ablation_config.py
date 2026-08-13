"""
Switchboard for the revision experiments. See experiments/experiments.md §1.

Every flag is off by default; with all of them off the hooks in ScanRefer_train.py /
ScanRefer_eval.py / predict.py / visualize.py fall through to ScannetReferenceDataset and
RefNet. Set a flag here or override it from the CLI (see `apply()`); no other file needs
editing.

    ABLATION.USE_CACHED_SCENES = True                      # train on cached detector output
    ABLATION.PARSING_FOLDER = "spacy_parsing_tokenized"    # parser ablation, variant B
    ABLATION.DISABLE_COPY_PASTE = True                     # copy-paste ablation
"""

import os


class _Config(dict):
    """dict with attribute access, so flags read as ABLATION.USE_CACHED_SCENES.

    Not easydict -- keeping this module dependency-free lets the standalone scripts
    inspect the config without the training environment installed.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    __setattr__ = dict.__setitem__


ABLATION = _Config()

# ======================================================================================
# Experiment 1 -- cached scene (detector) features
# ======================================================================================
# Skip PointNet++/VoteNet/DETR entirely and read their per-scene output from disk.
# Only valid for runs that change language input and/or fusion; the detector is frozen
# by construction. Build the cache first with experiments/ablation/scenes_cache.py.
ABLATION.USE_CACHED_SCENES = False

# Where the cache lives (train/, val/, meta.json).
ABLATION.CACHED_SCENES_ROOT = "cached_scenes"

# Seed the 40k-point subsample from the scene id instead of global numpy state. Implied
# by USE_CACHED_SCENES; set it alone for an end-to-end run that must reproduce the cache.
ABLATION.DETERMINISTIC_SUBSAMPLE = False

# Salt for that seed. Must match the value recorded in the cache's meta.json.
ABLATION.SUBSAMPLE_SALT = "v1"

# Host-RAM control, not an ablation -- it changes when the GloVe matrices are built, not
# what they contain. Eager precomputation needs ~36 GB for train+val and is SIGKILLed
# (exit 247) on smaller machines; the lazy LRU path gives identical tensors in ~1.8 GB.
ABLATION.LAZY_LANG_DATA = True


# ======================================================================================
# Experiment 2 -- parser ablation
# ======================================================================================
# Which parse cache lib/dataset.py reads, relative to data/scannet/.
#   "final_parsing_tokenized"          -> GPT-4o-mini (paper's main configuration)
#   "spacy_parsing_tokenized"          -> rule-based spaCy parser (variant B)
#   "llama_parsing_tokenized_clipped"  -> LLaMA, token caps enforced (variant C)
# The schema is identical in every case, so the language module and fusion net are
# untouched between arms.
ABLATION.PARSING_FOLDER = "final_parsing_tokenized"


# ======================================================================================
# Experiment 3 -- proposal copy-paste augmentation
# ======================================================================================
# True disables the two copy-paste blocks in models/match_module.py.
ABLATION.DISABLE_COPY_PASTE = False


# ======================================================================================
# Experiment 4 -- which fusion head to build
# ======================================================================================
# "current"  -> models/match_module.py, the architecture described in the paper
# "original" -> "models/match_module original.py", the frozen earlier variant
#
# The 2024-12-18 checkpoint was trained against the older head; loading it into the
# current module leaves 178 tensors randomly initialised, and eval loads with
# strict=False, so it happens silently. Use "original" for that checkpoint --
# ablation_hooks swaps the class at build time, models/match_module.py is never edited.
# Details and the verification numbers: experiments/experiments.md §6.3.
ABLATION.FUSION_VARIANT = "current"

# Refuse a checkpoint that would leave part of the model randomly initialised. For each
# top-level submodule present in both the model and the checkpoint, every parameter must
# be supplied. A detector-only file in a full RefNet is fine (lang./match. are absent
# entirely, so unchecked); a half-loading fusion head is an error.
ABLATION.STRICT_CHECKPOINT = True


# ======================================================================================
# Performance -- which PointNet++ implementation to use
# ======================================================================================
# pointnet/__init__.py walks a three-tier chain and takes the first that imports:
#
#   1. "pointnet2_ops"      pointnet/pointnet2_ops_lib/ -- C++/CUDA extension, fastest
#   2. "pointnet2_py_adv"   pointnet/pointnet2_py_adv/pn_py_torch.py -- optimized torch
#   3. "pointnet2_python"   pointnet/pointnet2_python/ -- original Python reference
#
# None walks the chain; a tier name pins that tier for an A/B comparison and raises if it
# cannot be imported. Read at model import time, so there is no CLI flag for it.
ABLATION.POINTNET_IMPL = None


# ======================================================================================

_DEFAULT_PARSING_FOLDER = "final_parsing_tokenized"

#: name in ABLATION -> argparse destination
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
    """True when any revision experiment is active."""
    return bool(
        ABLATION.USE_CACHED_SCENES
        or ABLATION.DETERMINISTIC_SUBSAMPLE
        or ABLATION.DISABLE_COPY_PASTE
        or ABLATION.PARSING_FOLDER != _DEFAULT_PARSING_FOLDER
        or ABLATION.FUSION_VARIANT != "current"
    )


def add_arguments(parser):
    """Optional CLI overrides, so a sweep does not need to edit this file per run."""
    group = parser.add_argument_group("revision ablations (experiments/ablation/ablation_config.py)")
    group.add_argument("--use_cached_scenes", action="store_true", default=None,
                       help="Train/eval on cached detector output (see experiments/ablation/scenes_cache.py).")
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
    """Fold CLI overrides into ABLATION, then propagate them onto ``args``, which is
    returned. Called after the `debug` block so values set there still win.
    """
    for key, dest in _CLI_FLAGS.items():
        value = getattr(args, dest, None)
        # store_true flags default to None, so "not passed" is distinguishable from False.
        if value is not None and value is not False:
            ABLATION[key] = value

    # These two default to True, so their CLI switches are the negative form and the
    # generic rule above cannot express them.
    if getattr(args, "no_strict_checkpoint", False):
        ABLATION.STRICT_CHECKPOINT = False

    if getattr(args, "no_lazy_lang_data", False):
        ABLATION.LAZY_LANG_DATA = False

    if ABLATION.FUSION_VARIANT not in FUSION_VARIANTS:
        raise ValueError(
            f"FUSION_VARIANT must be one of {FUSION_VARIANTS}, "
            f"got {ABLATION.FUSION_VARIANT!r}")

    if ABLATION.USE_CACHED_SCENES:
        # No detection branch to train or supervise; the cached tensors are constants.
        args.no_detection = True
        args.detection = False

        # comp_weight() copies every shape-matching key from --use_checkpoint, which
        # under CachedRefNet is the whole model -- every "different seed" would then share
        # one initialisation and the seed ablation would measure nothing. Cleared by
        # default; the warm-starting runners opt back in with --keep_checkpoint (§2.6).
        if getattr(args, "use_checkpoint", "") and not getattr(args, "keep_checkpoint", False):
            print(
                f"\n[ABLATION] ignoring --use_checkpoint {args.use_checkpoint!r}: in cached "
                f"mode it would preload the fusion weights and every run would share one\n"
                f"           initialisation. Pass --keep_checkpoint to override.\n"
            )
            args.use_checkpoint = ""

    # Mirror the resolved values back onto args so downstream code can read either.
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
