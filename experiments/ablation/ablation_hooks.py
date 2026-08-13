
import importlib.util
import os
import sys

from experiments.ablation.ablation_config import ABLATION
from lib.dataset import ScannetReferenceDataset

_augment_notice_shown = False
_fusion_variant_installed = None
_current_match_module = None

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ORIGINAL_MATCH_MODULE = os.path.join(_REPO_ROOT, "models", "match_module original.py")

_MATCH_MODULE_HOLDERS = (
    "models.match_module",
    "models.refnet",
    "experiments.ablation.cached_refnet",
)


def _cached_or_deterministic():
    return bool(ABLATION.USE_CACHED_SCENES or ABLATION.DETERMINISTIC_SUBSAMPLE)


def build_dataset(**kwargs):
    global _augment_notice_shown

    from experiments.ablation.cached_scenes import CachedSceneDataset

    if not _cached_or_deterministic():
        if not ABLATION.LAZY_LANG_DATA:
            return ScannetReferenceDataset(**kwargs)
        return CachedSceneDataset(
            use_cache=False, cached_scenes_root=None, deterministic=False,
            lazy_lang_data=True, **kwargs,
        )

    if ABLATION.USE_CACHED_SCENES and kwargs.get("augment", False):
        kwargs["augment"] = False
        if not _augment_notice_shown:
            _augment_notice_shown = True
            print(
                "\n[ABLATION] use_cached_scenes=True -> forcing augment=False.\n"
                "           The scene cache is augmentation-free, so geometric\n"
                "           augmentation (flip/rotate/scale/translate) is disabled and\n"
                "           the detector is frozen. Cached runs are comparable to each\n"
                "           other, NOT to the paper's end-to-end main table.\n"
            )

    return CachedSceneDataset(
        use_cache=bool(ABLATION.USE_CACHED_SCENES),
        cached_scenes_root=ABLATION.CACHED_SCENES_ROOT,
        deterministic=bool(ABLATION.DETERMINISTIC_SUBSAMPLE),
        subsample_salt=ABLATION.SUBSAMPLE_SALT,
        lazy_lang_data=bool(ABLATION.LAZY_LANG_DATA),
        **kwargs,
    )


def load_original_match_module():
    if not os.path.isfile(_ORIGINAL_MATCH_MODULE):
        raise FileNotFoundError(
            f"FUSION_VARIANT='original' needs {_ORIGINAL_MATCH_MODULE!r}, which is not "
            f"there. Either restore that file or set FUSION_VARIANT='current'.")
    spec = importlib.util.spec_from_file_location(
        "experiments.ablation._match_module_original", _ORIGINAL_MATCH_MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MatchModule


def _bind_match_module(cls):
    patched = []
    for name in _MATCH_MODULE_HOLDERS:
        module = sys.modules.get(name)
        if module is not None and hasattr(module, "MatchModule"):
            module.MatchModule = cls
            patched.append(name)
    return patched


def install_fusion_variant():
    global _fusion_variant_installed, _current_match_module
    variant = ABLATION.FUSION_VARIANT
    if variant not in ("current", "original"):
        raise ValueError(f"unknown FUSION_VARIANT {variant!r}")
    if _fusion_variant_installed == variant:
        return

    import models.match_module

    if _current_match_module is None:
        _current_match_module = models.match_module.MatchModule

    if variant == "current":
        if _fusion_variant_installed is not None:
            _bind_match_module(_current_match_module)
            print("\n[ABLATION] FUSION_VARIANT='current' -> restored "
                  "models/match_module.py\n")
        _fusion_variant_installed = variant
        return

    patched = _bind_match_module(load_original_match_module())
    _fusion_variant_installed = variant
    print(
        "\n[ABLATION] FUSION_VARIANT='original' -> building MatchModule from\n"
        f"           'models/match_module original.py' instead of models/match_module.py.\n"
        f"           Patched: {', '.join(patched)}\n"
        "           That older head is the one the 2024-12-18 checkpoint was trained\n"
        "           with; the current file has an extra post-graph attention stage and\n"
        "           would leave 178 tensors randomly initialised.\n"
    )


def _check_loaded(model, state_dict, report):
    incoming_prefixes = {key.split(".")[0] for key in state_dict}
    own_prefixes = {key.split(".")[0] for key in model.state_dict()}
    shared = incoming_prefixes & own_prefixes

    missing = [k for k in report.missing_keys if k.split(".")[0] in shared]
    if not missing:
        return missing

    affected = sorted({".".join(k.split(".")[:2]) for k in missing})
    message = (
        f"checkpoint does not fit this model: {len(missing)} parameter(s) in "
        f"{sorted(shared)} have no value in the file and would stay randomly "
        f"initialised.\n"
        f"  affected submodules : {', '.join(affected[:12])}"
        f"{' ...' if len(affected) > 12 else ''}\n"
        f"  first missing keys  : {missing[:3]}\n"
        f"\n"
        f"  If this is the 2024-12-18 checkpoint, it was trained with the older fusion\n"
        f"  head. Re-run with --fusion_variant original (or set\n"
        f"  ABLATION.FUSION_VARIANT = 'original').\n"
        f"\n"
        f"  To proceed anyway and accept randomly initialised weights, pass\n"
        f"  --no_strict_checkpoint. The resulting accuracy will not mean anything."
    )
    if ABLATION.STRICT_CHECKPOINT:
        raise RuntimeError(message)
    print(f"\n[ABLATION] WARNING -- {message}\n")
    return missing


def _enforce_strict_checkpoint(model):
    inner = model.load_state_dict

    def load_state_dict(state_dict, strict=False, **kwargs):
        report = inner(state_dict, strict=False, **kwargs)
        missing = _check_loaded(model, state_dict, report)
        kept = len(set(state_dict) & set(model.state_dict()))
        unused = len(report.unexpected_keys)
        print(f"[ABLATION] checkpoint: {kept} tensor(s) loaded, "
              f"{len(missing)} left at random init"
              + (f", {unused} key(s) in the file unused by this model" if unused else ""))
        return report

    model.load_state_dict = load_state_dict
    return model


def build_model(**kwargs):
    install_fusion_variant()

    if not ABLATION.USE_CACHED_SCENES:
        from models.refnet import RefNet

        model = RefNet(**kwargs)
    else:
        from experiments.ablation.cached_refnet import CachedRefNet

        model = CachedRefNet(**kwargs)

    return _enforce_strict_checkpoint(model)


def skip_pretrained_detector():
    return bool(ABLATION.USE_CACHED_SCENES)
