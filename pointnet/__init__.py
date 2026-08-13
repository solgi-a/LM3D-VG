
import importlib
import os
import sys

IMPLEMENTATIONS = (
    ("pointnet2_ops", "pointnet2_ops.pointnet2_modules",
     "CUDA extension"),
    ("pointnet2_py_adv", "pointnet.pointnet2_py_adv.pn_py_torch",
     "optimized pure PyTorch"),
    ("pointnet2_python", "pointnet.pointnet2_python.pointnet2_modules",
     "original pure-Python reference"),
)

ALIASES = {
    "pointnet_py_adv": "pointnet2_py_adv",
    "pn": "pointnet2_py_adv",
    "pn_py_torch": "pointnet2_py_adv",
    "cuda": "pointnet2_ops",
    "ops": "pointnet2_ops",
    "python": "pointnet2_python",
}

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_path():
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


def resolve(preferred=None, names=None):
    _ensure_path()

    if preferred:
        wanted = ALIASES.get(preferred, preferred)
        for tier, path, note in IMPLEMENTATIONS:
            if tier != wanted:
                continue
            try:
                return importlib.import_module(path), tier
            except ImportError as error:
                raise ImportError(
                    f"ABLATION.POINTNET_IMPL pins {preferred!r} ({note}), which "
                    f"cannot be imported: {error}. Either install it or choose "
                    f"another of {[t for t, _, _ in IMPLEMENTATIONS]}."
                ) from error
        raise ImportError(
            f"unknown PointNet++ implementation {preferred!r}. "
            f"Choose one of {[t for t, _, _ in IMPLEMENTATIONS]} "
            f"(aliases: {sorted(ALIASES)})."
        )

    failures = []
    for tier, path, note in IMPLEMENTATIONS:
        try:
            return importlib.import_module(path), tier
        except ImportError as error:
            failures.append(f"  {tier:18s} ({note}): {error}")

    raise ImportError(
        "no PointNet++ implementation could be imported:\n" + "\n".join(failures))


def load(*names, preferred=None):
    module, tier = resolve(preferred)
    missing = [n for n in names if not hasattr(module, n)]
    if missing:
        raise ImportError(
            f"PointNet++ implementation {tier!r} does not provide {missing}.")
    return tuple(getattr(module, n) for n in names), tier


def available():
    _ensure_path()
    rows = []
    for tier, path, note in IMPLEMENTATIONS:
        try:
            importlib.import_module(path)
            rows.append((tier, note, True, "ok"))
        except ImportError as error:
            rows.append((tier, note, False, str(error).splitlines()[0][:110]))
    return rows
