"""
Prove the scene cache is equivalent to the end-to-end model.

Runs a val subset through (a) the full model and (b) the cached path, computes Acc@0.25
both ways, and requires the two numbers to agree to within 1e-4. If they do not, it
reports *which tensor* diverged rather than just that the metric moved.

    python experiments/diagnostics/validate_scene_cache.py \
        --cached_scenes_root cached_scenes \
        --use_pretrained 2024-12-18_20-40-38_3DVG-FIXED \
        --use_color --use_normal --num_samples 200

Exit code 0 means the cache is safe to use for ablations; anything else means stop.

Why the comparison can be exact
------------------------------
Two sources of nondeterminism sit between the point cloud and Acc@0.25, and both are
pinned here:

1. The 40k-point subsample (utils/pc_utils.random_sampling) redraws on every
   __getitem__ even when augment=False. Both paths run with deterministic=True so they
   see the identical draw.
2. MatchModule's proposal copy-paste and LangModule's word masking are gated on
   ``istrain``. The val split sets istrain=0, so neither fires.

Dropout/BatchNorm are handled by model.eval(). What remains is fp16 storage of
detr_features and aggregated_vote_features, which is why the threshold is 1e-4 on a
metric rather than bitwise equality on the tensors.

Acc@0.25 is pooled over per-sample IoUs (data_dict["ref_iou"]), not averaged over
per-batch rates, so the number does not depend on how the batches happen to divide.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Resolve the repo root from this file, not the cwd, so the script works when
# invoked as `python experiments/diagnostics/<name>.py` from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from data.scannet.model_util_scannet import ScannetDatasetConfig
from experiments.ablation.cached_scenes import SCENE_CACHE_KEYS_REQUIRED, CachedSceneDataset, read_meta
from lib.config import CONF
from lib.eval_helper import get_eval
from lib.loss_helper import get_loss
from experiments.ablation.cached_refnet import CachedRefNet
from models.refnet import RefNet

DC = ScannetDatasetConfig()
TOLERANCE = 1e-4


def load_val(num_samples, lang_num_max):
    with open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")) as f:
        scanrefer = json.load(f)
    scanrefer = scanrefer[:num_samples]

    chunks, current, scene_id = [], [], ""
    for data in scanrefer:
        if data["scene_id"] != scene_id or len(current) >= lang_num_max:
            if current:
                chunks.append(current)
            current, scene_id = [], data["scene_id"]
        current.append(data)
    if current:
        chunks.append(current)

    scene_list = sorted({d["scene_id"] for d in scanrefer})
    return scanrefer, chunks, scene_list


def build_dataset(args, scanrefer, chunks, scene_list, use_cached):
    # Both paths use deterministic=True so they see the identical 40k-point draw; that is
    # what makes an exact comparison possible at all.
    return CachedSceneDataset(
        args=args,
        scanrefer=scanrefer,
        scanrefer_new=chunks,
        scanrefer_all_scene=scene_list,
        split="val",
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max,
        augment=False,
        shuffle=False,
        use_cache=use_cached,
        cached_scenes_root=args.cached_scenes_root if use_cached else None,
        deterministic=True,
        subsample_salt=args.subsample_salt,
        lazy_scene_data=True,
    )


def build_model(args, use_cached, device):
    input_channels = (
        int(args.use_multiview) * 128
        + int(args.use_normal) * 3
        + int(args.use_color) * 3
        + int(not args.no_height)
    )
    model_args = argparse.Namespace(**vars(args))
    model_cls = CachedRefNet if use_cached else RefNet

    model = model_cls(
        args=model_args,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=args.no_reference,
        dataset_config=DC,
    )

    folder = args.use_checkpoint or args.use_pretrained
    ckpt = os.path.join(CONF.PATH.OUTPUT, folder, "model_criteria_25.pth")
    weights = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = model.state_dict()
    for key, value in weights.items():
        if key in state and state[key].shape == value.shape:
            state[key] = value
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def run(args, use_cached, device, collect_tensors=False):
    scanrefer, chunks, scene_list = load_val(args.num_samples, args.lang_num_max)
    dataset = build_dataset(args, scanrefer, chunks, scene_list, use_cached)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    eval_args = argparse.Namespace(**vars(args))
    eval_args.detection = False

    model = build_model(args, use_cached, device)

    ious, snapshots = [], {}
    label = "cached" if use_cached else "end-to-end"
    for data_dict in tqdm(loader, desc=label, unit="batch"):
        for key in data_dict:
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key].to(device)

        data_dict = model(data_dict)
        _, data_dict = get_loss(
            args=eval_args, data_dict=data_dict, config=DC,
            detection=False, reference=True, use_lang_classifier=not args.no_lang_cls,
        )
        data_dict = get_eval(
            data_dict=data_dict, config=DC, reference=True,
            use_lang_classifier=not args.no_lang_cls,
        )
        # Pool the raw per-sample IoUs so the result is batch-partition independent.
        ious.extend(np.asarray(data_dict["ref_iou"]).reshape(-1).tolist())

        if collect_tensors and not snapshots:
            for key in SCENE_CACHE_KEYS_REQUIRED:
                value = data_dict.get(key)
                if value is not None and torch.is_tensor(value):
                    value = value.float()
                    snapshots[key] = {
                        "shape": list(value.shape),
                        "mean": value.mean().item(),
                        "std": value.std().item(),
                        "absmax": value.abs().max().item(),
                    }

    ious = np.asarray(ious)
    acc = float((ious >= 0.25).sum() / max(ious.shape[0], 1))
    return acc, snapshots


def diagnose(args, device):
    """Locate the divergence by comparing per-tensor statistics between the two paths."""
    print("\n--- per-tensor diagnostics (first batch) ---")
    _, fresh = run(args, use_cached=False, device=device, collect_tensors=True)
    _, cached = run(args, use_cached=True, device=device, collect_tensors=True)

    print(f"{'tensor':<32} {'status':<10} {'shape':<22} {'d(mean)':>12} {'d(std)':>12}")
    suspects = []
    for key in SCENE_CACHE_KEYS_REQUIRED:
        a, b = fresh.get(key), cached.get(key)
        if a is None or b is None:
            print(f"{key:<32} {'MISSING':<10} {'-':<22} {'-':>12} {'-':>12}")
            suspects.append((key, "missing from one path"))
            continue
        if a["shape"] != b["shape"]:
            print(f"{key:<32} {'SHAPE':<10} {str(a['shape']):<22} {'-':>12} {'-':>12}")
            suspects.append((key, f"shape {a['shape']} vs {b['shape']}"))
            continue
        dmean = abs(a["mean"] - b["mean"])
        dstd = abs(a["std"] - b["std"])
        scale = max(abs(a["mean"]), 1e-8)
        status = "OK" if dmean / scale < 1e-3 else "DIVERGED"
        if status != "OK":
            suspects.append((key, f"mean {a['mean']:.6g} vs {b['mean']:.6g}"))
        print(f"{key:<32} {status:<10} {str(a['shape']):<22} {dmean:>12.3e} {dstd:>12.3e}")

    print("\nMost likely culprits:")
    if suspects:
        for key, why in suspects:
            print(f"  - {key}: {why}")
    else:
        print("  No single tensor diverged. The difference is therefore NOT in the cached")
        print("  detector output itself. Check, in order:")
        print("   1. deterministic_subsample disagreement (different point draw per path)")
        print("   2. a stochastic op still active at eval: MatchModule copy-paste or")
        print("      LangModule masking firing because istrain != 0")
        print("   3. model.eval() not applied (dropout / BatchNorm running stats)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cached_scenes_root", type=str,
                        default="cached_scenes")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lang_num_max", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    parser.add_argument("--out-dir", dest="out_dir",
                        default=os.path.join("outputs", "diagnostics"),
                        help="Where to write the PASS/FAIL verdict. This script used to "
                             "print it and keep nothing.")
    parser.add_argument("--subsample_salt", type=str, default="v1")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_points", type=int, default=40000)
    parser.add_argument("--num_proposals", type=int, default=256)
    parser.add_argument("--detector", type=str, default="VN", choices=["VN", "GF"])
    parser.add_argument("--use_pretrained", type=str, default="")
    parser.add_argument("--use_checkpoint", type=str, default="")
    parser.add_argument("--no_height", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_multiview", action="store_true")
    parser.add_argument("--use_bidir", action="store_true")
    parser.add_argument("--no_lang_cls", action="store_true")
    parser.add_argument("--no_reference", action="store_true")
    parser.add_argument("--no_detection", action="store_true")
    parser.add_argument("--lang_input", type=str, default="glove+parse")
    parser.add_argument("--GF_path", type=str, default=None)

    args = parser.parse_args()
    args.detection = False

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    if device.type == "cpu":
        print("WARNING: CPU mode. Parts of the detection path call .cuda() directly "
              "(models/proposal_module.decode_dataset_config, lib/loss_helper), so the "
              "end-to-end leg needs a GPU. Use a GPU machine for a real validation.")

    meta = read_meta(args.cached_scenes_root)
    print(f"cache generated {meta.get('generated_at')} from {meta.get('checkpoint')}")
    print(f"validating on the first {args.num_samples} val descriptions\n")

    acc_full, _ = run(args, use_cached=False, device=device)
    acc_cached, _ = run(args, use_cached=True, device=device)
    delta = abs(acc_full - acc_cached)

    print("\n" + "=" * 60)
    print(f"  end-to-end   Acc@0.25 = {acc_full:.8f}")
    print(f"  cached path  Acc@0.25 = {acc_cached:.8f}")
    print(f"  |difference|          = {delta:.3e}   (tolerance {args.tolerance:.1e})")
    print("=" * 60)

    passed = delta <= args.tolerance

    # This verdict gates every cached-protocol ablation, so it has to outlive the
    # terminal it was printed in. Previously the script wrote nothing at all.
    report = {
        "cached_scenes_root": args.cached_scenes_root,
        "cache_generated_at": meta.get("generated_at"),
        "cache_checkpoint": meta.get("checkpoint"),
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "lang_num_max": args.lang_num_max,
        "device": str(device),
        "tolerance": args.tolerance,
        "acc_end_to_end": acc_full,
        "acc_cached": acc_cached,
        "abs_difference": delta,
        "passed": bool(passed),
        "verdict": ("PASS - the cached path reproduces the end-to-end model."
                    if passed else
                    "FAIL - the cached path does NOT reproduce the end-to-end model. "
                    "Do not use this cache for ablations until the cause is found."),
    }
    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "scene_cache_validation.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")

    if passed:
        print("\nPASS - the cached path reproduces the end-to-end model.")
        return 0

    print("\nFAIL - the cached path does NOT reproduce the end-to-end model.")
    print("Do not use this cache for ablations until the cause is found.")
    diagnose(args, device)
    return 1


if __name__ == "__main__":
    sys.exit(main())
