
import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.scannet.model_util_scannet import ScannetDatasetConfig  # noqa: E402
from lib.config import CONF  # noqa: E402

DC = ScannetDatasetConfig()

TGT_TOKENS, ADJ_TOKENS, NGH_TOKENS = 7, 17, 75
GLOVE_DIM = 300


def build_data_dict(args, batch_size, istrain, device, variant):
    L = args.lang_num_max
    B = batch_size
    d = {}

    if variant == "end2end":
        d["point_clouds"] = torch.rand(B, args.num_points, 3 + args.input_feature_dim)
    else:
        K, NH, NS = args.num_proposals, DC.num_heading_bin, DC.num_size_cluster
        d["detr_features"] = torch.randn(B, K, 288)
        d["center"] = torch.randn(B, K, 3)
        d["objectness_scores"] = torch.randn(B, K, 2)
        d["sem_cls_scores"] = torch.randn(B, K, DC.num_class)
        d["heading_scores"] = torch.randn(B, K, NH)
        d["heading_residuals"] = torch.randn(B, K, NH)
        d["heading_residuals_normalized"] = torch.randn(B, K, NH)
        d["size_scores"] = torch.randn(B, K, NS)
        d["size_residuals"] = torch.randn(B, K, NS, 3)
        d["size_residuals_normalized"] = torch.randn(B, K, NS, 3)
        d["aggregated_vote_xyz"] = torch.randn(B, K, 3)
        d["aggregated_vote_features"] = torch.randn(B, K, 128)

    d["lang_feat_list"] = torch.randn(B, L, CONF.TRAIN.MAX_DES_LEN, GLOVE_DIM)
    d["target"] = torch.randn(B, L, TGT_TOKENS, GLOVE_DIM)
    d["adjectives"] = torch.randn(B, L, ADJ_TOKENS, GLOVE_DIM)
    d["neighbors"] = torch.randn(B, L, NGH_TOKENS, GLOVE_DIM)
    d["lang_len_list"] = torch.full((B, L), CONF.TRAIN.MAX_DES_LEN, dtype=torch.int64)
    d["main_lang_len_list"] = torch.full((B, L), CONF.TRAIN.MAX_DES_LEN, dtype=torch.int64)
    d["tgt_len"] = torch.full((B, L), TGT_TOKENS, dtype=torch.int64)
    d["adj_len"] = torch.full((B, L), ADJ_TOKENS, dtype=torch.int64)
    d["ngh_len"] = torch.full((B, L), NGH_TOKENS, dtype=torch.int64)
    d["first_obj_list"] = torch.zeros(B, L, dtype=torch.int64)
    d["unk"] = torch.randn(B, GLOVE_DIM)
    d["istrain"] = torch.full((B,), int(istrain), dtype=torch.int64)

    return {k: v.to(device) for k, v in d.items()}


def verify_shapes_against_dataset(args, built):
    from experiments.ablation.cached_scenes import CachedSceneDataset

    with open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")) as f:
        scanrefer = json.load(f)
    scanrefer = scanrefer[: args.lang_num_max]
    ds = CachedSceneDataset(
        args=args, scanrefer=scanrefer, scanrefer_new=[scanrefer],
        scanrefer_all_scene=sorted({d["scene_id"] for d in scanrefer}),
        split="val", num_points=args.num_points,
        use_height=(not args.no_height), use_color=args.use_color,
        use_normal=args.use_normal, use_multiview=args.use_multiview,
        lang_num_max=args.lang_num_max, augment=False, shuffle=False,
        use_cache=False, deterministic=True, lazy_scene_data=True,
    )
    sample = ds[0]
    mismatches = []
    for key, ours in built.items():
        if key not in sample:
            continue
        real = np.asarray(sample[key])
        want, got = tuple(ours.shape[1:]), tuple(real.shape)
        if want != got:
            mismatches.append(f"{key}: synthetic{want} vs dataset{got}")
    return mismatches


def build_model(args, variant, device):
    if variant == "end2end":
        from models.refnet import RefNet as cls
    else:
        from experiments.ablation.cached_refnet import CachedRefNet as cls

    model = cls(
        args=args,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=args.input_feature_dim,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=False,
        dataset_config=DC,
    )
    return model.to(device)


def param_counts(model):
    def n(m):
        return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

    out = {"total": n(model)}
    for name in ("backbone_net", "vgen", "proposal", "detector", "lang", "match"):
        if hasattr(model, name):
            out[name] = n(getattr(model, name))
    return out


def measure_flops(model, data_dict, depth=2):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None, {}, [], "torch.utils.flop_counter unavailable (needs torch >= 2.1)"

    model.eval()
    counter = None
    for kwargs in ({"mods": model, "depth": depth, "display": False},
                   {"depth": depth, "display": False},
                   {"display": False},
                   {}):
        try:
            counter = FlopCounterMode(**kwargs)
            break
        except TypeError:
            continue
    if counter is None:
        return None, {}, [], "could not construct FlopCounterMode"

    with torch.no_grad():
        with counter:
            model(dict(data_dict))

    total = int(counter.get_total_flops())

    per_module = {}
    try:
        for mod_name, ops in counter.get_flop_counts().items():
            s = int(sum(ops.values()))
            if s:
                per_module[mod_name or "<global>"] = s
    except Exception:
        pass

    counted = set()
    for k in per_module:
        parts = k.split(".")
        for i in range(len(parts)):
            counted.add(".".join(parts[: i + 1]))
    uncounted = []
    for name, sub in model.named_modules():
        if not name or list(sub.children()):
            continue
        if not any(p.requires_grad for p in sub.parameters(recurse=False)):
            continue
        top = name.split(".")[0]
        if name not in counted and top not in counted and f"Global.{name}" not in counted:
            uncounted.append(f"{name} ({type(sub).__name__})")

    note = ("FLOPs = 2 x MACs for matmul/conv/bmm/scaled-dot-product-attention. "
            "Elementwise ops, softmax, normalisation and index gathers "
            "(ball-query / kNN / FPS) are NOT counted -- the usual convention.")
    return total, per_module, uncounted, note


def measure_peak_memory(model, data_dict, device, train_mode, surrogate_backward=True):
    if device.type != "cuda":
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    if train_mode:
        model.train()
        out = model(dict(data_dict))
        if surrogate_backward:
            loss = out["cluster_ref"].float().sum()
            loss.backward()
        model.zero_grad(set_to_none=True)
    else:
        model.eval()
        with torch.no_grad():
            model(dict(data_dict))

    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    torch.cuda.empty_cache()
    return round(float(peak), 1)


def enable_cpu_shim():
    if not hasattr(torch.Tensor, "_orig_cuda"):
        torch.Tensor._orig_cuda = torch.Tensor.cuda

        def _cuda(self, *a, **k):
            return self
        torch.Tensor.cuda = _cuda
    print("  [cpu-shim] Tensor.cuda() neutralised for this process so the language\n"
          "             branch can run on CPU (models/lang_module.py:56 hardcodes it).\n"
          "             FLOPs/params stay valid; latency here is NOT representative.")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def measure_latency(model, data_dict, device, iters=100, warmup=10):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dict(data_dict))
        _sync(device)

        times = []
        for _ in range(iters):
            _sync(device)
            t0 = time.perf_counter()
            model(dict(data_dict))
            _sync(device)
            times.append(time.perf_counter() - t0)

    t = np.asarray(times) * 1000.0
    return {
        "iters": int(iters),
        "warmup": int(warmup),
        "mean_ms": round(float(t.mean()), 3),
        "std_ms": round(float(t.std()), 3),
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "min_ms": round(float(t.min()), 3),
    }


def environment(device):
    env = {
        "device": device.type,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        env["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        env["gpu_total_mem_mb"] = round(props.total_memory / 1024 ** 2, 1)
        env["gpu_capability"] = f"{props.major}.{props.minor}"
    try:
        from experiments.ablation.ablation_config import ABLATION
        env["pointnet_impl"] = ABLATION.POINTNET_IMPL
    except Exception:
        pass
    try:
        import pointnet2_ops  # noqa: F401
        env["pointnet2_ops_cuda_ext"] = "installed"
    except ImportError:
        env["pointnet2_ops_cuda_ext"] = "NOT installed (Python fallback in use)"
    return env


def run_variant(args, variant, device):
    print(f"\n{'=' * 78}\n  VARIANT: {variant}\n{'=' * 78}")
    res = {"variant": variant}

    model = build_model(args, variant, device)
    res["params"] = param_counts(model)
    print(f"  parameters: {res['params']['total'] / 1e6:.2f} M  "
          + "  ".join(f"{k}={v / 1e6:.2f}M" for k, v in res["params"].items() if k != "total"))

    d1 = build_data_dict(args, 1, istrain=0, device=device, variant=variant)

    model.eval()
    try:
        with torch.no_grad():
            model(dict(d1))
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        hint = ""
        if "knn_graph" in msg or "pyg-lib" in msg or "torch-cluster" in msg:
            hint = ("MatchModule needs torch_geometric's knn_graph, which requires "
                    "pyg-lib or torch-cluster. Install them (they are required to "
                    "train this model at all) and re-run -- e.g. "
                    "pip install torch-cluster -f "
                    "https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html")
        print(f"  [SKIP] forward pass failed -- {msg}")
        if hint:
            print(f"         {hint}")
        print("         Parameter counts above are still valid.")
        res["error"] = msg
        res["hint"] = hint
        res["flops"] = None
        res["peak_memory_mb"] = None
        res["latency_bs1"] = None
        del model, d1
        return res
    if args.real_sample:
        mism = verify_shapes_against_dataset(args, d1)
        res["shape_check"] = mism or "all synthetic shapes match the dataset"
        print(f"  shape check vs dataset: {res['shape_check']}")

    total, per_module, uncounted, note = measure_flops(model, d1, depth=args.flop_depth)
    res["flops"] = {
        "batch_size": 1,
        "lang_num_max": args.lang_num_max,
        "total_flops": total,
        "total_gflops": round(total / 1e9, 3) if total else None,
        "total_gmacs": round(total / 2e9, 3) if total else None,
        "per_module_gflops": {k: round(v / 1e9, 4) for k, v in
                              sorted(per_module.items(), key=lambda x: -x[1])},
        "uncounted_modules_with_params": uncounted,
        "note": note,
    }
    if total:
        print(f"  FLOPs (batch 1): {total / 1e9:.3f} GFLOPs  ({total / 2e9:.3f} GMACs)")
        for k, v in list(res["flops"]["per_module_gflops"].items())[:12]:
            print(f"      {k:<44s} {v:10.4f} GFLOPs")
    else:
        print(f"  FLOPs: UNAVAILABLE -- {note}")
    if uncounted:
        print(f"  [!] {len(uncounted)} parameterised module(s) registered ZERO FLOPs "
              f"(listed in the JSON) -- review before quoting the total")

    res["peak_memory_mb"] = {}
    if device.type == "cuda":
        res["peak_memory_mb"]["inference_bs1"] = measure_peak_memory(
            model, d1, device, train_mode=False)
        d_tr = build_data_dict(args, args.train_batch_size, istrain=1,
                               device=device, variant=variant)
        res["peak_memory_mb"]["training_bs%d" % args.train_batch_size] = \
            measure_peak_memory(model, d_tr, device, train_mode=True)
        res["peak_memory_mb"]["training_note"] = (
            "forward + backward on a surrogate scalar (cluster_ref.sum()); "
            "EXCLUDES lib/loss_helper.get_loss, which needs ground-truth tensors")
        del d_tr
        for k, v in res["peak_memory_mb"].items():
            if isinstance(v, float):
                print(f"  peak memory [{k}]: {v:.1f} MB")
    else:
        res["peak_memory_mb"] = "CPU run -- GPU memory not measurable"
        print("  peak memory: skipped (no CUDA device)")

    res["latency_bs1"] = measure_latency(model, d1, device,
                                         iters=args.latency_iters, warmup=args.warmup)
    lat = res["latency_bs1"]
    print(f"  latency (batch 1): mean {lat['mean_ms']:.2f} ms  std {lat['std_ms']:.2f}  "
          f"p50 {lat['p50_ms']:.2f}  p95 {lat['p95_ms']:.2f}")

    if args.eval_batch_size and args.eval_batch_size != 1:
        d_ev = build_data_dict(args, args.eval_batch_size, istrain=0,
                               device=device, variant=variant)
        st = measure_latency(model, d_ev, device,
                             iters=max(10, args.latency_iters // 4), warmup=args.warmup)
        st["per_sample_ms"] = round(st["mean_ms"] / args.eval_batch_size, 3)
        res["latency_bs%d" % args.eval_batch_size] = st
        print(f"  latency (batch {args.eval_batch_size}): mean {st['mean_ms']:.2f} ms "
              f"-> {st['per_sample_ms']:.2f} ms/sample")
        del d_ev

    del model, d1
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=["end2end", "cached", "both"], default="both",
                   help="end2end = full RefNet (the paper's model); "
                        "cached = fusion-only on pre-computed detector output.")
    p.add_argument("--output", default=os.path.join("outputs", "complexity"),
                   help="Directory for complexity_report.json (repo convention: outputs/).")
    p.add_argument("--latency_iters", type=int, default=100)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--flop_depth", type=int, default=2)
    p.add_argument("--repeat", type=int, default=1,
                   help="Run the whole suite N times; FLOPs must be identical across "
                        "runs and latency within a few percent, else something else is "
                        "using the device.")
    p.add_argument("--real_sample", action="store_true",
                   help="Cross-check synthetic shapes against a real val sample "
                        "(loads the GloVe pickle; slow, needs data/).")
    p.add_argument("--cpu", action="store_true", help="Force CPU (FLOPs/params only).")

    p.add_argument("--train_batch_size", type=int, default=8,
                   help="Batch size used by the paper's training runs "
                        "(ScanRefer_train.py debug block sets 8).")
    p.add_argument("--eval_batch_size", type=int, default=8,
                   help="Batch size ScanRefer_eval.py uses, for a comparable number.")
    p.add_argument("--lang_num_max", type=int, default=1,
                   help="1 matches ScanRefer_eval.py (which asserts it) and the "
                        "single-query deployment scenario.")
    p.add_argument("--num_points", type=int, default=40000)
    p.add_argument("--num_proposals", type=int, default=256)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--detector", type=str, default="VN", choices=["VN", "GF"])
    p.add_argument("--no_height", action="store_true")
    p.add_argument("--use_color", action="store_true")
    p.add_argument("--use_normal", action="store_true")
    p.add_argument("--use_multiview", action="store_true")
    p.add_argument("--use_bidir", action="store_true")
    p.add_argument("--no_lang_cls", action="store_true")
    p.add_argument("--lang_input", type=str, default="glove+parse")
    p.add_argument("--GF_path", type=str, default=None)

    args = p.parse_args()
    args.detection = False
    args.no_reference = False

    args.input_feature_dim = (int(args.use_multiview) * 128 + int(args.use_normal) * 3
                              + int(args.use_color) * 3 + int(not args.no_height))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    env = environment(device)

    print("=" * 78)
    print("  COMPUTATIONAL COMPLEXITY  (Reviewer #2)")
    print("=" * 78)
    for k, v in env.items():
        print(f"  {k:<26s} {v}")
    print(f"  {'input_feature_dim':<26s} {args.input_feature_dim} "
          f"(point_clouds channels = {3 + args.input_feature_dim})")
    if device.type == "cpu":
        print("\n  NOTE: CPU run. FLOPs and parameter counts are hardware-independent\n"
              "        and valid; GPU memory is unmeasurable and latency is NOT\n"
              "        representative. Run on the paper's GPU for those two.")
        enable_cpu_shim()
        print()

    variants = ["end2end", "cached"] if args.variant == "both" else [args.variant]

    runs = []
    for r in range(args.repeat):
        if args.repeat > 1:
            print(f"\n########## repeat {r + 1}/{args.repeat} ##########")
        runs.append({v: run_variant(args, v, device) for v in variants})

    report = {
        "environment": env,
        "config": {
            "variant": args.variant,
            "lang_num_max": args.lang_num_max,
            "num_points": args.num_points,
            "num_proposals": args.num_proposals,
            "input_feature_dim": args.input_feature_dim,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "use_color": args.use_color, "use_normal": args.use_normal,
            "use_height": not args.no_height,
        },
        "results": runs[0],
        "repeats": runs if args.repeat > 1 else None,
        "caveats": [
            "FLOPs count matmul/conv/bmm/SDPA only; elementwise ops, softmax, "
            "normalisation and index gathers (ball-query/kNN/FPS) are excluded.",
            "Latency and memory are hardware-dependent -- always quote them with the "
            "'environment' block above.",
            "The existing figure in the manuscript (10.35 ms) was produced by "
            "ScanRefer_eval.py, which times model() with time.time() and NO "
            "torch.cuda.synchronize() and no warmup, at batch_size=8. CUDA is "
            "asynchronous, so that number measures kernel launch rather than "
            "execution and is not directly comparable to the synchronised numbers "
            "here. Report the new number, and state the measurement protocol.",
            "Baseline FLOPs/memory in the comparison table are taken from the "
            "original papers and were NOT reproduced on this hardware; that must be "
            "disclosed in a table footnote.",
            "Training peak memory uses a surrogate backward on the model output and "
            "excludes get_loss; treat it as a lower bound.",
        ],
    }

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "complexity_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 78}\n  SUMMARY  (paste into the complexity table)\n{'=' * 78}")
    hdr = f"  {'variant':<10s} {'params(M)':>10s} {'GFLOPs':>10s} {'infer MB':>10s} {'lat mean':>10s} {'lat p95':>9s}"
    print(hdr + "\n  " + "-" * (len(hdr) - 2))
    for v, r in runs[0].items():
        if r.get("error"):
            print(f"  {v:<10s} {r['params']['total'] / 1e6:>10.2f} "
                  f"{'FAILED':>10s} {'-':>10s} {'-':>10s} {'-':>9s}")
            continue
        mem = r["peak_memory_mb"]
        mem_s = (f"{mem.get('inference_bs1'):.0f}" if isinstance(mem, dict)
                 and mem.get("inference_bs1") else "n/a")
        g = r["flops"]["total_gflops"]
        print(f"  {v:<10s} {r['params']['total'] / 1e6:>10.2f} "
              f"{(f'{g:.2f}' if g else 'n/a'):>10s} {mem_s:>10s} "
              f"{r['latency_bs1']['mean_ms']:>9.2f}m {r['latency_bs1']['p95_ms']:>8.2f}m")
    print(f"\n  wrote {out_path}")
    print("  parsing latency is measured separately: "
          "python experiments/complexity/measure_parsing_latency.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
