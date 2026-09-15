"""
What the model costs to run, and whether the implementation does what the paper says.

    python experiments/measurements.py <command> --help

Commands: model, parsing-latency, online-latency, validate-scene-cache,
audit-scene-cache, cached-eval-cpu, dist-weights-sweep.

Latency is warmed up and synchronised everywhere; FLOPs count matmul/conv/bmm/attention
only. Conventions, and why parse and grounding latency are reported separately:
details.md §6.
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict

import numpy as np

REPO = REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from tqdm import tqdm  # noqa: E402

from data.scannet.model_util_scannet import ScannetDatasetConfig  # noqa: E402
from experiments.analysis import THRESHOLDS  # noqa: E402
from lib.config import CONF  # noqa: E402

DC = ScannetDatasetConfig()



# ---- model ----

_DOC_MODEL = """
FLOPs, peak GPU memory and grounding-network latency.

    python experiments/measurements.py model --variant both \
        --use_color --use_normal --latency_iters 100

Two variants, always labelled separately: ``end2end`` is the full RefNet, the deployed
model; ``cached`` is CachedRefNet, language + fusion only, which is what the ablations
train on and is much cheaper.

FLOPs come from torch.utils.flop_counter, which is dispatch-based and so survives the
data-dependent control flow here that trace-based counters choke on. Latency is warmed up
and synchronised. Conventions and caveats: details.md §6.
"""

#: Token caps enforced by lib/dataset.py (_transform_parsed allocates these rows).
TGT_TOKENS, ADJ_TOKENS, NGH_TOKENS = 7, 17, 75
GLOVE_DIM = 300



def build_data_dict(args, batch_size, istrain, device, variant):
    """A shape-exact data_dict, matching what lib/dataset.py emits.

    Shapes are taken from the real dataloader contract, not invented:
      point_clouds      (B, num_points, 3 + input_feature_dim)
      lang_feat_list    (B, L, MAX_DES_LEN, 300)
      target/adjectives/neighbors   (B, L, 7|17|75, 300)
      *_len             (B, L)
      istrain           (B,)
    FLOPs and latency depend only on these shapes, so synthetic values with the
    correct shapes give the same answer as a real batch at a fraction of the
    cost (loading a real batch pulls in the 978 MB GloVe pickle). Use
    --real_sample to cross-check the shapes against the actual dataset.
    """
    L = args.lang_num_max
    B = batch_size
    d = {}

    if variant == "end2end":
        d["point_clouds"] = torch.rand(B, args.num_points, 3 + args.input_feature_dim)
    else:
        # Cached detector output, exactly the keys experiments/ablation/cache.py stores.
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

    # Language branch inputs, identical for both variants.
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
    # istrain gates MatchModule's copy-paste and LangModule's word masking.
    d["istrain"] = torch.full((B,), int(istrain), dtype=torch.int64)

    return {k: v.to(device) for k, v in d.items()}


def verify_shapes_against_dataset(args, built):
    """Optional cross-check that the synthetic shapes match a real val sample."""
    from experiments.ablation.cache import CachedSceneDataset

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



def _cx_build_model(args, variant, device):
    if variant == "end2end":
        from models.refnet import RefNet as cls
    else:
        from experiments.ablation.cache import CachedRefNet as cls

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
    """Trainable parameters overall and per top-level branch.

    Mirrors get_num_params() in ScanRefer_train.py so the numbers in this
    report are directly comparable to what training already logs.
    """
    def n(m):
        return int(sum(p.numel() for p in m.parameters() if p.requires_grad))

    out = {"total": n(model)}
    for name in ("backbone_net", "vgen", "proposal", "detector", "lang", "match"):
        if hasattr(model, name):
            out[name] = n(getattr(model, name))
    return out



def measure_flops(model, data_dict, depth=2):
    """FLOPs for one forward pass via torch's built-in FlopCounterMode.

    Returns (total_flops, per_module_dict, uncounted_modules, note).
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None, {}, [], "torch.utils.flop_counter unavailable (needs torch >= 2.1)"

    model.eval()
    # Signature drifted across torch versions; try the richest form first.
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

    # Honesty check: modules that own parameters but registered no FLOPs.
    counted = set()
    for k in per_module:
        parts = k.split(".")
        for i in range(len(parts)):
            counted.add(".".join(parts[: i + 1]))
    uncounted = []
    for name, sub in model.named_modules():
        if not name or list(sub.children()):
            continue  # leaves only
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
    """Peak allocated memory for one forward (+ backward if train_mode)."""
    if device.type != "cuda":
        return None

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    if train_mode:
        model.train()
        out = model(dict(data_dict))
        if surrogate_backward:
            # A surrogate scalar on the model's own output, so this excludes
            # get_loss and is a labelled lower bound, not end-to-end training memory.
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
    """Let the model run on CPU by neutralising hardcoded .cuda() calls.

    models/lang_module.py:56 and a few sites in models/match_module.py call
    ``.cuda()`` unconditionally, so the language+fusion path cannot execute on a
    CPU-only machine. FLOPs and parameter counts are hardware-independent and
    worth having without a GPU, so this shim makes ``Tensor.cuda()`` a no-op
    **inside this measurement process only**.

    It changes nothing in the repository and nothing about the arithmetic: the
    same ops run on the same shapes, just on CPU. Latency and memory measured
    under the shim are meaningless and are suppressed/labelled accordingly.
    """
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
    """Warmed-up, synchronised per-forward latency. Returns a stats dict in ms.

    The synchronisation is the point: CUDA kernels are asynchronous, so timing
    a forward pass without torch.cuda.synchronize() measures kernel *launch*
    time, not execution. The existing measurement in ScanRefer_eval.py has no
    sync and no warmup, which is why its number is not directly comparable.
    """
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

    model = _cx_build_model(args, variant, device)
    res["params"] = param_counts(model)
    print(f"  parameters: {res['params']['total'] / 1e6:.2f} M  "
          + "  ".join(f"{k}={v / 1e6:.2f}M" for k, v in res["params"].items() if k != "total"))

    d1 = build_data_dict(args, 1, istrain=0, device=device, variant=variant)

    # Preflight: one forward, so a missing dependency is reported as an actionable
    # message instead of a traceback halfway through the measurement.
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

    # Batch 1: the single-query deployment scenario.
    res["latency_bs1"] = measure_latency(model, d1, device,
                                         iters=args.latency_iters, warmup=args.warmup)
    lat = res["latency_bs1"]
    print(f"  latency (batch 1): mean {lat['mean_ms']:.2f} ms  std {lat['std_ms']:.2f}  "
          f"p50 {lat['p50_ms']:.2f}  p95 {lat['p95_ms']:.2f}")

    # Latency at the eval script's batch size, for comparability.
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


def _add_model_args(parser):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=["end2end", "cached", "both"], default="both",
                   help="end2end = full RefNet (the paper's model); "
                        "cached = fusion-only on pre-computed detector output.")
    parser.add_argument("--output", default=os.path.join("outputs", "complexity"),
                   help="Directory for complexity_report.json (repo convention: outputs/).")
    parser.add_argument("--latency_iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--flop_depth", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=1,
                   help="Run the whole suite N times; FLOPs must be identical across "
                        "runs and latency within a few percent, else something else is "
                        "using the device.")
    parser.add_argument("--real_sample", action="store_true",
                   help="Cross-check synthetic shapes against a real val sample "
                        "(loads the GloVe pickle; slow, needs data/).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (FLOPs/params only).")

    # model / data configuration -- must match the training run being reported
    parser.add_argument("--train_batch_size", type=int, default=8,
                   help="Batch size used by the paper's training runs "
                        "(ScanRefer_train.py debug block sets 8).")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                   help="Batch size ScanRefer_eval.py uses, for a comparable number.")
    parser.add_argument("--lang_num_max", type=int, default=1,
                   help="1 matches ScanRefer_eval.py (which asserts it) and the "
                        "single-query deployment scenario.")
    parser.add_argument("--num_points", type=int, default=40000)
    parser.add_argument("--num_proposals", type=int, default=256)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--detector", type=str, default="VN", choices=["VN", "GF"])
    parser.add_argument("--no_height", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_multiview", action="store_true")
    parser.add_argument("--use_bidir", action="store_true")
    parser.add_argument("--no_lang_cls", action="store_true")
    parser.add_argument("--lang_input", type=str, default="glove+parse")
    parser.add_argument("--GF_path", type=str, default=None)
    return parser


def run_model(args):
    args.detection = False       # read by the dataset/model paths
    args.no_reference = False

    args.input_feature_dim = (int(args.use_multiview) * 128 + int(args.use_normal) * 3
                              + int(args.use_color) * 3 + int(not args.no_height))

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    env = environment(device)

    print("=" * 78)
    print("  COMPUTATIONAL COMPLEXITY")
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
          "python experiments/measurements.py parsing-latency")
    return 0


# ---- parsing-latency ----

_DOC_PARSING_LATENCY = """
Sentence-parsing latency and cost, and the offline-vs-online deployment question.

    python experiments/measurements.py parsing-latency --mode offline_check
    python experiments/measurements.py parsing-latency --mode spacy --num_samples 500
    python experiments/measurements.py parsing-latency --mode gpt --num_samples 100
    python experiments/measurements.py parsing-latency --mode all --num_samples 100

Parsing is offline for every benchmark number here -- lib/dataset.py loads a precomputed
JSON once at construction and no parser runs in the forward path -- so parse cost is
additive only for an unseen sentence at deployment. The two scenarios are measured and
kept apart; see details.md §6.

The GPT leg makes real API calls over distinct val descriptions (latency varies with input
length, so repeating one sentence would mislead) and reads token usage for $/query.
"""

#: Reconstructed to match the observed schema of data_parsing/final_parsing/:
#: three surface-phrase fields, the literal "not mentioned" when absent.
GPT_SYSTEM_PROMPT = (
    "You parse 3D scene referring expressions. Given one description, return a "
    "JSON object with exactly three string fields: \"target\" (the referred "
    "object), \"adjectives\" (its attributes as they appear in the sentence), "
    "and \"neighbors\" (the spatial-relation phrases describing nearby "
    "objects). Use the literal string \"not mentioned\" for any field the "
    "description does not supply. Return only the JSON object."
)

#: USD per 1M tokens. Defaults are gpt-4o-mini list prices; override via CLI,
#: prices change and the paper should quote the date they were taken.
DEFAULT_PRICE_IN = 0.150
DEFAULT_PRICE_OUT = 0.600


def load_val_descriptions(n, seed=42):
    """n distinct val descriptions (not one sentence repeated)."""
    path = os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")
    with open(path) as f:
        records = json.load(f)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(records), size=min(n, len(records)), replace=False)
    out = [records[int(i)]["description"] for i in idx]
    lens = [len(d.split()) for d in out]
    return out, {"count": len(out), "mean_words": round(float(np.mean(lens)), 1),
                 "min_words": int(min(lens)), "max_words": int(max(lens))}


def stats_ms(times_s):
    t = np.asarray(times_s) * 1000.0
    return {
        "n": int(t.size),
        "mean_ms": round(float(t.mean()), 3),
        "std_ms": round(float(t.std()), 3),
        "p50_ms": round(float(np.percentile(t, 50)), 3),
        "p95_ms": round(float(np.percentile(t, 95)), 3),
        "min_ms": round(float(t.min()), 3),
        "max_ms": round(float(t.max()), 3),
    }



def offline_check():
    """Establish from the code whether parsing is in the inference path."""
    repo = REPO_ROOT
    findings = {}

    ds = open(os.path.join(repo, "lib", "dataset.py"), encoding="utf-8").read()
    findings["dataset_loads_precomputed_json"] = "tokenized_parsed_result_" in ds
    findings["parse_cache_read_at_construction"] = "_tranform_des" in ds

    parser_names = ("openai", "OpenAI", "ChatCompletion", "gpt-4o", "spacy.load")
    hits = []
    for rel in ("lib/dataset.py", "models/lang_module.py", "models/match_module.py",
                "models/refnet.py", "experiments/ablation/cache.py", "lib/solver.py"):
        p = os.path.join(repo, rel)
        if not os.path.isfile(p):
            continue
        src = open(p, encoding="utf-8").read()
        for name in parser_names:
            if name in src:
                hits.append(f"{rel}: {name}")
    findings["parser_invocations_in_forward_path"] = hits or "none"

    caches = []
    pdir = os.path.join(repo, "data_parsing")
    if os.path.isdir(pdir):
        caches = sorted(d for d in os.listdir(pdir)
                        if os.path.isdir(os.path.join(pdir, d)))
    findings["available_parse_caches"] = caches

    findings["verdict"] = (
        "OFFLINE for all reported benchmarks. lib/dataset.py loads a precomputed "
        "tokenized_parsed_result_{split}.json once at dataset construction and "
        "indexes it by (scene_id, object_id, ann_id). No parser -- neither an LLM "
        "API call nor spaCy -- is invoked anywhere in the model forward path. "
        "Parsing therefore contributes ZERO to the per-query latency of every "
        "number reported on ScanRefer/ReferIt3D."
    ) if not hits else (
        "REVIEW NEEDED: a parser reference was found in the forward path -- see "
        "parser_invocations_in_forward_path."
    )
    findings["deployment_note"] = (
        "For a genuinely unseen sentence at deployment time, one parse call is "
        "required before grounding, and only then is parse latency additive. "
        "`measurements.py online-latency` measures that case end to end."
    )
    return findings



def measure_spacy(descriptions, model="en_core_web_sm", warmup=5):
    from experiments.ablation.parsing import load_parser, parse_with_spacy

    t0 = time.perf_counter()
    nlp = load_parser(model)
    load_s = time.perf_counter() - t0

    for d in descriptions[:warmup]:
        parse_with_spacy(d, nlp=nlp)

    times, samples = [], []
    for i, d in enumerate(descriptions):
        t0 = time.perf_counter()
        parsed = parse_with_spacy(d, nlp=nlp)
        times.append(time.perf_counter() - t0)
        if i < 3:
            samples.append({"description": d, "parsed": parsed})

    out = stats_ms(times)
    out.update({
        "model": model,
        "model_load_s": round(load_s, 2),
        "throughput_desc_per_s": round(1.0 / (float(np.mean(times)) or 1e-9), 1),
        "cost_usd_per_query": 0.0,
        "note": "Local CPU inference, per description, one at a time (the "
                "deployment scenario). Batched nlp.pipe() is faster and is what "
                "experiments/ablation/parsing.py spacy uses for bulk parsing.",
        "samples": samples,
    })
    return out



def measure_gpt(descriptions, model, price_in, price_out, warmup=2):
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed -- pip install openai"}
    if not os.environ.get("OPENAI_API_KEY"):
        return {"error": "OPENAI_API_KEY not set in the environment"}

    client = OpenAI()

    def one(desc):
        t0 = time.perf_counter()
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": GPT_SYSTEM_PROMPT},
                      {"role": "user", "content": desc}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        dt = time.perf_counter() - t0
        return dt, r

    for d in descriptions[:warmup]:
        try:
            one(d)
        except Exception:
            pass

    times, tok_in, tok_out, errors, samples = [], [], [], 0, []
    for i, d in enumerate(descriptions):
        try:
            dt, r = one(d)
        except Exception as exc:
            errors += 1
            if errors <= 3:
                print(f"    [api error] {type(exc).__name__}: {exc}")
            continue
        times.append(dt)
        u = getattr(r, "usage", None)
        if u is not None:
            tok_in.append(u.prompt_tokens)
            tok_out.append(u.completion_tokens)
        if i < 3:
            samples.append({"description": d,
                            "response": r.choices[0].message.content})
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(descriptions)} calls, "
                  f"running mean {np.mean(times) * 1000:.0f} ms")

    if not times:
        return {"error": f"all {len(descriptions)} calls failed"}

    out = stats_ms(times)
    mean_in = float(np.mean(tok_in)) if tok_in else None
    mean_out = float(np.mean(tok_out)) if tok_out else None
    cost = None
    if mean_in is not None and mean_out is not None:
        cost = mean_in / 1e6 * price_in + mean_out / 1e6 * price_out
    out.update({
        "model": model,
        "errors": errors,
        "mean_prompt_tokens": round(mean_in, 1) if mean_in else None,
        "mean_completion_tokens": round(mean_out, 1) if mean_out else None,
        "price_usd_per_1m_input": price_in,
        "price_usd_per_1m_output": price_out,
        "cost_usd_per_query": round(cost, 8) if cost else None,
        "cost_usd_per_1k_queries": round(cost * 1000, 5) if cost else None,
        "cost_usd_full_scanrefer_46173": round(cost * 46173, 3) if cost else None,
        "note": "Real API calls over distinct val descriptions. Latency includes "
                "network round-trip and therefore depends on location and load. "
                "The prompt is RECONSTRUCTED from the observed output schema -- "
                "the original parsing script is not in this repository.",
        "samples": samples,
    })
    return out


def _add_parsing_latency_args(parser):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["offline_check", "spacy", "gpt", "all"],
                   default="offline_check")
    parser.add_argument("--num_samples", type=int, default=100,
                   help="Distinct val descriptions to parse (>=100 recommended).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacy_model", default="en_core_web_sm")
    parser.add_argument("--gpt_model", default="gpt-4o-mini")
    parser.add_argument("--price_in", type=float, default=DEFAULT_PRICE_IN,
                   help="USD per 1M input tokens.")
    parser.add_argument("--price_out", type=float, default=DEFAULT_PRICE_OUT,
                   help="USD per 1M output tokens.")
    parser.add_argument("--output", default=os.path.join("outputs", "complexity"))
    return parser


def run_parsing_latency(args):

    print("=" * 78)
    print("  PARSING LATENCY & COST")
    print("=" * 78)

    report = {"config": vars(args)}

    # --- always: the deployment question ---
    dep = offline_check()
    report["deployment"] = dep
    print("\n-- deployment analysis --")
    for k, v in dep.items():
        if k in ("verdict", "deployment_note"):
            continue
        print(f"  {k:<38s} {v}")
    print(f"\n  VERDICT: {dep['verdict']}\n")
    print(f"  {dep['deployment_note']}\n")

    need_desc = args.mode in ("spacy", "gpt", "all")
    if need_desc:
        descs, dstats = load_val_descriptions(args.num_samples, args.seed)
        report["description_sample"] = dstats
        print(f"-- sampled {dstats['count']} distinct val descriptions "
              f"(mean {dstats['mean_words']} words, "
              f"range {dstats['min_words']}-{dstats['max_words']}) --")

    if args.mode in ("spacy", "all"):
        print("\n-- spaCy (variant B), local --")
        r = measure_spacy(descs, args.spacy_model)
        report["spacy"] = r
        print(f"  mean {r['mean_ms']:.2f} ms   std {r['std_ms']:.2f}   "
              f"p50 {r['p50_ms']:.2f}   p95 {r['p95_ms']:.2f}   "
              f"({r['throughput_desc_per_s']:.0f} desc/s, $0 per query)")

    if args.mode in ("gpt", "all"):
        print(f"\n-- {args.gpt_model} (variant A), REAL API calls --")
        r = measure_gpt(descs, args.gpt_model, args.price_in, args.price_out)
        report["gpt"] = r
        if "error" in r:
            print(f"  SKIPPED: {r['error']}")
        else:
            print(f"  mean {r['mean_ms']:.1f} ms   std {r['std_ms']:.1f}   "
                  f"p50 {r['p50_ms']:.1f}   p95 {r['p95_ms']:.1f}   "
                  f"(errors: {r['errors']})")
            print(f"  tokens: {r['mean_prompt_tokens']} in / "
                  f"{r['mean_completion_tokens']} out per query")
            if r["cost_usd_per_query"]:
                print(f"  cost:   ${r['cost_usd_per_query']:.6f}/query   "
                      f"${r['cost_usd_per_1k_queries']:.4f}/1k   "
                      f"${r['cost_usd_full_scanrefer_46173']:.2f} for all 46,173")

    report["reporting_guidance"] = [
        "Benchmark numbers: parsing is offline, so report it as a separate row and do "
        "not add it to grounding latency.",
        "Deployment (one unseen sentence): parse latency is additive. Measured "
        "end to end by `measurements.py online-latency`.",
    ]

    os.makedirs(args.output, exist_ok=True)
    # One file per mode; a shared filename let whichever mode ran second erase the first.
    out_path = os.path.join(args.output, f"parsing_latency_report_{args.mode}.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {out_path}")

    # The historical path stays as a merged view: top-level keys are hoisted from every
    # mode that has run, with the per-mode detail under "modes".
    merged_path = os.path.join(args.output, "parsing_latency_report.json")
    modes = {}
    if os.path.isfile(merged_path):
        try:
            with open(merged_path) as f:
                existing = json.load(f)
            modes = existing.get("modes") or {}
            if not modes and isinstance(existing.get("config"), dict):
                # A pre-fix single-mode file: keep it under its own mode name.
                previous = existing.get("config", {}).get("mode")
                if previous:
                    modes[previous] = existing
        except (ValueError, OSError):
            modes = {}
    modes[args.mode] = report

    merged = {}
    for name in sorted(modes, key=lambda m: m == args.mode):   # current mode last = wins
        for key, value in modes[name].items():
            merged[key] = value
    merged["modes"] = modes
    merged["_note"] = ("Top-level keys are hoisted from every --mode that has run, so "
                       "existing readers keep working; per-mode detail is under 'modes'. "
                       "Each mode also has its own parsing_latency_report_<mode>.json.")
    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"wrote {merged_path}  (modes: {', '.join(sorted(modes))})")
    return 0


# ---- validate-scene-cache ----

_DOC_VALIDATE_SCENE_CACHE = """
Prove the scene cache is equivalent to the end-to-end model.

    python experiments/measurements.py validate-scene-cache \
        --cached_scenes_root cached_scenes \
        --use_pretrained 2024-12-18_20-40-38_3DVG-FIXED \
        --use_color --use_normal --num_samples 200

Runs a val subset through the full model and the cached path, computes Acc@0.25 both ways,
and requires them to agree to 1e-4 -- reporting which tensor diverged if they do not. Exit
code 0 means the cache is safe to use for ablations; anything else means stop.

The comparison can be exact because both sources of nondeterminism are pinned: both paths
run with deterministic=True so the 40k-point draw is identical, and copy-paste and word
masking are gated on istrain, which val sets to 0. What remains is fp16 storage of two
cached tensors, which is why the threshold is 1e-4 on a metric rather than bitwise equality.
"""

# Resolve the repo root from this file, not the cwd, so the script works when
# invoked as `python experiments/measurements.py <command>` from anywhere.
sys.path.insert(0, REPO_ROOT)

from data.scannet.model_util_scannet import ScannetDatasetConfig
from experiments.ablation.cache import SCENE_CACHE_KEYS_REQUIRED, CachedSceneDataset, read_meta
from lib.config import CONF
from lib.eval_helper import get_eval
from lib.loss_helper import get_loss
from experiments.ablation.cache import CachedRefNet
from models.refnet import RefNet

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


def _vs_build_model(args, use_cached, device):
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

    model = _vs_build_model(args, use_cached, device)

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


def _add_validate_scene_cache_args(parser):
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
    return parser


def run_validate_scene_cache(args):
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


# ---- audit-scene-cache ----

_DOC_AUDIT_SCENE_CACHE = """
Is the scene cache complete, well-formed, and good enough to ground on? No model needed.

    python experiments/measurements.py audit-scene-cache
    python experiments/measurements.py audit-scene-cache --split train --limit 50

validate_scene_cache.py is the strict test but needs a GPU. This answers a weaker question
with no model at all: every scene present with the keys, shapes and dtypes meta.json
promises and no NaN; and the recall ceiling, since the fusion net can only return one of
the 256 cached proposals so max_i IoU(proposal_i, gt) bounds accuracy.

The cache subsamples deterministically while the original evaluation subsampled at random,
so an achieved IoU is not required to sit below its cached ceiling. How often it does not
is reported as a consistency statistic rather than pass/fail.
"""

def _audit_aabb_iou(a, b):
    """3-D IoU of two axis-aligned boxes given as (8, 3) corner arrays.

    ScanRefer's boxes are axis aligned, so the extent alone defines them and the IoU is
    exact -- no convex-hull intersection needed.
    """
    a_low, a_high = a.min(axis=0), a.max(axis=0)
    b_low, b_high = b.min(axis=0), b.max(axis=0)
    overlap = np.minimum(a_high, b_high) - np.maximum(a_low, b_low)
    if np.any(overlap <= 0):
        return 0.0
    intersection = float(np.prod(overlap))
    volume_a = float(np.prod(a_high - a_low))
    volume_b = float(np.prod(b_high - b_low))
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0 else 0.0


def batch_max_iou(proposals, gt):
    """Best IoU between one gt box and all (N, 8, 3) proposals, vectorised."""
    p_low, p_high = proposals.min(axis=1), proposals.max(axis=1)      # (N, 3)
    g_low, g_high = gt.min(axis=0), gt.max(axis=0)                    # (3,)

    overlap = np.minimum(p_high, g_high) - np.maximum(p_low, g_low)
    overlap = np.clip(overlap, 0.0, None)
    intersection = overlap.prod(axis=1)

    volume_p = np.clip(p_high - p_low, 0.0, None).prod(axis=1)
    volume_g = float(np.clip(g_high - g_low, 0.0, None).prod())
    union = volume_p + volume_g - intersection

    ious = np.where(union > 0, intersection / np.maximum(union, 1e-12), 0.0)
    return ious


def load_cached_scene(root, split, scene_id):
    import torch
    path = os.path.join(root, split, f"{scene_id}.p")
    if not os.path.isfile(path):
        return None, path
    return torch.load(path, map_location="cpu", weights_only=False), path


def _add_audit_scene_cache_args(parser):
    parser.add_argument("--cached-scenes-root", dest="root", default="cached_scenes")
    parser.add_argument("--split", default="val")
    parser.add_argument("--predictions",
                        default="outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p",
                        help="used for the gt boxes and for the achieved-IoU comparison")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--limit", type=int, default=0, help="0 = every scene")
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/diagnostics")
    return parser


def run_audit_scene_cache(args):

    root = os.path.join(REPO, args.root)
    print("=" * 84)
    print(f"Scene-cache audit -- split '{args.split}', root '{args.root}'")
    print("=" * 84)

    meta_path = os.path.join(root, "meta.json")
    if not os.path.isfile(meta_path):
        print(f"\nno cache found: {os.path.relpath(meta_path, REPO)}")
        print("Build it first with experiments/ablation/cache.py build (GPU).")
        return 1
    with open(meta_path) as handle:
        meta = json.load(handle)

    split_meta = (meta.get("splits") or {}).get(args.split, {})
    expected_keys = list(meta.get("cached_keys") or [])
    print(f"\n[meta] complete={meta.get('complete')}  "
          f"detector={meta.get('detector')}  num_points={meta.get('num_points')}  "
          f"num_proposals={meta.get('num_proposals')}  salt={meta.get('subsample_salt')!r}")
    print(f"[meta] checkpoint = {meta.get('checkpoint')}")
    print(f"[meta] {args.split}: {split_meta.get('num_scenes')} scenes, "
          f"complete={split_meta.get('complete')}")
    print(f"[meta] {len(expected_keys)} cached keys")

    records = load_scanrefer(args.split, args.data_root)
    scene_ids = sorted({r["scene_id"] for r in records})
    if args.limit:
        scene_ids = scene_ids[:args.limit]

    print(f"\n[1] Integrity over {len(scene_ids)} scene(s)\n" + "-" * 84)
    missing_files, missing_keys, bad_values, shapes = [], [], [], {}
    cache = {}
    for scene_id in scene_ids:
        payload, path = load_cached_scene(root, args.split, scene_id)
        if payload is None:
            missing_files.append(scene_id)
            continue
        for key in expected_keys:
            if key not in payload:
                missing_keys.append((scene_id, key))
                continue
            value = np.asarray(payload[key])
            shapes.setdefault(key, (value.shape, str(value.dtype)))
            if value.dtype.kind == "f":
                as_float = value.astype(np.float64, copy=False)
                if not np.isfinite(as_float).all():
                    bad_values.append((scene_id, key))
        cache[scene_id] = np.asarray(payload["pred_bboxes"], dtype=np.float64)

    print(f"  scenes present            : {len(cache)}/{len(scene_ids)}")
    print(f"  files missing             : {len(missing_files)}")
    print(f"  key/scene pairs missing   : {len(missing_keys)}")
    print(f"  tensors with NaN or Inf   : {len(bad_values)}")
    for key in sorted(shapes):
        shape, dtype = shapes[key]
        print(f"    {key:32s} {str(shape):18s} {dtype}")
    for scene_id in missing_files[:5]:
        print(f"    MISSING FILE {scene_id}")
    for scene_id, key in missing_keys[:5]:
        print(f"    MISSING KEY  {scene_id}/{key}")
    for scene_id, key in bad_values[:5]:
        print(f"    NON-FINITE   {scene_id}/{key}")

    integrity_ok = not (missing_files or missing_keys or bad_values)

    predictions_path = os.path.join(REPO, args.predictions)
    if not os.path.isfile(predictions_path):
        print(f"\n[2] SKIPPED -- {args.predictions} not found, so there are no gt boxes "
              f"to measure against.")
        return 0 if integrity_ok else 1

    predictions = load_predictions(predictions_path)
    print(f"\n[2] Recall ceiling from the cached proposals\n" + "-" * 84)

    # load_predictions returns a flat dict keyed by (scene_id, object_id, ann_id); join()
    # is the one place that key convention is written down, so reuse it.
    joined, no_prediction = join(predictions, records)
    if no_prediction:
        print(f"  {no_prediction} annotation(s) have no prediction and are excluded")

    ceilings, achieved, rows = [], [], 0
    for row in joined:
        scene_id = row["scene_id"]
        if scene_id not in cache:
            continue
        gt = np.asarray(row["_prediction"]["gt_bbox"], dtype=np.float64)
        ceilings.append(float(batch_max_iou(cache[scene_id], gt).max()))
        achieved.append(float(row["iou"]))
        rows += 1

    if not rows:
        print("  no annotation could be matched to both a cached scene and a prediction")
        return 1

    ceilings = np.asarray(ceilings)
    achieved = np.asarray(achieved)

    table_rows = []
    for threshold in THRESHOLDS:
        table_rows.append([
            f"@{threshold}",
            f"{100 * (ceilings >= threshold).mean():.2f}",
            f"{100 * (achieved >= threshold).mean():.2f}",
            f"{100 * ((ceilings >= threshold).mean() - (achieved >= threshold).mean()):+.2f}",
        ])
    table = md_table(["IoU", "cache ceiling %", "achieved %", "headroom pp"], table_rows)
    print("\n" + table)
    print(f"\n  annotations compared     : {rows}")
    print(f"  mean best-proposal IoU   : {ceilings.mean():.4f}")
    print(f"  median best-proposal IoU : {np.median(ceilings):.4f}")

    violations = int((achieved > ceilings + 1e-6).sum())
    print(f"  achieved > ceiling       : {violations} ({100 * violations / rows:.2f}%)"
          f"   <- resampling noise; large values mean a different checkpoint")

    print("\n" + "=" * 84)
    print("VERDICT")
    print("=" * 84)
    problems = []
    if not integrity_ok:
        problems.append("the cache is incomplete or malformed -- see section 1")
    for threshold in THRESHOLDS:
        ceiling = (ceilings >= threshold).mean()
        got = (achieved >= threshold).mean()
        if ceiling < got:
            problems.append(
                f"at IoU {threshold} the cached proposals top out at "
                f"{100 * ceiling:.2f}% but the run reports {100 * got:.2f}%. The fusion "
                f"network can only return one of these proposals, so the cache cannot "
                f"be the source of that number.")
    if violations / rows > 0.25:
        problems.append(
            f"{100 * violations / rows:.1f}% of annotations beat their own cached "
            f"ceiling, which is far more than resampling explains. Check that "
            f"meta.json's checkpoint matches the run that produced these predictions.")

    if problems:
        for index, item in enumerate(problems, 1):
            print(f"  {index}. {item}")
        print("\n  Do not run ablations on this cache.")
    else:
        print("  The cache is complete, finite, correctly shaped, and leaves "
              f"{100 * ((ceilings >= 0.25).mean() - (achieved >= 0.25).mean()):.1f} pp of "
              f"headroom above the reported Acc@0.25.")
        print("  Nothing here proves numerical equivalence -- that is "
              "validate_scene_cache.py's\n  job and it needs a GPU -- but every failure "
              "mode this can see is absent.")

    report = {
        "split": args.split, "root": args.root, "scenes": len(cache),
        "annotations": rows, "integrity_ok": integrity_ok,
        "missing_files": missing_files, "missing_keys": missing_keys[:50],
        "non_finite": bad_values[:50],
        "ceiling": {str(t): float((ceilings >= t).mean()) for t in THRESHOLDS},
        "achieved": {str(t): float((achieved >= t).mean()) for t in THRESHOLDS},
        "mean_best_iou": float(ceilings.mean()),
        "achieved_above_ceiling": violations,
        "problems": problems,
    }
    out_dir = os.path.join(REPO, args.out_dir)
    ensure_dir(out_dir)
    path = save_json(report, os.path.join(out_dir, f"scene_cache_audit_{args.split}.json"))
    print(f"\nwrote {os.path.relpath(path, REPO)}")
    return 1 if problems else 0


# ---- cached-eval-cpu ----

_DOC_CACHED_EVAL_CPU = """
Evaluate the cached path on a machine with no usable GPU, and time it.

    python experiments/measurements.py cached-eval-cpu --num-samples 128
    python experiments/measurements.py cached-eval-cpu --num-samples 0      # whole split

validate_scene_cache.py is authoritative but needs CUDA -- get_loss and get_eval contain 32
hardcoded .cuda() calls between them. This answers the narrower "does the cached path
reproduce the reported accuracy?" without entering them, via two shims confined to this
file: torch.Tensor.cuda becomes identity, and proposal selection is reproduced rather than
imported from lib/eval_helper.py.

The accuracy will not match end-to-end to 1e-4 -- deterministic vs random point subsampling
-- but lands within a point or two when the cache is sound.
"""

def build_args(cli):
    """The eval defaults ScanRefer_eval.py uses, as a plain namespace."""
    return argparse.Namespace(
        gpu="0", batch_size=cli.batch_size, lang_num_max=1,
        num_points=cli.num_points, num_proposals=cli.num_proposals, num_scenes=-1,
        seed=42, no_height=False, no_lang_cls=False, no_nms=True,
        use_color=True, use_normal=True, use_multiview=False, use_bidir=False,
        use_train=False, use_oracle=False, use_cat_rand=False, use_best=False,
        reference=True, detection=False, no_detection=True, no_reference=False,
        model_tag="", lang_input="glove+parse", detector="VN", GF_path=None,
        do_not_remove_empty_box=False, use_scanrefer_scenes=False,
        width=1, num_target=cli.num_proposals, sampling="kps",
        nhead=8, num_decoder_layers=6, dim_feedforward=2048,
        transformer_dropout=0.1, transformer_activation="relu",
        self_position_embedding="loc_learned", cross_position_embedding="xyz_learned",
        size_cls_agnostic=False, bn_momentum=0.1, syncbn=False,
        use_checkpoint="", use_pretrained=None,
        # ablation flags
        use_cached_scenes=True, cached_scenes_root=cli.cached_scenes_root,
        deterministic_subsample=True, subsample_salt=cli.subsample_salt,
        parsing_folder=cli.parsing_folder, disable_copy_paste=None, keep_checkpoint=True,
        fusion_variant=cli.fusion_variant, no_strict_checkpoint=False,
    )


def chunk_scanrefer(scanrefer, lang_num_max=1):
    """Reproduces get_scanrefer()'s chunking from scripts/ScanRefer_eval.py."""
    chunks, current, scene_id = [], [], ""
    for data in scanrefer:
        if scene_id != data["scene_id"]:
            scene_id = data["scene_id"]
            if current:
                chunks.append(current)
            current = []
        if len(current) >= lang_num_max:
            chunks.append(current)
            current = []
        current.append(data)
    if current:
        chunks.append(current)
    return chunks


def _cpu_aabb_iou(a, b):
    a_low, a_high = a.min(axis=0), a.max(axis=0)
    b_low, b_high = b.min(axis=0), b.max(axis=0)
    overlap = np.clip(np.minimum(a_high, b_high) - np.maximum(a_low, b_low), 0.0, None)
    intersection = float(np.prod(overlap))
    union = (float(np.prod(a_high - a_low)) + float(np.prod(b_high - b_low))
             - intersection)
    return intersection / union if union > 0 else 0.0


def _eval_once(model, loader, chunks, predictions, device):
    """One pass over the loader. Returns (ious, keys, seconds), aligned with `chunks`.

    Proposal selection is reproduced here rather than imported from lib/eval_helper.py,
    which is CUDA-only. The winning box comes from the cache's pred_bboxes and the ground
    truth from predictions.p, so nothing needs the detector.
    """
    import torch

    ious, keys, position = [], [], 0
    started = time.time()
    with torch.no_grad():
        for batch in loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device)
            batch = model(batch)

            cluster_ref = batch["cluster_ref"]                       # (B, num_proposal)
            objectness = batch["objectness_scores"]                  # (B, num_proposal, 2)
            mask = (torch.argmax(objectness, 2) == 1).float()
            selected = torch.argmax(cluster_ref * mask, 1).cpu().numpy()
            boxes = batch["pred_bboxes"].cpu().numpy()               # (B, num_proposal, 8, 3)

            for row in range(len(selected)):
                if position >= len(chunks):
                    break
                record = chunks[position][0]
                key = (str(record["scene_id"]), str(record["object_id"]),
                       str(record["ann_id"]))
                truth = predictions.get(key)
                if truth is not None:
                    ious.append(_cpu_aabb_iou(boxes[row][selected[row]],
                                         np.asarray(truth["gt_bbox"], dtype=np.float64)))
                    keys.append(key)
                position += 1
    return ious, keys, time.time() - started


def _add_cached_eval_cpu_args(parser):
    parser.add_argument("--folder", default="2024-12-18_20-40-38_3DVG-FIXED",
                        help="run under outputs/ holding model.pth and predictions.p")
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=128,
                        help="annotations to evaluate; 0 = the whole split")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-points", dest="num_points", type=int, default=40000)
    parser.add_argument("--num-proposals", dest="num_proposals", type=int, default=256)
    parser.add_argument("--cached-scenes-root", dest="cached_scenes_root",
                        default="cached_scenes")
    parser.add_argument("--subsample-salt", dest="subsample_salt", default="v1")
    parser.add_argument("--parsing-folder", dest="parsing_folder",
                        default="final_parsing_tokenized")
    parser.add_argument("--fusion-variant", dest="fusion_variant", default="original",
                        choices=["current", "original"],
                        help="Which MatchModule to build. Defaults to 'original', which "
                             "is what the 2024-12-18 checkpoint was trained with.")
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/diagnostics")
    return parser


def run_cached_eval_cpu(args):
    cli = args

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            probe = torch.randn(8, 8, device="cuda")
            (probe @ probe).sum().item()
            device = torch.device("cuda")
        except Exception:
            pass

    if device.type == "cpu":
        # Five call sites in the cached forward path hardcode .cuda(); everything is
        # already on CPU, so identity is correct.
        torch.Tensor.cuda = lambda self, *a, **k: self          # noqa: E731

    from data.scannet.model_util_scannet import ScannetDatasetConfig
    from experiments.ablation import ablation_config, ablation_hooks
    from experiments.analysis import ensure_dir, join, load_predictions, \
        load_scanrefer, md_table, save_json
    from lib.config import CONF

    print("=" * 84)
    print(f"Cached-path evaluation on {device.type.upper()}")
    print("=" * 84)

    args = build_args(cli)
    args = ablation_config.apply(args)
    print("\n" + ablation_config.describe())

    records = load_scanrefer("val", "data")
    chunks = chunk_scanrefer(records, lang_num_max=1)
    if cli.num_samples:
        chunks = chunks[:cli.num_samples]
    flat = [c[0] for c in chunks]
    scene_list = sorted({c[0]["scene_id"] for c in chunks})
    print(f"\nevaluating {len(chunks)} annotation(s) across {len(scene_list)} scene(s)")

    started = time.time()
    dataset = ablation_hooks.build_dataset(
        args=args, scanrefer=flat, scanrefer_new=chunks,
        scanrefer_all_scene=scene_list, split="val",
        num_points=args.num_points, use_color=args.use_color,
        use_height=(not args.no_height), use_normal=args.use_normal,
        use_multiview=args.use_multiview, lang_num_max=1)
    dataset_seconds = time.time() - started
    print(f"[timing] dataset ready in {dataset_seconds:.1f}s "
          f"(GloVe table dominates this)")

    # shuffle=False so batch order matches `chunks`, which is how each row is mapped
    # back to its annotation.
    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False)

    model = ablation_hooks.build_model(
        args=args, num_class=ScannetDatasetConfig().num_class,
        num_heading_bin=ScannetDatasetConfig().num_heading_bin,
        num_size_cluster=ScannetDatasetConfig().num_size_cluster,
        mean_size_arr=ScannetDatasetConfig().mean_size_arr,
        input_feature_dim=(int(args.use_multiview) * 128 + int(args.use_normal) * 3
                           + int(args.use_color) * 3 + int(not args.no_height)),
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir, no_reference=False,
        dataset_config=ScannetDatasetConfig()).to(device)

    checkpoint_path = os.path.join(CONF.PATH.OUTPUT, cli.folder, cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print(f"\ncheckpoint not found: {checkpoint_path}")
        return 1
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # STRICT_CHECKPOINT raises when a parameter would be left random. That is a
    # configuration problem with a known fix, so refuse cleanly rather than traceback.
    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        print("\n" + "!" * 84)
        print(f"ABORT: this checkpoint does not fit --fusion-variant "
              f"{cli.fusion_variant!r}.")
        print("!" * 84)
        print(f"\n{error}")
        return 2
    model.eval()

    predictions = load_predictions(
        os.path.join(REPO, "outputs", cli.folder, "predictions.p"))

    ious, _keys, forward_seconds = _eval_once(model, loader, chunks, predictions, device)

    if not ious:
        print("\nno annotation could be scored")
        return 1

    ious = np.asarray(ious)
    rate = len(ious) / forward_seconds
    total = len(chunk_scanrefer(records, 1))

    print(f"\n[timing] {len(ious)} annotations in {forward_seconds:.1f}s "
          f"= {rate:.2f}/s")
    print(f"[timing] full val ({total} annotations) would take about "
          f"{(total / rate) / 60:.0f} min on this {device.type.upper()}, "
          f"plus {dataset_seconds:.0f}s of setup")

    reference = {}
    joined, _ = join(predictions, records)
    end_to_end = np.asarray([row["iou"] for row in joined])

    rows = []
    for threshold in (0.25, 0.5):
        cached = 100 * (ious >= threshold).mean()
        full = 100 * (end_to_end >= threshold).mean()
        reference[str(threshold)] = {"cached_subset": cached, "end_to_end_full": full}
        rows.append([f"Acc@{threshold}", f"{cached:.2f}", f"{full:.2f}",
                     f"{cached - full:+.2f}"])
    print("\n" + md_table(
        [f"metric", f"cached ({len(ious)})", f"end-to-end (all {len(end_to_end)})",
         "diff pp"], rows))

    print("\nThe two columns are not directly comparable: the left is a subset evaluated "
          "under\nthe deterministic subsample, the right is the whole split under random "
          "subsampling.\nA gap of a point or two is expected; a large one is not.")

    ensure_dir(os.path.join(REPO, cli.out_dir))
    path = save_json(
        {"device": device.type, "folder": cli.folder, "annotations": len(ious),
         "seconds_forward": forward_seconds, "seconds_dataset": dataset_seconds,
         "rate_per_second": rate, "projected_full_val_minutes": (total / rate) / 60,
         "accuracy": reference, "mean_iou": float(ious.mean())},
        os.path.join(REPO, cli.out_dir, "cached_eval_cpu.json"))
    print(f"\nwrote {os.path.relpath(path, REPO)}")
    return 0


# ---- online-latency ----

_DOC_ONLINE_LATENCY = """
End-to-end latency with the parser inside the pipeline: sentence -> box.

    python experiments/measurements.py online-latency --num_samples 50
    python experiments/measurements.py online-latency --model qwen2.5 --variant cached

GPU. Every other latency number here assumes the parse cache is precomputed, which is true
for every benchmark table. This measures the other case -- a sentence the model has never
seen, parsed live by Qwen2.5 before grounding -- which is the number that matters if the
method is deployed rather than benchmarked.

Four numbers per query, timed separately and then together:

    parse     the LM generating the three fields, batch of one, greedy
    encode    tokenising that output and looking up its GloVe rows
    ground    one forward pass of the grounding network
    end2end   the same three as one wall-clock block

The offline row re-runs `ground` alone with a parse read from the cache, so the table
answers directly what putting the parser online costs per query. One-off setup -- LM
weights, the ~933 MB GloVe pickle -- is timed and reported apart from per-query latency,
never inside it.

This is the one place where summing parse and grounding latency is correct. It is the
online/deployment number and must be labelled as such, not as the cost of a benchmark
table. See details.md §6.
"""

#: Fixed by _transform_parsed in lib/dataset.py.
_ONLINE_CAPS = {"target": 7, "adjectives": 17, "neighbors": 75}
_GLOVE_DIM = 300


def _online_embed(tokens, num_token, glove):
    """Tokens -> (num_token, 300) GloVe rows, the same fill lib/dataset.py uses."""
    out = np.zeros((num_token, _GLOVE_DIM), dtype=np.float32)
    for i in range(min(len(tokens), num_token)):
        out[i] = glove.get(tokens[i], glove["unk"])
    return out


def _online_load_glove():
    import pickle
    path = os.path.join(CONF.PATH.DATA, "glove.p")
    with open(path, "rb") as f:
        return pickle.load(f)


def _online_splice(data_dict, parsed_tokens, glove, device):
    """Write one sentence's parsed fields into a batch-of-one data_dict."""
    import torch
    for field, cap in _ONLINE_CAPS.items():
        toks = parsed_tokens[field]
        emb = _online_embed(toks, cap, glove)
        data_dict[field] = torch.from_numpy(emb).unsqueeze(0).unsqueeze(0).to(device)
        data_dict[{"target": "tgt_len", "adjectives": "adj_len",
                   "neighbors": "ngh_len"}[field]] = torch.tensor(
            [[min(len(toks), cap)]], dtype=torch.int64, device=device)
    return data_dict


def _add_online_latency_args(parser):
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Distinct val descriptions to time (each parsed once).")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="qwen2.5",
                        help="Parser alias or HuggingFace id; see parsing_smalllm.py.")
    parser.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=96)
    parser.add_argument("--variant", default="cached", choices=["cached", "end2end"],
                        help="Which grounding network to time. 'end2end' includes the "
                             "detection branch and is the cost of the deployed method.")
    parser.add_argument("--offline-folder", dest="offline_folder",
                        default="smalllm_qwen2.5_parsing_tokenized",
                        help="Parse cache for the offline comparison row; skipped if absent.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--output", default=os.path.join("outputs", "complexity"))
    # shape of the synthetic batch, matching the `model` subcommand
    parser.add_argument("--num_points", type=int, default=40000)
    parser.add_argument("--num_proposals", type=int, default=256)
    parser.add_argument("--lang_num_max", type=int, default=1)
    parser.add_argument("--detector", default="VN", choices=["VN", "GF"])
    parser.add_argument("--no_height", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_multiview", action="store_true")
    parser.add_argument("--use_bidir", action="store_true")
    parser.add_argument("--no_lang_cls", action="store_true")
    parser.add_argument("--lang_input", default="glove+parse")
    parser.add_argument("--GF_path", default=None)
    return parser


def run_online_latency(args):
    import torch
    from experiments.ablation.parsing import tokenize_record
    from experiments.ablation.parsing_smalllm import (
        MODEL_ALIASES, check_requirements, cuda_is_usable, extract_fields, generate,
        load_model)

    args.input_feature_dim = (int(not args.no_height) + int(args.use_color) * 3
                              + int(args.use_normal) * 3 + int(args.use_multiview) * 128)
    args.detection = False
    args.no_reference = False

    print("=" * 78)
    print("  ONLINE PARSING + GROUNDING LATENCY")
    print("=" * 78)

    if not check_requirements(args.model):
        return 1
    if args.device == "cuda":
        usable, detail = cuda_is_usable()
        if not usable:
            print(f"ERROR: --device cuda requested but no CUDA kernel runs.\n  {detail}")
            print("  Re-run with --device cpu, or use a torch build matching this GPU.")
            return 1
        print(f"GPU: {detail}")
    # torch.device, not the string: environment(), _sync() and .to() all expect one.
    device = torch.device(args.device)

    report = {"config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
              "environment": environment(device), "setup_seconds": {}}

    # One-off setup, timed but never counted into per-query latency.
    model_id = MODEL_ALIASES.get(args.model, args.model)
    t0 = time.perf_counter()
    tokenizer, lm, kind = load_model(model_id, device, args.dtype)
    report["setup_seconds"]["parser_load"] = round(time.perf_counter() - t0, 2)
    print(f"parser: {model_id} ({kind})  loaded in "
          f"{report['setup_seconds']['parser_load']}s")

    t0 = time.perf_counter()
    glove = _online_load_glove()
    report["setup_seconds"]["glove_load"] = round(time.perf_counter() - t0, 2)
    print(f"glove : {len(glove)} vectors loaded in "
          f"{report['setup_seconds']['glove_load']}s")

    t0 = time.perf_counter()
    grounding = _cx_build_model(args, args.variant, device)
    grounding.eval()
    report["setup_seconds"]["grounding_build"] = round(time.perf_counter() - t0, 2)

    descriptions, dstats = load_val_descriptions(args.num_samples, args.seed)
    report["description_sample"] = dstats
    print(f"sample: {dstats['count']} distinct val descriptions "
          f"(mean {dstats['mean_words']} words)\n")

    base = build_data_dict(args, 1, istrain=0, device=device, variant=args.variant)

    def parse_once(sentence):
        raw = next(iter(generate([sentence], tokenizer, lm, kind, device, 1,
                                 args.max_new_tokens)))
        fields, status = extract_fields(raw)
        return tokenize_record(fields), status

    # Warm up both models, so neither pays first-call cost in the timed loop.
    for sentence in descriptions[:args.warmup]:
        toks, _ = parse_once(sentence)
        with torch.no_grad():
            grounding(dict(_online_splice(dict(base), toks, glove, device)))
    _sync(device)

    parse_t, encode_t, ground_t, e2e_t, statuses = [], [], [], [], []
    with torch.no_grad():
        for sentence in descriptions:
            _sync(device)
            t_start = time.perf_counter()

            t0 = time.perf_counter()
            toks, status = parse_once(sentence)
            _sync(device)
            parse_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            data_dict = _online_splice(dict(base), toks, glove, device)
            encode_t.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            grounding(dict(data_dict))
            _sync(device)
            ground_t.append(time.perf_counter() - t0)

            e2e_t.append(time.perf_counter() - t_start)
            statuses.append(status)

    report["online"] = {"parse": stats_ms(parse_t), "encode": stats_ms(encode_t),
                        "ground": stats_ms(ground_t), "end2end": stats_ms(e2e_t)}
    report["parse_status"] = {s: statuses.count(s) for s in sorted(set(statuses))}
    report["malformed_pct"] = round(
        100.0 * statuses.count("malformed") / max(len(statuses), 1), 2)

    # Offline row: the same grounding pass, parse read from the cache.
    cache_path = os.path.join("data_parsing", args.offline_folder,
                              "tokenized_parsed_result_val.json")
    if os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        records = [r for scene in cache.values() for obj in scene.values()
                   for r in obj.values()][:len(descriptions)]
        for r in records[:args.warmup]:
            grounding_input = _online_splice(dict(base), r, glove, device)
            with torch.no_grad():
                grounding(dict(grounding_input))
        _sync(device)
        off_t = []
        with torch.no_grad():
            for r in records:
                _sync(device)
                t0 = time.perf_counter()
                grounding(dict(_online_splice(dict(base), r, glove, device)))
                _sync(device)
                off_t.append(time.perf_counter() - t0)
        report["offline"] = {"ground_plus_encode": stats_ms(off_t),
                             "parse": {"mean_ms": 0.0,
                                       "note": "precomputed; zero at query time"},
                             "source": cache_path}
    else:
        report["offline"] = {"skipped": f"no parse cache at {cache_path}"}

    online_total = report["online"]["end2end"]["mean_ms"]
    print(f"{'leg':<12}{'mean ms':>10}{'p50':>9}{'p95':>9}")
    for leg in ("parse", "encode", "ground", "end2end"):
        s = report["online"][leg]
        print(f"{leg:<12}{s['mean_ms']:>10.2f}{s['p50_ms']:>9.2f}{s['p95_ms']:>9.2f}")
    if "ground_plus_encode" in report.get("offline", {}):
        off = report["offline"]["ground_plus_encode"]["mean_ms"]
        report["online_overhead_ms"] = round(online_total - off, 3)
        report["online_overhead_x"] = round(online_total / off, 2) if off else None
        print(f"\noffline (cached parse): {off:.2f} ms")
        print(f"online costs {report['online_overhead_ms']:.2f} ms more per query "
              f"({report['online_overhead_x']}x)")
    print(f"\nmalformed parses: {report['malformed_pct']}%")

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "online_latency_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nwrote {out_path}")
    return 0


# ---- dist-weights-sweep ----

_DOC_DIST_WEIGHTS_SWEEP = """
Sweep the geometric prior that biases the fusion self-attention.

    python experiments/measurements.py dist-weights-sweep --num-samples 200
    python experiments/measurements.py dist-weights-sweep --num-samples 64 \
        --arms baseline swapped

MatchModule builds a (B, 4, K, K) tensor and adds it to the pre-softmax logits of a 4-head
attention, so channel c biases head c and nothing else. Two of the four channels are
zeros -- heads 2 and 3 carry no geometric prior at all. The two live signals are on very
different scales:

    channel 0   1/(d+1e-2), normalised over dim=2   positive, about O(1/K)
    channel 1   -dist, raw                          0 to about -6, metres

This evaluates a chosen number of val annotations once per arm, holding everything else
fixed, and compares the arms pairwise:

    baseline    head0 = norm_inv   head1 = -dist      (what ships)
    both_dist   head0 = -dist      head1 = -dist
    both_norm   head0 = norm_inv   head1 = norm_inv
    swapped     head0 = -dist      head1 = norm_inv

Every arm is a re-selection of the two live channels, so it is applied by intercepting
attention_weights inside ScaledDotProductAttention -- which covers both duplicated
torch.cat sites and both fusion variants, with no edit to models/.

The checkpoint was trained with the baseline pairing, so a swapped arm measures how
sensitive the trained head is to a changed prior, not the accuracy a retrained model would
reach.
"""

#: (head0, head1) picked from the two live channels of the (B, 4, K, K) tensor.
_ARMS = {
    "baseline":  (0, 1),
    "both_dist": (1, 1),
    "both_norm": (0, 0),
    "swapped":   (1, 0),
}


def _install_arm(arm):
    """Re-select the two live channels on the way into every attention call."""
    import torch
    from models.transformer.attention import ScaledDotProductAttention

    if not hasattr(ScaledDotProductAttention, "_sweep_orig_forward"):
        ScaledDotProductAttention._sweep_orig_forward = ScaledDotProductAttention.forward
    original = ScaledDotProductAttention._sweep_orig_forward
    head0, head1 = _ARMS[arm]

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None,
                way="mul"):
        # Cross-attention passes attention_weights=None, so only the self-attention
        # stacks that carry the geometric prior are touched.
        if attention_weights is not None and attention_weights.shape[1] == 4:
            zeros = torch.zeros_like(attention_weights[:, :1])
            attention_weights = torch.cat(
                [attention_weights[:, head0:head0 + 1],
                 attention_weights[:, head1:head1 + 1], zeros, zeros], dim=1)
        return original(self, queries, keys, values, attention_mask, attention_weights, way)

    ScaledDotProductAttention.forward = forward


def _restore_attention():
    from models.transformer.attention import ScaledDotProductAttention
    if hasattr(ScaledDotProductAttention, "_sweep_orig_forward"):
        ScaledDotProductAttention.forward = \
            ScaledDotProductAttention._sweep_orig_forward
        del ScaledDotProductAttention._sweep_orig_forward


def _add_dist_weights_sweep_args(parser):
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=200,
                        help="annotations to evaluate per arm; 0 = the whole split")
    parser.add_argument("--arms", nargs="+", default=list(_ARMS),
                        choices=list(_ARMS),
                        help="which arms to run; 'baseline' is always included first.")
    parser.add_argument("--folder", default="2024-12-18_20-40-38_3DVG-FIXED",
                        help="run under outputs/ holding model.pth and predictions.p")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=4)
    parser.add_argument("--num-points", dest="num_points", type=int, default=40000)
    parser.add_argument("--num-proposals", dest="num_proposals", type=int, default=256)
    parser.add_argument("--cached-scenes-root", dest="cached_scenes_root",
                        default="cached_scenes")
    parser.add_argument("--subsample-salt", dest="subsample_salt", default="v1")
    parser.add_argument("--parsing-folder", dest="parsing_folder",
                        default="final_parsing_tokenized")
    parser.add_argument("--fusion-variant", dest="fusion_variant", default="original",
                        choices=["current", "original"],
                        help="Which MatchModule to build. Defaults to 'original', which "
                             "is what the 2024-12-18 checkpoint was trained with.")
    parser.add_argument("--checkpoint", default="model.pth")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="bootstrap resamples for the CI on each delta.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/diagnostics")
    return parser


def run_dist_weights_sweep(args):
    cli = args

    import torch
    from torch.utils.data import DataLoader

    device = torch.device("cpu")
    if torch.cuda.is_available():
        try:
            probe = torch.randn(8, 8, device="cuda")
            (probe @ probe).sum().item()
            device = torch.device("cuda")
        except Exception:
            pass

    if device.type == "cpu":
        # The cached forward path hardcodes .cuda(); everything is already on CPU.
        torch.Tensor.cuda = lambda self, *a, **k: self          # noqa: E731

    from data.scannet.model_util_scannet import ScannetDatasetConfig
    from experiments.ablation import ablation_config, ablation_hooks
    from experiments.analysis import (bootstrap_ci, ensure_dir, load_predictions,
                                      load_scanrefer, mcnemar, md_table, save_json)
    from lib.config import CONF

    arms = ["baseline"] + [a for a in cli.arms if a != "baseline"]

    print("=" * 84)
    print(f"dist_weights head sweep on {device.type.upper()}")
    print("=" * 84)

    args = build_args(cli)
    args = ablation_config.apply(args)

    records = load_scanrefer("val", "data")
    chunks = chunk_scanrefer(records, lang_num_max=1)
    if cli.num_samples:
        chunks = chunks[:cli.num_samples]
    flat = [c[0] for c in chunks]
    scene_list = sorted({c[0]["scene_id"] for c in chunks})
    print(f"\n{len(arms)} arm(s) x {len(chunks)} annotation(s) "
          f"across {len(scene_list)} scene(s)")

    # Dataset, loader, model and checkpoint are built ONCE; only the forward pass repeats.
    started = time.time()
    dataset = ablation_hooks.build_dataset(
        args=args, scanrefer=flat, scanrefer_new=chunks,
        scanrefer_all_scene=scene_list, split="val",
        num_points=args.num_points, use_color=args.use_color,
        use_height=(not args.no_height), use_normal=args.use_normal,
        use_multiview=args.use_multiview, lang_num_max=1)
    dataset_seconds = time.time() - started
    print(f"[timing] dataset ready in {dataset_seconds:.1f}s")

    loader = DataLoader(dataset, batch_size=cli.batch_size, shuffle=False)

    model = ablation_hooks.build_model(
        args=args, num_class=ScannetDatasetConfig().num_class,
        num_heading_bin=ScannetDatasetConfig().num_heading_bin,
        num_size_cluster=ScannetDatasetConfig().num_size_cluster,
        mean_size_arr=ScannetDatasetConfig().mean_size_arr,
        input_feature_dim=(int(args.use_multiview) * 128 + int(args.use_normal) * 3
                           + int(args.use_color) * 3 + int(not args.no_height)),
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir, no_reference=False,
        dataset_config=ScannetDatasetConfig()).to(device)

    checkpoint_path = os.path.join(CONF.PATH.OUTPUT, cli.folder, cli.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print(f"\ncheckpoint not found: {checkpoint_path}")
        return 1
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        print(f"\nABORT: this checkpoint does not fit --fusion-variant "
              f"{cli.fusion_variant!r}.\n\n{error}")
        return 2
    model.eval()

    predictions = load_predictions(
        os.path.join(REPO, "outputs", cli.folder, "predictions.p"))

    results = {}
    try:
        for arm in arms:
            head0, head1 = _ARMS[arm]
            _install_arm(arm)
            ious, keys, seconds = _eval_once(model, loader, chunks, predictions, device)
            if not ious:
                print("\nno annotation could be scored")
                return 1
            results[arm] = {"ious": np.asarray(ious), "keys": keys, "seconds": seconds}
            print(f"  {arm:10s} head0=ch{head0} head1=ch{head1}  "
                  f"{len(ious)} annotations in {seconds:.1f}s  "
                  f"Acc@0.25 {100 * (np.asarray(ious) >= 0.25).mean():.2f}%")
    finally:
        _restore_attention()

    base = results["baseline"]
    if any(results[a]["keys"] != base["keys"] for a in arms):
        print("\nABORT: the arms did not score the same annotations.")
        return 1

    report, rows, paired_rows = {}, [], []
    base_hit = base["ious"] >= 0.25

    for arm in arms:
        ious = results[arm]["ious"]
        hit25, hit50 = ious >= 0.25, ious >= 0.5
        acc25, acc50 = 100 * hit25.mean(), 100 * hit50.mean()
        entry = {"head0_channel": _ARMS[arm][0], "head1_channel": _ARMS[arm][1],
                 "annotations": int(len(ious)), "seconds": results[arm]["seconds"],
                 "acc_0.25": acc25, "acc_0.5": acc50, "mean_iou": float(ious.mean())}
        rows.append([arm, f"{acc25:.2f}", f"{acc50:.2f}", f"{ious.mean():.4f}",
                     "-" if arm == "baseline" else f"{acc25 - 100 * base_hit.mean():+.2f}"])

        if arm != "baseline":
            a_only, b_only, chi2, p = mcnemar(base_hit.tolist(), hit25.tolist())
            changed = int((hit25 != base_hit).sum())
            delta = acc25 - 100 * base_hit.mean()
            point, low, high = bootstrap_ci(
                lambda idx: 100 * (hit25[idx].mean() - base_hit[idx].mean()),
                len(ious), num_resamples=cli.bootstrap, seed=cli.seed)
            entry.update({"vs_baseline": {
                "delta_acc_0.25": delta, "ci_low": low, "ci_high": high,
                "mcnemar_p": p, "mcnemar_chi2": chi2,
                "broke": int(a_only), "fixed": int(b_only), "changed": changed,
                "significant": bool(low > 0 or high < 0)}})
            paired_rows.append([arm, f"{delta:+.2f}", f"[{low:+.2f}, {high:+.2f}]",
                                f"{p:.4f}", str(changed), str(int(a_only)),
                                str(int(b_only)),
                                "yes" if (low > 0 or high < 0) else "no"])
        report[arm] = entry

    print("\n" + md_table(["arm", "Acc@0.25", "Acc@0.5", "mean IoU", "delta pp"], rows))
    if paired_rows:
        print("\nPaired against baseline, same " + str(len(base["ious"])) +
              " annotations:\n")
        print(md_table(["arm", "delta pp", "bootstrap 95% CI", "McNemar p",
                        "changed", "broke", "fixed", "outside CI"], paired_rows))

    moved = [a for a in report if a != "baseline"
             and report[a].get("vs_baseline", {}).get("significant")]
    print("\nVERDICT: " + (
        f"{', '.join(moved)} moved accuracy outside the 95% CI."
        if moved else
        "no arm moved accuracy outside its 95% CI at this sample size."))
    print("\nHeads 2 and 3 receive zeros in every arm; only the two live heads are swept.")
    print("The checkpoint was trained with the baseline pairing, so a swapped arm measures")
    print("sensitivity of the trained head to a changed prior, not a retrained optimum.")

    ensure_dir(os.path.join(REPO, cli.out_dir))
    path = save_json(
        {"device": device.type, "folder": cli.folder, "fusion_variant": cli.fusion_variant,
         "annotations": int(len(base["ious"])), "seconds_dataset": dataset_seconds,
         "bootstrap": cli.bootstrap, "seed": cli.seed, "arms": report,
         "channel_meaning": {"0": "1/(d+1e-2) normalised over dim=2",
                             "1": "-dist, raw", "2": "zeros", "3": "zeros"}},
        os.path.join(REPO, cli.out_dir, "dist_weights_sweep.json"))
    print(f"\nwrote {os.path.relpath(path, REPO)}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python experiments/measurements.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    for key, doc, adder, runner in (
            ("model", _DOC_MODEL, _add_model_args, run_model),
            ("parsing-latency", _DOC_PARSING_LATENCY, _add_parsing_latency_args,
             run_parsing_latency),
            ("online-latency", _DOC_ONLINE_LATENCY, _add_online_latency_args,
             run_online_latency),
            ("validate-scene-cache", _DOC_VALIDATE_SCENE_CACHE,
             _add_validate_scene_cache_args, run_validate_scene_cache),
            ("audit-scene-cache", _DOC_AUDIT_SCENE_CACHE, _add_audit_scene_cache_args,
             run_audit_scene_cache),
            ("cached-eval-cpu", _DOC_CACHED_EVAL_CPU, _add_cached_eval_cpu_args,
             run_cached_eval_cpu),
            ("dist-weights-sweep", _DOC_DIST_WEIGHTS_SWEEP, _add_dist_weights_sweep_args,
             run_dist_weights_sweep)):
        p = sub.add_parser(key, help=doc.strip().splitlines()[0], description=doc,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        adder(p)
        p.set_defaults(func=runner, _parser=p)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
