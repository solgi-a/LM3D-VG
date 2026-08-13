
import argparse
import gc
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.scannet.model_util_scannet import ScannetDatasetConfig
from experiments.ablation.cached_scenes import (
    SCENE_CACHE_KEYS_OPTIONAL,
    SCENE_CACHE_KEYS_REQUIRED,
    CachedSceneDataset,
    cache_dir_for,
    cache_path_for,
    save_scene_cache,
    write_meta,
)
from lib.config import CONF
from models.refnet import RefNet

DC = ScannetDatasetConfig()


def _load_scanrefer(split):
    path = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{split}.json")
    with open(path) as f:
        return json.load(f)


def _one_chunk_per_scene(scanrefer, num_scenes=-1):
    seen, chunks = set(), []
    for data in scanrefer:
        scene_id = data["scene_id"]
        if scene_id in seen:
            continue
        seen.add(scene_id)
        chunks.append([data])
        if num_scenes != -1 and len(chunks) >= num_scenes:
            break
    return chunks


def build_dataset(args, split, todo_scene_ids=None):
    scanrefer = _load_scanrefer(split)
    chunks = _one_chunk_per_scene(scanrefer, args.num_scenes)

    if todo_scene_ids is not None:
        todo = set(todo_scene_ids)
        chunks = [c for c in chunks if c[0]["scene_id"] in todo]

    scanrefer = [chunk[0] for chunk in chunks]
    scene_list = sorted({c[0]["scene_id"] for c in chunks})

    dataset = CachedSceneDataset(
        args=args,
        scanrefer=scanrefer,
        scanrefer_new=chunks,
        scanrefer_all_scene=scene_list,
        split=split,
        num_points=args.num_points,
        use_height=(not args.no_height),
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        lang_num_max=1,
        augment=False,
        shuffle=False,
        use_cache=False,
        deterministic=True,
        subsample_salt=args.subsample_salt,
        lazy_scene_data=True,
        lazy_maxsize=args.lazy_maxsize,
    )
    return dataset, chunks


def _pending_scenes(args, split):
    scanrefer = _load_scanrefer(split)
    all_ids = [c[0]["scene_id"] for c in _one_chunk_per_scene(scanrefer, args.num_scenes)]
    if not args.resume:
        return all_ids, list(all_ids)
    todo = [s for s in all_ids
            if not os.path.isfile(cache_path_for(args.output, split, s))]
    return all_ids, todo


def _fmt_hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def build_model(args, device):
    input_channels = (
        int(args.use_multiview) * 128
        + int(args.use_normal) * 3
        + int(args.use_color) * 3
        + int(not args.no_height)
    )
    model = RefNet(
        args=args,
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        use_lang_classifier=(not args.no_lang_cls),
        use_bidir=args.use_bidir,
        no_reference=True,
        dataset_config=DC,
    )

    folder = args.use_pretrained or args.use_checkpoint
    if not folder:
        raise ValueError(
            "Refusing to cache features from randomly initialised weights. "
            "Pass --use_pretrained <folder> or --use_checkpoint <folder>."
        )
    ckpt_path = os.path.join(CONF.PATH.OUTPUT, folder, "model_criteria_25.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"loading detector weights from {ckpt_path}")
    weights = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = model.state_dict()
    loaded, skipped = [], []
    for key, value in weights.items():
        if key in state and state[key].shape == value.shape:
            state[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(state)

    vision_prefixes = ("backbone_net.", "vgen.", "proposal.")
    n_vision = sum(1 for k in state if k.startswith(vision_prefixes))
    missing_vision = [k for k in state if k.startswith(vision_prefixes) and k not in loaded]
    print(f"loaded {len(loaded)} tensors ({n_vision} detection-branch), "
          f"skipped {len(skipped)} unused (lang/match) from the checkpoint")
    if missing_vision:
        raise RuntimeError(
            "Detection-branch weights failed to load (missing or shape mismatch):\n  "
            + "\n  ".join(missing_vision[:20])
            + f"\n  ... ({len(missing_vision)} total)"
            + "\nCaching would capture untrained features. Aborting."
        )

    return model.to(device).eval(), ckpt_path


def dry_run(args, keys):
    print("=" * 78)
    print("DRY RUN -- no forward pass, no cache written")
    print("=" * 78)

    problems = []

    print("\n[1/4] model construction and checkpoint")
    try:
        model, ckpt_path = build_model(args, torch.device("cpu"))
        n_params = sum(p.numel() for p in model.parameters())
        print(f"      OK  RefNet built on CPU ({n_params/1e6:.1f}M params)")
        print(f"      OK  every backbone_net/vgen/proposal weight loaded from")
        print(f"          {ckpt_path}")
        del model
    except Exception as exc:
        problems.append(f"model/checkpoint: {exc}")
        print(f"      FAIL  {exc}")

    total_scenes = 0
    per_split_scenes = {}
    for split in args.splits:
        print(f"\n[2/4] dataset for split '{split}'")
        try:
            dataset, chunks = build_dataset(args, split)
            n = len(chunks)
            per_split_scenes[split] = n
            total_scenes += n
            print(f"      OK  {n} unique scenes, parse cache and glove loaded")
        except Exception as exc:
            problems.append(f"dataset[{split}]: {exc}")
            print(f"      FAIL  {exc}")
            continue

        print(f"[3/4] building one sample from '{split}' (exercises the .npy files)")
        try:
            t0 = time.time()
            sample = dataset[0]
            dt = time.time() - t0
            pc = sample["point_clouds"]
            print(f"      OK  scene {chunks[0][0]['scene_id']} in {dt:.2f}s, "
                  f"point_clouds {tuple(pc.shape)} {pc.dtype}")
        except Exception as exc:
            problems.append(f"sample[{split}]: {exc}")
            print(f"      FAIL  {exc}")

    print("\n[4/4] output estimate")
    per_scene_kb = 250 if not args.include_optional else 950
    est_mb = total_scenes * per_scene_kb / 1024
    for split, n in per_split_scenes.items():
        print(f"      {split:5s}  {n:4d} scenes -> {n * per_scene_kb / 1024:.0f} MB")
    print(f"      total  {total_scenes:4d} scenes -> {est_mb:.0f} MB "
          f"in {args.output}")
    print(f"      caching {len(keys)} tensors per scene")

    print("\n" + "=" * 78)
    if problems:
        print(f"DRY RUN FAILED -- {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("=" * 78)
        return 1
    print("DRY RUN PASSED -- re-run without --dry_run on a GPU machine to build the cache.")
    print("=" * 78)
    return 0


@torch.no_grad()
def cache_split(args, model, split, keys, device):
    out_dir = cache_dir_for(args.output, split)
    os.makedirs(out_dir, exist_ok=True)

    all_ids, todo = _pending_scenes(args, split)
    already = len(all_ids) - len(todo)
    if already:
        print(f"[{split}] resume: {already}/{len(all_ids)} scenes already cached, "
              f"{len(todo)} to go")
    if not todo:
        print(f"[{split}] nothing to do; all {len(all_ids)} scenes are cached")
        size_mb = sum(os.path.getsize(os.path.join(out_dir, f))
                      for f in os.listdir(out_dir) if f.endswith(".p")) / (1024 ** 2)
        return {"num_scenes": 0, "num_cached_total": len(all_ids),
                "elapsed_sec": 0.0, "size_mb": round(size_mb, 1),
                "tensor_shapes": {}, "skipped_existing": already}

    dataset, chunks = build_dataset(args, split, todo_scene_ids=todo)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)

    scene_ids = [c[0]["scene_id"] for c in chunks]
    written, cursor = 0, 0
    shapes_seen = {}

    t0 = time.time()
    bar = tqdm(total=len(todo), desc=f"caching {split}", unit="scene",
               dynamic_ncols=True, smoothing=0.05)
    for data_dict in loader:
        for key in data_dict:
            if torch.is_tensor(data_dict[key]):
                data_dict[key] = data_dict[key].to(device)

        data_dict = model(data_dict)

        batch = data_dict["center"].shape[0]
        for i in range(batch):
            scene_id = scene_ids[cursor + i]
            tensors = {}
            for key in keys:
                value = data_dict.get(key)
                if value is None or not torch.is_tensor(value):
                    continue
                tensors[key] = value[i]
                shapes_seen.setdefault(key, list(value[i].shape))
            save_scene_cache(cache_path_for(args.output, split, scene_id), tensors)
            written += 1
        cursor += batch

        del data_dict, tensors
        if args.gc_every and written % args.gc_every == 0:
            gc.collect()

        rate = written / max(time.time() - t0, 1e-9)
        bar.set_postfix_str(f"{rate * 60:.1f} scene/min, "
                            f"eta {_fmt_hms((len(todo) - written) / max(rate, 1e-9))}")
        bar.update(batch)
    bar.close()

    elapsed = time.time() - t0
    missing = [k for k in SCENE_CACHE_KEYS_REQUIRED if k not in shapes_seen]
    if missing:
        raise RuntimeError(
            "The detection branch did not produce these required keys: "
            + ", ".join(missing)
            + "\nThe cache would be incomplete; check models/proposal_module.py."
        )

    size_mb = sum(os.path.getsize(os.path.join(out_dir, f))
                  for f in os.listdir(out_dir) if f.endswith(".p")) / (1024 ** 2)
    print(
        f"[{split}] {written} scenes in {_fmt_hms(elapsed)} "
        f"({elapsed / max(written, 1):.2f}s/scene), {size_mb:.1f} MB on disk"
    )
    return {
        "num_scenes": written,
        "num_cached_total": already + written,
        "elapsed_sec": round(elapsed, 2),
        "sec_per_scene": round(elapsed / max(written, 1), 3),
        "size_mb": round(size_mb, 1),
        "tensor_shapes": shapes_seen,
        "skipped_existing": already,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=str,
                        default="cached_scenes")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Default 8 on GPU, 1 on CPU (lowest possible memory).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Default 4 on GPU, 0 on CPU (a worker copies the dataset).")
    parser.add_argument("--num_threads", type=int, default=None,
                        help="torch CPU threads. Default: leave one core free so the "
                             "machine stays responsive during a long run.")
    parser.add_argument("--lazy_maxsize", type=int, default=None,
                        help="How many scene meshes to keep resident. Default 4 on GPU, "
                             "1 on CPU.")
    parser.add_argument("--gc_every", type=int, default=32,
                        help="Force a gc.collect() every N scenes (0 disables).")
    parser.add_argument("--no_resume", dest="resume", action="store_false",
                        help="Re-cache scenes even if their .p file already exists. By "
                             "default existing scenes are skipped, so an interrupted run "
                             "can simply be restarted.")
    parser.add_argument("--include_optional", action="store_true",
                        help="Also cache large unused intermediates (seed/vote/fp2 features).")
    parser.add_argument("--no_validate", dest="validate", action="store_false",
                        help="Skip the automatic experiments/diagnostics/validate_scene_cache.py run at the end.")
    parser.add_argument("--validate_samples", type=int, default=200)
    parser.add_argument("--subsample_salt", type=str, default="v1")
    parser.add_argument("--dry_run", action="store_true",
                        help="Check checkpoint, data and one sample per split, then report "
                             "the scene count and estimated cache size. No forward pass, "
                             "nothing written. Safe and fast on a CPU-only machine.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU. NOTE: only useful with --dry_run -- a real caching "
                             "run needs CUDA, because models/proposal_module.py and "
                             "lib/loss_helper.py call .cuda() directly.")

    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_points", type=int, default=40000)
    parser.add_argument("--num_proposals", type=int, default=256)
    parser.add_argument("--num_scenes", type=int, default=-1,
                        help="Cache only the first N scenes of each split (smoke tests).")
    parser.add_argument("--detector", type=str, default="VN", choices=["VN", "GF"])
    parser.add_argument("--use_pretrained", type=str, default="")
    parser.add_argument("--use_checkpoint", type=str, default="")
    parser.add_argument("--no_height", action="store_true")
    parser.add_argument("--use_color", action="store_true")
    parser.add_argument("--use_normal", action="store_true")
    parser.add_argument("--use_multiview", action="store_true")
    parser.add_argument("--use_bidir", action="store_true")
    parser.add_argument("--no_lang_cls", action="store_true")
    parser.add_argument("--no_reference", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--lang_input", type=str, default="glove+parse")
    parser.add_argument("--GF_path", type=str, default=None)

    args = parser.parse_args()

    args.detection = True

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    keys = list(SCENE_CACHE_KEYS_REQUIRED)
    if args.include_optional:
        keys += SCENE_CACHE_KEYS_OPTIONAL

    device = torch.device("cpu" if (args.cpu or not torch.cuda.is_available()) else "cuda")
    on_cpu = device.type == "cpu"

    if args.batch_size is None:
        args.batch_size = 1 if on_cpu else 8
    if args.num_workers is None:
        args.num_workers = 0 if on_cpu else 4
    if args.lazy_maxsize is None:
        args.lazy_maxsize = 1 if on_cpu else 4

    if on_cpu:
        if args.num_threads is None:
            args.num_threads = max(1, (os.cpu_count() or 2) - 1)
        torch.set_num_threads(args.num_threads)
        print(
            f"\nCPU MODE\n"
            f"  batch_size   {args.batch_size}   (lowest memory)\n"
            f"  num_workers  {args.num_workers}   (in-process; a worker would copy the dataset)\n"
            f"  threads      {args.num_threads} of {os.cpu_count()}   (one core left free)\n"
            f"  resume       {'on' if args.resume else 'OFF'}   "
            f"(already-cached scenes are {'skipped' if args.resume else 're-run'})\n"
            f"  validation   skipped on CPU -- it needs ~37 more .cuda() call sites in\n"
            f"               lib/loss_helper.py and lib/eval_helper.py. Run\n"
            f"               experiments/diagnostics/validate_scene_cache.py on a GPU before trusting\n"
            f"               these numbers for the paper.\n"
            f"  This is expected to take hours. It is safe to Ctrl-C and restart:\n"
            f"  each scene is written atomically and completed scenes are skipped.\n"
        )
        args.validate = False
    elif args.num_threads:
        torch.set_num_threads(args.num_threads)

    if args.dry_run:
        return dry_run(args, keys)

    model, ckpt_path = build_model(args, device)

    t0 = time.time()
    per_split = {}
    incomplete = []
    for split in args.splits:
        per_split[split] = cache_split(args, model, split, keys, device)
        expected = len(_pending_scenes(args, split)[0])
        have = per_split[split].get("num_cached_total", per_split[split]["num_scenes"])
        per_split[split]["expected_scenes"] = expected
        per_split[split]["complete"] = (have >= expected)
        if have < expected:
            incomplete.append(f"{split}: {have}/{expected}")
    total = time.time() - t0

    write_meta(
        args.output,
        checkpoint=ckpt_path,
        detector=args.detector,
        num_points=args.num_points,
        num_proposals=args.num_proposals,
        subsample_salt=args.subsample_salt,
        augmentation="none (deterministic per-scene subsample, no flip/rot/scale/translate)",
        use_color=args.use_color,
        use_normal=args.use_normal,
        use_multiview=args.use_multiview,
        use_height=(not args.no_height),
        device=device.type,
        batch_size=args.batch_size,
        cached_keys=keys,
        splits=per_split,
        complete=(not incomplete),
        total_elapsed_sec=round(total, 2),
    )
    print(f"\nwrote {os.path.join(args.output, 'meta.json')}")
    print(f"total cache time: {_fmt_hms(total)}")

    if incomplete:
        print("\nWARNING: cache is INCOMPLETE -- " + ", ".join(incomplete))
        print("Re-run the same command to continue; finished scenes are skipped.")
        return 2

    if args.validate:
        print("\n" + "=" * 78)
        print("running cache validation -- the cache is not usable until this passes")
        print("=" * 78)
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "experiments", "diagnostics", "validate_scene_cache.py"),
            "--cached_scenes_root", args.output,
            "--num_samples", str(args.validate_samples),
            "--detector", args.detector,
            "--num_points", str(args.num_points),
            "--num_proposals", str(args.num_proposals),
            "--subsample_salt", args.subsample_salt,
        ]
        for flag in ("use_color", "use_normal", "use_multiview", "use_bidir",
                     "no_height", "no_lang_cls", "cpu"):
            if getattr(args, flag):
                cmd.append(f"--{flag}")
        if args.use_pretrained:
            cmd += ["--use_pretrained", args.use_pretrained]
        if args.use_checkpoint:
            cmd += ["--use_checkpoint", args.use_checkpoint]
        return subprocess.call(cmd)

    return 0


if __name__ == "__main__":
    sys.exit(main())
