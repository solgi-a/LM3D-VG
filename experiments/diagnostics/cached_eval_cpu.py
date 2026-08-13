"""
Evaluate the cached path on a machine with no usable GPU, and time it.

    RUNS ON: CPU. Sized by --num-samples; the default 128 takes about a minute after a
    one-off ~30 s load of the 933 MB GloVe table. The script measures its own throughput
    and extrapolates for the full split.

    python experiments/diagnostics/cached_eval_cpu.py --num-samples 128
    python experiments/diagnostics/cached_eval_cpu.py --num-samples 0      # whole split

``validate_scene_cache.py`` is the authoritative check but needs CUDA -- it calls
``get_loss`` and ``get_eval``, which between them contain 32 hardcoded ``.cuda()`` calls.
This answers the narrower "does the cached path reproduce the reported accuracy?" without
entering those functions, via two shims confined to this file:

* ``torch.Tensor.cuda`` becomes identity on CPU. Five call sites in the cached forward
  path hardcode it -- ``models/lang_module.py:56`` and
  ``models/match_module.py:158,190,214,245``. Every tensor is already on CPU, so this is a
  no-op on a real GPU run.
* Proposal selection is reproduced rather than imported. ``lib/eval_helper.py:129-134``
  picks ``argmax(cluster_ref * objectness_mask)`` with the mask
  ``argmax(objectness_scores, 2) == 1``, straight from the cache. The winning box is read
  from the cache's ``pred_bboxes`` rather than rebuilt from centre/heading/size, and
  ground truth comes from ``predictions.p``.

The accuracy will not match the same checkpoint's end-to-end number to 1e-4 -- deterministic
vs random point subsampling -- but lands within a point or two when the cache is sound.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def aabb_iou(a, b):
    a_low, a_high = a.min(axis=0), a.max(axis=0)
    b_low, b_high = b.min(axis=0), b.max(axis=0)
    overlap = np.clip(np.minimum(a_high, b_high) - np.maximum(a_low, b_low), 0.0, None)
    intersection = float(np.prod(overlap))
    union = (float(np.prod(a_high - a_low)) + float(np.prod(b_high - b_low))
             - intersection)
    return intersection / union if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    cli = parser.parse_args()

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
        # See the module docstring: five call sites in the cached forward path hardcode
        # .cuda(). Everything is already on CPU, so identity is the correct behaviour.
        torch.Tensor.cuda = lambda self, *a, **k: self          # noqa: E731

    from data.scannet.model_util_scannet import ScannetDatasetConfig
    from experiments.ablation import ablation_config, ablation_hooks
    from experiments.analysis.common import ensure_dir, join, load_predictions, \
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
    # ablation_hooks wraps load_state_dict on this instance and raises when any parameter
    # would be left at its random initialisation (ABLATION.STRICT_CHECKPOINT). Present
    # that as a clean refusal rather than a traceback -- it is a configuration problem
    # with a known fix, not a crash.
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

    ious, position = [], 0
    forward_started = time.time()
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
                    ious.append(aabb_iou(boxes[row][selected[row]],
                                         np.asarray(truth["gt_bbox"], dtype=np.float64)))
                position += 1
    forward_seconds = time.time() - forward_started

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


if __name__ == "__main__":
    sys.exit(main())
