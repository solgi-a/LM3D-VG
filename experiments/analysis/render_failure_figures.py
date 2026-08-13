"""
Render the qualitative failure figures.

    RUNS ON: CPU. About a minute for six cases. No GPU, no checkpoint.

    python experiments/analysis/render_failure_figures.py
    python experiments/analysis/render_failure_figures.py --cases 3 --dpi 300

``failure_cases.py`` picks and diagnoses the cases and writes ``pred.ply`` / ``gt.ply`` per
case; this turns them into finished panels, instead of wireframes somebody has to open in
MeshLab and screenshot.

Each case becomes one PNG with four panels:

  1  bird's-eye view of the whole room, so a predicted box landing on a different instance
     across the room is visible as such. Every other object of the target's class is
     outlined faintly, which makes distractor confusion legible at a glance.
  2  oblique 3-D close-up of the region containing both boxes, in scene colour.
  3  front elevation of the same crop. The orthographic projection resolves silhouettes
     more crisply than a 3-D scatter, and it is usually this panel that makes the referred
     object recognisable in print.
  4  description, parse, diagnosis and IoU, so the figure is self-contained.

Green is ground truth, red is the prediction throughout.

Point clouds are drawn with gamma 0.65 applied to the ScanNet colours, which are captured
dark enough that furniture otherwise merges into its own shadow; close-up points are
depth-sorted because matplotlib's 3-D scatter does not do it and far geometry would paint
over near geometry.

Coordinate frames
-----------------
The boxes in ``predictions.p`` live in the same axis-aligned frame as
``data/scannet/scannet_data/<scene>_aligned_vert.npy``, checked by matching a ground-truth
box against the corresponding row of ``<scene>_aligned_bbox.npy`` and re-checked at run
time for every case. A case that fails the check is still drawn but carries a visible
warning.
"""

import argparse
import json
import os
import pickle
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCANNET_DATA = os.path.join(REPO, "data", "scannet", "scannet_data")

GT_COLOR = "#12A150"
PRED_COLOR = "#D62828"
DISTRACTOR_COLOR = "#8A8FA3"
CROP_MARGIN = 0.9          # metres of context around the two boxes in the close-up


# ======================================================================================
# geometry
# ======================================================================================

def box_edges(corners):
    """Return the 12 edges of a box given its 8 corners, without assuming an ordering.

    ScanRefer's boxes are axis-aligned, so the common case is exact: rebuild the box
    from its extent. The fallback connects each corner to its three nearest
    neighbours, which recovers the edges of any convex box whatever the ordering.
    """
    corners = np.asarray(corners, dtype=float)
    unique_per_axis = [np.unique(np.round(corners[:, axis], 4)) for axis in range(3)]
    if all(len(values) == 2 for values in unique_per_axis):
        low = corners.min(axis=0)
        high = corners.max(axis=0)
        vertices = np.array([[x, y, z] for x in (low[0], high[0])
                             for y in (low[1], high[1])
                             for z in (low[2], high[2])])
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.count_nonzero(np.abs(vertices[i] - vertices[j]) > 1e-9) == 1:
                    edges.append((vertices[i], vertices[j]))
        return edges

    edges, seen = [], set()
    for i in range(8):
        distances = np.linalg.norm(corners - corners[i], axis=1)
        for j in np.argsort(distances)[1:4]:
            key = (min(i, int(j)), max(i, int(j)))
            if key not in seen:
                seen.add(key)
                edges.append((corners[i], corners[int(j)]))
    return edges


def draw_box_3d(axis, corners, color, linewidth=1.6, alpha=1.0):
    for start, end in box_edges(corners):
        axis.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                  color=color, linewidth=linewidth, alpha=alpha)


def draw_box_2d(axis, corners, color, linewidth=1.6, alpha=1.0, fill=False):
    corners = np.asarray(corners, dtype=float)
    low, high = corners.min(axis=0), corners.max(axis=0)
    xs = [low[0], high[0], high[0], low[0], low[0]]
    ys = [low[1], low[1], high[1], high[1], low[1]]
    axis.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)
    if fill:
        axis.fill(xs, ys, color=color, alpha=0.12)


def box_center(corners):
    corners = np.asarray(corners, dtype=float)
    return (corners.min(axis=0) + corners.max(axis=0)) / 2.0


# ======================================================================================
# data
# ======================================================================================

def load_scene(scene_id):
    path = os.path.join(SCANNET_DATA, f"{scene_id}_aligned_vert.npy")
    if not os.path.isfile(path):
        return None, None
    vertices = np.load(path)
    xyz = vertices[:, :3].astype(float)
    rgb = np.clip(vertices[:, 3:6] / 255.0, 0, 1) if vertices.shape[1] >= 6 else None
    return xyz, rgb


def load_scene_boxes(scene_id):
    """(N, 8) rows of [cx, cy, cz, dx, dy, dz, semantic_label, object_id]."""
    path = os.path.join(SCANNET_DATA, f"{scene_id}_aligned_bbox.npy")
    return np.load(path) if os.path.isfile(path) else None


def corners_from_row(row):
    center, size = np.asarray(row[:3], dtype=float), np.asarray(row[3:6], dtype=float)
    low, high = center - size / 2.0, center + size / 2.0
    return np.array([[x, y, z] for x in (low[0], high[0])
                     for y in (low[1], high[1])
                     for z in (low[2], high[2])])


def verify_frames(scene_id, object_id, gt_corners):
    """Confirm the prediction file and the .npy scene share a coordinate frame."""
    boxes = load_scene_boxes(scene_id)
    if boxes is None:
        return None
    match = boxes[boxes[:, -1] == float(object_id)]
    if not len(match):
        return None
    offset = np.linalg.norm(box_center(gt_corners) - np.asarray(match[0][:3], dtype=float))
    return float(offset)


def same_class_boxes(scene_id, object_id):
    """Other instances sharing the target's semantic label -- the distractors."""
    boxes = load_scene_boxes(scene_id)
    if boxes is None:
        return []
    target = boxes[boxes[:, -1] == float(object_id)]
    if not len(target):
        return []
    label = target[0][6]
    return [corners_from_row(row) for row in boxes
            if row[6] == label and row[-1] != float(object_id)]


def discover_cases(cases_dir, limit):
    """Prefer failure_cases.json; fall back to the per-case folders it wrote."""
    manifest = os.path.join(cases_dir, "failure_cases.json")
    if os.path.isfile(manifest):
        with open(manifest) as handle:
            payload = json.load(handle)
        selected = payload.get("selected", [])
        if selected:
            return payload, selected[:limit]

    cases = []
    for name in sorted(os.listdir(cases_dir)):
        folder = os.path.join(cases_dir, name)
        if not os.path.isdir(folder):
            continue
        parts = name.split("_")
        try:
            scene_id = "_".join(parts[2:4])
            object_id = parts[4].replace("obj", "")
            ann_id = parts[5].replace("ann", "")
        except IndexError:
            continue
        cases.append({"scene_id": scene_id, "object_id": object_id, "ann_id": ann_id,
                      "cause": "_".join(parts[1:2]), "description": "", "iou": None})
    return {}, cases[:limit]


# ======================================================================================
# rendering
# ======================================================================================

def render_case(case, predictions, out_dir, dpi, max_points):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_id = case["scene_id"]
    object_id, ann_id = str(case["object_id"]), str(case["ann_id"])

    entry = predictions.get(scene_id, {}).get(object_id, {}).get(ann_id)
    if entry is None:
        print(f"  ! {scene_id}/{object_id}/{ann_id} absent from predictions.p -- skipped")
        return None

    gt = np.asarray(entry["gt_bbox"], dtype=float)
    pred = np.asarray(entry["pred_bbox"], dtype=float)
    iou = float(entry.get("iou", case.get("iou") or 0.0))

    xyz, rgb = load_scene(scene_id)
    if xyz is None:
        print(f"  ! {scene_id}_aligned_vert.npy not found -- skipped")
        return None

    offset = verify_frames(scene_id, object_id, gt)
    misaligned = offset is not None and offset > 0.05

    figure = plt.figure(figsize=(16.5, 4.8))
    grid = figure.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.0, 0.80], wspace=0.13)

    # ---- panel 1: bird's eye of the whole room ---------------------------------------
    ax1 = figure.add_subplot(grid[0, 0])
    step = max(1, len(xyz) // max_points)
    sampled = xyz[::step]
    sampled_rgb = rgb[::step] if rgb is not None else None
    if sampled_rgb is not None:
        sampled_rgb = np.clip(sampled_rgb, 0, 1) ** 0.65      # same lift as the close-up
    ax1.scatter(sampled[:, 0], sampled[:, 1], s=2.0,
                c=sampled_rgb if sampled_rgb is not None else "#B8BEC9",
                linewidths=0, rasterized=True)

    distractors = same_class_boxes(scene_id, object_id)
    for box in distractors:
        draw_box_2d(ax1, box, DISTRACTOR_COLOR, linewidth=0.9, alpha=0.75)
    draw_box_2d(ax1, gt, GT_COLOR, linewidth=2.0, fill=True)
    draw_box_2d(ax1, pred, PRED_COLOR, linewidth=2.0, fill=True)

    gt_center, pred_center = box_center(gt), box_center(pred)
    ax1.annotate("", xy=pred_center[:2], xytext=gt_center[:2],
                 arrowprops=dict(arrowstyle="->", color="#333333",
                                 linewidth=1.2, linestyle=":"))
    drift = float(np.linalg.norm(gt_center - pred_center))
    ax1.set_title(f"{scene_id} -- top view   (centre error {drift:.2f} m)", fontsize=9)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (m)", fontsize=8)
    ax1.set_ylabel("y (m)", fontsize=8)
    ax1.tick_params(labelsize=7)

    # ---- panels 2 and 3: the close-up ------------------------------------------------
    # The crop is not thinned (it holds far fewer than `max_points` anyway), markers are
    # large enough to form a surface, and colour is gamma-corrected out of ScanNet's
    # shadows. Points are drawn back to front since matplotlib's 3-D scatter does not
    # depth-sort.
    both = np.vstack([gt, pred])
    low, high = both.min(axis=0) - CROP_MARGIN, both.max(axis=0) + CROP_MARGIN
    inside = np.all((xyz >= low) & (xyz <= high), axis=1)
    crop = xyz[inside]
    crop_rgb = rgb[inside] if rgb is not None else None
    if len(crop) > max_points:                    # only bites on very large crops
        keep = np.linspace(0, len(crop) - 1, max_points).astype(int)
        crop = crop[keep]
        crop_rgb = crop_rgb[keep] if crop_rgb is not None else None

    if crop_rgb is not None:
        # Gamma < 1 lifts the midtones; ScanNet scans are captured dark and the raw
        # values leave most surfaces indistinguishable from their own shadow.
        crop_rgb = np.clip(crop_rgb, 0, 1) ** 0.65
    colour = crop_rgb if crop_rgb is not None else "#B8BEC9"

    marker = 6.0 if len(crop) < 6000 else (4.0 if len(crop) < 20000 else 2.5)

    ax2 = figure.add_subplot(grid[0, 1], projection="3d")
    if len(crop):
        # back to front along the view direction at azim=-62, elev=24
        depth = crop[:, 0] * np.cos(np.deg2rad(-62)) + crop[:, 1] * np.sin(np.deg2rad(-62))
        order = np.argsort(-depth)
        ax2.scatter(crop[order, 0], crop[order, 1], crop[order, 2], s=marker,
                    c=colour[order] if crop_rgb is not None else colour,
                    linewidths=0, depthshade=False, rasterized=True)
    draw_box_3d(ax2, gt, GT_COLOR, linewidth=2.1)
    draw_box_3d(ax2, pred, PRED_COLOR, linewidth=2.1)
    ax2.view_init(elev=24, azim=-62)
    ax2.set_title(f"close-up, oblique  ({len(crop):,} points)", fontsize=9)
    ax2.set_box_aspect((high[0] - low[0], high[1] - low[1],
                        max(high[2] - low[2], 0.4)))
    for axis_setter in (ax2.set_xticks, ax2.set_yticks, ax2.set_zticks):
        axis_setter([])

    # Front elevation. A flat orthographic projection resolves object silhouettes far
    # more crisply than any 3-D scatter, which is what makes the referred object
    # actually recognisable in print.
    ax3d = figure.add_subplot(grid[0, 2])
    if len(crop):
        order = np.argsort(crop[:, 1])[::-1]          # far wall first
        ax3d.scatter(crop[order, 0], crop[order, 2], s=marker,
                     c=colour[order] if crop_rgb is not None else colour,
                     linewidths=0, rasterized=True)
    for box, shade in ((gt, GT_COLOR), (pred, PRED_COLOR)):
        corners = np.asarray(box, dtype=float)
        b_low, b_high = corners.min(axis=0), corners.max(axis=0)
        ax3d.plot([b_low[0], b_high[0], b_high[0], b_low[0], b_low[0]],
                  [b_low[2], b_low[2], b_high[2], b_high[2], b_low[2]],
                  color=shade, linewidth=2.1)
    ax3d.set_title("close-up, front elevation", fontsize=9)
    ax3d.set_aspect("equal")
    ax3d.set_xlabel("x (m)", fontsize=8)
    ax3d.set_ylabel("z (m)", fontsize=8)
    ax3d.tick_params(labelsize=7)

    # ---- panel 3: the words ------------------------------------------------------------
    ax3 = figure.add_subplot(grid[0, 3])
    ax3.axis("off")

    # Only the count this figure draws goes on the panel. failure_cases.py derives its
    # same-class count from the NYU40 mapping over the ScanRefer records, which need not
    # equal the number of boxes carrying this scene's semantic label. Both are in
    # figures.json.
    signals = case.get("signals", {}) or {}
    lines = [("cause", case.get("cause", "?")),
             ("IoU", f"{iou:.3f}"),
             ("target class", case.get("object_name", "?")),
             ("parsed target", case.get("parsed_target", "-")),
             ("parsed adjectives", case.get("parsed_adjectives", "-")),
             ("parsed neighbours", case.get("parsed_neighbors", "-")),
             ("same-label boxes drawn", str(len(distractors))),
             ("tokens", str(signals.get("num_tokens", "-"))),
             ("spatial relations", str(signals.get("num_spatial", "-")))]

    text = "\n".join(
        "\n".join(textwrap.wrap(f"{label:<22}: {value}", 44,
                                subsequent_indent=" " * 24))
        for label, value in lines)
    description = case.get("description", "")
    if description:
        text = ('"' + "\n".join(textwrap.wrap(description, 44)) + '"\n\n') + text

    ax3.text(0.0, 1.0, text, va="top", ha="left", fontsize=7.8, family="monospace",
             transform=ax3.transAxes)
    ax3.text(0.0, 0.02,
             "green = ground truth    red = prediction\ngrey = other objects of the "
             "same class", fontsize=7.6, color="#444444", transform=ax3.transAxes)

    if misaligned:
        figure.text(0.5, 0.015,
                    f"WARNING: gt box centre is {offset:.3f} m from the scene's "
                    f"aligned_bbox entry -- do not publish this panel",
                    ha="center", color=PRED_COLOR, fontsize=9)

    figure.suptitle(
        f"{case.get('cause', 'failure')}  --  {scene_id}  object {object_id}  "
        f"annotation {ann_id}  --  IoU {iou:.2f}", fontsize=10.5)
    # tight_layout cannot handle the 3D axes, so the margins are set explicitly.
    figure.subplots_adjust(left=0.045, right=0.99, top=0.88,
                           bottom=0.17 if misaligned else 0.12)

    filename = f"{case.get('cause', 'case')}_{scene_id}_obj{object_id}_ann{ann_id}.png"
    path = os.path.join(out_dir, filename)
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return {"path": path, "iou": iou, "drift": drift,
            "distractors": len(distractors), "frame_offset": offset,
            "misaligned": bool(misaligned)}


def contact_sheet(rendered, out_dir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    if not rendered:
        return None
    figure, axes = plt.subplots(len(rendered), 1,
                               figsize=(15.5, 5.0 * len(rendered)))
    if len(rendered) == 1:
        axes = [axes]
    for axis, item in zip(axes, rendered):
        axis.imshow(mpimg.imread(item["path"]))
        axis.axis("off")
    figure.tight_layout()
    path = os.path.join(out_dir, "failure_contact_sheet.png")
    figure.savefig(path, dpi=max(90, dpi // 3))
    plt.close(figure)
    return path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions",
                        default="outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p")
    parser.add_argument("--cases-dir", dest="cases_dir",
                        default="outputs/analysis/failure_cases",
                        help="output of experiments/analysis/failure_cases.py")
    parser.add_argument("--out-dir", dest="out_dir",
                        default="experiments/analysis/figures",
                        help="default keeps the figures beside this script; pass "
                             "outputs/diagnostics/failure_figures to write them "
                             "under outputs/ instead")
    parser.add_argument("--cases", type=int, default=12, help="maximum cases to render")
    parser.add_argument("--dpi", type=int, default=200,
                        help="use 300 for the camera-ready figure")
    parser.add_argument("--max-points", dest="max_points", type=int, default=25000)
    args = parser.parse_args()

    cases_dir = os.path.join(REPO, args.cases_dir)
    if not os.path.isdir(cases_dir):
        print(f"case folder not found: {args.cases_dir}")
        print("Run experiments/analysis/failure_cases.py first -- it selects and diagnoses the "
              "cases this script draws.")
        return 1

    predictions_path = os.path.join(REPO, args.predictions)
    if not os.path.isfile(predictions_path):
        print(f"predictions not found: {args.predictions}")
        return 1
    with open(predictions_path, "rb") as handle:
        predictions = pickle.load(handle)

    payload, cases = discover_cases(cases_dir, args.cases)
    if not cases:
        print(f"no cases found under {args.cases_dir}")
        return 1

    out_dir = os.path.join(REPO, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"rendering {len(cases)} case(s) at {args.dpi} dpi -> "
          f"{os.path.relpath(out_dir, REPO)}\n")

    rendered = []
    for index, case in enumerate(cases, 1):
        label = (f"{case['scene_id']}/{case['object_id']}/{case['ann_id']} "
                 f"[{case.get('cause', '?')}]")
        print(f"  [{index}/{len(cases)}] {label}")
        result = render_case(case, predictions, out_dir, args.dpi, args.max_points)
        if result:
            result["case"] = case
            rendered.append(result)
            flag = "  FRAME MISMATCH" if result["misaligned"] else ""
            print(f"        IoU {result['iou']:.3f}  centre error "
                  f"{result['drift']:.2f} m  distractors {result['distractors']}{flag}")

    sheet = contact_sheet(rendered, out_dir, args.dpi)

    manifest = {
        "predictions": args.predictions,
        "cases_dir": args.cases_dir,
        "model": payload.get("model"),
        "rendered": [{"file": os.path.basename(item["path"]),
                      "scene_id": item["case"]["scene_id"],
                      "object_id": item["case"]["object_id"],
                      "ann_id": item["case"]["ann_id"],
                      "cause": item["case"].get("cause"),
                      "iou": item["iou"],
                      "centre_error_m": item["drift"],
                      "same_label_boxes_drawn": item["distractors"],
                      "same_class_instances_from_analysis":
                          (item["case"].get("signals") or {}).get("same_class_instances"),
                      "frame_offset_m": item["frame_offset"]}
                     for item in rendered],
    }
    manifest_path = os.path.join(out_dir, "figures.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nwrote {len(rendered)} figure(s)")
    if sheet:
        print(f"wrote {os.path.relpath(sheet, REPO)}")
    print(f"wrote {os.path.relpath(manifest_path, REPO)}")

    bad = [item for item in rendered if item["misaligned"]]
    if bad:
        print(f"\n{len(bad)} panel(s) carry a coordinate-frame warning; do not publish "
              f"them without checking the scene's aligned_bbox entry.")
        return 1

    causes = sorted({item["case"].get("cause") for item in rendered})
    print("\nCauses covered: " + ", ".join(str(c) for c in causes))
    print("The paper needs at least three distinct scenarios with a stated cause; "
          "pick the panels whose causes differ, not the three lowest IoUs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
