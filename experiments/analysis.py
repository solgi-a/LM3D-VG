"""
Every post-hoc analysis of a finished run. CPU only -- no model is built and no GPU used.

    python experiments/analysis.py <command> --help

Commands: results-table, linguistic-complexity, parse-quality-split, failure-cases,
failure-figures, failure-scenes, annotation-sheet, error-taxonomy, error-propagation,
parse-field-comparison, complex-sentence-showdown, seed-aggregate.

Everything here consumes artifacts that already exist -- outputs/<run>/predictions.p, the
parse caches, and ScanRefer's JSON. Which statistical test goes with which table, and
why: details.md §7.
"""

import argparse
import csv
import fnmatch
import glob
import json
import math
import os
import pickle
import random
import re
import statistics
import subprocess
import sys
import textwrap
from collections import Counter, OrderedDict, defaultdict

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from experiments.ablation.parsing import (  # noqa: E402
    FIELD_MAX_TOKENS, NOT_MENTIONED_TOKENS, match_kinds)


#: Thresholds the paper reports.
THRESHOLDS = (0.25, 0.5)



def load_scanrefer(split="val", data_root="data"):
    """ScanRefer annotations for one split, as the list of raw records."""
    for candidate in (os.path.join(data_root, f"ScanRefer_filtered_{split}.json"),
                      os.path.join(REPO_ROOT, data_root, f"ScanRefer_filtered_{split}.json")):
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"ScanRefer_filtered_{split}.json not found under {data_root!r} "
        f"(cwd={os.getcwd()!r}, repo={REPO_ROOT!r})")


def load_predictions(path):
    """Flatten predictions.p to {(scene_id, object_id, ann_id): record}."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"predictions file not found: {path}\n"
            f"Generate it with:\n"
            f"    python scripts/ScanRefer_eval.py --folder <run> --reference --force "
            f"--use_color --use_normal --lang_num_max 1")
    with open(path, "rb") as f:
        raw = pickle.load(f)

    flat = {}
    for scene_id, objects in raw.items():
        for object_id, anns in objects.items():
            for ann_id, record in anns.items():
                flat[(str(scene_id), str(object_id), str(ann_id))] = record
    return flat


def load_parse_cache(folder, split="val", parsing_root="data_parsing"):
    """A tokenized parse cache as {scene_id: {object_id: {ann_id: {field: [tok]}}}}."""
    for base in (parsing_root, os.path.join(REPO_ROOT, parsing_root)):
        path = os.path.join(base, folder, f"tokenized_parsed_result_{split}.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"parse cache not found: {parsing_root}/{folder}/tokenized_parsed_result_{split}.json")


def parse_for(cache, scene_id, object_id, ann_id):
    """One annotation's parse, or None when the cache does not cover it."""
    try:
        return cache[str(scene_id)][str(object_id)][str(ann_id)]
    except (KeyError, TypeError):
        return None


def join(predictions, records):
    """Merge ScanRefer records with their prediction. Returns (rows, num_missing).

    Records with no prediction are counted, not silently dropped -- that would change the
    denominator of every accuracy in the report.
    """
    rows, missing = [], 0
    for record in records:
        key = (str(record["scene_id"]), str(record["object_id"]), str(record["ann_id"]))
        prediction = predictions.get(key)
        if prediction is None:
            missing += 1
            continue
        row = dict(record)
        row["iou"] = float(prediction["iou"])
        row["_prediction"] = prediction
        rows.append(row)
    return rows, missing


def unique_multiple_lookup(records, scannet_meta=None):
    """ScanRefer's Unique/Multiple split: {(scene, object, ann): 0 unique | 1 multiple}.

    Ported from lib/dataset.py rather than imported -- this package is model-free. The
    two details a naive reimplementation gets wrong: details.md §7.
    """
    import numpy as _np

    from data.scannet.model_util_scannet import ScannetDatasetConfig

    meta = scannet_meta or os.path.join(
        REPO_ROOT, "data", "scannet", "meta_data", "scannetv2-labels.combined.tsv")
    if not os.path.isfile(meta):
        raise FileNotFoundError(
            f"ScanNet label mapping not found: {meta}\n"
            f"It is required to reproduce ScanRefer's Unique/Multiple split.")

    labels = ScannetDatasetConfig().type2class.keys()
    label_set = set(labels)
    scannet2label = {label: i for i, label in enumerate(labels)}

    raw2label = {}
    for line in [line.rstrip() for line in open(meta)][1:]:
        elements = line.split("\t")
        raw_name, nyu40_name = elements[1], elements[7]
        raw2label[raw_name] = scannet2label[
            nyu40_name if nyu40_name in label_set else "others"]
    raw2label["shower_curtain"] = 13

    all_sem_labels, seen = {}, {}
    for record in records:
        scene_id, object_id = record["scene_id"], record["object_id"]
        name = " ".join(record["object_name"].split("_"))
        all_sem_labels.setdefault(scene_id, [])
        seen.setdefault(scene_id, set())
        if object_id not in seen[scene_id]:
            seen[scene_id].add(object_id)
            all_sem_labels[scene_id].append(raw2label.get(name, 17))
    all_sem_labels = {s: _np.array(v) for s, v in all_sem_labels.items()}

    lookup = {}
    for record in records:
        scene_id = record["scene_id"]
        name = " ".join(record["object_name"].split("_"))
        sem_label = raw2label.get(name, 17)
        multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1
        lookup[(str(scene_id), str(record["object_id"]), str(record["ann_id"]))] = multiple
    return lookup


def mcnemar(a_hits, b_hits):
    """Paired binary comparison of two models on the same items.

    Returns (a_only, b_only, chi2, p). Conditions on the discordant pairs, which is what
    an unpaired proportion test cannot do. See details.md §7.
    """
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, float("nan"), float("nan")
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n          # continuity-corrected
    return a_only, b_only, chi2, math.erfc(math.sqrt(chi2 / 2.0))


def bootstrap_ci(values_fn, n_items, num_resamples=2000, seed=42, alpha=0.05):
    """Percentile bootstrap CI. values_fn(indices) -> float; returns (point, low, high)."""
    import numpy as _np

    rng = _np.random.default_rng(seed)
    point = values_fn(_np.arange(n_items))
    samples = _np.empty(num_resamples, dtype=float)
    for i in range(num_resamples):
        samples[i] = values_fn(rng.integers(0, n_items, n_items))
    low, high = _np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(low), float(high)


def parse_predictions_arg(value):
    """``NAME=PATH`` -> (name, path). A bare path gets its run folder as the name."""
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), path.strip()
    # outputs/<run>/predictions.p -> "<run>"
    parent = os.path.basename(os.path.dirname(os.path.abspath(value)))
    return (parent or "model"), value


def find_default_predictions(output_root="outputs"):
    """Every outputs/*/predictions.p, newest first. Convenience for the default arg."""
    root = output_root if os.path.isdir(output_root) else os.path.join(REPO_ROOT, output_root)
    if not os.path.isdir(root):
        return []
    found = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry, "predictions.p")
        if os.path.isfile(path):
            found.append(path)
    found.sort(key=os.path.getmtime, reverse=True)
    return found



def accuracy(ious, threshold):
    """Acc@threshold as a fraction. Empty input -> 0.0."""
    ious = np.asarray(ious, dtype=float)
    if ious.size == 0:
        return 0.0
    return float((ious >= threshold).mean())


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval for a proportion.

    Wilson, not the normal approximation, which leaves [0, 1] for small bins near 0 or 1.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_proportion_z(k1, n1, k2, n2):
    """Two-sided two-proportion z-test. Returns (diff, z, p); (0,nan,nan) if degenerate."""
    if n1 == 0 or n2 == 0:
        return (0.0, float("nan"), float("nan"))
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2, float("nan"), float("nan"))
    z = (p1 - p2) / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return (p1 - p2, z, p)


def spearman(x, y):
    """Spearman rank correlation, (rho, p). Falls back to rho-only without scipy."""
    if len(x) < 3:
        return (float("nan"), float("nan"))
    try:
        from scipy.stats import spearmanr

        result = spearmanr(x, y)
        return (float(result.statistic), float(result.pvalue))
    except Exception:
        rx, ry = _rank(x), _rank(y)
        rx, ry = np.asarray(rx), np.asarray(ry)
        rx, ry = rx - rx.mean(), ry - ry.mean()
        denominator = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
        if denominator == 0:
            return (float("nan"), float("nan"))
        return (float((rx * ry).sum() / denominator), float("nan"))


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def quantile_edges(values, num_bins):
    """Interior quantile cut points, de-duplicated. Returns num_bins-1 edges at most."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return []
    qs = [i / num_bins for i in range(1, num_bins)]
    edges = [float(np.quantile(values, q)) for q in qs]
    unique = []
    for edge in edges:
        if not unique or edge > unique[-1]:
            unique.append(edge)
    return unique



def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_jsonable)
    return path


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def md_table(headers, rows):
    """A GitHub-flavoured markdown table, columns padded to a readable width."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    lines = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(cells[0])) + " |",
             "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for row in cells[1:]:
        lines.append("| " + " | ".join(row[i].ljust(widths[i])
                                       for i in range(len(headers))) + " |")
    return "\n".join(lines)


def pct(value, digits=2):
    return f"{100.0 * value:.{digits}f}"



#: The 12 edges, in the corner order utils/box_util.get_3d_box produces: 0-3 top face,
#: 4-7 bottom face, each in the same rotational order.
_BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))

#: Triangles of a unit prism's 6 faces, indexed into the 8 corners this module emits
#: per edge (0-3 the start cap, 4-7 the end cap).
_PRISM_FACES = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
                (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0))


def normalise_corners(corners):
    """An (8,3) box rebuilt from its min/max bounds in ``_BOX_EDGES`` order.

    Two corner orderings circulate in this module -- predictions.p and ``_BOX_EDGES`` put
    the top face first, while ``corners_from_row`` walks x/y/z lexicographically. Every
    box here is axis-aligned, so rebuilding from the bounds makes the caller's ordering
    stop mattering. ``draw_box_2d`` and ``draw_box_projected`` normalise the same way.
    """
    corners = np.asarray(corners, dtype=float).reshape(8, 3)
    low, high = corners.min(axis=0), corners.max(axis=0)
    # 0-3 the top face, 4-7 the bottom, each in rotational order -- what _BOX_EDGES indexes.
    return np.array([[high[0], high[1], high[2]], [high[0], low[1], high[2]],
                     [low[0], low[1], high[2]], [low[0], high[1], high[2]],
                     [high[0], high[1], low[2]], [high[0], low[1], low[2]],
                     [low[0], low[1], low[2]], [low[0], high[1], low[2]]])


def box_edge_points(corners, spacing=0.005):
    """The box's 12 edges sampled as points, dense enough to read as solid lines.

    This is how a box joins a *point cloud*. Writing it as triangles instead would put a
    ``face`` element in the file, and every viewer keys off that to load the whole thing
    as a mesh -- which hides the face-less scan points and loses the colour entirely.
    """
    corners = normalise_corners(corners)
    spacing = max(float(spacing), 1e-4)
    points = []
    for start_index, end_index in _BOX_EDGES:
        start, end = corners[start_index], corners[end_index]
        length = float(np.linalg.norm(end - start))
        steps = max(int(np.ceil(length / spacing)) + 1, 2)
        points.append(np.linspace(start, end, steps))
    return np.concatenate(points) if points else np.empty((0, 3))


def box_prism_geometry(corners, radius=0.02):
    """An (8,3) box as wireframe prisms: returns (vertices, faces), faces 0-based."""
    corners = normalise_corners(corners)
    vertices, faces = [], []

    for start_index, end_index in _BOX_EDGES:
        start, end = corners[start_index], corners[end_index]
        axis = end - start
        length = np.linalg.norm(axis)
        if length < 1e-9:
            continue
        axis = axis / length

        # Any two unit vectors orthogonal to the edge give the prism's cross-section.
        reference = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(reference, axis))) > 0.9:
            reference = np.array([1.0, 0.0, 0.0])
        u = np.cross(axis, reference)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)

        base = len(vertices)
        for point in (start, end):
            for du, dv in ((+1, +1), (+1, -1), (-1, -1), (-1, +1)):
                vertices.append(point + radius * (du * u + dv * v))
        for a, b, c in _PRISM_FACES:
            faces.append((base + a, base + b, base + c))
    return vertices, faces


def write_bbox_ply(corners, path, color=(255, 0, 0), radius=0.02):
    """Write an (8,3) box as a wireframe PLY: each edge becomes a thin square prism.

    A solid box would occlude the cloud it annotates, and PLY's ``edge`` element is not
    portably supported. Prisms render in MeshLab, CloudCompare and Open3D alike.
    """
    vertices, faces = box_prism_geometry(corners, radius)

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    r, g, b = (int(c) for c in color)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_index\n")
        f.write("end_header\n")
        for x, y, z in vertices:
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")
        for a, b_, c in faces:
            f.write(f"3 {a} {b_} {c}\n")
    return path


#: One merged scene: the scan's coloured points, then the boxes' edge points.
SCENE_PLY_VERTEX_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                   ("red", "u1"), ("green", "u1"), ("blue", "u1")])


def write_scene_ply(path, xyz, rgb=None, boxes=(), point_color=(180, 190, 201)):
    """Write a cloud and its boxes as one binary PLY, for a real 3-D viewer.

    ``boxes`` is a sequence of ``(corners, (r, g, b), spacing)``; each becomes points
    sampled along its 12 edges, appended to the cloud in its own colour.

    **No ``face`` element is written, and that is the point.** A PLY carrying faces is
    loaded as a *mesh*: CloudCompare then hides the face-less scan vertices behind a mesh
    entity, and MeshLab shades it with lighting and draws no isolated points at all. A
    scene exported that way looks empty or grey however much colour the file holds. As a
    pure cloud it opens coloured everywhere, with nothing to switch on.

    Binary rather than ascii: a 160k-point scene is ~12 MB as text and slow to parse,
    where ``tofile`` on a structured dtype writes the same data in a fraction of that.
    """
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    if rgb is None:
        colors = np.tile(np.asarray(point_color, dtype=np.uint8), (len(xyz), 1))
    else:
        colors = (np.clip(np.asarray(rgb, dtype=float), 0, 1) * 255).astype(np.uint8)

    parts_xyz, parts_rgb = [xyz], [colors]
    for corners, color, spacing in boxes:
        points = box_edge_points(corners, spacing)
        parts_xyz.append(points)
        parts_rgb.append(np.tile(np.asarray(color, dtype=np.uint8), (len(points), 1)))

    all_xyz = np.concatenate(parts_xyz)
    all_rgb = np.concatenate(parts_rgb)

    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "wb") as handle:
        handle.write(b"ply\nformat binary_little_endian 1.0\n")
        handle.write(f"element vertex {len(all_xyz)}\n".encode())
        handle.write(b"property float x\nproperty float y\nproperty float z\n")
        handle.write(b"property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(b"end_header\n")

        buffer = np.empty(len(all_xyz), dtype=SCENE_PLY_VERTEX_DTYPE)
        buffer["x"], buffer["y"], buffer["z"] = all_xyz[:, 0], all_xyz[:, 1], all_xyz[:, 2]
        buffer["red"], buffer["green"], buffer["blue"] = (all_rgb[:, 0], all_rgb[:, 1],
                                                          all_rgb[:, 2])
        buffer.tofile(handle)
    return path


def box_volume(corners):
    """Axis-aligned volume of an (8,3) box. Used as a proxy for object size."""
    corners = np.asarray(corners, dtype=float).reshape(8, 3)
    extent = corners.max(axis=0) - corners.min(axis=0)
    return float(np.prod(extent))


# ---- results-table ----

_DOC_RESULTS_TABLE = """
Main results table: Overall / Unique / Multiple, with a paired baseline test. CPU.

    python experiments/analysis.py results-table         --predictions ours=outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p         --predictions 3DVG-Trans=outputs/3DVG-TRANS-outputs/predictions.p

Every number is recomputed from a prediction file rather than copied from published tables.

An annotation is *unique* when its scene holds exactly one object of the referred object's
NYU40 class -- the mapped label, not the raw object_name. Models share the same 9,508
annotations, so significance is McNemar and the bootstrap CI carries the same pairing.
See details.md §7.
"""

SUBSETS = ("overall", "unique", "multiple")


def _add_results_table_args(parser):
    parser.add_argument("--predictions", action="append", required=True,
                        metavar="NAME=PATH",
                        help="Repeatable. The first is the reference every other model is "
                             "compared against.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for the difference CIs; 0 disables.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/analysis/results_table")
    parser.add_argument("--latex", action="store_true")
    return parser


def run_results_table(args):

    records = load_scanrefer(args.split, args.data_root)
    lookup = unique_multiple_lookup(records)

    models, hits, ious = [], {}, {}
    common_keys = None
    for spec in args.predictions:
        name, path = parse_predictions_arg(spec)
        rows, missing = join(load_predictions(path), records)
        print(f"[{name}] {len(rows)} annotations from {path}" +
              (f"  ({missing} without a prediction)" if missing else ""))
        models.append(name)
        keys = {(r["scene_id"], r["object_id"], r["ann_id"]): r["iou"] for r in rows}
        ious[name] = keys
        common_keys = set(keys) if common_keys is None else (common_keys & set(keys))

    order = [k for k in ((r["scene_id"], r["object_id"], r["ann_id"]) for r in records)
             if k in common_keys]
    if len(order) != len(records):
        print(f"[join] {len(order)}/{len(records)} annotations are predicted by all "
              f"{len(models)} model(s); the rest are excluded so every column has the "
              f"same denominator")

    subset_mask = {
        "overall": np.ones(len(order), dtype=bool),
        "unique": np.array([lookup[k] == 0 for k in order]),
        "multiple": np.array([lookup[k] == 1 for k in order]),
    }
    print(f"[split] unique={subset_mask['unique'].sum()}  "
          f"multiple={subset_mask['multiple'].sum()}  total={len(order)}")

    for name in models:
        series = np.array([ious[name][k] for k in order], dtype=float)
        hits[name] = {t: (series >= t) for t in THRESHOLDS}

    reference = models[0]
    report = {"split": args.split, "models": models, "n": len(order),
              "subset_sizes": {s: int(subset_mask[s].sum()) for s in SUBSETS},
              "results": {}}

    rows = []
    for name in models:
        row = [name]
        report["results"][name] = {}
        for subset in SUBSETS:
            mask = subset_mask[subset]
            entry = {}
            for threshold in THRESHOLDS:
                selected = hits[name][threshold][mask]
                k, n = int(selected.sum()), int(mask.sum())
                low, high = wilson_ci(k, n)
                entry[f"acc_{threshold}"] = k / n if n else 0.0
                entry[f"acc_{threshold}_ci"] = [low, high]
                row.append(pct(k / n if n else 0.0))
            report["results"][name][subset] = entry
        rows.append(row)

    headers = ["model"]
    for subset in SUBSETS:
        for threshold in THRESHOLDS:
            headers.append(f"{subset} @{threshold}")
    table = md_table(headers, rows)
    print("\n" + table)

    comparison_rows = []
    for name in models[1:]:
        for subset in SUBSETS:
            mask = subset_mask[subset]
            for threshold in THRESHOLDS:
                a = hits[reference][threshold][mask]
                b = hits[name][threshold][mask]
                a_only, b_only, chi2, p = mcnemar(a, b)
                difference = float(a.mean() - b.mean())

                ci_text = "-"
                if args.bootstrap:
                    def statistic(idx, a=a, b=b):
                        return float(a[idx].mean() - b[idx].mean())

                    _point, low, high = bootstrap_ci(
                        statistic, int(mask.sum()), args.bootstrap, args.seed)
                    ci_text = f"[{100 * low:+.2f}, {100 * high:+.2f}]"

                comparison_rows.append([
                    f"{subset} @{threshold}", f"{100 * difference:+.2f}", ci_text,
                    f"{a_only}/{b_only}", f"{p:.2e}",
                    "yes" if (p == p and p < 0.05) else "no",
                ])
                report["results"][name].setdefault("vs_" + reference, {})[
                    f"{subset}_{threshold}"] = {
                    "diff_pp": 100 * difference, "mcnemar_chi2": chi2, "p_value": p,
                    "reference_only": a_only, "other_only": b_only,
                }

        comparison = md_table(
            ["subset", f"{reference} - {name} (pp)", "95% CI (pp)",
             f"{reference}-only/{name}-only", "McNemar p", "significant"],
            comparison_rows)
        print(f"\nPaired comparison: {reference} vs {name}\n")
        print(comparison)
        comparison_rows = []

    lines = ["# Main results table\n",
             f"split: `{args.split}` | annotations: {len(order)} "
             f"(unique {int(subset_mask['unique'].sum())}, "
             f"multiple {int(subset_mask['multiple'].sum())})\n",
             "**All numbers below were computed on this machine from `predictions.p`, "
             "not quoted from the original papers.**\n",
             table]

    verdicts = []
    if len(models) > 1:
        for name in models[1:]:
            deltas = {}
            for subset in SUBSETS:
                mask = subset_mask[subset]
                deltas[subset] = float(hits[reference][0.25][mask].mean()
                                       - hits[name][0.25][mask].mean())
            if deltas["multiple"] > deltas["unique"] + 0.01:
                verdicts.append(
                    f"Against {name}, the advantage is concentrated in **multiple** "
                    f"({100 * deltas['multiple']:+.2f} pp) rather than unique "
                    f"({100 * deltas['unique']:+.2f} pp) at IoU 0.25. Ambiguous scenes are "
                    f"where adjacency reasoning should help, so this is the pattern the "
                    f"method predicts and it is worth stating explicitly.")
            elif deltas["unique"] > deltas["multiple"] + 0.01:
                verdicts.append(
                    f"Against {name}, the advantage is larger on **unique** "
                    f"({100 * deltas['unique']:+.2f} pp) than multiple "
                    f"({100 * deltas['multiple']:+.2f} pp). That is the opposite of what "
                    f"adjacency reasoning predicts -- report it and do not attribute the "
                    f"gain to disambiguation.")
            else:
                verdicts.append(
                    f"Against {name}, the advantage is roughly uniform across unique "
                    f"({100 * deltas['unique']:+.2f} pp) and multiple "
                    f"({100 * deltas['multiple']:+.2f} pp).")

    if verdicts:
        print("\nReading:")
        for verdict in verdicts:
            print(f"  {verdict}")
        lines.append("\n## Reading\n")
        lines += [f"- {v}" for v in verdicts]

    if args.latex:
        print("\nLaTeX rows:")
        latex = []
        for row in rows:
            latex.append("  " + row[0] + " & " + " & ".join(row[1:]) + r" \\")
        for line in latex:
            print(line)
        lines.append("\n## LaTeX\n\n```\n" + "\n".join(latex) + "\n```")

    ensure_dir(args.out_dir)
    report_path = os.path.join(args.out_dir, "results_table.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json(report, os.path.join(args.out_dir, "results_table.json"))
    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")
    return 0


# ---- linguistic-complexity ----

_DOC_LINGUISTIC_COMPLEXITY = """
Grounding accuracy split by linguistic complexity. CPU, ~20-60 s; re-reads predictions.p.

    python experiments/analysis.py linguistic-complexity         --predictions ours=outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p         --predictions 3DVG-Trans=outputs/<baseline>/predictions.p

The Unique/Multiple split measures object ambiguity; this bins the val set by language
complexity instead:

    tokens     description length, quartile bins
    depth      spaCy dependency-tree depth, quartile bins -- structural, not superficial
    neighbors  adjacent-object phrases the parser extracted
    spatial    spatial-relation cues, matched against spacy_parser's lexicon

With two or more models the per-bin gap is computed with a Spearman correlation of that gap
against bin index, so the trend across bins is visible.
"""

METRICS = ("tokens", "depth", "neighbors", "spatial")



def metric_tokens(rows, _args):
    return [len(row.get("token") or []) for row in rows]


def metric_depth(rows, args):
    """Dependency-tree depth per description, cached to disk (spaCy is the slow part)."""
    cache_path = args.depth_cache
    cache = {}
    if cache_path and os.path.isfile(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)

    def key_of(row):
        return f"{row['scene_id']}|{row['object_id']}|{row['ann_id']}"

    todo = [row for row in rows if key_of(row) not in cache]
    if todo:
        try:
            from experiments.ablation.parsing import load_parser
        except ImportError as error:
            raise SystemExit(f"the 'depth' metric needs spaCy: {error}")
        print(f"  computing dependency depth for {len(todo)} descriptions "
              f"({len(cache)} cached)...")
        nlp = load_parser(args.spacy_model)
        texts = [(row.get("description") or "").lower() for row in todo]
        for row, doc in zip(todo, nlp.pipe(texts, batch_size=256)):
            cache[key_of(row)] = _max_depth(doc)
        if cache_path:
            ensure_dir(os.path.dirname(os.path.abspath(cache_path)))
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"  cached depths -> {cache_path}")

    return [cache[key_of(row)] for row in rows]


def _max_depth(doc):
    """Longest head-chain from any token to its sentence root.

    Compared by token index, not identity: spaCy materialises a *new* ``Token`` object on
    every ``.head`` access, so ``node.head is not node`` is true even at the root and the
    walk never terminates.
    """
    best = 0
    for token in doc:
        depth, node, guard = 0, token, 0
        # `guard` protects against a malformed parse producing a head cycle.
        while node.head.i != node.i and guard < len(doc) + 1:
            depth += 1
            node = node.head
            guard += 1
        best = max(best, depth)
    return best


def metric_neighbors(rows, args):
    """Adjacent-object phrases our own parser extracted, per annotation."""
    cache = load_parse_cache(args.parse_folder, args.split, args.parsing_root)
    from experiments.ablation.parsing import NOT_MENTIONED_TOKENS

    values = []
    for row in rows:
        parsed = parse_for(cache, row["scene_id"], row["object_id"], row["ann_id"])
        if not parsed:
            values.append(0)
            continue
        tokens = parsed.get("neighbors") or []
        if not tokens or tokens == NOT_MENTIONED_TOKENS:
            values.append(0)
            continue
        # The field is a comma-separated list of relation phrases and the tokenizer
        # emits "," as its own token, so segments = commas + 1.
        values.append(tokens.count(",") + 1)
    return values


def _spatial_lexicon():
    from experiments.ablation.parsing import SPATIAL_PHRASES, SPATIAL_PREPS

    # Longest first, so "in front of" is consumed before "in".
    phrases = sorted((p.split() for p in SPATIAL_PHRASES), key=len, reverse=True)
    return phrases, set(SPATIAL_PREPS)


def spatial_hits(tokens):
    """Spatial-relation cues in a token list, as surface strings, non-overlapping."""
    phrases, preps = _spatial_lexicon()
    lowered = [str(t).lower() for t in tokens]
    hits, i = [], 0
    while i < len(lowered):
        matched = None
        for phrase in phrases:
            if lowered[i: i + len(phrase)] == phrase:
                matched = " ".join(phrase)
                i += len(phrase)
                break
        if matched is None:
            if lowered[i] in preps:
                matched = lowered[i]
            i += 1
        if matched:
            hits.append(matched)
    return hits


def metric_spatial(rows, _args):
    return [len(spatial_hits(row.get("token") or [])) for row in rows]


METRIC_FUNCTIONS = {
    "tokens": metric_tokens,
    "depth": metric_depth,
    "neighbors": metric_neighbors,
    "spatial": metric_spatial,
}

METRIC_LABELS = {
    "tokens": "description length (tokens)",
    "depth": "dependency-tree depth",
    "neighbors": "adjacent objects extracted by our parser",
    "spatial": "spatial-relation words (fixed lexicon)",
}

#: Metrics whose natural bins are the small integers themselves, not quantiles.
DISCRETE_METRICS = {"neighbors": 3, "spatial": 3}



def assign_bins(values, metric, num_bins):
    """Return (bin_index_per_row, bin_labels)."""
    values = np.asarray(values)
    if metric in DISCRETE_METRICS:
        top = DISCRETE_METRICS[metric]
        indices = np.minimum(values, top).astype(int)
        labels = [str(i) for i in range(top)] + [f"{top}+"]
        return indices, labels

    edges = quantile_edges(values, num_bins)
    if not edges:
        return np.zeros(len(values), dtype=int), ["all"]

    indices = np.digitize(values, edges, right=True)
    labels, low = [], int(values.min())
    for edge in edges:
        labels.append(f"{low}-{int(edge)}")
        low = int(edge) + 1
    labels.append(f"{low}+")
    return indices.astype(int), labels


def summarise(rows, indices, labels, models, thresholds):
    """Per-bin statistics for every model. Returns a list of bin dicts."""
    bins = []
    for index, label in enumerate(labels):
        mask = indices == index
        selected = [row for row, keep in zip(rows, mask) if keep]
        entry = {"bin": label, "n": len(selected), "models": {}}
        for name in models:
            ious = [row["ious"][name] for row in selected if name in row["ious"]]
            per_model = {"n": len(ious)}
            for threshold in thresholds:
                key = f"acc_{threshold}"
                value = accuracy(ious, threshold)
                hits = int(sum(1 for i in ious if i >= threshold))
                low, high = wilson_ci(hits, len(ious))
                per_model[key] = value
                per_model[f"{key}_ci"] = [low, high]
            entry["models"][name] = per_model
        bins.append(entry)
    return bins



def render_metric(metric, labels, bins, models, threshold, rows, values,
                  bin_indices, bootstrap=2000, seed=42):
    """(markdown table, findings dict) for one metric at one IoU threshold.

    Binned for the table, but every test runs on the per-sample data, not the bin means --
    a Spearman over four points reports spurious certainty. See details.md §7.
    """
    reference = models[0]
    others = models[1:]

    headers = ["bin", "n"] + [f"{m} Acc@{threshold}" for m in models]
    headers += [f"gap {reference}-{m}" for m in others]

    table_rows = []
    for entry in bins:
        line = [entry["bin"], entry["n"]]
        for name in models:
            stats = entry["models"][name]
            low, high = stats[f"acc_{threshold}_ci"]
            line.append(f"{pct(stats[f'acc_{threshold}'])} "
                        f"[{pct(low, 1)}-{pct(high, 1)}]")
        for name in others:
            gap = (entry["models"][reference][f"acc_{threshold}"]
                   - entry["models"][name][f"acc_{threshold}"])
            line.append(f"{100 * gap:+.2f}")
        table_rows.append(line)

    findings = {"threshold": threshold, "trends": {}, "gaps": {},
                "test": "per-sample Spearman (complexity value vs hit indicator)"}

    def hits(name):
        return [1.0 if row["ious"].get(name, -1.0) >= threshold else 0.0 for row in rows]

    reference_hits = hits(reference)
    rho, p = spearman(list(values), reference_hits)
    findings["trends"][reference] = {
        "spearman_rho": rho, "p_value": p, "n": len(rows),
        "significant": bool(p == p and p < 0.05),
    }

    for name in others:
        # +1 where only the reference is right, -1 where only the other is, 0 where they
        # agree -- correlating that against complexity is the question, at full n.
        other_hits = hits(name)
        advantage = [a - b for a, b in zip(reference_hits, other_hits)]
        rho_gap, p_gap = spearman(list(values), advantage)

        reference_array = np.asarray(reference_hits, dtype=bool)
        other_array = np.asarray(other_hits, dtype=bool)
        indices = np.asarray(bin_indices)

        # Per-bin McNemar: both models saw the same annotations in each bin, so the gap
        # inside a bin is a paired comparison and deserves its own p-value.
        per_bin = []
        for index in range(len(bins)):
            mask = indices == index
            if not mask.any():
                per_bin.append({"gap": 0.0, "n": 0, "p_value": float("nan")})
                continue
            a, b = reference_array[mask], other_array[mask]
            _a_only, _b_only, _chi2, p_bin = mcnemar(a, b)
            per_bin.append({"gap": float(a.mean() - b.mean()), "n": int(mask.sum()),
                            "p_value": p_bin})

        # Does the gap in the most complex bin exceed the gap in the least complex one?
        # A bootstrap over annotations gives a CI for that difference directly.
        populated = [i for i, entry in enumerate(per_bin) if entry["n"] > 0]
        widening = {"available": False}
        if len(populated) >= 2:
            low_bin, high_bin = populated[0], populated[-1]

            def difference_of_gaps(idx, low=low_bin, high=high_bin):
                sub_indices = indices[idx]
                a, b = reference_array[idx], other_array[idx]
                low_mask, high_mask = sub_indices == low, sub_indices == high
                if not low_mask.any() or not high_mask.any():
                    return float("nan")
                return float((a[high_mask].mean() - b[high_mask].mean())
                             - (a[low_mask].mean() - b[low_mask].mean()))

            point, ci_low, ci_high = bootstrap_ci(
                difference_of_gaps, len(rows), num_resamples=bootstrap, seed=seed)
            widening = {
                "available": True,
                "low_bin": bins[low_bin]["bin"], "high_bin": bins[high_bin]["bin"],
                "difference": point, "ci": [ci_low, ci_high],
                "significant": bool(ci_low > 0),
                "directional": bool(point > 0),
                "resamples": bootstrap,
            }

        findings["gaps"][name] = {
            "per_bin": [entry["gap"] for entry in per_bin],
            "per_bin_detail": per_bin,
            "spearman_rho": rho_gap,
            "p_value": p_gap,
            "n": len(rows),
            "widening": widening,
            "widens_with_complexity": bool(widening.get("significant")),
        }

    return md_table(headers, table_rows), findings


def verdict_lines(metric, findings, models):
    """The honest read of one metric, in plain sentences."""
    reference = models[0]
    lines = []
    trend = findings["trends"][reference]
    rho, p, n = trend["spearman_rho"], trend["p_value"], trend["n"]
    direction = "falls" if rho < 0 else "rises"
    strength = "significantly" if trend["significant"] else "but NOT significantly"
    lines.append(f"  {reference}: accuracy {direction} as {METRIC_LABELS[metric]} grows, "
                 f"{strength} (per-sample Spearman rho={rho:+.4f}, p={p:.2e}, n={n}).")

    if not findings["gaps"]:
        lines.append("  No baseline supplied, so this says nothing about whether the "
                     "advantage over prior work grows with complexity. Add --predictions "
                     "NAME=PATH for a baseline.")
        return lines

    for name, gap in findings["gaps"].items():
        widening = gap.get("widening") or {}
        if not widening.get("available"):
            lines.append(f"  vs {name}: too few populated bins to test widening.")
            continue

        low, high = widening["ci"]
        summary = (f"gap in '{widening['high_bin']}' minus gap in "
                   f"'{widening['low_bin']}' = {100 * widening['difference']:+.2f} pp, "
                   f"95% CI [{100 * low:+.2f}, {100 * high:+.2f}]")

        if widening["significant"]:
            lines.append(f"  vs {name}: the advantage WIDENS with complexity -- "
                         f"{summary}. The CI excludes zero, so this supports the paper's "
                         f"motivation directly. Report it.")
        elif widening["directional"]:
            lines.append(
                f"  vs {name}: the advantage widens DIRECTIONALLY but not significantly "
                f"-- {summary}. The CI includes zero. State the trend and the CI, and "
                f"soften the claim to 'consistent with' rather than 'demonstrates'. Do "
                f"not report the point estimate alone.")
        else:
            lines.append(f"  vs {name}: the advantage does NOT widen with complexity -- "
                         f"{summary}. Report this honestly and drop the claim that the "
                         f"method's advantage is specific to complex language.")

        significant_bins = [entry for entry in gap["per_bin_detail"]
                            if entry["p_value"] == entry["p_value"]
                            and entry["p_value"] < 0.05]
        lines.append(f"      per-bin gaps (McNemar): " + "  ".join(
            f"{100 * entry['gap']:+.2f}{'*' if entry['p_value'] == entry['p_value'] and entry['p_value'] < 0.05 else ''}"
            for entry in gap["per_bin_detail"]) +
            f"   ({len(significant_bins)}/{len(gap['per_bin_detail'])} bins significant, "
            f"* = p<0.05)")
    return lines


def relation_type_table(rows, models, threshold, top_k):
    """Accuracy per spatial-relation type, not just per relation count."""
    from collections import Counter

    counts = Counter()
    per_type = {}
    for row in rows:
        for relation in set(spatial_hits(row.get("token") or [])):
            counts[relation] += 1
            per_type.setdefault(relation, []).append(row)

    headers = ["relation", "n"] + [f"{m} Acc@{threshold}" for m in models]
    table_rows = []
    for relation, count in counts.most_common(top_k):
        line = [relation, count]
        for name in models:
            ious = [r["ious"][name] for r in per_type[relation] if name in r["ious"]]
            line.append(pct(accuracy(ious, threshold)))
        table_rows.append(line)
    return md_table(headers, table_rows), dict(counts.most_common(top_k))


def _lc_make_plot(results, models, threshold, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the plot")
        return None

    metrics = list(results)
    figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.2),
                                squeeze=False)
    for axis, metric in zip(axes[0], metrics):
        bins = results[metric]["bins"]
        labels = [entry["bin"] for entry in bins]
        x = range(len(labels))

        counts = [entry["n"] for entry in bins]
        twin = axis.twinx()
        twin.bar(x, counts, color="0.88", zorder=0)
        twin.set_ylabel("samples", color="0.5")
        twin.tick_params(axis="y", colors="0.5")

        for name in models:
            y = [100 * entry["models"][name][f"acc_{threshold}"] for entry in bins]
            axis.plot(x, y, marker="o", label=name, zorder=3)

        axis.set_zorder(twin.get_zorder() + 1)
        axis.patch.set_visible(False)
        axis.set_xticks(list(x))
        axis.set_xticklabels(labels, rotation=0)
        axis.set_xlabel(METRIC_LABELS[metric])
        axis.set_ylabel(f"Acc@{threshold} (%)")
        axis.grid(alpha=0.3)
        axis.legend(loc="lower left", fontsize=8)

    figure.suptitle(f"Grounding accuracy by linguistic complexity (Acc@{threshold})")
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _add_linguistic_complexity_args(parser):
    parser.add_argument("--predictions", action="append", default=None, metavar="NAME=PATH",
                        help="Repeatable. The first one is the reference model in every "
                             "gap column. Defaults to the newest outputs/*/predictions.p.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--metrics", nargs="+", default=list(METRICS), choices=METRICS)
    parser.add_argument("--num-bins", dest="num_bins", type=int, default=4,
                        help="Quantile bins for the continuous metrics (tokens, depth).")
    parser.add_argument("--threshold", type=float, default=0.25, choices=list(THRESHOLDS),
                        help="IoU threshold used for the tables, plot and verdicts.")
    parser.add_argument("--parse-folder", dest="parse_folder",
                        default="final_parsing_tokenized",
                        help="Parse cache backing the 'neighbors' metric.")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--spacy-model", dest="spacy_model", default="en_core_web_sm")
    parser.add_argument("--depth-cache", dest="depth_cache",
                        default="outputs/analysis/dep_depth_cache.json")
    parser.add_argument("--top-relations", dest="top_relations", type=int, default=12)
    parser.add_argument("--bootstrap", type=int, default=2000,
                        help="Bootstrap resamples for the widening CI; 0 is not "
                             "recommended -- the CI is the headline statistic.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/linguistic_complexity")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser


def run_linguistic_complexity(args):

    specs = args.predictions
    if not specs:
        found = find_default_predictions()
        if not found:
            args._parser.error("no --predictions given and no outputs/*/predictions.p found")
        specs = [found[0]]
        print(f"[auto] using {found[0]}")

    records = load_scanrefer(args.split, args.data_root)

    models, base_rows = [], None
    for spec in specs:
        name, path = parse_predictions_arg(spec)
        rows, missing = join(load_predictions(path), records)
        print(f"[{name}] {len(rows)} joined from {path}" +
              (f"  ({missing} annotations had no prediction)" if missing else ""))
        models.append(name)
        if base_rows is None:
            base_rows = {r_key(row): dict(row, ious={}) for row in rows}
        for row in rows:
            entry = base_rows.get(r_key(row))
            if entry is not None:
                entry["ious"][name] = row["iou"]

    rows = list(base_rows.values())
    if len(models) > 1:
        # Only annotations every model predicted can support a gap column.
        complete = [row for row in rows if len(row["ious"]) == len(models)]
        if len(complete) != len(rows):
            print(f"[join] restricting to {len(complete)}/{len(rows)} annotations "
                  f"predicted by all {len(models)} models")
        rows = complete

    ensure_dir(args.out_dir)
    results, report_lines = {}, []
    report_lines.append("# Grounding accuracy by linguistic complexity\n")
    report_lines.append(f"split: `{args.split}` | annotations: {len(rows)} | "
                        f"models: {', '.join(models)} | IoU threshold: {args.threshold}\n")

    for metric in args.metrics:
        print(f"\n=== {metric}: {METRIC_LABELS[metric]} ===")
        values = METRIC_FUNCTIONS[metric](rows, args)
        indices, labels = assign_bins(values, metric, args.num_bins)
        bins = summarise(rows, indices, labels, models, THRESHOLDS)
        table, findings = render_metric(metric, labels, bins, models, args.threshold,
                                        rows, values, indices, args.bootstrap,
                                        args.seed)

        print(table)
        lines = verdict_lines(metric, findings, models)
        for line in lines:
            print(line)

        results[metric] = {
            "label": METRIC_LABELS[metric],
            "values_summary": {
                "min": int(np.min(values)), "max": int(np.max(values)),
                "mean": float(np.mean(values)), "median": float(np.median(values)),
            },
            "bins": bins,
            "findings": findings,
        }
        report_lines.append(f"\n## {metric} — {METRIC_LABELS[metric]}\n")
        report_lines.append(table)
        report_lines.append("\n**Reading:**\n")
        report_lines.extend(line.strip() + "\n" for line in lines)

    if "spatial" in args.metrics:
        print("\n=== accuracy per spatial-relation type ===")
        table, counts = relation_type_table(rows, models, args.threshold, args.top_relations)
        print(table)
        results["relation_types"] = {"counts": counts}
        report_lines.append("\n## Accuracy by spatial-relation type\n")
        report_lines.append("The metric above covers the number of spatial relations; "
                            "this covers their type.\n")
        report_lines.append(table)

    json_path = save_json({
        "split": args.split,
        "models": models,
        "num_annotations": len(rows),
        "threshold": args.threshold,
        "num_bins": args.num_bins,
        "results": results,
    }, os.path.join(args.out_dir, "linguistic_complexity.json"))

    report_path = os.path.join(args.out_dir, "linguistic_complexity.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nwrote {json_path}")
    print(f"wrote {report_path}")

    if args.plot:
        plot_metrics = {m: results[m] for m in args.metrics}
        plot_path = _lc_make_plot(plot_metrics, models, args.threshold,
                              os.path.join(args.out_dir, "linguistic_complexity.png"))
        if plot_path:
            print(f"wrote {plot_path}")

    if len(models) == 1:
        print("\nNOTE: only one model was supplied. Whether the advantage over a "
              "baseline grows\n      with complexity needs a second --predictions "
              "NAME=PATH. See the docstring\n      for the required format.")
    return 0


def r_key(row):
    return (str(row["scene_id"]), str(row["object_id"]), str(row["ann_id"]))


# ---- parse-quality-split ----

_DOC_PARSE_QUALITY_SPLIT = """
Grounding accuracy on the correctly parsed subset vs the incorrectly parsed one. CPU.

    python experiments/analysis.py parse-quality-split         --predictions outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p         --parse gpt4o-mini=final_parsing_tokenized

ScanRefer's object_name is free ground truth for the target slot, which labels all 9,508
val annotations at no cost. The raw comparison is confounded -- descriptions the parser
gets wrong skew towards rare classes -- so two numbers come out: the raw difference, and
the same difference computed within each object_name class and pooled
(Cochran-Mantel-Haenszel). A controlled difference much smaller than the raw one means
class composition drove it.

Observational, not an intervention; parse_error_propagation.py is the intervening version.
"""

CRITERIA = ("exact", "substring", "fuzzy")


def label_rows(rows, cache, criterion):
    """Split rows by whether the parser's target matches object_name."""
    correct, wrong, uncovered = [], [], 0
    for row in rows:
        parsed = parse_for(cache, row["scene_id"], row["object_id"], row["ann_id"])
        if not parsed:
            uncovered += 1
            continue
        predicted = " ".join(parsed.get("target") or [])
        kinds = match_kinds(predicted, row["object_name"])
        row = dict(row, _predicted_target=predicted)
        (correct if kinds[criterion] else wrong).append(row)
    return correct, wrong, uncovered


def subset_stats(rows, threshold):
    ious = [row["iou"] for row in rows]
    hits = int(sum(1 for i in ious if i >= threshold))
    low, high = wilson_ci(hits, len(ious))
    return {"n": len(ious), "hits": hits, "acc": accuracy(ious, threshold),
            "ci": [low, high]}


def controlled_difference(correct, wrong, threshold):
    """Cochran-Mantel-Haenszel style difference, stratified by object_name.

    Within each class the two subsets are directly comparable; across classes they are
    not. Weighting each class by the harmonic mean of its two subset sizes gives the
    classes that actually carry information the most influence, and gives zero weight to
    classes present in only one subset.
    """
    by_class = defaultdict(lambda: {"correct": [], "wrong": []})
    for row in correct:
        by_class[row["object_name"]]["correct"].append(row["iou"])
    for row in wrong:
        by_class[row["object_name"]]["wrong"].append(row["iou"])

    total_weight, weighted_sum, used = 0.0, 0.0, 0
    observed, expected, variance = 0.0, 0.0, 0.0

    for _class_name, group in by_class.items():
        n1, n2 = len(group["correct"]), len(group["wrong"])
        if n1 == 0 or n2 == 0:
            continue
        weight = 2.0 * n1 * n2 / (n1 + n2)          # harmonic mean of the two sizes
        difference = (accuracy(group["correct"], threshold)
                      - accuracy(group["wrong"], threshold))
        weighted_sum += weight * difference
        total_weight += weight
        used += 1

        # Cochran-Mantel-Haenszel accumulators for the stratified 2x2xK test.
        a = sum(1 for i in group["correct"] if i >= threshold)
        c = sum(1 for i in group["wrong"] if i >= threshold)
        total = n1 + n2
        hit_total = a + c
        miss_total = total - hit_total
        observed += a
        expected += n1 * hit_total / total
        if total > 1:
            variance += (n1 * n2 * hit_total * miss_total) / (total * total * (total - 1))

    if total_weight == 0:
        return {"difference": float("nan"), "classes_used": 0, "coverage": 0.0,
                "cmh_chi2": float("nan"), "cmh_p": float("nan")}

    if variance > 0:
        chi2 = (abs(observed - expected) - 0.5) ** 2 / variance
        # Survival function of chi-square with 1 degree of freedom, in closed form.
        cmh_p = math.erfc(math.sqrt(chi2 / 2.0))
    else:
        chi2, cmh_p = float("nan"), float("nan")

    covered = sum(len(g["correct"]) + len(g["wrong"]) for g in by_class.values()
                  if g["correct"] and g["wrong"])
    return {
        "difference": weighted_sum / total_weight,
        "classes_used": used,
        "classes_total": len(by_class),
        "coverage": covered / max(len(correct) + len(wrong), 1),
        "cmh_chi2": chi2,
        "cmh_p": cmh_p,
    }


def composition(rows, top_k=8):
    counts = Counter(row["object_name"] for row in rows)
    total = max(sum(counts.values()), 1)
    return [(name, count, 100.0 * count / total) for name, count in counts.most_common(top_k)]


def _add_parse_quality_split_args(parser):
    parser.add_argument("--predictions", required=True,
                        help="outputs/<run>/predictions.p, optionally NAME=PATH.")
    parser.add_argument("--parse", action="append", required=True, metavar="NAME=FOLDER",
                        help="Repeatable. Parse cache under data_parsing/ to score.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--criterion", default="fuzzy", choices=CRITERIA,
                        help="Which target-match criterion counts as a correct parse.")
    parser.add_argument("--threshold", type=float, default=0.25, choices=list(THRESHOLDS))
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/parse_quality_split")
    return parser


def run_parse_quality_split(args):

    model_name, predictions_path = parse_predictions_arg(args.predictions)
    records = load_scanrefer(args.split, args.data_root)
    rows, missing = join(load_predictions(predictions_path), records)
    print(f"[{model_name}] {len(rows)} annotations joined from {predictions_path}" +
          (f"  ({missing} without a prediction)" if missing else ""))
    print(f"[criterion] a parse counts as correct when its target is a "
          f"{args.criterion!r} match to object_name\n")

    report, lines = {}, ["# Grounding accuracy vs parse correctness\n",
                         f"model: `{model_name}` | split: `{args.split}` | "
                         f"criterion: `{args.criterion}` | "
                         f"IoU threshold: {args.threshold}\n"]

    summary_rows = []
    for spec in args.parse:
        name, folder = (spec.split("=", 1) if "=" in spec else (spec, spec))
        name, folder = name.strip(), folder.strip()
        try:
            cache = load_parse_cache(folder, args.split, args.parsing_root)
        except FileNotFoundError as error:
            print(f"[{name}] SKIPPED: {error}")
            continue

        correct, wrong, uncovered = label_rows(rows, cache, args.criterion)
        good = subset_stats(correct, args.threshold)
        bad = subset_stats(wrong, args.threshold)
        difference, z, p = two_proportion_z(good["hits"], good["n"], bad["hits"], bad["n"])
        controlled = controlled_difference(correct, wrong, args.threshold)

        print(f"=== {name}  ({folder}) ===")
        if uncovered:
            print(f"  {uncovered} annotations absent from this parse cache; excluded")
        print(f"  parse correct : n={good['n']:>5}  Acc@{args.threshold}="
              f"{pct(good['acc'])}%  [{pct(good['ci'][0], 1)}-{pct(good['ci'][1], 1)}]")
        print(f"  parse wrong   : n={bad['n']:>5}  Acc@{args.threshold}="
              f"{pct(bad['acc'])}%  [{pct(bad['ci'][0], 1)}-{pct(bad['ci'][1], 1)}]")
        print(f"  raw difference          : {100 * difference:+.2f} pp "
              f"(z={z:.2f}, p={p:.3e})")
        print(f"  category-controlled     : {100 * controlled['difference']:+.2f} pp "
              f"(CMH chi2={controlled['cmh_chi2']:.2f}, p={controlled['cmh_p']:.3e}) "
              f"over {controlled['classes_used']}/{controlled.get('classes_total', 0)} "
              f"classes covering {100 * controlled['coverage']:.0f}% of annotations")

        if difference and controlled["difference"] == controlled["difference"]:
            ratio = controlled["difference"] / difference
            if ratio > 1.15:
                print(f"  -> controlling AMPLIFIES the gap ({ratio:.1f}x). Class "
                      f"composition was masking it: the misparsed subset happens to "
                      f"contain easier classes, so the raw number understates the "
                      f"association. Quote the controlled number.")
            elif ratio < 0.85:
                print(f"  -> controlling SHRINKS the gap to {100 * ratio:.0f}% of raw. "
                      f"Most of the raw difference was class composition, not the "
                      f"parse. Quote the controlled number and say so.")
            else:
                print("  -> controlling barely moves the gap; class composition is not "
                      "driving it.")

        print("  composition of each subset (top classes):")
        for label, subset in (("correct", correct), ("wrong", wrong)):
            top = ", ".join(f"{n} {c:.0f}%" for n, _k, c in composition(subset, 5))
            print(f"    {label:<8}: {top}")
        print()

        report[name] = {
            "folder": folder, "uncovered": uncovered,
            "correct": good, "wrong": bad,
            "raw_difference": difference, "z": z, "p_value": p,
            "category_controlled": controlled,
            "composition": {
                "correct": composition(correct), "wrong": composition(wrong)},
        }
        summary_rows.append([
            name, good["n"], f"{pct(good['acc'])}", bad["n"], f"{pct(bad['acc'])}",
            f"{100 * difference:+.2f}", f"{p:.2e}",
            f"{100 * controlled['difference']:+.2f}", f"{controlled['cmh_p']:.2e}",
        ])

    if not summary_rows:
        print("No parse cache could be loaded; nothing to report.")
        return 1

    table = md_table(
        ["parser", "n correct", f"Acc@{args.threshold} correct", "n wrong",
         f"Acc@{args.threshold} wrong", "raw diff (pp)", "raw p",
         "controlled diff (pp)", "CMH p"],
        summary_rows)
    print(table)

    lines.append(table)
    lines.append("\n**raw diff** is confounded: wrongly parsed descriptions skew towards "
                 "rare classes and unusual phrasing, which are harder to ground "
                 "regardless of the parse.\n")
    lines.append("**controlled diff** recomputes the same difference within each "
                 "`object_name` class and averages it with Cochran-Mantel-Haenszel "
                 "weights, so class composition cannot drive it.\n")
    lines.append("Neither is an intervention. Pair this with the corruption test "
                 "(`experiments/analysis.py error-propagation`), which holds the sample fixed "
                 "and changes only the parse.\n")

    ensure_dir(args.out_dir)
    json_path = save_json({
        "model": model_name, "predictions": predictions_path,
        "split": args.split, "criterion": args.criterion,
        "threshold": args.threshold, "parsers": report,
    }, os.path.join(args.out_dir, "parse_quality_split.json"))
    report_path = os.path.join(args.out_dir, "parse_quality_split.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\nwrote {json_path}")
    print(f"wrote {report_path}")
    return 0


# ---- failure-cases ----

_DOC_FAILURE_CASES = """
Select grounding failures and attribute a cause to each. CPU, seconds.

    python experiments/analysis.py failure-cases         --predictions outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p --top 6

Every signal is computable offline from artifacts already on disk. Rules are evaluated in
order and the first match wins, since an earlier cause subsumes the later ones -- if the
parser named the wrong class, the distractor count downstream is beside the point.

    parse_target_wrong    the parser's target does not match object_name
    distractor_confusion  low IoU with several instances of the target class in the scene
    small_object          GT box volume in the bottom --small-percentile
    localization_drift    IoU in [drift-low, 0.25): right region, imprecise box
    complex_language      long description or many spatial cues, no cause above
    unattributed          none of the above

Which cases get dumped is --strategy:

    diverse   (default) group by cause, sort each group by ascending IoU, then
              round-robin over causes ordered by how common they are. So each case is
              the *worst* example of its cause, commonest cause first -- the six default
              cases are that rule's output, not a hand-picked set.
    worst     simply the --top lowest IoUs, which in practice are all one cause.
    pick      list the failures grouped by cause and let you choose by number:

                  python experiments/analysis.py failure-cases --predictions ... \\
                      --strategy pick --cause small_object --iou-max 0.05

              Answer with a comma list or ranges ("3,17" / "1-6,20"); the order you type
              is the order they are numbered. --pick-file reads the same syntax from a
              file, so a chosen set re-runs without a terminal.

--cause, --scene, --iou-min and --iou-max narrow the pool before any strategy runs.

Writes outputs/analysis/failure_cases/ with a report, a JSON record, and per case a
pred.ply / gt.ply / case.txt. The report prints the matching scripts/visualize.py command
per case, for rendering the full scene mesh behind the boxes. To open a case in a real
point-cloud viewer instead, see failure-scenes.
"""

CAUSES = ("parse_target_wrong", "distractor_confusion", "small_object",
          "localization_drift", "complex_language", "unattributed")

CAUSE_TEXT = {
    "parse_target_wrong":
        "the parser extracted the wrong target class, so the grounding module was "
        "conditioned on the wrong object category",
    "distractor_confusion":
        "the correct category was identified but the wrong instance was selected; the "
        "scene contains several objects of that class",
    "small_object":
        "the referred object is among the smallest in the split, so the detector's "
        "proposals resolve it poorly",
    "localization_drift":
        "the correct region was found but the predicted box is too imprecise to pass "
        "the IoU threshold",
    "complex_language":
        "a long description with many spatial relations and no simpler failure cause",
    "unattributed":
        "none of the diagnosed causes applies",
}


def scene_class_counts(records):
    """{scene_id: {object_name: number of distinct instances}} from ScanRefer itself."""
    instances = defaultdict(lambda: defaultdict(set))
    for record in records:
        instances[record["scene_id"]][record["object_name"]].add(record["object_id"])
    return {scene: {name: len(ids) for name, ids in classes.items()}
            for scene, classes in instances.items()}


def attribute(row, signals, args):
    if not signals["parse_target_ok"]:
        return "parse_target_wrong"
    if row["iou"] < args.distractor_iou and signals["same_class_instances"] >= 2:
        return "distractor_confusion"
    if signals["gt_volume"] <= signals["small_threshold"]:
        return "small_object"
    if args.drift_low <= row["iou"] < 0.25:
        return "localization_drift"
    if (signals["num_tokens"] >= signals["long_threshold"]
            or signals["num_spatial"] >= args.many_spatial):
        return "complex_language"
    return "unattributed"


def select(cases, top, strategy):
    """Pick `top` cases, either the worst overall or spread across causes."""
    if strategy == "worst":
        return sorted(cases, key=lambda c: c["iou"])[:top]

    by_cause = defaultdict(list)
    for case in cases:
        by_cause[case["cause"]].append(case)
    for group in by_cause.values():
        group.sort(key=lambda c: c["iou"])

    # Round-robin over causes ordered by how common they are, so the sample is diverse
    # but still weighted towards the failure modes that actually dominate.
    order = [cause for cause, _ in Counter(c["cause"] for c in cases).most_common()]
    chosen, index = [], 0
    while len(chosen) < top and any(by_cause[c] for c in order):
        cause = order[index % len(order)]
        if by_cause[cause]:
            chosen.append(by_cause[cause].pop(0))
        index += 1
    return chosen


def filter_failures(failures, causes=(), scenes=(), iou_min=None, iou_max=None):
    """Narrow the failure list before it is listed or sampled. Order is preserved."""
    kept = failures
    if causes:
        kept = [c for c in kept if c["cause"] in set(causes)]
    if scenes:
        kept = [c for c in kept if c["scene_id"] in set(scenes)]
    if iou_min is not None:
        kept = [c for c in kept if c["iou"] >= iou_min]
    if iou_max is not None:
        kept = [c for c in kept if c["iou"] <= iou_max]
    return kept


def format_failure_listing(failures, limit=25):
    """(text, ordered_cases) -- failures grouped by cause, each group worst IoU first.

    Indices run globally across the whole listing so a single flat answer like "3,17-20"
    addresses any row regardless of which group it sits in.
    """
    by_cause = defaultdict(list)
    for case in failures:
        by_cause[case["cause"]].append(case)

    lines, ordered = [], []
    for cause, _ in Counter(c["cause"] for c in failures).most_common():
        group = sorted(by_cause[cause], key=lambda c: c["iou"])
        shown = group[:limit] if limit else group
        suffix = f", showing {len(shown)}" if len(shown) < len(group) else ""
        lines.append(f"\n== {cause}  ({len(group)} failures{suffix}) ==")
        for case in shown:
            ordered.append(case)
            description = (case.get("description") or "").replace("\n", " ")
            if len(description) > 58:
                description = description[:57] + "…"
            lines.append(
                f"  [{len(ordered):3d}] IoU {case['iou']:.3f}  {case['scene_id']} "
                f"obj{case['object_id']:<4} ann{case['ann_id']:<3} "
                f"{(case['object_name'] or '?')[:14]:<14} \"{description}\"")
    return "\n".join(lines), ordered


def parse_selection(text, count):
    """'1-3,17' -> [0, 1, 2, 16]. Raises ValueError naming the offending token.

    Duplicates are dropped but the order typed is kept, so the caller controls the
    numbering of what gets rendered.
    """
    chosen = []
    for token in (text or "").replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token[1:]:
            start, _, end = token.partition("-")
            bounds = (start, end)
        else:
            bounds = (token, token)
        try:
            low, high = int(bounds[0]), int(bounds[1])
        except ValueError:
            raise ValueError(f"{token!r} is not a number or a range")
        if low > high:
            low, high = high, low
        if low < 1 or high > count:
            raise ValueError(f"{token!r} is outside 1-{count}")
        for index in range(low - 1, high):
            if index not in chosen:
                chosen.append(index)
    if not chosen:
        raise ValueError("nothing selected")
    return chosen


def pick_cases(failures, limit=25, selection=None, top=None):
    """Print the grouped listing and resolve a selection into cases.

    A ``selection`` skips the prompt entirely -- that is what --pick-file and the
    notebook pass -- and a bad one fails rather than falling back to stdin, since
    neither caller has a person waiting at a terminal. Returns [] when the user
    declines, so the caller writes nothing.
    """
    listing, ordered = format_failure_listing(failures, limit)
    if not ordered:
        return []
    print(listing)
    print(f"\n{len(ordered)} case(s) listed. Select by number: "
          f"e.g. 3,17 or 1-6 (blank to abort).")

    if selection is not None:
        if not selection.strip():
            print("nothing selected")
            return []
        indices = parse_selection(selection, len(ordered))     # raises: no one to re-ask
        chosen = [ordered[i] for i in indices]
        return chosen[:top] if top else chosen

    while True:
        try:
            typed = input("cases> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nnothing selected")
            return []
        if not typed:
            print("nothing selected")
            return []
        try:
            indices = parse_selection(typed, len(ordered))
        except ValueError as error:
            print(f"  {error} -- try again")
            continue
        chosen = [ordered[i] for i in indices]
        return chosen[:top] if top else chosen


def dump_case(case, out_dir, args):
    directory = ensure_dir(os.path.join(
        out_dir, f"{case['rank']:02d}_{case['cause']}_"
                 f"{case['scene_id']}_obj{case['object_id']}_ann{case['ann_id']}"))

    write_bbox_ply(case["_pred_bbox"], os.path.join(directory, "pred.ply"),
                   color=(220, 30, 30), radius=args.box_radius)
    write_bbox_ply(case["_gt_bbox"], os.path.join(directory, "gt.ply"),
                   color=(30, 190, 60), radius=args.box_radius)

    with open(os.path.join(directory, "case.txt"), "w") as f:
        f.write(f"scene       : {case['scene_id']}\n")
        f.write(f"object      : {case['object_id']} ({case['object_name']})\n")
        f.write(f"annotation  : {case['ann_id']}\n")
        f.write(f"IoU         : {case['iou']:.4f}\n")
        f.write(f"cause       : {case['cause']}\n")
        f.write(f"              {CAUSE_TEXT[case['cause']]}\n\n")
        f.write(f"description : {case['description']}\n\n")
        f.write(f"parsed target     : {case['parsed_target']!r}"
                f"  (ground truth {case['object_name']!r})\n")
        f.write(f"parsed adjectives : {case['parsed_adjectives']!r}\n")
        f.write(f"parsed neighbors  : {case['parsed_neighbors']!r}\n\n")
        f.write("signals\n")
        for key, value in case["signals"].items():
            f.write(f"  {key:<22}: {value}\n")
        f.write("\nboxes: pred.ply (red) and gt.ply (green), wireframe PLY.\n")
        f.write("Full scene mesh:\n")
        f.write(f"  {case['visualize_command']}\n")
    return directory


def _add_failure_cases_args(parser):
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parse-folder", dest="parse_folder",
                        default="final_parsing_tokenized")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--criterion", default="fuzzy",
                        choices=["exact", "substring", "fuzzy"])
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="A failure is IoU below this.")
    parser.add_argument("--top", type=int, default=6,
                        help="How many cases to dump.")
    parser.add_argument("--strategy", default="diverse",
                        choices=["diverse", "worst", "pick"])
    parser.add_argument("--pick-limit", dest="pick_limit", type=int, default=25,
                        help="rows listed per cause under --strategy pick; 0 for all")
    parser.add_argument("--pick-file", dest="pick_file",
                        help="read the selection from a file instead of the prompt, so "
                             "a chosen set is reproducible and needs no terminal")
    parser.add_argument("--cause", dest="causes", action="append", choices=list(CAUSES),
                        help="keep only these causes; repeatable")
    parser.add_argument("--scene", dest="scenes", action="append",
                        help="keep only these scene ids; repeatable")
    parser.add_argument("--iou-min", dest="iou_min", type=float)
    parser.add_argument("--iou-max", dest="iou_max", type=float)
    parser.add_argument("--distractor-iou", dest="distractor_iou", type=float, default=0.05)
    parser.add_argument("--drift-low", dest="drift_low", type=float, default=0.10)
    parser.add_argument("--small-percentile", dest="small_percentile", type=float, default=10.0)
    parser.add_argument("--long-percentile", dest="long_percentile", type=float, default=75.0)
    parser.add_argument("--many-spatial", dest="many_spatial", type=int, default=3)
    parser.add_argument("--box-radius", dest="box_radius", type=float, default=0.02)
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/failure_cases")
    parser.add_argument("--no-ply", dest="write_ply", action="store_false")
    return parser


def run_failure_cases(args):

    model_name, predictions_path = parse_predictions_arg(args.predictions)
    records = load_scanrefer(args.split, args.data_root)
    rows, missing = join(load_predictions(predictions_path), records)
    print(f"[{model_name}] {len(rows)} annotations joined" +
          (f"  ({missing} without a prediction)" if missing else ""))

    try:
        cache = load_parse_cache(args.parse_folder, args.split, args.parsing_root)
    except FileNotFoundError as error:
        print(f"WARNING: {error}\n         parse-based causes will be unavailable")
        cache = {}

    counts = scene_class_counts(records)
    volumes = np.array([box_volume(row["_prediction"]["gt_bbox"]) for row in rows])
    small_threshold = float(np.percentile(volumes, args.small_percentile))
    lengths = np.array([len(row.get("token") or []) for row in rows])
    long_threshold = float(np.percentile(lengths, args.long_percentile))
    print(f"[thresholds] small object <= {small_threshold:.3f} m^3 "
          f"(p{args.small_percentile:g}) | long description >= {long_threshold:.0f} tokens "
          f"(p{args.long_percentile:g})")

    failures = []
    for row in rows:
        if row["iou"] >= args.threshold:
            continue
        parsed = parse_for(cache, row["scene_id"], row["object_id"], row["ann_id"]) or {}
        predicted_target = " ".join(parsed.get("target") or [])
        target_ok = (match_kinds(predicted_target, row["object_name"])[args.criterion]
                     if parsed else True)

        signals = {
            "parse_target_ok": bool(target_ok),
            "same_class_instances": counts[row["scene_id"]].get(row["object_name"], 1),
            "gt_volume": round(box_volume(row["_prediction"]["gt_bbox"]), 4),
            "small_threshold": round(small_threshold, 4),
            "num_tokens": int(len(row.get("token") or [])),
            "long_threshold": int(long_threshold),
            "num_spatial": len(spatial_hits(row.get("token") or [])),
        }
        cause = attribute(row, signals, args)

        failures.append({
            "scene_id": row["scene_id"], "object_id": row["object_id"],
            "ann_id": row["ann_id"], "object_name": row["object_name"],
            "description": row.get("description", ""),
            "iou": row["iou"], "cause": cause, "signals": signals,
            "parsed_target": predicted_target or "(no parse)",
            "parsed_adjectives": " ".join(parsed.get("adjectives") or []) or "(no parse)",
            "parsed_neighbors": " ".join(parsed.get("neighbors") or []) or "(no parse)",
            "_pred_bbox": row["_prediction"]["pred_bbox"],
            "_gt_bbox": row["_prediction"]["gt_bbox"],
        })

    total = len(rows)
    print(f"\n[failures] {len(failures)} of {total} annotations "
          f"({100.0 * len(failures) / max(total, 1):.2f}%) fall below IoU {args.threshold}\n")

    tally = Counter(case["cause"] for case in failures)
    distribution = md_table(
        ["cause", "count", "% of failures", "% of all"],
        [[cause, tally[cause],
          f"{100.0 * tally[cause] / max(len(failures), 1):.1f}",
          f"{100.0 * tally[cause] / max(total, 1):.1f}"]
         for cause in CAUSES if tally[cause]])
    print(distribution)

    # Filters narrow the pool for every strategy, not just the picker: --cause
    # small_object --strategy worst is a useful thing to ask for on its own.
    pool = filter_failures(failures, getattr(args, "causes", None) or (),
                           getattr(args, "scenes", None) or (),
                           getattr(args, "iou_min", None), getattr(args, "iou_max", None))
    if len(pool) != len(failures):
        print(f"[filter] {len(pool)} of {len(failures)} failures match the filters\n")
    if not pool:
        print("no failures match the filters -- nothing to do")
        return 1

    if args.strategy == "pick":
        selection = None
        if args.pick_file:
            with open(args.pick_file) as handle:
                selection = handle.read().strip()
        elif not sys.stdin.isatty():
            print("--strategy pick needs a terminal to prompt on; pass --pick-file "
                  "with a selection like '1-3,17' instead.")
            return 1
        try:
            chosen = pick_cases(pool, args.pick_limit, selection, args.top)
        except ValueError as error:
            print(f"\nbad selection in {args.pick_file}: {error}")
            return 1
        if not chosen:
            return 1
    else:
        chosen = select(pool, args.top, args.strategy)
    ensure_dir(args.out_dir)

    lines = ["# Failure cases\n",
             f"model: `{model_name}` | split: `{args.split}` | "
             f"failure = IoU < {args.threshold} | parse: `{args.parse_folder}`\n",
             f"{len(failures)} of {total} annotations "
             f"({100.0 * len(failures) / max(total, 1):.2f}%) are failures.\n",
             "\n## Distribution of causes\n", distribution,
             "\n## Selected cases\n"]

    print(f"\n[selected] {len(chosen)} case(s), strategy={args.strategy}\n")
    for rank, case in enumerate(chosen, 1):
        case["rank"] = rank
        case["visualize_command"] = (
            f"python scripts/visualize.py --folder {os.path.basename(os.path.dirname(os.path.abspath(predictions_path)))} "
            f"--scene_id {case['scene_id']} --use_color --use_normal")

        print(f"[{rank}] {case['cause']:<22} IoU={case['iou']:.3f}  "
              f"{case['scene_id']} obj={case['object_id']} ({case['object_name']})")
        print(f"     {case['description'][:110]}")
        print(f"     parsed target={case['parsed_target']!r}  "
              f"same-class instances={case['signals']['same_class_instances']}  "
              f"tokens={case['signals']['num_tokens']}  "
              f"spatial={case['signals']['num_spatial']}")

        lines.append(f"\n### {rank}. {case['cause']} — IoU {case['iou']:.3f}\n")
        lines.append(f"- **scene / object**: `{case['scene_id']}` / "
                     f"`{case['object_id']}` ({case['object_name']}), ann `{case['ann_id']}`")
        lines.append(f"- **description**: {case['description']}")
        lines.append(f"- **parsed**: target=`{case['parsed_target']}`, "
                     f"adjectives=`{case['parsed_adjectives']}`, "
                     f"neighbors=`{case['parsed_neighbors']}`")
        lines.append(f"- **why it failed**: {CAUSE_TEXT[case['cause']]}")
        lines.append(f"- **signals**: same-class instances "
                     f"{case['signals']['same_class_instances']}, GT volume "
                     f"{case['signals']['gt_volume']} m³, {case['signals']['num_tokens']} "
                     f"tokens, {case['signals']['num_spatial']} spatial cues")
        lines.append(f"- **render**: `{case['visualize_command']}`")

        if args.write_ply:
            directory = dump_case(case, args.out_dir, args)
            lines.append(f"- **boxes**: `{directory}/pred.ply` (red), "
                         f"`{directory}/gt.ply` (green)")

    report_path = os.path.join(args.out_dir, "failure_cases.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    json_path = save_json({
        "model": model_name, "predictions": predictions_path, "split": args.split,
        "threshold": args.threshold, "criterion": args.criterion,
        "num_annotations": total, "num_failures": len(failures),
        "cause_distribution": dict(tally),
        "cause_explanations": CAUSE_TEXT,
        "selected": [{k: v for k, v in case.items() if not k.startswith("_")}
                     for case in chosen],
    }, os.path.join(args.out_dir, "failure_cases.json"))

    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")
    if args.write_ply:
        print(f"wrote {len(chosen)} case directories under {args.out_dir}/")
    return 0


# ---- failure-figures ----

_DOC_FAILURE_FIGURES = """
Render the qualitative failure figures. CPU, about a minute for six cases.

    python experiments/analysis.py failure-figures --cases 3 --dpi 300

failure_cases.py picks and diagnoses the cases and writes pred.ply / gt.ply per case; this
turns them into finished panels. Each case becomes one PNG with three panels: a bird's-eye
view of the room with every other instance of the target's class outlined, a perspective
3-D view from an upper room corner as a point-cloud viewer would show it, and the
description/parse/diagnosis. Green is ground truth, red is the prediction.

Points come from the original scans under data/scannet/scans (~160k per scene, aligned
into the box frame with ScanNet's axisAlignment matrix). --source npy draws the detector's
fixed 50k cloud instead, which is what the model actually sees.

The boxes in predictions.p live in the same axis-aligned frame as <scene>_aligned_vert.npy;
that is re-checked at run time per case, and a case failing the check is still drawn but
carries a visible warning.
"""

REPO = REPO_ROOT
SCANNET_DATA = os.path.join(REPO, "data", "scannet", "scannet_data")
#: The original scans. Denser than SCANNET_DATA's fixed 50k cloud, and what the figures
#: draw by default; absolute, because this runs as a CLI from any working directory.
SCANNET_SCANS = os.path.join(REPO, "data", "scannet", "scans")

GT_COLOR = "#12A150"
PRED_COLOR = "#D62828"
DISTRACTOR_COLOR = "#8A8FA3"



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


# The 12 edges of a box whose 8 corners were built in the
# ``for x in (lo, hi) for y in (lo, hi) for z in (lo, hi)`` order used throughout here.
BOX_EDGE_PAIRS = ((0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
                  (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7))


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    """World-to-camera rotation. ScanNet's aligned frame is z-up, hence the default."""
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(up, dtype=float))
    right /= np.linalg.norm(right)
    return np.stack([right, np.cross(right, forward), -forward])


def project(points, eye, rotation, fov=55.0, width=1000.0, height=750.0):
    """Perspective-project world points to pixels. Returns (u, v, depth); depth<=0 is behind."""
    camera = (np.asarray(points, dtype=float) - eye) @ rotation.T
    depth = -camera[:, 2]
    focal = (height / 2.0) / np.tan(np.deg2rad(fov) / 2.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = width / 2.0 + focal * camera[:, 0] / depth
        v = height / 2.0 - focal * camera[:, 1] / depth
    return u, v, depth


def draw_box_projected(axis, corners, eye, rotation, color, linewidth=2.2):
    """Draw a box as a projected wireframe, dropping edges that cross behind the camera."""
    corners = np.asarray(corners, dtype=float)
    low, high = corners.min(axis=0), corners.max(axis=0)
    vertices = np.array([[x, y, z] for x in (low[0], high[0])
                         for y in (low[1], high[1])
                         for z in (low[2], high[2])])
    u, v, depth = project(vertices, eye, rotation)
    for i, j in BOX_EDGE_PAIRS:
        if depth[i] > 0 and depth[j] > 0:
            axis.plot([u[i], u[j]], [v[i], v[j]], color=color, linewidth=linewidth)


def load_scene(scene_id):
    path = os.path.join(SCANNET_DATA, f"{scene_id}_aligned_vert.npy")
    if not os.path.isfile(path):
        return None, None
    vertices = np.load(path)
    xyz = vertices[:, :3].astype(float)
    rgb = np.clip(vertices[:, 3:6] / 255.0, 0, 1) if vertices.shape[1] >= 6 else None
    return xyz, rgb


#: The vertex layout every ScanNet ``_vh_clean_2.ply`` uses.
PLY_VERTEX_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                             ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])
PLY_PROPERTIES = [(b"float", b"x"), (b"float", b"y"), (b"float", b"z"),
                  (b"uchar", b"red"), (b"uchar", b"green"), (b"uchar", b"blue"),
                  (b"uchar", b"alpha")]


def axis_alignment(scene_id, scans_root=None):
    """The 4x4 matrix that puts a raw scan into the frame the boxes live in."""
    root = scans_root or SCANNET_SCANS
    path = os.path.join(root, scene_id, f"{scene_id}.txt")
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        for line in handle:
            if line.startswith("axisAlignment"):
                values = [float(v) for v in line.split("=", 1)[1].split()]
                if len(values) == 16:
                    return np.asarray(values, dtype=float).reshape(4, 4)
    return None


def load_scan_ply(scene_id, scans_root=None):
    """The original scan, aligned into the box frame. (None, None) if unusable.

    Roughly 160k points against the 50k of ``load_scene``. The raw .ply sits in the
    scanner's own frame -- drawing it unaligned would put every box in the wrong place --
    so axisAlignment is required, not optional.
    """
    root = scans_root or SCANNET_SCANS
    path = os.path.join(root, scene_id, f"{scene_id}_vh_clean_2.ply")
    if not os.path.isfile(path):
        return None, None
    matrix = axis_alignment(scene_id, root)
    if matrix is None:
        return None, None

    with open(path, "rb") as handle:
        header = b""
        while b"end_header" not in header:
            line = handle.readline()
            if not line:                                  # truncated file
                return None, None
            header += line
        # Read bytes only under the layout PLY_VERTEX_DTYPE describes; any other layout
        # would be silently misread as geometry rather than raising.
        count = re.search(rb"element vertex (\d+)", header)
        if (b"binary_little_endian" not in header or not count
                or re.findall(rb"property (\w+) (\w+)", header)[:7] != PLY_PROPERTIES):
            return None, None
        vertices = np.fromfile(handle, dtype=PLY_VERTEX_DTYPE, count=int(count.group(1)))

    xyz = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(float)
    xyz = (np.concatenate([xyz, np.ones((len(xyz), 1))], axis=1) @ matrix.T)[:, :3]
    rgb = np.stack([vertices["r"], vertices["g"], vertices["b"]],
                   axis=1).astype(float) / 255.0
    return xyz, rgb


def load_points(scene_id, source="ply", scans_root=None):
    """(xyz, rgb, source_used). Falls back to the .npy cloud when the scan is unusable."""
    if source == "ply":
        xyz, rgb = load_scan_ply(scene_id, scans_root)
        if xyz is not None:
            return xyz, rgb, "ply"
        print(f"  ! {scene_id}: no usable scan under "
              f"{scans_root or SCANNET_SCANS} -- falling back to .npy")
    xyz, rgb = load_scene(scene_id)
    return xyz, rgb, ("npy" if xyz is not None else None)


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

    # Anchor on the scene id rather than counting underscores: a cause is one word
    # ("unattributed") or two ("distractor_confusion"), and splitting blind put the
    # boundary in the wrong place for the one-word ones. An optional NN_ rank prefix is
    # what dump_case adds; failure-scenes writes the same name without it.
    cases = []
    for name in sorted(os.listdir(cases_dir)):
        if not os.path.isdir(os.path.join(cases_dir, name)):
            continue
        match = re.match(r"^(?:\d+_)?(?P<cause>.+?)_"
                         r"(?P<scene>scene\d{4}_\d{2})_obj(?P<obj>\d+)_ann(?P<ann>\d+)$",
                         name)
        if not match:
            continue
        cases.append({"scene_id": match.group("scene"),
                      "object_id": match.group("obj"), "ann_id": match.group("ann"),
                      "cause": match.group("cause"), "description": "", "iou": None})
    return {}, cases[:limit]



def render_case(case, predictions, out_dir, dpi, max_points,
                source="ply", scans_root=None):
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

    xyz, rgb, source_used = load_points(scene_id, source, scans_root)
    if xyz is None:
        print(f"  ! {scene_id}: no point cloud found (tried {source}) -- skipped")
        return None

    offset = verify_frames(scene_id, object_id, gt)
    misaligned = offset is not None and offset > 0.05

    figure = plt.figure(figsize=(14.5, 5.2))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 0.85], wspace=0.13)

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

    # ---- panel 2: the 3-D view, as a point-cloud viewer would show it -------------------
    # A perspective camera at an upper room corner. Flat orthographic projections collapse
    # the depth cues that make a room legible; this keeps them, and painting the points
    # back to front gives real occlusion.
    scene_low, scene_high = xyz.min(axis=0), xyz.max(axis=0)
    diagonal = float(np.linalg.norm(scene_high - scene_low))
    eye = np.array([scene_high[0] + 0.30 * diagonal,
                    scene_low[1] - 0.30 * diagonal,
                    scene_high[2] + 0.30 * diagonal])
    rotation = look_at(eye, (scene_low + scene_high) / 2.0)

    step = max(1, len(xyz) // max(max_points, 1))
    shown, shown_rgb = xyz[::step], (rgb[::step] if rgb is not None else None)
    if shown_rgb is not None:
        # Gamma < 1 lifts the midtones; ScanNet scans are captured dark and the raw
        # values leave most surfaces indistinguishable from their own shadow.
        shown_rgb = np.clip(shown_rgb, 0, 1) ** 0.6

    ax2 = figure.add_subplot(grid[0, 1])
    u, v, depth = project(shown, eye, rotation)
    visible = depth > 0
    order = np.argsort(-depth[visible])               # far to near: painter's algorithm
    u, v = u[visible][order], v[visible][order]
    ax2.scatter(u, v, s=2.2,
                c=shown_rgb[visible][order] if shown_rgb is not None else "#B8BEC9",
                linewidths=0, rasterized=True)
    draw_box_projected(ax2, gt, eye, rotation, GT_COLOR)
    draw_box_projected(ax2, pred, eye, rotation, PRED_COLOR)

    if len(u):                                        # frame the room, not the empty canvas
        pad = 0.04 * max(u.max() - u.min(), 1.0)
        ax2.set_xlim(u.min() - pad, u.max() + pad)
        ax2.set_ylim(v.max() + pad, v.min() - pad)    # screen coordinates run downward
    ax2.set_title(f"3-D view from the room corner  ({len(u):,} points)", fontsize=9)
    ax2.set_aspect("equal")
    ax2.axis("off")

    # ---- panel 3: the words ------------------------------------------------------------
    ax3 = figure.add_subplot(grid[0, 2])
    ax3.axis("off")

    # Only the count this figure draws. failure_cases.py counts via the NYU40 mapping,
    # which need not agree with the boxes carrying this scene's semantic label.
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
            "misaligned": bool(misaligned),
            "source": source_used, "num_points": int(len(xyz))}


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


def _add_failure_figures_args(parser):
    parser.add_argument("--predictions",
                        default="outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p")
    parser.add_argument("--cases-dir", dest="cases_dir",
                        default="outputs/analysis/failure_cases",
                        help="output of experiments/analysis.py failure-cases")
    parser.add_argument("--out-dir", dest="out_dir",
                        default="experiments/figures",
                        help="default keeps the figures beside this script; pass "
                             "outputs/diagnostics/failure_figures to write them "
                             "under outputs/ instead")
    parser.add_argument("--cases", type=int, default=12, help="maximum cases to render")
    parser.add_argument("--dpi", type=int, default=200,
                        help="use 300 for the camera-ready figure")
    parser.add_argument("--source", default="ply", choices=["ply", "npy"],
                        help="ply: the original scans, ~160k points (default). "
                             "npy: the detector's fixed 50k cloud.")
    parser.add_argument("--scans-root", dest="scans_root", default=SCANNET_SCANS,
                        help="where the original scans live")
    parser.add_argument("--max-points", dest="max_points", type=int, default=200000,
                        help="cap on points drawn per panel; the default is above the "
                             "50k a ScanNet scene carries, so every point is drawn")
    return parser


def run_failure_figures(args):

    cases_dir = os.path.join(REPO, args.cases_dir)
    if not os.path.isdir(cases_dir):
        print(f"case folder not found: {args.cases_dir}")
        print("Run experiments/analysis.py failure-cases first -- it selects and diagnoses the "
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
        result = render_case(case, predictions, out_dir, args.dpi, args.max_points,
                             args.source, args.scans_root)
        if result:
            result["case"] = case
            rendered.append(result)
            flag = "  FRAME MISMATCH" if result["misaligned"] else ""
            print(f"        IoU {result['iou']:.3f}  centre error "
                  f"{result['drift']:.2f} m  distractors {result['distractors']}  "
                  f"{result['num_points']:,} points ({result['source']}){flag}")

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
                      "frame_offset_m": item["frame_offset"],
                      "source": item["source"], "num_points": item["num_points"]}
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


# ---- failure-scenes ----

_DOC_FAILURE_SCENES = """
Export each failure case as one 3-D scene to open in a point-cloud viewer. CPU, seconds.

    python experiments/analysis.py failure-scenes --cases 6

The PNG's 3-D panel is a matplotlib scatter with a painter's-algorithm depth sort. That is
as far as matplotlib goes -- no z-buffer, no splatting, no orbiting -- so for cases where
the panel is not legible enough, this writes the geometry instead and you take the
screenshot from a real viewer (MeshLab, CloudCompare).

Per case it writes a folder holding:

    scene.ply   the aligned scan with its real colour, plus the ground-truth box (green),
                the prediction (red) and every same-class distractor (grey) drawn as
                cages of points. One file: open it, orbit, screenshot.

                It is a pure point cloud -- no face element -- so every viewer opens it
                coloured with nothing to switch on. Writing the cages as triangles
                instead makes the file a mesh, and a mesh hides the scan points: the
                room then shows up grey, or not at all.
    case.txt    the description, the parse, the diagnosis and the frame check
    <cause>_<scene>_obj<id>_ann<id>.png   the usual three-panel figure, unless --no-figure

Points come from data/scannet/scans by default (~160k per scene); --source npy draws the
detector's fixed 50k cloud instead.
"""


def _hex_to_rgb(value):
    """'#12A150' -> (18, 161, 80). The figure's colours, reused by the PLY export."""
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def _add_failure_scenes_args(parser):
    parser.add_argument("--predictions",
                        default="outputs/2024-12-18_20-40-38_3DVG-FIXED/predictions.p")
    parser.add_argument("--cases-dir", dest="cases_dir",
                        default="outputs/analysis/failure_cases",
                        help="output of experiments/analysis.py failure-cases")
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/failure_scenes")
    parser.add_argument("--cases", type=int, default=12, help="maximum cases to export")
    parser.add_argument("--source", default="ply", choices=["ply", "npy"],
                        help="ply: the original scans, ~160k points (default). "
                             "npy: the detector's fixed 50k cloud.")
    parser.add_argument("--scans-root", dest="scans_root", default=SCANNET_SCANS)
    parser.add_argument("--box-spacing", dest="box_spacing", type=float, default=0.005,
                        help="gap in metres between the points making up a cage edge; "
                             "smaller draws a more solid line and costs more points")
    parser.add_argument("--no-figure", dest="write_figure", action="store_false",
                        help="skip the PNG and write only the 3-D scene")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--max-points", dest="max_points", type=int, default=200000,
                        help="cap on points drawn in the PNG; the scene.ply is never "
                             "subsampled")
    return parser


def export_scene(case, predictions, out_dir, args):
    """One case -> a folder with scene.ply, case.txt and (optionally) the PNG."""
    scene_id = case["scene_id"]
    object_id, ann_id = str(case["object_id"]), str(case["ann_id"])

    entry = predictions.get(scene_id, {}).get(object_id, {}).get(ann_id)
    if entry is None:
        print(f"  ! {scene_id}/{object_id}/{ann_id} absent from predictions.p -- skipped")
        return None

    gt = np.asarray(entry["gt_bbox"], dtype=float)
    pred = np.asarray(entry["pred_bbox"], dtype=float)
    iou = float(entry.get("iou", case.get("iou") or 0.0))

    xyz, rgb, source_used = load_points(scene_id, args.source, args.scans_root)
    if xyz is None:
        print(f"  ! {scene_id}: no point cloud found (tried {args.source}) -- skipped")
        return None

    offset = verify_frames(scene_id, object_id, gt)
    misaligned = offset is not None and offset > 0.05
    distractors = same_class_boxes(scene_id, object_id)

    folder = ensure_dir(os.path.join(
        out_dir, f"{case.get('cause', 'case')}_{scene_id}_obj{object_id}_ann{ann_id}"))

    # Distractors first, so the two boxes that matter draw over them in a viewer that
    # respects file order. They are also sampled more sparsely, which keeps a scene with
    # a dozen same-class instances from being a wall of grey.
    boxes = [(box, _hex_to_rgb(DISTRACTOR_COLOR), args.box_spacing * 2.0)
             for box in distractors]
    boxes.append((gt, _hex_to_rgb(GT_COLOR), args.box_spacing))
    boxes.append((pred, _hex_to_rgb(PRED_COLOR), args.box_spacing))

    scene_path = write_scene_ply(os.path.join(folder, "scene.ply"), xyz, rgb, boxes)

    signals = case.get("signals", {}) or {}
    with open(os.path.join(folder, "case.txt"), "w") as f:
        f.write(f"scene       : {scene_id}\n")
        f.write(f"object      : {object_id} ({case.get('object_name', '?')})\n")
        f.write(f"annotation  : {ann_id}\n")
        f.write(f"IoU         : {iou:.4f}\n")
        cause = case.get("cause", "?")
        f.write(f"cause       : {cause}\n")
        if cause in CAUSE_TEXT:
            f.write(f"              {CAUSE_TEXT[cause]}\n")
        f.write(f"\ndescription : {case.get('description', '')}\n\n")
        f.write(f"parsed target     : {case.get('parsed_target', '-')!r}"
                f"  (ground truth {case.get('object_name', '?')!r})\n")
        f.write(f"parsed adjectives : {case.get('parsed_adjectives', '-')!r}\n")
        f.write(f"parsed neighbors  : {case.get('parsed_neighbors', '-')!r}\n\n")
        if signals:
            f.write("signals\n")
            for key, value in signals.items():
                f.write(f"  {key:<22}: {value}\n")
            f.write("\n")
        f.write(f"points      : {len(xyz):,} from the {source_used} source\n")
        f.write(f"distractors : {len(distractors)} same-label box(es) drawn in grey\n")
        f.write(f"frame check : gt centre is {offset:.4f} m from the scene's "
                f"aligned_bbox entry\n" if offset is not None else
                "frame check : no aligned_bbox entry for this object\n")
        if misaligned:
            f.write("              WARNING: over 5 cm -- the boxes may sit in the "
                    "wrong frame\n")
        f.write("\nscene.ply: green = ground truth, red = prediction, "
                "grey = other objects of the same class.\n")
        f.write("Open it in MeshLab or CloudCompare and screenshot from there.\n")

    figure_path = None
    if args.write_figure:
        result = render_case(case, predictions, folder, args.dpi, args.max_points,
                             args.source, args.scans_root)
        figure_path = result["path"] if result else None

    return {"folder": folder, "scene": scene_path, "figure": figure_path,
            "iou": iou, "distractors": len(distractors), "frame_offset": offset,
            "misaligned": bool(misaligned), "source": source_used,
            "num_points": int(len(xyz)), "bytes": os.path.getsize(scene_path)}


def run_failure_scenes(args):

    cases_dir = os.path.join(REPO, args.cases_dir)
    if not os.path.isdir(cases_dir):
        print(f"case folder not found: {args.cases_dir}")
        print("Run experiments/analysis.py failure-cases first -- it selects and "
              "diagnoses the cases this exports.")
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

    out_dir = ensure_dir(os.path.join(REPO, args.out_dir))
    print(f"exporting {len(cases)} case(s) -> {os.path.relpath(out_dir, REPO)}\n")

    exported = []
    for index, case in enumerate(cases, 1):
        print(f"  [{index}/{len(cases)}] {case['scene_id']}/{case['object_id']}/"
              f"{case['ann_id']} [{case.get('cause', '?')}]")
        result = export_scene(case, predictions, out_dir, args)
        if result:
            result["case"] = case
            exported.append(result)
            flag = "  FRAME MISMATCH" if result["misaligned"] else ""
            print(f"        {result['num_points']:,} points ({result['source']})  "
                  f"{result['bytes'] / 1e6:.1f} MB  IoU {result['iou']:.3f}{flag}")
            print(f"        {result['scene']}")

    manifest = {
        "predictions": args.predictions,
        "cases_dir": args.cases_dir,
        "model": payload.get("model"),
        "source": args.source,
        "exported": [{"folder": os.path.relpath(item["folder"], REPO),
                      "scene_id": item["case"]["scene_id"],
                      "object_id": item["case"]["object_id"],
                      "ann_id": item["case"]["ann_id"],
                      "cause": item["case"].get("cause"),
                      "iou": item["iou"],
                      "same_label_boxes_drawn": item["distractors"],
                      "frame_offset_m": item["frame_offset"],
                      "source": item["source"], "num_points": item["num_points"],
                      "scene_ply_bytes": item["bytes"]}
                     for item in exported],
    }
    manifest_path = os.path.join(out_dir, "scenes.json")
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"\nwrote {len(exported)} scene(s) and {os.path.relpath(manifest_path, REPO)}")
    print("Open a scene.ply in MeshLab or CloudCompare; green is ground truth, red is "
          "the prediction, grey are same-class distractors.")

    bad = [item for item in exported if item["misaligned"]]
    if bad:
        print(f"\n{len(bad)} scene(s) carry a coordinate-frame warning -- check the "
              f"aligned_bbox entry before publishing a screenshot.")
        return 1
    return 0


# ---- annotation-sheet ----

_DOC_ANNOTATION_SHEET = """
Export a stratified sample of parses for manual labelling. CPU, seconds.

    python experiments/analysis.py annotation-sheet --num 200         --parse gpt4o-mini=final_parsing_tokenized         --parse spacy=spacy_parsing_tokenized

Target extraction is scored automatically against object_name; attributes and adjacent
objects have no ground truth in ScanRefer and need hand labelling.

The sample is paired across parsers, so the comparison is within-item, and stratified by
target correctness -- a uniform draw would be ~90% correct parses. --wrong-fraction forces
in a share of detected errors, and sampling_manifest.json carries the strata so
error_taxonomy.py can re-weight back to population rates. See details.md §7.

Writes outputs/analysis/annotation/{annotation_sheet_<parser>.csv,.txt, instructions.md,
sampling_manifest.json}. Fill the five blank columns, then tally with error_taxonomy.py.
"""

#: The five columns the annotator fills in. Kept blank in the exported sheet.
BLANK_COLUMNS = ("target_ok", "adjectives_ok", "neighbors_ok", "error_type", "notes")

FIXED_COLUMNS = ("sample_id", "parser", "scene_id", "object_id", "ann_id",
                 "object_name", "description",
                 "parsed_target", "parsed_adjectives", "parsed_neighbors",
                 "auto_target_match")

#: Error taxonomy. Also duplicated in experiments/analysis.py error-taxonomy, which validates it.
TAXONOMY = {
    "ok": "all three fields are acceptable",
    "wrong_target": "the target names the wrong object",
    "missed_attribute": "an attribute stated in the description is absent from the parse",
    "hallucinated_attribute": "an attribute appears that the description does not state",
    "missed_neighbor": "an adjacent object stated in the description is absent",
    "hallucinated_neighbor": "an adjacent object appears that the description does not state",
    "malformed_output": "the parse is structurally broken or empty where content exists",
}

INSTRUCTIONS = """# Manual parse annotation — instructions

Fill in the five blank columns of each `annotation_sheet_<parser>.csv`.

## Columns to fill

| column | values | meaning |
|---|---|---|
| `target_ok` | `1` / `0` | Does `parsed_target` name the object the description refers to? Judge against the description, not only against `object_name` — `object_name` is the annotated class and is sometimes coarser than the sentence. |
| `adjectives_ok` | `1` / `0` | Does `parsed_adjectives` capture the attributes the description gives for the target, without inventing any? `not mentioned` is **correct** when the description gives no attributes. |
| `neighbors_ok` | `1` / `0` | Does `parsed_neighbors` capture the spatial relations to other objects, without inventing any? `not mentioned` is **correct** when there are none. |
| `error_type` | see below | One or more codes, separated by `;`. Use `ok` when nothing is wrong. |
| `notes` | free text | Anything worth quoting in the paper. Optional. |

## Error codes

{taxonomy}

## Rules that keep the numbers honest

1. **Annotate every parser's sheet for the same `sample_id` before moving on.** The design
   is paired; breaking the pairing throws away most of its statistical power.
2. **Do not consult `auto_target_match`** when deciding `target_ok`. It is the automatic
   verdict, included only so disagreements can be found afterwards. Judging while looking
   at it makes the two measurements dependent and the comparison meaningless.
3. `not mentioned` is a legitimate, correct parse whenever the description genuinely does
   not supply that field. It is only an error when content exists and was dropped.
4. If two annotators are available, have both label the same sheet and save them
   separately. `experiments/analysis.py error-taxonomy` computes Cohen's kappa when given two files,
   and an agreement number makes the manual layer far harder to dismiss.
5. The sample is **stratified, not uniform** — failures are deliberately oversampled. Do
   not quote raw percentages from it as population rates; `error_taxonomy.py` applies the
   weights in `sampling_manifest.json` and reports both.
"""


def build_pool(records, cache, criterion):
    """Split the split into (target-correct, target-wrong) using the automatic check."""
    correct, wrong = [], []
    for record in records:
        parsed = parse_for(cache, record["scene_id"], record["object_id"], record["ann_id"])
        if not parsed:
            continue
        predicted = " ".join(parsed.get("target") or [])
        kinds = match_kinds(predicted, record["object_name"])
        (correct if kinds[criterion] else wrong).append(record)
    return correct, wrong


def _add_annotation_sheet_args(parser):
    parser.add_argument("--parse", action="append", required=True, metavar="NAME=FOLDER")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--num", type=int, default=200)
    parser.add_argument("--wrong-fraction", dest="wrong_fraction", type=float, default=0.4,
                        help="Share of the sample drawn from automatically-detected "
                             "target errors. 0 gives a uniform random sample.")
    parser.add_argument("--criterion", default="fuzzy",
                        choices=["exact", "substring", "fuzzy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/analysis/annotation")
    return parser


def run_annotation_sheet(args):

    parsers = []
    for spec in args.parse:
        name, folder = (spec.split("=", 1) if "=" in spec else (spec, spec))
        parsers.append((name.strip(), folder.strip()))

    records = load_scanrefer(args.split, args.data_root)
    caches = {}
    for name, folder in parsers:
        caches[name] = load_parse_cache(folder, args.split, args.parsing_root)

    # Strata are defined by the FIRST parser, so every parser gets the same sample_ids.
    anchor_name, anchor_folder = parsers[0]
    correct, wrong = build_pool(records, caches[anchor_name], args.criterion)
    print(f"[strata] anchored on {anchor_name} ({anchor_folder}): "
          f"{len(correct)} target-correct, {len(wrong)} target-wrong")

    rng = random.Random(args.seed)
    num_wrong = min(len(wrong), int(round(args.num * args.wrong_fraction)))
    num_correct = min(len(correct), args.num - num_wrong)
    sample = (rng.sample(wrong, num_wrong) + rng.sample(correct, num_correct))
    rng.shuffle(sample)
    print(f"[sample] {len(sample)} annotations: {num_wrong} from the target-wrong "
          f"stratum, {num_correct} from the target-correct stratum")

    ensure_dir(args.out_dir)
    written = []
    for name, folder in parsers:
        cache = caches[name]
        rows = []
        for index, record in enumerate(sample, 1):
            parsed = parse_for(cache, record["scene_id"], record["object_id"],
                               record["ann_id"]) or {}
            predicted = " ".join(parsed.get("target") or [])
            kinds = match_kinds(predicted, record["object_name"]) if parsed else None
            rows.append({
                "sample_id": f"S{index:03d}",
                "parser": name,
                "scene_id": record["scene_id"],
                "object_id": record["object_id"],
                "ann_id": record["ann_id"],
                "object_name": record["object_name"],
                "description": (record.get("description") or "").strip(),
                "parsed_target": predicted or "(absent from cache)",
                "parsed_adjectives": " ".join(parsed.get("adjectives") or []) or "(absent)",
                "parsed_neighbors": " ".join(parsed.get("neighbors") or []) or "(absent)",
                "auto_target_match": (args.criterion if kinds and kinds[args.criterion]
                                      else "MISS"),
                **{column: "" for column in BLANK_COLUMNS},
            })

        csv_path = os.path.join(args.out_dir, f"annotation_sheet_{name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(FIXED_COLUMNS) + list(BLANK_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)

        txt_path = os.path.join(args.out_dir, f"annotation_sheet_{name}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Manual annotation sheet - parser: {name} ({folder})\n")
            f.write(f"split={args.split}  n={len(rows)}  seed={args.seed}\n")
            f.write("Fill target_ok / adjectives_ok / neighbors_ok / error_type in the CSV.\n")
            f.write("=" * 92 + "\n\n")
            for row in rows:
                f.write(f"[{row['sample_id']}] {row['scene_id']} "
                        f"obj={row['object_id']} ann={row['ann_id']}\n")
                f.write(f"  description : {row['description']}\n")
                f.write(f"  object_name : {row['object_name']}\n")
                f.write(f"  target      : {row['parsed_target']}\n")
                f.write(f"  adjectives  : {row['parsed_adjectives']}\n")
                f.write(f"  neighbors   : {row['parsed_neighbors']}\n")
                f.write("  target_ok=__  adjectives_ok=__  neighbors_ok=__  error_type=______\n\n")

        written += [csv_path, txt_path]
        print(f"[{name}] -> {csv_path}")
        print(f"[{name}] -> {txt_path}")

    instructions_path = os.path.join(args.out_dir, "instructions.md")
    taxonomy_table = "\n".join(f"- `{code}` — {text}" for code, text in TAXONOMY.items())
    with open(instructions_path, "w") as f:
        f.write(INSTRUCTIONS.format(taxonomy=taxonomy_table))

    manifest_path = save_json({
        "split": args.split, "seed": args.seed, "criterion": args.criterion,
        "anchor_parser": anchor_name,
        "requested": args.num, "sampled": len(sample),
        "strata": {
            "target_wrong": {
                "population": len(wrong), "sampled": num_wrong,
                "weight": (len(wrong) / num_wrong) if num_wrong else 0.0},
            "target_correct": {
                "population": len(correct), "sampled": num_correct,
                "weight": (len(correct) / num_correct) if num_correct else 0.0},
        },
        "note": "Failures are oversampled. Re-weight with the per-stratum weights before "
                "quoting any rate as a population rate; error_taxonomy.py does this.",
        "parsers": {name: folder for name, folder in parsers},
        "sample": [{"sample_id": f"S{i:03d}", "scene_id": r["scene_id"],
                    "object_id": r["object_id"], "ann_id": r["ann_id"]}
                   for i, r in enumerate(sample, 1)],
    }, os.path.join(args.out_dir, "sampling_manifest.json"))

    print(f"\nwrote {instructions_path}")
    print(f"wrote {manifest_path}")
    print("\nAnnotate the CSV files, then run:")
    print("    python experiments/analysis.py error-taxonomy " +
          " ".join(f"--sheet {name}={os.path.join(args.out_dir, f'annotation_sheet_{name}.csv')}"
                   for name, _ in parsers))
    return 0


# ---- annotation-sheet-15 ----

_DOC_ANNOTATION_SHEET_15 = """
Export a small sheet stratified by description length, for hand labelling. CPU, seconds.

    python experiments/analysis.py annotation-sheet-15 --parse gpt4o-mini=final_parsing_tokenized

The 200-row `annotation-sheet` oversamples parser failures so its rates can be re-weighted
to population rates. This one does not: it draws uniformly inside three length bands, so
what you read is what you get, and the sheet is small enough to actually finish.

Difficulty is description length in words. The band edges are the tertiles of the 9,508
val descriptions -- <=14, 15-19, >=20 -- which split the population 37.5/30.7/31.8%.

With five rows a band this is a qualitative error analysis, not a rate estimate. Report the
failure modes it surfaces, not percentages; the 200-row sheet is the instrument for rates.

Writes outputs/analysis/annotation15/{annotation15_<parser>.csv,.txt, instructions.md,
sampling_manifest15.json}. Fill the blank columns, then tally with error-taxonomy.
"""

#: Length tertiles of the val split, in words. See _DOC_ANNOTATION_SHEET_15.
DIFFICULTY_BANDS = (("easy", 0, 14), ("medium", 15, 19), ("hard", 20, 10 ** 6))


def _add_annotation_sheet_15_args(parser):
    parser.add_argument("--parse", action="append", required=True, metavar="NAME=FOLDER")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--per-band", dest="per_band", type=int, default=5,
                        help="rows drawn from each of the three length bands")
    parser.add_argument("--criterion", default="fuzzy",
                        choices=["exact", "substring", "fuzzy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/annotation15")
    return parser


def run_annotation_sheet_15(args):

    parsers = []
    for spec in args.parse:
        name, folder = (spec.split("=", 1) if "=" in spec else (spec, spec))
        parsers.append((name.strip(), folder.strip()))

    records = load_scanrefer(args.split, args.data_root)
    caches = {name: load_parse_cache(folder, args.split, args.parsing_root)
              for name, folder in parsers}

    # Bands are defined by the FIRST parser's coverage, so every parser gets the same rows.
    anchor_name = parsers[0][0]
    bands = {label: [] for label, _, _ in DIFFICULTY_BANDS}
    for record in records:
        if not parse_for(caches[anchor_name], record["scene_id"],
                         record["object_id"], record["ann_id"]):
            continue
        words = len((record.get("description") or "").split())
        for label, low, high in DIFFICULTY_BANDS:
            if low <= words <= high:
                bands[label].append((words, record))
                break

    rng = random.Random(args.seed)
    sample = []
    for label, low, high in DIFFICULTY_BANDS:
        pool = bands[label]
        take = min(args.per_band, len(pool))
        if take < args.per_band:
            print(f"[warn] band {label!r} holds only {len(pool)} parsed records")
        # Sort before sampling: dict/JSON order is stable here, but sorting makes the
        # draw reproducible regardless of how the records arrived.
        pool = sorted(pool, key=lambda item: (item[1]["scene_id"], item[1]["object_id"],
                                              item[1]["ann_id"]))
        for words, record in rng.sample(pool, take):
            sample.append((label, words, record))
        print(f"[band] {label:6s} {low}-{high if high < 10 ** 6 else '+'} words: "
              f"{len(bands[label])} available, {take} drawn")

    ensure_dir(args.out_dir)
    columns = (list(FIXED_COLUMNS) + ["difficulty", "num_words"] + list(BLANK_COLUMNS))
    written = []
    for name, folder in parsers:
        cache = caches[name]
        rows = []
        for index, (label, words, record) in enumerate(sample, 1):
            parsed = parse_for(cache, record["scene_id"], record["object_id"],
                               record["ann_id"]) or {}
            predicted = " ".join(parsed.get("target") or [])
            kinds = match_kinds(predicted, record["object_name"]) if parsed else None
            rows.append({
                "sample_id": f"D{index:02d}",
                "parser": name,
                "scene_id": record["scene_id"],
                "object_id": record["object_id"],
                "ann_id": record["ann_id"],
                "object_name": record["object_name"],
                "description": (record.get("description") or "").strip(),
                "parsed_target": predicted or "(absent from cache)",
                "parsed_adjectives": " ".join(parsed.get("adjectives") or []) or "(absent)",
                "parsed_neighbors": " ".join(parsed.get("neighbors") or []) or "(absent)",
                "auto_target_match": (args.criterion if kinds and kinds[args.criterion]
                                      else "MISS"),
                "difficulty": label,
                "num_words": words,
                **{column: "" for column in BLANK_COLUMNS},
            })

        csv_path = os.path.join(args.out_dir, f"annotation15_{name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

        txt_path = os.path.join(args.out_dir, f"annotation15_{name}.txt")
        with open(txt_path, "w") as f:
            f.write(f"Manual annotation sheet ({len(rows)} rows) - parser: {name} ({folder})\n")
            f.write(f"split={args.split}  seed={args.seed}  stratified by description length\n")
            f.write("Fill target_ok / adjectives_ok / neighbors_ok / error_type in the CSV.\n")
            f.write("=" * 92 + "\n\n")
            for row in rows:
                f.write(f"[{row['sample_id']}] {row['difficulty']} "
                        f"({row['num_words']} words)  {row['scene_id']} "
                        f"obj={row['object_id']} ann={row['ann_id']}\n")
                f.write(f"  description : {row['description']}\n")
                f.write(f"  object_name : {row['object_name']}\n")
                f.write(f"  target      : {row['parsed_target']}\n")
                f.write(f"  adjectives  : {row['parsed_adjectives']}\n")
                f.write(f"  neighbors   : {row['parsed_neighbors']}\n")
                f.write("  target_ok=__  adjectives_ok=__  neighbors_ok=__  error_type=______\n\n")

        written += [csv_path, txt_path]
        print(f"[{name}] -> {csv_path}")
        print(f"[{name}] -> {txt_path}")

    instructions_path = os.path.join(args.out_dir, "instructions.md")
    taxonomy_table = "\n".join(f"- `{code}` — {text}" for code, text in TAXONOMY.items())
    with open(instructions_path, "w") as f:
        f.write(INSTRUCTIONS.format(taxonomy=taxonomy_table))
        f.write("\n## This sheet is not failure-oversampled\n\n"
                "Rule 5 above describes the 200-row sheet. This one draws uniformly inside\n"
                "each length band, so no re-weighting is needed -- but with five rows a band\n"
                "it supports a qualitative error analysis, not a rate estimate.\n")

    manifest_path = save_json({
        "split": args.split, "seed": args.seed, "criterion": args.criterion,
        "anchor_parser": anchor_name,
        "design": "uniform within description-length bands; NOT failure-oversampled",
        "sampled": len(sample),
        "bands": {label: {"words": [low, high if high < 10 ** 6 else None],
                          "population": len(bands[label]),
                          "sampled": sum(1 for item in sample if item[0] == label)}
                  for label, low, high in DIFFICULTY_BANDS},
        "note": "Uniform inside each band, so band rates need no re-weighting. The bands "
                "are not equal shares of the population; do not pool them into a single "
                "rate without weighting by the population column.",
        "parsers": {name: folder for name, folder in parsers},
        "sample": [{"sample_id": f"D{i:02d}", "difficulty": label, "num_words": words,
                    "scene_id": r["scene_id"], "object_id": r["object_id"],
                    "ann_id": r["ann_id"]}
                   for i, (label, words, r) in enumerate(sample, 1)],
    }, os.path.join(args.out_dir, "sampling_manifest15.json"))

    print(f"\nwrote {instructions_path}")
    print(f"wrote {manifest_path}")
    print("\nAnnotate the CSV, then run:")
    print("    python experiments/analysis.py error-taxonomy " +
          " ".join(f"--sheet {name}={os.path.join(args.out_dir, f'annotation15_{name}.csv')}"
                   for name, _ in parsers) +
          f" --manifest {manifest_path}")
    return 0


# ---- error-taxonomy ----

_DOC_ERROR_TAXONOMY = """
Tally the filled annotation sheets into the parse-error taxonomy table. CPU, instant.

    python experiments/analysis.py error-taxonomy         --sheet gpt4o-mini=outputs/analysis/annotation/annotation_sheet_gpt4o-mini.csv

    # two annotators on the same parser -> Cohen's kappa
    python experiments/analysis.py error-taxonomy --sheet gpt4o-mini=a.csv --sheet gpt4o-mini=b.csv

Unknown error codes, out-of-range *_ok values and blank rows are reported as problems
rather than coerced, so a typo cannot become a category with n=1.

The sample oversamples failures, so its raw field accuracy is not the dataset rate. Both
come out: ``sample`` for the annotated set, ``population`` re-weighted by stratum.
"""

_TAXONOMY_FIELDS = ("target_ok", "adjectives_ok", "neighbors_ok")

_TRUE = {"1", "y", "yes", "true", "t", "ok"}
_FALSE = {"0", "n", "no", "false", "f"}


def to_bool(value):
    """Parse a tolerant yes/no cell. Returns True, False, or None when blank/unreadable."""
    text = (value or "").strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def read_sheet(path):
    """Return (rows, problems). Rows keep their raw cells; validation is reported."""
    if not os.path.isfile(path):
        raise SystemExit(f"annotation sheet not found: {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    problems = []
    for row in rows:
        sample_id = row.get("sample_id", "?")
        for field in _TAXONOMY_FIELDS:
            if field not in row:
                problems.append(f"{sample_id}: column {field!r} missing from the sheet")
            elif to_bool(row[field]) is None:
                problems.append(f"{sample_id}: {field}={row[field]!r} is not yes/no")
        for code in split_codes(row.get("error_type")):
            if code not in TAXONOMY:
                problems.append(f"{sample_id}: unknown error code {code!r} "
                                f"(known: {', '.join(sorted(TAXONOMY))})")
    return rows, problems


def split_codes(value):
    return [c.strip().lower() for c in (value or "").split(";") if c.strip()]


def load_weights(manifest_path):
    """Per-stratum weights from the sampling manifest, plus the sample_id -> stratum map.

    The manifest does not record which stratum each sample came from directly, but
    ``auto_target_match`` in the sheet does: MISS means the target-wrong stratum.
    """
    if not manifest_path or not os.path.isfile(manifest_path):
        return None
    with open(manifest_path) as f:
        manifest = json.load(f)
    strata = manifest.get("strata") or {}
    return {
        "target_wrong": strata.get("target_wrong", {}).get("weight", 1.0),
        "target_correct": strata.get("target_correct", {}).get("weight", 1.0),
        "population_total": (strata.get("target_wrong", {}).get("population", 0)
                             + strata.get("target_correct", {}).get("population", 0)),
    }


def stratum_of(row):
    return "target_wrong" if (row.get("auto_target_match", "").strip().upper() == "MISS") \
        else "target_correct"


def field_rates(rows, weights):
    """Sample and population-weighted accuracy per field."""
    result = {}
    for field in _TAXONOMY_FIELDS:
        values = [(stratum_of(row), to_bool(row.get(field))) for row in rows]
        graded = [(s, v) for s, v in values if v is not None]
        n = len(graded)
        hits = sum(1 for _s, v in graded if v)
        low, high = wilson_ci(hits, n)
        entry = {"n": n, "hits": hits, "sample_rate": (hits / n) if n else 0.0,
                 "ci": [low, high], "ungraded": len(values) - n}

        if weights:
            numerator, denominator = 0.0, 0.0
            for stratum in ("target_wrong", "target_correct"):
                subset = [v for s, v in graded if s == stratum]
                if not subset:
                    continue
                weight = weights.get(stratum, 1.0)
                numerator += weight * sum(1 for v in subset if v)
                denominator += weight * len(subset)
            entry["population_rate"] = (numerator / denominator) if denominator else None
        result[field] = entry
    return result


def cohens_kappa(a, b):
    """Cohen's kappa for two aligned binary label lists, ignoring unpaired blanks."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return None, 0
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    pa_true = sum(1 for x, _ in pairs if x) / n
    pb_true = sum(1 for _, y in pairs if y) / n
    expected = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if expected >= 1.0:
        return 1.0, n
    return (observed - expected) / (1 - expected), n


def _add_error_taxonomy_args(parser):
    parser.add_argument("--sheet", action="append", required=True, metavar="NAME=PATH",
                        help="Repeatable. Two sheets with the same NAME are treated as "
                             "two annotators and compared.")
    parser.add_argument("--manifest",
                        default="outputs/analysis/annotation/sampling_manifest.json",
                        help="Sampling manifest, used to re-weight to population rates.")
    parser.add_argument("--out-dir", dest="out_dir", default="outputs/analysis/annotation")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero if any validation problem is found.")
    return parser


def run_error_taxonomy(args):

    weights = load_weights(args.manifest)
    if weights and weights["population_total"]:
        print(f"[weights] target_wrong x{weights['target_wrong']:.2f}, "
              f"target_correct x{weights['target_correct']:.2f} "
              f"(population {weights['population_total']})")
    elif weights:
        # A manifest with no correctness strata -- annotation-sheet-15 draws uniformly
        # inside length bands, so its counts are already unweighted.
        print("[weights] manifest carries no correctness strata; counts are unweighted")
        weights = None
    else:
        print("[weights] no sampling manifest; population rates will not be reported")

    sheets = defaultdict(list)
    all_problems = []
    for spec in args.sheet:
        name, path = (spec.split("=", 1) if "=" in spec else (spec, spec))
        rows, problems = read_sheet(path.strip())
        sheets[name.strip()].append((path.strip(), rows))
        all_problems += [f"{name.strip()} [{os.path.basename(path)}] {p}" for p in problems]
        print(f"[{name.strip()}] {len(rows)} rows from {path.strip()}")

    if all_problems:
        print(f"\n{len(all_problems)} validation problem(s):")
        for problem in all_problems[:25]:
            print("  -", problem)
        if len(all_problems) > 25:
            print(f"  ... and {len(all_problems) - 25} more")
        print("  (rows with unreadable cells are excluded from that field's rate)")

    report, summary_rows, taxonomy_rows = {}, [], []
    for name, versions in sheets.items():
        path, rows = versions[0]
        rates = field_rates(rows, weights)

        codes = Counter()
        for row in rows:
            for code in split_codes(row.get("error_type")):
                codes[code] += 1
        graded_rows = sum(1 for row in rows if split_codes(row.get("error_type")))

        print(f"\n=== {name} ===")
        for field in _TAXONOMY_FIELDS:
            entry = rates[field]
            population = entry.get("population_rate")
            population_text = (f"  population {100 * population:.1f}%"
                               if population is not None else "")
            print(f"  {field:<15}: sample {100 * entry['sample_rate']:.1f}% "
                  f"[{100 * entry['ci'][0]:.1f}-{100 * entry['ci'][1]:.1f}]  "
                  f"n={entry['n']}{population_text}"
                  + (f"  ({entry['ungraded']} ungraded)" if entry["ungraded"] else ""))

        if codes:
            print(f"  error codes over {graded_rows} labelled rows:")
            for code, count in codes.most_common():
                print(f"    {code:<24} {count:>4}  "
                      f"{100.0 * count / max(graded_rows, 1):.1f}%")
        else:
            print("  no error_type values filled in yet")

        agreement = {}
        if len(versions) > 1:
            other_path, other_rows = versions[1]
            index = {row.get("sample_id"): row for row in other_rows}
            print(f"  agreement with {os.path.basename(other_path)}:")
            for field in _TAXONOMY_FIELDS:
                a = [to_bool(row.get(field)) for row in rows]
                b = [to_bool(index.get(row.get("sample_id"), {}).get(field)) for row in rows]
                kappa, n = cohens_kappa(a, b)
                raw = (sum(1 for x, y in zip(a, b)
                           if x is not None and y is not None and x == y) / n) if n else 0.0
                agreement[field] = {"kappa": kappa, "raw": raw, "n": n}
                kappa_text = f"{kappa:.3f}" if kappa is not None else "n/a"
                print(f"    {field:<15}: raw {100 * raw:.1f}%  kappa {kappa_text}  n={n}")

        report[name] = {"sheet": path, "n_rows": len(rows), "fields": rates,
                        "error_codes": dict(codes), "labelled_rows": graded_rows,
                        "agreement": agreement}

        row = [name, len(rows)]
        for field in _TAXONOMY_FIELDS:
            entry = rates[field]
            population = entry.get("population_rate")
            row.append(f"{100 * entry['sample_rate']:.1f}" +
                       (f" / {100 * population:.1f}" if population is not None else ""))
        summary_rows.append(row)

        for code in TAXONOMY:
            if codes.get(code):
                taxonomy_rows.append([name, code, codes[code],
                                      f"{100.0 * codes[code] / max(graded_rows, 1):.1f}"])

    summary = md_table(
        ["parser", "n", "target_ok (sample/pop %)", "adjectives_ok (sample/pop %)",
         "neighbors_ok (sample/pop %)"], summary_rows)
    print("\n" + summary)

    lines = ["# Manual parse annotation — results\n",
             "Field accuracy, as `sample / population`. The sample oversamples target "
             "failures on purpose; the population column re-weights by stratum and is the "
             "one to quote as a dataset rate.\n", summary]
    if taxonomy_rows:
        taxonomy_table = md_table(["parser", "error code", "count", "% of labelled rows"],
                                  taxonomy_rows)
        print("\n" + taxonomy_table)
        lines += ["\n## Error taxonomy\n", taxonomy_table,
                  "\n" + "\n".join(f"- `{code}` — {text}" for code, text in TAXONOMY.items())]

    ensure_dir(args.out_dir)
    report_path = os.path.join(args.out_dir, "error_taxonomy.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json({"weights": weights, "problems": all_problems,
                           "parsers": report},
                          os.path.join(args.out_dir, "error_taxonomy.json"))

    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")

    if all_problems and args.strict:
        return 1
    return 0


# ---- error-propagation ----

_DOC_ERROR_PROPAGATION = """
Do parse errors propagate to the grounding result? CPU, seconds.

    python experiments/analysis.py error-propagation         --run-dir outputs/2024-12-18_20-40-38_3DVG-FIXED/corruption

Where parse_quality_split.py observes an association, this intervenes: sample, model and
weights are fixed and only the parse changes. Producing the input needs a GPU:

    1. python experiments/ablation/parsing.py corrupt --splits val --rates 0.10 0.25 0.50
    2. python experiments/ablation/train_arms.py corruption-sweep     # GPU, eval only
    3. this script --run-dir outputs/<run>/corruption

Three numbers come out: the global accuracy curve (at rate r only a fraction r was touched,
so the damage is diluted by 1 - r); the corrupted subset against the untouched one, which
acts as a control and should stay flat; and the paired flip rate over corrupted
annotations, tested with McNemar. See details.md §4.
"""

_RATE_IN_NAME = re.compile(r"_(\d{1,3})$")


def _prop_discover(run_dir):
    """Find every level directory under run_dir. Returns a list sorted by rate."""
    if not os.path.isdir(run_dir):
        raise SystemExit(
            f"run directory not found: {run_dir}\n"
            f"Produce it first (GPU):\n"
            f"    python experiments/ablation/train_arms.py corruption-sweep")

    levels = []
    for entry in sorted(os.listdir(run_dir)):
        directory = os.path.join(run_dir, entry)
        predictions = os.path.join(directory, "predictions.p")
        if not os.path.isfile(predictions):
            continue

        manifest_path = os.path.join(directory, "corruption_manifest.json")
        manifest = None
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)

        if manifest and manifest.get("requested_rate") is not None:
            rate = float(manifest["requested_rate"])
        elif entry.lower() in ("baseline", "clean", "rate_0", "00"):
            rate = 0.0
        else:
            match = _RATE_IN_NAME.search(entry)
            rate = int(match.group(1)) / 100.0 if match else float("nan")

        levels.append({"name": entry, "dir": directory, "rate": rate,
                       "predictions": predictions, "manifest": manifest})

    levels.sort(key=lambda level: (math.isnan(level["rate"]), level["rate"]))
    return levels


def corrupted_keys(manifest, split):
    """{(scene, object, ann)} that were actually rewritten at this level."""
    if not manifest:
        return None
    entries = (manifest.get("corrupted") or {}).get(split)
    if entries is None:
        # Fall back to whatever split the manifest does carry.
        buckets = manifest.get("corrupted") or {}
        entries = next(iter(buckets.values()), []) if buckets else []
    return {(str(e["scene_id"]), str(e["object_id"]), str(e["ann_id"])) for e in entries}


def _prop_mcnemar(broke, fixed):
    """Exact-ish McNemar with continuity correction. Returns (chi2, p)."""
    n = broke + fixed
    if n == 0:
        return float("nan"), float("nan")
    chi2 = (abs(broke - fixed) - 1) ** 2 / n
    return chi2, math.erfc(math.sqrt(chi2 / 2.0))


def _add_error_propagation_args(parser):
    parser.add_argument("--run-dir", dest="run_dir", required=True,
                        help="Directory holding one subdirectory per corruption level.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--threshold", type=float, default=0.25, choices=list(THRESHOLDS))
    parser.add_argument("--baseline", default=None,
                        help="Explicit path to the uncorrupted predictions.p. Defaults to "
                             "the rate-0 level inside --run-dir.")
    parser.add_argument("--out-dir", dest="out_dir",
                        default="outputs/analysis/parse_error_propagation")
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser


def run_error_propagation(args):

    levels = _prop_discover(args.run_dir)
    if not levels:
        raise SystemExit(f"no predictions.p found under {args.run_dir}")
    print(f"[levels] {len(levels)} found: " +
          ", ".join(f"{l['name']}({100 * l['rate']:.0f}%)" for l in levels))

    baseline_predictions = None
    if args.baseline:
        baseline_predictions = load_predictions(args.baseline)
    else:
        for level in levels:
            if level["rate"] == 0.0:
                baseline_predictions = load_predictions(level["predictions"])
                break
    if baseline_predictions is None:
        print("[baseline] no rate-0 level found; the paired flip analysis is unavailable. "
              "Add a 'baseline/' directory or pass --baseline.")

    results, table_rows = [], []
    for level in levels:
        predictions = load_predictions(level["predictions"])
        ious = [float(record["iou"]) for record in predictions.values()]
        hits = sum(1 for i in ious if i >= args.threshold)
        low, high = wilson_ci(hits, len(ious))

        entry = {
            "name": level["name"], "rate": level["rate"], "n": len(ious),
            "acc": accuracy(ious, args.threshold), "ci": [low, high],
            "acc_other": {str(t): accuracy(ious, t) for t in THRESHOLDS},
        }

        keys = corrupted_keys(level["manifest"], args.split)
        if keys:
            touched = [float(predictions[k]["iou"]) for k in keys if k in predictions]
            untouched = [float(record["iou"]) for key, record in predictions.items()
                         if key not in keys]
            entry["corrupted_subset"] = {
                "n": len(touched), "acc": accuracy(touched, args.threshold)}
            entry["untouched_subset"] = {
                "n": len(untouched), "acc": accuracy(untouched, args.threshold)}

            if baseline_predictions:
                broke = fixed = same = 0
                for key in keys:
                    if key not in predictions or key not in baseline_predictions:
                        continue
                    before = float(baseline_predictions[key]["iou"]) >= args.threshold
                    after = float(predictions[key]["iou"]) >= args.threshold
                    if before and not after:
                        broke += 1
                    elif after and not before:
                        fixed += 1
                    else:
                        same += 1
                chi2, p = _prop_mcnemar(broke, fixed)
                paired_n = broke + fixed + same
                entry["paired"] = {
                    "n": paired_n, "broke": broke, "fixed": fixed, "unchanged": same,
                    "net_broken": broke - fixed,
                    "net_rate": (broke - fixed) / paired_n if paired_n else 0.0,
                    "mcnemar_chi2": chi2, "mcnemar_p": p,
                }

        results.append(entry)

        row = [level["name"], f"{100 * level['rate']:.0f}", entry["n"],
               f"{pct(entry['acc'])} [{pct(low, 1)}-{pct(high, 1)}]"]
        if "corrupted_subset" in entry:
            row += [f"{entry['corrupted_subset']['n']}",
                    pct(entry["corrupted_subset"]["acc"]),
                    pct(entry["untouched_subset"]["acc"])]
        else:
            row += ["-", "-", "-"]
        if "paired" in entry:
            paired = entry["paired"]
            row += [f"{paired['broke']}/{paired['fixed']}",
                    f"{100 * paired['net_rate']:+.1f}", f"{paired['mcnemar_p']:.2e}"]
        else:
            row += ["-", "-", "-"]
        table_rows.append(row)

    table = md_table(
        ["level", "rate %", "n", f"global Acc@{args.threshold}",
         "n corrupted", "corrupted Acc", "untouched Acc",
         "broke/fixed", "net %", "McNemar p"],
        table_rows)
    print("\n" + table)

    lines = ["# Parse-error propagation\n",
             f"split: `{args.split}` | IoU threshold: {args.threshold} | "
             f"source: `{args.run_dir}`\n", table,
             "\n**global** dilutes the effect by (1 - rate): only the corrupted fraction "
             "was touched.\n",
             "**untouched Acc** is the control column — it should stay flat across "
             "levels.\n",
             "**broke/fixed** counts corrupted annotations that flipped against their own "
             "baseline outcome. `net %` is (broke - fixed) / n, and McNemar's test is the "
             "correct significance test for a paired binary flip.\n"]

    baseline_entry = next((r for r in results if r["rate"] == 0.0), None)
    verdicts = []
    if baseline_entry:
        worst = max((r for r in results if r["rate"] > 0),
                    key=lambda r: r["rate"], default=None)
        if worst:
            drop = baseline_entry["acc"] - worst["acc"]
            verdicts.append(
                f"At {100 * worst['rate']:.0f}% corruption the global accuracy falls "
                f"{100 * drop:.2f} pp (from {pct(baseline_entry['acc'])}% to "
                f"{pct(worst['acc'])}%).")
            if "paired" in worst:
                paired = worst["paired"]
                significant = paired["mcnemar_p"] == paired["mcnemar_p"] \
                    and paired["mcnemar_p"] < 0.05
                verdicts.append(
                    f"On the corrupted annotations themselves, {paired['broke']} flipped "
                    f"from correct to wrong and {paired['fixed']} the other way "
                    f"(net {100 * paired['net_rate']:+.1f}%, McNemar p="
                    f"{paired['mcnemar_p']:.2e})" +
                    (" — parse errors do propagate." if significant
                     else " — not statistically significant, so the model is fairly "
                          "robust to this corruption. Report that as the finding."))

    # If the control fails, the whole table is suspect.
    controls = [r["untouched_subset"]["acc"] for r in results if "untouched_subset" in r]
    if len(controls) > 1:
        spread = max(controls) - min(controls)
        verdicts.append(
            f"Control: accuracy on untouched annotations varies by {100 * spread:.2f} pp "
            f"across levels" +
            (" — flat, as it should be." if spread < 0.02 else
             " — this should be near zero; a large spread means the corruption affected "
             "annotations it was not supposed to, and the table cannot be trusted."))

    if verdicts:
        print("\nReading:")
        for verdict in verdicts:
            print(f"  {verdict}")
        lines.append("\n## Reading\n")
        lines += [f"- {v}" for v in verdicts]

    ensure_dir(args.out_dir)
    report_path = os.path.join(args.out_dir, "parse_error_propagation.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json({"run_dir": args.run_dir, "split": args.split,
                           "threshold": args.threshold, "levels": results,
                           "verdicts": verdicts},
                          os.path.join(args.out_dir, "parse_error_propagation.json"))
    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}")

    if args.plot:
        path = _prop_make_plot(results, args.threshold,
                         os.path.join(args.out_dir, "parse_error_propagation.png"))
        if path:
            print(f"wrote {path}")
    return 0


def _prop_make_plot(results, threshold, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the plot")
        return None

    usable = [r for r in results if not math.isnan(r["rate"])]
    if len(usable) < 2:
        return None
    rates = [100 * r["rate"] for r in usable]

    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    axis.plot(rates, [100 * r["acc"] for r in usable], marker="o", label="all annotations")
    if any("corrupted_subset" in r for r in usable):
        pairs = [(100 * r["rate"], 100 * r["corrupted_subset"]["acc"])
                 for r in usable if "corrupted_subset" in r]
        axis.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="s",
                  label="corrupted subset")
        pairs = [(100 * r["rate"], 100 * r["untouched_subset"]["acc"])
                 for r in usable if "untouched_subset" in r]
        axis.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="^",
                  linestyle="--", label="untouched subset (control)")

    axis.set_xlabel("parse corruption rate (%)")
    axis.set_ylabel(f"Acc@{threshold} (%)")
    axis.set_title("Grounding accuracy under deliberate parse corruption")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


# ---- parse-field-comparison ----

_DOC_PARSE_FIELD_COMPARISON = """
Adjectives and neighbors, scored across parsers -- the two fields nothing else measures.

    python experiments/analysis.py parse-field-comparison         --parse gpt4o-mini=final_parsing_tokenized         --parse spacy=spacy_parsing_tokenized

eval_parser_target_accuracy.py scores the target field against object_name. The other two
have no ground truth, yet they are what the fusion network consumes -- TAF reads the
attribute field and A2F the adjacency field. This scores them reference-free:

    coverage      how often the parser declines the slot ("not mentioned")
    faithfulness  fraction of emitted tokens occurring in the source description
    agreement     mean per-annotation Jaccard against every other parser

Faithfulness alone is not a quality signal -- a copy-only parser scores near 100% by
construction and one emitting nothing scores perfectly. Read it next to coverage.
"""

#: `target` is deliberately absent -- it has a scorer with real ground truth, and a
#: reference-free number should not be quoted when a referenced one exists.
FIELDS = ("adjectives", "neighbors")


def key_of(record):
    return (str(record["scene_id"]), str(record["object_id"]), str(record["ann_id"]))


def is_empty(tokens):
    """True when the parser declined the slot."""
    return not tokens or list(tokens) == list(NOT_MENTIONED_TOKENS)


def jaccard(a, b):
    """Token-set overlap. Two declined fields agree perfectly -- that is a real agreement."""
    set_a, set_b = set(a or []), set(b or [])
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 1.0


def phrase_count(tokens):
    """Comma-separated segments, exactly as linguistic_complexity.metric_neighbors counts
    them, so the numbers here line up with that report."""
    if is_empty(tokens):
        return 0
    return list(tokens).count(",") + 1


def collect(caches, records):
    """Per-parser per-field statistics over the annotations every parser covers.

    Restricting to the common key set is what makes the comparison paired: a parser that
    simply skipped the hard descriptions must not look good by having a smaller, easier
    denominator.
    """
    keys, parses = [], {name: {} for name in caches}
    for record in records:
        key = key_of(record)
        found = {name: parse_for(cache, *key) for name, cache in caches.items()}
        if any(parse is None for parse in found.values()):
            continue
        keys.append(key)
        for name, parse in found.items():
            parses[name][key] = parse
    return keys, parses


def field_stats(keys, parses, records_by_key, name, field):
    empty = faithful_hits = faithful_total = 0
    lengths, phrases, capped = [], [], 0
    for key in keys:
        tokens = parses[name][key].get(field) or []
        if is_empty(tokens):
            empty += 1
        else:
            source = {t.lower() for t in (records_by_key[key].get("token") or [])}
            faithful_hits += sum(1 for t in tokens if str(t).lower() in source)
            faithful_total += len(tokens)
            lengths.append(len(tokens))
        phrases.append(phrase_count(tokens))
        if len(tokens) >= FIELD_MAX_TOKENS[field]:
            capped += 1

    total = len(keys)
    low, high = wilson_ci(empty, total)
    return {
        "n": total,
        "empty": empty,
        "empty_rate": empty / total if total else 0.0,
        "empty_ci": [low, high],
        "capped": capped,
        "mean_tokens_when_present": (sum(lengths) / len(lengths)) if lengths else 0.0,
        "mean_phrases": (sum(phrases) / total) if total else 0.0,
        "faithfulness": (faithful_hits / faithful_total) if faithful_total else float("nan"),
        "faithful_tokens": faithful_hits,
        "emitted_tokens": faithful_total,
    }


def agreement_matrix(keys, parses, names, field):
    """Mean per-annotation Jaccard for every parser pair."""
    matrix = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            score = sum(jaccard(parses[a][k].get(field), parses[b][k].get(field))
                        for k in keys) / max(len(keys), 1)
            matrix[f"{a} vs {b}"] = score
    return matrix


def consensus_agreement(keys, parses, names, field, reference_names):
    """Each parser's mean Jaccard against the reference parsers (excluding itself)."""
    scores = {}
    for name in names:
        others = [r for r in reference_names if r != name]
        if not others:
            scores[name] = float("nan")
            continue
        scores[name] = sum(
            sum(jaccard(parses[name][k].get(field), parses[other][k].get(field))
                for other in others) / len(others)
            for k in keys) / max(len(keys), 1)
    return scores


def _pfc_make_plot(results, names, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the plot")
        return None

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2), squeeze=False)
    x = range(len(names))

    axis = axes[0][0]
    width = 0.38
    for offset, field in zip((-width / 2, width / 2), FIELDS):
        y = [100 * results[field]["parsers"][n]["empty_rate"] for n in names]
        axis.bar([i + offset for i in x], y, width, label=field)
    axis.set_ylabel("declined the slot (%)")
    axis.set_title("Coverage -- lower is better")

    axis = axes[0][1]
    for offset, field in zip((-width / 2, width / 2), FIELDS):
        y = [100 * results[field]["parsers"][n]["faithfulness"] for n in names]
        axis.bar([i + offset for i in x], y, width, label=field)
    axis.set_ylim(80, 101)
    axis.set_ylabel("tokens found in the description (%)")
    axis.set_title("Faithfulness -- near-tautological\nfor a copy-only parser")

    axis = axes[0][2]
    for offset, field in zip((-width / 2, width / 2), FIELDS):
        y = [results[field]["consensus"][n] for n in names]
        axis.bar([i + offset for i in x], y, width, label=field)
    axis.set_ylabel("mean Jaccard vs the other parsers")
    axis.set_title("Agreement -- higher is better")

    for axis in axes[0]:
        axis.set_xticks(list(x))
        axis.set_xticklabels(names, rotation=15, ha="right")
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)

    figure.suptitle("Parser comparison on the two fields with no ground truth")
    figure.tight_layout()
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _add_parse_field_comparison_args(parser):
    parser.add_argument("--parse", action="append", default=[], metavar="NAME=FOLDER",
                        help="Repeatable. Parse cache under data_parsing/.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--reference", action="append", default=None, metavar="NAME",
                        help="Parsers forming the consensus band. Default: every parser.")
    parser.add_argument("--out-dir", dest="out_dir",
                        default=os.path.join("outputs", "analysis", "parse_field_comparison"))
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser


def run_parse_field_comparison(args):

    if not args.parse:
        args._parser.error("at least one --parse NAME=FOLDER is required")

    caches, folders = {}, {}
    for item in args.parse:
        name, _, folder = item.partition("=")
        if not folder:
            name, folder = os.path.basename(item), item
        try:
            caches[name] = load_parse_cache(folder, args.split, args.parsing_root)
            folders[name] = folder
        except FileNotFoundError as error:
            # Skipping rather than dying: a missing optional variant should not stop the
            # report on the ones that are present.
            print(f"  skipping {name}: {error}")
    if not caches:
        print("no parse cache could be loaded")
        return 1

    names = list(caches)
    references = [n for n in (args.reference or names) if n in caches] or names

    records = load_scanrefer(args.split, args.data_root)
    records_by_key = {key_of(r): r for r in records}
    keys, parses = collect(caches, records)
    if not keys:
        print("no annotation is covered by every parse cache")
        return 1

    print(f"split: {args.split} | annotations covered by all parsers: {len(keys)} "
          f"of {len(records)}")
    print(f"parsers: {', '.join(names)}")
    print(f"consensus reference: {', '.join(references)}\n")

    lines = ["# Parser comparison -- adjectives and neighbors", "",
             f"split: `{args.split}` | annotations: {len(keys)} (covered by every parser) "
             f"| parsers: {', '.join(names)}", "",
             "These two fields have **no ground truth** -- ScanRefer's `object_name` only "
             "grounds the `target` field. Every number below is reference-free.", ""]

    results = {}
    for field in FIELDS:
        stats = {n: field_stats(keys, parses, records_by_key, n, field) for n in names}
        pairs = agreement_matrix(keys, parses, names, field)
        consensus = consensus_agreement(keys, parses, names, field, references)
        results[field] = {"parsers": stats, "pairwise": pairs, "consensus": consensus}

        rows = []
        for name in names:
            s = stats[name]
            rows.append([
                name,
                f"{100 * s['empty_rate']:.1f}",
                f"[{100 * s['empty_ci'][0]:.1f}, {100 * s['empty_ci'][1]:.1f}]",
                f"{s['mean_tokens_when_present']:.2f}",
                f"{s['mean_phrases']:.2f}",
                f"{100 * s['faithfulness']:.1f}",
                f"{consensus[name]:.3f}",
            ])
        table = md_table(
            ["parser", "declined %", "95% CI", "tokens when present", "phrases",
             "faithful %", "agreement"], rows)

        pair_table = md_table(["pair", "mean Jaccard"],
                              [[k, f"{v:.3f}"] for k, v in pairs.items()])

        print(f"--- {field} ---")
        print(table)
        print()
        print(pair_table)
        print()

        lines += [f"## {field}", "", table, "", "**Pairwise agreement**", "",
                  pair_table, ""]

    # Verdicts, written from the numbers rather than asserted.
    lines += ["## Reading", ""]
    for field in FIELDS:
        stats = results[field]["parsers"]
        worst = max(names, key=lambda n: stats[n]["empty_rate"])
        best = min(names, key=lambda n: stats[n]["empty_rate"])
        ratio = (stats[worst]["empty_rate"] / stats[best]["empty_rate"]
                 if stats[best]["empty_rate"] else float("inf"))
        line = (f"- `{field}`: **{worst}** declines the slot most often "
                f"({100 * stats[worst]['empty_rate']:.1f}%), "
                f"{ratio:.1f}x as often as {best} "
                f"({100 * stats[best]['empty_rate']:.1f}%).")
        lines.append(line)
        print(line)

        consensus = results[field]["consensus"]
        odd = min(names, key=lambda n: consensus[n])
        top = max(names, key=lambda n: consensus[n])
        if len(names) > 2:
            line = (f"- `{field}`: **{odd}** agrees least with the others "
                    f"(mean Jaccard {consensus[odd]:.3f} against {consensus[top]:.3f} "
                    f"for {top}).")
            lines.append(line)
            print(line)

        faith = {n: stats[n]["faithfulness"] for n in names}
        most_faithful = max(names, key=lambda n: faith[n])
        if most_faithful == worst:
            line = (f"- `{field}`: {worst} scores highest on faithfulness "
                    f"({100 * faith[worst]:.1f}%) **while** declining the slot most often. "
                    f"That combination is the signature of a copy-only parser -- it cannot "
                    f"invent a token, so faithfulness is near-tautological for it. Read it "
                    f"beside coverage, never alone.")
            lines.append(line)
            print(line)

    lines += ["",
              "Scope: this measures parse quality, not grounding accuracy. Whether a "
              "worse parse *causes* a worse box is the corruption experiment "
              "(`run_parse_corruption.py` -> `parse_error_propagation.py`), not this.", ""]

    ensure_dir(args.out_dir)
    md_path = os.path.join(args.out_dir, "parse_field_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    json_path = save_json({
        "split": args.split,
        "annotations": len(keys),
        "parsers": folders,
        "reference": references,
        "fields": results,
    }, os.path.join(args.out_dir, "parse_field_comparison.json"))

    print(f"\nwrote {md_path}")
    print(f"wrote {json_path}")
    if args.plot:
        png = _pfc_make_plot(results, names,
                        os.path.join(args.out_dir, "parse_field_comparison.png"))
        if png:
            print(f"wrote {png}")
    return 0


# ---- complex-sentence-showdown ----

_DOC_COMPLEX_SENTENCE_SHOWDOWN = """
Parsers head to head on the hardest descriptions in the dataset. CPU, seconds.

    python experiments/analysis.py complex-sentence-showdown --num 10

One number per parser over the whole split hides where the difference lives: parsers agree
on "there is a black chair" and diverge on long, deeply nested descriptions. Two things
come out together -- the N hardest descriptions side by side, and the same measurements
over the whole split in complexity quartiles, which keeps the former from being
cherry-picked.

Complexity is spaCy dependency-tree depth, tie-broken by token count, read from
outputs/analysis/dep_depth_cache.json which linguistic_complexity.py populates. Scope is
parse quality; no model is run and no prediction is read.
"""

RANKERS = {"depth": metric_depth, "tokens": metric_tokens, "spatial": metric_spatial}

#: Quartile labels, hardest last.
QUARTILES = ("Q1 simplest", "Q2", "Q3", "Q4 hardest")


def field_text(parse, field):
    tokens = parse.get(field) or []
    return " ".join(str(t) for t in tokens)


def quartile_indices(values):
    """Split rows into four equal-count buckets by rank, ties broken by position.

    Rank-based rather than value-based: dependency depth is a small integer and whole
    quartiles would otherwise collapse into a single value.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    buckets = [0] * len(values)
    size = max(len(values) / 4.0, 1e-9)
    for rank, index in enumerate(order):
        buckets[index] = min(int(rank / size), 3)
    return buckets


def stratified(keys, parses, names, buckets_by_key, field):
    """Declined-rate per parser per complexity quartile."""
    table = {name: [] for name in names}
    for q in range(4):
        subset = [k for k in keys if buckets_by_key[k] == q]
        for name in names:
            empty = sum(1 for k in subset if is_empty(parses[name][k].get(field)))
            table[name].append({
                "quartile": QUARTILES[q],
                "n": len(subset),
                "empty_rate": empty / len(subset) if subset else 0.0,
            })
    return table


def _css_make_plot(cases, names, strat, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available; skipping the figure")
        return None

    n_rows = len(cases)
    # One row per case, one column per parser, plus a full-width summary row on top.
    figure = plt.figure(figsize=(4.6 * len(names), 2.0 + 2.05 * n_rows))
    grid = figure.add_gridspec(n_rows + 1, len(names), height_ratios=[1.5] + [1] * n_rows)

    summary = figure.add_subplot(grid[0, :])
    x = range(4)
    for name in names:
        y = [100 * entry["empty_rate"] for entry in strat["adjectives"][name]]
        summary.plot(x, y, marker="o", label=name)
    summary.set_xticks(list(x))
    summary.set_xticklabels(QUARTILES)
    summary.set_ylabel("adjectives declined (%)")
    summary.set_title("Whole split: how often each parser leaves the attribute slot empty, "
                      "by description complexity")
    summary.grid(alpha=0.3)
    summary.legend(fontsize=8)

    for r, case in enumerate(cases):
        for c, name in enumerate(names):
            axis = figure.add_subplot(grid[r + 1, c])
            axis.axis("off")
            parse = case["parsers"][name]
            if r == 0:
                axis.set_title(name, fontsize=10, fontweight="bold")

            body = []
            for field in ("target",) + FIELDS:
                value = parse[field]
                missing = (field != "target" and parse[f"{field}_empty"])
                label = f"{field:11s}"
                body.append((label + (value if value else "-"), missing))

            axis.text(0.01, 0.97, f"depth {case['depth']} | {case['tokens']} tokens",
                      fontsize=7, color="0.45", va="top", family="monospace")
            y = 0.80
            for text, missing in body:
                for piece in textwrap.wrap(text, 46, subsequent_indent=" " * 13) or [text]:
                    axis.text(0.01, y, piece, fontsize=7.6, va="top", family="monospace",
                              color="#B00020" if missing else "0.15")
                    y -= 0.14
            axis.set_facecolor("#FFF6F6" if any(m for _, m in body) else "white")
            axis.patch.set_visible(True)

    figure.suptitle("Parser output on the hardest descriptions "
                    "(red = the parser declined the slot)", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _add_complex_sentence_showdown_args(parser):
    parser.add_argument("--parse", action="append", default=[], metavar="NAME=FOLDER")
    parser.add_argument("--num", type=int, default=10,
                        help="How many of the hardest descriptions to show.")
    parser.add_argument("--rank-by", dest="rank_by", default="depth", choices=sorted(RANKERS))
    parser.add_argument("--split", default="val")
    parser.add_argument("--data-root", dest="data_root", default="data")
    parser.add_argument("--parsing-root", dest="parsing_root", default="data_parsing")
    parser.add_argument("--parse-folder", dest="parse_folder", default="final_parsing_tokenized",
                        help="Cache the 'spatial'/'neighbors' rankers read.")
    parser.add_argument("--spacy-model", dest="spacy_model", default="en_core_web_sm")
    parser.add_argument("--depth-cache", dest="depth_cache",
                        default=os.path.join("outputs", "analysis", "dep_depth_cache.json"))
    parser.add_argument("--out-dir", dest="out_dir",
                        default=os.path.join("outputs", "analysis", "complex_sentence_showdown"))
    parser.add_argument("--no-plot", dest="plot", action="store_false")
    return parser


def run_complex_sentence_showdown(args):

    if not args.parse:
        args.parse = ["gpt4o-mini=final_parsing_tokenized",
                      "llama=llama_parsing_tokenized_clipped",
                      "spacy=spacy_parsing_tokenized"]

    caches, folders = {}, {}
    for item in args.parse:
        name, _, folder = item.partition("=")
        if not folder:
            name, folder = os.path.basename(item), item
        try:
            caches[name] = load_parse_cache(folder, args.split, args.parsing_root)
            folders[name] = folder
        except FileNotFoundError as error:
            print(f"  skipping {name}: {error}")
    if not caches:
        print("no parse cache could be loaded")
        return 1
    names = list(caches)

    records = load_scanrefer(args.split, args.data_root)
    rows = [r for r in records
            if all(parse_for(caches[n], *key_of(r)) is not None for n in names)]
    if not rows:
        print("no annotation is covered by every parse cache")
        return 1

    values = RANKERS[args.rank_by](rows, args)
    depths = values if args.rank_by == "depth" else metric_depth(rows, args)
    tokens = [len(r.get("token") or []) for r in rows]

    buckets = quartile_indices(values)
    keys = [key_of(r) for r in rows]
    parses = {n: {key_of(r): parse_for(caches[n], *key_of(r)) for r in rows} for n in names}
    buckets_by_key = {key_of(r): buckets[i] for i, r in enumerate(rows)}

    strat = {f: stratified(keys, parses, names, buckets_by_key, f) for f in FIELDS}

    # The hardest N, ranked by the chosen metric then by length.
    order = sorted(range(len(rows)), key=lambda i: (values[i], tokens[i]), reverse=True)
    cases = []
    for i in order[:args.num]:
        row, key = rows[i], key_of(rows[i])
        case = {
            "scene_id": row["scene_id"], "object_id": row["object_id"],
            "ann_id": row["ann_id"], "object_name": row.get("object_name"),
            "description": row.get("description", ""),
            "depth": int(depths[i]), "tokens": int(tokens[i]),
            "rank_value": int(values[i]), "parsers": {},
        }
        for name in names:
            parse = parses[name][key]
            entry = {"target": field_text(parse, "target")}
            for field in FIELDS:
                entry[field] = field_text(parse, field)
                entry[f"{field}_empty"] = is_empty(parse.get(field))
            case["parsers"][name] = entry
        cases.append(case)

    print(f"split: {args.split} | annotations: {len(rows)} | ranked by: {args.rank_by}")
    print(f"parsers: {', '.join(names)}\n")

    lines = ["# Parsers on the hardest descriptions", "",
             f"split: `{args.split}` | annotations: {len(rows)} | ranked by "
             f"`{args.rank_by}` | showing the {len(cases)} hardest", "",
             "No ground truth exists for `adjectives` or `neighbors`, so this reports "
             "**coverage** (how often a parser leaves the slot empty) and, for the "
             "selected cases, the raw output for inspection.", ""]

    # --- the statistic ---
    lines += ["## Whole split, by complexity quartile", ""]
    for field in FIELDS:
        header = ["parser"] + [f"{q} (declined %)" for q in QUARTILES] + ["Q4 - Q1"]
        table_rows = []
        for name in names:
            entries = strat[field][name]
            gap = 100 * (entries[3]["empty_rate"] - entries[0]["empty_rate"])
            table_rows.append([name] + [f"{100 * e['empty_rate']:.1f}" for e in entries]
                              + [f"{gap:+.1f}"])
        table = md_table(header, table_rows)
        print(f"--- {field}: declined rate by complexity quartile ---")
        print(table)
        print()
        lines += [f"### {field}", "", table, ""]

    # --- verdicts from the numbers ---
    lines += ["**Reading:**", ""]
    adj = strat["adjectives"]
    worst = max(names, key=lambda n: adj[n][3]["empty_rate"])
    best = min(names, key=lambda n: adj[n][3]["empty_rate"])
    v = [f"- On the hardest quartile, **{worst}** leaves the attribute slot empty "
         f"{100 * adj[worst][3]['empty_rate']:.1f}% of the time against "
         f"{100 * adj[best][3]['empty_rate']:.1f}% for {best}."]
    gap_worst = adj[worst][3]["empty_rate"] - adj[worst][0]["empty_rate"]
    gap_best = adj[best][3]["empty_rate"] - adj[best][0]["empty_rate"]
    if gap_worst <= gap_best:
        v.append(f"- Both degrade with complexity, and {worst} does **not** degrade faster "
                 f"({100 * gap_worst:+.1f}pp against {100 * gap_best:+.1f}pp from the "
                 f"simplest to the hardest quartile). The damning number is the level, not "
                 f"the slope: {worst} starts roughly twice as bad and stays there.")
    else:
        v.append(f"- {worst} also degrades faster ({100 * gap_worst:+.1f}pp against "
                 f"{100 * gap_best:+.1f}pp), so the gap widens with complexity.")
    for line in v:
        lines.append(line)
        print(line)
    print()

    # --- the cases ---
    lines += ["", "## The hardest descriptions", ""]
    case_lines = []
    for n, case in enumerate(cases, 1):
        head = (f"### {n}. {case['scene_id']} obj {case['object_id']} ann {case['ann_id']} "
                f"-- depth {case['depth']}, {case['tokens']} tokens, "
                f"class `{case['object_name']}`")
        lines += [head, "", f"> {case['description']}", ""]
        rows_md = []
        for name in names:
            entry = case["parsers"][name]
            rows_md.append([
                name, entry["target"],
                entry["adjectives"] + (" **(empty)**" if entry["adjectives_empty"] else ""),
                entry["neighbors"] + (" **(empty)**" if entry["neighbors_empty"] else ""),
            ])
        lines += [md_table(["parser", "target", "adjectives", "neighbors"], rows_md), ""]

        case_lines.append(f"[{n}] {case['scene_id']} obj{case['object_id']} "
                          f"ann{case['ann_id']}  depth={case['depth']} "
                          f"tokens={case['tokens']}  class={case['object_name']}")
        case_lines.append(f"    {case['description']}")
        for name in names:
            entry = case["parsers"][name]
            case_lines.append(f"    {name:12s} target={entry['target']!r} "
                              f"adjectives={entry['adjectives']!r} "
                              f"neighbors={entry['neighbors']!r}")
        case_lines.append("")

    lines += ["Scope: parse quality only. No model was run and no prediction was read, so "
              "nothing here speaks to grounding accuracy -- that is the corruption "
              "experiment.", ""]

    ensure_dir(args.out_dir)
    md_path = os.path.join(args.out_dir, "complex_sentence_showdown.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    txt_path = os.path.join(args.out_dir, "cases.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(case_lines) + "\n")
    json_path = save_json({
        "split": args.split, "annotations": len(rows), "rank_by": args.rank_by,
        "parsers": folders, "quartiles": strat, "cases": cases,
    }, os.path.join(args.out_dir, "complex_sentence_showdown.json"))

    print(f"wrote {md_path}")
    print(f"wrote {txt_path}")
    print(f"wrote {json_path}")
    if args.plot:
        png = _css_make_plot(cases, names, strat,
                        os.path.join(args.out_dir, "complex_sentence_showdown.png"))
        if png:
            print(f"wrote {png}")
    return 0


# ---- seed-aggregate ----

_DOC_SEED_AGGREGATE = """
Aggregate finished runs into mean +/- std. CPU, instant.

    python experiments/analysis.py seed-aggregate --pattern 'ABL-SEED*'
    python experiments/analysis.py seed-aggregate --pattern '*ABL-PARSER-*' --group-by tag

Reads outputs/<run>/best.txt, which training writes at the end holding the best epoch's
metrics. Every ``key: value`` pair on a tagged line is collected, so metrics beyond the
defaults are reachable via --metrics.

--paired additionally loads scores.p and, for exactly two groups, runs a paired
per-annotation McNemar comparison -- a sharper instrument than comparing two means over
three seeds when the difference at stake is under a point.

With one run the standard deviation is undefined and it says so rather than printing 0.00.
"""

sys.path.insert(0, REPO_ROOT)

DEFAULT_METRICS = ("iou_rate_0.25", "iou_rate_0.5", "ref_acc", "lang_acc", "obj_acc")

#: `[sco.] iou_rate_0.25: 0.49789, iou_rate_0.5: 0.3588` -> both pairs.
_PAIR = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*:\s*(-?[0-9]*\.?[0-9]+)")
_TAGGED_LINE = re.compile(r"^\s*\[(best|loss|sco\.|train|val)\]")


def read_best(path):
    """Every key: value pair in a best.txt, as {key: float}."""
    values = {}
    with open(path) as f:
        for line in f:
            if not _TAGGED_LINE.match(line):
                continue
            for key, value in _PAIR.findall(line):
                try:
                    values[key] = float(value)
                except ValueError:
                    continue
    return values


def _agg_discover(output_root, pattern):
    """[(run_name, best.txt path)] for runs matching the glob pattern."""
    root = output_root if os.path.isabs(output_root) \
        else os.path.join(REPO_ROOT, output_root)
    found = []
    for path in sorted(glob.glob(os.path.join(root, "*", "best.txt"))):
        name = os.path.basename(os.path.dirname(path))
        if pattern and not fnmatch.fnmatch(name, pattern):
            continue
        found.append((name, path))
    return found


def group_name(run_name, mode):
    """How runs are pooled. 'all' pools everything; 'tag' pools by trailing tag."""
    if mode == "all":
        return "all runs"
    # outputs are named "<timestamp>_<TAG>"; the tag is what identifies a configuration,
    # and seeds of one configuration differ only by a trailing digit.
    tag = run_name.split("_", 2)[-1] if "_" in run_name else run_name
    if mode == "tag":
        return tag
    return re.sub(r"\d+$", "", tag) or tag          # mode == "config": strip the seed


def mean_std(values):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n == 1:
        return mean, None, 1
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)   # sample std
    return mean, math.sqrt(variance), n


def format_cell(mean, std, n, scale=100.0, digits=2):
    """Short enough to stay inside a table column; the n=1 caveat is printed separately."""
    if mean is None:
        return "n/a"
    if std is None:
        return f"{scale * mean:.{digits}f} (n=1)"
    return f"{scale * mean:.{digits}f}+/-{scale * std:.{digits}f}"


def _agg_mcnemar(a_hits, b_hits):
    """Paired binary comparison. Returns (a_only, b_only, chi2, p)."""
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, float("nan"), float("nan")
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    return a_only, b_only, chi2, math.erfc(math.sqrt(chi2 / 2.0))


def load_scores(run_dir, threshold):
    """Per-annotation hit indicators from scores.p, or None when unavailable."""
    path = os.path.join(run_dir, "scores.p")
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        scores = pickle.load(f)
    ious = scores.get("ious")
    if not ious:
        return None
    return [1 if float(i) >= threshold else 0 for i in ious[0]]


def _add_seed_aggregate_args(parser):
    parser.add_argument("--output-root", dest="output_root", default="outputs")
    parser.add_argument("--pattern", default="*ABL-SEED*",
                        help="Glob over run folder names under outputs/.")
    parser.add_argument("--group-by", dest="group_by", default="all",
                        choices=["all", "tag", "config"],
                        help="'all' pools every match into one mean; 'tag' keeps each "
                             "run folder's tag separate; 'config' strips a trailing "
                             "seed number so ABL-SEED1/2/3 pool together.")
    parser.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="IoU threshold for the paired comparison.")
    parser.add_argument("--paired", action="store_true",
                        help="Also run a paired per-annotation McNemar test between the "
                             "two groups (requires scores.p and exactly two groups).")
    parser.add_argument("--latex", action="store_true", help="Emit a LaTeX table row.")
    parser.add_argument("--out-dir", dest="out_dir",
                        default=os.path.join("outputs", "analysis", "seed_aggregate"),
                        help="Where to write the aggregate.")
    return parser


def run_seed_aggregate(args):

    runs = _agg_discover(args.output_root, args.pattern)
    if not runs:
        print(f"No runs matched {args.pattern!r} under {args.output_root}/")
        print("Finished runs write outputs/<timestamp>_<TAG>/best.txt.")
        print("Launch the seed sweep with: python experiments/ablation/train_arms.py run seeds")
        return 1

    groups = defaultdict(list)
    for name, path in runs:
        values = read_best(path)
        if not values:
            print(f"[warn] {name}: best.txt has no parsable metrics; skipped")
            continue
        groups[group_name(name, args.group_by)].append((name, path, values))

    print(f"{len(runs)} run(s) matched {args.pattern!r}, "
          f"{len(groups)} group(s) by --group-by {args.group_by}\n")

    aggregated = {}
    for group, members in sorted(groups.items()):
        print(f"=== {group}  ({len(members)} run{'s' if len(members) != 1 else ''}) ===")
        width = max(len(name) for name, _p, _v in members)
        header = "  " + "run".ljust(width) + "  " + "  ".join(
            m.rjust(14) for m in args.metrics)
        print(header)
        for name, _path, values in sorted(members):
            cells = "  ".join(
                (f"{100 * values[m]:.2f}" if m in values else "-").rjust(14)
                for m in args.metrics)
            print("  " + name.ljust(width) + "  " + cells)

        stats = {}
        for metric in args.metrics:
            series = [values[metric] for _n, _p, values in members if metric in values]
            mean, std, n = mean_std(series)
            stats[metric] = {"mean": mean, "std": std, "n": n, "values": series}

        print("  " + "-" * (width + 2 + 16 * len(args.metrics)))
        print("  " + "mean +/- std".ljust(width) + "  " + "  ".join(
            format_cell(stats[m]["mean"], stats[m]["std"], stats[m]["n"]).rjust(14)
            for m in args.metrics))

        counts = {stats[m]["n"] for m in args.metrics}
        if counts == {1}:
            print("  NOTE: n=1. No variance can be estimated; run at least two more "
                  "seeds\n        (python experiments/ablation/train_arms.py run seeds).")
        elif counts == {2}:
            print("  NOTE: n=2. A standard deviation from two runs is a very weak "
                  "estimate;\n        three is materially better.")
        print()
        aggregated[group] = {"runs": [n for n, _p, _v in members], "stats": stats}

    latex_rows = []
    for group, data in sorted(aggregated.items()):
        cells = []
        for metric in args.metrics:
            entry = data["stats"][metric]
            if entry["mean"] is None:
                cells.append("--")
            elif entry["std"] is None:
                cells.append(f"{100 * entry['mean']:.2f}")
            else:
                cells.append(f"{100 * entry['mean']:.2f} $\\pm$ "
                             f"{100 * entry['std']:.2f}")
        latex_rows.append(f"  {group} & " + " & ".join(cells) + r" \\")

    if args.latex:
        print("LaTeX rows (values as percentages):")
        for row in latex_rows:
            print(row)
        print()

    paired_result = None
    if args.paired:
        if len(groups) != 2:
            print(f"--paired needs exactly two groups; got {len(groups)}. "
                  f"Use --group-by tag or narrow --pattern.")
        else:
            (name_a, members_a), (name_b, members_b) = sorted(groups.items())
            hits_a = load_scores(os.path.dirname(members_a[0][1]), args.threshold)
            hits_b = load_scores(os.path.dirname(members_b[0][1]), args.threshold)
            if hits_a is None or hits_b is None:
                print("--paired needs scores.p in both runs. Produce it with:")
                print("    python scripts/ScanRefer_eval.py --folder <run> --reference --force "
                      "--use_color --use_normal --lang_num_max 1")
            elif len(hits_a) != len(hits_b):
                print(f"--paired needs equal-length score vectors; got "
                      f"{len(hits_a)} vs {len(hits_b)}.")
            else:
                only_a, only_b, chi2, p = _agg_mcnemar(hits_a, hits_b)
                acc_a = sum(hits_a) / len(hits_a)
                acc_b = sum(hits_b) / len(hits_b)
                paired_result = {
                    "group_a": name_a, "group_b": name_b, "n": len(hits_a),
                    "threshold": args.threshold, "acc_a": acc_a, "acc_b": acc_b,
                    "difference": acc_a - acc_b, "only_a_correct": only_a,
                    "only_b_correct": only_b, "mcnemar_chi2": chi2, "p_value": p,
                    "significant": bool(p == p and p < 0.05),
                }
                print(f"Paired comparison at IoU {args.threshold} over {len(hits_a)} "
                      f"annotations:")
                print(f"  {name_a}: {100 * acc_a:.2f}%   {name_b}: {100 * acc_b:.2f}%   "
                      f"difference {100 * (acc_a - acc_b):+.2f} pp")
                print(f"  only {name_a} correct: {only_a}   only {name_b} correct: "
                      f"{only_b}")
                print(f"  McNemar chi2={chi2:.3f}, p={p:.3e}  -> " +
                      ("the difference is statistically significant"
                       if p == p and p < 0.05 else
                       "NOT significant; do not claim this difference"))

    out_dir = args.out_dir if os.path.isabs(args.out_dir) \
        else os.path.join(REPO_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    report = {
        "pattern": args.pattern,
        "group_by": args.group_by,
        "metrics": list(args.metrics),
        "threshold": args.threshold,
        "num_runs": len(runs),
        "groups": aggregated,
        "latex_rows": latex_rows,
        "paired": paired_result,
    }
    json_path = os.path.join(out_dir, "seed_aggregate.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    lines = ["# Seed aggregate", "",
             f"pattern `{args.pattern}` | {len(runs)} run(s) | grouped by `{args.group_by}`",
             ""]
    for group, data in sorted(aggregated.items()):
        stats = data["stats"]
        n = max((stats[m]["n"] or 0) for m in args.metrics)
        lines += [f"## {group}  (n = {n})", "",
                  "| metric | mean | std | n | values |", "|---|---|---|---|---|"]
        for m in args.metrics:
            e = stats[m]
            mean = f"{100 * e['mean']:.2f}" if e["mean"] is not None else "--"
            std = f"{100 * e['std']:.2f}" if e["std"] is not None else "--"
            vals = ", ".join(f"{100 * v:.2f}" for v in e["values"]) or "--"
            lines.append(f"| `{m}` | {mean} | {std} | {e['n']} | {vals} |")
        lines += ["", "runs: " + ", ".join(f"`{r}`" for r in data["runs"]), ""]
        if n == 1:
            lines += ["> **n=1.** No variance can be estimated from a single run.", ""]
        elif n == 2:
            lines += ["> **n=2.** A standard deviation from two runs is a very weak "
                      "estimate; three seeds is materially better.", ""]
    if latex_rows:
        lines += ["## LaTeX", "", "```", *latex_rows, "```", ""]
    if paired_result:
        pr = paired_result
        lines += ["## Paired comparison", "",
                  f"`{pr['group_a']}` vs `{pr['group_b']}` at IoU {pr['threshold']} "
                  f"over {pr['n']:,} annotations", "",
                  f"- {pr['group_a']}: {100 * pr['acc_a']:.2f}%  ·  "
                  f"{pr['group_b']}: {100 * pr['acc_b']:.2f}%  ·  "
                  f"difference {100 * pr['difference']:+.2f} pp",
                  f"- McNemar chi2 = {pr['mcnemar_chi2']:.3f}, p = {pr['p_value']:.3e} — "
                  + ("significant" if pr["significant"] else
                     "**not** significant; do not claim this difference"), ""]
    md_path = os.path.join(out_dir, "seed_aggregate.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python experiments/analysis.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    for key, doc, adder, runner in (
            ("results-table", _DOC_RESULTS_TABLE, _add_results_table_args, run_results_table),
            ("linguistic-complexity", _DOC_LINGUISTIC_COMPLEXITY, _add_linguistic_complexity_args, run_linguistic_complexity),
            ("parse-quality-split", _DOC_PARSE_QUALITY_SPLIT, _add_parse_quality_split_args, run_parse_quality_split),
            ("failure-cases", _DOC_FAILURE_CASES, _add_failure_cases_args, run_failure_cases),
            ("failure-figures", _DOC_FAILURE_FIGURES, _add_failure_figures_args, run_failure_figures),
            ("failure-scenes", _DOC_FAILURE_SCENES, _add_failure_scenes_args, run_failure_scenes),
            ("annotation-sheet", _DOC_ANNOTATION_SHEET, _add_annotation_sheet_args, run_annotation_sheet),
            ("annotation-sheet-15", _DOC_ANNOTATION_SHEET_15, _add_annotation_sheet_15_args,
             run_annotation_sheet_15),
            ("error-taxonomy", _DOC_ERROR_TAXONOMY, _add_error_taxonomy_args, run_error_taxonomy),
            ("error-propagation", _DOC_ERROR_PROPAGATION, _add_error_propagation_args, run_error_propagation),
            ("parse-field-comparison", _DOC_PARSE_FIELD_COMPARISON, _add_parse_field_comparison_args, run_parse_field_comparison),
            ("complex-sentence-showdown", _DOC_COMPLEX_SENTENCE_SHOWDOWN, _add_complex_sentence_showdown_args, run_complex_sentence_showdown),
            ("seed-aggregate", _DOC_SEED_AGGREGATE, _add_seed_aggregate_args, run_seed_aggregate)):
        p = sub.add_parser(key, help=doc.strip().splitlines()[0], description=doc,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
        adder(p)
        p.set_defaults(func=runner, _parser=p)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
