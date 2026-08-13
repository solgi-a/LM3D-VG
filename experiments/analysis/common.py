
import json
import math
import os
import pickle

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLDS = (0.25, 0.5)


def load_scanrefer(split="val", data_root="data"):
    for candidate in (os.path.join(data_root, f"ScanRefer_filtered_{split}.json"),
                      os.path.join(REPO_ROOT, data_root, f"ScanRefer_filtered_{split}.json")):
        if os.path.isfile(candidate):
            with open(candidate) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"ScanRefer_filtered_{split}.json not found under {data_root!r} "
        f"(cwd={os.getcwd()!r}, repo={REPO_ROOT!r})")


def load_predictions(path):
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
    for base in (parsing_root, os.path.join(REPO_ROOT, parsing_root)):
        path = os.path.join(base, folder, f"tokenized_parsed_result_{split}.json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"parse cache not found: {parsing_root}/{folder}/tokenized_parsed_result_{split}.json")


def parse_for(cache, scene_id, object_id, ann_id):
    try:
        return cache[str(scene_id)][str(object_id)][str(ann_id)]
    except (KeyError, TypeError):
        return None


def join(predictions, records):
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
    a_only = sum(1 for a, b in zip(a_hits, b_hits) if a and not b)
    b_only = sum(1 for a, b in zip(a_hits, b_hits) if b and not a)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, float("nan"), float("nan")
    chi2 = (abs(a_only - b_only) - 1) ** 2 / n
    return a_only, b_only, chi2, math.erfc(math.sqrt(chi2 / 2.0))


def bootstrap_ci(values_fn, n_items, num_resamples=2000, seed=42, alpha=0.05):
    import numpy as _np

    rng = _np.random.default_rng(seed)
    point = values_fn(_np.arange(n_items))
    samples = _np.empty(num_resamples, dtype=float)
    for i in range(num_resamples):
        samples[i] = values_fn(rng.integers(0, n_items, n_items))
    low, high = _np.nanpercentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(low), float(high)


def parse_predictions_arg(value):
    if "=" in value:
        name, path = value.split("=", 1)
        return name.strip(), path.strip()
    parent = os.path.basename(os.path.dirname(os.path.abspath(value)))
    return (parent or "model"), value


def find_default_predictions(output_root="outputs"):
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
    ious = np.asarray(ious, dtype=float)
    if ious.size == 0:
        return 0.0
    return float((ious >= threshold).mean())


def wilson_ci(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def two_proportion_z(k1, n1, k2, n2):
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


_BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0),
              (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))

_PRISM_FACES = ((0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6),
                (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2),
                (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0))


def write_bbox_ply(corners, path, color=(255, 0, 0), radius=0.02):
    corners = np.asarray(corners, dtype=float).reshape(8, 3)
    vertices, faces = [], []

    for start_index, end_index in _BOX_EDGES:
        start, end = corners[start_index], corners[end_index]
        axis = end - start
        length = np.linalg.norm(axis)
        if length < 1e-9:
            continue
        axis = axis / length

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


def box_volume(corners):
    corners = np.asarray(corners, dtype=float).reshape(8, 3)
    extent = corners.max(axis=0) - corners.min(axis=0)
    return float(np.prod(extent))
