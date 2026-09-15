"""
The frozen-detector scene cache: format, the model that consumes it, and the builder.

    python experiments/ablation/cache.py build \
        --use_pretrained 2024-12-18_20-40-38_3DVG-FIXED \
        --splits train val --use_color --use_normal

The detection branch never sees the description, so its output is computed once per scene
and reused by every description of it. ``build`` chains into the validator when it
finishes; --no_validate skips that. Protocol: details.md §3.
"""

import argparse
import gc
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from data.scannet.model_util_scannet import ScannetDatasetConfig  # noqa: E402
from experiments.ablation.ablation_config import ABLATION  # noqa: E402
from lib.config import CONF  # noqa: E402
from lib.dataset import GLOVE_PICKLE, SCANNET_V2_TSV, ScannetReferenceDataset  # noqa: E402
from models.lang_module import LangModule  # noqa: E402
from models.match_module import MatchModule  # noqa: E402
from models.refnet import RefNet  # noqa: E402

DC = ScannetDatasetConfig()


SCENE_CACHE_KEYS_REQUIRED = [
    "detr_features",                  # (K, 288)      -> MatchModule.features_concat
    "center",                         # (K, 3)
    "objectness_scores",              # (K, 2)
    "sem_cls_scores",                 # (K, num_class)
    "heading_scores",                 # (K, NH)
    "heading_residuals",              # (K, NH)
    "heading_residuals_normalized",   # (K, NH)
    "size_scores",                    # (K, NS)
    "size_residuals",                 # (K, NS, 3)
    "size_residuals_normalized",      # (K, NS, 3)
    "pred_obbs",                      # (K, 7)
    "pred_bboxes",                    # (K, 8, 3)
    "aggregated_vote_xyz",            # (K, 3)        -> compute_objectness_loss
    "aggregated_vote_features",       # (K, 128)
    "aggregated_vote_inds",           # (K,)
    "seed_xyz",                       # (num_seed, 3) -> compute_vote_loss
    "seed_inds",                      # (num_seed,)   -> compute_vote_loss
    "vote_xyz",                       # (num_seed*vote_factor, 3)
]

#: Nothing downstream reads these, and they roughly quadruple the cache size. Opt-in via
#: --include_optional, for debugging the cache itself.
SCENE_CACHE_KEYS_OPTIONAL = [
    "seed_features",
    "vote_features",
    "fp2_xyz",
    "fp2_features",
    "fp2_inds",
]

#: fp16 halves the cache size and these are activations, where the round-trip error sits
#: below the model's noise floor. Geometry and logits stay fp32 -- they feed IoU
#: thresholding and argmax, where the equivalence check needs exactness.
SCENE_CACHE_FP16_KEYS = {
    "detr_features",
    "aggregated_vote_features",
    "seed_features",
    "vote_features",
    "fp2_features",
}

CACHE_FORMAT_VERSION = 1



def scene_subsample_seed(scene_id, salt="v1"):
    """Stable 32-bit seed from the scene id. md5, not hash(), which is salted per process."""
    digest = hashlib.md5(f"{salt}:{scene_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)



def cache_dir_for(root, split):
    return os.path.join(root, split)


def cache_path_for(root, split, scene_id):
    return os.path.join(cache_dir_for(root, split), f"{scene_id}.p")


def meta_path_for(root):
    return os.path.join(root, "meta.json")



def save_scene_cache(path, tensors):
    """Persist one scene's detector output.

    ``tensors`` maps cache key -> torch tensor with the batch dimension already stripped
    (i.e. shape (K, ...), not (1, K, ...)).
    """
    import torch

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {}
    for key, value in tensors.items():
        value = value.detach().cpu()
        if key in SCENE_CACHE_FP16_KEYS and value.dtype == torch.float32:
            value = value.half()
        payload[key] = value
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_scene_cache(path, keys=None, dtype=None):
    """Load one scene's cached tensors, restoring fp16 entries to ``dtype``."""
    import torch

    if dtype is None:
        dtype = torch.float32
    payload = torch.load(path, map_location="cpu", weights_only=False)
    out = {}
    for key, value in payload.items():
        if keys is not None and key not in keys:
            continue
        if value.dtype == torch.float16:
            value = value.to(dtype)
        out[key] = value
    return out



def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        ).decode().strip()
    except Exception:
        return "unknown"


def write_meta(root, **fields):
    meta = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
    }
    meta.update(fields)
    os.makedirs(root, exist_ok=True)
    with open(meta_path_for(root), "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return meta


def read_meta(root):
    path = meta_path_for(root)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No meta.json at {path}. Generate the cache with\n"
            f"    python experiments/ablation/cache.py build --use_pretrained <folder> --splits train val\n"
            f"before enabling --use_cached_scenes."
        )
    with open(path) as f:
        return json.load(f)


def assert_meta_compatible(root, num_points, subsample_salt, detector, num_proposals=None):
    """Fail loudly when the training config does not match how the cache was built."""
    meta = read_meta(root)
    problems = []
    if meta.get("num_points") != num_points:
        problems.append(f"num_points: cache={meta.get('num_points')} run={num_points}")
    if meta.get("subsample_salt") != subsample_salt:
        problems.append(f"subsample_salt: cache={meta.get('subsample_salt')} run={subsample_salt}")
    if meta.get("detector") != detector:
        problems.append(f"detector: cache={meta.get('detector')} run={detector}")
    if num_proposals is not None and meta.get("num_proposals") != num_proposals:
        problems.append(f"num_proposals: cache={meta.get('num_proposals')} run={num_proposals}")
    if problems:
        raise RuntimeError(
            "Cached scenes were generated with a different configuration:\n  "
            + "\n  ".join(problems)
            + f"\nRegenerate the cache at {root} or fix the run configuration."
        )
    return meta



class _LazySceneData(dict):
    """A scene's npy arrays, loaded on first access, last few kept resident.

    Presents the mapping the base __getitem__ indexes, so lib/dataset.py works verbatim.
    """

    def __init__(self, scene_list, maxsize=4):
        super().__init__()
        self.scene_list = list(scene_list)
        self.maxsize = max(1, int(maxsize))
        self._cache = OrderedDict()

    def _load(self, scene_id):
        base = os.path.join(CONF.PATH.SCANNET_DATA, scene_id)
        return {
            "mesh_vertices": np.load(base + "_aligned_vert.npy"),
            "instance_labels": np.load(base + "_ins_label.npy"),
            "semantic_labels": np.load(base + "_sem_label.npy"),
            "instance_bboxes": np.load(base + "_aligned_bbox.npy"),
        }

    def __getitem__(self, scene_id):
        entry = self._cache.get(scene_id)
        if entry is None:
            entry = self._load(scene_id)
            self._cache[scene_id] = entry
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(scene_id)
        # The base class normalises colours in place, so hand back a fresh copy or the
        # second visit to a scene sees them normalised twice.
        return dict(entry, mesh_vertices=entry["mesh_vertices"].copy())

    def __contains__(self, scene_id):
        return scene_id in self.scene_list

    def __len__(self):
        return len(self.scene_list)

    def keys(self):
        return list(self.scene_list)


class _LazyLangData:
    """Builds one annotation's GloVe embeddings on access instead of all up front.

    The eager path holds ~823 KB per annotation for the dataset's lifetime, which is ~29 GB
    on train. The embeddings are a pure function of the tokens and the GloVe table, so this
    rebuilds them per annotation behind a small LRU instead. Numbers: details.md §5.
    """

    __slots__ = ("_owner", "_kind", "_cache", "_maxsize")

    def __init__(self, owner, kind, maxsize):
        self._owner = owner
        self._kind = kind            # "lang" | "lang_main" | "parsed"
        self._cache = OrderedDict()
        self._maxsize = max(1, int(maxsize))

    def __getitem__(self, scene_id):
        return _LazyLangLevel(self, (scene_id,))

    def _entry(self, key):
        """key = (scene_id, object_id, ann_id) -> the built record, LRU-cached."""
        record = self._cache.get(key)
        if record is None:
            record = self._owner._build_lang_entry(*key)
            self._cache[key] = record
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(key)
        return record[self._kind]

    def __contains__(self, scene_id):
        return scene_id in self._owner.lang_scene_ids

    def keys(self):
        return list(self._owner.lang_scene_ids)

    def __len__(self):
        return len(self._owner.lang_scene_ids)


class _LazyLangLevel:
    """One partially-applied level of the ``[scene][object][ann]`` chain."""

    __slots__ = ("_parent", "_key")

    def __init__(self, parent, key):
        self._parent = parent
        self._key = key

    def __getitem__(self, part):
        key = self._key + (str(part),)
        if len(key) == 3:
            return self._parent._entry(key)
        return _LazyLangLevel(self._parent, key)


class CachedSceneDataset(ScannetReferenceDataset):
    """ScannetReferenceDataset + deterministic subsampling + cached detector output.

    Subclasses rather than edits, so lib/dataset.py keeps its original content. The two
    behaviours switch independently: ``deterministic`` pins the 40k-point subsample per
    scene, ``use_cache`` splices in that scene's cached detector tensors.
    """

    def __init__(self, *args, use_cache=False, cached_scenes_root=None,
                 deterministic=False, subsample_salt="v1", lazy_scene_data=True,
                 lazy_maxsize=4, lazy_lang_data=True, **kwargs):
        self.use_cache = use_cache
        self.cached_scenes_root = cached_scenes_root
        self.subsample_salt = subsample_salt
        self.lazy_lang_data = lazy_lang_data
        # Reading the cache only makes sense against the un-augmented, fixed subsample it
        # was generated from.
        self.deterministic = deterministic or use_cache
        self.lazy_scene_data = lazy_scene_data
        self.lazy_maxsize = lazy_maxsize

        if self.use_cache:
            if not cached_scenes_root:
                raise ValueError("use_cache=True requires cached_scenes_root")
            if kwargs.get("augment", False):
                raise ValueError(
                    "Cached scenes are incompatible with augment=True: the cache is built "
                    "from un-augmented point clouds, so augmented GT boxes would not "
                    "correspond to the cached proposals. Train cached ablations with "
                    "augment=False."
                )

        super().__init__(*args, **kwargs)

        if self.use_cache:
            assert_meta_compatible(
                cached_scenes_root,
                num_points=self.num_points,
                subsample_salt=subsample_salt,
                detector=getattr(self.args, "detector", None),
            )

    def _load_data(self):
        """ScannetReferenceDataset._load_data with the eager scene preload made lazy.

        Every other step is reproduced verbatim; set lazy_scene_data=False to fall back.
        """
        if not self.lazy_scene_data:
            return super()._load_data()

        # Mirrors ScannetReferenceDataset._load_data, except that scene arrays are
        # materialised on demand instead of up front.
        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        self.scene_data = _LazySceneData(self.scene_list, maxsize=self.lazy_maxsize)

        # prepare class mapping
        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

        # load language features
        if self.lazy_lang_data:
            self.lang, self.lang_main = self._lazy_tranform_des()
        else:
            self.lang, self.lang_main = self._tranform_des()


    def _lazy_tranform_des(self):
        """``_tranform_des`` without materialising every annotation's embeddings.

        Returns the same lang / lang_main mappings; parsed_sentence and tokenized_parsed
        are set as a side effect, as in the base class.
        """
        with open(GLOVE_PICKLE, "rb") as f:
            self.glove = pickle.load(f)

        folder = ABLATION.PARSING_FOLDER
        self.tokenized_parsed = json.load(open(os.path.join(
            CONF.PATH.PARSING, f"{folder}/tokenized_parsed_result_{self.split}.json")))

        # The annotation index: everything _tranform_des reads off `data`, nothing more.
        self._lang_index = {}
        scene_ids = []
        for data in self.scanrefer:
            key = (data["scene_id"], str(data["object_id"]), str(data["ann_id"]))
            if key not in self._lang_index:
                self._lang_index[key] = (data["token"], data["object_name"])
                if not scene_ids or scene_ids[-1] != data["scene_id"]:
                    scene_ids.append(data["scene_id"])
        self.lang_scene_ids = sorted(set(scene_ids))

        # One LRU shared by all three views, so a single build serves all of them.
        # Sized to hold a whole batch of scenes' worth of annotations.
        maxsize = max(self.lang_num_max * 4, 64)
        lang = _LazyLangData(self, "lang", maxsize)
        lang_main = _LazyLangData(self, "lang_main", maxsize)
        self.parsed_sentence = _LazyLangData(self, "parsed", maxsize)
        # The three views must share one cache, or each build happens three times.
        lang_main._cache = self.parsed_sentence._cache = lang._cache
        return lang, lang_main

    def _build_lang_entry(self, scene_id, object_id, ann_id):
        """Reproduce one iteration of ``_tranform_des``'s loop body, verbatim."""
        try:
            tokens, object_name = self._lang_index[(scene_id, object_id, ann_id)]
        except KeyError:
            raise KeyError(
                f"no annotation {(scene_id, object_id, ann_id)} in split {self.split!r}"
            ) from None

        embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
        main_embeddings = np.zeros((CONF.TRAIN.MAX_DES_LEN, 300))
        main_len, first_obj, pd = 0, -1, 1

        main_object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
        for token_id in range(CONF.TRAIN.MAX_DES_LEN):
            if token_id < len(tokens):
                token = tokens[token_id]
                embeddings[token_id] = self.glove.get(token, self.glove["pad"])
                if pd == 1:
                    main_embeddings[token_id] = self.glove.get(token, self.glove["unk"])
                    if token == ".":
                        pd = 0
                        main_len = token_id + 1
                object_cat = self.raw2label[token] if token in self.raw2label else -1
                is_two_words = 0
                if token_id + 1 < len(tokens):
                    token_new = token + " " + tokens[token_id + 1]
                    object_cat_new = self.raw2label.get(token_new, -1)
                    if object_cat_new != -1:
                        object_cat = object_cat_new
                        is_two_words = 1
                if first_obj == -1 and object_cat == main_object_cat:
                    first_obj = (token_id + 1 if is_two_words == 1
                                 and token_id + 1 < len(tokens) else token_id)
        if pd == 1:
            main_len = len(tokens)

        parsed = {}
        if self.args.lang_input == 'glove+parse' and self.args.detection == False:
            entry = self.tokenized_parsed[scene_id][object_id][ann_id]
            for field, num_token in (("target", 7), ("adjectives", 17), ("neighbors", 75)):
                parsed[field] = self._transform_parsed(entry[field], num_token, 300)
        elif self.args.detection == True:
            for field in ("target", "adjectives", "neighbors"):
                parsed[field] = np.array([self.glove["unk"]])

        return {
            "lang": embeddings,
            "lang_main": {"main": main_embeddings, "len": main_len,
                          "first_obj": first_obj, "unk": self.glove["unk"]},
            "parsed": parsed,
        }

    def _scene_id_for(self, idx):
        return self.scanrefer_new[idx][0]["scene_id"]

    def __getitem__(self, idx):
        scene_id = self._scene_id_for(idx)

        if self.deterministic:
            # Pin the global RNG random_sampling draws from, then restore it so no fixed
            # stream leaks into anything else in this worker.
            state = np.random.get_state()
            np.random.seed(scene_subsample_seed(scene_id, self.subsample_salt))
            try:
                data_dict = super().__getitem__(idx)
            finally:
                np.random.set_state(state)
        else:
            data_dict = super().__getitem__(idx)

        if self.use_cache:
            data_dict.update(self._load_cached_scene(scene_id))

        return data_dict

    def _load_cached_scene(self, scene_id):
        path = cache_path_for(self.cached_scenes_root, self.split, scene_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No cached scene for {scene_id} (split={self.split}) at {path}.\n"
                f"Build it first:  python experiments/ablation/cache.py build --splits {self.split} "
                f"--use_pretrained <folder>"
            )
        return load_scene_cache(path, keys=set(SCENE_CACHE_KEYS_REQUIRED))


class CachedRefNet(nn.Module):
    """Language + fusion only. Signature matches RefNet so callers can swap them."""

    def __init__(self, args, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, num_proposal=128, vote_factor=1, sampling="vote_fps",
                 use_lang_classifier=True, use_bidir=False, no_reference=False,
                 emb_size=300, hidden_size=256, dataset_config=None):
        super().__init__()

        if no_reference:
            raise ValueError(
                "CachedRefNet trains the language and fusion branches; no_reference=True "
                "leaves it with nothing to train. Use RefNet for detection-only runs."
            )

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.use_lang_classifier = use_lang_classifier
        self.use_bidir = use_bidir
        self.no_reference = no_reference
        self.dataset_config = dataset_config
        self.args = args

        self.lang = LangModule(num_class, use_lang_classifier, use_bidir, emb_size)
        # det_channel=288 is the width of the cached detr_features.
        self.match = MatchModule(args, num_proposals=num_proposal, det_channel=288)

    def forward(self, data_dict):
        """RefNet.forward without the detection branch; data_dict carries the cache."""
        data_dict = self.lang(data_dict)
        data_dict = self.match(data_dict)

        return data_dict


def _load_scanrefer(split):
    path = os.path.join(CONF.PATH.DATA, f"ScanRefer_filtered_{split}.json")
    with open(path) as f:
        return json.load(f)


def _one_chunk_per_scene(scanrefer, num_scenes=-1):
    """Build a scanrefer_new whose every entry is a distinct scene.

    ScannetReferenceDataset indexes chunks of up to lang_num_max descriptions that all
    share a scene, and loads the point cloud by that scene alone. For caching we want
    each scene visited exactly once, so we hand it one single-description chunk per
    scene: the language fields are irrelevant here, only the point cloud is.
    """
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
        # Resume: only build the dataset over scenes that are not cached yet, so a
        # restart does not re-run the detector on work already on disk.
        todo = set(todo_scene_ids)
        chunks = [c for c in chunks if c[0]["scene_id"] in todo]

    # Keep only the one description per scene `chunks` references. Not an optimisation:
    # _tranform_des costs ~22 GB over the train split, and the detection branch never
    # touches those tensors.
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
        augment=False,               # the cache is augmentation-free by construction
        shuffle=False,
        use_cache=False,             # this script *creates* the cache, it does not read it
        deterministic=True,          # ...and the subsample must be reproducible
        subsample_salt=args.subsample_salt,
        lazy_scene_data=True,
        lazy_maxsize=args.lazy_maxsize,
    )
    return dataset, chunks


def _pending_scenes(args, split):
    """(all_scene_ids, not_yet_cached) for this split, honouring --num_scenes."""
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
        # Returns right after the detection branch, which is all the cache stores -- and
        # it avoids LangModule's hardcoded .cuda() breaking the CPU path.
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

    # comp_weight() from ScanRefer_train.py, but reporting what was skipped: a silent
    # shape mismatch here would fill the cache with untrained-network output.
    state = model.state_dict()
    loaded, skipped = [], []
    for key, value in weights.items():
        if key in state and state[key].shape == value.shape:
            state[key] = value
            loaded.append(key)
        else:
            skipped.append(key)
    model.load_state_dict(state)

    # Keyed on the model's own keys, so it asserts every detection weight this model has
    # was loaded; the skipped entries are the checkpoint's lang.*/match.*, as expected.
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
    """Verify everything the real run needs, without a single forward pass.

    Checks, in the order they would otherwise fail:
      1. the checkpoint exists and every detection-branch weight loads with matching shape
      2. the parse cache and scene .npy files the dataloader will ask for are present
      3. one sample can actually be built end to end by the dataset
      4. how many scenes will be written, and roughly how much disk that needs

    Cheap enough to run on a CPU-only machine in well under a minute: it constructs the
    model on CPU (no CUDA op is touched) and builds exactly one dataset sample per split.
    """
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
    # 18 required tensors at num_proposals=256, two of them fp16; ~250 KB/scene measured
    # against the tensor shapes in experiments/ablation/cache.py SCENE_CACHE_KEYS_REQUIRED.
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
    # unit="scene" with total=len(todo) so the bar and ETA count scenes, not batches --
    # with batch_size=1 on CPU a batch-based bar would be misleadingly granular.
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

        # Release the whole graph-free forward output before the next scene. On CPU with
        # batch_size=1 this is what keeps RSS flat instead of creeping over four hours.
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


def _add_build_args(parser):
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
                        help="Skip the automatic cache validation at the end.")
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

    # model / data configuration -- must match the training run that will read the cache
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
    # Accepted for signature compatibility with ScanRefer_train.py, but ignored: this
    # script always builds RefNet with no_reference=True (detection branch only).
    parser.add_argument("--no_reference", action="store_true",
                        help=argparse.SUPPRESS)
    parser.add_argument("--lang_input", type=str, default="glove+parse")
    parser.add_argument("--GF_path", type=str, default=None)
    return parser


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python experiments/ablation/cache.py",
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    build = sub.add_parser("build", help="build the scene cache",
                           description=__doc__,
                           formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_build_args(build)

    args = parser.parse_args(argv)

    # ScannetReferenceDataset reads args.detection to decide whether to build the parsed
    # language tensors. Caching never touches the language branch, so skip that work.
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

    # Defaults chosen per device: on CPU, minimise memory and keep the machine usable.
    if args.batch_size is None:
        args.batch_size = 1 if on_cpu else 8
    if args.num_workers is None:
        # A DataLoader worker forks the whole dataset (glove dict included), so on CPU
        # in-process loading uses far less RAM than any worker would.
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
            f"               experiments/measurements.py validate-scene-cache on a GPU before\n"
            f"               these numbers for the paper.\n"
            f"  This is expected to take hours. It is safe to Ctrl-C and restart:\n"
            f"  each scene is written atomically and completed scenes are skipped.\n"
        )
        # The end-to-end validation leg cannot run here; do not pretend otherwise.
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
        # meta.json is the contract the training path checks, so it must describe the
        # cache as it is on disk now -- not just what this invocation happened to write.
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
            os.path.join(REPO_ROOT, "experiments", "measurements.py"),
            "validate-scene-cache",
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
