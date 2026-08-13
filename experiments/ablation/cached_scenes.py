
import hashlib
import json
import os
import pickle
import subprocess
from collections import OrderedDict
from datetime import datetime

import numpy as np

from experiments.ablation.ablation_config import ABLATION
from lib.config import CONF
from lib.dataset import GLOVE_PICKLE, SCANNET_V2_TSV, ScannetReferenceDataset


SCENE_CACHE_KEYS_REQUIRED = [
    "detr_features",
    "center",
    "objectness_scores",
    "sem_cls_scores",
    "heading_scores",
    "heading_residuals",
    "heading_residuals_normalized",
    "size_scores",
    "size_residuals",
    "size_residuals_normalized",
    "pred_obbs",
    "pred_bboxes",
    "aggregated_vote_xyz",
    "aggregated_vote_features",
    "aggregated_vote_inds",
    "seed_xyz",
    "seed_inds",
    "vote_xyz",
]

SCENE_CACHE_KEYS_OPTIONAL = [
    "seed_features",
    "vote_features",
    "fp2_xyz",
    "fp2_features",
    "fp2_inds",
]

SCENE_CACHE_FP16_KEYS = {
    "detr_features",
    "aggregated_vote_features",
    "seed_features",
    "vote_features",
    "fp2_features",
}

CACHE_FORMAT_VERSION = 1


def scene_subsample_seed(scene_id, salt="v1"):
    digest = hashlib.md5(f"{salt}:{scene_id}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def cache_dir_for(root, split):
    return os.path.join(root, split)


def cache_path_for(root, split, scene_id):
    return os.path.join(cache_dir_for(root, split), f"{scene_id}.p")


def meta_path_for(root):
    return os.path.join(root, "meta.json")


def save_scene_cache(path, tensors):
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
            f"    python experiments/ablation/scenes_cache.py --use_pretrained <folder> --splits train val\n"
            f"before enabling --use_cached_scenes."
        )
    with open(path) as f:
        return json.load(f)


def assert_meta_compatible(root, num_points, subsample_salt, detector, num_proposals=None):
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
        return dict(entry, mesh_vertices=entry["mesh_vertices"].copy())

    def __contains__(self, scene_id):
        return scene_id in self.scene_list

    def __len__(self):
        return len(self.scene_list)

    def keys(self):
        return list(self.scene_list)


class _LazyLangData:

    __slots__ = ("_owner", "_kind", "_cache", "_maxsize")

    def __init__(self, owner, kind, maxsize):
        self._owner = owner
        self._kind = kind
        self._cache = OrderedDict()
        self._maxsize = max(1, int(maxsize))

    def __getitem__(self, scene_id):
        return _LazyLangLevel(self, (scene_id,))

    def _entry(self, key):
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

    def __init__(self, *args, use_cache=False, cached_scenes_root=None,
                 deterministic=False, subsample_salt="v1", lazy_scene_data=True,
                 lazy_maxsize=4, lazy_lang_data=True, **kwargs):
        self.use_cache = use_cache
        self.cached_scenes_root = cached_scenes_root
        self.subsample_salt = subsample_salt
        self.lazy_lang_data = lazy_lang_data
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
        if not self.lazy_scene_data:
            return super()._load_data()

        self.scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))

        self.scene_data = _LazySceneData(self.scene_list, maxsize=self.lazy_maxsize)

        lines = [line.rstrip() for line in open(SCANNET_V2_TSV)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

        if self.lazy_lang_data:
            self.lang, self.lang_main = self._lazy_tranform_des()
        else:
            self.lang, self.lang_main = self._tranform_des()


    def _lazy_tranform_des(self):
        with open(GLOVE_PICKLE, "rb") as f:
            self.glove = pickle.load(f)

        folder = ABLATION.PARSING_FOLDER
        self.tokenized_parsed = json.load(open(os.path.join(
            CONF.PATH.PARSING, f"{folder}/tokenized_parsed_result_{self.split}.json")))

        self._lang_index = {}
        scene_ids = []
        for data in self.scanrefer:
            key = (data["scene_id"], str(data["object_id"]), str(data["ann_id"]))
            if key not in self._lang_index:
                self._lang_index[key] = (data["token"], data["object_name"])
                if not scene_ids or scene_ids[-1] != data["scene_id"]:
                    scene_ids.append(data["scene_id"])
        self.lang_scene_ids = sorted(set(scene_ids))

        maxsize = max(self.lang_num_max * 4, 64)
        lang = _LazyLangData(self, "lang", maxsize)
        lang_main = _LazyLangData(self, "lang_main", maxsize)
        self.parsed_sentence = _LazyLangData(self, "parsed", maxsize)
        lang_main._cache = self.parsed_sentence._cache = lang._cache
        return lang, lang_main

    def _build_lang_entry(self, scene_id, object_id, ann_id):
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
                f"Build it first:  python experiments/ablation/scenes_cache.py --splits {self.split} "
                f"--use_pretrained <folder>"
            )
        return load_scene_cache(path, keys=set(SCENE_CACHE_KEYS_REQUIRED))
