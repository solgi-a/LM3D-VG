# Enhancing 3D Visual Grounding through Semantic-Guided Attention and Graph Networks with LLM-Based Sentence Parsing

The task is 3D visual grounding: given a 3D point cloud of an indoor scene and a
free-form natural-language description, localise the referred object with a 3D bounding
box. The method parses each description into three semantic fields with a language
model, then fuses them with detected object proposals through two complementary
attention paths and a spatial graph network.

Evaluated on **ScanRefer** (ScanNet v2 scenes) and **ReferIt3D / Sr3D**.

---

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Sentence parsing](#sentence-parsing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualisation](#visualisation)
- [Ablations](#ablations)
- [Analysis](#analysis)
- [Computational complexity](#computational-complexity)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

---

## Architecture

```
point cloud (N x 10)                 description ("the brown chair next to the table")
      |                                            |
      v                                            v
 PointNet++  (Pointnet2Backbone)            LLM sentence parser  [OFFLINE]
      |                                            |
      v                                     {target, adjectives, neighbors}
 Hough voting (VotingModule)                       |
      |                                            v
      v                                    GloVe + GRU (LangModule)
 DETR decoder, 2 layers (ProposalModule)           |
      |                                            |
      +---------> object proposals ----------------+
                  (256 x 288)                      |
                                                   v
                                     MatchModule:  LGA -> A2F + TAF
                                                   |
                                                   v
                                         referred-object scores
```

| Module | File | Role |
|---|---|---|
| `Pointnet2Backbone` | `models/backbone_module.py` | point-cloud feature extraction |
| `VotingModule` | `models/voting_module.py` | Hough voting towards object centres |
| `ProposalModule` | `models/proposal_module.py` | DETR decoder → 256 proposals, 288-d features |
| `LangModule` | `models/lang_module.py` | GloVe embedding + GRU over the parsed fields |
| `MatchModule` | `models/match_module.py` | LGA (language-guided graph attention), A2F (adjacency-to-feature), TAF (target-attribute fusion) |
| `RefNet` | `models/refnet.py` | assembles all of the above |

The three parsed fields are the interface between language and fusion:

| Field | Meaning | Token cap |
|---|---|---|
| `target` | the referred object | 7 |
| `adjectives` | its attributes as written | 17 |
| `neighbors` | spatial-relation phrases to nearby objects | 75 |

Each is a surface phrase, tokenised and embedded with GloVe. An absent field carries the
literal string `"not mentioned"`. The caps are load-bearing: `lib/dataset.py:173-175`
passes the *unclipped* token count to `pack_padded_sequence`, so a longer list raises
mid-training and an empty one raises immediately.

**Parsing is offline.** `lib/dataset.py` loads a precomputed
`tokenized_parsed_result_{split}.json` once at dataset construction. No parser is invoked
anywhere in the forward path, so parsing adds nothing to per-query inference latency.

---

## Repository layout

```
.
├── run_analysis_colab.ipynb          runs every experiment, in order
├── run_one_experiment.ipynb          runs ONE experiment, with a confirmation gate
├── run_parser_analysis.ipynb         the two parser-field analyses, inline figures
├── run_smalllm_parsing.ipynb         build a small-LM parse cache, one per model
│
├── scripts/                    entry points
│   ├── ScanRefer_train.py         training
│   ├── ScanRefer_eval.py          evaluation (writes predictions.p / scores.p)
│   ├── predict.py                 inference on new descriptions
│   ├── visualize.py               render a scene to .ply
│   ├── utils/                     AdamW, script_utils
│   └── multiview_features/        ENet multiview feature extraction/projection
│
├── models/                     the network
├── lib/                        dataset, solver, losses, config, eval helpers
├── utils/                      point-cloud / box / NMS utilities
├── data/                       ScanRefer JSON, GloVe, ScanNet (see Data preparation)
├── data_parsing/               parse caches, one folder per parser variant
├── cached_scenes/              frozen-detector output (built by experiments/ablation/scenes_cache.py)
├── outputs/                    runs, checkpoints, predictions, reports
│
├── experiments/                ablations, analyses, diagnostics
│   ├── experiments.md             the index for this tree — start there
│   │
│   ├── ablation/               experiments that rebuild a cache or retrain
│   │   ├── ablation_config.py     the switchboard — every flag lives here
│   │   ├── ablation_hooks.py      dataset/model selection; returns originals when off
│   │   ├── cached_scenes.py       cache format + CachedSceneDataset
│   │   ├── cached_refnet.py       CachedRefNet (language + fusion, no detector)
│   │   ├── scenes_cache.py        [entry] build the scene cache            GPU
│   │   ├── parsers/               everything that writes a parse cache
│   │   │   ├── spacy_parser.py        the rule-based parser itself
│   │   │   ├── tokenize_parse.py      shared tokenizer + the 7/17/75 caps
│   │   │   ├── run_spacy_parser.py    [entry] variant B parse              CPU
│   │   │   ├── run_smalllm_parser.py  [entry] variant E parse (small LM)   GPU
│   │   │   ├── parse_with_smalllm.py  [entry] variant E, one folder per model GPU
│   │   │   ├── make_noparse_cache.py  [entry] variant D parse (empty)      CPU
│   │   │   ├── corrupt_parse_cache.py [entry] deliberately corrupt a parse CPU
│   │   │   ├── clip_parse_cache.py    [entry] enforce the token caps       CPU
│   │   │   └── eval_parser_target_accuracy.py [entry] score a parser       CPU
│   │   └── runners/               one self-contained runner per training run
│   │
│   ├── analysis/               post-hoc — ALL CPU, no model needed
│   │   ├── common.py              shared loaders, statistics, PLY writer
│   │   ├── results_table.py       main table: Overall/Unique/Multiple
│   │   ├── linguistic_complexity.py   accuracy by linguistic complexity
│   │   ├── parse_quality_split.py grounding accuracy vs parse correctness
│   │   ├── parse_error_propagation.py accuracy under parse corruption
│   │   ├── parse_field_comparison.py  adjectives/neighbors across parsers
│   │   ├── complex_sentence_showdown.py parsers on the hardest sentences
│   │   ├── failure_cases.py       failure selection + cause attribution
│   │   ├── render_failure_figures.py  the qualitative figure
│   │   ├── annotation_sheet.py    export 200 samples for manual labelling
│   │   ├── error_taxonomy.py      tally the filled sheets
│   │   └── figures/               rendered qualitative panels
│   │
│   ├── complexity/             FLOPs, GPU memory, latency, parsing cost
│   │
│   └── diagnostics/            implementation checks
│       ├── validate_scene_cache.py      cached == end-to-end, to 1e-4     GPU
│       ├── audit_scene_cache.py         model-free cache audit            CPU
│       └── cached_eval_cpu.py           cached-path accuracy, no CUDA     CPU
│
└── pointnet/                   the three PointNet++ implementations
    ├── __init__.py                the resolution chain (1 -> 2 -> 3 below)
    ├── pointnet2_ops_lib/      1. C++/CUDA extension, fastest
    ├── pointnet2_py_adv/       2. optimised pure PyTorch (pn_py_torch.py)
    └── pointnet2_python/       3. original pure-Python reference
```

Everything under `experiments/` is self-contained. The core files carry ~28 changed lines
across 8 files, each tagged `# ABLATION`. With every flag off,
`ablation_hooks.build_dataset` returns `ScannetReferenceDataset` and `build_model` returns
`RefNet`.

---

## Installation

```bash
conda create -n 3dvg python=3.10 -y && conda activate 3dvg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r req/requirements.txt

# REQUIRED: torch_geometric.nn.knn_graph inside MatchModule needs pyg-lib. Without it the
# model cannot run at all. pyg-lib is not on PyPI -- it lives only on the PyG wheel index,
# keyed to the exact installed torch build, so resolve the version at run time.
pip install torch-geometric
TORCH=$(python -c "import torch; print(torch.__version__)")
pip install pyg_lib -f "https://data.pyg.org/whl/torch-${TORCH}.html"

python -m spacy download en_core_web_sm     # parser variant B and the depth metric
```

Optional:

```bash
pip install 'transformers>=4.40' accelerate   # parser variant E (Qwen2-0.5B)
pip install openai                            # GPT-4o-mini latency measurement
```

**PointNet++ ops.** `pointnet/__init__.py` resolves the three implementations as a chain,
trying each tier only when the one above cannot be imported:

| # | Tier | Where | Notes |
|---|---|---|---|
| 1 | `pointnet2_ops` | `pointnet/pointnet2_ops_lib/` | C++/CUDA extension, fastest; needs a matching toolchain |
| 2 | `pointnet2_py_adv` | `pointnet/pointnet2_py_adv/pn_py_torch.py` | optimised pure PyTorch: top-k ball query instead of a full sort, channel-first gathers, no redundant transposes |
| 3 | `pointnet2_python` | `pointnet/pointnet2_python/` | the original pure-Python reference |

Outputs are equivalent throughout: ball-query / gather / FPS indices are bitwise identical
for the same seed and features are `allclose`; only the ordering of exactly-equidistant
neighbours is unspecified.

Build the CUDA extension with `pip install ./pointnet/pointnet2_ops_lib`.

`ABLATION.POINTNET_IMPL` pins one tier by name for an A/B comparison; `None` (the default)
walks the chain. A pinned tier that cannot be imported raises rather than downgrading.

---

## Data preparation

```
data/
├── ScanRefer_filtered_train.json      36,665 annotations
├── ScanRefer_filtered_val.json         9,508 annotations
├── ScanRefer_filtered_test.json
├── glove.p                             400,003-word GloVe dictionary
└── scannet/
    ├── scannet_data/                   preprocessed scenes (_vert.npy, _bbox.npy, ...)
    ├── meta_data/                      scannetv2 splits + label mapping
    └── scans/                          raw ScanNet (only for visualisation)
```

1. Download ScanRefer from the [official
   repository](https://github.com/daveredrum/ScanRefer) into `data/`.
2. Download ScanNet v2 and place the scans under `data/scannet/scans/`.
3. Preprocess: `cd data/scannet && python batch_load_scannet_data.py`.

A ScanRefer record — every script joins on `scene_id`, `object_id`, `ann_id`:

```json
{
  "scene_id": "scene0011_00", "object_id": "5", "ann_id": "3",
  "object_name": "chair",
  "description": "there is a dark brown wooden and leather chair. placed in the table of the kitchen.",
  "token": ["there", "is", "a", "dark", "brown", ...]
}
```

`object_name` is the annotated class of the referred object, which makes it usable as
ground truth for the parsed `target` field across the whole dataset.

---

## Sentence parsing

Each parse cache is a folder under `data_parsing/` containing

```
tokenized_parsed_result_{train,val,test}.json
```

keyed `scene_id → object_id → ann_id → {target, adjectives, neighbors}`, each a token
list.

| Variant | Parser | Folder | Build it with |
|---|---|---|---|
| A | GPT-4o-mini (main) | `final_parsing_tokenized` | supplied |
| B | spaCy rule-based | `spacy_parsing_tokenized` | `experiments/ablation/parsers/run_spacy_parser.py` |
| C | LLaMA-3 | `llama_parsing_tokenized_clipped` | `experiments/ablation/parsers/clip_parse_cache.py` |
| D | none | `noparse_tokenized` | `experiments/ablation/parsers/make_noparse_cache.py` |
| E | small local LM (`--list-models`) | `smalllm_parsing_tokenized` | `experiments/ablation/parsers/run_smalllm_parser.py` |
| E′ | small local LM, one folder per model | `smalllm_<model>_parsing_tokenized` | `experiments/ablation/parsers/parse_with_smalllm.py` |

Use **E′** when comparing several small models. `run_smalllm_parser.py` writes every model
to the same pair of folders, so parsing with `qwen2.5` and then `smollm2` overwrites the
first result; `parse_with_smalllm.py` derives the folder name from the model and refuses to
resume a checkpoint written by a different one.

Every variant goes through the same tokenizer
(`experiments/ablation/parsers/tokenize_parse.py`) with the same caps, so the language
module and fusion network never change between arms.

Select one at run time:

```bash
python scripts/ScanRefer_train.py --parsing_folder spacy_parsing_tokenized ...
```

Score a parser against `object_name`:

```bash
python experiments/ablation/parsers/eval_parser_target_accuracy.py --splits train \
    --parsed-dir data_parsing/final_parsing_tokenized --tag gpt4o-mini
```

Target-extraction accuracy on the 36,665 train descriptions:

| Parser | fuzzy match |
|---|---|
| LLaMA-3 | 83.53% |
| GPT-4o-mini | 82.30% |
| spaCy | 73.12% |

---

## Training

```bash
python scripts/ScanRefer_train.py --use_color --use_normal \
    --batch_size 8 --epoch 100 --tag MY-RUN
```

| Flag | Meaning |
|---|---|
| `--use_color` / `--use_normal` | input channels; must match the scene cache |
| `--batch_size`, `--epoch`, `--seed` | the usual |
| `--tag` | suffix of the output folder `outputs/<timestamp>_<TAG>/` |
| `--lang_num_max` | descriptions per scene per sample (32 train, 1 eval) |
| `--parsing_folder` | which parse cache to read |
| `--use_cached_scenes` | train on cached detector output (see below) |
| `--disable_copy_paste` | turn off the proposal copy-paste augmentation |

Each run writes `outputs/<timestamp>_<TAG>/` with `model.pth`, `checkpoint.tar`,
`best.txt` (best-epoch metrics), `eval.txt` (per-epoch log) and `info.json`.

> **Note on `--tag`.** `scripts/ScanRefer_train.py` contains a `debug = True` block
> (lines 495–509) that overwrites `args.tag` after `parse_args()`. Any sweep relying on
> distinct tags must confirm each run landed in its own folder, or set the tag inside that
> block.

### Frozen-detector protocol

The detection branch reads only `point_clouds` and is identical across every ablation that
changes language or fusion. Caching it once turns a 1.5-day run into a 2–3 hour one:

```bash
python experiments/ablation/scenes_cache.py --splits train val --use_color --use_normal   # GPU, ~40 min
python experiments/diagnostics/validate_scene_cache.py --use_color --use_normal
python scripts/ScanRefer_train.py --use_cached_scenes --tag MY-ABLATION --use_color --use_normal
```

The cache is augmentation-free, so cached runs are comparable to each other rather than to
end-to-end runs. Details in
[`experiments/experiments.md`](experiments/experiments.md) §2.1.

---

## Evaluation

```bash
python scripts/ScanRefer_eval.py --folder <run> --reference --force \
    --use_color --use_normal --lang_num_max 1
```

Writes into `outputs/<run>/`:

| File | Contents |
|---|---|
| `predictions.p` | per-annotation `pred_bbox` (8,3), `gt_bbox` (8,3), `iou`, keyed `scene/object/ann` |
| `scores.p` | flat arrays: `ious`, `ref_acc`, `masks` (unique/multiple), `others`, `lang_acc` |

`predictions.p` is what the analysis suite runs on — most of `analysis/` is pure
post-processing of it, with no GPU and no model.

Detection metrics instead of grounding: `--detection` in place of `--reference`.

---

## Visualisation

```bash
python scripts/visualize.py --folder <run> --scene_id scene0011_00 \
    --use_color --use_normal
```

Writes `.ply` files (scene point cloud, predicted box, ground-truth box) under
`outputs/<run>/vis/`. To choose which scenes to show, use
`experiments/analysis/failure_cases.py` — it selects failures, attributes a cause, and
emits standalone box `.ply` files plus the exact `visualize.py` command per case.

---

## Ablations

Every flag lives in `experiments/ablation/ablation_config.py` and can be overridden from
the CLI. Each runner is standalone with a `CONFIG` block at the top; none imports from
another.

| Runner | Device | What it isolates |
|---|---|---|
| `run_parser_gpt.py` | GPU | variant A, GPT-4o-mini |
| `run_parser_spacy.py` | GPU | variant B, rule-based |
| `run_parser_llama.py` | GPU | variant C, LLaMA-3 |
| `run_parser_none.py` | GPU | variant D, no parser |
| `run_parser_smalllm.py` | GPU | variant E, small local LM |
| `run_seeds.py` | GPU | run-to-run variance |
| `run_no_copypaste.py` | GPU | proposal copy-paste augmentation |
| `run_parse_corruption.py` | GPU (eval only) | parse-error propagation |
| `run_eval_only_parser_swap.py` | GPU (eval only) | sensitivity to parse quality |
| `sweep_attention_layers.py` | GPU | number of attention layers |
| `aggregate_seed_results.py` | **CPU** | mean ± std over runs |

**Variant D** removes only the parse content and leaves A2F and TAF running, so the
difference against A/B/C/E is attributable to the parser alone. Its fields carry a single
`unk` token rather than zeros, which the existing code forces: `pack_padded_sequence`
rejects a zero-length sequence, and `_transform_parsed` maps every token — including
out-of-vocabulary ones — to a non-zero GloVe row (`glove["pad"]` has Σ|·| = 89.3, an
ordinary trained vector). A single `unk` is what `lib/dataset.py:687-689` already writes
for "no parse".

---

## Analysis

All CPU — no GPU, no model, no re-evaluation. These read `predictions.p`, the parse caches
and ScanRefer's JSON. Full detail in
[`experiments/experiments.md`](experiments/experiments.md) §4.

```bash
P=outputs/<run>/predictions.p
B=outputs/3DVG-TRANS-outputs/predictions.p

python experiments/analysis/results_table.py --predictions ours=$P --predictions 3DVG-Trans=$B --latex
python experiments/analysis/linguistic_complexity.py --predictions ours=$P --predictions 3DVG-Trans=$B
python experiments/analysis/parse_quality_split.py --predictions $P \
    --parse gpt4o-mini=final_parsing_tokenized --parse spacy=spacy_parsing_tokenized
python experiments/analysis/failure_cases.py --predictions $P --top 6
python experiments/analysis/annotation_sheet.py --num 200 --parse gpt4o-mini=final_parsing_tokenized
python experiments/analysis/error_taxonomy.py --sheet gpt4o-mini=<filled csv>
python experiments/analysis/parse_error_propagation.py --run-dir outputs/<run>/corruption
python experiments/ablation/runners/aggregate_seed_results.py --pattern '*ABL-SEED*' --latex

# the two fields that have no ground truth -- these need no predictions.p at all
python experiments/analysis/parse_field_comparison.py \
    --parse gpt4o-mini=final_parsing_tokenized \
    --parse llama=llama_parsing_tokenized_clipped \
    --parse spacy=spacy_parsing_tokenized
python experiments/analysis/complex_sentence_showdown.py --num 10
```

| Analysis | What it measures | Output |
|---|---|---|
| `results_table.py` | Overall/Unique/Multiple against a baseline | table + paired McNemar + LaTeX |
| `linguistic_complexity.py` | accuracy by language complexity, not object ambiguity | table + PNG + relation-type breakdown |
| `parse_quality_split.py` | grounding accuracy vs parse correctness | class-controlled difference + CMH test |
| `parse_error_propagation.py` | accuracy under deliberate parse corruption | decay curve + McNemar |
| `failure_cases.py` | failure selection with cause attribution | report + PLY boxes |
| `annotation_sheet.py` / `error_taxonomy.py` | attribute/neighbor validity, by hand | sheets + taxonomy table |
| `parse_field_comparison.py` | adjectives/neighbors across parsers, reference-free | coverage + faithfulness + agreement, PNG |
| `complex_sentence_showdown.py` | parsers on the hardest descriptions | quartile table + side-by-side figure |
| `aggregate_seed_results.py` | variance across seeds | mean ± std + LaTeX row |

---

## Computational complexity

```bash
python experiments/complexity/measure_parsing_latency.py --mode offline_check     # CPU, instant
python experiments/complexity/measure_parsing_latency.py --mode spacy --num_samples 500   # CPU
python experiments/complexity/measure_complexity.py --variant both --use_color --use_normal  # GPU
```

Reports FLOPs (via `torch.utils.flop_counter.FlopCounterMode`), peak GPU memory for
inference and training, and warmed, `cuda.synchronize()`-bracketed latency as
mean/std/p50/p95.

Parsing latency, measured on an 8-core CPU with `en_core_web_sm` over 200 distinct val
descriptions: **spaCy 3.31 ms mean, p95 4.40 ms, $0/query**. Parsing is offline, so it is
not part of per-query grounding latency.

Details in [`experiments/experiments.md`](experiments/experiments.md) §5.

---

## Running everything

[`run_analysis_colab.ipynb`](run_analysis_colab.ipynb) runs every experiment in dependency
order. Set the device and the stage toggles in the first cell:

```python
DEVICE = "cpu"        # or "cuda"
RUN = {"env_check": True, "parse_caches": True, "analyses": True, ...}
```

| Stage | What | Device |
|---|---|---|
| 0 | Environment check | CPU |
| 1 | Parse caches (B, D, E, corrupted) | CPU (E: GPU) |
| 2 | Scene cache build + validation | GPU |
| 3 | Parser target accuracy | CPU |
| 4 | Training runs | GPU |
| 5 | Evaluation → `predictions.p` | GPU |
| 6 | Parse-corruption sweep | GPU (eval only) |
| 7 | Eval-only parser swap | GPU (eval only) |
| 8 | Post-hoc analyses | **CPU** |
| 9 | FLOPs / memory / latency | GPU + CPU |
| 10 | Summary of artifacts | CPU |

GPU stages refuse to run when `DEVICE="cpu"` unless `ALLOW_GPU_STAGES_ON_CPU=True`. Stage 8
works with no GPU at all, from the `predictions.p` already in `outputs/`.

---

## Results

ScanRefer validation, ours vs the 3DVG-Trans baseline in `outputs/3DVG-TRANS-outputs/`.
Both were evaluated on the identical 9,508 annotations, so every comparison is paired
(McNemar); CIs are bootstrap over annotations.

| Model | Overall @0.25 | Overall @0.5 | Unique @0.25 | Unique @0.5 | Multiple @0.25 | Multiple @0.5 |
|---|---|---|---|---|---|---|
| ours | **48.91** | **35.54** | 79.62 | 55.83 | **41.51** | **30.65** |
| 3DVG-Trans | 46.23 | 30.87 | 79.51 | 53.71 | 38.22 | 25.37 |

| Difference | pp | 95% CI | McNemar p | significant |
|---|---|---|---|---|
| overall @0.25 | +2.67 | [+1.69, +3.70] | 1.5e-07 | yes |
| overall @0.5 | +4.67 | [+3.74, +5.60] | 8.1e-22 | yes |
| unique @0.25 | +0.11 | [−1.93, +1.98] | 0.96 | no |
| multiple @0.25 | +3.29 | [+2.11, +4.50] | 1.8e-08 | yes |
| multiple @0.5 | +5.29 | [+4.14, +6.38] | 2.9e-22 | yes |

The advantage is concentrated in `multiple` — scenes containing several objects of the
target class — and is statistically indistinguishable from zero on `unique`. Ambiguous
scenes are where adjacency reasoning is expected to help.

Reproduce with `python experiments/analysis/results_table.py --predictions ours=…
--predictions 3DVG-Trans=…`.

### Analysis findings

**Accuracy by linguistic complexity** (val, n = 9,508). Accuracy falls with description
length (per-sample Spearman ρ = −0.041, p = 5.4e-05) and dependency depth (ρ = −0.038,
p = 2.3e-04); the adjacency and spatial-word counts show the same direction but are not
significant.

The gap over the baseline per complexity bin:

| Measure | per-bin gap (pp) | highest − lowest bin | 95% CI |
|---|---|---|---|
| description length | +1.40, +2.59\*, +4.05\*, +3.15\* | +1.75 pp | [−1.03, +4.01] |
| dependency depth | +2.26\*, +2.16\*, +4.21\* | +1.96 pp | [−0.92, +4.50] |
| adjacent objects | +1.77, +2.37\*, +3.92\*, +3.92 | +2.15 pp | [−9.12, +12.37] |
| spatial-relation words | +1.43, +1.91\*, +4.52\*, +2.11 | +0.68 pp | [−4.53, +5.53] |

\* = that bin's gap is individually significant (McNemar, p < 0.05).

The gap is positive in every bin and larger in the complex bins, though every CI on the
widening includes zero.

**Grounding accuracy vs parse correctness** (Acc@0.25):

| Parser | raw diff | raw p | class-controlled diff | CMH p |
|---|---|---|---|---|
| GPT-4o-mini | +2.98 pp | 0.17 | **+7.54 pp** | 4.8e-04 |
| spaCy | +1.89 pp | 0.27 | **+5.72 pp** | 5.9e-04 |
| LLaMA-3 | +1.60 pp | 0.46 | **+6.47 pp** | 3.0e-03 |

The raw comparison is confounded and non-significant; controlling for object class makes
the effect larger and highly significant, because the misparsed subset happens to contain
easier classes.

Built on [ScanRefer](https://github.com/daveredrum/ScanRefer),
[VoteNet](https://github.com/facebookresearch/votenet) and
[3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer).
