# Enhancing 3D Visual Grounding through Semantic-Guided Attention and Graph Networks with LLM-Based Sentence Parsing

3D visual grounding: given a point cloud and a description, localise the referred object
with a 3D box. A language model parses the description into three semantic fields, fused
with detected object proposals via attention and a spatial graph network.

Evaluated on **ScanRefer** and **ReferIt3D / Sr3D**. Results are in the paper, not here.

---

## Architecture

```
point cloud (N x 10)                 description
      |                                   |
      v                                   v
 PointNet++ -> Hough voting -> DETR   LLM parser [offline] -> GloVe + GRU
      |                                   |
      +------------> proposals -----------+
                          |
                          v
                MatchModule: LGA -> A2F + TAF
                          |
                          v
                 referred-object scores
```

| Module | File |
|---|---|
| `Pointnet2Backbone` | `models/backbone_module.py` |
| `VotingModule` | `models/voting_module.py` |
| `ProposalModule` | `models/proposal_module.py` |
| `LangModule` | `models/lang_module.py` |
| `MatchModule` | `models/match_module.py` |
| `RefNet` | `models/refnet.py` (assembles all of the above) |

Parsed fields: `target` (cap 7), `adjectives` (cap 17), `neighbors` (cap 75) — tokenised,
GloVe-embedded; an absent field is `"not mentioned"`. Parsing is offline: `lib/dataset.py`
reads a precomputed `tokenized_parsed_result_{split}.json`; no parser runs in the forward
path.

---

## Repository layout

```
.
├── run_analysis.ipynb, run_one_experiment.ipynb, results_analysis.ipynb, ...   notebooks
├── scripts/            entry points: train, eval, predict, visualize
├── models/             the network
├── lib/                dataset, solver, losses, config, eval helpers
├── utils/              point-cloud / box / NMS utilities
├── data/                ScanRefer JSON, GloVe, ScanNet
├── data_parsing/       parse caches, one folder per parser variant
├── cached_scenes/      frozen-detector output
├── outputs/            runs, checkpoints, predictions, reports
├── experiments/        ablations, analyses, measurements (self-contained, `# ABLATION`-tagged)
│   └── ablation/       cache.py, parsing.py, parsing_smalllm.py, train_arms.py, sweep_attention_layers.py
└── pointnet/           three PointNet++ implementations (CUDA / optimised PyTorch / pure Python)
```

---

## Installation

```bash
conda create -n 3dvg python=3.10 -y && conda activate 3dvg
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r req/requirements.txt

# pyg-lib is required (MatchModule's knn_graph); not on PyPI, resolve against the torch build
pip install torch-geometric
TORCH=$(python -c "import torch; print(torch.__version__)")
pip install pyg_lib -f "https://data.pyg.org/whl/torch-${TORCH}.html"

python -m spacy download en_core_web_sm     # parser variant B
pip install ./pointnet/pointnet2_ops_lib    # CUDA PointNet++ ops (optional, falls back to pure Python)
```

Optional: `transformers accelerate` (parser variant E), `openai` (GPT latency measurement).

---

## Data preparation

```
data/
├── ScanRefer_filtered_{train,val,test}.json
├── glove.p
└── scannet/{scannet_data, meta_data, scans}/
```

1. Download [ScanRefer](https://github.com/daveredrum/ScanRefer) into `data/`.
2. Download ScanNet v2 into `data/scannet/scans/`.
3. `cd data/scannet && python batch_load_scannet_data.py`.

---

## Sentence parsing

| Variant | Parser | Folder |
|---|---|---|
| A | GPT-4o-mini (main) | `final_parsing_tokenized` |
| B | spaCy | `spacy_parsing_tokenized` |
| C | LLaMA-3 | `llama_parsing_tokenized_clipped` |
| D | none | `noparse_tokenized` |
| E | small local LM | `smalllm_parsing_tokenized` |

Build with `experiments/ablation/parsing.py {spacy,clip,noparse}` or `parsing_smalllm.py`.
Select at train time with `--parsing_folder`.

We used gpt-4o-mini with the maximum output length set to 4096 tokens and the temperature set to 0 to minimize variability in the generated outputs; all other parameters were left at their default values.

---

## The experiment phases

| phase | needs |
|---|---|
| A — retraining | GPU |
| B — evaluation | GPU + checkpoint |
| C — post-processing | nothing (reads `predictions.p`) |
| D — geometric prior | checkpoint |
| E — online parsing | GPU |

Order on a fresh machine: C1 (parse caches) → A → B → rest of C. Flags live in
`experiments/ablation/ablation_config.py`.

---

## Training

```bash
python scripts/ScanRefer_train.py --use_color --use_normal --batch_size 8 --epoch 100 --tag MY-RUN
```

Key flags: `--parsing_folder`, `--use_cached_scenes` (frozen-detector protocol),
`--disable_copy_paste`. Writes `outputs/<timestamp>_<TAG>/`.

Frozen-detector protocol (caches the detection branch, since it's identical across
language/fusion ablations):

```bash
python experiments/ablation/cache.py build --splits train val --use_color --use_normal
python scripts/ScanRefer_train.py --use_cached_scenes --tag MY-ABLATION --use_color --use_normal
```

---

## Evaluation

```bash
python scripts/ScanRefer_eval.py --folder <run> --reference --force --use_color --use_normal --lang_num_max 1
```

Writes `predictions.p` (per-annotation boxes + IoU) and `scores.p`, which the analysis
suite reads. `--detection` for detection metrics instead of grounding.

---

## Visualisation

```bash
python scripts/visualize.py --folder <run> --scene_id scene0011_00 --use_color --use_normal
```

Writes `.ply` files. `experiments/analysis.py failure-cases` picks which scenes to show.

---

## Ablations

Runners in `experiments/ablation/`: `train_arms.py run {parser-gpt,parser-spacy,parser-llama,
parser-none,parser-smalllm,seeds,no-copypaste}`, `sweep_attention_layers.py`. Each isolates
one variant against the reference arm (`parser-gpt`).

---

## Analysis

CPU-only, reads `predictions.p` and the parse caches. Key entry points in `experiments/analysis.py`:
`results-table`, `linguistic-complexity`, `parse-quality-split`, `failure-cases`,
`annotation-sheet`, `error-propagation`, `seed-aggregate`, `parse-field-comparison`. See
`--help` on the script for the full list and options.

---

## Complexity and latency

```bash
python experiments/measurements.py model --variant both --use_color --use_normal
python experiments/measurements.py online-latency --num_samples 50 --use_color --use_normal
```

Reports FLOPs, peak GPU memory, and latency. `online-latency` covers the live-parsing
deployment case; parsing is offline everywhere else.

---

## Diagnostics

```bash
python experiments/measurements.py audit-scene-cache        # cache integrity, no GPU
python experiments/measurements.py validate-scene-cache     # strict, needs CUDA
python experiments/measurements.py cached-eval-cpu --fusion-variant original
python experiments/measurements.py dist-weights-sweep --num-samples 200
```

The shipped checkpoint (`outputs/2024-12-18_20-40-38_3DVG-FIXED`) needs
`--fusion_variant original` — it predates the current `MatchModule`. `ABLATION.STRICT_CHECKPOINT`
(default on) refuses to silently load it otherwise.

`dist-weights-sweep` tests which attention head carries the geometric distance prior, over
four channel-assignment arms (`baseline`, `both_dist`, `both_norm`, `swapped`).

---

## Notebooks

| Notebook | For |
|---|---|
| `run_analysis.ipynb` | every experiment, in order |
| `run_one_experiment.ipynb` | one experiment, with a confirmation gate |
| `results_analysis.ipynb` | aggregates finished artifacts into tables/figures |
| `run_parser_analysis.ipynb` | adjectives/neighbors across parsers |
| `run_parse_quality.ipynb` | parse quality, failure taxonomy |
| `run_smalllm_parsing.ipynb` | build a small-LM parse cache |
| `run_dist_weights_sweep.ipynb`, `run_online_latency.ipynb` | phase D / E, with figures |

---

## Troubleshooting

**`pyg-lib` missing** — model can't build at all (`knn_graph` in `MatchModule`'s forward
pass). Install per the [Installation](#installation) section; CPU-only analyses are unaffected.

**`compute_37` build error** — `compute_37` (Kepler) was removed in CUDA 12. Set
`TORCH_CUDA_ARCH_LIST` explicitly, install `ninja`, and rebuild:
```bash
pip install ninja
rm -rf pointnet/pointnet2_ops_lib/build pointnet/pointnet2_ops_lib/pointnet2_ops.egg-info
pip install ./pointnet/pointnet2_ops_lib
```

**OOM building the dataset, not training** — host RAM, not GPU memory; a smaller batch
size doesn't help. Fixed by `ABLATION.LAZY_LANG_DATA` (default on); pass `--no_lazy_lang_data`
to restore the original eager behaviour if you have RAM to spare.

**`cuda.is_available() == True` but the GPU still fails** — the card's compute capability
may predate the installed torch build. `run_analysis.ipynb` probes with a real kernel launch.

---

Built on [ScanRefer](https://github.com/daveredrum/ScanRefer),
[VoteNet](https://github.com/facebookresearch/votenet) and
[3DVG-Transformer](https://github.com/zlccccc/3DVG-Transformer).
