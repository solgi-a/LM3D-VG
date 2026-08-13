# Experiments — the complete guide

What each experiment tests, how to run it, and what has already been measured.

**Run everything from the repository root**, either through
[`run_analysis_colab.ipynb`](../run_analysis_colab.ipynb) — which runs all of it in
dependency order and skips what the machine cannot do — or one script at a time.

Four notebooks, each for a different job:

| Notebook | Use it for |
|---|---|
| [`run_analysis_colab.ipynb`](../run_analysis_colab.ipynb) | every experiment, in dependency order |
| [`run_one_experiment.ipynb`](../run_one_experiment.ipynb) | **one** experiment, every knob exposed, with a dry run and a confirmation gate |
| [`run_parser_analysis.ipynb`](../run_parser_analysis.ipynb) | §4.4b and §4.4c, with the tables and figures rendered inline |
| [`run_smalllm_parsing.ipynb`](../run_smalllm_parsing.ipynb) | building a variant-E parse cache, one folder per model |

---

## Contents

| § | |
|---|---|
| [1](#1-orientation) | Orientation — the three phases, the layout, the switchboard |
| [2](#2-phase-a--retraining) | Phase A — retraining (7 arms + the sweep) |
| [3](#3-phase-b--evaluation) | Phase B — evaluation |
| [4](#4-phase-c--post-processing) | Phase C — post-processing and analysis |
| [5](#5-complexity-flops-memory-latency) | Complexity — FLOPs, memory, latency |
| [6](#6-diagnostics) | Diagnostics — implementation checks |
| [7](#7-environment-and-troubleshooting) | Environment and troubleshooting |
| [8](#8-full-command-list-in-dependency-order) | Full command list, in dependency order |

---

# 1. Orientation

## The three phases

Experiments are grouped by what they **cost** and what they **need**, because those are
the two things that decide whether you can run one right now.

| phase | what it does | needs | cost |
|---|---|---|---|
| **A — retraining** | trains a model from scratch, one arm per ablation | GPU | hours **each** |
| **B — evaluation** | runs a trained checkpoint over val | GPU + a checkpoint | minutes |
| **C — post-processing** | reads files off disk; builds no model | nothing | seconds to minutes |

Phase A is the long pole and everything downstream waits on it. Phase C is the one that
produces the paper's tables, and it works today on any machine — if no GPU is available,
those tables can still be regenerated in seconds.

> **Dependency order on a fresh machine** is C1 → A → B → the rest of C: the parse
> caches that phase A consumes are built in C1. The caches usually already exist, which
> is why the notebook lists phase A first.

## Layout

All experiment code lives under `experiments/`. The dependency runs one way only — `lib/`, `models/` and `scripts/` import
`experiments.ablation.ablation_config` for the handful of flags that switch an
experiment on, and nothing else.

```
experiments/
├── ablation/       rebuilds a cache or retrains          parsers/, runners/
├── analysis/       post-hoc on a finished run            CPU only, no model
├── complexity/     FLOPs, GPU memory, latency, parse cost
└── diagnostics/    implementation checks
```

In detail:

```
experiments/ablation/
  ablation_config.py              THE switchboard -- every flag lives here
  ablation_hooks.py               dataset/model selection; returns originals when off
  cached_scenes.py                cache format + CachedSceneDataset
  cached_refnet.py                CachedRefNet (language + fusion, no detector)
  scenes_cache.py                 [entry] build the scene cache
  parsers/
    spacy_parser.py               rule-based parser (variant B)
    tokenize_parse.py             shared tokenizer + caps
    run_spacy_parser.py           [entry] produce the spaCy parse            (variant B)
    run_smalllm_parser.py         [entry] produce the small-LM parse         (variant E)
    parse_with_smalllm.py         [entry] variant E, one folder per model    (variant E')
    make_noparse_cache.py         [entry] produce the empty parse            (variant D)
    corrupt_parse_cache.py        [entry] deliberately corrupt a parse cache
    clip_parse_cache.py           [entry] enforce 7/17/75 caps on a parse cache
    eval_parser_target_accuracy.py  [entry] score a parser vs object_name
  runners/
    run_*.py                      [entry] one independent runner per ablation
    aggregate_seed_results.py     [entry] mean +/- std across seed runs
    sweep_attention_layers.py     [entry] attention-layer sweep

experiments/diagnostics/
  validate_scene_cache.py         [entry] prove cached == end-to-end   (GPU, strict)
  audit_scene_cache.py            [entry] model-free integrity + recall ceiling (CPU)
  cached_eval_cpu.py              [entry] cached-path accuracy without CUDA

cached_scenes/                    generated scene cache, REPO ROOT (git-ignored)
  train/<scene_id>.p  val/<scene_id>.p  meta.json

data_parsing/                     ALL parse caches, REPO ROOT (git-ignored)
  final_parsing/                    GPT-4o-mini, raw phrase strings
  final_parsing_tokenized/          GPT-4o-mini, tokenized     <- variant A
  final_parsing_llama/              LLaMA, raw phrase strings
  final_parsing_tokenized_llama/    LLaMA, tokenized (EXCEEDS caps -- do not use directly)
  llama_parsing_tokenized_clipped/  LLaMA, caps enforced       <- variant C
  spacy_parsing/                    spaCy, raw phrase strings
  spacy_parsing_tokenized/          spaCy, tokenized           <- variant B
  noparse_tokenized/                all fields "unk"           <- variant D
  smalllm_parsing_tokenized/        small local LM             <- variant E
```

## The switchboard

Everything is off by default and controlled from one file,
`experiments/ablation/ablation_config.py`:

```python
ABLATION.USE_CACHED_SCENES       = False   # frozen-detector protocol
ABLATION.CACHED_SCENES_ROOT      = "cached_scenes"
ABLATION.DETERMINISTIC_SUBSAMPLE = False
ABLATION.SUBSAMPLE_SALT          = "v1"
ABLATION.PARSING_FOLDER          = "final_parsing_tokenized"   # parser variant
ABLATION.DISABLE_COPY_PASTE      = False   # copy-paste ablation
ABLATION.FUSION_VARIANT          = "current"   # "current" | "original"
ABLATION.STRICT_CHECKPOINT       = True    # refuse a partially-loaded checkpoint
ABLATION.LAZY_LANG_DATA          = True    # host-RAM control, not an ablation
```

`ABLATION.PARSING_FOLDER` names a subfolder of `data_parsing/`, resolved through
`CONF.PATH.PARSING` (`lib/config.py`) — one source of truth for the location.

Flip a flag, or override from the CLI (`--use_cached_scenes`, `--parsing_folder ...`,
`--disable_copy_paste`, `--fusion_variant original`, `--no_strict_checkpoint`,
`--no_lazy_lang_data`), and nothing else needs editing. Each entry-point script prints
the active config at startup.

The last two are **not experiments** and are deliberately excluded from `enabled()`:
they change *how* a run executes, never what it computes. `STRICT_CHECKPOINT` makes a
half-loaded checkpoint an error instead of a silently meaningless accuracy;
`LAZY_LANG_DATA` decides when the language tensors are built (see §7).

With every experiment flag `False`, `build_model` returns `RefNet` and `build_dataset`
yields the original pipeline. It returns a `CachedSceneDataset` with `use_cache=False`
and `deterministic=False` rather than a bare `ScannetReferenceDataset`, purely so the
lazy language loading applies; the tensors are identical either way, and
`--no_lazy_lang_data` restores the literal original class.

## Changes to pre-existing files

**22 changed lines across 6 files, every one tagged `# ABLATION`**, plus 6 lines in
`.gitignore`. `models/refnet.py`, `lib/solver.py`, `lib/loss_helper.py`,
`lib/eval_helper.py`, `lib/ap_helper.py` and `models/lang_module.py` are **untouched**.

| File | Lines | What |
|---|---|---|
| `scripts/ScanRefer_train.py` | 6 | import; two hooks; guard the pretrained-detector mount; register + apply config |
| `scripts/ScanRefer_eval.py` | 5 | import; two hooks; register + apply config |
| `scripts/predict.py` | 5 | same |
| `scripts/visualize.py` | 4 | same |
| `lib/dataset.py` | 2 | import; `folder = ABLATION.PARSING_FOLDER` |
| `models/match_module.py` | 3 | import; `and not ABLATION.DISABLE_COPY_PASTE` on the two copy-paste gates |

Separately from the ablation work, `scripts/ScanRefer_train.py` also received the
argument-handling, warm-start and dataloader fixes described in §2.

---

# 2. Phase A — retraining

Seven arms, plus a hyper-parameter sweep. Each is hours of GPU time, each has **its own
runner, its own CONFIG block and its own switch**, and none imports from another — so
changing one experiment cannot disturb the others, and a Colab session that dies in the
middle of one arm can be resumed by turning the finished ones off.

| | experiment | runner | ablates |
|---|---|---|---|
| **A1** | no copy-paste | `run_no_copypaste.py` | proposal copy-paste augmentation |
| **A2** | variant A | `run_parser_gpt.py` | — (reference arm) |
| **A3** | variant B | `run_parser_spacy.py` | LLM → rule-based parser |
| **A4** | variant C | `run_parser_llama.py` | GPT API → open LLaMA-3 |
| **A5** | variant D | `run_parser_none.py` | the parser entirely |
| **A6** | variant E | `run_parser_smalllm.py` | GPT → 0.5B local model |
| **A7** | seeds | `run_seeds.py` | nothing — repeats A2 under more seeds |
| **A8** | attention sweep | `sweep_attention_layers.py` | *(hyper-parameter, not an arm)* |

## 2.1 The frozen-detector protocol (scene caching)

### Why it is sound

`RefNet`'s detection branch (`Pointnet2Backbone` → `VotingModule` → `ProposalModule`,
i.e. PointNet++ → Hough voting → DETR decoder) reads only `point_clouds`. It never sees
the referring description. The dataloader already groups each sample as *one scene + up
to `lang_num_max` descriptions* and loads the point cloud by `scene_id` alone, so the
detector output is a function of `scene_id`. **`scene_id` is the correct cache key** —
verified against the code, not assumed.

### What gets cached

The 18 tensors in `experiments/ablation/cached_scenes.py :: SCENE_CACHE_KEYS_REQUIRED`,
derived by tracing every downstream consumer:

| Consumer | Reads |
|---|---|
| `models/match_module.py` | `detr_features`, `center`, `objectness_scores` |
| `lib/loss_helper.py` | `aggregated_vote_xyz`, `center`, `objectness_scores`, `heading_*`, `size_*`, `sem_cls_scores`, `seed_xyz`, `seed_inds`, `vote_xyz` |
| `lib/eval_helper.py` | `center`, `heading_*`, `size_*`, `objectness_scores` |
| `lib/ap_helper.py` | as above plus `sem_cls_scores` |

`detr_features` and `aggregated_vote_features` are stored fp16 (they feed a Conv1d /
attention input); geometry and logits stay fp32 because they drive IoU thresholds and
argmax. Roughly **250 KB/scene → ~180 MB** for all 703 train+val scenes.

### Two problems the cache had to solve

**Determinism.** `utils/pc_utils.random_sampling` redraws its 40 000-point subsample on
**every** `__getitem__`, including when `augment=False`. The detector output was
therefore *not* reproducible per scene, which would make the equivalence check
unsatisfiable. `CachedSceneDataset` seeds numpy's global RNG from an md5 of the scene id
(`hashlib`, not Python's per-process-salted `hash()`) before delegating to the base
class, then restores the previous state — pinning the draw *without editing*
`lib/dataset.py`.

**Memory.** `ScannetReferenceDataset._load_data` eagerly preloads every scene's mesh
vertices and per-point labels (~1.6 GB on train). `CachedSceneDataset` loads them
**lazily**, one scene at a time behind a small LRU. Identical GT tensors, a fraction of
the RAM. This matters because `compute_vote_loss` still runs (its result is discarded
when `detection=False`), so `vote_label` must still be produced — the arrays cannot
simply be dropped.

### Augmentation: what caching does and does not disturb

| Augmentation | Where | Cached mode |
|---|---|---|
| Flip / rotate / scale / translate | `lib/dataset.py`, raw point cloud | **disabled** |
| 40k-point random subsample | `utils/pc_utils.random_sampling` | **frozen** per scene |
| **Proposal copy-paste** | `models/match_module.py` — *feature level, inside the fusion network* | **unaffected**, still randomises every iteration |
| Language masking / reversal | `models/lang_module.py` | **unaffected** |

> Proposal copy-paste is **not** a point-cloud augmentation. It operates on proposal
> features downstream of the cache boundary, so the dedicated copy-paste ablation can
> run **on the cache** — no end-to-end bypass is needed.

### Building the cache

```bash
# 0. pre-flight, CPU-safe, no forward pass, nothing written
python experiments/ablation/scenes_cache.py --dry_run \
    --use_pretrained 2024-12-18_20-40-38_3DVG-FIXED \
    --splits train val --use_color --use_normal

# 1. build the cache (auto-runs validation at the end) -- GPU REQUIRED
python experiments/ablation/scenes_cache.py \
    --use_pretrained 2024-12-18_20-40-38_3DVG-FIXED \
    --splits train val --use_color --use_normal

# 2. validation only — must pass before any ablation run
python experiments/diagnostics/validate_scene_cache.py \
    --cached_scenes_root cached_scenes --num_samples 200 \
    --use_pretrained 2024-12-18_20-40-38_3DVG-FIXED --use_color --use_normal
```

`--use_color --use_normal` are **not optional** — they set `input_feature_dim`, and
omitting them makes the checkpoint's `backbone_net.*` weights fail to load. That is a
hard error by design, not a silent fallback.

`--dry_run` checks, in the order they would otherwise fail at runtime: the checkpoint
exists and every detection-branch weight loads with matching shape; the parse cache,
GloVe pickle and per-scene `.npy` files are present; one full sample can be built per
split; and how many scenes will be written and roughly how much disk that needs. It
builds the model on CPU and touches no CUDA op.

### Building the cache on CPU

Supported, and the default settings are tuned for it. Two things made it possible:
`pointnet2_ops` (the CUDA extension) may be absent, in which case
`models/proposal_module.py` and `models/backbone_module.py` fall through the chain in
`pointnet/__init__.py` — first `pointnet2_py_adv` (optimised pure PyTorch), then
`pointnet2_python` (the reference), both CPU-clean; and the only hard CUDA dependency in
the detection path was two `.cuda()` calls in `decode_dataset_config`, which now follow
the input's device.

```bash
python experiments/ablation/scenes_cache.py --cpu \
    --use_pretrained 2024-12-18_20-40-38_3DVG-FIXED \
    --splits train val --use_color --use_normal
```

`--cpu` auto-selects `batch_size=1`, `num_workers=0` (an in-process loader; a worker
would fork the whole dataset), `lazy_maxsize=1` scene mesh resident, `num_threads =
cpu_count-1` so the machine stays usable, and a `gc.collect()` every 32 scenes.

`scenes_cache.py` builds `RefNet` with **`no_reference=True`**, so `RefNet.forward`
returns straight after the detection branch. The cache only ever stores detector output,
so running the language and fusion branches would be pure waste — and `LangModule`
hardcodes `.cuda()` at `models/lang_module.py:56`, which would break the CPU path for no
reason. `--no_reference` on the CLI is accepted but ignored.

Measured on an 8-core CPU (7 threads), `--use_color --use_normal`:

| | value |
|---|---|
| per scene | **~20 s** |
| peak RSS | **~4.2 GB** |
| val (141 scenes) | ~47 min |
| train (562 scenes) | ~3 h 10 min |
| both | **~4 h** |

**Resumable.** Every scene is written atomically (`.tmp` then `os.replace`) and
already-cached scenes are skipped, so Ctrl-C and re-running the same command continues
where it stopped. `--no_resume` forces a rebuild. On exit, `meta.json` records
`complete: true/false` and per-split `have/expected` counts; an incomplete run exits 2.
The progress bar counts *scenes*, not batches, with live scene/min and ETA.

**Validation cannot run on CPU** and is skipped automatically there. It calls
`get_loss`/`get_eval`, which contain ~37 further hardcoded `.cuda()` calls
(`lib/loss_helper.py` 27, `lib/eval_helper.py` 5, `models/match_module.py` 4,
`models/lang_module.py` 1). Those were deliberately left alone. **Run
`validate_scene_cache.py` on a GPU before using a CPU-built cache for any number in the
paper.** It exits non-zero on failure and prints a per-tensor table (shape, mean, std
delta) identifying which tensor diverged; if none did, it lists the three non-tensor
causes in likelihood order. Acc@0.25 is pooled over per-sample IoUs, so the number does
not depend on how batches happen to divide.

## 2.2 The five parser variants

| Variant | Parser | Parse folder | Produced by | Runner |
|---|---|---|---|---|
| A | GPT-4o-mini (paper's main) | `final_parsing_tokenized` | supplied with the repo | `run_parser_gpt.py` |
| B | spaCy rule-based | `spacy_parsing_tokenized` | `run_spacy_parser.py` | `run_parser_spacy.py` |
| C | LLaMA-3 | `llama_parsing_tokenized_clipped` | `clip_parse_cache.py` | `run_parser_llama.py` |
| **D** | **none** | `noparse_tokenized` | `make_noparse_cache.py` | `run_parser_none.py` |
| **E** | **small local LM** | `smalllm_parsing_tokenized` | `run_smalllm_parser.py` | `run_parser_smalllm.py` |
| **E′** | **small local LM, per model** | `smalllm_<model>_parsing_tokenized` | `parse_with_smalllm.py` | `run_parser_smalllm.py` (edit `PARSING_FOLDER`) |

Use **E′** whenever more than one small model is compared. `run_smalllm_parser.py`
hardcodes a single output folder, so a second model silently overwrites the first and
nothing on disk records which model produced a cache. `parse_with_smalllm.py` derives both
folder names from the model, writes the model id into `parse_run_summary.json`, and
refuses to resume a `.partial_{split}.json` written by a different model — without that
guard, switching `--model` mid-run blends two models' output into one cache.

Every variant is written through the same tokenizer
(`experiments/ablation/parsers/tokenize_parse.py`) with the same 7 / 17 / 75 token caps,
so the language module and fusion network are untouched and the only thing that differs
between arms is the parser, which is what makes it a controlled comparison.

### Schema, derived from the real files

`lib/dataset.py` reads `data_parsing/<folder>/tokenized_parsed_result_{split}.json`. The
raw form GPT-4o-mini writes is:

```json
{"target": "chair",
 "adjectives": "dark brown wooden and leather",
 "neighbors": "in the kitchen, placed in the table"}
```

Two properties are load-bearing and were verified across all 46,173 annotations:

1. **Fields are surface phrases, not head-word lists** — they keep determiners,
   prepositions, conjunctions and commas. Emitting bare head nouns would make the
   ablation compare *representation format* rather than parser quality.
2. **An absent field is the literal string `"not mentioned"`** — never `null`, `""` or
   `[]`. After tokenisation it becomes `["not", "mentioned"]`.

Tokenisation (lowercase, whitespace split, punctuation as its own token — matching
ScanRefer's own `token` field) and the **7 / 17 / 75 token caps** are shared by every
variant so all of them go through one identical path. The caps are load-bearing:
`_transform_parsed` allocates only 7/17/75 embedding rows while `__getitem__` sets the
sequence lengths **unclipped** (`lib/dataset.py:173-175`), so a longer field raises
inside `pack_padded_sequence` mid-training.

### The LLaMA cache needed fixing

`data_parsing/final_parsing_tokenized_llama` violates those caps in **5 annotations**
(train: one `target` of 8, three `adjectives` of 20; val: one `adjectives` of 22) — each
one a hard crash. `clip_parse_cache.py` writes a corrected copy to
`llama_parsing_tokenized_clipped` and never modifies the original.

```bash
python experiments/ablation/parsers/clip_parse_cache.py \
    --input  data_parsing/final_parsing_tokenized_llama \
    --output data_parsing/llama_parsing_tokenized_clipped
```

### Variant D — how "no parser" is encoded, and why it is not zeros

The architecture is left completely intact — A2F and TAF still run — and only the parse
content is removed, so the difference against the other variants is attributable to the
parser alone.

Two facts in `lib/dataset.py` force the encoding:

1. `__getitem__` sets `tgt_len = len(tokens)` **unclipped** (lines 173–175) and the
   language module feeds that to `pack_padded_sequence`, which **raises on length 0**.
   An empty field is not representable.
2. `_transform_parsed` (lines 570–581) fills row *i* with `glove[token]`, or with
   `glove["unk"]` when the token is out of vocabulary. **No token maps to a zero row** —
   verified directly: `glove["pad"]` has Σ|·| = 89.3, it is an ordinary trained vector.

So a literal zero mask is unreachable without editing `lib/dataset.py`, which this code
does not do. The encoding used is a single `unk` token, which is exactly what
the codebase already writes when `args.detection == True` (lines 687–689). A second
cache with `["not", "mentioned"]` is generated alongside it as a robustness check.

Suggested sentence for the manuscript:

> In the no-parser variant the three parsed fields are replaced by a single
> out-of-vocabulary token whose GloVe embedding is the model's `<unk>` vector. A
> zero-length field is not admissible because the language encoder packs padded
> sequences, so this is the minimal content-free encoding that leaves the architecture —
> including the A2F and TAF sub-modules — untouched.

```bash
python experiments/ablation/parsers/make_noparse_cache.py --splits train val --both  # CPU, ~10 s
python experiments/ablation/runners/run_parser_none.py                               # GPU
```

### Variant E — small local language model

A small local model with the same prompt as the GPT reference, covering the middle of the
range between a rule chain and a hosted LLM. A 0.5B model landing close to GPT-4o-mini
removes the API from the method's critical path.

```bash
python experiments/ablation/parsers/run_smalllm_parser.py --list-models   # downloads nothing
```

| alias | model | params | download | needs |
|---|---|---|---|---|
| `qwen2.5` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49B | ~1.0 GB | — |
| `qwen2` | `Qwen/Qwen2-0.5B-Instruct` | 0.49B | ~1.0 GB | — |
| `flan-t5` | `google/flan-t5-base` | 0.25B | ~1.0 GB | `sentencepiece` |
| `smollm2` | `HuggingFaceTB/SmolLM2-360M-Instruct` | 0.36B | ~0.7 GB | — |

`qwen2.5` is the default — same size as `qwen2` but a later instruction-tuning
generation, so it adheres to the JSON schema more reliably and the malformed rate is
lower. `flan-t5` is fastest on CPU but weakest at the schema.

Dependencies are checked **before** any network call, so a missing `sentencepiece` fails
in a second rather than after a 1 GB download.

The script counts and reports the **malformed-output rate**. The handling is:
extract the first balanced `{...}`; on a JSON failure repair single quotes and trailing
commas; on a second failure fall back to `"not mentioned"` per field, and count it.
**Quote that rate in the paper.**

```bash
pip install 'transformers>=4.40'          # 5.x is supported; the dtype kwarg is detected
python experiments/ablation/parsers/run_smalllm_parser.py --splits train val \
    --model qwen2.5 --device cuda
python experiments/ablation/runners/run_parser_smalllm.py                    # GPU

# or, to keep one cache per model (recommended when comparing several):
python experiments/ablation/parsers/parse_with_smalllm.py --splits train val \
    --model qwen2.5 --device cuda      # -> data_parsing/smalllm_qwen2.5_parsing_tokenized/
```

Generation is greedy (reproducible) and resumable — progress is checkpointed every 500
records to `.partial_{split}.json` and reloaded automatically, so an interrupted CPU run
picks up where it stopped. On CPU expect roughly 1–2 descriptions/second, i.e. ~2 h for
val and ~8 h for train+val; `--device cuda` is refused outright if the GPU cannot
execute a kernel, rather than failing 40 minutes in.

## 2.3 The copy-paste ablation (A1)

The augmentation lives at the two blocks in `models/match_module.py` gated on
`data_dict["istrain"][0] == 1 and random_numer < 0.5`, which copy valid proposals'
features into invalid proposal slots. `--disable_copy_paste` turns both off.

Compare `run_no_copypaste.py` against `run_parser_gpt.py` (identical settings,
copy-paste on) to isolate its contribution.

## 2.4 Seeds (A7)

Runs the unchanged main configuration once per seed into its own output folder, so the
run-to-run spread can be reported as mean ± std beside the ablation table.

What the seed still controls under the frozen detector: geometric point-cloud
augmentation is off, but proposal copy-paste, the word masking and sentence reversal in
`LangModule`, weight initialisation and batch order all sit downstream of the cache and
keep randomising. Those are the run-to-run variance sources.

> **Warm start and this experiment.** With `WARM_START` on (§2.6) every arm fine-tunes
> from the same checkpoint, so the spread measured here is the spread *of fine-tuning*,
> narrower than a from-scratch spread. For the wider figure, set `WARM_START = False` and
> raise `EPOCH`.

Aggregate the runs afterwards with `aggregate_seed_results.py` (§4.6).

## 2.5 The attention-layer sweep (A8)

```bash
python experiments/ablation/runners/sweep_attention_layers.py     # GPU
```

The neighbour count and the graph depth were each chosen against a reported curve; the
attention-layer count was not. This sweeps it.

**`--nhead` and `--num_decoder_layers` do not work for this.** They are read only inside
the `args.detector == "GF"` branch of `models/refnet.py`, and the training script runs
with `detector = 'VN'`. The two knobs that are real — `MatchModule`'s `depth=3` and the
hard-coded `config_transformer['dec_layers'] = 2` — are not reachable from the command
line, so each arm writes a launcher that patches both constructors and hands off to the
unmodified training script through `runpy`.

Verified on CPU (`SELF_TEST = True`): depth 1→4 scales the module 816,518 → 2,208,902
parameters, so the override reaches the constructor.

## 2.6 Warm start, hyper-parameters and data loading

All three are set once in the notebook's configuration cell and pushed into the seven
phase-A runners by the *apply training configuration* cell, so you do not need to open
each runner. They are also plain CLI flags on `scripts/ScanRefer_train.py`.

### Warm start

Loading fusion weights from an already-trained run instead of starting from random
initialisation. Under the frozen-detector protocol with `FUSION_VARIANT="original"` the
shipped checkpoint covers the model **completely**:

| fusion variant | model tensors | warm-started | left at random init |
|---|---|---|---|
| `original` | 146 | **146 (100 %)** | **0** |
| `current` | 294 | 116 (39 %) | 178 |

So this is a true fine-tune, not a partial one, which is why 50 epochs is comfortable.
Every phase-A arm warm-starts from the **same** checkpoint: the arms share a starting
point and differ only in what each ablates, so the comparison between them stays fair.
These are therefore fine-tuned runs, not from-scratch runs.

Warm start goes through `--use_checkpoint`, **not** `--use_pretrained`: under the
frozen-detector protocol `ablation_hooks.skip_pretrained_detector()` returns `True` and
the detector is never built, so the pretrained-detector mount is skipped by design.
`--no_warm_start` trains from random initialisation instead.

**`--keep_checkpoint` is mandatory alongside it.** In cached mode
`ablation_config.apply()` clears `--use_checkpoint` by default, because an *accidental*
shared initialisation would silently destroy the seed ablation — every "different seed"
would start from identical weights and measure nothing. Here the shared start is
intended, so the runners opt back in. Without the flag the warm start is discarded and the
run trains from random init **while appearing to fine-tune**.

`comp_weight()` copies only tensors matching by **name and shape** and now reports how
many it loaded, how many were skipped on a shape mismatch, and warns when more than half
the model is left randomly initialised. A permissive load is what makes a warm start
possible across architectures; the danger is that permissiveness is silent.

### Recommended hyper-parameters

| parameter | value | why |
|---|---|---|
| `EPOCH` | **50** | Generous with a 100 % warm start. From random init it is too few. |
| `VAL_STEP` | **5000** | **In iterations, not epochs** (`lib/solver.py` checks `_global_iter_id % val_step`). A previously hardcoded `10` ran a full validation pass every 10 iterations — more expensive than the training it interrupted. |
| `BATCH_SIZE` | 8 | Matches how the scene cache was built. |
| `LR` | 0.002, cosine | Reasonable for fine-tuning. |
| `LANG_NUM_MAX` | 32 | Language samples per scene per batch. |

### Data loading

`__getitem__` costs **~20 ms**, and the `DataLoader` previously ran with `num_workers=0`
— every item prepared serially in the main process while the GPU idled, roughly 12
minutes per epoch of pure loading. Background workers overlap loading with compute:

| workers | loading per epoch (8-core machine, measured) |
|---|---|
| 0 | 13.6 min |
| 2 | 9.1 min |
| 4 | **6.2 min** |

`NUM_WORKERS = None` auto-selects `cpu_count - 1`, capped at 4; `PREFETCH_FACTOR = 3`
batches are kept ready per worker, with `persistent_workers` and `pin_memory`. **If
Colab reports OOM, lower `NUM_WORKERS` first** — each worker holds its own copy of the
dataset, which is only affordable because `LAZY_LANG_DATA` keeps that copy near 1.8 GB.

> **Why there is no third "partial preload" mode.** Measured: building one annotation's
> GloVe embeddings costs **0.06 ms — 0.3 %** of the ~20 ms `__getitem__` spends; the
> rest is point-cloud work (subsampling, `unique`, reductions). Precomputing the
> embeddings to disk would cost ~11 GB (float32) to address 0.3 %, and reading 591 KB
> per item back would be *slower* than recomputing it. The loading win is `NUM_WORKERS`,
> not preloading.

---

# 3. Phase B — evaluation

Needs a GPU and a trained checkpoint; trains nothing.

## 3.1 Main evaluation → `predictions.p`

`outputs/<run>/predictions.p` — written by `scripts/ScanRefer_eval.py` — holds the
**per-sample IoU** for all 9,508 val annotations, keyed by `(scene_id, object_id,
ann_id)`:

```python
predictions[scene_id][object_id][ann_id] = {
    "pred_bbox": ndarray (8, 3),
    "gt_bbox":   ndarray (8, 3),
    "iou":       float,
}
```

**Most of phase C is pure post-processing of that one file.** Those analyses need no
GPU, no model and no re-evaluation, and run in seconds.

```bash
python scripts/ScanRefer_eval.py --folder <run> --reference --force \
    --use_color --use_normal --lang_num_max 1
```

Build it with the fusion variant the checkpoint was trained with (`--fusion_variant
original` for the shipped run); see §6.2.

## 3.2 Parse-error propagation — the corruption sweep

No training, evaluation only. Unlike the parse-quality split (§4.3) — which *observes* an
association — this **intervenes**: sample, model and weights are fixed and only the parse
changes, which makes the result causal.

```bash
python experiments/ablation/parsers/corrupt_parse_cache.py --splits val \
    --rates 0.10 0.25 0.50 --mode all                              # CPU, ~15 s
python experiments/ablation/runners/run_parse_corruption.py        # GPU, eval only
python experiments/analysis/parse_error_propagation.py --run-dir outputs/<run>/corruption
```

Corruption modes: `swap` (target replaced with another annotation's), `drop` (all fields
→ "not mentioned", the malformed-output path), `shuffle` (neighbors replaced — a
hallucinated adjacency), `all` (one of the three per annotation, the default).

`corruption_manifest.json` records exactly which annotations were rewritten. That is
what lets the aggregation report three numbers of increasing strength:

- **global accuracy** — the headline curve; understates the effect by (1 − rate).
- **corrupted vs untouched subset** — the untouched subset is a control and should stay
  flat. If it does not, the corruption leaked and the table is not trustworthy.
- **paired flip rate** — for corrupted annotations only, each compared against its own
  baseline outcome. `broke` = right→wrong, `fixed` = wrong→right. McNemar's test is the
  correct test here because the same items are measured twice.

The runner archives each level's `predictions.p` before the next overwrites it —
`scripts/ScanRefer_eval.py` always writes the same filenames, so without archiving only
the last level would survive.

## 3.3 Evaluation-only parser swap

```bash
python experiments/ablation/runners/run_eval_only_parser_swap.py   # GPU, ~40 min
```

Takes the GPT-trained model and feeds it a *different* parser's output at test time,
with training held completely fixed. Separates two things the phase-A parser arms
necessarily conflate: dependence on the parser **during training** versus **at
inference**.

---

# 4. Phase C — post-processing

CPU unless noted. Reads files off disk and writes reports.

| # | Experiment | Script | Device | Time |
|---|---|---|---|---|
| C1 | variant B parse cache | `parsers/run_spacy_parser.py` | CPU | ~1 min |
| C1 | variant D parse cache | `parsers/make_noparse_cache.py` | CPU | ~10 s |
| C1 | corrupted parse caches | `parsers/corrupt_parse_cache.py` | CPU | ~15 s |
| C1 | variant E parse cache | `parsers/run_smalllm_parser.py` | **GPU** (CPU ~8 h) | 30–60 min |
| C2 | scene cache | `ablation/scenes_cache.py` | **GPU** | ~40 min |
| C3 | parser target accuracy | `parsers/eval_parser_target_accuracy.py` | CPU | ~1 min |
| C4 | main results table | `analysis/results_table.py` | CPU | ~10 s |
| C4 | linguistic complexity | `analysis/linguistic_complexity.py` | CPU | ~20 s |
| C4 | parse-quality split | `analysis/parse_quality_split.py` | CPU | ~10 s |
| C4 | failure cases | `analysis/failure_cases.py` | CPU | ~10 s |
| C4 | annotation sheets | `analysis/annotation_sheet.py` | CPU | ~5 s |
| C4 | error taxonomy tally | `analysis/error_taxonomy.py` | CPU | instant |
| C4 | propagation aggregation | `analysis/parse_error_propagation.py` | CPU | ~5 s |
| C4 | seed aggregation | `runners/aggregate_seed_results.py` | CPU | instant |
| C5 | complexity suite | `complexity/*` | CPU + **GPU** | minutes |
| C6 | diagnostics | `diagnostics/*` | CPU (one GPU) | seconds |

## 4.1 Parser target accuracy

How often does each parser recover the ground-truth target noun? This **isolates parser
quality from grounding quality**. Without it, a weak result for variant B is ambiguous —
bad parses, or good parses the fusion network cannot use?

```bash
python experiments/ablation/parsers/eval_parser_target_accuracy.py --splits train \
    --parsed-dir data_parsing/spacy_parsing_tokenized --tag spacy
```

Scored against ScanRefer's `object_name`, train split, 36,665 descriptions:

| Parser | exact | substring | fuzzy | no target |
|---|---|---|---|---|
| **spaCy** (variant B) | 73.12 % | 88.10 % | 89.30 % | 0.00 % |
| **GPT-4o-mini** (variant A) | 82.30 % | 92.42 % | 93.58 % | 0.40 % |
| **LLaMA** (variant C) | **83.53 %** | **92.95 %** | **94.04 %** | 0.29 % |

Read carefully before quoting: most of the residual "errors" for the two LLMs are
**synonym mismatches, not parse failures** — `couch → sofa`, `refrigerator → fridge`,
`trash can → bin`, `nightstand → night stand`. spaCy's errors include genuine failures
(`chair → object`, `door → object`), which is the real quality gap. LLaMA also edges out
GPT-4o-mini on this metric.

spaCy throughput: **~900 descriptions/s on CPU** (val 9,508 in 10.6 s; train 36,665 in
40.8 s) with `en_core_web_sm`. `en_core_web_trf` is selectable via `--model` and is more
accurate but ~30–50× slower.

The script also dumps 20–30 parsed examples for manual review, mixing random draws with
failure cases.

## 4.2 Linguistic complexity

The reported Unique/Multiple split measures *object ambiguity*, not *language
complexity*. These four measures cover the latter:

| Measure | Definition | Source |
|---|---|---|
| `tokens` | description length, quartile bins | ScanRefer `token` field |
| `depth` | dependency-tree depth, quartile bins | spaCy, cached to disk |
| `neighbors` | adjacent-object phrases our parser extracted | comma segments of the `neighbors` field |
| `spatial` | spatial-relation cues in the description | `SPATIAL_PHRASES` ∪ `SPATIAL_PREPS` from `parsers/spacy_parser.py` |

A per-relation-type accuracy table is produced as well, so both the number and the type
of spatial relations are covered.

```bash
python experiments/analysis/linguistic_complexity.py \
    --predictions ours=outputs/<run>/predictions.p \
    --predictions 3DVG-Trans=outputs/3DVG-TRANS-outputs/predictions.p
```

**Statistics note.** The table is binned for readability, but the significance tests are
**not** computed over the bins. A Spearman correlation over
four bin means has n = 4, and scipy's asymptotic p-value degenerates to exactly 0
whenever
|ρ| = 1 — which four points reach easily and which would report a spurious certainty.
Every test therefore runs per-sample: the continuous complexity value against the
per-sample hit indicator, n = 9,508.

With two or more models the script adds a gap column, a **per-bin McNemar** test (the
two models saw the same annotations inside each bin, so each bin's gap is paired and
gets its own p-value), and a **bootstrap CI on the difference between the gap in the
highest and the lowest bin** — the statistic that actually answers "does the advantage
widen", and far better powered than a rank correlation on a three-valued advantage
variable.

### The measured result

| Measure | per-bin gap (pp) | highest − lowest | 95% CI |
|---|---|---|---|
| description length | +1.40, +2.59\*, +4.05\*, +3.15\* | +1.75 pp | [−1.03, +4.01] |
| dependency depth | +2.26\*, +2.16\*, +4.21\* | +1.96 pp | [−0.92, +4.50] |
| adjacent objects | +1.77, +2.37\*, +3.92\*, +3.92 | +2.15 pp | [−9.12, +12.37] |
| spatial-relation words | +1.43, +1.91\*, +4.52\*, +2.11 | +0.68 pp | [−4.53, +5.53] |

\* = individually significant (McNemar, p < 0.05).

The gap is positive in every bin, individually significant in most, and larger in the
complex bins, though every CI on the widening includes zero.

## 4.3 Parse-correct vs parse-wrong split

ScanRefer's `object_name` labels every annotation's parse as correct or not for free.

```bash
python experiments/analysis/parse_quality_split.py \
    --predictions outputs/<run>/predictions.p \
    --parse gpt4o-mini=final_parsing_tokenized \
    --parse spacy=spacy_parsing_tokenized \
    --parse llama=llama_parsing_tokenized_clipped
```

**Read the controlled column, not the raw one.** The raw comparison is confounded:
misparsed descriptions are not a random sample, they skew towards particular classes.
The script therefore also computes the difference *within* each `object_name` class and
pools it with Cochran–Mantel–Haenszel weights, with a CMH significance test.

| parser | raw diff | raw p | class-controlled diff | CMH p |
|---|---|---|---|---|
| GPT-4o-mini | +2.98 pp | 0.17 | **+7.54 pp** | 4.8e-04 |
| spaCy | +1.89 pp | 0.27 | **+5.72 pp** | 5.9e-04 |
| LLaMA-3 | +1.60 pp | 0.46 | **+6.47 pp** | 3.0e-03 |

Class composition was *masking* the effect: the misparsed subset happens to contain
easier classes, so the raw number understates it. Reporting only the raw figure would
have thrown away a significant result — and reporting it without the control would have
been indefensible.

## 4.4 Manual annotation, 200 samples

Target extraction is scored automatically. **Attributes and adjacent objects cannot be**
— ScanRefer has no ground truth for them, and they are two of the three fields the
method depends on. This is the only way to put a number on them.

```bash
python experiments/analysis/annotation_sheet.py --num 200 \
    --parse gpt4o-mini=final_parsing_tokenized \
    --parse spacy=spacy_parsing_tokenized
# ... fill in the five blank columns by hand ...
python experiments/analysis/error_taxonomy.py \
    --sheet gpt4o-mini=outputs/analysis/annotation/annotation_sheet_gpt4o-mini.csv
```

Three properties make the sample defensible:

- **Paired across parsers** — the same annotations for every parser, so the comparison
  is within-item. This matters more than the sample size.
- **Stratified** — `--wrong-fraction 0.4` oversamples automatically-detected target
  errors, because a uniform draw would be ~94 % correct parses and would barely
  constrain the taxonomy. `sampling_manifest.json` records the strata so
  `error_taxonomy.py` can **re-weight back to population rates**. Use the population
  column; quoting the raw sample rate as a dataset rate would be a real error.
- **Two annotators supported** — pass two sheets for one parser and Cohen's kappa per
  field is reported. A manual evaluation with no agreement number is easy to discount.

Taxonomy: `ok`, `wrong_target`, `missed_attribute`, `hallucinated_attribute`,
`missed_neighbor`, `hallucinated_neighbor`, `malformed_output`. Unknown codes and
unreadable cells are reported, not silently coerced.

## 4.4b Adjectives and neighbors, scored automatically

```bash
python experiments/analysis/parse_field_comparison.py \
    --parse gpt4o-mini=final_parsing_tokenized \
    --parse llama=llama_parsing_tokenized_clipped \
    --parse spacy=spacy_parsing_tokenized
```

**Why this exists.** §4.1 scores `target` against `object_name`; §4.4 was meant to cover
the other two fields by hand. That never happened — all four sheets under
`outputs/analysis/annotation/` still have **0 of 200 rows filled**. So `adjectives` and
`neighbors` — the fields TAF and A2F actually consume — had no quantitative validation at
all. This closes that gap **without any ground truth**, which is the only option: ScanRefer
has no annotation for either field and none can be manufactured.

Three signals, on the identical 9,508 val annotations every parser covers (so every
comparison is paired):

| signal | what it catches |
|---|---|
| **coverage** | how often the parser declines the slot (`"not mentioned"`) |
| **faithfulness** | fraction of emitted tokens present in the source description — invention |
| **agreement** | mean per-annotation Jaccard against the other parsers |

Measured (val, n = 9,508):

| parser | adj declined | ngh declined | adj faithful | agreement (adj) |
|---|---|---|---|---|
| GPT-4o-mini | 20.7% | 1.2% | 95.3% | 0.759 |
| LLaMA-3 | 22.1% | 1.7% | 93.3% | 0.762 |
| **spaCy** | **46.3%** | **9.7%** | 99.7% | **0.630** |

Pairwise Jaccard on `adjectives`: GPT↔LLaMA **0.891**, GPT↔spaCy **0.627**,
LLaMA↔spaCy **0.634**. The two LLMs were built independently and agree with each other;
spaCy agrees with neither, and declines the attribute slot more than twice as often.

**Three things to keep in mind when reading this.**

- **Faithfulness is near-tautological for a rule-based parser.** spaCy's 99.7% is not a
  quality win: it can only copy tokens verbatim out of the sentence, so it *cannot* score
  low. A parser emitting nothing would score 100%; it only means something beside
  coverage.
  The script's own generated verdict says exactly this.
- **Agreement is not correctness.** Two parsers can agree and both be wrong. The LLM
  consensus is a reference *band*, not ground truth.
- **Spatial-cue recall does not separate the parsers** (GPT 96.2%, spaCy 94.4%), and spaCy
  emits *more* neighbor phrases (1.30 vs 1.08). The story is coverage and agreement, not
  recall.

## 4.4c Parsers on the hardest descriptions

```bash
python experiments/analysis/complex_sentence_showdown.py --num 10
```

One number per parser over the whole split hides *where* the difference lives: every
parser handles "there is a black chair" identically. This reports both halves —

- **the statistic**: the whole split in complexity quartiles, so the result generalises;
- **the anecdote**: the N hardest descriptions with each parser's output side by side,
  as a figure and as text.

Complexity is spaCy dependency-tree depth, tie-broken by token count, read from
`outputs/analysis/dep_depth_cache.json` (already populated for val by §4.2, so selection
costs nothing). `--rank-by tokens|spatial` switches the criterion.

Attribute slot left empty, by quartile:

| parser | Q1 simplest | Q2 | Q3 | Q4 hardest | Q4 − Q1 |
|---|---|---|---|---|---|
| GPT-4o-mini | 14.9% | 19.6% | 25.3% | 23.1% | +8.2 pp |
| spaCy | 47.6% | 43.6% | 45.0% | 49.1% | **+1.5 pp** |

The reading the script generates: spaCy does **not** degrade faster with complexity — its
slope is *flatter* than the LLMs'. The difference is in the **level, not the slope**: it
starts roughly twice as bad and stays there, and the LLMs degrade toward it without
reaching it.

Both scripts are **CPU-only and read no `predictions.p`** — they measure *parse quality,
not grounding accuracy*. The causal link between a worse parse and a worse box is §3.2's
corruption experiment, which needs a GPU.

## 4.5 Failure cases and the qualitative figure

Nothing previously *selected* failures — `scripts/visualize.py` renders whatever scene
it is pointed at, but choosing which one is the problem.

```bash
python experiments/analysis/failure_cases.py --predictions outputs/<run>/predictions.p --top 6
python experiments/analysis/render_failure_figures.py --dpi 300      # camera-ready
```

Causes are assigned by ordered rules over signals computable offline; earlier rules are
more fundamental. On the current checkpoint:

| cause | share of failures |
|---|---|
| `distractor_confusion` | 65.6 % |
| `unattributed` | 11.4 % |
| `localization_drift` | 7.1 % |
| `parse_target_wrong` | 6.3 % |
| `small_object` | 5.2 % |
| `complex_language` | 4.4 % |

**Two thirds of all failures are picking the wrong *instance* of the right class** —
precisely the failure mode the paper's adjacency reasoning is meant to address. That is
a usable finding for the discussion section, not just a figure.

`render_failure_figures.py` turns each case into a four-panel PNG in
`experiments/analysis/figures/` — room top view with every same-label object outlined,
oblique 3-D close-up, front elevation, and a text panel with the description, the parse
the model was conditioned on, the cause and the IoU. Each case also gets `pred.ply`
(red) and `gt.ply` (green) wireframe boxes, viewable in MeshLab / CloudCompare / Open3D
without the scene mesh.

The strongest panel is `parse_target_wrong_scene0011_00_obj17_ann4.png`: the description
is *"there is a large painting on the wall. this the long skinny table under the
painting…"*, the parser returned `painting` where the target was `table`, and the red
box sits on the wall painting 4.01 m from the green box on the table. **That single
figure shows parse error propagating into grounding failure — the mechanism the
about.**

Boxes come from `predictions.p`, points from
`data/scannet/scannet_data/<scene>_aligned_vert.npy`. The two share a coordinate frame;
this is re-checked for every case at run time and any panel that fails gets a visible
warning stamped on it.

## 4.6 Seed aggregation

```bash
python experiments/ablation/runners/aggregate_seed_results.py --pattern '*ABL-SEED*' --latex
python experiments/ablation/runners/aggregate_seed_results.py --pattern '*ABL-PARSER-*' --group-by tag
```

Reports mean ± sample std with n, and emits a LaTeX row. At n = 1 it says the std is
undefined rather than printing 0.00; at n = 2 it flags the estimate as weak. The
permits dropping to two seeds and forbids dropping to one.

`--paired` additionally loads `scores.p` for two groups and runs McNemar's test on the
same 9,508 annotations — a much sharper instrument than comparing two means with n = 3
when the question is whether a sub-one-point difference is real.

## 4.7 Main results table and baseline comparison

Nothing previously reproduced the paper's headline table from a finished run, or
compared it against a baseline on the standard split.

```bash
python experiments/analysis/results_table.py \
    --predictions ours=outputs/<run>/predictions.p \
    --predictions 3DVG-Trans=outputs/3DVG-TRANS-outputs/predictions.p --latex
```

Reproduces ScanRefer's Unique/Multiple split exactly (**1845 / 7663** on val, verified
against a real run's `scores.p["masks"]`). Two details make a naive reimplementation
wrong: the class compared is the **NYU40 mapping** of `object_name` through
`scannetv2-labels.combined.tsv`, not the raw string (counting raw strings gives
3759/5749), and each *object* contributes its label to the scene once, not each
annotation. `scores.p["masks"]` itself cannot be reused: those arrays are in dataloader
order with no saved `scan_idx`, so they cannot be keyed back to `(scene, object, ann)`.

Comparisons use **McNemar**, because both models are evaluated on the identical
annotation set — an unpaired test discards that structure and is materially less
sensitive at the one-point differences at stake.

| | ours | 3DVG-Trans | diff | McNemar p |
|---|---|---|---|---|
| overall @0.25 | 48.91 | 46.23 | +2.67 | 1.5e-07 |
| overall @0.5 | 35.54 | 30.87 | +4.67 | 8.1e-22 |
| unique @0.25 | 79.62 | 79.51 | +0.11 | 0.96 (n.s.) |
| multiple @0.25 | 41.51 | 38.22 | +3.29 | 1.8e-08 |
| multiple @0.5 | 30.65 | 25.37 | +5.29 | 2.9e-22 |

The advantage is concentrated in **multiple** and is indistinguishable from zero on
**unique** — the pattern adjacency reasoning predicts, and worth stating explicitly.

---

# 5. Complexity: FLOPs, memory, latency

FLOPs, GPU memory and inference latency for the grounding network, plus the per-query
latency and cost of the parsing step.

No pre-existing file was modified to add this suite — it is read-only with respect to the
model and training code.

## 5.1 Prior measurement in the repo

Only parameter counts, from `get_num_params()`. Everything else was missing or unusable:

| Metric | Status before this work |
|---|---|
| FLOPs | One dead line in a `__main__` block (`models/detr/transformer3D.py:560-568`) calling `thop.profile` on a **single decoder layer**, discarding the result. `thop` is not installed. |
| Peak GPU memory | **Nothing.** Zero occurrences of `max_memory_allocated`, `memory_reserved`, `memory_summary`, `nvidia-smi` or NVML anywhere in the repo. |
| Parsing latency / cost | **Nothing for the LLM path.** The GPT-4o-mini parsing script is *not in this repository* — only its outputs. No `openai` import exists anywhere. |
| Inference latency | Exists but unsound: `scripts/ScanRefer_eval.py:214-308` times `model(data)` with `time.time()`, no `torch.cuda.synchronize()` and no warmup, at `batch_size=8`, divided by a hardcoded `9508`. CUDA is asynchronous, so it measures kernel *launch*, not execution. |

## 5.2 Design decisions

### FLOPs via `torch.utils.flop_counter.FlopCounterMode`, not fvcore

The obvious choice is `fvcore.nn.FlopCountAnalysis`. It was rejected because nothing is
installed (`fvcore`, `thop`, `ptflops` are all absent) and because **fvcore is
trace-based**, so it would choke on exactly what this model does: `knn_graph` from
`torch_geometric`, `GCNConv`, the custom deformable attention in the DETR decoder,
data-dependent control flow (`if data_dict["istrain"][0] == 1`), and a `forward()`
taking a **dict** rather than tensors.

`FlopCounterMode` is a `TorchDispatchMode` counter built into torch ≥ 2.1 — it counts
ops as they are actually dispatched, so control flow, custom ops and dict inputs are all
irrelevant to it. **No new dependency.**

It counts matmul / conv / bmm / scaled-dot-product-attention. **Elementwise ops,
softmax, normalisation, and index gathers (ball-query / kNN / FPS) are not counted** —
the usual convention, but the paper must state it. FLOPs = 2 × MACs; both are reported.

**Silent-zero guard:** the script lists every module that *holds parameters but
registered zero FLOPs* under `uncounted_modules_with_params` in the JSON. Check that
list before quoting a total.

### Two variants, always labelled, never conflated

- **`end2end`** — full `RefNet`: PointNet++ → VoteNet → DETR decoder → language →
  fusion. The deployed model, and the number that belongs in the complexity table.
- **`cached`** — `CachedRefNet`: language + fusion only, reading pre-computed detector
  output. Substantially cheaper. Report it *only* to document the ablation protocol,
  never as the headline cost of the method.

### Training memory is a labelled lower bound

Inference memory is eval + `no_grad` + batch 1 — the real single-query scenario.
Training memory is train mode with a backward at the real training batch size, but on a
**surrogate scalar** (`cluster_ref.sum()`), *not* `lib/loss_helper.get_loss`, which
needs a large surface of ground-truth tensors. This under-reports slightly and is
**labelled as such in the JSON** rather than quietly presented as full training memory.

### Latency is warmed up and synchronised

10 warmup iterations discarded, then `torch.cuda.synchronize()` on both sides of every
timed forward, reported as **mean / std / p50 / p95** — not a bare mean. A p95 far above
the mean is itself worth knowing.

### CPU shim (contained, measurement-process only)

`models/lang_module.py:56` calls `.cuda()` unconditionally, so the language branch
cannot execute on a CPU-only machine. Since **FLOPs are hardware-independent and worth
having without a GPU**, `measure_complexity.py` neutralises `Tensor.cuda()` *inside its
own process* when `--cpu` is used. It changes nothing in the repository and nothing
about the arithmetic. Latency measured under the shim is meaningless and is labelled
accordingly.

## 5.3 Commands

```bash
# Step 0 — the deployment question (instant; no GPU, no network). Run this first.
python experiments/complexity/measure_parsing_latency.py --mode offline_check

# Step 1 — spaCy parse latency, variant B (local CPU, seconds)
python experiments/complexity/measure_parsing_latency.py --mode spacy --num_samples 500

# Step 2 — GPT-4o-mini parse latency + cost, variant A (real API calls)
export OPENAI_API_KEY=...
python experiments/complexity/measure_parsing_latency.py --mode gpt --num_samples 100
#   prices change; override and record the date taken
python experiments/complexity/measure_parsing_latency.py --mode gpt --num_samples 100 \
    --price_in 0.150 --price_out 0.600

# Step 3 — FLOPs / memory / latency (GPU box)
python experiments/complexity/measure_complexity.py --variant both \
    --use_color --use_normal --latency_iters 100 --repeat 2
```

Step 2 uses ≥100 **distinct** val descriptions — API latency varies with input length
and content, so repeating one sentence would be misleading.

`--repeat 2` is the stability check: **FLOPs must be identical across runs** (it is
deterministic) and latency mean within a few percent. If latency drifts, something else
is using the GPU or warmup was insufficient — investigate before reporting.

`--use_color --use_normal` are **not optional** — they set `input_feature_dim` (7 here,
so `point_clouds` has 10 channels) and therefore change both FLOPs and memory. They must
match the trained configuration.

Useful variations:

```bash
# only the paper's model
python experiments/complexity/measure_complexity.py --variant end2end --use_color --use_normal

# cross-check synthetic input shapes against a real val sample (loads the 978 MB GloVe pickle)
python experiments/complexity/measure_complexity.py --variant cached --real_sample \
    --use_color --use_normal

# deeper per-module FLOPs breakdown
python experiments/complexity/measure_complexity.py --variant end2end --flop_depth 3 \
    --use_color --use_normal

# FLOPs/params only, no GPU (uses the CPU shim; latency here is meaningless)
python experiments/complexity/measure_complexity.py --variant cached --cpu \
    --latency_iters 5 --warmup 2 --use_color --use_normal
```

Both scripts print a formatted table to stdout **and** write the same numbers to
`outputs/complexity/{complexity_report.json,parsing_latency_report.json}`, so figures
can be pulled into the manuscript table without retyping.

# 6. Diagnostics

Correctness checks on the scene cache and the checkpoint, rather than experiments that
produce a table.

## 6.1 Is the scene cache trustworthy?

```bash
python experiments/diagnostics/audit_scene_cache.py               # CPU, ~5 s
python experiments/diagnostics/validate_scene_cache.py            # GPU, strict
```

`validate_scene_cache.py` is the strict test — end-to-end vs cached Acc@0.25 to 1e-4 —
and it needs CUDA, because `lib/loss_helper.py` and `lib/eval_helper.py` contain 32
hardcoded `.cuda()` calls between them.

`audit_scene_cache.py` answers a weaker question anywhere, with no model: every scene
present, every key the right shape and dtype, nothing non-finite, and then the **recall
ceiling** — the fusion network can only return one of the 256 cached proposals, so
`max_i IoU(proposal_i, gt)` bounds grounding accuracy from above.

| IoU | cache ceiling | reported | headroom |
|---|---|---|---|
| 0.25 | 91.33 % | 48.91 % | +42.4 pp |
| 0.5 | 70.88 % | 35.54 % | +35.3 pp |

141/141 val scenes, 18/18 keys, zero non-finite values. **The cache is not the
bottleneck.**

## 6.2 Does the cached path reproduce the accuracy, without a GPU?

```bash
python experiments/diagnostics/cached_eval_cpu.py --num-samples 128 --fusion-variant original
```

Yes — **the checkpoint is intact and no retraining is needed.** This was an open
question for a while, so the resolution is worth recording.

`outputs/2024-12-18_20-40-38_3DVG-FIXED` was trained against the **older**
`MatchModule`, preserved in `models/match_module original.py`. Loading it into today's
`models/match_module.py` leaves 178 `lang.*` / `match.*` tensors missing, because the
current file adds `self_attn_post`, four more GCN layers, and a third layer in every
attention stack (126 keys → 274). The decisive fingerprint is the naming of the three
cross-attentions: the original emits `match.tgt_cross_attn.attention.fc_q.weight`
(un-indexed), the current emits `match.tgt_cross_attn.0.…`. **The checkpoint contains
the un-indexed form**, and all three `.pth` files load into the original module under
`strict=True` with 126/126 tensors and zero shape mismatches.

So the mismatch was never a damaged checkpoint — it was the architecture moving on.
Select the matching head instead of retraining:

```bash
--fusion_variant original      # or ABLATION.FUSION_VARIANT in ablation_config.py
```

`ABLATION.STRICT_CHECKPOINT` (on by default) enforces this. `scripts/ScanRefer_eval.py`
loads with `strict=False`, so without it those 178 tensors would stay randomly
initialised and a plausible-looking but meaningless accuracy would be reported. On a
paired subset — 816 annotations over 15 scenes, same annotations both sides — the two
heads score 51.96 % vs 49.02 % (−2.94 pp, 82.4 % per-item agreement), the residual being
deterministic vs random point subsampling rather than a weight problem.

---

# 7. Environment and troubleshooting

## `pyg-lib` is mandatory for the model

`models/match_module.py` calls `torch_geometric.nn.knn_graph` inside the forward pass,
so **without `pyg-lib` the model cannot be built at all** — not training, not
evaluation, not the FLOPs measurement. That is a pre-existing property of the codebase,
not of these scripts. Every CPU analysis in §4 is unaffected and runs without it.

`pyg-lib` is not on PyPI; it installs only from the PyG index keyed to the exact torch
build:

```bash
# torch.__version__ already carries the CUDA suffix (e.g. "2.11.0+cu128"), which is
# exactly how the PyG index is keyed -- so read it rather than composing it by hand.
TORCH=$(python -c "import torch; print(torch.__version__)")
pip install pyg_lib -f "https://data.pyg.org/whl/torch-${TORCH}.html"
```

PyG 2.8 routes `knn_graph` through `pyg_lib`, so `torch_cluster` is not required.
Verified on Colab: torch `2.11.0+cu128` resolves `pyg_lib-0.8.0+pt211cu128`. See
`req/req_env.txt`.

## `pointnet2_ops` and the `compute_37` build failure

If building the CUDA extension fails with

```
nvcc fatal : Unsupported gpu architecture 'compute_37'
```

the cause is the architecture list: `compute_37` is Kepler, which **CUDA 12 removed
entirely**. `pointnet/pointnet2_ops_lib/setup.py` used to hardcode that list *and assign it over*
`TORCH_CUDA_ARCH_LIST`, so exporting the variable could not work around it. It now
honours an explicit `TORCH_CUDA_ARCH_LIST` and otherwise picks a default from the
detected CUDA version (CUDA ≥ 12 → Pascal and newer). Build with `ninja` installed, or
the build falls back to a single-threaded backend and takes many minutes:

```bash
pip install ninja
rm -rf pointnet/pointnet2_ops_lib/build pointnet/pointnet2_ops_lib/pointnet2_ops.egg-info
pip install ./pointnet/pointnet2_ops_lib
```

Also worth checking: `python -c "import pointnet2_ops"`. If it fails, the model silently
falls back to the Python PointNet++ implementation, which changes latency substantially
(though not FLOPs). `measure_complexity.py` records which implementation was active in
the `environment` block (`pointnet_impl`, `pointnet2_ops_cuda_ext`).

## Host RAM is the real constraint, not GPU memory

`ScannetReferenceDataset._tranform_des` builds a `(MAX_DES_LEN=126, 300)` **float64**
array per annotation, twice (`lang` and `lang_main`), plus a `(7+17+75, 300)` array per
annotation for the parsed fields when `detection=False`:

| split | annotations | `lang` | `lang_main` | parsed | total |
|---|---|---|---|---|---|
| train | 36,665 | 11.1 GB | 11.1 GB | 8.7 GB | **30.9 GB** |
| val | 9,508 | 2.9 GB | 2.9 GB | 2.3 GB | **8.0 GB** |

This is pre-existing behaviour, unrelated to the ablation work, but it decides where
each job can run:

- **`scenes_cache.py`** sidesteps it entirely: it keeps only *one description per scene*
  (the detection branch never reads the language tensors). Measured on train: **2.1 GB
  peak RSS, 6 s** for the dry run.
- **Training runs** genuinely need all annotations. This is what killed the first Colab
  run: every training command died with **exit 247** (SIGKILL — the host OOM killer, not
  CUDA) about 40 s in, right after `train on 36665 samples` and before the first batch.
  Note that a smaller batch size cannot help: the memory is spent building the dataset,
  before any batch is formed, and it is host RAM rather than GPU memory.

**Fixed** by `ABLATION.LAZY_LANG_DATA` (default `True`), implemented as `_LazyLangData`
in `experiments/ablation/cached_scenes.py`, so `lib/dataset.py` is still untouched. The
embeddings are a pure function of the tokens and the GloVe table, both already resident,
so each annotation is rebuilt on access behind an LRU instead of being precomputed:

| | peak RSS | build time |
|---|---|---|
| eager (original) | ~36 GB for train+val | — |
| lazy (`_LazyLangData`) | **1.77 GB**, full train split | 2.7 s |

Cost is now flat in dataset size — the full 36,665-annotation train split costs the same
as 20 val scenes — and access runs at ~2,400 annotations/s. Verified elementwise against
the eager path: every annotation's five arrays identical, and `__getitem__` equal across
all 40 array keys. Pass `--no_lazy_lang_data` to restore the original behaviour on a
machine with ~40 GB to spare.

## A GPU that reports `is_available() == True` may still be unusable

A card whose compute capability predates the installed torch build reports `True` and
then fails on the first real operation. `run_analysis_colab.ipynb` probes this properly
by launching an actual kernel. Every CPU analysis is unaffected.

## Other dependencies

`transformers` is needed only for variant E (and `sentencepiece` only for its Flan-T5
option — `--list-models` shows which models need what, and the script checks before
downloading). `openai` only for the GPT latency measurement. `pandas` is not used
anywhere — the annotation sheets use the standard library `csv` module.

---

# 8. Full command list, in dependency order

```bash
# ---- CPU, no model needed -------------------------------------------------
python experiments/ablation/parsers/run_spacy_parser.py --splits train val          # variant B
python experiments/ablation/parsers/make_noparse_cache.py --splits train val --both # variant D
python experiments/ablation/parsers/corrupt_parse_cache.py --splits val --rates 0.10 0.25 0.50

python experiments/ablation/parsers/eval_parser_target_accuracy.py --splits train \
    --parsed-dir data_parsing/final_parsing_tokenized --tag gpt4o-mini   # repeat per parser

python experiments/complexity/measure_parsing_latency.py --mode offline_check
python experiments/complexity/measure_parsing_latency.py --mode spacy --num_samples 500

python experiments/diagnostics/audit_scene_cache.py

# ---- GPU: caches ----------------------------------------------------------
python experiments/ablation/parsers/run_smalllm_parser.py --list-models             # pick a model
python experiments/ablation/parsers/run_smalllm_parser.py --splits train val \
    --model qwen2.5 --device cuda                                                   # variant E
python experiments/ablation/scenes_cache.py --splits train val --use_color --use_normal
python experiments/diagnostics/validate_scene_cache.py --use_color --use_normal     # must pass

# ---- GPU: phase A, retraining (hours each) --------------------------------
python experiments/ablation/runners/run_no_copypaste.py                             # A1
python experiments/ablation/runners/run_parser_gpt.py                               # A2
python experiments/ablation/runners/run_parser_spacy.py                             # A3
python experiments/ablation/runners/run_parser_llama.py                             # A4
python experiments/ablation/runners/run_parser_none.py                              # A5
python experiments/ablation/runners/run_parser_smalllm.py                           # A6
python experiments/ablation/runners/run_seeds.py                                    # A7
python experiments/ablation/runners/sweep_attention_layers.py                       # A8

# ---- GPU: phase B, evaluation ---------------------------------------------
python scripts/ScanRefer_eval.py --folder <run> --reference --force \
    --use_color --use_normal --lang_num_max 1 --fusion_variant original    # per run
python experiments/ablation/runners/run_parse_corruption.py
python experiments/ablation/runners/run_eval_only_parser_swap.py
python experiments/complexity/measure_complexity.py --variant both --use_color --use_normal

# ---- CPU: phase C, after predictions.p exists -----------------------------
B=outputs/3DVG-TRANS-outputs/predictions.p
python experiments/analysis/results_table.py --predictions ours=outputs/<run>/predictions.p \
    --predictions 3DVG-Trans=$B --latex
python experiments/analysis/linguistic_complexity.py \
    --predictions ours=outputs/<run>/predictions.p --predictions 3DVG-Trans=$B
python experiments/analysis/parse_quality_split.py --predictions outputs/<run>/predictions.p \
    --parse gpt4o-mini=final_parsing_tokenized --parse spacy=spacy_parsing_tokenized
python experiments/analysis/parse_error_propagation.py --run-dir outputs/<run>/corruption
python experiments/analysis/failure_cases.py --predictions outputs/<run>/predictions.p --top 6
python experiments/analysis/render_failure_figures.py --dpi 300
python experiments/analysis/annotation_sheet.py --num 200 --parse gpt4o-mini=final_parsing_tokenized
python experiments/analysis/error_taxonomy.py --sheet gpt4o-mini=<filled csv>
python experiments/ablation/runners/aggregate_seed_results.py --pattern '*ABL-SEED*' --latex
python experiments/diagnostics/cached_eval_cpu.py --num-samples 128 --fusion-variant original
```
