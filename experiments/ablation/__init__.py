"""
Everything added for the revision lives in this package.

The pre-existing scripts carry only a handful of changed lines, every one tagged
``# ABLATION``; all new logic sits here, so the revision can be reviewed in isolation.

Layout
------
    ablation_config.py            THE switchboard -- every experiment flag lives here
    ablation_hooks.py             picks dataset/model class; returns the originals when off
    cached_scenes.py              cache format + CachedSceneDataset
    cached_refnet.py              CachedRefNet: language + fusion, no detection branch
    scenes_cache.py               build the scene cache          [entry point]
    validate_scene_cache.py       check the cache == end-to-end  [entry point]
    parsing/spacy_parser.py       rule-based parser (variant B)
    tokenize_parse.py             shared tokenizer + 7/17/75 caps
    run_spacy_parser.py           produce the variant-B parse    [entry point]
    clip_parse_cache.py           enforce token caps on a cache  [entry point]
    eval_parser_target_accuracy.py  score a parser               [entry point]
    runners/run_*.py              one independent runner per ablation

Only ``ablation_config`` is imported by the original code paths, and it has no third-party
dependencies so it can be inspected without the training environment.
"""
