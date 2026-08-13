"""
Post-hoc analyses.

    EVERYTHING IN THIS PACKAGE RUNS ON CPU.

Nothing here loads or runs the model. Every script consumes artifacts that already exist
-- ``outputs/<run>/predictions.p``, the parse caches under ``data_parsing/``, and
ScanRefer's JSON -- and produces tables, plots and reports.

    common.py                     shared loaders, statistics, PLY writer
    linguistic_complexity.py      accuracy split by linguistic complexity
    parse_quality_split.py        grounding accuracy vs parse correctness
    parse_error_propagation.py    accuracy vs deliberate parse corruption
    annotation_sheet.py           export 200 samples for manual labelling
    error_taxonomy.py             tally the filled annotation sheet
    failure_cases.py              failure selection + cause attribution

Separate from ``ablation/``, which changes what the model is trained on; this package only
reads what the model already produced.
"""
