"""Every experiment written for the IMAVIS revision (IMAVIS-D-26-00703).

Four sub-packages, grouped by what the code does rather than by when it was written:

    ablation/      experiments that rebuild a cache or retrain the network
                     parsers/   build the parse caches every variant is fed from
                     runners/   one file per training/evaluation run
    analysis/      post-hoc work on a finished run; reads predictions.p, needs no GPU
    complexity/    FLOPs, GPU memory, latency, parsing cost
    diagnostics/   does the implementation match what the paper claims?

Nothing here is imported by the model. The dependency runs the other way: ``lib/``,
``models/`` and ``scripts/`` import ``experiments.ablation.ablation_config`` for the
handful of flags that switch an experiment on, and that is the whole coupling.

Run everything from the repository root, either through ``run_analysis_colab.ipynb`` or as
``python experiments/<sub-package>/<script>.py``. See ``experiments/experiments.md``.
"""
