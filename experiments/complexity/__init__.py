"""
Computational-complexity measurement suite.

The paper's complexity table reports parameter counts and a single latency figure. This
package produces the rest:

    measure_complexity.py       FLOPs, peak GPU memory, grounding-network latency
    measure_parsing_latency.py  sentence-parsing latency + token cost, and the
                                offline-vs-online deployment statement

Both write JSON next to their stdout table. Nothing here imports from or modifies the
training/eval path.
"""
