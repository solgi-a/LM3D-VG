
import ast
import os
import sys
import textwrap

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MATCH_MODULE_SRC = os.path.join(REPO, "models", "match_module.py")

NUM_PROPOSALS = 256
HIDDEN_SIZE = 128
NUM_HEADS = 4
ROOM = (7.0, 7.0, 2.6)
SEED = 0


def _function_node(tree, class_name, func_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == func_name:
                    return child
    return None


def _assign_targets(node):
    found = {}
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                found.setdefault(target.id, []).append(ast.dump(stmt.value))
    return found


def check_source_drift():
    with open(MATCH_MODULE_SRC) as handle:
        tree = ast.parse(handle.read())

    dist_cal = _function_node(tree, "MatchModule", "dist_cal")
    forward = _function_node(tree, "MatchModule", "forward")
    if dist_cal is None or forward is None:
        print("  ! could not locate MatchModule.dist_cal / .forward -- skipping drift check")
        return None

    a, b = _assign_targets(dist_cal), _assign_targets(forward)
    shared = ["dist", "dist_weights", "norm"]
    disagreements = []
    for name in shared:
        if name not in a or name not in b:
            disagreements.append(f"{name}: present in only one copy")
            continue
        for expression in a[name]:
            if expression not in b[name]:
                disagreements.append(f"{name}: dist_cal form absent from forward()")

    if disagreements:
        print("  DRIFT between MatchModule.dist_cal and the inline copy in forward():")
        for item in disagreements:
            print(f"    - {item}")
        print("    The two code paths no longer compute the same bias. Fix before")
        print("    quoting any number from this script.")
        return False

    print("  dist_cal and the duplicated block inside forward() agree (AST-identical")
    print("  for dist, norm and dist_weights).")
    return True


def check_attention_mask_usage():
    with open(MATCH_MODULE_SRC) as handle:
        tree = ast.parse(handle.read())

    total, masked = 0, 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {kw.arg for kw in node.keywords}
        if "attention_weights" in keywords:
            total += 1
            if "attention_mask" in keywords:
                masked += 1
    return total, masked


def real_dist_weights(centers):
    from models.match_module import MatchModule
    return MatchModule.dist_cal(None, centers)


def describe_channels(dist_weights, dist):
    labels = [
        "0  normalised 1/(d+1e-2)   proximity, row-normalised",
        "1  -d                      raw negative distance, metres",
        "2  zeros                   inert",
        "3  zeros                   inert",
    ]
    rows = []
    for channel in range(dist_weights.shape[1]):
        values = dist_weights[:, channel]
        rows.append((labels[channel], float(values.min()), float(values.max()),
                     float(values.mean()), float(values.std())))
    return rows


def spearman_rows(attention, distance):
    def rank(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        ranks[order] = np.arange(len(values), dtype=float)
        sorted_values = values[order]
        start = 0
        for index in range(1, len(values) + 1):
            if index == len(values) or sorted_values[index] != sorted_values[start]:
                if index - start > 1:
                    ranks[order[start:index]] = ranks[order[start:index]].mean()
                start = index
        return ranks

    rhos = []
    for row in range(attention.shape[0]):
        x, y = rank(attention[row]), rank(distance[row])
        x = x - x.mean()
        y = y - y.mean()
        denominator = np.sqrt((x ** 2).sum() * (y ** 2).sum())
        if denominator > 0:
            rhos.append(float((x * y).sum() / denominator))
    return float(np.mean(rhos)) if rhos else float("nan")


def quartile_ratio(attention, distance):
    near, far = [], []
    for row in range(attention.shape[0]):
        order = np.argsort(distance[row])
        cut = max(1, len(order) // 4)
        near.append(attention[row][order[:cut]].mean())
        far.append(attention[row][order[-cut:]].mean())
    near, far = float(np.mean(near)), float(np.mean(far))
    return near / far if far > 0 else float("inf"), near, far


def attention_maps(features, dist_weights, way, attention_module):
    batch, length = features.shape[:2]
    module = attention_module
    q = module.fc_q(features).view(batch, length, module.h, module.d_k).permute(0, 2, 1, 3)
    k = module.fc_k(features).view(batch, length, module.h, module.d_k).permute(0, 2, 3, 1)

    logits = torch.matmul(q, k) / np.sqrt(module.d_k)
    raw = logits.clone()

    if dist_weights is not None:
        if way == "mul":
            logits = logits * dist_weights
        elif way == "add":
            logits = logits + dist_weights
        else:
            raise NotImplementedError(way)

    return torch.softmax(logits, -1), raw


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_grad_enabled(False)

    print("=" * 86)
    print("Eq. 7 -- distance bias in the fusion self-attention")
    print("=" * 86)

    print("\n[1] Source-level facts\n" + "-" * 86)
    check_source_drift()

    total_calls, masked_calls = check_attention_mask_usage()
    print(f"  self-attention calls carrying attention_weights : {total_calls}")
    print(f"  ... of which also pass attention_mask           : {masked_calls}")
    if masked_calls == 0:
        print("  => no attention_mask is ever supplied, so the -inf invalid-proposal")
        print("     removal in attention.py is dead code on this path. If the paper")
        print("     describes masking invalid proposals inside Eq. 7, it describes")
        print("     something the code does not do.")

    print("\n[2] The bias tensor, from MatchModule.dist_cal\n" + "-" * 86)
    centers = torch.rand(1, NUM_PROPOSALS, 3) * torch.tensor(ROOM)
    dist_weights, way = real_dist_weights(centers)

    pairwise = torch.cdist(centers, centers)[0].numpy()
    print(f"  returned shape : {tuple(dist_weights.shape)}   (batch, heads, N, N)")
    print(f"  returned way   : {way!r}   -> attention.py takes the "
          f"'{'att + w' if way == 'add' else 'att * w'}' branch")
    print(f"  proposal spread: {pairwise.min():.2f} m to {pairwise.max():.2f} m, "
          f"mean {pairwise.mean():.2f} m")

    print("\n  channel                                          min       max      mean       std")
    for label, low, high, mean, std in describe_channels(dist_weights, pairwise):
        print(f"    {label:<44} {low:8.4f}  {high:8.4f}  {mean:8.4f}  {std:8.4f}")

    from models.transformer.attention import ScaledDotProductAttention

    attention_module = ScaledDotProductAttention(
        d_model=HIDDEN_SIZE, d_k=HIDDEN_SIZE // NUM_HEADS,
        d_v=HIDDEN_SIZE // NUM_HEADS, h=NUM_HEADS)
    attention_module.eval()

    features = torch.randn(1, NUM_PROPOSALS, HIDDEN_SIZE)
    biased, raw_logits = attention_maps(features, dist_weights, way, attention_module)
    unbiased, _ = attention_maps(features, None, way, attention_module)

    logit_std = float(raw_logits.std())
    print("\n[3] Is each channel large enough to matter?\n" + "-" * 86)
    print(f"  spread of the raw logits q.k/sqrt(d_k)          : std {logit_std:.4f}")
    for channel in range(dist_weights.shape[1]):
        bias_std = float(dist_weights[:, channel].std())
        share = bias_std / logit_std if logit_std > 0 else float("inf")
        if share < 0.05:
            note = "INERT -- swamped by the logits"
        elif share < 0.5:
            note = "minor"
        else:
            note = "DOMINANT -- drives the softmax"
        print(f"  head {channel}: bias std {bias_std:8.4f}  "
              f"= {share:7.3f} x logit std   {note}")

    print("\n[4] Does attention fall off with distance?\n" + "-" * 86)
    print("  rho  = mean per-query Spearman(attention, distance); negative = near wins")
    print("  near/far = mean attention on nearest quartile / farthest quartile\n")
    print("  head  |        with bias        |       no bias (control)")
    print("        |    rho     near/far     |    rho     near/far")
    print("  " + "-" * 62)

    per_head = {}
    for head in range(NUM_HEADS):
        biased_head = biased[0, head].numpy()
        unbiased_head = unbiased[0, head].numpy()
        rho_b = spearman_rows(biased_head, pairwise)
        rho_u = spearman_rows(unbiased_head, pairwise)
        ratio_b, near_b, far_b = quartile_ratio(biased_head, pairwise)
        ratio_u, _, _ = quartile_ratio(unbiased_head, pairwise)
        per_head[head] = {"rho": rho_b, "ratio": ratio_b,
                          "rho_control": rho_u, "ratio_control": ratio_u}
        print(f"    {head}   |  {rho_b:+.4f}   {ratio_b:8.3f}x   |  "
              f"{rho_u:+.4f}   {ratio_u:8.3f}x")

    print("\n[5] Counterfactual -- +d instead of -d (Eq. 7 as printed)\n" + "-" * 86)
    flipped = dist_weights.clone()
    flipped[:, 1] = -flipped[:, 1]
    counterfactual, _ = attention_maps(features, flipped, way, attention_module)

    head1_real = spearman_rows(biased[0, 1].numpy(), pairwise)
    head1_flipped = spearman_rows(counterfactual[0, 1].numpy(), pairwise)
    ratio_real, _, _ = quartile_ratio(biased[0, 1].numpy(), pairwise)
    ratio_flipped, _, _ = quartile_ratio(counterfactual[0, 1].numpy(), pairwise)
    print(f"  head 1 as implemented (-d) : rho {head1_real:+.4f}   "
          f"near/far {ratio_real:10.3f}x")
    print(f"  head 1 as printed     (+d) : rho {head1_flipped:+.4f}   "
          f"near/far {ratio_flipped:10.3f}x")

    print("\n" + "=" * 86)
    print("VERDICT")
    print("=" * 86)

    inert = [h for h in range(NUM_HEADS)
             if float(dist_weights[:, h].std()) / logit_std < 0.05]
    active = [h for h in range(NUM_HEADS) if h not in inert]
    near_favoured = [h for h in active if per_head[h]["ratio"] > 1.0]

    lines = []
    if near_favoured:
        lines.append(
            f"The code favours NEAR objects. Heads {near_favoured} put "
            f"{max(per_head[h]['ratio'] for h in near_favoured):.1f}x more attention on "
            f"the nearest quarter of proposals than the farthest quarter, and the "
            f"control run without the bias shows no such preference "
            f"({per_head[near_favoured[0]]['ratio_control']:.2f}x). The reviewers' "
            f"conclusion -- that farther objects gain weight -- does NOT hold for the "
            f"implementation.")
        lines.append(
            "So this is a writing defect, not a modelling defect. Eq. 7 in the paper "
            "must be corrected to show the negative sign that the code applies "
            "(`att - d`, not `att + d`). Do not change match_module.py; change the "
            "equation, and say in the response letter that the implementation was "
            "always the intended one and the typeset equation dropped the sign.")
    else:
        lines.append(
            "The code favours FAR objects, exactly as the reviewers read it. This is a "
            "real bug, not a typesetting slip. Every number in the paper was produced "
            "by this code, so the equation cannot simply be corrected -- report the "
            "behaviour honestly and decide whether to retrain.")

    if inert:
        lines.append(
            f"Separately: head(s) {inert} contribute a bias whose standard deviation is "
            f"under 5% of the logit spread, because dist_cal row-normalises "
            f"1/(d+1e-2) so its entries are O(1/N) with N={NUM_PROPOSALS} while the "
            f"logits are O(1). Whatever Eq. 7 claims that channel does, it does almost "
            f"nothing. Heads 2 and 3 are literally zeros. Only head 1 carries a "
            f"working distance prior -- state that the prior is applied on one of four "
            f"heads rather than implying it shapes all of them.")

    lines.append(
        "The normalisation in dist_cal sums over dim=2 (the query axis) while softmax "
        "normalises over dim=-1 (the key axis), so the row-normalisation does not make "
        "each attention row sum to one. Worth one sentence in the paper so a reader "
        "reproducing Eq. 7 does not expect a stochastic matrix.")

    if masked_calls == 0:
        lines.append(
            "No attention_mask is passed anywhere in MatchModule, so invalid proposals "
            "are not removed inside the attention. If the paper's Eq. 7 includes a "
            "removal term, delete it or move it to where it actually happens.")

    for number, line in enumerate(lines, 1):
        print(f"\n{number}. " + textwrap.fill(line, 84, subsequent_indent="   "))

    out_dir = os.path.join(REPO, "outputs", "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        edges = np.linspace(pairwise.min(), pairwise.max(), 25)
        centres_x = 0.5 * (edges[:-1] + edges[1:])
        bin_index = np.clip(np.digitize(pairwise, edges) - 1, 0, len(centres_x) - 1)

        def binned(attention):
            means = np.zeros(len(centres_x))
            for b in range(len(centres_x)):
                selected = attention[bin_index == b]
                means[b] = selected.mean() if selected.size else np.nan
            return means

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for head in range(NUM_HEADS):
            axes[0].plot(centres_x, binned(biased[0, head].numpy()),
                         marker="o", markersize=3, label=f"head {head}")
        axes[0].plot(centres_x, binned(unbiased[0, 0].numpy()), "k--",
                     linewidth=1, label="no bias (control)")
        axes[0].set_title("As implemented (-d)")
        axes[0].set_xlabel("distance between proposals (m)")
        axes[0].set_ylabel("mean attention weight")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=8)

        axes[1].plot(centres_x, binned(biased[0, 1].numpy()),
                     marker="o", markersize=3, label="head 1, as implemented (-d)")
        axes[1].plot(centres_x, binned(counterfactual[0, 1].numpy()),
                     marker="s", markersize=3, label="head 1, as printed (+d)")
        axes[1].plot(centres_x, binned(unbiased[0, 1].numpy()), "k--",
                     linewidth=1, label="no bias (control)")
        axes[1].set_title("Sign of the distance term")
        axes[1].set_xlabel("distance between proposals (m)")
        axes[1].set_yscale("log")
        axes[1].legend(fontsize=8)

        figure.tight_layout()
        figure_path = os.path.join(out_dir, "eq7_distance_bias.png")
        figure.savefig(figure_path, dpi=170)
        plt.close(figure)
        print(f"\nwrote {os.path.relpath(figure_path, REPO)}")
    except Exception as error:
        print(f"\n(figure skipped: {error})")

    report = {
        "way": way,
        "num_proposals": NUM_PROPOSALS,
        "logit_std": logit_std,
        "channel_std": [float(dist_weights[:, c].std()) for c in range(NUM_HEADS)],
        "inert_heads": inert,
        "per_head": per_head,
        "counterfactual_head1": {"implemented_rho": head1_real,
                                 "printed_rho": head1_flipped,
                                 "implemented_near_far": ratio_real,
                                 "printed_near_far": ratio_flipped},
        "attention_calls": total_calls,
        "attention_calls_with_mask": masked_calls,
    }
    import json
    report_path = os.path.join(out_dir, "eq7_distance_bias.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"wrote {os.path.relpath(report_path, REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
