"""Matplotlib heatmap visualizations for token relevance."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from explainllm.utils import to_numpy, clean_token, clean_token_list
from explainllm.viz.stepwise import get_stepwise_relevance


def plot_relevance_heatmap(
    result: Dict,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "YlOrRd",
    title: str = "Token Relevance Heatmap  (full context)",
    save_path: str = "relevance_heatmap.png",
    dpi: int = 180,
    annot: bool = True,
    fmt: str = ".2f",
) -> str:
    """
    Full heatmap: rows = generated tokens, cols = full context at each step.

    Cells beyond the valid context at each step are greyed out.
    Column headers are colour-coded: red = prompt, blue = generated.
    """
    sw = get_stepwise_relevance(result)
    matrix = sw["matrix"]
    mask = sw["mask"]
    prompt_len = sw["prompt_len"]
    all_tokens = sw["all_tokens"]
    output_tokens = sw["output_tokens"]
    n_out, max_ctx = matrix.shape

    if n_out == 0 or max_ctx == 0:
        raise ValueError("No generation steps found; cannot plot heatmap.")

    col_labels = list(all_tokens[:max_ctx])
    while len(col_labels) < max_ctx:
        col_labels.append("")

    if figsize is None:
        w = max(10, max_ctx * 0.95 + 4)
        h = max(4, n_out * 0.85 + 3.5)
        figsize = (w, h)

    if max_ctx * n_out > 200:
        annot = False

    plot_matrix = matrix.copy()
    plot_matrix[~mask] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    current_cmap = plt.get_cmap(cmap).copy()
    current_cmap.set_bad(color="#f0f0f0")

    sns.heatmap(
        plot_matrix,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=[f"\u2192 {t}" for t in output_tokens],
        cmap=current_cmap,
        annot=annot if annot else False,
        fmt=fmt if annot else "",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Relevance Score", "shrink": 0.75},
        vmin=0.0,
        mask=~mask,
    )

    for i in range(n_out):
        for j in range(max_ctx):
            if not mask[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i), 1, 1,
                        fill=True,
                        facecolor="#e8e8e8",
                        edgecolor="white",
                        linewidth=0.5,
                    )
                )

    x_tick_labels = ax.get_xticklabels()
    for idx, label in enumerate(x_tick_labels):
        if idx < prompt_len:
            label.set_color("#c0392b")
            label.set_fontweight("bold")
        else:
            label.set_color("#2471a3")
            label.set_fontweight("bold")
            label.set_fontstyle("italic")

    if prompt_len < max_ctx:
        ax.axvline(
            x=prompt_len, color="#555555", linewidth=2, linestyle="--", alpha=0.7
        )
        ax.text(
            prompt_len + 0.15, -0.4,
            "\u25C0 prompt | generated \u25B6",
            fontsize=8, color="#555", ha="left", va="top",
            transform=ax.get_xaxis_transform(),
        )

    for i in range(n_out):
        ctx_len = int(mask[i].sum())
        if ctx_len < max_ctx:
            ax.plot(
                [ctx_len, ctx_len], [i, i + 1],
                color="#888", linewidth=1.5, alpha=0.5,
            )
            ax.plot(
                [ctx_len, max_ctx], [i + 0.5, i + 0.5],
                color="#ccc", linewidth=0.5, linestyle=":", alpha=0.5,
            )

    ax.set_xlabel(
        "Context Tokens  (red = prompt, blue = generated)",
        fontsize=11, labelpad=10,
    )
    ax.set_ylabel("Generated Tokens", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=10)

    legend_handles = [
        mpatches.Patch(facecolor="#c0392b", label="Prompt token", alpha=0.7),
        mpatches.Patch(facecolor="#2471a3", label="Generated token", alpha=0.7),
        mpatches.Patch(
            facecolor="#e8e8e8", edgecolor="#ccc", label="Not in context yet"
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=8,
        framealpha=0.9, edgecolor="#ccc",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap saved -> {save_path}")
    return save_path


def plot_single_step_heatmap(
    result: Dict,
    step: int = 0,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: str = "relevance_step_bar.png",
    dpi: int = 180,
    top_k: Optional[int] = None,
) -> str:
    """
    Horizontal bar chart for a single step.

    Bars are colour-coded: red/warm = prompt token, blue/cool = generated token.
    """
    sw = get_stepwise_relevance(result)
    if step >= len(sw["steps"]):
        raise IndexError(f"Step {step} not available (max {len(sw['steps'])-1}).")

    s = sw["steps"][step]
    ctx_tokens = s["context_tokens"]
    prompt_len = s["prompt_len"]
    rel = np.array(s["relevance_vector"])

    order = np.argsort(rel)[::-1]
    if top_k:
        order = order[:top_k]

    labels = []
    is_gen = []
    for idx in order:
        tok = ctx_tokens[idx] if idx < len(ctx_tokens) else "?"
        tag = "G" if idx >= prompt_len else "P"
        labels.append(f"[{tag}:{idx}] {tok}")
        is_gen.append(idx >= prompt_len)

    values = rel[order]

    if figsize is None:
        figsize = (11, max(3.5, len(order) * 0.5 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    cmap_prompt = plt.get_cmap("YlOrRd")
    cmap_gen = plt.get_cmap("YlGnBu")
    max_val = values.max() + 1e-10

    colours = []
    for v, g in zip(values, is_gen):
        norm_v = v / max_val
        if g:
            colours.append(cmap_gen(0.3 + 0.7 * norm_v))
        else:
            colours.append(cmap_prompt(0.3 + 0.7 * norm_v))

    bars = ax.barh(
        range(len(values)), values, color=colours,
        edgecolor="white", linewidth=0.6, height=0.75,
    )
    ax.set_yticks(range(len(values)))
    ax.set_yticklabels(labels, fontsize=9.5, fontfamily="monospace")
    ax.invert_yaxis()

    for idx, (label_obj, g) in enumerate(zip(ax.get_yticklabels(), is_gen)):
        label_obj.set_color("#2471a3" if g else "#c0392b")
        label_obj.set_fontweight("bold")

    ax.set_xlabel("Relevance", fontsize=11)

    gen_count = sum(is_gen)
    prompt_count = len(is_gen) - gen_count
    ax.set_title(
        f"Step {step}: Generating \u2192 '{s['generated_token']}'  "
        f"({prompt_count}P + {gen_count}G tokens in context)",
        fontsize=13, fontweight="bold", pad=12,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max_val * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center", fontsize=9, fontfamily="monospace",
        )

    ax.xaxis.grid(True, alpha=0.2, linestyle="--")
    ax.set_axisbelow(True)

    legend_handles = [
        mpatches.Patch(facecolor=cmap_prompt(0.7), label="Prompt token"),
        mpatches.Patch(facecolor=cmap_gen(0.7), label="Generated token"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower right", fontsize=9,
        framealpha=0.9, edgecolor="#ccc",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Step bar chart saved -> {save_path}")
    return save_path


def plot_layer_relevance_heatmap(
    result: Dict,
    step: int = 0,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "magma",
    save_path: str = "layer_relevance_heatmap.png",
    dpi: int = 180,
) -> str:
    """
    Per-layer relevance heatmap for a given step.

    Rows = transformer layers, Cols = full context tokens at that step.
    """
    relevance_per_token = result.get("relevance_per_token", [])
    if step >= len(relevance_per_token):
        raise IndexError(f"Step {step} not available.")

    layer_rel = relevance_per_token[step].get("layer_relevance", [])
    if not layer_rel:
        raise ValueError("No layer relevance data.")

    details = result.get("token_details", [])
    prompt_tokens = clean_token_list(
        result.get("prompt_tokens", result.get("input_tokens", []))
    )

    if step < len(details) and "context_tokens" in details[step]:
        ctx_tokens = clean_token_list(details[step]["context_tokens"])
        prompt_len = details[step].get(
            "prompt_len", result.get("prompt_len", len(ctx_tokens))
        )
    else:
        all_tok = clean_token_list(
            result.get("all_token_strings", result.get("input_tokens", []))
        )
        prompt_len = result.get(
            "prompt_len", len(result.get("input_tokens", []))
        )
        ctx_len = len(to_numpy(layer_rel[0]))
        ctx_tokens = list(all_tok[:ctx_len])

    matrix = np.array([to_numpy(lr) for lr in layer_rel])
    n_layers, n_tokens = matrix.shape

    col_labels = ctx_tokens[:n_tokens]
    while len(col_labels) < n_tokens:
        col_labels.append("")

    if figsize is None:
        figsize = (
            max(9, n_tokens * 0.8 + 4),
            max(5, n_layers * 0.38 + 2.5),
        )

    fig, ax = plt.subplots(figsize=figsize)
    annot = n_layers * n_tokens <= 200

    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=col_labels,
        yticklabels=[f"Layer {i}" for i in range(n_layers)],
        cmap=cmap,
        annot=annot,
        fmt=".2f" if annot else "",
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Relevance", "shrink": 0.75},
        vmin=0.0,
    )

    for idx, label in enumerate(ax.get_xticklabels()):
        if idx < prompt_len:
            label.set_color("#c0392b")
            label.set_fontweight("bold")
        else:
            label.set_color("#2471a3")
            label.set_fontweight("bold")
            label.set_fontstyle("italic")

    if prompt_len < n_tokens:
        ax.axvline(
            x=prompt_len, color="#fff", linewidth=2.5, linestyle="--", alpha=0.8
        )

    gen_tok = (
        clean_token(details[step]["generated_token"])
        if step < len(details)
        else "?"
    )
    n_gen = max(0, n_tokens - prompt_len)

    ax.set_xlabel(
        f"Context Tokens  ({prompt_len} prompt + {n_gen} generated)",
        fontsize=11, labelpad=10,
    )
    ax.set_ylabel("Transformer Layer", fontsize=11, labelpad=10)
    ax.set_title(
        f"Per-Layer Attention Relevance  |  Step {step}: \u2192 '{gen_tok}'",
        fontsize=13, fontweight="bold", pad=12,
    )

    mid_layer = n_layers // 2
    ax.annotate(
        "\u2190 early layers\n(attention sink)",
        xy=(n_tokens + 0.2, mid_layer / 2),
        fontsize=7.5, color="#888", style="italic",
        xycoords="data", ha="left", va="center",
    )
    ax.annotate(
        "\u2190 later layers\n(semantic)",
        xy=(n_tokens + 0.2, mid_layer + mid_layer / 2),
        fontsize=7.5, color="#888", style="italic",
        xycoords="data", ha="left", va="center",
    )

    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    legend_handles = [
        mpatches.Patch(facecolor="#c0392b", label="Prompt token", alpha=0.7),
        mpatches.Patch(facecolor="#2471a3", label="Generated token", alpha=0.7),
    ]
    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=8,
        framealpha=0.9, edgecolor="#ccc",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Layer heatmap saved -> {save_path}")
    return save_path
