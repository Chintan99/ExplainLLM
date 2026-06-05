"""Network graph visualizations for token relevance (arc, radial, force-directed)."""

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
from typing import Dict, List, Optional, Tuple

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

from explainllm.utils import to_numpy, clean_token


def _extract_steps(result: Dict) -> List[Dict]:
    """Pull out per-step info from result dict."""
    details = result.get("token_details", [])
    prompt_len = result.get("prompt_len", len(result.get("input_tokens", [])))
    prompt_tokens = result.get("prompt_tokens", result.get("input_tokens", []))

    steps = []
    for d in details:
        ctx_tokens = [clean_token(t) for t in d.get("context_tokens", prompt_tokens)]
        rel = to_numpy(d.get("full_relevance", [0.0] * len(ctx_tokens)))
        gen_tok = clean_token(d["generated_token"])
        pl = d.get("prompt_len", prompt_len)
        steps.append({
            "step": d["step"],
            "generated_token": gen_tok,
            "context_tokens": ctx_tokens,
            "prompt_len": pl,
            "relevance": rel,
        })
    return steps


# ---------------------------------------------------------------------------
# 1. Arc Diagram
# ---------------------------------------------------------------------------

def plot_relevance_graph(
    result: Dict,
    steps: Optional[List[int]] = None,
    top_k: int = 4,
    min_relevance: float = 0.02,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: str = "relevance_graph.png",
    dpi: int = 180,
    show_values: bool = True,
    title: Optional[str] = None,
) -> str:
    """
    Arc-diagram network graph showing token relevance flow.

    Tokens are placed sequentially on a horizontal axis. For each
    generated token, curved arcs connect back to the context tokens
    that contributed most, with arc thickness proportional to relevance.
    """
    all_steps = _extract_steps(result)
    if not all_steps:
        raise ValueError("No generation steps found in result.")

    if steps is not None:
        all_steps = [s for s in all_steps if s["step"] in steps]

    prompt_len = result.get("prompt_len", len(result.get("input_tokens", [])))
    all_tokens = [
        clean_token(t)
        for t in result.get("all_token_strings", result.get("input_tokens", []))
    ]

    n_gen = len(result.get("token_details", []))
    total_tokens = prompt_len + n_gen
    all_tokens = all_tokens[:total_tokens]
    while len(all_tokens) < total_tokens:
        all_tokens.append("?")

    cumulative_relevance = np.zeros(total_tokens)
    for s in all_steps:
        rel = s["relevance"]
        for i, r in enumerate(rel):
            if i < total_tokens:
                cumulative_relevance[i] += r

    max_cum = cumulative_relevance.max() + 1e-10
    node_sizes = 200 + 1800 * (cumulative_relevance / max_cum)

    edges: List[Dict] = []
    for s in all_steps:
        dst_pos = prompt_len + s["step"]
        rel = s["relevance"]
        k = min(top_k, len(rel))
        top_idx = np.argsort(rel)[::-1][:k]
        for src_pos in top_idx:
            w = float(rel[src_pos])
            if w >= min_relevance and int(src_pos) != dst_pos:
                edges.append({
                    "src": int(src_pos),
                    "dst": dst_pos,
                    "weight": w,
                    "step": s["step"],
                })

    if figsize is None:
        w = max(14, total_tokens * 1.3 + 2)
        h = max(7, total_tokens * 0.45 + 2)
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)

    x_positions = np.linspace(0, 1, total_tokens)
    y_base = 0.0

    prompt_color = "#d63031"
    gen_color = "#0984e3"
    prompt_color_light = "#fab1a0"
    gen_color_light = "#74b9ff"

    max_weight = max((e["weight"] for e in edges), default=1.0)

    for e in edges:
        x_src = x_positions[e["src"]]
        x_dst = x_positions[e["dst"]]
        w = e["weight"]

        x_mid = (x_src + x_dst) / 2
        span = abs(x_dst - x_src)
        arc_height = 0.08 + 0.42 * (
            span / (x_positions[-1] - x_positions[0] + 1e-10)
        )

        verts = [
            (x_src, y_base),
            (x_mid, y_base + arc_height),
            (x_dst, y_base),
        ]
        codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
        path = MplPath(verts, codes)

        lw = 1.0 + 6.0 * (w / (max_weight + 1e-10))
        alpha = 0.25 + 0.65 * (w / (max_weight + 1e-10))

        edge_color = prompt_color if e["src"] < prompt_len else gen_color

        patch = FancyArrowPatch(
            path=path,
            arrowstyle="->,head_width=5,head_length=4",
            lw=lw, color=edge_color, alpha=alpha,
            connectionstyle="arc3",
            zorder=1,
        )
        ax.add_patch(patch)

        if show_values and w >= 0.05:
            txt_x = x_mid
            txt_y = y_base + arc_height * 0.6
            ax.text(
                txt_x, txt_y, f"{w:.2f}",
                fontsize=7, ha="center", va="bottom",
                color=edge_color, alpha=min(1.0, alpha + 0.2),
                fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    for i in range(total_tokens):
        is_prompt = i < prompt_len
        color = prompt_color if is_prompt else gen_color
        light = prompt_color_light if is_prompt else gen_color_light
        size = node_sizes[i]

        circle = plt.Circle(
            (x_positions[i], y_base),
            radius=0.004 + 0.016 * (size / 2000),
            facecolor=light, edgecolor=color, linewidth=2.0,
            zorder=3,
        )
        ax.add_patch(circle)

        tok_text = all_tokens[i].strip() if all_tokens[i].strip() else all_tokens[i]
        if len(tok_text) > 15:
            tok_text = tok_text[:12] + "..."

        tag = "P" if is_prompt else "G"
        ax.text(
            x_positions[i], y_base - 0.045, tok_text,
            fontsize=9, ha="center", va="top",
            fontweight="bold", color=color,
            fontfamily="monospace",
            rotation=35, rotation_mode="anchor",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

        ax.text(
            x_positions[i], y_base - 0.09, f"[{tag}:{i}]",
            fontsize=6.5, ha="center", va="top",
            color="#888", fontfamily="monospace",
        )

        if cumulative_relevance[i] > 0.05:
            ax.text(
                x_positions[i], y_base + 0.025,
                f"{cumulative_relevance[i]:.2f}",
                fontsize=6.5, ha="center", va="bottom",
                color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    if prompt_len < total_tokens:
        sep_x = (x_positions[prompt_len - 1] + x_positions[prompt_len]) / 2
        ax.axvline(
            x=sep_x, color="#bbb", linewidth=1.5, linestyle="--",
            ymin=0.0, ymax=0.15, alpha=0.6,
        )
        ax.text(
            sep_x, y_base - 0.125, "prompt | generated",
            fontsize=7, ha="center", color="#999", style="italic",
        )

    ax.set_xlim(-0.05, 1.05)
    y_top = max(
        0.5,
        max(
            (
                0.08 + 0.42 * abs(x_positions[e["dst"]] - x_positions[e["src"]])
                for e in edges
            ),
            default=0.3,
        )
        + 0.1,
    )
    ax.set_ylim(-0.17, y_top + 0.05)
    ax.set_aspect("auto")
    ax.axis("off")

    if title is None:
        gen_text = clean_token(result.get("generated_text", ""))
        if len(gen_text) > 50:
            gen_text = gen_text[:47] + "..."
        title = f'Token Relevance Graph - "{gen_text}"'
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    legend_handles = [
        mpatches.Patch(
            facecolor=prompt_color_light, edgecolor=prompt_color,
            linewidth=1.5, label="Prompt token",
        ),
        mpatches.Patch(
            facecolor=gen_color_light, edgecolor=gen_color,
            linewidth=1.5, label="Generated token",
        ),
        plt.Line2D(
            [0], [0], color=prompt_color, lw=3, alpha=0.6,
            label="attends to prompt",
        ),
        plt.Line2D(
            [0], [0], color=gen_color, lw=3, alpha=0.6,
            label="attends to generated",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper left", fontsize=8,
        framealpha=0.9, edgecolor="#ccc", ncol=2,
    )

    info = (
        f"Prompt: {prompt_len} tokens  |  Generated: {n_gen} tokens  |  "
        f"top-{top_k} edges/step  |  min relevance: {min_relevance}"
    )
    ax.text(
        0.5, -0.14, info, transform=ax.transAxes,
        fontsize=7, ha="center", color="#999",
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Arc graph saved -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 2. Single-step radial graph
# ---------------------------------------------------------------------------

def plot_step_graph(
    result: Dict,
    step: int = 0,
    top_k: int = 6,
    min_relevance: float = 0.01,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: str = "step_graph.png",
    dpi: int = 180,
) -> str:
    """
    Radial network graph for a single generation step.

    The generated token is at the centre, with context tokens arranged
    in a circle around it. Edge thickness and node size proportional to relevance.
    """
    all_steps = _extract_steps(result)
    if step >= len(all_steps):
        raise IndexError(f"Step {step} not available (max {len(all_steps)-1}).")

    s = all_steps[step]
    ctx_tokens = s["context_tokens"]
    rel = s["relevance"]
    gen_tok = s["generated_token"]
    pl = s["prompt_len"]

    n = len(rel)
    order = np.argsort(rel)[::-1]
    shown_idx = [int(i) for i in order if rel[i] >= min_relevance][:top_k]

    if not shown_idx:
        shown_idx = [int(order[0])]

    if figsize is None:
        k = len(shown_idx)
        sz = max(8, k * 0.9 + 4)
        figsize = (sz, sz)

    fig, ax = plt.subplots(figsize=figsize)

    cx, cy = 0.5, 0.5
    radius = 0.35

    angles = np.linspace(
        np.pi / 2, np.pi / 2 + 2 * np.pi, len(shown_idx), endpoint=False
    )
    positions = {}
    for k_idx, (idx, angle) in enumerate(zip(shown_idx, angles)):
        px = cx + radius * np.cos(angle)
        py = cy + radius * np.sin(angle)
        positions[idx] = (px, py)

    max_rel = max(rel[i] for i in shown_idx)

    for idx in shown_idx:
        w = rel[idx]
        px, py = positions[idx]
        lw = 1.5 + 8.0 * (w / (max_rel + 1e-10))
        alpha = 0.3 + 0.6 * (w / (max_rel + 1e-10))

        is_prompt = idx < pl
        color = "#d63031" if is_prompt else "#0984e3"

        ax.annotate(
            "", xy=(cx, cy), xytext=(px, py),
            arrowprops=dict(
                arrowstyle="->,head_width=8,head_length=6",
                color=color, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.1",
            ),
            zorder=1,
        )

        mx = 0.55 * px + 0.45 * cx
        my = 0.55 * py + 0.45 * cy
        ax.text(
            mx, my, f"{w:.3f}",
            fontsize=8, ha="center", va="center",
            fontweight="bold", color=color, alpha=min(1, alpha + 0.2),
            bbox=dict(
                boxstyle="round,pad=0.15", facecolor="white",
                edgecolor="none", alpha=0.85,
            ),
            zorder=4,
        )

    for idx in shown_idx:
        px, py = positions[idx]
        w = rel[idx]
        is_prompt = idx < pl

        node_r = 0.025 + 0.04 * (w / (max_rel + 1e-10))
        face = "#fab1a0" if is_prompt else "#74b9ff"
        edge = "#d63031" if is_prompt else "#0984e3"
        tag = "P" if is_prompt else "G"

        circle = plt.Circle(
            (px, py), node_r, facecolor=face,
            edgecolor=edge, linewidth=2.5, zorder=5,
        )
        ax.add_patch(circle)

        tok = ctx_tokens[idx].strip() if idx < len(ctx_tokens) else "?"
        if len(tok) > 12:
            tok = tok[:10] + ".."

        label_r = node_r + 0.05
        lx = cx + (radius + label_r) * np.cos(angles[shown_idx.index(idx)])
        ly = cy + (radius + label_r) * np.sin(angles[shown_idx.index(idx)])

        ax.text(
            lx, ly, f"[{tag}:{idx}] {tok}",
            fontsize=10, ha="center", va="center",
            fontweight="bold", color=edge,
            fontfamily="monospace",
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            zorder=6,
        )

    centre_r = 0.05
    circle = plt.Circle(
        (cx, cy), centre_r, facecolor="#dfe6e9",
        edgecolor="#2d3436", linewidth=3, zorder=5,
    )
    ax.add_patch(circle)
    ax.text(
        cx, cy, gen_tok.strip(), fontsize=13, ha="center", va="center",
        fontweight="bold", color="#2d3436", zorder=6,
    )
    ax.text(
        cx, cy - centre_r - 0.025, "generated", fontsize=7,
        ha="center", color="#636e72", style="italic",
    )

    n_ctx = len(ctx_tokens)
    n_gen_ctx = n_ctx - pl
    ax.set_title(
        f"Step {step}: Generating '{gen_tok.strip()}'  "
        f"({pl}P + {n_gen_ctx}G = {n_ctx} context tokens, showing top {len(shown_idx)})",
        fontsize=13, fontweight="bold", pad=12,
    )

    legend_handles = [
        mpatches.Patch(
            facecolor="#fab1a0", edgecolor="#d63031",
            linewidth=1.5, label="Prompt token",
        ),
        mpatches.Patch(
            facecolor="#74b9ff", edgecolor="#0984e3",
            linewidth=1.5, label="Generated token",
        ),
        mpatches.Patch(
            facecolor="#dfe6e9", edgecolor="#2d3436",
            linewidth=1.5, label="Target (generated)",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="lower left", fontsize=8,
        framealpha=0.9, edgecolor="#ccc",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Step graph saved -> {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 3. Force-directed graph (requires networkx)
# ---------------------------------------------------------------------------

def plot_force_graph(
    result: Dict,
    steps: Optional[List[int]] = None,
    top_k: int = 3,
    min_relevance: float = 0.03,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: str = "force_graph.png",
    dpi: int = 180,
) -> str:
    """
    Force-directed network graph using networkx spring layout.

    Nodes with strong relevance connections are pulled closer together,
    revealing natural clusters of related tokens.
    """
    if not HAS_NX:
        raise ImportError("networkx is required: pip install networkx")

    all_steps = _extract_steps(result)
    if steps is not None:
        all_steps = [s for s in all_steps if s["step"] in steps]

    prompt_len = result.get("prompt_len", len(result.get("input_tokens", [])))
    all_tokens = [
        clean_token(t)
        for t in result.get("all_token_strings", result.get("input_tokens", []))
    ]
    n_gen = len(result.get("token_details", []))
    total = prompt_len + n_gen
    all_tokens = all_tokens[:total]
    while len(all_tokens) < total:
        all_tokens.append("?")

    G = nx.DiGraph()

    for i in range(total):
        is_p = i < prompt_len
        tag = "P" if is_p else "G"
        tok = all_tokens[i].strip() or all_tokens[i]
        G.add_node(i, label=f"[{tag}:{i}] {tok}", is_prompt=is_p, token=tok, tag=tag)

    cum_rel = np.zeros(total)

    for s in all_steps:
        dst = prompt_len + s["step"]
        rel = s["relevance"]
        order = np.argsort(rel)[::-1][:top_k]
        for src in order:
            src = int(src)
            w = float(rel[src])
            if w >= min_relevance and src != dst and src < total and dst < total:
                if G.has_edge(src, dst):
                    G[src][dst]["weight"] += w
                else:
                    G.add_edge(src, dst, weight=w)
                cum_rel[src] += w
                cum_rel[dst] += w

    if figsize is None:
        sz = max(10, total * 0.6 + 3)
        figsize = (sz, sz)

    fig, ax = plt.subplots(figsize=figsize)

    pos = nx.spring_layout(
        G, k=2.5 / math.sqrt(total + 1), iterations=80, weight="weight", seed=42
    )

    node_colors = []
    node_edge_colors = []
    node_sizes = []
    for n in G.nodes():
        is_p = G.nodes[n]["is_prompt"]
        node_colors.append("#fab1a0" if is_p else "#74b9ff")
        node_edge_colors.append("#d63031" if is_p else "#0984e3")
        node_sizes.append(300 + 2000 * (cum_rel[n] / (cum_rel.max() + 1e-10)))

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_ew = max(edge_weights) if edge_weights else 1.0
    edge_widths = [1.0 + 5.0 * (w / max_ew) for w in edge_weights]
    edge_alphas = [0.2 + 0.7 * (w / max_ew) for w in edge_weights]
    edge_colors_list = []
    for u, v in G.edges():
        edge_colors_list.append("#d63031" if G.nodes[u]["is_prompt"] else "#0984e3")

    for (u, v), lw, alpha, ec in zip(
        G.edges(), edge_widths, edge_alphas, edge_colors_list
    ):
        ax.annotate(
            "", xy=pos[v], xytext=pos[u],
            arrowprops=dict(
                arrowstyle="->,head_width=5,head_length=4",
                color=ec, lw=lw, alpha=alpha,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=1,
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        node_size=node_sizes,
        linewidths=2.0, alpha=0.9,
    )

    labels = {}
    for n in G.nodes():
        tok = G.nodes[n]["token"]
        tag = G.nodes[n]["tag"]
        if len(tok) > 10:
            tok = tok[:8] + ".."
        labels[n] = f"{tok}\n[{tag}:{n}]"

    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_size=8, font_weight="bold", font_family="monospace",
    )

    for (u, v), w in zip(G.edges(), edge_weights):
        if w / max_ew > 0.3:
            mx = 0.6 * pos[u][0] + 0.4 * pos[v][0]
            my = 0.6 * pos[u][1] + 0.4 * pos[v][1]
            ax.text(
                mx, my, f"{w:.2f}", fontsize=6.5, ha="center",
                va="center", fontweight="bold", color="#555",
                bbox=dict(
                    boxstyle="round,pad=0.1", facecolor="white",
                    edgecolor="none", alpha=0.8,
                ),
                zorder=4,
            )

    ax.set_title(
        "Force-Directed Relevance Graph", fontsize=14,
        fontweight="bold", pad=15,
    )

    legend_handles = [
        mpatches.Patch(
            facecolor="#fab1a0", edgecolor="#d63031",
            linewidth=1.5, label="Prompt token",
        ),
        mpatches.Patch(
            facecolor="#74b9ff", edgecolor="#0984e3",
            linewidth=1.5, label="Generated token",
        ),
        plt.Line2D(
            [0], [0], color="#d63031", lw=3, alpha=0.6, label="prompt -> gen",
        ),
        plt.Line2D(
            [0], [0], color="#0984e3", lw=3, alpha=0.6, label="gen -> gen",
        ),
    ]
    ax.legend(
        handles=legend_handles, loc="upper left", fontsize=8,
        framealpha=0.9, edgecolor="#ccc",
    )

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Force graph saved -> {save_path}")
    return save_path
