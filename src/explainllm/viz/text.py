"""Plain-text relevance visualizations."""

from typing import Dict


def visualize_relevance(result: Dict, step: int = 0, top_k: int = 10) -> str:
    """Render a single generation step as a text bar chart."""
    details = result.get("token_details", [])
    if step >= len(details):
        return f"Step {step} not available."

    d = details[step]
    context_tokens = d.get("context_tokens", result.get("input_tokens", []))
    prompt_len = d.get("prompt_len", len(result.get("input_tokens", [])))
    relevance = d.get("full_relevance", [])

    lines = [
        "=" * 70,
        f"  Generated: '{d['generated_token']}'  (step {step}, id {d['token_id']})",
        f"  Context: {len(context_tokens)} tokens "
        f"({prompt_len} prompt + {len(context_tokens) - prompt_len} generated)",
        "=" * 70,
    ]

    sorted_idx = sorted(
        range(len(relevance)), key=lambda i: relevance[i], reverse=True
    )
    for rank, idx in enumerate(sorted_idx[:top_k]):
        bar_len = int(relevance[idx] * 50)
        bar = "\u2588" * bar_len + "\u2591" * (50 - bar_len)
        tok = context_tokens[idx] if idx < len(context_tokens) else "???"
        tag = "P" if idx < prompt_len else "G"
        lines.append(
            f"  {rank+1:2d}. [{tag}:{idx:3d}] {tok:20s} {bar} {relevance[idx]:.4f}"
        )

    return "\n".join(lines)


def visualize_all_steps(result: Dict, top_k: int = 5) -> str:
    """Text visualization for every generation step."""
    parts = []
    for step in range(len(result.get("token_details", []))):
        parts.append(visualize_relevance(result, step=step, top_k=top_k))
        parts.append("")
    return "\n".join(parts)
