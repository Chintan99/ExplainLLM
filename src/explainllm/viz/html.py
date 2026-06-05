"""HTML relevance visualization."""

from html import escape
from typing import Dict


def relevance_to_html(result: Dict, step: int = 0) -> str:
    """
    Render a single step as an inline-styled HTML snippet.

    Prompt tokens are highlighted in red, generated tokens in blue,
    with opacity proportional to relevance.
    """
    details = result.get("token_details", [])
    if step >= len(details):
        return "<p>Step not available.</p>"

    d = details[step]
    context_tokens = d.get("context_tokens", result.get("input_tokens", []))
    prompt_len = d.get("prompt_len", len(result.get("input_tokens", [])))
    relevance = d.get("full_relevance", [])
    max_rel = max(relevance) if relevance else 1.0

    html_tokens = []
    for idx, tok in enumerate(context_tokens):
        rel_val = relevance[idx] if idx < len(relevance) else 0.0
        r = rel_val / (max_rel + 1e-10)
        if idx < prompt_len:
            bg = f"rgba(255, 0, 0, {r:.2f})"
        else:
            bg = f"rgba(0, 100, 255, {r:.2f})"
        border = "2px solid #888" if idx >= prompt_len else "none"
        html_tokens.append(
            f'<span style="background:{bg}; padding:2px 4px; margin:1px; '
            f'border-radius:3px; border:{border}; font-family:monospace;" '
            f'title="[{idx}] relevance: {rel_val:.4f}">{escape(str(tok))}</span>'
        )

    header = (
        f"<p><b>Generated token:</b> <code>{escape(str(d['generated_token']))}</code> "
        f"(step {step}) &mdash; "
        f"<span style='color:red'>&#9632; </span> prompt  "
        f"<span style='color:blue'>&#9632; </span> generated</p>"
    )
    return header + " ".join(html_tokens)
