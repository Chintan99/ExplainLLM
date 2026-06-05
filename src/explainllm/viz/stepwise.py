"""Step-wise relevance extraction and text display (v3: variable-length rows)."""

import numpy as np
from typing import Dict, List

from explainllm.utils import to_numpy, clean_token, clean_token_list


def get_stepwise_relevance(result: Dict) -> Dict:
    """
    Extract step-wise relevance from a v3 result dict.

    Returns a dict with:
        prompt_tokens, prompt_len, output_tokens, all_tokens,
        matrix (padded np.ndarray), mask (bool np.ndarray), steps (list of dicts).
    """
    details = result.get("token_details", [])
    prompt_len = result.get("prompt_len", len(result.get("input_tokens", [])))
    prompt_tokens = clean_token_list(
        result.get("prompt_tokens", result.get("input_tokens", []))
    )
    all_tokens = clean_token_list(
        result.get("all_token_strings", list(prompt_tokens))
    )

    steps_out: List[Dict] = []
    output_tokens: List[str] = []
    raw_rows: List[List[float]] = []

    for d in details:
        ctx_tokens = clean_token_list(d.get("context_tokens", prompt_tokens))
        rel_vec = d.get("full_relevance", [0.0] * len(ctx_tokens))
        gen_tok = clean_token(d["generated_token"])
        output_tokens.append(gen_tok)
        raw_rows.append(rel_vec)

        contributors = sorted(
            d.get(
                "top_contributing_tokens",
                d.get("top_contributing_input_tokens", []),
            ),
            key=lambda x: x["relevance"],
            reverse=True,
        )
        for c in contributors:
            c["token"] = clean_token(c["token"])

        steps_out.append({
            "step": d["step"],
            "generated_token": gen_tok,
            "token_id": d["token_id"],
            "context_tokens": ctx_tokens,
            "prompt_len": d.get("prompt_len", prompt_len),
            "relevance_vector": rel_vec,
            "top_contributors": contributors,
        })

    max_ctx = max((len(r) for r in raw_rows), default=0)
    n_out = len(raw_rows)
    matrix = np.zeros((n_out, max_ctx), dtype=np.float64)
    mask = np.zeros((n_out, max_ctx), dtype=bool)
    for i, row in enumerate(raw_rows):
        matrix[i, : len(row)] = row
        mask[i, : len(row)] = True

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len,
        "output_tokens": output_tokens,
        "all_tokens": all_tokens,
        "matrix": matrix,
        "mask": mask,
        "steps": steps_out,
    }


def print_stepwise_relevance(stepwise: Dict, top_k: int = 5) -> str:
    """Pretty-print step-wise relevance with P/G labels."""
    lines: List[str] = []

    for s in stepwise["steps"]:
        ctx = s["context_tokens"]
        pl = s["prompt_len"]
        lines.append("=" * 72)
        lines.append(
            f"  Step {s['step']}  |  Generated: '{s['generated_token']}'  "
            f"| Context: {len(ctx)} tokens ({pl}P + {len(ctx)-pl}G)"
        )
        lines.append("-" * 72)

        rel = s["relevance_vector"]
        sorted_idx = sorted(
            range(len(rel)), key=lambda i: rel[i], reverse=True
        )

        for rank, idx in enumerate(sorted_idx[:top_k]):
            bar_len = int(rel[idx] * 50)
            bar = "\u2588" * bar_len + "\u2591" * (50 - bar_len)
            tok = ctx[idx] if idx < len(ctx) else "???"
            tag = "P" if idx < pl else "G"
            lines.append(
                f"  {rank+1:2d}. [{tag}:{idx:3d}] {tok:20s} {bar} {rel[idx]:.4f}"
            )
        lines.append("")

    return "\n".join(lines)
