"""Shared fixtures for explainllm tests."""

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_attention_matrices():
    """Fake attention matrices: 4 layers, 1 batch, 8 heads, 10 seq len."""
    return tuple(torch.rand(1, 8, 10, 10) for _ in range(4))


@pytest.fixture
def sample_result():
    """
    A realistic result dict matching the generate_with_relevance output.
    Simulates 3 generation steps over a 5-token prompt.
    """
    prompt_tokens = ["<s>", "What", " is", " AI", "?"]
    generated = ["AI", " is", " a"]
    prompt_len = len(prompt_tokens)

    all_token_strings = list(prompt_tokens) + list(generated)

    token_details = []
    for step_i, gen_tok in enumerate(generated):
        ctx_len = prompt_len + step_i
        rel = np.random.dirichlet(np.ones(ctx_len)).tolist()
        ctx_tokens = all_token_strings[:ctx_len]

        sorted_idx = sorted(range(ctx_len), key=lambda i: rel[i], reverse=True)
        top_k = min(5, ctx_len)
        top_contributors = [
            {
                "token": ctx_tokens[idx],
                "position": idx,
                "is_prompt": idx < prompt_len,
                "relevance": rel[idx],
            }
            for idx in sorted_idx[:top_k]
        ]

        token_details.append({
            "step": step_i,
            "generated_token": gen_tok,
            "token_id": 100 + step_i,
            "full_relevance": rel,
            "context_tokens": ctx_tokens,
            "prompt_len": prompt_len,
            "top_contributing_tokens": top_contributors,
            "top_contributing_input_tokens": top_contributors,
        })

    return {
        "prompt": "What is AI?",
        "generated_text": "AI is a",
        "full_text": "What is AI? AI is a",
        "prompt_tokens": prompt_tokens,
        "prompt_len": prompt_len,
        "input_tokens": prompt_tokens,
        "generated_tokens": [100, 101, 102],
        "all_token_strings": all_token_strings,
        "relevance_per_token": [],
        "token_details": token_details,
    }
