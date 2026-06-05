import re
import math
import numpy as np
from typing import List

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def to_numpy(x) -> np.ndarray:
    """Convert a torch.Tensor or array-like to a NumPy array."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def clean_token(tok: str) -> str:
    """
    Convert raw tokenizer vocabulary strings to human-readable form.

    Handles:
      - G   (U+0120) - space prefix used by BPE tokenizers (GPT-2, LLaMA, Mistral)
      - _  (U+2581) - space prefix used by SentencePiece (T5, mBART)
      - newline markers in some tokenizers
      - <0x..> hex escapes used by LLaMA tokenizers for bytes
    """
    s = tok
    s = s.replace("\u0120", " ")
    s = s.replace("\u2581", " ")
    s = s.replace("\u010a", "\\n")
    s = re.sub(r"<0x([0-9A-Fa-f]{2})>", lambda m: chr(int(m.group(1), 16)), s)
    return s


def clean_token_list(tokens: List[str]) -> List[str]:
    """Clean a list of token strings."""
    return [clean_token(t) for t in tokens]


def entropy(probs, dim: int = -1):
    """Shannon entropy along `dim`."""
    if HAS_TORCH and isinstance(probs, torch.Tensor):
        log_p = torch.log(probs + 1e-10)
        return -(probs * log_p).sum(dim=dim)
    probs = np.asarray(probs)
    log_p = np.log(probs + 1e-10)
    return -(probs * log_p).sum(axis=dim)
