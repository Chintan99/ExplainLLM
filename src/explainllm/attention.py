import torch
from typing import List, Optional, Tuple

from explainllm.utils import entropy


def attention_rollout(
    attentions: Tuple[torch.Tensor, ...],
    sink_positions: Optional[List[int]] = None,
    sink_dampen: float = 0.1,
) -> torch.Tensor:
    """
    Compute attention rollout across all layers with optional sink dampening.

    Args:
        attentions:     Tuple of [batch, heads, seq, seq] tensors.
        sink_positions: Token positions to dampen (default: [0]).
        sink_dampen:    Factor for sink columns (0=remove, 1=keep).

    Returns:
        [batch, seq, seq] rolled-out attention matrix.
    """
    if sink_positions is None:
        sink_positions = [0]

    avg_attentions = [att.mean(dim=1) for att in attentions]

    rollout = None
    for att in avg_attentions:
        att_d = att.clone()
        for pos in sink_positions:
            if pos < att_d.size(-1):
                att_d[:, :, pos] *= sink_dampen
        att_d = att_d / (att_d.sum(dim=-1, keepdim=True) + 1e-10)

        identity = torch.eye(
            att.size(-1), device=att.device, dtype=att.dtype
        ).unsqueeze(0)
        att_r = 0.5 * att_d + 0.5 * identity
        att_r = att_r / (att_r.sum(dim=-1, keepdim=True) + 1e-10)

        rollout = att_r if rollout is None else torch.bmm(att_r, rollout)

    return rollout
