"""Tests for explainllm.attention."""

import torch
import pytest

from explainllm.attention import attention_rollout


class TestAttentionRollout:
    def test_output_shape(self, sample_attention_matrices):
        result = attention_rollout(sample_attention_matrices)
        assert result.shape == (1, 10, 10)

    def test_output_sums_to_one(self, sample_attention_matrices):
        result = attention_rollout(sample_attention_matrices)
        row_sums = result[0].sum(dim=-1)
        for s in row_sums:
            assert abs(s.item() - 1.0) < 0.1

    def test_sink_dampening(self, sample_attention_matrices):
        result_no_dampen = attention_rollout(
            sample_attention_matrices, sink_positions=[], sink_dampen=1.0
        )
        result_dampen = attention_rollout(
            sample_attention_matrices, sink_positions=[0], sink_dampen=0.01
        )
        # Position 0 should have lower attention with dampening
        assert result_dampen[0, -1, 0] <= result_no_dampen[0, -1, 0]

    def test_single_layer(self):
        att = (torch.eye(5).unsqueeze(0).unsqueeze(0),)  # [1, 1, 5, 5]
        result = attention_rollout(att, sink_positions=[])
        # With identity attention and no dampening, rollout should be close to identity
        assert result.shape == (1, 5, 5)
