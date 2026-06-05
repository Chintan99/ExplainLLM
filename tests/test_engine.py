"""Tests for explainllm.engine (mocked, no GPU required)."""

import pytest
from unittest.mock import MagicMock, patch
import torch

from explainllm.config import RelevanceConfig
from explainllm.engine import LLMRelevanceWrapper


class TestLLMRelevanceWrapperInit:
    def test_with_provided_model_and_tokenizer(self):
        """Test that a pre-loaded model and tokenizer can be injected."""
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([torch.tensor([1.0])])
        mock_model.eval.return_value = mock_model

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        wrapper = LLMRelevanceWrapper(
            model=mock_model,
            tokenizer=mock_tokenizer,
        )

        assert wrapper.tokenizer.pad_token == "<eos>"

    def test_config_defaults(self):
        config = RelevanceConfig()
        assert config.model_id == "meta-llama/Llama-3.2-1B-Instruct"
        assert config.device == "cpu"
        assert config.relevance_method == "combined"
        assert config.alpha == 0.5


class TestBuildTokenDetails:
    def test_basic(self):
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode = lambda ids: f"tok_{ids[0]}"

        predicted_ids = [10, 20]
        all_token_strings = ["a", "b", "c", "d"]
        prompt_len = 2

        rel_scores = [
            {"input_explainllm": torch.tensor([0.3, 0.7])},
            {"input_explainllm": torch.tensor([0.1, 0.5, 0.4])},
        ]

        details = LLMRelevanceWrapper._build_token_details(
            predicted_ids=predicted_ids,
            relevance_scores=rel_scores,
            all_token_strings=all_token_strings,
            prompt_len=prompt_len,
            tokenizer=mock_tokenizer,
        )

        assert len(details) == 2
        assert details[0]["step"] == 0
        assert details[0]["token_id"] == 10
        assert "top_contributing_tokens" in details[0]
        assert "full_relevance" in details[0]


class TestAttentionRelevanceEdgeCases:
    def test_uniform_attention_still_normalizes(self):
        """Uniform attention can make entropy-based weights all zero."""
        seq_len = 7
        uniform = torch.full((1, 2, seq_len, seq_len), 1.0 / seq_len)
        attentions = (uniform, uniform.clone())

        rel = LLMRelevanceWrapper._attention_relevance(
            None,
            attentions=attentions,
            context_len=seq_len,
            sink_positions=[],
            bos_dampen=0.01,
            use_top_layers_fraction=1.0,
        )

        assert rel.shape[0] == seq_len
        assert torch.isfinite(rel).all()
        assert rel.sum().item() == pytest.approx(1.0, abs=1e-6)
        assert torch.all(rel > 0)

    def test_zero_top_layers_fraction_keeps_last_layer(self):
        seq_len = 6
        attentions = (
            torch.rand(1, 2, seq_len, seq_len),
            torch.rand(1, 2, seq_len, seq_len),
        )

        rel = LLMRelevanceWrapper._attention_relevance(
            None,
            attentions=attentions,
            context_len=seq_len,
            sink_positions=[],
            bos_dampen=0.01,
            use_top_layers_fraction=0.0,
        )

        assert rel.shape[0] == seq_len
        assert torch.isfinite(rel).all()
        assert rel.sum().item() == pytest.approx(1.0, abs=1e-6)
