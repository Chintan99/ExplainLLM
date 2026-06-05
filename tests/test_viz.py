"""Tests for explainllm.viz subpackage."""

import os
import tempfile
import pytest

from explainllm.viz.text import visualize_relevance, visualize_all_steps
from explainllm.viz.html import relevance_to_html
from explainllm.viz.stepwise import get_stepwise_relevance, print_stepwise_relevance


class TestTextVisualization:
    def test_visualize_relevance_basic(self, sample_result):
        output = visualize_relevance(sample_result, step=0, top_k=3)
        assert "Generated:" in output
        assert "Context:" in output

    def test_visualize_relevance_invalid_step(self, sample_result):
        output = visualize_relevance(sample_result, step=999)
        assert "not available" in output

    def test_visualize_all_steps(self, sample_result):
        output = visualize_all_steps(sample_result, top_k=3)
        assert output.count("Generated:") == 3


class TestHtmlVisualization:
    def test_relevance_to_html(self, sample_result):
        html = relevance_to_html(sample_result, step=0)
        assert "<span" in html
        assert "Generated token:" in html

    def test_invalid_step(self, sample_result):
        html = relevance_to_html(sample_result, step=999)
        assert "not available" in html

    def test_handles_short_relevance_and_escapes_tokens(self):
        result = {
            "input_tokens": ["a", "b"],
            "token_details": [
                {
                    "generated_token": "<script>alert(1)</script>",
                    "token_id": 1,
                    "context_tokens": ["<b>x</b>", "safe"],
                    "prompt_len": 1,
                    "full_relevance": [1.0],
                }
            ],
        }
        html = relevance_to_html(result, step=0)
        assert "&lt;script&gt;" in html
        assert "&lt;b&gt;x&lt;/b&gt;" in html


class TestStepwise:
    def test_get_stepwise_relevance(self, sample_result):
        sw = get_stepwise_relevance(sample_result)
        assert sw["prompt_len"] == 5
        assert len(sw["output_tokens"]) == 3
        assert sw["matrix"].shape[0] == 3
        assert sw["mask"].shape == sw["matrix"].shape

    def test_print_stepwise_relevance(self, sample_result):
        sw = get_stepwise_relevance(sample_result)
        output = print_stepwise_relevance(sw, top_k=3)
        assert "Step 0" in output
        assert "Step 1" in output
        assert "Step 2" in output


class TestHeatmapPlots:
    def test_plot_relevance_heatmap(self, sample_result):
        from explainllm.viz.heatmap import plot_relevance_heatmap

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result_path = plot_relevance_heatmap(sample_result, save_path=path)
            assert os.path.exists(result_path)
            assert os.path.getsize(result_path) > 0
        finally:
            os.unlink(path)

    def test_plot_single_step_heatmap(self, sample_result):
        from explainllm.viz.heatmap import plot_single_step_heatmap

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result_path = plot_single_step_heatmap(
                sample_result, step=0, save_path=path
            )
            assert os.path.exists(result_path)
        finally:
            os.unlink(path)

    def test_plot_relevance_heatmap_empty_steps(self):
        from explainllm.viz.heatmap import plot_relevance_heatmap

        empty_result = {
            "prompt_len": 2,
            "input_tokens": ["a", "b"],
            "token_details": [],
            "all_token_strings": ["a", "b"],
        }
        with pytest.raises(ValueError, match="No generation steps found"):
            plot_relevance_heatmap(empty_result)


class TestGraphPlots:
    def test_plot_relevance_graph(self, sample_result):
        from explainllm.viz.graph import plot_relevance_graph

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result_path = plot_relevance_graph(
                sample_result, top_k=2, save_path=path
            )
            assert os.path.exists(result_path)
        finally:
            os.unlink(path)

    def test_plot_step_graph(self, sample_result):
        from explainllm.viz.graph import plot_step_graph

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            result_path = plot_step_graph(
                sample_result, step=0, top_k=3, save_path=path
            )
            assert os.path.exists(result_path)
        finally:
            os.unlink(path)
