"""Tests for explainllm.export."""

import json
import os
import tempfile
import pytest

from explainllm.export import export_relevance_json


class TestExportRelevanceJson:
    def test_returns_json_string(self, sample_result):
        json_str = export_relevance_json(sample_result)
        data = json.loads(json_str)
        assert data["prompt"] == "What is AI?"
        assert data["generated_text"] == "AI is a"
        assert data["prompt_len"] == 5
        assert len(data["token_details"]) == 3

    def test_writes_to_file(self, sample_result):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            export_relevance_json(sample_result, path)
            with open(path) as f:
                data = json.load(f)
            assert data["prompt"] == "What is AI?"
        finally:
            os.unlink(path)

    def test_roundtrip(self, sample_result):
        json_str = export_relevance_json(sample_result)
        data = json.loads(json_str)
        # Re-export from loaded data
        json_str2 = export_relevance_json(data)
        data2 = json.loads(json_str2)
        assert data["prompt"] == data2["prompt"]
        assert data["generated_text"] == data2["generated_text"]

    def test_includes_max_new_tokens(self, sample_result):
        result = dict(sample_result)
        result["max_new_tokens"] = 20
        json_str = export_relevance_json(result)
        data = json.loads(json_str)
        assert data["max_new_tokens"] == 20
