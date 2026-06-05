"""Tests for explainllm.utils."""

import numpy as np
import torch
import pytest

from explainllm.utils import to_numpy, clean_token, clean_token_list, entropy


class TestToNumpy:
    def test_from_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = to_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])

    def test_from_list(self):
        result = to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_from_ndarray(self):
        arr = np.array([1, 2])
        result = to_numpy(arr)
        assert result is arr


class TestCleanToken:
    def test_bpe_space(self):
        assert clean_token("\u0120hello") == " hello"

    def test_sentencepiece_space(self):
        assert clean_token("\u2581hello") == " hello"

    def test_newline(self):
        assert clean_token("\u010a") == "\\n"

    def test_hex_escape(self):
        assert clean_token("<0x41>") == "A"

    def test_plain(self):
        assert clean_token("hello") == "hello"


class TestCleanTokenList:
    def test_basic(self):
        result = clean_token_list(["\u0120the", "cat"])
        assert result == [" the", "cat"]


class TestEntropy:
    def test_uniform(self):
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        h = entropy(probs, dim=-1)
        expected = -4 * 0.25 * np.log(0.25)
        assert abs(h.item() - expected) < 1e-5

    def test_deterministic(self):
        probs = torch.tensor([1.0, 0.0, 0.0])
        h = entropy(probs, dim=-1)
        assert h.item() < 0.01

    def test_numpy(self):
        probs = np.array([0.5, 0.5])
        h = entropy(probs, dim=-1)
        expected = -2 * 0.5 * np.log(0.5)
        assert abs(h - expected) < 1e-5
