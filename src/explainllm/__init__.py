"""LLM Token Relevance Calculator using DL-Backtrace patterns."""

from explainllm._version import __version__
from explainllm.config import RelevanceConfig
from explainllm.engine import LLMRelevanceWrapper
from explainllm.convenience import calculate_relevance
from explainllm.export import export_relevance_json
from explainllm.attention import attention_rollout
from explainllm.utils import clean_token, clean_token_list

try:
    from explainllm import viz
except Exception:  # pragma: no cover - optional viz deps may be absent
    viz = None

__all__ = [
    "__version__",
    "RelevanceConfig",
    "LLMRelevanceWrapper",
    "calculate_relevance",
    "export_relevance_json",
    "attention_rollout",
    "clean_token",
    "clean_token_list",
    "viz",
]
