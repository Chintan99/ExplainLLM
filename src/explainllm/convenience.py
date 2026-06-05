from typing import Dict, Optional

from explainllm.config import RelevanceConfig
from explainllm.engine import LLMRelevanceWrapper


def calculate_relevance(
    prompt: str,
    config: Optional[RelevanceConfig] = None,
    model_id: Optional[str] = None,
    max_new_tokens: int = 20,
    token: Optional[str] = None,
    device: str = "cpu",
    torch_dtype: str = "float32",
    method: str = "combined",
    alpha: float = 0.5,
    verbose: bool = False,
    suppress_bos: bool = True,
    bos_dampen: float = 0.01,
    use_top_layers_fraction: float = 0.5,
) -> Dict:
    """
    One-call convenience function: loads model, runs relevance, returns results.

    Either pass a ``RelevanceConfig`` object via ``config``, or pass individual
    keyword arguments which will be used to build one.
    """
    if config is None:
        config = RelevanceConfig(
            model_id=model_id or "meta-llama/Llama-3.2-1B-Instruct",
            auth_token=token or "",
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            relevance_method=method,
            alpha=alpha,
            suppress_bos=suppress_bos,
            bos_dampen=bos_dampen,
            use_top_layers_fraction=use_top_layers_fraction,
        )

    wrapper = LLMRelevanceWrapper(config=config)
    return wrapper.generate_with_relevance(
        prompt=prompt,
        calculate_relevance=True,
        verbose=verbose,
    )
