import os
from dataclasses import dataclass, field


@dataclass
class RelevanceConfig:
    """Configuration for the LLM Token Relevance Calculator."""

    # Model
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    auth_token: str = field(
        default_factory=lambda: os.environ.get("HF_TOKEN", "")
    )
    device: str = "cpu"
    torch_dtype: str = "float32"

    # Generation
    max_new_tokens: int = 20

    # Relevance algorithm
    relevance_method: str = "combined"  # "attention" | "gradient" | "rollout" | "combined"
    alpha: float = 0.5
    suppress_bos: bool = True
    bos_dampen: float = 0.01
    use_top_layers_fraction: float = 0.5

    # Visualization defaults
    colormap: str = "YlOrRd"
    dpi: int = 180
    top_k: int = 5

    def get_torch_dtype(self):
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.float32)
