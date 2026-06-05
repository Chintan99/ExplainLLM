"""Model service for ExplainLLM: keeps models warm and runs inference."""

import os
from threading import Lock
from typing import Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from explainllm.config import RelevanceConfig
from explainllm.engine import LLMRelevanceWrapper


DEFAULT_MODEL_ID = os.environ.get(
    "MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"
)

app = FastAPI(title="ExplainLLM Model Service", version="0.1.0")

_MODEL_CACHE: Dict[Tuple[str, str, str, str], LLMRelevanceWrapper] = {}
_MODEL_CACHE_LOCK = Lock()


def _to_jsonable(obj):
    """Recursively convert tensors/arrays to JSON-serializable objects."""
    try:
        import torch
    except Exception:  # pragma: no cover
        torch = None
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None

    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class WarmupRequest(BaseModel):
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    dtype: str = "float32"
    hf_token: Optional[str] = ""


class RelevanceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    dtype: str = "float32"
    max_new_tokens: int = Field(default=20, ge=1, le=2048)
    method: str = Field(
        default="combined",
        pattern="^(attention|gradient|rollout|combined)$",
    )
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    hf_token: Optional[str] = ""


def _wrapper_key(
    model_id: str,
    device: str,
    dtype: str,
    hf_token: Optional[str],
) -> Tuple[str, str, str, str]:
    return (model_id, device, dtype, hf_token or "")


def _get_or_create_wrapper(
    model_id: str,
    device: str,
    dtype: str,
    hf_token: Optional[str],
) -> LLMRelevanceWrapper:
    key = _wrapper_key(model_id, device, dtype, hf_token)
    with _MODEL_CACHE_LOCK:
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]

        config = RelevanceConfig(
            model_id=model_id,
            auth_token=hf_token or "",
            device=device,
            torch_dtype=dtype,
        )
        wrapper = LLMRelevanceWrapper(config=config)
        _MODEL_CACHE[key] = wrapper
        return wrapper


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models() -> Dict[str, object]:
    with _MODEL_CACHE_LOCK:
        loaded = [
            {
                "model_id": k[0],
                "device": k[1],
                "dtype": k[2],
            }
            for k in _MODEL_CACHE.keys()
        ]
    return {
        "default_model_id": DEFAULT_MODEL_ID,
        "loaded_models": loaded,
    }


@app.post("/warmup")
def warmup(req: WarmupRequest) -> Dict[str, str]:
    try:
        _get_or_create_wrapper(
            model_id=req.model_id,
            device=req.device,
            dtype=req.dtype,
            hf_token=req.hf_token,
        )
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Warmup failed: {exc}",
        ) from exc


@app.post("/relevance")
def relevance(req: RelevanceRequest) -> Dict:
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    try:
        wrapper = _get_or_create_wrapper(
            model_id=req.model_id,
            device=req.device,
            dtype=req.dtype,
            hf_token=req.hf_token,
        )
        result = wrapper.generate_with_relevance(
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            calculate_relevance=True,
            relevance_method=req.method,
            alpha=req.alpha,
            verbose=False,
        )
        return _to_jsonable(result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Relevance generation failed: {exc}",
        ) from exc
