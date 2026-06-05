"""Backend gateway service for ExplainLLM UI."""

import json
import os
from typing import Dict, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


MODEL_SERVICE_URL = os.environ.get(
    "MODEL_SERVICE_URL", "http://modelservice:8001"
).rstrip("/")

app = FastAPI(title="ExplainLLM Backend", version="0.1.0")


class WarmupRequest(BaseModel):
    model_id: str = os.environ.get(
        "MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"
    )
    device: str = "cpu"
    dtype: str = "float32"
    hf_token: Optional[str] = ""


class RelevanceRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model_id: str = os.environ.get(
        "MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct"
    )
    device: str = "cpu"
    dtype: str = "float32"
    max_new_tokens: int = Field(default=20, ge=1, le=2048)
    method: str = Field(
        default="combined",
        pattern="^(attention|gradient|rollout|combined)$",
    )
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    hf_token: Optional[str] = ""


def _request_model_service(
    method: str,
    endpoint: str,
    payload: Optional[Dict] = None,
) -> Dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urlrequest.Request(
        f"{MODEL_SERVICE_URL}{endpoint}",
        data=data,
        headers=headers,
        method=method,
    )

    try:
        with urlrequest.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise HTTPException(
            status_code=exc.code,
            detail=f"Model service error on {endpoint}: {detail}",
        ) from exc
    except urlerror.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Model service unavailable: {exc}",
        ) from exc


@app.get("/health")
def health() -> Dict[str, object]:
    upstream = _request_model_service("GET", "/health")
    return {"status": "ok", "modelservice": upstream}


@app.get("/models")
def models() -> Dict:
    return _request_model_service("GET", "/models")


@app.post("/warmup")
def warmup(req: WarmupRequest) -> Dict:
    return _request_model_service("POST", "/warmup", req.model_dump())


@app.post("/relevance")
def relevance(req: RelevanceRequest) -> Dict:
    return _request_model_service("POST", "/relevance", req.model_dump())
