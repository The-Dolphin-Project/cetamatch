"""
FastAPI dependency providers for the shared model, device, and API key auth.
Loaded once at startup, reused across all requests.
"""

from __future__ import annotations

import os

import torch
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from .core.model import FinEmbedder, load_model

_model: FinEmbedder | None = None
_device: str = "cuda" if torch.cuda.is_available() else "cpu"

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def init_model() -> None:
    global _model
    _model = load_model(device=_device)
    print(f"[startup] Model ready on {_device}")


def get_model() -> FinEmbedder:
    if _model is None:
        raise RuntimeError("Model not initialised — call init_model() at startup")
    return _model


def get_device() -> str:
    return _device


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Validate X-API-Key header. No-op when API_KEY env var is not configured."""
    configured = os.getenv("API_KEY")
    if not configured:
        return  # auth disabled — allow all requests
    if key != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
