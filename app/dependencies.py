"""
FastAPI dependency providers for the shared model and API key auth.
The model is loaded once at startup and reused across all requests.
"""

from __future__ import annotations

import os

import tensorflow as tf
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

from .core.model import load_model

_model: tf.keras.Model | None = None

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def init_model() -> None:
    """Called once during application lifespan startup."""
    global _model
    _model = load_model()
    print("[startup] Model ready.")


def get_model() -> tf.keras.Model:
    if _model is None:
        raise RuntimeError("Model not initialised — call init_model() at startup")
    return _model


async def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    """Validate X-API-Key header. No-op when API_KEY env var is not configured."""
    configured = os.getenv("API_KEY")
    if not configured:
        return  # auth disabled — allow all requests
    if key != configured:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
