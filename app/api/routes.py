"""
FastAPI route definitions.

Endpoints:
  POST /identify          — upload a fin photo, get embedding + top catalogue matches
  POST /catalogue/add     — add a fin (with label) to the catalogue
  GET  /catalogue         — list catalogue entries
  GET  /catalogue/{id}    — get a single entry (with embedding)
  DELETE /catalogue/{id}  — remove a fin from the catalogue
  GET  /health            — liveness check
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..core import catalogue, pipeline
from ..dependencies import get_device, get_model, require_api_key

router = APIRouter()


# ---------------------------------------------------------------------------
# Identify
# ---------------------------------------------------------------------------

@router.post("/identify", dependencies=[Depends(require_api_key)])
async def identify(
    file: Annotated[UploadFile, File(description="Fin photograph (JPEG/PNG)")],
    top_n: int = 10,
    model=Depends(get_model),
    device=Depends(get_device),
):
    """
    Process a fin photograph and return:
      - the 512-D embedding hash
      - the detected trace coordinates
      - the top-N catalogue matches (empty if catalogue is unpopulated)
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are accepted")

    image_bytes = await file.read()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image must be ≤ 20 MB")

    result = pipeline.process_image(image_bytes, model, device)
    matches = catalogue.find_matches(result["hash"], top_n=top_n)

    return {
        "hash":        result["hash"],
        "trace":       result["trace"],
        "crop_bbox":   result["crop_bbox"],
        "image_shape": result["image_shape"],
        "matches":     matches,
    }


# ---------------------------------------------------------------------------
# Catalogue — write
# ---------------------------------------------------------------------------

@router.post("/catalogue/add", dependencies=[Depends(require_api_key)])
async def add_to_catalogue(
    file: Annotated[UploadFile, File(description="Fin photograph (JPEG/PNG)")],
    label: Annotated[str, Form(description="Individual identifier / name")],
    fin_id: Annotated[str | None, Form(description="Optional stable ID (UUID generated if omitted)")] = None,
    model=Depends(get_model),
    device=Depends(get_device),
):
    """Process a fin photo and add it to the catalogue under the given label."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=415, detail="Only JPEG and PNG images are accepted")

    image_bytes = await file.read()
    result = pipeline.process_image(image_bytes, model, device)

    entry_id = fin_id or str(uuid.uuid4())
    catalogue.add_fin(entry_id, label, result["hash"])

    return {
        "id":          entry_id,
        "label":       label,
        "hash":        result["hash"],
        "trace":       result["trace"],
        "crop_bbox":   result["crop_bbox"],
        "image_shape": result["image_shape"],
        "catalogue_size": catalogue.catalogue_size(),
    }


@router.delete("/catalogue/{fin_id}", dependencies=[Depends(require_api_key)])
async def remove_from_catalogue(fin_id: str):
    """Remove a fin from the catalogue by ID."""
    removed = catalogue.delete_fin(fin_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"No fin found with id '{fin_id}'")
    return {"deleted": fin_id}


# ---------------------------------------------------------------------------
# Catalogue — read
# ---------------------------------------------------------------------------

@router.get("/catalogue", dependencies=[Depends(require_api_key)])
async def list_catalogue(limit: int = 100, offset: int = 0):
    """List catalogue entries (without embeddings)."""
    entries = catalogue.list_fins(limit=limit, offset=offset)
    return {
        "total": catalogue.catalogue_size(),
        "limit": limit,
        "offset": offset,
        "entries": entries,
    }


@router.get("/catalogue/{fin_id}", dependencies=[Depends(require_api_key)])
async def get_catalogue_entry(fin_id: str):
    """Retrieve a single catalogue entry including its embedding."""
    entry = catalogue.get_fin(fin_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No fin found with id '{fin_id}'")
    return entry


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    return {"status": "ok", "catalogue_size": catalogue.catalogue_size()}
