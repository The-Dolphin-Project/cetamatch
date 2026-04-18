"""
PyTorch triplet-loss embedding network.

Architecture:
  - ResNet50 backbone (ImageNet pretrained)
  - Replace final FC layer with a projection head → 512-D L2-normalised embedding
  - Trained with online hard triplet mining

The original finFindR used MXNet with a 4096-D embedding trained from scratch.
This version uses a pretrained backbone for better data efficiency — important
when starting from a small fin catalogue.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

EMBEDDING_DIM = 512
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/fin_embedder.pt"))
_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "finfinder")
_STORAGE_OBJECT = "models/fin_embedder.pt"


def _download_model_if_needed() -> None:
    """Download model weights from Supabase Storage if not present locally."""
    if MODEL_PATH.exists():
        return
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not supabase_url or not supabase_key:
        print("[model] SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY not set — skipping download.")
        return
    print(f"[model] Downloading weights from Supabase Storage ({_STORAGE_BUCKET}/{_STORAGE_OBJECT}) …")
    try:
        from supabase import create_client
        client = create_client(supabase_url, supabase_key)
        data = client.storage.from_(_STORAGE_BUCKET).download(_STORAGE_OBJECT)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(data)
        print(f"[model] Weights saved to {MODEL_PATH}")
    except Exception as exc:
        print(f"[model] Could not download weights: {exc}")
        print("[model] Falling back to ImageNet pretrained backbone only.")


class FinEmbedder(nn.Module):
    """ResNet50 backbone with a 512-D L2-normalised projection head."""

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the original classification head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Projection head: 2048 → 1024 → embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embedding = self.projector(features)
        return F.normalize(embedding, p=2, dim=1)  # L2-normalise


def load_model(device: str | torch.device = "cpu") -> FinEmbedder:
    """Load model weights if they exist, otherwise return the pretrained backbone."""
    _download_model_if_needed()
    model = FinEmbedder(pretrained=True)
    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[model] Loaded weights from {MODEL_PATH}")
    else:
        print("[model] No saved weights found — using ImageNet pretrained backbone only.")
        print(f"[model] Train the model and save to {MODEL_PATH}")
    model.to(device)
    model.eval()
    return model


def embed_image_tensor(
    model: FinEmbedder,
    tensor: torch.Tensor,
    device: str | torch.device = "cpu",
) -> list[float]:
    """
    Run inference on a single preprocessed image tensor.

    Args:
        tensor: shape (1, 3, H, W), normalised to ImageNet stats
    Returns:
        embedding as a Python list of floats (length EMBEDDING_DIM)
    """
    model.eval()
    with torch.no_grad():
        emb = model(tensor.to(device))
    return emb.squeeze(0).cpu().tolist()
