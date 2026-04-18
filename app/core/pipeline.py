"""
Full end-to-end pipeline: image bytes → embedding hash + trace coordinates.

Mirrors the R traceFromImage() / hashFromImage() flow:
  bytes → preprocess → crop → edge detect → A* trace → embedding → result
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

from .preprocessing import load_and_preprocess_bytes
from .cropping import detect_and_crop_fin
from .edge_detection import trace_fin_edge
from .features import extract_annulus_features, features_to_vector
from .model import FinEmbedder, embed_image_tensor

# ImageNet normalisation for ResNet input
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _bgr_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def process_image(
    image_bytes: bytes,
    model: FinEmbedder,
    device: str | torch.device = "cpu",
    edge_start: tuple[int, int] | None = None,
    edge_end: tuple[int, int] | None = None,
) -> dict:
    """
    Process a single fin photograph.

    Returns a dict with:
      - hash        : list[float]          512-D embedding
      - trace       : list[list[int]]      [[row, col], ...] edge coordinates
      - edge_map    : list[list[float]]    edge magnitude map (for visualisation)
      - crop_bbox   : list[int] | None     [x, y, w, h]
      - image_shape : list[int]            [height, width] after preprocessing
    """
    # 1. Load + preprocess
    img_bgr, gray = load_and_preprocess_bytes(image_bytes)

    # 2. Detect and crop fin
    cropped_bgr, cropped_gray, bbox = detect_and_crop_fin(img_bgr, gray)

    # 3. Detect trailing edge via Canny + A*
    trace, edge_map = trace_fin_edge(
        cropped_gray,
        start=edge_start,
        end=edge_end,
    )

    # 4. Annulus features (kept for potential MLP pathway / downstream analysis)
    annulus = extract_annulus_features(cropped_gray, trace)
    _ = features_to_vector(annulus)  # available for future use

    # 5. Generate embedding via CNN
    pil_img = _bgr_to_pil(cropped_bgr)
    tensor = _transform(pil_img).unsqueeze(0)
    embedding = embed_image_tensor(model, tensor, device)

    return {
        "hash":        embedding,
        "trace":       [[int(r), int(c)] for r, c in trace],
        "edge_map":    edge_map.tolist(),
        "crop_bbox":   list(bbox) if bbox is not None else None,
        "image_shape": list(cropped_bgr.shape[:2]),
    }
