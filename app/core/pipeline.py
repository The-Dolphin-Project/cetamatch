"""
Full end-to-end pipeline: image bytes → embedding hash + trace coordinates.

Mirrors the R traceFromImage() / hashFromImage() flow:
  bytes → preprocess → crop → edge detect → A* trace → embedding → result
"""

from __future__ import annotations

import cv2
import numpy as np
import tensorflow as tf

from .preprocessing import load_and_preprocess_bytes
from .cropping import detect_and_crop_fin
from .edge_detection import trace_fin_edge
from .features import extract_annulus_features, features_to_vector
from .model import embed_image_array

_IMAGE_SIZE = 600  # EfficientNet-B7 input size used during Happywhale training


def _bgr_to_efficientnet_input(bgr_image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 OpenCV image to EfficientNet-B7 model input:
      - RGB channel order
      - Resize to (_IMAGE_SIZE, _IMAGE_SIZE) with area interpolation
      - EfficientNet preprocessing: scales [0, 255] → [-1, 1]
      - Shape (1, _IMAGE_SIZE, _IMAGE_SIZE, 3), float32
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (_IMAGE_SIZE, _IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


def process_image(
    image_bytes: bytes,
    model: tf.keras.Model,
    edge_start: tuple[int, int] | None = None,
    edge_end: tuple[int, int] | None = None,
) -> dict:
    """
    Process a single fin photograph end-to-end.

    Args:
        image_bytes: Raw JPEG/PNG bytes from an HTTP upload.
        model:       The embedding sub-model returned by load_model().
        edge_start:  Optional (row, col) hint for A* trace start.
        edge_end:    Optional (row, col) hint for A* trace end.

    Returns a dict with:
      - hash        : list[float]        L2-normalised embedding vector
      - trace       : list[list[int]]    [[row, col], ...] trailing-edge coordinates
      - edge_map    : list[list[float]]  Canny edge magnitude map (for visualisation)
      - crop_bbox   : list[int] | None   [x, y, w, h] bounding box in preprocessed image
      - image_shape : list[int]          [height, width] after size-constraining
    """
    # 1. Load + resize + glare removal
    img_bgr, gray = load_and_preprocess_bytes(image_bytes)

    # 2. Blob-detect and crop to the fin region
    cropped_bgr, cropped_gray, bbox = detect_and_crop_fin(img_bgr, gray)

    # 3. Canny edge map + A* trailing-edge trace
    trace, edge_map = trace_fin_edge(
        cropped_gray,
        start=edge_start,
        end=edge_end,
    )

    # 4. Annulus features (retained for potential downstream analysis)
    annulus = extract_annulus_features(cropped_gray, trace)
    _ = features_to_vector(annulus)

    # 5. Generate embedding via EfficientNet-B7
    img_input = _bgr_to_efficientnet_input(cropped_bgr)
    embedding = embed_image_array(model, img_input)

    return {
        "hash":        embedding,
        "trace":       [[int(r), int(c)] for r, c in trace],
        "edge_map":    edge_map.tolist(),
        "crop_bbox":   list(bbox) if bbox is not None else None,
        "image_shape": list(cropped_bgr.shape[:2]),
    }
