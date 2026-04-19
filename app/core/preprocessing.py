"""
Image preprocessing — port of R constrainSizeFinImage() and fillGlare().
"""

from __future__ import annotations

import cv2
import numpy as np


MAX_DIM = 300


def constrain_size(image: np.ndarray) -> np.ndarray:
    """Resize so the largest dimension is MAX_DIM, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale >= 1.0:
        return image
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def fill_glare(image: np.ndarray, threshold: int = 245, iterations: int = 3) -> np.ndarray:
    """
    Remove blown-out highlights by iteratively replacing over-exposed pixels
    with the mean of their valid (non-glare) neighbours.

    Port of R fillGlare() — iterative neighbour sampling.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    result = image.copy().astype(np.float32)

    glare_mask = gray >= threshold

    for _ in range(iterations):
        if not glare_mask.any():
            break
        # For each glare pixel, replace with mean of 3×3 non-glare neighbours.
        # Pad both the image and the glare mask by 1 so neighbourhood slices
        # always have shape (3, 3) even at image edges — avoiding the boolean
        # index dimension mismatch that otherwise occurs at boundary pixels.
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)) if result.ndim == 3 else ((1, 1), (1, 1)), mode="edge")
        padded_mask = np.pad(glare_mask, ((1, 1), (1, 1)), mode="constant", constant_values=False)
        ys, xs = np.where(glare_mask)
        for y, x in zip(ys, xs):
            neighbourhood = padded[y : y + 3, x : x + 3]
            neighbour_mask = ~padded_mask[y : y + 3, x : x + 3]
            if neighbour_mask.any():
                if result.ndim == 3:
                    valid = neighbourhood[neighbour_mask]
                    result[y, x] = valid.mean(axis=0)
                else:
                    valid = neighbourhood[neighbour_mask]
                    result[y, x] = valid.mean()
        # Update mask after repair
        repaired_gray = (
            cv2.cvtColor(result.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            if result.ndim == 3
            else result.astype(np.uint8)
        )
        glare_mask = repaired_gray >= threshold

    return np.clip(result, 0, 255).astype(np.uint8)


def load_and_preprocess(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image from disk, apply size constraint and glare removal.
    Returns (preprocessed_bgr, gray).
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = constrain_size(img)
    img = fill_glare(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray


def load_and_preprocess_bytes(data: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Load image from raw bytes (e.g. HTTP upload)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    img = constrain_size(img)
    img = fill_glare(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray
