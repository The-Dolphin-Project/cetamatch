"""
Fin detection and cropping — port of blobber.R.

Detects the dominant fin blob in an image and returns a tight crop.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_and_crop_fin(
    image: np.ndarray,
    gray: np.ndarray,
    padding: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int] | None]:
    """
    Attempt to isolate the dorsal fin from the background.

    Strategy (mirrors blobber.R):
      1. Threshold on brightness to find dark fin against lighter water
      2. Find the largest contour
      3. Return a padded bounding-box crop

    Returns:
        cropped_bgr  — cropped colour image (or original if detection fails)
        cropped_gray — cropped grayscale image
        bbox         — (x, y, w, h) of the detected crop, or None
    """
    h, w = gray.shape

    # Adaptive threshold — fin is typically darker than the background
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image, gray, None

    # Largest contour by area
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    # Add padding
    pad_x = int(cw * padding)
    pad_y = int(ch * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + cw + pad_x)
    y2 = min(h, y + ch + pad_y)

    cropped_bgr  = image[y1:y2, x1:x2]
    cropped_gray = gray[y1:y2,  x1:x2]

    if cropped_bgr.size == 0:
        return image, gray, None

    return cropped_bgr, cropped_gray, (x1, y1, x2 - x1, y2 - y1)
