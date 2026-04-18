"""
Annulus feature extraction — port of chromaAnnulus.cpp.

For each point along the traced edge, samples 16 evenly-spaced angular
positions at a radius that scales with trace length, collects a 15×15px
neighbourhood at each sample, and records mean intensity + std deviation.
"""

from __future__ import annotations

import math

import numpy as np


RING_POINTS = 16          # number of angular samples per annulus ring
NEIGHBOURHOOD = 15        # neighbourhood window size (pixels)
RADIUS_SCALE = 1 / 130.0  # radius = trace_length * RADIUS_SCALE


def _neighbourhood_stats(
    gray: np.ndarray,
    cy: float,
    cx: float,
    half: int,
) -> tuple[float, float]:
    """Return (mean, std) of a square neighbourhood centred at (cy, cx)."""
    rows, cols = gray.shape
    r0 = max(0, int(cy) - half)
    r1 = min(rows, int(cy) + half + 1)
    c0 = max(0, int(cx) - half)
    c1 = min(cols, int(cx) + half + 1)
    patch = gray[r0:r1, c0:c1].astype(np.float32)
    if patch.size == 0:
        return 0.0, 0.0
    return float(patch.mean()), float(patch.std())


def extract_annulus_features(
    gray: np.ndarray,
    trace: list[tuple[int, int]],
) -> np.ndarray:
    """
    Compute annulus features for every point along the trace.

    For each trace point:
      - place a ring of RING_POINTS sample positions at distance `radius`
      - collect neighbourhood mean + std at each sample position
      - result per point: 2 * RING_POINTS floats

    Returns shape (len(trace), 2 * RING_POINTS) float32 array.
    Returns empty array if trace is empty.
    """
    if not trace:
        return np.empty((0, 2 * RING_POINTS), dtype=np.float32)

    trace_length = len(trace)
    radius = max(1.0, trace_length * RADIUS_SCALE * gray.shape[0])
    half = NEIGHBOURHOOD // 2
    angles = [2 * math.pi * k / RING_POINTS for k in range(RING_POINTS)]

    features = np.zeros((trace_length, 2 * RING_POINTS), dtype=np.float32)

    for i, (row, col) in enumerate(trace):
        for j, angle in enumerate(angles):
            sy = row + radius * math.sin(angle)
            sx = col + radius * math.cos(angle)
            mean, std = _neighbourhood_stats(gray, sy, sx, half)
            features[i, 2 * j]     = mean
            features[i, 2 * j + 1] = std

    return features


def features_to_vector(features: np.ndarray, target_length: int = 256) -> np.ndarray:
    """
    Flatten and interpolate annulus feature matrix to a fixed-length vector
    suitable as CNN/MLP input.

    The original MXNet model used a fixed-size input; this resamples to
    `target_length` points regardless of original trace length.
    Output shape: (target_length * 2 * RING_POINTS,)
    """
    if features.shape[0] == 0:
        return np.zeros(target_length * 2 * RING_POINTS, dtype=np.float32)

    if features.shape[0] == target_length:
        return features.flatten().astype(np.float32)

    # Resample along trace axis using linear interpolation
    old_indices = np.linspace(0, features.shape[0] - 1, features.shape[0])
    new_indices = np.linspace(0, features.shape[0] - 1, target_length)
    resampled = np.zeros((target_length, features.shape[1]), dtype=np.float32)
    for col in range(features.shape[1]):
        resampled[:, col] = np.interp(new_indices, old_indices, features[:, col])

    return resampled.flatten().astype(np.float32)
