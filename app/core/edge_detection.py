"""
Canny edge detection + A* edge tracing.

Port of:
  - imageToFeatureProcessing.cpp  (Canny / non-max suppression)
  - astar.cpp                     (A* pathfinding on weighted edge map)
"""

from __future__ import annotations

import heapq
import math

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Edge map
# ---------------------------------------------------------------------------

def extract_edge_map(gray: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """
    Run Canny on a grayscale image and return the edge magnitude map.
    Pixels that survive non-maximum suppression have their gradient magnitude
    preserved; all others are 0.

    This mirrors extractEdgeMap() + simplifyAngles() in the original C++ code.
    """
    # Gaussian blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny gives a binary edge map; we also want magnitude for A* weights
    edges_binary = cv2.Canny(blurred, low, high)

    # Sobel gradients for magnitude
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    # Only keep magnitude where Canny kept edges
    edge_map = np.where(edges_binary > 0, magnitude, 0.0)

    # Normalise to [0, 255]
    if edge_map.max() > 0:
        edge_map = edge_map / edge_map.max() * 255.0

    return edge_map.astype(np.float32)


# ---------------------------------------------------------------------------
# A* pathfinding  (port of astar.cpp)
# ---------------------------------------------------------------------------

# Movement cost constants matching the original C++ implementation
_ORTHO_COST = 10
_DIAG_COST = 14   # ≈ 10 * sqrt(2)

# 8-connectivity: (dy, dx, base_cost)
_NEIGHBOURS = [
    (-1,  0, _ORTHO_COST),
    ( 1,  0, _ORTHO_COST),
    ( 0, -1, _ORTHO_COST),
    ( 0,  1, _ORTHO_COST),
    (-1, -1, _DIAG_COST),
    (-1,  1, _DIAG_COST),
    ( 1, -1, _DIAG_COST),
    ( 1,  1, _DIAG_COST),
]


def _heuristic(r1: int, c1: int, r2: int, c2: int) -> float:
    """Euclidean distance × 10 — matches the original C++ heuristic."""
    return math.hypot(r2 - r1, c2 - c1) * _ORTHO_COST


def astar_trace(
    edge_map: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """
    A* pathfinding on the edge map.

    Movement cost = base_cost * (1 + (255 - edge_magnitude) / 255)
    i.e. high-magnitude edges are cheap; off-edge pixels are expensive.

    Returns a list of (row, col) pixel coordinates from start to end,
    or an empty list if no path is found.
    """
    rows, cols = edge_map.shape
    sr, sc = start
    er, ec = end

    # g_cost[r][c] = cheapest cost found so far to reach (r, c)
    g_cost = np.full((rows, cols), np.inf, dtype=np.float64)
    g_cost[sr, sc] = 0.0

    # parent[r][c] = (pr, pc) for path reconstruction
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sr, sc): None}

    # Min-heap: (f_cost, g_cost, row, col)
    heap: list[tuple[float, float, int, int]] = []
    heapq.heappush(heap, (_heuristic(sr, sc, er, ec), 0.0, sr, sc))

    while heap:
        f, g, r, c = heapq.heappop(heap)

        if (r, c) == (er, ec):
            # Reconstruct path
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = (er, ec)
            while node is not None:
                path.append(node)
                node = parent.get(node)
            path.reverse()
            return path

        # Skip stale heap entries
        if g > g_cost[r, c]:
            continue

        for dr, dc, base in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue

            # Edge pixels are cheap; non-edge pixels are expensive
            edge_val = edge_map[nr, nc]
            move_cost = base * (1.0 + (255.0 - edge_val) / 255.0)

            new_g = g + move_cost
            if new_g < g_cost[nr, nc]:
                g_cost[nr, nc] = new_g
                parent[(nr, nc)] = (r, c)
                f_new = new_g + _heuristic(nr, nc, er, ec)
                heapq.heappush(heap, (f_new, new_g, nr, nc))

    return []  # No path found


# ---------------------------------------------------------------------------
# Automatic start/end detection for the trailing edge
# ---------------------------------------------------------------------------

def detect_trailing_edge_endpoints(
    edge_map: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Heuristic to find the start and end points for A* tracing.

    For a dorsal fin the trailing edge typically runs from a point near the
    top of the fin to the base. We find the two strongest edge pixels that
    are farthest apart along the vertical axis as a robust default.
    """
    rows, cols = edge_map.shape

    # Candidate pixels: top 5% by edge magnitude
    threshold = np.percentile(edge_map[edge_map > 0], 95) if (edge_map > 0).any() else 1
    candidates = np.argwhere(edge_map >= threshold)

    if len(candidates) < 2:
        # Fallback: top-centre → bottom-centre
        return (0, cols // 2), (rows - 1, cols // 2)

    # Pick topmost and bottommost candidate
    top_idx = candidates[:, 0].argmin()
    bot_idx = candidates[:, 0].argmax()
    return tuple(candidates[top_idx]), tuple(candidates[bot_idx])


def trace_fin_edge(
    gray: np.ndarray,
    start: tuple[int, int] | None = None,
    end: tuple[int, int] | None = None,
    canny_low: int = 50,
    canny_high: int = 150,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """
    Full edge-tracing pipeline for one fin image.

    Returns:
        trace  — list of (row, col) coordinates along the trailing edge
        edge_map — the computed edge magnitude map (useful for visualisation)
    """
    edge_map = extract_edge_map(gray, low=canny_low, high=canny_high)

    if start is None or end is None:
        start, end = detect_trailing_edge_endpoints(edge_map)

    trace = astar_trace(edge_map, start, end)
    return trace, edge_map
