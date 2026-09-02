"""Density-aware latent landscape smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class LandscapeGrid:
    """Smoothed response surface on bin centers."""

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    density: np.ndarray


def _edges(values: np.ndarray, grid_n: int, pad: float) -> np.ndarray:
    lo, hi = np.nanpercentile(values, [0.5, 99.5])
    width = max(float(hi - lo), 1e-6)
    return np.linspace(lo - pad * width, hi + pad * width, grid_n + 1)


def smooth_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    *,
    grid_n: int = 100,
    sigma: float = 2.2,
    mask_quantile: float = 0.03,
    pad: float = 0.06,
) -> LandscapeGrid:
    """Smooth weighted observations onto a 2D grid with low-density masking.

    This follows the final response-landscape implementation: weighted and count
    histograms are Gaussian-smoothed separately, divided, then masked by support.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    if not (x.shape == y.shape == values.shape):
        raise ValueError("x, y, and values must have identical shapes")
    if grid_n < 10:
        raise ValueError("grid_n must be at least 10")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 <= mask_quantile < 1:
        raise ValueError("mask_quantile must be in [0, 1)")

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if finite.sum() < 3:
        raise ValueError("At least three finite observations are required")
    x, y, values = x[finite], y[finite], values[finite]
    x_edges = _edges(x, grid_n, pad)
    y_edges = _edges(y, grid_n, pad)

    weighted, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=values)
    counts, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    smooth_weighted = gaussian_filter(weighted, sigma=sigma)
    smooth_counts = gaussian_filter(counts, sigma=sigma)
    surface = smooth_weighted / (smooth_counts + 1e-9)

    density = smooth_counts / (np.nanmax(smooth_counts) + 1e-9)
    positive = density[density > 0]
    threshold = float(np.nanquantile(positive, mask_quantile)) if positive.size else 0.0
    surface = np.where(density > threshold, surface, np.nan)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    return LandscapeGrid(x=x_centers, y=y_centers, values=surface, density=density)
