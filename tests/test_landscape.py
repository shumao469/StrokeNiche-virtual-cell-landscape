import numpy as np
import pytest

from strokeniche_vc.landscape import smooth_to_grid


def test_landscape_shape_and_support_mask():
    rng = np.random.default_rng(7)
    x = rng.normal(size=400)
    y = rng.normal(size=400)
    values = np.exp(-(x**2 + y**2))
    grid = smooth_to_grid(x, y, values, grid_n=50, sigma=1.5)
    assert grid.values.shape == (50, 50)
    assert grid.density.shape == (50, 50)
    assert np.isfinite(grid.values).any()
    assert np.nanmax(grid.density) <= 1.0 + 1e-9


def test_landscape_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        smooth_to_grid(np.arange(5), np.arange(4), np.arange(5))
