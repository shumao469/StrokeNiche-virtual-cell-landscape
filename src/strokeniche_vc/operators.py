"""Numerical gene-programme editing operators for StrokeNiche.

A counterfactual operator is a declared transformation of model inputs.  A
fixed classifier is evaluated before and after the transformation, and the
change in its state-probability vector is reported.  The operator itself is
not a treatment model or an experimentally observed perturbation.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


def _as_feature_indices(indices: Iterable[int], n_features: int) -> tuple[int, ...]:
    selected = tuple(dict.fromkeys(int(index) for index in indices))
    if not selected:
        raise ValueError("at least one feature index is required")
    if min(selected) < 0 or max(selected) >= n_features:
        raise ValueError("feature index is outside the expression matrix")
    return selected


def scale_log_expression(
    expression: np.ndarray,
    feature_indices: Iterable[int],
    *,
    factor: float,
) -> np.ndarray:
    """Scale selected log-normalised inputs and preserve all other features.

    ``factor`` is a multiplier on the supplied log-normalised value.  It must
    not be interpreted as a retained fraction of the original molecular count.
    """

    x = np.asarray(expression, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("expression must be a finite two-dimensional matrix")
    if not np.isfinite(factor) or factor < 0:
        raise ValueError("factor must be finite and non-negative")
    selected = _as_feature_indices(feature_indices, x.shape[1])
    edited = x.copy()
    edited[:, selected] *= float(factor)
    return edited


@dataclass(frozen=True)
class MonotoneUpOperator:
    """Training-derived, non-decreasing feature-edit operator.

    For selected feature ``g``, the transformation is

    ``x' = x + min(strength * s_g, max(c_g - x, 0))``.

    ``s_g`` is the training-set standard deviation and ``c_g`` is a
    training-derived upper reference.  Values at or above ``c_g`` are left
    unchanged, so the operator never reduces an input.  Strength zero is the
    exact identity transformation.
    """

    feature_indices: tuple[int, ...]
    scale: np.ndarray
    ceiling: np.ndarray

    def apply(self, expression: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
        x = np.asarray(expression, dtype=float)
        if x.ndim != 2 or not np.isfinite(x).all():
            raise ValueError("expression must be a finite two-dimensional matrix")
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("strength must be finite and non-negative")
        selected = _as_feature_indices(self.feature_indices, x.shape[1])
        if self.scale.shape != self.ceiling.shape or self.scale.shape != (len(selected),):
            raise ValueError("operator parameters do not match selected features")
        if not np.isfinite(self.scale).all() or not np.isfinite(self.ceiling).all():
            raise ValueError("operator parameters must be finite")
        if (self.scale < 0).any():
            raise ValueError("operator scale must be non-negative")

        edited = x.copy()
        block = edited[:, selected]
        increment = np.minimum(
            float(strength) * self.scale,
            np.maximum(self.ceiling - block, 0.0),
        )
        edited[:, selected] = block + increment
        return edited


def fit_monotone_up_operator(
    training_expression: np.ndarray,
    feature_indices: Iterable[int],
    *,
    upper_quantile: float = 0.995,
    ceiling_margin: float = 0.5,
    epsilon: float = 1e-6,
) -> MonotoneUpOperator:
    """Estimate a monotone up-modulation operator from training rows only."""

    x = np.asarray(training_expression, dtype=float)
    if x.ndim != 2 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("training_expression must be a non-empty finite matrix")
    if not 0 < upper_quantile <= 1:
        raise ValueError("upper_quantile must lie in (0, 1]")
    if not np.isfinite(ceiling_margin) or ceiling_margin < 0:
        raise ValueError("ceiling_margin must be finite and non-negative")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")

    selected = _as_feature_indices(feature_indices, x.shape[1])
    block = x[:, selected]
    scale = np.std(block, axis=0)
    ceiling = np.quantile(block, upper_quantile, axis=0) + ceiling_margin * np.maximum(
        scale, epsilon
    )
    return MonotoneUpOperator(selected, scale, ceiling)
