"""Leakage-aware expression classifier counterfactuals.

This module extracts the reusable idea behind the Step72C analysis: learn a
balanced state classifier from a predeclared expression feature set, then edit
only selected expression columns and compare predicted state probabilities.
The result is a classifier counterfactual, not a causal knockout estimate.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ExpressionCounterfactualModel:
    """Balanced multinomial state classifier with explicit feature names."""

    feature_names: tuple[str, ...]
    scaler: StandardScaler
    classifier: LogisticRegression

    @classmethod
    def fit(
        cls,
        expression: np.ndarray,
        labels: Sequence[str],
        feature_names: Sequence[str],
        *,
        max_iter: int = 3000,
        random_state: int = 42,
    ) -> "ExpressionCounterfactualModel":
        """Fit on an already isolated training split.

        Splitting is deliberately not performed here. Callers should create
        sample/section/animal-grouped splits before calling this method.
        """

        names = tuple(str(name).strip() for name in feature_names)
        if any(not name for name in names):
            raise ValueError("feature_names must not contain blanks")
        folded = [name.casefold() for name in names]
        if len(set(folded)) != len(folded):
            raise ValueError("feature_names must be unique ignoring case")
        x = _validate_expression(expression, names)
        y_raw = pd.Series(labels, dtype="object")
        if y_raw.isna().any():
            raise ValueError("labels must not contain missing values")
        y = y_raw.astype(str).str.strip().to_numpy()
        if np.any(y == ""):
            raise ValueError("labels must not contain blanks")
        if y.ndim != 1 or len(y) != len(x):
            raise ValueError("labels must be one-dimensional and match expression rows")
        if np.unique(y).size < 2:
            raise ValueError("at least two state labels are required")
        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=max_iter,
            random_state=random_state,
        ).fit(scaler.transform(x), y)
        return cls(names, scaler, classifier)

    @property
    def classes_(self) -> np.ndarray:
        return self.classifier.classes_

    def predict_proba(self, expression: np.ndarray) -> np.ndarray:
        """Return classifier state probabilities in ``classes_`` order."""

        x = _validate_expression(expression, self.feature_names)
        return self.classifier.predict_proba(self.scaler.transform(x))

    def perturb(
        self,
        expression: np.ndarray,
        genes: Iterable[str] | str,
        *,
        retained_fraction: float,
        allow_partial: bool = False,
    ) -> np.ndarray:
        """Scale selected genes while preserving all non-target features.

        ``retained_fraction=0`` is complete in-silico suppression, ``0.2``
        retains 20% expression, and ``1`` is the unchanged baseline. These are
        expression edits and must not be described as drug doses.
        """

        if not np.isfinite(retained_fraction) or not 0 <= retained_fraction <= 1:
            raise ValueError("retained_fraction must be between 0 and 1")
        x = _validate_expression(expression, self.feature_names).copy()
        index = {name.casefold(): i for i, name in enumerate(self.feature_names)}
        gene_values = [genes] if isinstance(genes, str) else list(genes)
        requested = [str(g).strip() for g in gene_values]
        if not requested or any(not gene for gene in requested):
            raise ValueError("requested genes must not be empty or blank")
        requested_keys = list(dict.fromkeys(gene.casefold() for gene in requested))
        missing = [gene for gene in requested if gene.casefold() not in index]
        if missing and not allow_partial:
            raise ValueError(
                "requested genes are absent from the fitted feature set: "
                f"{missing}"
            )
        matched = [index[key] for key in requested_keys if key in index]
        if not matched:
            raise ValueError("none of the requested genes match the fitted feature set")
        x[:, matched] *= float(retained_fraction)
        return x

    def counterfactual_probabilities(
        self,
        expression: np.ndarray,
        genes: Iterable[str] | str,
        *,
        retained_fraction: float,
        allow_partial: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return baseline and edited classifier probabilities."""

        baseline = self.predict_proba(expression)
        edited = self.perturb(
            expression,
            genes,
            retained_fraction=retained_fraction,
            allow_partial=allow_partial,
        )
        return baseline, self.predict_proba(edited)


def _validate_expression(expression: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    x = np.asarray(expression, dtype=float)
    if x.ndim != 2:
        raise ValueError("expression must be a two-dimensional matrix")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names must match expression columns")
    if not np.isfinite(x).all():
        raise ValueError("expression must contain only finite numeric values")
    return x
