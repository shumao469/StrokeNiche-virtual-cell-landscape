"""Section-isolated spatial-context features for StrokeNiche.

The v9 context extension uses a transparent one-hop operator.  Coordinates
define neighbours within each tissue section; labels are never used to build
the graph.  Standardisation must be fitted on training sections before these
features are constructed.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

ContextMode = Literal["expression", "self", "spatial"]


def build_section_knn_operator(
    coordinates: np.ndarray,
    sections: np.ndarray,
    *,
    k: int = 12,
) -> sparse.csr_matrix:
    """Return a row-normalised kNN operator with no self or cross-section edges."""

    coords = np.asarray(coordinates, dtype=float)
    section_values = np.asarray(sections).astype(str)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coordinates must have shape (n_observations, >=2)")
    if len(coords) != len(section_values) or len(coords) == 0:
        raise ValueError("coordinates and sections must have matching non-zero rows")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates must be finite")
    if k < 1:
        raise ValueError("k must be positive")

    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    for section in np.unique(section_values):
        members = np.flatnonzero(section_values == section)
        if len(members) < 2:
            raise ValueError(f"section {section!r} needs at least two observations")
        local_k = min(k, len(members) - 1)
        neighbours = NearestNeighbors(n_neighbors=local_k + 1).fit(coords[members])
        nearest = neighbours.kneighbors(coords[members], return_distance=False)
        for local_receiver, candidates in enumerate(nearest):
            selected = [int(value) for value in candidates if value != local_receiver][
                :local_k
            ]
            rows.extend([int(members[local_receiver])] * local_k)
            columns.extend(int(members[value]) for value in selected)
            weights.extend([1.0 / local_k] * local_k)

    operator = sparse.csr_matrix(
        (weights, (rows, columns)), shape=(len(coords), len(coords)), dtype=float
    )
    row_sums = np.asarray(operator.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 1.0):
        raise AssertionError("every context row must sum to one")
    if np.any(operator.diagonal() != 0):
        raise AssertionError("self edges are not permitted")
    return operator


def make_context_design(
    standardised_expression: np.ndarray,
    *,
    mode: ContextMode,
    operator: sparse.spmatrix | None = None,
) -> np.ndarray:
    """Construct expression, self-edge control, or spatial-context features.

    Equal ``1/sqrt(2)`` block scaling makes the duplicated self-edge design
    L2-equivalent to the expression-only design under the same regularisation.
    """

    z = np.asarray(standardised_expression, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("standardised_expression must be a finite matrix")
    if mode == "expression":
        return z.copy()
    if mode == "self":
        return np.concatenate([z, z], axis=1) / np.sqrt(2.0)
    if mode != "spatial":
        raise ValueError("mode must be 'expression', 'self', or 'spatial'")
    if operator is None or operator.shape != (len(z), len(z)):
        raise ValueError("a square operator matching expression rows is required")
    neighbour_mean = np.asarray(operator @ z)
    return np.concatenate([z, neighbour_mean], axis=1) / np.sqrt(2.0)
