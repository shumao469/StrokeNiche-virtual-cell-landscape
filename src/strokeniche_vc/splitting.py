"""Leakage-aware grouped splitting and graph isolation utilities."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def group_split_masks(
    groups: np.ndarray,
    *,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split observations so that each sample/section belongs to one split only.

    ``val_size`` is expressed as a fraction of all observations. Actual row
    fractions may differ because whole groups are assigned together.
    """

    groups = np.asarray(groups).astype(str)
    n = len(groups)
    if n < 3 or np.unique(groups).size < 3:
        raise ValueError("At least three observations and three distinct groups are required")
    if not 0 < test_size < 1 or not 0 < val_size < 1 or test_size + val_size >= 1:
        raise ValueError("test_size and val_size must be positive and sum to less than one")

    indices = np.arange(n)
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(outer.split(indices, groups=groups))
    relative_val = val_size / (1.0 - test_size)
    inner = GroupShuffleSplit(n_splits=1, test_size=relative_val, random_state=random_state + 1)
    train_local, val_local = next(
        inner.split(train_val_idx, groups=groups[train_val_idx])
    )
    train_idx = train_val_idx[train_local]
    val_idx = train_val_idx[val_local]

    masks = []
    for selected in (train_idx, val_idx, test_idx):
        mask = np.zeros(n, dtype=bool)
        mask[selected] = True
        masks.append(mask)
    return tuple(masks)  # type: ignore[return-value]


def split_labels(train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> np.ndarray:
    """Convert three disjoint masks into integer split labels."""

    masks = [np.asarray(m, dtype=bool) for m in (train_mask, val_mask, test_mask)]
    if not (masks[0].shape == masks[1].shape == masks[2].shape):
        raise ValueError("split masks must have identical shapes")
    membership = sum(m.astype(np.int8) for m in masks)
    if not np.all(membership == 1):
        raise ValueError("every observation must belong to exactly one split")
    labels = np.empty(masks[0].shape, dtype=np.int8)
    for value, mask in enumerate(masks):
        labels[mask] = value
    return labels


def cut_cross_split_edges(edge_index: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Remove graph edges that cross train/validation/test boundaries."""

    edges = np.asarray(edge_index)
    labels = np.asarray(labels)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, n_edges)")
    if edges.size and (edges.min() < 0 or edges.max() >= len(labels)):
        raise ValueError("edge_index contains an observation index outside split labels")
    keep = labels[edges[0]] == labels[edges[1]]
    return edges[:, keep]


def assert_group_isolation(groups: np.ndarray, labels: np.ndarray) -> None:
    """Raise if any group occurs in more than one split."""

    groups = np.asarray(groups).astype(str)
    labels = np.asarray(labels)
    if len(groups) != len(labels):
        raise ValueError("groups and labels must have the same length")
    for group in np.unique(groups):
        if np.unique(labels[groups == group]).size != 1:
            raise AssertionError(f"group {group!r} crosses splits")


def scale_coordinates_train_only(
    coordinates: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit coordinate scaling on training rows and transform every row."""

    coords = np.asarray(coordinates, dtype=float)
    train = np.asarray(train_mask, dtype=bool)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coordinates must have shape (n_observations, >=2)")
    if len(coords) != len(train) or not train.any():
        raise ValueError("train_mask must match coordinates and select at least one row")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates must be finite")
    scaler = StandardScaler().fit(coords[train])
    return scaler.transform(coords).astype(np.float32), scaler.mean_, scaler.scale_


def build_group_isolated_knn(
    coordinates: np.ndarray,
    groups: np.ndarray,
    *,
    k: int = 12,
) -> np.ndarray:
    """Build directed kNN edges separately inside each sample/section group."""

    coords = np.asarray(coordinates, dtype=float)
    group_values = np.asarray(groups).astype(str)
    if coords.ndim != 2 or len(coords) != len(group_values):
        raise ValueError("coordinates and groups must have matching rows")
    if not np.isfinite(coords).all():
        raise ValueError("coordinates must be finite")
    if k < 1:
        raise ValueError("k must be positive")
    source: list[int] = []
    target: list[int] = []
    for group in np.unique(group_values):
        indices = np.flatnonzero(group_values == group)
        if len(indices) < 2:
            continue
        n_neighbors = min(k + 1, len(indices))
        neighbours = NearestNeighbors(n_neighbors=n_neighbors).fit(coords[indices])
        local_indices = neighbours.kneighbors(coords[indices], return_distance=False)
        for local_query, local_neighbours in enumerate(local_indices):
            selected_neighbours = [
                local_neighbour
                for local_neighbour in local_neighbours
                if local_neighbour != local_query
            ][:k]
            for local_neighbour in selected_neighbours:
                # edge_index is source -> target: the query target attends to
                # each selected neighbour source in the graph-attention block.
                source.append(int(indices[local_neighbour]))
                target.append(int(indices[local_query]))
    if not source:
        return np.empty((2, 0), dtype=np.int64)
    return np.vstack([source, target]).astype(np.int64)
