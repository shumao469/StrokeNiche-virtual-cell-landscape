import numpy as np

from strokeniche_vc.splitting import (
    assert_group_isolation,
    build_group_isolated_knn,
    cut_cross_split_edges,
    group_split_masks,
    scale_coordinates_train_only,
    split_labels,
)


def test_group_split_and_edge_cut_are_inductive():
    groups = np.repeat(["s1", "s2", "s3", "s4", "s5", "s6"], 5)
    train, val, test = group_split_masks(groups, test_size=0.2, val_size=0.2, random_state=3)
    labels = split_labels(train, val, test)
    assert_group_isolation(groups, labels)

    source = np.arange(len(groups) - 1)
    target = source + 1
    edges = np.vstack([source, target])
    isolated = cut_cross_split_edges(edges, labels)
    assert np.all(labels[isolated[0]] == labels[isolated[1]])


def test_train_only_coordinate_scaling_and_group_knn():
    coords = np.array(
        [[0, 0], [1, 0], [100, 100], [101, 100], [200, 200], [201, 200]],
        dtype=float,
    )
    groups = np.array(["a", "a", "b", "b", "c", "c"])
    train = np.array([True, True, True, True, False, False])
    scaled, mean, scale = scale_coordinates_train_only(coords, train)
    assert np.allclose(mean, coords[train].mean(axis=0))
    assert np.all(scale > 0)
    edges = build_group_isolated_knn(scaled, groups, k=1)
    assert edges.shape[0] == 2
    assert np.all(groups[edges[0]] == groups[edges[1]])


def test_knn_edges_point_from_neighbour_to_query():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    edges = build_group_isolated_knn(coords, np.array(["one", "one", "one"]), k=1)
    edge_pairs = set(map(tuple, edges.T.tolist()))
    assert (1, 2) in edge_pairs  # point 1 is the nearest source for query point 2
    assert (2, 1) not in edge_pairs


def test_knn_ties_never_exceed_k_incoming_edges():
    coords = np.zeros((8, 2), dtype=float)
    edges = build_group_isolated_knn(coords, np.array(["same"] * 8), k=1)
    incoming = np.bincount(edges[1], minlength=8)
    assert np.all(incoming <= 1)
