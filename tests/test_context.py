import numpy as np

from strokeniche_vc.context import build_section_knn_operator, make_context_design


def test_section_operator_is_row_normalised_and_isolated():
    coordinates = np.array(
        [[0, 0], [1, 0], [2, 0], [100, 0], [101, 0], [102, 0]], dtype=float
    )
    sections = np.array(["D1", "D1", "D1", "D7", "D7", "D7"])
    operator = build_section_knn_operator(coordinates, sections, k=2)
    dense = operator.toarray()
    assert np.allclose(dense.sum(axis=1), 1.0)
    assert np.allclose(np.diag(dense), 0.0)
    assert np.allclose(dense[:3, 3:], 0.0)
    assert np.allclose(dense[3:, :3], 0.0)


def test_context_design_shapes_and_self_edge_l2_equivalence():
    z = np.array([[1.0, -1.0], [0.5, 2.0], [-2.0, 1.0]])
    coordinates = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    sections = np.array(["D1", "D1", "D1"])
    operator = build_section_knn_operator(coordinates, sections, k=1)
    expression = make_context_design(z, mode="expression")
    self_edge = make_context_design(z, mode="self")
    spatial = make_context_design(z, mode="spatial", operator=operator)
    assert expression.shape == (3, 2)
    assert self_edge.shape == spatial.shape == (3, 4)
    assert np.allclose(expression @ expression.T, self_edge @ self_edge.T)
    assert not np.allclose(self_edge, spatial)
