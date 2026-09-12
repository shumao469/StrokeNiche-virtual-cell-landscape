import numpy as np
import pytest

from strokeniche_vc.operators import (
    fit_monotone_up_operator,
    scale_log_expression,
)


def test_log_expression_scaling_is_explicit_and_non_mutating():
    x = np.array([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
    original = x.copy()
    edited = scale_log_expression(x, [0, 2], factor=0.25)
    assert np.allclose(edited[:, [0, 2]], x[:, [0, 2]] * 0.25)
    assert np.array_equal(edited[:, 1], x[:, 1])
    assert np.array_equal(x, original)


def test_monotone_operator_never_decreases_and_zero_is_identity():
    train = np.array(
        [[0.0, 1.0, 8.0], [1.0, 2.0, 9.0], [2.0, 3.0, 10.0]], dtype=float
    )
    x = np.array([[0.5, 1.5, 20.0], [4.0, 5.0, 9.5]], dtype=float)
    operator = fit_monotone_up_operator(train, [1, 2])
    edited = operator.apply(x, strength=1.0)
    identity = operator.apply(x, strength=0.0)
    assert np.all(edited >= x)
    assert np.array_equal(identity, x)
    assert np.array_equal(edited[:, 0], x[:, 0])
    assert edited[0, 2] == x[0, 2]


def test_operator_rejects_invalid_feature_indices():
    with pytest.raises(ValueError, match="outside"):
        fit_monotone_up_operator(np.ones((3, 2)), [2])
