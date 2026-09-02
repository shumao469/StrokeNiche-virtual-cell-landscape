import numpy as np
import pytest

from strokeniche_vc.classifier import ExpressionCounterfactualModel


def test_expression_counterfactual_changes_probabilities_without_mutating_input():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(120, 4))
    labels = np.where(x[:, 0] + 0.6 * x[:, 1] > 0, "core", "remote")
    original = x.copy()
    model = ExpressionCounterfactualModel.fit(x, labels, ["Ccl2", "Ccr2", "Gpx4", "Cd44"])
    before, after = model.counterfactual_probabilities(
        x,
        ["Ccl2", "Ccr2"],
        retained_fraction=0.0,
    )
    assert before.shape == after.shape == (120, 2)
    assert np.allclose(before.sum(axis=1), 1.0)
    assert np.max(np.abs(before - after)) > 0.01
    assert np.array_equal(x, original)


def test_expression_counterfactual_rejects_unknown_genes():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    model = ExpressionCounterfactualModel.fit(x, ["a", "b", "a", "b"], ["G1", "G2"])
    with pytest.raises(ValueError, match="missing"):
        model.perturb(x, ["missing"], retained_fraction=0.5)


def test_classifier_rejects_ambiguous_features_and_missing_labels():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    try:
        ExpressionCounterfactualModel.fit(x, ["a", "b", "a", "b"], ["G1", "g1"])
    except ValueError as exc:
        assert "unique ignoring case" in str(exc)
    else:
        raise AssertionError("case-insensitive duplicate features should fail")
    try:
        ExpressionCounterfactualModel.fit(x, ["a", None, "a", "b"], ["G1", "G2"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing labels should fail")


def test_expression_counterfactual_requires_every_requested_gene_by_default():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    model = ExpressionCounterfactualModel.fit(x, ["a", "b", "a", "b"], ["Ccl2", "Ccr2"])
    with pytest.raises(ValueError, match="typo_target"):
        model.perturb(x, ["Ccl2", "typo_target"], retained_fraction=0.5)
    partial = model.perturb(
        x,
        ["Ccl2", "typo_target"],
        retained_fraction=0.5,
        allow_partial=True,
    )
    assert np.allclose(partial[:, 0], x[:, 0] * 0.5)
    assert np.array_equal(partial[:, 1], x[:, 1])


def test_expression_counterfactual_treats_one_string_as_one_gene():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    model = ExpressionCounterfactualModel.fit(x, ["a", "b", "a", "b"], ["Ccl2", "Ccr2"])
    edited = model.perturb(x, "Ccl2", retained_fraction=0.25)
    assert np.allclose(edited[:, 0], x[:, 0] * 0.25)
    assert np.array_equal(edited[:, 1], x[:, 1])
