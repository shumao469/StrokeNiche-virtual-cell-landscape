import numpy as np
import pandas as pd
import pytest

from strokeniche_vc.response import (
    PerturbationEffect,
    classify_response,
    compute_susceptibility,
    simulate_counterfactual,
)


@pytest.fixture
def cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "obs_name": ["a", "b", "c", "d"],
            "latent1": [0.0, 1.0, 0.0, 1.0],
            "latent2": [0.0, 0.0, 1.0, 1.0],
            "state": ["core", "core", "peri", "remote"],
            "baseline_core": [0.95, 0.65, 0.35, 0.05],
            "baseline_repair": [0.05, 0.25, 0.55, 0.90],
        }
    )


def test_susceptibility_is_bounded_and_context_ordered(cells):
    susceptibility = compute_susceptibility(cells)
    assert np.all((0.25 <= susceptibility) & (susceptibility <= 1.0))
    assert susceptibility[0] > susceptibility[-1]


def test_rescue_effect_reduces_core_and_increases_repair(cells):
    effect = PerturbationEffect("rescue", delta_core=-0.2, delta_repair=0.3)
    out = simulate_counterfactual(cells, effect, strength=1.0)
    assert (out["delta_core"] <= 0).all()
    assert (out["delta_repair"] >= 0).all()
    assert (out["perturbed_core"].between(0, 1)).all()
    assert (out["perturbed_repair"].between(0, 1)).all()
    assert out["response_class"].eq("rescue_positive").all()


def test_reverse_classification():
    assert classify_response(np.array([0.2, 0.1]), np.array([-0.1, -0.2])) == "weak_or_reverse"


def test_opposing_sign_tie_is_mixed_not_rescue():
    assert classify_response(np.array([-0.1]), np.array([-0.1])) == "mixed_tradeoff"


def test_response_classification_rejects_mismatched_vectors():
    with pytest.raises(ValueError, match="identical shapes"):
        classify_response(np.array([-0.1, -0.2]), np.array([0.1]))


def test_negative_strength_is_rejected(cells):
    with pytest.raises(ValueError, match="non-negative"):
        simulate_counterfactual(cells, PerturbationEffect("x", -0.1, 0.1), strength=-1)


def test_effect_magnitude_is_preserved_in_priority(cells):
    weak = simulate_counterfactual(cells, PerturbationEffect("weak", -0.02, 0.03))
    strong = simulate_counterfactual(cells, PerturbationEffect("strong", -0.2, 0.3))
    assert strong["response_priority"].mean() > weak["response_priority"].mean() * 5


def test_nonfinite_effect_and_out_of_range_baseline_are_rejected(cells):
    with pytest.raises(ValueError, match="finite"):
        simulate_counterfactual(cells, PerturbationEffect("bad", np.nan, 0.1))
    invalid = cells.copy()
    invalid.loc[0, "baseline_core"] = 1.1
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        simulate_counterfactual(invalid, PerturbationEffect("x", -0.1, 0.1))
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        simulate_counterfactual(cells, PerturbationEffect("percent-error", -20.0, 10.0))
