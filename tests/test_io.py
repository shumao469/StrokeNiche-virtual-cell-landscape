import pandas as pd
import pytest

from strokeniche_vc.io import normalize_cell_table, normalize_effect_table


def test_cell_aliases_are_normalized():
    table = pd.DataFrame(
        {
            "x": [0, 1, 2],
            "y": [1, 2, 3],
            "core_probability": [0.2, 0.8, 0.5],
            "repair_score": [0.7, 0.1, 0.4],
        }
    )
    result = normalize_cell_table(table)
    assert {"latent1", "latent2", "baseline_core", "baseline_repair", "obs_name", "state"}.issubset(result.columns)


def test_duplicate_effect_names_are_rejected():
    table = pd.DataFrame(
        {
            "perturbation": ["same", "same"],
            "delta_core": [-0.1, -0.2],
            "delta_repair": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        normalize_effect_table(table)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_effects_are_rejected(bad):
    table = pd.DataFrame(
        {"perturbation": ["x"], "delta_core": [bad], "delta_repair": [0.1]}
    )
    with pytest.raises(ValueError, match="finite"):
        normalize_effect_table(table)


def test_out_of_range_baseline_is_rejected():
    table = pd.DataFrame(
        {
            "latent1": [0, 1, 2],
            "latent2": [0, 1, 2],
            "baseline_core": [0.2, 1.2, 0.4],
            "baseline_repair": [0.7, 0.3, 0.5],
        }
    )
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        normalize_cell_table(table)


def test_blank_names_and_too_few_cells_are_rejected():
    effects = pd.DataFrame(
        {"perturbation": ["  "], "delta_core": [-0.1], "delta_repair": [0.1]}
    )
    with pytest.raises(ValueError, match="blank"):
        normalize_effect_table(effects)
    cells = pd.DataFrame(
        {"latent1": [0, 1], "latent2": [0, 1], "baseline_core": [0.2, 0.3], "baseline_repair": [0.7, 0.6]}
    )
    with pytest.raises(ValueError, match="at least three"):
        normalize_cell_table(cells)


def test_missing_names_are_rejected_before_string_conversion():
    effects = pd.DataFrame(
        {"perturbation": [None], "delta_core": [-0.1], "delta_repair": [0.1]}
    )
    with pytest.raises(ValueError, match="missing"):
        normalize_effect_table(effects)


@pytest.mark.parametrize("column", ["delta_core", "delta_repair"])
def test_out_of_range_effect_fraction_is_rejected(column):
    effects = pd.DataFrame(
        {"perturbation": ["x"], "delta_core": [-0.1], "delta_repair": [0.1]}
    )
    effects.loc[0, column] = 20.0
    with pytest.raises(ValueError, match=r"within \[-1, 1\]"):
        normalize_effect_table(effects)
