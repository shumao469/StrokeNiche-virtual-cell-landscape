"""Counterfactual response primitives used by the virtual-cell visualizer.

The functions in this module adapt the transparent response-proxy logic used by
the final Step78D analysis. Unlike the legacy per-candidate priority normalization,
the reusable implementation keeps one absolute weighted scale so candidate and
strength magnitudes remain comparable. It does not claim causal effects, observed
state transitions, or wet-lab perturbation outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerturbationEffect:
    """Candidate-level mean shifts applied to susceptible virtual cells.

    Parameters
    ----------
    name:
        Human-readable perturbation name.
    delta_core:
        Mean signed shift in lesion-core probability. Negative values indicate
        a reduction.
    delta_repair:
        Mean signed shift in repair score. Positive values indicate a gain.
    modality:
        For display only, for example ``gene`` or ``drug_proxy``.
    target_genes:
        Optional comma-separated targets for display and provenance.
    source:
        Short provenance label. Use ``synthetic_demo`` for illustrative values.
    proxy_from_perturbation:
        For drug proxies, the gene/pathway perturbation that supplied the numeric
        effect. Blank for direct gene rows.
    """

    name: str
    delta_core: float
    delta_repair: float
    modality: str = "gene"
    target_genes: str = ""
    source: str = "user_supplied"
    proxy_from_perturbation: str = ""

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PerturbationEffect":
        proxy_from = value.get("proxy_from_perturbation", "")
        if pd.isna(proxy_from):
            proxy_from = ""
        return cls(
            name=str(value["perturbation"]),
            delta_core=float(value["delta_core"]),
            delta_repair=float(value["delta_repair"]),
            modality=str(value.get("modality", "gene")),
            target_genes=str(value.get("target_genes", "")),
            source=str(value.get("source", "user_supplied")),
            proxy_from_perturbation=str(proxy_from),
        )


def robust_minmax(values: np.ndarray | pd.Series, qlow: float = 1.0, qhigh: float = 99.0) -> np.ndarray:
    """Scale finite values to [0, 1] using robust percentile bounds."""

    x = np.asarray(values, dtype=float)
    out = np.zeros_like(x, dtype=float)
    finite = np.isfinite(x)
    if not finite.any():
        return out
    lo, hi = np.nanpercentile(x[finite], [qlow, qhigh])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out
    out[finite] = np.clip((x[finite] - lo) / (hi - lo), 0.0, 1.0)
    return out


def compute_susceptibility(
    cells: pd.DataFrame,
    *,
    core_col: str = "baseline_core",
    repair_col: str = "baseline_repair",
) -> np.ndarray:
    """Compute the Step78D per-cell susceptibility weight.

    The weight emphasizes high-core and non-maximal-repair cells. ``RMM`` is
    robust percentile min-max scaling, applied first to each input and then to
    their weighted combination:

    ``c_tilde=RMM(core); r_tilde=RMM(repair);``
    ``s=0.25+0.75*RMM(0.65*c_tilde+0.35*(1-r_tilde))``.
    """

    missing = [c for c in (core_col, repair_col) if c not in cells.columns]
    if missing:
        raise ValueError(f"Missing required cell columns: {missing}")
    core = robust_minmax(cells[core_col])
    repair = robust_minmax(cells[repair_col])
    susceptibility = robust_minmax(0.65 * core + 0.35 * (1.0 - repair))
    return 0.25 + 0.75 * susceptibility


def classify_response(delta_core: np.ndarray, delta_repair: np.ndarray) -> str:
    """Classify mean signed shifts without resolving trade-offs arbitrarily."""

    dcore = np.asarray(delta_core, dtype=float)
    drepr = np.asarray(delta_repair, dtype=float)
    if dcore.ndim != 1 or drepr.ndim != 1 or dcore.shape != drepr.shape:
        raise ValueError("response deltas must be one-dimensional arrays with identical shapes")
    if dcore.size == 0 or drepr.size == 0 or not np.isfinite(dcore).all() or not np.isfinite(drepr).all():
        raise ValueError("response deltas must be non-empty and finite")
    mean_core = float(np.mean(dcore))
    mean_repair = float(np.mean(drepr))
    tolerance = 1e-5
    if mean_core < -tolerance and mean_repair > tolerance:
        return "rescue_positive"
    if mean_core > tolerance and mean_repair < -tolerance:
        return "weak_or_reverse"
    if abs(mean_core) <= tolerance and abs(mean_repair) <= tolerance:
        return "near_zero"
    return "mixed_tradeoff"


def simulate_counterfactual(
    cells: pd.DataFrame,
    effect: PerturbationEffect,
    *,
    strength: float = 1.0,
    core_col: str = "baseline_core",
    repair_col: str = "baseline_repair",
) -> pd.DataFrame:
    """Apply a signed candidate-level response to each virtual cell.

    Candidate-level shifts are modulated by Step78D susceptibility and clipped
    to [0, 1]. The returned table contains both signed deltas and positive rescue
    components. Input rows and extra columns are preserved.
    """

    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be a finite non-negative number")
    if not np.isfinite([effect.delta_core, effect.delta_repair]).all():
        raise ValueError("effect deltas must be finite")
    if not (-1.0 <= effect.delta_core <= 1.0 and -1.0 <= effect.delta_repair <= 1.0):
        raise ValueError("effect deltas must be fractional shifts within [-1, 1]")
    susceptibility = compute_susceptibility(cells, core_col=core_col, repair_col=repair_col)
    baseline_core = pd.to_numeric(cells[core_col], errors="coerce").to_numpy(float)
    baseline_repair = pd.to_numeric(cells[repair_col], errors="coerce").to_numpy(float)
    if not np.isfinite(baseline_core).all() or not np.isfinite(baseline_repair).all():
        raise ValueError("baseline_core and baseline_repair must contain finite numeric values")
    if ((baseline_core < 0) | (baseline_core > 1)).any() or ((baseline_repair < 0) | (baseline_repair > 1)).any():
        raise ValueError("baseline_core and baseline_repair must be within [0, 1]")

    perturbed_core = np.clip(
        baseline_core + strength * effect.delta_core * susceptibility,
        0.0,
        1.0,
    )
    perturbed_repair = np.clip(
        baseline_repair + strength * effect.delta_repair * susceptibility,
        0.0,
        1.0,
    )
    delta_core = perturbed_core - baseline_core
    delta_repair = perturbed_repair - baseline_repair
    core_reduction = np.maximum(-delta_core, 0.0)
    repair_gain = np.maximum(delta_repair, 0.0)

    out = cells.copy()
    out["susceptibility"] = susceptibility
    out["perturbation"] = effect.name
    out["modality"] = effect.modality
    out["target_genes"] = effect.target_genes
    out["effect_source"] = effect.source
    out["proxy_from_perturbation"] = effect.proxy_from_perturbation
    out["strength"] = float(strength)
    out["perturbed_core"] = perturbed_core
    out["perturbed_repair"] = perturbed_repair
    out["delta_core"] = delta_core
    out["delta_repair"] = delta_repair
    out["core_reduction"] = core_reduction
    out["repair_gain"] = repair_gain
    # Keep an absolute scale so candidates and strengths remain comparable.
    # Per-candidate min-max normalization would erase effect-magnitude changes.
    out["response_priority"] = np.clip(
        0.55 * core_reduction + 0.45 * repair_gain,
        0.0,
        1.0,
    )
    out["response_class"] = classify_response(delta_core, delta_repair)
    return out
