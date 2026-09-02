"""Deterministic synthetic demo data; contains no patient or study records."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_cells(n: int = 1800, seed: int = 20260902) -> pd.DataFrame:
    """Create a three-state latent/spatial demo with realistic score structure."""

    if n < 30:
        raise ValueError("n must be at least 30")
    rng = np.random.default_rng(seed)
    states = rng.choice(
        ["lesion-core-like", "peri-infarct", "remote-like"],
        size=n,
        p=[0.32, 0.40, 0.28],
    )
    centers = {
        "lesion-core-like": (2.3, -1.4),
        "peri-infarct": (-0.2, 0.2),
        "remote-like": (-2.5, 1.3),
    }
    latent = np.array([centers[s] for s in states], dtype=float)
    latent += rng.normal(scale=[0.85, 0.72], size=(n, 2))

    state_core = np.select(
        [states == "lesion-core-like", states == "peri-infarct"],
        [0.82, 0.45],
        default=0.14,
    )
    baseline_core = np.clip(state_core + rng.normal(0, 0.09, n), 0, 1)
    baseline_repair = np.clip(
        0.72 - 0.48 * baseline_core + 0.10 * np.sin(latent[:, 1]) + rng.normal(0, 0.06, n),
        0,
        1,
    )
    timepoint = rng.choice(["D1", "D3", "D7"], size=n, p=[0.34, 0.33, 0.33])
    spatial_x = 55 + 14 * latent[:, 0] + rng.normal(0, 3.0, n)
    spatial_y = 48 - 12 * latent[:, 1] + 4 * np.sin(latent[:, 0]) + rng.normal(0, 3.0, n)
    return pd.DataFrame(
        {
            "obs_name": [f"demo_cell_{i:05d}" for i in range(n)],
            "latent1": latent[:, 0],
            "latent2": latent[:, 1],
            "spatial_x": spatial_x,
            "spatial_y": spatial_y,
            "state": states,
            "timepoint": timepoint,
            "baseline_core": baseline_core,
            "baseline_repair": baseline_repair,
        }
    )


def make_demo_effects() -> pd.DataFrame:
    """Return fully synthetic effects on biologically relevant example axes."""

    rows = [
        ("Ccl2/Ccr2-Ackr1 blockade", -0.035, 0.050, "gene", "Ccl2,Ccr2,Ackr1", "synthetic_demo_not_evidence"),
        ("Spp1-Cd44 blockade", -0.075, 0.105, "gene", "Spp1,Cd44", "synthetic_demo_not_evidence"),
        ("Vegfa-Flt1/Kdr blockade", -0.025, 0.040, "gene", "Vegfa,Flt1,Kdr", "synthetic_demo_not_evidence"),
        ("Ferroptosis down", -0.090, 0.135, "gene", "Hmox1,Slc7a11,Gpx4", "synthetic_demo_not_evidence"),
        ("Repair-ECM up", 0.070, -0.100, "gene", "Col1a1,Col3a1,Fn1", "synthetic_demo_not_evidence"),
        ("Demo CCR2 inhibitor proxy", -0.060, 0.070, "drug_proxy", "Ccr2", "synthetic_demo_not_evidence"),
        ("Demo CD44 inhibitor proxy", -0.080, 0.110, "drug_proxy", "Cd44", "synthetic_demo_not_evidence"),
        ("Demo ferroptosis inhibitor proxy", -0.100, 0.150, "drug_proxy", "Gpx4,Slc7a11", "synthetic_demo_not_evidence"),
        ("Demo VEGFR modulation proxy", -0.040, 0.050, "drug_proxy", "Flt1,Kdr", "synthetic_demo_not_evidence"),
    ]
    return pd.DataFrame(
        rows,
        columns=["perturbation", "delta_core", "delta_repair", "modality", "target_genes", "source"],
    )
