"""Schema adapters for visualizing demo, model-export, and Step78D tables."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


CELL_ALIASES = {
    "latent1": ("latent1", "latent_1", "x", "umap1", "UMAP_1"),
    "latent2": ("latent2", "latent_2", "y", "umap2", "UMAP_2"),
    "baseline_core": ("baseline_core", "core", "core_probability", "lesion_core_probability"),
    "baseline_repair": ("baseline_repair", "repair", "repair_score", "rescue_score"),
}


def _find_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    columns = list(columns)
    lower = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def normalize_cell_table(table: pd.DataFrame) -> pd.DataFrame:
    """Normalize common baseline cell-table aliases to the visualizer schema."""

    out = table.copy()
    if len(out) < 3:
        raise ValueError("Cell table must contain at least three rows")
    rename: dict[str, str] = {}
    missing = []
    for canonical, aliases in CELL_ALIASES.items():
        source = _find_column(out.columns, aliases)
        if source is None:
            missing.append(canonical)
        elif source != canonical:
            rename[source] = canonical
    if missing:
        raise ValueError(
            "Cell table is missing columns that can map to: " + ", ".join(missing)
        )
    out = out.rename(columns=rename)
    for column in CELL_ALIASES:
        out[column] = pd.to_numeric(out[column], errors="raise")
        if not np.isfinite(out[column].to_numpy(float)).all():
            raise ValueError(f"{column} must contain finite values")
    for column in ("baseline_core", "baseline_repair"):
        if not out[column].between(0.0, 1.0).all():
            raise ValueError(f"{column} must be within [0, 1]")
    if "obs_name" not in out.columns:
        out.insert(0, "obs_name", [f"cell_{i:05d}" for i in range(len(out))])
    if out["obs_name"].isna().any():
        raise ValueError("obs_name must not be missing")
    out["obs_name"] = out["obs_name"].astype(str).str.strip()
    if out["obs_name"].eq("").any():
        raise ValueError("obs_name must not be blank")
    if out["obs_name"].duplicated().any():
        raise ValueError("obs_name values must be unique")
    if "state" not in out.columns:
        out["state"] = "unassigned"
    else:
        out["state"] = out["state"].fillna("unassigned").astype(str)
    spatial_present = [column in out.columns for column in ("spatial_x", "spatial_y")]
    if any(spatial_present) and not all(spatial_present):
        raise ValueError("spatial_x and spatial_y must be supplied together")
    for column in ("spatial_x", "spatial_y"):
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="raise")
            if not np.isfinite(out[column].to_numpy(float)).all():
                raise ValueError(f"{column} must contain finite values")
    return out


def normalize_effect_table(table: pd.DataFrame) -> pd.DataFrame:
    """Validate a perturbation effect table and add optional metadata columns."""

    out = table.copy()
    if out.empty:
        raise ValueError("Effect table must not be empty")
    required = ["perturbation", "delta_core", "delta_repair"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Effect table is missing required columns: {missing}")
    out["delta_core"] = pd.to_numeric(out["delta_core"], errors="raise")
    out["delta_repair"] = pd.to_numeric(out["delta_repair"], errors="raise")
    if not np.isfinite(out[["delta_core", "delta_repair"]].to_numpy(float)).all():
        raise ValueError("Perturbation effects must contain finite values")
    if not out[["delta_core", "delta_repair"]].apply(lambda column: column.between(-1.0, 1.0)).all(axis=None):
        raise ValueError("delta_core and delta_repair must be fractional shifts within [-1, 1]")
    if out["perturbation"].isna().any():
        raise ValueError("Perturbation names must not be missing")
    out["perturbation"] = out["perturbation"].astype(str).str.strip()
    if out["perturbation"].eq("").any():
        raise ValueError("Perturbation names must not be blank")
    defaults = {
        "modality": "gene",
        "target_genes": "",
        "source": "user_supplied",
    }
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
        out[column] = out[column].fillna(value).astype(str)
    if "proxy_from_perturbation" in out.columns:
        out["proxy_from_perturbation"] = out["proxy_from_perturbation"].fillna("").astype(str)
    if out["perturbation"].duplicated().any():
        duplicates = out.loc[out["perturbation"].duplicated(), "perturbation"].tolist()
        raise ValueError(f"Perturbation names must be unique; duplicates: {duplicates}")
    primary = required + ["modality", "target_genes", "source"]
    extras = [column for column in out.columns if column not in primary]
    return out[primary + extras]
