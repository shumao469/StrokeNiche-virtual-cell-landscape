#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step78 | Virtual perturbation response landscape

Purpose
-------
Construct latent-space counterfactual response landscapes from Step72
expression-level in silico KO / blockade outputs.

Main quantities:
    delta_core(z)        = P_core_baseline(z) - P_core_perturbed(z)
    delta_repair(z)      = repair_perturbed(z) - repair_baseline(z)
    response_score(z)    = combined rescue response, if available

Outputs:
    1. Compact manuscript-ready 2D response landscape
    2. 3D core-reduction response surface
    3. Ridge / peak response map
    4. Per-cell merged response table
    5. Smoothed grid response table
    6. JSON audit report

Important interpretation
------------------------
This is a computational counterfactual response landscape derived from
expression-level in silico perturbation and surrogate/state-probability
smoothing. It is not a wet-lab KO result, not an observed cell-state
transition, and not a physical energy landscape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import glob
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import anndata as ad
except Exception as e:
    raise ImportError(
        "anndata is required. Install in your environment, e.g.:\n"
        "pip install anndata\n"
        f"Original error: {e}"
    )

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except Exception as e:
    raise ImportError(
        "scikit-learn is required. Install in your environment, e.g.:\n"
        "pip install scikit-learn\n"
        f"Original error: {e}"
    )

try:
    from scipy.ndimage import gaussian_filter, maximum_filter
except Exception as e:
    raise ImportError(
        "scipy is required. Install in your environment, e.g.:\n"
        "pip install scipy\n"
        f"Original error: {e}"
    )


# -----------------------------
# General helpers
# -----------------------------

def mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_name(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())


def find_col(
    df: pd.DataFrame,
    candidates: List[str],
    required: bool = False,
    label: str = "",
) -> Optional[str]:
    cols = list(df.columns)
    norm_map = {norm_key(c): c for c in cols}

    # Exact case-insensitive
    lower_map = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # Normalized exact
    for cand in candidates:
        k = norm_key(cand)
        if k in norm_map:
            return norm_map[k]

    # Normalized contains
    for cand in candidates:
        k = norm_key(cand)
        for col in cols:
            if k and k in norm_key(col):
                return col

    if required:
        raise ValueError(
            f"Could not find required column for {label}. "
            f"Candidates={candidates}. Available columns={cols[:80]}"
        )
    return None


def robust_minmax(x: np.ndarray, q_low=1, q_high=99) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = x.copy()
    finite = np.isfinite(out)
    if finite.sum() == 0:
        return np.zeros_like(out)
    lo, hi = np.nanpercentile(out[finite], [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(out[finite]), np.nanmax(out[finite])
    if hi <= lo:
        return np.zeros_like(out)
    out = (out - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out


def clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), 0, 1)


def sanitize_filename(s: str) -> str:
    s = clean_name(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    s = s.strip("_")
    return s if s else "auto"


# -----------------------------
# Load latent coordinates
# -----------------------------

def infer_latent_coords(
    adata,
    force_obsm_key: Optional[str] = None,
    pca_random_state: int = 0,
) -> Tuple[np.ndarray, Dict]:
    audit = {}

    if force_obsm_key:
        if force_obsm_key not in adata.obsm:
            raise ValueError(
                f"--force_obsm_key={force_obsm_key} not found in adata.obsm. "
                f"Available obsm keys: {list(adata.obsm.keys())}"
            )
        X = np.asarray(adata.obsm[force_obsm_key])
        audit["mode"] = "forced_obsm"
        audit["obsm_key"] = force_obsm_key
    else:
        priority = ["X_nicheformer", "X_scgpt", "X_umap", "X_pca", "spatial"]
        found = None
        for k in priority:
            if k in adata.obsm:
                found = k
                break
        if found is None:
            raise ValueError(f"No usable obsm key found. Available: {list(adata.obsm.keys())}")
        X = np.asarray(adata.obsm[found])
        audit["mode"] = "auto_obsm"
        audit["obsm_key"] = found

    audit["n_dim_raw"] = int(X.shape[1]) if X.ndim == 2 else None

    if X.ndim != 2:
        raise ValueError(f"obsm array must be 2D, got shape={X.shape}")

    if X.shape[1] >= 3:
        pca = PCA(n_components=2, random_state=pca_random_state)
        coords = pca.fit_transform(X)
        audit["coordinate_reduction"] = "PCA_to_2D"
        audit["pca_explained_variance_ratio"] = [
            float(v) for v in pca.explained_variance_ratio_
        ]
    elif X.shape[1] == 2:
        coords = X.copy()
        audit["coordinate_reduction"] = "already_2D"
    else:
        raise ValueError(f"Need at least 2 dimensions in obsm, got {X.shape[1]}")

    coords = np.asarray(coords, dtype=float)
    if not np.all(np.isfinite(coords)):
        good = np.isfinite(coords).all(axis=1)
        coords[~good, :] = np.nanmedian(coords[good, :], axis=0)

    return coords, audit


# -----------------------------
# State table merge
# -----------------------------

def read_csv_auto(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", **kwargs)
    return pd.read_csv(path, **kwargs)


def merge_state_table(
    adata,
    coords: np.ndarray,
    state_table_path: Optional[str],
    label_col: str,
    time_col: str,
) -> Tuple[pd.DataFrame, Dict]:
    meta = pd.DataFrame({
        "_adata_id_": adata.obs_names.astype(str),
        "latent_1": coords[:, 0],
        "latent_2": coords[:, 1],
    })

    for c in adata.obs.columns:
        if c not in meta.columns:
            meta[c] = np.asarray(adata.obs[c]).astype(str)

    audit = {
        "state_table_available": bool(state_table_path),
        "merge_mode": None,
        "state_merge_overlap": None,
    }

    if not state_table_path:
        return meta, audit

    st = read_csv_auto(state_table_path)
    st = st.copy()
    st.columns = [str(c) for c in st.columns]

    id_candidates = [
        "obs_name", "barcode", "spot_id", "cell_id", "index",
        "_obs_names_", "_adata_id_", "Unnamed: 0"
    ]

    best_key = None
    best_overlap = -1
    ad_ids = set(meta["_adata_id_"].astype(str))

    for c in st.columns:
        if c in id_candidates or norm_key(c) in [norm_key(x) for x in id_candidates]:
            vals = set(st[c].astype(str))
            ov = len(ad_ids.intersection(vals))
            if ov > best_overlap:
                best_overlap = ov
                best_key = c

    if best_key is not None and best_overlap > 0:
        st["_merge_id_"] = st[best_key].astype(str)
        meta["_merge_id_"] = meta["_adata_id_"].astype(str)
        merged = meta.merge(st, on="_merge_id_", how="left", suffixes=("", "_state"))
        audit["merge_mode"] = "id"
        audit["merge_key_state"] = best_key
        audit["state_merge_overlap"] = int(best_overlap)
    elif len(st) == len(meta):
        st = st.reset_index(drop=True)
        merged = pd.concat([meta.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
        audit["merge_mode"] = "row_order"
        audit["state_merge_overlap"] = int(len(meta))
    else:
        raise ValueError(
            "Could not merge state table by ID and row count does not match. "
            f"state rows={len(st)}, adata obs={len(meta)}"
        )

    # Keep important columns if present
    if label_col not in merged.columns:
        alt_label = find_col(
            merged,
            [label_col, "state_group", "region_refined", "region_manual_final", "region_auto"],
            required=False,
            label="label_col",
        )
        if alt_label:
            merged[label_col] = merged[alt_label]

    if time_col not in merged.columns:
        alt_time = find_col(
            merged,
            [time_col, "timepoint", "time", "day", "sample_time"],
            required=False,
            label="time_col",
        )
        if alt_time:
            merged[time_col] = merged[alt_time]

    return merged, audit


def infer_state_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = {}
    cols["core"] = find_col(df, [
        "core_probability", "core_prob", "p_core", "prob_core",
        "lesion_core_probability", "lesion-core-like_probability",
        "lesion_core_like_probability", "core_like_probability"
    ])
    cols["peri"] = find_col(df, [
        "peri_probability", "peri_prob", "p_peri", "prob_peri",
        "peri_infarct_probability", "peri-infarct_probability",
        "peri_infarct_like_probability"
    ])
    cols["remote"] = find_col(df, [
        "remote_probability", "remote_prob", "p_remote", "prob_remote",
        "remote_like_probability", "remote-like_probability"
    ])
    cols["repair"] = find_col(df, [
        "repair_score", "repair_probability", "repair_prob",
        "rescue_score", "repair_module_score", "repair_ecm_score"
    ])
    return cols


# -----------------------------
# Step72 perturbation table
# -----------------------------

ID_CANDIDATES = [
    "obs_name", "barcode", "spot_id", "cell_id", "cell", "spot",
    "index", "_obs_names_", "_adata_id_", "adata_id", "Unnamed: 0"
]

PERT_CANDIDATES = [
    "perturbation", "candidate", "target", "target_axis",
    "module", "intervention", "ko_target", "blockade", "perturb_name"
]

BASE_CORE_CANDIDATES = [
    "baseline_core_probability", "base_core_probability",
    "core_probability_baseline", "core_prob_baseline",
    "p_core_baseline", "baseline_core_prob", "base_core_prob"
]

PERT_CORE_CANDIDATES = [
    "perturbed_core_probability", "pert_core_probability",
    "core_probability_perturbed", "core_prob_perturbed",
    "p_core_perturbed", "after_core_probability",
    "pred_core_probability", "predicted_core_probability"
]

DELTA_CORE_CANDIDATES = [
    "delta_core_probability", "delta_core_prob", "delta_core",
    "d_core", "core_delta", "core_probability_delta"
]

CORE_REDUCTION_CANDIDATES = [
    "core_reduction", "delta_core_reduction",
    "core_prob_reduction", "core_probability_reduction",
    "reduced_core_probability"
]

BASE_REPAIR_CANDIDATES = [
    "baseline_repair_score", "base_repair_score",
    "repair_score_baseline", "repair_baseline",
    "base_rescue_score", "baseline_rescue_score"
]

PERT_REPAIR_CANDIDATES = [
    "perturbed_repair_score", "pert_repair_score",
    "repair_score_perturbed", "repair_perturbed",
    "after_repair_score", "pred_repair_score",
    "predicted_repair_score", "perturbed_rescue_score"
]

REPAIR_GAIN_CANDIDATES = [
    "repair_gain", "delta_repair", "delta_repair_score",
    "rescue_gain", "delta_rescue_score", "response_score",
    "state_shift_priority_score"
]


def score_step72_csv(path: Path, nrows: int = 50) -> Tuple[int, Dict]:
    info = {"path": str(path)}
    try:
        df = read_csv_auto(path, nrows=nrows)
    except Exception as e:
        info["error"] = str(e)
        return -999, info

    cols = set(df.columns)
    id_col = find_col(df, ID_CANDIDATES)
    pert_col = find_col(df, PERT_CANDIDATES)
    pert_core = find_col(df, PERT_CORE_CANDIDATES)
    delta_core = find_col(df, DELTA_CORE_CANDIDATES)
    core_red = find_col(df, CORE_REDUCTION_CANDIDATES)
    pert_rep = find_col(df, PERT_REPAIR_CANDIDATES)
    rep_gain = find_col(df, REPAIR_GAIN_CANDIDATES)

    score = 0
    name = path.name.lower()
    if "per_cell" in name or "cell" in name:
        score += 4
    if "counterfactual" in name or "insilico" in name or "ko" in name:
        score += 4
    if "summary" in name:
        score -= 3
    if "grid" in name:
        score -= 2
    if id_col:
        score += 5
    if pert_col:
        score += 3
    if pert_core or delta_core or core_red:
        score += 7
    if pert_rep or rep_gain:
        score += 5
    score += min(len(cols), 200) // 50

    info.update({
        "n_cols_sample": len(cols),
        "id_col": id_col,
        "pert_col": pert_col,
        "pert_core_col": pert_core,
        "delta_core_col": delta_core,
        "core_reduction_col": core_red,
        "pert_repair_col": pert_rep,
        "repair_gain_col": rep_gain,
        "score": score,
    })
    return score, info


def discover_step72_table(step72_dir: str | Path) -> Tuple[Path, List[Dict]]:
    step72_dir = Path(step72_dir)
    csvs = sorted(step72_dir.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under Step72 dir: {step72_dir}")

    scored = []
    for p in csvs:
        s, info = score_step72_csv(p)
        scored.append((s, p, info))

    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    best_score, best_path, _ = scored[0]
    if best_score < 5:
        raise ValueError(
            "Could not identify a suitable per-cell Step72 counterfactual table. "
            "Top candidates:\n"
            + json.dumps([x[2] for x in scored[:10]], indent=2, ensure_ascii=False)
        )
    return best_path, [x[2] for x in scored[:20]]


def load_step72_table(
    step72_cell_table: Optional[str],
    step72_dir: Optional[str],
) -> Tuple[pd.DataFrame, Dict]:
    audit = {}
    if step72_cell_table:
        path = Path(step72_cell_table)
        if not path.exists():
            raise FileNotFoundError(path)
        df = read_csv_auto(path)
        audit["source_mode"] = "explicit_cell_table"
        audit["table_path"] = str(path)
        audit["discovery_candidates"] = []
    else:
        if not step72_dir:
            raise ValueError("Provide either --step72_cell_table or --step72_dir")
        path, candidates = discover_step72_table(step72_dir)
        df = read_csv_auto(path)
        audit["source_mode"] = "discovered_from_step72_dir"
        audit["table_path"] = str(path)
        audit["discovery_candidates"] = candidates

    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    pert_col = find_col(df, PERT_CANDIDATES)
    if pert_col is None:
        df["perturbation"] = Path(audit["table_path"]).stem
        pert_col = "perturbation"

    audit["n_rows"] = int(len(df))
    audit["n_cols"] = int(df.shape[1])
    audit["perturbation_col"] = pert_col

    return df, audit


def select_perturbation(
    df: pd.DataFrame,
    perturb_col: str,
    requested: str,
) -> Tuple[str, Dict]:
    audit = {}
    df[perturb_col] = df[perturb_col].astype(str)

    available = sorted(df[perturb_col].dropna().astype(str).unique().tolist())
    audit["available_perturbations"] = available[:200]
    audit["n_available_perturbations"] = len(available)

    if requested and requested.lower() != "auto":
        # exact or fuzzy
        if requested in available:
            return requested, audit
        req_norm = norm_key(requested)
        fuzzy = [x for x in available if req_norm in norm_key(x) or norm_key(x) in req_norm]
        if fuzzy:
            audit["requested_fuzzy_matched"] = fuzzy[0]
            return fuzzy[0], audit
        raise ValueError(
            f"Requested perturbation '{requested}' not found. "
            f"Available examples: {available[:50]}"
        )

    # Auto-select using available response-like columns
    core_red_col = find_col(df, CORE_REDUCTION_CANDIDATES)
    delta_core_col = find_col(df, DELTA_CORE_CANDIDATES)
    repair_gain_col = find_col(df, REPAIR_GAIN_CANDIDATES)

    scores = []
    for name, g in df.groupby(perturb_col):
        s = 0.0
        n = len(g)
        if core_red_col:
            s += np.nanmedian(pd.to_numeric(g[core_red_col], errors="coerce").values)
        elif delta_core_col:
            # delta_core = perturbed - baseline; rescue means negative delta.
            s += -np.nanmedian(pd.to_numeric(g[delta_core_col], errors="coerce").values)

        if repair_gain_col:
            s += np.nanmedian(pd.to_numeric(g[repair_gain_col], errors="coerce").values)

        if not np.isfinite(s):
            s = -np.inf
        scores.append((s, n, name))

    scores = sorted(scores, key=lambda x: (x[0], x[1]), reverse=True)
    audit["auto_scores_top"] = [
        {"perturbation": str(n), "score": float(s), "n": int(nn)}
        for s, nn, n in scores[:20]
    ]

    if not scores or not np.isfinite(scores[0][0]):
        chosen = available[0]
    else:
        chosen = scores[0][2]

    audit["auto_selected_perturbation"] = chosen
    return chosen, audit


def merge_step72_response(
    meta: pd.DataFrame,
    step72: pd.DataFrame,
    perturbation: str,
    label_col: str,
) -> Tuple[pd.DataFrame, Dict]:
    audit = {}
    pert_col = find_col(step72, PERT_CANDIDATES, required=True, label="perturbation")
    step72[pert_col] = step72[pert_col].astype(str)

    sub = step72.loc[step72[pert_col].astype(str) == str(perturbation)].copy()
    if sub.empty:
        raise ValueError(f"No rows after filtering perturbation={perturbation}")

    audit["selected_perturbation"] = perturbation
    audit["n_rows_selected"] = int(len(sub))

    id_col = find_col(sub, ID_CANDIDATES)
    audit["step72_id_col"] = id_col

    if id_col:
        sub["_merge_id_"] = sub[id_col].astype(str)
        meta2 = meta.copy()
        meta2["_merge_id_"] = meta2["_adata_id_"].astype(str)
        merged = meta2.merge(sub, on="_merge_id_", how="left", suffixes=("", "_step72"))

        overlap = int(merged[pert_col].notna().sum())
        audit["merge_mode"] = "id"
        audit["merge_overlap"] = overlap

        if overlap < max(10, 0.1 * len(meta)):
            # Try row-order fallback if same length
            if len(sub) == len(meta):
                merged = pd.concat(
                    [meta.reset_index(drop=True), sub.reset_index(drop=True).add_suffix("_step72")],
                    axis=1
                )
                audit["merge_mode"] = "row_order_fallback"
                audit["merge_overlap"] = int(len(meta))
            else:
                raise ValueError(
                    f"Low Step72 merge overlap={overlap}. "
                    f"meta n={len(meta)}, selected perturbation rows={len(sub)}."
                )
    else:
        if len(sub) != len(meta):
            raise ValueError(
                "Step72 table has no ID column and selected perturbation row count "
                f"does not match adata obs. rows={len(sub)}, obs={len(meta)}"
            )
        merged = pd.concat(
            [meta.reset_index(drop=True), sub.reset_index(drop=True).add_suffix("_step72")],
            axis=1
        )
        audit["merge_mode"] = "row_order"
        audit["merge_overlap"] = int(len(meta))

    return merged, audit


def infer_response_columns(df: pd.DataFrame, state_cols: Dict[str, Optional[str]]) -> Tuple[pd.DataFrame, Dict]:
    audit = {}

    base_core_col = find_col(df, BASE_CORE_CANDIDATES)
    pert_core_col = find_col(df, PERT_CORE_CANDIDATES)
    delta_core_col = find_col(df, DELTA_CORE_CANDIDATES)
    core_reduction_col = find_col(df, CORE_REDUCTION_CANDIDATES)

    base_repair_col = find_col(df, BASE_REPAIR_CANDIDATES)
    pert_repair_col = find_col(df, PERT_REPAIR_CANDIDATES)
    repair_gain_col = find_col(df, REPAIR_GAIN_CANDIDATES)

    # Base core
    if base_core_col:
        base_core = pd.to_numeric(df[base_core_col], errors="coerce").values
        base_core_source = base_core_col
    elif state_cols.get("core") and state_cols["core"] in df.columns:
        base_core = pd.to_numeric(df[state_cols["core"]], errors="coerce").values
        base_core_source = state_cols["core"]
    else:
        raise ValueError(
            "Could not infer baseline core probability. "
            "Need Step72 baseline core column or state_table core_probability column."
        )

    # Perturbed core
    if pert_core_col:
        pert_core = pd.to_numeric(df[pert_core_col], errors="coerce").values
        pert_core_source = pert_core_col
    elif delta_core_col:
        delta_core_raw = pd.to_numeric(df[delta_core_col], errors="coerce").values
        pert_core = base_core + delta_core_raw
        pert_core_source = f"{base_core_source} + {delta_core_col}"
    elif core_reduction_col:
        core_red_raw = pd.to_numeric(df[core_reduction_col], errors="coerce").values
        pert_core = base_core - core_red_raw
        pert_core_source = f"{base_core_source} - {core_reduction_col}"
    else:
        raise ValueError(
            "Could not infer perturbed core probability. "
            "Need perturbed_core, delta_core, or core_reduction column in Step72 output."
        )

    base_core = clip01(base_core)
    pert_core = clip01(pert_core)

    # Core reduction positive means rescue-like decrease of core probability.
    if core_reduction_col:
        core_reduction = pd.to_numeric(df[core_reduction_col], errors="coerce").values
        core_reduction_source = core_reduction_col
    else:
        core_reduction = base_core - pert_core
        core_reduction_source = "baseline_core - perturbed_core"

    # Base repair
    if base_repair_col:
        base_repair = pd.to_numeric(df[base_repair_col], errors="coerce").values
        base_repair_source = base_repair_col
    elif state_cols.get("repair") and state_cols["repair"] in df.columns:
        base_repair = pd.to_numeric(df[state_cols["repair"]], errors="coerce").values
        base_repair_source = state_cols["repair"]
    else:
        base_repair = np.zeros(len(df), dtype=float)
        base_repair_source = "zeros_no_baseline_repair_available"

    # Normalize repair baseline to 0-1 for map comparability
    base_repair_norm = robust_minmax(base_repair)

    # Perturbed repair / repair gain
    if pert_repair_col:
        pert_repair = pd.to_numeric(df[pert_repair_col], errors="coerce").values
        pert_repair_source = pert_repair_col
        pert_repair_norm = robust_minmax(pert_repair)
        repair_gain = pert_repair_norm - base_repair_norm
        repair_gain_source = f"{pert_repair_col} normalized - baseline normalized"
    elif repair_gain_col:
        repair_gain_raw = pd.to_numeric(df[repair_gain_col], errors="coerce").values
        repair_gain = repair_gain_raw.copy()
        repair_gain_source = repair_gain_col
        # visualization-level perturbed repair
        rg_norm = robust_minmax(repair_gain_raw)
        pert_repair_norm = clip01(base_repair_norm + 0.35 * rg_norm)
        pert_repair_source = f"baseline_repair_norm + scaled({repair_gain_col})"
    else:
        repair_gain = np.zeros(len(df), dtype=float)
        repair_gain_source = "zeros_no_repair_gain_available"
        pert_repair_norm = base_repair_norm.copy()
        pert_repair_source = "baseline_repair_norm"

    out = df.copy()
    out["step78_base_core_probability"] = base_core
    out["step78_perturbed_core_probability"] = pert_core
    out["step78_core_reduction"] = core_reduction
    out["step78_base_repair_score_norm"] = base_repair_norm
    out["step78_perturbed_repair_score_norm"] = pert_repair_norm
    out["step78_repair_gain"] = repair_gain

    # Combined response score: positive means desirable rescue direction.
    cr_norm = robust_minmax(core_reduction)
    rg_norm = robust_minmax(repair_gain)
    out["step78_combined_response_score"] = 0.5 * cr_norm + 0.5 * rg_norm

    audit.update({
        "base_core_source": base_core_source,
        "pert_core_source": pert_core_source,
        "core_reduction_source": core_reduction_source,
        "base_repair_source": base_repair_source,
        "pert_repair_source": pert_repair_source,
        "repair_gain_source": repair_gain_source,
        "value_summary": {
            "base_core_mean": float(np.nanmean(base_core)),
            "pert_core_mean": float(np.nanmean(pert_core)),
            "core_reduction_mean": float(np.nanmean(core_reduction)),
            "core_reduction_q95": float(np.nanpercentile(core_reduction, 95)),
            "repair_gain_mean": float(np.nanmean(repair_gain)),
            "repair_gain_q95": float(np.nanpercentile(repair_gain, 95)),
        }
    })

    return out, audit


# -----------------------------
# Grid smoothing / landscapes
# -----------------------------

def auto_bandwidth(coords: np.ndarray) -> float:
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    k = min(15, max(3, n // 100))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(coords)
    dist, _ = nn.kneighbors(coords)
    d = dist[:, -1]
    bw = float(np.nanmedian(d) * 2.5)
    if not np.isfinite(bw) or bw <= 0:
        xr = np.nanmax(coords[:, 0]) - np.nanmin(coords[:, 0])
        yr = np.nanmax(coords[:, 1]) - np.nanmin(coords[:, 1])
        bw = 0.03 * max(xr, yr)
    return bw


def make_grid_landscape(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    grid_n: int,
    bandwidth: float,
    min_density_frac: float = 0.01,
    margin_frac: float = 0.05,
) -> Dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[good], y[good], v[good]

    xr = np.nanmax(x) - np.nanmin(x)
    yr = np.nanmax(y) - np.nanmin(y)
    margin_x = xr * margin_frac
    margin_y = yr * margin_frac

    xmin, xmax = np.nanmin(x) - margin_x, np.nanmax(x) + margin_x
    ymin, ymax = np.nanmin(y) - margin_y, np.nanmax(y) + margin_y

    x_edges = np.linspace(xmin, xmax, grid_n + 1)
    y_edges = np.linspace(ymin, ymax, grid_n + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    count, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    val_sum, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges], weights=v)

    pix_x = (xmax - xmin) / grid_n
    pix_y = (ymax - ymin) / grid_n
    pix = max(pix_x, pix_y)
    sigma_pix = max(float(bandwidth) / max(pix, 1e-12), 0.8)

    count_s = gaussian_filter(count, sigma=sigma_pix)
    val_s = gaussian_filter(val_sum, sigma=sigma_pix)

    Z = val_s / np.maximum(count_s, 1e-12)

    density = count_s / max(np.nanmax(count_s), 1e-12)
    mask = density < min_density_frac
    Z_masked = Z.copy()
    Z_masked[mask] = np.nan

    Xg, Yg = np.meshgrid(x_centers, y_centers)

    return {
        "X": Xg,
        "Y": Yg,
        "Z": Z_masked,
        "Z_raw": Z,
        "density": density,
        "mask": mask,
        "extent": [xmin, xmax, ymin, ymax],
        "x_centers": x_centers,
        "y_centers": y_centers,
        "bandwidth": float(bandwidth),
        "sigma_pix": float(sigma_pix),
        "n_used": int(len(v)),
    }


def find_peaks_on_grid(Z: np.ndarray, top_n: int = 12, min_quantile: float = 0.90) -> List[Tuple[int, int, float]]:
    Z = np.asarray(Z, dtype=float)
    finite = np.isfinite(Z)
    if finite.sum() == 0:
        return []
    threshold = np.nanquantile(Z[finite], min_quantile)
    Z_fill = np.where(finite, Z, -np.inf)
    local_max = maximum_filter(Z_fill, size=9) == Z_fill
    candidates = np.where(local_max & finite & (Z_fill >= threshold))
    peaks = [(int(i), int(j), float(Z_fill[i, j])) for i, j in zip(candidates[0], candidates[1])]
    peaks = sorted(peaks, key=lambda x: x[2], reverse=True)
    return peaks[:top_n]


# -----------------------------
# Plotting
# -----------------------------

STATE_COLORS = {
    "lesion-core-like": "#e23b42",
    "core": "#e23b42",
    "peri-infarct": "#f2b705",
    "peri": "#f2b705",
    "remote-like": "#45a9e8",
    "remote": "#45a9e8",
}


def style_dark_figure(fig):
    fig.patch.set_facecolor("#05070b")


def style_dark_ax(ax):
    ax.set_facecolor("#05070b")
    for spine in ax.spines.values():
        spine.set_color("#9aa4b2")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#d8dee9", labelsize=8)
    ax.xaxis.label.set_color("#d8dee9")
    ax.yaxis.label.set_color("#d8dee9")
    ax.title.set_color("#f1f5f9")
    ax.grid(True, color="#1b2430", linewidth=0.4, alpha=0.6)


def plot_state_points(ax, df: pd.DataFrame, label_col: str, alpha=0.28, s=2.0):
    if label_col not in df.columns:
        ax.scatter(df["latent_1"], df["latent_2"], s=s, c="#6aaed6", alpha=alpha, linewidths=0)
        return

    labels = df[label_col].astype(str).fillna("unknown")
    for lab in sorted(labels.unique()):
        idx = labels == lab
        key = lab.lower()
        color = None
        for k, c in STATE_COLORS.items():
            if k in key:
                color = c
                break
        if color is None:
            color = "#8aa1b4"
        ax.scatter(
            df.loc[idx, "latent_1"],
            df.loc[idx, "latent_2"],
            s=s,
            c=color,
            alpha=alpha,
            linewidths=0,
        )


def plot_landscape_2d(
    ax,
    grid: Dict,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    label_col: str,
    contour_color: str = "#cbd5e1",
    peak: bool = False,
    diverging: bool = False,
):
    style_dark_ax(ax)

    Z = grid["Z"]
    extent = grid["extent"]

    if diverging:
        max_abs = np.nanpercentile(np.abs(Z[np.isfinite(Z)]), 98) if np.isfinite(Z).sum() else 1
        max_abs = max(max_abs, 1e-6)
        norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
        im = ax.imshow(
            Z,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
            alpha=0.93,
            aspect="auto",
        )
    else:
        im = ax.imshow(
            Z,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            alpha=0.93,
            aspect="auto",
        )

    plot_state_points(ax, df, label_col=label_col, alpha=0.20, s=2.0)

    try:
        finite = np.isfinite(Z)
        if finite.sum() > 20:
            levels = np.nanpercentile(Z[finite], [60, 75, 85, 92, 97])
            levels = np.unique(levels)
            if len(levels) >= 2:
                ax.contour(
                    grid["X"], grid["Y"], Z,
                    levels=levels,
                    colors=contour_color,
                    linewidths=0.45,
                    alpha=0.55,
                )
    except Exception:
        pass

    if peak:
        peaks = find_peaks_on_grid(Z, top_n=12, min_quantile=0.90)
        for i, j, val in peaks:
            x = grid["X"][i, j]
            y = grid["Y"][i, j]
            ax.scatter(
                [x], [y],
                marker="*",
                s=70,
                c="#ffe66d",
                edgecolors="black",
                linewidths=0.5,
                zorder=10,
            )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Latent coordinate 1", fontsize=8)
    ax.set_ylabel("Latent coordinate 2", fontsize=8)
    return im


def plot_surface_3d(
    ax,
    grid: Dict,
    title: str,
    cmap: str = "magma",
    zlabel: str = "Smoothed response",
):
    ax.set_facecolor("#05070b")
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    Zp = np.where(np.isfinite(Z), Z, np.nan)

    finite = np.isfinite(Zp)
    if finite.sum() == 0:
        return None

    # Downsample for lighter vector/PDF output
    step = max(1, int(max(Zp.shape) / 120))
    Xs, Ys, Zs = X[::step, ::step], Y[::step, ::step], Zp[::step, ::step]

    zmin = np.nanmin(Zs)
    zmax = np.nanmax(Zs)
    if zmax > zmin:
        Zplot = (Zs - zmin) / (zmax - zmin)
    else:
        Zplot = Zs

    surf = ax.plot_surface(
        Xs, Ys, Zplot,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.95,
        rstride=1,
        cstride=1,
    )

    ax.set_title(title, color="#f1f5f9", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Latent 1", color="#d8dee9", fontsize=8)
    ax.set_ylabel("Latent 2", color="#d8dee9", fontsize=8)
    ax.set_zlabel(zlabel, color="#d8dee9", fontsize=8)

    ax.tick_params(colors="#d8dee9", labelsize=7)
    ax.xaxis.pane.set_facecolor((0.05, 0.06, 0.08, 1))
    ax.yaxis.pane.set_facecolor((0.05, 0.06, 0.08, 1))
    ax.zaxis.pane.set_facecolor((0.05, 0.06, 0.08, 1))
    ax.grid(True, color="#253040", linewidth=0.4)
    ax.view_init(elev=28, azim=-58)
    return surf


def add_panel_label(ax, label: str, is_3d: bool = False):
    if is_3d:
        ax.text2D(
            -0.08, 1.04, label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            color="white",
            va="top",
            ha="left",
        )
    else:
        ax.text(
            -0.08, 1.04, label,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            color="white",
            va="top",
            ha="left",
        )


def make_response_figures(
    df: pd.DataFrame,
    grids: Dict[str, Dict],
    outdir: Path,
    perturbation: str,
    label_col: str,
    dpi: int = 300,
) -> Dict[str, Dict[str, str]]:
    paths = {}
    safe_pert = sanitize_filename(perturbation)

    # 2D compact response landscape
    fig = plt.figure(figsize=(17, 12), dpi=dpi)
    style_dark_figure(fig)

    gs = fig.add_gridspec(
        2, 3,
        left=0.055,
        right=0.88,
        bottom=0.11,
        top=0.88,
        wspace=0.18,
        hspace=0.28,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[0, 2])
    axD = fig.add_subplot(gs[1, 0])
    axE = fig.add_subplot(gs[1, 1])
    axF = fig.add_subplot(gs[1, 2])

    imA = plot_landscape_2d(
        axA, grids["base_core"], df, "step78_base_core_probability",
        "Baseline core probability", cmap="magma",
        vmin=0, vmax=1, label_col=label_col
    )
    imB = plot_landscape_2d(
        axB, grids["pert_core"], df, "step78_perturbed_core_probability",
        "Perturbed core probability", cmap="magma",
        vmin=0, vmax=1, label_col=label_col
    )
    imC = plot_landscape_2d(
        axC, grids["core_reduction"], df, "step78_core_reduction",
        "Core-reduction response", cmap="YlGn",
        vmin=0, vmax=max(1e-6, np.nanpercentile(grids["core_reduction"]["Z"], 99)),
        label_col=label_col, contour_color="#fef08a", peak=True
    )
    imD = plot_landscape_2d(
        axD, grids["base_repair"], df, "step78_base_repair_score_norm",
        "Baseline repair score", cmap="magma",
        vmin=0, vmax=1, label_col=label_col
    )
    imE = plot_landscape_2d(
        axE, grids["pert_repair"], df, "step78_perturbed_repair_score_norm",
        "Perturbed repair score", cmap="magma",
        vmin=0, vmax=1, label_col=label_col
    )
    imF = plot_landscape_2d(
        axF, grids["repair_gain"], df, "step78_repair_gain",
        "Repair-gain response", cmap="YlGn",
        vmin=0, vmax=max(1e-6, np.nanpercentile(grids["repair_gain"]["Z"], 99)),
        label_col=label_col, contour_color="#fef08a", peak=True
    )

    for ax, lab in zip([axA, axB, axC, axD, axE, axF], list("ABCDEF")):
        add_panel_label(ax, lab)

    fig.suptitle(
        f"Virtual perturbation response landscape | {perturbation}",
        fontsize=24,
        fontweight="bold",
        color="white",
        y=0.955,
    )

    # Shared probability colorbar
    cax1 = fig.add_axes([0.895, 0.55, 0.018, 0.28])
    cb1 = fig.colorbar(imA, cax=cax1)
    cb1.set_label("Core probability / repair score", color="white", fontsize=10)
    cb1.ax.tick_params(colors="white", labelsize=8)
    cb1.outline.set_edgecolor("#d8dee9")

    cax2 = fig.add_axes([0.895, 0.18, 0.018, 0.28])
    cb2 = fig.colorbar(imC, cax=cax2)
    cb2.set_label("Positive response", color="white", fontsize=10)
    cb2.ax.tick_params(colors="white", labelsize=8)
    cb2.outline.set_edgecolor("#d8dee9")

    fig.text(
        0.50,
        0.035,
        "Counterfactual response landscapes summarize model-derived perturbation effects in latent space. "
        "Core reduction is baseline minus perturbed core probability; repair gain is perturbed minus baseline repair/rescue score. "
        "These are in silico state-editing summaries, not wet-lab KO/blockade results or observed cell-state transitions.",
        ha="center",
        va="center",
        color="#d8dee9",
        fontsize=9,
        wrap=True,
    )

    png = outdir / f"Fig_Step78_VirtualPerturbationResponseLandscape_{safe_pert}_compact.png"
    pdf = outdir / f"Fig_Step78_VirtualPerturbationResponseLandscape_{safe_pert}_compact.pdf"
    svg = outdir / f"Fig_Step78_VirtualPerturbationResponseLandscape_{safe_pert}_compact.svg"

    fig.savefig(png, dpi=dpi, facecolor=fig.get_facecolor())
    fig.savefig(pdf, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)

    paths["compact_2d"] = {"png": str(png), "pdf": str(pdf), "svg": str(svg)}

    # 3D response surface + ridge map
    fig = plt.figure(figsize=(17, 7), dpi=dpi)
    style_dark_figure(fig)
    gs = fig.add_gridspec(
        1, 2,
        left=0.055,
        right=0.88,
        bottom=0.14,
        top=0.82,
        wspace=0.20,
    )

    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1])

    surf = plot_surface_3d(
        ax1,
        grids["core_reduction"],
        title="Core-reduction response | 3D surface",
        cmap="YlGn",
        zlabel="Core reduction",
    )
    add_panel_label(ax1, "A", is_3d=True)

    im = plot_landscape_2d(
        ax2,
        grids["core_reduction"],
        df,
        "step78_core_reduction",
        "Core-reduction response | ridge / peak map",
        cmap="YlGn",
        vmin=0,
        vmax=max(1e-6, np.nanpercentile(grids["core_reduction"]["Z"], 99)),
        label_col=label_col,
        contour_color="#fef08a",
        peak=True,
    )
    add_panel_label(ax2, "B")

    fig.suptitle(
        f"Counterfactual response basin | {perturbation}",
        fontsize=24,
        fontweight="bold",
        color="white",
        y=0.95,
    )

    cax = fig.add_axes([0.90, 0.25, 0.018, 0.45])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Core-reduction response", color="white", fontsize=10)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.outline.set_edgecolor("#d8dee9")

    fig.text(
        0.50,
        0.05,
        "Ridges and peaks indicate latent regions with locally high predicted rescue-like response. "
        "This is a computational counterfactual landscape and should be interpreted as prioritization evidence only.",
        ha="center",
        va="center",
        color="#d8dee9",
        fontsize=9,
        wrap=True,
    )

    png = outdir / f"Fig_Step78_CoreReductionResponseBasin_{safe_pert}.png"
    pdf = outdir / f"Fig_Step78_CoreReductionResponseBasin_{safe_pert}.pdf"
    svg = outdir / f"Fig_Step78_CoreReductionResponseBasin_{safe_pert}.svg"

    fig.savefig(png, dpi=dpi, facecolor=fig.get_facecolor())
    fig.savefig(pdf, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)

    paths["core_response_basin"] = {"png": str(png), "pdf": str(pdf), "svg": str(svg)}

    return paths


def grid_to_dataframe(grids: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for name, g in grids.items():
        X = g["X"]
        Y = g["Y"]
        Z = g["Z"]
        D = g["density"]
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                rows.append({
                    "landscape": name,
                    "latent_1": float(X[i, j]),
                    "latent_2": float(Y[i, j]),
                    "value": safe_float(Z[i, j]),
                    "density": safe_float(D[i, j]),
                    "masked": bool(not np.isfinite(Z[i, j])),
                })
    return pd.DataFrame(rows)


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        description="Step78 virtual perturbation response landscape"
    )

    ap.add_argument("--h5ad", required=True, help="h5ad with X_nicheformer or other latent obsm")
    ap.add_argument("--state_table", required=True, help="state probability table, e.g. Step64b modeling input")
    ap.add_argument("--step72_dir", default="", help="Step72 output directory")
    ap.add_argument("--step72_cell_table", default="", help="Explicit Step72 per-cell counterfactual CSV")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--force_obsm_key", default="X_nicheformer")
    ap.add_argument("--label_col", default="state_group")
    ap.add_argument("--time_col", default="timepoint")
    ap.add_argument("--perturbation", default="auto")

    ap.add_argument("--grid_n", type=int, default=240)
    ap.add_argument("--bandwidth", default="auto", help="'auto' or float")
    ap.add_argument("--min_density_frac", type=float, default=0.008)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--random_state", type=int, default=0)

    return ap.parse_args()


def main():
    args = parse_args()
    outdir = mkdir(args.outdir)

    print("=" * 100)
    print("Step78 | Virtual perturbation response landscape")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"step72_dir={args.step72_dir}")
    print(f"step72_cell_table={args.step72_cell_table}")
    print(f"outdir={outdir}")

    adata = ad.read_h5ad(args.h5ad)

    coords, latent_audit = infer_latent_coords(
        adata,
        force_obsm_key=args.force_obsm_key,
        pca_random_state=args.random_state,
    )

    meta, state_merge_audit = merge_state_table(
        adata,
        coords,
        args.state_table,
        label_col=args.label_col,
        time_col=args.time_col,
    )

    state_cols = infer_state_columns(meta)

    step72, step72_audit = load_step72_table(
        step72_cell_table=args.step72_cell_table if args.step72_cell_table else None,
        step72_dir=args.step72_dir if args.step72_dir else None,
    )

    perturb_col = step72_audit["perturbation_col"]
    selected_pert, pert_audit = select_perturbation(
        step72,
        perturb_col=perturb_col,
        requested=args.perturbation,
    )

    merged, step72_merge_audit = merge_step72_response(
        meta,
        step72,
        perturbation=selected_pert,
        label_col=args.label_col,
    )

    response, response_audit = infer_response_columns(
        merged,
        state_cols=state_cols,
    )

    # Drop rows without latent coords
    response = response.loc[
        np.isfinite(response["latent_1"]) & np.isfinite(response["latent_2"])
    ].copy()

    x = response["latent_1"].values
    y = response["latent_2"].values
    coords2 = np.c_[x, y]

    if str(args.bandwidth).lower() == "auto":
        bw = auto_bandwidth(coords2)
    else:
        bw = float(args.bandwidth)

    grids = {}
    grid_specs = {
        "base_core": "step78_base_core_probability",
        "pert_core": "step78_perturbed_core_probability",
        "core_reduction": "step78_core_reduction",
        "base_repair": "step78_base_repair_score_norm",
        "pert_repair": "step78_perturbed_repair_score_norm",
        "repair_gain": "step78_repair_gain",
        "combined_response": "step78_combined_response_score",
    }

    for gname, col in grid_specs.items():
        grids[gname] = make_grid_landscape(
            x=x,
            y=y,
            values=response[col].values,
            grid_n=args.grid_n,
            bandwidth=bw,
            min_density_frac=args.min_density_frac,
        )

    fig_paths = make_response_figures(
        df=response,
        grids=grids,
        outdir=outdir,
        perturbation=selected_pert,
        label_col=args.label_col,
        dpi=args.dpi,
    )

    safe_pert = sanitize_filename(selected_pert)

    per_cell_csv = outdir / f"step78_per_cell_response_{safe_pert}.csv"
    response.to_csv(per_cell_csv, index=False)

    grid_csv = outdir / f"step78_grid_response_landscapes_{safe_pert}.csv"
    grid_df = grid_to_dataframe(grids)
    grid_df.to_csv(grid_csv, index=False)

    report = {
        "status": "ok",
        "h5ad": str(args.h5ad),
        "state_table": str(args.state_table),
        "step72_source": step72_audit,
        "selected_perturbation": selected_pert,
        "latent_coordinate_audit": latent_audit,
        "state_merge_audit": state_merge_audit,
        "state_cols": state_cols,
        "perturbation_selection_audit": pert_audit,
        "step72_merge_audit": step72_merge_audit,
        "response_column_audit": response_audit,
        "n_cells_for_landscape": int(len(response)),
        "bandwidth_used": float(bw),
        "grid_n": int(args.grid_n),
        "min_density_frac": float(args.min_density_frac),
        "outputs": {
            "figures": fig_paths,
            "per_cell_response_csv": str(per_cell_csv),
            "grid_response_csv": str(grid_csv),
        },
        "interpretation_note": (
            "Step78 constructs computational counterfactual response landscapes "
            "from Step72 in silico perturbation outputs. Core reduction is baseline "
            "minus perturbed core probability. Repair gain is perturbed minus baseline "
            "repair/rescue score or a Step72 response score when available. These "
            "landscapes are visualization-level summaries, not wet-lab KO/blockade "
            "results, not observed cell-state transitions, and not physical energy landscapes."
        ),
    }

    report_json = outdir / "step78_virtual_perturbation_response_landscape_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    report_txt = outdir / "step78_virtual_perturbation_response_landscape_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2, ensure_ascii=False))

    print("\nDONE Step78")
    print(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()