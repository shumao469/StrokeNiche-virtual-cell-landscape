#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step78C | Final virtual perturbation response landscape

Purpose
-------
Generate manuscript-ready latent-space virtual perturbation response landscapes.

Inputs
------
1. h5ad with X_nicheformer in .obsm
2. state probability table from Step64/Step66-compatible state modeling
3. Step72/Step72C summary table if available

Main outputs
------------
A. Atlas:
   rows = canonical perturbations
   columns = baseline core probability, core reduction, repair gain, response priority

B. Single-candidate 4-panel figures:
   baseline core, core reduction, repair gain, response-priority basin map

Notes
-----
This is a visualization-level counterfactual response landscape.
It is not wet-lab KO/blockade, not observed cell-fate transition, and not physical energy.
"""

import os
import re
import json
import math
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

try:
    import anndata as ad
except Exception:
    ad = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None

try:
    from scipy.ndimage import gaussian_filter, maximum_filter
except Exception:
    gaussian_filter = None
    maximum_filter = None


# -----------------------------
# Basic helpers
# -----------------------------

def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def norm_col(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def sanitize_filename(x):
    x = str(x)
    x = re.sub(r"[^\w\-.]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x[:120] if len(x) > 120 else x


def normalize01(v, qlo=1, qhi=99):
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return out
    lo = np.nanpercentile(v[finite], qlo)
    hi = np.nanpercentile(v[finite], qhi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = np.nanmax(v[finite])
        lo = np.nanmin(v[finite])
    if hi <= lo:
        out[finite] = 0.0
    else:
        out[finite] = (v[finite] - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def find_col(df, candidates, required=False):
    if df is None:
        return None
    cmap = {norm_col(c): c for c in df.columns}
    for c in candidates:
        nc = norm_col(c)
        if nc in cmap:
            return cmap[nc]
    for c in candidates:
        nc = norm_col(c)
        for k, orig in cmap.items():
            if nc in k or k in nc:
                return orig
    if required:
        raise ValueError(f"Cannot find required column among: {candidates}")
    return None


# -----------------------------
# Perturbation canonicalization
# -----------------------------

CANONICAL_ORDER = [
    "Ccl2/Ccr2-Ackr1 blockade",
    "Spp1-Cd44 blockade",
    "Vegfa-Flt1/Kdr blockade",
    "Ferroptosis down",
    "Repair-ECM up",
]

CANONICAL_ALIAS = {
    "Ccl2/Ccr2-Ackr1 blockade": [
        "ccl2", "ccr2", "ackr1", "ccl2_ccr2", "ccl2/ccr2", "ccl2-ccr2",
    ],
    "Spp1-Cd44 blockade": [
        "spp1", "cd44", "spp1_cd44", "spp1-cd44", "spp1/cd44",
    ],
    "Vegfa-Flt1/Kdr blockade": [
        "vegfa", "flt1", "kdr", "vegfa_flt1", "vegfa-flt1", "vegfa/fllt1",
    ],
    "Ferroptosis down": [
        "ferroptosis", "ferroptosis_down", "ferroptosis down",
        "ferroptosis_down_modulation", "ferroptosis down-modulation",
        "ferroptosis down modulation",
    ],
    "Repair-ECM up": [
        "repair_ecm", "repair ecm", "repair-ecm", "repair_ecm_up",
        "repair ecm up", "repair-ecm up", "repair_ecm_promotion",
        "repair ecm promotion", "repair-ecm promotion",
    ],
}


def canonicalize_perturbation_name(x):
    s = str(x).strip()
    sl = norm_col(s)

    for canon in CANONICAL_ORDER:
        aliases = CANONICAL_ALIAS[canon]
        for a in aliases:
            aa = norm_col(a)
            if aa and aa in sl:
                if canon == "Vegfa-Flt1/Kdr blockade":
                    if ("vegfa" in sl) or ("flt1" in sl) or ("kdr" in sl):
                        return canon
                else:
                    return canon

    return None


# Step72C screenshot-derived fallback values.
# Used only when no valid Step72 summary CSV can be discovered.
BUILTIN_EFFECTS = pd.DataFrame([
    {
        "perturbation": "Ccl2/Ccr2-Ackr1 blockade",
        "delta_core": -0.0290,
        "delta_peri": -0.0076,
        "delta_remote": 0.0368,
        "delta_repair": 0.0438,
        "source": "builtin_fallback_from_step72c_visual_summary",
    },
    {
        "perturbation": "Spp1-Cd44 blockade",
        "delta_core": -0.0953,
        "delta_peri": -0.0921,
        "delta_remote": 0.1874,
        "delta_repair": 0.1429,
        "source": "builtin_fallback_from_step72c_visual_summary",
    },
    {
        "perturbation": "Vegfa-Flt1/Kdr blockade",
        "delta_core": -0.0224,
        "delta_peri": -0.0102,
        "delta_remote": 0.0326,
        "delta_repair": 0.0337,
        "source": "builtin_fallback_from_step72c_visual_summary",
    },
    {
        "perturbation": "Ferroptosis down",
        "delta_core": -0.1084,
        "delta_peri": -0.0358,
        "delta_remote": 0.1442,
        "delta_repair": 0.1627,
        "source": "builtin_fallback_from_step72c_visual_summary",
    },
    {
        "perturbation": "Repair-ECM up",
        "delta_core": 0.0962,
        "delta_peri": 0.0521,
        "delta_remote": -0.1483,
        "delta_repair": -0.1443,
        "source": "builtin_fallback_from_step72c_visual_summary",
    },
])


# -----------------------------
# Input loading
# -----------------------------

def load_h5ad(path):
    if ad is None:
        raise ImportError("anndata is not installed. Please run in an environment with anndata.")
    return ad.read_h5ad(path)


def infer_latent_coordinates(adata, force_obsm_key="X_nicheformer", random_state=0):
    audit = {}

    if force_obsm_key:
        if force_obsm_key not in adata.obsm:
            raise ValueError(
                f"--force_obsm_key={force_obsm_key} not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        X = np.asarray(adata.obsm[force_obsm_key])
        audit["mode"] = "forced_obsm"
        audit["obsm_key"] = force_obsm_key
    elif "X_nicheformer" in adata.obsm:
        X = np.asarray(adata.obsm["X_nicheformer"])
        audit["mode"] = "auto_obsm"
        audit["obsm_key"] = "X_nicheformer"
    elif "X_umap" in adata.obsm:
        X = np.asarray(adata.obsm["X_umap"])
        audit["mode"] = "auto_obsm"
        audit["obsm_key"] = "X_umap"
    elif "spatial" in adata.obsm:
        X = np.asarray(adata.obsm["spatial"])
        audit["mode"] = "auto_obsm"
        audit["obsm_key"] = "spatial"
    else:
        raise ValueError(f"No usable obsm coordinate found. Available keys: {list(adata.obsm.keys())}")

    audit["n_dim_original"] = int(X.shape[1]) if X.ndim == 2 else None

    if X.ndim != 2:
        raise ValueError("Selected obsm must be a 2D matrix.")

    if X.shape[1] >= 3:
        if PCA is None:
            raise ImportError("sklearn is required for PCA projection from high-dimensional embeddings.")
        X2 = PCA(n_components=2, random_state=random_state).fit_transform(X)
        audit["projection"] = "PCA_to_2D"
    else:
        X2 = X[:, :2].copy()
        audit["projection"] = "direct_first_two_dims"

    # Robust scaling only for visualization.
    X2 = np.asarray(X2, dtype=float)
    for j in range(2):
        med = np.nanmedian(X2[:, j])
        q1, q99 = np.nanpercentile(X2[:, j], [1, 99])
        scale = (q99 - q1) / 8.0
        if not np.isfinite(scale) or scale <= 0:
            scale = np.nanstd(X2[:, j])
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0
        X2[:, j] = (X2[:, j] - med) / scale

    audit["scaled_for_visualization"] = True
    return X2, audit


def load_state_table(path, adata_obs_names):
    st = pd.read_csv(path)
    st = st.copy()

    meta = pd.DataFrame({"_adata_id_": pd.Index(adata_obs_names).astype(str)})

    key_candidates = [
        "obs_name", "spot", "barcode", "cell_id", "cell", "id",
        "_adata_id_", "adata_id", "index",
    ]

    merge_key = None
    best_overlap = -1

    for c in key_candidates:
        if c in st.columns:
            vals = st[c].astype(str)
            overlap = vals.isin(meta["_adata_id_"]).sum()
            if overlap > best_overlap:
                best_overlap = overlap
                merge_key = c

    if merge_key is not None and best_overlap > 0:
        st["_state_id_"] = st[merge_key].astype(str)
        out = meta.merge(st, left_on="_adata_id_", right_on="_state_id_", how="left")
        merge_mode = "id"
        overlap = int(best_overlap)
    elif len(st) == len(meta):
        out = pd.concat([meta.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
        merge_mode = "row_order"
        merge_key = None
        overlap = int(len(st))
    else:
        raise ValueError(
            f"Cannot merge state table. state rows={len(st)}, adata rows={len(meta)}, "
            f"candidate id overlap={best_overlap}"
        )

    audit = {
        "state_table": path,
        "state_rows": int(len(st)),
        "adata_rows": int(len(meta)),
        "merge_mode": merge_mode,
        "merge_key_state": merge_key,
        "overlap": overlap,
    }
    return out, audit


def infer_prob_columns(df, label_col=None):
    prob_cols = {}

    prob_cols["core"] = find_col(df, [
        "core_probability", "core_prob", "p_core", "prob_core",
        "lesion_core_probability", "lesion_core_prob",
    ])
    prob_cols["peri"] = find_col(df, [
        "peri_probability", "peri_prob", "p_peri", "prob_peri",
        "peri_infarct_probability", "peri_infarct_prob",
    ])
    prob_cols["remote"] = find_col(df, [
        "remote_probability", "remote_prob", "p_remote", "prob_remote",
        "remote_like_probability", "remote_like_prob",
    ])
    prob_cols["repair"] = find_col(df, [
        "repair_score", "rescue_score", "repair_probability", "p_repair",
        "repair_prob", "state_repair_score",
    ])

    missing = [k for k in ["core", "peri", "remote"] if prob_cols[k] is None]

    # If probabilities are missing, derive state one-hot from label.
    used_label = None
    if missing:
        label_candidates = []
        if label_col:
            label_candidates.append(label_col)
        label_candidates += [
            "state_group", "region_refined", "region_manual_final", "region_auto",
            "state_label", "label", "group",
        ]
        used_label = find_col(df, label_candidates)
        if used_label is None:
            raise ValueError(
                f"Cannot find probability columns {missing}, and no usable label column found."
            )

        lab = df[used_label].astype(str).str.lower()
        df["core_probability_from_label"] = lab.str.contains("core|lesion").astype(float)
        df["peri_probability_from_label"] = lab.str.contains("peri").astype(float)
        df["remote_probability_from_label"] = lab.str.contains("remote").astype(float)

        if prob_cols["core"] is None:
            prob_cols["core"] = "core_probability_from_label"
        if prob_cols["peri"] is None:
            prob_cols["peri"] = "peri_probability_from_label"
        if prob_cols["remote"] is None:
            prob_cols["remote"] = "remote_probability_from_label"

    if prob_cols["repair"] is None:
        # Conservative visualization fallback.
        df["repair_score_from_remote_minus_core"] = np.clip(
            df[prob_cols["remote"]].astype(float) - df[prob_cols["core"]].astype(float),
            -1, 1
        )
        prob_cols["repair"] = "repair_score_from_remote_minus_core"

    return prob_cols, used_label, df


# -----------------------------
# Step72 effect loading
# -----------------------------

def discover_csvs(path):
    if not path:
        return []
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".csv":
        return [str(p)]
    if not p.exists():
        return []
    return sorted([str(x) for x in p.rglob("*.csv")])


def extract_effects_from_csv(csv_path):
    out_rows = []

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return out_rows

    if df.empty:
        return out_rows

    name_col = find_col(df, [
        "perturbation", "candidate", "target", "target_axis", "module",
        "name", "label", "perturbation_name", "candidate_name",
    ])

    if name_col is None:
        return out_rows

    dcore_col = find_col(df, [
        "delta_core", "d_core", "mean_delta_core", "delta_core_probability",
        "delta_core_prob", "core_delta", "delta_core_prob_mean",
        "Δ core", "delta core",
    ])

    dperi_col = find_col(df, [
        "delta_peri", "d_peri", "mean_delta_peri", "delta_peri_probability",
        "delta_peri_prob", "peri_delta", "Δ peri", "delta peri",
    ])

    dremote_col = find_col(df, [
        "delta_remote", "d_remote", "mean_delta_remote", "delta_remote_probability",
        "delta_remote_prob", "remote_delta", "Δ remote", "delta remote",
    ])

    drepiar_col = find_col(df, [
        "delta_repair", "d_repair", "mean_delta_repair", "delta_repair_score",
        "repair_delta", "rescue_score", "delta_rescue_score",
        "mean_rescue_shift", "repair_shift", "state_shift_priority_score",
        "Δ repair", "delta repair", "rescue shift",
    ])

    # Also support before/after columns.
    base_core_col = find_col(df, ["baseline_core", "base_core", "core_before", "mean_core_before"])
    pert_core_col = find_col(df, ["perturbed_core", "core_after", "mean_core_after"])
    base_repair_col = find_col(df, ["baseline_repair", "base_repair", "repair_before", "mean_repair_before"])
    pert_repair_col = find_col(df, ["perturbed_repair", "repair_after", "mean_repair_after"])

    for _, r in df.iterrows():
        raw_name = r.get(name_col)
        canon = canonicalize_perturbation_name(raw_name)
        if canon is None:
            continue

        dc = safe_float(r.get(dcore_col)) if dcore_col else np.nan
        dp = safe_float(r.get(dperi_col)) if dperi_col else np.nan
        dr = safe_float(r.get(dremote_col)) if dremote_col else np.nan
        drep = safe_float(r.get(drepiar_col)) if drepiar_col else np.nan

        if not np.isfinite(dc) and base_core_col and pert_core_col:
            dc = safe_float(r.get(pert_core_col)) - safe_float(r.get(base_core_col))

        if not np.isfinite(drep) and base_repair_col and pert_repair_col:
            drep = safe_float(r.get(pert_repair_col)) - safe_float(r.get(base_repair_col))

        if not np.isfinite(dc):
            continue

        out_rows.append({
            "perturbation": canon,
            "raw_name": str(raw_name),
            "delta_core": dc,
            "delta_peri": dp,
            "delta_remote": dr,
            "delta_repair": drep,
            "source": csv_path,
        })

    return out_rows


def load_step72_effects(step72_dir, step72_summary=None, allow_builtin=True):
    csvs = []
    if step72_summary:
        csvs.append(step72_summary)
    csvs += discover_csvs(step72_dir)

    all_rows = []
    for p in csvs:
        all_rows.extend(extract_effects_from_csv(p))

    if len(all_rows) == 0:
        if not allow_builtin:
            raise FileNotFoundError(
                "No valid Step72 effect CSV found and builtin fallback is disabled."
            )
        effects = BUILTIN_EFFECTS.copy()
        audit = {
            "mode": "builtin_fallback",
            "n_effect_rows_raw": 0,
            "csvs_scanned": csvs,
            "warning": "No valid Step72 effect CSV found; using built-in Step72C visual-summary fallback values.",
        }
        return effects, audit

    raw = pd.DataFrame(all_rows)

    # Deduplicate: keep one row per canonical perturbation, choosing largest positive response priority.
    raw["core_reduction_mean"] = np.maximum(-raw["delta_core"].astype(float), 0.0)
    raw["repair_gain_mean"] = np.maximum(raw["delta_repair"].fillna(0).astype(float), 0.0)
    raw["response_score_for_dedup"] = raw["core_reduction_mean"] + raw["repair_gain_mean"]

    chosen = []
    for canon in CANONICAL_ORDER:
        sub = raw[raw["perturbation"] == canon].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("response_score_for_dedup", ascending=False)
        chosen.append(sub.iloc[0].to_dict())

    effects = pd.DataFrame(chosen)

    # Ensure canonical order and no duplicates.
    effects["order"] = effects["perturbation"].map({x: i for i, x in enumerate(CANONICAL_ORDER)})
    effects = effects.sort_values("order").drop(columns=["order"])

    audit = {
        "mode": "csv_discovered",
        "n_effect_rows_raw": int(len(raw)),
        "n_effect_rows_deduplicated": int(len(effects)),
        "csvs_scanned": csvs,
        "chosen_effect_sources": effects[["perturbation", "raw_name", "source"]].to_dict(orient="records"),
    }
    return effects, audit


# -----------------------------
# Response construction
# -----------------------------

def make_response_per_cell(df, prob_cols, effects, args):
    x = df["latent_x"].to_numpy(float)
    y = df["latent_y"].to_numpy(float)

    core = np.clip(df[prob_cols["core"]].to_numpy(float), 0, 1)
    peri = np.clip(df[prob_cols["peri"]].to_numpy(float), 0, 1)
    remote = np.clip(df[prob_cols["remote"]].to_numpy(float), 0, 1)

    repair_raw = df[prob_cols["repair"]].to_numpy(float)
    repair_norm = normalize01(repair_raw, 1, 99)

    # Susceptibility maps:
    # core susceptibility: high core-probability cells are more rescue-addressable.
    core_susc = np.power(np.clip(core, 0, 1), args.core_susc_power)
    core_susc = core_susc / (np.nanmean(core_susc) + 1e-9)

    # repair susceptibility: core/peri regions with low-to-intermediate repair receive more modeled gain.
    repair_susc = (
        0.55 * np.power(np.clip(core, 0, 1), 0.8)
        + 0.30 * np.power(np.clip(peri, 0, 1), 0.8)
        + 0.15 * (1.0 - repair_norm)
    )
    repair_susc = np.clip(repair_susc, 0, None)
    repair_susc = repair_susc / (np.nanmean(repair_susc) + 1e-9)

    rows = []
    for _, er in effects.iterrows():
        pert = er["perturbation"]
        dcore_mean = safe_float(er.get("delta_core"), 0.0)
        drep_mean = safe_float(er.get("delta_repair"), 0.0)

        # Visualization-level distributed effect.
        delta_core_cell = dcore_mean * core_susc
        delta_repair_cell = drep_mean * repair_susc

        core_reduction = np.maximum(-delta_core_cell, 0.0)
        repair_gain = np.maximum(delta_repair_cell, 0.0)

        for i in range(len(df)):
            rows.append({
                "_adata_id_": df["_adata_id_"].iloc[i],
                "perturbation": pert,
                "latent_x": x[i],
                "latent_y": y[i],
                "baseline_core": core[i],
                "baseline_peri": peri[i],
                "baseline_remote": remote[i],
                "baseline_repair_norm": repair_norm[i],
                "delta_core": delta_core_cell[i],
                "core_reduction": core_reduction[i],
                "delta_repair": delta_repair_cell[i],
                "repair_gain": repair_gain[i],
            })

    resp = pd.DataFrame(rows)

    # Normalize priority across all perturbations.
    cr = resp["core_reduction"].to_numpy(float)
    rg = resp["repair_gain"].to_numpy(float)
    crn = cr / (np.nanpercentile(cr, 99) + 1e-9)
    rgn = rg / (np.nanpercentile(rg, 99) + 1e-9)
    priority = args.priority_core_weight * crn + (1.0 - args.priority_core_weight) * rgn
    resp["response_priority"] = np.clip(priority, 0, None)
    resp["response_priority"] = normalize01(resp["response_priority"].to_numpy(float), 1, 99)

    return resp


# -----------------------------
# Landscape smoothing and plotting
# -----------------------------

def smooth_grid(x, y, values, grid_n=240, sigma=2.0, mask_quantile=0.03):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x = x[ok]
    y = y[ok]
    values = values[ok]

    if len(x) < 20:
        raise ValueError("Too few finite points for smoothing.")

    pad_x = 0.04 * (np.nanmax(x) - np.nanmin(x) + 1e-9)
    pad_y = 0.04 * (np.nanmax(y) - np.nanmin(y) + 1e-9)

    xmin, xmax = np.nanmin(x) - pad_x, np.nanmax(x) + pad_x
    ymin, ymax = np.nanmin(y) - pad_y, np.nanmax(y) + pad_y

    xb = np.linspace(xmin, xmax, grid_n + 1)
    yb = np.linspace(ymin, ymax, grid_n + 1)

    H, _, _ = np.histogram2d(x, y, bins=[xb, yb])
    W, _, _ = np.histogram2d(x, y, bins=[xb, yb], weights=values)

    if gaussian_filter is None:
        dens = H
        val = W / (H + 1e-9)
    else:
        dens = gaussian_filter(H, sigma=sigma)
        val = gaussian_filter(W, sigma=sigma) / (dens + 1e-9)

    positive = dens[dens > 0]
    if positive.size > 0:
        floor = np.nanquantile(positive, mask_quantile)
    else:
        floor = 0

    mask = dens <= floor
    val = np.where(mask, np.nan, val)

    Xc = 0.5 * (xb[:-1] + xb[1:])
    Yc = 0.5 * (yb[:-1] + yb[1:])
    extent = [xmin, xmax, ymin, ymax]

    return Xc, Yc, val.T, dens.T, extent


def find_peaks_on_grid(Z, Xc, Yc, n_peaks=10, min_quantile=0.92):
    if maximum_filter is None:
        return []

    Z2 = np.array(Z, dtype=float)
    finite = np.isfinite(Z2)
    if finite.sum() == 0:
        return []

    thr = np.nanquantile(Z2[finite], min_quantile)
    local = maximum_filter(np.nan_to_num(Z2, nan=-np.inf), size=9)
    peak_mask = finite & (Z2 == local) & (Z2 >= thr)

    yy, xx = np.where(peak_mask)
    vals = Z2[yy, xx]
    if len(vals) == 0:
        return []

    order = np.argsort(vals)[::-1][:n_peaks]
    peaks = []
    for k in order:
        peaks.append((Xc[xx[k]], Yc[yy[k]], vals[k]))
    return peaks


def set_dark_axis(ax):
    ax.set_facecolor("#05080d")
    ax.tick_params(colors="#d7dce5", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#4b5563")
        sp.set_linewidth(0.6)
    ax.grid(True, color="#1f2937", lw=0.35, alpha=0.55)
    ax.set_xlabel("Latent coordinate 1", color="#d7dce5", fontsize=8)
    ax.set_ylabel("Latent coordinate 2", color="#d7dce5", fontsize=8)


def plot_landscape(
    ax,
    x,
    y,
    values,
    title,
    cmap,
    vmin=None,
    vmax=None,
    grid_n=240,
    sigma=2.0,
    mask_quantile=0.03,
    scatter_values=None,
    scatter_cmap=None,
    scatter_alpha=0.18,
    scatter_size=1.8,
    add_peaks=False,
    peak_n=10,
):
    set_dark_axis(ax)

    Xc, Yc, Z, D, extent = smooth_grid(
        x, y, values,
        grid_n=grid_n,
        sigma=sigma,
        mask_quantile=mask_quantile,
    )

    im = ax.imshow(
        Z,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="auto",
    )

    finite = np.isfinite(Z)
    if finite.sum() > 20:
        try:
            levels = np.linspace(np.nanpercentile(Z[finite], 55), np.nanpercentile(Z[finite], 97), 7)
            ax.contour(
                Xc, Yc, Z,
                levels=levels,
                colors="#b7c3d0",
                linewidths=0.35,
                alpha=0.55,
            )
        except Exception:
            pass

    if scatter_values is not None:
        ax.scatter(
            x, y,
            c=scatter_values,
            s=scatter_size,
            cmap=scatter_cmap if scatter_cmap is not None else cmap,
            alpha=scatter_alpha,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
    else:
        ax.scatter(
            x, y,
            s=scatter_size,
            c="#65c7f7",
            alpha=scatter_alpha,
            linewidths=0,
        )

    if add_peaks:
        peaks = find_peaks_on_grid(Z, Xc, Yc, n_peaks=peak_n, min_quantile=0.93)
        for px, py, _ in peaks:
            ax.scatter(
                [px], [py],
                marker="*",
                s=45,
                c="#ffe81a",
                edgecolors="#111827",
                linewidths=0.35,
                zorder=10,
            )

    ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=5)
    return im


def make_colormaps():
    core_cmap = plt.get_cmap("magma")

    core_reduction_cmap = LinearSegmentedColormap.from_list(
        "core_reduction_dark",
        ["#05080d", "#12324a", "#2878a5", "#9bd4e4", "#fff7d1"],
        N=256,
    )

    repair_gain_cmap = LinearSegmentedColormap.from_list(
        "repair_gain_dark",
        ["#05080d", "#062815", "#0b6b3a", "#61c96d", "#f4fbc1"],
        N=256,
    )

    priority_cmap = LinearSegmentedColormap.from_list(
        "priority_dark",
        ["#05080d", "#241150", "#4b1d85", "#0891b2", "#f8e71c"],
        N=256,
    )

    return core_cmap, core_reduction_cmap, repair_gain_cmap, priority_cmap


# -----------------------------
# Figures
# -----------------------------

def make_atlas(df, resp, effects, outdir, args):
    core_cmap, core_red_cmap, repair_cmap, priority_cmap = make_colormaps()

    x = df["latent_x"].to_numpy(float)
    y = df["latent_y"].to_numpy(float)
    baseline_core = df["baseline_core"].to_numpy(float)

    atlas_effects = [p for p in CANONICAL_ORDER if p in set(effects["perturbation"])]
    nrows = len(atlas_effects)
    ncols = 4

    if nrows == 0:
        raise ValueError("No canonical perturbation rows available for atlas.")

    vmax_core_red = np.nanpercentile(resp["core_reduction"], 99)
    vmax_repair = np.nanpercentile(resp["repair_gain"], 99)
    vmax_pri = 1.0

    vmax_core_red = max(vmax_core_red, 1e-4)
    vmax_repair = max(vmax_repair, 1e-4)

    fig_w = 14.5
    fig_h = 2.25 * nrows + 1.7
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#05080d", dpi=args.dpi)

    gs = GridSpec(
        nrows=nrows,
        ncols=5,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 0.045],
        wspace=0.12,
        hspace=0.22,
        left=0.09,
        right=0.90,
        top=0.91,
        bottom=0.10,
    )

    axes = []
    last_ims = {}

    for i, pert in enumerate(atlas_effects):
        sub = resp[resp["perturbation"] == pert].copy()
        sx = sub["latent_x"].to_numpy(float)
        sy = sub["latent_y"].to_numpy(float)

        vals = {
            "core": baseline_core,
            "core_reduction": sub["core_reduction"].to_numpy(float),
            "repair_gain": sub["repair_gain"].to_numpy(float),
            "priority": sub["response_priority"].to_numpy(float),
        }

        titles = [
            "Baseline core map" if i == 0 else "",
            "Core reduction" if i == 0 else "",
            "Repair gain" if i == 0 else "",
            "Response priority" if i == 0 else "",
        ]

        ax0 = fig.add_subplot(gs[i, 0])
        im0 = plot_landscape(
            ax0, x, y, vals["core"],
            titles[0],
            cmap=core_cmap, vmin=0, vmax=1,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=baseline_core,
            scatter_cmap=core_cmap,
            scatter_alpha=0.18,
            scatter_size=args.scatter_size,
        )

        ax1 = fig.add_subplot(gs[i, 1])
        im1 = plot_landscape(
            ax1, sx, sy, vals["core_reduction"],
            titles[1],
            cmap=core_red_cmap, vmin=0, vmax=vmax_core_red,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=vals["core_reduction"],
            scatter_cmap=core_red_cmap,
            scatter_alpha=0.20,
            scatter_size=args.scatter_size,
        )

        ax2 = fig.add_subplot(gs[i, 2])
        im2 = plot_landscape(
            ax2, sx, sy, vals["repair_gain"],
            titles[2],
            cmap=repair_cmap, vmin=0, vmax=vmax_repair,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=vals["repair_gain"],
            scatter_cmap=repair_cmap,
            scatter_alpha=0.20,
            scatter_size=args.scatter_size,
        )

        ax3 = fig.add_subplot(gs[i, 3])
        im3 = plot_landscape(
            ax3, sx, sy, vals["priority"],
            titles[3],
            cmap=priority_cmap, vmin=0, vmax=vmax_pri,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=vals["priority"],
            scatter_cmap=priority_cmap,
            scatter_alpha=0.22,
            scatter_size=args.scatter_size,
            add_peaks=True,
            peak_n=8,
        )

        axes.append([ax0, ax1, ax2, ax3])
        last_ims = {
            "core": im0,
            "core_reduction": im1,
            "repair_gain": im2,
            "priority": im3,
        }

        # Row label
        ax0.text(
            -0.18, 0.5, pert,
            transform=ax0.transAxes,
            color="white",
            fontsize=9,
            fontweight="bold",
            ha="right",
            va="center",
        )

        if i != nrows - 1:
            for ax in [ax0, ax1, ax2, ax3]:
                ax.set_xlabel("")

    fig.suptitle(
        "Virtual perturbation response landscapes",
        color="white",
        fontsize=18,
        fontweight="bold",
        y=0.972,
    )

    # Colorbars stacked on right.
    cax_top = fig.add_axes([0.925, 0.70, 0.015, 0.19])
    cb0 = fig.colorbar(last_ims["core"], cax=cax_top)
    cb0.set_label("Core probability", color="white", fontsize=8)
    cb0.ax.tick_params(colors="white", labelsize=7)
    cb0.outline.set_edgecolor("#d1d5db")

    cax_mid1 = fig.add_axes([0.925, 0.49, 0.015, 0.15])
    cb1 = fig.colorbar(last_ims["core_reduction"], cax=cax_mid1)
    cb1.set_label("Core reduction", color="white", fontsize=8)
    cb1.ax.tick_params(colors="white", labelsize=7)
    cb1.outline.set_edgecolor("#d1d5db")

    cax_mid2 = fig.add_axes([0.925, 0.30, 0.015, 0.15])
    cb2 = fig.colorbar(last_ims["repair_gain"], cax=cax_mid2)
    cb2.set_label("Repair gain", color="white", fontsize=8)
    cb2.ax.tick_params(colors="white", labelsize=7)
    cb2.outline.set_edgecolor("#d1d5db")

    cax_bot = fig.add_axes([0.925, 0.12, 0.015, 0.15])
    cb3 = fig.colorbar(last_ims["priority"], cax=cax_bot)
    cb3.set_label("Response priority", color="white", fontsize=8)
    cb3.ax.tick_params(colors="white", labelsize=7)
    cb3.outline.set_edgecolor("#d1d5db")

    note = (
        "Gaussian-kernel-smoothed latent response landscapes. "
        "Core reduction = max(-Δcore probability, 0); repair gain = max(Δrepair score, 0). "
        "Maps are visualization-level in silico perturbation proxies, not wet-lab KO/blockade results "
        "and not observed cell-state transitions."
    )
    fig.text(
        0.50, 0.035,
        note,
        ha="center",
        va="center",
        color="#d1d5db",
        fontsize=8,
        wrap=True,
    )

    out_png = os.path.join(outdir, "Fig_Step78C_VirtualPerturbationResponseLandscape_Atlas_final.png")
    out_pdf = os.path.join(outdir, "Fig_Step78C_VirtualPerturbationResponseLandscape_Atlas_final.pdf")
    out_svg = os.path.join(outdir, "Fig_Step78C_VirtualPerturbationResponseLandscape_Atlas_final.svg")

    fig.savefig(out_png, dpi=args.dpi, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, facecolor=fig.get_facecolor())
    fig.savefig(out_svg, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {"png": out_png, "pdf": out_pdf, "svg": out_svg}


def make_single_candidate_figures(df, resp, effects, outdir, args):
    core_cmap, core_red_cmap, repair_cmap, priority_cmap = make_colormaps()

    x = df["latent_x"].to_numpy(float)
    y = df["latent_y"].to_numpy(float)
    baseline_core = df["baseline_core"].to_numpy(float)

    vmax_core_red = max(np.nanpercentile(resp["core_reduction"], 99), 1e-4)
    vmax_repair = max(np.nanpercentile(resp["repair_gain"], 99), 1e-4)

    outputs = []

    atlas_effects = [p for p in CANONICAL_ORDER if p in set(effects["perturbation"])]

    if args.single_candidate.lower() == "all":
        selected = atlas_effects
    else:
        canon = canonicalize_perturbation_name(args.single_candidate)
        if canon is None and args.single_candidate in atlas_effects:
            canon = args.single_candidate
        if canon is None:
            # Choose best by mean priority.
            mean_pri = resp.groupby("perturbation")["response_priority"].mean().sort_values(ascending=False)
            selected = [mean_pri.index[0]]
        else:
            selected = [canon]

    for pert in selected:
        sub = resp[resp["perturbation"] == pert].copy()
        sx = sub["latent_x"].to_numpy(float)
        sy = sub["latent_y"].to_numpy(float)

        fig = plt.figure(figsize=(12.5, 9.2), facecolor="#05080d", dpi=args.dpi)
        gs = GridSpec(
            nrows=2,
            ncols=3,
            figure=fig,
            width_ratios=[1, 1, 0.045],
            wspace=0.12,
            hspace=0.22,
            left=0.07,
            right=0.90,
            top=0.88,
            bottom=0.12,
        )

        axA = fig.add_subplot(gs[0, 0])
        imA = plot_landscape(
            axA, x, y, baseline_core,
            "Baseline core-probability landscape",
            cmap=core_cmap, vmin=0, vmax=1,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=baseline_core,
            scatter_cmap=core_cmap,
            scatter_alpha=0.18,
            scatter_size=args.scatter_size,
        )

        axB = fig.add_subplot(gs[0, 1])
        imB = plot_landscape(
            axB, sx, sy, sub["core_reduction"].to_numpy(float),
            "Core-reduction landscape",
            cmap=core_red_cmap, vmin=0, vmax=vmax_core_red,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=sub["core_reduction"].to_numpy(float),
            scatter_cmap=core_red_cmap,
            scatter_alpha=0.22,
            scatter_size=args.scatter_size,
        )

        axC = fig.add_subplot(gs[1, 0])
        imC = plot_landscape(
            axC, sx, sy, sub["repair_gain"].to_numpy(float),
            "Repair-gain landscape",
            cmap=repair_cmap, vmin=0, vmax=vmax_repair,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=sub["repair_gain"].to_numpy(float),
            scatter_cmap=repair_cmap,
            scatter_alpha=0.22,
            scatter_size=args.scatter_size,
        )

        axD = fig.add_subplot(gs[1, 1])
        imD = plot_landscape(
            axD, sx, sy, sub["response_priority"].to_numpy(float),
            "Response-priority basin map",
            cmap=priority_cmap, vmin=0, vmax=1,
            grid_n=args.grid_n, sigma=args.sigma_grid,
            mask_quantile=args.mask_quantile,
            scatter_values=sub["response_priority"].to_numpy(float),
            scatter_cmap=priority_cmap,
            scatter_alpha=0.24,
            scatter_size=args.scatter_size,
            add_peaks=True,
            peak_n=10,
        )

        for label, ax in zip(["A", "B", "C", "D"], [axA, axB, axC, axD]):
            ax.text(
                -0.08, 1.08, label,
                transform=ax.transAxes,
                color="white",
                fontsize=15,
                fontweight="bold",
                ha="left",
                va="top",
            )

        cax = fig.add_subplot(gs[:, 2])
        cb = fig.colorbar(imD, cax=cax)
        cb.set_label("Response priority", color="white", fontsize=10)
        cb.ax.tick_params(colors="white", labelsize=8)
        cb.outline.set_edgecolor("#d1d5db")

        fig.suptitle(
            f"Virtual perturbation response landscape | {pert}",
            color="white",
            fontsize=20,
            fontweight="bold",
            y=0.965,
        )

        note = (
            "Gaussian-kernel-smoothed latent response landscape. "
            "Peaks indicate local high-response basins. "
            "This is a visualization-level counterfactual proxy, not observed cell-state transition."
        )
        fig.text(
            0.50, 0.045,
            note,
            ha="center",
            va="center",
            color="#d1d5db",
            fontsize=9,
            wrap=True,
        )

        stem = sanitize_filename(pert)
        out_png = os.path.join(outdir, f"Fig_Step78C_ResponseLandscape_{stem}_final.png")
        out_pdf = os.path.join(outdir, f"Fig_Step78C_ResponseLandscape_{stem}_final.pdf")
        out_svg = os.path.join(outdir, f"Fig_Step78C_ResponseLandscape_{stem}_final.svg")

        fig.savefig(out_png, dpi=args.dpi, facecolor=fig.get_facecolor())
        fig.savefig(out_pdf, facecolor=fig.get_facecolor())
        fig.savefig(out_svg, facecolor=fig.get_facecolor())
        plt.close(fig)

        outputs.append({"perturbation": pert, "png": out_png, "pdf": out_pdf, "svg": out_svg})

    return outputs


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--h5ad",
        default="/mnt/h/vir/ST/results/step5_nicheformer/spatial_all_with_nicheformer.h5ad",
    )
    ap.add_argument(
        "--state_table",
        default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv",
    )
    ap.add_argument(
        "--step72_dir",
        default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/expression_level_insilico_ko_72c_strict_label_surrogate",
    )
    ap.add_argument("--step72_summary", default="")
    ap.add_argument(
        "--outdir",
        default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/virtual_perturbation_response_landscape_78c_final",
    )

    ap.add_argument("--force_obsm_key", default="X_nicheformer")
    ap.add_argument("--state_label_col", default="state_group")

    ap.add_argument("--grid_n", type=int, default=260)
    ap.add_argument("--sigma_grid", type=float, default=2.2)
    ap.add_argument("--mask_quantile", type=float, default=0.035)
    ap.add_argument("--scatter_size", type=float, default=1.4)
    ap.add_argument("--dpi", type=int, default=300)

    ap.add_argument("--core_susc_power", type=float, default=1.15)
    ap.add_argument("--priority_core_weight", type=float, default=0.58)

    ap.add_argument(
        "--single_candidate",
        default="all",
        help="all, or one candidate name such as ferroptosis_down / spp1_cd44 / ccl2",
    )
    ap.add_argument("--disable_builtin_fallback", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    mkdir(args.outdir)

    print("=" * 100)
    print("Step78C | Final virtual perturbation response landscape")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"step72_dir={args.step72_dir}")
    print(f"step72_summary={args.step72_summary}")
    print(f"outdir={args.outdir}")

    adata = load_h5ad(args.h5ad)
    latent, latent_audit = infer_latent_coordinates(
        adata,
        force_obsm_key=args.force_obsm_key,
    )

    state_df, state_audit = load_state_table(args.state_table, adata.obs_names)
    state_df["latent_x"] = latent[:, 0]
    state_df["latent_y"] = latent[:, 1]

    prob_cols, label_used, state_df = infer_prob_columns(state_df, args.state_label_col)

    state_df["baseline_core"] = np.clip(state_df[prob_cols["core"]].astype(float), 0, 1)
    state_df["baseline_peri"] = np.clip(state_df[prob_cols["peri"]].astype(float), 0, 1)
    state_df["baseline_remote"] = np.clip(state_df[prob_cols["remote"]].astype(float), 0, 1)

    effects, effect_audit = load_step72_effects(
        step72_dir=args.step72_dir,
        step72_summary=args.step72_summary if args.step72_summary else None,
        allow_builtin=not args.disable_builtin_fallback,
    )

    print("\n---- canonical perturbation effects after dedup ----")
    print(effects[["perturbation", "delta_core", "delta_repair", "source"]].to_string(index=False))

    resp = make_response_per_cell(state_df, prob_cols, effects, args)

    out_resp_csv = os.path.join(args.outdir, "step78c_per_cell_virtual_response_landscape.csv")
    resp.to_csv(out_resp_csv, index=False)

    out_effect_csv = os.path.join(args.outdir, "step78c_canonical_perturbation_effects_deduplicated.csv")
    effects.to_csv(out_effect_csv, index=False)

    atlas_paths = make_atlas(state_df, resp, effects, args.outdir, args)
    single_paths = make_single_candidate_figures(state_df, resp, effects, args.outdir, args)

    report = {
        "status": "ok",
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "step72_dir": args.step72_dir,
        "step72_summary": args.step72_summary,
        "outdir": args.outdir,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "latent_audit": latent_audit,
        "state_audit": state_audit,
        "prob_cols": prob_cols,
        "state_label_used_for_fallback": label_used,
        "step72_effect_audit": effect_audit,
        "outputs": {
            "atlas": atlas_paths,
            "single_candidate_figures": single_paths,
            "per_cell_response_csv": out_resp_csv,
            "effect_csv": out_effect_csv,
        },
        "interpretation_note": (
            "Step78C produces visualization-level virtual perturbation response landscapes. "
            "Core reduction is max(-delta_core_probability, 0). Repair gain is max(delta_repair_score, 0). "
            "Response priority combines core reduction and repair gain in latent space. "
            "These maps are not wet-lab KO/blockade results, not physical energy landscapes, "
            "and not observed cell-state transitions."
        ),
    }

    out_report = os.path.join(args.outdir, "step78c_report.json")
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDONE Step78C")
    print(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
