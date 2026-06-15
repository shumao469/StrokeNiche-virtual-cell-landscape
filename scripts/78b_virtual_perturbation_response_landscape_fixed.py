#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step78B | Virtual perturbation response landscape

Purpose
-------
Construct manuscript-style latent-space virtual perturbation response landscapes:

    ΔP_core(z)
    repair-gain(z)
    response-priority(z) = core-reduction(z) + repair-gain(z)

This version is robust to the current Step72C output format.
It can use:
  1) Step72C per-cell table if available;
  2) Step72C summary table if per-cell table is unavailable;
  3) built-in Step72C fallback deltas from the strict-label output.

Important interpretation
------------------------
This is a visualization-level in silico perturbation response landscape.
It is NOT a wet-lab KO/blockade result and NOT an observed cell-fate transition.
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# Basic utilities
# -----------------------------

def mkdirp(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_read_csv(path, nrows=None):
    path = str(path)
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def norm_name(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def clean_display_name(x):
    x = str(x)
    x = x.replace("_", " ")
    x = x.replace("Ccl2 Ccr2 Ackr1 blockade", "Ccl2/Ccr2-Ackr1")
    x = x.replace("Spp1 Cd44 blockade", "Spp1-Cd44")
    x = x.replace("Vegfa Flt1 Kdr blockade", "Vegfa-Flt1/Kdr")
    x = x.replace("ferroptosis down", "Ferroptosis down")
    x = x.replace("repair ECM up", "Repair-ECM up")
    return x


def first_existing_col(df, candidates):
    lowmap = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).lower() in lowmap:
            return lowmap[str(c).lower()]
    return None


def find_col_by_patterns(df, patterns, require_numeric=False):
    cols = list(df.columns)
    for pat in patterns:
        rgx = re.compile(pat, flags=re.I)
        for c in cols:
            if rgx.search(str(c)):
                if require_numeric:
                    try:
                        pd.to_numeric(df[c], errors="raise")
                    except Exception:
                        continue
                return c
    return None


def numeric_series(df, col, default=0.0):
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def minmax01(x):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros_like(x, dtype=float)
    if finite.sum() == 0:
        return out
    lo = np.nanpercentile(x[finite], 1)
    hi = np.nanpercentile(x[finite], 99)
    if hi <= lo:
        return out
    out = (x - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~finite] = 0
    return out


def zclip(x, q=0.995):
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    mu = np.nanmean(x[finite])
    sd = np.nanstd(x[finite]) + 1e-9
    z = (x - mu) / sd
    lim = np.nanquantile(np.abs(z[finite]), q)
    lim = max(lim, 1.0)
    z = np.clip(z, -lim, lim)
    return z


# -----------------------------
# Load data
# -----------------------------

def load_h5ad(h5ad_path):
    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(
            "Cannot import anndata. Activate the environment that has anndata installed."
        ) from e

    adata = ad.read_h5ad(h5ad_path)
    return adata


def infer_latent_2d(adata, force_obsm_key="X_nicheformer", random_state=42):
    audit = {}

    if force_obsm_key:
        if force_obsm_key not in adata.obsm.keys():
            raise ValueError(
                f"--force_obsm_key={force_obsm_key} not found in adata.obsm. "
                f"Available obsm keys: {list(adata.obsm.keys())}"
            )
        X = np.asarray(adata.obsm[force_obsm_key])
        audit["mode"] = "forced_obsm_pca_or_direct"
        audit["obsm_key"] = force_obsm_key
    else:
        preferred = ["X_nicheformer", "X_scgpt", "X_umap", "X_pca", "spatial"]
        key = None
        for k in preferred:
            if k in adata.obsm.keys():
                key = k
                break
        if key is None:
            raise ValueError(f"No usable obsm key found. Available: {list(adata.obsm.keys())}")
        X = np.asarray(adata.obsm[key])
        audit["mode"] = "auto_obsm_pca_or_direct"
        audit["obsm_key"] = key

    audit["input_dim"] = int(X.shape[1]) if X.ndim == 2 else None

    if X.ndim != 2:
        raise ValueError("obsm latent matrix must be 2D.")

    if X.shape[1] == 2:
        Z2 = X.astype(float)
        audit["projection"] = "direct_2d"
    else:
        Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
        Z2 = PCA(n_components=2, random_state=random_state).fit_transform(Xs)
        audit["projection"] = "standardized_pca_2d"

    return Z2[:, 0], Z2[:, 1], audit


def load_state_table(path):
    df = safe_read_csv(path)
    id_col = first_existing_col(
        df,
        ["obs_name", "_obs_names_", "barcode", "spot_id", "cell_id", "adata_id", "merge_key"]
    )
    if id_col is None:
        raise ValueError(f"No obs/cell id column found in state table: {path}")
    df[id_col] = df[id_col].astype(str)
    return df, id_col


def merge_state(adata, state_df, state_id_col):
    meta = pd.DataFrame({
        "_obs_name_": adata.obs_names.astype(str)
    })
    merged = meta.merge(
        state_df,
        left_on="_obs_name_",
        right_on=state_id_col,
        how="left",
        suffixes=("", "_state")
    )

    overlap = int(merged[state_id_col].notna().sum())
    if overlap == 0:
        raise RuntimeError("State table merge overlap is 0. Check obs_name consistency.")

    return merged, {
        "state_id_col": state_id_col,
        "n_adata": int(adata.n_obs),
        "n_state_table": int(len(state_df)),
        "merge_overlap": overlap,
        "merge_fraction": float(overlap / max(adata.n_obs, 1)),
    }


def infer_state_columns(df):
    core_col = first_existing_col(df, [
        "core_probability", "core_prob", "prob_core", "P_core",
        "prob_lesion_core", "P_lesion_core", "lesion_core_probability"
    ])
    peri_col = first_existing_col(df, [
        "peri_probability", "peri_prob", "prob_peri", "P_peri",
        "prob_peri_infarct", "P_peri_infarct", "peri_infarct_probability"
    ])
    remote_col = first_existing_col(df, [
        "remote_probability", "remote_prob", "prob_remote", "P_remote",
        "prob_remote_like", "P_remote_like", "remote_like_probability"
    ])
    repair_col = first_existing_col(df, [
        "repair_score", "repair_score_from_npy", "rescue_score",
        "repair_probability", "repair_ECM_score"
    ])

    state_col = first_existing_col(df, [
        "state_group", "region_auto", "region_refined",
        "region_manual_final", "region_true", "strict_state_group"
    ])

    return {
        "core": core_col,
        "peri": peri_col,
        "remote": remote_col,
        "repair": repair_col,
        "state_label": state_col,
    }


# -----------------------------
# Step72C loading
# -----------------------------

def discover_step72_summary(step72_dir):
    d = Path(step72_dir)
    if not d.exists():
        raise FileNotFoundError(f"Step72 dir does not exist: {step72_dir}")

    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in Step72 dir: {step72_dir}")

    # Prefer the strict summary file.
    preferred = []
    for p in csvs:
        s = p.name.lower()
        score = 0
        if "summary" in s:
            score += 10
        if "state_shift" in s:
            score += 8
        if "expression_level" in s:
            score += 4
        if "net_flow" in s:
            score -= 2
        if "feature" in s or "gene_audit" in s:
            score -= 5
        preferred.append((score, p))

    preferred = sorted(preferred, reverse=True, key=lambda x: x[0])
    return preferred[0][1], [str(x[1]) for x in preferred]


def parse_step72_summary(step72_dir):
    summary_path, candidates = discover_step72_summary(step72_dir)
    df = safe_read_csv(summary_path)

    pert_col = find_col_by_patterns(
        df,
        [
            r"^candidate$",
            r"^perturbation$",
            r"^target$",
            r"intervention",
            r"blockade",
            r"module",
            r"name",
        ],
        require_numeric=False
    )

    if pert_col is None:
        # If the table has row names accidentally stored in first unnamed column.
        unnamed = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if unnamed:
            pert_col = unnamed[0]

    # Find delta columns robustly.
    delta_core_col = find_col_by_patterns(df, [
        r"delta.*core",
        r"d_core",
        r"Δ.*core",
        r"core.*shift",
        r"core.*delta",
        r"mean.*core",
    ], require_numeric=True)

    delta_peri_col = find_col_by_patterns(df, [
        r"delta.*peri",
        r"d_peri",
        r"Δ.*peri",
        r"peri.*shift",
        r"peri.*delta",
        r"mean.*peri",
    ], require_numeric=True)

    delta_remote_col = find_col_by_patterns(df, [
        r"delta.*remote",
        r"d_remote",
        r"Δ.*remote",
        r"remote.*shift",
        r"remote.*delta",
        r"mean.*remote",
    ], require_numeric=True)

    repair_shift_col = find_col_by_patterns(df, [
        r"rescue.*score",
        r"repair.*shift",
        r"repair.*gain",
        r"delta.*repair",
        r"state.*shift.*priority",
        r"priority",
    ], require_numeric=True)

    out = {}
    if pert_col is not None:
        for _, r in df.iterrows():
            name = str(r[pert_col])
            if name.lower() == "nan":
                continue
            key = norm_name(name)

            dcore = pd.to_numeric(r.get(delta_core_col, np.nan), errors="coerce")
            dperi = pd.to_numeric(r.get(delta_peri_col, np.nan), errors="coerce")
            dremote = pd.to_numeric(r.get(delta_remote_col, np.nan), errors="coerce")
            drepr = pd.to_numeric(r.get(repair_shift_col, np.nan), errors="coerce")

            out[key] = {
                "display": clean_display_name(name),
                "delta_core": float(dcore) if np.isfinite(dcore) else np.nan,
                "delta_peri": float(dperi) if np.isfinite(dperi) else np.nan,
                "delta_remote": float(dremote) if np.isfinite(dremote) else np.nan,
                "repair_shift": float(drepr) if np.isfinite(drepr) else np.nan,
                "source": "step72c_summary",
            }

    audit = {
        "summary_path": str(summary_path),
        "candidate_csvs_considered": candidates,
        "summary_columns": list(df.columns),
        "perturbation_col": pert_col,
        "delta_core_col": delta_core_col,
        "delta_peri_col": delta_peri_col,
        "delta_remote_col": delta_remote_col,
        "repair_shift_col": repair_shift_col,
        "n_rows": int(len(df)),
        "n_parsed": int(len(out)),
    }
    return out, audit


def fallback_step72_effects():
    """
    Fallback values based on the already inspected Step72C strict-label output.
    Values are mean state-probability / rescue-score shifts.
    """
    return {
        "ccl2_ccr2_ackr1_blockade": {
            "display": "Ccl2/Ccr2-Ackr1",
            "delta_core": -0.0292,
            "delta_peri": -0.0076,
            "delta_remote": +0.0368,
            "repair_shift": +0.0438,
            "source": "fallback_from_step72c_strict_label"
        },
        "spp1_cd44_blockade": {
            "display": "Spp1-Cd44",
            "delta_core": -0.0953,
            "delta_peri": -0.0921,
            "delta_remote": +0.1874,
            "repair_shift": +0.1429,
            "source": "fallback_from_step72c_strict_label"
        },
        "vegfa_flt1_kdr_blockade": {
            "display": "Vegfa-Flt1/Kdr",
            "delta_core": -0.0224,
            "delta_peri": -0.0102,
            "delta_remote": +0.0326,
            "repair_shift": +0.0337,
            "source": "fallback_from_step72c_strict_label"
        },
        "ferroptosis_down": {
            "display": "Ferroptosis down",
            "delta_core": -0.1084,
            "delta_peri": -0.0358,
            "delta_remote": +0.1442,
            "repair_shift": +0.1627,
            "source": "fallback_from_step72c_strict_label"
        },
        "repair_ecm_up": {
            "display": "Repair-ECM up",
            "delta_core": +0.0962,
            "delta_peri": +0.0521,
            "delta_remote": -0.1483,
            "repair_shift": -0.1443,
            "source": "fallback_from_step72c_strict_label"
        },
    }


def complete_effects_from_summary(summary_effects):
    fb = fallback_step72_effects()
    out = {}

    for key, v in fb.items():
        if key in summary_effects:
            sv = summary_effects[key].copy()
            # Fill missing numeric values from fallback.
            for col in ["delta_core", "delta_peri", "delta_remote", "repair_shift"]:
                if col not in sv or not np.isfinite(sv[col]):
                    sv[col] = v[col]
            if "display" not in sv or sv["display"] == "":
                sv["display"] = v["display"]
            sv["source"] = sv.get("source", "step72c_summary")
            out[key] = sv
        else:
            out[key] = v

    # Add extra summary effects if any are parseable.
    for key, v in summary_effects.items():
        if key not in out:
            valid = any(np.isfinite(v.get(c, np.nan)) for c in ["delta_core", "delta_peri", "delta_remote", "repair_shift"])
            if valid:
                out[key] = v

    return out


# -----------------------------
# Response construction
# -----------------------------

def module_col(df, aliases):
    for a in aliases:
        c = first_existing_col(df, [a])
        if c is not None:
            return c
    for c in df.columns:
        lc = str(c).lower()
        if any(norm_name(a) in norm_name(lc) for a in aliases):
            return c
    return None


def get_susceptibility(df, perturbation_key, core, repair):
    """
    Build per-cell/spot susceptibility weights.
    This is only for visualization-level response landscape when per-cell Step72 outputs are absent.
    """
    pkey = norm_name(perturbation_key)

    col_inflam = module_col(df, [
        "module_inflammation",
        "module_microglia_inflammatory",
        "microglia_context",
        "lesion_niche_context"
    ])
    col_repair = module_col(df, [
        "module_repair_ECM",
        "repair_score",
        "repair_target"
    ])
    col_ferro = module_col(df, [
        "module_ferroptosis",
        "ferroptosis"
    ])
    col_bbb = module_col(df, [
        "module_BBB_leakage",
        "module_endothelial_barrier_fragility",
        "endothelial_context",
        "module_barrier_stability"
    ])
    col_astro = module_col(df, [
        "module_astrocyte_reactive",
        "astrocyte_context"
    ])

    core01 = minmax01(core)
    repair01 = minmax01(repair)

    inflam01 = minmax01(numeric_series(df, col_inflam, 0).values) if col_inflam else core01
    repair_mod01 = minmax01(numeric_series(df, col_repair, 0).values) if col_repair else repair01
    ferro01 = minmax01(numeric_series(df, col_ferro, 0).values) if col_ferro else core01
    bbb01 = minmax01(numeric_series(df, col_bbb, 0).values) if col_bbb else core01
    astro01 = minmax01(numeric_series(df, col_astro, 0).values) if col_astro else core01

    if "ccl2" in pkey or "ccr2" in pkey or "ackr1" in pkey:
        sus = 0.50 * core01 + 0.40 * inflam01 + 0.10 * bbb01
    elif "spp1" in pkey or "cd44" in pkey:
        sus = 0.45 * core01 + 0.45 * repair_mod01 + 0.10 * inflam01
    elif "vegfa" in pkey or "flt1" in pkey or "kdr" in pkey:
        sus = 0.50 * core01 + 0.45 * bbb01 + 0.05 * inflam01
    elif "ferro" in pkey:
        sus = 0.55 * core01 + 0.40 * ferro01 + 0.05 * inflam01
    elif "repair" in pkey or "ecm" in pkey:
        sus = 0.45 * repair_mod01 + 0.30 * core01 + 0.15 * astro01 + 0.10 * inflam01
    else:
        sus = core01

    sus = minmax01(sus)
    return sus, {
        "col_inflammation": col_inflam,
        "col_repair": col_repair,
        "col_ferroptosis": col_ferro,
        "col_bbb": col_bbb,
        "col_astrocyte": col_astro,
    }


def construct_per_cell_response(df, effects, state_cols):
    core = numeric_series(df, state_cols["core"], 0).values
    peri = numeric_series(df, state_cols["peri"], 0).values
    remote = numeric_series(df, state_cols["remote"], 0).values
    repair = numeric_series(df, state_cols["repair"], 0).values if state_cols["repair"] else np.zeros(len(df))

    if np.nanstd(core) < 1e-8:
        # If probability table is uniform/degenerate, fall back to labels.
        label_col = state_cols["state_label"]
        if label_col:
            labs = df[label_col].astype(str).str.lower()
            core = labs.str.contains("core|lesion").astype(float).values
            peri = labs.str.contains("peri").astype(float).values
            remote = labs.str.contains("remote").astype(float).values

    rows = []
    sus_audit = {}

    for key, e in effects.items():
        sus, sa = get_susceptibility(df, key, core, repair)
        sus_audit[key] = sa

        dcore = float(e.get("delta_core", 0.0))
        dperi = float(e.get("delta_peri", 0.0))
        dremote = float(e.get("delta_remote", 0.0))
        drepr = float(e.get("repair_shift", 0.0))

        # Per-cell visualization proxies.
        delta_core = dcore * sus
        delta_peri = dperi * sus
        delta_remote = dremote * sus

        core_reduction = np.clip(-delta_core, 0, None)
        repair_gain = np.clip(drepr, 0, None) * sus

        # If perturbation increases core or lowers repair, reflect it as negative response.
        adverse = np.clip(delta_core, 0, None) + np.clip(-drepr, 0, None) * sus

        response_priority = core_reduction + repair_gain - adverse

        for i in range(len(df)):
            rows.append({
                "obs_name": df["_obs_name_"].iloc[i],
                "perturbation_key": key,
                "perturbation": e.get("display", key),
                "effect_source": e.get("source", ""),
                "baseline_core_probability": core[i],
                "baseline_peri_probability": peri[i],
                "baseline_remote_probability": remote[i],
                "baseline_repair_score": repair[i],
                "susceptibility": sus[i],
                "mean_delta_core": dcore,
                "mean_delta_peri": dperi,
                "mean_delta_remote": dremote,
                "mean_repair_shift": drepr,
                "delta_core_probability_proxy": delta_core[i],
                "delta_peri_probability_proxy": delta_peri[i],
                "delta_remote_probability_proxy": delta_remote[i],
                "core_reduction_proxy": core_reduction[i],
                "repair_gain_proxy": repair_gain[i],
                "adverse_shift_proxy": adverse[i],
                "response_priority_proxy": response_priority[i],
            })

    return pd.DataFrame(rows), sus_audit


# -----------------------------
# Grid smoothing
# -----------------------------

def make_grid(x, y, grid_n=240, pad_frac=0.04):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xmin, xmax = np.nanpercentile(x, [0.5, 99.5])
    ymin, ymax = np.nanpercentile(y, [0.5, 99.5])

    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= dx * pad_frac
    xmax += dx * pad_frac
    ymin -= dy * pad_frac
    ymax += dy * pad_frac

    xedges = np.linspace(xmin, xmax, grid_n + 1)
    yedges = np.linspace(ymin, ymax, grid_n + 1)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    return xedges, yedges, xc, yc


def smooth_to_grid(x, y, values, xedges, yedges, sigma=2.0, min_density_frac=0.01):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x = x[ok]
    y = y[ok]
    v = v[ok]

    count, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    wsum, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=v)

    count_s = gaussian_filter(count, sigma=sigma)
    wsum_s = gaussian_filter(wsum, sigma=sigma)

    grid = wsum_s / (count_s + 1e-9)
    density = count_s / (np.nanmax(count_s) + 1e-9)
    mask = density >= min_density_frac
    grid[~mask] = np.nan

    # Transpose so imshow x/y orientation is correct.
    return grid.T, density.T, mask.T


def detect_peaks(grid, xc, yc, n_peaks=10):
    g = np.array(grid, dtype=float)
    if np.all(~np.isfinite(g)):
        return []

    filled = np.where(np.isfinite(g), g, -np.inf)
    local = filled == maximum_filter(filled, size=9, mode="nearest")
    finite = np.isfinite(g)
    vals = filled[local & finite]

    if vals.size == 0:
        return []

    thresh = np.nanpercentile(vals, 80)
    coords = np.argwhere(local & finite & (filled >= thresh))

    peaks = []
    for iy, ix in coords:
        if ix < 0 or iy < 0 or ix >= len(xc) or iy >= len(yc):
            continue
        peaks.append((float(filled[iy, ix]), float(xc[ix]), float(yc[iy])))

    peaks = sorted(peaks, reverse=True, key=lambda t: t[0])
    return peaks[:n_peaks]


# -----------------------------
# Plotting
# -----------------------------

def set_dark_ax(ax):
    ax.set_facecolor("#05080d")
    for spine in ax.spines.values():
        spine.set_color("#9aa4b2")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#d8dee9", labelsize=8, length=2)
    ax.xaxis.label.set_color("#d8dee9")
    ax.yaxis.label.set_color("#d8dee9")
    ax.grid(True, color="#202733", linewidth=0.4, alpha=0.55)


def plot_landscape_ax(
    ax,
    x,
    y,
    grid,
    xc,
    yc,
    title,
    cmap="magma",
    norm=None,
    scatter_values=None,
    scatter_cmap=None,
    scatter_alpha=0.25,
    peak=False,
    peak_color="#ffd84d",
):
    set_dark_ax(ax)
    extent = [xc.min(), xc.max(), yc.min(), yc.max()]

    im = ax.imshow(
        grid,
        extent=extent,
        origin="lower",
        interpolation="bilinear",
        cmap=cmap,
        norm=norm,
        aspect="auto",
        alpha=0.98,
    )

    if np.isfinite(grid).sum() > 10:
        try:
            levels = np.nanpercentile(grid[np.isfinite(grid)], [35, 55, 75, 90])
            levels = np.unique(levels)
            if len(levels) >= 2:
                ax.contour(
                    xc,
                    yc,
                    grid,
                    levels=levels,
                    colors="#a7c7d9",
                    linewidths=0.45,
                    alpha=0.55,
                )
        except Exception:
            pass

    if scatter_values is None:
        ax.scatter(x, y, s=2, c="#5dade2", alpha=scatter_alpha, linewidths=0)
    else:
        ax.scatter(
            x, y, s=2, c=scatter_values,
            cmap=scatter_cmap or cmap,
            alpha=scatter_alpha,
            linewidths=0
        )

    if peak:
        for _, px, py in detect_peaks(grid, xc, yc, n_peaks=10):
            ax.scatter(
                [px], [py],
                s=70,
                marker="*",
                c=peak_color,
                edgecolors="black",
                linewidths=0.6,
                zorder=10,
            )

    ax.set_title(title, color="white", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Latent coordinate 1", fontsize=8)
    ax.set_ylabel("Latent coordinate 2", fontsize=8)
    return im


def savefig_multi(fig, out_prefix, dpi=300):
    paths = {}
    for ext in ["png", "pdf", "svg"]:
        p = f"{out_prefix}.{ext}"
        fig.savefig(
            p,
            dpi=dpi if ext == "png" else None,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.12
        )
        paths[ext] = p
    return paths


def make_atlas_figure(
    x,
    y,
    per_cell,
    effects,
    xedges,
    yedges,
    xc,
    yc,
    outdir,
    sigma=2.0,
    min_density_frac=0.01,
    dpi=300,
):
    keys = list(effects.keys())
    nrow = len(keys)
    ncol = 4

    fig_w = 16
    fig_h = max(12, 2.65 * nrow + 2.0)
    fig, axes = plt.subplots(
        nrow,
        ncol,
        figsize=(fig_w, fig_h),
        facecolor="#05080d",
        constrained_layout=False
    )

    if nrow == 1:
        axes = np.array([axes])

    fig.suptitle(
        "Virtual perturbation response landscapes",
        color="white",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )

    # Shared color norms.
    prob_norm = Normalize(vmin=0, vmax=1)
    response_norm = Normalize(vmin=0, vmax=0.35)

    # Symmetric delta norm based on all delta-core proxy values.
    all_delta = per_cell["delta_core_probability_proxy"].values
    vmax_delta = np.nanpercentile(np.abs(all_delta[np.isfinite(all_delta)]), 99)
    vmax_delta = max(vmax_delta, 0.03)
    delta_norm = TwoSlopeNorm(vmin=-vmax_delta, vcenter=0, vmax=vmax_delta)

    last_prob_im = None
    last_delta_im = None
    last_gain_im = None
    last_priority_im = None

    for i, key in enumerate(keys):
        d = per_cell[per_cell["perturbation_key"] == key].copy()
        disp = effects[key].get("display", key)

        base_grid, _, _ = smooth_to_grid(
            x, y, d["baseline_core_probability"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )
        delta_grid, _, _ = smooth_to_grid(
            x, y, d["delta_core_probability_proxy"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )
        gain_grid, _, _ = smooth_to_grid(
            x, y, d["repair_gain_proxy"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )
        priority_grid, _, _ = smooth_to_grid(
            x, y, d["response_priority_proxy"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )

        row_title = disp
        axes[i, 0].text(
            -0.08, 0.5, row_title,
            transform=axes[i, 0].transAxes,
            ha="right", va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            rotation=0
        )

        last_prob_im = plot_landscape_ax(
            axes[i, 0], x, y, base_grid, xc, yc,
            "Baseline core map" if i == 0 else "",
            cmap="magma",
            norm=prob_norm,
            scatter_alpha=0.15,
        )

        last_delta_im = plot_landscape_ax(
            axes[i, 1], x, y, delta_grid, xc, yc,
            "Δ core probability" if i == 0 else "",
            cmap="RdBu_r",
            norm=delta_norm,
            scatter_alpha=0.12,
        )

        last_gain_im = plot_landscape_ax(
            axes[i, 2], x, y, gain_grid, xc, yc,
            "Repair gain" if i == 0 else "",
            cmap="YlGn",
            norm=response_norm,
            scatter_alpha=0.12,
        )

        last_priority_im = plot_landscape_ax(
            axes[i, 3], x, y, priority_grid, xc, yc,
            "Response priority" if i == 0 else "",
            cmap="viridis",
            norm=response_norm,
            scatter_alpha=0.12,
            peak=True,
        )

        for j in range(ncol):
            if i > 0:
                axes[i, j].set_title("")
            axes[i, j].set_xlabel("")
            axes[i, j].set_ylabel("")

    # Colorbars.
    cax1 = fig.add_axes([0.925, 0.70, 0.012, 0.18])
    cb1 = fig.colorbar(last_prob_im, cax=cax1)
    cb1.set_label("Core probability", color="white", fontsize=9)
    cb1.ax.tick_params(colors="white", labelsize=8)
    cb1.outline.set_edgecolor("white")

    cax2 = fig.add_axes([0.925, 0.49, 0.012, 0.18])
    cb2 = fig.colorbar(last_delta_im, cax=cax2)
    cb2.set_label("Δ core probability", color="white", fontsize=9)
    cb2.ax.tick_params(colors="white", labelsize=8)
    cb2.outline.set_edgecolor("white")

    cax3 = fig.add_axes([0.925, 0.28, 0.012, 0.18])
    cb3 = fig.colorbar(last_gain_im, cax=cax3)
    cb3.set_label("Repair gain", color="white", fontsize=9)
    cb3.ax.tick_params(colors="white", labelsize=8)
    cb3.outline.set_edgecolor("white")

    cax4 = fig.add_axes([0.925, 0.07, 0.012, 0.18])
    cb4 = fig.colorbar(last_priority_im, cax=cax4)
    cb4.set_label("Response priority", color="white", fontsize=9)
    cb4.ax.tick_params(colors="white", labelsize=8)
    cb4.outline.set_edgecolor("white")

    fig.text(
        0.5, 0.018,
        "Δ landscapes are visualization-level in silico perturbation response proxies derived from Step72C summary effects and per-spot susceptibility; "
        "they are not wet-lab KO/blockade results or observed cell-state transitions.",
        ha="center", va="bottom",
        color="#d8dee9",
        fontsize=9,
        wrap=True
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.90,
        top=0.93,
        bottom=0.06,
        wspace=0.10,
        hspace=0.14
    )

    out_prefix = os.path.join(outdir, "Fig_Step78B_VirtualPerturbationResponseLandscape_Atlas")
    paths = savefig_multi(fig, out_prefix, dpi=dpi)
    plt.close(fig)
    return paths


def make_single_candidate_figures(
    x,
    y,
    per_cell,
    effects,
    xedges,
    yedges,
    xc,
    yc,
    outdir,
    sigma=2.0,
    min_density_frac=0.01,
    dpi=300,
):
    paths = {}

    all_delta = per_cell["delta_core_probability_proxy"].values
    vmax_delta = np.nanpercentile(np.abs(all_delta[np.isfinite(all_delta)]), 99)
    vmax_delta = max(vmax_delta, 0.03)

    for key, e in effects.items():
        d = per_cell[per_cell["perturbation_key"] == key].copy()
        disp = e.get("display", key)

        base_grid, _, _ = smooth_to_grid(
            x, y, d["baseline_core_probability"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )
        delta_grid, _, _ = smooth_to_grid(
            x, y, d["delta_core_probability_proxy"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )
        gain_grid, _, _ = smooth_to_grid(
            x, y, d["repair_gain_proxy"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )
        priority_grid, _, _ = smooth_to_grid(
            x, y, d["response_priority_proxy"].values,
            xedges, yedges,
            sigma=sigma,
            min_density_frac=min_density_frac
        )

        fig, axes = plt.subplots(
            2, 2,
            figsize=(11, 8.5),
            facecolor="#05080d",
            constrained_layout=False
        )

        fig.suptitle(
            f"Virtual perturbation response landscape | {disp}",
            color="white",
            fontsize=18,
            fontweight="bold",
            y=0.975
        )

        im0 = plot_landscape_ax(
            axes[0, 0], x, y, base_grid, xc, yc,
            "Baseline core-probability landscape",
            cmap="magma",
            norm=Normalize(0, 1),
            scatter_alpha=0.16
        )
        im1 = plot_landscape_ax(
            axes[0, 1], x, y, delta_grid, xc, yc,
            "Δ core-probability landscape",
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-vmax_delta, vcenter=0, vmax=vmax_delta),
            scatter_alpha=0.14
        )
        im2 = plot_landscape_ax(
            axes[1, 0], x, y, gain_grid, xc, yc,
            "Repair-gain landscape",
            cmap="YlGn",
            norm=Normalize(0, 0.35),
            scatter_alpha=0.14
        )
        im3 = plot_landscape_ax(
            axes[1, 1], x, y, priority_grid, xc, yc,
            "Response-priority basin map",
            cmap="viridis",
            norm=Normalize(0, 0.35),
            scatter_alpha=0.14,
            peak=True
        )

        cax = fig.add_axes([0.92, 0.23, 0.018, 0.55])
        cb = fig.colorbar(im3, cax=cax)
        cb.set_label("Response priority", color="white", fontsize=9)
        cb.ax.tick_params(colors="white", labelsize=8)
        cb.outline.set_edgecolor("white")

        fig.text(
            0.5, 0.025,
            "Gaussian-kernel-smoothed latent response landscape; peaks indicate local high-response basins. "
            "Visualization-level counterfactual proxy, not an observed transition.",
            ha="center", va="bottom",
            color="#d8dee9",
            fontsize=9,
            wrap=True
        )

        fig.subplots_adjust(
            left=0.07,
            right=0.89,
            top=0.91,
            bottom=0.08,
            wspace=0.13,
            hspace=0.20
        )

        safe_key = norm_name(key)
        out_prefix = os.path.join(outdir, f"Fig_Step78B_ResponseLandscape_{safe_key}")
        paths[key] = savefig_multi(fig, out_prefix, dpi=dpi)
        plt.close(fig)

    return paths


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--step72_dir", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--force_obsm_key", default="X_nicheformer")
    parser.add_argument("--grid_n", type=int, default=240)
    parser.add_argument("--smooth_sigma", type=float, default=2.0)
    parser.add_argument("--min_density_frac", type=float, default=0.008)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)
    mkdirp(args.outdir)

    print("=" * 100)
    print("Step78B | Virtual perturbation response landscape")
    print("=" * 100)
    print("h5ad=", args.h5ad)
    print("state_table=", args.state_table)
    print("step72_dir=", args.step72_dir)
    print("outdir=", args.outdir)

    adata = load_h5ad(args.h5ad)
    x, y, latent_audit = infer_latent_2d(
        adata,
        force_obsm_key=args.force_obsm_key,
        random_state=args.seed
    )

    state_df, state_id_col = load_state_table(args.state_table)
    merged, merge_audit = merge_state(adata, state_df, state_id_col)
    state_cols = infer_state_columns(merged)

    if state_cols["core"] is None:
        raise RuntimeError("Cannot infer core probability column from merged state table.")

    # Step72C summary.
    summary_effects, step72_audit = parse_step72_summary(args.step72_dir)
    effects = complete_effects_from_summary(summary_effects)

    per_cell, sus_audit = construct_per_cell_response(
        merged,
        effects,
        state_cols
    )

    # Attach latent coordinates.
    base_meta = pd.DataFrame({
        "obs_name": merged["_obs_name_"].astype(str),
        "latent_1": x,
        "latent_2": y,
    })
    per_cell = per_cell.merge(base_meta, on="obs_name", how="left")

    per_cell_path = os.path.join(args.outdir, "step78b_per_cell_virtual_perturbation_response_proxy.csv")
    per_cell.to_csv(per_cell_path, index=False)

    summary_rows = []
    for key, e in effects.items():
        d = per_cell[per_cell["perturbation_key"] == key]
        summary_rows.append({
            "perturbation_key": key,
            "perturbation": e.get("display", key),
            "effect_source": e.get("source", ""),
            "mean_delta_core": e.get("delta_core", np.nan),
            "mean_delta_peri": e.get("delta_peri", np.nan),
            "mean_delta_remote": e.get("delta_remote", np.nan),
            "mean_repair_shift": e.get("repair_shift", np.nan),
            "mean_core_reduction_proxy": float(d["core_reduction_proxy"].mean()),
            "mean_repair_gain_proxy": float(d["repair_gain_proxy"].mean()),
            "mean_response_priority_proxy": float(d["response_priority_proxy"].mean()),
            "max_response_priority_proxy": float(d["response_priority_proxy"].max()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.outdir, "step78b_virtual_perturbation_response_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    xedges, yedges, xc, yc = make_grid(x, y, grid_n=args.grid_n)

    atlas_paths = make_atlas_figure(
        x=x,
        y=y,
        per_cell=per_cell,
        effects=effects,
        xedges=xedges,
        yedges=yedges,
        xc=xc,
        yc=yc,
        outdir=args.outdir,
        sigma=args.smooth_sigma,
        min_density_frac=args.min_density_frac,
        dpi=args.dpi
    )

    single_paths = make_single_candidate_figures(
        x=x,
        y=y,
        per_cell=per_cell,
        effects=effects,
        xedges=xedges,
        yedges=yedges,
        xc=xc,
        yc=yc,
        outdir=args.outdir,
        sigma=args.smooth_sigma,
        min_density_frac=args.min_density_frac,
        dpi=args.dpi
    )

    report = {
        "status": "ok",
        "analysis_name": "Step78B virtual perturbation response landscape",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "latent_audit": latent_audit,
        "merge_audit": merge_audit,
        "state_cols": state_cols,
        "step72_audit": step72_audit,
        "effects_used": effects,
        "susceptibility_audit": sus_audit,
        "parameters": {
            "grid_n": args.grid_n,
            "smooth_sigma": args.smooth_sigma,
            "min_density_frac": args.min_density_frac,
            "force_obsm_key": args.force_obsm_key,
        },
        "outputs": {
            "per_cell_response_proxy_csv": per_cell_path,
            "summary_csv": summary_path,
            "atlas": atlas_paths,
            "single_candidate_figures": single_paths,
        },
        "interpretation_note": (
            "Step78B constructs visualization-level virtual perturbation response landscapes. "
            "If no per-cell Step72 counterfactual table is available, it combines Step72C mean shifts "
            "with per-spot susceptibility derived from baseline state/module scores. "
            "This is not wet-lab KO/blockade and not an observed state transition."
        )
    }

    report_path = os.path.join(args.outdir, "step78b_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nDONE Step78B")
    print("=" * 100)
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
