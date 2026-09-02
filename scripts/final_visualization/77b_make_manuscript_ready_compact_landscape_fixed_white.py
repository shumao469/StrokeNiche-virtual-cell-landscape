#!/usr/bin/env python3
# WHITE-BACKGROUND VARIANT GENERATED 2026-09-01.
# Source: E:\vir\ST\77b_make_manuscript_ready_compact_landscape_fixed.py
# Scientific calculations and data mappings are unchanged; only visual theme literals were remapped.
# -*- coding: utf-8 -*-

"""
Step77B fixed | Manuscript-ready compact latent state-probability landscape

Panels:
A. Core-probability landscape
B. Repair-score landscape
C. Core probability 3D surface
D. Core ridge / peak map

Key fixes:
1. Prevents bottom caption clipping.
2. Smooths 3D surface to reduce artificial spikes.
3. Clarifies that core is state probability and repair is model-derived score.
4. Supports X_nicheformer forced latent embeddings.
"""

import os
import json
import textwrap
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -------------------------
# Optional anndata import
# -------------------------
def import_anndata():
    try:
        import anndata as ad
        return ad
    except Exception as e:
        raise ImportError(
            "Cannot import anndata. Please install it in the current environment:\n"
            "  pip install anndata\n"
            f"Original error: {repr(e)}"
        )


# -------------------------
# Basic utilities
# -------------------------
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def minmax01(x, qlow=0.01, qhigh=0.99):
    x = np.asarray(x, dtype=float)
    y = x.copy()
    good = np.isfinite(y)
    if good.sum() == 0:
        return np.zeros_like(y)
    lo, hi = np.nanquantile(y[good], [qlow, qhigh])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(y[good]), np.nanmax(y[good])
    if hi <= lo:
        out = np.zeros_like(y)
        out[good] = 0.5
        return out
    y = (y - lo) / (hi - lo)
    y = np.clip(y, 0, 1)
    y[~good] = np.nan
    return y


def clean_label(s):
    s = str(s)
    s = s.replace("_", " ")
    s = s.replace("probability", "prob.")
    s = s.replace("lesion core like", "lesion-core-like")
    s = s.replace("peri infarct", "peri-infarct")
    s = s.replace("remote like", "remote-like")
    return s


def add_panel_label(ax, label, color="#1f2937"):
    """
    Works for both 2D and 3D axes.
    Axes3D.text() requires x, y, z, s; use text2D for 3D axes.
    """
    if hasattr(ax, "text2D"):
        ax.text2D(
            -0.075, 1.035, label,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            color=color,
            va="top",
            ha="left"
        )
    else:
        ax.text(
            -0.075, 1.035, label,
            transform=ax.transAxes,
            fontsize=17,
            fontweight="bold",
            color=color,
            va="top",
            ha="left"
        )


# -------------------------
# Data loading / merging
# -------------------------
def read_h5ad(h5ad):
    ad = import_anndata()
    return ad.read_h5ad(h5ad)


def infer_merge_key(adata, state_df):
    """
    Try robust merge between adata.obs_names and state table.
    Preferred state columns: obs_name, barcode, cell_id, spot_id.
    """
    obs_names = pd.Index(adata.obs_names.astype(str))
    candidates = [
        "obs_name", "barcode", "cell_id", "spot_id", "spot", "cell",
        "adata_obs_name", "_obs_names_", "index"
    ]

    best = None
    best_overlap = -1

    for c in candidates:
        if c in state_df.columns:
            vals = state_df[c].astype(str)
            overlap = vals.isin(obs_names).sum()
            if overlap > best_overlap:
                best = c
                best_overlap = overlap

    if best is not None and best_overlap > 0:
        return {
            "mode": "id",
            "state_key": best,
            "overlap": int(best_overlap)
        }

    # Fallback: same length/order
    if state_df.shape[0] == adata.n_obs:
        return {
            "mode": "order",
            "state_key": None,
            "overlap": int(adata.n_obs)
        }

    raise ValueError(
        "Cannot merge state_table with adata.obs_names. "
        "Please ensure state_table has obs_name/barcode/cell_id or has the same row order."
    )


def merge_state_table(adata, state_table):
    state_df = pd.read_csv(state_table)
    state_df = state_df.copy()

    meta = pd.DataFrame({
        "_adata_id_": adata.obs_names.astype(str)
    })

    info = infer_merge_key(adata, state_df)

    if info["mode"] == "id":
        key = info["state_key"]
        state_df["_state_id_"] = state_df[key].astype(str)
        merged = meta.merge(
            state_df,
            left_on="_adata_id_",
            right_on="_state_id_",
            how="left",
            suffixes=("", "_state")
        )
    else:
        merged = pd.concat(
            [meta.reset_index(drop=True), state_df.reset_index(drop=True)],
            axis=1
        )

    overlap = int(merged.drop(columns=["_adata_id_"], errors="ignore").notna().any(axis=1).sum())

    audit = {
        "state_table_available": True,
        "state_table_rows": int(state_df.shape[0]),
        "merge_mode": info["mode"],
        "merge_key_state": info.get("state_key"),
        "state_merge_overlap": overlap,
        "adata_n_obs": int(adata.n_obs)
    }

    return merged, audit


# -------------------------
# Latent coordinates
# -------------------------
def get_latent_coords(
    adata,
    force_obsm_key="X_nicheformer",
    pca_random_state=1,
    max_pca_cells=50000
):
    if force_obsm_key is not None:
        if force_obsm_key not in adata.obsm.keys():
            raise ValueError(
                f"--force_obsm_key={force_obsm_key} not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        X = np.asarray(adata.obsm[force_obsm_key])
        source = force_obsm_key
        forced = True
    else:
        # fallback priority
        for k in ["X_nicheformer", "X_scgpt", "X_pca", "X_umap", "spatial"]:
            if k in adata.obsm.keys():
                X = np.asarray(adata.obsm[k])
                source = k
                forced = False
                break
        else:
            raise ValueError(f"No usable obsm key found. Available: {list(adata.obsm.keys())}")

    if X.ndim != 2:
        raise ValueError(f"Latent matrix from {source} is not 2D.")

    n, d = X.shape

    if d >= 2 and source.lower().endswith("umap"):
        coords = X[:, :2].astype(float)
        mode = "obsm_direct_2d"
    elif d == 2:
        coords = X.astype(float)
        mode = "obsm_direct_2d"
    else:
        # Standardize then PCA to 2D
        X2 = X.astype(np.float32)
        # replace non-finite
        X2[~np.isfinite(X2)] = 0.0
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X2)
        pca = PCA(n_components=2, random_state=pca_random_state)
        coords = pca.fit_transform(Xs)
        mode = "obsm_forced_pca" if forced else "obsm_pca"
        evr = pca.explained_variance_ratio_.tolist()

    # Orient for reproducible visual layout: longer horizontal axis
    if np.nanstd(coords[:, 1]) > np.nanstd(coords[:, 0]):
        coords = coords[:, [1, 0]]

    # Mild robust scaling for display stability
    for j in [0, 1]:
        med = np.nanmedian(coords[:, j])
        q1, q99 = np.nanquantile(coords[:, j], [0.01, 0.99])
        scale = max(q99 - q1, 1e-6)
        coords[:, j] = (coords[:, j] - med) / scale * 10.0

    audit = {
        "mode": mode,
        "obsm_key": source,
        "n_dim": int(d),
        "forced_obsm_priority": bool(force_obsm_key is not None),
    }
    if "evr" in locals():
        audit["pca_explained_variance_ratio"] = evr

    return coords, audit


# -------------------------
# Column inference
# -------------------------
def infer_col(df, user_value, candidates, required=True, label="column"):
    if user_value and user_value != "auto":
        if user_value not in df.columns:
            raise ValueError(f"Requested {label} '{user_value}' not found.")
        return user_value

    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise ValueError(
            f"Cannot infer {label}. Tried: {candidates}\n"
            f"Available columns include: {list(df.columns)[:80]}"
        )
    return None


def infer_state_label_col(df, user_col):
    candidates = [
        "state_group",
        "region_refined",
        "region_manual_final",
        "region_auto",
        "state_label",
        "label",
        "region"
    ]
    return infer_col(df, user_col, candidates, required=False, label="state label column")


def infer_core_col(df, user_col):
    candidates = [
        "core_probability",
        "core_prob",
        "prob_core",
        "p_core",
        "lesion_core_probability",
        "lesion_core_prob",
        "lesion-core-like_probability",
        "lesion_core_like_probability",
        "core_like_probability"
    ]
    return infer_col(df, user_col, candidates, required=False, label="core probability column")


def infer_repair_col(df, user_col):
    candidates = [
        "repair_score",
        "repair_probability",
        "repair_prob",
        "prob_repair",
        "repair",
        "repair_ECM_score",
        "repair_ecm_score",
        "repair_module_score",
        "model_repair_score"
    ]
    return infer_col(df, user_col, candidates, required=False, label="repair score column")


def make_core_values(meta, core_col, state_label_col):
    """
    Prefer numeric core probability column. If unavailable, derive one-hot from label.
    """
    if core_col is not None:
        vals = pd.to_numeric(meta[core_col], errors="coerce").to_numpy(dtype=float)
        vals = np.clip(vals, 0, 1)
        source = "model_probability"
        return vals, source

    if state_label_col is None:
        raise ValueError("No core probability column and no state_label_col to derive core one-hot.")

    lab = meta[state_label_col].astype(str).str.lower()
    core_mask = (
        lab.str.contains("core", regex=False)
        | lab.str.contains("lesion", regex=False)
        | lab.str.contains("infarct_core", regex=False)
    )
    vals = core_mask.astype(float).to_numpy()
    source = "label_onehot_smoothed"
    return vals, source


def make_repair_values(meta, repair_col):
    if repair_col is None:
        raise ValueError(
            "Cannot infer repair score column. Please specify --repair_col."
        )
    vals = pd.to_numeric(meta[repair_col], errors="coerce").to_numpy(dtype=float)
    vals = minmax01(vals, qlow=0.01, qhigh=0.99)
    source = "model_score"
    return vals, source


# -------------------------
# Landscape estimation
# -------------------------
def make_grid(coords, grid_n=260, pad_frac=0.04):
    x = coords[:, 0]
    y = coords[:, 1]
    xlo, xhi = np.nanquantile(x, [0.005, 0.995])
    ylo, yhi = np.nanquantile(y, [0.005, 0.995])

    xpad = (xhi - xlo) * pad_frac
    ypad = (yhi - ylo) * pad_frac

    x_edges = np.linspace(xlo - xpad, xhi + xpad, grid_n + 1)
    y_edges = np.linspace(ylo - ypad, yhi + ypad, grid_n + 1)
    x_cent = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_cent = 0.5 * (y_edges[:-1] + y_edges[1:])

    return x_edges, y_edges, x_cent, y_cent


def binned_smooth_landscape(
    coords,
    values,
    grid_n=260,
    smooth_sigma=3.2,
    density_sigma=2.0,
    mask_density_quantile=0.045,
    clip01=True
):
    """
    Nadaraya-Watson style smoothed landscape:
      smooth(sum(value)) / smooth(count)
    Then mask low-density regions to avoid extrapolation.
    """
    x_edges, y_edges, x_cent, y_cent = make_grid(coords, grid_n=grid_n)

    x = coords[:, 0]
    y = coords[:, 1]
    v = np.asarray(values, dtype=float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    xg, yg, vg = x[good], y[good], v[good]

    count, _, _ = np.histogram2d(yg, xg, bins=[y_edges, x_edges])
    weighted, _, _ = np.histogram2d(yg, xg, bins=[y_edges, x_edges], weights=vg)

    density = gaussian_filter(count.astype(float), sigma=density_sigma)
    weighted_s = gaussian_filter(weighted.astype(float), sigma=smooth_sigma)

    denom = gaussian_filter(count.astype(float), sigma=smooth_sigma)
    with np.errstate(divide="ignore", invalid="ignore"):
        grid = weighted_s / np.maximum(denom, 1e-8)

    if clip01:
        grid = np.clip(grid, 0, 1)

    den_good = density[np.isfinite(density)]
    if len(den_good) > 0:
        thr = np.quantile(den_good[den_good > 0], mask_density_quantile) if np.any(den_good > 0) else 0
    else:
        thr = 0
    mask = density >= thr

    grid_masked = grid.copy()
    grid_masked[~mask] = np.nan

    return {
        "grid": grid_masked,
        "grid_raw": grid,
        "density": density,
        "mask": mask,
        "x_cent": x_cent,
        "y_cent": y_cent,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "density_threshold": float(thr)
    }


def surface_smooth(grid, mask, sigma=4.6):
    z = grid.copy()
    valid = np.isfinite(z) & mask
    z0 = np.where(valid, z, 0.0)
    w0 = valid.astype(float)

    zs = gaussian_filter(z0, sigma=sigma)
    ws = gaussian_filter(w0, sigma=sigma)

    with np.errstate(divide="ignore", invalid="ignore"):
        out = zs / np.maximum(ws, 1e-8)

    out[~mask] = np.nan
    out = np.clip(out, 0, 1)
    return out


def find_peaks(grid, mask, peak_quantile=0.92, peak_neighborhood=15, max_peaks=16):
    arr = np.asarray(grid, dtype=float)
    valid = np.isfinite(arr) & mask
    if valid.sum() == 0:
        return []

    vals = arr[valid]
    thr = np.nanquantile(vals, peak_quantile)

    local_max = arr == maximum_filter(
        np.where(valid, arr, -np.inf),
        size=peak_neighborhood,
        mode="nearest"
    )
    peak_mask = valid & local_max & (arr >= thr)

    inds = np.argwhere(peak_mask)
    if inds.shape[0] == 0:
        return []

    scores = arr[peak_mask]
    order = np.argsort(scores)[::-1]
    inds = inds[order]
    scores = scores[order]

    peaks = []
    taken = []
    min_dist = peak_neighborhood * 0.85

    for (ij, score) in zip(inds, scores):
        i, j = int(ij[0]), int(ij[1])
        keep = True
        for ti, tj in taken:
            if np.sqrt((i - ti) ** 2 + (j - tj) ** 2) < min_dist:
                keep = False
                break
        if keep:
            peaks.append((i, j, float(score)))
            taken.append((i, j))
        if len(peaks) >= max_peaks:
            break

    return peaks


# -------------------------
# Plot helpers
# -------------------------
def setup_dark_ax(ax):
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#9ca3af")
        spine.set_linewidth(0.7)
    ax.tick_params(colors="#374151", labelsize=8)
    ax.xaxis.label.set_color("#374151")
    ax.yaxis.label.set_color("#374151")
    ax.grid(color="#e5e7eb", alpha=0.35, linewidth=0.5)


def plot_landscape_2d(
    ax,
    land,
    coords,
    state_labels=None,
    title="Landscape",
    cmap="magma",
    vmin=0,
    vmax=1,
    contour=True,
    scatter=True
):
    setup_dark_ax(ax)

    x_cent = land["x_cent"]
    y_cent = land["y_cent"]
    grid = land["grid"]

    extent = [x_cent.min(), x_cent.max(), y_cent.min(), y_cent.max()]
    im = ax.imshow(
        grid,
        origin="lower",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="auto",
        alpha=0.98
    )

    if contour:
        try:
            levels = np.linspace(vmin, vmax, 9)[2:-1]
            ax.contour(
                x_cent, y_cent, grid,
                levels=levels,
                colors="#9fd3ff",
                linewidths=0.35,
                alpha=0.42
            )
            ax.contour(
                x_cent, y_cent, grid,
                levels=[0.70, 0.85],
                colors=["#ffd166", "#fff6a8"],
                linewidths=[0.75, 0.95],
                alpha=0.80
            )
        except Exception:
            pass

    if scatter:
        x = coords[:, 0]
        y = coords[:, 1]
        if state_labels is None:
            ax.scatter(x, y, s=1.5, c="#66c2ff", alpha=0.28, linewidths=0)
        else:
            lab = pd.Series(state_labels).astype(str).str.lower()
            colors = np.array(["#3db7ff"] * len(lab), dtype=object)
            colors[lab.str.contains("peri", regex=False).to_numpy()] = "#f0b000"
            colors[
                lab.str.contains("core", regex=False).to_numpy()
                | lab.str.contains("lesion", regex=False).to_numpy()
            ] = "#ef4444"
            colors[lab.str.contains("remote", regex=False).to_numpy()] = "#4fc3f7"
            ax.scatter(x, y, s=2.0, c=colors, alpha=0.30, linewidths=0)

    ax.set_title(title, color="#1f2937", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Latent coordinate 1", fontsize=8)
    ax.set_ylabel("Latent coordinate 2", fontsize=8)

    return im


def plot_surface_3d(
    ax,
    land,
    title="Core probability 3D surface",
    cmap="magma",
    surface_smooth_sigma=4.8,
    stride=4
):
    ax.set_facecolor("#ffffff")

    x_cent = land["x_cent"]
    y_cent = land["y_cent"]
    z = surface_smooth(
        land["grid"],
        land["mask"],
        sigma=surface_smooth_sigma
    )

    Xg, Yg = np.meshgrid(x_cent, y_cent)

    # downsample for clean rendering
    Xs = Xg[::stride, ::stride]
    Ys = Yg[::stride, ::stride]
    Zs = z[::stride, ::stride]

    # Replace nan with zero for surface, but make invisible-ish by alpha through mask impossible;
    # keep z=0 outside support to avoid jagged vertical walls.
    Zplot = np.where(np.isfinite(Zs), Zs, 0.0)

    ax.plot_surface(
        Xs, Ys, Zplot,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.96
    )

    try:
        ax.contour(
            Xs, Ys, Zplot,
            zdir="z",
            offset=0,
            levels=np.linspace(0.15, 0.9, 6),
            cmap=cmap,
            linewidths=0.45,
            alpha=0.5
        )
    except Exception:
        pass

    ax.view_init(elev=26, azim=-58)
    ax.set_title(title, color="#1f2937", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Latent 1", color="#374151", fontsize=8, labelpad=4)
    ax.set_ylabel("Latent 2", color="#374151", fontsize=8, labelpad=4)
    ax.set_zlabel("Smoothed value", color="#374151", fontsize=8, labelpad=4)

    ax.tick_params(colors="#374151", labelsize=7)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))

    ax.set_zlim(0, 1)


def plot_peak_map(
    ax,
    land,
    coords,
    state_labels=None,
    title="Core ridge / peak map",
    cmap="magma",
    peak_quantile=0.92,
    peak_neighborhood=15,
    max_peaks=14
):
    im = plot_landscape_2d(
        ax,
        land,
        coords,
        state_labels=state_labels,
        title=title,
        cmap=cmap,
        vmin=0,
        vmax=1,
        contour=True,
        scatter=True
    )

    peaks = find_peaks(
        land["grid"],
        land["mask"],
        peak_quantile=peak_quantile,
        peak_neighborhood=peak_neighborhood,
        max_peaks=max_peaks
    )

    x_cent = land["x_cent"]
    y_cent = land["y_cent"]

    peak_rows = []
    for k, (i, j, score) in enumerate(peaks, start=1):
        x = x_cent[j]
        y = y_cent[i]
        ax.scatter(
            [x], [y],
            marker="*",
            s=78,
            c="#ffea00",
            edgecolors="#111111",
            linewidths=0.7,
            zorder=10
        )
        peak_rows.append({
            "peak_rank": k,
            "grid_i": i,
            "grid_j": j,
            "latent_x": float(x),
            "latent_y": float(y),
            "smoothed_value": float(score)
        })

    return im, pd.DataFrame(peak_rows)


# -------------------------
# Main figure
# -------------------------
def make_compact_figure(
    coords,
    core_land,
    repair_land,
    state_labels,
    outdir,
    surface_smooth_sigma=4.8,
    peak_quantile=0.92,
    peak_neighborhood=15,
    caption_note=None,
    dpi=420
):
    bg = "#ffffff"
    fg = "#111827"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "figure.facecolor": bg,
        "axes.facecolor": bg,
        "savefig.facecolor": bg,
        "text.color": fg,
        "axes.labelcolor": "#374151",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
    })

    fig = plt.figure(figsize=(13.2, 8.6), facecolor=bg)

    # Explicit layout to avoid clipping.
    gs = gridspec.GridSpec(
        nrows=2,
        ncols=2,
        figure=fig,
        left=0.060,
        right=0.865,
        bottom=0.140,
        top=0.870,
        wspace=0.160,
        hspace=0.285
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0], projection="3d")
    axD = fig.add_subplot(gs[1, 1])

    imA = plot_landscape_2d(
        axA,
        core_land,
        coords,
        state_labels=state_labels,
        title="Core-probability landscape",
        cmap="magma"
    )
    add_panel_label(axA, "A")

    imB = plot_landscape_2d(
        axB,
        repair_land,
        coords,
        state_labels=state_labels,
        title="Repair-score landscape",
        cmap="magma"
    )
    add_panel_label(axB, "B")

    plot_surface_3d(
        axC,
        core_land,
        title="Core probability 3D surface",
        cmap="magma",
        surface_smooth_sigma=surface_smooth_sigma,
        stride=4
    )
    add_panel_label(axC, "C")

    imD, peaks_df = plot_peak_map(
        axD,
        core_land,
        coords,
        state_labels=state_labels,
        title="Core ridge / peak map",
        cmap="magma",
        peak_quantile=peak_quantile,
        peak_neighborhood=peak_neighborhood
    )
    add_panel_label(axD, "D")

    # Shared colorbar: core probability and repair score both normalized to 0-1.
    cax = fig.add_axes([0.895, 0.245, 0.018, 0.515])
    cb = fig.colorbar(imD, cax=cax)
    cb.set_label("Smoothed probability / score", color=fg, fontsize=10, labelpad=10)
    cb.ax.tick_params(colors="#374151", labelsize=8)
    cb.outline.set_edgecolor("#374151")

    # Main title
    fig.suptitle(
        "StrokeNiche latent state-probability landscape",
        fontsize=22,
        fontweight="bold",
        color="#1f2937",
        y=0.948
    )

    if caption_note is None:
        caption_note = (
            "Gaussian-kernel smoothing in X_nicheformer latent space; core probability is derived from "
            "state-label/probability smoothing, whereas repair is a model-derived rescue/repair score. "
            "Peaks indicate local high-probability basins. This is a visualization-level landscape, "
            "not a physical energy landscape, not true Waddington potential, and not an observed cell-state transition."
        )

    wrapped = "\n".join(textwrap.wrap(caption_note, width=150))
    fig.text(
        0.5,
        0.052,
        wrapped,
        ha="center",
        va="center",
        fontsize=9.0,
        color="#1f2937",
        linespacing=1.25
    )

    outbase = Path(outdir) / "Fig_Step77B_ManuscriptReadyCompact_LatentLandscape_fixed"
    png = str(outbase) + ".png"
    pdf = str(outbase) + ".pdf"
    svg = str(outbase) + ".svg"

    # Do not use aggressive tight bbox; layout already reserves caption space.
    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    fig.savefig(svg)

    plt.close(fig)

    peaks_csv = Path(outdir) / "step77b_core_peak_coordinates_fixed.csv"
    peaks_df.to_csv(peaks_csv, index=False)

    return {
        "png": png,
        "pdf": pdf,
        "svg": svg,
        "peaks_csv": str(peaks_csv)
    }


# -------------------------
# Audit outputs
# -------------------------
def write_landscape_csv(land, out_csv, value_name):
    x_cent = land["x_cent"]
    y_cent = land["y_cent"]
    Xg, Yg = np.meshgrid(x_cent, y_cent)

    df = pd.DataFrame({
        "latent_x": Xg.ravel(),
        "latent_y": Yg.ravel(),
        value_name: land["grid"].ravel(),
        "density": land["density"].ravel(),
        "mask": land["mask"].ravel().astype(int)
    })
    df.to_csv(out_csv, index=False)


def make_report(
    args,
    adata,
    state_audit,
    latent_audit,
    core_col,
    repair_col,
    core_source,
    repair_source,
    core_land,
    repair_land,
    fig_paths
):
    report = {
        "status": "ok",
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "shape": [int(adata.n_obs), int(adata.n_vars)],
        "state_merge_audit": state_audit,
        "latent_coordinate_audit": latent_audit,
        "columns": {
            "core_col": core_col,
            "repair_col": repair_col,
            "state_label_col": args.state_label_col
        },
        "value_sources": {
            "core": core_source,
            "repair": repair_source
        },
        "smoothing": {
            "grid_n": args.grid_n,
            "smooth_sigma": args.smooth_sigma,
            "density_sigma": args.density_sigma,
            "surface_smooth_sigma": args.surface_smooth_sigma,
            "mask_density_quantile": args.mask_density_quantile,
            "peak_quantile": args.peak_quantile,
            "peak_neighborhood": args.peak_neighborhood
        },
        "density_thresholds": {
            "core": core_land["density_threshold"],
            "repair": repair_land["density_threshold"]
        },
        "outputs": fig_paths,
        "interpretation_note": (
            "Core probability is derived from state-label/probability smoothing. "
            "Repair is a model-derived rescue/repair score normalized for visualization. "
            "These Gaussian-kernel-smoothed latent landscapes are visualization-level maps, "
            "not physical energy landscapes and not observed cell-state transitions."
        )
    }
    return report


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--h5ad", required=True)
    p.add_argument("--state_table", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--force_obsm_key", default="X_nicheformer")
    p.add_argument("--state_label_col", default="state_group")
    p.add_argument("--core_col", default="auto")
    p.add_argument("--repair_col", default="auto")

    p.add_argument("--grid_n", type=int, default=260)
    p.add_argument("--smooth_sigma", type=float, default=3.4)
    p.add_argument("--density_sigma", type=float, default=2.0)
    p.add_argument("--surface_smooth_sigma", type=float, default=5.0)
    p.add_argument("--mask_density_quantile", type=float, default=0.045)

    p.add_argument("--peak_quantile", type=float, default=0.92)
    p.add_argument("--peak_neighborhood", type=int, default=15)

    p.add_argument("--dpi", type=int, default=420)
    p.add_argument("--pca_random_state", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    print("=" * 100)
    print("Step77B fixed | Manuscript-ready compact latent state-probability landscape")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={args.outdir}")

    adata = read_h5ad(args.h5ad)
    meta, state_audit = merge_state_table(adata, args.state_table)

    coords, latent_audit = get_latent_coords(
        adata,
        force_obsm_key=args.force_obsm_key,
        pca_random_state=args.pca_random_state
    )

    state_label_col = infer_state_label_col(meta, args.state_label_col)
    if state_label_col is None:
        print("WARNING: state label column not found; scatter overlay will use default color.")
        state_labels = None
    else:
        state_labels = meta[state_label_col].astype(str).to_numpy()

    core_col = infer_core_col(meta, args.core_col)
    repair_col = infer_repair_col(meta, args.repair_col)

    core_vals, core_source = make_core_values(meta, core_col, state_label_col)
    repair_vals, repair_source = make_repair_values(meta, repair_col)

    # Remove cells without values
    good = (
        np.isfinite(coords[:, 0])
        & np.isfinite(coords[:, 1])
        & np.isfinite(core_vals)
        & np.isfinite(repair_vals)
    )
    coords2 = coords[good]
    core2 = core_vals[good]
    repair2 = repair_vals[good]
    state_labels2 = state_labels[good] if state_labels is not None else None

    print(f"n cells used: {coords2.shape[0]} / {adata.n_obs}")
    print(f"core_col={core_col}; core_source={core_source}")
    print(f"repair_col={repair_col}; repair_source={repair_source}")

    core_land = binned_smooth_landscape(
        coords2,
        core2,
        grid_n=args.grid_n,
        smooth_sigma=args.smooth_sigma,
        density_sigma=args.density_sigma,
        mask_density_quantile=args.mask_density_quantile,
        clip01=True
    )

    repair_land = binned_smooth_landscape(
        coords2,
        repair2,
        grid_n=args.grid_n,
        smooth_sigma=args.smooth_sigma,
        density_sigma=args.density_sigma,
        mask_density_quantile=args.mask_density_quantile,
        clip01=True
    )

    fig_paths = make_compact_figure(
        coords=coords2,
        core_land=core_land,
        repair_land=repair_land,
        state_labels=state_labels2,
        outdir=args.outdir,
        surface_smooth_sigma=args.surface_smooth_sigma,
        peak_quantile=args.peak_quantile,
        peak_neighborhood=args.peak_neighborhood,
        dpi=args.dpi
    )

    # Write landscape grids
    core_grid_csv = Path(args.outdir) / "step77b_core_landscape_grid_fixed.csv"
    repair_grid_csv = Path(args.outdir) / "step77b_repair_landscape_grid_fixed.csv"
    write_landscape_csv(core_land, core_grid_csv, "core_probability_smoothed")
    write_landscape_csv(repair_land, repair_grid_csv, "repair_score_smoothed")

    fig_paths["core_grid_csv"] = str(core_grid_csv)
    fig_paths["repair_grid_csv"] = str(repair_grid_csv)

    report = make_report(
        args=args,
        adata=adata,
        state_audit=state_audit,
        latent_audit=latent_audit,
        core_col=core_col,
        repair_col=repair_col,
        core_source=core_source,
        repair_source=repair_source,
        core_land=core_land,
        repair_land=repair_land,
        fig_paths=fig_paths
    )

    report_json = Path(args.outdir) / "step77b_report_fixed.json"
    with open(report_json, "w") as f:
        json.dump(report, f, indent=2)

    report_txt = Path(args.outdir) / "step77b_report_fixed.txt"
    with open(report_txt, "w") as f:
        f.write("Step77B fixed report\n")
        f.write("=" * 80 + "\n")
        f.write(json.dumps(report, indent=2))

    print("\nDONE Step77B fixed")
    print(json.dumps({
        "figure": fig_paths,
        "report_json": str(report_json),
        "report_txt": str(report_txt)
    }, indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
