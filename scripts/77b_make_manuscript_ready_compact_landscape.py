#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step77B | Manuscript-ready compact latent state-probability landscape

Purpose
-------
Compact 4-panel manuscript-ready figure:
A. Core probability latent landscape
B. Repair-score latent landscape
C. Core probability 3D surface
D. Core ridge / peak map

Important interpretation
------------------------
core/peri/remote landscapes are generated from smoothed one-hot state labels,
not from potentially duplicated model-probability columns.
repair landscape is generated from repair_score/model_score.

This is a visualization-level latent conditional landscape, not a physical
energy surface, not true Waddington lineage potential, and not wet-lab validation.
"""

import os
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy import sparse
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import spearmanr

from sklearn.decomposition import PCA


# -----------------------------
# Basic utilities
# -----------------------------

def mkdir_p(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        if np.isnan(o):
            return None
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def read_h5ad(path):
    try:
        import anndata as ad
    except Exception as e:
        raise ImportError(
            "anndata is required. Run this script in the nicheformer/anndata environment."
        ) from e
    return ad.read_h5ad(path)


def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def lower_str(x):
    return clean_str(x).lower()


def find_col_case_insensitive(df, wanted):
    if wanted in df.columns:
        return wanted
    lowmap = {str(c).lower(): c for c in df.columns}
    return lowmap.get(str(wanted).lower(), None)


def first_existing_col(df, candidates):
    for c in candidates:
        cc = find_col_case_insensitive(df, c)
        if cc is not None:
            return cc
    return None


def robust01(v, q_low=0.01, q_high=0.99):
    v = np.asarray(v, dtype=float)
    out = np.full_like(v, np.nan, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return out
    lo, hi = np.nanquantile(v[ok], [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(v[ok]), np.nanmax(v[ok])
    if hi <= lo:
        out[ok] = 0.5
        return out
    out[ok] = (v[ok] - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return out


def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce").astype(float).values


# -----------------------------
# Merge state table
# -----------------------------

def merge_state_table(adata, state_table_path):
    obs = adata.obs.copy()
    obs["_adata_obs_name_"] = obs.index.astype(str)
    obs["_adata_order_"] = np.arange(obs.shape[0])

    if state_table_path is None or str(state_table_path).strip() == "":
        obs["_merge_key_"] = obs["_adata_obs_name_"]
        return obs, {
            "state_table_available": False,
            "merge_mode": "obs_only",
            "state_table_rows": None,
            "merge_overlap": None,
        }

    st = pd.read_csv(state_table_path)
    st = st.copy()

    candidate_keys = [
        "obs_name", "barcode", "cell_id", "spot_id", "index",
        "_obs_names_", "_adata_id_", "adata_id", "cell", "CellID"
    ]

    obs_keys = set(obs["_adata_obs_name_"].astype(str).values)
    best_key = None
    best_overlap = -1

    for k in candidate_keys:
        if k in st.columns:
            vals = st[k].astype(str).values
            overlap = len(obs_keys.intersection(set(vals)))
            if overlap > best_overlap:
                best_overlap = overlap
                best_key = k

    audit = {
        "state_table_available": True,
        "state_table_rows": int(st.shape[0]),
        "merge_mode": None,
        "merge_key_state": best_key,
        "merge_overlap": None,
    }

    if best_key is not None and best_overlap > 0:
        st["_merge_key_"] = st[best_key].astype(str)
        obs["_merge_key_"] = obs["_adata_obs_name_"].astype(str)

        st_nodup = st.drop_duplicates("_merge_key_", keep="first")
        meta = obs.merge(
            st_nodup,
            on="_merge_key_",
            how="left",
            suffixes=("", "_state")
        )
        meta = meta.sort_values("_adata_order_").reset_index(drop=True)

        matched = meta["_merge_key_"].isin(set(st_nodup["_merge_key_"].astype(str)))
        audit.update({
            "merge_mode": "id",
            "merge_overlap": int(matched.sum()),
            "merge_key_meta": "_adata_obs_name_",
        })
        return meta, audit

    if st.shape[0] == obs.shape[0]:
        meta = pd.concat([obs.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
        audit.update({
            "merge_mode": "row_order",
            "merge_overlap": int(obs.shape[0]),
            "merge_key_meta": "row_order",
        })
        return meta, audit

    raise ValueError(
        f"Could not merge state table. state_table rows={st.shape[0]}, "
        f"adata obs={obs.shape[0]}, best_key={best_key}, best_overlap={best_overlap}"
    )


# -----------------------------
# Latent coordinates
# -----------------------------

def infer_latent_coords(adata, force_obsm_key="X_nicheformer", seed=1, flip_x=False, flip_y=False):
    if force_obsm_key not in adata.obsm.keys():
        raise ValueError(
            f"--force_obsm_key={force_obsm_key} not found in adata.obsm. "
            f"Available keys: {list(adata.obsm.keys())}"
        )

    Z = adata.obsm[force_obsm_key]
    if sparse.issparse(Z):
        Z = Z.toarray()
    Z = np.asarray(Z)

    if Z.ndim != 2:
        raise ValueError(f"adata.obsm[{force_obsm_key}] must be 2D, got {Z.shape}")

    if Z.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(Z)
        audit = {
            "mode": "obsm_forced_pca",
            "obsm_key": force_obsm_key,
            "n_dim": int(Z.shape[1]),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        }
    else:
        coords = np.column_stack([Z[:, 0], np.zeros(Z.shape[0])])
        audit = {
            "mode": "obsm_forced_1d_padded",
            "obsm_key": force_obsm_key,
            "n_dim": int(Z.shape[1]),
        }

    if flip_x:
        coords[:, 0] *= -1
    if flip_y:
        coords[:, 1] *= -1

    return coords, audit


# -----------------------------
# State label one-hot targets
# -----------------------------

def infer_state_label(meta, state_label_col):
    col = find_col_case_insensitive(meta, state_label_col)
    if col is None:
        alt = find_col_case_insensitive(meta, f"{state_label_col}_state")
        if alt is not None:
            col = alt

    if col is None:
        candidates = [
            "state_group", "region_refined", "region_manual_final",
            "region", "state", "label", "class"
        ]
        col = first_existing_col(meta, candidates)

    if col is None:
        raise ValueError(
            f"Could not find state label column. Requested={state_label_col}. "
            f"Available columns include: {list(meta.columns)[:50]}"
        )

    labels_raw = meta[col].astype(str).values
    labels = np.array([lower_str(x) for x in labels_raw])

    core = np.array([
        ("core" in x) or ("lesion-core" in x) or ("lesion_core" in x)
        for x in labels
    ], dtype=float)

    peri = np.array([
        ("peri" in x) or ("penumbra" in x)
        for x in labels
    ], dtype=float)

    remote = np.array([
        ("remote" in x) or ("normal" in x) or ("healthy" in x)
        for x in labels
    ], dtype=float)

    # If labels are exactly simple names.
    core = np.maximum(core, np.array([x in ["core", "lesion", "lesioncore"] for x in labels], dtype=float))
    peri = np.maximum(peri, np.array([x in ["peri", "peri-infarct", "periinfarct"] for x in labels], dtype=float))
    remote = np.maximum(remote, np.array([x in ["remote", "remote-like"] for x in labels], dtype=float))

    counts = {
        "core_n": int(np.nansum(core)),
        "peri_n": int(np.nansum(peri)),
        "remote_n": int(np.nansum(remote)),
    }

    if min(counts.values()) == 0:
        raise ValueError(
            f"One or more state one-hot targets are empty using label column {col}. "
            f"Counts={counts}. Please check state_label_col."
        )

    return {
        "core": core,
        "peri": peri,
        "remote": remote,
    }, labels_raw, {
        "state_label_col_used": col,
        "state_counts": counts,
        "target_source": "label_onehot",
    }


def infer_repair_score(meta, repair_col="auto"):
    if repair_col is not None and repair_col != "auto":
        col = find_col_case_insensitive(meta, repair_col)
        if col is None:
            col = find_col_case_insensitive(meta, f"{repair_col}_state")
        if col is None:
            raise ValueError(f"--repair_col={repair_col} not found in merged metadata.")
    else:
        candidates = [
            "repair_score", "repair_probability", "repair_prob",
            "p_repair", "prob_repair", "repair",
            "rescue_score", "repair_ecm_score",
            "module_repair_score",
            "repair_score_state", "repair_probability_state",
        ]
        col = first_existing_col(meta, candidates)

    if col is None:
        raise ValueError(
            "Could not infer repair score column. Please provide --repair_col repair_score "
            "or the correct column name."
        )

    raw = safe_numeric(meta[col])
    if np.isfinite(raw).sum() == 0:
        raise ValueError(f"Repair column {col} has no numeric values.")

    val = robust01(raw)
    return val, {
        "repair_col_used": col,
        "repair_source": "model_score",
        "repair_raw_min": float(np.nanmin(raw)),
        "repair_raw_max": float(np.nanmax(raw)),
        "repair_normalized_for_plot": True,
    }


# -----------------------------
# Grid smoothing
# -----------------------------

def make_common_grid(coords, grid_n=260, crop_q=0.002, pad_frac=0.04):
    x = coords[:, 0]
    y = coords[:, 1]

    xmin, xmax = np.quantile(x, [crop_q, 1 - crop_q])
    ymin, ymax = np.quantile(y, [crop_q, 1 - crop_q])

    dx = xmax - xmin
    dy = ymax - ymin
    xmin -= dx * pad_frac
    xmax += dx * pad_frac
    ymin -= dy * pad_frac
    ymax += dy * pad_frac

    xbins = np.linspace(xmin, xmax, grid_n + 1)
    ybins = np.linspace(ymin, ymax, grid_n + 1)
    xc = (xbins[:-1] + xbins[1:]) / 2
    yc = (ybins[:-1] + ybins[1:]) / 2
    Xg, Yg = np.meshgrid(xc, yc)

    return {
        "xbins": xbins,
        "ybins": ybins,
        "xc": xc,
        "yc": yc,
        "Xg": Xg,
        "Yg": Yg,
        "extent": [xmin, xmax, ymin, ymax],
    }


def smooth_target_on_grid(coords, values, grid, smooth_sigma=3.0,
                          density_sigma=2.0, mask_density_quantile=0.05):
    x = coords[:, 0]
    y = coords[:, 1]
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)

    H_count, _, _ = np.histogram2d(
        y[ok], x[ok],
        bins=[grid["ybins"], grid["xbins"]]
    )
    H_sum, _, _ = np.histogram2d(
        y[ok], x[ok],
        bins=[grid["ybins"], grid["xbins"]],
        weights=v[ok]
    )

    H_count_s = gaussian_filter(H_count.astype(float), sigma=smooth_sigma)
    H_sum_s = gaussian_filter(H_sum.astype(float), sigma=smooth_sigma)
    Z = H_sum_s / (H_count_s + 1e-12)

    density = gaussian_filter(H_count.astype(float), sigma=density_sigma)
    if np.nanmax(density) > 0:
        density_norm = density / np.nanmax(density)
    else:
        density_norm = density

    positive = density_norm[density_norm > 0]
    if positive.size > 0:
        thr = np.quantile(positive, mask_density_quantile)
    else:
        thr = 0.0

    mask = density_norm >= thr
    Z_masked = np.array(Z, dtype=float)
    Z_masked[~mask] = np.nan

    return {
        "Z": Z,
        "Z_masked": Z_masked,
        "density": density_norm,
        "mask": mask,
        "density_threshold": float(thr),
    }


def compute_all_landscapes(coords, targets, grid, args):
    out = {}
    for name, val in targets.items():
        out[name] = smooth_target_on_grid(
            coords, val, grid,
            smooth_sigma=args.smooth_sigma,
            density_sigma=args.density_sigma,
            mask_density_quantile=args.mask_density_quantile
        )
    return out


# -----------------------------
# Peak detection
# -----------------------------

def find_peaks(Z_masked, grid, max_peaks=10, min_quantile=0.90, neighborhood=13):
    Z = np.array(Z_masked, dtype=float)
    finite = np.isfinite(Z)

    if finite.sum() == 0:
        return pd.DataFrame(columns=["peak_rank", "x", "y", "value", "grid_i", "grid_j"])

    threshold = np.nanquantile(Z[finite], min_quantile)
    Z_fill = np.where(finite, Z, -np.inf)

    local_max = Z_fill == maximum_filter(Z_fill, size=neighborhood, mode="nearest")
    keep = local_max & finite & (Z >= threshold)

    yy, xx = np.where(keep)
    vals = Z[yy, xx]

    if len(vals) == 0:
        return pd.DataFrame(columns=["peak_rank", "x", "y", "value", "grid_i", "grid_j"])

    order = np.argsort(vals)[::-1][:max_peaks]
    rows = []
    for rank, idx in enumerate(order, start=1):
        i = int(yy[idx])
        j = int(xx[idx])
        rows.append({
            "peak_rank": rank,
            "x": float(grid["xc"][j]),
            "y": float(grid["yc"][i]),
            "value": float(vals[idx]),
            "grid_i": i,
            "grid_j": j,
        })

    return pd.DataFrame(rows)


# -----------------------------
# Audit
# -----------------------------

def spearman_matrix(values_dict):
    keys = list(values_dict.keys())
    M = np.full((len(keys), len(keys)), np.nan)
    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            va = np.asarray(values_dict[a], dtype=float)
            vb = np.asarray(values_dict[b], dtype=float)
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() < 3:
                continue
            if np.nanstd(va[ok]) == 0 or np.nanstd(vb[ok]) == 0:
                continue
            M[i, j] = spearmanr(va[ok], vb[ok]).correlation
    return pd.DataFrame(M, index=keys, columns=keys)


def grid_vector(Z):
    z = np.asarray(Z, dtype=float).ravel()
    return z[np.isfinite(z)]


def landscape_spearman_matrix(landscapes):
    keys = list(landscapes.keys())
    M = np.full((len(keys), len(keys)), np.nan)

    for i, a in enumerate(keys):
        for j, b in enumerate(keys):
            Za = landscapes[a]["Z_masked"].ravel()
            Zb = landscapes[b]["Z_masked"].ravel()
            ok = np.isfinite(Za) & np.isfinite(Zb)
            if ok.sum() < 10:
                continue
            if np.nanstd(Za[ok]) == 0 or np.nanstd(Zb[ok]) == 0:
                continue
            M[i, j] = spearmanr(Za[ok], Zb[ok]).correlation

    return pd.DataFrame(M, index=keys, columns=keys)


# -----------------------------
# Plotting
# -----------------------------

def set_dark_axis(ax):
    ax.set_facecolor("#05070b")
    ax.tick_params(colors="#d8dee9", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#7b8794")
        spine.set_linewidth(0.8)
    ax.grid(color="#2c3440", alpha=0.30, linewidth=0.5)


def state_colors_from_labels(labels_raw):
    colors = []
    for x in labels_raw:
        lx = lower_str(x)
        if "core" in lx or "lesion-core" in lx:
            colors.append("#e63946")
        elif "peri" in lx or "penumbra" in lx:
            colors.append("#f2b701")
        elif "remote" in lx or "normal" in lx:
            colors.append("#39a9f9")
        else:
            colors.append("#9aa4b2")
    return np.array(colors)


def plot_landscape_2d(ax, coords, labels_raw, grid, Z_masked, title,
                      cmap="magma", vmin=0, vmax=1, overlay_points=True):
    set_dark_axis(ax)

    im = ax.imshow(
        Z_masked,
        origin="lower",
        extent=grid["extent"],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="auto",
    )

    finite = np.isfinite(Z_masked)
    if finite.sum() > 100:
        levels = np.linspace(np.nanquantile(Z_masked[finite], 0.25),
                             np.nanquantile(Z_masked[finite], 0.95), 7)
        try:
            ax.contour(
                grid["Xg"], grid["Yg"], Z_masked,
                levels=levels,
                colors="#9dd6df",
                linewidths=0.45,
                alpha=0.55
            )
            ax.contour(
                grid["Xg"], grid["Yg"], Z_masked,
                levels=[np.nanquantile(Z_masked[finite], 0.85)],
                colors="#ffd166",
                linewidths=0.9,
                alpha=0.85
            )
        except Exception:
            pass

    if overlay_points:
        c = state_colors_from_labels(labels_raw)
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=2.0,
            c=c,
            alpha=0.32,
            linewidths=0,
            rasterized=True
        )

    ax.set_title(title, color="white", fontsize=13, weight="bold", pad=8)
    ax.set_xlabel("Latent coordinate 1", color="#d8dee9", fontsize=9)
    ax.set_ylabel("Latent coordinate 2", color="#d8dee9", fontsize=9)
    ax.set_xlim(grid["extent"][0], grid["extent"][1])
    ax.set_ylim(grid["extent"][2], grid["extent"][3])
    return im


def plot_core_3d(ax, grid, Z_masked, title, cmap="magma"):
    ax.set_facecolor("#05070b")
    X = grid["Xg"]
    Y = grid["Yg"]
    Z = np.array(Z_masked, dtype=float)

    ax.plot_surface(
        X, Y, Z,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
        rcount=120,
        ccount=120
    )

    ax.contour(
        X, Y, Z,
        zdir="z",
        offset=0,
        cmap="gray",
        levels=8,
        linewidths=0.4,
        alpha=0.35
    )

    ax.set_title(title, color="white", fontsize=13, weight="bold", pad=8)
    ax.set_xlabel("Latent 1", color="#d8dee9", labelpad=5, fontsize=9)
    ax.set_ylabel("Latent 2", color="#d8dee9", labelpad=5, fontsize=9)
    ax.set_zlabel("Smoothed value", color="#d8dee9", labelpad=5, fontsize=9)
    ax.tick_params(colors="#d8dee9", labelsize=8)

    ax.set_zlim(0, 1)
    ax.view_init(elev=28, azim=-58)

    try:
        ax.xaxis.pane.set_facecolor((0.03, 0.04, 0.06, 1.0))
        ax.yaxis.pane.set_facecolor((0.03, 0.04, 0.06, 1.0))
        ax.zaxis.pane.set_facecolor((0.03, 0.04, 0.06, 1.0))
    except Exception:
        pass


def plot_peak_map(ax, coords, labels_raw, grid, Z_masked, peak_df, title,
                  cmap="magma"):
    im = plot_landscape_2d(
        ax, coords, labels_raw, grid, Z_masked,
        title=title,
        cmap=cmap,
        vmin=0,
        vmax=1,
        overlay_points=True
    )

    if peak_df is not None and peak_df.shape[0] > 0:
        ax.scatter(
            peak_df["x"], peak_df["y"],
            marker="*",
            s=115,
            c="#f8f32b",
            edgecolors="#1b1b1b",
            linewidths=0.6,
            zorder=10
        )
        for _, r in peak_df.head(6).iterrows():
            ax.text(
                r["x"], r["y"], str(int(r["peak_rank"])),
                color="black",
                fontsize=7,
                ha="center",
                va="center",
                zorder=11,
                weight="bold"
            )

    return im


def add_panel_label(ax, label):
    """
    Add panel label for both 2D and 3D axes.
    Axes3D.text() requires x, y, z, s, so use text2D for 3D axes.
    """
    if hasattr(ax, "text2D"):
        ax.text2D(
            -0.08, 1.06, label,
            transform=ax.transAxes,
            fontsize=16,
            weight="bold",
            color="white",
            va="top",
            ha="left"
        )
    else:
        ax.text(
            -0.08, 1.06, label,
            transform=ax.transAxes,
            fontsize=16,
            weight="bold",
            color="white",
            va="top",
            ha="left"
        )


def make_compact_figure(coords, labels_raw, grid, landscapes, peak_df, outdir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(15.5, 11.2), facecolor="#05070b")
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        left=0.055,
        right=0.90,
        top=0.90,
        bottom=0.09,
        wspace=0.16,
        hspace=0.26
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0], projection="3d")
    axD = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        "StrokeNiche latent state-probability landscape",
        fontsize=24,
        color="white",
        weight="bold",
        y=0.965
    )

    imA = plot_landscape_2d(
        axA, coords, labels_raw, grid,
        landscapes["core"]["Z_masked"],
        "Core-probability landscape",
        cmap="magma"
    )
    add_panel_label(axA, "A")

    imB = plot_landscape_2d(
        axB, coords, labels_raw, grid,
        landscapes["repair"]["Z_masked"],
        "Repair-score landscape",
        cmap="magma"
    )
    add_panel_label(axB, "B")

    plot_core_3d(
        axC, grid,
        landscapes["core"]["Z_masked"],
        "Core probability 3D surface",
        cmap="magma"
    )
    add_panel_label(axC, "C")

    imD = plot_peak_map(
        axD, coords, labels_raw, grid,
        landscapes["core"]["Z_masked"],
        peak_df,
        "Core ridge / peak map",
        cmap="magma"
    )
    add_panel_label(axD, "D")

    cax = fig.add_axes([0.925, 0.20, 0.018, 0.62])
    cb = fig.colorbar(imA, cax=cax)
    cb.set_label("Smoothed probability / score", color="white", fontsize=11)
    cb.ax.tick_params(colors="white", labelsize=9)
    cb.outline.set_edgecolor("#d8dee9")

    fig.text(
        0.50, 0.035,
        "Landscapes use target-specific Gaussian-kernel smoothing in X_nicheformer latent space; "
        "core/peri/remote are smoothed state-label indicators, whereas repair is a model-derived score. "
        "Peaks indicate local high-probability basins; this is a visualization-level landscape, not physical energy or functional validation.",
        color="#d8dee9",
        ha="center",
        va="center",
        fontsize=10
    )

    png = os.path.join(outdir, "Fig_Step77B_ManuscriptReadyCompact_LatentLandscape.png")
    pdf = os.path.join(outdir, "Fig_Step77B_ManuscriptReadyCompact_LatentLandscape.pdf")
    svg = os.path.join(outdir, "Fig_Step77B_ManuscriptReadyCompact_LatentLandscape.svg")

    fig.savefig(png, dpi=450, facecolor=fig.get_facecolor())
    fig.savefig(pdf, facecolor=fig.get_facecolor())
    fig.savefig(svg, facecolor=fig.get_facecolor())
    plt.close(fig)

    return {"png": png, "pdf": pdf, "svg": svg}


# -----------------------------
# Save tables
# -----------------------------

def save_grid_table(grid, landscapes, outdir):
    rows = []
    for name, obj in landscapes.items():
        Z = obj["Z_masked"]
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                val = Z[i, j]
                if np.isfinite(val):
                    rows.append({
                        "target": name,
                        "x": float(grid["xc"][j]),
                        "y": float(grid["yc"][i]),
                        "value": float(val),
                        "density": float(obj["density"][i, j]),
                        "mask": bool(obj["mask"][i, j]),
                    })
    df = pd.DataFrame(rows)
    path = os.path.join(outdir, "step77b_compact_landscape_grid_values.csv")
    df.to_csv(path, index=False)
    return path


def save_per_cell_table(coords, labels_raw, targets, outdir):
    df = pd.DataFrame({
        "latent_1": coords[:, 0],
        "latent_2": coords[:, 1],
        "state_label": labels_raw,
    })
    for k, v in targets.items():
        df[k] = v
    path = os.path.join(outdir, "step77b_per_cell_targets_used.csv")
    df.to_csv(path, index=False)
    return path


# -----------------------------
# Main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--h5ad",
        default="/mnt/h/vir/ST/results/step5_nicheformer/spatial_all_with_nicheformer.h5ad",
        help="h5ad containing X_nicheformer in obsm."
    )
    ap.add_argument(
        "--state_table",
        default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv",
        help="State/probability table used in Step64/77A."
    )
    ap.add_argument(
        "--outdir",
        default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/latent_state_probability_landscape_77b_manuscript_ready_compact",
        help="Output directory."
    )
    ap.add_argument("--force_obsm_key", default="X_nicheformer")
    ap.add_argument("--state_label_col", default="state_group")
    ap.add_argument("--repair_col", default="auto")

    ap.add_argument("--grid_n", type=int, default=260)
    ap.add_argument("--smooth_sigma", type=float, default=3.0)
    ap.add_argument("--density_sigma", type=float, default=2.0)
    ap.add_argument("--mask_density_quantile", type=float, default=0.05)
    ap.add_argument("--crop_q", type=float, default=0.002)
    ap.add_argument("--pad_frac", type=float, default=0.04)

    ap.add_argument("--max_peaks", type=int, default=10)
    ap.add_argument("--peak_quantile", type=float, default=0.90)
    ap.add_argument("--peak_neighborhood", type=int, default=13)

    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--flip_x", action="store_true")
    ap.add_argument("--flip_y", action="store_true")

    return ap.parse_args()


def main():
    args = parse_args()
    mkdir_p(args.outdir)

    print("=" * 100)
    print("Step77B | Manuscript-ready compact latent state-probability landscape")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={args.outdir}")

    adata = read_h5ad(args.h5ad)
    meta, merge_audit = merge_state_table(adata, args.state_table)
    coords, latent_audit = infer_latent_coords(
        adata,
        force_obsm_key=args.force_obsm_key,
        seed=args.seed,
        flip_x=args.flip_x,
        flip_y=args.flip_y
    )

    state_targets, labels_raw, state_audit = infer_state_label(
        meta,
        args.state_label_col
    )
    repair, repair_audit = infer_repair_score(meta, args.repair_col)

    targets = {
        "core": state_targets["core"],
        "peri": state_targets["peri"],
        "remote": state_targets["remote"],
        "repair": repair,
    }

    grid = make_common_grid(
        coords,
        grid_n=args.grid_n,
        crop_q=args.crop_q,
        pad_frac=args.pad_frac
    )

    landscapes = compute_all_landscapes(coords, targets, grid, args)

    peak_df = find_peaks(
        landscapes["core"]["Z_masked"],
        grid,
        max_peaks=args.max_peaks,
        min_quantile=args.peak_quantile,
        neighborhood=args.peak_neighborhood
    )

    fig_paths = make_compact_figure(
        coords, labels_raw, grid, landscapes, peak_df, args.outdir
    )

    per_cell_path = save_per_cell_table(coords, labels_raw, targets, args.outdir)
    grid_path = save_grid_table(grid, landscapes, args.outdir)

    peak_path = os.path.join(args.outdir, "step77b_core_peak_table.csv")
    peak_df.to_csv(peak_path, index=False)

    raw_corr = spearman_matrix(targets)
    raw_corr_path = os.path.join(args.outdir, "step77b_per_cell_target_spearman.csv")
    raw_corr.to_csv(raw_corr_path)

    grid_corr = landscape_spearman_matrix(landscapes)
    grid_corr_path = os.path.join(args.outdir, "step77b_grid_landscape_spearman.csv")
    grid_corr.to_csv(grid_corr_path)

    report = {
        "status": "ok",
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "outdir": args.outdir,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "obsm_keys": list(adata.obsm.keys()),
        "state_merge_audit": merge_audit,
        "latent_coordinate_audit": latent_audit,
        "state_label_audit": state_audit,
        "repair_audit": repair_audit,
        "landscape_parameters": {
            "grid_n": args.grid_n,
            "smooth_sigma": args.smooth_sigma,
            "density_sigma": args.density_sigma,
            "mask_density_quantile": args.mask_density_quantile,
            "crop_q": args.crop_q,
            "pad_frac": args.pad_frac,
            "peak_quantile": args.peak_quantile,
            "peak_neighborhood": args.peak_neighborhood,
        },
        "target_sources": {
            "core": "label_onehot_smoothed",
            "peri": "label_onehot_smoothed",
            "remote": "label_onehot_smoothed",
            "repair": "model_score",
        },
        "outputs": {
            "figure": fig_paths,
            "per_cell_targets_csv": per_cell_path,
            "grid_values_csv": grid_path,
            "core_peak_table_csv": peak_path,
            "per_cell_target_spearman_csv": raw_corr_path,
            "grid_landscape_spearman_csv": grid_corr_path,
        },
        "interpretation_note": (
            "Step77B is a compact manuscript-ready visualization. "
            "Core/peri/remote landscapes are smoothed state-label indicator landscapes. "
            "Repair is a model-derived repair score landscape. "
            "This is not physical energy, not true Waddington lineage potential, "
            "and not functional validation."
        ),
    }

    report_path = os.path.join(args.outdir, "step77b_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=json_default)

    print("\nDONE Step77B")
    print(json.dumps(report["outputs"], indent=2, default=json_default))


if __name__ == "__main__":
    main()