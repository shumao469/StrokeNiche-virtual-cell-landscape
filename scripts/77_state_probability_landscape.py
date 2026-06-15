#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step77 | Latent state-probability landscape (Scheme B2)
=======================================================

Construct smoothed probability / score landscapes directly in latent 2D space:
    P(core-like | z)
    P(peri-infarct | z)
    P(remote-like | z)
    repair_score(z)   [optional]

Main outputs
------------
1) 2D atlas: heatmap + contour for selected targets
2) Per-target 3D surface
3) Per-target ridge / peak map
4) CSV tables for smoothed grid values and peak coordinates
5) JSON audit report

Interpretation
--------------
This step smooths state probabilities / scores over latent space using
a Nadaraya-Watson Gaussian-kernel regression:
    f(z) = sum_i K(||z-z_i||/h) * y_i / sum_i K(||z-z_i||/h)

It is a visualization-level latent-state landscape, not a physical energy
surface and not functional validation.

Typical usage
-------------
python /mnt/h/vir/ST/77_state_probability_landscape.py \
  --h5ad /mnt/h/vir/ST/results/virtual_cell_h5ad/spatial_all.h5ad \
  --state_table /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv \
  --outdir /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/latent_state_probability_landscape_77_X_nicheformer \
  --force_obsm_key X_nicheformer \
  --state_label_col state_group \
  --time_col timepoint \
  --grid_n 220 \
  --bandwidth auto \
  --targets core,peri,remote,repair \
  --dark_style
"""

import os
import re
import json
import math
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def to_serializable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Series,)):
        return x.to_dict()
    if isinstance(x, (pd.DataFrame,)):
        return x.to_dict(orient="records")
    if isinstance(x, dict):
        return {k: to_serializable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_serializable(v) for v in x]
    return x


def pretty_name(key: str) -> str:
    mp = {
        "core": "Core probability",
        "peri": "Peri-infarct probability",
        "remote": "Remote-like probability",
        "repair": "Repair score",
        "core_probability": "Core probability",
        "peri_probability": "Peri-infarct probability",
        "remote_probability": "Remote-like probability",
        "repair_score": "Repair score",
    }
    if key in mp:
        return mp[key]
    x = key.replace("_", " ").replace("-", " ")
    return x[:1].upper() + x[1:]


def is_prob_like(vals: np.ndarray) -> bool:
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return False
    return (vals.min() >= -0.05) and (vals.max() <= 1.05)


# -----------------------------------------------------------------------------
# Reading and merging
# -----------------------------------------------------------------------------

def infer_state_id_col(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested is not None and requested in df.columns:
        return requested

    candidates = [
        "obs_name", "spot_id", "barcode", "cell_id", "id", "ID",
        "_obs_names_", "Unnamed: 0", "index"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # fallback: if first column looks like an id
    first_col = df.columns[0]
    return first_col


def build_obs_df(adata, adata_id_col: Optional[str] = None) -> pd.DataFrame:
    obs = adata.obs.copy()
    obs = obs.copy()
    obs["_obs_names_"] = obs.index.astype(str)

    if adata_id_col is not None and adata_id_col in obs.columns:
        obs["_adata_id_"] = obs[adata_id_col].astype(str)
        used = adata_id_col
    else:
        obs["_adata_id_"] = obs.index.astype(str)
        used = "_obs_names_"

    return obs, used


def merge_state_table(
    adata,
    state_table_path: Optional[str],
    adata_id_col: Optional[str] = None,
    state_id_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    obs_df, adata_id_used = build_obs_df(adata, adata_id_col=adata_id_col)

    audit = {
        "state_table_available": False,
        "state_table_rows": 0,
        "merge_mode": None,
        "merge_key_meta": "_adata_id_",
        "merge_key_state": None,
        "state_merge_overlap": 0
    }

    if state_table_path is None or (not os.path.exists(state_table_path)):
        merged = obs_df.copy()
        return merged, audit

    st = pd.read_csv(state_table_path)
    audit["state_table_available"] = True
    audit["state_table_rows"] = int(st.shape[0])

    state_key = infer_state_id_col(st, requested=state_id_col)
    audit["merge_key_state"] = state_key

    st = st.copy()
    st[state_key] = st[state_key].astype(str)

    merged = obs_df.merge(
        st,
        left_on="_adata_id_",
        right_on=state_key,
        how="left",
        suffixes=("", "_state")
    )
    audit["merge_mode"] = "id"

    overlap = merged[state_key].notna().sum() if state_key in merged.columns else 0
    audit["state_merge_overlap"] = int(overlap)

    return merged, audit


# -----------------------------------------------------------------------------
# Column inference
# -----------------------------------------------------------------------------

def infer_state_label_col(df: pd.DataFrame, requested: Optional[str] = None) -> Optional[str]:
    if requested is not None and requested in df.columns:
        return requested

    candidates = [
        "state_group",
        "region_manual_final",
        "region_refined",
        "region_auto",
        "state",
        "label",
        "state_label",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_time_col(df: pd.DataFrame, requested: Optional[str] = None) -> Optional[str]:
    if requested is not None and requested in df.columns:
        return requested

    candidates = ["timepoint", "time", "day", "stage"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_prob_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    out = {
        "core": pick([
            "core_probability", "core_prob", "prob_core", "p_core",
            "lesion_core_like_probability", "lesion-core-like_probability"
        ]),
        "peri": pick([
            "peri_probability", "peri_prob", "prob_peri", "p_peri",
            "peri_infarct_probability", "peri-infarct_probability"
        ]),
        "remote": pick([
            "remote_probability", "remote_prob", "prob_remote", "p_remote",
            "remote_like_probability", "remote-like_probability"
        ]),
        "repair": pick([
            "repair_score", "repair_probability", "repair_prob", "prob_repair"
        ])
    }
    return out


def parse_targets(targets_str: str, prob_cols: Dict[str, Optional[str]]) -> List[Tuple[str, str]]:
    """
    Return list of (short_key, actual_column_name)
    """
    out = []
    for tok in [x.strip() for x in targets_str.split(",") if x.strip()]:
        low = tok.lower()
        if low in prob_cols and prob_cols[low] is not None:
            out.append((low, prob_cols[low]))
        elif tok in prob_cols and prob_cols[tok] is not None:
            out.append((tok, prob_cols[tok]))
        else:
            # treat as direct column name if present later
            out.append((tok, tok))
    return out


# -----------------------------------------------------------------------------
# Latent coordinate inference
# -----------------------------------------------------------------------------

def infer_latent_coords(
    adata,
    force_obsm_key: Optional[str] = None,
    obs_x_col: Optional[str] = None,
    obs_y_col: Optional[str] = None,
) -> Tuple[np.ndarray, Dict]:
    audit = {
        "mode": None,
        "obsm_key": None,
        "n_dim": None,
        "forced_obsm_priority": bool(force_obsm_key),
        "obs_x_col": None,
        "obs_y_col": None
    }

    # 1) forced obsm
    if force_obsm_key is not None:
        if force_obsm_key not in adata.obsm.keys():
            raise ValueError(f"--force_obsm_key={force_obsm_key} not found in adata.obsm")
        X = np.asarray(adata.obsm[force_obsm_key])
        if X.ndim != 2:
            raise ValueError(f"adata.obsm['{force_obsm_key}'] is not 2D")
        audit["obsm_key"] = force_obsm_key
        audit["n_dim"] = int(X.shape[1])

        if X.shape[1] == 2:
            coords = X.astype(float)
            audit["mode"] = "obsm_forced_2d"
            return coords, audit
        else:
            pca = PCA(n_components=2, random_state=0)
            coords = pca.fit_transform(X.astype(float))
            audit["mode"] = "obsm_forced_pca"
            return coords, audit

    # 2) explicit obs cols
    if (obs_x_col is not None and obs_x_col in adata.obs.columns and
        obs_y_col is not None and obs_y_col in adata.obs.columns):
        coords = np.c_[
            pd.to_numeric(adata.obs[obs_x_col], errors="coerce").values,
            pd.to_numeric(adata.obs[obs_y_col], errors="coerce").values
        ]
        audit["mode"] = "obs_explicit"
        audit["obs_x_col"] = obs_x_col
        audit["obs_y_col"] = obs_y_col
        return coords, audit

    # 3) existing 2D obsm
    preferred_obsm = [
        "X_umap", "X_draw_graph_fa", "X_tsne", "X_diffmap",
        "X_nicheformer", "X_scgpt", "X_pca"
    ]
    for key in preferred_obsm:
        if key in adata.obsm.keys():
            X = np.asarray(adata.obsm[key])
            if X.ndim == 2:
                audit["obsm_key"] = key
                audit["n_dim"] = int(X.shape[1])
                if X.shape[1] == 2:
                    audit["mode"] = "obsm_auto_2d"
                    return X.astype(float), audit
                elif X.shape[1] > 2:
                    pca = PCA(n_components=2, random_state=0)
                    coords = pca.fit_transform(X.astype(float))
                    audit["mode"] = "obsm_auto_pca"
                    return coords, audit

    # 4) obs latent columns
    obs_pairs = [
        ("latent_x", "latent_y"),
        ("latent1", "latent2"),
        ("latent_1", "latent_2"),
        ("umap1", "umap2"),
        ("UMAP1", "UMAP2"),
        ("pca1", "pca2"),
        ("PC1", "PC2"),
    ]
    for xcol, ycol in obs_pairs:
        if xcol in adata.obs.columns and ycol in adata.obs.columns:
            coords = np.c_[
                pd.to_numeric(adata.obs[xcol], errors="coerce").values,
                pd.to_numeric(adata.obs[ycol], errors="coerce").values
            ]
            audit["mode"] = "obs_auto"
            audit["obs_x_col"] = xcol
            audit["obs_y_col"] = ycol
            return coords, audit

    raise ValueError("Could not infer latent 2D coordinates from obsm or obs columns.")


# -----------------------------------------------------------------------------
# Smoothing / landscape estimation
# -----------------------------------------------------------------------------

def auto_bandwidth(coords: np.ndarray, k: int = 15) -> float:
    n = coords.shape[0]
    kk = min(max(5, k), max(5, n - 1))
    nbrs = NearestNeighbors(n_neighbors=kk)
    nbrs.fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    # use distance to kth neighbor
    hk = np.median(dists[:, -1])
    if not np.isfinite(hk) or hk <= 0:
        xstd = np.std(coords[:, 0])
        ystd = np.std(coords[:, 1])
        hk = 0.15 * max(xstd, ystd, 1e-3)
    return float(hk)


def build_grid(coords: np.ndarray, grid_n: int = 220, pad_frac: float = 0.08):
    x = coords[:, 0]
    y = coords[:, 1]
    xmin, xmax = np.nanpercentile(x, [0.5, 99.5])
    ymin, ymax = np.nanpercentile(y, [0.5, 99.5])

    xr = xmax - xmin
    yr = ymax - ymin
    if xr <= 0:
        xr = 1.0
    if yr <= 0:
        yr = 1.0

    xmin -= pad_frac * xr
    xmax += pad_frac * xr
    ymin -= pad_frac * yr
    ymax += pad_frac * yr

    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    XX, YY = np.meshgrid(gx, gy)
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    return gx, gy, XX, YY, grid_points


def nadaraya_watson_grid(
    coords: np.ndarray,
    values: np.ndarray,
    grid_points: np.ndarray,
    bandwidth: float,
    radius_mult: float = 3.0,
    chunk_size: int = 2000,
    fallback_k: int = 32,
) -> np.ndarray:
    tree = cKDTree(coords)
    radius = bandwidth * radius_mult
    n_grid = grid_points.shape[0]
    out = np.full(n_grid, np.nan, dtype=float)

    values = values.astype(float)
    valid = np.isfinite(values)
    coords_valid = coords[valid]
    values_valid = values[valid]

    tree = cKDTree(coords_valid)

    for start in range(0, n_grid, chunk_size):
        end = min(start + chunk_size, n_grid)
        pts = grid_points[start:end]
        neigh = tree.query_ball_point(pts, r=radius)

        for i, idxs in enumerate(neigh):
            p = pts[i]
            if len(idxs) == 0:
                k = min(fallback_k, coords_valid.shape[0])
                d, idx = tree.query(p, k=k)
                d = np.atleast_1d(d)
                idx = np.atleast_1d(idx)
                ww = np.exp(-0.5 * (d / bandwidth) ** 2)
                vv = values_valid[idx]
            else:
                arr = np.asarray(idxs, dtype=int)
                cc = coords_valid[arr]
                d = np.sqrt(((cc - p) ** 2).sum(axis=1))
                ww = np.exp(-0.5 * (d / bandwidth) ** 2)
                vv = values_valid[arr]

            s = np.sum(ww)
            if s > 0:
                out[start + i] = np.sum(ww * vv) / s

    return out


def find_peaks_on_grid(
    Z: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    smooth_sigma: float = 1.0,
    max_peaks: int = 8,
    q_thresh: float = 0.92,
    size: int = 9
) -> pd.DataFrame:
    z = np.array(Z, dtype=float)
    z[np.isnan(z)] = np.nanmedian(z)
    zs = gaussian_filter(z, sigma=smooth_sigma)

    mx = maximum_filter(zs, size=size)
    mask = (zs == mx)
    thr = np.nanquantile(zs, q_thresh)
    mask &= (zs >= thr)

    rows, cols = np.where(mask)
    vals = zs[rows, cols]

    if len(vals) == 0:
        return pd.DataFrame(columns=["peak_rank", "x", "y", "value"])

    order = np.argsort(vals)[::-1][:max_peaks]
    rows = rows[order]
    cols = cols[order]
    vals = vals[order]

    peaks = pd.DataFrame({
        "peak_rank": np.arange(1, len(vals) + 1),
        "x": gx[cols],
        "y": gy[rows],
        "value": vals
    })
    return peaks


# -----------------------------------------------------------------------------
# Plot styling
# -----------------------------------------------------------------------------

STATE_COLORS = {
    "core": "#ff5a5a",
    "peri": "#ffb000",
    "remote": "#53b8ff"
}


def infer_state_colors(labels: pd.Series) -> np.ndarray:
    lab = labels.astype(str).str.lower().fillna("nan")
    out = np.full(len(lab), "#aaaaaa", dtype=object)

    core_mask = lab.str.contains("core|lesion")
    peri_mask = lab.str.contains("peri")
    remote_mask = lab.str.contains("remote")

    out[core_mask.values] = STATE_COLORS["core"]
    out[peri_mask.values] = STATE_COLORS["peri"]
    out[remote_mask.values] = STATE_COLORS["remote"]
    return out


def style_target(values: np.ndarray):
    vv = values[np.isfinite(values)]
    if len(vv) == 0:
        return {
            "cmap": "magma",
            "norm": Normalize(vmin=0, vmax=1),
            "vmin": 0, "vmax": 1
        }

    if is_prob_like(vv):
        return {
            "cmap": "magma",
            "norm": Normalize(vmin=0.0, vmax=1.0),
            "vmin": 0.0, "vmax": 1.0
        }

    vmin = np.nanpercentile(vv, 2)
    vmax = np.nanpercentile(vv, 98)

    if vmin < 0 < vmax:
        lim = max(abs(vmin), abs(vmax))
        return {
            "cmap": "coolwarm",
            "norm": TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
            "vmin": -lim, "vmax": lim
        }
    else:
        return {
            "cmap": "viridis",
            "norm": Normalize(vmin=vmin, vmax=vmax),
            "vmin": vmin, "vmax": vmax
        }


def apply_global_style(dark=True):
    if dark:
        plt.style.use("dark_background")
        plt.rcParams.update({
            "figure.facecolor": "black",
            "axes.facecolor": "black",
            "savefig.facecolor": "black",
            "axes.edgecolor": "#bbbbbb",
            "axes.labelcolor": "white",
            "xtick.color": "#dddddd",
            "ytick.color": "#dddddd",
            "text.color": "white",
            "grid.color": "#444444",
            "font.size": 10,
            "font.family": "DejaVu Sans",
        })
    else:
        plt.style.use("default")
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "font.family": "DejaVu Sans",
        })


# -----------------------------------------------------------------------------
# Plot functions
# -----------------------------------------------------------------------------

def plot_landscape_2d(
    ax,
    gx,
    gy,
    Z,
    coords=None,
    point_colors=None,
    style=None,
    title="",
    alpha_points=0.28,
    s_points=4,
):
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[gx.min(), gx.max(), gy.min(), gy.max()],
        cmap=style["cmap"],
        norm=style["norm"],
        aspect="auto"
    )

    zz = np.array(Z, dtype=float)
    finite = np.isfinite(zz)
    if finite.sum() > 0:
        q = np.quantile(zz[finite], [0.20, 0.40, 0.60, 0.80, 0.92])
        q = np.unique(q)
        if len(q) >= 2:
            ax.contour(
                gx, gy, Z,
                levels=q,
                colors="white",
                linewidths=0.45,
                alpha=0.25
            )

    if coords is not None:
        if point_colors is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=s_points, c="white", alpha=alpha_points, linewidths=0)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=s_points, c=point_colors, alpha=alpha_points, linewidths=0)

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Latent coordinate 1")
    ax.set_ylabel("Latent coordinate 2")
    ax.grid(alpha=0.15)
    return im


def plot_landscape_3d(
    ax,
    XX, YY, Z,
    style,
    title=""
):
    surf = ax.plot_surface(
        XX, YY, Z,
        cmap=style["cmap"],
        norm=style["norm"],
        linewidth=0,
        antialiased=True,
        alpha=0.98
    )
    try:
        ax.contour(
            XX, YY, Z,
            levels=8,
            zdir='z',
            offset=np.nanmin(Z),
            colors='white',
            linewidths=0.35,
            alpha=0.25
        )
    except Exception:
        pass

    ax.set_title(title, fontsize=12, weight="bold", pad=10)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Smoothed value")
    return surf


def plot_ridge_peak_map(
    ax,
    gx, gy, Z,
    peaks_df,
    coords=None,
    point_colors=None,
    style=None,
    title=""
):
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[gx.min(), gx.max(), gy.min(), gy.max()],
        cmap=style["cmap"],
        norm=style["norm"],
        aspect="auto"
    )

    zz = np.array(Z, dtype=float)
    finite = np.isfinite(zz)
    if finite.sum() > 0:
        q = np.quantile(zz[finite], [0.75, 0.85, 0.92, 0.97])
        q = np.unique(q)
        if len(q) >= 2:
            ax.contour(
                gx, gy, Z,
                levels=q,
                colors=["#88ccff", "#ffee88", "#ff9966", "#ffffff"][:len(q)],
                linewidths=[0.6, 0.8, 1.0, 1.2][:len(q)],
                alpha=0.9
            )

    if coords is not None:
        if point_colors is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=2, c="white", alpha=0.08, linewidths=0)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=2, c=point_colors, alpha=0.08, linewidths=0)

    if peaks_df is not None and peaks_df.shape[0] > 0:
        ax.scatter(
            peaks_df["x"], peaks_df["y"],
            marker="*",
            s=95,
            c="#ffea00",
            edgecolors="black",
            linewidths=0.8,
            zorder=10
        )
        for _, r in peaks_df.iterrows():
            ax.text(
                r["x"], r["y"], str(int(r["peak_rank"])),
                fontsize=8, weight="bold",
                ha="left", va="bottom",
                color="white"
            )

    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel("Latent coordinate 1")
    ax.set_ylabel("Latent coordinate 2")
    ax.grid(alpha=0.15)
    return im


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step77 latent state-probability landscape")
    parser.add_argument("--h5ad", required=True, help="Input h5ad")
    parser.add_argument("--state_table", required=True, help="Step64/66 etc state probability table")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--adata_id_col", default=None, help="Optional adata.obs ID col")
    parser.add_argument("--state_id_col", default=None, help="Optional state table ID col")

    parser.add_argument("--state_label_col", default=None, help="e.g. state_group")
    parser.add_argument("--time_col", default=None, help="e.g. timepoint")

    parser.add_argument("--force_obsm_key", default=None, help="Force a specific adata.obsm key, e.g. X_nicheformer")
    parser.add_argument("--obs_x_col", default=None, help="Optional explicit obs x col")
    parser.add_argument("--obs_y_col", default=None, help="Optional explicit obs y col")

    parser.add_argument("--targets", default="core,peri,remote,repair",
                        help="Comma-separated targets among core/peri/remote/repair or direct column names")

    parser.add_argument("--grid_n", type=int, default=220, help="Grid size")
    parser.add_argument("--bandwidth", default="auto", help="Kernel bandwidth or 'auto'")
    parser.add_argument("--radius_mult", type=float, default=3.0, help="Neighborhood radius = bandwidth * radius_mult")
    parser.add_argument("--dark_style", action="store_true", help="Use dark plotting style")

    parser.add_argument("--peak_q", type=float, default=0.92, help="Peak threshold quantile")
    parser.add_argument("--max_peaks", type=int, default=8, help="Max peak count per landscape")
    parser.add_argument("--show_points", action="store_true", help="Overlay all points")
    parser.add_argument("--dpi", type=int, default=220)

    args = parser.parse_args()
    ensure_dir(args.outdir)
    apply_global_style(dark=args.dark_style)

    print("=" * 98)
    print("Step77 | Latent state-probability landscape (Scheme B2)")
    print("=" * 98)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={args.outdir}")

    # -------------------------------------------------------------------------
    # Read inputs
    # -------------------------------------------------------------------------
    adata = ad.read_h5ad(args.h5ad)
    merged, state_merge_audit = merge_state_table(
        adata,
        args.state_table,
        adata_id_col=args.adata_id_col,
        state_id_col=args.state_id_col
    )

    latent_coords, latent_audit = infer_latent_coords(
        adata,
        force_obsm_key=args.force_obsm_key,
        obs_x_col=args.obs_x_col,
        obs_y_col=args.obs_y_col,
    )
    merged["latent_1"] = latent_coords[:, 0]
    merged["latent_2"] = latent_coords[:, 1]

    state_label_col = infer_state_label_col(merged, requested=args.state_label_col)
    time_col = infer_time_col(merged, requested=args.time_col)
    prob_cols = infer_prob_columns(merged)

    requested_targets = parse_targets(args.targets, prob_cols)

    # verify target columns
    valid_targets = []
    for short_key, col in requested_targets:
        if col in merged.columns:
            vv = pd.to_numeric(merged[col], errors="coerce").values
            if np.isfinite(vv).sum() > 0:
                valid_targets.append((short_key, col))
        else:
            warnings.warn(f"Target column '{col}' not found; skipped.")

    if len(valid_targets) == 0:
        raise ValueError("No valid target columns found for plotting.")

    # bandwith
    coords = merged[["latent_1", "latent_2"]].values.astype(float)
    if isinstance(args.bandwidth, str) and args.bandwidth.lower() == "auto":
        h = auto_bandwidth(coords)
    else:
        h = float(args.bandwidth)

    print(f"bandwidth_used={h:.6f}")
    gx, gy, XX, YY, grid_points = build_grid(coords, grid_n=args.grid_n)

    # point color overlay
    point_colors = None
    if state_label_col is not None:
        point_colors = infer_state_colors(merged[state_label_col])

    # -------------------------------------------------------------------------
    # Smooth each target onto grid
    # -------------------------------------------------------------------------
    grid_df = pd.DataFrame({"latent_1": grid_points[:, 0], "latent_2": grid_points[:, 1]})
    peaks_all = []
    landscapes = {}

    for short_key, col in valid_targets:
        vals = pd.to_numeric(merged[col], errors="coerce").values.astype(float)
        zhat = nadaraya_watson_grid(
            coords=coords,
            values=vals,
            grid_points=grid_points,
            bandwidth=h,
            radius_mult=args.radius_mult,
            chunk_size=2500,
            fallback_k=32
        )
        Z = zhat.reshape(len(gy), len(gx))
        landscapes[short_key] = {
            "column": col,
            "values": vals,
            "grid": Z
        }
        grid_df[col] = zhat

        peaks = find_peaks_on_grid(
            Z, gx, gy,
            smooth_sigma=1.0,
            max_peaks=args.max_peaks,
            q_thresh=args.peak_q,
            size=9
        )
        peaks["target_key"] = short_key
        peaks["target_column"] = col
        peaks_all.append(peaks)

    peaks_df = pd.concat(peaks_all, axis=0, ignore_index=True) if len(peaks_all) > 0 else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Figure 1: 2D atlas
    # -------------------------------------------------------------------------
    n_tgt = len(valid_targets)
    ncols = 2 if n_tgt <= 4 else 4
    nrows = int(math.ceil(n_tgt / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 5.4 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    ims = []
    for ax, (short_key, col) in zip(axes, valid_targets):
        style = style_target(landscapes[short_key]["values"])
        im = plot_landscape_2d(
            ax=ax,
            gx=gx, gy=gy,
            Z=landscapes[short_key]["grid"],
            coords=coords if args.show_points else None,
            point_colors=point_colors if args.show_points else None,
            style=style,
            title=pretty_name(short_key)
        )
        ims.append((im, style, short_key))

    for k in range(len(valid_targets), len(axes)):
        axes[k].axis("off")

    fig.suptitle("StrokeNiche latent state-probability landscapes", fontsize=22, weight="bold", y=0.995)

    # colorbars: one for probability-like, one for score-like if needed
    prob_like_present = any(is_prob_like(landscapes[k]["values"]) for k, _ in valid_targets)
    score_like_present = any(not is_prob_like(landscapes[k]["values"]) for k, _ in valid_targets)

    if prob_like_present:
        cax = fig.add_axes([0.92, 0.56, 0.015, 0.28])
        sm = plt.cm.ScalarMappable(cmap="magma", norm=Normalize(0, 1))
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("State probability")

    if score_like_present:
        # determine common symmetric range across score-like targets
        score_vals = []
        for short_key, _ in valid_targets:
            vv = landscapes[short_key]["values"]
            if not is_prob_like(vv):
                vv = vv[np.isfinite(vv)]
                if len(vv):
                    score_vals.append(vv)
        if len(score_vals):
            score_vals = np.concatenate(score_vals)
            lim = max(abs(np.nanpercentile(score_vals, 2)), abs(np.nanpercentile(score_vals, 98)))
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.28])
            sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim))
            cb = fig.colorbar(sm, cax=cax)
            cb.set_label("Score / mean value")

    fig.text(
        0.5, 0.01,
        "Each panel shows the smoothed conditional landscape in latent space; contours highlight ridge/peak structure.",
        ha="center", fontsize=11
    )

    out_png = os.path.join(args.outdir, "Fig_Step77_StateProbabilityLandscape_2D_Atlas.png")
    out_pdf = os.path.join(args.outdir, "Fig_Step77_StateProbabilityLandscape_2D_Atlas.pdf")
    out_svg = os.path.join(args.outdir, "Fig_Step77_StateProbabilityLandscape_2D_Atlas.svg")
    plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.97])
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2+: per-target 3-panel figure (2D / 3D / peak map)
    # -------------------------------------------------------------------------
    per_target_outputs = []
    for short_key, col in valid_targets:
        vals = landscapes[short_key]["values"]
        Z = landscapes[short_key]["grid"]
        style = style_target(vals)

        peaks_sub = peaks_df.loc[peaks_df["target_key"] == short_key].copy()

        fig = plt.figure(figsize=(18, 5.7))
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = plot_landscape_2d(
            ax=ax1,
            gx=gx, gy=gy, Z=Z,
            coords=coords if args.show_points else None,
            point_colors=point_colors if args.show_points else None,
            style=style,
            title=f"{pretty_name(short_key)} | 2D heatmap"
        )

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        surf = plot_landscape_3d(
            ax=ax2,
            XX=XX, YY=YY, Z=Z,
            style=style,
            title=f"{pretty_name(short_key)} | 3D surface"
        )

        ax3 = fig.add_subplot(1, 3, 3)
        im3 = plot_ridge_peak_map(
            ax=ax3,
            gx=gx, gy=gy, Z=Z,
            peaks_df=peaks_sub,
            coords=coords if args.show_points else None,
            point_colors=point_colors if args.show_points else None,
            style=style,
            title=f"{pretty_name(short_key)} | ridge / peak map"
        )

        cax = fig.add_axes([0.93, 0.16, 0.012, 0.68])
        cb = fig.colorbar(im1, cax=cax)
        cb.set_label(pretty_name(short_key))

        fig.suptitle(f"Step77 | {pretty_name(short_key)} latent landscape", fontsize=18, weight="bold", y=0.99)
        fig.text(
            0.5, 0.01,
            "Landscape estimated by Gaussian-kernel smoothing in latent space; peaks indicate high-probability / high-score basins.",
            ha="center", fontsize=10
        )

        base = f"Fig_Step77_{short_key}_Landscape_2D3DPeaks"
        f_png = os.path.join(args.outdir, base + ".png")
        f_pdf = os.path.join(args.outdir, base + ".pdf")
        f_svg = os.path.join(args.outdir, base + ".svg")
        plt.tight_layout(rect=[0.02, 0.04, 0.92, 0.96])
        fig.savefig(f_png, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(f_pdf, bbox_inches="tight")
        fig.savefig(f_svg, bbox_inches="tight")
        plt.close(fig)

        per_target_outputs.append({
            "target_key": short_key,
            "target_column": col,
            "png": f_png,
            "pdf": f_pdf,
            "svg": f_svg
        })

    # -------------------------------------------------------------------------
    # Write tables
    # -------------------------------------------------------------------------
    grid_csv = os.path.join(args.outdir, "step77_grid_state_probability_landscape.csv")
    peaks_csv = os.path.join(args.outdir, "step77_landscape_peaks.csv")
    merged_small_csv = os.path.join(args.outdir, "step77_per_spot_latent_targets.csv")

    grid_df.to_csv(grid_csv, index=False)
    peaks_df.to_csv(peaks_csv, index=False)

    keep_cols = ["_obs_names_", "_adata_id_", "latent_1", "latent_2"]
    if state_label_col is not None and state_label_col in merged.columns:
        keep_cols.append(state_label_col)
    if time_col is not None and time_col in merged.columns:
        keep_cols.append(time_col)
    for _, col in valid_targets:
        if col in merged.columns:
            keep_cols.append(col)

    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged[keep_cols].to_csv(merged_small_csv, index=False)

    # -------------------------------------------------------------------------
    # JSON report
    # -------------------------------------------------------------------------
    report = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "adata_id_col_used": args.adata_id_col if args.adata_id_col else "_obs_names_",
        "state_merge_audit": state_merge_audit,
        "latent_coordinate_audit": latent_audit,
        "state_label_source": state_label_col,
        "time_col": time_col,
        "prob_cols": prob_cols,
        "targets_requested": args.targets,
        "targets_used": [{"target_key": k, "target_column": c} for k, c in valid_targets],
        "bandwidth_used": safe_float(h),
        "grid_n": int(args.grid_n),
        "radius_mult": safe_float(args.radius_mult),
        "peak_q": safe_float(args.peak_q),
        "max_peaks": int(args.max_peaks),
        "outputs": {
            "atlas_png": out_png,
            "atlas_pdf": out_pdf,
            "atlas_svg": out_svg,
            "grid_csv": grid_csv,
            "peaks_csv": peaks_csv,
            "per_spot_csv": merged_small_csv,
            "per_target_figures": per_target_outputs
        },
        "formula": "f(z) = sum_i K(||z-z_i||/h) * y_i / sum_i K(||z-z_i||/h), where y_i is a state probability or repair score.",
        "interpretation_note": (
            "Step77 constructs smoothed latent-state probability / score landscapes directly in latent space. "
            "High values indicate latent regions enriched for the corresponding state probability or score. "
            "This is a visualization-level conditional landscape, not a physical energy surface and not functional validation."
        )
    }

    report_path = os.path.join(args.outdir, "step77_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(report), f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Atlas: {out_png}")
    print(f"Grid CSV: {grid_csv}")
    print(f"Peaks CSV: {peaks_csv}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()