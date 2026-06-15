#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step77A | Fixed latent state-probability landscape with audit
============================================================

Purpose
-------
Construct target-specific conditional landscapes in latent space:

    P(core-like | z)
    P(peri-infarct | z)
    P(remote-like | z)
    repair_score(z)

Compared with Step77, this corrected version adds:
1. explicit target-column audit
2. model-probability vs label-onehot probability source control
3. duplicate/correlation detection for core/peri/remote probability columns
4. low-density masking to prevent extrapolated polygon artifacts
5. target-specific grid/hash/allclose checks
6. cleaner 2D / 3D / ridge-peak figures

Recommended main run
--------------------
Use X_nicheformer h5ad:

python 77a_state_probability_landscape_fixed_audit.py \
  --h5ad /mnt/h/vir/ST/results/step5_nicheformer/spatial_all_with_nicheformer.h5ad \
  --state_table /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv \
  --outdir /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/latent_state_probability_landscape_77a_X_nicheformer_fixed \
  --force_obsm_key X_nicheformer \
  --state_label_col state_group \
  --time_col timepoint \
  --targets core,peri,remote,repair \
  --prob_source auto \
  --grid_n 240 \
  --bandwidth auto \
  --radius_mult 3.0 \
  --density_mask_q 0.025 \
  --min_neighbors 8 \
  --peak_q 0.94 \
  --max_peaks 6 \
  --dark_style \
  --show_points \
  --dpi 600

Interpretation
--------------
This is a visualization-level conditional state-probability landscape in latent space.
It is not a physical energy surface, not lineage potential, and not functional validation.
"""

import os
import re
import json
import math
import hashlib
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
from scipy.ndimage import maximum_filter, gaussian_filter
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# =============================================================================
# Basic utilities
# =============================================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def to_jsonable(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, pd.DataFrame):
        return x.to_dict(orient="records")
    if isinstance(x, pd.Series):
        return x.to_dict()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [to_jsonable(v) for v in x]
    return x


def value_hash(arr: np.ndarray, ndigits: int = 8) -> str:
    x = np.asarray(arr, dtype=float)
    x = np.nan_to_num(x, nan=-999999.0, posinf=999999.0, neginf=-999999.0)
    x = np.round(x, ndigits)
    return hashlib.sha1(x.tobytes()).hexdigest()[:16]


def pretty_target_name(key: str) -> str:
    mp = {
        "core": "Core probability",
        "peri": "Peri-infarct probability",
        "remote": "Remote-like probability",
        "repair": "Repair score",
    }
    return mp.get(key, key.replace("_", " ").replace("-", " ").title())


def is_prob_like(values: np.ndarray) -> bool:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return False
    return np.nanmin(v) >= -0.02 and np.nanmax(v) <= 1.02


# =============================================================================
# Input merge
# =============================================================================

def read_table_auto(path: str) -> pd.DataFrame:
    if path.endswith(".tsv") or path.endswith(".tsv.gz"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def infer_state_id_col(df: pd.DataFrame, requested: Optional[str] = None) -> str:
    if requested and requested in df.columns:
        return requested

    candidates = [
        "obs_name", "spot_id", "barcode", "cell_id", "cell", "id", "ID",
        "_obs_names_", "Unnamed: 0", "index"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    return df.columns[0]


def merge_state_table(
    adata,
    state_table_path: str,
    adata_id_col: Optional[str] = None,
    state_id_col: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    obs = adata.obs.copy()
    obs["_obs_names_"] = obs.index.astype(str)

    if adata_id_col and adata_id_col in obs.columns:
        obs["_adata_id_"] = obs[adata_id_col].astype(str)
        adata_id_used = adata_id_col
    else:
        obs["_adata_id_"] = obs.index.astype(str)
        adata_id_used = "_obs_names_"

    st = read_table_auto(state_table_path)
    key = infer_state_id_col(st, state_id_col)
    st = st.copy()
    st[key] = st[key].astype(str)

    before = st.shape[0]
    st = st.drop_duplicates(subset=[key], keep="first")
    after = st.shape[0]

    merged = obs.merge(
        st,
        left_on="_adata_id_",
        right_on=key,
        how="left",
        suffixes=("", "_state")
    )

    overlap = int(merged[key].notna().sum()) if key in merged.columns else 0

    audit = {
        "state_table_available": True,
        "state_table_path": state_table_path,
        "state_table_rows_raw": int(before),
        "state_table_rows_after_dedup": int(after),
        "merge_mode": "id",
        "adata_id_col_used": adata_id_used,
        "merge_key_meta": "_adata_id_",
        "merge_key_state": key,
        "state_merge_overlap": overlap,
        "n_obs": int(adata.n_obs),
        "overlap_fraction": float(overlap / max(1, adata.n_obs))
    }
    return merged, audit


# =============================================================================
# Latent coordinate inference
# =============================================================================

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

    if force_obsm_key:
        if force_obsm_key not in adata.obsm.keys():
            raise ValueError(
                f"--force_obsm_key={force_obsm_key} not found. "
                f"Available obsm keys: {list(adata.obsm.keys())}"
            )
        X = np.asarray(adata.obsm[force_obsm_key])
        if X.ndim != 2 or X.shape[0] != adata.n_obs:
            raise ValueError(f"Invalid obsm[{force_obsm_key}] shape: {X.shape}")

        audit["obsm_key"] = force_obsm_key
        audit["n_dim"] = int(X.shape[1])

        if X.shape[1] == 2:
            audit["mode"] = "obsm_forced_2d"
            return X.astype(float), audit

        coords = PCA(n_components=2, random_state=0).fit_transform(X.astype(float))
        audit["mode"] = "obsm_forced_pca"
        return coords, audit

    if obs_x_col and obs_y_col and obs_x_col in adata.obs.columns and obs_y_col in adata.obs.columns:
        coords = np.c_[
            pd.to_numeric(adata.obs[obs_x_col], errors="coerce").values,
            pd.to_numeric(adata.obs[obs_y_col], errors="coerce").values
        ]
        audit["mode"] = "obs_explicit"
        audit["obs_x_col"] = obs_x_col
        audit["obs_y_col"] = obs_y_col
        return coords.astype(float), audit

    preferred = [
        "X_umap", "X_nicheformer", "X_scgpt", "X_pca", "X_diffmap",
        "X_draw_graph_fa", "X_tsne"
    ]
    for k in preferred:
        if k in adata.obsm.keys():
            X = np.asarray(adata.obsm[k])
            audit["obsm_key"] = k
            audit["n_dim"] = int(X.shape[1])
            if X.shape[1] == 2:
                audit["mode"] = "obsm_auto_2d"
                return X.astype(float), audit
            coords = PCA(n_components=2, random_state=0).fit_transform(X.astype(float))
            audit["mode"] = "obsm_auto_pca"
            return coords, audit

    raise ValueError("No latent coordinate source found. Use --force_obsm_key.")


# =============================================================================
# Column inference and target construction
# =============================================================================

def infer_state_label_col(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested

    candidates = [
        "state_group", "region_manual_final", "region_refined", "region_auto",
        "state", "state_label", "label", "class"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_time_col(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested
    for c in ["timepoint", "time", "stage", "day"]:
        if c in df.columns:
            return c
    return None


def infer_prob_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    return {
        "core": pick([
            "core_probability", "core_prob", "prob_core", "p_core",
            "lesion_core_probability", "lesion_core_like_probability",
            "lesion-core-like_probability"
        ]),
        "peri": pick([
            "peri_probability", "peri_prob", "prob_peri", "p_peri",
            "peri_infarct_probability", "peri_infarct_prob",
            "peri-infarct_probability"
        ]),
        "remote": pick([
            "remote_probability", "remote_prob", "prob_remote", "p_remote",
            "remote_like_probability", "remote-like_probability"
        ]),
        "repair": pick([
            "repair_score", "repair_probability", "repair_prob",
            "prob_repair", "repair_state_score"
        ]),
    }


def label_to_onehot_probs(labels: pd.Series) -> pd.DataFrame:
    lab = labels.astype(str).str.lower().fillna("")

    core = lab.str.contains("core|lesion").astype(float)
    peri = lab.str.contains("peri").astype(float)
    remote = lab.str.contains("remote").astype(float)

    total = core + peri + remote
    unknown = total == 0

    # Unknown states receive NaN, not forced zeros, to avoid biasing smoothing.
    core[unknown] = np.nan
    peri[unknown] = np.nan
    remote[unknown] = np.nan

    return pd.DataFrame({
        "__label_onehot_core": core.values,
        "__label_onehot_peri": peri.values,
        "__label_onehot_remote": remote.values,
    })


def parse_targets(targets: str) -> List[str]:
    return [x.strip() for x in targets.split(",") if x.strip()]


def target_column_stats(df: pd.DataFrame, cols: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for k, c in cols.items():
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce").values.astype(float)
        finite = np.isfinite(v)
        rows.append({
            "target": k,
            "column": c,
            "n_finite": int(finite.sum()),
            "nan_fraction": float(1 - finite.sum() / max(1, len(v))),
            "min": float(np.nanmin(v)) if finite.sum() else np.nan,
            "p01": float(np.nanpercentile(v, 1)) if finite.sum() else np.nan,
            "p05": float(np.nanpercentile(v, 5)) if finite.sum() else np.nan,
            "mean": float(np.nanmean(v)) if finite.sum() else np.nan,
            "std": float(np.nanstd(v)) if finite.sum() else np.nan,
            "p50": float(np.nanpercentile(v, 50)) if finite.sum() else np.nan,
            "p95": float(np.nanpercentile(v, 95)) if finite.sum() else np.nan,
            "p99": float(np.nanpercentile(v, 99)) if finite.sum() else np.nan,
            "max": float(np.nanmax(v)) if finite.sum() else np.nan,
            "hash": value_hash(v),
            "prob_like": bool(is_prob_like(v))
        })
    return pd.DataFrame(rows)


def build_target_values(
    merged: pd.DataFrame,
    targets: List[str],
    prob_cols: Dict[str, Optional[str]],
    state_label_col: Optional[str],
    prob_source: str = "auto",
    duplicate_corr_threshold: float = 0.995,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Returns:
      target_df: columns are canonical target names.
      target_col_map: canonical target -> original/generated column used.
      target_source: canonical target -> source label.
    """
    prob_source = prob_source.lower()
    assert prob_source in ["auto", "model", "label_onehot"]

    work = merged.copy()

    onehot_cols = {}
    if state_label_col and state_label_col in work.columns:
        oh = label_to_onehot_probs(work[state_label_col])
        for c in oh.columns:
            work[c] = oh[c].values
        onehot_cols = {
            "core": "__label_onehot_core",
            "peri": "__label_onehot_peri",
            "remote": "__label_onehot_remote",
        }

    model_state_cols = {k: prob_cols.get(k) for k in ["core", "peri", "remote"] if prob_cols.get(k) in work.columns}

    model_state_duplicate = False
    model_state_corr = None
    if len(model_state_cols) >= 2:
        mat = pd.DataFrame({
            k: pd.to_numeric(work[c], errors="coerce")
            for k, c in model_state_cols.items()
        })
        model_state_corr = mat.corr(method="spearman")
        vals = model_state_corr.values
        off = vals[~np.eye(vals.shape[0], dtype=bool)]
        if len(off) > 0 and np.nanmin(off) >= duplicate_corr_threshold:
            model_state_duplicate = True

    use_label_for_states = False
    if prob_source == "label_onehot":
        use_label_for_states = True
    elif prob_source == "auto":
        # If model probs are missing or highly duplicate, use label-onehot smoothing for state probabilities.
        if len(model_state_cols) < 3:
            use_label_for_states = True
        elif model_state_duplicate:
            use_label_for_states = True

    target_df = pd.DataFrame(index=work.index)
    target_col_map = {}
    target_source = {}

    for t in targets:
        if t in ["core", "peri", "remote"]:
            if use_label_for_states:
                if t not in onehot_cols:
                    raise ValueError(
                        f"Cannot derive label one-hot for {t}; state_label_col is missing or unusable."
                    )
                col = onehot_cols[t]
                target_df[t] = pd.to_numeric(work[col], errors="coerce")
                target_col_map[t] = col
                target_source[t] = "label_onehot_smoothed"
            else:
                col = prob_cols.get(t)
                if not col or col not in work.columns:
                    raise ValueError(f"Model probability column for {t} not found.")
                target_df[t] = pd.to_numeric(work[col], errors="coerce")
                target_col_map[t] = col
                target_source[t] = "model_probability"

        elif t == "repair":
            col = prob_cols.get("repair")
            if not col or col not in work.columns:
                warnings.warn("repair target requested but repair_score/probability column not found; skipped.")
                continue
            target_df[t] = pd.to_numeric(work[col], errors="coerce")
            target_col_map[t] = col
            target_source[t] = "model_score"

        else:
            # Direct column target.
            if t not in work.columns:
                warnings.warn(f"Direct target column {t} not found; skipped.")
                continue
            target_df[t] = pd.to_numeric(work[t], errors="coerce")
            target_col_map[t] = t
            target_source[t] = "direct_column"

    audit = {
        "prob_source_requested": prob_source,
        "use_label_for_core_peri_remote": bool(use_label_for_states),
        "model_state_duplicate_detected": bool(model_state_duplicate),
        "duplicate_corr_threshold": float(duplicate_corr_threshold),
        "model_state_corr": None if model_state_corr is None else model_state_corr.to_dict(),
        "target_col_map": target_col_map,
        "target_source": target_source,
    }

    return target_df, target_col_map, target_source, audit


# =============================================================================
# Landscape smoothing
# =============================================================================

def auto_bandwidth(coords: np.ndarray, k: int = 20) -> float:
    n = coords.shape[0]
    k = min(max(5, k), max(5, n - 1))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(coords)
    d, _ = nn.kneighbors(coords)
    h = np.nanmedian(d[:, -1])
    if not np.isfinite(h) or h <= 0:
        h = 0.15 * max(np.nanstd(coords[:, 0]), np.nanstd(coords[:, 1]), 1e-3)
    return float(h)


def build_grid(coords: np.ndarray, grid_n: int = 240, pad_frac: float = 0.07):
    x = coords[:, 0]
    y = coords[:, 1]

    xmin, xmax = np.nanpercentile(x, [0.5, 99.5])
    ymin, ymax = np.nanpercentile(y, [0.5, 99.5])

    xr = max(xmax - xmin, 1e-6)
    yr = max(ymax - ymin, 1e-6)

    xmin -= xr * pad_frac
    xmax += xr * pad_frac
    ymin -= yr * pad_frac
    ymax += yr * pad_frac

    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    XX, YY = np.meshgrid(gx, gy)
    GP = np.c_[XX.ravel(), YY.ravel()]
    return gx, gy, XX, YY, GP


def compute_density_and_neighbors(
    coords: np.ndarray,
    grid_points: np.ndarray,
    bandwidth: float,
    radius_mult: float = 3.0,
    chunk_size: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(coords)
    radius = bandwidth * radius_mult
    density = np.zeros(grid_points.shape[0], dtype=float)
    n_neigh = np.zeros(grid_points.shape[0], dtype=int)

    for start in range(0, grid_points.shape[0], chunk_size):
        end = min(start + chunk_size, grid_points.shape[0])
        pts = grid_points[start:end]
        neigh = tree.query_ball_point(pts, r=radius)

        for i, idxs in enumerate(neigh):
            if len(idxs) == 0:
                density[start + i] = 0.0
                n_neigh[start + i] = 0
                continue
            arr = np.asarray(idxs, dtype=int)
            d = np.sqrt(((coords[arr] - pts[i]) ** 2).sum(axis=1))
            w = np.exp(-0.5 * (d / bandwidth) ** 2)
            density[start + i] = float(np.sum(w))
            n_neigh[start + i] = int(len(arr))

    return density, n_neigh


def kernel_smooth_no_extrapolate(
    coords: np.ndarray,
    values: np.ndarray,
    grid_points: np.ndarray,
    bandwidth: float,
    radius_mult: float = 3.0,
    chunk_size: int = 2000,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values) & np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    coords_v = coords[valid]
    vals_v = values[valid]

    tree = cKDTree(coords_v)
    radius = bandwidth * radius_mult
    out = np.full(grid_points.shape[0], np.nan, dtype=float)

    for start in range(0, grid_points.shape[0], chunk_size):
        end = min(start + chunk_size, grid_points.shape[0])
        pts = grid_points[start:end]
        neigh = tree.query_ball_point(pts, r=radius)

        for i, idxs in enumerate(neigh):
            if len(idxs) == 0:
                continue
            arr = np.asarray(idxs, dtype=int)
            d = np.sqrt(((coords_v[arr] - pts[i]) ** 2).sum(axis=1))
            w = np.exp(-0.5 * (d / bandwidth) ** 2)
            s = np.sum(w)
            if s <= 0:
                continue
            out[start + i] = float(np.sum(w * vals_v[arr]) / s)

    return out


def apply_density_mask(
    z_flat: np.ndarray,
    density: np.ndarray,
    n_neigh: np.ndarray,
    density_mask_q: float = 0.025,
    min_neighbors: int = 8,
) -> Tuple[np.ndarray, Dict]:
    z = z_flat.copy()
    positive = density > 0

    if positive.sum() == 0:
        z[:] = np.nan
        return z, {
            "density_threshold": np.nan,
            "density_mask_q": float(density_mask_q),
            "min_neighbors": int(min_neighbors),
            "n_grid_kept": 0,
            "n_grid_total": int(len(z))
        }

    thr = float(np.nanquantile(density[positive], density_mask_q))
    keep = (density >= thr) & (n_neigh >= min_neighbors)
    z[~keep] = np.nan

    audit = {
        "density_threshold": thr,
        "density_mask_q": float(density_mask_q),
        "min_neighbors": int(min_neighbors),
        "n_grid_kept": int(np.isfinite(z).sum()),
        "n_grid_total": int(len(z)),
        "kept_fraction": float(np.isfinite(z).sum() / max(1, len(z)))
    }
    return z, audit


def find_peaks(
    Z: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    q: float = 0.94,
    max_peaks: int = 6,
    filter_size: int = 11,
) -> pd.DataFrame:
    z = np.asarray(Z, dtype=float)
    finite = np.isfinite(z)

    if finite.sum() == 0:
        return pd.DataFrame(columns=["peak_rank", "x", "y", "value"])

    fill = np.nanmin(z[finite])
    z2 = z.copy()
    z2[~finite] = fill
    z2 = gaussian_filter(z2, sigma=0.8)

    mx = maximum_filter(z2, size=filter_size)
    thr = np.nanquantile(z2[finite], q)
    mask = (z2 == mx) & (z2 >= thr) & finite

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return pd.DataFrame(columns=["peak_rank", "x", "y", "value"])

    vals = z2[rows, cols]
    order = np.argsort(vals)[::-1][:max_peaks]

    out = pd.DataFrame({
        "peak_rank": np.arange(1, len(order) + 1),
        "x": gx[cols[order]],
        "y": gy[rows[order]],
        "value": vals[order]
    })
    return out


# =============================================================================
# Plotting
# =============================================================================

STATE_COLORS = {
    "core": "#ff5a5a",
    "peri": "#ffb000",
    "remote": "#50bfff",
    "other": "#aaaaaa",
}


def get_point_colors(labels: Optional[pd.Series]) -> Optional[np.ndarray]:
    if labels is None:
        return None
    lab = labels.astype(str).str.lower()
    colors = np.full(len(lab), STATE_COLORS["other"], dtype=object)
    colors[lab.str.contains("core|lesion").values] = STATE_COLORS["core"]
    colors[lab.str.contains("peri").values] = STATE_COLORS["peri"]
    colors[lab.str.contains("remote").values] = STATE_COLORS["remote"]
    return colors


def setup_style(dark: bool):
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
            "grid.color": "#333333",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        })
    else:
        plt.style.use("default")
        plt.rcParams.update({
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        })


def copy_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((0, 0, 0, 0))
    return cmap


def target_style(target: str, values: np.ndarray):
    v = np.asarray(values, dtype=float)
    if target in ["core", "peri", "remote"] or is_prob_like(v):
        return copy_cmap("magma"), Normalize(0, 1), "probability"

    finite = v[np.isfinite(v)]
    if len(finite) == 0:
        return copy_cmap("viridis"), Normalize(0, 1), "score"

    p2, p98 = np.nanpercentile(finite, [2, 98])
    if p2 < 0 < p98:
        lim = max(abs(p2), abs(p98))
        return copy_cmap("coolwarm"), TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim), "score"
    return copy_cmap("magma"), Normalize(p2, p98), "score"


def plot_2d(ax, gx, gy, Z, coords=None, point_colors=None, title="", cmap=None, norm=None, show_points=True):
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[gx.min(), gx.max(), gy.min(), gy.max()],
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="bilinear",
    )

    finite = np.isfinite(Z)
    if finite.sum() > 0:
        vals = Z[finite]
        levels = np.unique(np.nanquantile(vals, [0.30, 0.50, 0.70, 0.85, 0.94]))
        if len(levels) >= 2:
            ax.contour(
                gx, gy, Z,
                levels=levels,
                colors="white",
                linewidths=0.45,
                alpha=0.32
            )

    if show_points and coords is not None:
        if point_colors is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=3, c="#60c4ff", alpha=0.22, linewidths=0)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=3, c=point_colors, alpha=0.22, linewidths=0)

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Latent coordinate 1")
    ax.set_ylabel("Latent coordinate 2")
    ax.grid(alpha=0.16)
    return im


def plot_ridge(ax, gx, gy, Z, peaks, coords=None, point_colors=None, title="", cmap=None, norm=None):
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[gx.min(), gx.max(), gy.min(), gy.max()],
        cmap=cmap,
        norm=norm,
        aspect="auto",
        interpolation="bilinear",
    )

    finite = np.isfinite(Z)
    if finite.sum() > 0:
        vals = Z[finite]
        levels = np.unique(np.nanquantile(vals, [0.70, 0.82, 0.90, 0.96]))
        colors = ["#55ccff", "#ffee88", "#ff9966", "#ffffff"]
        if len(levels) >= 2:
            ax.contour(
                gx, gy, Z,
                levels=levels,
                colors=colors[:len(levels)],
                linewidths=[0.65, 0.85, 1.05, 1.25][:len(levels)],
                alpha=0.9
            )

    if coords is not None:
        if point_colors is None:
            ax.scatter(coords[:, 0], coords[:, 1], s=2, c="#60c4ff", alpha=0.10, linewidths=0)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], s=2, c=point_colors, alpha=0.10, linewidths=0)

    if peaks is not None and len(peaks) > 0:
        ax.scatter(
            peaks["x"], peaks["y"],
            s=120,
            marker="*",
            c="#ffe600",
            edgecolors="black",
            linewidths=0.8,
            zorder=20
        )
        for _, r in peaks.iterrows():
            ax.text(
                r["x"], r["y"], str(int(r["peak_rank"])),
                fontsize=8,
                weight="bold",
                color="white",
                ha="left",
                va="bottom"
            )

    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("Latent coordinate 1")
    ax.set_ylabel("Latent coordinate 2")
    ax.grid(alpha=0.16)
    return im


def plot_3d(ax, XX, YY, Z, cmap, norm, title=""):
    z = np.asarray(Z, dtype=float)
    surf = ax.plot_surface(
        XX, YY, z,
        cmap=cmap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        alpha=0.98
    )
    ax.set_title(title, fontsize=13, weight="bold", pad=8)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Smoothed value")
    return surf


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--state_table", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--adata_id_col", default=None)
    ap.add_argument("--state_id_col", default=None)
    ap.add_argument("--force_obsm_key", default=None)
    ap.add_argument("--obs_x_col", default=None)
    ap.add_argument("--obs_y_col", default=None)

    ap.add_argument("--state_label_col", default=None)
    ap.add_argument("--time_col", default=None)
    ap.add_argument("--targets", default="core,peri,remote,repair")

    ap.add_argument("--prob_source", default="auto", choices=["auto", "model", "label_onehot"])
    ap.add_argument("--duplicate_corr_threshold", type=float, default=0.995)

    ap.add_argument("--grid_n", type=int, default=240)
    ap.add_argument("--bandwidth", default="auto")
    ap.add_argument("--radius_mult", type=float, default=3.0)
    ap.add_argument("--density_mask_q", type=float, default=0.025)
    ap.add_argument("--min_neighbors", type=int, default=8)
    ap.add_argument("--peak_q", type=float, default=0.94)
    ap.add_argument("--max_peaks", type=int, default=6)

    ap.add_argument("--dark_style", action="store_true")
    ap.add_argument("--show_points", action="store_true")
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    ensure_dir(args.outdir)
    setup_style(args.dark_style)

    print("=" * 100)
    print("Step77A | Fixed latent state-probability landscape with audit")
    print("=" * 100)
    print("h5ad:", args.h5ad)
    print("state_table:", args.state_table)
    print("outdir:", args.outdir)

    # -------------------------------------------------------------------------
    # Load and merge
    # -------------------------------------------------------------------------
    adata = ad.read_h5ad(args.h5ad)
    merged, merge_audit = merge_state_table(
        adata,
        args.state_table,
        adata_id_col=args.adata_id_col,
        state_id_col=args.state_id_col
    )

    coords, coord_audit = infer_latent_coords(
        adata,
        force_obsm_key=args.force_obsm_key,
        obs_x_col=args.obs_x_col,
        obs_y_col=args.obs_y_col
    )

    finite_coord = np.isfinite(coords[:, 0]) & np.isfinite(coords[:, 1])
    if finite_coord.sum() < len(finite_coord):
        warnings.warn(f"Removing {len(finite_coord) - finite_coord.sum()} rows with non-finite latent coordinates.")

    merged = merged.loc[finite_coord].reset_index(drop=True)
    coords = coords[finite_coord]

    merged["latent_1"] = coords[:, 0]
    merged["latent_2"] = coords[:, 1]

    state_label_col = infer_state_label_col(merged, args.state_label_col)
    time_col = infer_time_col(merged, args.time_col)
    prob_cols = infer_prob_columns(merged)

    targets = parse_targets(args.targets)

    target_df, target_col_map, target_source, target_source_audit = build_target_values(
        merged,
        targets=targets,
        prob_cols=prob_cols,
        state_label_col=state_label_col,
        prob_source=args.prob_source,
        duplicate_corr_threshold=args.duplicate_corr_threshold
    )

    if target_df.shape[1] == 0:
        raise ValueError("No valid targets available after target construction.")

    # -------------------------------------------------------------------------
    # Target audit before smoothing
    # -------------------------------------------------------------------------
    target_stats = []
    for t in target_df.columns:
        v = target_df[t].values.astype(float)
        finite = np.isfinite(v)
        target_stats.append({
            "target": t,
            "source_column": target_col_map.get(t, ""),
            "source_type": target_source.get(t, ""),
            "n_finite": int(finite.sum()),
            "nan_fraction": float(1 - finite.sum() / max(1, len(v))),
            "min": float(np.nanmin(v)) if finite.sum() else np.nan,
            "p01": float(np.nanpercentile(v, 1)) if finite.sum() else np.nan,
            "p05": float(np.nanpercentile(v, 5)) if finite.sum() else np.nan,
            "mean": float(np.nanmean(v)) if finite.sum() else np.nan,
            "std": float(np.nanstd(v)) if finite.sum() else np.nan,
            "p50": float(np.nanpercentile(v, 50)) if finite.sum() else np.nan,
            "p95": float(np.nanpercentile(v, 95)) if finite.sum() else np.nan,
            "p99": float(np.nanpercentile(v, 99)) if finite.sum() else np.nan,
            "max": float(np.nanmax(v)) if finite.sum() else np.nan,
            "hash": value_hash(v),
            "prob_like": bool(is_prob_like(v))
        })
    target_stats_df = pd.DataFrame(target_stats)

    target_corr_pearson = target_df.corr(method="pearson")
    target_corr_spearman = target_df.corr(method="spearman")

    # -------------------------------------------------------------------------
    # Grid and smoothing
    # -------------------------------------------------------------------------
    if args.bandwidth == "auto":
        h = auto_bandwidth(coords, k=20)
    else:
        h = float(args.bandwidth)

    gx, gy, XX, YY, GP = build_grid(coords, grid_n=args.grid_n)

    density, n_neigh = compute_density_and_neighbors(
        coords,
        GP,
        bandwidth=h,
        radius_mult=args.radius_mult,
        chunk_size=2500
    )

    point_colors = None
    if state_label_col and state_label_col in merged.columns:
        point_colors = get_point_colors(merged[state_label_col])

    landscapes = {}
    grid_df = pd.DataFrame({
        "latent_1": GP[:, 0],
        "latent_2": GP[:, 1],
        "latent_density": density,
        "n_neighbors": n_neigh
    })

    peaks_all = []
    mask_audits = {}

    for t in target_df.columns:
        vals = target_df[t].values.astype(float)

        z_flat_raw = kernel_smooth_no_extrapolate(
            coords=coords,
            values=vals,
            grid_points=GP,
            bandwidth=h,
            radius_mult=args.radius_mult,
            chunk_size=2500
        )

        z_flat, mask_audit = apply_density_mask(
            z_flat_raw,
            density=density,
            n_neigh=n_neigh,
            density_mask_q=args.density_mask_q,
            min_neighbors=args.min_neighbors
        )

        Z = z_flat.reshape(len(gy), len(gx))
        landscapes[t] = Z
        grid_df[f"{t}_landscape"] = z_flat
        mask_audits[t] = mask_audit

        peaks = find_peaks(
            Z,
            gx=gx,
            gy=gy,
            q=args.peak_q,
            max_peaks=args.max_peaks,
            filter_size=11
        )
        peaks["target"] = t
        peaks["source_column"] = target_col_map.get(t, "")
        peaks["source_type"] = target_source.get(t, "")
        peaks_all.append(peaks)

    peaks_df = pd.concat(peaks_all, axis=0, ignore_index=True) if peaks_all else pd.DataFrame()

    # Grid-level diagnostics
    grid_target_cols = [f"{t}_landscape" for t in target_df.columns]
    grid_corr = grid_df[grid_target_cols].corr(method="spearman")

    grid_allclose_rows = []
    for i, t1 in enumerate(target_df.columns):
        for t2 in list(target_df.columns)[i + 1:]:
            a = grid_df[f"{t1}_landscape"].values
            b = grid_df[f"{t2}_landscape"].values
            common = np.isfinite(a) & np.isfinite(b)
            allclose = bool(np.allclose(a[common], b[common], rtol=1e-5, atol=1e-8)) if common.sum() else False
            diff_mean = float(np.nanmean(np.abs(a[common] - b[common]))) if common.sum() else np.nan
            grid_allclose_rows.append({
                "target_1": t1,
                "target_2": t2,
                "n_common_grid": int(common.sum()),
                "allclose": allclose,
                "mean_abs_grid_difference": diff_mean,
                "hash_1": value_hash(a),
                "hash_2": value_hash(b)
            })
    grid_allclose_df = pd.DataFrame(grid_allclose_rows)

    # -------------------------------------------------------------------------
    # Save audit tables
    # -------------------------------------------------------------------------
    target_stats_csv = os.path.join(args.outdir, "step77a_target_value_audit.csv")
    corr_pearson_csv = os.path.join(args.outdir, "step77a_target_corr_pearson.csv")
    corr_spearman_csv = os.path.join(args.outdir, "step77a_target_corr_spearman.csv")
    grid_csv = os.path.join(args.outdir, "step77a_grid_landscape_values.csv")
    peaks_csv = os.path.join(args.outdir, "step77a_landscape_peaks.csv")
    grid_corr_csv = os.path.join(args.outdir, "step77a_grid_corr_spearman.csv")
    grid_allclose_csv = os.path.join(args.outdir, "step77a_grid_allclose_audit.csv")
    spot_csv = os.path.join(args.outdir, "step77a_per_spot_latent_targets.csv")

    target_stats_df.to_csv(target_stats_csv, index=False)
    target_corr_pearson.to_csv(corr_pearson_csv)
    target_corr_spearman.to_csv(corr_spearman_csv)
    grid_df.to_csv(grid_csv, index=False)
    peaks_df.to_csv(peaks_csv, index=False)
    grid_corr.to_csv(grid_corr_csv)
    grid_allclose_df.to_csv(grid_allclose_csv, index=False)

    spot_out = merged[["_obs_names_", "_adata_id_", "latent_1", "latent_2"]].copy()
    if state_label_col and state_label_col in merged.columns:
        spot_out[state_label_col] = merged[state_label_col].values
    if time_col and time_col in merged.columns:
        spot_out[time_col] = merged[time_col].values
    for t in target_df.columns:
        spot_out[t] = target_df[t].values
    spot_out.to_csv(spot_csv, index=False)

    # -------------------------------------------------------------------------
    # Figure: 2D atlas
    # -------------------------------------------------------------------------
    n = len(target_df.columns)
    ncols = 2 if n <= 4 else 4
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 5.6 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.ravel()

    im_for_prob = None
    im_for_score = None

    for ax, t in zip(axes, target_df.columns):
        vals = target_df[t].values.astype(float)
        cmap, norm, kind = target_style(t, vals)
        im = plot_2d(
            ax, gx, gy, landscapes[t],
            coords=coords if args.show_points else None,
            point_colors=point_colors if args.show_points else None,
            title=f"{pretty_target_name(t)}\n{target_source.get(t, '')}",
            cmap=cmap,
            norm=norm,
            show_points=args.show_points
        )
        if kind == "probability" and im_for_prob is None:
            im_for_prob = im
        if kind == "score" and im_for_score is None:
            im_for_score = im

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Step77A fixed latent state-probability landscapes", fontsize=22, weight="bold", y=0.995)

    if im_for_prob is not None:
        cax = fig.add_axes([0.925, 0.56, 0.012, 0.28])
        cb = fig.colorbar(im_for_prob, cax=cax)
        cb.set_label("Probability")

    if im_for_score is not None:
        cax = fig.add_axes([0.925, 0.16, 0.012, 0.28])
        cb = fig.colorbar(im_for_score, cax=cax)
        cb.set_label("Score")

    fig.text(
        0.5, 0.014,
        "Landscapes use target-specific Gaussian-kernel smoothing with low-density masking; labels indicate value source.",
        ha="center",
        fontsize=10
    )

    atlas_png = os.path.join(args.outdir, "Fig_Step77A_Fixed_StateProbabilityLandscape_2D_Atlas.png")
    atlas_pdf = os.path.join(args.outdir, "Fig_Step77A_Fixed_StateProbabilityLandscape_2D_Atlas.pdf")
    atlas_svg = os.path.join(args.outdir, "Fig_Step77A_Fixed_StateProbabilityLandscape_2D_Atlas.svg")

    plt.tight_layout(rect=[0.02, 0.04, 0.90, 0.96])
    fig.savefig(atlas_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(atlas_pdf, bbox_inches="tight")
    fig.savefig(atlas_svg, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Per-target 2D/3D/ridge figures
    # -------------------------------------------------------------------------
    per_target_outputs = []
    for t in target_df.columns:
        vals = target_df[t].values.astype(float)
        cmap, norm, kind = target_style(t, vals)
        Z = landscapes[t]
        peaks_sub = peaks_df.loc[peaks_df["target"] == t].copy()

        fig = plt.figure(figsize=(18, 5.8))
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = plot_2d(
            ax1, gx, gy, Z,
            coords=coords if args.show_points else None,
            point_colors=point_colors if args.show_points else None,
            title=f"{pretty_target_name(t)} | 2D landscape",
            cmap=cmap,
            norm=norm,
            show_points=args.show_points
        )

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        plot_3d(
            ax2, XX, YY, Z,
            cmap=cmap,
            norm=norm,
            title=f"{pretty_target_name(t)} | 3D surface"
        )

        ax3 = fig.add_subplot(1, 3, 3)
        plot_ridge(
            ax3, gx, gy, Z, peaks_sub,
            coords=coords if args.show_points else None,
            point_colors=point_colors if args.show_points else None,
            title=f"{pretty_target_name(t)} | ridge / peak map",
            cmap=cmap,
            norm=norm
        )

        cax = fig.add_axes([0.93, 0.16, 0.012, 0.70])
        cb = fig.colorbar(im1, cax=cax)
        cb.set_label(pretty_target_name(t))

        fig.suptitle(
            f"Step77A | {pretty_target_name(t)} latent landscape",
            fontsize=20,
            weight="bold",
            y=0.995
        )
        fig.text(
            0.5, 0.012,
            f"Source: {target_source.get(t, '')}; masked outside low-density latent support.",
            ha="center",
            fontsize=10
        )

        base = f"Fig_Step77A_{t}_Fixed_Landscape_2D3DPeaks"
        out_png = os.path.join(args.outdir, base + ".png")
        out_pdf = os.path.join(args.outdir, base + ".pdf")
        out_svg = os.path.join(args.outdir, base + ".svg")
        plt.tight_layout(rect=[0.02, 0.04, 0.92, 0.95])
        fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        fig.savefig(out_svg, bbox_inches="tight")
        plt.close(fig)

        per_target_outputs.append({
            "target": t,
            "png": out_png,
            "pdf": out_pdf,
            "svg": out_svg
        })

    # -------------------------------------------------------------------------
    # Diagnostic figure: correlation and allclose audit
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    im = ax.imshow(target_corr_spearman.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Per-spot target Spearman correlation", weight="bold")
    ax.set_xticks(range(len(target_corr_spearman.columns)))
    ax.set_xticklabels(target_corr_spearman.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(target_corr_spearman.index)))
    ax.set_yticklabels(target_corr_spearman.index)
    for i in range(target_corr_spearman.shape[0]):
        for j in range(target_corr_spearman.shape[1]):
            ax.text(j, i, f"{target_corr_spearman.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1]
    im = ax.imshow(grid_corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Grid landscape Spearman correlation", weight="bold")
    ax.set_xticks(range(len(grid_corr.columns)))
    ax.set_xticklabels([c.replace("_landscape", "") for c in grid_corr.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(grid_corr.index)))
    ax.set_yticklabels([c.replace("_landscape", "") for c in grid_corr.index])
    for i in range(grid_corr.shape[0]):
        for j in range(grid_corr.shape[1]):
            ax.text(j, i, f"{grid_corr.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Step77A target-difference audit", fontsize=18, weight="bold")
    diag_png = os.path.join(args.outdir, "Fig_Step77A_TargetDifferenceAudit.png")
    diag_pdf = os.path.join(args.outdir, "Fig_Step77A_TargetDifferenceAudit.pdf")
    diag_svg = os.path.join(args.outdir, "Fig_Step77A_TargetDifferenceAudit.svg")

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.savefig(diag_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(diag_pdf, bbox_inches="tight")
    fig.savefig(diag_svg, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Final JSON report
    # -------------------------------------------------------------------------
    report = {
        "status": "ok",
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "n_obs_input": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "n_obs_used_after_finite_coord_filter": int(coords.shape[0]),
        "state_merge_audit": merge_audit,
        "latent_coordinate_audit": coord_audit,
        "state_label_source": state_label_col,
        "time_col": time_col,
        "prob_cols_inferred": prob_cols,
        "target_source_audit": target_source_audit,
        "target_stats": target_stats_df.to_dict(orient="records"),
        "target_corr_spearman": target_corr_spearman.to_dict(),
        "target_corr_pearson": target_corr_pearson.to_dict(),
        "grid_corr_spearman": grid_corr.to_dict(),
        "grid_allclose_audit": grid_allclose_df.to_dict(orient="records"),
        "bandwidth_used": float(h),
        "grid_n": int(args.grid_n),
        "radius_mult": float(args.radius_mult),
        "density_mask_q": float(args.density_mask_q),
        "min_neighbors": int(args.min_neighbors),
        "mask_audits": mask_audits,
        "outputs": {
            "atlas_png": atlas_png,
            "atlas_pdf": atlas_pdf,
            "atlas_svg": atlas_svg,
            "diagnostic_png": diag_png,
            "diagnostic_pdf": diag_pdf,
            "diagnostic_svg": diag_svg,
            "target_stats_csv": target_stats_csv,
            "target_corr_pearson_csv": corr_pearson_csv,
            "target_corr_spearman_csv": corr_spearman_csv,
            "grid_csv": grid_csv,
            "grid_corr_csv": grid_corr_csv,
            "grid_allclose_csv": grid_allclose_csv,
            "peaks_csv": peaks_csv,
            "per_spot_csv": spot_csv,
            "per_target_figures": per_target_outputs,
        },
        "formula": "f_t(z)=sum_i K(||z-z_i||/h)y_{it}/sum_i K(||z-z_i||/h), with low-density grid masking.",
        "interpretation_note": (
            "Step77A constructs target-specific smoothed conditional landscapes in latent space. "
            "If model state-probability columns are missing or nearly duplicate and --prob_source=auto, "
            "core/peri/remote landscapes are estimated from state_label one-hot smoothing. "
            "This is a visualization-level summary, not a physical energy surface, lineage potential, or functional validation."
        )
    }

    report_path = os.path.join(args.outdir, "step77a_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(report), f, indent=2, ensure_ascii=False)

    print("=" * 100)
    print("DONE Step77A")
    print("=" * 100)
    print(json.dumps({
        "outdir": args.outdir,
        "report_json": report_path,
        "atlas_png": atlas_png,
        "diagnostic_png": diag_png,
        "target_stats_csv": target_stats_csv,
        "grid_allclose_csv": grid_allclose_csv,
        "peaks_csv": peaks_csv,
        "prob_source_used_for_state_probs": target_source_audit.get("use_label_for_core_peri_remote"),
        "model_state_duplicate_detected": target_source_audit.get("model_state_duplicate_detected"),
        "latent_coordinate_audit": coord_audit,
        "state_merge_overlap": merge_audit.get("state_merge_overlap"),
    }, indent=2))


if __name__ == "__main__":
    main()
