#!/usr/bin/env python3
# WHITE-BACKGROUND VARIANT GENERATED 2026-09-01.
# Source: E:\vir\ST\76_latent_state_potential_landscape.py
# Scientific calculations and data mappings are unchanged; only visual theme literals were remapped.
# -*- coding: utf-8 -*-

"""
Step76 | StrokeNiche latent-state pseudo-energy landscape

Purpose
-------
Construct a density-derived pseudo-energy / potential landscape in latent space:

    U(z) = -log[p(z) + epsilon]

where p(z) is a 2D KDE density estimated from latent cell/spot coordinates.

Outputs
-------
1. 2D pseudo-energy landscape with state-colored cells and contours
2. 2D landscape with D1-D3-D7 trajectory / state centroid arrows
3. 2D landscape with virtual perturbation shift vectors
4. 3D surface view of U(z)
5. Per-cell latent/pseudo-energy table
6. Grid-level potential surface table
7. Audit report

Interpretation
--------------
This is a density-derived pseudo-energy visualization, not a physical energy,
not a true Waddington landscape, and not functional validation.
"""

from pathlib import Path
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from scipy import sparse
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA, TruncatedSVD


PERTURBATIONS = [
    {
        "name": "Ccl2/Ccr2-Ackr1 blockade",
        "short": "Ccl2/Ccr2-Ackr1",
        "genes": ["Ccl2", "Ccr2", "Ackr1", "Ccl7", "Ccl12"],
        "direction": "down",
    },
    {
        "name": "Spp1-Cd44 blockade",
        "short": "Spp1-Cd44",
        "genes": ["Spp1", "Cd44", "Itgav", "Itgb1", "Apoe"],
        "direction": "down",
    },
    {
        "name": "Vegfa-Flt1/Kdr blockade",
        "short": "Vegfa-Flt1/Kdr",
        "genes": ["Vegfa", "Flt1", "Kdr", "Vwf", "Pecam1"],
        "direction": "down",
    },
    {
        "name": "Ferroptosis down",
        "short": "Ferroptosis down",
        "genes": ["Hmox1", "Fth1", "Ftl1", "Slc7a11", "Gpx4", "Acsl4", "Tfrc", "Ptgs2"],
        "direction": "down",
    },
    {
        "name": "Repair-ECM up",
        "short": "Repair-ECM up",
        "genes": ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Spp1", "Apoe", "Vim", "Postn", "Timp1"],
        "direction": "up",
    },
]

REPAIR_GENES = ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Spp1", "Apoe", "Vim", "Postn", "Timp1"]


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path):
    if not path or str(path).strip() == "":
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def first_existing(cols, candidates):
    cols = list(cols)
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def dense_vector(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def zscore_clip(x, clip=2.5):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    z = (x - mu) / sd
    z = np.clip(z, -clip, clip)
    z[~np.isfinite(z)] = 0.0
    return z


def minmax01(x, q_low=0.02, q_high=0.98):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    lo = np.nanquantile(x[ok], q_low)
    hi = np.nanquantile(x[ok], q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        lo = np.nanmin(x[ok])
        hi = np.nanmax(x[ok])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        return out
    out[ok] = (x[ok] - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~np.isfinite(out)] = 0
    return out


def slugify(x):
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


# =============================================================================
# Colormaps
# =============================================================================

def cmap_energy():
    return LinearSegmentedColormap.from_list(
        "pseudo_energy",
        ["#12001f", "#1f0a53", "#3454a4", "#2fb8c6", "#e6f598", "#fff7bc"]
    )


def cmap_density():
    return LinearSegmentedColormap.from_list(
        "density_basin",
        ["#ffffff", "#071d3a", "#123c69", "#2b8cbe", "#a6cee3", "#111827"]
    )


STATE_COLORS = {
    "lesion-core-like": "#ef4444",
    "core": "#ef4444",
    "peri-infarct": "#f59e0b",
    "peri": "#f59e0b",
    "remote-like": "#38bdf8",
    "remote": "#38bdf8",
    "unknown": "#374151",
}


# =============================================================================
# Metadata merge
# =============================================================================

def get_obs_id_series(adata, obs, user_col=""):
    if user_col and user_col in obs.columns:
        return obs[user_col].astype(str), user_col

    candidates = [
        "obs_name", "cell_id", "spot_id", "barcode",
        "CellID", "SpotID", "cell", "spot", "id"
    ]
    col = first_existing(obs.columns, candidates)
    if col:
        return obs[col].astype(str), col

    return pd.Series(adata.obs_names.astype(str), index=obs.index), "_obs_names_"


def merge_state_table(meta, state_df, adata_id_col="_adata_id_", state_id_col=""):
    audit = {
        "state_table_available": False,
        "state_table_rows": 0,
        "merge_mode": "none",
        "merge_key_meta": "",
        "merge_key_state": "",
        "state_merge_overlap": 0,
    }

    if state_df is None or state_df.empty:
        return meta, audit

    audit["state_table_available"] = True
    audit["state_table_rows"] = int(len(state_df))

    st = state_df.copy()

    if state_id_col and state_id_col in st.columns:
        sid = state_id_col
    else:
        sid = first_existing(
            st.columns,
            [
                "obs_name", "cell_id", "spot_id", "barcode",
                "CellID", "SpotID", "cell", "spot", "id"
            ]
        )

    if sid:
        left = meta[adata_id_col].astype(str)
        right = st[sid].astype(str)
        overlap = len(set(left).intersection(set(right)))
        audit["state_merge_overlap"] = int(overlap)

        if overlap > 0:
            meta = meta.copy()
            st = st.copy()
            meta["_merge_key_"] = left
            st["_merge_key_"] = right
            st = st.drop_duplicates("_merge_key_")
            out = meta.merge(st, on="_merge_key_", how="left", suffixes=("", "_state"))
            out.drop(columns=["_merge_key_"], inplace=True, errors="ignore")
            audit["merge_mode"] = "id"
            audit["merge_key_meta"] = adata_id_col
            audit["merge_key_state"] = sid
            return out, audit

    if len(st) == len(meta):
        meta = meta.copy()
        st = st.copy()
        meta["_row_order_"] = np.arange(len(meta))
        st["_row_order_"] = np.arange(len(st))
        out = meta.merge(st, on="_row_order_", how="left", suffixes=("", "_state"))
        out.drop(columns=["_row_order_"], inplace=True, errors="ignore")
        audit["merge_mode"] = "row_order"
        audit["merge_key_meta"] = "row_order"
        audit["merge_key_state"] = "row_order"
        audit["state_merge_overlap"] = int(len(meta))
        return out, audit

    audit["merge_mode"] = "failed"
    return meta, audit


# =============================================================================
# Coordinate inference
# =============================================================================

def infer_latent_coordinates(adata, meta, xcol="", ycol="", obsm_key=""):
    """
    Priority:
    1. user-specified xcol/ycol from merged meta
    2. common 2D latent/UMAP columns in meta
    3. user-specified obsm_key
    4. common obsm 2D/high-dim embeddings
    5. PCA/SVD fallback from expression matrix
    """

    if xcol and ycol and xcol in meta.columns and ycol in meta.columns:
        x = pd.to_numeric(meta[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(meta[ycol], errors="coerce").to_numpy(dtype=float)
        return np.column_stack([x, y]), {"mode": "meta_user", "xcol": xcol, "ycol": ycol}

    candidate_pairs = [
        ("latent_1", "latent_2"),
        ("latent_x", "latent_y"),
        ("strict_latent_1", "strict_latent_2"),
        ("strict_z1", "strict_z2"),
        ("strict_z_1", "strict_z_2"),
        ("z1", "z2"),
        ("z_1", "z_2"),
        ("Z1", "Z2"),
        ("UMAP_1", "UMAP_2"),
        ("umap_1", "umap_2"),
        ("nicheformer_umap_1", "nicheformer_umap_2"),
        ("scgpt_umap_1", "scgpt_umap_2"),
        ("latent_umap_1", "latent_umap_2"),
        ("X_umap_1", "X_umap_2"),
        ("pca_1", "pca_2"),
        ("PC1", "PC2"),
    ]

    for a, b in candidate_pairs:
        if a in meta.columns and b in meta.columns:
            x = pd.to_numeric(meta[a], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(meta[b], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(x).sum() > 10 and np.isfinite(y).sum() > 10:
                return np.column_stack([x, y]), {"mode": "meta_auto", "xcol": a, "ycol": b}

    # columns with strict_z_* or latent_* high-dimensional values
    high_dim_prefixes = ["strict_z_", "strict_latent_", "latent_", "z_"]
    for prefix in high_dim_prefixes:
        cols = [c for c in meta.columns if str(c).startswith(prefix)]
        numeric_cols = []
        for c in cols:
            vals = pd.to_numeric(meta[c], errors="coerce")
            if vals.notna().sum() > 10:
                numeric_cols.append(c)
        if len(numeric_cols) >= 2:
            M = meta[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=float)
            if M.shape[1] == 2:
                return M, {"mode": "meta_highdim_2d", "columns": numeric_cols}
            Z = PCA(n_components=2, random_state=0).fit_transform(M)
            return Z, {"mode": "meta_highdim_pca", "columns": numeric_cols[:20], "n_columns": len(numeric_cols)}

    if obsm_key and obsm_key in adata.obsm:
        arr = np.asarray(adata.obsm[obsm_key])
        if arr.ndim == 2 and arr.shape[0] == adata.n_obs:
            if arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    return arr[:, :2].astype(float), {"mode": "obsm_user_2d", "obsm_key": obsm_key}
                Z = PCA(n_components=2, random_state=0).fit_transform(arr)
                return Z, {"mode": "obsm_user_pca", "obsm_key": obsm_key, "n_dim": int(arr.shape[1])}

    obsm_candidates = [
        "X_umap",
        "X_pca",
        "X_diffmap",
        "X_draw_graph_fa",
        "X_nicheformer",
        "nicheformer",
        "strict_z",
        "latent",
        "X_latent",
        "X_scgpt",
        "spatial",
    ]

    for key in obsm_candidates:
        if key in adata.obsm:
            arr = np.asarray(adata.obsm[key])
            if arr.ndim == 2 and arr.shape[0] == adata.n_obs and arr.shape[1] >= 2:
                if arr.shape[1] == 2:
                    return arr[:, :2].astype(float), {"mode": "obsm_auto_2d", "obsm_key": key}
                Z = PCA(n_components=2, random_state=0).fit_transform(arr)
                return Z, {"mode": "obsm_auto_pca", "obsm_key": key, "n_dim": int(arr.shape[1])}

    # last fallback: expression SVD/PCA
    X = adata.X
    if sparse.issparse(X):
        svd = TruncatedSVD(n_components=2, random_state=0)
        Z = svd.fit_transform(X)
        return Z, {"mode": "expression_truncated_svd_fallback", "n_vars": int(adata.n_vars)}
    else:
        arr = np.asarray(X)
        Z = PCA(n_components=2, random_state=0).fit_transform(arr)
        return Z, {"mode": "expression_pca_fallback", "n_vars": int(adata.n_vars)}


def infer_prob_cols(meta):
    def pick(cands):
        return first_existing(meta.columns, cands)

    return {
        "core": pick(["core_probability", "core_prob", "prob_core", "p_core", "lesion_core_probability"]),
        "peri": pick(["peri_probability", "peri_prob", "prob_peri", "p_peri", "peri_infarct_probability"]),
        "remote": pick(["remote_probability", "remote_prob", "prob_remote", "p_remote", "remote_like_probability"]),
        "repair": pick(["repair_score", "repair_ECM_score", "repair_ecm_score", "repair_module_score"]),
    }


def infer_state_label(meta, prob_cols):
    label_col = first_existing(
        meta.columns,
        [
            "state_group", "region_refined", "region_manual_final",
            "region_auto", "predicted_state", "state_label", "region"
        ]
    )

    if label_col:
        raw = meta[label_col].astype(str).str.lower()
        labels = []
        for v in raw:
            if "core" in v or "lesion" in v:
                labels.append("lesion-core-like")
            elif "peri" in v:
                labels.append("peri-infarct")
            elif "remote" in v:
                labels.append("remote-like")
            else:
                labels.append(str(v))
        return np.array(labels, dtype=object), label_col

    if all(prob_cols.get(k) in meta.columns for k in ["core", "peri", "remote"]):
        P = np.column_stack([
            pd.to_numeric(meta[prob_cols["core"]], errors="coerce").fillna(0).to_numpy(),
            pd.to_numeric(meta[prob_cols["peri"]], errors="coerce").fillna(0).to_numpy(),
            pd.to_numeric(meta[prob_cols["remote"]], errors="coerce").fillna(0).to_numpy(),
        ])
        idx = np.argmax(P, axis=1)
        names = np.array(["lesion-core-like", "peri-infarct", "remote-like"], dtype=object)
        return names[idx], "argmax_state_probability"

    return np.array(["unknown"] * len(meta), dtype=object), "unknown"


def infer_time_col(meta):
    return first_existing(
        meta.columns,
        [
            "timepoint", "time_point", "stage", "stage_numeric", "condition",
            "day", "dpi", "group", "sample_label"
        ]
    )


def stage_order_values(values):
    vals = pd.Series(values).dropna().astype(str).unique().tolist()

    def key(v):
        s = str(v).lower()
        if "sham" in s or "control" in s or s in ["0", "0.0"]:
            return 0
        m = re.search(r"d\s*([0-9]+)", s)
        if m:
            return int(m.group(1))
        try:
            return float(s)
        except Exception:
            return 999
    return sorted(vals, key=key)


# =============================================================================
# Gene score helpers
# =============================================================================

def gene_lookup(adata):
    mapping = {}
    for v in adata.var_names.astype(str):
        mapping[v.upper()] = v

    for col in ["gene", "genes", "gene_symbol", "symbol", "feature_name", "gene_name"]:
        if col in adata.var.columns:
            for var_name, val in zip(adata.var_names.astype(str), adata.var[col].astype(str)):
                key = str(val).upper()
                if key and key not in mapping:
                    mapping[key] = var_name

    return mapping


def get_gene_expr(adata, gene, gmap):
    var = gmap.get(str(gene).upper(), None)
    if var is None:
        return np.full(adata.n_obs, np.nan)
    return dense_vector(adata[:, var].X).astype(float)


def compute_gene_set_score(adata, genes, gmap, min_genes=1):
    exprs = []
    matched = []
    for g in genes:
        if str(g).upper() in gmap:
            exprs.append(get_gene_expr(adata, g, gmap))
            matched.append(g)

    if len(exprs) < min_genes:
        return np.full(adata.n_obs, np.nan), matched

    M = np.vstack(exprs).T
    Z = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        Z[:, j] = zscore_clip(M[:, j], clip=2.5)
    score = np.nanmean(Z, axis=1)
    return score, matched


# =============================================================================
# KDE / pseudo-energy
# =============================================================================

def compute_kde_landscape(Z, bandwidth=0.0, grid_n=220, eps=1e-12):
    ok = np.isfinite(Z[:, 0]) & np.isfinite(Z[:, 1])
    Z0 = Z[ok]

    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z0)

    n = Zs.shape[0]
    if bandwidth <= 0:
        # Silverman-like rule for 2D standardized coordinates.
        bandwidth = n ** (-1.0 / 6.0)
        bandwidth = float(np.clip(bandwidth, 0.08, 0.45))

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(Zs)

    x0, x1 = np.nanmin(Zs[:, 0]), np.nanmax(Zs[:, 0])
    y0, y1 = np.nanmin(Zs[:, 1]), np.nanmax(Zs[:, 1])
    px = (x1 - x0) * 0.08
    py = (y1 - y0) * 0.08
    x0 -= px
    x1 += px
    y0 -= py
    y1 += py

    gx = np.linspace(x0, x1, grid_n)
    gy = np.linspace(y0, y1, grid_n)
    GX, GY = np.meshgrid(gx, gy)
    grid_scaled = np.column_stack([GX.ravel(), GY.ravel()])

    logp = kde.score_samples(grid_scaled)
    p = np.exp(logp)
    U = -np.log(p + eps)
    U = U.reshape(grid_n, grid_n)
    p = p.reshape(grid_n, grid_n)

    # Smooth and normalize potential for visualization.
    U = gaussian_filter(U, sigma=1.0)
    U_norm = U - np.nanmin(U)
    q_hi = np.nanquantile(U_norm, 0.98)
    if q_hi > 0:
        U_norm = np.clip(U_norm / q_hi, 0, 1)

    grid_orig = scaler.inverse_transform(grid_scaled)
    Xorig = grid_orig[:, 0].reshape(grid_n, grid_n)
    Yorig = grid_orig[:, 1].reshape(grid_n, grid_n)

    # Per-cell pseudo-energy by KDE density at each cell.
    logp_cell = kde.score_samples(scaler.transform(Z))
    p_cell = np.exp(logp_cell)
    U_cell = -np.log(p_cell + eps)
    U_cell = U_cell - np.nanmin(U_cell)
    denom = np.nanquantile(U_cell, 0.98)
    if denom > 0:
        U_cell = np.clip(U_cell / denom, 0, 1)

    return {
        "X": Xorig,
        "Y": Yorig,
        "density": p,
        "U": U,
        "U_norm": U_norm,
        "U_cell": U_cell,
        "bandwidth": bandwidth,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


# =============================================================================
# Vector summaries
# =============================================================================

def compute_state_centroids(Z, labels, time_values=None):
    rows = []

    if time_values is None:
        for lab in sorted(pd.Series(labels).dropna().astype(str).unique()):
            mask = pd.Series(labels).astype(str).to_numpy() == lab
            if mask.sum() < 5:
                continue
            rows.append({
                "kind": "state",
                "state": lab,
                "time": "",
                "n": int(mask.sum()),
                "x": float(np.nanmean(Z[mask, 0])),
                "y": float(np.nanmean(Z[mask, 1])),
            })
    else:
        times = stage_order_values(time_values)
        arr_time = pd.Series(time_values).astype(str).to_numpy()
        arr_lab = pd.Series(labels).astype(str).to_numpy()
        for t in times:
            for lab in ["lesion-core-like", "peri-infarct", "remote-like"]:
                mask = (arr_time == str(t)) & (arr_lab == lab)
                if mask.sum() < 5:
                    continue
                rows.append({
                    "kind": "state_time",
                    "state": lab,
                    "time": str(t),
                    "n": int(mask.sum()),
                    "x": float(np.nanmean(Z[mask, 0])),
                    "y": float(np.nanmean(Z[mask, 1])),
                })

    return pd.DataFrame(rows)


def compute_perturbation_vectors(adata, meta, Z, gmap, prob_cols):
    if prob_cols["core"] in meta.columns:
        core = pd.to_numeric(meta[prob_cols["core"]], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        core = np.zeros(len(meta))

    if prob_cols["remote"] in meta.columns:
        remote = pd.to_numeric(meta[prob_cols["remote"]], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        remote = np.zeros(len(meta))

    if prob_cols["peri"] in meta.columns:
        peri = pd.to_numeric(meta[prob_cols["peri"]], errors="coerce").fillna(0).to_numpy(dtype=float)
    else:
        peri = np.zeros(len(meta))

    if prob_cols.get("repair") and prob_cols["repair"] in meta.columns:
        repair_raw = pd.to_numeric(meta[prob_cols["repair"]], errors="coerce").fillna(0).to_numpy(dtype=float)
        repair_source = f"column:{prob_cols['repair']}"
        repair_matched = ""
    else:
        repair_raw, repair_matched_list = compute_gene_set_score(adata, REPAIR_GENES, gmap, min_genes=1)
        repair_source = "computed_repair_gene_set_score"
        repair_matched = ";".join(repair_matched_list)

    repair = minmax01(repair_raw)
    core_n = minmax01(core)
    healthy_like = 0.45 * minmax01(remote) + 0.25 * minmax01(peri) + 0.30 * repair - 0.40 * core_n

    rows = []

    for p in PERTURBATIONS:
        score, matched = compute_gene_set_score(adata, p["genes"], gmap, min_genes=1)
        score_n = minmax01(score)

        if p["direction"] == "up":
            source_strength = (1.0 - repair) * (0.5 + 0.5 * core_n)
            dest_strength = repair * (0.4 + 0.6 * (minmax01(peri) + minmax01(remote)) / 2)
        else:
            source_strength = score_n * (0.5 + 0.5 * core_n)
            dest_strength = healthy_like

        # Weighted source/destination centroids. Robust top quantile selection.
        src_thr = np.nanquantile(source_strength, 0.88)
        dst_thr = np.nanquantile(dest_strength, 0.88)

        src_mask = source_strength >= src_thr
        dst_mask = dest_strength >= dst_thr

        if src_mask.sum() < 10:
            src_mask = source_strength >= np.nanquantile(source_strength, 0.80)
        if dst_mask.sum() < 10:
            dst_mask = dest_strength >= np.nanquantile(dest_strength, 0.80)

        src = np.nanmean(Z[src_mask], axis=0)
        dst = np.nanmean(Z[dst_mask], axis=0)

        rows.append({
            "perturbation": p["name"],
            "short": p["short"],
            "direction": p["direction"],
            "target_genes": ";".join(p["genes"]),
            "matched_target_genes": ";".join(matched),
            "n_matched_target_genes": len(matched),
            "repair_score_source": repair_source,
            "matched_repair_genes": repair_matched,
            "source_n": int(src_mask.sum()),
            "destination_n": int(dst_mask.sum()),
            "source_x": float(src[0]),
            "source_y": float(src[1]),
            "dest_x": float(dst[0]),
            "dest_y": float(dst[1]),
            "dx": float(dst[0] - src[0]),
            "dy": float(dst[1] - src[1]),
            "vector_norm": float(np.sqrt(np.sum((dst - src) ** 2))),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Plotting helpers
# =============================================================================

def setup_dark(ax):
    ax.set_facecolor("#ffffff")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors="#6b7280", labelsize=7)


def scatter_state_cells(ax, Z, labels, size=4, alpha=0.55):
    labels = pd.Series(labels).astype(str).to_numpy()
    for lab in ["lesion-core-like", "peri-infarct", "remote-like"]:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            s=size,
            c=STATE_COLORS.get(lab, "#374151"),
            alpha=alpha,
            linewidths=0,
            rasterized=True,
            label=lab,
            zorder=4,
        )

    other = ~np.isin(labels, ["lesion-core-like", "peri-infarct", "remote-like"])
    if other.sum() > 0:
        ax.scatter(
            Z[other, 0], Z[other, 1],
            s=size,
            c="#6b7280",
            alpha=0.25,
            linewidths=0,
            rasterized=True,
            label="other",
            zorder=3,
        )


def add_energy_background(ax, land, levels=18):
    X = land["X"]
    Y = land["Y"]
    U = land["U_norm"]
    cf = ax.contourf(
        X, Y, U,
        levels=np.linspace(0, 1, levels),
        cmap=cmap_energy(),
        alpha=0.86,
        zorder=1,
    )
    ax.contour(
        X, Y, U,
        levels=np.linspace(0.10, 0.95, 9),
        colors="#1f2937",
        linewidths=0.35,
        alpha=0.30,
        zorder=2,
    )
    return cf


def add_colorbar(fig, mappable, label, x=0.925, y=0.16, h=0.68):
    cax = fig.add_axes([x, y, 0.012, h])
    cb = fig.colorbar(mappable, cax=cax)
    cb.ax.tick_params(colors="#1f2937", labelsize=7)
    cb.outline.set_edgecolor("white")
    cb.set_label(label, color="#1f2937", fontsize=8)
    return cb


def draw_centroid_arrows(ax, centroids):
    if centroids is None or centroids.empty:
        return

    # Time-ordered arrows within each state if available.
    df = centroids.copy()
    if "time" in df.columns and df["time"].astype(str).str.len().max() > 0:
        for state, sub in df.groupby("state"):
            sub = sub.copy()
            sub["_ord"] = range(len(sub))
            sub = sub.sort_values("_ord")
            color = STATE_COLORS.get(state, "#1f2937")
            for i in range(len(sub) - 1):
                a = sub.iloc[i]
                b = sub.iloc[i + 1]
                ax.annotate(
                    "",
                    xy=(b["x"], b["y"]),
                    xytext=(a["x"], a["y"]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4, alpha=0.9),
                    zorder=8,
                )
            for _, r in sub.iterrows():
                ax.scatter(r["x"], r["y"], s=55, c=color, edgecolor="white", linewidth=0.7, zorder=9)
                ax.text(r["x"], r["y"], str(r["time"]), color="#1f2937", fontsize=6,
                        ha="center", va="center", zorder=10)
    else:
        for _, r in df.iterrows():
            color = STATE_COLORS.get(r["state"], "#1f2937")
            ax.scatter(r["x"], r["y"], s=60, c=color, edgecolor="white", linewidth=0.7, zorder=9)
            ax.text(r["x"], r["y"], str(r["state"])[:10], color="#1f2937", fontsize=6,
                    ha="center", va="center", zorder=10)


def draw_perturbation_vectors(ax, vec_df):
    if vec_df is None or vec_df.empty:
        return

    colors = ["#22c55e", "#a78bfa", "#38bdf8", "#f97316", "#facc15"]

    for i, (_, r) in enumerate(vec_df.iterrows()):
        color = colors[i % len(colors)]
        ax.annotate(
            "",
            xy=(r["dest_x"], r["dest_y"]),
            xytext=(r["source_x"], r["source_y"]),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                lw=2.0,
                alpha=0.92,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=12,
        )
        ax.scatter(r["source_x"], r["source_y"], s=42, c=color, edgecolor="white", linewidth=0.6, zorder=13)
        ax.text(
            r["dest_x"], r["dest_y"], r["short"],
            fontsize=7,
            color=color,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=14,
        )


def finalize_2d_ax(ax, title, xlabel="Latent coordinate 1", ylabel="Latent coordinate 2"):
    ax.set_title(title, color="#1f2937", fontsize=12, fontweight="bold", pad=6)
    ax.set_xlabel(xlabel, color="#374151", fontsize=8)
    ax.set_ylabel(ylabel, color="#374151", fontsize=8)
    ax.grid(color="#1f2937", alpha=0.06, linewidth=0.5)


# =============================================================================
# Figures
# =============================================================================

def make_main_figure(Z, labels, land, centroids, vec_df, outbase, dpi=600):
    fig = plt.figure(figsize=(14.2, 10.2), facecolor="#ffffff")
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.10)

    axA = fig.add_subplot(gs[0, 0])
    setup_dark(axA)
    cf = add_energy_background(axA, land)
    scatter_state_cells(axA, Z, labels, size=5, alpha=0.62)
    finalize_2d_ax(axA, "A  Density-derived pseudo-energy landscape")
    leg = axA.legend(loc="upper right", fontsize=7, frameon=False)
    for t in leg.get_texts():
        t.set_color("#1f2937")

    axB = fig.add_subplot(gs[0, 1])
    setup_dark(axB)
    add_energy_background(axB, land)
    scatter_state_cells(axB, Z, labels, size=4, alpha=0.38)
    draw_centroid_arrows(axB, centroids)
    finalize_2d_ax(axB, "B  State/timepoint centroid dynamics")

    axC = fig.add_subplot(gs[1, 0])
    setup_dark(axC)
    add_energy_background(axC, land)
    scatter_state_cells(axC, Z, labels, size=3, alpha=0.25)
    draw_perturbation_vectors(axC, vec_df)
    finalize_2d_ax(axC, "C  Virtual perturbation shift vectors")

    axD = fig.add_subplot(gs[1, 1])
    setup_dark(axD)
    # Density basin view: inverse potential = basin depth
    basin = 1.0 - land["U_norm"]
    cf2 = axD.contourf(
        land["X"], land["Y"], basin,
        levels=np.linspace(0, 1, 18),
        cmap=cmap_density(),
        alpha=0.92,
        zorder=1,
    )
    axD.contour(
        land["X"], land["Y"], land["U_norm"],
        levels=np.linspace(0.12, 0.92, 8),
        colors="#1f2937",
        linewidths=0.35,
        alpha=0.35,
        zorder=2,
    )
    scatter_state_cells(axD, Z, labels, size=3, alpha=0.35)
    finalize_2d_ax(axD, "D  Latent-state basins")

    fig.suptitle(
        "StrokeNiche latent-state pseudo-energy landscape",
        color="#1f2937",
        fontsize=22,
        fontweight="bold",
        y=0.985,
    )

    add_colorbar(fig, cf, "Pseudo-energy U(z), normalized", x=0.925, y=0.56, h=0.30)
    add_colorbar(fig, cf2, "Basin depth / density", x=0.925, y=0.15, h=0.30)

    fig.text(
        0.5, 0.018,
        "U(z) = -log[p(z)+ε], where p(z) is KDE density in latent space. This is a density-derived pseudo-energy visualization, not a physical energy or functional validation.",
        ha="center",
        va="bottom",
        color="#1f2937",
        fontsize=8.5,
    )

    fig.subplots_adjust(left=0.065, right=0.905, top=0.93, bottom=0.06)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{outbase}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_3d_surface_figure(Z, labels, land, outbase, dpi=600):
    fig = plt.figure(figsize=(9.8, 7.6), facecolor="#ffffff")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#ffffff")

    X = land["X"]
    Y = land["Y"]
    U = land["U_norm"]

    ax.plot_surface(
        X, Y, U,
        rstride=3,
        cstride=3,
        cmap=cmap_energy(),
        linewidth=0,
        antialiased=True,
        alpha=0.82,
    )

    # Add projected state clouds at z slightly below min.
    z_floor = -0.04
    labels = pd.Series(labels).astype(str).to_numpy()
    for lab in ["lesion-core-like", "peri-infarct", "remote-like"]:
        mask = labels == lab
        if mask.sum() == 0:
            continue
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            np.full(mask.sum(), z_floor),
            c=STATE_COLORS.get(lab, "#374151"),
            s=4,
            alpha=0.45,
            depthshade=False,
            label=lab,
        )

    ax.set_title("3D pseudo-energy surface of latent stroke-state organization",
                 color="#1f2937", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Latent coordinate 1", color="#374151", labelpad=8)
    ax.set_ylabel("Latent coordinate 2", color="#374151", labelpad=8)
    ax.set_zlabel("Pseudo-energy U(z)", color="#374151", labelpad=8)

    ax.tick_params(colors="#374151", labelsize=7)
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))

    leg = ax.legend(loc="upper right", fontsize=7, frameon=False)
    for t in leg.get_texts():
        t.set_color("#1f2937")

    fig.text(
        0.5, 0.025,
        "3D view is for visualization of density-derived potential basins only; it is not a physical energy landscape.",
        ha="center",
        color="#1f2937",
        fontsize=8,
    )

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{outbase}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--adata_id_col", default="")
    parser.add_argument("--state_id_col", default="")
    parser.add_argument("--latent_x_col", default="")
    parser.add_argument("--latent_y_col", default="")
    parser.add_argument("--obsm_key", default="")

    parser.add_argument("--bandwidth", type=float, default=0.0)
    parser.add_argument("--grid_n", type=int, default=220)
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    print("=" * 100)
    print("Step76 | StrokeNiche latent-state pseudo-energy landscape")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={outdir}")

    adata = ad.read_h5ad(args.h5ad)
    obs = adata.obs.copy()

    adata_id, adata_id_col_used = get_obs_id_series(adata, obs, user_col=args.adata_id_col)
    obs["_adata_id_"] = adata_id.astype(str).values

    state_df = read_table(args.state_table)
    meta, merge_audit = merge_state_table(
        obs,
        state_df,
        adata_id_col="_adata_id_",
        state_id_col=args.state_id_col,
    )

    Z, coord_audit = infer_latent_coordinates(
        adata,
        meta,
        xcol=args.latent_x_col,
        ycol=args.latent_y_col,
        obsm_key=args.obsm_key,
    )

    # Make finite coordinate mask.
    ok = np.isfinite(Z[:, 0]) & np.isfinite(Z[:, 1])
    if ok.sum() < 50:
        raise ValueError("Too few finite latent coordinates.")

    # Use all rows but replace nonfinite with median.
    for j in [0, 1]:
        med = np.nanmedian(Z[ok, j])
        Z[~np.isfinite(Z[:, j]), j] = med

    prob_cols = infer_prob_cols(meta)
    labels, label_source = infer_state_label(meta, prob_cols)
    time_col = infer_time_col(meta)
    time_values = meta[time_col].astype(str).to_numpy() if time_col else None

    land = compute_kde_landscape(
        Z,
        bandwidth=args.bandwidth,
        grid_n=args.grid_n,
        eps=1e-12,
    )

    gmap = gene_lookup(adata)
    centroids = compute_state_centroids(Z, labels, time_values=time_values)
    vec_df = compute_perturbation_vectors(adata, meta, Z, gmap, prob_cols)

    # Save tables
    cell_df = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "latent_1": Z[:, 0],
        "latent_2": Z[:, 1],
        "state_label": labels,
        "pseudo_energy_norm": land["U_cell"],
    })

    if time_col:
        cell_df["time_or_stage"] = time_values

    for k, col in prob_cols.items():
        if col and col in meta.columns:
            cell_df[f"{k}_column"] = pd.to_numeric(meta[col], errors="coerce").to_numpy()

    cell_path = outdir / "step76_per_cell_latent_pseudo_energy.csv"
    cell_df.to_csv(cell_path, index=False)

    grid_df = pd.DataFrame({
        "latent_1": land["X"].ravel(),
        "latent_2": land["Y"].ravel(),
        "density": land["density"].ravel(),
        "pseudo_energy_raw": land["U"].ravel(),
        "pseudo_energy_norm": land["U_norm"].ravel(),
    })
    grid_path = outdir / "step76_grid_pseudo_energy_surface.csv"
    grid_df.to_csv(grid_path, index=False)

    centroids_path = outdir / "step76_state_time_centroids.csv"
    centroids.to_csv(centroids_path, index=False)

    vectors_path = outdir / "step76_virtual_perturbation_shift_vectors.csv"
    vec_df.to_csv(vectors_path, index=False)

    # Figures
    main_outbase = outdir / "Fig_Step76_StrokeNiche_LatentStatePotentialLandscape_2D"
    make_main_figure(Z, labels, land, centroids, vec_df, str(main_outbase), dpi=args.dpi)

    surface_outbase = outdir / "Fig_Step76_StrokeNiche_LatentStatePotentialLandscape_3D"
    make_3d_surface_figure(Z, labels, land, str(surface_outbase), dpi=args.dpi)

    audit = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "adata_id_col_used": adata_id_col_used,
        "state_merge_audit": merge_audit,
        "latent_coordinate_audit": coord_audit,
        "state_label_source": label_source,
        "time_col": time_col,
        "prob_cols": prob_cols,
        "kde_bandwidth_used": land["bandwidth"],
        "grid_n": args.grid_n,
        "outputs": {
            "main_2d_png": str(main_outbase.with_suffix(".png")),
            "main_2d_pdf": str(main_outbase.with_suffix(".pdf")),
            "main_2d_svg": str(main_outbase.with_suffix(".svg")),
            "surface_3d_png": str(surface_outbase.with_suffix(".png")),
            "surface_3d_pdf": str(surface_outbase.with_suffix(".pdf")),
            "surface_3d_svg": str(surface_outbase.with_suffix(".svg")),
            "per_cell_energy_csv": str(cell_path),
            "grid_energy_csv": str(grid_path),
            "centroids_csv": str(centroids_path),
            "perturbation_vectors_csv": str(vectors_path),
        },
        "formula": "U(z) = -log[p(z) + epsilon], where p(z) is KDE density in latent space.",
        "interpretation_note": (
            "Step76 constructs a density-derived pseudo-energy landscape in latent/state space. "
            "Low U(z) indicates high-density latent basins; high U(z) indicates sparse regions or barriers. "
            "This should be interpreted as a visualization-level potential landscape, not a physical energy, "
            "not true Waddington lineage potential, and not functional validation."
        ),
    }

    (outdir / "step76_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step76_report.txt").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("---- audit ----")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("=" * 100)
    print("DONE Step76")
    print("=" * 100)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
