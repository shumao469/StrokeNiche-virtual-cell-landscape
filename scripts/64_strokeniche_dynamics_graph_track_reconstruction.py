#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
64_strokeniche_dynamics_graph_track_reconstruction.py

Purpose
-------
Step 64: StrokeNiche dynamics graph / track reconstruction.

This is designed to mimic the logic of UNAGI-like disease dynamics graph
construction, but adapted to the current StrokeNiche pipeline.

Core logic
----------
Inputs:
  - D1 / D3 / D7 timepoint
  - latent coordinates and/or strict z_i latent
  - state_group / region_true
  - celltype / RCTD composition
  - repair_score
  - core_probability
  - niche module scores

Steps:
  1. Standardize cell-level table.
  2. Cluster cells within each timepoint.
     Default: KMeans in latent/state/module feature space.
     Optional: state_celltype subgrouping.
  3. Treat each timepoint-cluster as a dynamics graph node.
  4. Compute node-node distances across adjacent timepoints:
       D1->D3 and D3->D7
     using:
       - symmetric Gaussian KL divergence
       - Gaussian 2-Wasserstein distance
       - Mahalanobis / centroid distance
       - module/state-score distance
  5. Min-max normalize distances within each adjacent timepoint pair.
  6. Construct dynamic edges by selecting top-k closest outgoing/incoming links.
  7. Reconstruct D1->D3->D7 tracks.
  8. Export nodes / edges / tracks / cell assignments and a publication-style figure.

Important
---------
This is a dynamics graph / pseudo-progression track analysis.
It is NOT true 3D / true 4D tissue reconstruction.

Outputs
-------
results/step8_strokeniche_perturbmap/dynamics_graph_64/
  step64_modeling_input_used.csv
  step64_clustered_cells.csv
  step64_dynamics_graph_nodes.csv
  step64_dynamics_graph_edges_all.csv
  step64_dynamics_graph_edges.csv
  step64_track_summary.csv
  step64_track_assignment_by_cell.csv
  step64_track_module_matrix.csv
  step64_distance_feature_metadata.json
  step64_report.json
  step64_report.txt
  Fig_StrokeNiche_DynamicsGraph.pdf/svg/png
"""

from pathlib import Path
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

try:
    from scipy.linalg import sqrtm
    SCIPY_AVAILABLE = True
except Exception:
    sqrtm = None
    SCIPY_AVAILABLE = False


BASE = Path("/mnt/h/vir/ST")

DEFAULT_INPUT = BASE / "results/step8_strokeniche_perturbmap/generalization_62/step62_modeling_input_table.csv"
DEFAULT_COORD = BASE / "results/figure5_trajectory/figures/fig5_strokeniche_trajectory_latent_state_coordinates.csv"
DEFAULT_TRUE_Z = BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54/adapter_true_hidden_latent_for_neighbor_head.csv"
DEFAULT_OUT = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def read_csv(path, required=True):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None
    return pd.read_csv(p, low_memory=False)


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def minmax(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn = s.min()
    mx = s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def extract_timepoint(x):
    if pd.isna(x):
        return ""
    s = str(x).upper()

    m = re.search(r"(?:^|[^A-Z0-9])D\s*([0-9]{1,2})(?:$|[^A-Z0-9])", s)
    if m:
        return f"D{int(m.group(1))}"

    m = re.search(r"DAY\s*([0-9]{1,2})", s)
    if m:
        return f"D{int(m.group(1))}"

    m = re.search(r"TIMEPOINT[_\-\s]*([0-9]{1,2})", s)
    if m:
        return f"D{int(m.group(1))}"

    return ""


def day_from_timepoint(tp):
    s = str(tp).upper()
    m = re.search(r"D\s*([0-9]{1,2})", s)
    if m:
        return int(m.group(1))
    m = re.search(r"([0-9]{1,2})", s)
    if m:
        return int(m.group(1))
    return np.nan


def detect_timepoint(df):
    if "timepoint" in df.columns:
        tp = df["timepoint"].map(extract_timepoint)
        if tp.astype(str).ne("").sum() > 0:
            return tp.replace("", np.nan)

    out = pd.Series("", index=df.index, dtype=object)

    for c in [
        "time_point", "time", "sample", "sample_id", "RCTD_sample",
        "batch", "orig.ident", "orig_ident"
    ]:
        if c in df.columns:
            v = df[c].map(extract_timepoint)
            out = out.where(out.astype(str).ne(""), v)

    if "obs_name" in df.columns:
        v = df["obs_name"].map(extract_timepoint)
        out = out.where(out.astype(str).ne(""), v)

    return out.replace("", np.nan)


def normalize_state(x):
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    if any(k in s for k in ["lesion_core", "lesion", "core_like", "core"]):
        return "lesion-core-like"
    if any(k in s for k in ["peri_infarct", "peri", "penumbra"]):
        return "peri-infarct"
    if any(k in s for k in ["remote_like", "remote", "normal", "intact"]):
        return "remote-like"
    return "other"


def detect_state(df):
    for c in ["state_group", "strict_state_group", "region_true", "region", "region_label", "region_manual_final"]:
        if c in df.columns:
            return df[c].map(normalize_state), c
    return pd.Series(["unknown"] * len(df), index=df.index), ""


def detect_celltype(df):
    preferred = [
        "predicted_celltype_filtered",
        "predicted_celltype",
        "dominant_celltype",
        "RCTD_predicted_celltype",
        "celltype",
        "cell_type",
    ]

    for c in preferred:
        if c in df.columns:
            s = df[c].astype(str).replace({"nan": "unknown", "None": "unknown"})
            return s, c

    # Derive dominant celltype from RCTD_prop_* if available.
    prop_cols = [c for c in df.columns if str(c).startswith("RCTD_prop_")]
    prop_cols = [c for c in prop_cols if pd.to_numeric(df[c], errors="coerce").notna().sum() > 0]

    if prop_cols:
        mat = df[prop_cols].apply(pd.to_numeric, errors="coerce")
        idx = mat.values.argmax(axis=1)
        labels = [prop_cols[i].replace("RCTD_prop_", "") for i in idx]
        return pd.Series(labels, index=df.index), "derived_from_RCTD_prop"

    return pd.Series(["unknown"] * len(df), index=df.index), ""


def numeric_series(df, candidates, default=np.nan):
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce"), c
    return pd.Series([default] * len(df), index=df.index, dtype=float), ""


def is_numeric_like(s, frac=0.80):
    x = pd.to_numeric(s, errors="coerce")
    return bool(x.notna().mean() >= frac)


def detect_coord_pair(df):
    pairs = [
        ("strict_coord_x", "strict_coord_y"),
        ("coord_x", "coord_y"),
        ("latent_x", "latent_y"),
        ("umap_1", "umap_2"),
        ("UMAP_1", "UMAP_2"),
        ("UMAP1", "UMAP2"),
        ("x", "y"),
    ]

    for x, y in pairs:
        if x in df.columns and y in df.columns:
            if pd.to_numeric(df[x], errors="coerce").notna().sum() > 0 and pd.to_numeric(df[y], errors="coerce").notna().sum() > 0:
                return x, y

    # Search coordinate-like columns.
    numeric_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c]):
            numeric_cols.append(c)

    coord_like = []
    for c in numeric_cols:
        nc = norm_col(c)
        if any(k in nc for k in ["coord", "umap", "latent_state", "embedding"]):
            if not any(b in nc for b in ["prob", "score", "repair", "core", "peri", "remote"]):
                coord_like.append(c)

    if len(coord_like) >= 2:
        return coord_like[0], coord_like[1]

    raise RuntimeError("Cannot detect latent coordinate columns.")


def maybe_merge_by_obs(df, other_path, suffix, required=False):
    other = read_csv(other_path, required=required)
    if other is None or other.empty:
        return df, {"merged": False, "path": str(other_path), "reason": "missing_or_empty"}

    if "obs_name" not in df.columns or "obs_name" not in other.columns:
        return df, {"merged": False, "path": str(other_path), "reason": "obs_name_missing"}

    df = df.copy()
    other = other.copy()

    df["obs_name"] = df["obs_name"].astype(str)
    other["obs_name"] = other["obs_name"].astype(str)

    # Avoid duplicating existing columns except obs_name.
    keep_cols = ["obs_name"]
    for c in other.columns:
        if c == "obs_name":
            continue
        if c not in df.columns:
            keep_cols.append(c)

    merged = df.merge(other[keep_cols], on="obs_name", how="left", suffixes=("", suffix))

    non_obs = [c for c in keep_cols if c != "obs_name"]
    matched = int(merged[non_obs].notna().any(axis=1).sum()) if non_obs else 0

    return merged, {
        "merged": True,
        "path": str(other_path),
        "other_shape": list(other.shape),
        "added_columns": non_obs[:100],
        "n_added_columns": len(non_obs),
        "matched_rows": matched,
    }


# -----------------------------------------------------------------------------
# Standardization and feature selection
# -----------------------------------------------------------------------------

def module_cols(df):
    cols = []
    for c in df.columns:
        if not str(c).startswith("module_"):
            continue
        nc = norm_col(c)
        if any(b in nc for b in ["obs", "barcode", "timepoint", "coord", "key", "matched", "source", "method"]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() >= 0.50 and x.nunique(dropna=True) > 1:
            cols.append(c)
    return list(dict.fromkeys(cols))


def strict_z_cols(df):
    prefixes = ["adapter_latent_", "true_z_", "z_", "latent_"]
    cols = []
    for c in df.columns:
        if any(str(c).startswith(p) for p in prefixes):
            if str(c) in ["latent_x", "latent_y"]:
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() >= 0.80 and x.nunique(dropna=True) > 1:
                cols.append(c)
    return list(dict.fromkeys(cols))


def state_score_cols(df):
    preferred = [
        "repair_score_std",
        "core_probability_std",
        "peri_probability_std",
        "remote_probability_std",
        "repair_score",
        "core_probability",
        "peri_probability",
        "remote_probability",
        "prob_lesion_core",
        "prob_peri_infarct",
        "prob_remote_like",
        "endothelial_context",
        "astrocyte_context",
        "microglia_context",
        "neuron_context",
        "lesion_niche_context",
        "remote_offtarget_context",
    ]
    cols = []
    for c in preferred:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() >= 0.50 and x.nunique(dropna=True) > 1:
                cols.append(c)
    return list(dict.fromkeys(cols))


def standardize_input(raw, coord_table=None, true_z_table=None, z_pca_dim=10, seed=42):
    df = raw.copy()

    if "obs_name" not in df.columns:
        raise RuntimeError("Input lacks obs_name.")

    df["obs_name"] = df["obs_name"].astype(str)

    # Merge coordinate table first if coordinates are absent.
    try:
        detect_coord_pair(df)
        coord_meta = {"merged": False, "reason": "coordinates_already_present"}
    except Exception:
        df, coord_meta = maybe_merge_by_obs(df, coord_table, suffix="_coord", required=False)

    # Merge strict z_i latent if available.
    df, z_meta = maybe_merge_by_obs(df, true_z_table, suffix="_z", required=False)

    # Timepoint.
    df["timepoint"] = detect_timepoint(df)
    df = df[df["timepoint"].notna()].copy()
    df["timepoint"] = df["timepoint"].astype(str)
    df["day"] = df["timepoint"].map(day_from_timepoint)

    # State/celltype.
    state, state_col = detect_state(df)
    df["state_group_std"] = state

    celltype, celltype_col = detect_celltype(df)
    df["celltype_std"] = celltype.astype(str)

    # Scores.
    repair, repair_col = numeric_series(df, [
        "repair_score", "predicted_repair_score", "repair_axis", "repair_state_score"
    ])
    core, core_col = numeric_series(df, [
        "core_probability", "prob_lesion_core", "lesion_core_probability", "predicted_core_probability"
    ])
    peri, peri_col = numeric_series(df, [
        "peri_probability", "prob_peri_infarct", "peri_infarct_probability", "predicted_peri_probability"
    ])
    remote, remote_col = numeric_series(df, [
        "remote_probability", "prob_remote_like", "remote_like_probability", "predicted_remote_probability"
    ])

    df["repair_score_std"] = repair
    df["core_probability_std"] = core
    df["peri_probability_std"] = peri
    df["remote_probability_std"] = remote

    # Coordinates.
    xcol, ycol = detect_coord_pair(df)
    df["latent_x"] = pd.to_numeric(df[xcol], errors="coerce")
    df["latent_y"] = pd.to_numeric(df[ycol], errors="coerce")

    # Strict z PCA features.
    zcols = strict_z_cols(df)
    z_pca_cols = []
    if zcols and z_pca_dim > 0:
        mat = df[zcols].apply(pd.to_numeric, errors="coerce")
        mat = mat.fillna(mat.median(numeric_only=True))
        n_comp = int(min(z_pca_dim, mat.shape[1], max(1, mat.shape[0] - 1)))
        if n_comp >= 1:
            scaler = StandardScaler()
            z_scaled = scaler.fit_transform(mat)
            pca = PCA(n_components=n_comp, random_state=seed)
            pcs = pca.fit_transform(z_scaled)
            for i in range(n_comp):
                col = f"strict_z_pca_{i+1}"
                df[col] = pcs[:, i]
                z_pca_cols.append(col)
            z_meta["z_pca_dim"] = n_comp
            z_meta["z_pca_explained_variance_sum"] = float(np.sum(pca.explained_variance_ratio_))

    # Keep only valid rows.
    df = df[
        df["day"].notna()
        & df["latent_x"].notna()
        & df["latent_y"].notna()
    ].copy()

    meta = {
        "coord_merge": coord_meta,
        "z_merge": z_meta,
        "state_col": state_col,
        "celltype_col": celltype_col,
        "repair_col": repair_col,
        "core_col": core_col,
        "peri_col": peri_col,
        "remote_col": remote_col,
        "coord_x_col": xcol,
        "coord_y_col": ycol,
        "module_cols": module_cols(df),
        "state_score_cols": state_score_cols(df),
        "strict_z_cols_n": len(zcols),
        "strict_z_pca_cols": z_pca_cols,
    }

    return df, meta


def build_feature_matrix(df, cols):
    mat = df[cols].apply(pd.to_numeric, errors="coerce")
    mat = mat.fillna(mat.median(numeric_only=True))
    # If still missing, fill with zero.
    mat = mat.fillna(0.0)
    return mat


# -----------------------------------------------------------------------------
# Clustering
# -----------------------------------------------------------------------------

def adaptive_k(n, min_k=3, max_k=8, target_cells_per_cluster=450):
    if n < 2 * target_cells_per_cluster:
        return max(2, min_k)
    k = int(round(n / target_cells_per_cluster))
    return int(max(min_k, min(max_k, k)))


def assign_clusters_kmeans(df, feature_cols, min_k=3, max_k=8, target_cells_per_cluster=450, seed=42):
    out = df.copy()
    out["cluster_local"] = ""
    out["cluster_method"] = "kmeans"

    cluster_rows = []

    for tp, sub in out.groupby("timepoint"):
        idx = sub.index
        n = len(sub)
        if n < min_k:
            out.loc[idx, "cluster_local"] = "C00"
            continue

        k = adaptive_k(n, min_k=min_k, max_k=max_k, target_cells_per_cluster=target_cells_per_cluster)
        k = min(k, n)

        X = build_feature_matrix(sub, feature_cols)
        Xs = StandardScaler().fit_transform(X)

        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        lab = km.fit_predict(Xs)

        out.loc[idx, "cluster_local"] = [f"C{int(x):02d}" for x in lab]

        cluster_rows.append({
            "timepoint": tp,
            "n_cells": n,
            "k": k,
            "feature_cols": ";".join(feature_cols),
        })

    out["node_id"] = out["timepoint"].astype(str) + "_" + out["cluster_local"].astype(str)

    return out, pd.DataFrame(cluster_rows)


def assign_clusters_state_celltype(df, min_cells=30):
    out = df.copy()
    combo = out["state_group_std"].astype(str) + "__" + out["celltype_std"].astype(str)

    # Collapse rare groups within each timepoint into state-only group.
    labels = []
    for tp, sub in out.groupby("timepoint"):
        counts = combo.loc[sub.index].value_counts()
        for idx in sub.index:
            lab = combo.loc[idx]
            if counts.get(lab, 0) < min_cells:
                lab = out.loc[idx, "state_group_std"]
            labels.append((idx, lab))

    label_series = pd.Series({idx: lab for idx, lab in labels})
    out["cluster_local"] = label_series.reindex(out.index).astype(str)

    # Make local compact names.
    new_labels = []
    for tp, sub in out.groupby("timepoint"):
        vals = sorted(sub["cluster_local"].unique())
        mapping = {v: f"G{i:02d}" for i, v in enumerate(vals)}
        out.loc[sub.index, "cluster_local_name"] = sub["cluster_local"].map(mapping)
        out.loc[sub.index, "cluster_annotation"] = sub["cluster_local"]

    out["cluster_local"] = out["cluster_local_name"].astype(str)
    out["cluster_method"] = "state_celltype"
    out["node_id"] = out["timepoint"].astype(str) + "_" + out["cluster_local"].astype(str)

    cluster_rows = (
        out.groupby(["timepoint", "cluster_local", "cluster_annotation"])
        .size()
        .reset_index(name="n_cells")
    )

    return out, cluster_rows


def cluster_cells(df, mode, cluster_feature_cols, args):
    if mode == "state_celltype":
        return assign_clusters_state_celltype(df, min_cells=args.min_cluster_cells)

    if mode == "auto":
        # Prefer KMeans unless explicitly asked for state_celltype.
        return assign_clusters_kmeans(
            df,
            feature_cols=cluster_feature_cols,
            min_k=args.min_k,
            max_k=args.max_k,
            target_cells_per_cluster=args.target_cells_per_cluster,
            seed=args.seed,
        )

    if mode == "kmeans":
        return assign_clusters_kmeans(
            df,
            feature_cols=cluster_feature_cols,
            min_k=args.min_k,
            max_k=args.max_k,
            target_cells_per_cluster=args.target_cells_per_cluster,
            seed=args.seed,
        )

    raise ValueError(f"Unknown cluster mode: {mode}")


# -----------------------------------------------------------------------------
# Node summaries
# -----------------------------------------------------------------------------

def proportion_dict(s, prefix, top_n=5):
    vc = s.astype(str).value_counts(normalize=True)
    out = {}
    for k, v in vc.head(top_n).items():
        key = f"{prefix}_prop_{re.sub(r'[^A-Za-z0-9]+', '_', str(k)).strip('_')}"
        out[key] = float(v)
    return out


def make_nodes(clustered, module_features, distance_features):
    rows = []

    for node_id, sub in clustered.groupby("node_id"):
        tp = sub["timepoint"].iloc[0]
        day = day_from_timepoint(tp)

        state_counts = sub["state_group_std"].astype(str).value_counts()
        cell_counts = sub["celltype_std"].astype(str).value_counts()

        dominant_state = state_counts.index[0] if len(state_counts) else "unknown"
        dominant_celltype = cell_counts.index[0] if len(cell_counts) else "unknown"

        row = {
            "node_id": node_id,
            "timepoint": tp,
            "day": day,
            "cluster_local": sub["cluster_local"].iloc[0],
            "cluster_method": sub["cluster_method"].iloc[0],
            "n_cells": int(len(sub)),
            "latent_x": float(sub["latent_x"].mean()),
            "latent_y": float(sub["latent_y"].mean()),
            "latent_x_sd": float(sub["latent_x"].std()),
            "latent_y_sd": float(sub["latent_y"].std()),
            "repair_score_mean": float(sub["repair_score_std"].mean()),
            "repair_score_sd": float(sub["repair_score_std"].std()),
            "core_probability_mean": float(sub["core_probability_std"].mean()),
            "core_probability_sd": float(sub["core_probability_std"].std()),
            "peri_probability_mean": float(sub["peri_probability_std"].mean()),
            "peri_probability_sd": float(sub["peri_probability_std"].std()),
            "remote_probability_mean": float(sub["remote_probability_std"].mean()),
            "remote_probability_sd": float(sub["remote_probability_std"].std()),
            "dominant_state": dominant_state,
            "dominant_state_fraction": float(state_counts.iloc[0] / len(sub)) if len(state_counts) else np.nan,
            "dominant_celltype": dominant_celltype,
            "dominant_celltype_fraction": float(cell_counts.iloc[0] / len(sub)) if len(cell_counts) else np.nan,
            "cell_indices": ";".join(map(str, sub.index.tolist()[:200])),
        }

        row.update(proportion_dict(sub["state_group_std"], "state", top_n=10))
        row.update(proportion_dict(sub["celltype_std"], "celltype", top_n=10))

        for c in module_features:
            if c in sub.columns:
                row[f"{c}_mean"] = float(pd.to_numeric(sub[c], errors="coerce").mean())

        for c in distance_features:
            if c in sub.columns:
                row[f"feature_mean__{c}"] = float(pd.to_numeric(sub[c], errors="coerce").mean())

        rows.append(row)

    nodes = pd.DataFrame(rows)
    return nodes.sort_values(["day", "node_id"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Gaussian distribution distances
# -----------------------------------------------------------------------------

def gaussian_stats(X, reg=1e-4):
    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    mu = np.nanmean(X, axis=0)

    # Fill NaNs by mean.
    X2 = X.copy()
    inds = np.where(~np.isfinite(X2))
    if inds[0].size:
        X2[inds] = np.take(mu, inds[1])

    if X2.shape[0] <= 1:
        cov = np.eye(X2.shape[1]) * reg
    else:
        cov = np.cov(X2, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[float(cov)]])

    cov = np.asarray(cov, dtype=float)
    cov = cov + np.eye(cov.shape[0]) * reg

    return mu, cov


def gaussian_kl(mu0, cov0, mu1, cov1):
    """
    KL(N0 || N1)
    """
    k = len(mu0)

    inv1 = np.linalg.pinv(cov1)

    diff = (mu1 - mu0).reshape(-1, 1)

    sign0, logdet0 = np.linalg.slogdet(cov0)
    sign1, logdet1 = np.linalg.slogdet(cov1)

    if sign0 <= 0 or sign1 <= 0:
        logdet0 = np.log(max(np.linalg.det(cov0 + np.eye(k) * 1e-4), 1e-12))
        logdet1 = np.log(max(np.linalg.det(cov1 + np.eye(k) * 1e-4), 1e-12))

    val = 0.5 * (
        np.trace(inv1 @ cov0)
        + float(diff.T @ inv1 @ diff)
        - k
        + (logdet1 - logdet0)
    )

    if not np.isfinite(val):
        return np.nan

    return float(max(val, 0.0))


def symmetric_kl(mu0, cov0, mu1, cov1):
    a = gaussian_kl(mu0, cov0, mu1, cov1)
    b = gaussian_kl(mu1, cov1, mu0, cov0)
    return float(np.nanmean([a, b]))


def gaussian_wasserstein2(mu0, cov0, mu1, cov1):
    diff = mu0 - mu1
    mean_term = float(diff @ diff)

    if SCIPY_AVAILABLE and sqrtm is not None:
        try:
            sqrt_cov1 = sqrtm(cov1)
            middle = sqrt_cov1 @ cov0 @ sqrt_cov1
            sqrt_middle = sqrtm(middle)
            if np.iscomplexobj(sqrt_middle):
                sqrt_middle = sqrt_middle.real
            cov_term = np.trace(cov0 + cov1 - 2 * sqrt_middle)
            val = mean_term + float(cov_term)
            if np.isfinite(val):
                return float(np.sqrt(max(val, 0.0)))
        except Exception:
            pass

    # fallback: diagonal approximation
    d0 = np.diag(cov0)
    d1 = np.diag(cov1)
    cov_term = np.sum(d0 + d1 - 2 * np.sqrt(np.maximum(d0 * d1, 0)))
    val = mean_term + cov_term
    return float(np.sqrt(max(val, 0.0)))


def mahalanobis_distance(mu0, cov0, mu1, cov1):
    pooled = (cov0 + cov1) / 2.0
    inv = np.linalg.pinv(pooled)
    diff = (mu0 - mu1).reshape(-1, 1)
    val = float(diff.T @ inv @ diff)
    return float(np.sqrt(max(val, 0.0))) if np.isfinite(val) else np.nan


def euclidean_distance(mu0, mu1):
    return float(np.linalg.norm(mu0 - mu1))


def node_distribution_stats(clustered, distance_features, node_id):
    sub = clustered[clustered["node_id"].eq(node_id)]
    X = build_feature_matrix(sub, distance_features)
    return gaussian_stats(X.values, reg=1e-4)


def module_vector(clustered, module_features, fallback_features, node_id):
    sub = clustered[clustered["node_id"].eq(node_id)]

    cols = [c for c in module_features if c in sub.columns]
    if not cols:
        cols = [c for c in fallback_features if c in sub.columns]

    if not cols:
        return np.array([]), []

    vals = []
    for c in cols:
        vals.append(float(pd.to_numeric(sub[c], errors="coerce").mean()))

    return np.asarray(vals, dtype=float), cols


def module_distance(v0, v1):
    if len(v0) == 0 or len(v1) == 0 or len(v0) != len(v1):
        return np.nan
    return float(np.linalg.norm(v0 - v1))


# -----------------------------------------------------------------------------
# Edge construction
# -----------------------------------------------------------------------------

def adjacent_timepoint_pairs(timepoints):
    tps = sorted(timepoints, key=day_from_timepoint)
    pairs = []
    for i in range(len(tps) - 1):
        pairs.append((tps[i], tps[i + 1]))
    return pairs


def compute_all_edges(clustered, nodes, distance_features, module_features, args):
    rows = []

    fallback_features = ["repair_score_std", "core_probability_std", "peri_probability_std", "remote_probability_std"]

    stats_cache = {}
    module_cache = {}

    for node_id in nodes["node_id"]:
        stats_cache[node_id] = node_distribution_stats(clustered, distance_features, node_id)
        module_cache[node_id] = module_vector(clustered, module_features, fallback_features, node_id)

    for tp0, tp1 in adjacent_timepoint_pairs(nodes["timepoint"].unique()):
        n0 = nodes[nodes["timepoint"].eq(tp0)].copy()
        n1 = nodes[nodes["timepoint"].eq(tp1)].copy()

        for _, a in n0.iterrows():
            for _, b in n1.iterrows():
                id0 = a["node_id"]
                id1 = b["node_id"]

                mu0, cov0 = stats_cache[id0]
                mu1, cov1 = stats_cache[id1]

                skl = symmetric_kl(mu0, cov0, mu1, cov1)
                wass = gaussian_wasserstein2(mu0, cov0, mu1, cov1)
                maha = mahalanobis_distance(mu0, cov0, mu1, cov1)
                euc = euclidean_distance(mu0, mu1)

                mv0, mcols0 = module_cache[id0]
                mv1, mcols1 = module_cache[id1]
                mdist = module_distance(mv0, mv1)

                d_repair = b["repair_score_mean"] - a["repair_score_mean"]
                d_core = b["core_probability_mean"] - a["core_probability_mean"]
                d_remote = b["remote_probability_mean"] - a["remote_probability_mean"]

                state_same = a["dominant_state"] == b["dominant_state"]
                celltype_same = a["dominant_celltype"] == b["dominant_celltype"]

                rows.append({
                    "edge_id": f"{id0}__to__{id1}",
                    "from_node": id0,
                    "to_node": id1,
                    "from_timepoint": tp0,
                    "to_timepoint": tp1,
                    "from_day": day_from_timepoint(tp0),
                    "to_day": day_from_timepoint(tp1),
                    "from_n_cells": int(a["n_cells"]),
                    "to_n_cells": int(b["n_cells"]),
                    "from_state": a["dominant_state"],
                    "to_state": b["dominant_state"],
                    "from_celltype": a["dominant_celltype"],
                    "to_celltype": b["dominant_celltype"],
                    "state_same": bool(state_same),
                    "celltype_same": bool(celltype_same),
                    "symmetric_kl": skl,
                    "gaussian_wasserstein": wass,
                    "mahalanobis": maha,
                    "centroid_euclidean": euc,
                    "module_distance": mdist,
                    "delta_repair_score": d_repair,
                    "delta_core_probability": d_core,
                    "delta_remote_probability": d_remote,
                    "module_cols_used": ";".join(mcols0),
                })

    edges = pd.DataFrame(rows)

    if edges.empty:
        return edges

    # Normalize distances separately for each adjacent transition.
    normed = []
    for pair, sub in edges.groupby(["from_timepoint", "to_timepoint"]):
        sub = sub.copy()

        for c in ["symmetric_kl", "gaussian_wasserstein", "mahalanobis", "centroid_euclidean", "module_distance"]:
            sub[f"{c}_norm"] = minmax(sub[c])

        sub["combined_distance_norm"] = (
            args.w_kl * sub["symmetric_kl_norm"]
            + args.w_wasserstein * sub["gaussian_wasserstein_norm"]
            + args.w_mahalanobis * sub["mahalanobis_norm"]
            + args.w_centroid * sub["centroid_euclidean_norm"]
            + args.w_module * sub["module_distance_norm"]
        )

        denom = args.w_kl + args.w_wasserstein + args.w_mahalanobis + args.w_centroid + args.w_module
        if denom > 0:
            sub["combined_distance_norm"] = sub["combined_distance_norm"] / denom

        # Continuity bonuses are subtracted from distance.
        sub["state_continuity_bonus"] = np.where(sub["state_same"], args.state_continuity_bonus, 0.0)
        sub["celltype_continuity_bonus"] = np.where(sub["celltype_same"], args.celltype_continuity_bonus, 0.0)

        sub["combined_distance_adjusted"] = (
            sub["combined_distance_norm"]
            - sub["state_continuity_bonus"]
            - sub["celltype_continuity_bonus"]
        ).clip(lower=0.0)

        sub["edge_score"] = 1.0 - minmax(sub["combined_distance_adjusted"])

        # Ranks.
        sub["edge_rank_from_source"] = sub.groupby("from_node")["combined_distance_adjusted"].rank(method="first", ascending=True)
        sub["edge_rank_to_target"] = sub.groupby("to_node")["combined_distance_adjusted"].rank(method="first", ascending=True)
        sub["mutual_best"] = (sub["edge_rank_from_source"].eq(1) & sub["edge_rank_to_target"].eq(1))

        normed.append(sub)

    edges = pd.concat(normed, ignore_index=True)

    return edges.sort_values(["from_day", "from_node", "combined_distance_adjusted"]).reset_index(drop=True)


def select_edges(edges, top_k_out=2, top_k_in=2, edge_score_quantile=0.0):
    if edges.empty:
        return edges

    selected = edges[
        (edges["edge_rank_from_source"] <= top_k_out)
        | (edges["edge_rank_to_target"] <= top_k_in)
        | (edges["mutual_best"] == True)
    ].copy()

    if edge_score_quantile > 0:
        keep = []
        for pair, sub in selected.groupby(["from_timepoint", "to_timepoint"]):
            q = sub["edge_score"].quantile(edge_score_quantile)
            keep.append(sub[sub["edge_score"] >= q])
        selected = pd.concat(keep, ignore_index=True) if keep else selected

    selected["selected_for_graph"] = True
    return selected.sort_values(["from_day", "from_node", "combined_distance_adjusted"]).reset_index(drop=True)


# -----------------------------------------------------------------------------
# Track reconstruction
# -----------------------------------------------------------------------------

def track_label_from_nodes(track_nodes, nodes):
    sub = nodes[nodes["node_id"].isin(track_nodes)].copy().sort_values("day")
    states = sub["dominant_state"].astype(str).tolist()
    celltypes = sub["dominant_celltype"].astype(str).tolist()

    repair_delta = safe_float(sub["repair_score_mean"].iloc[-1] - sub["repair_score_mean"].iloc[0]) if len(sub) >= 2 else np.nan
    core_delta = safe_float(sub["core_probability_mean"].iloc[-1] - sub["core_probability_mean"].iloc[0]) if len(sub) >= 2 else np.nan

    celltype_text = " ".join(celltypes).lower()

    # Module means if available.
    module_cols = [c for c in nodes.columns if c.startswith("module_") and c.endswith("_mean")]
    module_trend = {}
    for c in module_cols:
        if len(sub) >= 2:
            module_trend[c] = safe_float(sub[c].iloc[-1] - sub[c].iloc[0])

    if any("microglia" in x for x in celltypes) or any("inflamm" in k.lower() and v > 0 for k, v in module_trend.items()):
        return "microglia-inflammatory track"

    if any("endo" in x.lower() for x in celltypes) or any(("bbb" in k.lower() or "barrier" in k.lower()) and v > 0 for k, v in module_trend.items()):
        return "vascular-barrier-fragile track"

    if any(("repair_ecm" in k.lower() or "repair" in k.lower()) and v > 0 for k, v in module_trend.items()):
        return "repair-permissive ECM track"

    if states[0] == "lesion-core-like" and states[-1] in ["peri-infarct", "remote-like"]:
        return "core-to-repair/remote transition track"

    if states[0] == "lesion-core-like" and states[-1] == "lesion-core-like":
        return "persistent core-like injury track"

    if repair_delta > 0 and core_delta < 0:
        return "repair-gain core-reduction track"

    if repair_delta > 0:
        return "repair-gain track"

    return "state-transition track"


def reconstruct_tracks(nodes, selected_edges, max_tracks=30):
    if selected_edges.empty:
        return pd.DataFrame()

    tps = sorted(nodes["timepoint"].unique(), key=day_from_timepoint)

    rows = []

    if len(tps) < 2:
        return pd.DataFrame()

    if len(tps) == 2:
        for _, e in selected_edges.iterrows():
            track_nodes = [e["from_node"], e["to_node"]]
            rows.append({
                "track_id": f"Track_{len(rows)+1:03d}",
                "track_nodes": "->".join(track_nodes),
                "start_node": track_nodes[0],
                "mid_node": "",
                "end_node": track_nodes[-1],
                "n_steps": 1,
                "mean_edge_score": safe_float(e["edge_score"]),
                "mean_combined_distance": safe_float(e["combined_distance_adjusted"]),
                "min_edge_score": safe_float(e["edge_score"]),
            })
    else:
        # Build D1->D3->D7 paths using adjacent selected edges.
        first_pair = (tps[0], tps[1])
        second_pair = (tps[1], tps[2])

        e01 = selected_edges[
            selected_edges["from_timepoint"].eq(first_pair[0])
            & selected_edges["to_timepoint"].eq(first_pair[1])
        ].copy()

        e12 = selected_edges[
            selected_edges["from_timepoint"].eq(second_pair[0])
            & selected_edges["to_timepoint"].eq(second_pair[1])
        ].copy()

        for _, a in e01.iterrows():
            candidates = e12[e12["from_node"].eq(a["to_node"])]
            for _, b in candidates.iterrows():
                track_nodes = [a["from_node"], a["to_node"], b["to_node"]]
                scores = [safe_float(a["edge_score"]), safe_float(b["edge_score"])]
                dists = [safe_float(a["combined_distance_adjusted"]), safe_float(b["combined_distance_adjusted"])]

                rows.append({
                    "track_id": f"Track_{len(rows)+1:03d}",
                    "track_nodes": "->".join(track_nodes),
                    "start_node": track_nodes[0],
                    "mid_node": track_nodes[1],
                    "end_node": track_nodes[-1],
                    "n_steps": 2,
                    "edge_ids": f"{a['edge_id']};{b['edge_id']}",
                    "mean_edge_score": float(np.nanmean(scores)),
                    "mean_combined_distance": float(np.nanmean(dists)),
                    "min_edge_score": float(np.nanmin(scores)),
                    "edge_score_product": float(np.nanprod(scores)),
                })

    tracks = pd.DataFrame(rows)

    if tracks.empty:
        return tracks

    # Add node-derived summaries.
    track_rows = []
    for _, tr in tracks.iterrows():
        node_ids = str(tr["track_nodes"]).split("->")
        sub = nodes[nodes["node_id"].isin(node_ids)].copy().sort_values("day")

        if sub.empty:
            continue

        repair_delta = safe_float(sub["repair_score_mean"].iloc[-1] - sub["repair_score_mean"].iloc[0])
        core_delta = safe_float(sub["core_probability_mean"].iloc[-1] - sub["core_probability_mean"].iloc[0])
        peri_delta = safe_float(sub["peri_probability_mean"].iloc[-1] - sub["peri_probability_mean"].iloc[0])
        remote_delta = safe_float(sub["remote_probability_mean"].iloc[-1] - sub["remote_probability_mean"].iloc[0])

        n_cells_sum = int(sub["n_cells"].sum())
        state_seq = "->".join(sub["dominant_state"].astype(str).tolist())
        celltype_seq = "->".join(sub["dominant_celltype"].astype(str).tolist())

        row = tr.to_dict()
        row.update({
            "track_label": track_label_from_nodes(node_ids, nodes),
            "state_sequence": state_seq,
            "celltype_sequence": celltype_seq,
            "n_cells_sum": n_cells_sum,
            "delta_repair_score": repair_delta,
            "delta_core_probability": core_delta,
            "delta_peri_probability": peri_delta,
            "delta_remote_probability": remote_delta,
            "start_state": sub["dominant_state"].iloc[0],
            "end_state": sub["dominant_state"].iloc[-1],
            "start_celltype": sub["dominant_celltype"].iloc[0],
            "end_celltype": sub["dominant_celltype"].iloc[-1],
        })

        # Track rescue-like score: higher repair, lower core, higher remote/peri, better edge score.
        row["track_repair_core_score"] = (
            0.35 * max(repair_delta, 0)
            + 0.35 * max(-core_delta, 0)
            + 0.15 * max(peri_delta, 0)
            + 0.15 * max(remote_delta, 0)
        )

        row["track_priority_score"] = (
            0.60 * safe_float(row.get("mean_edge_score", 0))
            + 0.40 * row["track_repair_core_score"]
        )

        track_rows.append(row)

    tracks = pd.DataFrame(track_rows)

    if tracks.empty:
        return tracks

    tracks = tracks.sort_values(
        ["track_priority_score", "mean_edge_score", "n_cells_sum"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    tracks["track_rank"] = np.arange(1, len(tracks) + 1)

    if max_tracks and len(tracks) > max_tracks:
        tracks = tracks.head(max_tracks).copy()

    return tracks


def assign_cells_to_tracks(clustered, tracks):
    if tracks is None or tracks.empty:
        return pd.DataFrame()

    rows = []

    # Create node -> tracks mapping.
    node_to_tracks = {}
    for _, tr in tracks.iterrows():
        node_ids = str(tr["track_nodes"]).split("->")
        for node_id in node_ids:
            node_to_tracks.setdefault(node_id, []).append((tr["track_id"], tr["track_rank"], tr["track_label"]))

    for idx, row in clustered.iterrows():
        node_id = row["node_id"]
        if node_id not in node_to_tracks:
            continue
        for track_id, track_rank, track_label in node_to_tracks[node_id]:
            rows.append({
                "obs_name": row["obs_name"],
                "cell_index": idx,
                "timepoint": row["timepoint"],
                "day": row["day"],
                "node_id": node_id,
                "cluster_local": row["cluster_local"],
                "state_group": row["state_group_std"],
                "celltype": row["celltype_std"],
                "track_id": track_id,
                "track_rank": track_rank,
                "track_label": track_label,
                "repair_score": row["repair_score_std"],
                "core_probability": row["core_probability_std"],
                "peri_probability": row["peri_probability_std"],
                "remote_probability": row["remote_probability_std"],
                "latent_x": row["latent_x"],
                "latent_y": row["latent_y"],
            })

    return pd.DataFrame(rows)


def build_track_module_matrix(nodes, tracks):
    if tracks is None or tracks.empty:
        return pd.DataFrame()

    module_mean_cols = [c for c in nodes.columns if c.startswith("module_") and c.endswith("_mean")]

    if not module_mean_cols:
        return pd.DataFrame()

    rows = []
    for _, tr in tracks.iterrows():
        node_ids = str(tr["track_nodes"]).split("->")
        sub = nodes[nodes["node_id"].isin(node_ids)].copy().sort_values("day")
        if sub.empty:
            continue

        row = {
            "track_id": tr["track_id"],
            "track_rank": tr["track_rank"],
            "track_label": tr["track_label"],
            "track_nodes": tr["track_nodes"],
        }

        for c in module_mean_cols:
            vals = pd.to_numeric(sub[c], errors="coerce").values
            if len(vals) >= 2:
                row[c.replace("_mean", "_delta_end_minus_start")] = safe_float(vals[-1] - vals[0])
                row[c.replace("_mean", "_mean_over_track")] = safe_float(np.nanmean(vals))
            else:
                row[c.replace("_mean", "_delta_end_minus_start")] = np.nan
                row[c.replace("_mean", "_mean_over_track")] = safe_float(np.nanmean(vals))

        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def state_color_map():
    return {
        "lesion-core-like": "#D84A3A",
        "peri-infarct": "#E7A23B",
        "remote-like": "#4D91C6",
        "other": "#8A8A8A",
        "unknown": "#8A8A8A",
    }


def plot_dynamics_graph(clustered, nodes, selected_edges, tracks, outdir, dpi=600, max_scatter=6000, seed=42):
    outbase = Path(outdir) / "Fig_StrokeNiche_DynamicsGraph"

    cmap_state = state_color_map()

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.30)

    # Panel A: graph layered by timepoint.
    axA = fig.add_subplot(gs[0, 0])
    axA.set_title("A | StrokeNiche dynamics graph", loc="left", fontsize=12, fontweight="bold")

    plot_nodes = nodes.copy()
    tps = sorted(plot_nodes["timepoint"].unique(), key=day_from_timepoint)
    x_map = {tp: i for i, tp in enumerate(tps)}

    # vertical ordering by latent_y or cluster index within each timepoint
    y_positions = {}
    for tp in tps:
        sub = plot_nodes[plot_nodes["timepoint"].eq(tp)].sort_values("latent_y")
        if len(sub) == 1:
            ys = [0.5]
        else:
            ys = np.linspace(0.1, 0.9, len(sub))
        for y, (_, r) in zip(ys, sub.iterrows()):
            y_positions[r["node_id"]] = y

    # Draw edges.
    if selected_edges is not None and not selected_edges.empty:
        for _, e in selected_edges.iterrows():
            if e["from_node"] not in y_positions or e["to_node"] not in y_positions:
                continue
            x0 = x_map[e["from_timepoint"]]
            x1 = x_map[e["to_timepoint"]]
            y0 = y_positions[e["from_node"]]
            y1 = y_positions[e["to_node"]]
            lw = 0.6 + 3.2 * safe_float(e["edge_score"])
            alpha = 0.25 + 0.55 * safe_float(e["edge_score"])
            axA.plot([x0, x1], [y0, y1], linewidth=lw, alpha=alpha, color="#6E7781", zorder=1)

    # Draw nodes.
    max_n = max(plot_nodes["n_cells"].max(), 1)
    for _, r in plot_nodes.iterrows():
        x = x_map[r["timepoint"]]
        y = y_positions[r["node_id"]]
        size = 80 + 520 * (r["n_cells"] / max_n)
        color = cmap_state.get(r["dominant_state"], "#8A8A8A")
        axA.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=1.0, zorder=3)
        label = f"{r['cluster_local']}\n{r['dominant_state'].replace('-like','')}"
        axA.text(x, y, label, ha="center", va="center", fontsize=6, zorder=4)

    axA.set_xticks(list(x_map.values()))
    axA.set_xticklabels(tps, fontsize=10, fontweight="bold")
    axA.set_xlim(-0.4, len(tps) - 0.6)
    axA.set_ylim(0, 1)
    axA.set_yticks([])
    axA.set_xlabel("Timepoint")
    axA.text(
        0.01, -0.12,
        "Node size = cell count; edge width = latent/module similarity",
        transform=axA.transAxes,
        fontsize=8,
    )

    # Panel B: latent space with selected edges among centroids.
    axB = fig.add_subplot(gs[0, 1])
    axB.set_title("B | Latent-space node connectivity", loc="left", fontsize=12, fontweight="bold")

    if len(clustered) > max_scatter:
        scatter_df = clustered.sample(n=max_scatter, random_state=seed)
    else:
        scatter_df = clustered.copy()

    for state, sub in scatter_df.groupby("state_group_std"):
        axB.scatter(
            sub["latent_x"],
            sub["latent_y"],
            s=4,
            alpha=0.14,
            color=cmap_state.get(state, "#8A8A8A"),
            label=state,
        )

    if selected_edges is not None and not selected_edges.empty:
        node_pos = nodes.set_index("node_id")[["latent_x", "latent_y"]].to_dict(orient="index")
        for _, e in selected_edges.iterrows():
            if e["from_node"] not in node_pos or e["to_node"] not in node_pos:
                continue
            p0 = node_pos[e["from_node"]]
            p1 = node_pos[e["to_node"]]
            dx = p1["latent_x"] - p0["latent_x"]
            dy = p1["latent_y"] - p0["latent_y"]
            axB.arrow(
                p0["latent_x"], p0["latent_y"], dx, dy,
                length_includes_head=True,
                head_width=0.035,
                alpha=0.35 + 0.45 * safe_float(e["edge_score"]),
                linewidth=0.6 + 1.8 * safe_float(e["edge_score"]),
                color="#333333",
                zorder=2,
            )

    axB.scatter(nodes["latent_x"], nodes["latent_y"], s=80, color="white", edgecolor="black", linewidth=0.8, zorder=4)
    for _, r in nodes.iterrows():
        axB.text(r["latent_x"], r["latent_y"], r["node_id"], fontsize=6, ha="center", va="center", zorder=5)

    axB.set_xlabel("Latent coordinate 1")
    axB.set_ylabel("Latent coordinate 2")
    axB.legend(frameon=False, fontsize=7, loc="best")

    # Panel C: top tracks.
    axC = fig.add_subplot(gs[1, 0])
    axC.set_title("C | Reconstructed D1-D3-D7 tracks", loc="left", fontsize=12, fontweight="bold")

    if tracks is not None and not tracks.empty:
        top = tracks.sort_values("track_rank").head(12).copy()
        labels = top["track_id"].astype(str) + " | " + top["track_label"].astype(str)
        axC.barh(labels[::-1], top["track_priority_score"].values[::-1])
        axC.set_xlabel("Track priority score")
        axC.tick_params(axis="y", labelsize=7)
    else:
        axC.text(0.5, 0.5, "No tracks reconstructed", ha="center", va="center", transform=axC.transAxes)
        axC.set_axis_off()

    # Panel D: repair-core trajectory readout by track.
    axD = fig.add_subplot(gs[1, 1])
    axD.set_title("D | Track repair-core shifts", loc="left", fontsize=12, fontweight="bold")

    if tracks is not None and not tracks.empty:
        top = tracks.sort_values("track_rank").head(15).copy()
        x = top["delta_core_probability"]
        y = top["delta_repair_score"]
        sizes = 50 + 260 * minmax(top["mean_edge_score"])
        axD.scatter(x, y, s=sizes, alpha=0.82, edgecolor="black", linewidth=0.5)
        for _, r in top.iterrows():
            axD.text(r["delta_core_probability"], r["delta_repair_score"], r["track_id"].replace("Track_", "T"), fontsize=7)

        axD.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        axD.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        axD.set_xlabel("Δ core probability")
        axD.set_ylabel("Δ repair score")
        axD.text(
            0.02, 0.98,
            "Upper-left quadrant:\nrepair gain + core reduction",
            transform=axD.transAxes,
            va="top",
            fontsize=8,
        )
    else:
        axD.text(0.5, 0.5, "No tracks reconstructed", ha="center", va="center", transform=axD.transAxes)
        axD.set_axis_off()

    for ax in [axA, axB, axC, axD]:
        ax.grid(linestyle="--", linewidth=0.35, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Step 64 | StrokeNiche dynamics graph and progression tracks", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "pdf": str(outbase.with_suffix(".pdf")),
        "svg": str(outbase.with_suffix(".svg")),
        "png": str(outbase.with_suffix(".png")),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--coord_table", default=str(DEFAULT_COORD))
    ap.add_argument("--true_z", default=str(DEFAULT_TRUE_Z))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--cluster_mode", default="kmeans", choices=["auto", "kmeans", "state_celltype"])
    ap.add_argument("--min_k", type=int, default=3)
    ap.add_argument("--max_k", type=int, default=7)
    ap.add_argument("--target_cells_per_cluster", type=int, default=450)
    ap.add_argument("--min_cluster_cells", type=int, default=30)

    ap.add_argument("--z_pca_dim", type=int, default=10)
    ap.add_argument("--distance_use_z_pca", action="store_true")
    ap.add_argument("--distance_use_modules", action="store_true")
    ap.add_argument("--distance_use_state_scores", action="store_true")

    ap.add_argument("--top_k_out", type=int, default=2)
    ap.add_argument("--top_k_in", type=int, default=2)
    ap.add_argument("--edge_score_quantile", type=float, default=0.0)
    ap.add_argument("--max_tracks", type=int, default=30)

    ap.add_argument("--w_kl", type=float, default=0.30)
    ap.add_argument("--w_wasserstein", type=float, default=0.25)
    ap.add_argument("--w_mahalanobis", type=float, default=0.15)
    ap.add_argument("--w_centroid", type=float, default=0.10)
    ap.add_argument("--w_module", type=float, default=0.20)

    ap.add_argument("--state_continuity_bonus", type=float, default=0.03)
    ap.add_argument("--celltype_continuity_bonus", type=float, default=0.02)

    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 64: StrokeNiche dynamics graph / track reconstruction")
    log("=" * 100)
    log(f"input={args.input}")
    log(f"coord_table={args.coord_table}")
    log(f"true_z={args.true_z}")
    log(f"outdir={outdir}")
    log(f"cluster_mode={args.cluster_mode}")

    raw = read_csv(args.input, required=True)

    df, input_meta = standardize_input(
        raw,
        coord_table=args.coord_table,
        true_z_table=args.true_z,
        z_pca_dim=args.z_pca_dim,
        seed=args.seed,
    )

    df.to_csv(outdir / "step64_modeling_input_used.csv", index=False)

    # Feature sets.
    modules = module_cols(df)
    state_scores = [
        c for c in [
            "repair_score_std",
            "core_probability_std",
            "peri_probability_std",
            "remote_probability_std",
            "endothelial_context",
            "astrocyte_context",
            "microglia_context",
            "neuron_context",
            "lesion_niche_context",
            "remote_offtarget_context",
        ]
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 0
    ]

    z_pca_cols = input_meta.get("strict_z_pca_cols", [])
    coord_cols = ["latent_x", "latent_y"]

    # Clustering feature space.
    cluster_features = coord_cols + state_scores
    if modules:
        cluster_features += modules
    if z_pca_cols:
        cluster_features += z_pca_cols

    cluster_features = list(dict.fromkeys([c for c in cluster_features if c in df.columns]))

    if len(cluster_features) < 2:
        raise RuntimeError("Too few clustering features.")

    clustered, cluster_meta = cluster_cells(
        df,
        mode=args.cluster_mode,
        cluster_feature_cols=cluster_features,
        args=args,
    )

    clustered.to_csv(outdir / "step64_clustered_cells.csv", index=False)
    cluster_meta.to_csv(outdir / "step64_cluster_metadata.csv", index=False)

    # Distance feature space.
    distance_features = coord_cols.copy()

    if args.distance_use_state_scores or True:
        distance_features += state_scores

    if args.distance_use_modules and modules:
        distance_features += modules

    if args.distance_use_z_pca and z_pca_cols:
        distance_features += z_pca_cols

    # Always keep at least coordinate + state score distance.
    distance_features = list(dict.fromkeys([c for c in distance_features if c in clustered.columns]))

    if len(distance_features) < 2:
        raise RuntimeError("Too few distance features.")

    nodes = make_nodes(
        clustered=clustered,
        module_features=modules,
        distance_features=distance_features,
    )
    nodes.to_csv(outdir / "step64_dynamics_graph_nodes.csv", index=False)

    all_edges = compute_all_edges(
        clustered=clustered,
        nodes=nodes,
        distance_features=distance_features,
        module_features=modules,
        args=args,
    )
    all_edges.to_csv(outdir / "step64_dynamics_graph_edges_all.csv", index=False)

    selected_edges = select_edges(
        all_edges,
        top_k_out=args.top_k_out,
        top_k_in=args.top_k_in,
        edge_score_quantile=args.edge_score_quantile,
    )
    selected_edges.to_csv(outdir / "step64_dynamics_graph_edges.csv", index=False)

    tracks = reconstruct_tracks(
        nodes=nodes,
        selected_edges=selected_edges,
        max_tracks=args.max_tracks,
    )
    tracks.to_csv(outdir / "step64_track_summary.csv", index=False)

    assignments = assign_cells_to_tracks(clustered, tracks)
    assignments.to_csv(outdir / "step64_track_assignment_by_cell.csv", index=False)

    track_module_matrix = build_track_module_matrix(nodes, tracks)
    track_module_matrix.to_csv(outdir / "step64_track_module_matrix.csv", index=False)

    feature_meta = {
        "cluster_features": cluster_features,
        "distance_features": distance_features,
        "module_features": modules,
        "state_score_features": state_scores,
        "z_pca_cols": z_pca_cols,
        "input_meta": input_meta,
        "distance_weights": {
            "w_kl": args.w_kl,
            "w_wasserstein": args.w_wasserstein,
            "w_mahalanobis": args.w_mahalanobis,
            "w_centroid": args.w_centroid,
            "w_module": args.w_module,
            "state_continuity_bonus": args.state_continuity_bonus,
            "celltype_continuity_bonus": args.celltype_continuity_bonus,
        },
        "scipy_available_for_wasserstein_sqrtm": SCIPY_AVAILABLE,
    }

    (outdir / "step64_distance_feature_metadata.json").write_text(
        json.dumps(feature_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    plot_outputs = plot_dynamics_graph(
        clustered=clustered,
        nodes=nodes,
        selected_edges=selected_edges,
        tracks=tracks,
        outdir=outdir,
        dpi=args.dpi,
        seed=args.seed,
    )

    report = {
        "status": "ok",
        "analysis_name": "StrokeNiche dynamics graph / track reconstruction",
        "unagi_like_logic": [
            "timepoint-specific cell clusters",
            "cluster-level latent distributions",
            "adjacent-timepoint symmetric KL and module distance",
            "D1-D3-D7 dynamics graph",
            "progression track reconstruction",
        ],
        "not_claimed": [
            "true 3D reconstruction",
            "true 4D registered tissue reconstruction",
            "same-cell longitudinal tracking",
        ],
        "input": str(args.input),
        "outdir": str(outdir),
        "n_cells": int(len(df)),
        "timepoint_counts": df["timepoint"].value_counts(dropna=False).to_dict(),
        "state_counts": df["state_group_std"].value_counts(dropna=False).to_dict(),
        "celltype_top_counts": df["celltype_std"].value_counts(dropna=False).head(20).to_dict(),
        "cluster_mode": args.cluster_mode,
        "n_nodes": int(len(nodes)),
        "n_edges_all": int(len(all_edges)),
        "n_edges_selected": int(len(selected_edges)),
        "n_tracks": int(len(tracks)),
        "n_track_cell_assignments": int(len(assignments)),
        "plot_outputs": plot_outputs,
        "outputs": {
            "modeling_input_used": str(outdir / "step64_modeling_input_used.csv"),
            "clustered_cells": str(outdir / "step64_clustered_cells.csv"),
            "nodes": str(outdir / "step64_dynamics_graph_nodes.csv"),
            "edges_all": str(outdir / "step64_dynamics_graph_edges_all.csv"),
            "edges_selected": str(outdir / "step64_dynamics_graph_edges.csv"),
            "track_summary": str(outdir / "step64_track_summary.csv"),
            "track_assignment_by_cell": str(outdir / "step64_track_assignment_by_cell.csv"),
            "track_module_matrix": str(outdir / "step64_track_module_matrix.csv"),
            "distance_feature_metadata": str(outdir / "step64_distance_feature_metadata.json"),
            "report_json": str(outdir / "step64_report.json"),
            "report_txt": str(outdir / "step64_report.txt"),
        },
        "interpretation_note": (
            "Step 64 reconstructs a StrokeNiche dynamics graph across D1/D3/D7 by linking "
            "timepoint-specific cell-state clusters with latent-distribution and module-distance metrics. "
            "It should be interpreted as a disease-state dynamics graph / pseudo-progression track analysis, "
            "not as true longitudinal cell tracking or registered 4D tissue reconstruction."
        ),
    }

    (outdir / "step64_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step 64 StrokeNiche dynamics graph / track reconstruction report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Nodes:")
    lines.append(nodes.head(60).to_string(index=False))
    lines.append("")
    lines.append("Selected edges:")
    lines.append(selected_edges.head(80).to_string(index=False))
    lines.append("")
    lines.append("Tracks:")
    lines.append(tracks.head(80).to_string(index=False) if not tracks.empty else "No tracks reconstructed.")

    (outdir / "step64_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step 64")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Top tracks:")
    log(tracks.head(20).to_string(index=False) if not tracks.empty else "No tracks reconstructed.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
