#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
53c_add_RCTD_composition_to_knn_neighbor_profile.py

Purpose
-------
Enhance the 53b virtual-cell kNN neighborhood profile by merging RCTD
cell-composition columns into the virtual-cell table and rebuilding kNN
neighborhood features.

This step fixes the 53b limitation:
  composition_cols_n = 0

Inputs
------
1) 53b coordinate-merged virtual table, preferred:
   results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53b/
     virtual_cell_state_table_with_coords.csv

2) RCTD metadata:
   results/task9_step3_RCTD_deconvolution_main/
     all_spots_metadata_with_RCTD.csv

Outputs
-------
results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/
  virtual_cell_state_table_with_coords_RCTD.csv
  virtual_cell_knn_neighbor_profile_RCTD.csv
  virtual_cell_knn_neighbor_profile_RCTD_light.csv
  virtual_cell_knn_neighbor_profile_RCTD_summary.csv
  virtual_cell_knn_neighbor_profile_RCTD_metadata.json
  virtual_cell_knn_neighbor_profile_RCTD_report.txt

Interpretation
--------------
This is still a kNN-derived neighborhood profile/proxy, not a trained neural
NeighborHead. It now includes RCTD composition features in the local
neighborhood vector.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/mnt/h/vir/ST")
DEFAULT_VIRTUAL_WITH_COORDS = BASE / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53b/virtual_cell_state_table_with_coords.csv"
DEFAULT_VIRTUAL_RAW = BASE / "results/step8_strokeniche_perturbmap/virtual_cell_state_table.csv"
DEFAULT_RCTD = BASE / "results/task9_step3_RCTD_deconvolution_main/all_spots_metadata_with_RCTD.csv"
DEFAULT_OUTDIR = BASE / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c"


def log(msg: str):
    print(msg, flush=True)


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def safe_name(x: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(x).strip()).strip("_")
    return (s or "unknown")[:max_len]


def strip_time_suffix(obs) -> str:
    s = str(obs).strip()
    s = re.sub(r"-(D1|D3|D7)$", "", s, flags=re.IGNORECASE)
    return s


def get_timepoint_from_obs(obs) -> str:
    s = str(obs).strip()
    m = re.search(r"-(D1|D3|D7)$", s, flags=re.IGNORECASE)
    return m.group(1).upper() if m else ""


def normalize_timepoint(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    m = re.search(r"(D1|D3|D7)", s)
    return m.group(1) if m else s


def pick_col(df: pd.DataFrame, names):
    nmap = {norm_col(c): c for c in df.columns}
    for n in names:
        k = norm_col(n)
        if k in nmap:
            return nmap[k]
    return None


def numeric_nonnull(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(pd.to_numeric(df[col], errors="coerce").notna().sum())


def entropy_rows(mat: np.ndarray) -> np.ndarray:
    x = np.asarray(mat, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x[x < 0] = 0.0

    row_sum = x.sum(axis=1, keepdims=True)
    p = np.divide(x, row_sum, out=np.zeros_like(x), where=row_sum > 0)

    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(p * np.where(p > 0, np.log(p), 0.0)).sum(axis=1)

    denom = math.log(x.shape[1]) if x.shape[1] > 1 else 1.0
    return ent / denom if denom > 0 else ent


def detect_coordinate_cols(df: pd.DataFrame):
    pairs = [
        ("spatial_x", "spatial_y"),
        ("x_coord", "y_coord"),
        ("x", "y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("image_col", "image_row"),
        ("imagecol", "imagerow"),
        ("array_col", "array_row"),
    ]
    nmap = {norm_col(c): c for c in df.columns}
    for a, b in pairs:
        ka, kb = norm_col(a), norm_col(b)
        if ka in nmap and kb in nmap:
            ca, cb = nmap[ka], nmap[kb]
            if numeric_nonnull(df, ca) > 0 and numeric_nonnull(df, cb) > 0:
                return ca, cb
    return None, None


def detect_region_prob_cols(df: pd.DataFrame):
    cols = []
    exact = [
        "prob_lesion_core",
        "prob_peri_infarct",
        "prob_remote_like",
        "P_lesion_core",
        "P_peri_infarct",
        "P_remote_like",
        "core_probability",
        "peri_probability",
        "remote_probability",
    ]
    nmap = {norm_col(c): c for c in df.columns}
    for e in exact:
        k = norm_col(e)
        if k in nmap:
            c = nmap[k]
            if numeric_nonnull(df, c) > 0 and c not in cols:
                cols.append(c)

    if cols:
        return cols

    for c in df.columns:
        nc = norm_col(c)
        if "prob" in nc and any(t in nc for t in ["core", "lesion", "peri", "remote"]):
            if not any(t in nc for t in ["delta", "shift", "after", "before"]):
                if numeric_nonnull(df, c) > 0:
                    cols.append(c)
    return cols


def detect_celltype_col(df: pd.DataFrame):
    names = [
        "dominant_celltype",
        "predicted_celltype",
        "predicted_celltype_filtered",
        "celltype",
        "cell_type",
        "major_celltype",
        "RCTD_predicted_celltype",
    ]
    return pick_col(df, names)


def detect_region_label_col(df: pd.DataFrame):
    names = [
        "region_true",
        "region_auto",
        "region_refined",
        "region_manual_final",
        "region_label",
        "lesion_compartment",
    ]
    return pick_col(df, names)


def detect_score_cols(df: pd.DataFrame):
    cols = []
    names = [
        "repair_score",
        "repair_score_from_npy",
        "injury_score",
        "myeloid_score",
        "reactive_score",
        "neuronal_score",
        "virtual_penumbra_score",
    ]
    nmap = {norm_col(c): c for c in df.columns}
    for n in names:
        k = norm_col(n)
        if k in nmap:
            c = nmap[k]
            if numeric_nonnull(df, c) > 0 and c not in cols:
                cols.append(c)
    return cols


def detect_rctd_cols(df: pd.DataFrame):
    """
    Keep true RCTD cell-composition columns only.
    Exclude predicted labels, sample, max weights, and metadata.
    """
    cols = []
    exclude = [
        "predicted",
        "sample",
        "weight_max",
        "max",
        "barcode",
        "source",
        "timepoint",
    ]
    for c in df.columns:
        nc = norm_col(c)
        if nc.startswith("rctd_"):
            if any(e in nc for e in exclude):
                continue
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() > 0:
                cols.append(c)
    return cols


def make_merge_key_for_virtual(v: pd.DataFrame) -> pd.Series:
    if "obs_name" not in v.columns:
        raise RuntimeError("Virtual table must contain obs_name for this pipeline.")
    if "timepoint" in v.columns:
        tp = v["timepoint"].map(normalize_timepoint)
    else:
        tp = v["obs_name"].map(get_timepoint_from_obs)
    barcode = v["obs_name"].map(strip_time_suffix)
    return barcode + "::" + tp


def make_merge_key_for_rctd(r: pd.DataFrame) -> pd.Series:
    barcode_col = pick_col(r, ["barcode", "spot", "spot_id", "obs_name"])
    if barcode_col is None:
        raise RuntimeError("RCTD metadata has no barcode/spot/obs_name column.")

    tp_col = pick_col(r, ["sample", "timepoint", "RCTD_sample", "orig.ident", "orig_ident", "Sample"])
    if tp_col is None:
        # Fallback: infer blank timepoint, usually not enough.
        tp = pd.Series("", index=r.index)
    else:
        tp = r[tp_col].map(normalize_timepoint)

    barcode = r[barcode_col].astype(str).map(strip_time_suffix)
    return barcode + "::" + tp


def merge_rctd_into_virtual(virtual: pd.DataFrame, rctd: pd.DataFrame):
    v = virtual.copy()
    r = rctd.copy()

    v["merge_key_53c"] = make_merge_key_for_virtual(v)
    r["merge_key_53c"] = make_merge_key_for_rctd(r)

    rctd_cols = detect_rctd_cols(r)

    x_col, y_col = detect_coordinate_cols(r)
    coord_cols = []
    if x_col and y_col:
        r["rctd_spatial_x"] = pd.to_numeric(r[x_col], errors="coerce")
        r["rctd_spatial_y"] = pd.to_numeric(r[y_col], errors="coerce")
        coord_cols = ["rctd_spatial_x", "rctd_spatial_y"]

    extra_cols = []
    for c in [
        "RCTD_predicted_celltype",
        "RCTD_weight_max",
        "RCTD_sample",
        "barcode",
        "sample",
        "timepoint",
        "region_auto",
        "region_refined",
        "region_manual_final",
        "predicted_celltype",
        "predicted_celltype_filtered",
        "x_coord",
        "y_coord",
    ]:
        if c in r.columns and c not in extra_cols:
            extra_cols.append(c)

    keep = ["merge_key_53c"] + coord_cols + rctd_cols + extra_cols
    keep = [c for c in keep if c in r.columns]

    r_small = r[keep].copy()
    r_small = r_small.drop_duplicates("merge_key_53c", keep="first")

    merged = v.merge(r_small, on="merge_key_53c", how="left", suffixes=("", "_from_RCTD"))

    # If virtual table did not already have coordinates, use RCTD coordinates.
    if "spatial_x" not in merged.columns and "rctd_spatial_x" in merged.columns:
        merged["spatial_x"] = merged["rctd_spatial_x"]
    if "spatial_y" not in merged.columns and "rctd_spatial_y" in merged.columns:
        merged["spatial_y"] = merged["rctd_spatial_y"]

    if "spatial_x" in merged.columns and "rctd_spatial_x" in merged.columns:
        merged["spatial_x"] = pd.to_numeric(merged["spatial_x"], errors="coerce").fillna(merged["rctd_spatial_x"])
    if "spatial_y" in merged.columns and "rctd_spatial_y" in merged.columns:
        merged["spatial_y"] = pd.to_numeric(merged["spatial_y"], errors="coerce").fillna(merged["rctd_spatial_y"])

    # Normalize RCTD composition to row proportions.
    rctd_present = [c for c in rctd_cols if c in merged.columns]
    if rctd_present:
        mat = merged[rctd_present].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        mat[mat < 0] = 0.0
        row_sum = mat.sum(axis=1)

        for c in rctd_present:
            prop_col = "RCTD_prop_" + c.replace("RCTD_", "")
            merged[prop_col] = np.divide(
                mat[c],
                row_sum,
                out=np.zeros(len(mat), dtype=float),
                where=row_sum.values > 0,
            )

        merged["RCTD_total_weight_sum"] = row_sum
        merged["RCTD_composition_available"] = row_sum.gt(0)
    else:
        merged["RCTD_total_weight_sum"] = np.nan
        merged["RCTD_composition_available"] = False

    matched_rctd = int(merged["RCTD_composition_available"].fillna(False).astype(bool).sum())
    coord_matched = int(merged[["spatial_x", "spatial_y"]].notna().all(axis=1).sum()) if "spatial_x" in merged.columns and "spatial_y" in merged.columns else 0

    meta = {
        "rctd_cols_raw": rctd_cols,
        "rctd_prop_cols": [c for c in merged.columns if c.startswith("RCTD_prop_")],
        "rctd_rows": int(len(rctd)),
        "virtual_rows": int(len(virtual)),
        "rctd_merge_unique_keys": int(r_small["merge_key_53c"].nunique()),
        "rctd_composition_matched_rows": matched_rctd,
        "coordinate_matched_rows": coord_matched,
        "rctd_coordinate_cols": [x_col, y_col],
    }

    return merged, meta


def create_onehot(df: pd.DataFrame, col: str, prefix: str, max_levels: int = 30):
    out = pd.DataFrame(index=df.index)
    cols = []

    if col is None or col not in df.columns:
        return out, cols, []

    s = df[col].fillna("unknown").astype(str)
    vc = s.value_counts()
    levels = [
        x for x in vc.index.tolist()
        if x and str(x).lower() not in ["nan", "none", "null", "unknown"]
    ][:max_levels]

    for level in levels:
        c = f"{prefix}_{safe_name(level)}"
        out[c] = (s == level).astype(float)
        cols.append(c)

    return out, cols, levels


def build_feature_table(df: pd.DataFrame, max_celltypes: int = 30):
    features = pd.DataFrame(index=df.index)
    feature_cols = []

    region_prob_cols = detect_region_prob_cols(df)
    for c in region_prob_cols:
        fc = "feature_regionprob_" + safe_name(c)
        features[fc] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        feature_cols.append(fc)

    celltype_col = detect_celltype_col(df)
    cell_onehot, cell_cols, cell_levels = create_onehot(df, celltype_col, "feature_celltype", max_celltypes)
    for c in cell_cols:
        features[c] = cell_onehot[c]
        feature_cols.append(c)

    rctd_prop_cols = [c for c in df.columns if c.startswith("RCTD_prop_")]
    for c in rctd_prop_cols:
        fc = "feature_RCTD_" + safe_name(c.replace("RCTD_prop_", ""))
        features[fc] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        feature_cols.append(fc)

    score_cols = detect_score_cols(df)
    for c in score_cols:
        fc = "feature_score_" + safe_name(c)
        features[fc] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        feature_cols.append(fc)

    # Remove constant columns.
    keep = []
    for c in feature_cols:
        x = pd.to_numeric(features[c], errors="coerce").fillna(0.0)
        if x.nunique(dropna=True) > 1:
            keep.append(c)

    features = features[keep].copy()

    meta = {
        "region_prob_cols": region_prob_cols,
        "celltype_col": celltype_col,
        "celltype_levels": cell_levels,
        "rctd_prop_cols": rctd_prop_cols,
        "score_cols": score_cols,
        "feature_cols": keep,
    }

    return features, keep, meta


def compute_knn_indices(coords: np.ndarray, k: int):
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), metric="euclidean")
        nn.fit(coords)
        dists, inds = nn.kneighbors(coords)
        return dists[:, 1:], inds[:, 1:]
    except Exception:
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(coords)
            dists, inds = tree.query(coords, k=min(k + 1, len(coords)))
            if inds.ndim == 1:
                inds = inds[:, None]
                dists = dists[:, None]
            return dists[:, 1:], inds[:, 1:]
        except Exception as e:
            raise RuntimeError(f"Cannot compute kNN. Need sklearn or scipy. Error: {e}")


def build_knn_profile(df: pd.DataFrame, features: pd.DataFrame, feature_cols, k: int):
    out = df.copy()

    if "timepoint" in out.columns:
        sample_col = "timepoint"
    else:
        sample_col = None

    out["knn_neighbor_profile_available"] = False
    out["knn_k_requested"] = k
    out["knn_k_effective"] = np.nan
    out["knn_mean_distance"] = np.nan
    out["knn_median_distance"] = np.nan
    out["knn_local_density_proxy"] = np.nan

    for c in feature_cols:
        out["neighbor_mean_" + safe_name(c)] = np.nan
        out["neighbor_delta_vs_self_" + safe_name(c)] = np.nan

    coords_all = out[["spatial_x", "spatial_y"]].apply(pd.to_numeric, errors="coerce")
    valid = coords_all.notna().all(axis=1)

    if sample_col:
        groups = out.loc[valid].groupby(sample_col, dropna=False).groups
        group_indices = [list(v) for v in groups.values()]
    else:
        group_indices = [out.index[valid].tolist()]

    feat_all = features.reindex(out.index).fillna(0.0)

    n_processed = 0
    group_summaries = []

    for idx in group_indices:
        if len(idx) <= 1:
            continue

        k_eff = min(k, len(idx) - 1)
        if k_eff <= 0:
            continue

        coords = coords_all.loc[idx].values
        dists, neigh = compute_knn_indices(coords, k_eff)

        feat = feat_all.loc[idx, feature_cols].values if feature_cols else np.zeros((len(idx), 0))

        if len(feature_cols):
            neigh_mean = feat[neigh].mean(axis=1)
            delta = neigh_mean - feat

            for j, c in enumerate(feature_cols):
                out.loc[idx, "neighbor_mean_" + safe_name(c)] = neigh_mean[:, j]
                out.loc[idx, "neighbor_delta_vs_self_" + safe_name(c)] = delta[:, j]

        out.loc[idx, "knn_neighbor_profile_available"] = True
        out.loc[idx, "knn_k_effective"] = k_eff
        out.loc[idx, "knn_mean_distance"] = dists.mean(axis=1)
        out.loc[idx, "knn_median_distance"] = np.median(dists, axis=1)
        out.loc[idx, "knn_local_density_proxy"] = 1.0 / (dists.mean(axis=1) + 1e-8)

        n_processed += len(idx)
        group_summaries.append({
            "group_n": int(len(idx)),
            "k_eff": int(k_eff),
            "mean_knn_distance": float(np.nanmean(dists)),
        })

    meta = {
        "n_valid_coordinates": int(valid.sum()),
        "n_processed": int(n_processed),
        "n_groups": int(len(group_indices)),
        "group_summaries_head": group_summaries[:20],
    }

    return out, meta


def add_entropy_metrics(profile: pd.DataFrame):
    out = profile.copy()
    meta = {}

    groups = {
        "region": [c for c in out.columns if c.startswith("neighbor_mean_feature_regionprob_")],
        "celltype": [c for c in out.columns if c.startswith("neighbor_mean_feature_celltype_")],
        "RCTD": [c for c in out.columns if c.startswith("neighbor_mean_feature_RCTD_")],
    }

    for name, cols in groups.items():
        meta[f"{name}_neighbor_cols"] = cols

        if cols:
            mat = out[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
            ent = entropy_rows(mat)
            row_sum = mat.sum(axis=1, keepdims=True)
            p = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)
            dominant = np.nanmax(p, axis=1)
            mixedness = 1.0 - dominant

            out[f"knn_neighbor_{name}_entropy"] = ent
            out[f"knn_neighbor_{name}_dominant_fraction"] = dominant
            out[f"knn_neighbor_{name}_mixedness"] = mixedness
        else:
            out[f"knn_neighbor_{name}_entropy"] = np.nan
            out[f"knn_neighbor_{name}_dominant_fraction"] = np.nan
            out[f"knn_neighbor_{name}_mixedness"] = np.nan

    # Backward compatible aliases
    if "knn_neighbor_region_mixedness" in out.columns:
        out["knn_neighbor_region_boundary_score"] = out["knn_neighbor_region_mixedness"]
    if "knn_neighbor_RCTD_mixedness" in out.columns:
        out["knn_neighbor_RCTD_composition_mixedness"] = out["knn_neighbor_RCTD_mixedness"]

    return out, meta


def make_light_table(profile: pd.DataFrame):
    keep = []

    base_cols = [
        "obs_name",
        "timepoint",
        "region_true",
        "dominant_celltype",
        "spatial_x",
        "spatial_y",
        "RCTD_composition_available",
        "RCTD_total_weight_sum",
        "knn_neighbor_profile_available",
        "knn_k_effective",
        "knn_mean_distance",
        "knn_local_density_proxy",
        "knn_neighbor_region_entropy",
        "knn_neighbor_region_boundary_score",
        "knn_neighbor_celltype_entropy",
        "knn_neighbor_celltype_mixedness",
        "knn_neighbor_RCTD_entropy",
        "knn_neighbor_RCTD_composition_mixedness",
    ]

    for c in base_cols:
        if c in profile.columns and c not in keep:
            keep.append(c)

    # RCTD proportions and neighbor means
    for c in profile.columns:
        if c.startswith("RCTD_prop_"):
            keep.append(c)
    for c in profile.columns:
        if c.startswith("neighbor_mean_feature_RCTD_"):
            keep.append(c)
    for c in profile.columns:
        if c.startswith("neighbor_delta_vs_self_feature_RCTD_"):
            keep.append(c)

    # Region/celltype neighbor features
    for c in profile.columns:
        if c.startswith("neighbor_mean_feature_regionprob_"):
            keep.append(c)
    for c in profile.columns:
        if c.startswith("neighbor_mean_feature_celltype_"):
            keep.append(c)

    seen = set()
    keep2 = []
    for c in keep:
        if c in profile.columns and c not in seen:
            seen.add(c)
            keep2.append(c)

    return profile[keep2].copy()


def make_summary(profile: pd.DataFrame):
    group_cols = [c for c in ["timepoint", "region_true", "dominant_celltype"] if c in profile.columns]
    metric_cols = [
        "knn_neighbor_profile_available",
        "knn_mean_distance",
        "knn_local_density_proxy",
        "knn_neighbor_region_entropy",
        "knn_neighbor_region_boundary_score",
        "knn_neighbor_celltype_entropy",
        "knn_neighbor_celltype_mixedness",
        "knn_neighbor_RCTD_entropy",
        "knn_neighbor_RCTD_composition_mixedness",
        "RCTD_total_weight_sum",
    ]
    metric_cols = [c for c in metric_cols if c in profile.columns]

    rows = []
    grouped = profile.groupby(group_cols, dropna=False) if group_cols else [("all", profile)]

    for keys, sub in grouped:
        if group_cols:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {g: k for g, k in zip(group_cols, keys)}
        else:
            row = {"group": "all"}

        row["n"] = len(sub)

        for c in metric_cols:
            if c == "knn_neighbor_profile_available":
                row[c + "_sum"] = int(sub[c].fillna(False).astype(bool).sum())
            else:
                row[c + "_mean"] = pd.to_numeric(sub[c], errors="coerce").mean()

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/mnt/h/vir/ST")
    ap.add_argument("--virtual", default=str(DEFAULT_VIRTUAL_WITH_COORDS))
    ap.add_argument("--rctd", default=str(DEFAULT_RCTD))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--max_celltypes", type=int, default=30)
    args = ap.parse_args()

    virtual_path = Path(args.virtual)
    if not virtual_path.exists():
        virtual_path = DEFAULT_VIRTUAL_RAW

    rctd_path = Path(args.rctd)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 90)
    log("53c add RCTD composition to virtual-cell kNN neighborhood profile")
    log("=" * 90)
    log(f"virtual={virtual_path}")
    log(f"rctd={rctd_path}")
    log(f"outdir={outdir}")
    log(f"knn={args.knn}")

    if not virtual_path.exists():
        raise FileNotFoundError(virtual_path)
    if not rctd_path.exists():
        raise FileNotFoundError(rctd_path)

    virtual = pd.read_csv(virtual_path, low_memory=False)
    rctd = pd.read_csv(rctd_path, low_memory=False)

    merged, merge_meta = merge_rctd_into_virtual(virtual, rctd)

    log("[merge] done")
    log(f"[merge] virtual_rows={merge_meta['virtual_rows']}")
    log(f"[merge] rctd_cols_raw={merge_meta['rctd_cols_raw']}")
    log(f"[merge] rctd_composition_matched_rows={merge_meta['rctd_composition_matched_rows']}/{merge_meta['virtual_rows']}")
    log(f"[merge] coordinate_matched_rows={merge_meta['coordinate_matched_rows']}/{merge_meta['virtual_rows']}")

    if merge_meta["coordinate_matched_rows"] == 0:
        raise RuntimeError("No spatial coordinates after RCTD merge. Run 53b first or use RCTD metadata with coordinates.")

    features, feature_cols, feature_meta = build_feature_table(merged, max_celltypes=args.max_celltypes)

    log(f"[features] region_prob_cols={feature_meta['region_prob_cols']}")
    log(f"[features] celltype_col={feature_meta['celltype_col']}")
    log(f"[features] rctd_prop_cols_n={len(feature_meta['rctd_prop_cols'])}")
    log(f"[features] score_cols={feature_meta['score_cols']}")
    log(f"[features] final_feature_cols_n={len(feature_cols)}")

    profile, knn_meta = build_knn_profile(merged, features, feature_cols, args.knn)
    profile, entropy_meta = add_entropy_metrics(profile)

    light = make_light_table(profile)
    summary = make_summary(profile)

    out_merged = outdir / "virtual_cell_state_table_with_coords_RCTD.csv"
    out_profile = outdir / "virtual_cell_knn_neighbor_profile_RCTD.csv"
    out_light = outdir / "virtual_cell_knn_neighbor_profile_RCTD_light.csv"
    out_summary = outdir / "virtual_cell_knn_neighbor_profile_RCTD_summary.csv"
    out_meta = outdir / "virtual_cell_knn_neighbor_profile_RCTD_metadata.json"
    out_report = outdir / "virtual_cell_knn_neighbor_profile_RCTD_report.txt"

    merged.to_csv(out_merged, index=False)
    profile.to_csv(out_profile, index=False)
    light.to_csv(out_light, index=False)
    summary.to_csv(out_summary, index=False)

    metadata = {
        "status": "ok",
        "virtual_path": str(virtual_path),
        "rctd_path": str(rctd_path),
        "outdir": str(outdir),
        "merge_meta": merge_meta,
        "feature_meta": feature_meta,
        "knn_meta": knn_meta,
        "entropy_meta": entropy_meta,
        "outputs": {
            "merged": str(out_merged),
            "profile": str(out_profile),
            "light": str(out_light),
            "summary": str(out_summary),
            "report": str(out_report),
        },
    }
    out_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("53c RCTD-enhanced virtual-cell kNN neighborhood profile report")
    lines.append("=" * 90)
    lines.append(f"virtual_path: {virtual_path}")
    lines.append(f"rctd_path: {rctd_path}")
    lines.append(f"virtual_rows: {len(profile)}")
    lines.append(f"coordinate_matched_rows: {merge_meta['coordinate_matched_rows']}/{merge_meta['virtual_rows']}")
    lines.append(f"rctd_composition_matched_rows: {merge_meta['rctd_composition_matched_rows']}/{merge_meta['virtual_rows']}")
    lines.append("")
    lines.append("RCTD columns:")
    lines.append("; ".join(merge_meta["rctd_cols_raw"]))
    lines.append("")
    lines.append("Feature columns:")
    lines.append(f"region_prob_cols: {feature_meta['region_prob_cols']}")
    lines.append(f"celltype_col: {feature_meta['celltype_col']}")
    lines.append(f"rctd_prop_cols_n: {len(feature_meta['rctd_prop_cols'])}")
    lines.append(f"score_cols: {feature_meta['score_cols']}")
    lines.append(f"final_feature_cols_n: {len(feature_cols)}")
    lines.append("")
    lines.append("Key profile metrics:")
    for c in [
        "knn_neighbor_profile_available",
        "knn_mean_distance",
        "knn_local_density_proxy",
        "knn_neighbor_region_entropy",
        "knn_neighbor_region_boundary_score",
        "knn_neighbor_celltype_entropy",
        "knn_neighbor_celltype_mixedness",
        "knn_neighbor_RCTD_entropy",
        "knn_neighbor_RCTD_composition_mixedness",
        "RCTD_total_weight_sum",
    ]:
        if c in profile.columns:
            if c == "knn_neighbor_profile_available":
                lines.append(f"{c}: true={int(profile[c].fillna(False).astype(bool).sum())}/{len(profile)}")
            else:
                x = pd.to_numeric(profile[c], errors="coerce")
                lines.append(f"{c}: non-null={x.notna().sum()}, mean={x.mean()}, median={x.median()}, min={x.min()}, max={x.max()}")
    lines.append("")
    lines.append("Summary head:")
    lines.append(summary.head(50).to_string(index=False))
    lines.append("")
    lines.append("Light table head:")
    lines.append(light.head(20).to_string(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "53c extends 53b by adding RCTD cell-composition features to the "
        "coordinate-based kNN neighborhood profile. This supports a richer "
        "neighbor_i profile including region probability, dominant cell type, "
        "repair score and RCTD-derived local composition. It remains a "
        "kNN-derived profile/proxy rather than a trained neural NeighborHead."
    )

    out_report.write_text("\n".join(lines), encoding="utf-8")

    log("=" * 90)
    log("DONE 53c RCTD-enhanced kNN neighborhood profile")
    log("=" * 90)
    log(f"Saved: {out_merged}")
    log(f"Saved: {out_profile}")
    log(f"Saved: {out_light}")
    log(f"Saved: {out_summary}")
    log(f"Saved: {out_meta}")
    log(f"Saved: {out_report}")
    log("")
    log("Key checks:")
    log(f"coordinate_matched_rows={merge_meta['coordinate_matched_rows']}/{merge_meta['virtual_rows']}")
    log(f"rctd_composition_matched_rows={merge_meta['rctd_composition_matched_rows']}/{merge_meta['virtual_rows']}")
    log(f"knn_profile_available={int(profile['knn_neighbor_profile_available'].fillna(False).astype(bool).sum())}/{len(profile)}")
    if "knn_neighbor_RCTD_entropy" in profile.columns:
        x = pd.to_numeric(profile["knn_neighbor_RCTD_entropy"], errors="coerce")
        log(f"knn_neighbor_RCTD_entropy_nonnull={x.notna().sum()}")
    if "knn_neighbor_RCTD_composition_mixedness" in profile.columns:
        x = pd.to_numeric(profile["knn_neighbor_RCTD_composition_mixedness"], errors="coerce")
        log(f"knn_neighbor_RCTD_composition_mixedness_nonnull={x.notna().sum()}")

    print("")
    print("Summary head:")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
