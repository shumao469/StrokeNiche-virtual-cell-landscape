#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
53b_rebuild_virtual_knn_neighbor_profile.py

Purpose
-------
Fix the missing spatial coordinates in virtual_cell_state_table.csv and rebuild
a true spatial kNN neighborhood profile at the virtual-cell / spot level.

This script does two things:

1) Merge coordinates back into virtual_cell_state_table.csv
   from raw spatial files, Visium metadata, tissue_positions.csv,
   obs/metadata files, or user-specified coordinate metadata.

2) Reconstruct kNN neighborhood profiles using:
   - spatial x/y coordinates
   - region probability columns or region labels
   - celltype labels
   - RCTD / cell2location / abundance / composition columns if present

Outputs
-------
/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53b/
  virtual_cell_state_table_with_coords.csv
  virtual_cell_knn_neighbor_profile.csv
  virtual_cell_knn_neighbor_profile_light.csv
  virtual_cell_knn_neighbor_profile_summary.csv
  virtual_cell_knn_neighbor_profile_report.txt
  virtual_cell_knn_neighbor_profile_metadata.json

Important
---------
This is a spatial kNN neighborhood profile/proxy, not a trained neural NeighborHead.
Use the wording:
  "kNN-derived neighborhood profile / neighbor-shift proxy"
unless a formal NeighborHead is later trained.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DEFAULT = Path("/mnt/h/vir/ST")
OUTDIR_DEFAULT = BASE_DEFAULT / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53b"


# -----------------------------------------------------------------------------
# General utilities
# -----------------------------------------------------------------------------

def log(msg: str):
    print(msg, flush=True)


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def norm_key(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"[\u2010-\u2015]", "-", s)
    s = re.sub(r"\s+", "", s)
    return s


def safe_name(x: str, max_len: int = 70) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(x).strip()).strip("_")
    if not s:
        s = "unknown"
    return s[:max_len]


def pick_column(cols, exact=(), contains=()):
    col_map = {norm_col(c): c for c in cols}

    for e in exact:
        k = norm_col(e)
        if k in col_map:
            return col_map[k]

    for term in contains:
        t = norm_col(term)
        for c in cols:
            if t in norm_col(c):
                return c

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
    if denom <= 0:
        return ent
    return ent / denom


# -----------------------------------------------------------------------------
# File discovery and reading
# -----------------------------------------------------------------------------

def read_table(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Robust table reader for csv/tsv/txt/gz/parquet.
    Also handles 10x Visium tissue_positions_list.csv without header.
    """
    suffixes = "".join(path.suffixes).lower()
    name = path.name.lower()

    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)

    if suffixes.endswith(".xlsx") or suffixes.endswith(".xls"):
        return pd.read_excel(path, nrows=nrows)

    compression = "gzip" if suffixes.endswith(".gz") else None

    # First attempt: infer delimiter.
    try:
        df = pd.read_csv(path, sep=None, engine="python", compression=compression, nrows=nrows, low_memory=False)
    except Exception:
        sep = "\t" if (".tsv" in suffixes or name.endswith(".txt") or "positions" in name) else ","
        df = pd.read_csv(path, sep=sep, compression=compression, nrows=nrows, low_memory=False)

    # If this looks like headerless 10x tissue_positions_list.csv, reread with fixed names.
    # 10x old format:
    # barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    # 10x newer tissue_positions.csv often has header already.
    norm_cols = [norm_col(c) for c in df.columns]
    has_coord = any("pxl" in c or "array" in c or c in ["x", "y", "row", "col"] for c in norm_cols)
    if ("tissue_positions" in name or "positions" in name) and not has_coord:
        try:
            df2 = pd.read_csv(path, header=None, compression=compression, nrows=nrows, low_memory=False)
            if df2.shape[1] >= 6:
                df2 = df2.iloc[:, :6].copy()
                df2.columns = [
                    "barcode",
                    "in_tissue",
                    "array_row",
                    "array_col",
                    "pxl_row_in_fullres",
                    "pxl_col_in_fullres",
                ]
                return df2
        except Exception:
            pass

    # If first column is unnamed index, keep it but rename to possible id.
    for c in list(df.columns):
        if str(c).startswith("Unnamed"):
            df = df.rename(columns={c: "index_id"})
            break

    return df


def find_virtual_cell_table(base: Path, user_path: Optional[str] = None) -> Optional[Path]:
    if user_path:
        p = Path(user_path)
        return p if p.exists() else None

    candidates = [
        base / "results/step8_strokeniche_perturbmap/virtual_cell_state_table.csv",
        base / "results/step8_strokeniche_perturbmap/virtual_cell_state_table_with_coords.csv",
    ]
    for p in candidates:
        if p.exists():
            return p

    patterns = [
        "results/step8_strokeniche_perturbmap/**/*virtual*cell*state*.csv",
        "results/**/*virtual*cell*state*.csv",
    ]
    hits = []
    for pat in patterns:
        hits.extend(base.glob(pat))

    hits = [p for p in hits if p.exists() and p.is_file()]
    if hits:
        hits = sorted(hits, key=lambda p: (len(str(p)), str(p)))
        return hits[0]

    return None


def discover_coordinate_metadata_files(base: Path, user_paths: Optional[str] = None, max_files: int = 300) -> List[Path]:
    if user_paths:
        paths = []
        for x in user_paths.split(","):
            x = x.strip()
            if not x:
                continue
            p = Path(x)
            if p.exists():
                paths.append(p)
        return paths

    roots = [
        base,
        base / "data",
        base / "models",
        base / "results",
        base / "ST",
    ]

    patterns = [
        "**/spatial/tissue_positions*.csv",
        "**/spatial/tissue_positions*.csv.gz",
        "**/tissue_positions*.csv",
        "**/tissue_positions*.csv.gz",
        "**/*positions*.csv",
        "**/*positions*.csv.gz",
        "**/*coordinate*.csv",
        "**/*coordinates*.csv",
        "**/*spot*metadata*.csv",
        "**/*spot*meta*.csv",
        "**/*metadata*.csv",
        "**/*meta*.csv",
        "**/*obs*.csv",
        "**/*obs*.tsv",
        "**/*spatial*.csv",
        "**/*visium*.csv",
    ]

    hits = []
    seen = set()

    for root in roots:
        if not root.exists():
            continue
        for pat in patterns:
            for p in root.glob(pat):
                if not p.is_file():
                    continue
                sp = str(p)
                if sp in seen:
                    continue
                seen.add(sp)
                # Avoid huge unrelated result tables if possible.
                lname = p.name.lower()
                if any(bad in lname for bad in ["combined_drug", "chembl", "ctd_", "dgidb", "lincs_signature_long"]):
                    continue
                hits.append(p)
                if len(hits) >= max_files:
                    return hits

    return hits


# -----------------------------------------------------------------------------
# Column detection
# -----------------------------------------------------------------------------

def detect_id_col(df: pd.DataFrame) -> Optional[str]:
    cols = list(df.columns)

    exact = [
        "barcode",
        "barcodes",
        "spot_id",
        "spot",
        "cell_id",
        "cell",
        "cell_id_original",
        "cell_name",
        "cell_barcode",
        "obs_names",
        "obs_name",
        "index",
        "index_id",
        "id",
        "sample_barcode",
    ]

    c = pick_column(cols, exact=exact)
    if c:
        return c

    # Columns containing barcode / spot / cell and likely string-like.
    for term in ["barcode", "spot", "cell"]:
        for col in cols:
            if term in norm_col(col):
                s = df[col].dropna().astype(str)
                if len(s) and s.nunique() > min(10, max(1, len(s) // 10)):
                    return col

    return None


def detect_coordinate_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    norm_to_col = {norm_col(c): c for c in cols}

    candidate_pairs = [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("pixel_col", "pixel_row"),
        ("imagecol", "imagerow"),
        ("image_col", "image_row"),
        ("array_col", "array_row"),
        ("col", "row"),
        ("spot_x", "spot_y"),
        ("center_x", "center_y"),
        ("coord_x", "coord_y"),
    ]

    for a, b in candidate_pairs:
        ka, kb = norm_col(a), norm_col(b)
        if ka in norm_to_col and kb in norm_to_col:
            ca, cb = norm_to_col[ka], norm_to_col[kb]
            if numeric_nonnull(df, ca) > 0 and numeric_nonnull(df, cb) > 0:
                return ca, cb

    # Any x-like and y-like pair.
    x_like = []
    y_like = []
    for c in cols:
        nc = norm_col(c)
        if re.search(r"(^|_)(x|col|column|pxl_col|image_col|array_col)(_|$)", nc):
            x_like.append(c)
        if re.search(r"(^|_)(y|row|pxl_row|image_row|array_row)(_|$)", nc):
            y_like.append(c)

    for ca in x_like:
        for cb in y_like:
            if ca == cb:
                continue
            if numeric_nonnull(df, ca) > 0 and numeric_nonnull(df, cb) > 0:
                return ca, cb

    return None, None


def detect_sample_col(df: pd.DataFrame) -> Optional[str]:
    return pick_column(
        df.columns,
        exact=[
            "sample_id",
            "sample",
            "slice_id",
            "section_id",
            "section",
            "library_id",
            "batch",
            "timepoint",
            "orig_ident",
        ],
        contains=["sample", "slice", "section", "library", "batch", "orig"],
    )


def detect_region_probability_columns(df: pd.DataFrame) -> List[str]:
    selected = []

    patterns = [
        r"(core).*prob",
        r"(lesion).*prob",
        r"(peri).*prob",
        r"(remote).*prob",
        r"(normal).*prob",
        r"prob.*(core)",
        r"prob.*(lesion)",
        r"prob.*(peri)",
        r"prob.*(remote)",
        r"region.*prob",
    ]

    for c in df.columns:
        nc = norm_col(c)
        if any(re.search(p, nc) for p in patterns):
            if not any(x in nc for x in ["delta", "shift", "perturb", "after", "before"]):
                if numeric_nonnull(df, c) > 0:
                    selected.append(c)

    exacts = [
        "core_probability",
        "peri_probability",
        "remote_probability",
        "lesion_core_probability",
        "peri_infarct_probability",
        "remote_like_probability",
        "region_core_probability",
        "region_peri_probability",
        "region_remote_probability",
    ]

    for e in exacts:
        c = pick_column(df.columns, exact=[e])
        if c and c not in selected and numeric_nonnull(df, c) > 0:
            selected.append(c)

    # Deduplicate.
    seen = set()
    out = []
    for c in selected:
        if c not in seen:
            seen.add(c)
            out.append(c)

    return out


def detect_region_label_col(df: pd.DataFrame) -> Optional[str]:
    return pick_column(
        df.columns,
        exact=[
            "region",
            "region_label",
            "region_auto",
            "lesion_region",
            "lesion_compartment",
            "compartment",
            "zone",
            "spatial_domain",
        ],
        contains=["region", "compartment", "zone", "domain"],
    )


def detect_celltype_col(df: pd.DataFrame) -> Optional[str]:
    return pick_column(
        df.columns,
        exact=[
            "predicted_celltype",
            "predicted_cell_type",
            "celltype",
            "cell_type",
            "major_celltype",
            "major_cell_type",
            "annotation",
            "cell_annotation",
            "seurat_clusters",
            "cluster",
        ],
        contains=["celltype", "cell_type", "annotation", "cluster"],
    )


def detect_composition_columns(df: pd.DataFrame, max_cols: int = 120) -> List[str]:
    selected = []

    include_terms = [
        "rctd",
        "cell2location",
        "celltype_abundance",
        "cell_type_abundance",
        "abundance",
        "composition",
        "proportion",
        "fraction",
        "frac",
        "prop",
        "spotlight",
        "stereoscope",
        "tangram",
    ]

    exclude_terms = [
        "rank",
        "score",
        "probability",
        "delta",
        "shift",
        "distance",
        "x",
        "y",
        "row",
        "col",
        "time",
        "phase",
        "priority",
    ]

    for c in df.columns:
        nc = norm_col(c)
        if any(t in nc for t in include_terms) and not any(re.search(rf"(^|_){e}(_|$)", nc) for e in exclude_terms):
            if numeric_nonnull(df, c) > 0:
                selected.append(c)

    return selected[:max_cols]


def detect_score_columns(df: pd.DataFrame) -> List[str]:
    selected = []

    terms = [
        "repair_score",
        "core_score",
        "injury_score",
        "hypoxia_score",
        "inflammation_score",
        "bbb_score",
        "barrier_score",
        "ferroptosis_score",
        "pseudotime",
        "time_score",
    ]

    for c in df.columns:
        nc = norm_col(c)
        if any(t in nc for t in terms):
            if numeric_nonnull(df, c) > 0:
                selected.append(c)

    return selected[:40]


# -----------------------------------------------------------------------------
# Coordinate merge
# -----------------------------------------------------------------------------

def score_metadata_candidate(virtual: pd.DataFrame, meta_path: Path, n_preview: int = 1000) -> Dict:
    try:
        meta = read_table(meta_path, nrows=n_preview)
    except Exception as e:
        return {
            "path": str(meta_path),
            "ok": False,
            "error": str(e),
            "score": -1,
        }

    x_col, y_col = detect_coordinate_columns(meta)
    meta_id_col = detect_id_col(meta)

    v_id_col = detect_id_col(virtual)
    v_n = len(virtual)

    score = 0
    if x_col and y_col:
        score += 50
    if meta_id_col:
        score += 20
    if v_id_col:
        score += 10

    meta_n = len(meta)
    if meta_n == v_n:
        score += 20

    # Estimate overlap if IDs available.
    overlap_n = 0
    if v_id_col and meta_id_col:
        v_keys = set(virtual[v_id_col].dropna().astype(str).map(norm_key).head(n_preview * 5))
        m_keys = set(meta[meta_id_col].dropna().astype(str).map(norm_key))
        overlap_n = len(v_keys & m_keys)
        if overlap_n > 0:
            score += min(100, overlap_n)

    return {
        "path": str(meta_path),
        "ok": True,
        "score": score,
        "meta_shape": [int(meta.shape[0]), int(meta.shape[1])],
        "x_col": x_col,
        "y_col": y_col,
        "meta_id_col": meta_id_col,
        "virtual_id_col": v_id_col,
        "overlap_preview_n": int(overlap_n),
        "same_n_rows": bool(meta_n == v_n),
    }


def merge_coordinates(
    virtual: pd.DataFrame,
    metadata_paths: List[Path],
    user_coord_file: Optional[str] = None,
    allow_order_merge: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Try coordinate metadata files and choose best merge by:
      1. ID overlap
      2. same row count order merge
    """
    v = virtual.copy()
    v_id_col = detect_id_col(v)

    if "spatial_x" in v.columns and "spatial_y" in v.columns and numeric_nonnull(v, "spatial_x") > 0 and numeric_nonnull(v, "spatial_y") > 0:
        return v, {
            "status": "already_has_spatial_x_y",
            "virtual_id_col": v_id_col,
            "coordinate_source": "virtual_table_existing",
            "matched_rows": int(v[["spatial_x", "spatial_y"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).sum()),
        }

    # Also check if virtual already has other coordinate names.
    vx, vy = detect_coordinate_columns(v)
    if vx and vy:
        v["spatial_x"] = pd.to_numeric(v[vx], errors="coerce")
        v["spatial_y"] = pd.to_numeric(v[vy], errors="coerce")
        return v, {
            "status": "virtual_had_coordinate_columns_renamed",
            "virtual_id_col": v_id_col,
            "virtual_x_col": vx,
            "virtual_y_col": vy,
            "coordinate_source": "virtual_table_existing",
            "matched_rows": int(v[["spatial_x", "spatial_y"]].notna().all(axis=1).sum()),
        }

    candidate_reports = []

    for p in metadata_paths:
        rep = score_metadata_candidate(v, p)
        candidate_reports.append(rep)

    candidate_reports_sorted = sorted(candidate_reports, key=lambda d: d.get("score", -1), reverse=True)

    best = None
    best_merged = None
    best_matched = -1

    for rep in candidate_reports_sorted:
        if not rep.get("ok"):
            continue
        if not rep.get("x_col") or not rep.get("y_col"):
            continue

        p = Path(rep["path"])

        try:
            meta = read_table(p)
        except Exception:
            continue

        x_col, y_col = detect_coordinate_columns(meta)
        meta_id_col = detect_id_col(meta)

        if not x_col or not y_col:
            continue

        meta_small = meta.copy()
        meta_small["coord_spatial_x"] = pd.to_numeric(meta_small[x_col], errors="coerce")
        meta_small["coord_spatial_y"] = pd.to_numeric(meta_small[y_col], errors="coerce")

        # ID merge if possible.
        if v_id_col and meta_id_col:
            tmp_v = v.copy()
            tmp_v["_merge_key_for_coords"] = tmp_v[v_id_col].astype(str).map(norm_key)

            meta_small["_merge_key_for_coords"] = meta_small[meta_id_col].astype(str).map(norm_key)
            meta_small = meta_small[
                meta_small["_merge_key_for_coords"].astype(str).str.len().gt(0)
            ].copy()
            meta_small = meta_small.drop_duplicates("_merge_key_for_coords", keep="first")

            merged = tmp_v.merge(
                meta_small[["_merge_key_for_coords", "coord_spatial_x", "coord_spatial_y"]],
                on="_merge_key_for_coords",
                how="left",
            )

            matched = int(merged[["coord_spatial_x", "coord_spatial_y"]].notna().all(axis=1).sum())

            if matched > best_matched:
                best_matched = matched
                best = {
                    "status": "id_merge",
                    "coordinate_source": str(p),
                    "virtual_id_col": v_id_col,
                    "metadata_id_col": meta_id_col,
                    "metadata_x_col": x_col,
                    "metadata_y_col": y_col,
                    "matched_rows": matched,
                    "virtual_rows": int(len(v)),
                    "candidate_report": rep,
                }
                best_merged = merged.drop(columns=["_merge_key_for_coords"])

        # Order merge fallback.
        if allow_order_merge and len(meta_small) == len(v):
            merged = v.copy()
            merged["coord_spatial_x"] = meta_small["coord_spatial_x"].values
            merged["coord_spatial_y"] = meta_small["coord_spatial_y"].values
            matched = int(merged[["coord_spatial_x", "coord_spatial_y"]].notna().all(axis=1).sum())

            if matched > best_matched:
                best_matched = matched
                best = {
                    "status": "row_order_merge",
                    "coordinate_source": str(p),
                    "virtual_id_col": v_id_col,
                    "metadata_id_col": meta_id_col,
                    "metadata_x_col": x_col,
                    "metadata_y_col": y_col,
                    "matched_rows": matched,
                    "virtual_rows": int(len(v)),
                    "warning": "Coordinates merged by row order because direct ID merge was unavailable or weaker.",
                    "candidate_report": rep,
                }
                best_merged = merged

    if best_merged is None:
        return v, {
            "status": "failed_no_coordinate_match",
            "virtual_id_col": v_id_col,
            "candidate_reports": candidate_reports_sorted[:30],
            "matched_rows": 0,
            "virtual_rows": int(len(v)),
        }

    best_merged["spatial_x"] = pd.to_numeric(best_merged["coord_spatial_x"], errors="coerce")
    best_merged["spatial_y"] = pd.to_numeric(best_merged["coord_spatial_y"], errors="coerce")

    # Keep original source coordinate columns.
    best_merged["coordinate_source_file"] = best["coordinate_source"]
    best_merged["coordinate_merge_status"] = best["status"]

    best["candidate_reports_top30"] = candidate_reports_sorted[:30]

    return best_merged, best


# -----------------------------------------------------------------------------
# Feature construction
# -----------------------------------------------------------------------------

def create_label_onehot(df: pd.DataFrame, label_col: Optional[str], prefix: str, max_levels: int = 40) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if label_col is None or label_col not in df.columns:
        return pd.DataFrame(index=df.index), [], []

    s = df[label_col].fillna("unknown").astype(str)
    vc = s.value_counts()
    levels = [
        x for x in vc.index.tolist()
        if x and str(x).lower() not in ["nan", "none", "null"]
    ][:max_levels]

    out = pd.DataFrame(index=df.index)
    cols = []

    for level in levels:
        c = f"{prefix}_{safe_name(level)}"
        out[c] = (s == level).astype(float)
        cols.append(c)

    return out, cols, levels


def build_feature_matrix(df: pd.DataFrame, max_celltypes: int, max_regions: int) -> Tuple[pd.DataFrame, List[str], Dict]:
    region_prob_cols = detect_region_probability_columns(df)
    region_label_col = detect_region_label_col(df)
    celltype_col = detect_celltype_col(df)
    composition_cols = detect_composition_columns(df)
    score_cols = detect_score_columns(df)

    features = pd.DataFrame(index=df.index)
    feature_cols = []

    # Region probability columns.
    for c in region_prob_cols:
        out_c = f"feature_regionprob_{safe_name(c)}"
        features[out_c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        feature_cols.append(out_c)

    # If no explicit region probabilities, use region label one-hot.
    region_onehot_cols = []
    region_levels = []
    if not region_prob_cols:
        onehot, region_onehot_cols, region_levels = create_label_onehot(
            df,
            label_col=region_label_col,
            prefix="feature_region",
            max_levels=max_regions,
        )
        for c in region_onehot_cols:
            features[c] = onehot[c]
            feature_cols.append(c)

    # Celltype one-hot.
    cell_onehot, cell_onehot_cols, cell_levels = create_label_onehot(
        df,
        label_col=celltype_col,
        prefix="feature_celltype",
        max_levels=max_celltypes,
    )
    for c in cell_onehot_cols:
        features[c] = cell_onehot[c]
        feature_cols.append(c)

    # Composition columns.
    for c in composition_cols:
        out_c = f"feature_comp_{safe_name(c)}"
        features[out_c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        feature_cols.append(out_c)

    # Score columns, weak neighborhood context.
    for c in score_cols:
        out_c = f"feature_score_{safe_name(c)}"
        features[out_c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        feature_cols.append(out_c)

    # Remove all-zero / constant unusable cols.
    keep = []
    for c in feature_cols:
        x = pd.to_numeric(features[c], errors="coerce").fillna(0.0)
        if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
            keep.append(c)
    features = features[keep].copy()
    feature_cols = keep

    meta = {
        "region_probability_cols_detected": region_prob_cols,
        "region_label_col_detected": region_label_col,
        "region_onehot_cols": region_onehot_cols,
        "region_levels": region_levels,
        "celltype_col_detected": celltype_col,
        "celltype_onehot_cols": cell_onehot_cols,
        "celltype_levels": cell_levels,
        "composition_cols_detected": composition_cols,
        "score_cols_detected": score_cols,
        "feature_cols_final": feature_cols,
    }

    return features, feature_cols, meta


# -----------------------------------------------------------------------------
# kNN neighborhood construction
# -----------------------------------------------------------------------------

def compute_knn_indices(coords: np.ndarray, k: int):
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(coords)), metric="euclidean")
        nn.fit(coords)
        dists, inds = nn.kneighbors(coords)
        # Remove self.
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
            raise RuntimeError(
                "Neither sklearn.neighbors nor scipy.spatial.cKDTree could build kNN. "
                f"Original error: {e}"
            )


def build_knn_neighbor_profile(
    df: pd.DataFrame,
    features: pd.DataFrame,
    feature_cols: List[str],
    k: int,
    sample_col: Optional[str],
) -> Tuple[pd.DataFrame, Dict]:
    out = df.copy()

    out["knn_neighbor_profile_available"] = False
    out["knn_k_requested"] = k
    out["knn_k_effective"] = np.nan
    out["knn_mean_distance"] = np.nan
    out["knn_median_distance"] = np.nan
    out["knn_local_density_proxy"] = np.nan

    for c in feature_cols:
        out[f"neighbor_mean_{safe_name(c)}"] = np.nan
        out[f"neighbor_delta_vs_self_{safe_name(c)}"] = np.nan

    valid_coord = out[["spatial_x", "spatial_y"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)
    valid_idx_all = out.index[valid_coord].tolist()

    if not valid_idx_all:
        return out, {
            "status": "no_valid_coordinates",
            "n_valid_coordinates": 0,
            "feature_cols": feature_cols,
        }

    if sample_col and sample_col in out.columns:
        group_map = out.loc[valid_idx_all].groupby(sample_col, dropna=False).groups
        groups = [list(v) for v in group_map.values()]
    else:
        groups = [valid_idx_all]

    feature_values = features.reindex(out.index).fillna(0.0)

    n_processed = 0
    group_summaries = []

    for idx_labels in groups:
        idx_labels = list(idx_labels)
        if len(idx_labels) <= 1:
            continue

        k_eff = min(k, len(idx_labels) - 1)
        if k_eff <= 0:
            continue

        coords = out.loc[idx_labels, ["spatial_x", "spatial_y"]].apply(pd.to_numeric, errors="coerce").values
        dists, neigh = compute_knn_indices(coords, k=k_eff)

        feat = feature_values.loc[idx_labels, feature_cols].values if feature_cols else np.zeros((len(idx_labels), 0))

        if feat.shape[1] > 0:
            neigh_mean = feat[neigh].mean(axis=1)
            delta = neigh_mean - feat

            for j, c in enumerate(feature_cols):
                out.loc[idx_labels, f"neighbor_mean_{safe_name(c)}"] = neigh_mean[:, j]
                out.loc[idx_labels, f"neighbor_delta_vs_self_{safe_name(c)}"] = delta[:, j]

        out.loc[idx_labels, "knn_neighbor_profile_available"] = True
        out.loc[idx_labels, "knn_k_effective"] = k_eff
        out.loc[idx_labels, "knn_mean_distance"] = dists.mean(axis=1)
        out.loc[idx_labels, "knn_median_distance"] = np.median(dists, axis=1)
        out.loc[idx_labels, "knn_local_density_proxy"] = 1.0 / (dists.mean(axis=1) + 1e-8)

        n_processed += len(idx_labels)
        group_summaries.append({
            "n": int(len(idx_labels)),
            "k_eff": int(k_eff),
            "mean_knn_distance": float(np.nanmean(dists)),
        })

    meta = {
        "status": "ok",
        "n_valid_coordinates": int(len(valid_idx_all)),
        "n_processed": int(n_processed),
        "n_groups": int(len(groups)),
        "group_summaries_head": group_summaries[:20],
        "feature_cols": feature_cols,
    }

    return out, meta


def add_entropy_boundary_metrics(profile: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    out = profile.copy()

    neighbor_region_cols = [
        c for c in out.columns
        if c.startswith("neighbor_mean_feature_region") or c.startswith("neighbor_mean_feature_regionprob")
    ]
    neighbor_celltype_cols = [
        c for c in out.columns
        if c.startswith("neighbor_mean_feature_celltype")
    ]
    neighbor_comp_cols = [
        c for c in out.columns
        if c.startswith("neighbor_mean_feature_comp")
    ]

    meta = {
        "neighbor_region_cols": neighbor_region_cols,
        "neighbor_celltype_cols": neighbor_celltype_cols,
        "neighbor_comp_cols": neighbor_comp_cols,
    }

    if neighbor_region_cols:
        mat = out[neighbor_region_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        out["knn_neighbor_region_entropy"] = entropy_rows(mat)

        row_sum = mat.sum(axis=1, keepdims=True)
        p = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)
        out["knn_neighbor_region_boundary_score"] = 1.0 - np.nanmax(p, axis=1)
        out["knn_neighbor_region_dominant_fraction"] = np.nanmax(p, axis=1)
    else:
        out["knn_neighbor_region_entropy"] = np.nan
        out["knn_neighbor_region_boundary_score"] = np.nan
        out["knn_neighbor_region_dominant_fraction"] = np.nan

    if neighbor_celltype_cols:
        mat = out[neighbor_celltype_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        out["knn_neighbor_celltype_entropy"] = entropy_rows(mat)

        row_sum = mat.sum(axis=1, keepdims=True)
        p = np.divide(mat, row_sum, out=np.zeros_like(mat), where=row_sum > 0)
        out["knn_neighbor_celltype_mixedness"] = 1.0 - np.nanmax(p, axis=1)
        out["knn_neighbor_celltype_dominant_fraction"] = np.nanmax(p, axis=1)
    else:
        out["knn_neighbor_celltype_entropy"] = np.nan
        out["knn_neighbor_celltype_mixedness"] = np.nan
        out["knn_neighbor_celltype_dominant_fraction"] = np.nan

    if neighbor_comp_cols:
        mat = out[neighbor_comp_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        out["knn_neighbor_composition_entropy"] = entropy_rows(mat)
    else:
        out["knn_neighbor_composition_entropy"] = np.nan

    return out, meta


def make_light_table(profile: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    base_cols = []

    for c in [
        "coordinate_merge_status",
        "coordinate_source_file",
        "spatial_x",
        "spatial_y",
        "knn_neighbor_profile_available",
        "knn_k_requested",
        "knn_k_effective",
        "knn_mean_distance",
        "knn_median_distance",
        "knn_local_density_proxy",
        "knn_neighbor_region_entropy",
        "knn_neighbor_region_boundary_score",
        "knn_neighbor_region_dominant_fraction",
        "knn_neighbor_celltype_entropy",
        "knn_neighbor_celltype_mixedness",
        "knn_neighbor_celltype_dominant_fraction",
        "knn_neighbor_composition_entropy",
    ]:
        if c in profile.columns:
            base_cols.append(c)

    # Add likely identifiers / annotations.
    for c in [
        meta.get("virtual_id_col"),
        meta.get("sample_col"),
        meta.get("region_label_col_detected"),
        meta.get("celltype_col_detected"),
    ]:
        if c and c in profile.columns and c not in base_cols:
            base_cols.insert(0, c)

    # Add region and celltype neighbor means.
    extra = [
        c for c in profile.columns
        if c.startswith("neighbor_mean_feature_region")
        or c.startswith("neighbor_mean_feature_celltype")
        or c.startswith("neighbor_delta_vs_self_feature_region")
    ][:120]

    cols = []
    seen = set()
    for c in base_cols + extra:
        if c and c in profile.columns and c not in seen:
            seen.add(c)
            cols.append(c)

    return profile[cols].copy()


def summarize_profile(profile: pd.DataFrame, meta: Dict) -> pd.DataFrame:
    rows = []

    sample_col = meta.get("sample_col")
    region_col = meta.get("region_label_col_detected")
    celltype_col = meta.get("celltype_col_detected")

    group_cols = []
    if sample_col and sample_col in profile.columns:
        group_cols.append(sample_col)
    if region_col and region_col in profile.columns:
        group_cols.append(region_col)
    if celltype_col and celltype_col in profile.columns:
        group_cols.append(celltype_col)

    metric_cols = [
        "knn_neighbor_profile_available",
        "knn_mean_distance",
        "knn_local_density_proxy",
        "knn_neighbor_region_entropy",
        "knn_neighbor_region_boundary_score",
        "knn_neighbor_celltype_entropy",
        "knn_neighbor_celltype_mixedness",
        "knn_neighbor_composition_entropy",
    ]
    metric_cols = [c for c in metric_cols if c in profile.columns]

    if group_cols:
        grouped = profile.groupby(group_cols, dropna=False)
        for keys, sub in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {g: k for g, k in zip(group_cols, keys)}
            row["n"] = len(sub)
            for c in metric_cols:
                if c == "knn_neighbor_profile_available":
                    row[f"{c}_sum"] = int(sub[c].fillna(False).astype(bool).sum())
                else:
                    row[f"{c}_mean"] = pd.to_numeric(sub[c], errors="coerce").mean()
            rows.append(row)
    else:
        row = {"group": "all", "n": len(profile)}
        for c in metric_cols:
            if c == "knn_neighbor_profile_available":
                row[f"{c}_sum"] = int(profile[c].fillna(False).astype(bool).sum())
            else:
                row[f"{c}_mean"] = pd.to_numeric(profile[c], errors="coerce").mean()
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE_DEFAULT))
    ap.add_argument("--virtual_cell_table", default=None)
    ap.add_argument("--coord_metadata", default=None, help="Comma-separated coordinate metadata files. Optional.")
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT))
    ap.add_argument("--knn", type=int, default=12)
    ap.add_argument("--max_celltypes", type=int, default=30)
    ap.add_argument("--max_regions", type=int, default=20)
    ap.add_argument("--max_cells", type=int, default=0, help="Optional subsampling for debugging. 0 = all.")
    ap.add_argument("--no_order_merge", action="store_true", help="Disable row-order coordinate merge fallback.")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 90)
    log("53b rebuild virtual-cell kNN neighborhood profile")
    log("=" * 90)
    log(f"base={base}")
    log(f"outdir={outdir}")
    log(f"knn={args.knn}")

    virtual_path = find_virtual_cell_table(base, args.virtual_cell_table)
    if virtual_path is None:
        raise FileNotFoundError("Could not find virtual_cell_state_table.csv. Use --virtual_cell_table.")

    log(f"[virtual] using: {virtual_path}")
    virtual = pd.read_csv(virtual_path, low_memory=False)

    if args.max_cells and args.max_cells > 0 and len(virtual) > args.max_cells:
        virtual = virtual.sample(args.max_cells, random_state=0).reset_index(drop=True)
        log(f"[virtual] sampled max_cells={args.max_cells}")

    virtual_id_col = detect_id_col(virtual)
    sample_col_initial = detect_sample_col(virtual)

    log(f"[virtual] shape={virtual.shape}")
    log(f"[virtual] id_col={virtual_id_col}")
    log(f"[virtual] sample_col={sample_col_initial}")

    # Discover metadata files and merge coordinates.
    coord_paths = discover_coordinate_metadata_files(base, args.coord_metadata)
    log(f"[coords] candidate metadata files={len(coord_paths)}")

    merged, merge_meta = merge_coordinates(
        virtual,
        coord_paths,
        user_coord_file=args.coord_metadata,
        allow_order_merge=not args.no_order_merge,
    )

    matched = int(merged[["spatial_x", "spatial_y"]].apply(pd.to_numeric, errors="coerce").notna().all(axis=1).sum()) if "spatial_x" in merged.columns and "spatial_y" in merged.columns else 0

    log(f"[coords] merge_status={merge_meta.get('status')}")
    log(f"[coords] source={merge_meta.get('coordinate_source')}")
    log(f"[coords] matched_rows={matched}/{len(merged)}")

    # Save coordinate-merged virtual table.
    with_coords_path = outdir / "virtual_cell_state_table_with_coords.csv"
    merged.to_csv(with_coords_path, index=False)

    if matched == 0:
        # Write audit report and stop gracefully.
        metadata = {
            "status": "failed_no_coordinates",
            "virtual_path": str(virtual_path),
            "merge_meta": merge_meta,
            "n_virtual_rows": int(len(merged)),
            "n_coordinate_matched": int(matched),
        }
        metadata_path = outdir / "virtual_cell_knn_neighbor_profile_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        report_path = outdir / "virtual_cell_knn_neighbor_profile_report.txt"
        lines = []
        lines.append("53b virtual-cell kNN neighborhood profile report")
        lines.append("=" * 90)
        lines.append("FAILED: no spatial coordinates could be merged.")
        lines.append(f"virtual_path: {virtual_path}")
        lines.append(f"virtual_rows: {len(merged)}")
        lines.append(f"merge_status: {merge_meta.get('status')}")
        lines.append("")
        lines.append("Top coordinate metadata candidates:")
        for rep in merge_meta.get("candidate_reports", [])[:20]:
            lines.append(str(rep))
        report_path.write_text("\n".join(lines), encoding="utf-8")

        log("[DONE with warning] No coordinates merged; cannot build kNN profile.")
        log(f"Saved coordinate-merge attempt: {with_coords_path}")
        log(f"Saved metadata: {metadata_path}")
        log(f"Saved report: {report_path}")
        return

    # Detect sample/feature columns after coordinate merge.
    sample_col = detect_sample_col(merged)
    region_prob_cols = detect_region_probability_columns(merged)
    region_label_col = detect_region_label_col(merged)
    celltype_col = detect_celltype_col(merged)
    composition_cols = detect_composition_columns(merged)
    score_cols = detect_score_columns(merged)

    log(f"[features] sample_col={sample_col}")
    log(f"[features] region_prob_cols={region_prob_cols}")
    log(f"[features] region_label_col={region_label_col}")
    log(f"[features] celltype_col={celltype_col}")
    log(f"[features] composition_cols_n={len(composition_cols)}")
    log(f"[features] score_cols_n={len(score_cols)}")

    features, feature_cols, feature_meta = build_feature_matrix(
        merged,
        max_celltypes=args.max_celltypes,
        max_regions=args.max_regions,
    )

    log(f"[features] final feature cols={len(feature_cols)}")

    if not feature_cols:
        log("[features] WARNING: no region/celltype/composition/score features detected. kNN distances will still be computed, but profile features will be empty.")

    profile, knn_meta = build_knn_neighbor_profile(
        df=merged,
        features=features,
        feature_cols=feature_cols,
        k=args.knn,
        sample_col=sample_col,
    )

    profile, entropy_meta = add_entropy_boundary_metrics(profile)

    # Save outputs.
    profile_path = outdir / "virtual_cell_knn_neighbor_profile.csv"
    light_path = outdir / "virtual_cell_knn_neighbor_profile_light.csv"
    summary_path = outdir / "virtual_cell_knn_neighbor_profile_summary.csv"
    metadata_path = outdir / "virtual_cell_knn_neighbor_profile_metadata.json"
    report_path = outdir / "virtual_cell_knn_neighbor_profile_report.txt"

    profile.to_csv(profile_path, index=False)

    meta_for_light = {
        **feature_meta,
        "virtual_id_col": virtual_id_col,
        "sample_col": sample_col,
        "region_label_col_detected": region_label_col,
        "celltype_col_detected": celltype_col,
    }
    light = make_light_table(profile, meta_for_light)
    light.to_csv(light_path, index=False)

    summary = summarize_profile(profile, meta_for_light)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "status": "ok",
        "base": str(base),
        "virtual_path": str(virtual_path),
        "outdir": str(outdir),
        "n_virtual_rows": int(len(profile)),
        "n_coordinate_matched": int(matched),
        "coordinate_merge": merge_meta,
        "virtual_id_col": virtual_id_col,
        "sample_col": sample_col,
        "feature_meta": feature_meta,
        "knn_meta": knn_meta,
        "entropy_meta": entropy_meta,
        "outputs": {
            "virtual_cell_state_table_with_coords": str(with_coords_path),
            "virtual_cell_knn_neighbor_profile": str(profile_path),
            "virtual_cell_knn_neighbor_profile_light": str(light_path),
            "virtual_cell_knn_neighbor_profile_summary": str(summary_path),
            "report": str(report_path),
        },
    }

    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Write report.
    lines = []
    lines.append("53b virtual-cell kNN neighborhood profile report")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Purpose:")
    lines.append("Merge spatial coordinates into virtual_cell_state_table and rebuild a true spatial kNN neighborhood profile.")
    lines.append("")
    lines.append(f"virtual_path: {virtual_path}")
    lines.append(f"virtual_rows: {len(profile)}")
    lines.append(f"coordinate_matched_rows: {matched}")
    lines.append(f"coordinate_merge_status: {merge_meta.get('status')}")
    lines.append(f"coordinate_source: {merge_meta.get('coordinate_source')}")
    lines.append(f"coordinate_metadata_x_col: {merge_meta.get('metadata_x_col')}")
    lines.append(f"coordinate_metadata_y_col: {merge_meta.get('metadata_y_col')}")
    lines.append(f"coordinate_metadata_id_col: {merge_meta.get('metadata_id_col')}")
    lines.append("")
    lines.append("Detected feature columns:")
    lines.append(f"sample_col: {sample_col}")
    lines.append(f"region_probability_cols: {region_prob_cols}")
    lines.append(f"region_label_col: {region_label_col}")
    lines.append(f"celltype_col: {celltype_col}")
    lines.append(f"composition_cols_n: {len(composition_cols)}")
    lines.append(f"score_cols_n: {len(score_cols)}")
    lines.append(f"final_feature_cols_n: {len(feature_cols)}")
    lines.append("")
    lines.append("kNN:")
    lines.append(json.dumps(knn_meta, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("Entropy/boundary profile:")
    for c in [
        "knn_neighbor_region_entropy",
        "knn_neighbor_region_boundary_score",
        "knn_neighbor_region_dominant_fraction",
        "knn_neighbor_celltype_entropy",
        "knn_neighbor_celltype_mixedness",
        "knn_neighbor_celltype_dominant_fraction",
        "knn_neighbor_composition_entropy",
        "knn_mean_distance",
        "knn_local_density_proxy",
    ]:
        if c in profile.columns:
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
        "This output provides a true coordinate-based kNN neighborhood profile for virtual cells/spots. "
        "It can support neighbor_i in the algorithmic description. It remains a kNN-derived profile/proxy, "
        "not a trained neural NeighborHead(z_i), unless a formal NeighborHead is trained later."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    log("=" * 90)
    log("DONE 53b virtual-cell kNN neighborhood profile")
    log("=" * 90)
    log(f"Saved: {with_coords_path}")
    log(f"Saved: {profile_path}")
    log(f"Saved: {light_path}")
    log(f"Saved: {summary_path}")
    log(f"Saved: {metadata_path}")
    log(f"Saved: {report_path}")
    log("")
    log("Key checks:")
    log(f"coordinate_matched_rows={matched}/{len(profile)}")
    log(f"knn_profile_available={int(profile['knn_neighbor_profile_available'].fillna(False).astype(bool).sum())}/{len(profile)}")
    if "knn_neighbor_region_boundary_score" in profile.columns:
        log(f"region_boundary_score_nonnull={pd.to_numeric(profile['knn_neighbor_region_boundary_score'], errors='coerce').notna().sum()}")
    if "knn_neighbor_celltype_mixedness" in profile.columns:
        log(f"celltype_mixedness_nonnull={pd.to_numeric(profile['knn_neighbor_celltype_mixedness'], errors='coerce').notna().sum()}")

    print("")
    print("Summary head:")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
