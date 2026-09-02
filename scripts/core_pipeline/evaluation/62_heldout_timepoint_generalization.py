#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
62_heldout_timepoint_generalization.py

Purpose
-------
Step 62: held-out timepoint / held-out sample generalization.

This step evaluates whether StrokeNiche/NichePerturb readouts generalize across
timepoints and samples.

Primary splits
--------------
1. train D1 + D3 -> test D7
2. train D1 + D7 -> test D3
3. train D3 + D7 -> test D1

Additional split
----------------
leave-one-sample-out, if a sample column is available.
If no independent sample column is found, timepoint is used as a sample-like group.

Metrics
-------
Classification:
  - region macro-F1
  - balanced accuracy
  - accuracy

Regression:
  - repair score RMSE / MAE / R2 / Spearman
  - neighbor composition RMSE / MAE / R2
  - cell-type composition RMSE / MAE / R2

Trajectory:
  - held-out centroid distance to train trajectory line
  - held-out projection consistency
  - trajectory consistency score

Perturbation ranking stability:
  - train/test Spearman correlation
  - train/test top-k Jaccard overlap

Inputs
------
Default primary input:
  results/step8_strokeniche_perturbmap/niche_perturb_61_strict_fig5_umap_61d_v3/
    niche_module_scores_by_cell.csv
    niche_perturbation_cell_level_state_editing.csv
    niche_perturbation_ranking.csv

Optional neighbor input:
  results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/
    virtual_cell_knn_neighbor_profile_RCTD.csv

Outputs
-------
results/step8_strokeniche_perturbmap/generalization_62/
  step62_modeling_input_table.csv
  step62_feature_target_metadata.json
  step62_heldout_timepoint_metrics.csv
  step62_leave_one_sample_metrics.csv
  step62_all_split_metrics.csv
  step62_perturbation_ranking_stability.csv
  step62_prediction_examples.csv
  step62_repair_monotonicity_by_timepoint.csv
  step62_trajectory_consistency.csv
  step62_generalization_summary.csv
  step62_generalization_report.txt
  step62_generalization_report.json
  Fig_Step62_Generalization_Summary.pdf/svg/png
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

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder


BASE = Path("/mnt/h/vir/ST")

DEFAULT_IN61_STRICT = BASE / "results/step8_strokeniche_perturbmap/niche_perturb_61_strict_fig5_umap_61d_v3"
DEFAULT_NEIGHBOR53C = BASE / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/virtual_cell_knn_neighbor_profile_RCTD.csv"
DEFAULT_OUT = BASE / "results/step8_strokeniche_perturbmap/generalization_62"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def read_csv(path, required=True):
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(p)
        return None
    if p.stat().st_size == 0:
        if required:
            raise RuntimeError(f"Empty file: {p}")
        return None
    return pd.read_csv(p, low_memory=False)


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def normalize_state(x):
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")
    if any(k in s for k in ["lesion_core", "core_like", "lesion", "core"]):
        return "lesion-core-like"
    if any(k in s for k in ["peri_infarct", "peri", "penumbra"]):
        return "peri-infarct"
    if any(k in s for k in ["remote_like", "remote", "normal", "healthy", "intact"]):
        return "remote-like"
    return "other"


def extract_timepoint_from_text(x):
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


def timepoint_to_day(tp):
    s = str(tp).upper()
    m = re.search(r"D\s*([0-9]{1,2})", s)
    if m:
        return int(m.group(1))
    m = re.search(r"([0-9]{1,2})", s)
    if m:
        return int(m.group(1))
    return np.nan


def detect_timepoint(df):
    out = pd.Series("", index=df.index, dtype=object)

    for c in ["timepoint", "time_point", "time", "sample", "sample_id", "orig.ident", "orig_ident", "batch"]:
        if c in df.columns:
            v = df[c].map(extract_timepoint_from_text)
            out = out.where(out.astype(str).ne(""), v)

    if "obs_name" in df.columns:
        v = df["obs_name"].map(extract_timepoint_from_text)
        out = out.where(out.astype(str).ne(""), v)

    # fallback to unknown
    out = out.replace("", np.nan)
    return out


def detect_sample_col(df):
    candidates = ["sample", "sample_id", "orig.ident", "orig_ident", "batch", "library", "slide", "section"]
    for c in candidates:
        if c in df.columns:
            vals = df[c].astype(str)
            if vals.nunique() > 1:
                return c
    return None


def safe_numeric_df(df, cols):
    if not cols:
        return pd.DataFrame(index=df.index)
    return df[cols].apply(pd.to_numeric, errors="coerce")


def spearman_corr(a, b):
    x = pd.Series(a).astype(float)
    y = pd.Series(b).astype(float)
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return np.nan
    return float(x[valid].corr(y[valid], method="spearman"))


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def make_reg_metrics(y_true, y_pred, prefix):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 3:
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_r2": np.nan,
            f"{prefix}_spearman": np.nan,
            f"{prefix}_n": int(valid.sum()),
        }

    yt = y_true[valid]
    yp = y_pred[valid]

    return {
        f"{prefix}_rmse": rmse(yt, yp),
        f"{prefix}_mae": float(mean_absolute_error(yt, yp)),
        f"{prefix}_r2": float(r2_score(yt, yp)) if len(np.unique(yt)) > 1 else np.nan,
        f"{prefix}_spearman": spearman_corr(yt, yp),
        f"{prefix}_n": int(valid.sum()),
    }


def make_multioutput_metrics(Y_true, Y_pred, prefix):
    if Y_true is None or Y_pred is None:
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_r2_mean": np.nan,
            f"{prefix}_n_targets": 0,
            f"{prefix}_n_rows": 0,
        }

    Y_true = np.asarray(Y_true, dtype=float)
    Y_pred = np.asarray(Y_pred, dtype=float)

    valid = np.isfinite(Y_true) & np.isfinite(Y_pred)
    if valid.sum() == 0:
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_r2_mean": np.nan,
            f"{prefix}_n_targets": int(Y_true.shape[1]) if Y_true.ndim == 2 else 0,
            f"{prefix}_n_rows": int(Y_true.shape[0]) if Y_true.ndim == 2 else 0,
        }

    mse = np.nanmean((Y_true - Y_pred) ** 2)
    mae = np.nanmean(np.abs(Y_true - Y_pred))

    r2s = []
    for j in range(Y_true.shape[1]):
        yt = Y_true[:, j]
        yp = Y_pred[:, j]
        m = np.isfinite(yt) & np.isfinite(yp)
        if m.sum() >= 3 and len(np.unique(yt[m])) > 1:
            try:
                r2s.append(r2_score(yt[m], yp[m]))
            except Exception:
                pass

    return {
        f"{prefix}_rmse": float(np.sqrt(mse)),
        f"{prefix}_mae": float(mae),
        f"{prefix}_r2_mean": float(np.mean(r2s)) if r2s else np.nan,
        f"{prefix}_n_targets": int(Y_true.shape[1]),
        f"{prefix}_n_rows": int(Y_true.shape[0]),
    }


def minmax(s, fill=0.0):
    x = pd.to_numeric(pd.Series(s), errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.full(len(x), fill), index=x.index)
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.where(x.notna(), 0.5, fill), index=x.index)
    return ((x - mn) / (mx - mn)).fillna(fill)


# -----------------------------------------------------------------------------
# Load and prepare data
# -----------------------------------------------------------------------------

def load_strict_tables(in61):
    in61 = Path(in61)

    module = read_csv(in61 / "niche_module_scores_by_cell.csv")
    cell = read_csv(in61 / "niche_perturbation_cell_level_state_editing.csv")
    ranking = read_csv(in61 / "niche_perturbation_ranking.csv")

    if "obs_name" not in module.columns:
        raise RuntimeError("module table lacks obs_name.")
    if "obs_name" not in cell.columns:
        raise RuntimeError("cell-level perturbation table lacks obs_name.")

    return module, cell, ranking


def merge_neighbor53c(module, neighbor53c_path):
    p = Path(neighbor53c_path)
    if not p.exists():
        return module.copy(), {"neighbor53c_exists": False, "merged": False}

    nb = read_csv(p)
    if "obs_name" not in nb.columns:
        return module.copy(), {"neighbor53c_exists": True, "merged": False, "reason": "no obs_name"}

    base = module.copy()
    add_cols = ["obs_name"] + [c for c in nb.columns if c != "obs_name" and c not in base.columns]
    nb2 = nb[add_cols].copy()

    merged = base.merge(nb2, on="obs_name", how="left")

    meta = {
        "neighbor53c_exists": True,
        "merged": True,
        "neighbor53c_path": str(p),
        "neighbor53c_shape": list(nb.shape),
        "added_cols": len(add_cols) - 1,
        "matched_rows": int(merged[add_cols[1:]].notna().any(axis=1).sum()) if len(add_cols) > 1 else 0,
    }
    return merged, meta


def detect_targets_and_features(df, include_region_prob_features=False, feature_mode="full"):
    # Target: state group / region
    if "state_group" not in df.columns:
        raise RuntimeError("modeling table lacks state_group target.")
    df["state_group"] = df["state_group"].map(normalize_state)

    # Target: repair score
    repair_col = "repair_score" if "repair_score" in df.columns else None

    # Target: neighbor / cell type composition
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    ncols_norm = {c: norm_col(c) for c in numeric_cols}

    def is_composition_col(c):
        nc = ncols_norm[c]
        if any(k in nc for k in [
            "entropy", "mixedness", "boundary", "distance", "density",
            "delta", "rank", "score_norm", "repair_score", "core_probability",
            "peri_probability", "remote_probability", "coord", "umap", "strict"
        ]):
            return False
        if any(k in nc for k in ["rctd", "celltype", "neighbor_profile", "knn_neighbor"]):
            return True
        return False

    neighbor_cols = [c for c in numeric_cols if is_composition_col(c)]

    # RCTD/cell-type composition targets
    known_celltype_tokens = [
        "astro", "microglia", "endothelial", "neuron", "oligo", "opc",
        "pericyte", "macrophage", "myeloid", "ependymal", "fibroblast",
    ]
    celltype_cols = [
        c for c in neighbor_cols
        if ("rctd" in ncols_norm[c]) or any(t in ncols_norm[c] for t in known_celltype_tokens)
    ]

    # Feature columns
    module_cols = []
    for c in numeric_cols:
        if not c.startswith("module_"):
            continue
        nc = norm_col(c)
        if any(k in nc for k in ["id_col", "obs_name", "barcode", "timepoint", "key", "raw", "matched", "coord", "method", "source"]):
            continue
        module_cols.append(c)

    context_cols = [
        c for c in [
            "endothelial_context", "astrocyte_context", "microglia_context",
            "neuron_context", "lesion_niche_context", "remote_offtarget_context"
        ]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    coord_cols = []
    if feature_mode in ["full", "coords_only", "module_plus_coords"]:
        for c in ["strict_coord_x", "strict_coord_y", "coord_x", "coord_y", "umap_1", "umap_2"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                coord_cols.append(c)

    region_prob_cols = []
    if include_region_prob_features:
        for c in ["core_probability", "peri_probability", "remote_probability"]:
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
                region_prob_cols.append(c)

    if feature_mode == "coords_only":
        feature_cols = coord_cols
    elif feature_mode == "modules_only":
        feature_cols = module_cols + context_cols
    else:
        feature_cols = module_cols + context_cols + coord_cols + region_prob_cols

    # Exclude targets and IDs
    exclude = set(neighbor_cols + celltype_cols)
    if repair_col:
        exclude.add(repair_col)
    exclude.update([
        "core_probability", "peri_probability", "remote_probability",
        "niche_perturbation_score_norm",
    ])

    feature_cols = [c for c in feature_cols if c not in exclude]
    feature_cols = list(dict.fromkeys(feature_cols))

    # Drop nearly empty/constant features
    clean_features = []
    for c in feature_cols:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() < 0.50:
            continue
        if x.nunique(dropna=True) <= 1:
            continue
        clean_features.append(c)

    meta = {
        "feature_mode": feature_mode,
        "include_region_prob_features": include_region_prob_features,
        "n_features": len(clean_features),
        "feature_cols": clean_features,
        "region_target_col": "state_group",
        "repair_col": repair_col,
        "n_neighbor_targets": len(neighbor_cols),
        "neighbor_target_cols": neighbor_cols,
        "n_celltype_targets": len(celltype_cols),
        "celltype_target_cols": celltype_cols,
        "n_module_features": len(module_cols),
        "n_context_features": len(context_cols),
        "n_coord_features": len(coord_cols),
    }

    return clean_features, repair_col, neighbor_cols, celltype_cols, meta


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

def make_classifier(args):
    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        class_weight="balanced_subsample",
        n_jobs=args.n_jobs,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", clf),
    ])


def make_regressor(args):
    reg = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth if args.max_depth > 0 else None,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", reg),
    ])


def run_region_classification(df, feature_cols, train_mask, test_mask, args):
    out = {}
    pred_df = pd.DataFrame()

    y_train = df.loc[train_mask, "state_group"].astype(str)
    y_test = df.loc[test_mask, "state_group"].astype(str)

    valid_train = y_train.notna()
    valid_test = y_test.notna()

    if valid_train.sum() < 20 or valid_test.sum() < 5 or y_train[valid_train].nunique() < 2:
        out.update({
            "region_macro_f1": np.nan,
            "region_balanced_accuracy": np.nan,
            "region_accuracy": np.nan,
            "region_n_train": int(valid_train.sum()),
            "region_n_test": int(valid_test.sum()),
            "region_n_train_classes": int(y_train[valid_train].nunique()),
            "region_n_test_classes": int(y_test[valid_test].nunique()),
        })
        return out, pred_df

    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]

    model = make_classifier(args)
    model.fit(X_train, y_train)

    y_pred = pd.Series(model.predict(X_test), index=X_test.index)

    labels_union = sorted(set(y_train.unique()).union(set(y_test.unique())).union(set(y_pred.unique())))

    out.update({
        "region_macro_f1": float(f1_score(y_test, y_pred, labels=labels_union, average="macro", zero_division=0)),
        "region_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "region_accuracy": float(accuracy_score(y_test, y_pred)),
        "region_n_train": int(len(y_train)),
        "region_n_test": int(len(y_test)),
        "region_n_train_classes": int(y_train.nunique()),
        "region_n_test_classes": int(y_test.nunique()),
    })

    pred_df = pd.DataFrame({
        "obs_name": df.loc[test_mask, "obs_name"].values,
        "prediction_task": "region",
        "y_true": y_test.values,
        "y_pred": y_pred.values,
    })

    if "timepoint" in df.columns:
        pred_df["timepoint"] = df.loc[test_mask, "timepoint"].values
    if "sample_id_for_split" in df.columns:
        pred_df["sample_id_for_split"] = df.loc[test_mask, "sample_id_for_split"].values

    return out, pred_df


def run_repair_regression(df, feature_cols, repair_col, train_mask, test_mask, args):
    out = {}
    pred_df = pd.DataFrame()

    if repair_col is None:
        return {
            "repair_rmse": np.nan,
            "repair_mae": np.nan,
            "repair_r2": np.nan,
            "repair_spearman": np.nan,
            "repair_n": 0,
        }, pred_df

    y_train = pd.to_numeric(df.loc[train_mask, repair_col], errors="coerce")
    y_test = pd.to_numeric(df.loc[test_mask, repair_col], errors="coerce")

    valid_train = y_train.notna()
    valid_test = y_test.notna()

    if valid_train.sum() < 20 or valid_test.sum() < 5:
        return {
            "repair_rmse": np.nan,
            "repair_mae": np.nan,
            "repair_r2": np.nan,
            "repair_spearman": np.nan,
            "repair_n": int(valid_test.sum()),
        }, pred_df

    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]

    model = make_regressor(args)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = make_reg_metrics(y_test.values, y_pred, "repair")
    out.update(metrics)

    pred_df = pd.DataFrame({
        "obs_name": df.loc[test_mask, "obs_name"].values,
        "prediction_task": "repair_score",
        "y_true": y_test.values,
        "y_pred": y_pred,
    })

    if "timepoint" in df.columns:
        pred_df["timepoint"] = df.loc[test_mask, "timepoint"].values
    if "sample_id_for_split" in df.columns:
        pred_df["sample_id_for_split"] = df.loc[test_mask, "sample_id_for_split"].values

    return out, pred_df


def run_multioutput_regression(df, feature_cols, target_cols, train_mask, test_mask, args, prefix):
    if not target_cols:
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_r2_mean": np.nan,
            f"{prefix}_n_targets": 0,
            f"{prefix}_n_rows": 0,
        }

    Y_train = safe_numeric_df(df.loc[train_mask], target_cols)
    Y_test = safe_numeric_df(df.loc[test_mask], target_cols)

    # Drop target columns with insufficient train values
    keep_cols = []
    for c in target_cols:
        if Y_train[c].notna().sum() >= 20 and Y_test[c].notna().sum() >= 5:
            if Y_train[c].nunique(dropna=True) > 1:
                keep_cols.append(c)

    if not keep_cols:
        return {
            f"{prefix}_rmse": np.nan,
            f"{prefix}_mae": np.nan,
            f"{prefix}_r2_mean": np.nan,
            f"{prefix}_n_targets": 0,
            f"{prefix}_n_rows": int(test_mask.sum()),
        }

    Y_train = Y_train[keep_cols]
    Y_test = Y_test[keep_cols]

    X_train = df.loc[train_mask, feature_cols]
    X_test = df.loc[test_mask, feature_cols]

    model = make_regressor(args)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    metrics = make_multioutput_metrics(Y_test.values, Y_pred, prefix)
    metrics[f"{prefix}_target_cols_used"] = ";".join(keep_cols[:80])
    return metrics


# -----------------------------------------------------------------------------
# Trajectory and monotonicity
# -----------------------------------------------------------------------------

def compute_repair_monotonicity(df, repair_col):
    if repair_col is None or "timepoint" not in df.columns:
        return pd.DataFrame(), {"repair_monotonicity_spearman": np.nan}

    tmp = df.copy()
    tmp["day"] = tmp["timepoint"].map(timepoint_to_day)
    tmp[repair_col] = pd.to_numeric(tmp[repair_col], errors="coerce")

    g = (
        tmp.dropna(subset=["day", repair_col])
        .groupby("timepoint")
        .agg(
            day=("day", "median"),
            n_cells=("obs_name", "count"),
            mean_repair=(repair_col, "mean"),
            median_repair=(repair_col, "median"),
        )
        .reset_index()
        .sort_values("day")
    )

    mono = spearman_corr(g["day"], g["mean_repair"]) if len(g) >= 3 else np.nan

    return g, {"repair_monotonicity_spearman": mono}


def compute_trajectory_consistency(df, split_name, train_timepoints, test_timepoint):
    coord_x = None
    coord_y = None
    for x, y in [
        ("strict_coord_x", "strict_coord_y"),
        ("coord_x", "coord_y"),
        ("umap_1", "umap_2"),
    ]:
        if x in df.columns and y in df.columns:
            coord_x, coord_y = x, y
            break

    if coord_x is None or "timepoint" not in df.columns:
        return {
            "split": split_name,
            "trajectory_available": False,
            "trajectory_consistency_score": np.nan,
        }

    tmp = df.copy()
    tmp["day"] = tmp["timepoint"].map(timepoint_to_day)
    tmp[coord_x] = pd.to_numeric(tmp[coord_x], errors="coerce")
    tmp[coord_y] = pd.to_numeric(tmp[coord_y], errors="coerce")

    cent = (
        tmp.dropna(subset=["day", coord_x, coord_y])
        .groupby("timepoint")
        .agg(
            day=("day", "median"),
            x=(coord_x, "median"),
            y=(coord_y, "median"),
            n_cells=("obs_name", "count"),
        )
        .reset_index()
    )

    train_cent = cent[cent["timepoint"].isin(train_timepoints)].copy()
    test_cent = cent[cent["timepoint"].astype(str) == str(test_timepoint)].copy()

    if len(train_cent) < 2 or len(test_cent) != 1:
        return {
            "split": split_name,
            "trajectory_available": False,
            "trajectory_consistency_score": np.nan,
            "reason": "not enough centroids",
        }

    train_cent = train_cent.sort_values("day")
    p0 = train_cent.iloc[0][["x", "y"]].astype(float).values
    p1 = train_cent.iloc[-1][["x", "y"]].astype(float).values
    pt = test_cent.iloc[0][["x", "y"]].astype(float).values

    v = p1 - p0
    denom = float(np.dot(v, v))
    if denom <= 1e-12:
        return {
            "split": split_name,
            "trajectory_available": False,
            "trajectory_consistency_score": np.nan,
            "reason": "zero train trajectory length",
        }

    projection = float(np.dot(pt - p0, v) / denom)
    closest = p0 + projection * v
    dist = float(np.linalg.norm(pt - closest))
    traj_len = float(np.linalg.norm(v))

    test_day = float(test_cent.iloc[0]["day"])
    train_days = train_cent["day"].astype(float).tolist()

    if test_day < min(train_days):
        expected_region = "before_train_interval"
        position_ok = projection <= 0
        position_penalty = max(0.0, projection)
    elif test_day > max(train_days):
        expected_region = "after_train_interval"
        position_ok = projection >= 1
        position_penalty = max(0.0, 1.0 - projection)
    else:
        expected_region = "inside_train_interval"
        position_ok = 0 <= projection <= 1
        position_penalty = max(0.0, -projection, projection - 1.0)

    distance_score = float(np.exp(-dist / max(traj_len, 1e-6)))
    position_score = float(np.exp(-abs(position_penalty)))
    consistency = distance_score * position_score

    return {
        "split": split_name,
        "trajectory_available": True,
        "train_timepoints": "+".join(train_timepoints),
        "test_timepoint": test_timepoint,
        "coord_x": coord_x,
        "coord_y": coord_y,
        "train_start_timepoint": train_cent.iloc[0]["timepoint"],
        "train_end_timepoint": train_cent.iloc[-1]["timepoint"],
        "projection_on_train_trajectory": projection,
        "distance_to_train_trajectory_line": dist,
        "train_trajectory_length": traj_len,
        "expected_region": expected_region,
        "position_ok": bool(position_ok),
        "distance_score": distance_score,
        "position_score": position_score,
        "trajectory_consistency_score": consistency,
    }


# -----------------------------------------------------------------------------
# Perturbation ranking stability
# -----------------------------------------------------------------------------

def compute_perturbation_scores(cell_subset):
    if cell_subset.empty or "perturbation_id" not in cell_subset.columns:
        return pd.DataFrame()

    required = [
        "predicted_repair_shift",
        "predicted_core_reversal",
        "predicted_BBB_stability_gain",
        "predicted_ferroptosis_reduction",
        "predicted_inflammatory_chemotaxis_reduction",
        "predicted_spatial_safety",
        "remote_offtarget_penalty",
        "neuron_context_risk",
    ]

    missing = [c for c in required if c not in cell_subset.columns]
    if missing:
        return pd.DataFrame()

    agg = (
        cell_subset.groupby("perturbation_id")
        .agg(
            repair_shift=("predicted_repair_shift", "mean"),
            core_reversal=("predicted_core_reversal", "mean"),
            BBB_stability_gain=("predicted_BBB_stability_gain", "mean"),
            ferroptosis_reduction=("predicted_ferroptosis_reduction", "mean"),
            inflammatory_chemotaxis_reduction=("predicted_inflammatory_chemotaxis_reduction", "mean"),
            spatial_safety=("predicted_spatial_safety", "mean"),
            remote_offtarget_penalty=("remote_offtarget_penalty", "mean"),
            neuron_context_risk=("neuron_context_risk", "mean"),
            n_cells=("obs_name", "count"),
        )
        .reset_index()
    )

    if len(agg) == 0:
        return agg

    for c in [
        "repair_shift",
        "core_reversal",
        "BBB_stability_gain",
        "ferroptosis_reduction",
        "inflammatory_chemotaxis_reduction",
        "spatial_safety",
    ]:
        agg[c + "_norm"] = minmax(agg[c], fill=0.0)

    risk = (
        0.60 * minmax(agg["remote_offtarget_penalty"], fill=0.0)
        + 0.40 * minmax(agg["neuron_context_risk"], fill=0.0)
    ).clip(0, 1)

    agg["spatial_risk_penalty_norm"] = risk

    agg["subset_niche_perturbation_score"] = (
        0.22 * agg["repair_shift_norm"]
        + 0.22 * agg["core_reversal_norm"]
        + 0.16 * agg["BBB_stability_gain_norm"]
        + 0.14 * agg["ferroptosis_reduction_norm"]
        + 0.16 * agg["inflammatory_chemotaxis_reduction_norm"]
        + 0.10 * agg["spatial_safety_norm"]
        - 0.08 * agg["spatial_risk_penalty_norm"]
    )
    agg["subset_niche_perturbation_score_norm"] = minmax(agg["subset_niche_perturbation_score"], fill=0.0)
    agg = agg.sort_values("subset_niche_perturbation_score_norm", ascending=False).reset_index(drop=True)
    agg["subset_rank"] = np.arange(1, len(agg) + 1)

    return agg


def jaccard_topk(a, b, k=5):
    A = set(a.head(k)["perturbation_id"].astype(str))
    B = set(b.head(k)["perturbation_id"].astype(str))
    if not A and not B:
        return np.nan
    return float(len(A & B) / max(1, len(A | B)))


def compute_ranking_stability(cell_df, split_name, train_obs, test_obs):
    if cell_df.empty or "obs_name" not in cell_df.columns:
        return {}, pd.DataFrame(), pd.DataFrame()

    train_cell = cell_df[cell_df["obs_name"].astype(str).isin(train_obs)].copy()
    test_cell = cell_df[cell_df["obs_name"].astype(str).isin(test_obs)].copy()

    train_rank = compute_perturbation_scores(train_cell)
    test_rank = compute_perturbation_scores(test_cell)

    if train_rank.empty or test_rank.empty:
        return {
            "perturbation_ranking_available": False,
            "perturbation_rank_spearman": np.nan,
            "perturbation_score_spearman": np.nan,
            "perturbation_top3_jaccard": np.nan,
            "perturbation_top5_jaccard": np.nan,
        }, train_rank, test_rank

    merged = train_rank[["perturbation_id", "subset_rank", "subset_niche_perturbation_score_norm"]].merge(
        test_rank[["perturbation_id", "subset_rank", "subset_niche_perturbation_score_norm"]],
        on="perturbation_id",
        suffixes=("_train", "_test"),
    )

    if len(merged) < 3:
        rank_s = np.nan
        score_s = np.nan
    else:
        rank_s = spearman_corr(merged["subset_rank_train"], merged["subset_rank_test"])
        score_s = spearman_corr(
            merged["subset_niche_perturbation_score_norm_train"],
            merged["subset_niche_perturbation_score_norm_test"],
        )

    metrics = {
        "perturbation_ranking_available": True,
        "perturbation_rank_spearman": rank_s,
        "perturbation_score_spearman": score_s,
        "perturbation_top3_jaccard": jaccard_topk(train_rank, test_rank, k=3),
        "perturbation_top5_jaccard": jaccard_topk(train_rank, test_rank, k=5),
        "perturbation_n_shared": int(len(merged)),
    }

    train_rank["split"] = split_name
    train_rank["subset"] = "train"
    test_rank["split"] = split_name
    test_rank["subset"] = "test"

    return metrics, train_rank, test_rank


# -----------------------------------------------------------------------------
# Split runner
# -----------------------------------------------------------------------------

def run_split(df, cell_df, feature_cols, repair_col, neighbor_cols, celltype_cols, split_name, train_mask, test_mask, args):
    row = {
        "split": split_name,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }

    if row["n_train"] < args.min_train_cells or row["n_test"] < args.min_test_cells:
        row["status"] = "skipped_too_few_cells"
        return row, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    row["status"] = "ok"

    # Region classification
    region_metrics, pred_region = run_region_classification(df, feature_cols, train_mask, test_mask, args)
    row.update(region_metrics)

    # Repair regression
    repair_metrics, pred_repair = run_repair_regression(df, feature_cols, repair_col, train_mask, test_mask, args)
    row.update(repair_metrics)

    # Neighbor composition
    neighbor_metrics = run_multioutput_regression(
        df, feature_cols, neighbor_cols, train_mask, test_mask, args, "neighbor"
    )
    row.update(neighbor_metrics)

    # Celltype composition
    celltype_metrics = run_multioutput_regression(
        df, feature_cols, celltype_cols, train_mask, test_mask, args, "celltype"
    )
    row.update(celltype_metrics)

    # Perturbation ranking stability
    train_obs = set(df.loc[train_mask, "obs_name"].astype(str))
    test_obs = set(df.loc[test_mask, "obs_name"].astype(str))
    rank_metrics, rank_train, rank_test = compute_ranking_stability(cell_df, split_name, train_obs, test_obs)
    row.update(rank_metrics)

    pred_all = pd.concat([pred_region, pred_repair], ignore_index=True) if len(pred_region) or len(pred_repair) else pd.DataFrame()
    if not pred_all.empty:
        pred_all["split"] = split_name

    rank_all = pd.concat([rank_train, rank_test], ignore_index=True) if len(rank_train) or len(rank_test) else pd.DataFrame()

    return row, pred_all, rank_all, pd.DataFrame([row])


def build_timepoint_splits(df):
    tps = sorted(
        [x for x in df["timepoint"].dropna().astype(str).unique()],
        key=lambda x: timepoint_to_day(x) if not pd.isna(timepoint_to_day(x)) else 999,
    )

    splits = []

    # Prefer D1/D3/D7 if present.
    desired = ["D1", "D3", "D7"]
    if all(tp in tps for tp in desired):
        split_defs = [
            (["D1", "D3"], "D7"),
            (["D1", "D7"], "D3"),
            (["D3", "D7"], "D1"),
        ]
    else:
        split_defs = []
        for test_tp in tps:
            train_tps = [x for x in tps if x != test_tp]
            if len(train_tps) >= 1:
                split_defs.append((train_tps, test_tp))

    for train_tps, test_tp in split_defs:
        name = f"timepoint_train_{'+'.join(train_tps)}__test_{test_tp}"
        train_mask = df["timepoint"].astype(str).isin(train_tps)
        test_mask = df["timepoint"].astype(str).eq(str(test_tp))
        splits.append((name, train_tps, test_tp, train_mask, test_mask))

    return splits


def build_loso_splits(df):
    sample_col = "sample_id_for_split"
    vals = sorted(df[sample_col].dropna().astype(str).unique())
    splits = []
    for test_sample in vals:
        train_mask = ~df[sample_col].astype(str).eq(test_sample)
        test_mask = df[sample_col].astype(str).eq(test_sample)
        name = f"leave_one_sample__test_{test_sample}"
        splits.append((name, test_sample, train_mask, test_mask))
    return splits


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def plot_summary(all_metrics, outdir):
    if all_metrics.empty:
        return None

    outbase = outdir / "Fig_Step62_Generalization_Summary"

    df = all_metrics.copy()
    df = df[df["status"].eq("ok")].copy()
    if df.empty:
        return None

    # Use a compact 2x2 dashboard.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    axes = axes.ravel()

    # 1 region macro F1
    axes[0].barh(df["split"], df["region_macro_f1"])
    axes[0].set_xlabel("Region macro-F1")
    axes[0].set_title("Held-out region classification")
    axes[0].grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)

    # 2 repair RMSE
    axes[1].barh(df["split"], df["repair_rmse"])
    axes[1].set_xlabel("Repair RMSE")
    axes[1].set_title("Repair-score regression")
    axes[1].grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)

    # 3 neighbor RMSE
    axes[2].barh(df["split"], df["neighbor_rmse"])
    axes[2].set_xlabel("Neighbor RMSE")
    axes[2].set_title("Neighborhood composition")
    axes[2].grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)

    # 4 ranking stability
    axes[3].barh(df["split"], df["perturbation_score_spearman"])
    axes[3].set_xlabel("Train/test perturbation score Spearman")
    axes[3].set_title("Perturbation ranking stability")
    axes[3].grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=7)

    fig.suptitle("Step 62 | Held-out timepoint / sample generalization", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=500, bbox_inches="tight")
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
    ap.add_argument("--in61", default=str(DEFAULT_IN61_STRICT))
    ap.add_argument("--neighbor53c", default=str(DEFAULT_NEIGHBOR53C))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--feature_mode", default="full", choices=["full", "modules_only", "module_plus_coords", "coords_only"])
    ap.add_argument("--include_region_probability_features", action="store_true")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=12)
    ap.add_argument("--min_samples_leaf", type=int, default=3)
    ap.add_argument("--n_jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_train_cells", type=int, default=200)
    ap.add_argument("--min_test_cells", type=int, default=50)
    args = ap.parse_args()

    in61 = Path(args.in61)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 62: held-out timepoint / sample generalization")
    log("=" * 100)
    log(f"in61={in61}")
    log(f"neighbor53c={args.neighbor53c}")
    log(f"outdir={outdir}")

    module, cell, ranking = load_strict_tables(in61)
    module_merged, merge_meta = merge_neighbor53c(module, args.neighbor53c)

    df = module_merged.copy()
    df["obs_name"] = df["obs_name"].astype(str)

    df["timepoint"] = detect_timepoint(df)
    if df["timepoint"].isna().all():
        raise RuntimeError("Cannot detect timepoint from table.")

    df["timepoint"] = df["timepoint"].astype(str)

    sample_col = detect_sample_col(df)
    if sample_col is not None:
        df["sample_id_for_split"] = df[sample_col].astype(str)
        sample_mode = f"sample_col:{sample_col}"
    else:
        df["sample_id_for_split"] = df["timepoint"].astype(str)
        sample_mode = "fallback_timepoint_as_sample"

    if "state_group" not in df.columns:
        raise RuntimeError("Input strict module table lacks state_group.")
    df["state_group"] = df["state_group"].map(normalize_state)

    feature_cols, repair_col, neighbor_cols, celltype_cols, ft_meta = detect_targets_and_features(
        df,
        include_region_prob_features=args.include_region_probability_features,
        feature_mode=args.feature_mode,
    )

    if len(feature_cols) == 0:
        raise RuntimeError("No valid feature columns detected.")

    # Add timepoint to cell-level table for ranking subsets.
    cell = cell.copy()
    cell["obs_name"] = cell["obs_name"].astype(str)
    if "timepoint" not in cell.columns:
        cell["timepoint"] = detect_timepoint(cell).astype(str)

    # Save modeling input.
    df.to_csv(outdir / "step62_modeling_input_table.csv", index=False)

    feature_meta = {
        "input_in61": str(in61),
        "merge_meta": merge_meta,
        "sample_mode": sample_mode,
        "timepoint_counts": df["timepoint"].value_counts(dropna=False).to_dict(),
        "state_counts": df["state_group"].value_counts(dropna=False).to_dict(),
        "feature_target_meta": ft_meta,
        "model": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "feature_mode": args.feature_mode,
            "include_region_probability_features": args.include_region_probability_features,
        }
    }

    (outdir / "step62_feature_target_metadata.json").write_text(
        json.dumps(feature_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    all_rows = []
    prediction_rows = []
    ranking_rows = []
    trajectory_rows = []

    # Repair monotonicity
    repair_mono_df, repair_mono_meta = compute_repair_monotonicity(df, repair_col)
    repair_mono_df.to_csv(outdir / "step62_repair_monotonicity_by_timepoint.csv", index=False)

    # Held-out timepoint splits
    timepoint_rows = []
    for split_name, train_tps, test_tp, train_mask, test_mask in build_timepoint_splits(df):
        log(f"[timepoint split] {split_name}")

        row, pred, ranks, _ = run_split(
            df, cell, feature_cols, repair_col, neighbor_cols, celltype_cols,
            split_name, train_mask, test_mask, args
        )
        row["split_type"] = "heldout_timepoint"
        row["train_timepoints"] = "+".join(train_tps)
        row["test_timepoint"] = test_tp
        row.update(repair_mono_meta)

        traj = compute_trajectory_consistency(df, split_name, train_tps, test_tp)
        row.update({k: v for k, v in traj.items() if k not in ["split"]})
        trajectory_rows.append(traj)

        timepoint_rows.append(row)
        all_rows.append(row)

        if not pred.empty:
            prediction_rows.append(pred)
        if not ranks.empty:
            ranking_rows.append(ranks)

    heldout_tp = pd.DataFrame(timepoint_rows)
    heldout_tp.to_csv(outdir / "step62_heldout_timepoint_metrics.csv", index=False)

    # Leave-one-sample-out splits
    loso_rows = []
    for split_name, test_sample, train_mask, test_mask in build_loso_splits(df):
        log(f"[LOSO split] {split_name}")

        row, pred, ranks, _ = run_split(
            df, cell, feature_cols, repair_col, neighbor_cols, celltype_cols,
            split_name, train_mask, test_mask, args
        )
        row["split_type"] = "leave_one_sample"
        row["test_sample"] = test_sample
        row["sample_mode"] = sample_mode
        row.update(repair_mono_meta)

        loso_rows.append(row)
        all_rows.append(row)

        if not pred.empty:
            prediction_rows.append(pred)
        if not ranks.empty:
            ranking_rows.append(ranks)

    loso = pd.DataFrame(loso_rows)
    loso.to_csv(outdir / "step62_leave_one_sample_metrics.csv", index=False)

    all_metrics = pd.DataFrame(all_rows)
    all_metrics.to_csv(outdir / "step62_all_split_metrics.csv", index=False)

    if ranking_rows:
        all_rankings = pd.concat(ranking_rows, ignore_index=True)
    else:
        all_rankings = pd.DataFrame()
    all_rankings.to_csv(outdir / "step62_perturbation_ranking_stability.csv", index=False)

    if prediction_rows:
        preds = pd.concat(prediction_rows, ignore_index=True)
        # keep file manageable
        if len(preds) > 50000:
            preds = preds.sample(n=50000, random_state=args.seed)
    else:
        preds = pd.DataFrame()
    preds.to_csv(outdir / "step62_prediction_examples.csv", index=False)

    traj_df = pd.DataFrame(trajectory_rows)
    traj_df.to_csv(outdir / "step62_trajectory_consistency.csv", index=False)

    # Summary
    ok = all_metrics[all_metrics["status"].eq("ok")].copy()
    summary_rows = []
    for split_type, sub in ok.groupby("split_type"):
        row = {
            "split_type": split_type,
            "n_splits": int(len(sub)),
            "region_macro_f1_mean": float(sub["region_macro_f1"].mean()) if "region_macro_f1" in sub else np.nan,
            "region_balanced_accuracy_mean": float(sub["region_balanced_accuracy"].mean()) if "region_balanced_accuracy" in sub else np.nan,
            "repair_rmse_mean": float(sub["repair_rmse"].mean()) if "repair_rmse" in sub else np.nan,
            "neighbor_rmse_mean": float(sub["neighbor_rmse"].mean()) if "neighbor_rmse" in sub else np.nan,
            "celltype_rmse_mean": float(sub["celltype_rmse"].mean()) if "celltype_rmse" in sub else np.nan,
            "trajectory_consistency_score_mean": float(sub["trajectory_consistency_score"].mean()) if "trajectory_consistency_score" in sub else np.nan,
            "perturbation_score_spearman_mean": float(sub["perturbation_score_spearman"].mean()) if "perturbation_score_spearman" in sub else np.nan,
            "perturbation_top5_jaccard_mean": float(sub["perturbation_top5_jaccard"].mean()) if "perturbation_top5_jaccard" in sub else np.nan,
            "repair_monotonicity_spearman": repair_mono_meta.get("repair_monotonicity_spearman", np.nan),
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(outdir / "step62_generalization_summary.csv", index=False)

    plot_outputs = plot_summary(all_metrics, outdir)

    report = {
        "status": "ok",
        "outdir": str(outdir),
        "input_in61": str(in61),
        "n_cells": int(len(df)),
        "n_features": int(len(feature_cols)),
        "timepoint_counts": df["timepoint"].value_counts(dropna=False).to_dict(),
        "state_counts": df["state_group"].value_counts(dropna=False).to_dict(),
        "sample_mode": sample_mode,
        "n_neighbor_targets": len(neighbor_cols),
        "n_celltype_targets": len(celltype_cols),
        "repair_col": repair_col,
        "repair_monotonicity": repair_mono_meta,
        "n_timepoint_splits": int(len(heldout_tp)),
        "n_loso_splits": int(len(loso)),
        "plot_outputs": plot_outputs,
        "outputs": {
            "modeling_input": str(outdir / "step62_modeling_input_table.csv"),
            "feature_target_metadata": str(outdir / "step62_feature_target_metadata.json"),
            "heldout_timepoint_metrics": str(outdir / "step62_heldout_timepoint_metrics.csv"),
            "leave_one_sample_metrics": str(outdir / "step62_leave_one_sample_metrics.csv"),
            "all_split_metrics": str(outdir / "step62_all_split_metrics.csv"),
            "perturbation_ranking_stability": str(outdir / "step62_perturbation_ranking_stability.csv"),
            "prediction_examples": str(outdir / "step62_prediction_examples.csv"),
            "repair_monotonicity_by_timepoint": str(outdir / "step62_repair_monotonicity_by_timepoint.csv"),
            "trajectory_consistency": str(outdir / "step62_trajectory_consistency.csv"),
            "summary": str(outdir / "step62_generalization_summary.csv"),
        }
    }

    (outdir / "step62_generalization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    lines = []
    lines.append("Step 62 held-out timepoint / sample generalization report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Generalization summary:")
    lines.append(summary.to_string(index=False))
    lines.append("")
    lines.append("Held-out timepoint metrics:")
    lines.append(heldout_tp.to_string(index=False))
    lines.append("")
    lines.append("Leave-one-sample metrics:")
    lines.append(loso.to_string(index=False))
    lines.append("")
    lines.append("Repair monotonicity:")
    lines.append(repair_mono_df.to_string(index=False))
    lines.append("")
    lines.append("Trajectory consistency:")
    lines.append(traj_df.to_string(index=False))

    (outdir / "step62_generalization_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step 62")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Summary:")
    log(summary.to_string(index=False))
    log("")
    log("Held-out timepoint metrics:")
    log(heldout_tp.to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
