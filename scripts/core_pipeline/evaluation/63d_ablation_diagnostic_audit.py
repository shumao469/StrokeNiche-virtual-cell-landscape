#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
63d_ablation_diagnostic_audit.py

Purpose
-------
Step 63d: Diagnostic audit for mechanism-aware ablation interpretation.

This script does NOT retrain main models. It audits two uncertain ablation blocks:

1) Step7 no_neighbor_loss
   Question:
     Did no_neighbor_loss truly fail to show a neighbor-specific mechanism,
     or was the 63c leakage-free output-state readout too weak?

   Audits:
     A. training loss/history availability
     B. full vs no_neighbor_loss output-state difference
     C. association between output-state features and clean neighbor targets
     D. leakage-free post-hoc neighbor predictability from output-state features

2) L_domain
   Question:
     Does high domain leakage reflect undesirable batch/timepoint leakage,
     or biologically meaningful D1/D3/D7 temporal structure?

   Audits:
     A. timepoint predictability from biology/state/module/coordinate features
     B. comparison with previously computed L_domain variant leakage
     C. interpretation note: high D1/D3/D7 leakage may reflect true stroke evolution

Outputs
-------
results/step8_strokeniche_perturbmap/mechanism_ablation_63d_diagnostic_audit_<tag>/
  step63d_step7_training_loss_audit.csv
  step63d_no_neighbor_output_difference.csv
  step63d_no_neighbor_target_association.csv
  step63d_no_neighbor_readout_summary.csv
  step63d_ldomain_biology_vs_leakage_audit.csv
  step63d_ldomain_variant_leakage_imported.csv
  step63d_interpretation_summary.csv
  step63d_report.json
  step63d_report.txt
  Fig_Step63D_AblationDiagnosticAudit.pdf/svg/png
"""

from pathlib import Path
import argparse
import json
import re
from datetime import datetime
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, balanced_accuracy_score, accuracy_score


BASE = Path("/mnt/h/vir/ST")


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def read_csv(path, required=False):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception as e:
        if required:
            raise e
        return None


def read_json(path, required=False):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def rmse(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(np.sqrt(mean_squared_error(a[ok], b[ok])))


def mae(y_true, y_pred):
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    return float(mean_absolute_error(a[ok], b[ok]))


def spearman(a, b):
    x = pd.Series(a).astype(float)
    y = pd.Series(b).astype(float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return float(x[ok].corr(y[ok], method="spearman"))


def skill_vs_baseline(y_true, y_pred, y_base):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    yb = np.asarray(y_base, dtype=float)
    ok = np.isfinite(yt) & np.isfinite(yp) & np.isfinite(yb)
    if ok.sum() < 3:
        return np.nan
    mse_model = mean_squared_error(yt[ok], yp[ok])
    mse_base = mean_squared_error(yt[ok], yb[ok])
    if mse_base <= 1e-12:
        return np.nan
    return float(1.0 - mse_model / mse_base)


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

    return ""


def detect_timepoint(df):
    out = pd.Series("", index=df.index, dtype=object)

    for c in ["timepoint", "time_point", "time", "sample", "sample_id", "RCTD_sample", "batch", "orig.ident", "orig_ident"]:
        if c in df.columns:
            v = df[c].map(extract_timepoint)
            out = out.where(out.astype(str).ne(""), v)

    if "obs_name" in df.columns:
        v = df["obs_name"].map(extract_timepoint)
        out = out.where(out.astype(str).ne(""), v)

    return out.replace("", np.nan)


def is_numeric_like(s, frac=0.80):
    x = pd.to_numeric(s, errors="coerce")
    return bool(x.notna().mean() >= frac)


def numeric_cols(df, min_fraction=0.80):
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c], min_fraction):
            cols.append(c)
    return cols


# -----------------------------------------------------------------------------
# Input locating
# -----------------------------------------------------------------------------

def find_step7_prediction(step7_root, variant):
    d = Path(step7_root) / variant
    candidates = []

    if d.exists():
        candidates.extend(list(d.rglob("strokeniche_adapter_predictions.csv")))
        candidates.extend(list(d.rglob("*adapter_predictions*.csv")))
        candidates.extend(list(d.rglob("*predictions*.csv")))

    candidates = [p for p in candidates if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        return None

    return sorted(
        candidates,
        key=lambda p: (0 if p.name == "strokeniche_adapter_predictions.csv" else 1, len(str(p)))
    )[0]


def find_training_history_files(step7_root, variant):
    d = Path(step7_root) / variant
    if not d.exists():
        return []

    patterns = [
        "*history*.csv",
        "*loss*.csv",
        "*metrics*.csv",
        "*training*.csv",
        "*history*.json",
        "*metrics*.json",
        "*metadata*.json",
    ]

    files = []
    for pat in patterns:
        files.extend(list(d.rglob(pat)))

    files = [p for p in files if p.is_file() and p.stat().st_size > 0]
    return sorted(set(files), key=lambda p: str(p))


# -----------------------------------------------------------------------------
# Clean target and feature definitions
# -----------------------------------------------------------------------------

def clean_neighbor_targets(df):
    cols = []
    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c])):
            continue

        if (
            c.startswith("neighbor_mean_feature_RCTD_")
            or c.startswith("neighbor_delta_vs_self_feature_RCTD_")
            or c.startswith("neighbor_mean_feature_celltype_")
            or c.startswith("neighbor_delta_vs_self_feature_celltype_")
        ):
            nc = norm_col(c)
            bad = [
                "spatial", "coord", "umap", "weight", "total", "sum", "max",
                "available", "dominant", "entropy", "boundary", "density", "distance"
            ]
            if not any(b in nc for b in bad):
                cols.append(c)

    return list(dict.fromkeys(cols))


def clean_celltype_targets(df):
    cols = []
    for c in df.columns:
        if c.startswith("RCTD_prop_"):
            if pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c]):
                cols.append(c)
    return list(dict.fromkeys(cols))


def output_state_features(df, target_cols=None):
    target_cols = set(target_cols or [])

    preferred = [
        "repair_score",
        "prob_lesion_core",
        "prob_peri_infarct",
        "prob_remote_like",
        "core_probability",
        "peri_probability",
        "remote_probability",
        "delta_repair_score",
        "delta_core_probability",
        "delta_peri_probability",
        "delta_remote_probability",
        "rescue_score",
        "delta_norm",
    ]

    keep = []
    audit = []

    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c])):
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() < 0.80:
            audit.append({"column": c, "kept": False, "reason": "too_many_missing"})
            continue
        if x.nunique(dropna=True) <= 1:
            audit.append({"column": c, "kept": False, "reason": "constant"})
            continue

        nc = norm_col(c)

        forbidden = False
        reason = ""

        if c in target_cols:
            forbidden = True
            reason = "exact_target_column"
        elif c.startswith("neighbor_mean_feature_") or c.startswith("neighbor_delta_vs_self_feature_"):
            forbidden = True
            reason = "neighbor_target_prefix"
        elif c.startswith("RCTD_") or c.startswith("RCTD_prop_"):
            forbidden = True
            reason = "RCTD_target_or_composition_prefix"
        elif any(tok in nc for tok in [
            "obs", "barcode", "key", "timepoint", "sample", "train", "val", "test",
            "true", "label", "class", "region_true", "region_pred", "state_group",
            "predicted_celltype", "dominant_celltype", "cell_index", "target_gene",
            "ligand", "receptor", "neighbor", "rctd", "celltype", "composition",
            "rank"
        ]):
            forbidden = True
            reason = "forbidden_label_or_target_token"

        if forbidden:
            audit.append({"column": c, "kept": False, "reason": reason})
            continue

        if c in preferred or c.startswith("prob_") or c.startswith("delta_") or "repair" in nc or "rescue" in nc:
            keep.append(c)
            audit.append({"column": c, "kept": True, "reason": "allowed_output_state"})
        else:
            audit.append({"column": c, "kept": False, "reason": "not_output_state_feature"})

    return list(dict.fromkeys(keep)), pd.DataFrame(audit)


def module_feature_cols(df):
    cols = []
    for c in numeric_cols(df):
        if not str(c).startswith("module_"):
            continue
        nc = norm_col(c)
        if any(k in nc for k in ["obs", "barcode", "timepoint", "coord", "key", "matched", "source", "method"]):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() >= 0.80 and x.nunique(dropna=True) > 1:
            cols.append(c)
    return list(dict.fromkeys(cols))


def coord_feature_cols(df):
    cols = []
    for c in ["strict_coord_x", "strict_coord_y", "coord_x", "coord_y", "umap_1", "umap_2"]:
        if c in df.columns and (pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c])):
            cols.append(c)
    return list(dict.fromkeys(cols))


def context_feature_cols(df):
    candidates = [
        "endothelial_context",
        "astrocyte_context",
        "microglia_context",
        "neuron_context",
        "lesion_niche_context",
        "remote_offtarget_context",
        "repair_score",
        "core_probability",
        "peri_probability",
        "remote_probability",
        "prob_lesion_core",
        "prob_peri_infarct",
        "prob_remote_like",
    ]
    cols = []
    for c in candidates:
        if c in df.columns and (pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c])):
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() >= 0.80 and x.nunique(dropna=True) > 1:
                cols.append(c)
    return list(dict.fromkeys(cols))


def composition_feature_cols(df):
    cols = []
    for c in clean_neighbor_targets(df) + clean_celltype_targets(df):
        if c in df.columns:
            cols.append(c)
    return list(dict.fromkeys(cols))


# -----------------------------------------------------------------------------
# Step7 training loss audit
# -----------------------------------------------------------------------------

LOSS_TOKENS = [
    "loss",
    "neighbor",
    "repair",
    "contrast",
    "contrastive",
    "graph",
    "spatial",
    "domain",
    "coral",
    "adv",
]


def audit_history_csv(path, variant):
    df = read_csv(path)
    rows = []

    if df is None or df.empty:
        return rows

    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c])):
            continue

        nc = norm_col(c)
        if not any(tok in nc for tok in LOSS_TOKENS):
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() == 0:
            continue

        rows.append({
            "variant": variant,
            "file": str(path),
            "metric_col": c,
            "n_values": int(x.notna().sum()),
            "first": float(x.dropna().iloc[0]),
            "last": float(x.dropna().iloc[-1]),
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "contains_neighbor": "neighbor" in nc,
            "contains_repair": "repair" in nc,
            "contains_contrastive": "contrast" in nc,
            "contains_graph_or_spatial": ("graph" in nc or "spatial" in nc),
            "contains_domain": ("domain" in nc or "coral" in nc or "adv" in nc),
        })

    return rows


def flatten_json(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_json(v, key))
        else:
            out[key] = v
    return out


def audit_history_json(path, variant):
    js = read_json(path)
    rows = []
    if not js:
        return rows

    flat = flatten_json(js)

    for k, v in flat.items():
        nc = norm_col(k)
        if not any(tok in nc for tok in LOSS_TOKENS):
            continue
        val = safe_float(v)
        if pd.isna(val):
            continue
        rows.append({
            "variant": variant,
            "file": str(path),
            "metric_col": k,
            "n_values": 1,
            "first": val,
            "last": val,
            "min": val,
            "max": val,
            "mean": val,
            "contains_neighbor": "neighbor" in nc,
            "contains_repair": "repair" in nc,
            "contains_contrastive": "contrast" in nc,
            "contains_graph_or_spatial": ("graph" in nc or "spatial" in nc),
            "contains_domain": ("domain" in nc or "coral" in nc or "adv" in nc),
        })

    return rows


def run_step7_training_loss_audit(step7_root):
    rows = []
    for variant in ["full", "no_neighbor_loss", "no_repair_loss", "no_contrastive_loss", "no_spatial_graph"]:
        files = find_training_history_files(step7_root, variant)

        if not files:
            rows.append({
                "variant": variant,
                "file": "",
                "metric_col": "",
                "n_values": 0,
                "status": "no_history_or_metric_files_found",
            })
            continue

        any_rows = False
        for p in files:
            if p.suffix.lower() == ".csv":
                sub = audit_history_csv(p, variant)
            elif p.suffix.lower() == ".json":
                sub = audit_history_json(p, variant)
            else:
                sub = []

            for r in sub:
                r["status"] = "ok"
            rows.extend(sub)
            any_rows = any_rows or bool(sub)

        if not any_rows:
            rows.append({
                "variant": variant,
                "file": ";".join(map(str, files[:10])),
                "metric_col": "",
                "n_values": 0,
                "status": "files_found_but_no_loss_like_numeric_metrics",
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# no_neighbor_loss output difference audit
# -----------------------------------------------------------------------------

def compare_outputs(full_path, no_neighbor_path, target_df, target_cols):
    full = read_csv(full_path, required=True)
    nn = read_csv(no_neighbor_path, required=True)

    if "obs_name" not in full.columns or "obs_name" not in nn.columns:
        return pd.DataFrame(), pd.DataFrame()

    full["obs_name"] = full["obs_name"].astype(str)
    nn["obs_name"] = nn["obs_name"].astype(str)

    # Drop target columns from both predictions before comparing output-state.
    full = full.drop(columns=[c for c in target_cols if c in full.columns], errors="ignore")
    nn = nn.drop(columns=[c for c in target_cols if c in nn.columns], errors="ignore")

    merged = full.merge(nn, on="obs_name", suffixes=("_full", "_no_neighbor"), how="inner")

    # Candidate comparable columns: same original names present in both.
    base_cols = []
    for c in full.columns:
        if c == "obs_name":
            continue
        if c in nn.columns:
            if pd.api.types.is_numeric_dtype(full[c]) or is_numeric_like(full[c]):
                if pd.api.types.is_numeric_dtype(nn[c]) or is_numeric_like(nn[c]):
                    base_cols.append(c)

    # Focus on output-state columns.
    keep_cols, faudit = output_state_features(full[["obs_name"] + base_cols].copy(), target_cols=[])
    base_cols = [c for c in base_cols if c in keep_cols]

    rows = []
    for c in base_cols:
        cf = f"{c}_full"
        cn = f"{c}_no_neighbor"
        if cf not in merged.columns or cn not in merged.columns:
            continue

        x = pd.to_numeric(merged[cf], errors="coerce")
        y = pd.to_numeric(merged[cn], errors="coerce")
        ok = x.notna() & y.notna()

        if ok.sum() < 3:
            continue

        diff = y[ok] - x[ok]

        rows.append({
            "output_feature": c,
            "n": int(ok.sum()),
            "full_mean": float(x[ok].mean()),
            "no_neighbor_mean": float(y[ok].mean()),
            "mean_diff_no_neighbor_minus_full": float(diff.mean()),
            "median_abs_diff": float(diff.abs().median()),
            "mean_abs_diff": float(diff.abs().mean()),
            "rmse_between_variants": rmse(x[ok], y[ok]),
            "spearman_between_variants": spearman(x[ok], y[ok]),
            "pearson_between_variants": float(x[ok].corr(y[ok], method="pearson")),
            "fraction_nearly_identical_1e_6": float((diff.abs() < 1e-6).mean()),
            "full_sd": float(x[ok].std()),
            "no_neighbor_sd": float(y[ok].std()),
        })

    summary = pd.DataFrame({
        "n_merged_rows": [int(len(merged))],
        "n_output_features_compared": [int(len(rows))],
        "mean_abs_diff_over_features": [float(np.nanmean([r["mean_abs_diff"] for r in rows])) if rows else np.nan],
        "median_abs_diff_over_features": [float(np.nanmedian([r["median_abs_diff"] for r in rows])) if rows else np.nan],
        "mean_spearman_between_variants": [float(np.nanmean([r["spearman_between_variants"] for r in rows])) if rows else np.nan],
    })

    return pd.DataFrame(rows), summary


# -----------------------------------------------------------------------------
# no_neighbor_loss target association audit
# -----------------------------------------------------------------------------

def make_fixed_split(df, seed=42):
    tmp = df[["obs_name"]].copy()
    tp = detect_timepoint(df)
    if tp.notna().mean() >= 0.5:
        tmp["strata"] = tp.fillna("unknown").astype(str)
    else:
        tmp["strata"] = "all"

    y = tmp["strata"].astype(str)

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr, te = next(splitter.split(tmp, y))
    except Exception:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(tmp))
        rng.shuffle(idx)
        nte = int(round(0.2 * len(idx)))
        te = idx[:nte]
        tr = idx[nte:]

    return set(tmp.iloc[tr]["obs_name"].astype(str)), set(tmp.iloc[te]["obs_name"].astype(str))


def ridge_neighbor_readout(pred_path, target_df, target_cols, train_obs, test_obs, seed=42):
    pred = read_csv(pred_path, required=True)
    if "obs_name" not in pred.columns:
        return None, pd.DataFrame(), pd.DataFrame(), "prediction_lacks_obs_name"

    pred["obs_name"] = pred["obs_name"].astype(str)
    pred = pred.drop(columns=[c for c in target_cols if c in pred.columns], errors="ignore")

    merged = pred.merge(target_df[["obs_name"] + target_cols], on="obs_name", how="inner")
    if merged.empty:
        return None, pd.DataFrame(), pd.DataFrame(), "no_overlap"

    fcols, faudit = output_state_features(merged, target_cols=target_cols)
    if not fcols:
        return None, pd.DataFrame(), faudit, "no_output_state_features"

    train_mask = merged["obs_name"].isin(train_obs)
    test_mask = merged["obs_name"].isin(test_obs)

    keep_targets = []
    target_rows = []
    for c in target_cols:
        ytr = pd.to_numeric(merged.loc[train_mask, c], errors="coerce")
        yte = pd.to_numeric(merged.loc[test_mask, c], errors="coerce")
        ok = ytr.notna().sum() >= 50 and yte.notna().sum() >= 10 and ytr.nunique(dropna=True) > 1
        target_rows.append({
            "target": c,
            "kept": bool(ok),
            "train_nonnull": int(ytr.notna().sum()),
            "test_nonnull": int(yte.notna().sum()),
            "train_nunique": int(ytr.nunique(dropna=True)),
        })
        if ok:
            keep_targets.append(c)

    if not keep_targets:
        return None, pd.DataFrame(target_rows), faudit, "no_valid_targets"

    X_train = merged.loc[train_mask, fcols].apply(pd.to_numeric, errors="coerce")
    X_test = merged.loc[test_mask, fcols].apply(pd.to_numeric, errors="coerce")
    Y_train = merged.loc[train_mask, keep_targets].apply(pd.to_numeric, errors="coerce")
    Y_test = merged.loc[test_mask, keep_targets].apply(pd.to_numeric, errors="coerce")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])

    model.fit(X_train, Y_train)
    pred_y = model.predict(X_test)
    base_y = np.tile(Y_train.mean(axis=0).values.reshape(1, -1), (Y_test.shape[0], 1))

    rows = []
    for j, c in enumerate(keep_targets):
        yt = Y_test.values[:, j]
        yp = pred_y[:, j]
        yb = base_y[:, j]
        rows.append({
            "target": c,
            "rmse": rmse(yt, yp),
            "mae": mae(yt, yp),
            "spearman": spearman(yt, yp),
            "skill_vs_train_mean": skill_vs_baseline(yt, yp, yb),
            "target_mean": float(np.nanmean(yt)),
            "target_sd": float(np.nanstd(yt)),
            "target_range": float(np.nanmax(yt) - np.nanmin(yt)) if np.isfinite(yt).any() else np.nan,
        })

    per = pd.DataFrame(rows)
    summary = {
        "n_merged_rows": int(len(merged)),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_features": int(len(fcols)),
        "n_targets": int(len(keep_targets)),
        "feature_cols_used": ";".join(fcols),
        "mean_rmse": float(per["rmse"].mean()),
        "median_rmse": float(per["rmse"].median()),
        "mean_spearman": float(per["spearman"].mean()),
        "mean_skill_vs_train_mean": float(per["skill_vs_train_mean"].mean()),
    }

    return summary, per, faudit, "ok"


def output_target_correlation(pred_path, target_df, target_cols):
    pred = read_csv(pred_path, required=True)
    if "obs_name" not in pred.columns:
        return pd.DataFrame()

    pred["obs_name"] = pred["obs_name"].astype(str)
    pred = pred.drop(columns=[c for c in target_cols if c in pred.columns], errors="ignore")

    merged = pred.merge(target_df[["obs_name"] + target_cols], on="obs_name", how="inner")
    fcols, _ = output_state_features(merged, target_cols=target_cols)

    rows = []
    for f in fcols:
        for t in target_cols:
            if f not in merged.columns or t not in merged.columns:
                continue
            r = spearman(merged[f], merged[t])
            rows.append({
                "output_feature": f,
                "target": t,
                "spearman": r,
                "abs_spearman": abs(r) if not pd.isna(r) else np.nan,
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# L_domain biology-vs-leakage audit
# -----------------------------------------------------------------------------

def domain_cv(X, y, seed=42, max_rows=50000):
    y = pd.Series(y).astype(str)
    X = X.copy()

    ok = y.notna()
    y = y[ok]
    X = X.loc[ok]

    if len(X) > max_rows:
        idx = X.sample(n=max_rows, random_state=seed).index
        X = X.loc[idx]
        y = y.loc[idx]

    if y.nunique() < 2:
        return None, "less_than_two_domains"

    counts = y.value_counts()
    min_class = int(counts.min())
    if min_class < 10:
        return None, "too_few_rows_per_domain"

    n_splits = min(5, min_class)
    if n_splits < 2:
        return None, "not_enough_folds"

    clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    bal = []
    acc = []

    for tr, te in skf.split(X, y):
        clf.fit(X.iloc[tr], y.iloc[tr])
        yp = clf.predict(X.iloc[te])
        bal.append(balanced_accuracy_score(y.iloc[te], yp))
        acc.append(accuracy_score(y.iloc[te], yp))

    return {
        "domain_balanced_accuracy": float(np.mean(bal)),
        "domain_accuracy": float(np.mean(acc)),
        "domain_chance": float(1.0 / y.nunique()),
        "domain_leakage_minus_chance": float(np.mean(bal) - 1.0 / y.nunique()),
        "domain_confusion_resistance": float(1.0 - np.mean(bal)),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_domains": int(y.nunique()),
        "domain_counts": ";".join([f"{k}:{v}" for k, v in counts.items()]),
    }, "ok"


def run_ldomain_biology_audit(modeling_df, seed=42):
    df = modeling_df.copy()
    if "timepoint" not in df.columns or df["timepoint"].isna().all():
        df["timepoint"] = detect_timepoint(df)
    df = df[df["timepoint"].notna()].copy()
    y = df["timepoint"].astype(str)

    feature_groups = {}

    cols = context_feature_cols(df)
    if cols:
        feature_groups["output_state_context"] = df[cols].apply(pd.to_numeric, errors="coerce")

    cols = module_feature_cols(df)
    if cols:
        feature_groups["niche_module_scores"] = df[cols].apply(pd.to_numeric, errors="coerce")

    cols = coord_feature_cols(df)
    if cols:
        feature_groups["latent_spatial_coordinates"] = df[cols].apply(pd.to_numeric, errors="coerce")

    cols = composition_feature_cols(df)
    if cols:
        feature_groups["celltype_neighbor_composition"] = df[cols].apply(pd.to_numeric, errors="coerce")

    if "state_group" in df.columns:
        X = pd.get_dummies(df["state_group"].astype(str), prefix="state")
        if X.shape[1] > 1:
            feature_groups["state_label_onehot_audit"] = X

    if "region_true" in df.columns:
        X = pd.get_dummies(df["region_true"].astype(str), prefix="region")
        if X.shape[1] > 1:
            feature_groups["region_label_onehot_audit"] = X

    combined_cols = []
    for fun in [context_feature_cols, module_feature_cols, coord_feature_cols]:
        combined_cols.extend(fun(df))
    combined_cols = list(dict.fromkeys(combined_cols))
    if combined_cols:
        feature_groups["combined_state_module_coord"] = df[combined_cols].apply(pd.to_numeric, errors="coerce")

    rows = []
    for name, X in feature_groups.items():
        if X is None or X.empty or X.shape[1] == 0:
            continue
        metrics, status = domain_cv(X, y, seed=seed)
        row = {
            "feature_group": name,
            "status": status,
        }
        if metrics:
            row.update(metrics)
        row["feature_cols_head"] = ";".join(list(X.columns[:80]))
        rows.append(row)

    return pd.DataFrame(rows)


def import_ldomain_variant_leakage(step63b_dir):
    p = Path(step63b_dir) / "step63b_ldomain_domain_leakage_metrics.csv"
    df = read_csv(p, required=False)
    if df is None:
        return pd.DataFrame()
    keep = [c for c in [
        "component", "variant", "domain_leakage", "domain_balanced_accuracy",
        "domain_leakage_minus_chance", "domain_confusion_resistance",
        "domain_chance", "n_domains", "domain_counts", "table_path", "status"
    ] if c in df.columns]
    return df[keep].copy()


# -----------------------------------------------------------------------------
# Interpretation and plotting
# -----------------------------------------------------------------------------

def make_interpretation(training_audit, output_diff_summary, readout_summary, ldomain_bio, ldomain_variant):
    rows = []

    # no_neighbor_loss: training availability
    has_neighbor_metric_full = False
    has_neighbor_metric_nn = False
    if not training_audit.empty and "contains_neighbor" in training_audit.columns:
        has_neighbor_metric_full = bool(training_audit[(training_audit["variant"] == "full") & (training_audit["contains_neighbor"] == True)].shape[0] > 0)
        has_neighbor_metric_nn = bool(training_audit[(training_audit["variant"] == "no_neighbor_loss") & (training_audit["contains_neighbor"] == True)].shape[0] > 0)

    rows.append({
        "audit_block": "no_neighbor_loss_training_history",
        "finding": f"neighbor-like loss metric found in full={has_neighbor_metric_full}, no_neighbor_loss={has_neighbor_metric_nn}",
        "interpretation": (
            "training logs contain explicit neighbor loss readouts"
            if has_neighbor_metric_full or has_neighbor_metric_nn
            else
            "existing history files do not expose explicit neighbor loss readouts; cannot verify loss behavior from logs alone"
        ),
        "recommended_use": "audit"
    })

    # output difference
    if not output_diff_summary.empty:
        mad = safe_float(output_diff_summary.iloc[0].get("mean_abs_diff_over_features", np.nan))
        sp = safe_float(output_diff_summary.iloc[0].get("mean_spearman_between_variants", np.nan))
        if not pd.isna(mad) and mad < 0.02 and not pd.isna(sp) and sp > 0.95:
            interp = "full and no_neighbor_loss output-state predictions are highly similar"
            use = "supports downgrading no_neighbor_loss as weak mechanism evidence"
        else:
            interp = "full and no_neighbor_loss output-state predictions show measurable differences"
            use = "inspect feature-level differences"
        rows.append({
            "audit_block": "no_neighbor_output_difference",
            "finding": f"mean_abs_diff={mad}, mean_spearman={sp}",
            "interpretation": interp,
            "recommended_use": use,
        })

    # leakage-free readout
    if not readout_summary.empty:
        if set(readout_summary["variant"]) >= {"full", "no_neighbor_loss"}:
            rr = readout_summary.set_index("variant")
            b = safe_float(rr.loc["full", "mean_rmse"])
            a = safe_float(rr.loc["no_neighbor_loss", "mean_rmse"])
            delta = a - b
            if delta > 0.001:
                interp = "no_neighbor_loss worsens leakage-free neighbor readout"
                use = "can be used as directional mechanism evidence"
            elif delta < -0.001:
                interp = "no_neighbor_loss does not worsen leakage-free neighbor readout"
                use = "negative audit result; do not use as strong main-text evidence"
            else:
                interp = "no_neighbor_loss and full are nearly indistinguishable on leakage-free neighbor readout"
                use = "weak/inconclusive mechanism evidence"
            rows.append({
                "audit_block": "no_neighbor_leakage_free_readout",
                "finding": f"full_rmse={b}, no_neighbor_rmse={a}, delta={delta}",
                "interpretation": interp,
                "recommended_use": use,
            })

    # L_domain biology
    if not ldomain_bio.empty and "domain_balanced_accuracy" in ldomain_bio.columns:
        best = ldomain_bio.sort_values("domain_balanced_accuracy", ascending=False).head(1).iloc[0]
        ba = safe_float(best.get("domain_balanced_accuracy", np.nan))
        group = best.get("feature_group", "")
        if not pd.isna(ba) and ba > 0.80:
            interp = "D1/D3/D7 timepoint is strongly predictable from biological state features"
            use = "interpret high L_domain leakage as partly biological temporal structure, not purely technical batch leakage"
        else:
            interp = "timepoint is not strongly predictable from audited biological state features"
            use = "domain leakage may require stronger technical audit"
        rows.append({
            "audit_block": "ldomain_biology_vs_leakage",
            "finding": f"best_biology_feature_group={group}, balanced_accuracy={ba}",
            "interpretation": interp,
            "recommended_use": use,
        })

    # L_domain variant effect
    if not ldomain_variant.empty and "domain_leakage" in ldomain_variant.columns:
        sub = ldomain_variant.copy()
        if set(sub["variant"]) >= {"none", "adv_coral"}:
            m = sub.set_index("variant")
            none = safe_float(m.loc["none", "domain_leakage"])
            advc = safe_float(m.loc["adv_coral", "domain_leakage"])
            delta = none - advc
            rows.append({
                "audit_block": "ldomain_variant_effect",
                "finding": f"none_leakage={none}, adv_coral_leakage={advc}, reduction={delta}",
                "interpretation": (
                    "adv_coral modestly reduces leakage but leakage remains high"
                    if delta > 0 and advc > 0.80
                    else
                    "L_domain effect is weak or inconsistent"
                ),
                "recommended_use": "limitation / conservative mechanism evidence",
            })

    return pd.DataFrame(rows)


def plot_results(output_diff, readout_summary, ldomain_bio, ldomain_variant, outdir):
    outbase = Path(outdir) / "Fig_Step63D_AblationDiagnosticAudit"

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    axes = axes.ravel()

    # A: output diff
    ax = axes[0]
    if output_diff is not None and not output_diff.empty:
        top = output_diff.sort_values("mean_abs_diff", ascending=False).head(10)
        ax.barh(top["output_feature"], top["mean_abs_diff"])
        ax.set_xlabel("Mean absolute difference")
        ax.set_title("A | full vs no_neighbor_loss output-state difference")
    else:
        ax.text(0.5, 0.5, "No output difference table", ha="center", va="center")
        ax.set_axis_off()

    # B: leakage-free neighbor readout
    ax = axes[1]
    if readout_summary is not None and not readout_summary.empty:
        ax.bar(readout_summary["variant"], readout_summary["mean_rmse"])
        ax.set_ylabel("Mean clean neighbor RMSE")
        ax.set_title("B | Leakage-free neighbor readout")
        ax.tick_params(axis="x", labelrotation=20)
    else:
        ax.text(0.5, 0.5, "No readout summary", ha="center", va="center")
        ax.set_axis_off()

    # C: biology domain predictability
    ax = axes[2]
    if ldomain_bio is not None and not ldomain_bio.empty:
        tmp = ldomain_bio.sort_values("domain_balanced_accuracy", ascending=True)
        ax.barh(tmp["feature_group"], tmp["domain_balanced_accuracy"])
        ax.axvline(1/3, linestyle="--", linewidth=0.8)
        ax.set_xlabel("Timepoint balanced accuracy")
        ax.set_title("C | D1/D3/D7 predictability from biology features")
    else:
        ax.text(0.5, 0.5, "No L_domain biology audit", ha="center", va="center")
        ax.set_axis_off()

    # D: L_domain variant leakage
    ax = axes[3]
    if ldomain_variant is not None and not ldomain_variant.empty and "domain_leakage" in ldomain_variant.columns:
        tmp = ldomain_variant.copy()
        ax.bar(tmp["variant"], tmp["domain_leakage"])
        chance = safe_float(tmp["domain_chance"].dropna().iloc[0]) if "domain_chance" in tmp.columns and tmp["domain_chance"].notna().any() else 1/3
        ax.axhline(chance, linestyle="--", linewidth=0.8)
        ax.set_ylabel("Domain leakage\n(timepoint balanced accuracy)")
        ax.set_title("D | L_domain variant leakage")
        ax.tick_params(axis="x", labelrotation=20)
    else:
        ax.text(0.5, 0.5, "No imported L_domain leakage", ha="center", va="center")
        ax.set_axis_off()

    for ax in axes:
        if ax.has_data():
            ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("Step 63d | Ablation diagnostic audit", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

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
    ap.add_argument("--step7_root", default=str(BASE / "results/step7_strokeniche_ablation/runs"))
    ap.add_argument("--step62_dir", default=str(BASE / "results/step8_strokeniche_perturbmap/generalization_62"))
    ap.add_argument("--step63b_dir", default=str(BASE / "results/step8_strokeniche_perturbmap/mechanism_ablation_63b_debug_missing_final_20260601"))
    ap.add_argument("--step63c_dir", default=str(BASE / "results/step8_strokeniche_perturbmap/mechanism_ablation_63c_no_target_leakage_final_20260601"))
    ap.add_argument("--outroot", default=str(BASE / "results/step8_strokeniche_perturbmap"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outroot) / f"mechanism_ablation_63d_diagnostic_audit_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 63d: ablation diagnostic audit")
    log("=" * 100)
    log(f"outdir={outdir}")

    # Load targets/modeling table.
    modeling_path = Path(args.step62_dir) / "step62_modeling_input_table.csv"
    modeling = read_csv(modeling_path, required=True)

    if "obs_name" not in modeling.columns:
        raise RuntimeError("Step62 modeling table lacks obs_name.")

    modeling["obs_name"] = modeling["obs_name"].astype(str)
    if "timepoint" not in modeling.columns or modeling["timepoint"].isna().all():
        modeling["timepoint"] = detect_timepoint(modeling)

    neighbor_targets = clean_neighbor_targets(modeling)

    if not neighbor_targets:
        raise RuntimeError("No clean neighbor targets detected in Step62 modeling input.")

    # 1. Training history/loss audit.
    training_audit = run_step7_training_loss_audit(args.step7_root)

    # 2. full vs no_neighbor_loss output difference.
    full_pred = find_step7_prediction(args.step7_root, "full")
    nn_pred = find_step7_prediction(args.step7_root, "no_neighbor_loss")

    if full_pred is not None and nn_pred is not None:
        output_diff, output_diff_summary = compare_outputs(full_pred, nn_pred, modeling, neighbor_targets)
    else:
        output_diff = pd.DataFrame()
        output_diff_summary = pd.DataFrame({
            "n_merged_rows": [0],
            "n_output_features_compared": [0],
            "mean_abs_diff_over_features": [np.nan],
            "median_abs_diff_over_features": [np.nan],
            "mean_spearman_between_variants": [np.nan],
        })

    # 3. Leakage-free output-state -> neighbor target association/readout.
    train_obs, test_obs = make_fixed_split(modeling, seed=args.seed)

    readout_rows = []
    assoc_frames = []
    feature_audit_frames = []

    for variant, pred_path in [("full", full_pred), ("no_neighbor_loss", nn_pred)]:
        if pred_path is None:
            readout_rows.append({"variant": variant, "status": "missing_prediction"})
            continue

        summary, per, faudit, status = ridge_neighbor_readout(
            pred_path,
            modeling,
            neighbor_targets,
            train_obs,
            test_obs,
            seed=args.seed,
        )

        if summary is None:
            readout_rows.append({"variant": variant, "status": status})
        else:
            summary["variant"] = variant
            summary["status"] = status
            summary["prediction_path"] = str(pred_path)
            readout_rows.append(summary)

        assoc = output_target_correlation(pred_path, modeling, neighbor_targets)
        if not assoc.empty:
            assoc["variant"] = variant
            assoc_frames.append(assoc)

        if faudit is not None and not faudit.empty:
            faudit["variant"] = variant
            feature_audit_frames.append(faudit)

    readout_summary = pd.DataFrame(readout_rows)
    target_assoc = pd.concat(assoc_frames, ignore_index=True) if assoc_frames else pd.DataFrame()
    feature_audit = pd.concat(feature_audit_frames, ignore_index=True) if feature_audit_frames else pd.DataFrame()

    # Add aggregate association summary.
    if not target_assoc.empty:
        assoc_summary = (
            target_assoc.groupby(["variant", "output_feature"])
            .agg(
                mean_abs_spearman=("abs_spearman", "mean"),
                max_abs_spearman=("abs_spearman", "max"),
                n_targets=("target", "count"),
            )
            .reset_index()
        )
    else:
        assoc_summary = pd.DataFrame()

    # 4. L_domain biology-vs-leakage.
    ldomain_bio = run_ldomain_biology_audit(modeling, seed=args.seed)
    ldomain_variant = import_ldomain_variant_leakage(args.step63b_dir)

    # 5. Interpretation summary.
    interpretation = make_interpretation(
        training_audit=training_audit,
        output_diff_summary=output_diff_summary,
        readout_summary=readout_summary,
        ldomain_bio=ldomain_bio,
        ldomain_variant=ldomain_variant,
    )

    # 6. Plot.
    plot_outputs = plot_results(output_diff, readout_summary, ldomain_bio, ldomain_variant, outdir)

    # Save outputs.
    training_audit.to_csv(outdir / "step63d_step7_training_loss_audit.csv", index=False)
    output_diff.to_csv(outdir / "step63d_no_neighbor_output_difference.csv", index=False)
    output_diff_summary.to_csv(outdir / "step63d_no_neighbor_output_difference_summary.csv", index=False)
    target_assoc.to_csv(outdir / "step63d_no_neighbor_target_association.csv", index=False)
    assoc_summary.to_csv(outdir / "step63d_no_neighbor_target_association_summary.csv", index=False)
    readout_summary.to_csv(outdir / "step63d_no_neighbor_readout_summary.csv", index=False)
    feature_audit.to_csv(outdir / "step63d_no_neighbor_feature_audit.csv", index=False)
    ldomain_bio.to_csv(outdir / "step63d_ldomain_biology_vs_leakage_audit.csv", index=False)
    ldomain_variant.to_csv(outdir / "step63d_ldomain_variant_leakage_imported.csv", index=False)
    interpretation.to_csv(outdir / "step63d_interpretation_summary.csv", index=False)

    report = {
        "status": "ok",
        "outdir": str(outdir),
        "inputs": {
            "step7_root": str(args.step7_root),
            "step62_modeling_input": str(modeling_path),
            "full_prediction": str(full_pred) if full_pred else "",
            "no_neighbor_prediction": str(nn_pred) if nn_pred else "",
            "step63b_dir": str(args.step63b_dir),
            "step63c_dir": str(args.step63c_dir),
        },
        "n_clean_neighbor_targets": int(len(neighbor_targets)),
        "neighbor_targets_head": neighbor_targets[:20],
        "key_findings": interpretation.to_dict(orient="records") if not interpretation.empty else [],
        "plot_outputs": plot_outputs,
        "outputs": {
            "training_loss_audit": str(outdir / "step63d_step7_training_loss_audit.csv"),
            "output_difference": str(outdir / "step63d_no_neighbor_output_difference.csv"),
            "output_difference_summary": str(outdir / "step63d_no_neighbor_output_difference_summary.csv"),
            "target_association": str(outdir / "step63d_no_neighbor_target_association.csv"),
            "target_association_summary": str(outdir / "step63d_no_neighbor_target_association_summary.csv"),
            "readout_summary": str(outdir / "step63d_no_neighbor_readout_summary.csv"),
            "feature_audit": str(outdir / "step63d_no_neighbor_feature_audit.csv"),
            "ldomain_biology_audit": str(outdir / "step63d_ldomain_biology_vs_leakage_audit.csv"),
            "ldomain_variant_leakage_imported": str(outdir / "step63d_ldomain_variant_leakage_imported.csv"),
            "interpretation_summary": str(outdir / "step63d_interpretation_summary.csv"),
            "report_json": str(outdir / "step63d_report.json"),
            "report_txt": str(outdir / "step63d_report.txt"),
        },
        "recommended_interpretation": {
            "no_neighbor_loss": (
                "Use as negative or conservative audit result unless readout_summary shows clear worsening "
                "for no_neighbor_loss under leakage-free output-state features."
            ),
            "L_domain": (
                "Interpret as modest leakage reduction only. If biology/state features strongly predict D1/D3/D7, "
                "high leakage should be described as biologically structured temporal signal rather than purely technical leakage."
            ),
        },
    }

    (outdir / "step63d_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step 63d ablation diagnostic audit report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Interpretation summary:")
    lines.append(interpretation.to_string(index=False) if not interpretation.empty else "No interpretation rows.")
    lines.append("")
    lines.append("Step7 training loss audit head:")
    lines.append(training_audit.head(80).to_string(index=False) if not training_audit.empty else "No training loss audit.")
    lines.append("")
    lines.append("Output difference summary:")
    lines.append(output_diff_summary.to_string(index=False))
    lines.append("")
    lines.append("No-neighbor readout summary:")
    lines.append(readout_summary.to_string(index=False))
    lines.append("")
    lines.append("Output-target association summary:")
    lines.append(assoc_summary.to_string(index=False) if not assoc_summary.empty else "No association summary.")
    lines.append("")
    lines.append("L_domain biology-vs-leakage audit:")
    lines.append(ldomain_bio.to_string(index=False) if not ldomain_bio.empty else "No L_domain biology audit.")
    lines.append("")
    lines.append("Imported L_domain variant leakage:")
    lines.append(ldomain_variant.to_string(index=False) if not ldomain_variant.empty else "No imported variant leakage.")

    (outdir / "step63d_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step 63d")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Interpretation summary:")
    log(interpretation.to_string(index=False) if not interpretation.empty else "No interpretation rows.")
    log("")
    log("No-neighbor readout summary:")
    log(readout_summary.to_string(index=False))
    log("")
    log("L_domain biology-vs-leakage audit:")
    log(ldomain_bio.to_string(index=False) if not ldomain_bio.empty else "No L_domain biology audit.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
