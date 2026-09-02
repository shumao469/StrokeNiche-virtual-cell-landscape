#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
63c_recompute_no_neighbor_loss_no_target_leakage.py

Purpose
-------
Recompute Step7 no_neighbor_loss readout without target leakage.

Why
---
63b filled the missing no_neighbor_loss readout, but feature_cols_used included
neighbor_mean_feature_* and neighbor_delta_vs_self_feature_* columns that were
also target_cols_used. This creates target leakage and makes the tiny RMSE
unusable as strong evidence.

This script:
  1. Loads Step7 full and no_neighbor_loss adapter prediction files.
  2. Loads clean neighbor targets from Step62 modeling table.
  3. Merges by obs_name.
  4. Selects leakage-free features only:
       - repair_score
       - prob_lesion_core
       - prob_peri_infarct
       - prob_remote_like
       - other adapter output-state columns that are NOT neighbor/RCTD/celltype targets
  5. Explicitly excludes:
       - neighbor_mean_feature_*
       - neighbor_delta_vs_self_feature_*
       - RCTD_prop_*
       - RCTD_*
       - celltype composition columns
       - labels, region_true, predicted_celltype, dominant_celltype, train/val/test flags
  6. Computes clean neighbor RMSE / Spearman / skill.
  7. Patches the Step63/63b mechanism summary for step7_no_neighbor_loss.

Outputs
-------
results/step8_strokeniche_perturbmap/mechanism_ablation_63c_no_target_leakage_<tag>/
  step63c_no_neighbor_loss_leakage_free_metrics.csv
  step63c_no_neighbor_loss_leakage_free_per_target.csv
  step63c_feature_audit.csv
  step63c_target_audit.csv
  step63c_mechanism_summary_patched_no_leakage.csv
  step63c_report.json
  step63c_report.txt
  Fig_Step63C_NoNeighborLoss_NoTargetLeakage.pdf/svg/png
"""

from pathlib import Path
import argparse
import json
import re
from datetime import datetime

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error


BASE = Path("/mnt/h/vir/ST")


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


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    return float(np.sqrt(mean_squared_error(y_true[ok], y_pred[ok])))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    if ok.sum() < 3:
        return np.nan
    return float(mean_absolute_error(y_true[ok], y_pred[ok]))


def spearman(a, b):
    x = pd.Series(a).astype(float)
    y = pd.Series(b).astype(float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    return float(x[ok].corr(y[ok], method="spearman"))


def skill_vs_baseline(y_true, y_pred, y_base):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_base = np.asarray(y_base, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(y_base)
    if ok.sum() < 3:
        return np.nan
    mse_model = mean_squared_error(y_true[ok], y_pred[ok])
    mse_base = mean_squared_error(y_true[ok], y_base[ok])
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
    return ""


def detect_timepoint(df):
    out = pd.Series("", index=df.index, dtype=object)
    for c in ["timepoint", "sample", "sample_id", "batch", "orig.ident", "orig_ident"]:
        if c in df.columns:
            v = df[c].map(extract_timepoint)
            out = out.where(out.astype(str).ne(""), v)
    if "obs_name" in df.columns:
        v = df["obs_name"].map(extract_timepoint)
        out = out.where(out.astype(str).ne(""), v)
    return out.replace("", np.nan)


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
    return sorted(candidates, key=lambda p: (0 if p.name == "strokeniche_adapter_predictions.csv" else 1, len(str(p))))[0]


def is_numeric_like(s, frac=0.80):
    x = pd.to_numeric(s, errors="coerce")
    return bool(x.notna().mean() >= frac)


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
            if not any(x in nc for x in bad):
                cols.append(c)
    return list(dict.fromkeys(cols))


def is_forbidden_feature(c, target_cols):
    nc = norm_col(c)

    if c in target_cols:
        return True, "exact_target_column"

    forbidden_prefix = [
        "neighbor_mean_feature_",
        "neighbor_delta_vs_self_feature_",
        "RCTD_prop_",
        "RCTD_",
        "celltype_",
    ]
    if any(str(c).startswith(p) for p in forbidden_prefix):
        return True, "target_or_composition_prefix"

    forbidden_tokens = [
        "obs", "barcode", "key", "timepoint", "sample",
        "train", "val", "test",
        "true", "label", "class",
        "region_true", "region_pred", "state_group",
        "predicted_celltype", "dominant_celltype",
        "cell_index", "index",
        "target_gene", "ligand", "receptor",
        "neighbor", "rctd", "celltype",
        "composition", "dominant",
        "rank",
    ]
    if any(tok in nc for tok in forbidden_tokens):
        return True, "forbidden_token"

    return False, ""


def select_leakage_free_features(df, target_cols, mode="output_state"):
    rows = []
    keep = []

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

    for c in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[c]) or is_numeric_like(df[c])):
            continue

        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() < 0.80:
            rows.append({"column": c, "kept": False, "reason": "too_many_missing"})
            continue
        if x.nunique(dropna=True) <= 1:
            rows.append({"column": c, "kept": False, "reason": "constant"})
            continue

        forbidden, reason = is_forbidden_feature(c, target_cols)
        if forbidden:
            rows.append({"column": c, "kept": False, "reason": reason})
            continue

        if mode == "output_state":
            if c in preferred or c.startswith("prob_") or c.startswith("delta_") or c in ["repair_score", "rescue_score", "delta_norm"]:
                keep.append(c)
                rows.append({"column": c, "kept": True, "reason": "allowed_output_state"})
            else:
                rows.append({"column": c, "kept": False, "reason": "not_output_state_feature"})
        else:
            keep.append(c)
            rows.append({"column": c, "kept": True, "reason": "allowed_numeric_non_target"})

    return list(dict.fromkeys(keep)), pd.DataFrame(rows)


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


def per_target_metrics(y_true, y_pred, y_base, cols):
    rows = []
    for j, c in enumerate(cols):
        yt = y_true[:, j].astype(float)
        yp = y_pred[:, j].astype(float)
        yb = y_base[:, j].astype(float)

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
    return pd.DataFrame(rows)


def evaluate_variant(variant, pred_path, target_df, target_cols, train_obs, test_obs, args):
    if pred_path is None:
        return None, pd.DataFrame(), pd.DataFrame(), {"variant": variant, "status": "missing_prediction"}

    pred = read_csv(pred_path, required=True)
    if "obs_name" not in pred.columns:
        return None, pd.DataFrame(), pd.DataFrame(), {"variant": variant, "status": "prediction_lacks_obs_name"}

    pred["obs_name"] = pred["obs_name"].astype(str)

    # To avoid duplicated target names, remove any clean target columns that are already present in pred.
    pred_clean = pred.drop(columns=[c for c in target_cols if c in pred.columns], errors="ignore")

    merged = pred_clean.merge(
        target_df[["obs_name"] + target_cols],
        on="obs_name",
        how="inner",
    )

    if merged.empty:
        return None, pd.DataFrame(), pd.DataFrame(), {"variant": variant, "status": "no_obs_overlap"}

    fcols, feature_audit = select_leakage_free_features(
        merged,
        target_cols=target_cols,
        mode=args.feature_mode,
    )

    feature_audit["variant"] = variant

    if not fcols:
        return None, pd.DataFrame(), feature_audit, {"variant": variant, "status": "no_leakage_free_features"}

    train_mask = merged["obs_name"].isin(train_obs)
    test_mask = merged["obs_name"].isin(test_obs)

    keep_targets = []
    target_audit_rows = []
    for c in target_cols:
        ytr = pd.to_numeric(merged.loc[train_mask, c], errors="coerce")
        yte = pd.to_numeric(merged.loc[test_mask, c], errors="coerce")
        ok = ytr.notna().sum() >= args.min_train_target_values and yte.notna().sum() >= args.min_test_target_values and ytr.nunique(dropna=True) > 1
        target_audit_rows.append({
            "variant": variant,
            "target": c,
            "kept": bool(ok),
            "train_nonnull": int(ytr.notna().sum()),
            "test_nonnull": int(yte.notna().sum()),
            "train_nunique": int(ytr.nunique(dropna=True)),
        })
        if ok:
            keep_targets.append(c)

    target_audit = pd.DataFrame(target_audit_rows)

    if not keep_targets:
        return None, target_audit, feature_audit, {"variant": variant, "status": "no_valid_targets"}

    X_train = merged.loc[train_mask, fcols].apply(pd.to_numeric, errors="coerce")
    X_test = merged.loc[test_mask, fcols].apply(pd.to_numeric, errors="coerce")
    Y_train = merged.loc[train_mask, keep_targets].apply(pd.to_numeric, errors="coerce")
    Y_test = merged.loc[test_mask, keep_targets].apply(pd.to_numeric, errors="coerce")

    if len(X_train) < args.min_train_cells or len(X_test) < args.min_test_cells:
        return None, target_audit, feature_audit, {"variant": variant, "status": "too_few_rows"}

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(alpha=args.ridge_alpha)),
    ])

    model.fit(X_train, Y_train)
    pred_y = model.predict(X_test)
    base_y = np.tile(Y_train.mean(axis=0).values.reshape(1, -1), (Y_test.shape[0], 1))

    per = per_target_metrics(Y_test.values, pred_y, base_y, keep_targets)
    per["variant"] = variant

    metrics = {
        "component": "Step7_adapter",
        "variant": variant,
        "readout": "leakage_free_clean_neighbor_readout_from_adapter_outputs",
        "prediction_path": str(pred_path),
        "n_merged_rows": int(len(merged)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_features": int(len(fcols)),
        "n_targets": int(len(keep_targets)),
        "neighbor_rmse": float(per["rmse"].mean()),
        "neighbor_median_rmse": float(per["rmse"].median()),
        "neighbor_mae": float(per["mae"].mean()),
        "neighbor_spearman": float(per["spearman"].mean()),
        "neighbor_skill_vs_train_mean": float(per["skill_vs_train_mean"].mean()),
        "feature_cols_used": ";".join(fcols),
        "target_cols_used": ";".join(keep_targets),
        "status": "ok",
        "target_leakage_excluded": True,
    }

    return metrics, per, feature_audit, target_audit


def impairment_score(baseline, ablated):
    if pd.isna(baseline) or pd.isna(ablated):
        return np.nan, np.nan
    delta = ablated - baseline
    imp = ablated - baseline  # RMSE: higher is worse
    denom = max(abs(baseline), abs(ablated), 1e-12)
    return delta, imp / denom


def patch_mechanism_summary(summary_path, metrics_df):
    if summary_path is None or not Path(summary_path).exists():
        return pd.DataFrame()

    s = read_csv(summary_path, required=True)
    if metrics_df.empty or "variant" not in metrics_df.columns:
        return s

    m = metrics_df.set_index("variant")
    if "full" not in m.index or "no_neighbor_loss" not in m.index:
        return s

    b = safe_float(m.loc["full", "neighbor_rmse"])
    a = safe_float(m.loc["no_neighbor_loss", "neighbor_rmse"])
    delta, imp = impairment_score(b, a)

    mask = s["ablation_id"].eq("step7_no_neighbor_loss")
    if mask.any():
        s.loc[mask, "metric_used"] = "neighbor_rmse_leakage_free_output_state_readout"
        s.loc[mask, "baseline_value"] = b
        s.loc[mask, "ablated_value"] = a
        s.loc[mask, "delta_ablated_minus_baseline"] = delta
        s.loc[mask, "impairment_score_norm"] = imp
        s.loc[mask, "evidence_status"] = "ok_no_target_leakage"
        s.loc[mask, "mechanism_interpretation"] = (
            "leakage-free post-hoc readout excludes all neighbor/RCTD/celltype target columns from features; "
            "effect size should be interpreted conservatively"
        )
        s.loc[mask, "baseline_source"] = str(m.loc["full", "prediction_path"])
        s.loc[mask, "ablated_source"] = str(m.loc["no_neighbor_loss", "prediction_path"])

    return s


def find_summary_path(args):
    candidates = []
    if args.step63b_dir:
        candidates.append(Path(args.step63b_dir) / "step63b_mechanism_summary_patched.csv")
    if args.step63_dir:
        candidates.append(Path(args.step63_dir) / "step63_ablation_mechanism_summary.csv")

    candidates.extend(
        sorted(
            (BASE / "results/step8_strokeniche_perturbmap").glob("mechanism_ablation_63b_debug_missing_*/step63b_mechanism_summary_patched.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )
    candidates.extend(
        sorted(
            (BASE / "results/step8_strokeniche_perturbmap").glob("mechanism_ablation_63_*/step63_ablation_mechanism_summary.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    for p in candidates:
        if p.exists():
            return p
    return None


def plot(metrics_df, patched, outdir):
    outbase = Path(outdir) / "Fig_Step63C_NoNeighborLoss_NoTargetLeakage"

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

    ax = axes[0]
    if not metrics_df.empty:
        ax.bar(metrics_df["variant"], metrics_df["neighbor_rmse"])
        ax.set_ylabel("Clean neighbor RMSE")
        ax.set_title("A | Leakage-free neighbor readout")
        ax.tick_params(axis="x", labelrotation=25)
    else:
        ax.text(0.5, 0.5, "No metrics", ha="center", va="center")
        ax.set_axis_off()

    ax = axes[1]
    if not metrics_df.empty:
        ax.bar(metrics_df["variant"], metrics_df["neighbor_spearman"])
        ax.set_ylabel("Mean Spearman")
        ax.set_title("B | Rank-level target stability")
        ax.tick_params(axis="x", labelrotation=25)
    else:
        ax.set_axis_off()

    ax = axes[2]
    if patched is not None and not patched.empty:
        sub = patched[patched["ablation_id"].eq("step7_no_neighbor_loss")]
        if len(sub):
            vals = pd.to_numeric(sub["impairment_score_norm"], errors="coerce").fillna(0)
            ax.barh(["without neighbor loss"], vals)
            ax.axvline(0, linewidth=0.8)
            ax.set_xlabel("Normalized impairment")
            ax.set_title("C | Patched mechanism readout")
        else:
            ax.text(0.5, 0.5, "No patched row", ha="center", va="center")
            ax.set_axis_off()
    else:
        ax.set_axis_off()

    for ax in axes:
        if ax.has_data():
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.suptitle("Step 63c | no_neighbor_loss readout without target leakage", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=500, bbox_inches="tight")
    plt.close(fig)

    return {
        "pdf": str(outbase.with_suffix(".pdf")),
        "svg": str(outbase.with_suffix(".svg")),
        "png": str(outbase.with_suffix(".png")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step7_root", default=str(BASE / "results/step7_strokeniche_ablation/runs"))
    ap.add_argument("--step62_dir", default=str(BASE / "results/step8_strokeniche_perturbmap/generalization_62"))
    ap.add_argument("--step63_dir", default=str(BASE / "results/step8_strokeniche_perturbmap/mechanism_ablation_63_final_20260601"))
    ap.add_argument("--step63b_dir", default=str(BASE / "results/step8_strokeniche_perturbmap/mechanism_ablation_63b_debug_missing_final_20260601"))
    ap.add_argument("--outroot", default=str(BASE / "results/step8_strokeniche_perturbmap"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--feature_mode", default="output_state", choices=["output_state", "all_numeric_non_target"])
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min_train_cells", type=int, default=200)
    ap.add_argument("--min_test_cells", type=int, default=50)
    ap.add_argument("--min_train_target_values", type=int, default=50)
    ap.add_argument("--min_test_target_values", type=int, default=10)
    args = ap.parse_args()

    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outroot) / f"mechanism_ablation_63c_no_target_leakage_{tag}"
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 63c: recompute no_neighbor_loss readout without target leakage")
    log("=" * 100)
    log(f"outdir={outdir}")

    target_path = Path(args.step62_dir) / "step62_modeling_input_table.csv"
    target_df = read_csv(target_path, required=True)
    if "obs_name" not in target_df.columns:
        raise RuntimeError("Step62 modeling table lacks obs_name.")
    target_df["obs_name"] = target_df["obs_name"].astype(str)

    target_cols = clean_neighbor_targets(target_df)
    if not target_cols:
        raise RuntimeError("No clean neighbor targets found in Step62 modeling table.")

    train_obs, test_obs = make_fixed_split(target_df, seed=args.seed)

    all_metrics = []
    all_per = []
    all_feature_audit = []
    all_target_audit = []
    run_audit = []

    for variant in ["full", "no_neighbor_loss"]:
        pred_path = find_step7_prediction(args.step7_root, variant)

        metrics, per, feature_audit, target_audit = evaluate_variant(
            variant,
            pred_path,
            target_df,
            target_cols,
            train_obs,
            test_obs,
            args,
        )

        run_audit.append({
            "variant": variant,
            "prediction_path": str(pred_path) if pred_path else "",
            "status": metrics["status"] if metrics is not None else "failed_or_no_metrics",
            "n_features": metrics["n_features"] if metrics is not None else 0,
            "n_targets": metrics["n_targets"] if metrics is not None else 0,
        })

        if metrics is not None:
            all_metrics.append(metrics)
        if per is not None and not per.empty:
            all_per.append(per)
        if feature_audit is not None and not feature_audit.empty:
            all_feature_audit.append(feature_audit)
        if target_audit is not None and not target_audit.empty:
            all_target_audit.append(target_audit)

    metrics_df = pd.DataFrame(all_metrics)
    per_df = pd.concat(all_per, ignore_index=True) if all_per else pd.DataFrame()
    feature_audit_df = pd.concat(all_feature_audit, ignore_index=True) if all_feature_audit else pd.DataFrame()
    target_audit_df = pd.concat(all_target_audit, ignore_index=True) if all_target_audit else pd.DataFrame()
    run_audit_df = pd.DataFrame(run_audit)

    summary_path = find_summary_path(args)
    patched = patch_mechanism_summary(summary_path, metrics_df)

    metrics_df.to_csv(outdir / "step63c_no_neighbor_loss_leakage_free_metrics.csv", index=False)
    per_df.to_csv(outdir / "step63c_no_neighbor_loss_leakage_free_per_target.csv", index=False)
    feature_audit_df.to_csv(outdir / "step63c_feature_audit.csv", index=False)
    target_audit_df.to_csv(outdir / "step63c_target_audit.csv", index=False)
    run_audit_df.to_csv(outdir / "step63c_run_audit.csv", index=False)
    patched.to_csv(outdir / "step63c_mechanism_summary_patched_no_leakage.csv", index=False)

    plot_outputs = plot(metrics_df, patched, outdir)

    # Interpretation
    interpretation = "not_available"
    if not metrics_df.empty and set(metrics_df["variant"]) >= {"full", "no_neighbor_loss"}:
        m = metrics_df.set_index("variant")
        b = safe_float(m.loc["full", "neighbor_rmse"])
        a = safe_float(m.loc["no_neighbor_loss", "neighbor_rmse"])
        delta = a - b
        if pd.isna(delta):
            interpretation = "metric unavailable"
        elif delta > 0:
            interpretation = "no_neighbor_loss has higher leakage-free neighbor RMSE than full; directionally consistent with weakened neighborhood reconstruction"
        elif delta < 0:
            interpretation = "no_neighbor_loss does not worsen leakage-free neighbor RMSE; the neighbor-loss ablation is not supported by this post-hoc output-state readout"
        else:
            interpretation = "no measurable difference in leakage-free neighbor RMSE"

    report = {
        "status": "ok",
        "outdir": str(outdir),
        "target_source": str(target_path),
        "summary_input": str(summary_path) if summary_path else "",
        "n_clean_neighbor_targets": len(target_cols),
        "feature_mode": args.feature_mode,
        "target_leakage_policy": {
            "excluded_exact_target_columns": True,
            "excluded_neighbor_mean_feature_prefix": True,
            "excluded_neighbor_delta_vs_self_feature_prefix": True,
            "excluded_RCTD_and_celltype_prefixes": True,
            "excluded_train_val_test_and_label_columns": True,
        },
        "interpretation": interpretation,
        "plot_outputs": plot_outputs,
        "outputs": {
            "metrics": str(outdir / "step63c_no_neighbor_loss_leakage_free_metrics.csv"),
            "per_target": str(outdir / "step63c_no_neighbor_loss_leakage_free_per_target.csv"),
            "feature_audit": str(outdir / "step63c_feature_audit.csv"),
            "target_audit": str(outdir / "step63c_target_audit.csv"),
            "run_audit": str(outdir / "step63c_run_audit.csv"),
            "patched_summary": str(outdir / "step63c_mechanism_summary_patched_no_leakage.csv"),
        },
    }

    (outdir / "step63c_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = []
    lines.append("Step 63c no_neighbor_loss leakage-free readout report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Metrics:")
    lines.append(metrics_df.to_string(index=False) if not metrics_df.empty else "No metrics.")
    lines.append("")
    lines.append("Run audit:")
    lines.append(run_audit_df.to_string(index=False))
    lines.append("")
    lines.append("Kept leakage-free features:")
    if not feature_audit_df.empty:
        kept = feature_audit_df[feature_audit_df["kept"] == True]
        lines.append(kept.to_string(index=False) if not kept.empty else "No kept features.")
    else:
        lines.append("No feature audit.")
    lines.append("")
    lines.append("Patched no_neighbor_loss row:")
    if not patched.empty:
        sub = patched[patched["ablation_id"].eq("step7_no_neighbor_loss")]
        lines.append(sub.to_string(index=False) if not sub.empty else "No patched row.")
    else:
        lines.append("No patched summary.")

    (outdir / "step63c_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step 63c")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Metrics:")
    log(metrics_df.to_string(index=False) if not metrics_df.empty else "No metrics.")
    log("")
    log("Patched no_neighbor_loss row:")
    if not patched.empty:
        sub = patched[patched["ablation_id"].eq("step7_no_neighbor_loss")]
        log(sub.to_string(index=False) if not sub.empty else "No patched row.")
    else:
        log("No patched summary.")


if __name__ == "__main__":
    main()
