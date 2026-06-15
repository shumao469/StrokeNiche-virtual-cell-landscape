#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step81 | Surrogate feature-response audit for virtual perturbations

Purpose
-------
Plot candidate feature scores versus virtual perturbation response metrics:
  x = baseline candidate gene/module score
  y = core reduction or repair gain
  color = state_group or timepoint

This is designed as a small QC/supporting figure for Step72/Step78.

Important
---------
This is NOT SHAP unless actual SHAP values are supplied.
This is NOT wet-lab KO/blockade.
It is a surrogate feature-response association audit.

Required for true response audit
--------------------------------
A Step72 per-spot/per-cell counterfactual table with:
  - obs/cell/spot id
  - candidate/perturbation name
  - either delta_core / repair_gain columns
    OR baseline and perturbed core/repair columns.

If such a table is not found, the script can optionally run with:
  --allow_fallback

In fallback mode it generates expression-vs-baseline-state-score association,
not perturbation-response association.
"""

import os
import re
import json
import glob
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import anndata as ad
except Exception as e:
    raise ImportError("Please install anndata: pip install anndata") from e

try:
    from scipy import sparse
    from scipy.stats import spearmanr
except Exception as e:
    raise ImportError("Please install scipy: pip install scipy") from e


# -----------------------------
# Candidate feature definitions
# -----------------------------

CANDIDATE_FEATURES = {
    "Ccl2/Ccr2-Ackr1 blockade": ["Ccl2", "Ccr2", "Ackr1"],
    "Spp1-Cd44 blockade": ["Spp1", "Cd44"],
    "Vegfa-Flt1/Kdr blockade": ["Vegfa", "Flt1", "Kdr"],
    "Ferroptosis down": ["Hmox1", "Fth1", "Slc7a11", "Gpx4"],
    "Repair-ECM up": ["Col1a1", "Col3a1", "Fn1", "Spp1", "Vim", "Apoe", "Postn", "Timp1"],
}

CANON_PATTERNS = {
    "Ccl2/Ccr2-Ackr1 blockade": ["ccl2", "ccr2", "ackr1"],
    "Spp1-Cd44 blockade": ["spp1", "cd44"],
    "Vegfa-Flt1/Kdr blockade": ["vegfa", "flt1", "kdr"],
    "Ferroptosis down": ["ferroptosis"],
    "Repair-ECM up": ["repair", "ecm"],
}


STATE_COLORS = {
    "lesion-core-like": "#d62728",
    "core-like": "#d62728",
    "core": "#d62728",
    "peri-infarct": "#f0b000",
    "peri": "#f0b000",
    "remote-like": "#4aa3df",
    "remote": "#4aa3df",
}


# -----------------------------
# Utilities
# -----------------------------

def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_label(x: str) -> str:
    s = clean_str(x).lower()
    s = s.replace("_", "-").replace(" ", "-")
    if s in {"core", "core-like", "lesion-core", "lesion-core-like"}:
        return "lesion-core-like"
    if s in {"peri", "peri-infarct", "periinfarct"}:
        return "peri-infarct"
    if s in {"remote", "remote-like"}:
        return "remote-like"
    return clean_str(x)


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    med = np.nanmedian(x[ok])
    q1, q3 = np.nanpercentile(x[ok], [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        sd = np.nanstd(x[ok])
        if not np.isfinite(sd) or sd <= 0:
            out[ok] = 0.0
        else:
            out[ok] = (x[ok] - np.nanmean(x[ok])) / sd
    else:
        out[ok] = (x[ok] - med) / (iqr / 1.349)
    return np.clip(out, -5, 5)


def minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    lo, hi = np.nanpercentile(x[ok], [1, 99])
    if hi <= lo:
        out[ok] = 0
    else:
        out[ok] = np.clip((x[ok] - lo) / (hi - lo), 0, 1)
    return out


def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def read_state_table(path: str) -> pd.DataFrame:
    return read_table(path)


def find_merge_key(adata, state: pd.DataFrame) -> Tuple[str, Optional[str], str]:
    obs_names = pd.Index(adata.obs_names.astype(str))
    candidates = [
        "obs_name", "barcode", "spot", "spot_id", "cell_id", "id",
        "_obs_names_", "adata_obs_name", "index"
    ]

    best_col, best_overlap = None, -1

    for c in candidates:
        if c in state.columns:
            overlap = state[c].astype(str).isin(obs_names).sum()
            if overlap > best_overlap:
                best_col, best_overlap = c, overlap

    if best_overlap <= 0:
        for c in state.columns:
            if state[c].dtype == object or str(state[c].dtype).startswith("string"):
                overlap = state[c].astype(str).isin(obs_names).sum()
                if overlap > best_overlap:
                    best_col, best_overlap = c, overlap

    if best_col is not None and best_overlap > 0:
        return "_obs_names_", best_col, f"id_merge_overlap={best_overlap}"

    if len(state) == adata.n_obs:
        return "_row_order_", None, "row_order_merge"

    raise ValueError("Cannot merge state table to AnnData.")


def merge_metadata(adata, state: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    obs = adata.obs.copy()
    obs["_adata_id_"] = adata.obs_names.astype(str)

    mode, state_key, report = find_merge_key(adata, state)
    audit = {
        "n_obs": int(adata.n_obs),
        "state_rows": int(len(state)),
        "merge_mode_detail": report,
    }

    if mode == "_obs_names_":
        st = state.copy()
        st["_adata_id_"] = st[state_key].astype(str)
        merged = obs.merge(st, on="_adata_id_", how="left", suffixes=("", "_state"))
        audit["merge_mode"] = "id"
        audit["merge_key_state"] = state_key
        audit["state_merge_overlap"] = int(merged["_adata_id_"].isin(st["_adata_id_"]).sum())
        return merged.reset_index(drop=True), audit

    if mode == "_row_order_":
        merged = pd.concat([obs.reset_index(drop=True), state.reset_index(drop=True)], axis=1)
        audit["merge_mode"] = "row_order"
        audit["state_merge_overlap"] = int(len(state))
        return merged.reset_index(drop=True), audit

    raise RuntimeError("Unexpected merge mode")


def build_gene_index(adata) -> Dict[str, int]:
    gene_to_idx = {}

    def add(name, idx):
        s = clean_str(name)
        if s == "":
            return
        k = s.lower()
        if k not in gene_to_idx:
            gene_to_idx[k] = idx

    for i, g in enumerate(adata.var_names.astype(str)):
        add(g, i)

    for col in ["gene_symbol", "gene_symbols", "symbol", "gene_name", "feature_name", "name"]:
        if col in adata.var.columns:
            for i, g in enumerate(adata.var[col].astype(str).values):
                add(g, i)

    return gene_to_idx


def extract_gene_vector(adata, gene: str, gene_index: Dict[str, int]) -> Tuple[np.ndarray, Dict]:
    audit = {"gene": gene, "found": False, "matched_index": None}
    k = gene.lower()

    if k not in gene_index:
        return np.full(adata.n_obs, np.nan, dtype=float), audit

    idx = gene_index[k]
    X = adata.X[:, idx]

    if sparse is not None and sparse.issparse(X):
        arr = np.asarray(X.toarray()).ravel()
    else:
        arr = np.asarray(X).ravel()

    arr = arr.astype(float)
    audit["found"] = True
    audit["matched_index"] = int(idx)
    audit["var_name"] = str(adata.var_names[idx])
    return arr, audit


def compute_candidate_scores(adata) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_index = build_gene_index(adata)
    score_df = pd.DataFrame({"_adata_id_": adata.obs_names.astype(str)})
    audit_rows = []

    gene_cache = {}

    for cand, genes in CANDIDATE_FEATURES.items():
        gene_zs = []

        for g in genes:
            if g not in gene_cache:
                vals, aud = extract_gene_vector(adata, g, gene_index)
                aud["candidate"] = cand
                gene_cache[g] = vals
                audit_rows.append(aud)

            vals = gene_cache[g]
            if np.isfinite(vals).sum() > 0:
                gene_zs.append(robust_z(vals))

        if len(gene_zs) == 0:
            score = np.full(adata.n_obs, np.nan)
        else:
            score = np.nanmean(np.vstack(gene_zs), axis=0)

        score_df[cand + "__feature_score"] = score
        score_df[cand + "__feature_score01"] = minmax01(score)

    return score_df, pd.DataFrame(audit_rows)


def canonical_candidate_name(x: str) -> Optional[str]:
    s = clean_str(x).lower()
    s2 = s.replace("_", "-").replace(" ", "-")

    for canon, tokens in CANON_PATTERNS.items():
        if all(tok in s2 for tok in tokens):
            return canon

    # permissive matching
    if "ccl2" in s2 or "ccr2" in s2 or "ackr1" in s2:
        return "Ccl2/Ccr2-Ackr1 blockade"
    if "spp1" in s2 or "cd44" in s2:
        return "Spp1-Cd44 blockade"
    if "vegfa" in s2 or "flt1" in s2 or "kdr" in s2:
        return "Vegfa-Flt1/Kdr blockade"
    if "ferro" in s2:
        return "Ferroptosis down"
    if "repair" in s2 or "ecm" in s2:
        return "Repair-ECM up"

    return None


# -----------------------------
# Step72 response loading
# -----------------------------

def score_csv_for_response(path: str) -> Tuple[int, Dict]:
    """
    Lightweight scoring to identify likely Step72 per-cell response tables.
    """
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception:
        return -999, {"path": path, "error": "read_failed"}

    cols = [c.lower() for c in df.columns]
    joined = " ".join(cols)

    score = 0
    if any(k in joined for k in ["candidate", "perturb", "intervention", "blockade", "ko"]):
        score += 5
    if any(k in joined for k in ["obs", "cell", "spot", "barcode", "adata"]):
        score += 3
    if any(k in joined for k in ["delta", "shift", "gain", "rescue"]):
        score += 6
    if "core" in joined:
        score += 3
    if "repair" in joined:
        score += 2
    if any(k in os.path.basename(path).lower() for k in ["per_cell", "per-cell", "per_spot", "cell", "spot", "counterfactual", "perturb"]):
        score += 4

    return score, {"path": path, "score": score, "columns": df.columns.tolist()}


def discover_step72_response_table(step72_dir: str) -> Tuple[Optional[str], List[Dict]]:
    if step72_dir is None or step72_dir == "" or not os.path.exists(step72_dir):
        return None, []

    files = glob.glob(os.path.join(step72_dir, "**", "*.csv"), recursive=True)
    audits = []

    for f in files:
        score, aud = score_csv_for_response(f)
        audits.append(aud)

    audits_sorted = sorted(audits, key=lambda x: x.get("score", -999), reverse=True)
    if len(audits_sorted) == 0:
        return None, []

    best = audits_sorted[0]
    if best.get("score", -999) < 8:
        return None, audits_sorted

    return best["path"], audits_sorted


def find_col_by_keywords(cols: List[str], required: List[str], any_of: Optional[List[str]] = None) -> Optional[str]:
    for c in cols:
        lc = c.lower()
        if all(r in lc for r in required):
            if any_of is None or any(a in lc for a in any_of):
                return c
    return None


def find_id_col_for_response(resp: pd.DataFrame, meta: pd.DataFrame) -> Optional[str]:
    meta_ids = set(meta["_adata_id_"].astype(str).values)

    candidates = [
        "_adata_id_", "obs_name", "barcode", "spot_id", "spot", "cell_id",
        "adata_obs_name", "id", "index"
    ]

    best_col, best_overlap = None, -1

    for c in candidates:
        if c in resp.columns:
            overlap = resp[c].astype(str).isin(meta_ids).sum()
            if overlap > best_overlap:
                best_col, best_overlap = c, overlap

    if best_overlap <= 0:
        for c in resp.columns:
            if resp[c].dtype == object or str(resp[c].dtype).startswith("string"):
                overlap = resp[c].astype(str).isin(meta_ids).sum()
                if overlap > best_overlap:
                    best_col, best_overlap = c, overlap

    if best_col is not None and best_overlap > 0:
        return best_col

    return None


def find_candidate_col(resp: pd.DataFrame) -> Optional[str]:
    for c in resp.columns:
        lc = c.lower()
        if any(k in lc for k in ["candidate", "perturb", "intervention", "blockade", "ko_name", "target"]):
            return c
    return None


def standardize_response_table(resp: pd.DataFrame, meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Return long response table with:
      _adata_id_, candidate, delta_core, core_reduction, delta_repair, repair_gain
    """
    cols = resp.columns.tolist()
    audit = {"input_columns": cols}

    id_col = find_id_col_for_response(resp, meta)
    cand_col = find_candidate_col(resp)

    if id_col is None:
        if len(resp) == len(meta):
            resp = resp.copy()
            resp["_adata_id_"] = meta["_adata_id_"].astype(str).values
            id_col = "_adata_id_"
            audit["id_mode"] = "row_order"
        else:
            raise ValueError("Cannot identify obs/cell id column in Step72 response table.")

    if cand_col is None:
        raise ValueError("Cannot identify candidate/perturbation column in Step72 response table.")

    out = pd.DataFrame()
    out["_adata_id_"] = resp[id_col].astype(str)
    out["candidate_raw"] = resp[cand_col].astype(str)
    out["candidate"] = out["candidate_raw"].map(canonical_candidate_name)

    # delta core direct
    delta_core_col = None
    for pat in [
        ["delta", "core"],
        ["shift", "core"],
        ["d_core"],
    ]:
        delta_core_col = find_col_by_keywords(cols, pat)
        if delta_core_col is not None:
            break

    if delta_core_col is not None:
        out["delta_core"] = pd.to_numeric(resp[delta_core_col], errors="coerce")
        audit["delta_core_source"] = delta_core_col
    else:
        base_core = find_col_by_keywords(cols, ["core"], ["base", "before", "control", "orig"])
        pert_core = find_col_by_keywords(cols, ["core"], ["pert", "after", "counter", "ko", "block", "new"])
        if base_core is not None and pert_core is not None:
            out["delta_core"] = (
                pd.to_numeric(resp[pert_core], errors="coerce") -
                pd.to_numeric(resp[base_core], errors="coerce")
            )
            audit["delta_core_source"] = [pert_core, base_core]
        else:
            out["delta_core"] = np.nan
            audit["delta_core_source"] = None

    out["core_reduction"] = np.maximum(-out["delta_core"].astype(float), 0)

    # repair direct
    repair_gain_col = None
    for pat in [
        ["repair", "gain"],
        ["rescue", "score"],
        ["response", "priority"],
    ]:
        repair_gain_col = find_col_by_keywords(cols, pat)
        if repair_gain_col is not None:
            break

    if repair_gain_col is not None:
        out["repair_gain"] = pd.to_numeric(resp[repair_gain_col], errors="coerce")
        audit["repair_gain_source"] = repair_gain_col
        out["delta_repair"] = out["repair_gain"]
    else:
        delta_repair_col = None
        for pat in [
            ["delta", "repair"],
            ["shift", "repair"],
            ["d_repair"],
        ]:
            delta_repair_col = find_col_by_keywords(cols, pat)
            if delta_repair_col is not None:
                break

        if delta_repair_col is not None:
            out["delta_repair"] = pd.to_numeric(resp[delta_repair_col], errors="coerce")
            out["repair_gain"] = np.maximum(out["delta_repair"].astype(float), 0)
            audit["repair_gain_source"] = delta_repair_col
        else:
            base_repair = find_col_by_keywords(cols, ["repair"], ["base", "before", "control", "orig"])
            pert_repair = find_col_by_keywords(cols, ["repair"], ["pert", "after", "counter", "ko", "block", "new"])
            if base_repair is not None and pert_repair is not None:
                out["delta_repair"] = (
                    pd.to_numeric(resp[pert_repair], errors="coerce") -
                    pd.to_numeric(resp[base_repair], errors="coerce")
                )
                out["repair_gain"] = np.maximum(out["delta_repair"].astype(float), 0)
                audit["repair_gain_source"] = [pert_repair, base_repair]
            else:
                out["delta_repair"] = np.nan
                out["repair_gain"] = np.nan
                audit["repair_gain_source"] = None

    out = out[out["candidate"].notna()].copy()
    audit["standardized_rows"] = int(len(out))
    audit["candidate_counts"] = out["candidate"].value_counts().to_dict()

    if out["delta_core"].notna().sum() == 0 and out["repair_gain"].notna().sum() == 0:
        raise ValueError("No usable delta_core/core_reduction or repair_gain columns found.")

    return out, audit


def load_step72_response(
    step72_dir: str,
    step72_cell_table: Optional[str],
    meta: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    audit = {}

    if step72_cell_table is not None and step72_cell_table != "":
        path = step72_cell_table
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        candidates = []
    else:
        path, candidates = discover_step72_response_table(step72_dir)

    audit["step72_cell_table_used"] = path
    audit["candidate_csvs"] = candidates[:20] if candidates else []

    if path is None:
        return None, audit

    resp = read_table(path)
    standardized, std_audit = standardize_response_table(resp, meta)
    audit.update(std_audit)
    return standardized, audit


# -----------------------------
# Fallback mode
# -----------------------------

def infer_prob_col(meta: pd.DataFrame, keys: List[str]) -> Optional[str]:
    cols = meta.columns.tolist()
    for c in cols:
        lc = c.lower()
        if all(k in lc for k in keys):
            return c
    return None


def make_fallback_response(meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Expression-vs-baseline-state association table.
    This is not perturbation response.
    """
    core_col = (
        infer_prob_col(meta, ["core", "prob"]) or
        infer_prob_col(meta, ["core", "probability"]) or
        infer_prob_col(meta, ["core"])
    )
    repair_col = (
        infer_prob_col(meta, ["repair", "score"]) or
        infer_prob_col(meta, ["repair"])
    )

    if core_col is None:
        raise ValueError("Fallback requested but no core probability/score column found.")

    rows = []
    for cand in CANDIDATE_FEATURES:
        tmp = pd.DataFrame()
        tmp["_adata_id_"] = meta["_adata_id_"].astype(str)
        tmp["candidate"] = cand
        tmp["delta_core"] = np.nan
        tmp["core_reduction"] = pd.to_numeric(meta[core_col], errors="coerce")
        if repair_col is not None:
            tmp["delta_repair"] = np.nan
            tmp["repair_gain"] = pd.to_numeric(meta[repair_col], errors="coerce")
        else:
            tmp["delta_repair"] = np.nan
            tmp["repair_gain"] = np.nan
        rows.append(tmp)

    out = pd.concat(rows, axis=0, ignore_index=True)
    audit = {
        "fallback_mode": True,
        "core_metric_used_as_y": core_col,
        "repair_metric_used_as_y": repair_col,
        "interpretation": (
            "Fallback mode: y-axis is baseline state/repair score, not Step72 perturbation response."
        )
    }
    return out, audit


# -----------------------------
# Plotting
# -----------------------------

def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    if np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def add_trend(ax, x, y, color="white"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return
    try:
        coef = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(np.nanpercentile(x[ok], 2), np.nanpercentile(x[ok], 98), 100)
        ys = coef[0] * xs + coef[1]
        ax.plot(xs, ys, color=color, lw=1.5, alpha=0.9)
    except Exception:
        return


def plot_feature_response(
    plot_df: pd.DataFrame,
    outdir: str,
    color_col: str,
    fallback_mode: bool,
) -> Dict[str, str]:

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 180,
        "savefig.dpi": 450,
    })

    candidates = list(CANDIDATE_FEATURES.keys())
    n = len(candidates)

    fig, axes = plt.subplots(
        nrows=n, ncols=2,
        figsize=(10.5, max(9, 1.85 * n)),
        sharex=False, sharey=False
    )

    if n == 1:
        axes = np.array([axes])

    fig.patch.set_facecolor("#070b10")

    metric_cols = ["core_reduction", "repair_gain"]
    metric_titles = [
        "Core reduction" if not fallback_mode else "Baseline core score",
        "Repair gain" if not fallback_mode else "Baseline repair score"
    ]

    summary_rows = []

    for i, cand in enumerate(candidates):
        sub = plot_df[plot_df["candidate"] == cand].copy()
        xcol = cand + "__feature_score"

        for j, metric in enumerate(metric_cols):
            ax = axes[i, j]
            ax.set_facecolor("#070b10")

            if sub.empty or xcol not in sub.columns or metric not in sub.columns:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", color="white", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x = pd.to_numeric(sub[xcol], errors="coerce").values
            y = pd.to_numeric(sub[metric], errors="coerce").values

            if color_col in sub.columns:
                labels = sub[color_col].astype(str).map(normalize_label).values
                colors = [STATE_COLORS.get(v, "#aaaaaa") for v in labels]
            else:
                colors = "#aaaaaa"

            ax.scatter(
                x, y,
                s=5,
                c=colors,
                alpha=0.35,
                edgecolor="none",
                rasterized=True
            )
            add_trend(ax, x, y, color="white")

            rho = safe_spearman(x, y)
            n_ok = int((np.isfinite(x) & np.isfinite(y)).sum())

            ax.text(
                0.03, 0.94,
                f"ρ={rho:.2f}, n={n_ok}" if np.isfinite(rho) else f"ρ=NA, n={n_ok}",
                transform=ax.transAxes,
                ha="left", va="top",
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="#111820", ec="none", alpha=0.65)
            )

            if i == 0:
                ax.set_title(metric_titles[j], color="white", fontweight="bold", pad=6)

            if j == 0:
                ax.set_ylabel(cand + "\nresponse", color="white", fontweight="bold")
            else:
                ax.set_ylabel("")

            ax.set_xlabel("Candidate feature score (robust z)", color="white")
            ax.tick_params(colors="0.85")
            for spine in ax.spines.values():
                spine.set_color("0.35")
            ax.grid(True, color="0.18", linewidth=0.5, alpha=0.8)

            summary_rows.append({
                "candidate": cand,
                "metric": metric,
                "spearman_rho": rho,
                "n": n_ok,
                "fallback_mode": fallback_mode,
            })

    title = "Surrogate feature-response relationships"
    if fallback_mode:
        title += " | fallback baseline-score audit"

    fig.suptitle(title, fontsize=18, fontweight="bold", color="white", y=0.985)

    note = (
        "x-axis shows candidate gene/module feature score. y-axis shows Step72-derived virtual response "
        "when available. This is a surrogate association audit, not SHAP and not wet-lab KO/blockade."
    )
    if fallback_mode:
        note = (
            "Fallback mode: y-axis uses baseline state/repair score because no usable Step72 per-cell response table was found. "
            "Do not interpret this as perturbation response."
        )

    fig.text(0.5, 0.025, note, ha="center", va="center", color="0.85", fontsize=8.5)

    fig.subplots_adjust(left=0.22, right=0.97, top=0.94, bottom=0.07, hspace=0.55, wspace=0.30)

    paths = {}
    prefix = "Fig_Step81_SurrogateFeatureResponseAudit"
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(outdir, f"{prefix}.{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[ext] = path

    plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(outdir, "step81_feature_response_spearman_summary.csv")
    summary.to_csv(summary_path, index=False)
    paths["summary_csv"] = summary_path

    return paths


def plot_correlation_heatmap(summary_csv: str, outdir: str) -> Dict[str, str]:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return {}

    mat = df.pivot(index="candidate", columns="metric", values="spearman_rho")
    mat = mat.reindex(index=list(CANDIDATE_FEATURES.keys()))

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    fig.patch.set_facecolor("#070b10")
    ax.set_facecolor("#070b10")

    im = ax.imshow(mat.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=35, ha="right", color="white")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, color="white")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.values[i, j]
            txt = "NA" if not np.isfinite(val) else f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)

    ax.set_title("Feature-response Spearman correlation", color="white", fontweight="bold", pad=12)

    for spine in ax.spines.values():
        spine.set_color("0.35")
    ax.tick_params(colors="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Spearman ρ", color="white")
    cbar.ax.tick_params(colors="white")

    fig.tight_layout()

    paths = {}
    prefix = "Fig_Step81_FeatureResponseCorrelationHeatmap"
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(outdir, f"{prefix}.{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[ext] = path
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--step72_dir", default="")
    parser.add_argument("--step72_cell_table", default="")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label_col", default="state_group")
    parser.add_argument("--time_col", default="timepoint")
    parser.add_argument("--color_by", default="state_group", choices=["state_group", "timepoint"])
    parser.add_argument("--allow_fallback", action="store_true")
    args = parser.parse_args()

    mkdir(args.outdir)

    print("=" * 100)
    print("Step81 | Surrogate feature-response audit for virtual perturbations")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"step72_dir={args.step72_dir}")
    print(f"step72_cell_table={args.step72_cell_table}")
    print(f"outdir={args.outdir}")

    adata = read_h5ad(args.h5ad)
    state = read_state_table(args.state_table)
    meta, merge_audit = merge_metadata(adata, state)

    if args.label_col not in meta.columns:
        raise ValueError(f"--label_col={args.label_col} not found in merged metadata")
    if args.time_col not in meta.columns:
        raise ValueError(f"--time_col={args.time_col} not found in merged metadata")

    meta[args.label_col] = meta[args.label_col].map(normalize_label)
    meta[args.time_col] = meta[args.time_col].astype(str)

    score_df, gene_audit = compute_candidate_scores(adata)

    meta2 = meta.merge(score_df, on="_adata_id_", how="left")

    fallback_mode = False

    try:
        response_df, response_audit = load_step72_response(
            args.step72_dir,
            args.step72_cell_table,
            meta2
        )
        if response_df is None:
            raise FileNotFoundError("No usable Step72 per-cell response CSV found.")
    except Exception as e:
        if not args.allow_fallback:
            raise RuntimeError(
                "Could not load usable Step72 per-cell response table. "
                "Please pass --step72_cell_table explicitly, or rerun with --allow_fallback "
                "for expression-vs-baseline-score QC only. Original error: "
                f"{repr(e)}"
            )
        response_df, fb_audit = make_fallback_response(meta2)
        response_audit = {"fallback_reason": repr(e), **fb_audit}
        fallback_mode = True

    plot_df = response_df.merge(meta2, on="_adata_id_", how="left", suffixes=("", "_meta"))

    color_col = args.label_col if args.color_by == "state_group" else args.time_col
    fig_paths = plot_feature_response(plot_df, args.outdir, color_col=color_col, fallback_mode=fallback_mode)
    heatmap_paths = plot_correlation_heatmap(fig_paths["summary_csv"], args.outdir)

    gene_audit_path = os.path.join(args.outdir, "step81_candidate_feature_gene_matching_audit.csv")
    gene_audit.to_csv(gene_audit_path, index=False)

    response_out = os.path.join(args.outdir, "step81_merged_feature_response_table.csv")
    keep_cols = [
        "_adata_id_", "candidate", "delta_core", "core_reduction",
        "delta_repair", "repair_gain", args.label_col, args.time_col
    ]
    for cand in CANDIDATE_FEATURES:
        keep_cols.append(cand + "__feature_score")
        keep_cols.append(cand + "__feature_score01")
    keep_cols = [c for c in keep_cols if c in plot_df.columns]
    plot_df[keep_cols].to_csv(response_out, index=False)

    report = {
        "status": "ok",
        "fallback_mode": fallback_mode,
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "step72_dir": args.step72_dir,
        "step72_cell_table": args.step72_cell_table,
        "outdir": args.outdir,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "merge_audit": merge_audit,
        "response_audit": response_audit,
        "missing_feature_genes": gene_audit.loc[~gene_audit["found"], "gene"].drop_duplicates().tolist(),
        "outputs": {
            "main_figure": {k: v for k, v in fig_paths.items() if k in ["png", "pdf", "svg"]},
            "correlation_heatmap": heatmap_paths,
            "spearman_summary_csv": fig_paths["summary_csv"],
            "merged_feature_response_table": response_out,
            "gene_audit_csv": gene_audit_path,
        },
        "interpretation_note": (
            "Step81 is a surrogate feature-response association audit. It should not be described as SHAP "
            "unless true SHAP values are supplied, and should not be interpreted as wet-lab KO/blockade evidence."
        )
    }

    report_path = os.path.join(args.outdir, "step81_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()