#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step81B | Robust surrogate feature-response audit

This script generates:
1. Candidate feature score vs core-reduction scatter
2. Candidate feature score vs repair-gain scatter
3. Spearman correlation heatmap

Preferred input:
    per-cell/per-spot virtual perturbation response table with obs/cell/spot ID.

If no response table is found:
    use --allow_fallback to generate baseline feature-state association QC.
    Fallback is NOT perturbation-response evidence.

Interpretation:
    This is a surrogate computational association audit.
    It is not SHAP unless true SHAP values are supplied.
    It is not wet-lab KO/blockade validation.
"""

import os
import re
import json
import glob
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

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
    raise ImportError("Please install scipy") from e


CANDIDATE_FEATURES = {
    "Ccl2/Ccr2-Ackr1 blockade": ["Ccl2", "Ccr2", "Ackr1"],
    "Spp1-Cd44 blockade": ["Spp1", "Cd44"],
    "Vegfa-Flt1/Kdr blockade": ["Vegfa", "Flt1", "Kdr"],
    "Ferroptosis down": ["Hmox1", "Fth1", "Slc7a11", "Gpx4"],
    "Repair-ECM up": ["Col1a1", "Col3a1", "Fn1", "Spp1", "Vim", "Apoe", "Postn", "Timp1"],
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


def mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_label(x) -> str:
    s = clean_str(x).lower().replace("_", "-").replace(" ", "-")
    if s in {"core", "core-like", "lesion-core", "lesion-core-like"}:
        return "lesion-core-like"
    if s in {"peri", "peri-infarct", "periinfarct", "peri-infarct-like"}:
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
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out
    lo, hi = np.nanpercentile(x[ok], [1, 99])
    if hi <= lo:
        out[ok] = 0.0
    else:
        out[ok] = np.clip((x[ok] - lo) / (hi - lo), 0, 1)
    return out


def read_csv_any(path: str) -> pd.DataFrame:
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def build_gene_index(adata) -> Dict[str, int]:
    idx = {}

    def add(name, i):
        s = clean_str(name)
        if s == "":
            return
        k = s.lower()
        if k not in idx:
            idx[k] = i

    for i, g in enumerate(adata.var_names.astype(str)):
        add(g, i)

    for col in ["gene_symbol", "gene_symbols", "symbol", "gene_name", "feature_name", "name"]:
        if col in adata.var.columns:
            for i, g in enumerate(adata.var[col].astype(str).values):
                add(g, i)

    return idx


def extract_gene(adata, gene: str, gene_index: Dict[str, int]) -> Tuple[np.ndarray, Dict]:
    audit = {"gene": gene, "found": False, "matched_index": None, "var_name": None}
    key = gene.lower()

    if key not in gene_index:
        return np.full(adata.n_obs, np.nan), audit

    j = gene_index[key]
    X = adata.X[:, j]

    if sparse is not None and sparse.issparse(X):
        arr = np.asarray(X.toarray()).ravel()
    else:
        arr = np.asarray(X).ravel()

    arr = arr.astype(float)
    audit.update({"found": True, "matched_index": int(j), "var_name": str(adata.var_names[j])})
    return arr, audit


def compute_candidate_scores(adata) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gene_index = build_gene_index(adata)
    score_df = pd.DataFrame({"_adata_id_": adata.obs_names.astype(str)})
    audit_rows = []

    for cand, genes in CANDIDATE_FEATURES.items():
        zs = []
        for g in genes:
            vec, aud = extract_gene(adata, g, gene_index)
            aud["candidate"] = cand
            audit_rows.append(aud)
            if np.isfinite(vec).sum() > 0:
                zs.append(robust_z(vec))

        if len(zs) == 0:
            score = np.full(adata.n_obs, np.nan)
        else:
            score = np.nanmean(np.vstack(zs), axis=0)

        score_df[f"{cand}__feature_score"] = score
        score_df[f"{cand}__feature_score01"] = minmax01(score)

    return score_df, pd.DataFrame(audit_rows)


def find_merge_key(adata, state: pd.DataFrame) -> Tuple[str, Optional[str]]:
    obs_names = pd.Index(adata.obs_names.astype(str))
    candidates = [
        "obs_name", "barcode", "spot", "spot_id", "cell_id", "id",
        "_obs_names_", "adata_obs_name", "index"
    ]

    best_col, best_overlap = None, -1

    for c in candidates:
        if c in state.columns:
            ov = state[c].astype(str).isin(obs_names).sum()
            if ov > best_overlap:
                best_col, best_overlap = c, ov

    if best_overlap <= 0:
        for c in state.columns:
            if state[c].dtype == object or str(state[c].dtype).startswith("string"):
                ov = state[c].astype(str).isin(obs_names).sum()
                if ov > best_overlap:
                    best_col, best_overlap = c, ov

    if best_col is not None and best_overlap > 0:
        return "id", best_col

    if len(state) == adata.n_obs:
        return "row_order", None

    raise ValueError("Cannot merge state_table to adata.obs_names")


def merge_state(adata, state_path: str) -> Tuple[pd.DataFrame, Dict]:
    state = read_csv_any(state_path)
    obs = adata.obs.copy()
    obs["_adata_id_"] = adata.obs_names.astype(str)

    mode, key = find_merge_key(adata, state)

    audit = {
        "state_table": state_path,
        "state_rows": int(len(state)),
        "merge_mode": mode,
        "merge_key": key,
    }

    if mode == "id":
        st = state.copy()
        st["_adata_id_"] = st[key].astype(str)
        out = obs.merge(st, on="_adata_id_", how="left", suffixes=("", "_state"))
        audit["merge_overlap"] = int(out["_adata_id_"].isin(st["_adata_id_"]).sum())
    else:
        out = pd.concat([obs.reset_index(drop=True), state.reset_index(drop=True)], axis=1)
        audit["merge_overlap"] = int(len(out))

    return out.reset_index(drop=True), audit


def candidate_from_text(x: str) -> Optional[str]:
    s = clean_str(x).lower()
    s = s.replace("_", "-").replace(" ", "-").replace("/", "-")

    if any(t in s for t in ["ccl2", "ccr2", "ackr1"]):
        return "Ccl2/Ccr2-Ackr1 blockade"
    if any(t in s for t in ["spp1", "cd44"]):
        return "Spp1-Cd44 blockade"
    if any(t in s for t in ["vegfa", "flt1", "kdr"]):
        return "Vegfa-Flt1/Kdr blockade"
    if "ferro" in s:
        return "Ferroptosis down"
    if "repair" in s or "ecm" in s:
        return "Repair-ECM up"
    return None


def find_id_col(df: pd.DataFrame, meta_ids: set) -> Optional[str]:
    candidates = [
        "_adata_id_", "obs_name", "barcode", "spot", "spot_id", "cell_id",
        "adata_obs_name", "id", "index"
    ]

    best_col, best_overlap = None, -1
    for c in candidates:
        if c in df.columns:
            ov = df[c].astype(str).isin(meta_ids).sum()
            if ov > best_overlap:
                best_col, best_overlap = c, ov

    if best_overlap <= 0:
        for c in df.columns:
            if df[c].dtype == object or str(df[c].dtype).startswith("string"):
                ov = df[c].astype(str).isin(meta_ids).sum()
                if ov > best_overlap:
                    best_col, best_overlap = c, ov

    if best_col is not None and best_overlap > 0:
        return best_col
    return None


def find_candidate_col(df: pd.DataFrame) -> Optional[str]:
    keys = ["candidate", "perturb", "intervention", "blockade", "target", "ko", "axis"]
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in keys):
            return c
    return None


def col_contains(df: pd.DataFrame, include: List[str], exclude: List[str] = None) -> Optional[str]:
    exclude = exclude or []
    for c in df.columns:
        lc = c.lower()
        if all(k in lc for k in include) and not any(e in lc for e in exclude):
            return c
    return None


def standardize_response_table(path: str, meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    df = read_csv_any(path)
    meta_ids = set(meta["_adata_id_"].astype(str).values)

    id_col = find_id_col(df, meta_ids)
    cand_col = find_candidate_col(df)

    audit = {
        "path": path,
        "n_rows_raw": int(len(df)),
        "columns": df.columns.tolist(),
        "id_col": id_col,
        "candidate_col": cand_col,
    }

    if id_col is None:
        if len(df) == len(meta):
            df = df.copy()
            df["_adata_id_"] = meta["_adata_id_"].astype(str).values
            id_col = "_adata_id_"
            audit["id_col"] = id_col
            audit["id_mode"] = "row_order"
        else:
            raise ValueError(f"No usable obs/cell/spot id column in {path}")

    out = pd.DataFrame()
    out["_adata_id_"] = df[id_col].astype(str)

    if cand_col is not None:
        out["candidate_raw"] = df[cand_col].astype(str)
        out["candidate"] = out["candidate_raw"].map(candidate_from_text)
    else:
        inferred = candidate_from_text(os.path.basename(path))
        if inferred is None:
            inferred = candidate_from_text(path)
        out["candidate_raw"] = os.path.basename(path)
        out["candidate"] = inferred

    # Core reduction
    direct_core_reduction = (
        col_contains(df, ["core", "reduction"]) or
        col_contains(df, ["reduction", "core"]) or
        col_contains(df, ["core_reduction"])
    )

    delta_core = (
        col_contains(df, ["delta", "core"]) or
        col_contains(df, ["dcore"]) or
        col_contains(df, ["core", "delta"]) or
        col_contains(df, ["shift", "core"])
    )

    if direct_core_reduction is not None:
        out["core_reduction"] = pd.to_numeric(df[direct_core_reduction], errors="coerce")
        out["delta_core"] = -out["core_reduction"]
        audit["core_reduction_source"] = direct_core_reduction
    elif delta_core is not None:
        out["delta_core"] = pd.to_numeric(df[delta_core], errors="coerce")
        out["core_reduction"] = np.maximum(-out["delta_core"], 0)
        audit["delta_core_source"] = delta_core
    else:
        base_core = (
            col_contains(df, ["baseline", "core"]) or
            col_contains(df, ["base", "core"]) or
            col_contains(df, ["before", "core"])
        )
        pert_core = (
            col_contains(df, ["perturbed", "core"]) or
            col_contains(df, ["after", "core"]) or
            col_contains(df, ["counter", "core"])
        )
        if base_core is not None and pert_core is not None:
            base = pd.to_numeric(df[base_core], errors="coerce")
            pert = pd.to_numeric(df[pert_core], errors="coerce")
            out["delta_core"] = pert - base
            out["core_reduction"] = np.maximum(base - pert, 0)
            audit["core_sources"] = [base_core, pert_core]
        else:
            out["delta_core"] = np.nan
            out["core_reduction"] = np.nan
            audit["core_reduction_source"] = None

    # Repair gain
    repair_gain = (
        col_contains(df, ["repair", "gain"]) or
        col_contains(df, ["gain", "repair"]) or
        col_contains(df, ["rescue"]) or
        col_contains(df, ["response", "priority"])
    )

    delta_repair = (
        col_contains(df, ["delta", "repair"]) or
        col_contains(df, ["repair", "delta"]) or
        col_contains(df, ["shift", "repair"])
    )

    if repair_gain is not None:
        out["repair_gain"] = pd.to_numeric(df[repair_gain], errors="coerce")
        out["delta_repair"] = out["repair_gain"]
        audit["repair_gain_source"] = repair_gain
    elif delta_repair is not None:
        out["delta_repair"] = pd.to_numeric(df[delta_repair], errors="coerce")
        out["repair_gain"] = np.maximum(out["delta_repair"], 0)
        audit["delta_repair_source"] = delta_repair
    else:
        base_repair = (
            col_contains(df, ["baseline", "repair"]) or
            col_contains(df, ["base", "repair"]) or
            col_contains(df, ["before", "repair"])
        )
        pert_repair = (
            col_contains(df, ["perturbed", "repair"]) or
            col_contains(df, ["after", "repair"]) or
            col_contains(df, ["counter", "repair"])
        )
        if base_repair is not None and pert_repair is not None:
            base = pd.to_numeric(df[base_repair], errors="coerce")
            pert = pd.to_numeric(df[pert_repair], errors="coerce")
            out["delta_repair"] = pert - base
            out["repair_gain"] = np.maximum(pert - base, 0)
            audit["repair_sources"] = [base_repair, pert_repair]
        else:
            out["delta_repair"] = np.nan
            out["repair_gain"] = np.nan
            audit["repair_gain_source"] = None

    out = out[out["candidate"].notna()].copy()

    usable = int(out["core_reduction"].notna().sum() + out["repair_gain"].notna().sum())
    audit["n_rows_standardized"] = int(len(out))
    audit["usable_response_values"] = usable
    audit["candidate_counts"] = out["candidate"].value_counts().to_dict()

    if len(out) == 0 or usable == 0:
        raise ValueError(f"No usable standardized response values in {path}")

    return out, audit


def score_candidate_csv(path: str) -> int:
    try:
        tmp = pd.read_csv(path, nrows=5)
    except Exception:
        return -999

    cols = " ".join([c.lower() for c in tmp.columns])
    name = os.path.basename(path).lower()
    score = 0

    if any(k in cols for k in ["obs", "barcode", "spot", "cell", "adata"]):
        score += 4
    if any(k in cols for k in ["candidate", "perturb", "intervention", "blockade", "target"]):
        score += 4
    if any(k in cols for k in ["delta", "reduction", "gain", "rescue", "response"]):
        score += 8
    if "core" in cols:
        score += 3
    if "repair" in cols:
        score += 3
    if any(k in name for k in ["per_cell", "per-cell", "per_spot", "counterfactual", "response", "perturb"]):
        score += 4
    return score


def discover_response_tables(search_dirs: List[str]) -> List[str]:
    files = []
    for d in search_dirs:
        if d and os.path.exists(d):
            files.extend(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))

    scored = [(score_candidate_csv(f), f) for f in files]
    scored = sorted(scored, key=lambda x: x[0], reverse=True)
    return [f for s, f in scored if s >= 8]


def load_response(
    meta: pd.DataFrame,
    response_table: str,
    response_dir: str,
    extra_search_root: str,
) -> Tuple[Optional[pd.DataFrame], Dict]:
    audits = []

    candidate_paths = []
    if response_table:
        candidate_paths.append(response_table)

    search_dirs = []
    if response_dir:
        search_dirs.append(response_dir)
    if extra_search_root:
        search_dirs.append(extra_search_root)

    candidate_paths.extend(discover_response_tables(search_dirs))

    # remove duplicates
    candidate_paths = list(dict.fromkeys(candidate_paths))

    response_frames = []
    for p in candidate_paths:
        try:
            std, aud = standardize_response_table(p, meta)
            audits.append(aud)
            response_frames.append(std)
        except Exception as e:
            audits.append({"path": p, "error": repr(e)})

    if len(response_frames) == 0:
        return None, {"candidate_paths_checked": candidate_paths[:50], "audits": audits[:50]}

    response = pd.concat(response_frames, axis=0, ignore_index=True)
    response = response.drop_duplicates(subset=["_adata_id_", "candidate"], keep="first")

    return response, {
        "candidate_paths_checked": candidate_paths[:50],
        "audits": audits[:50],
        "n_rows_response": int(len(response)),
        "candidate_counts": response["candidate"].value_counts().to_dict(),
    }


def infer_col(meta: pd.DataFrame, include: List[str]) -> Optional[str]:
    for c in meta.columns:
        lc = c.lower()
        if all(k in lc for k in include):
            return c
    return None


def fallback_response(meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    core_col = (
        infer_col(meta, ["core", "probability"]) or
        infer_col(meta, ["core", "prob"]) or
        infer_col(meta, ["core"])
    )
    repair_col = (
        infer_col(meta, ["repair", "score"]) or
        infer_col(meta, ["repair"])
    )

    if core_col is None:
        raise ValueError("Fallback mode requested but no core probability column found.")

    frames = []
    for cand in CANDIDATE_FEATURES:
        tmp = pd.DataFrame({
            "_adata_id_": meta["_adata_id_"].astype(str).values,
            "candidate": cand,
            "delta_core": np.nan,
            "core_reduction": pd.to_numeric(meta[core_col], errors="coerce"),
            "delta_repair": np.nan,
            "repair_gain": pd.to_numeric(meta[repair_col], errors="coerce") if repair_col else np.nan,
        })
        frames.append(tmp)

    return pd.concat(frames, axis=0, ignore_index=True), {
        "fallback_mode": True,
        "core_col_used_as_y": core_col,
        "repair_col_used_as_y": repair_col,
        "note": "Fallback uses baseline state/repair metrics; not perturbation response."
    }


def safe_spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return np.nan
    if np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def add_linear_trend(ax, x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return
    try:
        coef = np.polyfit(x[ok], y[ok], 1)
        xs = np.linspace(np.nanpercentile(x[ok], 2), np.nanpercentile(x[ok], 98), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color="white", lw=1.6, alpha=0.9)
    except Exception:
        return


def plot_scatter_grid(df: pd.DataFrame, outdir: str, label_col: str, fallback: bool) -> Tuple[Dict, pd.DataFrame]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "figure.dpi": 180,
        "savefig.dpi": 450,
    })

    candidates = list(CANDIDATE_FEATURES.keys())
    metrics = ["core_reduction", "repair_gain"]
    titles = ["Core reduction" if not fallback else "Baseline core score",
              "Repair gain" if not fallback else "Baseline repair score"]

    fig, axes = plt.subplots(
        nrows=len(candidates),
        ncols=2,
        figsize=(11.2, 9.8),
        sharex=False,
        sharey=False
    )
    fig.patch.set_facecolor("#070b10")

    summary = []

    for i, cand in enumerate(candidates):
        sub = df[df["candidate"] == cand].copy()
        xcol = f"{cand}__feature_score"

        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            ax.set_facecolor("#070b10")

            if sub.empty or xcol not in sub.columns or metric not in sub.columns:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        color="white", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x = pd.to_numeric(sub[xcol], errors="coerce").values
            y = pd.to_numeric(sub[metric], errors="coerce").values

            if label_col in sub.columns:
                labels = sub[label_col].astype(str).map(normalize_label).values
                colors = [STATE_COLORS.get(v, "#9e9e9e") for v in labels]
            else:
                colors = "#9e9e9e"

            ax.scatter(x, y, s=5, c=colors, alpha=0.35, edgecolor="none", rasterized=True)
            add_linear_trend(ax, x, y)

            rho = safe_spearman(x, y)
            n_ok = int((np.isfinite(x) & np.isfinite(y)).sum())

            ax.text(
                0.03, 0.94,
                f"ρ={rho:.2f}, n={n_ok}" if np.isfinite(rho) else f"ρ=NA, n={n_ok}",
                ha="left", va="top", transform=ax.transAxes,
                color="white", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="#111820", ec="none", alpha=0.75)
            )

            if i == 0:
                ax.set_title(titles[j], color="white", fontweight="bold", fontsize=12)

            if j == 0:
                ax.set_ylabel(cand, color="white", fontweight="bold")
            ax.set_xlabel("Candidate feature score", color="white")

            ax.grid(True, color="0.18", lw=0.5)
            ax.tick_params(colors="0.86")
            for sp in ax.spines.values():
                sp.set_color("0.35")

            summary.append({
                "candidate": cand,
                "metric": metric,
                "spearman_rho": rho,
                "n": n_ok,
                "fallback_mode": bool(fallback),
            })

    title = "Surrogate feature-response audit"
    if fallback:
        title += " | baseline fallback"

    fig.suptitle(title, fontsize=19, color="white", fontweight="bold", y=0.985)

    if fallback:
        note = ("Fallback mode: y-axis uses baseline state/repair metrics because no per-cell perturbation-response table was found. "
                "Do not interpret this as virtual perturbation response.")
    else:
        note = ("Candidate feature score versus Step72/Step78-derived virtual response metrics. "
                "This is a surrogate computational audit, not SHAP and not wet-lab KO/blockade evidence.")

    fig.text(0.5, 0.025, note, ha="center", va="center", color="0.86", fontsize=8.5)

    fig.subplots_adjust(left=0.23, right=0.97, top=0.94, bottom=0.07, hspace=0.55, wspace=0.30)

    paths = {}
    prefix = "Fig_Step81B_SurrogateFeatureResponseAudit"
    for ext in ["png", "pdf", "svg"]:
        p = os.path.join(outdir, f"{prefix}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[ext] = p

    plt.close(fig)
    return paths, pd.DataFrame(summary)


def plot_heatmap(summary: pd.DataFrame, outdir: str) -> Dict:
    if summary.empty:
        return {}

    mat = summary.pivot(index="candidate", columns="metric", values="spearman_rho")
    mat = mat.reindex(index=list(CANDIDATE_FEATURES.keys()))

    fig, ax = plt.subplots(figsize=(5.8, 4.3))
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
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("0.35")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman ρ", color="white")
    cb.ax.tick_params(colors="white")

    fig.tight_layout()

    paths = {}
    prefix = "Fig_Step81B_FeatureResponseCorrelationHeatmap"
    for ext in ["png", "pdf", "svg"]:
        p = os.path.join(outdir, f"{prefix}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=fig.get_facecolor())
        paths[ext] = p

    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--response_table", default="")
    parser.add_argument("--response_dir", default="")
    parser.add_argument("--extra_search_root", default="")

    parser.add_argument("--label_col", default="state_group")
    parser.add_argument("--time_col", default="timepoint")
    parser.add_argument("--allow_fallback", action="store_true")

    args = parser.parse_args()
    mkdir(args.outdir)

    print("=" * 100)
    print("Step81B | Robust surrogate feature-response audit")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"response_table={args.response_table}")
    print(f"response_dir={args.response_dir}")
    print(f"extra_search_root={args.extra_search_root}")
    print(f"outdir={args.outdir}")

    adata = ad.read_h5ad(args.h5ad)
    meta, merge_audit = merge_state(adata, args.state_table)

    if args.label_col not in meta.columns:
        raise ValueError(f"{args.label_col} not found in merged metadata")
    if args.time_col not in meta.columns:
        raise ValueError(f"{args.time_col} not found in merged metadata")

    meta[args.label_col] = meta[args.label_col].map(normalize_label)
    meta[args.time_col] = meta[args.time_col].astype(str)

    score_df, gene_audit = compute_candidate_scores(adata)
    meta2 = meta.merge(score_df, on="_adata_id_", how="left")

    fallback = False
    response, response_audit = load_response(
        meta2,
        args.response_table,
        args.response_dir,
        args.extra_search_root,
    )

    if response is None:
        if not args.allow_fallback:
            raise RuntimeError(
                "No usable per-cell/per-spot response table found. "
                "Use --response_table if you know the exact file, or rerun with --allow_fallback "
                "for baseline association QC only."
            )
        response, fb_audit = fallback_response(meta2)
        response_audit["fallback"] = fb_audit
        fallback = True

    plot_df = response.merge(meta2, on="_adata_id_", how="left", suffixes=("", "_meta"))

    # Save merged table
    merged_path = os.path.join(args.outdir, "step81b_merged_feature_response_table.csv")
    plot_df.to_csv(merged_path, index=False)

    # Plot
    fig_paths, summary = plot_scatter_grid(plot_df, args.outdir, args.label_col, fallback)
    summary_path = os.path.join(args.outdir, "step81b_feature_response_spearman_summary.csv")
    summary.to_csv(summary_path, index=False)

    heatmap_paths = plot_heatmap(summary, args.outdir)

    gene_audit_path = os.path.join(args.outdir, "step81b_gene_matching_audit.csv")
    gene_audit.to_csv(gene_audit_path, index=False)

    report = {
        "status": "ok",
        "fallback_mode": bool(fallback),
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "merge_audit": merge_audit,
        "response_audit": response_audit,
        "missing_genes": gene_audit.loc[~gene_audit["found"], "gene"].drop_duplicates().tolist(),
        "outputs": {
            "scatter": fig_paths,
            "heatmap": heatmap_paths,
            "summary_csv": summary_path,
            "merged_table": merged_path,
            "gene_audit_csv": gene_audit_path,
        },
        "interpretation_note": (
            "This is a surrogate feature-response audit. It is not SHAP unless true SHAP values are supplied, "
            "and it is not wet-lab KO/blockade validation. If fallback_mode=true, it is only a baseline "
            "feature-state association QC."
        )
    }

    report_path = os.path.join(args.outdir, "step81b_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()