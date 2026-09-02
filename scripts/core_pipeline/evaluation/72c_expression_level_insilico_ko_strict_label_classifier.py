#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step72C | Strict label-driven expression-level in silico KO / blockade

Why Step72C?
------------
Step72B failed scientifically because the chosen state probability table was nearly
uniform:
    core/peri/remote ≈ 1/3, 1/3, 1/3
and hard argmax labels were all "core".
A regressor trained on such targets cannot learn meaningful state boundaries.

Step72C fixes this by:
1. Auditing state probabilities and rejecting degenerate probability-only labels.
2. Searching real multiclass state/region label columns.
3. Training an expression-to-state classifier only if >=2 or preferably 3 usable
   core/peri/remote classes are present.
4. Performing expression-level in silico KO/blockade-like perturbation.
5. Reclassifying state probabilities after perturbation.
6. Refusing to overclaim if effect size is near-zero.

Interpretation:
This is expression-level computational counterfactual analysis.
It is not wet-lab KO/blockade, not drug validation, and not observed cell-fate tracking.
"""

from pathlib import Path
import argparse
import json
import math
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.colors import TwoSlopeNorm

import anndata as ad
import scipy.sparse as sp

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


STATE_KEYS = ["core", "peri", "remote"]

STATE_DISPLAY = {
    "core": "lesion-core-like",
    "peri": "peri-infarct",
    "remote": "remote-like",
}

STATE_COLORS = {
    "core": "#d62728",
    "peri": "#f2b705",
    "remote": "#4aa8f0",
}

DEFAULT_CANDIDATES = [
    "Ccl2_Ccr2_Ackr1_blockade",
    "Spp1_Cd44_blockade",
    "Vegfa_Flt1_Kdr_blockade",
    "ferroptosis_down",
    "repair_ECM_up",
]

DEFAULT_GENE_SETS = {
    "Ccl2_Ccr2_Ackr1_blockade": {
        "display_name": "Ccl2/Ccr2-Ackr1 blockade",
        "direction": "down",
        "genes": [
            "Ccl2", "Ccr2", "Ackr1", "Ccl7", "Ccl12",
            "Cxcl1", "Cxcl2", "Cxcl10", "Il1b", "Tnf", "Ly6c2"
        ],
        "interpretation": "target-axis downscale; blockade-like computational counterfactual"
    },
    "Spp1_Cd44_blockade": {
        "display_name": "Spp1-Cd44 blockade",
        "direction": "down",
        "genes": [
            "Spp1", "Cd44", "Itgav", "Itgb1", "Itgb5",
            "Lgals3", "Trem2", "Apoe", "Cst7", "Lpl"
        ],
        "interpretation": "target-axis downscale; blockade-like computational counterfactual"
    },
    "Vegfa_Flt1_Kdr_blockade": {
        "display_name": "Vegfa-Flt1/Kdr blockade",
        "direction": "down",
        "genes": [
            "Vegfa", "Flt1", "Kdr", "Pecam1", "Vwf",
            "Klf2", "Klf4", "Angpt2", "Tek", "Adm", "Nos3"
        ],
        "interpretation": "target-axis downscale; blockade-like computational counterfactual"
    },
    "ferroptosis_down": {
        "display_name": "Ferroptosis down-modulation",
        "direction": "down",
        "genes": [
            "Acsl4", "Ptgs2", "Hmox1", "Tfrc", "Slc7a11",
            "Gpx4", "Fth1", "Ftl1", "Ncoa4", "Alox15", "Aifm2", "Nfe2l2"
        ],
        "interpretation": "module downscale; not single-gene KO"
    },
    "repair_ECM_up": {
        "display_name": "Repair-ECM promotion",
        "direction": "up",
        "genes": [
            "Fn1", "Col1a1", "Col1a2", "Col3a1", "Sparc",
            "Timp1", "Mmp9", "Mmp14", "Vim", "Postn", "Lgals3", "Itgb1"
        ],
        "interpretation": "module upshift; not single-gene overexpression experiment"
    },
}


def log(x):
    print(x, flush=True)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path, required=False):
    if not path:
        if required:
            raise FileNotFoundError("Empty path")
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, low_memory=False)


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return ""


def clean_gene(g):
    return re.sub(r"[^A-Za-z0-9_.-]", "", str(g)).strip()


def title_gene(g):
    g = clean_gene(g)
    if not g:
        return g
    return g[0].upper() + g[1:].lower()


def normalize_prob(arr):
    arr = np.asarray(arr, dtype=float)
    arr[~np.isfinite(arr)] = 0
    arr[arr < 0] = 0
    s = arr.sum(axis=1, keepdims=True)
    s[s <= 0] = 1
    return arr / s


def sparse_to_dense(X):
    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


def gene_symbol_map(adata):
    mapping = {}

    def add(sym, var_name):
        if sym is None:
            return
        s = str(sym).strip()
        if not s:
            return
        mapping.setdefault(s.upper(), var_name)

    for v in adata.var_names.astype(str):
        add(v, v)

    for col in ["gene", "genes", "gene_name", "gene_symbol", "symbol", "feature_name", "features"]:
        if col in adata.var.columns:
            for var_name, sym in zip(adata.var_names.astype(str), adata.var[col].astype(str)):
                add(sym, var_name)

    return mapping


def match_genes(genes, gmap):
    matched, missing = [], []
    for g in genes:
        g0 = clean_gene(g)
        if not g0:
            continue
        probes = [g0, g0.upper(), title_gene(g0), g0.capitalize()]
        hit = None
        for p in probes:
            if p.upper() in gmap:
                hit = gmap[p.upper()]
                break
        if hit is None:
            missing.append(g)
        else:
            if hit not in matched:
                matched.append(hit)
    return matched, missing


def detect_probability_cols(df):
    core = first_existing(df, [
        "core_probability", "prob_core", "prob_lesion_core",
        "baseline_core_probability", "lesion_core_probability",
        "lesion_core_like_probability"
    ])
    peri = first_existing(df, [
        "peri_probability", "prob_peri", "prob_peri_infarct",
        "baseline_peri_probability", "peri_infarct_probability"
    ])
    remote = first_existing(df, [
        "remote_probability", "prob_remote", "prob_remote_like",
        "baseline_remote_probability", "remote_like_probability"
    ])
    if all([core, peri, remote]):
        return core, peri, remote
    return "", "", ""


def probability_audit(df, min_prob_sd):
    core, peri, remote = detect_probability_cols(df)
    if not all([core, peri, remote]):
        return {
            "available": False,
            "reason": "no_core_peri_remote_probability_columns"
        }, None

    probs = df[[core, peri, remote]].apply(pd.to_numeric, errors="coerce")
    ok = probs.notna().all(axis=1)
    probs = probs.loc[ok].to_numpy(dtype=float)
    probs = normalize_prob(probs)

    argmax = np.array(STATE_KEYS)[np.argmax(probs, axis=1)]
    counts = pd.Series(argmax).value_counts().to_dict()
    sd = probs.std(axis=0)
    mean = probs.mean(axis=0)

    is_uniform_like = (
        float(np.max(sd)) < min_prob_sd or
        len(counts) < 2 or
        max(counts.values()) / max(sum(counts.values()), 1) > 0.98
    )

    audit = {
        "available": True,
        "columns": {"core": core, "peri": peri, "remote": remote},
        "n_non_na": int(len(probs)),
        "mean": {"core": float(mean[0]), "peri": float(mean[1]), "remote": float(mean[2])},
        "sd": {"core": float(sd[0]), "peri": float(sd[1]), "remote": float(sd[2])},
        "argmax_counts": counts,
        "min_prob_sd_threshold": float(min_prob_sd),
        "is_degenerate_or_uniform_like": bool(is_uniform_like),
    }
    return audit, probs


def map_to_state(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if not s or s in ["nan", "none", "na", "null", "unknown"]:
        return None

    s2 = re.sub(r"[^a-z0-9]+", "_", s)

    # Core / lesion
    core_tokens = [
        "core", "lesion_core", "lesion", "infarct_core", "ischemic_core",
        "necrotic", "damaged_core", "injury_core", "core_like"
    ]
    # Peri / penumbra
    peri_tokens = [
        "peri", "peri_infarct", "periinfarct", "penumbra", "border",
        "rim", "transition", "peri_like", "peri_infarct_like"
    ]
    # Remote / normal-like
    remote_tokens = [
        "remote", "remote_like", "normal", "control", "sham", "contralateral",
        "intact", "non_lesion", "nonlesion", "healthy_like", "remote_core",
        "distant"
    ]

    for t in core_tokens:
        if t in s2:
            return "core"
    for t in peri_tokens:
        if t in s2:
            return "peri"
    for t in remote_tokens:
        if t in s2:
            return "remote"

    return None


def candidate_label_columns(df, min_class_n=20):
    rows = []
    for col in df.columns:
        if col.startswith("_"):
            continue

        # skip numeric probability-like columns
        if pd.api.types.is_numeric_dtype(df[col]):
            continue

        mapped = df[col].map(map_to_state)
        counts = mapped.dropna().value_counts().to_dict()
        n_classes = len(counts)
        min_n = min(counts.values()) if counts else 0

        if n_classes >= 2 and min_n >= min_class_n:
            rows.append({
                "column": col,
                "n_mapped": int(mapped.notna().sum()),
                "n_classes": int(n_classes),
                "min_class_n": int(min_n),
                "counts": counts,
                "score": int(n_classes) * 100000 + int(mapped.notna().sum()) + int(min_n)
            })

    rows = sorted(rows, key=lambda x: x["score"], reverse=True)
    return rows


def merge_state_and_obs(adata, state_df, obs_names, sampled_indices=None):
    obs_meta = adata.obs.copy()
    if sampled_indices is not None:
        obs_meta = obs_meta.iloc[sampled_indices].copy()
    obs_meta = obs_meta.copy()
    obs_meta.insert(0, "obs_name", obs_names)

    # Prefix obs columns to avoid collision.
    rename = {}
    for c in obs_meta.columns:
        if c != "obs_name":
            rename[c] = f"obs__{c}"
    obs_meta = obs_meta.rename(columns=rename)

    if state_df.empty:
        merged = obs_meta.copy()
        merged["_adata_row"] = np.arange(len(obs_names))
        return merged, {
            "state_table_used": False,
            "match_mode": "adata_obs_only",
            "name_match_fraction": None
        }

    state_df = state_df.copy()
    obs_col = first_existing(state_df, ["obs_name", "cell_id", "barcode", "spot_id"])
    if not obs_col:
        obs_col = state_df.columns[0]
    state_df[obs_col] = state_df[obs_col].astype(str)

    left = pd.DataFrame({
        "obs_name": obs_names,
        "_adata_row": np.arange(len(obs_names))
    })

    merged = left.merge(state_df, left_on="obs_name", right_on=obs_col, how="left")
    name_match_fraction = float(merged[obs_col].notna().mean())

    if name_match_fraction < 0.2:
        if sampled_indices is not None:
            if len(state_df) >= int(np.max(sampled_indices)) + 1:
                state_sub = state_df.iloc[sampled_indices].reset_index(drop=True).copy()
                state_sub.insert(0, "obs_name", obs_names)
                state_sub["_adata_row"] = np.arange(len(obs_names))
                merged = state_sub
                match_mode = "order_fallback_using_sampled_indices"
            else:
                merged = left.copy()
                match_mode = "failed_state_table_alignment_using_adata_obs_only"
        elif len(state_df) == len(obs_names):
            state_sub = state_df.reset_index(drop=True).copy()
            state_sub.insert(0, "obs_name", obs_names)
            state_sub["_adata_row"] = np.arange(len(obs_names))
            merged = state_sub
            match_mode = "order_fallback_equal_length"
        else:
            merged = left.copy()
            match_mode = "failed_state_table_alignment_using_adata_obs_only"
    else:
        match_mode = "obs_name_merge"

    merged = merged.merge(obs_meta, on="obs_name", how="left")

    audit = {
        "state_table_used": True,
        "state_table_obs_col": obs_col,
        "match_mode": match_mode,
        "name_match_fraction": name_match_fraction,
        "n_merged": int(len(merged)),
    }
    return merged, audit


def choose_labels(merged, label_col, min_class_n, min_prob_sd, allow_probability_labels=False):
    prob_audit, probs = probability_audit(merged, min_prob_sd=min_prob_sd)

    if label_col:
        if label_col not in merged.columns:
            raise RuntimeError(f"--label_col '{label_col}' not found. Available columns include: {list(merged.columns)[:80]}")
        y = merged[label_col].map(map_to_state)
        counts = y.dropna().value_counts().to_dict()
        if len(counts) < 2 or min(counts.values()) < min_class_n:
            raise RuntimeError(
                f"Provided label_col={label_col} is not usable. Mapped counts: {counts}. "
                f"Need >=2 classes and min_class_n >= {min_class_n}."
            )
        selected = {
            "mode": "user_label_col",
            "label_col": label_col,
            "counts": counts,
            "probability_audit": prob_audit,
        }
        ok = y.notna()
        return y.loc[ok].astype(str).values, merged.loc[ok].copy(), selected

    label_candidates = candidate_label_columns(merged, min_class_n=min_class_n)

    if label_candidates:
        best = label_candidates[0]
        col = best["column"]
        y = merged[col].map(map_to_state)
        ok = y.notna()
        selected = {
            "mode": "auto_label_col",
            "label_col": col,
            "counts": best["counts"],
            "all_candidate_label_cols": label_candidates[:20],
            "probability_audit": prob_audit,
        }
        return y.loc[ok].astype(str).values, merged.loc[ok].copy(), selected

    if allow_probability_labels and prob_audit.get("available") and not prob_audit.get("is_degenerate_or_uniform_like"):
        core, peri, remote = detect_probability_cols(merged)
        probs_df = merged[[core, peri, remote]].apply(pd.to_numeric, errors="coerce")
        ok = probs_df.notna().all(axis=1)
        p = normalize_prob(probs_df.loc[ok].to_numpy(dtype=float))
        y = np.array(STATE_KEYS)[np.argmax(p, axis=1)]
        counts = pd.Series(y).value_counts().to_dict()
        if len(counts) >= 2 and min(counts.values()) >= min_class_n:
            selected = {
                "mode": "probability_argmax",
                "label_col": "",
                "counts": counts,
                "probability_audit": prob_audit,
            }
            return y, merged.loc[ok].copy(), selected

    selected = {
        "mode": "failed_no_usable_multiclass_label",
        "probability_audit": prob_audit,
        "all_candidate_label_cols": label_candidates[:20],
        "message": (
            "No usable core/peri/remote label column found, and probability labels are degenerate or disabled. "
            "Provide --label_col using a real region/state label column, or use a different state table."
        )
    }
    raise RuntimeError(json.dumps(selected, indent=2, ensure_ascii=False))


def read_step65_genes(path, max_genes=1000):
    df = read_table(path)
    if df.empty:
        return []
    gene_col = first_existing(df, ["gene", "genes", "gene_symbol", "symbol", "dynamic_gene"])
    if not gene_col:
        for c in df.columns:
            if "gene" in c.lower():
                gene_col = c
                break
    if not gene_col:
        return []

    genes = []
    for x in df[gene_col].astype(str):
        for g in re.split(r"[;,| \t]+", x):
            g = clean_gene(g)
            if g and g.lower() not in ["nan", "none"]:
                genes.append(g)
    return list(dict.fromkeys(genes))[:max_genes]


def build_feature_genes(specs, step65_genes, gmap, max_feature_genes):
    candidate_raw = []
    for spec in specs.values():
        candidate_raw.extend(spec["genes"])
    candidate_raw = list(dict.fromkeys([clean_gene(g) for g in candidate_raw if clean_gene(g)]))

    dynamic_raw = list(dict.fromkeys([clean_gene(g) for g in step65_genes if clean_gene(g)]))

    cand_matched, cand_missing = match_genes(candidate_raw, gmap)
    dyn_matched, dyn_missing = match_genes(dynamic_raw, gmap)

    # Always keep candidate genes first; fill with dynamic genes.
    feature_genes = list(dict.fromkeys(cand_matched + dyn_matched))
    if len(feature_genes) > max_feature_genes:
        keep = []
        for g in cand_matched:
            if g not in keep:
                keep.append(g)
        for g in dyn_matched:
            if g not in keep:
                keep.append(g)
            if len(keep) >= max_feature_genes:
                break
        feature_genes = keep

    audit = {
        "n_candidate_raw": len(candidate_raw),
        "n_candidate_matched": len(cand_matched),
        "candidate_missing_first100": cand_missing[:100],
        "n_dynamic_raw": len(dynamic_raw),
        "n_dynamic_matched": len(dyn_matched),
        "dynamic_missing_first100": dyn_missing[:100],
        "n_feature_genes": len(feature_genes),
    }
    return feature_genes, audit


def load_expression(adata, feature_genes, max_cells=0, seed=20260601):
    n = adata.n_obs
    sampled_indices = None

    if max_cells and max_cells > 0 and n > max_cells:
        rng = np.random.default_rng(seed)
        sampled_indices = np.sort(rng.choice(np.arange(n), size=max_cells, replace=False))
        adata_sub = adata[sampled_indices, feature_genes]
        obs_names = adata.obs_names[sampled_indices].astype(str).tolist()
    else:
        adata_sub = adata[:, feature_genes]
        obs_names = adata.obs_names.astype(str).tolist()

    X = sparse_to_dense(adata_sub.X).astype(np.float32)

    q99 = np.nanquantile(X[:min(X.shape[0], 2000), :min(X.shape[1], 200)], 0.99)
    log1p_applied = False
    if np.isfinite(q99) and q99 > 30:
        X = np.log1p(X)
        log1p_applied = True

    audit = {
        "n_obs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "sampled_indices_used": sampled_indices is not None,
        "auto_log1p_applied": bool(log1p_applied),
        "q99_for_log_check": float(q99) if np.isfinite(q99) else None,
    }
    return X, obs_names, sampled_indices, audit


def build_candidate_info(specs, gmap, feature_genes):
    var_to_idx = {g: i for i, g in enumerate(feature_genes)}
    rows = []
    info = {}

    for cid, spec in specs.items():
        matched_h5ad, missing = match_genes(spec["genes"], gmap)
        matched_feature = [g for g in matched_h5ad if g in var_to_idx]
        idx = [var_to_idx[g] for g in matched_feature]

        info[cid] = {
            "candidate_id": cid,
            "display_name": spec["display_name"],
            "direction": spec["direction"],
            "interpretation": spec["interpretation"],
            "input_genes": spec["genes"],
            "matched_h5ad_genes": matched_h5ad,
            "matched_feature_genes": matched_feature,
            "missing_genes": missing,
            "feature_indices": idx,
        }

        rows.append({
            "candidate_id": cid,
            "display_name": spec["display_name"],
            "direction": spec["direction"],
            "n_input_genes": len(spec["genes"]),
            "n_matched_h5ad_genes": len(matched_h5ad),
            "n_matched_feature_genes": len(matched_feature),
            "matched_feature_genes": ";".join(matched_feature),
            "missing_genes": ";".join(missing),
            "interpretation": spec["interpretation"],
            "is_usable": len(matched_feature) > 0,
        })

    return info, pd.DataFrame(rows)


def make_classifier(C=1.0):
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs",
            multi_class="auto",
            C=C,
            random_state=20260601,
        )
    )


def classifier_cv_audit(X, y, C=1.0):
    y = np.asarray(y)
    counts = pd.Series(y).value_counts().to_dict()
    min_count = min(counts.values()) if counts else 0
    n_classes = len(counts)

    if n_classes < 2:
        return {
            "status": "failed_one_class",
            "class_counts": counts
        }
    if min_count < 3:
        return {
            "status": "failed_min_class_too_small",
            "class_counts": counts
        }

    n_splits = min(5, min_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260601)
    clf = make_classifier(C=C)
    pred = cross_val_predict(clf, X, y, cv=cv, method="predict")

    labels = sorted(pd.Series(y).unique().tolist())
    cm = confusion_matrix(y, pred, labels=labels)

    return {
        "status": "ok",
        "model_type": "strict_multiclass_or_binary_expression_to_state_classifier",
        "n_splits": int(n_splits),
        "labels": labels,
        "class_counts": counts,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "confusion_matrix": cm.tolist(),
        "interpretation": (
            "Internal cross-validation of surrogate expression-to-state classifier. "
            "Use as counterfactual support only if balanced accuracy/macro-F1 are acceptable."
        ),
    }


def perturb_expression(X, idx, direction, ko_scale=0.0, up_shift_sd=1.0):
    Xp = X.copy()
    idx = list(idx)
    if not idx:
        return Xp

    idx = np.asarray(idx, dtype=int)

    if direction == "up":
        sd = np.nanstd(X[:, idx], axis=0)
        sd[~np.isfinite(sd)] = 0
        Xp[:, idx] = Xp[:, idx] + up_shift_sd * sd[None, :]
        cap = np.nanquantile(X[:, idx], 0.995, axis=0)
        cap[~np.isfinite(cap)] = np.nanmax(X[:, idx], axis=0)[~np.isfinite(cap)]
        Xp[:, idx] = np.minimum(Xp[:, idx], cap[None, :] + 0.5 * np.maximum(sd[None, :], 1e-6))
    else:
        Xp[:, idx] = Xp[:, idx] * ko_scale

    return Xp


def predict_state_probabilities(clf, X):
    probs_raw = clf.predict_proba(X)
    classes = list(clf.classes_)

    out = np.zeros((X.shape[0], 3), dtype=float)
    for i, key in enumerate(STATE_KEYS):
        if key in classes:
            out[:, i] = probs_raw[:, classes.index(key)]
        else:
            out[:, i] = 0.0

    # If classifier lacks one class, preserve probability mass across available states.
    out = normalize_prob(out)
    return out


def summarize_cell_prob(cell_df):
    rows = []
    for cid, sub in cell_df.groupby("candidate_id", sort=False):
        r = {
            "candidate_id": cid,
            "display_name": sub["display_name"].iloc[0],
            "direction": sub["direction"].iloc[0],
            "n_cells": int(len(sub)),
            "n_perturbed_genes": int(sub["n_perturbed_genes"].iloc[0]),
            "perturbed_feature_genes": sub["perturbed_feature_genes"].iloc[0],
        }

        for state in STATE_KEYS:
            r[f"baseline_{state}_mean"] = float(sub[f"baseline_{state}_probability"].mean())
            r[f"perturbed_{state}_mean"] = float(sub[f"perturbed_{state}_probability"].mean())
            r[f"delta_{state}_probability"] = float(sub[f"delta_{state}_probability"].mean())

        r["state_shift_priority_score"] = (
            -r["delta_core_probability"]
            + 0.5 * r["delta_peri_probability"]
            + 0.5 * r["delta_remote_probability"]
        )
        rows.append(r)

    return pd.DataFrame(rows)


def net_redistribution_flow(before, after):
    b = np.asarray(before, dtype=float)
    p = np.asarray(after, dtype=float)
    b = b / max(b.sum(), 1e-12)
    p = p / max(p.sum(), 1e-12)

    delta = p - b
    surplus = np.where(delta < 0, -delta, 0.0)
    deficit = np.where(delta > 0, delta, 0.0)

    M = np.zeros((3, 3), dtype=float)
    total_deficit = deficit.sum()

    if total_deficit > 1e-12:
        for i in range(3):
            if surplus[i] <= 1e-12:
                continue
            for j in range(3):
                if deficit[j] <= 1e-12:
                    continue
                M[i, j] = surplus[i] * deficit[j] / total_deficit
    return M


def make_flow_table(summary):
    rows = []
    for _, r in summary.iterrows():
        before = [r["baseline_core_mean"], r["baseline_peri_mean"], r["baseline_remote_mean"]]
        after = [r["perturbed_core_mean"], r["perturbed_peri_mean"], r["perturbed_remote_mean"]]
        M = net_redistribution_flow(before, after)

        for i, src in enumerate(STATE_KEYS):
            for j, dst in enumerate(STATE_KEYS):
                if M[i, j] > 1e-12:
                    rows.append({
                        "candidate_id": r["candidate_id"],
                        "display_name": r["display_name"],
                        "source_state": STATE_DISPLAY[src],
                        "target_state": STATE_DISPLAY[dst],
                        "source_key": src,
                        "target_key": dst,
                        "net_flow_weight": float(M[i, j]),
                    })
    return pd.DataFrame(rows)


def bary_to_xy(core, peri, remote):
    x = 0.5 * core + 0.0 * peri + 1.0 * remote
    y = (math.sqrt(3) / 2) * core
    return x, y


def draw_simplex(ax, summary, arrow_scale=12, clean=False):
    tri = np.array([
        bary_to_xy(1, 0, 0),
        bary_to_xy(0, 1, 0),
        bary_to_xy(0, 0, 1),
    ])
    ax.add_patch(Polygon(tri, closed=True, fill=False, edgecolor="black", linewidth=1.1))

    for t in [0.2, 0.4, 0.6, 0.8]:
        p1, p2 = bary_to_xy(t, 1 - t, 0), bary_to_xy(t, 0, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#e5e7eb", lw=0.6)
        p1, p2 = bary_to_xy(1 - t, t, 0), bary_to_xy(0, t, 1 - t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#e5e7eb", lw=0.6)
        p1, p2 = bary_to_xy(1 - t, 0, t), bary_to_xy(0, 1 - t, t)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#e5e7eb", lw=0.6)

    cmap = plt.cm.tab10

    for i, (_, r) in enumerate(summary.iterrows()):
        color = cmap(i % 10)

        b = np.array([r["baseline_core_mean"], r["baseline_peri_mean"], r["baseline_remote_mean"]])
        p = np.array([r["perturbed_core_mean"], r["perturbed_peri_mean"], r["perturbed_remote_mean"]])
        d = p - b

        p_vis = b + arrow_scale * d
        p_vis[p_vis < 0] = 0
        if p_vis.sum() <= 0:
            p_vis = p
        p_vis = p_vis / p_vis.sum()

        xb, yb = bary_to_xy(*b)
        xp, yp = bary_to_xy(*p_vis)

        ax.scatter(xb, yb, s=36, facecolor="white", edgecolor=color, lw=1.2, zorder=3)
        ax.scatter(xp, yp, s=68, marker="*", color=color, edgecolor="black", lw=0.5, zorder=4)
        ax.annotate("", xy=(xp, yp), xytext=(xb, yb),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

        if not clean:
            ax.text(0.02, 0.94 - i * 0.055,
                    f"{i+1}. {str(r['display_name'])[:28]}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=7.2, color=color)
            ax.text(xp + 0.010, yp + 0.010, str(i+1), fontsize=8, color=color, fontweight="bold")

    if not clean:
        ax.text(0.5, math.sqrt(3) / 2 + 0.055, "lesion-core-like",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=STATE_COLORS["core"])
        ax.text(-0.04, -0.04, "peri-infarct",
                ha="left", va="top", fontsize=9, fontweight="bold", color=STATE_COLORS["peri"])
        ax.text(1.04, -0.04, "remote-like",
                ha="right", va="top", fontsize=9, fontweight="bold", color=STATE_COLORS["remote"])
        ax.text(0.5, -0.12, f"Arrows magnified ×{arrow_scale}; centroids use true means",
                ha="center", va="top", fontsize=8, color="#475569")

    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.13, math.sqrt(3) / 2 + 0.13)
    ax.set_aspect("equal")
    ax.axis("off")


def ribbon(ax, x0, x1, y0a, y0b, y1a, y1b, color, alpha=0.50):
    verts = [
        (x0, y0a),
        (x0 + 0.35 * (x1 - x0), y0a),
        (x1 - 0.35 * (x1 - x0), y1a),
        (x1, y1a),
        (x1, y1b),
        (x1 - 0.35 * (x1 - x0), y1b),
        (x0 + 0.35 * (x1 - x0), y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def draw_net_flow(ax, summary, top_candidate, flow_scale=15, clean=False):
    r = summary[summary["candidate_id"].astype(str).eq(str(top_candidate))].iloc[0]

    b = np.array([r["baseline_core_mean"], r["baseline_peri_mean"], r["baseline_remote_mean"]])
    p = np.array([r["perturbed_core_mean"], r["perturbed_peri_mean"], r["perturbed_remote_mean"]])
    M = net_redistribution_flow(b, p)

    xL0, xL1 = 0.08, 0.19
    xR0, xR1 = 0.81, 0.92
    pad = 0.035
    usable = 1 - pad * 2

    def stack(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals / max(vals.sum(), 1e-12)
        pos = []
        y = 1.0
        for val in vals:
            h = val * usable
            pos.append((y - h, y))
            y -= h + pad
        return pos

    left_pos = stack(b)
    right_pos = stack(p)

    for i, state in enumerate(STATE_KEYS):
        col = STATE_COLORS[state]
        y0, y1 = left_pos[i]
        ax.add_patch(Rectangle((xL0, y0), xL1-xL0, y1-y0, facecolor=col, edgecolor="white", lw=0.7))
        y0, y1 = right_pos[i]
        ax.add_patch(Rectangle((xR0, y0), xR1-xR0, y1-y0, facecolor=col, edgecolor="white", lw=0.7))

    left_cursor = {i: left_pos[i][0] for i in range(3)}
    right_cursor = {j: right_pos[j][0] for j in range(3)}

    any_flow = False
    for i, src in enumerate(STATE_KEYS):
        for j, dst in enumerate(STATE_KEYS):
            w = M[i, j]
            if w <= 1e-12:
                continue
            any_flow = True
            h = min(w * flow_scale, 0.24)
            y0a, y0b = left_cursor[i], left_cursor[i] + h
            y1a, y1b = right_cursor[j], right_cursor[j] + h
            ribbon(ax, xL1, xR0, y0a, y0b, y1a, y1b, color=STATE_COLORS[src], alpha=0.50)

            if not clean:
                ax.text(0.50, (y0a + y0b + y1a + y1b) / 4,
                        f"{src}→{dst}: {w:+.3f}",
                        ha="center", va="center", fontsize=7.2,
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, boxstyle="round,pad=0.12"))
            left_cursor[i] += h + 0.003
            right_cursor[j] += h + 0.003

    if not clean:
        ax.text((xL0+xL1)/2, 1.045, "Baseline", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.text((xR0+xR1)/2, 1.045, "Perturbed", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for i, state in enumerate(STATE_KEYS):
            ax.text(xL0 - 0.025, np.mean(left_pos[i]), STATE_DISPLAY[state], ha="right", va="center", fontsize=8)
            ax.text(xR1 + 0.025, np.mean(right_pos[i]), STATE_DISPLAY[state], ha="left", va="center", fontsize=8)
        if any_flow:
            ax.text(0.5, -0.08,
                    f"Ribbons show net redistributed mass only; widths magnified ×{flow_scale}",
                    ha="center", va="top", fontsize=8, color="#475569")
        else:
            ax.text(0.5, 0.5, "No detectable net redistribution", ha="center", va="center",
                    fontsize=10, color="#475569")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.10, 1.08)
    ax.axis("off")


def draw_dumbbell(ax, summary, clean=False):
    rows = []
    for _, r in summary.iterrows():
        for state in STATE_KEYS:
            rows.append({
                "candidate": r["display_name"],
                "state": state,
                "baseline": r[f"baseline_{state}_mean"],
                "perturbed": r[f"perturbed_{state}_mean"],
            })
    df = pd.DataFrame(rows)
    df["y"] = np.arange(len(df))[::-1]

    for _, row in df.iterrows():
        col = STATE_COLORS[row["state"]]
        ax.plot([row["baseline"], row["perturbed"]], [row["y"], row["y"]],
                color=col, lw=1.5, alpha=0.75)
        ax.scatter(row["baseline"], row["y"], s=26, facecolor="white", edgecolor=col, lw=1.0, zorder=3)
        ax.scatter(row["perturbed"], row["y"], s=34, marker="s", facecolor=col, edgecolor="white", lw=0.4, zorder=4)

    if not clean:
        ax.set_yticks(df["y"])
        ax.set_yticklabels(
            [f"{str(x)[:18]} | {s}" for x, s in zip(df["candidate"], df["state"])],
            fontsize=7
        )
        ax.set_xlabel("Mean predicted state probability", fontsize=8)
        ax.text(0.01, 1.02, "○ baseline   ■ perturbed", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8, color="#475569")
    else:
        ax.set_yticks([])

    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.tick_params(axis="x", labelsize=7)
    for spn in ["top", "right", "left"]:
        ax.spines[spn].set_visible(False)


def draw_heatmap(ax, summary, clean=False):
    cols = ["delta_core_probability", "delta_peri_probability", "delta_remote_probability", "state_shift_priority_score"]
    labels = ["Δ core", "Δ peri", "Δ remote", "rescue score"]
    data = summary[cols].to_numpy(dtype=float)

    vmax = np.nanquantile(np.abs(data), 0.95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.02

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto")

    if not clean:
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(summary)))
        ax.set_yticklabels(summary["display_name"].astype(str).str[:30], fontsize=8)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:+.4f}", ha="center", va="center", fontsize=7, color="#111827")

        cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("mean shift", fontsize=8)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spn in ax.spines.values():
        spn.set_visible(False)


def build_figure(summary, outbase, top_candidate, clean=False, dpi=600, arrow_scale=12, flow_scale=15):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(16.5, 11.5), facecolor="white")
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[0.95, 1.25],
        height_ratios=[0.96, 1.08],
        hspace=0.30,
        wspace=0.25,
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    draw_simplex(axA, summary, arrow_scale=arrow_scale, clean=clean)
    draw_net_flow(axB, summary, top_candidate, flow_scale=flow_scale, clean=clean)
    draw_dumbbell(axC, summary, clean=clean)
    draw_heatmap(axD, summary, clean=clean)

    if not clean:
        fig.suptitle(
            "Strict expression-level in silico perturbation of StrokeNiche state labels",
            fontsize=16,
            fontweight="bold",
            y=0.985,
        )
        axA.set_title("A  Counterfactual state-probability shift", loc="left", fontsize=12, fontweight="bold")
        axB.set_title(f"B  Net redistributed probability mass | {top_candidate}", loc="left", fontsize=12, fontweight="bold")
        axC.set_title("C  Before-after predicted state probability", loc="left", fontsize=12, fontweight="bold")
        axD.set_title("D  Candidate-wise state shift summary", loc="left", fontsize=12, fontweight="bold")

        fig.text(
            0.5,
            0.002,
            "Expression values are computationally downscaled/upshifted and reclassified with a strict label-driven surrogate classifier; not wet-lab KO/blockade or observed cell-fate transition.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#475569",
        )

    outbase = Path(outbase)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--state_table", default="")
    ap.add_argument("--step65_genes", default="")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label_col", default="", help="Optional explicit real state/region label column.")
    ap.add_argument("--allow_probability_labels", action="store_true", help="Allow non-degenerate probability argmax labels.")
    ap.add_argument("--audit_only", action="store_true")
    ap.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    ap.add_argument("--top_flow_candidate", default="ferroptosis_down")
    ap.add_argument("--max_cells", type=int, default=0)
    ap.add_argument("--max_feature_genes", type=int, default=400)
    ap.add_argument("--min_class_n", type=int, default=30)
    ap.add_argument("--min_prob_sd", type=float, default=0.02)
    ap.add_argument("--min_abs_delta_for_main", type=float, default=0.002)
    ap.add_argument("--ko_scale", type=float, default=0.0)
    ap.add_argument("--up_shift_sd", type=float, default=1.0)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--simplex_arrow_scale", type=float, default=12)
    ap.add_argument("--flow_scale", type=float, default=15)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--save_celllevel", action="store_true")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    log("=" * 100)
    log("Step72C | Strict label-driven expression-level in silico KO/blockade")
    log("=" * 100)
    log(f"h5ad={args.h5ad}")
    log(f"state_table={args.state_table}")
    log(f"outdir={outdir}")

    adata = ad.read_h5ad(args.h5ad)
    gmap = gene_symbol_map(adata)

    requested = [x.strip() for x in args.candidates.split(",") if x.strip()]
    specs = {cid: DEFAULT_GENE_SETS[cid] for cid in requested if cid in DEFAULT_GENE_SETS}

    step65_genes = read_step65_genes(args.step65_genes)
    feature_genes, feature_audit = build_feature_genes(
        specs,
        step65_genes,
        gmap,
        max_feature_genes=args.max_feature_genes
    )

    if len(feature_genes) < 10:
        raise RuntimeError(f"Too few matched feature genes: {len(feature_genes)}")

    X0, obs_names, sampled_indices, expr_audit = load_expression(
        adata,
        feature_genes,
        max_cells=args.max_cells
    )

    state_df = read_table(args.state_table) if args.state_table else pd.DataFrame()
    merged, merge_audit = merge_state_and_obs(
        adata,
        state_df,
        obs_names,
        sampled_indices=sampled_indices
    )

    # Select strict labels.
    try:
        y, merged_used, label_audit = choose_labels(
            merged,
            label_col=args.label_col,
            min_class_n=args.min_class_n,
            min_prob_sd=args.min_prob_sd,
            allow_probability_labels=args.allow_probability_labels
        )
    except Exception as e:
        fail_report = {
            "status": "failed_no_usable_state_labels",
            "error": str(e),
            "expression_audit": expr_audit,
            "feature_audit": feature_audit,
            "merge_audit": merge_audit,
            "available_columns_first200": list(merged.columns)[:200],
            "candidate_label_columns": candidate_label_columns(merged, min_class_n=max(5, args.min_class_n // 3))[:50],
            "probability_audit": probability_audit(merged, min_prob_sd=args.min_prob_sd)[0],
            "recommendation": (
                "Use a real cell/spot-level state label column with values mappable to core/peri/remote. "
                "Try --label_col region_auto, --label_col state_group, --label_col refined_state, "
                "or provide a different state_table. Do not use the uniform fixed_region_probs table for Step72."
            )
        }
        (outdir / "step72c_failed_audit.json").write_text(
            json.dumps(fail_report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        log(json.dumps(fail_report, indent=2, ensure_ascii=False))
        raise RuntimeError(fail_report["recommendation"])

    if args.audit_only:
        audit = {
            "status": "audit_only_ok",
            "expression_audit": expr_audit,
            "feature_audit": feature_audit,
            "merge_audit": merge_audit,
            "label_audit": label_audit,
            "available_columns_first200": list(merged.columns)[:200],
        }
        (outdir / "step72c_audit_only.json").write_text(
            json.dumps(audit, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        log(json.dumps(audit, indent=2, ensure_ascii=False))
        return

    idx = merged_used["_adata_row"].astype(int).to_numpy()
    X = X0[idx, :]
    obs_used = merged_used["obs_name"].astype(str).tolist()

    candidate_info, candidate_audit = build_candidate_info(specs, gmap, feature_genes)
    usable = [cid for cid in requested if cid in candidate_info and len(candidate_info[cid]["feature_indices"]) > 0]

    if not usable:
        raise RuntimeError("No usable candidate has matched feature genes.")

    cv_audit = classifier_cv_audit(X, y, C=args.C)

    if cv_audit.get("status") != "ok":
        raise RuntimeError(f"Classifier CV failed: {cv_audit}")

    clf = make_classifier(C=args.C)
    clf.fit(X, y)

    baseline_prob = predict_state_probabilities(clf, X)

    rows = []
    for cid in usable:
        info = candidate_info[cid]
        Xp = perturb_expression(
            X,
            info["feature_indices"],
            direction=info["direction"],
            ko_scale=args.ko_scale,
            up_shift_sd=args.up_shift_sd
        )
        pert_prob = predict_state_probabilities(clf, Xp)

        tmp = pd.DataFrame({
            "obs_name": obs_used,
            "candidate_id": cid,
            "display_name": info["display_name"],
            "direction": info["direction"],
            "n_perturbed_genes": len(info["feature_indices"]),
            "perturbed_feature_genes": ";".join(info["matched_feature_genes"]),
            "baseline_core_probability": baseline_prob[:, 0],
            "baseline_peri_probability": baseline_prob[:, 1],
            "baseline_remote_probability": baseline_prob[:, 2],
            "perturbed_core_probability": pert_prob[:, 0],
            "perturbed_peri_probability": pert_prob[:, 1],
            "perturbed_remote_probability": pert_prob[:, 2],
        })

        for state in STATE_KEYS:
            tmp[f"delta_{state}_probability"] = (
                tmp[f"perturbed_{state}_probability"] -
                tmp[f"baseline_{state}_probability"]
            )

        rows.append(tmp)

    cell_prob = pd.concat(rows, ignore_index=True)
    summary = summarize_cell_prob(cell_prob)

    order_map = {cid: i for i, cid in enumerate(usable)}
    summary["plot_order"] = summary["candidate_id"].map(order_map)
    summary = summary.sort_values("plot_order").reset_index(drop=True)

    flow = make_flow_table(summary)

    max_abs_delta = float(summary[["delta_core_probability", "delta_peri_probability", "delta_remote_probability"]].abs().max().max())
    effect_status = (
        "detectable_expression_counterfactual_shift"
        if max_abs_delta >= args.min_abs_delta_for_main
        else "near_zero_shift_not_suitable_for_main_claim"
    )

    if args.top_flow_candidate in set(summary["candidate_id"]):
        top_candidate = args.top_flow_candidate
    else:
        top_candidate = summary.iloc[0]["candidate_id"]

    candidate_audit_path = outdir / "step72c_candidate_gene_audit.csv"
    summary_path = outdir / "step72c_expression_level_ko_state_shift_summary.csv"
    flow_path = outdir / "step72c_expression_level_ko_net_flow_links.csv"
    feature_path = outdir / "step72c_feature_genes_used.csv"
    model_audit_path = outdir / "step72c_classifier_audit.json"

    candidate_audit.to_csv(candidate_audit_path, index=False)
    summary.to_csv(summary_path, index=False)
    flow.to_csv(flow_path, index=False)
    pd.DataFrame({"feature_gene": feature_genes}).to_csv(feature_path, index=False)

    audit_report = {
        "expression_audit": expr_audit,
        "feature_audit": feature_audit,
        "merge_audit": merge_audit,
        "label_audit": label_audit,
        "classifier_cv_audit": cv_audit,
        "usable_candidates": usable,
        "max_abs_delta": max_abs_delta,
        "effect_status": effect_status,
        "interpretation": (
            "Strict label-driven expression-level in silico KO/blockade-like counterfactual. "
            "This uses real multiclass labels rather than the degenerate uniform probability table. "
            "It remains computational and is not a wet-lab perturbation."
        )
    }
    model_audit_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_celllevel:
        cell_prob.to_csv(outdir / "step72c_expression_level_ko_celllevel_state_probabilities.csv", index=False)
    else:
        cell_prob.sample(min(20000, len(cell_prob)), random_state=20260601).to_csv(
            outdir / "step72c_expression_level_ko_celllevel_state_probabilities.sampled.csv",
            index=False
        )

    annotated_base = outdir / "Fig_Step72C_StrictExpressionLevelKO_StateShift_annotated"
    clean_base = outdir / "Fig_Step72C_StrictExpressionLevelKO_StateShift_clean_no_text"

    build_figure(
        summary,
        annotated_base,
        top_candidate=top_candidate,
        clean=False,
        dpi=args.dpi,
        arrow_scale=args.simplex_arrow_scale,
        flow_scale=args.flow_scale
    )
    build_figure(
        summary,
        clean_base,
        top_candidate=top_candidate,
        clean=True,
        dpi=args.dpi,
        arrow_scale=args.simplex_arrow_scale,
        flow_scale=args.flow_scale
    )

    report = {
        "status": "ok",
        "analysis_name": "Step72C strict label-driven expression-level in silico KO/blockade",
        "top_flow_candidate": top_candidate,
        "effect_status": effect_status,
        "max_abs_delta": max_abs_delta,
        "audit_report": audit_report,
        "outputs": {
            "candidate_gene_audit": str(candidate_audit_path),
            "classifier_audit_json": str(model_audit_path),
            "summary": str(summary_path),
            "net_flow_links": str(flow_path),
            "feature_genes_used": str(feature_path),
            "annotated_pdf": str(annotated_base.with_suffix(".pdf")),
            "annotated_svg": str(annotated_base.with_suffix(".svg")),
            "annotated_png": str(annotated_base.with_suffix(".png")),
            "clean_pdf": str(clean_base.with_suffix(".pdf")),
            "clean_svg": str(clean_base.with_suffix(".svg")),
            "clean_png": str(clean_base.with_suffix(".png")),
        },
        "interpretation_note": (
            "Step72C is the corrected strict version. It must be used only if a real multiclass core/peri/remote label column is found. "
            "If no usable label exists, do not present expression-level KO state-transition figures."
        )
    }

    (outdir / "step72c_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step72c_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 100)
    log("DONE Step72C")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
