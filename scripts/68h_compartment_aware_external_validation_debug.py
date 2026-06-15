#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
68h_compartment_aware_external_validation_debug.py

Step68H: compartment-aware external validation/debug for GSE233815.

Purpose
-------
Resolve whether weak/inconsistent modules in Step68G are genuinely unsupported
or diluted by whole-cell averaging.

This script:
1. Loads Step68F filtered sparse cell-level h5ad.
2. Scores broad compartments using available module/marker genes.
3. Assigns each cell/barcode to a lightweight compartment:
   neuron_like, astrocyte_like, microglia_like, endothelial_like,
   vascular_ecm_like, stress_hypoxia_redox_like, other_like.
4. Scores Step66f candidate modules within:
   - all cells
   - module-relevant compartments
5. Aggregates to sample-level observations:
   modality × condition × compartment.
6. Runs:
   - D7-D1 repair-direction consistency
   - sc/sn modality consistency
   - balanced within-sample bootstrap
   - stage-label permutation within modality and compartment
7. Compares with Step68E and Step68G.
8. Produces final rescue/debug tier.

Interpretation
--------------
This is a compartment-specific external direction-level support layer.
It is not wet-lab validation and should not be used to force unsupported modules
into positive claims.
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


DROP_GENES = {
    "ACTB", "B2M", "GAPDH", "HPRT", "HPRT1", "PPIA", "MALAT1",
    "RPLP0", "RPS18"
}

COMPARTMENT_MARKERS = {
    "neuron_like": [
        "MAP2", "RBFOX3", "SNAP25", "SYN1", "SYT1", "DLG4",
        "GRIN1", "GRIA1", "GRIA2", "GAP43", "DCX", "NEFL", "NEFM", "SYP"
    ],
    "astrocyte_like": [
        "GFAP", "VIM", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
        "SOX9", "STAT3", "LCN2", "SERPINA3N", "CLU", "C3", "GJA1", "S100B"
    ],
    "microglia_like": [
        "AIF1", "TREM2", "APOE", "TYROBP", "CSF1R", "ITGAM",
        "CD68", "LST1", "C1QA", "C1QB", "C1QC", "LGALS3",
        "SPP1", "CCL2", "CCR2", "TNF", "IL1B", "IL6", "CXCL10"
    ],
    "endothelial_like": [
        "CLDN5", "OCLN", "TJP1", "TJP2", "PECAM1", "CDH5",
        "VWF", "KDR", "FLT1", "TEK", "ENG", "ICAM1", "VCAM1",
        "SELE", "SELP", "PLVAP", "MFSD2A", "ABCB1A", "ANGPT2", "ESAM"
    ],
    "vascular_ecm_like": [
        "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
        "SERPINE1", "CTGF", "FN1", "COL1A1", "COL1A2", "COL3A1",
        "COL4A1", "COL4A2", "MMP2", "MMP9", "TIMP1", "TIMP2",
        "THBS1", "SPP1", "CD44", "ITGAV", "ITGB1"
    ],
    "stress_hypoxia_redox_like": [
        "HIF1A", "EPAS1", "VEGFA", "SLC2A1", "SLC16A1", "SLC16A3",
        "LDHA", "PDK1", "PGK1", "ENO1", "BNIP3", "BNIP3L",
        "GPX4", "SLC7A11", "FTH1", "FTL", "NFE2L2", "HMOX1", "NQO1",
        "ACSL4", "TXNRD1", "PRDX1", "PRDX2", "SOD1", "SOD2"
    ],
}

MODULE_RELEVANT_COMPARTMENTS = {
    "synaptic_recovery": ["neuron_like"],
    "repair_ecm": ["vascular_ecm_like", "endothelial_like", "stress_hypoxia_redox_like"],
    "ferroptosis": ["stress_hypoxia_redox_like", "microglia_like", "endothelial_like"],
    "astrocyte_reactive": ["astrocyte_like"],
    "bbb_leakage": ["endothelial_like"],
    "barrier_stability": ["endothelial_like"],
    "microglia_inflammatory": ["microglia_like"],
    "inflammation": ["microglia_like", "astrocyte_like", "endothelial_like"],
    "hypoxia": ["stress_hypoxia_redox_like", "endothelial_like"],
    "endothelial_barrier_fragility": ["endothelial_like"],
}

MODULE_CATALOG = {
    "synaptic_recovery": [
        "BDNF", "NTRK2", "MAP2", "RBFOX3", "SNAP25", "SYN1", "SYT1",
        "DLG4", "GRIN1", "GRIA1", "GRIA2", "GAP43", "DCX", "NEFL",
        "NEFM", "SYP", "STMN1", "STMN2"
    ],
    "repair_ecm": [
        "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
        "SMAD4", "SERPINE1", "CTGF", "FN1", "COL1A1", "COL1A2",
        "COL3A1", "COL4A1", "COL4A2", "MMP2", "MMP9", "TIMP1",
        "TIMP2", "THBS1", "SPP1", "CD44", "ITGAV", "ITGB1"
    ],
    "ferroptosis": [
        "GPX4", "SLC7A11", "FTH1", "FTL", "NFE2L2", "KEAP1",
        "HMOX1", "NQO1", "GCLC", "GCLM", "ACSL4", "LPCAT3",
        "ALOX15", "TXNRD1", "PRDX1", "PRDX2", "SOD1", "SOD2"
    ],
    "astrocyte_reactive": [
        "GFAP", "VIM", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
        "SOX9", "STAT3", "LCN2", "SERPINA3N", "CLU", "C3",
        "TIMP1", "EMP1", "GJA1", "S100B"
    ],
    "bbb_leakage": [
        "CLDN5", "OCLN", "TJP1", "TJP2", "PECAM1", "CDH5",
        "VWF", "KDR", "FLT1", "TEK", "ENG", "ICAM1", "VCAM1",
        "SELE", "SELP", "PLVAP", "MFSD2A", "ABCB1A", "ANGPT2"
    ],
    "barrier_stability": [
        "CLDN5", "OCLN", "TJP1", "CDH5", "PECAM1", "KLF2",
        "KLF4", "NOS3", "TEK", "ANGPT1", "MFSD2A", "ABCB1A",
        "ABCG2", "ESAM", "ENG"
    ],
    "microglia_inflammatory": [
        "AIF1", "TREM2", "APOE", "TYROBP", "CSF1R", "ITGAM",
        "CD68", "LST1", "C1QA", "C1QB", "C1QC", "LGALS3",
        "SPP1", "CCL2", "CCR2", "TNF", "IL1B", "IL6", "CXCL10"
    ],
    "inflammation": [
        "TNF", "IL1B", "IL1A", "IL6", "IL10", "IL18", "CCL2",
        "CCL3", "CCL4", "CCL5", "CXCL1", "CXCL2", "CXCL10",
        "CCR2", "CCR5", "RELA", "NFKB1", "NFKBIA", "TLR2",
        "TLR4", "MYD88", "NLRP3", "CASP1", "PTGS2", "NOS2"
    ],
    "hypoxia": [
        "HIF1A", "EPAS1", "VEGFA", "SLC2A1", "SLC16A1",
        "SLC16A3", "LDHA", "PDK1", "PGK1", "ENO1", "BNIP3",
        "BNIP3L", "ADM", "EGLN1", "EGLN3", "CA9", "HK2", "MIF"
    ],
    "endothelial_barrier_fragility": [
        "KDR", "FLT1", "VWF", "PLVAP", "VCAM1", "ICAM1",
        "ANGPT2", "SELE", "SELP", "ENG", "ESAM", "PECAM1",
        "CDH5", "CLDN5"
    ],
}


def log(x):
    print(x, flush=True)


def gene_upper(x):
    return str(x).strip().upper()


def split_genes(x):
    if x is None or pd.isna(x):
        return []
    out = []
    for p in re.split(r"[;,\s|/]+", str(x)):
        g = gene_upper(p)
        if not g:
            continue
        if g in DROP_GENES:
            continue
        if g.startswith(("RPL", "RPS", "MRPL", "MRPS", "MT-", "HBA", "HBB", "HBG")):
            continue
        out.append(g)
    return list(dict.fromkeys(out))


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def bh_fdr(pvals):
    p = np.asarray([1.0 if pd.isna(x) else float(x) for x in pvals], dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[order[i]] = min(prev, 1.0)
    return q


def spearman_manual(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    rx = x[ok].rank(method="average").values
    ry = y[ok].rank(method="average").values
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def expected_repair_direction(module_key, perturbation_id=""):
    s = f"{module_key} {perturbation_id}".lower()

    if "synaptic" in s:
        return +1
    if "repair_ecm" in s or "ecm" in s:
        return +1
    if "barrier_stability" in s:
        return +1

    if "ferroptosis" in s:
        return -1
    if "hypoxia" in s:
        return -1
    if "inflammation" in s:
        return -1
    if "microglia" in s:
        return -1
    if "astrocyte" in s:
        return -1
    if "bbb_leakage" in s:
        return -1
    if "endothelial_barrier_fragility" in s:
        return -1

    return np.nan


def stage_numeric_from_condition(s):
    m = {
        "sham": 0.0,
        "ctrl": 0.0,
        "control": 0.0,
        "d1": 1.0,
        "1d": 1.0,
        "day1": 1.0,
        "d3": 2.0,
        "3d": 2.0,
        "day3": 2.0,
        "d7": 3.0,
        "7d": 3.0,
        "day7": 3.0,
    }
    return m.get(str(s).strip().lower(), np.nan)


def d7_minus_d1(df):
    d1 = df.loc[df["condition"].eq("D1"), "module_score"]
    d7 = df.loc[df["condition"].eq("D7"), "module_score"]
    if d1.empty or d7.empty:
        return np.nan
    return float(d7.mean() - d1.mean())


def d1_minus_sham(df):
    d1 = df.loc[df["condition"].eq("D1"), "module_score"]
    sh = df.loc[df["condition"].eq("sham"), "module_score"]
    if d1.empty or sh.empty:
        return np.nan
    return float(d1.mean() - sh.mean())


def load_step66f(step66f_dir):
    p = Path(step66f_dir) / "step66f_manuscript_ready_perturbation_summary.csv"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_csv(p, low_memory=False)

    if "module_key" not in df.columns:
        df["module_key"] = (
            df["perturbation_id"].astype(str)
            .str.replace("_up$", "", regex=True)
            .str.replace("_down$", "", regex=True)
            .str.lower()
        )

    if "display_name" not in df.columns:
        df["display_name"] = df["perturbation_id"].astype(str)

    if "target_genes" not in df.columns:
        df["target_genes"] = ""

    for i, r in df.iterrows():
        module = str(r.get("module_key", ""))
        genes = split_genes(r.get("target_genes", ""))
        if not genes and module in MODULE_CATALOG:
            genes = MODULE_CATALOG[module]
        df.loc[i, "target_genes"] = ";".join(split_genes(";".join(genes)))

    df["expected_repair_direction"] = df.apply(
        lambda r: expected_repair_direction(r.get("module_key", ""), r.get("perturbation_id", "")),
        axis=1
    )

    return df


def load_optional_summary(path_or_dir, preferred_names):
    p = Path(path_or_dir)
    if p.is_dir():
        for name in preferred_names:
            q = p / name
            if q.exists():
                return pd.read_csv(q, low_memory=False)
        return pd.DataFrame()
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return pd.DataFrame()


def load_h5ad(h5ad_path):
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(h5ad_path)

    X = a.X
    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    obs = a.obs.copy()
    obs.index = obs.index.astype(str)
    obs["obs_name"] = obs.index.astype(str)

    for c in ["condition", "modality", "sample_label", "gsm", "obs_name"]:
        if c in obs.columns:
            obs[c] = obs[c].astype(str)

    if "condition" not in obs.columns:
        raise RuntimeError("h5ad obs lacks condition.")
    if "stage_numeric" not in obs.columns:
        obs["stage_numeric"] = obs["condition"].map(stage_numeric_from_condition)
    else:
        obs["stage_numeric"] = pd.to_numeric(obs["stage_numeric"], errors="coerce")
    if "modality" not in obs.columns:
        obs["modality"] = "unknown"
    if "sample_label" not in obs.columns:
        obs["sample_label"] = obs["modality"].astype(str) + "_" + obs["condition"].astype(str)
    if "gsm" not in obs.columns:
        obs["gsm"] = obs["sample_label"].astype(str)

    var_names = [gene_upper(x) for x in a.var_names.astype(str)]

    return X, obs, var_names, a


def zscore_params(X):
    import scipy.sparse as sp
    if sp.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = mean_sq - mean ** 2
    else:
        mean = np.nanmean(X, axis=0)
        var = np.nanvar(X, axis=0)
    var[var < 1e-12] = np.nan
    sd = np.sqrt(var)
    return mean, sd


def score_genes(X, var_names, mean, sd, genes):
    import scipy.sparse as sp

    gene_to_idx = {}
    for i, g in enumerate(var_names):
        gene_to_idx.setdefault(g, []).append(i)

    idx = []
    used = []
    for g in split_genes(";".join(genes)):
        if g in gene_to_idx:
            idx.extend(gene_to_idx[g])
            used.append(g)

    if not idx:
        return np.full(X.shape[0], np.nan), []

    if sp.issparse(X):
        sub = X[:, idx].toarray().astype(np.float32)
    else:
        sub = np.asarray(X[:, idx], dtype=np.float32)

    mu = mean[idx]
    sig = sd[idx]
    z = (sub - mu) / sig
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.mean(axis=1), used


def assign_compartments(X, obs, var_names, mean, sd, outdir, min_margin=0.05):
    score_cols = {}
    matched_rows = []

    for comp, genes in COMPARTMENT_MARKERS.items():
        score, used = score_genes(X, var_names, mean, sd, genes)
        score_cols[comp] = score
        matched_rows.append({
            "compartment": comp,
            "n_marker_genes": len(genes),
            "n_matched_genes": len(used),
            "matched_genes": ";".join(used),
        })

    score_df = pd.DataFrame(score_cols, index=obs.index)
    score_df["max_score"] = score_df[list(score_cols)].max(axis=1)
    score_df["second_score"] = np.sort(score_df[list(score_cols)].values, axis=1)[:, -2]
    score_df["margin"] = score_df["max_score"] - score_df["second_score"]
    score_df["assigned_compartment"] = score_df[list(score_cols)].idxmax(axis=1)
    score_df.loc[score_df["margin"] < min_margin, "assigned_compartment"] = "other_like"

    out = obs[["obs_name", "condition", "stage_numeric", "modality", "sample_label", "gsm"]].copy()
    out["assigned_compartment"] = score_df["assigned_compartment"].values
    out["compartment_score_max"] = score_df["max_score"].values
    out["compartment_margin"] = score_df["margin"].values

    comp_summary = (
        out.groupby(["assigned_compartment", "modality", "condition"], observed=True)
        .size()
        .reset_index(name="n_cells")
        .sort_values(["assigned_compartment", "modality", "condition"])
    )

    matched_df = pd.DataFrame(matched_rows)
    matched_df.to_csv(Path(outdir) / "step68h_compartment_marker_gene_audit.csv", index=False)
    comp_summary.to_csv(Path(outdir) / "step68h_cell_compartment_assignment_summary.csv", index=False)

    return out, score_df, matched_df, comp_summary


def sample_level_direction(sample_scores, expected):
    pooled = d7_minus_d1(sample_scores)
    rho = spearman_manual(sample_scores["stage_numeric"], sample_scores["module_score"])
    repair_consistency = expected * pooled if np.isfinite(expected) and np.isfinite(pooled) else np.nan
    trend_consistency = expected * rho if np.isfinite(expected) and np.isfinite(rho) else np.nan

    d1sham = d1_minus_sham(sample_scores)

    modality = {}
    supports = []
    for mod in ["sc", "sn"]:
        sub = sample_scores[sample_scores["modality"].eq(mod)]
        md = d7_minus_d1(sub)
        mc = expected * md if np.isfinite(expected) and np.isfinite(md) else np.nan
        modality[f"{mod}_D7_minus_D1"] = md
        modality[f"{mod}_repair_consistency"] = mc
        supports.append(bool(mc > 0) if np.isfinite(mc) else False)

    modality_concordant = bool(all(supports)) if len(supports) == 2 else False
    one_modality_supported = bool(any(supports))

    return {
        "D7_minus_D1": pooled,
        "D1_minus_sham": d1sham,
        "stage_spearman_rho": rho,
        "repair_consistency_score": repair_consistency,
        "trend_consistency_score": trend_consistency,
        "modality_concordant": modality_concordant,
        "one_modality_supported": one_modality_supported,
        **modality,
    }


def permutation_p(sample_scores, expected, n_perm, seed):
    rng = np.random.default_rng(seed)
    obs = sample_level_direction(sample_scores, expected)["repair_consistency_score"]

    if not np.isfinite(obs) or len(sample_scores) < 6:
        return 1.0, np.nan, np.nan

    null = []
    for _ in range(n_perm):
        tmp = sample_scores.copy()
        for key, idx in tmp.groupby(["modality"], observed=True).groups.items():
            idx = list(idx)
            cond = tmp.loc[idx, "condition"].values.copy()
            stage = tmp.loc[idx, "stage_numeric"].values.copy()
            perm = rng.permutation(len(idx))
            tmp.loc[idx, "condition"] = cond[perm]
            tmp.loc[idx, "stage_numeric"] = stage[perm]
        val = sample_level_direction(tmp, expected)["repair_consistency_score"]
        null.append(val)

    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]

    if len(null) == 0:
        return 1.0, np.nan, np.nan

    p = float((np.sum(null >= obs) + 1) / (len(null) + 1)) if obs > 0 else 1.0
    return p, float(np.mean(null)), float(np.std(null, ddof=1)) if len(null) > 1 else 0.0


def balanced_bootstrap_compartment(cell_df, score_vec, expected, compartment, n_boot, n_per_sample, seed):
    rng = np.random.default_rng(seed)

    df = cell_df[[
        "obs_name", "condition", "stage_numeric", "modality",
        "sample_label", "gsm", "assigned_compartment"
    ]].copy().reset_index(drop=True)

    df["module_score"] = np.asarray(score_vec)
    if compartment != "all_cells":
        df = df[df["assigned_compartment"].eq(compartment)].copy()

    if df.empty:
        return {
            "bootstrap_available": False,
            "bootstrap_positive_fraction": np.nan,
            "bootstrap_median_repair_consistency": np.nan,
            "bootstrap_ci025": np.nan,
            "bootstrap_ci975": np.nan,
            "bootstrap_n_per_sample_used": 0,
        }

    groups = []
    for sample, idx in df.groupby("sample_label", observed=True).indices.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) > 0:
            groups.append((sample, idx))

    if len(groups) < 6:
        return {
            "bootstrap_available": False,
            "bootstrap_positive_fraction": np.nan,
            "bootstrap_median_repair_consistency": np.nan,
            "bootstrap_ci025": np.nan,
            "bootstrap_ci975": np.nan,
            "bootstrap_n_per_sample_used": 0,
        }

    min_n = min(len(idx) for _, idx in groups)
    take_n = min(n_per_sample, min_n)

    vals = []
    for b in range(n_boot):
        rows = []
        for sample, idx in groups:
            chosen = rng.choice(idx, size=take_n, replace=(len(idx) < take_n))
            sub = df.iloc[chosen]
            first = sub.iloc[0]
            rows.append({
                "sample_label": sample,
                "condition": first["condition"],
                "stage_numeric": first["stage_numeric"],
                "modality": first["modality"],
                "module_score": float(sub["module_score"].mean()),
            })
        st = pd.DataFrame(rows)
        val = sample_level_direction(st, expected)["repair_consistency_score"]
        vals.append(val)

    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]

    if len(vals) == 0:
        return {
            "bootstrap_available": False,
            "bootstrap_positive_fraction": np.nan,
            "bootstrap_median_repair_consistency": np.nan,
            "bootstrap_ci025": np.nan,
            "bootstrap_ci975": np.nan,
            "bootstrap_n_per_sample_used": int(take_n),
        }

    return {
        "bootstrap_available": True,
        "bootstrap_positive_fraction": float(np.mean(vals > 0)),
        "bootstrap_median_repair_consistency": float(np.median(vals)),
        "bootstrap_ci025": float(np.quantile(vals, 0.025)),
        "bootstrap_ci975": float(np.quantile(vals, 0.975)),
        "bootstrap_n_per_sample_used": int(take_n),
    }


def evaluate_candidate_compartments(X, obs_comp, var_names, mean, sd, candidates, n_boot, n_per_sample, n_perm, seed, min_cells_per_sample_compartment):
    all_rows = []

    available_genes = set(var_names)

    for ci, cand in candidates.iterrows():
        pid = str(cand["perturbation_id"])
        module = str(cand["module_key"])
        display = str(cand.get("display_name", pid))
        expected = safe_float(cand.get("expected_repair_direction"))

        genes = split_genes(cand.get("target_genes", ""))
        matched = [g for g in genes if g in available_genes]
        score_vec, used = score_genes(X, var_names, mean, sd, matched)

        relevant = MODULE_RELEVANT_COMPARTMENTS.get(module, [])
        test_compartments = ["all_cells"] + relevant

        for comp_i, comp in enumerate(test_compartments):
            if comp == "all_cells":
                sub_obs = obs_comp.copy()
            else:
                sub_obs = obs_comp[obs_comp["assigned_compartment"].eq(comp)].copy()

            if sub_obs.empty:
                continue

            # Attach scores by positional index from original obs_comp.
            sub_pos = obs_comp.index.get_indexer(sub_obs.index)
            sub_obs = sub_obs.copy()
            sub_obs["module_score"] = score_vec[sub_pos]

            sample_scores = (
                sub_obs.groupby(
                    ["sample_label", "gsm", "modality", "condition", "stage_numeric", "assigned_compartment"],
                    as_index=False,
                    observed=True
                )
                .agg(
                    module_score=("module_score", "mean"),
                    n_cells=("module_score", "size"),
                )
            )

            # Require at least two stages per modality and enough sample-compartment cells.
            n_stage = sample_scores["stage_numeric"].nunique(dropna=True)
            n_obs = len(sample_scores)
            min_cells = int(sample_scores["n_cells"].min()) if not sample_scores.empty else 0

            direction = sample_level_direction(sample_scores, expected)

            if n_obs >= 6 and n_stage >= 2 and min_cells >= min_cells_per_sample_compartment:
                p, null_mean, null_sd = permutation_p(
                    sample_scores,
                    expected,
                    n_perm=n_perm,
                    seed=seed + ci * 97 + comp_i * 13
                )
                boot = balanced_bootstrap_compartment(
                    cell_df=obs_comp,
                    score_vec=score_vec,
                    expected=expected,
                    compartment=comp,
                    n_boot=n_boot,
                    n_per_sample=n_per_sample,
                    seed=seed + ci * 211 + comp_i * 37
                )
            else:
                p, null_mean, null_sd = 1.0, np.nan, np.nan
                boot = {
                    "bootstrap_available": False,
                    "bootstrap_positive_fraction": np.nan,
                    "bootstrap_median_repair_consistency": np.nan,
                    "bootstrap_ci025": np.nan,
                    "bootstrap_ci975": np.nan,
                    "bootstrap_n_per_sample_used": 0,
                }

            row = {
                "perturbation_id": pid,
                "display_name": display,
                "module_key": module,
                "tested_compartment": comp,
                "is_relevant_compartment": bool(comp in relevant),
                "expected_repair_direction": expected,
                "n_module_genes": len(genes),
                "n_matched_genes": len(used),
                "matched_gene_fraction": len(used) / max(len(genes), 1),
                "matched_genes": ";".join(used),
                "n_sample_compartment_observations": int(n_obs),
                "n_unique_stages": int(n_stage),
                "min_cells_per_sample_compartment": int(min_cells),
                "total_cells_in_compartment": int(len(sub_obs)),
                "permutation_p_value": p,
                "permutation_null_mean": null_mean,
                "permutation_null_sd": null_sd,
                **direction,
                **boot,
            }
            all_rows.append(row)

    out = pd.DataFrame(all_rows)
    if out.empty:
        return out

    out["permutation_bh_fdr_global"] = bh_fdr(out["permutation_p_value"].values)

    out["compartment_direction_supported"] = (
        (pd.to_numeric(out["repair_consistency_score"], errors="coerce") > 0)
        & (pd.to_numeric(out["bootstrap_positive_fraction"], errors="coerce").fillna(0) >= 0.70)
        & (pd.to_numeric(out["matched_gene_fraction"], errors="coerce").fillna(0) >= 0.20)
        & (pd.to_numeric(out["n_unique_stages"], errors="coerce").fillna(0) >= 4)
    )

    out["strong_compartment_support"] = (
        out["compartment_direction_supported"]
        & (pd.to_numeric(out["bootstrap_positive_fraction"], errors="coerce").fillna(0) >= 0.80)
        & (
            (pd.to_numeric(out["permutation_p_value"], errors="coerce").fillna(1) <= 0.10)
            | (pd.to_numeric(out["permutation_bh_fdr_global"], errors="coerce").fillna(1) <= 0.25)
        )
    )

    return out


def prepare_prior_tables(step68e_dir, step68g_dir):
    e = load_optional_summary(
        step68e_dir,
        [
            "step68e_nature_ready_external_validation_summary.fixed.csv",
            "step68e_nature_ready_external_validation_summary.csv",
        ]
    )
    g = load_optional_summary(
        step68g_dir,
        ["step68g_balanced_celllevel_sensitivity_summary.csv"]
    )

    if not e.empty:
        keep = [
            "perturbation_id", "module_key", "final_nature_tier",
            "best_repair_consistency_score", "best_permutation_p_value",
            "best_permutation_bh_fdr_by_qc",
            "best_jackknife_positive_fraction",
        ]
        keep = [c for c in keep if c in e.columns]
        e = e[keep].rename(columns={
            "final_nature_tier": "step68e_final_tier",
            "best_repair_consistency_score": "step68e_best_repair_consistency",
            "best_permutation_p_value": "step68e_best_permutation_p",
            "best_permutation_bh_fdr_by_qc": "step68e_best_bh_fdr",
            "best_jackknife_positive_fraction": "step68e_best_jackknife",
        })

    if not g.empty:
        keep = [
            "perturbation_id", "module_key", "step68g_final_sensitivity_tier",
            "repair_consistency_score", "bootstrap_positive_fraction",
            "sample_level_permutation_p_value", "sample_level_permutation_bh_fdr",
        ]
        keep = [c for c in keep if c in g.columns]
        g = g[keep].rename(columns={
            "repair_consistency_score": "step68g_repair_consistency",
            "bootstrap_positive_fraction": "step68g_bootstrap_positive_fraction",
            "sample_level_permutation_p_value": "step68g_permutation_p",
            "sample_level_permutation_bh_fdr": "step68g_bh_fdr",
        })

    return e, g


def final_rescue_summary(comp_df, step68e, step68g):
    rows = []

    for (pid, module), sub in comp_df.groupby(["perturbation_id", "module_key"], observed=True):
        sub = sub.copy()

        relevant = sub[sub["is_relevant_compartment"].astype(bool)].copy()
        if relevant.empty:
            relevant = sub[sub["tested_compartment"].eq("all_cells")].copy()

        sort_cols = [
            "strong_compartment_support",
            "compartment_direction_supported",
            "is_relevant_compartment",
            "bootstrap_positive_fraction",
            "repair_consistency_score",
        ]
        relevant = relevant.sort_values(sort_cols, ascending=[False, False, False, False, False])
        best = relevant.iloc[0]

        row = {
            "perturbation_id": pid,
            "module_key": module,
            "display_name": best.get("display_name", pid),
            "best_compartment": best.get("tested_compartment", ""),
            "best_is_relevant_compartment": bool(best.get("is_relevant_compartment", False)),
            "best_repair_consistency_score": best.get("repair_consistency_score", np.nan),
            "best_modality_concordant": bool(best.get("modality_concordant", False)),
            "best_bootstrap_positive_fraction": best.get("bootstrap_positive_fraction", np.nan),
            "best_permutation_p_value": best.get("permutation_p_value", np.nan),
            "best_permutation_bh_fdr_global": best.get("permutation_bh_fdr_global", np.nan),
            "best_total_cells_in_compartment": best.get("total_cells_in_compartment", np.nan),
            "best_min_cells_per_sample_compartment": best.get("min_cells_per_sample_compartment", np.nan),
            "n_matched_genes": best.get("n_matched_genes", np.nan),
            "matched_gene_fraction": best.get("matched_gene_fraction", np.nan),
            "compartment_direction_supported_any": bool(sub["compartment_direction_supported"].any()),
            "strong_compartment_support_any": bool(sub["strong_compartment_support"].any()),
            "all_supported_compartments": ";".join(
                sub.loc[sub["compartment_direction_supported"], "tested_compartment"].astype(str).tolist()
            ),
        }
        rows.append(row)

    out = pd.DataFrame(rows)

    if not step68e.empty:
        out = out.merge(step68e, on=["perturbation_id", "module_key"], how="left")
    else:
        out["step68e_final_tier"] = "not_available"

    if not step68g.empty:
        out = out.merge(step68g, on=["perturbation_id", "module_key"], how="left")
    else:
        out["step68g_final_sensitivity_tier"] = "not_available"

    out["step68h_final_tier"] = out.apply(assign_step68h_tier, axis=1)

    tier_order = {
        "compartment_specific_rescue_with_68E_or_68G_support": 1,
        "compartment_specific_support_but_primary_discordant": 2,
        "pseudobulk_primary_only_no_compartment_rescue": 3,
        "celllevel_or_compartment_only_exploratory": 4,
        "not_rescued_after_compartment_debug": 5,
    }
    out["tier_order"] = out["step68h_final_tier"].map(tier_order).fillna(9)
    out = out.sort_values(
        ["tier_order", "best_bootstrap_positive_fraction", "best_repair_consistency_score"],
        ascending=[True, False, False]
    ).reset_index(drop=True)
    out["step68h_rank"] = np.arange(1, len(out) + 1)

    return out


def assign_step68h_tier(row):
    e_tier = str(row.get("step68e_final_tier", ""))
    g_tier = str(row.get("step68g_final_sensitivity_tier", ""))

    e_supported = e_tier in {
        "robust_external_direction_support",
        "modality_concordant_external_support",
        "partial_external_direction_support",
    }
    g_supported = g_tier in {
        "68E_and_68F_concordant_strong_sensitivity",
        "68E_supported_with_68F_balanced_sensitivity",
    }

    strong_comp = bool(row.get("strong_compartment_support_any", False))
    comp = bool(row.get("compartment_direction_supported_any", False))

    if strong_comp and (e_supported or g_supported):
        return "compartment_specific_rescue_with_68E_or_68G_support"

    if comp and (e_supported or g_supported):
        return "compartment_specific_rescue_with_68E_or_68G_support"

    if comp and not (e_supported or g_supported):
        return "compartment_specific_support_but_primary_discordant"

    if e_supported and not comp:
        return "pseudobulk_primary_only_no_compartment_rescue"

    if g_supported and not comp:
        return "celllevel_or_compartment_only_exploratory"

    return "not_rescued_after_compartment_debug"


def make_plots(final_df, comp_df, comp_summary, outdir, dpi=600):
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step68H_CompartmentAwareValidation_{mode}"

        fig = plt.figure(figsize=(17, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        counts = final_df["step68h_final_tier"].value_counts()
        ax1.barh(np.arange(len(counts))[::-1], counts.values[::-1])
        if annotate:
            ax1.set_yticks(np.arange(len(counts))[::-1])
            ax1.set_yticklabels(counts.index[::-1], fontsize=8)
            ax1.set_xlabel("Number of modules")
            ax1.set_title("A | Final compartment-aware rescue tiers", loc="left", fontweight="bold")
        else:
            strip_axis(ax1)

        ax2 = fig.add_subplot(gs[0, 1])
        y = np.arange(len(final_df))[::-1]
        ax2.barh(y, final_df["best_repair_consistency_score"].astype(float).values[::-1])
        if annotate:
            ax2.axvline(0, linestyle="--", linewidth=0.8)
            ax2.set_yticks(y)
            ax2.set_yticklabels(
                (final_df["module_key"].astype(str) + " | " + final_df["best_compartment"].astype(str)).values[::-1],
                fontsize=7
            )
            ax2.set_xlabel("Best compartment repair-direction consistency")
            ax2.set_title("B | Best relevant compartment direction", loc="left", fontweight="bold")
        else:
            strip_axis(ax2)

        ax3 = fig.add_subplot(gs[1, 0])
        pivot = comp_df.pivot_table(
            index="module_key",
            columns="tested_compartment",
            values="repair_consistency_score",
            aggfunc="mean"
        )
        im = ax3.imshow(pivot.fillna(0).values, aspect="auto")
        if annotate:
            ax3.set_yticks(np.arange(pivot.shape[0]))
            ax3.set_yticklabels(pivot.index, fontsize=7)
            ax3.set_xticks(np.arange(pivot.shape[1]))
            ax3.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
            ax3.set_title("C | Direction score by compartment", loc="left", fontweight="bold")
            fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        else:
            strip_axis(ax3)

        ax4 = fig.add_subplot(gs[1, 1])
        x = pd.to_numeric(final_df["best_repair_consistency_score"], errors="coerce")
        yv = pd.to_numeric(final_df["best_bootstrap_positive_fraction"], errors="coerce")
        ax4.scatter(x, yv, s=65, alpha=0.85)
        if annotate:
            ax4.axvline(0, linestyle="--", linewidth=0.8)
            ax4.axhline(0.70, linestyle="--", linewidth=0.8)
            for _, r in final_df.iterrows():
                ax4.text(
                    safe_float(r.get("best_repair_consistency_score")),
                    safe_float(r.get("best_bootstrap_positive_fraction")),
                    str(r.get("module_key"))[:20],
                    fontsize=7
                )
            ax4.set_xlabel("Best repair-direction consistency")
            ax4.set_ylabel("Bootstrap positive fraction")
            ax4.set_title("D | Compartment direction stability", loc="left", fontweight="bold")
        else:
            strip_axis(ax4)

        for ax in [ax1, ax2, ax3, ax4]:
            if annotate:
                ax.grid(linestyle="--", linewidth=0.35, alpha=0.25)
            for sp in ax.spines.values():
                sp.set_visible(False)

        if annotate:
            fig.suptitle("Step68H | Compartment-aware external validation/debug", fontsize=15, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.95] if annotate else [0, 0, 1, 1])
        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        outputs[mode] = {
            "pdf": str(outbase.with_suffix(".pdf")),
            "svg": str(outbase.with_suffix(".svg")),
            "png": str(outbase.with_suffix(".png")),
        }

    return outputs


def strip_axis(ax):
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def write_text(final_df, outdir):
    lines = []
    lines.append("Step68H manuscript-ready compartment-aware interpretation")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        "We performed a compartment-aware sensitivity analysis on the QC-filtered GSE233815 cell-level h5ad. "
        "Cells were assigned to lightweight marker-defined compartments using available module genes, and candidate "
        "module directions were re-evaluated within biologically relevant compartments after sample-level aggregation, "
        "stage-label permutation and balanced within-sample bootstrap."
    )
    lines.append("")
    lines.append(
        "This analysis is intended to diagnose whether weak modules in the whole-cell analysis are diluted by cell "
        "composition. It does not replace the primary Step68E pseudobulk robustness result or the Step68G balanced "
        "cell-level sensitivity audit."
    )
    lines.append("")
    lines.append("Module-level interpretation:")
    for _, r in final_df.iterrows():
        lines.append(
            f"- {r['module_key']}: {r['step68h_final_tier']}; "
            f"best compartment={r['best_compartment']}; "
            f"repair consistency={r['best_repair_consistency_score']:.4g}; "
            f"bootstrap positive fraction={r['best_bootstrap_positive_fraction']:.3g}; "
            f"permutation P={r['best_permutation_p_value']:.3g}; "
            f"68E={r.get('step68e_final_tier', 'NA')}; "
            f"68G={r.get('step68g_final_sensitivity_tier', 'NA')}."
        )
    lines.append("")
    lines.append("Conservative wording:")
    lines.append(
        "Only candidates supported in Step68E/Step68G and rescued in the relevant compartment should be described as "
        "compartment-specific external direction-supported hypotheses. Candidates supported only in compartment or "
        "only in direct cell-level analysis should remain exploratory."
    )
    lines.append("")
    lines.append("Forbidden wording:")
    lines.append("experimentally validated therapeutic candidate; clinically validated target; proven treatment response")

    (Path(outdir) / "step68h_manuscript_ready_compartment_text.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--step66f_dir", default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/track_level_perturbation_fdr_66f_manuscript_ready_summary")
    ap.add_argument("--step68e_dir", default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/validation_layer_68e_strict_GSE233815_pseudobulk_robustness")
    ap.add_argument("--step68g_dir", default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/validation_layer_68g_celllevel_balanced_sensitivity")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--min_margin", type=float, default=0.05)
    ap.add_argument("--min_cells_per_sample_compartment", type=int, default=50)
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--n_per_sample", type=int, default=500)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step68H compartment-aware external validation/debug")
    log("=" * 100)

    candidates = load_step66f(args.step66f_dir)
    step68e, step68g = prepare_prior_tables(args.step68e_dir, args.step68g_dir)

    X, obs, var_names, adata = load_h5ad(args.h5ad)
    mean, sd = zscore_params(X)

    obs_comp, comp_score_df, comp_marker_audit, comp_summary = assign_compartments(
        X=X,
        obs=obs,
        var_names=var_names,
        mean=mean,
        sd=sd,
        outdir=outdir,
        min_margin=args.min_margin,
    )

    comp_df = evaluate_candidate_compartments(
        X=X,
        obs_comp=obs_comp,
        var_names=var_names,
        mean=mean,
        sd=sd,
        candidates=candidates,
        n_boot=args.n_boot,
        n_per_sample=args.n_per_sample,
        n_perm=args.n_perm,
        seed=args.seed,
        min_cells_per_sample_compartment=args.min_cells_per_sample_compartment,
    )

    final_df = final_rescue_summary(comp_df, step68e, step68g)

    comp_df.to_csv(outdir / "step68h_candidate_compartment_direction_summary.csv", index=False)
    final_df.to_csv(outdir / "step68h_final_compartment_rescue_summary.csv", index=False)

    write_text(final_df, outdir)
    figs = make_plots(final_df, comp_df, comp_summary, outdir, dpi=args.dpi)

    report = {
        "status": "ok",
        "analysis_name": "Step68H compartment-aware external validation/debug",
        "h5ad": args.h5ad,
        "step66f_dir": args.step66f_dir,
        "step68e_dir": args.step68e_dir,
        "step68g_dir": args.step68g_dir,
        "outdir": str(outdir),
        "n_obs": int(X.shape[0]),
        "n_vars": int(X.shape[1]),
        "n_candidates": int(len(final_df)),
        "tier_counts": final_df["step68h_final_tier"].value_counts().to_dict(),
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "figure_outputs": figs,
        "outputs": {
            "compartment_marker_audit": str(outdir / "step68h_compartment_marker_gene_audit.csv"),
            "compartment_assignment_summary": str(outdir / "step68h_cell_compartment_assignment_summary.csv"),
            "candidate_compartment_direction_summary": str(outdir / "step68h_candidate_compartment_direction_summary.csv"),
            "final_compartment_rescue_summary": str(outdir / "step68h_final_compartment_rescue_summary.csv"),
            "manuscript_text": str(outdir / "step68h_manuscript_ready_compartment_text.txt"),
            "report_json": str(outdir / "step68h_report.json"),
            "report_txt": str(outdir / "step68h_report.txt"),
        },
        "interpretation_note": (
            "Step68H diagnoses whether weak whole-cell signals are rescued in relevant compartments. "
            "It does not constitute wet-lab therapeutic validation."
        ),
    }

    (outdir / "step68h_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    show = [
        "step68h_rank", "perturbation_id", "module_key",
        "step68h_final_tier", "best_compartment",
        "best_repair_consistency_score", "best_modality_concordant",
        "best_bootstrap_positive_fraction", "best_permutation_p_value",
        "best_permutation_bh_fdr_global",
        "step68e_final_tier", "step68g_final_sensitivity_tier",
        "n_matched_genes", "matched_gene_fraction",
    ]
    show = [c for c in show if c in final_df.columns]

    lines = []
    lines.append("Step68H compartment-aware external validation/debug report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Final summary:")
    lines.append(final_df[show].to_string(index=False))
    lines.append("")
    lines.append("Compartment assignment summary:")
    lines.append(comp_summary.head(100).to_string(index=False))
    lines.append("")
    lines.append("Manuscript text:")
    lines.append((outdir / "step68h_manuscript_ready_compartment_text.txt").read_text(encoding="utf-8"))

    (outdir / "step68h_report.txt").write_text("\n".join(lines), encoding="utf-8")

    try:
        adata.file.close()
    except Exception:
        pass

    log("=" * 100)
    log("DONE Step68H")
    log("=" * 100)
    log(final_df[show].to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
