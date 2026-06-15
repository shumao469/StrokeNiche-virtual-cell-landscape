#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
65_track_dynamic_marker_regulator_network.py

Purpose
-------
Step 65: Track-specific dynamic marker / module / regulator network.

This is a StrokeNiche version of a UNAGI-like trajectory-specific dynamic marker
and regulator analysis.

It does not require full iDREM. It computes:
  1. Track-specific D1-D3-D7 dynamic modules.
  2. Optional track-specific dynamic genes if an expression matrix is provided.
  3. Dynamic pattern classification:
       increasing / decreasing / transient_up / transient_down / stable
  4. Regulator enrichment using:
       - user-provided regulator-target table
       - GMT gene sets
       - built-in conservative stroke/niche regulator fallback
  5. Track-level regulator summary and marker tables.
  6. Annotated and clean-no-text figures.

Inputs
------
Required:
  --in64c
      Step64c output directory.
      Expected:
        step64c_track_label_refined.csv
        step64c_track_assignment_by_cell.refined_labels.csv

Optional:
  --in64b
      Step64b output directory.
      Used for module columns if needed.
      Expected:
        step64b_modeling_input_used.fixed_region_probs.csv
        step64b_clustered_cells.csv

  --expr_csv
      Optional expression table with obs_name and gene columns.
      If absent, Step65 runs module-level analysis only.

  --expr_h5ad
      Optional AnnData h5ad with obs_names and genes.
      Requires anndata installed.

  --regulator_table
      CSV/TSV with regulator-target pairs.
      Accepted columns:
        regulator,target
        tf,target
        source,target
        database optional

  --gmt
      GMT file with gene sets.
      Format:
        set_name <tab> description <tab> gene1 <tab> gene2 ...

Outputs
-------
outdir/
  step65_track_dynamic_modules.csv
  step65_track_dynamic_genes.csv
  step65_track_dynamic_marker_summary.csv
  step65_regulator_enrichment.csv
  step65_track_regulator_summary.csv
  step65_track_pathway_summary.csv
  step65_track_network_edges.csv
  step65_track_feature_timepoint_means.csv
  step65_report.json
  step65_report.txt
  Fig_StrokeNiche_TrackDynamicMarkerRegulator_annotated.pdf/svg/png
  Fig_StrokeNiche_TrackDynamicMarkerRegulator_clean_no_text.pdf/svg/png

Interpretation
--------------
This analysis produces computational track-specific dynamic-marker and
regulator hypotheses. It should not be described as experimental validation.
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

try:
    from scipy.stats import hypergeom
    SCIPY_AVAILABLE = True
except Exception:
    hypergeom = None
    SCIPY_AVAILABLE = False


BASE = Path("/mnt/h/vir/ST")

DEFAULT_IN64C = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64c_refined_labels_kmeans"
DEFAULT_IN64B = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_kmeans"
DEFAULT_OUT = BASE / "results/step8_strokeniche_perturbmap/track_dynamic_marker_regulator_65_kmeans"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def read_csv_auto(path, required=True):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None

    if p.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(p, sep="\t", low_memory=False)

    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.read_csv(p, sep="\t", low_memory=False)


def find_existing(indir, names, required=True):
    indir = Path(indir)
    for name in names:
        p = indir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    if required:
        raise FileNotFoundError(f"None found in {indir}: {names}")
    return None


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def gene_symbol(x):
    s = str(x).strip()
    s = re.sub(r"^gene_", "", s)
    return s.upper()


def extract_day(tp):
    m = re.search(r"D\s*([0-9]+)", str(tp).upper())
    if m:
        return int(m.group(1))
    m = re.search(r"([0-9]+)", str(tp))
    if m:
        return int(m.group(1))
    return np.nan


def bh_fdr(pvals):
    p = np.asarray([safe_float(x, np.nan) for x in pvals], dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out[order] = q
    return out


def zscore_df(df):
    x = df.apply(pd.to_numeric, errors="coerce")
    mu = x.mean(axis=0)
    sd = x.std(axis=0).replace(0, np.nan)
    z = (x - mu) / sd
    return z.fillna(0.0)


def minmax_series(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


# -----------------------------------------------------------------------------
# Load inputs
# -----------------------------------------------------------------------------

def load_step64c(in64c):
    in64c = Path(in64c)

    tracks_p = find_existing(in64c, ["step64c_track_label_refined.csv"])
    assign_p = find_existing(in64c, ["step64c_track_assignment_by_cell.refined_labels.csv"])

    tracks = read_csv_auto(tracks_p, required=True)
    assign = read_csv_auto(assign_p, required=True)

    if "obs_name" not in assign.columns:
        raise RuntimeError("Track assignment table lacks obs_name.")
    if "track_id" not in assign.columns:
        raise RuntimeError("Track assignment table lacks track_id.")
    if "timepoint" not in assign.columns:
        raise RuntimeError("Track assignment table lacks timepoint.")

    assign["obs_name"] = assign["obs_name"].astype(str)
    assign["track_id"] = assign["track_id"].astype(str)
    assign["timepoint"] = assign["timepoint"].astype(str)
    assign["day"] = assign["timepoint"].map(extract_day)

    return tracks, assign, {"tracks": str(tracks_p), "assignments": str(assign_p)}


def load_64b_cell_table(in64b):
    in64b = Path(in64b)
    p = find_existing(
        in64b,
        [
            "step64b_modeling_input_used.fixed_region_probs.csv",
            "step64b_clustered_cells.csv",
            "step64_modeling_input_used.csv",
            "step64_clustered_cells.csv",
        ],
        required=False,
    )
    if p is None:
        return pd.DataFrame(), ""
    df = read_csv_auto(p, required=False)
    if df is None or df.empty:
        return pd.DataFrame(), str(p)
    if "obs_name" in df.columns:
        df["obs_name"] = df["obs_name"].astype(str)
    return df, str(p)


def detect_module_cols(df):
    cols = []
    for c in df.columns:
        if not str(c).startswith("module_"):
            continue
        if str(c).endswith("_delta") or str(c).endswith("_mean"):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() >= 0.30 and x.nunique(dropna=True) > 1:
            cols.append(c)
    return list(dict.fromkeys(cols))


def detect_context_cols(df):
    candidates = [
        "repair_score_std", "core_probability_std", "peri_probability_std", "remote_probability_std",
        "repair_score", "core_probability", "peri_probability", "remote_probability",
        "prob_lesion_core", "prob_peri_infarct", "prob_remote_like",
        "endothelial_context", "astrocyte_context", "microglia_context", "neuron_context",
        "lesion_niche_context", "remote_offtarget_context",
    ]
    cols = []
    for c in candidates:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().mean() >= 0.30 and x.nunique(dropna=True) > 1:
                cols.append(c)
    return list(dict.fromkeys(cols))


# -----------------------------------------------------------------------------
# Expression loading
# -----------------------------------------------------------------------------

def load_expr_csv(expr_csv, max_genes=3000):
    if not expr_csv:
        return pd.DataFrame(), []

    df = read_csv_auto(expr_csv, required=True)
    if "obs_name" not in df.columns:
        # Try first column as obs_name.
        first = df.columns[0]
        df = df.rename(columns={first: "obs_name"})

    df["obs_name"] = df["obs_name"].astype(str)

    gene_cols = []
    for c in df.columns:
        if c == "obs_name":
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() >= 0.30 and x.nunique(dropna=True) > 1:
            gene_cols.append(c)

    if len(gene_cols) > max_genes:
        var = df[gene_cols].apply(pd.to_numeric, errors="coerce").var(axis=0).sort_values(ascending=False)
        gene_cols = var.head(max_genes).index.tolist()

    return df[["obs_name"] + gene_cols].copy(), gene_cols


def load_expr_h5ad(expr_h5ad, max_genes=3000):
    if not expr_h5ad:
        return pd.DataFrame(), []

    try:
        import anndata as ad
    except Exception:
        raise RuntimeError("anndata is not installed, cannot read h5ad. Use --expr_csv instead.")

    adata = ad.read_h5ad(expr_h5ad)

    obs_names = adata.obs_names.astype(str).tolist()
    var_names = adata.var_names.astype(str).tolist()

    X = adata.X
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            # Select variable genes first using sparse variance approximation.
            means = np.asarray(X.mean(axis=0)).ravel()
            sq_means = np.asarray(X.multiply(X).mean(axis=0)).ravel()
            vars_ = sq_means - means**2
        else:
            vars_ = np.var(np.asarray(X), axis=0)
    except Exception:
        X_dense = np.asarray(X)
        vars_ = np.var(X_dense, axis=0)

    top_idx = np.argsort(vars_)[::-1][:min(max_genes, len(var_names))]
    genes = [var_names[i] for i in top_idx]

    X_sub = X[:, top_idx]
    try:
        import scipy.sparse as sp
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()
    except Exception:
        pass

    expr = pd.DataFrame(np.asarray(X_sub), columns=genes)
    expr.insert(0, "obs_name", obs_names)

    return expr, genes


# -----------------------------------------------------------------------------
# Dynamic pattern scoring
# -----------------------------------------------------------------------------

def classify_dynamic_pattern(v1, v3, v7, min_abs_delta=0.05):
    vals = np.array([safe_float(v1), safe_float(v3), safe_float(v7)], dtype=float)
    if not np.isfinite(vals).all():
        return "missing"

    d31 = vals[1] - vals[0]
    d73 = vals[2] - vals[1]
    d71 = vals[2] - vals[0]

    if d31 >= min_abs_delta and d73 >= min_abs_delta:
        return "increasing"
    if d31 <= -min_abs_delta and d73 <= -min_abs_delta:
        return "decreasing"

    if vals[1] - max(vals[0], vals[2]) >= min_abs_delta:
        return "transient_up"
    if min(vals[0], vals[2]) - vals[1] >= min_abs_delta:
        return "transient_down"

    if d71 >= min_abs_delta:
        return "late_increase"
    if d71 <= -min_abs_delta:
        return "late_decrease"

    return "stable"


def trend_metrics(days, means):
    days = np.asarray(days, dtype=float)
    y = np.asarray(means, dtype=float)
    ok = np.isfinite(days) & np.isfinite(y)

    if ok.sum() < 2:
        return {"slope": np.nan, "spearman": np.nan, "dynamic_range": np.nan}

    d = days[ok]
    yy = y[ok]

    if len(np.unique(d)) >= 2:
        slope = np.polyfit(d, yy, 1)[0]
    else:
        slope = np.nan

    spearman = pd.Series(d).corr(pd.Series(yy), method="spearman") if len(d) >= 3 else np.nan
    dyn_range = np.nanmax(yy) - np.nanmin(yy)

    return {
        "slope": safe_float(slope),
        "spearman": safe_float(spearman),
        "dynamic_range": safe_float(dyn_range),
    }


def compute_track_feature_dynamics(assign, feature_table, feature_cols, feature_type, min_cells_per_timepoint=10, min_abs_delta=0.05):
    """
    assign: many-to-many track assignment table with obs_name, track_id, timepoint.
    feature_table: obs_name + feature columns.
    """
    if feature_table is None or feature_table.empty or not feature_cols:
        return pd.DataFrame(), pd.DataFrame()

    ft = feature_table[["obs_name"] + feature_cols].copy()
    ft["obs_name"] = ft["obs_name"].astype(str)

    # Many-to-many merge: one cell can contribute to several tracks.
    merged = assign[["obs_name", "track_id", "timepoint", "day"]].merge(ft, on="obs_name", how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    mean_rows = []

    for (track_id, feature), sub in merged.melt(
        id_vars=["obs_name", "track_id", "timepoint", "day"],
        value_vars=feature_cols,
        var_name="feature",
        value_name="value",
    ).groupby(["track_id", "feature"]):
        sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
        g = (
            sub.groupby(["timepoint", "day"])
            .agg(
                n_cells=("value", lambda x: int(x.notna().sum())),
                mean_value=("value", "mean"),
                median_value=("value", "median"),
                sd_value=("value", "std"),
            )
            .reset_index()
            .sort_values("day")
        )

        for _, r in g.iterrows():
            mean_rows.append({
                "track_id": track_id,
                "feature": feature,
                "feature_type": feature_type,
                "timepoint": r["timepoint"],
                "day": r["day"],
                "n_cells": r["n_cells"],
                "mean_value": r["mean_value"],
                "median_value": r["median_value"],
                "sd_value": r["sd_value"],
            })

        # Require all D1/D3/D7 if possible.
        time_means = {str(r["timepoint"]): safe_float(r["mean_value"]) for _, r in g.iterrows()}
        time_ns = {str(r["timepoint"]): int(r["n_cells"]) for _, r in g.iterrows()}

        # Use sorted days.
        days = g["day"].values
        means = g["mean_value"].values

        tm = {}
        for _, r in g.iterrows():
            tp = str(r["timepoint"])
            tm[tp] = safe_float(r["mean_value"])

        # Explicit D1/D3/D7 columns if present.
        v1 = tm.get("D1", np.nan)
        v3 = tm.get("D3", np.nan)
        v7 = tm.get("D7", np.nan)

        met = trend_metrics(days, means)

        if np.isfinite([v1, v3, v7]).all():
            pattern = classify_dynamic_pattern(v1, v3, v7, min_abs_delta=min_abs_delta)
            delta_d3_d1 = v3 - v1
            delta_d7_d3 = v7 - v3
            delta_d7_d1 = v7 - v1
        else:
            pattern = "partial_timepoints"
            delta_d3_d1 = np.nan
            delta_d7_d3 = np.nan
            delta_d7_d1 = np.nan

        min_n = min(time_ns.values()) if time_ns else 0

        score = (
            abs(safe_float(delta_d7_d1, 0.0))
            + 0.5 * abs(safe_float(delta_d3_d1, 0.0))
            + 0.5 * abs(safe_float(delta_d7_d3, 0.0))
            + 0.5 * abs(safe_float(met["slope"], 0.0))
            + 0.25 * safe_float(met["dynamic_range"], 0.0)
        )

        rows.append({
            "track_id": track_id,
            "feature": feature,
            "feature_type": feature_type,
            "D1_mean": v1,
            "D3_mean": v3,
            "D7_mean": v7,
            "delta_D3_minus_D1": delta_d3_d1,
            "delta_D7_minus_D3": delta_d7_d3,
            "delta_D7_minus_D1": delta_d7_d1,
            "slope": met["slope"],
            "spearman_time": met["spearman"],
            "dynamic_range": met["dynamic_range"],
            "dynamic_pattern": pattern,
            "min_cells_per_timepoint": min_n,
            "n_timepoints": int(g["timepoint"].nunique()),
            "dynamic_marker_score": safe_float(score),
            "passes_min_cells": bool(min_n >= min_cells_per_timepoint),
        })

    dyn = pd.DataFrame(rows)
    means = pd.DataFrame(mean_rows)

    if not dyn.empty:
        dyn["dynamic_marker_score_norm"] = dyn.groupby("feature_type")["dynamic_marker_score"].transform(lambda x: minmax_series(x).values)
        dyn = dyn.sort_values(["track_id", "dynamic_marker_score"], ascending=[True, False]).reset_index(drop=True)

    return dyn, means


# -----------------------------------------------------------------------------
# Built-in regulator/pathway sets
# -----------------------------------------------------------------------------

def builtin_regulator_sets():
    """
    Conservative fallback regulator/pathway sets focused on stroke niche biology.
    This is not a substitute for curated TRRUST/DoRothEA/ChEA databases, but it
    enables reproducible module/gene-level regulator hypotheses when no local DB
    is available.
    """
    sets = {
        "NFkB_RELA_inflammatory": [
            "TNF","IL1B","IL6","CCL2","CCL3","CCL4","CXCL10","ICAM1","VCAM1","NFKBIA","PTGS2","NOS2",
            "TLR2","TLR4","MYD88","RELA","NFKB1"
        ],
        "STAT3_JAK_injury_reactive": [
            "STAT3","SOCS3","IL6","LIF","OSM","JAK1","JAK2","GFAP","VIM","SERPINA3N","CLU","LGALS3"
        ],
        "HIF1A_hypoxia": [
            "HIF1A","VEGFA","SLC2A1","LDHA","ENO1","PGK1","CA9","BNIP3","ADM","EGLN1","ANGPTL4"
        ],
        "NFE2L2_redox_ferroptosis": [
            "NFE2L2","HMOX1","NQO1","GCLC","GCLM","SLC7A11","GPX4","FTH1","FTL","TXNRD1","PRDX1"
        ],
        "SPI1_microglia_myeloid": [
            "SPI1","TREM2","APOE","AIF1","CSF1R","TYROBP","ITGAM","CD68","LST1","C1QA","C1QB","C1QC"
        ],
        "IRF1_interferon_immune": [
            "IRF1","IRF7","STAT1","ISG15","IFIT1","IFIT3","CXCL10","GBP2","OAS1","MX1"
        ],
        "TGFB_SMAD_ECM": [
            "TGFB1","TGFBR1","TGFBR2","SMAD2","SMAD3","SERPINE1","COL1A1","COL3A1","FN1","CTGF","MMP2"
        ],
        "AP1_JUN_FOS_stress": [
            "JUN","FOS","FOSB","JUNB","JUND","ATF3","EGR1","DUSP1","IER2","IER3"
        ],
        "KLF2_KLF4_endothelial_barrier": [
            "KLF2","KLF4","NOS3","THBD","CLDN5","PECAM1","KDR","FLT1","ESAM","VWF","ENG"
        ],
        "VEGF_FLT1_KDR_vascular": [
            "VEGFA","FLT1","KDR","PGF","ANGPT2","TEK","VWF","PECAM1","ENG","ESAM","CLDN5"
        ],
        "SPP1_CD44_injury_remodeling": [
            "SPP1","CD44","ITGAV","ITGB1","LGALS3","MMP9","MMP14","FN1","COL1A1","COL3A1"
        ],
        "CCL2_CCR2_chemotaxis": [
            "CCL2","CCR2","ACKR1","CXCL10","CCL7","ITGAM","LST1","LYZ","AIF1","TNF","IL1B"
        ],
        "TREM2_APOE_phagocytic_repair": [
            "TREM2","APOE","TYROBP","LPL","CST7","CTSD","CTSB","LGALS3","SPP1","GPNMB"
        ],
        "astrocyte_reactive": [
            "GFAP","VIM","AQP4","SLC1A3","CLU","SERPINA3N","Lcn2".upper(),"C3","STAT3","SOX9"
        ],
        "synaptic_recovery": [
            "SNAP25","SYT1","SYN1","DLG4","GRIN1","GRIA1","MAP2","RBFOX3","BDNF","CAMK2A"
        ],
        "oligodendrocyte_myelin": [
            "MBP","PLP1","MOG","MAG","MOBP","OLIG1","OLIG2","SOX10","CNP","CLDN11"
        ],
    }

    rows = []
    for reg, targets in sets.items():
        for t in targets:
            rows.append({
                "regulator": reg,
                "target": gene_symbol(t),
                "database": "builtin_strokeniche_fallback",
            })

    return pd.DataFrame(rows)


def load_regulator_table(path):
    if not path:
        return pd.DataFrame()

    df = read_csv_auto(path, required=True)

    cols = {norm_key(c): c for c in df.columns}

    reg_col = None
    for k in ["regulator", "tf", "source", "term", "pathway"]:
        if k in cols:
            reg_col = cols[k]
            break

    target_col = None
    for k in ["target", "target_gene", "gene", "genesymbol"]:
        if k in cols:
            target_col = cols[k]
            break

    if reg_col is None or target_col is None:
        raise RuntimeError("Regulator table must contain regulator/tf/source and target/gene columns.")

    db_col = cols.get("database", None)

    out = pd.DataFrame({
        "regulator": df[reg_col].astype(str),
        "target": df[target_col].map(gene_symbol),
        "database": df[db_col].astype(str) if db_col else "user_regulator_table",
    })

    out = out.dropna().drop_duplicates()
    return out


def load_gmt(path):
    if not path:
        return pd.DataFrame()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    rows = []
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            genes = parts[2:]
            for g in genes:
                rows.append({
                    "regulator": name,
                    "target": gene_symbol(g),
                    "database": "gmt",
                })

    return pd.DataFrame(rows).drop_duplicates()


def module_to_pseudo_genes(feature):
    """
    Map module names to pseudo-target genes/pathway tokens for fallback enrichment
    if no expression genes are available.
    """
    f = norm_key(feature)

    mapping = {
        "hypoxia": ["HIF1A","VEGFA","SLC2A1","LDHA","BNIP3"],
        "inflammation": ["TNF","IL1B","IL6","CCL2","NFKB1","RELA"],
        "ferroptosis": ["GPX4","SLC7A11","FTH1","NFE2L2","HMOX1"],
        "bbb_leakage": ["CLDN5","PECAM1","KDR","FLT1","VCAM1","ICAM1"],
        "barrier_stability": ["KLF2","KLF4","CLDN5","PECAM1","NOS3"],
        "repair_ecm": ["TGFB1","SMAD3","FN1","COL1A1","MMP2","SPP1","CD44"],
        "microglia_inflammatory": ["SPI1","TREM2","APOE","TYROBP","CCL2","CCR2","AIF1"],
        "astrocyte_reactive": ["GFAP","VIM","AQP4","STAT3","SOX9","CLU"],
        "endothelial_barrier_fragility": ["KDR","FLT1","VWF","VCAM1","ICAM1","ANGPT2"],
        "synaptic_recovery": ["BDNF","SNAP25","SYT1","SYN1","DLG4","MAP2"],
    }

    genes = []
    for k, v in mapping.items():
        if k in f:
            genes.extend(v)

    return list(dict.fromkeys([gene_symbol(x) for x in genes]))


# -----------------------------------------------------------------------------
# Enrichment
# -----------------------------------------------------------------------------

def hypergeom_pval(M, K, n, x):
    if x <= 0:
        return 1.0
    if SCIPY_AVAILABLE and hypergeom is not None:
        return float(hypergeom.sf(x - 1, M, K, n))
    # conservative fallback: not exact; ranks only
    return float(1.0 / (1.0 + x))


def feature_to_gene_set(feature, feature_type):
    if feature_type == "gene":
        return [gene_symbol(feature)]
    if feature_type in ["module", "context"]:
        return module_to_pseudo_genes(feature)
    return []


def build_marker_gene_sets(dynamic_modules, dynamic_genes, top_n_modules=8, top_n_genes=50):
    rows = []

    if dynamic_genes is not None and not dynamic_genes.empty:
        dg = dynamic_genes.copy()
        dg = dg[dg["passes_min_cells"] == True]
        for tid, sub in dg.groupby("track_id"):
            sub = sub.sort_values("dynamic_marker_score", ascending=False).head(top_n_genes)
            genes = [gene_symbol(x) for x in sub["feature"].tolist()]
            rows.append({
                "track_id": tid,
                "marker_source": "dynamic_genes",
                "marker_features": ";".join(sub["feature"].astype(str).tolist()),
                "gene_set": sorted(set(genes)),
            })

    if dynamic_modules is not None and not dynamic_modules.empty:
        dm = dynamic_modules.copy()
        dm = dm[dm["passes_min_cells"] == True]
        for tid, sub in dm.groupby("track_id"):
            sub = sub.sort_values("dynamic_marker_score", ascending=False).head(top_n_modules)
            genes = []
            for f in sub["feature"]:
                genes.extend(feature_to_gene_set(f, "module"))
            rows.append({
                "track_id": tid,
                "marker_source": "dynamic_modules_pseudo_targets",
                "marker_features": ";".join(sub["feature"].astype(str).tolist()),
                "gene_set": sorted(set(genes)),
            })

    return pd.DataFrame(rows)


def run_regulator_enrichment(marker_sets, regulator_edges, universe_genes=None, min_overlap=1):
    if marker_sets is None or marker_sets.empty:
        return pd.DataFrame()

    if regulator_edges is None or regulator_edges.empty:
        regulator_edges = builtin_regulator_sets()

    reg = regulator_edges.copy()
    reg["target"] = reg["target"].map(gene_symbol)
    reg = reg.dropna().drop_duplicates()

    if universe_genes is None or len(universe_genes) == 0:
        universe = set(reg["target"].unique())
        for gs in marker_sets["gene_set"]:
            universe.update(gs)
    else:
        universe = set([gene_symbol(g) for g in universe_genes])

    M = max(len(universe), 1)

    rows = []
    reg_groups = reg.groupby(["regulator", "database"])["target"].apply(lambda x: set(x) & universe).reset_index()

    for _, ms in marker_sets.iterrows():
        tid = ms["track_id"]
        source = ms["marker_source"]
        genes = set(ms["gene_set"]) & universe
        n = len(genes)

        if n == 0:
            continue

        for _, rg in reg_groups.iterrows():
            targets = rg["target"]
            K = len(targets)
            if K == 0:
                continue
            overlap = genes & targets
            x = len(overlap)
            if x < min_overlap:
                continue

            p = hypergeom_pval(M, K, n, x)

            rows.append({
                "track_id": tid,
                "marker_source": source,
                "regulator": rg["regulator"],
                "database": rg["database"],
                "overlap_n": x,
                "marker_gene_n": n,
                "regulator_target_n": K,
                "universe_n": M,
                "p_value": p,
                "overlap_genes": ";".join(sorted(overlap)),
                "marker_features": ms.get("marker_features", ""),
            })

    enr = pd.DataFrame(rows)
    if enr.empty:
        return enr

    enr["fdr_bh"] = bh_fdr(enr["p_value"])
    enr["enrichment_score"] = -np.log10(enr["fdr_bh"].clip(lower=1e-300)) * enr["overlap_n"]
    enr = enr.sort_values(["track_id", "fdr_bh", "p_value", "overlap_n"], ascending=[True, True, True, False]).reset_index(drop=True)

    return enr


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def make_marker_summary(dynamic_modules, dynamic_genes, tracks, top_n_modules=8, top_n_genes=20):
    rows = []

    tmeta = tracks.set_index("track_id") if "track_id" in tracks.columns else pd.DataFrame()

    tids = sorted(set(dynamic_modules["track_id"].unique()) if dynamic_modules is not None and not dynamic_modules.empty else [])
    if dynamic_genes is not None and not dynamic_genes.empty:
        tids = sorted(set(tids) | set(dynamic_genes["track_id"].unique()))

    for tid in tids:
        row = {"track_id": tid}

        if not tmeta.empty and tid in tmeta.index:
            for c in [
                "track_rank", "track_label_refined", "biological_axis_refined",
                "label_confidence", "state_sequence", "celltype_sequence",
                "delta_repair_score", "delta_core_probability",
                "delta_peri_probability", "delta_remote_probability",
            ]:
                if c in tmeta.columns:
                    row[c] = tmeta.loc[tid, c]

        if dynamic_modules is not None and not dynamic_modules.empty:
            sub = dynamic_modules[(dynamic_modules["track_id"] == tid) & (dynamic_modules["passes_min_cells"] == True)]
            sub = sub.sort_values("dynamic_marker_score", ascending=False).head(top_n_modules)
            row["top_dynamic_modules"] = ";".join([
                f"{r.feature}:{r.dynamic_pattern}:ΔD7-D1={safe_float(r.delta_D7_minus_D1):.3g}"
                for r in sub.itertuples()
            ])
            row["n_dynamic_modules_available"] = int(len(sub))

        if dynamic_genes is not None and not dynamic_genes.empty:
            sub = dynamic_genes[(dynamic_genes["track_id"] == tid) & (dynamic_genes["passes_min_cells"] == True)]
            sub = sub.sort_values("dynamic_marker_score", ascending=False).head(top_n_genes)
            row["top_dynamic_genes"] = ";".join([
                f"{r.feature}:{r.dynamic_pattern}:ΔD7-D1={safe_float(r.delta_D7_minus_D1):.3g}"
                for r in sub.itertuples()
            ])
            row["n_dynamic_genes_available"] = int(len(sub))

        rows.append(row)

    return pd.DataFrame(rows)


def make_regulator_summary(enr, tracks, top_n=8):
    if enr is None or enr.empty:
        return pd.DataFrame()

    tmeta = tracks.set_index("track_id") if "track_id" in tracks.columns else pd.DataFrame()
    rows = []

    for tid, sub in enr.groupby("track_id"):
        sub = sub.sort_values(["fdr_bh", "p_value", "overlap_n"], ascending=[True, True, False]).head(top_n)

        row = {"track_id": tid}
        if not tmeta.empty and tid in tmeta.index:
            for c in ["track_rank", "track_label_refined", "biological_axis_refined", "label_confidence"]:
                if c in tmeta.columns:
                    row[c] = tmeta.loc[tid, c]

        row["top_regulators"] = ";".join([
            f"{r.regulator}|{r.database}|overlap={r.overlap_n}|FDR={safe_float(r.fdr_bh):.2g}|genes={r.overlap_genes}"
            for r in sub.itertuples()
        ])
        row["best_regulator"] = sub.iloc[0]["regulator"] if len(sub) else ""
        row["best_regulator_fdr"] = safe_float(sub.iloc[0]["fdr_bh"]) if len(sub) else np.nan
        row["n_enriched_regulators"] = int((enr[enr["track_id"] == tid]["fdr_bh"] <= 0.25).sum())
        rows.append(row)

    return pd.DataFrame(rows)


def make_network_edges(enr, dynamic_modules, dynamic_genes, max_regulators_per_track=8):
    rows = []

    if enr is None or enr.empty:
        return pd.DataFrame()

    for tid, sub in enr.groupby("track_id"):
        sub = sub.sort_values(["fdr_bh", "p_value", "overlap_n"], ascending=[True, True, False]).head(max_regulators_per_track)

        for _, r in sub.iterrows():
            rows.append({
                "source": tid,
                "target": r["regulator"],
                "edge_type": "track_to_regulator",
                "weight": -np.log10(max(safe_float(r["fdr_bh"], 1.0), 1e-300)),
                "overlap_n": r["overlap_n"],
                "overlap_genes": r["overlap_genes"],
                "database": r["database"],
            })

            for g in str(r["overlap_genes"]).split(";"):
                if g:
                    rows.append({
                        "source": r["regulator"],
                        "target": g,
                        "edge_type": "regulator_to_marker_gene",
                        "weight": 1.0,
                        "overlap_n": 1,
                        "overlap_genes": g,
                        "database": r["database"],
                    })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------

def strip_text(ax):
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


def pattern_colors():
    return {
        "increasing": "#2A9D8F",
        "decreasing": "#E76F51",
        "transient_up": "#F4A261",
        "transient_down": "#457B9D",
        "late_increase": "#8AB17D",
        "late_decrease": "#B56576",
        "stable": "#A8A8A8",
        "partial_timepoints": "#CCCCCC",
        "missing": "#DDDDDD",
    }


def axis_colors():
    return {
        "core_resolution": "#31688E",
        "vascular_barrier": "#35B779",
        "repair_remodeling": "#FDE725",
        "inflammatory": "#CC4678",
        "hypoxia_ferroptosis": "#440154",
        "astrocyte_reactive": "#1F9E89",
        "oligodendroglial_context": "#90D743",
        "remote_maintenance": "#6C757D",
        "persistent_core_injury": "#D94801",
        "state_transition": "#8C8C8C",
    }


def plot_step65(marker_summary, dynamic_modules, regulator_summary, enr, outdir, top_tracks=12, top_modules=12, dpi=600):
    outputs = {}

    # Pick top tracks by track_rank if available.
    ms = marker_summary.copy()
    if "track_rank" in ms.columns:
        ms = ms.sort_values("track_rank").head(top_tracks)
    else:
        ms = ms.head(top_tracks)

    tids = ms["track_id"].astype(str).tolist()

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_StrokeNiche_TrackDynamicMarkerRegulator_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34)

        # A: dynamic module heatmap.
        axA = fig.add_subplot(gs[0, 0])
        dm = dynamic_modules[dynamic_modules["track_id"].isin(tids)].copy() if dynamic_modules is not None and not dynamic_modules.empty else pd.DataFrame()

        if not dm.empty:
            # Select top module features by mean score.
            feat_order = (
                dm.groupby("feature")["dynamic_marker_score"]
                .mean()
                .sort_values(ascending=False)
                .head(top_modules)
                .index
                .tolist()
            )

            mat = []
            for tid in tids:
                row = []
                for f in feat_order:
                    sub = dm[(dm["track_id"] == tid) & (dm["feature"] == f)]
                    row.append(safe_float(sub["delta_D7_minus_D1"].iloc[0]) if len(sub) else 0.0)
                mat.append(row)

            mat = np.asarray(mat, dtype=float)
            vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 1e-6)
            im = axA.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

            if annotate:
                axA.set_title("A | Track-specific dynamic modules", loc="left", fontsize=12, fontweight="bold")
                axA.set_yticks(np.arange(len(tids)))
                axA.set_yticklabels(tids, fontsize=7)
                axA.set_xticks(np.arange(len(feat_order)))
                axA.set_xticklabels([f.replace("module_", "") for f in feat_order], rotation=45, ha="right", fontsize=7)
                cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
            else:
                strip_text(axA)
        else:
            if annotate:
                axA.text(0.5, 0.5, "No dynamic modules", ha="center", va="center")
            else:
                strip_text(axA)

        # B: dynamic pattern composition.
        axB = fig.add_subplot(gs[0, 1])
        if not dm.empty:
            pc = pattern_colors()
            pattern_tab = (
                dm.groupby(["track_id", "dynamic_pattern"])
                .size()
                .reset_index(name="n")
            )
            patterns = [p for p in pc.keys() if p in pattern_tab["dynamic_pattern"].unique()]
            bottom = np.zeros(len(tids))
            for pat in patterns:
                vals = []
                for tid in tids:
                    sub = pattern_tab[(pattern_tab["track_id"] == tid) & (pattern_tab["dynamic_pattern"] == pat)]
                    vals.append(int(sub["n"].iloc[0]) if len(sub) else 0)
                vals = np.asarray(vals)
                axB.bar(tids, vals, bottom=bottom, color=pc.get(pat, "#CCCCCC"), label=pat)
                bottom += vals

            if annotate:
                axB.set_title("B | Dynamic pattern composition", loc="left", fontsize=12, fontweight="bold")
                axB.set_ylabel("Number of dynamic modules")
                axB.tick_params(axis="x", rotation=45, labelsize=7)
                axB.legend(frameon=False, fontsize=7, ncol=2)
            else:
                strip_text(axB)
        else:
            if annotate:
                axB.text(0.5, 0.5, "No pattern table", ha="center", va="center")
            else:
                strip_text(axB)

        # C: top regulators by track.
        axC = fig.add_subplot(gs[1, 0])
        rs = regulator_summary[regulator_summary["track_id"].isin(tids)].copy() if regulator_summary is not None and not regulator_summary.empty else pd.DataFrame()

        if not rs.empty:
            rs["score"] = -np.log10(pd.to_numeric(rs["best_regulator_fdr"], errors="coerce").clip(lower=1e-300))
            rs = rs.set_index("track_id").reindex(tids).reset_index()
            axC.barh(np.arange(len(rs))[::-1], rs["score"].fillna(0).values[::-1])
            if annotate:
                labels = [
                    f"{r.track_id} | {str(r.best_regulator)[:35]}"
                    for r in rs.itertuples()
                ]
                axC.set_yticks(np.arange(len(rs))[::-1])
                axC.set_yticklabels(labels[::-1], fontsize=7)
                axC.set_xlabel("-log10(FDR)")
                axC.set_title("C | Top regulator enrichment", loc="left", fontsize=12, fontweight="bold")
            else:
                strip_text(axC)
        else:
            if annotate:
                axC.text(0.5, 0.5, "No regulator enrichment", ha="center", va="center")
            else:
                strip_text(axC)

        # D: track label distribution / axis summary.
        axD = fig.add_subplot(gs[1, 1])
        if not ms.empty and "biological_axis_refined" in ms.columns:
            ac = axis_colors()
            counts = ms["biological_axis_refined"].value_counts()
            labels = counts.index.tolist()
            vals = counts.values
            cols = [ac.get(x, "#AAAAAA") for x in labels]
            axD.barh(np.arange(len(labels))[::-1], vals[::-1], color=cols[::-1])

            if annotate:
                axD.set_yticks(np.arange(len(labels))[::-1])
                axD.set_yticklabels(labels[::-1], fontsize=8)
                axD.set_xlabel("Number of top tracks")
                axD.set_title("D | Biological axis distribution", loc="left", fontsize=12, fontweight="bold")
            else:
                strip_text(axD)
        else:
            if annotate:
                axD.text(0.5, 0.5, "No axis labels", ha="center", va="center")
            else:
                strip_text(axD)

        for ax in [axA, axB, axC, axD]:
            if annotate:
                ax.grid(linestyle="--", linewidth=0.35, alpha=0.25)
            else:
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if annotate:
            fig.suptitle("Step 65 | Track-specific dynamic marker and regulator network", fontsize=15, fontweight="bold")

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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in64c", default=str(DEFAULT_IN64C))
    ap.add_argument("--in64b", default=str(DEFAULT_IN64B))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--expr_csv", default="")
    ap.add_argument("--expr_h5ad", default="")
    ap.add_argument("--max_genes", type=int, default=3000)

    ap.add_argument("--regulator_table", default="")
    ap.add_argument("--gmt", default="")

    ap.add_argument("--min_cells_per_timepoint", type=int, default=10)
    ap.add_argument("--min_abs_delta_module", type=float, default=0.03)
    ap.add_argument("--min_abs_delta_gene", type=float, default=0.10)

    ap.add_argument("--top_n_modules_for_enrichment", type=int, default=8)
    ap.add_argument("--top_n_genes_for_enrichment", type=int, default=50)
    ap.add_argument("--min_overlap", type=int, default=1)

    ap.add_argument("--top_tracks_fig", type=int, default=12)
    ap.add_argument("--top_modules_fig", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 65: Track-specific dynamic marker / module / regulator network")
    log("=" * 100)
    log(f"in64c={args.in64c}")
    log(f"in64b={args.in64b}")
    log(f"outdir={outdir}")

    tracks, assign, input64c_meta = load_step64c(args.in64c)
    cell_table, cell_table_path = load_64b_cell_table(args.in64b)

    if cell_table.empty:
        raise RuntimeError("Could not load Step64b cell-level table.")

    cell_table["obs_name"] = cell_table["obs_name"].astype(str)

    # Module/context feature table.
    module_cols = detect_module_cols(cell_table)
    context_cols = detect_context_cols(cell_table)
    module_feature_cols = module_cols + context_cols
    module_feature_cols = list(dict.fromkeys(module_feature_cols))

    module_table = cell_table[["obs_name"] + module_feature_cols].copy()

    dynamic_modules, module_means = compute_track_feature_dynamics(
        assign=assign,
        feature_table=module_table,
        feature_cols=module_feature_cols,
        feature_type="module",
        min_cells_per_timepoint=args.min_cells_per_timepoint,
        min_abs_delta=args.min_abs_delta_module,
    )

    # Optional gene expression.
    expr_table = pd.DataFrame()
    gene_cols = []

    if args.expr_csv:
        expr_table, gene_cols = load_expr_csv(args.expr_csv, max_genes=args.max_genes)
    elif args.expr_h5ad:
        expr_table, gene_cols = load_expr_h5ad(args.expr_h5ad, max_genes=args.max_genes)

    if not expr_table.empty and gene_cols:
        dynamic_genes, gene_means = compute_track_feature_dynamics(
            assign=assign,
            feature_table=expr_table,
            feature_cols=gene_cols,
            feature_type="gene",
            min_cells_per_timepoint=args.min_cells_per_timepoint,
            min_abs_delta=args.min_abs_delta_gene,
        )
    else:
        dynamic_genes = pd.DataFrame()
        gene_means = pd.DataFrame()

    # Marker summary.
    marker_summary = make_marker_summary(
        dynamic_modules=dynamic_modules,
        dynamic_genes=dynamic_genes,
        tracks=tracks,
        top_n_modules=args.top_n_modules_for_enrichment,
        top_n_genes=args.top_n_genes_for_enrichment,
    )

    # Regulator database.
    reg_builtin = builtin_regulator_sets()
    reg_user = load_regulator_table(args.regulator_table) if args.regulator_table else pd.DataFrame()
    reg_gmt = load_gmt(args.gmt) if args.gmt else pd.DataFrame()

    regulator_edges = pd.concat([reg_builtin, reg_user, reg_gmt], ignore_index=True)
    regulator_edges = regulator_edges.drop_duplicates()

    # Universe.
    if gene_cols:
        universe = [gene_symbol(g) for g in gene_cols]
    else:
        universe = sorted(set(regulator_edges["target"].unique()))

    marker_sets = build_marker_gene_sets(
        dynamic_modules=dynamic_modules,
        dynamic_genes=dynamic_genes,
        top_n_modules=args.top_n_modules_for_enrichment,
        top_n_genes=args.top_n_genes_for_enrichment,
    )

    regulator_enrichment = run_regulator_enrichment(
        marker_sets=marker_sets,
        regulator_edges=regulator_edges,
        universe_genes=universe,
        min_overlap=args.min_overlap,
    )

    regulator_summary = make_regulator_summary(regulator_enrichment, tracks, top_n=8)
    pathway_summary = regulator_summary.copy()
    network_edges = make_network_edges(regulator_enrichment, dynamic_modules, dynamic_genes)

    # Save outputs.
    dynamic_modules.to_csv(outdir / "step65_track_dynamic_modules.csv", index=False)
    dynamic_genes.to_csv(outdir / "step65_track_dynamic_genes.csv", index=False)
    module_means.to_csv(outdir / "step65_track_feature_timepoint_means.modules.csv", index=False)
    gene_means.to_csv(outdir / "step65_track_feature_timepoint_means.genes.csv", index=False)
    marker_summary.to_csv(outdir / "step65_track_dynamic_marker_summary.csv", index=False)
    marker_sets.to_csv(outdir / "step65_track_marker_gene_sets.csv", index=False)
    regulator_edges.to_csv(outdir / "step65_regulator_database_used.csv", index=False)
    regulator_enrichment.to_csv(outdir / "step65_regulator_enrichment.csv", index=False)
    regulator_summary.to_csv(outdir / "step65_track_regulator_summary.csv", index=False)
    pathway_summary.to_csv(outdir / "step65_track_pathway_summary.csv", index=False)
    network_edges.to_csv(outdir / "step65_track_network_edges.csv", index=False)

    fig_outputs = plot_step65(
        marker_summary=marker_summary,
        dynamic_modules=dynamic_modules,
        regulator_summary=regulator_summary,
        enr=regulator_enrichment,
        outdir=outdir,
        top_tracks=args.top_tracks_fig,
        top_modules=args.top_modules_fig,
        dpi=args.dpi,
    )

    report = {
        "status": "ok",
        "analysis_name": "Step65 track-specific dynamic marker / regulator network",
        "in64c": str(args.in64c),
        "in64b": str(args.in64b),
        "cell_table_used": cell_table_path,
        "expression_mode": "gene_expression" if not dynamic_genes.empty else "module_only",
        "expr_csv": args.expr_csv,
        "expr_h5ad": args.expr_h5ad,
        "n_tracks": int(tracks["track_id"].nunique()) if "track_id" in tracks.columns else int(len(tracks)),
        "n_assignments": int(len(assign)),
        "n_module_features": int(len(module_feature_cols)),
        "n_dynamic_module_rows": int(len(dynamic_modules)),
        "n_gene_features": int(len(gene_cols)),
        "n_dynamic_gene_rows": int(len(dynamic_genes)),
        "n_marker_sets": int(len(marker_sets)),
        "n_regulator_edges_used": int(len(regulator_edges)),
        "n_regulator_enrichment_rows": int(len(regulator_enrichment)),
        "n_network_edges": int(len(network_edges)),
        "scipy_available_for_hypergeom": SCIPY_AVAILABLE,
        "figure_outputs": fig_outputs,
        "outputs": {
            "dynamic_modules": str(outdir / "step65_track_dynamic_modules.csv"),
            "dynamic_genes": str(outdir / "step65_track_dynamic_genes.csv"),
            "module_timepoint_means": str(outdir / "step65_track_feature_timepoint_means.modules.csv"),
            "gene_timepoint_means": str(outdir / "step65_track_feature_timepoint_means.genes.csv"),
            "marker_summary": str(outdir / "step65_track_dynamic_marker_summary.csv"),
            "marker_gene_sets": str(outdir / "step65_track_marker_gene_sets.csv"),
            "regulator_database_used": str(outdir / "step65_regulator_database_used.csv"),
            "regulator_enrichment": str(outdir / "step65_regulator_enrichment.csv"),
            "track_regulator_summary": str(outdir / "step65_track_regulator_summary.csv"),
            "pathway_summary": str(outdir / "step65_track_pathway_summary.csv"),
            "network_edges": str(outdir / "step65_track_network_edges.csv"),
            "report_json": str(outdir / "step65_report.json"),
            "report_txt": str(outdir / "step65_report.txt"),
        },
        "interpretation_note": (
            "Step65 identifies track-specific dynamic module/gene markers and regulator hypotheses. "
            "When no expression matrix or external regulator database is provided, regulator enrichment "
            "uses conservative built-in StrokeNiche fallback gene sets and should be interpreted as "
            "hypothesis generation, not validation."
        ),
    }

    (outdir / "step65_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step 65 track-specific dynamic marker / regulator network report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Top dynamic modules:")
    if not dynamic_modules.empty:
        lines.append(dynamic_modules.sort_values("dynamic_marker_score", ascending=False).head(80).to_string(index=False))
    else:
        lines.append("No dynamic modules.")
    lines.append("")
    lines.append("Top dynamic genes:")
    if not dynamic_genes.empty:
        lines.append(dynamic_genes.sort_values("dynamic_marker_score", ascending=False).head(80).to_string(index=False))
    else:
        lines.append("No dynamic genes. Module-only analysis was performed.")
    lines.append("")
    lines.append("Top regulator enrichment:")
    if not regulator_enrichment.empty:
        lines.append(regulator_enrichment.head(100).to_string(index=False))
    else:
        lines.append("No regulator enrichment.")
    lines.append("")
    lines.append("Track regulator summary:")
    if not regulator_summary.empty:
        lines.append(regulator_summary.head(80).to_string(index=False))
    else:
        lines.append("No track regulator summary.")

    (outdir / "step65_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step 65")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Top dynamic modules:")
    log(dynamic_modules.sort_values("dynamic_marker_score", ascending=False).head(30).to_string(index=False) if not dynamic_modules.empty else "No dynamic modules.")
    log("")
    log("Top regulator enrichment:")
    log(regulator_enrichment.head(30).to_string(index=False) if not regulator_enrichment.empty else "No regulator enrichment.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
