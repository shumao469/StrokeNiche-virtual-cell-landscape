#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
68b_external_mcao_validation_layer.py

Purpose
-------
Medium validation layer for StrokeNiche PerturbMap using local external
MCAO / ischemic stroke scRNA/snRNA/spatial datasets.

This script does NOT claim wet-lab validation.
It evaluates whether candidate modules show expected injury-to-repair direction
in an external dataset.

Inputs
------
Required:
  Step66f:
    step66f_manuscript_ready_perturbation_summary.csv
  Step67:
    step67_benchmark_scores.csv
    step67_benchmark_metrics.csv
    step67_topk_enrichment.csv
    step67_direction_reversal_consistency.csv

Optional external data:
  --external_h5ad
  --external_expr_csv
  --external_meta_csv
  --external_root

If no explicit external file is given, the script searches external_root
for h5ad/csv/tsv files whose names suggest MCAO/stroke/ischemia.

Outputs
-------
outdir/
  step68b_external_dataset_audit.csv
  step68b_validation_candidate_table.csv
  step68b_external_module_scores.csv
  step68b_external_module_direction_validation.csv
  step68b_external_celltype_direction_validation.csv
  step68b_validation_tier_summary.csv
  step68b_future_experimental_validation_plan.csv
  step68b_manuscript_validation_text.txt
  Fig_Step68B_ExternalMCAOValidation_annotated.pdf/svg/png
  Fig_Step68B_ExternalMCAOValidation_clean_no_text.pdf/svg/png
  step68b_report.json/txt

Interpretation
--------------
Use:
  external direction-level support
  public dataset consistency
  computational hypothesis support

Do NOT use:
  experimentally validated therapeutic candidate
  clinically validated target
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


ROOT = Path("/mnt/h/vir/ST/results/step8_strokeniche_perturbmap")
DEFAULT_STEP66F = ROOT / "track_level_perturbation_fdr_66f_manuscript_ready_summary"
DEFAULT_STEP67 = ROOT / "known_positive_decoy_benchmark_67"
DEFAULT_OUT = ROOT / "validation_layer_68b_external_mcao"

DEFAULT_EXTERNAL_ROOT = Path("/mnt/h/vir/ST/resources")

HOUSEKEEPING = {
    "ACTB", "B2M", "GAPDH", "HPRT", "HPRT1", "MALAT1", "PPIA",
    "RPLP0", "RPS18", "RPL13A", "TUBA1A", "TUBA1B", "TUBB",
    "EEF1A1", "EEF2", "PGK1", "LDHA", "LDHB",
}


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def log(x):
    print(x, flush=True)


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def gene_upper(x):
    return str(x).strip().upper()


def split_genes(x):
    if x is None or pd.isna(x):
        return []
    parts = re.split(r"[;,\s|/]+", str(x))
    genes = []
    for p in parts:
        g = gene_upper(p)
        if not g:
            continue
        if not re.match(r"^[A-Z][A-Z0-9\.\-]{1,24}$", g):
            continue
        if g in HOUSEKEEPING:
            continue
        if g.startswith(("RPL", "RPS", "MRPL", "MRPS", "MT-", "HBA", "HBB", "HBG")):
            continue
        genes.append(g)
    return list(dict.fromkeys(genes))


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_table_auto(path, required=False, nrows=None):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return pd.DataFrame()

    try:
        return pd.read_csv(p, low_memory=False, nrows=nrows)
    except Exception:
        try:
            return pd.read_csv(p, sep="\t", low_memory=False, nrows=nrows)
        except Exception:
            if required:
                raise
            return pd.DataFrame()


def read_json_auto(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def minmax(s):
    x = pd.to_numeric(pd.Series(s), errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    mn, mx = x.min(), x.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(x), 0.5), index=x.index)
    return (x - mn) / (mx - mn)


def spearmanr_manual(x, y):
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


def cohen_d(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sd = np.sqrt(((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1)) / max(len(a) + len(b) - 2, 1))
    if sd <= 1e-12:
        return np.nan
    return float((np.mean(b) - np.mean(a)) / sd)


# -----------------------------------------------------------------------------
# Module definitions
# -----------------------------------------------------------------------------

def module_gene_catalog():
    return {
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


def expected_acute_induction_direction(module_key):
    """
    Injury/stress modules should increase from control/sham to acute injury.
    Repair modules may decrease or be mixed, so not forced.
    """
    s = str(module_key).lower()
    if any(k in s for k in ["ferroptosis", "hypoxia", "inflammation", "microglia", "astrocyte", "bbb_leakage", "endothelial_barrier_fragility"]):
        return +1
    return np.nan


def validation_axis_label(module_key):
    d = {
        "synaptic_recovery": "repair-positive synaptic recovery",
        "repair_ecm": "repair-positive ECM remodeling",
        "ferroptosis": "injury-positive ferroptosis/redox stress",
        "astrocyte_reactive": "injury-positive astrocyte reactivity",
        "bbb_leakage": "injury-positive BBB leakage",
        "barrier_stability": "repair-positive barrier stability",
        "microglia_inflammatory": "injury-positive microglial inflammation",
        "inflammation": "injury-positive inflammatory signaling",
        "hypoxia": "injury-positive hypoxia/metabolism",
        "endothelial_barrier_fragility": "injury-positive endothelial fragility",
    }
    return d.get(str(module_key), str(module_key))


# -----------------------------------------------------------------------------
# Load Step66f and Step67
# -----------------------------------------------------------------------------

def load_step66f(step66f_dir):
    p = Path(step66f_dir) / "step66f_manuscript_ready_perturbation_summary.csv"
    df = read_table_auto(p, required=True)

    if "perturbation_id" not in df.columns:
        raise RuntimeError("Step66f summary lacks perturbation_id.")

    if "module_key" not in df.columns:
        df["module_key"] = df["perturbation_id"].map(lambda x: norm_key(x).replace("_up", "").replace("_down", ""))

    catalog = module_gene_catalog()

    if "target_genes" not in df.columns:
        df["target_genes"] = ""

    for i, r in df.iterrows():
        module = str(r.get("module_key", ""))
        genes = split_genes(r.get("target_genes", ""))
        if len(genes) == 0 and module in catalog:
            genes = catalog[module]
            df.loc[i, "target_genes"] = ";".join(genes)

    if "display_name" not in df.columns:
        df["display_name"] = df["perturbation_id"].astype(str).str.replace("_", " ")

    df["expected_repair_direction"] = df.apply(
        lambda r: expected_repair_direction(r.get("module_key", ""), r.get("perturbation_id", "")),
        axis=1
    )
    df["expected_acute_induction_direction"] = df["module_key"].map(expected_acute_induction_direction)
    df["validation_axis"] = df["module_key"].map(validation_axis_label)

    return df


def load_step67(step67_dir):
    d = Path(step67_dir)
    scores = read_table_auto(d / "step67_benchmark_scores.csv")
    metrics = read_table_auto(d / "step67_benchmark_metrics.csv")
    topk = read_table_auto(d / "step67_topk_enrichment.csv")
    drev = read_table_auto(d / "step67_direction_reversal_consistency.csv")
    report = read_json_auto(d / "step67_report.json")
    return scores, metrics, topk, drev, report


def attach_step67_support(candidates, step67_scores, step67_report):
    df = candidates.copy()

    if step67_scores.empty:
        df["step67_support"] = "not_available"
        df["step67_best_control_id"] = ""
        df["step67_best_control_rank"] = np.nan
        df["step67_best_control_score"] = np.nan
    else:
        rows = []
        for _, r in df.iterrows():
            module = str(r.get("module_key", ""))

            sub = step67_scores.copy()
            if "benchmark_label" in sub.columns:
                sub = sub[pd.to_numeric(sub["benchmark_label"], errors="coerce").fillna(0).eq(1)]

            hit = pd.DataFrame()
            if "mapped_step66_module" in sub.columns:
                hit = sub[sub["mapped_step66_module"].astype(str).eq(module)].copy()

            if hit.empty:
                key = module.replace("_", "").lower()
                hit = sub[sub["control_id"].astype(str).map(lambda x: key in norm_key(x).replace("_", ""))].copy()

            if hit.empty:
                best_id = ""
                best_rank = np.nan
                best_score = np.nan
                support = "no_matched_positive_control"
            else:
                hit["benchmark_rank"] = pd.to_numeric(hit["benchmark_rank"], errors="coerce")
                hit = hit.sort_values("benchmark_rank").head(1)
                best_id = str(hit["control_id"].iloc[0])
                best_rank = safe_float(hit["benchmark_rank"].iloc[0])
                best_score = safe_float(hit["benchmark_score"].iloc[0])
                support = "supported_by_step67_known_positive_benchmark"

            row = r.to_dict()
            row.update({
                "step67_support": support,
                "step67_best_control_id": best_id,
                "step67_best_control_rank": best_rank,
                "step67_best_control_score": best_score,
            })
            rows.append(row)

        df = pd.DataFrame(rows)

    df["step67_AUROC_positive_vs_decoy"] = step67_report.get("AUROC_positive_vs_decoy", np.nan)
    df["step67_AUPRC_positive_vs_decoy"] = step67_report.get("AUPRC_average_precision_positive_vs_decoy", np.nan)
    df["step67_direction_reversal_consistency"] = step67_report.get("direction_reversal_consistency", np.nan)
    df["step67_top10_positive_count"] = step67_report.get("top10_positive_count", np.nan)

    return df


# -----------------------------------------------------------------------------
# External discovery
# -----------------------------------------------------------------------------

def score_file_name(path):
    name = path.name.lower()
    full = str(path).lower()

    score = 0
    for k in ["mcao", "stroke", "ischemi", "ischaemi", "infarct", "brain", "cerebral"]:
        if k in full:
            score += 4
    for k in ["h5ad", "expression", "expr", "counts", "matrix", "data"]:
        if k in name:
            score += 3
    for k in ["meta", "obs", "anno", "annotation", "cellinfo", "sample"]:
        if k in name:
            score += 2
    for k in ["regulator", "encode", "msigdb", "dorothea", "collectri", "chea", "chipseq"]:
        if k in full:
            score -= 10
    return score


def discover_external_files(external_root, max_files=2000):
    root = Path(external_root)
    rows = []

    if not root.exists():
        return pd.DataFrame()

    suffixes = {".h5ad", ".csv", ".tsv", ".txt", ".mtx"}
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        low = p.name.lower()
        if low.endswith(".mtx.gz") or p.suffix.lower() in suffixes:
            files.append(p)
        if len(files) >= max_files:
            break

    for p in files:
        low = p.name.lower()
        kind = "unknown"

        if low.endswith(".h5ad"):
            kind = "h5ad"
        elif any(k in low for k in ["meta", "obs", "anno", "annotation", "cellinfo", "sample"]):
            kind = "metadata"
        elif any(k in low for k in ["expr", "expression", "counts", "matrix", "data"]):
            kind = "expression"
        elif low.endswith(".mtx") or low.endswith(".mtx.gz"):
            kind = "mtx"
        else:
            kind = "table_unknown"

        try:
            size_mb = p.stat().st_size / 1024 / 1024
        except Exception:
            size_mb = np.nan

        rows.append({
            "path": str(p),
            "name": p.name,
            "kind": kind,
            "score": score_file_name(p),
            "size_mb": size_mb,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["score", "size_mb"], ascending=[False, False]).reset_index(drop=True)
    return df


def choose_external_files(audit):
    if audit.empty:
        return "", ""

    h5 = audit[(audit["kind"].eq("h5ad")) & (audit["score"] > 0)]
    if not h5.empty:
        return str(h5.iloc[0]["path"]), ""

    exprs = audit[(audit["kind"].isin(["expression", "table_unknown"])) & (audit["score"] > 0)].copy()
    metas = audit[(audit["kind"].eq("metadata")) & (audit["score"] > 0)].copy()

    expr_path = str(exprs.iloc[0]["path"]) if not exprs.empty else ""
    meta_path = ""

    if expr_path and not metas.empty:
        expr_parent = Path(expr_path).parent
        metas["same_parent"] = metas["path"].map(lambda x: Path(x).parent == expr_parent)
        metas = metas.sort_values(["same_parent", "score"], ascending=[False, False])
        meta_path = str(metas.iloc[0]["path"])

    return expr_path, meta_path


# -----------------------------------------------------------------------------
# External data loading
# -----------------------------------------------------------------------------

def load_h5ad_subset(path, genes_needed):
    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(f"anndata not installed; cannot read h5ad: {e}")

    a = ad.read_h5ad(path)

    var_names = pd.Index([gene_upper(x) for x in a.var_names.astype(str)])
    gene_map = {}
    for i, g in enumerate(var_names):
        if g not in gene_map:
            gene_map[g] = i

    matched = [g for g in genes_needed if g in gene_map]
    if len(matched) == 0:
        # still create empty expr with obs
        expr = pd.DataFrame(index=a.obs_names.astype(str))
    else:
        idx = [gene_map[g] for g in matched]
        X = a.X[:, idx]
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                X = X.toarray()
        except Exception:
            pass
        expr = pd.DataFrame(X, index=a.obs_names.astype(str), columns=matched)

    meta = a.obs.copy()
    meta.index = meta.index.astype(str)
    meta["obs_name"] = meta.index

    return expr, meta


def read_external_expr_meta(expr_csv="", meta_csv="", h5ad="", genes_needed=None):
    genes_needed = sorted(set(gene_upper(g) for g in (genes_needed or [])))

    if h5ad:
        expr, meta = load_h5ad_subset(h5ad, genes_needed)
        return expr, meta, "h5ad", h5ad, meta_csv

    if not expr_csv:
        return pd.DataFrame(), pd.DataFrame(), "not_provided", "", ""

    raw = read_table_auto(expr_csv, required=True)

    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), "empty_expression_table", expr_csv, meta_csv

    if raw.shape[1] > 1:
        first = raw.columns[0]
        first_numeric = pd.to_numeric(raw[first].astype(str), errors="coerce").notna().mean()
        if first_numeric < 0.20:
            raw = raw.set_index(first)

    row_genes = set(gene_upper(x) for x in raw.index.astype(str))
    col_genes = set(gene_upper(x) for x in raw.columns.astype(str))

    row_overlap = len(row_genes & set(genes_needed))
    col_overlap = len(col_genes & set(genes_needed))

    if row_overlap > col_overlap:
        expr = raw.T.copy()
        expr.columns = [gene_upper(c) for c in expr.columns]
    else:
        expr = raw.copy()
        expr.columns = [gene_upper(c) for c in expr.columns]

    expr.index = expr.index.astype(str)
    keep_cols = [c for c in expr.columns if c in genes_needed] if genes_needed else list(expr.columns)

    # If no module gene matched, keep all columns for audit but probably no validation.
    if keep_cols:
        expr = expr[keep_cols].copy()

    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=1, how="all")
    expr = expr.dropna(axis=0, how="all")

    meta = pd.DataFrame({"obs_name": expr.index.astype(str)})

    if meta_csv:
        m = read_table_auto(meta_csv, required=False)
        if not m.empty:
            obs_col = detect_obs_col(m)
            if obs_col:
                m[obs_col] = m[obs_col].astype(str)
                meta = meta.merge(m, left_on="obs_name", right_on=obs_col, how="left")
            else:
                if len(m) == len(expr):
                    m = m.copy()
                    m["obs_name"] = expr.index.astype(str)
                    meta = m

    return expr, meta, "csv", expr_csv, meta_csv


def detect_obs_col(meta):
    for c in ["obs_name", "cell_id", "cell", "barcode", "sample_id", "sample", "spot_id"]:
        if c in meta.columns:
            return c
    return ""


def detect_stage_col(meta):
    if meta.empty:
        return ""

    priority = [
        "injury_repair_stage", "stage", "timepoint", "time_point", "day",
        "condition", "group", "region", "state_group", "disease_stage",
        "sample_group", "status", "treatment"
    ]

    for c in priority:
        if c in meta.columns:
            vals = meta[c].astype(str).map(stage_to_numeric)
            if vals.notna().mean() > 0.25 and vals.nunique(dropna=True) >= 2:
                return c

    for c in meta.columns:
        nk = norm_key(c)
        if any(k in nk for k in ["stage", "time", "day", "condition", "group", "region", "state", "treatment"]):
            vals = meta[c].astype(str).map(stage_to_numeric)
            if vals.notna().mean() > 0.25 and vals.nunique(dropna=True) >= 2:
                return c

    return ""


def detect_celltype_col(meta):
    if meta.empty:
        return ""

    priority = [
        "celltype", "cell_type", "cell_type_major", "annotation",
        "cell_annotation", "predicted_celltype", "cluster", "subcluster",
        "cell_class", "major_celltype"
    ]
    for c in priority:
        if c in meta.columns and meta[c].nunique(dropna=True) >= 2:
            return c

    for c in meta.columns:
        nk = norm_key(c)
        if any(k in nk for k in ["celltype", "cell_type", "annotation", "cluster", "class"]):
            if meta[c].nunique(dropna=True) >= 2:
                return c

    return ""


def stage_to_numeric(x):
    s = str(x).lower().strip()
    s2 = re.sub(r"[^a-z0-9]+", "_", s)

    if s in ["nan", "none", ""]:
        return np.nan

    if any(k in s2 for k in ["sham", "control", "normal", "naive", "contralateral", "vehicle"]):
        return 0.0

    if "24h" in s2 or "1d" in s2 or "d1" in s2 or "day1" in s2 or "acute" in s2:
        return 1.0

    if "48h" in s2 or "2d" in s2 or "d2" in s2 or "day2" in s2:
        return 1.5

    if "72h" in s2 or "3d" in s2 or "d3" in s2 or "day3" in s2 or "subacute" in s2:
        return 2.0

    if "7d" in s2 or "d7" in s2 or "day7" in s2 or "repair" in s2 or "recovery" in s2:
        return 3.0

    if "14d" in s2 or "d14" in s2 or "day14" in s2 or "chronic" in s2:
        return 4.0

    if "28d" in s2 or "d28" in s2 or "day28" in s2:
        return 5.0

    if "peri" in s2:
        return 2.0
    if "remote" in s2:
        return 3.0
    if "core" in s2 or "lesion" in s2 or "infarct" in s2 or "mcao" in s2 or "ischemi" in s2 or "ischaemi" in s2 or "stroke" in s2:
        return 1.0

    try:
        return float(s)
    except Exception:
        pass

    m = re.search(r"(\d+)", s2)
    if m:
        val = float(m.group(1))
        if val in [0, 1, 2, 3, 7, 14, 28]:
            if val == 7:
                return 3.0
            if val == 14:
                return 4.0
            if val == 28:
                return 5.0
            return val

    return np.nan


def maybe_log1p(expr):
    if expr.empty:
        return expr
    vals = expr.values
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return expr
    q99 = np.nanpercentile(finite, 99)
    if q99 > 50:
        return np.log1p(expr)
    return expr


def zscore_columns(expr):
    if expr.empty:
        return expr
    mu = expr.mean(axis=0)
    sd = expr.std(axis=0).replace(0, np.nan)
    z = (expr - mu) / sd
    return z.fillna(0)


# -----------------------------------------------------------------------------
# External module validation
# -----------------------------------------------------------------------------

def compute_module_scores(expr, meta, candidates):
    if expr.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "external_available": False,
            "reason": "no_expression_matrix_loaded"
        }

    expr = maybe_log1p(expr)
    expr.columns = [gene_upper(c) for c in expr.columns]
    expr = expr.loc[:, ~pd.Index(expr.columns).duplicated()].copy()
    z = zscore_columns(expr)

    meta = meta.copy()
    if "obs_name" not in meta.columns:
        meta["obs_name"] = z.index.astype(str)
    meta["obs_name"] = meta["obs_name"].astype(str)
    meta = meta.drop_duplicates("obs_name")
    meta = pd.DataFrame({"obs_name": z.index.astype(str)}).merge(meta, on="obs_name", how="left")

    stage_col = detect_stage_col(meta)
    celltype_col = detect_celltype_col(meta)

    if stage_col:
        meta["injury_repair_stage_numeric"] = meta[stage_col].astype(str).map(stage_to_numeric)
    else:
        meta["injury_repair_stage_numeric"] = np.nan

    catalog = module_gene_catalog()

    score_rows = []
    summary_rows = []
    celltype_rows = []

    for _, r in candidates.iterrows():
        pid = str(r.get("perturbation_id", ""))
        module = str(r.get("module_key", ""))
        genes = split_genes(r.get("target_genes", ""))

        if len(genes) == 0 and module in catalog:
            genes = catalog[module]

        matched = [g for g in genes if g in z.columns]

        tmp = meta.copy()
        tmp["perturbation_id"] = pid
        tmp["display_name"] = r.get("display_name", pid)
        tmp["module_key"] = module
        tmp["n_module_genes"] = len(genes)
        tmp["n_matched_genes"] = len(matched)
        tmp["matched_genes"] = ";".join(matched)
        tmp["external_stage_col"] = stage_col
        tmp["external_celltype_col"] = celltype_col

        if len(matched) == 0:
            tmp["module_score"] = np.nan
        else:
            tmp["module_score"] = z[matched].mean(axis=1).values

        score_rows.append(tmp)

        valid = tmp[["module_score", "injury_repair_stage_numeric"]].dropna()

        if len(valid) >= 10 and valid["injury_repair_stage_numeric"].nunique() >= 2:
            rho = spearmanr_manual(valid["injury_repair_stage_numeric"], valid["module_score"])

            stages = sorted(valid["injury_repair_stage_numeric"].unique())
            control_stage = 0.0 if 0.0 in stages else np.nan
            acute_candidates = [x for x in stages if x > 0]
            acute_stage = min(acute_candidates) if acute_candidates else np.nan
            late_stage = max(acute_candidates) if acute_candidates else np.nan

            control_mean = valid.loc[valid["injury_repair_stage_numeric"].eq(control_stage), "module_score"].mean() if np.isfinite(control_stage) else np.nan
            acute_mean = valid.loc[valid["injury_repair_stage_numeric"].eq(acute_stage), "module_score"].mean() if np.isfinite(acute_stage) else np.nan
            late_mean = valid.loc[valid["injury_repair_stage_numeric"].eq(late_stage), "module_score"].mean() if np.isfinite(late_stage) else np.nan

            acute_minus_control = acute_mean - control_mean if np.isfinite(control_mean) and np.isfinite(acute_mean) else np.nan
            late_minus_acute = late_mean - acute_mean if np.isfinite(late_mean) and np.isfinite(acute_mean) else np.nan

            d_acute_control = cohen_d(
                valid.loc[valid["injury_repair_stage_numeric"].eq(control_stage), "module_score"],
                valid.loc[valid["injury_repair_stage_numeric"].eq(acute_stage), "module_score"]
            ) if np.isfinite(control_stage) and np.isfinite(acute_stage) else np.nan

            d_late_acute = cohen_d(
                valid.loc[valid["injury_repair_stage_numeric"].eq(acute_stage), "module_score"],
                valid.loc[valid["injury_repair_stage_numeric"].eq(late_stage), "module_score"]
            ) if np.isfinite(acute_stage) and np.isfinite(late_stage) else np.nan

        else:
            rho = np.nan
            control_stage = np.nan
            acute_stage = np.nan
            late_stage = np.nan
            control_mean = np.nan
            acute_mean = np.nan
            late_mean = np.nan
            acute_minus_control = np.nan
            late_minus_acute = np.nan
            d_acute_control = np.nan
            d_late_acute = np.nan

        expected_repair = safe_float(r.get("expected_repair_direction"))
        expected_acute = safe_float(r.get("expected_acute_induction_direction"))

        repair_axis_consistency = expected_repair * rho if np.isfinite(expected_repair) and np.isfinite(rho) else np.nan
        late_vs_acute_consistency = expected_repair * late_minus_acute if np.isfinite(expected_repair) and np.isfinite(late_minus_acute) else np.nan
        acute_induction_consistency = expected_acute * acute_minus_control if np.isfinite(expected_acute) and np.isfinite(acute_minus_control) else np.nan

        matched_fraction = len(matched) / max(len(genes), 1)

        direction_supported = False
        if matched_fraction >= 0.20:
            support_terms = []
            if np.isfinite(late_vs_acute_consistency):
                support_terms.append(late_vs_acute_consistency > 0)
            if np.isfinite(repair_axis_consistency):
                support_terms.append(repair_axis_consistency > 0)
            if np.isfinite(acute_induction_consistency):
                support_terms.append(acute_induction_consistency > 0)
            direction_supported = any(support_terms)

        summary_rows.append({
            "perturbation_id": pid,
            "display_name": r.get("display_name", pid),
            "module_key": module,
            "validation_axis": r.get("validation_axis", module),
            "expected_repair_direction": expected_repair,
            "expected_acute_induction_direction": expected_acute,
            "n_module_genes": len(genes),
            "n_matched_genes": len(matched),
            "matched_gene_fraction": matched_fraction,
            "matched_genes": ";".join(matched),
            "external_stage_col": stage_col,
            "external_celltype_col": celltype_col,
            "n_external_observations": int(len(tmp)),
            "n_stage_valid_observations": int(len(valid)),
            "n_unique_stages": int(valid["injury_repair_stage_numeric"].nunique()) if not valid.empty else 0,
            "stage_spearman_rho": rho,
            "repair_axis_consistency_score": repair_axis_consistency,
            "acute_stage_numeric": acute_stage,
            "late_stage_numeric": late_stage,
            "control_stage_numeric": control_stage,
            "control_mean_module_score": control_mean,
            "acute_mean_module_score": acute_mean,
            "late_mean_module_score": late_mean,
            "acute_minus_control_module_score": acute_minus_control,
            "late_minus_acute_module_score": late_minus_acute,
            "cohen_d_acute_vs_control": d_acute_control,
            "cohen_d_late_vs_acute": d_late_acute,
            "late_vs_acute_consistency_score": late_vs_acute_consistency,
            "acute_induction_consistency_score": acute_induction_consistency,
            "external_direction_supported": bool(direction_supported),
        })

        if celltype_col and len(matched) > 0:
            tmp2 = tmp.dropna(subset=["module_score", "injury_repair_stage_numeric", celltype_col]).copy()
            for ct, sub in tmp2.groupby(celltype_col):
                if len(sub) < 10 or sub["injury_repair_stage_numeric"].nunique() < 2:
                    continue
                rho_ct = spearmanr_manual(sub["injury_repair_stage_numeric"], sub["module_score"])
                cons_ct = expected_repair * rho_ct if np.isfinite(expected_repair) and np.isfinite(rho_ct) else np.nan
                celltype_rows.append({
                    "perturbation_id": pid,
                    "module_key": module,
                    "celltype": ct,
                    "n_observations": int(len(sub)),
                    "n_unique_stages": int(sub["injury_repair_stage_numeric"].nunique()),
                    "stage_spearman_rho": rho_ct,
                    "repair_axis_consistency_score": cons_ct,
                    "external_direction_supported": bool(cons_ct > 0) if np.isfinite(cons_ct) else False,
                    "n_matched_genes": len(matched),
                    "matched_genes": ";".join(matched),
                })

    score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    celltype_df = pd.DataFrame(celltype_rows)

    audit = {
        "external_available": True,
        "n_observations": int(expr.shape[0]),
        "n_genes_loaded": int(expr.shape[1]),
        "stage_col": stage_col,
        "celltype_col": celltype_col,
        "n_candidates_scored": int(len(candidates)),
        "n_candidates_with_matched_genes": int((summary_df["n_matched_genes"] > 0).sum()) if not summary_df.empty else 0,
        "n_candidates_external_direction_supported": int(summary_df["external_direction_supported"].sum()) if not summary_df.empty else 0,
    }

    return score_df, summary_df, celltype_df, audit


# -----------------------------------------------------------------------------
# Weak support and final tiers
# -----------------------------------------------------------------------------

def build_weak_support(candidates, step67_metrics, step67_topk, step67_drev):
    rows = []

    auroc = np.nan
    auprc = np.nan
    if not step67_metrics.empty and "metric" in step67_metrics.columns:
        if "AUROC_positive_vs_decoy" in set(step67_metrics["metric"]):
            auroc = safe_float(step67_metrics.loc[step67_metrics["metric"].eq("AUROC_positive_vs_decoy"), "value"].iloc[0])
        if "AUPRC_average_precision_positive_vs_decoy" in set(step67_metrics["metric"]):
            auprc = safe_float(step67_metrics.loc[step67_metrics["metric"].eq("AUPRC_average_precision_positive_vs_decoy"), "value"].iloc[0])

    top10_enrich = np.nan
    if not step67_topk.empty and "top_k" in step67_topk.columns:
        hit = step67_topk[pd.to_numeric(step67_topk["top_k"], errors="coerce").eq(10)]
        if not hit.empty:
            top10_enrich = safe_float(hit["fold_enrichment"].iloc[0])

    direction_consistency = np.nan
    if not step67_drev.empty and "consistent" in step67_drev.columns:
        direction_consistency = float(step67_drev["consistent"].astype(str).str.lower().isin(["true", "1", "yes"]).mean())

    for _, r in candidates.iterrows():
        rank = safe_float(r.get("step67_best_control_rank"))
        rank_support = max(0.0, 1.0 - min(rank, 50) / 50.0) if np.isfinite(rank) else 0.2
        auroc_support = 1.0 if np.isfinite(auroc) and auroc >= 0.75 else 0.5
        dir_support = 1.0 if np.isfinite(direction_consistency) and direction_consistency >= 0.8 else 0.5

        weak_score = 0.50 * rank_support + 0.25 * auroc_support + 0.25 * dir_support

        rows.append({
            "perturbation_id": r.get("perturbation_id", ""),
            "module_key": r.get("module_key", ""),
            "weak_support_score": weak_score,
            "step67_best_control_id": r.get("step67_best_control_id", ""),
            "step67_best_control_rank": rank,
            "step67_best_control_score": r.get("step67_best_control_score", np.nan),
            "step67_AUROC_positive_vs_decoy": auroc,
            "step67_AUPRC_positive_vs_decoy": auprc,
            "step67_top10_fold_enrichment": top10_enrich,
            "step67_direction_reversal_consistency": direction_consistency,
        })

    return pd.DataFrame(rows)


def assign_tier(row):
    ext_avail = bool(row.get("external_available", False))
    ext_supported = bool(row.get("external_direction_supported", False))
    weak = safe_float(row.get("weak_support_score"), 0)

    if ext_avail and ext_supported:
        return "medium_external_direction_support"
    if ext_avail:
        return "external_dataset_inconclusive_or_direction_mismatch"
    if weak >= 0.70:
        return "weak_pathway_and_benchmark_support"
    return "weak_or_insufficient_support"


def build_validation_table(candidates, weak, external_summary, external_audit):
    df = candidates.copy()

    weak_cols = [
        "perturbation_id", "weak_support_score", "step67_best_control_id",
        "step67_best_control_rank", "step67_best_control_score",
        "step67_AUROC_positive_vs_decoy", "step67_AUPRC_positive_vs_decoy",
        "step67_top10_fold_enrichment", "step67_direction_reversal_consistency"
    ]
    df = df.merge(weak[[c for c in weak_cols if c in weak.columns]], on="perturbation_id", how="left")

    if not external_summary.empty:
        ext_cols = [
            "perturbation_id", "n_module_genes", "n_matched_genes",
            "matched_gene_fraction", "matched_genes", "external_stage_col",
            "external_celltype_col", "n_external_observations",
            "n_unique_stages", "stage_spearman_rho",
            "repair_axis_consistency_score",
            "acute_minus_control_module_score",
            "late_minus_acute_module_score",
            "late_vs_acute_consistency_score",
            "acute_induction_consistency_score",
            "external_direction_supported"
        ]
        df = df.merge(external_summary[[c for c in ext_cols if c in external_summary.columns]], on="perturbation_id", how="left")
        df["external_available"] = True
    else:
        df["external_available"] = False
        df["external_direction_supported"] = False
        df["n_matched_genes"] = np.nan
        df["matched_gene_fraction"] = np.nan
        df["stage_spearman_rho"] = np.nan
        df["late_vs_acute_consistency_score"] = np.nan

    df["validation_tier"] = df.apply(assign_tier, axis=1)

    df["recommended_claim"] = df["validation_tier"].map({
        "medium_external_direction_support": "externally direction-supported; not experimental validation",
        "external_dataset_inconclusive_or_direction_mismatch": "external dataset available but direction support is inconclusive or mismatched",
        "weak_pathway_and_benchmark_support": "pathway/benchmark-supported computational hypothesis",
        "weak_or_insufficient_support": "exploratory hypothesis requiring further validation",
    }).fillna("exploratory hypothesis requiring further validation")

    df["therapeutic_validation_claim_allowed"] = False
    df["allowed_language"] = "computationally prioritized / internally benchmarked / externally direction-supported if applicable"
    df["forbidden_language"] = "experimentally validated therapeutic candidate; clinically validated target"

    tier_order = {
        "medium_external_direction_support": 1,
        "weak_pathway_and_benchmark_support": 2,
        "external_dataset_inconclusive_or_direction_mismatch": 3,
        "weak_or_insufficient_support": 4,
    }
    df["validation_tier_order"] = df["validation_tier"].map(tier_order).fillna(9)

    sort_cols = ["validation_tier_order"]
    ascending = [True]

    if "weak_support_score" in df.columns:
        sort_cols.append("weak_support_score")
        ascending.append(False)
    if "s_overall_observed" in df.columns:
        sort_cols.append("s_overall_observed")
        ascending.append(False)

    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    df["validation_rank"] = np.arange(1, len(df) + 1)

    return df


def future_plan(validation_table):
    rows = []
    for _, r in validation_table.head(6).iterrows():
        module = str(r.get("module_key", ""))
        pid = str(r.get("perturbation_id", ""))

        if module == "ferroptosis":
            model = "OGD/R neuron-glia culture or organotypic brain slice"
            readout = "lipid ROS, GPX4/SLC7A11/FTH1/HMOX1, cell viability, core-like state reduction"
        elif module == "synaptic_recovery":
            model = "OGD/R neuronal culture or organotypic brain slice"
            readout = "MAP2/SYN1/SNAP25/DLG4 recovery, neuronal viability, synaptic module score"
        elif module == "repair_ecm":
            model = "organotypic brain slice or astrocyte-microglia-endothelial co-culture"
            readout = "repair-permissive ECM markers, SPP1/CD44/FN1/MMP/TIMP balance, scarring risk"
        elif "bbb" in module or "barrier" in module or "endothelial" in module:
            model = "microglia-endothelial-astrocyte BBB co-culture"
            readout = "TEER/permeability, CLDN5/OCLN/TJP1, PLVAP/VCAM1/ICAM1"
        elif "microglia" in module or "inflammation" in module:
            model = "microglia-endothelial/astrocyte co-culture after OGD/R"
            readout = "CCL2/CCR2/TNF/IL1B/CXCL10, phagocytosis and viability controls"
        else:
            model = "OGD/R or organotypic brain slice perturbation assay"
            readout = "module score reversal, latent/state shift, toxicity and cell-type specificity"

        rows.append({
            "perturbation_id": pid,
            "module_key": module,
            "suggested_validation_strength": "strong_wetlab_followup",
            "recommended_model": model,
            "primary_readout": readout,
            "secondary_readout": "latent/state shift, RRHO-like DEG overlap, module score reversal, toxicity/safety control",
            "interpretation_if_successful": "experimental support for computational prioritization; not automatically clinical validation",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting and text
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


def plot_validation(validation_table, external_summary, outdir, dpi=600):
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step68B_ExternalMCAOValidation_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.35)

        # A tier summary
        axA = fig.add_subplot(gs[0, 0])
        counts = validation_table["validation_tier"].value_counts()
        axA.barh(np.arange(len(counts))[::-1], counts.values[::-1])
        if annotate:
            axA.set_yticks(np.arange(len(counts))[::-1])
            axA.set_yticklabels(counts.index[::-1], fontsize=8)
            axA.set_xlabel("Number of candidates")
            axA.set_title("A | Validation tier summary", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axA)

        # B weak support score
        axB = fig.add_subplot(gs[0, 1])
        df = validation_table.sort_values("validation_rank")
        y = np.arange(len(df))[::-1]
        axB.barh(y, pd.to_numeric(df["weak_support_score"], errors="coerce").fillna(0).values[::-1])
        if annotate:
            axB.set_yticks(y)
            axB.set_yticklabels(df["display_name"].astype(str).values[::-1], fontsize=7)
            axB.set_xlabel("Step67 / pathway support score")
            axB.set_title("B | Internal benchmark support", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axB)

        # C external consistency
        axC = fig.add_subplot(gs[1, 0])
        if external_summary is not None and not external_summary.empty:
            ext = external_summary.copy()
            ext["late_vs_acute_consistency_score"] = pd.to_numeric(ext["late_vs_acute_consistency_score"], errors="coerce")
            ext = ext.sort_values("late_vs_acute_consistency_score", ascending=False)
            y2 = np.arange(len(ext))[::-1]
            axC.barh(y2, ext["late_vs_acute_consistency_score"].fillna(0).values[::-1])
            if annotate:
                axC.axvline(0, linestyle="--", linewidth=0.8)
                axC.set_yticks(y2)
                axC.set_yticklabels(ext["display_name"].astype(str).values[::-1], fontsize=7)
                axC.set_xlabel("Late-vs-acute expected-direction consistency")
                axC.set_title("C | External injury-to-repair direction", loc="left", fontsize=12, fontweight="bold")
            else:
                strip_text(axC)
        else:
            axC.text(0.5, 0.5, "No external dataset loaded", ha="center", va="center")
            if annotate:
                axC.set_title("C | External validation not available", loc="left", fontsize=12, fontweight="bold")
            else:
                strip_text(axC)

        # D Step66 score vs support
        axD = fig.add_subplot(gs[1, 1])
        x = pd.to_numeric(validation_table.get("s_overall_observed", pd.Series(np.nan, index=validation_table.index)), errors="coerce")
        yv = pd.to_numeric(validation_table.get("weak_support_score", pd.Series(np.nan, index=validation_table.index)), errors="coerce")
        axD.scatter(x, yv, s=55, alpha=0.8)
        if annotate:
            for _, r in validation_table.head(6).iterrows():
                axD.text(
                    safe_float(r.get("s_overall_observed")),
                    safe_float(r.get("weak_support_score")),
                    str(r.get("perturbation_id"))[:24],
                    fontsize=7,
                )
            axD.set_xlabel("Step66 perturbation score")
            axD.set_ylabel("Weak support score")
            axD.set_title("D | Perturbation score vs validation support", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axD)

        for ax in [axA, axB, axC, axD]:
            if annotate:
                ax.grid(linestyle="--", linewidth=0.35, alpha=0.25)
            else:
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if annotate:
            fig.suptitle(
                "Step68B | External MCAO / stroke validation-support layer",
                fontsize=15,
                fontweight="bold",
            )

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


def write_text(validation_table, external_audit, outdir):
    n_total = len(validation_table)
    n_medium = int(validation_table["validation_tier"].eq("medium_external_direction_support").sum())
    n_weak = int(validation_table["validation_tier"].eq("weak_pathway_and_benchmark_support").sum())

    lines = []
    lines.append("Step68B manuscript validation text")
    lines.append("=" * 100)
    lines.append("")

    if external_audit.get("external_available", False):
        lines.append(
            f"We implemented a validation-support layer using a local external MCAO/stroke expression dataset. "
            f"Among {n_total} candidate modules, {n_medium} showed direction-level support along the injury-to-repair axis. "
            f"This analysis provides external direction-level support, not experimental therapeutic validation."
        )
    else:
        lines.append(
            f"We implemented a validation-support layer integrating pathway plausibility and Step67 known-positive/decoy benchmark support. "
            f"No usable external MCAO/stroke expression dataset was loaded in this run; therefore, the analysis remains weak-support/hypothesis-prioritization only."
        )

    lines.append("")
    lines.append("Leading candidates:")
    for _, r in validation_table.head(5).iterrows():
        lines.append(
            f"- {r.get('display_name', r.get('perturbation_id'))}: "
            f"{r.get('validation_tier')}; {r.get('recommended_claim')}."
        )

    lines.append("")
    lines.append("Allowed wording:")
    lines.append("externally direction-supported; internally benchmarked; pathway-consistent; computationally prioritized; experimental-prioritization hypothesis")
    lines.append("")
    lines.append("Forbidden wording:")
    lines.append("experimentally validated therapeutic candidate; clinically validated target; validated treatment response")

    (Path(outdir) / "step68b_manuscript_validation_text.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--step66f_dir", default=str(DEFAULT_STEP66F))
    ap.add_argument("--step67_dir", default=str(DEFAULT_STEP67))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--external_root", default=str(DEFAULT_EXTERNAL_ROOT))
    ap.add_argument("--external_h5ad", default="")
    ap.add_argument("--external_expr_csv", default="")
    ap.add_argument("--external_meta_csv", default="")
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step68B external MCAO / stroke validation layer")
    log("=" * 100)
    log(f"outdir={outdir}")

    candidates = load_step66f(args.step66f_dir)

    step67_scores, step67_metrics, step67_topk, step67_drev, step67_report = load_step67(args.step67_dir)
    candidates = attach_step67_support(candidates, step67_scores, step67_report)

    weak = build_weak_support(candidates, step67_metrics, step67_topk, step67_drev)

    # External file discovery.
    audit_files = discover_external_files(args.external_root)
    audit_files.to_csv(outdir / "step68b_external_dataset_audit.csv", index=False)

    h5ad = args.external_h5ad
    expr_csv = args.external_expr_csv
    meta_csv = args.external_meta_csv

    if not h5ad and not expr_csv:
        chosen_expr, chosen_meta = choose_external_files(audit_files)
        if chosen_expr.lower().endswith(".h5ad"):
            h5ad = chosen_expr
            expr_csv = ""
            meta_csv = ""
        else:
            expr_csv = chosen_expr
            meta_csv = chosen_meta

    # Genes needed from all candidate target sets and catalogs.
    genes_needed = set()
    for x in candidates["target_genes"].fillna(""):
        genes_needed.update(split_genes(x))
    for gs in module_gene_catalog().values():
        genes_needed.update(gs)
    genes_needed = sorted(genes_needed)

    try:
        expr, meta, external_mode, used_expr, used_meta = read_external_expr_meta(
            expr_csv=expr_csv,
            meta_csv=meta_csv,
            h5ad=h5ad,
            genes_needed=genes_needed,
        )
        external_load_error = ""
    except Exception as e:
        expr = pd.DataFrame()
        meta = pd.DataFrame()
        external_mode = "load_failed"
        used_expr = expr_csv or h5ad
        used_meta = meta_csv
        external_load_error = str(e)

    external_scores, external_summary, external_celltype, external_audit = compute_module_scores(
        expr=expr,
        meta=meta,
        candidates=candidates,
    )

    external_audit.update({
        "external_mode": external_mode,
        "external_h5ad_used": h5ad,
        "external_expr_csv_used": used_expr if not h5ad else "",
        "external_meta_csv_used": used_meta,
        "external_load_error": external_load_error,
        "external_root": args.external_root,
    })

    validation_table = build_validation_table(
        candidates=candidates,
        weak=weak,
        external_summary=external_summary,
        external_audit=external_audit,
    )

    tier_summary = (
        validation_table["validation_tier"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "validation_tier", "validation_tier": "n_candidates"})
    )

    exp_plan = future_plan(validation_table)

    validation_table.to_csv(outdir / "step68b_validation_candidate_table.csv", index=False)
    weak.to_csv(outdir / "step68b_weak_validation_evidence.csv", index=False)
    external_scores.to_csv(outdir / "step68b_external_module_scores.csv", index=False)
    external_summary.to_csv(outdir / "step68b_external_module_direction_validation.csv", index=False)
    external_celltype.to_csv(outdir / "step68b_external_celltype_direction_validation.csv", index=False)
    tier_summary.to_csv(outdir / "step68b_validation_tier_summary.csv", index=False)
    exp_plan.to_csv(outdir / "step68b_future_experimental_validation_plan.csv", index=False)

    write_text(validation_table, external_audit, outdir)
    fig_outputs = plot_validation(validation_table, external_summary, outdir, dpi=args.dpi)

    report = {
        "status": "ok",
        "analysis_name": "Step68B external MCAO/stroke validation-support layer",
        "step66f_dir": args.step66f_dir,
        "step67_dir": args.step67_dir,
        "outdir": str(outdir),
        "n_candidates": int(len(validation_table)),
        "validation_tier_counts": validation_table["validation_tier"].value_counts().to_dict(),
        "external_audit": external_audit,
        "step67_report_summary": {
            "AUROC_positive_vs_decoy": step67_report.get("AUROC_positive_vs_decoy"),
            "AUPRC_average_precision_positive_vs_decoy": step67_report.get("AUPRC_average_precision_positive_vs_decoy"),
            "direction_reversal_consistency": step67_report.get("direction_reversal_consistency"),
            "top10_positive_count": step67_report.get("top10_positive_count"),
        },
        "figure_outputs": fig_outputs,
        "outputs": {
            "external_dataset_audit": str(outdir / "step68b_external_dataset_audit.csv"),
            "validation_candidate_table": str(outdir / "step68b_validation_candidate_table.csv"),
            "weak_validation_evidence": str(outdir / "step68b_weak_validation_evidence.csv"),
            "external_module_scores": str(outdir / "step68b_external_module_scores.csv"),
            "external_module_direction_validation": str(outdir / "step68b_external_module_direction_validation.csv"),
            "external_celltype_direction_validation": str(outdir / "step68b_external_celltype_direction_validation.csv"),
            "validation_tier_summary": str(outdir / "step68b_validation_tier_summary.csv"),
            "future_experimental_plan": str(outdir / "step68b_future_experimental_validation_plan.csv"),
            "manuscript_text": str(outdir / "step68b_manuscript_validation_text.txt"),
            "report_json": str(outdir / "step68b_report.json"),
            "report_txt": str(outdir / "step68b_report.txt"),
        },
        "interpretation_note": (
            "Step68B is a validation-support layer. External direction-level support is not wet-lab validation. "
            "Do not claim experimentally validated therapeutic efficacy."
        ),
    }

    (outdir / "step68b_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    show = [
        "validation_rank", "perturbation_id", "display_name", "module_key",
        "validation_tier", "recommended_claim", "weak_support_score",
        "step67_best_control_rank", "n_matched_genes", "matched_gene_fraction",
        "stage_spearman_rho", "late_vs_acute_consistency_score",
        "acute_induction_consistency_score", "external_direction_supported"
    ]
    show = [c for c in show if c in validation_table.columns]

    lines = []
    lines.append("Step68B external MCAO/stroke validation layer report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Validation table:")
    lines.append(validation_table[show].to_string(index=False))
    lines.append("")
    lines.append("External dataset audit top candidates:")
    if not audit_files.empty:
        lines.append(audit_files.head(80).to_string(index=False))
    else:
        lines.append("No external files discovered.")
    lines.append("")
    lines.append("Manuscript text:")
    lines.append((outdir / "step68b_manuscript_validation_text.txt").read_text(encoding="utf-8"))

    (outdir / "step68b_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step68B")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log(validation_table[show].to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
