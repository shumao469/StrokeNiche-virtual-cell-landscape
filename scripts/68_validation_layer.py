#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
68_validation_layer.py

Step 68: Validation layer for StrokeNiche PerturbMap.

Purpose
-------
Create a manuscript-ready validation/support layer without overstating
"validated" claims.

Validation tiers
----------------
Weak validation:
  - literature/pathway plausibility
  - Step67 known-positive/decoy benchmark support
  - BBB / safety / druggability / experimental-prioritization notes

Medium validation:
  - external MCAO / ischemic stroke scRNA/snRNA/spatial dataset
  - verify whether candidate target modules follow expected injury-to-repair
    direction across conditions/timepoints/states.

Strong validation:
  - not performed here
  - placeholder table for future OGD / co-culture / organotypic slice assays

Inputs
------
Required:
  Step66f manuscript-ready perturbation summary:
    step66f_manuscript_ready_perturbation_summary.csv

Optional:
  Step67 benchmark:
    step67_benchmark_scores.csv
    step67_benchmark_metrics.csv
    step67_topk_enrichment.csv
    step67_direction_reversal_consistency.csv

Optional external validation data:
  --external_expr_csv  expression matrix CSV/TSV
      supported orientations:
        cells/samples x genes
        genes x cells/samples
  --external_meta_csv  metadata CSV/TSV
  --external_h5ad      AnnData h5ad, if anndata is installed

Outputs
-------
outdir/
  step68_validation_candidate_table.csv
  step68_weak_validation_evidence.csv
  step68_external_module_scores.csv
  step68_external_module_direction_validation.csv
  step68_external_celltype_direction_validation.csv
  step68_validation_tier_summary.csv
  step68_future_experimental_validation_plan.csv
  step68_manuscript_validation_text.txt
  step68_report.json
  step68_report.txt
  Fig_Step68_ValidationLayer_annotated.pdf/svg/png
  Fig_Step68_ValidationLayer_clean_no_text.pdf/svg/png

Important interpretation
------------------------
Use wording:
  externally supported
  consistent with public dataset direction
  computationally prioritized
  hypothesis for experimental validation

Do NOT write:
  experimentally validated
  therapeutically validated
  clinically validated
unless wet-lab or independent therapeutic validation is actually performed.
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
DEFAULT_OUT = ROOT / "validation_layer_68"

DEFAULT_EXTERNAL_DIR = Path("/mnt/h/vir/ST/resources/external_validation_stroke")


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
        p = p.strip()
        if not p:
            continue
        g = gene_upper(p)
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


def rankdata_average(x):
    return pd.Series(x).rank(method="average").values


def spearmanr_manual(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    rx = rankdata_average(x[ok].values)
    ry = rankdata_average(y[ok].values)
    if np.std(rx) <= 1e-12 or np.std(ry) <= 1e-12:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# -----------------------------------------------------------------------------
# Candidate modules
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
    """
    Direction expected along injury-to-repair axis.

    +1: module expected to increase during repair/recovery.
    -1: pathological/stress module expected to decrease during repair/recovery.
    """
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

    df["expected_repair_direction"] = df.apply(
        lambda r: expected_repair_direction(r.get("module_key", ""), r.get("perturbation_id", "")),
        axis=1
    )
    df["validation_axis"] = df["module_key"].map(validation_axis_label)

    return df


def load_step67_support(step67_dir):
    step67_dir = Path(step67_dir)
    scores = read_table_auto(step67_dir / "step67_benchmark_scores.csv")
    metrics = read_table_auto(step67_dir / "step67_benchmark_metrics.csv")
    topk = read_table_auto(step67_dir / "step67_topk_enrichment.csv")
    drev = read_table_auto(step67_dir / "step67_direction_reversal_consistency.csv")
    report = read_json_auto(step67_dir / "step67_report.json")
    return scores, metrics, topk, drev, report


def attach_step67_support(candidates, step67_scores):
    df = candidates.copy()
    if step67_scores.empty:
        df["step67_support"] = "not_available"
        df["step67_best_control_rank"] = np.nan
        df["step67_best_control_score"] = np.nan
        return df

    # Map candidate module to best known positive control with same mapped module or related ID.
    rows = []
    for _, r in df.iterrows():
        module = str(r.get("module_key", ""))
        pid = str(r.get("perturbation_id", ""))

        sub = step67_scores.copy()
        if "benchmark_label" in sub.columns:
            sub = sub[sub["benchmark_label"].astype(int).eq(1)].copy()

        if "mapped_step66_module" in sub.columns:
            hit = sub[sub["mapped_step66_module"].astype(str).eq(module)].copy()
        else:
            hit = pd.DataFrame()

        if hit.empty:
            # fallback keyword search
            key = module.replace("_", "").lower()
            hit = sub[sub["control_id"].astype(str).map(lambda x: key in norm_key(x).replace("_", ""))].copy()

        if hit.empty:
            best_rank = np.nan
            best_score = np.nan
            best_id = ""
            support = "no_matched_positive_control"
        else:
            hit["benchmark_rank"] = pd.to_numeric(hit["benchmark_rank"], errors="coerce")
            hit = hit.sort_values("benchmark_rank").head(1)
            best_rank = safe_float(hit["benchmark_rank"].iloc[0])
            best_score = safe_float(hit["benchmark_score"].iloc[0])
            best_id = str(hit["control_id"].iloc[0])
            support = "supported_by_step67_known_positive_benchmark"

        row = r.to_dict()
        row.update({
            "step67_support": support,
            "step67_best_control_id": best_id,
            "step67_best_control_rank": best_rank,
            "step67_best_control_score": best_score,
        })
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Weak validation annotations
# -----------------------------------------------------------------------------

def weak_evidence_database():
    """
    Conservative built-in annotation layer.
    This is not a citation database; it is used to structure manuscript evidence.
    """
    return {
        "synaptic_recovery": {
            "pathway_consistency": "repair/recovery-aligned neuronal and synaptic module",
            "drug_safety_note": "experimental neurorecovery axis; avoid claiming direct druggability without perturbagen evidence",
            "bbb_note": "BBB penetration depends on perturbagen modality",
            "wetlab_priority": "medium-high",
            "risk_note": "requires neuronal viability and excitotoxicity controls",
        },
        "repair_ecm": {
            "pathway_consistency": "repair-associated ECM remodeling; may be beneficial or fibrotic depending on timing",
            "drug_safety_note": "context-dependent; excessive ECM activation may worsen scarring",
            "bbb_note": "local tissue effects and timing should be evaluated",
            "wetlab_priority": "medium-high",
            "risk_note": "interpret as repair-permissive ECM, not generic fibrosis activation",
        },
        "ferroptosis": {
            "pathway_consistency": "injury-associated redox/ferroptosis module expected to decline during repair",
            "drug_safety_note": "redox modulation can have broad off-target effects",
            "bbb_note": "BBB and dose window are critical",
            "wetlab_priority": "high",
            "risk_note": "requires ROS/lipid peroxidation and cell viability assays",
        },
        "astrocyte_reactive": {
            "pathway_consistency": "reactive astrocyte stress module expected to be injury-enriched",
            "drug_safety_note": "astrocyte reactivity has protective and harmful subcomponents",
            "bbb_note": "cell-type specificity is critical",
            "wetlab_priority": "medium",
            "risk_note": "avoid blanket suppression of astrocyte function",
        },
        "bbb_leakage": {
            "pathway_consistency": "vascular leakage module expected to be injury-enriched",
            "drug_safety_note": "vascular modulators need hemorrhage/edema risk assessment",
            "bbb_note": "directly related to barrier status",
            "wetlab_priority": "medium",
            "risk_note": "requires barrier integrity and permeability assays",
        },
        "barrier_stability": {
            "pathway_consistency": "repair-aligned endothelial barrier stabilization module",
            "drug_safety_note": "vascular normalization should be separated from angiogenic overactivation",
            "bbb_note": "direct BBB relevance",
            "wetlab_priority": "medium",
            "risk_note": "requires endothelial permeability and tight-junction readouts",
        },
        "microglia_inflammatory": {
            "pathway_consistency": "injury-associated microglial inflammatory module",
            "drug_safety_note": "microglia modulation can impair debris clearance if over-suppressed",
            "bbb_note": "immune-cell and CNS access depend on perturbagen",
            "wetlab_priority": "medium",
            "risk_note": "requires cytokine and phagocytosis controls",
        },
        "inflammation": {
            "pathway_consistency": "general inflammatory module expected to be injury-enriched",
            "drug_safety_note": "systemic immunosuppression risk",
            "bbb_note": "peripheral versus CNS compartment should be separated",
            "wetlab_priority": "medium",
            "risk_note": "requires cell-type resolved inflammatory readouts",
        },
        "hypoxia": {
            "pathway_consistency": "hypoxia/metabolic stress module expected to be injury-enriched",
            "drug_safety_note": "metabolic modulation may have broad systemic effects",
            "bbb_note": "tissue oxygenation and vascular confounding should be evaluated",
            "wetlab_priority": "medium",
            "risk_note": "requires oxygen/glucose and mitochondrial readouts",
        },
        "endothelial_barrier_fragility": {
            "pathway_consistency": "injury-associated endothelial fragility module",
            "drug_safety_note": "vascular off-target and bleeding risk must be considered",
            "bbb_note": "direct vascular/BBB relevance",
            "wetlab_priority": "medium",
            "risk_note": "requires endothelial and hemorrhagic risk assays",
        },
    }


def build_weak_validation(candidates, step67_metrics, step67_topk, step67_drev):
    db = weak_evidence_database()
    rows = []

    # Overall Step67 support summary.
    auroc = np.nan
    auprc = np.nan
    if not step67_metrics.empty and "metric" in step67_metrics.columns:
        m = step67_metrics
        if "AUROC_positive_vs_decoy" in set(m["metric"]):
            auroc = safe_float(m.loc[m["metric"].eq("AUROC_positive_vs_decoy"), "value"].iloc[0])
        if "AUPRC_average_precision_positive_vs_decoy" in set(m["metric"]):
            auprc = safe_float(m.loc[m["metric"].eq("AUPRC_average_precision_positive_vs_decoy"), "value"].iloc[0])

    top10_enrich = np.nan
    if not step67_topk.empty and "top_k" in step67_topk.columns:
        hit = step67_topk[pd.to_numeric(step67_topk["top_k"], errors="coerce").eq(10)]
        if not hit.empty:
            top10_enrich = safe_float(hit["fold_enrichment"].iloc[0])

    direction_consistency = np.nan
    if not step67_drev.empty and "consistent" in step67_drev.columns:
        direction_consistency = float(step67_drev["consistent"].astype(str).str.lower().isin(["true", "1", "yes"]).mean())

    for _, r in candidates.iterrows():
        module = str(r.get("module_key", ""))
        e = db.get(module, {})

        step67_rank = safe_float(r.get("step67_best_control_rank"))
        step67_score = safe_float(r.get("step67_best_control_score"))

        pathway_score = 1.0 if e else 0.5
        step67_score_norm = 0.0
        if np.isfinite(step67_rank):
            step67_score_norm = max(0.0, 1.0 - min(step67_rank, 50) / 50.0)

        weak_support_score = (
            0.40 * pathway_score
            + 0.40 * step67_score_norm
            + 0.10 * (1.0 if np.isfinite(auroc) and auroc >= 0.75 else 0.5)
            + 0.10 * (1.0 if np.isfinite(direction_consistency) and direction_consistency >= 0.8 else 0.5)
        )

        rows.append({
            "perturbation_id": r.get("perturbation_id", ""),
            "display_name": r.get("display_name", r.get("perturbation_id", "")),
            "module_key": module,
            "validation_level": "weak_support",
            "weak_support_score": weak_support_score,
            "pathway_consistency": e.get("pathway_consistency", "module-specific note unavailable"),
            "drug_safety_note": e.get("drug_safety_note", "annotation unavailable"),
            "bbb_note": e.get("bbb_note", "annotation unavailable"),
            "wetlab_priority": e.get("wetlab_priority", "medium"),
            "risk_note": e.get("risk_note", "annotation unavailable"),
            "step67_support": r.get("step67_support", ""),
            "step67_best_control_id": r.get("step67_best_control_id", ""),
            "step67_best_control_rank": step67_rank,
            "step67_best_control_score": step67_score,
            "step67_AUROC_positive_vs_decoy": auroc,
            "step67_AUPRC_positive_vs_decoy": auprc,
            "step67_top10_fold_enrichment": top10_enrich,
            "step67_direction_reversal_consistency": direction_consistency,
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# External dataset loading and module scoring
# -----------------------------------------------------------------------------

def auto_discover_external_files(external_dir):
    external_dir = Path(external_dir)
    if not external_dir.exists():
        return "", ""

    files = list(external_dir.glob("*"))
    expr_candidates = []
    meta_candidates = []

    for p in files:
        if not p.is_file():
            continue
        low = p.name.lower()
        if not low.endswith((".csv", ".tsv", ".txt")):
            continue
        if any(k in low for k in ["expr", "expression", "counts", "matrix", "data"]):
            expr_candidates.append(p)
        if any(k in low for k in ["meta", "obs", "anno", "annotation", "cellinfo", "sample"]):
            meta_candidates.append(p)

    expr = str(expr_candidates[0]) if expr_candidates else ""
    meta = str(meta_candidates[0]) if meta_candidates else ""
    return expr, meta


def load_h5ad(path):
    try:
        import anndata as ad
    except Exception as e:
        raise RuntimeError(f"anndata not installed; cannot read h5ad: {e}")

    a = ad.read_h5ad(path)

    X = a.X
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass

    expr = pd.DataFrame(X, index=a.obs_names.astype(str), columns=a.var_names.astype(str))
    meta = a.obs.copy()
    meta.index = meta.index.astype(str)
    meta["obs_name"] = meta.index
    return expr, meta


def read_external_expr_meta(expr_csv="", meta_csv="", h5ad="", external_dir=""):
    if h5ad:
        expr, meta = load_h5ad(h5ad)
        return expr, meta, "h5ad"

    if not expr_csv and external_dir:
        expr_csv, auto_meta = auto_discover_external_files(external_dir)
        if not meta_csv:
            meta_csv = auto_meta

    if not expr_csv:
        return pd.DataFrame(), pd.DataFrame(), "not_provided"

    raw = read_table_auto(expr_csv, required=True)

    # If first column looks like row names, use as index.
    if raw.shape[1] > 1:
        first = raw.columns[0]
        first_vals = raw[first].astype(str)
        numeric_fraction = pd.to_numeric(first_vals, errors="coerce").notna().mean()
        if numeric_fraction < 0.20:
            raw = raw.set_index(first)

    # Orientation detection by overlap with module genes.
    catalog_genes = set()
    for genes in module_gene_catalog().values():
        catalog_genes.update(genes)

    index_overlap = len(set(map(gene_upper, raw.index.astype(str))) & catalog_genes)
    col_overlap = len(set(map(gene_upper, raw.columns.astype(str))) & catalog_genes)

    if index_overlap > col_overlap:
        # genes x cells/samples -> transpose
        expr = raw.T.copy()
        expr.columns = [gene_upper(c) for c in expr.columns]
    else:
        expr = raw.copy()
        expr.columns = [gene_upper(c) for c in expr.columns]

    expr.index = expr.index.astype(str)

    # Convert numeric.
    expr = expr.apply(pd.to_numeric, errors="coerce")
    expr = expr.dropna(axis=1, how="all")
    expr = expr.dropna(axis=0, how="all")

    # Metadata.
    meta = pd.DataFrame(index=expr.index)
    meta["obs_name"] = meta.index

    if meta_csv:
        m = read_table_auto(meta_csv, required=False)
        if not m.empty:
            obs_col = detect_obs_col(m)
            if obs_col:
                m[obs_col] = m[obs_col].astype(str)
                meta = meta.merge(m, left_on="obs_name", right_on=obs_col, how="left")
                if obs_col != "obs_name" and "obs_name_x" in meta.columns:
                    meta["obs_name"] = meta["obs_name_x"]
            else:
                # If same length, align by order.
                if len(m) == len(expr):
                    m = m.copy()
                    m.index = expr.index
                    m["obs_name"] = expr.index
                    meta = m

    return expr, meta, "csv"


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
        "sample_group", "status"
    ]
    for c in priority:
        if c in meta.columns:
            vals = meta[c].astype(str).map(stage_to_numeric)
            if vals.notna().mean() > 0.30 and vals.nunique(dropna=True) >= 2:
                return c

    for c in meta.columns:
        nk = norm_key(c)
        if any(k in nk for k in ["stage", "time", "day", "condition", "group", "region", "state"]):
            vals = meta[c].astype(str).map(stage_to_numeric)
            if vals.notna().mean() > 0.30 and vals.nunique(dropna=True) >= 2:
                return c

    return ""


def detect_celltype_col(meta):
    if meta.empty:
        return ""
    priority = [
        "celltype", "cell_type", "cell_type_major", "annotation",
        "cell_annotation", "predicted_celltype", "cluster", "subcluster"
    ]
    for c in priority:
        if c in meta.columns and meta[c].nunique(dropna=True) >= 2:
            return c
    for c in meta.columns:
        nk = norm_key(c)
        if any(k in nk for k in ["celltype", "cell_type", "annotation", "cluster"]):
            if meta[c].nunique(dropna=True) >= 2:
                return c
    return ""


def stage_to_numeric(x):
    s = str(x).lower().strip()
    s2 = re.sub(r"[^a-z0-9]+", "_", s)

    if s in ["nan", "none", ""]:
        return np.nan

    # control / sham / normal.
    if any(k in s2 for k in ["sham", "control", "normal", "naive", "contralateral", "nc"]):
        return 0.0

    # acute injury.
    if re.search(r"(^|_)d?1($|_)", s2) or "day1" in s2 or "24h" in s2 or "acute" in s2:
        return 1.0
    if any(k in s2 for k in ["mcao", "ischemia", "ischaemia", "stroke", "lesion_core", "core"]):
        return 1.0

    # subacute/peri.
    if re.search(r"(^|_)d?3($|_)", s2) or "day3" in s2 or "72h" in s2 or "subacute" in s2:
        return 2.0
    if "peri" in s2:
        return 2.0

    # repair/remote.
    if re.search(r"(^|_)d?7($|_)", s2) or "day7" in s2 or "repair" in s2 or "recovery" in s2:
        return 3.0
    if "remote" in s2:
        return 3.0

    # later chronic.
    if re.search(r"(^|_)d?14($|_)", s2) or "day14" in s2 or "chronic" in s2:
        return 4.0
    if re.search(r"(^|_)d?28($|_)", s2) or "day28" in s2:
        return 5.0

    # numeric fallback.
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
    vals = expr.values
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return expr
    q99 = np.nanpercentile(finite, 99)
    if q99 > 50:
        return np.log1p(expr)
    return expr


def zscore_columns(expr):
    mu = expr.mean(axis=0)
    sd = expr.std(axis=0)
    sd = sd.replace(0, np.nan)
    z = (expr - mu) / sd
    return z.fillna(0)


def compute_external_module_scores(expr, meta, candidates):
    if expr.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {
            "external_available": False,
            "reason": "no_external_expression_matrix_provided"
        }

    expr = maybe_log1p(expr)
    expr.columns = [gene_upper(c) for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()].copy()

    z = zscore_columns(expr)

    meta = meta.copy()
    if "obs_name" not in meta.columns:
        meta["obs_name"] = z.index.astype(str)

    # Align meta.
    meta["obs_name"] = meta["obs_name"].astype(str)
    meta = meta.drop_duplicates("obs_name")
    meta = pd.DataFrame({"obs_name": z.index.astype(str)}).merge(meta, on="obs_name", how="left")

    stage_col = detect_stage_col(meta)
    celltype_col = detect_celltype_col(meta)

    if stage_col:
        meta["injury_repair_stage_numeric"] = meta[stage_col].map(stage_to_numeric)
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

        if len(matched) == 0:
            module_score = pd.Series(np.nan, index=z.index)
        else:
            module_score = z[matched].mean(axis=1)

        tmp = meta.copy()
        tmp["perturbation_id"] = pid
        tmp["module_key"] = module
        tmp["module_score"] = module_score.values
        tmp["n_module_genes"] = len(genes)
        tmp["n_matched_genes"] = len(matched)
        tmp["matched_genes"] = ";".join(matched)
        tmp["external_stage_col"] = stage_col
        tmp["external_celltype_col"] = celltype_col
        score_rows.append(tmp)

        # Overall direction validation.
        valid = tmp[["module_score", "injury_repair_stage_numeric"]].dropna()
        if valid["injury_repair_stage_numeric"].nunique() >= 2 and len(valid) >= 10:
            rho = spearmanr_manual(valid["injury_repair_stage_numeric"], valid["module_score"])
            stages = sorted(valid["injury_repair_stage_numeric"].dropna().unique())
            early = stages[0]
            late = stages[-1]
            early_mean = valid.loc[valid["injury_repair_stage_numeric"].eq(early), "module_score"].mean()
            late_mean = valid.loc[valid["injury_repair_stage_numeric"].eq(late), "module_score"].mean()
            late_minus_early = late_mean - early_mean
        else:
            rho = np.nan
            early = np.nan
            late = np.nan
            early_mean = np.nan
            late_mean = np.nan
            late_minus_early = np.nan

        expected_dir = safe_float(r.get("expected_repair_direction"))
        if np.isfinite(expected_dir) and np.isfinite(rho):
            consistency_score = expected_dir * rho
            consistent = consistency_score > 0
        else:
            consistency_score = np.nan
            consistent = False

        summary_rows.append({
            "perturbation_id": pid,
            "display_name": r.get("display_name", pid),
            "module_key": module,
            "validation_axis": r.get("validation_axis", module),
            "expected_repair_direction": expected_dir,
            "n_module_genes": len(genes),
            "n_matched_genes": len(matched),
            "matched_gene_fraction": len(matched) / max(len(genes), 1),
            "matched_genes": ";".join(matched),
            "external_stage_col": stage_col,
            "external_celltype_col": celltype_col,
            "n_external_observations": int(len(tmp)),
            "n_stage_valid_observations": int(len(valid)),
            "n_unique_stages": int(valid["injury_repair_stage_numeric"].nunique()) if not valid.empty else 0,
            "stage_spearman_rho": rho,
            "expected_direction_consistency_score": consistency_score,
            "direction_consistent_with_injury_to_repair": bool(consistent),
            "early_stage_numeric": early,
            "late_stage_numeric": late,
            "early_stage_mean_module_score": early_mean,
            "late_stage_mean_module_score": late_mean,
            "late_minus_early_module_score": late_minus_early,
        })

        # Celltype-level validation.
        if celltype_col and valid.shape[0] > 0:
            tmp2 = tmp.dropna(subset=["module_score", "injury_repair_stage_numeric", celltype_col]).copy()
            for ct, sub in tmp2.groupby(celltype_col):
                if len(sub) < 10 or sub["injury_repair_stage_numeric"].nunique() < 2:
                    continue
                rho_ct = spearmanr_manual(sub["injury_repair_stage_numeric"], sub["module_score"])
                cons_ct = expected_dir * rho_ct if np.isfinite(expected_dir) and np.isfinite(rho_ct) else np.nan
                celltype_rows.append({
                    "perturbation_id": pid,
                    "module_key": module,
                    "celltype": ct,
                    "n_observations": int(len(sub)),
                    "n_unique_stages": int(sub["injury_repair_stage_numeric"].nunique()),
                    "stage_spearman_rho": rho_ct,
                    "expected_direction_consistency_score": cons_ct,
                    "direction_consistent": bool(cons_ct > 0) if np.isfinite(cons_ct) else False,
                    "n_matched_genes": len(matched),
                    "matched_genes": ";".join(matched),
                })

    score_df = pd.concat(score_rows, ignore_index=True) if score_rows else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    celltype_df = pd.DataFrame(celltype_rows)

    audit = {
        "external_available": True,
        "n_observations": int(expr.shape[0]),
        "n_genes": int(expr.shape[1]),
        "stage_col": stage_col,
        "celltype_col": celltype_col,
        "n_candidates_scored": int(len(candidates)),
        "n_candidates_with_matched_genes": int((summary_df["n_matched_genes"] > 0).sum()) if not summary_df.empty else 0,
        "n_candidates_direction_consistent": int(summary_df["direction_consistent_with_injury_to_repair"].sum()) if not summary_df.empty else 0,
    }

    return score_df, summary_df, celltype_df, audit


# -----------------------------------------------------------------------------
# Validation tier and future experimental plan
# -----------------------------------------------------------------------------

def assign_validation_tier(row):
    weak = safe_float(row.get("weak_support_score"), 0.0)
    external_available = bool(row.get("external_available", False))
    direction_consistent = bool(row.get("direction_consistent_with_injury_to_repair", False))
    matched_frac = safe_float(row.get("matched_gene_fraction"), 0.0)

    if external_available and direction_consistent and matched_frac >= 0.25:
        return "medium_external_direction_support"

    if external_available and matched_frac > 0:
        return "external_dataset_inconclusive_or_direction_mismatch"

    if weak >= 0.70:
        return "weak_pathway_and_benchmark_support"

    return "weak_or_insufficient_support"


def build_validation_candidate_table(candidates, weak, external_summary, external_audit):
    df = candidates.copy()

    keep_weak = [
        "perturbation_id", "weak_support_score", "pathway_consistency",
        "drug_safety_note", "bbb_note", "wetlab_priority", "risk_note",
        "step67_best_control_id", "step67_best_control_rank",
        "step67_AUROC_positive_vs_decoy", "step67_top10_fold_enrichment",
        "step67_direction_reversal_consistency"
    ]
    weak2 = weak[[c for c in keep_weak if c in weak.columns]].copy()
    df = df.merge(weak2, on="perturbation_id", how="left")

    if not external_summary.empty:
        ext_cols = [
            "perturbation_id", "n_module_genes", "n_matched_genes",
            "matched_gene_fraction", "matched_genes", "external_stage_col",
            "external_celltype_col", "n_external_observations",
            "n_unique_stages", "stage_spearman_rho",
            "expected_direction_consistency_score",
            "direction_consistent_with_injury_to_repair",
            "early_stage_mean_module_score",
            "late_stage_mean_module_score",
            "late_minus_early_module_score"
        ]
        ext2 = external_summary[[c for c in ext_cols if c in external_summary.columns]].copy()
        df = df.merge(ext2, on="perturbation_id", how="left")
        df["external_available"] = True
    else:
        df["external_available"] = False
        df["n_matched_genes"] = np.nan
        df["matched_gene_fraction"] = np.nan
        df["direction_consistent_with_injury_to_repair"] = False

    df["validation_tier"] = df.apply(assign_validation_tier, axis=1)

    df["recommended_claim"] = df["validation_tier"].map({
        "medium_external_direction_support": "externally supported directionality; not experimental validation",
        "external_dataset_inconclusive_or_direction_mismatch": "external dataset available but direction support is inconclusive",
        "weak_pathway_and_benchmark_support": "pathway/benchmark-supported computational hypothesis",
        "weak_or_insufficient_support": "exploratory hypothesis requiring further validation",
    }).fillna("exploratory hypothesis requiring further validation")

    # Preserve Step66/66f conservative conclusion.
    df["therapeutic_validation_claim_allowed"] = False
    df["allowed_language"] = (
        "computationally prioritized / internally benchmarked / externally direction-supported if applicable"
    )
    df["forbidden_language"] = (
        "experimentally validated therapeutic candidate; clinically validated target"
    )

    sort_cols = []
    if "validation_tier" in df.columns:
        tier_order = {
            "medium_external_direction_support": 1,
            "weak_pathway_and_benchmark_support": 2,
            "external_dataset_inconclusive_or_direction_mismatch": 3,
            "weak_or_insufficient_support": 4,
        }
        df["validation_tier_order"] = df["validation_tier"].map(tier_order).fillna(9)
        sort_cols.append("validation_tier_order")
    if "weak_support_score" in df.columns:
        sort_cols.append("weak_support_score")
    if "s_overall_observed" in df.columns:
        sort_cols.append("s_overall_observed")

    if sort_cols:
        ascending = [True] + [False] * (len(sort_cols) - 1)
        df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    df["validation_rank"] = np.arange(1, len(df) + 1)

    return df


def future_experimental_plan(validation_table):
    rows = []
    priority_modules = [
        "synaptic_recovery",
        "repair_ecm",
        "ferroptosis",
        "spp1_cd44",
        "ccl2_ccr2",
        "vegfa_flt1",
    ]

    for _, r in validation_table.iterrows():
        module = str(r.get("module_key", ""))
        pid = str(r.get("perturbation_id", ""))
        if module not in priority_modules and r.get("validation_rank", 999) > 5:
            continue

        if module == "ferroptosis":
            assay = "OGD/R neuron-glia culture or organotypic slice; lipid ROS, GPX4/SLC7A11/FTH1, cell viability"
            readout = "reduced lipid peroxidation and core-like stress without toxicity"
        elif module == "repair_ecm":
            assay = "organotypic brain slice or astrocyte-microglia-endothelial co-culture; ECM remodeling markers"
            readout = "repair-permissive ECM markers without excessive scarring/fibrosis signature"
        elif module == "synaptic_recovery":
            assay = "OGD/R neuronal culture or organotypic slice; synaptic marker recovery"
            readout = "increased MAP2/SYN1/SNAP25/DLG4 and improved viability"
        elif "bbb" in module or "barrier" in module:
            assay = "microglia-endothelial-astrocyte BBB co-culture; TEER/permeability/tight junctions"
            readout = "improved CLDN5/OCLN/TJP1 and reduced leakage markers"
        elif "microglia" in module or "inflammation" in module:
            assay = "microglia-endothelial/astrocyte co-culture after OGD/R; cytokine and chemotaxis readouts"
            readout = "reduced CCL2/CCR2/TNF/IL1B without impaired debris-clearance markers"
        else:
            assay = "OGD/R or organotypic brain slice perturbation assay"
            readout = "directional rescue of module score and reduced core-like state probability"

        rows.append({
            "perturbation_id": pid,
            "module_key": module,
            "suggested_validation_strength": "strong_wetlab_followup",
            "recommended_model": assay,
            "primary_readout": readout,
            "secondary_readout": "latent/state shift, RRHO-like DEG overlap, module score reversal, toxicity/safety control",
            "interpretation_if_successful": "experimental support for computational prioritization, still requiring dose/time/cell-type validation",
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting
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


def plot_validation(validation_table, weak, external_summary, outdir, dpi=600):
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step68_ValidationLayer_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.35)

        # A: validation tier counts
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

        # B: weak support scores
        axB = fig.add_subplot(gs[0, 1])
        df = validation_table.sort_values("validation_rank")
        y = np.arange(len(df))[::-1]
        vals = pd.to_numeric(df["weak_support_score"], errors="coerce").fillna(0).values[::-1]
        axB.barh(y, vals)
        if annotate:
            axB.set_yticks(y)
            axB.set_yticklabels(df["display_name"].astype(str).values[::-1], fontsize=7)
            axB.set_xlabel("Weak support score")
            axB.set_title("B | Pathway / benchmark support", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axB)

        # C: external direction consistency
        axC = fig.add_subplot(gs[1, 0])
        if external_summary is not None and not external_summary.empty:
            ext = external_summary.copy()
            ext["expected_direction_consistency_score"] = pd.to_numeric(
                ext["expected_direction_consistency_score"], errors="coerce"
            )
            ext = ext.sort_values("expected_direction_consistency_score", ascending=False)
            y2 = np.arange(len(ext))[::-1]
            axC.barh(y2, ext["expected_direction_consistency_score"].fillna(0).values[::-1])
            if annotate:
                axC.axvline(0, linestyle="--", linewidth=0.8)
                axC.set_yticks(y2)
                axC.set_yticklabels(ext["display_name"].astype(str).values[::-1], fontsize=7)
                axC.set_xlabel("Expected-direction consistency score")
                axC.set_title("C | External injury-to-repair module direction", loc="left", fontsize=12, fontweight="bold")
            else:
                strip_text(axC)
        else:
            axC.text(0.5, 0.5, "No external dataset provided", ha="center", va="center")
            if annotate:
                axC.set_title("C | External validation not performed", loc="left", fontsize=12, fontweight="bold")
            else:
                strip_text(axC)

        # D: observed Step66 score vs validation support
        axD = fig.add_subplot(gs[1, 1])
        x = pd.to_numeric(validation_table.get("s_overall_observed", pd.Series(np.nan, index=validation_table.index)), errors="coerce")
        yv = pd.to_numeric(validation_table["weak_support_score"], errors="coerce")
        axD.scatter(x, yv, s=55, alpha=0.8)
        if annotate:
            for _, r in validation_table.head(6).iterrows():
                axD.text(
                    safe_float(r.get("s_overall_observed")),
                    safe_float(r.get("weak_support_score")),
                    str(r.get("perturbation_id"))[:24],
                    fontsize=7,
                )
            axD.set_xlabel("Step66 model-derived perturbation score")
            axD.set_ylabel("Weak validation support score")
            axD.set_title("D | Perturbation score versus support evidence", loc="left", fontsize=12, fontweight="bold")
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
                "Step68 | Validation layer for StrokeNiche perturbation hypotheses",
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


# -----------------------------------------------------------------------------
# Text output
# -----------------------------------------------------------------------------

def write_manuscript_text(validation_table, external_audit, outdir):
    n_medium = int(validation_table["validation_tier"].eq("medium_external_direction_support").sum())
    n_weak = int(validation_table["validation_tier"].eq("weak_pathway_and_benchmark_support").sum())
    n_total = int(len(validation_table))

    top = validation_table.head(4)

    lines = []
    lines.append("Step68 validation-layer manuscript text")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Recommended Results paragraph:")
    if external_audit.get("external_available", False):
        lines.append(
            f"We next implemented a validation-layer analysis integrating pathway plausibility, "
            f"known-positive/decoy benchmark support and an external injury-to-repair expression dataset. "
            f"Among {n_total} candidate perturbation modules, {n_medium} showed directionally consistent "
            f"external module behavior along the injury-to-repair axis, whereas the remaining candidates were "
            f"retained as pathway- and benchmark-supported computational hypotheses."
        )
    else:
        lines.append(
            f"We next implemented a validation-layer analysis integrating pathway plausibility and "
            f"known-positive/decoy benchmark support. No external MCAO/stroke expression dataset was provided "
            f"for this run; therefore, the analysis is interpreted as weak validation support rather than "
            f"external validation. Candidate perturbations are retained as computational hypotheses for "
            f"experimental prioritization."
        )

    lines.append("")
    lines.append("Leading candidates and support level:")
    for _, r in top.iterrows():
        lines.append(
            f"- {r.get('display_name', r.get('perturbation_id'))}: "
            f"{r.get('validation_tier')}; claim: {r.get('recommended_claim')}."
        )

    lines.append("")
    lines.append("Required conservative wording:")
    lines.append(
        "Use: internally benchmarked, pathway-consistent, externally direction-supported if applicable, "
        "computationally prioritized, experimental-prioritization hypothesis."
    )
    lines.append(
        "Avoid: experimentally validated, clinically validated, therapeutic candidate confirmed, "
        "validated treatment target."
    )

    lines.append("")
    lines.append("Suggested future wet-lab validation:")
    lines.append(
        "Prioritize 2–3 perturbations for OGD/R or organotypic brain-slice testing, with readouts including "
        "module score reversal, latent/state shift, RRHO-like DEG overlap, toxicity/safety and BBB/cell-type "
        "specificity controls."
    )

    (Path(outdir) / "step68_manuscript_validation_text.txt").write_text(
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

    ap.add_argument("--external_expr_csv", default="")
    ap.add_argument("--external_meta_csv", default="")
    ap.add_argument("--external_h5ad", default="")
    ap.add_argument("--external_dir", default=str(DEFAULT_EXTERNAL_DIR))

    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step68 validation layer")
    log("=" * 100)
    log(f"outdir={outdir}")

    # Load candidates.
    candidates = load_step66f(args.step66f_dir)

    # Load Step67 support.
    step67_scores, step67_metrics, step67_topk, step67_drev, step67_report = load_step67_support(args.step67_dir)
    candidates = attach_step67_support(candidates, step67_scores)

    # Weak validation.
    weak = build_weak_validation(candidates, step67_metrics, step67_topk, step67_drev)

    # External validation.
    expr, meta, external_mode = read_external_expr_meta(
        expr_csv=args.external_expr_csv,
        meta_csv=args.external_meta_csv,
        h5ad=args.external_h5ad,
        external_dir=args.external_dir,
    )

    external_scores, external_summary, external_celltype, external_audit = compute_external_module_scores(
        expr=expr,
        meta=meta,
        candidates=candidates,
    )
    external_audit["external_input_mode"] = external_mode
    external_audit["external_expr_csv"] = args.external_expr_csv
    external_audit["external_meta_csv"] = args.external_meta_csv
    external_audit["external_h5ad"] = args.external_h5ad
    external_audit["external_dir"] = args.external_dir

    # Final candidate validation table.
    validation_table = build_validation_candidate_table(
        candidates=candidates,
        weak=weak,
        external_summary=external_summary,
        external_audit=external_audit,
    )

    validation_tier_summary = (
        validation_table["validation_tier"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "validation_tier", "validation_tier": "n_candidates"})
    )

    future_plan = future_experimental_plan(validation_table)

    # Save tables.
    validation_table.to_csv(outdir / "step68_validation_candidate_table.csv", index=False)
    weak.to_csv(outdir / "step68_weak_validation_evidence.csv", index=False)
    external_scores.to_csv(outdir / "step68_external_module_scores.csv", index=False)
    external_summary.to_csv(outdir / "step68_external_module_direction_validation.csv", index=False)
    external_celltype.to_csv(outdir / "step68_external_celltype_direction_validation.csv", index=False)
    validation_tier_summary.to_csv(outdir / "step68_validation_tier_summary.csv", index=False)
    future_plan.to_csv(outdir / "step68_future_experimental_validation_plan.csv", index=False)

    # Text and figures.
    write_manuscript_text(validation_table, external_audit, outdir)
    fig_outputs = plot_validation(validation_table, weak, external_summary, outdir, dpi=args.dpi)

    # Report.
    report = {
        "status": "ok",
        "analysis_name": "Step68 validation layer",
        "step66f_dir": args.step66f_dir,
        "step67_dir": args.step67_dir,
        "outdir": str(outdir),
        "n_candidates": int(len(validation_table)),
        "validation_tier_counts": validation_table["validation_tier"].value_counts().to_dict(),
        "external_validation": external_audit,
        "step67_report_summary": {
            "AUROC_positive_vs_decoy": step67_report.get("AUROC_positive_vs_decoy"),
            "AUPRC_average_precision_positive_vs_decoy": step67_report.get("AUPRC_average_precision_positive_vs_decoy"),
            "direction_reversal_consistency": step67_report.get("direction_reversal_consistency"),
            "top10_positive_count": step67_report.get("top10_positive_count"),
        },
        "figure_outputs": fig_outputs,
        "outputs": {
            "validation_candidate_table": str(outdir / "step68_validation_candidate_table.csv"),
            "weak_validation_evidence": str(outdir / "step68_weak_validation_evidence.csv"),
            "external_module_scores": str(outdir / "step68_external_module_scores.csv"),
            "external_module_direction_validation": str(outdir / "step68_external_module_direction_validation.csv"),
            "external_celltype_direction_validation": str(outdir / "step68_external_celltype_direction_validation.csv"),
            "validation_tier_summary": str(outdir / "step68_validation_tier_summary.csv"),
            "future_experimental_plan": str(outdir / "step68_future_experimental_validation_plan.csv"),
            "manuscript_text": str(outdir / "step68_manuscript_validation_text.txt"),
            "report_json": str(outdir / "step68_report.json"),
            "report_txt": str(outdir / "step68_report.txt"),
        },
        "interpretation_note": (
            "Step68 is a validation/support layer. Without wet-lab experiments, do not use the word "
            "'validated' for therapeutic efficacy. If an external dataset is provided, use 'externally "
            "direction-supported' rather than 'experimentally validated'."
        ),
    }

    (outdir / "step68_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    lines = []
    lines.append("Step68 validation layer report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Validation table preview:")
    show = [
        "validation_rank", "perturbation_id", "display_name", "module_key",
        "validation_tier", "recommended_claim", "weak_support_score",
        "step67_best_control_rank", "n_matched_genes",
        "stage_spearman_rho", "expected_direction_consistency_score",
        "direction_consistent_with_injury_to_repair"
    ]
    show = [c for c in show if c in validation_table.columns]
    lines.append(validation_table[show].to_string(index=False))
    lines.append("")
    lines.append("Manuscript text:")
    lines.append((outdir / "step68_manuscript_validation_text.txt").read_text(encoding="utf-8"))

    (outdir / "step68_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step68")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log(validation_table[show].to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
