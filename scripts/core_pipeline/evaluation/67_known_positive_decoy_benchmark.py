#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
67_known_positive_decoy_benchmark.py

Purpose
-------
Step 67: Known-positive / decoy benchmark for StrokeNiche PerturbMap.

This is a UNAGI-like benchmark layer, but not a direct comparison to UNAGI,
scGPT, Geneformer, scVI or scGen.

Benchmark design
----------------
Positive controls:
  - SPP1-CD44 axis
  - CCL2-CCR2 / CCL2-ACKR1 axis
  - VEGFA-FLT1 / VEGFA-KDR axis
  - TNF inflammatory axis
  - TGFB1 repair/ECM axis
  - TREM2-APOE microglia axis
  - ferroptosis/redox gene set

Negative / decoy controls:
  - housekeeping genes
  - random gene sets matched by target-set size
  - shuffled target sets
  - direction-reversed perturbations

Main readouts:
  - AUROC
  - AUPRC / average precision
  - top-k enrichment
  - rank stability by bootstrap over tracks
  - direction reversal consistency

Important interpretation
------------------------
This benchmark evaluates whether the current StrokeNiche perturbation scoring
scheme ranks curated stroke-relevant controls above decoy/random controls.
It is a calibration/sanity benchmark, not external therapeutic validation.
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
DEFAULT_STEP66E_RERUN = ROOT / "track_level_perturbation_fdr_66e_probability_simplex_rerun66c/rerun66c_probability_simplex"
DEFAULT_STEP65C = ROOT / "track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata"
DEFAULT_STEP65E = ROOT / "track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered"
DEFAULT_STEP64C = ROOT / "dynamics_graph_64c_refined_labels_state_celltype"

DEFAULT_OUT = ROOT / "known_positive_decoy_benchmark_67"


HOUSEKEEPING = {
    "ACTB", "B2M", "GAPDH", "HPRT", "HPRT1", "MALAT1", "PPIA", "RPLP0",
    "RPS18", "RPL13A", "TUBA1A", "TUBA1B", "TUBB", "EEF1A1", "EEF2",
    "PGK1", "LDHA", "LDHB"
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(x):
    print(x, flush=True)


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def gene_upper(x):
    return str(x).strip().upper()


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


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
        if re.match(r"^[A-Z][A-Z0-9\.\-]{1,24}$", g):
            genes.append(g)
    return list(dict.fromkeys(genes))


def read_csv_optional(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


def read_json_optional(path):
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


def jaccard(a, b):
    a = set(a)
    b = set(b)
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def target_relevance(targets, markers):
    t = set(gene_upper(x) for x in targets if x)
    m = set(gene_upper(x) for x in markers if x)
    if not t or not m:
        return {
            "overlap_n": 0,
            "target_fraction": 0.0,
            "marker_fraction": 0.0,
            "jaccard": 0.0,
            "relevance": 0.0,
            "overlap_genes": "",
        }
    ov = t & m
    target_fraction = len(ov) / max(len(t), 1)
    marker_fraction = len(ov) / max(len(m), 1)
    jac = len(ov) / max(len(t | m), 1)
    relevance = 0.55 * target_fraction + 0.30 * marker_fraction + 0.15 * jac
    return {
        "overlap_n": len(ov),
        "target_fraction": target_fraction,
        "marker_fraction": marker_fraction,
        "jaccard": jac,
        "relevance": relevance,
        "overlap_genes": ";".join(sorted(ov)),
    }


# -----------------------------------------------------------------------------
# Module catalogs and controls
# -----------------------------------------------------------------------------

def module_gene_catalog():
    cat = {
        "spp1_cd44": [
            "SPP1", "CD44", "ITGAV", "ITGB1", "ITGB3", "FN1", "LGALS3", "THBS1"
        ],
        "ccl2_ccr2": [
            "CCL2", "CCR2", "CCR5", "CCL7", "CXCL10", "TNF", "IL1B", "RELA"
        ],
        "ccl2_ackr1": [
            "CCL2", "ACKR1", "VCAM1", "ICAM1", "SELE", "SELP", "CXCL1", "CXCL2"
        ],
        "vegfa_flt1": [
            "VEGFA", "FLT1", "KDR", "PECAM1", "CDH5", "VWF", "ANGPT2", "PLVAP"
        ],
        "vegfa_kdr": [
            "VEGFA", "KDR", "FLT1", "TEK", "NOS3", "PECAM1", "CDH5", "CLDN5"
        ],
        "tnf_axis": [
            "TNF", "TNFRSF1A", "TNFRSF1B", "RELA", "NFKB1", "NFKBIA",
            "IL1B", "IL6", "PTGS2", "ICAM1", "VCAM1"
        ],
        "tgfb1_axis": [
            "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
            "SMAD4", "SERPINE1", "CTGF", "FN1", "COL1A1", "COL3A1"
        ],
        "trem2_apoe": [
            "TREM2", "APOE", "TYROBP", "CSF1R", "C1QA", "C1QB", "C1QC",
            "LGALS3", "SPP1", "LST1", "ITGAM", "CD68"
        ],
        "ferroptosis_redox": [
            "GPX4", "SLC7A11", "FTH1", "FTL", "NFE2L2", "HMOX1", "NQO1",
            "GCLC", "GCLM", "ACSL4", "LPCAT3", "ALOX15", "TXNRD1", "SOD2"
        ],
        "synaptic_recovery": [
            "BDNF", "NTRK2", "MAP2", "RBFOX3", "SNAP25", "SYN1", "SYT1",
            "DLG4", "GRIN1", "GRIA1", "GRIA2", "GAP43", "DCX"
        ],
        "repair_ecm": [
            "TGFB1", "TGFBR1", "SMAD2", "SMAD3", "FN1", "SPP1", "CD44",
            "COL1A1", "COL1A2", "COL3A1", "MMP2", "MMP9", "TIMP1", "THBS1"
        ],
    }
    return {k: sorted(set(gene_upper(g) for g in v if gene_upper(g) not in HOUSEKEEPING)) for k, v in cat.items()}


def module_to_step66_key(module):
    mapping = {
        "spp1_cd44": "repair_ecm",
        "ccl2_ccr2": "microglia_inflammatory",
        "ccl2_ackr1": "bbb_leakage",
        "vegfa_flt1": "bbb_leakage",
        "vegfa_kdr": "barrier_stability",
        "tnf_axis": "inflammation",
        "tgfb1_axis": "repair_ecm",
        "trem2_apoe": "microglia_inflammatory",
        "ferroptosis_redox": "ferroptosis",
        "synaptic_recovery": "synaptic_recovery",
        "repair_ecm": "repair_ecm",
    }
    return mapping.get(module, module)


def known_positive_controls():
    cat = module_gene_catalog()
    rows = [
        ("Spp1_Cd44_blockade", "spp1_cd44", "known_positive", "SPP1-CD44 axis", +1),
        ("Ccl2_Ccr2_blockade", "ccl2_ccr2", "known_positive", "CCL2-CCR2 chemotaxis axis", +1),
        ("Ccl2_Ackr1_modulation", "ccl2_ackr1", "known_positive", "CCL2-ACKR1 endothelial chemokine axis", +1),
        ("Vegfa_Flt1_blockade", "vegfa_flt1", "known_positive", "VEGFA-FLT1 vascular axis", +1),
        ("Vegfa_Kdr_modulation", "vegfa_kdr", "known_positive", "VEGFA-KDR vascular axis", +1),
        ("Tnf_down", "tnf_axis", "known_positive", "TNF inflammatory axis", +1),
        ("Tgfb1_repair_ECM_modulation", "tgfb1_axis", "known_positive", "TGFβ repair/ECM axis", +1),
        ("Trem2_Apoe_microglia_modulation", "trem2_apoe", "known_positive", "TREM2-APOE microglia axis", +1),
        ("Ferroptosis_redox_down", "ferroptosis_redox", "known_positive", "ferroptosis/redox axis", +1),
        ("Synaptic_recovery_up_control", "synaptic_recovery", "known_positive", "synaptic recovery control", +1),
        ("Repair_ECM_up_control", "repair_ecm", "known_positive", "repair-permissive ECM control", +1),
    ]
    out = []
    for cid, module, ctype, desc, sign in rows:
        out.append({
            "control_id": cid,
            "control_type": ctype,
            "benchmark_label": 1,
            "module": module,
            "mapped_step66_module": module_to_step66_key(module),
            "target_genes": ";".join(cat[module]),
            "target_set_size": len(cat[module]),
            "direction_sign": sign,
            "description": desc,
        })
    return pd.DataFrame(out)


def direction_reversed_controls(pos):
    rows = []
    for _, r in pos.iterrows():
        row = r.to_dict()
        row["control_id"] = str(row["control_id"]) + "__direction_reversed"
        row["control_type"] = "direction_reversed_decoy"
        row["benchmark_label"] = 0
        row["direction_sign"] = -1
        row["description"] = "Direction-reversed negative control for " + str(r["control_id"])
        rows.append(row)
    return pd.DataFrame(rows)


def housekeeping_controls():
    genes = [
        "ACTB", "B2M", "GAPDH", "HPRT1", "MALAT1", "PPIA", "RPLP0", "RPS18",
        "RPL13A", "EEF1A1", "TUBA1A", "TUBB"
    ]
    rows = []
    for g in genes:
        rows.append({
            "control_id": f"{g}_housekeeping_decoy",
            "control_type": "housekeeping_decoy",
            "benchmark_label": 0,
            "module": "housekeeping",
            "mapped_step66_module": "none",
            "target_genes": g,
            "target_set_size": 1,
            "direction_sign": 1,
            "description": "housekeeping negative control",
        })

    grouped = [
        ("Housekeeping_panel_1", ["ACTB", "B2M", "GAPDH", "HPRT1", "PPIA"]),
        ("Ribo_panel", ["RPLP0", "RPS18", "RPL13A"]),
        ("RNA_abundance_panel", ["MALAT1", "EEF1A1", "TUBA1A", "TUBB"]),
    ]
    for cid, gs in grouped:
        rows.append({
            "control_id": cid,
            "control_type": "housekeeping_panel_decoy",
            "benchmark_label": 0,
            "module": "housekeeping",
            "mapped_step66_module": "none",
            "target_genes": ";".join(gs),
            "target_set_size": len(gs),
            "direction_sign": 1,
            "description": "housekeeping panel negative control",
        })

    return pd.DataFrame(rows)


def build_gene_universe(marker_genes, extra_genes=None):
    genes = set(marker_genes["gene"].dropna().astype(str).map(gene_upper)) if not marker_genes.empty else set()
    for gs in module_gene_catalog().values():
        genes.update(gs)
    if extra_genes:
        genes.update(gene_upper(g) for g in extra_genes)
    genes = {
        g for g in genes
        if g not in HOUSEKEEPING
        and not g.startswith(("RPL", "RPS", "MRPL", "MRPS", "MT-", "HBA", "HBB"))
        and len(g) >= 2
    }
    return sorted(genes)


def random_controls(universe, pos, n_random=200, seed=1):
    rng = np.random.default_rng(seed)
    sizes = pos["target_set_size"].astype(int).clip(lower=1).tolist()
    rows = []
    for i in range(n_random):
        k = int(rng.choice(sizes))
        k = min(k, len(universe))
        gs = rng.choice(universe, size=k, replace=False).tolist()
        rows.append({
            "control_id": f"random_size_matched_{i+1:04d}",
            "control_type": "random_gene_set_decoy",
            "benchmark_label": 0,
            "module": "random",
            "mapped_step66_module": "none",
            "target_genes": ";".join(sorted(gs)),
            "target_set_size": k,
            "direction_sign": 1,
            "description": "random gene-set negative control matched by target-set size",
        })
    return pd.DataFrame(rows)


def shuffled_controls(pos, seed=1, n_shuffles=50):
    rng = np.random.default_rng(seed)
    all_genes = []
    sizes = []
    for _, r in pos.iterrows():
        gs = split_genes(r["target_genes"])
        all_genes.extend(gs)
        sizes.append(len(gs))
    all_genes = sorted(set(all_genes))

    rows = []
    for i in range(n_shuffles):
        k = int(rng.choice(sizes))
        k = min(k, len(all_genes))
        gs = rng.choice(all_genes, size=k, replace=False).tolist()
        rows.append({
            "control_id": f"shuffled_positive_targets_{i+1:04d}",
            "control_type": "shuffled_target_decoy",
            "benchmark_label": 0,
            "module": "shuffled",
            "mapped_step66_module": "none",
            "target_genes": ";".join(sorted(gs)),
            "target_set_size": k,
            "direction_sign": 1,
            "description": "shuffled target-set negative control",
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Load marker genes and Step66 anchors
# -----------------------------------------------------------------------------

def load_marker_genes(step65e, step65c, top_n=50):
    p65e = Path(step65e) / "step65e_track_marker_genes_after_filter.csv"
    if p65e.exists():
        df = pd.read_csv(p65e, low_memory=False)
        rows = []
        if {"track_id", "marker_genes_after_filter"}.issubset(df.columns):
            for _, r in df.iterrows():
                tid = str(r["track_id"])
                genes = split_genes(r["marker_genes_after_filter"])
                for rank, g in enumerate(genes[:top_n], start=1):
                    if g not in HOUSEKEEPING:
                        rows.append({
                            "track_id": tid,
                            "gene": g,
                            "marker_rank": rank,
                            "marker_source": "step65e_housekeeping_filtered",
                            "marker_weight": 1.0 / rank,
                        })
            out = pd.DataFrame(rows)
            if not out.empty:
                return out

    p65 = Path(step65c) / "step65_track_dynamic_genes.csv"
    df = pd.read_csv(p65, low_memory=False)
    if "passes_min_cells" in df.columns:
        df = df[df["passes_min_cells"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if "dynamic_marker_score" not in df.columns:
        df["dynamic_marker_score"] = 1.0
    df["gene"] = df["feature"].map(gene_upper)
    rows = []
    for tid, sub in df.groupby("track_id"):
        sub = sub[~sub["gene"].isin(HOUSEKEEPING)].sort_values("dynamic_marker_score", ascending=False).drop_duplicates("gene").head(top_n)
        for rank, r in enumerate(sub.itertuples(), start=1):
            rows.append({
                "track_id": str(tid),
                "gene": r.gene,
                "marker_rank": rank,
                "marker_source": "step65c_dynamic_gene",
                "marker_weight": safe_float(getattr(r, "dynamic_marker_score", 1.0)),
            })
    return pd.DataFrame(rows)


def marker_map_by_track(marker_genes):
    out = {}
    for tid, sub in marker_genes.groupby("track_id"):
        out[str(tid)] = list(dict.fromkeys(sub.sort_values("marker_rank")["gene"].astype(str).map(gene_upper)))
    return out


def load_track_weights(step66e_rerun, step64c):
    p = Path(step66e_rerun) / "step66c_track_level_perturbation_scores.csv"
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        if {"track_id", "track_weight"}.issubset(df.columns):
            out = df[["track_id", "track_weight"]].drop_duplicates()
            out["track_id"] = out["track_id"].astype(str)
            out["track_weight"] = pd.to_numeric(out["track_weight"], errors="coerce")
            if out["track_weight"].notna().sum() > 0 and out["track_weight"].sum() > 0:
                out["track_weight"] = out["track_weight"].fillna(0)
                out["track_weight"] = out["track_weight"] / out["track_weight"].sum()
                return out

    p2 = Path(step64c) / "step64c_track_label_refined.csv"
    if p2.exists():
        tr = pd.read_csv(p2, low_memory=False)
        if "track_id" in tr.columns:
            if "track_priority_score" in tr.columns:
                w = pd.to_numeric(tr["track_priority_score"], errors="coerce").fillna(0)
            elif "n_cells_sum" in tr.columns:
                w = pd.to_numeric(tr["n_cells_sum"], errors="coerce").fillna(0)
            else:
                w = pd.Series(np.ones(len(tr)), index=tr.index)
            if w.sum() <= 0:
                w = pd.Series(np.ones(len(tr)), index=tr.index)
            return pd.DataFrame({
                "track_id": tr["track_id"].astype(str),
                "track_weight": w / w.sum(),
            })

    tids = sorted(marker_map_by_track(load_marker_genes(DEFAULT_STEP65E, DEFAULT_STEP65C)).keys())
    return pd.DataFrame({"track_id": tids, "track_weight": np.ones(len(tids)) / max(len(tids), 1)})


def load_step66_anchor(step66f_dir):
    p = Path(step66f_dir) / "step66f_manuscript_ready_perturbation_summary.csv"
    if not p.exists():
        return pd.DataFrame()

    df = pd.read_csv(p, low_memory=False)

    if "module_key" not in df.columns:
        return pd.DataFrame()

    score_col = "observed_minus_random_mean" if "observed_minus_random_mean" in df.columns else "s_overall_observed"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    rows = []
    for mod, sub in df.groupby("module_key"):
        val = sub[score_col].max()
        obs = pd.to_numeric(sub.get("s_overall_observed", pd.Series(np.nan, index=sub.index)), errors="coerce").max()
        fdr = pd.to_numeric(sub.get("bh_fdr", pd.Series(np.nan, index=sub.index)), errors="coerce").min()
        rows.append({
            "mapped_step66_module": str(mod),
            "step66_anchor_score_raw": safe_float(val, 0.0),
            "step66_observed_score_max": safe_float(obs, np.nan),
            "step66_min_bh_fdr": safe_float(fdr, np.nan),
        })

    out = pd.DataFrame(rows)
    out["step66_anchor_score_norm"] = minmax(out["step66_anchor_score_raw"]).values
    return out


# -----------------------------------------------------------------------------
# Scoring and metrics
# -----------------------------------------------------------------------------

def score_controls(controls, marker_genes, track_weights, anchor_df):
    mmap = marker_map_by_track(marker_genes)
    tw = track_weights.set_index("track_id")["track_weight"].to_dict()
    anchor_map = anchor_df.set_index("mapped_step66_module")["step66_anchor_score_norm"].to_dict() if not anchor_df.empty else {}
    anchor_raw = anchor_df.set_index("mapped_step66_module")["step66_anchor_score_raw"].to_dict() if not anchor_df.empty else {}

    catalog = module_gene_catalog()
    all_catalog_genes = {m: set(gs) for m, gs in catalog.items()}

    per_track_rows = []
    score_rows = []

    for _, r in controls.iterrows():
        cid = str(r["control_id"])
        genes = split_genes(r["target_genes"])
        direction_sign = safe_float(r.get("direction_sign", 1), 1)
        mapped_mod = str(r.get("mapped_step66_module", "none"))

        rel_sum = 0.0
        rel_max = 0.0
        overlap_all = []

        for tid, markers in mmap.items():
            rel = target_relevance(genes, markers)
            weight = safe_float(tw.get(tid, 1.0 / max(len(mmap), 1)), 0.0)
            weighted = rel["relevance"] * weight
            rel_sum += weighted
            rel_max = max(rel_max, rel["relevance"])
            if rel["overlap_genes"]:
                overlap_all.extend(split_genes(rel["overlap_genes"]))

            per_track_rows.append({
                "control_id": cid,
                "track_id": tid,
                "track_weight": weight,
                "track_target_relevance": rel["relevance"],
                "track_overlap_n": rel["overlap_n"],
                "track_overlap_genes": rel["overlap_genes"],
                "weighted_track_relevance": weighted,
            })

        # Catalog similarity: whether the target set resembles known stroke modules.
        catalog_jaccards = {m: jaccard(genes, gs) for m, gs in all_catalog_genes.items()}
        max_catalog_module = max(catalog_jaccards, key=catalog_jaccards.get) if catalog_jaccards else ""
        max_catalog_jaccard = catalog_jaccards.get(max_catalog_module, 0.0)

        anchor_score = safe_float(anchor_map.get(mapped_mod, 0.0), 0.0)
        anchor_score_raw = safe_float(anchor_raw.get(mapped_mod, np.nan), np.nan)

        # Final score:
        #   target-track relevance: 50%
        #   Step66f module anchor: 30%
        #   curated module similarity: 20%
        # Direction-reversed perturbations invert the biologically favorable direction.
        unsigned_score = (
            0.50 * rel_sum
            + 0.30 * anchor_score
            + 0.20 * max_catalog_jaccard
        )

        benchmark_score = direction_sign * unsigned_score

        score_rows.append({
            **r.to_dict(),
            "benchmark_score": benchmark_score,
            "benchmark_score_unsigned": unsigned_score,
            "target_track_relevance_weighted_sum": rel_sum,
            "target_track_relevance_max": rel_max,
            "step66_anchor_score_norm": anchor_score,
            "step66_anchor_score_raw": anchor_score_raw,
            "catalog_best_module": max_catalog_module,
            "catalog_best_jaccard": max_catalog_jaccard,
            "all_track_overlap_genes": ";".join(sorted(set(overlap_all))),
            "n_all_track_overlap_genes": len(set(overlap_all)),
        })

    scores = pd.DataFrame(score_rows)
    per_track = pd.DataFrame(per_track_rows)

    scores = scores.sort_values("benchmark_score", ascending=False).reset_index(drop=True)
    scores["benchmark_rank"] = np.arange(1, len(scores) + 1)

    return scores, per_track


def roc_auc_score_manual(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    pos = y == 1
    neg = y == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return np.nan

    # rank-sum AUC with average ranks for ties.
    order = np.argsort(s)
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)

    # tie correction: average ranks
    vals = pd.Series(s)
    avg_ranks = vals.rank(method="average").values
    rank_sum_pos = avg_ranks[pos].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def average_precision_manual(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = y.sum()
    if n_pos == 0:
        return np.nan
    tp = 0
    precisions = []
    for i, yy in enumerate(y_sorted, start=1):
        if yy == 1:
            tp += 1
            precisions.append(tp / i)
    return float(np.mean(precisions)) if precisions else np.nan


def pr_curve_points(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = max(y.sum(), 1)
    tp = 0
    fp = 0
    rec = [0.0]
    prec = [1.0]
    for yy in y_sorted:
        if yy == 1:
            tp += 1
        else:
            fp += 1
        rec.append(tp / n_pos)
        prec.append(tp / max(tp + fp, 1))
    return np.array(rec), np.array(prec)


def roc_curve_points(y_true, scores):
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    order = np.argsort(-s)
    y_sorted = y[order]
    n_pos = max(y.sum(), 1)
    n_neg = max((1 - y).sum(), 1)
    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    for yy in y_sorted:
        if yy == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / n_pos)
        fpr.append(fp / n_neg)
    return np.array(fpr), np.array(tpr)


def compute_metrics(scores):
    y = scores["benchmark_label"].astype(int).values
    s = scores["benchmark_score"].astype(float).values

    auroc = roc_auc_score_manual(y, s)
    auprc = average_precision_manual(y, s)

    rows = [{
        "metric": "AUROC_positive_vs_decoy",
        "value": auroc,
        "n_positive": int(y.sum()),
        "n_negative": int((1-y).sum()),
    }, {
        "metric": "AUPRC_average_precision_positive_vs_decoy",
        "value": auprc,
        "n_positive": int(y.sum()),
        "n_negative": int((1-y).sum()),
    }]

    # Additional per-negative-type metrics.
    for neg_type in sorted(scores.loc[scores["benchmark_label"].eq(0), "control_type"].unique()):
        sub = scores[(scores["benchmark_label"].eq(1)) | (scores["control_type"].eq(neg_type))].copy()
        yy = sub["benchmark_label"].astype(int).values
        ss = sub["benchmark_score"].astype(float).values
        rows.append({
            "metric": f"AUROC_positive_vs_{neg_type}",
            "value": roc_auc_score_manual(yy, ss),
            "n_positive": int(yy.sum()),
            "n_negative": int((1-yy).sum()),
        })
        rows.append({
            "metric": f"AUPRC_positive_vs_{neg_type}",
            "value": average_precision_manual(yy, ss),
            "n_positive": int(yy.sum()),
            "n_negative": int((1-yy).sum()),
        })

    return pd.DataFrame(rows)


def hypergeom_sf_manual(M, K, n, x):
    try:
        from scipy.stats import hypergeom
        return float(hypergeom.sf(x - 1, M, K, n))
    except Exception:
        return np.nan


def topk_enrichment(scores, ks=(1, 3, 5, 10, 20, 50, 100)):
    df = scores.sort_values("benchmark_score", ascending=False).reset_index(drop=True)
    M = len(df)
    K = int(df["benchmark_label"].sum())
    rows = []
    for k in ks:
        k = min(int(k), M)
        if k <= 0:
            continue
        sub = df.head(k)
        x = int(sub["benchmark_label"].sum())
        expected = k * K / max(M, 1)
        enrich = x / expected if expected > 0 else np.nan
        rows.append({
            "top_k": k,
            "positive_in_top_k": x,
            "expected_positive": expected,
            "fold_enrichment": enrich,
            "positive_fraction_top_k": x / k,
            "hypergeom_p_value": hypergeom_sf_manual(M, K, k, x),
            "top_k_ids": ";".join(sub["control_id"].astype(str).tolist()),
        })
    return pd.DataFrame(rows)


def rank_stability_bootstrap(scores, per_track, anchor_df, marker_genes, controls, n_boot=300, top_k=20, seed=1):
    rng = np.random.default_rng(seed)
    tids = sorted(per_track["track_id"].astype(str).unique())
    if not tids:
        return pd.DataFrame()

    base = scores.set_index("control_id")["benchmark_score"].to_dict()
    base_ranked = scores.sort_values("benchmark_score", ascending=False)["control_id"].astype(str).tolist()
    base_top = set(base_ranked[:min(top_k, len(base_ranked))])

    anchor_map = anchor_df.set_index("mapped_step66_module")["step66_anchor_score_norm"].to_dict() if not anchor_df.empty else {}
    catalog = module_gene_catalog()
    control_info = controls.set_index("control_id").to_dict(orient="index")

    pt = per_track.copy()
    pt["control_id"] = pt["control_id"].astype(str)
    pt["track_id"] = pt["track_id"].astype(str)

    rel_pivot = pt.pivot_table(
        index="control_id",
        columns="track_id",
        values="track_target_relevance",
        aggfunc="mean",
        fill_value=0.0,
    )

    rows = []
    control_ids = rel_pivot.index.astype(str).tolist()

    for b in range(n_boot):
        sampled = rng.choice(tids, size=len(tids), replace=True).tolist()

        boot_scores = {}
        for cid in control_ids:
            info = control_info.get(cid, {})
            direction = safe_float(info.get("direction_sign", 1), 1)
            mapped = str(info.get("mapped_step66_module", "none"))
            genes = split_genes(info.get("target_genes", ""))
            module = str(info.get("module", ""))

            track_rel = rel_pivot.loc[cid, sampled].mean() if cid in rel_pivot.index else 0.0
            anchor = safe_float(anchor_map.get(mapped, 0.0), 0.0)

            catalog_j = 0.0
            if module in catalog:
                catalog_j = jaccard(genes, catalog[module])
            else:
                catalog_j = max([jaccard(genes, gs) for gs in catalog.values()] + [0.0])

            boot_scores[cid] = direction * (0.50 * track_rel + 0.30 * anchor + 0.20 * catalog_j)

        common = list(base.keys() & boot_scores.keys())
        if len(common) >= 3:
            a = pd.Series([base[c] for c in common]).rank().values
            bb = pd.Series([boot_scores[c] for c in common]).rank().values
            corr = float(np.corrcoef(a, bb)[0, 1]) if np.std(a) > 0 and np.std(bb) > 0 else np.nan
        else:
            corr = np.nan

        boot_ranked = sorted(boot_scores, key=boot_scores.get, reverse=True)
        boot_top = set(boot_ranked[:min(top_k, len(boot_ranked))])
        jac = len(base_top & boot_top) / max(len(base_top | boot_top), 1)

        rows.append({
            "bootstrap_iter": b + 1,
            "spearman_rank_correlation": corr,
            "top_k": top_k,
            "top_k_jaccard": jac,
            "top_k_positive_count": int(scores[scores["control_id"].isin(boot_top)]["benchmark_label"].sum()),
        })

    return pd.DataFrame(rows)


def direction_reversal_consistency(scores):
    df = scores.set_index("control_id")
    rows = []
    for cid in scores["control_id"].astype(str):
        if cid.endswith("__direction_reversed"):
            continue
        rev = cid + "__direction_reversed"
        if rev in df.index and cid in df.index:
            pos = safe_float(df.loc[cid, "benchmark_score"])
            neg = safe_float(df.loc[rev, "benchmark_score"])
            rows.append({
                "positive_control_id": cid,
                "reversed_control_id": rev,
                "positive_score": pos,
                "reversed_score": neg,
                "delta_positive_minus_reversed": pos - neg,
                "consistent": bool(pos > neg),
                "positive_rank": int(df.loc[cid, "benchmark_rank"]),
                "reversed_rank": int(df.loc[rev, "benchmark_rank"]),
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


def plot_benchmark(scores, metrics, topk, stability, drev, outdir, dpi=600):
    outputs = {}

    y = scores["benchmark_label"].astype(int).values
    s = scores["benchmark_score"].astype(float).values
    fpr, tpr = roc_curve_points(y, s)
    rec, prec = pr_curve_points(y, s)

    auroc = metrics.loc[metrics["metric"].eq("AUROC_positive_vs_decoy"), "value"].iloc[0]
    auprc = metrics.loc[metrics["metric"].eq("AUPRC_average_precision_positive_vs_decoy"), "value"].iloc[0]

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step67_KnownPositiveDecoyBenchmark_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34)

        # A ROC/PR
        axA = fig.add_subplot(gs[0, 0])
        axA.plot(fpr, tpr, label=f"ROC AUC={auroc:.3f}")
        axA.plot(rec, prec, label=f"PR AP={auprc:.3f}")
        axA.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)
        if annotate:
            axA.set_xlabel("FPR / Recall")
            axA.set_ylabel("TPR / Precision")
            axA.set_title("A | Positive-vs-decoy benchmark", loc="left", fontsize=12, fontweight="bold")
            axA.legend(frameon=False, fontsize=8)
        else:
            strip_text(axA)

        # B ranked scores
        axB = fig.add_subplot(gs[0, 1])
        ranked = scores.sort_values("benchmark_score", ascending=False).reset_index(drop=True)
        x = np.arange(len(ranked))
        axB.scatter(x, ranked["benchmark_score"].values, s=20, alpha=0.75)
        pos_idx = ranked.index[ranked["benchmark_label"].eq(1)]
        axB.scatter(pos_idx, ranked.loc[pos_idx, "benchmark_score"].values, s=55, alpha=0.9)
        if annotate:
            for _, r in ranked.head(12).iterrows():
                axB.text(int(r.name), r["benchmark_score"], str(r["control_id"])[:24], fontsize=7, rotation=45)
            axB.set_xlabel("Benchmark rank")
            axB.set_ylabel("Benchmark score")
            axB.set_title("B | Ranked known-positive and decoy controls", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axB)

        # C top-k enrichment
        axC = fig.add_subplot(gs[1, 0])
        kk = topk["top_k"].astype(int).values
        enr = topk["fold_enrichment"].astype(float).values
        axC.plot(kk, enr, marker="o")
        if annotate:
            axC.set_xlabel("Top-k")
            axC.set_ylabel("Positive-control fold enrichment")
            axC.set_title("C | Top-k enrichment", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axC)

        # D stability / reversal
        axD = fig.add_subplot(gs[1, 1])
        if not stability.empty:
            vals = stability["spearman_rank_correlation"].dropna().values
            axD.hist(vals, bins=25, alpha=0.75, label="rank stability")
        if not drev.empty:
            cons = drev["consistent"].mean()
            axD.axvline(cons, linestyle="--", linewidth=1.2, label=f"direction consistency={cons:.2f}")
        if annotate:
            axD.set_xlabel("Bootstrap rank correlation / direction consistency")
            axD.set_ylabel("Count")
            axD.set_title("D | Rank stability and direction reversal", loc="left", fontsize=12, fontweight="bold")
            axD.legend(frameon=False, fontsize=8)
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
                "Step67 | Known-positive / decoy benchmark for StrokeNiche PerturbMap",
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
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--step66f_dir", default=str(DEFAULT_STEP66F))
    ap.add_argument("--step66e_rerun_dir", default=str(DEFAULT_STEP66E_RERUN))
    ap.add_argument("--step65c_dir", default=str(DEFAULT_STEP65C))
    ap.add_argument("--step65e_dir", default=str(DEFAULT_STEP65E))
    ap.add_argument("--step64c_dir", default=str(DEFAULT_STEP64C))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--n_random_decoys", type=int, default=200)
    ap.add_argument("--n_shuffled_decoys", type=int, default=50)
    ap.add_argument("--n_bootstrap", type=int, default=300)
    ap.add_argument("--top_marker_genes_per_track", type=int, default=50)
    ap.add_argument("--top_k_stability", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step67 known-positive / decoy benchmark")
    log("=" * 100)
    log(f"outdir={outdir}")

    marker_genes = load_marker_genes(
        step65e=args.step65e_dir,
        step65c=args.step65c_dir,
        top_n=args.top_marker_genes_per_track,
    )
    track_weights = load_track_weights(args.step66e_rerun_dir, args.step64c_dir)
    anchor_df = load_step66_anchor(args.step66f_dir)

    pos = known_positive_controls()
    rev = direction_reversed_controls(pos)
    hk = housekeeping_controls()

    universe = build_gene_universe(marker_genes)
    rand = random_controls(universe, pos, n_random=args.n_random_decoys, seed=args.seed)
    shuf = shuffled_controls(pos, seed=args.seed + 11, n_shuffles=args.n_shuffled_decoys)

    controls = pd.concat([pos, rev, hk, rand, shuf], ignore_index=True)
    controls["control_id"] = controls["control_id"].astype(str)

    scores, per_track = score_controls(controls, marker_genes, track_weights, anchor_df)

    metrics = compute_metrics(scores)
    topk = topk_enrichment(scores)
    stability = rank_stability_bootstrap(
        scores=scores,
        per_track=per_track,
        anchor_df=anchor_df,
        marker_genes=marker_genes,
        controls=controls,
        n_boot=args.n_bootstrap,
        top_k=args.top_k_stability,
        seed=args.seed + 101,
    )
    drev = direction_reversal_consistency(scores)

    fig_outputs = plot_benchmark(
        scores=scores,
        metrics=metrics,
        topk=topk,
        stability=stability,
        drev=drev,
        outdir=outdir,
        dpi=args.dpi,
    )

    # Save outputs.
    controls.to_csv(outdir / "step67_benchmark_controls.csv", index=False)
    scores.to_csv(outdir / "step67_benchmark_scores.csv", index=False)
    per_track.to_csv(outdir / "step67_per_track_relevance_matrix.csv", index=False)
    metrics.to_csv(outdir / "step67_benchmark_metrics.csv", index=False)
    topk.to_csv(outdir / "step67_topk_enrichment.csv", index=False)
    stability.to_csv(outdir / "step67_rank_stability_bootstrap.csv", index=False)
    drev.to_csv(outdir / "step67_direction_reversal_consistency.csv", index=False)
    marker_genes.to_csv(outdir / "step67_marker_genes_used.csv", index=False)
    track_weights.to_csv(outdir / "step67_track_weights_used.csv", index=False)
    anchor_df.to_csv(outdir / "step67_step66_anchor_used.csv", index=False)

    auroc = metrics.loc[metrics["metric"].eq("AUROC_positive_vs_decoy"), "value"].iloc[0]
    auprc = metrics.loc[metrics["metric"].eq("AUPRC_average_precision_positive_vs_decoy"), "value"].iloc[0]

    direction_consistency = float(drev["consistent"].mean()) if not drev.empty else np.nan
    stability_corr_median = float(stability["spearman_rank_correlation"].median()) if not stability.empty else np.nan
    stability_jaccard_median = float(stability["top_k_jaccard"].median()) if not stability.empty else np.nan

    report = {
        "status": "ok",
        "analysis_name": "Step67 known-positive / decoy benchmark",
        "n_positive_controls": int(scores["benchmark_label"].sum()),
        "n_negative_decoys": int((1 - scores["benchmark_label"]).sum()),
        "n_total_controls": int(len(scores)),
        "n_random_decoys": int(args.n_random_decoys),
        "n_shuffled_decoys": int(args.n_shuffled_decoys),
        "AUROC_positive_vs_decoy": safe_float(auroc),
        "AUPRC_average_precision_positive_vs_decoy": safe_float(auprc),
        "direction_reversal_consistency": direction_consistency,
        "bootstrap_rank_spearman_median": stability_corr_median,
        "bootstrap_topk_jaccard_median": stability_jaccard_median,
        "top10_positive_count": int(scores.sort_values("benchmark_score", ascending=False).head(10)["benchmark_label"].sum()),
        "top20_positive_count": int(scores.sort_values("benchmark_score", ascending=False).head(20)["benchmark_label"].sum()),
        "figure_outputs": fig_outputs,
        "outputs": {
            "controls": str(outdir / "step67_benchmark_controls.csv"),
            "scores": str(outdir / "step67_benchmark_scores.csv"),
            "metrics": str(outdir / "step67_benchmark_metrics.csv"),
            "topk_enrichment": str(outdir / "step67_topk_enrichment.csv"),
            "rank_stability": str(outdir / "step67_rank_stability_bootstrap.csv"),
            "direction_reversal": str(outdir / "step67_direction_reversal_consistency.csv"),
            "report_json": str(outdir / "step67_report.json"),
            "report_txt": str(outdir / "step67_report.txt"),
        },
        "interpretation_note": (
            "Step67 is a known-positive / decoy calibration benchmark. It tests whether curated "
            "stroke-relevant perturbation axes rank above housekeeping, random, shuffled and direction-reversed "
            "controls. It is not external therapeutic validation and should not be directly compared with UNAGI "
            "unless the same benchmark tasks and baselines are run."
        ),
    }

    (outdir / "step67_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step67 known-positive / decoy benchmark report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Benchmark metrics:")
    lines.append(metrics.to_string(index=False))
    lines.append("")
    lines.append("Top-k enrichment:")
    lines.append(topk.to_string(index=False))
    lines.append("")
    lines.append("Direction reversal consistency:")
    lines.append(drev.to_string(index=False) if not drev.empty else "No direction-reversal pairs found.")
    lines.append("")
    lines.append("Top ranked benchmark controls:")
    show = [
        "benchmark_rank", "control_id", "control_type", "benchmark_label",
        "benchmark_score", "target_track_relevance_weighted_sum",
        "step66_anchor_score_norm", "catalog_best_jaccard",
        "target_set_size", "mapped_step66_module", "all_track_overlap_genes"
    ]
    show = [c for c in show if c in scores.columns]
    lines.append(scores[show].head(80).to_string(index=False))

    (outdir / "step67_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step67")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log(scores[show].head(40).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
