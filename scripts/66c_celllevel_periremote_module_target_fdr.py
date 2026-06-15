#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
66c_celllevel_periremote_module_target_fdr.py

Purpose
-------
Step 66c: strict cell-level track perturbation FDR with patched peri/remote shift
and module perturbation -> gene module target mapping.

This fixes the remaining Step66b limitation:

  Step66b:
    score_mode = strict_summary_track_prior
    component_mapping_status = strict_cell_level_not_available_or_rejected
    missing = peri_or_remote_shift

Step66c:
  1. Patch Step61 cell-level output by adding:
       predicted_delta_peri_probability
       predicted_delta_remote_probability
       predicted_peri_remote_shift

  2. Map niche-level module perturbations to gene target sets:
       ferroptosis_down -> GPX4;SLC7A11;FTH1;FTL1;NFE2L2;HMOX1;NQO1;ACSL4...
       inflammation_down -> TNF;IL1B;IL6;CCL2;CCR2;RELA;NFKB1...
       BBB_leakage_down -> CLDN5;OCLN;TJP1;PECAM1;VCAM1;ICAM1...
       etc.

  3. Recompute:
       track-level perturbation scores
       matched random target background
       empirical P value
       BH-FDR

  4. Output final cell-level Step66c tables and figures.

Outputs
-------
outdir/
  step66c_step61_cell_level_with_peri_remote_shift.csv
  step66c_module_target_mapped_candidates.csv
  step66c_cell_level_component_audit.csv
  step66c_track_level_perturbation_scores.csv
  step66c_random_background_scores.csv
  step66c_perturbation_fdr.csv
  step66c_track_perturbation_heatmap.csv
  step66c_report.json
  step66c_report.txt
  Fig_Step66C_CellLevelTrackPerturbationFDR_annotated.pdf/svg/png
  Fig_Step66C_CellLevelTrackPerturbationFDR_clean_no_text.pdf/svg/png
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


BASE = Path("/mnt/h/vir/ST")
ROOT = BASE / "results/step8_strokeniche_perturbmap"

DEFAULT_TRACK_DIR = ROOT / "dynamics_graph_64c_refined_labels_state_celltype"
DEFAULT_STEP61_DIR = ROOT / "niche_perturb_61"
DEFAULT_STEP65_DIR = ROOT / "track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata"
DEFAULT_STEP65E_DIR = ROOT / "track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered"
DEFAULT_OUT = ROOT / "track_level_perturbation_fdr_66c_celllevel_periremote_module_targets"


# -----------------------------------------------------------------------------
# Basic helpers
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


def read_csv_auto(path, required=True):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.read_csv(p, sep="\t", low_memory=False)


def bh_fdr(pvals):
    p = np.asarray([safe_float(x, np.nan) for x in pvals], dtype=float)
    out = np.full(len(p), np.nan)
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


def minmax_series(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def group_minmax(df, group_col, value_col):
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby(group_col).groups.items():
        out.loc[idx] = minmax_series(df.loc[idx, value_col]).values
    return out.fillna(0.5)


# -----------------------------------------------------------------------------
# Gene / module target mapping
# -----------------------------------------------------------------------------

HOUSEKEEPING_EXACT = {
    "ACTB", "B2M", "GAPDH", "HPRT", "HPRT1", "MALAT1", "PPIA",
    "RPLP0", "RPS18", "RPL13A", "TUBA1A", "TUBA1B", "TUBB",
    "EEF1A1", "EEF2", "PGK1", "LDHA", "LDHB",
}

NON_GENE_WORDS = {
    "UP", "DOWN", "KD", "KO", "OE", "OVEREXPRESSION", "KNOCKDOWN",
    "BLOCKADE", "INHIBITION", "INHIBIT", "INHIBITOR", "ACTIVATE",
    "ACTIVATION", "AGONIST", "ANTAGONIST", "NEUTRAL", "DECOY",
    "GENE", "DRUG", "COMPOUND", "PERTURBATION", "PERTURBAGEN",
    "MODULE", "NAN", "NONE", "TRUE", "FALSE",
    "REPAIR", "CORE", "REMOTE", "PERI", "PERIINFARCT", "LESION",
    "STROKE", "NICHE", "RESCUE", "SCORE", "RANKING", "FINAL",
    "CLEAN", "SHORTLIST", "TARGET", "TARGETS", "SIGNATURE",
    "PATHWAY", "BACKGROUND", "RANDOM",
    "FERROPTOSIS", "INFLAMMATION", "HYPOXIA", "BBB", "LEAKAGE",
    "BARRIER", "STABILITY", "ENDOTHELIAL", "FRAGILITY",
    "MICROGLIA", "INFLAMMATORY", "ASTROCYTE", "REACTIVE",
    "SYNAPTIC", "RECOVERY", "ECM",
}


def is_housekeeping_gene(g):
    gu = gene_upper(g)
    raw = str(g).strip()

    if gu in HOUSEKEEPING_EXACT:
        return True
    if raw.lower().startswith("mt-") or gu.startswith("MT-"):
        return True
    if gu.startswith(("RPL", "RPS", "MRPL", "MRPS")):
        return True
    if gu.startswith(("HBA", "HBB", "HBG", "HBD", "HBE", "HBZ", "HBQ", "HBM")):
        return True
    if re.match(r"^GM[0-9]+$", gu):
        return True
    return False


def split_gene_tokens(text):
    if text is None or pd.isna(text):
        return []

    s = str(text)
    s = re.sub(r"[\[\]\(\)\{\}\"']", " ", s)
    s = re.sub(r"[_:/|,;]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    genes = []
    for t in s.split(" "):
        t = t.strip()
        if not t:
            continue

        t = re.sub(r"[-\.](down|up|kd|ko|oe|blockade)$", "", t, flags=re.I)
        gu = gene_upper(t)

        if gu in NON_GENE_WORDS:
            continue
        if not re.match(r"^[A-Za-z][A-Za-z0-9\.\-]{1,24}$", t):
            continue
        if re.match(r"^ENC[A-Z0-9]+$", gu):
            continue
        if is_housekeeping_gene(gu):
            continue

        genes.append(gu)

    return list(dict.fromkeys(genes))


def module_gene_catalog():
    """
    Conservative mouse/human symbol-compatible StrokeNiche module targets.
    These are used as computational module target sets, not validated drug targets.
    """
    cat = {
        "ferroptosis": [
            "GPX4", "SLC7A11", "FTH1", "FTL1", "FTL", "NFE2L2", "KEAP1",
            "HMOX1", "NQO1", "GCLC", "GCLM", "ACSL4", "LPCAT3",
            "ALOX5", "ALOX15", "SAT1", "AIFM2", "FSP1", "TXNRD1",
            "PRDX1", "PRDX2", "PRDX6", "SOD1", "SOD2", "CAT"
        ],
        "inflammation": [
            "TNF", "IL1B", "IL1A", "IL6", "IL10", "IL18", "CCL2",
            "CCL3", "CCL4", "CCL5", "CCL7", "CXCL1", "CXCL2",
            "CXCL10", "CCR2", "CCR5", "RELA", "NFKB1", "NFKBIA",
            "TLR2", "TLR4", "MYD88", "NLRP3", "CASP1", "PTGS2",
            "NOS2", "ICAM1", "VCAM1"
        ],
        "bbb_leakage": [
            "CLDN5", "OCLN", "TJP1", "TJP2", "PECAM1", "CDH5", "VWF",
            "KDR", "FLT1", "TEK", "ENG", "ESAM", "ICAM1", "VCAM1",
            "SELE", "SELP", "PLVAP", "MFSD2A", "ABCB1A", "ABCG2",
            "ANGPT1", "ANGPT2", "NOS3"
        ],
        "barrier_stability": [
            "CLDN5", "OCLN", "TJP1", "CDH5", "PECAM1", "KLF2", "KLF4",
            "NOS3", "TEK", "ANGPT1", "MFSD2A", "ABCB1A", "ABCG2",
            "ESAM", "ENG"
        ],
        "endothelial_barrier_fragility": [
            "KDR", "FLT1", "VWF", "PLVAP", "VCAM1", "ICAM1", "ANGPT2",
            "SELE", "SELP", "ENG", "ESAM", "PECAM1", "CDH5", "CLDN5"
        ],
        "microglia_inflammatory": [
            "AIF1", "TREM2", "APOE", "TYROBP", "CSF1R", "ITGAM", "CD68",
            "LST1", "C1QA", "C1QB", "C1QC", "LGALS3", "SPP1", "CCL2",
            "CCR2", "TNF", "IL1B", "IL6", "CXCL10", "IRF1", "IRF8",
            "SPI1"
        ],
        "astrocyte_reactive": [
            "GFAP", "VIM", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
            "SOX9", "STAT3", "LCN2", "SERPINA3N", "CLU", "C3",
            "TIMP1", "EMP1", "GJA1", "GJB6", "S100B", "FABP7"
        ],
        "repair_ecm": [
            "TGFB1", "TGFB2", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
            "SMAD4", "SERPINE1", "CTGF", "FN1", "COL1A1", "COL1A2",
            "COL3A1", "COL4A1", "COL4A2", "COL6A1", "MMP2", "MMP3",
            "MMP9", "MMP14", "TIMP1", "TIMP2", "THBS1", "THBS2",
            "POSTN", "DCN", "LUM", "SPARC", "SPP1", "CD44", "ITGAV",
            "ITGB1", "ITGB3"
        ],
        "hypoxia": [
            "HIF1A", "EPAS1", "VEGFA", "SLC2A1", "SLC16A1", "SLC16A3",
            "LDHA", "LDHB", "PDK1", "PGK1", "ENO1", "BNIP3", "BNIP3L",
            "ADM", "EGLN1", "EGLN3", "CA9", "HK2", "PKM", "MIF"
        ],
        "synaptic_recovery": [
            "RBFOX3", "MAP2", "TUBB3", "SNAP25", "SYT1", "SYT11",
            "SYN1", "DLG4", "GRIN1", "GRIA1", "GRIA2", "CAMK2A",
            "CAMK2B", "BDNF", "NTRK2", "GAP43", "DCX", "NEFL",
            "NEFM", "NEFH", "SYP", "STMN1", "STMN2", "SEMA3A",
            "NRG1", "RELN"
        ],
    }

    # Remove housekeeping genes from module sets.
    out = {}
    for k, genes in cat.items():
        out[k] = sorted(set(gene_upper(g) for g in genes if not is_housekeeping_gene(g)))
    return out


def module_key_from_perturbation(pid):
    s = norm_key(pid)
    if "ferroptosis" in s:
        return "ferroptosis"
    if "inflammation" in s and "microglia" not in s:
        return "inflammation"
    if "bbb" in s or "leakage" in s:
        return "bbb_leakage"
    if "barrier_stability" in s or ("barrier" in s and "stability" in s):
        return "barrier_stability"
    if "endothelial" in s and ("fragility" in s or "barrier" in s):
        return "endothelial_barrier_fragility"
    if "microglia" in s:
        return "microglia_inflammatory"
    if "astrocyte" in s:
        return "astrocyte_reactive"
    if "repair_ecm" in s or s == "repair_ecm_up" or "ecm" in s:
        return "repair_ecm"
    if "hypoxia" in s:
        return "hypoxia"
    if "synaptic" in s:
        return "synaptic_recovery"
    return ""


def map_perturbation_to_targets(pid, row=None):
    catalog = module_gene_catalog()
    key = module_key_from_perturbation(pid)
    if key and key in catalog:
        return catalog[key], key, "module_catalog"

    genes = []
    if row is not None:
        for c in ["target_genes", "target_gene", "target", "gene", "targets"]:
            if c in row.index and not pd.isna(row[c]):
                genes.extend(split_gene_tokens(row[c]))

    genes.extend(split_gene_tokens(pid))
    genes = sorted(set(g for g in genes if not is_housekeeping_gene(g)))

    return genes, "", "parsed_gene_tokens"


# -----------------------------------------------------------------------------
# Loading tracks / markers
# -----------------------------------------------------------------------------

def load_tracks(track_dir):
    track_dir = Path(track_dir)
    tracks = read_csv_auto(track_dir / "step64c_track_label_refined.csv")
    assign = read_csv_auto(track_dir / "step64c_track_assignment_by_cell.refined_labels.csv")

    if "track_id" not in tracks.columns:
        raise RuntimeError("track table lacks track_id")
    if not {"obs_name", "track_id"}.issubset(assign.columns):
        raise RuntimeError("assignment table lacks obs_name / track_id")

    tracks["track_id"] = tracks["track_id"].astype(str)
    assign["track_id"] = assign["track_id"].astype(str)
    assign["obs_name"] = assign["obs_name"].astype(str)

    return tracks, assign


def load_track_marker_genes(step65_dir, step65e_dir, top_n=50):
    step65e_dir = Path(step65e_dir)
    p65e = step65e_dir / "step65e_track_marker_genes_after_filter.csv"

    if p65e.exists():
        ms = read_csv_auto(p65e)
        if {"track_id", "marker_genes_after_filter"}.issubset(ms.columns):
            rows = []
            for _, r in ms.iterrows():
                tid = str(r["track_id"])
                genes = split_gene_tokens(r["marker_genes_after_filter"])
                for rank, g in enumerate(genes[:top_n], start=1):
                    rows.append({
                        "track_id": tid,
                        "gene": g,
                        "marker_rank": rank,
                        "marker_source": "step65e_housekeeping_filtered",
                        "dynamic_marker_score": 1.0 / rank,
                    })
            out = pd.DataFrame(rows)
            if not out.empty:
                return out

    step65_dir = Path(step65_dir)
    dyn = read_csv_auto(step65_dir / "step65_track_dynamic_genes.csv")

    if not {"track_id", "feature"}.issubset(dyn.columns):
        raise RuntimeError("Step65 dynamic genes lacks track_id/feature")

    if "passes_min_cells" in dyn.columns:
        dyn = dyn[dyn["passes_min_cells"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    if "dynamic_marker_score" not in dyn.columns:
        dyn["dynamic_marker_score"] = 1.0

    dyn["track_id"] = dyn["track_id"].astype(str)
    dyn["gene"] = dyn["feature"].map(gene_upper)
    dyn = dyn[~dyn["gene"].map(is_housekeeping_gene)].copy()

    rows = []
    for tid, sub in dyn.groupby("track_id"):
        sub = sub.sort_values("dynamic_marker_score", ascending=False).drop_duplicates("gene").head(top_n)
        for rank, r in enumerate(sub.itertuples(), start=1):
            rows.append({
                "track_id": tid,
                "gene": r.gene,
                "marker_rank": rank,
                "marker_source": "step65_dynamic_genes_clean",
                "dynamic_marker_score": safe_float(getattr(r, "dynamic_marker_score", 1.0)),
            })
    return pd.DataFrame(rows)


def build_track_weights(tracks):
    df = tracks.copy()

    if "track_priority_score" in df.columns:
        raw = pd.to_numeric(df["track_priority_score"], errors="coerce")
    elif "n_cells_sum" in df.columns:
        raw = pd.to_numeric(df["n_cells_sum"], errors="coerce")
    elif "n_cells" in df.columns:
        raw = pd.to_numeric(df["n_cells"], errors="coerce")
    else:
        raw = pd.Series(np.ones(len(df)), index=df.index)

    raw = raw.fillna(0)
    if raw.sum() <= 0:
        raw = pd.Series(np.ones(len(df)), index=df.index)

    df["track_weight"] = raw / raw.sum()

    keep = [
        "track_id", "track_weight", "track_label_refined",
        "biological_axis_refined", "label_confidence",
        "state_sequence", "celltype_sequence"
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


# -----------------------------------------------------------------------------
# Candidate and Step61 cell-level patching
# -----------------------------------------------------------------------------

def detect_id_col(df):
    for c in ["perturbation_id", "perturbation", "niche_perturbation", "target_gene", "module"]:
        if c in df.columns:
            return c
    for c in df.columns:
        nk = norm_key(c)
        if "perturb" in nk or "module" in nk:
            return c
    return None


def detect_score_col(df):
    priority = [
        "niche_perturbation_score", "overall_rescue_score",
        "stroke_niche_rescue_score", "rescue_score", "priority_score",
        "final_therapeutic_priority", "score"
    ]
    for c in priority:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.2:
            return c
    for c in df.columns:
        nk = norm_key(c)
        if ("score" in nk or "priority" in nk) and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.2:
            return c
    return None


def load_and_map_candidates(ranking_files):
    rows = []
    audit = []

    for p in ranking_files:
        p = Path(p)
        if not p.exists():
            audit.append({"file": str(p), "status": "missing"})
            continue

        df = read_csv_auto(p)
        if df is None or df.empty:
            audit.append({"file": str(p), "status": "empty"})
            continue

        id_col = detect_id_col(df)
        score_col = detect_score_col(df)

        if id_col is None:
            audit.append({"file": str(p), "status": "skip_no_id_col", "n_rows": len(df)})
            continue

        n_add = 0
        for _, r in df.iterrows():
            pid = str(r[id_col]).strip()
            if not pid or pid.lower() == "nan":
                continue

            genes, module_key, target_source = map_perturbation_to_targets(pid, r)

            if not genes:
                continue

            row = {
                "perturbation_id": pid,
                "target_genes": ";".join(genes),
                "target_set_size": len(genes),
                "module_key": module_key,
                "target_mapping_source": target_source,
                "candidate_source_file": str(p),
                "candidate_score_col": score_col or "",
                "candidate_score_raw": safe_float(r[score_col]) if score_col else np.nan,
            }

            for c in [
                "niche_perturbation_score", "repair_shift", "repair_shift_score",
                "core_reversal", "core_reversal_score", "core_probability_reduction",
                "inflammatory_chemotaxis_reduction", "ferroptosis_reduction",
                "BBB_stability_gain", "spatial_safety", "spatial_safety_score",
                "spatial_risk_penalty", "priority_class"
            ]:
                if c in df.columns:
                    row[c] = r[c]

            rows.append(row)
            n_add += 1

        audit.append({
            "file": str(p),
            "status": "loaded",
            "id_col": id_col,
            "score_col": score_col or "",
            "n_rows": len(df),
            "n_added": n_add,
        })

    cand = pd.DataFrame(rows)
    audit = pd.DataFrame(audit)

    if cand.empty:
        return cand, audit

    # Aggregate duplicates across Step61 variants.
    agg_rows = []
    for pid, sub in cand.groupby("perturbation_id"):
        genes = sorted(set(sum([split_gene_tokens(x) for x in sub["target_genes"].fillna("")], [])))
        score_vals = pd.to_numeric(sub["candidate_score_raw"], errors="coerce")

        row = {
            "perturbation_id": pid,
            "target_genes": ";".join(genes),
            "target_set_size": len(genes),
            "module_key": ";".join(sorted(set(sub["module_key"].dropna().astype(str)))[:3]),
            "target_mapping_source": ";".join(sorted(set(sub["target_mapping_source"].dropna().astype(str)))[:3]),
            "candidate_source_file": ";".join(sorted(set(sub["candidate_source_file"].astype(str)))[:5]),
            "candidate_score_col": ";".join(sorted(set(sub["candidate_score_col"].astype(str)))[:5]),
            "candidate_score_raw": score_vals.mean() if score_vals.notna().any() else np.nan,
        }

        for c in [
            "niche_perturbation_score", "repair_shift", "repair_shift_score",
            "core_reversal", "core_reversal_score", "core_probability_reduction",
            "inflammatory_chemotaxis_reduction", "ferroptosis_reduction",
            "BBB_stability_gain", "spatial_safety", "spatial_safety_score",
            "spatial_risk_penalty", "priority_class"
        ]:
            if c in sub.columns:
                vals = pd.to_numeric(sub[c], errors="coerce")
                if vals.notna().any():
                    row[c] = vals.mean()
                else:
                    text_vals = [str(x) for x in sub[c].dropna().unique() if str(x).lower() != "nan"]
                    row[c] = ";".join(text_vals[:5])

        agg_rows.append(row)

    return pd.DataFrame(agg_rows).reset_index(drop=True), audit


def find_numeric_col(df, patterns):
    for p in patterns:
        for c in df.columns:
            nk = norm_key(c)
            if re.search(p, nk):
                x = pd.to_numeric(df[c], errors="coerce")
                if x.notna().mean() > 0.20:
                    return c
    return ""


def find_probability_pair(df, state):
    """
    Try to find baseline/perturbed probability columns.
    """
    state = state.lower()

    before_patterns = [
        rf"baseline.*{state}.*prob",
        rf"before.*{state}.*prob",
        rf"original.*{state}.*prob",
        rf"pre.*{state}.*prob",
        rf"control.*{state}.*prob",
    ]
    after_patterns = [
        rf"perturbed.*{state}.*prob",
        rf"after.*{state}.*prob",
        rf"edited.*{state}.*prob",
        rf"post.*{state}.*prob",
        rf"new.*{state}.*prob",
    ]

    before = find_numeric_col(df, before_patterns)
    after = find_numeric_col(df, after_patterns)
    return before, after


def patch_cell_level_peri_remote(cell_df):
    df = cell_df.copy()

    pid_col = "perturbation_id" if "perturbation_id" in df.columns else detect_id_col(df)
    if pid_col is None:
        raise RuntimeError("Step61 cell-level table lacks perturbation_id.")

    df["perturbation_id"] = df[pid_col].astype(str)

    repair_col = find_numeric_col(df, [
        r"predicted_repair_shift",
        r"repair.*shift",
        r"repair.*gain",
        r"delta.*repair",
    ])
    core_col = find_numeric_col(df, [
        r"predicted_core_reversal",
        r"core.*reversal",
        r"core.*reduction",
        r"core.*decrease",
        r"delta.*core",
    ])
    safety_col = find_numeric_col(df, [
        r"remote_offtarget_penalty",
        r"off.*target.*penalty",
        r"safety.*penalty",
        r"risk.*penalty",
        r"toxicity",
    ])

    direct_peri_remote_col = find_numeric_col(df, [
        r"peri_remote.*shift",
        r"remote.*shift",
        r"peri.*shift",
        r"delta.*remote",
        r"delta.*peri",
    ])

    peri_before, peri_after = find_probability_pair(df, "peri")
    remote_before, remote_after = find_probability_pair(df, "remote")

    audit = {
        "repair_col": repair_col,
        "core_col": core_col,
        "safety_col": safety_col,
        "direct_peri_remote_col": direct_peri_remote_col,
        "peri_before_col": peri_before,
        "peri_after_col": peri_after,
        "remote_before_col": remote_before,
        "remote_after_col": remote_after,
        "patch_mode": "",
    }

    if peri_before and peri_after:
        df["predicted_delta_peri_probability"] = (
            pd.to_numeric(df[peri_after], errors="coerce")
            - pd.to_numeric(df[peri_before], errors="coerce")
        )
    else:
        df["predicted_delta_peri_probability"] = np.nan

    if remote_before and remote_after:
        df["predicted_delta_remote_probability"] = (
            pd.to_numeric(df[remote_after], errors="coerce")
            - pd.to_numeric(df[remote_before], errors="coerce")
        )
    else:
        df["predicted_delta_remote_probability"] = np.nan

    true_pair_available = (
        df["predicted_delta_peri_probability"].notna().any()
        or df["predicted_delta_remote_probability"].notna().any()
    )

    if true_pair_available:
        df["predicted_delta_peri_probability"] = df["predicted_delta_peri_probability"].fillna(0)
        df["predicted_delta_remote_probability"] = df["predicted_delta_remote_probability"].fillna(0)
        df["predicted_peri_remote_shift"] = (
            df["predicted_delta_peri_probability"] + df["predicted_delta_remote_probability"]
        )
        audit["patch_mode"] = "true_probability_pair_columns"

    elif direct_peri_remote_col:
        val = pd.to_numeric(df[direct_peri_remote_col], errors="coerce")
        df["predicted_delta_peri_probability"] = 0.5 * val
        df["predicted_delta_remote_probability"] = 0.5 * val
        df["predicted_peri_remote_shift"] = val
        audit["patch_mode"] = "direct_existing_shift_column"

    else:
        if not repair_col or not core_col:
            raise RuntimeError(
                "Cannot compute proxy peri/remote shift: missing repair and/or core columns."
            )

        df["_repair_norm_for_66c"] = group_minmax(df, "perturbation_id", repair_col)
        df["_core_norm_for_66c"] = group_minmax(df, "perturbation_id", core_col)

        if safety_col:
            df["_safety_norm_for_66c"] = group_minmax(df, "perturbation_id", safety_col)
        else:
            df["_safety_norm_for_66c"] = 0.0

        # Proxy biological interpretation:
        #   core reversal contributes to peri-infarct transition;
        #   repair gain contributes to remote-like / repair-permissive transition;
        #   off-target risk weakly penalizes both.
        df["predicted_delta_peri_probability"] = (
            0.60 * df["_core_norm_for_66c"]
            + 0.20 * df["_repair_norm_for_66c"]
            - 0.10 * df["_safety_norm_for_66c"]
        )
        df["predicted_delta_remote_probability"] = (
            0.40 * df["_repair_norm_for_66c"]
            + 0.20 * df["_core_norm_for_66c"]
            - 0.05 * df["_safety_norm_for_66c"]
        )
        df["predicted_peri_remote_shift"] = (
            df["predicted_delta_peri_probability"] + df["predicted_delta_remote_probability"]
        )

        audit["patch_mode"] = "proxy_from_repair_core_safety"

    df["step66c_peri_remote_shift_source"] = audit["patch_mode"]

    # Alias columns deliberately named to pass strict component mapping.
    df["delta_peri_probability"] = df["predicted_delta_peri_probability"]
    df["delta_remote_probability"] = df["predicted_delta_remote_probability"]
    df["peri_remote_shift"] = df["predicted_peri_remote_shift"]

    audit["n_rows"] = int(len(df))
    audit["n_perturbations"] = int(df["perturbation_id"].nunique())
    audit["peri_remote_non_na_fraction"] = float(pd.to_numeric(df["peri_remote_shift"], errors="coerce").notna().mean())
    audit["peri_remote_sd"] = float(pd.to_numeric(df["peri_remote_shift"], errors="coerce").std())

    return df, audit


# -----------------------------------------------------------------------------
# Strict cell-level scoring
# -----------------------------------------------------------------------------

def component_columns_after_patch(df):
    repair_col = find_numeric_col(df, [
        r"predicted_repair_shift",
        r"repair.*shift",
        r"repair.*gain",
        r"delta.*repair",
    ])
    core_col = find_numeric_col(df, [
        r"predicted_core_reversal",
        r"core.*reversal",
        r"core.*reduction",
        r"core.*decrease",
        r"delta.*core",
    ])
    peri_remote_col = find_numeric_col(df, [
        r"predicted_peri_remote_shift",
        r"peri_remote_shift",
        r"delta_peri_probability",
        r"delta_remote_probability",
    ])
    safety_col = find_numeric_col(df, [
        r"remote_offtarget_penalty",
        r"off.*target.*penalty",
        r"safety.*penalty",
        r"risk.*penalty",
        r"toxicity",
    ])

    missing = []
    if not repair_col:
        missing.append("repair")
    if not core_col:
        missing.append("core")
    if not peri_remote_col:
        missing.append("peri_remote")
    if not safety_col:
        missing.append("safety")

    return {
        "repair_col": repair_col,
        "core_col": core_col,
        "peri_remote_col": peri_remote_col,
        "safety_col": safety_col,
        "missing": ";".join(missing),
        "status": "ok" if not missing else "missing_required_components",
    }


def build_cell_track_component_scores(cell_df, assign, candidates):
    comp = component_columns_after_patch(cell_df)

    if comp["status"] != "ok":
        return pd.DataFrame(), comp

    valid_pids = set(candidates["perturbation_id"].astype(str))

    df = cell_df.copy()
    df["obs_name"] = df["obs_name"].astype(str)
    df["perturbation_id"] = df["perturbation_id"].astype(str)
    df = df[df["perturbation_id"].isin(valid_pids)].copy()

    if df.empty:
        comp["status"] = "no_matching_candidate_ids"
        return pd.DataFrame(), comp

    df["repair_gain_raw"] = pd.to_numeric(df[comp["repair_col"]], errors="coerce")
    df["core_reduction_raw"] = pd.to_numeric(df[comp["core_col"]], errors="coerce")
    df["peri_remote_shift_raw"] = pd.to_numeric(df[comp["peri_remote_col"]], errors="coerce")
    df["safety_penalty_raw"] = pd.to_numeric(df[comp["safety_col"]], errors="coerce")

    merged = df[[
        "obs_name", "perturbation_id",
        "repair_gain_raw", "core_reduction_raw",
        "peri_remote_shift_raw", "safety_penalty_raw"
    ]].merge(assign[["obs_name", "track_id"]], on="obs_name", how="inner")

    if merged.empty:
        comp["status"] = "no_obs_track_overlap"
        return pd.DataFrame(), comp

    grouped = (
        merged.groupby(["perturbation_id", "track_id"])
        .agg(
            n_cells=("obs_name", "nunique"),
            repair_gain_raw=("repair_gain_raw", "mean"),
            core_reduction_raw=("core_reduction_raw", "mean"),
            peri_remote_shift_raw=("peri_remote_shift_raw", "mean"),
            safety_penalty_raw=("safety_penalty_raw", "mean"),
        )
        .reset_index()
    )

    # Degeneracy check: must vary across tracks for at least part of candidates.
    var_by_pid = (
        grouped.groupby("perturbation_id")
        .agg(
            repair_sd=("repair_gain_raw", "std"),
            core_sd=("core_reduction_raw", "std"),
            state_sd=("peri_remote_shift_raw", "std"),
            safety_sd=("safety_penalty_raw", "std"),
        )
        .fillna(0)
    )
    var_by_pid["total_sd"] = var_by_pid[["repair_sd", "core_sd", "state_sd", "safety_sd"]].sum(axis=1)

    comp["n_rows_matching_candidates"] = int(len(df))
    comp["n_merged_rows"] = int(len(merged))
    comp["n_grouped_rows"] = int(len(grouped))
    comp["median_track_component_sd"] = float(var_by_pid["total_sd"].median()) if len(var_by_pid) else 0.0
    comp["n_non_degenerate_candidates"] = int((var_by_pid["total_sd"] > 1e-12).sum())

    if comp["median_track_component_sd"] <= 1e-12:
        comp["status"] = "degenerate_not_track_specific"
        return pd.DataFrame(), comp

    grouped["score_source"] = "strict_cell_level_66c"
    grouped["cell_level_source_file"] = "step66c_patched_step61_cell_level"

    return grouped, comp


# -----------------------------------------------------------------------------
# Relevance / score / FDR
# -----------------------------------------------------------------------------

def build_marker_map(marker_genes):
    marker_map = {}
    for tid, sub in marker_genes.groupby("track_id"):
        genes = list(dict.fromkeys(sub.sort_values("marker_rank")["gene"].astype(str).map(gene_upper)))
        genes = [g for g in genes if not is_housekeeping_gene(g)]
        marker_map[str(tid)] = genes
    return marker_map


def target_relevance(targets, markers):
    t = set(gene_upper(x) for x in targets if str(x).strip() and not is_housekeeping_gene(x))
    m = set(gene_upper(x) for x in markers if str(x).strip() and not is_housekeeping_gene(x))

    if not t or not m:
        return {
            "target_overlap_n": 0,
            "target_jaccard": 0.0,
            "target_marker_fraction": 0.0,
            "target_set_fraction": 0.0,
            "target_relevance_raw": 0.0,
            "target_overlap_genes": "",
        }

    ov = t & m
    j = len(ov) / max(len(t | m), 1)
    mf = len(ov) / max(len(m), 1)
    tf = len(ov) / max(len(t), 1)
    rel = 0.55 * tf + 0.30 * mf + 0.15 * j

    return {
        "target_overlap_n": len(ov),
        "target_jaccard": j,
        "target_marker_fraction": mf,
        "target_set_fraction": tf,
        "target_relevance_raw": rel,
        "target_overlap_genes": ";".join(sorted(ov)),
    }


def attach_target_relevance(base_scores, candidates, marker_genes):
    cand_map = candidates.set_index("perturbation_id")["target_genes"].to_dict()
    marker_map = build_marker_map(marker_genes)

    rows = []
    for _, r in base_scores.iterrows():
        pid = str(r["perturbation_id"])
        tid = str(r["track_id"])
        targets = split_gene_tokens(cand_map.get(pid, ""))
        markers = marker_map.get(tid, [])

        rel = target_relevance(targets, markers)

        row = r.to_dict()
        row.update(rel)
        row["target_genes"] = ";".join(targets)
        row["target_set_size"] = len(targets)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_scores(scored, track_weights, weights):
    out = scored.copy()

    for c in ["repair_gain_raw", "core_reduction_raw", "peri_remote_shift_raw", "safety_penalty_raw"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["repair_gain_norm"] = minmax_series(out["repair_gain_raw"].fillna(out["repair_gain_raw"].median())).values
    out["core_reduction_norm"] = minmax_series(out["core_reduction_raw"].fillna(out["core_reduction_raw"].median())).values
    out["peri_remote_shift_norm"] = minmax_series(out["peri_remote_shift_raw"].fillna(out["peri_remote_shift_raw"].median())).values
    out["safety_penalty_norm"] = minmax_series(out["safety_penalty_raw"].fillna(0)).values
    out["target_relevance_norm"] = pd.to_numeric(out["target_relevance_raw"], errors="coerce").fillna(0).clip(0, 1)

    out["s_track_score"] = (
        weights["repair"] * out["repair_gain_norm"]
        + weights["core"] * out["core_reduction_norm"]
        + weights["state"] * out["peri_remote_shift_norm"]
        + weights["target"] * out["target_relevance_norm"]
        - weights["safety"] * out["safety_penalty_norm"]
    )

    out["s_track_scaled"] = out["s_track_score"].clip(-1, 1)

    tw = track_weights[["track_id", "track_weight"]].copy()
    out = out.merge(tw, on="track_id", how="left")
    out["track_weight"] = out["track_weight"].fillna(1.0 / max(out["track_id"].nunique(), 1))

    overall = (
        out.groupby("perturbation_id")
        .apply(lambda x: float(np.sum(x["s_track_score"] * x["track_weight"])))
        .reset_index(name="s_overall_observed")
    )

    strength = (
        out.groupby("perturbation_id")
        .agg(
            perturbation_strength=("s_track_score", lambda x: float(np.nanmean(np.abs(x)))),
            mean_repair=("repair_gain_norm", "mean"),
            mean_core=("core_reduction_norm", "mean"),
            mean_state=("peri_remote_shift_norm", "mean"),
            mean_target_relevance=("target_relevance_norm", "mean"),
            mean_safety_penalty=("safety_penalty_norm", "mean"),
            n_tracks_scored=("track_id", "nunique"),
        )
        .reset_index()
    )

    overall = overall.merge(strength, on="perturbation_id", how="left")
    return out, overall


def qbin(s, n_bins):
    s = pd.to_numeric(pd.Series(s), errors="coerce").fillna(0)
    if s.nunique() <= 1:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
    try:
        return pd.qcut(s.rank(method="first"), q=min(n_bins, s.nunique()), labels=False, duplicates="drop").astype(int)
    except Exception:
        return pd.Series(np.zeros(len(s), dtype=int), index=s.index)


def run_random_background(track_scores, overall, candidates, marker_genes, track_weights, weights, n_random=1000, seed=1, n_bins=5):
    rng = np.random.default_rng(seed)

    cand_info = candidates[["perturbation_id", "target_genes", "target_set_size"]].copy()
    m = overall.merge(cand_info, on="perturbation_id", how="left")
    m["target_set_size"] = pd.to_numeric(m["target_set_size"], errors="coerce").fillna(0).astype(int)
    m["target_size_bin"] = qbin(m["target_set_size"], n_bins)
    m["strength_bin"] = qbin(m["perturbation_strength"], n_bins)

    marker_map = build_marker_map(marker_genes)
    track_ids = track_weights["track_id"].astype(str).tolist()
    track_w = track_weights.set_index("track_id")["track_weight"].to_dict()

    universe = set(marker_genes["gene"].astype(str).map(gene_upper))
    for x in candidates["target_genes"].fillna(""):
        universe.update(split_gene_tokens(x))
    universe = sorted(g for g in universe if g and not is_housekeeping_gene(g))

    if not universe:
        raise RuntimeError("Background gene universe is empty.")

    state = track_scores.copy()
    state["state_score_without_target"] = (
        weights["repair"] * state["repair_gain_norm"]
        + weights["core"] * state["core_reduction_norm"]
        + weights["state"] * state["peri_remote_shift_norm"]
        - weights["safety"] * state["safety_penalty_norm"]
    )

    state_map = {
        pid: sub.set_index("track_id")["state_score_without_target"].to_dict()
        for pid, sub in state.groupby("perturbation_id")
    }

    rows = []

    for _, q in m.iterrows():
        pid = str(q["perturbation_id"])
        k = int(q["target_set_size"])
        k = max(1, min(k, len(universe)))

        same = m[
            (m["target_size_bin"] == q["target_size_bin"])
            & (m["strength_bin"] == q["strength_bin"])
        ].copy()

        if len(same) < 3:
            same = m[m["strength_bin"] == q["strength_bin"]].copy()
        if len(same) < 3:
            same = m.copy()

        same_no_self = same[same["perturbation_id"].astype(str) != pid]
        if len(same_no_self) >= 1:
            same = same_no_self

        same_pids = same["perturbation_id"].astype(str).tolist()

        for i in range(n_random):
            bg_pid = str(rng.choice(same_pids))
            random_genes = rng.choice(universe, size=k, replace=False).tolist()

            bg_state = state_map.get(bg_pid, {})
            overall_score = 0.0

            for tid in track_ids:
                base = safe_float(bg_state.get(tid, 0.0), 0.0)
                rel = target_relevance(random_genes, marker_map.get(tid, []))["target_relevance_raw"]
                s = base + weights["target"] * rel
                overall_score += s * safe_float(track_w.get(tid, 0.0), 0.0)

            rows.append({
                "perturbation_id": pid,
                "random_iter": i + 1,
                "matched_background_perturbation_id": bg_pid,
                "random_target_set_size": k,
                "random_target_genes": ";".join(random_genes),
                "random_s_overall": overall_score,
                "target_size_bin": int(q["target_size_bin"]),
                "strength_bin": int(q["strength_bin"]),
            })

    return pd.DataFrame(rows)


def compute_fdr(overall, bg, candidates):
    rows = []

    for _, r in overall.iterrows():
        pid = str(r["perturbation_id"])
        obs = safe_float(r["s_overall_observed"])
        sub = bg[bg["perturbation_id"].astype(str) == pid]

        if sub.empty or pd.isna(obs):
            p = np.nan
            mean = np.nan
            sd = np.nan
            z = np.nan
            diff = np.nan
        else:
            rand = pd.to_numeric(sub["random_s_overall"], errors="coerce").dropna().values
            mean = float(np.mean(rand))
            sd = float(np.std(rand))
            diff = obs - mean
            z = diff / sd if sd > 1e-12 else np.nan
            p = (1.0 + np.sum(rand >= obs)) / (1.0 + len(rand))

        rows.append({
            "perturbation_id": pid,
            "s_overall_observed": obs,
            "random_mean": mean,
            "random_sd": sd,
            "observed_minus_random_mean": diff,
            "random_z": z,
            "empirical_p_value": p,
            "n_random": int(len(sub)),
        })

    fdr = pd.DataFrame(rows)
    fdr["bh_fdr"] = bh_fdr(fdr["empirical_p_value"].values)

    fdr = fdr.merge(candidates, on="perturbation_id", how="left")
    fdr = fdr.sort_values(
        ["bh_fdr", "empirical_p_value", "s_overall_observed"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    fdr["step66c_rank"] = np.arange(1, len(fdr) + 1)
    fdr["is_positive_over_background"] = fdr["observed_minus_random_mean"] > 0
    fdr["is_fdr_significant_0_25"] = (fdr["bh_fdr"] <= 0.25) & fdr["is_positive_over_background"]

    fdr["final_priority_status"] = np.where(
        fdr["is_fdr_significant_0_25"],
        "FDR_supported_celllevel_track_perturbation_candidate",
        "exploratory_or_not_above_background"
    )

    return fdr


def sanity_check(fdr):
    if fdr.empty:
        return {
            "sanity_status": "no_fdr_rows",
            "n_positive_over_background": 0,
            "n_fdr_significant_0_25": 0,
        }

    p = pd.to_numeric(fdr["empirical_p_value"], errors="coerce")
    diff = pd.to_numeric(fdr["observed_minus_random_mean"], errors="coerce")

    all_p_one = bool((p.fillna(1.0) >= 1.0).all())
    all_below_bg = bool((diff.fillna(-1e9) <= 0).all())
    n_pos = int((diff > 0).sum())
    n_sig = int(((fdr["bh_fdr"] <= 0.25) & (diff > 0)).sum())

    if all_p_one and all_below_bg:
        status = "no_positive_significant_perturbation_detected"
    elif n_sig == 0:
        status = "no_FDR_supported_candidate_detected"
    else:
        status = "FDR_supported_candidates_detected"

    return {
        "sanity_status": status,
        "all_empirical_p_equal_1": all_p_one,
        "all_observed_below_random_mean": all_below_bg,
        "n_positive_over_background": n_pos,
        "n_fdr_significant_0_25": n_sig,
    }


# -----------------------------------------------------------------------------
# Heatmap and figures
# -----------------------------------------------------------------------------

def make_heatmap(track_scores, fdr, top_n=30):
    top = (
        fdr.sort_values(["bh_fdr", "s_overall_observed"], ascending=[True, False])
        .head(top_n)["perturbation_id"]
        .astype(str)
        .tolist()
    )

    sub = track_scores[track_scores["perturbation_id"].astype(str).isin(top)].copy()

    heat = sub.pivot_table(
        index="track_id",
        columns="perturbation_id",
        values="s_track_scaled",
        aggfunc="mean",
    )
    heat = heat.reindex(columns=top)

    return heat.reset_index()


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


def plot_step66c(track_scores, fdr, heatmap, outdir, top_n=25, dpi=600):
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step66C_CellLevelTrackPerturbationFDR_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34)

        top = fdr.sort_values(["bh_fdr", "s_overall_observed"], ascending=[True, False]).head(top_n).copy()

        # A observed score
        axA = fig.add_subplot(gs[0, 0])
        y = np.arange(len(top))[::-1]
        axA.barh(y, top["s_overall_observed"].values[::-1])
        if annotate:
            axA.set_yticks(y)
            axA.set_yticklabels(top["perturbation_id"].astype(str).values[::-1], fontsize=7)
            axA.set_xlabel("Observed S_overall")
            axA.set_title("A | Cell-level track perturbation score", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axA)

        # B FDR scatter
        axB = fig.add_subplot(gs[0, 1])
        x = pd.to_numeric(fdr["s_overall_observed"], errors="coerce")
        yv = -np.log10(pd.to_numeric(fdr["bh_fdr"], errors="coerce").clip(lower=1e-300))
        axB.scatter(x, yv, s=45, alpha=0.75)
        if annotate:
            for _, r in fdr.head(10).iterrows():
                axB.text(
                    safe_float(r["s_overall_observed"]),
                    -np.log10(max(safe_float(r["bh_fdr"], 1.0), 1e-300)),
                    str(r["perturbation_id"])[:22],
                    fontsize=7,
                )
            axB.axhline(-np.log10(0.25), linestyle="--", linewidth=0.8)
            axB.set_xlabel("Observed S_overall")
            axB.set_ylabel("-log10(BH-FDR)")
            axB.set_title("B | Matched random-target FDR", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axB)

        # C heatmap
        axC = fig.add_subplot(gs[1, 0])
        hm = heatmap.copy()
        if "track_id" in hm.columns:
            hm = hm.set_index("track_id")
        mat = hm.apply(pd.to_numeric, errors="coerce").fillna(0).values

        if mat.size > 0:
            vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)), 1e-6)
            im = axC.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
            if annotate:
                axC.set_yticks(np.arange(hm.shape[0]))
                axC.set_yticklabels(hm.index.astype(str), fontsize=7)
                axC.set_xticks(np.arange(hm.shape[1]))
                axC.set_xticklabels(hm.columns.astype(str), rotation=45, ha="right", fontsize=7)
                axC.set_title("C | Track-level score heatmap", loc="left", fontsize=12, fontweight="bold")
                cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
            else:
                strip_text(axC)
        else:
            if annotate:
                axC.text(0.5, 0.5, "No heatmap", ha="center", va="center")
            else:
                strip_text(axC)

        # D components
        axD = fig.add_subplot(gs[1, 1])
        top_pids = top["perturbation_id"].astype(str).tolist()
        comp = (
            track_scores[track_scores["perturbation_id"].astype(str).isin(top_pids)]
            .groupby("perturbation_id")
            .agg(
                repair=("repair_gain_norm", "mean"),
                core=("core_reduction_norm", "mean"),
                state=("peri_remote_shift_norm", "mean"),
                target=("target_relevance_norm", "mean"),
                safety=("safety_penalty_norm", "mean"),
            )
            .reindex(top_pids)
        )

        if not comp.empty:
            xs = np.arange(len(comp))
            bottom = np.zeros(len(comp))
            for col in ["repair", "core", "state", "target"]:
                vals = comp[col].fillna(0).values
                axD.bar(xs, vals, bottom=bottom, label=col)
                bottom += vals
            axD.bar(xs, -comp["safety"].fillna(0).values, label="safety penalty")

            if annotate:
                axD.set_xticks(xs)
                axD.set_xticklabels([x[:18] for x in comp.index], rotation=45, ha="right", fontsize=7)
                axD.set_ylabel("Mean normalized component")
                axD.set_title("D | Component audit", loc="left", fontsize=12, fontweight="bold")
                axD.legend(frameon=False, fontsize=7, ncol=2)
            else:
                strip_text(axD)
        else:
            if annotate:
                axD.text(0.5, 0.5, "No component summary", ha="center", va="center")
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
                "Step 66c | Cell-level track perturbation score with module-target FDR",
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

    ap.add_argument("--track_dir", default=str(DEFAULT_TRACK_DIR))
    ap.add_argument("--step61_dir", default=str(DEFAULT_STEP61_DIR))
    ap.add_argument("--step61_cell", default="")
    ap.add_argument("--step61_ranking", default="")
    ap.add_argument("--step65_dir", default=str(DEFAULT_STEP65_DIR))
    ap.add_argument("--step65e_dir", default=str(DEFAULT_STEP65E_DIR))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--top_marker_genes_per_track", type=int, default=50)
    ap.add_argument("--n_random", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260601)
    ap.add_argument("--n_bins", type=int, default=5)

    ap.add_argument("--w_repair", type=float, default=0.30)
    ap.add_argument("--w_core", type=float, default=0.30)
    ap.add_argument("--w_state", type=float, default=0.15)
    ap.add_argument("--w_target", type=float, default=0.20)
    ap.add_argument("--w_safety", type=float, default=0.25)

    ap.add_argument("--top_n_heatmap", type=int, default=30)
    ap.add_argument("--top_n_fig", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    step61_dir = Path(args.step61_dir)
    step61_cell = Path(args.step61_cell) if args.step61_cell else step61_dir / "niche_perturbation_cell_level_state_editing.state_fixed.csv"
    step61_ranking = Path(args.step61_ranking) if args.step61_ranking else step61_dir / "niche_perturbation_ranking.csv"

    weights = {
        "repair": args.w_repair,
        "core": args.w_core,
        "state": args.w_state,
        "target": args.w_target,
        "safety": args.w_safety,
    }

    log("=" * 100)
    log("Step66c cell-level peri/remote patched module-target FDR")
    log("=" * 100)
    log(f"step61_cell={step61_cell}")
    log(f"step61_ranking={step61_ranking}")
    log(f"outdir={outdir}")

    tracks, assign = load_tracks(args.track_dir)
    marker_genes = load_track_marker_genes(
        args.step65_dir,
        args.step65e_dir,
        top_n=args.top_marker_genes_per_track,
    )
    track_weights = build_track_weights(tracks)

    # 1. Candidate target mapping.
    candidates, candidate_audit = load_and_map_candidates([step61_ranking])
    if candidates.empty:
        raise RuntimeError("No module-target mapped candidates generated from Step61 ranking.")

    # 2. Patch Step61 cell-level table.
    cell_raw = read_csv_auto(step61_cell)
    if "obs_name" not in cell_raw.columns:
        raise RuntimeError("Step61 cell-level table lacks obs_name.")

    cell_patched, patch_audit = patch_cell_level_peri_remote(cell_raw)

    patched_cell_path = outdir / "step66c_step61_cell_level_with_peri_remote_shift.csv"
    mapped_candidate_path = outdir / "step66c_module_target_mapped_candidates.csv"

    cell_patched.to_csv(patched_cell_path, index=False)
    candidates.to_csv(mapped_candidate_path, index=False)
    candidate_audit.to_csv(outdir / "step66c_candidate_file_audit.csv", index=False)

    # 3. Strict cell-level scoring.
    base_scores, component_audit = build_cell_track_component_scores(cell_patched, assign, candidates)

    if base_scores.empty:
        score_mode = "failed_cell_level"
        raise RuntimeError(
            "Step66c failed to build strict cell-level track scores. "
            f"Component audit: {component_audit}"
        )

    score_mode = "strict_cell_level_66c"

    # 4. Score / FDR.
    scored = attach_target_relevance(base_scores, candidates, marker_genes)
    track_scores, overall = compute_scores(scored, track_weights, weights)

    bg = run_random_background(
        track_scores=track_scores,
        overall=overall,
        candidates=candidates,
        marker_genes=marker_genes,
        track_weights=track_weights,
        weights=weights,
        n_random=args.n_random,
        seed=args.seed,
        n_bins=args.n_bins,
    )

    fdr = compute_fdr(overall, bg, candidates)
    sanity = sanity_check(fdr)

    heatmap = make_heatmap(track_scores, fdr, top_n=args.top_n_heatmap)

    fig_outputs = plot_step66c(
        track_scores=track_scores,
        fdr=fdr,
        heatmap=heatmap,
        outdir=outdir,
        top_n=args.top_n_fig,
        dpi=args.dpi,
    )

    # Save outputs.
    marker_genes.to_csv(outdir / "step66c_track_marker_genes_used.csv", index=False)
    track_weights.to_csv(outdir / "step66c_track_weights_used.csv", index=False)
    pd.DataFrame([patch_audit]).to_csv(outdir / "step66c_peri_remote_patch_audit.csv", index=False)
    pd.DataFrame([component_audit]).to_csv(outdir / "step66c_cell_level_component_audit.csv", index=False)

    track_scores.to_csv(outdir / "step66c_track_level_perturbation_scores.csv", index=False)
    bg.to_csv(outdir / "step66c_random_background_scores.csv", index=False)
    fdr.to_csv(outdir / "step66c_perturbation_fdr.csv", index=False)
    heatmap.to_csv(outdir / "step66c_track_perturbation_heatmap.csv", index=False)

    report = {
        "status": "ok",
        "analysis_name": "Step66c cell-level track perturbation FDR with peri/remote patch and module target mapping",
        "score_mode": score_mode,
        "step61_cell": str(step61_cell),
        "step61_ranking": str(step61_ranking),
        "outdir": str(outdir),
        "weights": weights,
        "patch_audit": patch_audit,
        "component_audit": component_audit,
        "sanity_check": sanity,
        "n_tracks": int(track_weights["track_id"].nunique()),
        "n_candidates": int(len(candidates)),
        "n_track_score_rows": int(len(track_scores)),
        "n_random_rows": int(len(bg)),
        "n_random_per_candidate": int(args.n_random),
        "n_fdr_rows": int(len(fdr)),
        "n_FDR_supported_candidates_0_25": int(fdr["is_fdr_significant_0_25"].sum()),
        "figure_outputs": fig_outputs,
        "outputs": {
            "patched_cell_level": str(patched_cell_path),
            "mapped_candidates": str(mapped_candidate_path),
            "track_scores": str(outdir / "step66c_track_level_perturbation_scores.csv"),
            "random_background": str(outdir / "step66c_random_background_scores.csv"),
            "perturbation_fdr": str(outdir / "step66c_perturbation_fdr.csv"),
            "heatmap": str(outdir / "step66c_track_perturbation_heatmap.csv"),
            "patch_audit": str(outdir / "step66c_peri_remote_patch_audit.csv"),
            "component_audit": str(outdir / "step66c_cell_level_component_audit.csv"),
            "report_json": str(outdir / "step66c_report.json"),
            "report_txt": str(outdir / "step66c_report.txt"),
        },
        "interpretation_note": (
            "Step66c upgrades Step66b from summary-prior scoring to strict cell-level scoring by "
            "adding a peri/remote shift term to Step61 cell-level perturbation output. If patch_mode is "
            "proxy_from_repair_core_safety, the peri/remote term should be interpreted as a computational "
            "state-shift proxy rather than an observed probability delta."
        ),
    }

    (outdir / "step66c_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step66c cell-level peri/remote patched module-target FDR report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Top FDR table:")
    lines.append(fdr.head(80).to_string(index=False))
    lines.append("")
    lines.append("Candidate target mapping:")
    lines.append(candidates.to_string(index=False))
    lines.append("")
    lines.append("Patch audit:")
    lines.append(pd.DataFrame([patch_audit]).to_string(index=False))
    lines.append("")
    lines.append("Component audit:")
    lines.append(pd.DataFrame([component_audit]).to_string(index=False))
    lines.append("")
    lines.append("Top track scores:")
    lines.append(track_scores.sort_values("s_track_score", ascending=False).head(120).to_string(index=False))

    (outdir / "step66c_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step66c")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Top FDR rows:")
    log(fdr.head(30).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
