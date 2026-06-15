#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
66b_strict_clean_track_perturbation_fdr.py

Purpose
-------
Strict clean version of Step66.

This fixes the initial Step66 issues:
  1. Do NOT auto-scan 37c backup / preexisting files.
  2. Use only final clean candidate shortlist / Step60 / Step61 / cleaned mini-CMap.
  3. Exclude decoy / housekeeping candidates from final FDR.
  4. Keep decoys only for calibration QC.
  5. Clean target parsing:
       Ccl2_down -> CCL2
       Spp1_Cd44_blockade -> SPP1;CD44
       Do not keep CCL2_DOWN as a gene.
  6. Strict cell-level component audit:
       must explicitly find repair_gain / delta_repair
       must explicitly find core_probability reduction
       must explicitly find peri/remote shift
       must explicitly find safety/off-target penalty
     If not, use strict_summary_track_prior and report it.
  7. Score sanity check:
       if all empirical_p = 1 and all observed < random_mean:
           no_positive_significant_perturbation_detected

Outputs
-------
outdir/
  step66b_final_candidate_perturbations_used.csv
  step66b_excluded_decoy_housekeeping_candidates_qc.csv
  step66b_candidate_file_audit.csv
  step66b_cell_level_component_audit.csv
  step66b_track_level_perturbation_scores.csv
  step66b_random_background_scores.csv
  step66b_perturbation_fdr.csv
  step66b_track_perturbation_heatmap.csv
  step66b_report.json
  step66b_report.txt
  Fig_Step66B_StrictCleanTrackPerturbationFDR_annotated.pdf/svg/png
  Fig_Step66B_StrictCleanTrackPerturbationFDR_clean_no_text.pdf/svg/png
"""

from pathlib import Path
import argparse
import json
import os
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
DEFAULT_STEP65_DIR = ROOT / "track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata"
DEFAULT_STEP65E_DIR = ROOT / "track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered"
DEFAULT_OUT = ROOT / "track_level_perturbation_fdr_66b_strict_clean"


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
    out[order] = np.clip(q, 0, 1)
    return out


def minmax(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def read_csv_auto(path, required=True, nrows=None):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None
    try:
        return pd.read_csv(p, nrows=nrows, low_memory=False)
    except Exception:
        return pd.read_csv(p, sep="\t", nrows=nrows, low_memory=False)


# -----------------------------------------------------------------------------
# Blacklist / target parsing
# -----------------------------------------------------------------------------

HOUSEKEEPING_EXACT = {
    "ACTB", "B2M", "GAPDH", "HPRT", "HPRT1", "MALAT1", "PPIA",
    "RPLP0", "RPS18", "RPL13A", "TUBA1A", "TUBA1B", "TUBB",
    "EEF1A1", "EEF2", "PGK1", "LDHA", "LDHB",
}

DIRECTION_WORDS = {
    "UP", "DOWN", "KD", "KO", "OE", "OVEREXPRESSION", "KNOCKDOWN",
    "BLOCKADE", "INHIBITION", "INHIBIT", "INHIBITOR", "ACTIVATE",
    "ACTIVATION", "AGONIST", "ANTAGONIST", "NEUTRAL", "DECOY",
    "GENE", "DRUG", "COMPOUND", "PERTURBATION", "PERTURBAGEN",
    "LR", "AXIS", "MODULE", "NAN", "NONE", "TRUE", "FALSE",
}

NON_GENE_WORDS = DIRECTION_WORDS | {
    "REPAIR", "CORE", "REMOTE", "PERI", "PERIINFARCT", "LESION",
    "STROKE", "NICHE", "RESCUE", "SCORE", "RANKING", "FINAL",
    "CLEAN", "SHORTLIST", "TARGET", "TARGETS", "SIGNATURE",
    "PATHWAY", "BACKGROUND", "RANDOM",
}

GENE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\.\-]{1,24}$")


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


def split_gene_like_tokens(text):
    """
    Clean parser:
      Ccl2_down -> CCL2
      Lgals9_Cd44_blockade -> LGALS9;CD44
      CCL2_DOWN is not retained as a gene.
    """
    if text is None or pd.isna(text):
        return []

    s = str(text)
    s = re.sub(r"[\[\]\(\)\{\}\"']", " ", s)
    s = re.sub(r"[_:/|,;]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    tokens = []
    for t in s.split(" "):
        t = t.strip()
        if not t:
            continue

        # Remove common suffixes if present as hyphenated form.
        t = re.sub(r"[-\.](down|up|kd|ko|oe|blockade)$", "", t, flags=re.I)

        gu = gene_upper(t)

        if gu in NON_GENE_WORDS:
            continue

        if not GENE_TOKEN_RE.match(t):
            continue

        if len(gu) < 2 or len(gu) > 25:
            continue

        # Avoid obvious numeric / accession-like tokens.
        if re.match(r"^ENC[A-Z0-9]+$", gu):
            continue
        if re.match(r"^[0-9]+$", gu):
            continue

        tokens.append(gu)

    return list(dict.fromkeys(tokens))


def clean_targets_from_row(row):
    priority_fields = [
        "target_gene", "target", "gene", "targets", "target_genes",
        "ligand", "receptor", "regulator", "perturbation_id",
        "perturbation", "compound", "drug",
    ]

    genes = []
    for c in priority_fields:
        if c in row.index and not pd.isna(row[c]):
            genes.extend(split_gene_like_tokens(row[c]))

    # Deduplicate.
    genes = list(dict.fromkeys(genes))

    # If perturbation_id generated token like CCL2 and DOWN separately, DOWN is removed.
    # Keep housekeeping in raw targets for QC; exclusion happens separately.
    return genes


def is_decoy_or_housekeeping_candidate(row):
    pid = str(row.get("perturbation_id", "")).upper()
    ptype = str(row.get("perturbation_type", "")).upper()
    mech = str(row.get("mechanism_hint", "")).upper()
    direction = str(row.get("direction", "")).upper()
    target_genes = split_gene_like_tokens(row.get("target_genes", ""))

    reasons = []

    if "DECOY" in pid or "DECOY" in ptype or "NEGATIVE" in ptype or "CALIBRATION" in mech:
        reasons.append("decoy_or_negative_control")

    if direction in {"NEUTRAL", "DECOY"}:
        reasons.append("neutral_decoy_direction")

    # Exclude if primary target is housekeeping.
    primary_target = ""
    if "target_gene" in row.index and not pd.isna(row.get("target_gene", np.nan)):
        parsed = split_gene_like_tokens(row.get("target_gene"))
        primary_target = parsed[0] if parsed else ""

    if primary_target and is_housekeeping_gene(primary_target):
        reasons.append("primary_target_housekeeping")

    if target_genes and all(is_housekeeping_gene(g) for g in target_genes):
        reasons.append("all_targets_housekeeping")

    # Explicit housekeeping ID, e.g. Gapdh_decoy or Actb.
    pid_tokens = split_gene_like_tokens(pid)
    if pid_tokens and any(is_housekeeping_gene(g) for g in pid_tokens):
        if "DECOY" in pid or len(pid_tokens) <= 2:
            reasons.append("housekeeping_like_perturbation_id")

    return len(reasons) > 0, ";".join(sorted(set(reasons)))


# -----------------------------------------------------------------------------
# Track / marker loading
# -----------------------------------------------------------------------------

def load_tracks(track_dir):
    track_dir = Path(track_dir)
    tracks_p = track_dir / "step64c_track_label_refined.csv"
    assign_p = track_dir / "step64c_track_assignment_by_cell.refined_labels.csv"

    tracks = read_csv_auto(tracks_p)
    assign = read_csv_auto(assign_p)

    if "track_id" not in tracks.columns:
        raise RuntimeError(f"{tracks_p} lacks track_id")
    if not {"obs_name", "track_id"}.issubset(assign.columns):
        raise RuntimeError(f"{assign_p} lacks obs_name/track_id")

    tracks["track_id"] = tracks["track_id"].astype(str)
    assign["track_id"] = assign["track_id"].astype(str)
    assign["obs_name"] = assign["obs_name"].astype(str)

    return tracks, assign


def load_track_marker_genes(step65_dir, step65e_dir, top_n=50):
    step65_dir = Path(step65_dir)
    step65e_dir = Path(step65e_dir)

    # Prefer Step65e housekeeping-filtered marker gene sets.
    p65e = step65e_dir / "step65e_track_marker_genes_after_filter.csv"
    if p65e.exists():
        ms = read_csv_auto(p65e)
        if {"track_id", "marker_genes_after_filter"}.issubset(ms.columns):
            rows = []
            for _, r in ms.iterrows():
                tid = str(r["track_id"])
                genes = split_gene_like_tokens(r["marker_genes_after_filter"])
                for rank, g in enumerate(genes[:top_n], start=1):
                    if is_housekeeping_gene(g):
                        continue
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

    p65 = step65_dir / "step65_track_dynamic_genes.csv"
    dyn = read_csv_auto(p65)

    if not {"track_id", "feature"}.issubset(dyn.columns):
        raise RuntimeError(f"{p65} lacks track_id/feature")

    if "passes_min_cells" in dyn.columns:
        dyn = dyn[dyn["passes_min_cells"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    dyn["track_id"] = dyn["track_id"].astype(str)
    dyn["gene"] = dyn["feature"].map(gene_upper)
    dyn = dyn[~dyn["gene"].map(is_housekeeping_gene)].copy()

    if "dynamic_marker_score" not in dyn.columns:
        dyn["dynamic_marker_score"] = 1.0

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
        "track_id", "track_weight", "track_label_refined", "biological_axis_refined",
        "label_confidence", "state_sequence", "celltype_sequence"
    ]
    keep = [c for c in keep if c in df.columns]

    return df[keep].copy()


# -----------------------------------------------------------------------------
# Strict candidate discovery
# -----------------------------------------------------------------------------

def strict_candidate_discovery(root):
    """
    Only final / clean / shortlist / Step61 files.
    Explicitly exclude 37c backup/preexisting/raw all_perturbation scans.
    """
    root = Path(root)

    allow_patterns = [
        "**/step60*clean*shortlist*.csv",
        "**/*step60*clean*.csv",
        "**/*cleaned*mini*cmap*.csv",
        "**/*mini*cmap*shortlist*.csv",
        "**/*manuscript*shortlist*.csv",
        "**/*final*shortlist*.csv",
        "**/*therapeutic*priority*clean*.csv",
        "**/*candidate*perturbagen*clean*.csv",
        "**/niche_perturb_61*/niche_perturbation_ranking.csv",
        "**/*niche_perturbation_ranking.csv",
    ]

    exclude_terms = [
        "37c_all_variant_retraining",
        "preexisting_backup",
        "/backup/",
        "negative_decoy",
        "decoy_rescue",
        "all_perturbation_rescue_ranking",
        "gene_perturbation_rescue_ranking",
        "lr_axis_rescue_ranking",
        "ordinary_graph",
        "boundary_aware_graph",
        "no_graph",
        "run_",
        "random_background",
        "step66",
    ]

    files = []
    for pat in allow_patterns:
        for p in root.glob(pat):
            if not p.is_file() or p.stat().st_size == 0:
                continue
            low = str(p).lower()
            if any(x in low for x in exclude_terms):
                continue
            files.append(p)

    seen = set()
    out = []
    for p in files:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)

    return out


def detect_id_col(df):
    candidates = [
        "perturbation_id", "perturbation", "niche_perturbation",
        "perturbagen", "compound", "drug", "drug_name",
        "target_gene", "gene", "ligand_receptor_axis", "lr_axis",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        nk = norm_key(c)
        if "perturb" in nk or "compound" in nk or "drug" in nk:
            return c
    return None


def detect_score_col(df):
    priority = [
        "final_therapeutic_priority",
        "overall_rescue_score",
        "stroke_niche_rescue_score",
        "niche_perturbation_score",
        "rescue_score",
        "reversal_score",
        "repair_shift_score",
        "candidate_priority_score",
        "priority_score",
        "score",
    ]
    for c in priority:
        if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.2:
            return c

    for c in df.columns:
        nk = norm_key(c)
        if ("score" in nk or "priority" in nk) and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.2:
            return c

    return None


def build_pid(row, id_col):
    if id_col and id_col in row.index and not pd.isna(row[id_col]):
        s = str(row[id_col]).strip()
        if s and s.lower() != "nan":
            return s

    parts = []
    for c in ["target_gene", "ligand", "receptor", "drug", "compound"]:
        if c in row.index and not pd.isna(row[c]):
            parts.append(str(row[c]).strip())
    return "_".join(parts)


def load_strict_candidates(candidate_files):
    rows = []
    audit = []

    for p in candidate_files:
        try:
            df = read_csv_auto(p)
        except Exception as e:
            audit.append({
                "file": str(p),
                "status": "read_failed",
                "reason": str(e),
                "n_rows_loaded": 0,
            })
            continue

        if df is None or df.empty:
            audit.append({
                "file": str(p),
                "status": "empty",
                "reason": "",
                "n_rows_loaded": 0,
            })
            continue

        id_col = detect_id_col(df)
        score_col = detect_score_col(df)

        if id_col is None and not any(c in df.columns for c in ["target_gene", "ligand", "receptor", "compound", "drug"]):
            audit.append({
                "file": str(p),
                "status": "skip_no_candidate_id",
                "id_col": "",
                "score_col": score_col or "",
                "n_rows_loaded": len(df),
            })
            continue

        n_added = 0
        for _, r in df.iterrows():
            pid = build_pid(r, id_col)
            if not pid:
                continue

            targets = clean_targets_from_row(r)

            raw_score = safe_float(r[score_col]) if score_col else np.nan

            row = {
                "perturbation_id": pid,
                "target_genes": ";".join(targets),
                "target_set_size": len(targets),
                "candidate_score_raw": raw_score,
                "candidate_score_col": score_col or "",
                "candidate_source_file": str(p),
            }

            for c in [
                "perturbation_type", "target_gene", "ligand", "receptor",
                "direction", "mechanism_hint", "drug", "compound", "database",
                "priority_class", "safety_flag", "bbb_flag",
                "spatial_safety_score", "safety_penalty", "toxicity_risk",
                "repair_shift_score", "core_reversal_score",
                "core_probability_reduction", "lesion_core_probability_decrease",
                "remote_shift_score", "peri_shift_score",
            ]:
                if c in df.columns:
                    row[c] = r[c]

            rows.append(row)
            n_added += 1

        audit.append({
            "file": str(p),
            "status": "loaded",
            "id_col": id_col or "",
            "score_col": score_col or "",
            "n_rows_loaded": len(df),
            "n_candidates_added": n_added,
        })

    cand = pd.DataFrame(rows)
    audit = pd.DataFrame(audit)

    if cand.empty:
        return cand, pd.DataFrame(), audit

    # Aggregate duplicates.
    agg_rows = []
    for pid, sub in cand.groupby("perturbation_id"):
        targets = sorted(set(sum([split_gene_like_tokens(x) for x in sub["target_genes"].fillna("")], [])))
        score_vals = pd.to_numeric(sub["candidate_score_raw"], errors="coerce")

        row = {
            "perturbation_id": pid,
            "target_genes": ";".join(targets),
            "target_set_size": len(targets),
            "candidate_score_raw": score_vals.mean() if score_vals.notna().any() else np.nan,
            "candidate_score_col": ";".join(sorted(set(sub["candidate_score_col"].astype(str)))[:10]),
            "candidate_source_file": ";".join(sorted(set(sub["candidate_source_file"].astype(str)))[:10]),
        }

        for c in [
            "perturbation_type", "target_gene", "ligand", "receptor",
            "direction", "mechanism_hint", "drug", "compound", "database",
            "priority_class", "safety_flag", "bbb_flag",
            "spatial_safety_score", "safety_penalty", "toxicity_risk",
            "repair_shift_score", "core_reversal_score",
            "core_probability_reduction", "lesion_core_probability_decrease",
            "remote_shift_score", "peri_shift_score",
        ]:
            if c in sub.columns:
                vals = [str(x) for x in sub[c].dropna().unique() if str(x).lower() != "nan"]
                row[c] = ";".join(vals[:5])

        agg_rows.append(row)

    cand = pd.DataFrame(agg_rows)

    # Re-parse targets if empty.
    for i, r in cand.iterrows():
        if int(r["target_set_size"]) == 0:
            targets = split_gene_like_tokens(r["perturbation_id"])
            cand.loc[i, "target_genes"] = ";".join(targets)
            cand.loc[i, "target_set_size"] = len(targets)

    # Flag decoy / housekeeping.
    flags = cand.apply(is_decoy_or_housekeeping_candidate, axis=1)
    cand["is_decoy_or_housekeeping"] = [x[0] for x in flags]
    cand["exclude_reason"] = [x[1] for x in flags]

    excluded = cand[cand["is_decoy_or_housekeeping"]].copy()
    final = cand[~cand["is_decoy_or_housekeeping"]].copy()

    return final.reset_index(drop=True), excluded.reset_index(drop=True), audit


# -----------------------------------------------------------------------------
# Strict component mapping
# -----------------------------------------------------------------------------

def detect_obs_col(df):
    for c in ["obs_name", "cell_id", "barcode", "spot_id"]:
        if c in df.columns:
            return c
    return None


def detect_pid_col(df):
    for c in ["perturbation_id", "perturbation", "niche_perturbation", "target_gene", "compound", "drug"]:
        if c in df.columns:
            return c
    return None


def numeric_candidate_columns(df):
    cols = []
    for c in df.columns:
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().mean() > 0.25 and x.nunique(dropna=True) > 1:
            cols.append(c)
    return cols


def find_col_by_patterns(cols, positive_patterns, negative_patterns=None):
    negative_patterns = negative_patterns or []
    scored = []
    for c in cols:
        nk = norm_key(c)
        if any(re.search(p, nk) for p in positive_patterns) and not any(re.search(p, nk) for p in negative_patterns):
            scored.append(c)
    return scored[0] if scored else ""


def strict_component_mapping(df):
    """
    Must find all four:
      repair_gain / delta_repair
      core_probability reduction
      peri/remote shift
      safety/off-target penalty

    Returns mapping dict and status.
    """
    num_cols = numeric_candidate_columns(df)

    repair_col = find_col_by_patterns(
        num_cols,
        [
            r"(^|_)delta_?repair($|_)",
            r"repair_?gain",
            r"repair_?shift",
            r"repair.*increase",
            r"delta.*repair",
        ],
    )

    core_reduction_col = find_col_by_patterns(
        num_cols,
        [
            r"core.*reduction",
            r"core.*decrease",
            r"core.*reversal",
            r"lesion.*core.*decrease",
            r"core_probability_reduction",
        ],
    )

    core_delta_col = ""
    if not core_reduction_col:
        core_delta_col = find_col_by_patterns(
            num_cols,
            [
                r"(^|_)delta_?core($|_)",
                r"delta.*core.*prob",
                r"core.*delta",
            ],
        )

    peri_col = find_col_by_patterns(
        num_cols,
        [
            r"(^|_)delta_?peri($|_)",
            r"peri.*shift",
            r"peri.*gain",
            r"prob_peri.*delta",
            r"peri.*increase",
        ],
    )

    remote_col = find_col_by_patterns(
        num_cols,
        [
            r"(^|_)delta_?remote($|_)",
            r"remote.*shift",
            r"remote.*gain",
            r"prob_remote.*delta",
            r"remote.*increase",
        ],
    )

    # Safety: can be direct penalty/risk, or positive safety score.
    safety_penalty_col = find_col_by_patterns(
        num_cols,
        [
            r"safety.*penalty",
            r"risk.*penalty",
            r"spatial.*risk",
            r"off.*target",
            r"offtarget",
            r"toxicity",
            r"bbb.*conflict",
            r"unfavorable",
        ],
    )

    safety_score_col = ""
    if not safety_penalty_col:
        safety_score_col = find_col_by_patterns(
            num_cols,
            [
                r"spatial.*safety",
                r"safety.*score",
                r"bbb.*compatible",
            ],
        )

    missing = []
    if not repair_col:
        missing.append("repair_gain_or_delta_repair")
    if not core_reduction_col and not core_delta_col:
        missing.append("core_probability_reduction_or_delta_core")
    if not peri_col and not remote_col:
        missing.append("peri_or_remote_shift")
    if not safety_penalty_col and not safety_score_col:
        missing.append("safety_or_offtarget_penalty")

    status = "ok" if not missing else "missing_required_components"

    mapping = {
        "repair_col": repair_col,
        "core_reduction_col": core_reduction_col,
        "core_delta_col": core_delta_col,
        "peri_col": peri_col,
        "remote_col": remote_col,
        "safety_penalty_col": safety_penalty_col,
        "safety_score_col": safety_score_col,
        "status": status,
        "missing": ";".join(missing),
    }

    return mapping


def apply_strict_component_mapping(df, mapping):
    out = df.copy()

    out["repair_gain_raw"] = pd.to_numeric(out[mapping["repair_col"]], errors="coerce")

    if mapping["core_reduction_col"]:
        out["core_reduction_raw"] = pd.to_numeric(out[mapping["core_reduction_col"]], errors="coerce")
    else:
        # delta_core means after - before; lower core is good, so reduction = -delta.
        out["core_reduction_raw"] = -pd.to_numeric(out[mapping["core_delta_col"]], errors="coerce")

    parts = []
    if mapping["peri_col"]:
        parts.append(pd.to_numeric(out[mapping["peri_col"]], errors="coerce"))
    if mapping["remote_col"]:
        parts.append(pd.to_numeric(out[mapping["remote_col"]], errors="coerce"))

    out["peri_remote_shift_raw"] = sum(parts) if parts else np.nan

    if mapping["safety_penalty_col"]:
        out["safety_penalty_raw"] = pd.to_numeric(out[mapping["safety_penalty_col"]], errors="coerce")
    else:
        # positive safety score -> penalty = 1 - normalized safety.
        safety = pd.to_numeric(out[mapping["safety_score_col"]], errors="coerce")
        out["safety_penalty_raw"] = 1.0 - minmax(safety)

    return out


def load_strict_cell_level_scores(cell_files, assign, final_candidates):
    audit_rows = []
    score_frames = []

    valid_pids = set(final_candidates["perturbation_id"].astype(str))

    for p in cell_files:
        p = Path(p)
        if not p.exists() or p.stat().st_size == 0:
            audit_rows.append({"file": str(p), "status": "missing_or_empty"})
            continue

        try:
            df = read_csv_auto(p)
        except Exception as e:
            audit_rows.append({"file": str(p), "status": "read_failed", "reason": str(e)})
            continue

        obs_col = detect_obs_col(df)
        pid_col = detect_pid_col(df)

        if not obs_col or not pid_col:
            audit_rows.append({
                "file": str(p),
                "status": "skip_missing_obs_or_perturbation_id",
                "obs_col": obs_col or "",
                "pid_col": pid_col or "",
            })
            continue

        mapping = strict_component_mapping(df)

        audit = {
            "file": str(p),
            "obs_col": obs_col,
            "pid_col": pid_col,
            **mapping,
            "n_rows": int(len(df)),
        }

        if mapping["status"] != "ok":
            audit["status"] = "rejected_missing_required_components"
            audit_rows.append(audit)
            continue

        tmp = df.copy()
        tmp["obs_name"] = tmp[obs_col].astype(str)
        tmp["perturbation_id"] = tmp[pid_col].astype(str)

        tmp = tmp[tmp["perturbation_id"].isin(valid_pids)].copy()
        if tmp.empty:
            audit["status"] = "rejected_no_matching_final_candidate_ids"
            audit_rows.append(audit)
            continue

        tmp = apply_strict_component_mapping(tmp, mapping)

        merged = tmp[[
            "obs_name", "perturbation_id",
            "repair_gain_raw", "core_reduction_raw",
            "peri_remote_shift_raw", "safety_penalty_raw",
        ]].merge(assign[["obs_name", "track_id"]], on="obs_name", how="inner")

        if merged.empty:
            audit["status"] = "rejected_no_obs_track_overlap"
            audit_rows.append(audit)
            continue

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

        # Degeneracy audit: if every track has same values, reject as non-track-specific.
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
        median_total_sd = float(var_by_pid["total_sd"].median()) if len(var_by_pid) else 0.0

        audit["n_rows_matching_final_candidates"] = int(len(tmp))
        audit["n_merged_rows"] = int(len(merged))
        audit["n_grouped_rows"] = int(len(grouped))
        audit["median_track_component_sd"] = median_total_sd

        if median_total_sd <= 1e-12:
            audit["status"] = "rejected_degenerate_not_track_specific"
            audit_rows.append(audit)
            continue

        grouped["score_source"] = "strict_cell_level"
        grouped["cell_level_source_file"] = str(p)
        score_frames.append(grouped)

        audit["status"] = "accepted_strict_cell_level"
        audit_rows.append(audit)

    audit_df = pd.DataFrame(audit_rows)

    if not score_frames:
        return pd.DataFrame(), audit_df

    out = pd.concat(score_frames, ignore_index=True)

    out = (
        out.groupby(["perturbation_id", "track_id"])
        .agg(
            n_cells=("n_cells", "sum"),
            repair_gain_raw=("repair_gain_raw", "mean"),
            core_reduction_raw=("core_reduction_raw", "mean"),
            peri_remote_shift_raw=("peri_remote_shift_raw", "mean"),
            safety_penalty_raw=("safety_penalty_raw", "mean"),
            score_source=("score_source", "first"),
            cell_level_source_file=("cell_level_source_file", lambda x: ";".join(sorted(set(map(str, x)))[:5])),
        )
        .reset_index()
    )

    return out, audit_df


# -----------------------------------------------------------------------------
# Strict summary track prior fallback
# -----------------------------------------------------------------------------

def find_component_col(candidates, positive_patterns, negative_patterns=None):
    negative_patterns = negative_patterns or []
    for c in candidates.columns:
        nk = norm_key(c)
        if any(re.search(p, nk) for p in positive_patterns) and not any(re.search(p, nk) for p in negative_patterns):
            if pd.to_numeric(candidates[c], errors="coerce").notna().mean() > 0.1:
                return c
    return ""


def candidate_component_matrix(candidates):
    df = candidates.copy()

    repair_col = find_component_col(df, [
        r"repair.*shift",
        r"repair.*gain",
        r"repair.*increase",
        r"delta.*repair",
    ])

    core_col = find_component_col(df, [
        r"core.*reversal",
        r"core.*reduction",
        r"core.*decrease",
        r"lesion.*core.*decrease",
    ])

    state_col = find_component_col(df, [
        r"peri.*shift",
        r"remote.*shift",
        r"state.*shift",
        r"remote.*gain",
        r"peri.*gain",
    ])

    safety_penalty_col = find_component_col(df, [
        r"safety.*penalty",
        r"risk.*penalty",
        r"toxicity",
        r"off.*target",
        r"bbb.*conflict",
    ])

    safety_score_col = ""
    if not safety_penalty_col:
        safety_score_col = find_component_col(df, [
            r"spatial.*safety",
            r"safety.*score",
            r"bbb.*compatible",
        ])

    score_col = detect_score_col(df)

    if score_col:
        base = pd.to_numeric(df[score_col], errors="coerce")
        if "rank" in norm_key(score_col):
            base_norm = 1.0 - minmax(base)
        else:
            base_norm = minmax(base)
    else:
        base_norm = pd.Series(np.full(len(df), 0.5), index=df.index)

    out = pd.DataFrame({
        "perturbation_id": df["perturbation_id"].astype(str),
        "base_priority_norm": base_norm,
    })

    out["repair_component_norm"] = minmax(pd.to_numeric(df[repair_col], errors="coerce")) if repair_col else base_norm
    out["core_component_norm"] = minmax(pd.to_numeric(df[core_col], errors="coerce")) if core_col else base_norm
    out["state_component_norm"] = minmax(pd.to_numeric(df[state_col], errors="coerce")) if state_col else base_norm

    if safety_penalty_col:
        out["safety_penalty_norm_pre"] = minmax(pd.to_numeric(df[safety_penalty_col], errors="coerce"))
        safety_source = safety_penalty_col
    elif safety_score_col:
        out["safety_penalty_norm_pre"] = 1.0 - minmax(pd.to_numeric(df[safety_score_col], errors="coerce"))
        safety_source = f"1_minus_{safety_score_col}"
    else:
        out["safety_penalty_norm_pre"] = 0.0
        safety_source = "zero_fallback"

    meta = {
        "repair_col": repair_col or "base_priority_norm_fallback",
        "core_col": core_col or "base_priority_norm_fallback",
        "state_col": state_col or "base_priority_norm_fallback",
        "safety_col": safety_source,
        "score_col": score_col or "",
    }

    return out, meta


def track_axis_prior(tracks):
    rows = []
    for _, r in tracks.iterrows():
        tid = str(r["track_id"])
        axis = str(r.get("biological_axis_refined", "")).lower()
        label = str(r.get("track_label_refined", "")).lower()
        state = str(r.get("state_sequence", "")).lower()
        celltype = str(r.get("celltype_sequence", "")).lower()
        txt = " ".join([axis, label, state, celltype])

        repair = 1.0 if "repair" in txt or "remote" in txt else 0.65
        core = 1.0 if "core" in txt or "lesion" in txt else 0.65
        state_shift = 1.0 if "transition" in txt or "peri" in txt or "remote" in txt else 0.65

        # Sensitive tracks get higher safety penalty weight.
        safety = 1.0 if any(x in txt for x in ["neuron", "synaptic", "endothelial", "barrier", "vascular"]) else 0.65

        rows.append({
            "track_id": tid,
            "track_repair_prior": repair,
            "track_core_prior": core,
            "track_state_prior": state_shift,
            "track_safety_prior": safety,
        })

    return pd.DataFrame(rows)


def make_strict_summary_track_prior_scores(candidates, tracks):
    comp, meta = candidate_component_matrix(candidates)
    pri = track_axis_prior(tracks)

    rows = []
    for _, c in comp.iterrows():
        for _, t in pri.iterrows():
            rows.append({
                "perturbation_id": c["perturbation_id"],
                "track_id": t["track_id"],
                "n_cells": np.nan,
                "repair_gain_raw": safe_float(c["repair_component_norm"]) * safe_float(t["track_repair_prior"]),
                "core_reduction_raw": safe_float(c["core_component_norm"]) * safe_float(t["track_core_prior"]),
                "peri_remote_shift_raw": safe_float(c["state_component_norm"]) * safe_float(t["track_state_prior"]),
                "safety_penalty_raw": safe_float(c["safety_penalty_norm_pre"]) * safe_float(t["track_safety_prior"]),
                "score_source": "strict_summary_track_prior",
                "cell_level_source_file": "",
            })

    return pd.DataFrame(rows), meta


# -----------------------------------------------------------------------------
# Target relevance and scoring
# -----------------------------------------------------------------------------

def build_marker_map(marker_genes):
    marker_map = {}
    for tid, sub in marker_genes.groupby("track_id"):
        genes = list(dict.fromkeys(sub.sort_values("marker_rank")["gene"].astype(str).map(gene_upper)))
        marker_map[str(tid)] = [g for g in genes if not is_housekeeping_gene(g)]
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
    jaccard = len(ov) / max(len(t | m), 1)
    marker_fraction = len(ov) / max(len(m), 1)
    target_fraction = len(ov) / max(len(t), 1)

    rel = 0.55 * target_fraction + 0.30 * marker_fraction + 0.15 * jaccard

    return {
        "target_overlap_n": len(ov),
        "target_jaccard": jaccard,
        "target_marker_fraction": marker_fraction,
        "target_set_fraction": target_fraction,
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
        targets = split_gene_like_tokens(cand_map.get(pid, ""))
        targets = [g for g in targets if not is_housekeeping_gene(g)]
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

    out["repair_gain_norm"] = minmax(out["repair_gain_raw"].fillna(out["repair_gain_raw"].median())).values
    out["core_reduction_norm"] = minmax(out["core_reduction_raw"].fillna(out["core_reduction_raw"].median())).values
    out["peri_remote_shift_norm"] = minmax(out["peri_remote_shift_raw"].fillna(out["peri_remote_shift_raw"].median())).values
    out["safety_penalty_norm"] = minmax(out["safety_penalty_raw"].fillna(0)).values
    out["target_relevance_norm"] = pd.to_numeric(out["target_relevance_raw"], errors="coerce").fillna(0).clip(0, 1)

    # Do not minmax the final score; keep interpretable additive score.
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


# -----------------------------------------------------------------------------
# Random background
# -----------------------------------------------------------------------------

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

    # Universe: dynamic marker genes + final candidate targets, no housekeeping.
    universe = set(marker_genes["gene"].astype(str).map(gene_upper))
    for x in candidates["target_genes"].fillna(""):
        universe.update(split_gene_like_tokens(x))
    universe = sorted(g for g in universe if g and not is_housekeeping_gene(g))

    if not universe:
        raise RuntimeError("Background gene universe is empty after housekeeping filtering.")

    # State scores without target relevance.
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
            obs_minus_mean = np.nan
        else:
            rand = pd.to_numeric(sub["random_s_overall"], errors="coerce").dropna().values
            mean = float(np.mean(rand))
            sd = float(np.std(rand))
            z = (obs - mean) / sd if sd > 1e-12 else np.nan
            obs_minus_mean = obs - mean
            p = (1.0 + np.sum(rand >= obs)) / (1.0 + len(rand))

        rows.append({
            "perturbation_id": pid,
            "s_overall_observed": obs,
            "random_mean": mean,
            "random_sd": sd,
            "observed_minus_random_mean": obs_minus_mean,
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

    fdr["step66b_rank"] = np.arange(1, len(fdr) + 1)

    fdr["is_positive_over_background"] = fdr["observed_minus_random_mean"] > 0
    fdr["is_fdr_significant_0_25"] = (fdr["bh_fdr"] <= 0.25) & fdr["is_positive_over_background"]

    fdr["final_priority_status"] = np.where(
        fdr["is_fdr_significant_0_25"],
        "FDR_supported_track_perturbation_candidate",
        "exploratory_or_not_above_background"
    )

    return fdr


def sanity_check_fdr(fdr):
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
# Heatmap and plotting
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


def plot_step66b(track_scores, fdr, heatmap, outdir, top_n=25, dpi=600):
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step66B_StrictCleanTrackPerturbationFDR_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34)

        # A observed score
        axA = fig.add_subplot(gs[0, 0])
        top = fdr.sort_values(["bh_fdr", "s_overall_observed"], ascending=[True, False]).head(top_n).copy()
        y = np.arange(len(top))[::-1]
        axA.barh(y, top["s_overall_observed"].values[::-1])
        if annotate:
            axA.set_yticks(y)
            axA.set_yticklabels(top["perturbation_id"].astype(str).values[::-1], fontsize=7)
            axA.set_xlabel("Observed S_overall")
            axA.set_title("A | Strict clean perturbation score", loc="left", fontsize=12, fontweight="bold")
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
            axB.set_title("B | Matched random-background FDR", loc="left", fontsize=12, fontweight="bold")
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
                axC.set_title("C | Track-level scores", loc="left", fontsize=12, fontweight="bold")
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
            fig.suptitle("Step 66b | Strict clean track-level perturbation score with random-background FDR", fontsize=15, fontweight="bold")

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

    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--track_dir", default=str(DEFAULT_TRACK_DIR))
    ap.add_argument("--step65_dir", default=str(DEFAULT_STEP65_DIR))
    ap.add_argument("--step65e_dir", default=str(DEFAULT_STEP65E_DIR))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--candidate_files", default="", help="Comma-separated clean final candidate files. Empty = strict auto-discovery.")
    ap.add_argument("--cell_level_files", default="", help="Comma-separated cell-level files. Empty = strict auto-discovery from known Step61/60 dirs.")

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

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    weights = {
        "repair": args.w_repair,
        "core": args.w_core,
        "state": args.w_state,
        "target": args.w_target,
        "safety": args.w_safety,
    }

    log("=" * 100)
    log("Step66b strict clean track-level perturbation FDR")
    log("=" * 100)
    log(f"root={root}")
    log(f"outdir={outdir}")

    tracks, assign = load_tracks(args.track_dir)
    marker_genes = load_track_marker_genes(args.step65_dir, args.step65e_dir, top_n=args.top_marker_genes_per_track)
    track_weights = build_track_weights(tracks)

    # Candidate files.
    if args.candidate_files.strip():
        candidate_files = [Path(x.strip()) for x in args.candidate_files.split(",") if x.strip()]
    else:
        candidate_files = strict_candidate_discovery(root)

    final_candidates, excluded_candidates, candidate_audit = load_strict_candidates(candidate_files)

    if final_candidates.empty:
        raise RuntimeError("No final clean non-decoy candidates found. Provide --candidate_files explicitly.")

    # Cell-level files: intentionally conservative.
    if args.cell_level_files.strip():
        cell_files = [Path(x.strip()) for x in args.cell_level_files.split(",") if x.strip()]
    else:
        possible = [
            root / "niche_perturb_61/niche_perturbation_cell_level_state_editing.state_fixed.csv",
            root / "niche_perturb_61_strict_fig5_umap_61d/niche_perturbation_cell_level_state_editing.state_fixed.csv",
            root / "niche_perturb_61_strict_fig5_umap_61d_v3/niche_perturbation_cell_level_state_editing.state_fixed.csv",
        ]
        cell_files = [p for p in possible if p.exists() and p.stat().st_size > 0]

    cell_scores, cell_audit = load_strict_cell_level_scores(cell_files, assign, final_candidates)

    if cell_scores.empty:
        base_scores, summary_meta = make_strict_summary_track_prior_scores(final_candidates, tracks)
        score_mode = "strict_summary_track_prior"
        component_mapping_status = "strict_cell_level_not_available_or_rejected"
    else:
        base_scores = cell_scores
        summary_meta = {}
        score_mode = "strict_cell_level"
        component_mapping_status = "strict_cell_level_accepted"

    scored = attach_target_relevance(base_scores, final_candidates, marker_genes)
    track_scores, overall = compute_scores(scored, track_weights, weights)

    bg = run_random_background(
        track_scores=track_scores,
        overall=overall,
        candidates=final_candidates,
        marker_genes=marker_genes,
        track_weights=track_weights,
        weights=weights,
        n_random=args.n_random,
        seed=args.seed,
        n_bins=args.n_bins,
    )

    fdr = compute_fdr(overall, bg, final_candidates)
    sanity = sanity_check_fdr(fdr)

    heatmap = make_heatmap(track_scores, fdr, top_n=args.top_n_heatmap)

    fig_outputs = plot_step66b(track_scores, fdr, heatmap, outdir, top_n=args.top_n_fig, dpi=args.dpi)

    # Save outputs.
    final_candidates.to_csv(outdir / "step66b_final_candidate_perturbations_used.csv", index=False)
    excluded_candidates.to_csv(outdir / "step66b_excluded_decoy_housekeeping_candidates_qc.csv", index=False)
    candidate_audit.to_csv(outdir / "step66b_candidate_file_audit.csv", index=False)
    cell_audit.to_csv(outdir / "step66b_cell_level_component_audit.csv", index=False)
    marker_genes.to_csv(outdir / "step66b_track_marker_genes_used.csv", index=False)
    track_weights.to_csv(outdir / "step66b_track_weights_used.csv", index=False)

    pd.DataFrame({"candidate_file": [str(x) for x in candidate_files]}).to_csv(
        outdir / "step66b_candidate_files_used.csv", index=False
    )
    pd.DataFrame({"cell_level_file": [str(x) for x in cell_files]}).to_csv(
        outdir / "step66b_cell_level_files_checked.csv", index=False
    )

    track_scores.to_csv(outdir / "step66b_track_level_perturbation_scores.csv", index=False)
    bg.to_csv(outdir / "step66b_random_background_scores.csv", index=False)
    fdr.to_csv(outdir / "step66b_perturbation_fdr.csv", index=False)
    heatmap.to_csv(outdir / "step66b_track_perturbation_heatmap.csv", index=False)

    report = {
        "status": "ok",
        "analysis_name": "Step66b strict clean track-level perturbation FDR",
        "score_mode": score_mode,
        "component_mapping_status": component_mapping_status,
        "summary_component_meta": summary_meta,
        "sanity_check": sanity,
        "weights": weights,
        "n_tracks": int(track_weights["track_id"].nunique()),
        "n_candidate_files": int(len(candidate_files)),
        "n_final_candidates": int(len(final_candidates)),
        "n_excluded_decoy_housekeeping_candidates": int(len(excluded_candidates)),
        "n_cell_level_files_checked": int(len(cell_files)),
        "n_track_score_rows": int(len(track_scores)),
        "n_random_rows": int(len(bg)),
        "n_random_per_candidate": int(args.n_random),
        "n_fdr_rows": int(len(fdr)),
        "n_FDR_supported_candidates_0_25": int(fdr["is_fdr_significant_0_25"].sum()) if "is_fdr_significant_0_25" in fdr.columns else 0,
        "figure_outputs": fig_outputs,
        "outputs": {
            "final_candidates": str(outdir / "step66b_final_candidate_perturbations_used.csv"),
            "excluded_candidates_qc": str(outdir / "step66b_excluded_decoy_housekeeping_candidates_qc.csv"),
            "candidate_file_audit": str(outdir / "step66b_candidate_file_audit.csv"),
            "cell_level_component_audit": str(outdir / "step66b_cell_level_component_audit.csv"),
            "track_scores": str(outdir / "step66b_track_level_perturbation_scores.csv"),
            "random_background": str(outdir / "step66b_random_background_scores.csv"),
            "perturbation_fdr": str(outdir / "step66b_perturbation_fdr.csv"),
            "heatmap": str(outdir / "step66b_track_perturbation_heatmap.csv"),
            "report_json": str(outdir / "step66b_report.json"),
            "report_txt": str(outdir / "step66b_report.txt"),
        },
        "interpretation_note": (
            "Use this strict 66b output as the corrected Step66 result. "
            "If sanity_status is no_positive_significant_perturbation_detected or "
            "no_FDR_supported_candidate_detected, report the analysis as a conservative "
            "random-background QC layer rather than a positive therapeutic ranking."
        ),
    }

    (outdir / "step66b_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step66b strict clean track-level perturbation FDR report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Top final FDR table:")
    lines.append(fdr.head(80).to_string(index=False))
    lines.append("")
    lines.append("Excluded decoy / housekeeping candidates:")
    lines.append(excluded_candidates.head(80).to_string(index=False) if not excluded_candidates.empty else "None")
    lines.append("")
    lines.append("Cell-level component audit:")
    lines.append(cell_audit.to_string(index=False) if not cell_audit.empty else "No cell-level files checked.")
    lines.append("")
    lines.append("Candidate file audit:")
    lines.append(candidate_audit.to_string(index=False) if not candidate_audit.empty else "No candidate audit.")
    lines.append("")
    lines.append("Top track scores:")
    lines.append(track_scores.sort_values("s_track_score", ascending=False).head(120).to_string(index=False))

    (outdir / "step66b_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step66b")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Top FDR rows:")
    log(fdr.head(30).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
