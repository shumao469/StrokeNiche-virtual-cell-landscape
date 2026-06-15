#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
66_track_level_perturbation_score_fdr.py

Purpose
-------
Step 66: Track-level perturbation score + random background FDR.

This makes Step59/60/61 perturbation analysis more UNAGI-like:
  1. Compute perturbation scores at the StrokeNiche dynamics-track level.
  2. Integrate:
       repair gain
       core probability reduction
       peri/remote shift
       target relevance to track-specific dynamic markers
       safety/off-target penalty
  3. Aggregate track scores into overall perturbation score.
  4. Generate matched random-target backgrounds.
  5. Compute empirical P value and BH-FDR.
  6. Export track-level heatmap and manuscript-ready score tables.

Core formula
------------
For each perturbation p and track t:

  S_track(p,t) =
      w_repair * repair_gain_norm
    + w_core   * core_reduction_norm
    + w_state  * peri_remote_shift_norm
    + w_target * target_marker_relevance
    - w_safety * safety_penalty_norm

Scaled score:
  S_track_scaled in [-1, 1]

Overall:
  S_overall(p) = weighted sum_t S_track_scaled(p,t)

Background
----------
For each perturbation:
  - keep target-set size
  - match perturbation strength bin
  - sample random target sets from Step65 clean dynamic gene universe
  - sample matched perturbation state-shift background from same strength bin
  - recompute S_overall
  - empirical P = (1 + #random >= observed) / (1 + n_random)
  - BH-FDR

Recommended final inputs
------------------------
Track labels:
  Step64c state-celltype refined track labels

Dynamic genes:
  Step65c no_encode_metadata clean gene-level results
  or Step65e housekeeping-filtered supplement genes

Perturbation inputs:
  Step59/60/61/37c outputs are auto-discovered.
  The script prioritizes cell-level perturbation outputs when available.
  If cell-level outputs are not found, it falls back to summary-level scoring.

Outputs
-------
outdir/
  step66_track_level_perturbation_scores.csv
  step66_random_background_scores.csv
  step66_perturbation_fdr.csv
  step66_track_perturbation_heatmap.csv
  step66_candidate_perturbations_used.csv
  step66_track_weights_used.csv
  step66_report.json
  step66_report.txt
  Fig_Step66_TrackPerturbationScoreFDR_annotated.pdf/svg/png
  Fig_Step66_TrackPerturbationScoreFDR_clean_no_text.pdf/svg/png
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

DEFAULT_ROOT = BASE / "results/step8_strokeniche_perturbmap"

DEFAULT_TRACK_DIR = DEFAULT_ROOT / "dynamics_graph_64c_refined_labels_state_celltype"

DEFAULT_STEP65_DIR = DEFAULT_ROOT / "track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata"

DEFAULT_STEP65E_DIR = DEFAULT_ROOT / "track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered"

DEFAULT_OUT = DEFAULT_ROOT / "track_level_perturbation_fdr_66"


# -----------------------------------------------------------------------------
# Basic helpers
# -----------------------------------------------------------------------------

def log(x):
    print(x, flush=True)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def gene_upper(x):
    return str(x).strip().upper()


def split_tokens(x):
    if pd.isna(x):
        return []
    s = str(x)
    s = re.sub(r"[\[\]\(\)\{\}\"']", " ", s)
    parts = re.split(r"[;,|/\s]+", s)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9\-_\.]{1,25}$", p):
            out.append(gene_upper(p))
    return list(dict.fromkeys(out))


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


def minmax(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def robust_z(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    med = s.median()
    mad = (s - med).abs().median()
    if pd.isna(mad) or mad <= 1e-12:
        sd = s.std()
        if pd.isna(sd) or sd <= 1e-12:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - s.mean()) / sd
    return 0.6745 * (s - med) / mad


def contains_any(name, keys):
    s = norm_key(name)
    return any(k in s for k in keys)


def read_csv_maybe(path, required=True, nrows=None):
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
# Track and dynamic marker loading
# -----------------------------------------------------------------------------

def load_tracks(track_dir):
    track_dir = Path(track_dir)

    track_p = track_dir / "step64c_track_label_refined.csv"
    assign_p = track_dir / "step64c_track_assignment_by_cell.refined_labels.csv"

    if not track_p.exists():
        raise FileNotFoundError(track_p)
    if not assign_p.exists():
        raise FileNotFoundError(assign_p)

    tracks = pd.read_csv(track_p, low_memory=False)
    assign = pd.read_csv(assign_p, low_memory=False)

    required = {"track_id"}
    if not required.issubset(tracks.columns):
        raise RuntimeError(f"{track_p} lacks track_id.")
    if not {"obs_name", "track_id"}.issubset(assign.columns):
        raise RuntimeError(f"{assign_p} lacks obs_name/track_id.")

    tracks["track_id"] = tracks["track_id"].astype(str)
    assign["track_id"] = assign["track_id"].astype(str)
    assign["obs_name"] = assign["obs_name"].astype(str)

    return tracks, assign


def load_dynamic_marker_genes(step65_dir, step65e_dir=None, top_n_per_track=50):
    """
    Prefer Step65e filtered marker genes if available.
    Fallback to Step65 dynamic genes.
    """
    step65e_dir = Path(step65e_dir) if step65e_dir else None
    step65_dir = Path(step65_dir)

    marker_sets_p = step65e_dir / "step65e_track_marker_genes_after_filter.csv" if step65e_dir else None

    if marker_sets_p and marker_sets_p.exists():
        ms = pd.read_csv(marker_sets_p, low_memory=False)
        if {"track_id", "marker_genes_after_filter"}.issubset(ms.columns):
            rows = []
            for _, r in ms.iterrows():
                genes = split_tokens(r["marker_genes_after_filter"])
                for rank, g in enumerate(genes[:top_n_per_track], start=1):
                    rows.append({
                        "track_id": str(r["track_id"]),
                        "gene": g,
                        "marker_rank": rank,
                        "marker_source": "step65e_housekeeping_filtered",
                        "dynamic_marker_score": 1.0 / rank,
                    })
            out = pd.DataFrame(rows)
            if not out.empty:
                return out

    dyn_p = step65_dir / "step65_track_dynamic_genes.csv"
    if not dyn_p.exists():
        raise FileNotFoundError(dyn_p)

    dyn = pd.read_csv(dyn_p, low_memory=False)
    if not {"track_id", "feature"}.issubset(dyn.columns):
        raise RuntimeError(f"{dyn_p} lacks track_id/feature.")

    if "passes_min_cells" in dyn.columns:
        dyn = dyn[dyn["passes_min_cells"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    if "dynamic_marker_score" not in dyn.columns:
        dyn["dynamic_marker_score"] = 1.0

    dyn["track_id"] = dyn["track_id"].astype(str)
    dyn["gene"] = dyn["feature"].map(gene_upper)

    rows = []
    for tid, sub in dyn.groupby("track_id"):
        sub = sub.sort_values("dynamic_marker_score", ascending=False).drop_duplicates("gene").head(top_n_per_track)
        for rank, r in enumerate(sub.itertuples(), start=1):
            rows.append({
                "track_id": tid,
                "gene": r.gene,
                "marker_rank": rank,
                "marker_source": "step65_dynamic_genes",
                "dynamic_marker_score": safe_float(getattr(r, "dynamic_marker_score", 1.0)),
            })

    return pd.DataFrame(rows)


def build_track_weights(tracks, marker_genes):
    df = tracks.copy()

    if "track_priority_score" in df.columns:
        w = pd.to_numeric(df["track_priority_score"], errors="coerce").fillna(0)
    elif "n_cells_sum" in df.columns:
        w = pd.to_numeric(df["n_cells_sum"], errors="coerce").fillna(0)
    elif "n_cells" in df.columns:
        w = pd.to_numeric(df["n_cells"], errors="coerce").fillna(0)
    else:
        w = pd.Series(np.ones(len(df)), index=df.index)

    if w.sum() <= 0:
        w = pd.Series(np.ones(len(df)), index=df.index)

    df["track_weight"] = w / w.sum()

    keep = [
        "track_id",
        "track_weight",
        "track_label_refined",
        "biological_axis_refined",
        "label_confidence",
        "state_sequence",
        "celltype_sequence",
    ]
    keep = [c for c in keep if c in df.columns]

    return df[keep].copy()


# -----------------------------------------------------------------------------
# Perturbation candidate discovery
# -----------------------------------------------------------------------------

def is_bad_auto_file(path):
    low = str(path).lower()
    bad = [
        "step66",
        "random_background",
        "track_perturbation_heatmap",
        "step65e_encode_chipseq",
        "step65_track_dynamic_genes",
        "step65_regulator",
        "step65_track_network_edges",
        "step64c_track_assignment",
        "step64c_track_label",
    ]
    return any(x in low for x in bad)


def auto_discover_candidate_files(root):
    root = Path(root)
    patterns = [
        "**/*shortlist*.csv",
        "**/*mini*cmap*.csv",
        "**/*perturb*ranking*.csv",
        "**/*rescue_ranking*.csv",
        "**/*niche_perturbation_ranking*.csv",
        "**/*final*prioritization*.csv",
        "**/*therapeutic*priority*.csv",
        "**/*drug*ranking*.csv",
        "**/*gene*perturbation*.csv",
        "**/*lr_axis*rescue*.csv",
    ]

    files = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p.stat().st_size > 0 and not is_bad_auto_file(p):
                files.append(p)

    # Deduplicate preserving order.
    seen = set()
    out = []
    for p in files:
        s = str(p)
        if s not in seen:
            seen.add(s)
            out.append(p)
    return out


def auto_discover_cell_level_files(root):
    root = Path(root)
    patterns = [
        "**/*cell_level*.csv",
        "**/*perturbed_state_predictions*.csv",
        "**/*niche_perturbation_cell_level*.csv",
        "**/*state_editing*.csv",
    ]

    files = []
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file() and p.stat().st_size > 0 and not is_bad_auto_file(p):
                files.append(p)

    seen = set()
    out = []
    for p in files:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)

        try:
            head = pd.read_csv(p, nrows=5, low_memory=False)
        except Exception:
            continue

        cols = set(map(str, head.columns))
        has_obs = "obs_name" in cols or "cell_id" in cols or "barcode" in cols
        has_pid = any(c in cols for c in [
            "perturbation_id", "perturbation", "target_gene",
            "drug", "compound", "perturbagen", "niche_perturbation"
        ])

        if has_obs and has_pid:
            out.append(p)

    return out


def detect_perturbation_id_col(df):
    candidates = [
        "perturbation_id", "perturbation", "niche_perturbation",
        "perturbagen", "compound", "drug", "drug_name",
        "target_gene", "gene", "ligand_receptor_axis", "lr_axis"
    ]
    for c in candidates:
        if c in df.columns:
            return c

    # Fuzzy.
    for c in df.columns:
        nk = norm_key(c)
        if "perturb" in nk or "compound" in nk or "drug" in nk:
            return c

    return None


def build_perturbation_id(row, id_col=None):
    if id_col and id_col in row.index and str(row[id_col]).strip() and str(row[id_col]) != "nan":
        return str(row[id_col]).strip()

    parts = []
    for c in ["target_gene", "ligand", "receptor", "drug", "compound", "perturbagen"]:
        if c in row.index and not pd.isna(row[c]):
            parts.append(str(row[c]).strip())
    if parts:
        return "_".join(parts)

    return ""


def extract_targets_from_row(row):
    fields = [
        "target_genes", "targets", "target_gene", "target", "gene",
        "ligand", "receptor", "regulator", "overlap_genes",
        "perturbation_id", "perturbation"
    ]

    genes = []
    for c in fields:
        if c in row.index and not pd.isna(row[c]):
            genes.extend(split_tokens(row[c]))

    # Remove non-gene words.
    bad = {
        "BLOCKADE", "UP", "DOWN", "KD", "KO", "OE", "OVEREXPRESSION",
        "INHIBITOR", "AGONIST", "ANTAGONIST", "DRUG", "GENE",
        "NAN", "NONE", "TRUE", "FALSE"
    }
    genes = [g for g in genes if g not in bad and 2 <= len(g) <= 25]
    return sorted(set(genes))


def find_score_col(df):
    priority = [
        "overall_rescue_score",
        "final_therapeutic_priority",
        "therapeutic_priority",
        "niche_perturbation_score",
        "rescue_score",
        "stroke_niche_rescue_score",
        "overall_score",
        "connectivity_score",
        "reversal_score",
        "priority_score",
        "score",
    ]

    for c in priority:
        if c in df.columns:
            return c

    for c in df.columns:
        nk = norm_key(c)
        if ("score" in nk or "rank" in nk) and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.5:
            return c

    return None


def load_candidate_tables(candidate_files):
    rows = []

    for p in candidate_files:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        id_col = detect_perturbation_id_col(df)
        score_col = find_score_col(df)

        # Require at least one sensible id or target column.
        if id_col is None and not any(c in df.columns for c in ["target_gene", "drug", "compound", "ligand", "receptor"]):
            continue

        for _, r in df.iterrows():
            pid = build_perturbation_id(r, id_col=id_col)
            if not pid:
                continue

            targets = extract_targets_from_row(r)
            score = safe_float(r[score_col]) if score_col else np.nan

            row = {
                "perturbation_id": pid,
                "target_genes": ";".join(targets),
                "target_set_size": len(targets),
                "candidate_source_file": str(p),
                "candidate_score_col": score_col or "",
                "candidate_score_raw": score,
            }

            for c in [
                "perturbation_type", "target_gene", "ligand", "receptor",
                "direction", "mechanism_hint", "drug", "compound", "database",
                "priority_class", "safety_flag", "bbb_flag"
            ]:
                if c in df.columns:
                    row[c] = r[c]

            rows.append(row)

    cand = pd.DataFrame(rows)

    if cand.empty:
        return cand

    # Aggregate duplicate perturbations.
    agg_rows = []
    for pid, sub in cand.groupby("perturbation_id"):
        targets = sorted(set(sum([split_tokens(x) for x in sub["target_genes"].fillna("")], [])))
        score_vals = pd.to_numeric(sub["candidate_score_raw"], errors="coerce")
        row = {
            "perturbation_id": pid,
            "target_genes": ";".join(targets),
            "target_set_size": len(targets),
            "candidate_score_raw": score_vals.mean() if score_vals.notna().any() else np.nan,
            "candidate_source_file": ";".join(sorted(set(sub["candidate_source_file"].astype(str)))[:10]),
            "candidate_score_col": ";".join(sorted(set(sub["candidate_score_col"].astype(str)))[:10]),
        }

        for c in [
            "perturbation_type", "target_gene", "ligand", "receptor",
            "direction", "mechanism_hint", "drug", "compound", "database",
            "priority_class", "safety_flag", "bbb_flag"
        ]:
            if c in sub.columns:
                vals = [str(x) for x in sub[c].dropna().unique() if str(x) != "nan"]
                row[c] = ";".join(vals[:5])

        agg_rows.append(row)

    cand = pd.DataFrame(agg_rows)

    # If no target genes were parsed, try perturbation_id itself.
    for i, r in cand.iterrows():
        if safe_float(r["target_set_size"], 0) == 0:
            genes = split_tokens(r["perturbation_id"])
            cand.loc[i, "target_genes"] = ";".join(genes)
            cand.loc[i, "target_set_size"] = len(genes)

    return cand


# -----------------------------------------------------------------------------
# Cell-level perturbation scoring
# -----------------------------------------------------------------------------

def detect_obs_col(df):
    for c in ["obs_name", "cell_id", "barcode", "spot_id"]:
        if c in df.columns:
            return c
    return None


def col_match(df, include, exclude=None):
    exclude = exclude or []
    hits = []
    for c in df.columns:
        nk = norm_key(c)
        if all(k in nk for k in include) and not any(e in nk for e in exclude):
            if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.2:
                hits.append(c)
    return hits


def first_col(df, patterns):
    for include, exclude in patterns:
        hits = col_match(df, include, exclude)
        if hits:
            return hits[0]
    return None


def compute_component_from_cell_table(df):
    """
    Adds:
      repair_gain_raw
      core_reduction_raw
      peri_remote_shift_raw
      safety_penalty_raw

    Flexible column detection.
    """
    out = df.copy()

    # Direct columns.
    repair_direct = first_col(out, [
        (["repair", "gain"], []),
        (["repair", "shift"], []),
        (["delta", "repair"], []),
    ])

    core_direct = first_col(out, [
        (["core", "reduction"], []),
        (["core", "reversal"], []),
        (["delta", "core"], ["probability_raw_bad"]),
    ])

    safety_direct = first_col(out, [
        (["safety", "penalty"], []),
        (["risk", "penalty"], []),
        (["toxicity"], []),
        (["offtarget"], []),
        (["off", "target"], []),
        (["unfavorable"], []),
    ])

    peri_direct = first_col(out, [
        (["delta", "peri"], []),
        (["peri", "shift"], []),
        (["peri", "gain"], []),
    ])

    remote_direct = first_col(out, [
        (["delta", "remote"], []),
        (["remote", "shift"], []),
        (["remote", "gain"], []),
    ])

    # Pairwise before/after detection.
    repair_before = first_col(out, [
        (["baseline", "repair"], []),
        (["before", "repair"], []),
        (["pred", "repair"], ["perturbed"]),
    ])
    repair_after = first_col(out, [
        (["perturbed", "repair"], []),
        (["after", "repair"], []),
        (["edited", "repair"], []),
    ])

    core_before = first_col(out, [
        (["baseline", "core"], []),
        (["before", "core"], []),
        (["pred", "core"], ["perturbed"]),
    ])
    core_after = first_col(out, [
        (["perturbed", "core"], []),
        (["after", "core"], []),
        (["edited", "core"], []),
    ])

    peri_before = first_col(out, [
        (["baseline", "peri"], []),
        (["before", "peri"], []),
        (["pred", "peri"], ["perturbed"]),
    ])
    peri_after = first_col(out, [
        (["perturbed", "peri"], []),
        (["after", "peri"], []),
        (["edited", "peri"], []),
    ])

    remote_before = first_col(out, [
        (["baseline", "remote"], []),
        (["before", "remote"], []),
        (["pred", "remote"], ["perturbed"]),
    ])
    remote_after = first_col(out, [
        (["perturbed", "remote"], []),
        (["after", "remote"], []),
        (["edited", "remote"], []),
    ])

    if repair_direct:
        out["repair_gain_raw"] = pd.to_numeric(out[repair_direct], errors="coerce")
        repair_source = repair_direct
    elif repair_before and repair_after:
        out["repair_gain_raw"] = pd.to_numeric(out[repair_after], errors="coerce") - pd.to_numeric(out[repair_before], errors="coerce")
        repair_source = f"{repair_after}-{repair_before}"
    else:
        out["repair_gain_raw"] = np.nan
        repair_source = ""

    if core_direct:
        val = pd.to_numeric(out[core_direct], errors="coerce")
        # If direct column is delta_core_probability, lower core is good; convert if likely delta.
        if "delta" in norm_key(core_direct) and "reduction" not in norm_key(core_direct) and "reversal" not in norm_key(core_direct):
            val = -val
        out["core_reduction_raw"] = val
        core_source = core_direct
    elif core_before and core_after:
        out["core_reduction_raw"] = pd.to_numeric(out[core_before], errors="coerce") - pd.to_numeric(out[core_after], errors="coerce")
        core_source = f"{core_before}-{core_after}"
    else:
        out["core_reduction_raw"] = np.nan
        core_source = ""

    parts = []
    src_parts = []

    if peri_direct:
        parts.append(pd.to_numeric(out[peri_direct], errors="coerce"))
        src_parts.append(peri_direct)
    elif peri_before and peri_after:
        parts.append(pd.to_numeric(out[peri_after], errors="coerce") - pd.to_numeric(out[peri_before], errors="coerce"))
        src_parts.append(f"{peri_after}-{peri_before}")

    if remote_direct:
        parts.append(pd.to_numeric(out[remote_direct], errors="coerce"))
        src_parts.append(remote_direct)
    elif remote_before and remote_after:
        parts.append(pd.to_numeric(out[remote_after], errors="coerce") - pd.to_numeric(out[remote_before], errors="coerce"))
        src_parts.append(f"{remote_after}-{remote_before}")

    if parts:
        out["peri_remote_shift_raw"] = sum(parts)
        state_source = "+".join(src_parts)
    else:
        out["peri_remote_shift_raw"] = np.nan
        state_source = ""

    if safety_direct:
        out["safety_penalty_raw"] = pd.to_numeric(out[safety_direct], errors="coerce")
        safety_source = safety_direct
    else:
        out["safety_penalty_raw"] = 0.0
        safety_source = "zero_fallback"

    meta = {
        "repair_source": repair_source,
        "core_source": core_source,
        "peri_remote_source": state_source,
        "safety_source": safety_source,
    }

    return out, meta


def load_cell_level_scores(cell_files, track_assign):
    all_rows = []
    meta_rows = []

    for p in cell_files:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue

        if df.empty:
            continue

        obs_col = detect_obs_col(df)
        pid_col = detect_perturbation_id_col(df)

        if obs_col is None or pid_col is None:
            continue

        df = df.copy()
        df["obs_name"] = df[obs_col].astype(str)
        df["perturbation_id"] = df[pid_col].astype(str)

        df, meta = compute_component_from_cell_table(df)

        # Require at least one useful component.
        useful = df[["repair_gain_raw", "core_reduction_raw", "peri_remote_shift_raw", "safety_penalty_raw"]].notna().any(axis=1).sum()
        if useful == 0:
            continue

        small = df[[
            "obs_name",
            "perturbation_id",
            "repair_gain_raw",
            "core_reduction_raw",
            "peri_remote_shift_raw",
            "safety_penalty_raw",
        ]].copy()

        merged = small.merge(track_assign[["obs_name", "track_id"]], on="obs_name", how="inner")
        if merged.empty:
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
        grouped["score_source"] = "cell_level"
        grouped["cell_level_source_file"] = str(p)

        all_rows.append(grouped)

        meta_rows.append({
            "file": str(p),
            "obs_col": obs_col,
            "perturbation_id_col": pid_col,
            "n_rows": int(len(df)),
            "n_merged_rows": int(len(merged)),
            **meta,
        })

    if not all_rows:
        return pd.DataFrame(), pd.DataFrame(meta_rows)

    out = pd.concat(all_rows, ignore_index=True)

    # Aggregate if multiple cell-level files cover same perturbation/track.
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

    return out, pd.DataFrame(meta_rows)


# -----------------------------------------------------------------------------
# Summary fallback scoring
# -----------------------------------------------------------------------------

def make_summary_fallback_scores(candidates, tracks):
    rows = []

    if candidates.empty:
        return pd.DataFrame()

    # Use candidate_score_raw if available; otherwise rank all equally.
    scores = pd.to_numeric(candidates["candidate_score_raw"], errors="coerce")
    if scores.notna().any():
        score_norm = minmax(scores.fillna(scores.median()))
    else:
        score_norm = pd.Series(np.full(len(candidates), 0.5), index=candidates.index)

    axis_boost = {}
    for _, tr in tracks.iterrows():
        tid = str(tr["track_id"])
        axis = str(tr.get("biological_axis_refined", "")).lower()
        label = str(tr.get("track_label_refined", "")).lower()

        # Track-axis prior; weak, only for fallback.
        axis_boost[tid] = {
            "repair": 1.0 if "repair" in axis or "repair" in label else 0.7,
            "core": 1.0 if "core" in axis or "core" in label else 0.7,
            "state": 1.0 if "transition" in axis or "remote" in axis or "peri" in label else 0.7,
            "safety": 0.2,
        }

    for i, cand in candidates.iterrows():
        pid = cand["perturbation_id"]
        s = safe_float(score_norm.loc[i], 0.5)

        for _, tr in tracks.iterrows():
            tid = str(tr["track_id"])
            b = axis_boost.get(tid, {"repair": 0.7, "core": 0.7, "state": 0.7, "safety": 0.2})

            rows.append({
                "perturbation_id": pid,
                "track_id": tid,
                "n_cells": np.nan,
                "repair_gain_raw": s * b["repair"],
                "core_reduction_raw": s * b["core"],
                "peri_remote_shift_raw": s * b["state"],
                "safety_penalty_raw": b["safety"],
                "score_source": "summary_fallback",
                "cell_level_source_file": "",
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Target relevance
# -----------------------------------------------------------------------------

def build_marker_dict(marker_genes):
    d = {}
    for tid, sub in marker_genes.groupby("track_id"):
        genes = list(dict.fromkeys(sub.sort_values("marker_rank")["gene"].astype(str).map(gene_upper)))
        d[str(tid)] = genes
    return d


def target_relevance(targets, markers):
    t = set(gene_upper(x) for x in targets if str(x).strip())
    m = set(gene_upper(x) for x in markers if str(x).strip())

    if len(t) == 0 or len(m) == 0:
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

    # Bounded in [0,1]-ish, emphasizes perturbation target coverage.
    rel = 0.55 * tf + 0.30 * mf + 0.15 * j

    return {
        "target_overlap_n": len(ov),
        "target_jaccard": j,
        "target_marker_fraction": mf,
        "target_set_fraction": tf,
        "target_relevance_raw": rel,
        "target_overlap_genes": ";".join(sorted(ov)),
    }


def attach_target_relevance(track_scores, candidates, marker_genes):
    cand_map = candidates.set_index("perturbation_id")["target_genes"].to_dict()
    marker_map = build_marker_dict(marker_genes)

    rows = []
    for _, r in track_scores.iterrows():
        pid = r["perturbation_id"]
        tid = r["track_id"]

        targets = split_tokens(cand_map.get(pid, ""))
        markers = marker_map.get(str(tid), [])

        rel = target_relevance(targets, markers)
        row = r.to_dict()
        row.update(rel)
        row["target_genes"] = ";".join(targets)
        row["target_set_size"] = len(targets)
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Score calculation
# -----------------------------------------------------------------------------

def compute_track_scores(df, track_weights, weights):
    out = df.copy()

    for c in ["repair_gain_raw", "core_reduction_raw", "peri_remote_shift_raw", "safety_penalty_raw"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["repair_gain_norm"] = minmax(out["repair_gain_raw"].fillna(out["repair_gain_raw"].median())).values
    out["core_reduction_norm"] = minmax(out["core_reduction_raw"].fillna(out["core_reduction_raw"].median())).values
    out["peri_remote_shift_norm"] = minmax(out["peri_remote_shift_raw"].fillna(out["peri_remote_shift_raw"].median())).values
    out["safety_penalty_norm"] = minmax(out["safety_penalty_raw"].fillna(0)).values

    # target_relevance_raw is already bounded; keep it as-is.
    out["target_relevance_norm"] = pd.to_numeric(out.get("target_relevance_raw", 0), errors="coerce").fillna(0).clip(0, 1)

    out["state_score_no_target"] = (
        weights["repair"] * out["repair_gain_norm"]
        + weights["core"] * out["core_reduction_norm"]
        + weights["state"] * out["peri_remote_shift_norm"]
        - weights["safety"] * out["safety_penalty_norm"]
    )

    out["s_track_raw"] = (
        out["state_score_no_target"]
        + weights["target"] * out["target_relevance_norm"]
    )

    # Scale to [-1,1] globally.
    mm = minmax(out["s_track_raw"])
    out["s_track_scaled"] = 2 * mm - 1

    # Attach track weights.
    tw = track_weights[["track_id", "track_weight"]].copy()
    out = out.merge(tw, on="track_id", how="left")
    out["track_weight"] = out["track_weight"].fillna(1.0 / max(out["track_id"].nunique(), 1))

    # Overall score.
    overall = (
        out.groupby("perturbation_id")
        .apply(lambda x: float(np.sum(x["s_track_scaled"] * x["track_weight"])))
        .reset_index(name="s_overall_observed")
    )

    # Strength for matched background.
    strength = (
        out.groupby("perturbation_id")
        .agg(
            perturbation_strength=("state_score_no_target", lambda x: float(np.nanmean(np.abs(x)))),
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

def add_bins(overall, candidates, n_bins=5):
    df = overall.merge(
        candidates[["perturbation_id", "target_set_size", "target_genes"]],
        on="perturbation_id",
        how="left",
    )

    df["target_set_size"] = pd.to_numeric(df["target_set_size"], errors="coerce").fillna(0).astype(int)
    df["perturbation_strength"] = pd.to_numeric(df["perturbation_strength"], errors="coerce").fillna(0)

    def qbin(s, nb):
        s = pd.to_numeric(s, errors="coerce").fillna(0)
        if s.nunique() <= 1:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)
        try:
            return pd.qcut(s.rank(method="first"), q=min(nb, s.nunique()), labels=False, duplicates="drop").astype(int)
        except Exception:
            return pd.Series(np.zeros(len(s), dtype=int), index=s.index)

    df["target_size_bin"] = qbin(df["target_set_size"], n_bins)
    df["strength_bin"] = qbin(df["perturbation_strength"], n_bins)

    return df


def random_target_relevance(random_genes, marker_map, track_ids):
    rels = {}
    for tid in track_ids:
        rels[tid] = target_relevance(random_genes, marker_map.get(str(tid), []))["target_relevance_raw"]
    return rels


def run_random_background(
    track_scores,
    overall,
    candidates,
    marker_genes,
    track_weights,
    weights,
    n_random=1000,
    seed=1,
    n_bins=5,
):
    rng = np.random.default_rng(seed)

    matched = add_bins(overall, candidates, n_bins=n_bins)

    marker_map = build_marker_dict(marker_genes)
    track_ids = list(track_weights["track_id"].astype(str))
    track_w = track_weights.set_index("track_id")["track_weight"].to_dict()

    # Gene universe from dynamic markers plus candidate targets.
    universe = set(marker_genes["gene"].astype(str).map(gene_upper))
    for tg in candidates["target_genes"].fillna(""):
        universe.update(split_tokens(tg))
    universe = sorted(g for g in universe if g and g != "NAN")

    if len(universe) == 0:
        raise RuntimeError("Random background gene universe is empty.")

    # Precompute state score table.
    state = track_scores[[
        "perturbation_id",
        "track_id",
        "state_score_no_target",
    ]].copy()

    state_map = {
        pid: sub.set_index("track_id")["state_score_no_target"].to_dict()
        for pid, sub in state.groupby("perturbation_id")
    }

    rows = []

    for _, q in matched.iterrows():
        pid = q["perturbation_id"]
        k = int(q["target_set_size"])
        k = max(1, min(k, len(universe)))

        same = matched[
            (matched["target_size_bin"] == q["target_size_bin"])
            & (matched["strength_bin"] == q["strength_bin"])
        ].copy()

        if len(same) < 3:
            same = matched[matched["strength_bin"] == q["strength_bin"]].copy()
        if len(same) < 3:
            same = matched.copy()

        # Prefer excluding itself, but allow fallback.
        same_no_self = same[same["perturbation_id"] != pid]
        if len(same_no_self) >= 1:
            same = same_no_self

        same_pids = same["perturbation_id"].astype(str).tolist()

        for b in range(n_random):
            bg_pid = rng.choice(same_pids)
            rand_genes = rng.choice(universe, size=k, replace=False).tolist()

            target_rel = random_target_relevance(rand_genes, marker_map, track_ids)
            bg_state = state_map.get(bg_pid, {})

            overall_score = 0.0
            for tid in track_ids:
                base = safe_float(bg_state.get(tid, 0.0), 0.0)
                rel = safe_float(target_rel.get(tid, 0.0), 0.0)
                s = base + weights["target"] * rel

                overall_score += s * safe_float(track_w.get(tid, 0.0), 0.0)

            rows.append({
                "perturbation_id": pid,
                "random_iter": b + 1,
                "matched_background_perturbation_id": bg_pid,
                "random_target_set_size": k,
                "random_target_genes": ";".join(rand_genes),
                "random_s_overall": overall_score,
                "target_size_bin": int(q["target_size_bin"]),
                "strength_bin": int(q["strength_bin"]),
            })

    bg = pd.DataFrame(rows)

    return bg


def compute_empirical_fdr(overall, bg, candidates):
    rows = []

    for _, r in overall.iterrows():
        pid = r["perturbation_id"]
        obs = safe_float(r["s_overall_observed"])
        sub = bg[bg["perturbation_id"] == pid]

        if sub.empty or pd.isna(obs):
            p = np.nan
            z = np.nan
            bg_mean = np.nan
            bg_sd = np.nan
        else:
            rand = pd.to_numeric(sub["random_s_overall"], errors="coerce").dropna().values
            p = (1.0 + np.sum(rand >= obs)) / (1.0 + len(rand))
            bg_mean = float(np.mean(rand))
            bg_sd = float(np.std(rand))
            z = (obs - bg_mean) / bg_sd if bg_sd > 1e-12 else np.nan

        rows.append({
            "perturbation_id": pid,
            "s_overall_observed": obs,
            "random_mean": bg_mean,
            "random_sd": bg_sd,
            "random_z": z,
            "empirical_p_value": p,
            "n_random": int(len(sub)),
        })

    fdr = pd.DataFrame(rows)
    fdr["bh_fdr"] = bh_fdr(fdr["empirical_p_value"].values)

    fdr = fdr.merge(
        candidates,
        on="perturbation_id",
        how="left",
    )

    fdr = fdr.sort_values(
        ["bh_fdr", "empirical_p_value", "s_overall_observed"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    fdr["step66_rank"] = np.arange(1, len(fdr) + 1)

    return fdr


# -----------------------------------------------------------------------------
# Heatmap and plotting
# -----------------------------------------------------------------------------

def make_heatmap_table(track_scores, fdr, top_n=30):
    top_pids = fdr.sort_values(["bh_fdr", "s_overall_observed"], ascending=[True, False])["perturbation_id"].head(top_n).tolist()

    sub = track_scores[track_scores["perturbation_id"].isin(top_pids)].copy()

    heat = sub.pivot_table(
        index="track_id",
        columns="perturbation_id",
        values="s_track_scaled",
        aggfunc="mean",
    )

    heat = heat.reindex(columns=top_pids)

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


def plot_step66(track_scores, fdr, heatmap, outdir, top_n=25, dpi=600):
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step66_TrackPerturbationScoreFDR_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.34)

        # A: observed vs random z / FDR.
        axA = fig.add_subplot(gs[0, 0])
        top = fdr.sort_values(["bh_fdr", "s_overall_observed"], ascending=[True, False]).head(top_n).copy()
        y = np.arange(len(top))[::-1]
        axA.barh(y, top["s_overall_observed"].values[::-1])

        if annotate:
            labels = top["perturbation_id"].astype(str).values[::-1]
            axA.set_yticks(y)
            axA.set_yticklabels(labels, fontsize=7)
            axA.set_xlabel("Observed S_overall")
            axA.set_title("A | Perturbation priority score", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axA)

        # B: empirical FDR scatter.
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
            axB.set_title("B | Empirical random-background FDR", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axB)

        # C: heatmap.
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
                axC.set_title("C | Track-level perturbation score heatmap", loc="left", fontsize=12, fontweight="bold")
                cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
            else:
                strip_text(axC)
        else:
            if annotate:
                axC.text(0.5, 0.5, "No heatmap", ha="center", va="center")
            else:
                strip_text(axC)

        # D: component summary for top candidates.
        axD = fig.add_subplot(gs[1, 1])
        top_pids = top["perturbation_id"].astype(str).tolist()
        comp = track_scores[track_scores["perturbation_id"].isin(top_pids)].groupby("perturbation_id").agg(
            repair=("repair_gain_norm", "mean"),
            core=("core_reduction_norm", "mean"),
            state=("peri_remote_shift_norm", "mean"),
            target=("target_relevance_norm", "mean"),
            safety=("safety_penalty_norm", "mean"),
        ).reindex(top_pids)

        if not comp.empty:
            bottom = np.zeros(len(comp))
            xs = np.arange(len(comp))
            for col in ["repair", "core", "state", "target"]:
                vals = comp[col].fillna(0).values
                axD.bar(xs, vals, bottom=bottom, label=col)
                bottom += vals
            vals = -comp["safety"].fillna(0).values
            axD.bar(xs, vals, label="safety penalty")
            if annotate:
                axD.set_xticks(xs)
                axD.set_xticklabels([x[:18] for x in comp.index], rotation=45, ha="right", fontsize=7)
                axD.set_ylabel("Mean normalized component")
                axD.set_title("D | Score components", loc="left", fontsize=12, fontweight="bold")
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
                for spine in ax.spines.values():
                    spine.set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if annotate:
            fig.suptitle("Step 66 | Track-level perturbation score with matched random-background FDR", fontsize=15, fontweight="bold")

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

    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--track_dir", default=str(DEFAULT_TRACK_DIR))
    ap.add_argument("--step65_dir", default=str(DEFAULT_STEP65_DIR))
    ap.add_argument("--step65e_dir", default=str(DEFAULT_STEP65E_DIR))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--candidate_files", default="", help="Comma-separated ranking/shortlist files. If empty, auto-discover.")
    ap.add_argument("--cell_level_files", default="", help="Comma-separated cell-level perturbation files. If empty, auto-discover.")

    ap.add_argument("--top_marker_genes_per_track", type=int, default=50)
    ap.add_argument("--n_random", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1)
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
    log("Step 66: Track-level perturbation score + random background FDR")
    log("=" * 100)
    log(f"root={root}")
    log(f"track_dir={args.track_dir}")
    log(f"step65_dir={args.step65_dir}")
    log(f"step65e_dir={args.step65e_dir}")
    log(f"outdir={outdir}")
    log(f"weights={weights}")

    tracks, assign = load_tracks(args.track_dir)
    marker_genes = load_dynamic_marker_genes(
        args.step65_dir,
        step65e_dir=args.step65e_dir,
        top_n_per_track=args.top_marker_genes_per_track,
    )
    track_weights = build_track_weights(tracks, marker_genes)

    if args.candidate_files.strip():
        candidate_files = [Path(x.strip()) for x in args.candidate_files.split(",") if x.strip()]
    else:
        candidate_files = auto_discover_candidate_files(root)

    candidates = load_candidate_tables(candidate_files)

    if candidates.empty:
        raise RuntimeError("No perturbation candidate table found. Provide --candidate_files.")

    if args.cell_level_files.strip():
        cell_files = [Path(x.strip()) for x in args.cell_level_files.split(",") if x.strip()]
    else:
        cell_files = auto_discover_cell_level_files(root)

    cell_scores, cell_meta = load_cell_level_scores(cell_files, assign)

    if cell_scores.empty:
        log("[WARN] No valid cell-level perturbation scores found. Using summary-level fallback.")
        base_scores = make_summary_fallback_scores(candidates, tracks)
        score_mode = "summary_fallback"
    else:
        base_scores = cell_scores
        score_mode = "cell_level"

    # Keep perturbations present in candidates; if cell-level has extra, append candidates.
    existing_pids = set(candidates["perturbation_id"].astype(str))
    base_scores = base_scores[base_scores["perturbation_id"].astype(str).isin(existing_pids)].copy()

    if base_scores.empty:
        log("[WARN] Cell-level perturbation ids did not match candidates. Falling back to summary-level scores.")
        base_scores = make_summary_fallback_scores(candidates, tracks)
        score_mode = "summary_fallback_due_to_id_mismatch"

    scored = attach_target_relevance(base_scores, candidates, marker_genes)
    track_scores, overall = compute_track_scores(scored, track_weights, weights)

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

    fdr = compute_empirical_fdr(overall, bg, candidates)

    heatmap = make_heatmap_table(track_scores, fdr, top_n=args.top_n_heatmap)

    # Save outputs.
    candidates.to_csv(outdir / "step66_candidate_perturbations_used.csv", index=False)
    track_weights.to_csv(outdir / "step66_track_weights_used.csv", index=False)
    marker_genes.to_csv(outdir / "step66_track_marker_genes_used.csv", index=False)

    pd.DataFrame({"candidate_file": [str(x) for x in candidate_files]}).to_csv(
        outdir / "step66_candidate_files_used.csv", index=False
    )
    pd.DataFrame({"cell_level_file": [str(x) for x in cell_files]}).to_csv(
        outdir / "step66_cell_level_files_scanned.csv", index=False
    )
    cell_meta.to_csv(outdir / "step66_cell_level_file_audit.csv", index=False)

    track_scores.to_csv(outdir / "step66_track_level_perturbation_scores.csv", index=False)
    bg.to_csv(outdir / "step66_random_background_scores.csv", index=False)
    fdr.to_csv(outdir / "step66_perturbation_fdr.csv", index=False)
    heatmap.to_csv(outdir / "step66_track_perturbation_heatmap.csv", index=False)

    fig_outputs = plot_step66(
        track_scores=track_scores,
        fdr=fdr,
        heatmap=heatmap,
        outdir=outdir,
        top_n=args.top_n_fig,
        dpi=args.dpi,
    )

    report = {
        "status": "ok",
        "analysis_name": "Step66 track-level perturbation score + matched random-background FDR",
        "score_mode": score_mode,
        "root": str(root),
        "track_dir": str(args.track_dir),
        "step65_dir": str(args.step65_dir),
        "step65e_dir": str(args.step65e_dir),
        "outdir": str(outdir),
        "weights": weights,
        "n_tracks": int(track_weights["track_id"].nunique()),
        "n_candidate_files": int(len(candidate_files)),
        "n_cell_level_files_scanned": int(len(cell_files)),
        "n_candidate_perturbations": int(candidates["perturbation_id"].nunique()),
        "n_track_score_rows": int(len(track_scores)),
        "n_random_rows": int(len(bg)),
        "n_random_per_perturbation": int(args.n_random),
        "n_perturbation_fdr_rows": int(len(fdr)),
        "figure_outputs": fig_outputs,
        "outputs": {
            "track_scores": str(outdir / "step66_track_level_perturbation_scores.csv"),
            "random_background": str(outdir / "step66_random_background_scores.csv"),
            "perturbation_fdr": str(outdir / "step66_perturbation_fdr.csv"),
            "heatmap": str(outdir / "step66_track_perturbation_heatmap.csv"),
            "candidate_perturbations": str(outdir / "step66_candidate_perturbations_used.csv"),
            "track_weights": str(outdir / "step66_track_weights_used.csv"),
            "marker_genes": str(outdir / "step66_track_marker_genes_used.csv"),
            "report_json": str(outdir / "step66_report.json"),
            "report_txt": str(outdir / "step66_report.txt"),
        },
        "interpretation_note": (
            "Step66 provides a UNAGI-like statistical prioritization layer. "
            "It should be interpreted as computational perturbation prioritization, "
            "not direct therapeutic recommendation. If score_mode is summary_fallback, "
            "track-specific effects are approximate and should be presented more conservatively."
        ),
    }

    (outdir / "step66_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step66 track-level perturbation score + random background FDR report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Top perturbations by FDR:")
    lines.append(fdr.head(80).to_string(index=False))
    lines.append("")
    lines.append("Top track-level perturbation scores:")
    lines.append(track_scores.sort_values("s_track_scaled", ascending=False).head(120).to_string(index=False))
    lines.append("")
    lines.append("Cell-level file audit:")
    lines.append(cell_meta.to_string(index=False) if not cell_meta.empty else "No valid cell-level files.")
    lines.append("")
    lines.append("Candidate files used:")
    lines.extend([str(x) for x in candidate_files[:100]])

    (outdir / "step66_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step66")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Top perturbations:")
    log(fdr.head(30).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
