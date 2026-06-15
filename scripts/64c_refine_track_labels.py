#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
64c_refine_track_labels.py

Purpose
-------
Refine Step 64b StrokeNiche dynamics graph track labels.

Why
---
Step64/64b generated pseudo-progression tracks, but track_label was heuristic.
Some tracks may be over-interpreted, e.g. labelled as "microglia-inflammatory"
even when the celltype sequence is OLs->OLs->OLs.

This script generates reviewer-ready labels using:
  1. state transition sequence
  2. celltype enrichment / context
  3. module delta / module mean evidence
  4. repair-core shift direction
  5. conservative confidence score

Inputs
------
Default input directory:
  results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_kmeans

Expected files:
  step64b_track_summary.csv
  step64b_track_module_matrix.csv
  step64b_track_assignment_by_cell.csv
  step64b_dynamics_graph_nodes.csv
  step64b_dynamics_graph_edges.csv

Also supports original Step64 names:
  step64_track_summary.csv
  step64_track_module_matrix.csv
  step64_track_assignment_by_cell.csv
  step64_dynamics_graph_nodes.csv
  step64_dynamics_graph_edges.csv

Outputs
-------
outdir/
  step64c_track_label_refined.csv
  step64c_track_label_evidence_long.csv
  step64c_track_axis_score_matrix.csv
  step64c_track_module_evidence_matrix.csv
  step64c_track_assignment_by_cell.refined_labels.csv
  step64c_dynamics_graph_nodes.refined_labels.csv
  step64c_report.json
  step64c_report.txt
  Fig_StrokeNiche_TrackLabelRefinement_annotated.pdf/svg/png
  Fig_StrokeNiche_TrackLabelRefinement_clean_no_text.pdf/svg/png
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
DEFAULT_IN64 = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_kmeans"
DEFAULT_OUT = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64c_refined_labels_kmeans"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(msg):
    print(msg, flush=True)


def read_csv(path, required=True):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None
    return pd.read_csv(p, low_memory=False)


def find_existing(indir, names, required=True):
    indir = Path(indir)
    for name in names:
        p = indir / name
        if p.exists() and p.stat().st_size > 0:
            return p
    if required:
        raise FileNotFoundError(f"None of these files found in {indir}: {names}")
    return None


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def minmax_series(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def zscore_series(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    mu = s.mean()
    sd = s.std()
    if pd.isna(sd) or sd <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd


def sigmoid(x):
    x = safe_float(x, 0.0)
    return 1.0 / (1.0 + np.exp(-x))


def split_seq(x):
    if pd.isna(x):
        return []
    return [s.strip() for s in str(x).split("->") if s.strip()]


def clean_token(x):
    return str(x).strip().lower().replace(" ", "_").replace("-", "_")


def contains_any(text, terms):
    s = str(text).lower()
    return any(t.lower() in s for t in terms)


def normalized_state(x):
    s = clean_token(x)
    if "lesion_core" in s or s == "core" or "core_like" in s:
        return "lesion-core-like"
    if "peri" in s or "penumbra" in s:
        return "peri-infarct"
    if "remote" in s:
        return "remote-like"
    return str(x)


def axis_color_map():
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
        "uncertain": "#B0B0B0",
    }


# -----------------------------------------------------------------------------
# Load Step64b files
# -----------------------------------------------------------------------------

def load_inputs(in64):
    in64 = Path(in64)

    track_p = find_existing(in64, ["step64b_track_summary.csv", "step64_track_summary.csv"])
    nodes_p = find_existing(in64, ["step64b_dynamics_graph_nodes.csv", "step64_dynamics_graph_nodes.csv"])
    edges_p = find_existing(in64, ["step64b_dynamics_graph_edges.csv", "step64_dynamics_graph_edges.csv"], required=False)
    assign_p = find_existing(in64, ["step64b_track_assignment_by_cell.csv", "step64_track_assignment_by_cell.csv"], required=False)
    module_p = find_existing(in64, ["step64b_track_module_matrix.csv", "step64_track_module_matrix.csv"], required=False)

    tracks = read_csv(track_p, required=True)
    nodes = read_csv(nodes_p, required=True)
    edges = read_csv(edges_p, required=False) if edges_p else pd.DataFrame()
    assignments = read_csv(assign_p, required=False) if assign_p else pd.DataFrame()
    module_matrix = read_csv(module_p, required=False) if module_p else pd.DataFrame()

    meta = {
        "track_summary": str(track_p),
        "nodes": str(nodes_p),
        "edges": str(edges_p) if edges_p else "",
        "assignments": str(assign_p) if assign_p else "",
        "module_matrix": str(module_p) if module_p else "",
    }

    return tracks, nodes, edges, assignments, module_matrix, meta


# -----------------------------------------------------------------------------
# Module evidence
# -----------------------------------------------------------------------------

MODULE_AXIS_PATTERNS = {
    "vascular_barrier": {
        "positive": [
            "module_BBB_leakage",
            "module_endothelial_barrier_fragility",
            "module_vascular",
            "module_endothelial",
        ],
        "negative": [
            "module_barrier_stability",
        ],
        "label_with_celltype": "vascular/endothelial-barrier transition track",
        "label_without_celltype": "barrier-remodeling latent track",
    },
    "repair_remodeling": {
        "positive": [
            "module_repair_ECM",
            "module_synaptic_recovery",
            "module_barrier_stability",
            "module_repair",
            "module_ECM",
        ],
        "negative": [],
        "label": "repair-permissive remodeling track",
    },
    "inflammatory": {
        "positive": [
            "module_inflammation",
            "module_microglia_inflammatory",
            "module_chemotaxis",
            "module_TNF",
            "module_CCL2",
        ],
        "negative": [],
        "label_with_celltype": "microglia-inflammatory track",
        "label_without_celltype": "inflammation-enriched latent track",
    },
    "hypoxia_ferroptosis": {
        "positive": [
            "module_hypoxia",
            "module_ferroptosis",
            "module_redox",
            "module_oxidative",
        ],
        "negative": [],
        "label": "hypoxia/ferroptosis injury track",
    },
    "astrocyte_reactive": {
        "positive": [
            "module_astrocyte_reactive",
            "module_reactive_astrocyte",
        ],
        "negative": [],
        "label": "astrocyte-reactive remodeling track",
    },
}


def extract_module_evidence_from_matrix(module_matrix, tracks, nodes):
    """
    Return module evidence matrix with one row per track_id.
    Columns:
      module_*_delta, module_*_mean
    """
    if module_matrix is not None and not module_matrix.empty and "track_id" in module_matrix.columns:
        mm = module_matrix.copy()

        keep = ["track_id"]
        for c in mm.columns:
            if c == "track_id":
                continue
            if c.startswith("module_") and (
                c.endswith("_delta_end_minus_start")
                or c.endswith("_mean_over_track")
            ):
                keep.append(c)

        if len(keep) > 1:
            out = mm[keep].copy()
            # Rename to shorter module names.
            rename = {}
            for c in out.columns:
                if c == "track_id":
                    continue
                if c.endswith("_delta_end_minus_start"):
                    rename[c] = c.replace("_delta_end_minus_start", "_delta")
                elif c.endswith("_mean_over_track"):
                    rename[c] = c.replace("_mean_over_track", "_mean")
            out = out.rename(columns=rename)
            return out

    # Fallback: reconstruct module deltas from nodes.
    if nodes is None or nodes.empty:
        return pd.DataFrame({"track_id": tracks["track_id"]})

    module_cols = [c for c in nodes.columns if c.startswith("module_") and c.endswith("_mean")]
    rows = []

    for _, tr in tracks.iterrows():
        node_ids = split_seq(tr.get("track_nodes", ""))
        sub = nodes[nodes["node_id"].isin(node_ids)].copy()
        if "day" in sub.columns:
            sub = sub.sort_values("day")

        row = {"track_id": tr["track_id"]}

        for c in module_cols:
            base = c.replace("_mean", "")
            vals = pd.to_numeric(sub[c], errors="coerce").values
            if len(vals) >= 2:
                row[f"{base}_delta"] = safe_float(vals[-1] - vals[0])
                row[f"{base}_mean"] = safe_float(np.nanmean(vals))
            elif len(vals) == 1:
                row[f"{base}_delta"] = np.nan
                row[f"{base}_mean"] = safe_float(vals[0])
            else:
                row[f"{base}_delta"] = np.nan
                row[f"{base}_mean"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def find_matching_module_cols(module_evidence, patterns, suffix="_delta"):
    cols = []
    norm_lookup = {norm_col(c): c for c in module_evidence.columns}

    for p in patterns:
        pn = norm_col(p + suffix)
        if pn in norm_lookup:
            cols.append(norm_lookup[pn])
            continue

        # Fuzzy contains.
        pcore = norm_col(p).replace("module_", "")
        for c in module_evidence.columns:
            if not c.endswith(suffix):
                continue
            cn = norm_col(c)
            if pcore and pcore in cn:
                cols.append(c)

    return list(dict.fromkeys(cols))


def build_module_axis_scores(module_evidence):
    """
    Produce axis module score columns and top-module evidence.
    """
    if module_evidence is None or module_evidence.empty:
        return pd.DataFrame(), pd.DataFrame()

    out = module_evidence[["track_id"]].copy()
    evidence_rows = []

    # Add z-scores for all delta columns.
    delta_cols = [c for c in module_evidence.columns if c.startswith("module_") and c.endswith("_delta")]
    mean_cols = [c for c in module_evidence.columns if c.startswith("module_") and c.endswith("_mean")]

    zdf = pd.DataFrame({"track_id": module_evidence["track_id"]})
    for c in delta_cols:
        zdf[f"{c}_z"] = zscore_series(module_evidence[c]).values
    for c in mean_cols:
        zdf[f"{c}_z"] = zscore_series(module_evidence[c]).values

    combined = module_evidence.merge(zdf, on="track_id", how="left")

    for axis, cfg in MODULE_AXIS_PATTERNS.items():
        pos_cols = find_matching_module_cols(module_evidence, cfg.get("positive", []), suffix="_delta")
        neg_cols = find_matching_module_cols(module_evidence, cfg.get("negative", []), suffix="_delta")

        raw_scores = []
        axis_top_modules = []

        for _, row in combined.iterrows():
            vals = []
            top_parts = []

            for c in pos_cols:
                zc = f"{c}_z"
                v = safe_float(row.get(zc, np.nan))
                vals.append(v)
                top_parts.append((c, safe_float(row.get(c, np.nan)), v, "positive"))

            for c in neg_cols:
                zc = f"{c}_z"
                v = -safe_float(row.get(zc, np.nan))
                vals.append(v)
                top_parts.append((c, safe_float(row.get(c, np.nan)), v, "negative_inverse"))

            if vals:
                score = np.nanmax(vals)
            else:
                score = 0.0

            raw_scores.append(score)

            if top_parts:
                top_parts = sorted(top_parts, key=lambda x: abs(x[2]) if not pd.isna(x[2]) else -1, reverse=True)
                axis_top_modules.append(";".join([
                    f"{m}:{raw:.3g}:z={z:.2f}:{direction}"
                    for m, raw, z, direction in top_parts[:5]
                ]))
            else:
                axis_top_modules.append("")

        out[f"module_axis_{axis}_z"] = raw_scores
        out[f"module_axis_{axis}_norm"] = minmax_series(raw_scores).values
        out[f"module_axis_{axis}_top_modules"] = axis_top_modules

        for i, tid in enumerate(out["track_id"]):
            evidence_rows.append({
                "track_id": tid,
                "evidence_type": "module_axis",
                "axis": axis,
                "raw_z_score": out.loc[i, f"module_axis_{axis}_z"],
                "norm_score": out.loc[i, f"module_axis_{axis}_norm"],
                "top_modules": out.loc[i, f"module_axis_{axis}_top_modules"],
            })

    return out, pd.DataFrame(evidence_rows)


# -----------------------------------------------------------------------------
# Track feature engineering
# -----------------------------------------------------------------------------

def add_basic_track_features(tracks):
    df = tracks.copy()

    for c in [
        "delta_repair_score",
        "delta_core_probability",
        "delta_peri_probability",
        "delta_remote_probability",
        "track_priority_score",
        "mean_edge_score",
        "n_cells_sum",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df["repair_gain_norm"] = minmax_series(df["delta_repair_score"]).values
    df["core_reduction_norm"] = minmax_series(-df["delta_core_probability"]).values
    df["peri_gain_norm"] = minmax_series(df["delta_peri_probability"]).values
    df["remote_gain_norm"] = minmax_series(df["delta_remote_probability"]).values
    df["edge_score_norm"] = minmax_series(df["mean_edge_score"]).values
    df["track_priority_norm"] = minmax_series(df["track_priority_score"]).values

    state_seq = df["state_sequence"].fillna("").astype(str) if "state_sequence" in df.columns else pd.Series([""] * len(df))
    cell_seq = df["celltype_sequence"].fillna("").astype(str) if "celltype_sequence" in df.columns else pd.Series([""] * len(df))

    df["state_start_std"] = df.get("start_state", "").map(normalized_state) if "start_state" in df.columns else state_seq.map(lambda x: normalized_state(split_seq(x)[0]) if split_seq(x) else "unknown")
    df["state_end_std"] = df.get("end_state", "").map(normalized_state) if "end_state" in df.columns else state_seq.map(lambda x: normalized_state(split_seq(x)[-1]) if split_seq(x) else "unknown")

    df["is_core_to_peri_or_remote"] = (
        df["state_start_std"].eq("lesion-core-like")
        & df["state_end_std"].isin(["peri-infarct", "remote-like"])
    )
    df["is_core_persistent"] = (
        df["state_start_std"].eq("lesion-core-like")
        & df["state_end_std"].eq("lesion-core-like")
    )
    df["is_peri_to_remote"] = (
        df["state_start_std"].eq("peri-infarct")
        & df["state_end_std"].eq("remote-like")
    )
    df["is_remote_to_remote"] = (
        df["state_start_std"].eq("remote-like")
        & df["state_end_std"].eq("remote-like")
    )
    df["is_remote_to_injury"] = (
        df["state_start_std"].eq("remote-like")
        & df["state_end_std"].isin(["peri-infarct", "lesion-core-like"])
    )

    df["celltype_sequence_lower"] = cell_seq.str.lower()
    df["has_endothelial_context"] = df["celltype_sequence_lower"].map(lambda x: contains_any(x, ["endo", "vascular"]))
    df["has_astrocyte_context"] = df["celltype_sequence_lower"].map(lambda x: contains_any(x, ["astro"]))
    df["has_microglia_context"] = df["celltype_sequence_lower"].map(lambda x: contains_any(x, ["micro", "macrophage", "immune", "myeloid"]))
    df["has_ol_context"] = df["celltype_sequence_lower"].map(lambda x: contains_any(x, ["ol", "oligodend", "opc"]))
    df["has_low_confidence_context"] = df["celltype_sequence_lower"].map(lambda x: contains_any(x, ["low_confidence", "unknown"]))

    return df


def merge_axis_scores(tracks, module_axis_scores):
    if module_axis_scores is None or module_axis_scores.empty:
        out = tracks.copy()
        for axis in MODULE_AXIS_PATTERNS.keys():
            out[f"module_axis_{axis}_norm"] = 0.0
            out[f"module_axis_{axis}_z"] = 0.0
            out[f"module_axis_{axis}_top_modules"] = ""
        return out

    return tracks.merge(module_axis_scores, on="track_id", how="left")


# -----------------------------------------------------------------------------
# Label refinement
# -----------------------------------------------------------------------------

def score_track_axes(df):
    out = df.copy()

    def m(axis):
        return pd.to_numeric(out.get(f"module_axis_{axis}_norm", 0.0), errors="coerce").fillna(0.0)

    out["score_core_resolution"] = (
        0.34 * out["repair_gain_norm"].fillna(0)
        + 0.34 * out["core_reduction_norm"].fillna(0)
        + 0.12 * out[["peri_gain_norm", "remote_gain_norm"]].max(axis=1).fillna(0)
        + 0.12 * out["is_core_to_peri_or_remote"].astype(float)
        + 0.08 * out["edge_score_norm"].fillna(0)
    )

    out["score_vascular_barrier"] = (
        0.52 * m("vascular_barrier")
        + 0.16 * out["has_endothelial_context"].astype(float)
        + 0.14 * out["core_reduction_norm"].fillna(0)
        + 0.10 * out["peri_gain_norm"].fillna(0)
        + 0.08 * out["edge_score_norm"].fillna(0)
    )

    out["score_repair_remodeling"] = (
        0.55 * m("repair_remodeling")
        + 0.22 * out["repair_gain_norm"].fillna(0)
        + 0.10 * out["remote_gain_norm"].fillna(0)
        + 0.08 * out["core_reduction_norm"].fillna(0)
        + 0.05 * out["edge_score_norm"].fillna(0)
    )

    out["score_inflammatory"] = (
        0.62 * m("inflammatory")
        + 0.18 * out["has_microglia_context"].astype(float)
        + 0.08 * out["has_low_confidence_context"].astype(float)
        + 0.07 * out["peri_gain_norm"].fillna(0)
        + 0.05 * out["edge_score_norm"].fillna(0)
    )

    out["score_hypoxia_ferroptosis"] = (
        0.65 * m("hypoxia_ferroptosis")
        + 0.14 * out["is_core_persistent"].astype(float)
        + 0.10 * (1.0 - out["core_reduction_norm"].fillna(0))
        + 0.06 * out["has_low_confidence_context"].astype(float)
        + 0.05 * out["edge_score_norm"].fillna(0)
    )

    out["score_astrocyte_reactive"] = (
        0.65 * m("astrocyte_reactive")
        + 0.20 * out["has_astrocyte_context"].astype(float)
        + 0.10 * out["repair_gain_norm"].fillna(0)
        + 0.05 * out["edge_score_norm"].fillna(0)
    )

    out["score_oligodendroglial_context"] = (
        0.40 * out["has_ol_context"].astype(float)
        + 0.20 * out[["peri_gain_norm", "remote_gain_norm"]].max(axis=1).fillna(0)
        + 0.20 * out["repair_gain_norm"].fillna(0)
        + 0.10 * out["is_peri_to_remote"].astype(float)
        + 0.10 * out["edge_score_norm"].fillna(0)
    )

    out["score_remote_maintenance"] = (
        0.55 * out["is_remote_to_remote"].astype(float)
        + 0.20 * out["edge_score_norm"].fillna(0)
        + 0.15 * (1.0 - out["core_reduction_norm"].fillna(0))
        + 0.10 * out["has_ol_context"].astype(float)
    )

    out["score_persistent_core_injury"] = (
        0.45 * out["is_core_persistent"].astype(float)
        + 0.22 * m("hypoxia_ferroptosis")
        + 0.18 * (1.0 - out["core_reduction_norm"].fillna(0))
        + 0.10 * m("inflammatory")
        + 0.05 * out["edge_score_norm"].fillna(0)
    )

    return out


def top_module_evidence_for_track(row, axis):
    col = f"module_axis_{axis}_top_modules"
    return str(row.get(col, "") or "")


def choose_label(row, conservative=True):
    axis_score_cols = [
        "score_core_resolution",
        "score_vascular_barrier",
        "score_repair_remodeling",
        "score_inflammatory",
        "score_hypoxia_ferroptosis",
        "score_astrocyte_reactive",
        "score_oligodendroglial_context",
        "score_remote_maintenance",
        "score_persistent_core_injury",
    ]

    scores = {c.replace("score_", ""): safe_float(row.get(c, 0.0), 0.0) for c in axis_score_cols}
    axis = max(scores, key=scores.get)
    top_score = scores[axis]

    # Secondary evidence.
    state_seq = str(row.get("state_sequence", ""))
    cell_seq = str(row.get("celltype_sequence", ""))
    drep = safe_float(row.get("delta_repair_score", np.nan))
    dcore = safe_float(row.get("delta_core_probability", np.nan))
    dperi = safe_float(row.get("delta_peri_probability", np.nan))
    dremote = safe_float(row.get("delta_remote_probability", np.nan))

    has_endo = bool(row.get("has_endothelial_context", False))
    has_micro = bool(row.get("has_microglia_context", False))
    has_astro = bool(row.get("has_astrocyte_context", False))
    has_ol = bool(row.get("has_ol_context", False))

    # Conservative override rules.
    # 1. Strong core resolution should be explicitly named.
    if bool(row.get("is_core_to_peri_or_remote", False)) and drep > 0 and dcore < 0:
        if top_score < scores.get("core_resolution", 0.0) + 0.10 or scores.get("core_resolution", 0.0) >= 0.55:
            axis = "core_resolution"
            top_score = scores[axis]

    # 2. Do not call microglia if there is no microglia/immune celltype evidence.
    if axis == "inflammatory" and not has_micro:
        label = "inflammation-enriched latent track"
    elif axis == "inflammatory" and has_micro:
        label = "microglia/immune-inflammatory track"

    # 3. Do not call vascular/endothelial if no endothelial context; use barrier-remodeling.
    elif axis == "vascular_barrier" and has_endo:
        label = "vascular/endothelial-barrier transition track"
    elif axis == "vascular_barrier":
        label = "barrier-remodeling latent track"

    elif axis == "core_resolution":
        label = "core-to-repair transition track"

    elif axis == "repair_remodeling":
        label = "repair-permissive remodeling track"

    elif axis == "hypoxia_ferroptosis":
        label = "hypoxia/ferroptosis injury track"

    elif axis == "astrocyte_reactive" and has_astro:
        label = "astrocyte-reactive remodeling track"
    elif axis == "astrocyte_reactive":
        label = "astrocyte-reactive module-enriched track"

    elif axis == "oligodendroglial_context":
        label = "OL-associated peri/remote remodeling track"

    elif axis == "remote_maintenance":
        label = "remote-state maintenance track"

    elif axis == "persistent_core_injury":
        label = "persistent core-injury track"

    else:
        label = "state-transition track"

    # Additional conservative state-based fallback.
    if top_score < 0.35:
        if bool(row.get("is_peri_to_remote", False)):
            axis = "state_transition"
            label = "peri-to-remote transition track"
        elif bool(row.get("is_core_persistent", False)):
            axis = "persistent_core_injury"
            label = "persistent core-injury track"
        elif bool(row.get("is_remote_to_injury", False)):
            axis = "state_transition"
            label = "remote-to-injury expansion track"
        elif bool(row.get("is_remote_to_remote", False)):
            axis = "remote_maintenance"
            label = "remote-state maintenance track"
        else:
            axis = "state_transition"
            label = "state-transition track"

    # Confidence.
    evidence_count = 0
    if abs(drep) > 0.10 or abs(dcore) > 0.10:
        evidence_count += 1
    if any([has_endo, has_micro, has_astro, has_ol]):
        evidence_count += 1
    if any(safe_float(row.get(f"module_axis_{a}_norm", 0), 0) > 0.60 for a in MODULE_AXIS_PATTERNS.keys()):
        evidence_count += 1
    if bool(row.get("is_core_to_peri_or_remote", False)) or bool(row.get("is_peri_to_remote", False)) or bool(row.get("is_core_persistent", False)):
        evidence_count += 1

    if top_score >= 0.70 and evidence_count >= 2:
        confidence = "high"
    elif top_score >= 0.45 and evidence_count >= 1:
        confidence = "medium"
    else:
        confidence = "low"

    # Review-ready evidence text.
    top_axis_modules = []
    for a in MODULE_AXIS_PATTERNS.keys():
        ev = top_module_evidence_for_track(row, a)
        if ev:
            top_axis_modules.append(f"{a}: {ev}")

    module_text = " | ".join(top_axis_modules[:3]) if top_axis_modules else "module evidence not available"

    evidence_text = (
        f"state={state_seq}; celltype={cell_seq}; "
        f"Δrepair={drep:.3g}, Δcore={dcore:.3g}, Δperi={dperi:.3g}, Δremote={dremote:.3g}; "
        f"top_module_evidence={module_text}"
    )

    short_label = label.replace(" track", "")

    return {
        "track_label_refined": label,
        "track_label_short": short_label,
        "biological_axis_refined": axis,
        "label_confidence": confidence,
        "label_score": top_score,
        "label_evidence_summary": evidence_text,
    }


def refine_labels(tracks, module_evidence, module_axis_scores):
    t = add_basic_track_features(tracks)
    t = merge_axis_scores(t, module_axis_scores)
    t = score_track_axes(t)

    rows = []
    for _, row in t.iterrows():
        lab = choose_label(row)
        base = row.to_dict()
        base.update(lab)
        rows.append(base)

    refined = pd.DataFrame(rows)

    # Add reviewer caution flags.
    refined["label_caution_flag"] = ""
    refined.loc[
        refined["biological_axis_refined"].eq("inflammatory") & (~refined["has_microglia_context"].astype(bool)),
        "label_caution_flag"
    ] = "inflammatory module evidence without microglia/immune celltype dominance"

    refined.loc[
        refined["biological_axis_refined"].eq("vascular_barrier") & (~refined["has_endothelial_context"].astype(bool)),
        "label_caution_flag"
    ] = "barrier module evidence without endothelial celltype dominance"

    refined.loc[
        refined["label_confidence"].eq("low"),
        "label_caution_flag"
    ] = refined["label_caution_flag"].where(
        refined["label_caution_flag"].astype(str).ne(""),
        "low-confidence label; interpret as state-transition audit"
    )

    # Stable ordering.
    sort_cols = [c for c in ["track_rank", "track_priority_score"] if c in refined.columns]
    if "track_rank" in sort_cols:
        refined = refined.sort_values("track_rank").reset_index(drop=True)
    else:
        refined = refined.sort_values("track_priority_score", ascending=False).reset_index(drop=True)

    return refined


def make_axis_score_matrix(refined):
    cols = ["track_id", "track_rank", "track_label_refined", "biological_axis_refined", "label_confidence"]
    score_cols = [c for c in refined.columns if c.startswith("score_")]
    keep = [c for c in cols if c in refined.columns] + score_cols
    return refined[keep].copy()


def make_evidence_long(refined, module_axis_scores):
    rows = []

    for _, r in refined.iterrows():
        tid = r["track_id"]

        for axis in [
            "core_resolution",
            "vascular_barrier",
            "repair_remodeling",
            "inflammatory",
            "hypoxia_ferroptosis",
            "astrocyte_reactive",
            "oligodendroglial_context",
            "remote_maintenance",
            "persistent_core_injury",
        ]:
            score_col = f"score_{axis}"
            if score_col in refined.columns:
                rows.append({
                    "track_id": tid,
                    "track_rank": r.get("track_rank", np.nan),
                    "evidence_type": "axis_score",
                    "axis": axis,
                    "score": safe_float(r.get(score_col, np.nan)),
                    "evidence_detail": "",
                })

        rows.append({
            "track_id": tid,
            "track_rank": r.get("track_rank", np.nan),
            "evidence_type": "final_label",
            "axis": r.get("biological_axis_refined", ""),
            "score": safe_float(r.get("label_score", np.nan)),
            "evidence_detail": r.get("label_evidence_summary", ""),
        })

        caution = str(r.get("label_caution_flag", ""))
        if caution:
            rows.append({
                "track_id": tid,
                "track_rank": r.get("track_rank", np.nan),
                "evidence_type": "caution",
                "axis": r.get("biological_axis_refined", ""),
                "score": np.nan,
                "evidence_detail": caution,
            })

    if module_axis_scores is not None and not module_axis_scores.empty:
        for _, r in module_axis_scores.iterrows():
            tid = r["track_id"]
            for axis in MODULE_AXIS_PATTERNS.keys():
                sc = safe_float(r.get(f"module_axis_{axis}_norm", np.nan))
                ev = str(r.get(f"module_axis_{axis}_top_modules", ""))
                rows.append({
                    "track_id": tid,
                    "track_rank": np.nan,
                    "evidence_type": "module_axis",
                    "axis": axis,
                    "score": sc,
                    "evidence_detail": ev,
                })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Assignment and node patching
# -----------------------------------------------------------------------------

def patch_assignments(assignments, refined):
    if assignments is None or assignments.empty:
        return pd.DataFrame()

    cols = [
        "track_id",
        "track_label_refined",
        "track_label_short",
        "biological_axis_refined",
        "label_confidence",
        "label_score",
        "label_caution_flag",
    ]
    cols = [c for c in cols if c in refined.columns]

    out = assignments.merge(refined[cols], on="track_id", how="left")
    return out


def patch_nodes(nodes, refined):
    if nodes is None or nodes.empty:
        return pd.DataFrame()

    rows = []
    for _, n in nodes.iterrows():
        node_id = n["node_id"]
        sub = refined[refined["track_nodes"].astype(str).str.contains(re.escape(str(node_id)), regex=True, na=False)]

        row = n.to_dict()
        if len(sub):
            top = sub.sort_values("track_priority_score", ascending=False).head(1).iloc[0]
            row["top_refined_track_id"] = top.get("track_id", "")
            row["top_refined_track_label"] = top.get("track_label_refined", "")
            row["top_refined_track_axis"] = top.get("biological_axis_refined", "")
            row["n_refined_tracks_passing_node"] = int(len(sub))
            row["refined_track_ids_passing_node"] = ";".join(sub["track_id"].astype(str).tolist()[:20])
        else:
            row["top_refined_track_id"] = ""
            row["top_refined_track_label"] = ""
            row["top_refined_track_axis"] = ""
            row["n_refined_tracks_passing_node"] = 0
            row["refined_track_ids_passing_node"] = ""

        rows.append(row)

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


def plot_label_refinement(refined, axis_matrix, module_evidence, outdir, top_n=18, dpi=600):
    colors = axis_color_map()
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_StrokeNiche_TrackLabelRefinement_{mode}"

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.32)

        # A: top tracks by priority, colored by refined axis.
        axA = fig.add_subplot(gs[0, 0])
        top = refined.sort_values("track_rank").head(top_n).copy()
        y = np.arange(len(top))[::-1]
        bar_colors = [colors.get(a, "#B0B0B0") for a in top["biological_axis_refined"]][::-1]
        axA.barh(y, top["track_priority_score"].values[::-1], color=bar_colors)

        if annotate:
            labels = (top["track_id"].astype(str) + " | " + top["track_label_short"].astype(str)).values[::-1]
            axA.set_yticks(y)
            axA.set_yticklabels(labels, fontsize=7)
            axA.set_xlabel("Track priority score")
            axA.set_title("A | Refined track labels", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axA)

        # B: repair-core scatter, colored by refined axis.
        axB = fig.add_subplot(gs[0, 1])
        top2 = refined.sort_values("track_rank").head(max(top_n, 20)).copy()
        for axis, sub in top2.groupby("biological_axis_refined"):
            axB.scatter(
                sub["delta_core_probability"],
                sub["delta_repair_score"],
                s=70 + 220 * minmax_series(sub["track_priority_score"]),
                color=colors.get(axis, "#B0B0B0"),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
                label=axis,
            )
            if annotate:
                for _, r in sub.iterrows():
                    axB.text(
                        r["delta_core_probability"],
                        r["delta_repair_score"],
                        str(r["track_id"]).replace("Track_", "T"),
                        fontsize=7,
                    )

        axB.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        axB.axvline(0, color="#777777", linestyle="--", linewidth=0.8)

        if annotate:
            axB.set_xlabel("Δ core probability")
            axB.set_ylabel("Δ repair score")
            axB.set_title("B | Repair-core shift by refined axis", loc="left", fontsize=12, fontweight="bold")
            axB.legend(frameon=False, fontsize=7, loc="best")
        else:
            strip_text(axB)

        # C: axis score heatmap.
        axC = fig.add_subplot(gs[1, 0])
        score_cols = [c for c in refined.columns if c.startswith("score_")]
        hm = refined.sort_values("track_rank").head(top_n)[["track_id"] + score_cols].copy()
        if score_cols:
            mat = hm[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            im = axC.imshow(mat, aspect="auto", interpolation="nearest", vmin=0, vmax=max(1.0, np.nanmax(mat)))
            if annotate:
                axC.set_yticks(np.arange(len(hm)))
                axC.set_yticklabels(hm["track_id"], fontsize=7)
                axC.set_xticks(np.arange(len(score_cols)))
                axC.set_xticklabels([c.replace("score_", "") for c in score_cols], rotation=45, ha="right", fontsize=7)
                axC.set_title("C | Evidence score matrix", loc="left", fontsize=12, fontweight="bold")
                cbar = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)
            else:
                strip_text(axC)
        else:
            if annotate:
                axC.text(0.5, 0.5, "No score columns", ha="center", va="center")
            else:
                strip_text(axC)

        # D: refined label distribution.
        axD = fig.add_subplot(gs[1, 1])
        counts = refined["biological_axis_refined"].value_counts()
        labels = counts.index.tolist()
        vals = counts.values
        cols = [colors.get(x, "#B0B0B0") for x in labels]
        axD.barh(np.arange(len(labels))[::-1], vals[::-1], color=cols[::-1])

        if annotate:
            axD.set_yticks(np.arange(len(labels))[::-1])
            axD.set_yticklabels(labels[::-1], fontsize=8)
            axD.set_xlabel("Number of tracks")
            axD.set_title("D | Refined biological-axis distribution", loc="left", fontsize=12, fontweight="bold")
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
            fig.suptitle("Step 64c | Reviewer-ready StrokeNiche track label refinement", fontsize=15, fontweight="bold")

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
    ap.add_argument("--in64", default=str(DEFAULT_IN64))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--top_n_fig", type=int, default=18)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    in64 = Path(args.in64)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 64c: track label refinement")
    log("=" * 100)
    log(f"in64={in64}")
    log(f"outdir={outdir}")

    tracks, nodes, edges, assignments, module_matrix, input_meta = load_inputs(in64)

    module_evidence = extract_module_evidence_from_matrix(module_matrix, tracks, nodes)
    module_evidence.to_csv(outdir / "step64c_track_module_evidence_matrix.csv", index=False)

    module_axis_scores, module_axis_evidence = build_module_axis_scores(module_evidence)

    refined = refine_labels(tracks, module_evidence, module_axis_scores)

    axis_matrix = make_axis_score_matrix(refined)
    evidence_long = make_evidence_long(refined, module_axis_scores)

    assignments_refined = patch_assignments(assignments, refined)
    nodes_refined = patch_nodes(nodes, refined)

    refined.to_csv(outdir / "step64c_track_label_refined.csv", index=False)
    evidence_long.to_csv(outdir / "step64c_track_label_evidence_long.csv", index=False)
    axis_matrix.to_csv(outdir / "step64c_track_axis_score_matrix.csv", index=False)
    assignments_refined.to_csv(outdir / "step64c_track_assignment_by_cell.refined_labels.csv", index=False)
    nodes_refined.to_csv(outdir / "step64c_dynamics_graph_nodes.refined_labels.csv", index=False)

    fig_outputs = plot_label_refinement(
        refined=refined,
        axis_matrix=axis_matrix,
        module_evidence=module_evidence,
        outdir=outdir,
        top_n=args.top_n_fig,
        dpi=args.dpi,
    )

    label_counts = refined["biological_axis_refined"].value_counts(dropna=False).to_dict()
    confidence_counts = refined["label_confidence"].value_counts(dropna=False).to_dict()
    caution_counts = refined["label_caution_flag"].replace("", "none").value_counts(dropna=False).to_dict()

    report = {
        "status": "ok",
        "analysis_name": "Step64c reviewer-ready track label refinement",
        "in64": str(in64),
        "outdir": str(outdir),
        "input_files": input_meta,
        "n_tracks": int(len(refined)),
        "n_assignments_refined": int(len(assignments_refined)) if assignments_refined is not None else 0,
        "label_counts": label_counts,
        "confidence_counts": confidence_counts,
        "caution_counts": caution_counts,
        "figure_outputs": fig_outputs,
        "outputs": {
            "track_label_refined": str(outdir / "step64c_track_label_refined.csv"),
            "track_label_evidence_long": str(outdir / "step64c_track_label_evidence_long.csv"),
            "track_axis_score_matrix": str(outdir / "step64c_track_axis_score_matrix.csv"),
            "track_module_evidence_matrix": str(outdir / "step64c_track_module_evidence_matrix.csv"),
            "track_assignment_by_cell_refined": str(outdir / "step64c_track_assignment_by_cell.refined_labels.csv"),
            "nodes_refined": str(outdir / "step64c_dynamics_graph_nodes.refined_labels.csv"),
            "report_json": str(outdir / "step64c_report.json"),
            "report_txt": str(outdir / "step64c_report.txt"),
        },
        "interpretation_note": (
            "Refined labels are reviewer-facing biological annotations derived from state transition, "
            "celltype context, module evidence and repair-core shifts. Labels with caution flags should "
            "be interpreted as module-enriched latent tracks rather than direct cell-lineage assignments."
        ),
    }

    (outdir / "step64c_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step 64c track label refinement report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Top refined tracks:")
    show = [
        "track_id",
        "track_nodes",
        "track_label",
        "track_label_refined",
        "biological_axis_refined",
        "label_confidence",
        "label_score",
        "label_caution_flag",
        "state_sequence",
        "celltype_sequence",
        "delta_repair_score",
        "delta_core_probability",
        "delta_peri_probability",
        "delta_remote_probability",
        "track_priority_score",
        "label_evidence_summary",
    ]
    show = [c for c in show if c in refined.columns]
    lines.append(refined[show].head(60).to_string(index=False))
    lines.append("")
    lines.append("Label counts:")
    lines.append(pd.Series(label_counts).to_string())
    lines.append("")
    lines.append("Confidence counts:")
    lines.append(pd.Series(confidence_counts).to_string())
    lines.append("")
    lines.append("Caution counts:")
    lines.append(pd.Series(caution_counts).to_string())

    (outdir / "step64c_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step 64c")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Top refined tracks:")
    log(refined[show].head(25).to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
