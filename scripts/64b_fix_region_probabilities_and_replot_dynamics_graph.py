#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
64b_fix_region_probabilities_and_replot_dynamics_graph.py

Purpose
-------
Fix Step 64 region-probability source issue and regenerate dynamics graph.

Why
---
Original Step 64 used core_probability / peri_probability / remote_probability,
but these columns may be identical in step62_modeling_input_table.csv or after
standardization. This caused:
  delta_core_probability == delta_peri_probability == delta_remote_probability

This script:
  1. Audits original Step64 outputs.
  2. Loads original cell-level input.
  3. Standardizes input using the original Step64 code.
  4. Re-selects correct lesion-core / peri-infarct / remote-like probability triplet.
  5. Overwrites:
       core_probability_std
       peri_probability_std
       remote_probability_std
     with corrected independent columns.
  6. Re-runs clustering, nodes, edges, tracks.
  7. Exports two figure versions:
       annotated: normal labels / titles
       clean_no_text: no text labels, no axes text, no legend text

Important
---------
This does NOT overwrite original Step64 results.

Outputs
-------
outdir/
  step64b_original_probability_audit.csv
  step64b_probability_source_audit.csv
  step64b_selected_probability_source.json
  step64b_modeling_input_used.fixed_region_probs.csv
  step64b_clustered_cells.csv
  step64b_dynamics_graph_nodes.csv
  step64b_dynamics_graph_edges_all.csv
  step64b_dynamics_graph_edges.csv
  step64b_track_summary.csv
  step64b_track_assignment_by_cell.csv
  step64b_track_module_matrix.csv
  step64b_report.json
  step64b_report.txt
  Fig_StrokeNiche_DynamicsGraph_annotated.pdf/svg/png
  Fig_StrokeNiche_DynamicsGraph_clean_no_text.pdf/svg/png
"""

from pathlib import Path
import argparse
import importlib.util
import json
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


BASE = Path("/mnt/h/vir/ST")

DEFAULT_STEP64_SCRIPT = BASE / "64_strokeniche_dynamics_graph_track_reconstruction.py"
DEFAULT_INPUT = BASE / "results/step8_strokeniche_perturbmap/generalization_62/step62_modeling_input_table.csv"
DEFAULT_COORD = BASE / "results/figure5_trajectory/figures/fig5_strokeniche_trajectory_latent_state_coordinates.csv"
DEFAULT_TRUE_Z = BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54/adapter_true_hidden_latent_for_neighbor_head.csv"
DEFAULT_ORIG64 = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64"
DEFAULT_OUT = BASE / "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_kmeans"


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


def safe_float(x):
    try:
        if pd.isna(x):
            return np.nan
        return float(x)
    except Exception:
        return np.nan


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def is_numeric_like(s, min_fraction=0.8):
    x = pd.to_numeric(s, errors="coerce")
    return bool(x.notna().mean() >= min_fraction)


def numeric_series(df, c):
    return pd.to_numeric(df[c], errors="coerce")


def minmax(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def load_step64_module(script_path):
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    spec = importlib.util.spec_from_file_location("step64_original_module", str(script_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def normalize_state(x):
    s = str(x).strip().lower().replace("-", "_").replace(" ", "_")
    if "lesion_core" in s or s == "core" or "core_like" in s:
        return "lesion-core-like"
    if "peri" in s or "penumbra" in s:
        return "peri-infarct"
    if "remote" in s or "normal" in s:
        return "remote-like"
    return "other"


# -----------------------------------------------------------------------------
# Original Step64 probability audit
# -----------------------------------------------------------------------------

def audit_original_step64(orig64_dir):
    orig64_dir = Path(orig64_dir)
    rows = []

    nodes_p = orig64_dir / "step64_dynamics_graph_nodes.csv"
    tracks_p = orig64_dir / "step64_track_summary.csv"

    nodes = read_csv(nodes_p, required=False)
    tracks = read_csv(tracks_p, required=False)

    if nodes is not None and not nodes.empty:
        cols = ["core_probability_mean", "peri_probability_mean", "remote_probability_mean"]
        if all(c in nodes.columns for c in cols):
            x = nodes[cols].apply(pd.to_numeric, errors="coerce")
            same = (
                x[cols[0]].round(12).eq(x[cols[1]].round(12))
                & x[cols[0]].round(12).eq(x[cols[2]].round(12))
            )
            corr = x.corr()
            rows.append({
                "source_table": "nodes",
                "path": str(nodes_p),
                "shape": str(nodes.shape),
                "checked_columns": ";".join(cols),
                "all_three_identical_rows": int(same.sum()),
                "n_rows": int(len(x)),
                "all_three_identical_fraction": float(same.mean()),
                "corr_core_peri": safe_float(corr.loc[cols[0], cols[1]]),
                "corr_core_remote": safe_float(corr.loc[cols[0], cols[2]]),
                "corr_peri_remote": safe_float(corr.loc[cols[1], cols[2]]),
                "status": "suspicious_identical" if same.mean() > 0.95 else "ok_not_identical",
            })

    if tracks is not None and not tracks.empty:
        cols = ["delta_core_probability", "delta_peri_probability", "delta_remote_probability"]
        if all(c in tracks.columns for c in cols):
            x = tracks[cols].apply(pd.to_numeric, errors="coerce")
            same = (
                x[cols[0]].round(12).eq(x[cols[1]].round(12))
                & x[cols[0]].round(12).eq(x[cols[2]].round(12))
            )
            corr = x.corr()
            rows.append({
                "source_table": "tracks",
                "path": str(tracks_p),
                "shape": str(tracks.shape),
                "checked_columns": ";".join(cols),
                "all_three_identical_rows": int(same.sum()),
                "n_rows": int(len(x)),
                "all_three_identical_fraction": float(same.mean()),
                "corr_core_peri": safe_float(corr.loc[cols[0], cols[1]]),
                "corr_core_remote": safe_float(corr.loc[cols[0], cols[2]]),
                "corr_peri_remote": safe_float(corr.loc[cols[1], cols[2]]),
                "status": "suspicious_identical" if same.mean() > 0.95 else "ok_not_identical",
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Region probability source selection
# -----------------------------------------------------------------------------

def candidate_triplets(df, allow_neighbor_regionprob=False):
    """
    Return candidate triplets:
      core_col, peri_col, remote_col, source_label, source_type
    """

    exact = [
        ("prob_lesion_core", "prob_peri_infarct", "prob_remote_like", "prob_region_state", "preferred_cell_state"),
        ("P_lesion_core", "P_peri_infarct", "P_remote_like", "P_region_state", "preferred_cell_state"),
        ("lesion_core_probability", "peri_infarct_probability", "remote_like_probability", "long_probability", "preferred_cell_state"),
        ("predicted_core_probability", "predicted_peri_probability", "predicted_remote_probability", "predicted_probability", "preferred_cell_state"),
        ("core_prob", "peri_prob", "remote_prob", "short_probability", "preferred_cell_state"),
        ("core_probability_raw", "peri_probability_raw", "remote_probability_raw", "raw_probability", "preferred_cell_state"),
        ("core_probability", "peri_probability", "remote_probability", "generic_probability", "generic"),
        ("core_probability_std", "peri_probability_std", "remote_probability_std", "existing_std_probability", "existing_std"),
    ]

    # Neighbor region probability is a fallback, not preferred, because it is
    # neighborhood mean rather than self state.
    neighbor = [
        (
            "neighbor_mean_feature_regionprob_prob_lesion_core",
            "neighbor_mean_feature_regionprob_prob_peri_infarct",
            "neighbor_mean_feature_regionprob_prob_remote_like",
            "neighbor_regionprob_prob",
            "neighbor_regionprob_fallback",
        ),
        (
            "neighbor_mean_feature_regionprob_P_lesion_core",
            "neighbor_mean_feature_regionprob_P_peri_infarct",
            "neighbor_mean_feature_regionprob_P_remote_like",
            "neighbor_regionprob_P",
            "neighbor_regionprob_fallback",
        ),
    ]

    out = []
    for a, b, c, label, typ in exact:
        if a in df.columns and b in df.columns and c in df.columns:
            out.append((a, b, c, label, typ))

    if allow_neighbor_regionprob:
        for a, b, c, label, typ in neighbor:
            if a in df.columns and b in df.columns and c in df.columns:
                out.append((a, b, c, label, typ))

    # Fuzzy normalized matching.
    lookup = {}
    for c in df.columns:
        lookup.setdefault(norm_col(c), c)

    fuzzy_sets = [
        ("prob_lesion_core", "prob_peri_infarct", "prob_remote_like", "fuzzy_prob_region_state", "preferred_cell_state"),
        ("p_lesion_core", "p_peri_infarct", "p_remote_like", "fuzzy_P_region_state", "preferred_cell_state"),
        ("lesion_core_prob", "peri_infarct_prob", "remote_like_prob", "fuzzy_short_prob", "preferred_cell_state"),
        ("lesion_core_probability", "peri_infarct_probability", "remote_like_probability", "fuzzy_long_probability", "preferred_cell_state"),
    ]

    for a, b, c, label, typ in fuzzy_sets:
        if a in lookup and b in lookup and c in lookup:
            trip = (lookup[a], lookup[b], lookup[c], label, typ)
            if trip not in out:
                out.append(trip)

    return out


def state_alignment_score(df, core, peri, remote):
    if "state_group_std" not in df.columns:
        return np.nan

    tmp = df[["state_group_std", core, peri, remote]].copy()
    for c in [core, peri, remote]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    tmp = tmp.dropna(subset=[core, peri, remote])
    if tmp.empty:
        return np.nan

    score_terms = []

    core_sub = tmp[tmp["state_group_std"].eq("lesion-core-like")]
    if len(core_sub):
        score_terms.append(core_sub[core].mean() - max(core_sub[peri].mean(), core_sub[remote].mean()))

    peri_sub = tmp[tmp["state_group_std"].eq("peri-infarct")]
    if len(peri_sub):
        score_terms.append(peri_sub[peri].mean() - max(peri_sub[core].mean(), peri_sub[remote].mean()))

    remote_sub = tmp[tmp["state_group_std"].eq("remote-like")]
    if len(remote_sub):
        score_terms.append(remote_sub[remote].mean() - max(remote_sub[core].mean(), remote_sub[peri].mean()))

    if not score_terms:
        return np.nan

    return float(np.nanmean(score_terms))


def audit_probability_triplet(df, core, peri, remote, label, typ):
    x = df[[core, peri, remote]].apply(pd.to_numeric, errors="coerce")
    nonnull_fraction = float(x.notna().all(axis=1).mean())

    x2 = x.dropna()
    if x2.empty:
        return {
            "source_label": label,
            "source_type": typ,
            "core_col": core,
            "peri_col": peri,
            "remote_col": remote,
            "status": "empty_numeric",
            "score": -999,
        }

    pairdiff = (
        (x2[core] - x2[peri]).abs().mean()
        + (x2[core] - x2[remote]).abs().mean()
        + (x2[peri] - x2[remote]).abs().mean()
    ) / 3.0

    identical_fraction = (
        x2[core].round(12).eq(x2[peri].round(12))
        & x2[core].round(12).eq(x2[remote].round(12))
    ).mean()

    corr = x2.corr()
    max_abs_corr = np.nanmax([
        abs(safe_float(corr.loc[core, peri])),
        abs(safe_float(corr.loc[core, remote])),
        abs(safe_float(corr.loc[peri, remote])),
    ])

    row_sum = x2[core] + x2[peri] + x2[remote]
    sum_median = float(row_sum.median())
    sum_mad_from1 = float((row_sum - 1.0).abs().median())

    variability = float(np.nanmean([x2[core].std(), x2[peri].std(), x2[remote].std()]))
    align = state_alignment_score(df, core, peri, remote)

    preferred_bonus = 0.0
    if typ == "preferred_cell_state":
        preferred_bonus = 2.0
    elif typ == "generic":
        preferred_bonus = 0.2
    elif typ == "existing_std":
        preferred_bonus = -0.5
    elif typ == "neighbor_regionprob_fallback":
        preferred_bonus = -1.0

    valid = True
    status = "ok_candidate"

    if nonnull_fraction < 0.5:
        valid = False
        status = "too_many_missing"
    if identical_fraction > 0.95 or pairdiff < 1e-8:
        valid = False
        status = "three_columns_identical"
    if variability < 1e-8:
        valid = False
        status = "near_constant"

    score = (
        preferred_bonus
        + 2.0 * nonnull_fraction
        + 1.5 * min(pairdiff, 1.0)
        + 0.8 * min(variability, 1.0)
        - 0.7 * max_abs_corr
        - 0.4 * min(sum_mad_from1, 5.0)
    )

    if not pd.isna(align):
        score += 1.5 * align

    if not valid:
        score -= 100

    return {
        "source_label": label,
        "source_type": typ,
        "core_col": core,
        "peri_col": peri,
        "remote_col": remote,
        "nonnull_fraction": nonnull_fraction,
        "pairwise_abs_diff_mean": float(pairdiff),
        "identical_fraction": float(identical_fraction),
        "max_abs_pairwise_corr": float(max_abs_corr),
        "sum_median": sum_median,
        "sum_median_abs_error_from1": sum_mad_from1,
        "mean_sd": variability,
        "state_alignment_score": align,
        "status": status,
        "score": float(score),
    }


def make_state_onehot_probs(df, smoothing=0.02):
    state = df["state_group_std"].map(normalize_state) if "state_group_std" in df.columns else pd.Series(["other"] * len(df), index=df.index)

    core = pd.Series(smoothing, index=df.index, dtype=float)
    peri = pd.Series(smoothing, index=df.index, dtype=float)
    remote = pd.Series(smoothing, index=df.index, dtype=float)

    core[state.eq("lesion-core-like")] = 1.0 - 2 * smoothing
    peri[state.eq("peri-infarct")] = 1.0 - 2 * smoothing
    remote[state.eq("remote-like")] = 1.0 - 2 * smoothing

    # Other stays approximately uniform.
    other = ~(state.isin(["lesion-core-like", "peri-infarct", "remote-like"]))
    core[other] = 1 / 3
    peri[other] = 1 / 3
    remote[other] = 1 / 3

    return core, peri, remote


def select_and_apply_region_probabilities(df, allow_neighbor_regionprob=False, state_onehot_smoothing=0.02):
    audits = []
    for core, peri, remote, label, typ in candidate_triplets(df, allow_neighbor_regionprob=allow_neighbor_regionprob):
        audits.append(audit_probability_triplet(df, core, peri, remote, label, typ))

    audit = pd.DataFrame(audits)
    if not audit.empty:
        audit = audit.sort_values("score", ascending=False).reset_index(drop=True)

    out = df.copy()

    selected = None
    if not audit.empty:
        ok = audit[audit["status"].eq("ok_candidate")].copy()
        if not ok.empty:
            selected = ok.iloc[0].to_dict()

    if selected is not None:
        core_col = selected["core_col"]
        peri_col = selected["peri_col"]
        remote_col = selected["remote_col"]

        out["core_probability_std"] = pd.to_numeric(out[core_col], errors="coerce")
        out["peri_probability_std"] = pd.to_numeric(out[peri_col], errors="coerce")
        out["remote_probability_std"] = pd.to_numeric(out[remote_col], errors="coerce")

        selected_meta = {
            "selection_status": "selected_existing_triplet",
            "source_label": selected["source_label"],
            "source_type": selected["source_type"],
            "core_col": core_col,
            "peri_col": peri_col,
            "remote_col": remote_col,
            "score": selected["score"],
            "audit_status": selected["status"],
        }
    else:
        core, peri, remote = make_state_onehot_probs(out, smoothing=state_onehot_smoothing)
        out["core_probability_std"] = core
        out["peri_probability_std"] = peri
        out["remote_probability_std"] = remote

        selected_meta = {
            "selection_status": "fallback_state_onehot",
            "source_label": "state_group_onehot_smoothed",
            "source_type": "state_onehot_fallback",
            "core_col": "state_group_std==lesion-core-like",
            "peri_col": "state_group_std==peri-infarct",
            "remote_col": "state_group_std==remote-like",
            "state_onehot_smoothing": state_onehot_smoothing,
            "warning": "No non-identical cell-level probability triplet was found. Used smoothed state one-hot fallback.",
        }

    # Fill occasional missing values by state-onehot fallback.
    miss = out[["core_probability_std", "peri_probability_std", "remote_probability_std"]].isna().any(axis=1)
    if miss.any():
        core, peri, remote = make_state_onehot_probs(out, smoothing=state_onehot_smoothing)
        out.loc[miss, "core_probability_std"] = out.loc[miss, "core_probability_std"].fillna(core.loc[miss])
        out.loc[miss, "peri_probability_std"] = out.loc[miss, "peri_probability_std"].fillna(peri.loc[miss])
        out.loc[miss, "remote_probability_std"] = out.loc[miss, "remote_probability_std"].fillna(remote.loc[miss])
        selected_meta["n_rows_filled_by_state_onehot"] = int(miss.sum())

    # Diagnostic columns.
    out["region_probability_source"] = selected_meta["source_label"]
    out["region_probability_selection_status"] = selected_meta["selection_status"]

    check = audit_probability_triplet(
        out,
        "core_probability_std",
        "peri_probability_std",
        "remote_probability_std",
        "selected_std_after_fix",
        "selected_output",
    )
    selected_meta["post_fix_identical_fraction"] = check.get("identical_fraction", np.nan)
    selected_meta["post_fix_pairwise_abs_diff_mean"] = check.get("pairwise_abs_diff_mean", np.nan)
    selected_meta["post_fix_sum_median"] = check.get("sum_median", np.nan)

    return out, audit, selected_meta


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def state_colors():
    return {
        "lesion-core-like": "#D84A3A",
        "peri-infarct": "#E7A23B",
        "remote-like": "#4D91C6",
        "other": "#8A8A8A",
        "unknown": "#8A8A8A",
    }


def strip_all_text(ax):
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def plot_graph_versions(clustered, nodes, selected_edges, tracks, outdir, dpi=600, seed=42):
    outputs = {}
    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_StrokeNiche_DynamicsGraph_{mode}"

        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.30)

        cmap = state_colors()

        # A: layered graph
        axA = fig.add_subplot(gs[0, 0])
        tps = sorted(nodes["timepoint"].unique(), key=lambda x: int(str(x).replace("D", "")))
        x_map = {tp: i for i, tp in enumerate(tps)}
        y_positions = {}

        for tp in tps:
            sub = nodes[nodes["timepoint"].eq(tp)].sort_values("latent_y")
            ys = [0.5] if len(sub) == 1 else np.linspace(0.08, 0.92, len(sub))
            for y, (_, r) in zip(ys, sub.iterrows()):
                y_positions[r["node_id"]] = y

        if selected_edges is not None and not selected_edges.empty:
            for _, e in selected_edges.iterrows():
                if e["from_node"] not in y_positions or e["to_node"] not in y_positions:
                    continue
                x0, x1 = x_map[e["from_timepoint"]], x_map[e["to_timepoint"]]
                y0, y1 = y_positions[e["from_node"]], y_positions[e["to_node"]]
                score = safe_float(e.get("edge_score", 0.5))
                axA.plot(
                    [x0, x1], [y0, y1],
                    linewidth=0.5 + 3.0 * score,
                    alpha=0.20 + 0.55 * score,
                    color="#6E7781",
                    zorder=1,
                )

        max_n = max(nodes["n_cells"].max(), 1)
        for _, r in nodes.iterrows():
            x, y = x_map[r["timepoint"]], y_positions[r["node_id"]]
            size = 60 + 520 * (r["n_cells"] / max_n)
            color = cmap.get(r["dominant_state"], "#8A8A8A")
            axA.scatter(x, y, s=size, color=color, edgecolor="white", linewidth=1.0, zorder=3)
            if annotate:
                label = f"{r['cluster_local']}\n{str(r['dominant_state']).replace('-like','')}"
                axA.text(x, y, label, ha="center", va="center", fontsize=6, zorder=4)

        if annotate:
            axA.set_title("A | StrokeNiche dynamics graph", loc="left", fontsize=12, fontweight="bold")
            axA.set_xticks(list(x_map.values()))
            axA.set_xticklabels(tps, fontsize=10, fontweight="bold")
            axA.set_xlabel("Timepoint")
            axA.set_yticks([])
            axA.text(
                0.01, -0.12,
                "Node size = cell count; edge width = latent/module similarity",
                transform=axA.transAxes,
                fontsize=8,
            )
        else:
            strip_all_text(axA)

        axA.set_xlim(-0.4, len(tps) - 0.6)
        axA.set_ylim(0, 1)

        # B: latent connectivity
        axB = fig.add_subplot(gs[0, 1])
        scatter_df = clustered.sample(n=min(len(clustered), 6000), random_state=seed)

        for state, sub in scatter_df.groupby("state_group_std"):
            axB.scatter(
                sub["latent_x"], sub["latent_y"],
                s=4,
                alpha=0.14,
                color=cmap.get(state, "#8A8A8A"),
                label=state if annotate else None,
            )

        node_pos = nodes.set_index("node_id")[["latent_x", "latent_y"]].to_dict(orient="index")
        if selected_edges is not None and not selected_edges.empty:
            for _, e in selected_edges.iterrows():
                if e["from_node"] not in node_pos or e["to_node"] not in node_pos:
                    continue
                p0, p1 = node_pos[e["from_node"]], node_pos[e["to_node"]]
                dx, dy = p1["latent_x"] - p0["latent_x"], p1["latent_y"] - p0["latent_y"]
                score = safe_float(e.get("edge_score", 0.5))
                axB.arrow(
                    p0["latent_x"], p0["latent_y"], dx, dy,
                    length_includes_head=True,
                    head_width=0.035,
                    alpha=0.25 + 0.55 * score,
                    linewidth=0.5 + 1.5 * score,
                    color="#333333",
                    zorder=2,
                )

        axB.scatter(nodes["latent_x"], nodes["latent_y"], s=70, color="white", edgecolor="black", linewidth=0.8, zorder=4)
        if annotate:
            for _, r in nodes.iterrows():
                axB.text(r["latent_x"], r["latent_y"], r["node_id"], fontsize=6, ha="center", va="center", zorder=5)
            axB.set_title("B | Latent-space node connectivity", loc="left", fontsize=12, fontweight="bold")
            axB.set_xlabel("Latent coordinate 1")
            axB.set_ylabel("Latent coordinate 2")
            axB.legend(frameon=False, fontsize=7, loc="best")
        else:
            strip_all_text(axB)

        # C: top tracks
        axC = fig.add_subplot(gs[1, 0])
        if tracks is not None and not tracks.empty:
            top = tracks.sort_values("track_rank").head(12).copy()
            y = np.arange(len(top))[::-1]
            axC.barh(y, top["track_priority_score"].values[::-1])
            if annotate:
                labels = (top["track_id"].astype(str) + " | " + top["track_label"].astype(str)).values[::-1]
                axC.set_yticks(y)
                axC.set_yticklabels(labels, fontsize=7)
                axC.set_xlabel("Track priority score")
                axC.set_title("C | Reconstructed D1-D3-D7 tracks", loc="left", fontsize=12, fontweight="bold")
            else:
                strip_all_text(axC)
        else:
            if annotate:
                axC.text(0.5, 0.5, "No tracks reconstructed", ha="center", va="center")
            else:
                strip_all_text(axC)

        # D: repair-core shifts
        axD = fig.add_subplot(gs[1, 1])
        if tracks is not None and not tracks.empty:
            top = tracks.sort_values("track_rank").head(15).copy()
            x = top["delta_core_probability"]
            y = top["delta_repair_score"]
            sizes = 45 + 260 * minmax(top["mean_edge_score"])
            axD.scatter(x, y, s=sizes, alpha=0.82, edgecolor="black", linewidth=0.5)
            axD.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
            axD.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
            if annotate:
                for _, r in top.iterrows():
                    axD.text(r["delta_core_probability"], r["delta_repair_score"], r["track_id"].replace("Track_", "T"), fontsize=7)
                axD.set_xlabel("Δ core probability")
                axD.set_ylabel("Δ repair score")
                axD.set_title("D | Track repair-core shifts", loc="left", fontsize=12, fontweight="bold")
                axD.text(0.02, 0.98, "Upper-left quadrant:\nrepair gain + core reduction", transform=axD.transAxes, va="top", fontsize=8)
            else:
                strip_all_text(axD)
        else:
            if annotate:
                axD.text(0.5, 0.5, "No tracks reconstructed", ha="center", va="center")
            else:
                strip_all_text(axD)

        for ax in [axA, axB, axC, axD]:
            if annotate:
                ax.grid(linestyle="--", linewidth=0.35, alpha=0.25)
            else:
                ax.grid(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if not annotate:
                ax.spines["bottom"].set_visible(False)
                ax.spines["left"].set_visible(False)

        if annotate:
            fig.suptitle("Step 64b | StrokeNiche dynamics graph with fixed region probabilities", fontsize=15, fontweight="bold")

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

    ap.add_argument("--step64_script", default=str(DEFAULT_STEP64_SCRIPT))
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--coord_table", default=str(DEFAULT_COORD))
    ap.add_argument("--true_z", default=str(DEFAULT_TRUE_Z))
    ap.add_argument("--orig64_dir", default=str(DEFAULT_ORIG64))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--cluster_mode", default="kmeans", choices=["auto", "kmeans", "state_celltype"])
    ap.add_argument("--min_k", type=int, default=3)
    ap.add_argument("--max_k", type=int, default=7)
    ap.add_argument("--target_cells_per_cluster", type=int, default=450)
    ap.add_argument("--min_cluster_cells", type=int, default=30)

    ap.add_argument("--z_pca_dim", type=int, default=10)
    ap.add_argument("--distance_use_z_pca", action="store_true")
    ap.add_argument("--distance_use_modules", action="store_true")
    ap.add_argument("--distance_use_state_scores", action="store_true")

    ap.add_argument("--top_k_out", type=int, default=2)
    ap.add_argument("--top_k_in", type=int, default=2)
    ap.add_argument("--edge_score_quantile", type=float, default=0.0)
    ap.add_argument("--max_tracks", type=int, default=30)

    ap.add_argument("--w_kl", type=float, default=0.30)
    ap.add_argument("--w_wasserstein", type=float, default=0.25)
    ap.add_argument("--w_mahalanobis", type=float, default=0.15)
    ap.add_argument("--w_centroid", type=float, default=0.10)
    ap.add_argument("--w_module", type=float, default=0.20)

    ap.add_argument("--state_continuity_bonus", type=float, default=0.03)
    ap.add_argument("--celltype_continuity_bonus", type=float, default=0.02)

    ap.add_argument("--allow_neighbor_regionprob_fallback", action="store_true")
    ap.add_argument("--state_onehot_smoothing", type=float, default=0.02)

    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step 64b: fix region probabilities and regenerate StrokeNiche dynamics graph")
    log("=" * 100)
    log(f"outdir={outdir}")

    step64 = load_step64_module(args.step64_script)

    original_audit = audit_original_step64(args.orig64_dir)
    original_audit.to_csv(outdir / "step64b_original_probability_audit.csv", index=False)

    raw = read_csv(args.input, required=True)

    df, input_meta = step64.standardize_input(
        raw,
        coord_table=args.coord_table,
        true_z_table=args.true_z,
        z_pca_dim=args.z_pca_dim,
        seed=args.seed,
    )

    fixed_df, prob_audit, selected_meta = select_and_apply_region_probabilities(
        df,
        allow_neighbor_regionprob=args.allow_neighbor_regionprob_fallback,
        state_onehot_smoothing=args.state_onehot_smoothing,
    )

    prob_audit.to_csv(outdir / "step64b_probability_source_audit.csv", index=False)
    (outdir / "step64b_selected_probability_source.json").write_text(
        json.dumps(selected_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    fixed_df.to_csv(outdir / "step64b_modeling_input_used.fixed_region_probs.csv", index=False)

    modules = step64.module_cols(fixed_df)
    state_scores = [
        c for c in [
            "repair_score_std",
            "core_probability_std",
            "peri_probability_std",
            "remote_probability_std",
            "endothelial_context",
            "astrocyte_context",
            "microglia_context",
            "neuron_context",
            "lesion_niche_context",
            "remote_offtarget_context",
        ]
        if c in fixed_df.columns and pd.to_numeric(fixed_df[c], errors="coerce").notna().sum() > 0
    ]

    z_pca_cols = input_meta.get("strict_z_pca_cols", [])
    coord_cols = ["latent_x", "latent_y"]

    cluster_features = coord_cols + state_scores
    if modules:
        cluster_features += modules
    if z_pca_cols:
        cluster_features += z_pca_cols
    cluster_features = list(dict.fromkeys([c for c in cluster_features if c in fixed_df.columns]))

    clustered, cluster_meta = step64.cluster_cells(
        fixed_df,
        mode=args.cluster_mode,
        cluster_feature_cols=cluster_features,
        args=args,
    )

    clustered.to_csv(outdir / "step64b_clustered_cells.csv", index=False)
    cluster_meta.to_csv(outdir / "step64b_cluster_metadata.csv", index=False)

    distance_features = coord_cols.copy()
    distance_features += state_scores

    if args.distance_use_modules and modules:
        distance_features += modules
    if args.distance_use_z_pca and z_pca_cols:
        distance_features += z_pca_cols

    distance_features = list(dict.fromkeys([c for c in distance_features if c in clustered.columns]))

    nodes = step64.make_nodes(
        clustered=clustered,
        module_features=modules,
        distance_features=distance_features,
    )
    nodes.to_csv(outdir / "step64b_dynamics_graph_nodes.csv", index=False)

    all_edges = step64.compute_all_edges(
        clustered=clustered,
        nodes=nodes,
        distance_features=distance_features,
        module_features=modules,
        args=args,
    )
    all_edges.to_csv(outdir / "step64b_dynamics_graph_edges_all.csv", index=False)

    selected_edges = step64.select_edges(
        all_edges,
        top_k_out=args.top_k_out,
        top_k_in=args.top_k_in,
        edge_score_quantile=args.edge_score_quantile,
    )
    selected_edges.to_csv(outdir / "step64b_dynamics_graph_edges.csv", index=False)

    tracks = step64.reconstruct_tracks(
        nodes=nodes,
        selected_edges=selected_edges,
        max_tracks=args.max_tracks,
    )
    tracks.to_csv(outdir / "step64b_track_summary.csv", index=False)

    assignments = step64.assign_cells_to_tracks(clustered, tracks)
    assignments.to_csv(outdir / "step64b_track_assignment_by_cell.csv", index=False)

    track_module_matrix = step64.build_track_module_matrix(nodes, tracks)
    track_module_matrix.to_csv(outdir / "step64b_track_module_matrix.csv", index=False)

    # Post-fix audit on tracks and nodes.
    post_rows = []
    if not nodes.empty:
        cols = ["core_probability_mean", "peri_probability_mean", "remote_probability_mean"]
        x = nodes[cols].apply(pd.to_numeric, errors="coerce")
        same = (
            x[cols[0]].round(12).eq(x[cols[1]].round(12))
            & x[cols[0]].round(12).eq(x[cols[2]].round(12))
        )
        post_rows.append({
            "table": "nodes_after_fix",
            "n_rows": len(nodes),
            "all_three_identical_rows": int(same.sum()),
            "all_three_identical_fraction": float(same.mean()),
        })

    if tracks is not None and not tracks.empty:
        cols = ["delta_core_probability", "delta_peri_probability", "delta_remote_probability"]
        x = tracks[cols].apply(pd.to_numeric, errors="coerce")
        same = (
            x[cols[0]].round(12).eq(x[cols[1]].round(12))
            & x[cols[0]].round(12).eq(x[cols[2]].round(12))
        )
        post_rows.append({
            "table": "tracks_after_fix",
            "n_rows": len(tracks),
            "all_three_identical_rows": int(same.sum()),
            "all_three_identical_fraction": float(same.mean()),
        })

    post_fix_audit = pd.DataFrame(post_rows)
    post_fix_audit.to_csv(outdir / "step64b_postfix_probability_audit.csv", index=False)

    fig_outputs = plot_graph_versions(
        clustered=clustered,
        nodes=nodes,
        selected_edges=selected_edges,
        tracks=tracks,
        outdir=outdir,
        dpi=args.dpi,
        seed=args.seed,
    )

    feature_meta = {
        "input_meta": input_meta,
        "cluster_features": cluster_features,
        "distance_features": distance_features,
        "module_features": modules,
        "state_score_features": state_scores,
        "z_pca_cols": z_pca_cols,
        "selected_probability_source": selected_meta,
        "distance_weights": {
            "w_kl": args.w_kl,
            "w_wasserstein": args.w_wasserstein,
            "w_mahalanobis": args.w_mahalanobis,
            "w_centroid": args.w_centroid,
            "w_module": args.w_module,
            "state_continuity_bonus": args.state_continuity_bonus,
            "celltype_continuity_bonus": args.celltype_continuity_bonus,
        },
    }

    (outdir / "step64b_distance_feature_metadata.json").write_text(
        json.dumps(feature_meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    report = {
        "status": "ok",
        "analysis_name": "Step64b fixed-region-probability dynamics graph",
        "outdir": str(outdir),
        "original_step64_audit": original_audit.to_dict(orient="records") if not original_audit.empty else [],
        "selected_probability_source": selected_meta,
        "post_fix_probability_audit": post_fix_audit.to_dict(orient="records"),
        "n_cells": int(len(fixed_df)),
        "n_nodes": int(len(nodes)),
        "n_edges_all": int(len(all_edges)),
        "n_edges_selected": int(len(selected_edges)),
        "n_tracks": int(len(tracks)),
        "n_track_assignments": int(len(assignments)),
        "cluster_mode": args.cluster_mode,
        "figure_outputs": fig_outputs,
        "outputs": {
            "original_probability_audit": str(outdir / "step64b_original_probability_audit.csv"),
            "probability_source_audit": str(outdir / "step64b_probability_source_audit.csv"),
            "selected_probability_source": str(outdir / "step64b_selected_probability_source.json"),
            "modeling_input_used_fixed_probs": str(outdir / "step64b_modeling_input_used.fixed_region_probs.csv"),
            "clustered_cells": str(outdir / "step64b_clustered_cells.csv"),
            "nodes": str(outdir / "step64b_dynamics_graph_nodes.csv"),
            "edges_all": str(outdir / "step64b_dynamics_graph_edges_all.csv"),
            "edges_selected": str(outdir / "step64b_dynamics_graph_edges.csv"),
            "track_summary": str(outdir / "step64b_track_summary.csv"),
            "track_assignment_by_cell": str(outdir / "step64b_track_assignment_by_cell.csv"),
            "track_module_matrix": str(outdir / "step64b_track_module_matrix.csv"),
            "postfix_probability_audit": str(outdir / "step64b_postfix_probability_audit.csv"),
            "report_json": str(outdir / "step64b_report.json"),
            "report_txt": str(outdir / "step64b_report.txt"),
        },
        "interpretation_note": (
            "Use Step64b outputs instead of the original Step64 outputs when interpreting "
            "core/peri/remote probability shifts. If fallback_state_onehot was used, describe "
            "peri/remote shifts as state-label-derived, not model-predicted independent probabilities."
        ),
    }

    (outdir / "step64b_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step 64b fixed-region-probability dynamics graph report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Original probability audit:")
    lines.append(original_audit.to_string(index=False) if not original_audit.empty else "No original audit.")
    lines.append("")
    lines.append("Probability source audit:")
    lines.append(prob_audit.head(40).to_string(index=False) if not prob_audit.empty else "No candidates.")
    lines.append("")
    lines.append("Post-fix probability audit:")
    lines.append(post_fix_audit.to_string(index=False))
    lines.append("")
    lines.append("Top tracks:")
    lines.append(tracks.head(30).to_string(index=False) if tracks is not None and not tracks.empty else "No tracks reconstructed.")

    (outdir / "step64b_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step 64b")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Post-fix probability audit:")
    log(post_fix_audit.to_string(index=False))
    log("")
    log("Top tracks:")
    log(tracks.head(20).to_string(index=False) if tracks is not None and not tracks.empty else "No tracks reconstructed.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
