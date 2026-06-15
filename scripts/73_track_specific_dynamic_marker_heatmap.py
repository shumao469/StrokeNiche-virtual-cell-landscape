#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step73 | Track-specific dynamic marker heatmap

Purpose
-------
Make a manuscript-ready heatmap for track-specific dynamic markers/modules/regulators.

Inputs are expected from Step65 / Step65e:
  - dynamic genes table
  - dynamic modules table
  - regulator enrichment table
  - optional track summary / dynamics graph table

Figure design
-------------
Rows:
  progression tracks

Columns:
  top dynamic genes and top modules

Heatmap value:
  signed D1->D3->D7 slope, signed marker score, or reconstructed signed score
  depending on available Step65 columns.

Right annotation:
  top regulators for each track.

Outputs
-------
  Fig_Step73_TrackSpecificDynamicMarkerHeatmap_annotated.pdf/svg/png
  Fig_Step73_TrackSpecificDynamicMarkerHeatmap_clean_no_text.pdf/svg/png
  step73_track_dynamic_feature_matrix.csv
  step73_track_top_regulators.csv
  step73_feature_selection_summary.csv
  step73_report.json

Interpretation
--------------
This is a visualization/summary of transcript-level dynamic markers and inferred
regulator enrichment. It is not protein validation or functional validation.
"""

from pathlib import Path
import argparse
import json
import re
import math
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


HOUSEKEEPING = {
    "GAPDH", "ACTB", "B2M", "HPRT", "HPRT1", "PPIA", "RPLP0", "RPS18",
    "MALAT1", "RPL", "RPS"
}


DEFAULT_STROKE_MODULE_ORDER = [
    "hypoxia",
    "ferroptosis",
    "redox",
    "bbb",
    "barrier",
    "endothelial",
    "inflammation",
    "chemotaxis",
    "microglia",
    "astrocyte",
    "reactive",
    "repair",
    "ecm",
    "synaptic",
    "recovery",
]


def log(x):
    print(x, flush=True)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path, required=False):
    if not path:
        if required:
            raise FileNotFoundError("Empty path")
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(str(p))
        return pd.DataFrame()
    if p.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(p, sep="\t", low_memory=False)
    return pd.read_csv(p, low_memory=False)


def find_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return str(Path(p))
    return ""


def discover_inputs(base, step65_dir, step65e_dir, track_dir):
    base = Path(base) if base else None
    step65_dir = Path(step65_dir) if step65_dir else None
    step65e_dir = Path(step65e_dir) if step65e_dir else None
    track_dir = Path(track_dir) if track_dir else None

    candidates = {
        "genes": [],
        "modules": [],
        "regulators": [],
        "network": [],
        "track_summary": [],
    }

    dirs = [d for d in [step65e_dir, step65_dir, track_dir, base] if d is not None]

    for d in dirs:
        candidates["genes"] += [
            d / "step65e_dynamic_genes_cleaned_for_chipseq.csv",
            d / "step65e_dynamic_genes_cleaned.csv",
            d / "step65_track_dynamic_genes.csv",
            d / "step65_dynamic_genes.csv",
            d / "dynamic_genes.csv",
        ]
        candidates["modules"] += [
            d / "step65_track_dynamic_modules.csv",
            d / "step65_dynamic_modules.csv",
            d / "dynamic_modules.csv",
            d / "step65e_track_dynamic_modules.csv",
        ]
        candidates["regulators"] += [
            d / "step65e_regulator_enrichment_housekeeping_filtered.csv",
            d / "step65_regulator_enrichment.csv",
            d / "regulator_enrichment.csv",
            d / "step65e_regulator_enrichment.csv",
        ]
        candidates["network"] += [
            d / "step65_track_network_edges.csv",
            d / "step65e_track_network_edges.csv",
            d / "track_network_edges.csv",
        ]
        candidates["track_summary"] += [
            d / "step64c_track_summary.csv",
            d / "step64b_track_summary.csv",
            d / "track_summary.csv",
            d / "step64b_dynamics_graph_nodes.csv",
            d / "step64c_dynamics_graph_nodes.refined_labels.csv",
            d / "step64c_dynamics_graph_nodes.csv",
        ]

    found = {k: find_first_existing(v) for k, v in candidates.items()}
    return found


def first_col(df, candidates, contains=None):
    if df is None or df.empty:
        return ""
    lower_map = {str(c).lower(): c for c in df.columns}

    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if contains:
        for c in df.columns:
            lc = str(c).lower()
            if any(x.lower() in lc for x in contains):
                return c

    return ""


def clean_str(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def clean_gene_symbol(g):
    g = clean_str(g)
    if not g:
        return ""
    # for strings like "GeneA;GeneB", keep first token for gene-level table
    g = re.split(r"[;,|/\s]+", g)[0]
    g = re.sub(r"[^A-Za-z0-9_.-]+", "", g)
    return g


def is_housekeeping_gene(g):
    gu = clean_gene_symbol(g).upper()
    if not gu:
        return True
    if gu in HOUSEKEEPING:
        return True
    if gu.startswith("RPL") or gu.startswith("RPS"):
        return True
    return False


def normalize_track_id(x):
    s = clean_str(x)
    if not s:
        return ""
    # keep Track 001 style if present
    m = re.search(r"(\d+)", s)
    if s.lower().startswith("track") and m:
        return f"Track {int(m.group(1)):03d}"
    if m and len(s) <= 4:
        return f"Track {int(m.group(1)):03d}"
    return s


def infer_track_col(df):
    return first_col(
        df,
        ["track", "track_id", "progression_track", "trajectory", "trajectory_id",
         "path", "path_id", "module_track"],
        contains=["track", "trajectory", "path"]
    )


def infer_gene_col(df):
    return first_col(
        df,
        ["gene", "genes", "gene_symbol", "symbol", "dynamic_gene", "marker", "feature"],
        contains=["gene", "symbol", "marker", "feature"]
    )


def infer_module_col(df):
    return first_col(
        df,
        ["module", "module_key", "module_name", "pathway", "program", "signature", "feature"],
        contains=["module", "pathway", "program", "signature"]
    )


def infer_regulator_col(df):
    return first_col(
        df,
        ["regulator", "tf", "TF", "transcription_factor", "source", "gene", "term"],
        contains=["regulator", "tf", "transcription", "source", "term"]
    )


def infer_score_col(df):
    candidates = [
        "signed_score", "signed_marker_score", "marker_score", "score",
        "dynamic_score", "slope", "trend_slope", "d1_d7_slope", "D1_D7_slope",
        "logFC", "avg_log2FC", "effect_size", "NES", "nes",
        "enrichment_score", "importance", "z", "z_score"
    ]
    col = first_col(df, candidates)
    if col:
        return col

    numeric_cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            lc = str(c).lower()
            if any(k in lc for k in ["score", "slope", "logfc", "effect", "nes", "importance", "z"]):
                numeric_cols.append(c)
    return numeric_cols[0] if numeric_cols else ""


def infer_p_col(df):
    return first_col(
        df,
        ["p_value", "pval", "p", "adjusted_p", "padj", "q_value", "fdr", "bh_fdr"],
        contains=["p_value", "pval", "padj", "fdr", "q_value"]
    )


def infer_direction_col(df):
    return first_col(
        df,
        ["direction", "trend", "pattern", "dynamic_pattern", "class", "category"],
        contains=["direction", "trend", "pattern", "class", "category"]
    )


def signed_from_direction(direction):
    s = clean_str(direction).lower()
    if not s:
        return 1.0
    if any(k in s for k in ["decreasing", "down", "negative", "reduced", "lower"]):
        return -1.0
    if any(k in s for k in ["increasing", "up", "positive", "induced", "higher"]):
        return 1.0
    if any(k in s for k in ["u-shape", "u_shape", "transient", "peak"]):
        return 1.0
    return 1.0


def compute_d1d7_slope_if_possible(row, cols):
    time_cols = {}
    for tp in ["D1", "D3", "D7", "d1", "d3", "d7"]:
        candidates = [c for c in cols if str(c).lower() in [
            tp.lower(), f"mean_{tp}".lower(), f"{tp}_mean".lower(),
            f"expr_{tp}".lower(), f"{tp}_expr".lower(),
            f"score_{tp}".lower(), f"{tp}_score".lower()
        ]]
        if candidates:
            time_cols[tp.upper()] = candidates[0]

    if all(k in time_cols for k in ["D1", "D3", "D7"]):
        vals = []
        for k in ["D1", "D3", "D7"]:
            try:
                vals.append(float(row[time_cols[k]]))
            except Exception:
                return np.nan
        x = np.array([1, 3, 7], dtype=float)
        y = np.array(vals, dtype=float)
        if np.any(~np.isfinite(y)):
            return np.nan
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    return np.nan


def robust_z_by_feature(df, value_col="raw_value"):
    out = df.copy()
    vals = pd.to_numeric(out[value_col], errors="coerce").astype(float)
    med = np.nanmedian(vals)
    mad = np.nanmedian(np.abs(vals - med))
    if not np.isfinite(mad) or mad < 1e-12:
        sd = np.nanstd(vals)
        if not np.isfinite(sd) or sd < 1e-12:
            out["z_value"] = vals.fillna(0.0)
            return out
        out["z_value"] = (vals - med) / sd
        return out
    out["z_value"] = 0.6745 * (vals - med) / mad
    out["z_value"] = out["z_value"].clip(-4, 4)
    return out


def prepare_gene_features(gene_df, top_genes_per_track=8, exclude_housekeeping=True):
    if gene_df.empty:
        return pd.DataFrame(), {}

    df = gene_df.copy()

    track_col = infer_track_col(df)
    gene_col = infer_gene_col(df)
    score_col = infer_score_col(df)
    p_col = infer_p_col(df)
    direction_col = infer_direction_col(df)

    audit = {
        "track_col": track_col,
        "gene_col": gene_col,
        "score_col": score_col,
        "p_col": p_col,
        "direction_col": direction_col,
        "n_input_rows": int(len(df)),
    }

    if not track_col or not gene_col:
        audit["status"] = "failed_missing_track_or_gene_col"
        return pd.DataFrame(), audit

    df["track_id"] = df[track_col].map(normalize_track_id)
    df["feature_name"] = df[gene_col].map(clean_gene_symbol)
    df["feature_type"] = "gene"

    df = df[(df["track_id"] != "") & (df["feature_name"] != "")]
    if exclude_housekeeping:
        df = df[~df["feature_name"].map(is_housekeeping_gene)].copy()

    raw_values = []
    for _, row in df.iterrows():
        val = np.nan
        if score_col:
            try:
                val = float(row[score_col])
            except Exception:
                val = np.nan
        if not np.isfinite(val):
            val = compute_d1d7_slope_if_possible(row, df.columns)
        if not np.isfinite(val):
            val = 1.0

        if direction_col:
            val = abs(val) * signed_from_direction(row[direction_col])

        raw_values.append(val)

    df["raw_value"] = raw_values

    if p_col:
        p = pd.to_numeric(df[p_col], errors="coerce")
        # Higher selection score for smaller p and larger effect.
        df["selection_score"] = np.abs(df["raw_value"]) * (-np.log10(p.clip(lower=1e-300))).replace([np.inf, -np.inf], np.nan).fillna(1)
    else:
        df["selection_score"] = np.abs(df["raw_value"])

    df = df.sort_values(["track_id", "selection_score"], ascending=[True, False])
    df = df.groupby("track_id", as_index=False, group_keys=False).head(top_genes_per_track)

    df = robust_z_by_feature(df, value_col="raw_value")

    keep_cols = [
        "track_id", "feature_name", "feature_type", "raw_value", "z_value",
        "selection_score"
    ]
    for c in [direction_col, p_col, score_col]:
        if c and c not in keep_cols and c in df.columns:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    audit["n_output_rows"] = int(len(out))
    audit["n_tracks"] = int(out["track_id"].nunique()) if not out.empty else 0
    audit["status"] = "ok"
    return out, audit


def module_display_name(x):
    s = clean_str(x)
    if not s:
        return ""
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > 24:
        s = s[:22] + "…"
    return s


def prepare_module_features(module_df, top_modules_per_track=5):
    if module_df.empty:
        return pd.DataFrame(), {"status": "empty"}

    df = module_df.copy()

    track_col = infer_track_col(df)
    module_col = infer_module_col(df)
    score_col = infer_score_col(df)
    p_col = infer_p_col(df)
    direction_col = infer_direction_col(df)

    audit = {
        "track_col": track_col,
        "module_col": module_col,
        "score_col": score_col,
        "p_col": p_col,
        "direction_col": direction_col,
        "n_input_rows": int(len(df)),
    }

    if not track_col or not module_col:
        audit["status"] = "failed_missing_track_or_module_col"
        return pd.DataFrame(), audit

    df["track_id"] = df[track_col].map(normalize_track_id)
    df["feature_name"] = df[module_col].map(module_display_name)
    df["feature_type"] = "module"

    df = df[(df["track_id"] != "") & (df["feature_name"] != "")].copy()

    raw_values = []
    for _, row in df.iterrows():
        val = np.nan
        if score_col:
            try:
                val = float(row[score_col])
            except Exception:
                val = np.nan
        if not np.isfinite(val):
            val = compute_d1d7_slope_if_possible(row, df.columns)
        if not np.isfinite(val):
            val = 1.0

        if direction_col:
            val = abs(val) * signed_from_direction(row[direction_col])
        raw_values.append(val)

    df["raw_value"] = raw_values

    if p_col:
        p = pd.to_numeric(df[p_col], errors="coerce")
        df["selection_score"] = np.abs(df["raw_value"]) * (-np.log10(p.clip(lower=1e-300))).replace([np.inf, -np.inf], np.nan).fillna(1)
    else:
        df["selection_score"] = np.abs(df["raw_value"])

    df = df.sort_values(["track_id", "selection_score"], ascending=[True, False])
    df = df.groupby("track_id", as_index=False, group_keys=False).head(top_modules_per_track)

    df = robust_z_by_feature(df, value_col="raw_value")

    keep_cols = [
        "track_id", "feature_name", "feature_type", "raw_value", "z_value",
        "selection_score"
    ]
    for c in [direction_col, p_col, score_col]:
        if c and c not in keep_cols and c in df.columns:
            keep_cols.append(c)

    out = df[keep_cols].copy()
    audit["n_output_rows"] = int(len(out))
    audit["n_tracks"] = int(out["track_id"].nunique()) if not out.empty else 0
    audit["status"] = "ok"
    return out, audit


def prepare_regulators(reg_df, top_regulators_per_track=3):
    if reg_df.empty:
        return pd.DataFrame(), {"status": "empty"}

    df = reg_df.copy()

    track_col = infer_track_col(df)
    reg_col = infer_regulator_col(df)
    score_col = infer_score_col(df)
    p_col = infer_p_col(df)

    audit = {
        "track_col": track_col,
        "regulator_col": reg_col,
        "score_col": score_col,
        "p_col": p_col,
        "n_input_rows": int(len(df)),
    }

    if not track_col or not reg_col:
        audit["status"] = "failed_missing_track_or_regulator_col"
        return pd.DataFrame(), audit

    df["track_id"] = df[track_col].map(normalize_track_id)
    df["regulator"] = df[reg_col].map(lambda x: clean_gene_symbol(x).upper() if clean_str(x) else "")

    df = df[(df["track_id"] != "") & (df["regulator"] != "")].copy()
    df = df[~df["regulator"].map(is_housekeeping_gene)].copy()

    if score_col:
        df["regulator_score"] = pd.to_numeric(df[score_col], errors="coerce").abs()
    elif p_col:
        p = pd.to_numeric(df[p_col], errors="coerce")
        df["regulator_score"] = -np.log10(p.clip(lower=1e-300))
    else:
        df["regulator_score"] = 1.0

    df["regulator_score"] = df["regulator_score"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    df = df.sort_values(["track_id", "regulator_score"], ascending=[True, False])
    df = df.groupby("track_id", as_index=False, group_keys=False).head(top_regulators_per_track)

    top_text = (
        df.groupby("track_id")["regulator"]
        .apply(lambda x: ", ".join(list(dict.fromkeys(x.astype(str).tolist()))))
        .reset_index()
        .rename(columns={"regulator": "top_regulators"})
    )

    audit["n_output_rows"] = int(len(df))
    audit["n_tracks"] = int(df["track_id"].nunique()) if not df.empty else 0
    audit["status"] = "ok"

    return top_text, audit


def load_track_order(track_summary_df, existing_tracks):
    if track_summary_df.empty:
        return sorted(existing_tracks, key=track_sort_key), {}

    df = track_summary_df.copy()
    track_col = infer_track_col(df)
    if not track_col:
        # maybe cluster/node table has track_id absent
        return sorted(existing_tracks, key=track_sort_key), {"status": "no_track_col"}

    df["track_id"] = df[track_col].map(normalize_track_id)
    tracks = [t for t in df["track_id"].dropna().astype(str).tolist() if t in existing_tracks]
    tracks = list(dict.fromkeys(tracks))

    for t in sorted(existing_tracks, key=track_sort_key):
        if t not in tracks:
            tracks.append(t)

    return tracks, {"status": "ok", "track_col": track_col, "n_tracks": len(tracks)}


def track_sort_key(x):
    s = str(x)
    m = re.search(r"(\d+)", s)
    if m:
        return (0, int(m.group(1)), s)
    return (1, 999999, s)


def feature_sort_key(f):
    s = str(f).lower()
    for i, k in enumerate(DEFAULT_STROKE_MODULE_ORDER):
        if k in s:
            return (0, i, s)
    return (1, 999, s)


def select_global_features(feature_df, top_total_genes=40, top_total_modules=20):
    if feature_df.empty:
        return []

    genes = feature_df[feature_df["feature_type"] == "gene"].copy()
    modules = feature_df[feature_df["feature_type"] == "module"].copy()

    selected = []

    if not genes.empty:
        gene_rank = (
            genes.groupby("feature_name")["selection_score"]
            .max()
            .sort_values(ascending=False)
            .head(top_total_genes)
            .index.tolist()
        )
        selected += gene_rank

    if not modules.empty:
        mod_rank = (
            modules.groupby("feature_name")["selection_score"]
            .max()
            .sort_values(ascending=False)
            .head(top_total_modules)
            .index.tolist()
        )
        mod_rank = sorted(mod_rank, key=feature_sort_key)
        selected += mod_rank

    selected = list(dict.fromkeys(selected))
    return selected


def build_matrix(feature_df, tracks, selected_features):
    mat = pd.DataFrame(0.0, index=tracks, columns=selected_features)

    raw_mat = pd.DataFrame(np.nan, index=tracks, columns=selected_features)
    type_map = {}

    for _, r in feature_df.iterrows():
        t = r["track_id"]
        f = r["feature_name"]
        if t in mat.index and f in mat.columns:
            # If multiple entries exist, keep the one with larger absolute z.
            z = float(r.get("z_value", 0))
            old = mat.loc[t, f]
            if abs(z) >= abs(old):
                mat.loc[t, f] = z
                raw_mat.loc[t, f] = float(r.get("raw_value", np.nan))
                type_map[f] = r.get("feature_type", "")

    return mat, raw_mat, type_map


def shorten_feature_name(f, max_len=18):
    s = str(f)
    if len(s) <= max_len:
        return s
    return s[:max_len-1] + "…"


def make_track_labels(tracks, regulator_table):
    reg_map = {}
    if not regulator_table.empty and "track_id" in regulator_table.columns:
        reg_map = dict(zip(regulator_table["track_id"], regulator_table["top_regulators"]))

    labels = []
    for t in tracks:
        regs = reg_map.get(t, "")
        if regs:
            labels.append(f"{t}")
        else:
            labels.append(f"{t}")
    return labels, reg_map


def make_feature_type_colors(selected_features, type_map):
    colors = []
    for f in selected_features:
        typ = type_map.get(f, "")
        if typ == "module":
            colors.append("#334155")
        else:
            colors.append("#94a3b8")
    return colors


def make_figure(
    mat,
    raw_mat,
    regulator_table,
    outbase,
    title="Track-specific dynamic marker / module / regulator map",
    clean=False,
    dpi=600,
    max_reg_text=42,
):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    tracks = mat.index.tolist()
    features = mat.columns.tolist()
    data = mat.to_numpy(dtype=float)

    if data.size == 0:
        raise RuntimeError("Empty heatmap matrix.")

    vmax = np.nanquantile(np.abs(data), 0.95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    n_rows, n_cols = data.shape
    fig_w = max(12.5, min(28, 0.36 * n_cols + 6.5))
    fig_h = max(6.5, min(20, 0.34 * n_rows + 3.2))

    if clean:
        fig_w = max(10, min(24, 0.30 * n_cols + 3.2))
        fig_h = max(5, min(18, 0.30 * n_rows + 2.0))

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    if clean:
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax_reg = None
        ax_top = None
    else:
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[0.35, 9],
            width_ratios=[max(8, 0.36 * n_cols), 3.2],
            hspace=0.03,
            wspace=0.05,
        )
        ax_top = fig.add_subplot(gs[0, 0])
        ax = fig.add_subplot(gs[1, 0])
        ax_reg = fig.add_subplot(gs[1, 1])

    im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")

    if clean:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([shorten_feature_name(f, 16) for f in features],
                           rotation=60, ha="right", fontsize=7)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels(tracks, fontsize=8)

        ax.set_xlabel("Top dynamic genes and modules", fontsize=9)
        ax.set_ylabel("Progression tracks", fontsize=9)

        # vertical line between genes and modules if possible
        # modules tend to be darker top bars
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color="white", lw=0.25, alpha=0.4)
        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color="white", lw=0.25, alpha=0.4)

        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("signed dynamic score / slope z", fontsize=8)

    for sp in ax.spines.values():
        sp.set_visible(False)

    if not clean and ax_top is not None:
        # top annotation: gene vs module
        types = []
        for f in features:
            if " " in f.lower() or any(k in f.lower() for k in DEFAULT_STROKE_MODULE_ORDER):
                # not perfect; visual only
                types.append("module" if len(f.split()) > 1 else "gene")
            else:
                types.append("gene")

        # Better infer module by lower names containing known module words.
        vals = np.array([[1 if any(k in str(f).lower() for k in DEFAULT_STROKE_MODULE_ORDER) else 0 for f in features]])
        ax_top.imshow(vals, cmap=matplotlib.colors.ListedColormap(["#cbd5e1", "#334155"]), aspect="auto")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_xlim(-0.5, n_cols - 0.5)
        ax_top.set_title(title, fontsize=13, fontweight="bold", pad=8)
        ax_top.text(0.0, 1.30, "feature class: gene / module",
                    transform=ax_top.transAxes, ha="left", va="bottom", fontsize=8, color="#475569")
        for sp in ax_top.spines.values():
            sp.set_visible(False)

    if not clean and ax_reg is not None:
        regulator_table = regulator_table.copy()
        reg_map = {}
        if not regulator_table.empty and "track_id" in regulator_table.columns:
            reg_map = dict(zip(regulator_table["track_id"], regulator_table["top_regulators"]))

        ax_reg.set_xlim(0, 1)
        ax_reg.set_ylim(-0.5, n_rows - 0.5)
        ax_reg.invert_yaxis()
        ax_reg.axis("off")
        ax_reg.set_title("Top regulators", fontsize=10, fontweight="bold", pad=8)

        for i, t in enumerate(tracks):
            regs = str(reg_map.get(t, ""))
            if len(regs) > max_reg_text:
                regs = regs[:max_reg_text - 1] + "…"

            # subtle row block
            if i % 2 == 0:
                ax_reg.add_patch(Rectangle((0, i - 0.5), 1, 1,
                                           facecolor="#f8fafc", edgecolor="none", zorder=0))
            ax_reg.text(0.02, i, regs if regs else "—",
                        ha="left", va="center", fontsize=8, color="#111827")

        ax_reg.text(
            0.02,
            n_rows - 0.15,
            "",
            ha="left",
            va="bottom",
            fontsize=7,
        )

    if not clean:
        fig.text(
            0.5,
            0.01,
            "Tracks are ordered from the dynamics graph. Heatmap values summarize signed D1-D3-D7 dynamic marker/module scores; regulators are enrichment-based and not functional validation.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#475569",
        )

    outbase = Path(outbase)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base", default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap")
    ap.add_argument("--step65_dir", default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata")
    ap.add_argument("--step65e_dir", default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered")
    ap.add_argument("--track_dir", default="/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--gene_table", default="")
    ap.add_argument("--module_table", default="")
    ap.add_argument("--regulator_table", default="")
    ap.add_argument("--track_summary", default="")

    ap.add_argument("--top_genes_per_track", type=int, default=8)
    ap.add_argument("--top_modules_per_track", type=int, default=5)
    ap.add_argument("--top_total_genes", type=int, default=45)
    ap.add_argument("--top_total_modules", type=int, default=18)
    ap.add_argument("--top_regulators_per_track", type=int, default=4)

    ap.add_argument("--exclude_housekeeping", action="store_true")
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    log("=" * 100)
    log("Step73 | Track-specific dynamic marker heatmap")
    log("=" * 100)
    log(f"outdir={outdir}")

    discovered = discover_inputs(args.base, args.step65_dir, args.step65e_dir, args.track_dir)

    gene_path = args.gene_table or discovered["genes"]
    module_path = args.module_table or discovered["modules"]
    regulator_path = args.regulator_table or discovered["regulators"]
    track_summary_path = args.track_summary or discovered["track_summary"]

    source_paths = {
        "gene_table": gene_path,
        "module_table": module_path,
        "regulator_table": regulator_path,
        "track_summary": track_summary_path,
        "discovered": discovered,
    }

    log("---- source paths ----")
    log(json.dumps(source_paths, indent=2, ensure_ascii=False))

    gene_df = read_table(gene_path) if gene_path else pd.DataFrame()
    module_df = read_table(module_path) if module_path else pd.DataFrame()
    reg_df = read_table(regulator_path) if regulator_path else pd.DataFrame()
    track_summary_df = read_table(track_summary_path) if track_summary_path else pd.DataFrame()

    gene_features, gene_audit = prepare_gene_features(
        gene_df,
        top_genes_per_track=args.top_genes_per_track,
        exclude_housekeeping=args.exclude_housekeeping
    )

    module_features, module_audit = prepare_module_features(
        module_df,
        top_modules_per_track=args.top_modules_per_track
    )

    regulator_table, regulator_audit = prepare_regulators(
        reg_df,
        top_regulators_per_track=args.top_regulators_per_track
    )

    all_features = pd.concat(
        [x for x in [gene_features, module_features] if not x.empty],
        ignore_index=True
    ) if (not gene_features.empty or not module_features.empty) else pd.DataFrame()

    if all_features.empty:
        fail_report = {
            "status": "failed_no_features",
            "source_paths": source_paths,
            "gene_audit": gene_audit,
            "module_audit": module_audit,
            "regulator_audit": regulator_audit,
            "message": "No dynamic gene/module features could be parsed. Please inspect Step65 output columns."
        }
        (outdir / "step73_failed_audit.json").write_text(
            json.dumps(fail_report, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        log(json.dumps(fail_report, indent=2, ensure_ascii=False))
        raise RuntimeError(fail_report["message"])

    selected_features = select_global_features(
        all_features,
        top_total_genes=args.top_total_genes,
        top_total_modules=args.top_total_modules
    )

    existing_tracks = set(all_features["track_id"].dropna().astype(str).tolist())
    tracks, track_order_audit = load_track_order(track_summary_df, existing_tracks)

    mat, raw_mat, type_map = build_matrix(all_features, tracks, selected_features)

    # Drop empty tracks/features after building matrix.
    mat = mat.loc[(mat.abs().sum(axis=1) > 0), :]
    mat = mat.loc[:, (mat.abs().sum(axis=0) > 0)]
    raw_mat = raw_mat.loc[mat.index, mat.columns]

    # Reorder tracks naturally.
    mat = mat.loc[sorted(mat.index.tolist(), key=track_sort_key), :]
    raw_mat = raw_mat.loc[mat.index, mat.columns]

    # Reorder features: genes by maximum score, modules by stroke order.
    gene_cols = [c for c in mat.columns if type_map.get(c, "gene") == "gene"]
    module_cols = [c for c in mat.columns if type_map.get(c, "gene") == "module"]

    gene_cols = sorted(gene_cols, key=lambda c: -float(mat[c].abs().max()))
    module_cols = sorted(module_cols, key=feature_sort_key)
    ordered_cols = gene_cols + module_cols
    mat = mat[ordered_cols]
    raw_mat = raw_mat[ordered_cols]

    feature_matrix_path = outdir / "step73_track_dynamic_feature_matrix.csv"
    raw_matrix_path = outdir / "step73_track_dynamic_feature_raw_matrix.csv"
    all_features_path = outdir / "step73_selected_dynamic_features_long.csv"
    regulator_path_out = outdir / "step73_track_top_regulators.csv"
    feature_selection_path = outdir / "step73_feature_selection_summary.csv"

    mat.to_csv(feature_matrix_path)
    raw_mat.to_csv(raw_matrix_path)
    all_features.to_csv(all_features_path, index=False)
    regulator_table.to_csv(regulator_path_out, index=False)

    feature_selection = pd.DataFrame({
        "feature_name": mat.columns,
        "feature_type": [type_map.get(f, "gene") for f in mat.columns],
        "max_abs_z": [float(mat[f].abs().max()) for f in mat.columns],
        "n_tracks_nonzero": [(mat[f].abs() > 0).sum() for f in mat.columns],
    })
    feature_selection.to_csv(feature_selection_path, index=False)

    annotated_base = outdir / "Fig_Step73_TrackSpecificDynamicMarkerHeatmap_annotated"
    clean_base = outdir / "Fig_Step73_TrackSpecificDynamicMarkerHeatmap_clean_no_text"

    make_figure(
        mat,
        raw_mat,
        regulator_table,
        annotated_base,
        title="Track-specific dynamic markers, modules and regulators",
        clean=False,
        dpi=args.dpi
    )

    make_figure(
        mat,
        raw_mat,
        regulator_table,
        clean_base,
        title="",
        clean=True,
        dpi=args.dpi
    )

    report = {
        "status": "ok",
        "analysis_name": "Step73 track-specific dynamic marker heatmap",
        "source_paths": source_paths,
        "gene_audit": gene_audit,
        "module_audit": module_audit,
        "regulator_audit": regulator_audit,
        "track_order_audit": track_order_audit,
        "n_tracks": int(mat.shape[0]),
        "n_features": int(mat.shape[1]),
        "n_gene_features": int(len(gene_cols)),
        "n_module_features": int(len(module_cols)),
        "outputs": {
            "feature_matrix_z": str(feature_matrix_path),
            "feature_matrix_raw": str(raw_matrix_path),
            "selected_features_long": str(all_features_path),
            "top_regulators": str(regulator_path_out),
            "feature_selection_summary": str(feature_selection_path),
            "annotated_pdf": str(annotated_base.with_suffix(".pdf")),
            "annotated_svg": str(annotated_base.with_suffix(".svg")),
            "annotated_png": str(annotated_base.with_suffix(".png")),
            "clean_pdf": str(clean_base.with_suffix(".pdf")),
            "clean_svg": str(clean_base.with_suffix(".svg")),
            "clean_png": str(clean_base.with_suffix(".png")),
        },
        "interpretation_note": (
            "Step73 visualizes transcript-level track-specific dynamic genes/modules and enrichment-derived regulators. "
            "It should be interpreted as dynamic marker/regulator prioritization, not functional or protein-level validation."
        )
    }

    (outdir / "step73_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    (outdir / "step73_report.txt").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    log("=" * 100)
    log("DONE Step73")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
