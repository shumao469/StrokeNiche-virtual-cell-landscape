#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step80 | Candidate perturbation-axis expression dotplot

Purpose
-------
Generate a manuscript-style dotplot showing expression support for candidate
virtual perturbation axes across StrokeNiche states and timepoints.

Rows:
  Ccl2, Ccr2, Ackr1
  Spp1, Cd44
  Vegfa, Flt1, Kdr
  Hmox1, Fth1, Slc7a11, Gpx4
  Col1a1, Col3a1, Fn1, Spp1

Columns:
  state_group: lesion-core-like / peri-infarct / remote-like
  timepoint: D1 / D3 / D7

Color:
  robust min-max scaled mean expression per gene.

Dot size:
  percentage of spots/cells with expression > threshold.

Interpretation
--------------
This figure supports expression availability of candidate perturbation axes.
It does not prove ligand-receptor directionality or wet-lab perturbation effect.
"""

import os
import json
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import anndata as ad
except Exception as e:
    raise ImportError("Please install anndata: pip install anndata") from e

try:
    from scipy import sparse
except Exception:
    sparse = None


# -----------------------------
# Candidate gene rows
# -----------------------------

ROW_DEFS = [
    ("Ccl2/Ccr2-Ackr1", "Ccl2", "Ccl2"),
    ("Ccl2/Ccr2-Ackr1", "Ccr2", "Ccr2"),
    ("Ccl2/Ccr2-Ackr1", "Ackr1", "Ackr1"),

    ("Spp1-Cd44", "Spp1", "Spp1"),
    ("Spp1-Cd44", "Cd44", "Cd44"),

    ("Vegfa-Flt1/Kdr", "Vegfa", "Vegfa"),
    ("Vegfa-Flt1/Kdr", "Flt1", "Flt1"),
    ("Vegfa-Flt1/Kdr", "Kdr", "Kdr"),

    ("Ferroptosis / hypoxia", "Hmox1", "Hmox1"),
    ("Ferroptosis / hypoxia", "Fth1", "Fth1"),
    ("Ferroptosis / hypoxia", "Slc7a11", "Slc7a11"),
    ("Ferroptosis / hypoxia", "Gpx4", "Gpx4"),

    ("Repair / ECM", "Col1a1", "Col1a1"),
    ("Repair / ECM", "Col3a1", "Col3a1"),
    ("Repair / ECM", "Fn1", "Fn1"),
    ("Repair / ECM", "Spp1", "Spp1 (ECM)"),
]


STATE_ORDER = [
    "lesion-core-like",
    "core-like",
    "core",
    "peri-infarct",
    "peri",
    "remote-like",
    "remote",
]

TIME_ORDER = ["D1", "D3", "D7"]


# -----------------------------
# Utilities
# -----------------------------

def mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_label(x: str) -> str:
    s = clean_str(x).lower()
    s = s.replace("_", "-").replace(" ", "-")
    if "core" in s and "lesion" not in s:
        return "lesion-core-like" if s == "core" else s
    if s in {"peri", "peri-infarct", "periinfarct"}:
        return "peri-infarct"
    if s in {"remote", "remote-like"}:
        return "remote-like"
    return clean_str(x)


def order_categories(values: List[str], preferred: List[str]) -> List[str]:
    values = [clean_str(v) for v in values if clean_str(v) != ""]
    unique = list(dict.fromkeys(values))
    low_to_orig = {v.lower(): v for v in unique}

    ordered = []
    for p in preferred:
        if p.lower() in low_to_orig:
            ordered.append(low_to_orig[p.lower()])

    for v in unique:
        if v not in ordered:
            ordered.append(v)

    return ordered


def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def read_state_table(path: str) -> pd.DataFrame:
    if path is None or path == "":
        return pd.DataFrame()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if path.endswith(".gz"):
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def find_merge_key(adata, state: pd.DataFrame) -> Tuple[str, Optional[str], str]:
    """
    Return:
      adata_key_mode, state_key, report
    adata_key_mode:
      "_obs_names_" means use adata.obs_names.
    """
    if state is None or state.empty:
        return "_obs_names_", None, "no_state_table"

    obs_names = pd.Index(adata.obs_names.astype(str))
    candidates = [
        "obs_name", "barcode", "spot", "spot_id", "cell_id", "id",
        "_obs_names_", "adata_obs_name", "index"
    ]

    best_col = None
    best_overlap = -1

    for c in candidates:
        if c in state.columns:
            vals = state[c].astype(str)
            overlap = vals.isin(obs_names).sum()
            if overlap > best_overlap:
                best_col = c
                best_overlap = overlap

    # also test all object columns when common candidates fail
    if best_overlap <= 0:
        for c in state.columns:
            if state[c].dtype == object or str(state[c].dtype).startswith("string"):
                vals = state[c].astype(str)
                overlap = vals.isin(obs_names).sum()
                if overlap > best_overlap:
                    best_col = c
                    best_overlap = overlap

    if best_col is not None and best_overlap > 0:
        return "_obs_names_", best_col, f"id_merge_overlap={best_overlap}"

    # If same row count, allow row-order merge
    if len(state) == adata.n_obs:
        return "_row_order_", None, "row_order_merge"

    raise ValueError(
        "Cannot match state_table to adata.obs_names and row counts differ. "
        "Please provide a state table containing obs_name/barcode/cell_id."
    )


def merge_metadata(adata, state: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    obs = adata.obs.copy()
    obs["_adata_id_"] = adata.obs_names.astype(str)

    audit = {
        "n_obs": int(adata.n_obs),
        "state_table_available": bool(state is not None and not state.empty),
    }

    if state is None or state.empty:
        audit["merge_mode"] = "obs_only"
        return obs.reset_index(drop=True), audit

    mode, state_key, report = find_merge_key(adata, state)
    audit["merge_mode_detail"] = report

    if mode == "_obs_names_":
        st = state.copy()
        st["_adata_id_"] = st[state_key].astype(str)
        merged = obs.merge(st, on="_adata_id_", how="left", suffixes=("", "_state"))
        audit["merge_mode"] = "id"
        audit["merge_key_state"] = state_key
        audit["state_rows"] = int(len(state))
        audit["state_merge_overlap"] = int(merged[state_key].notna().sum()) if state_key in merged.columns else int(merged.filter(regex="_state$").notna().any(axis=1).sum())
        return merged.reset_index(drop=True), audit

    if mode == "_row_order_":
        merged = pd.concat([obs.reset_index(drop=True), state.reset_index(drop=True)], axis=1)
        audit["merge_mode"] = "row_order"
        audit["state_rows"] = int(len(state))
        audit["state_merge_overlap"] = int(len(state))
        return merged.reset_index(drop=True), audit

    raise RuntimeError("Unexpected merge mode")


def build_gene_index(adata) -> Dict[str, int]:
    """
    Build case-insensitive gene name index from var_names and common var columns.
    """
    gene_to_idx = {}

    def add_name(name, idx):
        s = clean_str(name)
        if s == "":
            return
        key = s.lower()
        if key not in gene_to_idx:
            gene_to_idx[key] = idx

    for i, g in enumerate(adata.var_names.astype(str)):
        add_name(g, i)

    for col in ["gene_symbol", "gene_symbols", "symbol", "gene_name", "feature_name", "name"]:
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).values
            for i, g in enumerate(vals):
                add_name(g, i)

    return gene_to_idx


def extract_gene_vector(adata, gene: str, gene_index: Dict[str, int]) -> Tuple[np.ndarray, Dict]:
    key = gene.lower()
    audit = {"gene": gene, "found": False, "matched_index": None}

    if key not in gene_index:
        return np.full(adata.n_obs, np.nan, dtype=float), audit

    idx = gene_index[key]
    X = adata.X[:, idx]

    if sparse is not None and sparse.issparse(X):
        arr = np.asarray(X.toarray()).ravel()
    else:
        arr = np.asarray(X).ravel()

    arr = arr.astype(float)
    audit["found"] = True
    audit["matched_index"] = int(idx)
    audit["var_name"] = str(adata.var_names[idx])
    return arr, audit


def robust_minmax(values: np.ndarray, q_low=1.0, q_high=99.0) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    ok = np.isfinite(x)
    if ok.sum() == 0:
        return out

    lo, hi = np.nanpercentile(x[ok], [q_low, q_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out[ok] = 0.0
    else:
        out[ok] = np.clip((x[ok] - lo) / (hi - lo), 0, 1)
    return out


def compute_dot_stats(
    df: pd.DataFrame,
    expr_df: pd.DataFrame,
    row_defs: List[Tuple[str, str, str]],
    group_col: str,
    group_order: List[str],
    expr_threshold: float,
) -> pd.DataFrame:
    rows = []

    for program, gene, row_label in row_defs:
        if row_label not in expr_df.columns:
            continue

        x = expr_df[row_label].values.astype(float)

        for g in group_order:
            mask = df[group_col].astype(str).values == str(g)
            n = int(mask.sum())

            if n == 0:
                mean_expr = np.nan
                pct = np.nan
            else:
                vals = x[mask]
                vals_ok = vals[np.isfinite(vals)]
                if len(vals_ok) == 0:
                    mean_expr = np.nan
                    pct = np.nan
                else:
                    mean_expr = float(np.nanmean(vals_ok))
                    pct = float((vals_ok > expr_threshold).mean() * 100.0)

            rows.append({
                "program": program,
                "gene": gene,
                "row_label": row_label,
                "group_col": group_col,
                "group": g,
                "n_group": n,
                "mean_expression": mean_expr,
                "pct_expressed": pct,
            })

    return pd.DataFrame(rows)


def add_scaled_expression(stats: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    stats["relative_expression"] = np.nan

    for row_label, sub_idx in stats.groupby("row_label").groups.items():
        vals = stats.loc[sub_idx, "mean_expression"].values.astype(float)
        stats.loc[sub_idx, "relative_expression"] = robust_minmax(vals)

    return stats


def plot_dotplot(
    state_stats: pd.DataFrame,
    time_stats: pd.DataFrame,
    row_defs: List[Tuple[str, str, str]],
    outdir: str,
    prefix: str = "Fig_Step80_CandidatePerturbationAxisExpressionDotplot",
) -> Dict[str, str]:

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 180,
        "savefig.dpi": 450,
    })

    row_labels = [r[2] for r in row_defs]
    programs = [r[0] for r in row_defs]
    n_rows = len(row_labels)
    y_map = {lab: i for i, lab in enumerate(row_labels)}

    fig = plt.figure(figsize=(12.5, max(7.5, 0.36 * n_rows + 2.8)))
    gs = fig.add_gridspec(
        nrows=1, ncols=3,
        width_ratios=[1.08, 1.0, 0.08],
        left=0.20, right=0.91, bottom=0.16, top=0.86, wspace=0.18
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    cax = fig.add_subplot(gs[0, 2])

    def draw_panel(ax, stats, title):
        cats = list(dict.fromkeys(stats["group"].astype(str).tolist()))
        x_map = {c: i for i, c in enumerate(cats)}

        xs, ys, cs, ss = [], [], [], []
        for _, r in stats.iterrows():
            lab = r["row_label"]
            if lab not in y_map:
                continue
            xs.append(x_map[str(r["group"])])
            ys.append(y_map[lab])
            cs.append(r["relative_expression"])
            pct = r["pct_expressed"]
            if pd.isna(pct):
                ss.append(1)
            else:
                ss.append(12 + 130 * (pct / 100.0))

        sc = ax.scatter(
            xs, ys,
            c=cs, s=ss,
            cmap="RdBu_r",
            vmin=0, vmax=1,
            edgecolor="0.25",
            linewidth=0.25,
            alpha=0.95
        )

        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, rotation=45, ha="right")
        ax.set_xlim(-0.7, len(cats) - 0.3)
        ax.set_ylim(n_rows - 0.5, -0.5)
        ax.grid(True, axis="both", color="0.88", linewidth=0.6)
        ax.set_axisbelow(True)

        return sc

    sc = draw_panel(ax1, state_stats, "Across StrokeNiche states")
    draw_panel(ax2, time_stats, "Across timepoints")

    ax1.set_yticks(range(n_rows))
    ax1.set_yticklabels(row_labels)
    ax2.tick_params(axis="y", labelleft=False)

    # group labels and separators
    group_spans = []
    start = 0
    current = programs[0]
    for i, p in enumerate(programs + ["__END__"]):
        if p != current:
            group_spans.append((current, start, i - 1))
            start = i
            current = p

    for p, a, b in group_spans:
        mid = (a + b) / 2
        ax1.text(
            -1.35, mid, p,
            ha="right", va="center",
            fontsize=8.5, fontweight="bold",
            transform=ax1.transData
        )
        if b < n_rows - 1:
            for ax in [ax1, ax2]:
                ax.axhline(b + 0.5, color="0.70", linewidth=0.8)

    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Relative expression\n(row-scaled mean)", rotation=90)

    # dot-size legend
    for pct in [25, 50, 75]:
        ax2.scatter([], [], s=12 + 130 * pct / 100.0, c="lightgray",
                    edgecolor="0.25", linewidth=0.25, label=f"{pct}%")
    leg = ax2.legend(
        title="Percent expressed",
        loc="lower left",
        bbox_to_anchor=(1.02, 0.02),
        frameon=False,
        borderaxespad=0.0
    )
    leg._legend_box.align = "left"

    fig.suptitle(
        "Candidate perturbation-axis expression landscape",
        fontsize=17, fontweight="bold", y=0.96
    )

    fig.text(
        0.5, 0.055,
        "Dot color shows row-scaled mean expression; dot size shows percentage of spots/cells with non-zero expression. "
        "This supports expression availability of candidate axes and is not a ligand-receptor causality test.",
        ha="center", va="center", fontsize=8.5, color="0.25"
    )

    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(outdir, f"{prefix}.{ext}")
        fig.savefig(path, bbox_inches="tight", facecolor="white")
        paths[ext] = path
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label_col", default="state_group")
    parser.add_argument("--time_col", default="timepoint")
    parser.add_argument("--expr_threshold", type=float, default=0.0)
    args = parser.parse_args()

    mkdir(args.outdir)

    print("=" * 100)
    print("Step80 | Candidate perturbation-axis expression dotplot")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={args.outdir}")

    adata = read_h5ad(args.h5ad)
    state = read_state_table(args.state_table)
    meta, merge_audit = merge_metadata(adata, state)

    if args.label_col not in meta.columns:
        raise ValueError(f"--label_col={args.label_col} not found in merged metadata")
    if args.time_col not in meta.columns:
        raise ValueError(f"--time_col={args.time_col} not found in merged metadata")

    meta[args.label_col] = meta[args.label_col].map(normalize_label)
    meta[args.time_col] = meta[args.time_col].astype(str)

    state_order = order_categories(meta[args.label_col].dropna().unique().tolist(), STATE_ORDER)
    time_order = order_categories(meta[args.time_col].dropna().unique().tolist(), TIME_ORDER)

    gene_index = build_gene_index(adata)
    expr = {}
    gene_audit = []

    for program, gene, row_label in ROW_DEFS:
        vals, aud = extract_gene_vector(adata, gene, gene_index)
        aud["program"] = program
        aud["row_label"] = row_label
        expr[row_label] = vals
        gene_audit.append(aud)

    expr_df = pd.DataFrame(expr)

    state_stats = compute_dot_stats(
        meta, expr_df, ROW_DEFS,
        group_col=args.label_col,
        group_order=state_order,
        expr_threshold=args.expr_threshold,
    )
    time_stats = compute_dot_stats(
        meta, expr_df, ROW_DEFS,
        group_col=args.time_col,
        group_order=time_order,
        expr_threshold=args.expr_threshold,
    )

    all_stats = pd.concat([state_stats, time_stats], axis=0, ignore_index=True)
    all_stats = add_scaled_expression(all_stats)

    state_stats_scaled = all_stats[all_stats["group_col"] == args.label_col].copy()
    time_stats_scaled = all_stats[all_stats["group_col"] == args.time_col].copy()

    fig_paths = plot_dotplot(
        state_stats_scaled,
        time_stats_scaled,
        ROW_DEFS,
        args.outdir
    )

    stats_path = os.path.join(args.outdir, "step80_candidate_axis_expression_dotplot_summary.csv")
    all_stats.to_csv(stats_path, index=False)

    gene_audit_path = os.path.join(args.outdir, "step80_gene_matching_audit.csv")
    pd.DataFrame(gene_audit).to_csv(gene_audit_path, index=False)

    report = {
        "status": "ok",
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "outdir": args.outdir,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "merge_audit": merge_audit,
        "label_col": args.label_col,
        "time_col": args.time_col,
        "state_order": state_order,
        "time_order": time_order,
        "missing_genes": [x["gene"] for x in gene_audit if not x["found"]],
        "outputs": {
            "figure": fig_paths,
            "summary_csv": stats_path,
            "gene_audit_csv": gene_audit_path,
        },
        "interpretation_note": (
            "Dotplot supports expression availability of candidate perturbation axes. "
            "It is not a ligand-receptor causality test and not wet-lab perturbation evidence."
        )
    }

    report_path = os.path.join(args.outdir, "step80_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Done.")
    print(json.dumps(report["outputs"], indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()