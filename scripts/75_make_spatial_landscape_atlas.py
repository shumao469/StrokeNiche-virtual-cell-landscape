#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step75 | Spatial small-multiples atlas for StrokeNiche

Outputs:
1) marker atlas
2) module atlas
3) state probability atlas
4) virtual perturbation rescue atlas (optional)

Main idea:
- read h5ad
- extract spatial coordinates
- pull gene expression
- compute module scores from curated gene sets
- merge state probabilities from external table
- optionally merge virtual perturbation spatial maps
- render small-multiples manuscript-style figures

Author: ChatGPT (for Shumao Xu)
"""

import os
import re
import json
import math
import argparse
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

from scipy import sparse


# =========================================================
# Utility
# =========================================================

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def clean_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_\.]+", "_", str(s))
    return s.strip("_")

def first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return None

def robust_qmin_qmax(x: np.ndarray, q_low=0.01, q_high=0.99) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    lo = np.quantile(x, q_low)
    hi = np.quantile(x, q_high)
    if np.isclose(lo, hi):
        lo = np.min(x)
        hi = np.max(x)
    if np.isclose(lo, hi):
        hi = lo + 1e-6
    return float(lo), float(hi)

def zscore_clip(x: np.ndarray, clip=2.5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd < 1e-12:
        return np.zeros_like(x)
    z = (x - mu) / sd
    z = np.clip(z, -clip, clip)
    return z

def dense_vector(x):
    if sparse.issparse(x):
        x = x.toarray()
    x = np.asarray(x).reshape(-1)
    return x

def infer_obs_id_column(obs: pd.DataFrame, state_df: Optional[pd.DataFrame] = None) -> str:
    candidates = [
        "cell_id", "spot_id", "barcode", "obs_names", "index",
        "CellID", "SpotID", "cellid", "spotid"
    ]
    obs_cols = list(obs.columns)
    col = first_existing(obs_cols, candidates)
    if col is not None:
        return col

    # fallback: create from index
    obs["_obs_index_id_"] = obs.index.astype(str)
    return "_obs_index_id_"

def infer_xy_columns(obs: pd.DataFrame) -> Tuple[str, str]:
    x_candidates = [
        "x", "X", "x_coord", "xcoord", "array_col", "pxl_col_in_fullres",
        "imagecol", "col", "spatial_x", "coord_x"
    ]
    y_candidates = [
        "y", "Y", "y_coord", "ycoord", "array_row", "pxl_row_in_fullres",
        "imagerow", "row", "spatial_y", "coord_y"
    ]
    xcol = first_existing(list(obs.columns), x_candidates)
    ycol = first_existing(list(obs.columns), y_candidates)
    if xcol is None or ycol is None:
        raise ValueError(
            "Cannot infer spatial coordinates. Please add x/y-like columns into adata.obs "
            "or adapt infer_xy_columns()."
        )
    return xcol, ycol

def infer_prob_columns(df: pd.DataFrame) -> Dict[str, str]:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    def pick(cands):
        for c in cands:
            if c in cols:
                return c
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        return None

    core = pick([
        "lesion_core_like_prob", "lesion-core-like_prob", "core_prob",
        "p_core", "prob_core", "lesion_core_prob", "core_like_prob",
        "core probability", "core"
    ])
    peri = pick([
        "peri_infarct_prob", "peri-infarct_prob", "peri_prob",
        "p_peri", "prob_peri", "peri_like_prob", "peri probability", "peri"
    ])
    remote = pick([
        "remote_like_prob", "remote-like_prob", "remote_prob",
        "p_remote", "prob_remote", "remote probability", "remote"
    ])

    return {"core": core, "peri": peri, "remote": remote}

def infer_boundary_from_points(x: np.ndarray, y: np.ndarray,
                               xpad_ratio=0.05, ypad_ratio=0.05):
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)
    xpad = (xmax - xmin) * xpad_ratio
    ypad = (ymax - ymin) * ypad_ratio
    return xmin - xpad, xmax + xpad, ymin - ypad, ymax + ypad


# =========================================================
# Gene sets / programs
# =========================================================

DEFAULT_MARKER_GROUPS = {
    "Microglia/myeloid": ["C1qa", "C1qb", "Tyrobp", "Lgals3"],
    "Endothelial/BBB": ["Kdr", "Cldn5", "Pecam1", "Kcnj8"],
    "Ferroptosis/hypoxia": ["Hmox1", "Fth1", "Slc7a11", "Gpx4"],
    "Repair/ECM": ["Col1a1", "Col3a1", "Fn1", "Spp1"],
}

DEFAULT_MODULE_GENESETS = {
    "repair_ECM": ["Col1a1", "Col3a1", "Fn1", "Spp1", "Vim", "Apoe"],
    "ferroptosis": ["Hmox1", "Fth1", "Slc7a11", "Gpx4", "Tfrc", "Acsl4"],
    "microglia_inflammation": ["C1qa", "C1qb", "Tyrobp", "Lgals3", "Ctsd", "Ccl3"],
    "BBB_endothelial": ["Kdr", "Cldn5", "Pecam1", "Kcnj8", "Emcn", "Rgs5"],
    "astrocyte_reactive": ["Gfap", "Aqp4", "Vim", "Clu", "Apoe", "Serpina3n"],
    "hypoxia_redox": ["Hmox1", "Fth1", "Nfe2l2", "Hif1a", "Vegfa", "Sod2"],
}

DEFAULT_RESCUE_ORDER = [
    "Ccl2/Ccr2-Ackr1 blockade",
    "Spp1-Cd44 blockade",
    "Vegfa-Flt1/Kdr blockade",
    "ferroptosis down-modulation",
    "repair-ECM promotion",
]


# =========================================================
# I/O loaders
# =========================================================

def load_h5ad(h5ad_path: str) -> ad.AnnData:
    adata = ad.read_h5ad(h5ad_path)
    return adata

def load_state_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df

def load_rescue_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or not os.path.exists(path):
        return None
    # support csv / tsv / parquet
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".tsv") or path.endswith(".txt"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)

def maybe_load_json(path: Optional[str]) -> Optional[dict]:
    if path is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# Expression / module extraction
# =========================================================

def gene_name_lookup(adata: ad.AnnData) -> Dict[str, str]:
    """
    Map uppercase symbol -> exact var name in adata.
    """
    mapping = {}
    for g in adata.var_names.astype(str):
        mapping[g.upper()] = g
    return mapping

def extract_gene_expression(adata: ad.AnnData, gene: str, gene_map: Dict[str, str]) -> np.ndarray:
    g = gene_map.get(gene.upper(), None)
    if g is None:
        return np.full(adata.n_obs, np.nan)
    x = adata[:, g].X
    x = dense_vector(x)
    return x

def compute_module_score(adata: ad.AnnData,
                         genes: List[str],
                         gene_map: Dict[str, str],
                         mode: str = "mean_z") -> np.ndarray:
    expr_list = []
    for g in genes:
        if gene_map.get(g.upper(), None) is None:
            continue
        x = extract_gene_expression(adata, g, gene_map)
        expr_list.append(x)

    if len(expr_list) == 0:
        return np.full(adata.n_obs, np.nan)

    M = np.vstack(expr_list).T  # n_obs x n_genes

    if mode == "mean_raw":
        score = np.nanmean(M, axis=1)
    else:
        # default: mean z across genes
        Z = np.zeros_like(M, dtype=float)
        for j in range(M.shape[1]):
            Z[:, j] = zscore_clip(M[:, j], clip=2.5)
        score = np.nanmean(Z, axis=1)

    return score


# =========================================================
# Merge metadata
# =========================================================

def merge_state_metadata(obs: pd.DataFrame,
                         obs_id_col: str,
                         state_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    meta = obs.copy()

    if state_df is None:
        return meta

    state_df = state_df.copy()
    state_df_cols = list(state_df.columns)

    state_id_candidates = [
        "cell_id", "spot_id", "barcode", "obs_names", "index",
        "CellID", "SpotID", "cellid", "spotid"
    ]
    state_id_col = first_existing(state_df_cols, state_id_candidates)

    if state_id_col is None:
        # fallback: if row counts match, merge by index order
        if len(state_df) == len(meta):
            state_df["_row_order_"] = np.arange(len(state_df))
            meta["_row_order_"] = np.arange(len(meta))
            meta = meta.merge(state_df, on="_row_order_", how="left", suffixes=("", "_state"))
            meta.drop(columns=["_row_order_"], inplace=True, errors="ignore")
            return meta
        else:
            warnings.warn("No obvious ID column in state table; skip merging state metadata.")
            return meta

    meta["_merge_id_"] = meta[obs_id_col].astype(str)
    state_df["_merge_id_"] = state_df[state_id_col].astype(str)
    meta = meta.merge(state_df, on="_merge_id_", how="left", suffixes=("", "_state"))
    return meta

def merge_rescue_metadata(meta: pd.DataFrame,
                          rescue_df: Optional[pd.DataFrame],
                          obs_id_col: str) -> pd.DataFrame:
    """
    Expected rescue_df shapes:
    A) wide:
       columns contain one id column + many columns like
       "Ccl2/Ccr2-Ackr1 blockade__baseline", "__perturbed", "__delta_gain", "__core_reduction"
    B) long:
       id_col, perturbation, baseline, perturbed, delta_gain, core_reduction
    """
    if rescue_df is None:
        return meta

    df = rescue_df.copy()
    id_candidates = ["cell_id", "spot_id", "barcode", "obs_names", "index", "CellID", "SpotID"]
    rid = first_existing(list(df.columns), id_candidates)

    if rid is None:
        if len(df) == len(meta):
            df["_row_order_"] = np.arange(len(df))
            meta["_row_order_"] = np.arange(len(meta))
            meta = meta.merge(df, on="_row_order_", how="left", suffixes=("", "_rescue"))
            meta.drop(columns=["_row_order_"], inplace=True, errors="ignore")
            return meta
        warnings.warn("Cannot infer rescue id column; skip rescue merge.")
        return meta

    meta["_merge_id_"] = meta[obs_id_col].astype(str)
    df["_merge_id_"] = df[rid].astype(str)

    # if long format -> pivot wider
    lower_cols = {c.lower(): c for c in df.columns}
    perturb_col = lower_cols.get("perturbation", None)
    baseline_col = lower_cols.get("baseline", None)
    perturbed_col = lower_cols.get("perturbed", None)
    gain_col = lower_cols.get("delta_gain", lower_cols.get("repair_gain", None))
    core_red_col = lower_cols.get("core_reduction", lower_cols.get("delta_core", None))

    if perturb_col is not None and (baseline_col is not None or perturbed_col is not None):
        value_cols = [c for c in [baseline_col, perturbed_col, gain_col, core_red_col] if c is not None]
        wide_parts = []
        for vc in value_cols:
            tmp = df.pivot(index="_merge_id_", columns=perturb_col, values=vc)
            tmp.columns = [f"{pert}__{vc}" for pert in tmp.columns]
            wide_parts.append(tmp)
        wide = pd.concat(wide_parts, axis=1).reset_index()
        meta = meta.merge(wide, on="_merge_id_", how="left")
    else:
        meta = meta.merge(df, on="_merge_id_", how="left", suffixes=("", "_rescue"))

    return meta


# =========================================================
# Plotting helpers
# =========================================================

def make_dark_expression_cmap():
    # black-purple-yellow-like spatial vibe
    colors = ["#05070b", "#2a0a52", "#7d2e8d", "#d35f8d", "#f7c86b"]
    return LinearSegmentedColormap.from_list("dark_expr", colors)

def make_red_cmap():
    colors = ["#f8eee8", "#f7cfc0", "#f5a38d", "#e85a47", "#9e0000"]
    return LinearSegmentedColormap.from_list("soft_red", colors)

def make_diverging_cmap():
    colors = ["#2b6cb0", "#8bb9d9", "#e9e9e9", "#efb086", "#b00020"]
    return LinearSegmentedColormap.from_list("div_soft", colors)

def scatter_spatial(ax, x, y, val,
                    cmap="viridis",
                    norm=None,
                    s=4,
                    alpha=0.95,
                    background_color="#0b0d11",
                    outline=False,
                    title=None,
                    title_color="white",
                    text_size=8,
                    invert_y=True):
    ax.set_facecolor(background_color)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
    xg = x[good]
    yg = y[good]
    vg = val[good]

    ax.scatter(
        xg, yg,
        c=vg,
        cmap=cmap,
        norm=norm,
        s=s,
        linewidths=0,
        alpha=alpha,
        rasterized=True
    )

    if outline:
        # optional convex-ish outer cloud using faint points
        ax.scatter(xg, yg, s=s+0.5, c="white", alpha=0.02, linewidths=0)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    if invert_y:
        ax.invert_yaxis()

    if title is not None:
        ax.set_title(title, fontsize=text_size, color=title_color, pad=2, weight="bold")

def add_shared_colorbar(fig, axes, cmap, norm, label, color="white",
                        shrink=0.9, width=0.012, pad=0.008):
    # place on right of provided axes union
    boxes = [ax.get_position() for ax in axes]
    x1 = max([b.x1 for b in boxes])
    y0 = min([b.y0 for b in boxes])
    y1 = max([b.y1 for b in boxes])
    cax = fig.add_axes([x1 + pad, y0, width, (y1 - y0) * shrink])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label(label, color=color, fontsize=9)
    cb.ax.tick_params(colors=color, labelsize=8)
    cb.outline.set_edgecolor(color)
    return cb

def add_row_label(ax, text, color="white", fontsize=9):
    ax.text(-0.08, 0.5, text,
            transform=ax.transAxes,
            ha="right", va="center",
            fontsize=fontsize, color=color, weight="bold")

def set_figure_title(fig, title, color="white", fontsize=18):
    fig.suptitle(title, fontsize=fontsize, color=color, weight="bold", y=0.995)


# =========================================================
# Figure builders
# =========================================================

def build_marker_atlas(meta: pd.DataFrame,
                       xcol: str,
                       ycol: str,
                       adata: ad.AnnData,
                       marker_groups: Dict[str, List[str]],
                       out_prefix: str,
                       point_size=3.2,
                       dark_bg="#05070b",
                       dpi=300):
    gene_map = gene_name_lookup(adata)

    n_rows = len(marker_groups)
    n_cols = max(len(v) for v in marker_groups.values())

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 * n_cols, 2.7 * n_rows),
        facecolor=dark_bg
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, None]

    cmap = make_dark_expression_cmap()
    all_axes = []
    all_vals_for_global = []

    # first pass for global expression range
    extracted = {}
    for row_name, genes in marker_groups.items():
        for g in genes:
            val = extract_gene_expression(adata, g, gene_map)
            extracted[(row_name, g)] = val
            vv = val[np.isfinite(val)]
            if len(vv) > 0:
                all_vals_for_global.append(vv)
    if len(all_vals_for_global) == 0:
        global_lo, global_hi = 0.0, 1.0
    else:
        cat = np.concatenate(all_vals_for_global)
        global_lo, global_hi = robust_qmin_qmax(cat, 0.02, 0.99)
    norm = Normalize(vmin=global_lo, vmax=global_hi)

    x = meta[xcol].values
    y = meta[ycol].values

    row_names = list(marker_groups.keys())
    for i, row_name in enumerate(row_names):
        genes = marker_groups[row_name]
        for j in range(n_cols):
            ax = axes[i, j]
            ax.set_facecolor(dark_bg)
            if j < len(genes):
                g = genes[j]
                val = extracted[(row_name, g)]
                scatter_spatial(
                    ax, x, y, val,
                    cmap=cmap, norm=norm,
                    s=point_size,
                    background_color=dark_bg,
                    title=g,
                    title_color="white",
                    text_size=10
                )
            else:
                ax.axis("off")
                ax.set_facecolor(dark_bg)

            all_axes.append(ax)

        add_row_label(axes[i, 0], row_name, color="white", fontsize=10)

    set_figure_title(fig, "Spatial atlas of representative marker genes", color="white", fontsize=20)

    add_shared_colorbar(fig, all_axes, cmap, norm, label="Expression", color="white")

    fig.text(0.5, 0.01,
             "Rows denote compartments/programs; each panel shows spatial expression intensity.",
             ha="center", va="bottom", color="white", fontsize=9)

    plt.tight_layout(rect=[0.03, 0.03, 0.94, 0.97])

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{out_prefix}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_module_atlas(meta: pd.DataFrame,
                       xcol: str,
                       ycol: str,
                       module_scores: Dict[str, np.ndarray],
                       out_prefix: str,
                       point_size=3.2,
                       dark_bg="#f7f7f7",
                       ncols=3,
                       dpi=300):
    modules = list(module_scores.keys())
    n = len(modules)
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 3.6 * nrows),
        facecolor="white"
    )
    axes = np.array(axes).reshape(nrows, ncols)

    cmap = make_diverging_cmap()
    all_axes = []

    # global z-based norm
    vals = []
    for m in modules:
        x = module_scores[m]
        x = zscore_clip(x, clip=2.5)
        vals.append(x[np.isfinite(x)])
    if len(vals) == 0:
        v = 1.5
    else:
        cat = np.concatenate(vals)
        v = max(1.2, np.quantile(np.abs(cat), 0.98))
    norm = TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v)

    X = meta[xcol].values
    Y = meta[ycol].values

    k = 0
    for i in range(nrows):
        for j in range(ncols):
            ax = axes[i, j]
            if k < n:
                m = modules[k]
                val = zscore_clip(module_scores[m], clip=2.5)
                scatter_spatial(
                    ax, X, Y, val,
                    cmap=cmap, norm=norm,
                    s=point_size,
                    background_color="white",
                    title=m,
                    title_color="black",
                    text_size=11,
                    invert_y=True
                )
                all_axes.append(ax)
            else:
                ax.axis("off")
            k += 1

    set_figure_title(fig, "Spatial atlas of module activities", color="black", fontsize=20)
    add_shared_colorbar(fig, all_axes, cmap, norm, label="Module score (z)", color="black")

    fig.text(0.5, 0.01,
             "Module scores are computed from curated gene sets and shown as standardized spatial activities.",
             ha="center", va="bottom", color="black", fontsize=9)

    plt.tight_layout(rect=[0.03, 0.03, 0.94, 0.97])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{out_prefix}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def build_state_probability_atlas(meta: pd.DataFrame,
                                  xcol: str,
                                  ycol: str,
                                  prob_cols: Dict[str, str],
                                  out_prefix: str,
                                  point_size=3.2,
                                  dpi=300):
    state_order = [("core", "lesion-core-like"),
                   ("peri", "peri-infarct"),
                   ("remote", "remote-like")]

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.2), facecolor="white")
    cmap = make_red_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)

    X = meta[xcol].values
    Y = meta[ycol].values
    used_axes = []

    for ax, (k, lab) in zip(axes, state_order):
        col = prob_cols.get(k, None)
        if col is None or col not in meta.columns:
            ax.axis("off")
            ax.set_title(f"{lab}\n(missing)", fontsize=11)
            continue
        val = meta[col].values.astype(float)
        scatter_spatial(
            ax, X, Y, val,
            cmap=cmap, norm=norm,
            s=point_size,
            background_color="white",
            title=lab,
            title_color="black",
            text_size=11
        )
        used_axes.append(ax)

    set_figure_title(fig, "Spatial atlas of StrokeNiche state probabilities", color="black", fontsize=18)
    if len(used_axes) > 0:
        add_shared_colorbar(fig, used_axes, cmap, norm, label="Probability", color="black")

    fig.text(0.5, 0.01,
             "State probability maps summarize lesion-core-like, peri-infarct, and remote-like spatial organization.",
             ha="center", va="bottom", color="black", fontsize=9)

    plt.tight_layout(rect=[0.03, 0.04, 0.94, 0.95])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{out_prefix}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def resolve_rescue_panel_columns(meta: pd.DataFrame,
                                 perturbation_name: str) -> Dict[str, Optional[str]]:
    """
    Look for columns:
      "{perturbation}__baseline"
      "{perturbation}__perturbed"
      "{perturbation}__delta_gain"
      "{perturbation}__core_reduction"

    Also allow variants.
    """
    cols = list(meta.columns)
    lower_map = {c.lower(): c for c in cols}

    def pick(name_options):
        for nm in name_options:
            if nm in cols:
                return nm
            if nm.lower() in lower_map:
                return lower_map[nm.lower()]
        return None

    base_cands = [
        f"{perturbation_name}__baseline",
        f"{perturbation_name}__baseline_score",
        f"{perturbation_name}__core_like_baseline",
    ]
    pert_cands = [
        f"{perturbation_name}__perturbed",
        f"{perturbation_name}__perturbed_score",
        f"{perturbation_name}__rescue_score",
    ]
    gain_cands = [
        f"{perturbation_name}__delta_gain",
        f"{perturbation_name}__repair_gain",
        f"{perturbation_name}__gain",
    ]
    core_red_cands = [
        f"{perturbation_name}__core_reduction",
        f"{perturbation_name}__delta_core",
        f"{perturbation_name}__core_drop",
    ]

    return {
        "baseline": pick(base_cands),
        "perturbed": pick(pert_cands),
        "delta_gain": pick(gain_cands),
        "core_reduction": pick(core_red_cands),
    }

def build_rescue_atlas(meta: pd.DataFrame,
                       xcol: str,
                       ycol: str,
                       perturbations: List[str],
                       out_prefix: str,
                       point_size=3.2,
                       dpi=300):
    cols_needed = ["baseline", "perturbed", "delta_gain", "core_reduction"]
    n_rows = len(perturbations)
    n_cols = len(cols_needed)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 2.7 * n_rows),
                             facecolor="white")
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, None]

    X = meta[xcol].values
    Y = meta[ycol].values

    # collect ranges
    baseline_vals = []
    perturbed_vals = []
    gain_vals = []
    core_red_vals = []

    resolved = {}
    for p in perturbations:
        r = resolve_rescue_panel_columns(meta, p)
        resolved[p] = r
        if r["baseline"] in meta.columns:
            baseline_vals.append(meta[r["baseline"]].values)
        if r["perturbed"] in meta.columns:
            perturbed_vals.append(meta[r["perturbed"]].values)
        if r["delta_gain"] in meta.columns:
            gain_vals.append(meta[r["delta_gain"]].values)
        if r["core_reduction"] in meta.columns:
            core_red_vals.append(meta[r["core_reduction"]].values)

    red_cmap = make_red_cmap()
    purple_cmap = make_dark_expression_cmap()
    green_cmap = LinearSegmentedColormap.from_list("gain_green", ["#edf5ed", "#b7ddb7", "#62b862", "#0d7a33"])
    blue_cmap = LinearSegmentedColormap.from_list("core_blue", ["#edf2f7", "#b9d5f0", "#5da0db", "#0a4e9b"])

    def global_norm(vals, default=(0.0, 1.0)):
        if len(vals) == 0:
            return Normalize(vmin=default[0], vmax=default[1])
        cat = np.concatenate([v[np.isfinite(v)] for v in vals if np.any(np.isfinite(v))])
        if len(cat) == 0:
            return Normalize(vmin=default[0], vmax=default[1])
        lo, hi = robust_qmin_qmax(cat, 0.02, 0.98)
        return Normalize(vmin=lo, vmax=hi)

    norm_baseline = global_norm(baseline_vals, default=(0, 1))
    norm_perturbed = global_norm(perturbed_vals, default=(0, 1))
    norm_gain = global_norm(gain_vals, default=(0, 1))
    norm_core_red = global_norm(core_red_vals, default=(0, 1))

    col_titles = {
        "baseline": "Baseline",
        "perturbed": "Perturbed",
        "delta_gain": "Delta gain",
        "core_reduction": "Core reduction"
    }
    cmaps = {
        "baseline": red_cmap,
        "perturbed": purple_cmap,
        "delta_gain": green_cmap,
        "core_reduction": blue_cmap
    }
    norms = {
        "baseline": norm_baseline,
        "perturbed": norm_perturbed,
        "delta_gain": norm_gain,
        "core_reduction": norm_core_red
    }

    all_axes = []
    for i, p in enumerate(perturbations):
        r = resolved[p]
        for j, key in enumerate(cols_needed):
            ax = axes[i, j]
            c = r.get(key, None)
            if c is None or c not in meta.columns:
                ax.axis("off")
                if i == 0:
                    ax.set_title(col_titles[key], fontsize=11, weight="bold")
                continue
            val = meta[c].values.astype(float)
            scatter_spatial(
                ax, X, Y, val,
                cmap=cmaps[key], norm=norms[key],
                s=point_size,
                background_color="white",
                title=col_titles[key] if i == 0 else None,
                title_color="black",
                text_size=11
            )
            all_axes.append(ax)

        add_row_label(axes[i, 0], p, color="black", fontsize=9)

    set_figure_title(fig, "Virtual perturbation spatial rescue atlas", color="black", fontsize=18)

    fig.text(0.5, 0.012,
             "Rows denote in silico perturbations; columns summarize baseline, perturbed, rescue gain, and core-reduction maps.",
             ha="center", va="bottom", color="black", fontsize=9)

    plt.tight_layout(rect=[0.06, 0.04, 0.98, 0.96])
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{out_prefix}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Step75 spatial landscape atlas")

    parser.add_argument("--h5ad", required=True, help="Path to spatial_all.h5ad")
    parser.add_argument("--state_table", default=None, help="CSV with state probabilities")
    parser.add_argument("--rescue_table", default=None, help="Optional rescue spatial map table")
    parser.add_argument("--outdir", required=True, help="Output directory")

    parser.add_argument("--marker_json", default=None,
                        help="Optional JSON file for marker groups. Format: {row_name:[genes,...], ...}")
    parser.add_argument("--module_json", default=None,
                        help="Optional JSON file for module gene sets. Format: {module_name:[genes,...], ...}")

    parser.add_argument("--point_size", type=float, default=3.2)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--make_rescue", action="store_true",
                        help="Whether to build rescue atlas")

    args = parser.parse_args()

    safe_mkdir(args.outdir)

    print("=" * 100)
    print("Step75 | Spatial small-multiples atlas")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"rescue_table={args.rescue_table}")
    print(f"outdir={args.outdir}")

    adata = load_h5ad(args.h5ad)
    obs = adata.obs.copy()

    obs_id_col = infer_obs_id_column(obs)
    xcol, ycol = infer_xy_columns(obs)

    print(f"[INFO] obs_id_col = {obs_id_col}")
    print(f"[INFO] xcol = {xcol}, ycol = {ycol}")

    state_df = load_state_table(args.state_table)
    meta = merge_state_metadata(obs, obs_id_col, state_df)

    rescue_df = load_rescue_table(args.rescue_table)
    meta = merge_rescue_metadata(meta, rescue_df, obs_id_col)

    marker_groups = DEFAULT_MARKER_GROUPS
    if args.marker_json is not None and os.path.exists(args.marker_json):
        marker_groups = maybe_load_json(args.marker_json)

    module_genesets = DEFAULT_MODULE_GENESETS
    if args.module_json is not None and os.path.exists(args.module_json):
        module_genesets = maybe_load_json(args.module_json)

    # module scores
    gene_map = gene_name_lookup(adata)
    module_scores = {}
    for m, genes in module_genesets.items():
        module_scores[m] = compute_module_score(adata, genes, gene_map, mode="mean_z")

    prob_cols = infer_prob_columns(meta)
    print(f"[INFO] inferred probability columns: {prob_cols}")

    # 1) marker atlas
    marker_prefix = os.path.join(args.outdir, "Fig_Step75_SpatialMarkerAtlas")
    build_marker_atlas(
        meta=meta,
        xcol=xcol,
        ycol=ycol,
        adata=adata,
        marker_groups=marker_groups,
        out_prefix=marker_prefix,
        point_size=args.point_size,
        dpi=args.dpi
    )

    # 2) module atlas
    module_prefix = os.path.join(args.outdir, "Fig_Step75_SpatialModuleAtlas")
    build_module_atlas(
        meta=meta,
        xcol=xcol,
        ycol=ycol,
        module_scores=module_scores,
        out_prefix=module_prefix,
        point_size=args.point_size,
        dpi=args.dpi
    )

    # 3) probability atlas
    state_prefix = os.path.join(args.outdir, "Fig_Step75_SpatialStateProbabilityAtlas")
    build_state_probability_atlas(
        meta=meta,
        xcol=xcol,
        ycol=ycol,
        prob_cols=prob_cols,
        out_prefix=state_prefix,
        point_size=args.point_size,
        dpi=args.dpi
    )

    # 4) rescue atlas (optional)
    rescue_prefix = None
    if args.make_rescue:
        rescue_prefix = os.path.join(args.outdir, "Fig_Step75_VirtualPerturbationSpatialRescueAtlas")
        build_rescue_atlas(
            meta=meta,
            xcol=xcol,
            ycol=ycol,
            perturbations=DEFAULT_RESCUE_ORDER,
            out_prefix=rescue_prefix,
            point_size=args.point_size,
            dpi=args.dpi
        )

    report = {
        "marker_atlas": {
            ext: f"{marker_prefix}.{ext}" for ext in ["png", "pdf", "svg"]
        },
        "module_atlas": {
            ext: f"{module_prefix}.{ext}" for ext in ["png", "pdf", "svg"]
        },
        "state_probability_atlas": {
            ext: f"{state_prefix}.{ext}" for ext in ["png", "pdf", "svg"]
        },
        "rescue_atlas": None if rescue_prefix is None else {
            ext: f"{rescue_prefix}.{ext}" for ext in ["png", "pdf", "svg"]
        },
        "meta_rows": int(len(meta)),
        "obs_id_col": obs_id_col,
        "xcol": xcol,
        "ycol": ycol,
        "prob_cols": prob_cols,
        "modules": list(module_genesets.keys()),
        "marker_groups": marker_groups
    }

    report_json = os.path.join(args.outdir, "step75_report.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_txt = os.path.join(args.outdir, "step75_report.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write(json.dumps(report, indent=2))

    print("\nDONE Step75")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()