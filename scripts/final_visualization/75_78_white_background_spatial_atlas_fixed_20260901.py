#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
White-background manuscript redraw for:
1) Module and state-probability spatial landscape atlas
2) Spatial marker landscape atlas
3) Color-consistent before/after virtual perturbation spatial rescue atlas

Recommended use:
python /mnt/h/vir/ST/75_78_white_background_spatial_atlas.py \
  --h5ad /mnt/h/vir/ST/results/step5_nicheformer/spatial_all_with_nicheformer.h5ad \
  --state_table /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv \
  --step78_dir /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/virtual_perturbation_response_landscape_78d_final \
  --outdir /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/white_background_spatial_figures_75_78 \
  --timepoint D7 \
  --grid_n 360 \
  --sigma 1.35 \
  --force
"""

import os
import re
import json
import glob
import argparse
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore

try:
    import anndata as ad
except Exception as e:
    raise ImportError(
        "This script requires anndata. Install in your current environment, e.g.:\n"
        "pip install anndata h5py scipy pandas matplotlib"
    ) from e


# -----------------------------
# Biological marker/module sets
# -----------------------------
MARKER_GROUPS = OrderedDict({
    "Microglia / myeloid": ["C1qa", "C1qb", "Tyrobp", "Lgals3"],
    "Endothelial / BBB": ["Kdr", "Cldn5", "Pecam1", "Kcnj8"],
    "Ferroptosis / hypoxia": ["Hmox1", "Fth1", "Slc7a11", "Gpx4"],
    "Repair / ECM": ["Col1a1", "Col3a1", "Fn1", "Spp1"],
})

MODULE_GENESETS = OrderedDict({
    "Repair score": ["Spp1", "Fn1", "Col1a1", "Col3a1", "Col1a2", "Col4a1", "Mmp2", "Timp1"],
    "Ferroptosis": ["Hmox1", "Fth1", "Slc7a11", "Gpx4", "Acsl4", "Ptgs2", "Ncoa4"],
    "Inflammation": ["Ccl2", "C1qa", "C1qb", "Tyrobp", "Lgals3", "Tnf", "Il1b", "Trem2", "Apoe"],
    "BBB leakage": ["Kdr", "Flt1", "Cldn5", "Pecam1", "Kcnj8", "Vwf", "Ocln", "Tjp1"],
    "Reactive astrocyte": ["Gfap", "Aqp4", "Vim", "Serpina3n", "Lcn2", "C3"],
})

CANDIDATES = [
    "Ccl2/Ccr2-Ackr1",
    "Spp1-Cd44",
    "Vegfa-Flt1/Kdr",
    "Ferroptosis down",
    "Repair-ECM up",
]


# -----------------------------
# Utility
# -----------------------------
def mkdir(path):
    os.makedirs(path, exist_ok=True)


def clean_name(x):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(x)).strip("_")


def lower_no_space(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).lower())


def make_cmap(name, bad=(1, 1, 1, 0)):
    cmap = mpl.colormaps.get_cmap(name).copy()
    cmap.set_bad(bad)
    return cmap


CMAP_PROB = make_cmap("magma")
CMAP_EXPR = make_cmap("magma")
CMAP_MODULE = make_cmap("RdBu_r")
CMAP_REPAIR = make_cmap("YlGn")
CMAP_CORE_REDUCTION = make_cmap("Blues")
CMAP_PRIORITY = make_cmap("viridis")


def robust_limits(x, qlo=1, qhi=99):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(x, qlo)
    hi = np.nanpercentile(x, qhi)
    if not np.isfinite(lo):
        lo = np.nanmin(x)
    if not np.isfinite(hi):
        hi = np.nanmax(x)
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def robust_rescale01(x, qlo=1, qhi=99):
    x = np.asarray(x, dtype=float)
    lo, hi = robust_limits(x, qlo, qhi)
    y = (x - lo) / (hi - lo + 1e-12)
    return np.clip(y, 0, 1)


def safe_z(x):
    x = np.asarray(x, dtype=float)
    if np.nanstd(x) < 1e-12:
        return np.zeros_like(x, dtype=float)
    y = zscore(x, nan_policy="omit")
    y = np.asarray(y, dtype=float)
    y[~np.isfinite(y)] = 0.0
    return y


def save_figure(fig, outdir, stem, dpi=350):
    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.12,
            facecolor="white",
            edgecolor="white",
        )
        paths[ext] = path
    return paths


def infer_time_col(df):
    candidates = [
        "timepoint", "time_point", "time", "Timepoint", "Time", "stage", "Stage",
        "sample_timepoint", "orig.ident"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        vals = df[c].astype(str).unique()[:20]
        if any(v in {"D1", "D3", "D7"} for v in vals):
            return c
    return None


def order_timepoints(vals):
    order = {"D0": 0, "Ctrl": 0, "CTRL": 0, "control": 0, "D1": 1, "D3": 3, "D7": 7, "D14": 14}
    vals = list(pd.Series(vals).dropna().astype(str).unique())
    return sorted(vals, key=lambda x: order.get(x, 999))


def choose_timepoint(df, time_col, requested):
    if requested and requested.lower() != "auto":
        return requested
    if time_col is None:
        return None
    vals = order_timepoints(df[time_col].astype(str).unique())
    for t in ["D7", "D3", "D1"]:
        if t in vals:
            return t
    return vals[-1] if vals else None


def infer_state_label_col(df):
    preferred = [
        "state_group", "region_auto", "region", "state", "State",
        "stroke_state", "latent_state", "region_label"
    ]
    for c in preferred:
        if c in df.columns:
            vals = df[c].astype(str).str.lower()
            if vals.str.contains("core|peri|remote|lesion|infarct").any():
                return c
    for c in df.columns:
        vals = df[c].astype(str).str.lower()
        if vals.str.contains("core").any() and vals.str.contains("remote").any():
            return c
    return None


def label_onehot(df, label_col):
    if label_col is None:
        raise ValueError("No state/region label column found for label-derived state probability maps.")
    lab = df[label_col].astype(str).str.lower()

    core = lab.str.contains("core|lesion", regex=True).to_numpy(dtype=float)
    peri = lab.str.contains("peri|penumbra|infarct", regex=True).to_numpy(dtype=float)
    remote = lab.str.contains("remote", regex=True).to_numpy(dtype=float)

    # Avoid double assignment when label is "lesion-core-like"
    peri = np.where(core > 0, 0, peri)

    return {
        "Core probability": core,
        "Peri-infarct probability": peri,
        "Remote-like probability": remote,
    }


def get_xy(df):
    for xcol, ycol in [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("array_row", "array_col"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
    ]:
        if xcol in df.columns and ycol in df.columns:
            return df[xcol].to_numpy(float), df[ycol].to_numpy(float)

    raise ValueError(
        "Could not find spatial coordinates in merged dataframe. "
        "Expected spatial_x/spatial_y or x/y columns."
    )


# -----------------------------
# Data loading
# -----------------------------
def load_adata_and_metadata(h5ad_path, state_table_path, spatial_key="spatial", flip_y=False):
    adata = ad.read_h5ad(h5ad_path)

    obs = adata.obs.copy()
    obs["_adata_id_"] = adata.obs_names.astype(str)
    obs["obs_name"] = adata.obs_names.astype(str)

    if spatial_key not in adata.obsm:
        raise KeyError(f"Spatial key {spatial_key!r} not found. Available obsm keys: {list(adata.obsm.keys())}")
    xy = np.asarray(adata.obsm[spatial_key])
    obs["spatial_x"] = xy[:, 0].astype(float)
    obs["spatial_y"] = xy[:, 1].astype(float)
    if flip_y:
        obs["spatial_y"] = -obs["spatial_y"]

    state = pd.read_csv(state_table_path)
    if "_adata_id_" not in state.columns:
        if "obs_name" in state.columns:
            state["_adata_id_"] = state["obs_name"].astype(str)
        elif state.index.name is not None:
            state = state.reset_index().rename(columns={state.index.name: "_adata_id_"})
        else:
            first = state.columns[0]
            if state[first].astype(str).isin(obs["_adata_id_"]).mean() > 0.5:
                state["_adata_id_"] = state[first].astype(str)

    if "_adata_id_" in state.columns:
        state["_adata_id_"] = state["_adata_id_"].astype(str)
        df = obs.merge(state, on="_adata_id_", how="left", suffixes=("", "_state"))
    elif "obs_name" in state.columns:
        state["obs_name"] = state["obs_name"].astype(str)
        df = obs.merge(state, on="obs_name", how="left", suffixes=("", "_state"))
    else:
        raise ValueError("State table must contain _adata_id_ or obs_name column.")

    return adata, df


def build_gene_index(adata):
    gene_to_idx = {}
    for i, g in enumerate(adata.var_names.astype(str)):
        gene_to_idx[g] = i
        gene_to_idx[g.lower()] = i
    return gene_to_idx


def get_gene_expr(adata, gene, gene_to_idx=None):
    if gene_to_idx is None:
        gene_to_idx = build_gene_index(adata)

    idx = None
    if gene in gene_to_idx:
        idx = gene_to_idx[gene]
    elif gene.lower() in gene_to_idx:
        idx = gene_to_idx[gene.lower()]

    if idx is None:
        return None

    X = adata[:, idx].X
    if sparse.issparse(X):
        arr = X.toarray().ravel()
    else:
        arr = np.asarray(X).ravel()

    arr = arr.astype(float)
    if np.nanmax(arr) > 50:
        arr = np.log1p(arr)
    return arr


def compute_module_score(adata, genes, gene_to_idx=None):
    arrs = []
    used = []
    for g in genes:
        x = get_gene_expr(adata, g, gene_to_idx=gene_to_idx)
        if x is not None:
            arrs.append(safe_z(x))
            used.append(g)

    if not arrs:
        return None, []

    M = np.vstack(arrs)
    score = np.nanmean(M, axis=0)
    score = safe_z(score)
    return score, used


def add_module_scores_to_df(adata, df):
    gene_to_idx = build_gene_index(adata)
    module_used = {}
    for name, genes in MODULE_GENESETS.items():
        score, used = compute_module_score(adata, genes, gene_to_idx=gene_to_idx)
        module_used[name] = used
        if score is not None:
            df[f"module__{clean_name(name)}"] = score

    # Prefer model-derived repair_score if available; otherwise use module Repair score.
    if "repair_score" in df.columns:
        df["plot__Repair score"] = pd.to_numeric(df["repair_score"], errors="coerce").to_numpy(float)
        df["plot__Repair score"] = robust_rescale01(df["plot__Repair score"], 1, 99)
    elif "module__Repair_score" in df.columns:
        df["plot__Repair score"] = robust_rescale01(df["module__Repair_score"], 1, 99)

    return df, module_used


# -----------------------------
# Grid smoothing and plotting
# -----------------------------
def grid_smooth(x, y, values, grid_n=360, sigma=1.35, extent=None, density_q=0.05):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[keep], y[keep], v[keep]

    if extent is None:
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        padx = (xmax - xmin) * 0.04 + 1e-6
        pady = (ymax - ymin) * 0.04 + 1e-6
        extent = (xmin - padx, xmax + padx, ymin - pady, ymax + pady)

    xmin, xmax, ymin, ymax = extent
    xedges = np.linspace(xmin, xmax, grid_n + 1)
    yedges = np.linspace(ymin, ymax, grid_n + 1)

    H_count, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    H_sum, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=v)

    dens = gaussian_filter(H_count.astype(float), sigma=sigma)
    sm_sum = gaussian_filter(H_sum.astype(float), sigma=sigma)

    Z = sm_sum / (dens + 1e-8)
    positive = dens[dens > 0]
    if positive.size > 0:
        threshold = np.nanquantile(positive, density_q)
    else:
        threshold = 0
    mask = dens <= threshold
    Z = np.where(mask, np.nan, Z)

    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    Xg, Yg = np.meshgrid(xc, yc, indexing="ij")

    return {
        "Z": Z,
        "density": dens,
        "Xg": Xg,
        "Yg": Yg,
        "extent": extent,
        "support_threshold": threshold,
    }


def apply_clean_axis(ax, show_axes=False):
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    if show_axes:
        ax.tick_params(colors="black", labelsize=8, length=2)
        for s in ax.spines.values():
            s.set_color("0.25")
            s.set_linewidth(0.6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)


def add_support_contour(ax, grid, color="0.45", lw=0.65, alpha=0.9):
    try:
        ax.contour(
            grid["Xg"],
            grid["Yg"],
            grid["density"],
            levels=[grid["support_threshold"]],
            colors=[color],
            linewidths=lw,
            linestyles="--",
            alpha=alpha,
        )
    except Exception:
        pass


def add_state_contours(ax, df_t, state_maps, extent, grid_n, sigma):
    x, y = get_xy(df_t)
    contour_specs = [
        ("Core probability", "#d62728", 0.45),
        ("Peri-infarct probability", "#e69f00", 0.45),
    ]
    for name, color, fallback_level in contour_specs:
        if name not in state_maps:
            continue
        vals = np.asarray(state_maps[name], dtype=float)
        g = grid_smooth(x, y, vals, grid_n=grid_n, sigma=sigma, extent=extent, density_q=0.05)
        Z = g["Z"]
        valid = Z[np.isfinite(Z)]
        if valid.size == 0:
            continue
        level = max(fallback_level, np.nanquantile(valid, 0.78))
        if np.nanmax(valid) <= level:
            level = np.nanquantile(valid, 0.65)
        try:
            ax.contour(
                g["Xg"], g["Yg"], Z,
                levels=[level],
                colors=[color],
                linewidths=0.9,
                alpha=0.95,
            )
        except Exception:
            continue


def draw_landscape(
    ax,
    df_t,
    values,
    title=None,
    cmap=None,
    vmin=None,
    vmax=None,
    extent=None,
    grid_n=360,
    sigma=1.35,
    density_q=0.05,
    show_axes=False,
    raw_overlay=True,
    raw_s=1.2,
    raw_alpha=0.35,
    add_boundary=True,
    state_maps=None,
    add_region_contours=True,
):
    x, y = get_xy(df_t)
    values = np.asarray(values, dtype=float)

    grid = grid_smooth(
        x, y, values,
        grid_n=grid_n,
        sigma=sigma,
        extent=extent,
        density_q=density_q,
    )

    Zm = np.ma.masked_invalid(grid["Z"])
    im = ax.imshow(
        Zm.T,
        origin="lower",
        extent=grid["extent"],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )

    if raw_overlay:
        keep = np.isfinite(values)
        ax.scatter(
            x[keep], y[keep],
            c=values[keep],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            s=raw_s,
            alpha=raw_alpha,
            linewidths=0,
            rasterized=True,
        )

    if add_boundary:
        add_support_contour(ax, grid)

    if add_region_contours and state_maps is not None:
        add_state_contours(ax, df_t, state_maps, grid["extent"], grid_n, sigma)

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color="black", pad=5)

    apply_clean_axis(ax, show_axes=show_axes)
    return im, grid


# -----------------------------
# Step78 response table
# -----------------------------
def discover_step78_table(step78_dir):
    if not step78_dir or not os.path.isdir(step78_dir):
        return None, []

    patterns = [
        "**/*per*cell*.csv",
        "**/*spot*.csv",
        "**/*response*.csv",
        "**/*virtual*perturb*.csv",
        "**/*.csv",
    ]
    candidates = []
    for p in patterns:
        candidates.extend(glob.glob(os.path.join(step78_dir, p), recursive=True))
    candidates = sorted(set(candidates))

    scored = []
    for f in candidates:
        try:
            head = pd.read_csv(f, nrows=5)
            cols = [c.lower() for c in head.columns]
            score = 0
            if any(c in cols for c in ["perturbation", "candidate", "axis", "perturbation_name"]):
                score += 5
            if any("baseline" in c and "core" in c for c in cols):
                score += 3
            if any("perturbed" in c and "core" in c for c in cols):
                score += 3
            if any("repair" in c and "gain" in c for c in cols):
                score += 2
            if any(c in cols for c in ["obs_name", "_adata_id_", "cell_id", "spot_id"]):
                score += 2
            if score > 0:
                scored.append((score, f))
        except Exception:
            continue

    scored = sorted(scored, reverse=True)
    if scored:
        return scored[0][1], [x[1] for x in scored[:20]]
    return None, candidates[:20]


def find_col(df, patterns, required=False):
    cols = list(df.columns)
    lows = {c: c.lower() for c in cols}
    for pat in patterns:
        pat_l = pat.lower()
        for c, cl in lows.items():
            if pat_l == cl:
                return c
    for pat in patterns:
        rgx = re.compile(pat, flags=re.I)
        for c in cols:
            if rgx.search(c):
                return c
    if required:
        raise KeyError(f"Could not find column matching any of: {patterns}")
    return None


def normalize_candidate_name(x):
    s = str(x)
    key = lower_no_space(s)
    mapping = {
        "ccl2ccr2ackr1": "Ccl2/Ccr2-Ackr1",
        "ccl2ackr1": "Ccl2/Ccr2-Ackr1",
        "ccl2ccr2ackr1blockade": "Ccl2/Ccr2-Ackr1",
        "spp1cd44": "Spp1-Cd44",
        "spp1cd44blockade": "Spp1-Cd44",
        "vegfaflt1kdr": "Vegfa-Flt1/Kdr",
        "vegfaflt1": "Vegfa-Flt1/Kdr",
        "vegfaflt1kdrblockade": "Vegfa-Flt1/Kdr",
        "ferroptosisdown": "Ferroptosis down",
        "ferroptosisdownmodulation": "Ferroptosis down",
        "repairecmup": "Repair-ECM up",
        "repairecmpromotion": "Repair-ECM up",
    }
    if key in mapping:
        return mapping[key]
    for cand in CANDIDATES:
        ck = lower_no_space(cand)
        if ck in key or key in ck:
            return cand
    return s


def load_step78_response(step78_dir=None, step78_cell_table=None):
    path = step78_cell_table
    candidates = []
    if not path:
        path, candidates = discover_step78_table(step78_dir)
    if not path or not os.path.exists(path):
        return None, {"available": False, "path": None, "candidates": candidates}

    tab = pd.read_csv(path)
    cand_col = find_col(tab, ["perturbation", "candidate", "axis", "perturbation_name", "name"], required=False)
    if cand_col is None:
        return None, {
            "available": False,
            "path": path,
            "reason": "No perturbation/candidate column found",
            "columns": list(tab.columns),
        }

    tab["candidate_std"] = tab[cand_col].map(normalize_candidate_name)

    id_col = find_col(tab, ["_adata_id_", "obs_name", "cell_id", "spot_id", "barcode"], required=False)
    if id_col is None:
        return None, {
            "available": False,
            "path": path,
            "reason": "No cell/spot id column found",
            "columns": list(tab.columns),
        }

    tab["_response_id_"] = tab[id_col].astype(str)

    return tab, {
        "available": True,
        "path": path,
        "candidate_column": cand_col,
        "id_column": id_col,
        "n_rows": int(tab.shape[0]),
        "columns": list(tab.columns),
    }


def standardize_response_columns(resp):
    out = resp.copy()

    baseline_col = find_col(
        out,
        [
            "baseline_core_probability",
            "baseline_core",
            "core_baseline",
            "core_probability_baseline",
            "core_prob_baseline",
        ],
        required=False,
    )
    perturbed_col = find_col(
        out,
        [
            "perturbed_core_probability",
            "perturbed_core",
            "core_perturbed",
            "core_probability_perturbed",
            "post_core_probability",
        ],
        required=False,
    )
    delta_core_col = find_col(
        out,
        [
            "delta_core_probability",
            "delta_core",
            "d_core",
            "core_delta",
            "delta.*core",
        ],
        required=False,
    )
    core_reduction_col = find_col(
        out,
        [
            "core_reduction",
            "core_probability_reduction",
            "reduction_core",
            "core_drop",
        ],
        required=False,
    )
    repair_gain_col = find_col(
        out,
        [
            "repair_gain",
            "delta_repair_positive",
            "repair_score_gain",
            "positive_repair",
        ],
        required=False,
    )
    delta_repair_col = find_col(
        out,
        [
            "delta_repair",
            "delta_repair_score",
            "d_repair",
            "repair_delta",
            "delta.*repair",
        ],
        required=False,
    )

    if baseline_col is not None:
        out["baseline_core_std"] = pd.to_numeric(out[baseline_col], errors="coerce")
    else:
        out["baseline_core_std"] = np.nan

    if perturbed_col is not None:
        out["perturbed_core_std"] = pd.to_numeric(out[perturbed_col], errors="coerce")
    else:
        out["perturbed_core_std"] = np.nan

    if delta_core_col is not None:
        out["delta_core_std"] = pd.to_numeric(out[delta_core_col], errors="coerce")
    elif baseline_col is not None and perturbed_col is not None:
        out["delta_core_std"] = out["perturbed_core_std"] - out["baseline_core_std"]
    else:
        out["delta_core_std"] = np.nan

    if core_reduction_col is not None:
        out["core_reduction_std"] = pd.to_numeric(out[core_reduction_col], errors="coerce")
    else:
        out["core_reduction_std"] = np.maximum(-out["delta_core_std"], 0)

    if repair_gain_col is not None:
        out["repair_gain_std"] = pd.to_numeric(out[repair_gain_col], errors="coerce")
    elif delta_repair_col is not None:
        out["repair_gain_std"] = np.maximum(pd.to_numeric(out[delta_repair_col], errors="coerce"), 0)
    else:
        out["repair_gain_std"] = np.nan

    if out["perturbed_core_std"].isna().all():
        out["perturbed_core_std"] = out["baseline_core_std"] - out["core_reduction_std"]
    out["perturbed_core_std"] = np.clip(out["perturbed_core_std"], 0, 1)

    out["core_reduction_std"] = np.clip(out["core_reduction_std"], 0, None)
    out["repair_gain_std"] = np.clip(out["repair_gain_std"], 0, None)

    audit = {
        "baseline_col": baseline_col,
        "perturbed_col": perturbed_col,
        "delta_core_col": delta_core_col,
        "core_reduction_col": core_reduction_col,
        "repair_gain_col": repair_gain_col,
        "delta_repair_col": delta_repair_col,
    }
    return out, audit


# -----------------------------
# Figure 1: module/state atlas
# -----------------------------
def make_module_state_atlas(df, outdir, time_col, target_timepoint, grid_n, sigma, show_axes=False):
    if target_timepoint is not None and time_col is not None:
        df_t = df[df[time_col].astype(str) == str(target_timepoint)].copy()
    else:
        df_t = df.copy()

    label_col = infer_state_label_col(df_t)
    state_maps = label_onehot(df_t, label_col)

    maps = []
    maps.extend([
        ("Core probability", state_maps["Core probability"], "prob"),
        ("Peri-infarct probability", state_maps["Peri-infarct probability"], "prob"),
        ("Remote-like probability", state_maps["Remote-like probability"], "prob"),
    ])

    if "plot__Repair score" in df_t.columns:
        repair = df_t["plot__Repair score"].to_numpy(float)
    elif "module__Repair_score" in df_t.columns:
        repair = robust_rescale01(df_t["module__Repair_score"].to_numpy(float), 1, 99)
    else:
        repair = np.zeros(df_t.shape[0])
    maps.append(("Repair score", repair, "prob"))

    for name in ["Ferroptosis", "Inflammation", "BBB leakage", "Reactive astrocyte"]:
        col = f"module__{clean_name(name)}"
        if col in df_t.columns:
            maps.append((name, df_t[col].to_numpy(float), "module"))
        else:
            maps.append((name, np.zeros(df_t.shape[0]), "module"))

    x, y = get_xy(df_t)
    extent = (
        np.nanmin(x) - 0.04 * (np.nanmax(x) - np.nanmin(x)),
        np.nanmax(x) + 0.04 * (np.nanmax(x) - np.nanmin(x)),
        np.nanmin(y) - 0.04 * (np.nanmax(y) - np.nanmin(y)),
        np.nanmax(y) + 0.04 * (np.nanmax(y) - np.nanmin(y)),
    )

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5), facecolor="white")
    fig.subplots_adjust(left=0.04, right=0.90, top=0.86, bottom=0.08, wspace=0.12, hspace=0.18)
    fig.suptitle(
        "Module and state-probability spatial landscape atlas",
        fontsize=24,
        fontweight="bold",
        color="black",
        y=0.96,
    )

    prob_im = None
    module_im = None

    for ax, (title, vals, kind) in zip(axes.ravel(), maps):
        if kind == "prob":
            im, _ = draw_landscape(
                ax, df_t, vals, title=title,
                cmap=CMAP_PROB, vmin=0, vmax=1,
                extent=extent, grid_n=grid_n, sigma=sigma,
                show_axes=show_axes, raw_overlay=True,
                raw_s=1.1, raw_alpha=0.45,
                state_maps=state_maps,
            )
            prob_im = im
        else:
            im, _ = draw_landscape(
                ax, df_t, vals, title=title,
                cmap=CMAP_MODULE, vmin=-2.5, vmax=2.5,
                extent=extent, grid_n=grid_n, sigma=sigma,
                show_axes=show_axes, raw_overlay=True,
                raw_s=0.9, raw_alpha=0.28,
                state_maps=state_maps,
            )
            module_im = im

    cax1 = fig.add_axes([0.925, 0.56, 0.012, 0.28])
    cb1 = fig.colorbar(prob_im, cax=cax1)
    cb1.set_label("State probability", fontsize=10, color="black")
    cb1.ax.tick_params(labelsize=8, colors="black")

    cax2 = fig.add_axes([0.925, 0.18, 0.012, 0.28])
    cb2 = fig.colorbar(module_im, cax=cax2)
    cb2.set_label("Module score (z)", fontsize=10, color="black")
    cb2.ax.tick_params(labelsize=8, colors="black")

    fig.text(
        0.5, 0.035,
        "Smoothed spatial landscape with raw spot overlay. Dashed grey contour indicates tissue boundary; "
        "red/orange contours indicate core/peri regions when available.",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
    )

    paths = save_figure(fig, outdir, "Fig_white_Module_StateProbability_SpatialLandscapeAtlas")
    plt.close(fig)
    return paths


# -----------------------------
# Figure 2: marker atlas
# -----------------------------
def make_marker_atlas(adata, df, outdir, time_col, target_timepoint, grid_n, sigma, show_axes=False):
    if target_timepoint is not None and time_col is not None:
        df_t = df[df[time_col].astype(str) == str(target_timepoint)].copy()
        idx = df.index[df[time_col].astype(str) == str(target_timepoint)].to_numpy()
    else:
        df_t = df.copy()
        idx = np.arange(df.shape[0])

    gene_to_idx = build_gene_index(adata)
    all_genes = [g for genes in MARKER_GROUPS.values() for g in genes]

    expr_dict = {}
    used_genes = []
    for g in all_genes:
        expr = get_gene_expr(adata, g, gene_to_idx=gene_to_idx)
        if expr is None:
            expr_t = np.zeros(df_t.shape[0])
        else:
            expr_t = expr[idx]
        expr_dict[g] = expr_t
        used_genes.append(g)

    all_vals = np.concatenate([np.asarray(expr_dict[g], dtype=float) for g in used_genes])
    vmin, vmax = 0, np.nanpercentile(all_vals[np.isfinite(all_vals)], 99.3)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    label_col = infer_state_label_col(df_t)
    state_maps = label_onehot(df_t, label_col)

    x, y = get_xy(df_t)
    extent = (
        np.nanmin(x) - 0.04 * (np.nanmax(x) - np.nanmin(x)),
        np.nanmax(x) + 0.04 * (np.nanmax(x) - np.nanmin(x)),
        np.nanmin(y) - 0.04 * (np.nanmax(y) - np.nanmin(y)),
        np.nanmax(y) + 0.04 * (np.nanmax(y) - np.nanmin(y)),
    )

    nrows = len(MARKER_GROUPS)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(15.5, 14.5), facecolor="white")
    fig.subplots_adjust(left=0.14, right=0.90, top=0.91, bottom=0.07, wspace=0.10, hspace=0.16)
    fig.suptitle(
        "Spatial marker landscape atlas",
        fontsize=24,
        fontweight="bold",
        color="black",
        y=0.965,
    )

    last_im = None
    for r, (group, genes) in enumerate(MARKER_GROUPS.items()):
        for c, gene in enumerate(genes):
            ax = axes[r, c]
            vals = expr_dict[gene]
            im, _ = draw_landscape(
                ax, df_t, vals, title=gene,
                cmap=CMAP_EXPR, vmin=vmin, vmax=vmax,
                extent=extent, grid_n=grid_n, sigma=sigma,
                show_axes=show_axes, raw_overlay=True,
                raw_s=0.9, raw_alpha=0.40,
                state_maps=state_maps,
            )
            last_im = im

        axes[r, 0].text(
            -0.20, 0.5, group,
            transform=axes[r, 0].transAxes,
            rotation=0,
            ha="right",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    cax = fig.add_axes([0.925, 0.20, 0.012, 0.56])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Expression", fontsize=10, color="black")
    cb.ax.tick_params(labelsize=8, colors="black")

    fig.text(
        0.5, 0.035,
        "Smoothed spatial landscape with raw spot overlay; dashed outline indicates tissue boundary; "
        "colored contours indicate core/peri regions when available.",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
    )

    paths = save_figure(fig, outdir, "Fig_white_Spatial_Marker_LandscapeAtlas")
    plt.close(fig)
    return paths


# -----------------------------
# Figure 3: perturbation rescue atlas
# -----------------------------
def make_rescue_atlas(df, resp, outdir, time_col, target_timepoint, grid_n, sigma, show_axes=False):
    if resp is None:
        return {}

    resp, col_audit = standardize_response_columns(resp)

    if target_timepoint is not None and time_col is not None:
        df_t = df[df[time_col].astype(str) == str(target_timepoint)].copy()
    else:
        df_t = df.copy()

    # Merge response to spatial metadata.
    meta = df_t[["_adata_id_", "obs_name", "spatial_x", "spatial_y"]].copy()
    if time_col is not None and time_col in df_t.columns:
        meta[time_col] = df_t[time_col].astype(str)

    resp["_response_id_"] = resp["_response_id_"].astype(str)

    merged1 = resp.merge(
        meta,
        left_on="_response_id_",
        right_on="_adata_id_",
        how="inner",
        suffixes=("", "_meta"),
    )
    merged2 = resp.merge(
        meta,
        left_on="_response_id_",
        right_on="obs_name",
        how="inner",
        suffixes=("", "_meta"),
    )
    m = merged1 if merged1.shape[0] >= merged2.shape[0] else merged2

    if m.empty:
        raise ValueError("Step78 response table could not be merged to h5ad/state metadata by _adata_id_ or obs_name.")

    if time_col is not None and time_col in m.columns and target_timepoint is not None:
        m = m[m[time_col].astype(str) == str(target_timepoint)].copy()

    label_col = infer_state_label_col(df_t)
    state_maps = label_onehot(df_t, label_col)

    # Common extent from baseline df_t.
    x0, y0 = get_xy(df_t)
    extent = (
        np.nanmin(x0) - 0.04 * (np.nanmax(x0) - np.nanmin(x0)),
        np.nanmax(x0) + 0.04 * (np.nanmax(x0) - np.nanmin(x0)),
        np.nanmin(y0) - 0.04 * (np.nanmax(y0) - np.nanmin(y0)),
        np.nanmax(y0) + 0.04 * (np.nanmax(y0) - np.nanmin(y0)),
    )

    nrows = len(CANDIDATES)
    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 17), facecolor="white")
    fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.06, wspace=0.08, hspace=0.14)
    fig.suptitle(
        "Color-consistent before/after virtual perturbation spatial rescue atlas",
        fontsize=23,
        fontweight="bold",
        color="black",
        y=0.965,
    )

    col_titles = ["Baseline core map", "Perturbed core map", "Repair gain", "Core reduction"]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=12, fontweight="bold", color="black", pad=8)

    im_core = None
    im_repair = None
    im_reduction = None

    for r, cand in enumerate(CANDIDATES):
        sub = m[m["candidate_std"] == cand].copy()

        axes[r, 0].text(
            -0.18, 0.5, cand,
            transform=axes[r, 0].transAxes,
            ha="right",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="black",
        )

        if sub.empty:
            for c in range(ncols):
                ax = axes[r, c]
                ax.text(
                    0.5, 0.5, "No usable\nresponse table",
                    ha="center", va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="0.35",
                )
                apply_clean_axis(ax, show_axes=False)
            continue

        df_plot = sub.rename(columns={"spatial_x": "spatial_x", "spatial_y": "spatial_y"}).copy()

        vals_list = [
            ("baseline_core_std", CMAP_PROB, 0, 1),
            ("perturbed_core_std", CMAP_PROB, 0, 1),
            ("repair_gain_std", CMAP_REPAIR, 0, max(0.30, np.nanpercentile(m["repair_gain_std"], 99))),
            ("core_reduction_std", CMAP_CORE_REDUCTION, 0, max(0.30, np.nanpercentile(m["core_reduction_std"], 99))),
        ]

        for c, (vcol, cmap, vmin, vmax) in enumerate(vals_list):
            ax = axes[r, c]
            vals = pd.to_numeric(df_plot[vcol], errors="coerce").to_numpy(float)
            im, _ = draw_landscape(
                ax, df_plot, vals, title=None,
                cmap=cmap, vmin=vmin, vmax=vmax,
                extent=extent, grid_n=grid_n, sigma=sigma,
                show_axes=show_axes, raw_overlay=True,
                raw_s=0.8, raw_alpha=0.32,
                state_maps=state_maps,
            )
            if c in [0, 1]:
                im_core = im
            elif c == 2:
                im_repair = im
            elif c == 3:
                im_reduction = im

    cax1 = fig.add_axes([0.905, 0.70, 0.012, 0.18])
    cb1 = fig.colorbar(im_core, cax=cax1)
    cb1.set_label("Core probability", fontsize=10, color="black")
    cb1.ax.tick_params(labelsize=8, colors="black")

    cax2 = fig.add_axes([0.905, 0.43, 0.012, 0.18])
    cb2 = fig.colorbar(im_repair, cax=cax2)
    cb2.set_label("Repair gain", fontsize=10, color="black")
    cb2.ax.tick_params(labelsize=8, colors="black")

    cax3 = fig.add_axes([0.905, 0.17, 0.012, 0.18])
    cb3 = fig.colorbar(im_reduction, cax=cax3)
    cb3.set_label("Core reduction", fontsize=10, color="black")
    cb3.ax.tick_params(labelsize=8, colors="black")

    fig.text(
        0.5, 0.025,
        "Maps show visualization-level virtual perturbation/state-editing proxies; baseline and perturbed core maps share the same color scale. "
        "These are not wet-lab KO/blockade results or observed cell-state transitions.",
        ha="center",
        va="center",
        fontsize=9,
        color="black",
    )

    paths = save_figure(fig, outdir, "Fig_white_VirtualPerturbation_SpatialRescueAtlas")
    plt.close(fig)

    return {"paths": paths, "column_audit": col_audit}


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--step78_dir", default="")
    parser.add_argument("--step78_cell_table", default="")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--spatial_key", default="spatial")
    parser.add_argument("--timepoint", default="D7")
    parser.add_argument("--grid_n", type=int, default=360)
    parser.add_argument("--sigma", type=float, default=1.35)
    parser.add_argument("--flip_y", action="store_true")
    parser.add_argument("--show_axes", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    mkdir(args.outdir)

    print("=" * 100)
    print("White-background redraw for Step75/Step78 spatial atlas figures")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"step78_dir={args.step78_dir}")
    print(f"step78_cell_table={args.step78_cell_table}")
    print(f"outdir={args.outdir}")

    adata, df = load_adata_and_metadata(
        args.h5ad,
        args.state_table,
        spatial_key=args.spatial_key,
        flip_y=args.flip_y,
    )
    df, module_used = add_module_scores_to_df(adata, df)

    time_col = infer_time_col(df)
    target_timepoint = choose_timepoint(df, time_col, args.timepoint)

    if time_col is None:
        print("[WARN] No time column found; plotting all spots.")
    else:
        print(f"time_col={time_col}; selected_timepoint={target_timepoint}")

    label_col = infer_state_label_col(df)
    print(f"state_label_col={label_col}")

    outputs = OrderedDict()

    print("\n[1/3] Module/state-probability atlas...")
    outputs["module_state_atlas"] = make_module_state_atlas(
        df=df,
        outdir=args.outdir,
        time_col=time_col,
        target_timepoint=target_timepoint,
        grid_n=args.grid_n,
        sigma=args.sigma,
        show_axes=args.show_axes,
    )

    print("[2/3] Spatial marker atlas...")
    outputs["marker_atlas"] = make_marker_atlas(
        adata=adata,
        df=df,
        outdir=args.outdir,
        time_col=time_col,
        target_timepoint=target_timepoint,
        grid_n=args.grid_n,
        sigma=args.sigma,
        show_axes=args.show_axes,
    )

    print("[3/3] Virtual perturbation rescue atlas...")
    resp, resp_audit = load_step78_response(
        step78_dir=args.step78_dir,
        step78_cell_table=args.step78_cell_table,
    )
    outputs["step78_response_audit"] = resp_audit

    if resp is None:
        print("[WARN] No usable Step78 per-cell/per-spot response table found. Rescue atlas skipped.")
        print("       Please pass --step78_cell_table explicitly.")
        outputs["rescue_atlas"] = {}
    else:
        outputs["rescue_atlas"] = make_rescue_atlas(
            df=df,
            resp=resp,
            outdir=args.outdir,
            time_col=time_col,
            target_timepoint=target_timepoint,
            grid_n=args.grid_n,
            sigma=args.sigma,
            show_axes=args.show_axes,
        )

    report = {
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "step78_dir": args.step78_dir,
        "step78_cell_table": args.step78_cell_table,
        "outdir": args.outdir,
        "time_col": time_col,
        "selected_timepoint": target_timepoint,
        "state_label_col": label_col,
        "grid_n": args.grid_n,
        "sigma": args.sigma,
        "module_genes_used": module_used,
        "outputs": outputs,
        "interpretation_note": (
            "These are white-background manuscript redraws of smoothed spatial landscape visualizations. "
            "State maps are label-derived one-hot spatial smooths unless a model-derived repair score is available. "
            "Virtual perturbation maps are visualization-level computational proxies and are not wet-lab perturbation results."
        ),
    }

    report_path = os.path.join(args.outdir, "white_background_spatial_atlas_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"Report: {report_path}")
    print("Generated files:")
    for k, v in outputs.items():
        if isinstance(v, dict):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
