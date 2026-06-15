#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step75B | Dark manuscript-grade spatial landscape atlas

This script generates:
1. Dark spatial marker atlas
2. Dark spatial module atlas
3. Dark spatial state-probability atlas
4. Compact manuscript-grade spatial marker/module/state panel

Key improvements over Step75:
- robust coordinate detection from obs or obsm["spatial"]
- robust state-table merge
- supports core_probability / peri_probability / remote_probability
- tissue outline from spatial occupancy density
- lesion/core and peri outlines from state probabilities or labels
- smoothed spatial landscape layer + raw spot overlay
- unified black-background visual style
"""

from pathlib import Path
import argparse
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

from scipy import sparse
from scipy.ndimage import gaussian_filter


# =============================================================================
# Default marker and module definitions
# =============================================================================

DEFAULT_MARKER_GROUPS = {
    "Microglia / myeloid": ["C1qa", "C1qb", "Tyrobp", "Lgals3"],
    "Endothelial / BBB": ["Kdr", "Cldn5", "Pecam1", "Kcnj8"],
    "Ferroptosis / hypoxia": ["Hmox1", "Fth1", "Slc7a11", "Gpx4"],
    "Repair / ECM": ["Col1a1", "Col3a1", "Fn1", "Spp1"],
}

DEFAULT_MODULE_GENESETS = {
    "Repair / ECM": ["Col1a1", "Col3a1", "Fn1", "Spp1", "Vim", "Apoe", "Postn", "Timp1"],
    "Ferroptosis": ["Hmox1", "Fth1", "Slc7a11", "Gpx4", "Tfrc", "Acsl4", "Ptgs2"],
    "Microglia / inflammation": ["C1qa", "C1qb", "Tyrobp", "Lgals3", "Ctsd", "Ccl2", "Ccl3"],
    "BBB / endothelial": ["Kdr", "Cldn5", "Pecam1", "Kcnj8", "Vwf", "Flt1", "Rgs5"],
    "Reactive astrocyte": ["Gfap", "Aqp4", "Vim", "Clu", "Apoe", "Serpina3n", "Lcn2"],
    "Hypoxia / redox": ["Hmox1", "Fth1", "Nfe2l1", "Nfe2l2", "Hif1a", "Vegfa", "Sod2"],
}


# =============================================================================
# Basic utilities
# =============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_name(x):
    x = str(x)
    x = re.sub(r"[^\w\-\.]+", "_", x)
    return x.strip("_")


def read_table(path):
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def read_json(path):
    if path is None or str(path).strip() == "":
        return None
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing(cols, candidates):
    cols = list(cols)
    lower_map = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if str(c).lower() in lower_map:
            return lower_map[str(c).lower()]
    return None


def dense_vector(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def robust_limits(x, q_low=0.01, q_high=0.99, min_span=1e-6):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    lo = np.quantile(x, q_low)
    hi = np.quantile(x, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < min_span:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < min_span:
        lo, hi = 0.0, 1.0
    return float(lo), float(hi)


def zscore_clip(x, clip=2.5):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x)
    z = (x - mu) / sd
    z = np.clip(z, -clip, clip)
    z[~np.isfinite(z)] = 0.0
    return z


# =============================================================================
# Colormaps
# =============================================================================

def cmap_dark_expr():
    return LinearSegmentedColormap.from_list(
        "dark_expr",
        ["#02030a", "#1a0b3b", "#4a1486", "#9c2c93", "#e8669e", "#ffd67a"]
    )


def cmap_dark_green():
    return LinearSegmentedColormap.from_list(
        "dark_green",
        ["#02030a", "#092414", "#0f5132", "#2f9e44", "#a3e635", "#fff7ad"]
    )


def cmap_dark_blue():
    return LinearSegmentedColormap.from_list(
        "dark_blue",
        ["#02030a", "#071d3a", "#0b4f8a", "#2b8cbe", "#a6cee3", "#f0f9ff"]
    )


def cmap_dark_red():
    return LinearSegmentedColormap.from_list(
        "dark_red",
        ["#02030a", "#2a0611", "#5f0f1b", "#b11226", "#f26d5b", "#ffe0b2"]
    )


def cmap_dark_diverging():
    return LinearSegmentedColormap.from_list(
        "dark_div",
        ["#2166ac", "#67a9cf", "#05070b", "#ef8a62", "#b2182b"]
    )


# =============================================================================
# h5ad coordinate and metadata handling
# =============================================================================

def get_obs_id_series(adata, obs, user_col=""):
    if user_col and user_col in obs.columns:
        return obs[user_col].astype(str), user_col

    candidates = [
        "obs_name", "cell_id", "spot_id", "barcode", "CellID", "SpotID",
        "cell", "spot", "id"
    ]
    col = first_existing(obs.columns, candidates)
    if col is not None:
        return obs[col].astype(str), col

    return pd.Series(adata.obs_names.astype(str), index=obs.index), "_obs_names_"


def get_xy(adata, obs, xcol="", ycol=""):
    if xcol and ycol and xcol in obs.columns and ycol in obs.columns:
        x = pd.to_numeric(obs[xcol], errors="coerce").to_numpy()
        y = pd.to_numeric(obs[ycol], errors="coerce").to_numpy()
        return x, y, xcol, ycol, "obs_user"

    candidate_pairs = [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
        ("coord_x", "coord_y"),
        ("x_coord", "y_coord"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("baseline_coord_x_from_ref", "baseline_coord_y_from_ref"),
        ("UMAP_1", "UMAP_2"),
        ("umap_1", "umap_2"),
    ]

    for xc, yc in candidate_pairs:
        if xc in obs.columns and yc in obs.columns:
            x = pd.to_numeric(obs[xc], errors="coerce").to_numpy()
            y = pd.to_numeric(obs[yc], errors="coerce").to_numpy()
            if np.isfinite(x).sum() > 10 and np.isfinite(y).sum() > 10:
                return x, y, xc, yc, "obs_auto"

    if "spatial" in adata.obsm:
        arr = np.asarray(adata.obsm["spatial"])
        if arr.ndim == 2 and arr.shape[1] >= 2:
            x = arr[:, 0].astype(float)
            y = arr[:, 1].astype(float)
            return x, y, "obsm_spatial_0", "obsm_spatial_1", "obsm_spatial"

    numeric_like = []
    for c in obs.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["spatial", "coord", "array", "image", "pxl", "umap", "x", "y"]):
            vals = pd.to_numeric(obs[c], errors="coerce")
            if vals.notna().sum() > 10:
                numeric_like.append(c)
    if len(numeric_like) >= 2:
        xc, yc = numeric_like[0], numeric_like[1]
        x = pd.to_numeric(obs[xc], errors="coerce").to_numpy()
        y = pd.to_numeric(obs[yc], errors="coerce").to_numpy()
        return x, y, xc, yc, "obs_numeric_fallback"

    raise ValueError("Cannot infer x/y spatial coordinates. Please pass --xcol and --ycol.")


def merge_state_table(meta, state_df, adata_id_col="_adata_id_", user_state_id_col=""):
    audit = {
        "state_table_available": False,
        "state_table_rows": 0,
        "merge_mode": "none",
        "merge_key_meta": "",
        "merge_key_state": "",
        "state_merge_overlap": 0,
    }

    if state_df is None or state_df.empty:
        return meta, audit

    audit["state_table_available"] = True
    audit["state_table_rows"] = int(len(state_df))

    st = state_df.copy()

    if user_state_id_col and user_state_id_col in st.columns:
        state_id_col = user_state_id_col
    else:
        state_id_col = first_existing(
            st.columns,
            [
                "obs_name", "cell_id", "spot_id", "barcode", "CellID", "SpotID",
                "cell", "spot", "id", "_obs_names_", "_adata_id_"
            ]
        )

    if state_id_col is not None:
        left = meta[adata_id_col].astype(str)
        right = st[state_id_col].astype(str)
        overlap = len(set(left).intersection(set(right)))
        audit["state_merge_overlap"] = int(overlap)

        if overlap > 0:
            st["_merge_key_state_"] = right
            meta["_merge_key_state_"] = left
            st = st.drop_duplicates("_merge_key_state_")
            out = meta.merge(st, on="_merge_key_state_", how="left", suffixes=("", "_state"))
            out.drop(columns=["_merge_key_state_"], inplace=True, errors="ignore")
            audit["merge_mode"] = "id"
            audit["merge_key_meta"] = adata_id_col
            audit["merge_key_state"] = state_id_col
            return out, audit

    if len(st) == len(meta):
        st = st.copy()
        st["_row_order_state_"] = np.arange(len(st))
        meta = meta.copy()
        meta["_row_order_state_"] = np.arange(len(meta))
        out = meta.merge(st, on="_row_order_state_", how="left", suffixes=("", "_state"))
        out.drop(columns=["_row_order_state_"], inplace=True, errors="ignore")
        audit["merge_mode"] = "row_order"
        audit["merge_key_meta"] = "row_order"
        audit["merge_key_state"] = "row_order"
        audit["state_merge_overlap"] = int(len(meta))
        return out, audit

    audit["merge_mode"] = "failed"
    return meta, audit


def infer_prob_cols(meta, core_col="", peri_col="", remote_col=""):
    cols = list(meta.columns)
    lower = {str(c).lower(): c for c in cols}

    def pick(user, candidates):
        if user and user in cols:
            return user
        for c in candidates:
            if c in cols:
                return c
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    core = pick(core_col, [
        "core_probability",
        "peri_remote_core_probability",
        "lesion_core_probability",
        "lesion-core-like_probability",
        "lesion_core_like_probability",
        "core_prob",
        "p_core",
        "prob_core",
        "pred_core_probability",
        "region_core_probability",
        "core_like_prob",
        "core",
    ])

    peri = pick(peri_col, [
        "peri_probability",
        "peri_infarct_probability",
        "peri-infarct_probability",
        "peri_like_probability",
        "peri_prob",
        "p_peri",
        "prob_peri",
        "pred_peri_probability",
        "region_peri_probability",
        "peri",
    ])

    remote = pick(remote_col, [
        "remote_probability",
        "remote_like_probability",
        "remote-like_probability",
        "remote_prob",
        "p_remote",
        "prob_remote",
        "pred_remote_probability",
        "region_remote_probability",
        "remote",
    ])

    return {"core": core, "peri": peri, "remote": remote}


# =============================================================================
# Gene expression and module scores
# =============================================================================

def gene_lookup(adata):
    mapping = {}
    for v in adata.var_names.astype(str):
        mapping[v.upper()] = v

    for col in ["gene", "genes", "gene_symbol", "symbol", "feature_name", "gene_name"]:
        if col in adata.var.columns:
            for var_name, val in zip(adata.var_names.astype(str), adata.var[col].astype(str)):
                key = str(val).upper()
                if key and key not in mapping:
                    mapping[key] = var_name

    return mapping


def get_gene_expr(adata, gene, gmap):
    var = gmap.get(str(gene).upper(), None)
    if var is None:
        return np.full(adata.n_obs, np.nan)
    return dense_vector(adata[:, var].X).astype(float)


def compute_module_scores(adata, module_sets, gmap, min_genes=2):
    scores = {}
    audit_rows = []
    for module, genes in module_sets.items():
        matched = []
        exprs = []
        for g in genes:
            var = gmap.get(str(g).upper(), None)
            if var is not None:
                matched.append(g)
                exprs.append(get_gene_expr(adata, g, gmap))

        audit_rows.append({
            "module": module,
            "n_input_genes": len(genes),
            "n_matched_genes": len(matched),
            "matched_genes": ";".join(matched),
        })

        if len(exprs) < min_genes:
            scores[module] = np.full(adata.n_obs, np.nan)
        else:
            M = np.vstack(exprs).T
            Z = np.zeros_like(M, dtype=float)
            for j in range(M.shape[1]):
                Z[:, j] = zscore_clip(M[:, j], clip=2.5)
            scores[module] = np.nanmean(Z, axis=1)

    return scores, pd.DataFrame(audit_rows)


# =============================================================================
# Spatial smoothing and outlines
# =============================================================================

def spatial_grid_smooth(x, y, values=None, grid_size=260, sigma=2.2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)
    if values is not None:
        values = np.asarray(values, dtype=float)
        ok = ok & np.isfinite(values)

    x0, x1 = np.nanmin(x[ok]), np.nanmax(x[ok])
    y0, y1 = np.nanmin(y[ok]), np.nanmax(y[ok])
    dx = x1 - x0
    dy = y1 - y0
    pad_x = dx * 0.04
    pad_y = dy * 0.04
    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y

    xi = np.linspace(x0, x1, grid_size)
    yi = np.linspace(y0, y1, grid_size)

    gx = np.clip(((x[ok] - x0) / (x1 - x0) * (grid_size - 1)).astype(int), 0, grid_size - 1)
    gy = np.clip(((y[ok] - y0) / (y1 - y0) * (grid_size - 1)).astype(int), 0, grid_size - 1)

    count = np.zeros((grid_size, grid_size), dtype=float)
    np.add.at(count, (gy, gx), 1.0)
    count_s = gaussian_filter(count, sigma=sigma)

    if values is None:
        val_s = count_s
    else:
        num = np.zeros((grid_size, grid_size), dtype=float)
        np.add.at(num, (gy, gx), values[ok])
        num_s = gaussian_filter(num, sigma=sigma)
        val_s = num_s / np.maximum(count_s, 1e-8)

    extent = [x0, x1, y0, y1]
    return val_s, count_s, extent


def draw_tissue_outline(ax, x, y, color="white", alpha=0.8, lw=0.8, grid_size=260):
    occ, count_s, extent = spatial_grid_smooth(x, y, values=None, grid_size=grid_size, sigma=2.0)
    if np.nanmax(count_s) <= 0:
        return
    level = np.nanquantile(count_s[count_s > 0], 0.08)
    try:
        ax.contour(
            count_s,
            levels=[level],
            extent=extent,
            colors=color,
            linewidths=lw,
            alpha=alpha,
            linestyles="dashed"
        )
    except Exception:
        pass


def draw_probability_outline(ax, x, y, prob, level=0.5, color="#f8fafc", lw=1.0, alpha=0.9, grid_size=260):
    prob = np.asarray(prob, dtype=float)
    if np.isfinite(prob).sum() < 10:
        return
    sm, cnt, extent = spatial_grid_smooth(x, y, values=prob, grid_size=grid_size, sigma=2.2)
    if np.nanmax(sm) < level:
        level = np.nanquantile(sm[np.isfinite(sm)], 0.85)
    try:
        ax.contour(
            sm,
            levels=[level],
            extent=extent,
            colors=color,
            linewidths=lw,
            alpha=alpha,
            linestyles="solid"
        )
    except Exception:
        pass


def draw_label_outline(ax, x, y, labels, target_terms, color="#f8fafc", lw=1.0, alpha=0.85, grid_size=260):
    if labels is None:
        return
    s = pd.Series(labels).astype(str).str.lower()
    mask = np.zeros(len(s), dtype=float)
    for term in target_terms:
        mask = np.maximum(mask, s.str.contains(term.lower(), regex=False).astype(float).to_numpy())
    if mask.sum() < 10:
        return
    sm, cnt, extent = spatial_grid_smooth(x, y, values=mask, grid_size=grid_size, sigma=2.2)
    try:
        ax.contour(
            sm,
            levels=[0.45],
            extent=extent,
            colors=color,
            linewidths=lw,
            alpha=alpha,
            linestyles="solid"
        )
    except Exception:
        pass


def find_label_col(meta):
    for c in ["state_group", "region_refined", "region_manual_final", "region_auto", "condition", "timepoint"]:
        if c in meta.columns:
            return c
    return ""


# =============================================================================
# Plot helpers
# =============================================================================

def setup_dark_ax(ax):
    ax.set_facecolor("#05070b")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_spatial_landscape(
    ax,
    x,
    y,
    values,
    cmap,
    norm,
    title="",
    point_size=2.2,
    smooth=True,
    grid_size=260,
    sigma=2.2,
    tissue_outline=True,
    core_prob=None,
    peri_prob=None,
    label_values=None,
    dark=True,
):
    setup_dark_ax(ax)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y)

    if smooth and np.isfinite(values).sum() > 10:
        sm, cnt, extent = spatial_grid_smooth(x, y, values, grid_size=grid_size, sigma=sigma)
        alpha_mask = cnt / np.nanmax(cnt) if np.nanmax(cnt) > 0 else cnt
        alpha_mask = np.clip(alpha_mask, 0, 1)
        ax.imshow(
            sm,
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=0.70 * alpha_mask,
            interpolation="bilinear",
            zorder=1
        )

    good = ok & np.isfinite(values)
    ax.scatter(
        x[good],
        y[good],
        c=values[good],
        cmap=cmap,
        norm=norm,
        s=point_size,
        linewidths=0,
        alpha=0.78,
        rasterized=True,
        zorder=2
    )

    if tissue_outline:
        draw_tissue_outline(ax, x[ok], y[ok], color="white", alpha=0.65, lw=0.7, grid_size=grid_size)

    if core_prob is not None and np.isfinite(np.asarray(core_prob, dtype=float)).sum() > 10:
        draw_probability_outline(ax, x, y, core_prob, level=0.50, color="#ff4d6d", lw=1.0, alpha=0.85, grid_size=grid_size)

    if peri_prob is not None and np.isfinite(np.asarray(peri_prob, dtype=float)).sum() > 10:
        draw_probability_outline(ax, x, y, peri_prob, level=0.45, color="#ffd166", lw=0.9, alpha=0.80, grid_size=grid_size)

    if label_values is not None:
        draw_label_outline(ax, x, y, label_values, ["core", "lesion"], color="#ff4d6d", lw=1.0, alpha=0.80, grid_size=grid_size)
        draw_label_outline(ax, x, y, label_values, ["peri"], color="#ffd166", lw=0.9, alpha=0.75, grid_size=grid_size)

    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=9.5, color="white", fontweight="bold", pad=3)


def add_cbar(fig, axes, cmap, norm, label, color="white"):
    boxes = [ax.get_position() for ax in axes if ax is not None]
    x1 = max(b.x1 for b in boxes)
    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)
    cax = fig.add_axes([x1 + 0.010, y0 + 0.02, 0.012, y1 - y0 - 0.04])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(colors=color, labelsize=7)
    cb.outline.set_edgecolor(color)
    cb.set_label(label, color=color, fontsize=8)
    return cb


# =============================================================================
# Figure builders
# =============================================================================

def make_marker_atlas(meta, adata, x, y, marker_groups, gmap, prob_cols, outbase, point_size=2.1, dpi=600):
    n_rows = len(marker_groups)
    n_cols = max(len(v) for v in marker_groups.values())
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.7 * n_cols, 2.45 * n_rows),
        facecolor="#05070b"
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    cmap = cmap_dark_expr()
    values_all = []
    expr_cache = {}

    for group, genes in marker_groups.items():
        for g in genes:
            v = get_gene_expr(adata, g, gmap)
            expr_cache[g] = v
            if np.isfinite(v).sum() > 0:
                values_all.append(v[np.isfinite(v)])

    if values_all:
        cat = np.concatenate(values_all)
        lo, hi = robust_limits(cat, 0.02, 0.995)
    else:
        lo, hi = 0, 1
    norm = Normalize(lo, hi)

    core_prob = meta[prob_cols["core"]].to_numpy(dtype=float) if prob_cols.get("core") in meta.columns else None
    peri_prob = meta[prob_cols["peri"]].to_numpy(dtype=float) if prob_cols.get("peri") in meta.columns else None
    label_col = find_label_col(meta)
    labels = meta[label_col].astype(str).to_numpy() if label_col else None

    used_axes = []
    for i, (group, genes) in enumerate(marker_groups.items()):
        for j in range(n_cols):
            ax = axes[i, j]
            if j >= len(genes):
                ax.axis("off")
                continue
            g = genes[j]
            plot_spatial_landscape(
                ax, x, y, expr_cache[g],
                cmap=cmap, norm=norm,
                title=g,
                point_size=point_size,
                smooth=True,
                tissue_outline=True,
                core_prob=core_prob,
                peri_prob=peri_prob,
                label_values=labels,
            )
            used_axes.append(ax)

        axes[i, 0].text(
            -0.09, 0.5, group,
            transform=axes[i, 0].transAxes,
            ha="right", va="center",
            fontsize=9.5,
            color="white",
            fontweight="bold"
        )

    fig.suptitle("Spatial marker landscape atlas", fontsize=19, color="white", fontweight="bold", y=0.992)
    add_cbar(fig, used_axes, cmap, norm, "Expression", color="white")

    fig.text(
        0.5, 0.012,
        "Smoothed spatial landscape with raw spot overlay; dashed outline indicates tissue boundary; colored contours indicate core/peri regions when available.",
        ha="center", va="bottom",
        fontsize=7.5,
        color="#e5e7eb"
    )

    fig.subplots_adjust(left=0.13, right=0.92, top=0.94, bottom=0.055, wspace=0.05, hspace=0.16)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{outbase}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_module_atlas(meta, x, y, module_scores, prob_cols, outbase, point_size=2.1, dpi=600):
    modules = list(module_scores.keys())
    n_cols = 3
    n_rows = math.ceil(len(modules) / n_cols)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2 * n_cols, 2.75 * n_rows),
        facecolor="#05070b"
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    cmap = cmap_dark_diverging()
    all_vals = []
    score_cache = {}

    for m, v in module_scores.items():
        vv = zscore_clip(v, clip=2.5)
        score_cache[m] = vv
        all_vals.append(vv[np.isfinite(vv)])

    cat = np.concatenate(all_vals) if all_vals else np.array([-1, 1])
    vmax = max(1.2, np.quantile(np.abs(cat), 0.98))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    core_prob = meta[prob_cols["core"]].to_numpy(dtype=float) if prob_cols.get("core") in meta.columns else None
    peri_prob = meta[prob_cols["peri"]].to_numpy(dtype=float) if prob_cols.get("peri") in meta.columns else None
    label_col = find_label_col(meta)
    labels = meta[label_col].astype(str).to_numpy() if label_col else None

    used_axes = []
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]
            if k >= len(modules):
                ax.axis("off")
                k += 1
                continue
            m = modules[k]
            plot_spatial_landscape(
                ax, x, y, score_cache[m],
                cmap=cmap, norm=norm,
                title=m,
                point_size=point_size,
                smooth=True,
                tissue_outline=True,
                core_prob=core_prob,
                peri_prob=peri_prob,
                label_values=labels,
            )
            used_axes.append(ax)
            k += 1

    fig.suptitle("Spatial module activity landscapes", fontsize=19, color="white", fontweight="bold", y=0.992)
    add_cbar(fig, used_axes, cmap, norm, "Module score (z)", color="white")

    fig.text(
        0.5, 0.012,
        "Module scores are computed from curated gene sets; smoothed landscapes are for visualization and do not imply functional validation.",
        ha="center", va="bottom",
        fontsize=7.5,
        color="#e5e7eb"
    )

    fig.subplots_adjust(left=0.035, right=0.91, top=0.91, bottom=0.065, wspace=0.05, hspace=0.18)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{outbase}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_state_atlas(meta, x, y, prob_cols, outbase, point_size=2.1, dpi=600):
    states = [
        ("core", "lesion-core-like", cmap_dark_red()),
        ("peri", "peri-infarct", cmap_dark_expr()),
        ("remote", "remote-like", cmap_dark_blue()),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9.9, 3.45), facecolor="#05070b")
    axes = np.asarray(axes).reshape(1, 3)[0]

    norm = Normalize(0, 1)

    core_prob = meta[prob_cols["core"]].to_numpy(dtype=float) if prob_cols.get("core") in meta.columns else None
    peri_prob = meta[prob_cols["peri"]].to_numpy(dtype=float) if prob_cols.get("peri") in meta.columns else None
    label_col = find_label_col(meta)
    labels = meta[label_col].astype(str).to_numpy() if label_col else None

    used_axes = []
    for ax, (key, title, cmap) in zip(axes, states):
        col = prob_cols.get(key)
        if col is None or col not in meta.columns:
            setup_dark_ax(ax)
            ax.text(0.5, 0.5, f"{title}\\nmissing", ha="center", va="center",
                    color="white", fontsize=12, transform=ax.transAxes)
            continue
        v = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
        plot_spatial_landscape(
            ax, x, y, v,
            cmap=cmap,
            norm=norm,
            title=title,
            point_size=point_size,
            smooth=True,
            tissue_outline=True,
            core_prob=core_prob,
            peri_prob=peri_prob,
            label_values=labels,
        )
        used_axes.append(ax)

    fig.suptitle("Spatial state-probability landscapes", fontsize=18, color="white", fontweight="bold", y=0.992)

    # single representative probability colorbar
    if used_axes:
        add_cbar(fig, used_axes, cmap_dark_red(), norm, "Probability", color="white")

    fig.text(
        0.5, 0.012,
        "State probabilities are from the StrokeNiche dynamics/state model; contours denote core/peri regions when available.",
        ha="center", va="bottom",
        fontsize=7.5,
        color="#e5e7eb"
    )

    fig.subplots_adjust(left=0.035, right=0.91, top=0.82, bottom=0.11, wspace=0.04)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{outbase}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_compact_panel(meta, adata, x, y, marker_groups, module_scores, gmap, prob_cols, outbase, point_size=1.8, dpi=600):
    """
    Compact manuscript panel:
    A: selected marker genes
    B: module landscapes
    C: state probabilities
    """

    selected_markers = [
        ("C1qa", "Microglia"),
        ("Lgals3", "Myeloid"),
        ("Cldn5", "BBB"),
        ("Spp1", "Repair"),
        ("Hmox1", "Hypoxia"),
        ("Fth1", "Iron/redox"),
    ]

    selected_modules = [
        "Repair / ECM",
        "Microglia / inflammation",
        "Ferroptosis",
        "BBB / endothelial",
    ]

    selected_states = [
        ("core", "Core-like", cmap_dark_red()),
        ("peri", "Peri-infarct", cmap_dark_expr()),
        ("remote", "Remote-like", cmap_dark_blue()),
    ]

    fig = plt.figure(figsize=(13.5, 8.2), facecolor="#05070b")
    gs = fig.add_gridspec(
        3, 6,
        height_ratios=[1.0, 1.0, 0.78],
        hspace=0.20,
        wspace=0.05
    )

    core_prob = meta[prob_cols["core"]].to_numpy(dtype=float) if prob_cols.get("core") in meta.columns else None
    peri_prob = meta[prob_cols["peri"]].to_numpy(dtype=float) if prob_cols.get("peri") in meta.columns else None
    label_col = find_label_col(meta)
    labels = meta[label_col].astype(str).to_numpy() if label_col else None

    expr_cmap = cmap_dark_expr()
    expr_vals = []
    expr_cache = {}
    for g, _ in selected_markers:
        v = get_gene_expr(adata, g, gmap)
        expr_cache[g] = v
        if np.isfinite(v).sum() > 0:
            expr_vals.append(v[np.isfinite(v)])
    lo, hi = robust_limits(np.concatenate(expr_vals), 0.02, 0.995) if expr_vals else (0, 1)
    expr_norm = Normalize(lo, hi)

    # A markers: row 0, six columns
    marker_axes = []
    for j, (g, lab) in enumerate(selected_markers):
        ax = fig.add_subplot(gs[0, j])
        plot_spatial_landscape(
            ax, x, y, expr_cache[g],
            cmap=expr_cmap,
            norm=expr_norm,
            title=f"{g}\\n{lab}",
            point_size=point_size,
            smooth=True,
            tissue_outline=True,
            core_prob=core_prob,
            peri_prob=peri_prob,
            label_values=labels,
        )
        marker_axes.append(ax)
    marker_axes[0].text(
        -0.22, 0.5, "A  Marker genes",
        transform=marker_axes[0].transAxes,
        ha="right", va="center",
        color="white", fontsize=10.5, fontweight="bold"
    )

    # B modules: row 1, four panels occupying first 4 columns
    mod_cmap = cmap_dark_diverging()
    mod_vals = []
    mod_cache = {}
    for m in selected_modules:
        if m in module_scores:
            z = zscore_clip(module_scores[m], clip=2.5)
            mod_cache[m] = z
            mod_vals.append(z[np.isfinite(z)])
    cat = np.concatenate(mod_vals) if mod_vals else np.array([-1, 1])
    vmax = max(1.2, np.quantile(np.abs(cat), 0.98))
    mod_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    module_axes = []
    for j, m in enumerate(selected_modules):
        ax = fig.add_subplot(gs[1, j])
        plot_spatial_landscape(
            ax, x, y, mod_cache[m],
            cmap=mod_cmap,
            norm=mod_norm,
            title=m,
            point_size=point_size,
            smooth=True,
            tissue_outline=True,
            core_prob=core_prob,
            peri_prob=peri_prob,
            label_values=labels,
        )
        module_axes.append(ax)
    module_axes[0].text(
        -0.22, 0.5, "B  Module activities",
        transform=module_axes[0].transAxes,
        ha="right", va="center",
        color="white", fontsize=10.5, fontweight="bold"
    )

    # C states: row 2, three panels
    state_axes = []
    prob_norm = Normalize(0, 1)
    for j, (key, title, cmap) in enumerate(selected_states):
        ax = fig.add_subplot(gs[2, j])
        col = prob_cols.get(key)
        if col is not None and col in meta.columns:
            v = pd.to_numeric(meta[col], errors="coerce").to_numpy(dtype=float)
            plot_spatial_landscape(
                ax, x, y, v,
                cmap=cmap,
                norm=prob_norm,
                title=title,
                point_size=point_size,
                smooth=True,
                tissue_outline=True,
                core_prob=core_prob,
                peri_prob=peri_prob,
                label_values=labels,
            )
        else:
            setup_dark_ax(ax)
            ax.text(0.5, 0.5, f"{title}\\nmissing", ha="center", va="center",
                    color="white", transform=ax.transAxes)
        state_axes.append(ax)

    state_axes[0].text(
        -0.22, 0.5, "C  State probabilities",
        transform=state_axes[0].transAxes,
        ha="right", va="center",
        color="white", fontsize=10.5, fontweight="bold"
    )

    # empty remaining axes row 2 col 3-5 for note
    ax_note = fig.add_subplot(gs[2, 3:6])
    setup_dark_ax(ax_note)
    ax_note.text(
        0.02, 0.70,
        "Spatial landscape visualization",
        color="white",
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="center",
        transform=ax_note.transAxes
    )
    ax_note.text(
        0.02, 0.48,
        "Smoothed density layer + raw spot overlay",
        color="#cbd5e1",
        fontsize=9,
        ha="left",
        va="center",
        transform=ax_note.transAxes
    )
    ax_note.text(
        0.02, 0.32,
        "White dashed contour: tissue outline",
        color="#cbd5e1",
        fontsize=9,
        ha="left",
        va="center",
        transform=ax_note.transAxes
    )
    ax_note.text(
        0.02, 0.16,
        "Red/yellow contours: core/peri regions when available",
        color="#cbd5e1",
        fontsize=9,
        ha="left",
        va="center",
        transform=ax_note.transAxes
    )

    fig.suptitle("Spatial marker, module and state landscape atlas", fontsize=20, color="white", fontweight="bold", y=0.985)

    fig.text(
        0.5, 0.012,
        "Spatial landscapes are visualization-level summaries of gene expression, curated module scores and model-derived state probabilities.",
        ha="center", va="bottom", fontsize=8, color="#e5e7eb"
    )

    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.055)

    for ext in ["png", "pdf", "svg"]:
        fig.savefig(f"{outbase}.{ext}", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--marker_json", default="")
    parser.add_argument("--module_json", default="")

    parser.add_argument("--adata_id_col", default="")
    parser.add_argument("--state_id_col", default="")
    parser.add_argument("--xcol", default="")
    parser.add_argument("--ycol", default="")

    parser.add_argument("--core_col", default="")
    parser.add_argument("--peri_col", default="")
    parser.add_argument("--remote_col", default="")

    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    print("=" * 100)
    print("Step75B | Dark spatial landscape atlas")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={outdir}")

    adata = ad.read_h5ad(args.h5ad)
    obs = adata.obs.copy()

    adata_id, adata_id_col_used = get_obs_id_series(adata, obs, user_col=args.adata_id_col)
    obs["_adata_id_"] = adata_id.astype(str).values

    x, y, xcol_used, ycol_used, xy_mode = get_xy(adata, obs, xcol=args.xcol, ycol=args.ycol)
    obs["_spatial_x_"] = x
    obs["_spatial_y_"] = y

    state_df = read_table(args.state_table)
    meta, merge_audit = merge_state_table(
        obs,
        state_df,
        adata_id_col="_adata_id_",
        user_state_id_col=args.state_id_col
    )

    prob_cols = infer_prob_cols(
        meta,
        core_col=args.core_col,
        peri_col=args.peri_col,
        remote_col=args.remote_col
    )

    marker_groups = read_json(args.marker_json) if args.marker_json else None
    if marker_groups is None:
        marker_groups = DEFAULT_MARKER_GROUPS

    module_sets = read_json(args.module_json) if args.module_json else None
    if module_sets is None:
        module_sets = DEFAULT_MODULE_GENESETS

    gmap = gene_lookup(adata)
    module_scores, module_audit = compute_module_scores(adata, module_sets, gmap, min_genes=2)

    # use meta x/y after merge
    x = pd.to_numeric(meta["_spatial_x_"], errors="coerce").to_numpy()
    y = pd.to_numeric(meta["_spatial_y_"], errors="coerce").to_numpy()

    # save audits
    audit = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "adata_id_col_used": adata_id_col_used,
        "xy_mode": xy_mode,
        "xcol_used": xcol_used,
        "ycol_used": ycol_used,
        "state_merge_audit": merge_audit,
        "prob_cols": prob_cols,
        "marker_groups": marker_groups,
        "module_sets": module_sets,
    }

    (outdir / "step75b_audit_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    module_audit.to_csv(outdir / "step75b_module_gene_match_audit.csv", index=False)

    print("---- audit ----")
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    make_marker_atlas(
        meta=meta,
        adata=adata,
        x=x,
        y=y,
        marker_groups=marker_groups,
        gmap=gmap,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75B_DarkSpatialMarkerLandscapeAtlas"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    make_module_atlas(
        meta=meta,
        x=x,
        y=y,
        module_scores=module_scores,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75B_DarkSpatialModuleLandscapeAtlas"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    make_state_atlas(
        meta=meta,
        x=x,
        y=y,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75B_DarkSpatialStateProbabilityLandscapeAtlas"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    make_compact_panel(
        meta=meta,
        adata=adata,
        x=x,
        y=y,
        marker_groups=marker_groups,
        module_scores=module_scores,
        gmap=gmap,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75B_CompactSpatialLandscapeManuscriptPanel"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    report = {
        "status": "ok",
        "outputs": {
            "marker_atlas_png": str(outdir / "Fig_Step75B_DarkSpatialMarkerLandscapeAtlas.png"),
            "module_atlas_png": str(outdir / "Fig_Step75B_DarkSpatialModuleLandscapeAtlas.png"),
            "state_probability_atlas_png": str(outdir / "Fig_Step75B_DarkSpatialStateProbabilityLandscapeAtlas.png"),
            "compact_panel_png": str(outdir / "Fig_Step75B_CompactSpatialLandscapeManuscriptPanel.png"),
            "audit_json": str(outdir / "step75b_audit_report.json"),
            "module_gene_match_audit": str(outdir / "step75b_module_gene_match_audit.csv"),
        },
        "interpretation_note": (
            "Step75B generates dark-background spatial landscape atlases with smoothed visualization layers, "
            "raw spot overlays, tissue outlines, and core/peri contours when model-derived state probabilities "
            "or labels are available. These are visualization-level spatial landscapes, not direct energy landscapes."
        )
    }

    (outdir / "step75b_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step75b_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 100)
    print("DONE Step75B")
    print("=" * 100)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
