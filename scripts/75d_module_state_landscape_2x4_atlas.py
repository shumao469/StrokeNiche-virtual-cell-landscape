#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step75D | Main-text 2x4 module/state landscape atlas

Panels:
1. core probability
2. peri probability
3. remote probability
4. repair score
5. ferroptosis
6. inflammation
7. BBB leakage
8. astrocyte reactive

Purpose:
- Produce a compact manuscript-grade spatial landscape atlas.
- Combine model-derived state probabilities with module-level activity maps.
- Use dark background, smoothed spatial layer, raw spot overlay, tissue outline,
  and optional core/peri contours.

Interpretation:
- This is a visualization-level spatial landscape atlas.
- Module maps are gene-set score proxies unless a matched score column is found.
- This is not a direct energy landscape and not functional validation.
"""

from pathlib import Path
import argparse
import json
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
# Gene sets
# =============================================================================

MODULE_GENESETS = {
    "repair_score": [
        "Col1a1", "Col1a2", "Col3a1", "Fn1", "Spp1", "Apoe",
        "Vim", "Postn", "Timp1", "Mmp2", "Mmp9", "Thbs1"
    ],
    "ferroptosis": [
        "Hmox1", "Fth1", "Ftl1", "Slc7a11", "Gpx4", "Acsl4",
        "Tfrc", "Ptgs2", "Nfe2l1", "Nfe2l2"
    ],
    "inflammation": [
        "Ccl2", "Ccr2", "Ackr1", "Tnf", "Il1b", "Il6",
        "C1qa", "C1qb", "C1qc", "Tyrobp", "Lgals3", "Ctsd"
    ],
    "BBB_leakage": [
        "Vegfa", "Flt1", "Kdr", "Vwf", "Mmp9", "Timp1",
        "Pecam1", "Cldn5", "Kcnj8", "Rgs5", "Emcn"
    ],
    "astrocyte_reactive": [
        "Gfap", "Aqp4", "Vim", "Clu", "Apoe", "Serpina3n",
        "Lcn2", "C3", "S100b", "Slc1a2"
    ],
}


PANEL_ORDER = [
    {
        "key": "core_probability",
        "title": "Core probability",
        "kind": "prob",
        "preferred_cols": [
            "core_probability", "core_prob", "prob_core", "p_core",
            "lesion_core_probability", "lesion_core_like_probability"
        ],
    },
    {
        "key": "peri_probability",
        "title": "Peri-infarct probability",
        "kind": "prob",
        "preferred_cols": [
            "peri_probability", "peri_prob", "prob_peri", "p_peri",
            "peri_infarct_probability", "peri_like_probability"
        ],
    },
    {
        "key": "remote_probability",
        "title": "Remote-like probability",
        "kind": "prob",
        "preferred_cols": [
            "remote_probability", "remote_prob", "prob_remote", "p_remote",
            "remote_like_probability"
        ],
    },
    {
        "key": "repair_score",
        "title": "Repair score",
        "kind": "score",
        "preferred_cols": [
            "repair_score", "repair_ECM_score", "repair_ecm_score",
            "repair_module_score", "repair_gain"
        ],
    },
    {
        "key": "ferroptosis",
        "title": "Ferroptosis",
        "kind": "score",
        "preferred_cols": [
            "ferroptosis_score", "ferroptosis", "ferroptosis_module_score"
        ],
    },
    {
        "key": "inflammation",
        "title": "Inflammation",
        "kind": "score",
        "preferred_cols": [
            "inflammation_score", "inflammation", "microglia_inflammation_score",
            "microglia_inflammatory_score"
        ],
    },
    {
        "key": "BBB_leakage",
        "title": "BBB leakage",
        "kind": "score",
        "preferred_cols": [
            "BBB_leakage_score", "bbb_leakage_score", "bbb_leakage",
            "endothelial_barrier_fragility_score"
        ],
    },
    {
        "key": "astrocyte_reactive",
        "title": "Reactive astrocyte",
        "kind": "score",
        "preferred_cols": [
            "astrocyte_reactive_score", "astrocyte_reactive",
            "reactive_astrocyte_score"
        ],
    },
]


# =============================================================================
# Utility
# =============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path):
    if not path:
        return pd.DataFrame()
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", low_memory=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def first_existing(cols, candidates):
    cols = list(cols)
    lower = {str(c).lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def dense_vector(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x).reshape(-1)


def robust_limits(x, q_low=0.01, q_high=0.99):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0.0, 1.0
    lo = np.quantile(x, q_low)
    hi = np.quantile(x, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-8:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-8:
        return 0.0, 1.0
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

def cmap_prob_core():
    return LinearSegmentedColormap.from_list(
        "prob_core",
        ["#02030a", "#2a0611", "#5f0f1b", "#b11226", "#f26d5b", "#ffe0b2"]
    )


def cmap_prob_peri():
    return LinearSegmentedColormap.from_list(
        "prob_peri",
        ["#02030a", "#2b1700", "#664000", "#a66a00", "#f59f00", "#fff3bf"]
    )


def cmap_prob_remote():
    return LinearSegmentedColormap.from_list(
        "prob_remote",
        ["#02030a", "#071d3a", "#0b4f8a", "#2b8cbe", "#a6cee3", "#f0f9ff"]
    )


def cmap_score_diverging():
    return LinearSegmentedColormap.from_list(
        "score_div",
        ["#2166ac", "#67a9cf", "#05070b", "#ef8a62", "#b2182b"]
    )


# =============================================================================
# h5ad metadata
# =============================================================================

def get_obs_id_series(adata, obs, user_col=""):
    if user_col and user_col in obs.columns:
        return obs[user_col].astype(str), user_col

    candidates = [
        "obs_name", "cell_id", "spot_id", "barcode",
        "CellID", "SpotID", "cell", "spot", "id"
    ]
    col = first_existing(obs.columns, candidates)
    if col:
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
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
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
            return arr[:, 0].astype(float), arr[:, 1].astype(float), "obsm_spatial_0", "obsm_spatial_1", "obsm_spatial"

    raise ValueError("Cannot infer spatial x/y. Please pass --xcol and --ycol.")


def merge_state_table(meta, state_df, adata_id_col="_adata_id_", state_id_col=""):
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

    if state_id_col and state_id_col in st.columns:
        sid = state_id_col
    else:
        sid = first_existing(
            st.columns,
            [
                "obs_name", "cell_id", "spot_id", "barcode",
                "CellID", "SpotID", "cell", "spot", "id"
            ]
        )

    if sid:
        left = meta[adata_id_col].astype(str)
        right = st[sid].astype(str)
        overlap = len(set(left).intersection(set(right)))
        audit["state_merge_overlap"] = int(overlap)

        if overlap > 0:
            meta = meta.copy()
            st = st.copy()
            meta["_merge_key_"] = left
            st["_merge_key_"] = right
            st = st.drop_duplicates("_merge_key_")
            out = meta.merge(st, on="_merge_key_", how="left", suffixes=("", "_state"))
            out.drop(columns=["_merge_key_"], inplace=True, errors="ignore")
            audit["merge_mode"] = "id"
            audit["merge_key_meta"] = adata_id_col
            audit["merge_key_state"] = sid
            return out, audit

    if len(st) == len(meta):
        meta = meta.copy()
        st = st.copy()
        meta["_row_order_"] = np.arange(len(meta))
        st["_row_order_"] = np.arange(len(st))
        out = meta.merge(st, on="_row_order_", how="left", suffixes=("", "_state"))
        out.drop(columns=["_row_order_"], inplace=True, errors="ignore")
        audit["merge_mode"] = "row_order"
        audit["merge_key_meta"] = "row_order"
        audit["merge_key_state"] = "row_order"
        audit["state_merge_overlap"] = int(len(meta))
        return out, audit

    audit["merge_mode"] = "failed"
    return meta, audit


# =============================================================================
# Gene/module score
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


def compute_gene_set_score(adata, genes, gmap, min_genes=2):
    exprs = []
    matched = []
    for g in genes:
        if str(g).upper() in gmap:
            exprs.append(get_gene_expr(adata, g, gmap))
            matched.append(g)

    if len(exprs) < min_genes:
        return np.full(adata.n_obs, np.nan), matched

    M = np.vstack(exprs).T
    Z = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        Z[:, j] = zscore_clip(M[:, j], clip=2.5)
    score = np.nanmean(Z, axis=1)
    return score, matched


def infer_prob_cols(meta):
    cols = list(meta.columns)
    lower = {str(c).lower(): c for c in cols}

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return c
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    return {
        "core": pick(["core_probability", "core_prob", "prob_core", "p_core", "lesion_core_probability"]),
        "peri": pick(["peri_probability", "peri_prob", "prob_peri", "p_peri", "peri_infarct_probability"]),
        "remote": pick(["remote_probability", "remote_prob", "prob_remote", "p_remote", "remote_like_probability"]),
    }


def build_panel_values(meta, adata, gmap):
    rows = []
    values = {}

    for panel in PANEL_ORDER:
        key = panel["key"]
        preferred_col = first_existing(meta.columns, panel["preferred_cols"])

        if preferred_col is not None:
            raw = pd.to_numeric(meta[preferred_col], errors="coerce").to_numpy(dtype=float)
            source = f"column:{preferred_col}"
            matched = ""
        else:
            genes = MODULE_GENESETS.get(key, [])
            raw, matched_list = compute_gene_set_score(adata, genes, gmap, min_genes=2)
            source = "computed_gene_set_score"
            matched = ";".join(matched_list)

        if panel["kind"] == "prob":
            plot_val = raw.copy()
            lo, hi = 0.0, 1.0
            norm_kind = "probability_0_1"
        else:
            plot_val = zscore_clip(raw, clip=2.5)
            lo, hi = np.nan, np.nan
            norm_kind = "z_score"

        values[key] = plot_val

        rows.append({
            "panel_key": key,
            "title": panel["title"],
            "kind": panel["kind"],
            "source": source,
            "matched_genes_if_computed": matched,
            "n_finite": int(np.isfinite(plot_val).sum()),
            "min": float(np.nanmin(plot_val)) if np.isfinite(plot_val).sum() else np.nan,
            "max": float(np.nanmax(plot_val)) if np.isfinite(plot_val).sum() else np.nan,
            "norm_kind": norm_kind,
        })

    return values, pd.DataFrame(rows)


# =============================================================================
# Spatial smoothing / outline
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
    x0 -= dx * 0.04
    x1 += dx * 0.04
    y0 -= dy * 0.04
    y1 += dy * 0.04

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


def draw_tissue_outline(ax, x, y, grid_size=260):
    occ, cnt, extent = spatial_grid_smooth(x, y, values=None, grid_size=grid_size, sigma=2.0)
    if np.nanmax(cnt) <= 0:
        return
    level = np.nanquantile(cnt[cnt > 0], 0.08)
    try:
        ax.contour(
            cnt,
            levels=[level],
            extent=extent,
            colors="white",
            linewidths=0.65,
            alpha=0.65,
            linestyles="dashed",
            zorder=5,
        )
    except Exception:
        pass


def draw_prob_contour(ax, x, y, prob, level, color, grid_size=260):
    prob = np.asarray(prob, dtype=float)
    if np.isfinite(prob).sum() < 10:
        return
    sm, cnt, extent = spatial_grid_smooth(x, y, values=prob, grid_size=grid_size, sigma=2.2)
    if not np.isfinite(sm).any():
        return
    lvl = level
    if np.nanmax(sm) < lvl:
        lvl = np.nanquantile(sm[np.isfinite(sm)], 0.85)
    try:
        ax.contour(
            sm,
            levels=[lvl],
            extent=extent,
            colors=color,
            linewidths=0.85,
            alpha=0.85,
            linestyles="solid",
            zorder=6,
        )
    except Exception:
        pass


def setup_dark_ax(ax):
    ax.set_facecolor("#05070b")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def plot_landscape(ax, x, y, val, cmap, norm, title, core=None, peri=None, point_size=2.0, grid_size=260):
    setup_dark_ax(ax)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    val = np.asarray(val, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)

    if ok.sum() > 10:
        sm, cnt, extent = spatial_grid_smooth(x, y, val, grid_size=grid_size, sigma=2.2)
        alpha = cnt / np.nanmax(cnt) if np.nanmax(cnt) > 0 else cnt
        alpha = np.clip(alpha, 0, 1)

        ax.imshow(
            sm,
            extent=extent,
            origin="lower",
            cmap=cmap,
            norm=norm,
            alpha=0.72 * alpha,
            interpolation="bilinear",
            zorder=1,
        )

        ax.scatter(
            x[ok],
            y[ok],
            c=val[ok],
            cmap=cmap,
            norm=norm,
            s=point_size,
            linewidths=0,
            alpha=0.78,
            rasterized=True,
            zorder=2,
        )

    draw_tissue_outline(ax, x, y, grid_size=grid_size)

    if core is not None:
        draw_prob_contour(ax, x, y, core, level=0.50, color="#ff4d6d", grid_size=grid_size)
    if peri is not None:
        draw_prob_contour(ax, x, y, peri, level=0.45, color="#ffd166", grid_size=grid_size)

    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, color="white", fontweight="bold", pad=5)


def add_fixed_colorbar(fig, cmap, norm, label, x, y, h, color="white"):
    cax = fig.add_axes([x, y, 0.012, h])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(colors=color, labelsize=7)
    cb.outline.set_edgecolor(color)
    cb.set_label(label, color=color, fontsize=8)
    return cb


# =============================================================================
# Figure
# =============================================================================

def make_2x4_figure(meta, x, y, values, outbase, point_size=2.0, dpi=600):
    fig, axes = plt.subplots(
        2, 4,
        figsize=(13.8, 7.1),
        facecolor="#05070b"
    )
    axes = np.asarray(axes).reshape(2, 4)

    prob_norm = Normalize(0, 1)
    score_vals = []
    for panel in PANEL_ORDER:
        if panel["kind"] == "score":
            v = values[panel["key"]]
            score_vals.append(v[np.isfinite(v)])
    if score_vals:
        cat = np.concatenate(score_vals)
        vmax = max(1.2, np.nanquantile(np.abs(cat), 0.98))
    else:
        vmax = 2.0
    score_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    prob_cols = infer_prob_cols(meta)
    core = pd.to_numeric(meta[prob_cols["core"]], errors="coerce").to_numpy(dtype=float) if prob_cols["core"] in meta.columns else None
    peri = pd.to_numeric(meta[prob_cols["peri"]], errors="coerce").to_numpy(dtype=float) if prob_cols["peri"] in meta.columns else None

    for idx, panel in enumerate(PANEL_ORDER):
        i = idx // 4
        j = idx % 4
        ax = axes[i, j]

        key = panel["key"]
        val = values[key]

        if key == "core_probability":
            cmap = cmap_prob_core()
            norm = prob_norm
        elif key == "peri_probability":
            cmap = cmap_prob_peri()
            norm = prob_norm
        elif key == "remote_probability":
            cmap = cmap_prob_remote()
            norm = prob_norm
        else:
            cmap = cmap_score_diverging()
            norm = score_norm

        plot_landscape(
            ax,
            x,
            y,
            val,
            cmap=cmap,
            norm=norm,
            title=panel["title"],
            core=core,
            peri=peri,
            point_size=point_size,
        )

    fig.suptitle(
        "Module and state-probability spatial landscape atlas",
        fontsize=20,
        color="white",
        fontweight="bold",
        y=0.985,
    )

    # Colorbars: probability and module z-score.
    add_fixed_colorbar(
        fig,
        cmap_prob_core(),
        prob_norm,
        "State probability",
        x=0.928,
        y=0.56,
        h=0.30,
        color="white"
    )

    add_fixed_colorbar(
        fig,
        cmap_score_diverging(),
        score_norm,
        "Module score (z)",
        x=0.928,
        y=0.15,
        h=0.30,
        color="white"
    )

    fig.text(
        0.5,
        0.018,
        "Smoothed spatial landscape with raw spot overlay. White dashed contour indicates tissue boundary; red/yellow contours indicate core/peri regions when available.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#e5e7eb"
    )

    fig.subplots_adjust(left=0.035, right=0.905, top=0.90, bottom=0.075, wspace=0.055, hspace=0.16)

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

    parser.add_argument("--adata_id_col", default="")
    parser.add_argument("--state_id_col", default="")
    parser.add_argument("--xcol", default="")
    parser.add_argument("--ycol", default="")

    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    print("=" * 100)
    print("Step75D | 2x4 module/state landscape atlas")
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
        state_id_col=args.state_id_col,
    )

    x = pd.to_numeric(meta["_spatial_x_"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(meta["_spatial_y_"], errors="coerce").to_numpy(dtype=float)

    gmap = gene_lookup(adata)
    values, panel_audit = build_panel_values(meta, adata, gmap)

    outbase = outdir / "Fig_Step75D_ModuleStateLandscapeAtlas_2x4"

    make_2x4_figure(
        meta=meta,
        x=x,
        y=y,
        values=values,
        outbase=str(outbase),
        point_size=args.point_size,
        dpi=args.dpi,
    )

    # Save numeric panel values for audit / reproducibility.
    value_df = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "spatial_x": x,
        "spatial_y": y,
    })
    for panel in PANEL_ORDER:
        value_df[panel["key"]] = values[panel["key"]]

    value_path = outdir / "step75d_panel_values_used.csv"
    value_df.to_csv(value_path, index=False)

    panel_audit_path = outdir / "step75d_panel_source_audit.csv"
    panel_audit.to_csv(panel_audit_path, index=False)

    audit = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "adata_id_col_used": adata_id_col_used,
        "xy_mode": xy_mode,
        "xcol_used": xcol_used,
        "ycol_used": ycol_used,
        "state_merge_audit": merge_audit,
        "panel_order": PANEL_ORDER,
        "outputs": {
            "figure_png": str(outbase.with_suffix(".png")),
            "figure_pdf": str(outbase.with_suffix(".pdf")),
            "figure_svg": str(outbase.with_suffix(".svg")),
            "panel_values": str(value_path),
            "panel_source_audit": str(panel_audit_path),
        },
        "interpretation_note": (
            "Step75D generates a compact 2x4 spatial landscape atlas combining model-derived state probabilities "
            "and module/gene-set scores. Module panels are visualization-level gene-set score proxies when matched "
            "score columns are unavailable. This is not a direct energy landscape or functional validation."
        ),
    }

    (outdir / "step75d_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step75d_report.txt").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("---- panel source audit ----")
    print(panel_audit.to_string(index=False))
    print("=" * 100)
    print("DONE Step75D")
    print("=" * 100)
    print(json.dumps(audit["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
