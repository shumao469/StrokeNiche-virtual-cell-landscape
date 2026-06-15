#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step75E | Before/after virtual perturbation spatial rescue atlas

Rows:
  1. Ccl2/Ccr2-Ackr1 blockade
  2. Spp1-Cd44 blockade
  3. Vegfa-Flt1/Kdr blockade
  4. Ferroptosis down-modulation
  5. Repair-ECM promotion

Columns:
  1. Baseline core map
  2. Perturbed core map
  3. Repair gain map
  4. Core reduction map

Input:
  - spatial h5ad
  - Step64 state probability table
  - optional per-spot rescue table

If --rescue_table is unavailable or does not contain matched per-spot perturbation
columns, the script derives visualization-level counterfactual proxy maps from:
  baseline core_probability
  matched target/module expression scores
  matched repair score

Interpretation:
  These are virtual perturbation / state-editing visualization maps.
  They are not wet-lab KO/blockade results and not observed cell-fate transitions.
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


PERTURBATIONS = [
    {
        "name": "Ccl2/Ccr2-Ackr1 blockade",
        "short": "Ccl2/Ccr2-Ackr1",
        "genes": ["Ccl2", "Ccr2", "Ackr1", "Ccl7", "Ccl12"],
        "direction": "down",
        "effect_strength": 0.34,
        "claim": "chemokine-axis blockade proxy",
    },
    {
        "name": "Spp1-Cd44 blockade",
        "short": "Spp1-Cd44",
        "genes": ["Spp1", "Cd44", "Itgav", "Itgb1", "Apoe"],
        "direction": "down",
        "effect_strength": 0.38,
        "claim": "osteopontin-axis blockade proxy",
    },
    {
        "name": "Vegfa-Flt1/Kdr blockade",
        "short": "Vegfa-Flt1/Kdr",
        "genes": ["Vegfa", "Flt1", "Kdr", "Vwf", "Pecam1"],
        "direction": "down",
        "effect_strength": 0.30,
        "claim": "vascular signaling blockade proxy",
    },
    {
        "name": "Ferroptosis down-modulation",
        "short": "Ferroptosis down",
        "genes": ["Hmox1", "Fth1", "Ftl1", "Slc7a11", "Gpx4", "Acsl4", "Tfrc", "Ptgs2"],
        "direction": "down",
        "effect_strength": 0.40,
        "claim": "ferroptosis/redox down-modulation proxy",
    },
    {
        "name": "Repair-ECM promotion",
        "short": "Repair-ECM up",
        "genes": ["Col1a1", "Col1a2", "Col3a1", "Fn1", "Spp1", "Apoe", "Vim", "Postn", "Timp1"],
        "direction": "up",
        "effect_strength": 0.34,
        "claim": "repair-permissive ECM promotion proxy",
    },
]


REPAIR_GENES = [
    "Col1a1", "Col1a2", "Col3a1", "Fn1", "Spp1",
    "Apoe", "Vim", "Postn", "Timp1", "Mmp2", "Thbs1"
]


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def minmax01(x, q_low=0.02, q_high=0.98):
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() == 0:
        return np.zeros_like(x, dtype=float)

    lo = np.nanquantile(x[ok], q_low)
    hi = np.nanquantile(x[ok], q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        lo = np.nanmin(x[ok])
        hi = np.nanmax(x[ok])
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
        out[ok] = 0.0
        out[~ok] = 0.0
        return out

    out[ok] = (x[ok] - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~np.isfinite(out)] = 0
    return out


def slugify(x):
    x = str(x).lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    return x.strip("_")


# =============================================================================
# Colormaps
# =============================================================================

def cmap_core():
    return LinearSegmentedColormap.from_list(
        "core_map",
        ["#02030a", "#2a0611", "#5f0f1b", "#b11226", "#f26d5b", "#ffe0b2"]
    )


def cmap_perturbed():
    return LinearSegmentedColormap.from_list(
        "perturbed_map",
        ["#02030a", "#071d3a", "#0b4f8a", "#2b8cbe", "#a6cee3", "#f0f9ff"]
    )


def cmap_gain():
    return LinearSegmentedColormap.from_list(
        "gain_map",
        ["#02030a", "#092414", "#0f5132", "#2f9e44", "#a3e635", "#fff7ad"]
    )


def cmap_delta():
    return LinearSegmentedColormap.from_list(
        "delta_core",
        ["#02030a", "#2a0611", "#5f0f1b", "#b11226", "#f26d5b", "#ffe0b2"]
    )


# =============================================================================
# h5ad and metadata
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


def infer_prob_cols(meta):
    cols = list(meta.columns)
    lower = {str(c).lower(): c for c in cols}

    def pick(candidates):
        for c in candidates:
            if c in cols:
                return c
            if str(c).lower() in lower:
                return lower[str(c).lower()]
        return None

    return {
        "core": pick(["core_probability", "core_prob", "prob_core", "p_core", "lesion_core_probability"]),
        "peri": pick(["peri_probability", "peri_prob", "prob_peri", "p_peri", "peri_infarct_probability"]),
        "remote": pick(["remote_probability", "remote_prob", "prob_remote", "p_remote", "remote_like_probability"]),
        "repair": pick(["repair_score", "repair_ECM_score", "repair_ecm_score", "repair_module_score"]),
    }


# =============================================================================
# Expression scores
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


# =============================================================================
# Optional rescue table support
# =============================================================================

def merge_rescue_table(meta, rescue_df, adata_id_col="_adata_id_", rescue_id_col=""):
    audit = {
        "rescue_table_available": False,
        "rescue_table_rows": 0,
        "merge_mode": "none",
        "rescue_merge_overlap": 0,
    }

    if rescue_df is None or rescue_df.empty:
        return meta, audit

    audit["rescue_table_available"] = True
    audit["rescue_table_rows"] = int(len(rescue_df))

    df = rescue_df.copy()

    if rescue_id_col and rescue_id_col in df.columns:
        rid = rescue_id_col
    else:
        rid = first_existing(
            df.columns,
            [
                "obs_name", "cell_id", "spot_id", "barcode",
                "CellID", "SpotID", "cell", "spot", "id"
            ]
        )

    if rid:
        left = meta[adata_id_col].astype(str)
        right = df[rid].astype(str)
        overlap = len(set(left).intersection(set(right)))
        audit["rescue_merge_overlap"] = int(overlap)

        if overlap > 0:
            meta = meta.copy()
            df = df.copy()
            meta["_rescue_key_"] = left
            df["_rescue_key_"] = right
            df = df.drop_duplicates("_rescue_key_")
            out = meta.merge(df, on="_rescue_key_", how="left", suffixes=("", "_rescue"))
            out.drop(columns=["_rescue_key_"], inplace=True, errors="ignore")
            audit["merge_mode"] = "id"
            return out, audit

    if len(df) == len(meta):
        meta = meta.copy()
        df = df.copy()
        meta["_row_order_rescue_"] = np.arange(len(meta))
        df["_row_order_rescue_"] = np.arange(len(df))
        out = meta.merge(df, on="_row_order_rescue_", how="left", suffixes=("", "_rescue"))
        out.drop(columns=["_row_order_rescue_"], inplace=True, errors="ignore")
        audit["merge_mode"] = "row_order"
        audit["rescue_merge_overlap"] = int(len(meta))
        return out, audit

    audit["merge_mode"] = "failed"
    return meta, audit


def find_rescue_cols(meta, perturbation_name):
    """
    Try to find columns in wide table.

    Accepted examples:
      ccl2_ccr2_ackr1__baseline_core
      ccl2_ccr2_ackr1__perturbed_core
      ccl2_ccr2_ackr1__repair_gain
      ccl2_ccr2_ackr1__core_reduction
    """
    slug = slugify(perturbation_name)
    cols = list(meta.columns)
    lower = {str(c).lower(): c for c in cols}

    def pick(metric_terms):
        candidates = []
        for c in cols:
            lc = str(c).lower()
            if slug in slugify(lc) and all(t in slugify(lc) for t in metric_terms):
                candidates.append(c)
        if candidates:
            return candidates[0]

        # fallback: metric-only if table is known to contain a single perturbation
        for c in cols:
            lc = slugify(c)
            if all(t in lc for t in metric_terms):
                candidates.append(c)
        if candidates:
            return candidates[0]

        return None

    return {
        "baseline": pick(["baseline"]),
        "perturbed": pick(["perturbed"]),
        "repair_gain": pick(["repair", "gain"]),
        "core_reduction": pick(["core", "reduction"]),
    }


# =============================================================================
# Counterfactual proxy maps
# =============================================================================

def compute_perturbation_maps(meta, adata, gmap, prob_cols):
    if prob_cols["core"] is None or prob_cols["core"] not in meta.columns:
        raise ValueError("core_probability column is required.")

    core = pd.to_numeric(meta[prob_cols["core"]], errors="coerce").to_numpy(dtype=float)
    core = np.clip(core, 0, 1)

    if prob_cols.get("repair") and prob_cols["repair"] in meta.columns:
        repair_raw = pd.to_numeric(meta[prob_cols["repair"]], errors="coerce").to_numpy(dtype=float)
        repair_source = f"column:{prob_cols['repair']}"
        repair_matched = ""
    else:
        repair_raw, repair_matched_list = compute_gene_set_score(adata, REPAIR_GENES, gmap, min_genes=2)
        repair_source = "computed_repair_gene_set_score"
        repair_matched = ";".join(repair_matched_list)

    repair_norm = minmax01(repair_raw)

    maps = {}
    audit_rows = []

    for p in PERTURBATIONS:
        target_score, matched = compute_gene_set_score(adata, p["genes"], gmap, min_genes=1)
        target_norm = minmax01(target_score)

        if p["direction"] == "up":
            # Repair-ECM promotion is strongest where repair program is low
            # and core probability remains high.
            gain_seed = (1.0 - repair_norm) * (0.65 + 0.35 * core)
            repair_gain = p["effect_strength"] * minmax01(gain_seed)
            core_reduction = np.clip(0.75 * repair_gain * core, 0, 1)
        else:
            # Blockade/down-modulation is strongest where target axis is active
            # within core-like or transition areas.
            gain_seed = target_norm * (0.50 + 0.50 * core)
            repair_gain = p["effect_strength"] * 0.55 * minmax01(gain_seed)
            core_reduction = np.clip(p["effect_strength"] * target_norm * core, 0, 1)

        perturbed_core = np.clip(core - core_reduction, 0, 1)

        maps[p["name"]] = {
            "baseline": core,
            "perturbed": perturbed_core,
            "repair_gain": repair_gain,
            "core_reduction": core_reduction,
        }

        audit_rows.append({
            "perturbation": p["name"],
            "short": p["short"],
            "claim": p["claim"],
            "direction": p["direction"],
            "effect_strength": p["effect_strength"],
            "source": "derived_expression_state_proxy",
            "target_genes": ";".join(p["genes"]),
            "matched_target_genes": ";".join(matched),
            "n_matched_target_genes": len(matched),
            "repair_score_source": repair_source,
            "matched_repair_genes": repair_matched,
            "mean_baseline_core": float(np.nanmean(core)),
            "mean_perturbed_core": float(np.nanmean(perturbed_core)),
            "mean_repair_gain": float(np.nanmean(repair_gain)),
            "mean_core_reduction": float(np.nanmean(core_reduction)),
        })

    return maps, pd.DataFrame(audit_rows)


def compute_maps_from_rescue_table_if_available(meta, adata, gmap, prob_cols):
    """
    Prefer matched rescue-table columns if available; otherwise proxy maps.
    """
    core = pd.to_numeric(meta[prob_cols["core"]], errors="coerce").to_numpy(dtype=float)
    proxy_maps, proxy_audit = compute_perturbation_maps(meta, adata, gmap, prob_cols)

    maps = {}
    audit_rows = []

    for p in PERTURBATIONS:
        found = find_rescue_cols(meta, p["name"])

        use_external = all(found[k] is not None and found[k] in meta.columns for k in ["perturbed", "repair_gain", "core_reduction"])

        if use_external:
            baseline_col = found["baseline"]
            baseline = pd.to_numeric(meta[baseline_col], errors="coerce").to_numpy(dtype=float) if baseline_col else core
            perturbed = pd.to_numeric(meta[found["perturbed"]], errors="coerce").to_numpy(dtype=float)
            repair_gain = pd.to_numeric(meta[found["repair_gain"]], errors="coerce").to_numpy(dtype=float)
            core_reduction = pd.to_numeric(meta[found["core_reduction"]], errors="coerce").to_numpy(dtype=float)

            maps[p["name"]] = {
                "baseline": np.clip(baseline, 0, 1),
                "perturbed": np.clip(perturbed, 0, 1),
                "repair_gain": np.clip(repair_gain, 0, None),
                "core_reduction": np.clip(core_reduction, 0, None),
            }

            audit_rows.append({
                "perturbation": p["name"],
                "short": p["short"],
                "source": "rescue_table_columns",
                "baseline_col": baseline_col,
                "perturbed_col": found["perturbed"],
                "repair_gain_col": found["repair_gain"],
                "core_reduction_col": found["core_reduction"],
            })
        else:
            maps[p["name"]] = proxy_maps[p["name"]]
            row = proxy_audit.loc[proxy_audit["perturbation"] == p["name"]].iloc[0].to_dict()
            row.update({
                "baseline_col": "",
                "perturbed_col": "",
                "repair_gain_col": "",
                "core_reduction_col": "",
            })
            audit_rows.append(row)

    return maps, pd.DataFrame(audit_rows)


# =============================================================================
# Spatial smoothing and outline
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
    ax.set_title(title, fontsize=10.2, color="white", fontweight="bold", pad=4)


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

def make_a3_figure(meta, x, y, maps, outbase, point_size=2.0, dpi=600):
    row_names = [p["name"] for p in PERTURBATIONS]
    short_names = {p["name"]: p["short"] for p in PERTURBATIONS}

    n_rows = len(row_names)
    n_cols = 4

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(13.8, 2.35 * n_rows),
        facecolor="#05070b"
    )
    axes = np.asarray(axes).reshape(n_rows, n_cols)

    prob_norm = Normalize(0, 1)

    gain_vals = []
    delta_vals = []
    for rn in row_names:
        gain_vals.append(maps[rn]["repair_gain"][np.isfinite(maps[rn]["repair_gain"])])
        delta_vals.append(maps[rn]["core_reduction"][np.isfinite(maps[rn]["core_reduction"])])

    gain_max = np.nanquantile(np.concatenate(gain_vals), 0.99) if gain_vals else 1.0
    delta_max = np.nanquantile(np.concatenate(delta_vals), 0.99) if delta_vals else 1.0
    gain_max = float(max(gain_max, 1e-6))
    delta_max = float(max(delta_max, 1e-6))

    gain_norm = Normalize(0, gain_max)
    delta_norm = Normalize(0, delta_max)

    prob_cols = infer_prob_cols(meta)
    core = pd.to_numeric(meta[prob_cols["core"]], errors="coerce").to_numpy(dtype=float) if prob_cols["core"] in meta.columns else None
    peri = pd.to_numeric(meta[prob_cols["peri"]], errors="coerce").to_numpy(dtype=float) if prob_cols["peri"] in meta.columns else None

    col_titles = [
        "Baseline core map",
        "Perturbed core map",
        "Repair gain",
        "Core reduction",
    ]

    for i, rn in enumerate(row_names):
        vals = maps[rn]

        panel_specs = [
            ("baseline", cmap_core(), prob_norm),
            ("perturbed", cmap_perturbed(), prob_norm),
            ("repair_gain", cmap_gain(), gain_norm),
            ("core_reduction", cmap_delta(), delta_norm),
        ]

        for j, (key, cmap, norm) in enumerate(panel_specs):
            title = col_titles[j] if i == 0 else ""
            plot_landscape(
                axes[i, j],
                x,
                y,
                vals[key],
                cmap=cmap,
                norm=norm,
                title=title,
                core=core,
                peri=peri,
                point_size=point_size,
            )

        axes[i, 0].text(
            -0.12,
            0.5,
            short_names[rn],
            transform=axes[i, 0].transAxes,
            ha="right",
            va="center",
            color="white",
            fontsize=9.2,
            fontweight="bold",
        )

    fig.suptitle(
        "Before/after virtual perturbation spatial rescue atlas",
        fontsize=20,
        color="white",
        fontweight="bold",
        y=0.988,
    )

    add_fixed_colorbar(fig, cmap_core(), prob_norm, "Core probability", x=0.928, y=0.60, h=0.24, color="white")
    add_fixed_colorbar(fig, cmap_gain(), gain_norm, "Repair gain", x=0.928, y=0.34, h=0.20, color="white")
    add_fixed_colorbar(fig, cmap_delta(), delta_norm, "Core reduction", x=0.928, y=0.12, h=0.20, color="white")

    fig.text(
        0.5,
        0.012,
        "Maps show visualization-level virtual perturbation/state-editing proxies unless per-spot rescue-table columns are supplied; they are not wet-lab KO/blockade results or observed cell-fate transitions.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#e5e7eb"
    )

    fig.subplots_adjust(left=0.105, right=0.905, top=0.925, bottom=0.055, wspace=0.055, hspace=0.145)

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

    parser.add_argument("--rescue_table", default="")
    parser.add_argument("--adata_id_col", default="")
    parser.add_argument("--state_id_col", default="")
    parser.add_argument("--rescue_id_col", default="")
    parser.add_argument("--xcol", default="")
    parser.add_argument("--ycol", default="")

    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)

    print("=" * 100)
    print("Step75E | Before/after virtual perturbation spatial rescue atlas")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"rescue_table={args.rescue_table}")
    print(f"outdir={outdir}")

    adata = ad.read_h5ad(args.h5ad)
    obs = adata.obs.copy()

    adata_id, adata_id_col_used = get_obs_id_series(adata, obs, user_col=args.adata_id_col)
    obs["_adata_id_"] = adata_id.astype(str).values

    x, y, xcol_used, ycol_used, xy_mode = get_xy(adata, obs, xcol=args.xcol, ycol=args.ycol)
    obs["_spatial_x_"] = x
    obs["_spatial_y_"] = y

    state_df = read_table(args.state_table)

    meta, state_merge_audit = merge_state_table(
        obs,
        state_df,
        adata_id_col="_adata_id_",
        state_id_col=args.state_id_col,
    )

    rescue_df = read_table(args.rescue_table)
    meta, rescue_merge_audit = merge_rescue_table(
        meta,
        rescue_df,
        adata_id_col="_adata_id_",
        rescue_id_col=args.rescue_id_col,
    )

    prob_cols = infer_prob_cols(meta)
    if prob_cols["core"] is None or prob_cols["core"] not in meta.columns:
        raise ValueError("Cannot find core_probability after merge. Please check --state_table and --state_id_col.")

    x = pd.to_numeric(meta["_spatial_x_"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(meta["_spatial_y_"], errors="coerce").to_numpy(dtype=float)

    gmap = gene_lookup(adata)

    maps, perturb_audit = compute_maps_from_rescue_table_if_available(
        meta=meta,
        adata=adata,
        gmap=gmap,
        prob_cols=prob_cols,
    )

    outbase = outdir / "Fig_Step75E_BeforeAfterVirtualPerturbationSpatialRescueAtlas"

    make_a3_figure(
        meta=meta,
        x=x,
        y=y,
        maps=maps,
        outbase=str(outbase),
        point_size=args.point_size,
        dpi=args.dpi,
    )

    # Save per-spot values.
    values = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "spatial_x": x,
        "spatial_y": y,
    })

    for p in PERTURBATIONS:
        pname = p["name"]
        slug = slugify(pname)
        for metric in ["baseline", "perturbed", "repair_gain", "core_reduction"]:
            values[f"{slug}__{metric}"] = maps[pname][metric]

    value_path = outdir / "step75e_virtual_perturbation_spatial_values.csv"
    values.to_csv(value_path, index=False)

    perturb_audit_path = outdir / "step75e_perturbation_source_audit.csv"
    perturb_audit.to_csv(perturb_audit_path, index=False)

    audit = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "adata_id_col_used": adata_id_col_used,
        "xy_mode": xy_mode,
        "xcol_used": xcol_used,
        "ycol_used": ycol_used,
        "state_merge_audit": state_merge_audit,
        "rescue_merge_audit": rescue_merge_audit,
        "prob_cols": prob_cols,
        "perturbations": PERTURBATIONS,
        "outputs": {
            "figure_png": str(outbase.with_suffix(".png")),
            "figure_pdf": str(outbase.with_suffix(".pdf")),
            "figure_svg": str(outbase.with_suffix(".svg")),
            "spatial_values": str(value_path),
            "perturbation_source_audit": str(perturb_audit_path),
        },
        "interpretation_note": (
            "Step75E generates before/after virtual perturbation spatial maps. "
            "If no matched per-spot rescue-table columns are found, maps are derived from expression/state proxy scores. "
            "These visualizations should be described as virtual perturbation or state-editing hypotheses, "
            "not real KO/blockade experiments, therapeutic validation, or observed cell-fate transitions."
        ),
    }

    (outdir / "step75e_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step75e_report.txt").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    print("---- perturbation source audit ----")
    print(perturb_audit.to_string(index=False))
    print("=" * 100)
    print("DONE Step75E")
    print("=" * 100)
    print(json.dumps(audit["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
