#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step79 | State/timepoint-specific spatial progression atlas

Purpose
-------
Generate manuscript-style spatial progression atlases across disease timepoints
using true tissue coordinates.

Main outputs
------------
1) Fig_Step79A_StateModule_TimepointProgressionAtlas.{png,pdf,svg}
   Rows: state probabilities and module scores
   Columns: timepoints

2) Fig_Step79B_Marker_TimepointProgressionAtlas.{png,pdf,svg}
   Rows: representative marker genes
   Columns: timepoints

3) step79_per_spot_feature_table.csv
4) step79_audit_report.json

Interpretation
--------------
This is a visualization-level spatial progression atlas. State probabilities
come from the merged state table when available. Module scores are curated
gene-set scores and do not imply functional validation.
"""

from __future__ import annotations

import os
import re
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

try:
    import anndata as ad
except Exception as e:
    raise ImportError("Step79 requires anndata. Install with: pip install anndata") from e

try:
    from scipy import sparse
    from scipy.ndimage import gaussian_filter
except Exception as e:
    raise ImportError("Step79 requires scipy. Install with: pip install scipy") from e


STATE_PROB_CANDIDATES = {
    "core": [
        "core_probability", "core_prob", "p_core", "prob_core",
        "lesion_core_probability", "lesion_core_prob",
        "lesion-core-like_probability", "lesion_core_like_probability",
        "prob_lesion_core_like", "p_lesion_core_like",
    ],
    "peri": [
        "peri_probability", "peri_prob", "p_peri", "prob_peri",
        "peri_infarct_probability", "peri-infarct_probability",
        "peri_infarct_prob", "prob_peri_infarct",
    ],
    "remote": [
        "remote_probability", "remote_prob", "p_remote", "prob_remote",
        "remote_like_probability", "remote-like_probability",
        "remote_like_prob", "prob_remote_like",
    ],
    "repair": [
        "repair_score", "rescue_score", "repair_probability", "repair_prob",
        "repair_ecm_score", "repair_ECM_score", "repair_module_score",
        "state_repair_score",
    ],
}

DEFAULT_MODULE_SETS = {
    "repair_ECM": [
        "Col1a1", "Col3a1", "Fn1", "Spp1", "Vim", "Apoe", "Postn",
        "Timp1", "Col1a2", "Col5a1", "Sparc",
    ],
    "ferroptosis": [
        "Hmox1", "Fth1", "Slc7a11", "Gpx4", "Tfrc", "Acsl4", "Ptgs2",
        "Ftl1", "Ncoa4", "Nfe2l1", "Nfe2l2",
    ],
    "inflammation": [
        "C1qa", "C1qb", "Tyrobp", "Lgals3", "Ctsd", "Ccl2", "Ccl3",
        "Il1b", "Tnf", "Aif1", "Lyz2",
    ],
    "BBB_endothelial": [
        "Kdr", "Cldn5", "Pecam1", "Kcnj8", "Vwf", "Flt1", "Rgs5",
        "Klf2", "Tek", "Slc2a1",
    ],
    "astrocyte_reactive": [
        "Gfap", "Aqp4", "Vim", "Clu", "Apoe", "Serpina3n", "Lcn2",
        "S100b", "C3",
    ],
    "hypoxia_redox": [
        "Hmox1", "Fth1", "Nfe2l1", "Nfe2l2", "Hif1a", "Vegfa",
        "Sod2", "Nqo1", "Srxn1",
    ],
}

DEFAULT_MARKER_GROUPS = {
    "Microglia / inflammatory": ["C1qa", "C1qb", "Tyrobp", "Lgals3"],
    "Endothelial / BBB": ["Kdr", "Cldn5", "Pecam1", "Kcnj8"],
    "Ferroptosis / hypoxia": ["Hmox1", "Fth1", "Slc7a11", "Gpx4"],
    "Repair / ECM": ["Spp1", "Fn1", "Col1a1", "Col3a1"],
}

DEFAULT_FEATURES = [
    "core_probability",
    "peri_probability",
    "remote_probability",
    "repair_score",
    "repair_ECM",
    "ferroptosis",
    "inflammation",
    "BBB_endothelial",
    "astrocyte_reactive",
    "hypoxia_redox",
]

FEATURE_LABELS = {
    "core_probability": "Core probability",
    "peri_probability": "Peri-infarct probability",
    "remote_probability": "Remote-like probability",
    "repair_score": "Repair score",
    "repair_ECM": "Repair / ECM",
    "ferroptosis": "Ferroptosis",
    "inflammation": "Inflammation",
    "BBB_endothelial": "BBB / endothelial",
    "astrocyte_reactive": "Reactive astrocyte",
    "hypoxia_redox": "Hypoxia / redox",
}

TIMEPOINT_ORDER = {
    "sham": 0, "control": 0, "ctrl": 0, "normal": 0,
    "d0": 1, "day0": 1,
    "d1": 2, "day1": 2,
    "d3": 3, "day3": 3,
    "d5": 4, "day5": 4,
    "d7": 5, "day7": 5,
    "d14": 6, "day14": 6,
    "d21": 7, "day21": 7,
    "d28": 8, "day28": 8,
}


@dataclass
class FeatureSpec:
    name: str
    label: str
    kind: str
    source: str
    cmap: str
    vmin: float
    vmax: float


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clean_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).replace({"nan": np.nan, "None": np.nan, "NA": np.nan})


def robust_limits(x: np.ndarray, q_low: float = 1.0, q_high: float = 99.0,
                  symmetric: bool = False) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo = np.nanpercentile(x, q_low)
    hi = np.nanpercentile(x, q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0, 1.0
    if symmetric:
        m = max(abs(lo), abs(hi))
        if m == 0:
            m = 1.0
        return -m, m
    return float(lo), float(hi)


def robust_minmax(x: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = robust_limits(x, q_low=q_low, q_high=q_high, symmetric=False)
    y = (x - lo) / (hi - lo + 1e-12)
    return np.clip(y, 0, 1)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


def timepoint_sort_key(v: Any) -> Tuple[int, str]:
    s = str(v).strip()
    sl = s.lower().replace(" ", "").replace("_", "").replace("-", "")
    if sl in TIMEPOINT_ORDER:
        return TIMEPOINT_ORDER[sl], sl
    m = re.search(r"(?:d|day)(\d+)", sl)
    if m:
        return 10 + int(m.group(1)), sl
    m = re.search(r"(\d+)", sl)
    if m:
        return 100 + int(m.group(1)), sl
    return 999, sl


def parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None or str(s).strip() == "":
        return None
    return [x.strip() for x in str(s).split(",") if x.strip()]


def read_h5ad(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def infer_xy_columns(
    adata,
    force_x_col: Optional[str] = None,
    force_y_col: Optional[str] = None,
    force_obsm_spatial: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    obs = adata.obs

    if force_x_col and force_y_col:
        if force_x_col not in obs.columns or force_y_col not in obs.columns:
            raise ValueError(f"Forced xy columns not found: {force_x_col}, {force_y_col}")
        return (
            obs[force_x_col].astype(float).to_numpy(),
            obs[force_y_col].astype(float).to_numpy(),
            {"xy_mode": "obs_forced", "xcol": force_x_col, "ycol": force_y_col},
        )

    if force_obsm_spatial:
        if force_obsm_spatial not in adata.obsm:
            raise ValueError(f"--force_obsm_spatial={force_obsm_spatial} not found in adata.obsm")
        arr = np.asarray(adata.obsm[force_obsm_spatial])
        if arr.shape[1] < 2:
            raise ValueError(f"obsm[{force_obsm_spatial}] has fewer than 2 columns")
        return (
            arr[:, 0].astype(float),
            arr[:, 1].astype(float),
            {"xy_mode": "obsm_forced", "obsm_key": force_obsm_spatial},
        )

    xy_candidates = [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
        ("array_col", "array_row"),
        ("col", "row"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("imagecol", "imagerow"),
        ("spatial_col", "spatial_row"),
    ]
    for xcol, ycol in xy_candidates:
        if xcol in obs.columns and ycol in obs.columns:
            return (
                obs[xcol].astype(float).to_numpy(),
                obs[ycol].astype(float).to_numpy(),
                {"xy_mode": "obs_auto", "xcol": xcol, "ycol": ycol},
            )

    if "spatial" in adata.obsm:
        arr = np.asarray(adata.obsm["spatial"])
        if arr.shape[1] >= 2:
            return (
                arr[:, 0].astype(float),
                arr[:, 1].astype(float),
                {"xy_mode": "obsm_auto", "obsm_key": "spatial"},
            )

    raise ValueError(
        "Could not infer spatial coordinates. Use --x_col/--y_col or --force_obsm_spatial."
    )


def build_meta_from_adata(adata, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    meta = pd.DataFrame(index=adata.obs_names.copy())
    meta["_adata_id_"] = adata.obs_names.astype(str)
    meta["spatial_x"] = x
    meta["spatial_y"] = y

    for c in adata.obs.columns:
        if c not in meta.columns:
            meta[c] = adata.obs[c].values
    return meta


def read_state_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None or str(path).strip() == "":
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def merge_state_table(meta: pd.DataFrame, state: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    audit: Dict[str, Any] = {
        "state_table_available": state is not None,
        "merge_mode": None,
        "state_merge_overlap": None,
        "merge_key_meta": None,
        "merge_key_state": None,
    }
    if state is None:
        out = meta.copy()
        audit["merge_mode"] = "none"
        return out, audit

    st = state.copy()
    audit["state_table_rows"] = int(st.shape[0])

    key_candidates = [
        "obs_name", "_adata_id_", "barcode", "cell_id", "spot_id", "id",
        "cell", "spot", "index", "Unnamed: 0",
    ]

    meta_ids = meta["_adata_id_"].astype(str)
    best = None
    for k in key_candidates:
        if k in st.columns:
            st_ids = st[k].astype(str)
            overlap = len(set(meta_ids).intersection(set(st_ids)))
            if best is None or overlap > best[0]:
                best = (overlap, k)

    if best is not None and best[0] > 0:
        _, k = best
        st["_merge_id_"] = st[k].astype(str)
        tmp = meta.copy()
        tmp["_merge_id_"] = tmp["_adata_id_"].astype(str)
        out = tmp.merge(st, how="left", on="_merge_id_", suffixes=("", "_state"))
        out = out.drop(columns=["_merge_id_"])
        audit.update({
            "merge_mode": "id",
            "merge_key_meta": "_adata_id_",
            "merge_key_state": k,
            "state_merge_overlap": int(best[0]),
        })
        return out, audit

    if st.shape[0] == meta.shape[0]:
        out = meta.reset_index(drop=True).copy()
        st2 = st.reset_index(drop=True).copy()
        for c in st2.columns:
            if c in out.columns:
                out[f"{c}_state"] = st2[c]
            else:
                out[c] = st2[c]
        audit.update({
            "merge_mode": "row_order",
            "state_merge_overlap": int(meta.shape[0]),
        })
        return out, audit

    raise ValueError(
        "Could not merge state table by ID and row counts do not match. "
        "Add an obs_name/barcode column to state_table or use matching row order."
    )


def harmonize_time_col(meta: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, str]:
    candidates = [
        f"{time_col}_state", time_col,
        "timepoint_state", "timepoint",
        "time_point_state", "time_point",
        "day_state", "day",
        "sample_time_state", "sample_time",
    ]
    for c in candidates:
        if c in meta.columns:
            out = meta.copy()
            out["_step79_timepoint_"] = clean_str_series(out[c])
            return out, "_step79_timepoint_"
    raise ValueError(
        f"Could not find timepoint column. Tried {candidates}. "
        f"Use --time_col with a column present in adata.obs or state_table."
    )


def infer_state_prob_cols(meta: pd.DataFrame, args) -> Dict[str, Optional[str]]:
    forced = {
        "core": args.core_col,
        "peri": args.peri_col,
        "remote": args.remote_col,
        "repair": args.repair_col,
    }
    out = {}
    cols = list(meta.columns)
    lower_map = {c.lower(): c for c in cols}

    for k, forced_col in forced.items():
        if forced_col:
            if forced_col not in meta.columns:
                raise ValueError(f"Forced {k} column not found: {forced_col}")
            out[k] = forced_col
            continue

        found = None
        for cand in STATE_PROB_CANDIDATES[k]:
            if cand in meta.columns:
                found = cand
                break
            if cand.lower() in lower_map:
                found = lower_map[cand.lower()]
                break
            cand_state = f"{cand}_state"
            if cand_state in meta.columns:
                found = cand_state
                break
        out[k] = found

    return out


def make_gene_lookup(adata) -> Dict[str, str]:
    lookup = {}
    for g in map(str, adata.var_names):
        lookup[g.lower()] = g

    for col in ["gene_symbols", "gene_symbol", "symbol", "features", "feature_name"]:
        if col in adata.var.columns:
            vals = adata.var[col].astype(str).values
            for raw, actual in zip(vals, map(str, adata.var_names)):
                if raw and raw.lower() not in lookup:
                    lookup[raw.lower()] = actual
    return lookup


def get_gene_vector(adata, gene: str, lookup: Dict[str, str]) -> Optional[np.ndarray]:
    key = gene.lower()
    if key not in lookup:
        return None
    actual = lookup[key]
    idx = np.where(adata.var_names.astype(str) == actual)[0]
    if idx.size == 0:
        return None
    j = int(idx[0])
    Xj = adata.X[:, j]
    if sparse.issparse(Xj):
        arr = Xj.toarray().ravel()
    else:
        arr = np.asarray(Xj).ravel()
    return arr.astype(float)


def maybe_log1p_expr(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if mode == "yes":
        return np.log1p(np.maximum(x, 0))
    if mode == "no":
        return x

    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    if np.nanpercentile(finite, 99) > 20:
        return np.log1p(np.maximum(x, 0))
    return x


def compute_module_scores(
    adata,
    module_sets: Dict[str, List[str]],
    log1p_mode: str = "auto",
    min_genes: int = 2,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lookup = make_gene_lookup(adata)
    scores = pd.DataFrame(index=adata.obs_names.astype(str))
    audit: Dict[str, Any] = {}

    for module, genes in module_sets.items():
        used = []
        missing = []
        z_list = []

        for g in genes:
            v = get_gene_vector(adata, g, lookup)
            if v is None:
                missing.append(g)
                continue
            v = maybe_log1p_expr(v, log1p_mode)
            z_list.append(zscore(v))
            used.append(g)

        if len(z_list) >= min_genes:
            mat = np.vstack(z_list)
            sc = np.nanmean(mat, axis=0)
            sc = zscore(sc)
            scores[module] = sc
            status = "ok"
        elif len(z_list) > 0:
            mat = np.vstack(z_list)
            sc = np.nanmean(mat, axis=0)
            sc = zscore(sc)
            scores[module] = sc
            status = "few_genes"
        else:
            scores[module] = np.nan
            status = "missing_all"

        audit[module] = {
            "status": status,
            "n_genes_requested": len(genes),
            "n_genes_used": len(used),
            "genes_used": used,
            "genes_missing": missing,
        }

    return scores, audit


def compute_marker_values(
    adata,
    marker_groups: Dict[str, List[str]],
    log1p_mode: str = "auto",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    lookup = make_gene_lookup(adata)
    vals = pd.DataFrame(index=adata.obs_names.astype(str))
    audit: Dict[str, Any] = {}

    for group, genes in marker_groups.items():
        for g in genes:
            if g in vals.columns:
                continue
            v = get_gene_vector(adata, g, lookup)
            if v is None:
                vals[g] = np.nan
                audit[g] = {"group": group, "status": "missing"}
            else:
                v = maybe_log1p_expr(v, log1p_mode)
                vals[g] = v
                audit[g] = {
                    "group": group,
                    "status": "ok",
                    "display_limits": robust_limits(v, 0.5, 99.5),
                }

    return vals, audit


def make_global_extent(x: np.ndarray, y: np.ndarray, pad_frac: float = 0.04) -> Tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    ymin, ymax = np.nanmin(y), np.nanmax(y)

    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0:
        dx = 1.0
    if dy <= 0:
        dy = 1.0

    return (
        float(xmin - pad_frac * dx),
        float(xmax + pad_frac * dx),
        float(ymin - pad_frac * dy),
        float(ymax + pad_frac * dy),
    )


def grid_smooth(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    extent: Tuple[float, float, float, float],
    grid_n: int = 220,
    sigma: float = 1.8,
    density_quantile: float = 0.03,
    min_points: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(values, dtype=float)

    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[ok], y[ok], v[ok]

    xmin, xmax, ymin, ymax = extent
    nx = int(grid_n)
    aspect = (ymax - ymin) / (xmax - xmin + 1e-12)
    ny = max(40, int(round(nx * aspect)))
    ny = int(np.clip(ny, 60, grid_n * 2))

    x_edges = np.linspace(xmin, xmax, nx + 1)
    y_edges = np.linspace(ymin, ymax, ny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xg, Yg = np.meshgrid(x_centers, y_centers)

    if x.size < min_points:
        Z = np.full((ny, nx), np.nan)
        density = np.zeros((ny, nx), dtype=float)
        mask = np.zeros((ny, nx), dtype=bool)
        return Xg, Yg, Z, density, mask

    ix = np.searchsorted(x_edges, x, side="right") - 1
    iy = np.searchsorted(y_edges, y, side="right") - 1
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)

    ix, iy, v = ix[inside], iy[inside], v[inside]

    counts = np.zeros((ny, nx), dtype=float)
    sums = np.zeros((ny, nx), dtype=float)

    np.add.at(counts, (iy, ix), 1.0)
    np.add.at(sums, (iy, ix), v)

    density = gaussian_filter(counts, sigma=sigma, mode="nearest")
    smooth_sum = gaussian_filter(sums, sigma=sigma, mode="nearest")
    Z = smooth_sum / (density + 1e-12)

    if np.nanmax(density) > 0:
        vals = density[density > 0]
        thr = np.nanquantile(vals, density_quantile)
        mask = density >= thr
    else:
        mask = np.zeros_like(density, dtype=bool)

    Z = np.where(mask, Z, np.nan)
    return Xg, Yg, Z, density, mask


def draw_tissue_outline(ax, Xg, Yg, density, color="white", alpha=0.45, linewidth=0.7):
    if density is None or np.nanmax(density) <= 0:
        return

    vals = density[density > 0]
    if vals.size == 0:
        return

    level = np.nanquantile(vals, 0.03)
    try:
        ax.contour(
            Xg, Yg, density,
            levels=[level],
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            linestyles="dashed",
        )
    except Exception:
        pass


def draw_probability_contour(ax, Xg, Yg, Z, level, color, alpha=0.75, linewidth=0.8):
    if Z is None:
        return

    finite = Z[np.isfinite(Z)]
    if finite.size < 20:
        return

    zmin, zmax = np.nanmin(finite), np.nanmax(finite)
    if level <= zmin or level >= zmax:
        level2 = np.nanquantile(finite, 0.80)
    else:
        level2 = level

    try:
        ax.contour(
            Xg, Yg, Z,
            levels=[level2],
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
        )
    except Exception:
        pass


def setup_dark_axes(ax):
    ax.set_facecolor("#05080d")
    for spine in ax.spines.values():
        spine.set_color("#56606d")
        spine.set_linewidth(0.5)
    ax.tick_params(colors="#b8c0cc", labelsize=7, length=2)
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")


def prepare_probability_context(
    df_sub: pd.DataFrame,
    prob_cols: Dict[str, Optional[str]],
    extent,
    grid_n,
    sigma,
    density_quantile,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    ctx = {}

    for key in ["core", "peri"]:
        col = prob_cols.get(key)
        if col is None or col not in df_sub.columns:
            continue

        Xg, Yg, Z, density, mask = grid_smooth(
            df_sub["spatial_x"].to_numpy(float),
            df_sub["spatial_y"].to_numpy(float),
            df_sub[col].to_numpy(float),
            extent=extent,
            grid_n=grid_n,
            sigma=sigma,
            density_quantile=density_quantile,
        )
        ctx[key] = (Xg, Yg, Z)

    return ctx


def plot_spatial_panel(
    ax,
    df_sub: pd.DataFrame,
    value_col: str,
    spec: FeatureSpec,
    extent: Tuple[float, float, float, float],
    grid_n: int,
    sigma: float,
    density_quantile: float,
    scatter_size: float,
    scatter_alpha: float,
    show_spots: bool,
    show_outline: bool,
    contour_context: Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    core_contour_level: float = 0.45,
    peri_contour_level: float = 0.45,
):
    setup_dark_axes(ax)

    if df_sub.shape[0] == 0 or value_col not in df_sub.columns:
        ax.text(
            0.5, 0.5, "missing",
            transform=ax.transAxes,
            color="white",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    x = df_sub["spatial_x"].to_numpy(dtype=float)
    y = df_sub["spatial_y"].to_numpy(dtype=float)
    val = df_sub[value_col].to_numpy(dtype=float)

    Xg, Yg, Z, density, mask = grid_smooth(
        x, y, val,
        extent=extent,
        grid_n=grid_n,
        sigma=sigma,
        density_quantile=density_quantile,
    )

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[extent[0], extent[1], extent[2], extent[3]],
        cmap=spec.cmap,
        vmin=spec.vmin,
        vmax=spec.vmax,
        interpolation="bilinear",
        alpha=0.97,
        aspect="auto",
    )

    finite = Z[np.isfinite(Z)]
    if finite.size > 30:
        try:
            lo = max(spec.vmin, np.nanpercentile(finite, 10))
            hi = min(spec.vmax, np.nanpercentile(finite, 95))
            levels = np.linspace(lo, hi, 5)
            levels = np.unique(levels[np.isfinite(levels)])
            if levels.size > 1:
                ax.contour(
                    Xg, Yg, Z,
                    levels=levels,
                    colors="#c7d4df",
                    linewidths=0.35,
                    alpha=0.38,
                )
        except Exception:
            pass

    if show_spots:
        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
        if ok.sum() > 0:
            ax.scatter(
                x[ok], y[ok],
                c=val[ok],
                s=scatter_size,
                cmap=spec.cmap,
                vmin=spec.vmin,
                vmax=spec.vmax,
                alpha=scatter_alpha,
                linewidths=0,
                rasterized=True,
            )

    if show_outline:
        draw_tissue_outline(ax, Xg, Yg, density)

    if contour_context is not None:
        if "core" in contour_context:
            Xc, Yc, Zc = contour_context["core"]
            draw_probability_contour(
                ax, Xc, Yc, Zc,
                level=core_contour_level,
                color="#ff5c63",
                alpha=0.75,
                linewidth=0.75,
            )
        if "peri" in contour_context:
            Xp, Yp, Zp = contour_context["peri"]
            draw_probability_contour(
                ax, Xp, Yp, Zp,
                level=peri_contour_level,
                color="#ffd447",
                alpha=0.70,
                linewidth=0.65,
            )

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def make_state_module_atlas(
    df: pd.DataFrame,
    time_col: str,
    timepoints: List[str],
    features: List[str],
    feature_specs: Dict[str, FeatureSpec],
    prob_cols: Dict[str, Optional[str]],
    outdir: str,
    args,
) -> Dict[str, str]:
    n_rows = len(features)
    n_cols = len(timepoints)
    if n_rows == 0 or n_cols == 0:
        return {}

    extent = make_global_extent(
        df["spatial_x"].to_numpy(float),
        df["spatial_y"].to_numpy(float),
        pad_frac=args.extent_pad,
    )

    fig_w = max(9.0, args.panel_w * n_cols + 2.6)
    fig_h = max(7.0, args.panel_h * n_rows + 1.8)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#05080d", dpi=args.dpi)

    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        left=args.left_margin,
        right=0.86,
        bottom=args.bottom_margin,
        top=0.90,
        wspace=0.04,
        hspace=0.08,
    )

    time_ctx: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    for tp in timepoints:
        sub = df[df[time_col].astype(str) == str(tp)]
        time_ctx[tp] = prepare_probability_context(
            sub,
            prob_cols,
            extent,
            args.grid_n,
            args.sigma,
            args.density_quantile,
        )

    for i, feat in enumerate(features):
        spec = feature_specs[feat]
        for j, tp in enumerate(timepoints):
            ax = fig.add_subplot(gs[i, j])
            sub = df[df[time_col].astype(str) == str(tp)]

            plot_spatial_panel(
                ax,
                sub,
                feat,
                spec,
                extent,
                grid_n=args.grid_n,
                sigma=args.sigma,
                density_quantile=args.density_quantile,
                scatter_size=args.scatter_size,
                scatter_alpha=args.scatter_alpha,
                show_spots=bool(args.show_spots),
                show_outline=bool(args.show_outline),
                contour_context=time_ctx.get(tp) if args.show_roi_contours else None,
                core_contour_level=args.core_contour,
                peri_contour_level=args.peri_contour,
            )

            if i == 0:
                ax.set_title(
                    str(tp),
                    color="white",
                    fontsize=args.col_title_size,
                    weight="bold",
                    pad=8,
                )
            if j == 0:
                ax.text(
                    -0.10,
                    0.5,
                    spec.label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    color="white",
                    fontsize=args.row_label_size,
                    weight="bold",
                    clip_on=False,
                )

    fig.suptitle(
        args.title,
        color="white",
        fontsize=args.title_size,
        weight="bold",
        y=0.975,
    )

    has_prob = any(feature_specs[f].kind == "prob" for f in features)
    has_score = any(feature_specs[f].kind == "score" for f in features)

    if has_prob:
        cax1 = fig.add_axes([0.885, 0.56, 0.015, 0.28])
        sm1 = ScalarMappable(norm=Normalize(0, 1), cmap=args.prob_cmap)
        cb1 = fig.colorbar(sm1, cax=cax1)
        cb1.set_label("State probability", color="white", fontsize=9)
        cb1.ax.tick_params(colors="white", labelsize=8)
        cb1.outline.set_edgecolor("white")

    if has_score:
        cax2 = fig.add_axes([0.885, 0.19, 0.015, 0.28])
        vmax = args.module_zmax
        sm2 = ScalarMappable(norm=Normalize(-vmax, vmax), cmap=args.module_cmap)
        cb2 = fig.colorbar(sm2, cax=cax2)
        cb2.set_label("Module / score (z)", color="white", fontsize=9)
        cb2.ax.tick_params(colors="white", labelsize=8)
        cb2.outline.set_edgecolor("white")

    note = (
        "State/timepoint-specific spatial progression atlas. Smoothed maps use Gaussian-kernel "
        "spot aggregation; dashed contours indicate tissue support; colored contours indicate "
        "core/peri regions when available. Module scores are curated gene-set scores, not functional validation."
    )
    fig.text(
        0.5,
        0.025,
        note,
        color="#d8dde5",
        fontsize=args.note_size,
        ha="center",
        va="bottom",
        wrap=True,
    )

    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(outdir, f"Fig_Step79A_StateModule_TimepointProgressionAtlas.{ext}")
        if ext == "png":
            fig.savefig(
                path,
                dpi=args.dpi,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.22,
            )
        else:
            fig.savefig(
                path,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.22,
            )
        paths[ext] = path

    plt.close(fig)
    return paths


def make_marker_progression_atlas(
    df: pd.DataFrame,
    time_col: str,
    timepoints: List[str],
    marker_groups: Dict[str, List[str]],
    marker_specs: Dict[str, FeatureSpec],
    prob_cols: Dict[str, Optional[str]],
    outdir: str,
    args,
) -> Dict[str, str]:
    markers = []
    marker_to_group = {}

    for group, genes in marker_groups.items():
        for g in genes:
            if g not in markers:
                markers.append(g)
                marker_to_group[g] = group

    markers = [
        g for g in markers
        if g in df.columns and np.isfinite(df[g].to_numpy(float)).sum() > 0
    ]
    if not markers:
        return {}

    n_rows = len(markers)
    n_cols = len(timepoints)

    extent = make_global_extent(
        df["spatial_x"].to_numpy(float),
        df["spatial_y"].to_numpy(float),
        pad_frac=args.extent_pad,
    )

    fig_w = max(9.0, args.panel_w * n_cols + 2.2)
    fig_h = max(8.0, args.marker_panel_h * n_rows + 1.8)
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#05080d", dpi=args.dpi)

    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        left=args.left_margin,
        right=0.86,
        bottom=args.bottom_margin,
        top=0.91,
        wspace=0.04,
        hspace=0.08,
    )

    time_ctx = {}
    for tp in timepoints:
        sub = df[df[time_col].astype(str) == str(tp)]
        time_ctx[tp] = prepare_probability_context(
            sub,
            prob_cols,
            extent,
            args.grid_n,
            args.sigma,
            args.density_quantile,
        )

    last_group = None

    for i, gene in enumerate(markers):
        spec = marker_specs[gene]
        group = marker_to_group.get(gene, "")

        for j, tp in enumerate(timepoints):
            ax = fig.add_subplot(gs[i, j])
            sub = df[df[time_col].astype(str) == str(tp)]

            plot_spatial_panel(
                ax,
                sub,
                gene,
                spec,
                extent,
                grid_n=args.grid_n,
                sigma=args.sigma,
                density_quantile=args.density_quantile,
                scatter_size=args.scatter_size,
                scatter_alpha=args.scatter_alpha,
                show_spots=bool(args.show_spots),
                show_outline=bool(args.show_outline),
                contour_context=time_ctx.get(tp) if args.show_roi_contours else None,
                core_contour_level=args.core_contour,
                peri_contour_level=args.peri_contour,
            )

            if i == 0:
                ax.set_title(
                    str(tp),
                    color="white",
                    fontsize=args.col_title_size,
                    weight="bold",
                    pad=8,
                )

            if j == 0:
                ax.text(
                    -0.10,
                    0.5,
                    gene,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    color="white",
                    fontsize=args.row_label_size,
                    weight="bold",
                    clip_on=False,
                )

                if group != last_group:
                    ax.text(
                        -0.28,
                        0.5,
                        group,
                        transform=ax.transAxes,
                        ha="right",
                        va="center",
                        color="#d8dde5",
                        fontsize=max(7, args.row_label_size - 1),
                        weight="bold",
                        clip_on=False,
                    )
                    last_group = group

    fig.suptitle(
        "Spatial progression atlas of representative marker genes",
        color="white",
        fontsize=args.title_size,
        weight="bold",
        y=0.98,
    )

    cax = fig.add_axes([0.885, 0.32, 0.015, 0.38])
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=args.marker_cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Expression, robust-scaled per gene", color="white", fontsize=9)
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.outline.set_edgecolor("white")

    note = (
        "Rows show representative genes grouped by biological program; columns show disease timepoints. "
        "Expression is log/normalized as available and robust-scaled per gene for visualization."
    )
    fig.text(
        0.5,
        0.025,
        note,
        color="#d8dde5",
        fontsize=args.note_size,
        ha="center",
        va="bottom",
        wrap=True,
    )

    paths = {}
    for ext in ["png", "pdf", "svg"]:
        path = os.path.join(outdir, f"Fig_Step79B_Marker_TimepointProgressionAtlas.{ext}")
        if ext == "png":
            fig.savefig(
                path,
                dpi=args.dpi,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.22,
            )
        else:
            fig.savefig(
                path,
                facecolor=fig.get_facecolor(),
                bbox_inches="tight",
                pad_inches=0.22,
            )
        paths[ext] = path

    plt.close(fig)
    return paths


def main():
    p = argparse.ArgumentParser(
        description="Step79 | State/timepoint-specific spatial progression atlas"
    )

    p.add_argument("--h5ad", required=True)
    p.add_argument("--state_table", default="")
    p.add_argument("--outdir", required=True)

    p.add_argument("--time_col", default="timepoint")
    p.add_argument("--timepoints", default="", help="Comma-separated order, e.g. D1,D3,D7. Empty=auto.")
    p.add_argument("--x_col", default="")
    p.add_argument("--y_col", default="")
    p.add_argument("--force_obsm_spatial", default="")

    p.add_argument("--core_col", default="")
    p.add_argument("--peri_col", default="")
    p.add_argument("--remote_col", default="")
    p.add_argument("--repair_col", default="")

    p.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    p.add_argument("--make_marker_progression", type=int, default=1)

    p.add_argument("--log1p_expr", choices=["auto", "yes", "no"], default="auto")
    p.add_argument("--min_module_genes", type=int, default=2)

    p.add_argument("--grid_n", type=int, default=240)
    p.add_argument("--sigma", type=float, default=1.8)
    p.add_argument("--density_quantile", type=float, default=0.03)
    p.add_argument("--extent_pad", type=float, default=0.04)

    p.add_argument("--show_spots", type=int, default=1)
    p.add_argument("--show_outline", type=int, default=1)
    p.add_argument("--show_roi_contours", type=int, default=1)
    p.add_argument("--core_contour", type=float, default=0.45)
    p.add_argument("--peri_contour", type=float, default=0.45)
    p.add_argument("--scatter_size", type=float, default=1.6)
    p.add_argument("--scatter_alpha", type=float, default=0.24)

    p.add_argument("--prob_cmap", default="magma")
    p.add_argument("--module_cmap", default="RdBu_r")
    p.add_argument("--marker_cmap", default="magma")
    p.add_argument("--module_zmax", type=float, default=2.5)

    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--panel_w", type=float, default=3.0)
    p.add_argument("--panel_h", type=float, default=2.35)
    p.add_argument("--marker_panel_h", type=float, default=1.65)
    p.add_argument("--left_margin", type=float, default=0.18)
    p.add_argument("--bottom_margin", type=float, default=0.07)
    p.add_argument("--title_size", type=int, default=18)
    p.add_argument("--col_title_size", type=int, default=11)
    p.add_argument("--row_label_size", type=int, default=9)
    p.add_argument("--note_size", type=int, default=8)
    p.add_argument(
        "--title",
        default="State/timepoint-specific spatial progression landscape",
    )

    args = p.parse_args()

    ensure_dir(args.outdir)

    print("=" * 100)
    print("Step79 | State/timepoint-specific spatial progression atlas")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={args.outdir}")

    adata = read_h5ad(args.h5ad)

    x, y, xy_audit = infer_xy_columns(
        adata,
        force_x_col=args.x_col or None,
        force_y_col=args.y_col or None,
        force_obsm_spatial=args.force_obsm_spatial or None,
    )

    meta = build_meta_from_adata(adata, x, y)

    state = read_state_table(args.state_table) if args.state_table else None
    meta, merge_audit = merge_state_table(meta, state)
    meta, time_col = harmonize_time_col(meta, args.time_col)

    requested_tps = parse_csv_list(args.timepoints)
    observed_tps = [x for x in meta[time_col].dropna().astype(str).unique().tolist()]
    observed_tps = sorted(observed_tps, key=timepoint_sort_key)

    if requested_tps:
        timepoints = [tp for tp in requested_tps if tp in observed_tps]
        missing_tps = [tp for tp in requested_tps if tp not in observed_tps]
        if len(timepoints) == 0:
            raise ValueError(
                f"None of requested --timepoints were found. "
                f"requested={requested_tps}, observed={observed_tps}"
            )
    else:
        timepoints = observed_tps
        missing_tps = []

    prob_cols = infer_state_prob_cols(meta, args)
    for key, col in prob_cols.items():
        if col is not None and col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce")

    module_scores, module_audit = compute_module_scores(
        adata,
        DEFAULT_MODULE_SETS,
        log1p_mode=args.log1p_expr,
        min_genes=args.min_module_genes,
    )

    module_scores["_adata_id_"] = module_scores.index.astype(str)
    meta = meta.merge(module_scores.reset_index(drop=True), how="left", on="_adata_id_")

    marker_values, marker_audit = compute_marker_values(
        adata,
        DEFAULT_MARKER_GROUPS,
        log1p_mode=args.log1p_expr,
    )

    marker_values["_adata_id_"] = marker_values.index.astype(str)
    meta = meta.merge(marker_values.reset_index(drop=True), how="left", on="_adata_id_")

    alias_to_col = {
        "core_probability": prob_cols.get("core"),
        "peri_probability": prob_cols.get("peri"),
        "remote_probability": prob_cols.get("remote"),
        "repair_score": prob_cols.get("repair"),
    }

    if alias_to_col["repair_score"] is None and "repair_ECM" in meta.columns:
        alias_to_col["repair_score"] = "repair_ECM"

    for alias, col in alias_to_col.items():
        if col is not None and col in meta.columns:
            meta[alias] = pd.to_numeric(meta[col], errors="coerce")
        else:
            meta[alias] = np.nan

    features_requested = parse_csv_list(args.features) or DEFAULT_FEATURES
    features = []

    for f in features_requested:
        if f in meta.columns and np.isfinite(pd.to_numeric(meta[f], errors="coerce")).sum() > 0:
            features.append(f)
        else:
            warnings.warn(f"Feature not available or all missing; skipped: {f}")

    feature_specs: Dict[str, FeatureSpec] = {}

    for f in features:
        if f in ["core_probability", "peri_probability", "remote_probability"]:
            feature_specs[f] = FeatureSpec(
                name=f,
                label=FEATURE_LABELS.get(f, f),
                kind="prob",
                source=alias_to_col.get(f, f),
                cmap=args.prob_cmap,
                vmin=0.0,
                vmax=1.0,
            )
        elif f == "repair_score" and prob_cols.get("repair") is not None:
            vals = meta[f].to_numpy(float)
            if np.nanmin(vals) >= -0.05 and np.nanmax(vals) <= 1.05:
                feature_specs[f] = FeatureSpec(
                    name=f,
                    label=FEATURE_LABELS.get(f, f),
                    kind="prob",
                    source=alias_to_col.get(f, f),
                    cmap=args.prob_cmap,
                    vmin=0.0,
                    vmax=1.0,
                )
            else:
                vmax = args.module_zmax
                feature_specs[f] = FeatureSpec(
                    name=f,
                    label=FEATURE_LABELS.get(f, f),
                    kind="score",
                    source=alias_to_col.get(f, f),
                    cmap=args.module_cmap,
                    vmin=-vmax,
                    vmax=vmax,
                )
        else:
            vmax = args.module_zmax
            feature_specs[f] = FeatureSpec(
                name=f,
                label=FEATURE_LABELS.get(f, f),
                kind="score",
                source="curated_module_score",
                cmap=args.module_cmap,
                vmin=-vmax,
                vmax=vmax,
            )

    marker_specs: Dict[str, FeatureSpec] = {}

    for group, genes in DEFAULT_MARKER_GROUPS.items():
        for g in genes:
            if g not in meta.columns:
                continue
            raw = meta[g].to_numpy(float)
            if np.isfinite(raw).sum() == 0:
                continue

            disp_col = f"{g}__display01"
            meta[disp_col] = robust_minmax(raw, q_low=0.5, q_high=99.5)
            meta[g] = meta[disp_col]

            marker_specs[g] = FeatureSpec(
                name=g,
                label=g,
                kind="marker",
                source="expression_robust_scaled",
                cmap=args.marker_cmap,
                vmin=0.0,
                vmax=1.0,
            )

    keep_cols = ["_adata_id_", "spatial_x", "spatial_y", time_col]
    keep_cols += [
        c for c in [
            "core_probability",
            "peri_probability",
            "remote_probability",
            "repair_score",
        ]
        if c in meta.columns
    ]
    keep_cols += [m for m in DEFAULT_MODULE_SETS.keys() if m in meta.columns]
    keep_cols += [g for g in marker_specs.keys() if g in meta.columns]
    keep_cols = list(dict.fromkeys([c for c in keep_cols if c in meta.columns]))

    per_spot_path = os.path.join(args.outdir, "step79_per_spot_feature_table.csv")
    meta[keep_cols].to_csv(per_spot_path, index=False)

    state_paths = make_state_module_atlas(
        meta,
        time_col,
        timepoints,
        features,
        feature_specs,
        prob_cols,
        args.outdir,
        args,
    )

    marker_paths = {}
    if int(args.make_marker_progression) == 1:
        marker_paths = make_marker_progression_atlas(
            meta,
            time_col,
            timepoints,
            DEFAULT_MARKER_GROUPS,
            marker_specs,
            prob_cols,
            args.outdir,
            args,
        )

    audit = {
        "status": "ok",
        "h5ad": args.h5ad,
        "state_table": args.state_table,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "xy_audit": xy_audit,
        "state_merge_audit": merge_audit,
        "time_col_used": time_col,
        "timepoints_observed": observed_tps,
        "timepoints_used": timepoints,
        "timepoints_requested_missing": missing_tps,
        "prob_cols": prob_cols,
        "features_requested": features_requested,
        "features_used": features,
        "module_sets": DEFAULT_MODULE_SETS,
        "module_audit": module_audit,
        "marker_groups": DEFAULT_MARKER_GROUPS,
        "marker_audit": marker_audit,
        "outputs": {
            "state_module_atlas": state_paths,
            "marker_progression_atlas": marker_paths,
            "per_spot_feature_table": per_spot_path,
            "audit_report": os.path.join(args.outdir, "step79_audit_report.json"),
        },
        "interpretation_note": (
            "Step79 generates timepoint-specific spatial progression atlases. "
            "State probabilities are taken from the merged state table when available. "
            "Module scores are curated transcript-level gene-set scores. "
            "Smoothed maps are visualization-level spatial landscapes and do not imply "
            "functional validation or observed cell-state transition."
        ),
    }

    audit_path = os.path.join(args.outdir, "step79_audit_report.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)

    print("\nFinished Step79.")
    print(f"State/module atlas: {state_paths}")
    print(f"Marker atlas: {marker_paths}")
    print(f"Per-spot table: {per_spot_path}")
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
