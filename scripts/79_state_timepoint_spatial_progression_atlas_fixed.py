#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step79 fixed | State/timepoint-specific spatial progression atlas

Outputs:
  1) Fig_Step79A_StateModule_TimepointProgressionAtlas_FIXED.{png,pdf,svg}
  2) Fig_Step79B_Marker_TimepointProgressionAtlas_FIXED.{png,pdf,svg}
  3) Fig_Step79C_ProbabilityColumnAudit_FIXED.{png,pdf,svg}
  4) step79_fixed_audit_report.json
  5) step79_probability_column_audit.csv

Main fixes:
  - Explicitly audits core/peri/remote probability columns.
  - Stops if probability columns are duplicated.
  - Uses dedicated left-label column to avoid clipping.
  - Uses robust per-gene marker scaling for the marker atlas.
  - Adds tissue boundary and optional core/peri contours.
  - Saves publication-ready PNG/PDF/SVG.
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

try:
    import anndata as ad
except Exception as e:
    raise ImportError("Please install anndata first: pip install anndata") from e

try:
    from scipy import sparse
    from scipy.ndimage import gaussian_filter
except Exception as e:
    raise ImportError("Please install scipy first: pip install scipy") from e


# -----------------------------
# Basic style
# -----------------------------

BG = "#070b10"
FG = "#f2f2f2"
MUTED = "#b7c0cc"
GRID = "#1d2733"

CMAP_EXPR = "magma"
CMAP_PROB = "magma"
CMAP_MODULE = "RdBu_r"

CORE_CONTOUR_COLOR = "#ff9f40"
PERI_CONTOUR_COLOR = "#f4d35e"
TISSUE_CONTOUR_COLOR = "#9aa4b2"


# -----------------------------
# Utility
# -----------------------------

def mkdirp(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def clean_label(x):
    return str(x).replace("_", " ").replace("-", "–")


def natural_time_key(x):
    s = str(x)
    m = re.search(r"(\d+)", s)
    if m:
        return int(m.group(1))
    return s


def as_dense_1d(x):
    if sparse.issparse(x):
        return np.asarray(x.toarray()).ravel()
    return np.asarray(x).ravel()


def robust_minmax(v, q_low=1.0, q_high=99.0):
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return out
    lo = np.nanpercentile(v[finite], q_low)
    hi = np.nanpercentile(v[finite], q_high)
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = np.nanmax(v[finite])
        lo = np.nanmin(v[finite])
    if hi <= lo:
        return out
    out[finite] = np.clip((v[finite] - lo) / (hi - lo), 0, 1)
    out[~finite] = np.nan
    return out


def robust_z(v):
    v = np.asarray(v, dtype=float)
    out = np.zeros_like(v, dtype=float)
    finite = np.isfinite(v)
    if finite.sum() == 0:
        return out
    med = np.nanmedian(v[finite])
    mad = np.nanmedian(np.abs(v[finite] - med))
    if mad <= 1e-12:
        sd = np.nanstd(v[finite])
        if sd <= 1e-12:
            return out
        out[finite] = (v[finite] - np.nanmean(v[finite])) / sd
    else:
        out[finite] = (v[finite] - med) / (1.4826 * mad)
    out[~finite] = np.nan
    return out


def save_all(fig, outbase, dpi=450, formats=("png", "pdf", "svg")):
    paths = {}
    for ext in formats:
        p = f"{outbase}.{ext}"
        fig.savefig(
            p,
            dpi=dpi if ext == "png" else None,
            facecolor=fig.get_facecolor(),
            edgecolor="none",
            bbox_inches="tight",
            pad_inches=0.18
        )
        paths[ext] = p
    return paths


# -----------------------------
# Loading and metadata
# -----------------------------

def read_h5ad(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return ad.read_h5ad(path)


def infer_xy(adata, x_col="auto", y_col="auto"):
    obs = adata.obs

    if x_col != "auto" and y_col != "auto":
        if x_col not in obs.columns or y_col not in obs.columns:
            raise ValueError(f"Requested x/y columns not found: {x_col}, {y_col}")
        return np.asarray(obs[x_col], dtype=float), np.asarray(obs[y_col], dtype=float), {
            "mode": "obs_forced",
            "xcol": x_col,
            "ycol": y_col
        }

    candidate_pairs = [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
        ("array_col", "array_row"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("imagecol", "imagerow"),
    ]

    for xc, yc in candidate_pairs:
        if xc in obs.columns and yc in obs.columns:
            return np.asarray(obs[xc], dtype=float), np.asarray(obs[yc], dtype=float), {
                "mode": "obs_auto",
                "xcol": xc,
                "ycol": yc
            }

    if "spatial" in adata.obsm.keys():
        xy = np.asarray(adata.obsm["spatial"])
        if xy.shape[1] < 2:
            raise ValueError("adata.obsm['spatial'] has fewer than 2 columns.")
        return xy[:, 0].astype(float), xy[:, 1].astype(float), {
            "mode": "obsm_spatial",
            "obsm_key": "spatial",
            "xcol": "spatial[:,0]",
            "ycol": "spatial[:,1]"
        }

    raise ValueError("Could not infer spatial coordinates. Please provide --x_col and --y_col.")


def merge_state_table(adata, state_table_path, state_id_col="auto"):
    base = pd.DataFrame({"_adata_id_": adata.obs_names.astype(str)})
    obs = adata.obs.copy()
    obs["_adata_id_"] = adata.obs_names.astype(str)

    if not state_table_path:
        merged = base.merge(obs, on="_adata_id_", how="left")
        return merged, {
            "state_table_available": False,
            "merge_mode": "obs_only",
            "state_merge_overlap": None
        }

    if not os.path.exists(state_table_path):
        raise FileNotFoundError(state_table_path)

    st = pd.read_csv(state_table_path)
    st_cols = list(st.columns)

    if state_id_col != "auto":
        if state_id_col not in st.columns:
            raise ValueError(f"--state_id_col={state_id_col} not found in state table.")
        key = state_id_col
    else:
        candidates = [
            "obs_name", "barcode", "cell", "cell_id", "spot", "spot_id",
            "_adata_id_", "index", "Unnamed: 0"
        ]
        key = None
        for c in candidates:
            if c in st.columns:
                overlap = len(set(base["_adata_id_"]).intersection(set(st[c].astype(str))))
                if overlap > 0:
                    key = c
                    break
        if key is None:
            # Last fallback: first column
            first = st.columns[0]
            overlap = len(set(base["_adata_id_"]).intersection(set(st[first].astype(str))))
            if overlap > 0:
                key = first
            else:
                raise ValueError(
                    "Cannot infer state table ID column. "
                    "Please provide --state_id_col, e.g. --state_id_col obs_name"
                )

    st[key] = st[key].astype(str)
    merged = base.merge(st, left_on="_adata_id_", right_on=key, how="left", suffixes=("", "_state"))

    # add obs columns not already present
    for c in obs.columns:
        if c not in merged.columns:
            merged[c] = obs[c].values

    overlap = int(merged[key].notna().sum()) if key in merged.columns else int(merged.drop(columns=["_adata_id_"]).notna().any(axis=1).sum())

    audit = {
        "state_table_available": True,
        "state_table_rows": int(st.shape[0]),
        "state_table_cols": st_cols,
        "merge_mode": "id",
        "merge_key_meta": "_adata_id_",
        "merge_key_state": key,
        "state_merge_overlap": overlap,
        "n_obs": int(adata.n_obs)
    }

    if overlap < adata.n_obs * 0.5:
        warnings.warn(f"Low state-table merge overlap: {overlap}/{adata.n_obs}")

    return merged, audit


# -----------------------------
# Probability column inference and audit
# -----------------------------

def pick_col(df, forced, candidates, label):
    if forced and forced != "auto":
        if forced not in df.columns:
            raise ValueError(f"Forced {label} column not found: {forced}")
        return forced

    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    # contains-based fallback
    for c in df.columns:
        cl = c.lower()
        if label == "core" and ("core" in cl and ("prob" in cl or "probability" in cl)):
            return c
        if label == "peri" and (("peri" in cl or "penumbra" in cl) and ("prob" in cl or "probability" in cl)):
            return c
        if label == "remote" and ("remote" in cl and ("prob" in cl or "probability" in cl)):
            return c

    raise ValueError(f"Could not infer {label} probability column.")


def infer_prob_cols(df, args):
    core = pick_col(df, args.core_prob_col, [
        "core_probability", "core_prob", "prob_core", "p_core",
        "lesion_core_probability", "lesion_core_like_probability",
        "lesion-core-like_probability", "p_lesion_core_like"
    ], "core")

    peri = pick_col(df, args.peri_prob_col, [
        "peri_probability", "peri_prob", "prob_peri", "p_peri",
        "peri_infarct_probability", "periinfarct_probability",
        "peri-infarct_probability", "p_peri_infarct"
    ], "peri")

    remote = pick_col(df, args.remote_prob_col, [
        "remote_probability", "remote_prob", "prob_remote", "p_remote",
        "remote_like_probability", "remote-like_probability",
        "p_remote_like"
    ], "remote")

    cols = {"core": core, "peri": peri, "remote": remote}

    if len(set(cols.values())) < 3:
        raise ValueError(
            f"Probability columns are not distinct: {cols}. "
            "Please force correct columns with --core_prob_col, --peri_prob_col, --remote_prob_col."
        )

    return cols


def probability_audit(df, prob_cols, allow_identical=False, high_corr_warn=0.98):
    labels = list(prob_cols.keys())
    rows = []
    mat_pearson = pd.DataFrame(index=labels, columns=labels, dtype=float)
    mat_spearman = pd.DataFrame(index=labels, columns=labels, dtype=float)
    exact_duplicates = []
    high_corr_pairs = []

    for a in labels:
        va = pd.to_numeric(df[prob_cols[a]], errors="coerce").astype(float)
        for b in labels:
            vb = pd.to_numeric(df[prob_cols[b]], errors="coerce").astype(float)
            valid = np.isfinite(va) & np.isfinite(vb)
            if valid.sum() < 5:
                pear = np.nan
                spear = np.nan
                exact = False
                max_abs_diff = np.nan
            else:
                pear = np.corrcoef(va[valid], vb[valid])[0, 1]
                spear = pd.Series(va[valid]).corr(pd.Series(vb[valid]), method="spearman")
                max_abs_diff = float(np.nanmax(np.abs(va[valid] - vb[valid])))
                exact = bool(np.allclose(va[valid], vb[valid], equal_nan=True, atol=1e-12))

            mat_pearson.loc[a, b] = pear
            mat_spearman.loc[a, b] = spear
            rows.append({
                "a": a,
                "b": b,
                "col_a": prob_cols[a],
                "col_b": prob_cols[b],
                "pearson": pear,
                "spearman": spear,
                "max_abs_diff": max_abs_diff,
                "exact_equal": exact
            })

            if a < b and exact:
                exact_duplicates.append((a, b, prob_cols[a], prob_cols[b]))
            if a < b and np.isfinite(spear) and abs(spear) >= high_corr_warn:
                high_corr_pairs.append((a, b, float(spear)))

    if exact_duplicates and not allow_identical:
        raise ValueError(
            "Exact duplicated probability columns detected: "
            f"{exact_duplicates}. This would make Step79A state rows misleading."
        )

    summary = {
        "prob_cols": prob_cols,
        "exact_duplicate_pairs": exact_duplicates,
        "high_corr_pairs_abs_spearman_ge_threshold": high_corr_pairs,
        "high_corr_threshold": high_corr_warn,
        "pearson": mat_pearson.round(4).to_dict(),
        "spearman": mat_spearman.round(4).to_dict(),
        "per_column_range": {
            k: {
                "min": float(np.nanmin(pd.to_numeric(df[v], errors="coerce"))),
                "max": float(np.nanmax(pd.to_numeric(df[v], errors="coerce"))),
                "mean": float(np.nanmean(pd.to_numeric(df[v], errors="coerce")))
            }
            for k, v in prob_cols.items()
        }
    }

    return pd.DataFrame(rows), summary


def plot_probability_audit(audit_df, outbase, dpi=450):
    labels = ["core", "peri", "remote"]
    pear = pd.DataFrame(index=labels, columns=labels, dtype=float)
    spear = pd.DataFrame(index=labels, columns=labels, dtype=float)

    for _, r in audit_df.iterrows():
        pear.loc[r["a"], r["b"]] = r["pearson"]
        spear.loc[r["a"], r["b"]] = r["spearman"]

    fig = plt.figure(figsize=(9.5, 4.2), facecolor=BG)
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.92, bottom=0.18, top=0.82, wspace=0.22)
    mats = [(pear, "Pearson correlation"), (spear, "Spearman correlation")]

    for i, (mat, title) in enumerate(mats):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(BG)
        im = ax.imshow(mat.values.astype(float), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, color=FG, fontsize=9, rotation=35, ha="right")
        ax.set_yticklabels(labels, color=FG, fontsize=9)
        ax.set_title(title, color=FG, fontsize=12, fontweight="bold")
        for r in range(len(labels)):
            for c in range(len(labels)):
                val = mat.values[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color="white", fontsize=10)
        for spine in ax.spines.values():
            spine.set_color("#3b4654")

    cax = fig.add_axes([0.94, 0.22, 0.015, 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors=FG, labelsize=8)
    cb.outline.set_edgecolor(FG)

    fig.suptitle("Step79 probability-column audit", color=FG, fontsize=17, fontweight="bold", y=0.96)
    return save_all(fig, outbase, dpi=dpi)


# -----------------------------
# Expression / module extraction
# -----------------------------

def var_lookup(adata):
    return {str(g).lower(): i for i, g in enumerate(adata.var_names.astype(str))}


def get_gene_values(adata, gene, lookup, layer=None):
    key = gene.lower()
    if key not in lookup:
        return None
    idx = lookup[key]
    X = adata.layers[layer] if layer else adata.X
    vals = as_dense_1d(X[:, idx])
    vals = np.asarray(vals, dtype=float)
    return vals


def compute_marker_values(adata, marker_groups, layer=None):
    lookup = var_lookup(adata)
    values = {}
    audit = {}
    for group, genes in marker_groups.items():
        audit[group] = {"requested": genes, "used": [], "missing": []}
        for g in genes:
            vals = get_gene_values(adata, g, lookup, layer=layer)
            if vals is None:
                audit[group]["missing"].append(g)
                values[g] = None
            else:
                audit[group]["used"].append(g)
                values[g] = robust_minmax(vals, 1, 99.5)
    return values, audit


def compute_module_scores(adata, module_sets, layer=None):
    lookup = var_lookup(adata)
    module_scores = {}
    audit = {}
    for module, genes in module_sets.items():
        used = []
        missing = []
        mats = []
        for g in genes:
            vals = get_gene_values(adata, g, lookup, layer=layer)
            if vals is None:
                missing.append(g)
                continue
            used.append(g)
            mats.append(robust_z(vals))
        if len(mats) == 0:
            module_scores[module] = np.full(adata.n_obs, np.nan)
        else:
            score = np.nanmean(np.vstack(mats), axis=0)
            score = robust_z(score)
            module_scores[module] = score
        audit[module] = {
            "requested": genes,
            "used": used,
            "missing": missing,
            "n_used": len(used)
        }
    return module_scores, audit


# -----------------------------
# Spatial smoothing
# -----------------------------

def extent_from_xy(x, y, pad_frac=0.04):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    xmin, xmax = np.nanmin(x[finite]), np.nanmax(x[finite])
    ymin, ymax = np.nanmin(y[finite]), np.nanmax(y[finite])
    xr = xmax - xmin
    yr = ymax - ymin
    if xr <= 0:
        xr = 1
    if yr <= 0:
        yr = 1
    return (
        xmin - xr * pad_frac,
        xmax + xr * pad_frac,
        ymin - yr * pad_frac,
        ymax + yr * pad_frac
    )


def make_grid(extent, grid_n=180):
    xmin, xmax, ymin, ymax = extent
    nx = grid_n
    ny = max(40, int(grid_n * (ymax - ymin) / max(xmax - xmin, 1e-9)))
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    return xedges, yedges, Xg, Yg


def smooth_on_grid(x, y, v, xedges, yedges, sigma=1.4, support_frac=0.025):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)

    nx = len(xedges) - 1
    ny = len(yedges) - 1

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x = x[finite]
    y = y[finite]
    v = v[finite]

    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[keep]
    iy = iy[keep]
    v = v[keep]

    sum_grid = np.zeros((ny, nx), dtype=float)
    cnt_grid = np.zeros((ny, nx), dtype=float)

    np.add.at(sum_grid, (iy, ix), v)
    np.add.at(cnt_grid, (iy, ix), 1.0)

    sum_s = gaussian_filter(sum_grid, sigma=sigma, mode="constant", cval=0.0)
    cnt_s = gaussian_filter(cnt_grid, sigma=sigma, mode="constant", cval=0.0)

    z = np.full_like(sum_s, np.nan, dtype=float)
    valid = cnt_s > 1e-8
    z[valid] = sum_s[valid] / cnt_s[valid]

    if np.nanmax(cnt_s) > 0:
        support = cnt_s >= (np.nanmax(cnt_s) * support_frac)
    else:
        support = np.zeros_like(cnt_s, dtype=bool)

    z[~support] = np.nan
    return z, support, cnt_s


def draw_spatial_panel(
    ax,
    x, y, values,
    xedges, yedges, Xg, Yg,
    z, support,
    cmap,
    vmin,
    vmax,
    title=None,
    raw_alpha=0.20,
    raw_size=1.6,
    contour_pack=None,
    hide_ticks=True,
    diverging=False
):
    ax.set_facecolor(BG)

    if diverging:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax.imshow(
            np.ma.masked_invalid(z),
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            norm=norm,
            interpolation="bilinear",
            aspect="equal"
        )
    else:
        im = ax.imshow(
            np.ma.masked_invalid(z),
            origin="lower",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal"
        )

    # raw spot overlay
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if finite.sum() > 0 and raw_alpha > 0:
        if diverging:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            ax.scatter(
                x[finite], y[finite],
                c=values[finite],
                s=raw_size,
                cmap=cmap,
                norm=norm,
                alpha=raw_alpha,
                linewidths=0
            )
        else:
            ax.scatter(
                x[finite], y[finite],
                c=values[finite],
                s=raw_size,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                alpha=raw_alpha,
                linewidths=0
            )

    # tissue outline
    try:
        ax.contour(
            Xg, Yg, support.astype(float),
            levels=[0.5],
            colors=TISSUE_CONTOUR_COLOR,
            linewidths=0.45,
            linestyles="dashed",
            alpha=0.55
        )
    except Exception:
        pass

    # core / peri contours
    if contour_pack:
        for item in contour_pack:
            zz = item["z"]
            level = item["level"]
            color = item["color"]
            lw = item.get("lw", 0.55)
            alpha = item.get("alpha", 0.85)
            try:
                if np.isfinite(zz).sum() > 10 and np.nanmax(zz) > level:
                    ax.contour(
                        Xg, Yg, zz,
                        levels=[level],
                        colors=color,
                        linewidths=lw,
                        alpha=alpha
                    )
            except Exception:
                pass

    if title:
        ax.set_title(title, color=FG, fontsize=9.5, fontweight="bold", pad=5)

    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.tick_params(colors=MUTED, labelsize=7)
        ax.set_xlabel("Spatial x", color=MUTED, fontsize=7)
        ax.set_ylabel("Spatial y", color=MUTED, fontsize=7)

    for spine in ax.spines.values():
        spine.set_color("#2e3744")
        spine.set_linewidth(0.6)

    return im


def contour_pack_for_timepoint(tp_cache, tp):
    c = tp_cache[tp]
    pack = []
    if c.get("core_z") is not None:
        zz = c["core_z"]
        finite = np.isfinite(zz)
        if finite.sum() > 0:
            level = max(0.35, float(np.nanquantile(zz[finite], 0.78)))
            pack.append({"z": zz, "level": level, "color": CORE_CONTOUR_COLOR, "lw": 0.55, "alpha": 0.85})
    if c.get("peri_z") is not None:
        zz = c["peri_z"]
        finite = np.isfinite(zz)
        if finite.sum() > 0:
            level = max(0.35, float(np.nanquantile(zz[finite], 0.78)))
            pack.append({"z": zz, "level": level, "color": PERI_CONTOUR_COLOR, "lw": 0.45, "alpha": 0.75})
    return pack


# -----------------------------
# Figure A: state/module atlas
# -----------------------------

def make_state_module_atlas(df, x, y, timepoints, prob_cols, repair_values, module_scores, args, outdir):
    rows = [
        ("Core probability", "prob", pd.to_numeric(df[prob_cols["core"]], errors="coerce").astype(float).values),
        ("Peri-infarct probability", "prob", pd.to_numeric(df[prob_cols["peri"]], errors="coerce").astype(float).values),
        ("Remote-like probability", "prob", pd.to_numeric(df[prob_cols["remote"]], errors="coerce").astype(float).values),
        ("Repair score", "prob", repair_values),
        ("Repair / ECM", "module", module_scores["Repair / ECM"]),
        ("Ferroptosis", "module", module_scores["Ferroptosis"]),
        ("Inflammation", "module", module_scores["Inflammation"]),
        ("BBB / endothelial", "module", module_scores["BBB / endothelial"]),
        ("Reactive astrocyte", "module", module_scores["Reactive astrocyte"]),
        ("Hypoxia / redox", "module", module_scores["Hypoxia / redox"]),
    ]

    time_col = args.time_col
    nrows = len(rows)
    ncols = len(timepoints)

    # prepare per-timepoint grids and region contours
    tp_cache = {}
    for tp in timepoints:
        mask = df[time_col].astype(str).values == str(tp)
        xt = x[mask]
        yt = y[mask]
        extent = extent_from_xy(xt, yt, args.pad_frac)
        xedges, yedges, Xg, Yg = make_grid(extent, args.grid_n)
        core_z, support, _ = smooth_on_grid(
            xt, yt,
            pd.to_numeric(df.loc[mask, prob_cols["core"]], errors="coerce").astype(float).values,
            xedges, yedges,
            sigma=args.sigma,
            support_frac=args.support_frac
        )
        peri_z, _, _ = smooth_on_grid(
            xt, yt,
            pd.to_numeric(df.loc[mask, prob_cols["peri"]], errors="coerce").astype(float).values,
            xedges, yedges,
            sigma=args.sigma,
            support_frac=args.support_frac
        )
        tp_cache[tp] = {
            "mask": mask,
            "xedges": xedges,
            "yedges": yedges,
            "Xg": Xg,
            "Yg": Yg,
            "support": support,
            "core_z": core_z,
            "peri_z": peri_z
        }

    module_all = np.concatenate([np.asarray(v, dtype=float) for _, typ, v in rows if typ == "module"])
    finite_module = module_all[np.isfinite(module_all)]
    module_vlim = float(np.nanpercentile(np.abs(finite_module), args.module_vlim_quantile)) if finite_module.size else 2.5
    module_vlim = max(module_vlim, 1.0)

    fig_h = max(18, 2.0 * nrows)
    fig = plt.figure(figsize=(12.6, fig_h), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1.15] + [1] * ncols,
        left=0.06,
        right=0.86,
        bottom=0.055,
        top=0.92,
        hspace=0.10,
        wspace=0.055
    )

    for r, (row_label, typ, vals) in enumerate(rows):
        lab_ax = fig.add_subplot(gs[r, 0])
        lab_ax.set_facecolor(BG)
        lab_ax.axis("off")
        lab_ax.text(
            0.97, 0.50, row_label,
            ha="right", va="center",
            color=FG,
            fontsize=8.7,
            fontweight="bold"
        )

        for c, tp in enumerate(timepoints):
            ax = fig.add_subplot(gs[r, c + 1])
            cache = tp_cache[tp]
            mask = cache["mask"]
            xt = x[mask]
            yt = y[mask]
            vt = np.asarray(vals, dtype=float)[mask]

            z, support, _ = smooth_on_grid(
                xt, yt, vt,
                cache["xedges"], cache["yedges"],
                sigma=args.sigma,
                support_frac=args.support_frac
            )

            if typ == "prob":
                cmap = CMAP_PROB
                vmin, vmax = 0, 1
                diverging = False
                raw_alpha = 0.14
                raw_size = 1.15
            else:
                cmap = CMAP_MODULE
                vmin, vmax = -module_vlim, module_vlim
                diverging = True
                raw_alpha = 0.10
                raw_size = 1.0

            title = str(tp) if r == 0 else None

            draw_spatial_panel(
                ax,
                xt, yt, vt,
                cache["xedges"], cache["yedges"],
                cache["Xg"], cache["Yg"],
                z, support,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                title=title,
                raw_alpha=raw_alpha,
                raw_size=raw_size,
                contour_pack=contour_pack_for_timepoint(tp_cache, tp),
                hide_ticks=True,
                diverging=diverging
            )

    # colorbars
    cax1 = fig.add_axes([0.885, 0.61, 0.014, 0.22])
    cb1 = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_PROB),
        cax=cax1
    )
    cb1.set_label("State probability / repair score", color=FG, fontsize=8)
    cb1.ax.tick_params(colors=FG, labelsize=7)
    cb1.outline.set_edgecolor(FG)

    cax2 = fig.add_axes([0.885, 0.16, 0.014, 0.30])
    cb2 = fig.colorbar(
        ScalarMappable(norm=TwoSlopeNorm(vmin=-module_vlim, vcenter=0, vmax=module_vlim), cmap=CMAP_MODULE),
        cax=cax2
    )
    cb2.set_label("Module score (z)", color=FG, fontsize=8)
    cb2.ax.tick_params(colors=FG, labelsize=7)
    cb2.outline.set_edgecolor(FG)

    fig.suptitle(
        "State/timepoint-specific spatial progression landscape",
        color=FG,
        fontsize=18,
        fontweight="bold",
        y=0.975
    )

    fig.text(
        0.50, 0.025,
        "State/timepoint-specific spatial progression atlas. Smoothed maps use Gaussian spot aggregation; "
        "dashed contours indicate tissue support; colored contours indicate high core/peri regions when available. "
        "Module scores are curated gene-set scores, not functional validation.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.6
    )

    outbase = os.path.join(outdir, "Fig_Step79A_StateModule_TimepointProgressionAtlas_FIXED")
    paths = save_all(fig, outbase, dpi=args.dpi)
    plt.close(fig)
    return paths


# -----------------------------
# Figure B: marker atlas
# -----------------------------

def make_marker_atlas(df, x, y, timepoints, marker_groups, marker_values, prob_cols, args, outdir):
    marker_rows = []
    for group, genes in marker_groups.items():
        for i, g in enumerate(genes):
            if marker_values.get(g) is not None:
                marker_rows.append((group if i == 0 else "", g, marker_values[g]))

    nrows = len(marker_rows)
    ncols = len(timepoints)
    time_col = args.time_col

    tp_cache = {}
    for tp in timepoints:
        mask = df[time_col].astype(str).values == str(tp)
        xt = x[mask]
        yt = y[mask]
        extent = extent_from_xy(xt, yt, args.pad_frac)
        xedges, yedges, Xg, Yg = make_grid(extent, args.grid_n)
        core_z, support, _ = smooth_on_grid(
            xt, yt,
            pd.to_numeric(df.loc[mask, prob_cols["core"]], errors="coerce").astype(float).values,
            xedges, yedges,
            sigma=args.sigma,
            support_frac=args.support_frac
        )
        peri_z, _, _ = smooth_on_grid(
            xt, yt,
            pd.to_numeric(df.loc[mask, prob_cols["peri"]], errors="coerce").astype(float).values,
            xedges, yedges,
            sigma=args.sigma,
            support_frac=args.support_frac
        )
        tp_cache[tp] = {
            "mask": mask,
            "xedges": xedges,
            "yedges": yedges,
            "Xg": Xg,
            "Yg": Yg,
            "support": support,
            "core_z": core_z,
            "peri_z": peri_z
        }

    fig_h = max(22, 1.15 * nrows)
    fig = plt.figure(figsize=(11.8, fig_h), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 2,
        width_ratios=[1.45, 0.70] + [1] * ncols,
        left=0.055,
        right=0.86,
        bottom=0.055,
        top=0.925,
        hspace=0.075,
        wspace=0.045
    )

    for r, (group_label, gene, vals) in enumerate(marker_rows):
        group_ax = fig.add_subplot(gs[r, 0])
        group_ax.set_facecolor(BG)
        group_ax.axis("off")
        if group_label:
            group_ax.text(
                0.98, 0.5,
                group_label,
                ha="right",
                va="center",
                color=FG,
                fontsize=7.7,
                fontweight="bold"
            )

        gene_ax = fig.add_subplot(gs[r, 1])
        gene_ax.set_facecolor(BG)
        gene_ax.axis("off")
        gene_ax.text(
            0.96, 0.5,
            gene,
            ha="right",
            va="center",
            color=FG,
            fontsize=7.6,
            fontweight="bold"
        )

        for c, tp in enumerate(timepoints):
            ax = fig.add_subplot(gs[r, c + 2])
            cache = tp_cache[tp]
            mask = cache["mask"]
            xt = x[mask]
            yt = y[mask]
            vt = vals[mask]

            z, support, _ = smooth_on_grid(
                xt, yt, vt,
                cache["xedges"], cache["yedges"],
                sigma=args.sigma,
                support_frac=args.support_frac
            )

            title = str(tp) if r == 0 else None

            draw_spatial_panel(
                ax,
                xt, yt, vt,
                cache["xedges"], cache["yedges"],
                cache["Xg"], cache["Yg"],
                z, support,
                cmap=CMAP_EXPR,
                vmin=0,
                vmax=1,
                title=title,
                raw_alpha=0.22,
                raw_size=1.25,
                contour_pack=contour_pack_for_timepoint(tp_cache, tp),
                hide_ticks=True,
                diverging=False
            )

    cax = fig.add_axes([0.885, 0.25, 0.014, 0.50])
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_EXPR),
        cax=cax
    )
    cb.set_label("Expression, robust-scaled per gene", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=7)
    cb.outline.set_edgecolor(FG)

    fig.suptitle(
        "Spatial progression atlas of representative marker genes",
        color=FG,
        fontsize=18,
        fontweight="bold",
        y=0.975
    )

    fig.text(
        0.50, 0.025,
        "Rows show representative genes grouped by biological program; columns show disease timepoints. "
        "Expression is log-normalized as available and robust-scaled per gene for visualization.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=7.6
    )

    outbase = os.path.join(outdir, "Fig_Step79B_Marker_TimepointProgressionAtlas_FIXED")
    paths = save_all(fig, outbase, dpi=args.dpi)
    plt.close(fig)
    return paths


# -----------------------------
# Main
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--h5ad", required=True)
    p.add_argument("--state_table", required=True)
    p.add_argument("--outdir", required=True)

    p.add_argument("--state_id_col", default="auto")
    p.add_argument("--time_col", default="timepoint")
    p.add_argument("--timepoints", default="D1,D3,D7")

    p.add_argument("--x_col", default="auto")
    p.add_argument("--y_col", default="auto")

    p.add_argument("--core_prob_col", default="auto")
    p.add_argument("--peri_prob_col", default="auto")
    p.add_argument("--remote_prob_col", default="auto")
    p.add_argument("--repair_score_col", default="auto")

    p.add_argument("--layer", default=None)

    p.add_argument("--grid_n", type=int, default=180)
    p.add_argument("--sigma", type=float, default=1.35)
    p.add_argument("--support_frac", type=float, default=0.025)
    p.add_argument("--pad_frac", type=float, default=0.045)
    p.add_argument("--module_vlim_quantile", type=float, default=98.5)

    p.add_argument("--dpi", type=int, default=450)
    p.add_argument("--allow_identical_probability_cols", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    mkdirp(args.outdir)

    print("=" * 100)
    print("Step79 fixed | State/timepoint-specific spatial progression atlas")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={args.outdir}")

    adata = read_h5ad(args.h5ad)
    x, y, xy_audit = infer_xy(adata, args.x_col, args.y_col)

    df, merge_audit = merge_state_table(adata, args.state_table, args.state_id_col)

    if args.time_col not in df.columns:
        raise ValueError(
            f"time_col={args.time_col} not found after merge. "
            f"Available columns include: {list(df.columns)[:60]}"
        )

    prob_cols = infer_prob_cols(df, args)
    prob_audit_df, prob_audit = probability_audit(
        df,
        prob_cols,
        allow_identical=args.allow_identical_probability_cols
    )

    prob_audit_csv = os.path.join(args.outdir, "step79_probability_column_audit.csv")
    prob_audit_df.to_csv(prob_audit_csv, index=False)

    prob_audit_paths = plot_probability_audit(
        prob_audit_df,
        os.path.join(args.outdir, "Fig_Step79C_ProbabilityColumnAudit_FIXED"),
        dpi=args.dpi
    )

    requested_tps = [t.strip() for t in args.timepoints.split(",") if t.strip()]
    present_tps = set(df[args.time_col].dropna().astype(str))
    timepoints = [t for t in requested_tps if t in present_tps]

    if not timepoints:
        timepoints = sorted(list(present_tps), key=natural_time_key)

    timepoint_counts = {
        str(tp): int((df[args.time_col].astype(str) == str(tp)).sum())
        for tp in timepoints
    }

    marker_groups = {
        "Microglia / inflammatory": ["C1qa", "C1qb", "Tyrobp", "Lgals3"],
        "Endothelial / BBB": ["Kdr", "Cldn5", "Pecam1", "Kcnj8"],
        "Ferroptosis / hypoxia": ["Hmox1", "Fth1", "Slc7a11", "Gpx4"],
        "Repair / ECM": ["Spp1", "Fn1", "Col1a1", "Col3a1"],
    }

    module_sets = {
        "Repair / ECM": ["Col1a1", "Col3a1", "Fn1", "Spp1", "Vim", "Apoe", "Postn", "Timp1"],
        "Ferroptosis": ["Hmox1", "Fth1", "Slc7a11", "Gpx4", "Tfrc", "Acsl4", "Ptgs2"],
        "Inflammation": ["C1qa", "C1qb", "Tyrobp", "Lgals3", "Ctsd", "Ccl2", "Ccl3"],
        "BBB / endothelial": ["Kdr", "Cldn5", "Pecam1", "Kcnj8", "Vwf", "Flt1", "Rgs5"],
        "Reactive astrocyte": ["Gfap", "Aqp4", "Vim", "Clu", "Apoe", "Serpina3n", "Lcn2"],
        "Hypoxia / redox": ["Hmox1", "Fth1", "Nfe2l1", "Nfe2l2", "Hif1a", "Vegfa", "Sod2"],
    }

    marker_values, marker_audit = compute_marker_values(adata, marker_groups, layer=args.layer)
    module_scores, module_audit = compute_module_scores(adata, module_sets, layer=args.layer)

    # repair score source
    repair_source = "module_score_repair_ECM"
    repair_values = robust_minmax(module_scores["Repair / ECM"], 1, 99)

    if args.repair_score_col != "auto":
        if args.repair_score_col not in df.columns:
            raise ValueError(f"Forced repair score column not found: {args.repair_score_col}")
        repair_values = robust_minmax(pd.to_numeric(df[args.repair_score_col], errors="coerce").astype(float).values, 1, 99)
        repair_source = args.repair_score_col
    else:
        repair_candidates = [
            "repair_score", "rescue_score", "repair_probability", "repair_prob",
            "state_shift_priority_score", "response_priority"
        ]
        for c in repair_candidates:
            if c in df.columns:
                repair_values = robust_minmax(pd.to_numeric(df[c], errors="coerce").astype(float).values, 1, 99)
                repair_source = c
                break

    paths_A = make_state_module_atlas(
        df=df,
        x=x,
        y=y,
        timepoints=timepoints,
        prob_cols=prob_cols,
        repair_values=repair_values,
        module_scores=module_scores,
        args=args,
        outdir=args.outdir
    )

    paths_B = make_marker_atlas(
        df=df,
        x=x,
        y=y,
        timepoints=timepoints,
        marker_groups=marker_groups,
        marker_values=marker_values,
        prob_cols=prob_cols,
        args=args,
        outdir=args.outdir
    )

    report = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "xy_audit": xy_audit,
        "state_merge_audit": merge_audit,
        "time_col": args.time_col,
        "timepoints_used": timepoints,
        "timepoint_counts": timepoint_counts,
        "prob_cols": prob_cols,
        "probability_column_audit": prob_audit,
        "repair_score_source": repair_source,
        "marker_audit": marker_audit,
        "module_audit": module_audit,
        "outputs": {
            "state_module_atlas": paths_A,
            "marker_atlas": paths_B,
            "probability_audit": prob_audit_paths,
            "probability_audit_csv": prob_audit_csv
        },
        "interpretation_note": (
            "Step79 visualizes timepoint-specific spatial progression patterns. "
            "State probability rows use explicitly audited probability columns. "
            "Marker rows are robust-scaled per gene for visualization. "
            "Module rows are curated gene-set scores and should not be interpreted as full WGCNA or functional validation."
        )
    }

    report_path = os.path.join(args.outdir, "step79_fixed_audit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nFinished Step79 fixed.")
    print(f"Audit report: {report_path}")
    print("Outputs:")
    for k, v in report["outputs"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
