#!/usr/bin/env python3
# WHITE-BACKGROUND VARIANT GENERATED 2026-09-01.
# Source: E:\vir\ST\79_state_timepoint_spatial_progression_atlas_final_v2.py
# Scientific calculations and data mappings are unchanged; only visual theme literals were remapped.
# -*- coding: utf-8 -*-

"""
Step79 final v2
State/timepoint-specific spatial progression atlas

Main correction:
- If core/peri/remote probability columns are duplicated, do NOT use them.
- Use label-derived one-hot state maps from state_group / region labels, then spatially smooth.
- This avoids misleading identical state-probability rows.

Outputs:
1. Fig_Step79A_StateModule_TimepointProgressionAtlas_FINAL_v2.png/pdf/svg
2. Fig_Step79B_Marker_TimepointProgressionAtlas_FINAL_v2.png/pdf/svg
3. Fig_Step79C_ProbabilityColumnAudit_FINAL_v2.png/pdf/svg
4. step79_final_v2_audit_report.json
5. step79_probability_column_audit_final_v2.csv
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
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

try:
    import anndata as ad
except Exception as e:
    raise ImportError("Please install anndata: pip install anndata") from e

try:
    from scipy import sparse
    from scipy.ndimage import gaussian_filter
except Exception as e:
    raise ImportError("Please install scipy") from e


BG = "#ffffff"
FG = "#111827"
MUTED = "#4b5563"
TISSUE_CONTOUR_COLOR = "#6b7280"
CORE_CONTOUR_COLOR = "#ff9f40"
PERI_CONTOUR_COLOR = "#f4d35e"

CMAP_EXPR = "magma"
CMAP_PROB = "magma"
CMAP_MODULE = "RdBu_r"


def mkdirp(path):
    Path(path).mkdir(parents=True, exist_ok=True)


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
        lo = np.nanmin(v[finite])
        hi = np.nanmax(v[finite])
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
    if mad > 1e-12:
        out[finite] = (v[finite] - med) / (1.4826 * mad)
    else:
        sd = np.nanstd(v[finite])
        if sd <= 1e-12:
            return out
        out[finite] = (v[finite] - np.nanmean(v[finite])) / sd
    out[~finite] = np.nan
    return out


def save_all(fig, outbase, dpi=450):
    paths = {}
    for ext in ["png", "pdf", "svg"]:
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

    pairs = [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
        ("array_col", "array_row"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("imagecol", "imagerow"),
    ]
    for xc, yc in pairs:
        if xc in obs.columns and yc in obs.columns:
            return np.asarray(obs[xc], dtype=float), np.asarray(obs[yc], dtype=float), {
                "mode": "obs_auto",
                "xcol": xc,
                "ycol": yc
            }

    if "spatial" in adata.obsm.keys():
        xy = np.asarray(adata.obsm["spatial"])
        return xy[:, 0].astype(float), xy[:, 1].astype(float), {
            "mode": "obsm_spatial",
            "obsm_key": "spatial"
        }

    raise ValueError("Cannot infer spatial coordinates. Use --x_col and --y_col.")


def merge_state_table(adata, state_table_path, state_id_col="auto"):
    base = pd.DataFrame({"_adata_id_": adata.obs_names.astype(str)})
    obs = adata.obs.copy()
    obs["_adata_id_"] = adata.obs_names.astype(str)

    if not state_table_path:
        df = base.merge(obs, on="_adata_id_", how="left")
        return df, {
            "state_table_available": False,
            "merge_mode": "obs_only"
        }

    st = pd.read_csv(state_table_path)
    if state_id_col != "auto":
        if state_id_col not in st.columns:
            raise ValueError(f"--state_id_col={state_id_col} not in state table.")
        key = state_id_col
    else:
        key = None
        for c in ["obs_name", "barcode", "cell", "cell_id", "spot", "spot_id", "_adata_id_", "Unnamed: 0"]:
            if c in st.columns:
                overlap = len(set(base["_adata_id_"]).intersection(set(st[c].astype(str))))
                if overlap > 0:
                    key = c
                    break
        if key is None:
            raise ValueError("Cannot infer state table ID column. Use --state_id_col obs_name.")

    st[key] = st[key].astype(str)
    df = base.merge(st, left_on="_adata_id_", right_on=key, how="left", suffixes=("", "_state"))

    for c in obs.columns:
        if c not in df.columns:
            df[c] = obs[c].values

    overlap = int(df[key].notna().sum()) if key in df.columns else 0

    audit = {
        "state_table_available": True,
        "state_table_rows": int(st.shape[0]),
        "merge_mode": "id",
        "merge_key_meta": "_adata_id_",
        "merge_key_state": key,
        "state_merge_overlap": overlap,
        "n_obs": int(adata.n_obs)
    }

    if overlap < adata.n_obs * 0.8:
        warnings.warn(f"Low state table overlap: {overlap}/{adata.n_obs}")

    return df, audit


def pick_col(df, forced, candidates, label):
    if forced and forced != "auto":
        if forced not in df.columns:
            raise ValueError(f"Forced {label} column not found: {forced}")
        return forced

    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]

    for c in df.columns:
        cl = c.lower()
        if label == "core" and "core" in cl and "prob" in cl:
            return c
        if label == "peri" and ("peri" in cl or "penumbra" in cl) and "prob" in cl:
            return c
        if label == "remote" and "remote" in cl and "prob" in cl:
            return c

    return None


def infer_prob_cols(df, args):
    core = pick_col(df, args.core_prob_col, [
        "core_probability", "core_prob", "prob_core", "p_core",
        "lesion_core_probability", "lesion_core_like_probability",
        "lesion-core-like_probability"
    ], "core")
    peri = pick_col(df, args.peri_prob_col, [
        "peri_probability", "peri_prob", "prob_peri", "p_peri",
        "peri_infarct_probability", "peri-infarct_probability"
    ], "peri")
    remote = pick_col(df, args.remote_prob_col, [
        "remote_probability", "remote_prob", "prob_remote", "p_remote",
        "remote_like_probability", "remote-like_probability"
    ], "remote")

    if core is None or peri is None or remote is None:
        return None

    return {"core": core, "peri": peri, "remote": remote}


def probability_audit(df, prob_cols):
    labels = ["core", "peri", "remote"]
    rows = []
    exact_duplicates = []

    for a in labels:
        va = pd.to_numeric(df[prob_cols[a]], errors="coerce").astype(float).values
        for b in labels:
            vb = pd.to_numeric(df[prob_cols[b]], errors="coerce").astype(float).values
            valid = np.isfinite(va) & np.isfinite(vb)

            if valid.sum() < 5:
                pear = np.nan
                spear = np.nan
                exact = False
                max_abs_diff = np.nan
            else:
                pear = float(np.corrcoef(va[valid], vb[valid])[0, 1])
                spear = float(pd.Series(va[valid]).corr(pd.Series(vb[valid]), method="spearman"))
                max_abs_diff = float(np.nanmax(np.abs(va[valid] - vb[valid])))
                exact = bool(np.allclose(va[valid], vb[valid], atol=1e-12, equal_nan=True))

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

    return pd.DataFrame(rows), exact_duplicates


def plot_probability_audit(audit_df, outbase, dpi):
    labels = ["core", "peri", "remote"]
    mats = {}
    for metric in ["pearson", "spearman"]:
        mat = pd.DataFrame(index=labels, columns=labels, dtype=float)
        for _, r in audit_df.iterrows():
            mat.loc[r["a"], r["b"]] = r[metric]
        mats[metric] = mat

    fig = plt.figure(figsize=(8.8, 4.0), facecolor=BG)
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.90, bottom=0.20, top=0.78, wspace=0.28)

    im = None
    for i, metric in enumerate(["pearson", "spearman"]):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(BG)
        mat = mats[metric].values.astype(float)
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)

        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(labels, color=FG, rotation=35, ha="right", fontsize=9)
        ax.set_yticklabels(labels, color=FG, fontsize=9)
        ax.set_title(metric.capitalize(), color=FG, fontsize=12, fontweight="bold")

        for r in range(3):
            for c in range(3):
                ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center", color="#1f2937", fontsize=10)

        for sp in ax.spines.values():
            sp.set_color("#9ca3af")

    cax = fig.add_axes([0.92, 0.22, 0.015, 0.52])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(colors=FG, labelsize=8)
    cb.outline.set_edgecolor(FG)

    fig.suptitle("Step79 probability-column audit", color=FG, fontsize=16, fontweight="bold")
    return save_all(fig, outbase, dpi)


def infer_label_col(df, forced="auto"):
    if forced != "auto":
        if forced not in df.columns:
            raise ValueError(f"--label_col={forced} not found.")
        return forced

    for c in ["state_group", "region_refined", "region_manual_final", "region", "state", "label"]:
        if c in df.columns:
            return c

    raise ValueError("Cannot infer label_col. Please provide --label_col state_group.")


def normalize_label(s):
    s = str(s).strip().lower()
    s = s.replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    return s



def make_label_state_arrays(df, label_col):
    """
    Build label-derived one-hot state maps for Step79A.

    Important:
    - Uses boolean arrays internally, then converts to float.
    - Avoids duplicated probability columns.
    - State rows become:
        core   = I(label contains core / lesion-core)
        peri   = I(label contains peri / penumbra)
        remote = I(label contains remote)
    """

    def normalize_label(s):
        s = str(s).strip().lower()
        s = s.replace("_", "-")
        s = re.sub(r"\s+", "-", s)
        return s

    lab = (
        df[label_col]
        .fillna("")
        .astype(str)
        .map(normalize_label)
        .values
        .astype(str)
    )

    core_keys = [
        "lesion-core-like",
        "lesion-core",
        "infarct-core",
        "core-like",
        "core"
    ]

    peri_keys = [
        "peri-infarct",
        "peri-infarct-like",
        "peri-lesion",
        "peri-core",
        "penumbra",
        "peri"
    ]

    remote_keys = [
        "remote-like",
        "remote-normal",
        "normal-like",
        "remote"
    ]

    def match(keys):
        out = np.zeros(len(lab), dtype=bool)
        for k in keys:
            kk = normalize_label(k)
            out = out | np.array([kk in x for x in lab], dtype=bool)
        return out

    core_bool = match(core_keys)
    peri_bool = match(peri_keys)
    remote_bool = match(remote_keys)

    # Conservative fallback, useful when labels are slightly different
    if core_bool.sum() == 0:
        core_bool = np.array(["core" in x for x in lab], dtype=bool)

    if peri_bool.sum() == 0:
        peri_bool = np.array(
            [("peri" in x) or ("penumbra" in x) for x in lab],
            dtype=bool
        )

    if remote_bool.sum() == 0:
        remote_bool = np.array(["remote" in x for x in lab], dtype=bool)

    counts = {
        "core_positive": int(core_bool.sum()),
        "peri_positive": int(peri_bool.sum()),
        "remote_positive": int(remote_bool.sum())
    }

    if counts["core_positive"] == 0 or counts["peri_positive"] == 0 or counts["remote_positive"] == 0:
        raise ValueError(
            "Label-derived state maps contain an empty class. "
            f"label_col={label_col}; counts={counts}; "
            f"unique labels={sorted(pd.unique(df[label_col].astype(str)).tolist())}"
        )

    state_arrays = {
        "core": core_bool.astype(float),
        "peri": peri_bool.astype(float),
        "remote": remote_bool.astype(float)
    }

    label_audit = {
        "label_col": label_col,
        "unique_labels": sorted(pd.unique(df[label_col].astype(str)).tolist()),
        "counts": counts,
        "state_source": "label_onehot_then_spatial_smoothing"
    }

    return state_arrays, label_audit

def var_lookup(adata):
    return {str(g).lower(): i for i, g in enumerate(adata.var_names.astype(str))}


def get_gene_values(adata, gene, lookup, layer=None):
    key = gene.lower()
    if key not in lookup:
        return None
    idx = lookup[key]
    X = adata.layers[layer] if layer else adata.X
    return np.asarray(as_dense_1d(X[:, idx]), dtype=float)


def compute_marker_values(adata, marker_groups, layer=None):
    lookup = var_lookup(adata)
    values = {}
    audit = {}
    for group, genes in marker_groups.items():
        audit[group] = {"requested": genes, "used": [], "missing": []}
        for g in genes:
            vals = get_gene_values(adata, g, lookup, layer=layer)
            if vals is None:
                values[g] = None
                audit[group]["missing"].append(g)
            else:
                values[g] = robust_minmax(vals, 1, 99.5)
                audit[group]["used"].append(g)
    return values, audit


def compute_module_scores(adata, module_sets, layer=None):
    lookup = var_lookup(adata)
    scores = {}
    audit = {}
    for module, genes in module_sets.items():
        mats = []
        used = []
        missing = []
        for g in genes:
            vals = get_gene_values(adata, g, lookup, layer=layer)
            if vals is None:
                missing.append(g)
            else:
                mats.append(robust_z(vals))
                used.append(g)

        if len(mats) == 0:
            scores[module] = np.full(adata.n_obs, np.nan)
        else:
            scores[module] = robust_z(np.nanmean(np.vstack(mats), axis=0))

        audit[module] = {
            "requested": genes,
            "used": used,
            "missing": missing,
            "n_used": len(used)
        }
    return scores, audit


def extent_from_xy(x, y, pad_frac=0.04):
    finite = np.isfinite(x) & np.isfinite(y)
    xmin, xmax = np.nanmin(x[finite]), np.nanmax(x[finite])
    ymin, ymax = np.nanmin(y[finite]), np.nanmax(y[finite])
    xr = max(xmax - xmin, 1e-9)
    yr = max(ymax - ymin, 1e-9)
    return xmin - xr * pad_frac, xmax + xr * pad_frac, ymin - yr * pad_frac, ymax + yr * pad_frac


def make_grid(extent, grid_n):
    xmin, xmax, ymin, ymax = extent
    nx = grid_n
    ny = max(40, int(grid_n * (ymax - ymin) / max(xmax - xmin, 1e-9)))
    xedges = np.linspace(xmin, xmax, nx + 1)
    yedges = np.linspace(ymin, ymax, ny + 1)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xg, Yg = np.meshgrid(xc, yc)
    return xedges, yedges, Xg, Yg


def smooth_on_grid(x, y, v, xedges, yedges, sigma=1.35, support_frac=0.025):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    v = np.asarray(v, dtype=float)

    nx = len(xedges) - 1
    ny = len(yedges) - 1

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    x, y, v = x[finite], y[finite], v[finite]

    ix = np.searchsorted(xedges, x, side="right") - 1
    iy = np.searchsorted(yedges, y, side="right") - 1
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix, iy, v = ix[keep], iy[keep], v[keep]

    sums = np.zeros((ny, nx), dtype=float)
    counts = np.zeros((ny, nx), dtype=float)
    np.add.at(sums, (iy, ix), v)
    np.add.at(counts, (iy, ix), 1.0)

    sums_s = gaussian_filter(sums, sigma=sigma, mode="constant", cval=0.0)
    counts_s = gaussian_filter(counts, sigma=sigma, mode="constant", cval=0.0)

    z = np.full_like(sums_s, np.nan, dtype=float)
    valid = counts_s > 1e-8
    z[valid] = sums_s[valid] / counts_s[valid]

    support = counts_s >= np.nanmax(counts_s) * support_frac if np.nanmax(counts_s) > 0 else np.zeros_like(counts_s, dtype=bool)
    z[~support] = np.nan
    return z, support


def draw_panel(
    ax,
    x, y, values,
    xedges, yedges, Xg, Yg,
    z, support,
    cmap,
    vmin,
    vmax,
    title=None,
    diverging=False,
    raw_alpha=0.16,
    raw_size=1.0,
    contour_pack=None
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

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if finite.sum() > 0 and raw_alpha > 0:
        if diverging:
            ax.scatter(
                x[finite], y[finite], c=values[finite],
                s=raw_size, cmap=cmap,
                norm=TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax),
                alpha=raw_alpha, linewidths=0
            )
        else:
            ax.scatter(
                x[finite], y[finite], c=values[finite],
                s=raw_size, cmap=cmap,
                vmin=vmin, vmax=vmax,
                alpha=raw_alpha, linewidths=0
            )

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

    if contour_pack:
        for item in contour_pack:
            zz = item["z"]
            if zz is None or np.isfinite(zz).sum() < 10:
                continue
            level = item["level"]
            if np.nanmax(zz) <= level:
                continue
            try:
                ax.contour(
                    Xg, Yg, zz,
                    levels=[level],
                    colors=item["color"],
                    linewidths=item.get("lw", 0.55),
                    alpha=item.get("alpha", 0.85)
                )
            except Exception:
                pass

    if title:
        ax.set_title(title, color=FG, fontsize=9.5, fontweight="bold", pad=5)

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#d1d5db")
        sp.set_linewidth(0.6)

    return im


def make_timepoint_cache(df, x, y, timepoints, state_arrays, args):
    cache = {}
    for tp in timepoints:
        mask = df[args.time_col].astype(str).values == str(tp)
        xt, yt = x[mask], y[mask]
        extent = extent_from_xy(xt, yt, args.pad_frac)
        xedges, yedges, Xg, Yg = make_grid(extent, args.grid_n)

        core_z, support = smooth_on_grid(
            xt, yt, state_arrays["core"][mask],
            xedges, yedges,
            sigma=args.sigma,
            support_frac=args.support_frac
        )
        peri_z, _ = smooth_on_grid(
            xt, yt, state_arrays["peri"][mask],
            xedges, yedges,
            sigma=args.sigma,
            support_frac=args.support_frac
        )

        cache[tp] = {
            "mask": mask,
            "xedges": xedges,
            "yedges": yedges,
            "Xg": Xg,
            "Yg": Yg,
            "support": support,
            "core_z": core_z,
            "peri_z": peri_z
        }
    return cache


def contour_pack(cache, tp):
    c = cache[tp]
    pack = []
    for key, color in [("core_z", CORE_CONTOUR_COLOR), ("peri_z", PERI_CONTOUR_COLOR)]:
        zz = c.get(key)
        if zz is None or np.isfinite(zz).sum() < 10:
            continue
        finite = np.isfinite(zz)
        level = max(0.25, float(np.nanquantile(zz[finite], 0.80)))
        pack.append({"z": zz, "level": level, "color": color, "lw": 0.55, "alpha": 0.85})
    return pack


def make_state_module_atlas(df, x, y, timepoints, state_arrays, repair_values, module_scores, args, outdir):
    rows = [
        ("Core probability", "prob", state_arrays["core"]),
        ("Peri-infarct probability", "prob", state_arrays["peri"]),
        ("Remote-like probability", "prob", state_arrays["remote"]),
        ("Repair score", "prob", repair_values),
        ("Repair / ECM", "module", module_scores["Repair / ECM"]),
        ("Ferroptosis", "module", module_scores["Ferroptosis"]),
        ("Inflammation", "module", module_scores["Inflammation"]),
        ("BBB / endothelial", "module", module_scores["BBB / endothelial"]),
        ("Reactive astrocyte", "module", module_scores["Reactive astrocyte"]),
        ("Hypoxia / redox", "module", module_scores["Hypoxia / redox"]),
    ]

    cache = make_timepoint_cache(df, x, y, timepoints, state_arrays, args)

    all_module = np.concatenate([np.asarray(v, dtype=float) for _, typ, v in rows if typ == "module"])
    finite = all_module[np.isfinite(all_module)]
    module_vlim = float(np.nanpercentile(np.abs(finite), args.module_vlim_quantile)) if finite.size else 2.5
    module_vlim = max(module_vlim, 1.0)

    nrows = len(rows)
    ncols = len(timepoints)

    fig = plt.figure(figsize=(12.8, max(18, 2.0 * nrows)), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[1.25] + [1] * ncols,
        left=0.055,
        right=0.86,
        bottom=0.055,
        top=0.92,
        hspace=0.10,
        wspace=0.055
    )

    for r, (row_label, typ, vals) in enumerate(rows):
        lab_ax = fig.add_subplot(gs[r, 0])
        lab_ax.axis("off")
        lab_ax.set_facecolor(BG)
        lab_ax.text(
            0.98, 0.5, row_label,
            ha="right", va="center",
            color=FG, fontsize=8.4, fontweight="bold"
        )

        for c, tp in enumerate(timepoints):
            ax = fig.add_subplot(gs[r, c + 1])
            cc = cache[tp]
            mask = cc["mask"]
            xt, yt, vt = x[mask], y[mask], np.asarray(vals, dtype=float)[mask]

            z, support = smooth_on_grid(
                xt, yt, vt,
                cc["xedges"], cc["yedges"],
                sigma=args.sigma,
                support_frac=args.support_frac
            )

            if typ == "prob":
                draw_panel(
                    ax, xt, yt, vt,
                    cc["xedges"], cc["yedges"], cc["Xg"], cc["Yg"],
                    z, support,
                    cmap=CMAP_PROB, vmin=0, vmax=1,
                    title=str(tp) if r == 0 else None,
                    diverging=False,
                    raw_alpha=0.15,
                    raw_size=1.1,
                    contour_pack=contour_pack(cache, tp)
                )
            else:
                draw_panel(
                    ax, xt, yt, vt,
                    cc["xedges"], cc["yedges"], cc["Xg"], cc["Yg"],
                    z, support,
                    cmap=CMAP_MODULE, vmin=-module_vlim, vmax=module_vlim,
                    title=None,
                    diverging=True,
                    raw_alpha=0.10,
                    raw_size=1.0,
                    contour_pack=contour_pack(cache, tp)
                )

    cax1 = fig.add_axes([0.885, 0.61, 0.014, 0.22])
    cb1 = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_PROB), cax=cax1)
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
        color=FG, fontsize=18, fontweight="bold", y=0.975
    )
    fig.text(
        0.5, 0.026,
        "State rows are label-derived one-hot maps smoothed within each timepoint; "
        "module rows are curated gene-set scores. Dashed contours indicate tissue support; "
        "colored contours indicate high core/peri regions when available.",
        ha="center", va="center", color=MUTED, fontsize=7.5
    )

    outbase = os.path.join(outdir, "Fig_Step79A_StateModule_TimepointProgressionAtlas_FINAL_v2")
    paths = save_all(fig, outbase, dpi=args.dpi)
    plt.close(fig)
    return paths


def make_marker_atlas(df, x, y, timepoints, marker_groups, marker_values, state_arrays, args, outdir):
    marker_rows = []
    for group, genes in marker_groups.items():
        for i, g in enumerate(genes):
            if marker_values.get(g) is not None:
                marker_rows.append((group if i == 0 else "", g, marker_values[g]))

    cache = make_timepoint_cache(df, x, y, timepoints, state_arrays, args)

    nrows = len(marker_rows)
    ncols = len(timepoints)

    fig = plt.figure(figsize=(11.8, max(22, 1.15 * nrows)), facecolor=BG)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 2,
        width_ratios=[1.55, 0.72] + [1] * ncols,
        left=0.052,
        right=0.86,
        bottom=0.055,
        top=0.925,
        hspace=0.075,
        wspace=0.045
    )

    for r, (group_label, gene, vals) in enumerate(marker_rows):
        gax = fig.add_subplot(gs[r, 0])
        gax.axis("off")
        gax.set_facecolor(BG)
        if group_label:
            gax.text(
                0.98, 0.5, group_label,
                ha="right", va="center",
                color=FG, fontsize=7.6, fontweight="bold"
            )

        gene_ax = fig.add_subplot(gs[r, 1])
        gene_ax.axis("off")
        gene_ax.set_facecolor(BG)
        gene_ax.text(
            0.96, 0.5, gene,
            ha="right", va="center",
            color=FG, fontsize=7.5, fontweight="bold"
        )

        for c, tp in enumerate(timepoints):
            ax = fig.add_subplot(gs[r, c + 2])
            cc = cache[tp]
            mask = cc["mask"]
            xt, yt, vt = x[mask], y[mask], vals[mask]

            z, support = smooth_on_grid(
                xt, yt, vt,
                cc["xedges"], cc["yedges"],
                sigma=args.sigma,
                support_frac=args.support_frac
            )

            draw_panel(
                ax, xt, yt, vt,
                cc["xedges"], cc["yedges"], cc["Xg"], cc["Yg"],
                z, support,
                cmap=CMAP_EXPR, vmin=0, vmax=1,
                title=str(tp) if r == 0 else None,
                diverging=False,
                raw_alpha=0.22,
                raw_size=1.15,
                contour_pack=contour_pack(cache, tp)
            )

    cax = fig.add_axes([0.885, 0.25, 0.014, 0.50])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP_EXPR), cax=cax)
    cb.set_label("Expression, robust-scaled per gene", color=FG, fontsize=8)
    cb.ax.tick_params(colors=FG, labelsize=7)
    cb.outline.set_edgecolor(FG)

    fig.suptitle(
        "Spatial progression atlas of representative marker genes",
        color=FG, fontsize=18, fontweight="bold", y=0.975
    )
    fig.text(
        0.5, 0.026,
        "Rows show representative genes grouped by biological program; columns show disease timepoints. "
        "Expression is robust-scaled per gene for visualization.",
        ha="center", va="center", color=MUTED, fontsize=7.5
    )

    outbase = os.path.join(outdir, "Fig_Step79B_Marker_TimepointProgressionAtlas_FINAL_v2")
    paths = save_all(fig, outbase, dpi=args.dpi)
    plt.close(fig)
    return paths


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

    p.add_argument("--label_col", default="state_group")
    p.add_argument("--state_prob_mode", default="label", choices=["label", "columns", "auto"])

    p.add_argument("--core_prob_col", default="auto")
    p.add_argument("--peri_prob_col", default="auto")
    p.add_argument("--remote_prob_col", default="auto")
    p.add_argument("--repair_score_col", default="auto")

    p.add_argument("--layer", default=None)

    p.add_argument("--grid_n", type=int, default=190)
    p.add_argument("--sigma", type=float, default=1.35)
    p.add_argument("--support_frac", type=float, default=0.025)
    p.add_argument("--pad_frac", type=float, default=0.045)
    p.add_argument("--module_vlim_quantile", type=float, default=98.5)
    p.add_argument("--dpi", type=int, default=450)

    return p.parse_args()


def main():
    args = parse_args()
    mkdirp(args.outdir)

    print("=" * 100)
    print("Step79 final v2 | State/timepoint-specific spatial progression atlas")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"outdir={args.outdir}")

    adata = ad.read_h5ad(args.h5ad)
    x, y, xy_audit = infer_xy(adata, args.x_col, args.y_col)
    df, merge_audit = merge_state_table(adata, args.state_table, args.state_id_col)

    if args.time_col not in df.columns:
        raise ValueError(f"time_col={args.time_col} not found.")

    prob_cols = infer_prob_cols(df, args)
    prob_audit_df = None
    prob_exact_duplicates = []
    prob_audit_paths = None
    prob_audit_csv = None

    if prob_cols is not None:
        prob_audit_df, prob_exact_duplicates = probability_audit(df, prob_cols)
        prob_audit_csv = os.path.join(args.outdir, "step79_probability_column_audit_final_v2.csv")
        prob_audit_df.to_csv(prob_audit_csv, index=False)
        prob_audit_paths = plot_probability_audit(
            prob_audit_df,
            os.path.join(args.outdir, "Fig_Step79C_ProbabilityColumnAudit_FINAL_v2"),
            args.dpi
        )

    # Decide source for state rows
    state_source = None
    if args.state_prob_mode == "columns":
        if prob_cols is None:
            raise ValueError("Requested --state_prob_mode columns but probability columns were not found.")
        if prob_exact_duplicates:
            raise ValueError(
                f"Requested columns mode but duplicated probability columns found: {prob_exact_duplicates}. "
                "Use --state_prob_mode label."
            )
        state_arrays = {
            k: pd.to_numeric(df[v], errors="coerce").astype(float).values
            for k, v in prob_cols.items()
        }
        state_source = "probability_columns"
        label_audit = None
    else:
        # label mode or auto fallback
        if args.state_prob_mode == "auto" and prob_cols is not None and not prob_exact_duplicates:
            state_arrays = {
                k: pd.to_numeric(df[v], errors="coerce").astype(float).values
                for k, v in prob_cols.items()
            }
            state_source = "probability_columns_auto"
            label_audit = None
        else:
            label_col = infer_label_col(df, args.label_col)
            state_arrays, label_audit = make_label_state_arrays(df, label_col)
            state_source = "label_onehot_smoothed"
            if prob_exact_duplicates:
                print("\nWARNING: probability columns are exactly duplicated; using label-derived state maps instead.")
                print(f"Duplicated pairs: {prob_exact_duplicates}")

    requested_tps = [t.strip() for t in args.timepoints.split(",") if t.strip()]
    present_tps = set(df[args.time_col].dropna().astype(str))
    timepoints = [t for t in requested_tps if t in present_tps]
    if not timepoints:
        timepoints = sorted(list(present_tps), key=lambda z: int(re.search(r"\d+", z).group()) if re.search(r"\d+", z) else z)

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

    repair_values = robust_minmax(module_scores["Repair / ECM"], 1, 99)
    repair_source = "module_score_repair_ECM"

    if args.repair_score_col != "auto":
        if args.repair_score_col not in df.columns:
            raise ValueError(f"repair_score_col not found: {args.repair_score_col}")
        repair_values = robust_minmax(pd.to_numeric(df[args.repair_score_col], errors="coerce").astype(float).values, 1, 99)
        repair_source = args.repair_score_col
    else:
        for c in ["repair_score", "rescue_score", "response_priority", "state_shift_priority_score"]:
            if c in df.columns:
                repair_values = robust_minmax(pd.to_numeric(df[c], errors="coerce").astype(float).values, 1, 99)
                repair_source = c
                break

    paths_A = make_state_module_atlas(
        df, x, y, timepoints, state_arrays, repair_values, module_scores, args, args.outdir
    )
    paths_B = make_marker_atlas(
        df, x, y, timepoints, marker_groups, marker_values, state_arrays, args, args.outdir
    )

    report = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "xy_audit": xy_audit,
        "state_merge_audit": merge_audit,
        "time_col": args.time_col,
        "timepoints_used": timepoints,
        "timepoint_counts": {
            t: int((df[args.time_col].astype(str) == str(t)).sum())
            for t in timepoints
        },
        "prob_cols_detected": prob_cols,
        "probability_exact_duplicate_pairs": prob_exact_duplicates,
        "state_row_source_used": state_source,
        "label_audit": label_audit,
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
            "Because the provided core/peri/remote probability columns can be duplicated in this state table, "
            "state rows are preferably generated from label-derived one-hot maps and then spatially smoothed. "
            "This is a spatial visualization of state/timepoint progression, not functional validation."
        )
    }

    report_path = os.path.join(args.outdir, "step79_final_v2_audit_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nFinished Step79 final v2.")
    print(f"Audit report: {report_path}")
    print(f"state_row_source_used: {state_source}")
    print("Outputs:")
    for k, v in report["outputs"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
