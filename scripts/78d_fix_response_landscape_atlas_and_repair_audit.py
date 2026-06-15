#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step78D | Fixed virtual perturbation response landscape

Purpose
-------
1) Fix row-label clipping in the multi-candidate atlas.
2) Avoid misleading blank Repair-ECM-up rescue panels:
   - main manuscript atlas keeps only rescue-positive candidates;
   - weak/reverse candidates are moved to supplement/signed audit figures;
   - signed delta maps are shown for weak/reverse candidates.

Interpretation
--------------
These are Gaussian-kernel-smoothed latent response landscapes.
They are visualization-level in silico counterfactual proxies,
not wet-lab knockout/blockade results and not observed cell-state transitions.
"""

import os
import re
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.stats import spearmanr

try:
    import anndata as ad
except Exception as e:
    raise RuntimeError("Please install anndata in the active environment.") from e

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# -----------------------------
# general utilities
# -----------------------------

def mkdirp(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def safe_str(x):
    if x is None:
        return ""
    return str(x)


def slugify(x):
    x = safe_str(x)
    x = x.replace("/", "_")
    x = re.sub(r"[^A-Za-z0-9._+-]+", "_", x)
    x = re.sub(r"_+", "_", x).strip("_")
    return x


def find_col(df, candidates, required=False, default=None):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    if required:
        raise ValueError(f"Cannot find required column among candidates: {candidates}")
    return default


def robust_minmax(x, qlow=1, qhigh=99):
    x = np.asarray(x, dtype=float)
    good = np.isfinite(x)
    if good.sum() == 0:
        return np.zeros_like(x, dtype=float)
    lo, hi = np.nanpercentile(x[good], [qlow, qhigh])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanmin(x[good]), np.nanmax(x[good])
    if hi <= lo:
        out = np.zeros_like(x, dtype=float)
        out[good] = 0.0
        return out
    out = (x - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def robust_z(x, clip=3.0):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-9
    z = (x - med) / (1.4826 * mad)
    return np.clip(z, -clip, clip)


# -----------------------------
# load data
# -----------------------------

def load_h5ad_and_state(h5ad, state_table, force_obsm_key="X_nicheformer"):
    print(f"Reading h5ad: {h5ad}")
    adata = ad.read_h5ad(h5ad)

    print(f"Reading state table: {state_table}")
    st = pd.read_csv(state_table)

    # ID matching
    st_id_col = find_col(
        st,
        ["obs_name", "barcode", "cell_id", "spot_id", "index", "Unnamed: 0"],
        required=False,
    )

    obs = adata.obs.copy()
    obs["_adata_id_"] = obs.index.astype(str)

    if st_id_col is not None:
        st["_state_id_"] = st[st_id_col].astype(str)
        merged = obs.merge(st, left_on="_adata_id_", right_on="_state_id_", how="left")
        overlap = int(merged["_state_id_"].notna().sum())
        merge_mode = "id"
    else:
        if len(st) != adata.n_obs:
            raise ValueError(
                "State table has no ID column and row number differs from adata.n_obs."
            )
        merged = pd.concat([obs.reset_index(drop=True), st.reset_index(drop=True)], axis=1)
        overlap = len(merged)
        merge_mode = "row_order"

    if overlap < 0.8 * adata.n_obs:
        raise ValueError(
            f"Low state-table overlap: {overlap}/{adata.n_obs}. Please check obs_name/barcode matching."
        )

    # latent coordinates
    if force_obsm_key and force_obsm_key in adata.obsm:
        X = np.asarray(adata.obsm[force_obsm_key])
        if X.ndim != 2:
            raise ValueError(f"{force_obsm_key} is not 2D.")
        if X.shape[1] >= 2:
            Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
            Z = PCA(n_components=2, random_state=0).fit_transform(Xs)
            latent_mode = f"obsm_forced_pca:{force_obsm_key}"
        else:
            raise ValueError(f"{force_obsm_key} has <2 dimensions.")
    elif "X_umap" in adata.obsm:
        Z = np.asarray(adata.obsm["X_umap"])[:, :2]
        latent_mode = "X_umap"
    elif "spatial" in adata.obsm:
        Z = np.asarray(adata.obsm["spatial"])[:, :2]
        latent_mode = "spatial_fallback"
    else:
        raise ValueError("No usable latent coordinates found.")

    # choose probability columns
    core_col = find_col(
        merged,
        ["core_probability", "core_prob", "prob_core", "p_core", "lesion_core_probability"],
        required=True,
    )
    peri_col = find_col(
        merged,
        ["peri_probability", "peri_prob", "prob_peri", "p_peri", "peri_infarct_probability"],
        required=False,
    )
    remote_col = find_col(
        merged,
        ["remote_probability", "remote_prob", "prob_remote", "p_remote", "remote_like_probability"],
        required=False,
    )
    repair_col = find_col(
        merged,
        ["repair_score", "repair_probability", "repair_prob", "rescue_score", "state_repair_score"],
        required=False,
    )

    if repair_col is None:
        # fallback: remote-like as weak repair proxy
        repair_col = remote_col if remote_col is not None else core_col
        print(f"[WARN] No repair score found. Using {repair_col} as fallback repair proxy.")

    df = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "latent1": Z[:, 0],
        "latent2": Z[:, 1],
        "core": pd.to_numeric(merged[core_col], errors="coerce").values,
        "repair": pd.to_numeric(merged[repair_col], errors="coerce").values,
    })

    if peri_col is not None:
        df["peri"] = pd.to_numeric(merged[peri_col], errors="coerce").values
    if remote_col is not None:
        df["remote"] = pd.to_numeric(merged[remote_col], errors="coerce").values

    df["core"] = np.nan_to_num(df["core"], nan=np.nanmedian(df["core"]))
    df["repair"] = np.nan_to_num(df["repair"], nan=np.nanmedian(df["repair"]))

    audit = {
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "merge_mode": merge_mode,
        "state_table_overlap": overlap,
        "latent_mode": latent_mode,
        "prob_cols": {
            "core": core_col,
            "repair": repair_col,
            "peri": peri_col,
            "remote": remote_col,
        },
    }
    return df, audit


# -----------------------------
# Step72 effect discovery
# -----------------------------

DEFAULT_EFFECTS = pd.DataFrame([
    {
        "perturbation": "Ccl2/Ccr2-Ackr1 blockade",
        "short_label": "Ccl2/Ccr2-Ackr1",
        "delta_core": -0.029,
        "delta_repair": +0.044,
    },
    {
        "perturbation": "Spp1-Cd44 blockade",
        "short_label": "Spp1-Cd44",
        "delta_core": -0.095,
        "delta_repair": +0.143,
    },
    {
        "perturbation": "Vegfa-Flt1/Kdr blockade",
        "short_label": "Vegfa-Flt1/Kdr",
        "delta_core": -0.022,
        "delta_repair": +0.034,
    },
    {
        "perturbation": "Ferroptosis down",
        "short_label": "Ferroptosis down",
        "delta_core": -0.108,
        "delta_repair": +0.163,
    },
    {
        "perturbation": "Repair-ECM up",
        "short_label": "Repair-ECM up",
        "delta_core": +0.096,
        "delta_repair": -0.144,
    },
])


def discover_csvs(root):
    if root is None or not os.path.isdir(root):
        return []
    out = []
    for p, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".csv"):
                out.append(os.path.join(p, f))
    return out


def infer_effect_table_from_step72(step72_dir):
    """
    Try to load candidate-level effects from Step72.
    Supports many possible column names.
    Falls back to DEFAULT_EFFECTS if no usable file is found.
    """
    csvs = discover_csvs(step72_dir)
    best = None
    best_score = -1

    for f in csvs:
        try:
            d = pd.read_csv(f, nrows=50)
        except Exception:
            continue

        cols = [c.lower() for c in d.columns]
        score = 0
        if any(("candidate" in c or "perturb" in c or "module" in c) for c in cols):
            score += 2
        if any(("delta" in c and "core" in c) or ("d_core" in c) for c in cols):
            score += 3
        if any(("delta" in c and ("repair" in c or "rescue" in c)) or ("d_repair" in c) for c in cols):
            score += 3
        if score > best_score:
            best_score = score
            best = f

    if best is None or best_score < 3:
        print("[WARN] No usable Step72 summary CSV found. Using built-in fallback effects.")
        eff = DEFAULT_EFFECTS.copy()
        eff["source"] = "fallback_default_from_previous_step72_summary"
        return eff, {"mode": "fallback_default", "source_file": None, "all_csvs": csvs}

    print(f"Using Step72 effect table: {best}")
    raw = pd.read_csv(best)

    cand_col = find_col(
        raw,
        ["perturbation", "candidate", "module", "target", "name", "label"],
        required=False,
    )
    if cand_col is None:
        # choose first object-like col
        object_cols = [c for c in raw.columns if raw[c].dtype == object]
        cand_col = object_cols[0] if object_cols else raw.columns[0]

    core_delta_col = find_col(
        raw,
        [
            "delta_core", "mean_delta_core", "d_core", "delta_core_probability",
            "mean_core_shift", "core_shift", "Δ core", "Δcore",
        ],
        required=False,
    )

    repair_delta_col = find_col(
        raw,
        [
            "delta_repair", "mean_delta_repair", "d_repair", "delta_repair_score",
            "delta_rescue_score", "mean_rescue_shift", "rescue_shift",
            "repair_shift", "Δ repair", "Δrepair",
        ],
        required=False,
    )

    if core_delta_col is None:
        # try any column containing both delta and core
        for c in raw.columns:
            lc = c.lower()
            if ("delta" in lc or "shift" in lc or "d_" in lc) and "core" in lc:
                core_delta_col = c
                break

    if repair_delta_col is None:
        for c in raw.columns:
            lc = c.lower()
            if ("delta" in lc or "shift" in lc or "d_" in lc) and (
                "repair" in lc or "rescue" in lc
            ):
                repair_delta_col = c
                break

    if core_delta_col is None or repair_delta_col is None:
        print("[WARN] Could not infer delta_core/delta_repair from Step72 table. Using fallback effects.")
        eff = DEFAULT_EFFECTS.copy()
        eff["source"] = "fallback_default_due_to_missing_delta_columns"
        return eff, {
            "mode": "fallback_default",
            "source_file": best,
            "reason": "missing_delta_columns",
            "columns": list(raw.columns),
        }

    eff = raw[[cand_col, core_delta_col, repair_delta_col]].copy()
    eff.columns = ["perturbation", "delta_core", "delta_repair"]
    eff["delta_core"] = pd.to_numeric(eff["delta_core"], errors="coerce")
    eff["delta_repair"] = pd.to_numeric(eff["delta_repair"], errors="coerce")
    eff = eff.dropna(subset=["perturbation", "delta_core", "delta_repair"])

    # standardize labels, keep known order if possible
    def clean_label(x):
        s = str(x)
        s = s.replace("_", " ")
        s = re.sub(r"\s+", " ", s).strip()
        return s

    eff["perturbation"] = eff["perturbation"].map(clean_label)

    # map close labels to preferred labels
    label_map = {
        "Ccl2 Ccr2 Ackr1 blockade": "Ccl2/Ccr2-Ackr1 blockade",
        "Ccl2/Ccr2 Ackr1 blockade": "Ccl2/Ccr2-Ackr1 blockade",
        "Spp1 Cd44 blockade": "Spp1-Cd44 blockade",
        "Vegfa Flt1 Kdr blockade": "Vegfa-Flt1/Kdr blockade",
        "Ferroptosis down modulation": "Ferroptosis down",
        "Ferroptosis down-modulation": "Ferroptosis down",
        "Repair ECM promotion": "Repair-ECM up",
        "Repair ECM up": "Repair-ECM up",
    }
    eff["perturbation"] = eff["perturbation"].replace(label_map)

    # if many rows, keep known rows when available, otherwise top 6 by rescue magnitude
    preferred = list(DEFAULT_EFFECTS["perturbation"])
    if eff["perturbation"].isin(preferred).sum() >= 3:
        eff = eff[eff["perturbation"].isin(preferred)].copy()
        eff["order"] = eff["perturbation"].map({x: i for i, x in enumerate(preferred)})
        eff = eff.sort_values("order").drop(columns="order")
    else:
        eff["rank_score"] = np.maximum(-eff["delta_core"], 0) + np.maximum(eff["delta_repair"], 0)
        eff = eff.sort_values("rank_score", ascending=False).head(6).drop(columns="rank_score")

    short_map = {
        "Ccl2/Ccr2-Ackr1 blockade": "Ccl2/Ccr2-Ackr1",
        "Spp1-Cd44 blockade": "Spp1-Cd44",
        "Vegfa-Flt1/Kdr blockade": "Vegfa-Flt1/Kdr",
        "Ferroptosis down": "Ferroptosis down",
        "Repair-ECM up": "Repair-ECM up",
    }
    eff["short_label"] = eff["perturbation"].map(short_map).fillna(eff["perturbation"])
    eff["source"] = best

    return eff, {
        "mode": "step72_summary",
        "source_file": best,
        "candidate_col": cand_col,
        "delta_core_col": core_delta_col,
        "delta_repair_col": repair_delta_col,
        "n_effect_rows": int(len(eff)),
    }


# -----------------------------
# response model
# -----------------------------

def compute_susceptibility(df):
    """
    Per-spot susceptibility weight.
    Rescue-like effects are more visible in core/high-injury or transition-like regions.
    This does not claim cell-fate transition.
    """
    core = robust_minmax(df["core"].values)
    repair = robust_minmax(df["repair"].values)

    # high core and non-max repair are more susceptible to rescue
    sus = 0.65 * core + 0.35 * (1 - repair)
    sus = robust_minmax(sus)
    sus = 0.25 + 0.75 * sus
    return sus


def build_per_cell_response(df, effects, amp=1.0):
    sus = compute_susceptibility(df)

    rows = []
    for _, r in effects.iterrows():
        dcore_mean = float(r["delta_core"])
        drepr_mean = float(r["delta_repair"])
        pert = str(r["perturbation"])
        short = str(r["short_label"])

        # Per-spot response proxy:
        # preserve candidate-level sign, modulate by susceptibility.
        dcore = amp * dcore_mean * sus
        drepr = amp * drepr_mean * sus

        # clipped counterfactual values
        baseline_core = df["core"].values
        baseline_repair = df["repair"].values
        pert_core = np.clip(baseline_core + dcore, 0, 1)
        pert_repair = np.clip(baseline_repair + drepr, 0, 1)

        # signed actual after clipping
        dcore_clip = pert_core - baseline_core
        drepr_clip = pert_repair - baseline_repair

        core_reduction = np.maximum(-dcore_clip, 0)
        repair_gain = np.maximum(drepr_clip, 0)

        rescue_strength = float(np.nanmean(core_reduction) + np.nanmean(repair_gain))
        reverse_strength = float(np.nanmean(np.maximum(dcore_clip, 0)) + np.nanmean(np.maximum(-drepr_clip, 0)))

        if rescue_strength >= reverse_strength and rescue_strength > 1e-5:
            response_class = "rescue_positive"
        elif reverse_strength > rescue_strength and reverse_strength > 1e-5:
            response_class = "weak_or_reverse"
        else:
            response_class = "near_zero"

        tmp = pd.DataFrame({
            "obs_name": df["obs_name"].values,
            "latent1": df["latent1"].values,
            "latent2": df["latent2"].values,
            "perturbation": pert,
            "short_label": short,
            "baseline_core": baseline_core,
            "baseline_repair": baseline_repair,
            "perturbed_core": pert_core,
            "perturbed_repair": pert_repair,
            "delta_core": dcore_clip,
            "delta_repair": drepr_clip,
            "core_reduction": core_reduction,
            "repair_gain": repair_gain,
            "response_priority": robust_minmax(core_reduction) * 0.55 + robust_minmax(repair_gain) * 0.45,
            "response_class": response_class,
            "candidate_mean_delta_core": dcore_mean,
            "candidate_mean_delta_repair": drepr_mean,
        })
        rows.append(tmp)

    return pd.concat(rows, axis=0, ignore_index=True)


# -----------------------------
# landscape smoothing
# -----------------------------

def make_grid(x, y, grid_n=220, pad=0.06):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xr = np.nanpercentile(x, [0.5, 99.5])
    yr = np.nanpercentile(y, [0.5, 99.5])
    dx = xr[1] - xr[0]
    dy = yr[1] - yr[0]
    xmin, xmax = xr[0] - pad * dx, xr[1] + pad * dx
    ymin, ymax = yr[0] - pad * dy, yr[1] + pad * dy
    gx = np.linspace(xmin, xmax, grid_n)
    gy = np.linspace(ymin, ymax, grid_n)
    return gx, gy


def smooth_to_grid(x, y, val, gx, gy, sigma=2.2, mask_quantile=0.03):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    val = np.asarray(val, float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(val)
    x, y, val = x[good], y[good], val[good]

    H_sum, _, _ = np.histogram2d(y, x, bins=[gy, gx], weights=val)
    H_cnt, _, _ = np.histogram2d(y, x, bins=[gy, gx])

    S_sum = gaussian_filter(H_sum, sigma=sigma)
    S_cnt = gaussian_filter(H_cnt, sigma=sigma)
    Z = S_sum / (S_cnt + 1e-9)

    density = S_cnt / (np.nanmax(S_cnt) + 1e-9)
    positive = density[density > 0]
    if len(positive) > 0:
        thr = np.nanquantile(positive, mask_quantile)
    else:
        thr = 0
    mask = density > thr
    Z = np.where(mask, Z, np.nan)

    # grid centers
    xc = (gx[:-1] + gx[1:]) / 2
    yc = (gy[:-1] + gy[1:]) / 2
    return xc, yc, Z, density[:-1, :-1] if density.shape == Z.shape else density


def find_peaks(Z, xc, yc, top_n=10, min_value=0.08):
    A = np.array(Z, float)
    A[~np.isfinite(A)] = -np.inf
    mx = maximum_filter(A, size=11)
    peaks = (A == mx) & np.isfinite(A) & (A >= min_value)
    yy, xx = np.where(peaks)
    vals = A[yy, xx]
    if len(vals) == 0:
        return pd.DataFrame(columns=["x", "y", "value"])
    order = np.argsort(vals)[::-1][:top_n]
    return pd.DataFrame({
        "x": xc[xx[order]],
        "y": yc[yy[order]],
        "value": vals[order],
    })


# -----------------------------
# plotting style
# -----------------------------

def setup_dark():
    plt.rcParams.update({
        "figure.facecolor": "#05080d",
        "axes.facecolor": "#05080d",
        "savefig.facecolor": "#05080d",
        "text.color": "white",
        "axes.labelcolor": "#d8dee9",
        "xtick.color": "#c8d0dc",
        "ytick.color": "#c8d0dc",
        "axes.edgecolor": "#6b7280",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
    })


def plot_landscape(
    ax,
    xc,
    yc,
    Z,
    xraw=None,
    yraw=None,
    raw_color=None,
    title="",
    cmap="magma",
    vmin=0,
    vmax=1,
    contours=True,
    signed=False,
    label_axes=True,
    peak_df=None,
):
    ax.set_facecolor("#05080d")
    if signed:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.imshow(
        Z,
        origin="lower",
        extent=[xc.min(), xc.max(), yc.min(), yc.max()],
        cmap=cmap,
        norm=norm,
        interpolation="bilinear",
        aspect="auto",
        alpha=0.96,
    )

    if contours:
        finite = Z[np.isfinite(Z)]
        if len(finite) > 20 and np.nanmax(finite) > np.nanmin(finite):
            if signed:
                levels = np.linspace(vmin, vmax, 9)
            else:
                levels = np.linspace(max(vmin, np.nanpercentile(finite, 35)), vmax, 7)
            try:
                ax.contour(
                    xc,
                    yc,
                    Z,
                    levels=levels,
                    colors="#b8c7d9",
                    linewidths=0.35,
                    alpha=0.45,
                )
            except Exception:
                pass

    if xraw is not None and yraw is not None:
        if raw_color is None:
            ax.scatter(xraw, yraw, s=0.7, c="#2aa7d8", alpha=0.20, linewidths=0)
        else:
            ax.scatter(xraw, yraw, s=0.8, c=raw_color, alpha=0.15, linewidths=0)

    if peak_df is not None and len(peak_df) > 0:
        ax.scatter(
            peak_df["x"], peak_df["y"],
            s=34,
            marker="*",
            c="#ffe700",
            edgecolors="#111111",
            linewidths=0.4,
            zorder=8,
        )

    ax.set_title(title, pad=6)
    if label_axes:
        ax.set_xlabel("Latent coordinate 1")
        ax.set_ylabel("Latent coordinate 2")
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    ax.grid(True, color="#27313d", lw=0.35, alpha=0.55)
    return im


def add_panel_label(ax, label):
    ax.text(
        -0.075, 1.04, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=17, fontweight="bold",
        color="white",
        clip_on=False,
    )


# -----------------------------
# figures
# -----------------------------

def classify_candidates(per_cell):
    summary = (
        per_cell
        .groupby(["perturbation", "short_label", "response_class"], dropna=False)
        .agg(
            mean_delta_core=("delta_core", "mean"),
            mean_delta_repair=("delta_repair", "mean"),
            mean_core_reduction=("core_reduction", "mean"),
            mean_repair_gain=("repair_gain", "mean"),
            mean_response_priority=("response_priority", "mean"),
            max_response_priority=("response_priority", "max"),
        )
        .reset_index()
    )
    return summary


def make_atlas(
    per_cell,
    candidate_summary,
    outdir,
    basename,
    include_mode="main_rescue",
    grid_n=210,
    sigma=2.2,
):
    """
    include_mode:
    - main_rescue: only rescue_positive candidates
    - all_signed: all candidates, signed delta panels; weak/reverse included
    """
    setup_dark()

    if include_mode == "main_rescue":
        keep = candidate_summary[candidate_summary["response_class"].eq("rescue_positive")].copy()
        title = "Virtual perturbation response landscapes"
        note = (
            "Gaussian-kernel-smoothed latent response landscapes. "
            "Core reduction = max(-Δcore probability, 0); repair gain = max(Δrepair score, 0). "
            "Only rescue-positive candidates are shown; weak/reverse responses are provided as signed audit figures."
        )
        col_titles = ["Baseline core map", "Core reduction", "Repair gain", "Response priority"]
        signed_cols = [False, False, False, False]
    else:
        keep = candidate_summary.copy()
        title = "Virtual perturbation response landscapes | signed audit"
        note = (
            "Signed audit atlas. Δcore and Δrepair are shown with diverging scales; "
            "weak/reverse candidates are retained for interpretation rather than forced into rescue-positive maps."
        )
        col_titles = ["Baseline core map", "Signed Δcore", "Signed Δrepair", "Response priority"]
        signed_cols = [False, True, True, False]

    if len(keep) == 0:
        print(f"[WARN] No candidates for atlas mode: {include_mode}")
        return None

    # preserve original order from per_cell
    order = per_cell[["perturbation", "short_label"]].drop_duplicates()
    keep = order.merge(keep, on=["perturbation", "short_label"], how="inner")

    nrows = len(keep)
    ncols = 4

    # left label axis fixes clipping
    fig = plt.figure(figsize=(18, max(3.2, 2.75 * nrows)), facecolor="#05080d")
    gs = gridspec.GridSpec(
        nrows=nrows + 1,
        ncols=ncols + 2,
        height_ratios=[0.22] + [1] * nrows,
        width_ratios=[1.45, 4.0, 4.0, 4.0, 4.0, 0.32],
        left=0.045,
        right=0.91,
        top=0.92,
        bottom=0.09,
        wspace=0.10,
        hspace=0.16,
    )

    fig.suptitle(title, fontsize=24, fontweight="bold", y=0.975, color="white")

    # column titles
    for j, ct in enumerate(col_titles):
        axh = fig.add_subplot(gs[0, j + 1])
        axh.axis("off")
        axh.text(
            0.5, 0.25, ct,
            ha="center", va="center",
            fontsize=12.5, fontweight="bold",
            color="white",
        )

    ims = {}
    xall = per_cell["latent1"].values
    yall = per_cell["latent2"].values
    gx, gy = make_grid(xall, yall, grid_n=grid_n)

    for i, row in keep.iterrows():
        pert = row["perturbation"]
        short = row["short_label"]
        cls = row["response_class"]

        d = per_cell[per_cell["perturbation"].eq(pert)].copy()

        # row-label axis
        axlab = fig.add_subplot(gs[i + 1, 0])
        axlab.axis("off")
        label = short
        if include_mode != "main_rescue" and cls != "rescue_positive":
            label = f"{short}\n{cls.replace('_', '/')}"
        axlab.text(
            0.98, 0.50, label,
            ha="right", va="center",
            fontsize=11.2, fontweight="bold",
            color="white",
            linespacing=1.05,
            clip_on=False,
        )

        values = [
            d["baseline_core"].values,
            d["core_reduction"].values if include_mode == "main_rescue" else d["delta_core"].values,
            d["repair_gain"].values if include_mode == "main_rescue" else d["delta_repair"].values,
            d["response_priority"].values,
        ]

        cmaps = [
            "magma",
            "Blues_r" if include_mode == "main_rescue" else "coolwarm",
            "YlGn" if include_mode == "main_rescue" else "PiYG",
            "viridis",
        ]

        # positive scales
        vmaxs = [
            1.0,
            max(0.05, np.nanpercentile(np.abs(per_cell["core_reduction" if include_mode == "main_rescue" else "delta_core"]), 99)),
            max(0.05, np.nanpercentile(np.abs(per_cell["repair_gain" if include_mode == "main_rescue" else "delta_repair"]), 99)),
            1.0,
        ]

        for j in range(ncols):
            ax = fig.add_subplot(gs[i + 1, j + 1])
            val = values[j]
            xc, yc, Z, _ = smooth_to_grid(
                d["latent1"].values,
                d["latent2"].values,
                val,
                gx,
                gy,
                sigma=sigma,
                mask_quantile=0.03,
            )

            if signed_cols[j]:
                vmax = max(0.02, np.nanpercentile(np.abs(val[np.isfinite(val)]), 99))
                vmin = -vmax
            else:
                vmin = 0
                vmax = vmaxs[j]

            peaks = None
            if j == 3:
                peaks = find_peaks(Z, xc, yc, top_n=8, min_value=0.10)

            im = plot_landscape(
                ax,
                xc, yc, Z,
                xraw=d["latent1"].values,
                yraw=d["latent2"].values,
                title="",
                cmap=cmaps[j],
                vmin=vmin,
                vmax=vmax,
                contours=True,
                signed=signed_cols[j],
                label_axes=True,
                peak_df=peaks,
            )
            if i == 0:
                ims[j] = (im, cmaps[j], vmin, vmax, signed_cols[j])

    # colorbars
    cax1 = fig.add_subplot(gs[1: max(2, min(nrows + 1, 3)), -1])
    cb1 = fig.colorbar(ims[0][0], cax=cax1)
    cb1.set_label("Core probability", color="white")
    cb1.ax.tick_params(colors="white")

    if include_mode == "main_rescue":
        cax2 = fig.add_subplot(gs[max(2, nrows // 2): nrows + 1, -1])
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap="viridis")
        cb2 = fig.colorbar(sm, cax=cax2)
        cb2.set_label("Response priority", color="white")
        cb2.ax.tick_params(colors="white")
    else:
        cax2 = fig.add_subplot(gs[max(2, nrows // 2): nrows + 1, -1])
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(norm=norm, cmap="viridis")
        cb2 = fig.colorbar(sm, cax=cax2)
        cb2.set_label("Response priority", color="white")
        cb2.ax.tick_params(colors="white")

    fig.text(
        0.50, 0.035,
        note + " These are visualization-level in silico perturbation proxies, not observed cell-state transitions.",
        ha="center", va="center",
        fontsize=10.2,
        color="#d8dee9",
        wrap=True,
    )

    png = os.path.join(outdir, f"{basename}.png")
    pdf = os.path.join(outdir, f"{basename}.pdf")
    svg = os.path.join(outdir, f"{basename}.svg")

    fig.savefig(png, dpi=420, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)

    return {"png": png, "pdf": pdf, "svg": svg}


def make_single_candidate_figure(
    per_cell,
    candidate,
    outdir,
    grid_n=230,
    sigma=2.2,
    signed_audit=False,
):
    setup_dark()

    d = per_cell[per_cell["perturbation"].eq(candidate)].copy()
    if len(d) == 0:
        return None

    short = d["short_label"].iloc[0]
    response_class = d["response_class"].iloc[0]

    x = d["latent1"].values
    y = d["latent2"].values
    gx, gy = make_grid(x, y, grid_n=grid_n)

    if signed_audit or response_class != "rescue_positive":
        title_suffix = f"{short} | signed response audit"
        values = [
            d["baseline_core"].values,
            d["delta_core"].values,
            d["delta_repair"].values,
            d["response_priority"].values,
        ]
        titles = [
            "Baseline core-probability landscape",
            "Signed Δcore-probability landscape",
            "Signed Δrepair-score landscape",
            "Response-priority basin map",
        ]
        cmaps = ["magma", "coolwarm", "PiYG", "viridis"]
        signed = [False, True, True, False]
        note = (
            "Signed response audit. Positive Δcore or negative Δrepair indicates weak/reverse behavior under this surrogate setting."
        )
    else:
        title_suffix = short
        values = [
            d["baseline_core"].values,
            d["core_reduction"].values,
            d["repair_gain"].values,
            d["response_priority"].values,
        ]
        titles = [
            "Baseline core-probability landscape",
            "Core-reduction landscape",
            "Repair-gain landscape",
            "Response-priority basin map",
        ]
        cmaps = ["magma", "Blues_r", "YlGn", "viridis"]
        signed = [False, False, False, False]
        note = (
            "Gaussian-kernel-smoothed latent response landscape. Peaks indicate local high-response basins."
        )

    fig = plt.figure(figsize=(15.5, 10.4), facecolor="#05080d")
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[1, 1, 0.055],
        left=0.065,
        right=0.90,
        top=0.86,
        bottom=0.10,
        wspace=0.13,
        hspace=0.24,
    )

    fig.suptitle(
        f"Virtual perturbation response landscape | {title_suffix}",
        fontsize=24,
        fontweight="bold",
        y=0.965,
        color="white",
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]

    ims = []
    letters = ["A", "B", "C", "D"]

    for k, ax in enumerate(axes):
        val = values[k]
        xc, yc, Z, _ = smooth_to_grid(x, y, val, gx, gy, sigma=sigma, mask_quantile=0.03)

        if signed[k]:
            vmax = max(0.02, np.nanpercentile(np.abs(val[np.isfinite(val)]), 99))
            vmin = -vmax
        else:
            vmin = 0.0
            if k == 0:
                vmax = 1.0
            elif k == 3:
                vmax = 1.0
            else:
                vmax = max(0.05, np.nanpercentile(val[np.isfinite(val)], 99))
                if not np.isfinite(vmax) or vmax <= 0:
                    vmax = 0.05

        peaks = find_peaks(Z, xc, yc, top_n=10, min_value=0.10) if k == 3 else None

        im = plot_landscape(
            ax,
            xc, yc, Z,
            xraw=x,
            yraw=y,
            title=titles[k],
            cmap=cmaps[k],
            vmin=vmin,
            vmax=vmax,
            contours=True,
            signed=signed[k],
            label_axes=True,
            peak_df=peaks,
        )
        add_panel_label(ax, letters[k])
        ims.append(im)

    cax = fig.add_subplot(gs[:, 2])
    cb = fig.colorbar(ims[-1], cax=cax)
    cb.set_label("Response priority" if not signed_audit else "Response/audit value", color="white")
    cb.ax.tick_params(colors="white")

    if response_class != "rescue_positive":
        fig.text(
            0.50, 0.055,
            note + " This candidate is not included in the main rescue-positive atlas.",
            ha="center",
            va="center",
            fontsize=10.6,
            color="#d8dee9",
            wrap=True,
        )
    else:
        fig.text(
            0.50, 0.055,
            note + " This is a visualization-level counterfactual proxy, not a wet-lab KO/blockade result or observed cell-state transition.",
            ha="center",
            va="center",
            fontsize=10.6,
            color="#d8dee9",
            wrap=True,
        )

    tag = slugify(short)
    suffix = "signed_audit" if signed_audit or response_class != "rescue_positive" else "final"
    png = os.path.join(outdir, f"Fig_Step78D_ResponseLandscape_{tag}_{suffix}.png")
    pdf = os.path.join(outdir, f"Fig_Step78D_ResponseLandscape_{tag}_{suffix}.pdf")
    svg = os.path.join(outdir, f"Fig_Step78D_ResponseLandscape_{tag}_{suffix}.svg")

    fig.savefig(png, dpi=420, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.22)
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)

    return {"png": png, "pdf": pdf, "svg": svg}


# -----------------------------
# main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--state_table", required=True)
    ap.add_argument("--step72_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--force_obsm_key", default="X_nicheformer")
    ap.add_argument("--grid_n", type=int, default=220)
    ap.add_argument("--sigma", type=float, default=2.2)
    ap.add_argument("--amp", type=float, default=1.0)
    ap.add_argument("--main_exclude_weak_reverse", action="store_true")
    args = ap.parse_args()

    mkdirp(args.outdir)

    print("=" * 100)
    print("Step78D | Fixed response-landscape atlas and weak/reverse audit")
    print("=" * 100)
    print(f"h5ad={args.h5ad}")
    print(f"state_table={args.state_table}")
    print(f"step72_dir={args.step72_dir}")
    print(f"outdir={args.outdir}")

    df, data_audit = load_h5ad_and_state(
        args.h5ad,
        args.state_table,
        force_obsm_key=args.force_obsm_key,
    )

    effects, eff_audit = infer_effect_table_from_step72(args.step72_dir)
    effects_path = os.path.join(args.outdir, "step78d_candidate_effects_used.csv")
    effects.to_csv(effects_path, index=False)

    per_cell = build_per_cell_response(df, effects, amp=args.amp)
    per_cell_path = os.path.join(args.outdir, "step78d_per_cell_response_proxy.csv")
    per_cell.to_csv(per_cell_path, index=False)

    summary = classify_candidates(per_cell)
    summary_path = os.path.join(args.outdir, "step78d_candidate_response_classification.csv")
    summary.to_csv(summary_path, index=False)

    print("\n---- candidate response classification ----")
    print(summary.to_string(index=False))

    # Main manuscript atlas: exclude weak/reverse automatically.
    main_atlas = make_atlas(
        per_cell,
        summary,
        args.outdir,
        basename="Fig_Step78D_VirtualPerturbationResponseLandscape_main_rescue_positive_atlas",
        include_mode="main_rescue",
        grid_n=args.grid_n,
        sigma=args.sigma,
    )

    # Supplement signed atlas: includes Repair-ECM up and any weak/reverse candidate.
    signed_atlas = make_atlas(
        per_cell,
        summary,
        args.outdir,
        basename="Fig_Step78D_VirtualPerturbationResponseLandscape_signed_audit_atlas",
        include_mode="all_signed",
        grid_n=args.grid_n,
        sigma=args.sigma,
    )

    # Individual figures.
    single_figs = {}
    for pert in effects["perturbation"].tolist():
        cls = summary.loc[summary["perturbation"].eq(pert), "response_class"]
        cls = cls.iloc[0] if len(cls) else "unknown"
        single_figs[pert] = make_single_candidate_figure(
            per_cell,
            pert,
            args.outdir,
            grid_n=args.grid_n,
            sigma=args.sigma,
            signed_audit=(cls != "rescue_positive"),
        )

    report = {
        "status": "ok",
        "data_audit": data_audit,
        "effect_audit": eff_audit,
        "effects_csv": effects_path,
        "per_cell_response_csv": per_cell_path,
        "candidate_summary_csv": summary_path,
        "main_atlas": main_atlas,
        "signed_audit_atlas": signed_atlas,
        "single_figures": single_figs,
        "interpretation_note": (
            "Step78D fixes row-label clipping by using a dedicated label column. "
            "Repair-ECM up and other weak/reverse candidates are not forced into rescue-positive maps; "
            "they are shown as signed audits. All maps are Gaussian-kernel-smoothed latent response proxies, "
            "not wet-lab KO/blockade results and not observed cell-state transitions."
        ),
    }

    report_path = os.path.join(args.outdir, "step78d_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\nDONE Step78D")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
