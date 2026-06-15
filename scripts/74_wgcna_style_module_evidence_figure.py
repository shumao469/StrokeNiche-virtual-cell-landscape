#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step74 | WGCNA-style module evidence figure

Purpose
-------
Generate a WGCNA-style module evidence figure for StrokeNiche:
  A. module eigengene dendrogram
  B. module-trait correlation heatmap
  C. module score spatial maps
  D. module dotplot across time/state groups

Important interpretation
------------------------
This is WGCNA-style module visualization based on curated / Step65-derived module
gene sets and module score eigengenes. It is not a full R-WGCNA topological
overlap network unless explicitly run with WGCNA in R.

Outputs
-------
  Fig_Step74_WGCNAStyleModuleEvidence_annotated.pdf/svg/png
  Fig_Step74_WGCNAStyleModuleEvidence_clean_no_text.pdf/svg/png
  step74_module_gene_sets_used.csv
  step74_module_scores_by_observation.csv         optional if --save_cell_scores
  step74_module_trait_correlations.csv
  step74_module_dotplot_summary.csv
  step74_module_spatial_plot_cells.csv
  step74_audit_report.json
  step74_report.json
"""

from pathlib import Path
import argparse
import json
import re
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpecFromSubplotSpec

try:
    import anndata as ad
except Exception as e:
    ad = None

try:
    from scipy import sparse
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import squareform
    from scipy.stats import spearmanr
except Exception as e:
    sparse = None
    linkage = None
    dendrogram = None
    leaves_list = None
    squareform = None
    spearmanr = None


CURATED_MODULES = {
    "Hypoxia_redox": [
        "Hif1a", "Epas1", "Hmox1", "Nfe2l1", "Nfe2l2", "Sod1", "Sod2", "Txn1",
        "Prdx1", "Prdx2", "Gpx1", "Gpx4", "Nqo1", "Slc2a1", "Ldha", "Vegfa"
    ],
    "Ferroptosis": [
        "Acsl4", "Gpx4", "Slc7a11", "Tfrc", "Fth1", "Ftl1", "Ptgs2", "Alox5",
        "Alox15", "Hmox1", "Nfe2l2", "Sat1", "Chac1", "Lpcat3", "Cox2"
    ],
    "BBB_endothelial": [
        "Pecam1", "Cldn5", "Ocln", "Tjp1", "Kdr", "Flt1", "Vwf", "Klf2",
        "Klf4", "Cdh5", "Eng", "Nos3", "Slc2a1", "Abcb1a", "Mfsd2a"
    ],
    "Inflammation_chemotaxis": [
        "Ccl2", "Ccr2", "Ackr1", "Tnf", "Il1b", "Il6", "Cxcl1", "Cxcl2",
        "Ccl3", "Ccl4", "Nfkb1", "Rela", "Ctsd", "C1qa", "C1qb", "C1qc"
    ],
    "Microglia_myeloid": [
        "Aif1", "Tmem119", "P2ry12", "Cx3cr1", "Trem2", "Apoe", "Lpl", "Tyrobp",
        "Csf1r", "Itgam", "Cd68", "Cst7", "Ctsb", "Ctsd", "Lgals3"
    ],
    "Astrocyte_reactive": [
        "Gfap", "Vim", "Serpina3n", "Lcn2", "Aqp4", "S100b", "S100a10",
        "Clu", "C3", "Stat3", "Sparcl1", "Aldh1l1", "Slc1a2"
    ],
    "Repair_ECM": [
        "Fn1", "Col1a1", "Col1a2", "Col3a1", "Col4a1", "Col4a2", "Sparc",
        "Postn", "Mmp2", "Mmp9", "Timp1", "Timp2", "Thbs1", "Vcan", "Dcn", "Lum"
    ],
    "Synaptic_recovery": [
        "Snap25", "Syp", "Syt1", "Dlg4", "Map2", "Rbfox3", "Grin1", "Gria1",
        "Bdnf", "Gap43", "Tubb3", "Mbp", "Plp1", "Mog", "Mag", "Sox10", "Olig1", "Olig2"
    ],
}


PROGRAM_COLORS = {
    "Hypoxia_redox": "#8dd3c7",
    "Ferroptosis": "#fb8072",
    "BBB_endothelial": "#80b1d3",
    "Inflammation_chemotaxis": "#fdb462",
    "Microglia_myeloid": "#b3de69",
    "Astrocyte_reactive": "#ccebc5",
    "Repair_ECM": "#bebada",
    "Synaptic_recovery": "#fccde5",
}


def log(msg):
    print(msg, flush=True)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table(path, required=False):
    if not path:
        if required:
            raise FileNotFoundError("Empty path")
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(str(p))
        return pd.DataFrame()
    if p.suffix.lower() in [".tsv", ".txt"]:
        return pd.read_csv(p, sep="\t", low_memory=False)
    return pd.read_csv(p, low_memory=False)


def clean_symbol(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    x = re.split(r"[;,|/\s]+", x)[0]
    x = re.sub(r"[^A-Za-z0-9_.-]", "", x)
    return x


def norm_gene(x):
    return clean_symbol(x).upper()


def first_col(df, names, contains=None):
    if df is None or df.empty:
        return ""
    lower_map = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n in df.columns:
            return n
        if n.lower() in lower_map:
            return lower_map[n.lower()]
    if contains:
        for c in df.columns:
            lc = str(c).lower()
            if any(k.lower() in lc for k in contains):
                return c
    return ""


def infer_gene_col(df):
    return first_col(
        df,
        ["gene", "genes", "gene_symbol", "symbol", "marker", "feature", "dynamic_gene"],
        contains=["gene", "symbol", "marker", "feature"]
    )


def infer_module_col(df):
    return first_col(
        df,
        ["module", "module_key", "module_name", "pathway", "program", "signature"],
        contains=["module", "pathway", "program", "signature"]
    )


def build_module_sets(module_gene_table, min_genes=3):
    modules = {k: list(v) for k, v in CURATED_MODULES.items()}
    audit = {
        "source": "curated_default_plus_optional_table",
        "curated_modules": {k: len(v) for k, v in modules.items()},
        "table_loaded": False,
        "table_module_col": "",
        "table_gene_col": "",
        "n_table_rows": 0,
    }

    df = read_table(module_gene_table) if module_gene_table else pd.DataFrame()
    if not df.empty:
        audit["table_loaded"] = True
        audit["n_table_rows"] = int(len(df))
        mcol = infer_module_col(df)
        gcol = infer_gene_col(df)
        audit["table_module_col"] = mcol
        audit["table_gene_col"] = gcol

        if mcol and gcol:
            for module, sub in df.groupby(mcol):
                module = str(module).strip()
                if not module or module.lower() in {"nan", "none"}:
                    continue
                genes = [clean_symbol(x) for x in sub[gcol].dropna().astype(str)]
                genes = [g for g in genes if g]
                genes = list(dict.fromkeys(genes))
                if len(genes) >= min_genes:
                    safe_module = re.sub(r"[^A-Za-z0-9_.-]+", "_", module)
                    modules[safe_module] = genes

    # remove duplicated genes inside each module
    final = {}
    for m, genes in modules.items():
        clean = []
        for g in genes:
            g = clean_symbol(g)
            if g:
                clean.append(g)
        clean = list(dict.fromkeys(clean))
        if len(clean) >= min_genes:
            final[m] = clean

    return final, audit


def get_gene_name_map(adata):
    mapping = {}
    for v in list(adata.var_names):
        mapping[norm_gene(v)] = v

    for col in ["gene", "genes", "gene_symbol", "symbol", "feature_name", "gene_name"]:
        if col in adata.var.columns:
            for var_name, val in zip(adata.var_names, adata.var[col].astype(str).values):
                key = norm_gene(val)
                if key and key not in mapping:
                    mapping[key] = var_name
    return mapping


def subset_adata_cells(adata, max_cells=0, seed=20260601):
    n = adata.n_obs
    if not max_cells or max_cells <= 0 or n <= max_cells:
        return adata, np.arange(n)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_cells, replace=False))
    return adata[idx].copy(), idx


def matrix_to_dense(X):
    if sparse is not None and sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def zscore_columns(X):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[~np.isfinite(sd) | (sd < 1e-12)] = 1.0
    Z = (X - mu) / sd
    Z[~np.isfinite(Z)] = 0.0
    return Z


def compute_module_scores(adata, module_sets, min_matched_genes=3):
    gene_map = get_gene_name_map(adata)
    score_df = pd.DataFrame(index=adata.obs_names.astype(str))
    gene_set_rows = []

    for module, genes in module_sets.items():
        matched_var = []
        matched_symbol = []
        for g in genes:
            key = norm_gene(g)
            if key in gene_map:
                matched_var.append(gene_map[key])
                matched_symbol.append(g)

        matched_var = list(dict.fromkeys(matched_var))
        matched_symbol = list(dict.fromkeys(matched_symbol))

        gene_set_rows.append({
            "module": module,
            "n_input_genes": len(genes),
            "n_matched_genes": len(matched_var),
            "matched_genes": ";".join(matched_symbol),
            "matched_var_names": ";".join(matched_var),
        })

        if len(matched_var) < min_matched_genes:
            continue

        X = matrix_to_dense(adata[:, matched_var].X)
        Z = zscore_columns(X)
        score = np.nanmean(Z, axis=1)
        score_df[module] = score

    gene_sets_used = pd.DataFrame(gene_set_rows)
    return score_df, gene_sets_used


def find_obs_merge_key(adata, state_table):
    obs_names = pd.Index(adata.obs_names.astype(str))
    if state_table.empty:
        return "", 0

    candidates = ["obs_name", "cell_id", "spot_id", "barcode", "cell", "spot", "id"]
    best_col = ""
    best_overlap = 0

    for c in candidates:
        if c in state_table.columns:
            vals = pd.Index(state_table[c].astype(str))
            overlap = len(obs_names.intersection(vals))
            if overlap > best_overlap:
                best_overlap = overlap
                best_col = c

    return best_col, best_overlap


def merge_metadata(adata, state_table):
    meta = adata.obs.copy()
    meta = meta.reset_index().rename(columns={"index": "obs_name"})
    if "obs_name" not in meta.columns:
        meta["obs_name"] = adata.obs_names.astype(str)

    if not state_table.empty:
        key, overlap = find_obs_merge_key(adata, state_table)
        if key and overlap > 0:
            st = state_table.copy()
            st[key] = st[key].astype(str)
            meta["obs_name"] = meta["obs_name"].astype(str)
            st = st.drop_duplicates(key)
            meta = meta.merge(st, left_on="obs_name", right_on=key, how="left", suffixes=("", "_state"))
        else:
            overlap = 0
    else:
        key, overlap = "", 0

    audit = {"state_merge_key": key, "state_overlap": int(overlap)}
    return meta, audit


def find_xy_cols(meta):
    candidates = [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
        ("coord_x", "coord_y"),
        ("array_col", "array_row"),
        ("imagecol", "imagerow"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("baseline_coord_x_from_ref", "baseline_coord_y_from_ref"),
        ("UMAP_1", "UMAP_2"),
        ("umap_1", "umap_2"),
        ("latent_x", "latent_y"),
    ]

    for xcol, ycol in candidates:
        if xcol in meta.columns and ycol in meta.columns:
            x = pd.to_numeric(meta[xcol], errors="coerce")
            y = pd.to_numeric(meta[ycol], errors="coerce")
            if x.notna().sum() > 10 and y.notna().sum() > 10:
                return xcol, ycol

    # fallback: first two numeric columns containing coord/umap/spatial
    numeric = []
    for c in meta.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["coord", "umap", "spatial", "array", "image", "pxl"]):
            vals = pd.to_numeric(meta[c], errors="coerce")
            if vals.notna().sum() > 10:
                numeric.append(c)
    if len(numeric) >= 2:
        return numeric[0], numeric[1]
    return "", ""


def find_group_col(meta):
    candidates = [
        "state_group", "region_auto", "region_refined", "region_manual_final",
        "condition", "timepoint", "stage", "group", "day", "sample_label"
    ]
    for c in candidates:
        if c in meta.columns:
            vc = meta[c].astype(str).value_counts()
            if 2 <= len(vc) <= 20:
                return c
    return ""


def numeric_trait_table(meta):
    traits = pd.DataFrame(index=meta.index)

    numeric_candidates = [
        "stage_numeric", "core_probability", "peri_probability", "remote_probability",
        "repair_score", "injury_score", "predicted_repair_shift", "predicted_core_reversal",
        "module_score", "core_reduction", "repair_gain"
    ]

    for c in numeric_candidates:
        if c in meta.columns:
            vals = pd.to_numeric(meta[c], errors="coerce")
            if vals.notna().sum() > 10 and vals.nunique(dropna=True) > 2:
                traits[c] = vals

    categorical_candidates = [
        "timepoint", "condition", "stage", "group", "region_auto", "region_refined",
        "region_manual_final", "state_group", "sample_label"
    ]

    for c in categorical_candidates:
        if c in meta.columns:
            s = meta[c].astype(str)
            vc = s.value_counts()
            levels = [x for x in vc.index if x.lower() not in {"nan", "none"} and vc.loc[x] >= 10]
            if 2 <= len(levels) <= 8:
                for lev in levels:
                    name = f"{c}_{lev}"
                    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
                    traits[name] = (s == lev).astype(float)

    traits = traits.loc[:, traits.notna().sum(axis=0) > 10]
    traits = traits.loc[:, traits.nunique(dropna=True) > 1]
    return traits


def bh_fdr(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    vals = p[ok]
    order = np.argsort(vals)
    ranked = vals[order]
    n = len(ranked)
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    tmp = np.empty(n)
    tmp[order] = adj
    out[ok] = tmp
    return out


def module_trait_correlations(scores, traits):
    rows = []
    for m in scores.columns:
        x = pd.to_numeric(scores[m], errors="coerce")
        for t in traits.columns:
            y = pd.to_numeric(traits[t], errors="coerce")
            ok = x.notna() & y.notna()
            if ok.sum() < 10:
                rho, p = np.nan, np.nan
            else:
                try:
                    rho, p = spearmanr(x[ok], y[ok])
                except Exception:
                    rho, p = np.nan, np.nan
            rows.append({
                "module": m,
                "trait": t,
                "spearman_rho": rho,
                "p_value": p,
                "n": int(ok.sum()),
            })
    df = pd.DataFrame(rows)
    df["bh_fdr"] = bh_fdr(df["p_value"].values)
    return df


def module_order_by_dendrogram(scores):
    modules = list(scores.columns)
    if len(modules) <= 2 or linkage is None:
        return modules, None

    corr = scores.corr(method="spearman").fillna(0.0)
    dist = 1 - corr
    dist = dist.clip(lower=0, upper=2)
    np.fill_diagonal(dist.values, 0)
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)
    ordered = [modules[i] for i in order]
    return ordered, Z


def trait_order_by_signal(corr_df, max_traits=12):
    if corr_df.empty:
        return []
    tmp = (
        corr_df.groupby("trait")["spearman_rho"]
        .apply(lambda x: np.nanmax(np.abs(x)))
        .sort_values(ascending=False)
    )
    return tmp.head(max_traits).index.tolist()


def summarize_dotplot(scores, meta, group_col):
    if not group_col:
        return pd.DataFrame()
    df = scores.copy()
    df[group_col] = meta[group_col].astype(str).values

    rows = []
    for g, sub in df.groupby(group_col):
        if str(g).lower() in {"nan", "none"}:
            continue
        for m in scores.columns:
            vals = pd.to_numeric(sub[m], errors="coerce")
            rows.append({
                "group": g,
                "module": m,
                "mean_score": float(vals.mean()),
                "median_score": float(vals.median()),
                "pct_positive": float((vals > 0).mean() * 100.0),
                "n": int(vals.notna().sum()),
            })
    return pd.DataFrame(rows)


def make_clean_name(x):
    x = str(x)
    x = x.replace("_", " ")
    x = x.replace("probability", "prob.")
    x = x.replace("region ", "")
    x = x.replace("timepoint ", "")
    if len(x) > 24:
        x = x[:23] + "…"
    return x


def select_top_modules_for_maps(corr_df, scores, n=4):
    if not corr_df.empty:
        tmp = (
            corr_df.groupby("module")["spearman_rho"]
            .apply(lambda x: np.nanmax(np.abs(x)))
            .sort_values(ascending=False)
        )
        mods = [m for m in tmp.index if m in scores.columns]
        if mods:
            return mods[:n]
    return list(scores.columns[:n])


def make_figure(scores, meta, corr_df, dot_df, gene_sets, module_order, Z, outbase, clean=False, dpi=600, max_traits=12, top_map_modules=4):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    })

    ordered_modules = [m for m in module_order if m in scores.columns]
    traits = trait_order_by_signal(corr_df, max_traits=max_traits)

    corr_mat = pd.DataFrame(index=ordered_modules, columns=traits, dtype=float)
    fdr_mat = pd.DataFrame(index=ordered_modules, columns=traits, dtype=float)

    for _, r in corr_df.iterrows():
        if r["module"] in corr_mat.index and r["trait"] in corr_mat.columns:
            corr_mat.loc[r["module"], r["trait"]] = r["spearman_rho"]
            fdr_mat.loc[r["module"], r["trait"]] = r["bh_fdr"]

    xcol, ycol = find_xy_cols(meta)
    map_modules = select_top_modules_for_maps(corr_df, scores, n=top_map_modules)
    group_col = find_group_col(meta)

    if clean:
        fig = plt.figure(figsize=(12, 8), facecolor="white")
        gs = fig.add_gridspec(2, 2, hspace=0.24, wspace=0.25)
    else:
        fig = plt.figure(figsize=(17, 12), facecolor="white")
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[0.78, 1.0],
            width_ratios=[0.90, 1.25],
            hspace=0.30,
            wspace=0.26
        )
        fig.suptitle("WGCNA-style module evidence landscape", fontsize=18, fontweight="bold", y=0.98)

    # A. Dendrogram.
    axA = fig.add_subplot(gs[0, 0])
    if Z is not None and dendrogram is not None:
        dendrogram(
            Z,
            labels=[make_clean_name(m) for m in scores.columns],
            ax=axA,
            leaf_rotation=60,
            leaf_font_size=8,
            color_threshold=None,
            above_threshold_color="#334155"
        )
    else:
        axA.barh(np.arange(len(ordered_modules)), np.arange(len(ordered_modules)) + 1)
        axA.set_yticks(np.arange(len(ordered_modules)))
        axA.set_yticklabels([make_clean_name(m) for m in ordered_modules], fontsize=8)

    if not clean:
        axA.set_title("A  Module eigengene dendrogram", loc="left", fontsize=12, fontweight="bold")
        axA.set_ylabel("Module eigengene distance")
    else:
        axA.set_xticks([])
        axA.set_yticks([])
    for sp in ["top", "right"]:
        axA.spines[sp].set_visible(False)

    # B. Module-trait correlation heatmap.
    axB = fig.add_subplot(gs[0, 1])
    corr_values = corr_mat.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(corr_values)) if corr_values.size else 1
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1
    vmax = max(0.25, min(1.0, vmax))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    imB = axB.imshow(corr_values, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")

    if clean:
        axB.set_xticks([])
        axB.set_yticks([])
    else:
        axB.set_xticks(np.arange(len(traits)))
        axB.set_xticklabels([make_clean_name(t) for t in traits], rotation=45, ha="right", fontsize=8)
        axB.set_yticks(np.arange(len(ordered_modules)))
        axB.set_yticklabels([make_clean_name(m) for m in ordered_modules], fontsize=8)
        axB.set_title("B  Module-trait correlations", loc="left", fontsize=12, fontweight="bold")

        for i, m in enumerate(ordered_modules):
            for j, t in enumerate(traits):
                val = corr_mat.loc[m, t]
                fdr = fdr_mat.loc[m, t]
                if pd.notna(val):
                    star = ""
                    if pd.notna(fdr):
                        if fdr <= 0.05:
                            star = "**"
                        elif fdr <= 0.25:
                            star = "*"
                    axB.text(j, i, f"{val:+.2f}{star}", ha="center", va="center", fontsize=6.6, color="#111827")

        cb = fig.colorbar(imB, ax=axB, fraction=0.035, pad=0.02)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Spearman ρ", fontsize=8)

    for sp in axB.spines.values():
        sp.set_visible(False)

    # C. Spatial maps.
    axC_container = fig.add_subplot(gs[1, 0])
    axC_container.axis("off")
    if not clean:
        axC_container.set_title("C  Module score spatial maps", loc="left", fontsize=12, fontweight="bold", pad=8)

    sub = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 0], hspace=0.22, wspace=0.15)
    if xcol and ycol:
        plot_df = meta[["obs_name", xcol, ycol]].copy()
        for m in map_modules:
            plot_df[m] = scores[m].values

        x = pd.to_numeric(plot_df[xcol], errors="coerce")
        y = pd.to_numeric(plot_df[ycol], errors="coerce")
        ok_xy = x.notna() & y.notna()

        for k in range(4):
            ax = fig.add_subplot(sub[k // 2, k % 2])
            if k < len(map_modules):
                m = map_modules[k]
                vals = pd.to_numeric(plot_df[m], errors="coerce")
                vmax2 = np.nanquantile(np.abs(vals), 0.98)
                if not np.isfinite(vmax2) or vmax2 <= 0:
                    vmax2 = 1.0
                norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
                ax.scatter(
                    x[ok_xy], y[ok_xy],
                    c=vals[ok_xy],
                    s=4,
                    cmap="RdBu_r",
                    norm=norm2,
                    linewidths=0,
                    rasterized=True
                )
                ax.set_title(make_clean_name(m), fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            for sp in ax.spines.values():
                sp.set_visible(False)
    else:
        ax = fig.add_subplot(sub[:, :])
        ax.text(0.5, 0.5, "No spatial coordinates found", ha="center", va="center", fontsize=10)
        ax.axis("off")

    # D. Dotplot.
    axD = fig.add_subplot(gs[1, 1])
    if not dot_df.empty and group_col:
        groups = dot_df["group"].astype(str).value_counts().index.tolist()
        # order common stroke states/timepoints
        preferred = ["D1", "D3", "D7", "sham", "core", "peri", "remote", "lesion_core", "peri_infarct", "remote_like"]
        def gkey(g):
            gl = g.lower()
            for i, p in enumerate(preferred):
                if p.lower() in gl:
                    return (i, g)
            return (999, g)
        groups = sorted(groups, key=gkey)

        modules_d = ordered_modules
        mean_mat = pd.DataFrame(index=modules_d, columns=groups, dtype=float)
        pct_mat = pd.DataFrame(index=modules_d, columns=groups, dtype=float)

        for _, r in dot_df.iterrows():
            m = r["module"]
            g = str(r["group"])
            if m in mean_mat.index and g in mean_mat.columns:
                mean_mat.loc[m, g] = r["mean_score"]
                pct_mat.loc[m, g] = r["pct_positive"]

        xs, ys, cs, ss = [], [], [], []
        for i, m in enumerate(modules_d):
            for j, g in enumerate(groups):
                val = mean_mat.loc[m, g]
                pct = pct_mat.loc[m, g]
                if pd.notna(val):
                    xs.append(j)
                    ys.append(i)
                    cs.append(val)
                    ss.append(12 + float(pct) * 1.2)

        vmax3 = np.nanquantile(np.abs(cs), 0.95) if len(cs) else 1
        if not np.isfinite(vmax3) or vmax3 <= 0:
            vmax3 = 1
        norm3 = TwoSlopeNorm(vmin=-vmax3, vcenter=0, vmax=vmax3)
        sc = axD.scatter(xs, ys, c=cs, s=ss, cmap="RdBu_r", norm=norm3, edgecolor="#334155", linewidth=0.25)

        if clean:
            axD.set_xticks([])
            axD.set_yticks([])
        else:
            axD.set_xticks(np.arange(len(groups)))
            axD.set_xticklabels([make_clean_name(g) for g in groups], rotation=45, ha="right", fontsize=8)
            axD.set_yticks(np.arange(len(modules_d)))
            axD.set_yticklabels([make_clean_name(m) for m in modules_d], fontsize=8)
            axD.set_title("D  Module dotplot across states/timepoints", loc="left", fontsize=12, fontweight="bold")
            axD.set_xlabel(group_col)
            axD.set_ylabel("Module")
            cb = fig.colorbar(sc, ax=axD, fraction=0.035, pad=0.02)
            cb.ax.tick_params(labelsize=8)
            cb.set_label("mean module score", fontsize=8)

            # dot size legend
            for pct, ypos in zip([25, 50, 75], [0.12, 0.20, 0.28]):
                axD.scatter([1.05], [ypos], s=12 + pct * 1.2, transform=axD.transAxes,
                            facecolor="white", edgecolor="#334155", linewidth=0.4, clip_on=False)
                axD.text(1.10, ypos, f"{pct}% positive", transform=axD.transAxes,
                         va="center", fontsize=7, color="#475569")
    else:
        axD.text(0.5, 0.5, "No group column found for dotplot", ha="center", va="center")
        axD.axis("off")

    for sp in axD.spines.values():
        sp.set_visible(False)

    if not clean:
        fig.text(
            0.5,
            0.012,
            "WGCNA-style module evidence: module eigengene scores are derived from curated/Step65 gene sets. This visualization supports module-level consistency, not functional validation.",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color="#475569"
        )

    outbase = Path(outbase)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", default="")
    parser.add_argument("--module_gene_table", default="")
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--max_cells", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--min_matched_genes", type=int, default=3)
    parser.add_argument("--max_traits", type=int, default=12)
    parser.add_argument("--top_map_modules", type=int, default=4)
    parser.add_argument("--save_cell_scores", action="store_true")
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    if ad is None:
        raise ImportError("anndata is required. Please run inside nicheformer_env.")
    if spearmanr is None:
        raise ImportError("scipy is required.")

    outdir = ensure_dir(args.outdir)

    log("=" * 100)
    log("Step74 | WGCNA-style module evidence figure")
    log("=" * 100)
    log(f"h5ad={args.h5ad}")
    log(f"state_table={args.state_table}")
    log(f"module_gene_table={args.module_gene_table}")
    log(f"outdir={outdir}")

    module_sets, module_audit = build_module_sets(args.module_gene_table, min_genes=args.min_matched_genes)

    adata = ad.read_h5ad(args.h5ad)
    adata, sampled_idx = subset_adata_cells(adata, max_cells=args.max_cells, seed=args.seed)

    state_table = read_table(args.state_table) if args.state_table else pd.DataFrame()

    meta, merge_audit = merge_metadata(adata, state_table)

    scores, gene_sets_used = compute_module_scores(
        adata,
        module_sets,
        min_matched_genes=args.min_matched_genes
    )

    if scores.empty:
        fail = {
            "status": "failed_no_module_scores",
            "module_audit": module_audit,
            "gene_sets_used": gene_sets_used.to_dict(orient="records"),
            "message": "No modules had enough matched genes in h5ad."
        }
        (outdir / "step74_failed_audit_report.json").write_text(json.dumps(fail, indent=2), encoding="utf-8")
        raise RuntimeError(fail["message"])

    # Align meta to scores.
    meta = meta.iloc[:scores.shape[0]].copy()
    meta.index = scores.index

    traits = numeric_trait_table(meta)
    corr_df = module_trait_correlations(scores, traits) if not traits.empty else pd.DataFrame()

    module_order, Z = module_order_by_dendrogram(scores)

    group_col = find_group_col(meta)
    dot_df = summarize_dotplot(scores, meta, group_col) if group_col else pd.DataFrame()

    xcol, ycol = find_xy_cols(meta)

    # Save outputs.
    gene_sets_path = outdir / "step74_module_gene_sets_used.csv"
    corr_path = outdir / "step74_module_trait_correlations.csv"
    dot_path = outdir / "step74_module_dotplot_summary.csv"
    meta_path = outdir / "step74_metadata_used_for_traits.csv"
    scores_summary_path = outdir / "step74_module_score_summary.csv"
    spatial_cells_path = outdir / "step74_module_spatial_plot_cells.csv"

    gene_sets_used.to_csv(gene_sets_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    dot_df.to_csv(dot_path, index=False)
    meta.to_csv(meta_path, index=False)

    score_summary = scores.describe().T.reset_index().rename(columns={"index": "module"})
    score_summary["n_matched_genes"] = score_summary["module"].map(
        dict(zip(gene_sets_used["module"], gene_sets_used["n_matched_genes"]))
    )
    score_summary.to_csv(scores_summary_path, index=False)

    if args.save_cell_scores:
        scores.to_csv(outdir / "step74_module_scores_by_observation.csv")

    spatial_cols = ["obs_name"]
    if xcol and ycol:
        spatial_cols += [xcol, ycol]
    spatial_df = meta[[c for c in spatial_cols if c in meta.columns]].copy()
    for m in scores.columns:
        spatial_df[m] = scores[m].values
    spatial_df.to_csv(spatial_cells_path, index=False)

    # Figures.
    annotated_base = outdir / "Fig_Step74_WGCNAStyleModuleEvidence_annotated"
    clean_base = outdir / "Fig_Step74_WGCNAStyleModuleEvidence_clean_no_text"

    make_figure(
        scores=scores,
        meta=meta,
        corr_df=corr_df,
        dot_df=dot_df,
        gene_sets=gene_sets_used,
        module_order=module_order,
        Z=Z,
        outbase=annotated_base,
        clean=False,
        dpi=args.dpi,
        max_traits=args.max_traits,
        top_map_modules=args.top_map_modules,
    )

    make_figure(
        scores=scores,
        meta=meta,
        corr_df=corr_df,
        dot_df=dot_df,
        gene_sets=gene_sets_used,
        module_order=module_order,
        Z=Z,
        outbase=clean_base,
        clean=True,
        dpi=args.dpi,
        max_traits=args.max_traits,
        top_map_modules=args.top_map_modules,
    )

    audit = {
        "status": "ok",
        "h5ad": args.h5ad,
        "n_obs_used": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "max_cells": args.max_cells,
        "module_audit": module_audit,
        "merge_audit": merge_audit,
        "n_modules_scored": int(scores.shape[1]),
        "modules_scored": list(scores.columns),
        "n_traits": int(traits.shape[1]) if not traits.empty else 0,
        "traits": list(traits.columns) if not traits.empty else [],
        "group_col_for_dotplot": group_col,
        "spatial_x_col": xcol,
        "spatial_y_col": ycol,
        "module_order": module_order,
    }

    (outdir / "step74_audit_report.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        "status": "ok",
        "analysis_name": "Step74 WGCNA-style module evidence figure",
        "outputs": {
            "module_gene_sets_used": str(gene_sets_path),
            "module_trait_correlations": str(corr_path),
            "module_dotplot_summary": str(dot_path),
            "module_score_summary": str(scores_summary_path),
            "metadata_used_for_traits": str(meta_path),
            "module_spatial_plot_cells": str(spatial_cells_path),
            "audit_report": str(outdir / "step74_audit_report.json"),
            "annotated_pdf": str(annotated_base.with_suffix(".pdf")),
            "annotated_svg": str(annotated_base.with_suffix(".svg")),
            "annotated_png": str(annotated_base.with_suffix(".png")),
            "clean_pdf": str(clean_base.with_suffix(".pdf")),
            "clean_svg": str(clean_base.with_suffix(".svg")),
            "clean_png": str(clean_base.with_suffix(".png")),
        },
        "interpretation_note": (
            "Step74 provides WGCNA-style module evidence using curated/Step65-derived module gene sets, "
            "module eigengene-like scores, module-trait correlations, spatial maps and dotplots. "
            "This is module-level computational evidence, not full R-WGCNA TOM analysis or functional validation."
        ),
    }

    (outdir / "step74_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step74_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 100)
    log("DONE Step74")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
