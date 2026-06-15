#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step74B | Polished WGCNA-style module evidence figure

Input
-----
Step74 output directory containing:
  step74_module_trait_correlations.csv
  step74_module_dotplot_summary.csv
  step74_module_spatial_plot_cells.csv
  step74_module_gene_sets_used.csv
  step74_audit_report.json

Purpose
-------
Improve Step74 figure aesthetics:
  A. compact module dendrogram
  B. non-redundant module-trait correlation heatmap
  C. clean module score spatial maps
  D. compact dotplot across state/time groups

Interpretation
--------------
WGCNA-style module evidence based on module eigengene-like scores derived from
curated/Step65 gene sets. This is not a full R-WGCNA TOM analysis and not
functional validation.
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
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import squareform
except Exception:
    linkage = None
    dendrogram = None
    leaves_list = None
    squareform = None


MODULE_COLORS = {
    "Synaptic_recovery": "#fccde5",
    "Hypoxia_redox": "#8dd3c7",
    "Astrocyte_reactive": "#ccebc5",
    "Ferroptosis": "#fb8072",
    "Inflammation_chemotaxis": "#fdb462",
    "Microglia_myeloid": "#b3de69",
    "BBB_endothelial": "#80b1d3",
    "Repair_ECM": "#bebada",
}


MODULE_NAME_MAP = {
    "Synaptic_recovery": "Synaptic\nrecovery",
    "Hypoxia_redox": "Hypoxia/\nredox",
    "Astrocyte_reactive": "Reactive\nastrocyte",
    "Ferroptosis": "Ferroptosis",
    "Inflammation_chemotaxis": "Inflammation/\nchemotaxis",
    "Microglia_myeloid": "Microglia/\nmyeloid",
    "BBB_endothelial": "BBB/\nendothelial",
    "Repair_ECM": "Repair/\nECM",
}


TRAIT_PRIORITY = [
    "repair_score",
    "core_probability",
    "peri_probability",
    "remote_probability",
    "state_group_lesion-core-like",
    "state_group_peri-infarct",
    "state_group_remote-like",
    "region_manual_final_lesion_core",
    "region_manual_final_peri_infarct",
    "region_manual_final_remote_like",
    "stage_numeric",
]


TRAIT_LABEL_MAP = {
    "repair_score": "Repair\nscore",
    "core_probability": "Core\nprob.",
    "peri_probability": "Peri\nprob.",
    "remote_probability": "Remote\nprob.",
    "state_group_lesion-core-like": "Core-like",
    "state_group_peri-infarct": "Peri-\ninfarct",
    "state_group_remote-like": "Remote-\nlike",
    "region_manual_final_lesion_core": "Manual\ncore",
    "region_manual_final_peri_infarct": "Manual\nperi",
    "region_manual_final_remote_like": "Manual\nremote",
    "stage_numeric": "Stage",
}


def log(x):
    print(x, flush=True)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def nice_module(x):
    x = str(x)
    return MODULE_NAME_MAP.get(x, x.replace("_", "\n"))


def nice_trait(x):
    x = str(x)
    if x in TRAIT_LABEL_MAP:
        return TRAIT_LABEL_MAP[x]
    x = x.replace("probability", "prob.")
    x = x.replace("state_group_", "")
    x = x.replace("region_manual_final_", "")
    x = x.replace("region_refined_", "")
    x = x.replace("region_auto_", "")
    x = x.replace("_", "\n")
    if len(x) > 18:
        x = x[:17] + "…"
    return x


def select_nonredundant_traits(corr_df, max_traits=8):
    if corr_df.empty:
        return []

    available = list(corr_df["trait"].dropna().astype(str).unique())

    selected = []
    for t in TRAIT_PRIORITY:
        if t in available and t not in selected:
            selected.append(t)
        if len(selected) >= max_traits:
            return selected

    # add strongest remaining traits while reducing duplicates
    strength = (
        corr_df.groupby("trait")["spearman_rho"]
        .apply(lambda x: np.nanmax(np.abs(pd.to_numeric(x, errors="coerce"))))
        .sort_values(ascending=False)
    )

    blocked_keywords = []
    for t in selected:
        low = t.lower()
        if "core" in low:
            blocked_keywords.append("core")
        if "peri" in low:
            blocked_keywords.append("peri")
        if "remote" in low:
            blocked_keywords.append("remote")

    for t in strength.index:
        if t in selected:
            continue
        low = str(t).lower()

        # Avoid too many duplicate core/peri/remote variants.
        if any(k in low for k in ["core", "peri", "remote"]):
            key = "core" if "core" in low else "peri" if "peri" in low else "remote"
            if blocked_keywords.count(key) >= 2:
                continue
            blocked_keywords.append(key)

        selected.append(t)
        if len(selected) >= max_traits:
            break

    return selected


def infer_module_order(corr_df, spatial_df=None):
    modules = list(corr_df["module"].dropna().astype(str).unique()) if not corr_df.empty else []
    if not modules and spatial_df is not None and not spatial_df.empty:
        modules = [c for c in spatial_df.columns if c not in {"obs_name", "x", "y", "spatial_x", "spatial_y"}]

    if not modules:
        return [], None

    # Prefer module correlation based on spatial/cell scores if available.
    score_cols = []
    if spatial_df is not None and not spatial_df.empty:
        for m in modules:
            if m in spatial_df.columns:
                vals = pd.to_numeric(spatial_df[m], errors="coerce")
                if vals.notna().sum() > 10:
                    score_cols.append(m)

    if len(score_cols) >= 3 and linkage is not None:
        X = spatial_df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        corr = X.corr(method="spearman").fillna(0)
        dist = 1 - corr
        dist = dist.clip(lower=0, upper=2)
        np.fill_diagonal(dist.values, 0)
        Z = linkage(squareform(dist.values, checks=False), method="average")
        order = leaves_list(Z)
        ordered = [score_cols[i] for i in order]
        return ordered, Z

    # Fallback: order by similarity of trait correlations.
    if not corr_df.empty and linkage is not None:
        traits = list(corr_df["trait"].dropna().astype(str).unique())
        mat = pd.DataFrame(index=modules, columns=traits, dtype=float)
        for _, r in corr_df.iterrows():
            mat.loc[str(r["module"]), str(r["trait"])] = pd.to_numeric(r["spearman_rho"], errors="coerce")
        mat = mat.fillna(0)
        if mat.shape[0] >= 3:
            corr = mat.T.corr(method="spearman").fillna(0)
            dist = 1 - corr
            dist = dist.clip(lower=0, upper=2)
            np.fill_diagonal(dist.values, 0)
            Z = linkage(squareform(dist.values, checks=False), method="average")
            order = leaves_list(Z)
            ordered = [modules[i] for i in order]
            return ordered, Z

    return modules, None


def find_xy_cols(spatial_df):
    candidates = [
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
        ("coord_x", "coord_y"),
        ("array_col", "array_row"),
        ("baseline_coord_x_from_ref", "baseline_coord_y_from_ref"),
        ("UMAP_1", "UMAP_2"),
        ("umap_1", "umap_2"),
    ]
    for x, y in candidates:
        if x in spatial_df.columns and y in spatial_df.columns:
            xx = pd.to_numeric(spatial_df[x], errors="coerce")
            yy = pd.to_numeric(spatial_df[y], errors="coerce")
            if xx.notna().sum() > 10 and yy.notna().sum() > 10:
                return x, y

    numeric_cols = []
    for c in spatial_df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["x", "y", "coord", "spatial", "umap", "array"]):
            vals = pd.to_numeric(spatial_df[c], errors="coerce")
            if vals.notna().sum() > 10:
                numeric_cols.append(c)
    if len(numeric_cols) >= 2:
        return numeric_cols[0], numeric_cols[1]

    return "", ""


def select_map_modules(corr_df, module_order, top_n=4):
    preferred = ["Microglia_myeloid", "Repair_ECM", "Astrocyte_reactive", "Ferroptosis", "BBB_endothelial"]
    selected = [m for m in preferred if m in module_order]

    if len(selected) < top_n and not corr_df.empty:
        strength = (
            corr_df.groupby("module")["spearman_rho"]
            .apply(lambda x: np.nanmax(np.abs(pd.to_numeric(x, errors="coerce"))))
            .sort_values(ascending=False)
        )
        for m in strength.index:
            if m in module_order and m not in selected:
                selected.append(m)
            if len(selected) >= top_n:
                break

    for m in module_order:
        if m not in selected:
            selected.append(m)
        if len(selected) >= top_n:
            break

    return selected[:top_n]


def make_corr_matrix(corr_df, modules, traits):
    mat = pd.DataFrame(index=modules, columns=traits, dtype=float)
    fdr = pd.DataFrame(index=modules, columns=traits, dtype=float)

    for _, r in corr_df.iterrows():
        m = str(r["module"])
        t = str(r["trait"])
        if m in mat.index and t in mat.columns:
            mat.loc[m, t] = pd.to_numeric(r["spearman_rho"], errors="coerce")
            fdr.loc[m, t] = pd.to_numeric(r.get("bh_fdr", np.nan), errors="coerce")

    return mat, fdr


def compact_dotplot_data(dot_df, modules):
    if dot_df.empty:
        return pd.DataFrame(), []

    group_col = "group"
    module_col = "module"

    groups = list(dot_df[group_col].dropna().astype(str).unique())

    preferred_order = [
        "lesion-core-like", "peri-infarct", "remote-like",
        "core", "peri", "remote",
        "D1", "D3", "D7", "sham"
    ]

    def gkey(g):
        gl = g.lower()
        for i, p in enumerate(preferred_order):
            if p.lower() == gl or p.lower() in gl:
                return (i, g)
        return (999, g)

    groups = sorted(groups, key=gkey)
    modules = [m for m in modules if m in dot_df[module_col].astype(str).unique()]
    return dot_df, groups


def draw_dotplot(ax, dot_df, modules, groups):
    if dot_df.empty or not modules or not groups:
        ax.text(0.5, 0.5, "No dotplot summary", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return None

    mean_col = "mean_score"
    pct_col = "pct_positive"

    xs, ys, cs, ss = [], [], [], []
    for i, m in enumerate(modules):
        for j, g in enumerate(groups):
            sub = dot_df[(dot_df["module"].astype(str) == m) & (dot_df["group"].astype(str) == g)]
            if sub.empty:
                continue
            val = pd.to_numeric(sub[mean_col].iloc[0], errors="coerce")
            pct = pd.to_numeric(sub[pct_col].iloc[0], errors="coerce")
            if pd.isna(val):
                continue
            if pd.isna(pct):
                pct = 0
            xs.append(j)
            ys.append(i)
            cs.append(float(val))
            ss.append(18 + float(pct) * 1.25)

    vmax = np.nanquantile(np.abs(cs), 0.95) if cs else 1
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    sc = ax.scatter(xs, ys, c=cs, s=ss, cmap="RdBu_r", norm=norm,
                    edgecolor="#334155", linewidth=0.35)

    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels([nice_trait(g) for g in groups], rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(modules)))
    ax.set_yticklabels([nice_module(m) for m in modules], fontsize=8.5)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.5, len(modules) - 0.5)
    ax.invert_yaxis()
    ax.grid(True, axis="both", linestyle=":", linewidth=0.4, color="#cbd5e1", alpha=0.6)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sc


def make_figure(step74_dir, outdir, max_traits=8, top_map_modules=4, dpi=600, clean=False):
    step74_dir = Path(step74_dir)
    outdir = Path(outdir)

    corr_df = read_csv(step74_dir / "step74_module_trait_correlations.csv", required=True)
    dot_df = read_csv(step74_dir / "step74_module_dotplot_summary.csv", required=False)
    spatial_df = read_csv(step74_dir / "step74_module_spatial_plot_cells.csv", required=False)
    gene_sets = read_csv(step74_dir / "step74_module_gene_sets_used.csv", required=False)
    audit = read_json(step74_dir / "step74_audit_report.json")

    module_order, Z = infer_module_order(corr_df, spatial_df)
    if not module_order:
        raise RuntimeError("No modules found from Step74 outputs.")

    traits = select_nonredundant_traits(corr_df, max_traits=max_traits)
    corr_mat, fdr_mat = make_corr_matrix(corr_df, module_order, traits)

    map_modules = select_map_modules(corr_df, module_order, top_n=top_map_modules)

    dot_df2, groups = compact_dotplot_data(dot_df, module_order)
    dot_modules = [m for m in module_order if m in dot_df2["module"].astype(str).unique()] if not dot_df2.empty else module_order

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    })

    if clean:
        fig = plt.figure(figsize=(13, 8), facecolor="white")
        gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.30)
    else:
        fig = plt.figure(figsize=(17, 11.5), facecolor="white")
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            height_ratios=[0.92, 1.15],
            width_ratios=[0.88, 1.25],
            hspace=0.36,
            wspace=0.30
        )
        fig.suptitle("WGCNA-style module evidence landscape",
                     fontsize=18, fontweight="bold", y=0.985)

    # Panel A: dendrogram.
    axA = fig.add_subplot(gs[0, 0])
    if Z is not None and dendrogram is not None:
        dendrogram(
            Z,
            labels=[nice_module(m).replace("\n", " ") for m in module_order],
            leaf_rotation=45,
            leaf_font_size=8.3,
            color_threshold=None,
            above_threshold_color="#334155",
            ax=axA,
        )
    else:
        y = np.arange(len(module_order))
        axA.barh(y, np.arange(len(module_order)) + 1, color="#94a3b8")
        axA.set_yticks(y)
        axA.set_yticklabels([nice_module(m).replace("\n", " ") for m in module_order], fontsize=8)

    if not clean:
        axA.set_title("A  Module eigengene dendrogram",
                      loc="left", fontsize=12, fontweight="bold", pad=8)
        axA.set_ylabel("Module eigengene distance", fontsize=9)
    else:
        axA.set_xticks([])
        axA.set_yticks([])
    for sp in ["top", "right"]:
        axA.spines[sp].set_visible(False)

    # Panel B: correlation heatmap.
    axB = fig.add_subplot(gs[0, 1])
    corr_values = corr_mat.to_numpy(dtype=float)
    vmax = np.nanquantile(np.abs(corr_values), 0.98) if corr_values.size else 1
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1
    vmax = min(1.0, max(0.25, vmax))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = axB.imshow(corr_values, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")

    if clean:
        axB.set_xticks([])
        axB.set_yticks([])
    else:
        axB.set_xticks(np.arange(len(traits)))
        axB.set_xticklabels([nice_trait(t) for t in traits],
                            rotation=35, ha="right", fontsize=8.2)
        axB.set_yticks(np.arange(len(module_order)))
        axB.set_yticklabels([nice_module(m) for m in module_order], fontsize=8.5)
        axB.set_title("B  Module-trait correlation",
                      loc="left", fontsize=12, fontweight="bold", pad=8)

        for i, m in enumerate(module_order):
            for j, t in enumerate(traits):
                val = corr_mat.loc[m, t]
                fdr = fdr_mat.loc[m, t]
                if pd.isna(val):
                    continue
                star = ""
                if pd.notna(fdr):
                    if fdr <= 0.05:
                        star = "**"
                    elif fdr <= 0.25:
                        star = "*"
                txt_color = "white" if abs(float(val)) > 0.48 else "#111827"
                axB.text(j, i, f"{float(val):+.2f}{star}",
                         ha="center", va="center", fontsize=7.0, color=txt_color)

        cb = fig.colorbar(im, ax=axB, fraction=0.035, pad=0.018)
        cb.ax.tick_params(labelsize=8)
        cb.set_label("Spearman ρ", fontsize=8.5)

    for sp in axB.spines.values():
        sp.set_visible(False)

    # Panel C: spatial maps.
    containerC = fig.add_subplot(gs[1, 0])
    containerC.axis("off")
    if not clean:
        containerC.set_title("C  Module-score spatial maps",
                             loc="left", fontsize=12, fontweight="bold", pad=10)

    subC = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, 0], hspace=0.20, wspace=0.12)

    xcol, ycol = find_xy_cols(spatial_df) if not spatial_df.empty else ("", "")

    if xcol and ycol:
        x = pd.to_numeric(spatial_df[xcol], errors="coerce")
        y = pd.to_numeric(spatial_df[ycol], errors="coerce")
        ok = x.notna() & y.notna()

        for k in range(4):
            ax = fig.add_subplot(subC[k // 2, k % 2])
            if k < len(map_modules) and map_modules[k] in spatial_df.columns:
                m = map_modules[k]
                vals = pd.to_numeric(spatial_df[m], errors="coerce")
                vmax2 = np.nanquantile(np.abs(vals), 0.985)
                if not np.isfinite(vmax2) or vmax2 <= 0:
                    vmax2 = 1
                norm2 = TwoSlopeNorm(vmin=-vmax2, vcenter=0, vmax=vmax2)
                ax.scatter(x[ok], y[ok], c=vals[ok], s=3.6,
                           cmap="RdBu_r", norm=norm2, linewidths=0, rasterized=True)
                if not clean:
                    ax.set_title(nice_module(m).replace("\n", " "), fontsize=8.8, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")
            for sp in ax.spines.values():
                sp.set_visible(False)
    else:
        ax = fig.add_subplot(subC[:, :])
        ax.text(0.5, 0.5, "No spatial coordinates found",
                ha="center", va="center", fontsize=10)
        ax.axis("off")

    # Panel D: compact dotplot.
    axD = fig.add_subplot(gs[1, 1])
    sc = draw_dotplot(axD, dot_df2, dot_modules, groups)
    if not clean:
        axD.set_title("D  Module activity across states/timepoints",
                      loc="left", fontsize=12, fontweight="bold", pad=8)
        axD.set_xlabel("State / timepoint group", fontsize=9)
        axD.set_ylabel("Module", fontsize=9)

        if sc is not None:
            cb = fig.colorbar(sc, ax=axD, fraction=0.035, pad=0.018)
            cb.ax.tick_params(labelsize=8)
            cb.set_label("Mean module score", fontsize=8.5)

            # Size legend inside D panel.
            for pct, ypos in zip([25, 50, 75], [0.12, 0.20, 0.28]):
                axD.scatter([1.04], [ypos], s=18 + pct * 1.25,
                            transform=axD.transAxes, facecolor="white",
                            edgecolor="#334155", linewidth=0.4, clip_on=False)
                axD.text(1.095, ypos, f"{pct}% positive",
                         transform=axD.transAxes, va="center",
                         fontsize=7.2, color="#475569")

    if not clean:
        fig.text(
            0.5, 0.012,
            "WGCNA-style module evidence: module eigengene-like scores are derived from curated/Step65 gene sets; this supports module-level consistency, not full TOM-WGCNA or functional validation.",
            ha="center", va="bottom", fontsize=8.5, color="#475569"
        )

    suffix = "clean_no_text" if clean else "annotated"
    outbase = outdir / f"Fig_Step74B_PolishedWGCNAStyleModuleEvidence_{suffix}"
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    report_part = {
        "module_order": module_order,
        "selected_traits": traits,
        "selected_spatial_map_modules": map_modules,
        "dotplot_groups": groups,
        "dotplot_modules": dot_modules,
        "xcol": xcol,
        "ycol": ycol,
        "figure": {
            "pdf": str(outbase.with_suffix(".pdf")),
            "svg": str(outbase.with_suffix(".svg")),
            "png": str(outbase.with_suffix(".png")),
        }
    }

    return report_part


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--step74_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_traits", type=int, default=8)
    ap.add_argument("--top_map_modules", type=int, default=4)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    log("=" * 100)
    log("Step74B | Polished WGCNA-style module evidence figure")
    log("=" * 100)
    log(f"step74_dir={args.step74_dir}")
    log(f"outdir={outdir}")

    annotated = make_figure(
        step74_dir=args.step74_dir,
        outdir=outdir,
        max_traits=args.max_traits,
        top_map_modules=args.top_map_modules,
        dpi=args.dpi,
        clean=False,
    )

    clean = make_figure(
        step74_dir=args.step74_dir,
        outdir=outdir,
        max_traits=args.max_traits,
        top_map_modules=args.top_map_modules,
        dpi=args.dpi,
        clean=True,
    )

    report = {
        "status": "ok",
        "analysis_name": "Step74B polished WGCNA-style module evidence figure",
        "input_step74_dir": args.step74_dir,
        "outputs": {
            "annotated": annotated["figure"],
            "clean_no_text": clean["figure"],
            "report_json": str(outdir / "step74b_report.json"),
            "report_txt": str(outdir / "step74b_report.txt"),
        },
        "selected_content": {
            "module_order": annotated["module_order"],
            "selected_traits": annotated["selected_traits"],
            "selected_spatial_map_modules": annotated["selected_spatial_map_modules"],
            "dotplot_groups": annotated["dotplot_groups"],
            "dotplot_modules": annotated["dotplot_modules"],
            "spatial_x_col": annotated["xcol"],
            "spatial_y_col": annotated["ycol"],
        },
        "interpretation_note": (
            "Step74B is a polished visualization of Step74 WGCNA-style module evidence. "
            "It improves layout and readability but remains module-level computational evidence, "
            "not full WGCNA TOM analysis and not functional validation."
        )
    }

    (outdir / "step74b_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step74b_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 100)
    log("DONE Step74B")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
