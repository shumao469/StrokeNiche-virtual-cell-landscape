#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step73B | Compact manuscript-ready track-specific dynamic marker heatmap

Input:
  Step73 outputs:
    step73_track_dynamic_feature_matrix.csv
    step73_track_dynamic_feature_raw_matrix.csv
    step73_selected_dynamic_features_long.csv
    step73_track_top_regulators.csv
    step73_feature_selection_summary.csv

Output:
  Compact manuscript-ready heatmap:
    rows = representative tracks
    columns = top dynamic genes/modules
    right side = top regulators
    left side = dominant biological program

Interpretation:
  Transcript-level dynamic marker/module/regulator prioritization.
  Not protein validation and not functional perturbation validation.
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
from matplotlib.colors import TwoSlopeNorm, ListedColormap
from matplotlib.patches import Rectangle


PROGRAM_RULES = [
    ("Hypoxia/redox", ["hypoxia", "hif", "hmox", "nfe2l2", "redox", "oxid"]),
    ("Ferroptosis", ["ferroptosis", "acsl4", "gpx4", "tfrc", "slc7a11", "ptgs2", "fth", "ftl"]),
    ("BBB/endothelial", ["bbb", "barrier", "endothelial", "pecam", "kdr", "flt1", "vwf", "claudin", "occludin"]),
    ("Inflammation/chemotaxis", ["inflamm", "chemotaxis", "ccl", "cxcl", "tnf", "il1", "microglia"]),
    ("Astrocyte/reactive", ["astro", "reactive", "gfap", "vim", "serpina", "stat3"]),
    ("Repair/ECM", ["repair", "ecm", "collagen", "col1", "col3", "fn1", "sparc", "mmp", "timp", "postn"]),
    ("Synaptic/recovery", ["synaptic", "recovery", "neuron", "snap", "syt", "grin", "map2", "rbfox"]),
]

PROGRAM_COLORS = {
    "Hypoxia/redox": "#8dd3c7",
    "Ferroptosis": "#fb8072",
    "BBB/endothelial": "#80b1d3",
    "Inflammation/chemotaxis": "#fdb462",
    "Astrocyte/reactive": "#b3de69",
    "Repair/ECM": "#bebada",
    "Synaptic/recovery": "#fccde5",
    "Other": "#d9d9d9",
}

FEATURE_TYPE_COLORS = {
    "gene": "#cbd5e1",
    "module": "#334155",
    "unknown": "#e5e7eb",
}


def log(x):
    print(x, flush=True)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path, required=False, index_col=None):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    return pd.read_csv(path, index_col=index_col, low_memory=False)


def clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def short_text(x, n=28):
    x = clean_text(x)
    if len(x) <= n:
        return x
    return x[: n - 1] + "…"


def natural_track_sort_key(x):
    s = str(x)
    m = re.search(r"(\d+)", s)
    if m:
        return (0, int(m.group(1)), s)
    return (1, 999999, s)


def infer_feature_type(name, feature_selection=None, long_df=None):
    name = str(name)
    if feature_selection is not None and not feature_selection.empty:
        if "feature_name" in feature_selection.columns and "feature_type" in feature_selection.columns:
            hit = feature_selection[feature_selection["feature_name"].astype(str) == name]
            if not hit.empty:
                return str(hit["feature_type"].iloc[0])

    if long_df is not None and not long_df.empty:
        if "feature_name" in long_df.columns and "feature_type" in long_df.columns:
            hit = long_df[long_df["feature_name"].astype(str) == name]
            if not hit.empty:
                return str(hit["feature_type"].iloc[0])

    low = name.lower()
    if any(k in low for _, keys in PROGRAM_RULES for k in keys):
        if " " in name or "_" in name or len(name) > 12:
            return "module"
    return "gene"


def feature_program(name):
    low = str(name).lower()
    for program, keys in PROGRAM_RULES:
        if any(k in low for k in keys):
            return program
    return "Other"


def track_dominant_program(row):
    scores = {p: 0.0 for p, _ in PROGRAM_RULES}
    scores["Other"] = 0.0

    for feature, val in row.items():
        p = feature_program(feature)
        scores[p] += abs(float(val)) if np.isfinite(val) else 0.0

    best = max(scores.items(), key=lambda x: x[1])
    if best[1] <= 0:
        return "Other"
    return best[0]


def compact_regulators(reg_text, max_n=3, max_len=34):
    reg_text = clean_text(reg_text)
    if not reg_text:
        return "—"

    parts = []
    for x in re.split(r"[,;/|]+", reg_text):
        x = clean_text(x)
        x = re.sub(r"[^A-Za-z0-9_.-]+", "", x)
        if x:
            parts.append(x.upper())

    parts = list(dict.fromkeys(parts))[:max_n]
    out = ", ".join(parts) if parts else "—"
    return short_text(out, max_len)


def select_representative_tracks(mat, max_tracks=14, min_nonzero_features=2):
    mat = mat.copy()
    mat = mat.loc[:, mat.abs().sum(axis=0) > 0]
    mat = mat.loc[mat.abs().sum(axis=1) > 0, :]

    if mat.empty:
        return []

    row_abs = mat.abs()
    row_score = (
        row_abs.max(axis=1) * 1.0 +
        row_abs.mean(axis=1) * 0.35 +
        (row_abs > 0).sum(axis=1) * 0.02
    )

    anno = pd.DataFrame({
        "track_id": mat.index,
        "row_score": row_score.values,
        "n_nonzero_features": (row_abs > 0).sum(axis=1).values,
        "dominant_program": [track_dominant_program(mat.loc[t]) for t in mat.index],
    }).set_index("track_id")

    anno = anno[anno["n_nonzero_features"] >= min_nonzero_features]
    if anno.empty:
        anno = pd.DataFrame({
            "track_id": mat.index,
            "row_score": row_score.values,
            "n_nonzero_features": (row_abs > 0).sum(axis=1).values,
            "dominant_program": [track_dominant_program(mat.loc[t]) for t in mat.index],
        }).set_index("track_id")

    selected = []

    # Keep top track per biological program first.
    for program, _ in PROGRAM_RULES:
        sub = anno[anno["dominant_program"] == program]
        if not sub.empty:
            selected.append(sub.sort_values("row_score", ascending=False).index[0])

    # Fill by strongest remaining tracks.
    for t in anno.sort_values("row_score", ascending=False).index:
        if t not in selected:
            selected.append(t)
        if len(selected) >= max_tracks:
            break

    selected = selected[:max_tracks]

    # Order by dominant program, then natural track number.
    program_order = {p: i for i, (p, _) in enumerate(PROGRAM_RULES)}
    program_order["Other"] = 999

    selected = sorted(
        selected,
        key=lambda t: (
            program_order.get(anno.loc[t, "dominant_program"], 999),
            -anno.loc[t, "row_score"],
            natural_track_sort_key(t)
        )
    )

    return selected


def select_compact_features(mat, long_df, feature_selection, max_genes=28, max_modules=10):
    features = list(mat.columns)

    type_map = {}
    program_map = {}

    for f in features:
        ft = infer_feature_type(f, feature_selection=feature_selection, long_df=long_df)
        type_map[f] = ft if ft in ["gene", "module"] else "unknown"
        program_map[f] = feature_program(f)

    scores = []
    for f in features:
        vals = mat[f].to_numpy(dtype=float)
        max_abs = np.nanmax(np.abs(vals))
        mean_abs = np.nanmean(np.abs(vals))
        prevalence = np.sum(np.abs(vals) > 0)
        scores.append({
            "feature_name": f,
            "feature_type": type_map[f],
            "program": program_map[f],
            "score": max_abs + 0.35 * mean_abs + 0.03 * prevalence,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "n_tracks_nonzero": prevalence,
        })

    score_df = pd.DataFrame(scores)
    score_df = score_df.sort_values("score", ascending=False)

    selected = []

    # Select modules by program priority.
    modules = score_df[score_df["feature_type"] == "module"].copy()
    if not modules.empty:
        program_order = {p: i for i, (p, _) in enumerate(PROGRAM_RULES)}
        modules["_porder"] = modules["program"].map(program_order).fillna(999)
        modules = modules.sort_values(["_porder", "score"], ascending=[True, False])
        selected += modules["feature_name"].head(max_modules).tolist()

    # Select top genes.
    genes = score_df[score_df["feature_type"] != "module"].copy()
    selected += genes["feature_name"].head(max_genes).tolist()

    selected = list(dict.fromkeys(selected))

    # Sort final features by program, module first, then score.
    score_lookup = dict(zip(score_df["feature_name"], score_df["score"]))
    program_order = {p: i for i, (p, _) in enumerate(PROGRAM_RULES)}
    program_order["Other"] = 999

    selected = sorted(
        selected,
        key=lambda f: (
            program_order.get(program_map.get(f, "Other"), 999),
            0 if type_map.get(f) == "module" else 1,
            -score_lookup.get(f, 0),
            str(f)
        )
    )

    feature_anno = score_df[score_df["feature_name"].isin(selected)].copy()
    feature_anno["selected_order"] = feature_anno["feature_name"].map({f: i for i, f in enumerate(selected)})
    feature_anno = feature_anno.sort_values("selected_order").drop(columns=["selected_order"])

    return selected, feature_anno, type_map, program_map


def build_row_annotation(mat, regulators):
    rows = []

    reg_map = {}
    if not regulators.empty and "track_id" in regulators.columns:
        reg_col = "top_regulators" if "top_regulators" in regulators.columns else regulators.columns[-1]
        reg_map = dict(zip(regulators["track_id"].astype(str), regulators[reg_col].astype(str)))

    for t in mat.index:
        dom = track_dominant_program(mat.loc[t])
        row_abs = mat.loc[t].abs()
        top_features = row_abs.sort_values(ascending=False).head(3).index.tolist()
        rows.append({
            "track_id": t,
            "dominant_program": dom,
            "top_features": ";".join(top_features),
            "top_regulators_compact": compact_regulators(reg_map.get(t, ""), max_n=3, max_len=36),
            "row_score": float(row_abs.max() + 0.35 * row_abs.mean() + 0.03 * (row_abs > 0).sum())
        })

    return pd.DataFrame(rows)


def prepare_matrix(step73_dir):
    step73_dir = Path(step73_dir)

    matrix_path = step73_dir / "step73_track_dynamic_feature_matrix.csv"
    raw_path = step73_dir / "step73_track_dynamic_feature_raw_matrix.csv"
    long_path = step73_dir / "step73_selected_dynamic_features_long.csv"
    reg_path = step73_dir / "step73_track_top_regulators.csv"
    feature_selection_path = step73_dir / "step73_feature_selection_summary.csv"

    mat = read_csv(matrix_path, required=True, index_col=0)
    raw = read_csv(raw_path, required=False, index_col=0)
    long_df = read_csv(long_path, required=False)
    regulators = read_csv(reg_path, required=False)
    feature_selection = read_csv(feature_selection_path, required=False)

    # Convert numeric and collapse duplicated track rows if any.
    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if mat.index.duplicated().any():
        mat = mat.groupby(mat.index).mean()

    if raw.empty:
        raw = mat.copy()
    else:
        raw = raw.apply(pd.to_numeric, errors="coerce")
        if raw.index.duplicated().any():
            raw = raw.groupby(raw.index).mean()

    # Remove empty rows/columns.
    mat = mat.loc[mat.abs().sum(axis=1) > 0, mat.abs().sum(axis=0) > 0]
    raw = raw.reindex(index=mat.index, columns=mat.columns)

    # Normalize regulator track labels.
    if not regulators.empty and "track_id" in regulators.columns:
        regulators["track_id"] = regulators["track_id"].astype(str)

    return mat, raw, long_df, regulators, feature_selection


def make_compact_figure(
    mat,
    row_anno,
    feature_anno,
    outbase,
    clean=False,
    dpi=600,
    title="Track-specific dynamic marker, module and regulator landscape"
):
    tracks = mat.index.tolist()
    features = mat.columns.tolist()
    data = mat.to_numpy(dtype=float)

    vmax = np.nanquantile(np.abs(data), 0.95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    feature_type = dict(zip(feature_anno["feature_name"], feature_anno["feature_type"])) if not feature_anno.empty else {}
    feature_program_map = dict(zip(feature_anno["feature_name"], feature_anno["program"])) if not feature_anno.empty else {}

    row_program = dict(zip(row_anno["track_id"], row_anno["dominant_program"]))
    row_regs = dict(zip(row_anno["track_id"], row_anno["top_regulators_compact"]))

    program_list = [p for p, _ in PROGRAM_RULES] + ["Other"]
    program_to_idx = {p: i for i, p in enumerate(program_list)}
    program_colors = [PROGRAM_COLORS[p] for p in program_list]

    feature_type_list = ["gene", "module", "unknown"]
    feature_type_to_idx = {p: i for i, p in enumerate(feature_type_list)}
    feature_type_colors = [FEATURE_TYPE_COLORS[p] for p in feature_type_list]

    row_prog_values = np.array([[program_to_idx.get(row_program.get(t, "Other"), program_to_idx["Other"])] for t in tracks])
    col_type_values = np.array([[feature_type_to_idx.get(feature_type.get(f, "unknown"), 2) for f in features]])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    n_rows, n_cols = data.shape

    if clean:
        fig_w = max(8, min(17, 0.34 * n_cols + 2.0))
        fig_h = max(4.8, min(12, 0.42 * n_rows + 1.2))
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(1, 1)
        ax_hm = fig.add_subplot(gs[0, 0])
        ax_left = ax_top = ax_reg = None
    else:
        fig_w = max(12, min(22, 0.42 * n_cols + 6.4))
        fig_h = max(6.2, min(14, 0.46 * n_rows + 2.8))
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[0.28, 8.5],
            width_ratios=[0.35, max(7.5, 0.42 * n_cols), 3.0],
            hspace=0.04,
            wspace=0.04,
        )
        ax_blank = fig.add_subplot(gs[0, 0])
        ax_top = fig.add_subplot(gs[0, 1])
        ax_top_right = fig.add_subplot(gs[0, 2])
        ax_left = fig.add_subplot(gs[1, 0])
        ax_hm = fig.add_subplot(gs[1, 1])
        ax_reg = fig.add_subplot(gs[1, 2])
        ax_blank.axis("off")
        ax_top_right.axis("off")

    im = ax_hm.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")

    if clean:
        ax_hm.set_xticks([])
        ax_hm.set_yticks([])
    else:
        ax_hm.set_xticks(np.arange(n_cols))
        ax_hm.set_xticklabels([short_text(f, 16) for f in features], rotation=60, ha="right", fontsize=7.5)
        ax_hm.set_yticks(np.arange(n_rows))
        ax_hm.set_yticklabels([str(t).replace("Track ", "T") for t in tracks], fontsize=8.5)

        ax_hm.set_xlabel("Selected dynamic genes/modules", fontsize=9)
        ax_hm.set_ylabel("Representative progression tracks", fontsize=9)

        for i in range(n_rows + 1):
            ax_hm.axhline(i - 0.5, color="white", lw=0.35, alpha=0.65)
        for j in range(n_cols + 1):
            ax_hm.axvline(j - 0.5, color="white", lw=0.25, alpha=0.45)

        cb = fig.colorbar(im, ax=ax_hm, fraction=0.025, pad=0.012)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("signed dynamic score", fontsize=8)

    for sp in ax_hm.spines.values():
        sp.set_visible(False)

    if not clean:
        # Top annotation: feature class.
        ax_top.imshow(col_type_values, cmap=ListedColormap(feature_type_colors), aspect="auto")
        ax_top.set_xlim(-0.5, n_cols - 0.5)
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title(title, fontsize=13.5, fontweight="bold", pad=10)
        for sp in ax_top.spines.values():
            sp.set_visible(False)

        # Left annotation: dominant program.
        ax_left.imshow(row_prog_values, cmap=ListedColormap(program_colors), aspect="auto")
        ax_left.set_xticks([])
        ax_left.set_yticks(np.arange(n_rows))
        ax_left.set_yticklabels([""] * n_rows)
        for sp in ax_left.spines.values():
            sp.set_visible(False)
        ax_left.set_ylabel("Dominant program", fontsize=8)

        # Right regulator text.
        ax_reg.set_xlim(0, 1)
        ax_reg.set_ylim(-0.5, n_rows - 0.5)
        ax_reg.invert_yaxis()
        ax_reg.axis("off")
        ax_reg.set_title("Top regulators", fontsize=10, fontweight="bold", pad=8)

        for i, t in enumerate(tracks):
            if i % 2 == 0:
                ax_reg.add_patch(Rectangle((0, i - 0.5), 1, 1, facecolor="#f8fafc", edgecolor="none", zorder=0))
            regs = row_regs.get(t, "—")
            prog = row_program.get(t, "Other")
            ax_reg.text(0.02, i, regs, ha="left", va="center", fontsize=8.2, color="#111827")
            ax_reg.text(0.74, i, short_text(prog, 20), ha="left", va="center", fontsize=7.2, color="#64748b")

        # Legends.
        legend_y = 0.02
        x0 = 0.02
        dx = 0.092
        fig.text(x0, legend_y + 0.045, "Programs:", fontsize=7.5, color="#334155", ha="left")
        for k, p in enumerate(program_list[:7]):
            fig.patches.append(Rectangle((x0 + k * dx, legend_y + 0.018), 0.012, 0.012,
                                         transform=fig.transFigure,
                                         facecolor=PROGRAM_COLORS[p], edgecolor="none"))
            fig.text(x0 + k * dx + 0.015, legend_y + 0.017, p.replace("/", "/\n") if len(p) > 15 else p,
                     fontsize=6.5, ha="left", va="bottom", color="#475569")

        fig.text(
            0.5,
            0.002,
            "Compact view of transcript-level dynamic markers/modules across representative tracks; regulators are enrichment-derived and not functional validation.",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#475569",
        )

    outbase = Path(outbase)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--step73_dir", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--max_tracks", type=int, default=14)
    ap.add_argument("--max_genes", type=int, default=28)
    ap.add_argument("--max_modules", type=int, default=10)
    ap.add_argument("--min_nonzero_features_per_track", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)

    log("=" * 100)
    log("Step73B | Compact manuscript-ready track dynamic marker heatmap")
    log("=" * 100)
    log(f"step73_dir={args.step73_dir}")
    log(f"outdir={outdir}")

    mat, raw, long_df, regulators, feature_selection = prepare_matrix(args.step73_dir)

    selected_tracks = select_representative_tracks(
        mat,
        max_tracks=args.max_tracks,
        min_nonzero_features=args.min_nonzero_features_per_track
    )

    if not selected_tracks:
        raise RuntimeError("No representative tracks selected. Check Step73 matrix.")

    mat_sub = mat.loc[selected_tracks].copy()

    selected_features, feature_anno, type_map, program_map = select_compact_features(
        mat_sub,
        long_df=long_df,
        feature_selection=feature_selection,
        max_genes=args.max_genes,
        max_modules=args.max_modules
    )

    if not selected_features:
        raise RuntimeError("No compact features selected. Check Step73 matrix.")

    mat_compact = mat_sub[selected_features].copy()
    raw_compact = raw.reindex(index=selected_tracks, columns=selected_features)

    # Drop empty rows/columns again.
    mat_compact = mat_compact.loc[mat_compact.abs().sum(axis=1) > 0, :]
    mat_compact = mat_compact.loc[:, mat_compact.abs().sum(axis=0) > 0]
    raw_compact = raw_compact.reindex(index=mat_compact.index, columns=mat_compact.columns)

    feature_anno = feature_anno[feature_anno["feature_name"].isin(mat_compact.columns)].copy()
    feature_anno["plot_order"] = feature_anno["feature_name"].map({f: i for i, f in enumerate(mat_compact.columns)})
    feature_anno = feature_anno.sort_values("plot_order").drop(columns=["plot_order"])

    row_anno = build_row_annotation(mat_compact, regulators)

    # Save tables.
    z_path = outdir / "step73b_compact_dynamic_feature_matrix_z.csv"
    raw_path = outdir / "step73b_compact_dynamic_feature_matrix_raw.csv"
    row_path = outdir / "step73b_compact_track_annotations.csv"
    col_path = outdir / "step73b_compact_feature_annotations.csv"

    mat_compact.to_csv(z_path)
    raw_compact.to_csv(raw_path)
    row_anno.to_csv(row_path, index=False)
    feature_anno.to_csv(col_path, index=False)

    # Figures.
    annotated_base = outdir / "Fig_Step73B_CompactTrackDynamicMarkerHeatmap_annotated"
    clean_base = outdir / "Fig_Step73B_CompactTrackDynamicMarkerHeatmap_clean_no_text"

    make_compact_figure(
        mat_compact,
        row_anno,
        feature_anno,
        annotated_base,
        clean=False,
        dpi=args.dpi
    )

    make_compact_figure(
        mat_compact,
        row_anno,
        feature_anno,
        clean_base,
        clean=True,
        dpi=args.dpi
    )

    report = {
        "status": "ok",
        "analysis_name": "Step73B compact manuscript-ready track-specific dynamic marker heatmap",
        "inputs": {
            "step73_dir": str(args.step73_dir),
        },
        "n_tracks_input": int(mat.shape[0]),
        "n_features_input": int(mat.shape[1]),
        "n_tracks_compact": int(mat_compact.shape[0]),
        "n_features_compact": int(mat_compact.shape[1]),
        "selected_tracks": list(mat_compact.index.astype(str)),
        "outputs": {
            "compact_matrix_z": str(z_path),
            "compact_matrix_raw": str(raw_path),
            "track_annotations": str(row_path),
            "feature_annotations": str(col_path),
            "annotated_pdf": str(annotated_base.with_suffix(".pdf")),
            "annotated_svg": str(annotated_base.with_suffix(".svg")),
            "annotated_png": str(annotated_base.with_suffix(".png")),
            "clean_pdf": str(clean_base.with_suffix(".pdf")),
            "clean_svg": str(clean_base.with_suffix(".svg")),
            "clean_png": str(clean_base.with_suffix(".png")),
        },
        "interpretation_note": (
            "Step73B is a compact visualization of Step65/Step73 transcript-level dynamic markers/modules "
            "and enrichment-derived regulators. It is suitable for manuscript/Extended Data presentation, "
            "but does not represent protein-level or functional validation."
        )
    }

    (outdir / "step73b_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step73b_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 100)
    log("DONE Step73B")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
