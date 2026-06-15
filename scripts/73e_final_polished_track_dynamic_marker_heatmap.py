#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step73E | Final polished track-specific dynamic marker heatmap

Input:
  Step73D output directory:
    step73d_labelclean_dynamic_feature_matrix_z.csv
    step73d_labelclean_dynamic_feature_matrix_raw.csv
    step73d_labelclean_track_annotations.csv
    step73d_labelclean_feature_annotations.csv

Purpose:
  Final aesthetic/polished figure after Step73D:
    - no overlap between y-axis labels and program strip
    - dedicated colorbar axis
    - dedicated regulator axis
    - separate bottom legend row
    - cleaned regulator labels only
    - optional feature/track reduction for main-text display

Interpretation:
  Transcript-level track-specific dynamic marker/module/regulator prioritization.
  Regulator labels are enrichment-derived computational annotations, not functional validation.
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


PROGRAM_ORDER = [
    "Hypoxia/redox",
    "Ferroptosis",
    "BBB/endothelial",
    "Inflammation",
    "Astrocyte/reactive",
    "Repair/ECM",
    "Synaptic/recovery",
    "Unassigned",
]

PROGRAM_COLORS = {
    "Hypoxia/redox": "#8dd3c7",
    "Ferroptosis": "#fb8072",
    "BBB/endothelial": "#80b1d3",
    "Inflammation": "#fdb462",
    "Astrocyte/reactive": "#b3de69",
    "Repair/ECM": "#bebada",
    "Synaptic/recovery": "#fccde5",
    "Unassigned": "#d9d9d9",
}

FEATURE_TYPE_COLORS = {
    "gene": "#cbd5e1",
    "module": "#334155",
    "unknown": "#e5e7eb",
}

PROGRAM_RULES = [
    ("Hypoxia/redox", ["hypoxia", "hif", "hmox", "nfe2l", "redox", "oxid"]),
    ("Ferroptosis", ["ferroptosis", "acsl4", "gpx4", "tfrc", "slc7a11", "ptgs2", "fth", "ftl"]),
    ("BBB/endothelial", ["bbb", "barrier", "endothelial", "pecam", "kdr", "flt1", "vwf", "claudin", "occludin"]),
    ("Inflammation", ["inflamm", "chemotaxis", "ccl", "cxcl", "tnf", "il1", "microglia", "ctsd", "c1qa", "c1qb"]),
    ("Astrocyte/reactive", ["astro", "reactive", "gfap", "vim", "serpina", "stat3"]),
    ("Repair/ECM", ["repair", "ecm", "collagen", "col1", "col3", "fn1", "sparc", "mmp", "timp", "postn"]),
    ("Synaptic/recovery", ["synaptic", "recovery", "neuron", "snap", "syt", "grin", "map2", "rbfox", "olig"]),
]


def log(msg):
    print(msg, flush=True)


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


def short_text(x, n=15):
    x = "" if pd.isna(x) else str(x).strip()
    if len(x) <= n:
        return x
    return x[: n - 1] + "…"


def feature_program(feature):
    low = str(feature).lower()
    for program, keys in PROGRAM_RULES:
        if any(k in low for k in keys):
            return program
    return "Unassigned"


def recompute_row_program(row):
    scores = {p: 0.0 for p in PROGRAM_ORDER}
    for feature, value in row.items():
        p = feature_program(feature)
        try:
            scores[p] += abs(float(value))
        except Exception:
            pass
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "Unassigned"


def clean_reg_label(x):
    x = "" if pd.isna(x) else str(x).strip()
    if not x or x.lower() in {"nan", "none", "null", "other"}:
        return "—"

    parts = []
    for token in re.split(r"[,;/|]+", x):
        t = token.strip()
        if not t:
            continue
        if t in {"AGI", "AR", "ARCHS", "ARCHS4", "MURI", "MURIS", "MYELOI", "UTERU", "UTERUS", "OTHER"}:
            continue
        parts.append(t)

    parts = list(dict.fromkeys(parts))
    if not parts:
        return "—"
    return ", ".join(parts[:3])


def select_tracks(mat, row_anno, max_tracks):
    if max_tracks <= 0 or mat.shape[0] <= max_tracks:
        return list(mat.index)

    scores = {}
    for t in mat.index:
        x = mat.loc[t].abs()
        scores[t] = float(x.max() + 0.35 * x.mean() + 0.03 * (x > 0).sum())

    row_anno = row_anno.copy()
    row_anno["track_id"] = row_anno["track_id"].astype(str)

    if "dominant_program_clean" in row_anno.columns:
        prog_col = "dominant_program_clean"
    elif "dominant_program" in row_anno.columns:
        prog_col = "dominant_program"
    else:
        prog_col = None

    program_map = {}
    for t in mat.index:
        if prog_col:
            hit = row_anno[row_anno["track_id"] == t]
            if not hit.empty:
                p = str(hit[prog_col].iloc[0])
                if p and p not in {"Other", "nan", "NaN"}:
                    program_map[t] = p
                    continue
        program_map[t] = recompute_row_program(mat.loc[t])

    selected = []

    for p in PROGRAM_ORDER:
        sub = [t for t in mat.index if program_map.get(t, "Unassigned") == p]
        if sub:
            sub = sorted(sub, key=lambda t: scores.get(t, 0), reverse=True)
            selected.append(sub[0])

    for t in sorted(mat.index, key=lambda x: scores.get(x, 0), reverse=True):
        if t not in selected:
            selected.append(t)
        if len(selected) >= max_tracks:
            break

    selected = selected[:max_tracks]
    program_rank = {p: i for i, p in enumerate(PROGRAM_ORDER)}
    selected = sorted(selected, key=lambda t: (program_rank.get(program_map.get(t, "Unassigned"), 999), -scores.get(t, 0), str(t)))
    return selected


def select_features(mat, feature_anno, max_features):
    if max_features <= 0 or mat.shape[1] <= max_features:
        selected = list(mat.columns)
    else:
        rows = []
        for f in mat.columns:
            x = mat[f].abs()
            hit = feature_anno[feature_anno["feature_name"].astype(str) == str(f)] if "feature_name" in feature_anno.columns else pd.DataFrame()
            ftype = hit["feature_type"].iloc[0] if not hit.empty and "feature_type" in hit.columns else "gene"
            prog = hit["program"].iloc[0] if not hit.empty and "program" in hit.columns else feature_program(f)
            rows.append({
                "feature_name": f,
                "feature_type": ftype,
                "program": prog,
                "score": float(x.max() + 0.4 * x.mean() + 0.03 * (x > 0).sum()),
            })
        score_df = pd.DataFrame(rows)
        selected = []

        # preserve program diversity
        for p in PROGRAM_ORDER:
            sub = score_df[score_df["program"] == p].sort_values("score", ascending=False)
            if not sub.empty:
                selected.append(sub["feature_name"].iloc[0])

        for f in score_df.sort_values("score", ascending=False)["feature_name"]:
            if f not in selected:
                selected.append(f)
            if len(selected) >= max_features:
                break

        selected = selected[:max_features]

        program_rank = {p: i for i, p in enumerate(PROGRAM_ORDER)}
        score_lookup = score_df.set_index("feature_name").to_dict(orient="index")
        selected = sorted(
            selected,
            key=lambda f: (
                program_rank.get(score_lookup.get(f, {}).get("program", "Unassigned"), 999),
                0 if score_lookup.get(f, {}).get("feature_type", "gene") == "module" else 1,
                -score_lookup.get(f, {}).get("score", 0),
                str(f)
            )
        )

    return selected


def prepare_annotations(mat, row_anno, feature_anno):
    row_anno = row_anno.copy()
    row_anno["track_id"] = row_anno["track_id"].astype(str)

    if "dominant_program_clean" not in row_anno.columns:
        if "dominant_program" in row_anno.columns:
            row_anno["dominant_program_clean"] = row_anno["dominant_program"]
        else:
            row_anno["dominant_program_clean"] = "Unassigned"

    prog_map = {}
    for t in mat.index:
        hit = row_anno[row_anno["track_id"] == t]
        if hit.empty:
            prog_map[t] = recompute_row_program(mat.loc[t])
        else:
            p = str(hit["dominant_program_clean"].iloc[0])
            if p in {"", "Other", "nan", "NaN", "None"}:
                p = recompute_row_program(mat.loc[t])
            prog_map[t] = p if p in PROGRAM_COLORS else recompute_row_program(mat.loc[t])

    row_anno["dominant_program_final"] = row_anno["track_id"].map(prog_map).fillna("Unassigned")

    if "top_regulators_final" not in row_anno.columns:
        if "top_regulators_ultraclean" in row_anno.columns:
            row_anno["top_regulators_final"] = row_anno["top_regulators_ultraclean"]
        elif "top_regulators_compact" in row_anno.columns:
            row_anno["top_regulators_final"] = row_anno["top_regulators_compact"]
        elif "top_regulators" in row_anno.columns:
            row_anno["top_regulators_final"] = row_anno["top_regulators"]
        else:
            row_anno["top_regulators_final"] = "—"

    row_anno["top_regulators_final"] = row_anno["top_regulators_final"].map(clean_reg_label)

    row_anno = row_anno[row_anno["track_id"].isin(mat.index)].copy()
    row_anno["plot_order"] = row_anno["track_id"].map({t: i for i, t in enumerate(mat.index)})
    row_anno = row_anno.sort_values("plot_order").drop(columns=["plot_order"])

    feature_anno = feature_anno.copy()
    if "feature_name" not in feature_anno.columns:
        feature_anno = pd.DataFrame({"feature_name": list(mat.columns)})

    if "feature_type" not in feature_anno.columns:
        feature_anno["feature_type"] = "gene"

    if "program" not in feature_anno.columns:
        feature_anno["program"] = feature_anno["feature_name"].map(feature_program)

    feature_anno = feature_anno[feature_anno["feature_name"].astype(str).isin(mat.columns.astype(str))].copy()
    feature_anno["plot_order"] = feature_anno["feature_name"].map({f: i for i, f in enumerate(mat.columns)})
    feature_anno = feature_anno.sort_values("plot_order").drop(columns=["plot_order"])

    return row_anno, feature_anno


def make_polished_figure(mat, row_anno, feature_anno, outbase, clean=False, dpi=600):
    tracks = list(mat.index.astype(str))
    features = list(mat.columns.astype(str))
    data = mat.to_numpy(dtype=float)

    vmax = np.nanquantile(np.abs(data), 0.96)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    row_map = row_anno.set_index("track_id").to_dict(orient="index")
    feature_map = feature_anno.set_index("feature_name").to_dict(orient="index")

    program_to_idx = {p: i for i, p in enumerate(PROGRAM_ORDER)}
    program_cmap = ListedColormap([PROGRAM_COLORS[p] for p in PROGRAM_ORDER])

    ftypes = ["gene", "module", "unknown"]
    ftype_to_idx = {p: i for i, p in enumerate(ftypes)}
    ftype_cmap = ListedColormap([FEATURE_TYPE_COLORS[p] for p in ftypes])

    row_prog_values = np.array([
        [program_to_idx.get(row_map.get(t, {}).get("dominant_program_final", "Unassigned"), program_to_idx["Unassigned"])]
        for t in tracks
    ])

    col_type_values = np.array([[
        ftype_to_idx.get(str(feature_map.get(f, {}).get("feature_type", "unknown")), 2)
        for f in features
    ]])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    })

    n_rows, n_cols = data.shape

    if clean:
        fig = plt.figure(
            figsize=(max(7.6, 0.38 * n_cols + 1.3), max(4.1, 0.46 * n_rows + 0.8)),
            facecolor="white"
        )
        ax = fig.add_subplot(111)
        ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    else:
        fig_w = max(13.2, 0.50 * n_cols + 5.8)
        fig_h = max(6.4, 0.56 * n_rows + 3.0)

        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

        gs = fig.add_gridspec(
            nrows=4,
            ncols=4,
            height_ratios=[0.42, 0.22, 7.2, 1.05],
            width_ratios=[0.18, 8.4, 0.32, 2.05],
            hspace=0.10,
            wspace=0.08,
        )

        ax_title = fig.add_subplot(gs[0, :])
        ax_top = fig.add_subplot(gs[1, 1])
        ax_left = fig.add_subplot(gs[2, 0])
        ax_heat = fig.add_subplot(gs[2, 1])
        ax_cbar = fig.add_subplot(gs[2, 2])
        ax_reg = fig.add_subplot(gs[2, 3])
        ax_legend = fig.add_subplot(gs[3, :])

        ax_title.axis("off")
        ax_title.text(
            0.5, 0.55,
            "Track-specific dynamic marker landscape",
            ha="center", va="center",
            fontsize=15.5,
            fontweight="bold",
            color="#111827"
        )

        # feature type strip
        ax_top.imshow(col_type_values, cmap=ftype_cmap, aspect="auto")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        for sp in ax_top.spines.values():
            sp.set_visible(False)
        ax_top.text(
            0.0, 1.15,
            "feature class: gene / module",
            transform=ax_top.transAxes,
            ha="left", va="bottom",
            fontsize=7.5,
            color="#64748b"
        )

        # left program strip
        ax_left.imshow(row_prog_values, cmap=program_cmap, aspect="auto")
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        for sp in ax_left.spines.values():
            sp.set_visible(False)

        # heatmap
        im = ax_heat.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        ax_heat.set_xticks(np.arange(n_cols))
        ax_heat.set_xticklabels([short_text(f, 14) for f in features], rotation=55, ha="right", fontsize=8.2)
        ax_heat.set_yticks(np.arange(n_rows))
        ax_heat.set_yticklabels([t.replace("Track ", "T") for t in tracks], fontsize=9.2)
        ax_heat.set_xlabel("Selected dynamic genes / modules", fontsize=9.5, labelpad=8)
        ax_heat.set_ylabel("Representative tracks", fontsize=9.5, labelpad=10)

        for i in range(n_rows + 1):
            ax_heat.axhline(i - 0.5, color="white", lw=0.35, alpha=0.75)
        for j in range(n_cols + 1):
            ax_heat.axvline(j - 0.5, color="white", lw=0.25, alpha=0.45)

        for sp in ax_heat.spines.values():
            sp.set_visible(False)

        cb = fig.colorbar(im, cax=ax_cbar)
        cb.ax.tick_params(labelsize=7.5, length=2)
        cb.set_label("signed dynamic score", fontsize=8.5, labelpad=8)

        # regulator axis
        ax_reg.set_xlim(0, 1)
        ax_reg.set_ylim(-0.5, n_rows - 0.5)
        ax_reg.invert_yaxis()
        ax_reg.axis("off")
        ax_reg.set_title("Regulators", fontsize=10.5, fontweight="bold", pad=9, color="#111827")

        for i, t in enumerate(tracks):
            if i % 2 == 0:
                ax_reg.add_patch(Rectangle((0, i - 0.5), 1, 1, facecolor="#f8fafc", edgecolor="none", zorder=0))
            regs = row_map.get(t, {}).get("top_regulators_final", "—")
            color = "#111827" if regs != "—" else "#94a3b8"
            ax_reg.text(0.03, i, regs, ha="left", va="center", fontsize=9.0, color=color)

        # bottom legend and note
        ax_legend.axis("off")
        ax_legend.set_xlim(0, 1)
        ax_legend.set_ylim(0, 1)

        ax_legend.text(0.02, 0.72, "Dominant programs", fontsize=8.0, color="#334155", ha="left", va="center")

        x = 0.16
        y = 0.72
        dx = 0.112

        for k, p in enumerate([p for p in PROGRAM_ORDER if p != "Unassigned"]):
            xx = x + k * dx
            ax_legend.add_patch(Rectangle((xx, y - 0.055), 0.014, 0.11, facecolor=PROGRAM_COLORS[p], edgecolor="none"))
            ax_legend.text(xx + 0.018, y, p, fontsize=7.4, color="#475569", ha="left", va="center")

        ax_legend.text(
            0.5,
            0.18,
            "Transcript-level dynamic markers/modules; regulator labels are cleaned enrichment-derived annotations and do not imply functional validation.",
            ha="center",
            va="center",
            fontsize=8.2,
            color="#475569"
        )

        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.055, top=0.965)

    outbase = Path(outbase)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--step73d_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--max_tracks", type=int, default=8)
    parser.add_argument("--max_features", type=int, default=18)
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    step73d_dir = Path(args.step73d_dir)

    log("=" * 100)
    log("Step73E | Final polished dynamic marker heatmap")
    log("=" * 100)
    log(f"step73d_dir={step73d_dir}")
    log(f"outdir={outdir}")

    mat = read_csv(step73d_dir / "step73d_labelclean_dynamic_feature_matrix_z.csv", required=True, index_col=0)
    raw = read_csv(step73d_dir / "step73d_labelclean_dynamic_feature_matrix_raw.csv", required=False, index_col=0)
    row_anno = read_csv(step73d_dir / "step73d_labelclean_track_annotations.csv", required=True)
    feature_anno = read_csv(step73d_dir / "step73d_labelclean_feature_annotations.csv", required=True)

    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mat = mat.loc[mat.abs().sum(axis=1) > 0, mat.abs().sum(axis=0) > 0]

    if raw.empty:
        raw = mat.copy()
    else:
        raw = raw.apply(pd.to_numeric, errors="coerce")
        raw = raw.reindex(index=mat.index, columns=mat.columns)

    selected_tracks = select_tracks(mat, row_anno, max_tracks=args.max_tracks)
    mat = mat.loc[selected_tracks].copy()
    raw = raw.reindex(index=mat.index, columns=mat.columns)

    selected_features = select_features(mat, feature_anno, max_features=args.max_features)
    mat = mat[selected_features].copy()
    raw = raw.reindex(index=mat.index, columns=mat.columns)

    row_anno, feature_anno = prepare_annotations(mat, row_anno, feature_anno)

    z_path = outdir / "step73e_final_polished_dynamic_feature_matrix_z.csv"
    raw_path = outdir / "step73e_final_polished_dynamic_feature_matrix_raw.csv"
    row_path = outdir / "step73e_final_polished_track_annotations.csv"
    col_path = outdir / "step73e_final_polished_feature_annotations.csv"

    mat.to_csv(z_path)
    raw.to_csv(raw_path)
    row_anno.to_csv(row_path, index=False)
    feature_anno.to_csv(col_path, index=False)

    annotated_base = outdir / "Fig_Step73E_FinalPolishedTrackDynamicMarkerHeatmap_annotated"
    clean_base = outdir / "Fig_Step73E_FinalPolishedTrackDynamicMarkerHeatmap_clean_no_text"

    make_polished_figure(mat, row_anno, feature_anno, annotated_base, clean=False, dpi=args.dpi)
    make_polished_figure(mat, row_anno, feature_anno, clean_base, clean=True, dpi=args.dpi)

    report = {
        "status": "ok",
        "analysis_name": "Step73E final polished track-specific dynamic marker heatmap",
        "inputs": {
            "step73d_dir": str(step73d_dir),
        },
        "n_tracks": int(mat.shape[0]),
        "n_features": int(mat.shape[1]),
        "selected_tracks": list(mat.index.astype(str)),
        "selected_features": list(mat.columns.astype(str)),
        "outputs": {
            "matrix_z": str(z_path),
            "matrix_raw": str(raw_path),
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
            "Step73E is the final polished layout of the track-specific dynamic marker landscape. "
            "It preserves Step73D label-cleaned regulators while improving figure spacing, colorbar placement, "
            "program legend placement and manuscript readability. Regulator labels remain enrichment-derived "
            "computational annotations, not functional validation."
        ),
    }

    (outdir / "step73e_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step73e_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 100)
    log("DONE Step73E")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
