#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step73C | Ultra-clean manuscript-ready track-specific dynamic marker heatmap

Input:
  Step73B outputs:
    step73b_compact_dynamic_feature_matrix_z.csv
    step73b_compact_dynamic_feature_matrix_raw.csv
    step73b_compact_track_annotations.csv
    step73b_compact_feature_annotations.csv

Optional:
  Step73 original regulator table if available:
    step73_track_top_regulators.csv

Purpose:
  Make an ultra-clean main-text style panel:
    - fewer representative tracks
    - fewer top genes/modules
    - simplified regulator names
    - less crowded labels
    - clean PDF/SVG/PNG output

Interpretation:
  Transcript-level dynamic marker/module/regulator prioritization.
  Regulators are enrichment-derived annotations, not functional validation.
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
    ("Inflammation", ["inflamm", "chemotaxis", "ccl", "cxcl", "tnf", "il1", "microglia", "macrophage"]),
    ("Astrocyte/reactive", ["astro", "reactive", "gfap", "vim", "serpina", "stat3"]),
    ("Repair/ECM", ["repair", "ecm", "collagen", "col1", "col3", "fn1", "sparc", "mmp", "timp", "postn"]),
    ("Synaptic/recovery", ["synaptic", "recovery", "neuron", "snap", "syt", "grin", "map2", "rbfox"]),
]

PROGRAM_COLORS = {
    "Hypoxia/redox": "#8dd3c7",
    "Ferroptosis": "#fb8072",
    "BBB/endothelial": "#80b1d3",
    "Inflammation": "#fdb462",
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

BAD_REGULATOR_TOKENS = {
    "GOCC", "GOBP", "GOMF", "HALLMARK", "REACTOME", "KEGG", "WP", "PID",
    "ARCHS4", "PEARSON", "TABULA", "MURIS", "SENIS", "BRAIN", "SPLEEN",
    "UTERUS", "MACROPHAGE", "NEURON", "PROJECTION", "AXON", "MYELOID",
    "MICROGLIA", "AGING", "CEREBELLUM", "DESCARTES", "LEE", "ZHANG",
    "C5", "UP", "DOWN", "MODULE", "SIGNATURE", "GENESET", "GENES",
    "NON", "MYELOID", "ENDO", "ORG", "PEARSONR"
}

KNOWN_REGULATORS = {
    "NFE2L1", "NFE2L2", "BHLHE41", "OLIG1", "OLIG2", "SOX8", "SOX9",
    "MAFB", "MAF", "SPI1", "CEBPB", "CEBPA", "CENPB", "TSC22D1",
    "JUN", "FOS", "STAT3", "RELA", "NFKB1", "HIF1A", "EPAS1",
    "IRF1", "IRF8", "RUNX1", "KLF2", "KLF4", "ETS1", "ERG",
    "FOXO1", "ATF3", "ATF4", "CREB1", "PPARG", "RXRA", "SP1",
    "MYC", "EGR1", "SMAD3", "SMAD4", "TEAD1", "YAP1", "CTCF"
}


def log(x):
    print(x, flush=True)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_csv(path, required=False, index_col=None):
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(str(p))
        return pd.DataFrame()
    return pd.read_csv(p, index_col=index_col, low_memory=False)


def short_text(x, n=22):
    x = "" if pd.isna(x) else str(x).strip()
    if len(x) <= n:
        return x
    return x[:n - 1] + "…"


def feature_program(name):
    low = str(name).lower()
    for program, keys in PROGRAM_RULES:
        if any(k in low for k in keys):
            return program
    return "Other"


def natural_track_sort_key(x):
    s = str(x)
    m = re.search(r"(\d+)", s)
    if m:
        return (0, int(m.group(1)), s)
    return (1, 999999, s)


def clean_gene_like_token(tok):
    tok = str(tok).strip()
    tok = re.sub(r"[^A-Za-z0-9_.-]", "", tok)
    tok = tok.upper()
    tok = tok.replace(".", "")
    return tok


def is_plausible_regulator(tok):
    tok = clean_gene_like_token(tok)
    if not tok:
        return False
    if tok in BAD_REGULATOR_TOKENS:
        return False
    if tok in KNOWN_REGULATORS:
        return True

    # Gene-symbol-like: 2–8 chars, contains letters, optionally digits.
    if not (2 <= len(tok) <= 10):
        return False
    if not re.search(r"[A-Z]", tok):
        return False
    if re.fullmatch(r"C\d+", tok):
        return False
    if tok.startswith("GSE") or tok.startswith("GSM"):
        return False
    if tok.startswith("GO"):
        return False
    if tok in {"UP", "DOWN", "CORE", "PERI", "REMOTE", "AGING", "BRAIN"}:
        return False

    # Prefer TF-like symbols with digits or common capitalization patterns.
    if re.fullmatch(r"[A-Z]{2,6}\d{0,2}", tok):
        return True
    if re.fullmatch(r"[A-Z]{1,4}\d[A-Z0-9]{0,4}", tok):
        return True

    return False


def simplify_regulator_text(text, max_n=3):
    text = "" if pd.isna(text) else str(text)
    if not text.strip():
        return "—"

    # Split aggressively because many terms are like BHLHE41_ARCHS4_PEARSON_ZHANG...
    raw_parts = re.split(r"[,;/| \t]+", text)
    candidates = []

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        # Also split by underscores but keep original leading token.
        subparts = re.split(r"[_\-]+", part)

        for tok in [part] + subparts:
            tok2 = clean_gene_like_token(tok)
            if is_plausible_regulator(tok2):
                candidates.append(tok2)

    # Prefer known regulators first.
    candidates = list(dict.fromkeys(candidates))
    candidates = sorted(
        candidates,
        key=lambda x: (0 if x in KNOWN_REGULATORS else 1, len(x), x)
    )
    candidates = candidates[:max_n]

    if not candidates:
        return "—"

    return ", ".join(candidates)


def row_dominant_program(row):
    scores = {p: 0.0 for p, _ in PROGRAM_RULES}
    scores["Other"] = 0.0
    for f, v in row.items():
        p = feature_program(f)
        try:
            scores[p] += abs(float(v))
        except Exception:
            pass
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "Other"


def select_tracks(mat, row_anno, max_tracks=10):
    row_anno = row_anno.copy()
    if "track_id" not in row_anno.columns:
        row_anno = pd.DataFrame({"track_id": mat.index.astype(str)})

    row_anno["track_id"] = row_anno["track_id"].astype(str)

    if "dominant_program" not in row_anno.columns:
        row_anno["dominant_program"] = [
            row_dominant_program(mat.loc[t]) if t in mat.index else "Other"
            for t in row_anno["track_id"]
        ]

    if "row_score" not in row_anno.columns:
        score_map = {}
        for t in mat.index:
            x = mat.loc[t].abs()
            score_map[t] = float(x.max() + 0.35 * x.mean() + 0.03 * (x > 0).sum())
        row_anno["row_score"] = row_anno["track_id"].map(score_map).fillna(0.0)

    row_anno = row_anno[row_anno["track_id"].isin(mat.index)].copy()

    selected = []
    for program, _ in PROGRAM_RULES:
        sub = row_anno[row_anno["dominant_program"] == program]
        if not sub.empty:
            selected.append(sub.sort_values("row_score", ascending=False)["track_id"].iloc[0])

    for t in row_anno.sort_values("row_score", ascending=False)["track_id"]:
        if t not in selected:
            selected.append(t)
        if len(selected) >= max_tracks:
            break

    program_order = {p: i for i, (p, _) in enumerate(PROGRAM_RULES)}
    program_order["Other"] = 999
    selected = selected[:max_tracks]

    selected = sorted(
        selected,
        key=lambda t: (
            program_order.get(row_anno.set_index("track_id").loc[t, "dominant_program"], 999),
            natural_track_sort_key(t)
        )
    )

    return selected


def select_features(mat, feature_anno, max_features=24, max_modules=7):
    feature_anno = feature_anno.copy()

    if "feature_name" not in feature_anno.columns:
        feature_anno = pd.DataFrame({"feature_name": mat.columns.astype(str)})

    if "feature_type" not in feature_anno.columns:
        feature_anno["feature_type"] = [
            "module" if (" " in str(f) or any(k in str(f).lower() for k in ["hypoxia", "ferroptosis", "bbb", "repair", "synaptic", "inflamm", "astro"])) else "gene"
            for f in feature_anno["feature_name"]
        ]

    if "program" not in feature_anno.columns:
        feature_anno["program"] = feature_anno["feature_name"].map(feature_program)

    score_rows = []
    for f in mat.columns:
        x = mat[f].astype(float).abs()
        row = feature_anno[feature_anno["feature_name"].astype(str) == str(f)]
        ftype = row["feature_type"].iloc[0] if not row.empty else "gene"
        prog = row["program"].iloc[0] if not row.empty else feature_program(f)

        score_rows.append({
            "feature_name": f,
            "feature_type": ftype,
            "program": prog,
            "score": float(x.max() + 0.4 * x.mean() + 0.03 * (x > 0).sum()),
            "max_abs": float(x.max()),
            "n_tracks_nonzero": int((x > 0).sum()),
        })

    score_df = pd.DataFrame(score_rows)
    program_order = {p: i for i, (p, _) in enumerate(PROGRAM_RULES)}
    program_order["Other"] = 999

    selected = []

    modules = score_df[score_df["feature_type"].astype(str).eq("module")].copy()
    if not modules.empty:
        modules["_p"] = modules["program"].map(program_order).fillna(999)
        modules = modules.sort_values(["_p", "score"], ascending=[True, False])
        selected += modules["feature_name"].head(max_modules).tolist()

    genes = score_df[~score_df["feature_type"].astype(str).eq("module")].copy()
    genes = genes.sort_values("score", ascending=False)
    selected += genes["feature_name"].head(max_features - len(selected)).tolist()

    selected = list(dict.fromkeys(selected))

    selected = sorted(
        selected,
        key=lambda f: (
            program_order.get(score_df.set_index("feature_name").loc[f, "program"], 999),
            0 if score_df.set_index("feature_name").loc[f, "feature_type"] == "module" else 1,
            -score_df.set_index("feature_name").loc[f, "score"],
            str(f)
        )
    )

    out_anno = score_df[score_df["feature_name"].isin(selected)].copy()
    out_anno["plot_order"] = out_anno["feature_name"].map({f: i for i, f in enumerate(selected)})
    out_anno = out_anno.sort_values("plot_order").drop(columns=["plot_order"])

    return selected, out_anno


def build_ultra_row_anno(mat, row_anno):
    row_anno = row_anno.copy()
    row_anno["track_id"] = row_anno["track_id"].astype(str)

    if "dominant_program" not in row_anno.columns:
        row_anno["dominant_program"] = [
            row_dominant_program(mat.loc[t]) if t in mat.index else "Other"
            for t in row_anno["track_id"]
        ]

    if "top_regulators_compact" not in row_anno.columns:
        row_anno["top_regulators_compact"] = "—"

    row_anno["top_regulators_ultraclean"] = row_anno["top_regulators_compact"].map(
        lambda x: simplify_regulator_text(x, max_n=3)
    )

    # If current compact text was already truncated and no regulator found, try all text columns.
    for i, r in row_anno.iterrows():
        if r["top_regulators_ultraclean"] != "—":
            continue
        merged_text = " ".join([str(r[c]) for c in row_anno.columns if "reg" in c.lower() or c in ["top_features"]])
        row_anno.at[i, "top_regulators_ultraclean"] = simplify_regulator_text(merged_text, max_n=3)

    if "row_score" not in row_anno.columns:
        row_anno["row_score"] = [
            float(mat.loc[t].abs().max() + 0.35 * mat.loc[t].abs().mean())
            if t in mat.index else 0.0
            for t in row_anno["track_id"]
        ]

    return row_anno


def make_figure(mat, row_anno, feature_anno, outbase, clean=False, dpi=600):
    tracks = mat.index.astype(str).tolist()
    features = mat.columns.astype(str).tolist()
    data = mat.to_numpy(dtype=float)

    vmax = np.nanquantile(np.abs(data), 0.95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    row_map = row_anno.set_index("track_id").to_dict(orient="index")
    feat_map = feature_anno.set_index("feature_name").to_dict(orient="index")

    program_list = [p for p, _ in PROGRAM_RULES] + ["Other"]
    program_to_idx = {p: i for i, p in enumerate(program_list)}
    program_cmap = ListedColormap([PROGRAM_COLORS[p] for p in program_list])

    ftype_list = ["gene", "module", "unknown"]
    ftype_to_idx = {p: i for i, p in enumerate(ftype_list)}
    ftype_cmap = ListedColormap([FEATURE_TYPE_COLORS[p] for p in ftype_list])

    row_prog_values = np.array([
        [program_to_idx.get(row_map.get(t, {}).get("dominant_program", "Other"), program_to_idx["Other"])]
        for t in tracks
    ])
    col_type_values = np.array([[
        ftype_to_idx.get(str(feat_map.get(f, {}).get("feature_type", "unknown")), 2)
        for f in features
    ]])

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    n_rows, n_cols = data.shape

    if clean:
        fig = plt.figure(figsize=(max(7.5, 0.35 * n_cols + 2), max(4.0, 0.42 * n_rows + 1)), facecolor="white")
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    else:
        fig = plt.figure(figsize=(max(12, 0.45 * n_cols + 5.5), max(5.8, 0.50 * n_rows + 2.5)), facecolor="white")
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[0.28, 8.0],
            width_ratios=[0.25, max(7.5, 0.45 * n_cols), 2.4],
            hspace=0.04,
            wspace=0.04,
        )

        ax_blank = fig.add_subplot(gs[0, 0])
        ax_top = fig.add_subplot(gs[0, 1])
        ax_topr = fig.add_subplot(gs[0, 2])
        ax_left = fig.add_subplot(gs[1, 0])
        ax = fig.add_subplot(gs[1, 1])
        ax_reg = fig.add_subplot(gs[1, 2])

        ax_blank.axis("off")
        ax_topr.axis("off")

        ax_top.imshow(col_type_values, cmap=ftype_cmap, aspect="auto")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title("Ultra-clean track-specific dynamic marker landscape",
                         fontsize=13.5, fontweight="bold", pad=8)
        for sp in ax_top.spines.values():
            sp.set_visible(False)

        ax_left.imshow(row_prog_values, cmap=program_cmap, aspect="auto")
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_left.set_ylabel("Program", fontsize=8)
        for sp in ax_left.spines.values():
            sp.set_visible(False)

        im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([short_text(f, 14) for f in features], rotation=55, ha="right", fontsize=8)
        ax.set_yticks(np.arange(n_rows))
        ax.set_yticklabels([t.replace("Track ", "T") for t in tracks], fontsize=9)
        ax.set_xlabel("Selected dynamic genes / modules", fontsize=9)
        ax.set_ylabel("Representative tracks", fontsize=9)

        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color="white", lw=0.35, alpha=0.75)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color="white", lw=0.25, alpha=0.45)

        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("signed dynamic score", fontsize=8)

        for sp in ax.spines.values():
            sp.set_visible(False)

        ax_reg.set_xlim(0, 1)
        ax_reg.set_ylim(-0.5, n_rows - 0.5)
        ax_reg.invert_yaxis()
        ax_reg.axis("off")
        ax_reg.set_title("Regulators", fontsize=10, fontweight="bold", pad=8)

        for i, t in enumerate(tracks):
            if i % 2 == 0:
                ax_reg.add_patch(Rectangle((0, i - 0.5), 1, 1, facecolor="#f8fafc", edgecolor="none", zorder=0))
            regs = row_map.get(t, {}).get("top_regulators_ultraclean", "—")
            prog = row_map.get(t, {}).get("dominant_program", "Other")
            ax_reg.text(0.02, i, regs, ha="left", va="center", fontsize=8.8, color="#111827")
            ax_reg.text(0.66, i, short_text(prog, 17), ha="left", va="center", fontsize=7.5, color="#64748b")

        # Compact program legend.
        legend_items = [p for p in program_list if p != "Other"]
        x0, y0 = 0.04, 0.018
        fig.text(x0, y0 + 0.026, "Programs", fontsize=7.5, color="#334155", ha="left")
        for k, p in enumerate(legend_items):
            xx = x0 + k * 0.118
            fig.patches.append(Rectangle((xx, y0), 0.012, 0.012,
                                         transform=fig.transFigure,
                                         facecolor=PROGRAM_COLORS[p], edgecolor="none"))
            fig.text(xx + 0.015, y0 - 0.001, p, fontsize=6.6, ha="left", va="bottom", color="#475569")

        fig.text(
            0.5,
            0.002,
            "Ultra-clean compact view of transcript-level dynamic markers/modules; regulators are enrichment-derived annotations, not functional validation.",
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

    ap.add_argument("--step73b_dir", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--max_tracks", type=int, default=10)
    ap.add_argument("--max_features", type=int, default=24)
    ap.add_argument("--max_modules", type=int, default=7)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    step73b_dir = Path(args.step73b_dir)

    log("=" * 100)
    log("Step73C | Ultra-clean track dynamic marker heatmap")
    log("=" * 100)
    log(f"step73b_dir={step73b_dir}")
    log(f"outdir={outdir}")

    mat = read_csv(step73b_dir / "step73b_compact_dynamic_feature_matrix_z.csv", required=True, index_col=0)
    raw = read_csv(step73b_dir / "step73b_compact_dynamic_feature_matrix_raw.csv", required=False, index_col=0)
    row_anno = read_csv(step73b_dir / "step73b_compact_track_annotations.csv", required=True)
    feature_anno = read_csv(step73b_dir / "step73b_compact_feature_annotations.csv", required=True)

    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mat = mat.loc[mat.abs().sum(axis=1) > 0, mat.abs().sum(axis=0) > 0]

    if raw.empty:
        raw = mat.copy()
    else:
        raw = raw.apply(pd.to_numeric, errors="coerce")
        raw = raw.reindex(index=mat.index, columns=mat.columns)

    selected_tracks = select_tracks(mat, row_anno, max_tracks=args.max_tracks)
    mat_t = mat.loc[selected_tracks].copy()

    selected_features, feature_anno_ultra = select_features(
        mat_t,
        feature_anno,
        max_features=args.max_features,
        max_modules=args.max_modules
    )

    mat_ultra = mat_t[selected_features].copy()
    raw_ultra = raw.reindex(index=mat_ultra.index, columns=mat_ultra.columns)

    row_anno_ultra = build_ultra_row_anno(mat_ultra, row_anno)
    row_anno_ultra = row_anno_ultra[row_anno_ultra["track_id"].isin(mat_ultra.index)].copy()
    row_anno_ultra["plot_order"] = row_anno_ultra["track_id"].map({t: i for i, t in enumerate(mat_ultra.index)})
    row_anno_ultra = row_anno_ultra.sort_values("plot_order").drop(columns=["plot_order"])

    # Save tables.
    z_path = outdir / "step73c_ultraclean_dynamic_feature_matrix_z.csv"
    raw_path = outdir / "step73c_ultraclean_dynamic_feature_matrix_raw.csv"
    row_path = outdir / "step73c_ultraclean_track_annotations.csv"
    col_path = outdir / "step73c_ultraclean_feature_annotations.csv"

    mat_ultra.to_csv(z_path)
    raw_ultra.to_csv(raw_path)
    row_anno_ultra.to_csv(row_path, index=False)
    feature_anno_ultra.to_csv(col_path, index=False)

    annotated_base = outdir / "Fig_Step73C_UltraCleanTrackDynamicMarkerHeatmap_annotated"
    clean_base = outdir / "Fig_Step73C_UltraCleanTrackDynamicMarkerHeatmap_clean_no_text"

    make_figure(mat_ultra, row_anno_ultra, feature_anno_ultra, annotated_base, clean=False, dpi=args.dpi)
    make_figure(mat_ultra, row_anno_ultra, feature_anno_ultra, clean_base, clean=True, dpi=args.dpi)

    report = {
        "status": "ok",
        "analysis_name": "Step73C ultra-clean manuscript-ready track-specific dynamic marker heatmap",
        "inputs": {
            "step73b_dir": str(step73b_dir),
        },
        "n_tracks_input": int(mat.shape[0]),
        "n_features_input": int(mat.shape[1]),
        "n_tracks_ultraclean": int(mat_ultra.shape[0]),
        "n_features_ultraclean": int(mat_ultra.shape[1]),
        "selected_tracks": list(mat_ultra.index.astype(str)),
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
            "Step73C is an ultra-clean compact visualization for manuscript panel use. "
            "It summarizes transcript-level dynamic markers/modules and simplified enrichment-derived regulators. "
            "Regulator names are computational annotations and do not imply functional validation."
        )
    }

    (outdir / "step73c_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step73c_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 100)
    log("DONE Step73C")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
