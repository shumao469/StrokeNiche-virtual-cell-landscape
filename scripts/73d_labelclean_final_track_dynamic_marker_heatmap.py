#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step73D | Label-cleaned final track-specific dynamic marker heatmap

Purpose
-------
Final label-cleaned version after Step73C:
  - Use Step73C ultra-clean matrix
  - Simplify regulator labels strictly
  - Remove non-informative "Other" labels from right annotation
  - Keep only credible regulator / TF-like gene symbols
  - Produce final manuscript-ready PDF/SVG/PNG

Recommended use
---------------
Main-text panel or high-quality Extended Data panel.

Interpretation
--------------
Transcript-level dynamic marker/module/regulator prioritization.
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


PROGRAM_RULES = [
    ("Hypoxia/redox", ["hypoxia", "hif", "hmox", "nfe2l2", "nfe2l1", "redox", "oxid"]),
    ("Ferroptosis", ["ferroptosis", "acsl4", "gpx4", "tfrc", "slc7a11", "ptgs2", "fth", "ftl"]),
    ("BBB/endothelial", ["bbb", "barrier", "endothelial", "pecam", "kdr", "flt1", "vwf", "claudin", "occludin"]),
    ("Inflammation", ["inflamm", "chemotaxis", "ccl", "cxcl", "tnf", "il1", "microglia", "macrophage", "ctsd", "c1qa", "c1qb"]),
    ("Astrocyte/reactive", ["astro", "reactive", "gfap", "vim", "serpina", "stat3"]),
    ("Repair/ECM", ["repair", "ecm", "collagen", "col1", "col3", "fn1", "sparc", "mmp", "timp", "postn"]),
    ("Synaptic/recovery", ["synaptic", "recovery", "neuron", "snap", "syt", "grin", "map2", "rbfox", "olig"]),
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

# Strict allow-list for credible regulators / TFs.
# Add more if your regulator database contains additional trusted symbols.
KNOWN_REGULATORS = {
    "AHR", "ARNT", "ATF1", "ATF2", "ATF3", "ATF4", "ATF5", "ATF6",
    "BACH1", "BACH2", "BATF", "BCL11A", "BCL11B", "BCL6",
    "BHLHE40", "BHLHE41",
    "CEBPA", "CEBPB", "CEBPD", "CEBPE", "CEBPG",
    "CENPB", "CREB1", "CREB3", "CREM", "CTCF",
    "DLX1", "DLX2", "DLX5",
    "E2F1", "E2F2", "E2F3", "EGR1", "EGR2", "EGR3",
    "ELF1", "ELF2", "ELK1", "ERG", "ETS1", "ETV1", "ETV4", "ETV5",
    "FOS", "FOSB", "FOSL1", "FOSL2", "FOXO1", "FOXO3", "FOXP1", "FOXP2",
    "GATA1", "GATA2", "GATA3", "GATA4", "GATA6",
    "HES1", "HEY1", "HIF1A", "HIF3A", "HLF",
    "IRF1", "IRF2", "IRF3", "IRF4", "IRF7", "IRF8", "IRF9",
    "JUN", "JUNB", "JUND",
    "KLF2", "KLF4", "KLF5", "KLF6", "KLF7", "KLF9", "KLF10", "KLF13",
    "MAF", "MAFA", "MAFB", "MAFF", "MAFG", "MAFK", "MAX", "MEF2A", "MEF2C",
    "MITF", "MYB", "MYC", "MYCN", "MYOD1",
    "NANOG", "NFE2", "NFE2L1", "NFE2L2", "NFE2L3",
    "NFATC1", "NFATC2", "NFATC3", "NFATC4", "NFKB1", "NFKB2",
    "NR1H3", "NR1H4", "NR2F1", "NR2F2", "NR3C1", "NR4A1", "NR4A2",
    "OLIG1", "OLIG2", "OLIG3",
    "PAX6", "PBX1", "PPARA", "PPARD", "PPARG", "PRDM1",
    "RELA", "RELB", "REST", "RORA", "RUNX1", "RUNX2", "RUNX3", "RXRA",
    "SMAD1", "SMAD2", "SMAD3", "SMAD4", "SMAD5", "SMAD7",
    "SOX2", "SOX4", "SOX6", "SOX8", "SOX9", "SOX10", "SOX11",
    "SP1", "SP3", "SPI1", "SPIB", "STAT1", "STAT2", "STAT3", "STAT4", "STAT5A", "STAT6",
    "TBX21", "TCF3", "TCF4", "TCF7L2", "TEAD1", "TEAD2", "TEAD4",
    "TFAP2A", "TFAP2C", "TFE3", "TFEB", "TSC22D1",
    "USF1", "USF2", "YAP1", "YY1", "ZEB1", "ZEB2",
}

# Hard blacklist for tokens seen in enrichment database names or truncated terms.
BLACKLIST_TOKENS = {
    "AGI", "AR", "ARCHS", "ARCHS4", "MURI", "MURIS", "MYELOI", "MYELOID",
    "UTERU", "UTERUS", "TABULA", "SENIS", "CEREBELLUM", "AGING",
    "GOCC", "GOBP", "GOMF", "KEGG", "REACTOME", "HALLMARK", "PEARSON",
    "ZHANG", "LEE", "DESCARTES", "BRAIN", "SPLEEN", "MACROPHAGE",
    "NEURON", "PROJECTION", "AXON", "MICROGLIA", "ENDO", "NON", "UP", "DOWN",
    "OTHER", "NA", "NAN", "NONE", "NULL",
}


def log(x):
    print(x, flush=True)


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path, required=False, index_col=None):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return pd.DataFrame()
    return pd.read_csv(path, index_col=index_col, low_memory=False)


def short_text(x, n=16):
    x = "" if pd.isna(x) else str(x).strip()
    if len(x) <= n:
        return x
    return x[: n - 1] + "…"


def clean_token(tok):
    tok = "" if pd.isna(tok) else str(tok)
    tok = re.sub(r"[^A-Za-z0-9_.-]", "", tok)
    tok = tok.upper().replace(".", "")
    return tok


def feature_program(name):
    low = str(name).lower()
    for program, keys in PROGRAM_RULES:
        if any(k in low for k in keys):
            return program
    return "Unassigned"


def row_dominant_program(row):
    scores = {p: 0.0 for p, _ in PROGRAM_RULES}
    scores["Unassigned"] = 0.0
    for feature, val in row.items():
        p = feature_program(feature)
        try:
            scores[p] += abs(float(val))
        except Exception:
            pass
    best = max(scores.items(), key=lambda x: x[1])
    return best[0] if best[1] > 0 else "Unassigned"


def parse_regulator_candidates(text):
    text = "" if pd.isna(text) else str(text)
    if not text.strip():
        return []

    tokens = []

    # Split database terms aggressively.
    for chunk in re.split(r"[,;/| \t]+", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        tokens.append(chunk)
        tokens.extend(re.split(r"[_\-]+", chunk))

    cleaned = []
    for tok in tokens:
        t = clean_token(tok)
        if not t:
            continue
        if t in BLACKLIST_TOKENS:
            continue
        if t in KNOWN_REGULATORS:
            cleaned.append(t)

    cleaned = list(dict.fromkeys(cleaned))
    cleaned = sorted(cleaned, key=lambda x: (len(x), x))
    return cleaned


def clean_regulator_label(row, max_n=3):
    # Try all regulator-related fields first.
    fields = []
    for c in row.index:
        lc = str(c).lower()
        if "reg" in lc or "tf" in lc or "factor" in lc:
            fields.append(str(row[c]))

    # Then try compact regulator and top features as fallback.
    for c in ["top_regulators_ultraclean", "top_regulators_compact", "top_regulators", "top_features"]:
        if c in row.index:
            fields.append(str(row[c]))

    candidates = []
    for text in fields:
        candidates.extend(parse_regulator_candidates(text))

    candidates = list(dict.fromkeys(candidates))[:max_n]
    return ", ".join(candidates) if candidates else "—"


def build_clean_row_annotation(mat, row_anno, max_regulators=3):
    row_anno = row_anno.copy()
    row_anno["track_id"] = row_anno["track_id"].astype(str)

    if "dominant_program" not in row_anno.columns:
        row_anno["dominant_program"] = [
            row_dominant_program(mat.loc[t]) if t in mat.index else "Unassigned"
            for t in row_anno["track_id"]
        ]

    # Recompute dominant program from visible matrix if previous label is Other/Unassigned.
    program_map = {}
    for t in mat.index:
        old = ""
        hit = row_anno[row_anno["track_id"] == t]
        if not hit.empty and "dominant_program" in hit.columns:
            old = str(hit["dominant_program"].iloc[0])
        if old in ["", "Other", "Unassigned", "nan", "NaN"]:
            program_map[t] = row_dominant_program(mat.loc[t])
        else:
            program_map[t] = old

    row_anno["dominant_program_clean"] = row_anno["track_id"].map(program_map).fillna("Unassigned")

    cleaned_regs = []
    discarded = []

    for _, r in row_anno.iterrows():
        label = clean_regulator_label(r, max_n=max_regulators)
        cleaned_regs.append(label)

        raw_text = " ".join([str(r[c]) for c in row_anno.columns if "reg" in str(c).lower() or c in ["top_features"]])
        raw_tokens = [clean_token(x) for x in re.split(r"[,;/|_\-\s]+", raw_text)]
        raw_tokens = [x for x in raw_tokens if x]
        kept = set(label.replace(",", " ").split()) if label != "—" else set()
        bad = sorted(set([x for x in raw_tokens if x in BLACKLIST_TOKENS or (x not in kept and x not in KNOWN_REGULATORS)]))
        discarded.append(";".join(bad[:30]))

    row_anno["top_regulators_final"] = cleaned_regs
    row_anno["discarded_regulator_tokens_audit"] = discarded

    row_anno = row_anno[row_anno["track_id"].isin(mat.index)].copy()
    row_anno["plot_order"] = row_anno["track_id"].map({t: i for i, t in enumerate(mat.index)})
    row_anno = row_anno.sort_values("plot_order").drop(columns=["plot_order"])

    return row_anno


def load_inputs(step73c_dir):
    step73c_dir = Path(step73c_dir)

    mat = read_csv(step73c_dir / "step73c_ultraclean_dynamic_feature_matrix_z.csv", required=True, index_col=0)
    raw = read_csv(step73c_dir / "step73c_ultraclean_dynamic_feature_matrix_raw.csv", required=False, index_col=0)
    row_anno = read_csv(step73c_dir / "step73c_ultraclean_track_annotations.csv", required=True)
    feature_anno = read_csv(step73c_dir / "step73c_ultraclean_feature_annotations.csv", required=True)

    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    mat = mat.loc[mat.abs().sum(axis=1) > 0, mat.abs().sum(axis=0) > 0]

    if raw.empty:
        raw = mat.copy()
    else:
        raw = raw.apply(pd.to_numeric, errors="coerce")
        raw = raw.reindex(index=mat.index, columns=mat.columns)

    return mat, raw, row_anno, feature_anno


def select_final_matrix(mat, feature_anno, max_features=20):
    feature_anno = feature_anno.copy()
    if "feature_name" not in feature_anno.columns:
        feature_anno = pd.DataFrame({"feature_name": list(mat.columns)})

    if "feature_type" not in feature_anno.columns:
        feature_anno["feature_type"] = "gene"

    if "program" not in feature_anno.columns:
        feature_anno["program"] = feature_anno["feature_name"].map(feature_program)

    score_rows = []
    for f in mat.columns:
        x = mat[f].astype(float).abs()
        hit = feature_anno[feature_anno["feature_name"].astype(str) == str(f)]
        ftype = hit["feature_type"].iloc[0] if not hit.empty else "gene"
        program = hit["program"].iloc[0] if not hit.empty else feature_program(f)

        score_rows.append({
            "feature_name": f,
            "feature_type": ftype,
            "program": program,
            "score": float(x.max() + 0.4 * x.mean() + 0.03 * (x > 0).sum()),
            "max_abs": float(x.max()),
            "n_tracks_nonzero": int((x > 0).sum()),
        })

    score_df = pd.DataFrame(score_rows)

    program_order = {p: i for i, (p, _) in enumerate(PROGRAM_RULES)}
    program_order["Unassigned"] = 999

    # Keep strongest features while preserving program diversity.
    selected = []
    for program, _ in PROGRAM_RULES:
        sub = score_df[score_df["program"] == program].sort_values("score", ascending=False)
        if not sub.empty:
            selected.append(sub["feature_name"].iloc[0])

    for f in score_df.sort_values("score", ascending=False)["feature_name"]:
        if f not in selected:
            selected.append(f)
        if len(selected) >= max_features:
            break

    selected = selected[:max_features]
    selected = sorted(
        selected,
        key=lambda f: (
            program_order.get(score_df.set_index("feature_name").loc[f, "program"], 999),
            0 if score_df.set_index("feature_name").loc[f, "feature_type"] == "module" else 1,
            -score_df.set_index("feature_name").loc[f, "score"],
            str(f)
        )
    )

    mat_final = mat[selected].copy()
    feature_final = score_df[score_df["feature_name"].isin(selected)].copy()
    feature_final["plot_order"] = feature_final["feature_name"].map({f: i for i, f in enumerate(selected)})
    feature_final = feature_final.sort_values("plot_order").drop(columns=["plot_order"])

    return mat_final, feature_final


def make_final_figure(mat, row_anno, feature_anno, outbase, clean=False, dpi=600):
    tracks = list(mat.index.astype(str))
    features = list(mat.columns.astype(str))
    data = mat.to_numpy(dtype=float)

    vmax = np.nanquantile(np.abs(data), 0.95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    row_map = row_anno.set_index("track_id").to_dict(orient="index")
    feature_map = feature_anno.set_index("feature_name").to_dict(orient="index")

    program_list = [p for p, _ in PROGRAM_RULES] + ["Unassigned"]
    program_to_idx = {p: i for i, p in enumerate(program_list)}
    program_cmap = ListedColormap([PROGRAM_COLORS[p] for p in program_list])

    ftype_list = ["gene", "module", "unknown"]
    ftype_to_idx = {p: i for i, p in enumerate(ftype_list)}
    ftype_cmap = ListedColormap([FEATURE_TYPE_COLORS[p] for p in ftype_list])

    row_prog_values = np.array([
        [program_to_idx.get(row_map.get(t, {}).get("dominant_program_clean", "Unassigned"), program_to_idx["Unassigned"])]
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
    })

    n_rows, n_cols = data.shape

    if clean:
        fig = plt.figure(figsize=(max(7.2, 0.36 * n_cols + 1.5), max(4.0, 0.42 * n_rows + 1.0)), facecolor="white")
        ax = fig.add_subplot(111)
        ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    else:
        fig = plt.figure(figsize=(max(12.0, 0.46 * n_cols + 5.0), max(5.5, 0.48 * n_rows + 2.3)), facecolor="white")
        gs = fig.add_gridspec(
            2, 3,
            height_ratios=[0.25, 8.0],
            width_ratios=[0.22, max(7.2, 0.46 * n_cols), 1.9],
            hspace=0.04,
            wspace=0.035,
        )

        ax_blank = fig.add_subplot(gs[0, 0])
        ax_top = fig.add_subplot(gs[0, 1])
        ax_blank2 = fig.add_subplot(gs[0, 2])
        ax_left = fig.add_subplot(gs[1, 0])
        ax = fig.add_subplot(gs[1, 1])
        ax_reg = fig.add_subplot(gs[1, 2])

        ax_blank.axis("off")
        ax_blank2.axis("off")

        # Top feature type strip.
        ax_top.imshow(col_type_values, cmap=ftype_cmap, aspect="auto")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        ax_top.set_title("Track-specific dynamic marker landscape",
                         fontsize=13.5, fontweight="bold", pad=8)
        for sp in ax_top.spines.values():
            sp.set_visible(False)

        # Left program strip.
        ax_left.imshow(row_prog_values, cmap=program_cmap, aspect="auto")
        ax_left.set_xticks([])
        ax_left.set_yticks([])
        ax_left.set_ylabel("Program", fontsize=8)
        for sp in ax_left.spines.values():
            sp.set_visible(False)

        # Heatmap.
        im = ax.imshow(data, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        ax.set_xticks(np.arange(n_cols))
        ax.set_xticklabels([short_text(f, 14) for f in features],
                           rotation=55, ha="right", fontsize=8)
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

        # Regulator text only. No "Other".
        ax_reg.set_xlim(0, 1)
        ax_reg.set_ylim(-0.5, n_rows - 0.5)
        ax_reg.invert_yaxis()
        ax_reg.axis("off")
        ax_reg.set_title("Regulators", fontsize=10, fontweight="bold", pad=8)

        for i, t in enumerate(tracks):
            if i % 2 == 0:
                ax_reg.add_patch(Rectangle((0, i - 0.5), 1, 1, facecolor="#f8fafc", edgecolor="none", zorder=0))
            regs = row_map.get(t, {}).get("top_regulators_final", "—")
            ax_reg.text(0.02, i, regs, ha="left", va="center", fontsize=8.8, color="#111827")

        # Program legend.
        legend_items = [p for p in program_list if p != "Unassigned"]
        x0, y0 = 0.04, 0.020
        fig.text(x0, y0 + 0.026, "Programs", fontsize=7.5, color="#334155", ha="left")

        for k, p in enumerate(legend_items):
            xx = x0 + k * 0.118
            fig.patches.append(Rectangle(
                (xx, y0), 0.012, 0.012,
                transform=fig.transFigure,
                facecolor=PROGRAM_COLORS[p],
                edgecolor="none"
            ))
            fig.text(xx + 0.015, y0 - 0.001, p, fontsize=6.7,
                     ha="left", va="bottom", color="#475569")

        fig.text(
            0.5,
            0.002,
            "Transcript-level dynamic markers/modules; regulator labels are cleaned enrichment-derived annotations, not functional validation.",
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

    ap.add_argument("--step73c_dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_features", type=int, default=20)
    ap.add_argument("--max_regulators", type=int, default=3)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    step73c_dir = Path(args.step73c_dir)

    log("=" * 100)
    log("Step73D | Label-cleaned final dynamic marker heatmap")
    log("=" * 100)
    log(f"step73c_dir={step73c_dir}")
    log(f"outdir={outdir}")

    mat, raw, row_anno, feature_anno = load_inputs(step73c_dir)

    mat_final, feature_final = select_final_matrix(
        mat,
        feature_anno,
        max_features=args.max_features
    )

    raw_final = raw.reindex(index=mat_final.index, columns=mat_final.columns)

    row_final = build_clean_row_annotation(
        mat_final,
        row_anno,
        max_regulators=args.max_regulators
    )

    # Save final tables.
    z_path = outdir / "step73d_labelclean_dynamic_feature_matrix_z.csv"
    raw_path = outdir / "step73d_labelclean_dynamic_feature_matrix_raw.csv"
    row_path = outdir / "step73d_labelclean_track_annotations.csv"
    col_path = outdir / "step73d_labelclean_feature_annotations.csv"

    mat_final.to_csv(z_path)
    raw_final.to_csv(raw_path)
    row_final.to_csv(row_path, index=False)
    feature_final.to_csv(col_path, index=False)

    annotated_base = outdir / "Fig_Step73D_LabelCleanedFinalTrackDynamicMarkerHeatmap_annotated"
    clean_base = outdir / "Fig_Step73D_LabelCleanedFinalTrackDynamicMarkerHeatmap_clean_no_text"

    make_final_figure(mat_final, row_final, feature_final, annotated_base, clean=False, dpi=args.dpi)
    make_final_figure(mat_final, row_final, feature_final, clean_base, clean=True, dpi=args.dpi)

    report = {
        "status": "ok",
        "analysis_name": "Step73D label-cleaned final track-specific dynamic marker heatmap",
        "inputs": {
            "step73c_dir": str(step73c_dir),
        },
        "n_tracks": int(mat_final.shape[0]),
        "n_features": int(mat_final.shape[1]),
        "selected_tracks": list(mat_final.index.astype(str)),
        "selected_features": list(mat_final.columns.astype(str)),
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
            "Step73D is the label-cleaned final version. It removes database-term residues from regulator labels "
            "and keeps only strict known regulator/TF symbols. Regulator labels remain enrichment-derived "
            "computational annotations, not functional validation."
        )
    }

    (outdir / "step73d_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "step73d_report.txt").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log("=" * 100)
    log("DONE Step73D")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
