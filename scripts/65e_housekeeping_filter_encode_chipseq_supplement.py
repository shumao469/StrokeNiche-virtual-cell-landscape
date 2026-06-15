#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
65e_housekeeping_filter_encode_chipseq_supplement.py

Purpose
-------
Recompute Step65d ENCODE/ReMap/Literature ChIP-seq GMT supplement after removing
housekeeping / technical genes from:
  1. track-specific dynamic marker genes
  2. ChIP-seq GMT target sets
  3. overlap genes

This does NOT overwrite Step65c or Step65d.

Inputs
------
--step65_out:
  Step65c clean gene-level output directory.
  Required file:
    step65_track_dynamic_genes.csv

--step65d_out:
  Step65d ENCODE/ChIP-seq GMT supplement output directory.
  Required file:
    step65d_encode_chipseq_gmt_edges_parsed.csv
  Optional file:
    step65d_encode_chipseq_track_enrichment.csv

Outputs
-------
outdir/
  step65e_housekeeping_blacklist_used.csv
  step65e_dynamic_genes_cleaned_for_chipseq.csv
  step65e_encode_chipseq_gmt_edges_cleaned.csv
  step65e_encode_chipseq_track_enrichment.cleaned.csv
  step65e_encode_chipseq_track_summary.cleaned.csv
  step65e_removed_overlap_housekeeping_genes_from_65d.csv
  step65e_filter_audit_by_track.csv
  step65e_report.json
  step65e_report.txt
  Fig_Step65E_EncodeChipseq_HousekeepingFiltered_annotated.pdf/svg/png
  Fig_Step65E_EncodeChipseq_HousekeepingFiltered_clean_no_text.pdf/svg/png
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

try:
    from scipy.stats import hypergeom
    SCIPY_AVAILABLE = True
except Exception:
    hypergeom = None
    SCIPY_AVAILABLE = False


DEFAULT_STEP65_OUT = Path(
    "/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/"
    "track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata"
)

DEFAULT_STEP65D_OUT = Path(
    "/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/"
    "track_dynamic_marker_regulator_65d_encode_chipseq_gmt_supplement"
)

DEFAULT_OUT = Path(
    "/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/"
    "track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered"
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(x):
    print(x, flush=True)


def gene_upper(x):
    return str(x).strip().upper()


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def bh_fdr(pvals):
    p = np.asarray([safe_float(x, np.nan) for x in pvals], dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return out
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    m = len(ranked)
    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out[order] = np.clip(q, 0, 1)
    return out


def hypergeom_pval(M, K, n, x):
    if x <= 0:
        return 1.0
    if SCIPY_AVAILABLE and hypergeom is not None:
        return float(hypergeom.sf(x - 1, M, K, n))
    return float(1.0 / (1.0 + x))


# -----------------------------------------------------------------------------
# Housekeeping blacklist
# -----------------------------------------------------------------------------

def default_blacklist_exact():
    genes = [
        # classic housekeeping / technical
        "GAPDH", "ACTB", "B2M", "HPRT", "HPRT1", "PPIA", "RPLP0",
        "MALAT1", "XIST",

        # common additional housekeeping-like genes often dominating lists
        "TUBB", "TUBA1A", "TUBA1B", "TUBB2A", "TUBB2B", "TUBB3",
        "EEF1A1", "EEF1A2", "EEF2", "PGK1", "PKM", "LDHA", "LDHB",
        "HSP90AA1", "HSP90AB1", "HSPA1A", "HSPA1B", "HSPB1",

        # optional frequent high-abundance / stress technical genes
        "FOS", "JUN",
    ]
    return sorted(set(gene_upper(x) for x in genes))


def blacklist_reason(gene, extra_exact=None):
    """
    Return (is_blacklisted, reason)
    """
    g = str(gene).strip()
    gu = gene_upper(g)

    exact = set(default_blacklist_exact())
    if extra_exact:
        exact |= set(gene_upper(x) for x in extra_exact if str(x).strip())

    if gu in exact:
        return True, "exact_housekeeping_or_high_abundance"

    # mitochondrial
    if g.lower().startswith("mt-") or gu.startswith("MT-"):
        return True, "mitochondrial"

    # ribosomal
    if gu.startswith(("RPL", "RPS", "MRPL", "MRPS")):
        return True, "ribosomal"

    # hemoglobin
    if gu.startswith(("HBA", "HBB", "HBG", "HBD", "HBE", "HBZ", "HBQ", "HBM")):
        return True, "hemoglobin"

    # predicted Gm genes
    if re.match(r"^GM[0-9]+", gu):
        return True, "predicted_Gm_gene"

    return False, ""


def make_blacklist_table(extra_exact=None):
    rows = []
    for g in default_blacklist_exact():
        rows.append({
            "gene": g,
            "blacklist_type": "exact_housekeeping_or_high_abundance",
        })
    if extra_exact:
        for g in extra_exact:
            if str(g).strip():
                rows.append({
                    "gene": gene_upper(g),
                    "blacklist_type": "user_extra_exact",
                })
    for prefix, typ in [
        ("MT-", "mitochondrial_prefix"),
        ("RPL*", "ribosomal_prefix"),
        ("RPS*", "ribosomal_prefix"),
        ("MRPL*", "mitochondrial_ribosomal_prefix"),
        ("MRPS*", "mitochondrial_ribosomal_prefix"),
        ("HBA/HBB/HBG/HBD/HBE/HBZ/HBQ/HBM*", "hemoglobin_prefix"),
        ("GM[0-9]+", "predicted_Gm_regex"),
    ]:
        rows.append({"gene": prefix, "blacklist_type": typ})
    return pd.DataFrame(rows).drop_duplicates()


# -----------------------------------------------------------------------------
# Core calculation
# -----------------------------------------------------------------------------

def load_dynamic_genes(step65_out):
    p = Path(step65_out) / "step65_track_dynamic_genes.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    required = {"track_id", "feature", "dynamic_marker_score"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{p} lacks required columns: {missing}")
    df["feature_upper"] = df["feature"].map(gene_upper)
    df["dynamic_marker_score"] = pd.to_numeric(df["dynamic_marker_score"], errors="coerce")
    return df


def load_65d_edges(step65d_out):
    p = Path(step65d_out) / "step65d_encode_chipseq_gmt_edges_parsed.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p, low_memory=False)
    required = {"regulator", "term", "target", "source_name"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{p} lacks required columns: {missing}")
    df["target_upper"] = df["target"].map(gene_upper)
    return df


def load_old_65d_enrichment(step65d_out):
    p = Path(step65d_out) / "step65d_encode_chipseq_track_enrichment.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    return df


def filter_dynamic_genes(dyn, extra_exact=None, require_passes_min_cells=True):
    df = dyn.copy()

    if require_passes_min_cells and "passes_min_cells" in df.columns:
        df = df[df["passes_min_cells"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    reasons = df["feature_upper"].map(lambda g: blacklist_reason(g, extra_exact=extra_exact))
    df["is_housekeeping_blacklisted"] = [x[0] for x in reasons]
    df["blacklist_reason"] = [x[1] for x in reasons]

    clean = df[~df["is_housekeeping_blacklisted"]].copy()
    removed = df[df["is_housekeeping_blacklisted"]].copy()

    return clean, removed


def filter_edges(edges, extra_exact=None):
    df = edges.copy()
    reasons = df["target_upper"].map(lambda g: blacklist_reason(g, extra_exact=extra_exact))
    df["is_housekeeping_blacklisted"] = [x[0] for x in reasons]
    df["blacklist_reason"] = [x[1] for x in reasons]

    clean = df[~df["is_housekeeping_blacklisted"]].copy()
    removed = df[df["is_housekeeping_blacklisted"]].copy()

    return clean, removed


def audit_old_overlap_housekeeping(old_enr, extra_exact=None):
    if old_enr is None or old_enr.empty or "overlap_genes" not in old_enr.columns:
        return pd.DataFrame()

    rows = []
    for _, r in old_enr.iterrows():
        genes = [x for x in str(r.get("overlap_genes", "")).split(";") if x]
        bad = []
        reasons = []
        for g in genes:
            is_bad, reason = blacklist_reason(g, extra_exact=extra_exact)
            if is_bad:
                bad.append(gene_upper(g))
                reasons.append(reason)
        if bad:
            rows.append({
                "track_id": r.get("track_id", ""),
                "regulator": r.get("regulator", ""),
                "term": r.get("term", ""),
                "source_name": r.get("source_name", ""),
                "old_overlap_n": r.get("overlap_n", np.nan),
                "old_fdr_bh": r.get("fdr_bh", np.nan),
                "removed_housekeeping_genes": ";".join(sorted(set(bad))),
                "removed_reasons": ";".join(sorted(set(reasons))),
                "n_removed": len(set(bad)),
            })
    return pd.DataFrame(rows)


def recompute_enrichment(dynamic_clean, edges_clean, top_n_genes=50, min_overlap=2):
    """
    Use top dynamic genes per track after housekeeping filter.
    Recompute hypergeometric enrichment against cleaned ChIP-seq GMT edge sets.
    """
    if dynamic_clean.empty or edges_clean.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Universe after housekeeping filtering.
    universe = set(dynamic_clean["feature_upper"].unique()) | set(edges_clean["target_upper"].unique())
    M = len(universe)

    # Precompute GMT target sets.
    group_cols = ["regulator", "term", "source_name"]
    if "database" in edges_clean.columns:
        group_cols = ["regulator", "term", "source_name", "database"]

    target_sets = []
    for key, sub in edges_clean.groupby(group_cols):
        if len(group_cols) == 4:
            regulator, term, source_name, database = key
        else:
            regulator, term, source_name = key
            database = "ENCODE_ReMap_Literature_ChIPseq_GMT"

        targets = set(sub["target_upper"].unique()) & universe
        if len(targets) == 0:
            continue
        target_sets.append({
            "regulator": regulator,
            "term": term,
            "source_name": source_name,
            "database": database,
            "targets": targets,
            "target_gene_n": len(targets),
        })

    rows = []
    marker_rows = []

    for track_id, sub in dynamic_clean.groupby("track_id"):
        top = (
            sub.sort_values("dynamic_marker_score", ascending=False)
            .drop_duplicates(subset=["feature_upper"])
            .head(top_n_genes)
        )
        genes = set(top["feature_upper"].unique()) & universe
        n = len(genes)

        marker_rows.append({
            "track_id": track_id,
            "top_n_requested": top_n_genes,
            "marker_gene_n_after_filter": n,
            "marker_genes_after_filter": ";".join(sorted(genes)),
        })

        if n == 0:
            continue

        for ts in target_sets:
            targets = ts["targets"]
            K = ts["target_gene_n"]
            overlap = genes & targets
            x = len(overlap)
            if x < min_overlap:
                continue

            p = hypergeom_pval(M, K, n, x)

            rows.append({
                "track_id": track_id,
                "regulator": ts["regulator"],
                "term": ts["term"],
                "source_name": ts["source_name"],
                "database": ts["database"],
                "overlap_n": x,
                "marker_gene_n": n,
                "target_gene_n": K,
                "universe_n": M,
                "p_value": p,
                "overlap_genes": ";".join(sorted(overlap)),
            })

    enr = pd.DataFrame(rows)
    marker_gene_sets = pd.DataFrame(marker_rows)

    if not enr.empty:
        enr["fdr_bh"] = bh_fdr(enr["p_value"].values)
        enr["encode_chipseq_score"] = -np.log10(enr["fdr_bh"].clip(lower=1e-300)) * enr["overlap_n"]
        enr = enr.sort_values(
            ["fdr_bh", "p_value", "overlap_n"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

    # Summary per track.
    summary_rows = []
    if not enr.empty:
        for tid, sub in enr.groupby("track_id"):
            top = sub.sort_values(
                ["fdr_bh", "p_value", "overlap_n"],
                ascending=[True, True, False]
            ).head(10)
            summary_rows.append({
                "track_id": tid,
                "top_encode_chipseq_terms_housekeeping_filtered": ";".join(
                    f"{r.regulator}|{r.source_name}|overlap={r.overlap_n}|FDR={r.fdr_bh:.2g}|genes={r.overlap_genes}"
                    for r in top.itertuples()
                ),
                "n_encode_chipseq_terms_fdr_0_25": int((sub["fdr_bh"] <= 0.25).sum()),
                "best_regulator": top.iloc[0]["regulator"] if len(top) else "",
                "best_source_name": top.iloc[0]["source_name"] if len(top) else "",
                "best_fdr_bh": safe_float(top.iloc[0]["fdr_bh"]) if len(top) else np.nan,
            })
    summary = pd.DataFrame(summary_rows)

    return enr, summary, marker_gene_sets


def make_filter_audit(dynamic_raw, dynamic_clean, dynamic_removed, edges_raw, edges_clean, edges_removed, marker_sets):
    rows = []

    for tid, sub in dynamic_raw.groupby("track_id"):
        clean_sub = dynamic_clean[dynamic_clean["track_id"].eq(tid)]
        removed_sub = dynamic_removed[dynamic_removed["track_id"].eq(tid)]

        marker_sub = marker_sets[marker_sets["track_id"].eq(tid)] if marker_sets is not None and not marker_sets.empty else pd.DataFrame()

        rows.append({
            "track_id": tid,
            "dynamic_gene_rows_raw": int(len(sub)),
            "dynamic_gene_rows_clean": int(len(clean_sub)),
            "dynamic_gene_rows_removed_housekeeping": int(len(removed_sub)),
            "removed_housekeeping_genes": ";".join(sorted(removed_sub["feature_upper"].unique())[:100]),
            "marker_gene_n_after_filter": int(marker_sub["marker_gene_n_after_filter"].iloc[0]) if len(marker_sub) else 0,
        })

    audit = pd.DataFrame(rows)

    global_row = pd.DataFrame([{
        "track_id": "__GLOBAL__",
        "dynamic_gene_rows_raw": int(len(dynamic_raw)),
        "dynamic_gene_rows_clean": int(len(dynamic_clean)),
        "dynamic_gene_rows_removed_housekeeping": int(len(dynamic_removed)),
        "removed_housekeeping_genes": ";".join(sorted(dynamic_removed["feature_upper"].unique())[:200]),
        "marker_gene_n_after_filter": np.nan,
        "edge_rows_raw": int(len(edges_raw)),
        "edge_rows_clean": int(len(edges_clean)),
        "edge_rows_removed_housekeeping": int(len(edges_removed)),
    }])

    return pd.concat([global_row, audit], ignore_index=True)


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

def strip_text(ax):
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()


def plot_65e(enr, summary, audit, removed_old_overlap, outdir, top_n=20, dpi=600):
    outputs = {}

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step65E_EncodeChipseq_HousekeepingFiltered_{mode}"

        fig = plt.figure(figsize=(15, 9))
        gs = fig.add_gridspec(2, 2, wspace=0.32, hspace=0.34)

        # A top enrichment
        axA = fig.add_subplot(gs[0, 0])
        if not enr.empty:
            top = enr.head(top_n).copy()
            top["score"] = -np.log10(pd.to_numeric(top["fdr_bh"], errors="coerce").clip(lower=1e-300))
            y = np.arange(len(top))[::-1]
            axA.barh(y, top["score"].values[::-1])
            if annotate:
                labels = [
                    f"{r.track_id} | {str(r.regulator)[:25]}"
                    for r in top.itertuples()
                ][::-1]
                axA.set_yticks(y)
                axA.set_yticklabels(labels, fontsize=7)
                axA.set_xlabel("-log10(FDR)")
                axA.set_title("A | Top ChIP-seq enrichments after housekeeping filter", loc="left", fontsize=11, fontweight="bold")
            else:
                strip_text(axA)
        else:
            if annotate:
                axA.text(0.5, 0.5, "No enrichment after filter", ha="center", va="center")
            else:
                strip_text(axA)

        # B source counts
        axB = fig.add_subplot(gs[0, 1])
        if not enr.empty:
            counts = enr["source_name"].value_counts().head(12)
            y = np.arange(len(counts))[::-1]
            axB.barh(y, counts.values[::-1])
            if annotate:
                axB.set_yticks(y)
                axB.set_yticklabels(counts.index.tolist()[::-1], fontsize=7)
                axB.set_xlabel("Enrichment rows")
                axB.set_title("B | ChIP-seq source distribution", loc="left", fontsize=11, fontweight="bold")
            else:
                strip_text(axB)
        else:
            if annotate:
                axB.text(0.5, 0.5, "No source counts", ha="center", va="center")
            else:
                strip_text(axB)

        # C track summary
        axC = fig.add_subplot(gs[1, 0])
        if not summary.empty:
            s = summary.copy()
            s["score"] = -np.log10(pd.to_numeric(s["best_fdr_bh"], errors="coerce").clip(lower=1e-300))
            s = s.sort_values("score", ascending=False).head(20)
            y = np.arange(len(s))[::-1]
            axC.barh(y, s["score"].values[::-1])
            if annotate:
                labels = [
                    f"{r.track_id} | {str(r.best_regulator)[:25]}"
                    for r in s.itertuples()
                ][::-1]
                axC.set_yticks(y)
                axC.set_yticklabels(labels, fontsize=7)
                axC.set_xlabel("Best -log10(FDR)")
                axC.set_title("C | Best ChIP-seq support per track", loc="left", fontsize=11, fontweight="bold")
            else:
                strip_text(axC)
        else:
            if annotate:
                axC.text(0.5, 0.5, "No track summary", ha="center", va="center")
            else:
                strip_text(axC)

        # D audit removed old overlap
        axD = fig.add_subplot(gs[1, 1])
        if removed_old_overlap is not None and not removed_old_overlap.empty:
            counts = removed_old_overlap["removed_reasons"].value_counts().head(10)
            y = np.arange(len(counts))[::-1]
            axD.barh(y, counts.values[::-1])
            if annotate:
                axD.set_yticks(y)
                axD.set_yticklabels(counts.index.tolist()[::-1], fontsize=7)
                axD.set_xlabel("Old enrichment rows affected")
                axD.set_title("D | Housekeeping genes removed from old 65d overlaps", loc="left", fontsize=11, fontweight="bold")
            else:
                strip_text(axD)
        else:
            if annotate:
                axD.text(0.5, 0.5, "No old housekeeping overlaps detected", ha="center", va="center")
            else:
                strip_text(axD)

        for ax in [axA, axB, axC, axD]:
            if annotate:
                ax.grid(linestyle="--", linewidth=0.35, alpha=0.25)
            else:
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if annotate:
            fig.suptitle("Step 65e | ENCODE/ReMap/Literature ChIP-seq support after housekeeping filtering", fontsize=14, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.95] if annotate else [0, 0, 1, 1])

        fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
        fig.savefig(outbase.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        outputs[mode] = {
            "pdf": str(outbase.with_suffix(".pdf")),
            "svg": str(outbase.with_suffix(".svg")),
            "png": str(outbase.with_suffix(".png")),
        }

    return outputs


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step65_out", default=str(DEFAULT_STEP65_OUT))
    ap.add_argument("--step65d_out", default=str(DEFAULT_STEP65D_OUT))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--top_n_genes", type=int, default=50)
    ap.add_argument("--min_overlap", type=int, default=2)
    ap.add_argument("--require_passes_min_cells", action="store_true", default=True)
    ap.add_argument("--no_require_passes_min_cells", action="store_false", dest="require_passes_min_cells")

    ap.add_argument("--extra_blacklist", default="", help="Comma-separated extra exact gene symbols to blacklist.")
    ap.add_argument("--top_n_fig", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    step65_out = Path(args.step65_out)
    step65d_out = Path(args.step65d_out)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extra_exact = [x.strip() for x in args.extra_blacklist.split(",") if x.strip()]

    log("=" * 100)
    log("Step 65e: housekeeping-filtered ENCODE/ReMap/Literature ChIP-seq supplement")
    log("=" * 100)
    log(f"step65_out={step65_out}")
    log(f"step65d_out={step65d_out}")
    log(f"outdir={outdir}")

    blacklist_table = make_blacklist_table(extra_exact=extra_exact)
    blacklist_table.to_csv(outdir / "step65e_housekeeping_blacklist_used.csv", index=False)

    dynamic_raw = load_dynamic_genes(step65_out)
    edges_raw = load_65d_edges(step65d_out)
    old_enr = load_old_65d_enrichment(step65d_out)

    dynamic_clean, dynamic_removed = filter_dynamic_genes(
        dynamic_raw,
        extra_exact=extra_exact,
        require_passes_min_cells=args.require_passes_min_cells,
    )
    edges_clean, edges_removed = filter_edges(edges_raw, extra_exact=extra_exact)

    removed_old_overlap = audit_old_overlap_housekeeping(old_enr, extra_exact=extra_exact)

    enr, summary, marker_sets = recompute_enrichment(
        dynamic_clean=dynamic_clean,
        edges_clean=edges_clean,
        top_n_genes=args.top_n_genes,
        min_overlap=args.min_overlap,
    )

    audit = make_filter_audit(
        dynamic_raw=dynamic_raw,
        dynamic_clean=dynamic_clean,
        dynamic_removed=dynamic_removed,
        edges_raw=edges_raw,
        edges_clean=edges_clean,
        edges_removed=edges_removed,
        marker_sets=marker_sets,
    )

    # Save tables.
    dynamic_clean.to_csv(outdir / "step65e_dynamic_genes_cleaned_for_chipseq.csv", index=False)
    dynamic_removed.to_csv(outdir / "step65e_dynamic_genes_removed_housekeeping.csv", index=False)

    edges_clean.to_csv(outdir / "step65e_encode_chipseq_gmt_edges_cleaned.csv", index=False)
    edges_removed.to_csv(outdir / "step65e_encode_chipseq_gmt_edges_removed_housekeeping.csv", index=False)

    removed_old_overlap.to_csv(outdir / "step65e_removed_overlap_housekeeping_genes_from_65d.csv", index=False)

    marker_sets.to_csv(outdir / "step65e_track_marker_genes_after_filter.csv", index=False)
    enr.to_csv(outdir / "step65e_encode_chipseq_track_enrichment.cleaned.csv", index=False)
    summary.to_csv(outdir / "step65e_encode_chipseq_track_summary.cleaned.csv", index=False)
    audit.to_csv(outdir / "step65e_filter_audit_by_track.csv", index=False)

    fig_outputs = plot_65e(
        enr=enr,
        summary=summary,
        audit=audit,
        removed_old_overlap=removed_old_overlap,
        outdir=outdir,
        top_n=args.top_n_fig,
        dpi=args.dpi,
    )

    report = {
        "status": "ok",
        "analysis_name": "Step65e housekeeping-filtered ENCODE/ReMap/Literature ChIP-seq supplement",
        "step65_out": str(step65_out),
        "step65d_out": str(step65d_out),
        "outdir": str(outdir),
        "top_n_genes": args.top_n_genes,
        "min_overlap": args.min_overlap,
        "scipy_available": SCIPY_AVAILABLE,

        "dynamic_gene_rows_raw": int(len(dynamic_raw)),
        "dynamic_gene_rows_clean": int(len(dynamic_clean)),
        "dynamic_gene_rows_removed_housekeeping": int(len(dynamic_removed)),
        "dynamic_gene_unique_removed_housekeeping": int(dynamic_removed["feature_upper"].nunique()) if not dynamic_removed.empty else 0,

        "edge_rows_raw": int(len(edges_raw)),
        "edge_rows_clean": int(len(edges_clean)),
        "edge_rows_removed_housekeeping": int(len(edges_removed)),

        "old_65d_enrichment_rows_with_housekeeping_overlap": int(len(removed_old_overlap)),
        "new_enrichment_rows": int(len(enr)),
        "tracks_with_chipseq_support_after_filter": int(summary["track_id"].nunique()) if not summary.empty else 0,

        "source_counts_after_filter": enr["source_name"].value_counts().head(30).to_dict() if not enr.empty else {},
        "top_regulators_after_filter": enr["regulator"].value_counts().head(30).to_dict() if not enr.empty else {},

        "figure_outputs": fig_outputs,
        "outputs": {
            "blacklist": str(outdir / "step65e_housekeeping_blacklist_used.csv"),
            "dynamic_genes_cleaned": str(outdir / "step65e_dynamic_genes_cleaned_for_chipseq.csv"),
            "dynamic_genes_removed": str(outdir / "step65e_dynamic_genes_removed_housekeeping.csv"),
            "edges_cleaned": str(outdir / "step65e_encode_chipseq_gmt_edges_cleaned.csv"),
            "edges_removed": str(outdir / "step65e_encode_chipseq_gmt_edges_removed_housekeeping.csv"),
            "removed_old_overlap": str(outdir / "step65e_removed_overlap_housekeeping_genes_from_65d.csv"),
            "marker_genes_after_filter": str(outdir / "step65e_track_marker_genes_after_filter.csv"),
            "enrichment_cleaned": str(outdir / "step65e_encode_chipseq_track_enrichment.cleaned.csv"),
            "summary_cleaned": str(outdir / "step65e_encode_chipseq_track_summary.cleaned.csv"),
            "audit_by_track": str(outdir / "step65e_filter_audit_by_track.csv"),
            "report_json": str(outdir / "step65e_report.json"),
            "report_txt": str(outdir / "step65e_report.txt"),
        },
        "interpretation_note": (
            "This filtered supplement should be used for ENCODE/ReMap/Literature ChIP-seq support. "
            "Raw ENCODE metadata was not used. Housekeeping/technical genes were removed before "
            "recomputing marker-set overlaps and hypergeometric enrichment."
        ),
    }

    (outdir / "step65e_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step65e housekeeping-filtered ChIP-seq supplement report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Top cleaned enrichment:")
    if not enr.empty:
        lines.append(enr.head(120).to_string(index=False))
    else:
        lines.append("No enrichment after housekeeping filter.")
    lines.append("")
    lines.append("Removed old 65d overlap housekeeping genes:")
    if not removed_old_overlap.empty:
        lines.append(removed_old_overlap.head(120).to_string(index=False))
    else:
        lines.append("None detected.")
    lines.append("")
    lines.append("Filter audit:")
    lines.append(audit.head(80).to_string(index=False))

    (outdir / "step65e_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step65e")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    log("")
    log("Top cleaned enrichment:")
    log(enr.head(50).to_string(index=False) if not enr.empty else "No enrichment after filter.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
