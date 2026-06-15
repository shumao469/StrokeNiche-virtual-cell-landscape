#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
66f_manuscript_ready_perturbation_summary.py

Purpose
-------
Create manuscript-ready clean summary tables and figures from Step66e rerun66c results.

Input:
  Step66e/rerun66c_probability_simplex/
    step66c_perturbation_fdr.csv
    step66c_track_level_perturbation_scores.csv
    step66c_module_target_mapped_candidates.csv
    step66c_report.json

Main outputs:
  step66f_manuscript_ready_perturbation_summary.csv
  step66f_top_exploratory_candidates.csv
  step66f_not_above_background_candidates.csv
  step66f_calibration_qc_candidates.csv
  step66f_track_contribution_summary.csv
  step66f_manuscript_caption_and_results_text.txt
  Fig_Step66F_ManuscriptReadyPerturbationSummary_annotated.pdf/svg/png
  Fig_Step66F_ManuscriptReadyPerturbationSummary_clean_no_text.pdf/svg/png

Interpretation tiers:
  1. FDR-supported candidate
  2. Above random background but not FDR-significant
  3. Not above random background
  4. Calibration / QC only

Important:
  This script does not claim therapeutic significance.
  It preserves the conservative conclusion:
    no FDR-supported candidate detected
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


ROOT = Path("/mnt/h/vir/ST/results/step8_strokeniche_perturbmap")
DEFAULT_IN66 = ROOT / "track_level_perturbation_fdr_66e_probability_simplex_rerun66c/rerun66c_probability_simplex"
DEFAULT_66E = ROOT / "track_level_perturbation_fdr_66e_probability_simplex_rerun66c"
DEFAULT_66B = ROOT / "track_level_perturbation_fdr_66b_strict_clean"
DEFAULT_OUT = ROOT / "track_level_perturbation_fdr_66f_manuscript_ready_summary"


HOUSEKEEPING_EXACT = {
    "ACTB", "B2M", "GAPDH", "HPRT", "HPRT1", "MALAT1", "PPIA",
    "RPLP0", "RPS18", "RPL13A", "TUBA1A", "TUBA1B", "TUBB",
    "EEF1A1", "EEF2", "PGK1", "LDHA", "LDHB",
}


def log(x):
    print(x, flush=True)


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def gene_upper(x):
    return str(x).strip().upper()


def read_csv_required(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(p)
    return pd.read_csv(p, low_memory=False)


def read_csv_optional(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.DataFrame()


def is_housekeeping_or_decoy(pid, target_genes=""):
    s = str(pid).upper()
    if "DECOY" in s or "NEGATIVE" in s or "CONTROL" in s:
        return True

    genes = []
    for x in re.split(r"[;,\s|/_:-]+", str(target_genes)):
        x = gene_upper(x)
        if x:
            genes.append(x)

    if genes and all(g in HOUSEKEEPING_EXACT or g.startswith(("RPL", "RPS", "MRPL", "MRPS", "MT-", "HBA", "HBB")) for g in genes):
        return True

    toks = [gene_upper(x) for x in re.split(r"[;,\s|/_:-]+", str(pid)) if x]
    if toks and any(t in HOUSEKEEPING_EXACT for t in toks) and ("DECOY" in s or len(toks) <= 2):
        return True

    return False


def display_name(pid):
    mapping = {
        "synaptic_recovery_up": "Synaptic recovery enhancement",
        "repair_ECM_up": "Repair-permissive ECM promotion",
        "ferroptosis_down": "Ferroptosis down-modulation",
        "astrocyte_reactive_down": "Reactive astrocyte down-modulation",
        "BBB_leakage_down": "BBB leakage down-modulation",
        "barrier_stability_up": "Barrier stability enhancement",
        "microglia_inflammatory_down": "Microglia inflammatory down-modulation",
        "inflammation_down": "Inflammation down-modulation",
        "hypoxia_down": "Hypoxia down-modulation",
        "endothelial_barrier_fragility_down": "Endothelial barrier fragility down-modulation",
    }
    return mapping.get(str(pid), str(pid).replace("_", " "))


def module_axis(module_key, pid):
    s = f"{module_key} {pid}".lower()
    if "synaptic" in s:
        return "synaptic recovery"
    if "repair_ecm" in s or "ecm" in s:
        return "repair / ECM remodeling"
    if "ferroptosis" in s:
        return "ferroptosis / redox"
    if "astrocyte" in s:
        return "astrocyte reactivity"
    if "bbb" in s or "barrier" in s or "endothelial" in s:
        return "vascular / BBB"
    if "microglia" in s or "inflammation" in s:
        return "inflammation / immune"
    if "hypoxia" in s:
        return "hypoxia / metabolism"
    return "other"


def assign_tier(row, fdr_cutoff=0.25, p_trend_cutoff=0.10):
    pid = row.get("perturbation_id", "")
    target_genes = row.get("target_genes", "")

    if is_housekeeping_or_decoy(pid, target_genes):
        return "Calibration / QC only"

    positive = bool(row.get("is_positive_over_background", False))
    sig = bool(row.get("is_fdr_significant_0_25", False))
    fdr = safe_float(row.get("bh_fdr"))
    emp_p = safe_float(row.get("empirical_p_value"))

    if positive and sig and fdr <= fdr_cutoff:
        return "FDR-supported candidate"

    if positive:
        if emp_p <= p_trend_cutoff:
            return "Above random background; nominal trend only"
        return "Above random background but not FDR-significant"

    return "Not above random background"


def tier_order(tier):
    order = {
        "FDR-supported candidate": 1,
        "Above random background; nominal trend only": 2,
        "Above random background but not FDR-significant": 3,
        "Not above random background": 4,
        "Calibration / QC only": 5,
    }
    return order.get(tier, 99)


def interpretation_sentence(row):
    tier = row["interpretation_tier"]
    name = row["display_name"]
    obs = row["s_overall_observed"]
    rand = row["random_mean"]
    emp = row["empirical_p_value"]
    fdr = row["bh_fdr"]

    if tier == "FDR-supported candidate":
        return (
            f"{name} exceeded the matched random-target background and survived BH-FDR correction "
            f"(observed score={obs:.4f}, random mean={rand:.4f}, empirical P={emp:.4g}, BH-FDR={fdr:.4g})."
        )
    if tier == "Above random background; nominal trend only":
        return (
            f"{name} showed a nominal positive trend above the matched random-target background "
            f"(observed score={obs:.4f}, random mean={rand:.4f}, empirical P={emp:.4g}), "
            f"but did not survive BH-FDR correction (BH-FDR={fdr:.4g})."
        )
    if tier == "Above random background but not FDR-significant":
        return (
            f"{name} scored above the random-background mean but did not reach FDR-supported significance "
            f"(observed score={obs:.4f}, random mean={rand:.4f}, empirical P={emp:.4g}, BH-FDR={fdr:.4g})."
        )
    if tier == "Not above random background":
        return (
            f"{name} did not exceed the matched random-target background "
            f"(observed score={obs:.4f}, random mean={rand:.4f}, empirical P={emp:.4g}, BH-FDR={fdr:.4g})."
        )
    return f"{name} was retained as calibration/QC only and was not included in final perturbation prioritization."


def summarize_track_contributions(track_scores, top_k=5):
    if track_scores.empty:
        return pd.DataFrame()

    rows = []

    for pid, sub in track_scores.groupby("perturbation_id"):
        sub = sub.copy()
        sub["s_track_score"] = pd.to_numeric(sub["s_track_score"], errors="coerce")
        sub["repair_gain_norm"] = pd.to_numeric(sub.get("repair_gain_norm", np.nan), errors="coerce")
        sub["core_reduction_norm"] = pd.to_numeric(sub.get("core_reduction_norm", np.nan), errors="coerce")
        sub["peri_remote_shift_norm"] = pd.to_numeric(sub.get("peri_remote_shift_norm", np.nan), errors="coerce")
        sub["target_relevance_norm"] = pd.to_numeric(sub.get("target_relevance_norm", np.nan), errors="coerce")
        sub["safety_penalty_norm"] = pd.to_numeric(sub.get("safety_penalty_norm", np.nan), errors="coerce")

        pos = sub.sort_values("s_track_score", ascending=False).head(top_k)
        neg = sub.sort_values("s_track_score", ascending=True).head(top_k)

        rows.append({
            "perturbation_id": pid,
            "top_positive_tracks": "; ".join(
                f"{r.track_id}:{safe_float(r.s_track_score):.3f}" for r in pos.itertuples()
            ),
            "top_negative_tracks": "; ".join(
                f"{r.track_id}:{safe_float(r.s_track_score):.3f}" for r in neg.itertuples()
            ),
            "mean_track_score": float(sub["s_track_score"].mean()),
            "max_track_score": float(sub["s_track_score"].max()),
            "min_track_score": float(sub["s_track_score"].min()),
            "mean_repair_component": float(sub["repair_gain_norm"].mean()),
            "mean_core_component": float(sub["core_reduction_norm"].mean()),
            "mean_state_component": float(sub["peri_remote_shift_norm"].mean()),
            "mean_target_component": float(sub["target_relevance_norm"].mean()),
            "mean_safety_penalty": float(sub["safety_penalty_norm"].mean()),
            "n_tracks": int(sub["track_id"].nunique()) if "track_id" in sub.columns else len(sub),
        })

    return pd.DataFrame(rows)


def build_summary(fdr, track_scores, report, fdr_cutoff=0.25, p_trend_cutoff=0.10):
    df = fdr.copy()

    for c in [
        "s_overall_observed", "random_mean", "observed_minus_random_mean",
        "random_z", "empirical_p_value", "bh_fdr", "target_set_size"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "is_positive_over_background" not in df.columns:
        df["is_positive_over_background"] = df["observed_minus_random_mean"] > 0

    if "is_fdr_significant_0_25" not in df.columns:
        df["is_fdr_significant_0_25"] = (df["bh_fdr"] <= fdr_cutoff) & df["is_positive_over_background"]

    df["display_name"] = df["perturbation_id"].map(display_name)
    df["biological_axis"] = df.apply(lambda r: module_axis(r.get("module_key", ""), r.get("perturbation_id", "")), axis=1)
    df["interpretation_tier"] = df.apply(lambda r: assign_tier(r, fdr_cutoff=fdr_cutoff, p_trend_cutoff=p_trend_cutoff), axis=1)
    df["interpretation_tier_order"] = df["interpretation_tier"].map(tier_order)
    df["manuscript_interpretation"] = df.apply(interpretation_sentence, axis=1)

    # Add analysis provenance.
    df["analysis_mode"] = "Step66e probability-simplex model-derived sensitivity analysis"
    df["score_mode"] = report.get("rerun66c_report", {}).get("score_mode", report.get("score_mode", ""))
    df["fdr_cutoff"] = fdr_cutoff
    df["trend_p_cutoff"] = p_trend_cutoff
    df["n_random"] = report.get("rerun66c_report", {}).get("n_random_per_candidate", report.get("n_random_per_candidate", np.nan))

    # Track contribution summary.
    contrib = summarize_track_contributions(track_scores)
    if not contrib.empty:
        df = df.merge(contrib, on="perturbation_id", how="left")

    # Sort manuscript order.
    df = df.sort_values(
        ["interpretation_tier_order", "empirical_p_value", "bh_fdr", "s_overall_observed"],
        ascending=[True, True, True, False]
    ).reset_index(drop=True)

    df["manuscript_rank"] = np.arange(1, len(df) + 1)

    keep_first = [
        "manuscript_rank",
        "perturbation_id",
        "display_name",
        "biological_axis",
        "module_key",
        "interpretation_tier",
        "s_overall_observed",
        "random_mean",
        "observed_minus_random_mean",
        "random_z",
        "empirical_p_value",
        "bh_fdr",
        "is_positive_over_background",
        "is_fdr_significant_0_25",
        "target_set_size",
        "target_genes",
        "top_positive_tracks",
        "top_negative_tracks",
        "mean_repair_component",
        "mean_core_component",
        "mean_state_component",
        "mean_target_component",
        "mean_safety_penalty",
        "manuscript_interpretation",
        "analysis_mode",
        "score_mode",
        "n_random",
    ]

    keep = [c for c in keep_first if c in df.columns]
    rest = [c for c in df.columns if c not in keep]
    df = df[keep + rest]

    return df


def load_report(in66, in66e):
    rep = {}
    p1 = Path(in66e) / "step66e_report.json"
    p2 = Path(in66) / "step66c_report.json"

    if p1.exists():
        try:
            rep = json.loads(p1.read_text())
            return rep
        except Exception:
            pass

    if p2.exists():
        try:
            return json.loads(p2.read_text())
        except Exception:
            return {}

    return {}


def write_text_outputs(summary, report, outdir):
    n_fdr = int((summary["interpretation_tier"] == "FDR-supported candidate").sum())
    n_positive = int(summary["is_positive_over_background"].sum()) if "is_positive_over_background" in summary.columns else 0
    n_total = int(len(summary))

    top = summary.head(3)

    lines = []
    lines.append("Step66f manuscript-ready perturbation summary")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Recommended manuscript statement:")
    lines.append(
        "Track-level perturbation scoring with matched random-background calibration identified "
        f"{n_positive} of {n_total} candidates with model-derived scores above the random mean, "
        f"but {n_fdr} candidates passed BH-FDR correction. "
        "Therefore, the leading perturbations should be interpreted as exploratory computational "
        "hypotheses for experimental prioritization rather than statistically confirmed therapeutic candidates."
    )
    lines.append("")
    lines.append("Leading exploratory candidates:")
    for _, r in top.iterrows():
        lines.append(
            f"- {r['display_name']} ({r['perturbation_id']}): "
            f"observed={r['s_overall_observed']:.4f}, random mean={r['random_mean']:.4f}, "
            f"empirical P={r['empirical_p_value']:.4g}, BH-FDR={r['bh_fdr']:.4g}; "
            f"tier={r['interpretation_tier']}."
        )
    lines.append("")
    lines.append("Figure legend draft:")
    lines.append(
        "Step66f summarizes track-level perturbation scores calibrated against matched random-target backgrounds. "
        "Candidate niche perturbations were ranked by model-derived overall score, empirical P value and BH-FDR. "
        "No perturbation survived BH-FDR correction in the probability-simplex sensitivity analysis. "
        "Candidates with scores above the random background, including synaptic recovery enhancement, "
        "repair-permissive ECM promotion and ferroptosis down-modulation, were retained as exploratory hypotheses."
    )
    lines.append("")
    lines.append("Important limitation:")
    lines.append(
        "The Step66e/66f analysis uses probability-simplex model-derived state-probability shifts based on "
        "baseline core/peri/remote probabilities and Step61-predicted perturbation response components. "
        "It should not be described as direct observed perturbed state-probability transition."
    )
    lines.append("")
    lines.append("Full manuscript-ready table preview:")
    preview_cols = [
        "manuscript_rank", "perturbation_id", "display_name", "interpretation_tier",
        "s_overall_observed", "random_mean", "observed_minus_random_mean",
        "empirical_p_value", "bh_fdr", "target_set_size", "module_key"
    ]
    preview_cols = [c for c in preview_cols if c in summary.columns]
    lines.append(summary[preview_cols].to_string(index=False))

    (Path(outdir) / "step66f_manuscript_caption_and_results_text.txt").write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


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


def plot_summary(summary, track_scores, outdir, dpi=600):
    outputs = {}

    tier_colors = {
        "FDR-supported candidate": 0,
        "Above random background; nominal trend only": 1,
        "Above random background but not FDR-significant": 2,
        "Not above random background": 3,
        "Calibration / QC only": 4,
    }

    for mode in ["annotated", "clean_no_text"]:
        annotate = mode == "annotated"
        outbase = Path(outdir) / f"Fig_Step66F_ManuscriptReadyPerturbationSummary_{mode}"

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, wspace=0.34, hspace=0.35)

        # Panel A: observed vs random
        axA = fig.add_subplot(gs[0, 0])
        df = summary.sort_values("manuscript_rank").copy()
        y = np.arange(len(df))[::-1]

        axA.barh(y, df["random_mean"].values[::-1], alpha=0.45, label="random mean")
        axA.barh(y, df["s_overall_observed"].values[::-1], alpha=0.75, label="observed")

        if annotate:
            axA.set_yticks(y)
            axA.set_yticklabels(df["display_name"].values[::-1], fontsize=8)
            axA.set_xlabel("Track-level perturbation score")
            axA.set_title("A | Observed score versus matched random background", loc="left", fontsize=12, fontweight="bold")
            axA.legend(frameon=False, fontsize=8)
        else:
            strip_text(axA)

        # Panel B: FDR scatter
        axB = fig.add_subplot(gs[0, 1])
        x = df["observed_minus_random_mean"]
        yv = -np.log10(df["bh_fdr"].clip(lower=1e-300))
        c = df["interpretation_tier"].map(tier_colors).fillna(4)

        axB.scatter(x, yv, s=60, c=c, alpha=0.8)

        if annotate:
            for _, r in df.iterrows():
                if r["manuscript_rank"] <= 5:
                    axB.text(
                        r["observed_minus_random_mean"],
                        -np.log10(max(r["bh_fdr"], 1e-300)),
                        str(r["perturbation_id"])[:24],
                        fontsize=7,
                    )
            axB.axvline(0, linestyle="--", linewidth=0.8)
            axB.axhline(-np.log10(0.25), linestyle="--", linewidth=0.8)
            axB.set_xlabel("Observed minus random mean")
            axB.set_ylabel("-log10(BH-FDR)")
            axB.set_title("B | Random-background calibration", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axB)

        # Panel C: interpretation tiers
        axC = fig.add_subplot(gs[1, 0])
        counts = df["interpretation_tier"].value_counts().reindex([
            "FDR-supported candidate",
            "Above random background; nominal trend only",
            "Above random background but not FDR-significant",
            "Not above random background",
            "Calibration / QC only",
        ]).dropna()

        axC.barh(np.arange(len(counts))[::-1], counts.values[::-1])
        if annotate:
            axC.set_yticks(np.arange(len(counts))[::-1])
            axC.set_yticklabels(counts.index[::-1], fontsize=8)
            axC.set_xlabel("Number of candidates")
            axC.set_title("C | Manuscript interpretation tiers", loc="left", fontsize=12, fontweight="bold")
        else:
            strip_text(axC)

        # Panel D: component summary
        axD = fig.add_subplot(gs[1, 1])
        comp_cols = [
            "mean_repair_component",
            "mean_core_component",
            "mean_state_component",
            "mean_target_component",
            "mean_safety_penalty",
        ]
        comp_cols = [c for c in comp_cols if c in df.columns]
        comp = df.set_index("perturbation_id")[comp_cols].fillna(0)

        if not comp.empty:
            xloc = np.arange(len(comp))
            bottom = np.zeros(len(comp))
            for col in [c for c in comp_cols if c != "mean_safety_penalty"]:
                vals = comp[col].values
                axD.bar(xloc, vals, bottom=bottom, label=col.replace("mean_", "").replace("_component", ""))
                bottom += vals
            if "mean_safety_penalty" in comp.columns:
                axD.bar(xloc, -comp["mean_safety_penalty"].values, label="safety penalty")

            if annotate:
                axD.set_xticks(xloc)
                axD.set_xticklabels([x[:18] for x in comp.index], rotation=45, ha="right", fontsize=7)
                axD.set_ylabel("Mean normalized component")
                axD.set_title("D | Score component summary", loc="left", fontsize=12, fontweight="bold")
                axD.legend(frameon=False, fontsize=7, ncol=2)
            else:
                strip_text(axD)
        else:
            if annotate:
                axD.text(0.5, 0.5, "No component summary", ha="center", va="center")
            else:
                strip_text(axD)

        for ax in [axA, axB, axC, axD]:
            if annotate:
                ax.grid(linestyle="--", linewidth=0.35, alpha=0.25)
            else:
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if annotate:
            fig.suptitle(
                "Step66f | Manuscript-ready perturbation summary after random-background FDR",
                fontsize=15,
                fontweight="bold",
            )

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


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in66", default=str(DEFAULT_IN66))
    ap.add_argument("--in66e", default=str(DEFAULT_66E))
    ap.add_argument("--in66b", default=str(DEFAULT_66B))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))

    ap.add_argument("--fdr_cutoff", type=float, default=0.25)
    ap.add_argument("--p_trend_cutoff", type=float, default=0.10)
    ap.add_argument("--dpi", type=int, default=600)

    args = ap.parse_args()

    in66 = Path(args.in66)
    in66e = Path(args.in66e)
    in66b = Path(args.in66b)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step66f manuscript-ready perturbation summary")
    log("=" * 100)
    log(f"in66={in66}")
    log(f"in66e={in66e}")
    log(f"outdir={outdir}")

    fdr = read_csv_required(in66 / "step66c_perturbation_fdr.csv")
    track_scores = read_csv_optional(in66 / "step66c_track_level_perturbation_scores.csv")
    candidates = read_csv_optional(in66 / "step66c_module_target_mapped_candidates.csv")
    report = load_report(in66, in66e)

    summary = build_summary(
        fdr=fdr,
        track_scores=track_scores,
        report=report,
        fdr_cutoff=args.fdr_cutoff,
        p_trend_cutoff=args.p_trend_cutoff,
    )

    # Optional calibration/QC table from 66b.
    qc = read_csv_optional(in66b / "step66b_excluded_decoy_housekeeping_candidates_qc.csv")
    if not qc.empty:
        qc["interpretation_tier"] = "Calibration / QC only"
        qc["display_name"] = qc["perturbation_id"].map(display_name) if "perturbation_id" in qc.columns else ""
        qc["manuscript_interpretation"] = qc.apply(
            lambda r: f"{r.get('perturbation_id', '')} was retained as calibration/QC only and excluded from final FDR interpretation.",
            axis=1,
        )

    # Subtables.
    top_exploratory = summary[
        summary["interpretation_tier"].isin([
            "Above random background; nominal trend only",
            "Above random background but not FDR-significant",
            "FDR-supported candidate",
        ])
    ].copy()

    not_above = summary[summary["interpretation_tier"].eq("Not above random background")].copy()

    contrib = summarize_track_contributions(track_scores)

    # Save.
    summary.to_csv(outdir / "step66f_manuscript_ready_perturbation_summary.csv", index=False)
    top_exploratory.to_csv(outdir / "step66f_top_exploratory_candidates.csv", index=False)
    not_above.to_csv(outdir / "step66f_not_above_background_candidates.csv", index=False)
    qc.to_csv(outdir / "step66f_calibration_qc_candidates.csv", index=False)
    contrib.to_csv(outdir / "step66f_track_contribution_summary.csv", index=False)

    write_text_outputs(summary, report, outdir)
    fig_outputs = plot_summary(summary, track_scores, outdir, dpi=args.dpi)

    tier_counts = summary["interpretation_tier"].value_counts().to_dict()

    report_out = {
        "status": "ok",
        "analysis_name": "Step66f manuscript-ready perturbation summary",
        "input_66c_dir": str(in66),
        "input_66e_dir": str(in66e),
        "outdir": str(outdir),
        "fdr_cutoff": args.fdr_cutoff,
        "p_trend_cutoff": args.p_trend_cutoff,
        "n_candidates": int(len(summary)),
        "tier_counts": tier_counts,
        "n_FDR_supported_candidates": int((summary["interpretation_tier"] == "FDR-supported candidate").sum()),
        "n_above_random_candidates": int(summary["is_positive_over_background"].sum()) if "is_positive_over_background" in summary.columns else None,
        "top_candidates": summary[[
            "perturbation_id", "display_name", "interpretation_tier",
            "s_overall_observed", "random_mean", "empirical_p_value", "bh_fdr"
        ]].head(5).to_dict(orient="records"),
        "figure_outputs": fig_outputs,
        "outputs": {
            "summary": str(outdir / "step66f_manuscript_ready_perturbation_summary.csv"),
            "top_exploratory": str(outdir / "step66f_top_exploratory_candidates.csv"),
            "not_above_background": str(outdir / "step66f_not_above_background_candidates.csv"),
            "calibration_qc": str(outdir / "step66f_calibration_qc_candidates.csv"),
            "track_contribution": str(outdir / "step66f_track_contribution_summary.csv"),
            "text": str(outdir / "step66f_manuscript_caption_and_results_text.txt"),
            "report_json": str(outdir / "step66f_report.json"),
            "report_txt": str(outdir / "step66f_report.txt"),
        },
        "interpretation_note": (
            "Step66f is a manuscript-ready summary of Step66e probability-simplex model-derived "
            "sensitivity analysis. No statistically significant therapeutic candidate should be claimed "
            "unless the FDR-supported tier is non-empty."
        ),
    }

    (outdir / "step66f_report.json").write_text(
        json.dumps(report_out, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step66f manuscript-ready perturbation summary report")
    lines.append("=" * 100)
    lines.append(json.dumps(report_out, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Manuscript-ready summary:")
    preview_cols = [
        "manuscript_rank", "perturbation_id", "display_name", "interpretation_tier",
        "s_overall_observed", "random_mean", "observed_minus_random_mean",
        "empirical_p_value", "bh_fdr", "target_set_size", "module_key"
    ]
    preview_cols = [c for c in preview_cols if c in summary.columns]
    lines.append(summary[preview_cols].to_string(index=False))
    lines.append("")
    lines.append("Interpretation text:")
    text_path = outdir / "step66f_manuscript_caption_and_results_text.txt"
    lines.append(text_path.read_text(encoding="utf-8"))

    (outdir / "step66f_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE Step66f")
    log("=" * 100)
    log(json.dumps(report_out, indent=2, ensure_ascii=False, default=str))
    log("")
    log(summary[preview_cols].to_string(index=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
