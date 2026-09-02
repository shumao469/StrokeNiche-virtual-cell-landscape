#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step72D | Robustness check for Step72C expression-level in silico KO/blockade

Purpose
-------
Run Step72C across:
  1) label columns:
       region_auto, region_refined, region_manual_final, state_group
  2) KO strengths:
       ko_scale = 0.0, 0.2, 0.5

Then aggregate:
  - classifier CV stability
  - per-candidate delta core/peri/remote probability
  - rescue-like state shift score
  - direction consistency across label/KO settings
  - freeze recommendation

Interpretation
--------------
This is a robustness check for expression-level in silico counterfactuals.
It does not convert the result into wet-lab KO/blockade validation.
"""

from pathlib import Path
import argparse
import subprocess
import sys
import json
import re
import shutil
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def log(msg):
    print(msg, flush=True)


def ensure_dir(p):
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_csv_list(s):
    return [x.strip() for x in str(s).split(",") if x.strip()]


def parse_float_list(s):
    out = []
    for x in parse_csv_list(s):
        out.append(float(x))
    return out


def sanitize(x):
    x = str(x)
    x = x.replace(".", "p")
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x


def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def run_command(cmd, log_file, cwd=None):
    log_file = Path(log_file)
    with log_file.open("w", encoding="utf-8") as f:
        f.write("COMMAND:\n")
        f.write(" ".join(map(str, cmd)) + "\n\n")
        f.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
        proc.wait()

        f.write(f"\nRETURN_CODE={proc.returncode}\n")
        f.flush()

    return proc.returncode


def build_run_id(label_col, ko_scale):
    return f"label_{sanitize(label_col)}__ko_{sanitize(ko_scale)}"


def expected_outputs(run_dir):
    run_dir = Path(run_dir)
    return {
        "report": run_dir / "step72c_report.json",
        "classifier_audit": run_dir / "step72c_classifier_audit.json",
        "summary": run_dir / "step72c_expression_level_ko_state_shift_summary.csv",
        "candidate_audit": run_dir / "step72c_candidate_gene_audit.csv",
        "net_flow": run_dir / "step72c_expression_level_ko_net_flow_links.csv",
        "figure_png": run_dir / "Fig_Step72C_StrictExpressionLevelKO_StateShift_annotated.png",
        "figure_svg": run_dir / "Fig_Step72C_StrictExpressionLevelKO_StateShift_annotated.svg",
        "figure_pdf": run_dir / "Fig_Step72C_StrictExpressionLevelKO_StateShift_annotated.pdf",
    }


def is_successful_run(run_dir):
    outs = expected_outputs(run_dir)
    report = read_json(outs["report"])
    if report.get("status") == "ok" and outs["summary"].exists():
        return True
    return False


def run_step72c_one(args, label_col, ko_scale, run_dir):
    run_dir = ensure_dir(run_dir)
    outs = expected_outputs(run_dir)

    if args.skip_existing and is_successful_run(run_dir):
        log(f"[SKIP existing] {run_dir}")
        return 0

    cmd = [
        args.python,
        args.script72c,
        "--h5ad", args.h5ad,
        "--state_table", args.state_table,
        "--step65_genes", args.step65_genes,
        "--outdir", str(run_dir),
        "--label_col", label_col,
        "--candidates", args.candidates,
        "--top_flow_candidate", args.top_flow_candidate,
        "--max_feature_genes", str(args.max_feature_genes),
        "--ko_scale", str(ko_scale),
        "--up_shift_sd", str(args.up_shift_sd),
        "--simplex_arrow_scale", str(args.simplex_arrow_scale),
        "--flow_scale", str(args.flow_scale),
        "--min_abs_delta_for_main", str(args.min_abs_delta_for_main),
        "--dpi", str(args.dpi),
    ]

    if args.max_cells and args.max_cells > 0:
        cmd += ["--max_cells", str(args.max_cells)]

    if args.save_celllevel:
        cmd += ["--save_celllevel"]

    if args.dry_run:
        (run_dir / "DRY_RUN_command.sh").write_text(
            "#!/usr/bin/env bash\n" + " ".join(map(str, cmd)) + "\n",
            encoding="utf-8"
        )
        log("[DRY RUN] " + " ".join(map(str, cmd)))
        return 0

    log("=" * 100)
    log(f"RUN Step72C | label_col={label_col} | ko_scale={ko_scale}")
    log("=" * 100)
    return run_command(cmd, run_dir / f"run_step72c_{sanitize(label_col)}_ko{sanitize(ko_scale)}.log")


def extract_run_stats(run_dir, label_col, ko_scale):
    outs = expected_outputs(run_dir)
    report = read_json(outs["report"])
    audit = read_json(outs["classifier_audit"])

    audit_report = report.get("audit_report", audit)
    clf = audit_report.get("classifier_cv_audit", {})
    label_audit = audit_report.get("label_audit", {})
    feature_audit = audit_report.get("feature_audit", {})

    row = {
        "run_id": build_run_id(label_col, ko_scale),
        "label_col": label_col,
        "ko_scale": ko_scale,
        "run_dir": str(run_dir),
        "run_status": report.get("status", "missing_or_failed"),
        "effect_status": report.get("effect_status", audit_report.get("effect_status", "")),
        "max_abs_delta": report.get("max_abs_delta", audit_report.get("max_abs_delta", np.nan)),
        "classifier_status": clf.get("status", ""),
        "accuracy": clf.get("accuracy", np.nan),
        "balanced_accuracy": clf.get("balanced_accuracy", np.nan),
        "macro_f1": clf.get("macro_f1", np.nan),
        "n_splits": clf.get("n_splits", np.nan),
        "label_mode": label_audit.get("mode", ""),
        "label_counts_json": json.dumps(label_audit.get("counts", {}), ensure_ascii=False),
        "n_feature_genes": feature_audit.get("n_feature_genes", np.nan),
        "n_candidate_matched": feature_audit.get("n_candidate_matched", np.nan),
        "summary_exists": outs["summary"].exists(),
        "figure_exists": outs["figure_png"].exists(),
    }

    return row


def collect_all_runs(outdir, labels, ko_scales):
    run_stats = []
    summaries = []
    candidate_audits = []

    for label in labels:
        for scale in ko_scales:
            run_id = build_run_id(label, scale)
            run_dir = Path(outdir) / "runs" / run_id
            outs = expected_outputs(run_dir)

            run_stats.append(extract_run_stats(run_dir, label, scale))

            s = read_csv(outs["summary"])
            if not s.empty:
                s.insert(0, "run_id", run_id)
                s.insert(1, "label_col", label)
                s.insert(2, "ko_scale", scale)
                s.insert(3, "run_dir", str(run_dir))
                summaries.append(s)

            ca = read_csv(outs["candidate_audit"])
            if not ca.empty:
                ca.insert(0, "run_id", run_id)
                ca.insert(1, "label_col", label)
                ca.insert(2, "ko_scale", scale)
                ca.insert(3, "run_dir", str(run_dir))
                candidate_audits.append(ca)

    run_stats_df = pd.DataFrame(run_stats)
    all_summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    all_candidate_audit = pd.concat(candidate_audits, ignore_index=True) if candidate_audits else pd.DataFrame()

    return run_stats_df, all_summary, all_candidate_audit


def compute_consensus(all_summary, run_stats, effect_threshold=0.002, min_good_fraction=0.75):
    if all_summary.empty:
        return pd.DataFrame()

    good_runs = run_stats[
        (run_stats["run_status"] == "ok") &
        (pd.to_numeric(run_stats["balanced_accuracy"], errors="coerce") >= 0.60) &
        (pd.to_numeric(run_stats["macro_f1"], errors="coerce") >= 0.60)
    ]["run_id"].astype(str).tolist()

    df = all_summary[all_summary["run_id"].astype(str).isin(good_runs)].copy()
    if df.empty:
        df = all_summary.copy()

    rows = []
    for cid, sub in df.groupby("candidate_id", sort=False):
        dcore = pd.to_numeric(sub["delta_core_probability"], errors="coerce")
        dperi = pd.to_numeric(sub["delta_peri_probability"], errors="coerce")
        dremote = pd.to_numeric(sub["delta_remote_probability"], errors="coerce")
        score = pd.to_numeric(sub["state_shift_priority_score"], errors="coerce")

        n = len(sub)
        frac_core_down = float((dcore < -effect_threshold).mean()) if n else np.nan
        frac_peri_up = float((dperi > effect_threshold).mean()) if n else np.nan
        frac_remote_up = float((dremote > effect_threshold).mean()) if n else np.nan
        frac_score_pos = float((score > effect_threshold).mean()) if n else np.nan
        frac_counter = float((score < -effect_threshold).mean()) if n else np.nan

        median_score = float(np.nanmedian(score)) if n else np.nan
        median_dcore = float(np.nanmedian(dcore)) if n else np.nan
        median_dperi = float(np.nanmedian(dperi)) if n else np.nan
        median_dremote = float(np.nanmedian(dremote)) if n else np.nan

        if n == 0:
            tier = "failed_no_valid_run"
        elif frac_core_down >= min_good_fraction and frac_score_pos >= min_good_fraction and frac_remote_up >= min_good_fraction:
            tier = "robust_core_to_remote_rescue_like_shift"
        elif frac_core_down >= min_good_fraction and frac_score_pos >= min_good_fraction:
            tier = "robust_core_reduction_rescue_like_shift"
        elif frac_core_down >= 0.50 and frac_score_pos >= 0.50:
            tier = "partial_directional_support"
        elif frac_counter >= 0.50:
            tier = "counter_direction_or_potential_risk"
        else:
            tier = "inconsistent_or_near_zero"

        rows.append({
            "candidate_id": cid,
            "display_name": sub["display_name"].iloc[0] if "display_name" in sub else cid,
            "n_valid_runs_used": int(n),
            "n_total_runs_available": int(all_summary[all_summary["candidate_id"] == cid]["run_id"].nunique()),
            "median_rescue_score": median_score,
            "median_delta_core": median_dcore,
            "median_delta_peri": median_dperi,
            "median_delta_remote": median_dremote,
            "mean_rescue_score": float(np.nanmean(score)) if n else np.nan,
            "mean_delta_core": float(np.nanmean(dcore)) if n else np.nan,
            "mean_delta_peri": float(np.nanmean(dperi)) if n else np.nan,
            "mean_delta_remote": float(np.nanmean(dremote)) if n else np.nan,
            "frac_core_down": frac_core_down,
            "frac_peri_up": frac_peri_up,
            "frac_remote_up": frac_remote_up,
            "frac_rescue_score_positive": frac_score_pos,
            "frac_counter_direction": frac_counter,
            "min_rescue_score": float(np.nanmin(score)) if n else np.nan,
            "max_rescue_score": float(np.nanmax(score)) if n else np.nan,
            "consensus_tier": tier,
        })

    out = pd.DataFrame(rows)
    tier_order = {
        "robust_core_to_remote_rescue_like_shift": 1,
        "robust_core_reduction_rescue_like_shift": 2,
        "partial_directional_support": 3,
        "inconsistent_or_near_zero": 4,
        "counter_direction_or_potential_risk": 5,
        "failed_no_valid_run": 6,
    }
    out["_tier_order"] = out["consensus_tier"].map(tier_order).fillna(99)
    out = out.sort_values(
        ["_tier_order", "median_rescue_score", "frac_rescue_score_positive"],
        ascending=[True, False, False]
    ).drop(columns=["_tier_order"]).reset_index(drop=True)

    return out


def make_heatmap(df, value_col, outbase, title, cmap="RdBu_r", annotate=True):
    if df.empty or value_col not in df.columns:
        return

    plot_df = df.copy()
    plot_df["run_label"] = plot_df["label_col"].astype(str) + "\nKO=" + plot_df["ko_scale"].astype(str)
    plot_df["candidate_label"] = plot_df.get("display_name", plot_df["candidate_id"]).astype(str)

    pivot = plot_df.pivot_table(
        index="candidate_label",
        columns="run_label",
        values=value_col,
        aggfunc="mean"
    )

    if pivot.empty:
        return

    data = pivot.to_numpy(dtype=float)
    vmax = np.nanquantile(np.abs(data), 0.95)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.01

    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig_w = max(10, 0.85 * pivot.shape[1] + 4)
    fig_h = max(5, 0.42 * pivot.shape[0] + 2.2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    if annotate:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=7, color="#111827")

    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(value_col, fontsize=8)

    for sp in ax.spines.values():
        sp.set_visible(False)

    outbase = Path(outbase)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_classifier_stability_plot(run_stats, outbase):
    if run_stats.empty:
        return

    df = run_stats.copy()
    df["run_label"] = df["label_col"].astype(str) + "\nKO=" + df["ko_scale"].astype(str)
    df = df.sort_values(["label_col", "ko_scale"]).reset_index(drop=True)

    metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
    data = df[metrics].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(df) + 3), 4.2), facecolor="white")
    x = np.arange(len(df))

    for k, m in enumerate(metrics):
        ax.plot(x, data[:, k], marker="o", linewidth=1.4, label=m)

    ax.axhline(0.60, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
    ax.axhline(0.75, linestyle=":", linewidth=1.0, color="gray", alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["run_label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("CV metric")
    ax.set_title("Step72D classifier robustness across label columns and KO strengths",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    outbase = Path(outbase)
    fig.savefig(outbase.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(outbase.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_recommendation_text(run_stats, consensus, args):
    lines = []
    lines.append("Step72D robustness / freezing recommendation")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Design:")
    lines.append(f"- Label columns tested: {args.labels}")
    lines.append(f"- KO scales tested: {args.ko_scales}")
    lines.append("- Step72C was rerun for every label_col × ko_scale combination.")
    lines.append("")

    if not run_stats.empty:
        ok = run_stats[run_stats["run_status"] == "ok"].copy()
        lines.append("Run-level summary:")
        lines.append(f"- Successful runs: {len(ok)} / {len(run_stats)}")
        if len(ok):
            lines.append(f"- Minimum balanced accuracy: {pd.to_numeric(ok['balanced_accuracy'], errors='coerce').min():.3f}")
            lines.append(f"- Median balanced accuracy: {pd.to_numeric(ok['balanced_accuracy'], errors='coerce').median():.3f}")
            lines.append(f"- Minimum macro-F1: {pd.to_numeric(ok['macro_f1'], errors='coerce').min():.3f}")
            lines.append(f"- Median macro-F1: {pd.to_numeric(ok['macro_f1'], errors='coerce').median():.3f}")
            lines.append(f"- Maximum absolute state shift across runs: {pd.to_numeric(ok['max_abs_delta'], errors='coerce').max():.3f}")
        lines.append("")

    if not consensus.empty:
        lines.append("Candidate-level consensus:")
        for _, r in consensus.iterrows():
            lines.append(
                f"- {r['display_name']} ({r['candidate_id']}): {r['consensus_tier']}; "
                f"median rescue={r['median_rescue_score']:+.3f}, "
                f"median Δcore={r['median_delta_core']:+.3f}, "
                f"median Δremote={r['median_delta_remote']:+.3f}, "
                f"frac rescue-positive={r['frac_rescue_score_positive']:.2f}"
            )
        lines.append("")

        robust = consensus[
            consensus["consensus_tier"].isin([
                "robust_core_to_remote_rescue_like_shift",
                "robust_core_reduction_rescue_like_shift"
            ])
        ]
        partial = consensus[consensus["consensus_tier"].eq("partial_directional_support")]

        if len(robust):
            lines.append("Freeze recommendation:")
            lines.append("- FREEZE Step72C/72D as robust expression-level in silico counterfactual support.")
            lines.append("- Candidates in robust tiers can be described as robust KO/blockade-like computational counterfactuals.")
        elif len(partial):
            lines.append("Freeze recommendation:")
            lines.append("- FREEZE with cautious language: partial directional expression-level counterfactual support only.")
        else:
            lines.append("Freeze recommendation:")
            lines.append("- Do not use Step72 as a strong claim. Keep Step71C as sensitivity visualization only.")

    lines.append("")
    lines.append("Allowed wording:")
    lines.append("- expression-level in silico perturbation")
    lines.append("- KO/blockade-like computational counterfactual")
    lines.append("- robust/partial directional counterfactual support")
    lines.append("")
    lines.append("Forbidden wording:")
    lines.append("- wet-lab knockout validation")
    lines.append("- pharmacological blockade validation")
    lines.append("- observed cell-fate transition")
    lines.append("- experimentally validated therapeutic rescue")

    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--python", default="/home/shu/miniconda/envs/nicheformer_env/bin/python")
    ap.add_argument("--script72c", default="/mnt/h/vir/ST/72c_expression_level_insilico_ko_strict_label_classifier.py")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--state_table", required=True)
    ap.add_argument("--step65_genes", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--labels", default="region_auto,region_refined,region_manual_final,state_group")
    ap.add_argument("--ko_scales", default="0.0,0.2,0.5")

    ap.add_argument("--candidates", default="Ccl2_Ccr2_Ackr1_blockade,Spp1_Cd44_blockade,Vegfa_Flt1_Kdr_blockade,ferroptosis_down,repair_ECM_up")
    ap.add_argument("--top_flow_candidate", default="ferroptosis_down")

    ap.add_argument("--max_cells", type=int, default=0)
    ap.add_argument("--max_feature_genes", type=int, default=400)
    ap.add_argument("--up_shift_sd", type=float, default=1.0)
    ap.add_argument("--simplex_arrow_scale", type=float, default=12)
    ap.add_argument("--flow_scale", type=float, default=15)
    ap.add_argument("--min_abs_delta_for_main", type=float, default=0.002)
    ap.add_argument("--effect_threshold", type=float, default=0.002)
    ap.add_argument("--dpi", type=int, default=600)

    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--continue_on_error", action="store_true")
    ap.add_argument("--save_celllevel", action="store_true")

    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    runs_dir = ensure_dir(outdir / "runs")

    labels = parse_csv_list(args.labels)
    ko_scales = parse_float_list(args.ko_scales)

    log("=" * 100)
    log("Step72D | Robustness check for Step72C")
    log("=" * 100)
    log(f"outdir={outdir}")
    log(f"labels={labels}")
    log(f"ko_scales={ko_scales}")

    if not Path(args.script72c).exists():
        raise FileNotFoundError(f"Step72C script not found: {args.script72c}")

    manifest = {
        "analysis_name": "Step72D robustness check for Step72C",
        "created_at_unix": time.time(),
        "inputs": {
            "python": args.python,
            "script72c": args.script72c,
            "h5ad": args.h5ad,
            "state_table": args.state_table,
            "step65_genes": args.step65_genes,
        },
        "labels": labels,
        "ko_scales": ko_scales,
        "candidates": parse_csv_list(args.candidates),
        "run_dirs": [],
    }

    # Compile Step72C first.
    if not args.dry_run:
        compile_cmd = [args.python, "-m", "py_compile", args.script72c]
        rc = run_command(compile_cmd, outdir / "compile_step72c.log")
        if rc != 0:
            raise RuntimeError("Step72C script failed py_compile.")

    # Run all combinations.
    failures = []
    for label_col in labels:
        for ko_scale in ko_scales:
            run_id = build_run_id(label_col, ko_scale)
            run_dir = runs_dir / run_id
            manifest["run_dirs"].append(str(run_dir))

            rc = run_step72c_one(args, label_col, ko_scale, run_dir)
            if rc != 0:
                failures.append({
                    "run_id": run_id,
                    "label_col": label_col,
                    "ko_scale": ko_scale,
                    "run_dir": str(run_dir),
                    "return_code": rc,
                })
                if not args.continue_on_error:
                    raise RuntimeError(f"Step72C run failed: {run_id}")

    manifest["failures"] = failures
    (outdir / "step72d_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    if args.dry_run:
        log("DRY RUN completed. Commands were written into each run directory.")
        return

    # Aggregate.
    run_stats, all_summary, all_candidate_audit = collect_all_runs(outdir, labels, ko_scales)
    consensus = compute_consensus(
        all_summary,
        run_stats,
        effect_threshold=args.effect_threshold,
        min_good_fraction=0.75,
    )

    run_stats_path = outdir / "step72d_run_level_robustness_summary.csv"
    all_summary_path = outdir / "step72d_all_candidate_state_shift_summary.csv"
    candidate_audit_path = outdir / "step72d_all_candidate_gene_audit.csv"
    consensus_path = outdir / "step72d_candidate_robustness_consensus.csv"

    run_stats.to_csv(run_stats_path, index=False)
    all_summary.to_csv(all_summary_path, index=False)
    all_candidate_audit.to_csv(candidate_audit_path, index=False)
    consensus.to_csv(consensus_path, index=False)

    # Figures.
    figdir = ensure_dir(outdir / "figures")

    make_classifier_stability_plot(
        run_stats,
        figdir / "Fig_Step72D_ClassifierStabilityAcrossRobustnessRuns"
    )
    make_heatmap(
        all_summary,
        "state_shift_priority_score",
        figdir / "Fig_Step72D_RescueScoreRobustnessHeatmap",
        "Step72D rescue-score robustness across label columns and KO strengths"
    )
    make_heatmap(
        all_summary,
        "delta_core_probability",
        figdir / "Fig_Step72D_DeltaCoreRobustnessHeatmap",
        "Step72D Δcore robustness across label columns and KO strengths"
    )
    make_heatmap(
        all_summary,
        "delta_remote_probability",
        figdir / "Fig_Step72D_DeltaRemoteRobustnessHeatmap",
        "Step72D Δremote robustness across label columns and KO strengths"
    )

    recommendation = make_recommendation_text(run_stats, consensus, args)
    (outdir / "step72d_freeze_recommendation.txt").write_text(recommendation, encoding="utf-8")

    report = {
        "status": "ok",
        "analysis_name": "Step72D robustness check for Step72C",
        "n_label_columns": len(labels),
        "n_ko_scales": len(ko_scales),
        "n_planned_runs": len(labels) * len(ko_scales),
        "n_failures": len(failures),
        "failures": failures,
        "outputs": {
            "manifest": str(outdir / "step72d_manifest.json"),
            "run_level_summary": str(run_stats_path),
            "all_candidate_state_shift_summary": str(all_summary_path),
            "all_candidate_gene_audit": str(candidate_audit_path),
            "candidate_robustness_consensus": str(consensus_path),
            "freeze_recommendation": str(outdir / "step72d_freeze_recommendation.txt"),
            "figures_dir": str(figdir),
        },
        "interpretation_note": (
            "Step72D tests whether Step72C expression-level in silico KO/blockade-like counterfactuals "
            "are stable across region/state label definitions and KO strengths. Results remain computational "
            "counterfactuals, not wet-lab KO/blockade validation."
        )
    }

    (outdir / "step72d_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    (outdir / "step72d_report.txt").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    log("=" * 100)
    log("DONE Step72D")
    log("=" * 100)
    log(json.dumps(report["outputs"], indent=2, ensure_ascii=False))
    log("")
    log(recommendation)


if __name__ == "__main__":
    main()
