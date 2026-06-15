#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
66e_probability_simplex_patch_and_rerun66c.py

Purpose
-------
Create model-derived baseline -> perturbed core/peri/remote probabilities
from existing Step61 cell-level outputs, then rerun Step66c.

This is NOT a true observed state-probability delta.
It is a probability-simplex, model-derived sensitivity patch using:

  baseline:
    core_probability
    peri_probability
    remote_probability

  perturbation response summaries:
    predicted_core_reversal
    predicted_repair_shift
    remote_offtarget_penalty
    predicted_spatial_safety / neuron_context_risk if available

The output is suitable as:
  model-derived state probability sensitivity analysis

Not suitable to claim:
  direct cell-state probability transition
  true perturbed latent-coordinate transition

Outputs
-------
outdir/
  step66e_step61_cell_level_probability_simplex_patched.csv
  step66e_probability_simplex_patch_audit.csv
  step66e_run_66c_probability_simplex.sh
  rerun66c_probability_simplex/
"""

from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd


BASE = Path("/mnt/h/vir/ST")
ROOT = BASE / "results/step8_strokeniche_perturbmap"

DEFAULT_STEP61_CELL = ROOT / "niche_perturb_61/niche_perturbation_cell_level_state_editing.state_fixed.csv"
DEFAULT_STEP61_RANKING = ROOT / "niche_perturb_61/niche_perturbation_ranking.csv"
DEFAULT_TRACK_DIR = ROOT / "dynamics_graph_64c_refined_labels_state_celltype"
DEFAULT_STEP65_DIR = ROOT / "track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata"
DEFAULT_STEP65E_DIR = ROOT / "track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered"
DEFAULT_66C_SCRIPT = BASE / "66c_celllevel_periremote_module_target_fdr.py"
DEFAULT_OUT = ROOT / "track_level_perturbation_fdr_66e_probability_simplex_rerun66c"


def log(x):
    print(x, flush=True)


def norm_key(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def read_csv_auto(path):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(p)
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.read_csv(p, sep="\t", low_memory=False)


def safe_copy_no_metadata(src, dst):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            dst.unlink()
        except Exception:
            pass
    shutil.copyfile(src, dst)
    return dst


def shell_quote(x):
    s = str(x)
    if re.match(r"^[A-Za-z0-9_/\.\-:]+$", s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def find_numeric_col(df, patterns, required=False):
    for pat in patterns:
        for c in df.columns:
            nk = norm_key(c)
            if re.search(pat, nk):
                v = pd.to_numeric(df[c], errors="coerce")
                if v.notna().mean() > 0.20:
                    return c
    if required:
        raise RuntimeError(f"Missing required numeric column matching: {patterns}")
    return ""


def minmax_series(x):
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx <= mn:
        return pd.Series(np.full(len(s), 0.5), index=s.index)
    return (s - mn) / (mx - mn)


def group_minmax(df, group_col, value_col):
    out = pd.Series(index=df.index, dtype=float)
    for _, idx in df.groupby(group_col).groups.items():
        out.loc[idx] = minmax_series(df.loc[idx, value_col]).values
    return out.fillna(0.5)


def row_softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - np.nanmax(z, axis=1, keepdims=True)
    ez = np.exp(z)
    denom = np.nansum(ez, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return ez / denom


def normalize_probabilities(df, core_col, peri_col, remote_col):
    p = df[[core_col, peri_col, remote_col]].apply(pd.to_numeric, errors="coerce").fillna(0).values
    p[p < 0] = 0
    rowsum = p.sum(axis=1, keepdims=True)
    bad = rowsum[:, 0] <= 1e-12
    if bad.any():
        p[bad, :] = np.array([1/3, 1/3, 1/3])
        rowsum = p.sum(axis=1, keepdims=True)
    return p / rowsum


def probability_simplex_patch(cell, shift_scale=0.75, max_abs_delta=0.60):
    df = cell.copy()

    required = [
        "obs_name",
        "perturbation_id",
        "core_probability",
        "peri_probability",
        "remote_probability",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Step61 cell-level table is missing required columns: {missing}")

    df["obs_name"] = df["obs_name"].astype(str)
    df["perturbation_id"] = df["perturbation_id"].astype(str)

    core_col = "core_probability"
    peri_col = "peri_probability"
    remote_col = "remote_probability"

    repair_col = find_numeric_col(df, [
        r"predicted_repair_shift",
        r"repair.*shift",
        r"repair.*gain",
        r"delta.*repair",
    ], required=True)

    core_rev_col = find_numeric_col(df, [
        r"predicted_core_reversal",
        r"core.*reversal",
        r"core.*reduction",
        r"core.*decrease",
        r"delta.*core",
    ], required=True)

    safety_penalty_col = find_numeric_col(df, [
        r"remote_offtarget_penalty",
        r"off.*target.*penalty",
        r"safety.*penalty",
        r"risk.*penalty",
        r"toxicity",
    ], required=False)

    spatial_safety_col = find_numeric_col(df, [
        r"predicted_spatial_safety",
        r"spatial.*safety",
        r"safety.*score",
    ], required=False)

    neuron_risk_col = find_numeric_col(df, [
        r"neuron_context_risk",
        r"neuron.*risk",
        r"synapse.*risk",
    ], required=False)

    edit_col = find_numeric_col(df, [
        r"edit_intensity",
        r"editing_intensity",
        r"perturbation_strength",
        r"strength",
    ], required=False)

    p0 = normalize_probabilities(df, core_col, peri_col, remote_col)

    df["baseline_core_probability"] = p0[:, 0]
    df["baseline_peri_probability"] = p0[:, 1]
    df["baseline_remote_probability"] = p0[:, 2]

    # Normalize response components within each perturbation so a large global module
    # does not dominate purely by scale.
    df["_core_reversal_norm_66e"] = group_minmax(df, "perturbation_id", core_rev_col)
    df["_repair_shift_norm_66e"] = group_minmax(df, "perturbation_id", repair_col)

    if safety_penalty_col:
        df["_safety_penalty_norm_66e"] = group_minmax(df, "perturbation_id", safety_penalty_col)
    elif spatial_safety_col:
        df["_safety_penalty_norm_66e"] = 1.0 - group_minmax(df, "perturbation_id", spatial_safety_col)
    else:
        df["_safety_penalty_norm_66e"] = 0.0

    if neuron_risk_col:
        df["_neuron_risk_norm_66e"] = group_minmax(df, "perturbation_id", neuron_risk_col)
    else:
        df["_neuron_risk_norm_66e"] = 0.0

    if edit_col:
        df["_edit_intensity_norm_66e"] = group_minmax(df, "perturbation_id", edit_col)
    else:
        df["_edit_intensity_norm_66e"] = 1.0

    core_rev = df["_core_reversal_norm_66e"].values
    repair = df["_repair_shift_norm_66e"].values
    safety = df["_safety_penalty_norm_66e"].values
    neuron_risk = df["_neuron_risk_norm_66e"].values
    edit = df["_edit_intensity_norm_66e"].values

    # Probability-simplex perturbation model:
    # - core reversal decreases lesion-core probability.
    # - core reversal mainly shifts core -> peri.
    # - repair shift promotes peri/remote repair-like probability.
    # - off-target / neuron risk penalizes remote gain and weakly increases core persistence.
    shift_core = (
        -0.85 * core_rev
        -0.30 * repair
        +0.25 * safety
        +0.15 * neuron_risk
    )

    shift_peri = (
        +0.55 * core_rev
        +0.15 * repair
        -0.10 * safety
    )

    shift_remote = (
        +0.20 * core_rev
        +0.75 * repair
        -0.30 * safety
        -0.20 * neuron_risk
    )

    shifts = np.vstack([shift_core, shift_peri, shift_remote]).T
    shifts = shift_scale * edit[:, None] * shifts

    # Work on log probability; then softmax to preserve simplex.
    eps = 1e-6
    logits0 = np.log(np.clip(p0, eps, 1.0))
    p1_raw = row_softmax(logits0 + shifts)

    # Limit maximum absolute probability change per state to avoid artificial over-editing.
    delta = p1_raw - p0
    delta = np.clip(delta, -max_abs_delta, max_abs_delta)

    p1 = p0 + delta
    p1[p1 < eps] = eps
    p1 = p1 / p1.sum(axis=1, keepdims=True)

    df["perturbed_core_probability"] = p1[:, 0]
    df["perturbed_peri_probability"] = p1[:, 1]
    df["perturbed_remote_probability"] = p1[:, 2]

    df["predicted_delta_core_probability"] = df["perturbed_core_probability"] - df["baseline_core_probability"]
    df["predicted_delta_peri_probability"] = df["perturbed_peri_probability"] - df["baseline_peri_probability"]
    df["predicted_delta_remote_probability"] = df["perturbed_remote_probability"] - df["baseline_remote_probability"]

    df["predicted_peri_remote_shift"] = (
        df["predicted_delta_peri_probability"]
        + df["predicted_delta_remote_probability"]
    )

    df["true_core_probability_reduction"] = (
        df["baseline_core_probability"]
        - df["perturbed_core_probability"]
    )

    # Aliases used by Step66c.
    df["delta_peri_probability"] = df["predicted_delta_peri_probability"]
    df["delta_remote_probability"] = df["predicted_delta_remote_probability"]
    df["peri_remote_shift"] = df["predicted_peri_remote_shift"]

    df["core_probability_reduction_model_derived"] = df["true_core_probability_reduction"]
    df["step66e_stateprob_patch_mode"] = "probability_simplex_model_derived"

    audit = {
        "status": "ok",
        "patch_mode": "probability_simplex_model_derived",
        "n_rows": int(len(df)),
        "n_perturbations": int(df["perturbation_id"].nunique()),
        "baseline_probability_cols": [core_col, peri_col, remote_col],
        "repair_col": repair_col,
        "core_reversal_col": core_rev_col,
        "safety_penalty_col": safety_penalty_col,
        "spatial_safety_col": spatial_safety_col,
        "neuron_risk_col": neuron_risk_col,
        "edit_intensity_col": edit_col,
        "shift_scale": float(shift_scale),
        "max_abs_delta": float(max_abs_delta),
        "mean_delta_core": float(df["predicted_delta_core_probability"].mean()),
        "mean_delta_peri": float(df["predicted_delta_peri_probability"].mean()),
        "mean_delta_remote": float(df["predicted_delta_remote_probability"].mean()),
        "sd_delta_core": float(df["predicted_delta_core_probability"].std()),
        "sd_delta_peri": float(df["predicted_delta_peri_probability"].std()),
        "sd_delta_remote": float(df["predicted_delta_remote_probability"].std()),
        "peri_remote_shift_sd": float(df["predicted_peri_remote_shift"].std()),
        "probability_sum_min": float(df[["perturbed_core_probability", "perturbed_peri_probability", "perturbed_remote_probability"]].sum(axis=1).min()),
        "probability_sum_max": float(df[["perturbed_core_probability", "perturbed_peri_probability", "perturbed_remote_probability"]].sum(axis=1).max()),
        "interpretation": (
            "This is model-derived probability-simplex sensitivity analysis. "
            "It is not a direct observed perturbed probability delta."
        ),
    }

    if audit["peri_remote_shift_sd"] <= 1e-12:
        raise RuntimeError("Generated peri_remote_shift has zero variance; refusing to continue.")

    return df, audit


def write_run66c_script(args, patched_cell_path, mini_step61, out66c):
    run_sh = Path(args.outdir) / "step66e_run_66c_probability_simplex.sh"

    cmd = [
        args.python,
        str(args.script66c),
        "--track_dir", str(args.track_dir),
        "--step61_dir", str(mini_step61),
        "--step61_cell", str(patched_cell_path),
        "--step61_ranking", str(mini_step61 / "niche_perturbation_ranking.csv"),
        "--step65_dir", str(args.step65_dir),
        "--step65e_dir", str(args.step65e_dir),
        "--outdir", str(out66c),
        "--n_random", str(args.n_random),
        "--seed", str(args.seed),
        "--w_repair", str(args.w_repair),
        "--w_core", str(args.w_core),
        "--w_state", str(args.w_state),
        "--w_target", str(args.w_target),
        "--w_safety", str(args.w_safety),
        "--top_n_heatmap", str(args.top_n_heatmap),
        "--top_n_fig", str(args.top_n_fig),
        "--dpi", str(args.dpi),
    ]

    with open(run_sh, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write("cd /mnt/h/vir/ST\n\n")
        f.write(" ".join(shell_quote(x) for x in cmd))
        f.write(f" 2>&1 | tee {shell_quote(Path(args.outdir) / 'run_66e_rerun66c_probability_simplex.log')}\n")

    try:
        os.chmod(run_sh, 0o755)
    except Exception:
        pass

    return run_sh, cmd


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--step61_cell", default=str(DEFAULT_STEP61_CELL))
    ap.add_argument("--step61_ranking", default=str(DEFAULT_STEP61_RANKING))
    ap.add_argument("--track_dir", default=str(DEFAULT_TRACK_DIR))
    ap.add_argument("--step65_dir", default=str(DEFAULT_STEP65_DIR))
    ap.add_argument("--step65e_dir", default=str(DEFAULT_STEP65E_DIR))
    ap.add_argument("--script66c", default=str(DEFAULT_66C_SCRIPT))
    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--python", default="/home/shu/miniconda/envs/nicheformer_env/bin/python")

    ap.add_argument("--shift_scale", type=float, default=0.75)
    ap.add_argument("--max_abs_delta", type=float, default=0.60)

    ap.add_argument("--n_random", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260601)

    ap.add_argument("--w_repair", type=float, default=0.30)
    ap.add_argument("--w_core", type=float, default=0.30)
    ap.add_argument("--w_state", type=float, default=0.15)
    ap.add_argument("--w_target", type=float, default=0.20)
    ap.add_argument("--w_safety", type=float, default=0.25)

    ap.add_argument("--top_n_heatmap", type=int, default=30)
    ap.add_argument("--top_n_fig", type=int, default=25)
    ap.add_argument("--dpi", type=int, default=600)

    ap.add_argument("--run66c", action="store_true")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step66e probability-simplex model-derived patch and rerun Step66c")
    log("=" * 100)
    log(f"step61_cell={args.step61_cell}")
    log(f"outdir={outdir}")

    cell = read_csv_auto(args.step61_cell)

    patched, audit = probability_simplex_patch(
        cell,
        shift_scale=args.shift_scale,
        max_abs_delta=args.max_abs_delta,
    )

    patched_path = outdir / "step66e_step61_cell_level_probability_simplex_patched.csv"
    patched.to_csv(patched_path, index=False)

    pd.DataFrame([audit]).to_csv(outdir / "step66e_probability_simplex_patch_audit.csv", index=False)

    mini = outdir / "step61_probability_simplex_for_66c"
    mini.mkdir(parents=True, exist_ok=True)

    mini_cell = mini / "niche_perturbation_cell_level_state_editing.state_fixed.csv"
    safe_copy_no_metadata(patched_path, mini_cell)

    mini_rank = mini / "niche_perturbation_ranking.csv"
    safe_copy_no_metadata(args.step61_ranking, mini_rank)

    out66c = outdir / "rerun66c_probability_simplex"
    out66c.mkdir(parents=True, exist_ok=True)

    run_sh, cmd = write_run66c_script(args, mini_cell, mini, out66c)

    report = {
        "status": "prepared",
        "analysis_name": "Step66e probability-simplex model-derived state-probability patch and rerun Step66c",
        "step61_cell": str(args.step61_cell),
        "patched_cell_level": str(patched_path),
        "mini_step61_dir": str(mini),
        "rerun66c_outdir": str(out66c),
        "patch_audit": audit,
        "run66c_script": str(run_sh),
        "run66c_command": cmd,
        "run66c_executed": bool(args.run66c),
        "interpretation_note": (
            "This is not a true observed perturbed state probability. "
            "It is a probability-simplex model-derived sensitivity analysis based on baseline "
            "core/peri/remote probabilities and Step61 predicted perturbation response components."
        ),
    }

    (outdir / "step66e_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step66e probability-simplex patch report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Patch audit:")
    lines.append(pd.DataFrame([audit]).to_string(index=False))
    lines.append("")
    preview_cols = [
        "obs_name", "perturbation_id",
        "baseline_core_probability", "perturbed_core_probability",
        "baseline_peri_probability", "perturbed_peri_probability",
        "baseline_remote_probability", "perturbed_remote_probability",
        "predicted_delta_peri_probability",
        "predicted_delta_remote_probability",
        "predicted_peri_remote_shift",
        "step66e_stateprob_patch_mode",
    ]
    preview_cols = [c for c in preview_cols if c in patched.columns]
    lines.append("Preview:")
    lines.append(patched[preview_cols].head(30).to_string(index=False))

    (outdir / "step66e_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("PREPARED Step66e")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    if args.run66c:
        log("=" * 100)
        log("RUNNING Step66c with probability-simplex patched Step61 cell-level table")
        log("=" * 100)
        subprocess.run(cmd, check=True)

        rep66c = out66c / "step66c_report.json"
        if rep66c.exists():
            rep = json.loads(rep66c.read_text())
            report["status"] = "ok"
            report["run66c_executed"] = True
            report["rerun66c_report"] = rep
            (outdir / "step66e_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            log("=" * 100)
            log("DONE Step66e + rerun66c")
            log("=" * 100)
            log(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
