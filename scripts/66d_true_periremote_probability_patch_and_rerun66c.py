#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
66d_true_periremote_probability_patch_and_rerun66c.py

Purpose
-------
Fix Step66c's remaining limitation:

  Step66c patch_mode = proxy_from_repair_core_safety

This script creates a true peri/remote/core probability patch for Step61
cell-level perturbation output and then reruns Step66c.

It does NOT overwrite Step61 or previous Step66c results.

Core logic
----------
For each row in Step61 cell-level perturbation table:

1. If explicit probability columns are already available:
     baseline_peri_probability
     perturbed_peri_probability
     baseline_remote_probability
     perturbed_remote_probability
     baseline_core_probability
     perturbed_core_probability
   then use them directly.

2. Else, if baseline and perturbed latent/UMAP coordinates are available:
     baseline_x, baseline_y
     perturbed_x, perturbed_y
   or:
     obs_name + reference coordinate table + perturbed_x/perturbed_y
   then train a coordinate-based state probability classifier using a reference
   state-labeled coordinate table and predict:
     baseline_core/peri/remote probabilities
     perturbed_core/peri/remote probabilities

3. Else stop. No proxy fallback is allowed.

Outputs
-------
outdir/
  step66d_step61_cell_level_with_TRUE_state_probabilities.csv
  step66d_state_probability_patch_audit.csv
  step66d_state_classifier_reference_used.csv
  step66d_run_66c_true_probability.sh
  run_66d_true_probability_patch.log
  rerun66c_true_probability/
      step66c_report.json
      step66c_perturbation_fdr.csv
      Fig_Step66C_CellLevelTrackPerturbationFDR_*.pdf/svg/png

Recommended interpretation
--------------------------
If patch_mode = direct_probability_columns:
  true probability delta.

If patch_mode = coordinate_state_classifier:
  classifier-derived state probability delta.
  This is stronger than proxy_from_repair_core_safety, but should still be described
  as model-derived state probability, not experimentally observed state conversion.
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

DEFAULT_COORD_CANDIDATES = [
    ROOT / "dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv",
    ROOT / "dynamics_graph_64c_refined_labels_state_celltype/step64c_track_assignment_by_cell.refined_labels.csv",
    BASE / "results/figure5_trajectory/figures/fig5_strokeniche_trajectory_latent_state_coordinates.csv",
    BASE / "results/figure7_strokeniche_perturbmap_polished/strokeniche_true_umap_coordinates_standardized.csv",
]

DEFAULT_STEP65_DIR = ROOT / "track_dynamic_marker_regulator_65c_state_celltype_gene_clean_stroke_real_externaldb_no_encode_metadata"
DEFAULT_STEP65E_DIR = ROOT / "track_dynamic_marker_regulator_65e_encode_chipseq_housekeeping_filtered"

DEFAULT_66C_SCRIPT = BASE / "66c_celllevel_periremote_module_target_fdr.py"

DEFAULT_OUT = ROOT / "track_level_perturbation_fdr_66d_true_stateprob_rerun66c"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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


def read_csv_auto(path, required=True):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        if required:
            raise FileNotFoundError(p)
        return None
    try:
        return pd.read_csv(p, low_memory=False)
    except Exception:
        return pd.read_csv(p, sep="\t", low_memory=False)


def find_first_col(df, patterns, numeric=False):
    for pat in patterns:
        for c in df.columns:
            nk = norm_key(c)
            if re.search(pat, nk):
                if numeric:
                    x = pd.to_numeric(df[c], errors="coerce")
                    if x.notna().mean() < 0.20:
                        continue
                return c
    return ""


def normalize_state_label(x):
    s = str(x).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    if any(k in s for k in ["lesion_core", "lesioncore", "core_like", "core"]):
        return "lesion_core"
    if any(k in s for k in ["peri_infarct", "periinfarct", "peri"]):
        return "peri_infarct"
    if any(k in s for k in ["remote_like", "remotelike", "remote"]):
        return "remote_like"

    return ""


def softmax_negdist(d2, temperature):
    z = -d2 / max(temperature, 1e-8)
    z = z - np.nanmax(z, axis=1, keepdims=True)
    ez = np.exp(z)
    denom = np.nansum(ez, axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return ez / denom


# -----------------------------------------------------------------------------
# Direct probability detection
# -----------------------------------------------------------------------------

def detect_probability_columns(df):
    """
    Detect explicit baseline/perturbed core/peri/remote probability columns.

    Return mapping if all core/peri/remote pairs exist.
    """
    mapping = {}

    states = {
        "core": [
            "core", "lesion_core", "lesioncore"
        ],
        "peri": [
            "peri", "peri_infarct", "periinfarct"
        ],
        "remote": [
            "remote", "remote_like", "remotelike"
        ],
    }

    before_prefix = [
        r"baseline", r"before", r"original", r"pre", r"control", r"unperturbed"
    ]
    after_prefix = [
        r"perturbed", r"after", r"edited", r"post", r"new"
    ]

    for state, keys in states.items():
        before_patterns = []
        after_patterns = []

        for b in before_prefix:
            for k in keys:
                before_patterns.append(rf"{b}.*{k}.*prob")
                before_patterns.append(rf"{k}.*{b}.*prob")

        for a in after_prefix:
            for k in keys:
                after_patterns.append(rf"{a}.*{k}.*prob")
                after_patterns.append(rf"{k}.*{a}.*prob")

        before_col = find_first_col(df, before_patterns, numeric=True)
        after_col = find_first_col(df, after_patterns, numeric=True)

        mapping[f"baseline_{state}"] = before_col
        mapping[f"perturbed_{state}"] = after_col

    ok = all(mapping.get(k) for k in [
        "baseline_core", "perturbed_core",
        "baseline_peri", "perturbed_peri",
        "baseline_remote", "perturbed_remote",
    ])

    return ok, mapping


def apply_direct_probability_patch(df, mapping):
    out = df.copy()

    out["baseline_core_probability"] = pd.to_numeric(out[mapping["baseline_core"]], errors="coerce")
    out["perturbed_core_probability"] = pd.to_numeric(out[mapping["perturbed_core"]], errors="coerce")

    out["baseline_peri_probability"] = pd.to_numeric(out[mapping["baseline_peri"]], errors="coerce")
    out["perturbed_peri_probability"] = pd.to_numeric(out[mapping["perturbed_peri"]], errors="coerce")

    out["baseline_remote_probability"] = pd.to_numeric(out[mapping["baseline_remote"]], errors="coerce")
    out["perturbed_remote_probability"] = pd.to_numeric(out[mapping["perturbed_remote"]], errors="coerce")

    out["predicted_delta_peri_probability"] = out["perturbed_peri_probability"] - out["baseline_peri_probability"]
    out["predicted_delta_remote_probability"] = out["perturbed_remote_probability"] - out["baseline_remote_probability"]
    out["predicted_peri_remote_shift"] = out["predicted_delta_peri_probability"] + out["predicted_delta_remote_probability"]

    out["true_core_probability_reduction"] = out["baseline_core_probability"] - out["perturbed_core_probability"]

    # Keep original columns, but provide columns that Step66c will pick up.
    out["delta_peri_probability"] = out["predicted_delta_peri_probability"]
    out["delta_remote_probability"] = out["predicted_delta_remote_probability"]
    out["peri_remote_shift"] = out["predicted_peri_remote_shift"]

    # Make core reduction explicit; Step66c prefers predicted_core_reversal if present,
    # but this column records true probability reduction for audit.
    out["core_probability_reduction_true_stateprob"] = out["true_core_probability_reduction"]

    out["step66d_stateprob_patch_mode"] = "direct_probability_columns"

    return out


# -----------------------------------------------------------------------------
# Coordinate classifier path
# -----------------------------------------------------------------------------

def detect_obs_col(df):
    for c in ["obs_name", "cell_id", "barcode", "spot_id"]:
        if c in df.columns:
            return c
    return ""


def detect_state_col(df):
    candidates = [
        "state_group", "region_true", "region_label", "state_label",
        "predicted_state", "region", "state"
    ]

    for c in candidates:
        if c in df.columns:
            vals = df[c].astype(str).map(normalize_state_label)
            if vals.isin(["lesion_core", "peri_infarct", "remote_like"]).mean() > 0.30:
                return c

    for c in df.columns:
        nk = norm_key(c)
        if "state" in nk or "region" in nk:
            vals = df[c].astype(str).map(normalize_state_label)
            if vals.isin(["lesion_core", "peri_infarct", "remote_like"]).mean() > 0.30:
                return c

    return ""


def detect_coordinate_cols(df, prefix_hint=""):
    """
    Find 2D coordinates.
    """
    cols = list(df.columns)

    # High-priority exact pairs.
    pairs = [
        ("strict_coord_x", "strict_coord_y"),
        ("umap_1", "umap_2"),
        ("UMAP_1", "UMAP_2"),
        ("umap1", "umap2"),
        ("x", "y"),
        ("coord_x", "coord_y"),
        ("latent_x", "latent_y"),
        ("latent_1", "latent_2"),
        ("z1", "z2"),
        ("z_1", "z_2"),
    ]

    for x, y in pairs:
        if x in cols and y in cols:
            if pd.to_numeric(df[x], errors="coerce").notna().mean() > 0.20 and pd.to_numeric(df[y], errors="coerce").notna().mean() > 0.20:
                return x, y

    # Prefix-based patterns.
    hints = []
    if prefix_hint:
        hints.append(norm_key(prefix_hint))
    hints.extend(["baseline", "before", "original", "pre", "control", "perturbed", "after", "edited", "post", "new"])

    for hint in hints:
        x_col = ""
        y_col = ""

        for c in cols:
            nk = norm_key(c)
            if hint in nk and re.search(r"(^|_)(x|coord_x|umap_1|umap1|latent_1|z1|z_1)($|_)", nk):
                if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.20:
                    x_col = c
                    break

        for c in cols:
            nk = norm_key(c)
            if hint in nk and re.search(r"(^|_)(y|coord_y|umap_2|umap2|latent_2|z2|z_2)($|_)", nk):
                if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.20:
                    y_col = c
                    break

        if x_col and y_col:
            return x_col, y_col

    return "", ""


def detect_perturbed_coordinate_cols(df):
    """
    STRICT perturbed-coordinate detector.

    It only accepts:
      1. explicit perturbed/after/edited/post/new coordinate columns
      2. explicit delta/shift/vector coordinate columns

    It must NOT fall back to baseline/reference coordinates.
    This prevents invalid cases such as:
      perturbed_x_or_dx_col = baseline_coord_x_from_ref
      perturbed_y_or_dy_col = baseline_coord_y_from_ref
    """
    cols = list(df.columns)
    lowmap = {c: norm_key(c) for c in cols}

    # Explicit absolute perturbed coordinate pairs.
    explicit_pairs = [
        ("perturbed_strict_coord_x", "perturbed_strict_coord_y"),
        ("perturbed_umap_1", "perturbed_umap_2"),
        ("perturbed_umap1", "perturbed_umap2"),
        ("perturbed_x", "perturbed_y"),
        ("after_x", "after_y"),
        ("edited_x", "edited_y"),
        ("post_x", "post_y"),
        ("new_x", "new_y"),
        ("perturbed_coord_x", "perturbed_coord_y"),
        ("edited_coord_x", "edited_coord_y"),
        ("after_coord_x", "after_coord_y"),
        ("post_coord_x", "post_coord_y"),
    ]

    for x, y in explicit_pairs:
        if x in cols and y in cols:
            return x, y, "explicit_perturbed_xy"

    # Pattern-based absolute perturbed coordinate pairs.
    perturbed_prefixes = ["perturbed", "after", "edited", "post", "new"]
    x_patterns = [
        r"(x|coord_x|umap_1|umap1|latent_1|z1|z_1)$",
        r"(^|_)(x|coord_x|umap_1|umap1|latent_1|z1|z_1)($|_)",
    ]
    y_patterns = [
        r"(y|coord_y|umap_2|umap2|latent_2|z2|z_2)$",
        r"(^|_)(y|coord_y|umap_2|umap2|latent_2|z2|z_2)($|_)",
    ]

    for pref in perturbed_prefixes:
        x_col = ""
        y_col = ""
        for c in cols:
            nk = lowmap[c]
            if pref in nk and not any(bad in nk for bad in ["baseline", "before", "original", "control", "from_ref"]):
                if any(re.search(pat, nk) for pat in x_patterns):
                    if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.20:
                        x_col = c
                        break
        for c in cols:
            nk = lowmap[c]
            if pref in nk and not any(bad in nk for bad in ["baseline", "before", "original", "control", "from_ref"]):
                if any(re.search(pat, nk) for pat in y_patterns):
                    if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.20:
                        y_col = c
                        break
        if x_col and y_col:
            return x_col, y_col, "pattern_perturbed_xy"

    # Explicit delta/vector pairs. These are added to baseline coords.
    delta_pairs = [
        ("delta_x", "delta_y"),
        ("dx", "dy"),
        ("shift_x", "shift_y"),
        ("vector_x", "vector_y"),
        ("perturbation_vector_x", "perturbation_vector_y"),
        ("delta_umap_1", "delta_umap_2"),
        ("delta_umap1", "delta_umap2"),
        ("delta_coord_x", "delta_coord_y"),
        ("state_shift_x", "state_shift_y"),
        ("latent_shift_x", "latent_shift_y"),
    ]

    for dx, dy in delta_pairs:
        if dx in cols and dy in cols:
            if pd.to_numeric(df[dx], errors="coerce").notna().mean() > 0.20 and pd.to_numeric(df[dy], errors="coerce").notna().mean() > 0.20:
                return dx, dy, "delta_xy"

    # Pattern-based delta/vector columns.
    delta_prefixes = ["delta", "shift", "vector", "perturbation_vector", "state_shift", "latent_shift"]
    for pref in delta_prefixes:
        x_col = ""
        y_col = ""
        for c in cols:
            nk = lowmap[c]
            if pref in nk and any(re.search(pat, nk) for pat in x_patterns):
                if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.20:
                    x_col = c
                    break
        for c in cols:
            nk = lowmap[c]
            if pref in nk and any(re.search(pat, nk) for pat in y_patterns):
                if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.20:
                    y_col = c
                    break
        if x_col and y_col:
            return x_col, y_col, "delta_xy"

    return "", "", ""


def load_reference_coordinates(coord_files, track_dir):
    """
    Build state-labeled reference coordinate table.
    """
    frames = []

    # First, include track assignment as source of labels.
    assign_p = Path(track_dir) / "step64c_track_assignment_by_cell.refined_labels.csv"
    if assign_p.exists():
        assign = read_csv_auto(assign_p)
    else:
        assign = pd.DataFrame()

    for p in coord_files:
        p = Path(p)
        if not p.exists() or p.stat().st_size == 0:
            continue

        try:
            df = read_csv_auto(p)
        except Exception:
            continue

        obs_col = detect_obs_col(df)
        x_col, y_col = detect_coordinate_cols(df)
        state_col = detect_state_col(df)

        if not obs_col or not x_col or not y_col:
            continue

        tmp = df.copy()
        tmp["obs_name"] = tmp[obs_col].astype(str)
        tmp["coord_x"] = pd.to_numeric(tmp[x_col], errors="coerce")
        tmp["coord_y"] = pd.to_numeric(tmp[y_col], errors="coerce")

        if state_col:
            tmp["state_label"] = tmp[state_col].map(normalize_state_label)
        else:
            tmp["state_label"] = ""

        # Merge assignment if state not present.
        if (tmp["state_label"].eq("").mean() > 0.50) and not assign.empty and "obs_name" in assign.columns:
            a = assign.copy()
            a["obs_name"] = a["obs_name"].astype(str)
            a_state_col = detect_state_col(a)
            if a_state_col:
                a["assign_state_label"] = a[a_state_col].map(normalize_state_label)
                tmp = tmp.merge(a[["obs_name", "assign_state_label"]].drop_duplicates(), on="obs_name", how="left")
                tmp["state_label"] = tmp["state_label"].where(tmp["state_label"].ne(""), tmp["assign_state_label"].fillna(""))

        tmp = tmp[tmp["state_label"].isin(["lesion_core", "peri_infarct", "remote_like"])].copy()
        tmp = tmp.dropna(subset=["coord_x", "coord_y"])
        tmp = tmp[["obs_name", "coord_x", "coord_y", "state_label"]].drop_duplicates("obs_name")

        if len(tmp) >= 100:
            tmp["source_file"] = str(p)
            frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    ref = pd.concat(frames, ignore_index=True).drop_duplicates("obs_name")

    return ref


def train_centroid_state_classifier(ref):
    """
    Simple robust classifier:
      standardize coordinates
      centroid per state
      probability = softmax(-squared distance / temperature)
    """
    states = ["lesion_core", "peri_infarct", "remote_like"]

    X = ref[["coord_x", "coord_y"]].astype(float).values
    y = ref["state_label"].astype(str).values

    mean = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    std[std <= 1e-12] = 1.0

    Xz = (X - mean) / std

    centroids = []
    for s in states:
        sub = Xz[y == s]
        if len(sub) == 0:
            raise RuntimeError(f"No reference points for state {s}")
        centroids.append(np.nanmean(sub, axis=0))

    C = np.vstack(centroids)

    # Temperature based on median nearest-centroid squared distance.
    d2 = ((Xz[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    min_d2 = np.min(d2, axis=1)
    temp = float(np.nanmedian(min_d2))
    if not np.isfinite(temp) or temp <= 1e-8:
        temp = 1.0

    clf = {
        "states": states,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "centroids": C.tolist(),
        "temperature": temp,
        "n_reference": int(len(ref)),
        "state_counts": ref["state_label"].value_counts().to_dict(),
        "source_files": sorted(ref["source_file"].dropna().unique().tolist())[:10] if "source_file" in ref.columns else [],
    }

    return clf


def predict_state_probabilities(clf, x, y):
    X = np.vstack([pd.to_numeric(x, errors="coerce").values, pd.to_numeric(y, errors="coerce").values]).T.astype(float)

    mean = np.array(clf["mean"], dtype=float)
    std = np.array(clf["std"], dtype=float)
    C = np.array(clf["centroids"], dtype=float)
    temp = float(clf["temperature"])

    Xz = (X - mean) / std
    d2 = ((Xz[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    P = softmax_negdist(d2, temperature=temp)

    states = clf["states"]
    out = pd.DataFrame(P, columns=[f"prob_{s}" for s in states])

    return out


def apply_coordinate_classifier_patch(cell_df, coord_files, track_dir):
    df = cell_df.copy()

    obs_col = detect_obs_col(df)
    if not obs_col:
        raise RuntimeError("Cell-level table lacks obs_name/cell_id.")
    df["obs_name"] = df[obs_col].astype(str)

    ref = load_reference_coordinates(coord_files, track_dir)
    if ref.empty:
        raise RuntimeError("No usable reference coordinate table with state labels was found.")

    clf = train_centroid_state_classifier(ref)

    # Baseline coordinates.
    base_x_col, base_y_col = detect_coordinate_cols(df, prefix_hint="baseline")

    if not base_x_col or not base_y_col:
        # Merge baseline coordinates by obs_name from reference.
        df = df.merge(
            ref[["obs_name", "coord_x", "coord_y"]].rename(
                columns={"coord_x": "baseline_coord_x_from_ref", "coord_y": "baseline_coord_y_from_ref"}
            ),
            on="obs_name",
            how="left",
        )
        base_x_col = "baseline_coord_x_from_ref"
        base_y_col = "baseline_coord_y_from_ref"

    # Perturbed coordinates or delta coordinates.
    px_col, py_col, p_mode = detect_perturbed_coordinate_cols(df)

    if not px_col or not py_col:
        raise RuntimeError(
            "No perturbed coordinate columns or delta coordinate columns were found. "
            "Cannot compute true/model-derived peri/remote probability delta without proxy."
        )

    if p_mode == "delta_xy":
        pert_x = pd.to_numeric(df[base_x_col], errors="coerce") + pd.to_numeric(df[px_col], errors="coerce")
        pert_y = pd.to_numeric(df[base_y_col], errors="coerce") + pd.to_numeric(df[py_col], errors="coerce")
    else:
        pert_x = pd.to_numeric(df[px_col], errors="coerce")
        pert_y = pd.to_numeric(df[py_col], errors="coerce")

    base_prob = predict_state_probabilities(clf, df[base_x_col], df[base_y_col])
    pert_prob = predict_state_probabilities(clf, pert_x, pert_y)

    df["baseline_core_probability"] = base_prob["prob_lesion_core"].values
    df["baseline_peri_probability"] = base_prob["prob_peri_infarct"].values
    df["baseline_remote_probability"] = base_prob["prob_remote_like"].values

    df["perturbed_core_probability"] = pert_prob["prob_lesion_core"].values
    df["perturbed_peri_probability"] = pert_prob["prob_peri_infarct"].values
    df["perturbed_remote_probability"] = pert_prob["prob_remote_like"].values

    df["predicted_delta_peri_probability"] = df["perturbed_peri_probability"] - df["baseline_peri_probability"]
    df["predicted_delta_remote_probability"] = df["perturbed_remote_probability"] - df["baseline_remote_probability"]
    df["predicted_peri_remote_shift"] = df["predicted_delta_peri_probability"] + df["predicted_delta_remote_probability"]

    df["true_core_probability_reduction"] = df["baseline_core_probability"] - df["perturbed_core_probability"]

    df["delta_peri_probability"] = df["predicted_delta_peri_probability"]
    df["delta_remote_probability"] = df["predicted_delta_remote_probability"]
    df["peri_remote_shift"] = df["predicted_peri_remote_shift"]

    df["core_probability_reduction_true_stateprob"] = df["true_core_probability_reduction"]

    df["step66d_stateprob_patch_mode"] = "coordinate_state_classifier"

    audit = {
        "patch_mode": "coordinate_state_classifier",
        "obs_col": obs_col,
        "baseline_x_col": base_x_col,
        "baseline_y_col": base_y_col,
        "perturbed_x_or_dx_col": px_col,
        "perturbed_y_or_dy_col": py_col,
        "perturbed_coordinate_mode": p_mode,
        "n_rows": int(len(df)),
        "n_non_na_baseline_coord": int(pd.to_numeric(df[base_x_col], errors="coerce").notna().sum()),
        "n_non_na_perturbed_coord": int(pd.Series(pert_x).notna().sum()),
        "peri_remote_non_na_fraction": float(df["predicted_peri_remote_shift"].notna().mean()),
        "peri_remote_sd": float(pd.to_numeric(df["predicted_peri_remote_shift"], errors="coerce").std()),
        "classifier": clf,
    }

    return df, audit, ref, clf


# -----------------------------------------------------------------------------
# Running 66c
# -----------------------------------------------------------------------------

def write_run66c_script(args, patched_cell_path, outdir_66c):
    run_sh = Path(args.outdir) / "step66d_run_66c_true_probability.sh"

    cmd = [
        args.python,
        str(args.script66c),
        "--track_dir", str(args.track_dir),
        "--step61_dir", str(Path(args.outdir) / "step61_trueprob_for_66c"),
        "--step61_cell", str(patched_cell_path),
        "--step61_ranking", str(args.step61_ranking),
        "--step65_dir", str(args.step65_dir),
        "--step65e_dir", str(args.step65e_dir),
        "--outdir", str(outdir_66c),
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
        f.write(" ".join([shell_quote(x) for x in cmd]))
        f.write(f" 2>&1 | tee {shell_quote(Path(args.outdir) / 'run_66d_rerun66c_true_probability.log')}\n")

    try:
        os.chmod(run_sh, 0o755)
    except Exception:
        pass

    return run_sh, cmd



def safe_copy_no_metadata(src, dst):
    """
    Copy file content only. Do not preserve timestamp/permission metadata.
    This avoids PermissionError from shutil.copy2/copystat on /mnt/h Windows-mounted paths.
    """
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--step61_cell", default=str(DEFAULT_STEP61_CELL))
    ap.add_argument("--step61_ranking", default=str(DEFAULT_STEP61_RANKING))
    ap.add_argument("--track_dir", default=str(DEFAULT_TRACK_DIR))

    ap.add_argument("--coord_files", default="", help="Comma-separated coordinate reference files. Empty = default candidates.")
    ap.add_argument("--step65_dir", default=str(DEFAULT_STEP65_DIR))
    ap.add_argument("--step65e_dir", default=str(DEFAULT_STEP65E_DIR))
    ap.add_argument("--script66c", default=str(DEFAULT_66C_SCRIPT))

    ap.add_argument("--outdir", default=str(DEFAULT_OUT))
    ap.add_argument("--python", default="/home/shu/miniconda/envs/nicheformer_env/bin/python")

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

    ap.add_argument("--run66c", action="store_true", help="Actually rerun 66c after patching.")
    ap.add_argument("--strict_no_proxy", action="store_true", default=True)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("Step66d true peri/remote state-probability patch and rerun Step66c")
    log("=" * 100)
    log(f"step61_cell={args.step61_cell}")
    log(f"outdir={outdir}")

    cell = read_csv_auto(args.step61_cell)

    if "obs_name" not in cell.columns:
        raise RuntimeError("Step61 cell-level table lacks obs_name.")

    if "perturbation_id" not in cell.columns:
        id_col = find_first_col(cell, [r"perturb", r"niche_perturb", r"module"])
        if not id_col:
            raise RuntimeError("Step61 cell-level table lacks perturbation_id.")
        cell["perturbation_id"] = cell[id_col].astype(str)

    # 1. Try direct probability columns.
    direct_ok, direct_map = detect_probability_columns(cell)

    if direct_ok:
        patched = apply_direct_probability_patch(cell, direct_map)
        patch_audit = {
            "status": "ok",
            "patch_mode": "direct_probability_columns",
            "direct_probability_mapping": direct_map,
            "n_rows": int(len(patched)),
            "peri_remote_non_na_fraction": float(patched["predicted_peri_remote_shift"].notna().mean()),
            "peri_remote_sd": float(pd.to_numeric(patched["predicted_peri_remote_shift"], errors="coerce").std()),
        }
        ref = pd.DataFrame()
        clf = {}

    else:
        # 2. Try coordinate classifier.
        if args.coord_files.strip():
            coord_files = [Path(x.strip()) for x in args.coord_files.split(",") if x.strip()]
        else:
            coord_files = [p for p in DEFAULT_COORD_CANDIDATES if Path(p).exists()]

        try:
            patched, coord_audit, ref, clf = apply_coordinate_classifier_patch(
                cell,
                coord_files=coord_files,
                track_dir=args.track_dir,
            )
            patch_audit = {
                "status": "ok",
                **coord_audit,
            }
        except Exception as e:
            patch_audit = {
                "status": "failed_no_true_probability_or_coordinate_patch_available",
                "direct_probability_mapping": direct_map,
                "coordinate_patch_error": str(e),
                "message": (
                    "No true probability pair and no usable perturbed coordinates were found. "
                    "66d refuses to use proxy_from_repair_core_safety. "
                    "Please rerun Step61 to output either perturbed_peri/remote probabilities "
                    "or perturbed latent coordinates."
                ),
            }
            (outdir / "step66d_state_probability_patch_audit.csv").write_text(
                pd.DataFrame([patch_audit]).to_csv(index=False),
                encoding="utf-8",
            )
            (outdir / "step66d_report.json").write_text(
                json.dumps(patch_audit, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            log(json.dumps(patch_audit, indent=2, ensure_ascii=False, default=str))
            raise RuntimeError(patch_audit["message"])


    # Strict validation: coordinate_state_classifier is invalid if perturbed coords equal baseline
    # or if the resulting peri/remote shift has zero variance.
    if patch_audit.get("patch_mode") == "coordinate_state_classifier":
        bx = patch_audit.get("baseline_x_col", "")
        by = patch_audit.get("baseline_y_col", "")
        px = patch_audit.get("perturbed_x_or_dx_col", "")
        py = patch_audit.get("perturbed_y_or_dy_col", "")
        pr_sd = float(patch_audit.get("peri_remote_sd", 0.0) or 0.0)

        invalid_same_coord = (px == bx) or (py == by) or ("baseline" in str(px).lower()) or ("baseline" in str(py).lower()) or ("from_ref" in str(px).lower()) or ("from_ref" in str(py).lower())
        invalid_zero_shift = pr_sd <= 1e-12

        if invalid_same_coord or invalid_zero_shift:
            patch_audit["status"] = "failed_invalid_coordinate_patch"
            patch_audit["invalid_reason"] = (
                "Perturbed coordinates are identical to baseline/reference coordinates "
                "or peri_remote_shift has zero variance. Refuse to rerun 66c because this "
                "would not represent a true perturbation-induced state shift."
            )
            pd.DataFrame([patch_audit]).to_csv(outdir / "step66d_state_probability_patch_audit.csv", index=False)
            (outdir / "step66d_report.json").write_text(
                json.dumps(patch_audit, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8"
            )
            log(json.dumps(patch_audit, indent=2, ensure_ascii=False, default=str))
            raise RuntimeError(patch_audit["invalid_reason"])

    # 3. Save patched Step61 cell-level table.
    patched_cell_path = outdir / "step66d_step61_cell_level_with_TRUE_state_probabilities.csv"
    patched.to_csv(patched_cell_path, index=False)

    pd.DataFrame([patch_audit]).to_csv(outdir / "step66d_state_probability_patch_audit.csv", index=False)

    if not ref.empty:
        ref.to_csv(outdir / "step66d_state_classifier_reference_used.csv", index=False)
    else:
        pd.DataFrame().to_csv(outdir / "step66d_state_classifier_reference_used.csv", index=False)

    (outdir / "step66d_state_classifier_model.json").write_text(
        json.dumps(clf, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    # 4. Create a mini Step61 dir for 66c.
    mini61 = outdir / "step61_trueprob_for_66c"
    mini61.mkdir(parents=True, exist_ok=True)

    mini_cell = mini61 / "niche_perturbation_cell_level_state_editing.state_fixed.csv"
    safe_copy_no_metadata(patched_cell_path, mini_cell)

    ranking_src = Path(args.step61_ranking)
    mini_rank = mini61 / "niche_perturbation_ranking.csv"
    if ranking_src.exists():
        safe_copy_no_metadata(ranking_src, mini_rank)

    outdir_66c = outdir / "rerun66c_true_probability"
    outdir_66c.mkdir(parents=True, exist_ok=True)

    run_sh, cmd = write_run66c_script(args, mini_cell, outdir_66c)

    report = {
        "status": "prepared",
        "analysis_name": "Step66d true peri/remote probability patch and rerun Step66c",
        "step61_cell": str(args.step61_cell),
        "patched_cell_level": str(patched_cell_path),
        "mini_step61_dir": str(mini61),
        "rerun66c_outdir": str(outdir_66c),
        "patch_audit": patch_audit,
        "run66c_script": str(run_sh),
        "run66c_command": cmd,
        "run66c_executed": bool(args.run66c),
        "interpretation_note": (
            "This patch removes the Step66c proxy_from_repair_core_safety limitation. "
            "If patch_mode is coordinate_state_classifier, the state probabilities are model-derived "
            "from latent/UMAP coordinates and region labels. If patch_mode is direct_probability_columns, "
            "they are direct Step61 probability deltas."
        ),
    }

    (outdir / "step66d_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("Step66d true state-probability patch report")
    lines.append("=" * 100)
    lines.append(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Patch audit:")
    lines.append(pd.DataFrame([patch_audit]).to_string(index=False))
    lines.append("")
    lines.append("Patched columns preview:")
    preview_cols = [
        "obs_name", "perturbation_id",
        "baseline_core_probability", "perturbed_core_probability",
        "baseline_peri_probability", "perturbed_peri_probability",
        "baseline_remote_probability", "perturbed_remote_probability",
        "predicted_delta_peri_probability",
        "predicted_delta_remote_probability",
        "predicted_peri_remote_shift",
        "step66d_stateprob_patch_mode",
    ]
    preview_cols = [c for c in preview_cols if c in patched.columns]
    lines.append(patched[preview_cols].head(20).to_string(index=False))

    (outdir / "step66d_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("PREPARED Step66d")
    log("=" * 100)
    log(json.dumps(report, indent=2, ensure_ascii=False, default=str))

    if args.run66c:
        log("=" * 100)
        log("RUNNING Step66c with true state-probability patched Step61 cell-level table")
        log("=" * 100)
        subprocess.run(cmd, check=True)

        # Append 66c final report if available.
        rep66c = outdir_66c / "step66c_report.json"
        if rep66c.exists():
            rep66c_obj = json.loads(rep66c.read_text())
            report["status"] = "ok"
            report["run66c_executed"] = True
            report["rerun66c_report"] = rep66c_obj
            (outdir / "step66d_report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            log("=" * 100)
            log("DONE Step66d + rerun66c")
            log("=" * 100)
            log(json.dumps(report, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
