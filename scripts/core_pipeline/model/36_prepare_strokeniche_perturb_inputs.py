#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 8.1–8.2
Prepare StrokeNiche PerturbMap inputs.

This script keeps the original StrokeNiche Adapter untouched and constructs:
  1. virtual_cell_state_table.csv
  2. perturbation_library.csv
  3. state prototypes for core / peri / remote / repair
  4. positive and negative perturbation sets
  5. metadata JSON for downstream perturbation adapter training

Inputs:
  results/step7_strokeniche_adapter_clean/
  results/step7_strokeniche_adapter/
  results/task11_nichenet_virtual_penumbra_ligand_ranking/tables/
  results/task10_cellchat_region_communication/tables/
  results/figure6_regulator_communication/figures/fig6B_cell_cell_communication_network_top_edges.csv

Outputs:
  results/step8_strokeniche_perturbmap/
"""

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

PROJECT = Path("/mnt/h/vir/ST")

OUTDIR = PROJECT / "results/step8_strokeniche_perturbmap"
OUTDIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Step 7 StrokeNiche Adapter candidates
# -----------------------------------------------------------------------------
STEP7_CANDIDATES = [
    PROJECT / "results/step7_strokeniche_adapter_clean",
    PROJECT / "results/step7_strokeniche_adapter",
]

# -----------------------------------------------------------------------------
# Task10 / Task11 / Figure 6 inputs
# -----------------------------------------------------------------------------
TASK11_TABLE_DIR = PROJECT / "results/task11_nichenet_virtual_penumbra_ligand_ranking/tables"
TASK10_TABLE_DIR = PROJECT / "results/task10_cellchat_region_communication/tables"

RANKING_CANDIDATES = [
    TASK11_TABLE_DIR / "virtual_penumbra_candidate_perturbation_ranking.csv",
    TASK11_TABLE_DIR / "candidate_ligands_prior_table.csv",
]

FIG6B_TOP_EDGES = PROJECT / "results/figure6_regulator_communication/figures/fig6B_cell_cell_communication_network_top_edges.csv"

FIG6B_RANKED_EDGE_CANDIDATES = [
    PROJECT / "results/figure6_regulator_communication/figures/fig6B_cell_cell_communication_network_top_edges.csv",
    PROJECT / "results/figure6_regulator_communication/figures/fig6B_cell_cell_communication_network_all_ranked_edges.csv",
    PROJECT / "results/figure6_regulator_communication/figures/fig6B_cell_cell_communication_network_all_cellchat_all_ranked_edges.csv",
    PROJECT / "results/figure6_regulator_communication/figures_revised_final/fig6C_final_regulator_celltype_support_edges.csv",
]

CELLCHAT_CANDIDATES = [
    TASK10_TABLE_DIR / "Candidate_LR_hits_in_CellChat.csv",
    TASK10_TABLE_DIR / "All_samples_CellChat_LR_table.csv",
    TASK10_TABLE_DIR / "All_samples_CellChat_region_celltype_LR_summary.csv",
]

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
REGION_ORDER = ["lesion_core", "peri_infarct", "remote_like"]
REGION_CLASS_ORDER = ["lesion_core", "peri_infarct", "remote_like"]

REPAIR_PROTO_WEIGHTS = {
    "peri_infarct": 0.4,
    "remote_like": 0.6,
}

# Manually curated biologically meaningful perturbations to make sure
# redox / ferroptosis / barrier axes are represented even if absent from CellChat.
MANUAL_PERTURBATIONS = [
    {
        "perturbation_id": "Gpx4_up",
        "perturbation_type": "gene_up",
        "ligand": "",
        "receptor": "",
        "target_gene": "Gpx4",
        "direction": "activate",
        "candidate_score": 0.45,
        "celltype_context": "neurons;endothelial;astrocytes",
        "region_context": "lesion_core;peri_infarct",
        "source_evidence": "manual_redox_ferroptosis_axis",
        "mechanism_hint": "ferroptosis/redox protection",
    },
    {
        "perturbation_id": "Slc7a11_up",
        "perturbation_type": "gene_up",
        "ligand": "",
        "receptor": "",
        "target_gene": "Slc7a11",
        "direction": "activate",
        "candidate_score": 0.42,
        "celltype_context": "endothelial;astrocytes;neurons",
        "region_context": "lesion_core;peri_infarct",
        "source_evidence": "manual_redox_ferroptosis_axis",
        "mechanism_hint": "cystine transport / glutathione defense",
    },
    {
        "perturbation_id": "Cldn5_up",
        "perturbation_type": "gene_up",
        "ligand": "",
        "receptor": "",
        "target_gene": "Cldn5",
        "direction": "activate",
        "candidate_score": 0.40,
        "celltype_context": "endothelial",
        "region_context": "lesion_core;peri_infarct",
        "source_evidence": "manual_barrier_axis",
        "mechanism_hint": "BBB barrier stabilization",
    },
    {
        "perturbation_id": "Fth1_up",
        "perturbation_type": "gene_up",
        "ligand": "",
        "receptor": "",
        "target_gene": "Fth1",
        "direction": "activate",
        "candidate_score": 0.36,
        "celltype_context": "microglia;endothelial;astrocytes",
        "region_context": "lesion_core;peri_infarct",
        "source_evidence": "manual_iron_axis",
        "mechanism_hint": "iron buffering",
    },
]

# Negative decoys: used later for weakly-supervised ranking loss.
# These are not claims of biological neutrality; they are decoys for model calibration.
NEGATIVE_DECOY_GENES = [
    "Rplp0", "Rps18", "Actb", "Gapdh", "Malat1", "B2m", "Hprt", "Ppia"
]


# =============================================================================
# Utility functions
# =============================================================================
def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def safe_read_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as e:
        warnings.warn(f"Failed to read {path}: {repr(e)}")
        return None


def clean_gene_symbol(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", ""]:
        return ""
    return s


def gene_key(x):
    return re.sub(r"[^A-Za-z0-9]", "", str(x)).upper()


def split_receptor_complex(x):
    """
    Receptors may appear as:
      TREM2_TYROBP
      ACVR1B_TGFbR2
      ITGAV_ITGB1
      CD44
    Return compact primary receptor and list.
    """
    s = clean_gene_symbol(x)
    if s == "":
        return "", []

    parts = re.split(r"[_;/|,+\s]+", s)
    parts = [p for p in parts if p]
    primary = parts[0] if parts else s
    return primary, parts


def normalize_region(x):
    s = str(x)
    if "lesion_core" in s:
        return "lesion_core"
    if "peri_infarct" in s:
        return "peri_infarct"
    if "remote_like" in s:
        return "remote_like"
    return ""


def normalize_celltype(x):
    s = str(x)
    s_low = s.lower()

    if "microglia" in s_low or "myeloid" in s_low or "macrophage" in s_low:
        return "Microglia"
    if "astro" in s_low:
        return "Astrocytes"
    if "endothelial" in s_low or s_low in ["ec", "bbb"]:
        return "EndothelialCells"
    if "neuron" in s_low:
        return "Neurons"
    if "opc" in s_low:
        return "OPCs"
    if "ol" == s_low or "ols" in s_low or "oligodendro" in s_low:
        return "OLs"
    if "vlmc" in s_low:
        return "VLMCs"
    if "ependymal" in s_low:
        return "EpendymalCells"

    return s if s and s.lower() != "nan" else ""


def parse_state_label(label):
    """
    Example:
      lesion_core__Astrocytes
      peri_infarct__Microglia
      remote_like__EndothelialCells
    """
    s = str(label)
    region = normalize_region(s)

    if "__" in s:
        cell = s.split("__")[-1]
    else:
        cell = s

    return region, normalize_celltype(cell)


def score_to_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def minmax_norm(series):
    x = pd.to_numeric(pd.Series(series), errors="coerce")
    if x.notna().sum() == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    mn, mx = x.min(), x.max()
    if np.isclose(mn, mx):
        return pd.Series(np.ones(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


# =============================================================================
# Load Step 7 outputs
# =============================================================================
def find_step7_dir():
    for d in STEP7_CANDIDATES:
        z = d / "strokeniche_adapter_latent_z.npy"
        repair = d / "strokeniche_adapter_repair_score.npy"
        proba = d / "strokeniche_adapter_region_proba.npy"
        pred = d / "downstream/strokeniche_adapter_predictions.csv"

        if z.exists() and repair.exists() and proba.exists() and pred.exists():
            return d

    raise FileNotFoundError(
        "Cannot find complete Step 7 StrokeNiche Adapter outputs in clean or standard directories."
    )


def load_step7_outputs():
    d = find_step7_dir()

    z_path = d / "strokeniche_adapter_latent_z.npy"
    repair_path = d / "strokeniche_adapter_repair_score.npy"
    proba_path = d / "strokeniche_adapter_region_proba.npy"
    pred_path = d / "downstream/strokeniche_adapter_predictions.csv"

    Z = np.load(z_path)
    repair = np.load(repair_path)
    proba = np.load(proba_path)
    pred = pd.read_csv(pred_path)

    if len(pred) != Z.shape[0]:
        raise ValueError(f"predictions rows {len(pred)} != latent rows {Z.shape[0]}")

    if repair.shape[0] != Z.shape[0]:
        raise ValueError(f"repair rows {repair.shape[0]} != latent rows {Z.shape[0]}")

    if proba.shape[0] != Z.shape[0]:
        raise ValueError(f"proba rows {proba.shape[0]} != latent rows {Z.shape[0]}")

    print("Using Step 7 directory:", d)
    print("latent_z:", Z.shape)
    print("repair_score:", repair.shape)
    print("region_proba:", proba.shape)
    print("predictions:", pred.shape)

    return d, Z, repair, proba, pred


def build_virtual_cell_state_table(pred, repair, proba):
    df = pred.copy()

    # Add repair score if absent or inconsistent
    df["repair_score_from_npy"] = repair
    if "repair_score" not in df.columns:
        df["repair_score"] = repair

    # Add region probabilities
    if proba.shape[1] >= 3:
        df["P_lesion_core"] = proba[:, 0]
        df["P_peri_infarct"] = proba[:, 1]
        df["P_remote_like"] = proba[:, 2]
    else:
        for i in range(proba.shape[1]):
            df[f"P_region_{i}"] = proba[:, i]

    # Normalize common columns
    if "region_true" not in df.columns:
        if "region_auto" in df.columns:
            df["region_true"] = df["region_auto"]
        elif "region_pred" in df.columns:
            df["region_true"] = df["region_pred"]
        else:
            df["region_true"] = ""

    if "dominant_celltype" not in df.columns:
        for c in ["predicted_celltype", "predicted_celltype_filtered", "celltype", "cell_type"]:
            if c in df.columns:
                df["dominant_celltype"] = df[c]
                break
        if "dominant_celltype" not in df.columns:
            df["dominant_celltype"] = ""

    if "timepoint" not in df.columns:
        for c in ["time", "sample", "orig.ident", "day"]:
            if c in df.columns:
                df["timepoint"] = df[c]
                break
        if "timepoint" not in df.columns:
            df["timepoint"] = ""

    # Define lesion-core-like mask for perturbation training
    if "P_lesion_core" in df.columns:
        df["is_core_like"] = (
            (df["region_true"].astype(str) == "lesion_core")
            | (pd.to_numeric(df["P_lesion_core"], errors="coerce") >= 0.50)
        )
    else:
        df["is_core_like"] = df["region_true"].astype(str) == "lesion_core"

    df["is_repair_like"] = df["region_true"].astype(str).isin(["peri_infarct", "remote_like"])

    return df


# =============================================================================
# State prototypes
# =============================================================================
def compute_state_prototypes(Z, state_df):
    prototypes = {}

    for region in REGION_ORDER:
        mask = state_df["region_true"].astype(str).values == region
        if mask.sum() == 0:
            warnings.warn(f"No rows for region prototype: {region}")
            prototypes[region] = np.full(Z.shape[1], np.nan)
        else:
            prototypes[region] = np.nanmean(Z[mask], axis=0)

    # repair prototype = weighted combination of peri + remote
    repair_proto = np.zeros(Z.shape[1], dtype=float)
    weight_sum = 0.0

    for region, w in REPAIR_PROTO_WEIGHTS.items():
        v = prototypes.get(region)
        if v is not None and np.isfinite(v).all():
            repair_proto += w * v
            weight_sum += w

    if weight_sum > 0:
        repair_proto /= weight_sum
    else:
        repair_proto[:] = np.nan

    prototypes["repair_proto"] = repair_proto

    # Save centroid distances for QC
    rows = []
    keys = list(prototypes.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            v1, v2 = prototypes[k1], prototypes[k2]
            if np.isfinite(v1).all() and np.isfinite(v2).all():
                dist = float(np.linalg.norm(v1 - v2))
            else:
                dist = np.nan
            rows.append({"prototype_a": k1, "prototype_b": k2, "euclidean_distance": dist})

    pd.DataFrame(rows).to_csv(OUTDIR / "state_prototype_distances.csv", index=False)

    np.savez(
        OUTDIR / "strokeniche_state_prototypes.npz",
        lesion_core=prototypes["lesion_core"],
        peri_infarct=prototypes["peri_infarct"],
        remote_like=prototypes["remote_like"],
        repair_proto=prototypes["repair_proto"],
    )

    print("Saved:", OUTDIR / "strokeniche_state_prototypes.npz")
    print("Saved:", OUTDIR / "state_prototype_distances.csv")

    return prototypes


# =============================================================================
# Perturbation library construction
# =============================================================================
def load_candidate_ranking():
    p = first_existing(RANKING_CANDIDATES)
    if p is None:
        print("No Task11 ranking table found.")
        return pd.DataFrame()

    df = pd.read_csv(p)
    print("Using ranking table:", p, df.shape)

    if "ligand" not in df.columns:
        print("Ranking table has no ligand column. Ignoring.")
        return pd.DataFrame()

    # Candidate score
    score_col = None
    for c in ["total_score", "virtual_penumbra_score", "candidate_score", "ligand_total_score"]:
        if c in df.columns:
            score_col = c
            break

    if score_col is None:
        if "rank" in df.columns:
            df["candidate_score"] = 1.0 / pd.to_numeric(df["rank"], errors="coerce")
        else:
            df["candidate_score"] = 1.0
    else:
        df["candidate_score"] = pd.to_numeric(df[score_col], errors="coerce")

    # Receptor
    rec_col = None
    for c in ["main_receptor", "receptor", "candidate_receptor", "top_receptor", "receiver_hint"]:
        if c in df.columns:
            rec_col = c
            break

    if rec_col is None:
        df["main_receptor_for_library"] = ""
    else:
        df["main_receptor_for_library"] = df[rec_col].map(clean_gene_symbol)

    # Mechanism hint
    if "mechanism_hint" not in df.columns:
        if "pathway_hint" in df.columns:
            df["mechanism_hint"] = df["pathway_hint"].astype(str)
        else:
            df["mechanism_hint"] = ""

    if "rank" in df.columns:
        df = df.sort_values("rank", ascending=True)
    else:
        df = df.sort_values("candidate_score", ascending=False)

    return df


def load_fig6b_edges():
    p = first_existing(FIG6B_RANKED_EDGE_CANDIDATES)
    if p is None:
        print("No Fig6B edge table found.")
        return pd.DataFrame()

    df = pd.read_csv(p)
    print("Using Fig6B / CellChat edge table:", p, df.shape)

    if "ligand" not in df.columns:
        print("Fig6B edge table has no ligand column. Ignoring.")
        return pd.DataFrame()

    if "receptor" not in df.columns:
        for c in ["candidate_receptor", "cellchat_receptor", "receptor_cellchat"]:
            if c in df.columns:
                df["receptor"] = df[c]
                break

    if "receptor" not in df.columns:
        df["receptor"] = ""

    if "plot_priority" not in df.columns:
        for c in ["ligand_total_score", "candidate_score", "total_score", "max_score", "mean_score"]:
            if c in df.columns:
                df["plot_priority"] = pd.to_numeric(df[c], errors="coerce")
                break
        if "plot_priority" not in df.columns:
            df["plot_priority"] = 1.0

    df["plot_priority"] = pd.to_numeric(df["plot_priority"], errors="coerce").fillna(0)

    # Parse source/target state context
    if "source_region" not in df.columns:
        df["source_region"] = df["source"].map(lambda x: parse_state_label(x)[0]) if "source" in df.columns else ""
    if "target_region" not in df.columns:
        df["target_region"] = df["target"].map(lambda x: parse_state_label(x)[0]) if "target" in df.columns else ""

    if "source_celltype" not in df.columns:
        df["source_celltype"] = df["source"].map(lambda x: parse_state_label(x)[1]) if "source" in df.columns else ""
    if "target_celltype" not in df.columns:
        df["target_celltype"] = df["target"].map(lambda x: parse_state_label(x)[1]) if "target" in df.columns else ""

    return df


def build_gene_down_rows_from_ranking(rank_df):
    rows = []

    if rank_df is None or len(rank_df) == 0:
        return rows

    for _, r in rank_df.iterrows():
        ligand = clean_gene_symbol(r.get("ligand", ""))
        if ligand == "":
            continue

        score = score_to_float(r.get("candidate_score", 0.0), default=0.0)
        receptor = clean_gene_symbol(r.get("main_receptor_for_library", ""))

        rows.append({
            "perturbation_id": f"{ligand}_down",
            "perturbation_type": "gene_down",
            "ligand": ligand,
            "receptor": receptor,
            "target_gene": ligand,
            "direction": "inhibit",
            "candidate_score": score,
            "celltype_context": str(r.get("receiver_hint", "")),
            "region_context": "lesion_core;peri_infarct",
            "source_evidence": "NicheNet_virtual_penumbra_ranking",
            "mechanism_hint": str(r.get("mechanism_hint", "")),
            "rank": r.get("rank", np.nan),
        })

    return rows


def build_lr_blockade_rows_from_edges(edge_df):
    rows = []

    if edge_df is None or len(edge_df) == 0:
        return rows

    for _, r in edge_df.iterrows():
        ligand = clean_gene_symbol(r.get("ligand", ""))
        receptor = clean_gene_symbol(r.get("receptor", ""))

        if ligand == "":
            continue

        primary_receptor, receptor_parts = split_receptor_complex(receptor)
        receptor_for_id = receptor if receptor != "" else "receptor"

        score = score_to_float(r.get("plot_priority", 0.0), default=0.0)

        source_region = clean_gene_symbol(r.get("source_region", ""))
        target_region = clean_gene_symbol(r.get("target_region", ""))
        source_ct = normalize_celltype(r.get("source_celltype", ""))
        target_ct = normalize_celltype(r.get("target_celltype", ""))

        region_context = ";".join(
            sorted(set([x for x in [source_region, target_region] if x]))
        )

        celltype_context = ";".join(
            sorted(set([x for x in [source_ct, target_ct] if x]))
        )

        rows.append({
            "perturbation_id": f"{ligand}_{receptor_for_id}_blockade",
            "perturbation_type": "lr_blockade",
            "ligand": ligand,
            "receptor": receptor,
            "target_gene": primary_receptor if primary_receptor else receptor,
            "direction": "inhibit",
            "candidate_score": score,
            "celltype_context": celltype_context,
            "region_context": region_context,
            "source_evidence": "CellChat_LR_support;Fig6B_network",
            "mechanism_hint": str(r.get("pathway", r.get("pathway_hint", ""))),
            "rank": r.get("ligand_rank", np.nan),
        })

    return rows


def build_gene_up_rows_manual():
    return MANUAL_PERTURBATIONS.copy()


def build_negative_decoys():
    rows = []
    for g in NEGATIVE_DECOY_GENES:
        rows.append({
            "perturbation_id": f"{g}_decoy",
            "perturbation_type": "negative_decoy",
            "ligand": "",
            "receptor": "",
            "target_gene": g,
            "direction": "neutral",
            "candidate_score": 0.0,
            "celltype_context": "",
            "region_context": "",
            "source_evidence": "negative_decoy_for_ranking_loss",
            "mechanism_hint": "decoy / calibration",
            "rank": np.nan,
        })
    return rows


def deduplicate_library(df):
    """
    Deduplicate perturbations while keeping the highest candidate score.
    """
    df = df.copy()

    df["candidate_score"] = pd.to_numeric(df["candidate_score"], errors="coerce").fillna(0)
    df["perturbation_id"] = df["perturbation_id"].astype(str)

    # Clean IDs
    df["perturbation_id"] = (
        df["perturbation_id"]
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^A-Za-z0-9_\-\.]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )

    # Sort and deduplicate
    df = df.sort_values("candidate_score", ascending=False)
    df = df.drop_duplicates(subset=["perturbation_id"], keep="first")

    # Normalize scores to [0,1]
    df["candidate_score_norm"] = minmax_norm(df["candidate_score"]).values

    # Add perturbation class
    df["is_positive_candidate"] = df["perturbation_type"].isin(
        ["gene_down", "gene_up", "lr_blockade", "drug"]
    ) & (df["candidate_score"] > 0)

    df["is_negative_decoy"] = df["perturbation_type"].eq("negative_decoy")

    # Sort positive first
    df = df.sort_values(
        ["is_negative_decoy", "candidate_score"],
        ascending=[True, False]
    ).reset_index(drop=True)

    return df


def build_perturbation_library():
    rank_df = load_candidate_ranking()
    edge_df = load_fig6b_edges()

    rows = []

    rows.extend(build_gene_down_rows_from_ranking(rank_df))
    rows.extend(build_lr_blockade_rows_from_edges(edge_df))
    rows.extend(build_gene_up_rows_manual())
    rows.extend(build_negative_decoys())

    if len(rows) == 0:
        raise RuntimeError("No perturbations could be constructed.")

    lib = pd.DataFrame(rows)

    # Ensure all required columns
    required_cols = [
        "perturbation_id",
        "perturbation_type",
        "ligand",
        "receptor",
        "target_gene",
        "direction",
        "candidate_score",
        "celltype_context",
        "region_context",
        "source_evidence",
        "mechanism_hint",
        "rank",
    ]

    for c in required_cols:
        if c not in lib.columns:
            lib[c] = ""

    lib = lib[required_cols]
    lib = deduplicate_library(lib)

    out = OUTDIR / "perturbation_library.csv"
    lib.to_csv(out, index=False)
    print("Saved:", out)
    print("Perturbation library shape:", lib.shape)
    print(lib.head(20).to_string(index=False))

    positive = lib[lib["is_positive_candidate"]].copy()
    negative = lib[lib["is_negative_decoy"]].copy()

    positive.to_csv(OUTDIR / "positive_perturbations.csv", index=False)
    negative.to_csv(OUTDIR / "negative_decoy_perturbations.csv", index=False)

    print("Saved:", OUTDIR / "positive_perturbations.csv")
    print("Saved:", OUTDIR / "negative_decoy_perturbations.csv")

    return lib, rank_df, edge_df


# =============================================================================
# Save perturbation training input arrays
# =============================================================================
def save_training_inputs(Z, state_df, lib):
    """
    Save compact numpy arrays and tables for 37_train_strokeniche_perturb_adapter.py.
    """
    np.save(OUTDIR / "input_latent_z.npy", Z)

    # Convert categorical region into labels
    region_to_int = {r: i for i, r in enumerate(REGION_CLASS_ORDER)}
    y_region = state_df["region_true"].astype(str).map(region_to_int).fillna(-1).astype(int).values
    np.save(OUTDIR / "input_region_labels.npy", y_region)

    repair = pd.to_numeric(state_df["repair_score"], errors="coerce").fillna(
        pd.to_numeric(state_df["repair_score_from_npy"], errors="coerce")
    ).values
    np.save(OUTDIR / "input_repair_score.npy", repair)

    pcols = [c for c in ["P_lesion_core", "P_peri_infarct", "P_remote_like"] if c in state_df.columns]
    if len(pcols) == 3:
        np.save(OUTDIR / "input_region_proba.npy", state_df[pcols].values.astype(float))

    # Core-like indices for perturbation training
    core_idx = np.where(state_df["is_core_like"].values.astype(bool))[0]
    repair_idx = np.where(state_df["is_repair_like"].values.astype(bool))[0]

    np.save(OUTDIR / "input_core_like_indices.npy", core_idx)
    np.save(OUTDIR / "input_repair_like_indices.npy", repair_idx)

    # Perturbation numeric table
    pert_numeric = lib[[
        "perturbation_id",
        "perturbation_type",
        "target_gene",
        "direction",
        "candidate_score",
        "candidate_score_norm",
        "is_positive_candidate",
        "is_negative_decoy",
    ]].copy()

    pert_numeric.to_csv(OUTDIR / "perturbation_numeric_table.csv", index=False)

    print("Saved numeric training inputs:")
    print(" ", OUTDIR / "input_latent_z.npy")
    print(" ", OUTDIR / "input_region_labels.npy")
    print(" ", OUTDIR / "input_repair_score.npy")
    print(" ", OUTDIR / "input_core_like_indices.npy")
    print(" ", OUTDIR / "input_repair_like_indices.npy")
    print(" ", OUTDIR / "perturbation_numeric_table.csv")


# =============================================================================
# Metadata and QC summary
# =============================================================================
def save_metadata(step7_dir, Z, state_df, lib, rank_df, edge_df):
    region_counts = state_df["region_true"].astype(str).value_counts().to_dict()
    celltype_counts = state_df["dominant_celltype"].astype(str).value_counts().head(20).to_dict()

    perturb_counts = lib["perturbation_type"].value_counts().to_dict()

    meta = {
        "step": "Step 8.1-8.2 StrokeNiche PerturbMap input preparation",
        "project_dir": str(PROJECT),
        "output_dir": str(OUTDIR),
        "step7_source_dir": str(step7_dir),
        "n_virtual_cells": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
        "region_counts": region_counts,
        "top_celltype_counts": celltype_counts,
        "n_core_like_cells": int(state_df["is_core_like"].sum()),
        "n_repair_like_cells": int(state_df["is_repair_like"].sum()),
        "n_perturbations": int(len(lib)),
        "perturbation_type_counts": perturb_counts,
        "n_positive_perturbations": int(lib["is_positive_candidate"].sum()),
        "n_negative_decoys": int(lib["is_negative_decoy"].sum()),
        "ranking_table_rows": int(len(rank_df)) if rank_df is not None else 0,
        "cellchat_edge_rows": int(len(edge_df)) if edge_df is not None else 0,
        "region_class_order": REGION_CLASS_ORDER,
        "repair_proto_weights": REPAIR_PROTO_WEIGHTS,
        "manual_perturbations": MANUAL_PERTURBATIONS,
        "negative_decoy_genes": NEGATIVE_DECOY_GENES,
    }

    out_json = OUTDIR / "step8_input_metadata.json"
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("Saved:", out_json)

    # Compact human-readable summary
    lines = []
    lines.append("Step 8 StrokeNiche PerturbMap input preparation")
    lines.append("=" * 100)
    lines.append(f"Step7 source: {step7_dir}")
    lines.append(f"Virtual cells: {Z.shape[0]}")
    lines.append(f"Latent dimension: {Z.shape[1]}")
    lines.append("")
    lines.append("Region counts:")
    for k, v in region_counts.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("Perturbation type counts:")
    for k, v in perturb_counts.items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append(f"Positive perturbations: {int(lib['is_positive_candidate'].sum())}")
    lines.append(f"Negative decoys: {int(lib['is_negative_decoy'].sum())}")
    lines.append("")
    lines.append("Main outputs:")
    for f in [
        "virtual_cell_state_table.csv",
        "perturbation_library.csv",
        "positive_perturbations.csv",
        "negative_decoy_perturbations.csv",
        "strokeniche_state_prototypes.npz",
        "input_latent_z.npy",
        "input_core_like_indices.npy",
        "step8_input_metadata.json",
    ]:
        lines.append(f"  {OUTDIR / f}")

    out_txt = OUTDIR / "step8_input_summary.txt"
    out_txt.write_text("\n".join(lines))
    print("Saved:", out_txt)


# =============================================================================
# Main
# =============================================================================
def main():
    print("=" * 100)
    print("Step 8.1–8.2 | Prepare StrokeNiche PerturbMap inputs")
    print("=" * 100)

    step7_dir, Z, repair, proba, pred = load_step7_outputs()

    # State table
    state_df = build_virtual_cell_state_table(pred, repair, proba)
    state_out = OUTDIR / "virtual_cell_state_table.csv"
    state_df.to_csv(state_out, index=False)
    print("Saved:", state_out)

    # Prototypes
    compute_state_prototypes(Z, state_df)

    # Perturbation library
    lib, rank_df, edge_df = build_perturbation_library()

    # Training inputs
    save_training_inputs(Z, state_df, lib)

    # Metadata
    save_metadata(step7_dir, Z, state_df, lib, rank_df, edge_df)

    print("\nDone.")
    print("Output directory:", OUTDIR)


if __name__ == "__main__":
    main()
