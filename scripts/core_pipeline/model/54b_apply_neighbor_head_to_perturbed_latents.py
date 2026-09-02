#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
54b_apply_neighbor_head_to_perturbed_latents.py

Purpose
-------
Apply the trained strict-z NeighborHead to perturbed latent states z_i^p:

    neighbor_i     = observed / 53c kNN-RCTD neighborhood profile
    neighbor_pred_i = NeighborHead(z_i)
    neighbor_pred_i^p = NeighborHead(z_i^p)

Then compute:

    neural_neighbor_shift_distance = distance(neighbor_pred_i^p, neighbor_i)
    neural_neighbor_shift_from_baseline_pred = distance(neighbor_pred_i^p, neighbor_pred_i)

Inputs
------
1) strict-z NeighborHead model:
   results/step8_strokeniche_perturbmap/neighbor_head_54_true_z/neighbor_head_model.pt
   or final_release_20260528/neighbor_head_54_true_z/neighbor_head_model.pt

2) baseline strict z_i table:
   results/step8_strokeniche_perturbmap/neighbor_head_54/adapter_true_hidden_latent_for_neighbor_head.csv

3) 53c neighbor profile:
   results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/virtual_cell_knn_neighbor_profile_RCTD.csv

4) Perturbed latent table, one of:
   A. obs_name + perturbation columns + z_p columns
   B. obs_name + perturbation columns + delta_z columns
   C. perturbation-level delta_z columns, expanded over all baseline cells

Outputs
-------
results/step8_strokeniche_perturbmap/neighbor_head_54b_perturbed_neighbor_shift/
  perturbed_neighbor_head_predictions.csv
  perturbed_neighbor_head_predictions_light.csv
  perturbation_neural_neighbor_shift_summary.csv
  perturbation_neural_neighbor_shift_target_group_summary.csv
  perturbed_latent_candidate_audit.csv
  neighbor_head_54b_metadata.json
  neighbor_head_54b_report.txt

Important
---------
This script does not train a new model. It applies the already trained strict-z
NeighborHead to z_i^p.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE = Path("/mnt/h/vir/ST")

DEFAULT_MODEL_CANDIDATES = [
    BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54_true_z/neighbor_head_model.pt",
    BASE / "results/drug_repurposing/final_release_20260528/neighbor_head_54_true_z/neighbor_head_model.pt",
]

DEFAULT_BASELINE_LATENT = BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54/adapter_true_hidden_latent_for_neighbor_head.csv"
DEFAULT_PROFILE53C = BASE / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/virtual_cell_knn_neighbor_profile_RCTD.csv"
DEFAULT_OUTDIR = BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54b_perturbed_neighbor_shift"


def log(msg: str):
    print(msg, flush=True)


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def norm_id(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_name(x: str, max_len: int = 100) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", str(x).strip()).strip("_")
    return (s or "unknown")[:max_len]


def find_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def torch_load_trusted(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def pick_id_col(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "obs_name", "cell_id", "cell", "barcode", "spot_id", "spot",
        "id", "index_id", "index", "Unnamed: 0",
    ]
    nmap = {norm_col(c): c for c in df.columns}
    for p in preferred:
        if norm_col(p) in nmap:
            return nmap[norm_col(p)]
    for c in df.columns:
        nc = norm_col(c)
        if any(t in nc for t in ["obs", "cell", "barcode", "spot"]):
            return c
    return None


def detect_perturb_cols(df: pd.DataFrame) -> List[str]:
    preferred = [
        "perturbation_id",
        "perturbation_name",
        "perturbagen_id",
        "perturbagen_name",
        "drug_name",
        "drug",
        "compound",
        "ligand",
        "receptor",
        "target_gene",
        "gene",
        "direction",
        "mechanism_hint",
        "signature_source",
        "signature_source_type",
    ]
    out = []
    nmap = {norm_col(c): c for c in df.columns}
    for p in preferred:
        k = norm_col(p)
        if k in nmap and nmap[k] not in out:
            out.append(nmap[k])

    for c in df.columns:
        nc = norm_col(c)
        if any(t in nc for t in ["perturb", "drug", "compound", "ligand", "receptor", "target_gene"]):
            if c not in out and df[c].nunique(dropna=True) > 1:
                out.append(c)

    return out


def sorted_dim_cols(cols: List[str]) -> List[str]:
    def key(c):
        nc = norm_col(c)
        m = re.search(r"(\d+)$", nc)
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=key)


def detect_zp_cols(df: pd.DataFrame, expected_dim: Optional[int] = None) -> List[str]:
    patterns = [
        r"^z_p_?\d+$",
        r"^zp_?\d+$",
        r"^z_perturbed_?\d+$",
        r"^perturbed_z_?\d+$",
        r"^latent_p_?\d+$",
        r"^perturbed_latent_?\d+$",
        r"^adapter_latent_p_?\d+$",
        r"^adapter_perturbed_latent_?\d+$",
        r"^neighborhead_input_p_?\d+$",
    ]

    cols = []
    for c in df.columns:
        nc = norm_col(c)
        if any(re.search(p, nc) for p in patterns):
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() > 0:
                cols.append(c)

    cols = sorted_dim_cols(cols)

    # If the file is explicitly a perturbed latent table and adapter_latent_* exist,
    # allow adapter_latent_* as z_i^p.
    if not cols:
        name_hint_cols = [c for c in df.columns if re.search(r"^adapter_latent_?\d+$", norm_col(c))]
        name_hint_cols = sorted_dim_cols(name_hint_cols)
        if expected_dim is None or len(name_hint_cols) == expected_dim:
            cols = name_hint_cols

    if expected_dim is not None and len(cols) != expected_dim:
        # Do not silently accept wrong dimension.
        return []

    return cols


def detect_delta_cols(df: pd.DataFrame, expected_dim: Optional[int] = None) -> List[str]:
    patterns = [
        r"^delta_z_?\d+$",
        r"^z_delta_?\d+$",
        r"^perturb_delta_z_?\d+$",
        r"^delta_latent_?\d+$",
        r"^latent_delta_?\d+$",
        r"^adapter_delta_?\d+$",
        r"^perturb_decoder_delta_?\d+$",
    ]

    cols = []
    for c in df.columns:
        nc = norm_col(c)
        if any(re.search(p, nc) for p in patterns):
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() > 0:
                cols.append(c)

    cols = sorted_dim_cols(cols)

    if expected_dim is not None and len(cols) != expected_dim:
        return []

    return cols


def discover_perturbed_latent_files(base: Path, user_path: Optional[str] = None) -> List[Path]:
    if user_path:
        p = Path(user_path)
        return [p] if p.exists() else []

    patterns = [
        "results/step8_strokeniche_perturbmap/**/*perturb*latent*.csv",
        "results/step8_strokeniche_perturbmap/**/*perturbed*latent*.csv",
        "results/step8_strokeniche_perturbmap/**/*z_p*.csv",
        "results/step8_strokeniche_perturbmap/**/*delta_z*.csv",
        "results/step8_strokeniche_perturbmap/**/*perturb*cell*.csv",
        "results/step8_strokeniche_perturbmap/**/*perturb*prediction*.csv",
        "results/step8_strokeniche_perturbmap/**/*perturb*output*.csv",
        "results/step8_strokeniche_perturbmap/**/*decoder*.csv",
        "results/step8_strokeniche_perturbmap/**/*latent*.csv",
    ]

    hits = []
    seen = set()

    for pat in patterns:
        for p in base.glob(pat):
            if not p.is_file():
                continue
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)

            s = sp.lower()
            if any(bad in s for bad in [
                "neighbor_head_54/adapter_true_hidden",
                "neighbor_head_54/adapter_latent_for_neighbor",
                "neighbor_head_54_true_z",
                "neighbor_head_54_comparison",
                "chembl", "ctd", "dgidb", "drugcentral",
            ]):
                continue

            hits.append(p)

    return sorted(hits, key=lambda p: (len(str(p)), str(p)))


def audit_perturbed_candidates(paths: List[Path], expected_dim: int, baseline_obs: set) -> pd.DataFrame:
    rows = []
    for p in paths:
        row = {"path": str(p), "status": "unknown"}
        try:
            df = pd.read_csv(p, low_memory=False, nrows=2000)
        except Exception as e:
            row.update({"status": f"read_failed: {e}"})
            rows.append(row)
            continue

        id_col = pick_id_col(df)
        perturb_cols = detect_perturb_cols(df)
        zp_cols = detect_zp_cols(df, expected_dim=expected_dim)
        delta_cols = detect_delta_cols(df, expected_dim=expected_dim)

        matched_preview = 0
        if id_col:
            matched_preview = len(set(df[id_col].dropna().astype(str).map(norm_id)) & baseline_obs)

        score = 0
        if id_col:
            score += 100
        if perturb_cols:
            score += 100
        if len(zp_cols) == expected_dim:
            score += 1000
        if len(delta_cols) == expected_dim:
            score += 900
        score += matched_preview

        row.update({
            "status": "ok",
            "shape_preview": str(df.shape),
            "id_col": id_col,
            "matched_obs_preview": matched_preview,
            "perturb_cols_n": len(perturb_cols),
            "perturb_cols": ";".join(perturb_cols[:20]),
            "zp_cols_n": len(zp_cols),
            "delta_cols_n": len(delta_cols),
            "zp_cols_head": ";".join(zp_cols[:10]),
            "delta_cols_head": ";".join(delta_cols[:10]),
            "score": score,
        })
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["path", "status", "score"])

    return pd.DataFrame(rows).sort_values("score", ascending=False)


class NeighborHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, dropout, sigmoid_output):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        y = self.net(x)
        if self.sigmoid_output:
            y = torch.sigmoid(y)
        return y


def load_neighbor_head(model_path: Path, device: str = "cpu"):
    state = torch_load_trusted(model_path)

    input_dim = int(state["input_dim"])
    output_dim = int(state["output_dim"])
    hidden = list(state["hidden"])
    dropout = float(state.get("dropout", 0.0))
    sigmoid_output = bool(state.get("sigmoid_output", True))

    model = NeighborHead(input_dim, output_dim, hidden, dropout, sigmoid_output)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)

    meta = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "hidden": hidden,
        "dropout": dropout,
        "sigmoid_output": sigmoid_output,
        "latent_cols": state["latent_cols"],
        "target_cols": state["target_cols"],
        "x_mean": np.asarray(state["x_mean"], dtype=float),
        "x_std": np.asarray(state["x_std"], dtype=float),
        "target_groups": state.get("target_groups", {}),
        "best_epoch": state.get("epoch"),
        "best_val_loss": state.get("best_val_loss"),
    }

    return model, meta


def predict_neighbor(model, model_meta, X: np.ndarray, device: str = "cpu", batch_size: int = 4096) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mean = model_meta["x_mean"]
    std = model_meta["x_std"]
    std = np.where(std < 1e-8, 1.0, std)

    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    preds = []
    with torch.no_grad():
        for i in range(0, Xs.shape[0], batch_size):
            xb = torch.tensor(Xs[i:i+batch_size], dtype=torch.float32, device=device)
            yb = model(xb).cpu().numpy()
            preds.append(yb)
    return np.vstack(preds)


def cosine_distance_rows(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return 1.0 - cos


def group_indices_from_targets(target_cols: List[str]) -> Dict[str, List[int]]:
    groups = {"RCTD": [], "region": [], "celltype": [], "score": []}
    for i, c in enumerate(target_cols):
        if c.startswith("neighbor_mean_feature_RCTD_"):
            groups["RCTD"].append(i)
        elif c.startswith("neighbor_mean_feature_regionprob_"):
            groups["region"].append(i)
        elif c.startswith("neighbor_mean_feature_celltype_"):
            groups["celltype"].append(i)
        elif c.startswith("neighbor_mean_feature_score_"):
            groups["score"].append(i)
    return groups


def add_distance_metrics(out: pd.DataFrame, pred_p: np.ndarray, baseline_true: np.ndarray, baseline_pred: np.ndarray, target_cols: List[str], prefix: str = "neural_neighbor"):
    diff_true = pred_p - baseline_true
    diff_pred = pred_p - baseline_pred

    out[f"{prefix}_shift_l2_to_true_neighbor"] = np.sqrt((diff_true ** 2).sum(axis=1))
    out[f"{prefix}_shift_mae_to_true_neighbor"] = np.mean(np.abs(diff_true), axis=1)
    out[f"{prefix}_shift_cosine_to_true_neighbor"] = cosine_distance_rows(baseline_true, pred_p)

    out[f"{prefix}_shift_l2_from_baseline_pred"] = np.sqrt((diff_pred ** 2).sum(axis=1))
    out[f"{prefix}_shift_mae_from_baseline_pred"] = np.mean(np.abs(diff_pred), axis=1)
    out[f"{prefix}_shift_cosine_from_baseline_pred"] = cosine_distance_rows(baseline_pred, pred_p)

    groups = group_indices_from_targets(target_cols)

    for g, idx in groups.items():
        if not idx:
            continue
        idx = list(idx)
        out[f"{prefix}_shift_l2_to_true_{g}"] = np.sqrt((diff_true[:, idx] ** 2).sum(axis=1))
        out[f"{prefix}_shift_mae_to_true_{g}"] = np.mean(np.abs(diff_true[:, idx]), axis=1)
        out[f"{prefix}_shift_cosine_to_true_{g}"] = cosine_distance_rows(baseline_true[:, idx], pred_p[:, idx])

        out[f"{prefix}_shift_l2_from_baseline_pred_{g}"] = np.sqrt((diff_pred[:, idx] ** 2).sum(axis=1))
        out[f"{prefix}_shift_mae_from_baseline_pred_{g}"] = np.mean(np.abs(diff_pred[:, idx]), axis=1)

    # Direction proxies.
    repair_idx = [
        i for i, c in enumerate(target_cols)
        if "repair" in norm_col(c) or "score_repair" in norm_col(c)
    ]
    core_idx = [
        i for i, c in enumerate(target_cols)
        if "core" in norm_col(c) or "lesion_core" in norm_col(c)
    ]
    remote_idx = [
        i for i, c in enumerate(target_cols)
        if "remote" in norm_col(c)
    ]
    peri_idx = [
        i for i, c in enumerate(target_cols)
        if "peri" in norm_col(c)
    ]

    if repair_idx:
        out[f"{prefix}_delta_repair_score_vs_true"] = pred_p[:, repair_idx].mean(axis=1) - baseline_true[:, repair_idx].mean(axis=1)
    else:
        out[f"{prefix}_delta_repair_score_vs_true"] = np.nan

    if core_idx:
        out[f"{prefix}_delta_core_prob_vs_true"] = pred_p[:, core_idx].mean(axis=1) - baseline_true[:, core_idx].mean(axis=1)
    else:
        out[f"{prefix}_delta_core_prob_vs_true"] = np.nan

    if remote_idx:
        out[f"{prefix}_delta_remote_prob_vs_true"] = pred_p[:, remote_idx].mean(axis=1) - baseline_true[:, remote_idx].mean(axis=1)
    else:
        out[f"{prefix}_delta_remote_prob_vs_true"] = np.nan

    if peri_idx:
        out[f"{prefix}_delta_peri_prob_vs_true"] = pred_p[:, peri_idx].mean(axis=1) - baseline_true[:, peri_idx].mean(axis=1)
    else:
        out[f"{prefix}_delta_peri_prob_vs_true"] = np.nan

    def direction(row):
        dr = row.get(f"{prefix}_delta_repair_score_vs_true", np.nan)
        dc = row.get(f"{prefix}_delta_core_prob_vs_true", np.nan)
        drem = row.get(f"{prefix}_delta_remote_prob_vs_true", np.nan)

        if pd.notna(dr) and pd.notna(dc):
            if dr > 0 and dc < 0:
                return "toward_repair_away_from_core"
            if dr > 0 and dc >= 0:
                return "toward_repair_but_core_not_reduced"
            if dr <= 0 and dc < 0:
                return "away_from_core_without_repair_gain"
            if dr <= 0 and dc >= 0:
                return "unfavorable_or_no_repair_shift"

        if pd.notna(drem) and pd.notna(dc):
            if drem > 0 and dc < 0:
                return "toward_remote_away_from_core"

        return "undetermined"

    out[f"{prefix}_direction_proxy"] = out.apply(direction, axis=1)

    return out


def load_baseline_tables(baseline_latent_path: Path, profile53c_path: Path, target_cols: List[str]):
    z = pd.read_csv(baseline_latent_path, low_memory=False)
    profile = pd.read_csv(profile53c_path, low_memory=False)

    if "obs_name" not in z.columns:
        raise RuntimeError("baseline latent table lacks obs_name.")
    if "obs_name" not in profile.columns:
        raise RuntimeError("profile53c table lacks obs_name.")

    latent_cols = sorted_dim_cols([c for c in z.columns if c.startswith("adapter_latent_")])

    if not latent_cols:
        raise RuntimeError("No adapter_latent_* columns found in baseline latent table.")

    missing_targets = [c for c in target_cols if c not in profile.columns]
    if missing_targets:
        raise RuntimeError(f"profile53c lacks target columns: {missing_targets[:10]}")

    base = z[["obs_name"] + latent_cols].merge(
        profile[["obs_name"] + target_cols],
        on="obs_name",
        how="inner",
    )

    return base, latent_cols


def select_best_candidate(audit: pd.DataFrame) -> Optional[str]:
    """
    Strict selection: only accept files that actually contain z_i^p or delta_z
    with the expected model input dimension. Do not select state-prediction
    tables such as perturbed_state_predictions.csv.
    """
    if audit is None or audit.empty:
        return None

    ok = audit[audit["status"].eq("ok")].copy()
    if ok.empty:
        return None

    ok["zp_cols_n"] = pd.to_numeric(ok.get("zp_cols_n", 0), errors="coerce").fillna(0)
    ok["delta_cols_n"] = pd.to_numeric(ok.get("delta_cols_n", 0), errors="coerce").fillna(0)

    valid = ok[(ok["zp_cols_n"] > 0) | (ok["delta_cols_n"] > 0)].copy()
    if valid.empty:
        return None

    valid = valid.sort_values("score", ascending=False)
    return valid.iloc[0]["path"]


def load_perturbed_table(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def build_row_level_from_perturbed(
    pert: pd.DataFrame,
    baseline: pd.DataFrame,
    latent_cols: List[str],
    zp_cols: List[str],
    delta_cols: List[str],
    perturb_cols: List[str],
    expected_dim: int,
    expand_perturbation_level_delta: bool,
) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """
    Return metadata table and Xp matrix.
    """
    id_col = pick_id_col(pert)

    # Case A: cell-level z_i^p.
    if id_col and len(zp_cols) == expected_dim:
        meta_cols = [id_col] + [c for c in perturb_cols if c != id_col]
        meta = pert[meta_cols].copy()
        meta = meta.rename(columns={id_col: "obs_name"})
        Xp = pert[zp_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        mode = "cell_level_perturbed_z"

        return meta, Xp, {
            "mode": mode,
            "id_col": id_col,
            "zp_cols": zp_cols,
            "delta_cols": [],
            "n_rows": int(len(meta)),
        }

    # Case B: cell-level delta_z.
    if id_col and len(delta_cols) == expected_dim:
        b = baseline[["obs_name"] + latent_cols].copy()
        b["_obs_key"] = b["obs_name"].astype(str).map(norm_id)

        tmp = pert.copy()
        tmp["_obs_key"] = tmp[id_col].astype(str).map(norm_id)

        merged = tmp.merge(b, on="_obs_key", how="left", suffixes=("", "_baseline"))

        base_X = merged[latent_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        delta_X = merged[delta_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
        Xp = base_X + delta_X

        meta_cols = [id_col] + [c for c in perturb_cols if c != id_col]
        meta = merged[meta_cols].copy()
        meta = meta.rename(columns={id_col: "obs_name"})

        return meta, Xp, {
            "mode": "cell_level_delta_z",
            "id_col": id_col,
            "zp_cols": [],
            "delta_cols": delta_cols,
            "n_rows": int(len(meta)),
            "baseline_matched_rows": int(np.isfinite(base_X).all(axis=1).sum()),
        }

    # Case C: perturbation-level delta_z, expand over all cells.
    if (not id_col) and len(delta_cols) == expected_dim and expand_perturbation_level_delta:
        records = []
        Xps = []

        base_obs = baseline["obs_name"].astype(str).values
        base_X = baseline[latent_cols].values.astype(float)

        pert_cols_final = perturb_cols if perturb_cols else []
        for idx, row in pert.iterrows():
            delta = row[delta_cols].astype(float).values
            xp = base_X + delta[None, :]
            Xps.append(xp)

            m = pd.DataFrame({"obs_name": base_obs})
            for c in pert_cols_final:
                m[c] = row[c]
            m["_perturbation_row_index"] = idx
            records.append(m)

        meta = pd.concat(records, ignore_index=True)
        Xp = np.vstack(Xps)

        return meta, Xp, {
            "mode": "perturbation_level_delta_z_expanded",
            "id_col": None,
            "zp_cols": [],
            "delta_cols": delta_cols,
            "n_rows": int(len(meta)),
            "n_perturbations": int(len(pert)),
        }

    raise RuntimeError(
        "Could not construct z_i^p from perturbed table. "
        f"id_col={id_col}, zp_cols={len(zp_cols)}, delta_cols={len(delta_cols)}, expected_dim={expected_dim}. "
        "Need either cell-level z_i^p, cell-level delta_z, or perturbation-level delta_z."
    )


def summarize_by_perturbation(df: pd.DataFrame, perturb_cols: List[str], prefix: str = "neural_neighbor") -> pd.DataFrame:
    if not perturb_cols:
        perturb_cols = ["_perturbation_row_index"] if "_perturbation_row_index" in df.columns else []

    if not perturb_cols:
        df = df.copy()
        df["_all"] = "all"
        perturb_cols = ["_all"]

    metrics = [
        f"{prefix}_shift_l2_to_true_neighbor",
        f"{prefix}_shift_mae_to_true_neighbor",
        f"{prefix}_shift_cosine_to_true_neighbor",
        f"{prefix}_shift_l2_from_baseline_pred",
        f"{prefix}_shift_mae_from_baseline_pred",
        f"{prefix}_shift_cosine_from_baseline_pred",
        f"{prefix}_shift_l2_to_true_RCTD",
        f"{prefix}_shift_l2_to_true_region",
        f"{prefix}_shift_l2_to_true_celltype",
        f"{prefix}_delta_repair_score_vs_true",
        f"{prefix}_delta_core_prob_vs_true",
        f"{prefix}_delta_remote_prob_vs_true",
        f"{prefix}_delta_peri_prob_vs_true",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for keys, sub in df.groupby(perturb_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        row = {c: k for c, k in zip(perturb_cols, keys)}
        row["n_cells"] = int(len(sub))

        for m in metrics:
            x = pd.to_numeric(sub[m], errors="coerce")
            row[m + "_mean"] = float(x.mean())
            row[m + "_median"] = float(x.median())
            row[m + "_p90"] = float(x.quantile(0.90))

        dir_col = f"{prefix}_direction_proxy"
        if dir_col in sub.columns:
            vc = sub[dir_col].value_counts(dropna=False)
            row["dominant_neighbor_shift_direction"] = str(vc.index[0])
            row["dominant_neighbor_shift_direction_fraction"] = float(vc.iloc[0] / len(sub))
            for label in [
                "toward_repair_away_from_core",
                "toward_repair_but_core_not_reduced",
                "away_from_core_without_repair_gain",
                "unfavorable_or_no_repair_shift",
                "undetermined",
            ]:
                row[f"fraction_{label}"] = float((sub[dir_col] == label).mean())

        rows.append(row)

    out = pd.DataFrame(rows)

    rank_col = f"{prefix}_shift_l2_from_baseline_pred_mean"
    if rank_col in out.columns:
        out = out.sort_values(rank_col, ascending=False).reset_index(drop=True)
        out["neural_neighbor_shift_rank"] = np.arange(1, len(out) + 1)

    return out


def summarize_by_group_metrics(df: pd.DataFrame, perturb_cols: List[str], prefix: str = "neural_neighbor") -> pd.DataFrame:
    groups = ["RCTD", "region", "celltype", "score"]
    rows = []

    if not perturb_cols:
        perturb_cols = ["_perturbation_row_index"] if "_perturbation_row_index" in df.columns else []

    if not perturb_cols:
        return pd.DataFrame()

    for group in groups:
        cols = [
            f"{prefix}_shift_l2_to_true_{group}",
            f"{prefix}_shift_mae_to_true_{group}",
            f"{prefix}_shift_cosine_to_true_{group}",
            f"{prefix}_shift_l2_from_baseline_pred_{group}",
            f"{prefix}_shift_mae_from_baseline_pred_{group}",
        ]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            continue

        for keys, sub in df.groupby(perturb_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {c: k for c, k in zip(perturb_cols, keys)}
            row["target_group"] = group
            row["n_cells"] = int(len(sub))
            for c in cols:
                x = pd.to_numeric(sub[c], errors="coerce")
                row[c + "_mean"] = float(x.mean())
                row[c + "_median"] = float(x.median())
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--neighbor_head_model", default=None)
    ap.add_argument("--baseline_latent", default=str(DEFAULT_BASELINE_LATENT))
    ap.add_argument("--profile53c", default=str(DEFAULT_PROFILE53C))
    ap.add_argument("--perturbed_latent", default=None, help="Optional path to z_i^p or delta_z table.")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--expand_perturbation_level_delta", action="store_true")
    ap.add_argument("--no_save_row_level", action="store_true")
    args = ap.parse_args()

    base = Path(args.base)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.neighbor_head_model) if args.neighbor_head_model else find_existing(DEFAULT_MODEL_CANDIDATES)
    if model_path is None or not model_path.exists():
        raise FileNotFoundError("Cannot find strict-z NeighborHead model. Use --neighbor_head_model.")

    baseline_latent_path = Path(args.baseline_latent)
    profile53c_path = Path(args.profile53c)

    log("=" * 100)
    log("54b apply NeighborHead to perturbed latent z_i^p")
    log("=" * 100)
    log(f"model={model_path}")
    log(f"baseline_latent={baseline_latent_path}")
    log(f"profile53c={profile53c_path}")
    log(f"outdir={outdir}")

    model, model_meta = load_neighbor_head(model_path, device=args.device)

    expected_dim = int(model_meta["input_dim"])
    target_cols = list(model_meta["target_cols"])

    baseline, latent_cols = load_baseline_tables(
        baseline_latent_path=baseline_latent_path,
        profile53c_path=profile53c_path,
        target_cols=target_cols,
    )

    if len(latent_cols) != expected_dim:
        raise RuntimeError(f"Baseline latent dim {len(latent_cols)} != model input_dim {expected_dim}")

    baseline_obs = set(baseline["obs_name"].astype(str).map(norm_id))

    # Baseline prediction.
    X_base = baseline[latent_cols].values.astype(float)
    Y_true = baseline[target_cols].values.astype(float)
    Y_base_pred = predict_neighbor(
        model,
        model_meta,
        X_base,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Discover perturbed latent file.
    candidate_paths = discover_perturbed_latent_files(base, args.perturbed_latent)
    audit = audit_perturbed_candidates(candidate_paths, expected_dim=expected_dim, baseline_obs=baseline_obs)
    audit_path = outdir / "perturbed_latent_candidate_audit.csv"
    audit.to_csv(audit_path, index=False)

    selected_path = Path(args.perturbed_latent) if args.perturbed_latent else None
    if selected_path is None:
        best = select_best_candidate(audit)
        selected_path = Path(best) if best else None

    if selected_path is None or not selected_path.exists():
        metadata = {
            "status": "failed_no_perturbed_latent",
            "model": str(model_path),
            "baseline_latent": str(baseline_latent_path),
            "profile53c": str(profile53c_path),
            "expected_dim": expected_dim,
            "candidate_paths_n": len(candidate_paths),
            "audit": str(audit_path),
            "message": (
                "No z_i^p or delta_z table found. Need to export perturbed latent states "
                "from perturb_adapter / PerturbDecoder before running 54b."
            ),
        }

        meta_path = outdir / "neighbor_head_54b_metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        report_path = outdir / "neighbor_head_54b_report.txt"
        lines = []
        lines.append("54b NeighborHead perturbed latent application report")
        lines.append("=" * 100)
        lines.append("FAILED: no perturbed latent z_i^p or delta_z table found.")
        lines.append("")
        lines.append(f"Expected latent dim: {expected_dim}")
        lines.append(f"Candidate files checked: {len(candidate_paths)}")
        lines.append(f"Audit saved: {audit_path}")
        lines.append("")
        lines.append("Top audit rows:")
        if len(audit):
            lines.append(audit.head(50).to_string(index=False))
        else:
            lines.append("No candidate files found.")
        lines.append("")
        lines.append("Next step:")
        lines.append("Export one of the following from PerturbDecoder:")
        lines.append("1) cell-level z_i^p table: obs_name + perturbation_id + adapter_latent_p_0...adapter_latent_p_127")
        lines.append("2) cell-level delta_z table: obs_name + perturbation_id + delta_z_0...delta_z_127")
        lines.append("3) perturbation-level delta_z table: perturbation_id + delta_z_0...delta_z_127, then rerun with --expand_perturbation_level_delta")
        report_path.write_text("\n".join(lines), encoding="utf-8")

        print("=" * 100)
        print("FAILED 54b: no perturbed latent found")
        print("=" * 100)
        print("Audit:", audit_path)
        if len(audit):
            print(audit.head(30).to_string(index=False))
        raise SystemExit(2)

    log(f"selected perturbed latent file={selected_path}")

    pert = load_perturbed_table(selected_path)
    perturb_cols = detect_perturb_cols(pert)
    zp_cols = detect_zp_cols(pert, expected_dim=expected_dim)
    delta_cols = detect_delta_cols(pert, expected_dim=expected_dim)

    meta_table, Xp, construction_meta = build_row_level_from_perturbed(
        pert=pert,
        baseline=baseline,
        latent_cols=latent_cols,
        zp_cols=zp_cols,
        delta_cols=delta_cols,
        perturb_cols=perturb_cols,
        expected_dim=expected_dim,
        expand_perturbation_level_delta=args.expand_perturbation_level_delta,
    )

    log(f"construction mode={construction_meta['mode']}")
    log(f"row-level perturbed rows={len(meta_table)}")

    # Merge baseline truth and baseline prediction by obs_name.
    base_info = baseline[["obs_name"] + target_cols].copy()
    for j, c in enumerate(target_cols):
        base_info[f"NeighborHead_baseline_pred_{safe_name(c)}"] = Y_base_pred[:, j]

    meta_table["_obs_key"] = meta_table["obs_name"].astype(str).map(norm_id)
    base_info["_obs_key"] = base_info["obs_name"].astype(str).map(norm_id)

    merged = meta_table.merge(base_info, on="_obs_key", how="left", suffixes=("", "_baseline"))
    merged = merged.drop(columns=["_obs_key"])

    baseline_true = merged[target_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)
    baseline_pred_cols = [f"NeighborHead_baseline_pred_{safe_name(c)}" for c in target_cols]
    baseline_pred = merged[baseline_pred_cols].apply(pd.to_numeric, errors="coerce").values.astype(float)

    # Predict neighbor_i^p.
    Yp = predict_neighbor(
        model,
        model_meta,
        Xp,
        device=args.device,
        batch_size=args.batch_size,
    )

    out = merged.copy()

    # Add perturbed predictions.
    for j, c in enumerate(target_cols):
        out[f"NeighborHead_perturbed_pred_{safe_name(c)}"] = Yp[:, j]

    out = add_distance_metrics(
        out=out,
        pred_p=Yp,
        baseline_true=baseline_true,
        baseline_pred=baseline_pred,
        target_cols=target_cols,
        prefix="neural_neighbor",
    )

    # Remove duplicated obs_name_baseline if present.
    if "obs_name_baseline" in out.columns:
        out = out.drop(columns=["obs_name_baseline"])

    # Choose perturbation group columns for summary.
    summary_perturb_cols = [c for c in perturb_cols if c in out.columns]
    if not summary_perturb_cols and "_perturbation_row_index" in out.columns:
        summary_perturb_cols = ["_perturbation_row_index"]

    summary = summarize_by_perturbation(out, summary_perturb_cols, prefix="neural_neighbor")
    group_summary = summarize_by_group_metrics(out, summary_perturb_cols, prefix="neural_neighbor")

    # Light output columns.
    light_cols = []
    for c in [
        "obs_name",
        "timepoint",
        "region_true",
        "dominant_celltype",
    ]:
        if c in out.columns:
            light_cols.append(c)

    for c in summary_perturb_cols:
        if c in out.columns and c not in light_cols:
            light_cols.append(c)

    for c in [
        "neural_neighbor_shift_l2_to_true_neighbor",
        "neural_neighbor_shift_mae_to_true_neighbor",
        "neural_neighbor_shift_cosine_to_true_neighbor",
        "neural_neighbor_shift_l2_from_baseline_pred",
        "neural_neighbor_shift_mae_from_baseline_pred",
        "neural_neighbor_shift_cosine_from_baseline_pred",
        "neural_neighbor_shift_l2_to_true_RCTD",
        "neural_neighbor_shift_l2_to_true_region",
        "neural_neighbor_shift_l2_to_true_celltype",
        "neural_neighbor_delta_repair_score_vs_true",
        "neural_neighbor_delta_core_prob_vs_true",
        "neural_neighbor_delta_remote_prob_vs_true",
        "neural_neighbor_direction_proxy",
    ]:
        if c in out.columns:
            light_cols.append(c)

    light_cols = [c for c in light_cols if c in out.columns]

    row_path = outdir / "perturbed_neighbor_head_predictions.csv"
    light_path = outdir / "perturbed_neighbor_head_predictions_light.csv"
    summary_path = outdir / "perturbation_neural_neighbor_shift_summary.csv"
    group_summary_path = outdir / "perturbation_neural_neighbor_shift_target_group_summary.csv"
    meta_path = outdir / "neighbor_head_54b_metadata.json"
    report_path = outdir / "neighbor_head_54b_report.txt"

    if not args.no_save_row_level:
        out.to_csv(row_path, index=False)
    else:
        row_path = None

    out[light_cols].to_csv(light_path, index=False)
    summary.to_csv(summary_path, index=False)
    group_summary.to_csv(group_summary_path, index=False)

    metadata = {
        "status": "ok",
        "model": str(model_path),
        "baseline_latent": str(baseline_latent_path),
        "profile53c": str(profile53c_path),
        "perturbed_latent": str(selected_path),
        "model_meta": {
            "input_dim": model_meta["input_dim"],
            "output_dim": model_meta["output_dim"],
            "hidden": model_meta["hidden"],
            "best_epoch": model_meta["best_epoch"],
            "best_val_loss": model_meta["best_val_loss"],
        },
        "construction_meta": construction_meta,
        "perturb_cols": summary_perturb_cols,
        "target_cols": target_cols,
        "target_groups": group_indices_from_targets(target_cols),
        "n_rows": int(len(out)),
        "n_summary_rows": int(len(summary)),
        "audit": str(audit_path),
        "outputs": {
            "row_level": str(row_path) if row_path else None,
            "light": str(light_path),
            "summary": str(summary_path),
            "group_summary": str(group_summary_path),
            "metadata": str(meta_path),
            "report": str(report_path),
        },
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = []
    lines.append("54b NeighborHead application to perturbed latent z_i^p report")
    lines.append("=" * 100)
    lines.append("SUCCESS")
    lines.append("")
    lines.append(f"model: {model_path}")
    lines.append(f"baseline_latent: {baseline_latent_path}")
    lines.append(f"profile53c: {profile53c_path}")
    lines.append(f"perturbed_latent: {selected_path}")
    lines.append(f"construction_mode: {construction_meta['mode']}")
    lines.append(f"n_rows: {len(out)}")
    lines.append(f"n_perturbation_summary_rows: {len(summary)}")
    lines.append("")
    lines.append("Model:")
    lines.append(f"input_dim: {model_meta['input_dim']}")
    lines.append(f"output_dim: {model_meta['output_dim']}")
    lines.append(f"best_epoch: {model_meta['best_epoch']}")
    lines.append(f"best_val_loss: {model_meta['best_val_loss']}")
    lines.append("")
    lines.append("Distance metrics summary:")
    for c in [
        "neural_neighbor_shift_l2_to_true_neighbor",
        "neural_neighbor_shift_l2_from_baseline_pred",
        "neural_neighbor_shift_cosine_to_true_neighbor",
        "neural_neighbor_shift_cosine_from_baseline_pred",
        "neural_neighbor_delta_repair_score_vs_true",
        "neural_neighbor_delta_core_prob_vs_true",
    ]:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce")
            lines.append(f"{c}: n={x.notna().sum()}, mean={x.mean()}, median={x.median()}, p90={x.quantile(0.90)}, max={x.max()}")
    lines.append("")
    lines.append("Direction proxy:")
    if "neural_neighbor_direction_proxy" in out.columns:
        lines.append(out["neural_neighbor_direction_proxy"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("Summary head:")
    lines.append(summary.head(50).to_string(index=False))
    lines.append("")
    lines.append("Light output head:")
    lines.append(out[light_cols].head(30).to_string(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append(
        "neighbor_i^p was predicted by applying the strict-z NeighborHead to perturbed latent states z_i^p. "
        "neural_neighbor_shift_distance values quantify how much the predicted neighborhood profile changes "
        "relative to the baseline true 53c neighbor_i and to the baseline NeighborHead prediction."
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE 54b apply NeighborHead to perturbed latent z_i^p")
    log("=" * 100)
    if row_path:
        log(f"Saved: {row_path}")
    log(f"Saved: {light_path}")
    log(f"Saved: {summary_path}")
    log(f"Saved: {group_summary_path}")
    log(f"Saved: {meta_path}")
    log(f"Saved: {report_path}")
    log("")
    print("Summary head:")
    print(summary.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
