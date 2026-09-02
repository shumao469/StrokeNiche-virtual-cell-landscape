#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
54_train_neural_neighbor_head.py

Purpose
-------
Train a neural NeighborHead:

    NeighborHead(z_i) -> neighbor_i

where:
  z_i        = StrokeNiche Adapter latent / embedding for each virtual cell
  neighbor_i = 53c kNN-RCTD neighborhood vector

This is the first full neural NeighborHead training step after:
  53  perturbation-level neighbor_shift_distance_proxy
  53b coordinate-based virtual-cell kNN profile
  53c RCTD-enhanced kNN neighborhood profile

Inputs
------
1) 53c profile:
   /mnt/h/vir/ST/results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/
     virtual_cell_knn_neighbor_profile_RCTD.csv

2) Adapter latent table, ideally containing obs_name and numeric latent columns:
   examples:
     adapter_latent_0, adapter_latent_1, ...
     z_0, z_1, ...
     latent_0, latent_1, ...
     embedding_0, embedding_1, ...
     PC_1, PC_2, ...

Outputs
-------
/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/neighbor_head_54/
  neighbor_head_model.pt
  neighbor_head_metadata.json
  neighbor_head_training_history.csv
  neighbor_head_per_target_metrics.csv
  neighbor_head_predictions.csv
  neighbor_head_predictions_light.csv
  neighbor_head_report.txt

Important
---------
If no Adapter latent is found, the script will stop by default.
You may use --allow_feature_fallback for debugging only, but that is not a true
Adapter-latent NeighborHead.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


BASE_DEFAULT = Path("/mnt/h/vir/ST")
PROFILE53C_DEFAULT = BASE_DEFAULT / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/virtual_cell_knn_neighbor_profile_RCTD.csv"
OUTDIR_DEFAULT = BASE_DEFAULT / "results/step8_strokeniche_perturbmap/neighbor_head_54"


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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def pick_id_col(df: pd.DataFrame) -> Optional[str]:
    preferred = [
        "obs_name",
        "cell_id",
        "cell",
        "barcode",
        "spot_id",
        "spot",
        "id",
        "index_id",
        "index",
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


def find_latent_candidates(base: Path) -> List[Path]:
    patterns = [
        "results/step7_strokeniche_adapter/**/*latent*.csv",
        "results/step7_strokeniche_adapter/**/*embedding*.csv",
        "results/step7_strokeniche_adapter/**/*emb*.csv",
        "results/step7_strokeniche_adapter_clean/**/*latent*.csv",
        "results/step7_strokeniche_adapter_clean/**/*embedding*.csv",
        "results/step7_strokeniche_adapter_clean/**/*emb*.csv",
        "results/step8_strokeniche_perturbmap/**/*latent*.csv",
        "results/step8_strokeniche_perturbmap/**/*embedding*.csv",
        "results/virtual_cell_inputs_bridge/**/*latent*.csv",
        "results/**/*adapter*latent*.csv",
        "results/**/*nicheformer*latent*.csv",
        "results/**/*virtual*latent*.csv",
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
            # Avoid huge non-cell-level files if possible.
            lname = p.name.lower()
            if any(bad in lname for bad in ["drug", "chembl", "ctd", "lincs_signature"]):
                continue
            hits.append(p)

    return sorted(hits, key=lambda p: (len(str(p)), str(p)))


def detect_latent_cols(df: pd.DataFrame, min_cols: int = 4) -> List[str]:
    cols = []

    positive_patterns = [
        r"^z[_]?\d+$",
        r"^latent[_]?\d+$",
        r"^adapter_latent[_]?\d+$",
        r"^adapter_z[_]?\d+$",
        r"^embedding[_]?\d+$",
        r"^emb[_]?\d+$",
        r"^nicheformer[_]?\d+$",
        r"^x_scvi[_]?\d+$",
        r"^scvi[_]?\d+$",
        r"^pc[_]?\d+$",
        r"^pca[_]?\d+$",
        r"^dim[_]?\d+$",
    ]

    exclude_terms = [
        "neighbor",
        "rctd",
        "region",
        "celltype",
        "spatial",
        "coord",
        "distance",
        "score",
        "prob",
        "delta",
        "rank",
        "time",
        "phase",
        "label",
        "class",
        "id",
        "barcode",
    ]

    for c in df.columns:
        nc = norm_col(c)
        if any(ex in nc for ex in exclude_terms):
            continue
        if any(re.search(p, nc) for p in positive_patterns):
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
                cols.append(c)

    if len(cols) >= min_cols:
        return cols

    # Second pass: columns containing latent/embedding/z but not obvious target.
    for c in df.columns:
        if c in cols:
            continue
        nc = norm_col(c)
        if any(ex in nc for ex in exclude_terms):
            continue
        if any(t in nc for t in ["latent", "embedding", "adapter", "nicheformer"]):
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
                cols.append(c)

    return cols


def load_latent_table(base: Path, latent_arg: Optional[str], profile: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str], Dict]:
    """
    Return latent_df, latent_cols, metadata.
    If latent_arg is given, use it.
    Else, first check profile columns, then search candidate files.
    """
    meta = {
        "latent_arg": latent_arg,
        "source": None,
        "id_col": None,
        "latent_cols": [],
        "candidate_files_checked": [],
        "matched_rows": 0,
    }

    # 1. Check profile itself.
    cols_in_profile = detect_latent_cols(profile)
    if len(cols_in_profile) >= 4:
        meta["source"] = "profile53c_columns"
        meta["id_col"] = pick_id_col(profile)
        meta["latent_cols"] = cols_in_profile
        meta["matched_rows"] = len(profile)
        return profile.copy(), cols_in_profile, meta

    paths = []
    if latent_arg:
        paths = [Path(latent_arg)]
    else:
        paths = find_latent_candidates(base)

    profile_id = pick_id_col(profile)
    if profile_id is None:
        raise RuntimeError("Cannot detect obs/cell ID column in profile53c.")

    profile_keys = set(profile[profile_id].dropna().astype(str).map(norm_id))

    best = None
    best_cols = []
    best_meta = None
    best_match = -1

    for p in paths:
        if not p.exists():
            continue

        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue

        id_col = pick_id_col(df)
        latent_cols = detect_latent_cols(df)

        checked = {
            "path": str(p),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "id_col": id_col,
            "n_latent_cols": len(latent_cols),
            "latent_cols_head": latent_cols[:20],
        }

        match = 0
        if id_col:
            keys = set(df[id_col].dropna().astype(str).map(norm_id))
            match = len(profile_keys & keys)
            checked["matched_rows_preview"] = match

        meta["candidate_files_checked"].append(checked)

        if id_col and len(latent_cols) >= 4 and match > best_match:
            best = df
            best_cols = latent_cols
            best_match = match
            best_meta = {
                "source": str(p),
                "id_col": id_col,
                "latent_cols": latent_cols,
                "matched_rows": match,
            }

    if best is not None and best_match > 0:
        meta.update(best_meta)
        return best, best_cols, meta

    return None, [], meta


def detect_target_cols(profile: pd.DataFrame, target_mode: str) -> List[str]:
    cols = []

    if target_mode in ["all", "rctd_region_celltype_score"]:
        prefixes = [
            "neighbor_mean_feature_RCTD_",
            "neighbor_mean_feature_regionprob_",
            "neighbor_mean_feature_celltype_",
            "neighbor_mean_feature_score_",
        ]
    elif target_mode == "rctd_only":
        prefixes = ["neighbor_mean_feature_RCTD_"]
    elif target_mode == "rctd_region":
        prefixes = [
            "neighbor_mean_feature_RCTD_",
            "neighbor_mean_feature_regionprob_",
        ]
    elif target_mode == "rctd_celltype":
        prefixes = [
            "neighbor_mean_feature_RCTD_",
            "neighbor_mean_feature_celltype_",
        ]
    else:
        prefixes = [
            "neighbor_mean_feature_RCTD_",
            "neighbor_mean_feature_regionprob_",
            "neighbor_mean_feature_celltype_",
            "neighbor_mean_feature_score_",
        ]

    for c in profile.columns:
        if any(c.startswith(p) for p in prefixes):
            x = pd.to_numeric(profile[c], errors="coerce")
            if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
                cols.append(c)

    return cols


def build_feature_fallback(profile: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Debug-only fallback: use self state features instead of adapter latent.
    Not considered a true NeighborHead(z_i).
    """
    candidates = []

    prefixes = [
        "RCTD_prop_",
    ]
    for c in profile.columns:
        if any(c.startswith(p) for p in prefixes):
            candidates.append(c)

    for c in profile.columns:
        nc = norm_col(c)
        if (
            ("prob" in nc and any(t in nc for t in ["core", "peri", "remote", "lesion"]))
            or nc in ["repair_score", "repair_score_from_npy"]
            or "repair_score" in nc
        ):
            if not c.startswith("neighbor_"):
                candidates.append(c)

    for c in ["spatial_x", "spatial_y"]:
        if c in profile.columns:
            candidates.append(c)

    # Deduplicate numeric.
    cols = []
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        x = pd.to_numeric(profile[c], errors="coerce")
        if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
            cols.append(c)

    return profile.copy(), cols, {
        "source": "feature_fallback_from_profile53c",
        "id_col": pick_id_col(profile),
        "latent_cols": cols,
        "warning": "This is not a true Adapter-latent NeighborHead.",
        "matched_rows": len(profile),
    }


def merge_profile_and_latent(profile: pd.DataFrame, latent_df: pd.DataFrame, latent_cols: List[str], latent_meta: Dict) -> Tuple[pd.DataFrame, Dict]:
    profile_id = pick_id_col(profile)
    latent_id = latent_meta.get("id_col") or pick_id_col(latent_df)

    if profile_id is None:
        raise RuntimeError("Cannot detect profile ID column.")
    if latent_id is None:
        raise RuntimeError("Cannot detect latent ID column.")

    p = profile.copy()
    l = latent_df[[latent_id] + latent_cols].copy()

    p["_neighbor_head_id"] = p[profile_id].astype(str).map(norm_id)
    l["_neighbor_head_id"] = l[latent_id].astype(str).map(norm_id)
    l = l.drop_duplicates("_neighbor_head_id", keep="first")

    merged = p.merge(l[["_neighbor_head_id"] + latent_cols], on="_neighbor_head_id", how="left", suffixes=("", "_latent"))

    matched = int(merged[latent_cols].notna().any(axis=1).sum())

    meta = {
        "profile_id_col": profile_id,
        "latent_id_col": latent_id,
        "profile_rows": int(len(profile)),
        "latent_rows": int(len(latent_df)),
        "matched_rows": matched,
        "latent_cols": latent_cols,
    }

    return merged, meta


def train_val_split_indices(df: pd.DataFrame, val_fraction: float, seed: int, holdout_timepoint: Optional[str]):
    n = len(df)
    idx = np.arange(n)

    if holdout_timepoint and "timepoint" in df.columns:
        tp = df["timepoint"].astype(str).str.upper()
        val_mask = tp.eq(holdout_timepoint.upper()).values
        train_idx = idx[~val_mask]
        val_idx = idx[val_mask]
        if len(train_idx) > 0 and len(val_idx) > 0:
            return train_idx, val_idx, f"holdout_timepoint={holdout_timepoint}"

    try:
        from sklearn.model_selection import train_test_split

        stratify = None
        if "timepoint" in df.columns and "region_true" in df.columns:
            s = df["timepoint"].astype(str) + "_" + df["region_true"].astype(str)
            vc = s.value_counts()
            if vc.min() >= 2 and vc.shape[0] < n * 0.5:
                stratify = s

        train_idx, val_idx = train_test_split(
            idx,
            test_size=val_fraction,
            random_state=seed,
            stratify=stratify,
        )
        return train_idx, val_idx, "random_stratified" if stratify is not None else "random"
    except Exception:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_val = max(1, int(n * val_fraction))
        return idx[n_val:], idx[:n_val], "random_numpy"


def standardize_train_val_all(X: np.ndarray, train_idx: np.ndarray):
    mean = np.nanmean(X[train_idx], axis=0)
    std = np.nanstd(X[train_idx], axis=0)
    std[std < 1e-8] = 1.0

    Xs = (X - mean) / std
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    return Xs.astype(np.float32), mean.astype(float), std.astype(float)


def group_target_indices(target_cols: List[str]) -> Dict[str, List[int]]:
    groups = {
        "RCTD": [],
        "region": [],
        "celltype": [],
        "score": [],
    }
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


def compute_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    mse = mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)

    try:
        r2 = r2_score(yt, yp, multioutput="variance_weighted")
    except Exception:
        r2 = np.nan

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(math.sqrt(mse)),
        "r2_variance_weighted": float(r2) if np.isfinite(r2) else np.nan,
    }


def cosine_distance_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    num = np.sum(a * b, axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-12)
    return 1.0 - cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE_DEFAULT))
    ap.add_argument("--profile53c", default=str(PROFILE53C_DEFAULT))
    ap.add_argument("--latent", default=None, help="Optional Adapter latent CSV path.")
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT))
    ap.add_argument("--target_mode", default="all", choices=["all", "rctd_only", "rctd_region", "rctd_celltype", "rctd_region_celltype_score"])
    ap.add_argument("--allow_feature_fallback", action="store_true", help="Debug only: use profile features if no Adapter latent is found.")
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--hidden", default="256,128", help="Comma-separated hidden sizes.")
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_fraction", type=float, default=0.20)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--composition_weight", type=float, default=0.05)
    ap.add_argument("--holdout_timepoint", default=None, help="Optional, e.g. D7.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    set_seed(args.seed)

    base = Path(args.base)
    profile_path = Path(args.profile53c)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not profile_path.exists():
        raise FileNotFoundError(profile_path)

    profile = pd.read_csv(profile_path, low_memory=False)

    target_cols = detect_target_cols(profile, args.target_mode)
    if not target_cols:
        raise RuntimeError("No NeighborHead target columns detected from 53c profile.")

    latent_df, latent_cols, latent_meta = load_latent_table(base, args.latent, profile)

    used_feature_fallback = False
    if latent_df is None or len(latent_cols) < 4:
        if args.allow_feature_fallback:
            latent_df, latent_cols, latent_meta = build_feature_fallback(profile)
            used_feature_fallback = True
        else:
            meta_path = outdir / "neighbor_head_preflight_no_latent_found.json"
            latent_meta["error"] = "No usable Adapter latent table found. Provide --latent or use --allow_feature_fallback for debugging only."
            latent_meta["target_cols"] = target_cols
            meta_path.write_text(json.dumps(latent_meta, indent=2, ensure_ascii=False), encoding="utf-8")

            print("=" * 90)
            print("PRE-FLIGHT FAILED: no Adapter latent found")
            print("=" * 90)
            print("Target columns detected:", len(target_cols))
            print("Candidate latent files checked:", len(latent_meta.get("candidate_files_checked", [])))
            print("Saved audit:", meta_path)
            print("")
            print("Find latent files manually with:")
            print('find /mnt/h/vir/ST/results -type f \\( -iname "*latent*.csv" -o -iname "*embedding*.csv" -o -iname "*emb*.csv" \\) | head -100')
            print("")
            print("Then rerun with:")
            print("$PY 54_train_neural_neighbor_head.py --latent /path/to/adapter_latent.csv")
            raise SystemExit(2)

    merged, merge_meta = merge_profile_and_latent(profile, latent_df, latent_cols, latent_meta)

    # Prepare matrices.
    X_df = merged[latent_cols].apply(pd.to_numeric, errors="coerce")
    Y_df = merged[target_cols].apply(pd.to_numeric, errors="coerce")

    valid = X_df.notna().all(axis=1) & Y_df.notna().all(axis=1)
    train_df = merged.loc[valid].reset_index(drop=True)
    X = X_df.loc[valid].values.astype(float)
    Y = Y_df.loc[valid].values.astype(np.float32)

    if len(train_df) < 50:
        raise RuntimeError(f"Too few valid training rows: {len(train_df)}")

    train_idx, val_idx, split_mode = train_val_split_indices(
        train_df,
        val_fraction=args.val_fraction,
        seed=args.seed,
        holdout_timepoint=args.holdout_timepoint,
    )

    Xs, x_mean, x_std = standardize_train_val_all(X, train_idx)

    # Target range check.
    y_min = np.nanmin(Y)
    y_max = np.nanmax(Y)
    use_sigmoid = bool(y_min >= -1e-6 and y_max <= 1.000001)

    groups = group_target_indices(target_cols)

    metadata = {
        "status": "preflight_ok",
        "profile53c": str(profile_path),
        "profile_shape": [int(profile.shape[0]), int(profile.shape[1])],
        "latent_meta": latent_meta,
        "merge_meta": merge_meta,
        "used_feature_fallback": used_feature_fallback,
        "target_mode": args.target_mode,
        "target_cols": target_cols,
        "target_groups": groups,
        "n_valid_training_rows": int(len(train_df)),
        "input_dim": int(X.shape[1]),
        "output_dim": int(Y.shape[1]),
        "split_mode": split_mode,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "use_sigmoid_output": use_sigmoid,
        "latent_cols": latent_cols,
    }

    preflight_path = outdir / "neighbor_head_preflight_metadata.json"
    preflight_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 90)
    print("54 neural NeighborHead preflight")
    print("=" * 90)
    print(f"profile53c: {profile_path}")
    print(f"rows profile: {profile.shape}")
    print(f"latent source: {latent_meta.get('source')}")
    print(f"used_feature_fallback: {used_feature_fallback}")
    print(f"matched rows: {merge_meta.get('matched_rows')} / {merge_meta.get('profile_rows')}")
    print(f"input_dim: {X.shape[1]}")
    print(f"output_dim: {Y.shape[1]}")
    print(f"target columns: {len(target_cols)}")
    print(f"valid training rows: {len(train_df)}")
    print(f"split: {split_mode}, train={len(train_idx)}, val={len(val_idx)}")
    print(f"use_sigmoid_output: {use_sigmoid}")
    print(f"preflight metadata saved: {preflight_path}")

    if args.preflight_only:
        print("Preflight only. Stop.")
        return

    # Import torch after preflight.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    hidden = [int(x) for x in args.hidden.split(",") if x.strip()]
    input_dim = X.shape[1]
    output_dim = Y.shape[1]

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

    model = NeighborHead(input_dim, output_dim, hidden, args.dropout, use_sigmoid).to(device)

    X_train = torch.tensor(Xs[train_idx], dtype=torch.float32)
    Y_train = torch.tensor(Y[train_idx], dtype=torch.float32)
    X_val = torch.tensor(Xs[val_idx], dtype=torch.float32)
    Y_val = torch.tensor(Y[val_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    group_indices = {
        k: torch.tensor(v, dtype=torch.long, device=device)
        for k, v in groups.items()
        if len(v) > 1 and k in ["RCTD", "celltype"]
    }

    def loss_fn(pred, y):
        mse = F.mse_loss(pred, y)
        comp = torch.tensor(0.0, device=pred.device)

        for g, idx in group_indices.items():
            ps = pred.index_select(1, idx).sum(dim=1)
            ys = y.index_select(1, idx).sum(dim=1)
            comp = comp + F.mse_loss(ps, ys)

        return mse + args.composition_weight * comp, mse.detach(), comp.detach()

    best_val = float("inf")
    best_state = None
    best_epoch = -1
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        train_mse = []
        train_comp = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, mse_part, comp_part = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            train_losses.append(float(loss.item()))
            train_mse.append(float(mse_part.item()))
            train_comp.append(float(comp_part.item()))

        model.eval()
        with torch.no_grad():
            pv = model(X_val.to(device))
            val_loss, val_mse, val_comp = loss_fn(pv, Y_val.to(device))

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_mse": float(np.mean(train_mse)),
            "train_comp": float(np.mean(train_comp)),
            "val_loss": float(val_loss.item()),
            "val_mse": float(val_mse.item()),
            "val_comp": float(val_comp.item()),
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(row)

        if row["val_loss"] < best_val - 1e-8:
            best_val = row["val_loss"]
            best_epoch = epoch
            best_state = {
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "hidden": hidden,
                "dropout": args.dropout,
                "sigmoid_output": use_sigmoid,
                "latent_cols": latent_cols,
                "target_cols": target_cols,
                "x_mean": x_mean,
                "x_std": x_std,
                "target_groups": groups,
                "used_feature_fallback": used_feature_fallback,
            }
            no_improve = 0
        else:
            no_improve += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d} "
                f"train_loss={row['train_loss']:.6f} "
                f"val_loss={row['val_loss']:.6f} "
                f"val_mse={row['val_mse']:.6f} "
                f"best_epoch={best_epoch}"
            )

        if no_improve >= args.patience:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}, best_val_loss={best_val:.6f}")
            break

    # Restore best.
    model.load_state_dict(best_state["model_state_dict"])

    # Predict all valid rows.
    model.eval()
    with torch.no_grad():
        pred_all = []
        batch = 4096
        Xt = torch.tensor(Xs, dtype=torch.float32)
        for i in range(0, len(Xt), batch):
            pred = model(Xt[i:i+batch].to(device)).cpu().numpy()
            pred_all.append(pred)
        pred_all = np.vstack(pred_all)

    # Metrics.
    train_metrics = compute_metrics_np(Y[train_idx], pred_all[train_idx])
    val_metrics = compute_metrics_np(Y[val_idx], pred_all[val_idx])
    all_metrics = compute_metrics_np(Y, pred_all)

    per_target = []
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    for j, c in enumerate(target_cols):
        yt = Y[:, j]
        yp = pred_all[:, j]
        if np.unique(yt).shape[0] > 1:
            try:
                r2 = float(r2_score(yt, yp))
            except Exception:
                r2 = np.nan
        else:
            r2 = np.nan
        per_target.append({
            "target_col": c,
            "target_group": (
                "RCTD" if c.startswith("neighbor_mean_feature_RCTD_")
                else "region" if c.startswith("neighbor_mean_feature_regionprob_")
                else "celltype" if c.startswith("neighbor_mean_feature_celltype_")
                else "score" if c.startswith("neighbor_mean_feature_score_")
                else "other"
            ),
            "mse": float(mean_squared_error(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "r2": r2,
            "target_mean": float(np.mean(yt)),
            "pred_mean": float(np.mean(yp)),
        })

    per_target_df = pd.DataFrame(per_target)

    # Build predictions table.
    pred_df = train_df.copy()

    for j, c in enumerate(target_cols):
        pname = "NeighborHead_pred_" + safe_name(c.replace("neighbor_mean_", ""))
        ename = "NeighborHead_error_" + safe_name(c.replace("neighbor_mean_", ""))
        pred_df[pname] = pred_all[:, j]
        pred_df[ename] = pred_all[:, j] - Y[:, j]

    pred_df["NeighborHead_error_l2"] = np.sqrt(((pred_all - Y) ** 2).sum(axis=1))
    pred_df["NeighborHead_error_mae_row"] = np.mean(np.abs(pred_all - Y), axis=1)
    pred_df["NeighborHead_error_cosine_distance"] = cosine_distance_rows(Y, pred_all)

    # Group-wise distances.
    for group_name, idxs in groups.items():
        if idxs:
            idxs = list(idxs)
            pred_df[f"NeighborHead_error_l2_{group_name}"] = np.sqrt(((pred_all[:, idxs] - Y[:, idxs]) ** 2).sum(axis=1))
            pred_df[f"NeighborHead_error_mae_{group_name}"] = np.mean(np.abs(pred_all[:, idxs] - Y[:, idxs]), axis=1)
            pred_df[f"NeighborHead_cosine_distance_{group_name}"] = cosine_distance_rows(Y[:, idxs], pred_all[:, idxs])

    # Normalized error.
    e = pred_df["NeighborHead_error_l2"]
    if e.max() > e.min():
        pred_df["NeighborHead_error_l2_norm"] = (e - e.min()) / (e.max() - e.min())
    else:
        pred_df["NeighborHead_error_l2_norm"] = 0.0

    pred_df["NeighborHead_model_available"] = True
    pred_df["NeighborHead_training_mode"] = (
        "AdapterLatent_to_kNN_RCTD_neighbor_profile"
        if not used_feature_fallback
        else "FEATURE_FALLBACK_not_true_adapter_latent"
    )

    # Light output.
    light_cols = [
        "obs_name",
        "timepoint",
        "region_true",
        "dominant_celltype",
        "spatial_x",
        "spatial_y",
        "NeighborHead_model_available",
        "NeighborHead_training_mode",
        "NeighborHead_error_l2",
        "NeighborHead_error_l2_norm",
        "NeighborHead_error_mae_row",
        "NeighborHead_error_cosine_distance",
        "NeighborHead_error_l2_RCTD",
        "NeighborHead_error_mae_RCTD",
        "NeighborHead_cosine_distance_RCTD",
        "NeighborHead_error_l2_region",
        "NeighborHead_error_l2_celltype",
        "knn_neighbor_RCTD_entropy",
        "knn_neighbor_RCTD_composition_mixedness",
        "knn_neighbor_region_boundary_score",
        "knn_neighbor_celltype_mixedness",
    ]
    light_cols = [c for c in light_cols if c in pred_df.columns]

    # Save files.
    model_path = outdir / "neighbor_head_model.pt"
    history_path = outdir / "neighbor_head_training_history.csv"
    per_target_path = outdir / "neighbor_head_per_target_metrics.csv"
    pred_path = outdir / "neighbor_head_predictions.csv"
    pred_light_path = outdir / "neighbor_head_predictions_light.csv"
    meta_path = outdir / "neighbor_head_metadata.json"
    report_path = outdir / "neighbor_head_report.txt"
    scaler_path = outdir / "neighbor_head_input_scaler.pkl"

    torch.save(best_state, model_path)
    pd.DataFrame(history).to_csv(history_path, index=False)
    per_target_df.to_csv(per_target_path, index=False)
    pred_df.to_csv(pred_path, index=False)
    pred_df[light_cols].to_csv(pred_light_path, index=False)

    with open(scaler_path, "wb") as f:
        pickle.dump({"x_mean": x_mean, "x_std": x_std, "latent_cols": latent_cols}, f)

    final_meta = metadata.copy()
    final_meta.update({
        "status": "ok",
        "device": device,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "all_metrics": all_metrics,
        "model_path": str(model_path),
        "history_path": str(history_path),
        "per_target_metrics_path": str(per_target_path),
        "predictions_path": str(pred_path),
        "predictions_light_path": str(pred_light_path),
        "scaler_path": str(scaler_path),
    })

    meta_path.write_text(json.dumps(final_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Report.
    lines = []
    lines.append("54 neural NeighborHead training report")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Model:")
    lines.append("  NeighborHead(z_i) -> neighbor_i")
    lines.append("")
    lines.append(f"profile53c: {profile_path}")
    lines.append(f"latent source: {latent_meta.get('source')}")
    lines.append(f"used_feature_fallback: {used_feature_fallback}")
    lines.append(f"input_dim: {input_dim}")
    lines.append(f"output_dim: {output_dim}")
    lines.append(f"target_mode: {args.target_mode}")
    lines.append(f"n_valid_training_rows: {len(train_df)}")
    lines.append(f"split_mode: {split_mode}")
    lines.append(f"n_train: {len(train_idx)}")
    lines.append(f"n_val: {len(val_idx)}")
    lines.append(f"hidden: {hidden}")
    lines.append(f"dropout: {args.dropout}")
    lines.append(f"best_epoch: {best_epoch}")
    lines.append(f"best_val_loss: {best_val}")
    lines.append("")
    lines.append("Metrics:")
    lines.append(f"train: {train_metrics}")
    lines.append(f"val: {val_metrics}")
    lines.append(f"all: {all_metrics}")
    lines.append("")
    lines.append("Target groups:")
    lines.append(json.dumps({k: len(v) for k, v in groups.items()}, indent=2))
    lines.append("")
    lines.append("Top per-target metrics by worst R2:")
    lines.append(per_target_df.sort_values("r2", ascending=True, na_position="last").head(50).to_string(index=False))
    lines.append("")
    lines.append("Prediction light head:")
    lines.append(pred_df[light_cols].head(30).to_string(index=False))
    lines.append("")
    if used_feature_fallback:
        lines.append("WARNING:")
        lines.append("  This run used --allow_feature_fallback. It is useful for debugging but should not be described as a true Adapter-latent NeighborHead.")
    else:
        lines.append("Interpretation:")
        lines.append("  This run trained a neural NeighborHead from Adapter latent features to the 53c kNN-RCTD neighborhood vector.")
    lines.append("")
    lines.append("Downstream:")
    lines.append("  To predict neighbor_i^p, apply this trained NeighborHead to perturbed latent states z_i^p from the PerturbDecoder.")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 90)
    print("DONE 54 neural NeighborHead training")
    print("=" * 90)
    print(f"Saved: {model_path}")
    print(f"Saved: {history_path}")
    print(f"Saved: {per_target_path}")
    print(f"Saved: {pred_path}")
    print(f"Saved: {pred_light_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {report_path}")
    print("")
    print("Metrics:")
    print("train:", train_metrics)
    print("val:", val_metrics)
    print("all:", all_metrics)
    print("")
    print("Top light predictions:")
    print(pred_df[light_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
