#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
54a_true_export_adapter_latent_from_checkpoint_v2.py

Fixes v1 failure:
  checkpoint expected input_dim=522
  but input NPZ stores separate arrays:
    X_nf          (7756, 512)
    coords_scaled (7756, 2)
    Y_neighbor    (7756, 8)

522 = 512 + 2 + 8

This script reconstructs Adapter input by concatenating the correct arrays,
then exports strict hidden Adapter latent z_i from Step7 checkpoint.

Output:
  results/step8_strokeniche_perturbmap/neighbor_head_54/
    adapter_true_hidden_latent_for_neighbor_head.csv
    adapter_true_hidden_latent_full.csv
    adapter_true_hidden_latent_for_neighbor_head_metadata.json
    adapter_true_hidden_latent_for_neighbor_head_report.txt
"""

from __future__ import annotations

import argparse
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


BASE = Path("/mnt/h/vir/ST")

DEFAULT_CKPT = BASE / "results/step7_strokeniche_ablation/runs/full/checkpoints/strokeniche_adapter_best.pt"
DEFAULT_PROFILE53C = BASE / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/virtual_cell_knn_neighbor_profile_RCTD.csv"
DEFAULT_OUTDIR = BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54"


def log(x):
    print(x, flush=True)


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def norm_id(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_resolve_path(p, base=BASE):
    if p is None:
        return None
    p = str(p)
    if not p or p.lower() == "none":
        return None
    q = Path(p)
    if q.exists():
        return q
    if not q.is_absolute():
        q2 = base / q
        if q2.exists():
            return q2
    q3 = base / p.lstrip("./")
    if q3.exists():
        return q3
    return q


def torch_load_trusted(path: Path):
    """
    For local self-generated checkpoints only.
    PyTorch 2.6 may require weights_only=False when checkpoint contains numpy objects.
    """
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def infer_dims_from_state(state: Dict[str, torch.Tensor]) -> Dict:
    w = state["input_proj.0.weight"]
    hidden_dim = int(w.shape[0])
    input_dim = int(w.shape[1])

    block_ids = []
    for k in state:
        m = re.match(r"blocks\.(\d+)\.q\.weight", k)
        if m:
            block_ids.append(int(m.group(1)))
    block_ids = sorted(set(block_ids))

    if block_ids and f"blocks.{block_ids[0]}.ffn.0.weight" in state:
        ffn_dim = int(state[f"blocks.{block_ids[0]}.ffn.0.weight"].shape[0])
    else:
        ffn_dim = hidden_dim * 2

    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "n_layers": len(block_ids),
        "block_ids": block_ids,
        "ffn_dim": ffn_dim,
    }


def pick_id_col(df):
    preferred = [
        "obs_name", "cell_id", "cell", "barcode", "spot_id", "spot",
        "id", "index_id", "index", "Unnamed: 0"
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


def load_metadata_obj(path: Optional[Path]):
    if path is None or not path.exists():
        return None
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix.lower() in [".csv", ".tsv", ".txt"]:
        sep = "\t" if path.suffix.lower() in [".tsv", ".txt"] else ","
        return pd.read_csv(path, sep=sep, low_memory=False)
    return None


def find_obs_names(npz, n_obs, metadata_json_path: Optional[Path], checkpoint_path: Path):
    preferred = [
        "obs_name", "obs_names", "cell_id", "cell_ids",
        "barcode", "barcodes", "spot_id", "spot_ids"
    ]

    for k in preferred:
        if k in npz:
            arr = npz[k]
            if len(arr) == n_obs:
                return [str(x) for x in arr], f"npz:{k}"

    obj = load_metadata_obj(metadata_json_path)

    if isinstance(obj, pd.DataFrame):
        id_col = pick_id_col(obj)
        if id_col and len(obj) == n_obs:
            return [str(x) for x in obj[id_col]], f"metadata_table:{metadata_json_path}:{id_col}"

    if isinstance(obj, dict):
        for k in preferred:
            if k in obj and isinstance(obj[k], list) and len(obj[k]) == n_obs:
                return [str(x) for x in obj[k]], f"metadata_json:{k}"

        for parent in ["obs", "metadata", "cell_metadata"]:
            if parent in obj and isinstance(obj[parent], dict):
                sub = obj[parent]
                for k in preferred:
                    if k in sub and isinstance(sub[k], list) and len(sub[k]) == n_obs:
                        return [str(x) for x in sub[k]], f"metadata_json:{parent}.{k}"

    possible = [
        checkpoint_path.parent.parent / "downstream/strokeniche_adapter_predictions.csv",
        checkpoint_path.parent / "downstream/strokeniche_adapter_predictions.csv",
        BASE / "results/step7_strokeniche_ablation/runs/full/downstream/strokeniche_adapter_predictions.csv",
        BASE / "results/step7_strokeniche_adapter/downstream/strokeniche_adapter_predictions.csv",
    ]

    for p in possible:
        if p.exists():
            df = pd.read_csv(p, low_memory=False)
            id_col = pick_id_col(df)
            if id_col and len(df) == n_obs:
                return [str(x) for x in df[id_col]], f"downstream_predictions:{p}:{id_col}"

    return [str(i) for i in range(n_obs)], "fallback_integer_order"


def get_2d_arrays(npz):
    out = {}
    for k in npz.keys():
        arr = npz[k]
        if hasattr(arr, "ndim") and arr.ndim == 2:
            out[k] = np.asarray(arr)
    return out


def build_input_matrix_from_npz(npz, expected_dim: int, ckpt_args: dict, ckpt_meta: dict):
    """
    Main fix.
    If no single array has expected_dim, concatenate stored components.

    For full Step7 adapter:
      X_nf + coords_scaled + Y_neighbor = 512 + 2 + 8 = 522
    """
    arrays = get_2d_arrays(npz)

    # 1. exact single matrix
    preferred_single = [
        "X", "x", "features", "input_features", "adapter_input",
        "X_input", "X_adapter", "X_full"
    ]
    for k in preferred_single:
        if k in arrays and arrays[k].shape[1] == expected_dim:
            return k, arrays[k].astype(np.float32), [(k, arrays[k].shape)]

    for k, arr in arrays.items():
        if arr.shape[1] == expected_dim and arr.shape[0] != 2:
            return k, arr.astype(np.float32), [(k, arr.shape)]

    # 2. known Step7 full order.
    known_orders = [
        ["X_nf", "coords_scaled", "Y_neighbor"],
        ["X_nf", "coords", "Y_neighbor"],
        ["X_nf", "Y_neighbor", "coords_scaled"],
        ["X_nf", "Y_neighbor", "coords"],
    ]

    for order in known_orders:
        if all(k in arrays for k in order):
            n = arrays[order[0]].shape[0]
            if all(arrays[k].shape[0] == n for k in order):
                dim = sum(arrays[k].shape[1] for k in order)
                if dim == expected_dim:
                    X = np.concatenate([arrays[k] for k in order], axis=1).astype(np.float32)
                    return "+".join(order), X, [(k, arrays[k].shape) for k in order]

    # 3. use ckpt meta hints.
    n_nf = ckpt_meta.get("n_nf_features", None)
    n_neighbor = ckpt_meta.get("n_neighbor_targets", None)

    if "X_nf" in arrays:
        candidate_keys = ["coords_scaled", "coords", "Y_neighbor"]
        existing = [k for k in candidate_keys if k in arrays]

        for r in range(1, len(existing) + 1):
            for comb in combinations(existing, r):
                order = ["X_nf"] + list(comb)
                n = arrays["X_nf"].shape[0]
                if all(arrays[k].shape[0] == n for k in order):
                    dim = sum(arrays[k].shape[1] for k in order)
                    if dim == expected_dim:
                        X = np.concatenate([arrays[k] for k in order], axis=1).astype(np.float32)
                        return "+".join(order), X, [(k, arrays[k].shape) for k in order]

    # 4. exhaustive small combination excluding edge_index-like arrays.
    keys = [k for k, a in arrays.items() if a.shape[0] != 2 and a.shape[1] < expected_dim]
    for r in range(2, min(5, len(keys)) + 1):
        for comb in combinations(keys, r):
            n = arrays[comb[0]].shape[0]
            if not all(arrays[k].shape[0] == n for k in comb):
                continue
            dim = sum(arrays[k].shape[1] for k in comb)
            if dim == expected_dim:
                X = np.concatenate([arrays[k] for k in comb], axis=1).astype(np.float32)
                return "+".join(comb), X, [(k, arrays[k].shape) for k in comb]

    two_d = [(k, v.shape) for k, v in arrays.items()]
    raise RuntimeError(
        f"Cannot reconstruct input_dim={expected_dim}. "
        f"2D arrays found: {two_d}. "
        f"For your checkpoint, expected likely X_nf + coords_scaled + Y_neighbor."
    )


class AdapterBlock(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, dropout):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, h):
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        gate = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) / math.sqrt(h.shape[-1]))
        h = self.norm1(h + self.o(gate * v))
        h = self.norm2(h + self.ffn(h))
        return h


class AdapterEncoderOnly(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, ffn_dim, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.blocks = nn.ModuleList([
            AdapterBlock(hidden_dim, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return h


def profile_match_stats(latent_df, profile_path):
    if profile_path is None or not profile_path.exists():
        return {"profile_exists": False}
    prof = pd.read_csv(profile_path, low_memory=False)
    id_col = pick_id_col(prof)
    if id_col is None:
        return {"profile_exists": True, "profile_id_col": None, "matched_rows": 0}

    a = set(latent_df["obs_name"].astype(str).map(norm_id))
    b = set(prof[id_col].astype(str).map(norm_id))
    return {
        "profile_exists": True,
        "profile_id_col": id_col,
        "profile_rows": int(len(prof)),
        "latent_rows": int(len(latent_df)),
        "matched_rows": int(len(a & b)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    ap.add_argument("--input_npz", default=None)
    ap.add_argument("--metadata_json", default=None)
    ap.add_argument("--profile53c", default=str(DEFAULT_PROFILE53C))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("54a true export Adapter hidden latent z_i from checkpoint v2")
    log("=" * 100)
    log(f"checkpoint={ckpt_path}")

    ckpt = torch_load_trusted(ckpt_path)
    state = ckpt["model_state_dict"]
    ckpt_args = ckpt.get("args", {}) or {}
    ckpt_meta = ckpt.get("meta", {}) or {}

    dims = infer_dims_from_state(state)
    input_dim = dims["input_dim"]
    hidden_dim = dims["hidden_dim"]
    n_layers = dims["n_layers"]
    ffn_dim = dims["ffn_dim"]
    dropout = float(ckpt_args.get("dropout", 0.0))

    input_npz_path = safe_resolve_path(args.input_npz or ckpt_args.get("input_npz"))
    metadata_json_path = safe_resolve_path(args.metadata_json or ckpt_args.get("metadata_json"))

    if input_npz_path is None or not Path(input_npz_path).exists():
        raise FileNotFoundError(f"input_npz not found: {input_npz_path}")

    log(f"inferred input_dim={input_dim}, hidden_dim={hidden_dim}, n_layers={n_layers}, ffn_dim={ffn_dim}")
    log(f"input_npz={input_npz_path}")
    log(f"metadata_json={metadata_json_path}")

    npz = np.load(input_npz_path, allow_pickle=True)
    input_key, X, component_info = build_input_matrix_from_npz(npz, input_dim, ckpt_args, ckpt_meta)

    n_obs = X.shape[0]
    obs_names, obs_source = find_obs_names(npz, n_obs, metadata_json_path, ckpt_path)

    log(f"reconstructed_input={input_key}")
    log(f"component_info={component_info}")
    log(f"X shape={X.shape}")
    log(f"obs_source={obs_source}")

    model = AdapterEncoderOnly(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)

    model.eval()
    device = torch.device(args.device)
    model.to(device)

    Zs = []
    with torch.no_grad():
        for i in range(0, n_obs, args.batch_size):
            xb = torch.tensor(X[i:i+args.batch_size], dtype=torch.float32, device=device)
            zb = model(xb).cpu().numpy()
            Zs.append(zb)
    Z = np.vstack(Zs)

    full = pd.DataFrame({"obs_name": obs_names})
    compat = pd.DataFrame({"obs_name": obs_names})

    for j in range(Z.shape[1]):
        full[f"adapter_true_hidden_z_{j}"] = Z[:, j]
        compat[f"adapter_latent_{j}"] = Z[:, j]

    profile_path = Path(args.profile53c) if args.profile53c else None
    match = profile_match_stats(compat, profile_path)

    out_full = outdir / "adapter_true_hidden_latent_full.csv"
    out_compat = outdir / "adapter_true_hidden_latent_for_neighbor_head.csv"
    out_meta = outdir / "adapter_true_hidden_latent_for_neighbor_head_metadata.json"
    out_report = outdir / "adapter_true_hidden_latent_for_neighbor_head_report.txt"

    full.to_csv(out_full, index=False)
    compat.to_csv(out_compat, index=False)

    meta = {
        "status": "ok",
        "checkpoint": str(ckpt_path),
        "input_npz": str(input_npz_path),
        "metadata_json": str(metadata_json_path) if metadata_json_path else None,
        "inferred_dims": dims,
        "dropout": dropout,
        "reconstructed_input_key": input_key,
        "component_info": [(k, list(shape)) for k, shape in component_info],
        "X_shape": list(X.shape),
        "obs_source": obs_source,
        "latent_shape": list(Z.shape),
        "profile_match": match,
        "load_state_missing_keys": list(missing),
        "load_state_unexpected_keys": list(unexpected),
        "checkpoint_args": ckpt_args,
        "checkpoint_meta": ckpt_meta,
        "outputs": {
            "full": str(out_full),
            "compat": str(out_compat),
            "metadata": str(out_meta),
            "report": str(out_report),
        },
        "note": "Strict hidden z_i exported from reconstructed Step7 Adapter encoder. Input was reconstructed by concatenating NPZ components.",
    }

    out_meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = []
    lines.append("54a true Adapter hidden latent export v2 report")
    lines.append("=" * 100)
    lines.append("SUCCESS")
    lines.append("")
    lines.append(f"checkpoint: {ckpt_path}")
    lines.append(f"input_npz: {input_npz_path}")
    lines.append(f"metadata_json: {metadata_json_path}")
    lines.append("")
    lines.append(f"input_dim: {input_dim}")
    lines.append(f"hidden_dim: {hidden_dim}")
    lines.append(f"n_layers: {n_layers}")
    lines.append(f"ffn_dim: {ffn_dim}")
    lines.append("")
    lines.append(f"reconstructed_input_key: {input_key}")
    lines.append(f"component_info: {component_info}")
    lines.append(f"X shape: {X.shape}")
    lines.append(f"obs_source: {obs_source}")
    lines.append(f"latent shape: {Z.shape}")
    lines.append("")
    lines.append("Profile53c matching:")
    lines.append(json.dumps(match, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("load_state_dict:")
    lines.append(f"missing keys: {list(missing)}")
    lines.append(f"unexpected keys count: {len(list(unexpected))}")
    lines.append(f"unexpected keys head: {list(unexpected)[:60]}")
    lines.append("")
    lines.append(f"Saved full latent: {out_full}")
    lines.append(f"Saved NeighborHead-compatible latent: {out_compat}")
    lines.append("")
    lines.append("Preview:")
    lines.append(compat.head(10).to_string(index=False))
    lines.append("")
    lines.append("Interpretation:")
    lines.append("This file can now be used as strict z_i input for NeighborHead(z_i) -> neighbor_i training.")

    out_report.write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE 54a true hidden z_i export v2")
    log("=" * 100)
    log(f"Saved: {out_full}")
    log(f"Saved: {out_compat}")
    log(f"Saved: {out_meta}")
    log(f"Saved: {out_report}")
    log(f"profile matched rows: {match.get('matched_rows')} / {match.get('profile_rows')}")
    log(f"latent shape: {Z.shape}")
    print(compat.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
