#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 8.3
Train StrokeNiche Perturbation Adapter.

Goal
----
Keep the original StrokeNiche Adapter latent space fixed and learn a lightweight
perturbation module that predicts how a candidate gene / ligand-receptor / drug-like
perturbation shifts lesion-core-like virtual cells toward a repair-permissive state.

Inputs
------
results/step8_strokeniche_perturbmap/
  input_latent_z.npy
  input_region_labels.npy
  input_repair_score.npy
  input_region_proba.npy
  input_core_like_indices.npy
  input_repair_like_indices.npy
  perturbation_library.csv
  positive_perturbations.csv
  negative_decoy_perturbations.csv
  strokeniche_state_prototypes.npz
  virtual_cell_state_table.csv

Outputs
-------
results/step8_strokeniche_perturbmap/perturb_adapter/
  perturb_adapter_model.pt
  perturb_adapter_training_history.csv
  all_perturbation_rescue_ranking.csv
  gene_perturbation_rescue_ranking.csv
  lr_axis_rescue_ranking.csv
  perturbed_state_predictions.csv
  virtual_perturbation_vectors.npy
  virtual_perturbation_vectors.csv
  perturb_adapter_metadata.json

Concept
-------
For each core-like virtual cell i and perturbation p:

    z_i^p = z_i + delta_z_i,p

A perturbation is considered repair-favorable if it:
  1. increases model-derived repair score
  2. decreases lesion-core probability
  3. shifts latent state closer to repair prototype
  4. ranks positive candidates above negative decoys
  5. keeps delta_z bounded

This is weakly supervised virtual perturbation learning, not experimental drug validation.
"""

from pathlib import Path
import argparse
import json
import math
import random
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Paths
# =============================================================================
PROJECT = Path("/mnt/h/vir/ST")
BASE = PROJECT / "results/step8_strokeniche_perturbmap"
OUTDIR = BASE / "perturb_adapter"
OUTDIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Utilities
# =============================================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def read_required(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def safe_json(obj):
    def convert(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, (pd.Series,)):
            return x.tolist()
        return x
    return json.loads(json.dumps(obj, default=convert))


def l2_dist(a, b):
    return torch.norm(a - b, dim=-1)


def batch_cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def masked_mean(x, mask):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=x.device)
    return x[mask].mean()


# =============================================================================
# Data loading
# =============================================================================
def load_inputs():
    z_path = read_required(BASE / "input_latent_z.npy")
    region_path = read_required(BASE / "input_region_labels.npy")
    repair_path = read_required(BASE / "input_repair_score.npy")
    proba_path = read_required(BASE / "input_region_proba.npy")
    core_idx_path = read_required(BASE / "input_core_like_indices.npy")
    repair_idx_path = read_required(BASE / "input_repair_like_indices.npy")
    lib_path = read_required(BASE / "perturbation_library.csv")
    state_path = read_required(BASE / "virtual_cell_state_table.csv")
    proto_path = read_required(BASE / "strokeniche_state_prototypes.npz")

    Z = np.load(z_path).astype(np.float32)
    y_region = np.load(region_path).astype(np.int64)
    repair = np.load(repair_path).astype(np.float32)
    region_proba = np.load(proba_path).astype(np.float32)
    core_idx = np.load(core_idx_path).astype(np.int64)
    repair_idx = np.load(repair_idx_path).astype(np.int64)

    lib = pd.read_csv(lib_path)
    state = pd.read_csv(state_path)
    proto = np.load(proto_path)

    required_proto = ["lesion_core", "peri_infarct", "remote_like", "repair_proto"]
    for k in required_proto:
        if k not in proto.files:
            raise KeyError(f"Missing prototype: {k} in {proto_path}")

    prototypes = {k: proto[k].astype(np.float32) for k in required_proto}

    if Z.shape[0] != len(y_region) or Z.shape[0] != len(repair) or Z.shape[0] != region_proba.shape[0]:
        raise ValueError("Input arrays have inconsistent numbers of cells.")

    if len(state) != Z.shape[0]:
        warnings.warn(f"state table rows {len(state)} != latent rows {Z.shape[0]}")

    # Positive / negative perturbations
    if "is_positive_candidate" not in lib.columns or "is_negative_decoy" not in lib.columns:
        raise KeyError("perturbation_library.csv must contain is_positive_candidate and is_negative_decoy.")

    pos_ids = lib.index[lib["is_positive_candidate"].astype(bool)].to_numpy()
    neg_ids = lib.index[lib["is_negative_decoy"].astype(bool)].to_numpy()

    if len(pos_ids) == 0:
        raise RuntimeError("No positive perturbations found.")
    if len(neg_ids) == 0:
        raise RuntimeError("No negative decoy perturbations found.")

    print("Loaded inputs")
    print("  Z:", Z.shape)
    print("  region labels:", y_region.shape)
    print("  repair:", repair.shape)
    print("  region_proba:", region_proba.shape)
    print("  core-like cells:", len(core_idx))
    print("  repair-like cells:", len(repair_idx))
    print("  perturbation library:", lib.shape)
    print("  positive perturbations:", len(pos_ids))
    print("  negative decoys:", len(neg_ids))

    return {
        "Z": Z,
        "y_region": y_region,
        "repair": repair,
        "region_proba": region_proba,
        "core_idx": core_idx,
        "repair_idx": repair_idx,
        "lib": lib,
        "state": state,
        "prototypes": prototypes,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
    }


def build_standardized_latent(Z):
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    Zs = (Z - mu) / sd
    return Zs.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


def build_perturbation_features(lib: pd.DataFrame):
    """
    Build numeric features for each perturbation.

    Features:
      perturbation_type one-hot
      direction one-hot
      candidate_score_norm
      is_positive_candidate
      is_negative_decoy
      source evidence flags
    """
    df = lib.copy()

    for c in ["perturbation_type", "direction", "source_evidence"]:
        if c not in df.columns:
            df[c] = ""

    if "candidate_score_norm" not in df.columns:
        x = pd.to_numeric(df.get("candidate_score", 0), errors="coerce").fillna(0)
        if x.max() > x.min():
            df["candidate_score_norm"] = (x - x.min()) / (x.max() - x.min())
        else:
            df["candidate_score_norm"] = 0.0

    type_values = sorted(df["perturbation_type"].astype(str).unique().tolist())
    direction_values = sorted(df["direction"].astype(str).unique().tolist())

    type_to_idx = {v: i for i, v in enumerate(type_values)}
    direction_to_idx = {v: i for i, v in enumerate(direction_values)}

    feats = []

    for _, r in df.iterrows():
        type_oh = np.zeros(len(type_values), dtype=np.float32)
        type_oh[type_to_idx[str(r["perturbation_type"])]] = 1.0

        dir_oh = np.zeros(len(direction_values), dtype=np.float32)
        dir_oh[direction_to_idx[str(r["direction"])]] = 1.0

        source = str(r.get("source_evidence", ""))
        source_flags = np.array([
            float("NicheNet" in source),
            float("CellChat" in source),
            float("manual" in source.lower()),
            float("decoy" in source.lower()),
        ], dtype=np.float32)

        score = float(pd.to_numeric(pd.Series([r.get("candidate_score_norm", 0)]), errors="coerce").fillna(0).iloc[0])
        is_pos = float(bool(r.get("is_positive_candidate", False)))
        is_neg = float(bool(r.get("is_negative_decoy", False)))

        feat = np.concatenate([
            type_oh,
            dir_oh,
            np.array([score, is_pos, is_neg], dtype=np.float32),
            source_flags,
        ])
        feats.append(feat)

    feats = np.vstack(feats).astype(np.float32)

    meta = {
        "type_values": type_values,
        "direction_values": direction_values,
        "feature_dim": int(feats.shape[1]),
    }

    return feats, meta


# =============================================================================
# Dataset
# =============================================================================
class CorePerturbPairDataset(Dataset):
    """
    Each item contains:
      core-like cell index
      positive perturbation index
      negative perturbation index
    """
    def __init__(self, core_indices, pos_ids, neg_ids, n_samples_per_epoch=20000, seed=42):
        self.core_indices = np.asarray(core_indices, dtype=np.int64)
        self.pos_ids = np.asarray(pos_ids, dtype=np.int64)
        self.neg_ids = np.asarray(neg_ids, dtype=np.int64)
        self.n_samples_per_epoch = int(n_samples_per_epoch)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples_per_epoch

    def __getitem__(self, idx):
        cell_idx = int(self.core_indices[self.rng.integers(0, len(self.core_indices))])
        pos_id = int(self.pos_ids[self.rng.integers(0, len(self.pos_ids))])
        neg_id = int(self.neg_ids[self.rng.integers(0, len(self.neg_ids))])
        return cell_idx, pos_id, neg_id


# =============================================================================
# Model
# =============================================================================
class StateHeads(nn.Module):
    """
    Lightweight heads on frozen StrokeNiche latent state.

    These heads are trained to approximate:
      - region probability
      - repair score

    They are later frozen and used to evaluate virtual perturbation effects.
    """
    def __init__(self, z_dim, hidden=128, dropout=0.10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.region_head = nn.Linear(hidden, 3)
        self.repair_head = nn.Sequential(
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        h = self.backbone(z)
        region_logits = self.region_head(h)
        repair = self.repair_head(h).squeeze(-1)
        return region_logits, repair


class PerturbationAdapter(nn.Module):
    """
    Predicts bounded delta_z for a given latent z and perturbation p.

    z_p = z + delta_scale * tanh(delta_raw)
    """
    def __init__(
        self,
        z_dim,
        n_perturb,
        perturb_feature_dim,
        perturb_emb_dim=64,
        hidden=256,
        dropout=0.10,
        delta_scale=0.12,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.n_perturb = n_perturb
        self.delta_scale = delta_scale

        self.perturb_id_emb = nn.Embedding(n_perturb, perturb_emb_dim)
        self.perturb_feat_mlp = nn.Sequential(
            nn.Linear(perturb_feature_dim, perturb_emb_dim),
            nn.LayerNorm(perturb_emb_dim),
            nn.GELU(),
            nn.Linear(perturb_emb_dim, perturb_emb_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim + perturb_emb_dim * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
        )

    def perturb_embedding(self, perturb_id, perturb_feat):
        e_id = self.perturb_id_emb(perturb_id)
        e_feat = self.perturb_feat_mlp(perturb_feat)
        return e_id, e_feat

    def forward(self, z, perturb_id, perturb_feat):
        e_id, e_feat = self.perturb_embedding(perturb_id, perturb_feat)
        x = torch.cat([z, e_id, e_feat], dim=-1)
        delta_raw = self.decoder(x)
        delta = self.delta_scale * torch.tanh(delta_raw)
        z_p = z + delta
        return z_p, delta


# =============================================================================
# Training heads
# =============================================================================
def train_state_heads(
    Z_train,
    y_region,
    repair,
    train_mask,
    val_mask,
    device,
    epochs=250,
    batch_size=512,
    lr=1e-3,
    weight_decay=1e-4,
    seed=42,
):
    z_dim = Z_train.shape[1]
    model = StateHeads(z_dim=z_dim).to(device)

    X = torch.tensor(Z_train, dtype=torch.float32)
    y = torch.tensor(y_region, dtype=torch.long)
    r = torch.tensor(repair, dtype=torch.float32)

    train_idx = np.where(train_mask & (y_region >= 0))[0]
    val_idx = np.where(val_mask & (y_region >= 0))[0]

    if len(train_idx) == 0:
        train_idx = np.where(y_region >= 0)[0]
    if len(val_idx) == 0:
        val_idx = train_idx

    # Class weights
    class_counts = np.bincount(y_region[train_idx], minlength=3).astype(float)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    rng = np.random.default_rng(seed)
    history = []
    best_val = -np.inf
    best_state = None
    patience = 30
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(train_idx)
        losses = []

        for start in range(0, len(perm), batch_size):
            idx = perm[start:start + batch_size]
            xb = X[idx].to(device)
            yb = y[idx].to(device)
            rb = r[idx].to(device)

            logits, rhat = model(xb)

            loss_region = F.cross_entropy(logits, yb, weight=class_weights_t)
            loss_repair = F.mse_loss(rhat, rb)
            loss = loss_region + 2.0 * loss_repair

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad():
            xv = X[val_idx].to(device)
            yv = y[val_idx].to(device)
            rv = r[val_idx].to(device)

            logits_v, rhat_v = model(xv)
            pred_v = logits_v.argmax(dim=1)

            acc = (pred_v == yv).float().mean().item()
            repair_mse = F.mse_loss(rhat_v, rv).item()
            val_score = acc - repair_mse

        history.append({
            "phase": "pretrain_heads",
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_region_acc": acc,
            "val_repair_mse": repair_mse,
            "val_score": val_score,
        })

        if val_score > best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 25 == 0 or epoch == 1:
            print(f"[heads] epoch {epoch:03d} loss={np.mean(losses):.4f} val_acc={acc:.4f} repair_mse={repair_mse:.4f}")

        if bad >= patience:
            print(f"[heads] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist = pd.DataFrame(history)
    hist.to_csv(OUTDIR / "state_heads_pretraining_history.csv", index=False)

    # Freeze heads for perturbation training
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    return model, hist


# =============================================================================
# Perturbation training
# =============================================================================
def compute_rescue_components(z_base, z_p, heads, repair_proto, core_proto):
    """
    Returns components:
      repair_delta
      core_delta
      dist_repair_delta
      dist_core_delta
      rescue_score
    """
    with torch.no_grad():
        logits0, repair0 = heads(z_base)
        prob0 = F.softmax(logits0, dim=1)
        pcore0 = prob0[:, 0]

    logits_p, repair_p = heads(z_p)
    prob_p = F.softmax(logits_p, dim=1)
    pcore_p = prob_p[:, 0]

    repair_delta = repair_p - repair0
    core_delta = pcore_p - pcore0

    d_repair0 = l2_dist(z_base, repair_proto)
    d_repair_p = l2_dist(z_p, repair_proto)
    d_core0 = l2_dist(z_base, core_proto)
    d_core_p = l2_dist(z_p, core_proto)

    # positive if closer to repair and farther from core
    repair_dist_improvement = d_repair0 - d_repair_p
    core_dist_departure = d_core_p - d_core0

    rescue_score = (
        repair_delta
        - 1.0 * core_delta
        + 0.25 * repair_dist_improvement
        + 0.10 * core_dist_departure
    )

    return {
        "repair_delta": repair_delta,
        "core_delta": core_delta,
        "repair_dist_improvement": repair_dist_improvement,
        "core_dist_departure": core_dist_departure,
        "rescue_score": rescue_score,
        "repair_p": repair_p,
        "pcore_p": pcore_p,
    }


def train_perturb_adapter(
    Zs,
    inputs,
    perturb_feats,
    heads,
    device,
    epochs=300,
    batch_size=256,
    samples_per_epoch=20000,
    lr=5e-4,
    weight_decay=1e-4,
    margin=0.05,
    delta_reg_weight=0.02,
    decoy_null_weight=0.50,
    align_weight=0.20,
    seed=42,
):
    Z_t = torch.tensor(Zs, dtype=torch.float32, device=device)
    perturb_feats_t = torch.tensor(perturb_feats, dtype=torch.float32, device=device)

    # Candidate-score calibration: prevents all positive perturbations from learning
    # the same universal rescue shift. Higher candidate_score_norm is allowed to
    # induce stronger repair/core-reversal effects, while decoys remain near zero.
    perturb_score_norm = pd.to_numeric(
        inputs["lib"].get("candidate_score_norm", 0.0),
        errors="coerce"
    ).fillna(0.0).values.astype(np.float32)
    perturb_score_t = torch.tensor(perturb_score_norm, dtype=torch.float32, device=device)

    z_dim = Zs.shape[1]
    n_perturb = perturb_feats.shape[0]

    model = PerturbationAdapter(
        z_dim=z_dim,
        n_perturb=n_perturb,
        perturb_feature_dim=perturb_feats.shape[1],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    core_idx = inputs["core_idx"]
    pos_ids = inputs["pos_ids"]
    neg_ids = inputs["neg_ids"]

    # Prototypes should be standardized in the same space
    repair_proto = torch.tensor(inputs["prototypes_standardized"]["repair_proto"], dtype=torch.float32, device=device).view(1, -1)
    core_proto = torch.tensor(inputs["prototypes_standardized"]["lesion_core"], dtype=torch.float32, device=device).view(1, -1)

    dataset = CorePerturbPairDataset(
        core_indices=core_idx,
        pos_ids=pos_ids,
        neg_ids=neg_ids,
        n_samples_per_epoch=samples_per_epoch,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    history = []
    best_score = -np.inf
    best_state = None
    patience = 40
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        epoch_rank = []
        epoch_rescue_pos = []
        epoch_rescue_neg = []
        epoch_delta_norm = []

        for cell_idx, pos_id, neg_id in loader:
            cell_idx = cell_idx.to(device)
            pos_id = pos_id.to(device)
            neg_id = neg_id.to(device)

            z = Z_t[cell_idx]

            pos_feat = perturb_feats_t[pos_id]
            neg_feat = perturb_feats_t[neg_id]

            pos_score = perturb_score_t[pos_id].clamp(0.0, 1.0)
            neg_score = perturb_score_t[neg_id].clamp(0.0, 1.0)

            z_pos, delta_pos = model(z, pos_id, pos_feat)
            z_neg, delta_neg = model(z, neg_id, neg_feat)

            comp_pos = compute_rescue_components(z, z_pos, heads, repair_proto, core_proto)
            comp_neg = compute_rescue_components(z, z_neg, heads, repair_proto, core_proto)

            rescue_pos = comp_pos["rescue_score"]
            rescue_neg = comp_neg["rescue_score"]

            # Ranking loss: positive perturbations should rescue better than decoys.
            loss_rank = F.relu(margin - rescue_pos + rescue_neg).mean()

            # Calibrated positive objective.
            # Instead of maximizing rescue without bound, we ask each perturbation
            # to produce a bounded, candidate-score-calibrated state shift.
            # This prevents all positive perturbations from collapsing to the same
            # generic repair vector.
            target_rescue_pos = 0.06 + 0.18 * pos_score
            target_repair_delta_pos = 0.02 + 0.08 * pos_score
            target_core_delta_pos = -(0.05 + 0.16 * pos_score)

            loss_repair_direction = F.relu(
                target_repair_delta_pos - comp_pos["repair_delta"]
            ).mean()

            loss_core_reversal = F.relu(
                comp_pos["core_delta"] - target_core_delta_pos
            ).mean()

            loss_rescue_calibration = F.mse_loss(
                rescue_pos,
                target_rescue_pos
            )

            # Mild latent alignment toward repair prototype, not an unbounded pull.
            loss_latent_alignment = F.relu(
                -comp_pos["repair_dist_improvement"] + 0.01
            ).mean()

            loss_pos = (
                loss_repair_direction
                + loss_core_reversal
                + 0.75 * loss_rescue_calibration
                + 0.20 * loss_latent_alignment
            )

            # Decoys should produce minimal shift / minimal apparent rescue.
            loss_decoy_null = (
                F.mse_loss(rescue_neg, torch.zeros_like(rescue_neg))
                + 1.0 * delta_neg.pow(2).mean()
            )

            # Alignment: positive z_p should be closer to repair proto than original.
            d_repair_pos = l2_dist(z_pos, repair_proto)
            d_repair_base = l2_dist(z, repair_proto)
            loss_align = F.relu(d_repair_pos - d_repair_base + 0.02).mean()

            # Regularize perturbation magnitude
            loss_delta = delta_pos.pow(2).mean() + delta_neg.pow(2).mean()

            loss = (
                loss_rank
                + loss_pos
                + decoy_null_weight * loss_decoy_null
                + align_weight * loss_align
                + delta_reg_weight * loss_delta
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_losses.append(float(loss.detach().cpu()))
            epoch_rank.append(float(loss_rank.detach().cpu()))
            epoch_rescue_pos.append(float(rescue_pos.mean().detach().cpu()))
            epoch_rescue_neg.append(float(rescue_neg.mean().detach().cpu()))
            epoch_delta_norm.append(float(torch.norm(delta_pos, dim=1).mean().detach().cpu()))

        mean_loss = float(np.mean(epoch_losses))
        mean_rank = float(np.mean(epoch_rank))
        mean_pos = float(np.mean(epoch_rescue_pos))
        mean_neg = float(np.mean(epoch_rescue_neg))
        mean_margin = mean_pos - mean_neg
        mean_delta = float(np.mean(epoch_delta_norm))

        # validation-like score from training samples; weakly-supervised
        val_score = mean_margin - 0.05 * mean_delta - 0.1 * mean_rank

        history.append({
            "phase": "perturb_adapter",
            "epoch": epoch,
            "loss": mean_loss,
            "rank_loss": mean_rank,
            "mean_rescue_pos": mean_pos,
            "mean_rescue_neg": mean_neg,
            "mean_rescue_margin": mean_margin,
            "mean_delta_norm": mean_delta,
            "val_score_proxy": val_score,
        })

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 25 == 0 or epoch == 1:
            print(
                f"[perturb] epoch {epoch:03d} "
                f"loss={mean_loss:.4f} rank={mean_rank:.4f} "
                f"pos={mean_pos:.4f} neg={mean_neg:.4f} "
                f"margin={mean_margin:.4f} delta={mean_delta:.4f}"
            )

        if bad >= patience:
            print(f"[perturb] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist = pd.DataFrame(history)
    hist.to_csv(OUTDIR / "perturb_adapter_training_history.csv", index=False)

    return model, hist


# =============================================================================
# Ranking / inference
# =============================================================================
@torch.no_grad()
def score_all_perturbations(
    model,
    heads,
    Zs,
    inputs,
    perturb_feats,
    device,
    max_cells_per_perturb=1200,
):
    model.eval()
    heads.eval()

    Z_t = torch.tensor(Zs, dtype=torch.float32, device=device)
    perturb_feats_t = torch.tensor(perturb_feats, dtype=torch.float32, device=device)

    core_idx = inputs["core_idx"]
    lib = inputs["lib"].copy()

    if len(core_idx) > max_cells_per_perturb:
        rng = np.random.default_rng(42)
        eval_idx = rng.choice(core_idx, size=max_cells_per_perturb, replace=False)
    else:
        eval_idx = core_idx

    z = Z_t[eval_idx]

    repair_proto = torch.tensor(inputs["prototypes_standardized"]["repair_proto"], dtype=torch.float32, device=device).view(1, -1)
    core_proto = torch.tensor(inputs["prototypes_standardized"]["lesion_core"], dtype=torch.float32, device=device).view(1, -1)

    # Base state
    logits0, repair0 = heads(z)
    prob0 = F.softmax(logits0, dim=1)
    pcore0 = prob0[:, 0]
    pperi0 = prob0[:, 1]
    premote0 = prob0[:, 2]
    d_repair0 = l2_dist(z, repair_proto)
    d_core0 = l2_dist(z, core_proto)

    rows = []
    pred_rows = []
    mean_deltas = []

    for pid in range(len(lib)):
        pids = torch.full((len(eval_idx),), pid, dtype=torch.long, device=device)
        pfeat = perturb_feats_t[pids]

        z_p, delta = model(z, pids, pfeat)

        comp = compute_rescue_components(z, z_p, heads, repair_proto, core_proto)

        logits_p, repair_p = heads(z_p)
        prob_p = F.softmax(logits_p, dim=1)

        pcore_p = prob_p[:, 0]
        pperi_p = prob_p[:, 1]
        premote_p = prob_p[:, 2]

        d_repair_p = l2_dist(z_p, repair_proto)
        d_core_p = l2_dist(z_p, core_proto)

        delta_repair = repair_p - repair0
        delta_core = pcore_p - pcore0
        delta_peri = pperi_p - pperi0
        delta_remote = premote_p - premote0
        delta_d_repair = d_repair0 - d_repair_p
        delta_d_core = d_core_p - d_core0

        rescue = (
            delta_repair
            - delta_core
            + 0.25 * delta_d_repair
            + 0.10 * delta_d_core
        )

        delta_norm = torch.norm(delta, dim=1)

        mean_delta_vec = delta.mean(dim=0).detach().cpu().numpy()
        mean_deltas.append(mean_delta_vec)

        r = lib.iloc[pid].to_dict()

        strict_rescued = (
            (rescue > 0.08)
            & (delta_repair > 0.02)
            & (delta_core < -0.05)
        )

        moderate_rescued = (
            (rescue > 0.04)
            & (delta_repair > 0.01)
            & (delta_core < -0.025)
        )

        rows.append({
            "perturbation_id": r.get("perturbation_id", f"perturb_{pid}"),
            "perturbation_type": r.get("perturbation_type", ""),
            "ligand": r.get("ligand", ""),
            "receptor": r.get("receptor", ""),
            "target_gene": r.get("target_gene", ""),
            "direction": r.get("direction", ""),
            "candidate_score": r.get("candidate_score", np.nan),
            "candidate_score_norm": r.get("candidate_score_norm", np.nan),
            "source_evidence": r.get("source_evidence", ""),
            "mechanism_hint": r.get("mechanism_hint", ""),
            "is_positive_candidate": bool(r.get("is_positive_candidate", False)),
            "is_negative_decoy": bool(r.get("is_negative_decoy", False)),

            "n_eval_core_like_cells": int(len(eval_idx)),
            "mean_delta_repair_score": float(delta_repair.mean().detach().cpu()),
            "mean_delta_core_probability": float(delta_core.mean().detach().cpu()),
            "mean_delta_peri_probability": float(delta_peri.mean().detach().cpu()),
            "mean_delta_remote_probability": float(delta_remote.mean().detach().cpu()),
            "mean_repair_distance_improvement": float(delta_d_repair.mean().detach().cpu()),
            "mean_core_distance_departure": float(delta_d_core.mean().detach().cpu()),
            "mean_delta_norm": float(delta_norm.mean().detach().cpu()),
            "mean_rescue_score": float(rescue.mean().detach().cpu()),
            "median_rescue_score": float(rescue.median().detach().cpu()),

            # legacy permissive metric
            "fraction_cells_rescued": float((rescue > 0).float().mean().detach().cpu()),

            # stricter metrics for main figures
            "fraction_cells_rescued_moderate": float(moderate_rescued.float().mean().detach().cpu()),
            "fraction_cells_rescued_strict": float(strict_rescued.float().mean().detach().cpu()),

            "fraction_core_probability_decreased": float((delta_core < 0).float().mean().detach().cpu()),
            "fraction_repair_score_increased": float((delta_repair > 0).float().mean().detach().cpu()),
        })

        # Save per-cell predictions for compact top-level inspection.
        # To avoid huge files, store all perturbations x evaluated core-like cells.
        for j, cell_idx in enumerate(eval_idx):
            pred_rows.append({
                "perturbation_id": r.get("perturbation_id", f"perturb_{pid}"),
                "cell_index": int(cell_idx),
                "delta_repair_score": float(delta_repair[j].detach().cpu()),
                "delta_core_probability": float(delta_core[j].detach().cpu()),
                "delta_peri_probability": float(delta_peri[j].detach().cpu()),
                "delta_remote_probability": float(delta_remote[j].detach().cpu()),
                "rescue_score": float(rescue[j].detach().cpu()),
                "delta_norm": float(delta_norm[j].detach().cpu()),
            })

    ranking = pd.DataFrame(rows)

    # Composite score for presentation.
    # Penalize extremely large delta to avoid artifact-like shifts.
    # Presentation score:
    # combines model-predicted rescue with a small evidence prior and penalizes
    # excessive latent movement. The candidate prior helps break biologically
    # implausible ties but does not dominate the learned rescue score.
    candidate_prior = pd.to_numeric(
        ranking.get("candidate_score_norm", 0.0),
        errors="coerce"
    ).fillna(0.0)

    ranking["overall_rescue_score"] = (
        ranking["mean_rescue_score"]
        + 0.16 * ranking["fraction_cells_rescued_strict"]
        + 0.08 * ranking["fraction_cells_rescued_moderate"]
        + 0.06 * ranking["fraction_core_probability_decreased"]
        + 0.06 * ranking["fraction_repair_score_increased"]
        + 0.10 * candidate_prior
        - 0.14 * ranking["mean_delta_norm"]
    )

    ranking = ranking.sort_values("overall_rescue_score", ascending=False).reset_index(drop=True)
    ranking["rescue_rank"] = np.arange(1, len(ranking) + 1)

    pred_df = pd.DataFrame(pred_rows)
    mean_delta_arr = np.vstack(mean_deltas).astype(np.float32)

    return ranking, pred_df, mean_delta_arr


def save_rankings(ranking, pred_df, mean_delta_arr):
    all_path = OUTDIR / "all_perturbation_rescue_ranking.csv"
    ranking.to_csv(all_path, index=False)

    gene_mask = ranking["perturbation_type"].astype(str).isin(["gene_down", "gene_up"])
    lr_mask = ranking["perturbation_type"].astype(str).eq("lr_blockade")
    decoy_mask = ranking["perturbation_type"].astype(str).eq("negative_decoy")

    ranking[gene_mask].to_csv(OUTDIR / "gene_perturbation_rescue_ranking.csv", index=False)
    ranking[lr_mask].to_csv(OUTDIR / "lr_axis_rescue_ranking.csv", index=False)
    ranking[decoy_mask].to_csv(OUTDIR / "negative_decoy_rescue_ranking.csv", index=False)

    pred_df.to_csv(OUTDIR / "perturbed_state_predictions.csv", index=False)

    np.save(OUTDIR / "virtual_perturbation_vectors.npy", mean_delta_arr)

    vector_df = ranking[[
        "perturbation_id",
        "perturbation_type",
        "target_gene",
        "overall_rescue_score",
        "mean_rescue_score",
        "mean_delta_repair_score",
        "mean_delta_core_probability",
    ]].copy()

    for k in range(mean_delta_arr.shape[1]):
        vector_df[f"delta_z_{k}"] = mean_delta_arr[:, k]

    vector_df.to_csv(OUTDIR / "virtual_perturbation_vectors.csv", index=False)

    print("Saved:", all_path)
    print("Saved:", OUTDIR / "gene_perturbation_rescue_ranking.csv")
    print("Saved:", OUTDIR / "lr_axis_rescue_ranking.csv")
    print("Saved:", OUTDIR / "perturbed_state_predictions.csv")
    print("Saved:", OUTDIR / "virtual_perturbation_vectors.npy")
    print("Saved:", OUTDIR / "virtual_perturbation_vectors.csv")

    print("\nTop perturbations by overall_rescue_score:")
    cols = [
        "rescue_rank",
        "perturbation_id",
        "perturbation_type",
        "target_gene",
        "overall_rescue_score",
        "mean_delta_repair_score",
        "mean_delta_core_probability",
        "fraction_cells_rescued",
    ]
    print(ranking[cols].head(20).to_string(index=False))


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--head-epochs", type=int, default=250)
    parser.add_argument("--perturb-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--samples-per-epoch", type=int, default=20000)
    parser.add_argument("--max-cells-per-perturb", type=int, default=1200)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-perturb", type=float, default=5e-4)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--delta-reg-weight", type=float, default=0.10)
    parser.add_argument("--decoy-null-weight", type=float, default=2.00)
    parser.add_argument("--align-weight", type=float, default=0.10)
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 100)
    print("Step 8.3 | Train StrokeNiche Perturbation Adapter")
    print("=" * 100)
    print("Device:", device)
    print("Output:", OUTDIR)

    inputs = load_inputs()

    Z = inputs["Z"]
    Zs, z_mu, z_sd = build_standardized_latent(Z)

    # Standardize prototypes with same mu/sd
    proto_std = {}
    for k, v in inputs["prototypes"].items():
        proto_std[k] = ((v.reshape(1, -1) - z_mu) / z_sd).reshape(-1).astype(np.float32)
    inputs["prototypes_standardized"] = proto_std

    # Split masks from state table when available
    state = inputs["state"]
    if {"train", "val", "test"}.issubset(state.columns):
        train_mask = state["train"].astype(bool).values
        val_mask = state["val"].astype(bool).values
    else:
        rng = np.random.default_rng(args.seed)
        idx = np.arange(Z.shape[0])
        rng.shuffle(idx)
        train_mask = np.zeros(Z.shape[0], dtype=bool)
        val_mask = np.zeros(Z.shape[0], dtype=bool)
        train_mask[idx[: int(0.8 * len(idx))]] = True
        val_mask[idx[int(0.8 * len(idx)) :]] = True

    perturb_feats, perturb_meta = build_perturbation_features(inputs["lib"])

    # Save feature metadata
    (OUTDIR / "perturbation_feature_metadata.json").write_text(
        json.dumps(safe_json(perturb_meta), indent=2)
    )

    # Save latent normalization
    np.savez(OUTDIR / "latent_standardization_stats.npz", mean=z_mu, std=z_sd)

    # 1. Train heads on original StrokeNiche latent state
    heads, head_hist = train_state_heads(
        Z_train=Zs,
        y_region=inputs["y_region"],
        repair=inputs["repair"],
        train_mask=train_mask,
        val_mask=val_mask,
        device=device,
        epochs=args.head_epochs,
        batch_size=512,
        lr=args.lr_head,
        seed=args.seed,
    )

    # 2. Train perturbation adapter
    adapter, perturb_hist = train_perturb_adapter(
        Zs=Zs,
        inputs=inputs,
        perturb_feats=perturb_feats,
        heads=heads,
        device=device,
        epochs=args.perturb_epochs,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        lr=args.lr_perturb,
        margin=args.margin,
        delta_reg_weight=args.delta_reg_weight,
        decoy_null_weight=args.decoy_null_weight,
        align_weight=args.align_weight,
        seed=args.seed,
    )

    # 3. Save model
    model_path = OUTDIR / "perturb_adapter_model.pt"
    torch.save({
        "state_heads": heads.state_dict(),
        "perturb_adapter": adapter.state_dict(),
        "z_dim": int(Zs.shape[1]),
        "n_perturb": int(len(inputs["lib"])),
        "perturb_feature_dim": int(perturb_feats.shape[1]),
        "perturb_feature_metadata": perturb_meta,
        "latent_mean": z_mu,
        "latent_std": z_sd,
        "args": vars(args),
    }, model_path)
    print("Saved:", model_path)

    # 4. Score perturbations
    ranking, pred_df, mean_delta_arr = score_all_perturbations(
        model=adapter,
        heads=heads,
        Zs=Zs,
        inputs=inputs,
        perturb_feats=perturb_feats,
        device=device,
        max_cells_per_perturb=args.max_cells_per_perturb,
    )

    save_rankings(ranking, pred_df, mean_delta_arr)

    # 5. Metadata
    metadata = {
        "step": "Step 8.3 StrokeNiche Perturbation Adapter",
        "input_dir": str(BASE),
        "output_dir": str(OUTDIR),
        "n_virtual_cells": int(Z.shape[0]),
        "latent_dim": int(Z.shape[1]),
        "n_perturbations": int(len(inputs["lib"])),
        "n_positive_perturbations": int(inputs["lib"]["is_positive_candidate"].sum()),
        "n_negative_decoys": int(inputs["lib"]["is_negative_decoy"].sum()),
        "n_core_like_cells": int(len(inputs["core_idx"])),
        "device": str(device),
        "args": vars(args),
        "important_note": (
            "This is weakly supervised virtual perturbation learning. "
            "Scores are computational prioritization metrics, not experimental drug efficacy evidence."
        ),
    }

    (OUTDIR / "perturb_adapter_metadata.json").write_text(
        json.dumps(safe_json(metadata), indent=2, ensure_ascii=False)
    )

    print("Saved:", OUTDIR / "perturb_adapter_metadata.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
