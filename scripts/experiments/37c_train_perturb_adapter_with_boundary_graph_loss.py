#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
37c Train StrokeNiche Perturbation Adapter with boundary-aware graph loss.

This is the real training version after the post hoc 37b ablation.

It reuses functions/classes from:
  37_train_strokeniche_perturb_adapter.py

and adds graph regularization during PerturbationAdapter training:

  L_graph = sum_ij w_ij * Huber(||delta_z_i,p - delta_z_j,p||)

Default graph mode:
  boundary_aware_graph

Variants:
  no_graph
  ordinary_graph
  boundary_aware_graph

Outputs:
  results/step8_strokeniche_perturbmap/perturb_adapter_graph_regularized/<variant>/

Important:
  This is weakly supervised virtual perturbation learning.
  It is for computational prioritization, not experimental drug efficacy evidence.
"""

from pathlib import Path
import argparse
import importlib.util
import json
import random
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


# =============================================================================
# Paths
# =============================================================================
PROJECT = Path("/mnt/h/vir/ST")
BASE = PROJECT / "results/step8_strokeniche_perturbmap"
BASE_ADAPTER = BASE / "perturb_adapter"
OUTROOT = BASE / "perturb_adapter_graph_regularized"
OUTROOT.mkdir(parents=True, exist_ok=True)

SCRIPT37 = PROJECT / "37_train_strokeniche_perturb_adapter.py"

STATE_CSV = BASE / "virtual_cell_state_table.csv"
PROBA_NPY = BASE / "input_region_proba.npy"
REPAIR_NPY = BASE / "input_repair_score.npy"

SPATIAL_H5AD_CANDIDATES = [
    PROJECT / "results/step5_nicheformer/spatial_all_with_nicheformer.h5ad",
    PROJECT / "results/step5_nicheformer/nicheformer_spatial_with_embeddings.h5ad",
]

BASELINE_RANK = BASE_ADAPTER / "all_perturbation_rescue_ranking.csv"


# =============================================================================
# Graph parameters
# =============================================================================
K_NEIGHBORS = 8
STATE_TAU = 0.65
REPAIR_TAU = 0.30


# =============================================================================
# Utilities
# =============================================================================
def load_module37():
    if not SCRIPT37.exists():
        raise FileNotFoundError(SCRIPT37)

    spec = importlib.util.spec_from_file_location("m37", SCRIPT37)
    m37 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m37)
    return m37


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return None


def safe_json(obj):
    def conv(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        return x
    return json.loads(json.dumps(obj, default=conv))


def get_state_col(state, pairs):
    for a, b in pairs:
        if a in state.columns and b in state.columns:
            return a, b
    return None


def find_coordinates_from_state(state):
    pairs = [
        ("x", "y"),
        ("X", "Y"),
        ("spatial_x", "spatial_y"),
        ("array_col", "array_row"),
        ("pxl_col_in_fullres", "pxl_row_in_fullres"),
        ("imagecol", "imagerow"),
        ("imagerow", "imagecol"),
        ("col", "row"),
        ("row", "col"),
    ]
    p = get_state_col(state, pairs)
    if p is None:
        return None

    coords = state[list(p)].values.astype(float)
    if np.isfinite(coords).all():
        print(f"Using coordinates from state table: {p}")
        return coords

    return None


def find_coordinates_from_h5ad(n_expected):
    p = first_existing(SPATIAL_H5AD_CANDIDATES)
    if p is None:
        return None

    try:
        import anndata as ad
        a = ad.read_h5ad(p)
    except Exception as e:
        warnings.warn(f"Could not read h5ad for coordinates: {e}")
        return None

    if "spatial" in a.obsm:
        coords = np.asarray(a.obsm["spatial"], dtype=float)
        if coords.shape[0] == n_expected:
            print(f"Using coordinates from h5ad obsm['spatial']: {p}")
            return coords

    for pair in [("array_col", "array_row"), ("x", "y"), ("imagecol", "imagerow")]:
        if pair[0] in a.obs.columns and pair[1] in a.obs.columns:
            coords = a.obs[[pair[0], pair[1]]].values.astype(float)
            if coords.shape[0] == n_expected:
                print(f"Using coordinates from h5ad obs columns {pair}: {p}")
                return coords

    return None


def get_coordinates(state, n_expected):
    coords = find_coordinates_from_state(state)
    if coords is not None:
        return coords

    coords = find_coordinates_from_h5ad(n_expected)
    if coords is not None:
        return coords

    raise RuntimeError("No spatial coordinates found.")


def build_boundary_graph(coords, proba, repair, k=8):
    coords = np.asarray(coords, dtype=float)
    coords_s = StandardScaler().fit_transform(coords)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nn.fit(coords_s)
    dist, ind = nn.kneighbors(coords_s)

    rows = []
    for i in range(coords_s.shape[0]):
        for j, d in zip(ind[i, 1:], dist[i, 1:]):
            rows.append((i, int(j), float(d)))

    edges = pd.DataFrame(rows, columns=["i", "j", "dist"])

    # undirected unique
    rev = edges.rename(columns={"i": "j", "j": "i"})
    edges = pd.concat([edges, rev], ignore_index=True)
    edges["a"] = np.minimum(edges["i"], edges["j"])
    edges["b"] = np.maximum(edges["i"], edges["j"])
    edges = edges.drop_duplicates(["a", "b"]).drop(columns=["a", "b"]).reset_index(drop=True)

    sigma = np.median(edges["dist"].values)
    if sigma <= 0:
        sigma = 1.0

    i = edges["i"].values.astype(int)
    j = edges["j"].values.astype(int)

    w_spatial = np.exp(-(edges["dist"].values ** 2) / (sigma ** 2))

    dp = np.linalg.norm(proba[i] - proba[j], axis=1)
    dr = np.abs(repair[i] - repair[j])

    w_state = np.exp(-(dp ** 2) / (STATE_TAU ** 2))
    w_repair = np.exp(-(dr ** 2) / (REPAIR_TAU ** 2))

    edges["w_ordinary"] = w_spatial
    edges["w_boundary_aware"] = w_spatial * w_state * w_repair

    # Normalize each weight so average nonzero edge weight is close to 1.
    for c in ["w_ordinary", "w_boundary_aware"]:
        x = edges[c].values.astype(float)
        m = np.mean(x[x > 0])
        if m <= 0:
            m = 1.0
        edges[c] = x / m
        edges[c] = np.clip(edges[c], 1e-6, 5.0)

    print("Graph built:")
    print("  edges:", edges.shape[0])
    print("  median coordinate distance:", sigma)
    print("  ordinary weight mean:", edges["w_ordinary"].mean())
    print("  boundary-aware weight mean:", edges["w_boundary_aware"].mean())

    return edges


def load_graph_inputs(n_expected):
    state = pd.read_csv(STATE_CSV)
    proba = np.load(PROBA_NPY).astype(np.float32)
    repair = np.load(REPAIR_NPY).astype(np.float32)

    if proba.shape[0] != n_expected:
        raise ValueError(f"proba rows {proba.shape[0]} != n cells {n_expected}")
    if repair.shape[0] != n_expected:
        raise ValueError(f"repair rows {repair.shape[0]} != n cells {n_expected}")

    coords = get_coordinates(state, n_expected)

    edges = build_boundary_graph(coords, proba, repair, k=K_NEIGHBORS)
    return state, proba, repair, coords, edges


def graph_tensors(edges, variant, device):
    if variant == "ordinary_graph":
        wcol = "w_ordinary"
    elif variant == "boundary_aware_graph":
        wcol = "w_boundary_aware"
    elif variant == "no_graph":
        wcol = None
    else:
        raise ValueError(f"Unknown variant: {variant}")

    if wcol is None:
        return None

    ei = torch.tensor(edges["i"].values.astype(np.int64), dtype=torch.long, device=device)
    ej = torch.tensor(edges["j"].values.astype(np.int64), dtype=torch.long, device=device)
    ew = torch.tensor(edges[wcol].values.astype(np.float32), dtype=torch.float32, device=device)

    return {
        "i": ei,
        "j": ej,
        "w": ew,
        "n_edges": len(edges),
        "wcol": wcol,
    }


# =============================================================================
# Graph-regularized perturb adapter training
# =============================================================================
def compute_graph_delta_loss(
    model,
    Z_t,
    perturb_feats_t,
    pos_ids_t,
    graph,
    graph_batch_edges=1024,
    graph_on="delta",
):
    """
    Boundary-aware graph loss on perturbation response field.

    Default:
      graph_on = "delta"
      L_graph = Σ w_ij Huber(||delta_i,p - delta_j,p||)

    This regularizes spatial continuity of the perturbation response,
    not the original latent state itself.
    """
    if graph is None:
        return torch.tensor(0.0, device=Z_t.device)

    n_edges = graph["n_edges"]
    if n_edges <= 0:
        return torch.tensor(0.0, device=Z_t.device)

    sel = torch.randint(0, n_edges, (graph_batch_edges,), device=Z_t.device)

    ei = graph["i"][sel]
    ej = graph["j"][sel]
    ew = graph["w"][sel]

    # sample positive perturbation ids for graph regularization
    psel = torch.randint(0, len(pos_ids_t), (graph_batch_edges,), device=Z_t.device)
    pid = pos_ids_t[psel]

    pfeat = perturb_feats_t[pid]

    zi = Z_t[ei]
    zj = Z_t[ej]

    zpi, di = model(zi, pid, pfeat)
    zpj, dj = model(zj, pid, pfeat)

    if graph_on == "zp":
        d = torch.norm(zpi - zpj, dim=1)
    else:
        d = torch.norm(di - dj, dim=1)

    loss_vec = F.smooth_l1_loss(d, torch.zeros_like(d), reduction="none")
    loss = (ew * loss_vec).sum() / (ew.sum() + 1e-8)

    return loss


def train_perturb_adapter_graph_regularized(
    m37,
    Zs,
    inputs,
    perturb_feats,
    heads,
    device,
    variant="boundary_aware_graph",
    graph=None,
    epochs=300,
    batch_size=256,
    samples_per_epoch=20000,
    lr=5e-4,
    weight_decay=1e-4,
    margin=0.08,
    delta_reg_weight=0.12,
    decoy_null_weight=2.5,
    align_weight=0.08,
    graph_weight=0.05,
    graph_batch_edges=1024,
    graph_on="delta",
    seed=42,
):
    Z_t = torch.tensor(Zs, dtype=torch.float32, device=device)
    perturb_feats_t = torch.tensor(perturb_feats, dtype=torch.float32, device=device)

    perturb_score_norm = pd.to_numeric(
        inputs["lib"].get("candidate_score_norm", 0.0),
        errors="coerce"
    ).fillna(0.0).values.astype(np.float32)

    perturb_score_t = torch.tensor(perturb_score_norm, dtype=torch.float32, device=device)

    z_dim = Zs.shape[1]
    n_perturb = perturb_feats.shape[0]

    model = m37.PerturbationAdapter(
        z_dim=z_dim,
        n_perturb=n_perturb,
        perturb_feature_dim=perturb_feats.shape[1],
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    core_idx = inputs["core_idx"]
    pos_ids = inputs["pos_ids"]
    neg_ids = inputs["neg_ids"]

    pos_ids_t = torch.tensor(pos_ids, dtype=torch.long, device=device)

    repair_proto = torch.tensor(
        inputs["prototypes_standardized"]["repair_proto"],
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    core_proto = torch.tensor(
        inputs["prototypes_standardized"]["lesion_core"],
        dtype=torch.float32,
        device=device,
    ).view(1, -1)

    dataset = m37.CorePerturbPairDataset(
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
    patience = 45
    bad = 0

    for epoch in range(1, epochs + 1):
        model.train()

        losses = []
        rank_losses = []
        graph_losses = []
        rescue_pos_vals = []
        rescue_neg_vals = []
        delta_norm_vals = []

        for cell_idx, pos_id, neg_id in loader:
            cell_idx = cell_idx.to(device)
            pos_id = pos_id.to(device)
            neg_id = neg_id.to(device)

            z = Z_t[cell_idx]

            pos_feat = perturb_feats_t[pos_id]
            neg_feat = perturb_feats_t[neg_id]

            pos_score = perturb_score_t[pos_id].clamp(0.0, 1.0)

            z_pos, delta_pos = model(z, pos_id, pos_feat)
            z_neg, delta_neg = model(z, neg_id, neg_feat)

            comp_pos = m37.compute_rescue_components(z, z_pos, heads, repair_proto, core_proto)
            comp_neg = m37.compute_rescue_components(z, z_neg, heads, repair_proto, core_proto)

            rescue_pos = comp_pos["rescue_score"]
            rescue_neg = comp_neg["rescue_score"]

            # Ranking loss
            loss_rank = F.relu(margin - rescue_pos + rescue_neg).mean()

            # Candidate-score calibrated perturbation targets
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
                target_rescue_pos,
            )

            loss_latent_alignment = F.relu(
                -comp_pos["repair_dist_improvement"] + 0.01
            ).mean()

            loss_pos = (
                loss_repair_direction
                + loss_core_reversal
                + 0.75 * loss_rescue_calibration
                + 0.20 * loss_latent_alignment
            )

            # Decoys should stay close to null
            loss_decoy_null = (
                F.mse_loss(rescue_neg, torch.zeros_like(rescue_neg))
                + 1.0 * delta_neg.pow(2).mean()
            )

            loss_delta = delta_pos.pow(2).mean() + delta_neg.pow(2).mean()

            # New graph loss
            if variant == "no_graph" or graph is None or graph_weight <= 0:
                loss_graph = torch.tensor(0.0, device=device)
            else:
                loss_graph = compute_graph_delta_loss(
                    model=model,
                    Z_t=Z_t,
                    perturb_feats_t=perturb_feats_t,
                    pos_ids_t=pos_ids_t,
                    graph=graph,
                    graph_batch_edges=graph_batch_edges,
                    graph_on=graph_on,
                )

            loss = (
                loss_rank
                + loss_pos
                + decoy_null_weight * loss_decoy_null
                + align_weight * loss_latent_alignment
                + delta_reg_weight * loss_delta
                + graph_weight * loss_graph
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            rank_losses.append(float(loss_rank.detach().cpu()))
            graph_losses.append(float(loss_graph.detach().cpu()))
            rescue_pos_vals.append(float(rescue_pos.mean().detach().cpu()))
            rescue_neg_vals.append(float(rescue_neg.mean().detach().cpu()))
            delta_norm_vals.append(float(torch.norm(delta_pos, dim=1).mean().detach().cpu()))

        mean_loss = float(np.mean(losses))
        mean_rank = float(np.mean(rank_losses))
        mean_graph = float(np.mean(graph_losses))
        mean_pos = float(np.mean(rescue_pos_vals))
        mean_neg = float(np.mean(rescue_neg_vals))
        mean_margin = mean_pos - mean_neg
        mean_delta = float(np.mean(delta_norm_vals))

        # Proxy validation: keep margin, penalize large delta and graph instability
        val_score = mean_margin - 0.05 * mean_delta - 0.08 * mean_rank - 0.02 * mean_graph

        history.append({
            "variant": variant,
            "epoch": epoch,
            "loss": mean_loss,
            "rank_loss": mean_rank,
            "graph_loss": mean_graph,
            "mean_rescue_pos": mean_pos,
            "mean_rescue_neg": mean_neg,
            "mean_rescue_margin": mean_margin,
            "mean_delta_norm": mean_delta,
            "val_score_proxy": val_score,
            "graph_weight": graph_weight,
            "graph_on": graph_on,
        })

        if val_score > best_score:
            best_score = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"[{variant}] epoch {epoch:03d} "
                f"loss={mean_loss:.4f} rank={mean_rank:.4f} graph={mean_graph:.4f} "
                f"pos={mean_pos:.4f} neg={mean_neg:.4f} "
                f"margin={mean_margin:.4f} delta={mean_delta:.4f}"
            )

        if bad >= patience:
            print(f"[{variant}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist = pd.DataFrame(history)

    return model, hist


# =============================================================================
# Saving / comparison
# =============================================================================
def save_graph_regularized_outputs(
    variant_outdir,
    variant,
    ranking,
    pred_df,
    mean_delta_arr,
    lib,
):
    variant_outdir.mkdir(parents=True, exist_ok=True)

    ranking.to_csv(variant_outdir / "all_perturbation_rescue_ranking.csv", index=False)

    ranking[ranking["perturbation_type"].astype(str).isin(["gene_down", "gene_up"])].to_csv(
        variant_outdir / "gene_perturbation_rescue_ranking.csv",
        index=False,
    )

    ranking[ranking["perturbation_type"].astype(str).eq("lr_blockade")].to_csv(
        variant_outdir / "lr_axis_rescue_ranking.csv",
        index=False,
    )

    ranking[ranking["perturbation_type"].astype(str).eq("negative_decoy")].to_csv(
        variant_outdir / "negative_decoy_rescue_ranking.csv",
        index=False,
    )

    pred_df.to_csv(variant_outdir / "perturbed_state_predictions.csv", index=False)
    np.save(variant_outdir / "virtual_perturbation_vectors.npy", mean_delta_arr)

    # aligned vector file in library order, with metrics merged by perturbation_id
    delta_df = lib[[
        "perturbation_id",
        "perturbation_type",
        "target_gene",
        "ligand",
        "receptor",
        "direction",
        "candidate_score",
        "candidate_score_norm",
        "source_evidence",
        "mechanism_hint",
    ]].copy()

    for k in range(mean_delta_arr.shape[1]):
        delta_df[f"delta_z_{k}"] = mean_delta_arr[:, k]

    metric_cols = [
        "perturbation_id",
        "rescue_rank",
        "overall_rescue_score",
        "mean_rescue_score",
        "mean_delta_repair_score",
        "mean_delta_core_probability",
        "mean_delta_peri_probability",
        "mean_delta_remote_probability",
        "fraction_cells_rescued_moderate",
        "fraction_cells_rescued_strict",
        "mean_delta_norm",
    ]
    metric_cols = [c for c in metric_cols if c in ranking.columns]

    aligned = delta_df.merge(
        ranking[metric_cols],
        on="perturbation_id",
        how="left",
        validate="one_to_one",
    ).sort_values("rescue_rank")

    aligned.to_csv(variant_outdir / "virtual_perturbation_vectors_aligned.csv", index=False)

    print("Saved graph-regularized outputs to:", variant_outdir)


def compare_with_baseline(variant, variant_ranking):
    if not BASELINE_RANK.exists():
        return {
            "variant": variant,
            "baseline_spearman": np.nan,
            "top10_overlap": np.nan,
            "decoy_max_rank": np.nan,
            "positive_min_rank": np.nan,
        }

    base = pd.read_csv(BASELINE_RANK)

    m = base[["perturbation_id", "overall_rescue_score"]].merge(
        variant_ranking[["perturbation_id", "overall_rescue_score", "rescue_rank", "perturbation_type"]],
        on="perturbation_id",
        suffixes=("_baseline", "_graph"),
        how="inner",
    )

    if len(m) < 5:
        sp = np.nan
        overlap = np.nan
    else:
        sp = spearmanr(
            m["overall_rescue_score_baseline"],
            m["overall_rescue_score_graph"],
        ).correlation

        top_base = set(
            m.sort_values("overall_rescue_score_baseline", ascending=False)
            .head(10)["perturbation_id"]
        )
        top_graph = set(
            m.sort_values("overall_rescue_score_graph", ascending=False)
            .head(10)["perturbation_id"]
        )

        overlap = len(top_base & top_graph) / max(1, len(top_base | top_graph))

    decoy = variant_ranking[variant_ranking["perturbation_type"].astype(str).eq("negative_decoy")]
    pos = variant_ranking[~variant_ranking["perturbation_type"].astype(str).eq("negative_decoy")]

    return {
        "variant": variant,
        "baseline_spearman": float(sp) if pd.notna(sp) else np.nan,
        "top10_overlap": float(overlap) if pd.notna(overlap) else np.nan,
        "decoy_min_rank": int(decoy["rescue_rank"].min()) if len(decoy) else np.nan,
        "decoy_max_rank": int(decoy["rescue_rank"].max()) if len(decoy) else np.nan,
        "positive_max_rank": int(pos["rescue_rank"].max()) if len(pos) else np.nan,
        "mean_overall_score_positive": float(pos["overall_rescue_score"].mean()) if len(pos) else np.nan,
        "mean_overall_score_decoy": float(decoy["overall_rescue_score"].mean()) if len(decoy) else np.nan,
        "mean_strict_fraction_positive": float(pos["fraction_cells_rescued_strict"].mean()) if len(pos) else np.nan,
        "mean_strict_fraction_decoy": float(decoy["fraction_cells_rescued_strict"].mean()) if len(decoy) else np.nan,
        "mean_delta_norm_positive": float(pos["mean_delta_norm"].mean()) if len(pos) else np.nan,
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--variant",
        type=str,
        default="boundary_aware_graph",
        choices=["no_graph", "ordinary_graph", "boundary_aware_graph", "all"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--head-epochs", type=int, default=180)
    parser.add_argument("--perturb-epochs", type=int, default=240)
    parser.add_argument("--samples-per-epoch", type=int, default=16000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-cells-per-perturb", type=int, default=1200)

    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-perturb", type=float, default=5e-4)

    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--delta-reg-weight", type=float, default=0.12)
    parser.add_argument("--decoy-null-weight", type=float, default=2.5)
    parser.add_argument("--align-weight", type=float, default=0.08)

    parser.add_argument("--graph-weight", type=float, default=0.05)
    parser.add_argument("--graph-batch-edges", type=int, default=1024)
    parser.add_argument("--graph-on", type=str, default="delta", choices=["delta", "zp"])

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("=" * 100)
    print("37c | Train StrokeNiche Perturbation Adapter with graph loss")
    print("=" * 100)
    print("Device:", device)
    print("Variant:", args.variant)

    m37 = load_module37()

    inputs = m37.load_inputs()
    Z = inputs["Z"]
    Zs, z_mu, z_sd = m37.build_standardized_latent(Z)

    # Standardize prototypes
    proto_std = {}
    for k, v in inputs["prototypes"].items():
        proto_std[k] = ((v.reshape(1, -1) - z_mu) / z_sd).reshape(-1).astype(np.float32)
    inputs["prototypes_standardized"] = proto_std

    perturb_feats, perturb_meta = m37.build_perturbation_features(inputs["lib"])

    # Load graph
    state, proba, repair, coords, edges = load_graph_inputs(Z.shape[0])
    OUTROOT.mkdir(parents=True, exist_ok=True)
    edges.to_csv(OUTROOT / "training_boundary_graph_edges.csv", index=False)
    print("Saved:", OUTROOT / "training_boundary_graph_edges.csv")

    # Train / val split for state heads
    rng = np.random.default_rng(args.seed)
    idx = np.arange(Z.shape[0])
    rng.shuffle(idx)
    train_mask = np.zeros(Z.shape[0], dtype=bool)
    val_mask = np.zeros(Z.shape[0], dtype=bool)
    train_mask[idx[: int(0.8 * len(idx))]] = True
    val_mask[idx[int(0.8 * len(idx)) :]] = True

    # Train state heads once
    heads, head_hist = m37.train_state_heads(
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

    variants = ["no_graph", "ordinary_graph", "boundary_aware_graph"] if args.variant == "all" else [args.variant]

    comparison_rows = []

    for variant in variants:
        print("\n" + "=" * 100)
        print("Training graph-regularized variant:", variant)
        print("=" * 100)

        variant_outdir = OUTROOT / variant
        variant_outdir.mkdir(parents=True, exist_ok=True)

        g = graph_tensors(edges, variant=variant, device=device)
        effective_graph_weight = 0.0 if variant == "no_graph" else args.graph_weight

        adapter, hist = train_perturb_adapter_graph_regularized(
            m37=m37,
            Zs=Zs,
            inputs=inputs,
            perturb_feats=perturb_feats,
            heads=heads,
            device=device,
            variant=variant,
            graph=g,
            epochs=args.perturb_epochs,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            lr=args.lr_perturb,
            margin=args.margin,
            delta_reg_weight=args.delta_reg_weight,
            decoy_null_weight=args.decoy_null_weight,
            align_weight=args.align_weight,
            graph_weight=effective_graph_weight,
            graph_batch_edges=args.graph_batch_edges,
            graph_on=args.graph_on,
            seed=args.seed,
        )

        hist.to_csv(variant_outdir / "perturb_adapter_graph_training_history.csv", index=False)
        print("Saved:", variant_outdir / "perturb_adapter_graph_training_history.csv")

        # Save model
        model_path = variant_outdir / "perturb_adapter_graph_model.pt"
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
            "variant": variant,
        }, model_path)
        print("Saved:", model_path)

        # Score perturbations
        ranking, pred_df, mean_delta_arr = m37.score_all_perturbations(
            model=adapter,
            heads=heads,
            Zs=Zs,
            inputs=inputs,
            perturb_feats=perturb_feats,
            device=device,
            max_cells_per_perturb=args.max_cells_per_perturb,
        )

        save_graph_regularized_outputs(
            variant_outdir=variant_outdir,
            variant=variant,
            ranking=ranking,
            pred_df=pred_df,
            mean_delta_arr=mean_delta_arr,
            lib=inputs["lib"],
        )

        comp = compare_with_baseline(variant, ranking)
        comparison_rows.append(comp)

        # variant metadata
        meta = {
            "step": "37c graph-regularized perturbation adapter",
            "variant": variant,
            "graph_weight": effective_graph_weight,
            "graph_on": args.graph_on,
            "important_note": (
                "Graph loss is applied to perturbation-induced delta fields by default, "
                "not directly to original latent states, to reduce risk of lesion-boundary blurring."
            ),
            "args": vars(args),
        }

        (variant_outdir / "graph_regularized_metadata.json").write_text(
            json.dumps(safe_json(meta), indent=2, ensure_ascii=False)
        )

        print("Top graph-regularized perturbations:")
        cols = [
            "rescue_rank",
            "perturbation_id",
            "perturbation_type",
            "overall_rescue_score",
            "mean_rescue_score",
            "fraction_cells_rescued_strict",
            "mean_delta_norm",
        ]
        print(ranking[cols].head(20).to_string(index=False))

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = OUTROOT / "graph_regularized_perturb_adapter_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    print("\nSaved:", comp_path)
    print(comp_df.to_string(index=False))

    # top-level metadata
    meta = {
        "step": "37c graph-regularized perturbation adapter",
        "variants": variants,
        "output_root": str(OUTROOT),
        "graph_parameters": {
            "k_neighbors": K_NEIGHBORS,
            "state_tau": STATE_TAU,
            "repair_tau": REPAIR_TAU,
            "graph_weight": args.graph_weight,
            "graph_batch_edges": args.graph_batch_edges,
            "graph_on": args.graph_on,
        },
        "interpretation": (
            "Use boundary_aware_graph if it preserves baseline ranking stability, "
            "keeps decoys at the bottom, and reduces excessive delta variation without "
            "collapsing lesion-state perturbation specificity."
        ),
    }

    (OUTROOT / "graph_regularized_perturb_adapter_metadata.json").write_text(
        json.dumps(safe_json(meta), indent=2, ensure_ascii=False)
    )

    print("Saved:", OUTROOT / "graph_regularized_perturb_adapter_metadata.json")
    print("\nDone. Outputs saved to:", OUTROOT)


if __name__ == "__main__":
    main()
