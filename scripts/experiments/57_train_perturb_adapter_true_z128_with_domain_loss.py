#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
57_train_perturb_adapter_true_z128_with_domain_loss.py

Purpose
-------
Train a strict-z128 PerturbAdapter with optional L_domain.

This is a non-overwriting extension of Route B:

    z_i = strict Step7 Adapter hidden latent
    z_i^p = z_i + PerturbAdapter128(z_i, perturbation)
    neighbor_i^p = frozen strict NeighborHead(z_i^p)

Main task:
    Positive perturbations should increase repair-like neighbor state
    and reduce core-like neighbor state.
    Decoys should have minimal effect.

L_domain options:
    none       : no domain loss
    adv        : adversarial domain classifier with gradient reversal
    coral      : CORAL domain alignment on delta_z or z_p
    adv_coral  : adversarial + CORAL

Outputs
-------
results/step8_strokeniche_perturbmap/perturb_adapter_true_z128_Ldomain/<variant>/
  perturb_adapter_true_z128_Ldomain_model.pt
  perturb_adapter_true_z128_Ldomain_training_history.csv
  perturbed_true_z128_Ldomain_cell_level.csv
  virtual_perturbation_vectors_true_z128_Ldomain.csv
  domain_loss_report.txt
  domain_loss_metadata.json

Then run 54b with:
  --perturbed_latent perturbed_true_z128_Ldomain_cell_level.csv
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = Path("/mnt/h/vir/ST")

Z128_DEFAULT = BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54/adapter_true_hidden_latent_for_neighbor_head.csv"
PROFILE53C_DEFAULT = BASE / "results/step8_strokeniche_perturbmap/neighbor_shift_proxy_53c/virtual_cell_knn_neighbor_profile_RCTD.csv"
NEIGHBOR_HEAD_DEFAULT = BASE / "results/step8_strokeniche_perturbmap/neighbor_head_54_true_z/neighbor_head_model.pt"
PERTLIB_DEFAULT = BASE / "results/step8_strokeniche_perturbmap/perturbation_library.csv"
CORE_IDX_DEFAULT = BASE / "results/step8_strokeniche_perturbmap/input_core_like_indices.npy"
OUTROOT_DEFAULT = BASE / "results/step8_strokeniche_perturbmap/perturb_adapter_true_z128_Ldomain"


def log(msg: str):
    print(msg, flush=True)


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def safe_name(x):
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(x)).strip("_")[:120]


def sorted_latent_cols(df, prefix="adapter_latent_"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


def torch_load_trusted(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class NeighborHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, dropout, sigmoid_output):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        y = self.net(x)
        return torch.sigmoid(y) if self.sigmoid_output else y


def load_neighbor_head(path: Path, device: str):
    st = torch_load_trusted(path)
    model = NeighborHead(
        st["input_dim"],
        st["output_dim"],
        st["hidden"],
        st["dropout"],
        st["sigmoid_output"],
    )
    model.load_state_dict(st["model_state_dict"])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model, st


def normalize_z(z, nh_state):
    mean = torch.tensor(np.asarray(nh_state["x_mean"], float), dtype=torch.float32, device=z.device)
    std = torch.tensor(np.asarray(nh_state["x_std"], float), dtype=torch.float32, device=z.device)
    std = torch.where(std < 1e-8, torch.ones_like(std), std)
    return (z - mean) / std


def target_groups(target_cols):
    groups = {"repair": [], "core": [], "remote": [], "peri": []}
    for i, c in enumerate(target_cols):
        nc = norm_col(c)
        if "repair" in nc:
            groups["repair"].append(i)
        if "core" in nc or "lesion_core" in nc:
            groups["core"].append(i)
        if "remote" in nc:
            groups["remote"].append(i)
        if "peri" in nc:
            groups["peri"].append(i)
    return groups


def build_perturb_features(lib: pd.DataFrame):
    type_values = sorted(lib["perturbation_type"].fillna("unknown").astype(str).unique().tolist()) if "perturbation_type" in lib.columns else ["unknown"]
    direction_values = sorted(lib["direction"].fillna("unknown").astype(str).unique().tolist()) if "direction" in lib.columns else ["unknown"]

    rows = []
    for _, r in lib.iterrows():
        feat = []
        t = str(r.get("perturbation_type", "unknown"))
        d = str(r.get("direction", "unknown"))

        feat += [1.0 if t == x else 0.0 for x in type_values]
        feat += [1.0 if d == x else 0.0 for x in direction_values]

        for c in ["candidate_score", "candidate_score_norm", "is_positive_candidate", "is_negative_decoy"]:
            v = pd.to_numeric(r.get(c, 0.0), errors="coerce")
            feat.append(float(v) if pd.notna(v) else 0.0)

        feat.append(1.0 if str(r.get("ligand", "")).strip().lower() not in ["", "nan", "none"] else 0.0)
        feat.append(1.0 if str(r.get("receptor", "")).strip().lower() not in ["", "nan", "none"] else 0.0)
        feat.append(1.0)

        rows.append(feat)

    X = np.asarray(rows, dtype=np.float32)
    meta = {
        "type_values": type_values,
        "direction_values": direction_values,
        "feature_dim": int(X.shape[1]),
        "feature_schema": "type_onehot + direction_onehot + candidate_score + candidate_score_norm + positive/decoy flags + ligand/receptor flags + bias",
    }
    return X, meta


class PerturbAdapter128(nn.Module):
    def __init__(self, z_dim, n_perturb, feat_dim, emb_dim=128, hidden=256, dropout=0.10):
        super().__init__()
        self.perturb_id_emb = nn.Embedding(n_perturb, emb_dim)
        self.perturb_feat_mlp = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + emb_dim + emb_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z, pidx, pfeat):
        eid = self.perturb_id_emb(pidx)
        ef = self.perturb_feat_mlp(pfeat)
        x = torch.cat([z, eid, ef], dim=1)
        delta = self.decoder(x)
        return delta


class DomainClassifier(nn.Module):
    def __init__(self, input_dim, n_domain, hidden=128, dropout=0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domain),
        )

    def forward(self, x):
        return self.net(x)


def choose_domain_col(zdf: pd.DataFrame, prof: pd.DataFrame, domain_col: str):
    merged = zdf[["obs_name"]].copy()
    if "obs_name" in prof.columns:
        prof_small = prof.copy()
        cols = ["obs_name"] + [c for c in prof.columns if c != "obs_name"]
        prof_small = prof_small[cols]
        merged = merged.merge(prof_small, on="obs_name", how="left")

    if domain_col != "auto":
        if domain_col in merged.columns:
            return domain_col, merged[domain_col].fillna("unknown").astype(str).values, merged
        if domain_col in zdf.columns:
            return domain_col, zdf[domain_col].fillna("unknown").astype(str).values, merged
        raise RuntimeError(f"Requested domain_col={domain_col} not found.")

    candidates = [
        "sample_id", "sample", "sample_name", "batch", "batch_id",
        "section_id", "slice_id", "timepoint", "time", "donor", "replicate",
    ]
    for c in candidates:
        if c in merged.columns and merged[c].notna().sum() > 0 and merged[c].nunique(dropna=True) >= 2:
            return c, merged[c].fillna("unknown").astype(str).values, merged

    return None, np.array(["single_domain"] * len(zdf), dtype=object), merged


def encode_labels(values):
    vals = pd.Series(values).fillna("unknown").astype(str)
    classes = sorted(vals.unique().tolist())
    mapping = {c: i for i, c in enumerate(classes)}
    y = vals.map(mapping).values.astype(np.int64)
    return y, classes, mapping


def coral_loss(x, domain_y):
    """
    CORAL across domains. Align covariance/mean of representations across sampled domains.
    """
    unique = torch.unique(domain_y)
    if len(unique) < 2:
        return torch.tensor(0.0, device=x.device)

    losses = []
    stats = []
    for d in unique:
        idx = torch.where(domain_y == d)[0]
        if idx.numel() < 4:
            continue
        xd = x[idx]
        mean = xd.mean(dim=0)
        xc = xd - mean
        cov = (xc.t() @ xc) / max(1, xd.shape[0] - 1)
        stats.append((mean, cov))

    if len(stats) < 2:
        return torch.tensor(0.0, device=x.device)

    for i in range(len(stats)):
        for j in range(i + 1, len(stats)):
            mi, ci = stats[i]
            mj, cj = stats[j]
            losses.append(F.mse_loss(mi, mj) + F.mse_loss(ci, cj))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=x.device)


def domain_accuracy(logits, y):
    if logits is None or logits.numel() == 0:
        return np.nan
    pred = logits.argmax(dim=1)
    return float((pred == y).float().mean().detach().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z128", default=str(Z128_DEFAULT))
    ap.add_argument("--profile53c", default=str(PROFILE53C_DEFAULT))
    ap.add_argument("--neighbor_head", default=str(NEIGHBOR_HEAD_DEFAULT))
    ap.add_argument("--perturbation_library", default=str(PERTLIB_DEFAULT))
    ap.add_argument("--core_idx", default=str(CORE_IDX_DEFAULT))
    ap.add_argument("--outroot", default=str(OUTROOT_DEFAULT))
    ap.add_argument("--variant_name", default=None)
    ap.add_argument("--domain_col", default="auto")
    ap.add_argument("--domain_loss_type", choices=["none", "adv", "coral", "adv_coral"], default="adv_coral")
    ap.add_argument("--domain_on", choices=["zp", "delta"], default="zp")
    ap.add_argument("--domain_weight", type=float, default=0.05)
    ap.add_argument("--coral_weight", type=float, default=0.02)
    ap.add_argument("--grl_lambda", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--samples_per_epoch", type=int, default=12000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--delta_reg_weight", type=float, default=0.10)
    ap.add_argument("--decoy_null_weight", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--export_all_cells", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    variant = args.variant_name or f"{args.domain_loss_type}_domaincol_{args.domain_col}_on_{args.domain_on}_dw{args.domain_weight}_cw{args.coral_weight}"
    variant = safe_name(variant)

    outdir = Path(args.outroot) / variant
    outdir.mkdir(parents=True, exist_ok=True)

    log("=" * 100)
    log("57 train PerturbAdapter true-z128 with L_domain")
    log("=" * 100)
    log(f"outdir={outdir}")
    log(f"device={device}")

    zdf = pd.read_csv(args.z128, low_memory=False)
    zcols = sorted_latent_cols(zdf)
    if len(zcols) != 128:
        raise RuntimeError(f"Expected 128 adapter_latent_* columns, found {len(zcols)}")

    if "obs_name" not in zdf.columns:
        raise RuntimeError("z128 table lacks obs_name.")

    z_np = zdf[zcols].values.astype(np.float32)
    obs = zdf["obs_name"].astype(str).values

    prof = pd.read_csv(args.profile53c, low_memory=False)

    domain_col_used, domain_values, domain_merged = choose_domain_col(zdf, prof, args.domain_col)
    domain_y_np, domain_classes, domain_mapping = encode_labels(domain_values)
    n_domain = len(domain_classes)
    use_domain = args.domain_loss_type != "none" and n_domain >= 2

    log(f"domain_col_used={domain_col_used}")
    log(f"n_domain={n_domain}")
    log(f"domain_classes={domain_classes}")
    log(f"use_domain={use_domain}")

    nh, nh_state = load_neighbor_head(Path(args.neighbor_head), device)
    target_cols = list(nh_state["target_cols"])
    groups = target_groups(target_cols)

    lib = pd.read_csv(args.perturbation_library, low_memory=False)
    pfeat_np, pfeat_meta = build_perturb_features(lib)

    core_idx = np.load(args.core_idx).astype(int)
    train_idx = core_idx

    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    pfeat = torch.tensor(pfeat_np, dtype=torch.float32, device=device)
    domain_y = torch.tensor(domain_y_np, dtype=torch.long, device=device)

    with torch.no_grad():
        base_pred = nh(normalize_z(z, nh_state)).detach()

    is_pos = torch.tensor(
        pd.to_numeric(lib.get("is_positive_candidate", 1), errors="coerce").fillna(0).values.astype(np.float32),
        dtype=torch.float32,
        device=device,
    )
    is_neg = torch.tensor(
        pd.to_numeric(lib.get("is_negative_decoy", 0), errors="coerce").fillna(0).values.astype(np.float32),
        dtype=torch.float32,
        device=device,
    )

    model = PerturbAdapter128(
        z_dim=128,
        n_perturb=len(lib),
        feat_dim=pfeat_np.shape[1],
        emb_dim=128,
        hidden=256,
        dropout=args.dropout,
    ).to(device)

    domain_clf = None
    if use_domain and args.domain_loss_type in ["adv", "adv_coral"]:
        domain_clf = DomainClassifier(128, n_domain, hidden=128, dropout=args.dropout).to(device)

    params = list(model.parameters())
    if domain_clf is not None:
        params += list(domain_clf.parameters())

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    repair_idx = groups["repair"]
    core_idx2 = groups["core"]
    remote_idx = groups["remote"]

    hist = []

    for ep in range(1, args.epochs + 1):
        losses = []
        task_losses = []
        domain_losses = []
        coral_losses = []
        reg_losses = []
        decoy_losses = []
        domain_accs = []
        pos_scores = []

        steps = max(1, args.samples_per_epoch // args.batch_size)

        model.train()
        if domain_clf is not None:
            domain_clf.train()

        for _ in range(steps):
            cidx = rng.choice(train_idx, size=args.batch_size, replace=True)
            pidx_np = rng.integers(0, len(lib), size=args.batch_size)

            cidx_t = torch.tensor(cidx, dtype=torch.long, device=device)
            pidx = torch.tensor(pidx_np, dtype=torch.long, device=device)

            z0 = z[cidx_t]
            pf = pfeat[pidx]
            dy = domain_y[cidx_t]

            delta = model(z0, pidx, pf)
            zp = z0 + delta

            yp = nh(normalize_z(zp, nh_state))
            y0 = base_pred[cidx_t]

            score = torch.zeros(args.batch_size, dtype=torch.float32, device=device)

            if repair_idx:
                score = score + (yp[:, repair_idx].mean(1) - y0[:, repair_idx].mean(1))
            if core_idx2:
                score = score - 0.75 * (yp[:, core_idx2].mean(1) - y0[:, core_idx2].mean(1))
            if remote_idx:
                score = score + 0.25 * (yp[:, remote_idx].mean(1) - y0[:, remote_idx].mean(1))

            pos_mask = is_pos[pidx] > 0.5
            neg_mask = is_neg[pidx] > 0.5

            if pos_mask.any():
                pos_loss = -score[pos_mask].mean()
                pos_scores.append(float(score[pos_mask].mean().detach().cpu()))
            else:
                pos_loss = torch.tensor(0.0, device=device)

            shift = ((yp - y0) ** 2).mean(1)

            if neg_mask.any():
                decoy_loss = shift[neg_mask].mean() + (delta[neg_mask] ** 2).mean()
            else:
                decoy_loss = torch.tensor(0.0, device=device)

            reg_loss = (delta ** 2).mean()

            task_loss = pos_loss + args.decoy_null_weight * decoy_loss + args.delta_reg_weight * reg_loss

            dom_loss = torch.tensor(0.0, device=device)
            cor_loss = torch.tensor(0.0, device=device)
            dom_acc = np.nan

            domain_repr = zp if args.domain_on == "zp" else delta

            if use_domain and args.domain_loss_type in ["adv", "adv_coral"] and domain_clf is not None:
                logits = domain_clf(grad_reverse(domain_repr, args.grl_lambda))
                dom_loss = F.cross_entropy(logits, dy)
                dom_acc = domain_accuracy(logits, dy)

            if use_domain and args.domain_loss_type in ["coral", "adv_coral"]:
                cor_loss = coral_loss(domain_repr, dy)

            loss = (
                task_loss
                + args.domain_weight * dom_loss
                + args.coral_weight * cor_loss
            )

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

            losses.append(float(loss.detach().cpu()))
            task_losses.append(float(task_loss.detach().cpu()))
            domain_losses.append(float(dom_loss.detach().cpu()))
            coral_losses.append(float(cor_loss.detach().cpu()))
            reg_losses.append(float(reg_loss.detach().cpu()))
            decoy_losses.append(float(decoy_loss.detach().cpu()))
            domain_accs.append(dom_acc)

        row = {
            "epoch": ep,
            "loss": float(np.nanmean(losses)),
            "task_loss": float(np.nanmean(task_losses)),
            "domain_loss": float(np.nanmean(domain_losses)),
            "coral_loss": float(np.nanmean(coral_losses)),
            "delta_reg": float(np.nanmean(reg_losses)),
            "decoy_loss": float(np.nanmean(decoy_losses)),
            "domain_acc": float(np.nanmean(domain_accs)) if np.isfinite(np.nanmean(domain_accs)) else np.nan,
            "mean_positive_neighbor_score": float(np.nanmean(pos_scores)) if pos_scores else np.nan,
        }
        hist.append(row)

        if ep == 1 or ep % 20 == 0 or ep == args.epochs:
            log(
                f"epoch={ep} "
                f"loss={row['loss']:.6f} "
                f"task={row['task_loss']:.6f} "
                f"domain={row['domain_loss']:.6f} "
                f"coral={row['coral_loss']:.6f} "
                f"domain_acc={row['domain_acc']} "
                f"pos_score={row['mean_positive_neighbor_score']}"
            )

    hist_df = pd.DataFrame(hist)
    hist_df.to_csv(outdir / "perturb_adapter_true_z128_Ldomain_training_history.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "domain_classifier_state_dict": domain_clf.state_dict() if domain_clf is not None else None,
            "z_dim": 128,
            "n_perturb": len(lib),
            "perturb_feature_dim": pfeat_np.shape[1],
            "perturb_feature_metadata": pfeat_meta,
            "neighbor_head": str(args.neighbor_head),
            "domain_col_used": domain_col_used,
            "domain_classes": domain_classes,
            "domain_mapping": domain_mapping,
            "domain_loss_type": args.domain_loss_type,
            "domain_on": args.domain_on,
            "args": vars(args),
        },
        outdir / "perturb_adapter_true_z128_Ldomain_model.pt",
    )

    # Export perturbed z for 54b.
    export_idx = np.arange(len(z_np)) if args.export_all_cells else core_idx
    rows = []
    mean_delta_rows = []

    model.eval()
    with torch.no_grad():
        for pi, prow in lib.iterrows():
            pidx = torch.full((len(export_idx),), pi, dtype=torch.long, device=device)
            cidx_t = torch.tensor(export_idx, dtype=torch.long, device=device)

            z0 = z[cidx_t]
            pf = pfeat[pidx]
            delta = model(z0, pidx, pf)
            zp = z0 + delta

            delta_np = delta.detach().cpu().numpy()
            zp_np = zp.detach().cpu().numpy()

            m = pd.DataFrame({
                "perturbation_id": prow.get("perturbation_id", f"perturb_{pi}"),
                "cell_index": export_idx,
                "obs_name": obs[export_idx],
                "domain_label": domain_values[export_idx],
            })

            for c in [
                "perturbation_type", "target_gene", "ligand", "receptor",
                "direction", "candidate_score", "candidate_score_norm",
                "source_evidence", "mechanism_hint",
                "is_positive_candidate", "is_negative_decoy",
            ]:
                if c in lib.columns:
                    m[c] = prow.get(c)

            delta_df = pd.DataFrame(delta_np, columns=[f"delta_z_{j}" for j in range(128)])
            zp_df = pd.DataFrame(zp_np, columns=[f"adapter_latent_p_{j}" for j in range(128)])
            m = pd.concat([m.reset_index(drop=True), delta_df, zp_df], axis=1)
            rows.append(m)

            mr = {
                "perturbation_id": prow.get("perturbation_id", f"perturb_{pi}"),
            }
            for c in [
                "perturbation_type", "target_gene", "ligand", "receptor",
                "direction", "candidate_score", "candidate_score_norm",
                "is_positive_candidate", "is_negative_decoy",
            ]:
                if c in lib.columns:
                    mr[c] = prow.get(c)

            for j in range(128):
                mr[f"delta_z_{j}"] = float(delta_np[:, j].mean())
            mr["mean_delta_norm"] = float(np.linalg.norm(delta_np, axis=1).mean())
            mean_delta_rows.append(mr)

    cell = pd.concat(rows, ignore_index=True)
    cell_path = outdir / "perturbed_true_z128_Ldomain_cell_level.csv"
    cell.to_csv(cell_path, index=False)

    mean_delta = pd.DataFrame(mean_delta_rows)
    mean_delta_path = outdir / "virtual_perturbation_vectors_true_z128_Ldomain.csv"
    mean_delta.to_csv(mean_delta_path, index=False)

    meta = {
        "status": "ok",
        "mode": "RouteB_true_z128_with_L_domain",
        "variant": variant,
        "z128": str(args.z128),
        "profile53c": str(args.profile53c),
        "neighbor_head": str(args.neighbor_head),
        "perturbation_library": str(args.perturbation_library),
        "domain_col_used": domain_col_used,
        "domain_classes": domain_classes,
        "n_domain": n_domain,
        "use_domain": use_domain,
        "domain_loss_type": args.domain_loss_type,
        "domain_on": args.domain_on,
        "domain_weight": args.domain_weight,
        "coral_weight": args.coral_weight,
        "n_cell_level_rows": int(len(cell)),
        "n_perturbations": int(len(lib)),
        "cell_level_output": str(cell_path),
        "mean_delta_output": str(mean_delta_path),
        "model": str(outdir / "perturb_adapter_true_z128_Ldomain_model.pt"),
        "history": str(outdir / "perturb_adapter_true_z128_Ldomain_training_history.csv"),
    }
    (outdir / "domain_loss_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    lines = []
    lines.append("57 RouteB true-z128 PerturbAdapter with L_domain report")
    lines.append("=" * 100)
    lines.append("SUCCESS")
    lines.append(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    lines.append("")
    lines.append("Training tail:")
    lines.append(hist_df.tail(30).to_string(index=False))
    lines.append("")
    lines.append("Mean delta head:")
    lines.append(mean_delta.head(30).to_string(index=False))
    (outdir / "domain_loss_report.txt").write_text("\n".join(lines), encoding="utf-8")

    log("=" * 100)
    log("DONE 57 L_domain")
    log("=" * 100)
    log(json.dumps(meta, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
