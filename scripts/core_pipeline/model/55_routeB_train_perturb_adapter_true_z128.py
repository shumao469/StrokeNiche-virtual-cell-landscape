#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Route B:
Train a new PerturbAdapter in strict 128-d z_i space, using the frozen strict-z NeighborHead
as differentiable readout. Export cell-level z_i^p for 54b.

Does NOT overwrite existing 64-d PerturbAdapter.

Outputs:
  results/step8_strokeniche_perturbmap/perturb_adapter_true_z128/
    perturb_adapter_true_z128_model.pt
    perturbed_true_z128_cell_level.csv
    virtual_perturbation_vectors_true_z128.csv
    perturb_adapter_true_z128_report.txt
"""

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
OUTDIR_DEFAULT = BASE / "results/step8_strokeniche_perturbmap/perturb_adapter_true_z128"


def norm_col(c):
    return re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")


def safe(c):
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(c)).strip("_")[:100]


def sorted_latent_cols(df, prefix="adapter_latent_"):
    cols = [c for c in df.columns if c.startswith(prefix)]
    return sorted(cols, key=lambda x: int(x.split("_")[-1]))


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


class NeighborHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, dropout, sigmoid_output):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        y = self.net(x)
        return torch.sigmoid(y) if self.sigmoid_output else y


def load_neighbor_head(path, device):
    try:
        st = torch.load(path, map_location="cpu")
    except Exception:
        # Local self-generated checkpoint; needed for PyTorch >=2.6 when numpy objects are stored.
        st = torch.load(path, map_location="cpu", weights_only=False)

    model = NeighborHead(st["input_dim"], st["output_dim"], st["hidden"], st["dropout"], st["sigmoid_output"])
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


def build_perturb_features(lib):
    type_values = sorted(lib["perturbation_type"].fillna("unknown").astype(str).unique().tolist()) if "perturbation_type" in lib.columns else ["unknown"]
    direction_values = sorted(lib["direction"].fillna("unknown").astype(str).unique().tolist()) if "direction" in lib.columns else ["unknown"]

    rows = []
    for _, r in lib.iterrows():
        feat = []
        t = str(r.get("perturbation_type", "unknown"))
        d = str(r.get("direction", "unknown"))
        feat += [1.0 if t == x else 0.0 for x in type_values]
        feat += [1.0 if d == x else 0.0 for x in direction_values]
        feat.append(float(pd.to_numeric(r.get("candidate_score", 0), errors="coerce") if pd.notna(r.get("candidate_score", 0)) else 0))
        feat.append(float(pd.to_numeric(r.get("candidate_score_norm", 0), errors="coerce") if pd.notna(r.get("candidate_score_norm", 0)) else 0))
        feat.append(float(pd.to_numeric(r.get("is_positive_candidate", 0), errors="coerce") if pd.notna(r.get("is_positive_candidate", 0)) else 0))
        feat.append(float(pd.to_numeric(r.get("is_negative_decoy", 0), errors="coerce") if pd.notna(r.get("is_negative_decoy", 0)) else 0))
        feat.append(1.0 if str(r.get("ligand", "")) not in ["", "nan", "None"] else 0.0)
        feat.append(1.0 if str(r.get("receptor", "")) not in ["", "nan", "None"] else 0.0)
        feat.append(1.0)
        rows.append(feat)

    X = np.asarray(rows, dtype=np.float32)
    meta = {
        "type_values": type_values,
        "direction_values": direction_values,
        "feature_dim": int(X.shape[1]),
        "feature_schema": "type_onehot + direction_onehot + candidate_score + candidate_score_norm + is_positive + is_negative + ligand_present + receptor_present + bias",
    }
    return X, meta


class PerturbAdapter128(nn.Module):
    def __init__(self, z_dim, n_perturb, feat_dim, emb_dim=128, hidden=256, dropout=0.1):
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
        return self.decoder(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--z128", default=str(Z128_DEFAULT))
    ap.add_argument("--profile53c", default=str(PROFILE53C_DEFAULT))
    ap.add_argument("--neighbor_head", default=str(NEIGHBOR_HEAD_DEFAULT))
    ap.add_argument("--perturbation_library", default=str(PERTLIB_DEFAULT))
    ap.add_argument("--core_idx", default=str(CORE_IDX_DEFAULT))
    ap.add_argument("--outdir", default=str(OUTDIR_DEFAULT))
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--samples_per_epoch", type=int, default=16000)
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

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"

    zdf = pd.read_csv(args.z128, low_memory=False)
    zcols = sorted_latent_cols(zdf)
    if len(zcols) != 128:
        raise RuntimeError(f"Expected 128 adapter_latent columns, found {len(zcols)}")

    z_np = zdf[zcols].values.astype(np.float32)
    obs = zdf["obs_name"].astype(str).values

    prof = pd.read_csv(args.profile53c, low_memory=False)
    nh, nh_state = load_neighbor_head(Path(args.neighbor_head), device)
    target_cols = list(nh_state["target_cols"])
    groups = target_groups(target_cols)

    lib = pd.read_csv(args.perturbation_library, low_memory=False)
    pfeat_np, pfeat_meta = build_perturb_features(lib)

    core_idx = np.load(args.core_idx).astype(int)
    train_idx = core_idx

    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    pfeat = torch.tensor(pfeat_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        base_pred = nh(normalize_z(z, nh_state)).detach()

    is_pos = torch.tensor(pd.to_numeric(lib.get("is_positive_candidate", 1), errors="coerce").fillna(0).values.astype(np.float32), device=device)
    is_neg = torch.tensor(pd.to_numeric(lib.get("is_negative_decoy", 0), errors="coerce").fillna(0).values.astype(np.float32), device=device)

    model = PerturbAdapter128(128, len(lib), pfeat_np.shape[1], emb_dim=128, hidden=256, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    repair_idx = groups["repair"]
    core_idx2 = groups["core"]
    remote_idx = groups["remote"]

    hist = []
    rng = np.random.default_rng(args.seed)

    for ep in range(1, args.epochs + 1):
        losses = []
        pos_scores = []
        for _ in range(max(1, args.samples_per_epoch // args.batch_size)):
            cidx = rng.choice(train_idx, size=args.batch_size, replace=True)
            pidx_np = rng.integers(0, len(lib), size=args.batch_size)

            cidx_t = torch.tensor(cidx, dtype=torch.long, device=device)
            pidx = torch.tensor(pidx_np, dtype=torch.long, device=device)

            z0 = z[cidx_t]
            pf = pfeat[pidx]
            delta = model(z0, pidx, pf)
            zp = z0 + delta

            yp = nh(normalize_z(zp, nh_state))
            y0 = base_pred[cidx_t]

            score = torch.zeros(args.batch_size, device=device)
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
            loss = pos_loss + args.decoy_null_weight * decoy_loss + args.delta_reg_weight * reg_loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))

        row = {
            "epoch": ep,
            "loss": float(np.mean(losses)),
            "mean_positive_neighbor_score": float(np.mean(pos_scores)) if pos_scores else np.nan,
        }
        hist.append(row)
        if ep == 1 or ep % 20 == 0:
            print(f"epoch={ep} loss={row['loss']:.6f} pos_score={row['mean_positive_neighbor_score']}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "z_dim": 128,
        "n_perturb": len(lib),
        "perturb_feature_dim": pfeat_np.shape[1],
        "perturb_feature_metadata": pfeat_meta,
        "neighbor_head": str(args.neighbor_head),
        "args": vars(args),
    }, outdir / "perturb_adapter_true_z128_model.pt")

    pd.DataFrame(hist).to_csv(outdir / "perturb_adapter_true_z128_training_history.csv", index=False)

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

            delta_np = delta.cpu().numpy()
            zp_np = zp.cpu().numpy()

            m = pd.DataFrame({
                "perturbation_id": prow.get("perturbation_id", f"perturb_{pi}"),
                "cell_index": export_idx,
                "obs_name": obs[export_idx],
            })

            for c in ["perturbation_type", "target_gene", "ligand", "receptor", "direction", "candidate_score", "candidate_score_norm", "source_evidence", "mechanism_hint", "is_positive_candidate", "is_negative_decoy"]:
                if c in lib.columns:
                    m[c] = prow.get(c)

            for j in range(128):
                m[f"delta_z_{j}"] = delta_np[:, j]
            for j in range(128):
                m[f"adapter_latent_p_{j}"] = zp_np[:, j]

            rows.append(m)

            mr = {c: prow.get(c) for c in lib.columns if c in ["perturbation_id", "perturbation_type", "target_gene", "ligand", "receptor", "direction", "candidate_score", "candidate_score_norm", "is_positive_candidate", "is_negative_decoy"]}
            for j in range(128):
                mr[f"delta_z_{j}"] = float(delta_np[:, j].mean())
            mr["mean_delta_norm"] = float(np.linalg.norm(delta_np, axis=1).mean())
            mean_delta_rows.append(mr)

    cell = pd.concat(rows, ignore_index=True)
    cell.to_csv(outdir / "perturbed_true_z128_cell_level.csv", index=False)

    mean_delta = pd.DataFrame(mean_delta_rows)
    mean_delta.to_csv(outdir / "virtual_perturbation_vectors_true_z128.csv", index=False)

    meta = {
        "status": "ok",
        "mode": "RouteB_true_z128_PerturbAdapter",
        "z128": str(args.z128),
        "neighbor_head": str(args.neighbor_head),
        "profile53c": str(args.profile53c),
        "perturbation_library": str(args.perturbation_library),
        "export_cells": "all" if args.export_all_cells else str(args.core_idx),
        "n_cell_level_rows": int(len(cell)),
        "n_perturbations": int(len(lib)),
        "cell_level_output": str(outdir / "perturbed_true_z128_cell_level.csv"),
        "mean_delta_output": str(outdir / "virtual_perturbation_vectors_true_z128.csv"),
    }
    (outdir / "perturb_adapter_true_z128_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    report = []
    report.append("55 Route B true-z128 PerturbAdapter report")
    report.append("=" * 90)
    report.append("SUCCESS")
    report.append(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    report.append("")
    report.append("Training tail:")
    report.append(pd.DataFrame(hist).tail(20).to_string(index=False))
    report.append("")
    report.append("Mean delta head:")
    report.append(mean_delta.head(30).to_string(index=False))
    (outdir / "perturb_adapter_true_z128_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("DONE Route B train/export true z128")
    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
