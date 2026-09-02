#!/usr/bin/env python3
"""Train the baseline StrokeNiche adapter.

The neighbour-composition target is disabled as an input by default because
feeding it back into the model leaks the reconstruction target, especially at
validation/test time.  The legacy behaviour remains available only behind an
explicit acknowledgement flag so historical experiments can be reproduced.
"""
from pathlib import Path
import json
import math
import argparse
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import label_binarize


PROJECT = Path("/mnt/h/vir/ST")
OUTDIR = PROJECT / "results/step7_strokeniche_adapter_clean"
CKPT_DIR = OUTDIR / "checkpoints"
DOWN = OUTDIR / "downstream"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
DOWN.mkdir(parents=True, exist_ok=True)


class SparseGraphAttentionBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(dim)

    def forward(self, h, edge_index):
        # edge_index: [2, E], source -> target
        src = edge_index[0]
        dst = edge_index[1]
        n = h.size(0)

        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        score = (q[dst] * k[src]).sum(dim=-1) / self.scale

        # Sparse attention matrix: target attends to source.
        indices = torch.stack([dst, src], dim=0)
        attn = torch.sparse_coo_tensor(
            indices,
            score,
            size=(n, n),
            device=h.device,
        )
        attn = torch.sparse.softmax(attn, dim=1)
        out = torch.sparse.mm(attn, v)
        out = self.o(out)

        h = self.norm1(h + self.dropout(out))
        h = self.norm2(h + self.dropout(self.ffn(h)))
        return h


class StrokeNicheAdapter(nn.Module):
    def __init__(
        self,
        nf_dim,
        coord_dim,
        neighbor_dim,
        n_regions,
        n_time,
        hidden_dim=128,
        latent_dim=64,
        n_layers=2,
        dropout=0.1,
        use_neighbor_input=False,
        use_time_input=False,
    ):
        super().__init__()

        self.neighbor_dim = neighbor_dim
        self.n_time = n_time
        self.use_neighbor_input = use_neighbor_input
        self.use_time_input = use_time_input

        input_dim = nf_dim + coord_dim

        if use_neighbor_input:
            input_dim += neighbor_dim

        if use_time_input and n_time > 0:
            input_dim += n_time

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.blocks = nn.ModuleList([
            SparseGraphAttentionBlock(hidden_dim, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.region_head = nn.Linear(latent_dim, n_regions)

        self.time_head = nn.Linear(latent_dim, n_time) if n_time > 0 else None

        self.neighbor_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, neighbor_dim),
            nn.Softplus(),
        ) if neighbor_dim > 0 else None

        self.repair_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        X_nf,
        coords,
        edge_index,
        neighbor_input=None,
        time_onehot=None,
    ):
        xs = [X_nf, coords]

        if self.use_neighbor_input:
            if neighbor_input is None:
                raise ValueError("neighbor_input is required when use_neighbor_input=True")
            xs.append(neighbor_input)

        if self.use_time_input and self.n_time > 0:
            if time_onehot is None:
                raise ValueError("time_onehot is required when use_time_input=True")
            xs.append(time_onehot)

        h = torch.cat(xs, dim=1)
        h = self.input_proj(h)

        for block in self.blocks:
            h = block(h, edge_index)

        z = self.proj_head(h)
        z_norm = F.normalize(z, dim=1)

        out = {
            "z": z,
            "z_norm": z_norm,
            "region_logits": self.region_head(z),
            "repair_score": torch.sigmoid(self.repair_head(z)).squeeze(-1),
        }

        if self.time_head is not None:
            out["time_logits"] = self.time_head(z)

        if self.neighbor_head is not None:
            neigh = self.neighbor_head(z)
            neigh = neigh / (neigh.sum(dim=1, keepdim=True) + 1e-8)
            out["neighbor_pred"] = neigh

        return out


def supervised_contrastive_loss(z, labels, mask, temperature=0.2, max_samples=2048):
    idx = torch.where(mask)[0]

    if idx.numel() < 4:
        return z.new_tensor(0.0)

    if idx.numel() > max_samples:
        perm = torch.randperm(idx.numel(), device=z.device)[:max_samples]
        idx = idx[perm]

    z = z[idx]
    labels = labels[idx]

    sim = torch.matmul(z, z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()

    same = labels[:, None].eq(labels[None, :])
    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)

    positives = same & (~eye)

    exp_sim = torch.exp(sim) * (~eye).float()
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = positives.sum(dim=1)
    valid = pos_count > 0

    if valid.sum() == 0:
        return z.new_tensor(0.0)

    loss = -(log_prob * positives.float()).sum(dim=1) / (pos_count.float() + 1e-8)
    return loss[valid].mean()


def make_contrastive_labels(region_raw, time_raw, dominant_celltype):
    labels = []
    for i in range(len(region_raw)):
        r = str(region_raw[i])
        ct = str(dominant_celltype[i])
        if time_raw is not None and len(time_raw) == len(region_raw):
            t = str(time_raw[i])
        else:
            t = "NA"
        labels.append(f"{r}|{t}|{ct}")

    uniq = {v: i for i, v in enumerate(sorted(set(labels)))}
    return np.array([uniq[v] for v in labels], dtype=np.int64), uniq


def masked_neighbor_input(Y_neighbor, mask_prob=0.5):
    if Y_neighbor.size(1) == 0:
        return Y_neighbor

    keep = (torch.rand_like(Y_neighbor) > mask_prob).float()
    x = Y_neighbor * keep
    row_sum = x.sum(dim=1, keepdim=True)
    x = torch.where(row_sum > 0, x / (row_sum + 1e-8), x)
    return x


def compute_classification_metrics(y_true, logits, prefix="region"):
    y_pred = logits.argmax(axis=1)

    metrics = {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        f"{prefix}_macro_f1": f1_score(y_true, y_pred, average="macro"),
        f"{prefix}_weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    }

    try:
        proba = torch.softmax(torch.tensor(logits), dim=1).numpy()
        classes = np.arange(logits.shape[1])
        yy = label_binarize(y_true, classes=classes)
        if yy.shape[1] == proba.shape[1]:
            metrics[f"{prefix}_auroc_ovr_macro"] = roc_auc_score(
                yy,
                proba,
                average="macro",
                multi_class="ovr",
            )
    except Exception as e:
        metrics[f"{prefix}_auroc_error"] = repr(e)

    return metrics, y_pred


def evaluate(
    model,
    tensors,
    masks,
    meta,
    device,
    use_neighbor_input=False,
    use_time_input=False,
):
    model.eval()

    X_nf = tensors["X_nf"]
    coords = tensors["coords"]
    edge_index = tensors["edge_index"]
    Y_neighbor = tensors["Y_neighbor"]
    y_time = tensors["y_time"]
    time_onehot = tensors["time_onehot"]

    if use_neighbor_input and Y_neighbor.shape[1] > 0:
        neighbor_input = Y_neighbor
    else:
        neighbor_input = None

    if use_time_input and y_time is not None:
        time_in = time_onehot
    else:
        time_in = None

    with torch.no_grad():
        out = model(
            X_nf=X_nf,
            coords=coords,
            edge_index=edge_index,
            neighbor_input=neighbor_input,
            time_onehot=time_in,
        )

    result = {}
    arrays = {}

    for split_name, mask in masks.items():
        m = mask.cpu().numpy().astype(bool)
        y_region = tensors["y_region"].cpu().numpy()

        logits_region = out["region_logits"].detach().cpu().numpy()
        metrics, y_pred = compute_classification_metrics(
            y_region[m],
            logits_region[m],
            prefix=f"{split_name}_region",
        )
        result.update(metrics)

        cm = confusion_matrix(y_region[m], y_pred)
        arrays[f"{split_name}_region_cm"] = cm

        rep = classification_report(y_region[m], y_pred, output_dict=True)
        arrays[f"{split_name}_region_report"] = pd.DataFrame(rep).T

        if "time_logits" in out and tensors["y_time"] is not None:
            y_t = tensors["y_time"].cpu().numpy()
            logits_time = out["time_logits"].detach().cpu().numpy()
            t_metrics, t_pred = compute_classification_metrics(
                y_t[m],
                logits_time[m],
                prefix=f"{split_name}_time",
            )
            result.update(t_metrics)

        if "neighbor_pred" in out and Y_neighbor.shape[1] > 0:
            yp = out["neighbor_pred"].detach().cpu().numpy()[m]
            yt = Y_neighbor.detach().cpu().numpy()[m]

            result[f"{split_name}_neighbor_mae"] = mean_absolute_error(yt, yp)
            result[f"{split_name}_neighbor_rmse"] = mean_squared_error(yt, yp) ** 0.5

            by_target = []
            neighbor_cols = meta.get("neighbor_cols", [])
            for j in range(yt.shape[1]):
                name = neighbor_cols[j] if j < len(neighbor_cols) else f"target_{j}"
                by_target.append({
                    "split": split_name,
                    "target": name,
                    "r2": r2_score(yt[:, j], yp[:, j]),
                    "mae": mean_absolute_error(yt[:, j], yp[:, j]),
                    "rmse": mean_squared_error(yt[:, j], yp[:, j]) ** 0.5,
                })
            arrays[f"{split_name}_neighbor_by_target"] = pd.DataFrame(by_target)

        repair = out["repair_score"].detach().cpu().numpy()[m]
        target = tensors["repair_target"].detach().cpu().numpy()[m]
        result[f"{split_name}_repair_mae"] = mean_absolute_error(target, repair)
        result[f"{split_name}_repair_rmse"] = mean_squared_error(target, repair) ** 0.5

    arrays["z"] = out["z"].detach().cpu().numpy()
    arrays["repair_score"] = out["repair_score"].detach().cpu().numpy()
    arrays["region_proba"] = torch.softmax(out["region_logits"], dim=1).detach().cpu().numpy()

    if "neighbor_pred" in out:
        arrays["neighbor_pred"] = out["neighbor_pred"].detach().cpu().numpy()

    return result, arrays


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-npz", default=str(OUTDIR / "strokeniche_adapter_inputs.npz"))
    ap.add_argument("--metadata-json", default=str(OUTDIR / "strokeniche_adapter_input_metadata.json"))
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--latent-dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument(
        "--use-neighbor-input",
        type=int,
        default=0,
        choices=(0, 1),
        help="Legacy target-derived feature. Keep at 0 for leakage-safe runs.",
    )
    ap.add_argument(
        "--allow-target-derived-neighbor-input",
        action="store_true",
        help="Acknowledge leakage risk and reproduce the historical neighbour-input run.",
    )
    ap.add_argument("--neighbor-mask-prob", type=float, default=0.5)
    ap.add_argument("--use-time-input", type=int, default=0)
    ap.add_argument("--w-region", type=float, default=1.0)
    ap.add_argument("--w-time", type=float, default=0.3)
    ap.add_argument("--w-neighbor", type=float, default=0.5)
    ap.add_argument("--w-contrastive", type=float, default=0.1)
    ap.add_argument("--w-repair", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--trust-pickle-input",
        action="store_true",
        help="Allow object arrays in an NPZ only when the local file is trusted.",
    )
    args = ap.parse_args()

    if args.use_neighbor_input and not args.allow_target_derived_neighbor_input:
        ap.error(
            "--use-neighbor-input=1 exposes the neighbour reconstruction target "
            "to the model. Add --allow-target-derived-neighbor-input only for an "
            "explicitly labelled legacy reproduction."
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data = np.load(args.input_npz, allow_pickle=args.trust_pickle_input)
    meta = json.loads(Path(args.metadata_json).read_text())

    X_nf = data["X_nf"].astype(np.float32)
    coords = data["coords_scaled"].astype(np.float32)
    Y_neighbor = data["Y_neighbor"].astype(np.float32)
    y_region = data["y_region"].astype(np.int64)
    has_time = bool(data["has_time"][0])
    y_time = data["y_time"].astype(np.int64) if has_time else None
    repair_target = data["repair_target"].astype(np.float32)
    edge_index = data["edge_index"].astype(np.int64)
    train_mask = data["train_mask"].astype(bool)
    val_mask = data["val_mask"].astype(bool)
    test_mask = data["test_mask"].astype(bool)

    region_raw = data["region_raw"].astype(str)
    dominant_celltype = data["dominant_celltype"].astype(str)
    time_raw = data["time_raw"].astype(str) if has_time else None

    contrast_labels, contrast_map = make_contrastive_labels(
        region_raw=region_raw,
        time_raw=time_raw,
        dominant_celltype=dominant_celltype,
    )

    n_regions = len(meta["region_map"])
    n_time = len(meta["time_map"]) if meta.get("time_map") is not None else 0

    time_onehot = None
    if has_time and n_time > 0:
        time_onehot_np = np.zeros((len(y_time), n_time), dtype=np.float32)
        time_onehot_np[np.arange(len(y_time)), y_time] = 1.0
        time_onehot = torch.tensor(time_onehot_np, dtype=torch.float32, device=device)

    tensors = {
        "X_nf": torch.tensor(X_nf, dtype=torch.float32, device=device),
        "coords": torch.tensor(coords, dtype=torch.float32, device=device),
        "Y_neighbor": torch.tensor(Y_neighbor, dtype=torch.float32, device=device),
        "y_region": torch.tensor(y_region, dtype=torch.long, device=device),
        "y_time": torch.tensor(y_time, dtype=torch.long, device=device) if has_time else None,
        "time_onehot": time_onehot,
        "repair_target": torch.tensor(repair_target, dtype=torch.float32, device=device),
        "edge_index": torch.tensor(edge_index, dtype=torch.long, device=device),
        "contrast_labels": torch.tensor(contrast_labels, dtype=torch.long, device=device),
    }

    masks = {
        "train": torch.tensor(train_mask, dtype=torch.bool, device=device),
        "val": torch.tensor(val_mask, dtype=torch.bool, device=device),
        "test": torch.tensor(test_mask, dtype=torch.bool, device=device),
    }

    model = StrokeNicheAdapter(
        nf_dim=X_nf.shape[1],
        coord_dim=coords.shape[1],
        neighbor_dim=Y_neighbor.shape[1],
        n_regions=n_regions,
        n_time=n_time if has_time else 0,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_neighbor_input=bool(args.use_neighbor_input and Y_neighbor.shape[1] > 0),
        use_time_input=bool(args.use_time_input and has_time),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_state = None
    best_val = -1
    best_epoch = -1
    bad = 0

    history = []

    print(model)
    print("Training samples:", int(train_mask.sum()))
    print("Validation samples:", int(val_mask.sum()))
    print("Test samples:", int(test_mask.sum()))

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        neighbor_input = None
        if model.use_neighbor_input:
            neighbor_input = masked_neighbor_input(
                tensors["Y_neighbor"],
                mask_prob=args.neighbor_mask_prob,
            )

        time_input = tensors["time_onehot"] if model.use_time_input else None

        out = model(
            X_nf=tensors["X_nf"],
            coords=tensors["coords"],
            edge_index=tensors["edge_index"],
            neighbor_input=neighbor_input,
            time_onehot=time_input,
        )

        train = masks["train"]

        loss_region = F.cross_entropy(
            out["region_logits"][train],
            tensors["y_region"][train],
        )

        if has_time and "time_logits" in out:
            loss_time = F.cross_entropy(
                out["time_logits"][train],
                tensors["y_time"][train],
            )
        else:
            loss_time = out["region_logits"].new_tensor(0.0)

        if "neighbor_pred" in out and Y_neighbor.shape[1] > 0:
            loss_neighbor = F.smooth_l1_loss(
                out["neighbor_pred"][train],
                tensors["Y_neighbor"][train],
            )
        else:
            loss_neighbor = out["region_logits"].new_tensor(0.0)

        loss_contrast = supervised_contrastive_loss(
            out["z_norm"],
            tensors["contrast_labels"],
            train,
            temperature=0.2,
            max_samples=2048,
        )

        loss_repair = F.mse_loss(
            out["repair_score"][train],
            tensors["repair_target"][train],
        )

        loss = (
            args.w_region * loss_region
            + args.w_time * loss_time
            + args.w_neighbor * loss_neighbor
            + args.w_contrastive * loss_contrast
            + args.w_repair * loss_repair
        )

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        if epoch % 5 == 0 or epoch == 1:
            metrics, _ = evaluate(
                model,
                tensors,
                masks,
                meta,
                device,
                use_neighbor_input=model.use_neighbor_input,
                use_time_input=model.use_time_input,
            )

            val_macro = metrics.get("val_region_macro_f1", 0.0)

            row = {
                "epoch": epoch,
                "loss_total": float(loss.detach().cpu()),
                "loss_region": float(loss_region.detach().cpu()),
                "loss_time": float(loss_time.detach().cpu()),
                "loss_neighbor": float(loss_neighbor.detach().cpu()),
                "loss_contrastive": float(loss_contrast.detach().cpu()),
                "loss_repair": float(loss_repair.detach().cpu()),
                **metrics,
            }
            history.append(row)

            print(
                f"Epoch {epoch:04d} | "
                f"loss={row['loss_total']:.4f} | "
                f"val_region_macro_f1={val_macro:.4f} | "
                f"test_region_macro_f1={metrics.get('test_region_macro_f1', np.nan):.4f}"
            )

            if val_macro > best_val:
                best_val = val_macro
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 5

            if bad >= args.patience:
                print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}, best val macro-F1: {best_val:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics, arrays = evaluate(
        model,
        tensors,
        masks,
        meta,
        device,
        use_neighbor_input=model.use_neighbor_input,
        use_time_input=model.use_time_input,
    )

    # Save model
    model_path = CKPT_DIR / "strokeniche_adapter_best.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "meta": meta,
        "best_epoch": best_epoch,
        "best_val_region_macro_f1": best_val,
    }, model_path)

    # Save history and metrics
    hist = pd.DataFrame(history)
    hist.to_csv(DOWN / "strokeniche_adapter_training_history.csv", index=False)

    final_df = pd.DataFrame([{
        "method": "StrokeNiche_Adapter",
        "best_epoch": best_epoch,
        "best_val_region_macro_f1": best_val,
        **final_metrics,
        "note": (
            "Lightweight stroke-specific spatial virtual-cell adapter trained on frozen Nicheformer embeddings."
        ),
    }])
    final_df.to_csv(DOWN / "strokeniche_adapter_final_metrics.csv", index=False)

    # Save embeddings and predictions
    np.save(OUTDIR / "strokeniche_adapter_latent_z.npy", arrays["z"])
    np.save(OUTDIR / "strokeniche_adapter_repair_score.npy", arrays["repair_score"])
    np.save(OUTDIR / "strokeniche_adapter_region_proba.npy", arrays["region_proba"])

    pred = pd.DataFrame({
        "obs_name": data["obs_names"].astype(str),
        "region_true": data["region_raw"].astype(str),
        "dominant_celltype": data["dominant_celltype"].astype(str),
        "repair_target": repair_target,
        "repair_score": arrays["repair_score"],
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    })

    inv_region = {v: k for k, v in meta["region_map"].items()}
    region_pred_idx = arrays["region_proba"].argmax(axis=1)
    pred["region_pred"] = [inv_region.get(int(i), str(i)) for i in region_pred_idx]

    for name, idx in meta["region_map"].items():
        pred[f"prob_{name}"] = arrays["region_proba"][:, idx]

    if has_time:
        pred["timepoint"] = data["time_raw"].astype(str)

    pred.to_csv(DOWN / "strokeniche_adapter_predictions.csv", index=False)

    # Save confusion matrices and reports
    for split in ["train", "val", "test"]:
        if f"{split}_region_cm" in arrays:
            cm = pd.DataFrame(arrays[f"{split}_region_cm"])
            cm.to_csv(DOWN / f"strokeniche_adapter_{split}_region_confusion_matrix.csv", index=False)

        if f"{split}_region_report" in arrays:
            arrays[f"{split}_region_report"].to_csv(DOWN / f"strokeniche_adapter_{split}_region_classification_report.csv")

        if f"{split}_neighbor_by_target" in arrays:
            arrays[f"{split}_neighbor_by_target"].to_csv(DOWN / f"strokeniche_adapter_{split}_neighbor_by_target.csv", index=False)

    print("\nFinal metrics:")
    print(final_df.T)
    print("\nSaved model:", model_path)
    print("Saved metrics:", DOWN / "strokeniche_adapter_final_metrics.csv")
    print("Saved predictions:", DOWN / "strokeniche_adapter_predictions.csv")
    print("Saved latent:", OUTDIR / "strokeniche_adapter_latent_z.npy")


if __name__ == "__main__":
    main()
