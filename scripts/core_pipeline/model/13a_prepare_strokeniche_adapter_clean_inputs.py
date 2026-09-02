#!/usr/bin/env python3
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/mnt/h/vir/ST")
IN_H5AD = PROJECT / "results/step5_nicheformer/spatial_all_with_nicheformer.h5ad"

OUTDIR = PROJECT / "results/step7_strokeniche_adapter_clean"
OUTDIR.mkdir(parents=True, exist_ok=True)

REGION_CANDIDATES = [
    "region_auto",
    "region_label",
    "region",
    "lesion_region",
    "spatial_region",
]

TIME_CANDIDATES = [
    "timepoint",
    "Timepoint",
    "time_point",
    "day",
    "Day",
    "sample_time",
    "Sample",
    "sample",
    "orig.ident",
]

# Clean 8-neighborhood composition targets
# Neurons = NeuronsGABA + NeuronsGLUT
CLEAN_NEIGHBOR_SPECS = {
    "Microglia": ["prediction.score.Microglia"],
    "Astrocytes": ["prediction.score.Astrocytes"],
    "EndothelialCells": ["prediction.score.EndothelialCells"],
    "OLs": ["prediction.score.OLs"],
    "OPCs": ["prediction.score.OPCs"],
    "VLMCs": ["prediction.score.VLMCs"],
    "EpendymalCells": ["prediction.score.EpendymalCells"],
    "Neurons": [
        "prediction.score.NeuronsGABA",
        "prediction.score.NeuronsGLUT",
    ],
}


def pick_obsm_key(adata):
    for key in ["X_nicheformer", "nicheformer", "X_nf", "Nicheformer"]:
        if key in adata.obsm:
            return key
    for key in adata.obsm.keys():
        if "niche" in key.lower():
            return key
    raise KeyError(f"No Nicheformer embedding found. obsm keys: {list(adata.obsm.keys())}")


def pick_col(obs, candidates):
    for c in candidates:
        if c in obs.columns:
            return c
    return None


def parse_timepoint_from_value(x):
    s = str(x)
    m = re.search(r"D\s*([137])", s, flags=re.IGNORECASE)
    if m:
        return f"D{m.group(1)}"
    m = re.search(r"day\s*([137])", s, flags=re.IGNORECASE)
    if m:
        return f"D{m.group(1)}"
    return s


def get_coords(adata):
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            return coords[:, :2], "obsm['spatial']"

    candidates = [
        ("array_row", "array_col"),
        ("pxl_row_in_fullres", "pxl_col_in_fullres"),
        ("spatial_x", "spatial_y"),
        ("x", "y"),
        ("X", "Y"),
    ]

    for a, b in candidates:
        if a in adata.obs.columns and b in adata.obs.columns:
            coords = adata.obs[[a, b]].values.astype(np.float32)
            return coords, f"obs[['{a}', '{b}']]"

    raise KeyError("No spatial coordinates found.")


def build_clean_neighbor_matrix(adata):
    obs = adata.obs
    missing = []
    used_source_cols = []

    mats = []
    clean_names = []

    for clean_name, source_cols in CLEAN_NEIGHBOR_SPECS.items():
        vec = np.zeros(adata.n_obs, dtype=np.float32)

        for c in source_cols:
            if c not in obs.columns:
                missing.append(c)
                continue

            vals = pd.to_numeric(obs[c], errors="coerce").fillna(0).values.astype(np.float32)
            vals = np.nan_to_num(vals)
            vals[vals < 0] = 0
            vec += vals
            used_source_cols.append(c)

        mats.append(vec)
        clean_names.append(clean_name)

    if len(mats) == 0:
        raise ValueError("No clean prediction.score.* neighbor columns were found.")

    Y = np.vstack(mats).T.astype(np.float32)

    # Row-normalize to composition-like targets.
    row_sum = Y.sum(axis=1, keepdims=True)
    ok = row_sum[:, 0] > 0

    Y_norm = Y.copy()
    Y_norm[ok] = Y_norm[ok] / row_sum[ok]

    # If a row is all zero, keep zeros.
    Y_norm = np.nan_to_num(Y_norm).astype(np.float32)

    info = {
        "clean_neighbor_names": clean_names,
        "used_source_cols": sorted(set(used_source_cols)),
        "missing_source_cols": sorted(set(missing)),
        "note": "Neurons = prediction.score.NeuronsGABA + prediction.score.NeuronsGLUT; all 8 targets row-normalized.",
    }

    return Y_norm, clean_names, info


def build_knn_edges(coords_scaled, k=12):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(coords_scaled)
    _, idx = nbrs.kneighbors(coords_scaled)

    edges = []
    n = coords_scaled.shape[0]

    for i in range(n):
        for j in idx[i, 1:]:
            edges.append((j, i))  # source j -> target i
        edges.append((i, i))      # self-loop

    edge_index = np.array(edges, dtype=np.int64).T
    return edge_index


def encode_labels(values, preferred_order=None):
    values = pd.Series(values).astype(str).fillna("NA").values

    if preferred_order is not None:
        classes = [c for c in preferred_order if c in set(values)]
        classes += sorted([c for c in set(values) if c not in classes])
    else:
        classes = sorted(pd.unique(values))

    mapping = {c: i for i, c in enumerate(classes)}
    y = np.array([mapping[v] for v in values], dtype=np.int64)
    return y, mapping


def make_splits(region_labels, time_labels=None, test_size=0.15, val_size=0.15, random_state=42):
    n = len(region_labels)

    strat = region_labels.astype(str)

    if time_labels is not None:
        combined = np.array([f"{r}|{t}" for r, t in zip(region_labels, time_labels)])
        counts = pd.Series(combined).value_counts()
        if counts.min() >= 3:
            strat = combined

    idx = np.arange(n)

    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )

    strat_train_val = strat[train_val_idx]

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=strat_train_val,
    )

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def main():
    print("Step 7b: prepare clean-neighborhood StrokeNiche Adapter inputs")
    print("Reading:", IN_H5AD)

    adata = ad.read_h5ad(IN_H5AD)
    print(adata)

    emb_key = pick_obsm_key(adata)
    X_nf = np.asarray(adata.obsm[emb_key], dtype=np.float32)

    print("Nicheformer embedding key:", emb_key)
    print("X_nf:", X_nf.shape, "finite:", np.isfinite(X_nf).all())

    coords, coords_source = get_coords(adata)
    print("Coordinates:", coords.shape, "source:", coords_source)

    region_col = pick_col(adata.obs, REGION_CANDIDATES)
    if region_col is None:
        raise KeyError(f"No region label column found. Tried: {REGION_CANDIDATES}")

    region_raw = adata.obs[region_col].astype(str).values
    y_region, region_map = encode_labels(
        region_raw,
        preferred_order=["lesion_core", "peri_infarct", "remote_like"],
    )

    print("Region column:", region_col)
    print(pd.Series(region_raw).value_counts())

    time_col = pick_col(adata.obs, TIME_CANDIDATES)
    time_raw = None
    y_time = None
    time_map = None

    if time_col is not None:
        parsed = [parse_timepoint_from_value(x) for x in adata.obs[time_col].astype(str).values]
        time_raw = np.array(parsed, dtype=str)
        y_time, time_map = encode_labels(
            time_raw,
            preferred_order=["D1", "D3", "D7"],
        )
        print("Time column:", time_col)
        print(pd.Series(time_raw).value_counts())
    else:
        print("No timepoint column found. L_time will be disabled.")

    print("\nBuilding clean 8-class neighborhood targets...")
    Y_neighbor, neighbor_cols, neighbor_info = build_clean_neighbor_matrix(adata)

    print("Clean neighbor targets:", neighbor_cols)
    print("Y_neighbor:", Y_neighbor.shape)
    print("Used source columns:")
    for c in neighbor_info["used_source_cols"]:
        print("  ", c)

    if neighbor_info["missing_source_cols"]:
        print("Missing source columns:")
        for c in neighbor_info["missing_source_cols"]:
            print("  ", c)

    dominant_idx = Y_neighbor.argmax(axis=1)
    dominant_celltype = np.array([neighbor_cols[i] for i in dominant_idx], dtype=str)

    print("\nDominant clean celltype:")
    print(pd.Series(dominant_celltype).value_counts())

    # Directional repair target:
    # lesion_core = 0, peri_infarct = 0.5, remote_like = 1
    repair_target = np.zeros(adata.n_obs, dtype=np.float32)
    for i, r in enumerate(region_raw):
        if r == "lesion_core":
            repair_target[i] = 0.0
        elif r == "peri_infarct":
            repair_target[i] = 0.5
        elif r == "remote_like":
            repair_target[i] = 1.0
        else:
            repair_target[i] = 0.5

    coord_scaler = StandardScaler()
    coords_scaled = coord_scaler.fit_transform(coords).astype(np.float32)

    print("\nBuilding spatial kNN graph...")
    edge_index = build_knn_edges(coords_scaled, k=12)
    print("edge_index:", edge_index.shape)

    train_mask, val_mask, test_mask = make_splits(
        region_labels=region_raw,
        time_labels=time_raw,
        test_size=0.15,
        val_size=0.15,
        random_state=42,
    )

    print("\nSplit sizes:")
    print("train:", int(train_mask.sum()))
    print("val:", int(val_mask.sum()))
    print("test:", int(test_mask.sum()))

    out_npz = OUTDIR / "strokeniche_adapter_inputs.npz"
    np.savez_compressed(
        out_npz,
        X_nf=X_nf,
        coords=coords.astype(np.float32),
        coords_scaled=coords_scaled,
        Y_neighbor=Y_neighbor,
        y_region=y_region,
        y_time=np.array([] if y_time is None else y_time, dtype=np.int64),
        has_time=np.array([y_time is not None], dtype=bool),
        repair_target=repair_target,
        edge_index=edge_index,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        dominant_celltype=dominant_celltype,
        obs_names=adata.obs_names.astype(str).values,
        region_raw=region_raw,
        time_raw=np.array([] if time_raw is None else time_raw, dtype=str),
    )

    meta = {
        "version": "Step7b_clean_neighborhood",
        "input_h5ad": str(IN_H5AD),
        "embedding_key": emb_key,
        "coords_source": coords_source,
        "region_col": region_col,
        "time_col": time_col,
        "neighbor_cols": neighbor_cols,
        "neighbor_target_definition": neighbor_info,
        "region_map": region_map,
        "time_map": time_map,
        "n_obs": int(adata.n_obs),
        "n_nf_features": int(X_nf.shape[1]),
        "n_neighbor_targets": int(Y_neighbor.shape[1]),
        "note": (
            "Clean-neighborhood StrokeNiche Adapter. "
            "Only prediction.score.* cell-type scores are used. "
            "NeuronsGABA and NeuronsGLUT are merged into Neurons. "
            "Non-composition variables such as spatial_imagecol, neuronal_score, and neuronal_z are excluded."
        ),
    }

    out_json = OUTDIR / "strokeniche_adapter_input_metadata.json"
    out_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    summary = pd.DataFrame({
        "obs_name": adata.obs_names.astype(str),
        "region": region_raw,
        "dominant_celltype": dominant_celltype,
        "repair_target": repair_target,
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
    })

    if time_raw is not None:
        summary["timepoint"] = time_raw

    for j, c in enumerate(neighbor_cols):
        summary[f"neighbor_{c}"] = Y_neighbor[:, j]

    summary.to_csv(OUTDIR / "strokeniche_adapter_input_obs_summary.csv", index=False)

    print("\nSaved:")
    print(out_npz)
    print(out_json)
    print(OUTDIR / "strokeniche_adapter_input_obs_summary.csv")


if __name__ == "__main__":
    main()
