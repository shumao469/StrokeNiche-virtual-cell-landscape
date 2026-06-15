#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step76B | Force true latent obsm for StrokeNiche pseudo-energy landscape

This wrapper imports Step76, patches infer_latent_coordinates(),
and forces --obsm_key to be used before meta columns such as latent_x/latent_y.

Why:
Original Step76 may select latent_x/latent_y from state_table first.
If those columns are spatial-like or graph-layout-like, the figure can look like tissue coordinates.
Step76B forces true h5ad obsm embeddings, e.g. X_nicheformer or X_scgpt.
"""

import sys
import importlib.util
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


STEP76_SCRIPT = "/mnt/h/vir/ST/76_latent_state_potential_landscape.py"


spec = importlib.util.spec_from_file_location("step76_mod", STEP76_SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

_original_infer = mod.infer_latent_coordinates


def infer_latent_coordinates_force_obsm(adata, meta, xcol="", ycol="", obsm_key=""):
    """
    New priority:
    1. explicit user xcol/ycol, if provided
    2. explicit obsm_key, if provided
    3. original Step76 fallback logic
    """

    # 1. Explicit user columns still allowed.
    if xcol and ycol and xcol in meta.columns and ycol in meta.columns:
        x = pd.to_numeric(meta[xcol], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(meta[ycol], errors="coerce").to_numpy(dtype=float)
        return np.column_stack([x, y]), {
            "mode": "meta_user",
            "xcol": xcol,
            "ycol": ycol,
            "forced_obsm_priority": True,
        }

    # 2. Force obsm_key before any meta_auto latent_x/latent_y.
    if obsm_key:
        if obsm_key not in adata.obsm:
            raise ValueError(
                f"--obsm_key {obsm_key} not found. Available obsm keys: {list(adata.obsm.keys())}"
            )

        arr = np.asarray(adata.obsm[obsm_key])
        if arr.ndim != 2 or arr.shape[0] != adata.n_obs or arr.shape[1] < 2:
            raise ValueError(
                f"Invalid obsm[{obsm_key}] shape: {arr.shape}; expected n_obs × >=2."
            )

        arr = arr.astype(float)

        if arr.shape[1] == 2:
            return arr[:, :2], {
                "mode": "obsm_forced_2d",
                "obsm_key": obsm_key,
                "n_dim": int(arr.shape[1]),
                "forced_obsm_priority": True,
            }

        Z = PCA(n_components=2, random_state=0).fit_transform(arr)
        return Z, {
            "mode": "obsm_forced_pca",
            "obsm_key": obsm_key,
            "n_dim": int(arr.shape[1]),
            "forced_obsm_priority": True,
        }

    # 3. Fall back to original Step76 only if no obsm_key is provided.
    return _original_infer(adata, meta, xcol=xcol, ycol=ycol, obsm_key="")


mod.infer_latent_coordinates = infer_latent_coordinates_force_obsm

if __name__ == "__main__":
    mod.main()
