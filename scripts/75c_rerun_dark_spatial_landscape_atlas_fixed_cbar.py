#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step75C | Re-run Step75B with colorbar-safe layout

Purpose
-------
Fix the Step75B issue where the right-side colorbar overlaps the last column
of the spatial marker/module/state atlas.

This script imports 75b_dark_spatial_landscape_atlas.py, overrides add_cbar()
with a safer fixed-position colorbar, then reruns the same figure builders.

Input
-----
Same as Step75B.

Output
------
Colorbar-safe versions:
  Fig_Step75C_DarkSpatialMarkerLandscapeAtlas.*
  Fig_Step75C_DarkSpatialModuleLandscapeAtlas.*
  Fig_Step75C_DarkSpatialStateProbabilityLandscapeAtlas.*
  Fig_Step75C_CompactSpatialLandscapeManuscriptPanel.*
"""

from pathlib import Path
import argparse
import json
import importlib.util
import warnings

import numpy as np
import pandas as pd
import anndata as ad

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


def load_step75b_module(script_path):
    spec = importlib.util.spec_from_file_location("step75b_mod", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_cbar_safe(fig, axes, cmap, norm, label, color="white"):
    """
    Colorbar-safe replacement.

    The original Step75B placed colorbar based on axes positions before final
    subplot adjustment. Here we use a fixed right-side colorbar position so the
    last panel is never covered.
    """
    axes = [ax for ax in axes if ax is not None]
    boxes = [ax.get_position() for ax in axes]

    if not boxes:
        return None

    y0 = min(b.y0 for b in boxes)
    y1 = max(b.y1 for b in boxes)

    # Fixed external position.
    # Axes usually end near 0.90–0.92; colorbar starts at 0.955.
    cbar_x = 0.955
    cbar_w = 0.012

    # Slight vertical shrink for better appearance.
    ypad = 0.025
    cax = fig.add_axes([cbar_x, y0 + ypad, cbar_w, max(0.05, y1 - y0 - 2 * ypad)])

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(colors=color, labelsize=7)
    cb.outline.set_edgecolor(color)
    cb.set_label(label, color=color, fontsize=8)
    return cb


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--step75b_script", default="/mnt/h/vir/ST/75b_dark_spatial_landscape_atlas.py")
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--state_table", required=True)
    parser.add_argument("--outdir", required=True)

    parser.add_argument("--marker_json", default="")
    parser.add_argument("--module_json", default="")

    parser.add_argument("--adata_id_col", default="")
    parser.add_argument("--state_id_col", default="")
    parser.add_argument("--xcol", default="")
    parser.add_argument("--ycol", default="")

    parser.add_argument("--core_col", default="core_probability")
    parser.add_argument("--peri_col", default="peri_probability")
    parser.add_argument("--remote_col", default="remote_probability")

    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=600)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Step75C | Dark spatial landscape atlas with fixed colorbar")
    print("=" * 100)

    mod = load_step75b_module(args.step75b_script)

    # Monkey patch Step75B colorbar function.
    mod.add_cbar = add_cbar_safe

    adata = ad.read_h5ad(args.h5ad)
    obs = adata.obs.copy()

    adata_id, adata_id_col_used = mod.get_obs_id_series(adata, obs, user_col=args.adata_id_col)
    obs["_adata_id_"] = adata_id.astype(str).values

    x, y, xcol_used, ycol_used, xy_mode = mod.get_xy(adata, obs, xcol=args.xcol, ycol=args.ycol)
    obs["_spatial_x_"] = x
    obs["_spatial_y_"] = y

    state_df = mod.read_table(args.state_table)
    meta, merge_audit = mod.merge_state_table(
        obs,
        state_df,
        adata_id_col="_adata_id_",
        user_state_id_col=args.state_id_col
    )

    prob_cols = mod.infer_prob_cols(
        meta,
        core_col=args.core_col,
        peri_col=args.peri_col,
        remote_col=args.remote_col
    )

    marker_groups = mod.read_json(args.marker_json) if args.marker_json else None
    if marker_groups is None:
        marker_groups = mod.DEFAULT_MARKER_GROUPS

    module_sets = mod.read_json(args.module_json) if args.module_json else None
    if module_sets is None:
        module_sets = mod.DEFAULT_MODULE_GENESETS

    gmap = mod.gene_lookup(adata)
    module_scores, module_audit = mod.compute_module_scores(adata, module_sets, gmap, min_genes=2)

    x = pd.to_numeric(meta["_spatial_x_"], errors="coerce").to_numpy()
    y = pd.to_numeric(meta["_spatial_y_"], errors="coerce").to_numpy()

    audit = {
        "status": "ok",
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "adata_id_col_used": adata_id_col_used,
        "xy_mode": xy_mode,
        "xcol_used": xcol_used,
        "ycol_used": ycol_used,
        "state_merge_audit": merge_audit,
        "prob_cols": prob_cols,
        "layout_fix": "fixed external colorbar at x=0.955",
    }

    (outdir / "step75c_audit_report.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    module_audit.to_csv(outdir / "step75c_module_gene_match_audit.csv", index=False)

    print("---- audit ----")
    print(json.dumps(audit, indent=2, ensure_ascii=False))

    # Re-render all Step75B panels with corrected colorbar.
    mod.make_marker_atlas(
        meta=meta,
        adata=adata,
        x=x,
        y=y,
        marker_groups=marker_groups,
        gmap=gmap,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75C_DarkSpatialMarkerLandscapeAtlas"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    mod.make_module_atlas(
        meta=meta,
        x=x,
        y=y,
        module_scores=module_scores,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75C_DarkSpatialModuleLandscapeAtlas"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    mod.make_state_atlas(
        meta=meta,
        x=x,
        y=y,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75C_DarkSpatialStateProbabilityLandscapeAtlas"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    mod.make_compact_panel(
        meta=meta,
        adata=adata,
        x=x,
        y=y,
        marker_groups=marker_groups,
        module_scores=module_scores,
        gmap=gmap,
        prob_cols=prob_cols,
        outbase=str(outdir / "Fig_Step75C_CompactSpatialLandscapeManuscriptPanel"),
        point_size=args.point_size,
        dpi=args.dpi
    )

    report = {
        "status": "ok",
        "outputs": {
            "marker_atlas": str(outdir / "Fig_Step75C_DarkSpatialMarkerLandscapeAtlas.png"),
            "module_atlas": str(outdir / "Fig_Step75C_DarkSpatialModuleLandscapeAtlas.png"),
            "state_probability_atlas": str(outdir / "Fig_Step75C_DarkSpatialStateProbabilityLandscapeAtlas.png"),
            "compact_panel": str(outdir / "Fig_Step75C_CompactSpatialLandscapeManuscriptPanel.png"),
            "audit_json": str(outdir / "step75c_audit_report.json"),
        },
        "interpretation_note": (
            "Step75C fixes the colorbar-overlap issue in Step75B. "
            "State metadata are successfully merged when state_merge_overlap equals n_obs."
        )
    }

    (outdir / "step75c_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    (outdir / "step75c_report.txt").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("=" * 100)
    print("DONE Step75C")
    print("=" * 100)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
