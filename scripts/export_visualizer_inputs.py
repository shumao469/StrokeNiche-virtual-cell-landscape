#!/usr/bin/env python3
"""Export compact visualizer inputs from a Step78D per-cell response table."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from strokeniche_vc.io import normalize_cell_table, normalize_effect_table  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-table", required=True)
    parser.add_argument("--cells-out", required=True)
    parser.add_argument("--effects-out", required=True)
    parser.add_argument(
        "--context-table",
        default=None,
        help=(
            "Optional audited Step64 context CSV. When provided, valid "
            "prob_lesion_core, repair_score, state, timepoint, and spatial columns "
            "replace the legacy Step78 baseline fields."
        ),
    )
    parser.add_argument(
        "--allow-legacy-baseline",
        action="store_true",
        help="Explicitly retain Step78 baseline fields when no audited context table is available.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional JSON path; defaults to visualizer_manifest.json beside cells-out.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    args = parser.parse_args()
    if not args.context_table and not args.allow_legacy_baseline:
        parser.error(
            "--context-table is required by default because legacy core/peri/remote "
            "probability columns may be degenerate. Use --allow-legacy-baseline only "
            "for an explicitly audited exception."
        )

    source = Path(args.response_table).resolve()
    context_source = Path(args.context_table).resolve() if args.context_table else None
    cells_out = Path(args.cells_out).resolve()
    effects_out = Path(args.effects_out).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else (cells_out.parent / "visualizer_manifest.json").resolve()
    )
    input_paths = {source, *([context_source] if context_source is not None else [])}
    output_paths = [cells_out, effects_out, manifest_path]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("cells, effects, and manifest outputs must be distinct")
    collisions = [path for path in output_paths if path in input_paths]
    if collisions:
        raise ValueError(f"Output path collides with an input: {collisions}")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        raise FileExistsError(f"Outputs already exist; use --force to replace them: {existing}")

    table = pd.read_csv(
        source,
        dtype={"obs_name": "string", "perturbation": "string"},
        low_memory=False,
    )
    required = {
        "obs_name", "latent1", "latent2", "perturbation",
        "baseline_core", "baseline_repair", "candidate_mean_delta_core",
        "candidate_mean_delta_repair",
    }
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Step78D table is missing columns: {missing}")
    if table.empty:
        raise ValueError("Step78D table must contain at least one data row")
    if table["obs_name"].isna().any() or table["perturbation"].isna().any():
        raise ValueError("obs_name and perturbation must not be missing")
    table["obs_name"] = table["obs_name"].astype(str).str.strip()
    table["perturbation"] = table["perturbation"].astype(str).str.strip()
    if table["obs_name"].eq("").any() or table["perturbation"].eq("").any():
        raise ValueError("obs_name and perturbation must not be blank")
    duplicate_pairs = table.duplicated(["perturbation", "obs_name"], keep=False)
    if duplicate_pairs.any():
        examples = table.loc[duplicate_pairs, ["perturbation", "obs_name"]].head(10)
        raise ValueError(
            "Step78D table must contain exactly one row per perturbation and observation; "
            f"duplicate examples: {examples.to_dict('records')}"
        )
    coverage = table.groupby("perturbation", sort=False)["obs_name"].agg(frozenset)
    reference_coverage = coverage.iloc[0]
    incomplete = [
        perturbation
        for perturbation, observations in coverage.items()
        if observations != reference_coverage
    ]
    if incomplete:
        raise ValueError(
            "Every perturbation must cover the identical observation set; "
            f"mismatched perturbations: {incomplete[:10]}"
        )
    effect_value_columns = [
        "candidate_mean_delta_core",
        "candidate_mean_delta_repair",
        *(["short_label"] if "short_label" in table.columns else []),
    ]
    variation = table.groupby("perturbation")[effect_value_columns].nunique(dropna=False)
    if (variation > 1).any(axis=None):
        bad_columns = variation.columns[(variation > 1).any(axis=0)].tolist()
        raise ValueError(
            "Candidate effect values and labels are not constant within perturbation: "
            f"{bad_columns}"
        )

    optional_cell = [c for c in ["state", "timepoint", "spatial_x", "spatial_y"] if c in table.columns]
    cell_cols = ["obs_name", "latent1", "latent2", "baseline_core", "baseline_repair", *optional_cell]
    cell_variation = table.groupby("obs_name")[cell_cols[1:]].nunique(dropna=False)
    if (cell_variation > 1).any(axis=None):
        bad_columns = cell_variation.columns[(cell_variation > 1).any(axis=0)].tolist()
        raise ValueError(f"Conflicting baseline values across perturbations: {bad_columns}")
    cells = table[cell_cols].drop_duplicates("obs_name")
    if context_source is not None:
        wanted = {
            "obs_name", "prob_lesion_core", "repair_score", "state_group",
            "timepoint", "dominant_celltype", "spatial_x", "spatial_y",
        }
        context = pd.read_csv(
            context_source,
            usecols=lambda column: column in wanted,
            dtype={"obs_name": "string"},
            low_memory=False,
        )
        required_context = {"obs_name", "prob_lesion_core", "repair_score"}
        missing_context = sorted(required_context.difference(context.columns))
        if missing_context:
            raise ValueError(f"Context table is missing columns: {missing_context}")
        if context["obs_name"].isna().any():
            raise ValueError("Context obs_name must not be missing")
        context["obs_name"] = context["obs_name"].astype(str).str.strip()
        if context["obs_name"].eq("").any():
            raise ValueError("Context obs_name must not be blank")
        context_value_cols = [column for column in context.columns if column != "obs_name"]
        context_variation = context.groupby("obs_name")[context_value_cols].nunique(dropna=False)
        if (context_variation > 1).any(axis=None):
            raise ValueError("Context table has conflicting duplicate observation rows")
        context = context.drop_duplicates("obs_name").rename(
            columns={
                "prob_lesion_core": "baseline_core",
                "repair_score": "baseline_repair",
                "state_group": "state",
            }
        )
        replacement = [
            column for column in
            ["baseline_core", "baseline_repair", "state", "timepoint", "dominant_celltype", "spatial_x", "spatial_y"]
            if column in context.columns
        ]
        cells = cells.drop(columns=[column for column in replacement if column in cells.columns])
        cells = cells.merge(context[["obs_name", *replacement]], on="obs_name", how="left", validate="one_to_one")
        if cells[["baseline_core", "baseline_repair"]].isna().any(axis=None):
            raise ValueError("Context table does not cover every Step78D observation")
    cells = normalize_cell_table(cells)
    effects = (
        table[
            [
                "perturbation", "candidate_mean_delta_core", "candidate_mean_delta_repair",
                *(["short_label"] if "short_label" in table.columns else []),
            ]
        ]
        .drop_duplicates("perturbation")
        .rename(
            columns={
                "candidate_mean_delta_core": "delta_core",
                "candidate_mean_delta_repair": "delta_repair",
            }
        )
    )
    effects["modality"] = "gene"
    effects["target_genes"] = effects.get("short_label", "")
    effects["source"] = "Step78D heuristic visualization proxy"
    effects = effects[["perturbation", "delta_core", "delta_repair", "modality", "target_genes", "source"]]
    effects = normalize_effect_table(effects)

    cells_out.parent.mkdir(parents=True, exist_ok=True)
    effects_out.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(cells_out, index=False)
    effects.to_csv(effects_out, index=False)
    manifest = {
        "schema_version": 1,
        "interpretation": "Step78D smoothed heuristic proxy; not a causal or wet-lab perturbation result",
        "source": {"file": source.name, "sha256": sha256(source)},
        "outputs": {
            "cells": {"file": cells_out.name, "rows": len(cells), "sha256": sha256(cells_out)},
            "effects": {"file": effects_out.name, "rows": len(effects), "sha256": sha256(effects_out)},
        },
        "coverage": {
            "policy": "identical observation set for every perturbation",
            "perturbations": len(effects),
            "observations_per_perturbation": len(cells),
        },
    }
    if context_source is not None:
        manifest["context"] = {
            "file": context_source.name,
            "sha256": sha256(context_source),
            "core_probability_column": "prob_lesion_core",
        }
    else:
        manifest["warning"] = (
            "No audited context table supplied; Step78D baseline fields were retained. "
            "Do not use a legacy table where core/peri/remote probability columns are identical."
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"cells={len(cells)} -> {cells_out}")
    print(f"effects={len(effects)} -> {effects_out}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
