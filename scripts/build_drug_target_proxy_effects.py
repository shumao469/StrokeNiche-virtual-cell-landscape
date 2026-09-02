#!/usr/bin/env python3
"""Map reviewed drugs to gene effects without claiming drug-specific simulation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from strokeniche_vc import __version__  # noqa: E402
from strokeniche_vc.io import normalize_effect_table  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-effects", required=True)
    parser.add_argument("--drug-bridge", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default=None, help="Optional existing bundle manifest to update")
    parser.add_argument("--force", action="store_true", help="Replace an existing combined-effects output")
    parser.add_argument(
        "--drug-name-col",
        default="drug_name",
    )
    parser.add_argument(
        "--target-col",
        default="neighbor_bridge_matched_target_gene",
    )
    args = parser.parse_args()

    gene_effects_path = Path(args.gene_effects).resolve()
    drug_bridge_path = Path(args.drug_bridge).resolve()
    mapping_path = Path(args.mapping).resolve()
    output = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve() if args.manifest else None
    input_paths = {gene_effects_path, drug_bridge_path, mapping_path}
    if output in input_paths or (manifest_path is not None and manifest_path in input_paths | {output}):
        raise ValueError("Input, output, and manifest paths must be distinct")
    if output.exists() and not args.force:
        raise FileExistsError(f"Output already exists; use --force to replace it: {output}")

    manifest: dict[str, object] | None = None
    if manifest_path is not None:
        if manifest_path.exists():
            try:
                loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read a valid JSON manifest: {manifest_path.name}") from exc
            if not isinstance(loaded_manifest, dict):
                raise ValueError("Existing manifest must contain a JSON object")
            manifest = loaded_manifest
        else:
            manifest = {}
        outputs = manifest.get("outputs")
        if outputs is None:
            manifest["outputs"] = {}
        elif not isinstance(outputs, dict):
            raise ValueError("Existing manifest outputs field must contain a JSON object")

    # Identifiers are semantic strings.  Numeric effect columns are converted
    # explicitly by normalize_effect_table after IDs have been preserved.
    effects = normalize_effect_table(pd.read_csv(gene_effects_path, dtype=str))
    bridge = pd.read_csv(drug_bridge_path, dtype=str)
    mapping = pd.read_csv(mapping_path, dtype=str)
    effect_required = {"perturbation", "delta_core", "delta_repair"}
    bridge_required = {args.drug_name_col, args.target_col}
    mapping_required = {"target_gene", "gene_perturbation"}
    for label, table, required in [
        ("gene effects", effects, effect_required),
        ("drug bridge", bridge, bridge_required),
        ("mapping", mapping, mapping_required),
    ]:
        missing = sorted(required.difference(table.columns))
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")

    for label, table, columns in [
        ("drug bridge", bridge, [args.drug_name_col, args.target_col]),
        ("mapping", mapping, ["target_gene", "gene_perturbation"]),
    ]:
        if table[columns].isna().any(axis=None):
            raise ValueError(f"{label} contains missing identifiers")
        for column in columns:
            table[column] = table[column].astype(str).str.strip()
            if table[column].eq("").any():
                raise ValueError(f"{label} contains blank {column} values")

    mapping = mapping.copy()
    mapping["target_key"] = mapping["target_gene"].astype(str).str.casefold()
    mapping_conflicts = mapping.groupby("target_key")["gene_perturbation"].nunique()
    mapping_conflicts = mapping_conflicts[mapping_conflicts > 1].index.tolist()
    if mapping_conflicts:
        raise ValueError(f"Targets map to conflicting perturbations: {mapping_conflicts}")
    mapping = mapping.drop_duplicates(["target_key", "gene_perturbation"])
    bridge = bridge.copy()
    bridge["target_key"] = bridge[args.target_col].astype(str).str.casefold()
    merged = bridge.merge(mapping, on="target_key", how="left", validate="many_to_one", indicator=True)
    unmatched = merged.loc[merged["_merge"] != "both", args.target_col].astype(str).unique().tolist()
    if unmatched:
        raise ValueError(f"Reviewed drug targets are absent from the mapping table: {unmatched}")
    merged = merged.drop(columns="_merge")
    effect_lookup = effects[["perturbation", "delta_core", "delta_repair"]].rename(
        columns={"perturbation": "gene_perturbation"}
    )
    merged = merged.merge(effect_lookup, on="gene_perturbation", how="left", validate="many_to_one")
    if merged[["delta_core", "delta_repair"]].isna().any(axis=None):
        missing_effects = merged.loc[
            merged["delta_core"].isna() | merged["delta_repair"].isna(),
            "gene_perturbation",
        ].unique().tolist()
        raise ValueError(f"Mapped gene perturbations are absent from effect table: {missing_effects}")
    if merged.empty:
        raise ValueError("No reviewed drug targets matched the mapping table")
    per_drug_targets = merged.groupby(args.drug_name_col)["gene_perturbation"].nunique()
    ambiguous = per_drug_targets[per_drug_targets > 1].index.astype(str).tolist()
    if ambiguous:
        raise ValueError(
            "Drugs map to multiple perturbation effects and require an explicit "
            f"aggregation policy: {ambiguous}"
        )
    tier_col = "neighbor_bridge_evidence_tier"
    def joined_unique(values: pd.Series) -> str:
        return ",".join(sorted({str(value).strip() for value in values if pd.notna(value) and str(value).strip()}))

    aggregations = {
        args.target_col: joined_unique,
        "delta_core": "first",
        "delta_repair": "first",
    }
    if tier_col in merged.columns:
        aggregations[tier_col] = joined_unique
    merged = (
        merged.groupby([args.drug_name_col, "gene_perturbation"], as_index=False)
        .agg(aggregations)
        .sort_values([args.drug_name_col, "gene_perturbation"])
    )

    proxy = pd.DataFrame(
        {
            "perturbation": merged[args.drug_name_col].astype(str) + " (target-bridged proxy)",
            "delta_core": merged["delta_core"],
            "delta_repair": merged["delta_repair"],
            "modality": "drug_proxy",
            "target_genes": merged[args.target_col].astype(str),
            "source": (
                "target-bridged proxy via "
                + merged["gene_perturbation"].astype(str)
                + "; "
                + (merged[tier_col].astype(str) if tier_col in merged.columns else "reviewed target match")
            ),
            "proxy_from_perturbation": merged["gene_perturbation"].astype(str),
        }
    )
    gene = effects.copy()
    for column, default in {
        "modality": "gene",
        "target_genes": "",
        "source": "user_supplied_gene_effect",
        "proxy_from_perturbation": "",
    }.items():
        if column not in gene.columns:
            gene[column] = default
        else:
            gene[column] = gene[column].fillna(default)
    combined = pd.concat(
        [gene[["perturbation", "delta_core", "delta_repair", "modality", "target_genes", "source", "proxy_from_perturbation"]], proxy],
        ignore_index=True,
    )
    combined = normalize_effect_table(combined)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    if args.manifest:
        assert manifest_path is not None
        assert manifest is not None
        manifest.setdefault("outputs", {})["combined_effects"] = {
            "file": output.name,
            "rows": len(combined),
            "gene_effects": len(gene),
            "target_bridged_drug_proxies": len(proxy),
            "sha256": sha256(output),
        }
        manifest["drug_bridge"] = {
            "file": drug_bridge_path.name,
            "sha256": sha256(drug_bridge_path),
            "mapping_file": mapping_path.name,
            "mapping_sha256": sha256(mapping_path),
            "gene_effects_file": gene_effects_path.name,
            "gene_effects_sha256": sha256(gene_effects_path),
            "schema_version": 1,
            "strokeniche_vc_version": __version__,
            "interpretation": "drug rows inherit gene effects through reviewed targets; not drug-specific simulation",
            "aggregation_policy": "all targets and evidence tiers are retained when they map to one gene perturbation; multiple perturbation effects fail",
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"gene effects={len(gene)}, target-bridged drug proxies={len(proxy)} -> {output}")


if __name__ == "__main__":
    main()
