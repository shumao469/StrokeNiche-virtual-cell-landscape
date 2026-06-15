#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step99 | Prepare GitHub + Zenodo release bundle for StrokeNiche manuscript.

This script:
1. Collects final figure files from Step64-81 result folders.
2. Collects scripts from project root.
3. Generates code, figure, table, input-data, and step-ledger manifests.
4. Creates a GitHub-ready code skeleton and Zenodo-ready data/results bundle.

It does not upload to GitHub or Zenodo.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


DEFAULT_PROJECT_ROOT = "/mnt/h/vir/ST"
DEFAULT_RESULTS_ROOT = "/mnt/h/vir/ST/results/step8_strokeniche_perturbmap"
DEFAULT_OUTDIR = "/mnt/h/vir/ST/release_StrokeNiche_NCS_v1"


STEP_DIR_PATTERNS = {
    "Step64_dynamics_graph_state_modeling": [
        "dynamics_graph_64b_fixed_region_probs_state_celltype",
        "dynamics_graph_64c_refined_labels_state_celltype",
    ],
    "Step65_track_dynamic_markers_regulators": [
        "track_dynamic_marker_regulator_65*",
        "*65*dynamic*",
    ],
    "Step66_perturbation_fdr_summary": [
        "track_level_perturbation_fdr_66f*",
        "*66f*",
    ],
    "Step68_validation_layer": [
        "validation_layer_68h*",
        "*68h*",
    ],
    "Step69_trajectory_virtual_perturbation_main": [
        "main_figure_69*",
        "*trajectory_aware_virtual_perturbation*",
    ],
    "Step70_virtual_KO_spatial_rescue": [
        "virtual_ko_spatial_rescue_70*",
        "*70b*",
    ],
    "Step71_state_probability_shift_sensitivity": [
        "*71*",
    ],
    "Step72_expression_level_insilico_KO": [
        "expression_level_insilico_ko_72*",
        "*72c*",
    ],
    "Step73_track_dynamic_marker_heatmap": [
        "*73*",
    ],
    "Step74_wgcna_style_module_evidence": [
        "*74*",
        "*wgcna*",
    ],
    "Step75_spatial_marker_module_atlas": [
        "spatial_landscape_atlas_75*",
        "*75b*",
        "*75d*",
        "*75e*",
    ],
    "Step76_latent_pseudo_energy_landscape": [
        "latent_pseudo_energy_landscape_76*",
        "*pseudo_energy*",
    ],
    "Step77_latent_state_probability_landscape": [
        "latent_state_probability_landscape_77*",
        "*77b*",
    ],
    "Step78_virtual_perturbation_response_landscape": [
        "virtual_perturbation_response_landscape_78*",
        "*78d*",
    ],
    "Step79_state_timepoint_spatial_progression": [
        "state_timepoint_spatial_progression_atlas_79*",
        "*79_final*",
    ],
    "Step80_candidate_perturbation_dotplot": [
        "candidate_perturbation_axis_dotplot_80*",
        "*dotplot_80*",
    ],
    "Step81_surrogate_feature_response_audit": [
        "surrogate_feature_response_audit_81*",
        "*81b*",
    ],
}


KEY_INPUTS = [
    "results/virtual_cell_h5ad/spatial_all.h5ad",
    "results/step5_nicheformer/spatial_all_with_nicheformer.h5ad",
    "results/step5_nicheformer/nicheformer_spatial_with_embeddings.h5ad",
    "results/step6_scgpt/scgpt_spatial_with_embeddings.h5ad",
    "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv",
    "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_dynamics_graph_nodes.csv",
    "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_dynamics_graph_edges.csv",
    "results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_track_summary.csv",
]


FIG_EXTS = {".png", ".pdf", ".svg", ".tif", ".tiff", ".jpg", ".jpeg"}
TABLE_EXTS = {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".xlsx", ".parquet"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def safe_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    if src.is_file():
        shutil.copy2(src, dst)


def file_row(src: Path, dst: Path = None, step: str = "", role: str = "") -> dict:
    row = {
        "step": step,
        "role": role,
        "source_path": str(src),
        "file_name": src.name,
        "file_ext": src.suffix,
        "size_bytes": src.stat().st_size if src.exists() else "",
        "sha256": sha256_file(src) if src.exists() and src.is_file() else "",
    }
    if dst is not None:
        row["release_path"] = str(dst)
    return row


def find_dirs(root: Path, patterns: list) -> list:
    hits = []
    for pat in patterns:
        hits.extend(root.glob(pat))
        hits.extend(root.glob(f"**/{pat}"))
    hits = [p for p in hits if p.exists() and p.is_dir()]
    return sorted(set(hits))


def collect_files(d: Path, exts: set) -> list:
    out = []
    for ext in exts:
        out.extend(d.glob(f"*{ext}"))
        out.extend(d.glob(f"**/*{ext}"))
    return sorted(set([p for p in out if p.is_file()]))


def figure_role(step_name: str, file_name: str) -> str:
    s = (step_name + " " + file_name).lower()

    if "step75" in s:
        return "main_candidate_spatial_marker_module_atlas"
    if "step76" in s:
        return "main_candidate_latent_pseudo_energy"
    if "step77" in s and "77b" in s:
        return "main_candidate_latent_state_probability"
    if "step78" in s and ("main" in s or "78d" in s or "spp1" in s or "ferroptosis" in s):
        return "main_or_extended_virtual_perturbation_response"
    if "step79" in s:
        return "extended_state_timepoint_progression"
    if "step80" in s:
        return "extended_candidate_expression_dotplot"
    if "step81" in s:
        return "extended_surrogate_response_qc"
    if "step73" in s or "step74" in s:
        return "extended_module_or_track_evidence"
    if "step69" in s or "step70" in s:
        return "supplementary_virtual_perturbation_qc"
    if "step71" in s or "step72" in s:
        return "supplementary_sensitivity_or_surrogate_qc"
    return "supplementary_or_qc"


def write_text(path: Path, lines):
    ensure_dir(path.parent)
    if isinstance(lines, str):
        text = lines
    else:
        text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")


def export_environment(dst: Path):
    try:
        txt = subprocess.check_output(
            ["bash", "-lc", "conda env export --no-builds 2>/dev/null || true"],
            text=True,
            timeout=90,
        )
    except Exception:
        txt = ""

    if txt.strip():
        write_text(dst, txt)
    else:
        write_text(dst, [
            "# Environment export placeholder",
            "# Run in the correct environment:",
            "# conda env export --no-builds > environment.yml",
        ])


def make_readme(github_dir: Path, created: str):
    lines = [
        "# StrokeNiche virtual-cell landscape analysis",
        "",
        f"Prepared: {created}",
        "",
        "This repository contains scripts used to generate the StrokeNiche spatial, latent-state,",
        "state-probability, and virtual perturbation response landscapes.",
        "",
        "## Main figure-generating steps",
        "",
        "- Step75B/75D: spatial marker and module landscape atlas",
        "- Step76B: latent-state pseudo-energy landscape",
        "- Step77B: latent state-probability landscape",
        "- Step78D: virtual perturbation response landscape",
        "- Step79 final v2: state/timepoint-specific spatial progression atlas",
        "- Step80: candidate perturbation-axis expression dotplot",
        "- Step81B: surrogate feature-response audit",
        "",
        "## Data",
        "",
        "Large processed inputs and result tables are deposited separately in the Zenodo data/results bundle.",
        "See `docs/input_data_manifest.tsv`, `docs/figure_manifest.tsv`, and `docs/table_manifest.tsv`.",
        "",
        "## Reproducibility",
        "",
        "1. Create or activate the analysis environment.",
        "2. Edit local paths in `configs/paths.example.yaml` if needed.",
        "3. Run the final figure scripts or the shell wrappers in `workflow/`.",
        "",
        "## Important interpretation note",
        "",
        "Virtual perturbation response landscapes are computational surrogate analyses.",
        "They should not be interpreted as wet-lab knockout/blockade experiments, causal ligand-receptor validation,",
        "physical energy landscapes, true Waddington potentials, or directly observed cell-state transitions.",
    ]
    write_text(github_dir / "README.md", lines)


def make_citation(github_dir: Path):
    lines = [
        "cff-version: 1.2.0",
        'title: "StrokeNiche virtual-cell landscape analysis"',
        'message: "If you use this code, please cite the associated article and Zenodo DOI."',
        "type: software",
        "authors:",
        "  - family-names: Xu",
        "    given-names: Shumao",
        'repository-code: "https://github.com/shumaoxu/StrokeNiche-virtual-cell-landscape"',
        "license: MIT",
        "version: 1.0.0",
        "date-released: YYYY-MM-DD",
        'doi: "10.5281/zenodo.TO_BE_ADDED"',
    ]
    write_text(github_dir / "CITATION.cff", lines)


def make_misc_files(github_dir: Path, zenodo_dir: Path):
    write_text(github_dir / ".gitignore", [
        "# Large data",
        "*.h5ad",
        "*.h5",
        "*.loom",
        "*.zarr/",
        "*.npz",
        "*.npy",
        "*.pt",
        "*.pth",
        "*.ckpt",
        "results/",
        "data/",
        "",
        "# Cache",
        "__pycache__/",
        ".ipynb_checkpoints/",
        "*.log",
        ".DS_Store",
    ])

    write_text(github_dir / "LICENSE", [
        "MIT License",
        "",
        "Copyright (c) 2026",
        "",
        "Permission is hereby granted, free of charge, to any person obtaining a copy",
        "of this software and associated documentation files, to deal in the Software",
        "without restriction, subject to the conditions of the MIT License.",
        "",
        "Replace this placeholder with the final institutionally approved license text before public release.",
    ])

    ensure_dir(github_dir / "configs")
    write_text(github_dir / "configs" / "paths.example.yaml", [
        "project_root: /path/to/ST",
        "results_root: /path/to/ST/results/step8_strokeniche_perturbmap",
        "h5ad_nicheformer: /path/to/spatial_all_with_nicheformer.h5ad",
        "state_table: /path/to/step64b_modeling_input_used.fixed_region_probs.csv",
    ])

    ensure_dir(github_dir / "workflow")
    write_text(github_dir / "workflow" / "run_main_figures.sh", [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "echo 'Edit paths before running final figure scripts.'",
        "python scripts/75_spatial_landscape_atlas.py || true",
        "python scripts/76_latent_pseudo_energy_landscape.py || true",
        "python scripts/77b_make_manuscript_ready_compact_landscape.py || true",
        "python scripts/78d_fix_response_landscape_atlas_and_repair_audit.py || true",
        "python scripts/79_state_timepoint_spatial_progression_atlas_final_v2.py || true",
        "python scripts/80_candidate_perturbation_axis_expression_dotplot.py || true",
        "python scripts/81b_surrogate_feature_response_audit_fixed.py || true",
    ])

    write_text(zenodo_dir / "README_ZENODO.md", [
        "# StrokeNiche data/results bundle",
        "",
        "This Zenodo bundle contains manuscript-ready figures, source tables, audit reports,",
        "and manifests for the StrokeNiche virtual-cell landscape analysis.",
        "",
        "## Contents",
        "",
        "- figures/: manuscript-ready and extended/supplementary figure files",
        "- tables/: figure source tables, audit reports and QC outputs",
        "- manifests/: SHA256 and provenance manifests",
        "- processed_inputs/: optional processed h5ad/csv inputs if copied with --copy_large_inputs",
        "",
        "## Interpretation note",
        "",
        "Virtual perturbation response maps are in silico surrogate response landscapes.",
        "They are not wet-lab knockout/blockade experiments and not observed cell-state transitions.",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_root", default=DEFAULT_PROJECT_ROOT)
    ap.add_argument("--results_root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR)
    ap.add_argument("--copy_large_inputs", action="store_true")
    ap.add_argument("--hash_large_inputs", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root).resolve()
    outdir = Path(args.outdir).resolve()

    github_dir = outdir / "github_repo_skeleton"
    zenodo_dir = outdir / "zenodo_data_results_bundle"
    manifest_dir = outdir / "manifests"

    for d in [github_dir, zenodo_dir, manifest_dir]:
        ensure_dir(d)

    created = datetime.now().isoformat(timespec="seconds")

    # ------------------------------------------------------------------
    # Collect scripts
    # ------------------------------------------------------------------
    script_rows = []
    script_dst_dir = github_dir / "scripts"
    ensure_dir(script_dst_dir)

    for ext in ["*.py", "*.R", "*.sh"]:
        for p in sorted(project_root.glob(ext)):
            if p.is_file() and not p.name.startswith("."):
                dst = script_dst_dir / p.name
                safe_copy(p, dst)
                script_rows.append(file_row(p, dst.relative_to(outdir), step="code", role="analysis_script"))

    # ------------------------------------------------------------------
    # Collect figures and tables
    # ------------------------------------------------------------------
    fig_rows = []
    table_rows = []

    for step_name, patterns in STEP_DIR_PATTERNS.items():
        dirs = find_dirs(results_root, patterns)

        for d in dirs:
            figs = collect_files(d, FIG_EXTS)
            tabs = collect_files(d, TABLE_EXTS)

            for fig in figs:
                role = figure_role(step_name, fig.name)
                dst = zenodo_dir / "figures" / step_name / fig.name
                safe_copy(fig, dst)
                row = file_row(fig, dst.relative_to(outdir), step=step_name, role=role)
                row["source_dir"] = str(d)
                fig_rows.append(row)

            for tab in tabs:
                dst = zenodo_dir / "tables" / step_name / tab.name
                safe_copy(tab, dst)
                row = file_row(tab, dst.relative_to(outdir), step=step_name, role="source_table_or_audit")
                row["source_dir"] = str(d)
                table_rows.append(row)

    # ------------------------------------------------------------------
    # Key input manifest
    # ------------------------------------------------------------------
    input_rows = []
    for rel in KEY_INPUTS:
        p = project_root / rel
        exists = p.exists()
        row = {
            "input_name": Path(rel).name,
            "relative_path": rel,
            "source_path": str(p),
            "exists": bool(exists),
            "size_bytes": p.stat().st_size if exists else "",
            "sha256": "",
            "copied_to_zenodo_bundle": False,
            "release_path": "",
        }

        if exists and (args.hash_large_inputs or p.stat().st_size < 500 * 1024 * 1024):
            row["sha256"] = sha256_file(p)

        if exists and args.copy_large_inputs:
            dst = zenodo_dir / "processed_inputs" / Path(rel).name
            safe_copy(p, dst)
            row["copied_to_zenodo_bundle"] = True
            row["release_path"] = str(dst.relative_to(outdir))

        input_rows.append(row)

    # ------------------------------------------------------------------
    # Step ledger
    # ------------------------------------------------------------------
    step_ledger = [
        ["Step1-20", "Data preprocessing and spatial object construction", "Methods", "summarize, not all files needed"],
        ["Step21-40", "Annotation, region/state definition and QC", "Methods/Fig1", "summarize"],
        ["Step41-55", "Embedding and latent representation", "Methods/Fig1/Fig3", "summarize"],
        ["Step56-64", "State modeling and dynamics graph", "Methods/Results", "include final state table and graph files"],
        ["Step65-68", "Dynamic regulators, perturbation FDR, validation layer", "Extended/Supplementary", "include source tables and reports"],
        ["Step69", "Trajectory-aware virtual perturbation main figure", "Supplementary/QC or method schematic", "optional"],
        ["Step70", "Virtual KO/blockade spatial rescue maps", "Supplementary/QC", "optional"],
        ["Step71", "State probability shift sensitivity", "Supplementary/QC", "freeze as sensitivity only"],
        ["Step72", "Expression-level in silico KO/blockade", "Methods/Supplementary", "include report and response source if used"],
        ["Step73", "Track-specific dynamic marker heatmap", "Extended Data", "freeze"],
        ["Step74", "WGCNA-style module evidence figure", "Extended Data", "freeze"],
        ["Step75B/75D", "Spatial marker/module landscape atlas", "Main Fig2", "freeze"],
        ["Step76B", "Latent-state pseudo-energy landscape", "Main Fig3", "freeze"],
        ["Step77B", "Latent state-probability landscape", "Main Fig4", "freeze"],
        ["Step78D", "Virtual perturbation response landscape", "Main Fig5", "freeze"],
        ["Step79 final v2", "State/timepoint-specific spatial progression atlas", "Extended Data", "freeze; label-derived state maps"],
        ["Step80", "Candidate perturbation-axis expression dotplot", "Extended Data", "freeze"],
        ["Step81B", "Surrogate feature-response audit", "Extended Data/QC", "freeze with caveat"],
    ]
    step_df = pd.DataFrame(step_ledger, columns=["step", "description", "recommended_location", "status"])

    # ------------------------------------------------------------------
    # Write manifests
    # ------------------------------------------------------------------
    pd.DataFrame(script_rows).to_csv(manifest_dir / "code_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(fig_rows).to_csv(manifest_dir / "figure_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(table_rows).to_csv(manifest_dir / "table_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(input_rows).to_csv(manifest_dir / "input_data_manifest.tsv", sep="\t", index=False)
    step_df.to_csv(manifest_dir / "step_ledger.tsv", sep="\t", index=False)

    # Figure inventory copy
    if fig_rows:
        pd.DataFrame(fig_rows).to_csv(manifest_dir / "figure_inventory.tsv", sep="\t", index=False)

    # Copy manifests into GitHub and Zenodo bundles
    for f in sorted(manifest_dir.glob("*.tsv")):
        safe_copy(f, github_dir / "docs" / f.name)
        safe_copy(f, zenodo_dir / "manifests" / f.name)

    # Environment and docs
    export_environment(github_dir / "environment.yml")
    make_readme(github_dir, created)
    make_citation(github_dir)
    make_misc_files(github_dir, zenodo_dir)

    summary = {
        "status": "ok",
        "created": created,
        "project_root": str(project_root),
        "results_root": str(results_root),
        "outdir": str(outdir),
        "github_repo_skeleton": str(github_dir),
        "zenodo_data_results_bundle": str(zenodo_dir),
        "manifest_dir": str(manifest_dir),
        "n_scripts": len(script_rows),
        "n_figures": len(fig_rows),
        "n_tables": len(table_rows),
        "n_key_inputs": len(input_rows),
    }

    write_text(outdir / "release_summary.json", json.dumps(summary, indent=2, ensure_ascii=False))

    print("=" * 100)
    print("DONE Step99 release bundle preparation")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
