#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
PROJECT_ROOT=${PROJECT_ROOT:-/mnt/h/vir/ST}
RESULTS_ROOT=${RESULTS_ROOT:-/mnt/h/vir/ST/results/step8_strokeniche_perturbmap}
H5AD=${H5AD:-/mnt/h/vir/ST/results/step5_nicheformer/spatial_all_with_nicheformer.h5ad}
H5AD_SPATIAL=${H5AD_SPATIAL:-/mnt/h/vir/ST/results/virtual_cell_h5ad/spatial_all.h5ad}
STATE_TABLE=${STATE_TABLE:-/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv}

echo "This wrapper documents the final figure-generation order."
echo "Run each command after checking paths and dependencies."

# Representative final figure scripts:
python scripts/73e_final_polished_track_dynamic_marker_heatmap.py --help || true
python scripts/74b_polished_wgcna_style_module_evidence_figure.py --help || true
python scripts/75b_dark_spatial_landscape_atlas.py --help || true
python scripts/75d_module_state_landscape_2x4_atlas.py --help || true
python scripts/76b_force_obsm_latent_state_potential_landscape.py --help || true
python scripts/77b_make_manuscript_ready_compact_landscape_fixed.py --help || true
python scripts/78d_fix_response_landscape_atlas_and_repair_audit.py --help || true
python scripts/79_state_timepoint_spatial_progression_atlas_final_v2.py --help || true
python scripts/80_candidate_perturbation_axis_expression_dotplot.py --help || true
python scripts/81b_surrogate_feature_response_audit_fixed.py --help || true
