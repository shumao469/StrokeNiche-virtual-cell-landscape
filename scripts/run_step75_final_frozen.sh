#!/usr/bin/env bash
set -euo pipefail

cd /mnt/h/vir/ST

PY=/home/shu/miniconda/envs/nicheformer_env/bin/python

H5AD=/mnt/h/vir/ST/results/virtual_cell_h5ad/spatial_all.h5ad
STATE=/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/dynamics_graph_64b_fixed_region_probs_state_celltype/step64b_modeling_input_used.fixed_region_probs.csv

OUT=/mnt/h/vir/ST/results/step8_strokeniche_perturbmap/spatial_landscape_atlas_75_final_frozen
mkdir -p "$OUT"

echo "===================================================================================================="
echo "Step75 final frozen | dark spatial marker/module/state landscape atlas"
echo "===================================================================================================="
echo "H5AD=$H5AD"
echo "STATE=$STATE"
echo "OUT=$OUT"

# Keep a frozen copy of the exact scripts used.
cp -f /mnt/h/vir/ST/75b_dark_spatial_landscape_atlas.py "$OUT/75b_dark_spatial_landscape_atlas.frozen.py"
cp -f /mnt/h/vir/ST/75c_rerun_dark_spatial_landscape_atlas_fixed_cbar.py "$OUT/75c_final_frozen_runner.py"

$PY -m py_compile "$OUT/75b_dark_spatial_landscape_atlas.frozen.py"
$PY -m py_compile "$OUT/75c_final_frozen_runner.py"

$PY "$OUT/75c_final_frozen_runner.py" \
  --step75b_script "$OUT/75b_dark_spatial_landscape_atlas.frozen.py" \
  --h5ad "$H5AD" \
  --state_table "$STATE" \
  --outdir "$OUT" \
  --core_col core_probability \
  --peri_col peri_probability \
  --remote_col remote_probability \
  --point_size 2.0 \
  --dpi 600 \
  2>&1 | tee "$OUT/run_step75_final_frozen.log"

echo "===================================================================================================="
echo "Freeze manifest"
echo "===================================================================================================="

{
  echo "Step75 final frozen"
  echo "Date: $(date)"
  echo "H5AD: $H5AD"
  echo "STATE: $STATE"
  echo "OUT: $OUT"
  echo ""
  echo "Expected key audit:"
  echo "  n_obs = 7756"
  echo "  state_merge_overlap = 7756"
  echo "  prob_cols = core_probability / peri_probability / remote_probability"
  echo ""
  echo "Frozen scripts:"
  sha256sum "$OUT/75b_dark_spatial_landscape_atlas.frozen.py"
  sha256sum "$OUT/75c_final_frozen_runner.py"
  echo ""
  echo "Main outputs:"
  ls -lh "$OUT"/Fig_Step75C_*.png
  ls -lh "$OUT"/Fig_Step75C_*.pdf
  ls -lh "$OUT"/Fig_Step75C_*.svg
} | tee "$OUT/step75_final_freeze_manifest.txt"

echo "===================================================================================================="
echo "DONE Step75 final frozen"
echo "===================================================================================================="
