# StrokeNiche virtual-cell landscape analysis

Prepared: 2026-06-15T17:55:05

This repository contains scripts used to generate the StrokeNiche spatial, latent-state,
state-probability, and virtual perturbation response landscapes.

## Main figure-generating steps

- Step75B/75D: spatial marker and module landscape atlas
- Step76B: latent-state pseudo-energy landscape
- Step77B: latent state-probability landscape
- Step78D: virtual perturbation response landscape
- Step79 final v2: state/timepoint-specific spatial progression atlas
- Step80: candidate perturbation-axis expression dotplot
- Step81B: surrogate feature-response audit

## Data

Large processed inputs and result tables are deposited separately in the Zenodo data/results bundle.
See `docs/input_data_manifest.tsv`, `docs/figure_manifest.tsv`, and `docs/table_manifest.tsv`.

## Reproducibility

1. Create or activate the analysis environment.
2. Edit local paths in `configs/paths.example.yaml` if needed.
3. Run the final figure scripts or the shell wrappers in `workflow/`.

## Important interpretation note

Virtual perturbation response landscapes are computational surrogate analyses.
They should not be interpreted as wet-lab knockout/blockade experiments, causal ligand-receptor validation,
physical energy landscapes, true Waddington potentials, or directly observed cell-state transitions.
# StrokeNiche-virtual-cell-landscape
