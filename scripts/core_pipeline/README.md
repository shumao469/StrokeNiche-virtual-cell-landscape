# Core research pipeline

Selected numbered scripts from the original StrokeNiche working project are kept
here for analysis provenance. `model/` contains input assembly, graph-adapter,
latent, neighbour-head, and perturbation routes. `evaluation/` contains held-out,
leakage, ablation, benchmark, and expression-counterfactual audits.

These scripts retain historical absolute paths and study-specific schemas. Treat
them as reference implementations; use `src/strokeniche_vc/` for reusable code.
Run legacy deserialization only on trusted local files.

## Rebuild a leakage-aware input

The group table must contain one row per observation (identical duplicate rows are
allowed) and three columns such as:

```text
obs_name,animal_id,section_id
cell_001,mouse_01,mouse_01_section_A
```

Use the animal/sample as the independent split unit and the nested physical
section as the coordinate-frame unit:

```bash
python scripts/make_inductive_split.py \
  --input-npz local_bundle/trusted_legacy_inputs.npz \
  --group-table local_bundle/observation_groups.csv \
  --split-group-col animal_id \
  --graph-group-col section_id \
  --output-npz local_bundle/rebuilt_inputs.npz
```

Add `--trust-pickle-input` only after verifying an NPZ that contains legacy object
arrays. The rebuilt masks/scaling/graph change the training data: retrain the model.
Do not reuse old checkpoints or their metrics as evidence for the rebuilt graph.
Keep the observation-to-animal/section table in ignored `local_bundle/`; do not
commit study or sample identifiers to the repository.

## Build reviewed target-bridged drug rows

```bash
python scripts/build_drug_target_proxy_effects.py \
  --gene-effects local_bundle/perturbation_effects.csv \
  --drug-bridge reviewed_drug_target_bridge.csv \
  --mapping configs/drug_target_mapping.example.csv \
  --output local_bundle/gene_and_drug_proxy_effects.csv
```

The bridge must contain `drug_name` and
`neighbor_bridge_matched_target_gene` (or pass alternative column names). A drug
mapping to more than one perturbation effect fails until an explicit aggregation
policy is supplied; multiple targets for the same effect are all retained.
