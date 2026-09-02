# Scientific and release limitations

## Required interpretation boundaries

- A classifier counterfactual estimates how its predicted probabilities respond
  to an expression edit; it does not establish a causal biological response.
- `retained_fraction=0`, `0.2`, or `0.5` means 0%, 20%, or 50% of the selected
  expression value is retained. These values are not drug doses.
- Probability arrows, bars, and Sankey flows describe **predicted probability
  redistribution**, not observed migration or cell-state transition.
- Step78D uses candidate-level mean effects, heuristic susceptibility, forced PCA,
  and Gaussian smoothing. The available audited run reports fallback/default
  effects; display it as a smoothed visualization proxy.
- Step81 feature-response Spearman association is not SHAP, feature attribution,
  target engagement, or causality.
- LINCS reversal scores measure cross-context transcriptional similarity. They do
  not establish BBB exposure, safety, target engagement, animal benefit, or
  clinical benefit.
- A drug-to-target-to-perturbation map is a target-bridged proxy, not a
  drug-specific cell simulation.

## Leakage and independence controls

1. Split by independent biological source (animal/sample/section), not by spots.
2. Build spatial neighbours separately within each source; do not create one global
   coordinate graph across unrelated sections.
   When the biological split unit and coordinate frame differ, supply both (for
   example split by animal and build graph by nested tissue section).
3. Remove graph edges crossing split boundaries.
4. Fit scalers, PCA, feature selection, and thresholds on training data only.
5. Never provide a reconstruction target as a model input. The historical
   neighbour-composition input in Step13B is now disabled by default.
6. Do not use candidate score, positive/negative label, or downstream selection
   label as an input when estimating generalization to unseen perturbations.
7. Report held-out groups and held-out perturbations separately; a random held-out
   cell from a known perturbation does not test unknown-perturbation generalization.

## Data-column guardrail

In one legacy state table, `core_probability`, `peri_probability`, and
`remote_probability` are identical row by row. Do not treat them as a valid
three-class distribution. Prefer audited `prob_lesion_core`, `prob_peri_infarct`,
and `prob_remote_like`, or probabilities recomputed by the Step72C classifier.

## File integrity and serialization

- Multiple legacy result folders contain duplicated or zero-filled artifacts.
  Select a validated release copy, record SHA-256 hashes, and avoid scanning the
  entire project tree at app runtime.
- `numpy.load(..., allow_pickle=True)` and `torch.load(..., weights_only=False)`
  can execute unsafe deserialization paths. Use them only with trusted local
  artifacts. New wrappers default to safer loading or require explicit trust.
- Patient/study inputs, checkpoints, and result bundles are intentionally excluded
  from Git. Apply the relevant consent, institutional, and data-sharing controls.

## Release checklist

- Confirm institutional license text and author ownership.
- Replace article/Zenodo placeholders only with issued identifiers.
- Lock the validated data bundle and publish its SHA-256 manifest.
- Run grouped external/temporal validation, calibration, subgroup checks, and
  decision-utility analysis before making translational claims.
- Validate prioritized perturbations experimentally; this repository alone cannot
  support treatment recommendations.
