# Core innovation and code map

## 1. Context-aware virtual-cell representation

The historical baseline adapter combines pretrained Nicheformer features, scaled spatial
coordinates, and a sparse graph-attention layer. Multitask heads reconstruct
region/state, time, neighbour composition, and repair-related outputs. The reusable
safety rule is that a reconstruction target must not also be provided as an input.

### Transparent v9 spatial-context extension

The additive v9 model encodes each spot as
`[standardised local expression, within-section neighbour-mean expression] / sqrt(2)`.
Its section-isolated 12-neighbour operator contains no self-edges or cross-section
edges. An exactly matched duplicated self-edge representation separates the
contribution of tissue topology from the change in feature dimension and L2
regularisation. The implementation is `strokeniche_vc.context`; equations,
nested whole-section selection and locked aggregate metrics are documented in
[`V9_METHODS_AND_VALIDATION.md`](V9_METHODS_AND_VALIDATION.md).

Legacy provenance scripts (not a validated default pipeline):

| Stage | Script | Role |
|---|---|---|
| Input assembly | `scripts/core_pipeline/model/13a_prepare_strokeniche_adapter_clean_inputs.py` | Align embeddings, annotations, coordinates, graph, and targets |
| Baseline adapter | `scripts/core_pipeline/model/13b_train_strokeniche_adapter_clean.py` | Sparse graph adapter and multitask heads; neighbour target input is now off by default |
| Neighbour head | `scripts/core_pipeline/model/54_train_neural_neighbor_head.py` | Learn the latent-to-neighbour-composition mapping |
| Latent export | `scripts/core_pipeline/model/54a_true_export_adapter_latent_from_checkpoint_v2.py` | Export adapter latent state for downstream analysis |

## 2. Perturbation-aware state editing

Two complementary routes are retained:

- **Expression classifier counterfactual (preferred interpretable analysis route):** a
  predeclared feature panel trains a balanced state classifier. Target genes are
  scaled and the change in class probability is reported. The reusable API is
  `strokeniche_vc.ExpressionCounterfactualModel`; the full research analysis is
  `scripts/core_pipeline/evaluation/72c_expression_level_insilico_ko_strict_label_classifier.py`.
- **Perturbation adapter (experimental):** candidate identity and curated response
  labels guide latent editing. Source scripts are Steps 36, 37, and 55. Because
  labels and candidate features are weakly supervised, this route is suitable for
  prioritization and hypothesis generation, not unknown-drug efficacy claims.

Boundary/domain-loss variants are kept under `scripts/experiments/` rather than
presented as validated defaults.

The v9 numerical operators are defined in `strokeniche_vc.operators`. Programme
down-modulation scales the selected log-normalised inputs by a declared factor.
ECM-associated up-modulation uses a training-derived, non-decreasing increment;
values already above the upper reference remain unchanged and strength zero is
the exact identity. The resulting probability differences are model-predicted
feature sensitivities, which can be decomposed into single-gene and
leave-one-gene-out contributions.

The current Streamlit release does not call this classifier live. It visualizes
precomputed candidate-level effects exported from an audited analysis. Arbitrary
uploaded-gene inference is a future extension that requires a versioned classifier,
feature matrix, grouped validation metadata, and model provenance.

## 3. Leakage-aware validation

`strokeniche_vc.splitting` and `scripts/make_inductive_split.py` create grouped
train/validation/test masks, fit coordinate scaling on training rows only, and
rebuild spatial kNN separately inside each group.
Use separate grouping units when necessary: split by the independent biological
source (often animal/sample), and construct kNN only within one physical coordinate
frame (often tissue section). Every graph group must be nested in one split group.

Additional audit routes include held-out timepoint evaluation, no-target-leakage
recomputation, ablation diagnostics, known-positive/decoy benchmarking, and the
Step72D freeze/robustness check under `scripts/core_pipeline/evaluation/`.

The legacy Step13A preparer used full-data coordinate scaling, one global kNN, and
spot-level random masks. Step13B plus Step13A must therefore be treated as
historical/transductive provenance unless inputs are rebuilt with the safe wrapper.

## 4. Response landscape

The white-background explorer uses a transparent two-stage visualization:

\[
\tilde c_i=\operatorname{RMM}(c_i),\quad
\tilde r_i=\operatorname{RMM}(r_i),\quad
s_i=0.25+0.75\operatorname{RMM}\{0.65\tilde c_i+0.35(1-\tilde r_i)\},
\]

\[
c'_i=\operatorname{clip}(c_i+\alpha\Delta c\,s_i,0,1),\qquad
r'_i=\operatorname{clip}(r_i+\alpha\Delta r\,s_i,0,1).
\]

Here `c` is the baseline core-like score and `r` is the ordered injury-state
display score (retained as `baseline_repair` in the legacy app input schema). Candidate
mean shifts are supplied by a model export. The 2-D surface is a density-masked,
Gaussian-smoothed display proxy; it is not a physical or Waddington potential.
`RMM` is 1st–99th percentile robust min-max scaling, applied to each baseline
score and then to the weighted combination.
The response-priority value remains on one absolute weighted scale
`0.55*core_reduction + 0.45*repair_gain`; it is not independently normalized per
candidate, so candidate and strength comparisons retain magnitude information.
The 0.55/0.45 values are heuristic display weights, not learned or clinically
calibrated utility weights. This is a deliberate visualization-safe deviation from
the exact legacy Step78D priority normalization; the unchanged legacy script is
retained under `scripts/final_visualization/`.

The six audited white-background publication scripts are preserved in
`scripts/final_visualization/`.

## 5. Drug visualization

Drug evidence should be represented as an evidence graph:

`drug → LINCS signature / knowledge evidence → target → gene perturbation proxy → state-probability redistribution`

Only target-matched compounds should inherit a spatial gene-perturbation map, and
the map must remain labelled **target-bridged proxy**. LINCS cosine/Spearman
signature reversal is not animal or clinical efficacy.
