# Manuscript v9 methods and validation map

## Scientific objective

StrokeNiche maps how declared numerical edits to ischaemic-stroke-associated
gene programmes change model-predicted tissue-state probabilities, and locates
those changes in the tissue section. A **counterfactual operator** is the exact
mathematical transformation applied to selected model inputs before the fixed
model is evaluated again. The reported quantity is

\[
\Delta \mathbf p_i
= f_\theta\!\left(T_{\mathcal G}(\mathbf x_i)\right)
- f_\theta(\mathbf x_i),
\]

where \(T_{\mathcal G}\) edits the declared feature set \(\mathcal G\),
\(f_\theta\) is held fixed, and \(\Delta\mathbf p_i\) is the change in the
complete state-probability vector for spot \(i\).

## Spatial-context extension

The v9 extension separates local expression from its one-hop tissue context.
Within each section, each spot receives the unweighted mean of its 12 nearest
other spots. The resulting row-normalised operator \(A\) contains neither
self-edges nor cross-section edges. Standardisation is estimated from training
sections only. The three comparable feature maps are

\[
\phi_i^{\mathrm{expression}}=\mathbf z_i,\qquad
\phi_i^{\mathrm{self}}=2^{-1/2}[\mathbf z_i,\mathbf z_i],\qquad
\phi_i^{\mathrm{spatial}}=2^{-1/2}[\mathbf z_i,(A\mathbf Z)_i].
\]

All three use the same class-balanced, L2-regularised multinomial logistic
readout, optimiser and candidate values of \(C\). Equal block scaling makes the
self-edge control L2-equivalent to the expression-only model. The spatial
linear predictor can also be written as

\[
W_0\mathbf z_i+W_1(A\mathbf Z)_i
=(W_0+W_1)\mathbf z_i+W_1\{(A\mathbf Z)_i-\mathbf z_i\},
\]

which exposes the learned neighbourhood residual. The implementation is in
[`strokeniche_vc.context`](../src/strokeniche_vc/context.py).

## Numerical gene-programme edits

Down-modulation multiplies selected log-normalised inputs by a declared scaling
factor \(\kappa\):

\[
x'_{ig}=\kappa x_{ig},\qquad g\in\mathcal G.
\]

Thus \(\kappa\) describes scaling in the supplied transformed feature space;
it is not a percentage change in raw molecular abundance. The ECM-associated
up-modulation uses training-derived scale \(s_g\) and ceiling \(c_g\):

\[
x'_{ig}=x_{ig}+\min\!\left\{\eta s_g,
\max(c_g-x_{ig},0)\right\},
\quad
c_g=Q_{0.995,g}+0.5\max(s_g,10^{-6}).
\]

The transformation is non-decreasing, leaves values above the ceiling
unchanged, and becomes the identity at \(\eta=0\). Reusable implementations
and unit tests are provided in
[`strokeniche_vc.operators`](../src/strokeniche_vc/operators.py).

## Nested whole-section analysis

Each outer analysis holds out D1, D3 or D7. For the two remaining sections,
the inner loop trains on one whole section and validates on the other, then
swaps them. \(C\in\{0.01,0.1,1,10\}\) is selected by equal-section mean
macro-F1, with ties assigned to the smaller value. No held-out labels enter
feature scaling, graph construction, parameter selection or fitting.

The locked v9 equal-section means were:

| Model | Macro-F1 | Balanced accuracy | Log loss | Brier score | AUROC |
|---|---:|---:|---:|---:|---:|
| Expression only | 0.6780 | 0.6934 | 0.6802 | 0.3669 | 0.8787 |
| Self-edge control | 0.6780 | 0.6934 | 0.6802 | 0.3669 | 0.8787 |
| StrokeNiche spatial context | **0.6809** | 0.6899 | **0.6606** | **0.3424** | **0.8948** |

The spatial-context model therefore improves probability quality and AUROC in
this locked comparison while retaining comparable macro-F1. A separate
64-spot edit confirms the designed one-hop mechanism: only the spatial model
propagates a probability change to non-seed neighbours, and no change occurs
beyond the declared one-hop support.

## Programme specificity and evidence tracing

Each programme is reported as an **associated programme edit**, because its
genes can occupy different biological roles. Single-gene and
leave-one-gene-out decompositions identify which inputs contribute most to each
predicted endpoint. Expression- and detection-matched random programmes use
5,000 draws per programme and section, followed by the declared 30-test
Benjamini-Hochberg procedure.

Knowledge resources enter as evidence layers with recorded roles: CellChat-
and OmniPath-associated interaction resources, gene-set and transcription-
factor-target libraries, DGIdb, CTD, DrugCentral, ChEMBL and LINCS. They support
programme construction, target annotation or candidate tracing; they are not
substituted for the probability model.

## Reproducibility boundary

The public repository contains data-agnostic algorithms, example configuration,
tests, the Streamlit explorer and curated provenance scripts. Study matrices,
spot-level exports and checkpoints remain in the separately checksummed source-
data package. The D1/D3/D7 values above are section-level development analyses;
independent biological cohorts should be used when extending the model to new
animals, platforms or clinical specimens.
