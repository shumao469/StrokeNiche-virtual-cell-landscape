# v0.3.0 / manuscript-v9 release manifest

This release is an additive update to the original repository. It retains the
Streamlit explorer, reusable landscape and response utilities, curated legacy
research scripts, white-background figure scripts, exporters, tests and CI.

Added or updated:

- `src/strokeniche_vc/context.py`: section-isolated kNN operator and the
  expression, self-edge and spatial feature maps.
- `src/strokeniche_vc/operators.py`: explicit log-expression scaling and the
  training-derived monotone up-modulation operator.
- `tests/test_context.py` and `tests/test_operators.py`: graph-isolation,
  L2-equivalence, identity and monotonicity checks.
- `configs/v9_analysis.example.json`: path-free locked analysis parameters.
- `docs/V9_METHODS_AND_VALIDATION.md`: equations, evaluation design, aggregate
  metrics and the evidence-layer map.
- package and citation version: `0.3.0`.

Excluded from GitHub:

- raw or processed study matrices;
- spot-level prediction and edit tables;
- fitted checkpoints and local filesystem paths;
- manuscript source-data archives.

These larger artifacts remain governed by the manuscript data/code-availability
package and its SHA-256 manifests.
