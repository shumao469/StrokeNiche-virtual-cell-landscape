# Interactive perturbation explorer

The Streamlit app shows how an uploaded, precomputed candidate gene/pathway or
target-bridged drug-mechanism effect changes virtual-cell state scores across
latent and spatial context. It does not run the expression classifier or graph
adapter live and cannot infer an arbitrary new gene/drug in this release.

## Run locally

```bash
python -m pip install -e ".[visualizer]"
python -m streamlit run app/perturbation_explorer.py
```

On Windows, after installing the dependencies once, double-click `run_visualizer.bat`.

## Data modes

### Safe synthetic demo

The built-in cell coordinates and effect magnitudes are deterministic synthetic
data. No patient or study records are included. Gene axes are biologically relevant
examples; all built-in effects, including drug entries, are illustrative only.

### Model-exported tables

Upload two CSV files in the sidebar.

Use patient- or study-derived files only in an institutionally governed local
deployment. Do not upload sensitive data to a public Streamlit host.

Cell table, required columns:

```text
latent1,latent2,baseline_core,baseline_repair
```

Optional columns:

```text
obs_name,state,timepoint,dominant_celltype,spatial_x,spatial_y
```

Effect table, required columns:

```text
perturbation,delta_core,delta_repair
```

Optional columns:

```text
modality,target_genes,source
```

Use `modality=gene` for gene/pathway perturbations and `modality=drug_proxy` (or another descriptive label) for model-derived drug effects.

When exporting the audited legacy data, prefer `prob_lesion_core` over the older
`core_probability` field. Use `scripts/export_visualizer_inputs.py` with an
audited Step64 context table to apply this guardrail and create a SHA-256 manifest.

## Scientific boundary

The app visualizes computational counterfactual or surrogate outputs. It does not establish causal treatment effects, wet-lab efficacy, safety, true cell-fate transitions, or physical energy landscapes.
