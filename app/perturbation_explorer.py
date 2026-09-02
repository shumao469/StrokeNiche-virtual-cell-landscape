"""Interactive gene/drug perturbation explorer for StrokeNiche virtual cells."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strokeniche_vc.demo import make_demo_cells, make_demo_effects
from strokeniche_vc.io import normalize_cell_table, normalize_effect_table
from strokeniche_vc.landscape import smooth_to_grid
from strokeniche_vc.response import PerturbationEffect, classify_response, simulate_counterfactual


STATE_COLORS = {
    "lesion-core-like": "#ef5b5b",
    "peri-infarct": "#f2b134",
    "remote-like": "#39a9db",
    "unassigned": "#8b95a5",
}
PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["lasso2d"],
}
MIN_LANDSCAPE_OBSERVATIONS = 3


def has_minimum_landscape_support(frame: pd.DataFrame) -> bool:
    """Return whether a filtered view can support 2-D landscape smoothing."""

    return len(frame) >= MIN_LANDSCAPE_OBSERVATIONS


def page_style() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f7f9fc; }
        .block-container { max-width: 1500px; padding-top: 1.4rem; }
        [data-testid="stMetric"] {
          background: white; border: 1px solid #e2e8f0; border-radius: 14px;
          padding: 0.75rem 1rem; box-shadow: 0 4px 14px rgba(15,23,42,.04);
        }
        .hero {
          padding: 1.15rem 1.4rem; border-radius: 18px;
          background: linear-gradient(120deg,#102a43,#1f5f78 52%,#2f8f83);
          color: white; margin-bottom: 1rem;
          box-shadow: 0 12px 30px rgba(16,42,67,.18);
        }
        .hero h1 { margin: 0; font-size: 2rem; }
        .hero p { margin: .45rem 0 0; color: #d9f0f0; }
        .science-note {
          background: #fff8e6; border-left: 5px solid #e0a11a;
          padding: .8rem 1rem; border-radius: 8px; color: #5b4714;
        }
        .pipeline {
          display: grid; grid-template-columns: repeat(5,1fr); gap: .55rem;
          align-items: center; margin: .5rem 0 1rem;
        }
        .pipeline div {
          background: white; border: 1px solid #dbe5ee; border-radius: 12px;
          padding: .65rem; text-align: center; color: #22364a; font-weight: 600;
        }
        @media (max-width:900px){ .pipeline{grid-template-columns:1fr;} }
        </style>
        """,
        unsafe_allow_html=True,
    )


def plot_theme(fig: go.Figure, height: int = 560) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=40, r=25, t=65, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial", color="#26384a"),
        hoverlabel=dict(bgcolor="white", font_size=12),
        legend=dict(bgcolor="rgba(255,255,255,.85)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#edf1f5", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#edf1f5", zeroline=False)
    return fig


def scatter_trace(
    table: pd.DataFrame,
    color: str,
    *,
    name: str,
    colorbar_title: str,
    showscale: bool,
) -> go.Scattergl:
    hover = np.column_stack(
        [
            table["obs_name"].astype(str),
            table["state"].astype(str),
            table["baseline_core"],
            table["baseline_repair"],
            table["perturbed_core"],
            table["perturbed_repair"],
        ]
    )
    return go.Scattergl(
        x=table["latent1"],
        y=table["latent2"],
        mode="markers",
        name=name,
        customdata=hover,
        marker=dict(
            size=5,
            opacity=0.72,
            color=table[color],
            colorscale="Magma",
            cmin=0,
            cmax=1,
            showscale=showscale,
            colorbar=dict(title=colorbar_title, thickness=13) if showscale else None,
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>State: %{customdata[1]}"
            "<br>Baseline core: %{customdata[2]:.3f}"
            "<br>Baseline repair: %{customdata[3]:.3f}"
            "<br>Perturbed core: %{customdata[4]:.3f}"
            "<br>Perturbed repair: %{customdata[5]:.3f}<extra></extra>"
        ),
    )


def latent_comparison(table: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Baseline lesion-core probability", "After perturbation"),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        scatter_trace(table, "baseline_core", name="Baseline", colorbar_title="Core probability", showscale=False),
        row=1,
        col=1,
    )
    fig.add_trace(
        scatter_trace(table, "perturbed_core", name="Perturbed", colorbar_title="Core probability", showscale=True),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Latent coordinate 1")
    fig.update_yaxes(title_text="Latent coordinate 2")
    fig.update_layout(title="Baseline vs counterfactual score on fixed latent coordinates", showlegend=False)
    return plot_theme(fig, 600)


def response_landscapes(
    table: pd.DataFrame,
    grid_n: int,
    sigma: float,
    locked_scales: dict[str, float],
) -> go.Figure:
    metrics = [
        ("core_reduction", "Core reduction", "Blues"),
        ("repair_gain", "Repair gain", "Greens"),
        ("response_priority", "Response priority", "Viridis"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[m[1] for m in metrics], horizontal_spacing=0.07)
    for col, (metric, title, scale) in enumerate(metrics, start=1):
        grid = smooth_to_grid(
            table["latent1"].to_numpy(),
            table["latent2"].to_numpy(),
            table[metric].to_numpy(),
            grid_n=grid_n,
            sigma=sigma,
        )
        fig.add_trace(
            go.Heatmap(
                x=grid.x,
                y=grid.y,
                z=grid.values,
                zmin=0,
                zmax=locked_scales[metric],
                colorscale=scale,
                colorbar=dict(
                    title=title,
                    thickness=10,
                    len=0.72,
                    x={1: 0.295, 2: 0.652, 3: 1.01}[col],
                ),
                hovertemplate="L1=%{x:.2f}<br>L2=%{y:.2f}<br>Value=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="Latent coordinate 1")
    fig.update_yaxes(title_text="Latent coordinate 2")
    fig.update_layout(title="Density-aware response landscapes")
    fig = plot_theme(fig, 560)
    fig.update_layout(margin=dict(l=40, r=95, t=65, b=45))
    return fig


def state_shift_plot(table: pd.DataFrame) -> go.Figure:
    summary = (
        table.groupby("state", dropna=False)
        .agg(
            baseline_core=("baseline_core", "mean"),
            perturbed_core=("perturbed_core", "mean"),
            baseline_repair=("baseline_repair", "mean"),
            perturbed_repair=("perturbed_repair", "mean"),
            mean_priority=("response_priority", "mean"),
            n=("obs_name", "size"),
        )
        .reset_index()
    )
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Lesion-core probability", "Repair score"))
    for col, pair in enumerate(
        [("baseline_core", "perturbed_core"), ("baseline_repair", "perturbed_repair")],
        start=1,
    ):
        fig.add_trace(
            go.Bar(x=summary["state"], y=summary[pair[0]], name="Baseline", marker_color="#64748b", legendgroup="baseline", showlegend=col == 1),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Bar(x=summary["state"], y=summary[pair[1]], name="Perturbed", marker_color="#17a398", legendgroup="perturbed", showlegend=col == 1),
            row=1,
            col=col,
        )
    fig.update_yaxes(range=[0, 1], title_text="Mean score")
    fig.update_layout(title="State-stratified counterfactual summary", barmode="group")
    return plot_theme(fig, 520)


def spatial_response_plot(table: pd.DataFrame, locked_scales: dict[str, float]) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Core reduction", "Repair gain"), horizontal_spacing=0.08)
    for col, metric, scale, title in [
        (1, "core_reduction", "Blues", "Core reduction"),
        (2, "repair_gain", "Greens", "Repair gain"),
    ]:
        fig.add_trace(
            go.Scattergl(
                x=table["spatial_x"],
                y=table["spatial_y"],
                mode="markers",
                customdata=np.column_stack([table["obs_name"], table["state"], table[metric]]),
                marker=dict(
                    size=5,
                    color=table[metric],
                    colorscale=scale,
                    cmin=0,
                    cmax=locked_scales[metric],
                    opacity=0.82,
                    showscale=True,
                    colorbar=dict(title=title, thickness=10, x=0.47 if col == 1 else 1.01),
                ),
                hovertemplate="<b>%{customdata[0]}</b><br>State: %{customdata[1]}<br>Response: %{customdata[2]:.4f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="Spatial x")
    fig.update_yaxes(title_text="Spatial y")
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_yaxes(autorange="reversed", scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(title="Spatial response projection")
    fig = plot_theme(fig, 600)
    fig.update_layout(margin=dict(l=40, r=90, t=65, b=45))
    return fig


def strength_curve(
    cells: pd.DataFrame,
    effect: PerturbationEffect,
    max_strength: float,
    selected_index: pd.Index | None = None,
) -> go.Figure:
    strengths = np.linspace(0, max(0.25, max_strength), 25)
    rows = []
    for value in strengths:
        simulated = simulate_counterfactual(cells, effect, strength=float(value))
        if selected_index is not None:
            simulated = simulated.loc[selected_index]
        rows.append(
            {
                "strength": value,
                "mean_core": simulated["perturbed_core"].mean(),
                "mean_repair": simulated["perturbed_repair"].mean(),
                "mean_priority": simulated["response_priority"].mean(),
            }
        )
    curve = pd.DataFrame(rows)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["strength"], y=curve["mean_core"], mode="lines+markers", name="Mean core probability", line=dict(color="#d1495b", width=3)))
    fig.add_trace(go.Scatter(x=curve["strength"], y=curve["mean_repair"], mode="lines+markers", name="Mean repair score", line=dict(color="#17a398", width=3)))
    fig.add_trace(go.Scatter(x=curve["strength"], y=curve["mean_priority"], mode="lines", name="Mean response priority", line=dict(color="#3f51b5", width=2, dash="dot")))
    fig.update_xaxes(title="Relative perturbation strength")
    fig.update_yaxes(title="Mean score", range=[0, 1])
    fig.update_layout(title="Relative expression/effect-strength sensitivity")
    return plot_theme(fig, 520)


def read_csv_upload(uploaded_file, *, max_bytes: int = 100 * 1024 * 1024, max_rows: int = 250_000) -> pd.DataFrame:
    if uploaded_file.size > max_bytes:
        raise ValueError(f"{uploaded_file.name} exceeds the 100 MB local-app limit")
    # Read every upload field as text first so semantic identifiers such as
    # 001 are never rewritten as 1. Schema normalizers convert numeric columns.
    table = pd.read_csv(io.BytesIO(uploaded_file.getvalue()), dtype=str)
    if len(table) > max_rows:
        raise ValueError(f"{uploaded_file.name} exceeds the 250,000-row local-app limit")
    return table


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    st.sidebar.header("1 · Data source")
    source = st.sidebar.radio(
        "Choose input",
        ["Synthetic demo", "Upload baseline cells + effects"],
        help="The repository contains no patient-level or study-level data.",
    )
    if source == "Synthetic demo":
        n_cells = st.sidebar.slider("Demo cells", 600, 4000, 1800, 200)
        return make_demo_cells(n=n_cells), normalize_effect_table(make_demo_effects()), "synthetic_demo"

    cells_file = st.sidebar.file_uploader("Baseline cell CSV", type="csv")
    effects_file = st.sidebar.file_uploader("Perturbation effect CSV", type="csv")
    st.sidebar.warning(
        "Use study or patient-derived tables only in an institutionally governed local deployment. "
        "Do not upload them to a public Streamlit host."
    )
    with st.sidebar.expander("Required columns"):
        st.code(
            "Cell table:\nlatent1, latent2, baseline_core, baseline_repair\n"
            "Optional: obs_name, state, timepoint, spatial_x, spatial_y\n\n"
            "Effect table:\nperturbation, delta_core, delta_repair\n"
            "Optional: modality, target_genes, source",
            language="text",
        )
    if cells_file is None or effects_file is None:
        st.info("Upload both CSV files in the sidebar. A downloadable template is available below.")
        st.download_button(
            "Download effect-table template",
            make_demo_effects().head(0).to_csv(index=False),
            "perturbation_effect_template.csv",
            "text/csv",
        )
        st.stop()
    return normalize_cell_table(read_csv_upload(cells_file)), normalize_effect_table(read_csv_upload(effects_file)), "uploaded"


def main() -> None:
    st.set_page_config(page_title="StrokeNiche Virtual Cell Explorer", page_icon="🧬", layout="wide")
    page_style()
    st.markdown(
        """
        <div class="hero">
          <h1>StrokeNiche Virtual Cell Explorer</h1>
          <p>Interactive display of precomputed gene and target-bridged drug-effect proxies across latent and spatial context</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cells, effects, data_mode = load_inputs()
    for optional_context in ("state", "timepoint", "dominant_celltype"):
        if optional_context in cells.columns:
            cells[optional_context] = (
                cells[optional_context].astype("string").fillna("(missing)").astype(str)
            )
    st.sidebar.header("2 · Perturbation")
    modality_labels = {
        "gene": "Gene / pathway",
        "drug_proxy": "Drug-mechanism proxy",
    }
    available_modalities = list(dict.fromkeys(effects["modality"].tolist()))
    modality = st.sidebar.selectbox(
        "Modality",
        available_modalities,
        format_func=lambda x: modality_labels.get(x, x),
    )
    candidates = effects.loc[effects["modality"] == modality, "perturbation"].tolist()
    selected = st.sidebar.selectbox("Candidate", candidates)
    strength = st.sidebar.slider("Relative strength", 0.0, 2.0, 1.0, 0.05)
    grid_n = st.sidebar.slider("Landscape resolution", 45, 150, 90, 5)
    sigma = st.sidebar.slider("Smoothing", 0.8, 4.0, 2.2, 0.1)

    st.sidebar.header("3 · Context filters")
    filtered_cells = cells
    for column, label in [
        ("state", "State"),
        ("timepoint", "Timepoint"),
        ("dominant_celltype", "Cell type"),
    ]:
        if column in filtered_cells.columns:
            values = sorted(filtered_cells[column].dropna().astype(str).unique().tolist())
            selected_values = st.sidebar.multiselect(label, values, default=values)
            filtered_cells = filtered_cells.loc[
                filtered_cells[column].astype(str).isin(selected_values)
            ]
    if filtered_cells.empty:
        st.warning("The current context filters select no virtual cells.")
        st.stop()

    effect_row = effects.loc[effects["perturbation"] == selected].iloc[0].to_dict()
    effect = PerturbationEffect.from_mapping(effect_row)
    # Susceptibility is computed once against the complete uploaded baseline so
    # the same virtual cell does not change merely because a display filter does.
    response_all = simulate_counterfactual(cells, effect, strength=strength)
    response = response_all.loc[filtered_cells.index]
    response_class = classify_response(response["delta_core"], response["delta_repair"])
    response["response_class"] = response_class
    max_strength = 2.0
    core_scale = min(1.0, max_strength * float(np.maximum(-effects["delta_core"], 0).max()))
    repair_scale = min(1.0, max_strength * float(np.maximum(effects["delta_repair"], 0).max()))
    locked_scales = {
        "core_reduction": max(core_scale, 1e-6),
        "repair_gain": max(repair_scale, 1e-6),
        "response_priority": max(0.55 * core_scale + 0.45 * repair_scale, 1e-6),
    }

    st.markdown(
        '<div class="pipeline"><div>Baseline cells</div><div>Context susceptibility</div><div>Gene / drug effect</div><div>Counterfactual state</div><div>Response landscape</div></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Virtual cells", f"{len(response):,}")
    c2.metric("Mean core change", f"{response['delta_core'].mean():+.3f}")
    c3.metric("Mean repair change", f"{response['delta_repair'].mean():+.3f}")
    c4.metric("Mean priority", f"{response['response_priority'].mean():.3f}")
    class_labels = {
        "rescue_positive": "Rescue",
        "weak_or_reverse": "Reverse/adverse",
        "mixed_tradeoff": "Mixed",
        "near_zero": "Near zero",
    }
    c5.metric("Response class", class_labels.get(response_class, response_class))

    st.caption(
        f"Targets: {effect.target_genes or 'not supplied'} · Effect source: {effect.source} · Data mode: {data_mode}"
    )
    if effect.proxy_from_perturbation:
        st.caption(f"Drug proxy inherits numeric effects from: {effect.proxy_from_perturbation}")
    if data_mode == "synthetic_demo":
        st.markdown(
            '<div class="science-note"><b>Demo mode:</b> cell coordinates and all effect magnitudes are synthetic. '
            'Gene axes and drug-mechanism proxies are illustrative, not experimental evidence or treatment recommendations. '
            'Upload model-exported cell and effect tables for study-specific visualization.</div>',
            unsafe_allow_html=True,
        )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["Latent before/after", "Response landscapes", "State summary", "Spatial map", "Strength sensitivity", "Model card"]
    )
    with tab1:
        st.plotly_chart(latent_comparison(response), width="stretch", config=PLOT_CONFIG)
        st.caption("The latent coordinates are held fixed; marker colour shows the recomputed score, not latent movement.")
    with tab2:
        if has_minimum_landscape_support(response):
            st.plotly_chart(
                response_landscapes(response, grid_n, sigma, locked_scales),
                width="stretch",
                config=PLOT_CONFIG,
            )
            st.caption(
                "Low-density regions are masked. Colour limits are locked across all candidates and strengths in the uploaded effect table; "
                "surfaces are response proxies, not physical energy."
            )
        else:
            st.info(
                "At least three virtual cells are required for a smoothed landscape. "
                "The other filtered views remain available."
            )
    with tab3:
        st.plotly_chart(state_shift_plot(response), width="stretch", config=PLOT_CONFIG)
        summary = response.groupby("state", dropna=False).agg(
            n=("obs_name", "size"),
            mean_delta_core=("delta_core", "mean"),
            mean_delta_repair=("delta_repair", "mean"),
            mean_response_priority=("response_priority", "mean"),
        )
        st.dataframe(summary.reset_index(), width="stretch", hide_index=True)
    with tab4:
        if {"spatial_x", "spatial_y"}.issubset(response.columns):
            st.plotly_chart(
                spatial_response_plot(response, locked_scales),
                width="stretch",
                config=PLOT_CONFIG,
            )
        else:
            st.info("Add spatial_x and spatial_y to the cell CSV to enable this view.")
    with tab5:
        st.plotly_chart(
            strength_curve(cells, effect, max(2.0, strength), filtered_cells.index),
            width="stretch",
            config=PLOT_CONFIG,
        )
        st.caption("The x-axis is a relative computational scaling factor, not a pharmacological dose.")
    with tab6:
        st.subheader("Transparent response model")
        st.latex(r"\tilde c_i=\operatorname{RMM}(c_i),\quad \tilde r_i=\operatorname{RMM}(r_i)")
        st.latex(r"s_i = 0.25 + 0.75\,\operatorname{RMM}(0.65\tilde c_i + 0.35(1-\tilde r_i))")
        st.latex(r"c_i' = \operatorname{clip}(c_i + \alpha\Delta c\,s_i,0,1),\quad r_i' = \operatorname{clip}(r_i + \alpha\Delta r\,s_i,0,1)")
        st.latex(r"p_i = 0.55\max(-\Delta c_i,0)+0.45\max(\Delta r_i,0)")
        st.markdown(
            "- $c_i$: baseline lesion-core probability for virtual cell $i$.\n"
            "- $r_i$: baseline repair score.\n"
            "- $s_i$: context susceptibility derived from baseline state.\n"
            "- $\\Delta c, \\Delta r$: candidate-level mean effects supplied by the model or an effect table.\n"
            "- $\\alpha$: user-selected relative perturbation strength."
        )
        st.caption("RMM denotes robust 1st–99th percentile min-max scaling, applied first to each baseline score and again to their weighted combination.")
        st.caption("The 0.55/0.45 priority weights are heuristic display weights, not learned or clinically calibrated utility weights.")
        st.warning(
            "Interpretation boundary: this is a computational counterfactual/surrogate visualization. "
            "It is not wet-lab validation, a causal treatment effect, a physical energy landscape, "
            "or an observed cell-fate transition."
        )
        st.info(
            "This release visualizes uploaded candidate-level delta_core/delta_repair values. "
            "It does not run the expression classifier or graph adapter live, and it cannot infer an arbitrary new gene or drug."
        )
        st.markdown(
            "**Evidence ladder used by this project**\n\n"
            "1. Observed baseline expression and spatial context.\n"
            "2. Classifier counterfactual after a defined gene-expression edit.\n"
            "3. Step78D smoothed response proxy for display.\n"
            "4. Drug-to-target bridge, labelled as a target-level proxy.\n"
            "5. LINCS signature reversal as external transcriptional evidence."
        )

    st.divider()
    export_cols = [
        "obs_name", "state", "latent1", "latent2", "baseline_core", "baseline_repair",
        "perturbation", "modality", "target_genes", "effect_source", "proxy_from_perturbation", "strength", "susceptibility",
        "perturbed_core", "perturbed_repair", "delta_core", "delta_repair",
        "core_reduction", "repair_gain", "response_priority", "response_class",
    ]
    st.download_button(
        "Download current counterfactual table",
        response[[c for c in export_cols if c in response.columns]].to_csv(index=False),
        "strokeniche_virtual_cell_response.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
