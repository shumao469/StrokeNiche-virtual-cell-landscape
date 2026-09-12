"""Reusable, data-agnostic components of the StrokeNiche virtual-cell workflow."""

from .classifier import ExpressionCounterfactualModel
from .landscape import LandscapeGrid, smooth_to_grid
from .context import build_section_knn_operator, make_context_design
from .operators import (
    MonotoneUpOperator,
    fit_monotone_up_operator,
    scale_log_expression,
)
from .response import (
    PerturbationEffect,
    classify_response,
    compute_susceptibility,
    simulate_counterfactual,
)

__all__ = [
    "ExpressionCounterfactualModel",
    "LandscapeGrid",
    "MonotoneUpOperator",
    "PerturbationEffect",
    "build_section_knn_operator",
    "classify_response",
    "compute_susceptibility",
    "fit_monotone_up_operator",
    "make_context_design",
    "scale_log_expression",
    "simulate_counterfactual",
    "smooth_to_grid",
]

__version__ = "0.3.0"
