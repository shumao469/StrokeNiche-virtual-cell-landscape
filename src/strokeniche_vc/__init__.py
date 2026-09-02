"""Reusable, data-agnostic components of the StrokeNiche virtual-cell workflow."""

from .classifier import ExpressionCounterfactualModel
from .landscape import LandscapeGrid, smooth_to_grid
from .response import (
    PerturbationEffect,
    classify_response,
    compute_susceptibility,
    simulate_counterfactual,
)

__all__ = [
    "ExpressionCounterfactualModel",
    "LandscapeGrid",
    "PerturbationEffect",
    "classify_response",
    "compute_susceptibility",
    "simulate_counterfactual",
    "smooth_to_grid",
]

__version__ = "0.2.0"
