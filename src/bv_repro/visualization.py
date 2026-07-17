"""Visualization utilities for benchmark and experimental figures."""

from .benchmark_visualization import (
    load_benchmark_artifacts,
    plot_all,
    plot_experimental_prediction_comparison,
    plot_experimental_topk_heatmaps,
    plot_noise_robustness,
    plot_training_history,
    plot_uncertainty,
    print_model_metrics,
    print_model_metrics_latex,
    print_summary,
)
from .crystallographic_benchmark import plot_uncertainty_distributions
from .edf_rocking_roi import (
    LayerFrameROISelector,
    RockingCurveROIViewer,
    make_viewer_for_frame,
    plot_layer_frame_previews,
)

__all__ = [
    "load_benchmark_artifacts",
    "plot_all",
    "plot_experimental_prediction_comparison",
    "plot_experimental_topk_heatmaps",
    "plot_noise_robustness",
    "plot_training_history",
    "plot_uncertainty",
    "print_model_metrics",
    "print_model_metrics_latex",
    "print_summary",
    "plot_uncertainty_distributions",
    "LayerFrameROISelector",
    "RockingCurveROIViewer",
    "make_viewer_for_frame",
    "plot_layer_frame_previews",
]

