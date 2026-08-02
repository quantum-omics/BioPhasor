"""biophasor.viz — merged plotting namespace for the BioPhasor platform.

Hosts the spectral-omics publication figures (migrated from
``spectralomics.viz.figures`` → ``biophasor.viz.figures``) alongside
biophasor's native ``biophasor.visualization.phasor_plot``.
"""

from biophasor.viz.figures import (
    plot_phasor_snapshot,
    plot_harmonic_timeline,
    plot_eigenvalue_heatmap,
    plot_ccm_heatmap,
    plot_compartment_weights,
    plot_indicator_dashboard,
    plot_state_history,
    plot_workflow,
)

__all__ = [
    "plot_phasor_snapshot",
    "plot_harmonic_timeline",
    "plot_eigenvalue_heatmap",
    "plot_ccm_heatmap",
    "plot_compartment_weights",
    "plot_indicator_dashboard",
    "plot_state_history",
    "plot_workflow",
]
