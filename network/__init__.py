"""
biophasor.network — biological graph networks.

The canonical implementation now lives in :mod:`biophasor.core.graph`
(migrated from phnn-omics). This module re-exports it.
"""
from biophasor.core.graph import (  # noqa: F401
    build_biological_graph, build_compartment_structure,
    compute_plv_prior, print_graph_summary,
)

__all__ = [
    "build_biological_graph", "build_compartment_structure",
    "compute_plv_prior", "print_graph_summary",
]
