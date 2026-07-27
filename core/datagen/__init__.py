"""
biophasor.core.datagen — synthetic multi-omics generation, real-data adapters,
rhythmicity detection, and compartment/state assembly.

Migrated from the former phnn-omics ``data/`` package; self-contained (no
cross-domain coupling). Shared by biophasor.phnn and available platform-wide.
"""

from biophasor.core.datagen.omics_data_generator import (
    generate_multi_omics, get_total_nodes, get_layer_slices,
    LAYER_CONFIG,
)
from biophasor.core.datagen.data_adapter import (
    validate_omics_data, load_synthetic_omics, load_real_omics, get_omics_data,
)
from biophasor.core.datagen.rhythmicity_gate import (
    detect_rhythmicity, detect_all_layers,
)
from biophasor.core.datagen.two_layer_state import (
    assemble_two_layer_state, verify_conservation,
)
from biophasor.core.datagen.compartments import build_compartments

__all__ = [
    "generate_multi_omics", "get_total_nodes", "get_layer_slices", "LAYER_CONFIG",
    "validate_omics_data", "load_synthetic_omics", "load_real_omics", "get_omics_data",
    "detect_rhythmicity", "detect_all_layers",
    "assemble_two_layer_state", "verify_conservation",
    "build_compartments",
]
