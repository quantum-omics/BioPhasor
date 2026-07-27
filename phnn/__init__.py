"""biophasor.phnn — port-Hamiltonian neural network domain.

Migrated from the standalone 3-phnn-omics project. Shared pieces (data
generation, biological graph, physics losses) now live in biophasor.core
and are imported from there; this subpackage holds the pHNN-specific model
stack, training pipeline, and validation utilities.
"""

from biophasor.phnn.models import (
    Generic_pHNN,
    AbundanceGNN_EnergyNet,
    State_Dependent_R_Net,
    Sparse_Dynamic_J_Net,
    ModulatedPort_Net,
    NODE_FEAT_DIM,
)
from biophasor.phnn.utils import (
    CascadePredictor,
    verify_passivity,
    evaluate_edge_recovery,
    evaluate_held_out_perturbation,
    print_validation_report,
)

__all__ = [
    "Generic_pHNN",
    "AbundanceGNN_EnergyNet",
    "State_Dependent_R_Net",
    "Sparse_Dynamic_J_Net",
    "ModulatedPort_Net",
    "NODE_FEAT_DIM",
    "CascadePredictor",
    "verify_passivity",
    "evaluate_edge_recovery",
    "evaluate_held_out_perturbation",
    "print_validation_report",
]
