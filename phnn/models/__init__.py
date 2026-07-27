"""biophasor.phnn.models — port-Hamiltonian model stack.

Public API
----------
Generic_pHNN           : Full GNN-surrogate port-Hamiltonian model
AbundanceGNN_EnergyNet : H(x) Lyapunov energy network with sparse GNN layers
State_Dependent_R_Net  : k_deg-initialized dissipation operator R(x)
Sparse_Dynamic_J_Net   : Biologically sparse state-dependent skew J(x)
ModulatedPort_Net      : Zero-net-power enzymatic/feedback ports Γ(x)

Rollout / generation (migrated from the generative-omics manuscript):
vector_field, rk4_step, rollout        : deterministic RK4 integration of the pH field
conservation_projector, phsde_step,
phsde_rollout                          : port-Hamiltonian Langevin SDE generator
"""

from biophasor.phnn.models.phnn import (
    Generic_pHNN,
    State_Dependent_R_Net,
    Sparse_Dynamic_J_Net,
    ModulatedPort_Net,
)
from biophasor.phnn.models.energy_net import (
    AbundanceGNN_EnergyNet,
    NODE_FEAT_DIM,
)
from biophasor.phnn.models.integrator import (
    vector_field,
    rk4_step,
    rollout,
)
from biophasor.phnn.models.phsde import (
    conservation_projector,
    phsde_step,
    phsde_rollout,
)

__all__ = [
    "Generic_pHNN",
    "State_Dependent_R_Net",
    "Sparse_Dynamic_J_Net",
    "ModulatedPort_Net",
    "AbundanceGNN_EnergyNet",
    "NODE_FEAT_DIM",
    "vector_field",
    "rk4_step",
    "rollout",
    "conservation_projector",
    "phsde_step",
    "phsde_rollout",
]
