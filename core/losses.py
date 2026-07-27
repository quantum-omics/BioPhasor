"""
biophasor.core.losses — unified loss library.

Two disjoint families, both shared platform-wide:

* **Circular family** (phase-native, from biophasor.ml.losses):
  circular_mse_loss, coherence_loss, von_mises_kl_loss.
* **Physics family** (port-Hamiltonian / conservation, from phnn training.losses):
  loss_passivity, loss_power_balance, loss_conservation, loss_coherence_prior,
  loss_homeostasis, loss_kinematic, compute_composite_loss, etc.

`loss_coherence_prior` (physics) and `coherence_loss` (circular) are distinct:
the former is a PLV-prior penalty used in pHNN training, the latter maximises
Kuramoto order. Both retained under their original names.
"""
from biophasor.ml.losses import (  # circular family
    circular_mse_loss, coherence_loss, von_mises_kl_loss,
)
from biophasor.core._physics_losses import (  # physics family
    loss_kinematic, loss_passivity, loss_passivity_per_compartment,
    loss_power_balance, loss_coherence_prior, loss_conservation,
    loss_homeostasis, compute_composite_loss, LOSS_WEIGHTS,
)

__all__ = [
    "circular_mse_loss", "coherence_loss", "von_mises_kl_loss",
    "loss_kinematic", "loss_passivity", "loss_passivity_per_compartment",
    "loss_power_balance", "loss_coherence_prior", "loss_conservation",
    "loss_homeostasis", "compute_composite_loss", "LOSS_WEIGHTS",
]
