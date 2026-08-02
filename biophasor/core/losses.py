"""
biophasor.core.losses — the circular (phase-native) loss family.

Training objectives for models whose prediction target is an angle rather than
a magnitude. All three treat phase as living on S¹, so they are invariant to
the 2π wrap that an ordinary MSE on raw angles gets wrong: a prediction of
0.01 rad against a target of 6.27 rad is a small error, not a large one.

    circular_mse_loss  mean(1 − cos Δφ); the wrap-safe substitute for MSE
    coherence_loss     1 − R², R = |mean e^{iφ}|; drives a population toward
                       a common phase (maximises Kuramoto order)
    von_mises_kl_loss  KL against a von Mises target; used when the model
                       predicts a phase *distribution* (μ, κ), not a point

These are re-exported from :mod:`biophasor.ml.losses`, which holds the
implementations. This module is the stable import path for them.

Physics family — removed from this repository
─────────────────────────────────────────────
This module previously also carried a port-Hamiltonian family —
``loss_passivity``, ``loss_passivity_per_compartment``, ``loss_power_balance``,
``loss_conservation``, ``loss_coherence_prior``, ``loss_homeostasis``,
``loss_kinematic``, ``compute_composite_loss`` and the ``LOSS_WEIGHTS``
dictionary. That family now lives in ``cvomics.training.losses`` in the
Classical-Virtual-Omics repository, which is the single home for the
port-Hamiltonian work. The implementation as it stood here is retained
maintained in ``cvomics.training.losses``; do not import
it. Nothing else in ``biophasor`` referenced it.

The trap that motivated splitting the two families rather than merging them:
``loss_coherence_prior`` (physics) and ``coherence_loss`` (circular) sound
interchangeable and are not, in either sign or role.

* ``loss_coherence_prior`` is a *weak prior* (λ = 0.01) pulling the learned
  cross-layer entries of the pHNN interconnection matrix J toward a phase-
  locking-value matrix computed from the same data. It is deliberately weak
  precisely so that PLV recovery stays a testable outcome of the fitted
  connectome; raising λ trains J toward a statistic of its own training data
  and the recovery result becomes circular.
* ``coherence_loss`` is a *fitting objective* that maximises the Kuramoto order
  parameter of a predicted phase population. There is no prior and no external
  reference matrix; it acts on the phases themselves.

One regularises a matrix against an external target, the other optimises a
population statistic to its extremum. Substituting either for the other
silently changes what is being fitted, which is why both kept their original
names when the families were briefly co-located here.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""
from biophasor.ml.losses import (
    circular_mse_loss, coherence_loss, von_mises_kl_loss,
)

__all__ = [
    "circular_mse_loss", "coherence_loss", "von_mises_kl_loss",
]
