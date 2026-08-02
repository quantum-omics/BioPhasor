"""
biophasor.spectral.quantum — second-quantised representation of the omics
connectome spectrum.

What is actually modelled
-------------------------
The classical layer of ``biophasor.spectral`` reduces an expression matrix to a
Hermitian Omics Connectome Matrix, ``H_ij = c_ij e^{i(theta_i - theta_j)}``,
whose eigenvalues ``lambda_k`` define collective harmonic frequencies
``omega_k = sqrt(|lambda_k|)``. That spectrum is a static object: a list of
eigenvalues, with no notion of excitation, interaction, or a ground state.

This subpackage promotes it to a second-quantised model. Each of the leading
harmonics is treated as one bosonic mode; the occupation number ``n_k`` of that
mode is the quantum of collective activity carried by that harmonic; and the
modes are coupled by a bounded, particle-number-conserving Bose-Hubbard
Hamiltonian. The five leading harmonics carry the interpretive labels Clock,
Redox, Energy, Signalling and Biosynthesis — an explicit overlay on the five
dominant modes, not a mechanistic decomposition of cellular physiology.

The occupation numbers are the representation's variables; they are not counts
of molecules, and nothing here asserts that a cell is a quantum system or that
biological systems perform quantum computation. This is a formalism applied to
the omics correlation spectrum, in the same sense that a normal-mode expansion
is a formalism applied to a coupled-oscillator network.

What the representation buys
----------------------------
Three things that the raw eigenvalue list does not provide.

1. A bounded model with a genuine ground state. A Hamiltonian assembled naively
   from odd-order ladder products (``interactions.InteractionHamiltonians.cubic_reference``,
   kept here precisely so the failure is demonstrable) does not conserve
   excitation number and is unbounded below, so its "ground state" is an
   artifact of wherever the Fock truncation was placed. The even-order Kerr term
   of the Bose-Hubbard form removes that: for ``U >= 0`` the spectrum is bounded
   below and the ground state is a property of the model rather than of the
   cutoff.
2. Exact, cutoff-independent observables. Because ``[H, N] = 0``, the dynamics
   stays inside a fixed total-excitation sector, and that sector can be
   diagonalised exactly at a dimension far below the full Fock space. Raising
   the per-mode cutoff leaves sector observables unchanged, which is checkable
   and is checked.
3. A positive semi-definite readout. The compartment covariance matrix (CCM) is
   the quantum covariance of the five compartment observables in a state, so it
   is PSD by construction. Compartment weights, a dominance ranking, and the
   coherence measure ``kappa = ||diag(M)||_2 / ||M||_F`` in [0, 1] then follow
   without any normalisation chosen after the fact.

Layout
------
The upstream ``spectralomicsquantum`` package split these modules into
``quantum/`` and ``compartments/`` subdirectories. That split is flattened here:
as a subpackage the nesting would read ``biophasor.spectral.quantum.quantum.fock_space``,
and nine modules do not need a second level of grouping. The two compartment
modules that would have collided under flattening are renamed
(``compartments.covariance`` -> ``compartment_covariance``,
``compartments.weights`` -> ``compartment_weights``).

Note that ``compartment_weights.CompartmentWeights`` here is NOT the classical
``biophasor.spectral.omics.compartment_weights.CompartmentWeights``. The
classical one reads the eigenvector-projection weights off the OCM; this one
reads the diagonal of the quantum CCM. They answer the same question in the two
layers and are deliberately not unified — import them by qualified path.

``duality`` predates the promotion and is unrelated to the Fock-space
construction: it is the gate-level correspondence between classical phasor
operations and single-/two-qubit unitaries.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from biophasor.spectral.quantum.omics_spectrum import (
    ocm_spectrum,
    omics_harmonic_frequencies,
    compartment_self_energies,
)
from biophasor.spectral.quantum.fock_space import FockSpace
from biophasor.spectral.quantum.hamiltonian import OmicsHamiltonian
from biophasor.spectral.quantum.interactions import InteractionHamiltonians
from biophasor.spectral.quantum.dynamics import QuantumDynamics
from biophasor.spectral.quantum.compartment_model import (
    CompartmentModel,
    COMPARTMENTS,
    COMPARTMENT_EDGES,
)
from biophasor.spectral.quantum.compartment_covariance import CompartmentCovariance
from biophasor.spectral.quantum.compartment_weights import CompartmentWeights
from biophasor.spectral.quantum.duality import (
    rz_matrix,
    qft_matrix,
    phasor_to_statevector,
)

__all__ = [
    # classical -> quantum bridge
    "ocm_spectrum",
    "omics_harmonic_frequencies",
    "compartment_self_energies",
    # Fock-space algebra and Hamiltonians
    "FockSpace",
    "OmicsHamiltonian",
    "InteractionHamiltonians",
    "QuantumDynamics",
    # Bose-Hubbard compartment model and its readout
    "CompartmentModel",
    "COMPARTMENTS",
    "COMPARTMENT_EDGES",
    "CompartmentCovariance",
    "CompartmentWeights",
    # phasor <-> gate duality (independent of the Fock construction)
    "rz_matrix",
    "qft_matrix",
    "phasor_to_statevector",
]

# ``figures`` is not imported here: it pulls in matplotlib, and importing
# biophasor.spectral.quantum should not require a plotting backend. Import
# biophasor.spectral.quantum.figures explicitly when you need it.
