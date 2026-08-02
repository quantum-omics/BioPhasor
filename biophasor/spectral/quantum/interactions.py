"""
interactions.py — Interaction Hamiltonians for the omics quantum model
=======================================================================

This module builds the interaction terms that promote the free omics
Hamiltonian to an interacting many-body model.  All operators are expressed
as dense matrices in the full truncated Fock basis (dimension =
(max_occupation+1)^n_modes).

Two families are provided:

1. **Bounded, number-conserving Bose--Hubbard terms** — the feasible
   many-body model:

       * ``coherent_hopping``      Σ_{i,j} g_ij (a†_i a_j + a_i a†_j)
       * ``onsite_kerr``           (1/2) Σ_k U_k n_k (n_k − 1)
       * ``density_density``       Σ_{i,j} V_ij n_i n_j

   Each conserves total excitation number ``N̂ = Σ_k n_k`` and (for
   ``U_k >= 0``) is bounded below, so a genuine ground state exists.

2. **Unbounded odd-order cubic reference** — ``cubic_reference``, kept for
   contrast.  Its three-operator products ``a†_i a†_j a_k`` do NOT conserve
   excitation number and make the spectrum unbounded below; the truncated
   "ground state" is then a cutoff artifact.  It is provided only to show
   why the bounded Bose--Hubbard model is the physically feasible choice.

Notation
--------
a†_i, a_i  — creation / annihilation operators for mode i
n_i = a†_i a_i — number operator for mode i
h.c.       — Hermitian conjugate

The returned matrices are always Hermitian because each term is built to
include its own h.c.  Units: hbar = 1 throughout.

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import numpy as np

from biophasor.spectral.quantum.fock_space import FockSpace


class InteractionHamiltonians:
    """Interaction Hamiltonian terms in the omics Fock space.

    Parameters
    ----------
    fock_space : FockSpace
        Shared Fock-space object defining the Hilbert space.  All returned
        matrices live in the same space with dimension fock_space.dim().
    """

    def __init__(self, fock_space: FockSpace) -> None:
        self.fock_space = fock_space
        self._n = fock_space.n_modes
        self._dim = fock_space.dim()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_ops(self):
        """Cache and return lists of creation/annihilation/number operators."""
        if not hasattr(self, "_adag"):
            fs = self.fock_space
            self._adag = [self._to_dense(fs.creation_op(k)) for k in range(self._n)]
            self._a = [self._to_dense(fs.annihilation_op(k)) for k in range(self._n)]
            self._n_op = [self._to_dense(fs.number_op(k)) for k in range(self._n)]
        return self._adag, self._a, self._n_op

    @staticmethod
    def _to_dense(op) -> np.ndarray:
        """Convert sparse matrix to dense if needed."""
        if hasattr(op, "toarray"):
            return op.toarray().astype(complex)
        return np.asarray(op, dtype=complex)

    def _zero(self) -> np.ndarray:
        return np.zeros((self._dim, self._dim), dtype=complex)

    def _make_hermitian(self, H: np.ndarray) -> np.ndarray:
        """Symmetrise a matrix to enforce exact Hermiticity."""
        return 0.5 * (H + H.conj().T)

    def _check_shape(self, arr: np.ndarray, expected_shape, name: str) -> None:
        if arr.shape != expected_shape:
            raise ValueError(
                f"Parameter '{name}' has shape {arr.shape}, "
                f"expected {expected_shape}."
            )

    # ==================================================================
    # 1. Bounded, number-conserving Bose--Hubbard terms
    # ==================================================================

    def coherent_hopping(self, g: np.ndarray) -> np.ndarray:
        """Coherent hopping (beam-splitter) coupling between omics modes.

        H = sum_{i,j} g_{ij} * (a†_i a_j + a_i a†_j)

        Conserves the total number of excitations and describes coherent
        redistribution of activity between omics compartments.

        Parameters
        ----------
        g : np.ndarray, shape (n_modes, n_modes)
            Hopping matrix.  Off-diagonals give inter-mode coupling.
        """
        g = np.asarray(g, dtype=complex)
        self._check_shape(g, (self._n, self._n), "g")
        adag, a, _ = self._get_ops()
        H = self._zero()
        for i in range(self._n):
            for j in range(self._n):
                if g[i, j] != 0:
                    H += g[i, j] * (adag[i] @ a[j])
                    H += g[i, j].conj() * (adag[j] @ a[i])
        return self._make_hermitian(H)

    def onsite_kerr(self, U: np.ndarray) -> np.ndarray:
        """On-site Kerr nonlinearity (anharmonicity) per mode.

        H = (1/2) sum_i U_i * a†_i a†_i a_i a_i
          = (1/2) sum_i U_i * n_i (n_i - 1)

        Raises the energy of multiply-occupied states.  For U_i >= 0 this
        bounds the spectrum from below (the interaction is repulsive).
        Diagonal in the Fock basis.

        Parameters
        ----------
        U : np.ndarray, shape (n_modes,)
            On-site interaction strength for each mode.
        """
        U = np.asarray(U, dtype=complex)
        self._check_shape(U, (self._n,), "U")
        _, _, n_op = self._get_ops()
        I = np.eye(self._dim, dtype=complex)
        H = self._zero()
        for i in range(self._n):
            if U[i] != 0:
                ni = n_op[i]
                H += 0.5 * U[i] * (ni @ (ni - I))
        return self._make_hermitian(H)

    def density_density(self, V: np.ndarray) -> np.ndarray:
        """Density-density co-activation coupling.

        H = sum_{i,j} V_{ij} * n_i * n_j

        The occupation of one mode shifts the effective energy of another
        (static co-activation cost).  Conserves excitation number and is
        diagonal in the Fock basis.

        Parameters
        ----------
        V : np.ndarray, shape (n_modes, n_modes)
            Co-activation matrix (symmetric for physical interpretation).
        """
        V = np.asarray(V, dtype=complex)
        self._check_shape(V, (self._n, self._n), "V")
        _, _, n_op = self._get_ops()
        H = self._zero()
        for i in range(self._n):
            for j in range(self._n):
                if V[i, j] != 0:
                    H += V[i, j] * (n_op[i] @ n_op[j])
        return self._make_hermitian(H)

    # ==================================================================
    # 2. Unbounded odd-order cubic reference (kept for contrast only)
    # ==================================================================

    def cubic_reference(self, lam: np.ndarray) -> np.ndarray:
        """Odd-order cubic (three-wave mixing) reference interaction.

        H = sum_{i,j,k} lam_{ijk} * a†_i a†_j a_k + h.c.

        WARNING — reference term only.  This does NOT conserve excitation
        number and its odd-order structure makes the spectrum unbounded
        below: the truncated "ground state" is a cutoff artifact, so E0
        drifts with max_occupation and never converges.  It is provided to
        contrast with the bounded Bose--Hubbard model above, which has a
        genuine, cutoff-independent ground state in a fixed-N sector.

        Parameters
        ----------
        lam : np.ndarray, shape (n_modes, n_modes, n_modes)
            Three-index coupling tensor.
        """
        lam = np.asarray(lam, dtype=complex)
        self._check_shape(lam, (self._n, self._n, self._n), "lam")
        adag, a, _ = self._get_ops()
        H = self._zero()
        for i in range(self._n):
            for j in range(self._n):
                for k in range(self._n):
                    if lam[i, j, k] != 0:
                        term = lam[i, j, k] * (adag[i] @ adag[j] @ a[k])
                        H += term
                        H += term.conj().T   # h.c.
        return self._make_hermitian(H)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"InteractionHamiltonians(n_modes={self._n}, "
            f"dim={self._dim})"
        )
