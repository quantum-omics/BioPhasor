"""Bose--Hubbard compartment model for the omics spectral connectome.

This module realises the five biological compartments as five bosonic modes
governed by a **Bose--Hubbard Hamiltonian** — the canonical bounded,
particle-number-conserving many-body model of quantum simulation.

Compartments
------------
    0  Clock         — circadian / temporal regulation
    1  Redox         — oxidative balance
    2  Energy        — energy metabolism
    3  Signalling    — signal transduction
    4  Biosynthesis  — anabolic / biosynthetic activity

Model
-----
For five modes ``k = 0..4``::

    H_omics =  Σ_k ε_k n_k                          (compartment self-energy)
            +  Σ_{i<j} J_ij (a_i† a_j + a_j† a_i)    (coherent hopping)
            +  (U/2) Σ_k n_k (n_k − 1)               (on-site Kerr nonlinearity)
            +  Σ_{i<j} V_ij n_i n_j                  (density–density co-activation)

Properties that make the model *feasible* (in the quantum-physics sense):

* **Bounded below** for ``U >= 0`` and bounded ``J, V`` — a genuine ground
  state exists (an unbounded odd-order cubic model has none).
* **Particle-number conserving**: ``[H_omics, N̂] = 0`` with
  ``N̂ = Σ_k n_k``.  The dynamics stays inside a fixed total-excitation
  sector, so results are computed in a small, *exact* subspace rather than a
  cutoff-dependent full Fock space.
* **Rooted in the omics spectrum**: the self-energies ``ε_k`` are the leading
  omics harmonic frequencies ``ω_k`` from the Omics Connectome Matrix
  (``omics_spectrum.compartment_self_energies``), closing the loop between
  the classical spectral connectome and the quantum model.

The coherent-hopping and co-activation terms live on a small, biologically
motivated **compartment interaction graph**, so every non-zero parameter has
an interpretation.

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from biophasor.spectral.quantum.fock_space import FockSpace

# Canonical compartment order.
COMPARTMENTS: List[str] = [
    "Clock",
    "Redox",
    "Energy",
    "Signalling",
    "Biosynthesis",
]

# Biologically motivated compartment interaction graph (undirected edges).
#   Clock–Energy         : circadian gating of metabolic output
#   Redox–Energy         : oxidative balance coupled to energy metabolism
#   Energy–Biosynthesis  : energy budget driving anabolic activity
#   Signalling–Energy    : signalling modulating metabolic flux
#   Signalling–Biosynthesis : signalling driving biosynthetic programs
COMPARTMENT_EDGES: List[Tuple[int, int]] = [
    (0, 2),  # Clock–Energy
    (1, 2),  # Redox–Energy
    (2, 4),  # Energy–Biosynthesis
    (3, 2),  # Signalling–Energy
    (3, 4),  # Signalling–Biosynthesis
]


class CompartmentModel:
    """Bose--Hubbard Hamiltonian for the five omics compartments.

    Parameters
    ----------
    fock_space:
        A :class:`~quantum.fock_space.FockSpace` with ``n_modes == 5``.
    epsilon:
        Length-5 array of compartment self-energies ``ε_k`` (the leading
        omics harmonic frequencies ``ω_k`` from
        ``omics_spectrum.compartment_self_energies``).
    J:
        Coherent-hopping strengths.  Either a scalar (applied on every
        compartment edge) or a symmetric ``(5, 5)`` matrix.
    U:
        On-site Kerr nonlinearity (scalar ``>= 0`` for a bounded spectrum)
        or a length-5 array of per-compartment values.
    V:
        Density–density co-activation.  Scalar (applied on every edge) or a
        symmetric ``(5, 5)`` matrix.
    edges:
        Interaction-graph edges; defaults to :data:`COMPARTMENT_EDGES`.
    """

    def __init__(
        self,
        fock_space: FockSpace,
        epsilon: Sequence[float],
        J: "float | np.ndarray" = 0.0,
        U: "float | Sequence[float]" = 0.0,
        V: "float | np.ndarray" = 0.0,
        edges: Optional[List[Tuple[int, int]]] = None,
    ) -> None:
        if fock_space.n_modes != 5:
            raise ValueError("CompartmentModel requires a 5-mode FockSpace.")
        self.fock = fock_space
        self.dim = fock_space.dim()
        self.n = 5
        self.edges = list(edges) if edges is not None else list(COMPARTMENT_EDGES)

        self.epsilon = np.asarray(epsilon, dtype=float).reshape(5)
        self.J = self._as_edge_matrix(J)
        self.V = self._as_edge_matrix(V)
        self.U = (np.full(5, float(U)) if np.isscalar(U)
                  else np.asarray(U, dtype=float).reshape(5))

        # Cache single-mode operators as dense arrays (5 modes, small dim).
        self._a = [self._dense(fock_space.annihilation_op(k)) for k in range(5)]
        self._ad = [self._dense(fock_space.creation_op(k)) for k in range(5)]
        self._num = [self._dense(fock_space.number_op(k)) for k in range(5)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _dense(op) -> np.ndarray:
        return op.toarray() if hasattr(op, "toarray") else np.asarray(op, dtype=complex)

    def _as_edge_matrix(self, val: "float | np.ndarray") -> np.ndarray:
        """Promote a scalar edge weight to a symmetric (5,5) matrix."""
        if np.isscalar(val):
            M = np.zeros((5, 5), dtype=float)
            for (i, j) in self.edges:
                M[i, j] = M[j, i] = float(val)
            return M
        M = np.asarray(val, dtype=float).reshape(5, 5)
        return 0.5 * (M + M.T)

    # ------------------------------------------------------------------
    # Individual terms (each Hermitian)
    # ------------------------------------------------------------------
    def self_energy_term(self) -> np.ndarray:
        """``Σ_k ε_k n_k`` — baseline activation cost per compartment."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for k in range(5):
            H += self.epsilon[k] * self._num[k]
        return H

    def hopping_term(self) -> np.ndarray:
        """``Σ_{i<j} J_ij (a_i† a_j + h.c.)`` — coherent compartment coupling."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for (i, j) in self.edges:
            if self.J[i, j] == 0:
                continue
            term = self._ad[i] @ self._a[j]
            H += self.J[i, j] * (term + term.conj().T)
        return H

    def kerr_term(self) -> np.ndarray:
        """``(U/2) Σ_k n_k (n_k − 1)`` — on-site nonlinear self-interaction."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        I = np.eye(self.dim, dtype=complex)
        for k in range(5):
            nk = self._num[k]
            H += 0.5 * self.U[k] * (nk @ (nk - I))
        return H

    def density_density_term(self) -> np.ndarray:
        """``Σ_{i<j} V_ij n_i n_j`` — static co-activation cost."""
        H = np.zeros((self.dim, self.dim), dtype=complex)
        for (i, j) in self.edges:
            if self.V[i, j] == 0:
                continue
            H += self.V[i, j] * (self._num[i] @ self._num[j])
        return H

    def compartment_hamiltonian(self, k: int) -> np.ndarray:
        """The single-compartment observable ``Ĥ_k`` used to build the CCM.

        Defined as the self-energy plus on-site Kerr for compartment ``k``
        and one half of every coupling term incident on ``k`` (so that
        ``Σ_k Ĥ_k = H_omics``).  This gives a clean additive decomposition of
        the total Hamiltonian into compartment contributions.
        """
        H = np.zeros((self.dim, self.dim), dtype=complex)
        I = np.eye(self.dim, dtype=complex)
        H += self.epsilon[k] * self._num[k]
        H += 0.5 * self.U[k] * (self._num[k] @ (self._num[k] - I))
        for (i, j) in self.edges:
            if k not in (i, j):
                continue
            if self.J[i, j] != 0:
                term = self._ad[i] @ self._a[j]
                H += 0.5 * self.J[i, j] * (term + term.conj().T)
            if self.V[i, j] != 0:
                H += 0.5 * self.V[i, j] * (self._num[i] @ self._num[j])
        return H

    def total_hamiltonian(self) -> np.ndarray:
        """Full Bose--Hubbard compartment Hamiltonian ``H_omics`` (Hermitian)."""
        H = (self.self_energy_term() + self.hopping_term()
             + self.kerr_term() + self.density_density_term())
        return 0.5 * (H + H.conj().T)

    # ------------------------------------------------------------------
    # Fixed particle-number sector
    # ------------------------------------------------------------------
    def sector_indices(self, n_total: int) -> np.ndarray:
        """Indices of Fock states with total occupation ``Σ_k n_k == n_total``."""
        labels = self.fock.basis_labels()
        return np.array([i for i, occ in enumerate(labels) if sum(occ) == n_total],
                        dtype=int)

    def project_to_sector(self, H: np.ndarray, n_total: int) -> Tuple[np.ndarray, np.ndarray]:
        """Project operator ``H`` onto the fixed-``N`` sector.

        Returns ``(H_sector, idx)`` — the projected operator and the basis
        indices, so a sector eigenvector can be embedded back into the full
        space.
        """
        idx = self.sector_indices(n_total)
        return H[np.ix_(idx, idx)], idx

    def ground_state(self, n_total: Optional[int] = None) -> Tuple[float, np.ndarray]:
        """Ground state of ``H_omics``.

        If ``n_total`` is given, diagonalize within that fixed-excitation
        sector (exact and cutoff-independent) and embed the eigenvector back
        into the full Fock space; otherwise diagonalize the full space.

        Returns ``(E0, psi)`` with ``psi`` a full-dimension state vector.
        """
        H = self.total_hamiltonian()
        if n_total is None:
            E, V = np.linalg.eigh(H)
            return float(E[0]), V[:, 0].astype(complex)
        Hs, idx = self.project_to_sector(H, n_total)
        E, V = np.linalg.eigh(Hs)
        psi = np.zeros(self.dim, dtype=complex)
        psi[idx] = V[:, 0]
        return float(E[0]), psi

    def commutes_with_number(self, tol: float = 1e-10) -> float:
        """Return ``max|[H_omics, N̂]|`` — should be ~0 (number conservation)."""
        H = self.total_hamiltonian()
        N = sum(self._num)
        comm = H @ N - N @ H
        return float(np.max(np.abs(comm)))

    # ------------------------------------------------------------------
    # Compartment covariance matrix (quantum covariance of observables)
    # ------------------------------------------------------------------
    def compute_ccm(self, psi: np.ndarray) -> np.ndarray:
        """Compartment covariance matrix as the symmetrized covariance matrix.

        ``M_ab = (1/2)⟨{H_a, H_b}⟩ − ⟨H_a⟩⟨H_b⟩`` for the five compartment
        observables ``H_a = compartment_hamiltonian(a)`` in state ``psi``.
        Real, symmetric, and positive semi-definite by construction.
        """
        psi = np.asarray(psi, dtype=complex).ravel()
        nrm = float(np.real(psi.conj() @ psi))
        if nrm > 0:
            psi = psi / np.sqrt(nrm)
        Hs = [self.compartment_hamiltonian(k) for k in range(5)]
        Hpsi = [H @ psi for H in Hs]
        means = np.array([float(np.real(psi.conj() @ hp)) for hp in Hpsi])
        M = np.zeros((5, 5), dtype=float)
        for a in range(5):
            for b in range(a, 5):
                sym = float(np.real(Hpsi[a].conj() @ Hpsi[b]))  # (1/2)⟨{H_a,H_b}⟩
                cov = sym - means[a] * means[b]
                M[a, b] = M[b, a] = cov
        return 0.5 * (M + M.T)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"CompartmentModel(dim={self.dim}, "
            f"edges={len(self.edges)})"
        )
