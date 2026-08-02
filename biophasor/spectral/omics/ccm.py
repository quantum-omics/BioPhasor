"""
omics.ccm — Compartment Coupling Matrix (theory.md §5).

The CCM is a 5×5 Hermitian matrix that aggregates the OCM over the five
biological compartments:

    Clock, Redox, Energy, Signalling, Biosynthesis

    M_ab = (1/√(N_a N_b)) Σ_{i∈G_a} Σ_{j∈G_b} H_ij ,   M = M†.

The covariance form G = M† M ⪰ 0 supplies a positive semi-definite readout for
the compartment spectral weights.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

# Canonical compartment order (theory.md §5.1) — the five CCM axes.
COMPARTMENTS: List[str] = ["Clock", "Redox", "Energy", "Signalling", "Biosynthesis"]


class CompartmentCouplingMatrix:
    """Build the 5×5 Hermitian Compartment Coupling Matrix from an OCM.

    Parameters
    ----------
    compartments : sequence of str, optional
        Compartment axis labels (default: the five biological compartments).
    """

    def __init__(self, compartments: Optional[Sequence[str]] = None) -> None:
        self.compartments = list(compartments) if compartments is not None else list(COMPARTMENTS)
        self.n_comp = len(self.compartments)
        self.M_: Optional[np.ndarray] = None
        self.assignment_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def assign_by_harmonic(self, eigenvectors: np.ndarray) -> np.ndarray:
        """Fallback assignment of features to compartments by dominant harmonic.

        Feature i is assigned to compartment (n mod n_comp) where n is the
        harmonic on which |φ_{n,i}| is largest. Used when marker membership is
        not provided.

        Returns
        -------
        np.ndarray, shape (N,), int in [0, n_comp).
        """
        V = np.abs(np.asarray(eigenvectors))              # (N, k)
        dominant_mode = np.argmax(V, axis=1)              # (N,)
        assign = dominant_mode % self.n_comp
        self.assignment_ = assign
        return assign

    # ------------------------------------------------------------------
    def _membership_to_index(
        self,
        membership: Dict[str, Sequence[int]],
        N: int,
    ) -> np.ndarray:
        """Convert {compartment: [feature indices]} → per-feature label array.

        Features absent from every list get label -1 (unassigned).
        """
        assign = np.full(N, -1, dtype=int)
        for a, name in enumerate(self.compartments):
            for i in membership.get(name, []):
                if 0 <= i < N:
                    assign[i] = a
        return assign

    # ------------------------------------------------------------------
    def build(
        self,
        H: np.ndarray,
        membership: Optional[Dict[str, Sequence[int]]] = None,
        eigenvectors: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the 5×5 Hermitian CCM M from the OCM H (theory.md §5.2).

        Parameters
        ----------
        H : np.ndarray, complex, shape (N, N)
            Hermitian OCM.
        membership : dict {compartment: [feature indices]}, optional
            Marker/pathway assignment of features to compartments.
        eigenvectors : np.ndarray, shape (N, k), optional
            OCM eigenvectors, used for harmonic-based assignment of features
            not covered by `membership` (or when membership is None).

        Returns
        -------
        np.ndarray, complex, shape (n_comp, n_comp) — Hermitian.
        """
        H = np.asarray(H, dtype=complex)
        N = H.shape[0]

        if membership is not None:
            assign = self._membership_to_index(membership, N)
            # fill unassigned features via harmonics if available
            if (assign < 0).any() and eigenvectors is not None:
                fallback = self.assign_by_harmonic(eigenvectors)
                assign = np.where(assign < 0, fallback, assign)
            # any still-unassigned features: round-robin
            if (assign < 0).any():
                idx = np.where(assign < 0)[0]
                assign[idx] = idx % self.n_comp
        elif eigenvectors is not None:
            assign = self.assign_by_harmonic(eigenvectors)
        else:
            # deterministic round-robin partition
            assign = np.arange(N) % self.n_comp

        self.assignment_ = assign
        groups = [np.where(assign == a)[0] for a in range(self.n_comp)]

        M = np.zeros((self.n_comp, self.n_comp), dtype=complex)
        for a in range(self.n_comp):
            Ga = groups[a]
            for b in range(self.n_comp):
                Gb = groups[b]
                if Ga.size == 0 or Gb.size == 0:
                    continue
                block_sum = H[np.ix_(Ga, Gb)].sum()
                M[a, b] = block_sum / np.sqrt(Ga.size * Gb.size)

        # enforce exact Hermiticity against float noise
        M = 0.5 * (M + M.conj().T)
        self.M_ = M
        return M

    # ------------------------------------------------------------------
    @staticmethod
    def covariance_form(M: np.ndarray) -> np.ndarray:
        """PSD covariance form G = M† M ⪰ 0 (theory.md §5.2)."""
        M = np.asarray(M, dtype=complex)
        return M.conj().T @ M
