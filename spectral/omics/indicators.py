"""
omics.indicators — Spectral indicators of the omics harmonics (theory.md §4).

From the eigenpairs {λ_n, φ_n} of the OCM and the phasor field, compute a panel
of scalar indicators of collective molecular state:

    H_spec  spectral entropy          ∈ [0,1]
    Δ_F     spectral (Fiedler) gap    = λ_1 − λ_2
    a_conn  algebraic connectivity    (2nd-smallest eig of graph Laplacian)
    PR      participation ratio       (effective number of active modes)
    L_1     mode localisation (IPR of φ_1)
    R       Kuramoto order parameter  = phase coherence |⟨e^{iθ}⟩|

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class SpectralIndicators:
    """Compute the spectral-indicator panel (theory.md §4)."""

    # ------------------------------------------------------------------
    @staticmethod
    def spectral_entropy(eigenvalues: np.ndarray, eps: float = 1e-12) -> float:
        """Normalised spectral entropy H_spec = −Σ p_n log p_n / log N ∈ [0,1].

        p_n = |λ_n| / Σ|λ_k|. 0 = one dominant mode, 1 = flat spectrum.
        """
        a = np.abs(np.asarray(eigenvalues, dtype=float))
        s = a.sum()
        if s <= 0:
            return 0.0
        p = a / s
        p = p[p > eps]
        N = len(eigenvalues)
        if N <= 1:
            return 0.0
        return float(-np.sum(p * np.log(p)) / np.log(N))

    # ------------------------------------------------------------------
    @staticmethod
    def fiedler_gap(eigenvalues: np.ndarray) -> float:
        """Leading spectral gap Δ_F = λ_1 − λ_2 (eigenvalues sorted descending)."""
        v = np.asarray(eigenvalues, dtype=float)
        if v.size < 2:
            return 0.0
        vs = np.sort(v)[::-1]
        return float(vs[0] - vs[1])

    # ------------------------------------------------------------------
    @staticmethod
    def algebraic_connectivity(coupling: np.ndarray) -> float:
        """Algebraic connectivity: 2nd-smallest eigenvalue of L = D − C.

        Parameters
        ----------
        coupling : np.ndarray, shape (N, N)
            Non-negative symmetric coupling matrix (off-diagonal weights).
        """
        C = np.asarray(coupling, dtype=float).copy()
        np.fill_diagonal(C, 0.0)
        d = C.sum(axis=1)
        L = np.diag(d) - C
        w = np.linalg.eigvalsh(0.5 * (L + L.T))
        w = np.sort(w)
        return float(w[1]) if w.size >= 2 else 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def participation_ratio(eigenvalues: np.ndarray) -> float:
        """PR = (Σ λ_n²)² / (Σ λ_n⁴): effective number of active modes."""
        v = np.asarray(eigenvalues, dtype=float)
        num = (np.sum(v ** 2)) ** 2
        den = np.sum(v ** 4)
        return float(num / den) if den > 0 else 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def mode_localisation(eigenvector: np.ndarray) -> float:
        """Inverse participation ratio of a mode: L = Σ_i |φ_i|⁴ ∈ (0,1].

        Near 1/N = delocalised (spread over all features); near 1 = localised
        on a single feature.
        """
        phi = np.asarray(eigenvector, dtype=complex).ravel()
        p = np.abs(phi) ** 2
        p = p / (p.sum() + 1e-12)
        return float(np.sum(p ** 2))

    # ------------------------------------------------------------------
    @staticmethod
    def kuramoto_R(psi: np.ndarray) -> float:
        """Kuramoto order parameter R = |⟨e^{iθ}⟩| ∈ [0,1] from phasors ψ."""
        psi = np.asarray(psi, dtype=complex).ravel()
        if psi.size == 0:
            return 0.0
        return float(np.abs(np.mean(np.exp(1j * np.angle(psi)))))

    # ------------------------------------------------------------------
    def compute(
        self,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        psi: np.ndarray,
        coupling: Optional[np.ndarray] = None,
    ) -> dict:
        """Full indicator panel as a dict (theory.md §4).

        Parameters
        ----------
        eigenvalues : (k,)   descending OCM eigenvalues.
        eigenvectors : (N,k) orthonormal OCM eigenvectors (columns).
        psi : (N,)           phasor vector for the slice.
        coupling : (N,N), optional  coupling matrix (for algebraic connectivity).
        """
        eigenvalues = np.asarray(eigenvalues, dtype=float)
        phi1 = np.asarray(eigenvectors)[:, 0] if np.asarray(eigenvectors).ndim == 2 else eigenvectors
        out = {
            "spectral_entropy": self.spectral_entropy(eigenvalues),
            "fiedler_gap": self.fiedler_gap(eigenvalues),
            "participation_ratio": self.participation_ratio(eigenvalues),
            "mode_localisation": self.mode_localisation(phi1),
            "coherence_R": self.kuramoto_R(psi),
        }
        if coupling is not None:
            out["algebraic_connectivity"] = self.algebraic_connectivity(coupling)
        return out
