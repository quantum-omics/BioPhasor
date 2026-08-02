"""
connectome.harmonics — Eigendecomposition of the OCM → omics harmonics
(theory.md §3).

    H φ_n = λ_n φ_n ,     n = 1…N

Uses numpy.linalg.eigh (Hermitian solver ⇒ real eigenvalues). Eigenvalues are
sorted in descending order so φ_1 is the dominant collective mode ("first omics
harmonic"). Eigenvectors are orthonormal.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class OmicsHarmonics:
    """Compute and store omics harmonics from a Hermitian OCM.

    Parameters
    ----------
    n_harmonics : int or None
        Number of leading harmonics to retain (by descending eigenvalue).
        None keeps all N.
    force_real_eigenvalues : bool
        Take np.real() of eigenvalues (suppresses tiny imaginary float noise).
    """

    def __init__(
        self,
        n_harmonics: Optional[int] = None,
        force_real_eigenvalues: bool = True,
    ) -> None:
        self.n_harmonics = n_harmonics
        self.force_real_eigenvalues = force_real_eigenvalues
        self.eigenvalues_: Optional[np.ndarray] = None    # (k,)
        self.eigenvectors_: Optional[np.ndarray] = None   # (N, k)

    # ------------------------------------------------------------------
    def decompose(self, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Eigendecompose one Hermitian OCM H (N,N) → (λ, Φ), descending λ.

        Returns
        -------
        eigenvalues : np.ndarray, shape (k,)
        eigenvectors : np.ndarray, complex, shape (N, k) — orthonormal columns.
        """
        H = np.asarray(H)
        vals, vecs = np.linalg.eigh(H)         # ascending, real vals
        order = np.argsort(vals)[::-1]         # descending
        vals = vals[order]
        vecs = vecs[:, order]
        if self.force_real_eigenvalues:
            vals = np.real(vals)
        k = self.n_harmonics or len(vals)
        vals = vals[:k]
        vecs = vecs[:, :k]
        self.eigenvalues_ = vals
        self.eigenvectors_ = vecs
        return vals, vecs

    # ------------------------------------------------------------------
    def spectral_energy(self, eigenvalues: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalised spectral energy p_n = |λ_n| / Σ|λ_k| (theory.md §3)."""
        vals = self.eigenvalues_ if eigenvalues is None else np.asarray(eigenvalues)
        if vals is None:
            raise RuntimeError("Call decompose() first or pass eigenvalues.")
        a = np.abs(vals)
        s = a.sum()
        return a / s if s > 0 else np.full_like(a, 1.0 / len(a))

    # ------------------------------------------------------------------
    def reconstruct(self, k: Optional[int] = None) -> np.ndarray:
        """Rank-k reconstruction H_k = Σ_{n≤k} λ_n φ_n φ_n† (Hermitian)."""
        if self.eigenvalues_ is None:
            raise RuntimeError("Call decompose() first.")
        k = k or len(self.eigenvalues_)
        vals = self.eigenvalues_[:k]
        vecs = self.eigenvectors_[:, :k]
        return (vecs * vals) @ vecs.conj().T

    # ------------------------------------------------------------------
    def decompose_series(self, H_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Decompose a stack of OCMs H_series (T,N,N) → (λ (T,k), Φ (T,N,k))."""
        H_series = np.asarray(H_series)
        T = H_series.shape[0]
        eigvals, eigvecs = [], []
        for t in range(T):
            v, V = self.decompose(H_series[t])
            eigvals.append(v)
            eigvecs.append(V)
        return np.array(eigvals), np.array(eigvecs)
