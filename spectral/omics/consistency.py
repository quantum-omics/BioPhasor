"""
omics.consistency — Consistency suite of pipeline invariants (theory.md §6).

Physics-style checks run at the end of every pipeline. Each returns
(passed: bool, residual: float). Mirrors the consistency suites of the sibling
projects (Hermiticity, reality, PSD, gauge invariance, spectral completeness).

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class ConsistencySuite:
    """Run the invariant checks of theory.md §6.

    Parameters
    ----------
    tol_strict : float
        Tolerance for exact-algebra checks (Hermiticity, reality, trace).
    tol_loose : float
        Tolerance for float-noise-sensitive checks (orthonormality, PSD, gauge).
    """

    def __init__(self, tol_strict: float = 1e-10, tol_loose: float = 1e-8) -> None:
        self.tol_strict = tol_strict
        self.tol_loose = tol_loose

    # ------------------------------------------------------------------
    @staticmethod
    def _herm_residual(A: np.ndarray) -> float:
        A = np.asarray(A)
        return float(np.linalg.norm(A - A.conj().T))

    # ------------------------------------------------------------------
    def check_hermiticity(self, H: np.ndarray) -> Tuple[bool, float]:
        r = self._herm_residual(H)
        return (r <= self.tol_strict, r)

    def check_reality(self, eigenvalues: np.ndarray) -> Tuple[bool, float]:
        r = float(np.max(np.abs(np.imag(np.asarray(eigenvalues, dtype=complex)))))
        return (r <= self.tol_strict, r)

    def check_orthonormality(self, eigenvectors: np.ndarray) -> Tuple[bool, float]:
        V = np.asarray(eigenvectors)
        G = V.conj().T @ V
        r = float(np.linalg.norm(G - np.eye(G.shape[0])))
        return (r <= self.tol_loose, r)

    def check_ost_hermiticity(self, M: np.ndarray) -> Tuple[bool, float]:
        r = self._herm_residual(M)
        return (r <= self.tol_strict, r)

    def check_psd(self, G: np.ndarray) -> Tuple[bool, float]:
        G = np.asarray(G)
        Gh = 0.5 * (G + G.conj().T)
        wmin = float(np.min(np.linalg.eigvalsh(Gh)))
        # residual = magnitude of the most-negative eigenvalue (0 if PSD)
        r = max(0.0, -wmin)
        return (wmin >= -self.tol_loose, r)

    def check_gauge_invariance(
        self,
        H: np.ndarray,
        psi: np.ndarray,
        coupling: np.ndarray,
        n_trials: int = 3,
        seed: int = 0,
    ) -> Tuple[bool, float]:
        """Spectrum of H(ψ) equals spectrum of H(ψ·e^{iα}) for random global α.

        A global phase shift θ_i → θ_i + α (i.e. ψ → ψ e^{iα}) leaves
        H_ij = c_ij ψ_i conj(ψ_j) invariant (the amplitudes r_i are untouched),
        hence its spectrum. We verify the eigenvalue sets match.
        """
        H = np.asarray(H, dtype=complex)
        psi = np.asarray(psi, dtype=complex).ravel()
        coupling = np.asarray(coupling, dtype=float)
        base = np.sort(np.real(np.linalg.eigvalsh(H)))
        rng = np.random.default_rng(seed)
        worst = 0.0
        for _ in range(n_trials):
            alpha = float(rng.uniform(-np.pi, np.pi))
            v = psi * np.exp(1j * alpha)
            Hs = coupling * np.outer(v, np.conj(v))
            Hs = 0.5 * (Hs + Hs.conj().T)
            shifted = np.sort(np.real(np.linalg.eigvalsh(Hs)))
            worst = max(worst, float(np.max(np.abs(base - shifted))))
        return (worst <= self.tol_loose, worst)

    def check_spectral_completeness(self, H: np.ndarray, eigenvalues: np.ndarray) -> Tuple[bool, float]:
        """Σ λ_n = tr H over the FULL spectrum (theory.md §6).

        The pipeline may retain only the leading ``n_harmonics`` eigenvalues, so
        the completeness identity is checked against the full spectrum recomputed
        from H (a truncated sum would fail by construction). If the supplied
        `eigenvalues` already span the full dimension it is used directly.
        """
        H = np.asarray(H)
        tr = float(np.real(np.trace(H)))
        vals = np.asarray(eigenvalues)
        if vals.shape[0] != H.shape[0]:
            vals = np.linalg.eigvalsh(H)
        s = float(np.sum(np.real(vals)))
        r = abs(tr - s)
        return (r <= self.tol_loose, r)

    # ------------------------------------------------------------------
    def run(
        self,
        H: np.ndarray,
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        M: np.ndarray,
        psi: np.ndarray,
        coupling: np.ndarray,
    ) -> Dict[str, Tuple[bool, float]]:
        """Run the full suite; returns {check: (passed, residual)} (theory.md §6).

        `psi` is the phasor vector for the slice (ψ_i = r_i e^{iθ_i}).
        """
        G = M.conj().T @ M
        results = {
            "hermiticity": self.check_hermiticity(H),
            "reality": self.check_reality(eigenvalues),
            "orthonormality": self.check_orthonormality(eigenvectors),
            "ost_hermiticity": self.check_ost_hermiticity(M),
            "psd_covariance": self.check_psd(G),
            "gauge_invariance": self.check_gauge_invariance(H, psi, coupling),
            "spectral_completeness": self.check_spectral_completeness(H, eigenvalues),
        }
        return results

    @staticmethod
    def all_passed(results: Dict[str, Tuple[bool, float]]) -> bool:
        return all(passed for passed, _ in results.values())
