"""
biophasor.dynamics.synchrony — Synchrony and phase-locking metrics.

Implements:
  - Kuramoto order parameter  R = |(1/N) Σ e^{iφ}|
  - Phase Locking Value (PLV)
  - Coherence (mean resultant length)
  - Phase-lag index (PLI) for directed coupling

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class SynchronyMetrics:
    """
    Collection of synchrony and phase-locking statistics.

    Parameters
    ----------
    phase : np.ndarray, shape (n_samples, n_features)
        Phase arrays where axis-0 = samples/time and axis-1 = oscillators/genes.
    """

    def __init__(self, phase: np.ndarray) -> None:
        self.phase = np.asarray(phase, dtype=np.float64)

    # ── Order parameter ────────────────────────────────────────────────────────

    def order_parameter(self, axis: int = 0) -> np.ndarray:
        """
        Kuramoto order parameter R per feature:

            R = |(1/N) Σ_{j} e^{iφ_j}|   ∈ [0, 1]

        R → 1 : all features perfectly in phase.
        R → 0 : phases uniformly distributed (incoherent).
        """
        return np.abs(np.exp(1j * self.phase).mean(axis=axis))

    def mean_phase(self, axis: int = 0) -> np.ndarray:
        """Global mean phase Ψ = arg( mean(e^{iφ}) ) per feature."""
        return np.angle(np.exp(1j * self.phase).mean(axis=axis))

    # ── Phase Locking Value ────────────────────────────────────────────────────

    def plv_matrix(self) -> np.ndarray:
        """
        Compute the (n_features × n_features) Phase Locking Value matrix.

            PLV_{jk} = |(1/N) Σ_t e^{i(φ_{tj} − φ_{tk})}|

        Returns
        -------
        np.ndarray, shape (n_features, n_features), values ∈ [0, 1]
        """
        z = np.exp(1j * self.phase)            # (N, F)
        # Cross-covariance in complex domain → PLV
        PLV = np.abs(z.T @ z.conj()) / self.phase.shape[0]  # (F, F)
        return PLV

    def plv_pair(self, i: int, j: int) -> float:
        """PLV between two specific features (columns i and j)."""
        delta = self.phase[:, i] - self.phase[:, j]
        return float(np.abs(np.exp(1j * delta).mean()))

    # ── Phase-lag index ────────────────────────────────────────────────────────

    def pli_matrix(self) -> np.ndarray:
        """
        Phase-Lag Index (PLI) matrix — asymmetric; insensitive to zero-lag.

            PLI_{jk} = |<sign(sin(φ_j − φ_k))>|

        Returns
        -------
        np.ndarray, shape (n_features, n_features), values ∈ [0, 1]
        """
        F = self.phase.shape[1]
        PLI = np.zeros((F, F))
        for j in range(F):
            for k in range(F):
                if j == k:
                    continue
                delta = self.phase[:, j] - self.phase[:, k]
                PLI[j, k] = np.abs(np.sign(np.sin(delta)).mean())
        return PLI

    # ── Coherence ─────────────────────────────────────────────────────────────

    def coherence(self, axis: int = 0) -> np.ndarray:
        """Alias for order_parameter (mean resultant length)."""
        return self.order_parameter(axis=axis)

    # ── Synchronisation index ─────────────────────────────────────────────────

    def synchronisation_index(self) -> float:
        """
        Mean pairwise PLV over all feature pairs — a scalar network
        synchronisation index.
        """
        PLV = self.plv_matrix()
        n = PLV.shape[0]
        # Off-diagonal elements only
        mask = ~np.eye(n, dtype=bool)
        return float(PLV[mask].mean())

    # ── Rayleigh test ─────────────────────────────────────────────────────────

    def rayleigh_per_feature(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Rayleigh test for uniformity per feature (column).

        Returns
        -------
        R_vals : np.ndarray, shape (n_features,)
        p_vals : np.ndarray, shape (n_features,)
        """
        from biophasor.utils.math_utils import rayleigh_test
        R_list, p_list = [], []
        for j in range(self.phase.shape[1]):
            R, p = rayleigh_test(self.phase[:, j])
            R_list.append(R)
            p_list.append(p)
        return np.array(R_list), np.array(p_list)
