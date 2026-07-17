"""
biophasor.core.manifold — U(1)^N phasor manifold geometry.

Implements geometric operations on the N-torus:
  - Fréchet mean (iterative)
  - Geodesic interpolation
  - Angular distance matrix
  - Tangent-space projections

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class PhasorManifold:
    """
    Geometry class for the N-torus U(1)^N.

    All operations work on phase arrays of shape (n_samples, n_features)
    where each feature lives on an independent unit circle.

    Parameters
    ----------
    n_features : int
        Dimensionality of the manifold (number of independent circles).
    """

    def __init__(self, n_features: int) -> None:
        self.n_features = n_features

    # ── Distance ──────────────────────────────────────────────────────────────

    @staticmethod
    def angular_distance(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
        """
        Element-wise **circular distance** on U(1):

            d(φ_a, φ_b) = 1 − cos(φ_a − φ_b)  ∈ [0, 2]

        Parameters
        ----------
        phi_a, phi_b : np.ndarray   (same shape)

        Returns
        -------
        np.ndarray   same shape as inputs, values ∈ [0, 2]
        """
        return 1.0 - np.cos(phi_a - phi_b)

    @staticmethod
    def pairwise_distance(phase: np.ndarray) -> np.ndarray:
        """
        Compute the (n_samples × n_samples) pairwise angular distance matrix.

            D_{ij} = (1/n_features) Σ_k (1 − cos(φ_{ik} − φ_{jk}))

        Parameters
        ----------
        phase : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        np.ndarray, shape (n_samples, n_samples), values ∈ [0, 2]
        """
        # Use broadcasting: (N, 1, F) vs (1, N, F)
        diff = phase[:, np.newaxis, :] - phase[np.newaxis, :, :]  # (N, N, F)
        return (1.0 - np.cos(diff)).mean(axis=-1)

    # ── Fréchet mean ──────────────────────────────────────────────────────────

    @staticmethod
    def frechet_mean(
        phase: np.ndarray,
        weights: Optional[np.ndarray] = None,
        n_iter: int = 10,
    ) -> np.ndarray:
        """
        Compute the **Fréchet mean** on U(1)^N (iterative geodesic update).

        For each feature independently:
            μ^{(t+1)} = μ^{(t)} + (1/N) Σ_j w_j · log_{μ^{(t)}}( e^{iφ_j} )

        In practice this simplifies to the circular mean:

            μ = arg( Σ_j w_j · e^{iφ_j} )

        which is already the minimiser of the sum of squared geodesic distances.

        Parameters
        ----------
        phase : np.ndarray, shape (n_samples, n_features)
        weights : np.ndarray | None, shape (n_samples,)
        n_iter : int   (unused — kept for API consistency; circular mean is exact)

        Returns
        -------
        np.ndarray, shape (n_features,)   — mean phase per feature
        """
        z = np.exp(1j * phase)  # (n_samples, n_features)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)[:, np.newaxis]
            mean_z = (w * z).sum(axis=0) / (w.sum() + 1e-12)
        else:
            mean_z = z.mean(axis=0)
        return np.angle(mean_z)

    # ── Geodesic interpolation ────────────────────────────────────────────────

    @staticmethod
    def geodesic_interp(
        phi_start: np.ndarray,
        phi_end: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """
        Geodesic interpolation on U(1)^N at parameter ``t`` ∈ [0, 1].

            φ(t) = phi_start + t · wrap(phi_end − phi_start)

        wrapping keeps the shortest arc on the circle.

        Parameters
        ----------
        phi_start, phi_end : np.ndarray (same shape)
        t : float   interpolation parameter ∈ [0, 1]

        Returns
        -------
        np.ndarray   same shape, values in (−π, π]
        """
        delta = ((phi_end - phi_start + np.pi) % (2 * np.pi)) - np.pi
        result = phi_start + t * delta
        return ((result + np.pi) % (2 * np.pi)) - np.pi

    # ── Tangent space ─────────────────────────────────────────────────────────

    @staticmethod
    def log_map(base: np.ndarray, point: np.ndarray) -> np.ndarray:
        """
        Logarithmic map: project ``point`` onto the tangent space at ``base``.

            log_{base}(point) = wrap(point − base)

        Parameters
        ----------
        base : np.ndarray   base phase (any shape)
        point : np.ndarray  target phase (same shape)

        Returns
        -------
        np.ndarray   tangent vector in (−π, π]
        """
        delta = point - base
        return ((delta + np.pi) % (2 * np.pi)) - np.pi

    @staticmethod
    def exp_map(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
        """
        Exponential map: move from ``base`` along ``tangent``.

            exp_{base}(v) = base + v  (wrapped)
        """
        result = base + tangent
        return ((result + np.pi) % (2 * np.pi)) - np.pi

    # ── Curvature / dispersion ────────────────────────────────────────────────

    @staticmethod
    def concentration(phase: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Von Mises **concentration parameter** κ (approximate).

        Uses the approximation of Mardia & Jupp (1999):
            R = |mean(e^{iφ})|
            κ ≈ (2R - R³ - R⁵/6) / (1 - R²)   for R < 0.9

        Parameters
        ----------
        phase : np.ndarray
        axis : int

        Returns
        -------
        np.ndarray   concentration per feature
        """
        z = np.exp(1j * phase)
        R = np.abs(z.mean(axis=axis))
        R = np.clip(R, 0, 1 - 1e-6)
        # Mardia & Jupp approximation
        kappa = (2 * R - R ** 3 - R ** 5 / 6.0) / (1.0 - R ** 2)
        return kappa
