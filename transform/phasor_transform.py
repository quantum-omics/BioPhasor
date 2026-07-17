"""
biophasor.transform.phasor_transform — Biological Phasor Transform (BPT).

For time-series or oscillatory omics the classical phasor transform maps
intensity I(t) to the complex plane (G, S):

    G = Σ_k I_k cos(2π·j·k/N) / Σ_k I_k
    S = Σ_k I_k sin(2π·j·k/N) / Σ_k I_k

where j is the harmonic index (j=1 for fundamental frequency).

The resulting phasor  z = G + iS  lies on or inside the unit semicircle
(G²+S² ≤ 1) when the intensity is non-negative.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class BPT:
    """
    Biological Phasor Transform — maps intensity time-series to (G, S) space.

    Parameters
    ----------
    n_harmonics : int
        Number of harmonics to compute (default: 1 = fundamental).
    frequency : float
        Fundamental frequency in Hz (e.g. 1/24 for circadian, 1/cell_cycle_h).

    Examples
    --------
    >>> bpt = BPT(n_harmonics=1)
    >>> G, S = bpt.fit_transform(X_timeseries)  # X shape (n_timepoints, n_genes)
    """

    def __init__(self, n_harmonics: int = 1, frequency: float = 1.0) -> None:
        self.n_harmonics = n_harmonics
        self.frequency = frequency

    def fit_transform(
        self,
        X: np.ndarray,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the (G, S) phasor coordinates for each feature.

        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_features)
            Intensity (expression) time series.
        normalize : bool
            If True, normalise by total intensity (FLIM convention).

        Returns
        -------
        G : np.ndarray, shape (n_harmonics, n_features)
        S : np.ndarray, shape (n_harmonics, n_features)
        """
        X = np.asarray(X, dtype=np.float64)
        T, F = X.shape
        total = X.sum(axis=0)                       # (F,)
        normaliser = (total + 1e-12) if normalize else np.ones(F)

        G_list, S_list = [], []
        for j in range(1, self.n_harmonics + 1):
            t_idx = np.arange(T)
            cos_w = np.cos(2.0 * np.pi * j * t_idx / T)[:, np.newaxis]  # (T,1)
            sin_w = np.sin(2.0 * np.pi * j * t_idx / T)[:, np.newaxis]
            G_j = (X * cos_w).sum(axis=0) / normaliser  # (F,)
            S_j = (X * sin_w).sum(axis=0) / normaliser
            G_list.append(G_j)
            S_list.append(S_j)

        return np.array(G_list), np.array(S_list)

    def to_phase_amplitude(
        self,
        X: np.ndarray,
        normalize: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return phase φ and amplitude A = sqrt(G² + S²) for the first harmonic.

        Returns
        -------
        phase     : np.ndarray, shape (n_features,)   ∈ (−π, π]
        amplitude : np.ndarray, shape (n_features,)   ∈ [0, 1]
        """
        G, S = self.fit_transform(X, normalize=normalize)
        phase = np.arctan2(S[0], G[0])
        amplitude = np.sqrt(G[0] ** 2 + S[0] ** 2)
        return phase, amplitude

    @staticmethod
    def semicircle_constraint(G: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Check which features satisfy G²+S² ≤ 1 (semicircle constraint).

        Returns boolean mask of shape (n_features,).
        """
        return (G ** 2 + S ** 2) <= 1.0 + 1e-9


class PhasorWavelet:
    """
    Multi-scale phasor decomposition using Morlet wavelets on the circle.

    Parameters
    ----------
    scales : list[float]
        Wavelet scale factors (equivalent to periods).
    """

    def __init__(self, scales: Optional[list] = None) -> None:
        self.scales = scales or [1, 2, 4, 8, 16]

    def transform(self, X: np.ndarray) -> dict[float, np.ndarray]:
        """
        Apply Morlet wavelet transform at each scale.

        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_features)

        Returns
        -------
        dict mapping scale → complex wavelet coefficient array (n_timepoints, n_features)
        """
        T, F = X.shape
        result = {}
        for s in self.scales:
            t = np.arange(T)
            # Normalised Morlet wavelet
            sigma = s / 2.0
            psi = (
                np.exp(-0.5 * (t / sigma) ** 2)
                * np.exp(2j * np.pi * t / s)
                / (sigma * np.sqrt(2 * np.pi))
            )
            # Convolve along time axis for each feature
            coeffs = np.zeros((T, F), dtype=complex)
            for f in range(F):
                coeffs[:, f] = np.convolve(X[:, f], psi[::-1].conj(), mode='same')
            result[s] = coeffs
        return result
