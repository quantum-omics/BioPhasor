"""
connectome.phasor — Phasor-vertex encoding of omics data (theory.md §1).

Each molecular feature i (gene, protein, peak, metabolite) in each sample is
encoded as a complex phasor vertex

    ψ_i = r_i · exp(i · θ_i)

with the phase θ_i from the tanh-phase encoding
(theory.md Eq. §1.1) and a non-negative amplitude r_i (theory.md §1.2).

This is a phase-native signal-processing model, NOT a physical quantum claim.

Migration note (BioPhasor unification, Phase 3)
-----------------------------------------------
The former module-level ``tanh_phase_encode``, ``phase_coherence`` and
``phasor_statistics`` defined here were verified equivalent to the canonical
core implementations — ``tanh_phase_encode`` numerically (max|diff|=0.0 on
gamma/normal inputs, integration_map §7); ``phase_coherence`` equals
``core.operators.coherence`` on flattened input, and ``phasor_statistics``
delegates to core with identical output (both confirmed during Phase 5
verification, docs/VERIFICATION.md §3). They have been DELETED here and are now
re-exported from ``biophasor.core``:

    tanh_phase_encode  ← biophasor.core.encoder
    phase_coherence, phasor_statistics ← biophasor.core.operators

so downstream ``biophasor.spectral.connectome.{tanh_phase_encode,
phase_coherence}`` continue to resolve. Only the ``PhasorEncoder`` class
(amplitude modes + encoding, which is spectral-specific) lives here now, and it
delegates its phase encoding to the canonical core encoder.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# ── canonical shared implementations (single source of truth in core) ────────
from biophasor.core.encoder import tanh_phase_encode
from biophasor.core.operators import phase_coherence, phasor_statistics


class PhasorEncoder:
    """Encode an omics matrix into complex phasor vertices ψ_i = r_i e^{iθ_i}.

    Parameters
    ----------
    log_transform : bool
        Apply log1p in the phase encoding (default True; False for ATAC /
        methylation β values already in log/bounded space).
    amplitude_mode : {'expression', 'rhythm', 'unit'}
        How to compute the amplitude r_i.

        * ``'expression'`` — per-feature min–max of log1p(x) into [0,1] (default).
        * ``'rhythm'``     — amplitude of the leading Fourier component along the
          sample axis (for time-course data); rescaled per feature to [0,1].
        * ``'unit'``       — r_i = 1 for all features (pure phase encoding).
    eps : float
        Numerical floor.
    """

    _VALID_AMP = {"expression", "rhythm", "unit"}

    def __init__(
        self,
        log_transform: bool = True,
        amplitude_mode: str = "expression",
        eps: float = 1e-8,
    ) -> None:
        if amplitude_mode not in self._VALID_AMP:
            raise ValueError(
                f"amplitude_mode must be one of {self._VALID_AMP}, got '{amplitude_mode}'."
            )
        self.log_transform = log_transform
        self.amplitude_mode = amplitude_mode
        self.eps = eps

    # ------------------------------------------------------------------
    # phase
    # ------------------------------------------------------------------
    def compute_phase(self, X: np.ndarray) -> np.ndarray:
        """Phase matrix θ ∈ (−π, π], shape (S, N) — theory.md §1.1.

        Delegates to the canonical ``biophasor.core.encoder.tanh_phase_encode``
        (``eps`` is accepted as the deprecated alias for ``epsilon``).
        """
        return tanh_phase_encode(X, log_transform=self.log_transform, eps=self.eps)

    # ------------------------------------------------------------------
    # amplitude
    # ------------------------------------------------------------------
    def compute_amplitude(self, X: np.ndarray) -> np.ndarray:
        """Non-negative amplitude r ∈ [0,1], shape (S, N) — theory.md §1.2."""
        X = np.asarray(X, dtype=float)
        S, N = X.shape

        if self.amplitude_mode == "unit":
            return np.ones_like(X)

        if self.amplitude_mode == "expression":
            L = np.log1p(X) if self.log_transform else X
            lo = L.min(axis=0, keepdims=True)
            hi = L.max(axis=0, keepdims=True)
            return (L - lo) / (hi - lo + self.eps)

        # rhythm: amplitude of leading (non-DC) Fourier component along samples
        L = np.log1p(X) if self.log_transform else X
        Lc = L - L.mean(axis=0, keepdims=True)
        F = np.fft.rfft(Lc, axis=0)                       # (S//2+1, N)
        mag = np.abs(F)
        if mag.shape[0] > 1:
            mag = mag[1:]                                  # drop DC
        amp_peak = mag.max(axis=0, keepdims=True)          # (1, N)
        # broadcast the (feature-level) rhythm amplitude to all samples, rescaled
        amp = np.repeat(amp_peak, S, axis=0)
        col_max = amp.max(axis=0, keepdims=True) + self.eps
        return amp / col_max

    # ------------------------------------------------------------------
    # phasor
    # ------------------------------------------------------------------
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Return the complex phasor matrix Ψ of shape (S, N) — theory.md §1.

            ψ_i(s) = r_i(s) · exp(i · θ_i(s))
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (S, N); got shape {X.shape}.")
        theta = self.compute_phase(X)
        r = self.compute_amplitude(X)
        return r * np.exp(1j * theta)

    # ------------------------------------------------------------------
    # per-slice statistics
    # ------------------------------------------------------------------
    @staticmethod
    def phasor_statistics(psi: np.ndarray) -> dict:
        """Aggregate phasor statistics for one sample slice ψ ∈ ℂ^N.

        Delegates to the canonical ``biophasor.core.operators.phasor_statistics``
        (VERIFIED identical to the former in-module implementation). Kept as a
        static method so ``PhasorEncoder.phasor_statistics`` remains part of the
        spectral API.
        """
        return phasor_statistics(psi)
