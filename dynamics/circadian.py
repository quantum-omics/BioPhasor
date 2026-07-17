"""
biophasor.dynamics.circadian — Circadian rhythm phasor oscillator.

Models circadian rhythmicity as a phasor on the unit circle with period T = 24 h.

The core clock model:
    dφ/dt = 2π/T  (free running)

Phase entrainment by zeitgeber (light) is modelled via a forcing term.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class CircadianPhasor:
    """
    Circadian rhythm phasor oscillator.

    Maps gene expression time-series (sampled every Δt hours) to circadian
    phase using the Biological Phasor Transform at the circadian frequency.

    Parameters
    ----------
    period : float
        Circadian period in hours (default 24.0).
    sample_interval : float
        Sampling interval in hours (default 2.0, i.e. every 2 h).
    """

    def __init__(
        self,
        period: float = 24.0,
        sample_interval: float = 2.0,
        zt_origin: float = 0.0,
    ) -> None:
        self.period = period
        self.dt = sample_interval
        # Zeitgeber time of the first sample (acquisition-clock origin). Used to
        # calibrate absolute peak-ZT so it is anchored to the sampling clock
        # rather than to arbitrary sample position within the window.
        self.zt_origin = zt_origin

    # ── Phase inference ────────────────────────────────────────────────────────

    def infer_phase(self, X: np.ndarray) -> np.ndarray:
        """
        Infer the circadian phase of each gene from its time-series expression.

        Uses the Biological Phasor Transform at the fundamental frequency
        f = 1/period:

            φ_gene = arg( Σ_t I_t · e^{2πijt/T} )

        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_genes)

        Returns
        -------
        np.ndarray, shape (n_genes,)   phase ∈ (−π, π]
        """
        from biophasor.transform.phasor_transform import BPT
        bpt = BPT(n_harmonics=1, frequency=1.0 / self.period)
        phase, _ = bpt.to_phase_amplitude(X, normalize=True)
        return phase

    def peak_zt(
        self,
        X: np.ndarray,
        zt_times: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Absolute peak Zeitgeber time (h) of each gene, calibrated to the
        sampling clock.

        The single-harmonic BPT returns a phase in *sample-index* units. The
        fundamental cosine ``A·cos(2π k/T − θ)`` (k = sample index, T = number
        of samples) peaks at sample position ``k* = θ·T/(2π)``; the absolute
        Zeitgeber time is therefore

            ZT_peak = (ZT_origin + k*·Δt) mod period

        so that peak-ZT is anchored to when the samples were actually taken,
        rather than offset by an arbitrary constant.

        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_genes)
        zt_times : np.ndarray | None
            Explicit acquisition ZT of each sample (length n_timepoints). If
            given, its first entry sets the ZT origin and its spacing sets Δt
            (overriding the instance ``zt_origin`` / ``sample_interval``);
            otherwise the instance values are used.

        Returns
        -------
        np.ndarray, shape (n_genes,)   peak ZT in hours ∈ [0, period)
        """
        from biophasor.transform.phasor_transform import BPT

        X = np.asarray(X, dtype=np.float64)
        T = X.shape[0]
        if zt_times is not None:
            zt_times = np.asarray(zt_times, dtype=np.float64)
            zt0 = float(zt_times[0])
            dt = float(zt_times[1] - zt_times[0]) if T > 1 else self.dt
        else:
            zt0, dt = self.zt_origin, self.dt

        bpt = BPT(n_harmonics=1, frequency=1.0 / self.period)
        G, S = bpt.fit_transform(X, normalize=True)
        theta = np.arctan2(S[0], G[0])                 # (n_genes,), sample-index phase
        kstar = (theta / (2.0 * np.pi)) * T            # peak sample index
        return (zt0 + kstar * dt) % self.period

    def amplitude(self, X: np.ndarray) -> np.ndarray:
        """
        Circadian amplitude (rhythm strength) for each gene.

        Returns sqrt(G² + S²) ∈ [0, 1]; higher = stronger rhythmicity.
        """
        from biophasor.transform.phasor_transform import BPT
        bpt = BPT(n_harmonics=1)
        _, amp = bpt.to_phase_amplitude(X, normalize=True)
        return amp

    # ── Zeitgeber time mapping ─────────────────────────────────────────────────

    @staticmethod
    def zt_to_phase(zt_hours: float, period: float = 24.0) -> float:
        """Convert Zeitgeber time (ZT) in hours to phase in radians."""
        return 2.0 * np.pi * zt_hours / period - np.pi   # → (−π, π]

    @staticmethod
    def phase_to_zt(phi: float, period: float = 24.0) -> float:
        """Convert phasor phase (radians) to ZT hours."""
        phi_pos = (phi + np.pi) % (2 * np.pi)            # [0, 2π)
        return phi_pos * period / (2.0 * np.pi)

    # ── Rhythmicity score ──────────────────────────────────────────────────────

    def rhythmicity_score(self, X: np.ndarray) -> np.ndarray:
        """
        Score each gene for circadian rhythmicity using amplitude + coherence.

        Score ∈ [0, 1]; threshold ≥ 0.3 is commonly used to call rhythmic genes.

        Parameters
        ----------
        X : np.ndarray, shape (n_timepoints, n_genes)

        Returns
        -------
        np.ndarray, shape (n_genes,)
        """
        A = self.amplitude(X)
        # Normalise by the maximum possible amplitude
        return np.clip(A, 0, 1)

    # ── Simulate damped oscillator ────────────────────────────────────────────

    def simulate(
        self,
        n_cycles: int = 3,
        damping: float = 0.0,
        noise: float = 0.0,
        n_genes: int = 1,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Simulate circadian expression time-series.

            I(t) = A · e^{−γt} · cos(2πt/T + φ_0) + ε(t)

        Parameters
        ----------
        n_cycles : int   number of circadian cycles to simulate
        damping : float  exponential damping factor γ (0 = undamped)
        noise : float    Gaussian noise standard deviation
        n_genes : int    number of independent genes to simulate
        seed : int

        Returns
        -------
        np.ndarray, shape (n_timepoints, n_genes)
        """
        rng = np.random.RandomState(seed)
        n_timepoints = int(n_cycles * self.period / self.dt)
        t = np.arange(n_timepoints, dtype=float) * self.dt

        # Random initial phases and amplitudes
        phi0 = rng.uniform(-np.pi, np.pi, n_genes)
        A = rng.uniform(0.5, 1.5, n_genes)

        result = np.zeros((n_timepoints, n_genes))
        for g in range(n_genes):
            signal = A[g] * np.exp(-damping * t) * np.cos(2 * np.pi * t / self.period + phi0[g])
            if noise > 0:
                signal += rng.normal(0, noise, n_timepoints)
            result[:, g] = signal

        return result
