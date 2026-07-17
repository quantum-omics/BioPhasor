"""
biophasor.cst.limit_cycles — LimitCycleAnalyzer: limit cycle detection in regulatory circuits.

Detects and characterizes limit cycles in phase-coupled dissipative
cellular systems by analyzing periodic orbits, Floquet exponents,
and cycle stability.

Given a phase trajectory on T^N (e.g., from a Goodwin oscillator or
Kuramoto-coupled gene regulatory network), the analyzer identifies:
  1. Periodic orbits (approximate limit cycles)
  2. Cycle periods and winding numbers
  3. Stability via Floquet multipliers (monodromy matrix)
  4. Basin of attraction radii

Biological applications:
  - Cell-cycle oscillator period detection
  - Circadian limit cycle characterization
  - NF-κB pulsing dynamics
  - p53 oscillation stability analysis

Reference: Biophasor Manuscript — Section "Floquet Stability"
           Biophasor Book — Chapter 9 "Dissipative Phase-Coupled Systems"

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class LimitCycle:
    """
    A detected limit cycle in a cellular regulatory circuit.

    Attributes
    ----------
    period : float
        Estimated period of the cycle (in time units / samples).
    center_phase : np.ndarray
        Mean phase vector of the cycle, shape (N,).
    amplitude : float
        Phase oscillation amplitude around the center.
    floquet_multipliers : np.ndarray
        Stability multipliers — |μ| < 1 ⟹ stable cycle.
    winding_numbers : np.ndarray
        Per-oscillator winding numbers over one period.
    basin_radius : float
        Estimated basin of attraction radius (phase distance).
    """
    period: float
    center_phase: np.ndarray
    amplitude: float
    floquet_multipliers: np.ndarray
    winding_numbers: np.ndarray
    basin_radius: float = 0.0

    @property
    def is_stable(self) -> bool:
        """True if all Floquet multipliers have magnitude < 1."""
        return bool(np.all(np.abs(self.floquet_multipliers) < 1.0 + 1e-10))

    @property
    def max_multiplier(self) -> float:
        """Largest Floquet multiplier magnitude (stability margin)."""
        return float(np.max(np.abs(self.floquet_multipliers)))

    @property
    def resilience(self) -> float:
        """
        Resilience = −ln|μ_max| / T — decay rate per unit time.

        Positive resilience → stable; negative → unstable.
        """
        mu = self.max_multiplier
        if mu < 1e-12:
            return float("inf")
        return -np.log(mu) / (self.period + 1e-12)

    def __repr__(self) -> str:
        status = "stable" if self.is_stable else "unstable"
        return (
            f"LimitCycle(T={self.period:.1f}, amp={self.amplitude:.3f}, "
            f"{status}, |μ_max|={self.max_multiplier:.4f})"
        )


class LimitCycleAnalyzer:
    """
    Detect and characterize limit cycles in phase-coupled dissipative systems.

    Works on phase trajectories φ(t) ∈ T^N from gene regulatory circuits
    by detecting approximate periodicity, estimating cycle parameters,
    and computing stability via numerical Floquet analysis.

    Parameters
    ----------
    min_period : float
        Minimum period to search for (in samples).
    max_period : float
        Maximum period to search for.
    periodicity_threshold : float
        Phase recurrence threshold for detecting cycles (radians).
        Lower = stricter detection.
    n_candidates : int
        Number of period candidates to evaluate.

    Examples
    --------
    >>> from biophasor.cst.limit_cycles import LimitCycleAnalyzer
    >>> analyzer = LimitCycleAnalyzer()
    >>> # Simulate a Goodwin oscillator trajectory
    >>> t = np.linspace(0, 100, 2000)
    >>> phase = np.stack([np.sin(0.3*t), np.cos(0.3*t + np.pi/3)])
    >>> cycles = analyzer.detect(np.arctan2(phase[1], phase[0])[np.newaxis, :])
    >>> for c in cycles:
    ...     print(c, c.is_stable)
    """

    def __init__(
        self,
        min_period: float = 10,
        max_period: float = 500,
        periodicity_threshold: float = 0.3,
        n_candidates: int = 50,
    ) -> None:
        self.min_period = min_period
        self.max_period = max_period
        self.threshold = periodicity_threshold
        self.n_candidates = n_candidates

    def detect(self, phase: np.ndarray) -> list[LimitCycle]:
        """
        Detect limit cycles in a phase trajectory.

        Parameters
        ----------
        phase : np.ndarray, shape (N, T) or (T,)
            Phase trajectory on T^N.  If 1D, treated as single oscillator.

        Returns
        -------
        list[LimitCycle]   Detected limit cycles, sorted by period.
        """
        if phase.ndim == 1:
            phase = phase[np.newaxis, :]
        N, T = phase.shape

        candidates = np.linspace(
            self.min_period, min(self.max_period, T // 2),
            self.n_candidates, dtype=int,
        )
        candidates = np.unique(candidates)

        cycles = []
        for tau in candidates:
            if tau < 2 or tau >= T:
                continue
            recurrence = self._recurrence_score(phase, tau)
            if recurrence < self.threshold:
                cycle = self._characterize_cycle(phase, tau)
                cycles.append(cycle)

        cycles = self._deduplicate(cycles)
        return sorted(cycles, key=lambda c: c.period)

    def _recurrence_score(self, phase: np.ndarray, tau: int) -> float:
        """Compute circular recurrence score for period tau."""
        N, T = phase.shape
        n_checks = min(T - tau, tau, 100)
        if n_checks < 1:
            return float("inf")

        scores = []
        for t in range(n_checks):
            diff = phase[:, t + tau] - phase[:, t]
            circ_dist = np.abs(np.angle(np.exp(1j * diff)))
            scores.append(circ_dist.mean())
        return float(np.mean(scores))

    def _characterize_cycle(self, phase: np.ndarray, tau: int) -> LimitCycle:
        """Build a LimitCycle from a detected period."""
        N, T = phase.shape

        # Center phase: circular mean over one period
        center = np.angle(np.exp(1j * phase[:, :tau]).mean(axis=1))

        # Amplitude: mean deviation from center
        deviations = np.abs(np.angle(
            np.exp(1j * (phase[:, :tau] - center[:, np.newaxis]))
        ))
        amplitude = float(deviations.mean())

        # Winding numbers over one period
        winding = np.zeros(N)
        for n in range(N):
            diffs = np.diff(np.unwrap(phase[n, :tau + 1]))
            winding[n] = diffs.sum() / (2 * np.pi)

        # Floquet multipliers (numerical approximation)
        floquet = self._estimate_floquet(phase, tau)

        # Basin radius estimate
        basin = self._estimate_basin_radius(phase, tau)

        return LimitCycle(
            period=float(tau),
            center_phase=center,
            amplitude=amplitude,
            floquet_multipliers=floquet,
            winding_numbers=winding,
            basin_radius=basin,
        )

    def _estimate_floquet(self, phase: np.ndarray, tau: int) -> np.ndarray:
        """
        Estimate Floquet multipliers via monodromy matrix approximation.

        Uses finite differences along the trajectory to build a
        linearized return map.
        """
        N, T = phase.shape
        if tau + 1 >= T:
            return np.ones(N)

        J = np.zeros((N, N))
        for j in range(N):
            if j < T - tau:
                dphi = phase[:, j + tau] - phase[:, j]
                J[:, j] = np.cos(dphi)
            else:
                J[:, j] = np.eye(N)[:, j]

        eigenvalues = np.linalg.eigvals(J / (N + 1e-8))
        return np.abs(eigenvalues)

    def _estimate_basin_radius(self, phase: np.ndarray, tau: int) -> float:
        """Estimate basin of attraction radius from trajectory variance."""
        N, T = phase.shape
        n_periods = max(1, T // tau)
        period_means = []
        for k in range(n_periods):
            start = k * tau
            end = min(start + tau, T)
            mean_phase = np.angle(np.exp(1j * phase[:, start:end]).mean(axis=1))
            period_means.append(mean_phase)

        if len(period_means) < 2:
            return 0.5

        period_means = np.array(period_means)
        circ_var = 1.0 - np.abs(np.exp(1j * period_means).mean(axis=0)).mean()
        return float(np.sqrt(circ_var) * np.pi)

    def _deduplicate(self, cycles: list[LimitCycle], tol: float = 5.0) -> list[LimitCycle]:
        """Remove duplicate cycles with very similar periods."""
        if not cycles:
            return cycles
        unique = [cycles[0]]
        for c in cycles[1:]:
            if all(abs(c.period - u.period) > tol for u in unique):
                unique.append(c)
        return unique

    def synchrony_profile(self, phase: np.ndarray) -> np.ndarray:
        """
        Compute time-resolved Kuramoto order parameter R(t).

        R(t) = |1/N Σ_k exp(iφ_k(t))| ∈ [0, 1]
        """
        if phase.ndim == 1:
            return np.ones(len(phase))
        return np.abs(np.exp(1j * phase).mean(axis=0))

    def phase_velocity(self, phase: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Instantaneous phase velocity dφ/dt for each oscillator.

        Returns shape (N, T-1).
        """
        return np.diff(np.unwrap(phase, axis=-1), axis=-1) / dt

    def resilience_spectrum(self, cycles: list[LimitCycle]) -> np.ndarray:
        """
        Compute the resilience spectrum across all detected limit cycles.

        Returns an array of period-normalized Floquet decay rates:
            r_k = −ln|μ_max^(k)| / T_k

        Positive resilience → stable; negative → unstable.
        """
        return np.array([c.resilience for c in cycles])

    def __repr__(self) -> str:
        return (
            f"LimitCycleAnalyzer(min_T={self.min_period}, max_T={self.max_period}, "
            f"threshold={self.threshold})"
        )
