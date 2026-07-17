"""
biophasor.dynamics.kuramoto — Biological Kuramoto oscillator.

Implements the Kuramoto model tuned for biological gene networks:

    dφ_i/dt = ω_i + (K/N) Σ_j A_ij sin(φ_j − φ_i) + η_i(t)

where:
    ω_i  = natural frequency of gene i (e.g. from periodicity analysis)
    K    = coupling constant
    A_ij = gene regulatory network adjacency matrix
    η_i  = Gaussian noise term

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional

import numpy as np


class BioKuramoto:
    """
    Biological Kuramoto oscillator network.

    Parameters
    ----------
    n_oscillators : int
        Number of genes / biological oscillators.
    coupling : float
        Global coupling constant K.  K > K_c (critical) → synchronisation.
    omega : np.ndarray | None
        Natural frequencies (rad/s), shape (n_oscillators,).
        If None, drawn from a Lorentzian distribution.
    adjacency : np.ndarray | None
        Weighted adjacency matrix (n_oscillators × n_oscillators) representing
        gene regulatory connections.  If None, all-to-all coupling is used.
    noise : float
        Standard deviation of the Gaussian noise term σ_η.
    """

    def __init__(
        self,
        n_oscillators: int,
        coupling: float = 1.0,
        omega: Optional[np.ndarray] = None,
        adjacency: Optional[np.ndarray] = None,
        noise: float = 0.0,
        seed: int = 42,
    ) -> None:
        self.n = n_oscillators
        self.K = coupling
        self.rng = np.random.RandomState(seed)
        self.noise = noise

        if omega is None:
            # Lorentzian (Cauchy) distribution — classic Kuramoto assumption
            gamma = 0.5
            self.omega = self.rng.standard_cauchy(n_oscillators) * gamma
        else:
            self.omega = np.asarray(omega, dtype=np.float64)

        if adjacency is None:
            # All-to-all coupling (standard Kuramoto)
            self.A = np.ones((n_oscillators, n_oscillators), dtype=np.float64)
            np.fill_diagonal(self.A, 0.0)
        else:
            self.A = np.asarray(adjacency, dtype=np.float64)

        # Initial phase (random)
        self.phi = self.rng.uniform(-np.pi, np.pi, n_oscillators)

    # ── Simulation ────────────────────────────────────────────────────────────

    def step(self, dt: float = 0.01) -> np.ndarray:
        """
        Advance the system by one Euler step.

        Returns
        -------
        np.ndarray, shape (n_oscillators,)   updated phases
        """
        # Vectorised coupling: dφ_i = Σ_j A_ij sin(φ_j − φ_i)
        diff = self.phi[np.newaxis, :] - self.phi[:, np.newaxis]   # (N, N)
        coupling_term = (self.A * np.sin(diff)).sum(axis=1) * self.K / self.n
        eta = self.rng.normal(0, self.noise, self.n) if self.noise > 0 else 0.0
        self.phi += dt * (self.omega + coupling_term + eta)
        self.phi = ((self.phi + np.pi) % (2 * np.pi)) - np.pi   # wrap
        return self.phi.copy()

    def simulate(
        self,
        n_steps: int = 1000,
        dt: float = 0.01,
        record_every: int = 1,
    ) -> np.ndarray:
        """
        Run the full simulation.

        Parameters
        ----------
        n_steps : int
        dt : float
        record_every : int   record state every N steps

        Returns
        -------
        np.ndarray, shape (n_recorded, n_oscillators)
        """
        trajectory = []
        for step_i in range(n_steps):
            self.step(dt)
            if step_i % record_every == 0:
                trajectory.append(self.phi.copy())
        return np.array(trajectory)

    # ── Metrics ───────────────────────────────────────────────────────────────

    @property
    def order_parameter(self) -> float:
        """
        Kuramoto order parameter R:

            R·e^{iΨ} = (1/N) Σ_j e^{iφ_j}

        R → 1 : full synchronisation; R → 0 : incoherence.
        """
        return float(np.abs(np.exp(1j * self.phi).mean()))

    @property
    def mean_phase(self) -> float:
        """Global mean phase Ψ in (−π, π]."""
        return float(np.angle(np.exp(1j * self.phi).mean()))

    def critical_coupling(self) -> float:
        """
        Estimate the critical coupling Kc above which synchronisation emerges.

        For the all-to-all network with Lorentzian frequency distribution:
            Kc = 2 · γ    (where γ is the half-width at half-maximum)

        Here we use an empirical estimate from the frequency variance.
        """
        return 2.0 * np.abs(self.omega).mean()
