"""
biophasor.cst.dynamics — CSTDynamics: closed-loop Cell State Tensor evolution.

Evolves the CellStateTensor under Biophasor dissipative dynamics:

    ∂CST/∂t = F_BP(CST, I_ext(t))

where F_BP combines:
  1. Intrinsic phase dynamics (Kuramoto-like coupling within regulatory modules)
  2. Cross-regulatory modulation (epigenomic → transcriptomic → proteomic cascade)
  3. External perturbation forcing I_ext (drug, growth factor, cytokine)
  4. Stochastic noise (thermal + transcriptional burst noise)

Biological applications:
  - Cell-cycle progression simulation
  - Drug-response trajectory prediction
  - In-silico perturbation screening (phase-flip synthetic lethality)
  - Stem cell differentiation simulation via attractor transitions

Reference: Biophasor Manuscript — Section "Biophasor Formulation"
           Biophasor Book — Ch 10 "Cell State Tensor"

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from biophasor.cst.tensor import CellStateTensor


class CSTDynamics:
    """
    Closed-loop Cell State Tensor evolution.

    Simulates how the CST evolves over time given:
    - Intrinsic regulatory module Kuramoto coupling
    - Cross-regulatory cascade (epigenomic drives transcriptomic drives proteomic)
    - External perturbation (drug, cytokine, siRNA knockdown)
    - Stochastic transcriptional noise

    Parameters
    ----------
    cst : CellStateTensor
        Initial cell state.
    coupling : float
        Intra-module Kuramoto coupling constant K.
    cross_coupling : float
        Cross-regulatory coupling strength (cascade).
    noise : float
        Stochastic noise amplitude σ.
    seed : int

    Examples
    --------
    >>> from biophasor.cst import CellStateTensor, CSTDynamics
    >>> cst0 = CellStateTensor.random()
    >>> dyn = CSTDynamics(cst0, coupling=1.5, noise=0.05)
    >>> cst_evolved = dyn.simulate(n_steps=500, dt=0.01)
    >>> print(cst_evolved.global_coherence())
    """

    # Cross-regulatory cascade order (upstream drives downstream)
    _CASCADE_ORDER = [
        "epigenetic_mark",
        "chromatin_state",
        "TF_activity",
        "signalling_pathway",
        "metabolic_flux",
        "splicing_program",
    ]

    def __init__(
        self,
        cst: CellStateTensor,
        coupling: float = 1.0,
        cross_coupling: float = 0.1,
        noise: float = 0.01,
        seed: int = 42,
    ) -> None:
        self.cst = cst
        self.K = coupling
        self.K_cr = cross_coupling
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self._phase = self.cst.phase.copy()     # (R, T, H)
        self._amp = self.cst.amplitude.copy()

    def _intra_module_step(
        self,
        phase: np.ndarray,
        r: int,
        dt: float,
    ) -> np.ndarray:
        """Kuramoto coupling within one regulatory module (over homeostatic axis)."""
        phi = phase[r]   # (T, H)
        diff = phi[:, np.newaxis, :] - phi[np.newaxis, :, :]   # (T, T, H)
        coupling = np.sin(diff).mean(axis=1) * self.K / self.cst.n_temporal
        return phi + dt * coupling

    def _cross_regulatory_step(
        self,
        phase: np.ndarray,
        r: int,
        dt: float,
    ) -> np.ndarray:
        """Cross-regulatory cascade from upstream module r-1 → module r."""
        if r == 0:
            return np.zeros_like(phase[r])
        phi_upstream = phase[r - 1]
        phi_current = phase[r]
        return self.K_cr * np.sin(phi_upstream - phi_current) * dt

    def step(
        self,
        phase: np.ndarray,
        dt: float,
        I_ext: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        One integration step of the CST dynamics.

        Parameters
        ----------
        phase : np.ndarray, shape (R, T, H)   current phase snapshot
        dt : float
        I_ext : np.ndarray | None, shape (T,)
            External perturbation forcing at this step
            (e.g., drug effect on transcription factor activity).

        Returns
        -------
        np.ndarray  updated phase, same shape
        """
        new_phase = phase.copy()
        R = phase.shape[0]
        for r in range(R):
            intra = self._intra_module_step(phase, r, dt)
            cross = self._cross_regulatory_step(phase, r, dt)
            eta = (
                self.rng.normal(0, self.noise, phase[r].shape)
                if self.noise > 0 else 0.0
            )

            # External forcing (applied to TF_activity module, index 2)
            ext = 0.0
            if I_ext is not None and r == min(2, R - 1):
                ext = I_ext[:, np.newaxis] * dt

            new_phase[r] = intra + cross + eta + ext

        # Wrap to (−π, π]
        new_phase = ((new_phase + np.pi) % (2 * np.pi)) - np.pi
        return new_phase

    def simulate(
        self,
        n_steps: int = 500,
        dt: float = 0.01,
        perturbation_stream: Optional[Callable[[int], np.ndarray]] = None,
        record_every: int = 1,
    ) -> CellStateTensor:
        """
        Run closed-loop CST simulation.

        Parameters
        ----------
        n_steps : int   Number of time integration steps.
        dt : float    Integration step (arbitrary time units).
        perturbation_stream : callable | None
            Function f(step_i) → np.ndarray, shape (n_temporal,)
            providing external perturbation at each step.
        record_every : int   Save state every N steps.

        Returns
        -------
        CellStateTensor  evolved state trajectory
        """
        # Use last homeostatic slice to initialise
        phase_snapshot = self._phase[:, :, -1:]   # (R, T, 1)
        trajectory = []

        for step_i in range(n_steps):
            I_ext = (
                perturbation_stream(step_i)
                if perturbation_stream is not None
                else None
            )
            phase_snapshot = self.step(phase_snapshot, dt, I_ext=I_ext)
            if step_i % record_every == 0:
                trajectory.append(phase_snapshot[:, :, 0].copy())

        # Stack trajectory → (R, T, H_new)
        traj_arr = np.stack(trajectory, axis=-1)   # (R, T, H_rec)
        amp_const = self._amp[:, :, -1:] * np.ones_like(traj_arr)
        tensor = amp_const * np.exp(1j * traj_arr)

        return CellStateTensor(
            tensor=tensor,
            regulatory_names=self.cst.regulatory_names,
            temporal_names=self.cst.temporal_names,
            metadata={
                "source": "CSTDynamics.simulate",
                "n_steps": n_steps,
                "dt": dt,
                "K": self.K,
                "K_cr": self.K_cr,
            },
        )

    def phase_flip(self, gene_module: int) -> np.ndarray:
        """
        Apply a phase-flip (π-rotation) to a gene module — models gene knockout.

        Parameters
        ----------
        gene_module : int   Index of the regulatory module to knock out.

        Returns
        -------
        np.ndarray  modified phase array
        """
        phase = self._phase.copy()
        phase[gene_module] += np.pi
        phase = ((phase + np.pi) % (2 * np.pi)) - np.pi
        self._phase = phase
        return phase

    def synthetic_lethality_screen(
        self,
        n_steps: int = 200,
        dt: float = 0.01,
        coherence_threshold: float = 0.2,
    ) -> list[tuple[int, int, float]]:
        """
        Screen all pairwise double phase-flips for synthetic lethality.

        A gene pair (i, j) is flagged as synthetically lethal when the
        post-perturbation global coherence drops below the threshold.

        Parameters
        ----------
        n_steps : int   Simulation steps per perturbation.
        dt : float      Integration step.
        coherence_threshold : float   Lethality threshold.

        Returns
        -------
        list[tuple[int, int, float]]
            (module_i, module_j, post_perturbation_coherence)
            for all lethal pairs.
        """
        R = self.cst.n_regulatory
        lethal_pairs = []
        original_phase = self._phase.copy()

        for i in range(R):
            for j in range(i + 1, R):
                # Reset
                self._phase = original_phase.copy()
                # Double phase flip
                self._phase[i] += np.pi
                self._phase[j] += np.pi
                self._phase = ((self._phase + np.pi) % (2 * np.pi)) - np.pi
                # Simulate
                cst_post = self.simulate(n_steps=n_steps, dt=dt)
                gcm = cst_post.global_coherence()
                if gcm < coherence_threshold:
                    lethal_pairs.append((i, j, gcm))

        # Restore original
        self._phase = original_phase
        return lethal_pairs

    def __repr__(self) -> str:
        return (
            f"CSTDynamics(K={self.K}, K_cr={self.K_cr}, "
            f"noise={self.noise}, shape={self.cst.shape})"
        )
