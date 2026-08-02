"""
omics.compartment_weights — Compartment spectral-weight readout from the Compartment Coupling Matrix (theory.md §5.3).

Eigendecomposes the PSD covariance form G = M† M of the Compartment Coupling
Matrix (CCM) and extracts:

    weights (π^diag_a)   per-compartment diagonal energy of G, normalised to
                         Σ = 1 — how much inter-compartment coupling each named
                         compartment carries; drives the dominance ranking.
                         Returned as `weights` / `weight_vector`.
    spectral_weights     the PSD eigenvalue distribution π_a = w_a / Σ w_a
                         (the abstract spectral mode weights, Σ = 1).
    κ    global coherence = w_1 / Σ w_a  ∈ [0,1]  (concentration in mode 1).

This turns the CCM into interpretable per-compartment weights and a single
coherence scalar.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from biophasor.spectral.omics.ccm import COMPARTMENTS


class CompartmentWeights:
    """Compartment-weight / coherence extractor from the CCM.

    Parameters
    ----------
    compartments : sequence of str, optional
        Compartment labels (default: the five biological compartments).
    """

    def __init__(self, compartments: Optional[Sequence[str]] = None) -> None:
        self.compartments = list(compartments) if compartments is not None else list(COMPARTMENTS)
        self.weights_: Optional[np.ndarray] = None
        self.coherence_: Optional[float] = None

    # ------------------------------------------------------------------
    def analyze(self, M: np.ndarray) -> dict:
        """Compute the compartment-weight readout from the CCM M (theory.md §5.3).

        Parameters
        ----------
        M : np.ndarray, complex, shape (n_comp, n_comp)
            Hermitian Compartment Coupling Matrix.

        Returns
        -------
        dict with keys:
            weights            : {compartment: π_a}
            weight_vector      : np.ndarray (n_comp,) π_a
            dominance_ranking  : [compartment, ...] descending π_a
            dominant           : compartment with largest π_a
            coherence_kappa    : κ ∈ [0,1]
            eigenvalues        : PSD eigenvalues w_a (descending)
        """
        M = np.asarray(M, dtype=complex)
        G = M.conj().T @ M                                 # PSD (theory.md §5.2)
        G = 0.5 * (G + G.conj().T)
        w = np.linalg.eigvalsh(G)                          # ascending, real ≥ 0
        w = np.clip(np.real(w), 0.0, None)[::-1]           # descending, non-negative
        total = w.sum()
        if total <= 0:
            pi = np.full(self.n_comp, 1.0 / self.n_comp)
            kappa = 1.0 / self.n_comp
        else:
            pi = w / total
            kappa = float(w[0] / total)

        # Per-compartment weight: diagonal energy of G, normalised (interpretable
        # as how much spectral stress each named compartment carries).
        diag_energy = np.clip(np.real(np.diag(G)), 0.0, None)
        d_total = diag_energy.sum()
        comp_weights = diag_energy / d_total if d_total > 0 else np.full(self.n_comp, 1.0 / self.n_comp)

        self.weights_ = comp_weights
        self.coherence_ = kappa

        order = np.argsort(comp_weights)[::-1]
        ranking = [self.compartments[i] for i in order]

        return {
            "weights": {self.compartments[i]: float(comp_weights[i]) for i in range(self.n_comp)},
            "weight_vector": comp_weights,
            "dominance_ranking": ranking,
            "dominant": ranking[0],
            "coherence_kappa": kappa,
            "spectral_weights": pi,
            "eigenvalues": w,
        }

    # ------------------------------------------------------------------
    @property
    def n_comp(self) -> int:
        return len(self.compartments)
