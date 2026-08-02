"""
compartments.weights — compartment-weight readout derived from the CCM.

The *CompartmentWeights* class extracts a compact profile from the
compartment covariance matrix (CCM): per-compartment weights, pairwise
coupling strengths, dominance ranking, and an aggregate signature. It
summarises how energy fluctuation is distributed across the five omics
compartments (Clock, Redox, Energy, Signalling, Biosynthesis).

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import numpy as np

from biophasor.spectral.quantum.compartment_covariance import CompartmentCovariance
from biophasor.spectral.quantum.compartment_model import COMPARTMENTS


class CompartmentWeights:
    """Extract a compartment-weight profile from the CCM.

    The CCM diagonal gives the per-compartment energy-fluctuation variances;
    this class normalises them into compartment weights and summarises the
    off-diagonal coupling structure and an aggregate covariance signature.

    Parameters
    ----------
    ccm:
        A :class:`CompartmentCovariance` instance (analysis helper).
    """

    def __init__(self, ccm: CompartmentCovariance) -> None:
        self.ccm = ccm

    # ------------------------------------------------------------------
    # Weight & coupling extraction
    # ------------------------------------------------------------------

    def compartment_weights(self, ccm_matrix: np.ndarray) -> dict:
        """Normalise the CCM diagonal into per-compartment weights.

        ``w_a = M_{aa} / Σ_b M_{bb}``

        Values are non-negative and sum to 1 (or are uniform if the diagonal
        is zero).
        """
        diag = np.maximum(np.diag(ccm_matrix), 0.0)   # guard negatives
        total = diag.sum()
        if total < 1e-12:
            weights = np.ones(len(COMPARTMENTS)) / len(COMPARTMENTS)
        else:
            weights = diag / total
        return {comp: float(w) for comp, w in zip(COMPARTMENTS, weights)}

    def coupling_strengths(self, ccm_matrix: np.ndarray) -> dict:
        """Off-diagonal CCM elements as pairwise coupling strengths.

        ``I_{ab} = |M_{ab}|`` for ``a ≠ b`` (upper triangle, 10 entries).
        """
        n = len(COMPARTMENTS)
        strengths = {}
        for i in range(n):
            for j in range(i + 1, n):
                key = f"{COMPARTMENTS[i]}-{COMPARTMENTS[j]}"
                strengths[key] = float(abs(ccm_matrix[i, j]))
        return strengths

    def signature(self, ccm_matrix: np.ndarray) -> np.ndarray:
        """Flattened CCM vector ``σ = vec(M) ∈ ℝ^{25}`` — covariance fingerprint."""
        return ccm_matrix.flatten().astype(float)

    # ------------------------------------------------------------------
    # Aggregate config
    # ------------------------------------------------------------------

    def compartment_config(self, ccm_matrix: np.ndarray) -> dict:
        """Build a serialisable compartment-weight configuration dictionary.

        Keys:
            * ``weights``       — per-compartment weights (dict)
            * ``couplings``     — pairwise off-diagonal strengths (dict)
            * ``signature``     — 25-element covariance fingerprint (list)
            * ``coupling_norm`` — off-diagonal Frobenius norm (float)
            * ``coherence``     — coherence kappa in [0, 1] (float)
            * ``dominant``      — dominant compartment name (str)
            * ``ranking``       — dominance ranking [(name, variance), ...]
        """
        return {
            "weights":       self.compartment_weights(ccm_matrix),
            "couplings":     self.coupling_strengths(ccm_matrix),
            "signature":     self.signature(ccm_matrix).tolist(),
            "coupling_norm": self.ccm.coupling_norm(ccm_matrix),
            "coherence":     self.ccm.coherence(ccm_matrix),
            "dominant":      self.ccm.dominant_compartment(ccm_matrix),
            "ranking":       self.ccm.dominance_ranking(ccm_matrix),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"CompartmentWeights(ccm={self.ccm!r})"
