"""
compartments.covariance — the compartment covariance matrix (CCM).

The CCM is a 5x5 real symmetric, positive semi-definite matrix: the quantum
covariance of the five compartment observables in the ground state. Diagonal
elements give the energy-fluctuation variance of each compartment;
off-diagonal elements capture cross-compartment covariance. It is the quantum
analog of the classical omics Compartment Coupling Matrix.

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from biophasor.spectral.quantum.compartment_model import COMPARTMENTS


class CompartmentCovariance:
    """Readout helpers on the compartment covariance matrix (CCM).

    The CCM itself is produced by
    :meth:`compartments.compartment_model.CompartmentModel.compute_ccm`
    (the quantum covariance of the compartment observables in a state
    ``|ψ⟩``).  This class provides analysis helpers that operate on that
    matrix: dominant compartment, off-diagonal coupling, coherence, and a
    labelled DataFrame view.

    Parameters
    ----------
    state:
        Optional quantum state vector ``|ψ⟩`` (retained for reference / repr).
    """

    def __init__(self, state: Optional[np.ndarray] = None) -> None:
        self.state = None if state is None else np.asarray(state, dtype=complex)

    # ------------------------------------------------------------------
    # Core analysis helpers
    # ------------------------------------------------------------------

    def off_diagonal_coupling(self, ccm: np.ndarray) -> float:
        """Sum of squared off-diagonal elements — cross-compartment coupling.

        ``C = Σ_{a≠b} |M_{ab}|²``

        Measures the degree to which the compartments co-fluctuate: large
        ``C`` means energy flows freely between compartments (integrated
        regulation); small ``C`` means the compartments fluctuate
        independently.
        """
        off = ccm.copy()
        np.fill_diagonal(off, 0.0)
        return float(np.sum(np.abs(off) ** 2))

    def coupling_norm(self, ccm: np.ndarray) -> float:
        """Off-diagonal Frobenius norm ``‖M_off‖_F`` — total coupling strength."""
        off = ccm.copy()
        np.fill_diagonal(off, 0.0)
        return float(np.linalg.norm(off, "fro"))

    def dominant_compartment(self, ccm: np.ndarray) -> str:
        """Name of the compartment carrying the most energy fluctuation."""
        diag = np.diag(ccm)
        return COMPARTMENTS[int(np.argmax(diag))]

    def dominance_ranking(self, ccm: np.ndarray) -> list:
        """Compartments ranked by diagonal variance, descending.

        Returns
        -------
        list of (str, float)
            ``[(compartment_name, variance), ...]`` sorted high-to-low.
        """
        diag = np.maximum(np.diag(ccm), 0.0)
        order = np.argsort(diag)[::-1]
        return [(COMPARTMENTS[i], float(diag[i])) for i in order]

    def coherence(self, ccm: np.ndarray) -> float:
        """Coherence measure ``kappa`` in ``[0, 1]``.

        ``kappa = ‖diag(M)‖_2 / ‖M‖_F
                = sqrt(Σ_a M_{aa}²) / sqrt(Σ_{a,b} M_{ab}²)``

        Because the CCM is positive semi-definite with a non-negative
        diagonal, this ratio is guaranteed to lie in ``[0, 1]``: it is 1 when
        the CCM is purely diagonal (compartments fluctuate independently) and
        decreases toward 0 as off-diagonal covariance grows.
        """
        total_norm = float(np.linalg.norm(ccm, "fro"))
        if total_norm < 1e-12:
            return 1.0
        diag_norm = float(np.linalg.norm(np.diag(ccm)))
        return diag_norm / total_norm

    # ------------------------------------------------------------------
    # Deviation diagnostics (relative to a reference CCM)
    # ------------------------------------------------------------------

    def deviation_from_reference(
        self,
        ccm: np.ndarray,
        ccm_reference: np.ndarray,
    ) -> tuple:
        """Deviation tensor ``δM = M − M⁰`` and its Frobenius norm.

        Returns ``(delta_ccm, frob_norm)``.
        """
        delta = np.asarray(ccm, dtype=float) - np.asarray(ccm_reference, dtype=float)
        frob = float(np.linalg.norm(delta, "fro"))
        return delta, frob

    def ccm_as_dataframe(self, ccm: np.ndarray) -> pd.DataFrame:
        """Return the CCM as a labelled pandas DataFrame (COMPARTMENTS index)."""
        return pd.DataFrame(ccm, index=COMPARTMENTS, columns=COMPARTMENTS)

    def __repr__(self) -> str:  # pragma: no cover
        if self.state is None:
            return "CompartmentCovariance(state=None)"
        norm = float(np.linalg.norm(self.state))
        return (
            f"CompartmentCovariance(fock_dim={self.state.shape[0]}, "
            f"|ψ| norm={norm:.4f})"
        )
