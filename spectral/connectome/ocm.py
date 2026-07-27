"""
connectome.ocm — Omics Connectome Matrix (theory.md §2).

Builds the Hermitian matrix

    H_ij = c_ij · exp( i · (θ_i − θ_j) )

from a phasor vector ψ (one sample / time slice) and a non-negative symmetric
coupling matrix c_ij. Hermitian by construction (H = H†) ⇒ real spectrum, a
prerequisite for the harmonic interpretation. The phase factor e^{i(θ_i−θ_j)}
is gauge-invariant under a global shift θ_i → θ_i + α.

This is a phase-native signal-processing model, NOT a physical quantum claim.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class OmicsConnectomeMatrix:
    """Construct the Hermitian OCM H from omics phasor data.

    Parameters
    ----------
    coupling_mode : {'pearson', 'coexpression', 'prior', 'uniform'}
        How to compute the coupling strength c_ij (theory.md §2.2).

        * ``'pearson'``      — |Pearson corr| of feature profiles across samples.
        * ``'coexpression'`` — soft-thresholded |corr|^β (WGCNA-style).
        * ``'prior'``        — a supplied non-negative adjacency (GRN/STRING).
        * ``'uniform'``      — c_ij = 1 for i≠j (unweighted graph).
    beta : float
        Soft-threshold power for 'coexpression' mode.
    min_coupling : float
        Floor for coupling strengths (clip below this to it).
    self_coupling : float
        Diagonal value c_ii (real; sets H_ii). Default 1.0.
    """

    _VALID_MODES = {"pearson", "coexpression", "prior", "uniform"}

    def __init__(
        self,
        coupling_mode: str = "pearson",
        beta: float = 6.0,
        min_coupling: float = 0.0,
        self_coupling: float = 1.0,
    ) -> None:
        if coupling_mode not in self._VALID_MODES:
            raise ValueError(
                f"coupling_mode must be one of {self._VALID_MODES}, got '{coupling_mode}'."
            )
        self.coupling_mode = coupling_mode
        self.beta = beta
        self.min_coupling = min_coupling
        self.self_coupling = self_coupling
        self.coupling_: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # coupling matrix c_ij
    # ------------------------------------------------------------------
    def compute_coupling(
        self,
        X: np.ndarray,
        prior: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Non-negative symmetric coupling matrix c ∈ ℝ^{N×N} (theory.md §2.2).

        Parameters
        ----------
        X : np.ndarray, shape (S, N)
            Omics matrix used to estimate feature-feature coupling.
        prior : np.ndarray, shape (N, N), optional
            Required for coupling_mode='prior'.
        """
        X = np.asarray(X, dtype=float)
        S, N = X.shape

        if self.coupling_mode == "uniform":
            C = np.ones((N, N))

        elif self.coupling_mode == "prior":
            if prior is None:
                raise ValueError("coupling_mode='prior' requires a `prior` adjacency.")
            C = np.abs(np.asarray(prior, dtype=float))
            if C.shape != (N, N):
                raise ValueError(f"prior must be (N,N)=({N},{N}); got {C.shape}.")

        else:  # pearson / coexpression
            if S < 2:
                C = np.ones((N, N))
            else:
                C = np.corrcoef(X.T)                       # (N, N)
                C = np.nan_to_num(C, nan=0.0)
                C = np.abs(C)
                if self.coupling_mode == "coexpression":
                    C = C ** self.beta

        C = np.clip(C, self.min_coupling, None)
        # symmetrise defensively and set the real diagonal
        C = 0.5 * (C + C.T)
        np.fill_diagonal(C, self.self_coupling)
        self.coupling_ = C
        return C

    # ------------------------------------------------------------------
    # OCM
    # ------------------------------------------------------------------
    def build(
        self,
        psi: np.ndarray,
        coupling: Optional[np.ndarray] = None,
        X: Optional[np.ndarray] = None,
        prior: Optional[np.ndarray] = None,
        amplitude_weighted: bool = True,
    ) -> np.ndarray:
        """Build the Hermitian OCM H for a single slice (theory.md §2.1).

            H_ij = c_ij · ψ_i · conj(ψ_j) = c_ij · r_i r_j · exp( i (θ_i − θ_j) )

        With ``amplitude_weighted=True`` (default) the full phasor vertex
        ψ_i = r_i e^{iθ_i} enters, so H = diag(ψ) C diag(ψ)† is a *congruence* of
        the coupling — the spectrum is sample-specific (amplitudes differ across
        samples). With ``amplitude_weighted=False`` only the phase enters
        (r_i≡1), giving the pure-phase form H = diag(u) C diag(u)†, a unitary
        similarity whose spectrum equals that of C for every slice. In both
        cases H is Hermitian and the phase factor is gauge-invariant under a
        global shift θ_i → θ_i + α.

        Parameters
        ----------
        psi : np.ndarray, complex, shape (N,)
            Phasor vector for one sample/time slice (ψ_i = r_i e^{iθ_i}).
        coupling : np.ndarray, shape (N, N), optional
            Precomputed coupling matrix. If None, computed from X (or prior).
        X : np.ndarray, shape (S, N), optional
            Omics matrix to derive coupling (used when coupling is None).
        prior : np.ndarray, optional
            Prior adjacency for coupling_mode='prior'.
        amplitude_weighted : bool
            Include the phasor amplitudes r_i (default True; see above).

        Returns
        -------
        np.ndarray, complex, shape (N, N) — Hermitian.
        """
        psi = np.asarray(psi, dtype=complex).ravel()
        N = psi.size
        theta = np.angle(psi)

        if coupling is None:
            if X is None:
                raise ValueError("Provide either `coupling` or `X` to build the OCM.")
            coupling = self.compute_coupling(X, prior=prior)
        coupling = np.asarray(coupling, dtype=float)
        if coupling.shape != (N, N):
            raise ValueError(f"coupling shape {coupling.shape} != ({N},{N}).")

        # vertex factor v_i v_j*:  v = ψ (amplitude-weighted) or v = e^{iθ} (phase-only)
        v = psi if amplitude_weighted else np.exp(1j * theta)
        vertex_factor = np.outer(v, np.conj(v))            # (N, N), Hermitian
        H = coupling * vertex_factor
        # enforce exact Hermiticity against float noise
        H = 0.5 * (H + H.conj().T)
        return H

    # ------------------------------------------------------------------
    # non-zero-flux (magnetic-Laplacian-style) variant — ADDITIVE
    # ------------------------------------------------------------------
    def build_magnetic(
        self,
        coupling: np.ndarray,
        antisym: np.ndarray,
        q: float = 0.1,
        amplitude: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build the Hermitian *non-zero-flux* magnetic OCM (see connectome.magnetic).

            H_ij = w_ij · exp( i · 2π q · A_ij )   (A antisymmetric ⇒ non-zero flux)

        Unlike :meth:`build` (gradient phase ⇒ zero cycle flux, spectrum = spec C),
        the edge phase here carries an antisymmetric orientation A that is not a
        coboundary, so the cycle flux 2π q (A_ij+A_jk+A_ki) is generally non-zero
        and the spectrum DEPENDS on q. Delegates to
        :func:`spectralomics.connectome.magnetic.build_magnetic`; does not alter
        :meth:`build`.
        """
        from biophasor.spectral.connectome.magnetic import build_magnetic
        return build_magnetic(coupling, antisym, q=q, amplitude=amplitude)
