"""
connectome.magnetic — non-zero-flux (magnetic-Laplacian-style) OCM variant.

Motivation
----------
The default OCM (``connectome.ocm``) builds

    H_ij = c_ij · exp( i · (θ_i − θ_j) )        (gradient phase)

whose edge phase is a *pure gradient* of the vertex potential θ. The holonomy
(flux) around any cycle telescopes to zero (mod 2π):

    (θ_i−θ_j) + (θ_j−θ_k) + (θ_k−θ_i) ≡ 0 .

Hence the pure-phase gradient OCM is a *unitary congruence* of the coupling C
(``H = diag(u) C diag(u)†``, u_i = e^{iθ_i}) and its spectrum equals that of C
for every slice — the phase does no *topological* work. This is a deliberate,
interpretable special case, but it means the OCM cannot, by itself, resolve
*directed* structure.

This module provides a genuinely magnetic operator whose edge phase carries an
antisymmetric part A (A_ji = −A_ij) that is NOT a coboundary, so the cycle flux

    Φ_ijk = 2π q (A_ij + A_jk + A_ki)

is generally non-zero. The construction follows the magnetic-Laplacian /
Hermitian-adjacency lineage for directed graphs (Guo & Mohar 2017; Furutani
et al. 2019; Zhang et al. MagNet 2021; He et al. MSGNN 2022; Böttcher & Porter
2024). The operator stays Hermitian (real spectrum) but its spectrum now
DEPENDS on the flux — an Aharonov–Bohm-type dependence that a gradient phase
can never reproduce.

    H_ij = w_ij · exp( i · 2π q · A_ij )         (w symmetric ≥0, A antisymmetric)

Optionally amplitude-weighted:  H_ij → r_i r_j · H_ij .

The antisymmetric part A is estimated from a directed/signed relationship on a
time-course, e.g. the LEAD–LAG matrix (cross-correlation argmax lag) or a
SIGNED-correlation orientation.

This is a phase-native signal-processing model, NOT a physical quantum claim.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import correlate


# ----------------------------------------------------------------------
# antisymmetric orientation matrices A (A_ji = −A_ij)
# ----------------------------------------------------------------------
def lead_lag_antisymmetry(
    Xt: np.ndarray,
    max_lag: Optional[int] = None,
    detrend: bool = True,
) -> np.ndarray:
    """Antisymmetric lead–lag matrix A from a time-ordered omics matrix.

    A_ij > 0 means feature i *leads* feature j (its profile, shifted forward by
    the estimated lag, best aligns with j). A_ji = −A_ij by construction, so A
    is exactly antisymmetric and its cycle sums are generally non-zero.

    Parameters
    ----------
    Xt : np.ndarray, shape (S, N)
        Time-ORDERED omics matrix (rows are successive time points). Sort the
        samples by circadian/collection time before calling.
    max_lag : int, optional
        Search window (in samples) for the cross-correlation argmax. Default
        ``S // 4``.
    detrend : bool
        z-score each feature series before cross-correlating (default True).

    Returns
    -------
    np.ndarray, shape (N, N) — antisymmetric (float lag in samples).
    """
    Xt = np.asarray(Xt, dtype=float)
    if Xt.ndim != 2:
        raise ValueError(f"Xt must be 2-D (S, N); got shape {Xt.shape}.")
    S, N = Xt.shape
    if max_lag is None:
        max_lag = max(1, S // 4)
    max_lag = int(min(max_lag, S - 1))

    Z = Xt
    if detrend:
        Z = (Xt - Xt.mean(axis=0)) / (Xt.std(axis=0) + 1e-12)

    lags = np.arange(-(S - 1), S)
    center = S - 1
    win = slice(center - max_lag, center + max_lag + 1)
    wl = lags[win]

    A = np.zeros((N, N))
    for i in range(N):
        zi = Z[:, i]
        for j in range(i + 1, N):
            cc = correlate(zi, Z[:, j], mode="full") / S
            best = wl[int(np.argmax(cc[win]))]
            A[i, j] = best
            A[j, i] = -best
    return A


def signed_antisymmetry(
    Xt: np.ndarray,
    threshold: float = 0.0,
) -> np.ndarray:
    """Antisymmetric orientation from the *sign of the lag-1 lead–lag* asymmetry.

    A_ij = corr(x_i[t], x_j[t+1]) − corr(x_i[t+1], x_j[t]), the difference of
    one-step-ahead cross-correlations. This is antisymmetric (A_ji = −A_ij) and
    encodes a signed Granger-flavoured direction: positive when i's past
    predicts j's future more than the reverse. Entries with |A_ij| ≤ threshold
    are set to zero.

    Parameters
    ----------
    Xt : np.ndarray, shape (S, N)
        Time-ordered omics matrix.
    threshold : float
        Magnitude floor below which an orientation is set to 0.

    Returns
    -------
    np.ndarray, shape (N, N) — antisymmetric.
    """
    Xt = np.asarray(Xt, dtype=float)
    S, N = Xt.shape
    Z = (Xt - Xt.mean(axis=0)) / (Xt.std(axis=0) + 1e-12)
    past, future = Z[:-1], Z[1:]                       # (S-1, N)
    # C_fwd[i,j] = corr(x_i[t], x_j[t+1])
    C_fwd = (past.T @ future) / (S - 1)
    A = C_fwd - C_fwd.T                                # antisymmetric
    if threshold > 0.0:
        A = np.where(np.abs(A) > threshold, A, 0.0)
    return A


def cycle_flux(A: np.ndarray, q: float, i: int, j: int, k: int) -> float:
    """Signed flux 2π q (A_ij + A_jk + A_ki) through triangle (i, j, k)."""
    A = np.asarray(A, dtype=float)
    return 2.0 * np.pi * q * (A[i, j] + A[j, k] + A[k, i])


# ----------------------------------------------------------------------
# magnetic OCM builder
# ----------------------------------------------------------------------
def build_magnetic(
    coupling: np.ndarray,
    antisym: np.ndarray,
    q: float = 0.1,
    amplitude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build the Hermitian magnetic OCM with genuinely non-zero cycle flux.

        H_ij = w_ij · exp( i · 2π q · A_ij )                (amplitude=None)
        H_ij = r_i r_j · w_ij · exp( i · 2π q · A_ij )      (amplitude given)

    where ``w = coupling`` is symmetric ≥ 0 and ``A = antisym`` is antisymmetric
    (A_ji = −A_ij). Because A is not a coboundary, the flux 2π q (A_ij+A_jk+A_ki)
    around a cycle is generally non-zero, so the spectrum DEPENDS on q — unlike
    the gradient-phase OCM. H is Hermitian by construction (real spectrum).

    At ``q = 0`` this reduces exactly to the amplitude-weighted real coupling
    operator (zero flux). Varying q sweeps the holonomy (Aharonov–Bohm phase).

    Parameters
    ----------
    coupling : np.ndarray, shape (N, N)
        Non-negative symmetric coupling w_ij (e.g. |Pearson|).
    antisym : np.ndarray, shape (N, N)
        Antisymmetric orientation A_ij (lead–lag or signed). Symmetrisation of
        its symmetric part is discarded; only the antisymmetric part enters the
        phase (enforced defensively).
    q : float
        Charge / flux parameter. q = 0 ⇒ zero flux (gradient special case).
    amplitude : np.ndarray, shape (N,), optional
        Per-feature amplitudes r_i (e.g. |ψ_i|). If given, H is amplitude-
        weighted (a congruence), making the spectrum sample-specific.

    Returns
    -------
    np.ndarray, complex, shape (N, N) — Hermitian.
    """
    w = np.asarray(coupling, dtype=float)
    A = np.asarray(antisym, dtype=float)
    if w.shape != A.shape or w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"coupling {w.shape} and antisym {A.shape} must be equal square (N,N).")
    N = w.shape[0]

    # keep only the antisymmetric part of A so H is exactly Hermitian
    A = 0.5 * (A - A.T)
    # keep only the symmetric part of the weight
    w = 0.5 * (w + w.T)

    H = w * np.exp(1j * 2.0 * np.pi * q * A)
    if amplitude is not None:
        r = np.asarray(amplitude, dtype=float).ravel()
        if r.size != N:
            raise ValueError(f"amplitude size {r.size} != N={N}.")
        H = np.outer(r, r) * H

    # enforce exact Hermiticity against float noise
    H = 0.5 * (H + H.conj().T)
    return H
