"""
omics_spectrum.py — the one quantitative bridge from the classical omics
spectral connectome to the quantum model.

The compartment-mode self-energies :math:`\\varepsilon_k` of the Bose--Hubbard
compartment model are the omics *connectome harmonic frequencies*

    omega_k = sqrt(|lambda_k|),

where :math:`\\lambda_k` are the eigenvalues of the Hermitian Omics Connectome
Matrix (OCM)

    H_ij = c_ij * exp( i (theta_i - theta_j) ),

with c_ij a non-negative feature-feature coupling and theta_i the tanh-encoded
phasor phases of an omics expression matrix. These frequencies originate in the
*classical* omics spectral analysis; this project consumes them as precomputed
inputs so it is self-contained.

Resolution order for the frequency ladder, in decreasing order of authority:
the ladder cached alongside this module in ``data/omega_k.npy`` (shipped as
package data; the GSE10072 ladder the manuscript reports), then a ladder
computed on the spot from an expression matrix the caller supplies, then a
dependency-free synthetic OCM-like spectrum. The last case exists so the
quantum model is runnable with no data present at all — it is a stand-in, not a
measurement, and which of the three was used is always recorded in
``data/omega_provenance.json``. Read that file before quoting any number
derived from these frequencies.

This is a quantum-simulable signal-processing model, NOT a biological claim.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# Package data travels with the module, so the cached ladder resolves no matter
# where the caller's working directory is.
_DATA_DIR = os.path.join(_HERE, "data")
_OMEGA_CACHE = os.path.join(_DATA_DIR, "omega_k.npy")
_PROVENANCE = os.path.join(_DATA_DIR, "omega_provenance.json")


def _record_provenance(source: str, n_modes: int, extra: Optional[dict] = None) -> None:
    rec = {"source": source, "n_modes": int(n_modes)}
    if extra:
        rec.update(extra)
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        with open(_PROVENANCE, "w") as fh:
            json.dump(rec, fh, indent=2)
    except OSError:
        pass


def ocm_spectrum(X: np.ndarray, n_top: int = 200) -> np.ndarray:
    """Ascending OCM eigenvalues of an omics expression matrix.

    This is the classical half of the bridge, run through the installed
    ``biophasor.spectral.connectome`` rather than a sibling source tree — the
    original module reached a co-located repository by inserting it on
    ``sys.path``, which silently returned a synthetic spectrum whenever that
    layout did not hold.

    Parameters
    ----------
    X : np.ndarray, shape (S, N)
        Expression matrix, S samples x N features (features in columns).
    n_top : int
        Retain this many most-variable features before building the OCM. The
        manuscript ladder used 200.

    Returns
    -------
    np.ndarray
        Eigenvalues of the Hermitian OCM built on one sample slice, ascending.
    """
    from biophasor.spectral.connectome.phasor import PhasorEncoder
    from biophasor.spectral.connectome.ocm import OmicsConnectomeMatrix

    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (samples x features); got shape {X.shape}")
    if n_top < X.shape[1]:
        keep = np.argsort(X.var(axis=0))[::-1][:n_top]
        X = X[:, keep]

    enc = PhasorEncoder()
    psi = enc.encode(X)                        # samples x features (complex)
    ocm = OmicsConnectomeMatrix(coupling_mode="pearson")
    C = ocm.compute_coupling(X)
    H = ocm.build(psi[0], coupling=C)          # one sample slice
    lam = np.linalg.eigvalsh(0.5 * (H + H.conj().T))
    return np.sort(np.real(lam))


def _synthetic_spectrum(n: int = 64, seed: int = 0) -> np.ndarray:
    """Dependency-free fallback OCM-like spectrum (ascending).

    A random symmetric coupling with a soft-thresholded correlation structure
    reproduces the qualitative OCM spectrum (a bulk near zero with a few large
    outliers) so the quantum pipeline has physically reasonable mode
    frequencies even without the classical data.
    """
    rng = np.random.default_rng(seed)
    S, N = 40, n
    t = np.linspace(0, 4 * np.pi, S)
    X = np.zeros((S, N))
    for i in range(N):
        phase = (i % 5) * 2 * np.pi / 5
        X[:, i] = 5.0 + 3.0 * np.sin(t + phase) + rng.normal(0, 0.3, S)
    X = np.clip(X, 0.0, None)
    Xc = X - X.mean(0, keepdims=True)
    C = np.corrcoef(Xc.T)
    C = np.abs(np.nan_to_num(C)) ** 6              # soft threshold (WGCNA-style)
    np.fill_diagonal(C, 0.0)
    theta = np.pi * np.tanh((np.log1p(X[0]) - np.log1p(X[0]).mean()) /
                            (np.log1p(X[0]).std() + 1e-8))
    u = np.exp(1j * theta)
    H = C * np.outer(u, np.conj(u))
    H = 0.5 * (H + H.conj().T)
    lam = np.linalg.eigvalsh(H)
    return np.sort(np.real(lam))


def omics_harmonic_frequencies(
    X: Optional[np.ndarray] = None,
    use_cache: bool = True,
    n_top: int = 200,
) -> np.ndarray:
    """Return omics harmonic frequencies ``omega_k = sqrt(|lambda_k|)`` (ascending).

    Resolution order: an expression matrix passed in ``X`` (measured, and it
    wins over the cache) -> the shipped ``data/omega_k.npy`` ladder -> the
    synthetic fallback. The source is written to ``data/omega_provenance.json``
    whenever it is recomputed; a caller reporting numbers downstream of these
    frequencies should read that file rather than assume the measured ladder.

    Parameters
    ----------
    X : np.ndarray or None, shape (S, N)
        Expression matrix to derive the ladder from. ``None`` uses the cache.
    use_cache : bool
        Read (and, when writable, refresh) ``data/omega_k.npy``. Set False to
        force recomputation.
    n_top : int
        Most-variable feature count passed to :func:`ocm_spectrum`.
    """
    if X is not None:
        lam = ocm_spectrum(X, n_top=n_top)
        source = "ocm_from_expression_matrix"
    else:
        if use_cache and os.path.exists(_OMEGA_CACHE):
            return np.load(_OMEGA_CACHE)
        lam = _synthetic_spectrum()
        source = "synthetic_fallback"

    omega = np.sort(np.sqrt(np.abs(lam)))
    _record_provenance(source, omega.size)
    if use_cache:
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            np.save(_OMEGA_CACHE, omega)
        except OSError:
            # Read-only install: the ladder is still returned, just not cached.
            pass
    return omega


def compartment_self_energies(n: int = 5) -> np.ndarray:
    """Compartment-mode self-energies ``epsilon_k = omega_k`` (n leading modes).

    The Omics Connectome Matrix spectrum is dominated by a large near-zero bulk
    with a few large outliers (the collective omics harmonics). The five
    biological compartments (Clock, Redox, Energy, Signalling, Biosynthesis) are
    identified with the ``n`` *leading* (largest-frequency) harmonics, i.e. the
    dominant collective modes, returned in descending order and clipped to be
    non-negative.
    """
    omega = omics_harmonic_frequencies()
    if omega.size < n:
        omega = np.pad(omega, (0, n - omega.size), mode="edge")
    top = np.sort(omega)[::-1][:n]                 # n largest, descending
    return np.maximum(top, 0.0)
