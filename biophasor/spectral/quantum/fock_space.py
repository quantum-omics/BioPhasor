"""
fock_space.py — Truncated Fock-space algebra for the omics quantum model
=========================================================================

Physical picture
----------------
The Omics Connectome Matrix (OCM) has a Hermitian spectrum with eigenvalues
{lambda_k} whose leading harmonics give the omics normal-mode frequencies
omega_k = sqrt(|lambda_k|).  We treat each omics harmonic mode as an
independent quantum harmonic oscillator:

    H_0 = sum_k  hbar * omega_k * (n_k + 1/2)

where omega_k are the mode self-energies (from
``omics_spectrum.compartment_self_energies``) and n_k = a†_k a_k is the
number operator.

The Fock (occupation-number) basis for a single mode is
    {|0>, |1>, ..., |N_max>}    (N_max = max_occupation)

giving a per-mode Hilbert space of dimension (N_max + 1).  The full
N-mode Hilbert space is the tensor product, with dimension (N_max + 1)^N.

WARNING: Hilbert-space dimension grows exponentially.
    * n_modes <= 4             → dense matrices (full tensor product)
    * n_modes > 4              → sparse matrices via scipy.sparse

Creation / annihilation operators
----------------------------------
In the single-mode truncated basis:

    a† |n> = sqrt(n+1) |n+1>  (returns 0 if n+1 > N_max)
    a  |n> = sqrt(n)   |n-1>  (returns 0 if n = 0)

In the full N-mode space the operators are embedded via Kronecker products:

    A†_k = I ⊗ ... ⊗ I ⊗ a†  ⊗ I ⊗ ... ⊗ I
                         ^ mode k

Units:  hbar = 1,  m = 1  throughout (natural units).

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import warnings
from itertools import product as itertools_product
from math import factorial
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Threshold for switching between dense and sparse representations
# ---------------------------------------------------------------------------
_DENSE_MODE_LIMIT = 4   # n_modes <= this → dense; otherwise → sparse


class FockSpace:
    """Truncated Fock space for N quantum harmonic oscillator modes.

    Parameters
    ----------
    n_modes : int
        Number of omics harmonic modes (oscillators) to include.
    max_occupation : int, optional
        Maximum occupation number per mode (Fock-space truncation).
        The per-mode basis is {|0>, |1>, ..., |max_occupation>}, giving
        (max_occupation + 1) states per mode.  Default is 4.

    Notes
    -----
    For n_modes > 4 the operators are returned as ``scipy.sparse.csr_matrix``
    objects.  For n_modes <= 4 they are returned as dense ``numpy.ndarray``.
    In either case the caller can convert via ``op.toarray()`` or
    ``sp.csr_matrix(op)``.
    """

    def __init__(self, n_modes: int, max_occupation: int = 4) -> None:
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1.")
        if max_occupation < 1:
            raise ValueError("max_occupation must be >= 1.")

        self.n_modes = n_modes
        self.max_occupation = max_occupation
        self._local_dim = max_occupation + 1          # per-mode Hilbert dim
        self._use_sparse = n_modes > _DENSE_MODE_LIMIT

        if self._use_sparse:
            warnings.warn(
                f"n_modes={n_modes} > {_DENSE_MODE_LIMIT}: using sparse "
                "scipy matrices.  Total Hilbert-space dimension is "
                f"{self.dim()}.  Operations may still be slow for very "
                "large systems.",
                ResourceWarning,
                stacklevel=2,
            )
        elif self.dim() > 10_000:
            warnings.warn(
                f"Hilbert-space dimension is {self.dim()}, which is large. "
                "Dense matrix operations may be slow or memory-intensive.",
                ResourceWarning,
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _single_mode_creation(self) -> np.ndarray:
        """Single-mode creation operator a† in the local truncated basis."""
        d = self._local_dim
        mat = np.zeros((d, d), dtype=complex)
        for n in range(d - 1):
            mat[n + 1, n] = np.sqrt(n + 1.0)
        return mat

    def _single_mode_annihilation(self) -> np.ndarray:
        """Single-mode annihilation operator a in the local truncated basis."""
        return self._single_mode_creation().conj().T

    def _embed_single_mode_op(self, op_local: np.ndarray, mode: int):
        """Embed a single-mode operator into the full N-mode Hilbert space.

        Returns a dense ndarray (n_modes <= _DENSE_MODE_LIMIT) or a sparse
        csr_matrix (n_modes > _DENSE_MODE_LIMIT).
        """
        d = self._local_dim
        if self._use_sparse:
            eye = sp.eye(d, format="csr", dtype=complex)
            op_sp = sp.csr_matrix(op_local.astype(complex))
            result = sp.eye(1, format="csr", dtype=complex)
            for k in range(self.n_modes):
                result = sp.kron(result, op_sp if k == mode else eye, format="csr")
            return result
        else:
            eye = np.eye(d, dtype=complex)
            result = np.array([[1.0 + 0j]])
            for k in range(self.n_modes):
                result = np.kron(result, op_local if k == mode else eye)
            return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def dim(self) -> int:
        """Total Hilbert-space dimension: (max_occupation + 1) ** n_modes."""
        return self._local_dim ** self.n_modes

    def creation_op(self, mode: int) -> np.ndarray:
        """Full Fock-space creation operator a†_mode.

        a†_mode |n_mode> = sqrt(n_mode + 1) |n_mode + 1>
        (returns zero if n_mode + 1 > max_occupation).
        """
        self._check_mode(mode)
        return self._embed_single_mode_op(self._single_mode_creation(), mode)

    def annihilation_op(self, mode: int) -> np.ndarray:
        """Full Fock-space annihilation operator a_mode.

        a_mode |n_mode> = sqrt(n_mode) |n_mode - 1>
        (returns zero if n_mode = 0, i.e. vacuum is annihilated).
        """
        self._check_mode(mode)
        return self._embed_single_mode_op(self._single_mode_annihilation(), mode)

    def number_op(self, mode: int) -> np.ndarray:
        """Number operator n_mode = a†_mode a_mode.

        Eigenvalues are the occupation numbers {0, 1, ..., max_occupation}.
        """
        self._check_mode(mode)
        adag = self._single_mode_creation()
        a = self._single_mode_annihilation()
        n_local = adag @ a
        return self._embed_single_mode_op(n_local, mode)

    def position_op(self, mode: int, omega: float = 1.0) -> np.ndarray:
        """Dimensionless position (quadrature) operator Q_mode.

        In natural units (hbar = 1, m = 1):
            Q_mode = sqrt(1 / (2 * omega)) * (a_mode + a†_mode)
        """
        self._check_mode(mode)
        if omega <= 0:
            raise ValueError("omega must be positive.")
        adag = self.creation_op(mode)
        a = self.annihilation_op(mode)
        prefactor = np.sqrt(1.0 / (2.0 * omega))
        return prefactor * (a + adag)

    def momentum_op(self, mode: int, omega: float = 1.0) -> np.ndarray:
        """Dimensionless momentum (quadrature) operator P_mode.

        In natural units (hbar = 1, m = 1):
            P_mode = i * sqrt(omega / 2) * (a†_mode - a_mode)
        """
        self._check_mode(mode)
        if omega <= 0:
            raise ValueError("omega must be positive.")
        adag = self.creation_op(mode)
        a = self.annihilation_op(mode)
        prefactor = 1j * np.sqrt(omega / 2.0)
        return prefactor * (adag - a)

    def vacuum_state(self) -> np.ndarray:
        """Return the global vacuum state |0, 0, ..., 0> as a column vector."""
        psi = np.zeros(self.dim(), dtype=complex)
        psi[0] = 1.0
        return psi

    def fock_state(self, occupation: List[int]) -> np.ndarray:
        """Return the Fock state |n_0, n_1, ..., n_{N-1}> as a vector."""
        if len(occupation) != self.n_modes:
            raise ValueError(
                f"occupation must have length {self.n_modes}, "
                f"got {len(occupation)}."
            )
        for k, n in enumerate(occupation):
            if not (0 <= n <= self.max_occupation):
                raise ValueError(
                    f"occupation[{k}]={n} out of range "
                    f"[0, {self.max_occupation}]."
                )
        d = self._local_dim
        idx = 0
        for n in occupation:
            idx = idx * d + n
        psi = np.zeros(self.dim(), dtype=complex)
        psi[idx] = 1.0
        return psi

    def coherent_state(self, mode: int, alpha: complex) -> np.ndarray:
        """Approximate coherent state |alpha> for a single mode.

        |alpha> ≈ exp(-|alpha|^2 / 2)
                  * sum_{n=0}^{N_max} (alpha^n / sqrt(n!)) |n>

        embedded in the full N-mode space (vacuum in all other modes).
        """
        self._check_mode(mode)
        d = self._local_dim
        amp = np.exp(-0.5 * abs(alpha) ** 2)
        local_psi = np.zeros(d, dtype=complex)
        for n in range(d):
            local_psi[n] = amp * (alpha ** n) / np.sqrt(float(factorial(n)))
        norm = np.linalg.norm(local_psi)
        if norm > 0:
            local_psi /= norm

        vacuum_local = np.zeros(d, dtype=complex)
        vacuum_local[0] = 1.0

        psi = np.array([1.0 + 0j])
        for k in range(self.n_modes):
            psi = np.kron(psi, local_psi if k == mode else vacuum_local)
        return psi

    def basis_labels(self) -> List[Tuple[int, ...]]:
        """Return all occupation tuples in Fock-basis order (mixed-radix)."""
        ranges = [range(self._local_dim)] * self.n_modes
        return list(itertools_product(*ranges))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_mode(self, mode: int) -> None:
        if not (0 <= mode < self.n_modes):
            raise ValueError(
                f"mode={mode} out of range [0, {self.n_modes - 1}]."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"FockSpace(n_modes={self.n_modes}, "
            f"max_occupation={self.max_occupation}, "
            f"dim={self.dim()}, "
            f"sparse={self._use_sparse})"
        )
