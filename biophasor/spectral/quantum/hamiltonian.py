"""
hamiltonian.py — Free omics Hamiltonian in Fock space
======================================================

Physical picture
----------------
The free Hamiltonian of the omics quantum model is a collection of
independent quantum harmonic oscillators, one for each omics harmonic mode:

    H_0 = sum_k  hbar * eps_k * (n_k + 1/2)

where
    - eps_k are the mode self-energies, the leading omics harmonic
      frequencies omega_k = sqrt(|lambda_k|) from the Omics Connectome
      Matrix spectrum (see ``omics_spectrum.compartment_self_energies``).
    - n_k = a†_k a_k  is the number operator for mode k.
    - hbar = 1  throughout (natural units).

The zero-point energy is  E_0 = (1/2) * sum_k eps_k.

Thermal states
--------------
At inverse temperature beta = 1/(k_B T) the system occupies a Gibbs state:

    rho_th = exp(-beta * H_0) / Z,    Z = Tr[exp(-beta * H_0)]

which factorises over modes into independent Bose-Einstein distributions.

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
import scipy.linalg

from biophasor.spectral.quantum.fock_space import FockSpace


class OmicsHamiltonian:
    """Free Hamiltonian of the omics quantum model in truncated Fock space.

    H_0 = sum_k  eps_k * (n_k + 1/2)   (hbar = 1)

    Parameters
    ----------
    frequencies : np.ndarray, shape (n_modes,)
        Mode self-energies eps_k >= 0 for each omics harmonic mode.
        Typically the leading omics harmonic frequencies
        eps_k = omega_k = sqrt(|lambda_k|).
    fock_space : FockSpace
        Pre-built Fock-space object that matches the number of modes.

    Raises
    ------
    ValueError
        If len(frequencies) != fock_space.n_modes.
    """

    def __init__(
        self,
        frequencies: np.ndarray,
        fock_space: FockSpace,
    ) -> None:
        frequencies = np.asarray(frequencies, dtype=float)
        if frequencies.ndim != 1:
            raise ValueError("frequencies must be a 1-D array.")
        if len(frequencies) != fock_space.n_modes:
            raise ValueError(
                f"len(frequencies)={len(frequencies)} does not match "
                f"fock_space.n_modes={fock_space.n_modes}."
            )
        if np.any(frequencies < 0):
            warnings.warn(
                "Some frequencies are negative.  Setting them to zero.",
                RuntimeWarning,
                stacklevel=2,
            )
            frequencies = np.maximum(frequencies, 0.0)

        self.frequencies = frequencies
        self.fock_space = fock_space
        self._H0: np.ndarray | None = None   # cached

    # ------------------------------------------------------------------
    # Core Hamiltonian
    # ------------------------------------------------------------------

    def free_hamiltonian(self) -> np.ndarray:
        """Construct and return the free Hamiltonian H_0.

        H_0 = sum_k  eps_k * (n_k + 1/2)

        where n_k = a†_k a_k is the number operator and hbar = 1.
        The result is cached after the first call.
        """
        if self._H0 is not None:
            return self._H0

        fs = self.fock_space
        dim = fs.dim()

        H = np.zeros((dim, dim), dtype=complex)

        for k, eps_k in enumerate(self.frequencies):
            nk = fs.number_op(k)
            if hasattr(nk, "toarray"):
                nk = nk.toarray()
            H += eps_k * nk
            H += (eps_k / 2.0) * np.eye(dim, dtype=complex)

        self._H0 = H
        return H

    # ------------------------------------------------------------------
    # Classmethod constructor from an omics connectome matrix
    # ------------------------------------------------------------------

    @classmethod
    def from_connectome(
        cls,
        connectome: np.ndarray,
        n_modes: int = 5,
        max_occupation: int = 3,
    ) -> "OmicsHamiltonian":
        """Build an OmicsHamiltonian from an Omics Connectome Matrix (OCM).

        Pipeline:
            H (Hermitian OCM)
            → eigenvalues {lambda_k}
            → omega_k = sqrt(|lambda_k|)   (leading n_modes, descending)
            → FockSpace(n_modes, max_occupation)
            → OmicsHamiltonian(frequencies, fock_space)

        Parameters
        ----------
        connectome : np.ndarray, shape (M, M)
            Hermitian Omics Connectome Matrix.
        n_modes : int, optional
            Number of leading omics harmonic modes to retain.  Default 5.
        max_occupation : int, optional
            Fock-space truncation per mode.  Default 3.
        """
        H_ocm = np.asarray(connectome)
        if H_ocm.ndim != 2 or H_ocm.shape[0] != H_ocm.shape[1]:
            raise ValueError("connectome must be a square 2-D matrix.")

        H_ocm = 0.5 * (H_ocm + H_ocm.conj().T)
        eigenvalues = np.linalg.eigvalsh(H_ocm)
        omega = np.sqrt(np.abs(np.real(eigenvalues)))

        if n_modes > len(omega):
            warnings.warn(
                f"n_modes={n_modes} exceeds available modes "
                f"({len(omega)}).  Using {len(omega)} modes.",
                RuntimeWarning,
                stacklevel=2,
            )
            n_modes = len(omega)

        # leading (largest-frequency) modes, descending
        frequencies = np.sort(omega)[::-1][:n_modes]

        fock_space = FockSpace(n_modes=n_modes, max_occupation=max_occupation)
        return cls(frequencies=frequencies, fock_space=fock_space)

    # ------------------------------------------------------------------
    # Ground state
    # ------------------------------------------------------------------

    def ground_state(self) -> Tuple[float, np.ndarray]:
        """Compute the ground state of H_0 via exact diagonalisation."""
        H = self.free_hamiltonian()
        eigenvalues, eigenvectors = scipy.linalg.eigh(H)
        E_0 = float(eigenvalues[0].real)
        psi_0 = eigenvectors[:, 0]
        return E_0, psi_0

    def ground_state_energy(self) -> float:
        """Zero-point (vacuum) energy of the free Hamiltonian.

        E_0 = (1/2) * sum_k  eps_k
        """
        return float(0.5 * np.sum(self.frequencies))

    # ------------------------------------------------------------------
    # Density matrices
    # ------------------------------------------------------------------

    def density_matrix_from_state(self, psi: np.ndarray) -> np.ndarray:
        """Compute the pure-state density matrix rho = |psi><psi|."""
        psi = np.asarray(psi, dtype=complex).ravel()
        norm = np.linalg.norm(psi)
        if not np.isclose(norm, 1.0, atol=1e-6):
            warnings.warn(
                f"State vector norm {norm:.6f} != 1.  "
                "Normalising before constructing density matrix.",
                RuntimeWarning,
                stacklevel=2,
            )
            psi = psi / norm
        return np.outer(psi, psi.conj())

    def thermal_density_matrix(self, beta: float) -> np.ndarray:
        """Thermal (Gibbs) density matrix at inverse temperature beta.

        rho_th = exp(-beta * H_0) / Z,   Z = Tr[exp(-beta * H_0)]
        """
        if beta < 0:
            raise ValueError("beta must be >= 0.")

        H = self.free_hamiltonian()
        rho_unnorm = scipy.linalg.expm(-beta * H)
        rho_unnorm = 0.5 * (rho_unnorm + rho_unnorm.conj().T)
        Z = np.trace(rho_unnorm).real
        if Z <= 0:
            raise RuntimeError(
                "Partition function Z <= 0.  Numerical instability detected."
            )
        return rho_unnorm / Z

    # ------------------------------------------------------------------
    # Expectation values
    # ------------------------------------------------------------------

    def expectation_value(
        self,
        operator: np.ndarray,
        state: np.ndarray,
    ) -> float:
        """Compute the expectation value <psi|O|psi>."""
        psi = np.asarray(state, dtype=complex).ravel()
        norm2 = np.dot(psi.conj(), psi).real
        if norm2 == 0:
            raise ValueError("State vector has zero norm.")
        O_psi = operator @ psi
        ev = np.dot(psi.conj(), O_psi) / norm2
        if abs(ev.imag) > 1e-6 * (abs(ev.real) + 1e-12):
            warnings.warn(
                f"Expectation value has non-negligible imaginary part: "
                f"{ev.imag:.3e}.  Is the operator Hermitian?",
                RuntimeWarning,
                stacklevel=2,
            )
        return float(ev.real)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"OmicsHamiltonian(n_modes={self.fock_space.n_modes}, "
            f"max_occupation={self.fock_space.max_occupation}, "
            f"dim={self.fock_space.dim()}, "
            f"E0={self.ground_state_energy():.4f})"
        )
