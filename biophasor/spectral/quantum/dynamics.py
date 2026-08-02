"""
dynamics.py — Quantum time evolution and entropy for the omics model
=====================================================================

Physical picture
----------------
Given a Hamiltonian H (free + interaction terms), the unitary
time-evolution operator is

    U(t) = exp(-i H t / hbar)

State evolution:
    |psi(t)> = U(t) |psi(0)>

Density-matrix evolution (closed system):
    rho(t) = U(t) rho(0) U†(t)

Open systems (Lindblad master equation):
    d rho / dt = -i [H, rho]
                 + gamma * sum_k (L_k rho L†_k  - 1/2 {L†_k L_k, rho})

Trotter decomposition:
    exp(-i (H_1 + H_2 + ...) dt) ≈ prod_k exp(-i H_k dt)   + O(dt^2)

Entanglement entropy:
    For a bipartition H = H_A ⊗ H_B, the reduced density matrix is
    rho_A = Tr_B[|psi><psi|] and the entanglement entropy is the von
    Neumann entropy  S_A = -Tr_A[rho_A log rho_A].

Units:  hbar user-configurable (default 1).

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import scipy.linalg


class QuantumDynamics:
    """Unitary and open-system dynamics of an omics state in Fock space.

    Parameters
    ----------
    hamiltonian : np.ndarray, shape (dim, dim)
        Full system Hamiltonian (may be free + interaction terms).
        Must be Hermitian.
    hbar : float, optional
        Reduced Planck constant.  Default 1.0 (natural units).
    """

    def __init__(
        self,
        hamiltonian: np.ndarray,
        hbar: float = 1.0,
    ) -> None:
        H = np.asarray(hamiltonian, dtype=complex)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("hamiltonian must be a square 2-D matrix.")
        if not np.allclose(H, H.conj().T, atol=1e-8):
            warnings.warn(
                "Hamiltonian is not exactly Hermitian (max deviation "
                f"{np.max(np.abs(H - H.conj().T)):.2e}).  "
                "Symmetrising before use.",
                RuntimeWarning,
                stacklevel=2,
            )
            H = 0.5 * (H + H.conj().T)

        self.hamiltonian = H
        self.dim = H.shape[0]
        self.hbar = float(hbar)

    # ------------------------------------------------------------------
    # Unitary evolution
    # ------------------------------------------------------------------

    def time_evolve_exact(self, psi0: np.ndarray, t: float) -> np.ndarray:
        """Exact state evolution via matrix exponential.

        |psi(t)> = exp(-i H t / hbar) |psi(0)>
        """
        psi0 = np.asarray(psi0, dtype=complex).ravel()
        self._check_dim(psi0.shape[0], "psi0")
        U = scipy.linalg.expm(-1j * self.hamiltonian * t / self.hbar)
        return U @ psi0

    def time_evolve_trotter(
        self,
        psi0: np.ndarray,
        t: float,
        n_steps: int,
        H_terms: List[np.ndarray],
    ) -> np.ndarray:
        """First-order Trotter-product approximation to time evolution.

            U(dt) ≈ prod_k exp(-i H_k dt / hbar)

        iterated n_steps times with dt = t / n_steps.
        """
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        psi = np.asarray(psi0, dtype=complex).ravel()
        self._check_dim(psi.shape[0], "psi0")

        dt = t / n_steps
        U_terms = [
            scipy.linalg.expm(-1j * np.asarray(Hk, dtype=complex) * dt / self.hbar)
            for Hk in H_terms
        ]

        for _ in range(n_steps):
            for Uk in U_terms:
                psi = Uk @ psi
        return psi

    def evolve_trajectory(
        self,
        psi0: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute state vectors at each time in a trajectory.

        Diagonalises H once and evaluates
        U(t) = V diag(exp(-i E t / hbar)) V† for each requested time.
        """
        psi0 = np.asarray(psi0, dtype=complex).ravel()
        times = np.asarray(times, dtype=float)
        self._check_dim(psi0.shape[0], "psi0")

        E, V = scipy.linalg.eigh(self.hamiltonian)
        c0 = V.conj().T @ psi0

        T = len(times)
        trajectory = np.zeros((T, self.dim), dtype=complex)

        for idx, t in enumerate(times):
            phases = np.exp(-1j * E * t / self.hbar)
            c_t = phases * c0
            trajectory[idx] = V @ c_t

        return trajectory

    # ------------------------------------------------------------------
    # Density matrix dynamics
    # ------------------------------------------------------------------

    def density_matrix_evolution(
        self,
        rho0: np.ndarray,
        t: float,
    ) -> np.ndarray:
        """Unitary density matrix evolution: rho(t) = U rho0 U†."""
        rho0 = np.asarray(rho0, dtype=complex)
        self._check_dim(rho0.shape[0], "rho0")
        U = scipy.linalg.expm(-1j * self.hamiltonian * t / self.hbar)
        rho_t = U @ rho0 @ U.conj().T
        rho_t = 0.5 * (rho_t + rho_t.conj().T)
        return rho_t

    # ------------------------------------------------------------------
    # Entropy measures
    # ------------------------------------------------------------------

    def von_neumann_entropy(self, rho: np.ndarray) -> float:
        """Von Neumann entropy S = -Tr[rho log(rho)] (nats)."""
        rho = np.asarray(rho, dtype=complex)
        eigenvalues = scipy.linalg.eigh(rho, eigvals_only=True)
        eigenvalues = np.maximum(eigenvalues.real, 0.0)
        mask = eigenvalues > 0
        S = -np.sum(eigenvalues[mask] * np.log(eigenvalues[mask]))
        return float(S)

    def entanglement_entropy(
        self,
        psi: np.ndarray,
        subsystem_dim: int,
    ) -> float:
        """Entanglement entropy of a bipartitioned pure state (nats).

        Reshapes psi into a (subsystem_dim, complement_dim) matrix, takes
        the Schmidt decomposition, and returns the von Neumann entropy of
        the reduced density matrix:
            S_A = -sum_k lambda_k^2 log(lambda_k^2)
        """
        psi = np.asarray(psi, dtype=complex).ravel()
        if self.dim % subsystem_dim != 0:
            raise ValueError(
                f"subsystem_dim={subsystem_dim} does not evenly divide "
                f"dim={self.dim}."
            )
        complement_dim = self.dim // subsystem_dim
        psi_mat = psi.reshape(subsystem_dim, complement_dim)
        singular_values = np.linalg.svd(psi_mat, compute_uv=False)
        probs = np.maximum(singular_values ** 2, 0.0)
        mask = probs > 0
        S = -np.sum(probs[mask] * np.log(probs[mask]))
        return float(S)

    # ------------------------------------------------------------------
    # Observable trajectories
    # ------------------------------------------------------------------

    def expectation_trajectory(
        self,
        observable: np.ndarray,
        states: np.ndarray,
    ) -> np.ndarray:
        """Compute <O>(t) for a trajectory of states."""
        observable = np.asarray(observable, dtype=complex)
        states = np.asarray(states, dtype=complex)
        if states.ndim != 2:
            raise ValueError("states must be 2-D array of shape (T, dim).")
        T, d = states.shape
        self._check_dim(d, "states")

        O_psi = (observable @ states.T).T
        ev = np.einsum("ti,ti->t", states.conj(), O_psi).real
        norms2 = np.einsum("ti,ti->t", states.conj(), states).real
        norms2 = np.where(norms2 > 0, norms2, 1.0)
        return ev / norms2

    # ------------------------------------------------------------------
    # Open quantum system (Lindblad)
    # ------------------------------------------------------------------

    def lindblad_evolve(
        self,
        rho0: np.ndarray,
        t: float,
        jump_ops: List[np.ndarray],
        gamma: float = 0.01,
    ) -> np.ndarray:
        """Lindblad master equation via simple Euler integration.

        d rho / dt = -i/hbar [H, rho]
                     + gamma * sum_k (L_k rho L†_k
                                      - (1/2) {L†_k L_k, rho})
        """
        rho0 = np.asarray(rho0, dtype=complex)
        self._check_dim(rho0.shape[0], "rho0")
        if t < 0:
            raise ValueError("t must be >= 0.")

        max_dt = 0.1 / max(gamma, 1e-10)
        n_steps = max(1, int(np.ceil(t / max_dt)))
        dt = t / n_steps

        Lops = [np.asarray(L, dtype=complex) for L in jump_ops]
        LdagL = [L.conj().T @ L for L in Lops]

        H = self.hamiltonian
        rho = rho0.copy()

        for _ in range(n_steps):
            drho = (-1j / self.hbar) * (H @ rho - rho @ H)
            for L, LdL in zip(Lops, LdagL):
                drho += gamma * (
                    L @ rho @ L.conj().T
                    - 0.5 * (LdL @ rho + rho @ LdL)
                )
            rho = rho + dt * drho
            rho = 0.5 * (rho + rho.conj().T)

        return rho

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_dim(self, d: int, name: str) -> None:
        if d != self.dim:
            raise ValueError(
                f"Dimension of '{name}' ({d}) does not match "
                f"Hamiltonian dimension ({self.dim})."
            )

    def __repr__(self) -> str:  # pragma: no cover
        return f"QuantumDynamics(dim={self.dim}, hbar={self.hbar})"
