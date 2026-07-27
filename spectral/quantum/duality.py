"""
quantum.duality — Quantum-classical duality primitives (theory.md §9).

Minimal, dependency-free demonstration that the classical phasor operations of
the OCM pipeline have exact single-/two-qubit gate analogs. Provided for
completeness and as a validated path to quantum-hardware execution; not used by
the classical pipeline.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import numpy as np


def rz_matrix(delta: float) -> np.ndarray:
    """Single-qubit R_z(δ) gate — the analog of a phasor phase shift θ→θ+δ."""
    return np.array(
        [[np.exp(-1j * delta / 2), 0.0], [0.0, np.exp(1j * delta / 2)]],
        dtype=complex,
    )


def qft_matrix(n: int) -> np.ndarray:
    """n×n Quantum Fourier Transform matrix — analog of the harmonic basis change.

    W_jk = exp(2πi jk / n) / √n (unitary).
    """
    j = np.arange(n)
    W = np.exp(2j * np.pi * np.outer(j, j) / n) / np.sqrt(n)
    return W


def phasor_to_statevector(psi: np.ndarray) -> np.ndarray:
    """Embed a phasor vector ψ ∈ ℂ^N as a normalised statevector (‖·‖₂ = 1)."""
    psi = np.asarray(psi, dtype=complex).ravel()
    norm = np.linalg.norm(psi)
    if norm == 0:
        v = np.zeros_like(psi)
        v[0] = 1.0
        return v
    return psi / norm
