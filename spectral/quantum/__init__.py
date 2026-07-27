"""
quantum — quantum-classical duality helpers (theory.md §9).

Each classical phasor operation maps one-to-one onto a quantum gate:
    phase shift θ→θ+δ  ↔  R_z(δ)
    pairwise coupling   ↔  CNOT
    harmonic (DFT) basis change ↔ QFT

This module is optional and not required by the classical pipeline.
"""

from biophasor.spectral.quantum.duality import (
    rz_matrix,
    qft_matrix,
    phasor_to_statevector,
)

__all__ = ["rz_matrix", "qft_matrix", "phasor_to_statevector"]
