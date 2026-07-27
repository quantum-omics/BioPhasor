"""
BioPhasor: Phasor Dynamics Library for Omics Data.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("biophasor")
except PackageNotFoundError:
    __version__ = "0.3.0"

# ── Core conveniences ─────────────────────────────────────────────────────────
from biophasor.core.phasor import BioPhasor
from biophasor.core.manifold import PhasorManifold
from biophasor.core.operators import coherence, phasor_mean, phase_couple

# ── Top-level encoding shortcut ───────────────────────────────────────────────
from biophasor.transform.encoder import tanh_phase_encode, OmicsPhasorEncoder

# ── Multi-omics integration shortcut ─────────────────────────────────────────
from biophasor.integration.multiomics import integrate

# ── Cell State Tensor (CST) — dissipative dynamics & attractor landscape ─────
from biophasor.cst.tensor import CellStateTensor
from biophasor.cst.dynamics import CSTDynamics
from biophasor.cst.attractor import AttractorLandscape
from biophasor.cst.limit_cycles import LimitCycleAnalyzer
from biophasor.cst.geometry import AttractorGeometry

__all__ = [
    "__version__",
    # core
    "BioPhasor",
    "PhasorManifold",
    "coherence",
    "phasor_mean",
    "phase_couple",
    # transform
    "tanh_phase_encode",
    "OmicsPhasorEncoder",
    # integration
    "integrate",
    # CST
    "CellStateTensor",
    "CSTDynamics",
    "AttractorLandscape",
    "LimitCycleAnalyzer",
    "AttractorGeometry",
]

