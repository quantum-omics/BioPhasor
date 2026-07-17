"""
biophasor.cst — Cell State Tensor (CST) sub-package.

Core components for phase-coupled dissipative system analysis in cellular biology:
  - CellStateTensor: 3D phasor tensor (R, T, H) — regulatory × temporal × homeostatic
  - CSTDynamics: closed-loop CST evolution via Kuramoto coupling on gene networks
  - AttractorLandscape: cell-state basin-of-attraction analysis
  - LimitCycleAnalyzer: limit cycle detection and characterization in regulatory circuits
  - AttractorGeometry: attractor basin geometry and inter-state transitions

The CST is the biological analogue of the Mental State Tensor (MST) in the
Neurophasor framework, adapted for multi-omics cellular data.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from biophasor.cst.tensor import CellStateTensor
from biophasor.cst.dynamics import CSTDynamics
from biophasor.cst.attractor import AttractorLandscape
from biophasor.cst.limit_cycles import LimitCycleAnalyzer
from biophasor.cst.geometry import AttractorGeometry

__all__ = [
    "CellStateTensor",
    "CSTDynamics",
    "AttractorLandscape",
    "LimitCycleAnalyzer",
    "AttractorGeometry",
]
