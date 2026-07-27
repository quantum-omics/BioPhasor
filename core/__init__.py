"""
biophasor.core — Phasor math primitives.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from biophasor.core.phasor import BioPhasor
from biophasor.core.manifold import PhasorManifold
from biophasor.core.operators import (
    coherence, phasor_mean, phase_couple, bio_shift, bio_mix,
    phase_coherence, phasor_statistics,
)
from biophasor.core.encoder import (
    tanh_phase_encode,
    log_linear_encode,
    linear_encode,
    OmicsPhasorEncoder,
)
from biophasor.core.constants import (
    CELL_CYCLE_PHASES,
    CIRCADIAN_PHASES,
    PHASE_REFS,
    CANONICAL_MARKER_GENES,
)

__all__ = [
    "BioPhasor",
    "PhasorManifold",
    "coherence",
    "phasor_mean",
    "phase_couple",
    "bio_shift",
    "bio_mix",
    "phase_coherence",
    "phasor_statistics",
    "tanh_phase_encode",
    "log_linear_encode",
    "linear_encode",
    "OmicsPhasorEncoder",
    "CELL_CYCLE_PHASES",
    "CIRCADIAN_PHASES",
    "PHASE_REFS",
    "CANONICAL_MARKER_GENES",
]
