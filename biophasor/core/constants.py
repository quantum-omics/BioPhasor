"""
biophasor.core.constants — Biological phase references.

Phase landmarks for cell cycle, circadian, and canonical marker genes.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

import numpy as np

# ── Cell-cycle phase reference angles (radians) ───────────────────────────────
# Mapped to the unit circle: G1 → 0, S → π/2, G2 → π, M → 3π/2
CELL_CYCLE_PHASES: dict[str, float] = {
    "G1":  0.0,
    "S":   np.pi / 2,
    "G2":  np.pi,
    "M":   3 * np.pi / 2,
}

# ── Circadian phase reference angles (hours → radians; 24-h period) ───────────
CIRCADIAN_PHASES: dict[str, float] = {
    "ZT0":   0.0,           # lights on (zeitgeber 0)
    "ZT6":   np.pi / 2,
    "ZT12":  np.pi,         # lights off
    "ZT18":  3 * np.pi / 2,
}

# ── Generic phase references ──────────────────────────────────────────────────
PHASE_REFS: dict[str, dict[str, float]] = {
    "cell_cycle": CELL_CYCLE_PHASES,
    "circadian":  CIRCADIAN_PHASES,
}

# ── Canonical cell-cycle marker genes ─────────────────────────────────────────
# Used by CellCyclePhasor for phase assignment.
# Reference: Tirosh et al. 2016 (Seurat v2 gene sets)
CANONICAL_MARKER_GENES: dict[str, list[str]] = {
    "G1": [
        "CDK4", "CDK6", "CCND1", "CCND2", "CCND3",
        "RB1", "E2F1", "E2F2", "E2F3",
    ],
    "S": [
        "PCNA", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6",
        "RRM2", "TYMS", "POLA1", "RFC2", "RFC4", "RFC5",
        "CCNE1", "CCNE2", "CDK2",
    ],
    "G2": [
        "CCNB1", "CCNB2", "CDK1", "AURKA", "AURKB",
        "PLK1", "BUB1", "BUB1B", "BUB3",
    ],
    "M": [
        "MKI67", "TOP2A", "CENPF", "SMC4", "KIF11",
        "KIF2C", "NUSAP1", "UBE2C", "TPX2",
    ],
}

# ── Circadian core-clock genes (mammals) ──────────────────────────────────────
CIRCADIAN_CLOCK_GENES: list[str] = [
    "CLOCK", "BMAL1", "PER1", "PER2", "PER3",
    "CRY1", "CRY2", "RORA", "REV-ERBA", "CSNK1E",
]
