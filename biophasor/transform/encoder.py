"""
biophasor.transform.encoder — backward-compatible re-export shim.

The canonical implementations now live in :mod:`biophasor.core.encoder` (single
source of truth for the whole platform). This module re-exports them so that
existing imports `from biophasor.transform.encoder import ...` keep working.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from biophasor.core.encoder import (  # noqa: F401
    tanh_phase_encode,
    log_linear_encode,
    linear_encode,
    OmicsPhasorEncoder,
)

__all__ = [
    "tanh_phase_encode",
    "log_linear_encode",
    "linear_encode",
    "OmicsPhasorEncoder",
]
