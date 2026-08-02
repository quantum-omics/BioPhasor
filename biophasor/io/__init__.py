"""
biophasor.io — Omics data loaders.

Each loader returns AnnData with phasor representations attached.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from biophasor.io.rnaseq import load_rnaseq
from biophasor.io.singlecell import load_singlecell
from biophasor.io.proteomics import load_proteomics
from biophasor.io.metabolomics import load_metabolomics
from biophasor.io.formats import auto_load
from biophasor.io.loader import auto_detect

__all__ = [
    "load_rnaseq",
    "load_singlecell",
    "load_proteomics",
    "load_metabolomics",
    "auto_load",
    "auto_detect",
]

