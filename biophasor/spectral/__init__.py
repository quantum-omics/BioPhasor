"""
biophasor.spectral — phasor graph-spectral analysis of multi-omics data.

Migrated from the standalone ``spectralomics`` package into the unified
BioPhasor platform (Phase 3). The former module-level ``tanh_phase_encode`` /
``phase_coherence`` / ``phasor_statistics`` are now imported from
``biophasor.core`` (single canonical source) rather than re-implemented here.

Pipeline
--------
    omics matrix X
        -> PhasorEncoder            (ψ_i = r_i e^{iθ_i};  tanh-phase)
        -> OmicsConnectomeMatrix    (H_ij = c_ij e^{i(θ_i-θ_j)};  Hermitian)
        -> OmicsHarmonics           (H φ_n = λ_n φ_n;  real spectrum)
        -> SpectralIndicators       (entropy, Fiedler gap, PR, Kuramoto R)
        -> CompartmentCouplingMatrix        (5x5 Hermitian over compartments)
        -> CompartmentWeights                (compartment weights, dominance, coherence κ)
        -> SpectralStateClassifier            (7-class cellular/disease state)
        -> SpectralStateRecord                    (serialisable spectral record)
    + ConsistencySuite              (Hermiticity, reality, PSD, gauge invariance)

The "quantum" vocabulary denotes a quantum-simulable signal-processing formalism,
NOT a claim of physical quantum computation in cells.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

__version__ = "0.1.0"

# submodules exposed as attributes (tests use e.g. so.connectome.tanh_phase_encode)
from biophasor.spectral import connectome, omics, quantum

from biophasor.spectral.connectome.phasor import PhasorEncoder
from biophasor.spectral.connectome.ocm import OmicsConnectomeMatrix
from biophasor.spectral.connectome.harmonics import OmicsHarmonics
from biophasor.spectral.omics.indicators import SpectralIndicators
from biophasor.spectral.omics.ccm import CompartmentCouplingMatrix, COMPARTMENTS
from biophasor.spectral.omics.compartment_weights import CompartmentWeights
from biophasor.spectral.omics.state_classes import SpectralStateClassifier
from biophasor.spectral.omics.state_record import SpectralStateRecord
from biophasor.spectral.omics.consistency import ConsistencySuite
from biophasor.spectral.pipeline import SpectralOmicsPipeline

__all__ = [
    "__version__",
    "connectome",
    "omics",
    "quantum",
    "SpectralOmicsPipeline",
    "PhasorEncoder",
    "OmicsConnectomeMatrix",
    "OmicsHarmonics",
    "SpectralIndicators",
    "CompartmentCouplingMatrix",
    "COMPARTMENTS",
    "CompartmentWeights",
    "SpectralStateClassifier",
    "SpectralStateRecord",
    "ConsistencySuite",
]
