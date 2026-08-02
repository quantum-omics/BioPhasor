"""connectome — phasor encoding, Omics Connectome Matrix, omics harmonics.

``tanh_phase_encode`` and ``phase_coherence`` are re-exported from the trimmed
``phasor`` module, which now delegates them to ``biophasor.core`` (the single
canonical source).
"""

from biophasor.spectral.connectome.phasor import (
    PhasorEncoder,
    tanh_phase_encode,
    phase_coherence,
)
from biophasor.spectral.connectome.ocm import OmicsConnectomeMatrix
from biophasor.spectral.connectome.harmonics import OmicsHarmonics
from biophasor.spectral.connectome.magnetic import (
    build_magnetic,
    lead_lag_antisymmetry,
    signed_antisymmetry,
    cycle_flux,
)

__all__ = [
    "PhasorEncoder",
    "tanh_phase_encode",
    "phase_coherence",
    "OmicsConnectomeMatrix",
    "OmicsHarmonics",
    "build_magnetic",
    "lead_lag_antisymmetry",
    "signed_antisymmetry",
    "cycle_flux",
]
