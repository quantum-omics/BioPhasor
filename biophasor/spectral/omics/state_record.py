"""
omics.state_record — Spectral state record (theory.md §8).

A JSON-serialisable snapshot of the full spectral state at one sample / time
point. Omics analog of the Q-NEHR (BCI) and Q-MEHR (Finance) records.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np


def _complex_matrix_to_list(M: np.ndarray) -> List[List[List[float]]]:
    """Serialise a complex matrix as nested [re, im] pairs."""
    M = np.asarray(M, dtype=complex)
    return [[[float(v.real), float(v.imag)] for v in row] for row in M]


@dataclass
class SpectralStateRecord:
    """Serialisable spectral-state record for one sample/time point (theory.md §8)."""

    id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    coherence_R: float = 0.0
    spectral_entropy: float = 0.0
    fiedler_gap: float = 0.0
    participation_ratio: float = 0.0
    coherence_kappa: float = 0.0
    eigenvalues: List[float] = field(default_factory=list)
    ccm_matrix: List[List[List[float]]] = field(default_factory=list)
    compartment_weights: Dict[str, float] = field(default_factory=dict)
    state_class: str = ""
    state_label: str = ""
    recommended_intervention: str = ""
    consistency: Dict[str, List[float]] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def from_pipeline(
        cls,
        record_id: str,
        indicators: dict,
        eigenvalues: np.ndarray,
        ccm: np.ndarray,
        compartment_weights: dict,
        state_class: dict,
        consistency: Optional[Dict[str, tuple]] = None,
        timestamp: Optional[str] = None,
    ) -> "SpectralStateRecord":
        """Assemble a state record from the pipeline outputs (theory.md §8)."""
        cons_ser: Dict[str, List[float]] = {}
        if consistency is not None:
            for k, (passed, residual) in consistency.items():
                cons_ser[k] = [float(bool(passed)), float(residual)]

        kwargs = dict(
            id=record_id,
            coherence_R=float(indicators.get("coherence_R", 0.0)),
            spectral_entropy=float(indicators.get("spectral_entropy", 0.0)),
            fiedler_gap=float(indicators.get("fiedler_gap", 0.0)),
            participation_ratio=float(indicators.get("participation_ratio", 0.0)),
            coherence_kappa=float(compartment_weights.get("coherence_kappa", 0.0)),
            eigenvalues=[float(v) for v in np.asarray(eigenvalues, dtype=float)],
            ccm_matrix=_complex_matrix_to_list(ccm),
            compartment_weights={k: float(v) for k, v in compartment_weights.get("weights", {}).items()},
            state_class=str(state_class.get("class", "")),
            state_label=str(state_class.get("label", "")),
            recommended_intervention=str(state_class.get("recommended_intervention", "")),
            consistency=cons_ser,
        )
        if timestamp is not None:
            kwargs["timestamp"] = timestamp
        return cls(**kwargs)

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as fh:
            fh.write(self.to_json(indent=indent))

    @classmethod
    def load(cls, path: str) -> "SpectralStateRecord":
        with open(path) as fh:
            return cls(**json.load(fh))
