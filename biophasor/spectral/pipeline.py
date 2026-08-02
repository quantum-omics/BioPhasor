"""
pipeline — End-to-end Spectral-Omics pipeline convenience wrapper.

Chains phasor encoding → OCM → harmonics → indicators → CCM → CompartmentWeights →
state classes → state record, with the consistency suite. Operates per-sample-slice on a
shared feature-feature coupling estimated once across all samples.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from biophasor.spectral.connectome.phasor import PhasorEncoder
from biophasor.spectral.connectome.ocm import OmicsConnectomeMatrix
from biophasor.spectral.connectome.harmonics import OmicsHarmonics
from biophasor.spectral.omics.indicators import SpectralIndicators
from biophasor.spectral.omics.ccm import CompartmentCouplingMatrix, COMPARTMENTS
from biophasor.spectral.omics.compartment_weights import CompartmentWeights
from biophasor.spectral.omics.state_classes import SpectralStateClassifier
from biophasor.spectral.omics.state_record import SpectralStateRecord
from biophasor.spectral.omics.consistency import ConsistencySuite


class SpectralOmicsPipeline:
    """Full spectral-omics pipeline over an omics matrix.

    Parameters
    ----------
    X : np.ndarray, shape (S, N)
        Omics matrix, S samples × N features (non-negative for RNA-seq).
    membership : dict {compartment: [feature indices]}, optional
        Marker assignment of features to the five compartments.
    coupling_mode : str
        OCM coupling mode ('pearson', 'coexpression', 'prior', 'uniform').
    n_harmonics : int or None
        Number of leading harmonics retained.
    log_transform : bool
        Passed to the phasor encoder (False for ATAC/methylation).
    amplitude_mode : str
        Phasor amplitude mode ('expression', 'rhythm', 'unit').
    """

    def __init__(
        self,
        X: np.ndarray,
        membership: Optional[Dict[str, Sequence[int]]] = None,
        coupling_mode: str = "pearson",
        n_harmonics: Optional[int] = None,
        log_transform: bool = True,
        amplitude_mode: str = "expression",
    ) -> None:
        self.X = np.asarray(X, dtype=float)
        self.S, self.N = self.X.shape
        self.membership = membership
        self.encoder = PhasorEncoder(log_transform=log_transform, amplitude_mode=amplitude_mode)
        self.ocm = OmicsConnectomeMatrix(coupling_mode=coupling_mode)
        self.harmonics = OmicsHarmonics(n_harmonics=n_harmonics)
        self.indicators = SpectralIndicators()
        self.ccm = CompartmentCouplingMatrix()
        self.compartment_weights = CompartmentWeights()
        self.state_class = SpectralStateClassifier()
        self.suite = ConsistencySuite()

        # shared objects computed once
        self.Psi = self.encoder.encode(self.X)               # (S, N)
        self.coupling = self.ocm.compute_coupling(self.X)    # (N, N)

    # ------------------------------------------------------------------
    def run_slice(self, s: int, record_id: Optional[str] = None,
                  check_consistency: bool = False) -> dict:
        """Run the full pipeline on sample slice s; return a results dict."""
        psi = self.Psi[s]
        H = self.ocm.build(psi, coupling=self.coupling)
        vals, vecs = self.harmonics.decompose(H)
        panel = self.indicators.compute(vals, vecs, psi, coupling=self.coupling)
        M = self.ccm.build(H, membership=self.membership, eigenvectors=vecs)
        readout = self.compartment_weights.analyze(M)
        cls = self.state_class.classify(
            panel["coherence_R"], panel["spectral_entropy"], panel["fiedler_gap"],
            readout["coherence_kappa"], float(np.max(readout["weight_vector"])),
        )
        consistency = None
        if check_consistency:
            consistency = self.suite.run(H, vals, vecs, M, psi, self.coupling)
        state_record = SpectralStateRecord.from_pipeline(
            record_id or f"slice_{s}", panel, vals, M, readout, cls, consistency,
        )
        return dict(
            slice=s, eigenvalues=vals, eigenvectors=vecs, H=H, M=M,
            indicators=panel, compartment_weights=readout, state_class=cls,
            consistency=consistency, state_record=state_record,
        )

    # ------------------------------------------------------------------
    def run_all(self, check_consistency_on: str = "first") -> List[dict]:
        """Run every sample slice.

        check_consistency_on : {'first', 'all', 'none'}
            Where to run the (more expensive) consistency suite.
        """
        out = []
        for s in range(self.S):
            do_check = (check_consistency_on == "all") or (check_consistency_on == "first" and s == 0)
            out.append(self.run_slice(s, check_consistency=do_check))
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def stack_series(results: List[dict]) -> dict:
        """Collect per-slice results into arrays for plotting."""
        eig = np.array([r["eigenvalues"] for r in results])       # (S, k)
        ind_keys = list(results[0]["indicators"].keys())
        ind = {k: np.array([r["indicators"][k] for r in results]) for k in ind_keys}
        classes = [r["state_class"]["class"] for r in results]
        kappa = np.array([r["compartment_weights"]["coherence_kappa"] for r in results])
        return dict(eigenvalues=eig, indicators=ind, classes=classes, kappa=kappa)
