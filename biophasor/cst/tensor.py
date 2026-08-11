"""
biophasor.cst.tensor — CellStateTensor: core Biophasor state representation.

The Cell State Tensor (CST) is a 3D complex phasor tensor:

    CST[r, t, h] = A[r,t,h] · exp(i·θ[r,t,h])

    r ∈ {0..R-1}   regulatory modules   (TF activity, chromatin state, signalling)
    t ∈ {0..T-1}   temporal structure    (cell cycle, circadian, ultradian)
    h ∈ {0..H-1}   homeostatic factors   (coherence, redox, membrane potential)

The CST is the canonical state object of the Biophasor framework,
encoding the full attractor geometry of phase-coupled dissipative
regulatory networks — the biological analogue of the MST in Neurophasor.

Reference: Biophasor Manuscript — Section "Cell State Tensor (CST)"

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# Default axis labels for the three CST dimensions
REGULATORY_LABELS = [
    "TF_activity", "chromatin_state", "signalling_pathway",
    "epigenetic_mark", "metabolic_flux", "splicing_program",
]

TEMPORAL_LABELS = [
    "cell_cycle", "circadian", "ultradian",
    "developmental", "stress_response", "senescence",
]

HOMEOSTATIC_LABELS = [
    "global_coherence", "redox_balance", "membrane_potential",
    "proteostasis", "autophagy", "apoptotic_priming",
]


@dataclass
class CellStateTensor:
    """
    Cell State Tensor — complex phasor tensor (R, T, H).

    CST[r, t, h] = A[r,t,h] · exp(i·θ[r,t,h])

    Parameters
    ----------
    tensor : np.ndarray, complex, shape (n_regulatory, n_temporal, n_homeostatic)
        Complex-valued phasor tensor encoding the full cellular state.
    regulatory_names : list[str]
        Labels for the regulatory axis (TF programs, chromatin states).
    temporal_names : list[str]
        Labels for the temporal axis (cell cycle, circadian).
    homeostatic_names : list[str]
        Labels for the homeostatic axis (coherence, redox).
    metadata : dict
        Cell type, condition, patient, timepoint info.

    Properties
    ----------
    phase     : np.ndarray (R, T, H)   instantaneous regulatory phase
    amplitude : np.ndarray (R, T, H)   instantaneous amplitude
    n_regulatory, n_temporal, n_homeostatic : int

    Examples
    --------
    >>> cst = CellStateTensor.random(n_regulatory=6, n_temporal=4, n_homeostatic=4)
    >>> C = cst.coherence_map()         # (R, T) coherence
    >>> plv = cst.plv_map()             # (R, R) PLV per temporal axis
    >>> features = cst.attractor_features()  # dict of scalar features
    """

    tensor: np.ndarray                              # complex (R, T, H)
    regulatory_names: list = field(default_factory=lambda: list(REGULATORY_LABELS))
    temporal_names: list = field(default_factory=lambda: list(TEMPORAL_LABELS))
    homeostatic_names: list = field(default_factory=lambda: list(HOMEOSTATIC_LABELS))
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tensor = np.asarray(self.tensor, dtype=np.complex128)

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def phase(self) -> np.ndarray:
        """Instantaneous phase θ = arg(CST), shape (R, T, H)."""
        return np.angle(self.tensor)

    @property
    def amplitude(self) -> np.ndarray:
        """Instantaneous amplitude A = |CST|, shape (R, T, H)."""
        return np.abs(self.tensor)

    @property
    def n_regulatory(self) -> int:
        """Number of regulatory modules."""
        return self.tensor.shape[0]

    @property
    def n_temporal(self) -> int:
        """Number of temporal structure dimensions."""
        return self.tensor.shape[1]

    @property
    def n_homeostatic(self) -> int:
        """Number of homeostatic factors."""
        return self.tensor.shape[2]

    @property
    def shape(self) -> tuple:
        """Tensor shape (R, T, H)."""
        return self.tensor.shape

    # ── Analysis ──────────────────────────────────────────────────────────────

    def coherence_map(self) -> np.ndarray:
        """
        Per-regulatory, per-temporal coherence (mean resultant over homeostatic axis).

        Returns
        -------
        np.ndarray, shape (n_regulatory, n_temporal)  C ∈ [0, 1]
        """
        return np.abs(self.tensor.mean(axis=-1))

    def plv_map(self) -> np.ndarray:
        """
        Per-temporal-axis Phase Locking Value matrix across regulatory modules.

        Returns
        -------
        np.ndarray, shape (n_temporal, n_regulatory, n_regulatory)
        """
        R, T, H = self.tensor.shape
        PLV = np.zeros((T, R, R), dtype=np.float64)
        for t in range(T):
            z = self.tensor[:, t, :]     # (R, H)
            PLV[t] = np.abs(z @ z.conj().T) / H
        return PLV

    def regulatory_slice(self, name: str) -> np.ndarray:
        """Extract the complex tensor for a single regulatory module."""
        if name not in self.regulatory_names:
            raise ValueError(f"Module '{name}' not found. Available: {self.regulatory_names}")
        idx = self.regulatory_names.index(name)
        return self.tensor[idx]

    def global_coherence(self) -> float:
        """
        Global Coherence Metric (GCM): scalar summary of whole-cell phase synchrony.

        GCM = |Σ_i z_i| / Σ_i |z_i|,  the amplitude-weighted circular mean
        resultant length over every element of the tensor.

        The denominator is the sum of moduli, not the element count. Dividing by
        N instead — which is what ``|mean(tensor)|`` does — leaves the statistic
        scaled by the mean amplitude and therefore *unbounded above*: a tensor
        with identical phases and amplitudes around 1.6 returned 1.63, which no
        coherence may do. With the sum of moduli the triangle inequality gives
        |Σ z_i| <= Σ|z_i| exactly, so the value is in [0, 1] for any input, and
        equals 1 iff every element shares one phase.

        An all-zero tensor has no defined phase; 0.0 is returned for it rather
        than raising, so callers sweeping a knockout to extinction do not have
        to special-case the endpoint.

        Returns
        -------
        float ∈ [0, 1]
        """
        total_amplitude = float(np.abs(self.tensor).sum())
        if total_amplitude == 0.0:
            return 0.0
        return float(np.abs(self.tensor.sum()) / total_amplitude)

    def phase_entropy(self, n_bins: int = 36) -> float:
        """
        Phase entropy — complexity of the attractor landscape.

        Higher entropy = more diverse attractor basins occupied.
        """
        phases = self.phase.flatten() % (2 * np.pi)
        counts = np.histogram(phases, bins=n_bins)[0]
        probs = counts / (counts.sum() + 1e-12)
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def synchrony_index(self) -> float:
        """
        Pairwise phase synchrony S = (1/N²) Σ_{i,j} cos(φ_i - φ_j).

        Measures limit cycle coherence across all regulatory modules. The double
        sum collapses to the squared modulus of the resultant, since
        Σ_{i,j} cos(φ_i - φ_j) = Re[(Σ_i e^{iφ_i})(Σ_j e^{-iφ_j})] = |Σ_i e^{iφ_i}|²,
        which is what is computed here.

        Note the contraction that must NOT be used: ``z @ z.conj()`` is
        Σ_i z_i z_i* = Σ_i |z_i|² = N for unit phasors, so ``Re(z @ z.conj())/N²``
        is identically 1/N for *every* phase configuration and measures nothing.
        The outer product |Σ z|², not the inner product Σ|z|², is the pairwise
        sum this docstring describes.

        Returns
        -------
        float ∈ [0, 1], equal to 1 for perfectly aligned phases and to O(1/N)
        for phases drawn uniformly at random.
        """
        phases = self.phase.flatten()
        N = len(phases)
        if N == 0:
            return 0.0
        z = np.exp(1j * phases)
        return float(np.abs(z.sum()) ** 2 / N**2)

    def state_velocity(self, cst_prev: "CellStateTensor") -> float:
        """
        Inter-attractor transition velocity: ||Δφ||₁ / N.

        Parameters
        ----------
        cst_prev : CellStateTensor   previous timepoint CST.
        """
        dphi = np.angle(np.exp(1j * (self.phase - cst_prev.phase)))
        return float(np.abs(dphi).mean())

    def attractor_features(self, cst_prev: Optional["CellStateTensor"] = None) -> dict:
        """
        Extract attractor-geometric features for downstream analysis.

        Returns a dict of scalar features characterizing the current
        attractor landscape — the core CST feature vector.

        Returns
        -------
        dict with keys: global_coherence, phase_entropy, synchrony_index,
                       state_velocity (if cst_prev provided)
        """
        features = {
            "global_coherence": self.global_coherence(),
            "phase_entropy": self.phase_entropy(),
            "synchrony_index": self.synchrony_index(),
        }
        if cst_prev is not None:
            features["state_velocity"] = self.state_velocity(cst_prev)
        return features

    def energy(self) -> np.ndarray:
        """Total energy per regulatory module and temporal axis: E = mean(A²)."""
        return (np.abs(self.tensor) ** 2).mean(axis=-1)

    # ── Tensor Operations ─────────────────────────────────────────────────────

    def ema_update(self, cst_new: "CellStateTensor", lam: float = 0.9) -> "CellStateTensor":
        """
        Exponential moving average update: CST ← λ·CST + (1-λ)·CST_new.

        Parameters
        ----------
        cst_new : CellStateTensor   new observation.
        lam : float   smoothing factor ∈ [0, 1).

        Returns
        -------
        CellStateTensor   smoothed state.
        """
        new_tensor = lam * self.tensor + (1 - lam) * cst_new.tensor
        # Pull-back to unit circle
        new_tensor = new_tensor / (np.abs(new_tensor) + 1e-12)
        return CellStateTensor(
            tensor=new_tensor,
            regulatory_names=self.regulatory_names,
            temporal_names=self.temporal_names,
            homeostatic_names=self.homeostatic_names,
            metadata={**self.metadata, "ema_lambda": lam},
        )

    def to_real_features(self) -> np.ndarray:
        """
        Flatten CST to real-valued feature vector for ML classifiers.

        Concatenates phase and amplitude:
            features = [phase.flatten(), amplitude.flatten()]

        Returns
        -------
        np.ndarray, shape (2 * R * T * H,)
        """
        return np.concatenate([self.phase.flatten(), self.amplitude.flatten()])

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_omics_phases(
        cls,
        phases: dict,
        n_homeostatic: int = 4,
    ) -> "CellStateTensor":
        """
        Construct CST from multi-omics phase dictionaries.

        Parameters
        ----------
        phases : dict  {modality_name: np.ndarray of shape (n_genes,)}
            Phase vectors from different omics layers.
        n_homeostatic : int
            Number of homeostatic bins to partition features into.

        Returns
        -------
        CellStateTensor
        """
        modality_names = list(phases.keys())
        arrays = [np.asarray(v) for v in phases.values()]

        # Stack modalities as regulatory axis
        R = len(arrays)
        max_genes = max(a.shape[0] for a in arrays)

        # Pad to equal length and partition into homeostatic bins
        padded = np.zeros((R, max_genes))
        for i, a in enumerate(arrays):
            padded[i, :len(a)] = a

        # Reshape: (R, max_genes) → (R, T, H) via partitioning
        T = max(1, max_genes // n_homeostatic)
        H = n_homeostatic
        total = T * H
        padded_trimmed = padded[:, :total].reshape(R, T, H)

        tensor = np.exp(1j * padded_trimmed)
        return cls(
            tensor=tensor,
            regulatory_names=modality_names,
            metadata={"source": "from_omics_phases"},
        )

    @classmethod
    def random(
        cls,
        n_regulatory: int = 6,
        n_temporal: int = 6,
        n_homeostatic: int = 6,
        seed: int = 42,
    ) -> "CellStateTensor":
        """Create a random test CST (for development and testing)."""
        rng = np.random.default_rng(seed)
        phase = rng.uniform(-np.pi, np.pi, (n_regulatory, n_temporal, n_homeostatic))
        amp = rng.exponential(1.0, (n_regulatory, n_temporal, n_homeostatic))
        tensor = amp * np.exp(1j * phase)
        return cls(
            tensor=tensor,
            regulatory_names=REGULATORY_LABELS[:n_regulatory],
            temporal_names=TEMPORAL_LABELS[:n_temporal],
            homeostatic_names=HOMEOSTATIC_LABELS[:n_homeostatic],
        )

    # ── Display ───────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        gcm = self.global_coherence()
        return (
            f"CellStateTensor(R={self.n_regulatory}, T={self.n_temporal}, "
            f"H={self.n_homeostatic}, GCM={gcm:.3f})"
        )
