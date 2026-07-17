"""
biophasor.core.phasor — BioPhasor: the fundamental data type.

Every omics feature is represented as a complex phasor  z = A · e^{iφ}
where A is the expression amplitude and φ is the biological phase.

Encoding strategies (Manuscript Eq. 15):

    Strategy 1 — Rank-based (uniform coverage):
        φ = 2π · rank(x) / N

    Strategy 2 — Tanh-phase (DEFAULT; outlier-robust):
        φ = π · tanh( (log1p(x) − μ) / σ )     per-feature z-score in log space

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import torch


@dataclass
class BioPhasor:
    """
    Core BioPhasor container.

    Stores omics data as complex phasors z = amplitude * exp(i * phase).

    Parameters
    ----------
    data : np.ndarray | torch.Tensor, shape (n_samples, n_features)
        Raw or pre-processed omics matrix.  May be provided instead of
        explicit ``amplitude`` / ``phase`` when those are computed lazily.
    amplitude : np.ndarray | torch.Tensor, shape (n_samples, n_features)
        Amplitude A = |z|.  If None it is inferred as |data| or set to 1.
    phase : np.ndarray | torch.Tensor, shape (n_samples, n_features)
        Phase φ = arg(z) in radians (range [−π, π]).
    modality : str
        Omics modality label, e.g. "RNA", "ATAC", "protein", "metabolite".
    feature_names : list[str] | None
        Gene / feature identifiers aligned with axis-1.
    sample_names : list[str] | None
        Sample / cell identifiers aligned with axis-0.
    metadata : dict
        Arbitrary metadata (batch, condition, time-point, …).
    """

    data: Optional[Union[np.ndarray, torch.Tensor]] = None
    amplitude: Optional[Union[np.ndarray, torch.Tensor]] = None
    phase: Optional[Union[np.ndarray, torch.Tensor]] = None
    modality: str = "unknown"
    feature_names: Optional[list] = None
    sample_names: Optional[list] = None
    metadata: dict = field(default_factory=dict)

    # ── Post-init validation ─────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.phase is not None:
            self.phase = self._ensure_numpy(self.phase)
        if self.amplitude is not None:
            self.amplitude = self._ensure_numpy(self.amplitude)
        if self.data is not None:
            self.data = self._ensure_numpy(self.data)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def complex(self) -> np.ndarray:
        """Return phasors as complex NumPy array z = A · e^{iφ}.

        If amplitude is not provided, unit amplitude (A = 1) is assumed.
        """
        if self.phase is None:
            raise ValueError("Phase is not set.  Call an encoder first.")
        A = self.amplitude if self.amplitude is not None else np.ones_like(self.phase)
        return A * np.exp(1j * self.phase)

    @property
    def shape(self) -> tuple:
        """Shape of the phasor array (n_samples, n_features)."""
        if self.phase is not None:
            return self.phase.shape
        if self.data is not None:
            return self.data.shape
        return ()

    @property
    def n_samples(self) -> int:
        return self.shape[0] if self.shape else 0

    @property
    def n_features(self) -> int:
        return self.shape[1] if len(self.shape) > 1 else 0

    # ── Encoding helpers ─────────────────────────────────────────────────────

    def encode_tanh(
        self,
        epsilon: float = 1e-8,
    ) -> "BioPhasor":
        """
        Encode raw data as phasors using the **canonical tanh-phase** formula
        (Manuscript Eq. 15):

            φ = π · tanh( (log1p(x) − μ) / σ )

        Returns
        -------
        BioPhasor
            Self (phase and amplitude are set in place) for method chaining.
        """
        if self.data is None:
            raise ValueError("data must be set before calling encode_tanh().")
        X = self.data.astype(np.float64)
        log_x = np.log1p(X)
        mu = log_x.mean(axis=0, keepdims=True)
        sig = log_x.std(axis=0, keepdims=True) + epsilon
        self.phase = np.pi * np.tanh((log_x - mu) / sig)
        # Unit amplitude for pure-phase mode
        self.amplitude = np.ones_like(self.phase)
        return self

    def encode_rank(self) -> "BioPhasor":
        """
        Rank-based phase encoding (uniform distribution on circle):

            φ = 2π · rank(x) / N
        """
        if self.data is None:
            raise ValueError("data must be set before calling encode_rank().")
        X = self.data.astype(np.float64)
        from scipy.stats import rankdata
        ranks = np.apply_along_axis(rankdata, 0, X)  # rank per feature
        N = X.shape[0]
        self.phase = 2.0 * np.pi * ranks / N - np.pi  # centre on 0
        self.amplitude = np.ones_like(self.phase)
        return self

    # ── Arithmetic ───────────────────────────────────────────────────────────

    def __add__(self, other: "BioPhasor") -> "BioPhasor":
        """Element-wise addition in the complex domain."""
        z = self.complex + other.complex
        return BioPhasor.from_complex(z, modality=self.modality, metadata=self.metadata)

    def __mul__(self, scalar: float) -> "BioPhasor":
        """Scalar multiplication (scales amplitude)."""
        z = scalar * self.complex
        return BioPhasor.from_complex(z, modality=self.modality, metadata=self.metadata)

    def __repr__(self) -> str:
        return (
            f"BioPhasor(modality={self.modality!r}, shape={self.shape}, "
            f"features={self.n_features})"
        )

    # ── Constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_complex(
        cls,
        z: np.ndarray,
        modality: str = "unknown",
        feature_names: Optional[list] = None,
        sample_names: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> "BioPhasor":
        """Construct from a complex array z = A·e^{iφ}."""
        return cls(
            amplitude=np.abs(z),
            phase=np.angle(z),
            modality=modality,
            feature_names=feature_names,
            sample_names=sample_names,
            metadata=metadata or {},
        )

    @classmethod
    def from_expression(
        cls,
        X: np.ndarray,
        modality: str = "RNA",
        encoding: str = "tanh",
        feature_names: Optional[list] = None,
        sample_names: Optional[list] = None,
        metadata: Optional[dict] = None,
    ) -> "BioPhasor":
        """
        Convenience factory: raw expression matrix → encoded BioPhasor.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        modality : str
        encoding : {'tanh', 'rank'}
            'tanh' — canonical tanh-phase (default, Manuscript Eq. 15)
            'rank' — uniform rank-based encoding
        """
        bp = cls(
            data=X,
            modality=modality,
            feature_names=feature_names,
            sample_names=sample_names,
            metadata=metadata or {},
        )
        if encoding == "tanh":
            bp.encode_tanh()
        elif encoding == "rank":
            bp.encode_rank()
        else:
            raise ValueError(f"Unknown encoding '{encoding}'. Choose 'tanh' or 'rank'.")
        return bp

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _ensure_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float64)

    def to_torch(self) -> torch.Tensor:
        """Return complex phasor as a PyTorch complex tensor."""
        z = self.complex
        return torch.view_as_complex(
            torch.tensor(np.stack([z.real, z.imag], axis=-1), dtype=torch.float32)
        )
