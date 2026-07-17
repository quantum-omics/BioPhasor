"""
biophasor.transform.encoder — OmicsPhasorEncoder and encoding functions.

Three strategies (validated in Notebook 1.1):

    1. linear       : φ = π · (2·minmax(x) − 1)
    2. log_linear   : φ = π · (2·minmax(log1p(x)) − 1)
    3. tanh_phase   : φ = π · tanh((log1p(x) − μ) / σ)      ← DEFAULT

The tanh-phase strategy (Manuscript Eq. 15) is the most robust across all
omics modalities; it handles high dynamic range and is outlier-resistant.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional, Literal

import numpy as np


# ── Module-level encoding functions ──────────────────────────────────────────

def tanh_phase_encode(
    X: np.ndarray,
    epsilon: float = 1e-8,
    log_transform: bool = True,
) -> np.ndarray:
    """
    **Canonical tanh-phase encoding** (Manuscript Eq. 15, Notebook 1.1 default).

        φ = π · tanh( (log1p(x) − μ) / σ )

    Produces phases in (−π, π] with approximately Gaussian spread (~1.88 rad std
    for typical RNA-seq data), robust to count outliers.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Raw or pre-processed omics matrix (non-negative).
    epsilon : float
        Small constant added to σ to avoid division by zero.
    log_transform : bool
        If True, apply log1p before standardising.  Set False for data already
        in log space (ATAC peaks, methylation β values).

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)   phases in (−π, π]
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.log1p(X) if log_transform else X.copy()
    mu = Z.mean(axis=0, keepdims=True)
    sig = Z.std(axis=0, keepdims=True) + epsilon
    phi = np.pi * np.tanh((Z - mu) / sig)
    return phi


def log_linear_encode(X: np.ndarray) -> np.ndarray:
    """
    Log-linear encoding: min-max scale log1p(x) then shift to (−π, π].

        φ = π · (2 · minmax(log1p(x)) − 1)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)   phases in (−π, π]
    """
    X = np.asarray(X, dtype=np.float64)
    Z = np.log1p(X)
    z_min = Z.min(axis=0, keepdims=True)
    z_max = Z.max(axis=0, keepdims=True)
    norm = (Z - z_min) / (z_max - z_min + 1e-12)   # [0, 1]
    return np.pi * (2.0 * norm - 1.0)               # (−π, π]


def linear_encode(X: np.ndarray) -> np.ndarray:
    """
    Naive linear min-max encoding.

        φ = π · (2 · minmax(x) − 1)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)

    Returns
    -------
    np.ndarray, shape (n_samples, n_features)   phases in (−π, π]
    """
    X = np.asarray(X, dtype=np.float64)
    x_min = X.min(axis=0, keepdims=True)
    x_max = X.max(axis=0, keepdims=True)
    norm = (X - x_min) / (x_max - x_min + 1e-12)
    return np.pi * (2.0 * norm - 1.0)


# ── OmicsPhasorEncoder class ──────────────────────────────────────────────────

class OmicsPhasorEncoder:
    """
    Unified encoder that converts any omics modality matrix into phasor phases.

    Parameters
    ----------
    modality : str
        Omics type: 'RNA', 'ATAC', 'protein', 'metabolite', 'methylation'.
    strategy : {'tanh', 'log_linear', 'linear'}
        Encoding strategy.  'tanh' is the default and recommended choice.
    epsilon : float
        Numerical stabiliser for tanh encoding.

    Examples
    --------
    >>> enc = OmicsPhasorEncoder(modality='RNA', strategy='tanh')
    >>> phi = enc.encode(X_rna)   # shape (n_samples, n_genes)
    """

    # Default strategies per modality (can be overridden)
    _MODALITY_DEFAULTS: dict[str, str] = {
        "RNA":         "tanh",
        "ATAC":        "tanh",
        "protein":     "tanh",
        "metabolite":  "tanh",
        "methylation": "log_linear",   # β values ∈ [0,1] — already in [0,1]
        "drug":        "linear",       # viability scores ∈ [0,1]
        "mutations":   "linear",       # binary matrix ∈ {0,1}
    }

    def __init__(
        self,
        modality: str = "RNA",
        strategy: Optional[Literal["tanh", "log_linear", "linear"]] = None,
        epsilon: float = 1e-8,
    ) -> None:
        self.modality = modality
        self.strategy = strategy or self._MODALITY_DEFAULTS.get(modality, "tanh")
        self.epsilon = epsilon

    def encode(
        self,
        X: np.ndarray,
        log_transform: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode the expression matrix X into phasor phases.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        log_transform : bool | None
            Override for log1p pre-processing (tanh strategy only).

        Returns
        -------
        np.ndarray   phases ∈ (−π, π]
        """
        if self.strategy == "tanh":
            # For methylation / drug / mutations, skip log_transform
            do_log = (
                log_transform
                if log_transform is not None
                else self.modality not in ("methylation", "drug", "mutations")
            )
            return tanh_phase_encode(X, epsilon=self.epsilon, log_transform=do_log)
        elif self.strategy == "log_linear":
            return log_linear_encode(X)
        elif self.strategy == "linear":
            return linear_encode(X)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'. "
                             f"Choose 'tanh', 'log_linear', or 'linear'.")

    def encode_multiomics(
        self,
        data_dict: dict[str, np.ndarray],
        concat: bool = True,
    ) -> dict[str, np.ndarray] | np.ndarray:
        """
        Encode multiple omics layers and optionally concatenate them.

        Follows the pipeline from Notebook 2.1 (CLL multi-omics):
            1. Encode each layer with its appropriate strategy.
            2. Concatenate columns → single feature matrix per sample.

        Parameters
        ----------
        data_dict : dict[str, np.ndarray]
            Keys are modality names; values are expression matrices
            (n_samples, n_features_m).
        concat : bool
            If True, return concatenated (n_samples, sum_features) array.
            If False, return dict of per-modality phase arrays.

        Returns
        -------
        np.ndarray (concat=True) or dict[str, np.ndarray]
        """
        encoded: dict[str, np.ndarray] = {}
        for mod, X in data_dict.items():
            enc = OmicsPhasorEncoder(modality=mod, strategy=self.strategy)
            encoded[mod] = enc.encode(X)

        if concat:
            return np.concatenate(list(encoded.values()), axis=1)
        return encoded

    def phase_std(self, X: np.ndarray) -> float:
        """Return the mean standard deviation of encoded phases (diagnostic)."""
        phi = self.encode(X)
        return float(phi.std(axis=0).mean())

    def __repr__(self) -> str:
        return f"OmicsPhasorEncoder(modality={self.modality!r}, strategy={self.strategy!r})"
