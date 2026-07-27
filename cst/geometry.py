"""
biophasor.cst.geometry — AttractorGeometry: attractor basin geometry and transitions.

Characterizes the attractor landscape of phase-coupled dissipative cellular systems:
  1. Basin topology: number, size, and overlap of cell-state attractor basins
  2. Transition dynamics: inter-state transition rates and pathways (Markov matrix)
  3. Geometric invariants: Lyapunov exponents, fractal dimension proxies
  4. CST feature extraction from attractor geometry

These features feed directly into the Cell State Tensor, providing the
geometric foundation for cell-state representation and cell-fate prediction.

Biological applications:
  - Waddington landscape reconstruction from single-cell data
  - Cell-fate bifurcation detection
  - Cancer phenotypic plasticity quantification (transition entropy)
  - Stem cell attractor stability monitoring

Reference: Biophasor Book — Ch 11 "Attractor Landscape"
           Biophasor Manuscript — Section "Markov Transition Dynamics"

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BasinMetrics:
    """
    Metrics for a single cell-state attractor basin.

    Attributes
    ----------
    label : int
        Basin identifier.
    center : np.ndarray
        Phase-space center of the basin.
    radius : float
        Estimated basin radius (circular std of members).
    residence_time : float
        Mean time spent in this basin (in samples).
    occupancy : float
        Fraction of total trajectory in this basin.
    stability : float
        Inverse of the mean escape rate from this basin.
    cell_state : str
        Inferred cell-state label (if available).
    """
    label: int
    center: np.ndarray
    radius: float
    residence_time: float
    occupancy: float
    stability: float
    cell_state: str = ""

    def __repr__(self) -> str:
        state = f" ({self.cell_state})" if self.cell_state else ""
        return (
            f"Basin(label={self.label}{state}, R={self.radius:.3f}, "
            f"occ={self.occupancy:.2%}, τ_res={self.residence_time:.1f})"
        )


class AttractorGeometry:
    """
    Attractor basin geometry analysis for phase-coupled dissipative cellular systems.

    Analyzes CST or raw phase trajectories to characterize:
    - Basin structure (number, size, stability) — the Waddington landscape
    - Inter-basin transition dynamics (Markov matrix, transition entropy)
    - Geometric complexity (Lyapunov exponents, entropy)

    Parameters
    ----------
    n_basins : int
        Expected number of cell-state attractor basins.
    window_size : int
        Sliding window for local geometry analysis (in samples).
    overlap : float
        Window overlap fraction.
    cell_state_labels : dict | None
        Optional mapping {basin_id: "cell_state_name"}.

    Examples
    --------
    >>> from biophasor.cst.geometry import AttractorGeometry
    >>> geom = AttractorGeometry(n_basins=4)
    >>> geom.fit(phase_trajectory)
    >>> metrics = geom.basin_metrics()
    >>> T = geom.transition_matrix()
    >>> H_T = geom.transition_entropy()
    >>> features = geom.cst_features()
    """

    # Default cell-state labels for common experimental settings
    _CELL_STATE_LABELS = {
        0: "quiescent_G0",
        1: "proliferating",
        2: "differentiated",
        3: "apoptotic",
        4: "senescent",
        5: "stem_like",
        6: "mesenchymal",
        7: "drug_resistant",
    }

    def __init__(
        self,
        n_basins: int = 4,
        window_size: int = 64,
        overlap: float = 0.5,
        cell_state_labels: Optional[dict] = None,
    ) -> None:
        self.n_basins = n_basins
        self.window_size = window_size
        self.overlap = overlap
        self.state_labels = cell_state_labels or self._CELL_STATE_LABELS
        self._labels: Optional[np.ndarray] = None
        self._centers: Optional[np.ndarray] = None
        self._phase: Optional[np.ndarray] = None

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(self, phase: np.ndarray) -> "AttractorGeometry":
        """
        Fit attractor basins from a phase trajectory.

        Parameters
        ----------
        phase : np.ndarray, shape (N, T) or (T,)
            Phase trajectory on T^N.

        Returns
        -------
        self
        """
        if phase.ndim == 1:
            phase = phase[np.newaxis, :]
        self._phase = phase
        N, T = phase.shape

        step = max(1, int(self.window_size * (1.0 - self.overlap)))
        features = []
        for start in range(0, T - self.window_size + 1, step):
            window = phase[:, start:start + self.window_size]
            features.append(self._window_features(window))
        features = np.array(features)

        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=self.n_basins, n_init=10, random_state=42)
        self._labels = km.fit_predict(features)
        self._centers = km.cluster_centers_
        return self

    def _window_features(self, window: np.ndarray) -> np.ndarray:
        """
        Extract phase-geometric features from a single window.

        Features: [synchrony, entropy, velocity, coherence, per-osc circular_mean]
        """
        N, W = window.shape

        # Kuramoto order parameter (synchrony)
        R = float(np.abs(np.exp(1j * window).mean()))

        # Phase entropy
        n_bins = 36
        counts = np.histogram(window.flatten() % (2 * np.pi), bins=n_bins)[0]
        probs = counts / (counts.sum() + 1e-12)
        entropy = -np.sum(probs * np.log(probs + 1e-12))

        # Mean velocity
        velocity = float(np.abs(np.diff(np.unwrap(window, axis=-1), axis=-1)).mean())

        # Global coherence
        coherence = float(np.abs(np.exp(1j * window).mean(axis=0)).mean())

        # Per-oscillator circular means
        circ_means = np.angle(np.exp(1j * window).mean(axis=1))

        return np.concatenate([[R, entropy, velocity, coherence], circ_means])

    # ── Basin Metrics ─────────────────────────────────────────────────────────

    def basin_metrics(self) -> list[BasinMetrics]:
        """
        Compute metrics for each cell-state attractor basin.

        Returns
        -------
        list[BasinMetrics]
        """
        if self._labels is None:
            raise RuntimeError("Call fit() first.")

        metrics = []
        total = len(self._labels)
        for k in range(self.n_basins):
            mask = self._labels == k
            count = mask.sum()
            occupancy = count / total
            center = self._centers[k] if self._centers is not None else np.zeros(1)

            if count > 1:
                radius = float(np.std(np.where(mask)[0]))
            else:
                radius = 0.0

            runs = self._run_lengths(mask)
            residence = float(np.mean(runs)) if len(runs) > 0 else 0.0
            stability = residence / (total + 1e-8)

            metrics.append(BasinMetrics(
                label=k,
                center=center,
                radius=radius,
                residence_time=residence,
                occupancy=occupancy,
                stability=stability,
                cell_state=self.state_labels.get(k, f"state_{k}"),
            ))
        return metrics

    def _run_lengths(self, mask: np.ndarray) -> np.ndarray:
        """Compute consecutive run lengths of True values in a boolean array."""
        if not mask.any():
            return np.array([])
        diffs = np.diff(mask.astype(int))
        starts = np.where(diffs == 1)[0] + 1
        ends = np.where(diffs == -1)[0] + 1
        if mask[0]:
            starts = np.concatenate([[0], starts])
        if mask[-1]:
            ends = np.concatenate([ends, [len(mask)]])
        return ends - starts[:len(ends)]

    # ── Transition Matrix ─────────────────────────────────────────────────────

    def transition_matrix(self) -> np.ndarray:
        """
        Compute the Markov transition matrix between cell-state attractor basins.

        T[i,j] = P(basin j at t+1 | basin i at t)

        Returns
        -------
        np.ndarray, shape (n_basins, n_basins)
        """
        if self._labels is None:
            raise RuntimeError("Call fit() first.")

        T = np.zeros((self.n_basins, self.n_basins), dtype=np.float64)
        for i in range(len(self._labels) - 1):
            T[self._labels[i], self._labels[i + 1]] += 1

        row_sums = T.sum(axis=1, keepdims=True)
        T = np.where(row_sums > 0, T / row_sums, 0)
        return T

    def transition_entropy(self) -> float:
        """
        Entropy of the transition matrix.

        Higher H_T during active differentiation (multiple fates accessible)
        vs. lower H_T during terminal commitment.
        In cancer, H_T is pathologically elevated (phenotypic plasticity).
        """
        T = self.transition_matrix()
        T_flat = T.flatten()
        T_flat = T_flat[T_flat > 0]
        return float(-np.sum(T_flat * np.log(T_flat + 1e-12)))

    # ── Lyapunov Exponent ─────────────────────────────────────────────────────

    def max_lyapunov_exponent(self, phase: Optional[np.ndarray] = None) -> float:
        """
        Estimate the maximal Lyapunov exponent (Rosenstein algorithm).

        Positive λ_max → chaotic switching between attractor basins
        Negative λ_max → stable limit cycle dynamics

        Parameters
        ----------
        phase : np.ndarray | None
            Phase trajectory.  Uses fitted trajectory if None.

        Returns
        -------
        float  Estimated λ_max
        """
        if phase is None:
            phase = self._phase
        if phase is None:
            raise RuntimeError("No trajectory available. Call fit() or pass phase.")

        if phase.ndim == 1:
            phase = phase[np.newaxis, :]
        N, T = phase.shape

        embed = np.diff(np.unwrap(phase, axis=-1), axis=-1).T
        n = len(embed)
        if n < 20:
            return 0.0

        exclude_band = max(10, n // 20)
        divergences = []
        for i in range(min(n - exclude_band, 200)):
            dists = np.linalg.norm(embed - embed[i], axis=1)
            dists[max(0, i - exclude_band):min(n, i + exclude_band)] = np.inf
            j = np.argmin(dists)
            if j + 1 < n and i + 1 < n:
                d0 = dists[j]
                d1 = np.linalg.norm(embed[i + 1] - embed[j + 1])
                if d0 > 1e-12:
                    divergences.append(np.log(d1 / d0))

        return float(np.mean(divergences)) if divergences else 0.0

    # ── Quasi-Potential ────────────────────────────────────────────────────────

    def quasi_potential(self, n_grid: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate the Waddington quasi-potential landscape on a 2D grid.

        Uses log-density estimation from the fitted embedding as a
        proxy for the Freidlin-Wentzell quasi-potential U(φ).

        Returns
        -------
        X, Y : np.ndarray  meshgrid coordinates
        U : np.ndarray      quasi-potential values, shape (n_grid, n_grid)
        """
        if self._phase is None:
            raise RuntimeError("Call fit() first.")

        from sklearn.decomposition import PCA
        embed = np.diff(np.unwrap(self._phase, axis=-1), axis=-1).T
        if embed.shape[1] > 2:
            embed = PCA(n_components=2).fit_transform(embed)

        # KDE-based density estimation
        xmin, xmax = embed[:, 0].min(), embed[:, 0].max()
        ymin, ymax = embed[:, 1].min(), embed[:, 1].max()
        pad = 0.1 * max(xmax - xmin, ymax - ymin)
        xi = np.linspace(xmin - pad, xmax + pad, n_grid)
        yi = np.linspace(ymin - pad, ymax + pad, n_grid)
        X, Y = np.meshgrid(xi, yi)

        # Gaussian KDE
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(embed.T)
        positions = np.vstack([X.ravel(), Y.ravel()])
        density = kde(positions).reshape(X.shape)

        # Quasi-potential: U = -log(density)
        U = -np.log(density + 1e-12)
        U -= U.min()  # Normalize: deepest basin at U=0
        return X, Y, U

    # ── CST Feature Extraction ────────────────────────────────────────────────

    def cst_features(self) -> dict:
        """
        Extract attractor-geometric features for CST computation.

        Returns
        -------
        dict with keys:
            'n_active_basins': int — basins with occupancy > 5%
            'dominant_basin': int — most occupied basin
            'dominant_state': str — cell-state label of dominant basin
            'transition_entropy': float — complexity of state switching
            'mean_residence_time': float — average time in each basin
            'max_lyapunov_exponent': float — stability/chaos indicator
            'basin_occupancies': np.ndarray — occupancy per basin
        """
        metrics = self.basin_metrics()
        occupancies = np.array([m.occupancy for m in metrics])
        residences = np.array([m.residence_time for m in metrics])
        dominant = int(np.argmax(occupancies))

        return {
            "n_active_basins": int(np.sum(occupancies > 0.05)),
            "dominant_basin": dominant,
            "dominant_state": self.state_labels.get(dominant, f"state_{dominant}"),
            "transition_entropy": self.transition_entropy(),
            "mean_residence_time": float(residences.mean()),
            "max_lyapunov_exponent": self.max_lyapunov_exponent(),
            "basin_occupancies": occupancies,
        }

    def __repr__(self) -> str:
        status = f"fitted, {self.n_basins} basins" if self._labels is not None else "not fitted"
        return f"AttractorGeometry({status})"
