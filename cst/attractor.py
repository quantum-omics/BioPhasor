"""
biophasor.cst.attractor — AttractorLandscape: cell-state basin analysis.

Maps the CST phase space to an attractor landscape, discovering
recurring cell states as energy minima in the Waddington quasi-potential.

Uses dimensionality reduction (PCA/UMAP) and density-based clustering
to visualise the basin-of-attraction structure of cellular regulatory
networks.

Biological applications:
  - Single-cell state classification (stem, progenitor, differentiated)
  - Cancer subtype discovery via attractor geometry
  - Drug response prediction from attractor transitions
  - Waddington landscape reconstruction

Reference: Biophasor Book — Ch 11 "Attractor Landscape"
           Huang et al. (2005) Phys. Rev. Lett. 94:128701

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from biophasor.cst.tensor import CellStateTensor


class AttractorLandscape:
    """
    Basin-of-attraction analysis in CST phase space.

    Fits an attractor model from a collection of CST observations,
    identifies stable cell states as attractors, and classifies
    new CSTs into their nearest attractor basin.

    Parameters
    ----------
    n_attractors : int   Number of cell-state attractors.
    method : {'kmeans', 'gmm'}   Clustering method.
    reducer : {'pca', 'umap', None}   Dimensionality reducer.
    seed : int

    Examples
    --------
    >>> landscape = AttractorLandscape(n_attractors=4)
    >>> landscape.fit(cst_list)
    >>> label = landscape.nearest_attractor(cst_new)
    >>> landscape.basin_plot()
    """

    # Default cell-state labels
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
        n_attractors: int = 4,
        method: str = "kmeans",
        reducer: Optional[str] = "pca",
        seed: int = 42,
    ) -> None:
        self.n_attractors = n_attractors
        self.method = method
        self.reducer = reducer
        self.seed = seed
        self._cluster_model = None
        self._reducer_model = None
        self._embeddings: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None

    # ── Feature extraction ────────────────────────────────────────────────────

    def _extract_features(self, cst: CellStateTensor) -> np.ndarray:
        """
        Extract a fixed-size feature vector from a CST.

        Features: [coherence_map, energy, global_coherence, phase_entropy, synchrony_index]
        """
        coh = cst.coherence_map().flatten()
        eng = cst.energy().flatten()
        scalars = np.array([
            cst.global_coherence(),
            cst.phase_entropy(),
            cst.synchrony_index(),
        ])
        return np.concatenate([coh, eng, scalars])

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, cst_list: list[CellStateTensor]) -> "AttractorLandscape":
        """
        Fit the attractor landscape from a collection of CSTs.

        Parameters
        ----------
        cst_list : list[CellStateTensor]   training CSTs

        Returns
        -------
        self
        """
        X = np.stack([self._extract_features(cst) for cst in cst_list])
        X_reduced = self._reduce(X, fit=True)

        if self.method == "kmeans":
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=self.n_attractors, n_init=10, random_state=self.seed)
        elif self.method == "gmm":
            from sklearn.mixture import GaussianMixture
            model = GaussianMixture(n_components=self.n_attractors, random_state=self.seed)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'kmeans' or 'gmm'.")

        self._cluster_model = model.fit(X_reduced)
        self._labels = model.predict(X_reduced)
        self._embeddings = X_reduced
        return self

    def _reduce(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply dimensionality reduction (PCA, UMAP, or identity)."""
        if self.reducer is None:
            return X
        if self.reducer == "pca":
            from sklearn.decomposition import PCA
            if fit:
                self._reducer_model = PCA(
                    n_components=min(50, X.shape[1]),
                    random_state=self.seed,
                )
                return self._reducer_model.fit_transform(X)
            return self._reducer_model.transform(X)
        elif self.reducer == "umap":
            try:
                import umap
            except ImportError:
                raise ImportError("Install umap-learn:  pip install umap-learn")
            if fit:
                self._reducer_model = umap.UMAP(n_components=2, random_state=self.seed)
                return self._reducer_model.fit_transform(X)
            return self._reducer_model.transform(X)
        else:
            raise ValueError(f"Unknown reducer '{self.reducer}'. Use 'pca' or 'umap'.")

    # ── Prediction ────────────────────────────────────────────────────────────

    def nearest_attractor(self, cst: CellStateTensor) -> int:
        """
        Classify a CST into its nearest cell-state attractor.

        Returns
        -------
        int  attractor label (0 to n_attractors - 1)
        """
        if self._cluster_model is None:
            raise RuntimeError("Call fit() first.")
        x = self._extract_features(cst)[np.newaxis, :]
        x_reduced = self._reduce(x, fit=False)
        return int(self._cluster_model.predict(x_reduced)[0])

    def attractor_name(self, label: int) -> str:
        """Map numeric label to cell-state name."""
        return self._CELL_STATE_LABELS.get(label, f"state_{label}")

    def attractor_probability(self, cst: CellStateTensor) -> np.ndarray:
        """
        Probability over all attractors.

        Returns
        -------
        np.ndarray, shape (n_attractors,)
        """
        if self.method != "gmm":
            x = self._extract_features(cst)[np.newaxis, :]
            x_reduced = self._reduce(x, fit=False)
            centers = self._cluster_model.cluster_centers_
            dists = np.linalg.norm(centers - x_reduced, axis=1)
            probs = 1.0 / (dists + 1e-8)
            return probs / probs.sum()
        x = self._extract_features(cst)[np.newaxis, :]
        x_reduced = self._reduce(x, fit=False)
        return self._cluster_model.predict_proba(x_reduced)[0]

    # ── Visualisation ─────────────────────────────────────────────────────────

    def basin_2d(self) -> tuple[np.ndarray, np.ndarray]:
        """
        2D projection of the attractor landscape for plotting.

        Returns
        -------
        embeddings_2d : np.ndarray, shape (n_samples, 2)
        labels : np.ndarray, shape (n_samples,)
        """
        if self._embeddings is None:
            raise RuntimeError("Call fit() first.")
        if self._embeddings.shape[1] > 2:
            from sklearn.decomposition import PCA
            emb2d = PCA(n_components=2).fit_transform(self._embeddings)
        else:
            emb2d = self._embeddings
        return emb2d, self._labels

    def basin_plot(self, ax=None, title: str = "Cell-State Attractor Landscape"):
        """Quick 2D scatter plot of the attractor landscape."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Install matplotlib:  pip install matplotlib")

        emb2d, labels = self.basin_2d()
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            emb2d[:, 0], emb2d[:, 1],
            c=labels, cmap="Set2", alpha=0.7, s=40, edgecolors="k", linewidths=0.3,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Phasor PC-1")
        ax.set_ylabel("Phasor PC-2")

        for k in range(self.n_attractors):
            mask = labels == k
            if mask.any():
                cx, cy = emb2d[mask].mean(axis=0)
                ax.annotate(
                    self.attractor_name(k),
                    (cx, cy),
                    fontsize=8,
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
                )
        plt.colorbar(scatter, ax=ax, label="Basin ID")
        return ax

    def __repr__(self) -> str:
        status = f"fitted, n_attractors={self.n_attractors}" if self._cluster_model else "not fitted"
        return f"AttractorLandscape(method={self.method!r}, {status})"
