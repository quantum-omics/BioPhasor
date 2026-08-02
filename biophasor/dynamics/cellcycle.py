"""
biophasor.dynamics.cellcycle — Cell-cycle phase assignment via phasor signatures.

Maps single cells to G1/S/G2/M phases using canonical marker gene phasors:

    φ_cell = arg( Σ_{g ∈ markers} w_g · z_g )

Reference marker sets: Tirosh et al. 2016 (Seurat v2 gene sets).

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations
from typing import Optional, Union

import numpy as np


class CellCyclePhasor:
    """
    Assign cells to cell-cycle phases using phasor aggregation of marker genes.

    The phase of each cell is computed as the weighted circular mean over its
    marker gene phasors.  The closest reference phase determines the label.

    Parameters
    ----------
    marker_genes : dict[str, list[str]] | None
        Override canonical marker sets.  Keys must be 'G1', 'S', 'G2', 'M'.
    weights : dict[str, dict[str, float]] | None
        Optional per-gene weights within each phase.

    Examples
    --------
    >>> cc = CellCyclePhasor()
    >>> phase_labels, phi_cells = cc.assign(adata)
    """

    # Reference phase angles (radians) for the four cell-cycle phases
    REFERENCE_PHASES: dict[str, float] = {
        "G1": 0.0,
        "S":  np.pi / 2,
        "G2": np.pi,
        "M":  3 * np.pi / 2,
    }

    def __init__(
        self,
        marker_genes: Optional[dict] = None,
        weights: Optional[dict] = None,
    ) -> None:
        from biophasor.core.constants import CANONICAL_MARKER_GENES
        self.marker_genes = marker_genes or CANONICAL_MARKER_GENES
        self.weights = weights  # None → uniform weights

    # ── Fitting and assignment ─────────────────────────────────────────────────

    def assign(
        self,
        adata: "anndata.AnnData",
        layer: Optional[str] = None,
        encoding: str = "tanh",
        add_to_obs: bool = True,
        method: str = "continuous",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assign cells to cell-cycle phases.

        Parameters
        ----------
        adata : AnnData   must have var_names matching marker gene names
        layer : str | None   layer key for expression; uses .X if None
        encoding : str   phasor encoding strategy
        add_to_obs : bool   if True, add 'cell_cycle_phase' and 'cell_cycle_phi'
                            columns to adata.obs in place
        method : {'continuous', 'fixed'}
            'continuous' (default) — data-driven continuous cell-cycle axis: the
            per-phase marker module scores are embedded on a circle and each cell
            is labelled by its nearest *data-derived* phase anchor, so the phase
            floats with the biology instead of snapping to fixed reference angles.
            'fixed' — the legacy behaviour that snaps the circular mean of the
            four per-phase marker phasors to fixed reference angles
            (G1=0, S=π/2, G2=π, M=3π/2). Retained for reproducibility of the
            original feasibility result; on real cells the four marker phasors
            overlap and the mean collapses, so 'continuous' is preferred.

        Returns
        -------
        phase_labels : np.ndarray[str], shape (n_cells,)   'G1', 'S', 'G2', 'M'
        phi_cells    : np.ndarray[float], shape (n_cells,) ∈ (−π, π]
        """
        if method not in ("continuous", "fixed"):
            raise ValueError(f"method must be 'continuous' or 'fixed', got {method!r}")

        var_names = list(adata.var_names)
        X = np.array(adata.layers[layer] if layer else adata.X, dtype=np.float64)

        if method == "fixed":
            phase_labels, phi_cells = self._assign_fixed(X, var_names)
        else:
            phase_labels, phi_cells = self._assign_continuous(X, var_names)

        if add_to_obs:
            adata.obs["cell_cycle_phase"] = phase_labels
            adata.obs["cell_cycle_phi"]   = phi_cells

        return np.array(phase_labels), phi_cells

    # ── Continuous data-driven cell-cycle axis (default) ───────────────────────

    def _assign_continuous(
        self,
        X: np.ndarray,
        var_names: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Data-driven continuous cell-cycle phase.

        1. Per-phase marker *module score* = mean of z-scored marker expression
           (one score per cell per phase present).
        2. Embed the module-score matrix on a circle via its leading two
           principal components → a continuous angle φ per cell that traces the
           cell-cycle trajectory rather than snapping to fixed angles.
        3. Derive each phase's anchor angle as the circular mean of φ over the
           cells whose dominant (arg-max) module is that phase — anchors come
           from the marker biology, not from any external reference label.
        4. Label each cell by its nearest phase anchor.
        """
        phase_scores, phases_present = self._module_scores(X, var_names)
        if phase_scores.shape[1] == 0:
            raise ValueError(
                "No marker genes found in adata.  Check adata.var_names or "
                "supply custom marker_genes to CellCyclePhasor()."
            )

        # Centre and embed the module scores on a circle (leading 2 PCs).
        Sc = phase_scores - phase_scores.mean(axis=0, keepdims=True)
        U, s, _ = np.linalg.svd(Sc, full_matrices=False)
        k = min(2, U.shape[1])
        emb = U[:, :k] * s[:k]
        if emb.shape[1] < 2:                       # degenerate: single phase
            emb = np.column_stack([emb[:, 0], np.zeros(emb.shape[0])])
        phi_cells = np.arctan2(emb[:, 1], emb[:, 0])   # ∈ (−π, π]

        # Data-driven anchors from the dominant-module cells.
        dominant = np.array(phases_present)[phase_scores.argmax(axis=1)]
        anchor_labels, anchor_angles = [], []
        for ph in phases_present:
            m = dominant == ph
            if m.any():
                anchor_labels.append(ph)
                anchor_angles.append(np.angle(np.exp(1j * phi_cells[m]).mean()))
        anchor_angles = np.array(anchor_angles)

        phase_labels = self._closest_phase(phi_cells, anchor_angles, anchor_labels)
        return np.array(phase_labels), phi_cells

    def _module_scores(
        self,
        X: np.ndarray,
        var_names: list,
    ) -> tuple[np.ndarray, list]:
        """Per-cell z-scored marker module score for each phase present."""
        cols, present = [], []
        for phase, genes in self.marker_genes.items():
            gene_idx = [i for i, g in enumerate(var_names) if g in genes]
            if not gene_idx:
                continue
            Xs = X[:, gene_idx]
            z = (Xs - Xs.mean(axis=0)) / (Xs.std(axis=0) + 1e-9)
            cols.append(z.mean(axis=1))
            present.append(phase)
        if not cols:
            return np.empty((X.shape[0], 0)), []
        return np.stack(cols, axis=1), present

    # ── Legacy fixed reference-angle assignment ────────────────────────────────

    def _assign_fixed(
        self,
        X: np.ndarray,
        var_names: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Legacy: snap the circular mean of marker phasors to fixed angles."""
        from biophasor.transform.encoder import tanh_phase_encode

        phase_z = {}
        for phase, genes in self.marker_genes.items():
            gene_idx = [i for i, g in enumerate(var_names) if g in genes]
            if not gene_idx:
                continue
            X_sub = X[:, gene_idx]
            phi_sub = tanh_phase_encode(X_sub)
            phase_z[phase] = np.exp(1j * phi_sub).mean(axis=1)

        if not phase_z:
            raise ValueError(
                "No marker genes found in adata.  Check adata.var_names or "
                "supply custom marker_genes to CellCyclePhasor()."
            )

        all_z = np.stack(list(phase_z.values()), axis=1)
        z_cell = all_z.mean(axis=1)
        phi_cells = np.angle(z_cell)

        ref_phases = np.array(list(self.REFERENCE_PHASES.values()))
        labels = list(self.REFERENCE_PHASES.keys())
        phase_labels = self._closest_phase(phi_cells, ref_phases, labels)
        return np.array(phase_labels), phi_cells

    @staticmethod
    def _closest_phase(
        phi: np.ndarray,
        ref_phases: np.ndarray,
        labels: list,
    ) -> list:
        """Assign each cell to the closest reference phase."""
        assigned = []
        for p in phi:
            # Circular distance to each reference
            diffs = np.abs((p - ref_phases + np.pi) % (2 * np.pi) - np.pi)
            assigned.append(labels[int(np.argmin(diffs))])
        return assigned

    # ── Score ──────────────────────────────────────────────────────────────────

    def phase_scores(
        self,
        adata: "anndata.AnnData",
        layer: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """
        Compute a score for each cell-cycle phase as coherence of its markers.

        Returns
        -------
        dict[str, np.ndarray]   {phase: (n_cells,) float score ∈ [0,1]}
        """
        from biophasor.transform.encoder import tanh_phase_encode
        from biophasor.core.operators import coherence

        X = np.array(adata.layers[layer] if layer else adata.X, dtype=np.float64)
        var_names = list(adata.var_names)
        scores = {}

        for phase, genes in self.marker_genes.items():
            gene_idx = [i for i, g in enumerate(var_names) if g in genes]
            if not gene_idx:
                continue
            X_sub = X[:, gene_idx]
            phi_sub = tanh_phase_encode(X_sub)
            # Per-cell coherence over the markers of this phase
            C = np.abs(np.exp(1j * phi_sub).mean(axis=1))
            scores[phase] = C

        return scores
