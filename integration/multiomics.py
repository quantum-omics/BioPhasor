"""
biophasor.integration.multiomics — Multi-omics coherence fusion.

Integrates N phasor layers (mRNA, ATAC, protein, metabolite …) into a single
coherent representation using weighted circular mean:

    Z_fused = Σ_m α_m · Z_m / |Σ_m α_m · Z_m|     (normalised phasor sum)

Cross-layer coherence:
    C(m1, m2) = |<Z_m1 · conj(Z_m2)>| / (||Z_m1|| · ||Z_m2||)

Following the pipeline in Notebooks 2.1 / 2.2 (CLL multi-omics):
    1. Encode each modality → phase matrix
    2. Concatenate or fuse
    3. Measure per-layer and cross-layer coherence

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional, Union

import numpy as np


class MultiOmicsIntegrator:
    """
    Integrate multiple omics phasor layers into one representation.

    Parameters
    ----------
    modalities : list[str]
        Names of the omics modalities (e.g. ['RNA', 'ATAC', 'protein']).
    weights : list[float] | None
        Per-modality weights α_m.  If None, uniform weights are used.

    Examples
    --------
    >>> integrator = MultiOmicsIntegrator(['RNA', 'protein'], weights=[0.6, 0.4])
    >>> fused = integrator.fuse({'RNA': phi_rna, 'protein': phi_protein})
    >>> C_matrix = integrator.cross_coherence({'RNA': phi_rna, 'protein': phi_protein})
    """

    def __init__(
        self,
        modalities: Optional[list] = None,
        weights: Optional[list] = None,
    ) -> None:
        self.modalities = modalities or []
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            self.weights = w / w.sum()
        else:
            self.weights = None

    # ── Fusion ────────────────────────────────────────────────────────────────

    def fuse(
        self,
        phase_dict: dict[str, np.ndarray],
        method: str = "circular_mean",
    ) -> np.ndarray:
        """
        Fuse multiple phase matrices into one.

        Parameters
        ----------
        phase_dict : dict[str, np.ndarray]
            Keys = modality names; values = phase arrays (n_samples, n_features_m).
        method : {'circular_mean', 'concat'}
            'circular_mean' — weighted circular mean across same features.
            'concat'        — concatenate features from all layers.

        Returns
        -------
        np.ndarray   fused phase array (n_samples, n_features) or (n_samples, sum_features)
        """
        keys = list(phase_dict.keys())
        phases = [phase_dict[k] for k in keys]

        if method == "concat":
            return np.concatenate(phases, axis=1)

        # For circular_mean: arrays must have the same number of features
        n_features_list = [p.shape[1] for p in phases]
        if len(set(n_features_list)) > 1:
            raise ValueError(
                "For circular_mean fusion, all phase arrays must have the same number "
                f"of features.  Got: {dict(zip(keys, n_features_list))}. "
                "Use method='concat' or align features first."
            )

        # Weighted complex mean
        if self.weights is not None:
            w = self.weights[:len(phases)]
        else:
            w = np.ones(len(phases)) / len(phases)

        z_sum = sum(w[i] * np.exp(1j * phases[i]) for i in range(len(phases)))
        return np.angle(z_sum)

    # ── Cross-layer coherence ─────────────────────────────────────────────────

    def cross_coherence(
        self,
        phase_dict: dict[str, np.ndarray],
    ) -> dict[tuple[str, str], float]:
        """
        Compute pairwise cross-layer coherence.

            C(m1, m2) = |(1/F) Σ_f <z_{f}^{m1} · conj(z_{f}^{m2})>_{samples}|

        Returns
        -------
        dict[(str, str), float]   coherence for all ordered pairs
        """
        result = {}
        keys = list(phase_dict.keys())
        for i, k1 in enumerate(keys):
            z1 = np.exp(1j * phase_dict[k1])       # (n_samples, n_features)
            for k2 in keys[i + 1:]:
                z2 = np.exp(1j * phase_dict[k2])
                # Take minimum feature dimension for cross-layer
                F = min(z1.shape[1], z2.shape[1])
                cross = np.abs((z1[:, :F] * z2[:, :F].conj()).mean(axis=0)).mean()
                result[(k1, k2)] = float(cross)
                result[(k2, k1)] = float(cross)
        return result

    def coherence_matrix(
        self,
        phase_dict: dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Return an (n_modalities × n_modalities) coherence matrix.

        Diagonal entries = per-layer mean coherence (Kuramoto R).
        Off-diagonal entries = cross-layer coherence C(mi, mj).
        """
        keys = list(phase_dict.keys())
        n = len(keys)
        from biophasor.core.operators import coherence

        C = np.eye(n, dtype=np.float64)
        cross = self.cross_coherence(phase_dict)

        for i, k in enumerate(keys):
            C[i, i] = float(coherence(phase_dict[k], axis=0).mean())

        for i, k1 in enumerate(keys):
            for j, k2 in enumerate(keys):
                if i != j:
                    C[i, j] = cross.get((k1, k2), 0.0)

        return C

    # ── Per-layer statistics ──────────────────────────────────────────────────

    def layer_stats(
        self,
        phase_dict: dict[str, np.ndarray],
    ) -> dict[str, dict]:
        """
        Compute per-layer statistics: mean coherence, phase spread (std), min/max.

        Returns
        -------
        dict[str, dict]   {modality: {'coherence': float, 'phase_std': float}}
        """
        from biophasor.core.operators import coherence

        stats = {}
        for key, phi in phase_dict.items():
            C = coherence(phi, axis=0)
            stats[key] = {
                "mean_coherence":  float(C.mean()),
                "median_coherence": float(np.median(C)),
                "phase_std":       float(phi.std(axis=0).mean()),
                "phase_range":     (float(phi.min()), float(phi.max())),
            }
        return stats


# ── Module-level convenience function ────────────────────────────────────────

def integrate(
    phase_arrays: list[np.ndarray],
    weights: Optional[list] = None,
    method: str = "circular_mean",
) -> np.ndarray:
    """
    Shortcut: fuse a list of per-modality phase arrays.

    Parameters
    ----------
    phase_arrays : list[np.ndarray]   each (n_samples, n_features)
    weights : list[float] | None
    method : str   'circular_mean' or 'concat'

    Returns
    -------
    np.ndarray   fused phase
    """
    names = [f"mod_{i}" for i in range(len(phase_arrays))]
    integrator = MultiOmicsIntegrator(names, weights=weights)
    phase_dict = dict(zip(names, phase_arrays))
    return integrator.fuse(phase_dict, method=method)
