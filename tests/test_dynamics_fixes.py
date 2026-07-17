"""
Tests for the two Plan-II dynamics fixes:

  1. CellCyclePhasor continuous data-driven axis (replaces fixed-angle snap).
  2. CircadianPhasor.peak_zt — absolute peak-ZT anchored to the sampling clock.

Both use small synthetic data with known structure so they run fast and need
no external downloads.
"""

import numpy as np
import pytest

from biophasor.dynamics.cellcycle import CellCyclePhasor
from biophasor.dynamics.circadian import CircadianPhasor


# ── Cell-cycle: continuous axis ──────────────────────────────────────────────

def _synthetic_cellcycle_adata(n_per_phase=60, seed=0):
    """Build an AnnData where each phase's marker module is genuinely elevated
    in its own block of cells, so a working assigner must recover the blocks."""
    import anndata as ad
    import pandas as pd

    cc = CellCyclePhasor()
    phases = list(cc.marker_genes.keys())               # G1, S, G2, M
    genes = sorted({g for gs in cc.marker_genes.values() for g in gs})
    gi = {g: i for i, g in enumerate(genes)}
    rng = np.random.RandomState(seed)

    blocks, truth = [], []
    for ph in phases:
        X = rng.lognormal(0.0, 0.5, size=(n_per_phase, len(genes)))
        for g in cc.marker_genes[ph]:                   # elevate this phase's markers
            X[:, gi[g]] *= 6.0
        blocks.append(X)
        truth += [ph] * n_per_phase
    X = np.vstack(blocks)
    adata = ad.AnnData(X=X, var=pd.DataFrame(index=genes))
    return adata, np.array(truth)


class TestCellCycleContinuous:
    def test_continuous_beats_fixed_and_recovers_g1(self):
        from sklearn.metrics import adjusted_rand_score
        adata, truth = _synthetic_cellcycle_adata()

        lab_new, phi_new = adata_copy_assign(adata, method="continuous")
        lab_old, _ = adata_copy_assign(adata, method="fixed")

        def acc3(lab):
            t = np.array(["G2M" if p in ("G2", "M") else p for p in truth])
            l = np.array(["G2M" if p in ("G2", "M") else p for p in lab])
            return (t == l).mean(), adjusted_rand_score(t, l)

        acc_new, ari_new = acc3(lab_new)
        acc_old, ari_old = acc3(lab_old)

        # Continuous axis must beat the fixed-angle snap on this structured data.
        assert ari_new > ari_old
        assert acc_new > 0.55
        # G1 must actually be called (the fixed-angle failure mode was ~never).
        g1_calls = int((lab_new == "G1").sum())
        assert g1_calls > 10
        # Phase must be continuous, not snapped to 4 discrete values.
        assert len(np.unique(np.round(phi_new, 3))) > 20

    def test_shapes_and_obs(self):
        adata, _ = _synthetic_cellcycle_adata(n_per_phase=20)
        cc = CellCyclePhasor()
        labels, phi = cc.assign(adata, add_to_obs=True)
        assert labels.shape == (adata.n_obs,)
        assert phi.shape == (adata.n_obs,)
        assert "cell_cycle_phase" in adata.obs
        assert np.all(phi >= -np.pi - 1e-9) and np.all(phi <= np.pi + 1e-9)

    def test_bad_method_raises(self):
        adata, _ = _synthetic_cellcycle_adata(n_per_phase=10)
        with pytest.raises(ValueError):
            CellCyclePhasor().assign(adata, method="nonsense")


def adata_copy_assign(adata, method):
    """assign() on a fresh copy (assign writes obs in place)."""
    a = adata.copy()
    return CellCyclePhasor().assign(a, method=method, add_to_obs=False)


# ── Circadian: ZT anchoring ──────────────────────────────────────────────────

class TestCircadianPeakZT:
    def test_peak_zt_recovers_known_peaks(self):
        """A cosine peaking at a known ZT must be recovered to within one
        sampling step, anchored to the acquisition clock."""
        period = 24.0
        zt_times = np.arange(0, 24, 2.0)               # ZT0..22, Δt=2h
        cp = CircadianPhasor(period=period, sample_interval=2.0, zt_origin=0.0)

        for zt_peak in (2.0, 8.0, 14.0, 22.0):
            # cos(2π(t − zt_peak)/T) peaks at t = zt_peak; keep it non-negative.
            signal = 1.0 + np.cos(2 * np.pi * (zt_times - zt_peak) / period)
            X = signal[:, None]
            est = float(cp.peak_zt(X, zt_times=zt_times)[0])
            err = abs((est - zt_peak + period / 2) % period - period / 2)
            assert err <= 2.0 + 1e-6, f"peak {zt_peak}: est {est}, err {err}"

    def test_zt_origin_shifts_result(self):
        """A non-zero ZT origin shifts the reported peak by that origin."""
        period = 24.0
        T = 12
        zt = np.arange(T) * 2.0
        signal = 1.0 + np.cos(2 * np.pi * (zt - 6.0) / period)
        X = signal[:, None]
        cp0 = CircadianPhasor(period, 2.0, zt_origin=0.0)
        p0 = float(cp0.peak_zt(X)[0])
        p6 = float(cp0.peak_zt(X, zt_times=zt + 6.0)[0])
        assert abs((p6 - p0 - 6.0 + period / 2) % period - period / 2) <= 1e-6

    def test_infer_phase_unchanged_shape(self):
        cp = CircadianPhasor()
        X = np.abs(np.random.RandomState(0).randn(12, 30)) + 1.0
        phase = cp.infer_phase(X)
        assert phase.shape == (30,)
