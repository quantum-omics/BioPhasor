"""
Tests for biophasor.integration — small, fast samples.
"""

import numpy as np
import pytest
from biophasor.integration.multiomics import MultiOmicsIntegrator, integrate


RNG = np.random.RandomState(3)
N, F = 20, 25   # small!


class TestMultiOmicsIntegrator:
    def setup_method(self):
        self.phases = {
            "RNA":     RNG.uniform(-np.pi, np.pi, (N, F)),
            "ATAC":    RNG.uniform(-np.pi, np.pi, (N, F)),
            "protein": RNG.uniform(-np.pi, np.pi, (N, F)),
        }

    def test_fuse_circular_mean_shape(self):
        fused = MultiOmicsIntegrator().fuse(self.phases, method="circular_mean")
        assert fused.shape == (N, F)

    def test_fuse_circular_mean_range(self):
        fused = MultiOmicsIntegrator().fuse(self.phases, method="circular_mean")
        assert fused.min() >= -np.pi - 1e-9 and fused.max() <= np.pi + 1e-9

    def test_fuse_concat_shape(self):
        fused = MultiOmicsIntegrator().fuse(self.phases, method="concat")
        assert fused.shape == (N, 3 * F)

    def test_cross_coherence_values_in_range(self):
        cc = MultiOmicsIntegrator().cross_coherence(self.phases)
        for pair, val in cc.items():
            assert 0.0 <= val <= 1.0 + 1e-9

    def test_cross_coherence_symmetric(self):
        cc = MultiOmicsIntegrator().cross_coherence(self.phases)
        assert cc[("RNA", "ATAC")] == pytest.approx(cc[("ATAC", "RNA")])

    def test_coherence_matrix_shape(self):
        C = MultiOmicsIntegrator().coherence_matrix(self.phases)
        assert C.shape == (3, 3)

    def test_coherence_matrix_diagonal_in_01(self):
        C = MultiOmicsIntegrator().coherence_matrix(self.phases)
        diag = np.diag(C)
        assert diag.min() >= 0.0 - 1e-9 and diag.max() <= 1.0 + 1e-9

    def test_layer_stats(self):
        stats = MultiOmicsIntegrator().layer_stats(self.phases)
        assert set(stats.keys()) == {"RNA", "ATAC", "protein"}
        for v in stats.values():
            assert "mean_coherence" in v

    def test_mismatched_features_raises(self):
        bad = {"RNA": RNG.uniform(-np.pi, np.pi, (N, F)),
               "ATAC": RNG.uniform(-np.pi, np.pi, (N, F + 5))}
        with pytest.raises(ValueError):
            MultiOmicsIntegrator().fuse(bad, method="circular_mean")


class TestIntegrateFunction:
    def test_integrate_shape(self):
        arrays = [RNG.uniform(-np.pi, np.pi, (N, F)) for _ in range(3)]
        assert integrate(arrays, method="circular_mean").shape == (N, F)

    def test_integrate_concat(self):
        arrays = [RNG.uniform(-np.pi, np.pi, (N, F)) for _ in range(2)]
        assert integrate(arrays, method="concat").shape == (N, 2 * F)


class TestCoherenceGatedFusion:
    """Coherence-gated / coherence-weighted fusion (the diagnosed fix for the
    'fused beats best single layer' claim). The core scientific property is
    that per-feature gating can exceed the more coherent input, whereas a
    global circular mean can never exceed it."""

    def setup_method(self):
        from biophasor.core.operators import coherence
        self.coherence = coherence
        # Build two layers with DIFFERENT per-feature coherence structure so
        # that gating has something to choose between. Layer A is coherent on
        # the first half of features, layer B on the second half.
        rng = np.random.RandomState(7)
        base = rng.uniform(-np.pi, np.pi, F)
        A = np.tile(base, (N, 1)).copy()
        B = np.tile(base, (N, 1)).copy()
        # add noise so first-half is tight in A, second-half tight in B
        A[:, F // 2:] += rng.normal(0, 1.5, (N, F - F // 2))
        B[:, :F // 2] += rng.normal(0, 1.5, (N, F // 2))
        self.phases = {"RNA": A, "protein": B}

    def test_gated_shape_and_range(self):
        fused = MultiOmicsIntegrator().fuse(self.phases, method="coherence_gated")
        assert fused.shape == (N, F)
        assert fused.min() >= -np.pi - 1e-9 and fused.max() <= np.pi + 1e-9

    def test_weighted_shape_and_range(self):
        fused = MultiOmicsIntegrator().fuse(self.phases, method="coherence_weighted")
        assert fused.shape == (N, F)
        assert fused.min() >= -np.pi - 1e-9 and fused.max() <= np.pi + 1e-9

    def test_gated_beats_best_single_layer(self):
        """The key property: gated fusion coherence >= max single-layer coherence,
        while the legacy uniform circular mean does NOT."""
        mi = MultiOmicsIntegrator()
        cR = float(self.coherence(self.phases["RNA"], axis=0).mean())
        cP = float(self.coherence(self.phases["protein"], axis=0).mean())
        best_single = max(cR, cP)
        gated = mi.fuse(self.phases, method="coherence_gated")
        c_gated = float(self.coherence(gated, axis=0).mean())
        assert c_gated >= best_single - 1e-9, (c_gated, best_single)

    def test_gated_selects_per_feature_max(self):
        """Each fused feature must equal the phase of whichever layer has the
        higher across-sample coherence for that feature."""
        mi = MultiOmicsIntegrator()
        gated = mi.fuse(self.phases, method="coherence_gated")
        cA = self.coherence(self.phases["RNA"], axis=0)
        cB = self.coherence(self.phases["protein"], axis=0)
        expect = np.where(cA >= cB, self.phases["RNA"], self.phases["protein"])
        np.testing.assert_allclose(gated, expect, atol=1e-9)
