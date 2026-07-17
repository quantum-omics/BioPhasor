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
