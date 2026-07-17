"""
Tests for biophasor.core — small, fast samples.
"""

import numpy as np
import pytest
from biophasor.core.phasor import BioPhasor
from biophasor.core.manifold import PhasorManifold
from biophasor.core.operators import coherence, phasor_mean, bio_shift, bio_mix, coherence_filter
from biophasor.core.constants import CELL_CYCLE_PHASES, CIRCADIAN_PHASES, CANONICAL_MARKER_GENES


RNG = np.random.RandomState(42)
N, F = 30, 40   # small!


class TestBioPhasor:
    def setup_method(self):
        self.X = RNG.lognormal(0, 1, (N, F))

    def test_from_expression_tanh(self):
        bp = BioPhasor.from_expression(self.X, modality="RNA", encoding="tanh")
        assert bp.phase.shape == (N, F)
        assert bp.phase.max() <= np.pi + 1e-9
        assert bp.phase.min() >= -np.pi - 1e-9

    def test_from_expression_rank(self):
        bp = BioPhasor.from_expression(self.X, encoding="rank")
        assert bp.phase.max() <= np.pi + 1e-9

    def test_complex_property_amplitude_one(self):
        bp = BioPhasor.from_expression(self.X, encoding="tanh")
        np.testing.assert_allclose(np.abs(bp.complex).mean(), 1.0, atol=0.01)

    def test_from_complex_round_trip(self):
        bp = BioPhasor.from_expression(self.X, encoding="tanh")
        bp2 = BioPhasor.from_complex(bp.complex)
        np.testing.assert_allclose(bp2.phase, bp.phase, atol=1e-9)

    def test_shape(self):
        bp = BioPhasor.from_expression(self.X)
        assert bp.n_samples == N and bp.n_features == F

    def test_invalid_encoding(self):
        with pytest.raises(ValueError):
            BioPhasor.from_expression(self.X, encoding="bad")


class TestOperators:
    def setup_method(self):
        self.phi = RNG.uniform(-np.pi, np.pi, (N, F))

    def test_coherence_range(self):
        C = coherence(self.phi, axis=0)
        assert C.min() >= 0.0 - 1e-9 and C.max() <= 1.0 + 1e-9

    def test_coherence_perfect_sync(self):
        C = coherence(np.full((N, F), 1.23), axis=0)
        np.testing.assert_allclose(C, 1.0, atol=1e-9)

    def test_phasor_mean_direction(self):
        mu = phasor_mean(np.full((N, F), np.pi / 4), axis=0)
        np.testing.assert_allclose(mu, np.pi / 4, atol=1e-9)

    def test_bio_shift_wrapping(self):
        shifted = bio_shift(np.array([np.pi - 0.1]), 0.5)
        assert shifted[0] < 0

    def test_bio_mix_shape(self):
        assert bio_mix(self.phi, self.phi).shape == (N, F)

    def test_coherence_filter_keeps_coherent(self):
        phi = self.phi.copy()
        phi[:, :F//2] = 0.0   # perfect coherence for first half
        _, mask = coherence_filter(phi, threshold=0.9)
        assert mask[:F//2].all()


class TestManifold:
    def setup_method(self):
        self.phi = RNG.uniform(-np.pi, np.pi, (20, 10))

    def test_pairwise_distance_shape_symmetry(self):
        D = PhasorManifold.pairwise_distance(self.phi)
        assert D.shape == (20, 20)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_frechet_mean(self):
        mu = PhasorManifold.frechet_mean(np.full((N, F), 0.7))
        np.testing.assert_allclose(mu, 0.7, atol=1e-9)

    def test_geodesic_interpolation(self):
        mid = PhasorManifold.geodesic_interp(np.zeros(5), np.full(5, np.pi/2), 0.5)
        np.testing.assert_allclose(mid, np.pi / 4, atol=1e-9)

    def test_log_exp_round_trip(self):
        base = np.zeros(10)
        v = np.random.uniform(-0.5, 0.5, 10)
        np.testing.assert_allclose(PhasorManifold.exp_map(base, PhasorManifold.log_map(base, v)), v, atol=1e-9)


class TestConstants:
    def test_cell_cycle_keys(self):
        assert set(CELL_CYCLE_PHASES.keys()) == {"G1", "S", "G2", "M"}

    def test_marker_genes_not_empty(self):
        for genes in CANONICAL_MARKER_GENES.values():
            assert len(genes) > 0
