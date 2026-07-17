"""
Tests for biophasor.dynamics — small, fast samples.
"""

import numpy as np
import pytest
from biophasor.dynamics.kuramoto import BioKuramoto
from biophasor.dynamics.synchrony import SynchronyMetrics
from biophasor.dynamics.circadian import CircadianPhasor


RNG = np.random.RandomState(7)


class TestBioKuramoto:
    def test_order_parameter_range(self):
        km = BioKuramoto(n_oscillators=20, coupling=0.0)
        assert 0.0 <= km.order_parameter <= 1.0

    def test_high_coupling_synchronises(self):
        km = BioKuramoto(n_oscillators=20, coupling=20.0, noise=0.0, seed=0)
        km.phi = np.zeros(20)
        km.simulate(n_steps=100, dt=0.05)
        assert km.order_parameter > 0.85

    def test_simulate_shape(self):
        km = BioKuramoto(n_oscillators=10, coupling=1.0)
        traj = km.simulate(n_steps=10, dt=0.01, record_every=2)
        assert traj.shape == (5, 10)

    def test_phases_wrapped(self):
        km = BioKuramoto(n_oscillators=15, coupling=5.0)
        km.simulate(n_steps=50, dt=0.01)
        assert km.phi.max() <= np.pi + 1e-9 and km.phi.min() >= -np.pi - 1e-9


class TestSynchronyMetrics:
    def setup_method(self):
        self.phi_sync   = np.full((20, 10), np.pi / 3)
        self.phi_random = RNG.uniform(-np.pi, np.pi, (20, 10))

    def test_order_parameter_sync(self):
        sm = SynchronyMetrics(self.phi_sync)
        np.testing.assert_allclose(sm.order_parameter(axis=0), 1.0, atol=1e-9)

    def test_order_parameter_range(self):
        sm = SynchronyMetrics(self.phi_random)
        R = sm.order_parameter(axis=0)
        assert R.min() >= 0.0 - 1e-9 and R.max() <= 1.0 + 1e-9

    def test_plv_matrix_shape(self):
        sm = SynchronyMetrics(self.phi_random)
        PLV = sm.plv_matrix()
        assert PLV.shape == (10, 10)

    def test_plv_diagonal_is_one(self):
        sm = SynchronyMetrics(self.phi_random)
        np.testing.assert_allclose(np.diag(sm.plv_matrix()), 1.0, atol=1e-9)

    def test_synchronisation_index_range(self):
        SI = SynchronyMetrics(self.phi_random).synchronisation_index()
        assert 0.0 <= SI <= 1.0


class TestCircadianPhasor:
    def test_simulate_shape(self):
        cp = CircadianPhasor(period=24.0, sample_interval=2.0)
        ts = cp.simulate(n_cycles=1, n_genes=5, seed=0)
        assert ts.shape == (12, 5)

    def test_infer_phase_range(self):
        cp = CircadianPhasor(period=24.0, sample_interval=2.0)
        ts = cp.simulate(n_cycles=2, n_genes=8)
        phi = cp.infer_phase(ts)
        assert phi.shape == (8,)
        assert phi.min() >= -np.pi - 1e-9 and phi.max() <= np.pi + 1e-9

    def test_zt_round_trip(self):
        zt = 18.0
        assert abs(CircadianPhasor.phase_to_zt(CircadianPhasor.zt_to_phase(zt)) - zt) < 1e-9

    def test_rhythmicity_score_range(self):
        cp = CircadianPhasor(period=24.0, sample_interval=2.0)
        ts = cp.simulate(n_cycles=2, n_genes=6)
        score = cp.rhythmicity_score(ts)
        assert score.min() >= 0.0 and score.max() <= 1.001
