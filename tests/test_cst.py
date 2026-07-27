"""
Test suite for biophasor.cst — Cell State Tensor sub-package.

Tests all five core modules:
  - CellStateTensor
  - LimitCycleAnalyzer
  - AttractorGeometry
  - AttractorLandscape
  - CSTDynamics

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pytest


# ════════════════════════════════════════════════════════════════════════════════
# CellStateTensor
# ════════════════════════════════════════════════════════════════════════════════

class TestCellStateTensor:

    def test_random_creation(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random(n_regulatory=4, n_temporal=3, n_homeostatic=5)
        assert cst.shape == (4, 3, 5)
        assert cst.n_regulatory == 4
        assert cst.n_temporal == 3
        assert cst.n_homeostatic == 5

    def test_phase_amplitude(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random(n_regulatory=3, n_temporal=3, n_homeostatic=3)
        assert cst.phase.shape == (3, 3, 3)
        assert cst.amplitude.shape == (3, 3, 3)
        assert np.all(cst.amplitude >= 0)

    def test_global_coherence(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random()
        gcm = cst.global_coherence()
        assert 0 <= gcm <= 10  # Can exceed 1 if amplitudes vary

    def test_phase_entropy(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random()
        H = cst.phase_entropy()
        assert H > 0

    def test_synchrony_index(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random()
        S = cst.synchrony_index()
        assert isinstance(S, float)

    def test_coherence_map(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random(n_regulatory=4, n_temporal=3, n_homeostatic=5)
        cm = cst.coherence_map()
        assert cm.shape == (4, 3)

    def test_plv_map(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random(n_regulatory=4, n_temporal=3, n_homeostatic=5)
        plv = cst.plv_map()
        assert plv.shape == (3, 4, 4)

    def test_ema_update(self):
        from biophasor.cst.tensor import CellStateTensor
        cst1 = CellStateTensor.random(n_regulatory=3, n_temporal=3, n_homeostatic=3, seed=1)
        cst2 = CellStateTensor.random(n_regulatory=3, n_temporal=3, n_homeostatic=3, seed=2)
        cst_s = cst1.ema_update(cst2, lam=0.5)
        assert cst_s.shape == (3, 3, 3)

    def test_from_omics_phases(self):
        from biophasor.cst.tensor import CellStateTensor
        phases = {
            "RNA": np.random.uniform(-np.pi, np.pi, 100),
            "Protein": np.random.uniform(-np.pi, np.pi, 100),
        }
        cst = CellStateTensor.from_omics_phases(phases, n_homeostatic=5)
        assert cst.n_regulatory == 2
        assert cst.n_homeostatic == 5

    def test_to_real_features(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random(n_regulatory=3, n_temporal=3, n_homeostatic=3)
        rf = cst.to_real_features()
        assert rf.shape == (2 * 3 * 3 * 3,)

    def test_attractor_features(self):
        from biophasor.cst.tensor import CellStateTensor
        cst = CellStateTensor.random()
        feats = cst.attractor_features()
        assert "global_coherence" in feats
        assert "phase_entropy" in feats
        assert "synchrony_index" in feats


# ════════════════════════════════════════════════════════════════════════════════
# LimitCycleAnalyzer
# ════════════════════════════════════════════════════════════════════════════════

class TestLimitCycleAnalyzer:

    def test_detect_periodic(self):
        from biophasor.cst.limit_cycles import LimitCycleAnalyzer
        t = np.linspace(0, 100, 2000)
        phase = np.stack([np.sin(0.5 * t), np.cos(0.5 * t)])
        analyzer = LimitCycleAnalyzer(min_period=5, max_period=100)
        cycles = analyzer.detect(phase)
        assert len(cycles) > 0
        assert all(c.period > 0 for c in cycles)

    def test_floquet_stability(self):
        from biophasor.cst.limit_cycles import LimitCycleAnalyzer
        t = np.linspace(0, 100, 2000)
        phase = np.stack([np.sin(0.3 * t), np.cos(0.3 * t)])
        analyzer = LimitCycleAnalyzer()
        cycles = analyzer.detect(phase)
        for c in cycles:
            assert c.max_multiplier >= 0

    def test_synchrony_profile(self):
        from biophasor.cst.limit_cycles import LimitCycleAnalyzer
        phase = np.random.randn(3, 500)
        analyzer = LimitCycleAnalyzer()
        R_t = analyzer.synchrony_profile(phase)
        assert R_t.shape == (500,)
        assert np.all(R_t >= 0) and np.all(R_t <= 1)

    def test_phase_velocity(self):
        from biophasor.cst.limit_cycles import LimitCycleAnalyzer
        phase = np.random.randn(3, 500)
        analyzer = LimitCycleAnalyzer()
        vel = analyzer.phase_velocity(phase)
        assert vel.shape == (3, 499)

    def test_resilience_spectrum(self):
        from biophasor.cst.limit_cycles import LimitCycleAnalyzer, LimitCycle
        cycles = [
            LimitCycle(period=20, center_phase=np.zeros(2), amplitude=0.5,
                       floquet_multipliers=np.array([0.8, 0.9]),
                       winding_numbers=np.array([1.0, 1.0])),
        ]
        analyzer = LimitCycleAnalyzer()
        res = analyzer.resilience_spectrum(cycles)
        assert len(res) == 1
        assert res[0] > 0  # Stable cycle → positive resilience


# ════════════════════════════════════════════════════════════════════════════════
# AttractorGeometry
# ════════════════════════════════════════════════════════════════════════════════

class TestAttractorGeometry:

    def test_fit_and_metrics(self):
        from biophasor.cst.geometry import AttractorGeometry
        np.random.seed(42)
        phase = np.random.uniform(-3, 3, (2, 300))
        geom = AttractorGeometry(n_basins=2, window_size=16)
        geom.fit(phase)
        metrics = geom.basin_metrics()
        assert len(metrics) == 2
        total_occ = sum(m.occupancy for m in metrics)
        assert abs(total_occ - 1.0) < 0.01

    def test_transition_matrix(self):
        from biophasor.cst.geometry import AttractorGeometry
        np.random.seed(42)
        phase = np.random.uniform(-3, 3, (2, 300))
        geom = AttractorGeometry(n_basins=2, window_size=16)
        geom.fit(phase)
        T = geom.transition_matrix()
        assert T.shape == (2, 2)
        assert np.allclose(T.sum(axis=1), 1.0, atol=0.01)

    def test_transition_entropy(self):
        from biophasor.cst.geometry import AttractorGeometry
        np.random.seed(42)
        phase = np.random.uniform(-3, 3, (2, 300))
        geom = AttractorGeometry(n_basins=2, window_size=16)
        geom.fit(phase)
        H = geom.transition_entropy()
        assert H >= 0

    def test_cst_features(self):
        from biophasor.cst.geometry import AttractorGeometry
        np.random.seed(42)
        phase = np.random.uniform(-3, 3, (2, 300))
        geom = AttractorGeometry(n_basins=2, window_size=16)
        geom.fit(phase)
        feats = geom.cst_features()
        assert "n_active_basins" in feats
        assert "dominant_state" in feats
        assert "transition_entropy" in feats


# ════════════════════════════════════════════════════════════════════════════════
# CSTDynamics
# ════════════════════════════════════════════════════════════════════════════════

class TestCSTDynamics:

    def test_simulate(self):
        from biophasor.cst.tensor import CellStateTensor
        from biophasor.cst.dynamics import CSTDynamics
        cst0 = CellStateTensor.random(n_regulatory=3, n_temporal=3, n_homeostatic=3)
        dyn = CSTDynamics(cst0, coupling=1.0, noise=0.03)
        cst_ev = dyn.simulate(n_steps=50, dt=0.01)
        assert cst_ev.n_regulatory == 3
        assert cst_ev.n_temporal == 3

    def test_phase_flip(self):
        from biophasor.cst.tensor import CellStateTensor
        from biophasor.cst.dynamics import CSTDynamics
        cst0 = CellStateTensor.random(n_regulatory=3, n_temporal=3, n_homeostatic=3)
        dyn = CSTDynamics(cst0, coupling=1.0, noise=0.01)
        phase_before = dyn._phase[0].copy()
        dyn.phase_flip(0)
        phase_after = dyn._phase[0]
        # Phase should have shifted by ~π
        diff = np.abs(np.angle(np.exp(1j * (phase_after - phase_before))))
        assert diff.mean() > 2.0  # Should be ~π


# ════════════════════════════════════════════════════════════════════════════════
# AttractorLandscape
# ════════════════════════════════════════════════════════════════════════════════

class TestAttractorLandscape:

    def test_fit_and_classify(self):
        from biophasor.cst.tensor import CellStateTensor
        from biophasor.cst.attractor import AttractorLandscape
        csts = [CellStateTensor.random(n_regulatory=3, n_temporal=3,
                                       n_homeostatic=3, seed=i) for i in range(15)]
        landscape = AttractorLandscape(n_attractors=3, reducer=None)
        landscape.fit(csts)
        label = landscape.nearest_attractor(csts[0])
        assert 0 <= label < 3

    def test_attractor_probability(self):
        from biophasor.cst.tensor import CellStateTensor
        from biophasor.cst.attractor import AttractorLandscape
        csts = [CellStateTensor.random(n_regulatory=3, n_temporal=3,
                                       n_homeostatic=3, seed=i) for i in range(15)]
        landscape = AttractorLandscape(n_attractors=3, reducer=None)
        landscape.fit(csts)
        probs = landscape.attractor_probability(csts[0])
        assert len(probs) == 3
        assert abs(probs.sum() - 1.0) < 1e-6
