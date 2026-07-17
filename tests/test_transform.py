"""
Tests for biophasor.transform — small, fast samples.
"""

import numpy as np
import pytest
from biophasor.transform.encoder import tanh_phase_encode, log_linear_encode, linear_encode, OmicsPhasorEncoder
from biophasor.transform.phasor_transform import BPT


RNG = np.random.RandomState(0)
N, F = 25, 30   # small!


class TestEncoders:
    def setup_method(self):
        self.X = RNG.lognormal(0, 2, (N, F))

    def test_tanh_range(self):
        phi = tanh_phase_encode(self.X)
        assert phi.shape == (N, F)
        assert phi.min() >= -np.pi - 1e-9 and phi.max() <= np.pi + 1e-9

    def test_tanh_higher_spread_than_linear(self):
        """Notebook 1.1 finding: tanh std > linear std."""
        assert tanh_phase_encode(self.X).std() > linear_encode(self.X).std()

    def test_log_linear_range(self):
        phi = log_linear_encode(self.X)
        assert phi.min() >= -np.pi - 1e-9 and phi.max() <= np.pi + 1e-9

    def test_linear_range(self):
        phi = linear_encode(self.X)
        assert phi.min() >= -np.pi - 1e-9 and phi.max() <= np.pi + 1e-9

    def test_tanh_no_log_transform(self):
        phi = tanh_phase_encode(RNG.uniform(0, 1, (N, F)), log_transform=False)
        assert phi.shape == (N, F)

    def test_omics_encoder_defaults(self):
        enc = OmicsPhasorEncoder(modality="RNA")
        assert enc.strategy == "tanh"
        assert enc.encode(self.X).shape == (N, F)

    def test_omics_encoder_methylation_strategy(self):
        assert OmicsPhasorEncoder(modality="methylation").strategy == "log_linear"

    def test_omics_encoder_multiomics_concat(self):
        X2 = RNG.lognormal(0, 1, (N, 10))
        enc = OmicsPhasorEncoder(modality="RNA", strategy="tanh")
        out = enc.encode_multiomics({"RNA": self.X, "ATAC": X2}, concat=True)
        assert out.shape == (N, F + 10)


class TestBPT:
    def setup_method(self):
        # 24 timepoints (1 circadian cycle at 1h resolution), 20 genes
        self.X_ts = np.abs(RNG.randn(24, 20)) + 0.1

    def test_gs_shape(self):
        G, S = BPT(n_harmonics=2).fit_transform(self.X_ts)
        assert G.shape == (2, 20) and S.shape == (2, 20)

    def test_semicircle_constraint(self):
        G, S = BPT(n_harmonics=1).fit_transform(self.X_ts, normalize=True)
        assert BPT.semicircle_constraint(G[0], S[0]).mean() > 0.9

    def test_phase_range(self):
        phi, A = BPT(n_harmonics=1).to_phase_amplitude(self.X_ts)
        assert phi.min() >= -np.pi - 1e-9 and phi.max() <= np.pi + 1e-9
        assert A.min() >= 0.0
