"""
Test suite for the spectral-omics pipeline invariants (theory.md §6).

Locks in: Hermiticity, real spectrum, orthonormality, gauge invariance,
spectral completeness, CCM Hermiticity, CompartmentWeights bounds, state_class coverage.
"""

import numpy as np
import pytest

import biophasor.spectral as so


@pytest.fixture
def synthetic_omics():
    rng = np.random.default_rng(42)
    S, N = 30, 45
    t = np.linspace(0, 4 * np.pi, S)
    X = np.zeros((S, N))
    for i in range(N):
        phase = (i % 3) * 2 * np.pi / 3
        X[:, i] = 5 + 3 * np.sin(t + phase) + rng.normal(0, 0.3, S)
    return np.clip(X, 0, None)


@pytest.fixture
def pipeline(synthetic_omics):
    X = synthetic_omics
    enc = so.PhasorEncoder(amplitude_mode="expression")
    Psi = enc.encode(X)
    psi = Psi[0]
    ocm = so.OmicsConnectomeMatrix(coupling_mode="pearson")
    C = ocm.compute_coupling(X)
    H = ocm.build(psi, coupling=C)
    har = so.OmicsHarmonics()
    vals, vecs = har.decompose(H)
    ccm = so.CompartmentCouplingMatrix()
    M = ccm.build(H, eigenvectors=vecs)
    return dict(X=X, Psi=Psi, psi=psi, C=C, H=H, vals=vals, vecs=vecs, M=M)


# ── phasor encoding ──────────────────────────────────────────────────────────

def test_phase_range(synthetic_omics):
    theta = so.connectome.tanh_phase_encode(synthetic_omics)
    assert theta.min() >= -np.pi - 1e-9
    assert theta.max() <= np.pi + 1e-9


def test_amplitude_nonnegative(synthetic_omics):
    enc = so.PhasorEncoder(amplitude_mode="expression")
    r = enc.compute_amplitude(synthetic_omics)
    assert (r >= 0).all() and (r <= 1 + 1e-9).all()


def test_phase_coherence_bounds(pipeline):
    R = so.connectome.phase_coherence(np.angle(pipeline["psi"]))
    assert 0.0 <= R <= 1.0


# ── OCM Hermiticity ──────────────────────────────────────────────────────────

def test_ocm_hermitian(pipeline):
    H = pipeline["H"]
    assert np.linalg.norm(H - H.conj().T) < 1e-10


def test_ocm_real_diagonal(pipeline):
    H = pipeline["H"]
    assert np.max(np.abs(np.imag(np.diag(H)))) < 1e-12


@pytest.mark.parametrize("mode", ["pearson", "coexpression", "uniform"])
def test_ocm_modes_hermitian(synthetic_omics, mode):
    enc = so.PhasorEncoder()
    psi = enc.encode(synthetic_omics)[0]
    ocm = so.OmicsConnectomeMatrix(coupling_mode=mode)
    H = ocm.build(psi, X=synthetic_omics)
    assert np.linalg.norm(H - H.conj().T) < 1e-10


# ── harmonics ────────────────────────────────────────────────────────────────

def test_real_spectrum(pipeline):
    vals = pipeline["vals"]
    assert np.max(np.abs(np.imag(vals.astype(complex)))) < 1e-10


def test_eigenvectors_orthonormal(pipeline):
    V = pipeline["vecs"]
    G = V.conj().T @ V
    assert np.linalg.norm(G - np.eye(G.shape[0])) < 1e-8


def test_eigenvalues_descending(pipeline):
    vals = pipeline["vals"]
    assert np.all(np.diff(vals) <= 1e-9)


def test_spectral_completeness(pipeline):
    H, vals = pipeline["H"], pipeline["vals"]
    assert abs(np.trace(H).real - vals.sum()) < 1e-8


# ── gauge invariance ─────────────────────────────────────────────────────────

def test_gauge_invariance(pipeline):
    """Global phase shift ψ→ψe^{iα} leaves the OCM spectrum invariant (theory.md §6)."""
    C, psi = pipeline["C"], pipeline["psi"]
    ocm = so.OmicsConnectomeMatrix()
    base = np.sort(np.linalg.eigvalsh(ocm.build(psi, coupling=C)))
    for alpha in [0.3, 1.7, -2.1]:
        shifted = np.sort(np.linalg.eigvalsh(ocm.build(psi * np.exp(1j * alpha), coupling=C)))
        assert np.max(np.abs(base - shifted)) < 1e-8


def test_amplitude_weighting_makes_spectrum_sample_specific(synthetic_omics):
    """Amplitude-weighted OCM spectra differ across samples; phase-only do not."""
    enc = so.PhasorEncoder(amplitude_mode="expression")
    Psi = enc.encode(synthetic_omics)
    ocm = so.OmicsConnectomeMatrix(coupling_mode="pearson")
    C = ocm.compute_coupling(synthetic_omics)
    # phase-only: spectrum identical across slices (unitary similarity of C)
    sp0 = np.sort(np.linalg.eigvalsh(ocm.build(Psi[0], coupling=C, amplitude_weighted=False)))
    sp1 = np.sort(np.linalg.eigvalsh(ocm.build(Psi[1], coupling=C, amplitude_weighted=False)))
    assert np.max(np.abs(sp0 - sp1)) < 1e-8
    # amplitude-weighted: spectrum varies across slices (congruence)
    aw0 = np.sort(np.linalg.eigvalsh(ocm.build(Psi[0], coupling=C)))
    aw1 = np.sort(np.linalg.eigvalsh(ocm.build(Psi[1], coupling=C)))
    assert np.max(np.abs(aw0 - aw1)) > 1e-6


# ── CCM + CompartmentWeights ──────────────────────────────────────────────────────────

def test_ost_hermitian(pipeline):
    M = pipeline["M"]
    assert M.shape == (5, 5)
    assert np.linalg.norm(M - M.conj().T) < 1e-10


def test_covariance_psd(pipeline):
    G = so.CompartmentCouplingMatrix.covariance_form(pipeline["M"])
    assert np.min(np.linalg.eigvalsh(0.5 * (G + G.conj().T))) > -1e-8


def test_compartment_weights_normalised(pipeline):
    readout = so.CompartmentWeights().analyze(pipeline["M"])
    w = readout["weight_vector"]
    assert abs(w.sum() - 1.0) < 1e-8
    assert (w >= -1e-12).all()
    assert 0.0 <= readout["coherence_kappa"] <= 1.0


# ── state_class coverage ────────────────────────────────────────────────────────

@pytest.mark.parametrize("R,H,gap,kappa,pmax", [
    (0.9, 0.2, 5.0, 0.3, 0.3),   # I healthy
    (0.9, 0.5, 5.0, 0.8, 0.3),   # III hyper-coupled
    (0.2, 0.9, 1.0, 0.5, 0.3),   # IV desync
    (0.5, 0.9, 1.0, 0.2, 0.3),   # V fragmented
    (0.5, 0.5, 1.0, 0.5, 0.6),   # VI imbalanced
    (0.5, 0.5, 1.0, 0.5, 0.3),   # II balanced
])
def test_state_class_returns_valid_class(R, H, gap, kappa, pmax):
    cls = so.SpectralStateClassifier().classify(R, H, gap, kappa, pmax)
    assert cls["class"] in {"I", "II", "III", "IV", "V", "VI", "VII"}
    assert cls["label"] and cls["recommended_intervention"]


# ── full consistency suite ───────────────────────────────────────────────────

def test_full_consistency_suite(pipeline):
    cs = so.ConsistencySuite()
    res = cs.run(pipeline["H"], pipeline["vals"], pipeline["vecs"],
                 pipeline["M"], pipeline["psi"], pipeline["C"])
    assert cs.all_passed(res), {k: r for k, (p, r) in res.items() if not p}


# ── state-record round-trip ──────────────────────────────────────────────────

def test_state_record_roundtrip(pipeline, tmp_path):
    ind = so.SpectralIndicators()
    panel = ind.compute(pipeline["vals"], pipeline["vecs"], pipeline["psi"], coupling=pipeline["C"])
    readout = so.CompartmentWeights().analyze(pipeline["M"])
    cls = so.SpectralStateClassifier().classify(panel["coherence_R"], panel["spectral_entropy"],
                                      panel["fiedler_gap"], readout["coherence_kappa"],
                                      max(readout["weight_vector"]))
    rec = so.SpectralStateRecord.from_pipeline("t0", panel, pipeline["vals"], pipeline["M"], readout, cls)
    p = tmp_path / "rec.json"
    rec.save(str(p))
    rec2 = so.SpectralStateRecord.load(str(p))
    assert rec2.id == "t0"
    assert len(rec2.ccm_matrix) == 5
