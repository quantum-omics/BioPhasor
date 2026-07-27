"""
Tests for the non-zero-flux magnetic OCM variant (connectome.magnetic).

Locks in: Hermiticity, exact antisymmetry of the orientation matrices, and the
key property that distinguishes the magnetic operator from the gradient-phase
OCM — the spectrum DEPENDS on the cycle flux (charge parameter q). A gradient
phase has zero cycle flux and cannot reproduce a non-zero-flux spectrum.
"""

import numpy as np
import pytest

import biophasor.spectral as so
from biophasor.spectral.connectome.magnetic import (
    build_magnetic,
    lead_lag_antisymmetry,
    signed_antisymmetry,
    cycle_flux,
)


@pytest.fixture
def triangle():
    """3-node directed cycle: unit weights, antisymmetric A around the loop."""
    w = np.ones((3, 3))
    np.fill_diagonal(w, 0.0)
    A = np.array([[0.0, 1.0, -1.0],
                  [-1.0, 0.0, 1.0],
                  [1.0, -1.0, 0.0]])
    return w, A


# ── Hermiticity ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 1.0 / 6.0, 0.5])
def test_magnetic_hermitian(triangle, q):
    w, A = triangle
    H = build_magnetic(w, A, q=q)
    assert np.linalg.norm(H - H.conj().T) < 1e-12


def test_magnetic_hermitian_amplitude_weighted(triangle):
    w, A = triangle
    r = np.array([0.4, 0.9, 0.7])
    H = build_magnetic(w, A, q=0.2, amplitude=r)
    assert np.linalg.norm(H - H.conj().T) < 1e-12


def test_magnetic_real_spectrum(triangle):
    w, A = triangle
    vals = np.linalg.eigvals(build_magnetic(w, A, q=0.13))
    assert np.max(np.abs(vals.imag)) < 1e-10


# ── antisymmetry of the orientation estimators ───────────────────────────────

def test_lead_lag_antisymmetric():
    rng = np.random.default_rng(1)
    S, N = 40, 8
    t = np.linspace(0, 4 * np.pi, S)
    Xt = np.column_stack([np.sin(t - 0.3 * i) + rng.normal(0, 0.1, S) for i in range(N)])
    A = lead_lag_antisymmetry(Xt, max_lag=8)
    assert np.linalg.norm(A + A.T) < 1e-12
    assert np.allclose(np.diag(A), 0.0)


def test_signed_antisymmetric():
    rng = np.random.default_rng(2)
    S, N = 60, 6
    t = np.linspace(0, 6 * np.pi, S)
    Xt = np.column_stack([np.sin(t - 0.5 * i) + rng.normal(0, 0.1, S) for i in range(N)])
    A = signed_antisymmetry(Xt)
    assert np.linalg.norm(A + A.T) < 1e-12


# ── the central property: spectrum depends on flux ───────────────────────────

def test_spectrum_depends_on_flux(triangle):
    """Non-zero flux (q>0) changes the spectrum vs the zero-flux case (q=0).

    At q=0 the operator is the real coupling matrix (spectrum of C); a genuine
    magnetic flux must move the eigenvalues. A pure gradient phase could not.
    """
    w, A = triangle
    sp0 = np.sort(np.linalg.eigvalsh(build_magnetic(w, A, q=0.0)))
    sp1 = np.sort(np.linalg.eigvalsh(build_magnetic(w, A, q=1.0 / 6.0)))  # flux = pi
    assert np.max(np.abs(sp0 - sp1)) > 1e-3


def test_zero_flux_equals_real_coupling(triangle):
    """q=0 ⇒ zero flux ⇒ H is exactly the real symmetric coupling."""
    w, A = triangle
    H = build_magnetic(w, A, q=0.0)
    assert np.max(np.abs(H.imag)) < 1e-12
    assert np.allclose(H.real, 0.5 * (w + w.T))


def test_nonzero_flux_triangle_not_reproducible_by_gradient(triangle):
    """A non-zero-flux triangle spectrum is unreachable by any gradient phase.

    Any gradient-phase (diag(u) C diag(u)†) operator is a unitary congruence of
    C, hence isospectral with C for EVERY choice of vertex phases u. So if the
    magnetic spectrum differs from spec(C), no gradient phase can reproduce it.
    """
    w, A = triangle
    C = 0.5 * (w + w.T)
    spec_C = np.sort(np.linalg.eigvalsh(C))
    spec_mag = np.sort(np.linalg.eigvalsh(build_magnetic(w, A, q=1.0 / 6.0)))
    assert np.max(np.abs(spec_C - spec_mag)) > 1e-3


def test_cycle_flux_matches_phase_holonomy(triangle):
    """cycle_flux() equals the summed edge phases of H around the triangle."""
    w, A = triangle
    q = 0.11
    H = build_magnetic(w, A, q=q)
    ph = np.angle(H)
    holonomy = ph[0, 1] + ph[1, 2] + ph[2, 0]
    assert abs(cycle_flux(A, q, 0, 1, 2) - holonomy) < 1e-9


def test_flux_gauge_invariance(triangle):
    """Vertex gauge u_i does not change the magnetic spectrum (only flux does)."""
    w, A = triangle
    q = 0.2
    H = build_magnetic(w, A, q=q)
    rng = np.random.default_rng(7)
    u = np.exp(1j * rng.uniform(-np.pi, np.pi, H.shape[0]))
    Hg = np.outer(u, np.conj(u)) * H         # gauge transform (does not touch flux)
    sp = np.sort(np.linalg.eigvalsh(H))
    spg = np.sort(np.linalg.eigvalsh(0.5 * (Hg + Hg.conj().T)))
    assert np.max(np.abs(sp - spg)) < 1e-9


def test_class_method_matches_function(triangle):
    """OmicsConnectomeMatrix.build_magnetic delegates to the module function."""
    w, A = triangle
    ocm = so.OmicsConnectomeMatrix()
    H_method = ocm.build_magnetic(w, A, q=0.15)
    H_func = build_magnetic(w, A, q=0.15)
    assert np.allclose(H_method, H_func)
