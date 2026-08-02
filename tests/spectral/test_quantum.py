"""
Algebraic invariants and reference values for biophasor.spectral.quantum.

These lock the numbers the spectral-quantum manuscript reports, so a refactor
that silently changes the Fock-space conventions (operator ordering, the
sector projection, the CCM symmetrisation) fails here rather than in a figure
regenerated months later.

Reference values, from manuscripts/spectral-quantum:
    epsilon      = (6.2192, 1.9766, 1.9158, 1.5905, 1.4092)
    E0_free      = 6.5556
    E0(N=3)      = 3.85811176

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

import numpy as np
import pytest

from biophasor.spectral.quantum import (
    FockSpace,
    OmicsHamiltonian,
    InteractionHamiltonians,
    QuantumDynamics,
    CompartmentModel,
    CompartmentCovariance,
    CompartmentWeights,
    COMPARTMENTS,
    compartment_self_energies,
)

J, U, V = 0.4, 0.3, 0.15


@pytest.fixture(scope="module")
def eps():
    return compartment_self_energies(5)


@pytest.fixture(scope="module")
def model(eps):
    return CompartmentModel(FockSpace(5, 3), epsilon=eps, J=J, U=U, V=V)


@pytest.fixture(scope="module")
def ground(model):
    return model.ground_state(n_total=3)


# ── the classical → quantum bridge ───────────────────────────────────────────

def test_self_energies_match_manuscript(eps):
    assert np.allclose(
        eps, [6.2192, 1.9766, 1.9158, 1.5905, 1.4092], atol=1e-4
    )
    assert eps.size == len(COMPARTMENTS)


def test_free_zero_point_energy(eps):
    H0 = OmicsHamiltonian(eps, FockSpace(5, 3))
    assert float(H0.free_hamiltonian().diagonal().min().real) == pytest.approx(
        6.5556, abs=1e-4
    )


# ── Fock-space algebra ───────────────────────────────────────────────────────

def test_ladder_commutator_inside_truncation():
    # [a, a+] = 1 holds everywhere except the top rung, where the truncation
    # kills the outgoing state — checking the whole space would fail for a
    # correct implementation, so restrict to basis states below the cutoff.
    fs = FockSpace(n_modes=3, max_occupation=3)
    a, ad = fs.annihilation_op(0), fs.creation_op(0)
    keep = [i for i, lab in enumerate(fs.basis_labels()) if lab[0] < 3]
    comm = (a @ ad - ad @ a - np.eye(fs.dim()))[np.ix_(keep, keep)]
    assert np.max(np.abs(comm)) < 1e-12


# ── boundedness and number conservation ──────────────────────────────────────

def test_bose_hubbard_terms_conserve_number():
    fs = FockSpace(5, 3)
    inter = InteractionHamiltonians(fs)
    N = sum(fs.number_op(k) for k in range(5))
    g = np.zeros((5, 5))
    g[0, 2] = g[2, 0] = J
    for H in (inter.coherent_hopping(g), inter.onsite_kerr(np.full(5, U))):
        assert np.max(np.abs(H - H.conj().T)) < 1e-12
        assert np.max(np.abs(H @ N - N @ H)) < 1e-10


def test_cubic_reference_does_not_conserve_number():
    # The unbounded odd-order term is kept precisely so this contrast is
    # demonstrable; if it ever starts commuting with N, it is not the
    # pathological reference the manuscript argues against.
    fs = FockSpace(3, 3)
    inter = InteractionHamiltonians(fs)
    N = sum(fs.number_op(k) for k in range(3))
    H = inter.cubic_reference(np.full((3, 3, 3), 0.1))
    assert np.max(np.abs(H @ N - N @ H)) > 1e-6


def test_ground_state_energy_and_occupations(model, ground):
    E0, psi = ground
    assert E0 == pytest.approx(3.85811176, abs=1e-8)
    assert model.commutes_with_number() < 1e-10
    occ = np.array([float(np.real(psi.conj() @ model._num[k] @ psi))
                    for k in range(5)])
    assert occ.sum() == pytest.approx(3.0, abs=1e-9)
    # Clock carries the largest self-energy and is therefore nearly empty;
    # excitation concentrates on the lowest-energy mode.
    assert occ[0] < 0.01
    assert np.argmax(occ) == COMPARTMENTS.index("Biosynthesis")


def test_ground_state_is_cutoff_independent(eps, ground):
    E0, _ = ground
    E0_hi, _ = CompartmentModel(
        FockSpace(5, 4), epsilon=eps, J=J, U=U, V=V
    ).ground_state(n_total=3)
    assert E0_hi == pytest.approx(E0, abs=1e-9)


# ── CCM readout ──────────────────────────────────────────────────────────────

def test_ccm_is_symmetric_psd_and_kappa_bounded(model, ground):
    _, psi = ground
    M = model.compute_ccm(psi)
    assert np.max(np.abs(M - M.T)) < 1e-12
    assert np.linalg.eigvalsh(M).min() > -1e-9
    cov = CompartmentCovariance(psi)
    kappa = cov.coherence(M)
    assert 0.0 <= kappa <= 1.0
    w = CompartmentWeights(cov).compartment_weights(M)
    assert set(w) == set(COMPARTMENTS)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-9)
    assert min(w.values()) >= 0.0


# ── dynamics ─────────────────────────────────────────────────────────────────

def test_evolution_is_unitary_and_entropy_non_negative(model):
    dyn = QuantumDynamics(model.total_hamiltonian())
    psi0 = model.fock.fock_state([0, 0, 1, 1, 1])
    traj = dyn.evolve_trajectory(psi0, np.linspace(0.0, 2.0, 5))
    assert np.allclose([np.linalg.norm(s) for s in traj], 1.0, atol=1e-10)
    S = dyn.entanglement_entropy(traj[-1], 16)
    assert S >= -1e-12
