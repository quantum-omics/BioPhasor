"""
test_omics_phnn.py  —  test suite for Generic_pHNN

Tests
-----
1. Model instantiation with correct state dimensions
2. Forward pass shape correctness
3. J skew-symmetry (structural guarantee)
4. R positive-definiteness (softplus guarantee)
5. H non-negativity (softplus construction)
6. Passivity: Ḣ|_{u=0} = −∇H^T R ∇H ≤ 0  for PSD R
7. Port energy: G maps 3 port inputs to abundance block only
"""

import torch
import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biophasor.phnn.models.phnn import Generic_pHNN
from biophasor.core.graph.bio_graph import build_biological_graph
from biophasor.core.datagen.omics_data_generator import LAYER_CONFIG


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model_and_graph():
    """Instantiate a small Generic_pHNN with N_rhythmic=10 for fast tests."""
    N_total    = sum(cfg["n_nodes"] for cfg in LAYER_CONFIG.values())  # 100
    N_rhythmic = 10
    model = Generic_pHNN(N_total=N_total, N_rhythmic=N_rhythmic, hidden_dim=32, n_ports=3)
    model.eval()
    bio_graph = build_biological_graph(seed=0)
    state_dim = N_total + 3 * N_rhythmic   # 130
    rhy_idx   = torch.arange(N_rhythmic)   # first 10 nodes are "rhythmic"
    return model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic


@pytest.fixture(scope="module")
def dummy_batch(model_and_graph):
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    B = 4
    x = torch.randn(B, state_dim).requires_grad_(True)
    u = torch.randn(B, 3)
    return x, u


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_model_instantiation(model_and_graph):
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    assert model.N   == N_total
    assert model.N_r == N_rhythmic
    assert model.state_dim == state_dim


def test_forward_output_shapes(model_and_graph):
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    B  = 3
    x  = torch.randn(B, state_dim).requires_grad_(True)
    u  = torch.randn(B, 3)

    with torch.enable_grad():
        dx_pred, H, sub_H, nabla_H = model(x, u, rhy_idx, bio_graph)

    assert dx_pred.shape  == (B, state_dim), f"dx_pred shape {dx_pred.shape}"
    assert H.shape        == (B, 1),          f"H shape {H.shape}"
    assert nabla_H.shape  == (B, state_dim),  f"nabla_H shape {nabla_H.shape}"
    assert "G" in sub_H and "P" in sub_H and "M" in sub_H


def test_J_skew_symmetry(model_and_graph):
    """J(x) + J(x)^T must be zero (skew-symmetric)."""
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    x = torch.randn(2, state_dim).requires_grad_(True)
    u = torch.zeros(2, 3)

    with torch.enable_grad():
        model(x, u, rhy_idx, bio_graph)

    J = model._last_J  # (B, sd, sd)
    skew_err = (J + J.transpose(1, 2)).abs().max().item()
    assert skew_err < 1e-5, f"J not skew-symmetric: max |J+J^T| = {skew_err:.2e}"


def test_R_positive_diagonal(model_and_graph):
    """R(x) diagonal must be non-negative (softplus guarantee)."""
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    x = torch.randn(2, state_dim).requires_grad_(True)
    u = torch.zeros(2, 3)

    with torch.enable_grad():
        model(x, u, rhy_idx, bio_graph)

    R = model._last_R   # (B, sd, sd)
    R_diag = torch.diagonal(R, dim1=-2, dim2=-1)   # (B, sd)
    assert (R_diag >= 0).all(), f"R has negative diagonal entries: min={R_diag.min():.4f}"


def test_H_non_negative(model_and_graph):
    """H(x) must be non-negative for all inputs (softplus readout)."""
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    # Test across many random states
    x = torch.randn(20, state_dim).requires_grad_(True)
    u = torch.zeros(20, 3)

    with torch.enable_grad():
        _, H, _, _ = model(x, u, rhy_idx, bio_graph)

    assert (H >= 0).all(), f"H has negative values: min={H.min():.4f}"


def test_passivity_sign(model_and_graph):
    """Passivity: Ḣ|_{{u=0}} = −∇H^T R ∇H ≤ 0 since R is PSD."""
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    x = torch.randn(4, state_dim).requires_grad_(True)
    u = torch.zeros(4, 3)

    with torch.enable_grad():
        _, H, _, nabla_H = model(x, u, rhy_idx, bio_graph)

    R = model._last_R   # (B, sd, sd)
    Rg      = torch.bmm(R, nabla_H.unsqueeze(-1)).squeeze(-1)   # (B, sd)
    H_dot   = -(nabla_H * Rg).sum(dim=-1)   # (B,)  ≤ 0 if R PSD
    violations = (H_dot > 1e-4).sum().item()
    assert violations == 0, \
        f"{violations}/4 samples violate passivity (Ḣ > 0). max Ḣ = {H_dot.max():.4f}"


def test_port_G_maps_to_abundance_only(model_and_graph):
    """Port matrix G should be non-zero only in the abundance block [0:N]."""
    model, bio_graph, state_dim, rhy_idx, N_total, N_rhythmic = model_and_graph
    G = model.G   # (N_total, 3)
    # G should have shape (N_total, n_ports)
    assert G.shape == (N_total, 3), f"G shape {G.shape}"
    # No structural constraint that G is zero outside N, but the forward only uses
    # G to drive x[:, :N]; check that G is sensibly initialized
    assert G.abs().max() > 0, "G is all zeros — no port input!"


def test_state_dim_consistency():
    """Verify state_dim = N_total + 3 * N_rhythmic for several N_rhythmic values."""
    N_total = sum(cfg["n_nodes"] for cfg in LAYER_CONFIG.values())
    for N_r in [0, 5, 20]:
        model = Generic_pHNN(N_total=N_total, N_rhythmic=N_r, hidden_dim=32)
        assert model.state_dim == N_total + 3 * N_r, \
            f"state_dim mismatch for N_r={N_r}: {model.state_dim} != {N_total + 3*N_r}"
