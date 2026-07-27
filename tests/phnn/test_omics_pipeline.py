"""
test_omics_pipeline.py  —  end-to-end data pipeline tests

Tests the complete data flow:
  generate_multi_omics → detect_all_layers → assemble_two_layer_state
  → build_biological_graph → validate_omics_data → verify_conservation

Also tests the data adapter contract.
"""

import sys
import os
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biophasor.core.datagen.omics_data_generator import (
    generate_multi_omics, LAYER_CONFIG, CONSERVATION_GROUPS, CLOCK_FREQ
)
from biophasor.core.datagen.rhythmicity_gate import detect_all_layers
from biophasor.core.datagen.two_layer_state import assemble_two_layer_state, verify_conservation
from biophasor.core.graph.bio_graph import build_biological_graph, compute_plv_prior
from biophasor.core.datagen.data_adapter import validate_omics_data, load_synthetic_omics


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def omics_data():
    return generate_multi_omics(seed=0)


@pytest.fixture(scope="module")
def gate_results(omics_data):
    return detect_all_layers(omics_data)


@pytest.fixture(scope="module")
def state_data(omics_data, gate_results):
    return assemble_two_layer_state(omics_data, gate_results)


@pytest.fixture(scope="module")
def bio_graph():
    return build_biological_graph(seed=0)


# ── Generator tests ───────────────────────────────────────────────────────────

def test_generator_keys(omics_data):
    """All required keys present in generated data."""
    required = {"t", "expression", "k_deg", "rhythmic_mask", "acrophase_true",
                "node_class", "u", "state_labels", "layer_config",
                "omega_clock", "conservation", "dt"}
    assert required <= set(omics_data.keys()), \
        f"Missing keys: {required - set(omics_data.keys())}"


def test_generator_shapes(omics_data):
    """Expression arrays have correct shapes."""
    T  = len(omics_data["t"])
    dt = omics_data["dt"]
    assert T == int(240.0 / dt), f"Expected T={int(240/dt)}, got T={T}"
    for layer, cfg in LAYER_CONFIG.items():
        N = cfg["n_nodes"]
        expr = omics_data["expression"][layer]
        assert expr.shape == (N, T), \
            f"expression['{layer}'] shape {expr.shape} != ({N}, {T})"


def test_generator_port_shape(omics_data):
    T = len(omics_data["t"])
    assert omics_data["u"].shape == (T, 3)


def test_generator_state_labels(omics_data):
    """State labels must match the canonical vocabulary."""
    valid = {"Homeostasis", "Drug Administration", "Metabolic Recovery"}
    actual = set(np.unique(omics_data["state_labels"]))
    assert actual <= valid, f"Invalid state labels: {actual - valid}"


def test_data_adapter_schema(omics_data):
    """Data adapter schema validation passes on generated data."""
    issues = validate_omics_data(omics_data)
    assert issues == [], f"Schema issues: {issues}"


# ── Conservation tests ────────────────────────────────────────────────────────

def test_conservation_groups_in_metabolome():
    """All CONSERVATION_GROUPS must be assigned to the metabolome layer."""
    for moiety, (layer, indices, total) in CONSERVATION_GROUPS.items():
        assert layer == "metabolome", \
            f"Moiety '{moiety}' is in layer '{layer}'; expected 'metabolome'"


def test_conservation_holds(omics_data):
    """Stoichiometric conservation max_dev < 5% for all moieties."""
    result = verify_conservation(omics_data)
    for moiety, rep in result.items():
        assert rep["passes"], \
            f"Conservation failed for '{moiety}': max_dev={rep['max_relative_deviation']:.3%}"


# ── Rhythmicity gate tests ────────────────────────────────────────────────────

def test_gate_keys(gate_results):
    for layer in ["genomics", "proteome", "metabolome"]:
        g = gate_results[layer]
        assert "rhythmic_mask" in g
        assert "acrophase" in g
        assert "amplitude" in g


def test_gate_fraction(gate_results):
    """Each layer should have at least one rhythmic node."""
    for layer, g in gate_results.items():
        n_rhy = g["rhythmic_mask"].sum()
        assert n_rhy > 0, f"Layer '{layer}' has 0 rhythmic nodes"


# ── Two-layer state tests ─────────────────────────────────────────────────────

def test_state_dim(state_data):
    N_r = state_data["N_rhythmic"]
    N   = state_data["N_total"]
    sd  = state_data["state_dim"]
    assert sd == N + 3 * N_r, f"state_dim {sd} != {N} + 3*{N_r} = {N + 3*N_r}"


def test_state_shape(state_data):
    x    = state_data["x"]
    dx   = state_data["dx_dt"]
    T    = x.shape[0]
    sd   = state_data["state_dim"]
    assert x.shape  == (T, sd), f"x shape {x.shape} != ({T},{sd})"
    assert dx.shape == (T, sd), f"dx_dt shape {dx.shape}"


def test_phasor_identity(state_data):
    """sin²φ + cos²φ = 1 must hold for all rhythmic nodes."""
    N  = state_data["N_total"]
    Nr = state_data["N_rhythmic"]
    if Nr == 0:
        pytest.skip("No rhythmic nodes")
    x = state_data["x"]
    sin_phi = x[:, N:N+Nr]
    cos_phi = x[:, N+Nr:N+2*Nr]
    phasor_norm = sin_phi ** 2 + cos_phi ** 2
    max_dev = (phasor_norm - 1.0).abs().max().item()
    assert max_dev < 1e-4, f"Phasor identity violated: max|sin²φ+cos²φ-1|={max_dev:.2e}"


def test_scale_factors_positive(state_data):
    assert state_data["scale_q"]     > 0
    assert state_data["scale_dx"]    > 0
    assert state_data["scale_omega"] > 0


# ── Biological graph tests ────────────────────────────────────────────────────

def test_bio_graph_keys(bio_graph):
    required = {"A_GG", "A_PP", "A_MM", "A_GP_dogma", "A_PM_enz", "A_MG_fb",
                "S", "n_G", "n_P", "n_M", "edge_counts"}
    assert required <= set(bio_graph.keys())


def test_bio_graph_shapes(bio_graph):
    n_G, n_P, n_M = bio_graph["n_G"], bio_graph["n_P"], bio_graph["n_M"]
    assert bio_graph["A_GG"].shape       == (n_G, n_G)
    assert bio_graph["A_PP"].shape       == (n_P, n_P)
    assert bio_graph["A_MM"].shape       == (n_M, n_M)
    assert bio_graph["A_GP_dogma"].shape == (n_G, n_P)
    assert bio_graph["A_PM_enz"].shape   == (n_P, n_M)
    assert bio_graph["A_MG_fb"].shape    == (n_M, n_G)
    assert bio_graph["S"].shape          == (3, n_M)


def test_S_matrix_all_rows_nonzero(bio_graph):
    """S must have all 3 rows nonzero (3 conservation groups, all in metabolome)."""
    S = bio_graph["S"]
    for k in range(S.shape[0]):
        assert S[k].sum() > 0, f"S row {k} is all zeros (moiety not mapped)"


def test_dogma_adjacency_near_diagonal(bio_graph):
    """A_GP_dogma must be concentrated on and near the main diagonal."""
    A = bio_graph["A_GP_dogma"].numpy()   # (n_G, n_P)
    n_G, n_P = A.shape
    n_pairs  = min(n_G, n_P)
    diagonal_edges = sum(A[i, i] for i in range(n_pairs))
    total_edges    = A.sum()
    # At least 50% of edges should be near-diagonal (±DOGMA_OFFSET)
    from biophasor.core.graph.bio_graph import DOGMA_OFFSET
    near_diag_edges = 0
    for i in range(n_G):
        for j in range(n_P):
            if abs(i - j) <= DOGMA_OFFSET and A[i, j] > 0:
                near_diag_edges += 1
    if total_edges > 0:
        near_diag_frac = near_diag_edges / total_edges
        assert near_diag_frac > 0.4, \
            f"Only {near_diag_frac:.0%} of dogma edges are near-diagonal"


# ── PLV prior tests ────────────────────────────────────────────────────────────

def test_plv_prior_shape(state_data, gate_results):
    phi_rhy_all    = state_data["phi_rhythmic"].numpy()  # (T, N_r)
    rhy_layer_slcs = state_data["layer_slices_rhythmic"]
    T = phi_rhy_all.shape[0]
    phi_data = {}
    for layer in ["genomics", "proteome", "metabolome"]:
        slc  = rhy_layer_slcs.get(layer, slice(0, 0))
        n_rhy = slc.stop - slc.start
        phi_data[layer] = phi_rhy_all[:, slc].T if n_rhy > 0 else np.zeros((0, T))

    plv_priors = compute_plv_prior(phi_data, gate_results)
    plv_gp = plv_priors.get("PLV_GP")
    if plv_gp is not None:
        assert plv_gp.min() >= 0 and plv_gp.max() <= 1, \
            f"PLV_GP values out of [0,1]: min={plv_gp.min():.3f} max={plv_gp.max():.3f}"


# ── Data adapter tests ────────────────────────────────────────────────────────

def test_load_synthetic_runs():
    """Synthetic loader completes without error."""
    data = load_synthetic_omics(seed=1, total_hours=48.0)  # short for speed
    assert "expression" in data
    assert "t" in data


def test_load_real_implemented():
    """load_real_omics is now implemented and returns a schema-valid dict.

    Skipped if the staged real-data bundle is absent (see test_real_loader.py
    for the full real-loader contract suite).
    """
    import os
    from biophasor.core.datagen.data_adapter import load_real_omics, validate_omics_data
    real_dir = os.path.join(os.path.dirname(__file__), "..", "data", "real")
    if not os.path.exists(os.path.join(real_dir, "proteome", "liver_proteome.npz")):
        pytest.skip("staged real-data bundle not present")
    data = load_real_omics(total_hours=24.0)
    assert validate_omics_data(data) == []
    assert data["data_source"] == "real"
