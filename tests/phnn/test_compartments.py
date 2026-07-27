"""
test_compartments.py  —  Phase A regression tests for the compartmental
                          multi-clock scaffold.
"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biophasor.core.datagen.compartments import (
    build_compartments, concat_compartment_arrays,
    CLOCK_BANK, CLOCK_NAMES, N_COMPARTMENTS, COMPARTMENT_CONFIG,
)
from biophasor.core.datagen.omics_data_generator import generate_multi_omics, LAYER_CONFIG
from biophasor.core.graph.bio_graph import build_biological_graph, build_compartment_structure
from biophasor.core.datagen.data_adapter import validate_omics_data


LAYER_SIZES = {name: cfg["n_nodes"] for name, cfg in LAYER_CONFIG.items()}


# ── Compartment partition ─────────────────────────────────────────────────────

def test_two_clocks_in_bank():
    assert set(CLOCK_NAMES) == {"circadian", "redox"}
    assert CLOCK_BANK["circadian"] != CLOCK_BANK["redox"]   # distinct frequencies


def test_partition_complete_cover():
    """Every node assigned to exactly one compartment (build raises otherwise)."""
    built = build_compartments(LAYER_SIZES)
    for L, n in LAYER_SIZES.items():
        assert (built["comp_id"][L] != 0).all(), f"{L} has unassigned nodes"
        assert built["comp_id"][L].shape == (n,)


def test_every_layer_has_rhythmic_node():
    built = build_compartments(LAYER_SIZES)
    for L in LAYER_SIZES:
        assert built["rhythmic"][L].sum() > 0, f"{L} has 0 rhythmic nodes"


def test_rhythmic_iff_has_clock():
    built = build_compartments(LAYER_SIZES)
    for L in LAYER_SIZES:
        rhy = built["rhythmic"][L]
        has = built["clock_label"][L] != "none"
        assert np.array_equal(rhy, has)


# ── Generator two-clock output ────────────────────────────────────────────────

def test_generator_emits_clock_bank():
    d = generate_multi_omics(seed=0)
    for k in ("clock_bank", "compartment", "clock_label",
              "comp_id_global", "clock_label_global", "omega_node"):
        assert k in d, f"generator missing key '{k}'"
    assert set(d["clock_bank"]) == {"circadian", "redox"}


def test_generator_clock_labels_consistent():
    d = generate_multi_omics(seed=0)
    N = sum(e.shape[0] for e in d["expression"].values())
    assert len(d["comp_id_global"]) == N
    assert len(d["omega_node"]) == N
    # omega_node is the clock freq for rhythmic nodes, 0 otherwise
    for i, c in enumerate(d["clock_label_global"]):
        if c == "none":
            assert d["omega_node"][i] == 0.0
        else:
            assert np.isclose(d["omega_node"][i], d["clock_bank"][c])


def test_clocks_separable_by_period():
    """Circadian nodes peak near 24 h; redox nodes near 20 h (PoC separability)."""
    from scipy.signal import lombscargle
    d = generate_multi_omics(seed=0)
    t = d["t"]
    def dom_period(sig):
        freqs = np.linspace(1/40, 1/10, 600); w = 2*np.pi*freqs
        s = sig - sig.mean()
        pg = lombscargle(t, s, w, normalize=True)
        return 1.0/freqs[pg.argmax()]
    # proteome[0] is circadian (c1); proteome[12] is redox (c2)
    assert abs(dom_period(d["expression"]["proteome"][0])  - 24.0) < 2.0
    assert abs(dom_period(d["expression"]["proteome"][12]) - 20.0) < 2.0


def test_schema_accepts_compartment_keys():
    d = generate_multi_omics(seed=0)
    assert validate_omics_data(d) == []


# ── Bio-graph compartment structure ───────────────────────────────────────────

def test_compartment_masks_partition_offdiagonal():
    g = build_biological_graph(seed=0)
    N = g["n_G"] + g["n_P"] + g["n_M"]
    Mi, Mo = g["M_intra"], g["M_inter"]
    assert int(Mi.sum() + Mo.sum()) == N*N - N        # exact off-diagonal partition
    assert bool((Mi == Mi.T).all())                    # intra symmetric
    assert int((Mi * Mo).sum()) == 0                   # disjoint supports


def test_clock_couple_subset_of_inter():
    g = build_biological_graph(seed=0)
    Mo, Mc = g["M_inter"], g["M_clock_couple"]
    # every clock-couple edge is an inter-compartment edge
    assert int(((Mc > 0) & (Mo == 0)).sum()) == 0
    assert int(Mc.sum()) > 0                            # circadian↔redox coupling exists


def test_n_compartments():
    g = build_biological_graph(seed=0)
    assert g["n_compartments"] == N_COMPARTMENTS == 5


# ── Phase B: composite model (clock port + per-compartment energy) ────────────

@pytest.fixture(scope="module")
def model_ctx():
    import torch
    from biophasor.core.datagen.rhythmicity_gate import detect_all_layers
    from biophasor.core.datagen.two_layer_state import assemble_two_layer_state
    from biophasor.phnn.models.phnn import Generic_pHNN
    d = generate_multi_omics(seed=0)
    gate = detect_all_layers(d)
    sd = assemble_two_layer_state(d, gate)
    bg = build_biological_graph(seed=0)
    kdeg = torch.tensor(
        np.concatenate([d["k_deg"][l] for l in ["genomics", "proteome", "metabolome"]]),
        dtype=torch.float32)
    m = Generic_pHNN(N_total=sd["N_total"], N_rhythmic=sd["N_rhythmic"],
                     hidden_dim=64, n_ports=3, k_deg_prior=kdeg)
    x = sd["x"][10:14].clone().requires_grad_(True)
    u = torch.zeros(4, 3)
    rhy = torch.tensor(sd["rhythmic_indices"], dtype=torch.long)
    dx, H, subH, gH = m(x, u, rhy, bg)
    return dict(m=m, x=x, gH=gH, H=H, subH=subH, bg=bg, torch=torch)


def test_clock_coupling_port_present(model_ctx):
    pm = model_ctx["m"].port_net(model_ctx["x"], model_ctx["bg"])
    assert "Gamma_clock" in pm
    assert int((pm["Gamma_clock"].abs() > 0).sum()) > 0   # circadian↔redox coupling active


def test_clock_coupling_zero_net_power(model_ctx):
    """The clock-coupling port must transmit zero net power (skew construction)."""
    torch = model_ctx["torch"]
    m, x, gH, bg = model_ctx["m"], model_ctx["x"], model_ctx["gH"], model_ctx["bg"]
    N = m.n_G + m.n_P + m.n_M
    pm = m.port_net(x, bg)
    Gck = pm["Gamma_clock"]
    Gck_skew = Gck - Gck.transpose(1, 2)
    gH_q = gH[:, :N]
    power = torch.einsum("bi,bij,bj->b", gH_q, Gck_skew, gH_q).detach()
    assert float(power.abs().max()) < 1e-4, f"clock port net power {power.abs().max():.2e} != 0"


def test_passivity_holds_with_clock_port(model_ctx):
    """max Ḣ|u=0 ≤ 0 with the clock-coupling port active."""
    torch = model_ctx["torch"]
    m, x, gH, bg = model_ctx["m"], model_ctx["x"], model_ctx["gH"], model_ctx["bg"]
    J = m._last_J; R = m._last_R
    internal = torch.bmm(J - R, gH.unsqueeze(-1)).squeeze(-1)
    mp = m._modulated_port_term(x, gH, bg)
    Hdot = (gH * (internal + mp)).sum(-1).detach()
    assert float(Hdot.max()) <= 1e-4, f"passivity violated: max Hdot={float(Hdot.max()):.2e}"


def test_per_compartment_energy_nonneg(model_ctx):
    cH = model_ctx["subH"]["compartment"]
    assert sorted(cH.keys()) == [1, 2, 3, 4, 5]
    for cid, h in cH.items():
        assert bool((h >= 0).all()), f"H_{cid} has negative entries"


# ── Phase C: dual cascade + per-compartment passivity loss ────────────────────

def test_gate_assigns_two_clocks():
    """The rhythmicity gate labels nodes by clock; both clocks appear."""
    from biophasor.core.datagen.rhythmicity_gate import detect_all_layers
    d = generate_multi_omics(seed=0)
    gate = detect_all_layers(d)
    labels = np.concatenate([gate[L]["clock_label"] for L in
                             ["genomics", "proteome", "metabolome"]])
    present = set(labels.tolist())
    assert "circadian" in present and "redox" in present


def test_dual_cascade_recovers_both_clocks():
    """Both cascades — same R, different clock — recover their emergent lag."""
    from biophasor.core.datagen.rhythmicity_gate import detect_all_layers
    from biophasor.phnn.utils.cascade_predictor import CascadePredictor
    d = generate_multi_omics(seed=0)
    gate = detect_all_layers(d)
    pred = CascadePredictor(d["k_deg"]["genomics"], d["k_deg"]["proteome"],
                            d["k_deg"]["metabolome"], omega_clock=d["omega_clock"],
                            clock_bank=d["clock_bank"])
    aG, aP, aM = (gate["genomics"]["acrophase"], gate["proteome"]["acrophase"],
                  gate["metabolome"]["acrophase"])
    circ = pred.evaluate_cascade(aG[0:12], aP[0:12], d["k_deg"]["proteome"][0:12],
                                 omega=CLOCK_BANK["circadian"], label="circadian")
    redox = pred.evaluate_cascade(aP[12:20], aM[12:20], d["k_deg"]["metabolome"][12:20],
                                  omega=CLOCK_BANK["redox"], label="redox")
    assert circ["pearson_r"]  > 0.9, f"circadian cascade r={circ['pearson_r']:.3f}"
    assert redox["pearson_r"] > 0.9, f"redox cascade r={redox['pearson_r']:.3f}"
    assert circ["dose_response_direction_correct"]
    assert redox["dose_response_direction_correct"]


def test_per_compartment_passivity_loss_nonneg(model_ctx):
    """Per-compartment passivity loss is ≥ 0 and finite."""
    from biophasor.core.losses import loss_passivity_per_compartment
    m, gH, bg = model_ctx["m"], model_ctx["gH"], model_ctx["bg"]
    N = m.n_G + m.n_P + m.n_M
    L = loss_passivity_per_compartment(gH, m._last_R, bg["comp_id"], N).detach()
    assert float(L) >= 0.0 and np.isfinite(float(L))
