"""
test_real_loader.py — contract tests for the REAL circadian multi-omics loader.

Mirrors the synthetic-pipeline contract tests but exercises load_real_omics(),
which assembles a real mouse-liver tri-omic dataset (GSE54650 transcriptome +
Robles 2014 proteome + Metabolomics-Workbench ST002079 metabolome).

These tests are skipped automatically if the staged real-data bundle is absent
(so the suite still runs on a clone without the ~34 MB raw downloads).
"""
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from biophasor.core.datagen.data_adapter import validate_omics_data, load_real_omics, get_omics_data
from biophasor.core.datagen.two_layer_state import assemble_two_layer_state, verify_conservation
from biophasor.core.datagen.rhythmicity_gate import detect_all_layers

_REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")
_HAVE_REAL = all(os.path.exists(os.path.join(_REAL_DIR, p)) for p in [
    "transcriptome/liver_transcriptome_genelevel.npz",
    "proteome/liver_proteome.npz",
    "proteome/measured_cascade_lag.npz",
    "metabolome/liver_metabolome.npz",
])
pytestmark = pytest.mark.skipif(not _HAVE_REAL,
                                reason="staged real-data bundle not present")

LAYERS = ("genomics", "proteome", "metabolome")
SIZES = {"genomics": 40, "proteome": 35, "metabolome": 25}


@pytest.fixture(scope="module")
def real_data():
    return load_real_omics(total_hours=48.0, seed=0)


# ── Schema contract ───────────────────────────────────────────────────────────

def test_schema_valid(real_data):
    issues = validate_omics_data(real_data)
    assert issues == [], f"Real-data schema issues: {issues}"


def test_layer_shapes(real_data):
    T = len(real_data["t"])
    for L in LAYERS:
        assert real_data["expression"][L].shape == (SIZES[L], T)


def test_port_shape(real_data):
    T = len(real_data["t"])
    assert real_data["u"].shape == (T, 3)


def test_state_labels_valid(real_data):
    from biophasor.core.datagen.data_adapter import VALID_STATE_LABELS
    assert set(np.unique(real_data["state_labels"])) <= VALID_STATE_LABELS


def test_dispatch_source_real(monkeypatch):
    d = get_omics_data(source="real", total_hours=24.0)
    assert d["data_source"] == "real"


# ── Biological / physics contract ─────────────────────────────────────────────

def test_every_layer_has_a_rhythmic_pool(real_data):
    for L in LAYERS:
        assert real_data["rhythmic_mask"][L].sum() >= 1, \
            f"layer {L} has no rhythmic pool"


def test_kdeg_finite_and_physiological(real_data):
    # published half-life ordering: mRNA faster than protein; all finite, positive
    for L in LAYERS:
        k = real_data["k_deg"][L]
        assert np.all(np.isfinite(k)) and np.all(k > 0)
    assert np.median(real_data["k_deg"]["genomics"]) > \
           np.median(real_data["k_deg"]["proteome"]), \
        "mRNA should degrade faster than protein"


def test_acrophase_nan_iff_not_rhythmic(real_data):
    for L in LAYERS:
        mask = real_data["rhythmic_mask"][L]
        acr = real_data["acrophase_true"][L]
        assert np.all(np.isfinite(acr[mask]))
        assert np.all(np.isnan(acr[~mask]))


def test_conservation_holds(real_data):
    for name, (L, idxs, total) in real_data["conservation"].items():
        s = real_data["expression"][L][idxs].sum(axis=0)
        assert np.allclose(s, total, atol=1e-6), \
            f"moiety {name} not conserved (dev {np.abs(s-total).max():.2e})"


def test_measured_cascade_present(real_data):
    mc = real_data["measured_cascade"]
    assert len(mc["pairs"]) >= 8, "need matched transcript->protein pairs for cascade"
    assert np.all(np.isfinite(mc["lag_hours"]))
    # measured median lag is physiological (a few hours), not degenerate
    assert 1.0 < np.median(mc["lag_hours"]) < 15.0


# ── Downstream assembly contract ──────────────────────────────────────────────

def test_two_layer_state_assembles(real_data):
    gate = detect_all_layers(real_data)
    state = assemble_two_layer_state(real_data, gate)
    x = state["x"].numpy()
    assert np.all(np.isfinite(x))
    assert x.shape[1] == state["state_dim"]
    # phasor identity sin^2 + cos^2 = 1 for the rhythmic phasor block
    N, Nr = state["N_total"], state["N_rhythmic"]
    if Nr > 0:
        sinb = x[:, N:N + Nr]
        cosb = x[:, N + Nr:N + 2 * Nr]
        assert np.allclose(sinb ** 2 + cosb ** 2, 1.0, atol=1e-3)
