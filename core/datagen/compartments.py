"""
compartments.py  —  Phase A (new): functional-compartment membership + clock bank

The composite port-Hamiltonian upgrade decomposes the cell by FUNCTIONAL
COMPARTMENT (biological process), not by measurement layer.  Each compartment
is a self-contained pH subsystem; the oscillatory ones carry their OWN
biological clock.  This module is the single source of truth for:

  1. CLOCK_BANK          — the set of biological clocks (frequencies).
  2. COMPARTMENT_CONFIG  — the five functional compartments and their clock.
  3. build_compartments()— per-node compartment id + clock label, keyed by the
                           existing G/P/M layer axis (a compartment spans layers).

Design decisions (fixed by the researcher):
  * Compartments = functional modules.
  * Clock bank   = circadian + redox (two genuinely distinct oscillators;
                   the redox/peroxiredoxin clock is transcription-INDEPENDENT).

Single-formalism discipline: clocks are coupled INSIDE one classical pH model
via skew mass bonds and zero-power modulated ports — not as separate engines.

Note on synthetic periods
──────────────────────────
Biologically both the circadian TTFL and the peroxiredoxin redox oscillator
run at ~24 h.  For the synthetic proof-of-concept we give the redox clock a
SEPARABLE period (20 h) so the two clocks are identifiable by periodogram and
the model can be shown to carry and separate them.  In real data the clocks are
distinguished by compartment membership and transcription-(in)dependence, not by
a period gap; this is stated as a PoC simplification in the manuscript.

Design reference: notebooks/compartmental_upgrade_plan.md §2
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Clock bank — the biological clocks carried by the model
# ─────────────────────────────────────────────────────────────────────────────

CLOCK_BANK = {
    # name        angular frequency (rad/h)   biology
    "circadian": 2.0 * np.pi / 24.0,   # BMAL1/CLOCK–PER/CRY TTFL (transcription-dependent)
    "redox":     2.0 * np.pi / 20.0,   # peroxiredoxin/NADPH (transcription-INDEPENDENT)
}
CLOCK_NAMES = list(CLOCK_BANK.keys())          # ["circadian", "redox"]
N_CLOCKS    = len(CLOCK_BANK)

# Backward-compatible scalar (the "primary" clock) for any code still expecting one.
PRIMARY_CLOCK      = "circadian"
PRIMARY_CLOCK_FREQ = CLOCK_BANK[PRIMARY_CLOCK]


# ─────────────────────────────────────────────────────────────────────────────
#  Functional compartments
# ─────────────────────────────────────────────────────────────────────────────
#
# Five functional modules.  Each has an id (1..5), a clock ("circadian" /
# "redox" / "none"), and the set of nodes it owns per omic layer.  A compartment
# spans whichever layers its biology requires — membership is ORTHOGONAL to the
# G/P/M layer axis.
#
# Partition of the fixed synthetic node counts (n_G=40, n_P=35, n_M=25):
#
#   c1 core_clock   (circadian) : G[0:12],  P[0:12]
#   c2 redox        (redox)     : P[12:20], M[12:20]   ← free redox-responsive metabolites
#   c3 energy       (none)      : P[20:28], M[0:12]    ← adenylate+NAD+cofactor (conserved)
#   c4 signalling   (none)      : G[12:24], P[28:35]
#   c5 biosynthesis (none)      : G[24:40], M[20:25]
#
# Rhythmic ⟺ clock != "none".  Only c1 (circadian) and c2 (redox) oscillate.
# Every omic layer contains ≥1 rhythmic node (G via c1, P via c1+c2, M via c2).
#
# Redox metabolites vs conserved pools — a deliberate separation:
#   The redox compartment drives FREE redox-responsive metabolites M[12:20]
#   (e.g. glutathione, redox-sensitive intermediates) that can relax with a
#   clean first-order lag behind their enzymes — giving 8 matched P→M cascade
#   pairs, comparable to the 12-pair circadian G→P cascade.
#   The three CONSERVED moiety pools (adenylate M[0:4], NAD M[4:8], cofactor
#   M[8:12]) live in the clockless ENERGY compartment: a conserved pool
#   oscillates only as a RATIO (its members are sum-locked and cannot each show
#   an independent first-order lag), so it is homeostatically maintained here
#   rather than used as a cascade readout.  This keeps moiety conservation and
#   the falsifiable cascade test from contradicting each other.

COMPARTMENT_CONFIG = {
    1: {"name": "core_clock",   "clock": "circadian",
        "members": {"genomics": list(range(0, 12)),  "proteome": list(range(0, 12))}},
    2: {"name": "redox",        "clock": "redox",
        "members": {"proteome": list(range(12, 20)), "metabolome": list(range(12, 20))}},
    3: {"name": "energy",       "clock": "none",
        "members": {"proteome": list(range(20, 28)),
                    "metabolome": list(range(0, 12))}},
    4: {"name": "signalling",   "clock": "none",
        "members": {"genomics": list(range(12, 24)), "proteome": list(range(28, 35))}},
    5: {"name": "biosynthesis", "clock": "none",
        "members": {"genomics": list(range(24, 40)), "metabolome": list(range(20, 25))}},
}
COMPARTMENT_IDS   = list(COMPARTMENT_CONFIG.keys())
N_COMPARTMENTS    = len(COMPARTMENT_CONFIG)
COMPARTMENT_NAMES = {cid: c["name"] for cid, c in COMPARTMENT_CONFIG.items()}
COMPARTMENT_CLOCK = {cid: c["clock"] for cid, c in COMPARTMENT_CONFIG.items()}


def build_compartments(layer_sizes: dict) -> dict:
    """
    Build per-node compartment id and clock label arrays for each omic layer.

    Parameters
    ----------
    layer_sizes : dict  layer_name -> n_nodes (e.g. {"genomics":40,...})

    Returns
    -------
    dict with:
      'comp_id'     : dict layer -> (N_layer,) int   compartment id (1..5)
      'clock_label' : dict layer -> (N_layer,) object  "circadian"/"redox"/"none"
      'rhythmic'    : dict layer -> (N_layer,) bool   True ⟺ clock != "none"

    Raises
    ------
    ValueError if the partition does not exactly cover every node in a layer
    (each node assigned to exactly one compartment).
    """
    comp_id     = {L: np.zeros(n, dtype=int)        for L, n in layer_sizes.items()}
    clock_label = {L: np.array(["none"] * n, dtype=object) for L, n in layer_sizes.items()}

    for cid, cfg in COMPARTMENT_CONFIG.items():
        for layer, idxs in cfg["members"].items():
            for i in idxs:
                if comp_id[layer][i] != 0:
                    raise ValueError(
                        f"Node {layer}[{i}] assigned to >1 compartment "
                        f"({comp_id[layer][i]} and {cid})")
                comp_id[layer][i]     = cid
                clock_label[layer][i] = cfg["clock"]

    # Verify complete cover
    for layer, arr in comp_id.items():
        unassigned = np.where(arr == 0)[0]
        if len(unassigned) > 0:
            raise ValueError(
                f"Layer '{layer}' has unassigned nodes at indices {unassigned.tolist()}")

    rhythmic = {L: (clock_label[L] != "none") for L in layer_sizes}
    return {"comp_id": comp_id, "clock_label": clock_label, "rhythmic": rhythmic}


def concat_compartment_arrays(built: dict, layer_order: list) -> dict:
    """
    Concatenate per-layer compartment arrays into global (N,) arrays in the
    canonical layer order (genomics, proteome, metabolome).

    Returns dict with 'comp_id' (N,), 'clock_label' (N,), 'rhythmic' (N,).
    """
    comp_id  = np.concatenate([built["comp_id"][L]     for L in layer_order])
    clock    = np.concatenate([built["clock_label"][L] for L in layer_order])
    rhythmic = np.concatenate([built["rhythmic"][L]    for L in layer_order])
    return {"comp_id": comp_id, "clock_label": clock, "rhythmic": rhythmic}
