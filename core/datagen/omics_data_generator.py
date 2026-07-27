"""
omics_data_generator.py

Biologically grounded multi-omics data generator for the GNN-pHNN framework.

Design notes:
─────────────────────────
* Two-layer state: abundance trajectories q_i(t) are the primitive state;
  phasors (φ, ω) are *derived* from the oscillatory subset, not injected.
* Rhythmicity fraction: only RHYTHMIC_FRACTION (~30%) of nodes in each layer
  genuinely oscillate on a limit cycle.  The rest follow first-order
  production–degradation with slow trend or flat homeostasis.
* No injected phase cascade: the transcript→protein→metabolite lag is an
  *emergent* property of the dissipation rates (k_deg), not baked into the
  generator.  The cascade direction comes from the first-order filter property:
      Δφ = arctan(ω_clock / k_deg).
* Stoichiometric conservation: three conserved moieties (adenylate pool,
  redox pool, cofactor pool) are preserved as linear invariants of the
  true dynamics.
* Named node classes: each node carries a biological class label.
* k_deg exported per node — the independent data needed for the cascade test.
* PLV is NOT injected.  Cross-layer PLV is computed from the resulting
  trajectories and used only as a weak prior, not as a training target.

Design rules (from 4-Regorous.ipynb):
  R1  pH structure only on genuine mass-flux couplings.
  R2  No fictitious negative damping.  Self-sustained oscillations draw
      energy from an explicit metabolic port.
  R3  Phasors only for genuinely rhythmic features.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
#  Layer / node configuration
# ─────────────────────────────────────────────────────────────────────────────

LAYER_CONFIG = {
    "genomics":   {"n_nodes": 40, "label": "G"},
    "proteome":   {"n_nodes": 35, "label": "P"},
    "metabolome": {"n_nodes": 25, "label": "M"},
}

# Fraction of nodes that are genuinely rhythmic — Phase A: now DETERMINED by
# compartment membership (clock != "none"), not a random draw.  Kept as a
# nominal reference for documentation/back-compat only.
RHYTHMIC_FRACTION = 0.30   # ~10-40% of transcriptome in clocked liver tissue

# ── Clock bank (Phase A) ──────────────────────────────────────────────────────
# The model now carries MULTIPLE biological clocks.  CLOCK_FREQ is retained as a
# backward-compatible alias for the primary (circadian) clock; new code should
# use CLOCK_BANK / a node's compartment clock.
try:
    from .compartments import (
        CLOCK_BANK, CLOCK_NAMES, PRIMARY_CLOCK, PRIMARY_CLOCK_FREQ,
        build_compartments, concat_compartment_arrays, COMPARTMENT_CLOCK,
    )
except ImportError:
    from data.compartments import (
        CLOCK_BANK, CLOCK_NAMES, PRIMARY_CLOCK, PRIMARY_CLOCK_FREQ,
        build_compartments, concat_compartment_arrays, COMPARTMENT_CLOCK,
    )

CLOCK_FREQ = PRIMARY_CLOCK_FREQ          # rad / h  (circadian; alias)

# Pseudo-time: dt = 0.1 h, total 240 h = 10 circadian cycles
DT   = 0.1    # hours per step
T_H  = 240.0  # total hours simulated

# Node class labels per layer (biologically typed)
NODE_CLASSES = {
    "genomics":   ["TF", "mRNA", "ncRNA", "mRNA"],          # cycling through labels
    "proteome":   ["enzyme", "signaling_protein", "TF_protein", "structural"],
    "metabolome": ["metabolite", "cofactor", "energy_carrier"],
}

# Conserved moiety indices: three linear invariants
# All three are metabolite-level moieties (the correct biochemical layer):
#   Adenylate pool: ATP+ADP+AMP (energy currency) — metabolome nodes 0-3
#   Redox pool: NAD+/NADH proxy — metabolome nodes 4-7
#   Cofactor pool: CoA/acetyl-CoA proxy — metabolome nodes 8-11
# Using metabolome-only assignment because moiety conservation is a
# property of the *metabolic* network, not the gene expression network.
CONSERVATION_GROUPS = {
    "adenylate": ("metabolome", list(range(0, 4)),    8.0),   # (layer, idx, total)
    "redox":     ("metabolome", list(range(4, 8)),    5.0),
    "cofactor":  ("metabolome", list(range(8, 12)),   6.0),
}


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _first_order_response(q_ss: float, k_prod: float, k_deg: float,
                          drive: np.ndarray, dt: float,
                          rng: np.random.Generator,
                          noise_std: float) -> np.ndarray:
    """
    Integrate first-order production-degradation:
        dq/dt = k_prod * drive(t) - k_deg * q + noise
    This is the **abundance base layer** for all nodes.
    For non-rhythmic nodes, drive(t) ≈ 1 (constant or slow trend).
    For rhythmic nodes, drive(t) includes a sinusoidal forcing term.
    The dissipation rate k_deg is what sets the phase lag for rhythmic nodes.
    """
    T = len(drive)
    q = np.zeros(T)
    q[0] = q_ss + rng.normal(0, noise_std * 0.5)
    for t in range(T - 1):
        dq = k_prod * drive[t] - k_deg * q[t] + rng.normal(0, noise_std)
        q[t + 1] = max(0.0, q[t] + dt * dq)
    return q


def _make_drive(is_rhythmic: bool, omega: float, phi0: float,
                t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Clock drive for a node.
    Rhythmic: sinusoidal + harmonic at 2ω (biological clock waveform).
    Non-rhythmic: slow sigmoid trend or flat with weak stochastic fluctuation.
    """
    if is_rhythmic:
        drive = (
            1.0
            + 0.6 * np.cos(omega * t + phi0)
            + 0.15 * np.cos(2 * omega * t + 2 * phi0)   # second harmonic: phase 2φ₀
        )
    else:
        # Slow trend: monotone or quasi-flat with weak noise
        trend = 1.0 + 0.1 * rng.normal(0, 1, len(t)).cumsum() / len(t)
        drive = np.clip(trend, 0.3, 2.0)
    return drive


# ─────────────────────────────────────────────────────────────────────────────
#  Pharmacological port protocol
# ─────────────────────────────────────────────────────────────────────────────

def _build_port_signal(T: int, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    """
    Three-port time course:
      u_G  : Zeitgeber/epigenetic input (light/feeding cycle entrainment)
      u_P  : Pharmacological drug (transient, ramp-in/ramp-out)
      u_M  : Nutrient influx (recovery phase)

    Returns
    -------
    u            : (T, 3) port signals
    state_labels : (T,) string array
    """
    u = np.zeros((T, 3))
    state_labels = np.full(T, "Homeostasis", dtype=object)

    # Zeitgeber always on (circadian entrainment via port 0)
    t_arr = np.arange(T) * dt
    u[:, 0] = 0.3 * np.cos(CLOCK_FREQ * t_arr)

    # Drug: t in [80h, 160h]
    t_drug_start = min(T, int(80.0  / dt))
    t_drug_end   = min(T, int(160.0 / dt))
    ramp_len     = int(10.0  / dt)

    window = t_drug_end - t_drug_start
    if window > 0:
        # We must be careful if the window is shorter than 2*ramp_len (e.g. very short T)
        actual_ramp = min(ramp_len, window // 2)
        plateau_len = max(0, window - 2 * actual_ramp)
        ramp_up   = np.linspace(0, 1, actual_ramp)
        plateau   = np.ones(plateau_len)
        ramp_down = np.linspace(1, 0, window - actual_ramp - plateau_len)
        profile   = np.concatenate([ramp_up, plateau, ramp_down])
        u[t_drug_start:t_drug_end, 1] = 0.8 * profile
        state_labels[t_drug_start:t_drug_end] = "Drug Administration"

    # Recovery/Nutrient influx: t in [160h, 220h]
    t_rec_start = min(T, int(160.0 / dt))
    t_rec_end   = min(T, int(220.0 / dt))
    window_rec = t_rec_end - t_rec_start
    if window_rec > 0:
        actual_ramp_rec = min(ramp_len, window_rec // 2)
        plateau_rec_len = max(0, window_rec - 2 * actual_ramp_rec)
        ramp_up_rec   = np.linspace(0, 1, actual_ramp_rec)
        plateau_rec   = np.ones(plateau_rec_len)
        ramp_down_rec = np.linspace(1, 0, window_rec - actual_ramp_rec - plateau_rec_len)
        profile_rec   = np.concatenate([ramp_up_rec, plateau_rec, ramp_down_rec])
        u[t_rec_start:t_rec_end, 2] = 0.6 * profile_rec
        state_labels[t_rec_start:t_rec_end] = "Metabolic Recovery"
    state_labels[t_rec_end:]    = "Homeostasis"

    return u, state_labels


# ─────────────────────────────────────────────────────────────────────────────
#  Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_multi_omics(
    dt: float = DT,
    total_hours: float = T_H,
    noise_std: float = 0.08,
    seed: int = 42,
) -> dict:
    """
    Generate biologically grounded two-layer multi-omics data.

    Biological design (Phase A: compartmental, multi-clock)
    -------------------------------------------------------
    * Abundance q_i(t) follows first-order production-degradation.  Rhythmicity
      is DETERMINED by compartment membership: a node oscillates iff its
      functional compartment carries a clock, and it is driven by THAT
      compartment's clock — circadian (2π/24) or redox (2π/20).  Non-clocked
      compartments follow a slow homeostatic trend.
    * Two genuinely distinct clocks (clock bank): the circadian TTFL is
      transcription-dependent (drives compartment c1 across G,P); the redox /
      peroxiredoxin oscillator is transcription-INDEPENDENT (drives c2 across
      P,M).  For the synthetic PoC the redox period (20 h) is offset from the
      circadian (24 h) so the two clocks are periodogram-separable.
    * The phase cascade is NOT injected.  It emerges from the first-order
      filter, now per clock: a downstream pool with degradation rate k_deg
      lags its rhythmic input by Δφ = arctan(ω_clock / k_deg), where ω_clock is
      the driving compartment's clock frequency.  Two cascades exist: circadian
      G→P (compartment c1) and redox P→M (compartment c2).
    * Three conserved moieties hold exactly (modulo noise clamp).
    * k_deg values differ systematically across layers:
        Genomics (mRNA):   fast turnover, k_deg ∈ [1.5, 4.0] /h  → small lag
        Proteome:          slow turnover, k_deg ∈ [0.1, 0.5] /h  → large lag
        Metabolome:        intermediate, k_deg ∈ [0.5, 1.5] /h   → medium lag
      This is consistent with measured mRNA/protein half-lives (Schwanhäusser
      et al. 2011, Nature).

    Parameters
    ----------
    dt           : time step in hours
    total_hours  : total simulation duration in hours
    noise_std    : additive Gaussian noise standard deviation
    seed         : random seed

    Returns
    -------
    dict with keys:
      't'                  : (T,)          time axis in hours
      'expression'         : dict layer → (N_layer, T) abundance q_i(t)
      'k_deg'              : dict layer → (N_layer,) degradation rates
      'rhythmic_mask'      : dict layer → (N_layer,) bool (clock != "none")
      'acrophase_true'     : dict layer → (N_layer,) true clock acrophase (radians)
      'node_class'         : dict layer → (N_layer,) str class labels
      'u'                  : (T, 3) port signals
      'state_labels'       : (T,) state label array
      'layer_config'       : LAYER_CONFIG
      'omega_clock'        : scalar, circadian (primary) angular frequency — back-compat
      'clock_bank'         : dict clock_name → angular frequency (rad/h)
      'compartment'        : dict layer → (N_layer,) int compartment id (1..5)
      'clock_label'        : dict layer → (N_layer,) str clock name / "none"
      'comp_id_global'     : (N,) int compartment id, concatenated node axis
      'clock_label_global' : (N,) str clock label, concatenated node axis
      'omega_node'         : (N,) per-node clock ω (0 if non-rhythmic)
      'conservation'       : dict moiety → {'layer', 'idx', 'total'}
      'dt'                 : float time step (hours)
    """
    rng = np.random.default_rng(seed)
    T   = int(total_hours / dt)
    t   = np.arange(T) * dt

    # k_deg ranges per layer (physiologically grounded half-lives)
    kdeg_range = {
        "genomics":   (1.5, 4.0),   # mRNA:     t½ ≈ 10–30 min
        "proteome":   (0.1, 0.5),   # protein:  t½ ≈ 1–7 h
        "metabolome": (0.5, 1.5),   # metabolite: t½ ≈ 30–80 min
    }

    expression    = {}
    k_deg_all     = {}
    rhythmic_mask = {}
    acrophase_true = {}
    node_class    = {}

    # ── Compartment / clock assignment (Phase A) ─────────────────────────────
    # Rhythmicity is now DETERMINED by compartment membership: a node oscillates
    # iff its compartment carries a clock.  Each rhythmic node is driven by ITS
    # compartment's clock frequency (circadian or redox).
    layer_sizes = {name: cfg["n_nodes"] for name, cfg in LAYER_CONFIG.items()}
    comp_built  = build_compartments(layer_sizes)
    comp_id_L   = comp_built["comp_id"]       # layer -> (N,) int
    clock_L     = comp_built["clock_label"]   # layer -> (N,) "circadian"/"redox"/"none"

    # ── Cascade wiring (Phase C) ─────────────────────────────────────────────
    # For the falsifiable cascade test the DOWNSTREAM pool must be driven by the
    # UPSTREAM node's TRAJECTORY (not an independent sinusoid), so that a single
    # first-order filter produces exactly the predicted lag relative to its
    # source:  tan(Δφ_{src→tgt}) = ω_clock / k_deg_tgt.
    #   circadian cascade — G[i] (clock-driven) → P[i], i∈[0,12).
    #   redox cascade     — P[12+j] (clock-driven enzyme) → M[12+j], j∈[0,8).
    # Sources are clock-driven; targets are trajectory-driven.  Because layers
    # are generated in order (genomics → proteome → metabolome) each source is
    # computed before its target.  Redox targets M[12:20] are FREE metabolites
    # (not in a conserved moiety pool), so they show a clean first-order lag.
    cascade_source = {}   # (target_layer, target_idx) -> (src_layer, src_idx)
    acro_src_lookup = {}   # (layer, idx) -> base acrophase (for downstream targets)
    for i in range(12):
        cascade_source[("proteome", i)]         = ("genomics", i)      # circadian G→P
    for j in range(8):
        cascade_source[("metabolome", 12 + j)]  = ("proteome", 12 + j)  # redox P→M

    # Base acrophases for the clock-driven SOURCES (targets inherit via trajectory)
    shared_acro_circ  = rng.uniform(-np.pi, np.pi, size=12)   # G[0:12] circadian sources
    shared_acro_redox = rng.uniform(-np.pi, np.pi, size=8)    # P[12:20] redox sources

    def _source_acrophase(layer_name: str, i: int, clock: str) -> float:
        """Base acrophase for a clock-driven source node."""
        if clock == "circadian" and layer_name == "genomics" and i < 12:
            return shared_acro_circ[i]
        if clock == "redox" and layer_name == "proteome" and 12 <= i < 20:
            return shared_acro_redox[i - 12]
        return rng.uniform(-np.pi, np.pi)     # other rhythmic nodes: independent

    for layer_name, cfg in LAYER_CONFIG.items():
        N     = cfg["n_nodes"]
        klo, khi = kdeg_range[layer_name]
        classes  = NODE_CLASSES[layer_name]

        expr     = np.zeros((N, T))
        k_deg    = rng.uniform(klo, khi, size=N)
        clocks_i = clock_L[layer_name]                       # (N,) clock label per node
        is_rhy   = (clocks_i != "none")                      # rhythmic ⟺ has a clock
        acro     = np.full(N, np.nan)
        nclass   = np.array([classes[i % len(classes)] for i in range(N)])

        for i in range(N):
            k_prod_ss = k_deg[i] * rng.uniform(2.0, 8.0)  # set-point abundance
            q_ss      = k_prod_ss / k_deg[i]

            clock = clocks_i[i]
            src   = cascade_source.get((layer_name, i))
            if src is not None:
                # Cascade TARGET: driven by the upstream node's trajectory.
                # The lag relative to the source emerges from the first-order
                # filter with THIS node's k_deg.  Normalize source trajectory to
                # oscillate around 1.0 (the drive convention).
                src_layer, src_i = src
                q_src   = expression[src_layer][src_i]           # already computed
                drive   = q_src / (q_src.mean() + 1e-8)
                acro[i] = acro_src_lookup.get((src_layer, src_i), np.nan)
            elif clock != "none":
                # Clock-driven SOURCE (or independent rhythmic node).
                omega_i = CLOCK_BANK[clock]
                phi0    = _source_acrophase(layer_name, i, clock)
                acro[i] = phi0
                drive   = _make_drive(True, omega_i, phi0, t, rng)
            else:
                drive = _make_drive(False, CLOCK_FREQ, 0.0, t, rng)

            expr[i] = _first_order_response(
                q_ss, k_prod_ss, k_deg[i], drive, dt, rng, noise_std
            )
            # Record this node's base acrophase for any downstream target
            acro_src_lookup[(layer_name, i)] = acro[i]

        # Enforce stoichiometric conservation on designated pools
        for moiety, (mlayer, midx, mtotal) in CONSERVATION_GROUPS.items():
            if mlayer != layer_name:
                continue
            pool_sum = expr[midx, :].sum(axis=0)   # (T,)
            # Normalize so pool sum = mtotal at each timepoint
            scale = mtotal / (pool_sum + 1e-8)
            expr[np.ix_(midx, np.arange(T))] *= scale[np.newaxis, :]

        expression[layer_name]     = expr
        k_deg_all[layer_name]      = k_deg
        rhythmic_mask[layer_name]  = is_rhy
        acrophase_true[layer_name] = acro
        node_class[layer_name]     = nclass

    u, state_labels = _build_port_signal(T, dt=dt)

    # Global (concatenated) compartment + clock label arrays and per-node ω
    comp_cat   = concat_compartment_arrays(comp_built, list(LAYER_CONFIG.keys()))
    omega_node = np.array([
        CLOCK_BANK[c] if c != "none" else 0.0 for c in comp_cat["clock_label"]
    ])

    return {
        "t":               t,
        "expression":      expression,
        "k_deg":           k_deg_all,
        "rhythmic_mask":   rhythmic_mask,
        "acrophase_true":  acrophase_true,
        "node_class":      node_class,
        "u":               u,
        "state_labels":    state_labels,
        "layer_config":    LAYER_CONFIG,
        "omega_clock":     CLOCK_FREQ,           # circadian (primary) — back-compat
        "clock_bank":      dict(CLOCK_BANK),     # {name: ω}  (Phase A)
        "compartment":     comp_id_L,            # layer -> (N,) int compartment id
        "clock_label":     clock_L,              # layer -> (N,) clock name / "none"
        "comp_id_global":  comp_cat["comp_id"],  # (N,) int
        "clock_label_global": comp_cat["clock_label"],  # (N,) str
        "omega_node":      omega_node,           # (N,) per-node clock ω (0 if non-rhythmic)
        "conservation":    CONSERVATION_GROUPS,
        "dt":              dt,
    }


def get_total_nodes() -> int:
    return sum(cfg["n_nodes"] for cfg in LAYER_CONFIG.values())


def get_layer_slices() -> dict:
    """Returns dict: layer_name → slice into concatenated node axis."""
    slices, idx = {}, 0
    for name, cfg in LAYER_CONFIG.items():
        slices[name] = slice(idx, idx + cfg["n_nodes"])
        idx += cfg["n_nodes"]
    return slices
