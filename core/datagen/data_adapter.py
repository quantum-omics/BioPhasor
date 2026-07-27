"""
data_adapter.py  —  Data Source Abstraction Layer

Purpose
-------
This module defines the CANONICAL DATA CONTRACT between data sources (synthetic or
real) and the GNN-pHNN training pipeline.  All downstream code (two_layer_state.py,
rhythmicity_gate.py, bio_graph.py, train_surrogate.py) expects data in the schema
defined here.

Swapping from synthetic to real data ONLY requires implementing a new loader function
that returns a dict matching OmicsDataSchema below.  No model or training code changes.

Current status
--------------
The model uses synthetic data from generate_multi_omics() (see omics_data_generator.py).
Real data integration is planned using public circadian multi-omics datasets:
  * MATS  (Hughes et al., 2009)  — liver circadian transcriptome
  * CYCLOPS (Anafi et al., 2017) — order-recovery algorithm for bulk RNA-seq
  * CircaDB (Hughes et al.)      — curated clock gene data portal

When real data is available, implement load_real_omics() below and update the
data source in train_surrogate.py from generate_multi_omics() to load_real_omics().

Data Schema
-----------
All loader functions must return a dict with the following keys:

  Key                 Type / Shape            Description
  ─────────────────── ─────────────────────── ────────────────────────────────────
  't'                 (T,) float              Time axis in hours
  'dt'                float                   Sampling interval in hours
  'expression'        dict[str → (N_l, T)]    Abundance q_i(t) per omic layer
  'k_deg'             dict[str → (N_l,)]      Degradation rates per layer (1/h)
  'rhythmic_mask'     dict[str → (N_l,) bool] True if node is clock-rhythmic
  'acrophase_true'    dict[str → (N_l,)]      Clock acrophase (rad), NaN if not rhythmic
  'node_class'        dict[str → (N_l,) str]  Biological class labels per node
  'u'                 (T, 3) float            Port signals [Zeitgeber, Drug, Nutrient]
  'state_labels'      (T,) str                Perturbation state labels
  'layer_config'      dict                    Layer metadata (n_nodes, label)
  'omega_clock'       float                   Circadian angular frequency (rad/h)
  'conservation'      dict                    Moiety conservation groups
  ─────────────────── ─────────────────────── ────────────────────────────────────

Layer names must be exactly: 'genomics', 'proteome', 'metabolome'.

State label vocabulary:
  'Homeostasis', 'Drug Administration', 'Metabolic Recovery'

Design reference: 4-Regorous.ipynb §0 (Phase 0 roadmap), §1 (Gap A)
"""

import numpy as np
from typing import Optional

# Import synthetic generator as the default source
try:
    from .omics_data_generator import generate_multi_omics, LAYER_CONFIG, CLOCK_FREQ
except ImportError:
    from data.omics_data_generator import generate_multi_omics, LAYER_CONFIG, CLOCK_FREQ


# ─────────────────────────────────────────────────────────────────────────────
#  Schema validator
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {
    "t", "dt", "expression", "k_deg", "rhythmic_mask",
    "acrophase_true", "node_class", "u", "state_labels",
    "layer_config", "omega_clock", "conservation",
}

# Phase A: compartmental / multi-clock keys.  Optional for back-compat — a
# legacy single-clock loader may omit them — but the synthetic generator now
# always provides them.  Validated for shape when present.
COMPARTMENT_KEYS = {
    "clock_bank", "compartment", "clock_label",
    "comp_id_global", "clock_label_global", "omega_node",
}

REQUIRED_LAYERS = {"genomics", "proteome", "metabolome"}

VALID_STATE_LABELS = {"Homeostasis", "Drug Administration", "Metabolic Recovery"}


def validate_omics_data(data: dict, strict: bool = False) -> list[str]:
    """
    Validate an omics data dict against the canonical schema.

    Parameters
    ----------
    data   : dict returned by a loader function
    strict : if True, raise ValueError on any violation; if False, return a list
             of warning strings.

    Returns
    -------
    List of warning/error strings. Empty list = valid.
    """
    issues = []

    # Top-level keys
    missing_keys = REQUIRED_KEYS - set(data.keys())
    if missing_keys:
        issues.append(f"Missing required keys: {missing_keys}")

    # Layer names
    for key in ["expression", "k_deg", "rhythmic_mask", "acrophase_true", "node_class"]:
        if key in data:
            missing_layers = REQUIRED_LAYERS - set(data[key].keys())
            if missing_layers:
                issues.append(f"'{key}' missing layers: {missing_layers}")

    # Shape consistency
    if "t" in data and "expression" in data:
        T = len(data["t"])
        for layer, expr in data["expression"].items():
            if expr.shape[1] != T:
                issues.append(
                    f"expression['{layer}'] has T={expr.shape[1]} but t has T={T}"
                )

    # Port signal shape
    if "u" in data and "t" in data:
        T = len(data["t"])
        u = data["u"]
        if u.shape != (T, 3):
            issues.append(f"'u' shape {u.shape} != ({T}, 3)")

    # State labels
    if "state_labels" in data:
        unique_labels = set(np.unique(data["state_labels"]))
        invalid = unique_labels - VALID_STATE_LABELS
        if invalid:
            issues.append(
                f"Unrecognised state_labels: {invalid}. "
                f"Valid: {VALID_STATE_LABELS}"
            )

    # ── Phase A: compartment / clock-bank consistency (when present) ─────────
    present_comp = COMPARTMENT_KEYS & set(data.keys())
    if present_comp:
        missing_comp = COMPARTMENT_KEYS - present_comp
        if missing_comp:
            issues.append(
                f"Partial compartment metadata: present {sorted(present_comp)}, "
                f"missing {sorted(missing_comp)}. Provide all or none.")
        # clock_bank must be a non-empty {name: freq} mapping
        if "clock_bank" in data:
            cb = data["clock_bank"]
            if not isinstance(cb, dict) or len(cb) == 0:
                issues.append("'clock_bank' must be a non-empty {name: frequency} dict")
        # per-node global arrays must match total node count
        if "comp_id_global" in data and "expression" in data:
            N = sum(e.shape[0] for e in data["expression"].values())
            for key in ("comp_id_global", "clock_label_global", "omega_node"):
                if key in data and len(data[key]) != N:
                    issues.append(f"'{key}' length {len(data[key])} != N={N}")

    if strict and issues:
        raise ValueError(
            "Omics data schema validation failed:\n" +
            "\n".join(f"  - {issue}" for issue in issues)
        )

    return issues


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic data source (current default)
# ─────────────────────────────────────────────────────────────────────────────

def load_synthetic_omics(
    dt: float = 0.1,
    total_hours: float = 240.0,
    noise_std: float = 0.08,
    seed: int = 42,
) -> dict:
    """
    Load synthetic multi-omics data that represents realistic biology.

    This is a thin wrapper around generate_multi_omics() that:
    1. Calls the generator with validated parameters.
    2. Runs schema validation to confirm the output is contract-compliant.
    3. Returns the dict ready for the downstream pipeline.

    Parameters
    ----------
    dt          : sampling interval (hours), default 0.1 h
    total_hours : total simulation duration (hours), default 240 h (10 circadian cycles)
    noise_std   : Gaussian measurement noise standard deviation
    seed        : random seed for reproducibility

    Returns
    -------
    Validated omics data dict matching OmicsDataSchema.
    """
    data = generate_multi_omics(dt=dt, total_hours=total_hours,
                                noise_std=noise_std, seed=seed)
    issues = validate_omics_data(data)
    if issues:
        import warnings
        for issue in issues:
            warnings.warn(f"[data_adapter] Schema warning: {issue}", UserWarning)
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Real data loader (placeholder — not yet implemented)
# ─────────────────────────────────────────────────────────────────────────────

def load_real_omics(
    data_path: Optional[str] = None,
    dataset: str = "mouse_liver_triomic",
    total_hours: float = 48.0,
    dt: float = 0.1,
    seed: int = 0,
    layer_config: Optional[dict] = None,
) -> dict:
    """
    Load the REAL circadian multi-omics dataset (assembled mouse liver, tri-omic).

    Returns a dict matching the canonical OmicsDataSchema, built entirely from
    publicly available mouse-liver circadian time-series. Full provenance,
    retrieval routes and the cross-cohort caveat are in data/real/PROVENANCE.md.
    Accessions and full citations below were each confirmed live via GEO /
    EuropePMC / Metabolomics Workbench:

      * transcriptome : Zhang et al. 2014, "A circadian gene expression atlas in
                        mammals", PNAS 111:16219-16224 (PMID 25349387);
                        GEO GSE54650 (GEO uid 200054650; liver, CT18-64, 2 h).
      * proteome      : Robles, Cox & Mann 2014, "In-vivo quantitative proteomics
                        reveals a key contribution of post-transcriptional ...",
                        PLoS Genet 10(1):e1004047 (PMID 24391516,
                        DOI 10.1371/journal.pgen.1004047; SILAC, CT0-45, 3 h).
      * metabolome    : Metabolomics Workbench ST002079, "Defining the mammalian
                        coactivation of hepatic 12-hour clock and lipid metabolism"
                        (WT liver, ZT0-20, 4 h).

    Pipeline (no quantity the validation later scores is fit here):
      1. select_nodes() maps real pools onto the model's fixed compartment layout
         (40 G + 35 P + 25 M), using Robles Table S4 matched transcript->protein
         pairs (with MEASURED lag) to populate the core-clock cascade slots.
      2. A time-referenced 24 h cosinor gives each pool's measured amplitude,
         acrophase and rhythmicity; the acrophase is preserved when the trajectory
         is reconstructed on the model's dense grid, so the cross-omic phase cascade
         is carried by REAL measured phases.
      3. k_deg comes from PUBLISHED half-lives (Schwanhausser 2011), never fit —
         the independent quantity the cascade prediction consumes.
      4. Conserved moiety pools (adenylate/redox/cofactor) are imposed on the
         metabolome energy slots so moiety conservation holds by construction.
      5. Zeitgeber port u = [light/dark drive, 0, 0]; single 'Homeostasis' state
         (the assembled dataset is a baseline circadian time-course, no drug arm).

    The returned dict additionally carries 'measured_cascade' (the real matched
    transcript->protein lags) so the cascade test can score the learned R against
    a model-independent target.
    """
    import numpy as _np
    try:
        from .real_features import (
            select_nodes, compile_k_deg, reconstruct_trajectory, OMEGA_CIRCADIAN,
        )
        from .compartments import (
            CLOCK_BANK, build_compartments, concat_compartment_arrays,
            COMPARTMENT_CONFIG,
        )
    except ImportError:
        from data.real_features import (
            select_nodes, compile_k_deg, reconstruct_trajectory, OMEGA_CIRCADIAN,
        )
        from data.compartments import (
            CLOCK_BANK, build_compartments, concat_compartment_arrays,
            COMPARTMENT_CONFIG,
        )

    layer_sizes = {"genomics": 40, "proteome": 35, "metabolome": 25}
    layer_order = ["genomics", "proteome", "metabolome"]
    labels = {"genomics": "G", "proteome": "P", "metabolome": "M"}

    # Model dense time grid
    t = _np.arange(0.0, total_hours, dt)
    T = len(t)

    # Compartment / clock structure (fixed layout)
    built = build_compartments(layer_sizes)
    comp = built["comp_id"]; clocklab = built["clock_label"]; rhy = built["rhythmic"]

    sel = select_nodes(seed=seed)

    expression, k_deg, rhythmic_mask, acrophase_true, node_class = {}, {}, {}, {}, {}
    for L in layer_order:
        s = sel[L]
        n = layer_sizes[L]
        clabel = clocklab[L]
        omega_L = _np.array([CLOCK_BANK.get(c, OMEGA_CIRCADIAN) if c != "none"
                             else OMEGA_CIRCADIAN for c in clabel])
        # rhythmic iff (a) the compartment carries a clock AND (b) the real pool
        # passed the cosinor rhythmicity test (p<0.05)
        real_rhythmic = (s["pval"] < 0.05)
        mask = rhy[L] & real_rhythmic
        traj = _np.empty((n, T))
        for i in range(n):
            traj[i] = reconstruct_trajectory(
                t, s["mesor"][i], s["amp"][i], s["acro"][i], omega_L[i], bool(mask[i]))
        expression[L] = traj
        k_deg[L] = compile_k_deg(L, s["symbols"], seed=seed)
        rhythmic_mask[L] = mask
        acr = _np.where(mask, s["acro"], _np.nan)
        acrophase_true[L] = acr
        node_class[L] = _np.array([f"{labels[L]}:{sym}" for sym in s["symbols"]], dtype=object)

    # ---- Conservation: impose moiety sums on metabolome energy slots ----
    # compartments.py energy compartment owns metabolome[0:12]; group as
    # adenylate[0:4], redox[4:8], cofactor[8:12] to mirror the synthetic contract.
    conservation = {
        "adenylate": ("metabolome", [0, 1, 2, 3], 8.0),
        "redox":     ("metabolome", [4, 5, 6, 7], 5.0),
        "cofactor":  ("metabolome", [8, 9, 10, 11], 6.0),
    }
    for _name, (L, idxs, total) in conservation.items():
        block = expression[L][idxs]                      # (k, T)
        # rescale each timepoint so the moiety sums to `total`, preserving shape
        colsum = block.sum(axis=0, keepdims=True)
        colsum[colsum == 0] = 1.0
        expression[L][idxs] = block / colsum * total

    # ---- Port signal: 12:12 light/dark Zeitgeber drive on channel 0 ----
    u = _np.zeros((T, 3))
    u[:, 0] = 0.5 * (1.0 + _np.cos(OMEGA_CIRCADIAN * t))   # bounded [0,1] Zeitgeber
    state_labels = _np.array(["Homeostasis"] * T, dtype=object)

    # ---- Global compartment arrays ----
    gcat = concat_compartment_arrays(built, layer_order)
    omega_node = _np.array([CLOCK_BANK[c] if c in CLOCK_BANK else OMEGA_CIRCADIAN
                            for c in gcat["clock_label"]])

    # ---- Measured cross-omic cascade target (model-independent) ----
    measured_cascade = {
        "pairs": sel["cascade"],                     # (symbol, g_pos, p_pos, lag_h)
        "lag_hours": _np.array([c[3] for c in sel["cascade"]]),
        "gene_pos":  _np.array([c[1] for c in sel["cascade"]]),
        "prot_pos":  _np.array([c[2] for c in sel["cascade"]]),
        "symbols":   [c[0] for c in sel["cascade"]],
        "lag_population": sel.get("measured_lag_population"),
        "lag_population_symbols": sel.get("measured_lag_symbols"),
    }

    data = {
        "t": t, "dt": dt,
        "expression": expression, "k_deg": k_deg,
        "rhythmic_mask": rhythmic_mask, "acrophase_true": acrophase_true,
        "node_class": node_class, "u": u, "state_labels": state_labels,
        "layer_config": {L: {"n_nodes": layer_sizes[L], "label": labels[L]}
                         for L in layer_order},
        "omega_clock": float(OMEGA_CIRCADIAN),
        "conservation": conservation,
        # compartment / multi-clock metadata
        "clock_bank": dict(CLOCK_BANK),
        "compartment": {L: comp[L] for L in layer_order},
        "clock_label": {L: clocklab[L] for L in layer_order},
        "comp_id_global": gcat["comp_id"],
        "clock_label_global": gcat["clock_label"],
        "omega_node": omega_node,
        # real-data extras (ignored by the schema validator, used by the cascade test)
        "measured_cascade": measured_cascade,
        "data_source": "real",
        "dataset": dataset,
        "node_symbols": {L: sel[L]["symbols"] for L in layer_order},
    }

    issues = validate_omics_data(data)
    if issues:
        import warnings
        for issue in issues:
            warnings.warn(f"[data_adapter] Real-data schema warning: {issue}", UserWarning)
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  Convenience: get the active data source
# ─────────────────────────────────────────────────────────────────────────────

def get_omics_data(source: str = "synthetic", **kwargs) -> dict:
    """
    Unified entry point for data loading.

    Parameters
    ----------
    source : 'synthetic' (default) or 'real'
    **kwargs : passed to the appropriate loader

    Returns
    -------
    Validated omics data dict.
    """
    if source == "synthetic":
        return load_synthetic_omics(**kwargs)
    elif source == "real":
        return load_real_omics(**kwargs)
    else:
        raise ValueError(f"Unknown data source '{source}'. Use 'synthetic' or 'real'.")
