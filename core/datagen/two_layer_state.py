"""
two_layer_state.py

Two-layer state assembly for the biologically grounded GNN-pHNN.

Layer structure
───────────────
Base layer   — abundance deviations q_i for ALL N nodes.
              Directly measurable from omics; conserved under stoichiometry.
              Chemical potential proxy: μ_i = q_i / C_i.

Derived layer — phasor coordinates [sin(φ_i), cos(φ_i)] for the RHYTHMIC
              subset only (as identified by rhythmicity_gate.py).
              These are derived via Hilbert transform of the bandpass-filtered
              abundance, NOT primitive state variables.

State vector assembled for the model
─────────────────────────────────────
x = [ q_1, …, q_N,                           ← abundance (all nodes, dim N)
      sin(φ_{r1}), cos(φ_{r1}), ω_{r1},      ← phasor (rhythmic nodes, dim 3·N_r)
      … ]

Total dim: N + 3 * N_r   where N_r = rhythmic_mask.sum()

Targets (dx/dt)
───────────────
dx/dt base   = dq/dt  (numerical derivative of abundance)
dx/dt phasor = [cos(φ)·ω, -sin(φ)·ω, dω/dt]  (phasor kinematics on circle)

Conservation check
──────────────────
For each conserved moiety defined in omics_data_generator.CONSERVATION_GROUPS,
verify |Sq(t) - const|² < TOL across all time points.  This is the runtime
test that the generator enforced the constraint.

Design reference: 4-Regorous.ipynb §2.1, §2.2
"""

import numpy as np
import torch
from scipy.signal import hilbert as sp_hilbert

try:
    from .omics_data_generator import LAYER_CONFIG, CONSERVATION_GROUPS, get_layer_slices
except ImportError:
    from data.omics_data_generator import LAYER_CONFIG, CONSERVATION_GROUPS, get_layer_slices

# Thermodynamic capacitance default (sets μ = q/C proportionality)
DEFAULT_CAPACITANCE = 1.0


# ─────────────────────────────────────────────────────────────────────────────
#  Phasor extraction from bandpass-filtered abundance
# ─────────────────────────────────────────────────────────────────────────────

def _extract_phasors(
    expr_filtered: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hilbert-transform bandpass-filtered abundance to get phasors.
    Only called for rhythmic nodes.

    Parameters
    ----------
    expr_filtered : (N_r, T) bandpass-filtered abundance
    dt            : time step in hours

    Returns
    -------
    phi   : (N_r, T) unwrapped instantaneous phase
    omega : (N_r, T) instantaneous angular frequency
    amp   : (N_r, T) envelope amplitude
    """
    analytic = sp_hilbert(expr_filtered, axis=-1)
    phi      = np.unwrap(np.angle(analytic), axis=-1)
    amp      = np.abs(analytic)
    omega    = np.gradient(phi, dt, axis=-1)
    return phi, omega, amp


def _bandpass_circadian(signal: np.ndarray, dt: float) -> np.ndarray:
    """Circadian bandpass (20–28 h)."""
    from scipy.signal import butter, filtfilt
    fs  = 1.0 / dt
    nyq = 0.5 * fs
    lo  = np.clip((1.0 / 28.0) / nyq, 1e-4, 0.499)
    hi  = np.clip((1.0 / 20.0) / nyq, 1e-4, 0.499)
    b, a = butter(3, [lo, hi], btype="band")
    return filtfilt(b, a, signal - signal.mean(axis=-1, keepdims=True), axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  State assembly
# ─────────────────────────────────────────────────────────────────────────────

def assemble_two_layer_state(
    omics_data: dict,
    gate_results: dict,
    capacitance: float = DEFAULT_CAPACITANCE,
) -> dict:
    """
    Assemble the two-layer state from abundance data and rhythmicity gate.

    Parameters
    ----------
    omics_data   : dict from generate_multi_omics()
    gate_results : dict from detect_all_layers() — per-layer rhythmicity dicts
    capacitance  : thermodynamic capacitance C_i (uniform default)

    Returns
    -------
    dict with:
      'x'                : (T, N + 3*N_r)  full state tensor (float32)
      'dx_dt'            : (T, N + 3*N_r)  state derivatives
      'q_raw'            : (T, N)           raw abundance deviations
      'phi_rhythmic'     : (T, N_r)         instantaneous phase (rhythmic nodes)
      'omega_rhythmic'   : (T, N_r)         angular freq (rhythmic nodes)
      'amp_rhythmic'     : (T, N_r)         amplitude envelope (rhythmic nodes)
      'mu'               : (T, N)           chemical potential proxy q/C
      'rhythmic_indices' : (N_r,) global indices of rhythmic nodes in [0, N)
      'N_total'          : int, total nodes N
      'N_rhythmic'       : int, N_r
      'state_dim'        : int, N + 3*N_r
      'layer_slices'     : dict  (layer_name → slice in concatenated dim)
      'layer_slices_rhythmic' : dict (layer_name → slice in rhythmic-only dim)
      'scale_q'          : float (normalization factor for q)
      'scale_dx'         : float (normalization factor for dx_dt)
    """
    dt          = omics_data["dt"]
    expression  = omics_data["expression"]
    layer_names = list(LAYER_CONFIG.keys())
    t           = omics_data["t"]
    T           = len(t)
    layer_slices = get_layer_slices()

    # ── Collect per-node data ────────────────────────────────────────────────
    all_expr      = []           # (N, T)
    all_is_rhy    = []           # (N,)
    for name in layer_names:
        all_expr.append(expression[name])
        all_is_rhy.append(gate_results[name]["rhythmic_mask"])

    all_expr   = np.vstack(all_expr)      # (N, T)
    all_is_rhy = np.concatenate(all_is_rhy)  # (N,)
    N          = all_expr.shape[0]
    rhythmic_indices = np.where(all_is_rhy)[0]
    N_r        = len(rhythmic_indices)

    # ── Base layer: abundance deviations q = expr - homeostasis ─────────────
    # Homeostasis = mean over first 10% of trajectory (before drug)
    homeostasis  = all_expr[:, : max(1, T // 10)].mean(axis=1, keepdims=True)
    q            = all_expr - homeostasis             # (N, T), centred
    mu           = q / capacitance                    # chemical potential proxy

    # ── Derived layer: phasors for rhythmic nodes only ───────────────────────
    phi_rhy   = np.zeros((N_r, T))
    omega_rhy = np.zeros((N_r, T))
    amp_rhy   = np.zeros((N_r, T))

    if N_r > 0:
        expr_rhy      = all_expr[rhythmic_indices]          # (N_r, T)
        filtered_rhy  = _bandpass_circadian(expr_rhy, dt)   # (N_r, T)
        phi_rhy, omega_rhy, amp_rhy = _extract_phasors(filtered_rhy, dt)

    # ── Assemble state x = [q (N), sin_phi (N_r), cos_phi (N_r), omega (N_r)] ──
    if N_r > 0:
        sin_phi = np.sin(phi_rhy)   # (N_r, T)
        cos_phi = np.cos(phi_rhy)
        x_np    = np.vstack([q, sin_phi, cos_phi, omega_rhy]).T   # (T, N+3Nr)
    else:
        x_np = q.T                                                  # (T, N)

    # ── Target dx/dt ─────────────────────────────────────────────────────────
    dq_dt = np.gradient(q, dt, axis=1)   # (N, T)

    if N_r > 0:
        # Phasor kinematics on S¹: d(sin φ)/dt = cos(φ)·ω, etc.
        d_sinphi = cos_phi * omega_rhy                                 # (N_r, T)
        d_cosphi = -sin_phi * omega_rhy
        d_omega  = np.gradient(omega_rhy, dt, axis=1)
        dx_np = np.vstack([dq_dt, d_sinphi, d_cosphi, d_omega]).T     # (T, N+3Nr)
    else:
        dx_np = dq_dt.T                                                # (T, N)

    # ── Normalise abundance and phasor blocks SEPARATELY ────────────────────
    # IMPORTANT: sin φ and cos φ are bounded in [-1, 1] by definition.
    # Normalizing them by scale_q (an abundance scale) would break the
    # trigonometric identity sin²φ + cos²φ = 1 in the normalized space.
    # ω (rad/h) has physical units different from abundance.
    # Strategy: normalize the abundance block only; leave phasors in natural units.
    scale_q  = max(float(np.max(np.abs(q))),  1.0) + 1e-8   # ≥1 to prevent explosion
    scale_dx = max(float(np.max(np.abs(dq_dt))), 1e-4) + 1e-8

    q_norm   = q / scale_q
    dq_norm  = dq_dt / scale_dx

    if N_r > 0:
        # Phasor block: sin/cos already in [-1,1]; ω needs its own scale
        scale_omega = max(float(np.max(np.abs(omega_rhy))), 1e-4) + 1e-8
        omega_norm  = omega_rhy / scale_omega
        d_omega_norm = d_omega  / scale_omega

        x_np  = np.vstack([q_norm, sin_phi, cos_phi, omega_norm]).T    # (T, N+3Nr)
        dx_np = np.vstack([dq_norm, d_sinphi, d_cosphi, d_omega_norm]).T
    else:
        x_np  = q_norm.T
        dx_np = dq_norm.T
        scale_omega = 1.0

    # ── Layer slices in rhythmic-only dimension ──────────────────────────────
    layer_slices_rhy = {}
    rhy_cursor       = 0
    for name in layer_names:
        mask_layer = gate_results[name]["rhythmic_mask"]
        n_rhy_layer = mask_layer.sum()
        layer_slices_rhy[name] = slice(rhy_cursor, rhy_cursor + n_rhy_layer)
        rhy_cursor += n_rhy_layer

    return {
        "x":                   torch.tensor(x_np,          dtype=torch.float32),
        "dx_dt":               torch.tensor(dx_np,         dtype=torch.float32),
        "q_raw":               torch.tensor(q_norm.T,      dtype=torch.float32),  # (T, N)
        "phi_rhythmic":        torch.tensor(phi_rhy.T,     dtype=torch.float32),  # (T, N_r)
        "omega_rhythmic":      torch.tensor(omega_rhy.T,   dtype=torch.float32),  # (T, N_r)
        "amp_rhythmic":        torch.tensor(amp_rhy.T,     dtype=torch.float32),  # (T, N_r)
        "rhythmic_indices":    rhythmic_indices,
        "N_total":             N,
        "N_rhythmic":          N_r,
        "state_dim":           N + 3 * N_r,
        "layer_slices":        layer_slices,
        "layer_slices_rhythmic": layer_slices_rhy,
        "scale_q":             scale_q,
        "scale_dx":            scale_dx,
        "scale_omega":         scale_omega if N_r > 0 else 1.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Conservation verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_conservation(omics_data: dict, tol: float = 0.1) -> dict:
    """
    Verify that stoichiometric conservation constraints hold in the data.

    For each conserved moiety defined in CONSERVATION_GROUPS, compute:
      deviation(t) = |Σ_i q_i(t) - target| / target

    Returns dict: moiety_name → max_relative_deviation (float)
    """
    results = {}
    layer_slices = get_layer_slices()
    for moiety, (layer, indices, total) in CONSERVATION_GROUPS.items():
        expr   = omics_data["expression"][layer]   # (N_layer, T)
        pool_q = expr[indices, :].sum(axis=0)       # (T,)
        dev    = np.abs(pool_q - total) / total
        results[moiety] = {
            "max_relative_deviation": float(dev.max()),
            "mean_relative_deviation": float(dev.mean()),
            "passes": bool(dev.max() < tol),
        }
    return results
