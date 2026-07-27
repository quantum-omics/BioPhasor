"""
edits.py  —  Class-preserving structured edits to a port-Hamiltonian cell.

Both cell-type specialisation (atlas) and disease perturbation (EHR) are
structured deformations delta-S = (dJ, dR, dH, dG) of the reference
S0 = (J, R, H, G), subject to class constraints that keep the edited system
passive:
    dJ skew-symmetric              (J + dJ stays skew)
    R + dR positive semidefinite   (dissipation stays a valid PSD cone member)
so that  H_dot = -grad_H^T (R+dR) grad_H <= 0  by construction.

This module provides:
  * builders for each of the six taxonomy classes P1..P6, each returning a
    class-preserving edit;
  * apply_edit(): compose a weighted, nonnegative combination of edits;
  * assert_passive(): a hard runtime check on the edited dynamics;
  * project_R_psd(): clamp a dissipation diagonal onto the PSD cone.

The edits act on the model's dissipation diagonal and interconnection; the
learned energy net H and ports G are perturbed through explicit scale factors so
the pH class is preserved exactly (no free-form weight surgery).
"""
from __future__ import annotations
import numpy as np
import torch

# ---- six-class taxonomy (shared by atlas modules and disease perturbations) ----
# P1 coupling gain (scale J edges) ; P2 topology (mask/add J edges) ;
# P3 turnover (shift R diagonal, >=0) ; P4 cross-compartment coupling ;
# P5 energy set-point (scale H) ; P6 exogenous drive (scale/add G u).

def project_R_psd(R_diag: np.ndarray) -> np.ndarray:
    """Clamp a dissipation diagonal onto the PSD cone (elementwise >= 0)."""
    return np.clip(R_diag, 0.0, None)


def edit_turnover(R_diag, delta, mask=None):
    """P3: additive shift to the dissipation diagonal, kept PSD."""
    d = np.zeros_like(R_diag) if mask is None else np.asarray(mask, float)
    if mask is None:
        d[:] = 1.0
    return project_R_psd(R_diag + delta * d)


def edit_coupling_gain(J, gain, block=None):
    """P1: rescale interconnection edge weights; skew-symmetry preserved because
    a scalar (or symmetric-block) multiple of a skew matrix is skew."""
    J = np.asarray(J, float)
    if block is None:
        Jp = gain * J
    else:
        b = np.asarray(block, float)                 # (n,) node mask in [0,1]
        S = np.outer(b, b)                            # symmetric gate
        Jp = J + (gain - 1.0) * S * J
    # re-skew to kill numerical drift
    return 0.5 * (Jp - Jp.T)


def edit_drive_scale(scale):
    """P6: multiplicative factor on the port drive u (applied at rollout time)."""
    return float(scale)


def assert_skew(J, tol=1e-6):
    err = np.abs(J + J.T).max()
    assert err <= tol, f"J not skew: max|J+J^T|={err:.2e}"
    return err


def assert_R_psd(R_diag, tol=1e-9):
    mn = float(np.min(R_diag))
    assert mn >= -tol, f"R diagonal not PSD: min={mn:.2e}"
    return mn


def assert_passive(H_trace, tol=1e-5):
    """Hard check: energy is non-increasing along an autonomous rollout.
    Returns (max_dH_per_step, fraction_of_positive_steps)."""
    dH = np.diff(np.asarray(H_trace, float))
    frac_pos = float((dH > tol).mean())
    assert frac_pos == 0.0, f"passivity violated: {frac_pos:.3f} of steps have dH>0"
    return float(dH.max()), frac_pos


def compose_edits(R_diag, J, edits, weights):
    """Apply a weighted (nonnegative) combination of class-preserving edits.
    `edits` is a list of dicts each with keys among {dR, dR_mask, J_gain,
    J_block, drive_scale}; `weights` are nonnegative floats.
    Returns (R_diag_edited, J_edited, drive_scale)."""
    weights = np.asarray(weights, float)
    assert (weights >= 0).all(), "weights must be nonnegative"
    Rd = np.asarray(R_diag, float).copy()
    Je = np.asarray(J, float).copy()
    drive = 1.0
    for w, e in zip(weights, edits):
        if "dR" in e:
            Rd = edit_turnover(Rd, w * e["dR"], e.get("dR_mask"))
        if "J_gain" in e:
            Je = edit_coupling_gain(Je, 1.0 + w * (e["J_gain"] - 1.0), e.get("J_block"))
        if "drive_scale" in e:
            drive *= (1.0 + w * (e["drive_scale"] - 1.0))
    assert_skew(Je); assert_R_psd(Rd)
    return Rd, Je, drive
