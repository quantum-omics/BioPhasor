"""
biophasor.core.operators — Biological phasor operators.

Implements the fundamental phasor operations:
  - coherence()      : mean resultant length C = |<e^{iφ}>|  ∈ [0, 1]
  - phasor_mean()    : weighted circular mean on U(1)
  - phase_couple()   : pairwise Kuramoto-style coupling
  - bio_shift()      : phase rotation by a fixed angle
  - bio_mix()        : 50/50 beam-splitter (complex average)
  - coherence_filter(): retain only features with C > threshold

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional, Union

import numpy as np


# ── Coherence ─────────────────────────────────────────────────────────────────

def coherence(
    phase: np.ndarray,
    amplitude: Optional[np.ndarray] = None,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute the **phase coherence** (mean resultant length) along an axis.

        C = |mean_amplitude_weighted( e^{iφ} )|

    For amplitude=None (pure-phase mode), this is the Kuramoto order parameter:

        C = |(1/N) Σ_j e^{iφ_j}|  ∈ [0, 1]

    Parameters
    ----------
    phase : np.ndarray  (... N ...)
        Phase values in radians.
    amplitude : np.ndarray | None
        Weighting amplitudes.  If None, unit amplitude is assumed.
    axis : int
        Axis over which to compute coherence (default: samples axis = 0).

    Returns
    -------
    np.ndarray
        Coherence values, shape = phase.shape with ``axis`` removed.
    """
    if amplitude is not None:
        z = amplitude * np.exp(1j * phase)
        w = amplitude.sum(axis=axis, keepdims=True)
        mean_z = z.sum(axis=axis) / (w.squeeze(axis=axis) + 1e-12)
    else:
        z = np.exp(1j * phase)
        mean_z = z.mean(axis=axis)
    return np.abs(mean_z)


# ── Circular mean ─────────────────────────────────────────────────────────────

def phasor_mean(
    phase: np.ndarray,
    weights: Optional[np.ndarray] = None,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute weighted **circular mean phase** along an axis.

        φ_mean = arg( Σ_j w_j · e^{iφ_j} )

    Parameters
    ----------
    phase : np.ndarray
    weights : np.ndarray | None
        Weights per sample.  If None, uniform weights are used.
    axis : int

    Returns
    -------
    np.ndarray
        Mean phases in radians ∈ (−π, π].
    """
    z = np.exp(1j * phase)
    if weights is not None:
        z = weights * z
    return np.angle(z.sum(axis=axis))


# ── Pairwise coupling ─────────────────────────────────────────────────────────

def phase_couple(
    phi_i: np.ndarray,
    phi_j: np.ndarray,
    coupling: float = 1.0,
    dt: float = 0.01,
) -> np.ndarray:
    """
    Apply one step of Kuramoto-style phase coupling:

        Δφ_i = (K/N) Σ_j sin(φ_j − φ_i)  ·  dt

    Parameters
    ----------
    phi_i : np.ndarray, shape (n_samples, n_features)
        Current phases of population i.
    phi_j : np.ndarray, shape (n_samples, n_features)
        Phases of the coupling population j (same shape as phi_i).
    coupling : float
        Coupling constant K.
    dt : float
        Integration step.

    Returns
    -------
    np.ndarray
        Updated phases (phi_i + Δφ).
    """
    delta = phi_j - phi_i
    dphi = coupling * np.sin(delta).mean(axis=0) * dt
    return phi_i + dphi


# ── Bio-inspired gate operations ──────────────────────────────────────────────

def bio_shift(phase: np.ndarray, delta: float) -> np.ndarray:
    """
    Phase rotation by a fixed angle δ (biological phase shift):

        φ' = (φ + δ) mod 2π   wrapped to (−π, π]

    Parameters
    ----------
    phase : np.ndarray
    delta : float   (radians)

    Returns
    -------
    np.ndarray   same shape as phase, values in (−π, π]
    """
    return ((phase + delta + np.pi) % (2 * np.pi)) - np.pi


def bio_mix(
    phase_a: np.ndarray,
    phase_b: np.ndarray,
    amplitude_a: Optional[np.ndarray] = None,
    amplitude_b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    50/50 beam-splitter in phasor space (complex average):

        z_out = (z_a + z_b) / 2

    Returns the phase of the mixture:  φ_out = arg(z_out).
    """
    A_a = amplitude_a if amplitude_a is not None else np.ones_like(phase_a)
    A_b = amplitude_b if amplitude_b is not None else np.ones_like(phase_b)
    z_a = A_a * np.exp(1j * phase_a)
    z_b = A_b * np.exp(1j * phase_b)
    z_mix = 0.5 * (z_a + z_b)
    return np.angle(z_mix)


# ── Feature filtering ─────────────────────────────────────────────────────────

def coherence_filter(
    phase: np.ndarray,
    threshold: float = 0.3,
    amplitude: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retain only features (columns) whose coherence exceeds ``threshold``.

    Parameters
    ----------
    phase : np.ndarray, shape (n_samples, n_features)
    threshold : float   default 0.3
    amplitude : np.ndarray | None

    Returns
    -------
    filtered_phase : np.ndarray
    mask : np.ndarray[bool], shape (n_features,)
    """
    C = coherence(phase, amplitude=amplitude, axis=0)
    mask = C >= threshold
    return phase[:, mask], mask
