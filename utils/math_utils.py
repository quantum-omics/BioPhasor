"""
biophasor.utils.math_utils — Circular statistics utilities.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations
from typing import Optional

import numpy as np


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to (−π, π]."""
    return ((angle + np.pi) % (2.0 * np.pi)) - np.pi


def circular_mean(phi: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute the circular (directional) mean of phase angles.

        μ = arg( mean( e^{iφ} ) )

    Parameters
    ----------
    phi : np.ndarray   phase values in radians
    axis : int

    Returns
    -------
    np.ndarray   mean phases in (−π, π]
    """
    return np.angle(np.exp(1j * phi).mean(axis=axis))


def circular_std(phi: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Circular standard deviation:

        σ_c = sqrt( −2 · ln(R) )

    where R = |mean(e^{iφ})| is the mean resultant length.

    Returns
    -------
    np.ndarray   values ≥ 0 (0 = all in phase; → ∞ = uniform)
    """
    R = np.abs(np.exp(1j * phi).mean(axis=axis))
    R = np.clip(R, 1e-12, 1.0 - 1e-12)
    return np.sqrt(-2.0 * np.log(R))


def angular_distance_wrap(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Absolute angular distance in [0, π]."""
    diff = wrap_to_pi(phi_a - phi_b)
    return np.abs(diff)


def vonmises_kl(
    mu_p: np.ndarray,
    kappa_p: float,
    mu_q: np.ndarray,
    kappa_q: float,
) -> np.ndarray:
    """
    Approximate KL divergence between two Von Mises distributions.

        KL(p||q) ≈ κ_p·(1 - I₁(κ_p)/I₀(κ_p)) + ln(I₀(κ_p)/I₀(κ_q))
                   + κ_q·cos(μ_p - μ_q)·I₁(κ_p)/I₀(κ_p)
                   − κ_p·I₁(κ_p)/I₀(κ_p)

    (Sadeghi & Dansereau, 2013 approximation)

    Parameters
    ----------
    mu_p, mu_q : np.ndarray   mean phases
    kappa_p, kappa_q : float   concentration params (≥ 0)

    Returns
    -------
    np.ndarray   element-wise approximate KL
    """
    from scipy.special import i0, i1

    I0p = i0(kappa_p)
    I1p = i1(kappa_p)
    I0q = i0(kappa_q)
    ratio_p = I1p / (I0p + 1e-12)

    kl = (
        np.log(I0p + 1e-12)
        - np.log(I0q + 1e-12)
        + kappa_q * ratio_p * np.cos(mu_p - mu_q)
        - kappa_p * ratio_p
    )
    return kl


def rayleigh_test(phi: np.ndarray) -> tuple[float, float]:
    """
    Rayleigh test for uniformity of circular data.

    H₀: the population is uniformly distributed on the circle.

    Parameters
    ----------
    phi : np.ndarray, shape (N,)   phase values

    Returns
    -------
    (R, p_value)
        R : float   mean resultant length
        p_value : float   approximate p-value (Zar, 2010)
    """
    N = len(phi)
    R = float(np.abs(np.exp(1j * phi).mean()))
    # Test statistic
    Z = N * R ** 2
    # p-value approximation valid for N > 4
    p = np.exp(
        np.sqrt(1 + 4 * N + 4 * (N ** 2 - Z ** 2)) - (1 + 2 * N)
    )
    return R, float(p)
