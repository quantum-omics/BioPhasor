"""
biophasor.ml.losses — Phasor-aware training objectives.

    circular_mse_loss : MSE on the unit circle (wraps angular difference)
    coherence_loss    : 1 − R²   where R = |mean(e^{iφ})|

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

import numpy as np


def circular_mse_loss(pred_phase: "torch.Tensor", target_phase: "torch.Tensor") -> "torch.Tensor":
    """
    Circular MSE loss — accounts for angle wrapping.

        L = mean( 1 - cos(φ_pred - φ_target) )

    ∈ [0, 2].  L = 0 → perfect alignment; L = 2 → anti-phase.

    Parameters
    ----------
    pred_phase, target_phase : torch.Tensor, any shape

    Returns
    -------
    torch.Tensor   scalar loss
    """
    import torch
    diff = pred_phase - target_phase
    return (1.0 - torch.cos(diff)).mean()


def coherence_loss(phase: "torch.Tensor") -> "torch.Tensor":
    """
    Coherence loss — penalises incoherent (spread-out) phase distributions.

        L = 1 - R²   where R = |mean(e^{iφ})|

    ∈ [0, 1].  L → 0 → all phases aligned; L → 1 → uniform distribution.

    Parameters
    ----------
    phase : torch.Tensor, shape (batch, features)

    Returns
    -------
    torch.Tensor   scalar loss
    """
    import torch
    z = torch.exp(1j * phase.to(torch.complex64))
    R = z.mean(dim=0).abs()         # (features,)
    return (1.0 - R.pow(2)).mean()


def von_mises_kl_loss(
    mu_pred: "torch.Tensor",
    kappa_pred: "torch.Tensor",
    mu_prior: float = 0.0,
    kappa_prior: float = 0.0,
) -> "torch.Tensor":
    """
    Approximate KL divergence for Von Mises VAE latent space.

    For κ_prior = 0 (uniform prior):
        KL(VonMises(μ, κ) || uniform) = −ln(I₀(κ)) + κ·I₁(κ)/I₀(κ)

    Parameters
    ----------
    mu_pred    : torch.Tensor   predicted mean phases
    kappa_pred : torch.Tensor   predicted concentrations (≥ 0)
    mu_prior   : float
    kappa_prior : float

    Returns
    -------
    torch.Tensor   scalar KL loss
    """
    import torch
    from scipy.special import i0, i1
    kappa_np = kappa_pred.detach().cpu().numpy()
    I0 = torch.tensor(i0(kappa_np), dtype=torch.float32, device=kappa_pred.device)
    I1 = torch.tensor(i1(kappa_np), dtype=torch.float32, device=kappa_pred.device)

    if kappa_prior == 0:
        # KL to uniform
        kl = -torch.log(I0 + 1e-8) + kappa_pred * I1 / (I0 + 1e-8)
    else:
        I0_prior = torch.tensor(i0(kappa_prior), dtype=torch.float32)
        kl = (
            torch.log(I0_prior + 1e-8)
            - torch.log(I0 + 1e-8)
            + kappa_prior * (I1 / (I0 + 1e-8)) * torch.cos(mu_pred - mu_prior)
            - kappa_pred * (I1 / (I0 + 1e-8))
        )
    return kl.mean()
