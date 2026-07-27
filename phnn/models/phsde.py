"""
phsde.py  —  Thermodynamically consistent stochastic generator for multi-omic
cell dynamics (port-Hamiltonian Langevin SDE).

The trained port-Hamiltonian model defines a deterministic drift

    f(x) = (J(x) - R(x)) grad H(x) + G u,

with R(x) >= 0 the learned dissipation.  The fluctuation-dissipation theorem
pairs that same dissipation with noise: adding a diffusion sqrt(2 T R(x)) dW
gives the port-Hamiltonian Langevin equation

    dx = [ (J - R) grad H + G u ] dt + sqrt(2 T R(x)) dW .

Because the noise amplitude is set by the *same* R that damps the dynamics,
the system has (in idealized coordinates) the Gibbs stationary density
p(x) ∝ exp(-H(x) / T): the skew part J transports probability along level sets
of H without changing the density, while R and the matched noise balance to the
Boltzmann measure.  T is a sampling temperature that sets fluctuation amplitude.

R(x) is diagonal in this model (positive on the abundance block, zero on the
derived phasor block), so the diffusion is coordinate-wise sqrt(2 T R_ii).

Conserved moiety pools (adenylate, redox, cofactor) are exact linear invariants
S x = const.  The noise increment is projected into the null-space of S,
P = I - S^T (S S^T)^{-1} S, so stochastic sampling does not violate the
stoichiometric conservation laws.

Functions
---------
conservation_projector : build P from the stoichiometry S and state dimension.
phsde_step             : one Euler-Maruyama step (optionally conservation-safe).
phsde_rollout          : integrate the SDE forward, returning states and energy.
"""

from __future__ import annotations
import numpy as np
import torch


def conservation_projector(S_full: torch.Tensor) -> torch.Tensor:
    """Null-space projector P = I - S^T (S S^T)^{-1} S for constraints S x = const.

    S_full : (n_constraints, state_dim) stoichiometry acting on the full state.
    Returns P : (state_dim, state_dim) with S_full @ P = 0 and P @ P = P.
    """
    sd = S_full.shape[1]
    SSt = S_full @ S_full.T
    return torch.eye(sd) - S_full.T @ torch.linalg.inv(SSt) @ S_full


def phsde_step(model, x, u, rhythmic_indices, bio_graph, dt, temperature,
               projector=None, generator=None):
    """One Euler-Maruyama step of the port-Hamiltonian Langevin SDE.

    Parameters
    ----------
    temperature : sampling temperature T (sets noise amplitude sqrt(2 T R)).
    projector   : optional (state_dim, state_dim) conservation null-space
                  projector; if given, the noise increment is projected so the
                  conserved moiety pools are preserved.
    generator   : optional torch.Generator for reproducible noise.

    Returns
    -------
    x_next : (1, state_dim) next state.
    H      : float energy at the current state.
    """
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        dx, H, _sub_H, _grad_H = model(x, u, rhythmic_indices, bio_graph)
    dx = dx.detach()
    H_val = float(H.detach())
    R_diag = torch.diagonal(model._last_R.detach()[0]).clamp(min=0.0)
    sigma = torch.sqrt(2.0 * temperature * R_diag)
    dW = torch.randn(x.shape, generator=generator) * np.sqrt(dt)
    noise = sigma.unsqueeze(0) * dW
    if projector is not None:
        noise = noise @ projector.T
    return x.detach() + dx * dt + noise, H_val


def phsde_rollout(model, x0, u_seq, rhythmic_indices, bio_graph, dt, n_steps,
                  temperature, projector=None, seed=0):
    """Integrate the pH-Langevin SDE forward.

    Parameters
    ----------
    x0          : (1, state_dim) initial condition.
    u_seq       : (n_steps, n_ports) time-varying, or (1, n_ports)/(n_ports,)
                  constant port drive.
    temperature : sampling temperature T.
    projector   : optional conservation null-space projector.
    seed        : RNG seed for reproducible noise.

    Returns
    -------
    traj : (n_steps + 1, state_dim) tensor of states.
    H    : (n_steps,) array of energies.
    """
    g = torch.Generator().manual_seed(seed)
    xs = [x0.clone()]
    H_series = []
    x = x0.clone()

    def get_u(s):
        if u_seq.dim() == 2 and u_seq.shape[0] > 1:
            return u_seq[s:s + 1]
        return u_seq.reshape(1, -1)

    for s in range(n_steps):
        x, H = phsde_step(model, x, get_u(s), rhythmic_indices, bio_graph, dt,
                          temperature, projector=projector, generator=g)
        xs.append(x.clone())
        H_series.append(H)
    return torch.cat(xs, 0), np.asarray(H_series)
