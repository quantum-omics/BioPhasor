"""
integrator.py  —  Deterministic multi-step integrator and rollout for the
trained port-Hamiltonian cell model (Generic_pHNN).

The trained model provides the vector field

    dx/dt = (J(x) - R(x)) grad H(x) + G u + modulated_port(x)

which is integrated forward in time.  Because the model enforces R(x) >= 0
by construction, autonomous (u = 0) rollouts satisfy the passivity inequality
Hdot = grad H^T dx <= 0 at every step and remain bounded.

Functions
---------
vector_field : evaluate dx/dt (and optionally H, grad H, R) at a state.
rk4_step     : one fourth-order Runge-Kutta step.
rollout      : integrate n_steps forward, optionally monitoring energy and Hdot.
"""

from __future__ import annotations
import numpy as np
import torch


def vector_field(model, x, u, rhythmic_indices, bio_graph, return_energy=False):
    """Evaluate the learned vector field at state x.

    Parameters
    ----------
    model : Generic_pHNN
    x     : (B, state_dim) tensor
    u     : (B, n_ports) tensor
    return_energy : if True also return (H, grad H, R) detached.
    """
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        dx, H, _sub_H, grad_H = model(x, u, rhythmic_indices, bio_graph)
    if return_energy:
        R = model._last_R.detach()
        return dx.detach(), H.detach(), grad_H.detach(), R
    return dx.detach()


def rk4_step(model, x, u, rhythmic_indices, bio_graph, dt):
    """One classical RK4 step of the deterministic dynamics."""
    k1 = vector_field(model, x, u, rhythmic_indices, bio_graph)
    k2 = vector_field(model, x + 0.5 * dt * k1, u, rhythmic_indices, bio_graph)
    k3 = vector_field(model, x + 0.5 * dt * k2, u, rhythmic_indices, bio_graph)
    k4 = vector_field(model, x + dt * k3, u, rhythmic_indices, bio_graph)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def _select_u(u_seq, s):
    """Return the (1, n_ports) port input at step s.

    u_seq may be time-varying (n_steps, n_ports) or a constant (1, n_ports)
    / (n_ports,) drive that is held for the whole rollout.
    """
    if u_seq.dim() == 2 and u_seq.shape[0] > 1:
        return u_seq[s:s + 1]
    return u_seq.reshape(1, -1)


def rollout(model, x0, u_seq, rhythmic_indices, bio_graph, dt, n_steps,
            monitor=True):
    """Deterministic forward rollout.

    Parameters
    ----------
    x0      : (1, state_dim) initial condition.
    u_seq   : (n_steps, n_ports) time-varying, or (1, n_ports)/(n_ports,)
              constant port drive.
    dt      : timestep (hours).
    n_steps : number of integration steps.
    monitor : if True, also return the energy series H(t) and the power
              series Hdot(t) = grad H^T dx (<= 0 when u = 0).

    Returns
    -------
    traj  : (n_steps + 1, state_dim) tensor of states.
    H     : (n_steps,) array of energies (empty if monitor is False).
    Hdot  : (n_steps,) array of instantaneous powers (empty if False).
    """
    xs = [x0.clone()]
    H_series, Hdot_series = [], []
    x = x0.clone()
    for s in range(n_steps):
        u = _select_u(u_seq, s)
        if monitor:
            dx, H, grad_H, _R = vector_field(
                model, x, u, rhythmic_indices, bio_graph, return_energy=True)
            H_series.append(float(H))
            Hdot_series.append(float((grad_H * dx).sum()))
        x = rk4_step(model, x, u, rhythmic_indices, bio_graph, dt)
        xs.append(x.clone())
    return torch.cat(xs, 0), np.asarray(H_series), np.asarray(Hdot_series)
