"""
losses.py

Physics-Informed Composite Loss for the Multi-Omics GNN-pHNN.

Design notes
──────────────────────────
1. L_kinematic targets ABUNDANCE TRAJECTORY MSE (not phase-derivative MSE).
   The model is now evaluated on whether it predicts dq/dt correctly —
   the measurable, biologically interpretable quantity.

2. L_coherence (was L_plv): PLV is now a WEAK PRIOR (λ = 0.01), NOT a
   fitting target.  The circular training of J toward a statistic of the
   same data is broken.  Recovery of PLV structure is now a testable
   OUTCOME of the learned connectome, not a guaranteed result.

3. L_conservation (new): penalizes violation of stoichiometric moiety
   conservation.  |S q(t)|² should be constant.  This enforces the
   biochemical constraint that the model's predicted trajectory preserves
   the adenylate pool, redox pool, and cofactor pool.

4. L_homeostasis (new): soft constraint H(q=0) ≈ 0.  Ensures the storage
   function is minimized at the homeostatic set-point.

5. L_passivity: unchanged in structure; now computed over abundance block
   of the state.

6. L_balance: pH power balance, unchanged.

Loss weights (λ) — with biological interpretation:
  λ_kinematic    : 1.0   — primary fit objective (abundance trajectory)
  λ_passivity    : 0.5   — thermodynamic invariant (second law)
  λ_balance      : 0.3   — pH power identity
  λ_coherence    : 0.01  — PLV prior (weak)
  λ_conservation : 0.2   — stoichiometric invariant (mass balance)
  λ_homeostasis  : 0.1   — Lyapunov minimum at homeostasis

Design reference: 4-Regorous.ipynb §3.6, §6.1
"""

import torch
import torch.nn.functional as F
from typing import Optional


# Default loss weights
LOSS_WEIGHTS = {
    "kinematic":       1.0,
    "passivity":       0.5,
    "passivity_comp":  0.25,   # per-compartment passivity (Phase C)
    "balance":         0.3,
    "coherence":       0.01,   # ← weak prior
    "conservation":    0.2,
    "homeostasis":     0.1,
}


def loss_kinematic(
    dx_dt_pred: torch.Tensor,   # (B, state_dim)
    dx_dt_true: torch.Tensor,   # (B, state_dim)
) -> torch.Tensor:
    """
    Abundance trajectory kinematic loss.
    Primary fit objective: predicted state derivative vs. observed.
    MSE on the full state derivative (abundance + phasor if present).
    """
    return F.mse_loss(dx_dt_pred, dx_dt_true)


def loss_passivity(
    H:       torch.Tensor,   # (B, 1) Hamiltonian
    nabla_H: torch.Tensor,   # (B, state_dim)
    R:       torch.Tensor,   # (B, state_dim, state_dim)
) -> torch.Tensor:
    """
    Passivity invariant: Ḣ|_{u=0} ≤ 0  ←→  ∇H^T R ∇H ≥ 0.

    Penalty: max(0, -∇H^T R ∇H)  →  should be 0 for PSD R.
    Since R is PSD by construction (softplus), this loss monitors
    numerical violations and can tighten training if they occur.
    """
    # Compute dissipation rate: ∇H^T R ∇H ≥ 0 (should be ≥ 0)
    Rg = torch.bmm(R, nabla_H.unsqueeze(-1)).squeeze(-1)   # (B, sd)
    dissipation = (nabla_H * Rg).sum(dim=-1)                # (B,) ≥ 0 if R PSD
    return F.relu(-dissipation).mean()


def loss_passivity_per_compartment(
    nabla_H:  torch.Tensor,   # (B, state_dim)
    R:        torch.Tensor,   # (B, state_dim, state_dim)
    comp_id:  torch.Tensor,   # (N,) long compartment id per abundance node
    N_total:  int,
) -> torch.Tensor:
    """
    Per-compartment passivity penalty (Phase C).

    A composite pH system is passive iff each compartment dissipates:
    restricting ∇H and R to a compartment's abundance-node block, the local
    dissipation ∇H_c^T R_cc ∇H_c must be ≥ 0.  Penalize max(0, -diss_c) summed
    over compartments — a tighter thermodynamic check than the global one.

    Only the abundance block (first N_total dims) carries compartment identity.
    """
    gH_q = nabla_H[:, :N_total]                       # (B, N)
    R_qq = R[:, :N_total, :N_total]                   # (B, N, N)
    dev  = nabla_H.device
    total = torch.zeros((), device=dev)
    for cid in torch.unique(comp_id):
        idx = (comp_id == cid).nonzero(as_tuple=True)[0].to(dev)
        if idx.numel() == 0:
            continue
        gH_c = gH_q[:, idx]                            # (B, n_c)
        R_c  = R_qq[:, idx][:, :, idx]                 # (B, n_c, n_c)
        Rg   = torch.bmm(R_c, gH_c.unsqueeze(-1)).squeeze(-1)
        diss_c = (gH_c * Rg).sum(dim=-1)              # (B,)
        total = total + F.relu(-diss_c).mean()
    return total


def loss_power_balance(
    H:       torch.Tensor,   # (B, 1) scalar H
    nabla_H: torch.Tensor,   # (B, state_dim)
    J:       torch.Tensor,   # (B, state_dim, state_dim)
    R:       torch.Tensor,   # (B, state_dim, state_dim)
    G:       torch.Tensor,   # (N_total, n_ports)  port matrix
    u:       torch.Tensor,   # (B, n_ports)
    N_total: int,
) -> torch.Tensor:
    """
    Port-Hamiltonian power balance:
      Ḣ = -∇H^T R ∇H + y^T u
    where y = G^T ∇H_abundance.

    Penalize |Ḣ - (-diss + port_power)|.
    """
    # Dissipation
    Rg          = torch.bmm(R, nabla_H.unsqueeze(-1)).squeeze(-1)
    diss        = (nabla_H * Rg).sum(dim=-1, keepdim=True)   # (B, 1) ≥ 0

    # Port power: y = G^T ∇H_q  (only abundance part)
    nabla_H_q   = nabla_H[:, :N_total]                        # (B, N)
    y           = torch.matmul(nabla_H_q, G)                  # (B, n_ports)
    port_power  = (y * u).sum(dim=-1, keepdim=True)           # (B, 1)

    # Predicted Ḣ = -diss + port_power
    H_dot_pred  = -diss + port_power                          # (B, 1)

    # Actual Ḣ: would require d/dt of H along trajectory.
    # Approximation: use J contribution (should be zero for skew J)
    Jg          = torch.bmm(J, nabla_H.unsqueeze(-1)).squeeze(-1)
    j_contrib   = (nabla_H * Jg).sum(dim=-1, keepdim=True)   # should ≈ 0
    # Loss: J contribution should be zero (skew-symmetric); balance loss
    return F.mse_loss(j_contrib, torch.zeros_like(j_contrib)) + \
           0.3 * F.relu(-port_power + diss).mean()


def loss_coherence_prior(
    J_blocks:   dict,        # from Sparse_Dynamic_J_Net.forward
    plv_GP:     Optional[torch.Tensor] = None,  # (n_G_rhythmic, n_P_rhythmic)
    n_G:        int = 40,
    n_P:        int = 35,
) -> torch.Tensor:
    """
    Weak PLV coherence prior on the G↔P connectome block.

    This is a REGULARIZER, not a fitting target:
      L_coherence = λ * ||J_GP[rhythmic pairs] - PLV_GP||²
    with λ = LOSS_WEIGHTS['coherence'] = 0.01.

    By using a small λ, the J_GP entries are biased toward phase-coherent
    pairs but are NOT forced to match PLV.  Whether the learned connectome
    recovers PLV structure is now a testable outcome, not guaranteed.

    Returns zero if no PLV prior is provided.
    """
    if plv_GP is None or "J_GP" not in J_blocks:
        return torch.zeros(1, device=next(iter(J_blocks.values())).device
                           if J_blocks else torch.device('cpu')).squeeze()

    J_GP = J_blocks["J_GP"]   # (B, n_G, n_P)
    plv  = plv_GP.to(J_GP.device)

    # Scale J_GP to [0, 1] range for comparison with PLV
    J_abs     = J_GP.abs().mean(dim=0)         # (n_G, n_P)
    J_scaled  = J_abs / (J_abs.max() + 1e-8)  # (n_G, n_P)

    # Restrict to full-layer PLV (may be different size from rhythmic-only)
    min_G = min(J_scaled.size(0), plv.size(0))
    min_P = min(J_scaled.size(1), plv.size(1))

    return F.mse_loss(J_scaled[:min_G, :min_P], plv[:min_G, :min_P])


def loss_conservation(
    dx_dt_pred: torch.Tensor,   # (B, state_dim)
    S:          torch.Tensor,   # (n_moieties, n_M)  stoichiometric matrix
    n_G:        int,
    n_P:        int,
) -> torch.Tensor:
    """
    Stoichiometric moiety conservation penalty.

    For conserved moieties, the stoichiometric null-space condition requires:
      S * (dq_M/dt) = 0  for all t

    If this is violated, the predicted dynamics violate mass balance.
    We penalize |S * dq_M|² where dq_M is the metabolome abundance derivative.

    Parameters
    ----------
    dx_dt_pred : predicted state derivative (B, state_dim)
    S          : stoichiometric matrix (n_moieties, n_M)
    n_G, n_P   : node counts for indexing into state
    """
    n_M    = S.size(1)
    n_G_P  = n_G + n_P

    # Extract metabolome abundance-flux derivative block.
    # NOTE: dx_dt_pred is in NORMALIZED units (divided by scale_dx during state assembly).
    # S was built from integer stoichiometric coefficients (unitless ratios).
    # The constraint S @ dq_M = 0 is scale-invariant: if dq_M → dq_M / scale_dx,
    # then S @ (dq_M / scale_dx) = 0  iff  S @ dq_M = 0.
    # Therefore we can use the normalized derivative directly without rescaling S.
    dq_M   = dx_dt_pred[:, n_G_P:n_G_P + n_M]    # (B, n_M)  normalized fluxes

    # Apply stoichiometric constraint: S * dq_M should be 0
    Sdq    = torch.matmul(dq_M, S.T.to(dq_M.device))  # (B, n_moieties)
    return (Sdq ** 2).mean()


def loss_homeostasis(
    H:     torch.Tensor,   # (B, 1) evaluated at x
    x:     torch.Tensor,   # (B, state_dim)
    N:     int,
) -> torch.Tensor:
    """
    Soft Lyapunov constraint: H ≈ 0 when abundance q ≈ 0.

    We penalize H evaluated at states where the abundance deviation
    (x[:, :N]) is small — the model should recognize these as homeostasis.

    |q|² < threshold → H should be near 0.
    """
    q_sq   = (x[:, :N] ** 2).sum(dim=-1, keepdim=True)   # (B, 1)
    # Weight: high weight when q is near zero
    weight = torch.exp(-10.0 * q_sq)                      # (B, 1) ∈ (0, 1]
    return (weight * H ** 2).mean()


def compute_composite_loss(
    dx_dt_pred:  torch.Tensor,
    dx_dt_true:  torch.Tensor,
    H:           torch.Tensor,
    nabla_H:     torch.Tensor,
    J:           torch.Tensor,
    R:           torch.Tensor,
    J_blocks:    dict,
    G:           torch.Tensor,
    u:           torch.Tensor,
    S:           torch.Tensor,
    x:           torch.Tensor,
    N_total:     int,
    n_G:         int,
    n_P:         int,
    plv_GP:      Optional[torch.Tensor] = None,
    weights:     Optional[dict] = None,
    comp_id:     Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    """
    Full composite loss for the Cellular GNN-pHNN.

    Returns
    -------
    total_loss : scalar
    loss_dict  : dict of individual loss terms (for logging)
    """
    w = LOSS_WEIGHTS.copy()
    if weights:
        w.update(weights)

    L_kin  = loss_kinematic(dx_dt_pred, dx_dt_true)
    L_pass = loss_passivity(H, nabla_H, R)
    L_bal  = loss_power_balance(H, nabla_H, J, R, G, u, N_total)
    L_coh  = loss_coherence_prior(J_blocks, plv_GP, n_G, n_P)
    L_cons = loss_conservation(dx_dt_pred, S, n_G, n_P)
    L_home = loss_homeostasis(H, x, N_total)

    # Per-compartment passivity (Phase C) — zero if compartment map not provided
    if comp_id is not None:
        L_pass_c = loss_passivity_per_compartment(nabla_H, R, comp_id, N_total)
    else:
        L_pass_c = torch.zeros((), device=nabla_H.device)

    total = (
        w["kinematic"]      * L_kin  +
        w["passivity"]      * L_pass +
        w["passivity_comp"] * L_pass_c +
        w["balance"]        * L_bal  +
        w["coherence"]      * L_coh  +   # weak prior only
        w["conservation"]   * L_cons +
        w["homeostasis"]    * L_home
    )

    loss_dict = {
        "total":          total.item(),
        "kinematic":      L_kin.item(),
        "passivity":      L_pass.item(),
        "passivity_comp": float(L_pass_c.item()),
        "balance":        L_bal.item(),
        "coherence":      L_coh.item(),   # should drop: not a built-in guarantee
        "conservation":   L_cons.item(),
        "homeostasis":    L_home.item(),
    }
    return total, loss_dict
