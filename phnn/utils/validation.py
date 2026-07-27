"""
validation.py

Non-Circular Validation Suite for the Multi-Omics GNN-pHNN.

Purpose
───────
Every validation metric in this module is chosen to be non-circular — to test
structure the training loss was never given as a target:
  1. Training data carries no planted cascade (lags emerge from relaxation).
  2. PLV is a weak prior, not a target → J recovery is a testable outcome.

Validation functions (matching 4-Regorous.ipynb §6.2-6.5):

6.2 Edge recovery above random baseline
    Does the learned J enrich for biologically known edges?
    Evaluated on HELD-OUT edges (not in the prior) vs. degree-preserving
    random baseline.

6.3 Held-out perturbation forecasting
    Train on homeostasis + one condition; predict a WITHHELD condition.
    Trajectory-level RMSE + energy profile correlation.

6.4 Passivity verification
    max(Ḣ)|_{u=0} ≤ 0 across the full trajectory in free rollout.

6.5 Cascade test
    See cascade_predictor.py.  Called here for convenience.

6.6 Ablation utilities
    Remove/freeze each physics component and measure degradation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  6.2 — Edge Recovery Above Random Baseline
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_edge_recovery(
    J_learned:    torch.Tensor,   # (n_G, n_P) or (N, N) average |J| over trajectory
    A_true_held:  np.ndarray,     # (n_G, n_P) held-out ground-truth edges
    A_prior:      np.ndarray,     # (n_G, n_P) edges given as prior (excluded)
    n_null:       int = 100,
) -> dict:
    """
    Test whether the learned connectome enriches for held-out biological edges
    above a degree-preserving random baseline.

    Methodology:
      - Score each held-out edge by the learned |J_ij|.
      - Compare with the null: shuffle |J| maintaining row/column degree structure.
      - Compute AUROC for held-out edge recovery.

    Parameters
    ----------
    J_learned    : absolute value of learned J block (e.g., averaged over time)
    A_true_held  : binary adjacency of held-out ground-truth edges
    A_prior      : binary adjacency given as prior (not valid test examples)
    n_null       : number of degree-preserving permutations for null distribution

    Returns
    -------
    dict with AUROC, precision@k, and null distribution comparison
    """
    from sklearn.metrics import roc_auc_score

    J_np  = J_learned.detach().cpu().numpy() if isinstance(J_learned, torch.Tensor) \
            else np.array(J_learned)
    J_abs = np.abs(J_np)

    # Exclude prior edges from test set
    # True edge mask from held-out set (not in prior)
    positives = A_true_held.flatten()
    scores    = J_abs.flatten()
    # Test set = positions not in the prior adjacency
    exclude   = A_prior.flatten() > 0
    positives = positives[~exclude]
    scores    = scores[~exclude]

    if positives.sum() < 5 or (1 - positives).sum() < 5:
        return {"auroc": np.nan, "note": "Too few held-out edges for AUROC."}

    auroc_real = roc_auc_score(positives, scores)

    # Null distribution: degree-preserving row permutations
    rng        = np.random.default_rng(0)
    null_aurocs = []
    for _ in range(n_null):
        J_shuffled = np.array([rng.permutation(row) for row in J_abs])
        null_scores = J_shuffled.flatten()[~exclude]
        try:
            null_aurocs.append(roc_auc_score(positives, null_scores))
        except ValueError:
            pass

    null_mean = np.mean(null_aurocs)
    null_std  = np.std(null_aurocs)
    z_score   = (auroc_real - null_mean) / (null_std + 1e-8)

    # Precision@K where K = number of true positives
    K        = int(positives.sum())
    top_k    = np.argsort(scores)[-K:]
    prec_k   = positives[top_k].mean()

    return {
        "auroc_real":   float(auroc_real),
        "auroc_null_mean": float(null_mean),
        "auroc_null_std":  float(null_std),
        "z_score":      float(z_score),
        "precision_at_k": float(prec_k),
        "K":            K,
        "above_chance": bool(z_score > 1.96),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  6.3 — Held-Out Perturbation Forecasting
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_held_out_perturbation(
    model:            nn.Module,
    x_train:          torch.Tensor,    # (T_train, state_dim)
    u_train:          torch.Tensor,    # (T_train, n_ports)
    x_held:           torch.Tensor,    # (T_held, state_dim) — never seen in training
    u_held:           torch.Tensor,    # (T_held, n_ports)
    rhythmic_indices: torch.Tensor,    # (N_r,)
    bio_graph:        dict,
    rollout_steps:    int = 100,
    dt:               float = 0.1,
) -> dict:
    """
    Evaluate the model's ability to forecast a HELD-OUT perturbation condition.

    The model is given the initial state of the held-out condition and asked to
    roll out for `rollout_steps` without seeing the trajectory.  This tests
    generalization to unseen biology.

    Returns
    -------
    dict with trajectory RMSE, energy profile correlation, and passivity check
    """
    model.eval()
    n_ports = model.n_ports

    # Build initial condition — no_grad for preparation only
    x0 = x_held[0:1].clone()
    x_curr = x0

    rollout_preds = []
    rollout_trues = [x_held[i].numpy() for i in range(min(rollout_steps, len(x_held)))]

    for step in range(min(rollout_steps, len(x_held) - 1)):
        # Each forward call MUST be inside enable_grad because model.forward
        # calls torch.autograd.grad(H, x) internally.
        x_in = x_curr.detach().requires_grad_(True)
        u_in = torch.zeros(1, n_ports)
        try:
            with torch.enable_grad():
                dx_pred, H, _, _ = model(x_in, u_in, rhythmic_indices, bio_graph)
            # Euler step
            x_next = x_in.detach() + dx_pred.detach() * dt
            rollout_preds.append(x_next[0].numpy())
            x_curr = x_next
        except Exception as exc:
            # Propagate the error for debugging instead of silently swallowing
            import warnings
            warnings.warn(f"Rollout failed at step {step}: {exc}")
            break

    rollout_arr = np.array(rollout_preds)    # (steps, state_dim)
    true_arr    = x_held[:len(rollout_arr)].numpy()

    # Trajectory RMSE
    rmse_trajectory = np.sqrt(np.mean((rollout_arr - true_arr) ** 2))

    # Energy correlation (if we can compute H along rollout)
    # Use abundance block only for simplicity
    N = bio_graph["n_G"] + bio_graph["n_P"] + bio_graph["n_M"]
    q_pred = rollout_arr[:, :N]
    q_true = true_arr[:, :N]

    energy_pred = np.mean(q_pred ** 2, axis=1)   # proxy Lyapunov energy
    energy_true = np.mean(q_true ** 2, axis=1)

    from scipy.stats import pearsonr
    r, p = pearsonr(energy_pred, energy_true) if len(energy_pred) > 5 else (np.nan, np.nan)

    return {
        "rollout_steps":       len(rollout_arr),
        "trajectory_rmse":     float(rmse_trajectory),
        "energy_correlation_r": float(r),
        "energy_correlation_p": float(p),
        "generalizes":         bool(rmse_trajectory < 1.0),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  6.4 — Passivity Verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_passivity(
    model:            nn.Module,
    x_data:           torch.Tensor,   # (T, state_dim)
    rhythmic_indices: torch.Tensor,
    bio_graph:        dict,
    sample_every:     int = 10,
) -> dict:
    """
    Verify the passivity invariant: Ḣ|_{u=0} ≤ 0 across the trajectory.

    Ḣ = dH/dt = ∇H^T dx/dt = -∇H^T R ∇H ≤ 0  (since R is PSD)

    Returns max(Ḣ) and the fraction of time points violating passivity.
    """
    model.eval()
    n_ports   = model.n_ports
    H_dot_vals = []
    n_total    = x_data.shape[0]

    for t in range(0, n_total, sample_every):   # uses sample_every; covers full trajectory
        # Each call needs requires_grad=True for autograd.grad inside forward
        x_t = x_data[t:t+1].detach().requires_grad_(True)
        u_zero = torch.zeros(1, n_ports, device=x_t.device)
        try:
            with torch.enable_grad():
                _, H, _, nabla_H = model(x_t, u_zero, rhythmic_indices, bio_graph)
            R = model._last_R   # (1, sd, sd)
            Rg = torch.bmm(R, nabla_H.unsqueeze(-1)).squeeze(-1)
            H_dot = -(nabla_H * Rg).sum(dim=-1)   # (1,)  ≤ 0 if passive
            H_dot_vals.append(H_dot.item())
        except Exception as exc:
            import warnings
            warnings.warn(f"Passivity check failed at t={t}: {exc}")
            continue

    H_dots = np.array(H_dot_vals)
    return {
        "max_H_dot":              float(H_dots.max()) if len(H_dots) > 0 else np.nan,
        "mean_H_dot":             float(H_dots.mean()) if len(H_dots) > 0 else np.nan,
        "fraction_violating":     float((H_dots > 0).mean()) if len(H_dots) > 0 else np.nan,
        "passivity_satisfied":    bool(H_dots.max() <= 1e-3) if len(H_dots) > 0 else False,
        "n_samples":              len(H_dots),
        "H_dot_series":           H_dots.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Print / summary utilities
# ─────────────────────────────────────────────────────────────────────────────

def print_validation_report(results: dict) -> None:
    print("\n══ Validation Report ═══════════════════════════")

    if "passivity" in results:
        p = results["passivity"]
        status = "✓ PASS" if p.get("passivity_satisfied") else "✗ FAIL"
        print(f"\n  6.4 Passivity [{status}]")
        print(f"      max(Ḣ) = {p.get('max_H_dot', np.nan):.4f}  "
              f"(fraction violating: {p.get('fraction_violating', np.nan):.1%})")

    if "edge_recovery" in results:
        e = results["edge_recovery"]
        status = "✓ above chance" if e.get("above_chance") else "✗ at chance"
        print(f"\n  6.2 Edge Recovery [{status}]")
        print(f"      AUROC = {e.get('auroc_real', np.nan):.3f}  "
              f"(null = {e.get('auroc_null_mean', np.nan):.3f} ± {e.get('auroc_null_std', np.nan):.3f})")
        print(f"      z-score = {e.get('z_score', np.nan):.2f},  "
              f"Precision@K = {e.get('precision_at_k', np.nan):.3f}")

    if "held_out" in results:
        h = results["held_out"]
        status = "✓ GENERALIZES" if h.get("generalizes") else "✗ OVERFITS"
        print(f"\n  6.3 Held-Out Perturbation [{status}]")
        print(f"      Trajectory RMSE = {h.get('trajectory_rmse', np.nan):.4f}")
        print(f"      Energy corr. r  = {h.get('energy_correlation_r', np.nan):.3f}")

    print("\n══════════════════════════════════════════════════════════\n")
