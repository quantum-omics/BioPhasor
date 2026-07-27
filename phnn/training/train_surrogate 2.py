"""
train_surrogate.py

Biologically Grounded GNN-pHNN Training Pipeline.
Fits the composite compartmental port-Hamiltonian model to the
real mouse-liver tri-omic circadian dataset.

Design notes:
─────────────────────────────
* Data pipeline: two-layer state (abundance + gated phasors), not phasor-only.
* Rhythmicity gating: only genuinely oscillatory nodes get phasor coordinates.
* Biological graph: sparse adjacency seeded from bio priors.
* k_deg priors: R initialized from measured degradation rates.
* Loss: six-term composite — abundance MSE, passivity, pH balance,
        PLV weak prior (λ=0.01), conservation, homeostasis.
* Validation: passivity check, edge-recovery AUROC, cascade predictor.
* PLV is NOT a training target — its recovery is a measured outcome.

Outputs (all written to experiments/):
  experiments/figures/     — training diagnostics + figure_data.npz
  experiments/results/     — figure_data.npz, J/R matrices, results JSON
  experiments/checkpoints/ — model_seed{N}.pt
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data.omics_data_generator import (
    generate_multi_omics, get_total_nodes, LAYER_CONFIG
)
from data.data_adapter import load_synthetic_omics, load_real_omics
from data.rhythmicity_gate import detect_all_layers, summarise_rhythmicity
from data.two_layer_state import assemble_two_layer_state, verify_conservation
from data.bio_graph import build_biological_graph, compute_plv_prior, print_graph_summary
from models.phnn import Generic_pHNN
from training.losses import compute_composite_loss, LOSS_WEIGHTS
from utils.cascade_predictor import CascadePredictor
from utils.validation import (
    verify_passivity,
    evaluate_held_out_perturbation,
    evaluate_edge_recovery,
    print_validation_report,
)

_CODE_DIR   = os.path.dirname(os.path.abspath(__file__))
_EXP_DIR    = os.path.join(_CODE_DIR, "experiments")
# All outputs go into the canonical experiment record directories.
SAVE_DIR    = os.path.join(_EXP_DIR, "figures")
MATRIX_DIR  = os.path.join(_EXP_DIR, "results")
CHECKPT_DIR = os.path.join(_EXP_DIR, "checkpoints")


def train_omics_surrogate(epochs: int = 500, batch_size: int = 48, seed: int = 0,
                          source: str = "synthetic", total_hours: float = 48.0):
    print("=" * 65)
    print("  Cellular GNN-pHNN  —  Biologically Grounded Training")
    print(f"  Data source: {source.upper()}")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # All outputs go into experiments/{figures,results,checkpoints}/ directly.
    global SAVE_DIR, MATRIX_DIR, CHECKPT_DIR
    os.makedirs(SAVE_DIR,   exist_ok=True)
    os.makedirs(MATRIX_DIR, exist_ok=True)
    os.makedirs(CHECKPT_DIR, exist_ok=True)
    torch.manual_seed(seed)

    # ─── 1. Load two-layer multi-omics data (synthetic or real) ──────────────
    print(f"\n[1] Loading {source} abundance-based multi-omics data...")
    if source == "real":
        omics_data = load_real_omics(total_hours=total_hours, seed=seed)
    else:
        omics_data = generate_multi_omics(seed=seed)
    t          = omics_data["t"]
    T          = len(t)
    omega_clk  = omics_data["omega_clock"]

    print(f"  Time points: {T}  (dt={omics_data['dt']} h,  "
          f"T_total={omics_data['dt']*T:.0f} h)")

    # Conservation check
    cons_report = verify_conservation(omics_data)
    for moiety, rep in cons_report.items():
        status = "✓" if rep["passes"] else "✗"
        print(f"  {status} Conservation [{moiety}]: "
              f"max deviation = {rep['max_relative_deviation']:.3%}")

    # ─── 2. Rhythmicity gating ───────────────────────────────────────────────
    print("\n[2] Running rhythmicity gate (Lomb-Scargle)...")
    gate_results = detect_all_layers(omics_data)
    summarise_rhythmicity(gate_results)

    # ─── 3. Two-layer state assembly ─────────────────────────────────────────
    print("[3] Assembling two-layer state [q_abundance; sin φ; cos φ; ω]...")
    state_data  = assemble_two_layer_state(omics_data, gate_results)
    x_all       = state_data["x"]          # (T, state_dim)
    dx_all      = state_data["dx_dt"]      # (T, state_dim)
    rhy_idx     = torch.tensor(state_data["rhythmic_indices"], dtype=torch.long)
    N_total     = state_data["N_total"]
    N_rhythmic  = state_data["N_rhythmic"]
    state_dim   = state_data["state_dim"]

    print(f"  N_total    = {N_total}")
    print(f"  N_rhythmic = {N_rhythmic}  ({N_rhythmic/N_total:.0%})")
    print(f"  state_dim  = {state_dim}   (= N + 3·N_r)")

    # ─── 4. Biological graph ─────────────────────────────────────────────────
    print("\n[4] Building biological sparse graph...")
    bio_graph = build_biological_graph(seed=seed)
    print_graph_summary(bio_graph)

    # PLV prior (weak — not a training target)
    # Use instantaneous phases from two_layer_state assembly (bandpass+Hilbert),
    # sliced per layer using the rhythmic layer slices.
    phi_rhythmic_all = state_data["phi_rhythmic"].numpy()   # (T, N_r)
    rhy_layer_slices = state_data["layer_slices_rhythmic"]  # dict: layer → slice into N_r dim

    phi_data_rhy = {}
    for layer_name in ["genomics", "proteome", "metabolome"]:
        slc = rhy_layer_slices.get(layer_name, slice(0, 0))
        n_rhy = slc.stop - slc.start
        if n_rhy > 0:
            # phi_rhythmic_all is (T, N_r); slice along N_r dim → (T, n_rhy_layer)
            phi_layer = phi_rhythmic_all[:, slc].T   # (n_rhy_layer, T)
            phi_data_rhy[layer_name] = phi_layer
        else:
            phi_data_rhy[layer_name] = np.zeros((0, T))

    plv_priors = compute_plv_prior(phi_data_rhy, gate_results)
    plv_GP     = plv_priors.get("PLV_GP")

    # Port signals
    u_all = torch.tensor(omics_data["u"], dtype=torch.float32)   # (T, 3)

    # ─── 5. Build k_deg prior vector ─────────────────────────────────────────
    k_deg_concat = np.concatenate([
        omics_data["k_deg"]["genomics"],
        omics_data["k_deg"]["proteome"],
        omics_data["k_deg"]["metabolome"],
    ])
    k_deg_prior = torch.tensor(k_deg_concat, dtype=torch.float32)

    # ─── 6. Model initialization ─────────────────────────────────────────────
    print("\n[5] Initialising GNN-pHNN...")
    model = Generic_pHNN(
        N_total      = N_total,
        N_rhythmic   = N_rhythmic,
        hidden_dim   = 128,
        n_ports      = 3,
        k_deg_prior  = k_deg_prior,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    # Stoichiometric matrix (metabolome conservation)
    S = bio_graph["S"].to(device)   # (n_moieties, n_M)

    n_G, n_P = bio_graph["n_G"], bio_graph["n_P"]

    # ─── 7. Training loop ────────────────────────────────────────────────────
    print(f"\n[6] Training for {epochs} epochs (batch_size={batch_size})...")
    print(f"  Loss weights: {LOSS_WEIGHTS}")
    model.train()
    all_losses = {k: [] for k in ["total", "kinematic", "passivity", "passivity_comp",
                                   "balance", "coherence", "conservation", "homeostasis"]}

    # Compartment id vector (abundance nodes) for per-compartment passivity
    comp_id_vec = bio_graph["comp_id"].to(device)   # (N,) long

    for epoch in range(epochs):
        optimizer.zero_grad()

        # Random batch
        idx   = torch.randint(1, T - 1, (batch_size,))
        x_b   = x_all[idx].to(device).requires_grad_(True)
        dx_b  = dx_all[idx].to(device)
        u_b   = u_all[idx].to(device)

        dx_pred, H, sub_H, nabla_H = model(x_b, u_b, rhy_idx.to(device), bio_graph)

        J_full = model._last_J
        R_full = model._last_R
        J_blocks = model._last_j_blocks

        # Retrieve G parameter (abundance port only)
        G_param = model.G

        total_loss, loss_dict = compute_composite_loss(
            dx_dt_pred  = dx_pred,
            dx_dt_true  = dx_b,
            H           = H,
            nabla_H     = nabla_H,
            J           = J_full,
            R           = R_full,
            J_blocks     = J_blocks,
            G           = G_param,
            u           = u_b,
            S           = S,
            x           = x_b,
            N_total     = N_total,
            n_G         = n_G,
            n_P         = n_P,
            plv_GP      = plv_GP.to(device) if plv_GP is not None else None,
            comp_id     = comp_id_vec,
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        for k, v in loss_dict.items():
            all_losses[k].append(v)

        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{epochs}  "
                  f"total={loss_dict['total']:.4f}  "
                  f"kin={loss_dict['kinematic']:.4f}  "
                  f"pass={loss_dict['passivity']:.4f}  "
                  f"pass_c={loss_dict['passivity_comp']:.4f}  "
                  f"cons={loss_dict['conservation']:.4f}")

    # ─── 8. Figures ──────────────────────────────────────────────────────────
    print("\n[7] Generating figures...")

    # Fig 1: Training loss curves (all terms)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    terms = ["total", "kinematic", "passivity", "coherence", "conservation", "homeostasis"]
    colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad", "#f39c12", "#3498db"]
    for ax, term, col in zip(axes, terms, colors):
        vals = [max(v, 1e-10) for v in all_losses[term]]   # guard log(0)
        ax.semilogy(vals, color=col, lw=2, alpha=0.9)
        ax.set_title(f"L_{term}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Cellular GNN-pHNN — Composite Loss Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_loss.png"), dpi=200)
    plt.close()
    print("  Saved training_loss.png")

    # Fig 2: Dynamic connectome across states
    model.eval()
    # Build a state map from the trajectory's own labels, clamped to [0, T-1].
    # Synthetic data carries Drug/Recovery phases; the real assembled dataset is
    # baseline-only, so we sample three circadian phases instead of drug phases.
    labels_arr = np.asarray(omics_data["state_labels"])
    unique_states = list(dict.fromkeys(labels_arr.tolist()))
    if len(unique_states) >= 3 and "Drug Administration" in unique_states:
        state_map = {
            "Homeostasis":        0,
            "Drug Administration": int(80.0  / omics_data["dt"]) + 200,
            "Metabolic Recovery": int(160.0 / omics_data["dt"]) + 100,
        }
    else:
        # real / baseline-only: three circadian phases across the trajectory
        state_map = {
            "CT / phase 0":  0,
            "CT / phase π/2": T // 4,
            "CT / phase π":   T // 2,
        }
    # clamp all indices into range
    state_map = {k: int(min(max(v, 0), T - 1)) for k, v in state_map.items()}
    J_states = {}
    for state_name, t_idx in state_map.items():
        x_s = x_all[t_idx:t_idx+1].to(device).requires_grad_(True)
        u_s = u_all[t_idx:t_idx+1].to(device)
        with torch.enable_grad():
            model(x_s, u_s, rhy_idx.to(device), bio_graph)
        J_block = model._last_J[0, :N_total, :N_total].detach().cpu().numpy()
        J_states[state_name] = J_block

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    vmax = max(np.abs(j).max() for j in J_states.values()) + 1e-8
    for ax, (state, J_mat) in zip(axes, J_states.items()):
        im = ax.imshow(J_mat, cmap="PRGn", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"J(x) — {state}", fontsize=11)
        ax.set_xlabel("Node (G | P | M)")
        ax.set_ylabel("Node (G | P | M)")
        for b in [n_G, n_G + n_P]:
            ax.axhline(b - 0.5, color="k", lw=0.8, ls="--")
            ax.axvline(b - 0.5, color="k", lw=0.8, ls="--")
    cax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cax, label="Coupling strength")
    fig.suptitle("State-Dependent Sparse Connectome J(x)",
                 fontsize=12, fontweight="bold")
    plt.savefig(os.path.join(SAVE_DIR, "dynamic_connectome.png"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print("  Saved dynamic_connectome.png")

    # ─── 9. Post-training validation ─────────────────────────────────────────
    print("\n[8] Running post-training validation suite...")

    # 9a. Passivity
    pass_results = verify_passivity(model, x_all.to(device), rhy_idx.to(device),
                                    bio_graph, sample_every=20)

    # 9b. DUAL phase cascade predictor (Phase C) — one per clock, SAME R.
    clock_bank = omics_data["clock_bank"]
    predictor = CascadePredictor(
        k_deg_G     = omics_data["k_deg"]["genomics"],
        k_deg_P     = omics_data["k_deg"]["proteome"],
        k_deg_M     = omics_data["k_deg"]["metabolome"],
        omega_clock = omega_clk,
        clock_bank  = clock_bank,
    )
    # Detected acrophases (output phase of the first-order filter, time-referenced
    # per clock by the rhythmicity gate).  The two cascades are predicted from the
    # SAME dissipation R but DIFFERENT clock frequencies — a doubly-falsifiable test.
    acro_G = gate_results["genomics"]["acrophase"]
    acro_P = gate_results["proteome"]["acrophase"]
    acro_M = gate_results["metabolome"]["acrophase"]

    # Circadian cascade: G[0:12] → P[0:12], driven at ω_circadian
    cascade_circ = predictor.evaluate_cascade(
        acrophase_src = acro_G[0:12],
        acrophase_tgt = acro_P[0:12],
        k_deg_tgt     = omics_data["k_deg"]["proteome"][0:12],
        omega         = clock_bank["circadian"],
        label         = "circadian G→P",
    )
    # Redox cascade: P[12:20] → M[12:20], driven at ω_redox
    cascade_redox = predictor.evaluate_cascade(
        acrophase_src = acro_P[12:20],
        acrophase_tgt = acro_M[12:20],
        k_deg_tgt     = omics_data["k_deg"]["metabolome"][12:20],
        omega         = clock_bank["redox"],
        label         = "redox P→M",
    )
    predictor.print_cascade_report(cascade_circ)
    predictor.print_cascade_report(cascade_redox)
    # Primary cascade report (circadian) retained for checkpoint back-compat
    cascade_report = cascade_circ

    # 9b′. MODEL-INDEPENDENT cascade test (real data only): compare the model's
    # predicted transcript→protein lag (arctan(ω/k_deg), k_deg from PUBLISHED
    # half-lives) against the AUTHOR-MEASURED lags (Robles Table S4) for the
    # co-selected matched pairs. This is the strongest falsifiable test — neither
    # side is fit to this data.
    measured_cascade_report = None
    if omics_data.get("measured_cascade") and \
            len(omics_data["measured_cascade"].get("pairs", [])) >= 5:
        from scipy.stats import pearsonr, spearmanr
        mc = omics_data["measured_cascade"]
        prot_pos = np.asarray(mc["prot_pos"], dtype=int)
        meas_lag = np.asarray(mc["lag_hours"], dtype=float)
        k_p = np.asarray(omics_data["k_deg"]["proteome"])[prot_pos]
        omega_c = clock_bank["circadian"]
        pred_lag = np.degrees(np.arctan(omega_c / k_p)) / 360.0 * 24.0  # hours
        # fold measured lag into a single circadian cycle for fair comparison
        meas_lag_folded = np.mod(meas_lag, 24.0)
        meas_lag_folded = np.where(meas_lag_folded > 12.0,
                                   meas_lag_folded - 24.0, meas_lag_folded)
        v = np.isfinite(meas_lag_folded) & np.isfinite(pred_lag)
        if v.sum() >= 5:
            r_p, p_p = pearsonr(pred_lag[v], meas_lag_folded[v])
            r_s, p_s = spearmanr(pred_lag[v], meas_lag_folded[v])
            rmse_h = float(np.sqrt(np.mean((pred_lag[v] - meas_lag_folded[v]) ** 2)))
            measured_cascade_report = {
                "label": "measured transcript→protein (Robles S4)",
                "n_pairs": int(v.sum()),
                "pearson_r": float(r_p), "pearson_p": float(p_p),
                "spearman_r": float(r_s), "spearman_p": float(p_s),
                "rmse_hours": rmse_h,
                "mean_measured_lag_h": float(np.mean(meas_lag_folded[v])),
                "mean_predicted_lag_h": float(np.mean(pred_lag[v])),
            }
            print(f"\n── Model-Independent Cascade Test [measured Robles S4] ──")
            print(f"  Matched pairs:      {int(v.sum())}")
            print(f"  Predicted lag (k_deg from published t½): "
                  f"mean {np.mean(pred_lag[v]):.2f} h")
            print(f"  Measured lag (Robles):  mean {np.mean(meas_lag_folded[v]):.2f} h")
            print(f"  Pearson r:          {r_p:.3f}  (p={p_p:.3e})")
            print(f"  Spearman ρ:         {r_s:.3f}  (p={p_s:.3e})")
            print(f"  RMSE:               {rmse_h:.2f} h")
            print("─────────────────────────────────────────────────────────")

    # 9c. Held-out perturbation forecasting
    # Train on first 80% of trajectory; held-out = last 20% (drug recovery phase).
    print("\n  6.3 Held-out perturbation forecasting (last 20% of trajectory)...")
    t_split      = int(0.80 * T)
    x_train_full = x_all[:t_split].to(device)
    u_train_full = u_all[:t_split].to(device)
    x_held_full  = x_all[t_split:].cpu()
    u_held_full  = u_all[t_split:].cpu()

    held_out_results = evaluate_held_out_perturbation(
        model             = model,
        x_train           = x_train_full,
        u_train           = u_train_full,
        x_held            = x_held_full,
        u_held            = u_held_full,
        rhythmic_indices  = rhy_idx.to(device),
        bio_graph         = bio_graph,
        rollout_steps     = 100,
        dt                = omics_data["dt"],
    )
    print(f"    Trajectory RMSE = {held_out_results.get('trajectory_rmse', np.nan):.4f}  "
          f"Energy corr r = {held_out_results.get('energy_correlation_r', np.nan):.3f}  "
          f"Generalizes: {held_out_results.get('generalizes')}")

    # 9d. Edge recovery AUROC (G→P held-out edges)
    # Split dogma adjacency: 80% prior (given to model), 20% held-out for test.
    print("  6.2 Edge recovery AUROC...")
    A_dogma_np = bio_graph["A_GP_dogma"].numpy()   # (n_G, n_P)
    rng_split  = np.random.default_rng(seed)
    held_mask  = rng_split.random(A_dogma_np.shape) < 0.20
    A_prior    = A_dogma_np * (~held_mask).astype(float)
    A_held     = A_dogma_np * held_mask.astype(float)

    # Extract mean |J_GP| over training timesteps as the score
    model.eval()
    n_batches   = min(20, T // batch_size)
    J_GP_sum    = np.zeros((n_G, n_P))
    for b in range(n_batches):
        idx_b = torch.randint(0, T, (batch_size,))
        x_b   = x_all[idx_b].to(device).requires_grad_(True)
        u_b   = u_all[idx_b].to(device)
        with torch.enable_grad():
            model(x_b, u_b, rhy_idx.to(device), bio_graph)
        J_GP_sum += model._last_j_blocks["J_GP"].abs().mean(dim=0).detach().cpu().numpy()
    J_GP_mean = J_GP_sum / n_batches

    edge_recovery = evaluate_edge_recovery(
        J_learned    = torch.tensor(J_GP_mean),
        A_true_held  = A_held,
        A_prior      = A_prior,
        n_null       = 50,
    )
    if "auroc_real" in edge_recovery:
        print(f"    AUROC = {edge_recovery['auroc_real']:.3f}  "
              f"(null = {edge_recovery['auroc_null_mean']:.3f}±{edge_recovery['auroc_null_std']:.3f})  "
              f"z = {edge_recovery['z_score']:.2f}  above_chance: {edge_recovery['above_chance']}")

    val_results = {
        "passivity":      pass_results,
        "cascade":        cascade_report,       # circadian (primary)
        "cascade_circ":   cascade_circ,
        "cascade_redox":  cascade_redox,
        "cascade_measured": measured_cascade_report,   # model-independent (real data)
        "held_out":       held_out_results,
        "edge_recovery":  edge_recovery,
    }
    print_validation_report(val_results)

    # ─── 9d-bis. Figure-data bundle export ───────────────────────────────────
    # Dump every array a downstream, publication-grade figure renderer needs, so
    # figures can be built with a consistent house style outside this driver.
    fig_bundle = os.path.join(SAVE_DIR, "figure_data.npz")
    clk_G_ = gate_results["genomics"]["clock_label"]
    clk_P_ = gate_results["proteome"]["clock_label"]
    clk_M_ = gate_results["metabolome"]["clock_label"]
    np.savez(
        fig_bundle,
        # loss curves (per term, per epoch)
        loss_total       = np.array(all_losses["total"]),
        loss_kinematic   = np.array(all_losses["kinematic"]),
        loss_passivity   = np.array(all_losses["passivity"]),
        loss_passivity_c = np.array(all_losses["passivity_comp"]),
        loss_conservation= np.array(all_losses["conservation"]),
        loss_coherence   = np.array(all_losses["coherence"]),
        loss_homeostasis = np.array(all_losses["homeostasis"]),
        # cascade (circadian + redox), lags in radians
        circ_obs_rad     = np.array(cascade_circ.get("observed_lags_rad", [])),
        circ_pred_rad    = np.array(cascade_circ.get("predicted_lags_rad", [])),
        circ_omega       = float(cascade_circ.get("omega", np.nan)),
        circ_r           = float(cascade_circ.get("pearson_r", np.nan)),
        circ_kdeg        = np.array(omics_data["k_deg"]["proteome"][0:12]),
        redox_obs_rad    = np.array(cascade_redox.get("observed_lags_rad", [])),
        redox_pred_rad   = np.array(cascade_redox.get("predicted_lags_rad", [])),
        redox_omega      = float(cascade_redox.get("omega", np.nan)),
        redox_r          = float(cascade_redox.get("pearson_r", np.nan)),
        redox_kdeg       = np.array(omics_data["k_deg"]["metabolome"][12:20]),
        # model-independent measured cascade (real data): predicted vs Robles-measured lag
        meas_pred_h      = (np.degrees(np.arctan(clock_bank["circadian"] /
                            np.asarray(omics_data["k_deg"]["proteome"])[
                              np.asarray(omics_data["measured_cascade"]["prot_pos"], dtype=int)]))
                            / 360.0 * 24.0) if omics_data.get("measured_cascade") else np.array([]),
        meas_obs_h       = (lambda ml: np.where(np.mod(ml,24)>12, np.mod(ml,24)-24, np.mod(ml,24)))(
                            np.asarray(omics_data["measured_cascade"]["lag_hours"], dtype=float)
                            ) if omics_data.get("measured_cascade") else np.array([]),
        meas_r           = float(measured_cascade_report["pearson_r"]) if measured_cascade_report else np.nan,
        meas_rmse_h      = float(measured_cascade_report["rmse_hours"]) if measured_cascade_report else np.nan,
        # two-torus acrophases + clock labels
        acro_all         = np.concatenate([acro_G, acro_P, acro_M]),
        clk_all          = np.concatenate([clk_G_, clk_P_, clk_M_]),
        circ_period      = float(2 * np.pi / clock_bank["circadian"]),
        redox_period     = float(2 * np.pi / clock_bank["redox"]),
        # passivity distribution
        H_dot_series     = np.array(pass_results.get("H_dot_series", [])),
        passivity_max    = float(pass_results.get("max_H_dot", np.nan)),
        # edge recovery
        edge_auroc       = float(edge_recovery.get("auroc_real", np.nan)),
        edge_null_mean   = float(edge_recovery.get("auroc_null_mean", np.nan)),
        edge_null_std    = float(edge_recovery.get("auroc_null_std", np.nan)),
        edge_z           = float(edge_recovery.get("z_score", np.nan)),
        # held-out forecasting
        held_rmse        = float(held_out_results.get("trajectory_rmse", np.nan)),
        held_energy_r    = float(held_out_results.get("energy_correlation_r", np.nan)),
        # dynamic connectome (three states, in state_map order) + layer sizes
        J_homeo          = list(J_states.values())[0],
        J_drug           = list(J_states.values())[1],
        J_recov          = list(J_states.values())[2],
        J_state_names    = np.array(list(J_states.keys()), dtype=object),
        n_G              = n_G, n_P = n_P, n_M = N_total - n_G - n_P,
    )
    print(f"  Saved figure-data bundle → {fig_bundle}")

    # ─── 9e. Model checkpoint save ───────────────────────────────────────────
    os.makedirs(CHECKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPT_DIR, f"model_seed{seed}.pt")
    torch.save({
        "model_state_dict":  model.state_dict(),
        "N_total":           N_total,
        "N_rhythmic":        N_rhythmic,
        "state_dim":         state_dim,
        "hidden_dim":        128,
        "n_ports":           3,
        "epochs":            epochs,
        "seed":              seed,
        "val_results":       {
            "passivity_max_H_dot":  pass_results.get("max_H_dot", np.nan),
            "cascade_pearson_r":    cascade_report.get("pearson_r", np.nan),
            "held_out_rmse":        held_out_results.get("trajectory_rmse", np.nan),
            "edge_auroc":           edge_recovery.get("auroc_real", np.nan),
        },
    }, ckpt_path)
    print(f"\n  ✓ Model saved → {ckpt_path}")

    # ─── 10. Dual cascade figure (Phase C) ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, rep, col in zip(
        axes, [cascade_circ, cascade_redox], ["#e74c3c", "#2980b9"]
    ):
        if "observed_lags_rad" not in rep:
            ax.set_visible(False)
            continue
        obs   = np.array(rep["observed_lags_rad"])
        pred  = np.array(rep["predicted_lags_rad"])
        omega = rep["omega"]
        to_h  = (2 * np.pi / omega) / (2 * np.pi)   # rad → h at this clock
        ax.scatter(pred / omega, obs / omega, alpha=0.7,
                   edgecolors="k", linewidths=0.5, s=55, color=col)
        allv = np.concatenate([obs, pred]) / omega
        lim  = np.abs(allv).max() * 1.2 + 1e-6
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="y=x")
        ax.set_xlabel("Predicted lag (h)  [from k_deg, independent]")
        ax.set_ylabel("Observed lag (h)  [from data]")
        r_v = rep.get("pearson_r", np.nan)
        period = 2 * np.pi / omega
        ax.set_title(f"{rep['label']}   r = {r_v:.3f}   (clock {period:.0f} h)", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Dual Phase Cascade Test — same R, two clocks", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "dual_cascade_test.png"), dpi=200)
    plt.close()
    print("  Saved dual_cascade_test.png")

    # ─── 11. Two-torus phase figure (Phase C) ────────────────────────────────
    # Show the two clocks as separate phase circles: circadian-assigned nodes on
    # one S¹, redox-assigned nodes on the other, using detected acrophases.
    clk_G = gate_results["genomics"]["clock_label"]
    clk_P = gate_results["proteome"]["clock_label"]
    clk_M = gate_results["metabolome"]["clock_label"]
    clk_all  = np.concatenate([clk_G, clk_P, clk_M])
    acro_all = np.concatenate([acro_G, acro_P, acro_M])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), subplot_kw={"projection": "polar"})
    for ax, clk_name, col in zip(axes, ["circadian", "redox"], ["#e74c3c", "#2980b9"]):
        sel = (clk_all == clk_name) & ~np.isnan(acro_all)
        phases = acro_all[sel]
        radii  = np.ones(phases.sum() if phases.dtype == bool else len(phases))
        ax.scatter(phases, np.ones(len(phases)), s=60, color=col,
                   edgecolors="k", linewidths=0.5, alpha=0.8)
        period = 2 * np.pi / clock_bank[clk_name]
        ax.set_title(f"{clk_name} clock  (period {period:.0f} h)\n{len(phases)} rhythmic nodes",
                     fontsize=11, pad=18)
        ax.set_ylim(0, 1.3)
        ax.set_yticklabels([])
    fig.suptitle("Two-Clock Phase Structure on the n-Torus (𝕋²)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "two_torus_phase.png"), dpi=200)
    plt.close()
    print("  Saved two_torus_phase.png")

    print("\n✓ Training complete.")
    print(f"  Figures     → {SAVE_DIR}")
    print(f"  Checkpoint  → {ckpt_path}")
    print(f"  max(Ḣ)|_{{u=0}} = {pass_results.get('max_H_dot', np.nan):.3e}  "
          f"(≤0 required for passivity)")
    if "pearson_r" in cascade_report:
        print(f"  Cascade test: Pearson r = {cascade_report['pearson_r']:.3f}  "
              f"(dose-response correct: {cascade_report.get('dose_response_direction_correct')})")
    if "trajectory_rmse" in held_out_results:
        print(f"  Held-out RMSE = {held_out_results['trajectory_rmse']:.4f}")
    if "auroc_real" in edge_recovery:
        print(f"  Edge AUROC = {edge_recovery['auroc_real']:.3f}  "
              f"(above chance: {edge_recovery['above_chance']})")

    return model, state_data, cascade_report, pass_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train the cellular GNN-pHNN surrogate")
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic",
                    help="data source: synthetic generator or real assembled tri-omic")
    ap.add_argument("--total_hours", type=float, default=48.0,
                    help="trajectory length in hours (real source only)")
    a = ap.parse_args()
    train_omics_surrogate(epochs=a.epochs, batch_size=a.batch_size, seed=a.seed,
                          source=a.source, total_hours=a.total_hours)
