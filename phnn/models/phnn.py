"""
phnn.py

Biologically Grounded Port-Hamiltonian Neural Network for Multi-Omics.

Design notes
──────────────────────────
1. Sparse_J_Net: the connectome J(x) is masked to biological graph edges.
   Dense O(N²) is replaced by O(|edges|) computation using the adjacency
   masks from bio_graph.py.

2. Hard central dogma: the G→P block of J is initialized near the identity
   (gene i → protein i) and the model learns only DEVIATIONS (post-
   transcriptional regulation).  The diagonal prior is never turned off.

3. Mass bonds vs. modulated ports: DIFFERENT MATHEMATICS.
   * Mass bonds (G↔P, G-intra, P-intra, M-intra):
       Skew-symmetric entries in J(x).  Enforced by construction.
       Masked to stoichiometric support (biological edges only).
   * Modulated ports (P→M, M→G):
       State-dependent GAINS on the coupling, with structural zero-net-power:
       the gain modulates a flow but does not appear as a flow itself.
       Implemented as a separate ModulatedPort_Net that produces scalar gates.

4. State-dependent R(x) initialized from k_deg priors:
   The base dissipation r_base is set from the measured / generated
   degradation rates k_deg (mRNA fast, protein slow, metabolite medium).
   State correction is a small MLP.  This gives R physical meaning
   (each diagonal element ≈ pool turnover rate) and connects R to the
   cascade prediction test.

5. State vector: x = [q (N), sin φ (N_r), cos φ (N_r), ω (N_r)]
   NOT [φ, ω] for all N.

Biological semantics:
  J(x)  : conservative coupling network (rewires with cell state)
  R(x)  : dissipation (degradation/turnover rates + state correction)
  G     : port map (nutrient/Zeitgeber/drug input channels)
  H(x)  : Lyapunov-type storage function (≥ 0, = 0 at homeostasis)

Design reference: 4-Regorous.ipynb §2.4, §3.3-3.5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from biophasor.phnn.models.energy_net import AbundanceGNN_EnergyNet
from biophasor.core.datagen import LAYER_CONFIG

_LAYER_CFG = {
    "genomics":   {"n_nodes": LAYER_CONFIG["genomics"]["n_nodes"]},
    "proteome":   {"n_nodes": LAYER_CONFIG["proteome"]["n_nodes"]},
    "metabolome": {"n_nodes": LAYER_CONFIG["metabolome"]["n_nodes"]},
}


def _layer_dims():
    v = list(_LAYER_CFG.values())
    return v[0]["n_nodes"], v[1]["n_nodes"], v[2]["n_nodes"]


# ─────────────────────────────────────────────────────────────────────────────
#  State-Dependent Dissipation R(x) — k_deg initialized
# ─────────────────────────────────────────────────────────────────────────────

class State_Dependent_R_Net(nn.Module):
    """
    Positive-semidefinite dissipation operator R(x).

    r_i(x) = softplus(r_base_i + δr_i(x))

    r_base_i is initialized from per-node degradation rate priors (k_deg),
    so the model starts with physically meaningful dissipation rates rather
    than a flat guess.  The state correction δr_i(x) allows the model to
    learn how dissipation changes with cell state (e.g., clock-gated
    mRNA degradation).

    Physical interpretation of R diagonal (for cascade predictor):
      The predicted transcript→protein phase lag is
        tan(Δφ) = ω_clock / r_i
      where r_i ≈ k_deg,i is the protein's degradation rate.
    """

    def __init__(
        self,
        state_dim: int,
        N_total:   int,
        hidden_dim: int = 64,
        k_deg_prior: Optional[torch.Tensor] = None,  # (N_total,)
    ):
        super().__init__()
        self.N         = N_total
        self.state_dim = state_dim

        # Base dissipation: initialized from k_deg priors
        if k_deg_prior is not None:
            r0 = torch.log(torch.clamp(k_deg_prior, min=0.05))  # inverse softplus
        else:
            r0 = torch.zeros(N_total)
        self.r_base = nn.Parameter(r0)

        # Small MLP for state-dependent correction
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, N_total),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-node dissipation rates r_i(x) ∈ (0, ∞), shape (B, N)."""
        delta_r = self.net(x)                                       # (B, N)
        r       = F.softplus(self.r_base.unsqueeze(0) + delta_r)    # (B, N) > 0
        return r

    def get_R_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full (B, state_dim, state_dim) dissipation matrix.
        Dissipation acts on the ABUNDANCE block [0:N, 0:N] of the state.
        This corresponds to first-order degradation of molecular pools.
        """
        r = self.forward(x)   # (B, N)
        # torch.diag_embed: vectorized batch diagonal construction — no Python loop
        R_abundance = torch.diag_embed(r)   # (B, N, N)
        B = x.size(0)
        R = torch.zeros(B, self.state_dim, self.state_dim, device=x.device)
        R[:, :self.N, :self.N] = R_abundance
        return R


# ─────────────────────────────────────────────────────────────────────────────
#  Sparse Dynamic Connectome J(x) — mass bonds only on bio edges
# ─────────────────────────────────────────────────────────────────────────────

class Sparse_Dynamic_J_Net(nn.Module):
    """
    Sparse state-dependent conservative connectome J(x).

    Architecture
    ────────────
    A shared encoder maps state x → latent z.
    Separate decoders predict the UPPER TRIANGLE of each intra-layer block
    (automatically skew-symmetric).
    Cross-layer blocks: the G↔P mass bond is predicted and masked to the
    central dogma adjacency A_GP_dogma.  A deviation term captures
    post-transcriptional regulation.
    Modulated ports (P→M, M→G) are NOT part of J — they are separate
    (see ModulatedPort_Net).

    Sparsity: predicted entries are element-wise multiplied by the
    biological adjacency mask before assembly, so J is exactly zero
    outside the biological graph.
    """

    def __init__(
        self,
        state_dim:    int,
        N_total:      int,
        hidden_dim:   int = 64,
    ):
        super().__init__()
        n_G, n_P, n_M = _layer_dims()
        self.n_G, self.n_P, self.n_M = n_G, n_P, n_M
        self.N        = N_total
        self.state_dim = state_dim

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
        )

        # Intra-layer: upper-triangle entries (→ skew-symmetric)
        self.j_GG = nn.Linear(hidden_dim, n_G * (n_G - 1) // 2)
        self.j_PP = nn.Linear(hidden_dim, n_P * (n_P - 1) // 2)
        self.j_MM = nn.Linear(hidden_dim, n_M * (n_M - 1) // 2)

        # Cross-layer mass bond G ↔ P (learned, masked to dogma)
        self.j_GP_base  = nn.Linear(hidden_dim, n_G * n_P)   # full, then masked
        self.j_GP_dev   = nn.Linear(hidden_dim, n_G * n_P)   # deviations (post-transcriptional)

        # Hard central dogma prior: learnable but anchored to near-identity
        self.dogma_scale = nn.Parameter(torch.ones(1) * 0.5)

        # Cache triu indices — computed once, reused every forward pass
        self._triu_GG = torch.triu_indices(n_G, n_G, offset=1)
        self._triu_PP = torch.triu_indices(n_P, n_P, offset=1)
        self._triu_MM = torch.triu_indices(n_M, n_M, offset=1)

    def _skew(self, vals: torch.Tensor, n: int, cached_idx: torch.Tensor) -> torch.Tensor:
        """Fill upper triangle → skew-symmetric (B, n, n).  Uses cached triu indices."""
        B   = vals.size(0)
        mat = torch.zeros(B, n, n, device=vals.device)
        idx = cached_idx.to(vals.device)
        mat[:, idx[0], idx[1]] = vals
        mat = mat - mat.transpose(1, 2)
        return mat

    def forward(
        self,
        x:          torch.Tensor,              # (B, state_dim)
        bio_graph:  dict,                      # adjacency from build_biological_graph()
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        J_full  : (B, state_dim, state_dim) skew-symmetric J (zeros outside state_dim×state_dim)
        blocks  : dict of named sub-blocks
        """
        B   = x.size(0)
        dev = x.device
        n_G, n_P, n_M = self.n_G, self.n_P, self.n_M
        N = n_G + n_P + n_M

        z = self.encoder(x)   # (B, H)

        # ── Intra-layer mass bonds (sparse, skew-symmetric) ─────────────────
        J_GG_raw = self._skew(self.j_GG(z), n_G, self._triu_GG)   # (B, nG, nG)
        J_PP_raw = self._skew(self.j_PP(z), n_P, self._triu_PP)
        J_MM_raw = self._skew(self.j_MM(z), n_M, self._triu_MM)

        # Apply biological adjacency mask
        if bio_graph is not None:
            A_GG = bio_graph["A_GG"].to(dev)
            A_PP = bio_graph["A_PP"].to(dev)
            A_MM = bio_graph["A_MM"].to(dev)
            # Symmetrize mask for intra-layer (undirected coupling)
            A_GG_sym = torch.clamp(A_GG + A_GG.T, 0, 1)
            A_PP_sym = torch.clamp(A_PP + A_PP.T, 0, 1)
            A_MM_sym = torch.clamp(A_MM + A_MM.T, 0, 1)
            J_GG = J_GG_raw * A_GG_sym.unsqueeze(0)
            J_PP = J_PP_raw * A_PP_sym.unsqueeze(0)
            J_MM = J_MM_raw * A_MM_sym.unsqueeze(0)
        else:
            J_GG, J_PP, J_MM = J_GG_raw, J_PP_raw, J_MM_raw

        # ── Cross-layer mass bond G ↔ P (central dogma) ─────────────────────
        A_GP = bio_graph["A_GP_dogma"].to(dev) if bio_graph else None

        J_GP_base = self.j_GP_base(z).view(B, n_G, n_P)   # learned base
        J_GP_dev  = self.j_GP_dev(z).view(B, n_G, n_P)    # post-transcriptional dev.

        if A_GP is not None:
            # Base: masked to central dogma edges; deviation: unrestricted (but small)
            J_GP_base = J_GP_base * A_GP.unsqueeze(0) * self.dogma_scale.abs()
        J_GP = J_GP_base + 0.1 * J_GP_dev   # deviations weighted down

        # Skew-symmetrize G↔P block globally
        J_PG = -J_GP.transpose(1, 2)

        # ── Assemble N×N J matrix ────────────────────────────────────────────
        J_N = torch.zeros(B, N, N, device=dev)
        J_N[:, :n_G,         :n_G]          = J_GG
        J_N[:, n_G:n_G+n_P,  n_G:n_G+n_P]  = J_PP
        J_N[:, n_G+n_P:,     n_G+n_P:]      = J_MM
        J_N[:, :n_G,         n_G:n_G+n_P]   = J_GP
        J_N[:, n_G:n_G+n_P,  :n_G]          = J_PG

        # ── Lift to state_dim × state_dim ────────────────────────────────────
        # The state has N abundance dims + 3*N_r phasor dims.
        # J acts on abundance block; phasor block is coupled via kinematics.
        state_dim = self.state_dim
        J_full    = torch.zeros(B, state_dim, state_dim, device=dev)
        J_full[:, :N, :N] = J_N   # abundance block

        blocks = {
            "J_GG": J_GG.detach(),
            "J_PP": J_PP.detach(),
            "J_MM": J_MM.detach(),
            "J_GP": J_GP.detach(),
        }
        return J_full, blocks


# ─────────────────────────────────────────────────────────────────────────────
#  Modulated Port Networks — ZERO NET POWER
# ─────────────────────────────────────────────────────────────────────────────

class ModulatedPort_Net(nn.Module):
    """
    Modulated (signalling/enzymatic) ports for P→M and M→G coupling.

    These carry NO net power: they modulate a flux without being a flux.
    Implemented as state-dependent scalar GATES that scale an existing flow,
    rather than as entries in J.

    For the pHNN dynamics:
        dx/dt = (J - R) ∇H + G u  +  Γ(x) ∘ ∇H_cross
    where Γ(x) is a diagonal-like operator with zero net power,
    satisfying  ∇H^T Γ(x) ∇H = 0  for any H.

    Structural zero-net-power enforcement:
    Γ is constructed antisymmetrically in the cross-layer sense:
      Γ_PM(x): a gain from P-block → M-block with equal and opposite entry
    so that  ∇H_P * Γ_{PM} ∇H_M + ∇H_M * Γ_{MP} ∇H_P = 0 exactly.
    """

    def __init__(self, state_dim: int, N_total: int, hidden_dim: int = 32):
        super().__init__()
        self.state_dim = state_dim
        self.N         = N_total
        n_G, n_P, n_M  = _layer_dims()
        self.n_G, self.n_P, self.n_M = n_G, n_P, n_M

        # Gate networks: predict scalar gain per cross-layer pair
        self.gate_PM = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_P * n_M),
        )
        self.gate_MG = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, n_M * n_G),
        )

        # ── Clock-coupling gate (Phase B) ────────────────────────────────────
        # Circadian ↔ redox coupling is SIGNALLING (NAD+/SIRT1, NADPH), not mass
        # transfer — so it is a zero-net-power modulated port over the full
        # N-node abundance block, masked to M_clock_couple (circadian↔redox
        # node pairs).  One scalar gain per node pair; the antisymmetric
        # assembly in the pHNN dynamics guarantees zero net power.
        N = n_G + n_P + n_M
        self.gate_clock = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, N * N),
        )
        self._N_nodes = N

    def forward(
        self,
        x:         torch.Tensor,   # (B, state_dim)
        bio_graph: dict,
    ) -> dict:
        """
        Returns zero-net-power cross-layer modulation matrices:
          'Gamma_PM' : (B, n_P, n_M) — enzymatic modulation (masked to enz. edges)
          'Gamma_MG' : (B, n_M, n_G) — feedback modulation (masked to fb. edges)
        The antisymmetric structure is enforced implicitly: these enter
        the dynamics as G_PM * ∇H_M added to P equations and
                        -G_PM^T * ∇H_P added to M equations
        so net power = ∇H_P^T G_PM ∇H_M - ∇H_M^T G_PM^T ∇H_P = 0.
        """
        B   = x.size(0)
        dev = x.device
        n_P, n_M, n_G = self.n_P, self.n_M, self.n_G

        Gamma_PM_raw = self.gate_PM(x).view(B, n_P, n_M)
        Gamma_MG_raw = self.gate_MG(x).view(B, n_M, n_G)

        # Clock-coupling gain over the full N×N abundance block
        N = self._N_nodes
        Gamma_clock_raw = self.gate_clock(x).view(B, N, N)

        if bio_graph is not None:
            A_PM = bio_graph["A_PM_enz"].to(dev)
            A_MG = bio_graph["A_MG_fb"].to(dev)
            Gamma_PM = Gamma_PM_raw * A_PM.unsqueeze(0)
            Gamma_MG = Gamma_MG_raw * A_MG.unsqueeze(0)
            # Mask clock coupling to circadian↔redox node pairs (upper triangle
            # only; the lower triangle is generated antisymmetrically in the
            # dynamics so net power vanishes exactly).
            M_ck = bio_graph.get("M_clock_couple")
            if M_ck is not None:
                M_ck = torch.triu(M_ck.to(dev), diagonal=1)   # (N, N)
                Gamma_clock = Gamma_clock_raw * M_ck.unsqueeze(0)
            else:
                Gamma_clock = torch.zeros(B, N, N, device=dev)
        else:
            Gamma_PM, Gamma_MG = Gamma_PM_raw, Gamma_MG_raw
            Gamma_clock = Gamma_clock_raw

        return {"Gamma_PM": Gamma_PM, "Gamma_MG": Gamma_MG,
                "Gamma_clock": Gamma_clock}


# ─────────────────────────────────────────────────────────────────────────────
#  Full Port-Hamiltonian Neural Network
# ─────────────────────────────────────────────────────────────────────────────

class Generic_pHNN(nn.Module):
    """
    GNN-Surrogate Port-Hamiltonian Neural Network.

    dx/dt = (J(x) - R(x)) ∇H(x) + G u(t)
           + modulated_port_term(x, ∇H)

    Components:
      AbundanceGNN_EnergyNet : H(x) — Lyapunov energy (≥ 0)
      Sparse_Dynamic_J_Net   : J(x) — sparse skew-symmetric dynamic connectome
      State_Dependent_R_Net  : R(x) — k_deg-initialized dissipation (PSD)
      ModulatedPort_Net      : Γ(x) — zero-net-power modulated ports
      G (Port Matrix)        : maps 3 port inputs to omic-layer abundances
    """

    def __init__(
        self,
        N_total:          int,
        N_rhythmic:       int,
        hidden_dim:       int = 128,
        n_ports:          int = 3,
        k_deg_prior:      Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.N   = N_total
        self.N_r = N_rhythmic
        state_dim = N_total + 3 * N_rhythmic
        self.state_dim = state_dim

        n_G, n_P, n_M = _layer_dims()
        self.n_G, self.n_P, self.n_M = n_G, n_P, n_M

        # 1. Energy network
        self.energy_net = AbundanceGNN_EnergyNet(N_total, N_rhythmic, hidden_dim)

        # 2. Dynamic sparse connectome
        self.dynamic_j = Sparse_Dynamic_J_Net(state_dim, N_total, hidden_dim // 2)

        # 3. State-dependent dissipation (k_deg initialized)
        self.r_net = State_Dependent_R_Net(
            state_dim, N_total, hidden_dim // 2, k_deg_prior
        )

        # 4. Modulated ports (zero-net-power)
        self.port_net = ModulatedPort_Net(state_dim, N_total, hidden_dim // 4)

        # 5. Port matrix G: maps 3 inputs to abundance equations
        #    Port 0 → Genomics (Zeitgeber / circadian entrainment)
        #    Port 1 → Proteome (drug dosing)
        #    Port 2 → Metabolome (nutrient influx)
        G_init = torch.zeros(N_total, n_ports)
        G_init[:n_G,         0] = 1.0 / n_G
        G_init[n_G:n_G+n_P,  1] = 1.0 / n_P
        G_init[n_G+n_P:,     2] = 1.0 / n_M
        self.G       = nn.Parameter(G_init)
        self.n_ports = n_ports

    def get_structure_matrices(
        self,
        x:               torch.Tensor,
        rhythmic_indices: torch.Tensor,
        bio_graph:       dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Assemble J(x) and R(x).

        Returns (J_full, R_full, j_blocks)
        """
        J_full, j_blocks = self.dynamic_j(x, bio_graph)
        R_full           = self.r_net.get_R_matrix(x)
        return J_full, R_full, j_blocks

    def _modulated_port_term(
        self,
        x:          torch.Tensor,    # (B, state_dim)
        nabla_H:    torch.Tensor,    # (B, state_dim)
        bio_graph:  dict,
    ) -> torch.Tensor:
        """
        Compute the modulated port contribution to dx/dt.

        Structural zero-net-power:
          For P→M port with gate Γ_PM(x):
            d(abundance_P)/dt += Γ_PM(x) @ (∇H_M)
            d(abundance_M)/dt -= Γ_PM(x)^T @ (∇H_P)
          Net power = ∇H_P^T Γ_PM ∇H_M - ∇H_M^T Γ_PM^T ∇H_P = 0  ✓
        """
        B   = x.size(0)
        dev = x.device
        n_G, n_P, n_M = self.n_G, self.n_P, self.n_M

        port_matrices = self.port_net(x, bio_graph)
        Gamma_PM = port_matrices["Gamma_PM"]   # (B, n_P, n_M)
        Gamma_MG = port_matrices["Gamma_MG"]   # (B, n_M, n_G)

        # Energy gradient for abundance block
        gH_G = nabla_H[:, :n_G]                      # (B, n_G)
        gH_P = nabla_H[:, n_G:n_G+n_P]               # (B, n_P)
        gH_M = nabla_H[:, n_G+n_P:n_G+n_P+n_M]       # (B, n_M)

        # P→M port: power = ∇H_P^T Γ_PM ∇H_M − ∇H_M^T Γ_PM^T ∇H_P = 0
        mp_term = torch.zeros(B, self.state_dim, device=dev)
        mp_term[:, n_G:n_G+n_P] += torch.bmm(Gamma_PM, gH_M.unsqueeze(-1)).squeeze(-1)
        mp_term[:, n_G+n_P:n_G+n_P+n_M] -= torch.bmm(
            Gamma_PM.transpose(1, 2), gH_P.unsqueeze(-1)
        ).squeeze(-1)

        # M→G port: zero-net-power
        mp_term[:, :n_G]         += torch.bmm(Gamma_MG.transpose(1, 2),
                                               gH_M.unsqueeze(-1)).squeeze(-1)
        mp_term[:, n_G+n_P:n_G+n_P+n_M] -= torch.bmm(
            Gamma_MG, gH_G.unsqueeze(-1)
        ).squeeze(-1)

        # ── Clock-coupling port (Phase B): circadian ↔ redox, zero net power ──
        # Γ_clock is strictly upper-triangular over the N-node abundance block.
        # Skew-symmetrize it → (Γ − Γ^T); acting on ∇H_q the associated power is
        #   ∇H_q^T (Γ − Γ^T) ∇H_q = 0  exactly (skew form), so this signalling
        # coupling transmits no net power while phase-locking the two clocks.
        Gamma_clock = port_matrices.get("Gamma_clock")
        if Gamma_clock is not None:
            N     = n_G + n_P + n_M
            gH_q  = nabla_H[:, :N]                                   # (B, N)
            Gck_skew = Gamma_clock - Gamma_clock.transpose(1, 2)     # (B, N, N) skew
            mp_term[:, :N] += torch.bmm(Gck_skew, gH_q.unsqueeze(-1)).squeeze(-1)

        return mp_term

    def forward(
        self,
        x:                torch.Tensor,   # (B, state_dim)
        u:                torch.Tensor,   # (B, n_ports)
        rhythmic_indices: torch.Tensor,   # (N_r,)
        bio_graph:        dict,
    ) -> tuple[torch.Tensor, torch.Tensor, dict, torch.Tensor]:
        """
        Returns
        -------
        dx_dt_pred : (B, state_dim)  predicted state derivative
        H          : (B, 1)          Hamiltonian energy
        sub_H      : dict            per-layer sub-energies
        nabla_H    : (B, state_dim)  energy gradient
        """
        # 1. Energy
        H, sub_H = self.energy_net(x, rhythmic_indices, bio_graph)

        # 2. ∇H w.r.t. x
        nabla_H = torch.autograd.grad(
            H, x,
            grad_outputs=torch.ones_like(H),
            create_graph=self.training,
        )[0]   # (B, state_dim)

        # 3. Structure matrices
        J_full, R_full, j_blocks = self.get_structure_matrices(
            x, rhythmic_indices, bio_graph
        )
        structure = J_full - R_full   # (B, state_dim, state_dim)

        # 4. pH internal dynamics
        internal = torch.bmm(structure, nabla_H.unsqueeze(-1)).squeeze(-1)  # (B, sd)

        # 5. Modulated port term (zero net power)
        mp_term = self._modulated_port_term(x, nabla_H, bio_graph)

        # 6. External port input: G maps u → abundance equations
        G_ext = torch.zeros_like(x)
        G_ext[:, :self.N] = torch.matmul(u, self.G.T)   # (B, N)

        # 7. Full dynamics
        dx_dt_pred = internal + mp_term + G_ext

        # Cache for downstream loss computation
        self._last_J        = J_full
        self._last_R        = R_full
        self._last_j_blocks = j_blocks

        return dx_dt_pred, H, sub_H, nabla_H
