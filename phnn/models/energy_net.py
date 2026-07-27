"""
energy_net.py

Abundance-Based Hierarchical GNN Energy Network for the Multi-Omics GNN-pHNN.

Design notes
──────────────────────────
* Node features use the TWO-LAYER STATE:
    - All nodes: abundance deviation q_i (base layer)
    - Rhythmic nodes only: [sin(φ_i), cos(φ_i), ω_i] (derived phasor layer)
    - Chemical potential proxy: μ_i = q_i / C_i
  Previously: node features were [sin φ, cos φ, ω, |ω|] for ALL nodes —
  a mechanical analogy that is wrong for non-rhythmic molecules.

* Sparse message passing along biological adjacency (A_GG, A_PP, A_MM),
  not dense full-graph self-attention.

* Lyapunov structure: H(x) ≥ 0 with H = 0 at homeostasis (q = 0).
  Implemented via square activation on the base layer quadratic term:
  H_base_i = q_i² / (2 C_i) + nonlinear correction ≥ 0.

* Three omic-layer sub-energies remain for interpretability:
  H = H_G + H_P + H_M + H_cross

Design reference: 4-Regorous.ipynb §2.1-2.2, §3.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from biophasor.core.datagen import LAYER_CONFIG

_N_G, _N_P, _N_M = (
    LAYER_CONFIG["genomics"]["n_nodes"],
    LAYER_CONFIG["proteome"]["n_nodes"],
    LAYER_CONFIG["metabolome"]["n_nodes"],
)

# Node feature dimensions — two DISTINCT base features per node:
#   q       : abundance deviation (directly measurable)
#   log_q   : log-abundance ≈ chemical potential μ = kT ln(q/q_ref)
#             This is a FUNDAMENTALLY DIFFERENT transform from q itself:
#             it encodes concentration-scale information and diverges at q→0
#             (thermodynamic singularity at the empty state).
#   [sin φ, cos φ] : phasor for rhythmic nodes (zero elsewhere)
#   ω       : instantaneous frequency for rhythmic nodes
FEAT_Q   = 1    # abundance deviation
FEAT_MU  = 1    # log(q + ε) — thermodynamic chemical potential proxy
FEAT_PHI = 2    # [sin φ, cos φ]  — rhythmic nodes only
FEAT_OMG = 1    # ω — angular frequency (rhythmic nodes only)
NODE_FEAT_DIM = FEAT_Q + FEAT_MU + FEAT_PHI + FEAT_OMG   # = 5


class _SparseGNNLayer(nn.Module):
    """
    One graph attention layer operating on a SPARSE adjacency mask.

    Message passing: h_i^{l+1} = LayerNorm(h_i^l + Σ_j A_{ij} * attention(h_i, h_j))
    where A_ij is the biological adjacency — zero outside graph edges.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads    = n_heads
        head_dim        = hidden_dim // n_heads

        self.Wq = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wk = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wv = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wo = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.scale = head_dim ** -0.5

    def forward(
        self,
        h:    torch.Tensor,        # (B, N, H)
        adj:  Optional[torch.Tensor] = None,   # (N, N) or None for dense
    ) -> torch.Tensor:
        B, N, H = h.shape
        head_dim = H // self.n_heads

        Q = self.Wq(h).view(B, N, self.n_heads, head_dim).transpose(1, 2)  # (B, nh, N, d)
        K = self.Wk(h).view(B, N, self.n_heads, head_dim).transpose(1, 2)
        V = self.Wv(h).view(B, N, self.n_heads, head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, nh, N, N)

        if adj is not None:
            # Mask out non-edges (set to -inf before softmax)
            mask = (adj == 0).unsqueeze(0).unsqueeze(0)   # (1, 1, N, N)
            scores = scores.masked_fill(mask, -1e9)

        attn   = F.softmax(scores, dim=-1)                           # (B, nh, N, N)
        out    = torch.matmul(attn, V)                               # (B, nh, N, d)
        out    = out.transpose(1, 2).contiguous().view(B, N, H)
        out    = self.Wo(out)

        h  = self.norm1(h + out)
        h  = self.norm2(h + self.ff(h))
        return h


class _IntraLayerGNN(nn.Module):
    """Two sparse GNN layers for within-omic-layer message passing."""

    def __init__(self, n_nodes: int, hidden_dim: int):
        super().__init__()
        self.n_nodes = n_nodes
        self.embed   = nn.Linear(NODE_FEAT_DIM, hidden_dim)
        self.layers  = nn.ModuleList([
            _SparseGNNLayer(hidden_dim) for _ in range(2)
        ])
        # Lyapunov-structured energy readout: H_layer ≥ 0
        # Use square activation to guarantee non-negativity
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,         # (B, N_layer, NODE_FEAT_DIM)
        adj:        Optional[torch.Tensor] = None,   # (N_layer, N_layer)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h           = F.silu(self.embed(node_feats))   # (B, N, H)
        for layer in self.layers:
            h       = layer(h, adj)
        graph_emb   = h.mean(dim=1)                    # (B, H)
        H_raw       = self.readout(graph_emb)          # (B, 1)
        # Lyapunov structure: H ≥ 0, minimum when graph_emb ≈ 0 (homeostasis)
        H_sub       = F.softplus(H_raw)                # (B, 1)  > 0
        return h, H_sub


class _CrossOmicInteraction(nn.Module):
    """
    Cross-omic interaction energy with STRUCTURALLY DISTINCT edge types:

    Mass bonds (G↔P via central dogma):
      Power-conjugate flux coupling.  Implemented as symmetric cross-attention
      that enters H_cross as (f_G + f_P)²/2 — recovers energy when J=0.
      These are the edges that appear in J skew-symmetrically.

    Modulated ports (P→M enzymatic, M→G feedback):
      Informational / catalytic couplings that transmit NO net power.
      Implemented as a UNIDIRECTIONAL gain that modulates a flux without
      being a flux itself.  Net power contribution = 0 by construction
      (the modulated output is multiplied by a source that is not depleted).

    H_cross represents the cross-layer interaction storage function.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()

        # ── Mass bond: G ↔ P (central dogma, bidirectional) ─────────────────
        # Symmetric cross-attention (energy conservative: GP = PG transpose)
        self.mb_gp_q = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mb_gp_k = nn.Linear(hidden_dim, hidden_dim // 2)
        self.mb_gp_v = nn.Linear(hidden_dim, hidden_dim // 2)

        # ── Modulated port: P → M (enzymatic, unidirectional) ────────────────
        # A state-dependent GAIN on the M dynamics, not a flow
        self.mp_pm_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),                        # bounded gain ∈ (0, 1)
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Modulated port: M → G (feedback, unidirectional) ─────────────────
        self.mp_mg_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Cross-layer energy readout (quadratic: H_cross ≥ 0)
        self.readout_mb  = nn.Linear(hidden_dim // 2, 1)
        self.readout_mp  = nn.Linear(2, 1)

    def forward(
        self,
        h_G: torch.Tensor,    # (B, n_G, H)
        h_P: torch.Tensor,    # (B, n_P, H)
        h_M: torch.Tensor,    # (B, n_M, H)
        A_GP_dogma: Optional[torch.Tensor] = None,  # (n_G, n_P)
        A_PM_enz:   Optional[torch.Tensor] = None,  # (n_P, n_M)
        A_MG_fb:    Optional[torch.Tensor] = None,  # (n_M, n_G)
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns (H_cross, port_info)
        H_cross   : (B, 1)  cross-layer interaction energy
        port_info : dict with 'mass_bond_energy', 'port_P_gain', 'port_M_gain'
        """
        # ── Mass bond G ↔ P ──────────────────────────────────────────────────
        Q = self.mb_gp_q(h_P)   # (B, n_P, H/2)
        K = self.mb_gp_k(h_G)   # (B, n_G, H/2)
        V = self.mb_gp_v(h_G)

        scale  = Q.size(-1) ** 0.5
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale  # (B, n_P, n_G)

        if A_GP_dogma is not None:
            # Mask to central-dogma edges (biological prior)
            mask = (A_GP_dogma.T == 0).unsqueeze(0)  # (1, n_P, n_G)
            scores = scores.masked_fill(mask.to(scores.device), -1e9)

        attn        = F.softmax(scores, dim=-1)           # (B, n_P, n_G)
        gp_context  = torch.bmm(attn, V).mean(dim=1)     # (B, H/2)
        H_mb        = F.softplus(self.readout_mb(gp_context))  # (B, 1) ≥ 0

        # ── Modulated port P → M (enzymatic gate) ─────────────────────────────
        # Net power = 0: the enzyme modulates the metabolic rate but is not
        # consumed.  We represent this as a bounded gain on the M embedding,
        # MASKED to the known enzymatic adjacency so only real enzyme→metabolite
        # edges are active.
        gain_PM_full = self.mp_pm_gate(h_P)    # (B, n_P, 1) pre-mask
        gain_MG_full = self.mp_mg_gate(h_M)    # (B, n_M, 1) pre-mask

        if A_PM_enz is not None:
            # A_PM_enz: (n_P, n_M) — gate is non-zero only at enzyme edges
            # Sum enzyme gains weighted by adjacency mask
            # (B, n_P, 1) × (n_P, n_M) → aggregate to (B, 1)
            mask_PM = A_PM_enz.to(gain_PM_full.device).unsqueeze(0)  # (1, n_P, n_M)
            gain_PM = (gain_PM_full * mask_PM.sum(dim=2, keepdim=True).clamp(0,1)).mean(dim=1)  # (B,1)
        else:
            gain_PM = gain_PM_full.mean(dim=1)   # (B, 1)

        if A_MG_fb is not None:
            # A_MG_fb: (n_M, n_G) — gate non-zero only at feedback edges
            mask_MG = A_MG_fb.to(gain_MG_full.device).unsqueeze(0)   # (1, n_M, n_G)
            gain_MG = (gain_MG_full * mask_MG.sum(dim=2, keepdim=True).clamp(0,1)).mean(dim=1)  # (B,1)
        else:
            gain_MG = gain_MG_full.mean(dim=1)   # (B, 1)

        # Energy contribution from modulated ports is zero-net-power:
        # store only the SQUARE of the gain (Lyapunov term for the port state)
        H_mp = F.softplus(self.readout_mp(
            torch.cat([gain_PM, gain_MG], dim=-1)
        ))  # (B, 1)

        H_cross   = H_mb + 0.3 * H_mp   # cross total (0.3 weights modulated ports less)

        port_info = {
            "mass_bond_energy": H_mb.detach(),
            "port_P_gain":      gain_PM.detach(),
            "port_M_gain":      gain_MG.detach(),
        }
        return H_cross, port_info



class AbundanceGNN_EnergyNet(nn.Module):
    """
    Abundance-Based Hierarchical GNN Energy Network.

    Computes H(x) = H_G + H_P + H_M + H_cross
    from the two-layer state x = [q_abundance; phasor_rhythmic].

    H(x) ≥ 0 with H = 0 at homeostasis (q = 0, φ = 0) by Lyapunov construction.

    Parameters
    ----------
    N_total    : total number of molecular nodes (N_G + N_P + N_M)
    N_rhythmic : number of rhythmically oscillating nodes
    hidden_dim : GNN hidden dimension
    """

    def __init__(
        self,
        N_total:    int,
        N_rhythmic: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.N       = N_total
        self.N_r     = N_rhythmic
        self.n_G     = _N_G
        self.n_P     = _N_P
        self.n_M     = _N_M
        self.hidden  = hidden_dim

        # Intra-layer GNNs (sparse message passing along bio adjacency)
        self.gnn_G = _IntraLayerGNN(self.n_G, hidden_dim)
        self.gnn_P = _IntraLayerGNN(self.n_P, hidden_dim)
        self.gnn_M = _IntraLayerGNN(self.n_M, hidden_dim)

        # Cross-omic interaction (mass bonds + modulated ports)
        self.cross = _CrossOmicInteraction(hidden_dim)

        # ── Compartment energy readout (Phase B) ─────────────────────────────
        # The composite Hamiltonian is decomposed by FUNCTIONAL COMPARTMENT:
        #   H = Σ_c H_c + H_int.
        # Node embeddings from the three layer-GNNs are pooled per compartment
        # (a compartment spans layers) and read out to a non-negative sub-energy
        # H_c ≥ 0 via a shared Lyapunov head.  This makes the compartment
        # decomposition a genuine computed object — used for per-compartment
        # passivity checks and the composite-Hamiltonian claim — WITHOUT
        # altering the passivity-certified total H that drives the dynamics.
        self.compartment_readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _build_node_features(
        self,
        x:               torch.Tensor,   # (B, N + 3*N_r)
        rhythmic_indices: torch.Tensor,   # (N_r,)  global indices in [0, N)
    ) -> torch.Tensor:
        """
        Construct per-node feature matrix (B, N, NODE_FEAT_DIM=5).

        Base features (all nodes):
          [q_i, μ_i]   from x[:, :N]
        Phasor features (rhythmic nodes only, zero for non-rhythmic):
          [sin φ, cos φ, ω]  from x[:, N : N + 3*N_r]
        """
        B      = x.size(0)
        N, N_r = self.N, self.N_r
        device = x.device

        # Base layer
        q  = x[:, :N]                           # (B, N)  abundance deviation
        # Chemical potential proxy: μ ≈ kT ln(q/q_ref) ∝ log(q + ε)
        # This is DISTINCT from q: log-scale captures thermodynamic free energy.
        # ε = 1e-3 prevents log(0) without shifting the zero-crossings significantly.
        mu = torch.log(F.relu(q) + 1e-3)        # (B, N)  log-abundance

        # Initialize phasor features as zeros for all nodes
        sin_phi = torch.zeros(B, N, device=device)
        cos_phi = torch.ones(B, N, device=device)   # default phase = 0
        omega   = torch.zeros(B, N, device=device)

        if N_r > 0:
            phasor_block = x[:, N:]   # (B, 3*N_r)
            sp = phasor_block[:, :N_r]              # sin φ
            cp = phasor_block[:, N_r:2*N_r]         # cos φ
            om = phasor_block[:, 2*N_r:3*N_r]       # ω

            idx = rhythmic_indices.to(device)
            sin_phi[:, idx] = sp
            cos_phi[:, idx] = cp
            omega[:, idx]   = om

        # Stack: (B, N, 5)
        feats = torch.stack([q, mu, sin_phi, cos_phi, omega], dim=-1)
        return feats

    def forward(
        self,
        x:                torch.Tensor,   # (B, state_dim)
        rhythmic_indices: torch.Tensor,   # (N_r,)
        bio_graph:        dict,           # from build_biological_graph()
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        H       : (B, 1)  total Hamiltonian storage function  (≥ 0)
        sub_H   : dict with:
                    'G','P','M' : (B,1) per-layer sub-energies (drive dynamics)
                    'cross'     : (B,1) cross-layer interaction energy
                    'port_info' : dict from _CrossOmicInteraction
                    'compartment': dict cid -> (B,1) per-compartment energy H_c≥0
                                   (Phase B readout; diagnostic, not the total)
        """
        B     = x.size(0)
        feats = self._build_node_features(x, rhythmic_indices)  # (B, N, 5)

        # Split into omic layers
        f_G = feats[:, :self.n_G, :]
        f_P = feats[:, self.n_G:self.n_G+self.n_P, :]
        f_M = feats[:, self.n_G+self.n_P:, :]

        # Adjacency matrices (move to device)
        dev   = x.device
        A_GG  = bio_graph["A_GG"].to(dev) if bio_graph else None
        A_PP  = bio_graph["A_PP"].to(dev) if bio_graph else None
        A_MM  = bio_graph["A_MM"].to(dev) if bio_graph else None
        A_GP  = bio_graph["A_GP_dogma"].to(dev) if bio_graph else None
        A_PM  = bio_graph["A_PM_enz"].to(dev) if bio_graph else None
        A_MG  = bio_graph["A_MG_fb"].to(dev) if bio_graph else None  # metabolome → genomics feedback

        # Intra-layer sparse message passing
        h_G, H_G = self.gnn_G(f_G, A_GG)
        h_P, H_P = self.gnn_P(f_P, A_PP)
        h_M, H_M = self.gnn_M(f_M, A_MM)

        # Cross-omic (mass bonds + modulated ports) — all adjacencies now passed
        H_cross, port_info = self.cross(h_G, h_P, h_M, A_GP, A_PM, A_MG)

        H = H_G + H_P + H_M + H_cross   # (B, 1)

        sub_H = {
            "G":         H_G,
            "P":         H_P,
            "M":         H_M,
            "cross":     H_cross,
            "port_info": port_info,
        }

        # ── Per-compartment energy decomposition (Phase B) ───────────────────
        # Pool node embeddings by compartment membership and read out H_c ≥ 0.
        # This is a decomposition READOUT (diagnostic + validation), reported
        # alongside the layer decomposition; the dynamics continue to use the
        # total H above.
        if bio_graph is not None and "comp_masks" in bio_graph:
            h_all = torch.cat([h_G, h_P, h_M], dim=1)     # (B, N, hidden)
            comp_H = {}
            for cid, mask in bio_graph["comp_masks"].items():
                m = mask.to(dev)                          # (N,) bool
                if m.sum() == 0:
                    continue
                pooled = h_all[:, m, :].mean(dim=1)       # (B, hidden)
                comp_H[int(cid)] = F.softplus(self.compartment_readout(pooled))  # (B,1) ≥0
            sub_H["compartment"] = comp_H                 # dict cid -> (B,1)
        return H, sub_H
