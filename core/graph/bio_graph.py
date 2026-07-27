"""
bio_graph.py

Sparse biological graph construction for the GNN-pHNN.

Purpose
-------
The GNN surrogate's "acceleration" claim is only real with biological sparsity:
a dense O(N²) J matrix doesn't scale.  This module builds the sparse edge
structure from curated biological priors so that:
  • Message passing is O(|edges|), not O(N²).
  • The learned connectome can be evaluated against known biology.
  • Impossible couplings are excluded from the hypothesis space.

Graph components
────────────────
1. Intra-layer GRN (G→G):
   Erdős-Rényi with p=GRN_DENSITY, then hub amplification (scale-free topology).
   Represents TF→target regulatory edges.

2. Central dogma (G→P):
   Near-diagonal correspondence: gene i → protein i, ± DOGMA_OFFSET off-diagonal.
   This is the KNOWN, HARD-WIRED mapping.  It is NOT learned.
   Only deviations (post-transcriptional regulation) are learnable.

3. PPI (P→P):
   Cluster-structured graph (3 functional clusters + sparse cross-cluster links).
   Represents protein–protein interaction network.

4. Metabolic (M→M):
   Pathway-structured (3 linear pathways + sparse cross-pathway).
   Represents metabolic reaction adjacency.

5. Enzymatic port (P→M):
   Sparse bipartite: each enzyme in P connects to its 1-3 substrate metabolites.
   Modulated port (not mass bond) — used for the P→M coupling.

6. Feedback (M→G):
   Sparse bipartite: metabolite pool levels feed back to modulate TF activity.

Stoichiometric matrix S
───────────────────────
Three conserved moieties in the metabolome layer.  S encodes the stoichiometric
coefficients so that the mass-bond block of J preserves Sq = const exactly.

PLV prior weights
─────────────────
Phase-locking-value coupling matrix used as weak regularization prior on
cross-layer J entries.  Computed after data generation — passed in as
float array in [0, 1].

Design reference: 4-Regorous.ipynb §1 (Gap B), §2.4
"""

import numpy as np
import torch

from biophasor.core.datagen.omics_data_generator import (
    LAYER_CONFIG, CONSERVATION_GROUPS, get_layer_slices,
)
from biophasor.core.datagen.compartments import (
    build_compartments, concat_compartment_arrays,
    COMPARTMENT_IDS, N_COMPARTMENTS, COMPARTMENT_CLOCK, CLOCK_NAMES,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Graph density parameters
# ─────────────────────────────────────────────────────────────────────────────

GRN_DENSITY     = 0.05    # ~5% intra-genomics TF connectivity
# Note: PPI uses cluster structure (not ER); metabolic uses pathway structure (not ER)
# MET_DENSITY and PPI_DENSITY are intentionally not used; topology is structured, not random
ENZ_FANOUT      = 2       # each enzyme connects to this many metabolites (±1)
FB_DENSITY      = 0.04    # metabolite→TF feedback (sparse)
DOGMA_OFFSET    = 2       # gene i → protein i ± DOGMA_OFFSET off-diagonal


def _er_adjacency(n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    """Erdős-Rényi directed adjacency matrix (no self-loops)."""
    A = (rng.random((n, n)) < density).astype(float)
    np.fill_diagonal(A, 0)
    return A


def _cluster_adjacency(n: int, n_clusters: int,
                        within_p: float, across_p: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Cluster-structured adjacency: dense within clusters, sparse across."""
    A = np.zeros((n, n))
    sizes  = np.array_split(np.arange(n), n_clusters)
    for c, idx in enumerate(sizes):
        for i in idx:
            for j in idx:
                if i != j and rng.random() < within_p:
                    A[i, j] = 1.0
    # Sparse cross-cluster
    for i in range(n):
        for j in range(n):
            if A[i, j] == 0 and i != j and rng.random() < across_p:
                A[i, j] = 1.0
    return A


def _pathway_adjacency(n: int, n_pathways: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Linear pathway chains + sparse cross-pathway links."""
    A = np.zeros((n, n))
    sizes = np.array_split(np.arange(n), n_pathways)
    for path_nodes in sizes:
        for k in range(len(path_nodes) - 1):
            i, j = path_nodes[k], path_nodes[k + 1]
            A[i, j] = 1.0
            A[j, i] = 1.0   # bidirectional pathway
    # Sparse cross links
    cross_mask = rng.random((n, n)) < 0.03
    np.fill_diagonal(cross_mask, False)
    A[cross_mask] = 1.0
    return A


def _central_dogma_adjacency(n_G: int, n_P: int,
                              offset: int,
                              rng: np.random.Generator) -> np.ndarray:
    """
    Near-diagonal G → P correspondence.
    Gene i → Protein i (if i < min(n_G, n_P)), with ± offset random assignments.
    This is the hard-wired central dogma.  The returned binary matrix is NOT
    updated during training — it seeds the J_GP block.
    """
    A = np.zeros((n_G, n_P))
    for i in range(min(n_G, n_P)):
        A[i, i] = 1.0                        # primary diagonal
        for d in range(1, offset + 1):
            if i + d < n_P and rng.random() < 0.3:
                A[i, i + d] = 1.0           # off-diagonal (splicing isoforms)
            if i - d >= 0 and rng.random() < 0.2:
                A[i, i - d] = 1.0
    return A


def _enzymatic_adjacency(n_P: int, n_M: int, fanout: int,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Sparse P → M enzymatic port.
    Each protein (enzyme) connects to fanout ± 1 metabolite substrates.
    These are MODULATED PORTS (not mass bonds).
    """
    A = np.zeros((n_P, n_M))
    for i in range(n_P):
        # Each enzyme catalyses 1-3 reactions
        n_targets = max(1, fanout + rng.integers(-1, 2))
        targets   = rng.choice(n_M, size=min(n_targets, n_M), replace=False)
        A[i, targets] = 1.0
    return A


def _feedback_adjacency(n_M: int, n_G: int, density: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Sparse M → G feedback (metabolites modulating TF activity)."""
    A = (rng.random((n_M, n_G)) < density).astype(float)
    return A


def build_compartment_structure() -> dict:
    """
    Build the compartment block structure of J over the concatenated N-node axis.

    Returns
    -------
    dict with (N = n_G + n_P + n_M):
      'comp_id'          : (N,) int    compartment id (1..N_COMPARTMENTS) per node
      'clock_label'      : (N,) str    "circadian" / "redox" / "none" per node
      'comp_masks'       : dict cid -> (N,) bool   membership indicator per compartment
      'M_intra'          : (N, N) float  intra-compartment block mask (1 within a
                                         compartment, off-diagonal, symmetric)
      'M_inter'          : (N, N) float  inter-compartment mask (complement of intra,
                                         off-diagonal) — support of inter-compartment
                                         ports (mass bonds + modulated + clock coupling)
      'M_clock_couple'   : (N, N) float  inter-compartment mask restricted to node
                                         pairs whose compartments carry DIFFERENT
                                         clocks (both != "none") — the clock-coupling
                                         port support (e.g. circadian ↔ redox)
      'omega_node'       : (N,) float  per-node clock ω (0 if non-rhythmic)

    The masks are the biological support for the block-diagonal-plus-ports J
    assembly:  J = blkdiag(J_c) [on M_intra] + J_inter [on M_inter].
    """
    import numpy as _np
    from biophasor.core.datagen.compartments import CLOCK_BANK as _CB  # local import to avoid cycle at top
    layer_sizes = {name: cfg["n_nodes"] for name, cfg in LAYER_CONFIG.items()}
    built  = build_compartments(layer_sizes)
    cat    = concat_compartment_arrays(built, list(LAYER_CONFIG.keys()))
    comp_id = cat["comp_id"]         # (N,)
    clock   = cat["clock_label"]     # (N,)
    N       = comp_id.shape[0]

    comp_masks = {cid: (comp_id == cid) for cid in COMPARTMENT_IDS}

    same_comp = (comp_id[:, None] == comp_id[None, :])
    off_diag  = ~_np.eye(N, dtype=bool)
    M_intra   = (same_comp & off_diag).astype(_np.float32)
    M_inter   = ((~same_comp) & off_diag).astype(_np.float32)

    # Clock-coupling support: both endpoints rhythmic, different clocks
    has_clock = (clock != "none")
    diff_clock = (clock[:, None] != clock[None, :])
    both_clock = has_clock[:, None] & has_clock[None, :]
    M_clock_couple = ((~same_comp) & off_diag & both_clock & diff_clock).astype(_np.float32)

    omega_node = _np.array([_CB[c] if c != "none" else 0.0 for c in clock],
                           dtype=_np.float32)

    # Per-clock global node indices (for the directed clock-coupling port)
    clock_indices = {name: _np.where(clock == name)[0] for name in CLOCK_NAMES}

    return {
        "comp_id":        comp_id,
        "clock_label":    clock,
        "comp_masks":     comp_masks,
        "M_intra":        M_intra,
        "M_inter":        M_inter,
        "M_clock_couple": M_clock_couple,
        "omega_node":     omega_node,
        "clock_indices":  clock_indices,   # {clock_name: (n_k,) global indices}
    }


def build_biological_graph(
    seed: int = 42,
) -> dict:
    """
    Build the full sparse biological graph.

    Returns
    -------
    dict with:
      'A_GG'      : (n_G, n_G)  GRN adjacency (directed)
      'A_PP'      : (n_P, n_P)  PPI adjacency
      'A_MM'      : (n_M, n_M)  metabolic adjacency
      'A_GP_dogma': (n_G, n_P)  hard-wired central dogma (FIXED, not learned)
      'A_PM_enz'  : (n_P, n_M)  enzymatic port adjacency (modulated port)
      'A_MG_fb'   : (n_M, n_G)  feedback adjacency (modulated port)
      'S'         : (3, n_M)    stoichiometric matrix for 3 conserved moieties
      'n_G', 'n_P', 'n_M': layer sizes
      'edge_counts': dict  — number of edges per block

    Compartment structure (Phase A; N = n_G+n_P+n_M):
      'comp_id'       : (N,) long   compartment id (1..5) per node
      'clock_label'   : (N,) str    "circadian"/"redox"/"none" per node (numpy)
      'comp_masks'    : dict cid -> (N,) bool membership indicator
      'M_intra'       : (N, N) float intra-compartment block mask (symmetric)
      'M_inter'       : (N, N) float inter-compartment mask (port support)
      'M_clock_couple': (N, N) float circadian↔redox coupling-port support
      'omega_node'    : (N,) float per-node clock ω (0 if non-rhythmic)
      'n_compartments': int         number of compartments (5)
    """
    rng = np.random.default_rng(seed)
    cfg = list(LAYER_CONFIG.values())
    n_G, n_P, n_M = cfg[0]["n_nodes"], cfg[1]["n_nodes"], cfg[2]["n_nodes"]

    # Compartment block structure (Phase A) — deterministic (no rng)
    comp_struct = build_compartment_structure()

    A_GG = _er_adjacency(n_G, GRN_DENSITY, rng)
    # Hub amplification: top-5 out-degree nodes gain extra edges
    out_deg  = A_GG.sum(axis=1)
    hub_idx  = np.argsort(out_deg)[-5:]
    for h in hub_idx:
        extra = rng.choice(n_G, size=int(n_G * 0.1), replace=False)
        A_GG[h, extra] = 1.0
    np.fill_diagonal(A_GG, 0)

    A_PP       = _cluster_adjacency(n_P, 3, 0.25, 0.02, rng)
    A_MM       = _pathway_adjacency(n_M, 3, rng)
    A_GP_dogma = _central_dogma_adjacency(n_G, n_P, DOGMA_OFFSET, rng)
    A_PM_enz   = _enzymatic_adjacency(n_P, n_M, ENZ_FANOUT, rng)
    A_MG_fb    = _feedback_adjacency(n_M, n_G, FB_DENSITY, rng)

    # ── Stoichiometric matrix S: (n_moieties, n_M) ──────────────────────────
    # Row k = stoichiometric coefficients for moiety k in the metabolite pool
    # Three moieties from CONSERVATION_GROUPS (all in metabolome)
    n_moieties = 3
    S = np.zeros((n_moieties, n_M))
    for k, (moiety, (layer, indices, total)) in \
            enumerate(CONSERVATION_GROUPS.items()):
        if layer == "metabolome":
            S[k, indices] = 1.0
    # If fewer than n_moieties are in metabolome, leave remaining rows sparse
    # Add a random sparse row for any extra moiety slots
    for k in range(n_moieties):
        if S[k].sum() == 0:
            extra_idx = rng.choice(n_M, size=4, replace=False)
            S[k, extra_idx] = 1.0

    edge_counts = {
        "GRN (G→G)":      int(A_GG.sum()),
        "PPI (P→P)":      int(A_PP.sum()),
        "Metabolic (M→M)": int(A_MM.sum()),
        "Dogma (G→P)":    int(A_GP_dogma.sum()),
        "Enzymatic (P→M)": int(A_PM_enz.sum()),
        "Feedback (M→G)": int(A_MG_fb.sum()),
    }

    return {
        "A_GG":       torch.tensor(A_GG,       dtype=torch.float32),
        "A_PP":       torch.tensor(A_PP,       dtype=torch.float32),
        "A_MM":       torch.tensor(A_MM,       dtype=torch.float32),
        "A_GP_dogma": torch.tensor(A_GP_dogma, dtype=torch.float32),
        "A_PM_enz":   torch.tensor(A_PM_enz,   dtype=torch.float32),
        "A_MG_fb":    torch.tensor(A_MG_fb,    dtype=torch.float32),
        "S":          torch.tensor(S,           dtype=torch.float32),
        "n_G":        n_G,
        "n_P":        n_P,
        "n_M":        n_M,
        "edge_counts": edge_counts,
        # ── Compartment structure (Phase A) ──────────────────────────────────
        "comp_id":        torch.tensor(comp_struct["comp_id"],        dtype=torch.long),
        "M_intra":        torch.tensor(comp_struct["M_intra"],        dtype=torch.float32),
        "M_inter":        torch.tensor(comp_struct["M_inter"],        dtype=torch.float32),
        "M_clock_couple": torch.tensor(comp_struct["M_clock_couple"], dtype=torch.float32),
        "omega_node":     torch.tensor(comp_struct["omega_node"],     dtype=torch.float32),
        "clock_label":    comp_struct["clock_label"],          # (N,) numpy str (kept as-is)
        "comp_masks":     {cid: torch.tensor(m, dtype=torch.bool)
                           for cid, m in comp_struct["comp_masks"].items()},
        "n_compartments": N_COMPARTMENTS,
    }


def compute_plv_prior(phi_data: dict, gate_results: dict) -> dict:
    """
    Compute cross-layer Phase Locking Value matrices as WEAK PRIOR weights.

    These are used as regularization on J (not as training targets).
    Gated by amplitude so low-expressed nodes do not spuriously couple.

    Parameters
    ----------
    phi_data     : dict layer_name → (N_r_layer, T) phase array for rhythmic nodes
    gate_results : dict from detect_all_layers()

    Returns
    -------
    dict with:
      'PLV_GP' : (n_G_rhythmic, n_P_rhythmic) prior weights
      'PLV_PM' : (n_P_rhythmic, n_M_rhythmic) prior weights
      'PLV_MG' : (n_M_rhythmic, n_G_rhythmic) prior weights
    """
    def _plv(phi1, phi2, amp1, amp2):
        if phi1.shape[0] == 0 or phi2.shape[0] == 0:
            return np.zeros((phi1.shape[0], phi2.shape[0]))
        # Amplitude gate: weight by geometric mean of amplitudes
        amp_gate = np.sqrt(
            amp1.mean(axis=1)[:, None] * amp2.mean(axis=1)[None, :]
        )
        amp_gate /= amp_gate.max() + 1e-8
        plv_mat  = np.zeros((phi1.shape[0], phi2.shape[0]))
        for i in range(phi1.shape[0]):
            diff = phi1[i, :] - phi2          # (N2, T)
            plv_mat[i] = np.abs(np.mean(np.exp(1j * diff), axis=1))
        return plv_mat * amp_gate

    out = {}
    layer_names = list(LAYER_CONFIG.keys())
    pairs = [("genomics", "proteome", "PLV_GP"),
             ("proteome", "metabolome", "PLV_PM"),
             ("metabolome", "genomics", "PLV_MG")]

    for l1, l2, key in pairs:
        if l1 in phi_data and l2 in phi_data:
            amp1 = gate_results[l1]["amplitude"][gate_results[l1]["rhythmic_mask"]]
            amp2 = gate_results[l2]["amplitude"][gate_results[l2]["rhythmic_mask"]]
            plv  = _plv(phi_data[l1], phi_data[l2], amp1[:, None], amp2[:, None])
            out[key] = torch.tensor(plv, dtype=torch.float32)
        else:
            out[key] = None

    return out


def print_graph_summary(bio_graph: dict) -> None:
    print("\n── Biological Graph Summary ─────────────────────────────")
    n_G = bio_graph["n_G"]
    n_P = bio_graph["n_P"]
    n_M = bio_graph["n_M"]
    N   = n_G + n_P + n_M
    print(f"  Nodes: {N}  (G={n_G}, P={n_P}, M={n_M})")

    # Explicit max_possible lookup — avoids fragile string matching
    max_possible_map = {
        "GRN (G→G)":       n_G * n_G,
        "PPI (P→P)":       n_P * n_P,
        "Metabolic (M→M)": n_M * n_M,
        "Dogma (G→P)":     n_G * n_P,
        "Enzymatic (P→M)": n_P * n_M,
        "Feedback (M→G)":  n_M * n_G,
    }

    total_edges = 0
    for name, count in bio_graph["edge_counts"].items():
        max_possible = max_possible_map.get(name, 1)
        density      = count / max_possible if max_possible > 0 else 0.0
        print(f"  {name:20s}: {count:4d} edges  ({density:.1%} density)")
        total_edges += count
    print(f"  Total edges: {total_edges}  vs dense O(N²) = {N**2}")
    print(f"  Sparsity:    {total_edges / N**2:.2%}")
    print("─────────────────────────────────────────────────────────\n")

