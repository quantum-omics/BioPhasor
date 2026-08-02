"""
biophasor.cst.pathway_cst — pathway-resolved Cell State Tensor builder.

NON-BREAKING addition. This module does NOT modify ``CellStateTensor`` or any
existing CST axis labels; it provides a NEW builder that feeds a
*pathway-structured* complex tensor into the existing ``CellStateTensor`` class.

Motivation
----------
The flat gene-resolved CST (exp07) uses the raw gene axis as its regulatory
mode. That axis carries genuinely high-dimensional inter-gene structure and is
NOT low-rank (mode-1 SVD: rank-50 captures only ~50 % of energy). The sibling
NeuroPhasor MST is instead rooted on the AAL-90 anatomical brain-region atlas —
an axis with real, coarse structure — and compresses to CP rank-3 at ~1.9 %
error.

This builder gives the CST regulatory axis its biological atlas (MSigDB
Hallmark, ``biophasor.core.pathways``) so we can test the direct counterpart:
does aggregating genes into pathway programs make the CST low-rank?

Two designs are provided:

  (i) "aggregate" (DEFAULT, recommended) — pathway-AGGREGATED. One regulatory
      index per pathway. The phasor for a pathway is the circular mean of its
      member genes' phasors (per modality, per sample):
          Z[p, m, s] = R_pms * exp(i * mean_angle)
      where the mean is over member genes present in the data. The amplitude is
      the resultant-vector length (mean phase coherence, PLV) of the member
      genes — a bounded [0,1] readout of how phase-aligned a program is. This is
      the clean counterpart of the region-aggregated MST: regulatory axis length
      = n_pathways (~50), not n_genes (~7000).
      Tensor shape: (n_pathways, 2 modalities, n_samples).

  (ii) "block" — pathway-BLOCKED. Genes are kept but REORDERED/grouped so the
      regulatory axis has contiguous pathway blocks (a gene may appear in more
      than one pathway, so it can be duplicated across blocks). Unit-modulus
      per-gene phasors, exactly as in the flat CST, just permuted/grouped.
      Tensor shape: (n_grouped_genes, 2 modalities, n_samples). Reported as a
      secondary view; block ordering alone does not change the SVD spectrum of
      the unfolding (a permutation of rows), so this design is mainly for
      block-structured / grouped decompositions.

Both return an unmodified ``CellStateTensor`` plus a metadata dict.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd

from biophasor.transform.encoder import tanh_phase_encode
from biophasor.cst.tensor import CellStateTensor


# ── helpers ──────────────────────────────────────────────────────────────────
def _co_observed_genes(rna_df: pd.DataFrame, protein_df: pd.DataFrame) -> np.ndarray:
    """Genes present in both matrices with no NaN in the protein layer.

    Matches the exp07 convention: intersect columns, then drop genes with any
    NaN in the (aligned) protein matrix.
    """
    common = [g for g in rna_df.columns if g in set(protein_df.columns)]
    common = list(dict.fromkeys(common))  # dedupe, preserve order
    prot = protein_df[common]
    complete = ~np.isnan(prot.values).any(axis=0)
    return np.asarray(common, dtype=object)[complete]


def _circular_mean_phasor(unit_phasors: np.ndarray, axis: int) -> np.ndarray:
    """Resultant vector of a set of unit phasors along ``axis``.

    Returns the *mean* complex phasor: magnitude = mean phase coherence (PLV,
    in [0,1]), angle = circular mean angle. For n unit phasors z_k:
        R * exp(i*mu) = (1/n) * sum_k z_k
    """
    return unit_phasors.mean(axis=axis)


# ── primary builder: pathway-aggregated CST ──────────────────────────────────
def build_pathway_cst(
    rna_df: pd.DataFrame,
    protein_df: pd.DataFrame,
    atlas: dict[str, list[str]],
    design: str = "aggregate",
    min_genes: int = 5,
    log_transform_rna: bool = True,
    log_transform_protein: bool = False,
    return_meta: bool = True,
):
    """Build a pathway-resolved CST from matched RNA + protein matrices.

    Parameters
    ----------
    rna_df, protein_df : pd.DataFrame
        Samples (rows) x gene SYMBOLS (columns). The RNA index order is used and
        protein is reindexed to match.
    atlas : dict[str, list[str]]
        Pathway name -> member gene symbols (e.g. ``get_pathway_atlas()``).
    design : {"aggregate", "block"}
        "aggregate" (default): one regulatory index per pathway, phasor = circular
        mean of member-gene phasors. Shape (n_pathways, 2, n_samples).
        "block": genes kept but grouped by pathway (rows may duplicate across
        pathways). Shape (n_grouped_genes, 2, n_samples).
    min_genes : int
        Drop pathways with fewer than this many co-observed member genes.
    log_transform_rna, log_transform_protein : bool
        Passed to ``tanh_phase_encode`` per modality (RNA is raw counts -> log;
        protein is already log-space -> no log). Matches exp07.
    return_meta : bool
        If True, return (CellStateTensor, meta_dict); else just the tensor.

    Returns
    -------
    CellStateTensor  (regulatory = pathways or grouped genes,
                      temporal = 2 modalities [RNA, protein],
                      homeostatic = samples)
    meta : dict  (only if return_meta)
    """
    if design not in ("aggregate", "block"):
        raise ValueError("design must be 'aggregate' or 'block'")

    # Align samples: use RNA index order, reindex protein to it.
    protein_df = protein_df.reindex(index=rna_df.index)
    genes = _co_observed_genes(rna_df, protein_df)
    gene_set = set(genes.tolist())
    gidx = {g: i for i, g in enumerate(genes)}

    # Encode both modalities on the co-observed gene columns -> phases (samples, genes)
    rna_sub = rna_df[list(genes)].values
    prot_sub = protein_df[list(genes)].values
    phi_rna = tanh_phase_encode(rna_sub, log_transform=log_transform_rna)
    phi_prot = tanh_phase_encode(prot_sub, log_transform=log_transform_protein)
    # Unit phasors, shape (genes, samples) per modality
    zr = np.exp(1j * phi_rna.T)
    zp = np.exp(1j * phi_prot.T)
    n_samples = zr.shape[1]

    # Map pathways -> co-observed member gene indices
    pathway_members: dict[str, np.ndarray] = {}
    for name in sorted(atlas.keys()):
        members = [g for g in atlas[name] if g in gene_set]
        if len(members) >= min_genes:
            pathway_members[name] = np.array([gidx[g] for g in members], dtype=int)

    if not pathway_members:
        raise ValueError("No pathway has >= min_genes co-observed members.")

    pathway_names = sorted(pathway_members.keys())

    # Gene->pathway coverage stats
    covered_genes = set()
    for idxs in pathway_members.values():
        covered_genes.update(idxs.tolist())

    if design == "aggregate":
        R = len(pathway_names)
        Z = np.zeros((R, 2, n_samples), dtype=np.complex128)
        coherence_rna = np.zeros(R)
        coherence_prot = np.zeros(R)
        pathway_sizes = np.zeros(R, dtype=int)
        for i, name in enumerate(pathway_names):
            idxs = pathway_members[name]
            pathway_sizes[i] = len(idxs)
            mr = _circular_mean_phasor(zr[idxs, :], axis=0)  # (samples,)
            mp = _circular_mean_phasor(zp[idxs, :], axis=0)
            Z[i, 0, :] = mr
            Z[i, 1, :] = mp
            # mean coherence across samples (PLV of the program)
            coherence_rna[i] = float(np.abs(mr).mean())
            coherence_prot[i] = float(np.abs(mp).mean())
        reg_names = pathway_names
        meta = {
            "design": "aggregate",
            "pathway_names": pathway_names,
            "pathway_sizes": pathway_sizes.tolist(),
            "coherence_rna": coherence_rna.tolist(),
            "coherence_protein": coherence_prot.tolist(),
        }
    else:  # block
        # Concatenate member phasors grouped by pathway (duplicates allowed).
        blocks = []
        block_names = []
        block_bounds = {}
        cursor = 0
        for name in pathway_names:
            idxs = pathway_members[name]
            block_bounds[name] = (cursor, cursor + len(idxs))
            cursor += len(idxs)
            blocks.append(idxs)
            block_names.extend([name] * len(idxs))
        order = np.concatenate(blocks)
        Zr = zr[order, :]
        Zp = zp[order, :]
        Z = np.stack([Zr, Zp], axis=1)  # (n_grouped_genes, 2, samples)
        reg_names = block_names
        meta = {
            "design": "block",
            "pathway_names": pathway_names,
            "block_bounds": block_bounds,
            "n_grouped_genes": int(Z.shape[0]),
        }

    meta.update({
        "n_pathways_kept": len(pathway_names),
        "n_co_observed_genes": int(len(genes)),
        "n_genes_in_any_pathway": int(len(covered_genes)),
        "gene_coverage_fraction": float(len(covered_genes) / len(genes)),
        "n_samples": int(n_samples),
        "min_genes": int(min_genes),
        "cst_shape": tuple(int(x) for x in Z.shape),
    })

    cst = CellStateTensor(
        tensor=Z.astype(np.complex128),
        regulatory_names=list(reg_names),
        temporal_names=["RNA", "protein"],
        metadata={"builder": "pathway_cst", "design": design},
    )
    if return_meta:
        return cst, meta
    return cst


__all__ = ["build_pathway_cst"]
