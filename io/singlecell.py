"""
biophasor.io.singlecell — Single-cell omics loader (scRNA-seq, scATAC-seq).

Wraps scanpy's IO and attaches phasor representations.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np


def load_singlecell(
    path: Union[str, Path],
    n_top_genes: int = 2000,
    min_cells: int = 3,
    min_genes: int = 200,
    normalize: bool = True,
    modality: str = "RNA",
    encode: bool = True,
    encoding: str = "tanh",
) -> "anndata.AnnData":
    """
    Load a single-cell omics dataset (h5ad / MEX format) with standard QC.

    Pipeline:
        1. Load data
        2. Filter cells (min_genes) and genes (min_cells)
        3. Library-size normalise + log1p
        4. Select highly variable genes (n_top_genes)
        5. Encode phasors (tanh-phase by default)

    Parameters
    ----------
    path : str | Path
    n_top_genes : int    number of highly variable genes to retain
    min_cells : int      filter genes expressed in < min_cells cells
    min_genes : int      filter cells with < min_genes detected genes
    normalize : bool
    modality : str       'RNA' or 'ATAC'
    encode : bool
    encoding : str

    Returns
    -------
    AnnData   preprocessed with phasors in obsm
    """
    try:
        import scanpy as sc
        import anndata as ad
    except ImportError:
        raise ImportError("scanpy and anndata are required. pip install scanpy anndata")

    from biophasor.core.phasor import BioPhasor
    from biophasor.utils.anndata_utils import attach_phasor

    path = Path(path)

    # ── Load ──────────────────────────────────────────────────────────────────
    if path.suffix in (".h5ad",):
        adata = sc.read_h5ad(str(path))
    elif path.is_dir():
        adata = sc.read_10x_mtx(str(path), var_names="gene_symbols", cache=False)
    elif path.suffix in (".h5", ".hdf5"):
        adata = sc.read_10x_h5(str(path))
    elif path.suffix in (".loom",):
        adata = sc.read_loom(str(path))
    else:
        raise ValueError(f"Unsupported single-cell format: {path}")

    # ── QC Filtering ──────────────────────────────────────────────────────────
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)

    # ── Normalise ─────────────────────────────────────────────────────────────
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # ── Highly variable genes ─────────────────────────────────────────────────
    if n_top_genes is not None:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        adata = adata[:, adata.var["highly_variable"]].copy()

    # ── Encode phasors ────────────────────────────────────────────────────────
    if encode:
        X = np.array(adata.X, dtype=np.float64)
        bp = BioPhasor.from_expression(X, modality=modality, encoding=encoding)
        attach_phasor(adata, bp.phase, modality=modality, amplitude=bp.amplitude)

    adata.uns.setdefault("biophasor", {})["modality"] = modality
    return adata
