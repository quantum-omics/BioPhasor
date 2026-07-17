"""
biophasor.io.rnaseq — Bulk RNA-seq loader.

Loads counts matrix, applies library-size normalisation, and encodes as phasors.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np


def load_rnaseq(
    path: Union[str, Path],
    n_top_genes: Optional[int] = None,
    min_counts: int = 5,
    normalize: bool = True,
    encode: bool = True,
    encoding: str = "tanh",
) -> "anndata.AnnData":
    """
    Load a bulk RNA-seq counts matrix and return an AnnData with phasor phases.

    Accepts the following formats:
    - TSV / CSV (genes as rows or columns)
    - MEX (matrix.mtx + barcodes.tsv + features.tsv directory)
    - H5AD (already an AnnData)

    Parameters
    ----------
    path : str | Path
        Path to the counts file or directory.
    n_top_genes : int | None
        If set, retain only the top-N most variable genes.
    min_counts : int
        Minimum total counts per gene (filter low-expressed genes).
    normalize : bool
        Library-size normalise counts to 10 000 (CPM-style).
    encode : bool
        If True, attach tanh-phase pasphasor to adata.obsm.
    encoding : {'tanh', 'rank'}

    Returns
    -------
    AnnData
        Raw counts in .X, phasor in .obsm['X_phasor_RNA'] (if encode=True).
    """
    import anndata as ad
    import pandas as pd
    from biophasor.core.phasor import BioPhasor
    from biophasor.utils.anndata_utils import attach_phasor

    path = Path(path)

    # ── Load ──────────────────────────────────────────────────────────────────
    if path.suffix in (".h5ad", ".h5"):
        adata = ad.read_h5ad(path)
    elif path.is_dir():
        adata = ad.read_10x_mtx(path, var_names="gene_symbols", cache=False)
    elif path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        df = pd.read_csv(path, index_col=0, sep=sep)
        adata = ad.AnnData(X=df.values.T.astype(np.float32),
                           obs=pd.DataFrame(index=df.columns),
                           var=pd.DataFrame(index=df.index))
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")

    # ── Pre-process ───────────────────────────────────────────────────────────
    X = np.array(adata.X, dtype=np.float64)

    # Filter low-expressed genes
    gene_totals = X.sum(axis=0)
    mask = gene_totals >= min_counts
    X = X[:, mask]
    adata = adata[:, mask].copy()

    # Library-size normalise
    if normalize:
        lib_sizes = X.sum(axis=1, keepdims=True)
        X = X / (lib_sizes + 1e-8) * 1e4  # CPM-like, scaled to 10 000

    # Select highly variable genes
    if n_top_genes is not None and n_top_genes < X.shape[1]:
        gene_var = X.var(axis=0)
        top_idx = np.argsort(gene_var)[-n_top_genes:]
        X = X[:, top_idx]
        adata = adata[:, top_idx].copy()

    # Store processed counts back
    adata.X = X.astype(np.float32)

    # ── Encode phasors ────────────────────────────────────────────────────────
    if encode:
        bp = BioPhasor.from_expression(X, modality="RNA", encoding=encoding)
        attach_phasor(adata, bp.phase, modality="RNA", amplitude=bp.amplitude)

    adata.uns.setdefault("biophasor", {})["modality"] = "RNA"
    return adata
