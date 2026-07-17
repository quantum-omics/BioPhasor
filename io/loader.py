"""
biophasor.io.loader
===================
Auto-detect and dispatch to the correct omics data loader based on file
extension, directory structure, or explicit modality argument.
"""
from __future__ import annotations

import os
import pathlib
from typing import Optional

import anndata as ad


def auto_detect(
    path: str | os.PathLike,
    modality: Optional[str] = None,
    encode: bool = True,
    **loader_kwargs,
) -> ad.AnnData:
    """Automatically detect the omics data format and load it.

    Dispatch rules (in order):
    1. If `modality` is given, use the matching loader directly.
    2. If `path` is a directory → 10x CellRanger MEX format (singlecell).
    3. If extension is ``.h5ad`` → scanpy AnnData (singlecell).
    4. If extension is ``.loom`` → loom file (singlecell).
    5. If filename contains ``protein`` or ``proteom`` → proteomics loader.
    6. If filename contains ``metabol`` → metabolomics loader.
    7. Otherwise → RNA-seq loader.

    Parameters
    ----------
    path:
        Path to the data file or 10x directory.
    modality:
        Explicit modality override: ``"RNA"``, ``"singlecell"``,
        ``"proteomics"``, or ``"metabolomics"``.
    encode:
        Whether to automatically encode phasors (adds ``obsm['biophasor_phi']``).
    **loader_kwargs:
        Forwarded to the detected loader function.

    Returns
    -------
    anndata.AnnData
        Loaded (and optionally encoded) AnnData object.

    Examples
    --------
    >>> from biophasor.io.loader import auto_detect
    >>> adata = auto_detect("data.h5ad")
    >>> adata = auto_detect("counts.tsv", modality="RNA")
    >>> adata = auto_detect("10x_dir/", modality="singlecell")
    """
    p = pathlib.Path(path)
    ext = p.suffix.lower()
    stem = p.stem.lower()
    name = p.name.lower()

    # ── Explicit override ─────────────────────────────────────────────────────
    if modality is not None:
        return _dispatch(modality, path, encode, **loader_kwargs)

    # ── Directory → 10x MEX ──────────────────────────────────────────────────
    if p.is_dir():
        from biophasor.io.singlecell import load_singlecell
        return load_singlecell(path, encode=encode, **loader_kwargs)

    # ── Extension-based dispatch ──────────────────────────────────────────────
    if ext in (".h5ad",):
        from biophasor.io.singlecell import load_singlecell
        return load_singlecell(path, encode=encode, **loader_kwargs)

    if ext in (".loom",):
        import scanpy as sc
        adata = sc.read_loom(path)
        if encode:
            from biophasor.transform.encoder import tanh_phase_encode
            import numpy as np
            X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
            adata.obsm["biophasor_phi"] = tanh_phase_encode(X)
        return adata

    # ── Name-based dispatch ───────────────────────────────────────────────────
    if any(kw in name for kw in ("proteom", "protein", "prot_groups", "pg_matrix")):
        from biophasor.io.proteomics import load_proteomics
        return load_proteomics(path, encode=encode, **loader_kwargs)

    if any(kw in name for kw in ("metabol", "metab", "lipid", "nmr")):
        from biophasor.io.metabolomics import load_metabolomics
        return load_metabolomics(path, encode=encode, **loader_kwargs)

    # ── Default: RNA-seq ──────────────────────────────────────────────────────
    from biophasor.io.rnaseq import load_rnaseq
    return load_rnaseq(path, encode=encode, **loader_kwargs)


def _dispatch(modality: str, path, encode: bool, **kwargs) -> ad.AnnData:
    """Dispatch to loader by explicit modality name."""
    m = modality.lower()
    if m in ("rna", "rnaseq", "transcriptomics"):
        from biophasor.io.rnaseq import load_rnaseq
        return load_rnaseq(path, encode=encode, **kwargs)
    if m in ("singlecell", "sc", "scrnaseq", "10x"):
        from biophasor.io.singlecell import load_singlecell
        return load_singlecell(path, encode=encode, **kwargs)
    if m in ("proteomics", "protein", "proteome"):
        from biophasor.io.proteomics import load_proteomics
        return load_proteomics(path, encode=encode, **kwargs)
    if m in ("metabolomics", "metabolite", "metabolome"):
        from biophasor.io.metabolomics import load_metabolomics
        return load_metabolomics(path, encode=encode, **kwargs)
    raise ValueError(
        f"Unknown modality '{modality}'. Choose from: RNA, singlecell, proteomics, metabolomics."
    )
