"""
biophasor.io.formats — Auto-loader and format detection.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union


def auto_load(
    path: Union[str, Path],
    modality: Optional[str] = None,
    encode: bool = True,
    encoding: str = "tanh",
    **kwargs,
) -> "anndata.AnnData":
    """
    Automatically detect omics modality and load the dataset.

    Detection rules:
    - h5ad: single-cell (load_singlecell)
    - Directory with .mtx: single-cell MEX (load_singlecell)
    - CSV/TSV: infers from filename keywords (rna, mrna, atac, protein, metabol)
    - Falls back to RNA-seq loader

    Parameters
    ----------
    path : str | Path
    modality : str | None   override auto-detection ('RNA', 'protein', etc.)
    encode : bool
    encoding : str
    **kwargs : passed to the underlying loader

    Returns
    -------
    AnnData
    """
    from biophasor.io.rnaseq import load_rnaseq
    from biophasor.io.singlecell import load_singlecell
    from biophasor.io.proteomics import load_proteomics
    from biophasor.io.metabolomics import load_metabolomics

    path = Path(path)
    name_lower = path.name.lower()

    if modality is None:
        if "protein" in name_lower or "proteom" in name_lower:
            modality = "protein"
        elif "metabol" in name_lower:
            modality = "metabolite"
        elif "atac" in name_lower:
            modality = "ATAC"
        else:
            modality = "RNA"

    # ── Route to correct loader ───────────────────────────────────────────────
    if path.suffix == ".h5ad" or path.is_dir():
        return load_singlecell(path, modality=modality, encode=encode,
                               encoding=encoding, **kwargs)
    if modality == "protein":
        return load_proteomics(path, encode=encode, encoding=encoding, **kwargs)
    if modality == "metabolite":
        return load_metabolomics(path, encode=encode, encoding=encoding, **kwargs)
    # Default: bulk RNA-seq
    return load_rnaseq(path, encode=encode, encoding=encoding, **kwargs)
