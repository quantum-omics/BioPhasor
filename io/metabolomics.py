"""
biophasor.io.metabolomics — Metabolite concentration loader.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np


def load_metabolomics(
    path: Union[str, Path],
    log_transform: bool = True,
    scale: bool = True,
    encode: bool = True,
    encoding: str = "tanh",
) -> "anndata.AnnData":
    """
    Load a metabolite abundance matrix and encode as phasors.

    Accepts CSV/TSV files with metabolites as rows or columns.

    Parameters
    ----------
    path : str | Path
    log_transform : bool   apply log(x+1e-6) to handle zeros
    scale : bool           unit-variance scale each metabolite across samples
    encode : bool
    encoding : str

    Returns
    -------
    AnnData   metabolites as vars, samples as obs; phasor in obsm
    """
    import anndata as ad
    import pandas as pd
    from biophasor.core.phasor import BioPhasor
    from biophasor.utils.anndata_utils import attach_phasor

    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, index_col=0, sep=sep)

    if df.shape[0] > df.shape[1]:
        df = df.T   # (samples × metabolites)

    X = df.values.astype(np.float64)

    if log_transform:
        X = np.log(X + 1e-6)

    if scale:
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    adata = ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame(index=list(df.index)),
        var=pd.DataFrame(index=list(df.columns)),
    )

    if encode:
        bp = BioPhasor.from_expression(
            X, modality="metabolite", encoding=encoding
        )
        attach_phasor(adata, bp.phase, modality="metabolite", amplitude=bp.amplitude)

    adata.uns["biophasor"] = {"modality": "metabolite"}
    return adata
