"""
biophasor.io.proteomics — Protein abundance loader (TMT/LFQ).

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Union

import numpy as np


def load_proteomics(
    path: Union[str, Path],
    log_transform: bool = True,
    impute_missing: bool = True,
    encode: bool = True,
    encoding: str = "tanh",
) -> "anndata.AnnData":
    """
    Load a protein abundance matrix (CSV/TSV) and encode as phasors.

    Handles:
    - TMT / iTRAQ quantification tables (proteins × samples)
    - LFQ intensity tables (MaxQuant format)
    - Generic CSV/TSV (proteins × samples or samples × proteins)

    Parameters
    ----------
    path : str | Path
    log_transform : bool   apply log2 transformation (set False if already log)
    impute_missing : bool  replace NaN with minimum observed value (minimum imputation)
    encode : bool
    encoding : str

    Returns
    -------
    AnnData   proteins as vars, samples as obs; phasor in obsm['X_phasor_protein']
    """
    import anndata as ad
    import pandas as pd
    from biophasor.core.phasor import BioPhasor
    from biophasor.utils.anndata_utils import attach_phasor

    path = Path(path)
    sep = "\t" if path.suffix == ".tsv" else ","

    df = pd.read_csv(path, index_col=0, sep=sep)

    # Determine orientation: more proteins than samples → transpose
    if df.shape[0] > df.shape[1]:
        df = df.T   # now (samples × proteins)

    X = df.values.astype(np.float64)

    # Missing value imputation
    if impute_missing:
        col_min = np.nanmin(X, axis=0)
        nan_mask = np.isnan(X)
        X[nan_mask] = np.take(col_min, np.where(nan_mask)[1])

    # Log2 transform
    if log_transform:
        X = np.log2(X + 1.0)

    obs_names = list(df.index)
    var_names = list(df.columns)

    adata = ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )

    if encode:
        bp = BioPhasor(data=X, modality="protein")
        # Protein data is already log-transformed; skip extra log in encode_tanh
        bp.phase = np.pi * np.tanh((X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8))
        bp.amplitude = np.ones_like(bp.phase)
        attach_phasor(adata, bp.phase, modality="protein", amplitude=bp.amplitude)

    adata.uns["biophasor"] = {"modality": "protein"}
    return adata
