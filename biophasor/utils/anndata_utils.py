"""
biophasor.utils.anndata_utils — AnnData integration helpers.

Phasor representations are stored in AnnData objects:
  - obsm['X_phasor_<modality>']  : (n_cells, n_features) complex  — per-cell phasors
  - varm['phasor_<modality>']    : (n_genes, n_features) complex  — per-gene phasors
  - uns['biophasor']             : metadata dict

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata as ad


def attach_phasor(
    adata: "ad.AnnData",
    phase: np.ndarray,
    modality: str = "RNA",
    amplitude: Optional[np.ndarray] = None,
    key: Optional[str] = None,
) -> "ad.AnnData":
    """
    Attach phasor phase (and optional amplitude) to an AnnData object.

    Stores the complex phasor z = A·e^{iφ} in ``adata.obsm[key]``.

    Parameters
    ----------
    adata : AnnData
    phase : np.ndarray, shape (n_obs, n_vars)   phase values
    modality : str   e.g. 'RNA', 'ATAC', 'protein'
    amplitude : np.ndarray | None   if None, unit amplitude assumed
    key : str | None   custom obsm key; defaults to 'X_phasor_<modality>'

    Returns
    -------
    AnnData (modified in place, also returned)
    """
    obsm_key = key or f"X_phasor_{modality}"
    if amplitude is None:
        amplitude = np.ones_like(phase)
    z = amplitude * np.exp(1j * phase)
    adata.obsm[obsm_key] = z

    if "biophasor" not in adata.uns:
        adata.uns["biophasor"] = {}
    adata.uns["biophasor"][modality] = {
        "obsm_key": obsm_key,
        "n_features": phase.shape[1],
        "encoding": "phasor",
    }
    return adata


def get_phasor(
    adata: "ad.AnnData",
    modality: str = "RNA",
    key: Optional[str] = None,
    return_phase: bool = True,
) -> np.ndarray:
    """
    Retrieve phasor data from AnnData.

    Parameters
    ----------
    adata : AnnData
    modality : str
    key : str | None   custom obsm key
    return_phase : bool   if True return phase; else return complex array

    Returns
    -------
    np.ndarray   shape (n_obs, n_features)
    """
    obsm_key = key or f"X_phasor_{modality}"
    if obsm_key not in adata.obsm:
        raise KeyError(f"No phasor data found in adata.obsm['{obsm_key}']. "
                       f"Call attach_phasor() first.")
    z = adata.obsm[obsm_key]
    if return_phase:
        return np.angle(z)
    return z


def phasor_to_adata(
    phase: np.ndarray,
    amplitude: Optional[np.ndarray] = None,
    obs_names: Optional[list] = None,
    var_names: Optional[list] = None,
    modality: str = "RNA",
) -> "ad.AnnData":
    """
    Create an AnnData from phasor arrays.

    Parameters
    ----------
    phase : np.ndarray, shape (n_obs, n_vars)
    amplitude : np.ndarray | None
    obs_names, var_names : list | None
    modality : str

    Returns
    -------
    AnnData with X = phase, obsm['X_phasor_<modality>'] = complex phasor
    """
    try:
        import anndata as ad
        import pandas as pd
    except ImportError:
        raise ImportError("anndata is required for AnnData utilities. pip install anndata")

    n_obs, n_vars = phase.shape
    obs_names = obs_names or [f"cell_{i}" for i in range(n_obs)]
    var_names = var_names or [f"feature_{j}" for j in range(n_vars)]

    adata = ad.AnnData(
        X=phase.astype(np.float32),
        obs=pd.DataFrame(index=obs_names),
        var=pd.DataFrame(index=var_names),
    )
    adata.uns["biophasor"] = {"modality": modality}
    attach_phasor(adata, phase, modality=modality, amplitude=amplitude)
    return adata


def adata_to_phasor(
    adata: "ad.AnnData",
    layer: Optional[str] = None,
    modality: str = "RNA",
    encode: bool = True,
    encoding: str = "tanh",
) -> np.ndarray:
    """
    Extract expression matrix from AnnData and encode as phasor phases.

    Parameters
    ----------
    adata : AnnData
    layer : str | None   layer key; uses .X if None
    modality : str   label for the BioPhasor
    encode : bool   if True, apply tanh/rank encoding
    encoding : {'tanh', 'rank'}

    Returns
    -------
    np.ndarray   phase matrix, shape (n_obs, n_vars)
    """
    from biophasor.core.phasor import BioPhasor

    if layer is not None:
        X = np.array(adata.layers[layer])
    else:
        X = np.array(adata.X)

    if not encode:
        return X

    bp = BioPhasor(data=X, modality=modality)
    if encoding == "tanh":
        bp.encode_tanh()
    else:
        bp.encode_rank()
    return bp.phase
