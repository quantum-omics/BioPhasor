"""
experiments._shared.revision_data — the matched CPTAC UCEC matrix every
revision experiment shares.

Builds the identical matched CPTAC UCEC complete-case matrix (7,083 genes with
zero protein NaN across the 109 samples) that the benchmark scripts
(``biophasor`` suite: exp12, exp12b, exp13) and the hardened Omics-PAC
statistics (``tumor`` suite: exp09b) run on, so their numbers are comparable
by construction.

Lives in ``_shared`` because two suites use it. It reads the shared cache
through ``common.CACHE`` and writes nothing: each script asks
``common.results_dir(<its own suite>)`` where its results go, since the two
suites have separate ``results/`` trees.

Provenance and limitations of this cohort: ``_shared/data/PROVENANCE.md``.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from experiments._shared import common

DATADIR = os.path.join(common.CACHE, "cptac_ucec")


def load_matched_cptac(complete_case: bool = True):
    """Return (rna_df, prot_df, y, genes) on identical samples/genes.

    rna/prot: (109, G) DataFrames, rows = CPTAC case ids, cols = gene symbols.
    y: int array, 1 = Tumor, 0 = Normal (95/14).
    complete_case=True restricts to the 7083 genes with zero protein NaN.
    """
    rna = pd.read_pickle(os.path.join(DATADIR, "rna.pkl.gz"))
    prot = pd.read_pickle(os.path.join(DATADIR, "protein.pkl.gz"))
    labels = pd.read_pickle(os.path.join(DATADIR, "labels.pkl.gz"))
    y = labels.iloc[:, 0].astype(str).str.contains("Tumor").astype(int).values
    if complete_case:
        cc = prot.columns[prot.notna().all(axis=0)]
        rna, prot = rna[cc].copy(), prot[cc].copy()
    return rna, prot, y, rna.columns.values


def load_clinical():
    """CPTAC UCEC clinical table (mssm), cached alongside the omics.
    Falls back to the live `cptac` package if the cache is absent."""
    cache = os.path.join(DATADIR, "clinical_mssm.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    import cptac
    clin = cptac.Ucec().get_clinical("mssm")
    try:
        clin.to_pickle(cache)
    except Exception:
        pass
    return clin
