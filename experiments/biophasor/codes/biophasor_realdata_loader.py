#!/usr/bin/env python
"""
biophasor_realdata_loader.py

Pulls two small public datasets from NCBI GEO and runs them through the ACTUAL
biophasor package to reproduce the real-data feasibility test:

  1. Cell-cycle phase assignment  — GSE293316, REH human leukemia scRNA-seq (10x .h5)
  2. Circadian rhythmicity         — GSE171432, WT mouse liver time-series (FPKM table)

Requires the `biophasor` package installed (editable is fine) plus scanpy/anndata/pandas.
No biophasor source is modified; every result comes from the unmodified package API.

Usage:
    python biophasor_realdata_loader.py --outdir ./out
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request
import numpy as np, pandas as pd

import biophasor  # noqa: F401

from experiments._shared import common

FTP = {
    "cellcycle_h5": ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293316/"
                     "suppl/GSE293316_reh_filtered_feature_bc_matrix.h5"),
    "circadian_fpkm": ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171432/"
                       "suppl/GSE171432_genes_fpkm_table.tsv.gz"),
}

# BioPhasor circadian markers are HUMAN symbols; GSE171432 is MOUSE.
HUMAN_TO_MOUSE = {
    "CLOCK": "Clock", "BMAL1": "Arntl", "PER1": "Per1", "PER2": "Per2", "PER3": "Per3",
    "CRY1": "Cry1", "CRY2": "Cry2", "RORA": "Rora", "REV-ERBA": "Nr1d1", "CSNK1E": "Csnk1e",
}
CIRC_POS_EXTRA = ["Dbp", "Nr1d2", "Tef", "Npas2", "Ciart"]     # rhythmic positive controls
CIRC_NEG = ["Actb", "Gapdh", "Hprt", "Tbp"]                    # housekeeping negatives


def fetch(url: str, dst: str) -> str:
    if not os.path.exists(dst):
        print(f"  downloading {os.path.basename(dst)} ...")
        urllib.request.urlretrieve(url, dst)
    print(f"  {dst}: {os.path.getsize(dst):,} bytes")
    return dst


# ----------------------------------------------------------------------------- cell cycle
def run_cellcycle(h5_path: str, n_cells: int = 2000, seed: int = 0) -> dict:
    import scanpy as sc
    from biophasor.dynamics.cellcycle import CellCyclePhasor
    from biophasor.core.constants import CANONICAL_MARKER_GENES

    adata = sc.read_10x_h5(h5_path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    if adata.n_obs > n_cells:
        idx = np.random.RandomState(seed).choice(adata.n_obs, n_cells, replace=False)
        adata = adata[idx].copy()
    adata.layers["counts"] = adata.X.copy()

    # reference: scanpy score_genes_cell_cycle (regev_lab / Tirosh S & G2M lists)
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    import scipy.sparse as sp
    if sp.issparse(adata.X):        # CellCyclePhasor.assign expects a dense .X
        adata.X = adata.X.toarray()
    s_genes = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6","CDCA7",
               "DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN","WDR76",
               "SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45","CDC6",
               "EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
    g2m_genes = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2",
                 "CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2",
                 "AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1",
                 "CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2",
                 "KIF23","HMMR","AURKA","PSRC1","ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3",
                 "GAS2L3","CBX5","CENPA"]
    sc.tl.score_genes_cell_cycle(adata, s_genes=[g for g in s_genes if g in adata.var_names],
                                 g2m_genes=[g for g in g2m_genes if g in adata.var_names])
    ref = adata.obs["phase"].astype(str).values  # {G1,S,G2M}

    # biophasor assignment (unmodified); merge G2+M -> G2M for 3-way comparison
    cc = CellCyclePhasor()
    labels, phi = cc.assign(adata)
    bp = np.array(["G2M" if l in ("G2", "M") else l for l in labels])

    from sklearn.metrics import adjusted_rand_score, confusion_matrix
    cats = ["G1", "S", "G2M"]
    cm = confusion_matrix(ref, bp, labels=cats)
    acc = float((ref == bp).mean())
    recall = {c: float(cm[i].max() and cm[i, i] / cm[i].sum()) for i, c in enumerate(cats)}
    markers = set(sum(CANONICAL_MARKER_GENES.values(), []))
    return {"n_cells_used": int(adata.n_obs),
            "markers_present": f"{len(markers & set(adata.var_names))}/{len(markers)}",
            "agreement_accuracy": round(acc, 4),
            "adjusted_rand_index": round(float(adjusted_rand_score(ref, bp)), 4),
            "per_phase_recall": {k: round(v, 4) for k, v in recall.items()},
            "biophasor_counts": {c: int((bp == c).sum()) for c in cats},
            "reference_counts": {c: int((ref == c).sum()) for c in cats}}


# ----------------------------------------------------------------------------- circadian
def run_circadian(fpkm_path: str) -> dict:
    from biophasor.dynamics.circadian import CircadianPhasor

    df = pd.read_csv(fpkm_path, sep="\t", index_col=0)
    wt = [c for c in df.columns if c.startswith("WT_")]     # exclude KO columns
    zts = [0, 4, 8, 12, 16, 20]
    # replicate-mean per ZT -> (6 timepoints x genes)
    mat = pd.DataFrame({
        zt: df[[c for c in wt if c.startswith(f"WT_ZT{zt}_")]].mean(axis=1) for zt in zts
    }).T                                                     # rows = ZT, cols = genes
    expressed = mat.columns[(mat.mean(axis=0) > 1.0)]
    X = np.log1p(mat[expressed].values)                     # (6, G)

    circ = CircadianPhasor(period=24.0, sample_interval=4.0)
    score = circ.rhythmicity_score(X)
    phase = circ.infer_phase(X)
    score = pd.Series(score, index=expressed)
    peak_zt = pd.Series([CircadianPhasor.phase_to_zt(p) for p in phase], index=expressed)

    pos = [HUMAN_TO_MOUSE[h] for h in HUMAN_TO_MOUSE] + CIRC_POS_EXTRA
    pos = [g for g in pos if g in score.index]
    neg = [g for g in CIRC_NEG if g in score.index]
    recall = float((score[pos] >= 0.3).mean())
    spec = float((score[neg] < 0.3).mean())
    return {"n_timepoints": 6, "n_wt_columns": len(wt), "genes_tested": int(len(expressed)),
            "positive_recall": round(recall, 4), "negative_specificity": round(spec, 4),
            "peak_ZT": {g: round(float(peak_zt[g]), 2) for g in pos},
            "score": {g: round(float(score[g]), 3) for g in pos + neg}}


def main():
    ap = argparse.ArgumentParser()
    # Default to the persistent cache at experiments/data/raw (shared with the
    # other experiment scripts) so re-runs reuse already-downloaded datasets
    # instead of re-fetching from GEO.
    ap.add_argument("--outdir", default=common.CACHE)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[1/2] cell-cycle (GSE293316)")
    h5 = fetch(FTP["cellcycle_h5"], os.path.join(args.outdir, "GSE293316_reh.h5"))
    cc = run_cellcycle(h5)
    print("  ->", json.dumps(cc))

    print("[2/2] circadian (GSE171432)")
    fpkm = fetch(FTP["circadian_fpkm"], os.path.join(args.outdir, "GSE171432_fpkm.tsv.gz"))
    circ = run_circadian(fpkm)
    print("  ->", json.dumps(circ))

    out = os.path.join(common.results_dir("biophasor"),
                       "biophasor_realdata_results.json")
    json.dump({"cellcycle": cc, "circadian": circ}, open(out, "w"), indent=1)
    print("done -> data", args.outdir, "| results", out)


if __name__ == "__main__":
    main()
