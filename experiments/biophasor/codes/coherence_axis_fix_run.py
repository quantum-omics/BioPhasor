"""
coherence_axis_fix_run.py
=========================
Manuscript revision fix: scope the coherence-based gene-selection operator to a
SUPPORTED axis.

Baseline (from exp03): coherence-over-CELLS on sparse scRNA-seq is a dropout
statistic -- C anti-correlates with detection rate (~-0.975), C>0.30 passes
~99% of genes, and the selection shows zero biological enrichment.

Fix hypothesis: coherence needs an axis with genuine repeated support.
Variants evaluated:
  (A) metacell pseudobulk -- pool cells into ~100 metacells (k-means on PCA),
      per-gene coherence ACROSS metacells (repeated support removes dropout).
  (B) detection-filtered cells -- keep genes expressed in >=10% of cells, then
      coherence over cells (removes the zero-inflation confound directly).
  (C) module/feature-axis coherence -- coherence across genes within pathway
      modules (feature axis with support); discussed for tractability.

Winner criterion: (i) coherence decouples from detection rate, and
(ii) the high-coherence selection is biologically enriched (ribosomal /
cell-cycle / housekeeping) above chance.
"""
from __future__ import annotations
import os, sys, json
import numpy as np

from experiments._shared import common
from biophasor.transform.encoder import tanh_phase_encode
from biophasor.core.operators import coherence

import scanpy as sc
import scipy.sparse as sp
sc.settings.verbosity = 0

DATADIR = common.CACHE
H5 = os.path.join(DATADIR, "GSE293316_reh.h5")
N_CELLS = 2000
SEED = 0
C_THRESHOLD = 0.30
N_TOP = 2000

# ---------------------------------------------------------------------------
# curated reference gene sets (biological grounding)
# ---------------------------------------------------------------------------
HOUSEKEEPING = ["ACTB", "GAPDH", "B2M", "TUBB", "PGK1", "TBP",
                "HPRT1", "LDHA", "PPIA", "RPLP0"]  # Eisenberg & Levanon 2013
# Tirosh et al. 2016 cell-cycle genes (scanpy regev_lab list)
CC_S = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
        "CDCA7","DTL","PRIM1","UHRF1","MLF1IP","HELLS","RFC2","RPA2","NASP",
        "RAD51AP1","GMNN","WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2",
        "RAD51","RRM2","CDC45","CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2",
        "USP1","CLSPN","POLA1","CHAF1B","BRIP1","E2F8"]
CC_G2M = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2",
          "NUF2","CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2",
          "CKAP2L","CKAP2","AURKB","BUB1","KIF11","ANP32E","TUBB4B","GTSE1",
          "KIF20B","HJURP","CDCA3","HN1","CDC20","TTK","CDC25C","KIF2C","RANGAP1",
          "NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2","KIF23","HMMR","AURKA","PSRC1",
          "ANLN","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3","GAS2L3","CBX5","CENPA"]
CELLCYCLE = CC_S + CC_G2M


def enrich(selection: set, cat_genes: list, all_genes, n_genes: int) -> dict:
    cat = set(cat_genes) & set(all_genes)
    obs = len(cat & selection)
    exp = len(selection) / n_genes * len(cat) if n_genes else 0.0
    return {"observed": int(obs), "in_data": int(len(cat)),
            "expected_by_chance": round(exp, 2),
            "fold": round(obs / exp, 3) if exp > 0 else None}


def enrich_block(selection, all_genes, n_genes):
    ribo = [g for g in all_genes if g.startswith(("RPL", "RPS"))]
    sets = {"ribosomal_RPL_RPS": ribo,
            "cell_cycle_Tirosh": CELLCYCLE,
            "housekeeping": HOUSEKEEPING}
    return {k: enrich(selection, v, all_genes, n_genes) for k, v in sets.items()}


def load():
    adata = sc.read_10x_h5(H5)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    np.random.seed(SEED)
    if adata.n_obs > N_CELLS:
        idx = np.sort(np.random.choice(adata.n_obs, N_CELLS, replace=False))
        adata = adata[idx].copy()
    # keep raw counts for pseudobulk pooling
    raw = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata, raw


def main():
    adata, raw = load()
    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)  # log-norm, cells x genes
    raw = raw.toarray() if sp.issparse(raw) else np.asarray(raw)            # raw counts
    genes = np.array(adata.var_names)
    n_genes = len(genes)
    frac_expr = (X > 0).mean(axis=0)   # per-gene detection rate over cells

    out = {
        "dataset": "GSE293316 (REH human B-ALL scRNA-seq, 10x)",
        "n_cells_used": int(adata.n_obs),
        "n_genes": int(n_genes),
        "preprocessing": "subsample 2000 (seed 0), min_genes=200/min_cells=3, "
                         "normalize_total(1e4)+log1p (as exp03)",
        "reference_sets": {"ribosomal": "RPL/RPS prefix",
                           "cell_cycle": "Tirosh 2016 S+G2M",
                           "housekeeping": "Eisenberg & Levanon 2013 (10)"},
    }

    # ---- BASELINE: coherence over CELLS (axis=0) --------------------------
    phi = tanh_phase_encode(X)
    C = coherence(phi, axis=0)
    corr_base = float(np.corrcoef(C, frac_expr)[0, 1])
    pass_base = float((C > C_THRESHOLD).mean())
    top_base = set(genes[np.argsort(-C)[:N_TOP]])
    out["baseline_over_cells"] = {
        "axis": "cells (axis=0), full gene set",
        "corr_C_vs_detection_rate": round(corr_base, 4),
        "pass_rate_C_gt_0.30": round(pass_base, 4),
        "coherence_mean": round(float(C.mean()), 4),
        "enrichment_topN": enrich_block(top_base, genes, n_genes),
    }
    print(f"[baseline] corr(C,detect)={corr_base:.4f}  pass={pass_base:.4f}")

    # ---- VARIANT A: metacell pseudobulk ----------------------------------
    # k-means on PCA of log-norm data -> ~100 metacells; sum RAW counts per
    # metacell -> pseudobulk; renormalize; coherence ACROSS metacells.
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    K_META = 100
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    pcs = PCA(n_components=50, random_state=SEED).fit_transform(Xs)
    km = KMeans(n_clusters=K_META, random_state=SEED, n_init=10).fit(pcs)
    lab = km.labels_
    pb = np.zeros((K_META, n_genes))
    for k in range(K_META):
        pb[k] = raw[lab == k].sum(axis=0)
    sizes = np.array([(lab == k).sum() for k in range(K_META)])
    # normalize each metacell to 1e4 then log1p (same recipe, metacell axis)
    pb_norm = pb / (pb.sum(axis=1, keepdims=True) + 1e-12) * 1e4
    pb_log = np.log1p(pb_norm)
    frac_meta = (pb_log > 0).mean(axis=0)         # detection over metacells
    phi_m = tanh_phase_encode(pb_log)
    C_m = coherence(phi_m, axis=0)                # per-gene coherence over metacells
    # correlation vs BOTH original per-cell detection and metacell detection
    corr_m_cell = float(np.corrcoef(C_m, frac_expr)[0, 1])
    corr_m_meta = float(np.corrcoef(C_m, frac_meta)[0, 1])
    pass_m = float((C_m > C_THRESHOLD).mean())
    top_m = set(genes[np.argsort(-C_m)[:N_TOP]])
    out["variant_A_metacell_pseudobulk"] = {
        "axis": f"{K_META} metacells (k-means/PCA), coherence across metacells",
        "median_cells_per_metacell": int(np.median(sizes)),
        "corr_C_vs_detection_rate_percell": round(corr_m_cell, 4),
        "corr_C_vs_detection_rate_metacell": round(corr_m_meta, 4),
        "pass_rate_C_gt_0.30": round(pass_m, 4),
        "coherence_mean": round(float(C_m.mean()), 4),
        "enrichment_topN_highC": enrich_block(top_m, genes, n_genes),
    }
    print(f"[A metacell] corr(C,detect_cell)={corr_m_cell:.4f} corr(C,detect_meta)={corr_m_meta:.4f} pass={pass_m:.4f}")

    # ---- VARIANT B: detection-filtered cells -----------------------------
    # keep genes expressed in >=10% of cells, coherence over cells on subset.
    keep = frac_expr >= 0.10
    genes_k = genes[keep]
    ng_k = len(genes_k)
    phi_k = tanh_phase_encode(X[:, keep])
    C_k = coherence(phi_k, axis=0)
    corr_k = float(np.corrcoef(C_k, frac_expr[keep])[0, 1])
    pass_k = float((C_k > C_THRESHOLD).mean())
    top_k = set(genes_k[np.argsort(-C_k)[:min(N_TOP, ng_k)]])
    out["variant_B_detection_filtered_cells"] = {
        "axis": "cells (axis=0), genes expressed in >=10% of cells",
        "n_genes_kept": int(ng_k),
        "corr_C_vs_detection_rate": round(corr_k, 4),
        "pass_rate_C_gt_0.30": round(pass_k, 4),
        "coherence_mean": round(float(C_k.mean()), 4),
        "enrichment_topN_highC": enrich_block(top_k, genes_k, ng_k),
    }
    print(f"[B detect>=10%] n_genes={ng_k} corr(C,detect)={corr_k:.4f} pass={pass_k:.4f}")

    # ---- VARIANT C: metacell + detection-filter combined -----------------
    # supported axis (metacells) AND remove very-sparse genes: the clean fix.
    keepm = frac_expr >= 0.10
    pb_logk = pb_log[:, keepm]
    genes_mk = genes[keepm]
    ng_mk = len(genes_mk)
    frac_meta_k = (pb_logk > 0).mean(axis=0)
    phi_mk = tanh_phase_encode(pb_logk)
    C_mk = coherence(phi_mk, axis=0)
    corr_mk = float(np.corrcoef(C_mk, frac_expr[keepm])[0, 1])
    pass_mk = float((C_mk > C_THRESHOLD).mean())
    top_mk = set(genes_mk[np.argsort(-C_mk)[:min(N_TOP, ng_mk)]])
    out["variant_C_metacell_plus_detectionfilter"] = {
        "axis": "metacells across, genes expressed in >=10% of cells",
        "n_genes_kept": int(ng_mk),
        "corr_C_vs_detection_rate_percell": round(corr_mk, 4),
        "pass_rate_C_gt_0.30": round(pass_mk, 4),
        "coherence_mean": round(float(C_mk.mean()), 4),
        "enrichment_topN_highC": enrich_block(top_mk, genes_mk, ng_mk),
    }
    print(f"[C metacell+detect] n_genes={ng_mk} corr(C,detect)={corr_mk:.4f} pass={pass_mk:.4f}")

    _out_path = os.path.join(common.results_dir("biophasor"), "coherence_axis_fix.json")
    json.dump(out, open(_out_path, "w"), indent=1)
    print("saved", _out_path)
    return out


if __name__ == "__main__":
    main()
