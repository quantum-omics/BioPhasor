"""
exp01_cellcycle_assignment.py
==============================
Experiment 1: Cell-Cycle Phase Assignment on real single-cell RNA-seq.

Reproduces the measured cell-cycle result in the BioPhasor manuscript
(Section "Cell Cycle Phase Assignment", GSE293316 REH human B-ALL scRNA-seq).
Runs the *unmodified* `biophasor.dynamics.cellcycle.CellCyclePhasor.assign`
against a scanpy `score_genes_cell_cycle` method reference and reports the
(near-chance) agreement, diagnosing the fixed reference-angle aggregation.

Generates:
  cellcycle_real_confusion.png  -- confusion matrix + per-cell phase-angle rose
  cellcycle_real_results.json   -- accuracy, ARI, per-phase recall, counts

Data: GSE293316 (10x .h5). Uses the cached copy under experiments/data/raw/ if
present, otherwise downloads it from NCBI GEO.

Run from project root:
    python biophasor/experiments/codes/exp01_cellcycle_assignment.py
"""
from __future__ import annotations
import os
import sys
import json
import urllib.request

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path / import bootstrap — make `biophasor` importable from a source checkout
# ---------------------------------------------------------------------------
import biophasor  # noqa: F401

from biophasor.dynamics.cellcycle import CellCyclePhasor
from biophasor.core.constants import CANONICAL_MARKER_GENES
from sklearn.metrics import confusion_matrix, adjusted_rand_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUITE = "biophasor"
from experiments._shared import common
DATADIR = common.CACHE
OUTDIR = common.results_dir(SUITE)
FIGDIR = common.manuscript_figs(SUITE)   # figures are written ONCE, where the .tex reads them
os.makedirs(DATADIR, exist_ok=True)

H5_NAME = "GSE293316_reh.h5"
H5_URL = ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293316/"
          "suppl/GSE293316_reh_filtered_feature_bc_matrix.h5")
N_CELLS = 2000
SEED = 0

# Regev-lab / Tirosh S and G2M reference gene sets (scanpy method reference)
S_GENES = ["MCM5","PCNA","TYMS","FEN1","MCM2","MCM4","RRM1","UNG","GINS2","MCM6",
           "CDCA7","DTL","PRIM1","UHRF1","HELLS","RFC2","RPA2","NASP","RAD51AP1","GMNN",
           "WDR76","SLBP","CCNE2","UBR7","POLD3","MSH2","ATAD2","RAD51","RRM2","CDC45",
           "CDC6","EXO1","TIPIN","DSCC1","BLM","CASP8AP2","USP1","CLSPN","POLA1","CHAF1B",
           "BRIP1","E2F8"]
G2M_GENES = ["HMGB2","CDK1","NUSAP1","UBE2C","BIRC5","TPX2","TOP2A","NDC80","CKS2","NUF2",
             "CKS1B","MKI67","TMPO","CENPF","TACC3","FAM64A","SMC4","CCNB2","CKAP2L","CKAP2",
             "AURKB","BUB1","KIF11","ANLN","TUBB4B","GTSE1","KIF20B","HJURP","CDCA3","HN1",
             "CDC20","TTK","CDC25C","KIF2C","RANGAP1","NCAPD2","DLGAP5","CDCA2","CDCA8","ECT2",
             "KIF23","HMMR","AURKA","PSRC1","LBR","CKAP5","CENPE","CTCF","NEK2","G2E3",
             "GAS2L3","CBX5","CENPA"]


def _fetch(url: str, dst: str) -> str:
    if not os.path.exists(dst):
        print(f"  downloading {os.path.basename(dst)} from GEO ...")
        urllib.request.urlretrieve(url, dst)
    print(f"  data: {dst} ({os.path.getsize(dst):,} bytes)")
    return dst


def run():
    import scanpy as sc
    sc.settings.verbosity = 0

    h5 = _fetch(H5_URL, os.path.join(DATADIR, H5_NAME))
    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    np.random.seed(SEED)
    # Recorded before the subsample: the manuscript quotes the pool this draw
    # came out of, and after the next line adata.n_obs is the drawn count.
    n_cells_available = int(adata.n_obs)
    if adata.n_obs > N_CELLS:
        idx = np.sort(np.random.choice(adata.n_obs, N_CELLS, replace=False))
        adata = adata[idx].copy()

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # --- method reference: scanpy score_genes_cell_cycle ---
    s_present = [g for g in S_GENES if g in adata.var_names]
    g2m_present = [g for g in G2M_GENES if g in adata.var_names]
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g2m_present)
    ref = adata.obs["phase"].values.astype(str)  # {G1, S, G2M}

    # --- BioPhasor assignment: legacy fixed-angle vs fixed continuous axis ---
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    order = ["G1", "S", "G2M"]
    markers = set(sum(CANONICAL_MARKER_GENES.values(), []))

    def _score(method):
        a = adata.copy()
        labels, phi = CellCyclePhasor().assign(a, add_to_obs=False, method=method)
        bp = np.array(["G2M" if p in ("G2", "M") else p for p in labels])
        cm = confusion_matrix(ref, bp, labels=order)
        acc = float((bp == ref).mean())
        ari = float(adjusted_rand_score(ref, bp))
        recall = {c: float(cm[i].sum() and cm[i, i] / cm[i].sum()) for i, c in enumerate(order)}
        return dict(labels=labels, phi=phi, bp=bp, cm=cm, acc=acc, ari=ari, recall=recall)

    old = _score("fixed")        # legacy fixed reference-angle snap (feasibility result)
    new = _score("continuous")   # data-driven continuous cell-cycle axis (fix)

    result = {
        "dataset": "GSE293316 (REH human B-ALL scRNA-seq, 10x)",
        "n_cells_used": int(adata.n_obs),
        # The manuscript states the subsample fraction — "2,000 cells subsampled
        # from 7,433" — and both halves of it are measurements of this dataset.
        # Only the post-subsample count was being written, so the pool size the
        # text quotes had no receipt. n_cells_available is read before the
        # subsample; it is a property of the filtered matrix, not a constant.
        "n_cells_available": int(n_cells_available),
        "markers_present": f"{len(markers & set(adata.var_names))}/{len(markers)}",
        # markers_present is a string, so its two counts are invisible to the
        # number guard. The manuscript quotes the count ("all 42 canonical
        # Tirosh markers present"), so store it as a number too.
        "n_markers_present": int(len(markers & set(adata.var_names))),
        "n_markers_total": int(len(markers)),
        "method": "continuous data-driven cell-cycle axis (Plan-II fix)",
        "agreement_accuracy": round(new["acc"], 4),
        "adjusted_rand_index": round(new["ari"], 4),
        "per_phase_recall": {k: round(v, 4) for k, v in new["recall"].items()},
        "biophasor_counts": {c: int((new["bp"] == c).sum()) for c in order},
        "reference_counts": {c: int((ref == c).sum()) for c in order},
        "baseline_fixed_angle": {
            "agreement_accuracy": round(old["acc"], 4),
            "adjusted_rand_index": round(old["ari"], 4),
            "per_phase_recall": {k: round(v, 4) for k, v in old["recall"].items()},
            "biophasor_counts": {c: int((old["bp"] == c).sum()) for c in order},
        },
        # The manuscript quotes the CONTRAST between the two assignments — "a
        # 34-point accuracy gain on identical inputs", and the fraction of
        # cells the legacy baseline collapsed into G2/M — not just the two
        # accuracies. A derived form the text quotes in its own right gets its
        # own receipt; the guard matches values, not arithmetic.
        #
        # KNOWN DISCREPANCY, deliberately not smoothed over: the gain computes
        # to 34.65 points (0.69 - 0.3435), which is 35 at the two significant
        # figures the manuscript writes, not 34. The text is off by one point.
        # check_numbers does NOT currently flag it, because an unrelated
        # receipt — the pathway atlas's 33.95% gene coverage — rounds to 34 and
        # occupies the same slot in the value pool. That collision is the
        # guard's known blind spot at low precision, not a pass. Storing the
        # honest 34.65 here so the disagreement is on disk rather than hidden
        # behind a coincidence.
        "continuous_vs_fixed": {
            "accuracy_gain_points": round(100.0 * (new["acc"] - old["acc"]), 1),
            "fixed_frac_cells_called_G2M": round(
                float((old["bp"] == "G2M").mean()), 4),
            "fixed_pct_cells_called_G2M": round(
                100.0 * float((old["bp"] == "G2M").mean()), 1),
        },
        "verdict": (
            f"reproduces (continuous axis: acc {new['acc']:.2f}, ARI {new['ari']:.2f}, "
            f"G1 recall {new['recall']['G1']:.2f}); legacy fixed-angle was near-chance "
            f"(acc {old['acc']:.2f}, ARI {old['ari']:.2f}, G1 recall {old['recall']['G1']:.3f})"
        ),
    }

    _plot(new["cm"], new["acc"], new["ari"], new["labels"], new["phi"], old=old, order=order)
    json.dump(result, open(os.path.join(OUTDIR, "cellcycle_real_results.json"), "w"), indent=1)
    print("  ->", json.dumps(result))
    return result


def _plot(cm, acc, ari, phase_labels, phi_cells, old=None, order=("G1", "S", "G2M")):
    """Emit three single-panel PNGs (combined later in LaTeX):

      cellcycle_confusion_fixed.png       fixed-angle vs scanpy reference (recall)
      cellcycle_confusion_continuous.png  continuous axis vs reference (recall)
      cellcycle_phase_rose.png            per-cell continuous phase φ by call
    """
    from experiments._shared.figstyle import apply_style
    apply_style()
    order = list(order)

    def _confusion_png(cm_, acc_, title, fname):
        cmn = cm_ / np.clip(cm_.sum(axis=1, keepdims=True), 1, None)
        fig, ax = plt.subplots(figsize=(2.65, 2.7))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(3)); ax.set_xticklabels(order)
        ax.set_yticks(range(3)); ax.set_yticklabels(order)
        ax.set_xlabel("BioPhasor call"); ax.set_ylabel("scanpy reference")
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm_[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "#222", fontsize=7.5)
        ax.set_title(title, loc="left")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("recall (row-normalized)")
        path = os.path.join(FIGDIR, fname)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [figure] {path}")

    if old is not None:
        _confusion_png(old["cm"], old["acc"], f"Fixed-angle (acc {old['acc']:.2f})",
                       "cellcycle_confusion_fixed.png")
    _confusion_png(cm, acc, f"Continuous axis (acc {acc:.2f})",
                   "cellcycle_confusion_continuous.png")

    # Polar rose of per-cell continuous phase angle φ, coloured by BioPhasor call
    fig = plt.figure(figsize=(3.6, 3.4))
    axC = fig.add_subplot(111, projection="polar")
    phase_colors = {"G1": "#4C72B0", "S": "#55A868", "G2": "#C44E52", "M": "#8172B2"}
    nb = 36
    bins = np.linspace(-np.pi, np.pi, nb + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    for ph in ["G1", "S", "G2", "M"]:
        mask = phase_labels == ph
        if mask.sum() == 0:
            continue
        h, _ = np.histogram(phi_cells[mask], bins=bins)
        axC.bar(centers, h, width=(2 * np.pi / nb), bottom=0.0,
                color=phase_colors[ph], alpha=0.75, edgecolor="none",
                label=f"{ph} (n={int(mask.sum())})")
    axC.set_title("Continuous cell-cycle axis φ", loc="left", pad=12)
    axC.legend(loc="upper left", bbox_to_anchor=(-0.05, 1.08), borderaxespad=0.4,
               frameon=False, handletextpad=0.3, labelspacing=0.3)

    path = os.path.join(FIGDIR, "cellcycle_phase_rose.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


if __name__ == "__main__":
    print("=== Experiment 1: Cell-Cycle Phase Assignment (GSE293316) ===")
    run()
    print("Done. Outputs in:", OUTDIR)
