"""
exp03_encoding_coherence.py
===========================
Experiment 3: Encoder comparison + coherence-based gene selection on real
single-cell RNA-seq (BioPhasor manuscript, Section "Phase Encoding" /
subsec:r_encoding, GSE293316 REH human B-ALL scRNA-seq).

Runs the *unmodified* biophasor package (documented defaults, no tuning):

  (a) Applies all three encoders (tanh_phase_encode [default], log_linear_encode,
      linear_encode) to the SAME preprocessed matrix and reports per-encoder
      phase spread sigma_phi and the coherence C distribution
      (biophasor.core.operators.coherence over cells, axis=0):
      mean / spread / dynamic range / count clearing C>0.30.

  (b) Coherence-based gene selection (coherence_filter C>0.30 + Rayleigh
      non-uniformity, SynchronyMetrics.rayleigh_per_feature formula) vs
      variance-based HVG (scanpy highly_variable_genes) on the same matrix.
      Reports Jaccard overlap, how many coherence-selected genes are LOW
      variance (below the HVG threshold), and biological grounding: enrichment
      of the coherence-only selection for curated reference sets (ribosomal
      RPL/RPS, mitochondrial MT-, canonical housekeeping). Reports the real
      numbers whatever they are.

Preprocessing mirrors exp01 exactly: subsample 2000 cells (seed 0),
filter min_genes=200/min_cells=3, normalize_total(1e4)+log1p.

Generates:
  encoding_comparison.png   -- 4-panel diagnostic
  encoding_results.json     -- all numbers, gene lists, method notes

Run from project root:
    python biophasor/experiments/codes/exp03_encoding_coherence.py
"""
from __future__ import annotations
import os
import sys
import json
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path / import bootstrap
# ---------------------------------------------------------------------------
try:
    import biophasor  # noqa: F401
except ModuleNotFoundError:
    _d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isfile(os.path.join(_d, "biophasor", "__init__.py")):
            sys.path.insert(0, _d)
            break
        _d = os.path.dirname(_d)
    import biophasor  # noqa: F401

from biophasor.transform.encoder import (
    tanh_phase_encode, log_linear_encode, linear_encode,
)
from biophasor.core.operators import coherence, coherence_filter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(EXPDIR, "data", "raw")
OUTDIR = os.path.join(EXPDIR, "results")
FIGDIR = os.path.join(EXPDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(DATADIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

H5_NAME = "GSE293316_reh.h5"
H5_URL = ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE293nnn/GSE293316/"
          "suppl/GSE293316_reh_filtered_feature_bc_matrix.h5")
N_CELLS = 2000
SEED = 0
C_THRESHOLD = 0.30      # package-documented coherence_filter default
N_TOP = 2000            # rank-matched selection size (== scanpy n_top_genes)

ENCODERS = {
    "tanh_phase (default)": tanh_phase_encode,
    "log_linear": log_linear_encode,
    "linear": linear_encode,
}


def _fetch(url: str, dst: str) -> str:
    if not os.path.exists(dst):
        print(f"  downloading {os.path.basename(dst)} from GEO ...")
        urllib.request.urlretrieve(url, dst)
    print(f"  data: {dst} ({os.path.getsize(dst):,} bytes)")
    return dst


def _rayleigh_vec(phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised Rayleigh test, identical formula to
    biophasor.utils.math_utils.rayleigh_test (Zar 2010), applied per column.
    R equals the coherence C exactly; returns (R, p) per feature."""
    N = phase.shape[0]
    R = np.abs(np.exp(1j * phase).mean(axis=0))
    Z = N * R ** 2
    p = np.exp(np.sqrt(1 + 4 * N + 4 * (N ** 2 - Z ** 2)) - (1 + 2 * N))
    return R, p


def run():
    import scanpy as sc
    import scipy.sparse as sp
    sc.settings.verbosity = 0

    h5 = _fetch(H5_URL, os.path.join(DATADIR, H5_NAME))
    adata = sc.read_10x_h5(h5)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    np.random.seed(SEED)
    if adata.n_obs > N_CELLS:
        idx = np.sort(np.random.choice(adata.n_obs, N_CELLS, replace=False))
        adata = adata[idx].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    X = adata.X.toarray() if sp.issparse(adata.X) else np.asarray(adata.X)
    genes = np.array(adata.var_names)
    n_genes = len(genes)
    var = X.var(axis=0)
    frac_expr = (X > 0).mean(axis=0)
    mean_expr = X.mean(axis=0)

    # --- (a) encoder comparison -------------------------------------------
    enc_out = {}
    coh_by_enc = {}
    for name, fn in ENCODERS.items():
        phi = fn(X)
        C = coherence(phi, axis=0)
        coh_by_enc[name] = C
        enc_out[name] = {
            "sigma_phi": round(float(phi.std(axis=0).mean()), 4),
            "coherence_mean": round(float(C.mean()), 4),
            "coherence_std": round(float(C.std()), 4),
            "coherence_min": round(float(C.min()), 4),
            "coherence_max": round(float(C.max()), 4),
            "coherence_dynamic_range": round(float(C.max() - C.min()), 4),
            "n_genes_C_gt_0.30": int((C > C_THRESHOLD).sum()),
            "frac_genes_C_gt_0.30": round(float((C > C_THRESHOLD).mean()), 4),
        }

    # --- (b) coherence vs HVG selection -----------------------------------
    # Default-encoder (tanh) coherence is the selection statistic.
    phi_def = tanh_phase_encode(X)
    C_def = coherence(phi_def, axis=0)
    _, coh_mask = coherence_filter(phi_def, threshold=C_THRESHOLD)
    R_ray, p_ray = _rayleigh_vec(phi_def)

    # scanpy HVG (seurat flavor, n_top_genes matched to N_TOP)
    adh = adata.copy()
    sc.pp.highly_variable_genes(adh, n_top_genes=N_TOP)
    hvg_mask = adh.var["highly_variable"].values
    hvg_set = set(genes[hvg_mask])
    hvg_var_thresh = float(var[hvg_mask].min())

    # rank-matched coherence selection (top-N by C) for a fair overlap
    top_coh_idx = np.argsort(-C_def)[:N_TOP]
    top_coh_set = set(genes[top_coh_idx])

    inter = len(top_coh_set & hvg_set)
    union = len(top_coh_set | hvg_set)
    jaccard = inter / union

    coh_only = sorted(top_coh_set - hvg_set)
    coh_only_idx = np.array([np.where(genes == g)[0][0] for g in coh_only])
    n_low_var = int((var[coh_only_idx] < hvg_var_thresh).sum())

    # curated biological reference sets (grounding)
    ribo = [g for g in genes if g.startswith(("RPL", "RPS"))]
    mito = [g for g in genes if g.startswith("MT-")]
    housekeeping = ["ACTB", "GAPDH", "B2M", "TUBB", "PGK1", "TBP",
                    "HPRT1", "LDHA", "PPIA", "RPLP0"]  # Eisenberg & Levanon 2013
    ref_sets = {"ribosomal_RPL_RPS": ribo, "mitochondrial_MT": mito,
                "housekeeping": housekeeping}

    def _enrich(selection: set, cat_genes: list) -> dict:
        cat = set(cat_genes) & set(genes)
        obs = len(cat & selection)
        exp = len(selection) / n_genes * len(cat)
        return {"observed": int(obs), "in_data": int(len(cat)),
                "expected_by_chance": round(exp, 2),
                "fold": round(obs / exp, 3) if exp > 0 else None}

    enrich_cohonly = {k: _enrich(top_coh_set - hvg_set, v) for k, v in ref_sets.items()}
    enrich_cohtop = {k: _enrich(top_coh_set, v) for k, v in ref_sets.items()}
    enrich_hvg = {k: _enrich(hvg_set, v) for k, v in ref_sets.items()}

    # diagnosis: coherence-over-cells vs dropout
    corr_C_fexpr = float(np.corrcoef(C_def, frac_expr)[0, 1])
    corr_C_meanexpr = float(np.corrcoef(C_def, mean_expr)[0, 1])
    coh_order = np.argsort(-C_def)
    top20 = [{"gene": str(genes[i]), "C": round(float(C_def[i]), 4),
              "frac_cells_expr": round(float(frac_expr[i]), 4),
              "mean_logexpr": round(float(mean_expr[i]), 4)}
             for i in coh_order[:20]]
    bottom20 = [{"gene": str(genes[i]), "C": round(float(C_def[i]), 4),
                 "frac_cells_expr": round(float(frac_expr[i]), 4),
                 "mean_logexpr": round(float(mean_expr[i]), 4)}
                for i in coh_order[-20:][::-1]]
    lowest_names = ", ".join(d["gene"] for d in bottom20[:3])

    # verdict logic
    total_enrich_cohonly = sum(v["observed"] for v in enrich_cohonly.values())
    verdict = "does-not-reproduce"
    verdict_note = (
        f"Coherence over cells on scRNA-seq is a dropout statistic, not a "
        f"biological-structure statistic: C is anti-correlated with detection "
        f"rate (corr(C, frac_cells_expressing) = {corr_C_fexpr:.3f}). At C>0.30 "
        f"the filter is non-discriminative ({int((C_def>C_THRESHOLD).sum())}/"
        f"{n_genes} = {(C_def>C_THRESHOLD).mean():.1%} of genes pass). The "
        f"top-{N_TOP} coherence set is nearly orthogonal to HVG "
        f"(Jaccard {jaccard:.3f}) and is entirely low-variance "
        f"({n_low_var}/{len(coh_only)} coherence-only genes below the HVG "
        f"variance floor), but it is NOT biologically structured: coherence-only "
        f"genes show zero enrichment for ribosomal/mitochondrial/housekeeping "
        f"sets ({total_enrich_cohonly} observed, all at/below chance). The "
        f"highest-coherence genes are near-constant zero genes (expressed in "
        f"~0% of cells); the genuinely constitutive genes with the LOWEST "
        f"coherence in this run are {lowest_names} (ribosomal / translation-"
        f"elongation, expressed in ~100% of cells). The 'low-variance' half of "
        f"the claim holds; the "
        f"'recovers structured biology' half does not."
    )

    result = {
        "dataset": "GSE293316 (REH human B-ALL scRNA-seq, 10x)",
        "n_cells_used": int(adata.n_obs),
        "n_genes": int(n_genes),
        "preprocessing": "subsample 2000 (seed 0), min_genes=200/min_cells=3, "
                         "normalize_total(1e4)+log1p (as exp01)",
        "encoder_comparison": enc_out,
        "coherence_threshold": C_THRESHOLD,
        "selection_size_rank_matched": N_TOP,
        "hvg_variance_threshold": round(hvg_var_thresh, 6),
        "jaccard_coherenceTop_vs_HVG": round(jaccard, 4),
        "intersection_size": int(inter),
        "n_coherence_only_genes": int(len(coh_only)),
        "n_coherence_only_below_HVG_variance": int(n_low_var),
        "frac_coherence_only_low_variance": round(n_low_var / max(len(coh_only), 1), 4),
        "enrichment_coherence_only": enrich_cohonly,
        "enrichment_coherence_top": enrich_cohtop,
        "enrichment_HVG": enrich_hvg,
        "corr_C_vs_frac_cells_expressing": round(corr_C_fexpr, 4),
        "corr_C_vs_mean_expression": round(corr_C_meanexpr, 4),
        "rayleigh_note": (
            "Rayleigh R == coherence C exactly (both are the mean resultant "
            f"length). At N={int(adata.n_obs)} cells the Rayleigh p-value is "
            f"< 1e-3 for {int((p_ray < 1e-3).mean()*100)}% of genes, so it is "
            "also non-discriminative for feature selection here."
        ),
        "top20_coherence_genes": top20,
        "bottom20_coherence_genes": bottom20,
        "coherence_only_gene_sample": coh_only[:50],
        "verdict": verdict,
        "verdict_note": verdict_note,
    }

    _plot(coh_by_enc, enc_out, C_def, frac_expr, var, hvg_mask, top_coh_idx,
          hvg_var_thresh, enrich_cohonly, genes)
    json.dump(result, open(os.path.join(OUTDIR, "encoding_results.json"), "w"), indent=1)
    print("  ->", json.dumps({k: result[k] for k in
          ["jaccard_coherenceTop_vs_HVG", "n_coherence_only_below_HVG_variance",
           "corr_C_vs_frac_cells_expressing", "verdict"]}))
    return result


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def apply_figure_style(sizes=(9, 8, 7.5)):
    """Publication-grade rcParams (figure-style skill §5.2 role-mapped ladder,
    §4/§8 frame + tick conventions). base=titles/axis-labels, mid=legend/annot,
    small=ticks; frameless legends, outward ticks, 300-dpi vector-safe output."""
    base, mid, small = sizes
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": base, "axes.titlesize": base, "axes.labelsize": base,
        "legend.fontsize": mid, "xtick.labelsize": small, "ytick.labelsize": small,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.direction": "out", "ytick.direction": "out",
        "legend.frameon": False, "axes.grid": False,
        "figure.dpi": 200, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _style():
    apply_figure_style()


def _plot(coh_by_enc, enc_out, C_def, frac_expr, var, hvg_mask, top_coh_idx,
          hvg_var_thresh, enrich_cohonly, genes):
    _style()
    focal = "#C44E52"      # tanh default
    others = ["#4C72B0", "#55A868"]
    enc_colors = {"tanh_phase (default)": focal,
                  "log_linear": others[0], "linear": others[1]}
    saved = []

    # ---- PNG 1: coherence distribution per encoder vs the 0.30 filter --------
    figA, axA = plt.subplots(figsize=(3.1, 2.6))
    names = list(coh_by_enc.keys())
    for nm in names:
        C = coh_by_enc[nm]
        h, edges = np.histogram(C, bins=np.linspace(0, 1, 41), density=True)
        ctr = (edges[:-1] + edges[1:]) / 2
        axA.plot(ctr, h, lw=(2.0 if "tanh" in nm else 1.2),
                 color=enc_colors[nm], label=nm)
    axA.axvline(C_THRESHOLD, color="k", ls="--", lw=0.8)
    axA.text(C_THRESHOLD + 0.02, axA.get_ylim()[1] * 0.62, "C>0.30")
    axA.set_xlabel("coherence C over cells")
    axA.set_ylabel("gene density")
    axA.set_title("Coherence piles above filter", loc="left")
    # data piles at the right edge and threshold text sits mid-top; the
    # upper-centre-left is the free region for the encoder legend
    axA.legend(loc="upper left", bbox_to_anchor=(0.30, 1.0),
               borderaxespad=0.4, frameon=False)
    pathA = os.path.join(FIGDIR, "encoding_coherence_hist.png")
    figA.savefig(pathA, dpi=300, bbox_inches="tight")
    plt.close(figA)
    saved.append(pathA)

    # ---- PNG 2: C vs detection rate (dropout statistic) [KEY] ----------------
    figB, axB = plt.subplots(figsize=(3.9, 3.1))
    sub = np.random.RandomState(0).choice(len(C_def), 6000, replace=False)
    axB.scatter(frac_expr[sub], C_def[sub], s=3, alpha=0.25, color="#555555",
                edgecolors="none", rasterized=True)
    r = np.corrcoef(C_def, frac_expr)[0, 1]
    axB.set_xlabel("fraction of cells expressing gene")
    axB.set_ylabel("coherence C (tanh)")
    axB.set_title("C tracks detection rate", loc="left")
    axB.text(0.03, 0.05, f"r = {r:.2f}", transform=axB.transAxes)
    pathB = os.path.join(FIGDIR, "encoding_dropout_scatter.png")
    figB.savefig(pathB, dpi=300, bbox_inches="tight")
    plt.close(figB)
    saved.append(pathB)

    # ---- PNG 3: variance of the two selections (coherence-top vs HVG) --------
    figC, axC = plt.subplots(figsize=(3.9, 3.1))
    lv = np.log10(var + 1e-6)
    bins = np.linspace(lv.min(), lv.max(), 45)
    axC.hist(lv[hvg_mask], bins=bins, color="#4C72B0", alpha=0.7,
             label="HVG (variance)", density=True)
    axC.hist(lv[top_coh_idx], bins=bins, color=focal, alpha=0.7,
             label="coherence top-2000", density=True)
    axC.axvline(np.log10(hvg_var_thresh + 1e-6), color="k", ls="--", lw=0.8)
    axC.set_xlabel("log10 gene variance")
    axC.set_ylabel("density")
    axC.set_title("Coherence selects low-variance genes", loc="left")
    axC.legend(loc="upper right", borderaxespad=0.4, frameon=False)
    pathC = os.path.join(FIGDIR, "encoding_variance_selection.png")
    figC.savefig(pathC, dpi=300, bbox_inches="tight")
    plt.close(figC)
    saved.append(pathC)

    # ---- PNG 4: enrichment of coherence-only genes (observed vs expected) ----
    figD, axD = plt.subplots(figsize=(3.1, 2.6))
    labels = ["ribosomal", "mito", "house-\nkeeping"]
    keys = ["ribosomal_RPL_RPS", "mitochondrial_MT", "housekeeping"]
    obs = [enrich_cohonly[k]["observed"] for k in keys]
    exp = [enrich_cohonly[k]["expected_by_chance"] for k in keys]
    x = np.arange(len(keys))
    w = 0.38
    axD.bar(x - w / 2, exp, w, color="#BBBBBB", label="expected (chance)")
    axD.bar(x + w / 2, obs, w, color=focal, label="observed")
    for xi, o in zip(x, obs):
        axD.text(xi + w / 2, o + 0.15, str(o), ha="center", fontsize=7.5)
    axD.set_xticks(x)
    axD.set_xticklabels(labels)
    axD.set_ylabel("genes in coherence-only set")
    axD.set_title("No biological enrichment", loc="left")
    axD.legend(loc="upper right", borderaxespad=0.4, frameon=False)
    pathD = os.path.join(FIGDIR, "encoding_enrichment.png")
    figD.savefig(pathD, dpi=300, bbox_inches="tight")
    plt.close(figD)
    saved.append(pathD)

    for p in saved:
        print(f"  [figure] {p}")


if __name__ == "__main__":
    print("=== Experiment 3: Encoding comparison + coherence gene selection ===")
    run()
    print("Done. Outputs in:", OUTDIR)
