"""
exp04_manifold_geometry.py
==========================
Experiment 4: Phasor-manifold geometry validation on real measured phase
angles (BioPhasor manuscript, subsec:r_manifold).

Uses phase angles extracted from BOTH local datasets with the *unmodified*
biophasor package:
  - GSE293316 cells : per-cell continuous cell-cycle phase
    (CellCyclePhasor.assign, method="continuous").
  - GSE171432 genes : per-gene circadian phase
    (CircadianPhasor.infer_phase, single-harmonic BPT).

Validates that circular statistics are REQUIRED (not optional) on these
measured angles, via three tests:

  (a) log/exp map round-trip on the torus (PhasorManifold.log_map / exp_map):
      exp_map(base, log_map(base, point)) must recover point to machine
      precision. Reports max round-trip error.

  (b) Geodesic vs Euclidean distance near the +/-pi wrap boundary: real phase
      pairs that straddle the branch cut are close on the circle but far in
      naive Euclidean angle space. Quantifies the discrepancy on measured
      boundary-straddling pairs and via full pairwise distance matrices
      (PhasorManifold.pairwise_distance vs a flat-angle Euclidean matrix).

  (c) Frechet (circular) mean vs arithmetic mean at the branch cut: real
      genes/cells whose phases cluster near +/-pi. The arithmetic mean of the
      raw angles gives a ~180 deg error; the circular mean (phasor_mean /
      PhasorManifold.frechet_mean) is correct. Reports the error in degrees.

Generates:
  manifold_geometry.png    -- 3-panel diagnostic
  manifold_results.json    -- all numbers + method notes

Run from project root:
    python biophasor/experiments/codes/exp04_manifold_geometry.py
"""
from __future__ import annotations
import os
import sys
import json
import urllib.request

import numpy as np
import pandas as pd
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

from biophasor.dynamics.cellcycle import CellCyclePhasor
from biophasor.dynamics.circadian import CircadianPhasor
from biophasor.core.manifold import PhasorManifold
from biophasor.core.operators import phasor_mean

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
FPKM_NAME = "GSE171432_fpkm.tsv.gz"
FPKM_URL = ("ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE171nnn/GSE171432/"
            "suppl/GSE171432_genes_fpkm_table.tsv.gz")
N_CELLS = 2000
SEED = 0
ZTS = [0, 4, 8, 12, 16, 20]
BOUNDARY = np.pi - 0.30    # "near the branch cut" band: |phi| > pi - 0.30


def _fetch(url: str, dst: str) -> str:
    if not os.path.exists(dst):
        print(f"  downloading {os.path.basename(dst)} from GEO ...")
        urllib.request.urlretrieve(url, dst)
    print(f"  data: {dst} ({os.path.getsize(dst):,} bytes)")
    return dst


def _wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def _circ_err_deg(a, b):
    """Smallest circular error between two angles, in degrees."""
    return float(np.rad2deg(abs(_wrap(a - b))))


# ---------------------------------------------------------------------------
# Phase extraction from the two real datasets
# ---------------------------------------------------------------------------
def _cell_phases():
    import scanpy as sc
    import scipy.sparse as sp
    sc.settings.verbosity = 0
    h5 = _fetch(H5_URL, os.path.join(DATADIR, H5_NAME))
    ad = sc.read_10x_h5(h5)
    ad.var_names_make_unique()
    sc.pp.filter_cells(ad, min_genes=200)
    sc.pp.filter_genes(ad, min_cells=3)
    np.random.seed(SEED)
    idx = np.sort(np.random.choice(ad.n_obs, N_CELLS, replace=False))
    ad = ad[idx].copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)
    if sp.issparse(ad.X):
        ad.X = ad.X.toarray()
    _, phi = CellCyclePhasor().assign(ad, add_to_obs=False, method="continuous")
    return _wrap(np.asarray(phi, dtype=np.float64))


def _gene_phases():
    fpkm = _fetch(FPKM_URL, os.path.join(DATADIR, FPKM_NAME))
    df = pd.read_csv(fpkm, sep="\t", index_col=0)
    wt = {zt: [f"WT_ZT{zt}_{r}" for r in (0, 1, 2)] for zt in ZTS}
    avg = pd.DataFrame({zt: df[cs].mean(axis=1) for zt, cs in wt.items()}).T.loc[ZTS]
    X = np.log1p(avg.values)
    genes = avg.columns.values
    meanf = df[[c for cs in wt.values() for c in cs]].mean(axis=1)
    expr = (meanf > 1.0).values
    cp = CircadianPhasor(period=24.0, sample_interval=4.0, zt_origin=0.0)
    phi = cp.infer_phase(X[:, expr])
    return _wrap(np.asarray(phi, dtype=np.float64)), genes[expr]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run():
    phi_cells = _cell_phases()
    phi_genes, gene_names = _gene_phases()
    man = PhasorManifold(n_features=1)

    # ---- (a) log/exp round-trip on measured angles -----------------------
    rng = np.random.RandomState(SEED)
    roundtrip = {}
    for name, phi in [("GSE293316_cells", phi_cells),
                      ("GSE171432_genes", phi_genes)]:
        base = _wrap(rng.uniform(-np.pi, np.pi, size=phi.shape))
        tangent = man.log_map(base, phi)          # project to tangent at base
        recovered = man.exp_map(base, tangent)    # move back
        # circular error between recovered and original
        err = np.abs(_wrap(recovered - phi))
        roundtrip[name] = {
            "n": int(phi.size),
            "max_roundtrip_error_rad": float(err.max()),
            "mean_roundtrip_error_rad": float(err.mean()),
            "tangent_in_range": bool(np.all(np.abs(tangent) <= np.pi + 1e-9)),
        }
    max_rt = max(v["max_roundtrip_error_rad"] for v in roundtrip.values())

    # ---- (b) geodesic vs Euclidean near the wrap boundary ----------------
    # Real boundary-straddling PAIRS: one angle just below +pi, one just above
    # -pi. On the circle they are neighbours; in flat-angle space they look
    # ~2pi apart.
    def _straddle_pairs(phi, k=200):
        hi = np.sort(phi[phi > BOUNDARY])          # just below +pi
        lo = np.sort(phi[phi < -BOUNDARY])         # just above -pi
        m = min(len(hi), len(lo), k)
        hi = hi[-m:]; lo = lo[:m]                   # closest to the cut
        euclid = np.abs(hi - lo)                    # naive angle difference
        geo = np.abs(_wrap(hi - lo))                # true circular difference
        # PhasorManifold angular distance (1 - cos), monotone in geodesic
        ang = man.angular_distance(hi, lo)
        return hi, lo, euclid, geo, ang

    boundary = {}
    for name, phi in [("GSE293316_cells", phi_cells),
                      ("GSE171432_genes", phi_genes)]:
        hi, lo, euclid, geo, ang = _straddle_pairs(phi)
        boundary[name] = {
            "n_pairs": int(len(hi)),
            "euclidean_dist_mean_rad": round(float(euclid.mean()), 4),
            "geodesic_dist_mean_rad": round(float(geo.mean()), 4),
            "euclidean_over_geodesic_ratio": round(
                float(euclid.mean() / max(geo.mean(), 1e-9)), 2),
            "max_euclidean_error_rad": round(float((euclid - geo).max()), 4),
            "max_euclidean_error_deg": round(float(np.rad2deg((euclid - geo).max())), 1),
        }

    # Full pairwise-matrix view on a boundary-enriched cell subset: compare a
    # naive Euclidean (flat-angle) distance matrix vs the manifold geodesic
    # matrix, and quantify how many pairs the Euclidean matrix mis-orders.
    near = phi_cells[np.abs(phi_cells) > BOUNDARY]
    rng2 = np.random.RandomState(1)
    sub = near[rng2.choice(len(near), min(120, len(near)), replace=False)]
    Dgeo = man.pairwise_distance(sub[:, None])                       # (n,n) circular
    Deuc = np.abs(sub[:, None] - sub[None, :])                       # flat-angle
    iu = np.triu_indices(len(sub), k=1)
    geo_pairs = Dgeo[iu]; euc_pairs = Deuc[iu]
    # pairs that are near on the manifold (small geodesic) but far in Euclidean
    near_on_circle = geo_pairs < 0.1     # (1 - cos) < 0.1  ->  < ~26 deg
    far_in_euclid = euc_pairs > np.pi    # naive distance > half turn
    n_misordered = int((near_on_circle & far_in_euclid).sum())
    # Pick the single worst real pair in this run (largest Euclidean/geodesic
    # gap) so the note reports measured numbers, not illustrative ones.
    geo_true = np.abs(_wrap(sub[:, None] - sub[None, :]))[iu]  # radians on circle
    gap = euc_pairs - geo_true
    w = int(np.argmax(gap))
    ii, jj = iu[0][w], iu[1][w]
    matrix_view = {
        "n_cells_boundary_subset": int(len(sub)),
        "n_pairs": int(len(geo_pairs)),
        "n_pairs_near_on_circle_but_far_in_euclid": n_misordered,
        "frac_pairs_misordered": round(n_misordered / len(geo_pairs), 4),
        "worst_pair_angles_rad": [round(float(sub[ii]), 4), round(float(sub[jj]), 4)],
        "worst_pair_euclidean_rad": round(float(euc_pairs[w]), 4),
        "worst_pair_geodesic_rad": round(float(geo_true[w]), 4),
        "note": (
            f"Euclidean flat-angle distance treats measured angles "
            f"{sub[ii]:+.2f} and {sub[jj]:+.2f} rad as {euc_pairs[w]:.2f} rad "
            f"apart; the manifold geodesic correctly reports them as "
            f"{geo_true[w]:.3f} rad apart."
        ),
    }

    # ---- (c) Frechet vs arithmetic mean at the branch cut ----------------
    def _mean_test(phi, label, extra=None):
        near = phi[np.abs(phi) > BOUNDARY]         # cluster near +/-pi
        # circular spread to confirm they are genuinely one cluster on the circle
        Rbar = float(np.abs(np.exp(1j * near).mean()))
        arithmetic = float(near.mean())            # WRONG estimator
        circular = float(phasor_mean(near))        # biophasor circular mean
        frechet = float(man.frechet_mean(near[:, None])[0])  # manifold Frechet
        # true centre on the circle is +/-pi (the cluster direction)
        target = np.pi if abs(circular) > np.pi / 2 else 0.0
        out = {
            "label": label,
            "n_near_boundary": int(near.size),
            "resultant_length_R": round(Rbar, 4),
            "arithmetic_mean_rad": round(arithmetic, 4),
            "circular_mean_rad": round(circular, 4),
            "frechet_mean_rad": round(frechet, 4),
            "circular_vs_frechet_agree_deg": round(_circ_err_deg(circular, frechet), 6),
            "arithmetic_error_vs_circular_deg": round(_circ_err_deg(arithmetic, circular), 2),
        }
        if extra is not None:
            out.update(extra)
        return out, near

    mean_cells, near_cells = _mean_test(phi_cells, "GSE293316 cells")
    mean_genes, near_genes = _mean_test(phi_genes, "GSE171432 genes")

    # verdict
    a_ok = max_rt < 1e-9
    b_ok = all(v["euclidean_over_geodesic_ratio"] > 10 for v in boundary.values())
    c_ok = (mean_cells["arithmetic_error_vs_circular_deg"] > 90 and
            mean_genes["arithmetic_error_vs_circular_deg"] > 90 and
            mean_cells["circular_vs_frechet_agree_deg"] < 1e-6 and
            mean_genes["circular_vs_frechet_agree_deg"] < 1e-6)
    verdict = "reproduces" if (a_ok and b_ok and c_ok) else "partial"

    result = {
        "datasets": {
            "cells": "GSE293316 (REH B-ALL scRNA-seq): per-cell continuous "
                     "cell-cycle phase",
            "genes": "GSE171432 (WT mouse liver): per-gene circadian BPT phase",
        },
        "n_cell_phases": int(phi_cells.size),
        "n_gene_phases": int(phi_genes.size),
        "boundary_band_rad": round(float(BOUNDARY), 4),
        "a_logexp_roundtrip": {
            "per_dataset": roundtrip,
            "max_roundtrip_error_rad": max_rt,
            "holds": bool(a_ok),
        },
        "b_geodesic_vs_euclidean": {
            "straddling_pairs": boundary,
            "pairwise_matrix": matrix_view,
            "holds": bool(b_ok),
        },
        "c_frechet_vs_arithmetic": {
            "cells": mean_cells,
            "genes": mean_genes,
            "holds": bool(c_ok),
        },
        "verdict": verdict,
        "verdict_note": (
            f"(a) log/exp round-trip is exact on {int(phi_cells.size+phi_genes.size)} "
            f"measured angles (max error {max_rt:.1e} rad, machine precision). "
            f"(b) At the +/-pi branch cut, naive Euclidean distance overstates "
            f"true circular distance by "
            f"{boundary['GSE293316_cells']['euclidean_over_geodesic_ratio']:.0f}x "
            f"(cells) / "
            f"{boundary['GSE171432_genes']['euclidean_over_geodesic_ratio']:.0f}x "
            f"(genes); {matrix_view['n_pairs_near_on_circle_but_far_in_euclid']} "
            f"boundary cell-pairs are near on the circle but ~2pi apart in "
            f"Euclidean space. (c) For real angles clustered at +/-pi, the "
            f"arithmetic mean errs by "
            f"{mean_cells['arithmetic_error_vs_circular_deg']:.0f} deg (cells) / "
            f"{mean_genes['arithmetic_error_vs_circular_deg']:.0f} deg (genes) "
            f"vs the circular mean; biophasor phasor_mean and the manifold "
            f"Frechet mean agree to machine precision. Circular statistics are "
            f"required, not optional, on measured phase angles."
        ),
    }

    _plot(phi_cells, phi_genes, boundary, near_cells, mean_cells,
          near_genes, mean_genes)
    json.dump(result, open(os.path.join(OUTDIR, "manifold_results.json"), "w"), indent=1)
    print("  ->", json.dumps({"verdict": verdict, "max_roundtrip": max_rt,
          "cells_arith_err_deg": mean_cells["arithmetic_error_vs_circular_deg"],
          "genes_arith_err_deg": mean_genes["arithmetic_error_vs_circular_deg"]}))
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


def _plot(phi_cells, phi_genes, boundary, near_cells, mean_cells,
          near_genes, mean_genes):
    """Emit three single-panel PNGs (one panel per file); combining into a
    multi-panel figure happens later in LaTeX. Data and colors are unchanged
    from the previous combined figure."""
    _style()
    c_cell = "#4C72B0"; c_gene = "#C44E52"; c_ref = "#333333"

    # ---- PNG 1: Euclidean vs geodesic distance near the cut (stands alone) --
    fig = plt.figure(figsize=(3.6, 3.0))
    ax = fig.add_subplot(1, 1, 1)
    labels, euc, geo = [], [], []
    for name, disp in [("GSE293316_cells", "cells"), ("GSE171432_genes", "genes")]:
        labels.append(disp)
        euc.append(boundary[name]["euclidean_dist_mean_rad"])
        geo.append(boundary[name]["geodesic_dist_mean_rad"])
    x = np.arange(len(labels)); w = 0.36
    ax.bar(x - w / 2, euc, w, color="#BBBBBB", label="Euclidean (flat angle)")
    ax.bar(x + w / 2, geo, w, color=c_ref, label="geodesic (circular)")
    ax.axhline(np.pi, color="k", ls=":", lw=0.7)
    ax.text(1.35, np.pi + 0.06, "π", fontsize=7.5)
    for xi, (e, g) in enumerate(zip(euc, geo)):
        ax.text(xi - w / 2, e + 0.06, f"{e:.2f}", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, g + 0.06, f"{g:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 7.2)
    ax.set_ylabel("mean distance across the cut (rad)")
    ax.set_title("Euclidean vs geodesic near ±π", loc="left")
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.4)
    path = os.path.join(FIGDIR, "manifold_euclid_vs_geodesic.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")

    # ---- PNG 2: branch-cut cluster, cells (polar) -------------------------
    fig = plt.figure(figsize=(3.1, 3.3))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_theta_zero_location("E")
    theta = near_cells
    r = np.ones_like(theta) + np.linspace(0, 0.25, len(theta))
    ax.scatter(theta, r, s=5, color=c_cell, alpha=0.4, edgecolors="none",
               label=f"cells near ±π (n={len(theta)})")
    am = mean_cells["arithmetic_mean_rad"]; cm = mean_cells["circular_mean_rad"]
    ax.plot([am, am], [0, 1.5], color="#888888", lw=2.0, ls="--",
            label="arithmetic mean (wrong)")
    ax.plot([cm, cm], [0, 1.5], color=c_ref, lw=2.2,
            label="circular / Fréchet mean")
    ax.set_rticks([]); ax.set_rlim(0, 1.5)
    ax.set_title(f"Branch-cut cluster (cells): "
                 f"arith. errs {mean_cells['arithmetic_error_vs_circular_deg']:.0f}°",
                 loc="left", pad=16)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.02),
              borderaxespad=0.4)
    path = os.path.join(FIGDIR, "manifold_branchcut_cells.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")

    # ---- PNG 3: branch-cut cluster, genes (polar) -------------------------
    fig = plt.figure(figsize=(3.1, 3.3))
    ax = fig.add_subplot(1, 1, 1, projection="polar")
    ax.set_theta_zero_location("E")
    theta = near_genes
    r = np.ones_like(theta) + np.linspace(0, 0.25, len(theta))
    ax.scatter(theta, r, s=6, color=c_gene, alpha=0.5, edgecolors="none",
               label=f"genes near ±π (n={len(theta)})")
    am = mean_genes["arithmetic_mean_rad"]; cm = mean_genes["circular_mean_rad"]
    ax.plot([am, am], [0, 1.5], color="#888888", lw=2.0, ls="--",
            label="arithmetic mean (wrong)")
    ax.plot([cm, cm], [0, 1.5], color=c_ref, lw=2.2,
            label="circular / Fréchet mean")
    ax.set_rticks([]); ax.set_rlim(0, 1.5)
    ax.set_title(f"Branch-cut cluster (genes): "
                 f"arith. errs {mean_genes['arithmetic_error_vs_circular_deg']:.0f}°",
                 loc="left", pad=16)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.02),
              borderaxespad=0.4)
    path = os.path.join(FIGDIR, "manifold_branchcut_genes.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


if __name__ == "__main__":
    print("=== Experiment 4: Manifold geometry validation ===")
    run()
    print("Done. Outputs in:", OUTDIR)
