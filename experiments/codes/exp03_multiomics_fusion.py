"""
exp03_multiomics_fusion.py
==========================
Experiment 3: Multi-omics coherence fusion on a MATCHED CPTAC UCEC cohort.

Turns the "Multi-Omics Integration" planned scenario (Manuscript
\S subsec:r_multiomics) into a measured result on real, co-assayed data:
matched RNA-seq + mass-spec proteomics on the SAME 109 CPTAC endometrial
carcinoma samples (95 tumour / 14 normal), 9200 shared gene symbols.

Runs the *unmodified* biophasor phasor encoder + MultiOmicsIntegrator and tests
the two paper claims:
  (a) the cross-layer coherence matrix recovers RNA<->protein coupling
      (mRNA-to-protein flow) above a sample-shuffled null;
  (b) coherence-weighted fusion yields a fused layer whose mean coherence
      exceeds each single layer's mean coherence.

Missingness handling: the umich proteomics layer has real NaN (4.9% overall,
MNAR/MCAR). We use COMPLETE-CASE genes (columns with zero protein NaN across
all 109 samples; 7083/9200) so that no imputation can inflate coherence. A
mean-imputed sensitivity pass over all 9200 genes is also reported.

Encoding: RNA is washu transcriptomics on a LINEAR count-like scale
(0..14839) -> tanh_phase_encode(log_transform=True). Protein is umich
normalized log-ratio, already log-scale and signed (-6.9..5.7) ->
tanh_phase_encode(log_transform=False) (log1p on negatives is invalid).

Generates:
  multiomics_fusion.png    -- coherence matrix, cross-coh vs null, per-layer vs fused
  multiomics_results.json  -- all coherence numbers, null, method notes

Run from project root:
    PYTHONPATH=.. python experiments/codes/exp03_multiomics_fusion.py
"""
from __future__ import annotations
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _figstyle import apply_style as _apply_style

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

from biophasor.transform.encoder import tanh_phase_encode
from biophasor.integration.multiomics import MultiOmicsIntegrator
from biophasor.core.operators import coherence

EXPDIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATADIR = os.path.join(EXPDIR, "data", "raw", "cptac_ucec")
OUTDIR = os.path.join(EXPDIR, "results")
os.makedirs(OUTDIR, exist_ok=True)
FIGDIR = os.path.join(EXPDIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)

SEED = 0
N_NULL = 200   # sample-shuffle null repeats


def _load():
    rna = pd.read_pickle(os.path.join(DATADIR, "rna.pkl.gz"))
    prot = pd.read_pickle(os.path.join(DATADIR, "protein.pkl.gz"))
    lab = pd.read_pickle(os.path.join(DATADIR, "labels.pkl.gz"))
    assert (rna.index == prot.index).all() and (rna.columns == prot.columns).all()
    assert (rna.index == lab.index).all()
    return rna, prot, lab


def _encode(R, P):
    """RNA linear-scale -> log1p+tanh; protein log-ratio -> tanh, no log1p."""
    phi_rna = tanh_phase_encode(R, log_transform=True)
    phi_prot = tanh_phase_encode(P, log_transform=False)
    return phi_rna, phi_prot


def _coherence_suite(phi_rna, phi_prot, seed=SEED, n_null=N_NULL):
    integ = MultiOmicsIntegrator(["RNA", "protein"])
    phase_dict = {"RNA": phi_rna, "protein": phi_prot}

    cc = integ.cross_coherence(phase_dict)[("RNA", "protein")]
    Cm = integ.coherence_matrix(phase_dict)   # 2x2: diag=per-layer, offdiag=cross

    # Sample-shuffle null: destroy sample pairing between layers
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_null):
        perm = rng.permutation(phi_prot.shape[0])
        ccn = integ.cross_coherence({"RNA": phi_rna, "protein": phi_prot[perm]})
        nulls.append(ccn[("RNA", "protein")])
    nulls = np.array(nulls)
    z = (cc - nulls.mean()) / (nulls.std() + 1e-12)
    p_emp = float((nulls >= cc).mean())

    fused = integ.fuse(phase_dict, method="circular_mean")
    c_rna = float(coherence(phi_rna, axis=0).mean())
    c_prot = float(coherence(phi_prot, axis=0).mean())
    c_fused = float(coherence(fused, axis=0).mean())

    return dict(
        cross_coherence=float(cc),
        coherence_matrix=Cm.tolist(),
        null_mean=float(nulls.mean()), null_std=float(nulls.std()),
        cross_z=float(z), cross_p_emp=p_emp,
        mean_coh_rna=c_rna, mean_coh_protein=c_prot, mean_coh_fused=c_fused,
        fused_exceeds_max_single=bool(c_fused > max(c_rna, c_prot)),
        fused_vs_max_single_delta=float(c_fused - max(c_rna, c_prot)),
        fused=fused,
    )


def run():
    rna, prot, lab = _load()
    genes = rna.columns.values

    # --- primary: complete-case genes (zero protein NaN) ---
    complete = ~np.isnan(prot.values).any(axis=0)
    Rc = rna.values[:, complete]
    Pc = prot.values[:, complete]
    phi_rna_c, phi_prot_c = _encode(Rc, Pc)
    primary = _coherence_suite(phi_rna_c, phi_prot_c)

    # --- sensitivity: all 9200 genes, protein NaN mean-imputed per gene ---
    P_imp = prot.values.copy()
    col_means = np.nanmean(P_imp, axis=0)
    inds = np.where(np.isnan(P_imp))
    P_imp[inds] = np.take(col_means, inds[1])
    phi_rna_a, phi_prot_a = _encode(rna.values, P_imp)
    sens = _coherence_suite(phi_rna_a, phi_prot_a)

    fused = primary.pop("fused"); sens.pop("fused")

    verdict = (
        "partial: cross-layer RNA-protein coherence recovers real coupling "
        f"({primary['cross_coherence']:.3f} vs null {primary['null_mean']:.3f}"
        f"+-{primary['null_std']:.3f}, z={primary['cross_z']:.1f}, "
        f"p_emp={primary['cross_p_emp']:.3f}) -- the mRNA->protein flow claim "
        f"holds; BUT coherence-weighted fusion does NOT exceed single layers "
        f"(fused {primary['mean_coh_fused']:.3f} < RNA {primary['mean_coh_rna']:.3f}, "
        f"protein {primary['mean_coh_protein']:.3f}; delta="
        f"{primary['fused_vs_max_single_delta']:+.3f}). The paper's core "
        "integration claim (fused>single) is refuted on this cohort."
    )

    result = {
        "dataset": "CPTAC UCEC matched RNA-seq (washu) + proteomics (umich), same 109 samples",
        "n_samples": int(rna.shape[0]),
        "n_genes_shared": int(rna.shape[1]),
        "class_counts": lab["sample_type"].value_counts().to_dict(),
        "encoding": {
            "RNA": "tanh_phase_encode(log_transform=True)  [linear count-scale]",
            "protein": "tanh_phase_encode(log_transform=False)  [already log-ratio, signed]",
        },
        "missingness_handling": {
            "protein_nan_fraction_overall": float(np.isnan(prot.values).mean()),
            "primary": "complete-case genes (zero protein NaN across all samples)",
            "n_complete_case_genes": int(complete.sum()),
            "sensitivity": "all genes, protein NaN mean-imputed per gene",
        },
        "coherence_definition": "coherence(phi, axis=0): |mean_samples e^{i phi}| per gene, averaged over genes",
        "claim_a_cross_layer_coupling": {
            "cross_coherence_RNA_protein": round(primary["cross_coherence"], 4),
            "sample_shuffle_null_mean": round(primary["null_mean"], 4),
            "sample_shuffle_null_std": round(primary["null_std"], 4),
            "z_score": round(primary["cross_z"], 2),
            "empirical_p": primary["cross_p_emp"],
            "n_null_repeats": N_NULL,
            "coherence_matrix_RNA_protein": [[round(v, 4) for v in row]
                                             for row in primary["coherence_matrix"]],
            "reproduces": True,
        },
        "claim_b_fusion_exceeds_single": {
            "mean_coherence_RNA": round(primary["mean_coh_rna"], 4),
            "mean_coherence_protein": round(primary["mean_coh_protein"], 4),
            "mean_coherence_fused": round(primary["mean_coh_fused"], 4),
            "fused_exceeds_max_single": primary["fused_exceeds_max_single"],
            "delta_fused_minus_max_single": round(primary["fused_vs_max_single_delta"], 4),
            "reproduces": primary["fused_exceeds_max_single"],
        },
        "sensitivity_imputed_all_genes": {
            "n_genes": int(rna.shape[1]),
            "cross_coherence_RNA_protein": round(sens["cross_coherence"], 4),
            "null_mean": round(sens["null_mean"], 4),
            "mean_coherence_RNA": round(sens["mean_coh_rna"], 4),
            "mean_coherence_protein": round(sens["mean_coh_protein"], 4),
            "mean_coherence_fused": round(sens["mean_coh_fused"], 4),
            "fused_exceeds_max_single": sens["fused_exceeds_max_single"],
        },
        "method_note": (
            "Unmodified biophasor.transform.encoder.tanh_phase_encode + "
            "biophasor.integration.multiomics.MultiOmicsIntegrator (cross_coherence, "
            "coherence_matrix, fuse) with documented defaults. No tuning. "
            "Null = sample-label shuffle of the protein layer (breaks co-assay pairing)."
        ),
        "verdict": verdict,
    }

    _plot(primary, sens, result)
    json.dump(result, open(os.path.join(OUTDIR, "multiomics_results.json"), "w"), indent=1)
    print("  ->", json.dumps({k: result[k] for k in
          ["claim_a_cross_layer_coupling", "claim_b_fusion_exceeds_single", "verdict"]}, indent=1))
    return result


def _plot(primary, sens, result):
    """Emit three single-panel PNGs (one panel per file); combining into a
    multi-panel figure happens later in LaTeX. Data and colors are unchanged
    from the previous combined figure."""
    _apply_style()

    # ---- PNG 1: RNA/protein coherence matrix heatmap (stands alone) --------
    fig = plt.figure(figsize=(2.5, 2.3))
    ax = fig.add_subplot(1, 1, 1)
    Cm = np.array(primary["coherence_matrix"])
    im = ax.imshow(Cm, cmap="viridis", vmin=0, vmax=max(0.35, Cm.max() * 1.1))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["RNA", "protein"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["RNA", "protein"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{Cm[i, j]:.3f}", ha="center", va="center",
                    color="white" if Cm[i, j] < 0.22 else "black", fontsize=10)
    ax.set_title("RNA–protein coherence matrix", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path = os.path.join(FIGDIR, "multiomics_coherence_matrix.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")

    # ---- PNG 2: cross-coherence vs sample-shuffle null --------------------
    fig = plt.figure(figsize=(3.5, 2.8))
    ax = fig.add_subplot(1, 1, 1)
    nm, ns = primary["null_mean"], primary["null_std"]
    xs = np.linspace(nm - 5 * ns, primary["cross_coherence"] + 2 * ns, 300)
    ax.fill_between(xs, np.exp(-(xs - nm) ** 2 / (2 * ns ** 2)),
                    color="#bbbbbb", alpha=0.7, label=f"shuffled null\n{nm:.3f}±{ns:.3f}")
    ax.axvline(primary["cross_coherence"], color="#C44E52", lw=2.2,
               label=f"observed\n{primary['cross_coherence']:.3f}")
    ax.set_yticks([])
    ax.set_xlabel("RNA–protein cross-coherence")
    ax.set_title("Cross-layer coupling vs null", loc="left")
    ax.legend(frameon=False, loc="upper right", borderaxespad=0.4)
    path = os.path.join(FIGDIR, "multiomics_null_test.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")

    # ---- PNG 3: per-layer vs fused coherence (the refuted claim) ----------
    fig = plt.figure(figsize=(3.1, 2.6))
    ax = fig.add_subplot(1, 1, 1)
    labels = ["RNA", "protein", "fused"]
    vals = [primary["mean_coh_rna"], primary["mean_coh_protein"], primary["mean_coh_fused"]]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    mx = max(primary["mean_coh_rna"], primary["mean_coh_protein"])
    ax.axhline(mx, color="k", ls="--", lw=1)
    ax.text(2.4, mx, "max single-layer", va="bottom", ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
                ha="center", va="bottom")
    ax.set_ylabel("mean coherence (across samples)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title(f"Fused < best single (Δ={primary['fused_vs_max_single_delta']:+.3f})",
                 loc="left")
    path = os.path.join(FIGDIR, "multiomics_fusion_bars.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] {path}")


if __name__ == "__main__":
    print("=== Experiment 3: Multi-Omics Coherence Fusion (CPTAC UCEC) ===")
    run()
    print("Done. Outputs in:", OUTDIR)
