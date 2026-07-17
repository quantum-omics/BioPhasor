# BioPhasor Experiments — Real-Data Validation (Plan II)

**Restart-point document.** This folder holds every experiment script, cached
dataset, result file, and figure for the BioPhasor first real-data draft. The
compiled manuscript lives in the project root at
[`../../biophasor-manuscript/`](../../biophasor-manuscript) (organized like
`Y-Omics/3-hnn-omics`). Everything regenerates from source with the run recipe
below.

Status: **all 9 framework scenarios measured on small/local real data**, each
with an honest verdict — **4 reproduce, 3 partial, 2 do-not-reproduce**. The
next iteration scales to full data on a GPU/cloud VM
([reports/NEXT_STEPS.md](reports/NEXT_STEPS.md)).

Start here: [reports/FINDINGS.md](reports/FINDINGS.md) (results),
[reports/RUN_LOG.md](reports/RUN_LOG.md) (what was done),
[reports/NEXT_STEPS.md](reports/NEXT_STEPS.md) (what's next),
[reports/plan2_verdicts.md](reports/plan2_verdicts.md) (per-scenario detail).

---

## Folder layout

```
2-biophasor/
├── biophasor-manuscript/     ← MANUSCRIPT at root: main.tex, main.pdf, references.bib, figure PNGs
└── biophasor/
    └── experiments/          ← you are here
        ├── README.md
        ├── codes/            ← one script per scenario (exp01–exp05) + _figstyle.py
        ├── data/
        │   ├── raw/          ← public source datasets (do NOT edit); cptac_ucec cache
        │   └── README.md     ← dataset provenance + sha256
        ├── results/          ← per-scenario result files (JSON/CSV/…), one per scenario
        ├── figures/          ← 30 publication figure PNGs (one panel per file) + 1 legacy summary
        ├── reports/          ← FINDINGS, RUN_LOG, NEXT_STEPS, plan2_verdicts
        ├── feasibility-and-plan/  ← real-data loader (tool) + plan-II.md (GPU roadmap)
        └── notebooks/        ← 15 exploratory notebooks (pre-validation)
```

Result files (`results/`) and figures (`figures/`) are kept separate: `results/`
holds whatever data format a scenario emits (JSON here, CSV where appropriate);
`figures/` holds the rendered PNGs the manuscript embeds.

---

## Scenario → script → results → figures → manuscript map

Each scenario has a script in `codes/`, a `<name>_results.*` file in `results/`,
one or more single-panel PNGs in `figures/`, a manuscript subsection
(`§subsec:r_*`), and a verdict in `reports/plan2_verdicts.md`.

| # | Scenario | Script (`codes/`) | Result (`results/`) | Figures (`figures/`) | § | Verdict |
|---|---|---|---|---|---|---|
| 1 | Cell-cycle | `exp01_cellcycle_assignment.py` | `cellcycle_real_results.json` | `cellcycle_confusion_fixed`, `_continuous`, `cellcycle_phase_rose` | `r_cellcycle` | **reproduces** (acc 0.34→0.69, G1 recall 0.002→0.98) |
| 2 | Circadian | `exp02_circadian_rhythm.py` | `circadian_real_results.json` | `circadian_zt_traces`, `_peak_phase`, `_score_dist` | `r_circadian` | **partial** (MAE 10.6h→1.4h; recall 0.43 sampling-limited) |
| 3 | Encoding | `exp03_encoding_coherence.py` | `encoding_results.json` | `encoding_coherence_hist`, `_dropout_scatter`, `_variance_selection`, `_enrichment` | `r_encoding` | **does-not-reproduce** (dropout statistic, r=−0.975) |
| 4 | Kuramoto | `exp03_kuramoto_synchrony.py` | `kuramoto_results.json` | `kuramoto_transition`, `_topology`, `_plv_matrix` | `r_kuramoto` | **reproduces** 3/3 (R∞ 0.12→0.98) |
| 5 | Multi-omics fusion | `exp03_multiomics_fusion.py` | `multiomics_results.json` | `multiomics_coherence_matrix`, `_null_test`, `_fusion_bars` | `r_multiomics` | **partial** (coupling z=145 reproduces, fusion refuted) |
| 6 | Manifold geometry | `exp04_manifold_geometry.py` | `manifold_results.json` | `manifold_euclid_vs_geodesic`, `_branchcut_cells`, `_branchcut_genes` | `r_manifold` | **reproduces** (arith-mean err up to 178°) |
| 7 | ML / VPC | `exp04_ml_classification.py` | `ml_results.json` | `ml_auc_folds`, `_param_efficiency`, `_coherence_score` | `r_ml` | **partial** (VPC 8× leaner than MLP, beaten by LogReg) |
| 8 | Attractor + Floquet | `exp04_attractor_floquet.py` | `attractor_results.json`, `floquet_results.json` | `attractor_quasipotential`, `_markov`, `_lyapunov`, `floquet_stability_curve`, `_rotating_state` | `r_attractor`, `r_floquet` | **reproduces** (method demo) |
| 9 | Cell State Tensor + knockout | `exp05_cst_knockout.py` | `cst_results.json` | `cst_evolution`, `_knockout_rank`, `_enrichment` | `r_cst` | **does-not-reproduce** (0 essentials in top-100) |
| 10 | CST temporal profile | `exp06_cst_temporal.py` | `cst_temporal_results.json` | `cst_temporal_cellcycle`, `_velocity`, `_circadian`, `cst_ema_smoothing` | `r_cst_temporal` | **partial** (cell-cycle reproduces G1→G2M peak 0.845; circadian Nyquist-limited at 6 ZT; EMA rule holds) |
| 11 | CST tensor-network + uncertainty | `exp07_cst_tensornetwork.py` | `cst_tensornetwork_results.json` | `cst_tt_energy_spectrum`, `_compression_error`, `_storage_scaling`, `cst_uncertainty` | `r_cst_tn` | **partial** (single CST not low-rank, best TT 65% err @3.2×; history storage sublinear 2.7×→5.1×; uncertainty=heterogeneity not dropout, ρ=0.08) |
| 12 | CST pathway atlas + low-rank test | `exp08_cst_pathway.py` | `cst_pathway_results.json` | `cst_pathway_spectrum`, `_cp_error`, `_coherence` | `r_cst_pathway` | **partial** (pathway atlas collapses regulatory spectrum rank-1=52.5% vs flat rank-50=50%; CP rank-3=62% err, full-tensor bounded by across-sample heterogeneity) |
| 13 | Central-dogma cross-modal coupling (omics-PAC) | `exp09_cst_omics_pac.py` | `cst_omics_pac_results.json` | `cst_omics_pac_comodulogram`, `_null`, `_ranking` | `r_cst_pac` | **reproduces** (mRNA phase→protein amplitude MI z=190 vs sample-permuted null; 72.6% of genes sig; tumour-specific r=0.30 vs normal 0.03) |
| 14 | CST density-matrix quantum correspondence | `exp10_cst_quantum_bridge.py` | `cst_quantum_bridge_results.json` | `cst_quantum_density`, `_entropy`, `_coherence`, `_fidelity` | `r_cst_quantum` | **reproduces** (S(ρ)~ℰ r=0.66 with ℰ≥S(ρ); C_ℓ1~𝒢 r=0.90; tumour/normal fidelity F=0.50, D=0.63; correspondence is definitional, no quantum advantage) |
| 15 | VPC→VQC gate correspondence + complexity | `exp11_vpc_vqc_complexity.py` | `vpc_vqc_complexity_results.json` | `cst_vpc_vqc_circuit`, `cst_complexity_crossover`, `cst_quantum_kernel_classification` | `r_cst_quantum` | **partial** (Shift/Mix/DFT→Rz/CNOT+Rz/QFT exact; favourable scaling above crossover; quantum-kernel shows no robust advantage after fixing class-imbalance artefact) |

`_figstyle.py` is the shared figure-style helper. `results/loader_reproduced_results.json`
and `figures/biophasor_realdata_summary.png` are earlier feasibility-stage
provenance artifacts (scenarios 1–2 only).

---

## Datasets

| Dataset | Used by | Source | Cached at |
|---|---|---|---|
| GSE293316 (REH B-ALL scRNA-seq) | 1, 3, 4, 6, 8 | NCBI GEO | `data/raw/GSE293316_reh.h5` |
| GSE171432 (WT mouse-liver circadian) | 2, 6 | NCBI GEO | `data/raw/GSE171432_fpkm.tsv.gz` |
| CPTAC UCEC (matched RNA+protein) | 5, 7, 9, 12, 13, 14, 15 | `cptac` package | `data/raw/cptac_ucec/` (see its README) |

Provenance and sha256 in [`data/README.md`](data/README.md); CPTAC cache in
[`data/raw/cptac_ucec/README.md`](data/raw/cptac_ucec/README.md). The h5 and
cptac cache are gitignored (regeneratable).

---

## How to re-run

**Environment.** Package venv (`biophasor/.venv`, Python 3.11.15, runtime env
`biophasor-dev`). The venv's `pip` shebang is broken from a repo move — use
`python -m pip` and set `PYTHONPATH`. `phasorflow` (scenario 7) is imported from
the sibling `../../phasorflow` on the same `PYTHONPATH`.

**Run recipe** (from the package root, `biophasor/`):
```bash
cd /Users/vasu/Desktop/CHUB/Y-Omics/2-biophasor/biophasor
PYTHONPATH=/Users/vasu/Desktop/CHUB/Y-Omics/2-biophasor \
  ./.venv/bin/python experiments/codes/exp01_cellcycle_assignment.py
```
Each script reads `data/raw/`, writes its result file to `results/` and its PNGs
to `figures/`. Scripts are seeded → result files are byte-identical across runs.

**Change a figure without recomputing.** Edit only the `_plot` function in the
scenario's `codes/exp*.py`; the computation above it is untouched. Re-run the
script — the result file stays byte-identical (verify with
`git diff --stat results/<name>_results.json`) while `figures/*.png` refresh.

**Unit tests** for the two package fixes:
```bash
PYTHONPATH=/Users/vasu/Desktop/CHUB/Y-Omics/2-biophasor \
  ./.venv/bin/python -m pytest tests/test_dynamics_fixes.py -q   # 6 tests; full suite 92
```

**Rebuild the manuscript** (host MacTeX, TeX Live 2026):
```bash
cd /Users/vasu/Desktop/CHUB/Y-Omics/2-biophasor/biophasor-manuscript
export PATH=/Library/TeX/texbin:$PATH
pdflatex -interaction=nonstopmode main.tex && bibtex main && \
  pdflatex -interaction=nonstopmode main.tex && \
  pdflatex -interaction=nonstopmode main.tex
# → main.pdf, 62 pages, 0 undefined refs, 82 bibitems
```
When a figure changes, copy the refreshed PNG from `experiments/figures/` into
the manuscript folder before rebuilding.

---

## The two package fixes (Plan II §0)

1. **Cell-cycle continuous axis** (`biophasor/dynamics/cellcycle.py`) —
   `assign` defaults to `method="continuous"`; legacy `"fixed"` retained.
2. **Circadian ZT anchoring** (`biophasor/dynamics/circadian.py`) — added
   `zt_origin` and `peak_zt()` mapping BPT phase to absolute Zeitgeber time.

Both with unit tests (`tests/test_dynamics_fixes.py`), committed `d1ea985`.
Phase 2 (6 scenarios): `be94d13`. Verdicts fix: `f5e0eab`.
