# biophasor — the platform / method paper's experiment suite

Backs `manuscripts/biophasor` (*A Phase-Geometric Platform for Decoding Cellular
State Tensors from Multi-Omics, toward Quantum-Ready Systems Biology*), which
presents BioPhasor as a formulation plus released software and demonstrates it
across illustrative applications.

```bash
PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.run_all
PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.run_all --list
PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.run_all --check   # dry run
PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.run_all exp01 exp07   # a subset
```

Every scenario keeps an honest verdict — reproduces / partial /
does-not-reproduce — in the table below and in
[reports/plan2_verdicts.md](reports/plan2_verdicts.md). Start with
[reports/FINDINGS.md](reports/FINDINGS.md) for results,
[reports/CHANGELOG_revision.md](reports/CHANGELOG_revision.md) for how each
review finding was answered.

---

## Layout

```
experiments/biophasor/
├── codes/      one script per scenario + run_all.py
├── results/    the numbers the manuscript quotes (tracked)
├── data/       dataset notes (the data itself is in _shared/data/raw)
└── reports/    FINDINGS, RUN_LOG, NEXT_STEPS, plan2_verdicts, CHANGELOG_revision
```

There is no `figures/` here. **Figures are written once, into
`manuscripts/biophasor/`**, which is what `main.tex` resolves
`\includegraphics` against (`\graphicspath{{./}}`).

Shared plumbing — the data cache, the path convention, the figure style, the
CPTAC complete-case loader — lives in `experiments/_shared/`, not here.

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
| 13 | Central-dogma cross-modal coupling (omics-PAC) — **moved to `experiments/tumor/`** | `../tumor/codes/exp09_cst_omics_pac.py` | `cst_omics_pac_results.json` | `cst_omics_pac_comodulogram`, `_null`, `_ranking` | `r_cst_pac` | **reproduces** (mRNA phase→protein amplitude MI z=190 vs sample-permuted null; 72.6% of genes sig; tumour-specific r=0.30 vs normal 0.03) |
| 14 | CST density-matrix quantum correspondence | `exp10_cst_quantum_bridge.py` | `cst_quantum_bridge_results.json` | `cst_quantum_density`, `_entropy`, `_coherence`, `_fidelity` | `r_cst_quantum` | **reproduces** (S(ρ)~ℰ r=0.66 with ℰ≥S(ρ); C_ℓ1~𝒢 r=0.90; tumour/normal fidelity F=0.50, D=0.63; correspondence is definitional, no quantum advantage) |
| 15 | VPC→VQC gate correspondence + complexity | `exp11_vpc_vqc_complexity.py` | `vpc_vqc_complexity_results.json` | `cst_vpc_vqc_circuit`, `cst_complexity_crossover`, `cst_quantum_kernel_classification` | `r_cst_quantum` | **partial** (Shift/Mix/DFT→Rz/CNOT+Rz/QFT exact; favourable scaling above crossover; quantum-kernel shows no robust advantage after fixing class-imbalance artefact) |

Scenario 13 (Omics-PAC) is the tumour paper's headline result and moved with it
to `experiments/tumor/`, along with its hardened-statistics companion
(`exp09b`) and their result files. Scenario 12's pathway-atlas figures are
printed by BOTH manuscripts; `exp08_cst_pathway.py` stays here (the method
paper's suite owns shared scripts) and the tumour driver copies its three
figures across.

Beyond the numbered scenarios, `codes/` also holds the revision experiments the
review round added: `exp12_multiomics_benchmark` and
`exp12b_benchmark_competitors_fix` (head-to-head against MOFA+ and SNF on the
identical matrix), `exp13_hardtask_clinical` (histologic grade and subtype), and
`coherence_axis_fix_run` (coherence rescoped to a supported axis).
`biophasor_realdata_loader.py` is a data-fetch / feasibility tool, not a result
producer, and is deliberately not in `run_all.py`'s order — run it directly to
refill the shared cache from GEO.

The figure style helper is now `experiments/_shared/figstyle.py` (two suites use
it). `results/loader_reproduced_results.json` is an earlier feasibility-stage
provenance artifact (scenarios 1–2 only).

### Figures the manuscript prints that nothing regenerates

`fig_platform_overview.png` (the Figure 1 pipeline schematic),
`fig2_benchmark.png` and `fig3_hardtask_roc.png` are in
`manuscripts/biophasor/` and are `\includegraphics`'d by `main.tex`, but **no
script in this repository, in `1-Spectral`, or in `5-Biophasor-Local` draws
them**. `reports/CHANGELOG_revision.md` describes them being built, so the code
existed and was never committed. The *numbers* behind the latter two are
regenerated — `exp12`/`exp12b` rewrite `multiomics_benchmark_results.json` and
`exp13` rewrites `hardtask_clinical_results.json` — so only the rendering step
is missing. The schematic is a drawn asset rather than a measured one and, on
the Classical-Virtual-Omics convention, would belong in
`manuscripts/biophasor/codes/` rather than in this suite.

---

## Datasets

| Dataset | Used by | Source | Cached at |
|---|---|---|---|
| GSE293316 (REH B-ALL scRNA-seq) | 1, 3, 4, 6, 8 | NCBI GEO | `../_shared/data/raw/GSE293316_reh.h5` |
| GSE171432 (WT mouse-liver circadian) | 2, 6 | NCBI GEO | `../_shared/data/raw/GSE171432_fpkm.tsv.gz` |
| CPTAC UCEC (matched RNA+protein) | 5, 7, 9, 12, 14, 15 | `cptac` package | `../_shared/data/raw/cptac_ucec/` |

All three live in the ONE shared cache and are reached by
`common.CACHE`, never by a hard-coded path. **Citations, sampling designs and
the limitation each dataset imposes on what can be concluded are in
[`../_shared/data/PROVENANCE.md`](../_shared/data/PROVENANCE.md)** — read it
before quoting a number. In short: the cell-cycle reference is algorithmic
rather than measured, the circadian series is 6 points over one cycle (at the
Nyquist floor), and the CPTAC cohort is cross-sectional with n = 14 normals.

---

## How to re-run

**Environment.** The repository venv, `.venv` (Python 3.11.15), with `biophasor`
installed editable. `phasorflow` (scenario 7) is an optional PyPI package
(`pip install phasorflow`); experiments that use it record
`phasorflow_available` in their results and fall back cleanly when it is absent.

**Whole suite**, from the repository root:
```bash
cd /Users/vasu/Desktop/CHUB/Y-Omics/BioPhasor
PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.run_all
```

**One script:**
```bash
PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.exp01_cellcycle_assignment
```

`PYTHONPATH=.` is required: the scripts import `experiments._shared` for
plumbing and the installed `biophasor` for science, and none of them touch
`sys.path`. Each reads the shared cache, writes its result file to `results/`
and its PNGs to `manuscripts/biophasor/`. Scripts are seeded → result files are
byte-identical across runs.

**Change a figure without recomputing.** Edit only the `_plot` function in the
scenario's `codes/exp*.py`; the computation above it is untouched. Re-run the
script — the result file stays byte-identical (verify with
`git diff --stat experiments/biophasor/results/<name>_results.json`) while the
manuscript PNGs refresh.

**Unit tests** for the two package fixes:
```bash
.venv/bin/python -m pytest tests/test_dynamics_fixes.py -q
```
Use `.venv/bin/python`, not the system `python3` — the latter lacks `anndata`
and reports three spurious failures in that file.

**Rebuild the manuscript** (host MacTeX):
```bash
cd manuscripts/biophasor
export PATH=/Library/TeX/texbin:$PATH
pdflatex -interaction=nonstopmode main.tex && bibtex main && \
  pdflatex -interaction=nonstopmode main.tex && \
  pdflatex -interaction=nonstopmode main.tex
```
No figure copying step: the experiments already wrote their PNGs into that
directory.

---

## The two package fixes (Plan II §0)

1. **Cell-cycle continuous axis** (`biophasor/dynamics/cellcycle.py`) —
   `assign` defaults to `method="continuous"`; legacy `"fixed"` retained.
2. **Circadian ZT anchoring** (`biophasor/dynamics/circadian.py`) — added
   `zt_origin` and `peak_zt()` mapping BPT phase to absolute Zeitgeber time.

Both with unit tests (`tests/test_dynamics_fixes.py`), committed `d1ea985`.
Phase 2 (6 scenarios): `be94d13`. Verdicts fix: `f5e0eab`.
