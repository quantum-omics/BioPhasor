# Changelog

All notable changes to **BioPhasor** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

## [0.1.0] — 2026-08-02

First release. BioPhasor fits phasor geometry to multi-omics measurements:
abundances are encoded onto the unit circle, and the resulting phase structure
is analysed with circular statistics, coupled-oscillator dynamics, and the
spectral decomposition of an omics connectome.

### Added

**`biophasor.core`** — the shared foundation. Phasor encoding
(`tanh_phase_encode`, `log_linear_encode`, `linear_encode`), circular operators
(`coherence`, `phasor_mean`, `phase_couple`, phase-locking value), manifold
geometry on the $N$-torus (geodesic distance, Fréchet mean, log/exp maps),
curated pathway sets, and the circular loss family (`circular_mse_loss`,
`coherence_loss`, `von_mises_kl_loss`).

**`biophasor.cst`** — the Cell State Tensor: tensor construction, attractor
landscape and geometry, limit-cycle detection, and pathway-resolved
decomposition.

**`biophasor.dynamics`** — coupled-oscillator models: Kuramoto dynamics on
biological graphs, cell-cycle phase assignment, circadian phase inference with
ZT mapping and rhythmicity scoring, and synchrony metrics (PLV, PLI,
synchronisation index, per-feature Rayleigh tests).

**`biophasor.spectral`** — the omics connectome and its spectrum. The classical
layer reduces an expression matrix to a Hermitian Omics Connectome Matrix whose
eigenvalues define collective harmonic frequencies (`connectome.harmonics`,
`.magnetic`, `.ocm`, `.phasor`), with compartment weights, consistency
indicators and state records over it (`omics.*`) and an end-to-end `pipeline`.

**`biophasor.spectral.quantum`** — the second-quantised layer: a truncated Fock
space over five compartment modes, a particle-number-conserving Bose–Hubbard
Hamiltonian, exact sector dynamics, and the compartment covariance readout with
its coherence measure. The occupation numbers are the representation's
variables; nothing here asserts that a cell is a quantum system.

**`biophasor.integration`** — multi-omics fusion across modalities, with
circular-mean, concatenation and coherence-weighted strategies plus
cross-coherence matrices.

**`biophasor.transform`** — the Biological Phasor Transform with multi-harmonic
support, phasor wavelets, and an auto-dispatching encoder.

**`biophasor.ml`** — `PhasorClassifier`, an sklearn-compatible classifier over
phase-encoded features, and circular loss functions.

**`biophasor.io`** — loaders for RNA-seq (TSV/CSV/H5AD/MEX), single-cell (10x
CellRanger), proteomics (MaxQuant, Proteome Discoverer) and metabolomics
(LC-MS, NMR), with format auto-detection.

**`biophasor.utils`** — circular-statistics helpers including `rayleigh_test`
and angle handling, AnnData utilities, and the manuscript number guard
(`GuardConfig`, `check_numbers`, `run_guard`).

**`biophasor.viz` / `.visualization`** — phasor plots and figure primitives.

### Experiments

Four suites, one per manuscript, each with `codes/`, `results/` and a
`run_all.py` supporting `--list` and a `--check` dry run that validates inputs
and prints the plan without executing anything:

| suite | subject | runtime |
|---|---|---|
| `biophasor` | the platform: encoding, cell cycle, circadian, synchrony, CST | ~3 min |
| `tumor` | tumour cross-modal phase–amplitude coupling along the central dogma | ~24 min |
| `spectral-classical` | spectral connectome on cancer and circadian cohorts | ~28 s |
| `spectral-quantum` | the second-quantised compartment model | ~136 s |

`experiments/_shared/common.py` is the single home for experiment plumbing —
`results_dir(suite)`, `manuscript_figs(suite)`, `save_fig()`, `check_plan()`,
the shared GEO fetch and the preprocessing gates. Raw data lives in one cache,
`experiments/_shared/data/raw/`, with every accession, citation and sampling
limitation recorded in `PROVENANCE.md`.

Each suite ends with a **number guard**: every numeric literal in that
manuscript's results section must round-trip to a value in the suite's
`results/*.json` at the precision written.

### Known issues

Recorded rather than papered over.

- **SNF is broken by a library incompatibility.** `snfpy` 0.2.2 — the newest
  release, and unmaintained — calls
  `sklearn.utils.check_array(force_all_finite=...)`; scikit-learn renamed that
  keyword in 1.6 and removed it in 1.7. The `experiments` extra pins
  `scikit-learn<1.7` so a fresh environment reproduces the quoted numbers, but
  `experiments/biophasor/results/multiomics_benchmark_results.json` currently
  holds the error string where the SNF adjusted Rand index and silhouette used
  to be — it was written under 1.9 and has not been regenerated.
- **MOFA+ factor extraction fails in exp12.** The script reads the expectation
  keyed `"group0"` while labelling its single group `"g1"`, so mofapy2 raises
  `KeyError`. The failure is caught and recorded as a benchmark error; the
  MOFA+ row that does carry numbers comes from the separate `exp12b` path.
- **Three result sets have no generating code**, and are carried forward as
  prior results rather than results this repository can regenerate:
  `spectral-classical` `circadian_rigor/`, `directed_flux/` and `enrichment/`
  (two of the eight figures `main.tex` prints come from the first, so that
  figure deck cannot be rebuilt in full — see
  `experiments/spectral-classical/results/ORPHANED.md`); `tumor`
  `fig1_omics_pac.png`; and `biophasor` `fig_platform_overview.png`,
  `fig2_benchmark.png`, `fig3_hardtask_roc.png`. In each case the *numbers* are
  regenerated by scripts that do run; only the rendering step is missing.
- **The DMRG solver has never been run here.** `physics-tenpy` is not installed
  in the development environment, so `run_dmrg.py` has only taken its
  cached-scan branch. The exact-diagonalisation path runs live on every
  pipeline run and cross-checks the cached energies, so the agreement the
  manuscript reports is re-verified each time; but the DMRG side of that
  comparison is read from disk. `--check --dmrg` reports the missing dependency
  and exits non-zero rather than claiming OK.
- `biophasor.analysis` imports but exposes no public API. It is a reserved
  placeholder, not a module.

### Project setup

- Apache-2.0 licence with `NOTICE`; SPDX identifiers in every source file.
- `CITATION.cff` naming the platform manuscript.
- 171 tests spanning the core operators, Cell State Tensor, dynamics,
  integration, transform, classifier, the spectral and quantum modules, and the
  number guard.
- Contribution guide with DCO sign-off; black, ruff, mypy and pytest
  configuration; a CI workflow covering lint, the test matrix, a packaging
  check and the fast suites.
