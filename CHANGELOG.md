# Changelog

All notable changes to **BioPhasor** are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned
- Phasor graph neural networks and gene-regulatory-network inference in `network`
- Spectral decomposition and differential-phase feature selection in `analysis`
- Hosted API documentation
- Publication to PyPI

## [0.1.0] — 2026-07-26

First public release under the Quantum Omics Foundation, licensed Apache-2.0.

### Added

**`core`** — the shared scientific foundation.
Phasor encoding (`tanh_phase_encode`, `log_linear_encode`, `linear_encode`),
circular operators (`coherence`, `phasor_mean`, phase-locking value), manifold
geometry on the $N$-torus (geodesic distance, Fréchet mean, log/exp maps),
biological graph construction, a circular and port-Hamiltonian loss library,
curated pathway sets, and synthetic data generation.

**`cst`** — the Cell State Tensor: tensor construction, geometry, attractor
analysis, and limit-cycle detection.

**`dynamics`** — coupled-oscillator models: Kuramoto dynamics on biological
graphs, cell-cycle phase assignment, circadian phase inference with ZT mapping
and rhythmicity scoring, and synchrony metrics (PLV, PLI, synchronisation
index, per-feature Rayleigh tests).

**`phnn`** — port-Hamiltonian neural networks for cellular dynamics: models,
training loops, and integrators.

**`spectral`** — spectral connectome, spectral omics, and quantum-duality
analyses.

**`integration`** — multi-omics fusion across modalities, with circular-mean,
concatenation, and coherence-weighted strategies plus cross-coherence matrices.

**`transform`** — the Biological Phasor Transform with multi-harmonic support,
phasor wavelets, and an auto-dispatching encoder.

**`ml`** — `PhasorClassifier`, an sklearn-compatible classifier over
phase-encoded features, and circular loss functions.

**`io`** — loaders for RNA-seq (TSV/CSV/H5AD/MEX), single-cell (10x
CellRanger), proteomics (MaxQuant, Proteome Discoverer), and metabolomics
(LC-MS, NMR), with format auto-detection.

**`network`**, **`analysis`**, **`viz`**, **`visualization`**, **`utils`** —
network construction, analysis helpers, phasor plots, and circular-statistics
utilities including `rayleigh_test` and angle handling.

### Project setup
- Apache-2.0 licence with `NOTICE`; SPDX identifiers in every source file.
- `CITATION.cff` with the two bioRxiv preprints describing the methods.
- Test suite spanning the core operators, Cell State Tensor, dynamics,
  integration, the port-Hamiltonian layer, and the spectral module.
- Contribution guide with DCO sign-off; black, ruff, mypy, and pytest
  configuration.
