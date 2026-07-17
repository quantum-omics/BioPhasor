# Changelog

All notable changes to **BioPhasor** are documented in this file.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Version numbers follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Planned
- `network` module: phasor graph neural networks, GRN inference
- `analysis` module: spectral decomposition, differential-phase genes
- Sphinx + Read the Docs build pipeline
- PyPI publishing workflow

---

## [0.1.0] — 2026-03-21

### Added
- **`core`** sub-package: `BioPhasor` data class, `PhasorManifold` geometry (geodesic distance, Fréchet mean, log/exp maps), circular operators (`coherence`, `phasor_mean`, `bio_shift`, `bio_mix`, `PLV`), biological constants (`CANONICAL_MARKER_GENES`, `CIRCADIAN_CORE_GENES`, `ENCODING_MAP`)
- **`io`** sub-package: loaders for RNA-seq (TSV/CSV/H5AD/MEX), single-cell (10x CellRanger), proteomics (MaxQuant/PD output), metabolomics (LC-MS/NMR), and `auto_detect` dispatcher
- **`transform`** sub-package: three encoding strategies (`tanh_phase_encode`, `log_linear_encode`, `linear_encode`), `OmicsPhasorEncoder` auto-dispatch, `BPT` (Biological Phasor Transform) with multi-harmonic support, `PhasorWavelet`
- **`dynamics`** sub-package: `BioKuramoto` coupled oscillator model, `CellCyclePhasor` (G1/S/G2/M assignment + phase scores), `CircadianPhasor` (BPT-based phase inference, ZT mapping, rhythmicity score, simulation), `SynchronyMetrics` (PLV matrix, PLI matrix, synchronisation index, Rayleigh per feature)
- **`integration`** sub-package: `MultiOmicsIntegrator` (circular_mean / concat / coherence_weighted fusion), cross-coherence matrix, `integrate()` convenience function
- **`ml`** sub-package: `PhasorClassifier` (VPC + sklearn fallback, cross-validation, `force_fallback`), circular loss functions (`circular_mse_loss`, `coherence_loss`, `von_mises_kl_loss`)
- **`visualization`** sub-package: `PhasorPlot` (G×S scatter, polar histogram, coherence bar chart)
- **`utils`** sub-package: `rayleigh_test`, circular mean/variance/std, `wrap_angle`, `angular_distance`, AnnData helpers, logging
- **`tests/`**: 62 pytest unit tests covering all modules (pass in ~12 seconds)
- **`notebooks/`**: 11 executed demonstration notebooks covering encoding, coherence gene selection, cell cycle, Kuramoto dynamics, circadian rhythms, BPT time-series, multi-omics integration, classification, synchrony metrics, proteomics, and manifold geometry
- **`docs/`**: 15-chapter documentation (index + chapters 01–15), mkdocs.yml with Material theme, MathJax, and mermaid diagram support
- `pyproject.toml` with full metadata, optional dependency groups (`tda`, `spatial`, `ml`, `dev`, `docs`), black/ruff/mypy/pytest configuration

### Fixed
- `pyproject.toml`: corrected `build-backend` and `where=[".."]` for package discovery
- `PhasorClassifier`: `force_fallback` parameter for reliable testing, AUC fallback to accuracy on single-class folds, `cross_val_score` propagates `force_fallback`

---

## [0.0.1] — 2026-03-15 *(internal)*

- Initial project scaffold and `pyproject.toml`
