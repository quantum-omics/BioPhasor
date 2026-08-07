<div align="center">

# BioPhasor

**Phasor geometry fitted to omics measurements.**


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21831941.svg)](https://doi.org/10.5281/zenodo.21831941)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Stewarded by Quantum Omics Foundation](https://img.shields.io/badge/stewarded%20by-Quantum%20Omics%20Foundation-8a90ff.svg)](https://github.com/quantum-omics-foundation)

</div>

---

## The representation

BioPhasor encodes each omics feature as a **phasor** — a complex number
$\psi = r\,e^{i\theta}$ carrying an amplitude and a phase. A sample becomes a
point on the $N$-torus $\mathbb{T}^N$, and cellular state is analysed with
circular statistics, coupled-oscillator dynamics and spectral graph theory
rather than with methods that assume a Euclidean feature space.

Abundance sets the phase through a tanh encoder on the log scale, so
heavy-tailed distributions spread evenly around the circle instead of bunching:

$$\varphi = \pi \cdot \tanh\\left(\frac{\log(1+x) - \mu}{\sigma}\right)$$

Alignment of a feature set is then a single number, the mean resultant length:

$$C = \left|\frac{1}{N}\sum_{n=1}^{N} e^{i\varphi_n}\right| \in [0, 1]$$

Because both are defined on the circle, downstream analysis inherits circular
means, phase differences that wrap correctly, and synchrony measures that do not
depend on an arbitrary origin. Questions that are awkward in abundance space —
how synchronised is a pathway, which genes lead and which lag, whether a
transcript's phase organises its protein's amplitude — become direct
measurements.

## Two derived objects

**The Cell State Tensor** is an order-3 latent field over regulatory, temporal
and homeostatic axes. Its regulatory slice is the phase-coherence density matrix

$$\rho = \frac{1}{M}\sum_m \boldsymbol z_m \boldsymbol z_m^{\dagger}$$

whose off-diagonals are phase-locking values between features. From it follow
the attractor landscape, limit cycles, knockout rankings and pathway-resolved
decompositions of cellular state.

**The Omics Connectome Matrix** is the Hermitian operator

$$H_{ij} = c_{ij}\,\psi_i\psi_j^{*}$$

built on a co-expression graph whose edges carry a phase. Its eigenvectors are
the collective normal modes of that graph, and its spectrum yields compartment
weights, coupling matrices and state classification.

## Installation

Requires Python 3.10 or newer.

```bash
git clone <repo>/BioPhasor && cd BioPhasor
pip install -e .
```

The core install carries numpy, scipy, pandas, matplotlib, scikit-learn,
anndata, scanpy and torch — enough to import the whole package and run the
tests. Optional extras: `experiments` (dataset access and comparison methods),
`dev`, `docs`, `tda` (gudhi), `spatial` (squidpy), `ml` (torch-geometric), and
`vpc` (phasorflow, the variational phasor circuit backend).

```bash
pip install -e ".[dev]"
pytest
```

## Quick start

```python
import numpy as np
from biophasor.core.encoder import tanh_phase_encode
from biophasor.core.operators import coherence

rng = np.random.default_rng(0)
x = rng.lognormal(size=(20, 200))      # 20 samples x 200 features

phi = tanh_phase_encode(x)             # phases in (-pi, pi], same shape as x
C = coherence(phi, axis=1)             # per-sample phase coherence in [0, 1]

print(C.shape, round(float(C.min()), 3), round(float(C.max()), 3))
# (20,) 0.13 0.345
```

`coherence` is the mean resultant length — the Kuramoto order parameter when no
amplitude weighting is given. A value near 0 means the features are dispersed
around the circle; near 1 means they are aligned.

## Package layout

| Module | Contents |
|---|---|
| `biophasor.core` | encoding, circular operators, circular losses, curated pathway sets, phasor manifold geometry on $\mathbb{T}^N$ |
| `biophasor.transform` | the Biological Phasor Transform, phasor wavelets, the auto-dispatching encoder |
| `biophasor.cst` | Cell State Tensor — construction, geometry, attractors, limit cycles, the pathway-atlas CST |
| `biophasor.dynamics` | coupled-oscillator models: Kuramoto on biological graphs, cell-cycle and circadian phase assignment, synchrony metrics |
| `biophasor.integration` | multi-omics fusion across modalities and cross-coherence matrices |
| `biophasor.spectral.connectome` | phasor vertices, the Hermitian OCM, omics harmonics, the magnetic-OCM variant |
| `biophasor.spectral.omics` | spectral indicators, the compartment coupling matrix, compartment weights, state classification |
| `biophasor.spectral.quantum` | second-quantized compartment model: Fock space over the five leading harmonics, a number-conserving Bose–Hubbard Hamiltonian, and the compartment covariance readout |
| `biophasor.ml` | `PhasorClassifier` and circular loss functions |
| `biophasor.io` | loaders for RNA-seq, single-cell, proteomics, metabolomics, with format auto-detection |
| `biophasor.viz`, `.visualization` | phasor plots and figure helpers |
| `biophasor.utils` | AnnData helpers, circular statistics, the manuscript number guard |

`biophasor.analysis` imports but exposes no public API; it is a reserved
placeholder.

## Experiments

Per-manuscript suites live under `experiments/`, each importing the installed
package and driven by its own `run_all.py`. See
[`experiments/README.md`](experiments/README.md).

## Scope

BioPhasor analyses the phase geometry of a measurement matrix. For
dynamical-systems modelling of cellular state — a port-Hamiltonian twin
$(J, R, H, G)$ fitted to multi-omic time courses, with cell types and diseases
as deformations of it — see the sibling repository **Classical-Virtual-Omics**
(package `cvomics`). The two are separate installs; the coupled-oscillator
machinery lives here, in `biophasor.dynamics`.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the core-versus-experiment
decision rule and the number-guard discipline. Commits carry a DCO sign-off:

```bash
git commit -s -m "feat: your change"
```

## Citing

See [`CITATION.cff`](CITATION.cff).

## Governance

Stewarded by the **Quantum Omics Foundation**, a nonprofit advancing open
research and education at the interface of quantum computing and the life
sciences.

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and
[`LICENSING.md`](LICENSING.md). Datasets are not redistributed here and carry
their own terms.
