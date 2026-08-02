<div align="center">

# BioPhasor

**Phasor geometry fitted to omics measurements.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Stewarded by Quantum Omics Foundation](https://img.shields.io/badge/stewarded%20by-Quantum%20Omics%20Foundation-8a90ff.svg)](https://github.com/quantum-omics-foundation)

</div>

---

## What this is, and what it is not

This repository **fits phasor geometry to omics measurements**. Each feature is
mapped to a complex number $\psi = r\,e^{i\theta}$, so a sample becomes a point
on the $N$-torus $\mathbb{T}^N$ and the tools of circular statistics, spectral
graph theory and coupled-oscillator dynamics apply directly to it.

That is a representation choice, not a claim about biology. A cell is not a
phasor, a gene is not an oscillator with a phase waiting to be read off, and
nothing here is a quantum system. Where the code and the manuscripts use
quantum vocabulary — density operators, Fock space, a Bose–Hubbard Hamiltonian —
they mean a *quantum-simulable formalism applied to omics data*, with an exact
classical↔quantum correspondence and **no quantum advantage claimed**. The
distinction decides which sentences are defensible: a claim about the fitted
geometry can be checked against the fit; a claim about the cell would need
evidence the data do not supply.

Treating a measurement as a phase makes a set of otherwise awkward questions
natural to ask:

- How synchronised is a pathway, as a single number in $[0, 1]$?
- Which genes lead and which lag, and by how much?
- Does a gene's transcript *phase* organise its protein *amplitude*?
- What are the collective normal modes of a co-expression graph once the edges
  carry a phase?

## Installation

Requires Python 3.10 or newer.

```bash
git clone <repo>/BioPhasor && cd BioPhasor
pip install -e .
```

The core install carries numpy, scipy, pandas, matplotlib, scikit-learn,
anndata, scanpy and torch — enough to import the whole package and run the test
suite. Reproducing the manuscripts needs the `experiments` extra, which is
where the datasets and the comparison methods live.

| Extra | Adds | For |
|---|---|---|
| `experiments` | GEOparse, cptac, mofapy2, snfpy, lifelines, tensorly, joblib, networkx, h5py, physics-tenpy, `scikit-learn<1.7` | reproducing the four manuscripts |
| `dev` | pytest, black, ruff, mypy | development |
| `docs` | mkdocs-material, mkdocstrings, mike | building the documentation |
| `tda` | gudhi | topological data analysis |
| `spatial` | squidpy | spatial transcriptomics |
| `ml` | torch-geometric | graph neural networks |
| `vpc` | phasorflow | the variational phasor circuit backend (the classifier degrades without it) |

```bash
pip install -e ".[dev,experiments]"
```

The `experiments` extra caps scikit-learn below 1.7 deliberately. `snfpy`
0.2.2 calls a scikit-learn keyword that 1.7 removed, and without the cap the
SNF comparison in the biophasor benchmark raises instead of producing the
numbers the manuscript quotes. See [`CHANGELOG.md`](CHANGELOG.md) under Known
issues.

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

## How it works

**Tanh-phase encoding** maps a continuous measurement to phase, standardising on
the log scale so that heavy-tailed omics distributions spread evenly around the
circle rather than bunching:

$$\varphi = \pi \cdot \tanh\!\left(\frac{\log(1+x) - \mu}{\sigma}\right)$$

**Phase coherence** summarises how aligned a set of phases is:

$$C = \left|\frac{1}{N}\sum_{n=1}^{N} e^{i\varphi_n}\right| \in [0, 1]$$

Because both are defined on the circle, downstream analysis inherits circular
statistics for free — circular means, phase differences that wrap correctly, and
synchrony measures that do not depend on an arbitrary origin.

Two derived objects carry most of the science. The **Cell State Tensor** is an
order-3 latent field over regulatory, temporal and homeostatic axes whose
regulatory slice is the phase-coherence density matrix
$\rho = \frac{1}{M}\sum_m \boldsymbol z_m \boldsymbol z_m^{\dagger}$, whose
off-diagonals are phase-locking values. The **Omics Connectome Matrix** is the
Hermitian $H_{ij} = c_{ij}\,\psi_i\psi_j^{*}$ whose eigenvectors are the
collective normal modes of a co-expression graph with phase-bearing edges.

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
| `biophasor.spectral.quantum` | the second-quantized layer: Fock space over five compartment modes, the Bose–Hubbard Hamiltonian, compartment covariance, and the shipped omics harmonic ladder |
| `biophasor.ml` | `PhasorClassifier` and circular loss functions |
| `biophasor.io` | loaders for RNA-seq, single-cell, proteomics, metabolomics, with format auto-detection |
| `biophasor.viz`, `.visualization` | phasor plots and figure helpers |
| `biophasor.utils` | AnnData helpers, circular statistics, the manuscript number guard |

`biophasor.analysis` imports but exposes no public API; it is a reserved
placeholder.

## Experiments

Four suites live under `experiments/<suite>/`: `biophasor`, `tumor`,
`spectral-classical` and `spectral-quantum`. Each imports the installed package,
reads from the shared cache in `experiments/_shared/data/`, writes its numbers to
its own `results/`, and is driven by

```bash
python experiments/<suite>/codes/run_all.py --list    # what it will run
python experiments/<suite>/codes/run_all.py --check   # validate inputs, run nothing
python experiments/<suite>/codes/run_all.py           # run
```

Every suite ends with a number guard: each numeric literal quoted in the
corresponding write-up must round-trip to a value in that suite's `results/*.json`
at the precision written, or the guard fails with the nearest candidates.

See [`experiments/README.md`](experiments/README.md).

## Scope, and the sibling repository

BioPhasor fits a *representation* to a measurement matrix: abundances are
encoded onto the unit circle and the resulting phase structure is analysed.

Dynamical-systems modelling of cellular state is a different question, and it
has its own repository. **Classical-Virtual-Omics** (package `cvomics`)
fits a port-Hamiltonian twin $(J, R, H, G)$ to multi-omic time courses and
treats cell types, diseases and therapies as deformations of that quadruple.
Reach for it when the object of interest is a trajectory or an intervention
rather than the phase geometry of a snapshot.

The two are deliberately separate installs: `cvomics` carries a training
loop and its dependencies, `biophasor` needs neither. The coupled-oscillator
machinery they might have shared lives here, in `biophasor.dynamics`.

Superseded prior-generation code is not carried in this repository; it remains
in the local source trees it came from.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the core-versus-experiment
decision rule and the number-guard discipline. Commits carry a DCO sign-off:

```bash
git commit -s -m "feat: your change"
```

## Citing

See [`CITATION.cff`](CITATION.cff).

## Governance

BioPhasor is stewarded by the **Quantum Omics Foundation**, a nonprofit
organisation advancing open research and education at the interface of quantum
computing and the life sciences. The Foundation's remit is to keep this work
openly available, reproducible, and usable by the wider research community.

## Licence

Apache-2.0 — see [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and
[`LICENSING.md`](LICENSING.md). Datasets are not redistributed here and carry
their own terms.
