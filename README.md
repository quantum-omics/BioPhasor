<div align="center">

# BioPhasor

**A phase-geometric library for multi-omics systems biology.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Stewarded by Quantum Omics Foundation](https://img.shields.io/badge/stewarded%20by-Quantum%20Omics%20Foundation-8a90ff.svg)](https://github.com/quantum-omics-foundation)

</div>

---

## Overview

Omics measurements are usually treated as magnitudes. BioPhasor treats them as
**phases**: each feature is mapped to a point on the unit circle, so a sample
becomes a point on the $N$-torus $\mathbb{T}^N$. Relationships between features
then become geometric — alignment, coherence, and synchrony — and the tools of
circular statistics and coupled-oscillator dynamics apply directly.

This makes a set of otherwise awkward questions natural to ask:

- How synchronised is a pathway, as a single number in $[0, 1]$?
- Which genes lead and which lag, and by how much?
- Is a cell population approaching a limit cycle, or drifting away from one?
- How do transcript and protein layers fall in and out of step over time?

BioPhasor provides the shared machinery for that: encoding, circular operators,
biological graphs, oscillator dynamics, the Cell State Tensor, and a
port-Hamiltonian modelling layer.

## Installation

Requires Python 3.10 or newer.

```bash
pip install -e .
```

Optional extras:

| Extra | Adds | For |
|---|---|---|
| `dev` | pytest, black, ruff, mypy | development |
| `tda` | gudhi | topological data analysis |
| `spatial` | squidpy | spatial transcriptomics |
| `ml` | torch-geometric | graph neural networks |
| `vpc` | phasorflow | variational phasor circuits |

```bash
pip install -e ".[dev]"
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

## Package layout

| Module | Contents |
|---|---|
| `biophasor.core` | encoding, circular operators, biological graphs, loss functions, pathway sets, data generation |
| `biophasor.transform` | phasor transforms and alternative encoders |
| `biophasor.cst` | Cell State Tensor — geometry, attractors, limit cycles |
| `biophasor.dynamics` | coupled-oscillator models: Kuramoto, circadian, cell cycle, synchrony |
| `biophasor.integration` | multi-omics fusion across modalities |
| `biophasor.phnn` | port-Hamiltonian neural networks — models, training, integrators |
| `biophasor.spectral` | spectral connectome, spectral omics, quantum duality |
| `biophasor.ml` | classifiers and circular loss functions |
| `biophasor.io` | loaders for RNA-seq, single-cell, proteomics, metabolomics |
| `biophasor.network` | network construction and analysis |
| `biophasor.viz`, `.visualization` | phasor plots and figure helpers |
| `biophasor.utils` | AnnData helpers, math utilities, logging |

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite covers the core operators, the Cell State Tensor, oscillator dynamics,
multi-omics integration, the port-Hamiltonian layer, and the spectral module.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
development workflow, coding conventions, and the DCO sign-off we ask for on
each commit:

```bash
git commit -s -m "feat: your change"
```

Please open an issue before starting substantial work, so we can check it fits
the library's scope and is not already in progress.

## Citing

If BioPhasor supports your research, please cite it — see
[`CITATION.cff`](CITATION.cff), which GitHub renders as a ready-to-use citation
in the sidebar.

## Governance

BioPhasor is stewarded by the **Quantum Omics Foundation**, a nonprofit
organisation advancing open research and education at the interface of quantum
computing and the life sciences. The Foundation's remit is to keep this work
openly available, reproducible, and usable by the wider research community.

## License

Licensed under the **Apache License 2.0** — see [`LICENSE`](LICENSE) and
[`NOTICE`](NOTICE). [`LICENSING.md`](LICENSING.md) records why Apache-2.0 was
chosen over a shorter permissive licence.

In brief: you may use, modify, distribute, and build commercial work on this
code. Retain the copyright notice, the `NOTICE` file, and a copy of the licence
in any redistribution, and state any changes you made.

Datasets used in research built on this library carry their own terms — check
the original source before redistributing data.
