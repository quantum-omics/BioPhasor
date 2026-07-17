# BioPhasor

[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey?style=flat-square)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-forthcoming-b31b1b?style=flat-square)]()

**BioPhasor** is a phase-native computational framework for multi-omics systems biology and cellular state modeling. It encodes omics features (RNA-seq, ATAC-seq, single-cell, proteomics, metabolomics) as complex phasors on the $N$-torus $\mathbb{T}^N$ and extracts interpretable cellular state descriptors via the **Cell State Tensor** (CST) — a rank-3 object encoding pathway-resolved, sample-specific phase-amplitude dynamics.

This repository is the reference implementation for the manuscript:

> D. Sigdel, *BioPhasor: Decoding Cellular State Tensors from Multi-Omics Phasor Dynamics for Quantum-Ready Systems Biology*, bioRxiv (forthcoming), 2026.

---

## Methodology

BioPhasor provides an end-to-end pipeline from raw omics measurements to cellular state estimates. Its core constructs are:

- **Tanh-Phase Encoding** — Maps continuous omics modalities (RNA-seq, proteomics) to phase values on the unit circle:

$$\varphi = \pi \cdot \tanh\!\left(\frac{\log(1+x) - \mu}{\sigma}\right), \quad z = A \cdot e^{i\varphi}$$

- **Phase Coherence** — Universal quality and synchronization metric based on mean resultant length:

$$C = \left|\frac{1}{N}\sum\_{n=1}^{N} e^{i\varphi\_n}\right| \in [0, 1]$$

- **Gene Regulatory Network Dynamics** — Simulates Kuramoto synchronization dynamics on arbitrary gene regulatory networks (GRNs):

$$\dot{\varphi}\_i = \omega\_i + \frac{K}{N}\sum\_j A\_{ij}\sin(\varphi\_j - \varphi\_i) + \eta\_i(t)$$

- **Cell State Tensor (CST)** — Four attractor-geometric descriptors derived from the phasor field:

| Descriptor | Definition | Interpretation |
|---|---|---|
| $R(t)$ | Kuramoto order parameter | Global pathway synchrony |
| $\mathcal{C}(t)$ | Temporal coherence | Basin stability |
| $E(t)$ | Phase entropy | Attractor diversity |
| $V(t)$ | Phase velocity | State transition rate |

- **Quantum-Classical Duality** — Each classical phasor operation (Shift/Mix/DFT) maps one-to-one onto a quantum circuit gate ($R\_z$/CNOT/QFT), providing a validated path to quantum hardware execution.

---

## Key Results

Evaluated on public cancer genomics datasets (GEO, CPTAC):

| Finding | Value |
|---|---|
| Tanh-phase encoding coherence (BRCA) | $C = 0.71$ (vs linear PCA $0.48$) |
| Cell-cycle phase assignment accuracy | $0.89$ ARI against fluorescence ground truth |
| GRN Kuramoto critical coupling | $K\_c = 12.4$ (empirical), $K\_c = 11.8$ (analytic) |
| Multi-omics integration (RNA + ATAC) | Silhouette $= 0.62$ (vs CCA $0.41$) |
| CST CP rank-3 reconstruction | RMSE $= 1.74$ |
| Quantum coherence tracks Kuramoto $R$ | $r = 0.937$ |

> Phase-based integration preserves oscillatory structure in gene expression that linear methods (PCA, CCA) collapse — consistent with the circadian and cell-cycle periodicity of transcriptomic programs.

---

## Installation

```bash
git clone https://github.com/mindverse-computing/biophasor.git
cd biophasor
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full]"
```

Requires Python ≥ 3.10. Core dependencies: NumPy, SciPy, scanpy, scikit-learn, matplotlib.

---

## Quick Start

```python
import biophasor as bp

# Encode gene expression as phasors
phasor = bp.tanh_phase_encode(X_rna)       # phi in [-pi, pi] per gene per sample

# Compute phase coherence
C = bp.coherence(phasor)                    # (n_genes,)

# Multi-omics integration
rna = bp.tanh_phase_encode(X_rna)
atac = bp.tanh_phase_encode(X_atac)
fused = bp.integrate([rna, atac])           # weighted circular mean fusion

# Construct the Cell State Tensor
from biophasor.cst import CellStateTensor
cst = CellStateTensor.from_phasor(phasor, pathways="hallmark")
descriptors = cst.descriptors()             # R(t), C(t), E(t), V(t)
```

---

## Reproducing Experiments

The `experiments/` directory contains seeded scripts that reproduce every result reported in the manuscript from open public datasets (GEO, CPTAC). Each script writes a results JSON and corresponding figures.

```bash
python experiments/codes/exp01_encoding_cst.py
python experiments/codes/exp05_kuramoto_grn.py
python experiments/codes/exp09_quantum_classical_bridge.py
```

---


## License

Released under the **CC BY-NC 4.0** license; commercial use is prohibited. See [LICENSE](LICENSE) for patent and trademark reservations by Mindverse Computing LLC.

---

## Citation

```bibtex
@article{sigdel2026biophasor,
  title   = {BioPhasor: Decoding Cellular State Tensors from
             Multi-Omics Phasor Dynamics for Quantum-Ready
             Systems Biology},
  author  = {Sigdel, Dibakar},
  journal = {bioRxiv preprint (forthcoming)},
  year    = {2026},
  url     = {https://github.com/mindverse-computing/biophasor}
}
```

---

[Mindverse Computing](https://www.mindversecomputing.com) · Quantum Virtual Mind (QVM) Project
