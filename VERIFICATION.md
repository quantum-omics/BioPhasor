# BioPhasor Platform — Phase 5 Verification

Three checks certify the unified platform is coherent and genuinely deduplicated.

## 1. Full regression — PASS
`make test` → **181 passed, 12 skipped** (core 96 + phnn 45/12skip + spectral 40).
Identical to the pre-migration baseline; the 12 phnn skips are pre-existing and preserved.

## 2. One-import smoke — PASS
`import biophasor` + exercising the shared core once:
- `tanh_phase_encode` (30×40) → coherence 0.2318
- `build_biological_graph` → 19 adjacency blocks
- `generate_multi_omics` → 18 layers
- both domains reachable (`biophasor.phnn.models`, `biophasor.spectral.connectome.phasor`)
- all 18 subpackages import from a clean editable install.

## 3. Duplication scan — CLEAN
AST scan of every package module for repeated top-level definitions. Every
canonical shared symbol resolves to exactly ONE implementation:

| symbol | single home |
|---|---|
| tanh_phase_encode, log_linear_encode, linear_encode, OmicsPhasorEncoder | `core/encoder.py` |
| coherence, phase_coherence, phasor_statistics | `core/operators.py` |
| build_biological_graph, build_compartment_structure | `core/graph/bio_graph.py` |
| generate_multi_omics | `core/datagen/omics_data_generator.py` |
| circular_mse_loss, coherence_loss | `ml/losses.py` |

Two names appear in a second module but are **not** duplicate implementations
(verified):
- `Synchrony.coherence` (dynamics/synchrony.py) — a 1-line instance-method alias
  to `order_parameter`, not a copy of the free function.
- `PhasorEncoder.phasor_statistics` (spectral/connectome/phasor.py) — body is
  `return phasor_statistics(psi)`, importing and **delegating** to
  `core.operators` (identical output verified). The class keeps its method
  surface; the implementation lives once in core.

Benign per-class method names (`__init__`, `forward`, `fit`, `step`, `simulate`,
`build`) are legitimately distinct across classes and not flagged as concerns.
