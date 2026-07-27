# Regression Baseline — pre-migration

Recorded before any core canonicalisation or import rewiring, using the
unified BioPhasor venv (`BioPhasor/.venv`, torch 2.10.0 + scanpy). This is the
target every migration step must hold or exceed.

| Suite | Source tree (baseline run) | Result |
|---|---|---|
| biophasor (core/cst/dynamics/integration/transform/ml) | `BioPhasor/tests/` | **96 passed** |
| phnn | `3-phnn-omics/codes/tests/` | **45 passed, 12 skipped** |
| spectral | `7-spectral-omics/codes/tests/` | **40 passed** |
| **TOTAL** | | **181 passed, 12 skipped** |

## How the migrated suites are gated
- `tests/` — biophasor-native, already runs green under the installed package.
- `tests/phnn/` — copied verbatim; imports (`from data.X`, `from models.X`)
  currently point at the OLD phnn layout and will FAIL until Phase 3 rewires
  them to `biophasor.core` / `biophasor.phnn`. That failure→green transition is
  the regression signal for the phnn migration.
- `tests/spectral/` — copied verbatim; `import spectralomics` fails until Phase 3
  rewires to `biophasor.spectral`. Same signal.

## Skips (phnn, 12) — carried over, not introduced
The 12 phnn skips exist in the source suite (e.g. tests requiring real-data
files or optional deps). Migration must preserve the skip count, not silently
convert skips to passes or failures.
