# v2 Manuscript — Use-Case Deep-Dive Restructure

`main.tex` — *Tumour-Specific Cross-Modal Phase--Amplitude Coupling Along the
Central Dogma in Matched Multi-Omics* — 25 pp,
compiles clean (0 errors, 0 undefined citations/references, 52 cited entries).

## Title

Retitled to lead with the biological finding, not the framework: the method
name (BioPhasor) is credited in the abstract's opening sentences and throughout,
not advertised in the title, since the framework itself is the method paper's
contribution.

## Role in the version arc

This is **v2**, reshaped into the **use-case deep-dive** paper for the
tumour-specific central-dogma cross-modal coupling. It leans on the BioPhasor
**method/platform paper** (v3, `BioPhasor/manuscript`) for the framework rather
than re-deriving it. An untouched arxiv backup of the pre-restructure v2 is kept
at `arxiv/manuscript-local/` (md5 8d5c6b57e404d45c2ef653eba0a56405) and was NOT
edited.

## What was removed (now lives in the method paper / backup)

- **Framework re-derivation** — the full Theory, BioPhasor Formulation, CST
  derivation, and Methods/architecture (~1,255 lines) were compressed into one
  compact "BioPhasor in Brief" section that states only what the coupling use case
  needs and cites the method paper (`biophasor_platform`) for all derivations.
- **Eight non-coupling result scenarios** — encoding/coherence selection,
  cell-cycle, circadian, Kuramoto, multi-omics fusion, head-to-head benchmark,
  hard clinical tasks, phasor classification, manifold geometry, generic CST
  dynamics/temporal/tensor-network, and attractor/Floquet.
- **Classical–Quantum Correspondence appendix** — belongs in the method paper.
- **Roadmap-style Future Work** (11 subsections) and the framework-wide Discussion
  (14 subsections) — folded into a focused 4-part Discussion.

## What was retained / expanded

- **Results = two subsections**: the pathway atlas (supporting, roots the
  regulatory axis) and the **central-dogma coupling deep dive** (the main use
  case), expanded from 113 lines to a structured 7-part treatment: measurement
  definition, effect + surrogate null, genome-wide/FDR structure,
  regulatory-module structure, tumour-specificity, symmetry/directionality, and
  effect-size scoping.
- **Headline numbers (unchanged, verbatim from existing results)**: aggregate MI
  0.0177 vs null 0.0077, z=180, p≈1e-4, ~2.3× null; pooled circular–linear
  r=0.295 (95% CI [0.291,0.299]); 4,988/7,083 genes = 70.4% at BH-FDR<0.05
  (raw 73.1%); tumour r=0.295 vs normal r=0.029; reverse coupling r=0.285
  (ratio 1.04, symmetric).
- **Figures**: 7 used — 3 pathway-atlas (spectrum, CP error, coherence) + 4
  coupling (fig1_omics_pac composite, comodulogram, null, ranking).
- Abstract, introduction (contributions), discussion, and conclusion rewritten
  around the single use case, all pointing to the method paper for the framework.

## Future use cases in the series
Spectral connectome (`7-spectral-omics`) and port-Hamiltonian dynamics
(`3-phnn-omics`) are separate use-case manuscripts on the same shared formulation.
