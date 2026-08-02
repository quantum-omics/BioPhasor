# BioPhasor Manuscript (v3) — Focus, Provenance & Version Arc

`main.tex` — *BioPhasor: Decoding Cellular State Tensors from Multi-Omics Phasor
Dynamics for Quantum Ready Systems Biology* — 93 pp, compiles clean (0 errors,
0 undefined citations/references).

## Version arc (author's plan)

- **v1** — `2-biophasor-local/arxiv/manuscript-quantum`: quantum-forward first
  version; judged not publishable.
- **v2** — `2-biophasor-local/manuscript`: findings brought to the front, quantum
  pushed to an appendix. **Seed of the future results paper.**
- **v3 (THIS)** — `BioPhasor/manuscript`: the **platform / method paper**.
  BioPhasor is presented as a phase-geometric *formulation + released software
  ecosystem*, demonstrated across nine illustrative applications. Quantum
  correspondence promoted back into the main narrative (matches the title).
- **v4 (future)** — a revision of **v2**, focused into a results paper led by
  "Tumour-Specific Cross-Modal Coupling Along the Central Dogma" as the first
  use case of the BioPhasor formulation.

## This paper's job: platform / method paper

Thesis: establish the BioPhasor formulation and release it as one reproducible
ecosystem; results are *illustrative capability demonstrations*, not a single
headline study. The platform is explicitly intended to **host many use-case
manuscripts** (v4 being the first spun out).

Key reframing edits (all narrative; no content restructuring):
- **Abstract** — platform/method is the thesis; nine illustrative applications;
  Omics-PAC demoted to one vignette with a companion-study pointer.
- **Introduction** — added a "Scope of this paper" paragraph declaring it a
  platform/method paper that hosts standalone use-case studies.
- **Results lead** — the nine scenarios framed as capability demonstrations of
  the platform, not a definitive empirical study.
- **Omics-PAC subsection** — compressed from ~113 lines / 3 figure blocks to a
  compact vignette + one full-width figure + forward pointer to the dedicated
  (v4) use-case study, to avoid collision with that paper.
- **Discussion "BioPhasor as an Ecosystem"** — extended to tie modularity to the
  use-case-manuscript roadmap.
- **Future Work** — reframed opening: engineering milestones vs. use-case studies
  the shared formulation is designed to host.

## Content provenance

All content derives from the v2 backbone (`2-biophasor-local/manuscript`),
retitled and with the quantum material promoted from appendix to a main section.
The spectral-omics fold that was briefly explored has been fully reverted — that
material belongs in its own manuscript (`7-spectral-omics/manuscript`).

## The CST -> quantum through-line (one cross-linked arc)

1. Theory §VPC-as-VQC (`subsec:vpc_vqc`) — gate algebra maps 1:1 to a VQC.
2. CST §Quantum-Information Interpretation (`subsec:cst_quantum`) — phase-coherence
   density matrix rho (Hermitian, PSD, unit-trace); PLV = |rho_jk|.
3. CST §Decoherence Limit (`subsec:cst_decoherence`) — CPTP regime where classical = quantum.
4. §Classical–Quantum Correspondence of the CST (`sec:quantum_correspondence`,
   promoted appendix->main) — state-level, density-matrix distinguishability,
   gate-level, validation. Lead paragraph converges the three threads above.
5. Discussion §The CST as a Quantum-Computable Object — the CST *is* a density
   operator; structural correspondence, no empirical advantage claimed.

## References & figures
- `references.bib` = 214 entries (backbone only; spectral entries removed with the fold).
- Figures: backbone set + the 3 CST/quantum PNGs (`cst_vpc_vqc_circuit`,
  `cst_complexity_crossover`, `cst_quantum_kernel_classification`) wired into the
  correspondence section. Orphan `spec_*` figures removed.

## Polishing pass (platform-identity)

- **Title** — made "platform" explicit: *A Phase-Geometric Platform for Decoding
  Cellular State Tensors from Multi-Omics, toward Quantum-Ready Systems Biology*
  (CST + quantum still forward).
- **Figure 1 (NEW)** — `fig_platform_overview.png`: encode -> circular statistics
  -> oscillator dynamics -> CST -> quantum-ready pipeline schematic, with the
  shared-core band and pip-install framing. Placed at the top of the Introduction
  as the paper's pitch-as-schematic; referenced in the "Scope" paragraph.
- **System Architecture (§Methods)** — expanded from a bare module table to a
  design-principles subsection (one data layer; layered core imported-never-copied;
  scikit-learn / AnnData standard interfaces; reproducibility as a build target).
  Module table updated to add the first-class CST subpackage and mark Core as
  shared. Grounded in the real package (193 tests, Makefile + lockfile + reproduce.sh).
