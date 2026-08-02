# BioPhasor Platform Manuscript — Integration Blueprint

**Target:** `BioPhasor/manuscript/main.tex`
**New title:** *BioPhasor: Decoding Cellular State Tensors from Multi-Omics Phasor
Dynamics for Quantum Ready Systems Biology*

Two sources fuse into one integrative platform paper:
- **BACKBONE** = `2-biophasor-local/manuscript` (recent, 3636 lines, 214 refs, 55 figs)
- **EXPANSION** = `7-spectral-omics/manuscript` (spectral CST variables, 730 lines, 31 refs, 12 figs)

The guiding principle: keep **everything from the recent paper**, **fold the spectral
framework in as an expanded CST regulatory/spectral layer** (not a separate paper),
and **promote the quantum material from appendix into the main narrative** to match
the quantum-forward title. Major focus threaded throughout: *decoding the CST from
multi-omics data* + *quantum-computing integration*.

---

## The core unification insight

The recent paper's **phase-coherence density matrix**
  ρ = (1/M) Σ_m z_m z_m†,  (z_m)_u = e^{iθ_u}   [Hermitian, PSD, unit-trace]
and spectral's **Omics Connectome Matrix**
  H = diag(ψ) C diag(ψ)†,  ψ_i = r_i e^{iθ_i}   [Hermitian, gauge-invariant]
are the **same class of Hermitian ψψ*-type operator**. ρ is the sample-averaged
outer product of pure-phase vectors; H is the coupling-weighted, amplitude-carrying
single-slice version. This is the seam along which the two papers join:

- ρ (recent) → the *quantum-information* reading of the CST (density operator, PLV
  off-diagonals, von Neumann entropy).
- H (spectral) → the *spectral/graph* reading of the CST regulatory axis (harmonics =
  collective normal modes, spectral indicators, compartment coupling).

Both diagonalise a Hermitian phasor operator; we present them as two faces of one
CST object and unify notation (see below).

---

## Notation unification

| Concept | Recent (backbone) | Spectral | Unified in new ms |
|---|---|---|---|
| phasor | z = e^{iφ} (unit) / A e^{iφ} | ψ = r e^{iθ} | ψ = r e^{iθ}; z = e^{iθ} the pure-phase special case |
| phase encoding | π·tanh((log(1+x)−μ)/σ) | π·tanh((log(1+x)−μ)/(σ+ε)) | identical — state once in Theory, cite both |
| order parameter | R (coherence) | R (Kuramoto) | R — one definition |
| Hermitian operator | ρ (density matrix) | H (OCM) | keep both symbols; state the ρ↔H relationship explicitly |
| harmonics | (not present) | φ_n eigenvectors | adopt spectral's φ_n; rename to avoid clash with phase φ → use θ for phase everywhere, φ_n for harmonics |

**Notation fix:** backbone uses φ for phase; spectral uses θ for phase and φ_n for
harmonics. To avoid collision in the merged doc, **phase = θ**, **harmonic modes = φ_n**.
The abstract/intro of the backbone use φ for phase in a few places — sweep those to θ,
OR (lower-risk) keep backbone's φ-as-phase within backbone-derived sections and use
spectral's θ/φ_n only inside the new expanded-CST section, with a one-line notation
bridge. **Decision: low-risk path** — localise spectral notation to the new section,
add an explicit bridge sentence ("we write the phasor phase as θ in this section,
equivalent to φ elsewhere").

---

## Target section outline (source → target map)

| # | Target section | Source | Action |
|---|---|---|---|
| — | Title / authors / abstract | both | REWRITE (Phase 4): quantum-forward title; abstract adds expanded-CST + quantum |
| 1 | Introduction | backbone §Intro | KEEP; add 1 para on spectral connectome/harmonics as CST expansion + strengthen quantum-integration thread |
| 2 | Related Work | backbone §Related + spectral intro refs | MERGE; add spectral-graph/Hermitian-adjacency line (guo2017hermitian, chung1997, alter2000svd, langfelder2008) |
| 3 | Theory: Phasor Dynamics on N-torus | backbone §Theory | KEEP verbatim (encoding, circular stats, Riemannian, VPC, **VPC→VQC promoted-emphasis**, Kuramoto, BPT, PLV, fusion, dissipative, bifurcations) |
| 4 | Biophasor Formulation | backbone §Formulation | KEEP verbatim |
| 5 | **Cell State Tensor (expanded)** | backbone §CST + **spectral §Theory** | **FUSE — the heart of the integration.** See CST sub-map below |
| 6 | Methods | backbone §Methods + spectral §Methods (coupling est., compartment assignment, datasets) | MERGE methods; add spectral coupling/compartment/circadian-quantification subsections |
| 7 | Results | backbone §Results + spectral §Results | KEEP backbone results; ADD spectral results subsections (consistency at machine precision; spectral tumour/normal structure; circadian harmonics) |
| 8 | Discussion | backbone §Discussion + spectral §Discussion | MERGE into one integrative discussion |
| 9 | Conclusion | backbone | KEEP, broaden to platform |
| A | Appendix | backbone appendix | KEEP quantum-validation detail; but PROMOTE the core classical↔quantum correspondence + ρ density matrix + VPC→VQC into main §5/§3 |

### CST section sub-map (§5 — the fusion core)

| Target subsection | Source | Notes |
|---|---|---|
| 5.1 Definition (rank-3 R×T×H) | backbone §CST Definition | KEEP |
| 5.2 Measured instantiation of axes | backbone §cst_measured | KEEP (pathway atlas R, central-dogma modality, sample/time axis) |
| 5.3 **Expanded regulatory axis: the Omics Connectome** | **spectral §theory-ocm** | NEW — H = diag(ψ)C diag(ψ)†; Prop (amplitude weighting); gauge invariance |
| 5.4 **Omics harmonics (collective normal modes)** | **spectral §theory-harmonics** | NEW — eigendecomposition, rank-k truncation, spectral energy |
| 5.5 **Spectral indicators** | **spectral §theory-indicators** | NEW — spectral entropy, gap Δ_F, participation ratio, localisation |
| 5.6 **Compartment Coupling Matrix** | **spectral §theory-ost** | NEW — 5×5 Hermitian compartment contraction, compartment weights, κ |
| 5.7 **Spectral state classes + state record** | **spectral §theory-taxonomy** | NEW — 7-class table, JSON state record (honest: descriptive not validated) |
| 5.8 Construction from attractor geometry | backbone | KEEP |
| 5.9 Tensor-network factorization | backbone | KEEP |
| 5.10 Temporal update / uncertainty-aware CST | backbone | KEEP |
| 5.11 **Quantum-information interpretation (PROMOTED)** | backbone §cst_quantum + spectral §theory-quantum | FUSE — ρ density matrix ↔ H OCM as one Hermitian phasor operator; Rz/CNOT/QFT gate correspondence; classical↔quantum CST proposition |
| 5.12 Floquet / Markov / decoherence limit | backbone | KEEP |
| 5.13 CST axis semantics | backbone | KEEP |
| 5.14 Consistency suite (of the spectral construction) | spectral §theory-consistency | NEW — Hermiticity/real-spectrum/gauge machine-precision invariants |

---

## Figures

- **Backbone 55 figures** — copy all, keep filenames.
- **Spectral figures** to wire into the expanded-CST + spectral-results sections
  (copy from `7-spectral-omics/manuscript/fig/`, namespace with `spec_` prefix to
  avoid any collision):
  - `cancer_omicatome.png`, `cancer_ost_heatmap.png` → OCM / compartment coupling
  - `cancer_indicator_separation.png`, `bench_auroc_comparison.png` → spectral indicators / tumour-normal
  - `circadian_leading_harmonic_fit.png`, `circ2_phase_lag.png`, `circ2_spectral_vs_mean.png`, `circadian_indicator_dashboard.png` → circadian harmonics
  - `enrich_harmonic_enrichment.png`, `enrich_clock_loading.png` → harmonic enrichment
  - `flux_spectrum_dependence.png` → magnetic-OCM flux variant
  - `workflow.png` → optional platform workflow overview
- **Already in target folder** (keep): `cst_complexity_crossover.png`,
  `cst_quantum_kernel_classification.png`, `cst_vpc_vqc_circuit.png` → the promoted
  quantum §5.11 / §3 VPC→VQC.

---

## References

- Start from backbone `references.bib` (214). Merge spectral's 31, de-dup by DOI.
- Spectral-specific keys to bring in (used by new §5 material): kuramoto1975,
  langfelder2008wgcna, guo2017hermitian, chung1997spectral, alter2000svd,
  newman2006modularity, barabasi2011network, nielsen2010quantum (check if backbone
  already has nielsen/kuramoto → reuse, don't duplicate).

## Compile
Local TeX: `export PATH="/Library/TeX/texbin:$PATH"` then pdflatex→bibtex→pdflatex×2.
Backbone already compiles clean, so breakage will come only from the merged bib keys
and the new spectral figure paths — fix those to reach 0 undefined.

---

## VERIFIED reference-merge facts (checked against both bibs)

Spectral has **31** keys. Overlap check vs backbone:
- **kuramoto1975** — backbone already cites Kuramoto (2 matches). Reuse backbone's key; map spectral `\cite{kuramoto1975}` → backbone's Kuramoto key (or keep both if DOIs differ; de-dup by DOI at merge).
- **nielsen2010quantum** — backbone already cites Nielsen (1 match). Reuse backbone key; remap.
- **All 29 others are NEW to the backbone** and must be added, notably the graph-spectral core for the expanded CST: `guo2017hermitian`, `chung1997spectral`, `alter2000svd`, `langfelder2008wgcna`, `newman2006modularity`, `barabasi2011network`, plus magnetic-Laplacian refs (`fanuel2017/2018magnetic`, `zhang2021magnet`, `he2022msgnn`), graph-signal (`shuman2013graph`, `furutani2019graph`, `vonluxburg2007tutorial`), connectome (`atasoy2016connectome`), circadian/rhythm callers (`hughes2009harmonics`, `hughes2010jtk`, `cornelissen2014cosinor`, `zhang2014circadian`, `oneill2011circadian`, `vanderplas2018lombscargle`), phasor-imaging (`digman2008phasor`, `stringari2011cell`, `ranjit2018fitfree`), and misc (`eisen1998cluster`, `spellman1998cellcycle`, `landi2008gene`, `haghverdi2015diffusion`, `edgar2002geo`, `bottcher2024complex`).

**Merge rule:** append spectral entries to backbone `references.bib`, DROP the 2
duplicates (kuramoto, nielsen) and rewrite their `\cite` calls in spectral-derived
text to the backbone keys, keep the other 29. Net expected: 214 + 29 = ~243 entries.

**Snippets:** spectral `main.tex` uses `\input{snippets/flux_}` (the magnetic-OCM flux
variant). Either inline that snippet into §5.3 or copy `7-spectral-omics/manuscript/snippets/`
into the target. Decision: inline the flux paragraph into the OCM subsection (keep the
target folder flat, no snippets/ dependency).

---

## CORRECTION (Phase 2 execution — supersedes the merge estimate above)

A rigorous DOI+title+key dedup found the two bibs are **completely disjoint**:
**0 true duplicates** (no shared DOI, no shared title, no key collision). The
earlier "kuramoto/nielsen already in backbone" note was a false-positive substring
match — the backbone's only "Nielsen" is a metagenomics paper (`ref_145`), and its
Kuramoto is `ref_kuramoto1984` (1984 book), distinct from spectral's `kuramoto1975`
incollection. **All 31 spectral entries were appended verbatim with original keys.**
Final `references.bib` = **245 entries** (214 + 31). No citation remapping needed.
