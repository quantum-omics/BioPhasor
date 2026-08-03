# experiments/ — one suite per manuscript

Each suite is a thin layer over the installed `biophasor` package: it runs the
science, writes its numbers to `results/`, and regenerates its figures into the
manuscript that prints them.

| suite | manuscript | data | driver |
|---|---|---|---|
| `biophasor` | `manuscripts/biophasor` — the platform/method paper | GSE293316, GSE171432, CPTAC UCEC | `codes/run_all.py` |
| `tumor` | `manuscripts/tumor` — cross-modal PAC use case | CPTAC UCEC | `codes/run_all.py` |
| `spectral-classical` | `manuscripts/spectral-classical` — Spectral-Omics | GSE10072, GSE11923 | `codes/run_all.py` |
| `spectral-quantum` | `manuscripts/spectral-quantum` — the quantum paper | none (package data) | `codes/run_all.py` |

## Running a suite

Every driver needs the repository root on `sys.path`, because the suites import
`experiments._shared`:

```bash
# biophasor and tumor: importable package names, so -m works
PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.run_all
PYTHONPATH=. .venv/bin/python -m experiments.tumor.codes.run_all

# spectral-classical and spectral-quantum: the directory names contain a
# hyphen, so run the file directly (it reaches its own modules through runpy,
# which accepts the hyphen)
PYTHONPATH=. .venv/bin/python experiments/spectral-classical/codes/run_all.py
PYTHONPATH=. .venv/bin/python experiments/spectral-quantum/codes/run_all.py
```

`--list` on any driver prints what it will run, in order, and — for
`spectral-classical` and `tumor` — what it explicitly cannot run.

`--check` on any driver is a **dry run**: it prints the ordered plan,
byte-compiles each script's source *without importing it* (several experiment
modules load their data at import time, which is the expensive work this flag
exists to avoid), confirms every input it needs is on disk, and confirms the
results and manuscript-figure directories are writable — then exits without
executing anything. It also restates what that suite cannot regenerate. Use it
before committing to a multi-minute run. `--check` respects the other flags:
`--check --dmrg` reports that the recompute would fail if `physics-tenpy` is
absent, and exits non-zero rather than claiming OK.

Two steps are behind flags because they are slow, and neither is on the default
path: `spectral-classical --with-benchmark` (repeated CV + learning curve,
minutes) and `spectral-quantum --dmrg` (recompute the DMRG scan, needs TeNPy;
without it the cached scan is used and the output says "CACHED").

## Shared, not copied

`_shared/` holds what more than one suite needs and nothing any single suite
owns:

- `_shared/data/raw/` — the ONE measurement cache, keyed by accession, reached
  through `common.CACHE`, never by a hard-coded path. Gitignored. Provenance,
  citations and per-dataset limitations in `_shared/data/PROVENANCE.md`.
- `_shared/common.py` — the output-path convention (`results_dir(suite)`,
  `manuscript_figs(suite)`, `save_fig`), the GEO fetch, and the preprocessing
  gates (top-variable selection, cosinor rhythmicity, probe→symbol mapping).
- `_shared/figstyle.py` — the publication figure style.
- `_shared/revision_data.py` — the matched CPTAC UCEC complete-case matrix
  (7,083 genes × 109 samples) that the `biophasor` benchmark scripts and the
  `tumor` hardened statistics both run on, so their numbers are comparable by
  construction.

Reusable *science* does not live in `_shared/`. It lives in the package. If a
function there would interest someone who is not running an experiment, it
belongs in `biophasor`.

## Suite layout

    <suite>/
    ├── codes/      experiment scripts + run_all.py
    ├── results/    the numbers the manuscript quotes (tracked)
    └── reports/    methods notes, run logs

There is no suite-local `figures/`. **Result figures are written once, into
`manuscripts/<name>/…`** — the directory that manuscript's `main.tex` actually
resolves `\includegraphics` against, which `common.manuscript_figs(suite)`
knows per suite (`manuscripts/biophasor/` and `manuscripts/tumor/` declare
`\graphicspath{{./}}`; the two spectral papers declare `{{fig/}}`). The
A figure is written once, into the manuscript that prints it. Keeping a second
copy in a suite-local `figures/` drifts the moment one is regenerated alone.

## No sys.path manipulation

No script inserts anything on `sys.path`. Scripts import the installed package
(`from biophasor… import …`) and the shared plumbing
(`from experiments._shared import common`). Run the drivers with `PYTHONPATH=.`
from the repository root, as above.

## What cannot be reproduced

Recorded rather than papered over — each driver says so at the end of its run:

- **`spectral-classical`**: three result sets (`circadian_rigor`,
  `directed_flux`, `enrichment`) exist as figures + JSON with no generating code
  in this repository *or* in the `1-Spectral` source tree. Two of the eight
  figures `main.tex` prints come from that set, so its figure deck cannot be
  fully rebuilt. Details: `spectral-classical/results/ORPHANED.md`.
- **`tumor`**: `fig1_omics_pac.png` — the four-panel Figure 1 — has no
  generating script anywhere available. Details: `tumor/README.md`.
- **`biophasor`**: `fig_platform_overview.png`, `fig2_benchmark.png` and
  `fig3_hardtask_roc.png` are printed by `manuscripts/biophasor/main.tex` and
  likewise have no generator; the experiments that supply their *numbers*
  (exp12, exp12b, exp13) do run and rewrite their results JSON.

