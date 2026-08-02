# Contributing to the BioPhasor platform

BioPhasor is a **platform**: one shared package plus per-manuscript experiment
suites. The single most important rule is *do not duplicate code or data*. This
guide says where each kind of contribution belongs.

That rule is not abstract here. Earlier generations of this tree carried the
same figure library in several places and the same analysis under two import
roots. Duplicates drift, and a
fix then has to be made twice or it is made once and silently disagrees with
the other copy.

---

## Getting started

```bash
git clone <repo>/BioPhasor && cd BioPhasor
make env                      # venv + pinned deps + editable install
# or: python -m venv .venv && .venv/bin/pip install -e ".[dev,experiments]"
.venv/bin/python -m pytest tests/ -q      # expect: 171 passed
```

Use `.venv/bin/python`. A system interpreter will not have `anndata`, and the
failure surfaces as an import error a long way from its cause.

---

## The decision rule: package, plumbing, or experiment

Before writing code, ask **"could a second manuscript ever reuse this?"**

| If it is… | it goes in… | example |
|---|---|---|
| Reusable by any manuscript | `biophasor.core` | phasor encoding, circular statistics, manifold geometry, circular losses, pathway sets |
| Specific to one domain's science, reused across that domain | `biophasor.<domain>` (`spectral`, `cst`, `dynamics`, `integration`, `ml`, `transform`) | an OCM operator, a compartment model, a Kuramoto variant |
| Experiment plumbing shared across suites | `experiments/_shared/common.py` | output paths, the GEO fetch, seeding, the preprocessing gates |
| One-off analysis, figure or number for a single manuscript | `experiments/<suite>/codes/` | a script that *imports* the above and writes a results JSON |

**Never paste a reusable function into an experiment script.** If you are
copying, stop and promote it. If two suites implement the same idea, reconcile
them into one function in the package.

Reusable *science* never lives in `experiments/_shared/`. That module is glue —
paths, the data cache, provenance stamps, preprocessing gates. If a function
there would interest someone who is not running an experiment, it belongs in
the package. The converse also holds: `results_dir()`, `manuscript_figs()` and
`save_fig()` are not science and do not belong in `biophasor`.

A new subpackage needs nothing added to `pyproject.toml` — packages are
discovered, not listed. **Non-Python files are a different matter:** anything
that is not a `.py` must be declared in `[tool.setuptools.package-data]` or it
will be missing from a wheel while remaining present in your editable install,
which is how a shipped data file silently becomes a synthetic fallback. See the
comment on that block.

---

## Data: stored once, referenced by accession

- Raw measurements live in **one** cache, `experiments/_shared/data/raw/`,
  keyed by accession and reached through `common.CACHE` — never by a
  hard-coded path. It is gitignored and rebuilt on demand.
- Derived per-suite outputs go in that suite's `results/`, which **is**
  tracked: those are the receipts the manuscript quotes.
- Figures are written once, into the directory that manuscript's `main.tex`
  resolves `\includegraphics` against, which `common.manuscript_figs(suite)`
  knows per suite. There is no suite-local `figures/`; the previous
  arrangement kept a byte-identical copy in both places and they drifted the
  moment one was regenerated alone.
- **Never commit a raw dataset or a model checkpoint.** If a second suite needs
  a dataset, it is by definition shared.

Adding a data source means adding its fetch/cache entry to
`experiments/_shared/common.py` and a provenance block in
`experiments/_shared/data/PROVENANCE.md` — accession, citation, processing
steps, and any sampling limitation that constrains what can be concluded.

---

## Every number traces to a results file

A manuscript's numerics section claims its numbers come from `results/*.json`.
The **number guard** enforces it: each numeric literal in that section must
round-trip to a stored value *at the precision written*. Rounding is part of the
claim — `0.480` is correct where the source says `0.48046875`; `0.481` is a
defect. A suite's `codes/check_numbers.py` supplies its section markers and
whitelist; the guard algorithm is shared, in `biophasor.utils.number_guard`.

Three of the four suites carry a guard: `tumor`, `spectral-classical` and
`spectral-quantum`. The `biophasor` suite does not yet, which means the
platform manuscript — the longest of the four, and the one quoting the
benchmark numbers currently affected by the SNF break — is the one manuscript
whose literals are unchecked. Adding it is the highest-value contribution to
this area.

When the guard reports an unmatched literal, **the fix is almost never to widen
the whitelist.** In practice it is one of:

- **The value is computed but never stored.** Write it to the JSON.
- **The text quotes a derived form** — a percentage of a stored fraction, or the
  magnitude of a signed value. Store the derived form too; do not make the
  reader perform the conversion mentally.
- **The number genuinely drifted.** Update the manuscript.

The whitelist is for values that are not measurements at all: structural counts,
class labels, model dimensions, figure panel numbers. Keep it short and
explicit. An over-broad whitelist silently disables the guard, which is worse
than not having one — a guard that passes because it checks nothing is a claim
of verification you cannot support.

If a number cannot be reproduced, say so where a reader will find it rather than
whitelisting it away. `CHANGELOG.md` under Known issues, the suite's
`README.md`, and `experiments/spectral-classical/results/ORPHANED.md` are the
existing precedent for how that is recorded.

---

## Adding a manuscript

1. Create the folders:
   ```bash
   mkdir -p experiments/<name>/{codes,results,reports}
   mkdir -p manuscripts/<name>
   ```
2. Import shared science from the package and plumbing from the helper — no
   `sys.path` manipulation anywhere:
   ```python
   from biophasor.core.operators import coherence
   from experiments._shared import common
   ```
3. Register the suite's figure directory in `_MANUSCRIPT_FIGDIR` in
   `experiments/_shared/common.py`, read off that manuscript's
   `\graphicspath`.
4. Write `codes/run_all.py` with `--list` and `--check`, listing the scripts in
   order and ending with `check_numbers.py`.
5. Write `codes/check_numbers.py` as a guard config — section markers and
   whitelist only.
6. Add a `repro-<name>` target to the `Makefile`, a case in `reproduce.sh`, and
   a line to `make check-suites`.
7. Add it to the tables in `README.md`, `experiments/README.md` and
   `REPRODUCE.md`, including its runtime.

---

## Development workflow

- **Tests:** `make test`. The baseline is **171 passed** — do not reduce it.
  New package code needs tests in `tests/` or `tests/spectral/`.
- **Style:** `black` and `ruff` at line length 100; `mypy` on the package.
- **Exports:** update the relevant `__init__.py` when adding public symbols.
- **Docs:** a new public module needs a `docs/` chapter and an entry in the
  `mkdocs.yml` nav, or it is undiscoverable.

## Style

- Docstrings say what a thing is **for** and name the trap it avoids, not just
  what its arguments are. `omics_spectrum.py` is the model: it documents the
  resolution order and states plainly what goes wrong when the shipped ladder
  is missing.
- Comments explain **why**, not what. A comment restating the code is noise; a
  comment recording why a pin is `<1.7` rather than unbounded is worth keeping.
- Register of a methods section: one idea per sentence, no editorialising
  adjectives, no emoji.

### The tone rule

BioPhasor **fits phasor geometry to omics measurements**. Write "the phasor
representation of these data", never "the cell is a phasor". The quantum layer
is a quantum-simulable formalism with an exact classical correspondence: write
that, never "the cell is a quantum system" and never any claim of quantum
advantage. Reviewers accept the first form and reject the second, and the code
should not teach the wrong habit.

## Pull requests

- Open an issue before substantial work.
- One concern per PR; conventional-commit titles (`feat:`, `fix:`, `docs:`,
  `test:`).
- Run `pytest` and the affected suite's `run_all.py` before submitting; a green
  guard is part of a green build. Use `--check` first if the suite is slow —
  `tumor` is about 24 minutes.
- No new duplication: a shared implementation has exactly one home.

## Licence and sign-off

BioPhasor is licensed under the **Apache License 2.0** (see `LICENSE`).

Under Apache-2.0 § 5, any contribution you intentionally submit for inclusion is
licensed under those same terms unless you state otherwise explicitly. There is
no separate CLA to sign.

We do ask for a **Developer Certificate of Origin** sign-off on each commit —
you assert that you wrote the contribution or have the right to submit it under
the project licence:

```bash
git commit -s -m "feat: your change"     # appends the Signed-off-by line
```

If your contribution includes code you did not write, say so in the PR and name
its licence — we can only accept Apache-2.0-compatible terms (MIT, BSD, ISC,
Apache-2.0). GPL-licensed code cannot be merged.

New source files start with:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Quantum Omics Foundation
```
