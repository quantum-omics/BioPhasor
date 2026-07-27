# Contributing to the BioPhasor platform

BioPhasor is a **platform**: one shared scientific core plus per-manuscript
experiments. The single most important rule is *do not duplicate code or data*.
This guide explains where each kind of contribution belongs and how a new
manuscript plugs in.

---

## Getting started

```bash
git clone <repo>/BioPhasor && cd BioPhasor
make env                      # venv + pinned deps + editable install
# or: python -m venv .venv && .venv/bin/pip install -e ".[dev,experiments]"
.venv/bin/python -m pytest tests/ -q      # expect: 181 passed, 12 skipped
```

---

## The decision rule: core vs domain vs experiment

Before writing code, decide where it lives. Ask **"could a second manuscript
ever reuse this?"**

| If it is… | it goes in… | example |
|---|---|---|
| Reusable by ANY domain | `biophasor.core.*` | phasor encoding, circular stats, graph builders, data generators, losses |
| Specific to one domain's science, but reused across that domain's manuscripts | `biophasor.<domain>.*` (`phnn`, `spectral`, `cst`, `dynamics`, `integration`) | a pHNN model, a spectral connectome operator |
| One-off analysis / figure / number for a single manuscript | `experiments/<manuscript>/codes/` | an experiment script that *imports* the above |

**Never paste a reusable function into an experiment script.** If you find
yourself copying, stop and promote it to `core` (or the domain subpackage).
If two domains implement the same idea, reconcile them into one `core` function
(see `docs/integration_map.md` for how the initial reconciliation was done).

---

## Data: stored once, referenced by accession

- **Public datasets used by >1 manuscript** → `experiments/_shared/data/raw/<accession>/`
  (gitignored, rebuilt on demand). Reach them via `experiments/_shared/common.py`
  (`common.CACHE`, `common.load_expression(acc)`), never by copying files.
- **Manuscript-specific derived data** → that manuscript's own `data/` or `results/`.

Never commit a raw dataset into a manuscript folder. If a second manuscript
needs it, it is by definition shared.

---

## Adding a manuscript (the plug-in model)

1. **Create the folder:**
   ```bash
   mkdir -p experiments/<name>/{codes,results,figures,reports}
   ```
2. **Import shared science** from the package; **import experiment glue** from
   the shared helper:
   ```python
   from biophasor.core.encoder import tanh_phase_encode
   from biophasor.core.graph import build_biological_graph
   import sys; sys.path.insert(0, "<repo>/BioPhasor/experiments/_shared")
   import common                       # common.CACHE, common.exp_dir("<name>")
   ```
3. **Reference datasets by accession** through `common`; if the dataset is new,
   add its fetch/cache entry to `experiments/_shared/common.py` (do not hardcode
   a path).
4. **Write reusable functions into the package, not the experiment.** If your
   manuscript needs a new model or operator others could use, add it to the
   appropriate `biophasor.<domain>` subpackage (and register it in
   `pyproject.toml` `packages` if it's a new subpackage), then import it.
5. **Register the manuscript** in `reproduce.sh` (add a `case` branch) and, if
   desired, a `repro-<name>` target in the `Makefile`.
6. **Add it to the tables** in `README.md` and `experiments/README.md`.

A new *domain* (a whole new sibling project joining later) additionally gets a
`biophasor.<domain>` subpackage: reconcile its shared pieces into `core` first
(record decisions in `docs/integration_map.md`), copy its test suite into
`tests/<domain>/`, and rewire imports to `biophasor.*` — the same procedure
used for `phnn` and `spectral`.

---

## Development workflow

- **Tests:** `make test` (full unified suite). New package code needs tests in
  `tests/`, `tests/phnn/`, or `tests/spectral/`. The baseline is **181 passed,
  12 skipped** — do not reduce it or convert the 12 pre-existing skips.
- **Style:** `black` (line length 100), `ruff`, optional `mypy`.
- **Exports:** update the relevant `__init__.py` when adding public symbols.
- **New subpackage:** add it to `pyproject.toml` `[tool.setuptools] packages`
  and reinstall (`pip install -e .`).

## Pull requests

- One feature/fix per PR; conventional-commit titles (`feat:`, `fix:`, `docs:`, `test:`).
- All tests green; no new duplication (a shared implementation must have exactly
  one home — `docs/VERIFICATION.md` shows the duplication scan used to certify this).

## License and sign-off

BioPhasor is licensed under the **Apache License 2.0** (see `LICENSE`).

Under Apache-2.0 § 5, any contribution you intentionally submit for inclusion is
licensed under those same terms, unless you state otherwise explicitly. There is
no separate CLA to sign.

We do ask for a **Developer Certificate of Origin** sign-off on each commit —
you assert that you wrote the contribution or have the right to submit it under
the project licence:

```bash
git commit -s -m "feat: your change"     # appends the Signed-off-by line
```

If your contribution includes code you did not write, say so in the PR and name
its licence — we can only accept code under Apache-2.0-compatible terms (MIT,
BSD, ISC, Apache-2.0). GPL-licensed code cannot be merged.

If you add a new source file, start it with:

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2026 Quantum Omics Foundation
```
