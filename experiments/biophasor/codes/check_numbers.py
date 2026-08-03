#!/usr/bin/env python3
"""
check_numbers.py — the manuscript-number guard for the `biophasor` suite.

Every numeric literal in the manuscript's Results section must round-trip to a
value in ``experiments/biophasor/results/*.json`` at the precision written. The
algorithm lives in ``biophasor.utils.number_guard`` so all four suites enforce
the same contract; only the section markers and the whitelist are
suite-specific.

    PYTHONPATH=. .venv/bin/python experiments/biophasor/codes/check_numbers.py

Exit 1 on any unmatched literal, printing the nearest candidates.

The bound runs from ``\\section{Results}`` to the ``Data Sources and
Reproducibility`` subsection that closes it, NOT to the next ``\\section``.
Everything between those two markers is a measured vignette. Data Sources is
provenance prose — accessions, cell counts as described by the depositor,
software versions — and guarding it would force a dozen whitelist entries that
are not measurements, which is how a whitelist stops being a guard. The
``Classical--Quantum Correspondence`` section that follows is derivation.
"""
from __future__ import annotations

import sys
from pathlib import Path

from biophasor.utils.number_guard import GuardConfig, run_guard

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent                              # experiments/biophasor
ROOT = SUITE.parents[1]                          # repository root
MANUSCRIPT = ROOT / "manuscripts" / "biophasor" / "main.tex"

#: Values that are not measurements. Deliberately short and explicit — an
#: over-broad whitelist silently disables the guard.
WHITELIST = {
    # Small structural counts, axis and mode indices, and figure-panel numbers.
    # The CST is 3-way; the modality axis has 2 entries; the nine scenarios of
    # the results table are counted, not measured.
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    # Model and encoder dimensions chosen by configuration, not measured:
    # phase-bin counts, latent widths, epoch and fold counts, the 24 h period,
    # the top-N gene cutoffs the selection protocols declare.
    12.0, 16.0, 20.0, 24.0, 32.0, 50.0, 64.0, 100.0, 128.0, 150.0, 200.0,
    300.0, 500.0, 1000.0, 2000.0,
    # A literature citation year written in prose rather than \cite{} —
    # "Hart et al.\ 2015 core-essential" — which the LaTeX strip cannot see.
    # Significance and coherence thresholds written as thresholds, not as
    # attained values: "$\\mathcal{C} > 0.30$", "FDR $<0.05$", "$p<0.01$".
    0.30, 0.05, 0.01,
    2015.0,
    # GEO accession digits that survive the LaTeX strip.
    293316.0, 171432.0,
}

#: Two results directories. The Results section carries an "Illustrative
#: Application" subsection on the central-dogma PAC use case, which is
#: developed in full in the tumour manuscript and whose receipts are written by
#: ``experiments/tumor/`` — the two papers quote the same measurement.
RESULTS = [
    SUITE / "results",
    SUITE.parent / "tumor" / "results",
]

CONFIG = GuardConfig(
    results_dir=RESULTS,
    tex=MANUSCRIPT,
    start=r"\section{Results}",
    end=r"\subsection{Data Sources and Reproducibility}",
    whitelist=WHITELIST,
)

if __name__ == "__main__":
    sys.exit(run_guard(CONFIG))
