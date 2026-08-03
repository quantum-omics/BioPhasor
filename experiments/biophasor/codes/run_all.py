#!/usr/bin/env python3
"""
run_all.py — regenerate every result and figure of the `biophasor` suite.

    python -m experiments.biophasor.codes.run_all           everything
    python -m experiments.biophasor.codes.run_all --list    print the order and exit
    python -m experiments.biophasor.codes.run_all --check   validate inputs + the
                                                            plan, run NOTHING
    python -m experiments.biophasor.codes.run_all exp01 exp07   a subset (prefix match)

Run from the repository root, with the repository root on ``sys.path``:

    PYTHONPATH=. .venv/bin/python -m experiments.biophasor.codes.run_all

The suite imports the installed ``biophasor`` package for all science and
``experiments._shared`` for plumbing; no script manipulates ``sys.path``.
Numbers land in ``experiments/biophasor/results/``, figures in
``manuscripts/biophasor/`` (once — the ``.tex`` declares
``\\graphicspath{{./}}``, so that directory *is* the figure directory).

This suite belongs to the platform/method manuscript. The tumour-specific
Omics-PAC scripts live in ``experiments/tumor/``.

Data: GSE293316, GSE171432 and the CPTAC UCEC cache, all read from the shared
cache via ``experiments._shared.common``. See
``experiments/_shared/data/PROVENANCE.md``. If the cache is missing, the driver
fails loudly on the first script that needs it rather than skipping quietly.

The last step of every full run is ``check_numbers``, the manuscript-number
guard: it re-reads ``results/*.json`` and requires every numeric literal in the
manuscript's guarded section to round-trip at the precision written. It runs
last because it checks what the run just produced, and it reads files rather
than computing, so it costs nothing. A non-zero exit means a quoted number no
longer traces to a receipt.
"""
from __future__ import annotations

import os
import runpy
import sys
import time
import traceback

from experiments._shared import common

HERE = os.path.dirname(os.path.abspath(__file__))
SUITE = "biophasor"

# (module, one-line description). Ordered so that a script's dependencies run
# first: exp04_attractor imports the co-expression graph built by exp03_kuramoto,
# and exp12b refines the benchmark table exp12 writes.
ORDER = [
    ("exp03_encoding_coherence",      "phasor encoding + coherence gene selection (GSE293316)"),
    ("coherence_axis_fix_run",        "coherence rescoped to a supported axis (metacell / detection-filtered)"),
    ("exp01_cellcycle_assignment",    "cell-cycle phase assignment vs an algorithmic reference (GSE293316)"),
    ("exp02_circadian_rhythm",        "circadian rhythmicity, ZT0-20 WT liver (GSE171432)"),
    ("exp03_kuramoto_synchrony",      "Kuramoto synchrony on the real co-expression graph"),
    ("exp04_attractor_floquet",       "attractor landscape + Floquet stability (seeded by exp03's graph)"),
    ("exp04_manifold_geometry",       "phasor manifold: geodesics and branch cuts"),
    ("exp03_multiomics_fusion",       "coherence-gated multi-omics fusion (CPTAC UCEC)"),
    ("exp04_ml_classification",       "phasor classification, cross-validated"),
    ("exp05_cst_knockout",            "cell-state tensor evolution + in-silico knockout ranking"),
    ("exp06_cst_temporal",            "CST temporal dynamics (cell-cycle, circadian, velocity)"),
    ("exp07_cst_tensornetwork",       "tensor-train compression of the CST + uncertainty"),
    ("exp08_cst_pathway",             "pathway-atlas CST: spectrum, CP error, coherence"),
    ("exp10_cst_quantum_bridge",      "CST as a density operator: entropy, coherence, fidelity"),
    ("exp11_vpc_vqc_complexity",      "VPC->VQC gate correspondence + complexity crossover"),
    ("exp12_multiomics_benchmark",    "head-to-head vs MOFA+ / SNF on the identical matrix"),
    ("exp12b_benchmark_competitors_fix", "benchmark competitor fixes, same matrix"),
    ("exp13_hardtask_clinical",       "hard clinical tasks: histologic grade and subtype"),
    ("check_numbers",                 "manuscript-number guard: every quoted literal traces to results/"),
]

# Not in ORDER, and why:
#   biophasor_realdata_loader.py — a data-fetch / feasibility script, not a
#   result producer. Run it directly to (re)fill the shared cache from GEO.


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for mod, desc in ORDER:
            print(f"{mod:36s} {desc}")
        return 0

    if "--check" in argv:
        return common.check_plan(
            SUITE,
            plan=[(os.path.join(HERE, f"{m}.py"), d) for m, d in ORDER],
            inputs=[
                (os.path.join(common.CACHE, "GSE293316_reh.h5"),
                 "GSE293316 REH scRNA-seq (exp01, exp03*, exp04*)"),
                (os.path.join(common.CACHE, "GSE171432_fpkm.tsv.gz"),
                 "GSE171432 WT mouse-liver circadian FPKM (exp02)"),
                (os.path.join(common.CACHE, "cptac_ucec", "rna.pkl.gz"),
                 "CPTAC UCEC RNA matrix (exp03_multiomics, exp05, exp07, exp10-13)"),
                (os.path.join(common.CACHE, "cptac_ucec", "protein.pkl.gz"),
                 "CPTAC UCEC protein matrix (same)"),
                (os.path.join(common.CACHE, "cptac_ucec", "labels.pkl.gz"),
                 "CPTAC UCEC tumour/normal labels"),
                (os.path.join(common.CACHE, "cptac_ucec", "clinical_mssm.pkl"),
                 "CPTAC UCEC clinical table (exp13 hard tasks)"),
            ],
            notes=[
                "\nnot regenerated by this suite (no generating script in this "
                "repository,\n1-Spectral or 5-Biophasor-Local):",
                "  fig_platform_overview.png, fig2_benchmark.png, fig3_hardtask_roc.png",
                "  — printed by manuscripts/biophasor/main.tex; their NUMBERS are "
                "regenerated\n    (exp12/exp12b/exp13), only the rendering step is missing. "
                "See README.md.",
            ],
        )

    wanted = [a for a in argv if not a.startswith("-")]
    todo = [(m, d) for m, d in ORDER
            if not wanted or any(m.startswith(w) for w in wanted)]
    if wanted and not todo:
        print(f"no script matches {wanted}; --list shows the order", file=sys.stderr)
        return 2

    failures = []
    for mod, desc in todo:
        print(f"\n{'=' * 72}\n{mod}  --  {desc}\n{'=' * 72}", flush=True)
        t0 = time.time()
        try:
            runpy.run_module(f"experiments.{SUITE}.codes.{mod}", run_name="__main__")
            status = "ok"
        except SystemExit as e:
            code = e.code or 0
            status = "ok" if code == 0 else f"exit {code}"
            if code:
                failures.append(mod)
        except BaseException:
            traceback.print_exc()
            status = "FAILED"
            failures.append(mod)
        print(f"[{mod}: {status} in {time.time() - t0:.1f}s]", flush=True)

    print(f"\n{'=' * 72}")
    if failures:
        print("FAILED:", ", ".join(failures))
        return 1
    print(f"all {len(todo)} scripts completed")
    print("results -> experiments/biophasor/results/ | figures -> manuscripts/biophasor/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
