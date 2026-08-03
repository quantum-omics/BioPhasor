"""
experiments._shared.common — cross-suite experiment glue for the BioPhasor
platform.

The SINGLE home for experiment infrastructure shared across the four suites:

  * the shared raw-data cache (one copy per accession under ``data/raw/``)
    and the GEO fetch that fills it
  * the output-path convention — ``results_dir(suite)`` and
    ``manuscript_figs(suite)`` — so no script hard-codes ``../figures`` or
    reaches sideways into a manuscript folder
  * cosinor rhythmicity gate, probe→symbol mapping, top-variable selection

Reusable *science* does NOT live here — it lives in the installed ``biophasor``
package and is imported. This module is only experiment plumbing.

Where outputs go
----------------
Numbers go to ``experiments/<suite>/results/``; they are the receipts the
manuscript quotes and they are tracked.

Figures are written **once**, into the manuscript that prints them. Nothing is
written to a suite-local ``figures/`` any more: the two copies were byte
identical and drifted the moment one was regenerated alone.

``manuscript_figs(suite)`` returns the directory the suite's ``main.tex``
actually resolves ``\\includegraphics`` against, which is not the same for all
four:

    biophasor          manuscripts/biophasor/            \\graphicspath{{./}}
    tumor              manuscripts/tumor/                \\graphicspath{{./}}
    spectral-classical manuscripts/spectral-classical/fig/     {{fig/}}
    spectral-quantum   manuscripts/spectral-quantum/manuscript/fig/  {{fig/}}

If a ``.tex`` preamble is later given ``\\graphicspath{{./}{./figs/}}`` (the
Classical-Virtual-Omics convention), change the mapping here — not the scripts.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# experiments/_shared/common.py  →  HERE = experiments/_shared, EXP = experiments/
HERE = os.path.dirname(os.path.abspath(__file__))          # experiments/_shared
EXP  = os.path.dirname(HERE)                               # experiments/
ROOT = os.path.dirname(EXP)                                # repository root
MANUSCRIPTS = os.path.join(ROOT, "manuscripts")
# ONE shared raw-data cache, keyed by accession, used by every suite.
CACHE = os.path.join(HERE, "data", "raw")
os.makedirs(CACHE, exist_ok=True)

# suite → the directory that suite's main.tex resolves \includegraphics against.
# Read off the \graphicspath in each manuscript; see the module docstring.
_MANUSCRIPT_FIGDIR = {
    "biophasor":          ("biophasor",),
    "tumor":              ("tumor",),
    "spectral-classical": ("spectral-classical", "fig"),
    "spectral-quantum":   ("spectral-quantum", "manuscript", "fig"),
}


def suite_dir(suite: str) -> str:
    """Return ``experiments/<suite>``, checking it exists."""
    d = os.path.join(EXP, suite)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"no such experiment suite: {d}")
    return d


def results_dir(suite: str) -> str:
    """Return (and create) ``experiments/<suite>/results``."""
    d = os.path.join(suite_dir(suite), "results")
    os.makedirs(d, exist_ok=True)
    return d


def manuscript_figs(suite: str) -> str:
    """Return (and create) the ONE directory this suite's figures belong in.

    That is the manuscript directory the ``.tex`` reads figures from — result
    figures are not duplicated into a suite-local ``figures/``.
    """
    try:
        parts = _MANUSCRIPT_FIGDIR[suite]
    except KeyError:
        raise KeyError(
            f"unknown suite {suite!r}; known: {sorted(_MANUSCRIPT_FIGDIR)}"
        ) from None
    d = os.path.join(MANUSCRIPTS, *parts)
    os.makedirs(d, exist_ok=True)
    return d


def exp_dir(suite: str):
    """Return ``(manuscript_figure_dir, results_dir)`` for a suite.

    Kept as the one call a script makes to learn where its two kinds of output
    go. The first element is the manuscript figure directory (figures are
    written once, where the ``.tex`` reads them); the second is the suite's
    ``results/``.
    """
    return manuscript_figs(suite), results_dir(suite)


def save_fig(fig, name: str, suite: str, close: bool = True) -> str:
    """Write ``fig`` once, into the manuscript directory for ``suite``."""
    path = os.path.join(manuscript_figs(suite), name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path


def check_plan(suite: str, plan, inputs=(), notes=(), problems=()) -> int:
    """Validate a suite's run plan WITHOUT running any of it. Returns an exit code.

    Backs the ``--check`` flag on every suite driver: it answers "would this
    run, and on what?" cheaply, so nobody has to start a multi-minute
    experiment to discover a missing input or a syntax error.

    What it does, in order:

    1. prints the ordered plan, so the driver's intent is visible;
    2. byte-compiles each script's SOURCE — this catches syntax errors without
       importing, which matters because several experiment modules load their
       data at import time and importing them would be the expensive run this
       flag exists to avoid;
    3. checks each declared input path exists (cache files, package data);
    4. checks the suite's results dir and its manuscript figure dir are
       writable.

    Parameters
    ----------
    suite : the suite name, as known to ``manuscript_figs``.
    plan  : iterable of ``(script_path, description)``. ``script_path`` is an
            absolute path to the .py file that would run.
    inputs: iterable of ``(path, description)`` the suite needs on disk.
    notes : extra lines to print verbatim — used to state what a suite cannot
            regenerate at all.
    problems : failures the caller already found (e.g. a missing optional
            dependency that the requested flags DO require). These fail the
            check like any other, so ``--check`` never reports OK for a plan
            that would not run.
    """
    plan = list(plan)
    inputs = list(inputs)
    print(f"{'=' * 72}\ncheck: {suite}  (validating only — nothing is executed)\n{'=' * 72}")

    print(f"\nplan — {len(plan)} script(s), in order:")
    for i, (path, desc) in enumerate(plan, 1):
        print(f"  {i:2d}. {os.path.basename(path):<40s} {desc}")

    problems = list(problems)

    print("\nsource compiles:")
    for path, _ in plan:
        if not os.path.isfile(path):
            print(f"  MISSING  {path}")
            problems.append(f"missing script {path}")
            continue
        try:
            # compile in memory: no .pyc is written and, crucially, the module
            # is NOT imported — several experiment modules load their data at
            # import time, which is exactly the expensive work --check avoids.
            with open(path, "rb") as fh:
                compile(fh.read(), path, "exec")
            print(f"  ok       {os.path.basename(path)}")
        except SyntaxError as e:
            print(f"  FAILED   {os.path.basename(path)}: line {e.lineno}: {e.msg}")
            problems.append(f"syntax error in {path}")

    if inputs:
        print("\ninputs:")
        for path, desc in inputs:
            if os.path.exists(path):
                print(f"  ok       {desc}")
            else:
                print(f"  MISSING  {desc}\n             expected at {path}")
                problems.append(f"missing input {path}")

    print("\noutput destinations:")
    for label, d in (("results", results_dir(suite)),
                     ("figures", manuscript_figs(suite))):
        writable = os.access(d, os.W_OK)
        print(f"  {'ok      ' if writable else 'NOT WRITABLE'} {label}: {d}")
        if not writable:
            problems.append(f"{label} dir not writable: {d}")

    for line in notes:
        print(line)

    print(f"\n{'=' * 72}")
    if problems:
        print(f"check FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("check OK — the plan above would run; nothing was executed")
    return 0


def _ensure_geo(acc: str):
    """Load a GEO series (from cache if present) and return the GEOparse object."""
    import GEOparse
    return GEOparse.get_GEO(geo=acc, destdir=CACHE, how="full", silent=True)


def load_expression(acc: str) -> Tuple[pd.DataFrame, "GEOparse.GSE"]:
    """Return (probe × sample expression DataFrame, GSE object). Cached as pickle."""
    pkl = os.path.join(CACHE, f"{acc}_expr.pkl")
    g = _ensure_geo(acc)
    if os.path.exists(pkl):
        df = pd.read_pickle(pkl)
    else:
        df = g.pivot_samples("VALUE")
        df.to_pickle(pkl)
    return df, g


def probe_to_symbol(g) -> Dict[str, str]:
    """Map probe id → gene symbol using the platform annotation table."""
    gpl = list(g.gpls.values())[0]
    tbl = gpl.table
    # find the symbol-like column
    sym_col = None
    for cand in ["Gene Symbol", "Gene symbol", "GENE_SYMBOL", "Symbol", "gene_assignment"]:
        if cand in tbl.columns:
            sym_col = cand
            break
    id_col = "ID" if "ID" in tbl.columns else tbl.columns[0]
    if sym_col is None:
        return {}
    mapping: Dict[str, str] = {}
    for pid, sym in zip(tbl[id_col].astype(str), tbl[sym_col].astype(str)):
        if not sym or sym.lower() in ("nan", "---", ""):
            continue
        if sym_col == "gene_assignment":
            # RefSeq-style: "acc // SYMBOL // desc // ..."; symbol is 2nd token
            parts = [p.strip() for p in sym.split("//")]
            token = parts[1] if len(parts) > 1 else parts[0]
        else:
            # Affymetrix "Gene Symbol": "DDR1 /// MIR4640" — first triple-slash token
            token = sym.split("///")[0].strip()
        if token:
            mapping[pid] = token
    return mapping


def select_top_variable(df: pd.DataFrame, n_top: int = 200,
                        force_probes=None) -> pd.DataFrame:
    """Keep the n_top most-variable probes (rows), drop all-NaN rows first.

    force_probes : iterable of probe ids to always include (e.g. marker probes),
    so the compartment axes are populated even if the markers are not among the
    most-variable features.
    """
    d = df.dropna(how="any")
    v = d.var(axis=1)
    keep = list(v.sort_values(ascending=False).head(n_top).index)
    if force_probes is not None:
        for p in force_probes:
            if p in d.index and p not in keep:
                keep.append(p)
    return d.loc[keep]


def rhythmicity_gate(df: pd.DataFrame, ct: np.ndarray, period: float = 24.0,
                     n_top: int = 300, force_probes=None) -> pd.DataFrame:
    """Keep the n_top most-rhythmic probes by 24h cosinor R² (theory.md §1.2).

    A lightweight cosinor-based rhythmicity gate:
    each probe's log-expression is regressed on {cos, sin, 1} at the given
    period, and probes are ranked by the fraction of variance explained.

    Parameters
    ----------
    df : probe × sample DataFrame (samples already ordered by ct).
    ct : per-sample circadian time (hours), same length as df.columns.
    force_probes : probe ids to always include (e.g. clock markers).
    """
    d = df.dropna(how="any")
    ct = np.asarray(ct, dtype=float)
    w = 2 * np.pi / period
    A = np.column_stack([np.cos(w * ct), np.sin(w * ct), np.ones_like(ct)])
    L = np.log1p(d.values.T)                                # (T, N)
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    fit = A @ coef
    ss_res = ((L - fit) ** 2).sum(axis=0)
    ss_tot = ((L - L.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    ranked = d.index[np.argsort(r2)[::-1][:n_top]]
    keep = list(ranked)
    if force_probes is not None:
        for p in force_probes:
            if p in d.index and p not in keep:
                keep.append(p)
    return d.loc[keep]


def marker_probes(g) -> list:
    """Probe ids whose gene symbol is one of the compartment markers."""
    from biophasor.spectral.omics.markers import marker_to_compartment
    lookup = marker_to_compartment()
    p2s = probe_to_symbol(g)
    return [pid for pid, sym in p2s.items() if str(sym).strip().upper() in lookup]


def membership_from_probes(probe_ids, g) -> Dict[str, List[int]]:
    """Build compartment membership for a probe list (row order) via markers."""
    from biophasor.spectral.omics.markers import build_membership
    p2s = probe_to_symbol(g)
    symbols = [p2s.get(str(pid), "") for pid in probe_ids]
    return build_membership(symbols)
