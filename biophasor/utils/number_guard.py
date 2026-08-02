"""
biophasor.utils.number_guard — every quoted number traces to a results file.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation

A manuscript's results section claims its numbers were read from that suite's
``experiments/<suite>/results/*.json``. The claim is only worth making if it is
enforced: quoted values drift silently when a suite is rerun after a fix, and a
stale number is indistinguishable from a fresh one by eye.

This module parses every numeric literal out of the guarded section of a
manuscript and requires each to round-trip to a value in the results JSON **at
the precision written**. Rounding is part of the claim: where the source says
``0.48046875``, the literal ``0.480`` is correct and ``0.481`` is a defect.

Failures print the nearest candidates, so the fix is a substitution rather than
a search.

Ported from ``cvomics.utils.number_guard`` so that both repositories
enforce the identical contract; the tokenisation fixes it carries are the
substance of the port and are documented at each regex below. Two extensions
are BioPhasor-specific and exist because this repository's manuscripts are laid
out differently: ``tex`` may name several files (``spectral-quantum`` splits its
manuscript into ``sections/*.tex``), and ``end`` may be omitted to guard from
the start marker to the end of the file (a section that runs to EOF has no
following marker to stop at).

Usage from a suite's ``check_numbers.py``::

    from biophasor.utils.number_guard import GuardConfig, run_guard

    raise SystemExit(run_guard(GuardConfig(
        results_dir=..., tex=..., start=r"\\section{Results}",
        end=r"\\section{Discussion}", whitelist={17.0, 5.0, ...},
    )))
"""

from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

__all__ = ["GuardConfig", "GuardResult", "check_numbers", "run_guard", "run_guards"]

#: LaTeX scientific notation, folded to a plain float before anything else so
#: that ``$4.8\times10^{-8}$`` reads as one literal rather than as 4.8 and -8.
_SCI = re.compile(r"(-?\d+\.?\d*)\s*\\times\s*10\^\{?(-?\d+)\}?")

#: A bare power of ten, ``10^{-12}``, with no mantissa. Without this the
#: exponent leaks out as its own literal and the guard reports a spurious -12.
_POW10 = re.compile(r"(?<![.\d])10\^\{?(-?\d+)\}?")

#: LaTeX thousands separators: ``12{,}600`` and ``12\,600`` are ONE number.
#: Left unhandled they split into 12 and 600, and the guard then hunts for a
#: value of 600 that was never claimed.
_THOUSANDS = re.compile(r"(?<=\d)(?:\{,\}|\\[,;:!]|~)(?=\d{3}(?!\d))")

#: A LaTeX dash RUN is punctuation, never a minus sign: an en-dash marks a
#: range (``CT18--65``, the bootstrap CI ``0.989--1.000``) and an em-dash marks
#: a clause break (``the logistic fallback---4 layers``). A minus in LaTeX is a
#: single hyphen. Without this the literal pattern reads the trailing dash as a
#: sign and the guard reports an unmatched -65, -1.000 and -4 that no one ever
#: claimed — the same defect class as the thousands separator, and live on all
#: four of these manuscripts.
_DASHRUN = re.compile(r"-{2,}")

#: LaTeX constructs whose digits are not measurements.
#:
#: The first two entries are BioPhasor additions to the ported list; both are
#: the same class of defect as the thousands separator, found on these
#: manuscripts:
#:
#:   * a length argument, ``{0.48\linewidth}`` on a ``subfigure`` or
#:     ``\includegraphics``, is page geometry. The ported list stripped
#:     ``width=0.48`` but not the brace form, so every two-column figure
#:     reported a spurious claim of 0.48.
#:   * ``\tfrac12`` is the two-argument shorthand for one half. Removing the
#:     macro name alone leaves ``12`` behind, and the guard then hunts for a
#:     measured value of twelve.
_STRIP = [
    (re.compile(r"-?[0-9.]+\s*\\(linewidth|textwidth|columnwidth|paperwidth|"
                r"hsize|baselineskip|textheight)"), " "),
    # A confidence LEVEL is the interval's definition, not a measurement: the
    # "95" in "95\% CI $[0.291,0.299]$" is not a number anyone measured, while
    # the endpoints are and stay checked. Stripping the idiom is safer than
    # whitelisting 95.0, because these manuscripts also quote a measured
    # 95-sample tumour arm and a value-keyed whitelist cannot tell the two
    # apart. The match requires the level to be immediately followed by CI or
    # "confidence", so a bare 95 is untouched.
    (re.compile(r"\$?\s*\d{2}(?:\.\d+)?\s*\\?%\$?\s*"
                r"(?:\\[A-Za-z]+\s*)*"
                r"(?:bootstrap\s+)?(?:CI\b|confidence)"), " "),
    # A math sub/superscript on a SYMBOL is index notation, not a measurement:
    # the 0 of $E_0$, the 2 of $R^2$, the 1 of $\lambda_1$ and $\varphi_1$, the
    # 3 of $H^3_{ij}$. These are the most common digits in a theory-heavy
    # manuscript, and the alternative — whitelisting 0, 1, 2, 3 — would exempt
    # every genuinely measured small integer along with them. Runs after _SCI
    # and _POW10, so a real exponent (10^{-12}, \times10^{4}) is already folded
    # into its mantissa and is not touched here.
    (re.compile(r"(?<=[A-Za-z}\)\]])\s*[_^]\s*(?:\{[^{}]*\}|\d+|[A-Za-z])"), " "),
    # A percentile ORDINAL names which order statistic was taken; the value at
    # that percentile is the measurement and is checked. "null 95th percentile
    # $R^2=0.26$" claims 0.26, not 95.
    (re.compile(r"\b\d+\s*(?:st|nd|rd|th)\s+percentile"), " "),
    # A number fused to a TeX length unit is page geometry: the column widths
    # of `\begin{tabular}{p{2.6cm}p{4.4cm}...}`. Adjacency is required — a
    # space would let "$50$ in each arm" lose its fifty.
    (re.compile(r"-?[0-9.]+(cm|mm|pt|em|ex|bp|pc|dd|sp|in)\b"), " "),
    (re.compile(r"\\[dt]?frac\s*\d\s*\d"), " "),
    (re.compile(r"\\(label|ref|eqref|cite|includegraphics|Tlab)\s*\[[^\]]*\]\{[^}]*\}"), " "),
    (re.compile(r"\\(label|ref|eqref|cite|includegraphics|Tlab)\{[^}]*\}"), " "),
    (re.compile(r"\\begin\{[^}]*\}|\\end\{[^}]*\}"), " "),
    (re.compile(r"%.*"), " "),
    (re.compile(r"\\[A-Za-z]+"), " "),
    (re.compile(r"width=[0-9.]+"), " "),
]

#: A leading minus is a sign only when it is not a hyphen inside a word:
#: "arm-3" is not the number minus three.
_LITERAL = re.compile(
    r"(?<![A-Za-z0-9])(-?\d+\.\d+(?:e[-+]?\d+)?|-?\d+e[-+]?\d+|-?\d+)")


@dataclass
class GuardConfig:
    """What to guard, and where the numbers should be found.

    results_dir : directory of ``*.json`` result files, or a sequence of them.
                  A sequence is for the case where a manuscript's numbers are
                  produced by more than one suite — the ``tumor`` paper's
                  pathway-atlas subsection is written by the ``biophasor``
                  suite's ``exp08`` and lands in that suite's ``results/``.
                  Widening the search is not a loophole: the contract is that
                  a quoted number traces to a results file this repository
                  produced, and a value has to exist in one of them to match.
    tex         : the manuscript source; a path, or a sequence of paths when
                  the manuscript is split into ``sections/*.tex``. Each file is
                  searched for the markers and the sections are concatenated;
                  it is an error for NO file to carry the start marker, so a
                  renamed section still fails loudly rather than guarding
                  nothing.
    start, end  : literal markers bounding the guarded section. ``end`` may be
                  None or "" when the guarded section runs to the end of file.
    whitelist   : values that are not measurements — structural counts, class
                  labels, model dimensions. Keep it short and explicit; an
                  over-broad whitelist silently disables the guard.
    """

    results_dir: Path | str | Sequence[Path | str]
    tex: Path | str | Sequence[Path | str]
    start: str
    end: str | None = None
    whitelist: set[float] = field(default_factory=set)


@dataclass
class GuardResult:
    n_checked: int
    n_values: int
    n_files: int
    unmatched: list[tuple[str, list[tuple[str, float]]]]

    @property
    def ok(self) -> bool:
        return not self.unmatched


def _flatten(o: Any, p: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flatten(v, f"{p}.{k}" if p else k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(_flatten(v, f"{p}[{i}]"))
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        # NaN and +-Inf are recorded by several scripts where a statistic was
        # undefined (an empty stratum, a zero-variance gene). They are not
        # candidate matches for any literal, and letting one through makes the
        # rounding arithmetic raise instead of report — a crash that reads like
        # a broken guard rather than an unmatched number.
        if math.isfinite(o):
            out[p] = float(o)
    return out


def _results_dirs(results_dir: Path | str | Sequence[Path | str]) -> list[str]:
    if isinstance(results_dir, (str, Path)):
        return [str(results_dir)]
    return [str(d) for d in results_dir]


def _result_files(results_dir: Path | str | Sequence[Path | str]) -> list[str]:
    out: list[str] = []
    for d in _results_dirs(results_dir):
        out.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    return out


def _load_values(results_dir: Path | str | Sequence[Path | str]) -> dict[str, float]:
    vals: dict[str, float] = {}
    for f in _result_files(results_dir):
        stem = os.path.basename(f)[:-5]
        with open(f) as fh:
            for k, v in _flatten(json.load(fh)).items():
                vals[f"{stem}.{k}"] = v
    return vals


def _sig_of(lit_str: str) -> int:
    """Significant figures the literal was written to."""
    m = re.search(r"(-?\d+\.?\d*)", lit_str.replace("\\times10^", "e"))
    if not m:
        return 3
    digits = m.group(1).replace("-", "").replace(".", "")
    stripped = digits.lstrip("0")
    return len(stripped) if stripped else 1


def _rnd(x: float, sig: int) -> float:
    if x == 0:
        return 0.0
    d = sig - int(math.floor(math.log10(abs(x)))) - 1
    return round(x, d)


def _matches(lit: float, val: float, sig: int) -> bool:
    if val == 0:
        return abs(lit) < 1e-30
    return _rnd(val, sig) == _rnd(lit, sig)


def _section(text: str, start: str, end: str | None) -> str:
    i = text.find(start)
    if i < 0:
        raise ValueError(f"guard: section marker {start!r} not found")
    if not end:
        # A section that runs to EOF has no following marker to stop at.
        return text[i:]
    j = text.find(end, i)
    return text[i:j if j > 0 else len(text)]


def _read_sections(cfg: GuardConfig) -> str:
    """Concatenate the guarded section from every file that carries it."""
    paths = [cfg.tex] if isinstance(cfg.tex, (str, Path)) else list(cfg.tex)
    parts: list[str] = []
    for p in paths:
        with open(p) as fh:
            text = fh.read()
        try:
            parts.append(_section(text, cfg.start, cfg.end))
        except ValueError:
            # Only some of a multi-file manuscript carries the marker; a
            # missing marker is an error only if NO file has it.
            continue
    if not parts:
        raise ValueError(
            f"guard: section marker {cfg.start!r} not found in "
            f"{', '.join(str(p) for p in paths)}"
        )
    return "\n".join(parts)


def check_numbers(cfg: GuardConfig) -> GuardResult:
    """Run the guard and return the result without printing or exiting."""
    vals = _load_values(cfg.results_dir)
    if not vals:
        raise FileNotFoundError(
            "guard: no results JSON in "
            + ", ".join(_results_dirs(cfg.results_dir))
            + " — run the experiments first"
        )
    sec = _read_sections(cfg)

    # Order matters: join thousands groups before any digit-bearing construct is
    # stripped, fold mantissa-and-exponent before bare powers of ten, and only
    # then remove the LaTeX whose digits are not measurements.
    clean = _THOUSANDS.sub("", sec)
    clean = _DASHRUN.sub(" ", clean)
    clean = _SCI.sub(lambda m: f"{m.group(1)}e{m.group(2)}", clean)
    clean = _POW10.sub(lambda m: f"1e{m.group(1)}", clean)
    for pat, rep in _STRIP:
        clean = pat.sub(rep, clean)

    unmatched: list[tuple[str, list[tuple[str, float]]]] = []
    n_checked = 0
    for m in _LITERAL.finditer(clean):
        lit_str = m.group(0)
        try:
            lit = float(lit_str)
        except ValueError:
            continue
        if lit in cfg.whitelist:
            continue
        n_checked += 1
        sig = _sig_of(lit_str)
        if not any(_matches(lit, v, sig) for v in vals.values()):
            near = sorted(vals.items(), key=lambda kv: abs(kv[1] - lit))[:3]
            unmatched.append((lit_str, near))

    n_files = len(_result_files(cfg.results_dir))
    return GuardResult(n_checked, len(vals), n_files, unmatched)


def run_guard(cfg: GuardConfig) -> int:
    """Run the guard, report, and return a process exit code."""
    try:
        res = check_numbers(cfg)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        return 1

    print(f"guard: {res.n_checked} literals checked against {res.n_values} "
          f"JSON values from {res.n_files} files")
    if res.ok:
        print("guard: every numeric literal in the guarded section traces to a "
              "results file at the precision written")
        return 0
    for lit_str, near in res.unmatched:
        print(f"  UNMATCHED {lit_str}")
        for k, v in near:
            print(f"      nearest: {k} = {v!r}")
    print(f"guard: {len(res.unmatched)} unmatched literal(s)")
    return 1


def run_guards(configs: dict[str, GuardConfig]) -> int:
    """Run several named guards over one manuscript and OR their exit codes.

    A manuscript whose measured numbers are split across files — the
    ``spectral-classical`` main text plus its ``snippets/*.tex``, or the
    ``spectral-quantum`` ``sections/*.tex`` — needs more than one section
    bound. Running them separately keeps each failure attributable to a file;
    a single concatenated blob would report an unmatched literal without
    saying which part of the manuscript wrote it.
    """
    rc = 0
    for name, cfg in configs.items():
        print(f"\n--- {name} ---")
        rc |= run_guard(cfg)
    return rc
