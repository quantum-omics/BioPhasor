"""
test_number_guard.py — the guard that keeps manuscript numbers honest.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation

Most of these tests encode defects the guard actually had, on these four
manuscripts or on the Classical-Virtual-Omics ones it was ported from. Each is
the same failure mode: a LaTeX construct whose digits are not a measurement is
read as one, and the guard then reports a confident failure against a number no
one ever claimed. A false alarm is worse than no guard, because it trains the
reader to ignore the output.

The tokenisation cases are therefore not decoration — they are the reason the
guard can be left switched on.
"""
import json

import pytest

from biophasor.utils.number_guard import GuardConfig, check_numbers, run_guards


def _write(tmp_path, tex_body: str, values: dict):
    res = tmp_path / "results"
    res.mkdir(parents=True)
    (res / "summary.json").write_text(json.dumps(values))
    tex = tmp_path / "m.tex"
    tex.write_text("\\section{Numerics}\n" + tex_body + "\n\\section{Discussion}\n")
    return GuardConfig(results_dir=res, tex=tex,
                       start=r"\section{Numerics}", end=r"\section{Discussion}")


# --- the contract -----------------------------------------------------------

def test_matching_literal_passes(tmp_path):
    cfg = _write(tmp_path, "The coefficient is $0.5063$.", {"coef": 0.50627})
    assert check_numbers(cfg).ok


def test_rounding_is_part_of_the_claim(tmp_path):
    """0.480 is correct where the source says 0.48046875; 0.481 is a defect."""
    ok = _write(tmp_path / "ok", "value $0.480$", {"v": 0.48046875})
    assert check_numbers(ok).ok

    bad = _write(tmp_path / "bad", "value $0.481$", {"v": 0.48046875})
    assert not check_numbers(bad).ok


def test_drifted_number_is_caught(tmp_path):
    cfg = _write(tmp_path, "The coefficient is $0.9999$.", {"coef": 0.50627})
    res = check_numbers(cfg)
    assert not res.ok
    assert res.unmatched[0][0] == "0.9999"
    # the failure must name candidates, so the fix is a substitution
    assert res.unmatched[0][1]


def test_only_the_guarded_section_is_checked(tmp_path):
    """A number in the discussion is not the guard's business."""
    res = tmp_path / "results"
    res.mkdir()
    (res / "s.json").write_text(json.dumps({"v": 1.5}))
    tex = tmp_path / "m.tex"
    tex.write_text("\\section{Numerics}\n$1.5$\n\\section{Discussion}\n$99.9$\n")
    cfg = GuardConfig(results_dir=res, tex=tex,
                      start=r"\section{Numerics}", end=r"\section{Discussion}")
    assert check_numbers(cfg).ok


# --- tokenisation: each case is a defect the guard actually had --------------

def test_thousands_separator_is_one_number(tmp_path):
    """12{,}600 is one literal. Split, it invents a claim of 600."""
    cfg = _write(tmp_path, "there are $12{,}600$ rows", {"rows": 12600})
    assert check_numbers(cfg).ok


def test_backslash_thousands_separator(tmp_path):
    cfg = _write(tmp_path, "there are $12\\,600$ rows", {"rows": 12600})
    assert check_numbers(cfg).ok


def test_bare_power_of_ten_does_not_leak_its_exponent(tmp_path):
    """10^{-12} is a threshold, not a claim that -12 appears in the results."""
    cfg = _write(tmp_path, "a defect below $10^{-12}$", {"tol": 1e-12})
    assert check_numbers(cfg).ok


def test_scientific_notation_is_one_literal(tmp_path):
    cfg = _write(tmp_path, "cross-talk $5.2\\times10^{-4}$", {"x": 5.194e-4})
    assert check_numbers(cfg).ok


def test_en_dash_range_is_not_a_negative_number(tmp_path):
    """CT18--65 is a range. Read as a sign it invents a claim of -65."""
    cfg = _write(tmp_path, "samples over CT18--65", {"lo": 18.0, "hi": 65.0})
    assert check_numbers(cfg).ok


def test_em_dash_clause_break_is_not_a_negative_number(tmp_path):
    """"...fallback---4 layers" claims 4, not -4."""
    cfg = _write(tmp_path, "the logistic fallback---4 layers", {"layers": 4.0})
    assert check_numbers(cfg).ok


def test_figure_width_is_page_geometry_not_a_measurement(tmp_path):
    cfg = _write(tmp_path, "\\begin{subfigure}[t]{0.48\\textwidth} $1.5$",
                 {"v": 1.5})
    assert check_numbers(cfg).ok


def test_tabular_column_width_is_page_geometry(tmp_path):
    """p{2.6cm} is a column width, not a claim of 2.6."""
    cfg = _write(tmp_path, "\\begin{tabular}{p{2.6cm}p{4.4cm}} $1.5$", {"v": 1.5})
    assert check_numbers(cfg).ok


def test_tfrac_shorthand_does_not_leak_twelve(tmp_path):
    """\\tfrac12 is one half; stripping the macro alone leaves a bare 12."""
    cfg = _write(tmp_path, "$E_0=\\tfrac12\\sum_k\\varepsilon_k=6.5556$",
                 {"E0": 6.5556})
    assert check_numbers(cfg).ok


def test_confidence_level_is_stripped_but_endpoints_are_checked(tmp_path):
    """95\\% CI names the level; the interval's endpoints are the measurement.

    The level cannot be whitelisted by value: these manuscripts also quote a
    measured 95-sample tumour arm, and the whitelist matches on value alone.
    """
    ok = _write(tmp_path / "ok", "$r=0.295$ (95\\% CI $[0.291,0.299]$)",
                {"r": 0.2952, "lo": 0.2912, "hi": 0.2993})
    assert check_numbers(ok).ok

    # a drifted endpoint must still fail — the strip must not swallow the CI
    bad = _write(tmp_path / "bad", "$r=0.295$ (95\\% CI $[0.291,0.999]$)",
                 {"r": 0.2952, "lo": 0.2912, "hi": 0.2993})
    assert not check_numbers(bad).ok
    assert "0.999" in [u[0] for u in check_numbers(bad).unmatched]


def test_a_bare_ninety_five_is_still_checked(tmp_path):
    """The CI strip must not exempt 95 everywhere — it is also a sample count."""
    cfg = _write(tmp_path, "across the $95$ tumour samples", {"n_normal": 14.0})
    assert not check_numbers(cfg).ok


def test_percentile_ordinal_is_not_a_measurement(tmp_path):
    """"null 95th percentile $R^2=0.26$" claims 0.26, not 95."""
    cfg = _write(tmp_path, "null 95th percentile $R^2=0.26$", {"r2": 0.2597})
    assert check_numbers(cfg).ok


# --- whitelist, and the ways a guard is silently disabled -------------------

def test_whitelist_exempts_structural_counts(tmp_path):
    cfg = _write(tmp_path, "the $17$ modules", {"unrelated": 1.0})
    cfg.whitelist = {17.0}
    assert check_numbers(cfg).ok


def test_whitelist_does_not_exempt_a_different_value(tmp_path):
    """A whitelist is exact, never a tolerance band."""
    cfg = _write(tmp_path, "the coefficient $17.4$", {"unrelated": 1.0})
    cfg.whitelist = {17.0}
    assert not check_numbers(cfg).ok


def test_missing_results_is_an_error_not_a_pass(tmp_path):
    """An empty results directory must fail; silence would look like success."""
    tex = tmp_path / "m.tex"
    tex.write_text("\\section{Numerics}\n$1.234$\n\\section{Discussion}\n")
    empty = tmp_path / "empty"
    empty.mkdir()
    cfg = GuardConfig(results_dir=empty, tex=tex,
                      start=r"\section{Numerics}", end=r"\section{Discussion}")
    with pytest.raises(FileNotFoundError):
        check_numbers(cfg)


def test_missing_section_marker_is_an_error(tmp_path):
    cfg = _write(tmp_path, "$1.0$", {"v": 1.0})
    cfg.start = r"\section{Nonexistent}"
    with pytest.raises(ValueError, match="not found"):
        check_numbers(cfg)


def test_nan_in_results_does_not_crash_the_guard(tmp_path):
    """A NaN receipt must report an unmatched literal, not raise.

    Several scripts record NaN where a statistic was undefined. A crash there
    reads as a broken guard rather than as a number that does not trace.
    """
    res = tmp_path / "results"
    res.mkdir()
    (res / "s.json").write_text('{"undefined": NaN, "v": 1.5}')
    tex = tmp_path / "m.tex"
    tex.write_text("\\section{Numerics}\n$1.5$ and $9.9$\n\\section{Discussion}\n")
    cfg = GuardConfig(results_dir=res, tex=tex,
                      start=r"\section{Numerics}", end=r"\section{Discussion}")
    r = check_numbers(cfg)
    assert [u[0] for u in r.unmatched] == ["9.9"]


# --- the BioPhasor-specific extensions --------------------------------------

def test_no_end_marker_guards_to_end_of_file(tmp_path):
    """A section that runs to EOF has no following marker to stop at."""
    res = tmp_path / "results"
    res.mkdir()
    (res / "s.json").write_text(json.dumps({"v": 1.5}))
    tex = tmp_path / "m.tex"
    tex.write_text("\\section{Results}\n$1.5$ and $2.5$\n")
    cfg = GuardConfig(results_dir=res, tex=tex, start=r"\section{Results}",
                      end=None)
    r = check_numbers(cfg)
    assert [u[0] for u in r.unmatched] == ["2.5"]


def test_several_results_dirs_are_searched(tmp_path):
    """Two suites can write one manuscript's numbers (tumour + pathway atlas)."""
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    (a / "one.json").write_text(json.dumps({"v": 1.5}))
    (b / "two.json").write_text(json.dumps({"w": 2.5}))
    tex = tmp_path / "m.tex"
    tex.write_text("\\section{Results}\n$1.5$ and $2.5$\n")
    cfg = GuardConfig(results_dir=[a, b], tex=tex, start=r"\section{Results}")
    r = check_numbers(cfg)
    assert r.ok and r.n_files == 2


def test_several_tex_files_are_concatenated(tmp_path):
    """spectral-quantum splits its manuscript into sections/*.tex."""
    res = tmp_path / "results"
    res.mkdir()
    (res / "s.json").write_text(json.dumps({"v": 1.5, "w": 2.5}))
    one = tmp_path / "one.tex"
    one.write_text("\\section{Results}\n$1.5$\n")
    two = tmp_path / "two.tex"
    two.write_text("prose with no marker and a stray $77.7$\n")
    three = tmp_path / "three.tex"
    three.write_text("\\section{Results}\n$2.5$\n")
    cfg = GuardConfig(results_dir=res, tex=[one, two, three],
                      start=r"\section{Results}")
    # the marker-less file contributes nothing, and its 77.7 is not checked
    assert check_numbers(cfg).ok


def test_marker_absent_from_every_tex_file_is_an_error(tmp_path):
    """Missing from ALL files is a renamed section, not a section to skip."""
    res = tmp_path / "results"
    res.mkdir()
    (res / "s.json").write_text(json.dumps({"v": 1.5}))
    one = tmp_path / "one.tex"
    one.write_text("no marker here\n")
    two = tmp_path / "two.tex"
    two.write_text("nor here\n")
    cfg = GuardConfig(results_dir=res, tex=[one, two],
                      start=r"\section{Results}")
    with pytest.raises(ValueError, match="not found"):
        check_numbers(cfg)


def test_run_guards_ors_the_exit_codes(tmp_path):
    """One failing bound must fail the whole run, not be masked by a passing one."""
    res = tmp_path / "results"
    res.mkdir()
    (res / "s.json").write_text(json.dumps({"v": 1.5}))
    good = tmp_path / "good.tex"
    good.write_text("\\section{A}\n$1.5$\n")
    bad = tmp_path / "bad.tex"
    bad.write_text("\\section{B}\n$9.9$\n")
    cfgs = {
        "good": GuardConfig(results_dir=res, tex=good, start=r"\section{A}"),
        "bad": GuardConfig(results_dir=res, tex=bad, start=r"\section{B}"),
    }
    assert run_guards(cfgs) == 1
    assert run_guards({"good": cfgs["good"]}) == 0
