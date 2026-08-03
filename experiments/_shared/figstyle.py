"""_figstyle.py -- publication figure style shared across BioPhasor experiments.

Mirrors the figure-style skill's role-mapped size ladder (8/7/6), outward ticks,
frameless legends, open frame (no top/right spine), 300-dpi output, and
editable (Type-42) fonts. Applied via _apply_style() so each experiment figure
is publication-grade AND reproducible standalone (no external skill dependency).
"""
import matplotlib as mpl


def apply_style():
    mpl.rcParams.update({
        "font.size": 9.0, "axes.titlesize": 9.0, "axes.labelsize": 9.0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 8.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.linewidth": 0.6, "figure.dpi": 200.0, "savefig.dpi": 300.0,
        "font.family": ["sans-serif"], "legend.frameon": False,
        "axes.titlelocation": "left", "axes.grid": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
