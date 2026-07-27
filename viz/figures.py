"""
viz.figures — Publication figures for the spectral-omics pipeline.

Self-contained matplotlib functions; each returns a Figure. Colour is threaded
consistently across figures via COMPARTMENT_COLORS and a small focal palette.
No dependency on kernel helpers, so the package stays importable stand-alone.

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from biophasor.spectral.omics.ccm import COMPARTMENTS

# ── colour threading (stable across all figures) ─────────────────────────────
COMPARTMENT_COLORS: Dict[str, str] = {
    "Clock":        "#4C72B0",   # blue
    "Redox":        "#DD8452",   # orange
    "Energy":       "#55A868",   # green
    "Signalling":   "#C44E52",   # red
    "Biosynthesis": "#8172B3",   # purple
}
# Focal hue for the leading harmonic / dominant mode / resultant vector.
# Deliberately distinct from every COMPARTMENT_COLORS hue (in particular not
# Signalling's #C44E52) so the focal mark never collides with a categorical
# compartment colour in the same figure (figure-style §4.2).
FOCAL = "#111111"     # near-black focal accent
MUTED = "#8899A6"     # comparator grey


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    return ax


# ── 1. phasor snapshot (polar) ───────────────────────────────────────────────
def plot_phasor_snapshot(psi: np.ndarray, title: str = "Phasor vertices",
                         color_by: Optional[Sequence[str]] = None):
    """Polar scatter of the phasor vertices ψ_i = r_i e^{iθ_i} for one slice.

    color_by : optional per-feature compartment labels for colour threading.
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    theta = np.angle(psi)
    r = np.abs(psi)
    fig = plt.figure(figsize=(4.2, 4.2))
    ax = fig.add_subplot(111, projection="polar")
    if color_by is not None:
        for comp in COMPARTMENTS:
            m = np.array([c == comp for c in color_by])
            if m.any():
                ax.scatter(theta[m], r[m], s=14, alpha=0.75,
                           color=COMPARTMENT_COLORS[comp], label=comp, edgecolors="none")
        ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.10), fontsize=7,
                  frameon=False)
    else:
        ax.scatter(theta, r, s=14, alpha=0.7, color=MUTED, edgecolors="none")
    # mean resultant vector (Kuramoto)
    z = np.mean(np.exp(1j * theta))
    ax.annotate("", xy=(np.angle(z), np.abs(z)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=FOCAL, lw=2))
    ax.set_title(f"{title}\nR = {np.abs(z):.2f}", fontsize=9, pad=14)
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ── 2. harmonic timeline ─────────────────────────────────────────────────────
def plot_harmonic_timeline(eigenvalue_series: np.ndarray, x: Optional[np.ndarray] = None,
                           n_show: int = 5, xlabel: str = "sample / time",
                           title: str = "Omics harmonic timeline"):
    """Leading-eigenvalue trajectories λ_n(t): first mode focal, rest muted."""
    E = np.asarray(eigenvalue_series, dtype=float)     # (T, k)
    T, k = E.shape
    if x is None:
        x = np.arange(T)
    n_show = min(n_show, k)
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for n in range(n_show - 1, 0, -1):
        ax.plot(x, E[:, n], color=MUTED, lw=1.0, alpha=0.55)
    ax.plot(x, E[:, 0], color=FOCAL, lw=2.2, label="λ₁ (dominant mode)")
    # direct label the leading line at its right end
    ax.annotate("λ₁", xy=(x[-1], E[-1, 0]), xytext=(4, 0),
                textcoords="offset points", color=FOCAL, fontsize=9, va="center")
    ax.annotate("λ₂…λ₅", xy=(x[-1], E[-1, 1]), xytext=(4, 0),
                textcoords="offset points", color=MUTED, fontsize=8, va="center")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("eigenvalue λ", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    ax.margins(x=0.02)
    fig.tight_layout()
    return fig


# ── 3. eigenvalue heatmap ────────────────────────────────────────────────────
def plot_eigenvalue_heatmap(eigenvalue_series: np.ndarray, x: Optional[np.ndarray] = None,
                            n_show: int = 15, xlabel: str = "sample / time",
                            title: str = "Spectral energy over samples"):
    """Heatmap of normalised spectral energy p_n(t) = |λ_n| / Σ|λ| (sequential map)."""
    E = np.abs(np.asarray(eigenvalue_series, dtype=float))
    P = E / (E.sum(axis=1, keepdims=True) + 1e-12)      # (T, k)
    n_show = min(n_show, P.shape[1])
    P = P[:, :n_show].T                                  # (n_show, T)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    im = ax.imshow(P, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("harmonic index n", fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("spectral energy pₙ", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    return fig


# ── 4. CCM heatmap ───────────────────────────────────────────────────────────
def plot_ccm_heatmap(M: np.ndarray, compartments: Sequence[str] = COMPARTMENTS,
                     title: str = "Compartment Coupling Matrix |M|"):
    """Heatmap of |M_ab| (magnitude of the 5×5 Hermitian CCM), value in each cell."""
    A = np.abs(np.asarray(M, dtype=complex))
    n = A.shape[0]
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    im = ax.imshow(A, cmap="magma")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(compartments, rotation=40, ha="right", fontsize=7)
    ax.set_yticklabels(compartments, fontsize=7)
    thr = A.max() * 0.5
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if A[i, j] < thr else "black")
    ax.set_title(title, fontsize=10, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("|M_ab|", fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


# ── 5. CompartmentWeights weights ─────────────────────────────────────────────────────
def plot_compartment_weights(weights: Dict[str, float], kappa: float,
                           title: str = "CompartmentWeights compartment weights"):
    """Lollipop of compartment weights π_a, coloured by compartment, κ in title."""
    comps = list(weights.keys())
    vals = np.array([weights[c] for c in comps])
    order = np.argsort(vals)
    comps = [comps[i] for i in order]; vals = vals[order]
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    y = np.arange(len(comps))
    for yi, c, v in zip(y, comps, vals):
        ax.hlines(yi, 0, v, color=COMPARTMENT_COLORS.get(c, MUTED), lw=2, alpha=0.8)
        ax.plot(v, yi, "o", color=COMPARTMENT_COLORS.get(c, MUTED), ms=9)
        ax.annotate(f"{v:.2f}", xy=(v, yi), xytext=(5, 0), textcoords="offset points",
                    va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(comps, fontsize=8)
    ax.set_xlabel("compartment weight πₐ", fontsize=9)
    ax.set_title(f"{title}   (κ = {kappa:.2f})", fontsize=10, loc="left")
    _style(ax); ax.margins(x=0.12)
    fig.tight_layout()
    return fig


# ── 6. indicator dashboard ───────────────────────────────────────────────────
def plot_indicator_dashboard(indicator_series: Dict[str, np.ndarray],
                             x: Optional[np.ndarray] = None,
                             xlabel: str = "sample / time",
                             title: str = "Spectral indicators"):
    """Small-multiples of the indicator panel over samples (shared x)."""
    # algebraic_connectivity depends only on the shared coupling (constant across
    # slices), so it carries no per-slice signal and is omitted from the dashboard.
    keys = [k for k in ["coherence_R", "spectral_entropy", "fiedler_gap",
                        "participation_ratio", "mode_localisation"]
            if k in indicator_series and np.ptp(np.asarray(indicator_series[k])) > 1e-9]
    n = len(keys)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4, 1.7 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    labels = {"coherence_R": "Kuramoto R", "spectral_entropy": "spectral entropy",
              "fiedler_gap": "Fiedler gap Δ_F", "participation_ratio": "participation ratio",
              "algebraic_connectivity": "algebraic connectivity", "mode_localisation": "mode localisation"}
    for i, k in enumerate(keys):
        v = np.asarray(indicator_series[k], dtype=float)
        xx = np.arange(len(v)) if x is None else x
        axes[i].plot(xx, v, color=FOCAL if k == "coherence_R" else "#33608C", lw=1.6)
        axes[i].set_title(labels.get(k, k), fontsize=8, loc="left")
        _style(axes[i]); axes[i].margins(x=0.02)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    for i in range(n):
        if i >= n - ncol:
            axes[i].set_xlabel(xlabel, fontsize=8)
    fig.suptitle(title, fontsize=10, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ── 7. state_class history ──────────────────────────────────────────────────────
def plot_state_history(classes: Sequence[str], x: Optional[np.ndarray] = None,
                          xlabel: str = "sample / time",
                          title: str = "Cellular/disease state class"):
    """Step plot of the 7-class state_class code over samples."""
    order = ["I", "II", "III", "IV", "V", "VI", "VII"]
    idx = np.array([order.index(c) if c in order else len(order) for c in classes])
    xx = np.arange(len(idx)) if x is None else np.asarray(x)
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.step(xx, idx, where="mid", color="#33608C", lw=1.6)
    ax.scatter(xx, idx, s=16, color=FOCAL, zorder=3)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8)
    ax.set_ylabel("state class", fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(title, fontsize=10, loc="left")
    _style(ax); ax.margins(x=0.02)
    fig.tight_layout()
    return fig


# ── 8. workflow schematic ────────────────────────────────────────────────────
def plot_workflow(title: str = "Spectral-Omics pipeline"):
    """Simple boxed-flow schematic of the pipeline stages."""
    stages = ["Omics\nmatrix X", "Phasor\nvertices ψ", "OCM\nH = c·e^{iΔθ}",
              "Omics\nharmonics", "CCM 5×5\n+ weights", "State\nclasses"]
    fig, ax = plt.subplots(figsize=(7.6, 1.9))
    ax.axis("off")
    n = len(stages)
    xs = np.linspace(0.06, 0.94, n)
    w = 0.115
    for i, (xc, s) in enumerate(zip(xs, stages)):
        col = FOCAL if i in (1, 2, 3) else "#33608C"
        ax.add_patch(mpl.patches.FancyBboxPatch((xc - w / 2, 0.35), w, 0.32,
                     boxstyle="round,pad=0.02", fc="white", ec=col, lw=1.6))
        ax.text(xc, 0.51, s, ha="center", va="center", fontsize=7.5)
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - w / 2, 0.51), xytext=(xc + w / 2, 0.51),
                        arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    return fig
