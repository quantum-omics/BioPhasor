"""
viz.figures — self-contained publication figures for the omics quantum model.

Every function is self-contained (imports matplotlib internally, builds and
returns a :class:`matplotlib.figure.Figure`; it does not call ``plt.show`` or
save to disk — the caller saves). A single compartment colour palette is
threaded consistently across all figures so that the same colour always denotes
the same compartment (Clock, Redox, Energy, Signalling, Biosynthesis).

Figures
-------
* ``fock_occupation``      — mean per-compartment occupation ⟨n_k⟩ of a state.
* ``quantum_evolution``    — ⟨n_k(t)⟩ real-time trajectories.
* ``entropy_evolution``    — bipartite entanglement entropy S(t).
* ``ccm_heatmap``          — the 5x5 compartment covariance matrix.
* ``compartment_weights``  — weight readout (bar + radar).
* ``dmrg_scaling``         — exact sector dimension vs DMRG bond dimension.

This is a quantum-simulable signal-processing model, NOT a biological claim.
SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# The five omics compartments and their fixed display colours. The ordering
# matches ``compartments.compartment_model.COMPARTMENTS``.
COMPARTMENTS = ["Clock", "Redox", "Energy", "Signalling", "Biosynthesis"]

# A colourblind-safe categorical palette (Okabe-Ito-derived); one hue per
# compartment, reused for every mark denoting that compartment.
COMPARTMENT_COLORS = {
    "Clock":        "#0072B2",   # blue
    "Redox":        "#D55E00",   # vermillion
    "Energy":       "#009E73",   # green
    "Signalling":   "#CC79A7",   # purple/pink
    "Biosynthesis": "#E69F00",   # orange
}
_COLORS = [COMPARTMENT_COLORS[c] for c in COMPARTMENTS]

_ALARM = "#B00020"       # reserved for reference/exact marks, never a compartment


def _fig(figsize):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


# ---------------------------------------------------------------------------
# 1) Fock occupation
# ---------------------------------------------------------------------------
def fock_occupation(occupations: Sequence[float],
                    labels: Sequence[str] = COMPARTMENTS,
                    title: str = "Mean compartment occupation"):
    """Bar chart of the mean per-compartment occupation ``⟨n_k⟩``.

    Parameters
    ----------
    occupations : length-5 array of ⟨n_k⟩ values.
    """
    occ = np.asarray(occupations, dtype=float).ravel()
    fig, ax = _fig((6.4, 4.2))
    x = np.arange(len(labels))
    ax.bar(x, occ, color=[COMPARTMENT_COLORS[c] for c in labels],
           edgecolor="black", linewidth=0.6, width=0.7)
    for xi, v in zip(x, occ):
        ax.text(xi, v + 0.01 * max(occ.max(), 1e-9), f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(r"$\langle n_k \rangle$")
    ax.set_title(title)
    ax.margins(y=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2) Quantum evolution of occupations
# ---------------------------------------------------------------------------
def quantum_evolution(times: Sequence[float],
                      occupations: np.ndarray,
                      labels: Sequence[str] = COMPARTMENTS,
                      title: str = "Real-time compartment occupation"):
    """Line plot of ``⟨n_k(t)⟩`` for each compartment.

    Parameters
    ----------
    times : (T,) time grid.
    occupations : (T, 5) array; column k is ⟨n_k(t)⟩.
    """
    t = np.asarray(times, dtype=float).ravel()
    Y = np.asarray(occupations, dtype=float)
    if Y.shape[0] != t.size and Y.shape[1] == t.size:
        Y = Y.T
    fig, ax = _fig((6.8, 4.2))
    for k, lab in enumerate(labels):
        ax.plot(t, Y[:, k], color=COMPARTMENT_COLORS[lab], lw=2.0, label=lab)
    # de-collide end-of-line labels: sort by final value and enforce a minimum
    # vertical gap so close-lying series (e.g. Redox vs Signalling) stay legible.
    order = sorted(range(len(labels)), key=lambda k: Y[-1, k])
    span = float(Y.max() - Y.min()) or 1.0
    min_gap = 0.055 * span
    x_lab = t[-1] + 0.02 * (t[-1] - t[0])
    y_prev = -np.inf
    for k in order:
        y_lab = max(Y[-1, k], y_prev + min_gap)
        ax.annotate(labels[k], xy=(t[-1], Y[-1, k]), xytext=(x_lab, y_lab),
                    textcoords="data", va="center", fontsize=8,
                    color=COMPARTMENT_COLORS[labels[k]], annotation_clip=False)
        y_prev = y_lab
    ax.set_xlabel("time  $t$")
    ax.set_ylabel(r"$\langle n_k(t) \rangle$")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1] + 0.14 * (t[-1] - t[0]))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3) Entanglement entropy evolution
# ---------------------------------------------------------------------------
def entropy_evolution(times: Sequence[float],
                      entropy: Sequence[float],
                      title: str = "Bipartite entanglement entropy"):
    """Line plot of the bipartite entanglement entropy ``S(t)``."""
    t = np.asarray(times, dtype=float).ravel()
    S = np.asarray(entropy, dtype=float).ravel()
    fig, ax = _fig((6.8, 4.0))
    ax.plot(t, S, color="#333333", lw=2.0)
    ax.fill_between(t, 0, S, color="#333333", alpha=0.10)
    imax = int(np.argmax(S))
    ax.annotate(f"max $S$ = {S[imax]:.2f}", xy=(t[imax], S[imax]),
                xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=8)
    ax.set_xlabel("time  $t$")
    ax.set_ylabel(r"$S = -\mathrm{Tr}\,\rho_A \ln \rho_A$")
    ax.set_title(title)
    ax.set_xlim(t[0], t[-1])
    ax.margins(y=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4) CCM heatmap
# ---------------------------------------------------------------------------
def ccm_heatmap(ccm: np.ndarray,
                labels: Sequence[str] = COMPARTMENTS,
                title: str = "Compartment covariance matrix (CCM)"):
    """Annotated heatmap of the 5x5 compartment covariance matrix.

    The CCM is symmetric and positive-semi-definite with a non-negative
    diagonal, but its off-diagonal covariances are signed. A diverging map
    centered at the semantic zero (no covariance) is therefore used, with a
    symmetric range +/- max|M|, and every cell value is printed (only 25 cells).
    """
    M = np.asarray(ccm, dtype=float)
    fig, ax = _fig((5.8, 5.0))
    vmax = np.abs(M).max()
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # white text only on the darkest cells at either end of the ramp
            dark = abs(M[i, j]) > 0.6 * vmax
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if dark else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$M_{ab} = \frac{1}{2}\langle\{H_a,H_b\}\rangle "
                 r"- \langle H_a\rangle\langle H_b\rangle$  (0 = no covariance)",
                 fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5) Compartment weights (bar + radar)
# ---------------------------------------------------------------------------
def compartment_weights(weights: dict,
                        title: str = "Compartment-weight readout"):
    """Compartment-weight profile as a bar chart (left) and a radar/spider
    plot (right).

    Parameters
    ----------
    weights : dict compartment -> weight (non-negative, sums to 1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [c for c in COMPARTMENTS if c in weights] or list(weights.keys())
    w = np.array([weights[c] for c in labels], dtype=float)

    fig = plt.figure(figsize=(10.5, 4.4))
    ax1 = fig.add_subplot(1, 2, 1)
    x = np.arange(len(labels))
    ax1.bar(x, w, color=[COMPARTMENT_COLORS[c] for c in labels],
            edgecolor="black", linewidth=0.6, width=0.7)
    for xi, v in zip(x, w):
        ax1.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("weight  $w_a$")
    ax1.set_title("Weights")
    ax1.margins(y=0.14)
    ax1.spines[["top", "right"]].set_visible(False)

    # radar
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    ang_c = np.concatenate([ang, ang[:1]])
    w_c = np.concatenate([w, w[:1]])
    ax2.plot(ang_c, w_c, color="#333333", lw=1.8)
    ax2.fill(ang_c, w_c, color="#0072B2", alpha=0.18)
    ax2.scatter(ang, w, c=[COMPARTMENT_COLORS[c] for c in labels],
                s=48, zorder=5, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(ang)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_yticklabels([])
    ax2.set_title("Radar")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# ---------------------------------------------------------------------------
# 6) DMRG scaling
# ---------------------------------------------------------------------------
def dmrg_scaling(scan: Sequence[dict],
                 title: str = "Tensor-network scaling of the compartment model"):
    """Exact fixed-N sector dimension vs DMRG bond dimension.

    Parameters
    ----------
    scan : list of dicts with keys ``N``, ``ed_dim``, ``chi`` (as produced by
        ``run_dmrg.run``).
    """
    Ns = [r["N"] for r in scan]
    dims = [r["ed_dim"] for r in scan]
    chis = [r["chi"] for r in scan]
    fig, ax = _fig((7.0, 4.6))
    ax.plot(Ns, dims, "o-", color=_ALARM, lw=2.0, ms=7,
            label=r"exact sector dim  $\binom{N+4}{4}$")
    ax.plot(Ns, chis, "s-", color="#0072B2", lw=2.0, ms=7,
            label=r"DMRG bond dim  $\chi$")
    for n, d in zip(Ns, dims):
        ax.annotate(str(d), (n, d), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color=_ALARM)
    for n, c in zip(Ns, chis):
        ax.annotate(str(c), (n, c), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7, color="#0072B2")
    ax.set_yscale("log")
    ax.set_xlabel("total excitation number  $N$")
    ax.set_ylabel("dimension")
    ax.set_xticks(Ns)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(True, which="both", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
