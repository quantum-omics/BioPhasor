"""
biophasor.visualization.phasor_plot — Core phasor visualisation functions.

Provides:
  - PhasorPlot  : classic G vs S scatter on unit semicircle
  - plot_polar_histogram : phase rose diagram
  - plot_coherence_bar   : per-layer coherence bar chart

Color palette: NavyDeep (#0D1B2A) / TealAccent (#00B4D8) as in the Book.

(c) 2026 Mindverse Computing LLC. Licensed under CC BY-NC 4.0.
"""

from __future__ import annotations
from typing import Optional, Union

import numpy as np


# ── Color palette (matches Book LaTeX color scheme) ──────────────────────────
PALETTE = {
    "background":  "#0D1B2A",
    "primary":     "#00B4D8",
    "secondary":   "#90E0EF",
    "accent":      "#F4A261",
    "G1":          "#4CAF50",
    "S":           "#2196F3",
    "G2":          "#FF9800",
    "M":           "#F44336",
}

LAYER_COLORS = ["#00B4D8", "#F4A261", "#A8DADC", "#E76F51", "#457B9D", "#2A9D8F"]


class PhasorPlot:
    """
    Classic biological phasor plot: G vs S on the unit semicircle.

    In FLIM-style phasor analysis:
        G = A·cos(φ)    x-axis
        S = A·sin(φ)    y-axis
    All phasors from non-negative intensities lie on or inside the semicircle
    S ≥ 0,  G²+S² ≤ 1.

    Parameters
    ----------
    figsize : tuple
    style : str  'dark' or 'light'
    """

    def __init__(
        self,
        figsize: tuple = (7, 5),
        style: str = "dark",
    ) -> None:
        self.figsize = figsize
        self.style = style

    def plot(
        self,
        phase: np.ndarray,
        amplitude: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        label_map: Optional[dict] = None,
        title: str = "Phasor Plot",
        ax=None,
    ):
        """
        Scatter plot of phasors on the unit semicircle.

        Parameters
        ----------
        phase : np.ndarray, shape (n_samples,) or (n_samples, n_features)
            If 2D, the mean phase per sample is used.
        amplitude : np.ndarray | None
        labels : np.ndarray | None   class labels for colour-coding
        label_map : dict | None   {label: name} for legend
        title : str
        ax : matplotlib Axes | None

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if self.style == "dark":
            plt.style.use("dark_background")
        else:
            plt.style.use("default")

        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure

        # Compute G, S from mean phase
        if phase.ndim == 2:
            # Average over features using circular mean
            z = np.exp(1j * phase).mean(axis=1)
            G = z.real
            S = np.abs(z.imag)
        else:
            A = amplitude if amplitude is not None else np.ones_like(phase)
            G = A * np.cos(phase)
            S = A * np.abs(np.sin(phase))

        # ── Draw unit semicircle ──────────────────────────────────────────────
        theta = np.linspace(0, np.pi, 300)
        ax.plot(np.cos(theta), np.sin(theta), color=PALETTE["secondary"],
                lw=1.5, alpha=0.6, zorder=1)
        ax.axhline(0, color=PALETTE["secondary"], lw=0.8, ls="--", alpha=0.4)
        ax.axvline(0, color=PALETTE["secondary"], lw=0.8, ls="--", alpha=0.4)

        # ── Scatter ───────────────────────────────────────────────────────────
        if labels is not None:
            unique_labels = np.unique(labels)
            for i, lbl in enumerate(unique_labels):
                mask = labels == lbl
                color = LAYER_COLORS[i % len(LAYER_COLORS)]
                name = label_map.get(lbl, str(lbl)) if label_map else str(lbl)
                ax.scatter(G[mask], S[mask], s=18, alpha=0.7, color=color,
                           label=name, zorder=3)
            ax.legend(fontsize=9, framealpha=0.3)
        else:
            ax.scatter(G, S, s=18, alpha=0.7, color=PALETTE["primary"], zorder=3)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-0.15, 1.15)
        ax.set_aspect("equal")
        ax.set_xlabel("G = A·cos(φ)", fontsize=11)
        ax.set_ylabel("S = A·sin(φ)", fontsize=11)
        ax.set_title(title, fontsize=13, pad=10)

        plt.tight_layout()
        return fig


def plot_phasor(
    phase: np.ndarray,
    amplitude: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    label_map: Optional[dict] = None,
    title: str = "Phasor Plot",
    figsize: tuple = (7, 5),
    style: str = "dark",
    save_path: Optional[str] = None,
):
    """
    Convenience wrapper around PhasorPlot.

    Returns matplotlib Figure.
    """
    pp = PhasorPlot(figsize=figsize, style=style)
    fig = pp.plot(phase, amplitude=amplitude, labels=labels,
                  label_map=label_map, title=title)
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_polar_histogram(
    phase: np.ndarray,
    n_bins: int = 36,
    title: str = "Phase Distribution",
    figsize: tuple = (5, 5),
    color: str = PALETTE["primary"],
    save_path: Optional[str] = None,
):
    """
    Rose diagram (polar histogram) of phase distribution.

    Parameters
    ----------
    phase : np.ndarray, shape (N,)   phase values ∈ (−π, π]
    n_bins : int
    title : str
    figsize : tuple
    color : str
    save_path : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    counts, _ = np.histogram(phase, bins=bin_edges)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = 2 * np.pi / n_bins

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.bar(bin_centers, counts, width=width, alpha=0.8, color=color, edgecolor="white", lw=0.5)
    ax.set_title(title, pad=20, fontsize=12)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def plot_coherence_bar(
    coherence_dict: dict[str, float],
    title: str = "Per-Layer Coherence",
    figsize: tuple = (6, 4),
    save_path: Optional[str] = None,
):
    """
    Bar chart of per-modality phase coherence.

    Parameters
    ----------
    coherence_dict : dict[str, float]   {modality: coherence}
    title : str
    figsize : tuple
    save_path : str | None

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    names = list(coherence_dict.keys())
    values = list(coherence_dict.values())
    colors = LAYER_COLORS[:len(names)]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values, color=colors, alpha=0.85, edgecolor="white", lw=0.7)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Coherence C = |mean(e^{iφ})|", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.axhline(1.0, color="white", lw=0.5, ls="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig
