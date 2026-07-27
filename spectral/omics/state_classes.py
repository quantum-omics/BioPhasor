"""
omics.state_classes — Spectral state classes (theory.md §7).

Assigns one of seven state classes from the spectral-indicator panel
(R, H_spec, Δ_F, κ) and returns a candidate perturbation. This is the omics
analog of the Neural Perturbation Taxonomy (BCI) and the Market Regime Taxonomy
(Finance).

SPDX-License-Identifier: Apache-2.0
Copyright 2024-2026 Quantum Omics Foundation
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StateClass:
    code: str
    label: str
    intervention: str


# Seven classes (theory.md §7)
CLASSES = {
    "I":   StateClass("I",   "Healthy-synchronised",   "none (homeostatic)"),
    "II":  StateClass("II",  "Balanced-modular",       "monitor"),
    "III": StateClass("III", "Hyper-coupled",          "de-repress dominant compartment"),
    "IV":  StateClass("IV",  "Desynchronised",         "restore clock coupling"),
    "V":   StateClass("V",   "Fragmented",             "pathway-targeted rescue"),
    "VI":  StateClass("VI",  "Compartment-imbalanced", "modulate dominant compartment"),
    "VII": StateClass("VII", "Transitional",           "re-sample / observe"),
}


class SpectralStateClassifier:
    """Seven-class cellular/disease state classifier (theory.md §7).

    Parameters
    ----------
    R_hi, R_lo : float
        High/low thresholds on the Kuramoto order parameter R.
    H_hi, H_lo : float
        High/low thresholds on the normalised spectral entropy H_spec.
    kappa_hi, kappa_lo : float
        Coherence-κ thresholds for hyper-coupling / fragmentation.
    pi_dominant : float
        Single-compartment weight above which the state is imbalanced.
    """

    def __init__(
        self,
        R_hi: float = 0.7,
        R_lo: float = 0.4,
        H_hi: float = 0.7,
        H_lo: float = 0.4,
        kappa_hi: float = 0.6,
        kappa_lo: float = 0.3,
        pi_dominant: float = 0.5,
    ) -> None:
        self.R_hi, self.R_lo = R_hi, R_lo
        self.H_hi, self.H_lo = H_hi, H_lo
        self.kappa_hi, self.kappa_lo = kappa_hi, kappa_lo
        self.pi_dominant = pi_dominant

    # ------------------------------------------------------------------
    def classify(
        self,
        coherence_R: float,
        spectral_entropy: float,
        fiedler_gap: float,
        coherence_kappa: float,
        max_compartment_weight: float = 0.0,
    ) -> dict:
        """Return the state class for one indicator panel (theory.md §7).

        Decision order (first match wins):
          III Hyper-coupled          : R≥R_hi and κ≥κ_hi
          I   Healthy-synchronised   : R≥R_hi and H_spec≤H_lo
          IV  Desynchronised         : R<R_lo and H_spec≥H_hi
          V   Fragmented             : H_spec≥H_hi and κ≤κ_lo
          VI  Compartment-imbalanced : max π_a ≥ pi_dominant
          II  Balanced-modular       : R_lo≤R<R_hi and H_lo<H_spec<H_hi
          VII Transitional           : otherwise
        """
        R = float(coherence_R)
        H = float(spectral_entropy)
        kappa = float(coherence_kappa)
        pmax = float(max_compartment_weight)

        if R >= self.R_hi and kappa >= self.kappa_hi:
            code = "III"
        elif R >= self.R_hi and H <= self.H_lo:
            code = "I"
        elif R < self.R_lo and H >= self.H_hi:
            code = "IV"
        elif H >= self.H_hi and kappa <= self.kappa_lo:
            code = "V"
        elif pmax >= self.pi_dominant:
            code = "VI"
        elif (self.R_lo <= R < self.R_hi) and (self.H_lo < H < self.H_hi):
            code = "II"
        else:
            code = "VII"

        sc = CLASSES[code]
        return {
            "class": sc.code,
            "label": sc.label,
            "recommended_intervention": sc.intervention,
            "indicators": {
                "coherence_R": R,
                "spectral_entropy": H,
                "fiedler_gap": float(fiedler_gap),
                "coherence_kappa": kappa,
                "max_compartment_weight": pmax,
            },
        }
