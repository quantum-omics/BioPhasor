"""
rhythmicity_gate.py

Periodicity detection and rhythmicity gating for the two-layer state.

Purpose
-------
Only features that genuinely oscillate on a limit cycle referenced to the
circadian clock are placed on the torus as phasor state variables.
Non-rhythmic pools remain abundance variables in the base layer — they
participate in the dynamics but are NOT assigned a phase coordinate.

This directly enforces Design Rule R3 from 4-Regorous.ipynb:
  "Clock phase is both a model state and an observation coordinate, but
   ONLY for genuinely oscillatory features."

Method
------
We use Lomb-Scargle periodogram scoring centred on the circadian period
band (20–28 h). A node is classified rhythmic if:
  1. Its spectral power at the dominant period in [20, 28] h exceeds
     a threshold relative to the background spectrum (SNR > SNR_THRESH).
  2. Its median oscillatory amplitude exceeds AMP_THRESH × max amplitude
     in that layer (amplitude gating from the original phasor_extractor).

For synthetic data these criteria reduce to checking whether the node was
driven by the clock generator (is_rhy flag from the generator).
For real data, only the periodogram + amplitude check applies.

Outputs
-------
rhythmic_mask  : (N,)   bool array — True ↔ node is rhythmic
acrophase      : (N,)   float — estimated clock acrophase (radians), NaN for
                         non-rhythmic nodes
amplitude      : (N,)   float — median oscillatory amplitude (after bandpass)
period_h       : (N,)   float — dominant period in hours, NaN for non-rhythmic
"""

import numpy as np
from scipy.signal import lombscargle, butter, filtfilt, hilbert as sp_hilbert

# ─────────────────────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────────────────────

PERIOD_BAND_H  = (20.0, 28.0)   # circadian band in hours (default / back-compat)
SNR_THRESH     = 5.0            # minimum power-vs-background SNR for rhythmicity
AMP_THRESH     = 0.05           # minimum relative amplitude (fraction of layer max)
CIRCADIAN_H    = 24.0           # target clock period

# ── Clock bank (Phase C) ──────────────────────────────────────────────────────
# Per-clock detection bands.  Each clock is detected in a band bracketing its
# period; a node is assigned to the clock whose band contains its dominant
# period.  Bands are chosen NON-overlapping so the two clocks are separable.
try:
    from .compartments import CLOCK_BANK as _CLOCK_BANK
except ImportError:
    from data.compartments import CLOCK_BANK as _CLOCK_BANK

def _period_of(clock_name: str) -> float:
    return 2.0 * np.pi / _CLOCK_BANK[clock_name]

# Detection band per clock: ±10% around the clock period, split at the midpoint
# between clocks so bands do not overlap (circadian 24 h, redox 20 h → split ~22 h).
CLOCK_BANDS = {
    "circadian": (22.0, 26.5),   # brackets 24 h, above the split
    "redox":     (18.0, 22.0),   # brackets 20 h, below the split
}


def _bandpass(signal: np.ndarray, dt: float, band_h: tuple) -> np.ndarray:
    """Zero-phase 3rd-order Butterworth bandpass around a given period band."""
    fs  = 1.0 / dt          # samples per hour
    nyq = 0.5 * fs
    lo  = 1.0 / band_h[1]   # lower freq cutoff
    hi  = 1.0 / band_h[0]   # upper freq cutoff
    lo_n = np.clip(lo / nyq, 1e-4, 0.499)
    hi_n = np.clip(hi / nyq, 1e-4, 0.499)
    if lo_n >= hi_n:
        return signal
    b, a = butter(3, [lo_n, hi_n], btype="band")
    return filtfilt(b, a, signal, axis=-1)


def _bandpass_circadian(signal: np.ndarray, dt: float) -> np.ndarray:
    """Back-compat wrapper: bandpass around the default circadian band."""
    return _bandpass(signal, dt, PERIOD_BAND_H)


def _fit_acrophase(signal: np.ndarray, t: np.ndarray, omega: float) -> float:
    """
    Time-referenced acrophase at the clock frequency via least-squares fit of
    a cos/sin pair:  signal ≈ A cos(ωt) + B sin(ωt).  Acrophase = atan2(B, A) is
    the phase of the peak relative to t=0 — the correct reference for the
    cross-layer cascade lag (φ_target − φ_source).
    """
    x = signal - signal.mean()
    M = np.c_[np.cos(omega * t), np.sin(omega * t)]
    coef, _, _, _ = np.linalg.lstsq(M, x, rcond=None)
    return float(np.arctan2(coef[1], coef[0]))


def _lomb_scargle_score(signal: np.ndarray, t: np.ndarray) -> tuple:
    """
    Compute Lomb-Scargle periodogram and return:
      (dominant_period_h, snr, max_power)
    where SNR = max_power_in_band / median_background_power.
    """
    # Frequency grid: focus on 1/40 to 1/10 cycles/hour
    freqs = np.linspace(1.0 / 40.0, 1.0 / 10.0, 400)   # cycles/hour
    angular_freqs = 2.0 * np.pi * freqs

    sig_centered = signal - signal.mean()
    if sig_centered.std() < 1e-8:
        return np.nan, 0.0, 0.0

    pgram = lombscargle(t, sig_centered, angular_freqs, normalize=True)

    # Band mask: span the full clock bank (union of all clock detection bands),
    # so both circadian (~24 h) and redox (~20 h) oscillations are scored.
    band_lo_h = min(b[0] for b in CLOCK_BANDS.values())   # e.g. 18 h
    band_hi_h = max(b[1] for b in CLOCK_BANDS.values())   # e.g. 26.5 h
    in_band = (freqs >= 1.0 / band_hi_h) & (freqs <= 1.0 / band_lo_h)
    if not in_band.any():
        return np.nan, 0.0, 0.0

    max_power   = pgram[in_band].max()
    bkg_power   = np.median(pgram[~in_band]) if (~in_band).any() else 1e-8
    snr         = max_power / (bkg_power + 1e-8)
    peak_freq   = freqs[in_band][pgram[in_band].argmax()]
    period_h    = 1.0 / peak_freq

    return period_h, snr, max_power


def detect_rhythmicity(
    expression: np.ndarray,
    t: np.ndarray,
    dt: float,
    snr_thresh: float = SNR_THRESH,
    amp_thresh: float = AMP_THRESH,
) -> dict:
    """
    Classify each node in one omic layer as rhythmic or non-rhythmic.

    Parameters
    ----------
    expression : (N, T) abundance array
    t          : (T,) time axis in hours
    dt         : time step in hours
    snr_thresh : minimum Lomb-Scargle SNR to call rhythmic
    amp_thresh : minimum relative amplitude (fraction of layer max)

    Returns
    -------
    dict with:
      'rhythmic_mask' : (N,) bool
      'acrophase'     : (N,) float, NaN for non-rhythmic
      'amplitude'     : (N,) float — median bandpass amplitude
      'period_h'      : (N,) float, NaN for non-rhythmic
      'snr'           : (N,) float Lomb-Scargle SNR score
    """
    N = expression.shape[0]

    # Per-clock bandpass amplitude (used for both gating and clock assignment)
    clock_names   = list(CLOCK_BANDS.keys())
    filtered_by_c = {c: _bandpass(expression, dt, CLOCK_BANDS[c]) for c in clock_names}
    amp_by_c      = {c: np.std(filtered_by_c[c], axis=1) for c in clock_names}
    # Amplitude gate references the strongest per-node band
    amp_best      = np.maximum.reduce([amp_by_c[c] for c in clock_names])   # (N,)
    max_amp       = amp_best.max() + 1e-8
    amp_mask      = amp_best > amp_thresh * max_amp

    rhythmic    = np.zeros(N, dtype=bool)
    acrophase   = np.full(N, np.nan)
    period_h    = np.full(N, np.nan)
    snr_arr     = np.zeros(N)
    clock_label = np.array(["none"] * N, dtype=object)

    for i in range(N):
        p_h, snr, _ = _lomb_scargle_score(expression[i], t)   # dominant period + SNR
        snr_arr[i]  = snr
        if snr < snr_thresh or not amp_mask[i]:
            continue
        # Assign to the clock whose band contains the dominant period; if the
        # period falls between bands, pick the clock with the larger bandpass
        # amplitude (nearest oscillatory energy).
        assigned = None
        for c in clock_names:
            lo, hi = CLOCK_BANDS[c]
            if lo <= p_h <= hi:
                assigned = c
                break
        if assigned is None:
            assigned = max(clock_names, key=lambda c: amp_by_c[c][i])
        omega = _CLOCK_BANK[assigned]
        rhythmic[i]    = True
        period_h[i]    = p_h
        clock_label[i] = assigned
        acrophase[i]   = _fit_acrophase(expression[i], t, omega)   # time-referenced

    return {
        "rhythmic_mask": rhythmic,
        "acrophase":     acrophase,
        "amplitude":     amp_best,
        "period_h":      period_h,
        "snr":           snr_arr,
        "clock_label":   clock_label,   # (N,) "circadian"/"redox"/"none" (Phase C)
    }


def detect_all_layers(omics_data: dict) -> dict:
    """
    Run rhythmicity detection on all three omic layers.

    Parameters
    ----------
    omics_data : dict from generate_multi_omics()

    Returns
    -------
    dict: layer_name → rhythmicity dict (as returned by detect_rhythmicity)
    """
    t  = omics_data["t"]
    dt = omics_data["dt"]
    results = {}
    for layer_name, expr in omics_data["expression"].items():
        results[layer_name] = detect_rhythmicity(expr, t, dt)
    return results


def summarise_rhythmicity(gate_results: dict) -> None:
    """Print a summary of the rhythmicity classification per layer."""
    print("\n── Rhythmicity Summary ──────────────────────────────────")
    for layer, res in gate_results.items():
        mask = res["rhythmic_mask"]
        N    = len(mask)
        n_r  = mask.sum()
        frac = n_r / N
        mean_snr = res["snr"][mask].mean() if n_r > 0 else 0.0
        print(f"  {layer:12s}: {n_r:2d}/{N} rhythmic ({frac:.0%}), "
              f"mean SNR = {mean_snr:.1f}")
    print("─────────────────────────────────────────────────────────\n")
