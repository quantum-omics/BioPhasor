"""
cascade_predictor.py

The Falsifiable Phase Cascade Prediction Test.

Purpose
───────
This is the framework's SIGNATURE PREDICTION — the one test that
distinguishes the GNN-pHNN from RNA-velocity and Waddington-landscape models.

The prediction (from 4-Regorous.ipynb §2.6, §6.5):
  A rhythmic mRNA drives its protein through a first-order production-
  degradation pool.  Such a pool is a low-pass filter that phase-delays
  its rhythmic input by an amount FIXED by the pool's degradation rate:

      tan(Δφ_{mRNA→protein}) = ω_clock / k_deg

  where ω_clock is the circadian angular frequency and k_deg is the
  protein's degradation/turnover rate — an ENTRY OF THE DISSIPATION R.

WHY THIS IS FALSIFIABLE:
  The predicted lags come from INDEPENDENTLY MEASURED k_deg values (not
  fit to the observed lags).  If they systematically disagree with the
  observed acrophase differences, the pH dissipation interpretation of
  the cascade is wrong.  RNA-velocity makes no such quantitative lag
  prediction — so agreement would be genuinely distinguishing.

Usage
─────
  predictor = CascadePredictor(k_deg_G, k_deg_P, k_deg_M, omega_clock)
  predictions = predictor.predict_gp_lags()   # transcript→protein lags
  observed    = predictor.extract_observed_lags(acrophase_G, acrophase_P)
  report      = predictor.evaluate(predictions, observed)

  # Or use the model's R diagonal (learned values):
  predictions_learned = predictor.predict_from_R_diag(R_diag)

Design reference: 4-Regorous.ipynb §2.6, §4.1-4.3, §6.5
"""

import numpy as np
import torch
from typing import Optional


class CascadePredictor:
    """
    First-order pool phase-lag predictor.

    For a rhythmic pool driven at frequency ω_clock with degradation rate
    k_deg, the steady-state phase lag relative to the driving signal is:
        Δφ = arctan(ω_clock / k_deg)       [radians]
        Δφ_h = arctan(ω_clock / k_deg) / ω_clock  [hours]

    Notes on the prediction:
      - Short-lived proteins (large k_deg): small lag  → tracks mRNA closely
      - Long-lived proteins (small k_deg): large lag   → substantially delayed
      - This dose-response is the theory's explicit prediction.

    Parameters
    ----------
    k_deg_G      : (n_G,) mRNA degradation rates (rad/h or 1/h)
    k_deg_P      : (n_P,) protein degradation rates
    k_deg_M      : (n_M,) metabolite turnover rates
    omega_clock  : scalar, circadian angular frequency (default: 2π/24 rad/h)
    """

    def __init__(
        self,
        k_deg_G:     np.ndarray,
        k_deg_P:     np.ndarray,
        k_deg_M:     np.ndarray,
        omega_clock: float = 2.0 * np.pi / 24.0,
        clock_bank:  Optional[dict] = None,
    ):
        self.k_deg_G     = np.asarray(k_deg_G)
        self.k_deg_P     = np.asarray(k_deg_P)
        self.k_deg_M     = np.asarray(k_deg_M)
        self.omega_clock = omega_clock          # circadian (primary) — back-compat
        # Clock bank (Phase C): per-clock angular frequencies for the dual
        # cascade test.  Defaults to a single circadian clock if not supplied.
        self.clock_bank  = clock_bank if clock_bank is not None else {
            "circadian": omega_clock
        }

    # ── Phase lag from k_deg ────────────────────────────────────────────────

    def _lag_from_k(self, k_deg: np.ndarray, omega: Optional[float] = None) -> np.ndarray:
        """
        Δφ = arctan(ω / k_deg) in radians (positive = lag behind driver).

        omega : driving clock angular frequency; defaults to the primary
                (circadian) clock for backward compatibility.
        """
        w = self.omega_clock if omega is None else omega
        return np.arctan(w / (k_deg + 1e-8))

    def predict_G_lags(self) -> np.ndarray:
        """Predicted mRNA lags relative to the Zeitgeber (circadian input)."""
        return self._lag_from_k(self.k_deg_G)   # (n_G,)

    def predict_P_lags(self) -> np.ndarray:
        """Predicted protein lags relative to their driving mRNA."""
        return self._lag_from_k(self.k_deg_P)   # (n_P,)

    def predict_M_lags(self) -> np.ndarray:
        """Predicted metabolite lags relative to their enzyme."""
        return self._lag_from_k(self.k_deg_M)   # (n_M,)

    def predict_total_P_lag(self) -> np.ndarray:
        """
        Total mRNA→protein lag = mRNA lag + protein lag.
        For node pairs (i, j) where gene i → protein j (central dogma),
        the total lag is the cascade:
            Δφ_total = Δφ_G[i] + Δφ_P[j]
        Returns (min(n_G, n_P),) array for the diagonal pairs.
        """
        n_pairs = min(len(self.k_deg_G), len(self.k_deg_P))
        lag_G   = self.predict_G_lags()[:n_pairs]
        lag_P   = self.predict_P_lags()[:n_pairs]
        return lag_G + lag_P   # cascade

    def predict_from_R_diag(
        self,
        R_diag: np.ndarray,    # (N_total,) diagonal of R learned from model
        n_G:    int,
        n_P:    int,
        n_M:    int,
    ) -> dict:
        """
        Use the MODEL'S learned R diagonal to predict lags.
        This tests consistency: do the learned dissipation rates predict lags
        that match the observed acrophase differences?

        Parameters
        ----------
        R_diag : diagonal entries of R(x) averaged over the trajectory

        Returns
        -------
        dict with predicted lags for G, P, M layers
        """
        k_G = R_diag[:n_G]
        k_P = R_diag[n_G:n_G+n_P]
        k_M = R_diag[n_G+n_P:n_G+n_P+n_M]
        return {
            "lag_G_rad":       self._lag_from_k(k_G),
            "lag_P_rad":       self._lag_from_k(k_P),
            "lag_M_rad":       self._lag_from_k(k_M),
            "lag_G_hours":     self._lag_from_k(k_G) / self.omega_clock,
            "lag_P_hours":     self._lag_from_k(k_P) / self.omega_clock,
            "lag_M_hours":     self._lag_from_k(k_M) / self.omega_clock,
        }

    # ── Observed lag extraction ─────────────────────────────────────────────

    @staticmethod
    def extract_observed_gp_lags(
        acrophase_G: np.ndarray,   # (n_G,) observed acrophase (radians)
        acrophase_P: np.ndarray,   # (n_P,) observed acrophase
    ) -> np.ndarray:
        """
        Compute observed transcript→protein lag for diagonal gene–protein pairs.
        Lag = acrophase_P[i] - acrophase_G[i]  (wrapped to [−π, π]).

        Parameters
        ----------
        acrophase_G : (n_G,) acrophase of mRNA peaks
        acrophase_P : (n_P,) acrophase of protein peaks (matched by index)

        Returns
        -------
        (min(n_G, n_P),) observed lags in radians
        """
        n_pairs = min(len(acrophase_G), len(acrophase_P))
        # Filter out NaN (non-rhythmic nodes)
        valid   = (~np.isnan(acrophase_G[:n_pairs])) & (~np.isnan(acrophase_P[:n_pairs]))
        raw_lag = acrophase_P[:n_pairs] - acrophase_G[:n_pairs]
        # Wrap to [−π, π]
        lag     = np.where(valid, ((raw_lag + np.pi) % (2 * np.pi)) - np.pi, np.nan)
        return lag   # (n_pairs,)

    # ── Evaluation ──────────────────────────────────────────────────────────

    def evaluate_gp_cascade(
        self,
        acrophase_G: np.ndarray,
        acrophase_P: np.ndarray,
        k_deg_P_independent: Optional[np.ndarray] = None,
    ) -> dict:
        """
        The falsifiable test: compare INDEPENDENTLY predicted lags with
        OBSERVED acrophase differences.

        Steps:
          1. Observed lags from data: acrophase_P − acrophase_G.
          2. Predicted lags from k_deg (independent of observed lags).
          3. Correlation test: does predicted match observed?
          4. Dose-response test: do long-lived proteins show larger lags?

        Parameters
        ----------
        acrophase_G          : (n_G,) observed mRNA acrophase
        acrophase_P          : (n_P,) observed protein acrophase
        k_deg_P_independent  : (n_P,) protein degradation rates.
                               If None, uses self.k_deg_P (from generator).

        Returns
        -------
        dict with correlation, RMSE, dose-response test, and per-node data
        """
        if k_deg_P_independent is None:
            k_deg_P_independent = self.k_deg_P

        n_pairs = min(len(acrophase_G), len(acrophase_P),
                      len(k_deg_P_independent))

        observed  = self.extract_observed_gp_lags(acrophase_G, acrophase_P)[:n_pairs]
        predicted = self._lag_from_k(k_deg_P_independent[:n_pairs])   # independent

        # Filter valid pairs (both rhythmic)
        valid = ~np.isnan(observed)
        n_v   = valid.sum()

        if n_v < 5:
            return {
                "n_valid_pairs": int(n_v),
                "note": "Too few rhythmic pairs for reliable cascade test.",
            }

        obs_v  = observed[valid]
        pred_v = predicted[valid]

        # Pearson correlation
        from scipy.stats import pearsonr, spearmanr
        r_pearson,  p_pearson  = pearsonr(pred_v,  obs_v)
        r_spearman, p_spearman = spearmanr(pred_v, obs_v)
        rmse = np.sqrt(np.mean((pred_v - obs_v) ** 2))

        # Dose-response: stratify by protein half-life
        k_v         = k_deg_P_independent[:n_pairs][valid]
        half_life_h = np.log(2) / (k_v + 1e-8)
        short_mask  = half_life_h < np.median(half_life_h)
        long_mask   = ~short_mask
        mean_lag_short = obs_v[short_mask].mean() if short_mask.sum() > 0 else np.nan
        mean_lag_long  = obs_v[long_mask].mean()  if long_mask.sum()  > 0 else np.nan

        report = {
            "n_valid_pairs":       int(n_v),
            "pearson_r":           float(r_pearson),
            "pearson_p":           float(p_pearson),
            "spearman_r":          float(r_spearman),
            "spearman_p":          float(p_spearman),
            "rmse_rad":            float(rmse),
            "rmse_hours":          float(rmse / self.omega_clock),
            "mean_obs_lag_rad":    float(obs_v.mean()),
            "mean_pred_lag_rad":   float(pred_v.mean()),
            # Dose-response: theory predicts mean_lag_long > mean_lag_short
            "mean_lag_short_lived_h": float(mean_lag_short / self.omega_clock),
            "mean_lag_long_lived_h":  float(mean_lag_long  / self.omega_clock),
            "dose_response_direction_correct": bool(mean_lag_long > mean_lag_short),
            "observed_lags_rad":   obs_v.tolist(),
            "predicted_lags_rad":  pred_v.tolist(),
            "k_deg_P":             k_v.tolist(),
            "half_lives_h":        half_life_h[valid].tolist() if valid.any() else [],
        }
        return report

    # ── Generic per-clock cascade (Phase C: dual cascade) ───────────────────

    def evaluate_cascade(
        self,
        acrophase_src:  np.ndarray,   # (n,) source-layer acrophase (radians)
        acrophase_tgt:  np.ndarray,   # (n,) target-layer acrophase (radians), matched by index
        k_deg_tgt:      np.ndarray,   # (n,) target-pool degradation rates (independent)
        omega:          float,        # driving clock angular frequency (rad/h)
        label:          str = "cascade",
    ) -> dict:
        """
        Per-clock falsifiable cascade test for an arbitrary source→target pair.

        Generalizes evaluate_gp_cascade to any clock: the predicted lag comes
        from arctan(omega / k_deg_tgt) using the SPECIFIED clock frequency, and
        is compared against the observed acrophase difference
        (acrophase_tgt − acrophase_src).  Both the circadian transcript→protein
        cascade (omega = ω_circadian) and the redox protein→metabolite cascade
        (omega = ω_redox) are evaluated with this one method — predicted from
        the SAME dissipation R but DIFFERENT clock frequencies.

        Returns the same report schema as evaluate_gp_cascade, plus 'label' and
        'omega'.
        """
        n = min(len(acrophase_src), len(acrophase_tgt), len(k_deg_tgt))
        src = np.asarray(acrophase_src[:n], dtype=float)
        tgt = np.asarray(acrophase_tgt[:n], dtype=float)
        kdt = np.asarray(k_deg_tgt[:n], dtype=float)

        # Observed lag = tgt − src, wrapped to [−π, π]
        valid   = (~np.isnan(src)) & (~np.isnan(tgt))
        raw_lag = tgt - src
        observed = np.where(valid, ((raw_lag + np.pi) % (2 * np.pi)) - np.pi, np.nan)

        predicted = self._lag_from_k(kdt, omega=omega)   # independent of observed

        v = ~np.isnan(observed)
        n_v = int(v.sum())
        if n_v < 5:
            return {"label": label, "omega": float(omega),
                    "n_valid_pairs": n_v,
                    "note": f"Too few rhythmic pairs for {label} cascade test."}

        obs_v, pred_v = observed[v], predicted[v]
        from scipy.stats import pearsonr, spearmanr
        r_pearson,  p_pearson  = pearsonr(pred_v,  obs_v)
        r_spearman, p_spearman = spearmanr(pred_v, obs_v)
        rmse = float(np.sqrt(np.mean((pred_v - obs_v) ** 2)))

        k_v = kdt[v]
        half_life_h = np.log(2) / (k_v + 1e-8)
        short_mask  = half_life_h < np.median(half_life_h)
        long_mask   = ~short_mask
        mean_lag_short = obs_v[short_mask].mean() if short_mask.sum() > 0 else np.nan
        mean_lag_long  = obs_v[long_mask].mean()  if long_mask.sum()  > 0 else np.nan

        return {
            "label":               label,
            "omega":               float(omega),
            "n_valid_pairs":       n_v,
            "pearson_r":           float(r_pearson),
            "pearson_p":           float(p_pearson),
            "spearman_r":          float(r_spearman),
            "spearman_p":          float(p_spearman),
            "rmse_rad":            rmse,
            "rmse_hours":          float(rmse / omega),
            "mean_obs_lag_rad":    float(obs_v.mean()),
            "mean_pred_lag_rad":   float(pred_v.mean()),
            "mean_lag_short_lived_h": float(mean_lag_short / omega),
            "mean_lag_long_lived_h":  float(mean_lag_long  / omega),
            "dose_response_direction_correct": bool(mean_lag_long > mean_lag_short),
            "observed_lags_rad":   obs_v.tolist(),
            "predicted_lags_rad":  pred_v.tolist(),
            "k_deg_tgt":           k_v.tolist(),
            "half_lives_h":        half_life_h.tolist(),
        }

    def print_cascade_report(self, report: dict) -> None:
        label = report.get("label", "G→P")
        print(f"\n── Phase Cascade Test [{label}] (Falsifiable Prediction) ──")
        if "note" in report:
            print(f"  {report['note']}")
            return
        if "omega" in report:
            period = 2 * np.pi / report["omega"]
            print(f"  Driving clock:      ω={report['omega']:.4f} rad/h  (period {period:.1f} h)")
        print(f"  Valid pairs:        {report['n_valid_pairs']}")
        print(f"  Pearson r:          {report['pearson_r']:+.3f}  (p={report['pearson_p']:.3e})")
        print(f"  Spearman ρ:         {report['spearman_r']:+.3f}  (p={report['spearman_p']:.3e})")
        print(f"  RMSE:               {report['rmse_rad']:.3f} rad  "
              f"({report['rmse_hours']:.1f} h)")
        print(f"  Mean observed lag:  {report['mean_obs_lag_rad']:.3f} rad")
        print(f"  Mean predicted lag: {report['mean_pred_lag_rad']:.3f} rad")
        print(f"  Dose-response:      short-lived = {report['mean_lag_short_lived_h']:.2f} h  "
              f"long-lived = {report['mean_lag_long_lived_h']:.2f} h  "
              f"[correct direction: {report['dose_response_direction_correct']}]")
        print("─────────────────────────────────────────────────────────\n")
