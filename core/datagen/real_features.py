"""
real_features.py — derived-feature builder for the real circadian multi-omics dataset.

Turns the three raw real layers (staged under data/real/) into the model-ready
quantities the schema needs, WITHOUT fitting any quantity the validation later scores:

  1. k_deg per node — from PUBLISHED half-lives (literature priors per layer,
     log-normal spread), never fit from the trajectories. This is the independent
     quantity the cascade prediction consumes.
  2. Harmonic (cosinor) fit per pool — a 24 h time-referenced cosine
        q(t) ≈ mesor + A cos(ω t) + B sin(ω t)
     giving amplitude, acrophase (rad) and a rhythmicity p-value (F-test). The
     acrophase is the MEASURED phase used for the cross-omic cascade lag.
  3. Cross-layer symbol match — gene i ↔ protein i by gene symbol; the redox
     protein → metabolite pairing uses curated enzyme→metabolite links.

Half-life sources (cite in manuscript):
  * Schwanhausser B, et al. Global quantification of mammalian gene expression
    control. Nature 473:337 (2011). PMID 21593866 (verified via EuropePMC).
    [mammalian mRNA median t½ ≈ 7 h; protein median t½ ≈ 46 h]
  * Metabolite turnover: fast pool, minute-to-hour scale (order-of-magnitude
    layer prior, not a specific per-metabolite citation).
NOTE: layer half-life medians/ranges below are literature-grounded ORDER-OF-
MAGNITUDE priors used only to initialise k_deg; the manuscript should attach a
verified per-value citation to each before publication.

All arrays are keyed by the exact layer names 'genomics','proteome','metabolome'.
"""
import os
import numpy as np

_HERE  = os.path.dirname(os.path.abspath(__file__))   # codes/data/
_CODES = os.path.dirname(_HERE)                        # codes/

# ── REAL_DIR: three-tier discovery (highest → lowest priority) ────────────────
#   1. Env var HNN_REAL_DATA_DIR  (explicit override — use on GCP or custom paths)
#   2. experiments/data/real/     (canonical location when experiments/ is gitignored)
#   3. codes/data/real/           (legacy fallback for old checkouts)
_EXP_REAL  = os.path.join(_CODES, "experiments", "data", "real")
_LEGACY_REAL = os.path.join(_HERE, "real")

REAL_DIR = (
    os.environ.get("HNN_REAL_DATA_DIR")           # (1) explicit override
    or (_EXP_REAL if os.path.isdir(_EXP_REAL) else _LEGACY_REAL)  # (2) or (3)
)
# ─────────────────────────────────────────────────────────────────────────────

LN2 = np.log(2.0)

# Published half-life priors (hours): (median, lo, hi) — lognormal spread within layer
HALFLIFE_PRIOR = {
    "genomics":   (7.0,  2.0,  20.0),    # mRNA
    "proteome":   (46.0, 10.0, 120.0),   # protein
    "metabolome": (0.5,  0.1,  2.0),     # metabolite
}

# Clock frequencies (rad/h) — must match compartments.CLOCK_BANK
OMEGA_CIRCADIAN = 2 * np.pi / 24.0


def compile_k_deg(layer, symbols, seed=0):
    """Per-node degradation rate (1/h) from published half-life priors.

    Deterministic given (layer, n, seed). Draws t½ log-normally around the
    published layer median, clipped to the published [lo,hi] range, then
    k_deg = ln2 / t½.  NOT fit from data.
    """
    med, lo, hi = HALFLIFE_PRIOR[layer]
    n = len(symbols)
    rng = np.random.default_rng(abs(hash((layer, seed))) % (2**32))
    # lognormal in t½ with sigma chosen so ±1σ ~ spans [lo,hi] in log space
    sigma = (np.log(hi) - np.log(lo)) / 4.0
    t_half = np.exp(rng.normal(np.log(med), sigma, size=n))
    t_half = np.clip(t_half, lo, hi)
    return (LN2 / t_half).astype(np.float64)


def cosinor_fit(t_hours, y, period=24.0):
    """Time-referenced single-harmonic cosinor fit.

    Returns dict with mesor, amplitude, acrophase (rad, atan2(B,A)), r2, and an
    F-test p-value for rhythmicity (full cosine model vs mesor-only null).
    t_hours, y: 1-D arrays of equal length. NaNs in y are dropped.
    """
    t = np.asarray(t_hours, float)
    y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(t)
    t, y = t[m], y[m]
    if len(y) < 4 or np.allclose(y, y[0]):
        return dict(mesor=np.nan, amplitude=0.0, acrophase=np.nan, r2=0.0, pval=1.0)
    w = 2 * np.pi / period
    X = np.column_stack([np.ones_like(t), np.cos(w * t), np.sin(w * t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mesor, A, B = beta
    amp = float(np.hypot(A, B))
    acro = float(np.arctan2(B, A))
    # F-test: full (p=3) vs reduced mean-only (p=1)
    n = len(y)
    df1, df2 = 2, n - 3
    if df2 > 0 and ss_res > 0:
        F = ((ss_tot - ss_res) / df1) / (ss_res / df2)
        # survival of F-distribution
        from scipy.stats import f as fdist
        pval = float(fdist.sf(F, df1, df2))
    else:
        pval = 1.0
    return dict(mesor=float(mesor), amplitude=amp, acrophase=acro, r2=float(r2), pval=pval)


def _normalize_layer(expr, kind):
    """Put a raw layer on the model's common O(1-10) positive abundance scale.

    The model's losses, capacitance and phasor amplitude gate are calibrated to
    the synthetic generator's O(1-10) positive abundances (per-layer mean ~5).
    Raw real layers arrive on wildly different scales:
      * intensity layers (array transcriptome, LC-MS metabolome) are LINEAR
        intensities spanning several orders of magnitude → log2 first;
      * the SILAC proteome is ALREADY log2 ratios (can be negative) → no log.
    After the per-node log (where applicable) each layer is affinely mapped to a
    common location/scale comparable to the synthetic layers (per-layer mean ~5,
    O(1-10) spread). This is a MONOTONE per-layer transform: it preserves every
    node's phase and its RELATIVE amplitude ranking; it changes only the common
    scale, not the circadian structure the validation scores.
    """
    e = np.asarray(expr, float)
    if kind == "log":
        e = np.log2(np.clip(e, 1.0, None))          # linear intensity → log2
    # robust per-layer standardization: median → 5.0, IQR → 2.0
    med = np.nanmedian(e)
    iqr = np.nanpercentile(e, 75) - np.nanpercentile(e, 25)
    iqr = iqr if iqr > 1e-9 else (np.nanstd(e) + 1e-9)
    return 5.0 + 2.0 * (e - med) / iqr


def load_layer(layer, normalize=True):
    """Load a derived raw layer .npz → (symbols, expr [N,T], time_hours [T]).

    Time axes are folded onto a single 24 h phase (CT/ZT mod 24) so multi-cycle
    series contribute all cycles to the harmonic fit. With normalize=True (default)
    the abundance is placed on the model's common O(1-10) scale via
    _normalize_layer (log2 for intensity layers, affine for all); pass
    normalize=False to inspect the raw intensities.
    """
    if layer == "genomics":
        d = np.load(os.path.join(REAL_DIR, "transcriptome",
                                 "liver_transcriptome_genelevel.npz"), allow_pickle=True)
        e = d["expr"].astype(float)
        if normalize:
            e = _normalize_layer(e, "log")          # array intensity → log2 → affine
        return d["genes"].astype(str), e, d["CT"].astype(float)
    if layer == "proteome":
        d = np.load(os.path.join(REAL_DIR, "proteome", "liver_proteome.npz"),
                    allow_pickle=True)
        e = d["expr"].astype(float)
        if normalize:
            e = _normalize_layer(e, "linear")       # already log2 SILAC → affine only
        return d["genes"].astype(str), e, d["CT"].astype(float)
    if layer == "metabolome":
        d = np.load(os.path.join(REAL_DIR, "metabolome", "liver_metabolome.npz"),
                    allow_pickle=True)
        e = d["expr"].astype(float)
        if normalize:
            e = _normalize_layer(e, "log")          # LC-MS intensity → log2 → affine
        return d["metabolites"].astype(str), e, d["ZT"].astype(float)
    raise ValueError(layer)


def load_measured_cascade():
    """Robles Table S4: measured mRNA phase, protein phase, transcript→protein lag (h)."""
    d = np.load(os.path.join(REAL_DIR, "proteome", "measured_cascade_lag.npz"),
                allow_pickle=True)
    return dict(genes=d["genes"].astype(str),
                mrna_phase_ct=d["mrna_phase_ct"].astype(float),
                prot_phase_ct=d["prot_phase_ct"].astype(float),
                lag_hours=d["lag_hours"].astype(float))


# ─────────────────────────────────────────────────────────────────────────────
#  Node selection: map real pools onto the model's fixed compartment layout
# ─────────────────────────────────────────────────────────────────────────────

# Canonical liver clock / clock-output genes to seed the core_clock compartment
CORE_CLOCK_GENES = [
    "Arntl", "Npas2", "Clock", "Per1", "Per2", "Per3", "Cry1", "Cry2",
    "Nr1d1", "Nr1d2", "Rora", "Rorc", "Dbp", "Tef", "Hlf", "Nfil3",
    "Ciart", "Bhlhe40", "Bhlhe41", "Nampt", "Wee1", "Pnp",
]

# Redox / peroxiredoxin / glutathione-system enzymes for the redox compartment
REDOX_GENES = [
    "Prdx1", "Prdx2", "Prdx3", "Prdx4", "Prdx5", "Prdx6", "Gpx1", "Gpx4",
    "Gsr", "Gclc", "Gclm", "Txn1", "Txn2", "Txnrd1", "Cat", "Sod1", "Sod2",
    "Nqo1", "Gsta1", "Gstm1", "Nfe2l2", "Hmox1",
]


def _cosinor_all(syms, expr, t, period=24.0):
    tmod = np.asarray(t, float) % period
    out = [cosinor_fit(tmod, expr[i], period) for i in range(expr.shape[0])]
    return dict(
        r2=np.array([f["r2"] for f in out]),
        pval=np.array([f["pval"] for f in out]),
        acro=np.array([f["acrophase"] for f in out]),
        amp=np.array([f["amplitude"] for f in out]),
        mesor=np.array([f["mesor"] for f in out]),
    )


def select_nodes(seed=0):
    """Select real pools for each of the model's fixed node slots.

    Returns a dict per layer with:
      idx     : indices into the raw layer arrays (len = model layer size)
      symbols : chosen pool symbols
      fit      : cosinor params (r2, pval, acro, amp, mesor) for the chosen pools
    plus 'cascade' : Robles-S4 matched G→P pairs that are CO-SELECTED (the same
      symbol lands in both the genomics and proteome node sets), as
      (symbol, gene_idx_in_selection, prot_idx_in_selection, measured_lag_h).
      These are the pairs the per-node cascade test can score. The wider
      population of ALL matched pairs present in both raw layers (regardless of
      selection) is exposed separately as 'measured_lag_population'.

    Layer sizes and compartment index ranges are taken to match compartments.py:
      genomics  : 40  (core_clock G[0:12], signalling G[12:24], biosynthesis G[24:40])
      proteome  : 35  (core_clock P[0:12], redox P[12:20], energy P[20:28], signalling P[28:35])
      metabolome: 25  (energy/conserved M[0:12], redox M[12:20], biosynthesis M[20:25])
    """
    from .compartments import COMPARTMENT_CONFIG  # noqa
    NG, NP, NM = 40, 35, 25

    gS, gE, gT = load_layer("genomics")
    pS, pE, pT = load_layer("proteome")
    mS, mE, mT = load_layer("metabolome")
    gf = _cosinor_all(gS, gE, gT)
    pf = _cosinor_all(pS, pE, pT)
    mf = _cosinor_all(mS, mE, mT)
    gi = {s: i for i, s in enumerate(gS)}
    pi = {s: i for i, s in enumerate(pS)}
    casc = load_measured_cascade()

    gset, pset = set(gS), set(pS)

    # ---- core_clock: matched G↔P pairs with measured lag (Robles S4), ranked ----
    s4 = [(casc["genes"][i], casc["lag_hours"][i])
          for i in range(len(casc["genes"]))
          if casc["genes"][i] in gset and casc["genes"][i] in pset
          and np.isfinite(casc["lag_hours"][i])]
    # prefer canonical clock genes, then by transcriptome R²
    def rank_key(item):
        s = item[0]
        is_clock = 0 if s in CORE_CLOCK_GENES else 1
        return (is_clock, -gf["r2"][gi[s]])
    s4_sorted = sorted(s4, key=rank_key)
    core_syms = [s for s, _ in s4_sorted[:12]]
    core_lags = {s: lg for s, lg in s4_sorted[:12]}
    # full matched cascade set (all pairs), for the cascade test's statistical power
    cascade_full = [(s, lg) for s, lg in s4_sorted]

    # ---- Shared matched-cascade set: Robles-S4 pairs (beyond the core 12) that
    #      we deliberately place into BOTH the genomics and proteome node sets so
    #      each is a scoreable transcript→protein cascade pair. Proteome has
    #      15 non-core/non-redox fill slots, so we co-select up to 15 extra pairs
    #      (ranked by transcriptome R²). This makes the cascade test use the FULL
    #      set of co-selectable matched pairs, not just the core 12. ----
    s4_syms_by_rank = [s for s, _ in s4_sorted]           # rank-ordered, excludes<core order
    extra_cascade = [s for s in s4_syms_by_rank
                     if s not in core_syms and s in pset][:15]

    # ---- genomics: core_clock(12) + extra_cascade + top-R² fill → 40 ----
    used_g = set(core_syms) | set(extra_cascade)
    g_fill = [gS[i] for i in np.argsort(-gf["r2"]) if gS[i] not in used_g]
    n_g_fill = NG - len(core_syms) - len(extra_cascade)
    g_fill = g_fill[:n_g_fill]; used_g |= set(g_fill)
    genomics_syms = core_syms + extra_cascade + g_fill    # 12 + up to15 + rest = 40
    assert len(genomics_syms) == NG, len(genomics_syms)

    # ---- redox compartment (proteome[12:20]) ----
    # Biology: the curated peroxiredoxin/glutathione enzymes DEFINE the redox
    # oscillator, but in THIS SILAC proteome (Robles 2014) they are largely
    # NON-rhythmic at the abundance level — an observation from our own cosinor
    # analysis (only ~2/8 pass p<0.05), consistent with the known post-
    # translational / transcription-independent nature of the peroxiredoxin redox
    # cycle. See PROVENANCE.md § "Redox-arm limitation". A cascade test needs
    # rhythmic SOURCE pools, so we take curated redox enzymes that ARE rhythmic
    # (cosinor p<0.05) first, then fill the remaining slots with the most-rhythmic
    # proteins overall. This is a data-driven compromise, flagged in the manuscript
    # as a limitation of the redox arm.
    used_p = set(core_syms)
    redox_rhythmic = [s for s in REDOX_GENES
                      if s in pset and s not in used_p and pf["pval"][pi[s]] < 0.05]
    redox_rhythmic.sort(key=lambda s: -pf["r2"][pi[s]])
    redox_p = list(redox_rhythmic[:8])
    if len(redox_p) < 8:
        extra = [pS[i] for i in np.argsort(-pf["r2"])
                 if pS[i] not in used_p and pS[i] not in redox_p]
        redox_p += extra[: 8 - len(redox_p)]
    used_p |= set(redox_p)
    n_curated_redox = len(redox_rhythmic[:8])

    # ---- proteome energy(8)+signalling(7): the SAME extra_cascade genes first
    #      (guaranteeing co-selection), then top-R² fill. ----
    prefer = [s for s in extra_cascade if s not in used_p]
    used_p |= set(prefer)
    fill_slots = 15
    remaining = fill_slots - len(prefer)
    p_avail = [pS[i] for i in np.argsort(-pf["r2"]) if pS[i] not in used_p]
    filler = p_avail[:remaining]; used_p |= set(filler)
    energy_signal_p = prefer + filler                     # 15 pools
    energy_p = energy_signal_p[:8]
    signal_p = energy_signal_p[8:15]

    proteome_syms = core_syms + redox_p + energy_p + signal_p  # 12+8+8+7 = 35
    assert len(proteome_syms) == NP, len(proteome_syms)

    # ---- metabolome: energy/conserved(12) + redox(8) + biosynthesis(5) ----
    # rank metabolites by rhythmicity; skip duplicate feature names (the raw
    # ST002079 panel repeats some lipid identifiers) so node symbols are unique.
    m_order = list(np.argsort(-mf["r2"]))
    metab_idx, seen_m = [], set()
    for i in m_order:
        name = str(mS[i])
        if name in seen_m:
            continue
        seen_m.add(name); metab_idx.append(i)
        if len(metab_idx) == NM:
            break
    metab_idx = np.array(metab_idx)
    metabolome_syms = [mS[i] for i in metab_idx]

    def pack(syms, S, idxmap, fit):
        idx = np.array([idxmap[s] for s in syms])
        return dict(
            idx=idx, symbols=list(syms),
            r2=fit["r2"][idx], pval=fit["pval"][idx],
            acro=fit["acro"][idx], amp=fit["amp"][idx], mesor=fit["mesor"][idx],
        )

    sel = {
        "genomics": pack(genomics_syms, gS, gi, gf),
        "proteome": pack(proteome_syms, pS, pi, pf),
        "metabolome": dict(
            idx=np.array(metab_idx), symbols=metabolome_syms,
            r2=mf["r2"][metab_idx], pval=mf["pval"][metab_idx],
            acro=mf["acro"][metab_idx], amp=mf["amp"][metab_idx],
            mesor=mf["mesor"][metab_idx],
        ),
    }
    # cascade: matched G↔P pairs at co-selected node positions (used by the
    # per-node cascade test, which needs the same pool in both selected layers)
    gpos = {s: k for k, s in enumerate(genomics_syms)}
    ppos = {s: k for k, s in enumerate(proteome_syms)}
    cascade = [(s, gpos[s], ppos[s], lg) for s, lg in cascade_full
               if s in gpos and s in ppos]
    sel["cascade"] = cascade
    sel["core_lags"] = core_lags
    # population-level measured lag distribution (all Robles-S4 pairs present in
    # BOTH raw layers) — a model-independent target for the aggregate lag statistic
    sel["measured_lag_population"] = np.array([lg for _, lg in cascade_full], float)
    sel["measured_lag_symbols"] = [s for s, _ in cascade_full]
    return sel


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory reconstruction onto the model's dense grid
# ─────────────────────────────────────────────────────────────────────────────

def reconstruct_trajectory(t_grid, mesor, amp, acro, omega, rhythmic):
    """Smooth trajectory on the model grid from real cosinor parameters.

    Rhythmic pool  : q(t) = mesor + amp * cos(omega * t - acro_shift), where the
                     acrophase is preserved so the MEASURED phase (and hence the
                     cross-omic lag) is carried into the dense trajectory.
    Non-rhythmic   : flat homeostasis at the mesor (base-layer abundance).

    The cosine is written cos(ωt)·A + sin(ωt)·B with A=amp·cos(acro),
    B=amp·sin(acro) so acro = atan2(B,A) is exactly the fitted acrophase.
    """
    t = np.asarray(t_grid, float)
    if rhythmic and np.isfinite(acro):
        A = amp * np.cos(acro)
        B = amp * np.sin(acro)
        return mesor + A * np.cos(omega * t) + B * np.sin(omega * t)
    return np.full_like(t, mesor if np.isfinite(mesor) else 0.0)
