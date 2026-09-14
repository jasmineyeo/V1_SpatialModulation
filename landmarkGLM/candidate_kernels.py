# -*- coding: utf-8 -*-
"""
landmarkGLM/candidate_kernels.py

Builds the gauss / comb / adapt / free candidate kernel families used by
compare_kernels.

DMM, Aug 2026
"""

import numpy as np

from basis_funcs import lag_expand, raised_cosine_value_basis


def fit_span(cfg):
    """ (lo, hi) in cm: the stretch of track the position kernels are fit on.
    """

    sp = getattr(cfg, "fit_span_cm", None)
    if sp is None:
        return float(cfg.zones[0][0]), float(cfg.zones[-1][1])
    return float(sp[0]), float(sp[1])


def fit_span_mask(beh, cfg):
    """ Frames inside fit_span. Combine with run_mask, never a replacement.
    """

    lo, hi = fit_span(cfg)
    pos = np.asarray(beh["pos"], float)
    return (pos >= lo) & (pos <= hi)


def va_slope_grid(cfg):
    """ The slopes the adapting-visual envelope is allowed to take.
    """

    n = getattr(cfg, "va_slope_steps", None)
    if n is None:
        n = len(cfg.delta_grid)
    lo, hi = getattr(cfg, "va_slope_range", (0.1, 2.5))
    return np.linspace(float(lo), float(hi), int(n))


def lap_decay_ramp(beh, cfg, rate=1.0):
    """ The adaptation envelope: 1 at start, falling linearly toward 0.

    Returns (ramp, T), T = reference length in seconds. `rate` is the SLOPE
    multiplier: envelope = 1 - rate * t / T, so rate=1 hits zero exactly at T,
    rate=0.1 barely decays by then, rate=2.5 is zero by 40% of it. v05 pinned
    rate=1.
    """

    t_lap = np.asarray(beh["t_since_lap"], dtype=float)
    n = len(t_lap)
    lap_id = np.asarray(beh["lap_id"], dtype=int)
    pos = np.asarray(beh["pos"], dtype=float)
    origin = str(getattr(cfg, "adapt_ramp_origin", "span"))
    lo, hi = fit_span(cfg)
    laps = np.unique(lap_id[lap_id >= 0])

    if origin == "span":
        t0 = np.full(n, np.nan)
        for L in laps:
            m = lap_id == L
            ins = m & (pos >= lo) & np.isfinite(pos)
            if ins.any():
                t0[m] = float(np.nanmin(t_lap[ins]))
        t = np.maximum(np.nan_to_num(t_lap - t0, nan=0.0), 0.0)

        def _window(L):
            """ Latest ramp time reached inside fit_span on lap L.
            """

            m = (lap_id == L) & (pos >= lo) & (pos <= hi) & np.isfinite(pos)
            return float(np.nanmax(t[m])) if m.any() else np.nan
    else:
        t = t_lap

        def _window(L):
            """ Latest ramp time reached anywhere on lap L. """

            m = lap_id == L
            return float(np.nanmax(t[m])) if np.isfinite(t[m]).any() else np.nan

    if bool(getattr(cfg, "adapt_per_lap_ramp", False)):
        dur = np.full(n, np.nan)
        for L in laps:
            w = _window(L)
            dur[lap_id == L] = max(w if np.isfinite(w) else 1e-6, 1e-6)
        with np.errstate(invalid="ignore", divide="ignore"):
            ramp = 1.0 - float(rate) * t / dur
        T = float(np.nanmedian(dur))
    else:
        T = getattr(cfg, "adapt_ramp_s", None)
        if T is None:
            durs = [w for w in (_window(L) for L in laps) if np.isfinite(w)]
            T = float(np.median(durs)) if durs else 20.0
        T = float(T)
        ramp = 1.0 - float(rate) * t / max(T, 1e-6)

    ramp = np.clip(np.nan_to_num(ramp, nan=0.0), 0.0, 1.0)
    return ramp, T


def build_candidates(beh, cfg, lagB):
    """ Pre-build every candidate kernel ONCE, already lag-expanded.

    Three hypotheses plus a reference:
      gauss : ONE bump at position mu, width sigma -- the 'place' hypothesis.
      comb  : FOUR bumps at the landmark centers, slid rigidly by delta, all
              width sigma -- the 'landmark' hypothesis.
      adapt : the same four bumps, each scaled by a ramp falling linearly
              from 1 at lap onset to 0 over adapt_ramp_s seconds -- the
              'habituating landmark' hypothesis.
      free  : an unconstrained position basis. Not a hypothesis -- a CEILING
              showing how much position-locked signal the restricted models
              leave on the table. If all sit far below it, one-vs-four-peaks-
              vs-adaptation is the wrong frame for those cells.

    """

    pos = beh["pos"]
    n = len(pos)
    ok = (pos >= 0) & (pos <= cfg.corridor_cm)
    iti = beh.get("iti")
    rm = beh["run_mask"]
    centers = np.array([(a + b) / 2.0 for a, b in cfg.zones])

    def _prep_col(col):
        """ Zero non-running and ITI frames, then lag-expand. """

        col = np.asarray(col, float).copy()
        col[~ok] = 0.0
        if iti is not None and iti.any():
            col[iti] = 0.0
        return lag_expand(col[:, None], lagB).astype(np.float32)

    _lo, _hi = fit_span(cfg)
    _mus = cfg.mu_grid
    if bool(getattr(cfg, "restrict_mu_to_span", True)):
        _mus = np.asarray([m for m in cfg.mu_grid if _lo <= m <= _hi])
    gX, gpar = [], []
    for mu in _mus:
        for sg in cfg.sigma_grid:
            gX.append(_prep_col(np.exp(-0.5 * ((pos - mu) / sg) ** 2)))
            gpar.append((float(mu), float(sg)))

    use_ad = bool(getattr(cfg, "adapt_in_candidates", False))

    va_rates = va_slope_grid(cfg) if use_ad else np.array([])
    va_ramps, T_ramp = [], np.nan
    if use_ad:
        for _r in va_rates:
            _rm_, T_ramp = lap_decay_ramp(beh, cfg, rate=float(_r))
            va_ramps.append(_rm_)
    ramp = va_ramps[len(va_ramps) // 2] if use_ad else None
    match_scale = bool(getattr(cfg, "adapt_match_scale", True))

    def _rms(v):
        """ Root-mean-square of v on running frames. """

        w = np.asarray(v, float)[rm]
        return float(np.sqrt(np.mean(w ** 2))) if w.size else 1.0

    cX, cpar = [], []
    aX, apar = [], []
    for dl in cfg.delta_grid:
        for sg in cfg.sigma_grid:
            c = np.zeros(n)
            for m0 in centers:
                c += np.exp(-0.5 * ((pos - (m0 + dl)) / sg) ** 2)
            cX.append(_prep_col(c))
            cpar.append((float(dl), float(sg)))
            if use_ad:
                for _r, _rmp in zip(va_rates, va_ramps):
                    a = c * _rmp
                    if match_scale:
                        r0, r1 = _rms(c), _rms(a)
                        if r1 > 1e-12:
                            a = a * (r0 / r1)
                    aX.append(_prep_col(a))
                    apar.append((float(dl), float(sg), float(_r)))

    fb = raised_cosine_value_basis(pos, cfg.n_free_basis, lo=0.0,
                                   hi=cfg.corridor_cm, valid=ok)
    fb = np.asarray(fb, float).copy()
    fb[~ok] = 0.0
    if iti is not None and iti.any():
        fb[iti] = 0.0
    fX = lag_expand(fb, lagB).astype(np.float32)

    _insp = fit_span_mask(beh, cfg) & rm

    if use_ad:
        _org = str(getattr(cfg, "adapt_ramp_origin", "span"))

        _z = float(np.mean(ramp[rm & fit_span_mask(beh, cfg)] <= 0.0))
        if _z > 0.05:

            pass
        mode = ("each lap's OWN duration (median {:.1f} s) -- DISTANCE-LIKE, "
                "see Config".format(T_ramp)
                if getattr(cfg, "adapt_per_lap_ramp", False)
                else "a fixed {:.1f} s on every lap{}".format(
                    T_ramp, "" if getattr(cfg, "adapt_ramp_s", None) is not None
                    else " (= the median lap duration, measured)"))

        env, tmed = [], []
        for z0, z1 in cfg.zones:
            s = rm & (pos >= z0) & (pos <= z1)
            env.append(float(np.nanmean(ramp[s])) if s.any() else np.nan)
            tmed.append(float(np.nanmedian(beh["t_since_lap"][s]))
                        if s.any() else np.nan)

        _dep = (np.nanmax(env) - np.nanmin(env)) / max(np.nanmax(env), 1e-9)

        if _dep < 0.15:
            pass

        _nsg, _nrt = len(cfg.sigma_grid), len(va_rates)
        _dmid = len(cfg.delta_grid) // 2
        for _ri in (0, _nrt // 2, _nrt - 1):
            _e = []
            for _z0, _z1 in cfg.zones:
                _sel = rm & (pos >= _z0) & (pos <= _z1)
                _e.append(float(np.nanmean(va_ramps[_ri][_sel]))
                          if _sel.any() else np.nan)
            # aX is (delta, sigma, rate) in that nesting order.
            _ia = (_dmid * _nsg + _nsg // 2) * _nrt + _ri
            _ic = _dmid * _nsg + _nsg // 2
            _cc = float(np.corrcoef(cX[_ic][rm].ravel().astype(float),
                                    aX[_ia][rm].ravel().astype(float))[0, 1])

        if match_scale:

            pass

    out = dict(gauss=dict(X=gX, par=gpar),
               comb=dict(X=cX, par=cpar),
               free=dict(X=fX))
    if use_ad:
        out["adapt"] = dict(X=aX, par=apar, ramp=ramp, ramp_s=T_ramp,
                            rates=va_rates, ramps=va_ramps)
    return out
