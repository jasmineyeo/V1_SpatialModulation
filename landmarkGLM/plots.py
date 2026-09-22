# -*- coding: utf-8 -*-
"""
landmarkGLM/plots.py

DMM, Aug 2026
"""

import os
import warnings

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from glm import PoissonFold, _pick_lambda_PB


SHAPE_NAMES_DEFAULT = ["four equal peaks"] + [
    "L{} preference".format(j + 1) for j in range(4)]


def _fmt(v, spec="{:+.2f}", na="n/a"):
    v = float(v)
    return spec.format(v) if np.isfinite(v) else na


def _lap_stats(V, gix, nlap, nb, min_laps=3, return_laps=False):

    nc = V.shape[1]
    o  = np.argsort(gix, kind="stable")
    g  = gix[o]

    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    tot    = np.add.reduceat(V[o], starts, axis=0)
    cnt    = np.diff(np.r_[starts, g.size]).astype(float)

    L = np.full((nlap * nb, nc), np.nan)
    L[g[starts]] = tot / cnt[:, None]
    L = L.reshape(nlap, nb, nc)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n  = np.isfinite(L).sum(axis=0).astype(float)
        mu = np.nanmean(L, axis=0)
        se = np.nanstd(L, axis=0) / np.sqrt(np.maximum(n, 1.0))

    bad = n < min_laps
    mu[bad] = np.nan
    se[bad] = np.nan

    return (mu, se, L) if return_laps else (mu, se)


def _compute_display_curves(beh, Y, XPB, cfg, scalePB=None, bin_cm=3.0):

    rm     = beh["run_mask"]
    pos    = beh["pos"]
    lap    = beh["lap_id"]
    lap_r  = lap[rm]
    nc     = Y.shape[1]
    onePB  = (np.ones(XPB.shape[1]) if scalePB is None
              else np.asarray(scalePB, float))

    if getattr(cfg, "use_pure_behavior", True):
        lam  = _pick_lambda_PB(XPB[rm], Y[rm],
                               np.ones(rm.sum(), dtype=bool), lap_r, cfg,
                               scale=onePB)
        MU   = next(PoissonFold(XPB[rm], Y[rm]).predict(XPB, onePB, [lam]))
    else:
        MU   = np.broadcast_to(Y[rm].mean(axis=0), Y.shape)
    R    = Y - MU

    edges = np.arange(0.0, cfg.corridor_cm + bin_cm, bin_cm)
    xcm   = (edges[:-1] + edges[1:]) / 2.0
    nb    = len(xcm)

    use  = rm & np.isfinite(pos) & (lap >= 0)
    ulap = np.unique(lap[use])
    gix  = (np.searchsorted(ulap, lap[use]) * nb
            + np.clip(np.digitize(pos[use], edges) - 1, 0, nb - 1))

    CU,  CU_se  = _lap_stats(R[use],  gix, len(ulap), nb)
    RAW, RAW_se = _lap_stats(Y[use], gix, len(ulap), nb)

    ZN     = ["L{}".format(j + 1) for j in range(4)]
    pk     = np.zeros(nc, int)
    pk_bin = np.full(nc, -1, int)
    for c in range(nc):
        trace = CU[:, c]
        if not np.isfinite(trace).any():
            continue
        best_amp = -np.inf
        for z, (z0, z1) in enumerate(cfg.zones):
            in_z = np.where((xcm >= z0) & (xcm < z1))[0]
            if not in_z.size:
                continue
            t_z = trace[in_z]
            if not np.isfinite(t_z).any():
                continue
            i_max = int(np.nanargmax(np.abs(t_z)))
            amp   = float(np.abs(t_z[i_max]))
            if amp > best_amp:
                best_amp  = amp
                pk[c]     = z
                pk_bin[c] = int(in_z[i_max])

    gpk_bin = np.full(nc, -1, int)
    for c in range(nc):
        t = RAW[:, c]
        if np.isfinite(t).any():
            gpk_bin[c] = int(np.nanargmax(t))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _inz = np.zeros(nb, bool)
        for _z0, _z1 in cfg.zones:
            _inz |= (xcm >= _z0) & (xcm < _z1)

        def _frac(sel):
            vr = np.nanvar(RAW[sel], axis=0)
            vc = np.nanvar(CU[sel], axis=0)
            return np.where(vr > 1e-12, 1.0 - vc / np.where(vr > 1e-12, vr, 1.0),
                            np.nan)

        pb = dict(removed=_frac(np.ones(nb, bool)),
                  removed_zone=_frac(_inz))

    _spd = np.asarray(beh["speed"], float)[use][:, None]
    _sm, _sse, _sL = _lap_stats(_spd, gix, len(ulap), nb, return_laps=True)
    speed = dict(mean=_sm[:, 0], se=_sse[:, 0], xcm=xcm,
                 laps=_sL[:, :, 0], lap_ids=ulap)

    return (xcm, CU, CU_se, RAW, RAW_se, pk, pk_bin, gpk_bin, ZN, pb, speed)

def plot_fig3c(km, tx, cfg, cand, outdir, fig3_helpers,
               gain_min=0.15, per_page=8, shape_names=None):

    required = (
        "_curve_rows", "Rdisp", "Cres", "xcm", "mu_mode", "gpar", "cpar",
        "_trace_title", "_BAR", "_HAS_AD", "_clip_span", "FIT_LO", "FIT_HI",
        "_predicted_curve", "_predicted_curve_pair", "_adapt_idx",
        "_in_span_bin", "_AD_COLOR", "_metric_bars", "_metric_lims")
    missing = [k for k in required if k not in fig3_helpers]

    h = fig3_helpers
    xcm       = h["xcm"];      mu_mode  = h["mu_mode"]
    gpar      = h["gpar"];     cpar     = h["cpar"]
    Cres      = h["Cres"]
    _trace_title     = h["_trace_title"]
    _BAR             = h["_BAR"]
    _HAS_AD          = h["_HAS_AD"]
    _clip_span       = h["_clip_span"]
    FIT_LO           = h["FIT_LO"];    FIT_HI   = h["FIT_HI"]
    _predicted_curve = h["_predicted_curve"]
    _predicted_curve_pair = h["_predicted_curve_pair"]
    _adapt_idx       = h["_adapt_idx"]
    _AD_COLOR        = h["_AD_COLOR"]
    _metric_bars     = h["_metric_bars"]
    _metric_lims     = h["_metric_lims"]

    if shape_names is None:
        shape_names = SHAPE_NAMES_DEFAULT

    _has_tx = isinstance(tx, dict)
    if _has_tx:
        _sel = (np.asarray(tx["driven"], bool)
                & np.isin(np.asarray(tx["type"], dtype=object),
                          ["visual", "spatial", "mixed"]))
    else:
        _sel = np.asarray(km["well_fit"], bool)
    cells   = np.sort(np.where(_sel)[0])
    _MLIMS  = _metric_lims(cells)
    pdf_path = os.path.join(outdir, "v07_cells_by_number.pdf")
    n_pages  = int(np.ceil(len(cells) / per_page))

    with PdfPages(pdf_path) as pdf:
        for pg in range(n_pages):
            page = cells[pg * per_page:(pg + 1) * per_page]
            fig, axes = plt.subplots(
                len(page), 3, figsize=(12.5, 2.8 * len(page)),
                gridspec_kw={"width_ratios": [3.4, 0.5, 0.5]},
                squeeze=False)
            for r, c in enumerate(page):
                c = int(c)
                a = axes[r, 0]
                a.axvspan(0.0, FIT_LO, color="#c9c9c9", alpha=0.55, lw=0, zorder=0)
                a.axvspan(FIT_HI, cfg.corridor_cm, color="#c9c9c9",
                          alpha=0.55, lw=0, zorder=0)
                for z0, z1 in cfg.zones:
                    a.axvspan(z0, z1, color="#f0d8a8", alpha=0.55, lw=0)
                gi = int(np.argmin(
                    np.abs(gpar[:, 0] - mu_mode[c]) * 10
                    + np.abs(gpar[:, 1] - km["sigma"][c])))
                ci = int(np.argmin(
                    np.abs(cpar[:, 0] - km["delta"][c]) * 10
                    + np.abs(cpar[:, 1] - km["sigma"][c])))
                a.plot(xcm, Cres[:, c], color="#222222", lw=1.9,
                       label="pure-behavior residual")
                a.plot(xcm,
                       _clip_span(_predicted_curve(c, cand["gauss"]["X"],
                                                   gi, key="gauss")),
                       color="#2f6b52", lw=1.5, label="spatial (1 peak)")
                a.plot(xcm,
                       _clip_span(_predicted_curve(c, cand["comb"]["X"],
                                                   ci, key="comb")),
                       color="#3a6ea8", lw=1.5, label="visual (4 peaks)")
                if _HAS_AD:
                    a.plot(xcm,
                           _clip_span(_predicted_curve(c, cand["adapt"]["X"],
                                                       _adapt_idx(c),
                                                       key="adapt")),
                           color=_AD_COLOR, lw=1.5, ls=":", label="adapting visual")
                a.plot(xcm,
                       _clip_span(_predicted_curve_pair(
                           c, cand["gauss"]["X"], gi, cand["comb"]["X"], ci)),
                       color="#00b8c4", lw=1.3, ls="--", alpha=0.9, label="both")
                a.set_title(_trace_title(
                    c, "#{} of {}".format(pg * per_page + r + 1, len(cells)),
                    show_adapting=False), fontsize=8.5, loc="left")
                a.set_ylabel("std of mean activity\n(rate units)")
                if r == 0:
                    a.legend(frameon=False, ncol=5, loc="upper right",
                             fontsize=8.5)
                if r == len(page) - 1:
                    a.set_xlabel("position (cm)")
                m = axes[r, 1]
                _metric_bars(m, c, _MLIMS)
                m.set_ylabel("index")
                m.set_ylim([-1, 1])
                b = axes[r, 2]
                b.bar(range(len(_BAR)),
                      [km[n][c] for n, _, _ in _BAR],
                      color=[col for _, _, col in _BAR])
                b.axhline(0, color="#222222", lw=0.8)
                b.set_xticks(range(len(_BAR)))
                b.set_xticklabels([t for _, t, _ in _BAR],
                                  fontsize=7, rotation=90)
                b.set_ylabel("dev. expl.", fontsize=8)
                b.tick_params(labelsize=7.5)
            fig.tight_layout(rect=[0, 0, 1, 0.995])
            pdf.savefig(fig)
            plt.close(fig)

    print("  -> Wrote {} ({} cells, {} pages).".format(
        pdf_path, len(cells), n_pages))

def _draw_trace(a, cfg, xcm, raw, raw_se, color, edge_cm, start_cm, reward_cm):

    for zl, zh in cfg.zones:
        a.axvspan(zl, zh, color="#f2c14e", alpha=.25, lw=0)
    a.axvline(edge_cm,  color="#1f4e79", lw=.8, ls=":")
    a.axvline(start_cm, color="#1f4e79", lw=.8, ls=":")
    a.axvline(reward_cm, color="#7d3c98", lw=.8, ls="--")

    a.fill_between(xcm, raw - raw_se, raw + raw_se, color=color, alpha=.30, lw=0)
    a.plot(xcm, raw, color=color, lw=1.3, label="raw")

    top = np.nanmax(raw + raw_se)
    a.set_ylim(0.0, float(top) * 1.10 if np.isfinite(top) and top > 0 else 1.0)
    a.set_xlim(0.0, max(cfg.corridor_cm, reward_cm) + 1.5)
    a.tick_params(labelsize=5, length=2)


EXCLUDED_ROWS = 6
EXCLUDED_COLS = 2

EXCLUSION_REASONS = ["unreliable + poorly fit", "unreliable", "poorly fit"]
EXCLUSION_COLORS = {
    "unreliable + poorly fit": "#7f1d1d",
    "unreliable": "#c0392b",
    "poorly fit": "#d68910",
}


def exclusion_reasons(rel, lad, cfg, wf):

    reliable = np.asarray(rel["ok"], bool)
    fit_ok = np.asarray(lad["landmark_gains"], float) > cfg.r2_threshold

    kept = reliable & fit_ok
    if not np.array_equal(kept, np.asarray(wf, bool)):
        raise RuntimeError("Exclusion rule no longer matches main.py well_fit "
                           "({} vs {} kept) -- update exclusion_reasons.".format(
                               int(kept.sum()), int(np.sum(wf))))

    reasons = np.full(len(kept), "", dtype=object)
    reasons[~reliable & ~fit_ok] = "unreliable + poorly fit"
    reasons[~reliable & fit_ok] = "unreliable"
    reasons[reliable & ~fit_ok] = "poorly fit"

    return reasons


def plot_excluded_pdf(lad, rel, cfg, cell_ids, outdir, wf, curves,
                      edge_cm=124.5, start_cm=9.0, reward_cm=134.4):

    xcm, CU, CU_se, RAW, RAW_se = curves[:5]
    reasons = exclusion_reasons(rel, lad, cfg, wf)
    excluded = np.flatnonzero(reasons != "")

    rank = {r: i for i, r in enumerate(EXCLUSION_REASONS)}
    r_rel = np.nan_to_num(np.asarray(rel["r"], float), nan=-np.inf)
    order = sorted(excluded, key=lambda c: (rank[reasons[c]], -r_rel[c]))

    counts = {r: int(np.sum(reasons == r)) for r in EXCLUSION_REASONS}
    summary = "  |  ".join("{} {}".format(r, counts[r]) for r in EXCLUSION_REASONS)
    per_page = EXCLUDED_ROWS * EXCLUDED_COLS
    n_pages = int(np.ceil(len(order) / per_page))
    pdf_path = os.path.join(outdir, "v09_excluded_cells.pdf")
    r2_fit = np.asarray(lad["landmark_gains"], float)

    with PdfPages(pdf_path) as pdf:
        for pg in tqdm(range(n_pages)):
            sel = order[pg * per_page:(pg + 1) * per_page]
            fig, axs = plt.subplots(EXCLUDED_ROWS, EXCLUDED_COLS,
                                    figsize=(8.5, 11.0), squeeze=False)
            fig.subplots_adjust(left=.07, right=.98, top=.93, bottom=.04,
                                hspace=.62, wspace=.18)

            for k in range(per_page):
                a = axs[k // EXCLUDED_COLS, k % EXCLUDED_COLS]
                if k >= len(sel):
                    a.set_visible(False)
                    continue
                i = sel[k]
                _draw_trace(a, cfg, xcm, RAW[:, i], RAW_se[:, i], "0.15",
                            edge_cm, start_cm, reward_cm)
                if k % EXCLUDED_COLS == 0:
                    a.set_ylabel("rate", fontsize=6)
                if k // EXCLUDED_COLS == EXCLUDED_ROWS - 1 or k + EXCLUDED_COLS >= len(sel):
                    a.set_xlabel("position (cm)", fontsize=6)
                a.set_title("#{}  {}   rel r {:.2f}   fit R$^2$ {:+.3f}".format(
                    int(cell_ids[i]), reasons[i], rel["r"][i], r2_fit[i]),
                    fontsize=6.5, color=EXCLUSION_COLORS[reasons[i]], loc="left")

            pdf.savefig(fig)
            plt.close(fig)

    print("  -> Wrote {} ({} excluded cells on {} pages: {})".format(
        pdf_path, len(order), n_pages, summary))

    return pdf_path


def plot_review_pdf(lad, shp, rel, cfg, cell_ids, outdir, wf, wf_kernel, curves,
                    shape_names=None, edge_cm=124.5, start_cm=9.0, reward_cm=134.4):

    if shape_names is None:
        shape_names = SHAPE_NAMES_DEFAULT

    EDGE_CM   = edge_cm
    START_CM  = start_cm
    REWARD_CM = reward_cm
    ROWS      = 6

    (_xcm, _CU, _CU_se, _RAW, _RAW_se,
     _pk, _pk_bin, _gpk_bin, _ZN, _PB, _SPD) = curves

    _gmax = np.nanmax(lad["gain"] * lad["sgn"][None, :], axis=0)

    _ctr    = np.array([(a + b) / 2.0 for a, b in cfg.zones])
    _pk_pos = np.where(_pk_bin >= 0, _xcm[np.clip(_pk_bin, 0, None)], np.nan)

    _gpk_pos = np.where(_gpk_bin >= 0, _xcm[np.clip(_gpk_bin, 0, None)], np.nan)
    _is_edge  = _gpk_pos > EDGE_CM
    _is_onset = _gpk_pos < START_CM
    _G  = lad["gain"] * lad["sgn"][None, :]

    _Gf  = lad["gain_per_fold"] * lad["sgn"][None, None, :]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _Gse = np.nanstd(_Gf, axis=0) / np.sqrt(
            np.maximum(np.sum(np.isfinite(_Gf), axis=0), 1))

    _dgain    = lad["d_gain"]
    _dgain_se = lad["d_gain_se"]
    _grun     = lad["gain_runner_up"]

    _z_shape = shp["shape_dr2"] / np.where(
        shp["shape_dr2_se"] < 1e-12, np.nan, shp["shape_dr2_se"])

    with np.errstate(invalid="ignore"):
        _mir_neg = np.isclose(shp["shape_dr2"], -shp["flat_dr2"],
                              rtol=0.0, atol=1e-15)
        _mir_eq = np.isclose(shp["shape_dr2"], shp["flat_dr2"],
                             rtol=0.0, atol=1e-15)
    _mirror = np.where(_mir_neg, "   = -vs flat",
                       np.where(_mir_eq, "   = vs flat", ""))
    _z_gain  = lad["gain_z"]

    _shape_j = np.where(wf, shp["shape_mode"] - 1, -99)
    _gain_j  = np.asarray(lad["pref"], int)
    _cmp     = wf & np.isfinite(_z_shape) & np.isfinite(_z_gain)

    _margin_min = float(getattr(cfg, "shape_margin_min", 0.20))
    _dpref_min  = float(getattr(cfg, "pref_log_ratio_min", 0.25))

    _Gs = np.sort(_G, axis=0)
    _d_pref = _Gs[3] - np.nanmean(_Gs[:3], axis=0)

    _has_pref = (wf
                 & (np.nan_to_num(lad["margin"], nan=0.0) >= _margin_min)
                 & (np.nan_to_num(_d_pref, nan=0.0) >= _dpref_min))

    _use_shape = (_shape_j >= 0) & (np.nan_to_num(_z_shape, nan=-np.inf)
                                    >= np.nan_to_num(_z_gain, nan=-np.inf))
    _ident_j = np.where(_use_shape, _shape_j, _gain_j)

    _trace_agrees = _ident_j == _pk
    if bool(getattr(cfg, "require_trace_agreement", True)):
        _has_pref = _has_pref & _trace_agrees

    _cat_j = np.where(_has_pref, _ident_j, -1)

    _agree   = _cmp & (_shape_j == _gain_j)
    _dis     = _cmp & (_shape_j != _gain_j)
    _det     = _dis & (_shape_j < 0)
    _ident   = _dis & (_shape_j >= 0)
    _win_shape = _dis & _has_pref & _use_shape
    _win_gain  = _dis & _has_pref & ~_use_shape
    _floored   = wf & ~_has_pref
    _arb = np.full(len(wf), "n/a", dtype=object)
    _arb[_agree & _has_pref] = "agree"
    _arb[_win_shape] = "shape"
    _arb[_win_gain]  = "gain"

    _f_margin = np.nan_to_num(lad["margin"], nan=0.0) < _margin_min
    _f_dpref  = np.nan_to_num(_d_pref, nan=0.0) < _dpref_min
    _f_trace  = (~_trace_agrees) & bool(
        getattr(cfg, "require_trace_agreement", True))
    for _i in np.flatnonzero(_floored):
        _why = ([] if not _f_margin[_i] else ["margin"]) \
             + ([] if not _f_dpref[_i]  else ["d_pref"]) \
             + ([] if not _f_trace[_i]  else ["trace veto"])
        _arb[_i] = " + ".join(_why) if _why else "floored"

    _tmpl_shape = np.array(["unlabeled"] * len(wf), dtype=object)
    _tmpl_shape[wf] = np.array(shape_names, dtype=object)[shp["shape_mode"][wf]]
    _tmpl = _tmpl_shape.copy()
    _tmpl[wf] = np.array(shape_names, dtype=object)[_cat_j[wf] + 1]

    # end zones
    _amp_lm = np.asarray(rel.get("amp_landmark", np.full(len(wf), -np.inf)),
                         float)
    _amp_rw = np.asarray(rel.get("amp_reward", np.full(len(wf), -np.inf)),
                         float)
    _amp_on = np.asarray(rel.get("amp_onset", np.full(len(wf), -np.inf)),
                         float)
    _is_reward = _is_edge & (_amp_rw > _amp_lm)
    _is_onset_only = _is_onset & ~_is_reward & (_amp_on > _amp_lm)

    _label = _tmpl.copy()
    _label[_is_onset_only] = "onset only"
    _label[_is_reward]     = "reward only"
    _tj  = _shape_j

    _pstr = shp["pref_ratio"]

    _shown = np.asarray(wf, bool)

    df9 = pd.DataFrame({
        "suite2p_cell_id":    cell_ids,
        "well_fit":           wf,
        "reliable":           rel["ok"],
        "reliability_r":      rel["r"],
        "reliability_r_full": rel["r_full"],
        "reliable_full":      rel["ok_full"],
        "rel_type":           rel["type"],
        "label":              _label,
        "template":           _tmpl,
        "template_shape_search": _tmpl_shape,
        "arbitrated_by":      _arb,
        "is_reward":          _is_reward,
        "is_onset":           _is_onset_only,
        "amp_landmark":       _amp_lm,
        "amp_reward":         _amp_rw,
        "amp_onset":          _amp_on,
        "shape_stability":    shp["shape_stability"],
        "r2_shape":           shp["r2"],
        "shape_delta":        shp["delta"],
        "shape_sigma":        shp["sigma"],
        "shape_pref_ratio":   shp["pref_ratio"],
        "max_gain":           _gmax,
        "well_fit_kernel":    wf_kernel,
        "argmax_gain":        np.where(wf, lad["pref"] + 1, -1),
        "trace_peak_zone":    np.array(_ZN, dtype=object)[_pk],
        "trace_peak_cm":      _pk_pos,
        "track_peak_cm":      _gpk_pos,
        "edge_ramp":          _is_edge,
        "onset_ramp":         _is_onset,
        "pref_strength":      _pstr,
        "d_pref":             _d_pref,
        "trace_agrees":       _trace_agrees,
        "drive":              lad["drive"],
        "sign":               lad["sgn"],
        "adapt":              lad["adapt"],
        "quadratic":          lad["quadratic"],
        "margin":             lad["margin"],
        "r2_comb":            lad["comb"],
        "r2_landmark_gains":  lad["landmark_gains"],
        "shape_dR2":          shp["shape_dr2"],
        "shape_dR2_se":       shp["shape_dr2_se"],
        "shape_flat_dR2":     shp["flat_dr2"],
        "shape_flat_dR2_se":  shp["flat_dr2_se"],
        "shape_flat_z":       shp["flat_z"],
        "shape_runner_up":    np.array(shape_names, dtype=object)[
                                  shp["shape_runner_up"]],
        "shape_z":            _z_shape,
        "d_gain":             _dgain,
        "d_gain_se":          _dgain_se,
        "gain_runner_up":     _grun + 1,
        "gain_z":             _z_gain,
        **{"gain_L{}".format(j + 1): _G[j] for j in range(4)},
        **{"prof_L{}".format(j + 1): lad["profile"][j] for j in range(4)},
    })[_shown].reset_index(drop=True)
    _csv = os.path.join(outdir, "v09_per_cell_full.csv")
    df9.to_csv(_csv, index=False)
    print("  -> Wrote {} ({} rows of {} cells; the {} that earn a PDF page)"
          .format(_csv, len(df9), len(_shown), int(_shown.sum())))

    _wn = int(wf.sum())

    _DISP  = {"four equal peaks": "all equal",
              "L1 preference": "L1-preferring",
              "L2 preference": "L2-preferring",
              "L3 preference": "L3-preferring",
              "L4 preference": "L4-preferring",
              "onset only": "onset only",
              "reward only": "reward only",
              "unlabeled": "unlabeled"}
    _SHORTC = dict(zip(shape_names,
                       plt.cm.tab10(np.linspace(0, 1, 10))[[7, 0, 2, 4, 1]]))
    _HDRC   = dict(zip(shape_names,
                       ["#555555", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]))
    _HDRC["onset only"]  = "#17becf"
    _HDRC["reward only"] = "#7d3c98"
    _HDRC["unlabeled"]   = "#999999"

    def _tpage(pdf, body, title):
        f = plt.figure(figsize=(8.5, 11.0))
        f.text(.06, .965, title, fontsize=11, weight="bold", va="top")
        f.text(.06, .935, body, fontsize=6.2, family="monospace", va="top")
        pdf.savefig(f)
        plt.close(f)

    _cat   = np.where(_is_reward, 6,
                      np.where(_is_onset_only, 5,
                               np.where(wf, _cat_j + 1, 99)))
    _order = np.lexsort((-np.abs(lad["drive"]), _pk, _cat))
    _order = _order[_shown[_order]]
    _PDF   = os.path.join(outdir, "v09_cell_review.pdf")

    with PdfPages(_PDF) as pdf:
        _np = int(np.ceil(_order.size / ROWS))
        for pg in tqdm(range(_np)):
            sel = _order[pg * ROWS:(pg + 1) * ROWS]
            fig = plt.figure(figsize=(8.5, 11.0))
            gs  = fig.add_gridspec(ROWS, 3,
                                   width_ratios=[3.0, 1.0, 2.1],
                                   left=.06, right=.97,
                                   top=.945, bottom=.035,
                                   hspace=.55, wspace=.30)
            for r, i in enumerate(sel):
                a = fig.add_subplot(gs[r, 0])
                _rw = _RAW[:, i]
                _draw_trace(a, cfg, _xcm, _rw, _RAW_se[:, i],
                            _SHORTC.get(_tmpl[i], "0.4"),
                            EDGE_CM, START_CM, REWARD_CM)

                if _gpk_bin[i] >= 0:
                    a.plot([_gpk_pos[i]], [_rw[_gpk_bin[i]]],
                           marker="v", ms=3, color="0.35", clip_on=False)
                if _pk_bin[i] >= 0:
                    a.plot([_pk_pos[i]], [_rw[_pk_bin[i]]],
                           marker="v", ms=4, color="#c0392b", clip_on=False)
                a.set_ylabel("rate", fontsize=6)
                if r == ROWS - 1 or i == sel[-1]:
                    a.set_xlabel("position (cm)", fontsize=6)

                a = fig.add_subplot(gs[r, 1])
                g  = _G[:, i]
                ge = np.nan_to_num(_Gse[:, i])
                _t = int(np.nanargmax(g)) if np.isfinite(g).all() else -1
                a.bar(range(4), g, width=.72, yerr=ge, capsize=1.4,
                      error_kw=dict(lw=.6, capthick=.6, ecolor="0.25"),
                      color=["#c0392b" if k == _t else "0.62"
                             for k in range(4)])
                a.axhline(0, color="0.4", lw=.6)
                _glo = min(0.0, float(np.nanmin(g - ge))) * 1.18 - 0.02
                _ghi = max(0.0, float(np.nanmax(g + ge))) * 1.18 + 0.02
                a.set_ylim(_glo, _ghi)
                a.set_xticks(range(4))
                a.set_xticklabels("1234", fontsize=5.5)
                a.tick_params(axis="y", labelsize=5, length=2)

                a.set_ylabel('log gain', fontsize=5.5)

                a = fig.add_subplot(gs[r, 2])
                a.axis("off")
                a.add_patch(plt.Rectangle(
                    (0, .88), 1, .13,
                    color=_HDRC.get(_label[i], "#555555"),
                    alpha=.9, transform=a.transAxes, lw=0))
                _flag = ("   [edge ramp]"  if _is_edge[i]  else
                         "   [onset ramp]" if _is_onset[i] else "")
                a.text(.02, .945,
                       "#{}   {}{}".format(int(cell_ids[i]),
                                           _DISP.get(_label[i], _label[i]),
                                           _flag),
                       transform=a.transAxes, fontsize=5.8,
                       color="w", va="center",
                       family="monospace", weight="bold")
                a.text(.02, .84, "\n".join([
                    "template   {}{}".format(
                        _DISP.get(_tmpl_shape[i], _tmpl_shape[i]),
                        "" if not np.isfinite(shp["pref_ratio"][i])
                        else "  ({:.2f}x)".format(shp["pref_ratio"][i])),
                    "  delta {}  sigma {}  stab {}  r2 {}".format(
                        _fmt(shp["delta"][i], "{:+.1f}"),
                        _fmt(shp["sigma"][i], "{:.1f}"),
                        _fmt(shp["shape_stability"][i], "{:.2f}"),
                        _fmt(shp["r2"][i], "{:+.3f}")),
                    "reliab. r  {:.2f}  (full {:.2f}){}".format(
                        rel["r"][i], rel["r_full"][i],
                        "" if rel["ok"][i] else "   NOT RELIABLE"),
                    "rel type   {}".format(rel["type"][i]),
                    "pb removes {} track  {} zones".format(
                        _fmt(100 * _PB["removed"][i], "{:.0f}%"),
                        _fmt(100 * _PB["removed_zone"][i], "{:.0f}%")),
                    "argmax(g)  L{}".format(lad["pref"][i] + 1),

                    "adapt      {:+.3f}".format(lad["adapt"][i]),
 
                    "margin     {:.4f}  (unit g, floor {:.4f}){}".format(
                        lad["margin"][i], _margin_min,
                        "   BELOW" if np.nan_to_num(lad["margin"][i], nan=0.0)
                        < _margin_min else ""),
                    "d_pref     {:+.4f}  ({:.2f}x, floor {:.4f}){}".format(
                        _d_pref[i], float(np.exp(_d_pref[i])), _dpref_min,
                        "   BELOW" if np.nan_to_num(_d_pref[i], nan=0.0)
                        < _dpref_min else ""),

                    "trace      {}{}".format(
                        "agrees" if _trace_agrees[i] else "DISAGREES",
                        "" if _trace_agrees[i] else
                        "  -- ident L{} ({}), trace {}".format(
                            _ident_j[i] + 1,
                            "shape" if _use_shape[i] else "gain",
                            _ZN[_pk[i]])),
                    "gates      {}".format(
                        "all pass -- L{}".format(_cat_j[i] + 1)
                        if _has_pref[i] else
                        "FLAT, failed: " + ", ".join(
                            ([] if np.nan_to_num(lad["margin"][i], nan=0.0)
                             >= _margin_min else ["margin"])
                            + ([] if np.nan_to_num(_d_pref[i], nan=0.0)
                               >= _dpref_min else ["d_pref"])
                            + ([] if _trace_agrees[i] else ["trace"]))),

                    "dR2 shape  {:+.4f} +/-{:.4f}  (z {:+.1f}){}{}".format(
                        shp["shape_dr2"][i], shp["shape_dr2_se"][i],
                        _z_shape[i],
                        "  <- chosen" if _win_shape[i] else "",
                        _mirror[i]),
                    "dGain      {:+.3f} +/-{:.3f}  (z {:+.1f}){}".format(
                        _dgain[i], _dgain_se[i], _z_gain[i],
                        "  <- chosen" if _win_gain[i] else ""),
                    ]),
                    transform=a.transAxes, fontsize=5.4,
                    family="monospace", va="top", linespacing=1.42)

            pdf.savefig(fig)
            plt.close(fig)

    print("  -> Wrote {} (1 behavior page + {} cell pages)".format(_PDF, _np))
    print("     EDGE-flagged  (peak past {:.0f} cm): {} of {} well-fit cells".format(
        EDGE_CM, int((_is_edge & wf).sum()), _wn))
    print("     ONSET-flagged (peak before {:.0f} cm): {} of {} well-fit cells".format(
        START_CM, int((_is_onset & wf).sum()), _wn))
    print("     LABEL FLOORS (margin {:.2f}, d_pref {:.2f}, trace veto {}): {} of {} "
          "well-fit cells called flat, {} kept a preferred landmark".format(
              _margin_min, _dpref_min,
              "on" if bool(getattr(cfg, "require_trace_agreement", True)) else "off",
              int(_floored.sum()), _wn, int(_has_pref.sum())))
    print("     TRACE VETO: {} of {} well-fit cells name a landmark the raw trace "
          "does not".format(int((wf & ~_trace_agrees).sum()), _wn))
    print("     ARBITRATION: {} agree, {} disagree, shape {} / gain {}".format(
        int(_agree.sum()), int(_dis.sum()),
        int(_win_shape.sum()), int(_win_gain.sum())))
    print("     END ZONES: {} reward, {} onset ({} of them not well_fit)".format(
        int(_is_reward.sum()), int(_is_onset_only.sum()),
        int(((_is_reward | _is_onset_only) & ~wf).sum())))

    return df9
