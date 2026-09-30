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


def _pref_gates(G, j):

    G = np.asarray(G, float)
    j = np.asarray(j, int)
    nc = G.shape[1]
    nrm = np.sqrt(np.nansum(G ** 2, axis=0))
    U = G / np.where(nrm < 1e-12, np.nan, nrm)[None, :]
    margin = np.full(nc, np.nan)
    d_pref = np.full(nc, np.nan)
    for c in np.flatnonzero(j >= 0):
        o = [k for k in range(G.shape[0]) if k != j[c]]
        margin[c] = U[j[c], c] - np.nanmax(U[o, c])
        d_pref[c] = G[j[c], c] - np.nanmean(G[o, c])

    return margin, d_pref


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

EXCLUDED_COLOR = "#c0392b"


def plot_excluded_pdf(lad, rel, cfg, cell_ids, outdir, wf, curves,
                      edge_cm=124.5, start_cm=9.0, reward_cm=134.4, stamp=None):

    xcm, CU, CU_se, RAW, RAW_se = curves[:5]
    excluded = np.flatnonzero(~np.asarray(wf, bool))
    r_rel = np.nan_to_num(np.asarray(rel["r"], float), nan=-np.inf)
    order = sorted(excluded, key=lambda c: -r_rel[c])
    per_page = EXCLUDED_ROWS * EXCLUDED_COLS
    n_pages = int(np.ceil(len(order) / per_page))
    pdf_path = os.path.join(outdir, "excluded_cells_{}.pdf".format(stamp)
                            if stamp else "excluded_cells.pdf")

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
                a.set_title("#{}   rel r {:.2f}".format(
                    int(cell_ids[i]), rel["r"][i]),
                    fontsize=6.5, color=EXCLUDED_COLOR, loc="left")

            pdf.savefig(fig)
            plt.close(fig)

    print("  -> Wrote {} ({} excluded cells on {} pages)".format(
        pdf_path, len(order), n_pages))

    return pdf_path


def plot_review_pdf(lad, shp, rel, cfg, cell_ids, outdir, wf, wf_kernel, curves,
                    shape_names=None, edge_cm=124.5, start_cm=9.0, reward_cm=134.4,
                    stamp=None):

    if shape_names is None:
        shape_names = SHAPE_NAMES_DEFAULT

    _has_shape = shp is not None
    if not _has_shape:
        _nc = len(cell_ids)
        _nan = np.full(_nc, np.nan)
        shp = dict(shape_dr2=_nan, shape_dr2_se=_nan, flat_dr2=_nan,
                   flat_dr2_se=_nan, flat_z=_nan,
                   shape_mode=np.zeros(_nc, int), shape_stability=_nan,
                   r2=_nan, delta=_nan, sigma=_nan, pref_ratio=_nan,
                   shape_runner_up=np.zeros(_nc, int),
                   land_gain=np.full((4, _nc), np.nan))

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

    _shape_j = np.where(wf, shp["shape_mode"] - 1, -99)  # -1 = four equal peaks
    _gain_j  = np.asarray(lad["pref"], int)
    _r2_shape = np.asarray(shp["r2"], float)
    _r2_gain  = np.asarray(lad["landmark_gains"], float)
    _cmp     = wf & np.isfinite(_r2_shape) & np.isfinite(_r2_gain)

    _dpref_min  = float(getattr(cfg, "pref_log_ratio_min", 0.25))

    _compete = bool(getattr(cfg, "compete_template_vs_gain", False))
    _rawmin = float(getattr(cfg, "min_raw_peak_ratio", 1.2))
    _ident_src = str(getattr(cfg, "label_identity", "raw"))
    _lsrc = str(getattr(cfg, "label_source", "gain"))

    _dgain_min = float(lad["gap_thr"]) if "gap_thr" in lad else float(
        getattr(cfg, "min_d_gain", 0.235))

    _zin = [((_xcm >= z0) & (_xcm < z1)) for z0, z1 in cfg.zones]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _zpk = np.array([np.nanmax(_RAW[m], axis=0) for m in _zin])   # (4, n)
        _top = np.nanmax(_zpk, axis=0)
    _fin4 = np.isfinite(_zpk).all(axis=0)
    _raw_j = np.where(_fin4, np.argmax(np.nan_to_num(_zpk, nan=-np.inf), axis=0), -1)
    _oth = (np.nansum(_zpk, axis=0) - np.nan_to_num(_top, nan=0.0)) / 3.0
    with np.errstate(invalid="ignore", divide="ignore"):
        _raw_ratio = np.where(_fin4 & (_top > 1e-9),
                              _top / np.where(_oth > 1e-9, _oth, np.nan), np.nan)
    _raw_ratio = np.where(_fin4 & (_top > 1e-9) & ~(_oth > 1e-9), np.inf, _raw_ratio)

    if _compete:
        _use_shape = (np.nan_to_num(_r2_shape, nan=-np.inf)
                      >= np.nan_to_num(_r2_gain, nan=-np.inf))
    else:
        _use_shape = np.zeros(len(wf), bool)     # gain fit names the landmark
    _ident_j = np.where(_use_shape, _shape_j, _gain_j)
    if not _compete and _lsrc == "raw" and _ident_src == "raw":
        _ident_j = _raw_j.copy()       # name the landmark the raw curve peaks at
    _shape_flat = wf & _use_shape & (_shape_j == -1)

    _G_gate = np.where(_use_shape[None, :], shp["land_gain"], _G)
    _margin, _d_pref = _pref_gates(_G_gate, _ident_j)

    # gain-only rule: the tallest zone has to stand min_raw_peak_ratio above
    # the mean of the other three (NaN compares False, so it drops out)
    if _compete:
        _has_pref = (wf & (_ident_j >= 0)
                     & (np.nan_to_num(_d_pref, nan=0.0) >= _dpref_min))
    elif _lsrc == "raw":
        _has_pref = wf & (_raw_ratio >= _rawmin) & (_ident_j >= 0)
    else:
        _has_pref = (wf & (_ident_j >= 0)
                     & (np.nan_to_num(lad["d_gain"], nan=-np.inf) >= _dgain_min))

    _trace_agrees = _ident_j == _pk

    _cat_j = np.where(_has_pref, _ident_j, -1)

    _, _dp_gain = _pref_gates(_G, _gain_j)

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

    _f_dpref  = np.nan_to_num(_d_pref, nan=0.0) < _dpref_min
    for _i in np.flatnonzero(_floored):
        if _shape_flat[_i]:
            _arb[_i] = "shape (flat)"
            continue
        _arb[_i] = (("d_pref" if _f_dpref[_i] else "floored") if _compete
                    else "d_gain / gain_z")

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

    # reward / onset are flags layered on the category, not categories
    _label = _tmpl.copy()

    # adapting flags
    _gd = np.diff(_G, axis=0)         # (3, n_cells)
    with np.errstate(invalid="ignore"):
        _is_visual = wf & (_label == shape_names[0])
        _is_adapting = _is_visual & (_gd < 0).all(axis=0)       # L1>L2>L3>L4
        _is_rev_adapting = _is_visual & (_gd > 0).all(axis=0)   # L1<L2<L3<L4
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
        "is_adapting":        _is_adapting,
        "is_reverse_adapting": _is_rev_adapting,
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
        "gap_threshold":      _dgain_min,
        "raw_peak_ratio":     _raw_ratio,
        "raw_peak_zone":      np.where(_raw_j >= 0, np.array(_ZN, dtype=object)[
                                  np.clip(_raw_j, 0, 3)], ""),
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
        "margin":             _margin,
        "gain_fit_margin":    lad["margin"],
        "gate_source":        np.where(_use_shape, "shape", "gain"),
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
    if not _has_shape:
        df9 = df9.drop(columns=[
            "template_shape_search", "gate_source", "shape_stability",
            "r2_shape", "shape_delta", "shape_sigma", "shape_pref_ratio",
            "shape_dR2", "shape_dR2_se", "shape_flat_dR2", "shape_flat_dR2_se",
            "shape_flat_z", "shape_runner_up", "shape_z", "r2_comb"])
    _csv = os.path.join(outdir, "v09_per_cell_full.csv")
    df9.to_csv(_csv, index=False)
    print("  -> Wrote {} ({} rows of {} cells; the {} that earn a PDF page)"
          .format(_csv, len(df9), len(_shown), int(_shown.sum())))

    _wn = int(wf.sum())

    _DISP  = {"four equal peaks": "visual",
              "L1 preference": "L1-preferring",
              "L2 preference": "L2-preferring",
              "L3 preference": "L3-preferring",
              "L4 preference": "L4-preferring",
              "unlabeled": "unlabeled"}
    _SHORTC = dict(zip(shape_names,
                       plt.cm.tab10(np.linspace(0, 1, 10))[[7, 0, 2, 4, 1]]))
    _HDRC   = dict(zip(shape_names,
                       ["#555555", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]))
    _HDRC["unlabeled"]   = "#999999"

    def _row(key, val):
        return "{:<17s}{}".format(key, val)

    def _tf(v):
        return "true" if bool(v) else "false"

    def _tpage(pdf, body, title):
        f = plt.figure(figsize=(8.5, 11.0))
        f.text(.06, .965, title, fontsize=11, weight="bold", va="top")
        f.text(.06, .935, body, fontsize=6.2, family="monospace", va="top")
        pdf.savefig(f)
        plt.close(f)

    _rel_r = np.nan_to_num(np.asarray(rel["r"], float), nan=-np.inf)

    _cat = np.where(wf, _cat_j + 1, 99)
    _sub = np.where(_is_adapting, 0, np.where(_is_rev_adapting, 1, 2))
    _order = np.lexsort((-_rel_r, _sub, _cat))
    _order = _order[_shown[_order]]
    _PDF   = os.path.join(outdir, "good_cells_{}.pdf".format(stamp)
                          if stamp else "good_cells.pdf")

    def _info_rows(i):
        rows = [_row("reliability", "{:.2f}{}".format(
            rel["r"][i], "" if rel["ok"][i] else "  NOT RELIABLE")), ""]
        if _compete:
            rows += [
                _row("best template",
                     "shape-fit" if _use_shape[i] else "gain-fit"),
                _row("template R^2", _fmt(_r2_shape[i], "{:.3f}")),
                "",
                _row("argmax(gains)",
                     "none (all too similar)"
                     if not np.nan_to_num(_dp_gain[i], nan=0.0) >= _dpref_min
                     else "L{}  ({:.2f}x)".format(
                         _gain_j[i] + 1, np.exp(_dp_gain[i]))),
                _row("gain R^2", _fmt(_r2_gain[i], "{:.3f}")),
                "",
                _row("decision uses",
                     "template" if _use_shape[i] else "gain"),
                ""]
        elif _lsrc == "raw":
            rows += [
                _row("raw peak ratio", "{}  (need {:.2f})".format(
                    _fmt(np.minimum(_raw_ratio[i], 99.0), "{:.2f}"), _rawmin)),
                _row("raw peak zone",
                     "L{}".format(_raw_j[i] + 1) if _raw_j[i] >= 0 else "n/a"),
                _row("argmax(gains)", "L{}  (gap {})".format(
                    _gain_j[i] + 1, _fmt(lad["d_gain"][i], "{:+.2f}"))),
                _row("gain R^2", _fmt(_r2_gain[i], "{:.3f}")),
                ""]
        else:
            rows += [
                _row("argmax(gains)", "{}  (gap {}, need {:.3f})".format(
                    "L{}".format(_gain_j[i] + 1) if _has_pref[i] else "none",
                    _fmt(lad["d_gain"][i], "{:+.2f}"), _dgain_min)),
                _row("gain R^2", _fmt(_r2_gain[i], "{:.3f}")),
                _row("raw peak ratio",
                     _fmt(np.minimum(_raw_ratio[i], 99.0), "{:.2f}")),
                _row("raw peak zone",
                     "L{}".format(_raw_j[i] + 1) if _raw_j[i] >= 0 else "n/a"),
                ""]
        rows += [_row("adapting", _tf(_is_adapting[i])),
                 _row("reverse adapting", _tf(_is_rev_adapting[i])),
                 _row("onset", _tf(_is_onset_only[i])),
                 _row("reward", _tf(_is_reward[i]))]

        return rows

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
                a.text(.02, .945,
                       "#{}   {}".format(int(cell_ids[i]),
                                         _DISP.get(_label[i], _label[i])),
                       transform=a.transAxes, fontsize=5.8,
                       color="w", va="center",
                       family="monospace", weight="bold")
                a.text(.02, .84, "\n".join(_info_rows(i)),
                    transform=a.transAxes, fontsize=5.4,
                    family="monospace", va="top", linespacing=1.17)

            pdf.savefig(fig)
            plt.close(fig)

    print("  -> Wrote {} (1 behavior page + {} cell pages)".format(_PDF, _np))

    return df9
