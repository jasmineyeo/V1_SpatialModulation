# -*- coding: utf-8 -*-
"""
landmarkGLM/template_matching.py

Builds and cross-validates the shape candidate family: a flat template, or
that same template plus a fitted preferred-landmark contrast.

DMM, Aug 2026
"""

import warnings

import numpy as np
from tqdm import tqdm

from candidate_kernels import fit_span_mask
from basis_funcs import lag_expand
from glm import (PoissonFold, _fit_chosen, _glm_w, _null_mu, _pick_lambda_PB,
                 _select_kernel, active_backend, dev_explained, poisson_dev)

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

PREF_CONTRAST = np.array([1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])

SHAPE_NAMES = ["four equal peaks"] + ["L{} preference".format(j + 1)
                                      for j in range(4)]


def shape_contrast(j):
    """ Mean-zero contrast putting landmark j against the other three. """

    return np.roll(PREF_CONTRAST, j)


def build_shapes(beh, cfg, lagB, XPB=None, quiet=False):

    pos = beh["pos"]
    ok = (pos >= 0) & (pos <= cfg.corridor_cm)
    iti = beh.get("iti")
    rm = beh["run_mask"] & fit_span_mask(beh, cfg)
    centers = np.array([(a + b) / 2.0 for a, b in cfg.zones])

    def _comb(w, dl, sg):
        """ Weighted sum of the four landmark bumps, zeroed outside ok/ITI. """

        c = np.zeros(len(pos))
        for wj, m0 in zip(w, centers):
            c += wj * np.exp(-0.5 * ((pos - (m0 + dl)) / sg) ** 2)
        c[~ok] = 0.0
        if iti is not None and iti.any():
            c[iti] = 0.0
        return c

    _flat = np.ones(4)
    X, par, shp = [], [], []
    for dl in cfg.delta_grid:
        for sg in cfg.sigma_grid:
            flat = _comb(_flat, dl, sg)
            D = lag_expand(flat[:, None], lagB)
            X.append(D.astype(np.float32))
            par.append((float(dl), float(sg)))
            shp.append(0)
            for j in range(4):
                con = _comb(shape_contrast(j), dl, sg)
                C = lag_expand(con[:, None], lagB)

                # orthogonalizing the contrast against the drive
                _blocks = [D[rm]]
                if XPB is not None:
                    _blocks.append(np.asarray(XPB, float)[rm])
                _blocks.append(np.ones((int(rm.sum()), 1)))
                A = np.hstack(_blocks)
                coef, *_ = np.linalg.lstsq(A, C[rm], rcond=None)
                _P = [D] + ([np.asarray(XPB, float)] if XPB is not None else [])
                _w = sum(b.shape[1] for b in _P)
                C = C - np.hstack(_P) @ coef[:_w] - coef[_w]

                rC = float(np.sqrt(np.mean(C[rm] ** 2))) if C[rm].size else 0.0
                rD = float(np.sqrt(np.mean(D[rm] ** 2))) if D[rm].size else 1.0
                if rC > 1e-12:
                    C = C * (rD / rC)

                X.append(np.hstack([D, C]).astype(np.float32))
                par.append((float(dl), float(sg)))
                shp.append(j + 1)

    return dict(X=X, par=par, shape=np.asarray(shp))


def _landmark_log_gains(Xlist, idx, Y, off, tr, lam, land_mask):
    """ Fitted log gain at each landmark, read off the model rather than a coefficient.
    """

    nc = Y.shape[1]
    G = np.full((4, nc), np.nan)
    for i in np.unique(idx):
        cells = np.where(idx == i)[0]
        W = _glm_w(Xlist[i][tr], Y[tr][:, cells], off[tr][:, cells], lam)
        eta = np.asarray(Xlist[i][tr], float) @ W          # (n_tr, len(cells))
        lm = land_mask[tr]
        for k in range(4):
            if lm[:, k].any():
                G[k, cells] = np.nanmax(eta[lm[:, k]], axis=0)

    return G


def pref_ratio_from_gains(G, pick):
    """ Rate ratio of each cell's chosen landmark against the other three.

    Parameters
    ----------
    G : np.ndarray
        (4, n_cells) log gains from _landmark_log_gains.
    pick : np.ndarray
        Chosen shape per cell, 0 = flat (returns NaN), 1..4 = L1..L4.

    Returns
    -------
    np.ndarray
        exp(gain_pref - mean(gain_other)), NaN for flat.
    """

    G = np.asarray(G, float)
    nc = G.shape[1]
    out = np.full(nc, np.nan)
    for c in range(nc):
        j = int(pick[c]) - 1
        if j < 0 or not np.isfinite(G[:, c]).all():
            continue
        others = [k for k in range(4) if k != j]
        out[c] = np.exp(G[j, c] - float(np.mean(G[others, c])))

    return out


def _shapes_fold_worker(te_lap, lap_r, Yr, XPBr, Xs, onePB, cfg,
                        shape_arr, par, LADDER_LAMBDAS, nc, nsh, land_mask):
    """Single outer fold for fit_shapes. Returns fold contribution or None if skipped."""

    te = np.isin(lap_r, te_lap)
    tr = ~te
    if tr.sum() < 500 or te.sum() < 50:
        return None

    lamPB = _pick_lambda_PB(XPBr, Yr, tr, lap_r, cfg, scale=onePB)

    OFF = next(PoissonFold(XPBr[tr], Yr[tr]).eta(XPBr, onePB, [lamPB]))

    tl = np.unique(lap_r[tr])
    nv = max(1, len(tl) // cfg.n_inner_folds)
    val = tr & np.isin(lap_r, tl[-nv:])
    fit = tr & ~val
    if fit.sum() < 200 or val.sum() < 50:
        return None

    idx, lam, _, Sful = _select_kernel(Xs, Yr, OFF, fit, val, LADDER_LAMBDAS,
                                       return_scores=True)
    Ssh_fi = np.full((nsh, nc), np.nan)
    for _s in range(nsh):
        mask = shape_arr == _s
        if mask.any():
            Ssh_fi[_s] = np.nanmax(Sful[mask], axis=0)

    Yte = Yr[te]
    dnull = poisson_dev(Yte, _null_mu(Yr[tr], OFF[tr], OFF[te],
                                      n_eval=int(te.sum())))
    dmod = poisson_dev(Yte, _fit_chosen(Xs, idx, Yr, OFF, tr, te, lam))

    pick_raw = shape_arr[idx]
    dl_fi    = np.array([par[idx[c]][0] for c in range(nc)], float)
    sg_fi    = np.array([par[idx[c]][1] for c in range(nc)], float)
    G_fi     = _landmark_log_gains(Xs, idx, Yr, OFF, tr, lam, land_mask)

    pr_fi    = pref_ratio_from_gains(G_fi, pick_raw)
    pick_fi  = np.where((pick_raw > 0) & ~(pr_fi > 1.0), 0, pick_raw)
    pr_fi    = pref_ratio_from_gains(G_fi, pick_fi)

    return dict(dnull=dnull, dmod=dmod, Ssh=Ssh_fi, pick=pick_fi,
                pick_raw=pick_raw, dl=dl_fi, sg=sg_fi, pref=pr_fi, G=G_fi)


if HAS_RAY:
    _shapes_fold_remote = ray.remote(_shapes_fold_worker)


def fit_shapes(Y, beh, cfg, XPB, fam, scalePB=None, lap_subset=None,
               verbose=True, LADDER_LAMBDAS=None):
    """ Per cell, per fold: which of the five shapes cross-validation picks.

    Poisson GLM with a log link, pure behavior carried as an offset; `Y` is the
    non-negative rate from main.
    """

    if LADDER_LAMBDAS is None:
        LADDER_LAMBDAS = np.array([1.0, 10.0, 100.0, 1000.0])

    rm = beh["run_mask"] & fit_span_mask(beh, cfg)
    if lap_subset is not None:
        rm = rm & np.isin(beh["lap_id"], lap_subset)
    Yr, XPBr = np.asarray(Y, float)[rm], XPB[rm]
    if np.any(Yr < 0):
        raise ValueError("Poisson GLM needs a non-negative target; got negatives.")
    lap_r = beh["lap_id"][rm]
    laps = np.unique(lap_r[lap_r >= 0])
    folds = np.array_split(laps, min(cfg.n_outer_folds, max(len(laps) // 3, 2)))
    nc = Yr.shape[1]

    Xs = [x[rm] for x in fam["X"]]
    onePB = np.ones(XPBr.shape[1]) if scalePB is None else np.asarray(scalePB, float)

    nsh  = int(np.max(fam["shape"])) + 1
    dmod  = np.zeros(nc)
    dnull = np.zeros(nc)

    Ssh  = np.full((len(folds), nsh, nc), np.nan)
    pick = np.full((len(folds), nc), -1, dtype=int)
    pick_raw = np.full((len(folds), nc), -1, dtype=int)
    dl_s = np.full((len(folds), nc), np.nan)
    sg_s = np.full((len(folds), nc), np.nan)
    pr_s = np.full((len(folds), nc), np.nan)
    G_s  = np.full((len(folds), 4, nc), np.nan)
    used = 0

    shape_arr  = fam["shape"]
    par        = fam["par"]

    land_mask = np.zeros((len(lap_r), 4), bool)
    _pos_r = beh["pos"][rm]
    for _k, (_z0, _z1) in enumerate(cfg.zones):
        land_mask[:, _k] = (_pos_r >= _z0) & (_pos_r < _z1)

    if HAS_RAY and active_backend() != "torch":
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        if verbose:
            print("  Shape templates: {} folds in parallel (Ray)...".format(len(folds)))
        Yr_ref   = ray.put(Yr)
        XPBr_ref = ray.put(XPBr)
        Xs_ref   = ray.put(Xs)
        lr_ref   = ray.put(lap_r)
        lm_ref   = ray.put(land_mask)
        futures = [
            _shapes_fold_remote.remote(
                list(map(int, te_lap)), lr_ref, Yr_ref, XPBr_ref,
                Xs_ref, onePB, cfg, shape_arr, par,
                LADDER_LAMBDAS, nc, nsh, lm_ref)
            for te_lap in folds]
        results = ray.get(futures)
        for fi, res in enumerate(results):
            if res is None:
                continue
            dnull    += res["dnull"]
            dmod     += res["dmod"]
            Ssh[fi]   = res["Ssh"]
            pick[fi]  = res["pick"]
            pick_raw[fi] = res["pick_raw"]
            dl_s[fi]  = res["dl"]
            sg_s[fi]  = res["sg"]
            pr_s[fi]  = res["pref"]
            G_s[fi]   = res["G"]
            used += 1
    else:
        for fi, te_lap in tqdm(enumerate(folds), total=len(folds), desc="  Shape templates",
                               disable=not verbose):
            res = _shapes_fold_worker(
                list(map(int, te_lap)), lap_r, Yr, XPBr,
                Xs, onePB, cfg, shape_arr, par,
                LADDER_LAMBDAS, nc, nsh, land_mask)
            if res is None:
                continue
            dnull    += res["dnull"]
            dmod     += res["dmod"]
            Ssh[fi]   = res["Ssh"]
            pick[fi]  = res["pick"]
            pick_raw[fi] = res["pick_raw"]
            dl_s[fi]  = res["dl"]
            sg_s[fi]  = res["sg"]
            pr_s[fi]  = res["pref"]
            G_s[fi]   = res["G"]
            used += 1

    ok     = pick >= 0
    cnt    = np.stack([(pick == s).sum(axis=0) for s in range(nsh)])

    mode   = np.argmax(cnt, axis=0)
    nvalid = np.maximum(ok.sum(axis=0), 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _prm = np.nanmedian(pr_s, axis=0)
        _Gm = np.nanmedian(G_s, axis=0)

    _flip = int(np.sum((pick_raw > 0) & (pick == 0)))
    _npref = int(np.sum(pick_raw > 0))

    _ar = np.arange(nc)
    _zmin = float(getattr(cfg, "shape_flat_z_min", 1.0))
    with np.errstate(invalid="ignore"):
        Sm    = np.nanmean(Ssh, axis=0)   # (nsh, nc)
        _fin  = np.isfinite(Ssh)
        _tot  = np.nansum(Ssh, axis=0)
        _cnt  = _fin.sum(axis=0)

        fl_f = np.full((len(folds), nc), np.nan)
        for _f in range(len(folds)):
            if not _fin[_f].any():
                continue
            _loo = (_tot - np.nan_to_num(Ssh[_f])) / np.maximum(
                _cnt - _fin[_f], 1)
            _loo[0] = -np.inf
            _pf = np.argmax(_loo, axis=0)
            fl_f[_f] = Ssh[_f, _pf, _ar] - Ssh[_f, 0, _ar]
        flat_dr2 = np.nanmean(fl_f, axis=0)
        flat_dr2_se = np.nanstd(fl_f, axis=0) / np.sqrt(
            np.maximum(np.sum(np.isfinite(fl_f), axis=0), 1))
        flat_z = flat_dr2 / np.where(flat_dr2_se < 1e-12, np.nan, flat_dr2_se)

        mode = np.where(np.nan_to_num(flat_z, nan=-np.inf) >= _zmin, mode, 0)

        dr2_f = np.full((len(folds), nc), np.nan)
        for _f in range(len(folds)):
            if not _fin[_f].any():
                continue
            _loo = (_tot - np.nan_to_num(Ssh[_f])) / np.maximum(
                _cnt - _fin[_f], 1)
            _loo[mode, _ar] = -np.inf
            _rf = np.argmax(_loo, axis=0)
            dr2_f[_f] = Ssh[_f, mode, _ar] - Ssh[_f, _rf, _ar]
        Sm2    = Sm.copy(); Sm2[mode, _ar] = -np.inf
        runner = np.argmax(Sm2, axis=0)
        dr2    = np.nanmean(dr2_f, axis=0)
        dr2_se = np.nanstd(dr2_f, axis=0) / np.sqrt(
            np.maximum(np.sum(np.isfinite(dr2_f), axis=0), 1))

    _prm = np.where(mode > 0, _prm, np.nan)

    if verbose:
        print("  Contrast sign: {} of {} preference picks came back negative "
              "and were handed to flat.".format(_flip, max(_npref, 1)))
        print("  Flat-as-null gate at z >= {:.1f}: {} of {} cells kept a "
              "preference.".format(
                  _zmin, int(np.sum(mode > 0)), nc))

    return dict(r2=dev_explained(dmod, dnull),
                pick=pick, pick_raw=pick_raw, shape_mode=mode,
                flat_dr2=flat_dr2, flat_dr2_se=flat_dr2_se, flat_z=flat_z,
                shape_stability=cnt[mode, np.arange(nc)] / nvalid,
                shape_counts=cnt, r2_by_shape=Sm, shape_dr2=dr2,
                shape_dr2_se=dr2_se, shape_runner_up=runner,
                delta=np.nanmedian(dl_s, axis=0), sigma=np.nanmedian(sg_s, axis=0),
                pref_ratio=_prm, pref_ratio_per_fold=pr_s,
                land_gain=_Gm, land_gain_per_fold=G_s,
                n_folds_used=used)
