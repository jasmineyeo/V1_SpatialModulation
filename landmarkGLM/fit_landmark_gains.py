# -*- coding: utf-8 -*-
"""
landmarkGLM/fit_landmark_gains.py

Cross-validated comparison of the tied comb against the untied landmark-gains
family, and the four per-landmark gains the latter reads off its fitted kernels.

DMM, Aug 2026
"""

import numpy as np
from tqdm import tqdm

from candidate_kernels import fit_span_mask
from glm import (PoissonFold, _fit_chosen, _glm_w, _null_mu, _pick_lambda_PB,
                 _select_kernel, active_backend, dev_explained, poisson_dev)
from build_landmark_gains import shape_stats

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def _peak_gains(Xlist, idx, Y, off, tr, lam, lagB, p):
    """ Peak of each feature's temporal kernel, per cell. IN LOG UNITS.
    """

    nc = Y.shape[1]
    G = np.full((p, nc), np.nan)
    for i in np.unique(idx):
        cells = np.where(idx == i)[0]
        W = _glm_w(Xlist[i][tr], Y[tr][:, cells], off[tr][:, cells], lam)
        for j in range(p):
            K = lagB @ W[j::p, :]                             # (n_lags, cells)
            G[j, cells] = np.nanmax(K, axis=0)

    return G


def _landmark_gains_fold_worker(te_lap, lap_r, Yr, XPBr, X_comb, X_lgain, onePB, cfg,
                       par, lagB, LADDER_LAMBDAS, nc):
    """Single outer fold for fit_landmark_gains. Returns fold contribution or None if skipped.
    """
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

    X = {"comb": X_comb, "landmark_gains": X_lgain}
    idx_k, lam_k = {}, {}
    for k in X:
        idx_k[k], lam_k[k], _ = _select_kernel(X[k], Yr, OFF, fit, val,
                                               LADDER_LAMBDAS)

    Yte = Yr[te]
    dnull = poisson_dev(Yte, _null_mu(Yr[tr], OFF[tr], OFF[te],
                                      n_eval=int(te.sum())))
    dmod = {k: poisson_dev(Yte, _fit_chosen(X[k], idx_k[k], Yr, OFF, tr, te,
                                            lam_k[k])) for k in X}
    gains_fi = _peak_gains(X_lgain, idx_k["landmark_gains"], Yr, OFF, tr,
                           lam_k["landmark_gains"], lagB, 4)
    dl_fi = np.array([par[idx_k["landmark_gains"][c]][0] for c in range(nc)])

    return dict(dnull=dnull, dmod=dmod, gains=gains_fi, dl=dl_fi)


if HAS_RAY:
    _landmark_gains_fold_remote = ray.remote(_landmark_gains_fold_worker)


def fit_landmark_gains(Y, beh, cfg, XPB, cand, lgain_fam, lagB, scalePB=None,
              lap_subset=None, verbose=True, LADDER_LAMBDAS=None):
    """ Held-out deviance explained for comb and landmark_gains, plus the four
    per-landmark gains per fold.
    """

    if LADDER_LAMBDAS is None:
        LADDER_LAMBDAS = np.array([1.0, 10.0, 100.0, 1000.0])

    rm = beh["run_mask"] & fit_span_mask(beh, cfg)
    if lap_subset is not None:
        rm = rm & np.isin(beh["lap_id"], lap_subset)
    Yr, XPBr = np.asarray(Y, float)[rm], XPB[rm]
    if np.any(Yr < 0):
        raise ValueError("Poisson GLM needs a non-negative target... got negatives.")
    lap_r = beh["lap_id"][rm]
    laps = np.unique(lap_r[lap_r >= 0])
    folds = np.array_split(laps, min(cfg.n_outer_folds, max(len(laps) // 3, 2)))
    nc = Yr.shape[1]

    X_comb  = [x[rm] for x in cand["comb"]["X"]]
    X_lgain = [x[rm] for x in lgain_fam["X"]]
    onePB = np.ones(XPBr.shape[1]) if scalePB is None else np.asarray(scalePB, float)

    dmod  = {"comb": np.zeros(nc), "landmark_gains": np.zeros(nc)}
    dnull = np.zeros(nc)
    gains  = np.full((len(folds), 4, nc), np.nan)
    dl_sel = np.full((len(folds), nc), np.nan)
    used   = 0

    if HAS_RAY and active_backend() != "torch":
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        if verbose:
            print("  Landmark gains: {} folds in parallel (Ray)...".format(len(folds)))
        Yr_ref   = ray.put(Yr)
        XPBr_ref = ray.put(XPBr)
        Xc_ref   = ray.put(X_comb)
        Xlg_ref  = ray.put(X_lgain)
        lr_ref   = ray.put(lap_r)
        lagB_ref = ray.put(lagB)
        futures = [
            _landmark_gains_fold_remote.remote(
                list(map(int, te_lap)), lr_ref, Yr_ref, XPBr_ref,
                Xc_ref, Xlg_ref, onePB, cfg, lgain_fam["par"], lagB_ref,
                LADDER_LAMBDAS, nc)
            for te_lap in folds]
        results = ray.get(futures)
        for fi, res in enumerate(results):
            if res is None:
                continue
            dnull         += res["dnull"]
            dmod["comb"]  += res["dmod"]["comb"]
            dmod["landmark_gains"] += res["dmod"]["landmark_gains"]
            gains[fi]      = res["gains"]
            dl_sel[fi]     = res["dl"]
            used += 1
    else:
        for fi, te_lap in tqdm(enumerate(folds), total=len(folds), desc="  Landmark gains",
                               disable=not verbose):
            res = _landmark_gains_fold_worker(
                list(map(int, te_lap)), lap_r, Yr, XPBr,
                X_comb, X_lgain, onePB, cfg, lgain_fam["par"], lagB,
                LADDER_LAMBDAS, nc)
            if res is None:
                continue
            dnull         += res["dnull"]
            dmod["comb"]  += res["dmod"]["comb"]
            dmod["landmark_gains"] += res["dmod"]["landmark_gains"]
            gains[fi]      = res["gains"]
            dl_sel[fi]     = res["dl"]
            used += 1

    out = {k: dev_explained(dmod[k], dnull) for k in dmod}
    out["gain"]          = np.nanmedian(gains, axis=0)
    out["gain_per_fold"] = gains
    out["delta"]         = np.nanmedian(dl_sel, axis=0)
    out["n_folds_used"]  = used
    out["d_pref"]        = out["landmark_gains"] - out["comb"]
    out.update(shape_stats(gains))

    return out
