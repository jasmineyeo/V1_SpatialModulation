# -*- coding: utf-8 -*-
"""
landmarkGLM/fit_landmark_gains.py

Cross-validated comparison of the tied comb against the untied landmark-gains
family, and the four per-landmark gains the latter reads off its fitted log rate
in each landmark zone.

DMM, Aug 2026
"""

import warnings

import numpy as np
from tqdm import tqdm

from candidate_kernels import fit_span_mask
from glm import (_fit_chosen, _null_mu, behavior_offset,
                 _select_kernel, active_backend, dev_explained, score_loss)
from build_landmark_gains import shape_stats
from template_matching import (_landmark_kernel_gains, _landmark_log_gains,
                                zone_mask)

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def _fold_split(te_lap, lap_r, cfg):

    te = np.isin(lap_r, te_lap)
    tr = ~te
    if tr.sum() < 500 or te.sum() < 50:
        return None
    tl = np.unique(lap_r[tr])
    nv = max(1, len(tl) // cfg.n_inner_folds)
    val = tr & np.isin(lap_r, tl[-nv:])
    fit = tr & ~val
    if fit.sum() < 200 or val.sum() < 50:
        return None

    return te, tr, val, fit


def _select_fold_worker(te_lap, lap_r, Yr, XPBr, X_lgain, onePB, cfg,
                        LADDER_LAMBDAS):

    sp = _fold_split(te_lap, lap_r, cfg)
    if sp is None:
        return None
    te, tr, val, fit = sp
    OFF = behavior_offset(XPBr, Yr, tr, lap_r, cfg, scale=onePB)
    metric = getattr(cfg, "score_metric", "mse")
    _, lam, _, Sj = _select_kernel(X_lgain, Yr, OFF, tr, te, LADDER_LAMBDAS,
                                   return_scores=True, metric=metric)

    return dict(S=Sj, lam=float(lam))


def _landmark_gains_fold_worker(te_lap, lap_r, Yr, XPBr, X_comb, X_lgain, onePB, cfg,
                       par, land_mask, LADDER_LAMBDAS, nc, lagB=None,
                       fixed_idx=None, fixed_lam=None):

    sp = _fold_split(te_lap, lap_r, cfg)
    if sp is None:
        return None
    te, tr, val, fit = sp

    OFF = behavior_offset(XPBr, Yr, tr, lap_r, cfg, scale=onePB)

    metric = getattr(cfg, "score_metric", "mse")
    X = {"landmark_gains": X_lgain}
    if X_comb is not None:
        X["comb"] = X_comb
    idx_k, lam_k = {}, {}
    for k in X:
        if k == "landmark_gains" and fixed_idx is not None:
            idx_k[k], lam_k[k] = fixed_idx, fixed_lam
            continue
        idx_k[k], lam_k[k], _ = _select_kernel(X[k], Yr, OFF, fit, val,
                                               LADDER_LAMBDAS, metric=metric)

    Yte = Yr[te]
    dnull = score_loss(Yte, _null_mu(Yr[tr], OFF[tr], OFF[te],
                                     n_eval=int(te.sum())), metric)
    dmod = {k: score_loss(Yte, _fit_chosen(X[k], idx_k[k], Yr, OFF, tr, te,
                                           lam_k[k]), metric) for k in X}

    if getattr(cfg, "gain_method", "kernel") == "kernel" and lagB is not None:
        gains_fi = _landmark_kernel_gains(X_lgain, idx_k["landmark_gains"], Yr,
                                          OFF, tr, lam_k["landmark_gains"], lagB)
    else:
        gains_fi = _landmark_log_gains(X_lgain, idx_k["landmark_gains"], Yr, OFF,
                                       tr, lam_k["landmark_gains"], land_mask)
    dl_fi = np.array([par[idx_k["landmark_gains"][c]][0] for c in range(nc)])

    return dict(dnull=dnull, dmod=dmod, gains=gains_fi, dl=dl_fi)


if HAS_RAY:
    _landmark_gains_fold_remote = ray.remote(_landmark_gains_fold_worker)
    _select_fold_remote = ray.remote(_select_fold_worker)


def _run_folds(worker, remote, folds, shared, shared_ref, per_fold, use_ray,
               verbose, desc):
    """ Run one worker over every outer fold, on Ray or serially. """

    if use_ray:
        futs = [remote.remote(list(map(int, te)), *shared_ref, *pf)
                for te, pf in zip(folds, per_fold)]
        return ray.get(futs)

    return [worker(list(map(int, te)), *shared, *pf)
            for te, pf in tqdm(zip(folds, per_fold), total=len(folds),
                               desc=desc, disable=not verbose)]


def fit_landmark_gains(Y, beh, cfg, XPB, cand, lgain_fam, scalePB=None,
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

    compete = bool(getattr(cfg, "compete_template_vs_gain", False))
    X_comb  = [x[rm] for x in cand["comb"]["X"]] if compete else None
    X_lgain = [x[rm] for x in lgain_fam["X"]]
    onePB = np.ones(XPBr.shape[1]) if scalePB is None else np.asarray(scalePB, float)
    land_mask = zone_mask(beh["pos"][rm], cfg)

    dmod  = {"landmark_gains": np.zeros(nc)}
    if compete:
        dmod["comb"] = np.zeros(nc)
    dnull = np.zeros(nc)
    gains  = np.full((len(folds), 4, nc), np.nan)
    dl_sel = np.full((len(folds), nc), np.nan)
    used   = 0

    use_ray = HAS_RAY and active_backend() != "torch"
    if use_ray:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        if verbose:
            print("  Landmark gains: {} folds in parallel (Ray)...".format(len(folds)))
        Yr_ref   = ray.put(Yr)
        XPBr_ref = ray.put(XPBr)
        Xc_ref   = ray.put(X_comb)
        Xlg_ref  = ray.put(X_lgain)
        lr_ref   = ray.put(lap_r)
        lm_ref   = ray.put(land_mask)
    else:
        Yr_ref = XPBr_ref = Xc_ref = Xlg_ref = lr_ref = lm_ref = None

    par = lgain_fam["par"]
    lagB = lgain_fam.get("lagB")
    fixed = bool(getattr(cfg, "fixed_hyperparams_per_cell", True))
    idx_fixed = None
    lam_f = [None] * len(folds)
    if fixed:
        # score every shift/width candidate on each fold's held-out
        # laps, then take one candidate per cell across folds
        sel = _run_folds(
            _select_fold_worker, _select_fold_remote if HAS_RAY else None, folds,
            (lap_r, Yr, XPBr, X_lgain, onePB, cfg, LADDER_LAMBDAS),
            (lr_ref, Yr_ref, XPBr_ref, Xlg_ref, onePB, cfg, LADDER_LAMBDAS),
            [()] * len(folds), use_ray, verbose, "  Selecting shift/width")
        got = [r for r in sel if r is not None]
        if got:
            Ss = np.stack([r["S"] for r in got])          # (folds, cand, cells)
            Ss = np.where(np.isfinite(Ss), Ss, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                Sm = np.nanmean(Ss, axis=0)
            idx_fixed = np.argmax(np.nan_to_num(Sm, nan=-np.inf), axis=0)
            lam_f = [None if r is None else r["lam"] for r in sel]
        else:
            fixed = False

    # another pass... refit, score held-out laps, read gains
    if fixed:
        keep = [fi for fi, r in enumerate(sel) if r is not None]
        per_fold = [(lam_f[fi],) for fi in keep]
        fl = [folds[fi] for fi in keep]
        shared_extra = (idx_fixed,)
    else:
        keep = list(range(len(folds)))
        per_fold = [()] * len(folds)
        fl = folds
        shared_extra = ()
    res_all = _run_folds(
        _landmark_gains_fold_worker,
        _landmark_gains_fold_remote if HAS_RAY else None, fl,
        (lap_r, Yr, XPBr, X_comb, X_lgain, onePB, cfg, par, land_mask,
         LADDER_LAMBDAS, nc, lagB) + shared_extra,
        (lr_ref, Yr_ref, XPBr_ref, Xc_ref, Xlg_ref, onePB, cfg, par, lm_ref,
         LADDER_LAMBDAS, nc, lagB) + shared_extra,
        per_fold, use_ray, verbose, "  Landmark gains")
    for fi, res in zip(keep, res_all):
        if res is None:
            continue
        dnull += res["dnull"]
        for k in dmod:
            dmod[k] += res["dmod"][k]
        gains[fi]  = res["gains"]
        dl_sel[fi] = res["dl"]
        used += 1

    out = {k: dev_explained(dmod[k], dnull) for k in dmod}
    out["gain"]          = np.nanmedian(gains, axis=0)
    out["gain_per_fold"] = gains
    out["delta"]         = np.nanmedian(dl_sel, axis=0)
    if idx_fixed is not None:
        out["sigma"]     = np.array([par[i][1] for i in idx_fixed], float)
        out["cand_idx"]  = idx_fixed
    out["n_folds_used"]  = used
    if "comb" not in out:
        out["comb"] = np.full(nc, np.nan)
    out["d_pref"]        = out["landmark_gains"] - out["comb"]
    out.update(shape_stats(gains))

    return out


def gain_gap_null(Y, beh, cfg, XPB, cand, lgain_fam, scalePB, LADDER_LAMBDAS,
                  cells, n_shuffles=5, seed=0, verbose=True):
    """ How big a top-landmark gap does noise alone produce?
    """

    Yc = np.asarray(Y, float)[:, cells]
    nf, nc = Yc.shape
    lo = int(round(cfg.rel_min_shift_s * cfg.fps))
    rng = np.random.default_rng(seed)
    out = np.full((n_shuffles, nc), np.nan)
    for s in range(n_shuffles):
        if verbose:
            print("  Gap null: shuffle {} of {}...".format(s + 1, n_shuffles))
        sh = rng.integers(lo, nf - lo, size=nc)
        ix = (np.arange(nf)[:, None] - sh[None, :]) % nf
        lad = fit_landmark_gains(np.take_along_axis(Yc, ix, axis=0), beh, cfg,
                                 XPB, cand, lgain_fam, scalePB=scalePB,
                                 verbose=False, LADDER_LAMBDAS=LADDER_LAMBDAS)
        out[s] = lad["d_gain"]

    return out
