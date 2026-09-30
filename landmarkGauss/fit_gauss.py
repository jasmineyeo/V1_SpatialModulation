# -*- coding: utf-8 -*-
"""
landmarkGauss/fit_gauss.py

DMM, Sept 2026
"""

import itertools
import warnings

import numpy as np
import scipy.sparse
from scipy.optimize import least_squares
from tqdm import tqdm


def gauss_model(x, centers, delta, sigma, A, b):

    G = np.exp(-(np.asarray(x)[:, None] - (centers + delta)[None, :]) ** 2
               / (2.0 * sigma ** 2))
    return G @ A + b


def lap_bin_sums(Y, beh, edges, use):

    nb = len(edges) - 1
    pos = beh["pos"]
    lap = beh["lap_id"]
    use = use & (pos >= edges[0]) & (pos < edges[-1])
    laps = np.unique(lap[use])
    li = np.searchsorted(laps, lap[use])
    bi = np.clip(np.digitize(pos[use], edges) - 1, 0, nb - 1)
    rows = li * nb + bi
    fr = np.flatnonzero(use)
    M = scipy.sparse.csr_matrix(
        (np.ones(len(fr)), (rows, fr)), shape=(len(laps) * nb, Y.shape[0]))
    S = np.asarray(M @ Y).reshape(len(laps), nb, Y.shape[1])
    N = np.asarray(M.sum(axis=1)).reshape(len(laps), nb)

    return S, N, laps


def mean_trace(S, N, sel=None):

    if sel is not None:
        S, N = S[sel], N[sel]
    s, n = S.sum(axis=0), N.sum(axis=0)
    T = np.full(s.shape, np.nan)
    ok = n > 0
    T[ok] = s[ok] / n[ok, None]
    if not ok.all() and ok.sum() >= 2:
        x = np.arange(len(n))
        for c in range(T.shape[1]):
            T[~ok, c] = np.interp(x[~ok], x[ok], T[ok, c])

    return T


def lap_sem(S, N):

    with np.errstate(invalid="ignore", divide="ignore"):
        L = S / N[:, :, None]
    L[N == 0] = np.nan
    n = np.sum(N > 0, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sem = np.nanstd(L, axis=0) / np.sqrt(np.maximum(n, 1))[:, None]

    return mean_trace(S, N), sem


class GaussGrid:

    def __init__(self, x, centers, cfg, tied=False):

        self.x = np.asarray(x, float)
        self.centers = np.asarray(centers, float)
        self.tied = tied
        self.fit_base = cfg.gauss_baseline_mode == "fit"
        self.deltas = np.arange(-cfg.gauss_delta_max_cm,
                                cfg.gauss_delta_max_cm + 1e-9,
                                cfg.gauss_delta_step_cm)
        self.sigmas = np.arange(cfg.gauss_sigma_min_cm,
                                cfg.gauss_sigma_max_cm + 1e-9,
                                cfg.gauss_sigma_step_cm)

        nk = 1 if tied else len(self.centers)
        self.ncol = nk + int(self.fit_base)
        subsets = [s for r in range(1, self.ncol + 1)
                   for s in itertools.combinations(range(self.ncol), r)]

        self.grid = []
        for d in self.deltas:
            for s in self.sigmas:
                X = self.design(d, s)
                subs = [(np.array(sub), X[:, sub], np.linalg.pinv(X[:, sub]))
                        for sub in subsets]
                self.grid.append((d, s, subs))

    def design(self, delta, sigma):

        G = np.exp(-(self.x[:, None] - (self.centers + delta)[None, :]) ** 2
                   / (2.0 * sigma ** 2))
        if self.tied:
            G = G.sum(axis=1, keepdims=True)
        if self.fit_base:
            G = np.hstack([G, np.ones((len(self.x), 1))])
        return G

    def fit(self, T):

        nc = T.shape[1]
        best_sse = np.sum(T ** 2, axis=0)
        best_d = np.zeros(nc)
        best_s = np.full(nc, self.sigmas[0])
        best_C = np.zeros((self.ncol, nc))

        for d, s, subs in self.grid:
            for sub, Xs, P in subs:
                C = P @ T
                feas = np.all(C >= -1e-10, axis=0)
                if not feas.any():
                    continue
                sse = np.sum((T - Xs @ C) ** 2, axis=0)
                win = feas & (sse < best_sse)
                if win.any():
                    best_sse[win] = sse[win]
                    best_d[win] = d
                    best_s[win] = s
                    best_C[:, win] = 0.0
                    best_C[np.ix_(sub, np.flatnonzero(win))] = C[:, win]

        return dict(delta=best_d, sigma=best_s,
                    coef=np.maximum(best_C, 0.0), sse=best_sse)

    def unpack(self, fit, b0=None):

        nc = fit["coef"].shape[1]
        nk = len(self.centers)
        if self.tied:
            A = np.repeat(fit["coef"][:1], nk, axis=0)
        else:
            A = fit["coef"][:nk]
        b = fit["coef"][-1] if self.fit_base else np.zeros(nc)
        if b0 is not None:
            b = b + b0
        return A, b

    def predict(self, fit, x, b0=None):

        A, b = self.unpack(fit, b0)
        out = np.empty((len(x), A.shape[1]))
        for c in range(A.shape[1]):
            out[:, c] = gauss_model(x, self.centers, fit["delta"][c],
                                    fit["sigma"][c], A[:, c], b[c])
        return out


def _baseline(T, cfg):

    if cfg.gauss_baseline_mode == "percentile":
        return np.percentile(T, cfg.gauss_baseline_pct, axis=0)
    return None


def _fit_trace(grid, T, cfg):

    b0 = _baseline(T, cfg)
    fit = grid.fit(T if b0 is None else T - b0[None, :])
    fit["b0"] = b0
    return fit


def refine(grid, fit, T, cfg):

    x, cen = grid.x, grid.centers
    dmax = cfg.gauss_delta_max_cm
    smin, smax = cfg.gauss_sigma_min_cm, cfg.gauss_sigma_max_cm
    b0 = fit["b0"]
    A, b = grid.unpack(fit)
    nc = T.shape[1]
    out = dict(delta=fit["delta"].copy(), sigma=fit["sigma"].copy(),
               coef=fit["coef"].copy(), sse=fit["sse"].copy(), b0=b0)

    for c in range(nc):
        y = T[:, c] - (0.0 if b0 is None else b0[c])
        p0 = [fit["delta"][c], fit["sigma"][c], *A[:, c]]
        lo = [-dmax, smin, 0, 0, 0, 0]
        hi = [dmax, smax, np.inf, np.inf, np.inf, np.inf]
        if grid.fit_base:
            p0.append(b[c])
            lo.append(0.0)
            hi.append(np.inf)
        p0 = np.clip(p0, np.array(lo) + 1e-9,
                     np.where(np.isfinite(hi), np.array(hi) - 1e-9, np.inf))

        def res(p):
            bb = p[6] if grid.fit_base else 0.0
            return gauss_model(x, cen, p[0], p[1], p[2:6], bb) - y

        try:
            r = least_squares(res, p0, bounds=(lo, hi), method="trf")
        except Exception:
            continue
        sse = float(np.sum(r.fun ** 2))
        if sse < out["sse"][c]:
            out["delta"][c], out["sigma"][c] = r.x[0], r.x[1]
            out["coef"][:4, c] = r.x[2:6]
            if grid.fit_base:
                out["coef"][4, c] = r.x[6]
            out["sse"][c] = sse

    return out


def amplitude_indices(A):

    A = np.maximum(np.asarray(A, float), 0.0)
    nk, nc = A.shape
    srt = -np.sort(-A, axis=0)
    a1, a2, amin = srt[0], srt[1], srt[-1]
    aoth = (A.sum(axis=0) - a1) / (nk - 1)

    def _ratio(p, q):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(p + q > 0, (p - q) / (p + q), np.nan)

    tot = A.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        neff = np.where(tot > 0, tot ** 2 / np.sum(A ** 2, axis=0), np.nan)
        # 0 = equal amplitudes at all landmarks, 1 = a single landmark
        sparse = (nk - neff) / (nk - 1)

    pref = np.where(tot > 0, np.argmax(A, axis=0) + 1, 0)

    return dict(pref=pref, psi_2nd=_ratio(a1, a2), psi_min=_ratio(a1, amin),
                psi_mean=_ratio(a1, aoth), n_eff=neff, sparseness=sparse)


def _r2(T, P):

    sst = np.sum((T - T.mean(axis=0)) ** 2, axis=0)
    sse = np.sum((T - P) ** 2, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sst > 0, 1.0 - sse / sst, np.nan)


def _corr(a, b):

    a = a - a.mean(axis=0)
    b = b - b.mean(axis=0)
    d = np.sqrt(np.sum(a ** 2, axis=0) * np.sum(b ** 2, axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(d > 1e-12, np.sum(a * b, axis=0) / d, np.nan)


def fit_all(Y, beh, cfg):

    centers = cfg.gauss_centers()
    lo, hi = cfg.gauss_span()
    nb = max(int(round((hi - lo) / cfg.gauss_bin_cm)), 8)
    edges = np.linspace(lo, hi, nb + 1)
    x = 0.5 * (edges[:-1] + edges[1:])

    use = beh["run_mask"] & (beh["lap_id"] >= 0) & np.isfinite(beh["pos"])
    S, N, laps = lap_bin_sums(Y, beh, edges, use)

    nbd = int(round(cfg.corridor_cm / cfg.gauss_bin_cm))
    edges_d = np.linspace(0, cfg.corridor_cm, nbd + 1)
    Sd, Nd, _ = lap_bin_sums(Y, beh, edges_d, use)
    Td, semd = lap_sem(Sd, Nd)
    del Sd, Nd
    nl, nc = len(laps), Y.shape[1]
    T = mean_trace(S, N)

    g_free = GaussGrid(x, centers, cfg, tied=False)
    g_tied = GaussGrid(x, centers, cfg, tied=True)
    print("    {} grid points x {} support sets (free)".format(
        len(g_free.grid), len(g_free.grid[0][2])))

    f_free = _fit_trace(g_free, T, cfg)
    if cfg.gauss_refine:
        f_free = refine(g_free, f_free, T, cfg)
    f_tied = _fit_trace(g_tied, T, cfg)
    A, b = g_free.unpack(f_free, f_free["b0"])
    At, bt = g_tied.unpack(f_tied, f_tied["b0"])
    r2_free = _r2(T, g_free.predict(f_free, x, f_free["b0"]))
    r2_tied = _r2(T, g_tied.predict(f_tied, x, f_tied["b0"]))
    idx = amplitude_indices(A)

    rng = np.random.default_rng(cfg.gauss_seed)
    ns = cfg.gauss_n_splits
    cv_free = np.zeros((ns, 2, nc))
    cv_tied = np.zeros((ns, 2, nc))
    pref_h = np.zeros((ns, 2, nc), int)
    psi_h = np.zeros((ns, 2, nc))
    split_r = np.zeros((ns, nc))
    splits = []
    for s in tqdm(range(ns), desc="    CV splits"):
        perm = rng.permutation(nl)
        h = [np.sort(perm[:nl // 2]), np.sort(perm[nl // 2:])]
        splits.append(h)
        Th = [mean_trace(S, N, h[0]), mean_trace(S, N, h[1])]
        split_r[s] = _corr(Th[0], Th[1])
        for i in range(2):
            tr, te = Th[i], Th[1 - i]
            ff = _fit_trace(g_free, tr, cfg)
            ft = _fit_trace(g_tied, tr, cfg)
            cv_free[s, i] = _r2(te, g_free.predict(ff, x, ff["b0"]))
            cv_tied[s, i] = _r2(te, g_tied.predict(ft, x, ft["b0"]))
            ih = amplitude_indices(g_free.unpack(ff)[0])
            pref_h[s, i] = ih["pref"]
            psi_h[s, i] = ih["psi_2nd"]

    r2cv_free = np.nanmean(cv_free, axis=(0, 1))
    r2cv_tied = np.nanmean(cv_tied, axis=(0, 1))
    dr2 = np.nanmean(cv_free - cv_tied, axis=(0, 1))
    consistency = np.mean((pref_h[:, 0] == pref_h[:, 1]) & (pref_h[:, 0] > 0),
                          axis=0)
    rel_r = np.nanmean(split_r, axis=0)

    nf = Y.shape[0]
    shift_lo = int(round(cfg.rel_min_shift_s * cfg.fps))
    null = []
    for k in range(cfg.gauss_rel_n_shuffles):
        sh = rng.integers(shift_lo, nf - shift_lo, size=nc)
        ix = (np.arange(nf)[:, None] - sh[None, :]) % nf
        Ssh, Nsh, _ = lap_bin_sums(np.take_along_axis(Y, ix, axis=0),
                                   beh, edges, use)
        rr = np.zeros((ns, nc))
        for s, h in enumerate(splits):
            rr[s] = _corr(mean_trace(Ssh, Nsh, h[0]), mean_trace(Ssh, Nsh, h[1]))
        null.append(np.nanmean(rr, axis=0))
    null = np.concatenate(null)
    rel_thr = float(np.nanpercentile(null, cfg.gauss_rel_null_pct))
    reliable = (np.nan_to_num(rel_r, nan=-1) > rel_thr) & (
        np.nan_to_num(rel_r, nan=-1) >= cfg.gauss_rel_min_r)
    good = reliable & (np.nan_to_num(r2_free) >= cfg.gauss_min_fit_r2)

    return dict(
        disp_x=0.5 * (edges_d[:-1] + edges_d[1:]), disp_T=Td, disp_sem=semd,
        x=x, edges=edges, centers=centers, span=(lo, hi), T=T, laps=laps,
        delta=f_free["delta"], sigma=f_free["sigma"], A=A, b=b,
        delta_tied=f_tied["delta"], sigma_tied=f_tied["sigma"],
        A_tied=At[0], b_tied=bt,
        r2_free=r2_free, r2_tied=r2_tied,
        r2cv_free=r2cv_free, r2cv_tied=r2cv_tied, dr2_cv=dr2,
        pref_consistency=consistency,
        psi_2nd_half_mean=np.nanmean(psi_h, axis=(0, 1)),
        psi_2nd_half_sd=np.nanstd(psi_h, axis=(0, 1)),
        rel_r=rel_r, rel_null=null, rel_thr=rel_thr,
        reliable=reliable, good=good, **idx)


def label_cells(res, cfg):

    xd, Td = res["disp_x"], res["disp_T"]
    lo, hi = res["span"]
    b = res["b"][None, :]
    with np.errstate(invalid="ignore"):
        amp_on = np.nanmax(Td[xd < lo] - b, axis=0)
        amp_rw = np.nanmax(Td[xd > hi] - b, axis=0)
    amp_on = np.maximum(np.nan_to_num(amp_on, nan=0.0), 0.0)
    amp_rw = np.maximum(np.nan_to_num(amp_rw, nan=0.0), 0.0)

    A_eff = res["A"].copy()
    A_eff[0] = np.maximum(A_eff[0], amp_on)
    A_eff[3] = np.maximum(A_eff[3], amp_rw)
    idx = amplitude_indices(A_eff)

    at_bound = np.abs(res["delta"]) >= (cfg.gauss_delta_max_cm
                                        - 0.5 * cfg.gauss_delta_step_cm)
    good = res["good"]
    visual = good & (np.nan_to_num(res["dr2_cv"], nan=0.0)
                     <= cfg.gauss_visual_max_dr2)

    label = np.full(len(good), "none", dtype=object)
    for k in range(4):
        label[good & (idx["pref"] == k + 1)] = "L{}-pref".format(k + 1)
    label[visual] = "visual"


    res.update(idx)
    res.update(A_eff=A_eff, amp_reward=amp_rw, amp_onset=amp_on,
               delta_at_bound=at_bound, label=label)
    
    return res


