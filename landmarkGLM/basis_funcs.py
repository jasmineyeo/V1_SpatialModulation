# -*- coding: utf-8 -*-
"""
landmarkGLM/basis_funcs.py

Raised-cosine basis functions for lag expansion and nonlinear value tiling.

DMM, Aug 2026
"""

import numpy as np


def raised_cosine_lag_basis(n_basis, t_max_s, fps, c=0.05):
    """ Log-spaced raised-cosine temporal basis (Pillow style)

        phi_k(tau) = 0.5 * cos(clip((log(tau+c) - ck) / w * pi, -pi, pi)) + 0.5

    Log spacing puts fine resolution at short lags (at calcium kernel)
    and coarse resolution further out. Returns (n_lags, n_basis).
    """

    n_lags = int(round(t_max_s * fps)) + 1
    tau = np.arange(n_lags) / fps
    nl = np.log(tau + c)
    lo, hi = nl[0], nl[-1]
    centers = np.linspace(lo, hi, n_basis)
    width = (centers[1] - centers[0]) * 2 if n_basis > 1 else (hi - lo)

    B = np.zeros((n_lags, n_basis))
    for k, ck in enumerate(centers):
        arg = np.clip((nl - ck) / width * np.pi, -np.pi, np.pi)
        B[:, k] = 0.5 * np.cos(arg) + 0.5
    B /= np.maximum(B.sum(axis=0, keepdims=True), 1e-12)   # unit area per bump
    return B


def _rc_value_range(x, lo=None, hi=None, valid=None):
    """ The [lo, hi] a value basis will tile.
    """

    x = np.asarray(x, dtype=float)
    if valid is None:
        valid = np.ones(len(x), dtype=bool)
    v = x[valid]
    if lo is None:
        lo = np.nanpercentile(v, 0.5) if v.size else 0.0
    if hi is None:
        hi = np.nanpercentile(v, 99.5) if v.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def rc_value_spans(x, n_basis, name, unit, lo=None, hi=None, valid=None,
                   fmt="{:.0f}"):
    """ The value span each bump of a value basis covers, as label strings.
    """

    lo, hi = _rc_value_range(x, lo, hi, valid)
    ctr = np.linspace(lo, hi, n_basis)
    width = (ctr[1] - ctr[0]) * 2 if n_basis > 1 else (hi - lo)
    tmpl = "{} " + fmt + " to " + fmt + " {}"
    return [tmpl.format(name, max(c - width / 2, lo), min(c + width / 2, hi),
                        unit)
            for c in ctr]


def raised_cosine_value_basis(x, n_basis, lo=None, hi=None, valid=None):
    """ Raised-cosine tiling of a scalar variable's range (not its lag).
    """

    x = np.asarray(x, dtype=float)
    if valid is None:
        valid = np.ones(len(x), dtype=bool)
    lo, hi = _rc_value_range(x, lo, hi, valid)

    centers = np.linspace(lo, hi, n_basis)
    width = (centers[1] - centers[0]) * 2 if n_basis > 1 else (hi - lo)

    B = np.zeros((len(x), n_basis))
    xc = np.clip(x, lo, hi)
    for k, ck in enumerate(centers):
        arg = np.clip((xc - ck) / width * np.pi, -np.pi, np.pi)
        B[:, k] = 0.5 * np.cos(arg) + 0.5
    B[~valid, :] = 0.0
    B[~np.isfinite(x), :] = 0.0
    
    return B


def lag_expand(M, lag_basis):
    """ Convolve every column of M with every lag basis function.

    Column layout: [basis 0: feats 0..p-1][basis 1: feats 0..p-1] ...
    so feature j's full kernel is out[:, j::p]
    """

    M = np.asarray(M, dtype=float)
    n, p = M.shape
    K = lag_basis.shape[1]
    out = np.zeros((n, p * K))
    for k in range(K):
        f = lag_basis[:, k]
        for j in range(p):
            out[:, k * p + j] = np.convolve(M[:, j], f)[:n]

    return out