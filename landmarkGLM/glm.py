# -*- coding: utf-8 -*-
"""
landmarkGLM/glm.py

Poisson GLM core (log link, ridge-penalized): per-fold IRLS solves, candidate
kernel selection by cross-validation, and held-out deviance comparisons.

DMM, Aug 2026
"""

import warnings

import numpy as np
from tqdm import tqdm

from candidate_kernels import fit_span, fit_span_mask, va_slope_grid


ETA_CLIP = 30.0
GRAM_BYTES = 512 * 1024 ** 2
STEP_MAX = 4.0

IRLS_MAX_ITER = 40
IRLS_TOL = 1e-7


def poisson_dev(Y, MU):
    """ Poisson deviance per cell, 2 * sum(y*log(y/mu) - (y - mu)).

    y*log(y) is taken as 0 at y = 0, its limit. Valid for continuous
    non-negative y as a quasi-deviance, which is what deconvolved spikes are.

    Parameters
    ----------
    Y : np.ndarray
        Observed, (n, n_cells), non-negative.
    MU : np.ndarray
        Predicted rate, (n, n_cells), strictly positive.

    Returns
    -------
    np.ndarray
        Deviance per cell, (n_cells,).
    """

    MU = np.maximum(MU, 1e-12)
    t = np.where(Y > 0, Y * np.log(np.maximum(Y, 1e-12) / MU), 0.0)

    return 2.0 * np.sum(t - (Y - MU), axis=0)

DEV_NULL_FLOOR = 1e-9


def dev_explained(dmod, dnull):
    """ 1 - D_model / D_null, with silent cells returned as NaN rather than 1.0.
    summary.
    """

    ok = np.asarray(dnull, float) > DEV_NULL_FLOOR

    return np.where(ok, 1.0 - np.asarray(dmod, float) / np.where(ok, dnull, 1.0),
                    np.nan)


def dev_r2(Y, MU, MU0):
    """ Fraction of null deviance explained -- the log-link analogue of R2.

    Parameters
    ----------
    Y : np.ndarray
        Observed, (n, n_cells).
    MU : np.ndarray
        Model rate, (n, n_cells).
    MU0 : np.ndarray
        Null (intercept-only, same offset) rate, (n, n_cells).

    Returns
    -------
    np.ndarray
        1 - D_model / D_null, per cell. Negative when the model does worse
        out of sample than the null, same as the old held-out R2. NaN for a
        cell with no null deviance to explain.
    """

    return dev_explained(poisson_dev(Y, MU), poisson_dev(Y, MU0))


def _as_offset(off, n):
    """ Normalize an offset to (n, 1) or (n, n_cells), or None.
    """

    if off is None:
        return None
    O = np.asarray(off, np.float64)
    if O.ndim == 1:
        O = O[:, None]      # shared across cells, broadcasts
    if O.shape[0] != n:
        raise ValueError("offset has {} rows, design has {}".format(O.shape[0], n))

    return O


def _off_cols(O, cols):
    """ The offset columns for an active subset of cells.
    """

    if O is None:
        return 0.0

    return O if O.shape[1] == 1 else O[:, cols]


def _gram_table(Z, gram_bytes):
    """ Row-wise outer products of Z, upper triangle only, or None if too big.
    """

    n, q = Z.shape
    iu = np.triu_indices(q)
    if n * len(iu[0]) * 4 > gram_bytes:
        return None
    Z32 = np.ascontiguousarray(Z, dtype=np.float32)

    return np.ascontiguousarray(Z32[:, iu[0]] * Z32[:, iu[1]]), iu


def _scatter_gram(T, iu, q):
    """ Rebuild full symmetric (m, q, q) Grams from packed upper triangles.
    """

    m = T.shape[0]
    H = np.empty((m, q, q), dtype=np.float64)
    H[:, iu[0], iu[1]] = T
    H[:, iu[1], iu[0]] = T

    return H


_BACKEND = None
_DEV_CACHE = []
_DEV_CACHE_MAX = 8


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def set_backend(name="auto"):

    global _BACKEND
    name = (name or "auto").lower()
    if name == "numpy":
        _BACKEND = "numpy"
    elif name == "torch":
        t = _torch()
        if t is None:
            raise RuntimeError("glm_backend='torch' but torch is not installed.")
        if not t.cuda.is_available():
            raise RuntimeError("glm_backend='torch' but CUDA is unavailable.")
        _BACKEND = "torch"
    elif name == "auto":
        t = _torch()
        _BACKEND = "torch" if (t is not None and t.cuda.is_available()) else "numpy"
    else:
        raise ValueError("unknown glm backend {!r}".format(name))
    _DEV_CACHE.clear()

    return _BACKEND


def active_backend():
    """ The backend in use, resolving 'auto' on first call. """
    return _BACKEND if _BACKEND is not None else set_backend("auto")


def _dev_get(key_arr, tag, build):

    k = (id(key_arr), tag, getattr(key_arr, "shape", None))
    for i, ent in enumerate(_DEV_CACHE):
        if ent[0] == k:
            _DEV_CACHE.append(_DEV_CACHE.pop(i))
            return ent[2]
    val = build(key_arr)
    _DEV_CACHE.append((k, key_arr, val))
    while len(_DEV_CACHE) > _DEV_CACHE_MAX:
        _DEV_CACHE.pop(0)

    return val


def _torch_design(Z32, q):
    """ Augmented design's outer-product table on device, upper triangle only.
    """

    torch = _torch()
    iu = torch.triu_indices(q, q, device=Z32.device)

    return (Z32[:, iu[0]] * Z32[:, iu[1]]).contiguous(), iu


def _irls_core_torch(Z32, Zd, P, iu, Yd, Od, pen_a, TH0=None,
                     max_iter=IRLS_MAX_ITER, tol=IRLS_TOL, step_max=STEP_MAX):
    """
    Parameters
    ----------
    Z32, Zd : torch.Tensor
        Augmented design (n, q), fp32 and fp64.
    P, iu : torch.Tensor
        Outer-product table and its triu index pair, from _torch_design.
    Yd : torch.Tensor
        Target (n, n_cells), fp64.
    Od : torch.Tensor or None
        Offset (n, n_cells) or (n, 1), fp64.
    pen_a : torch.Tensor
        Per-column penalty (q,), fp64, with the intercept entry zero.
    TH0 : torch.Tensor, optional
        Warm start (q, n_cells), fp64.

    Returns
    -------
    torch.Tensor
        Coefficients (q, n_cells), fp64; the last row is the intercept.
    """

    torch = _torch()
    dev = Z32.device
    n, q = Z32.shape
    nc = Yd.shape[1]

    if TH0 is not None:
        TH = TH0.clone()
    else:
        TH = torch.zeros((q, nc), device=dev, dtype=torch.float64)
        E0 = (torch.ones((n, 1), device=dev, dtype=torch.float64) if Od is None
              else torch.exp(torch.clamp(Od, -ETA_CLIP, ETA_CLIP)))
        TH[q - 1] = torch.log(Yd.sum(dim=0).clamp_min(1e-12)
                              / E0.sum(dim=0).clamp_min(1e-12))

    ar = torch.arange(q, device=dev)
    for _ in range(max_iter):
        eta = Zd @ TH
        if Od is not None:
            eta = eta + Od
        eta = torch.clamp(eta, -ETA_CLIP, ETA_CLIP)
        MU = torch.exp(eta)

        g = Zd.T @ (Yd - MU) - pen_a[:, None] * TH
        T = (MU.float().T @ P).double()
        H = torch.zeros((nc, q, q), device=dev, dtype=torch.float64)
        H[:, iu[0], iu[1]] = T
        H[:, iu[1], iu[0]] = T
        jit = 1e-9 * (torch.diagonal(H, dim1=1, dim2=2).sum(dim=1) / q + 1.0)
        H[:, ar, ar] += pen_a[None, :] + jit[:, None]

        try:
            L = torch.linalg.cholesky(H)
            step = torch.cholesky_solve(g.T.unsqueeze(-1), L).squeeze(-1).T
        except Exception:
            step = torch.linalg.solve(H, g.T.unsqueeze(-1)).squeeze(-1).T

        dmax = (Zd @ step).abs().amax(dim=0)
        step = step * torch.clamp(step_max / dmax.clamp_min(1e-12), max=1.0)[None, :]

        TH = TH + step
        if float(step.abs().amax()) < tol:
            break

    return TH


def _pois_dev_torch(Y, MU):
    """ Poisson deviance per cell, on device. Mirrors poisson_dev exactly. """

    torch = _torch()
    MU = MU.clamp_min(1e-12)
    z = torch.zeros((), dtype=Y.dtype, device=Y.device)
    t = torch.where(Y > 0, Y * torch.log(Y.clamp_min(1e-12) / MU), z)

    return 2.0 * (t - (Y - MU)).sum(dim=0)


def _pen_vec_torch(pen, p, q, dev):
    """ Per-column penalty as a (q,) fp64 tensor, intercept unpenalized. """

    torch = _torch()
    v = torch.zeros(q, device=dev, dtype=torch.float64)
    if p:
        pv = np.array(np.broadcast_to(np.asarray(pen, float), (p,)), copy=True)
        v[:p] = torch.as_tensor(pv, device=dev, dtype=torch.float64)

    return v


def _irls_torch(X, Y, off=None, pen=0.0, W0=None, b0=None,
                max_iter=IRLS_MAX_ITER, tol=IRLS_TOL, step_max=STEP_MAX):
    """ Ridge-penalized Poisson regression on the GPU, numpy in and numpy out.
    """

    torch = _torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")

    n, p = np.asarray(X).shape
    q = p + 1
    nc = int(np.asarray(Y).shape[1])

    def _mk_Z(a):
        t = torch.as_tensor(np.ascontiguousarray(a, dtype=np.float32), device=dev)
        return torch.cat([t, torch.ones((t.shape[0], 1), device=dev,
                                        dtype=torch.float32)], dim=1)

    Z32 = _dev_get(X, "Z", _mk_Z)
    Zd = _dev_get(X, "Zd", lambda a: Z32.double())
    P, iu = _dev_get(X, "P", lambda a: _torch_design(Z32, q))
    Yd = _dev_get(Y, "Y", lambda a: torch.as_tensor(
        np.ascontiguousarray(a, dtype=np.float64), device=dev))
    Od = None
    if off is not None:
        O = _as_offset(off, n)
        Od = _dev_get(off, "O", lambda a: torch.as_tensor(
            np.ascontiguousarray(O, dtype=np.float64), device=dev))

    TH0 = None
    if W0 is not None or b0 is not None:
        TH0 = torch.zeros((q, nc), device=dev, dtype=torch.float64)
        if W0 is not None:
            TH0[:p] = torch.as_tensor(np.asarray(W0, float), device=dev,
                                      dtype=torch.float64)
        if b0 is not None:
            TH0[p] = torch.as_tensor(np.asarray(b0, float), device=dev,
                                     dtype=torch.float64)

    TH = _irls_core_torch(Z32, Zd, P, iu, Yd, Od,
                          _pen_vec_torch(pen, p, q, dev), TH0=TH0,
                          max_iter=max_iter, tol=tol, step_max=step_max)
    out = TH.cpu().numpy()

    return out[:p], out[p]


def _select_kernel_torch(Xlist, Y, off, fit, val, lams, order, denom):
    """ _select_kernel's candidate loop, without leaving the device.
    """

    torch = _torch()
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda")

    nc = Y.shape[1]
    fit_t = torch.as_tensor(np.asarray(fit), device=dev)
    val_t = torch.as_tensor(np.asarray(val), device=dev)
    Yt = torch.as_tensor(np.ascontiguousarray(Y, dtype=np.float64), device=dev)
    Ot = torch.as_tensor(np.ascontiguousarray(_as_offset(off, Y.shape[0]),
                                              dtype=np.float64), device=dev)
    Yf, Yv = Yt[fit_t], Yt[val_t]
    Of, Ov = Ot[fit_t], Ot[val_t]
    den = torch.as_tensor(np.ascontiguousarray(denom), device=dev,
                          dtype=torch.float64)
    del Yt, Ot

    S = np.full((len(Xlist), len(lams), nc), -np.inf)
    for i, X in enumerate(Xlist):
        Xt = torch.as_tensor(np.ascontiguousarray(X, dtype=np.float32), device=dev)
        Xf, Xv = Xt[fit_t], Xt[val_t]
        n, p = Xf.shape
        q = p + 1
        Z32 = torch.cat([Xf, torch.ones((n, 1), device=dev,
                                        dtype=torch.float32)], dim=1)
        Zd = Z32.double()
        P, iu = _torch_design(Z32, q)
        Xvd = Xv.double()

        TH = None
        for j in order:
            TH = _irls_core_torch(Z32, Zd, P, iu, Yf, Of,
                                  _pen_vec_torch(lams[j], p, q, dev), TH0=TH)
            eta = torch.clamp(Xvd @ TH[:p] + TH[p] + Ov, -ETA_CLIP, ETA_CLIP)
            d = _pois_dev_torch(Yv, torch.exp(eta))
            S[i, j] = torch.where(den > DEV_NULL_FLOOR, 1.0 - d / den,
                                  torch.full_like(den, float("nan"))).cpu().numpy()
        del Xt, Xf, Xv, Xvd, Z32, Zd, P

    return S


def poisson_irls(X, Y, off=None, pen=0.0, W0=None, b0=None,
                 max_iter=IRLS_MAX_ITER, tol=IRLS_TOL, step_max=STEP_MAX,
                 gram_bytes=GRAM_BYTES, P=None, Z=None):
    """ Ridge-penalized Poisson regression, all cells at once, by Newton/IRLS.

    Maximizes  sum_i [ y_i * eta_i - exp(eta_i) ] - 0.5 * sum_j pen_j * w_j^2
    with       eta = off + X @ w + b,
    the intercept b unpenalized (it carries the cell's mean rate, which no
    amount of shrinkage should touch). Concave in (w, b), so the Newton
    iteration has a single fixed point and the starting value only affects how
    fast it is reached -- which is what makes the lambda-path warm starts, and
    the two backends, safe to treat as interchangeable.
    """

    if active_backend() == "torch" and np.asarray(X).shape[1] > 0:
        return _irls_torch(X, Y, off=off, pen=pen, W0=W0, b0=b0,
                           max_iter=max_iter, tol=tol, step_max=step_max)

    return _irls_numpy(X, Y, off=off, pen=pen, W0=W0, b0=b0, max_iter=max_iter,
                       tol=tol, step_max=step_max, gram_bytes=gram_bytes,
                       P=P, Z=Z)


def _irls_numpy(X, Y, off=None, pen=0.0, W0=None, b0=None,
                max_iter=IRLS_MAX_ITER, tol=IRLS_TOL, step_max=STEP_MAX,
                gram_bytes=GRAM_BYTES, P=None, Z=None):

    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    n, p = X.shape
    nc = Y.shape[1]
    q = p + 1

    if Z is None:
        Z = np.concatenate([X, np.ones((n, 1))], axis=1)
    if P is None:
        P = _gram_table(Z, gram_bytes)

    O = _as_offset(off, n)
    pen_a = np.zeros(q)
    pen_a[:p] = np.asarray(pen, float)

    TH = np.zeros((q, nc))
    if W0 is not None:
        TH[:p] = W0
    if b0 is not None:
        TH[p] = b0
    else:
        E0 = (np.ones((n, 1)) if O is None
              else np.exp(np.clip(O, -ETA_CLIP, ETA_CLIP)))
        TH[p] = np.log(np.maximum(Y.sum(axis=0), 1e-12)
                       / np.maximum(E0.sum(axis=0), 1e-12))

    ar = np.arange(q)
    act = np.arange(nc)
    for _ in range(max_iter):
        TA = TH[:, act]
        eta = Z @ TA + _off_cols(O, act)
        np.clip(eta, -ETA_CLIP, ETA_CLIP, out=eta)
        MU = np.exp(eta)

        g = Z.T @ (Y[:, act] - MU) - pen_a[:, None] * TA
        if P is not None:
            Ptab, iu = P
            H = _scatter_gram(np.asarray(MU, np.float32).T @ Ptab, iu, q)
        else:
            H = np.empty((len(act), q, q))
            for k in range(len(act)):
                Zs = Z * np.sqrt(MU[:, k])[:, None]
                H[k] = Zs.T @ Zs

        H[:, ar, ar] += pen_a + 1e-9 * (np.trace(H, axis1=1, axis2=2) / q + 1.0)[:, None]

        step = np.linalg.solve(H, g.T[:, :, None])[:, :, 0].T      # (q, m)
        dmax = np.max(np.abs(Z @ step), axis=0)
        step *= np.minimum(1.0, step_max / np.maximum(dmax, 1e-12))

        TH[:, act] = TA + step
        act = act[np.max(np.abs(step), axis=0) >= tol]
        if act.size == 0:
            break

    return TH[:p], TH[p]


def _null_mu(Yfit, off_fit, off_eval, n_eval=None):
    """ Intercept-only Poisson fit on `fit` rows, evaluated where told.
    """

    _, b = poisson_irls(np.zeros((Yfit.shape[0], 0)), Yfit, off_fit)
    n_eval = Yfit.shape[0] if n_eval is None else n_eval
    O = _as_offset(off_eval, n_eval)
    eta = np.broadcast_to(b, (n_eval, len(b))).copy()
    if O is not None:
        eta = eta + O

    return np.exp(np.clip(eta, -ETA_CLIP, ETA_CLIP))


def _glm_eta(Xtr, Ytr, off_tr, Xte, off_te, lam):
    """ Fit a penalized Poisson GLM on train rows, return the test linear predictor. """

    W, b = poisson_irls(Xtr, Ytr, off_tr, np.full(Xtr.shape[1], float(lam)))
    eta = Xte @ W + b
    O = _as_offset(off_te, Xte.shape[0])
    if O is not None:
        eta = eta + O

    return np.clip(eta, -ETA_CLIP, ETA_CLIP)


def _glm_pred(Xtr, Ytr, off_tr, Xte, off_te, lam):
    """ As _glm_eta, on the rate scale. The log-link twin of the old _ridge_pred. """

    return np.exp(_glm_eta(Xtr, Ytr, off_tr, Xte, off_te, lam))


def _glm_w(Xtr, Ytr, off_tr, lam):
    """ Penalized Poisson weights with an unpenalized intercept. LOG GAINS. """

    W, _ = poisson_irls(Xtr, Ytr, off_tr, np.full(Xtr.shape[1], float(lam)))

    return W


class PoissonFold:

    def __init__(self, Xtr, Ytr, off=None, gram_bytes=GRAM_BYTES):
        """

        Parameters
        ----------
        Xtr : np.ndarray
            Training design, (n_train, p).
        Ytr : np.ndarray
            Training targets, (n_train, n_cells), non-negative.
        off : np.ndarray, optional
            Training-row offset, (n_train,) or (n_train, n_cells).
        """

        self.X = np.asarray(Xtr, np.float64)
        self.Y = np.asarray(Ytr, np.float64)
        self.off = off
        self.gram_bytes = gram_bytes

    def eta(self, Xte, scale, lambdas, off_te=None, cells=None):
        """ Get held-out linear predictors at each lambda.

        Parameters
        ----------
        Xte : np.ndarray
            Test design, (n_test, p).
        scale : np.ndarray
            Per-column penalty scale. Ridge multiplied columns by scale; the
            equivalent here is an effective penalty of lambda / scale^2 per
            column, which is what that scaling amounted to.
        lambdas : iterable of float
            Ridge penalties to yield at, in the order given.
        off_te : np.ndarray, optional
            Test-row offset.
        cells : np.ndarray, optional
            Column subset of Ytr to fit. Defaults to all cells.

        Yields
        ------
        eta : np.ndarray
            Held-out linear predictor at each successive lambda, (n_test, n_cells).
        """

        Y = self.Y if cells is None else self.Y[:, cells]
        O = _as_offset(self.off, self.X.shape[0])
        if O is not None and cells is not None and O.shape[1] > 1:
            O = O[:, cells]
        Ote = _as_offset(off_te, Xte.shape[0])
        if Ote is not None and cells is not None and Ote.shape[1] > 1:
            Ote = Ote[:, cells]

        sc = np.asarray(scale, float)
        base = 1.0 / np.maximum(sc, 1e-12) ** 2

        n, p = self.X.shape
        Z = np.concatenate([self.X, np.ones((n, 1))], axis=1)
        P = _gram_table(Z, self.gram_bytes)

        W = b = None
        for lam in lambdas:
            W, b = poisson_irls(self.X, Y, O, float(lam) * base, W0=W, b0=b,
                                P=P, Z=Z)
            eta = Xte @ W + b
            if Ote is not None:
                eta = eta + Ote
            yield np.clip(eta, -ETA_CLIP, ETA_CLIP)

    def predict(self, Xte, scale, lambdas, off_te=None, cells=None):
        """ As eta(), on the rate scale. """

        for e in self.eta(Xte, scale, lambdas, off_te=off_te, cells=cells):
            yield np.exp(e)


def _select_kernel(Xlist, Y, off, fit, val, lambdas, return_scores=False):
    """ Choose, per cell, which candidate kernel to use -- ON TRAINING LAPS ONLY.

    Parameters
    ----------
    Xlist : list of np.ndarray
        Candidate designs, each (n_frames, p).
    Y : np.ndarray
        Non-negative target, (n_frames, n_cells). Under the old ridge code this
        argument was the pure-behavior RESIDUAL; it is now the raw target, with
        pure behavior carried in `off`.
    off : np.ndarray
        Pure-behavior linear predictor, (n_frames, n_cells), held fixed.
    fit, val : np.ndarray of bool
        Row masks for the inner fit/validation split, both inside training laps.
    lambdas : array-like
        Ridge penalties to try; one is chosen for the whole fold.
    return_scores : bool
        If True, also return the full (candidate, cell) validation score matrix.

    Returns
    -------
    best : np.ndarray
        Chosen candidate index per cell.
    lam : float
        Chosen lambda, shared across cells.
    score : np.ndarray
        Validation deviance explained of the chosen candidate, per cell.
    S : np.ndarray, optional
        Full (n_candidate, n_cell) validation score at the chosen lambda. Only
        returned when return_scores is True.
    """

    Yf, Yv = Y[fit], Y[val]
    Of, Ov = off[fit], off[val]
    nc = Y.shape[1]

    MU0v = _null_mu(Yf, Of, Ov, n_eval=int(val.sum()))
    denom = poisson_dev(Yv, MU0v)

    lams = np.asarray(lambdas, float)
    order = np.argsort(lams)[::-1]

    S = np.full((len(Xlist), len(lams), nc), -np.inf)
    if active_backend() == "torch":
        S = _select_kernel_torch(Xlist, Y, off, fit, val, lams, order, denom)
    else:
        for i, X in enumerate(Xlist):
            Xf, Xv = X[fit], X[val]
            n, p = Xf.shape
            Z = np.concatenate([np.asarray(Xf, np.float64), np.ones((n, 1))],
                               axis=1)
            P = _gram_table(Z, GRAM_BYTES)
            W = b = None
            for j in order:
                W, b = poisson_irls(Xf, Yf, Of, np.full(p, lams[j]), W0=W,
                                    b0=b, P=P, Z=Z)
                MU = np.exp(np.clip(Xv @ W + b + Ov, -ETA_CLIP, ETA_CLIP))
                S[i, j] = dev_explained(poisson_dev(Yv, MU), denom)

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_lam = np.nanmedian(np.nanmax(S, axis=0), axis=1)
    j = int(np.nanargmax(per_lam))

    Sj = S[:, j, :]
    best = np.argmax(np.nan_to_num(Sj, nan=-np.inf, neginf=-np.inf), axis=0)
    if return_scores:

        return best, lams[j], Sj[best, np.arange(nc)], Sj

    return best, lams[j], Sj[best, np.arange(nc)]


def _fit_chosen(Xlist, idx, Y, off, tr, te, lam):
    """ Refit each cell's chosen kernel on ALL training laps; predict test laps.

    Returns the test-row RATE, (n_test, n_cells).
    """

    MU = np.zeros((int(te.sum()), Y.shape[1]))
    for i in np.unique(idx):
        cells = np.where(idx == i)[0]
        X = Xlist[i]
        MU[:, cells] = _glm_pred(X[tr], Y[tr][:, cells], off[tr][:, cells],
                                 X[te], off[te][:, cells], lam)

    return MU


def _fit_chosen_pair(Xa, ia, Xb, ib, Y, off, tr, te, lam):
    """ Both kernels together: 14 columns, refit on training laps. """

    MU = np.zeros((int(te.sum()), Y.shape[1]))
    key = ia.astype(np.int64) * (max(len(Xb), 1) + 1) + ib.astype(np.int64)
    for k in np.unique(key):
        cells = np.where(key == k)[0]
        i, j = int(ia[cells[0]]), int(ib[cells[0]])
        X = np.hstack([Xa[i], Xb[j]])
        MU[:, cells] = _glm_pred(X[tr], Y[tr][:, cells], off[tr][:, cells],
                                 X[te], off[te][:, cells], lam)

    return MU


def _pick_lambda_PB(XPBr, Yr, tr, lap_r, cfg, scale=None):
    """ Lambda for the pure behavior block, on an inner split of the training laps.

    `scale` is build_purebehavior's per-column penalty scale (1.0 except the
    position bases). Must match the vector used in the final fit, or lambda
    is chosen for a different model than the one that runs.

    Parameters
    ----------
    XPBr : np.ndarray
        Pure-behavior design, running frames only.
    Yr : np.ndarray
        Non-negative target, running frames only.
    tr : np.ndarray of bool
        Training-lap mask; the inner split is drawn from within it.
    lap_r : np.ndarray
        Lap id per row.
    cfg : Config
        Supplies lambda_grid and n_inner_folds.
    scale : np.ndarray, optional
        Per-column penalty scale. Defaults to all ones.

    Returns
    -------
    blam : float
        The chosen lambda.
    """

    tl = np.unique(lap_r[tr])
    nv = max(1, len(tl) // cfg.n_inner_folds)
    val = tr & np.isin(lap_r, tl[-nv:])
    fit = tr & ~val
    if fit.sum() < 100 or val.sum() < 20:
        return cfg.lambda_grid[len(cfg.lambda_grid) // 2]

    pf = PoissonFold(XPBr[fit], Yr[fit])
    one = np.ones(XPBr.shape[1]) if scale is None else np.asarray(scale, float)
    MU0 = _null_mu(Yr[fit], None, None, n_eval=int(val.sum()))

    grid = np.asarray(cfg.lambda_grid, float)
    order = np.argsort(grid)[::-1]
    best, blam = -np.inf, grid[0]
    for lam, MU in zip(grid[order], pf.predict(XPBr[val], one, grid[order])):
        s = np.nanmedian(dev_r2(Yr[val], MU, MU0))
        if np.isfinite(s) and s > best:
            best, blam = s, lam

    return float(blam)


def compare_kernels(Y, beh, cfg, XPB, cand, scalePB=None, reliable=None):

    _span = fit_span_mask(beh, cfg)
    rm = beh["run_mask"] & _span
    _lo, _hi = fit_span(cfg)
    Yr = np.asarray(Y, np.float64)[rm]
    if np.any(Yr < 0):
        raise ValueError("Poisson GLM needs a non-negative target; got negatives. "
                         "Use cfg.zone_target='spks' and main.prep's rate scaling, "
                         "not a z-scored trace.")
    XPBr = XPB[rm]
    lap_r = beh["lap_id"][rm]
    laps = np.unique(lap_r[lap_r >= 0])
    folds = np.array_split(laps, cfg.n_outer_folds)
    nc = Yr.shape[1]

    Gx = [x[rm] for x in cand["gauss"]["X"]]
    Cx = [x[rm] for x in cand["comb"]["X"]]
    has_ad = "adapt" in cand and len(cand["adapt"]["X"]) > 0
    Ax = [x[rm] for x in cand["adapt"]["X"]] if has_ad else []
    Fx = cand["free"]["X"][rm]
    onePB = np.ones(XPBr.shape[1]) if scalePB is None else np.asarray(scalePB, float)

    hyp = ["gauss", "comb"] + (["adapt"] if has_ad else [])
    names = hyp + ["both", "free"]
    dev = {k: np.zeros(nc) for k in names}
    dev_null = np.zeros(nc)
    lam_f_hist, nsig_hist = [], []
    mu_sel = np.full((len(folds), nc), np.nan)
    sg_sel = np.full((len(folds), nc), np.nan)
    dl_sel = np.full((len(folds), nc), np.nan)
    ad_dl_sel = np.full((len(folds), nc), np.nan)
    ad_sg_sel = np.full((len(folds), nc), np.nan)
    ad_rt_sel = np.full((len(folds), nc), np.nan)   # the chosen envelope slope
    used = 0

    for fi, te_lap in tqdm(enumerate(folds), total=len(folds), desc="  Kernel comparison"):
        te = np.isin(lap_r, te_lap)
        tr = ~te
        if tr.sum() < 500 or te.sum() < 50:
            continue

        lamPB = _pick_lambda_PB(XPBr, Yr, tr, lap_r, cfg, scale=onePB)
        pf = PoissonFold(XPBr[tr], Yr[tr])

        OFF = next(pf.eta(XPBr, onePB, [lamPB]))

        tl = np.unique(lap_r[tr])
        nv = max(1, len(tl) // cfg.n_inner_folds)
        val = tr & np.isin(lap_r, tl[-nv:])
        fit = tr & ~val
        if fit.sum() < 200 or val.sum() < 50:
            continue

        ig, lam_g, sg_ = _select_kernel(Gx, Yr, OFF, fit, val, cfg.kernel_lambdas)
        ic, lam_c, sc_ = _select_kernel(Cx, Yr, OFF, fit, val, cfg.kernel_lambdas)
        if has_ad:
            ia, lam_a, sa_ = _select_kernel(Ax, Yr, OFF, fit, val, cfg.kernel_lambdas)
        for c in range(nc):
            mu_sel[fi, c], sg_sel[fi, c] = cand["gauss"]["par"][ig[c]]
            dl_sel[fi, c] = cand["comb"]["par"][ic[c]][0]
            if has_ad:
                (ad_dl_sel[fi, c], ad_sg_sel[fi, c],
                 ad_rt_sel[fi, c]) = cand["adapt"]["par"][ia[c]]

        Yte = Yr[te]

        dev_null += poisson_dev(Yte, _null_mu(Yr[tr], OFF[tr], OFF[te],
                                              n_eval=int(te.sum())))
        dev["gauss"] += poisson_dev(Yte, _fit_chosen(Gx, ig, Yr, OFF, tr, te, lam_g))
        dev["comb"] += poisson_dev(Yte, _fit_chosen(Cx, ic, Yr, OFF, tr, te, lam_c))
        if has_ad:
            dev["adapt"] += poisson_dev(Yte, _fit_chosen(Ax, ia, Yr, OFF, tr, te, lam_a))
        dev["both"] += poisson_dev(Yte, _fit_chosen_pair(Gx, ig, Cx, ic, Yr, OFF,
                                                         tr, te, max(lam_g, lam_c)))

        _sc_all = [sg_, sc_] + ([sa_] if has_ad else [])
        sig = np.maximum.reduce(_sc_all) > getattr(cfg, "free_lambda_sig_r2",
                                                   cfg.r2_threshold)
        if reliable is not None:
            sig = sig & np.asarray(reliable, dtype=bool)
        if sig.sum() < 20:
            sig = np.ones(nc, dtype=bool)
        MU0v = _null_mu(Yr[fit], OFF[fit], OFF[val], n_eval=int(val.sum()))
        bf, lam_f = -np.inf, cfg.free_lambdas[0]
        for lam in np.sort(np.asarray(cfg.free_lambdas, float))[::-1]:
            MU = _glm_pred(Fx[fit], Yr[fit], OFF[fit], Fx[val], OFF[val], lam)
            s = np.nanmedian(dev_r2(Yr[val], MU, MU0v)[sig])
            if np.isfinite(s) and s > bf:
                bf, lam_f = s, lam
        lam_f_hist.append(lam_f)
        nsig_hist.append(int(sig.sum()))
        dev["free"] += poisson_dev(Yte, _glm_pred(Fx[tr], Yr[tr], OFF[tr],
                                                  Fx[te], OFF[te], lam_f))

        used += 1

    out = {k: dev_explained(dev[k], dev_null) for k in names}
    out["mu"] = np.nanmedian(mu_sel, axis=0)
    out["sigma"] = np.nanmedian(sg_sel, axis=0)
    out["delta"] = np.nanmedian(dl_sel, axis=0)
    out["mu_stability"] = np.nanstd(mu_sel, axis=0)
    out["mu_per_fold"] = mu_sel
    out["n_folds_used"] = used
    out["hypotheses"] = hyp
    if has_ad:
        out["adapt_delta"] = np.nanmedian(ad_dl_sel, axis=0)
        out["adapt_sigma"] = np.nanmedian(ad_sg_sel, axis=0)

        out["adapt_rate"] = np.nanmedian(ad_rt_sel, axis=0)
        out["adapt_rate_per_fold"] = ad_rt_sel
        out["adapt_ramp_s"] = cand["adapt"].get("ramp_s", np.nan)

    g = np.maximum(out["gauss"], 0.0)
    c = np.maximum(out["comb"], 0.0)
    v = np.maximum(c, np.maximum(out["adapt"], 0.0)) if has_ad else c
    tot = g + v
    out["visual_r2"] = v
    out["place_index"] = np.where(tot > 1e-12, (g - v) / np.maximum(tot, 1e-12), np.nan)

    if has_ad:

        a = np.maximum(out["adapt"], 0.0)
        den = a + c
        out["adapt_index"] = np.where(den > 1e-12, (a - c) / np.maximum(den, 1e-12),
                                      np.nan)
        out["adapt_index"][out["place_index"] > 0] = np.nan

    _b = np.maximum(out["both"], 0.0)
    _v = np.maximum(out["comb"], 0.0)
    _den = _b + _v
    out["spatial_index"] = np.where(_den > 1e-12, (_b - _v) / np.maximum(_den, 1e-12),
                                    np.nan)

    if has_ad:
        _a = np.maximum(out["adapt"], 0.0)
        _base = np.maximum(_v, _b)
        _den2 = _a + _base
        out["adaptation_index"] = np.where(
            _den2 > 1e-12, (_a - _base) / np.maximum(_den2, 1e-12), np.nan)
        out["adaptation_base"] = np.where(_b > _v, "both", "visual")

    _H = np.vstack([out[h] for h in hyp])
    out["best_r2"] = np.nanmax(_H, axis=0)
    out["best_model"] = np.array(hyp)[np.nanargmax(np.nan_to_num(_H, nan=-np.inf),
                                                   axis=0)]

    out["fit_above_null"] = out["best_r2"] > cfg.r2_threshold
    if reliable is None:
        out["reliable"] = np.ones(nc, dtype=bool)
    else:
        out["reliable"] = np.asarray(reliable, dtype=bool)
    out["well_fit"] = out["fit_above_null"] & out["reliable"]
    if has_ad:
        out["adapt_gain"] = out["adapt"] - out["comb"]

    out["lambda_free_per_fold"] = np.asarray(lam_f_hist, float)
    out["n_signal_per_fold"] = np.asarray(nsig_hist, int)

    return out


