# -*- coding: utf-8 -*-
"""
landmarkGLM/build_landmark_gains.py

Builds the landmark-gains candidate family (the four landmark bumps as four
separate regressors, so each gets its own amplitude, where `comb` ties all four
to one) and summarizes those per-landmark gains into shape statistics.
Used to be called comb4 but i renamed it to avoid confusion with the comb basis itself.

DMM, Aug 2026
"""

import numpy as np

from basis_funcs import lag_expand

# Four landmark bumps as four independent regressors.
CON = {
    "linear": np.array([-3.0, -1.0, 1.0, 3.0]) / np.sqrt(20.0),
    "quadratic": np.array([1.0, -1.0, -1.0, 1.0]) / 2.0,
    "cubic": np.array([-1.0, 3.0, -3.0, 1.0]) / np.sqrt(20.0)
}
CON_ORDER = ["linear", "quadratic", "cubic"]


def build_landmark_gains(beh, cfg, lagB):
    """ The four landmark bumps as four regressors, over comb's own grid.
    """

    pos = beh["pos"]
    ok = (pos >= 0) & (pos <= cfg.corridor_cm)
    iti = beh.get("iti")
    centers = np.array([(a + b) / 2.0 for a, b in cfg.zones])

    def _prep(M):
        """ Zero non-running and ITI frames, then lag-expand. """

        M = np.asarray(M, float).copy()
        M[~ok] = 0.0
        if iti is not None and iti.any():
            M[iti] = 0.0
        return lag_expand(M, lagB).astype(np.float32)

    X, par = [], []
    for dl in cfg.delta_grid:
        for sg in cfg.sigma_grid:
            X.append(_prep(np.column_stack(
                [np.exp(-0.5 * ((pos - (m0 + dl)) / sg) ** 2) for m0 in centers])))
            par.append((float(dl), float(sg)))

    return dict(X=X, par=par, lagB=lagB)


def shape_stats(gain_per_fold):
    """ Per-fold unit-gain contrasts, and their across-fold resultant.

    Parameters
    ----------
    gain_per_fold : (n_folds, 4, n_cells)

    Returns
    -------
    dict with, per cell:
        drive      mean gain over landmarks, median across folds. LOG GAIN:
                   exp(drive) is the rate ratio the landmarks buy over pure
                   behavior, so 0 is 'no effect' and the scale is unitless.
        sgn        ALWAYS +1. There is no suppression in this dataset, so there
                   is no response direction to align to. Kept as an array of
                   ones so every downstream `* sgn` is a documented no-op
                   rather than a silent deletion.
        profile    mean-1 amplitude profile
        dev        ||C||, in [0, 1]. 0 = flat or inconsistent, 1 = strong+stable
        adapt      -C[linear], positive = declining L1 > L2 > L3 > L4
        pref       argmax of the median gains, in the cell's own sign
        margin     largest minus second largest, on the unit-scaled gains
        d_gain     winner minus best rival, in the gains' own (log) units,
                   averaged over folds
        d_gain_se  across-fold SE of d_gain
        gain_z     d_gain / d_gain_se -- how decisively the free four-gain fit
                   names a preferred landmark. Computed here rather than in
                   plots.py so main.py's shape-search floor check and the
                   figures cannot drift apart.
        gain_runner_up  index of the rival landmark d_gain is measured against

    """

    Gf = np.asarray(gain_per_fold, float)
    gmed = np.nanmedian(Gf, axis=0)
    drive = np.nanmean(gmed, axis=0)
    sgn = np.ones_like(drive)

    S = Gf * sgn[None, None, :]
    nrm = np.sqrt(np.nansum(S ** 2, axis=1))
    U = S / np.where(nrm[:, None, :] < 1e-12, np.nan, nrm[:, None, :])

    V = np.stack([np.einsum("fjc,j->fc", U, CON[k]) for k in CON_ORDER], axis=1)
    with np.errstate(invalid="ignore"):
        C = np.nanmean(V, axis=0)
    Um = gmed * sgn[None, :]
    Um = Um / np.where(np.sqrt(np.nansum(Um ** 2, axis=0)) < 1e-12, np.nan,
                       np.sqrt(np.nansum(Um ** 2, axis=0)))
    Us = np.sort(Um, axis=0)[::-1]

    pref = np.nanargmax(np.nan_to_num(Um, nan=-np.inf), axis=0)

    nc = gmed.shape[1]
    ar = np.arange(nc)
    Gw = gmed * sgn[None, :]
    fin = np.isfinite(S)
    tot = np.nansum(S, axis=0)
    cnt = fin.sum(axis=0)
    dg_f = np.full((S.shape[0], nc), np.nan)
    for f in range(S.shape[0]):
        if not fin[f].any():
            continue
        loo = (tot - np.nan_to_num(S[f])) / np.maximum(cnt - fin[f], 1)
        loo[pref, ar] = -np.inf
        rf = np.argmax(loo, axis=0)
        dg_f[f] = S[f, pref, ar] - S[f, rf, ar]
    with np.errstate(invalid="ignore"):
        d_gain = np.nanmean(dg_f, axis=0)
        d_gain_se = np.nanstd(dg_f, axis=0) / np.sqrt(
            np.maximum(np.sum(np.isfinite(dg_f), axis=0), 1))
    G2 = Gw.copy()
    G2[pref, ar] = -np.inf

    return dict(
        drive=drive, sgn=sgn,
        profile=np.where(np.abs(drive) > 1e-9, gmed * sgn[None, :] / np.abs(drive),
                         np.nan),
        dev=np.sqrt(np.nansum(C ** 2, axis=0)),
        adapt=-C[0], quadratic=C[1], cubic=C[2],
        pref=pref,
        margin=Us[0] - Us[1],
        d_gain=d_gain, d_gain_se=d_gain_se,
        gain_z=d_gain / np.where(d_gain_se < 1e-12, np.nan, d_gain_se),
        gain_runner_up=np.argmax(G2, axis=0)
    )
