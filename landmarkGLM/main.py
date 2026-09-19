# -*- coding: utf-8 -*-
"""
landmarkGLM/main.py

DMM, Aug 2026
"""

import os
import numpy as np
import argparse
import scipy.stats
from glob import glob

from config import Config
from load_data import load_suite2p, load_vrlog
from pure_behavior_block import build_behavior, build_purebehavior
from candidate_kernels import build_candidates, fit_span, fit_span_mask
from glm import _pick_lambda_PB, PoissonFold, _glm_pred, set_backend
from build_landmark_gains import build_landmark_gains
from fit_landmark_gains import fit_landmark_gains
from template_matching import build_shapes, fit_shapes
from plots import _compute_display_curves, plot_review_pdf, plot_excluded_pdf
from gui_funcs import select_directory, select_file


def main(SUITE2P, VRLOG, OUTDIR):

    os.makedirs(OUTDIR, exist_ok=True)

    REWARD_CM   = 134.4

    cfg = Config()

    _bk = set_backend(getattr(cfg, "glm_backend", "auto"))
    print(" -> GLM backend: {}{}".format(
        _bk, "" if _bk == "torch" else "  (no CUDA torch found; set cfg.glm_backend "
        "or run in an env that has it)"))

    print(" -> Loading suite2p and VR log...")
    dff, spks, cell_ids = load_suite2p(SUITE2P, cfg)
    vr_df = load_vrlog(VRLOG)

    print(" -> Building behavior and candidate kernels...")
    beh = build_behavior(vr_df, dff.shape[1], cfg)
    XPB, lagB, penPB, pbinfo = build_purebehavior(beh, cfg)
    cand = build_candidates(beh, cfg, lagB)


    def prep(target):

        Y = (dff if target == "dff" else spks).T
        neg = float(np.mean(Y < 0))
        if neg > 0.01:
            raise ValueError(
                "target '{}' is {:.0f}% negative -- a Poisson GLM cannot fit "
                "it. Set cfg.zone_target = 'spks'.".format(target, 100 * neg))
        Y = np.maximum(Y, 0.0)
        if getattr(cfg, "glm_rate_norm", True):
            m = Y[beh["run_mask"]].mean(axis=0)
            Y = Y / np.where(m <= 0, 1.0, m) * float(
                getattr(cfg, "glm_mean_rate", 1.0))

        return Y

    Y_rate = prep(cfg.zone_target)

    REL_BIN_CM = 3.0
    REL_MIN_FRAMES = 10
    REL_MIN_BINS = 10
    REL_N_SPLITS = 25
    REL_N_SHUFFLES = 5
    REL_MIN_SHIFT_S = 60.0
    REL_NULL_PCT = 99.0
    REL_MIN_R = 0.60


    _ZGAP_CM = float(
        np.median([cfg.zones[j + 1][0] - cfg.zones[j][1]
        for j in range(len(cfg.zones) - 1)])
    )
    REL_ONSET_CM = float(cfg.zones[0][0]) - _ZGAP_CM
    REL_REWARD_CM = float(cfg.zones[-1][1]) + _ZGAP_CM


    print(" -> Computing lap-to-lap reliability ({} cells, {} laps)...".format(
        Y_rate.shape[1], len(np.unique(beh["lap_id"][beh["lap_id"] >= 0])))
    )

    _rel_lo, _rel_hi = fit_span(cfg)
    _rel_use = (beh["run_mask"] & fit_span_mask(beh, cfg) & (beh["lap_id"] >= 0) & np.isfinite(beh["pos"]))

    _rel_Y = Y_rate
    _rel_nf, _rel_nc = _rel_Y.shape
    _rel_lap = beh["lap_id"]
    _rel_laps = np.unique(_rel_lap[_rel_use])

    _rel_nb = max(int(round((_rel_hi - _rel_lo) / REL_BIN_CM)), 4)
    _rel_edges = np.linspace(_rel_lo, _rel_hi, _rel_nb + 1)
    _rel_bin = np.clip(np.digitize(beh["pos"], _rel_edges) - 1, 0, _rel_nb - 1)

    _rel_ctr = 0.5 * (_rel_edges[:-1] + _rel_edges[1:])
    _rel_inzone = np.zeros(_rel_nb, bool)
    for _zl, _zh in cfg.zones:
        _rel_inzone |= (_rel_ctr >= _zl) & (_rel_ctr < _zh)
    _rel_lmbins = (_rel_inzone & (_rel_ctr >= REL_ONSET_CM)
                   & (_rel_ctr <= REL_REWARD_CM))

    def _rel_curve(Y, rows, bin_ix=None, nb=None):

        bin_ix = _rel_bin if bin_ix is None else bin_ix
        nb = _rel_nb if nb is None else nb
        b = bin_ix[rows]
        Yu = Y[rows]
        C = np.full((nb, Y.shape[1]), np.nan)
        for i in range(nb):
            s = b == i
            if int(s.sum()) >= REL_MIN_FRAMES:
                C[i] = Yu[s].mean(axis=0)

        return C


    def _pairwise_corr(A, B, bins=None):

        out = np.full(A.shape[1], np.nan)
        for c in range(A.shape[1]):
            a, b = A[:, c], B[:, c]
            m = np.isfinite(a) & np.isfinite(b)

            if bins is not None:
                m = m & bins
            if int(m.sum()) < REL_MIN_BINS:
                continue
            a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
            d = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
            if d > 1e-12:
                out[c] = float(np.dot(a, b) / d)

        return out


    def _split_half_r(Y, rng):

        acc = np.zeros((REL_N_SPLITS, Y.shape[1]))
        acc_f = np.zeros((REL_N_SPLITS, Y.shape[1]))

        for s in range(REL_N_SPLITS):
            perm = rng.permutation(_rel_laps)

            h1 = np.isin(_rel_lap, perm[:len(perm) // 2]) & _rel_use
            h2 = np.isin(_rel_lap, perm[len(perm) // 2:]) & _rel_use

            C1, C2 = _rel_curve(Y, h1), _rel_curve(Y, h2)

            acc[s] = _pairwise_corr(C1, C2, _rel_lmbins)
            acc_f[s] = _pairwise_corr(C1, C2)

        return np.nanmean(acc, axis=0), np.nanmean(acc_f, axis=0)


    _rel_rng = np.random.default_rng(0)
    rel_r, rel_r_full = _split_half_r(_rel_Y, _rel_rng)

    # null
    _rel_shift_lo = int(round(REL_MIN_SHIFT_S * cfg.fps))
    _rel_null, _rel_null_f = [], []
    for _s in range(REL_N_SHUFFLES):
        _sh = _rel_rng.integers(_rel_shift_lo, _rel_nf - _rel_shift_lo, size=_rel_nc)
        _ix = (np.arange(_rel_nf)[:, None] - _sh[None, :]) % _rel_nf
        _Ysh = np.take_along_axis(_rel_Y, _ix, axis=0)
        _a, _b = _split_half_r(_Ysh, np.random.default_rng(100 + _s))
        _rel_null.append(_a)
        _rel_null_f.append(_b)
        del _Ysh, _ix
    _rel_null_by_cell = np.vstack(_rel_null) # (n_shuffles, n_cells)
    _rel_null = _rel_null_by_cell.ravel() # pooled, for the threshold
    _rel_null_full = np.vstack(_rel_null_f).ravel()

    rel_thr = float(np.nanpercentile(_rel_null, REL_NULL_PCT))
    rel_thr_full = float(np.nanpercentile(_rel_null_full, REL_NULL_PCT))

    # rel_z is (r - cell's own null mean) / own null SD
    _nn = int(np.sum(np.isfinite(_rel_null)))
    _rn = np.asarray(rel_r, float)
    rel_pct = np.array([100.0 * float(np.nanmean(_rel_null < v)) if np.isfinite(v)
                        else np.nan for v in _rn])
    rel_p = np.array([max(float(np.nanmean(_rel_null >= v)), 1.0 / max(_nn, 1))
                      if np.isfinite(v) else np.nan for v in _rn])
    _nmu = np.nanmean(_rel_null_by_cell, axis=0)
    _nsd = np.nanstd(_rel_null_by_cell, axis=0)
    rel_z = (_rn - _nmu) / np.where(_nsd > 1e-9, _nsd, np.nan)
    _rel_r0 = np.nan_to_num(rel_r, nan=-np.inf)
    _rel_rf0 = np.nan_to_num(rel_r_full, nan=-np.inf)

    reliable = np.asarray((_rel_r0 > rel_thr) & (_rel_r0 >= REL_MIN_R))

    # this one includes reward nad onset zones
    reliable_full = np.asarray((_rel_rf0 > rel_thr_full) & (_rel_rf0 >= REL_MIN_R))

    end_only = reliable_full & ~reliable

    _end_nb = max(int(round(cfg.corridor_cm / REL_BIN_CM)), 4)
    _end_edges = np.linspace(0.0, cfg.corridor_cm, _end_nb + 1)
    _end_ctr = 0.5 * (_end_edges[:-1] + _end_edges[1:])
    _end_bin = np.clip(np.digitize(beh["pos"], _end_edges) - 1, 0, _end_nb - 1)
    _end_use = beh["run_mask"] & (beh["lap_id"] >= 0) & np.isfinite(beh["pos"])

    _end_inzone = np.zeros(_end_nb, bool)
    for _zl, _zh in cfg.zones:
        _end_inzone |= (_end_ctr >= _zl) & (_end_ctr < _zh)
    _end_onbins = _end_ctr < REL_ONSET_CM
    _end_rwbins = _end_ctr > REL_REWARD_CM

    _rel_pool = _rel_curve(_rel_Y, _end_use, _end_bin, _end_nb)
    with np.errstate(invalid="ignore"):
        _rel_base = np.nanmedian(_rel_pool[_end_inzone], axis=0)
        _amp_on = np.nanmax(np.abs(_rel_pool[_end_onbins] - _rel_base), axis=0)
        _amp_rw = np.nanmax(np.abs(_rel_pool[_end_rwbins] - _rel_base), axis=0)
        _amp_lm = np.nanmax(np.abs(_rel_pool[_end_inzone] - _rel_base), axis=0)
    _amp_on = np.nan_to_num(_amp_on, nan=-np.inf)
    _amp_rw = np.nan_to_num(_amp_rw, nan=-np.inf)
    _amp_lm = np.nan_to_num(_amp_lm, nan=-np.inf)

    rel_type = np.full(_rel_nc, "none", dtype=object)
    rel_type[reliable] = "landmark"
    rel_type[end_only & (_amp_rw >= _amp_on)] = "reward"
    rel_type[end_only & (_amp_rw < _amp_on)] = "onset"

    rel = dict(
        r=rel_r, ok=reliable, thr=rel_thr, null=_rel_null, floor=REL_MIN_R,
        pct=rel_pct, p=rel_p, z=rel_z, null_by_cell=_rel_null_by_cell,
        n_null=_nn, n_bins=_rel_nb, span=(_rel_lo, _rel_hi),
        n_splits=REL_N_SPLITS,
        r_full=rel_r_full, ok_full=reliable_full, thr_full=rel_thr_full,
        end_only=end_only, type=rel_type,
        lm_span=(REL_ONSET_CM, REL_REWARD_CM),
        n_lm_bins=int(_rel_lmbins.sum()),
        amp_onset=_amp_on, amp_reward=_amp_rw, amp_landmark=_amp_lm
    )

    print(" -> Fitting per-landmark gains...")
    lgain_fam = build_landmark_gains(beh, cfg, lagB)
    lad = fit_landmark_gains(Y_rate, beh, cfg, XPB, cand, lgain_fam, lagB,
                             scalePB=penPB)

    _wf_kernel = ((np.asarray(lad["landmark_gains"], float) > cfg.r2_threshold)
                  & np.asarray(rel["ok"], bool))
    _wf = _wf_kernel

    SHAPE_NAMES = ["four equal peaks"] + ["L{} preference".format(j + 1)
                                          for j in range(4)]

    LADDER_LAMBDAS = np.array([1.0, 10.0, 100.0, 1000.0])


    print(" -> Building and fitting shape templates...")
    # shape contrasts are orthogonalized against behavior only if it's in the model
    shp_fam = build_shapes(beh, cfg, lagB,
                           XPB=XPB if cfg.use_pure_behavior else None)
    shp = fit_shapes(Y_rate, beh, cfg, XPB, shp_fam, scalePB=penPB,
                     LADDER_LAMBDAS=LADDER_LAMBDAS)

    # floor check
    _flat = _wf & (shp["shape_mode"] == 0)
    _decisive = np.nan_to_num(lad["gain_z"], nan=0.0) > 3.0
    _conflict = _flat & _decisive
    _nflat = int(_flat.sum())
    # now flat is the null, updated sept 9
    _missed = _conflict & (np.nan_to_num(shp["flat_z"], nan=0.0) > 3.0)
    if _nflat and _conflict.sum() > 0.25 * _nflat:
        print("  NOTE: {} of {} well-fit cells called 'four equal peaks' have "
              "gain_z > 3 from the free per-landmark fit ({:.0f}%), and {} of "
              "those also have flat_z > 3 -- those are the ones the shape "
              "family may genuinely not reach; the rest are the flat-as-null "
              "gate doing its job.".format(
                  int(_conflict.sum()), _nflat,
                  100 * _conflict.sum() / max(_nflat, 1), int(_missed.sum())))

    _pr = shp["pref_ratio"][_wf & (shp["shape_mode"] >= 1)]
    if np.isfinite(_pr).any():
        print("  Fitted preference ratio, cells given a preferred landmark: "
              "median {:.2f}x (IQR {:.2f}-{:.2f}x).".format(
                  float(np.nanmedian(_pr)),
                  float(np.nanpercentile(_pr, 25)),
                  float(np.nanpercentile(_pr, 75))))


    # one pure-behavior refit feeds both PDFs
    print(" -> Computing display tuning curves...")
    curves = _compute_display_curves(beh, Y_rate, XPB, cfg, scalePB=penPB)

    print(" -> Generating PDF of good cells...")
    plot_review_pdf(
        lad=lad, shp=shp, rel=rel, cfg=cfg, cell_ids=cell_ids, outdir=OUTDIR,
        wf=_wf, wf_kernel=_wf_kernel, curves=curves,
        shape_names=SHAPE_NAMES,
        edge_cm=REL_REWARD_CM, start_cm=REL_ONSET_CM, reward_cm=REWARD_CM
    )

    print(" -> Generating PDF of excluded cells...")
    plot_excluded_pdf(
        lad=lad, rel=rel, cfg=cfg, cell_ids=cell_ids, outdir=OUTDIR,
        wf=_wf, curves=curves,
        edge_cm=REL_REWARD_CM, start_cm=REL_ONSET_CM, reward_cm=REWARD_CM
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-s2p', '--suite2p_dir', type=str, default=None,
                        help='Path to suite2p directory')
    parser.add_argument('-vr', '--vrlog_file', type=str, default=None,
                        help='Path to VR log file')
    parser.add_argument('-b', '--batch', action='store_true', help='Run in batch mode')
    parser.add_argument('-bd', '--batch_dir', type=str, default=None, help='Directory for batch recordings')
    args = parser.parse_args()


    if not args.batch:

        if args.suite2p_dir is None:
            SUITE2P = select_directory('Select suite2p plane directory.')
        else:
            SUITE2P = args.suite2p_dir
        if args.vrlog_file is None:
            VRLOG = select_file('Select VR log text file.', filetypes=[('TXT', '*.txt'),])
        else:
            VRLOG = args.vrlog_file
            
        outdir = os.path.split(VRLOG)[0]
        if outdir == '':
            outdir = os.getcwd()

        main(SUITE2P, VRLOG, outdir)

    elif args.batch:

        bdir = args.batch_dir
        logfiles = glob(os.path.join(bdir, '*/*.txt'), recursive=True)
        for i, logfile in enumerate(logfiles):
            logbase = os.path.split(logfile)[0]
            suite2p_dir = os.path.join(logbase, 'suite2p/plane0')

            main(suite2p_dir, logfile, logbase)
