# -*- coding: utf-8 -*-
"""
landmarkGLM/pure_behavior_block.py

Builds the behavior dict from the VR log, and the pure-behavior design
matrix.

DMM, Aug 2026
"""

import numpy as np
import pandas as pd

from basis_funcs import raised_cosine_value_basis, raised_cosine_lag_basis, lag_expand, rc_value_spans


def build_behavior(vr_df, n_frames, cfg):

    t_2p = np.arange(n_frames) / cfg.fps + getattr(cfg, "t_offset_s", 0.0)

    pos_rows = vr_df["EventType"].values == "p"
    vt = vr_df["ElapsedTime(seconds)"].values[pos_rows]
    vp = vr_df["location_cm"].values[pos_rows]
    ok = np.isfinite(vt) & np.isfinite(vp)
    vt, vp = vt[ok], vp[ok]
    order = np.argsort(vt)
    vt, vp = vt[order], vp[order]

    pos_2p = np.interp(t_2p, vt, vp)

    gap_i = np.where(np.diff(vt) > cfg.iti_gap_s)[0]
    iti = np.zeros(n_frames, dtype=bool)
    for i in gap_i:
        iti |= (t_2p > vt[i]) & (t_2p < vt[i + 1])
    if gap_i.size:

        for i in gap_i:
            m = (t_2p > vt[i]) & (t_2p < vt[i + 1])
            pos_2p[m] = vp[i]

    d = np.diff(pos_2p, prepend=pos_2p[0])
    d[d < -50] = np.nan
    d = pd.Series(d).ffill().bfill().values
    speed = np.abs(d) * cfg.fps
    accel = np.gradient(speed) * cfg.fps

    lap_times = vr_df["ElapsedTime(seconds)"].values[vr_df["EventType"].values == "n"]
    lap_times = lap_times[np.isfinite(lap_times)]
    lap_frames = np.searchsorted(t_2p, lap_times)
    lap_frames = np.clip(lap_frames, 0, n_frames - 1)
    lap_frames = np.unique(lap_frames)

    lap_id = np.zeros(n_frames, dtype=int)
    t_since_lap = np.zeros(n_frames)
    for i, f0 in enumerate(lap_frames):
        f1 = lap_frames[i + 1] if i + 1 < len(lap_frames) else n_frames
        lap_id[f0:f1] = i
        t_since_lap[f0:f1] = (np.arange(f0, f1) - f0) / cfg.fps
    if len(lap_frames) and lap_frames[0] > 0:
        lap_id[:lap_frames[0]] = -1
        t_since_lap[:lap_frames[0]] = np.arange(lap_frames[0]) / cfg.fps

    rew_times = vr_df["ElapsedTime(seconds)"].values[vr_df["EventType"].values == "r"]
    rew_times = np.sort(rew_times[np.isfinite(rew_times)])
    if len(rew_times):
        idx = np.searchsorted(rew_times, t_2p, side="left")
        t_to_reward = np.where(idx < len(rew_times),
                               rew_times[np.clip(idx, 0, len(rew_times) - 1)] - t_2p,
                               np.nan)
        finite = np.isfinite(t_to_reward)
        t_to_reward[~finite] = np.nanmax(t_to_reward[finite]) if finite.any() else 0.0
        t_to_reward = np.clip(t_to_reward, 0, np.nanpercentile(t_to_reward, 99))
    else:
        t_to_reward = np.zeros(n_frames)

    rew_pos_med = np.nan
    if len(rew_times):
        rew_frames = np.unique(np.clip(np.searchsorted(t_2p, rew_times),
                                       0, n_frames - 1))
    else:
        rew_frames = np.zeros(0, dtype=int)

    if len(rew_times):

        j = np.searchsorted(vt, rew_times, side="right") - 1
        ok_j = j >= 0
        rew_pos = vp[np.clip(j, 0, len(vp) - 1)][ok_j]
        lag = (rew_times[ok_j] - vt[np.clip(j, 0, len(vp) - 1)][ok_j])
        rew_pos_med = float(np.median(rew_pos))
        frac = np.median(rew_pos) / max(cfg.corridor_cm, 1e-9)
        if np.std(rew_pos) < 2.0:

            if frac > 0.85:
                pass
        else:

            pass

    zone_idx = np.full(n_frames, -1, dtype=int)
    zone_phase = np.zeros(n_frames)
    for z, (z0, z1) in enumerate(cfg.zones):
        inside = (pos_2p >= z0) & (pos_2p <= z1)
        zone_idx[inside] = z
        zone_phase[inside] = (pos_2p[inside] - z0) / (z1 - z0)

    stationary = speed < cfg.speed_thresh
    min_len = int(cfg.min_stationary_s * cfg.fps)
    excluded = np.zeros(n_frames, dtype=bool)
    i = 0
    while i < n_frames:
        if stationary[i]:
            j = i
            while j < n_frames and stationary[j]:
                j += 1
            if j - i >= min_len:
                excluded[i:j] = True
            i = j
        else:
            i += 1
    run_mask = ~excluded & (lap_id >= 0) & ~iti

    return dict(t_2p=t_2p, pos=pos_2p, speed=speed, accel=accel,
                lap_id=lap_id, t_since_lap=t_since_lap, t_to_reward=t_to_reward,
                zone_idx=zone_idx, zone_phase=zone_phase, run_mask=run_mask,
                lap_frames=lap_frames, iti=iti, rew_pos_cm=rew_pos_med,
                rew_frames=rew_frames)


def build_purebehavior(beh, cfg):

    n = len(beh["pos"])
    m_run = beh["run_mask"]

    def _span(name):

        s = getattr(cfg, name, None)
        return (None, None) if s is None else (float(s[0]), float(s[1]))

    _sp = _span("speed_span_cms")
    _ac = _span("accel_span_cms2")
    _lt = _span("lapt_span_s")
    _rt = _span("rewt_span_s")

    _lin = bool(getattr(cfg, "linear_speed_accel", False))

    _use_accel = bool(getattr(cfg, "accel_in_PB", True))

    def _zcol(v):
        """ One linear column, standardized on the frames the model is fit on. """

        v = np.asarray(v, dtype=float)
        mu = np.nanmean(v[m_run]) if m_run.any() else 0.0
        sd = np.nanstd(v[m_run]) if m_run.any() else 1.0
        out = (v - mu) / (sd if sd > 0 else 1.0)
        out[~np.isfinite(out)] = 0.0
        return out[:, None]

    if _lin:
        speed_col = _zcol(beh["speed"])
        accel_col = _zcol(beh["accel"]) if _use_accel else np.zeros((n, 0))
    else:
        speed_col = raised_cosine_value_basis(beh["speed"], cfg.n_speed_basis,
                                              lo=_sp[0], hi=_sp[1], valid=m_run)
        accel_col = (raised_cosine_value_basis(beh["accel"], cfg.n_accel_basis,
                                               lo=_ac[0], hi=_ac[1], valid=m_run)
                     if _use_accel else np.zeros((n, 0)))
    N_run = np.hstack([speed_col, accel_col])

    on_imp = np.zeros((n, 0))
    if bool(getattr(cfg, "onset_impulse_in_PB", True)):
        on_imp = np.zeros((n, 1))
        on_imp[np.asarray(beh["lap_frames"], dtype=int), 0] = 1.0

    lapt_col = np.zeros((n, 0))
    if bool(getattr(cfg, "lap_time_in_PB", True)):
        lapt_col = raised_cosine_value_basis(beh["t_since_lap"], cfg.n_lapt_basis,
                                             lo=_lt[0], hi=_lt[1], valid=m_run)
    rewt_col = np.zeros((n, 0))
    if bool(getattr(cfg, "reward_time_in_PB", True)):
        rewt_col = raised_cosine_value_basis(beh["t_to_reward"], cfg.n_rewt_basis,
                                             lo=_rt[0], hi=_rt[1], valid=m_run)
    N_lap = np.hstack([lapt_col, rewt_col])

    on_col = np.zeros((n, 0))
    on_mu = np.zeros(0)
    sig_o = float(getattr(cfg, "onset_sigma_cm", 1.0))
    use_on = bool(getattr(cfg, "onset_gauss_in_PB", False))
    if use_on:
        lo_o, hi_o = cfg.onset_span_cm
        k_o = int(cfg.onset_n_basis)
        on_mu = (np.array([0.5 * (lo_o + hi_o)]) if k_o == 1
                 else np.linspace(lo_o, hi_o, k_o))
        on_col = np.exp(-0.5 * ((beh["pos"][:, None] - on_mu[None, :])
                                / sig_o) ** 2)

    rew_col = np.zeros((n, 0))
    mu_r = beh.get("rew_pos_cm", np.nan)
    use_rew = bool(getattr(cfg, "reward_gauss_in_PB", False)) and np.isfinite(mu_r)
    rew_mu = np.zeros(0)
    sig_r = float(getattr(cfg, "reward_sigma_cm", 1.0))
    cut = getattr(cfg, "reward_cut_cm", None)
    cut = cfg.corridor_cm if cut is None else float(cut)
    if use_rew:
        lo, hi = cfg.reward_span_cm
        k_r = int(cfg.reward_n_basis)
        rew_mu = (np.array([0.5 * (lo + hi)]) if k_r == 1
                  else np.linspace(lo, hi, k_r))

        rew_col = np.exp(-0.5 * ((beh["pos"][:, None] - rew_mu[None, :])
                                 / sig_r) ** 2)
        rew_col[beh["pos"] > cut, :] = 0.0
    elif getattr(cfg, "reward_gauss_in_PB", False):
        pass

    rew_f = np.asarray(beh.get("rew_frames", np.zeros(0, dtype=int)), dtype=int)
    use_rimp = bool(getattr(cfg, "reward_impulse_in_PB", False)) and rew_f.size > 0
    rimp = np.zeros((n, 1))
    if use_rimp:
        rimp[rew_f, 0] = 1.0
    elif getattr(cfg, "reward_impulse_in_PB", False):
        pass

    w_speed = speed_col.shape[1]
    w_accel = accel_col.shape[1]
    w_onimp = on_imp.shape[1]
    w_onpos = on_col.shape[1]
    w_rewpos = rew_col.shape[1]
    w_lapt = lapt_col.shape[1]
    w_rewt = rewt_col.shape[1]

    PB_raw = np.hstack([N_run, on_imp, on_col, rew_col, N_lap])
    if PB_raw.shape[1] == 0:
        raise ValueError("the pure behavior block is empty -- every flag is off")

    iti = beh.get("iti")
    if iti is not None and iti.any():
        PB_raw = PB_raw.copy()
        PB_raw[iti] = 0.0

    lagB = raised_cosine_lag_basis(cfg.n_lag_basis, cfg.lag_max_s, cfg.fps,
                                   c=cfg.lag_log_offset)
    XPB = lag_expand(PB_raw, lagB)
    n_lagged = XPB.shape[1]    # end of the ordinary PB block

    XR = np.zeros((n, 0))
    if use_rimp:
        lagB_r = raised_cosine_lag_basis(cfg.n_reward_lag_basis,
                                         cfg.reward_lag_max_s, cfg.fps,
                                         c=getattr(cfg, "reward_lag_offset",
                                                   cfg.lag_log_offset))
        XR = lag_expand(rimp, lagB_r)
        XPB = np.hstack([XPB, XR])

    if _lin:
        pass
    _off = [nm for nm, w in (("acceleration", w_accel),
                             ("onset-impulse", w_onimp),
                             ("onset-position", w_onpos),
                             ("reward-position", w_rewpos),
                             ("lap-time", w_lapt),
                             ("reward-time", w_rewt)) if w == 0]
    if _off:
        pass

    if use_rimp:
        rm0 = beh["run_mask"]
        itim = beh.get("iti")
        n_iti = int(itim[rew_f].sum()) if itim is not None else 0

        live = float((XR[rm0] ** 2).sum() / max((XR ** 2).sum(), 1e-12))

        if live < 0.05:
            pass

        lt = beh["t_2p"][np.asarray(beh["lap_frames"], dtype=int)]
        rt = beh["t_2p"][rew_f]
        gaps = np.array([lt[lt > r][0] - r for r in rt if (lt > r).any()])
        if gaps.size:
            iqr = np.subtract(*np.percentile(gaps, [75, 25]))

            if iqr < 0.25:

                pass

        e = (XR ** 2).sum(axis=1) * rm0
        if e.sum() > 0:
            edges = [(0.0, cfg.zones[0][0])] + list(cfg.zones) + \
                    [(cfg.zones[-1][1], cfg.corridor_cm)]
            labs = ["pre-A"] + ["ABCD"[i] for i in range(len(cfg.zones))] + ["reward end"]
            share = {l: 100 * e[(beh["pos"] >= a) & (beh["pos"] < b)].sum() / e.sum()
                     for l, (a, b) in zip(labs, edges)}
            if share[labs[-1]] < 25:
                pass

    def _report_pos_basis(label, col, mus, sig, zi, core_lo, core_hi, hint,
                          extra=""):
        
        rmv = beh["run_mask"]
        env = col.max(axis=1)
        pr = beh["pos"][rmv]
        z0, z1 = cfg.zones[zi]
        mid = 0.5 * (z0 + z1)
        zname = "ABCD"[zi] if zi < 4 else str(zi + 1)

        core = (pr >= core_lo) & (pr <= core_hi)
        env_mid = float(np.exp(-0.5 * ((mid - mus) / sig) ** 2).max())
        enc = float((env[rmv][core] > 0.5).mean()) if core.any() else 0.0

        if enc > 0.05 or env_mid > 0.25:
            pass

    if use_on:
        zA0, zA1 = cfg.zones[0]
        _report_pos_basis("lap-onset basis", on_col, on_mu, sig_o, 0,
                          zA0 + 5.0, zA1,
                          "Lower onset_span_cm[1] or shrink onset_sigma_cm.")

    if use_rew:
        z0, z1 = cfg.zones[-1]
        _report_pos_basis("reward-approach basis", rew_col, rew_mu, sig_r,
                          len(cfg.zones) - 1, z0, z1 - 5.0,
                          "Raise reward_span_cm[0] or shrink reward_sigma_cm.",
                          extra=", hard zero past {:.0f} cm "
                                "(reward is at {:.1f})".format(cut, mu_r))

    m = beh["run_mask"]
    pos_c = beh["pos"][m] - beh["pos"][m].mean()
    A = XPB[m] - XPB[m].mean(axis=0)
    coef, *_ = np.linalg.lstsq(A, pos_c, rcond=None)
    r2p = 1 - np.sum((pos_c - A @ coef) ** 2) / np.sum(pos_c ** 2)

    p = PB_raw.shape[1]
    j_on = w_speed + w_accel + w_onimp
    j_r = j_on + w_onpos
    for _lab, _j0, _w in (("lap-onset", j_on, w_onpos),
                          ("reward-approach", j_r, w_rewpos)):
        if _w == 0:
            continue
        keep = np.ones(XPB.shape[1], dtype=bool)
        for j in range(_j0, _j0 + _w):
            keep[j:n_lagged:p] = False
        A0 = XPB[m][:, keep] - XPB[m][:, keep].mean(axis=0)
        c, *_ = np.linalg.lstsq(A0, pos_c, rcond=None)
        r2p0 = 1 - np.sum((pos_c - A0 @ c) ** 2) / np.sum(pos_c ** 2)

    penPB = np.ones(XPB.shape[1])
    _sc = float(getattr(cfg, "purebehavior_pos_scale", 1.0))
    _npos = 0
    if _sc != 1.0 and (w_onpos + w_rewpos) > 0:
        for j in range(j_on, j_r + w_rewpos):
            penPB[j:n_lagged:p] = _sc
            _npos += len(range(j, n_lagged, p))

    _mid = beh["run_mask"]

    _fw = 1.1774
    if _lin:
        _lab_speed = ["speed (linear, z-scored)"]
        _lab_accel = ["acceleration (linear, z-scored)"] if _use_accel else []
    else:
        _lab_speed = rc_value_spans(beh["speed"], cfg.n_speed_basis, "speed",
                                    "cm/s", lo=_sp[0], hi=_sp[1], valid=_mid)
        _lab_accel = (rc_value_spans(beh["accel"], cfg.n_accel_basis, "accel",
                                     "cm/s2", lo=_ac[0], hi=_ac[1], valid=_mid,
                                     fmt="{:+.0f}")
                      if _use_accel else [])
    labels = (_lab_speed
              + _lab_accel
              + ["lap-onset impulse (t = 0)"] * w_onimp
              + ["onset-pos {:.1f} to {:.1f} cm".format(v - _fw * sig_o,
                                                        v + _fw * sig_o)
                 for v in on_mu]
              + ["reward-pos {:.1f} to {:.1f} cm".format(
                     v - _fw * sig_r, min(v + _fw * sig_r, cut))
                 for v in rew_mu]
              + (rc_value_spans(beh["t_since_lap"], cfg.n_lapt_basis, "lap-time",
                                "s", lo=_lt[0], hi=_lt[1], valid=_mid)
                 if w_lapt else [])
              + (rc_value_spans(beh["t_to_reward"], cfg.n_rewt_basis,
                                "reward-time", "s", lo=_rt[0], hi=_rt[1],
                                valid=_mid) if w_rewt else []))
    assert len(labels) == PB_raw.shape[1], (len(labels), PB_raw.shape[1])

    _widths = [("speed", w_speed),
               ("acceleration", w_accel),
               ("lap-onset impulse", w_onimp),
               ("onset-position", w_onpos),
               ("reward-position", w_rewpos),
               ("time since lap start", w_lapt),
               ("time to reward", w_rewt)]
    blocks, _j = [], 0
    for _nm, _wd in _widths:
        if _wd:
            blocks.append((_nm, _j, _wd))
        _j += _wd
    assert _j == PB_raw.shape[1], (_j, PB_raw.shape[1])
    pbinfo = dict(raw=PB_raw, labels=labels, p=PB_raw.shape[1], n_lagged=n_lagged,
                 blocks=blocks,
                 lag_basis=lagB, penalty=penPB,
                 impulse=(rimp if use_rimp else None))
    return XPB, lagB, penPB, pbinfo
