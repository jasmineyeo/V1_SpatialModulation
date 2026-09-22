"""
lick_metrics.py
===============

Per-session licking metrics for the question:
**does anticipatory licking narrow onto the reward zone as the animal learns?**

Corridor
--------
The animal runs a ~130 cm corridor; the reward zone is the far end. On
reaching it: black screen (~1.5 s, no VR position logged) -> water (valve,
LED r-flash) -> drink (~1.5 s) -> teleport (LED n-flash, next trial). A
drinking bout straddles the valve and can spill past the next teleport.

AU -> cm : `cm = au * 130.0 / (min(max_au, 393) - min_au)`  (matches
`helper/BehavioralDataFiltering.py`, so licking cm == neural-analysis cm).

Lick categories
---------------
A **drinking bout** = a chin bout that spans a trial's valve; the whole
bout (incl. frames after the next teleport) belongs to that trial's reward.

    consummatory_pre_reward   in a drinking bout, before the valve
                              (black-screen temporal anticipation)
    consummatory_post_reward  in a drinking bout, at/after the valve
                              (drinking; absorbs the old "carryover")
    approaching_reward        not a drinking bout, has position,
                              RZ-30cm <= pos < RZ-3cm  (anticipation while running)
    approaching_nonreward     not a drinking bout, has position, pos < RZ-30cm
                              (scattered / exploratory)

The **lick-rate-vs-position curve** uses the two `approaching_*` classes
(occupancy-normalised). The **reward PSTH** uses all licks by time-from-valve.

Functions
---------
au_to_cm_factor, classify_licks, lick_rate_vs_position,
speed_accel_vs_position, reward_psth, first_approach_lick,
lick_scatter_data, session_metrics, rolling_metrics,
vr_speed_trace, tm_vr_speed_alignment, speed_decel_onset, drinking_bouts,
trial_durations_s, trial_approach_speed_cms, spatial_temporal_onset_table,
spatial_vs_temporal_fit, classify_trial_anticipation_strategy,
frac_trials_with_category, frac_trials_post_vr_onset_licking,
black_screen_running_speed, trial_cluster_bootstrap, bootstrap_pvalue

JSY / V1_SpatialModulation - 09.LickingBehavior
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lick_sync import frame_to_vr, FPS


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def au_to_cm_factor(vr, phys_len_cm: float = 130.0, track_cap_au: float = 393.0) -> float:
    """cm per AU, per session (helper/BehavioralDataFiltering.py convention)."""
    loc = vr.position.location_au.to_numpy()
    track_au = min(float(np.max(loc)), track_cap_au)
    return phys_len_cm / (track_au - float(np.min(loc)))


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------

def classify_licks(licks: pd.DataFrame, trials: pd.DataFrame, bouts: pd.DataFrame,
                   cm_per_au: float, reward_zone_au: float,
                   rz_pad_cm: float = 3.0, antic_cm: float = 30.0,
                   valve_pad_frames: int = 6, carryover_frames: int = 45) -> pd.DataFrame:
    """Add `category` (+ `drink_bout`, corrected `trial`) to the licks table.

    bouts : DataFrame with start_frame / end_frame (from notebook 2).
    carryover_frames : licks up to this many frames after a teleport belong to
      the PREVIOUS trial's reward (the animal is still drinking; the chin bout
      is often broken by the ~0.5 s teleport screen-flash so these would
      otherwise be mis-labelled `approaching_nonreward` at position ~ 0).
    """
    out = licks.copy()
    out["trial"] = out["trial"].astype(int)
    out["bout_id"] = out["bout_id"].astype(int)
    frames = out["frame"].to_numpy()

    r_anchor = trials.set_index("trial")["r_anchor_frame"].to_dict()
    n_anchor = trials.set_index("trial")["n_anchor_frame"].to_dict()

    # --- which chin bouts are drinking bouts, and for which trial ---
    bs = bouts["start_frame"].to_numpy()
    be = bouts["end_frame"].to_numpy()
    bout_trial = np.full(len(bouts), -1)
    for tr, rf in r_anchor.items():
        if not np.isfinite(rf):
            continue
        hit = np.where((bs - valve_pad_frames <= rf) & (rf <= be + valve_pad_frames))[0]
        for b in hit:
            bout_trial[b] = tr

    # --- carry-over window: licks around each teleport belong to trial-1 ---
    carry = np.full(len(out), -1)
    for tr in trials["trial"].astype(int):
        tele = n_anchor.get(tr, np.nan)              # teleport INTO trial tr = end of (tr-1)'s reward
        if tr - 1 < 1 or not np.isfinite(tele):
            continue
        m = (frames >= tele - valve_pad_frames) & (frames <= tele + carryover_frames)
        carry[m] = tr - 1

    rz_lo_au = reward_zone_au - (antic_cm / cm_per_au)      # start of "approaching_reward"
    rz_hi_au = reward_zone_au - (rz_pad_cm / cm_per_au)     # end of it / start of consummatory-by-position

    cat = np.empty(len(out), dtype=object)
    drink = np.full(len(out), -1)
    trial_fixed = out["trial"].to_numpy().copy()

    for i, row in enumerate(out.itertuples()):
        bid = int(row.bout_id)
        dtr = bout_trial[bid] if bid >= 0 else -1
        if dtr < 0 and carry[i] >= 0:               # post-teleport carry-over drinking
            dtr = int(carry[i])
        if dtr >= 0:
            drink[i] = dtr
            trial_fixed[i] = dtr
            rf = r_anchor.get(dtr, np.inf)
            cat[i] = "consummatory_pre_reward" if row.frame < rf else "consummatory_post_reward"
        else:
            pos = row.position_au
            if not np.isfinite(pos) or pos >= rz_hi_au:
                cat[i] = "consummatory_post_reward"        # at the end / black screen, no drinking bout matched
            elif pos >= rz_lo_au:
                cat[i] = "approaching_reward"
            else:
                cat[i] = "approaching_nonreward"

    out["category"] = cat
    out["drink_bout"] = drink
    out["trial"] = trial_fixed
    return out


# --------------------------------------------------------------------------
# lick rate vs position  (occupancy-normalised)
# --------------------------------------------------------------------------

def _approach_windows(trials: pd.DataFrame):
    for row in trials.itertuples():
        if np.isfinite(row.r_time_vr):
            yield int(row.trial), float(row.n_time_vr), float(row.r_time_vr)


def _freeze_intervals(vr, freeze_thresh_s: float):
    """(starts, ends) of stretches where the logged position is unchanged for
    >= freeze_thresh_s -- the animal is stationary/paused, not in transit
    (some sessions have this for 30-50%+ of their duration, growing across
    the B1->...->DCZ200 sequence for both animals)."""
    loc = vr.position.location_au.to_numpy()
    t = vr.position.elapsed_s.to_numpy()
    if len(t) < 2:
        return np.empty(0), np.empty(0)
    change = np.r_[True, loc[1:] != loc[:-1]]      # True at the first row of each run
    run_start = np.where(change)[0]
    run_end = np.r_[run_start[1:] - 1, len(loc) - 1]
    dur = t[run_end] - t[run_start]
    keep = dur >= freeze_thresh_s
    return t[run_start][keep], t[run_end][keep]


def _in_intervals(x, starts, ends):
    """Boolean mask: is each value of x inside any [starts[i], ends[i]]?"""
    x = np.asarray(x, float)
    if len(starts) == 0:
        return np.zeros(len(x), dtype=bool)
    j = np.clip(np.searchsorted(starts, x, side="right") - 1, 0, len(starts) - 1)
    return (x >= starts[j]) & (x <= ends[j])


def position_frozen_fraction(vr, freeze_thresh_s: float = 2.0) -> dict:
    """QC summary of how much of the session the logged position sat still
    for >= freeze_thresh_s (excluded from `lick_rate_vs_position`'s dwell)."""
    fz_s, fz_e = _freeze_intervals(vr, freeze_thresh_s)
    pe = vr.position.elapsed_s.to_numpy()
    total = float(pe.max() - pe.min()) if len(pe) else np.nan
    frozen = float((fz_e - fz_s).sum())
    return dict(n_freezes=int(len(fz_s)),
               frozen_s=frozen, session_s=total,
               pct_frozen=float(100 * frozen / total) if total else np.nan)


def lick_rate_vs_position(licks: pd.DataFrame, vr, trials: pd.DataFrame,
                          cm_per_au: float, binw_cm: float = 5.0,
                          corridor_cm: float = 132.0, grid_dt: float = 0.01,
                          trial_mask=None, freeze_thresh_s: float = 2.0) -> pd.DataFrame:
    """Occupancy-normalised lick rate per position bin, over the approach.

    freeze_thresh_s : stretches where the logged position doesn't change for
      at least this long are excluded from BOTH dwell and the lick count --
      real elapsed time, but a pause, not transit (same treatment as the
      black-screen gap: real, but not counted as time-at-that-position)."""
    pe = vr.position.elapsed_s.to_numpy()
    pl_cm = vr.position.location_au.to_numpy() * cm_per_au
    bins = np.arange(0, corridor_cm + binw_cm, binw_cm)
    bc = (bins[:-1] + bins[1:]) / 2

    keep = (set(trials.trial.astype(int)) if trial_mask is None
            else set(np.asarray(trials.trial.astype(int))[np.asarray(trial_mask)]))

    fz_s, fz_e = _freeze_intervals(vr, freeze_thresh_s)

    dwell = np.zeros(len(bc))
    for tr, t0, t1 in _approach_windows(trials):
        if tr not in keep or t1 <= t0:
            continue
        # stop at the last logged position of this trial: position isn't
        # logged during the black screen before reward, so interpolating
        # past it (or across the teleport) draws a fake ramp from ~RZ down
        # to the next trial's reset position, smearing "dwell" backward
        # across the whole corridor and suppressing rate near the RZ.
        seg = (pe >= t0) & (pe <= t1)
        if seg.sum() < 2:
            continue
        t1e = float(pe[seg][-1])
        tg = np.arange(t0, t1e, grid_dt)
        if len(tg) == 0:
            continue
        tg = tg[~_in_intervals(tg, fz_s, fz_e)]         # drop freeze-stretch time
        if len(tg):
            dwell += np.histogram(np.interp(tg, pe, pl_cm), bins=bins)[0] * grid_dt

    ap = licks[licks.category.isin(["approaching_nonreward", "approaching_reward"])
               & licks.trial.isin(keep)]
    ap = ap[~_in_intervals(ap.vr_time.to_numpy(), fz_s, fz_e)]   # drop licks made while frozen
    n_licks = np.histogram((ap.position_au.to_numpy() * cm_per_au), bins=bins)[0]

    rate = np.where(dwell > 0.3, n_licks / dwell, np.nan)
    return pd.DataFrame({"bin_cm": bc, "dwell_s": dwell, "n_licks": n_licks, "rate_hz": rate})


def vr_speed_trace(vr, cm_per_au: float, trials: pd.DataFrame,
                   freeze_thresh_s: float = 2.0, dt: float = 0.1):
    """Session-wide running speed (cm/s) derived from the VR position log, as
    a plain (elapsed_s, cm/s) trace over every trial's actual transit window
    -- same exclusions as `speed_accel_vs_position` (stop at each trial's last
    logged position; drop freeze stretches), but returned as a time series
    (not binned by position) so it can be compared directly against the TM
    log's own speed trace on the same clock. This is the GROUND-TRUTH speed
    (AU->cm is an exact per-session factor); the TM log is the thing being
    cross-checked against it, not the other way round."""
    pe = vr.position.elapsed_s.to_numpy()
    pl_cm = vr.position.location_au.to_numpy() * cm_per_au
    fz_s, fz_e = _freeze_intervals(vr, freeze_thresh_s)
    times, speeds = [], []
    for tr, t0, t1 in _approach_windows(trials):
        seg = (pe >= t0) & (pe <= t1)
        if seg.sum() < 3:
            continue
        t1e = float(pe[seg][-1])
        tg = np.arange(t0, t1e, dt)
        if len(tg) < 3:
            continue
        pos = np.interp(tg, pe, pl_cm)
        spd = np.clip(np.gradient(pos, dt), 0, 150)
        keep = ~_in_intervals(tg, fz_s, fz_e)
        if keep.any():
            times.append(tg[keep]); speeds.append(spd[keep])
    if not times:
        return np.empty(0), np.empty(0)
    return np.concatenate(times), np.concatenate(speeds)


def tm_vr_speed_alignment(vr, tm, trials: pd.DataFrame, cm_per_au: float,
                          freeze_thresh_s: float = 2.0, dt: float = 0.1,
                          lag_search_s: float = 3.0, lag_step_s: float = 0.1) -> dict:
    """Cross-check the TM log's running speed (raw encoder units, unknown
    scale) against the VR-position-derived speed (cm/s, exact). The two logs
    are supposed to share one wall-clock zero (`lick_io.read_tmlog`), but
    rather than assume that, this searches a small window of relative time
    lag for the offset that makes the two traces agree best, then fits a
    robust linear calibration (cm/s = scale*tm_raw + intercept) at that lag.

    Returns `lag_s` (query-time shift applied to the VR trace before sampling
    TM at it; near 0 confirms the shared-clock assumption), `scale`,
    `intercept`, `r2` (fit quality = how well the two independent
    measurements of running speed agree -- the actual "do they align"
    answer), `n`, `tm_mean_cms` / `vr_mean_cms` (session-average speed by
    each method, tm's using this session's own calibration).
    """
    vt, vs = vr_speed_trace(vr, cm_per_au, trials, freeze_thresh_s, dt)
    empty = dict(lag_s=np.nan, scale=np.nan, intercept=np.nan, r2=np.nan, n=0,
                tm_mean_cms=np.nan, vr_mean_cms=float(np.mean(vs)) if len(vs) else np.nan)
    if len(vt) < 20 or tm is None:
        return empty

    lags = np.arange(-lag_search_s, lag_search_s + lag_step_s, lag_step_s)
    best = None
    for lag in lags:
        tm_raw = tm.speed_at(vt + lag)
        ok = np.isfinite(tm_raw) & np.isfinite(vs)
        if ok.sum() < 20:
            continue
        r = np.corrcoef(tm_raw[ok], vs[ok])[0, 1]
        if np.isfinite(r) and (best is None or r > best[0]):
            best = (r, lag)
    if best is None:
        return empty

    _, lag = best
    tm_raw = tm.speed_at(vt + lag)
    ok = np.isfinite(tm_raw) & np.isfinite(vs)
    x, y = tm_raw[ok], vs[ok]
    scale, intercept = np.polyfit(x, y, 1)
    pred = scale * x + intercept
    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return dict(lag_s=float(lag), scale=float(scale), intercept=float(intercept),
               r2=r2, n=int(ok.sum()), tm_mean_cms=float(np.mean(pred)),
               vr_mean_cms=float(np.mean(y)))


def speed_decel_onset(spd: pd.DataFrame, reward_zone_cm: float,
                      frac_of_peak: float = 0.7) -> float:
    """Position (cm) where running speed first drops below `frac_of_peak` of
    its own plateau value, scanning backward from the reward zone -- the
    motor/approach analogue of the lick-based `anticipatory_onset_cm`,
    independent of licking. "Plateau" = median of the top-20% speed bins
    (robust to a single spuriously-fast bin)."""
    bc = spd.bin_cm.to_numpy(); sp = spd.speed_cm_s.to_numpy()
    ok = np.isfinite(sp) & (bc < reward_zone_cm)
    if ok.sum() < 3:
        return np.nan
    top = np.sort(sp[ok])[-max(1, ok.sum() // 5):]
    peak = float(np.median(top))
    thr = frac_of_peak * peak
    for k in np.argsort(bc)[::-1]:
        if bc[k] >= reward_zone_cm or not np.isfinite(sp[k]):
            continue
        if sp[k] < thr:
            return float(bc[min(k + 1, len(bc) - 1)])
    return float(bc[ok].min())


# --------------------------------------------------------------------------
# drinking bouts (consummatory readout, bout-level)
# --------------------------------------------------------------------------

def drinking_bouts(licks: pd.DataFrame, bouts: pd.DataFrame, model,
                   fps: float = FPS) -> pd.DataFrame:
    """One row per DRINKING bout (a chin bout spanning some trial's valve --
    same bouts `classify_licks` already tagged via the licks table's
    `drink_bout` column, trial index or -1). duration_s / n_licks /
    lick_rate_hz per bout -- the consummatory-side counterpart to the
    anticipatory spatial metrics: does DCZ change how much/how long the
    animal drinks once it's earned the reward, independent of anticipation?
    """
    dl = licks[licks.drink_bout >= 0]
    if not len(dl):
        return pd.DataFrame(columns=["bout_id", "trial", "start_frame", "end_frame",
                                     "duration_s", "n_licks", "lick_rate_hz"])
    rows = []
    for bid, g in dl.groupby("bout_id"):
        b = bouts.iloc[int(bid)]
        dur_s = float(frame_to_vr(b.end_frame, model, "generic") -
                      frame_to_vr(b.start_frame, model, "generic"))
        rows.append(dict(bout_id=int(bid), trial=int(g["drink_bout"].iloc[0]),
                         start_frame=float(b.start_frame), end_frame=float(b.end_frame),
                         duration_s=dur_s, n_licks=int(len(g)),
                         lick_rate_hz=float(len(g) / dur_s) if dur_s > 0 else np.nan))
    return pd.DataFrame(rows)


def speed_accel_vs_position(vr, trials: pd.DataFrame, cm_per_au: float,
                            binw_cm: float = 5.0, corridor_cm: float = 132.0,
                            grid_dt: float = 0.05, trial_mask=None) -> pd.DataFrame:
    """Mean running speed (cm/s) and acceleration (cm/s^2) per position bin."""
    pe = vr.position.elapsed_s.to_numpy()
    pl_cm = vr.position.location_au.to_numpy() * cm_per_au
    bins = np.arange(0, corridor_cm + binw_cm, binw_cm)
    bc = (bins[:-1] + bins[1:]) / 2

    keep = (set(trials.trial.astype(int)) if trial_mask is None
            else set(np.asarray(trials.trial.astype(int))[np.asarray(trial_mask)]))

    s_sum = np.zeros(len(bc)); a_sum = np.zeros(len(bc)); cnt = np.zeros(len(bc))
    for tr, t0, t1 in _approach_windows(trials):
        if tr not in keep or t1 - t0 < 0.3:
            continue
        # stop at the last logged position of this trial (black screen has no
        # data; interpolating past it, or across the teleport, gives spikes)
        seg = (pe >= t0) & (pe <= t1)
        if seg.sum() < 3:
            continue
        t1e = float(pe[seg][-1])
        tg = np.arange(t0, t1e, grid_dt)
        pos = np.interp(tg, pe, pl_cm)
        spd = np.gradient(pos, grid_dt)
        spd = np.clip(spd, 0, 150)                      # forward, plausible cm/s
        acc = np.clip(np.gradient(spd, grid_dt), -300, 300)
        idx = np.clip(np.digitize(pos, bins) - 1, 0, len(bc) - 1)
        np.add.at(s_sum, idx, spd); np.add.at(a_sum, idx, acc); np.add.at(cnt, idx, 1)
    return pd.DataFrame({"bin_cm": bc,
                         "speed_cm_s": np.where(cnt > 0, s_sum / np.maximum(cnt, 1), np.nan),
                         "accel_cm_s2": np.where(cnt > 0, a_sum / np.maximum(cnt, 1), np.nan)})


# --------------------------------------------------------------------------
# reward-aligned PSTH
# --------------------------------------------------------------------------

def reward_psth(licks: pd.DataFrame, trials: pd.DataFrame, model,
                window=(-4.0, 3.0), binw: float = 0.1) -> pd.DataFrame:
    edges = np.arange(window[0], window[1] + binw, binw)
    tc = (edges[:-1] + edges[1:]) / 2
    counts = np.zeros(len(tc)); n_tr = 0
    lick_t = frame_to_vr(licks.frame.to_numpy(), model, "generic")
    for row in trials.itertuples():
        if not np.isfinite(row.r_anchor_frame):
            continue
        rel = lick_t - frame_to_vr(row.r_anchor_frame, model, "generic")
        counts += np.histogram(rel[(rel >= window[0]) & (rel < window[1])], bins=edges)[0]
        n_tr += 1
    return pd.DataFrame({"t_from_reward_s": tc, "rate_hz": counts / max(n_tr, 1) / binw})


# --------------------------------------------------------------------------
# per-trial first approach lick
# --------------------------------------------------------------------------

def first_approach_lick(licks: pd.DataFrame, trials: pd.DataFrame,
                        cm_per_au: float) -> pd.DataFrame:
    ap = licks[licks.category.isin(["approaching_nonreward", "approaching_reward"])]
    rows = []
    for row in trials.itertuples():
        g = ap[ap.trial == int(row.trial)]
        if len(g):
            f = g.loc[g.frame.idxmin()]
            rows.append(dict(trial=int(row.trial),
                             first_lick_cm=float(f.position_au * cm_per_au),
                             first_lick_frame=int(f.frame)))
        else:
            rows.append(dict(trial=int(row.trial), first_lick_cm=np.nan, first_lick_frame=-1))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# per-session lick scatter data
# --------------------------------------------------------------------------

def lick_scatter_data(licks: pd.DataFrame, trials: pd.DataFrame, model,
                      cm_per_au: float):
    """(approach_df, iti_df) for the per-session raster.

    approach_df : one row per approaching_* lick   [trial, cm, category]
    iti_df      : one row per consummatory_* lick   [trial, t_from_valve_s, category]
    """
    ap = licks[licks.category.isin(["approaching_nonreward", "approaching_reward"])].copy()
    ap["cm"] = ap.position_au * cm_per_au
    approach_df = ap[["trial", "cm", "category"]].reset_index(drop=True)

    r_anchor = trials.set_index("trial")["r_anchor_frame"].to_dict()
    it = licks[licks.category.str.startswith("consummatory")].copy()
    lt = frame_to_vr(it.frame.to_numpy(), model, "generic")
    rvt = np.array([frame_to_vr(r_anchor.get(int(t), np.nan), model, "generic")
                    for t in it.trial])
    it["t_from_valve_s"] = lt - rvt
    iti_df = it[["trial", "t_from_valve_s", "category"]].reset_index(drop=True)
    return approach_df, iti_df


# --------------------------------------------------------------------------
# session metrics
# --------------------------------------------------------------------------

_CATS = ["approaching_nonreward", "approaching_reward",
         "consummatory_pre_reward", "consummatory_post_reward"]


def trial_durations_s(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial time from teleport (n_time_vr) to reward (r_time_vr) -- how
    long it takes the animal to complete one corridor traversal and reach the
    RZ. A simple engagement/efficiency readout, independent of licking."""
    d = (trials.r_time_vr - trials.n_time_vr).to_numpy()
    return np.where(d > 0, d, np.nan)   # NaN r_time_vr (no reward that trial) -> NaN


# --------------------------------------------------------------------------
# reviewer-requested analyses (2026-09-22): spatial-vs-temporal anticipation
# strategy, trial-level anticipation classification, black-screen running
# speed, and the trial-cluster bootstrap used in notebook 4 §9.
# --------------------------------------------------------------------------

POST_VALVE_S = 1.5   # approx. black-screen duration after the valve, before VR reappears


def trial_approach_speed_cms(vr, trials: pd.DataFrame, cm_per_au: float) -> np.ndarray:
    """Per-trial average running speed (cm/s) during the VR-visible approach
    only (n_time_vr -> this trial's last logged position, NOT the black
    screen/reward window) -- the natural trial-to-trial pace variation used
    to dissociate spatial vs. temporal anticipation strategies (see
    `spatial_temporal_onset_table`)."""
    pe = vr.position.elapsed_s.to_numpy()
    pl_cm = vr.position.location_au.to_numpy() * cm_per_au
    out = np.full(len(trials), np.nan)
    for i, row in enumerate(trials.itertuples()):
        t0 = row.n_time_vr
        t1 = row.r_time_vr
        if not np.isfinite(t1):
            continue
        seg = (pe >= t0) & (pe <= t1)
        if seg.sum() < 2:
            continue
        dur = pe[seg][-1] - t0
        dist = pl_cm[seg][-1] - pl_cm[seg][0]
        if dur > 0.3 and dist > 5:
            out[i] = dist / dur
    return out


def spatial_temporal_onset_table(licks: pd.DataFrame, trials: pd.DataFrame, vr,
                                 model, cm_per_au: float) -> pd.DataFrame:
    """Per-trial (first_lick_cm, first_lick_time_s, approach_speed_cms) --
    the data needed to dissociate a spatial-map strategy (lick onset fixed in
    POSITION regardless of speed) from an interval-timing strategy (onset
    fixed in TIME regardless of speed). `first_lick_time_s` is elapsed time
    from teleport (n_time_vr) to the first approach lick."""
    fl = first_approach_lick(licks, trials, cm_per_au)
    speed = trial_approach_speed_cms(vr, trials, cm_per_au)
    n_t = trials.set_index("trial")["n_time_vr"].to_dict()
    t_s = np.full(len(fl), np.nan)
    for i, row in enumerate(fl.itertuples()):
        if row.first_lick_frame >= 0:
            t0 = n_t.get(int(row.trial), np.nan)
            if np.isfinite(t0):
                t_s[i] = float(frame_to_vr(row.first_lick_frame, model, "generic")) - t0
    out = fl.copy()
    out["first_lick_time_s"] = t_s
    out["approach_speed_cms"] = speed
    return out


def spatial_vs_temporal_fit(onset_table: pd.DataFrame) -> dict:
    """Fits first_lick_cm ~ speed and first_lick_time_s ~ speed (simple OLS,
    plus the standardised/z-scored slope so the two fits are comparable
    despite different units). A near-zero slope + low R^2 for POSITION vs.
    speed means onset sits at a fixed position regardless of pace (spatial
    strategy); a near-zero slope for TIME vs. speed means onset sits at a
    fixed delay regardless of pace (temporal/interval strategy)."""
    d = onset_table.dropna(subset=["first_lick_cm", "first_lick_time_s", "approach_speed_cms"])
    out = dict(n=len(d), slope_cm_per_speed=np.nan, r2_cm=np.nan, z_slope_cm=np.nan,
              slope_time_per_speed=np.nan, r2_time=np.nan, z_slope_time=np.nan)
    if len(d) < 8:
        return out
    x = d.approach_speed_cms.to_numpy()

    def _fit(y):
        b, _ = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 and np.std(y) > 0 else np.nan
        xz = (x - x.mean()) / x.std() if x.std() > 0 else x - x.mean()
        yz = (y - y.mean()) / y.std() if y.std() > 0 else y - y.mean()
        bz = np.polyfit(xz, yz, 1)[0] if np.std(xz) > 0 else np.nan
        return float(b), float(r ** 2) if np.isfinite(r) else np.nan, float(bz)

    b_cm, r2_cm, bz_cm = _fit(d.first_lick_cm.to_numpy())
    b_t, r2_t, bz_t = _fit(d.first_lick_time_s.to_numpy())
    out.update(slope_cm_per_speed=b_cm, r2_cm=r2_cm, z_slope_cm=bz_cm,
              slope_time_per_speed=b_t, r2_time=r2_t, z_slope_time=bz_t)
    return out


def classify_trial_anticipation_strategy(licks: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """Per trial: which kind of anticipatory lick happened FIRST -- 'spatial'
    (an approach_* lick -- licking started while the corridor was still
    visible), 'transition_triggered' (the first anticipatory lick is
    consummatory_pre_reward -- licking only started after the black screen
    appeared), or 'none' (no anticipatory lick that trial)."""
    antic = licks[licks.category.isin(["approaching_nonreward", "approaching_reward",
                                       "consummatory_pre_reward"])]
    rows = []
    for t in trials.trial.astype(int):
        g = antic[antic.trial == t]
        if not len(g):
            rows.append(dict(trial=t, strategy="none", first_antic_frame=-1))
            continue
        f = g.loc[g.frame.idxmin()]
        strat = ("spatial" if f.category in ("approaching_nonreward", "approaching_reward")
                 else "transition_triggered")
        rows.append(dict(trial=t, strategy=strat, first_antic_frame=int(f.frame)))
    return pd.DataFrame(rows)


def frac_trials_with_category(licks: pd.DataFrame, trials: pd.DataFrame, category: str) -> float:
    """Fraction of trials with >= 1 lick of the given category."""
    have = set(licks.loc[licks.category == category, "trial"].unique())
    n = len(trials)
    return float(len(have) / n) if n else np.nan


def frac_trials_post_vr_onset_licking(licks: pd.DataFrame, trials: pd.DataFrame,
                                      post_valve_s: float = POST_VALVE_S, fps: float = FPS) -> float:
    """Fraction of trials whose drinking bout has a lick more than
    `post_valve_s` after the valve -- i.e. licking persisted past when VR
    should have reappeared. A consumption/motivation readout, independent of
    anticipation (the DCZ drinking-bout-duration finding suggests this
    should go UP under DCZ)."""
    r_anchor = trials.set_index("trial")["r_anchor_frame"].to_dict()
    persisted = n = 0
    for t in trials.trial.astype(int):
        rf = r_anchor.get(t, np.nan)
        if not np.isfinite(rf):
            continue
        n += 1
        g = licks[licks.drink_bout == t]
        if len(g) and (g.frame.to_numpy() > rf + post_valve_s * fps).any():
            persisted += 1
    return float(persisted / n) if n else np.nan


def black_screen_running_speed(tm, trials: pd.DataFrame, tm_align: dict,
                               epoch=(-1.5, 0.0)) -> float:
    """Average TM-log running speed (calibrated to cm/s via `tm_align`, from
    `tm_vr_speed_alignment`) during the pre-valve black screen (`epoch`
    relative to the valve, seconds). The TM log is a separate hardware
    stream sampled continuously regardless of VR state, so unlike the VR
    position log it should still have real samples through the dark period.
    NaN if there's no TM log or the calibration failed for this session."""
    if tm is None or not np.isfinite(tm_align.get("scale", np.nan)):
        return np.nan
    rvt = trials.r_time_vr.to_numpy()
    rvt = rvt[np.isfinite(rvt)]
    if not len(rvt):
        return np.nan
    lag = tm_align.get("lag_s", 0.0)
    t_mid = rvt + (epoch[0] + epoch[1]) / 2.0
    raw = tm.speed_at(t_mid + lag)
    cms = tm_align["scale"] * raw + tm_align["intercept"]
    return float(np.nanmean(np.clip(cms, 0, 150)))


def trial_cluster_bootstrap(per_trial_values: list, n_draw: int, statistic,
                            n_boot: int = 2000, rng=None) -> np.ndarray:
    """Cluster (trial-level) bootstrap: `per_trial_values` is a list of
    arrays, one per trial (e.g. that trial's approach-lick cm positions, or
    a length-1 array for a single per-trial scalar like running speed). Each
    draw resamples `n_draw` TRIALS with replacement, pools their values, and
    applies `statistic` to the pooled array -- respects trial-level
    clustering (licks within a trial aren't independent observations)
    instead of resampling individual licks directly."""
    rng = rng or np.random.default_rng()
    per_trial_values = [np.asarray(v) for v in per_trial_values if len(v)]
    if len(per_trial_values) < 5:
        return np.full(n_boot, np.nan)
    n = len(per_trial_values)
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=max(n_draw, 1))
        pooled = np.concatenate([per_trial_values[i] for i in idx])
        out[b] = statistic(pooled) if len(pooled) else np.nan
    return out


def bootstrap_pvalue(observed: float, null_dist: np.ndarray) -> float:
    """Two-tailed empirical p-value: fraction of the null distribution at
    least as far from the null median as `observed` is."""
    null_dist = null_dist[np.isfinite(null_dist)]
    if len(null_dist) < 5 or not np.isfinite(observed):
        return np.nan
    med = np.median(null_dist)
    return float(np.mean(np.abs(null_dist - med) >= abs(observed - med)))


def session_metrics(curve: pd.DataFrame, psth: pd.DataFrame, licks: pd.DataFrame,
                    trials: pd.DataFrame, model, cm_per_au: float,
                    reward_zone_cm: float, first_licks: pd.DataFrame,
                    antic_cm: float = 30.0, neutral_cm=(20.0, 80.0),
                    corridor_cap_cm: float = None) -> dict:
    if corridor_cap_cm is None:
        corridor_cap_cm = reward_zone_cm + 30.0    # matches lick_rate_vs_position's default
                                                     # binning cap -- excludes the rare
                                                     # position-log excursion/glitch (300+ cm)
    trial_dur = trial_durations_s(trials)
    bc = curve.bin_cm.to_numpy(); rate = curve.rate_hz.to_numpy()
    rz_lo = reward_zone_cm - antic_cm
    rz_zone = (bc >= rz_lo) & (bc < reward_zone_cm)
    neu_zone = (bc >= neutral_cm[0]) & (bc < neutral_cm[1])

    rz_rate = float(np.nanmean(rate[rz_zone])) if rz_zone.any() else np.nan
    neu_rate = float(np.nanmean(rate[neu_zone])) if neu_zone.any() else np.nan
    baseline = float(np.nanmedian(rate[neu_zone])) if neu_zone.any() else np.nan

    # anticipatory onset: scanning from the reward end backwards, first bin
    # (moving away) that drops below 2x the neutral baseline
    onset = np.nan
    thr = 2 * baseline if np.isfinite(baseline) else np.nan
    if np.isfinite(thr):
        for k in np.argsort(bc)[::-1]:
            if bc[k] >= reward_zone_cm:
                continue
            if np.isfinite(rate[k]) and rate[k] < thr:
                onset = float(bc[min(k + 1, len(bc) - 1)])
                break
        if np.isnan(onset):
            onset = float(bc.min())
        onset = min(onset, float(reward_zone_cm))       # cannot be past the RZ

    ap_cm = (licks.loc[licks.category.isin(["approaching_nonreward", "approaching_reward"]),
                       "position_au"].to_numpy() * cm_per_au)
    ap_cm = ap_cm[np.isfinite(ap_cm) & (ap_cm <= corridor_cap_cm)]
    com = float(np.mean(ap_cm)) if len(ap_cm) else np.nan
    median_cm = float(np.median(ap_cm)) if len(ap_cm) else np.nan
    spread = float(np.std(ap_cm)) if len(ap_cm) else np.nan
    frac_rz = float(np.mean(ap_cm >= rz_lo)) if len(ap_cm) else np.nan

    per_trial = licks.trial.to_numpy(); cat = licks.category.to_numpy()
    counts = {c: np.array([np.sum((per_trial == t) & (cat == c)) for t in trials.trial])
              for c in _CATS}

    # latency: black-screen onset (~valve - 1.5 s) -> first consummatory lick
    lat = []
    lt = frame_to_vr(licks.frame.to_numpy(), model, "generic")
    for row in trials.itertuples():
        if not np.isfinite(row.r_anchor_frame):
            continue
        rvt = frame_to_vr(row.r_anchor_frame, model, "generic")
        m = (licks.trial.to_numpy() == int(row.trial)) & \
            np.isin(cat, ["consummatory_pre_reward", "consummatory_post_reward"])
        if m.any():
            lat.append(float(np.min(lt[m]) - (rvt - 1.5)))
    first_consum_latency = float(np.nanmedian(lat)) if lat else np.nan

    pr = psth.rate_hz.to_numpy(); pt = psth.t_from_reward_s.to_numpy()
    far = pt < pt.min() + 1.0
    psth_base = float(np.nanmean(pr[far])) if far.any() else np.nan
    lead = np.nan
    if np.isfinite(psth_base):
        above = (pt < 0) & (pr > 2 * psth_base)
        if above.any():
            lead = float(-pt[above][0])

    # time-based (temporal) anticipatory index -- the PSTH counterpart to the
    # position-based `anticipatory_index` above. That one is approach-licks
    # only (occupancy-normalised by position, so it structurally can't
    # include black-screen licks -- there's no position logged there to bin
    # them into). This uses the reward-aligned PSTH instead: lick rate in the
    # pre-valve black-screen window (-1.5..0 s, matches the black-screen
    # span marked throughout the notebooks) vs the same far-from-reward
    # baseline as `psth_anticipatory_lead_s`, same ratio structure as the
    # spatial index so the two are directly comparable.
    prevalve = (pt >= -1.5) & (pt < 0.0)
    psth_prevalve_hz = float(np.nanmean(pr[prevalve])) if prevalve.any() else np.nan
    psth_anticipatory_index = (
        float((psth_prevalve_hz - psth_base) / (psth_prevalve_hz + psth_base))
        if np.isfinite(psth_prevalve_hz + psth_base) and (psth_prevalve_hz + psth_base) > 0
        else np.nan)

    # reviewer-requested (2026-09-22): fraction of trials with any pre-valve
    # dark-period lick (#4), fraction with licking persisting past when VR
    # should reappear (#6), and the spatial/transition-triggered/none trial
    # classification (#5) -- all per-trial, so they thread through
    # `rolling_metrics` automatically like everything else here.
    frac_prevalve = frac_trials_with_category(licks, trials, "consummatory_pre_reward")
    frac_post_vr = frac_trials_post_vr_onset_licking(licks, trials)
    strat = classify_trial_anticipation_strategy(licks, trials)
    n_strat = len(strat) if len(strat) else 1
    frac_spatial = float((strat.strategy == "spatial").sum() / n_strat)
    frac_transition = float((strat.strategy == "transition_triggered").sum() / n_strat)
    frac_no_antic = float((strat.strategy == "none").sum() / n_strat)

    return dict(
        n_trials=int(np.isfinite(trials.r_time_vr).sum()),
        n_licks=int(len(licks)),
        **{f"n_{c}": int(np.sum(cat == c)) for c in _CATS},
        **{f"{c}_per_trial": float(counts[c].mean()) for c in _CATS},
        rz_lick_rate_hz=rz_rate,
        neutral_lick_rate_hz=neu_rate,
        anticipatory_ratio=float(rz_rate / max(neu_rate, 0.05)),   # floored denom
        anticipatory_index=float((rz_rate - neu_rate) / (rz_rate + neu_rate))
        if np.isfinite(rz_rate + neu_rate) and (rz_rate + neu_rate) > 0 else np.nan,
        anticipatory_onset_cm=onset,
        dist_onset_to_reward_cm=float(reward_zone_cm - onset) if np.isfinite(onset) else np.nan,
        lick_com_cm=com,
        lick_median_cm=median_cm,
        lick_spread_cm=spread,
        frac_approach_licks_in_rz=frac_rz,
        first_lick_cm_median=float(np.nanmedian(first_licks.first_lick_cm)),
        first_lick_cm_sd=float(np.nanstd(first_licks.first_lick_cm)),
        trials_with_approach_lick=int(np.sum(counts["approaching_nonreward"] +
                                             counts["approaching_reward"] >= 1)),
        first_consummatory_lick_latency_s=first_consum_latency,
        psth_peak_hz=float(np.nanmax(pr)),
        psth_baseline_hz=psth_base,
        psth_anticipatory_lead_s=lead,
        psth_prevalve_hz=psth_prevalve_hz,
        psth_anticipatory_index=psth_anticipatory_index,
        trial_duration_s_median=float(np.nanmedian(trial_dur)) if np.isfinite(trial_dur).any() else np.nan,
        trial_duration_s_sd=float(np.nanstd(trial_dur)) if np.isfinite(trial_dur).any() else np.nan,
        frac_trials_with_prevalve_lick=frac_prevalve,
        frac_trials_post_vr_onset_licking=frac_post_vr,
        frac_trials_spatial_strategy=frac_spatial,
        frac_trials_transition_triggered=frac_transition,
        frac_trials_no_anticipation=frac_no_antic,
    )


# --------------------------------------------------------------------------
# rolling within-session trajectory
# --------------------------------------------------------------------------

def rolling_metrics(licks: pd.DataFrame, vr, trials: pd.DataFrame, model,
                    cm_per_au: float, reward_zone_cm: float,
                    window: int = 20, step: int = 5, antic_cm: float = 30.0,
                    neutral_cm=(20.0, 80.0), freeze_thresh_s: float = 2.0) -> pd.DataFrame:
    """The key session metrics recomputed over a sliding window of trials."""
    tnums = trials.trial.astype(int).to_numpy()
    rows = []
    for lo in range(0, max(1, len(tnums) - window + 1), step):
        sub = tnums[lo:lo + window]
        mask = np.isin(tnums, sub)
        if mask.sum() < 5:
            continue
        tsub = trials[mask]
        curve = lick_rate_vs_position(licks, vr, tsub, cm_per_au, freeze_thresh_s=freeze_thresh_s)
        fl = first_approach_lick(licks, tsub, cm_per_au)
        psth = reward_psth(licks, tsub, model)
        m = session_metrics(curve, psth, licks[licks.trial.isin(set(sub))], tsub,
                            model, cm_per_au, reward_zone_cm, fl,
                            antic_cm=antic_cm, neutral_cm=neutral_cm)
        # locomotor readout per window -- lets the "does DCZ's motor effect
        # build over the session" question use the same rolling machinery as
        # the licking metrics, independent of licking
        spd = speed_accel_vs_position(vr, tsub, cm_per_au)
        _, vs = vr_speed_trace(vr, cm_per_au, tsub, freeze_thresh_s=freeze_thresh_s)
        m["run_speed_vr_cms"] = float(np.mean(vs)) if len(vs) else np.nan
        m["decel_onset_cm"] = speed_decel_onset(spd, reward_zone_cm)
        m["trial_lo"] = int(sub[0]); m["trial_hi"] = int(sub[-1])
        m["trial_mid"] = float(np.mean(sub))
        rows.append(m)
    return pd.DataFrame(rows)
