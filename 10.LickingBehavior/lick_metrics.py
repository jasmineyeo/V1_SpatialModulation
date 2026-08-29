"""
lick_metrics.py
===============

Per-session licking metrics for the "does anticipatory licking narrow onto
the reward zone with learning?" question.

    classify_licks(licks, trials, bouts)      -> licks + 'category' column
    lick_rate_vs_position(licks, vr, trials)  -> occupancy-normalised curve
    speed_vs_position(vr, trials)             -> mean running speed per bin
    reward_psth(licks, trials, model)         -> lick rate vs time-from-reward
    session_metrics(curve, psth, licks, ...)  -> one dict / row

Lick categories
---------------
carryover     : the lick's chin bout started before this trial's teleport
                (still drinking from the previous reward) -> excluded
approach      : has a valid VR position (the animal is running the corridor)
consummatory  : no position (at the reward zone / in the ITI) -> time-aligned

The **lick-rate-vs-position curve** uses `approach` licks, normalised by
dwell time per position bin (slowing near the reward would otherwise
inflate the count there). The **reward PSTH** uses all non-carryover licks.

JSY / V1_SpatialModulation - 10.LickingBehavior
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lick_sync import frame_to_vr, FPS


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------

def classify_licks(licks: pd.DataFrame, trials: pd.DataFrame,
                   bout_start_frame: np.ndarray) -> pd.DataFrame:
    """Add a `category` column: carryover / approach / consummatory."""
    out = licks.copy()
    n_anchor = trials.set_index("trial")["n_anchor_frame"]

    cat = np.empty(len(out), dtype=object)
    for i, row in enumerate(out.itertuples()):
        bid = int(row.bout_id)
        tri = int(row.trial)
        teleport_f = n_anchor.get(tri, np.inf)
        if bid >= 0 and bout_start_frame[bid] < teleport_f:
            cat[i] = "carryover"
        elif np.isfinite(row.position_au):
            cat[i] = "approach"
        else:
            cat[i] = "consummatory"
    out["category"] = cat
    return out


# --------------------------------------------------------------------------
# lick rate vs position (occupancy-normalised)
# --------------------------------------------------------------------------

def _approach_windows(trials: pd.DataFrame):
    """(t0, t1) VR-elapsed times of each trial's run: teleport -> reward."""
    for row in trials.itertuples():
        if np.isfinite(row.r_time_vr):
            yield row.trial, float(row.n_time_vr), float(row.r_time_vr)


def lick_rate_vs_position(licks: pd.DataFrame, vr, trials: pd.DataFrame,
                          binw: float = 15.0, pos_max: float = 405.0,
                          grid_dt: float = 0.01, trial_mask=None) -> pd.DataFrame:
    """Occupancy-normalised lick rate per position bin, over the approach.

    trial_mask : optional bool array/Series over `trials.trial` to restrict
                 (e.g. first third of the session).
    """
    pe = vr.position.elapsed_s.to_numpy()
    pl = vr.position.location_au.to_numpy()
    bins = np.arange(0, pos_max + binw, binw)
    bc = (bins[:-1] + bins[1:]) / 2

    if trial_mask is None:
        keep_trials = set(trials.trial)
    else:
        keep_trials = set(trials.trial[np.asarray(trial_mask)])

    dwell = np.zeros(len(bc))
    for tr, t0, t1 in _approach_windows(trials):
        if tr not in keep_trials or t1 <= t0:
            continue
        tg = np.arange(t0, t1, grid_dt)
        pos = np.interp(tg, pe, pl)
        dwell += np.histogram(pos, bins=bins)[0] * grid_dt

    ap = licks[(licks.category == "approach") & licks.trial.isin(keep_trials)]
    n_licks = np.histogram(ap.position_au.to_numpy(), bins=bins)[0]

    rate = np.where(dwell > 0.5, n_licks / dwell, np.nan)
    return pd.DataFrame({"bin_center": bc, "dwell_s": dwell,
                         "n_licks": n_licks, "rate_hz": rate})


def speed_vs_position(vr, trials: pd.DataFrame, binw: float = 15.0,
                      pos_max: float = 405.0, grid_dt: float = 0.05,
                      trial_mask=None) -> pd.DataFrame:
    """Mean running speed (au/s, from VR position) per position bin."""
    pe = vr.position.elapsed_s.to_numpy()
    pl = vr.position.location_au.to_numpy()
    bins = np.arange(0, pos_max + binw, binw)
    bc = (bins[:-1] + bins[1:]) / 2

    keep = (set(trials.trial) if trial_mask is None
            else set(trials.trial[np.asarray(trial_mask)]))

    sums = np.zeros(len(bc))
    counts = np.zeros(len(bc))
    for tr, t0, t1 in _approach_windows(trials):
        if tr not in keep or t1 - t0 < 0.2:
            continue
        tg = np.arange(t0, t1, grid_dt)
        pos = np.interp(tg, pe, pl)
        spd = np.abs(np.gradient(pos, grid_dt))
        idx = np.clip(np.digitize(pos, bins) - 1, 0, len(bc) - 1)
        np.add.at(sums, idx, spd)
        np.add.at(counts, idx, 1)
    mean_speed = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return pd.DataFrame({"bin_center": bc, "mean_speed_au_s": mean_speed})


# --------------------------------------------------------------------------
# reward-aligned PSTH
# --------------------------------------------------------------------------

def reward_psth(licks: pd.DataFrame, trials: pd.DataFrame, model,
                window=(-4.0, 3.0), binw: float = 0.1) -> pd.DataFrame:
    """Lick rate (Hz) vs time from the reward valve, all non-carryover licks."""
    edges = np.arange(window[0], window[1] + binw, binw)
    tc = (edges[:-1] + edges[1:]) / 2
    counts = np.zeros(len(tc))
    n_tr = 0
    lick_t = frame_to_vr(licks.frame.to_numpy(), model, "generic")
    cat = licks.category.to_numpy()
    for row in trials.itertuples():
        if not np.isfinite(row.r_anchor_frame):
            continue
        r_vr = frame_to_vr(row.r_anchor_frame, model, "generic")
        rel = lick_t - r_vr
        m = (cat != "carryover") & (rel >= window[0]) & (rel < window[1])
        counts += np.histogram(rel[m], bins=edges)[0]
        n_tr += 1
    return pd.DataFrame({"t_from_reward_s": tc,
                         "rate_hz": counts / max(n_tr, 1) / binw})


# --------------------------------------------------------------------------
# per-trial first (approach) lick
# --------------------------------------------------------------------------

def first_approach_lick(licks: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ap = licks[licks.category == "approach"]
    for row in trials.itertuples():
        g = ap[ap.trial == row.trial]
        if len(g):
            j = g.position_au.idxmin() if False else g.frame.idxmin()
            first = g.loc[j]
            rows.append(dict(trial=row.trial,
                             first_lick_pos_au=float(first.position_au),
                             first_lick_frame=int(first.frame)))
        else:
            rows.append(dict(trial=row.trial, first_lick_pos_au=np.nan,
                             first_lick_frame=-1))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# session metrics
# --------------------------------------------------------------------------

def session_metrics(curve: pd.DataFrame, psth: pd.DataFrame,
                    licks: pd.DataFrame, trials: pd.DataFrame,
                    reward_zone_au: float, first_licks: pd.DataFrame,
                    antic_span: float = 60.0, neutral: tuple = (60.0, None),
                    neutral_pad: float = 120.0) -> dict:
    """One row of metrics for the learning trajectory."""
    bc = curve.bin_center.to_numpy()
    rate = curve.rate_hz.to_numpy()

    rz_lo = reward_zone_au - antic_span
    neu_hi = (reward_zone_au - neutral_pad) if neutral[1] is None else neutral[1]
    rz_zone = (bc >= rz_lo) & (bc < reward_zone_au)
    neu_zone = (bc >= neutral[0]) & (bc < neu_hi)

    rz_rate = np.nanmean(rate[rz_zone]) if rz_zone.any() else np.nan
    neu_rate = np.nanmean(rate[neu_zone]) if neu_zone.any() else np.nan
    baseline = np.nanmedian(rate[neu_zone]) if neu_zone.any() else np.nan

    # anticipatory onset: scanning from the reward zone backward, the first
    # bin (going away from reward) whose rate drops below 2x baseline
    onset = np.nan
    thr = 2 * baseline if np.isfinite(baseline) else np.nan
    order = np.argsort(bc)[::-1]           # reward end -> start
    below = False
    for k in order:
        if bc[k] >= reward_zone_au:
            continue
        if np.isfinite(rate[k]) and rate[k] < thr:
            onset = bc[k + 1] if (k + 1) < len(bc) else bc[k]
            below = True
            break
    if not below:
        onset = bc[order[-1]]              # licks above threshold all the way

    # center of mass + spread of approach-lick positions
    ap_pos = licks.loc[licks.category == "approach", "position_au"].to_numpy()
    ap_pos = ap_pos[np.isfinite(ap_pos)]
    com = float(np.mean(ap_pos)) if len(ap_pos) else np.nan
    spread = float(np.std(ap_pos)) if len(ap_pos) else np.nan
    frac_in_rz = float(np.mean(ap_pos >= rz_lo)) if len(ap_pos) else np.nan

    # per-trial lick counts
    cat = licks.category.to_numpy()
    per_trial = licks.trial.to_numpy()
    n_ap = np.array([np.sum((per_trial == t) & (cat == "approach")) for t in trials.trial])
    n_co = np.array([np.sum((per_trial == t) & (cat == "consummatory")) for t in trials.trial])

    # PSTH anticipatory lead: how far before the valve the rate first (going
    # forward in time) stays above 2x the far-pre-reward baseline
    pr = psth.rate_hz.to_numpy()
    pt = psth.t_from_reward_s.to_numpy()
    pk = float(np.nanmax(pr))
    far = (pt >= pt.min()) & (pt < pt.min() + 1.0)      # first ~1 s of the window
    psth_baseline = float(np.nanmean(pr[far])) if far.any() else np.nan
    lead = np.nan
    thr_p = 2 * psth_baseline if np.isfinite(psth_baseline) else np.nan
    pre = pt < 0
    if pre.any() and np.isfinite(thr_p):
        above = pre & (pr > thr_p)
        if above.any():
            # first sustained crossing: earliest t<0 after which it stays above
            idx = np.where(pre)[0]
            for j in idx:
                if pr[j] > thr_p and np.all(pr[j:idx[-1] + 1] > 0.5 * thr_p):
                    lead = float(-pt[j])
                    break
            if np.isnan(lead):
                lead = float(-pt[above][0])

    return dict(
        n_trials=int(np.isfinite(trials.r_time_vr).sum()),
        n_licks=int(len(licks)),
        n_approach_licks=int(np.sum(cat == "approach")),
        n_consummatory_licks=int(np.sum(cat == "consummatory")),
        n_carryover_licks=int(np.sum(cat == "carryover")),
        rz_lick_rate_hz=float(rz_rate),
        neutral_lick_rate_hz=float(neu_rate),
        anticipatory_ratio=float(rz_rate / neu_rate) if neu_rate else np.nan,
        anticipatory_onset_au=float(onset),
        dist_onset_to_reward_au=float(reward_zone_au - onset),
        lick_com_au=com,
        lick_spread_au=spread,
        frac_approach_licks_in_rz=frac_in_rz,
        approach_licks_per_trial=float(n_ap.mean()),
        consummatory_licks_per_trial=float(n_co.mean()),
        first_lick_pos_median_au=float(np.nanmedian(first_licks.first_lick_pos_au)),
        first_lick_pos_sd_au=float(np.nanstd(first_licks.first_lick_pos_au)),
        trials_with_approach_lick=int(np.sum(n_ap >= 1)),
        psth_peak_hz=pk,
        psth_baseline_hz=psth_baseline,
        psth_anticipatory_lead_s=lead,
    )
