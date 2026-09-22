"""
lick_figures_per_animal.py
===========================

Corrected, per-animal figure set addressing the reviewer critique of the
`3c_*` figures in `4.SessionComparison.ipynb`. Consolidated from
`lick_figures_per_animal.ipynb` (Phase 3).

Reuses the existing per-session outputs (`{rec}_lickproc.h5`,
`{rec}_lickmetrics.h5`) and existing `lick_io.py`/`lick_sync.py`/
`lick_metrics.py` functions wherever the logic already existed correctly;
only the pieces the critique flagged as wrong/undefined are new.

Method fixes vs. the original `3c_*` figures
---------------------------------------------
M1  baseline z-score reference = B5-B8 per-day values (not B7+B8, not all
    of B1-B8) -- `baseline_zscore_reference`
M2  hierarchical (day -> trial) bootstrap over B5-B8, trial-clustered,
    10000 iterations, with a leave-one-day-out calibration check reported
    before any p-value is plotted -- `hierarchical_day_trial_bootstrap`,
    `bootstrap_calibration_check`
M3  "spatial" strategy restricted to a first-lick position within
    `SPATIAL_WINDOW_CM` of the corridor end, computed alongside the
    original (unrestricted) definition -- `classify_strategy_restricted`
M4  drinking bouts split into pre-/post-valve parts; persistence measured
    against the trial's ACTUAL next-trial VR-on frame (`n_anchor_frame`),
    not a nominal 1.5 s -- `split_bout_pre_post`,
    `frac_trials_post_vr_onset_exact`
M5  ONE VR running-speed definition used everywhere (`per_trial_vr_speed_cms`,
    freeze-excluded instantaneous mean -- matches the session-level
    `run_speed_vr_cms` used throughout the rest of the pipeline);
    `trial_approach_speed_cms` (net displacement / elapsed time, freeze NOT
    excluded) is kept ONLY as the per-trial pace covariate for the spatial-
    vs-temporal regression, never reported as "running speed."
    `speed_matched_lick_spread` restricts a spread comparison to the
    overlapping per-trial speed range of the two sessions being compared.
M6  deceleration onset reported/labelled as distance BEFORE the corridor
    end (`reward_zone_cm - decel_onset_cm`; higher = earlier deceleration =
    more anticipatory), never the raw position, to remove sign ambiguity.
M7  approach licks within `CARRYOVER_CM` of the corridor start excluded as
    likely previous-trial carryover -- on top of, not instead of,
    `classify_licks`'s existing time-based `carryover_frames` fix.

JSY / V1_SpatialModulation - 09.LickingBehavior
"""
from __future__ import annotations

import os
import sys
import glob

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
from matplotlib import rcParams

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

from lick_sync import FPS
from lick_io import read_vrlog, read_tmlog
from lick_metrics import (_freeze_intervals, _in_intervals,
                          classify_trial_anticipation_strategy,
                          trial_cluster_bootstrap, bootstrap_pvalue)

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

DATASET_ROOT = r"F:\dlc\lick-detector_v0-JSY-2026-06-16\videos"
BASELINE_DAYS = [f"B{i}" for i in range(1, 9)]
ZSCORE_DAYS = ["B5", "B6", "B7", "B8"]          # M1
CONDITIONS = ["Saline", "DCZ100", "DCZ200"]
SESSIONS = BASELINE_DAYS + CONDITIONS
ANIMALS = ["JSY083", "JSY084"]

CARRYOVER_CM = 10.0        # M7
SPATIAL_WINDOW_CM = 30.0   # M3
IN_ZONE_WINDOW_CM = 30.0   # matches ANTIC_CM elsewhere / reward zone 104-134cm

_BASELINE_GREYS = _cm.Greys(np.linspace(0.35, 0.85, len(BASELINE_DAYS)))
SESSION_COLOR = {d: _BASELINE_GREYS[i] for i, d in enumerate(BASELINE_DAYS)}
SESSION_COLOR.update({"Saline": "tab:green", "DCZ100": "tab:orange", "DCZ200": "tab:red"})


def set_figure_style():
    rcParams["legend.fontsize"] = 20
    rcParams["axes.labelsize"] = 20
    rcParams["axes.titlesize"] = 25
    rcParams["xtick.labelsize"] = 20
    rcParams["ytick.labelsize"] = 20


def _finalize(fig):
    """Rotate x-tick labels on every axis (fontsize 20 + up to 7-8 session
    labels per panel overlap badly unrotated) and tighten layout. Called at
    the end of every fig_* function instead of a bare `return fig`."""
    for ax in fig.axes:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(40)
            lbl.set_ha("right")
    fig.tight_layout()
    return fig


def save_fig(fig, animal, name, output_root):
    d = os.path.join(output_root, animal)
    os.makedirs(d, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(d, f"{name}.{ext}"), dpi=300, bbox_inches="tight")


def label_n1(ax, x, y, **kw):
    """Mark a drug-session point/panel with its actual n (always 1 session)."""
    ax.annotate("n = 1 session", (x, y), fontsize=11, color="dimgray",
               xytext=(4, 4), textcoords="offset points", **kw)


# --------------------------------------------------------------------------
# session loading -- reuses lickproc.h5 (n_anchor_frame) + lickmetrics.h5
# (licks_full / trials_full / onset_table / trial_strategy), no
# re-derivation of classification from scratch
# --------------------------------------------------------------------------

def _load_h5_group(f, name):
    if name not in f:
        return None
    g = f[name]
    out = {}
    for k in g:
        v = g[k][:]
        if v.dtype.kind in ("S", "O"):
            v = np.array([x.decode() if isinstance(x, bytes) else x for x in v])
        out[k] = v
    return pd.DataFrame(out)


def load_session_data(rec: str, animal: str) -> dict:
    """Everything needed for the M1-M7 analyses, for one recording."""
    proc_path = glob.glob(os.path.join(DATASET_ROOT, animal, rec, f"{rec}_lickproc.h5"))[0]
    met_path = glob.glob(os.path.join(DATASET_ROOT, animal, rec, f"{rec}_lickmetrics.h5"))[0]

    with h5py.File(proc_path, "r") as f:
        trials_raw = pd.DataFrame({k: f["trials"][k][:] for k in f["trials"]})
        vrlog_path = f.attrs["vrlog_path"]
        tmlog_path = str(f.attrs.get("tmlog_path", ""))
    trials_raw["trial"] = trials_raw["trial"].astype(int)

    with h5py.File(met_path, "r") as f:
        licks_full = _load_h5_group(f, "licks_full")
        trials_full = _load_h5_group(f, "trials_full")
        onset_table = _load_h5_group(f, "onset_table")
        trial_strategy = _load_h5_group(f, "trial_strategy")
        attrs = dict(f.attrs)

    licks_full["trial"] = licks_full["trial"].astype(int)
    licks_full["drink_bout"] = licks_full["drink_bout"].astype(int)
    trials_full["trial"] = trials_full["trial"].astype(int)

    # n_anchor_frame isn't saved in trials_full -- pull it (and the NEXT
    # trial's, for the exact VR-on-timestamp persistence metric, M4) from
    # the original lickproc trials table
    n_anchor = trials_raw.set_index("trial")["n_anchor_frame"].to_dict()
    trials_full["n_anchor_frame"] = trials_full["trial"].map(n_anchor)
    trials_full["next_n_anchor_frame"] = trials_full["trial"].map(
        {t: n_anchor.get(t + 1, np.nan) for t in trials_full.trial})

    vr = read_vrlog(vrlog_path)
    tm = read_tmlog(tmlog_path) if tmlog_path else None

    sess = dict(recording=rec, animal=animal, vr=vr, tm=tm,
               licks_full=licks_full, trials_full=trials_full,
               onset_table=onset_table, trial_strategy=trial_strategy,
               cm_per_au=float(attrs["cm_per_au"]), reward_zone_cm=float(attrs["reward_zone_cm"]),
               n_trials=int(attrs["n_trials"]))
    sess["trials_full"]["vr_speed_cms"] = per_trial_vr_speed_cms(vr, trials_full, sess["cm_per_au"])
    return sess


def load_all_sessions(M: pd.DataFrame) -> dict:
    """{animal: {session_label: session_dict}} for every session in M."""
    out = {a: {} for a in ANIMALS}
    for _, row in M.iterrows():
        out[row.animal][row.session_label] = load_session_data(row.recording, row.animal)
    return out


# --------------------------------------------------------------------------
# M1 -- baseline z-score reference (B5-B8)
# --------------------------------------------------------------------------

def baseline_zscore_reference(M: pd.DataFrame, animal: str, metric: str,
                              days=tuple(ZSCORE_DAYS)) -> dict:
    gm = M[M.animal == animal].set_index("session_label")
    have = [d for d in days if d in gm.index]
    vals = gm.loc[have, metric]
    return dict(days=have, mean=float(vals.mean()), sd=float(vals.std()),
               values={d: float(vals[d]) for d in have})


def zscore_vs_baseline(M: pd.DataFrame, animal: str, session: str, metric: str,
                       days=tuple(ZSCORE_DAYS)) -> float:
    ref = baseline_zscore_reference(M, animal, metric, days)
    gm = M[M.animal == animal].set_index("session_label")
    if session not in gm.index or not ref["sd"]:
        return np.nan
    return float((gm.loc[session, metric] - ref["mean"]) / ref["sd"])


# --------------------------------------------------------------------------
# M3 -- restricted "spatial" strategy classification
# --------------------------------------------------------------------------

def classify_strategy_restricted(licks_full: pd.DataFrame, trials_full: pd.DataFrame,
                                 reward_zone_cm: float, cm_per_au: float,
                                 window_cm: float = SPATIAL_WINDOW_CM) -> pd.DataFrame:
    """Same idea as `classify_trial_anticipation_strategy`, but 'spatial'
    only counts if the qualifying approach lick sits within `window_cm` of
    the corridor end -- a scattered lick at 20 cm isn't anticipation of
    THIS reward. consummatory_pre_reward licks always qualify as
    transition-triggered candidates. Classification uses whichever
    QUALIFYING candidate (spatial-in-window OR consummatory_pre_reward)
    came first in time; trials with neither are 'none'."""
    lf = licks_full.copy()
    lf["cm"] = lf.position_au * cm_per_au
    is_qual_spatial = (lf.category.isin(["approaching_nonreward", "approaching_reward"])
                       & (lf.cm >= reward_zone_cm - window_cm))
    is_transition = lf.category == "consummatory_pre_reward"
    cand = lf[is_qual_spatial | is_transition]
    by_trial = {}
    for t, g in cand.groupby("trial"):
        f = g.loc[g.frame.idxmin()]
        strat = ("spatial" if f.category in ("approaching_nonreward", "approaching_reward")
                 else "transition_triggered")
        by_trial[int(t)] = (strat, int(f.frame))
    rows = []
    for t in trials_full.trial.astype(int):
        strat, fr = by_trial.get(t, ("none", -1))
        rows.append(dict(trial=t, strategy=strat, first_antic_frame=fr))
    return pd.DataFrame(rows)


def strategy_proportions(strategy_df: pd.DataFrame) -> dict:
    n = len(strategy_df) if len(strategy_df) else 1
    vc = strategy_df.strategy.value_counts()
    return dict(spatial=float(vc.get("spatial", 0) / n),
               transition_triggered=float(vc.get("transition_triggered", 0) / n),
               none=float(vc.get("none", 0) / n), n_trials=len(strategy_df))


# --------------------------------------------------------------------------
# M4 -- bout pre/post-valve split, exact VR-on-timestamp persistence
# --------------------------------------------------------------------------

def split_bout_pre_post(licks_full: pd.DataFrame, trials_full: pd.DataFrame,
                        fps: float = FPS) -> pd.DataFrame:
    """Per-trial pre-valve / post-valve lick counts and duration for that
    trial's drinking bout, split exactly at the valve frame
    (`r_anchor_frame`) -- bout duration conflates the two if not split
    (a long PRE-valve bout and a long POST-valve bout both inflate the same
    number)."""
    r_anchor = trials_full.set_index("trial")["r_anchor_frame"].to_dict()
    rows = []
    for t in trials_full.trial:
        rf = r_anchor.get(t, np.nan)
        g = licks_full[licks_full.drink_bout == t]
        if not np.isfinite(rf) or not len(g):
            rows.append(dict(trial=t, n_licks_pre=0, n_licks_post=0,
                             pre_duration_s=np.nan, post_duration_s=np.nan))
            continue
        pre, post = g[g.frame <= rf], g[g.frame > rf]
        pre_dur = (float(pre.frame.max() - pre.frame.min()) / fps) if len(pre) > 1 else 0.0
        post_dur = (float(post.frame.max() - post.frame.min()) / fps) if len(post) > 1 else 0.0
        rows.append(dict(trial=t, n_licks_pre=len(pre), n_licks_post=len(post),
                         pre_duration_s=pre_dur, post_duration_s=post_dur))
    return pd.DataFrame(rows)


def frac_trials_post_vr_onset_exact(licks_full: pd.DataFrame, trials_full: pd.DataFrame) -> float:
    """Fraction of trials whose drinking bout has a lick AFTER the ACTUAL
    next trial's VR-reappearance frame (`next_n_anchor_frame`), not a
    nominal 1.5 s -- exact-timestamp version of
    `lick_metrics.frac_trials_post_vr_onset_licking`."""
    n = persisted = 0
    for row in trials_full.itertuples():
        nxt = row.next_n_anchor_frame
        if not np.isfinite(nxt):
            continue
        n += 1
        g = licks_full[licks_full.drink_bout == row.trial]
        if len(g) and (g.frame.to_numpy() > nxt).any():
            persisted += 1
    return float(persisted / n) if n else np.nan


# --------------------------------------------------------------------------
# M5 -- one VR running-speed definition + speed-matched lick spread
# --------------------------------------------------------------------------

def per_trial_vr_speed_cms(vr, trials_full: pd.DataFrame, cm_per_au: float,
                           freeze_thresh_s: float = 2.0, dt: float = 0.1) -> np.ndarray:
    """Per-trial average VR running speed (cm/s), freeze-excluded,
    instantaneous-grid mean -- THE canonical speed definition (M5), the
    same one `lick_metrics.vr_speed_trace`/`run_speed_vr_cms` use
    session-wide, just broken out per trial. `trial_approach_speed_cms`
    (net displacement / total elapsed time, freeze NOT excluded) is a
    DIFFERENT number kept only as the per-trial pace covariate for the
    spatial-vs-temporal regression -- never reported as "running speed" in
    any figure here."""
    pe = vr.position.elapsed_s.to_numpy()
    pl_cm = vr.position.location_au.to_numpy() * cm_per_au
    fz_s, fz_e = _freeze_intervals(vr, freeze_thresh_s)
    out = np.full(len(trials_full), np.nan)
    for i, row in enumerate(trials_full.itertuples()):
        t0, t1 = row.n_time_vr, row.r_time_vr
        if not np.isfinite(t1):
            continue
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
            out[i] = float(np.mean(spd[keep]))
    return out


def speed_matched_lick_spread(sess_a: dict, sess_b: dict) -> dict:
    """lick_spread_cm on session A and B, each restricted to only the
    trials whose per-trial VR speed (M5's canonical definition) falls
    within the OVERLAP of the two sessions' speed ranges -- removes "the
    drug session just runs at a different pace" as a confound on the
    spread comparison."""
    ta, tb = sess_a["trials_full"], sess_b["trials_full"]
    lo = max(np.nanmin(ta.vr_speed_cms), np.nanmin(tb.vr_speed_cms))
    hi = min(np.nanmax(ta.vr_speed_cms), np.nanmax(tb.vr_speed_cms))

    def _spread(sess):
        tf = sess["trials_full"]
        keep = set(tf.loc[(tf.vr_speed_cms >= lo) & (tf.vr_speed_cms <= hi), "trial"])
        ap = sess["licks_full"]
        ap = ap[ap.category.isin(["approaching_nonreward", "approaching_reward"]) & ap.trial.isin(keep)]
        spread = float(np.std(ap.position_au.to_numpy() * sess["cm_per_au"])) if len(ap) else np.nan
        return spread, len(keep)

    spread_a, n_a = _spread(sess_a)
    spread_b, n_b = _spread(sess_b)
    return dict(speed_lo=float(lo), speed_hi=float(hi),
               spread_a=spread_a, n_trials_a=n_a, spread_b=spread_b, n_trials_b=n_b)


# --------------------------------------------------------------------------
# M7 -- exclude corridor-start carryover licks
# --------------------------------------------------------------------------

def filter_carryover_start_licks(licks_full: pd.DataFrame, cm_per_au: float,
                                 cutoff_cm: float = CARRYOVER_CM) -> pd.DataFrame:
    """Drop approach-category licks within `cutoff_cm` of the corridor
    start -- residual previous-trial carryover not caught by
    `classify_licks`'s existing time-based `carryover_frames` window."""
    lf = licks_full.copy()
    cm = lf.position_au * cm_per_au
    drop = lf.category.isin(["approaching_nonreward", "approaching_reward"]) & (cm < cutoff_cm)
    return lf[~drop].reset_index(drop=True)


def lick_spread_and_in_zone(licks_full: pd.DataFrame, reward_zone_cm: float, cm_per_au: float,
                            in_zone_window_cm: float = IN_ZONE_WINDOW_CM,
                            apply_carryover_filter: bool = True,
                            cutoff_cm: float = CARRYOVER_CM) -> dict:
    """lick_spread_cm / frac_approach_licks_in_rz recomputed from
    `licks_full`, honest about M7's carryover filter (on by default here --
    the batch-pipeline `lick_metrics.session_metrics` values do NOT have
    this filter applied, so these numbers can differ slightly from `M`)."""
    lf = filter_carryover_start_licks(licks_full, cm_per_au, cutoff_cm) if apply_carryover_filter else licks_full
    ap = lf[lf.category.isin(["approaching_nonreward", "approaching_reward"])]
    cm = ap.position_au.to_numpy() * cm_per_au
    if not len(cm):
        return dict(lick_spread_cm=np.nan, frac_approach_licks_in_rz=np.nan, n_licks=0)
    return dict(lick_spread_cm=float(np.std(cm)),
               frac_approach_licks_in_rz=float(np.mean(cm >= reward_zone_cm - in_zone_window_cm)),
               n_licks=len(cm))


# --------------------------------------------------------------------------
# M6 -- deceleration-onset sign convention
# --------------------------------------------------------------------------

DECEL_ONSET_LABEL = "deceleration onset: cm before corridor end (higher = earlier anticipation)"


def dist_decel_onset_to_reward(decel_onset_cm: float, reward_zone_cm: float) -> float:
    """Distance-before-corridor-end version of decel_onset_cm -- ALWAYS use
    this (not the raw position) in figures/labels, per M6: a LOWER raw
    position means the animal started slowing down FARTHER from the
    reward, i.e. EARLIER anticipation, which reads backwards if plotted as
    a raw position without this conversion."""
    return float(reward_zone_cm - decel_onset_cm) if np.isfinite(decel_onset_cm) else np.nan


# --------------------------------------------------------------------------
# M2 -- hierarchical (day -> trial) bootstrap + calibration check
# --------------------------------------------------------------------------

BOOT_STAT = {"lick_spread_cm": np.std, "frac_approach_licks_in_rz": np.mean,
            "run_speed_vr_cms": np.mean, "dist_decel_onset_to_reward_cm": np.mean}


def per_trial_value_lists(sess: dict, metric: str) -> list:
    """List of per-trial value arrays (one array per trial) for one of the
    bootstrap-scoped metrics -- the trial-clustered resampling unit."""
    lf, tf = sess["licks_full"], sess["trials_full"]
    cm_per_au, rz_cm = sess["cm_per_au"], sess["reward_zone_cm"]
    out = []
    if metric in ("lick_spread_cm", "frac_approach_licks_in_rz"):
        ap = lf[lf.category.isin(["approaching_nonreward", "approaching_reward"])]
        by_trial = {t: g.position_au.to_numpy() * cm_per_au for t, g in ap.groupby("trial")}
        for t in tf.trial:
            pos = by_trial.get(t, np.array([]))
            out.append(pos if metric == "lick_spread_cm" else (pos >= rz_cm - 30.0).astype(float))
    elif metric == "run_speed_vr_cms":
        for v in tf.vr_speed_cms:
            out.append(np.array([v]) if np.isfinite(v) else np.array([]))
    elif metric == "dist_decel_onset_to_reward_cm":
        # not a per-trial quantity (position-binned aggregate across many
        # trials) -- see the calibration-scoping note in the notebook;
        # included here as a per-SESSION single value duplicated is NOT
        # valid for trial-clustered resampling, so this metric is excluded
        # from BOOT_STAT's actual usage below (kept only for completeness)
        raise NotImplementedError("decel_onset is a multi-trial aggregate, not trial-resamplable this way")
    else:
        raise ValueError(metric)
    return out


def hierarchical_day_trial_bootstrap(per_trial_by_day: dict, n_trials_target: int,
                                     statistic, n_boot: int = 10000, rng=None) -> np.ndarray:
    """Resample DAYS (with replacement) from `per_trial_by_day`'s keys,
    then TRIALS (with replacement, trial-clustered) within each resampled
    day, so the null reflects both between-day and between-trial variance
    (fixes M2: the original pooled-B7+B8 bootstrap only had trial-level
    variance, which is anti-conservative when days differ substantially).
    Each draw pools `n_trials_target` trials total, spread evenly (ceil)
    across the resampled days, then applies `statistic` to the pooled
    values."""
    rng = rng or np.random.default_rng()
    days = list(per_trial_by_day.keys())
    n_days = len(days)
    if n_days < 2:
        return np.full(n_boot, np.nan)
    per_day_draw = max(1, int(np.ceil(n_trials_target / n_days)))
    out = np.full(n_boot, np.nan)
    for b in range(n_boot):
        drawn_days = rng.choice(days, size=n_days, replace=True)
        pooled = []
        for d in drawn_days:
            trial_vals = per_trial_by_day[d]
            if not len(trial_vals):
                continue
            idx = rng.integers(0, len(trial_vals), size=per_day_draw)
            pooled.extend(trial_vals[i] for i in idx)
        pooled = [v for v in pooled if len(v)]
        if pooled:
            out[b] = statistic(np.concatenate(pooled))
    return out


def bootstrap_calibration_check(per_trial_by_day: dict, statistic, alpha: float = 0.05,
                                n_boot: int = 2000, rng=None) -> tuple:
    """Leave-one-day-out calibration: for each day, test it (its own
    trials, pooled) against a hierarchical-bootstrap null built from the
    REMAINING days. A well-calibrated test rejects at rate ~alpha; with
    only `len(per_trial_by_day)` held-out days this estimate is itself
    noisy (report it as such, don't over-read a single failure). Returns
    (false_positive_rate, per_day_results_df)."""
    rng = rng or np.random.default_rng()
    days = list(per_trial_by_day.keys())
    rows = []
    for held_out in days:
        remaining = {d: v for d, v in per_trial_by_day.items() if d != held_out}
        held_vals = [v for v in per_trial_by_day[held_out] if len(v)]
        if not held_vals or len(remaining) < 2:
            continue
        observed = statistic(np.concatenate(held_vals))
        n_target = sum(len(v) for v in per_trial_by_day[held_out])
        null = hierarchical_day_trial_bootstrap(remaining, n_target, statistic, n_boot=n_boot, rng=rng)
        p = bootstrap_pvalue(observed, null)
        rows.append(dict(held_out_day=held_out, observed=observed, p_value=p, significant=bool(p < alpha)))
    df = pd.DataFrame(rows)
    fpr = float(df.significant.mean()) if len(df) else np.nan
    return fpr, df


def run_hierarchical_bootstrap_for_condition(sessions_by_label: dict, animal_days: list,
                                             cond_sess: dict, metric: str,
                                             n_boot: int = 10000, rng=None) -> dict:
    """Full M2 pipeline for one (animal, condition, metric): build the
    per-day per-trial value lists for `animal_days` (the B5-B8 reference),
    run the hierarchical bootstrap, and compare `cond_sess`'s own observed
    value against it."""
    per_day = {d: per_trial_value_lists(sessions_by_label[d], metric) for d in animal_days
              if d in sessions_by_label}
    stat = BOOT_STAT[metric]
    n_target = cond_sess["n_trials"]
    null = hierarchical_day_trial_bootstrap(per_day, n_target, stat, n_boot=n_boot, rng=rng)
    cond_vals = [v for v in per_trial_value_lists(cond_sess, metric) if len(v)]
    observed = float(stat(np.concatenate(cond_vals))) if cond_vals else np.nan
    p = bootstrap_pvalue(observed, null)
    return dict(metric=metric, observed=observed, null_median=float(np.nanmedian(null)),
               null_lo=float(np.nanpercentile(null[np.isfinite(null)], 2.5)) if np.isfinite(null).any() else np.nan,
               null_hi=float(np.nanpercentile(null[np.isfinite(null)], 97.5)) if np.isfinite(null).any() else np.nan,
               p_value=p)


# --------------------------------------------------------------------------
# enriched per-session metrics table -- everything the figures read from,
# computed once per session rather than re-derived per figure
# --------------------------------------------------------------------------

def build_enriched_table(sessions_by_animal: dict, M: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for animal, by_label in sessions_by_animal.items():
        for label, sess in by_label.items():
            lf, tf = sess["licks_full"], sess["trials_full"]
            rz, cm_per_au = sess["reward_zone_cm"], sess["cm_per_au"]

            filt = lick_spread_and_in_zone(lf, rz, cm_per_au, apply_carryover_filter=True)
            unfilt = lick_spread_and_in_zone(lf, rz, cm_per_au, apply_carryover_filter=False)

            strat_o = classify_trial_anticipation_strategy(lf, tf)
            strat_r = classify_strategy_restricted(lf, tf, rz, cm_per_au)
            po, pr = strategy_proportions(strat_o), strategy_proportions(strat_r)

            frac_post_exact = frac_trials_post_vr_onset_exact(lf, tf)
            split = split_bout_pre_post(lf, tf)

            mrow = M[(M.animal == animal) & (M.session_label == label)]
            decel_cm = float(mrow.decel_onset_cm.iloc[0]) if len(mrow) else np.nan
            dist_decel = dist_decel_onset_to_reward(decel_cm, rz)

            rows.append(dict(
                animal=animal, session_label=label,
                session_order=(int(mrow.session_order.iloc[0]) if len(mrow) else np.nan),
                n_trials=sess["n_trials"], reward_zone_cm=rz, cm_per_au=cm_per_au,
                lick_spread_cm=filt["lick_spread_cm"], frac_approach_licks_in_rz=filt["frac_approach_licks_in_rz"],
                lick_spread_cm_unfiltered=unfilt["lick_spread_cm"],
                frac_approach_licks_in_rz_unfiltered=unfilt["frac_approach_licks_in_rz"],
                strategy_spatial=po["spatial"], strategy_transition=po["transition_triggered"], strategy_none=po["none"],
                strategy_spatial_restricted=pr["spatial"], strategy_transition_restricted=pr["transition_triggered"],
                strategy_none_restricted=pr["none"],
                frac_trials_post_vr_onset_exact=frac_post_exact,
                pre_valve_duration_s_median=float(split.pre_duration_s.median()),
                post_valve_duration_s_median=float(split.post_duration_s.median()),
                n_licks_pre_median=float(split.n_licks_pre.median()),
                n_licks_post_median=float(split.n_licks_post.median()),
                run_speed_vr_cms=float(np.nanmean(tf.vr_speed_cms)),
                decel_onset_cm=decel_cm, dist_decel_onset_to_reward_cm=dist_decel,
                trials_with_approach_lick=(float(mrow.trials_with_approach_lick.iloc[0]) / sess["n_trials"]
                                          if len(mrow) else np.nan),
                psth_prevalve_hz=(float(mrow.psth_prevalve_hz.iloc[0]) if len(mrow) else np.nan),
                psth_anticipatory_index=(float(mrow.psth_anticipatory_index.iloc[0]) if len(mrow) else np.nan),
            ))
    E = pd.DataFrame(rows)
    E["session_label"] = pd.Categorical(E["session_label"], categories=SESSIONS, ordered=True)
    return E.sort_values(["animal", "session_order"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# figures -- one function per figure, fixed condition colors, 300dpi PNG+SVG
# --------------------------------------------------------------------------

def _baseline_band(ax, E, animal, metric, days=tuple(ZSCORE_DAYS), color="gray"):
    vals = E[(E.animal == animal) & E.session_label.isin(days)][metric].dropna()
    if len(vals):
        ax.axhspan(vals.min(), vals.max(), color=color, alpha=0.12)
        for d in days:
            v = E[(E.animal == animal) & (E.session_label == d)][metric]
            if len(v):
                ax.scatter([d], v, color=SESSION_COLOR[d], s=60, zorder=3, marker="D")


def _mark_drug_points(ax, xvals, is_drug):
    for x, drug in zip(xvals, is_drug):
        if drug:
            ax.annotate("n=1", (x, ax.get_ylim()[1]), fontsize=11, color="dimgray",
                       ha="center", va="bottom")


def fig_A1_B1_spread_inzone(E, animal, days=tuple(BASELINE_DAYS)):
    """A1 / first two panels of B1: per-day lick spread + in-zone fraction, B1-B8."""
    g = E[(E.animal == animal) & E.session_label.isin(days)]
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    for ax, metric, ylab in zip(axs, ["lick_spread_cm", "frac_approach_licks_in_rz"],
                                ["lick spread (cm)", "fraction of approach licks in RZ"]):
        axs_x = [str(s) for s in g.session_label]
        colors = [SESSION_COLOR[s] for s in g.session_label]
        ax.bar(axs_x, g[metric], color=colors)
        ax.set_xlabel("baseline day"); ax.set_ylabel(ylab)
    return _finalize(fig)


def fig_A2_strategy_by_day(E, animal, sessions=tuple(BASELINE_DAYS) + ("Saline",)):
    """A2: stacked strategy proportions, original (M3 unrestricted) vs restricted, B1-B8 + Saline."""
    g = E[(E.animal == animal) & E.session_label.isin(sessions)]
    fig, axs = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    specs = [("original", ["strategy_spatial", "strategy_transition", "strategy_none"]),
            (f"restricted (final {SPATIAL_WINDOW_CM:g} cm)",
             ["strategy_spatial_restricted", "strategy_transition_restricted", "strategy_none_restricted"])]
    labels = ["spatial", "transition-triggered", "none"]
    colors = ["tab:orange", "tab:cyan", "tab:gray"]
    for ax, (title, cols) in zip(axs, specs):
        x = [str(s) for s in g.session_label]
        bot = np.zeros(len(g))
        for c, lbl, col in zip(cols, labels, colors):
            ax.bar(x, g[c], 0.8, bottom=bot, color=col, label=lbl)
            bot += g[c].to_numpy()
        ax.set_title(title); ax.set_xlabel("session")
        is_drug = [s in CONDITIONS for s in g.session_label]
        _mark_drug_points(ax, x, is_drug)
    axs[0].set_ylabel("fraction of trials"); axs[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _finalize(fig)


def fig_A3_B3_dcz_spread_inzone(E, sessions_by_animal, animal, speed_matched=True):
    """A3 (JSY083) / B3 (JSY084, using frac trials-with-any-VR-lick in place
    of raw spread per the B3 title): lick spread (raw + speed-matched) and
    in-zone fraction for Saline/DCZ100/DCZ200, B5-B8 reference points +
    shaded range."""
    fig, axs = plt.subplots(1, 3, figsize=(24, 7))
    g_ref = E[(E.animal == animal) & E.session_label.isin(ZSCORE_DAYS)]

    x = ["B5", "B6", "B7", "B8"] + list(CONDITIONS)
    gg = E[(E.animal == animal) & E.session_label.isin(x)].set_index("session_label").loc[x]
    colors = [SESSION_COLOR[s] for s in x]

    axs[0].bar(x, gg.lick_spread_cm, color=colors)
    _baseline_band(axs[0], E, animal, "lick_spread_cm")
    axs[0].set_ylabel("lick spread (cm)"); axs[0].set_title("raw")

    if speed_matched:
        ref_pool_label = "B7"  # single reference session for the pairwise speed-matched comparison
        ref_sess = sessions_by_animal[animal][ref_pool_label]
        sm_vals = []
        for s in CONDITIONS:
            if s not in sessions_by_animal[animal]:
                sm_vals.append(np.nan); continue
            r = speed_matched_lick_spread(ref_sess, sessions_by_animal[animal][s])
            sm_vals.append(r["spread_b"])
        axs[1].bar(list(CONDITIONS), sm_vals, color=[SESSION_COLOR[s] for s in CONDITIONS])
        ref_spread = E[(E.animal == animal) & (E.session_label == ref_pool_label)].lick_spread_cm
        if len(ref_spread):
            axs[1].axhline(float(ref_spread.iloc[0]), color="k", ls=":", lw=2, label=f"{ref_pool_label} (unmatched)")
        axs[1].set_ylabel("lick spread (cm)"); axs[1].set_title(f"speed-matched to {ref_pool_label}"); axs[1].legend()
    else:
        axs[1].axis("off")

    axs[2].bar(x, gg.frac_approach_licks_in_rz, color=colors)
    _baseline_band(axs[2], E, animal, "frac_approach_licks_in_rz")
    axs[2].set_ylabel("fraction of approach licks in RZ")
    _mark_drug_points(axs[0], x, [s in CONDITIONS for s in x])
    _mark_drug_points(axs[2], x, [s in CONDITIONS for s in x])
    return _finalize(fig)


def fig_B3_any_vr_lick(E, animal, sessions=("B5", "B6", "B7", "B8") + tuple(CONDITIONS)):
    """B3: fraction of trials with ANY VR (approach-phase) lick, next to
    in-zone fraction, per condition, baseline range shaded."""
    g = E[(E.animal == animal) & E.session_label.isin(sessions)].set_index("session_label").loc[list(sessions)]
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    x = list(sessions); colors = [SESSION_COLOR[s] for s in x]
    axs[0].bar(x, g.trials_with_approach_lick, color=colors)
    _baseline_band(axs[0], E, animal, "trials_with_approach_lick")
    axs[0].set_ylabel("fraction of trials with any VR lick")
    axs[1].bar(x, g.frac_approach_licks_in_rz, color=colors)
    _baseline_band(axs[1], E, animal, "frac_approach_licks_in_rz")
    axs[1].set_ylabel("fraction of approach licks in RZ")
    for ax in axs:
        _mark_drug_points(ax, x, [s in CONDITIONS for s in x])
    return _finalize(fig)


def fig_persistence(E, animal, sessions=("B5", "B6", "B7", "B8") + tuple(CONDITIONS)):
    """A4 / first half of B4: fraction of trials with post-VR-onset licking
    (exact timestamp, M4) + post-valve bout duration."""
    g = E[(E.animal == animal) & E.session_label.isin(sessions)].set_index("session_label").loc[list(sessions)]
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    x = list(sessions); colors = [SESSION_COLOR[s] for s in x]
    axs[0].bar(x, g.frac_trials_post_vr_onset_exact, color=colors)
    axs[0].set_ylabel("fraction of trials"); axs[0].set_title("licking past VR reappearance")
    axs[1].bar(x, g.post_valve_duration_s_median, color=colors)
    axs[1].set_ylabel("post-valve duration, median (s)"); axs[1].set_title("post-valve bout duration")
    for ax in axs:
        _mark_drug_points(ax, x, [s in CONDITIONS for s in x])
    return _finalize(fig)


def fig_dark_anticipation(E, animal, sessions=("B5", "B6", "B7", "B8") + tuple(CONDITIONS)):
    """A5 / second half of B4: pre-valve lick rate + temporal anticipatory index."""
    g = E[(E.animal == animal) & E.session_label.isin(sessions)].set_index("session_label").loc[list(sessions)]
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    x = list(sessions); colors = [SESSION_COLOR[s] for s in x]
    axs[0].bar(x, g.psth_prevalve_hz, color=colors)
    _baseline_band(axs[0], E, animal, "psth_prevalve_hz")
    axs[0].set_ylabel("pre-valve lick rate (Hz)")
    axs[1].bar(x, g.psth_anticipatory_index, color=colors)
    _baseline_band(axs[1], E, animal, "psth_anticipatory_index")
    axs[1].set_ylabel("temporal anticipatory index")
    for ax in axs:
        _mark_drug_points(ax, x, [s in CONDITIONS for s in x])
    return _finalize(fig)


def fig_B2_strategy_late(E, animal, sessions=("B7", "B8") + tuple(CONDITIONS)):
    """B2: strategy proportions for B7, B8, Saline, DCZ100, DCZ200, both definitions."""
    return fig_A2_strategy_by_day(E, animal, sessions=sessions)


def fig_C1_comparison(E, metrics=("dist_decel_onset_to_reward_cm", "lick_spread_cm",
                                  "frac_approach_licks_in_rz", "frac_trials_post_vr_onset_exact",
                                  "psth_prevalve_hz")):
    """C1: z-scores (M1, B5-B8 basis) for the 5 metrics, one row per metric,
    both animals side by side, Saline/DCZ100/DCZ200 on x."""
    # z computed directly from E (the corrected per-session table), not M,
    # so it uses the SAME M5/M7/M4-fixed values as the rest of this
    # notebook's figures, on the M1 B5-B8 baseline basis
    fig, axs = plt.subplots(len(metrics), 1, figsize=(14, 5 * len(metrics)), sharex=True)
    for ax, metric in zip(axs, metrics):
        for animal, marker in zip(ANIMALS, ("o", "s")):
            ge = E[E.animal == animal].set_index("session_label")
            base = ge.loc[ge.index.isin(ZSCORE_DAYS), metric]
            sd = float(base.std())
            zvals = [(float(ge.loc[s, metric]) - float(base.mean())) / sd
                    if s in ge.index and sd else np.nan for s in CONDITIONS]
            ax.plot(CONDITIONS, zvals, marker=marker, ms=14, lw=2.5, label=animal)
        ax.axhline(0, color="k", lw=1)
        ax.axhline(2, color="gray", ls="--", lw=1); ax.axhline(-2, color="gray", ls="--", lw=1)
        ax.set_ylabel(metric, fontsize=14)
    axs[0].legend()
    return _finalize(fig)


# --------------------------------------------------------------------------
# per-animal CSV export
# --------------------------------------------------------------------------

def export_animal_csv(E: pd.DataFrame, animal: str, output_root: str) -> str:
    d = os.path.join(output_root, animal)
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, f"{animal}_lick_figures_metrics.csv")
    E[E.animal == animal].to_csv(out, index=False)
    return out


def run_bootstrap_summary(sessions_by_animal: dict, alpha: float = 0.05,
                          n_boot: int = 10000, calib_n_boot: int = 2000,
                          rng=None) -> tuple:
    """M2, full pipeline: per animal x metric, run the leave-one-day-out
    calibration check first; only run (and only mark as plottable) the
    condition-vs-baseline hierarchical bootstrap if the calibration passes
    (FPR == 0 across the B5-B8 held-out days -- strict, since with only 4
    held-out days any non-zero FPR is already >= 0.25, 5x nominal alpha).
    Returns (results_df, calibration_df)."""
    rng = rng or np.random.default_rng(0)
    calib_rows, result_rows = [], []
    for animal in ANIMALS:
        by_label = sessions_by_animal[animal]
        for metric, stat in BOOT_STAT.items():
            if metric == "dist_decel_onset_to_reward_cm":
                continue   # not trial-resamplable this way, see per_trial_value_lists
            per_day = {d: per_trial_value_lists(by_label[d], metric) for d in ZSCORE_DAYS if d in by_label}
            fpr, calib_df = bootstrap_calibration_check(per_day, stat, alpha=alpha, n_boot=calib_n_boot, rng=rng)
            calib_df = calib_df.assign(animal=animal, metric=metric)
            calib_rows.append(calib_df)
            passed = (fpr == 0.0)
            for cond in CONDITIONS:
                if cond not in by_label:
                    continue
                res = run_hierarchical_bootstrap_for_condition(by_label, ZSCORE_DAYS, by_label[cond],
                                                                metric, n_boot=n_boot, rng=rng)
                res.update(animal=animal, session_label=cond, calibration_fpr=fpr,
                          calibration_passed=passed,
                          p_value_plottable=(res["p_value"] if passed else np.nan))
                result_rows.append(res)
    calib_all = pd.concat(calib_rows, ignore_index=True) if calib_rows else pd.DataFrame()
    results = pd.DataFrame(result_rows)
    return results, calib_all
