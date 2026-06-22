"""
SpeedAnalysis_Behavioral_Comparison.py
=======================================
Extract running speed metrics from preprocessed HDF5 files and compare
across animals and recording days.

Metrics computed per session:
  - Mean / median / max speed (VR + TM)
  - First-half vs second-half speed (split by lap index)
  - Lap-to-lap speed trend (OLS slope within session)
  - Running consistency (CV = std / mean)
  - Fraction of time running (speed > threshold)
  - Speed profile binned by position (115 bins, matching neural data)
  - Speed at each landmark window (±15 cm)
  - Reward zone deceleration index (last 15 cm / mean track speed)
  - Number of valid laps, mean lap duration

All metrics saved to:
  {OUTPUT_DIR}/{animal_id}_speed_metrics.h5

Cross-animal comparison figures saved to OUTPUT_DIR.

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from scipy.stats import linregress, mannwhitneyu

# ============================================================
# CONFIGURATION
# ============================================================

LANDMARK_POSITIONS_CM  = [25, 55, 85, 115]   # cm
LANDMARK_WINDOW_CM     = 15.0                 # ±cm around each landmark
REWARD_ZONE_START_CM   = 125.0               # start of reward zone (last landmark)
REWARD_ZONE_END_CM     = 130.0               # end of track
MIN_RUNNING_SPEED      = 2.0                 # cm/s threshold for "running"
N_POSITION_BINS        = 115                 # must match neural data binning

# Animal groups for comparison
GROUP1 = ['JSY044', 'JSY051', 'JSY055', 'JSY041', 'JSY040']
GROUP2 = ['JSY052', 'JSY054']
# in the group_colors below, add JSY041 AND JSY040 with distinct colors if you want to plot them individually instead of lumping with group1
GROUP_COLORS = {'JSY044': '#1f77b4', 'JSY051': '#ff7f0e', 'JSY052': '#2ca02c',
                'JSY054': '#d62728', 'JSY055': '#9467bd', 'JSY041': '#8c564b', 'JSY040':'gray'}

OUTPUT_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\SpeedAnalysis_BehavioralComparison"

# ── Per-animal session h5 paths ────────────────────────────────────────────
# Format: {animal_id: {day: h5_path}}
SESSIONS = {
    'JSY054': {
        1: r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001\10302025_JSY038_preproc_multi.h5",
        2: r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001\10312025_JSY038_preproc.h5",
        3: r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001\11012025_JSY038_preproc.h5",
        4: r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001\11022025_JSY038_preproc.h5",
        5: r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001\11032025_JSY038_preproc.h5",
        6: r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001\11042025_JSY038_preproc.h5",
        7: r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001\11052025_JSY038_preproc.h5",
    },
    'JSY055': {
        1: r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251205_JSY_JSY055_SpatialModulation_Day1\TSeries-12052025-1740-001\12052025_JSY038_preproc.h5",
        2: r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251206_JSY_JSY055_SpatialModulation_Day2\TSeries-12062025-1810-001\12062025_JSY038_preproc.h5",
        3: r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251207_JSY_JSY055_SpatialModulation_Day3\TSeries-12072025-1825-001\12072025_JSY038_preproc.h5",
        4: r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251208_JSY_JSY055_SpatialModulation_Day4\TSeries-12082025-1633-001\12082025_JSY038_preproc.h5",
        5: r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251209_JSY_JSY055_SpatialModualtion_Day5\TSeries-12092025-2000-001\12092025_JSY038_preproc.h5",
        6: r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251210_JSY_JSY055_SpatialModulation_Day6\TSeries-12102025-1702-001\12102025_JSY038_preproc.h5",
        7: r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251211_JSY_JSY055_SpatialModulation_Day7\TSeries-12112025-1631-001\12112025_JSY038_preproc.h5",
    },
    
    'JSY052': {
        1: r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002\10092025_JSY038_preproc.h5",
        2: r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001\10102025_JSY038_preproc.h5",
        3: r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002\10112025_JSY038_preproc.h5",
        4: r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001\10122025_JSY038_preproc_multi.h5",
        5: r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001\10132025_JSY038_preproc.h5",
        6: r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003\10142025_JSY038_preproc.h5",
        7: r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001\10152025_JSY038_preproc.h5",
    },
    
    'JSY051': {
        1: r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001\11012025_JSY038_preproc.h5",
        2: r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001\11022025_JSY038_preproc.h5",
        3: r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001\11032025_JSY038_preproc.h5",
        4: r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001\11042025_JSY038_preproc.h5",
        5: r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002\11052025_JSY038_preproc.h5",
    },
    
    'JSY044': {
        1: r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001\09062025_JSY038_preproc_multi.h5",
        2: r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001\09072025_JSY038_preproc_multi.h5",
        3: r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001\09082025_JSY038_preproc_multi.h5",
        4: r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001\09092025_JSY038_preproc_multi.h5",
        5: r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001\09102025_JSY038_preproc_multi.h5",
        6: r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001\09112025_JSY038_preproc_multi.h5",
        7: r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001\09122025_JSY038_preproc_multi.h5",
    },
    
    'JSY041': {
        1: r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001\06162025_JSY038_preproc.h5",
        3: r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001\06182025_JSY038_preproc.h5",
        5: r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001\06202025_JSY038_preproc.h5",
        7: r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001\06222025_JSY038_preproc.h5",
    },
    
    'JSY040': {
        1: r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001\06202025_JSY038_preproc.h5",
        3: r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001\06222025_JSY038_preproc.h5",
    },

    # Add other animals:
    # 'JSY052': {1: r"...", 2: r"...", ...},
    # 'JSY044': {1: r"...", ...},
    # 'JSY051': {1: r"...", ...},
    # 'JSY055': {1: r"...", ...},
}

# ============================================================


def compute_speed_profile(speed, location, n_bins=N_POSITION_BINS,
                           track_min=0.0, track_max=130.0):
    """Mean speed in each spatial bin."""
    bins = np.linspace(track_min, track_max, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    profile = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (location >= bins[i]) & (location < bins[i + 1])
        if np.sum(mask) > 0:
            profile[i] = np.mean(speed[mask])
    return profile, centers


def compute_landmark_speed(speed, location, landmark_positions, window_cm):
    """Mean speed within ±window_cm of each landmark."""
    lm_speed = np.full(len(landmark_positions), np.nan)
    for i, lm in enumerate(landmark_positions):
        mask = (location >= lm - window_cm) & (location <= lm + window_cm)
        if np.sum(mask) > 0:
            lm_speed[i] = np.mean(speed[mask])
    return lm_speed


def compute_reward_decel(speed, location, reward_start, reward_end, track_mean_speed):
    """Reward deceleration index = mean speed in reward zone / mean track speed."""
    mask = (location >= reward_start) & (location <= reward_end)
    if np.sum(mask) == 0 or track_mean_speed == 0:
        return np.nan
    reward_speed = np.mean(speed[mask])
    return reward_speed / track_mean_speed


def extract_session_metrics(h5_path):
    """
    Load a preproc h5 and compute all speed metrics for that session.

    Returns a dict of scalar and array metrics.
    """
    with h5py.File(h5_path, 'r') as f:
        speed_vr    = f['speed_cm_s'][:]
        location    = f['location_cm'][:]
        lap_starts  = f['lap_starts'][:]
        lap_ends    = f['lap_ends'][:]
        framerate   = float(f['processing_params/framerate'][()])
        has_tmlog   = 'speed_tmlog_cm_s' in f
        speed_tm    = f['speed_tmlog_cm_s'][:] if has_tmlog else None

    n_frames  = len(speed_vr)
    n_laps    = len(lap_starts)

    # ── Lap durations
    lap_dur_s = (lap_ends - lap_starts) / framerate

    # ── Per-lap mean speed (VR)
    lap_mean_speed_vr = np.array([
        np.mean(speed_vr[lap_starts[i]:lap_ends[i]])
        for i in range(n_laps)
    ])

    # ── First half vs second half (split by lap index)
    half = n_laps // 2
    first_half_speed = np.mean(lap_mean_speed_vr[:half]) if half > 0 else np.nan
    second_half_speed = np.mean(lap_mean_speed_vr[half:]) if half > 0 else np.nan

    # ── Lap-to-lap speed trend (OLS slope, cm/s per lap)
    if n_laps >= 3:
        slope, _, r, pval, _ = linregress(np.arange(n_laps), lap_mean_speed_vr)
        lap_trend_slope = slope
        lap_trend_r2    = r ** 2
        lap_trend_p     = pval
    else:
        lap_trend_slope = lap_trend_r2 = lap_trend_p = np.nan

    # ── Global session metrics (VR)
    mean_speed_vr   = np.mean(speed_vr)
    median_speed_vr = np.median(speed_vr)
    std_speed_vr    = np.std(speed_vr)
    cv_speed_vr     = std_speed_vr / mean_speed_vr if mean_speed_vr > 0 else np.nan
    frac_running_vr = np.mean(speed_vr > MIN_RUNNING_SPEED)

    # ── Position-binned speed profile
    speed_profile_vr, bin_centers = compute_speed_profile(speed_vr, location)

    # ── Landmark zone speeds
    lm_speed_vr  = compute_landmark_speed(speed_vr, location,
                                           LANDMARK_POSITIONS_CM, LANDMARK_WINDOW_CM)
    lm_speed_rel = lm_speed_vr / mean_speed_vr if mean_speed_vr > 0 else lm_speed_vr

    # ── Reward zone deceleration
    reward_decel_idx = compute_reward_decel(speed_vr, location,
                                             REWARD_ZONE_START_CM, REWARD_ZONE_END_CM,
                                             mean_speed_vr)
    reward_speed_abs = np.mean(speed_vr[(location >= REWARD_ZONE_START_CM) &
                                         (location <= REWARD_ZONE_END_CM)]) \
                       if np.any((location >= REWARD_ZONE_START_CM) &
                                  (location <= REWARD_ZONE_END_CM)) else np.nan

    metrics = {
        # Scalars
        'n_laps':             n_laps,
        'mean_lap_dur_s':     np.mean(lap_dur_s),
        'mean_speed_vr':      mean_speed_vr,
        'median_speed_vr':    median_speed_vr,
        'std_speed_vr':       std_speed_vr,
        'cv_speed_vr':        cv_speed_vr,
        'frac_running_vr':    frac_running_vr,
        'first_half_speed':   first_half_speed,
        'second_half_speed':  second_half_speed,
        'lap_trend_slope':    lap_trend_slope,
        'lap_trend_r2':       lap_trend_r2,
        'lap_trend_p':        lap_trend_p,
        'reward_decel_idx':   reward_decel_idx,
        'reward_speed_abs':   reward_speed_abs,
        # Arrays
        'speed_profile_vr':   speed_profile_vr,
        'bin_centers':        bin_centers,
        'lm_speed_vr':        lm_speed_vr,
        'lm_speed_rel':       lm_speed_rel,
        'lap_mean_speed_vr':  lap_mean_speed_vr,
        'lap_dur_s':          lap_dur_s,
    }

    # ── TMlog metrics (if available)
    if speed_tm is not None:
        mean_tm   = np.mean(speed_tm)
        speed_profile_tm, _ = compute_speed_profile(speed_tm, location)
        lm_speed_tm = compute_landmark_speed(speed_tm, location,
                                              LANDMARK_POSITIONS_CM, LANDMARK_WINDOW_CM)
        r_vr_tm = np.corrcoef(speed_vr, speed_tm)[0, 1]

        metrics.update({
            'mean_speed_tm':    mean_tm,
            'median_speed_tm':  np.median(speed_tm),
            'cv_speed_tm':      np.std(speed_tm) / mean_tm if mean_tm > 0 else np.nan,
            'r_vr_tm':          r_vr_tm,
            'speed_profile_tm': speed_profile_tm,
            'lm_speed_tm':      lm_speed_tm,
            'reward_decel_idx_tm': compute_reward_decel(speed_tm, location,
                                                         REWARD_ZONE_START_CM,
                                                         REWARD_ZONE_END_CM, mean_tm),
        })

    return metrics


def save_animal_metrics(animal_id, all_metrics, output_dir):
    """Save all session metrics for one animal to HDF5."""
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{animal_id}_speed_metrics.h5")

    with h5py.File(out_path, 'w') as f:
        for day, m in all_metrics.items():
            grp = f.create_group(f"day{day}")
            for key, val in m.items():
                if val is None:
                    continue
                if isinstance(val, np.ndarray):
                    grp.create_dataset(key, data=val.astype(np.float64))
                else:
                    grp.create_dataset(key, data=float(val))

    print(f"  Saved: {out_path}")
    return out_path


# ── Plotting functions ─────────────────────────────────────────────────────

def plot_mean_speed_trajectories(all_animal_metrics, output_dir):
    """Line plot: mean speed per day per animal, with group shading."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, (ax, metric_key, ylabel) in enumerate(zip(
            axes,
            ['mean_speed_vr', 'mean_speed_vr'],
            ['Mean speed VR (cm/s)', 'Mean speed VR (cm/s)'])):

        for animal_id, day_metrics in all_animal_metrics.items():
            days  = sorted(day_metrics.keys())
            vals  = [day_metrics[d].get(metric_key, np.nan) for d in days]
            color = GROUP_COLORS.get(animal_id, 'gray')
            ls    = '-' if animal_id in GROUP1 else '--'
            ax.plot(days, vals, marker='o', color=color, ls=ls,
                    label=animal_id, lw=1.5, ms=5)

        ax.set_xlabel('Recording day')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Left: all animals; Right: first/second half comparison
    axes[1].cla()
    axes[1].set_title('First vs second half of session')
    for animal_id, day_metrics in all_animal_metrics.items():
        days = sorted(day_metrics.keys())
        fh   = [day_metrics[d].get('first_half_speed',  np.nan) for d in days]
        sh   = [day_metrics[d].get('second_half_speed', np.nan) for d in days]
        color = GROUP_COLORS.get(animal_id, 'gray')
        axes[1].plot(days, fh, marker='o', color=color, ls='-',  lw=1.5, ms=5,
                     label=f'{animal_id} 1st')
        axes[1].plot(days, sh, marker='s', color=color, ls='--', lw=1.5, ms=5,
                     label=f'{animal_id} 2nd', alpha=0.6)
    axes[1].set_xlabel('Recording day')
    axes[1].set_ylabel('Mean speed (cm/s)')
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.3)

    axes[0].set_title('Mean running speed across days')
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'speed_trajectories.png'), dpi=150)
    plt.close(fig)
    print("  Saved: speed_trajectories.png")


def plot_speed_profiles(all_animal_metrics, output_dir):
    """Speed-by-position profiles per animal, early (day1-2) vs late (day6-7)."""
    n_animals = len(all_animal_metrics)
    fig, axes = plt.subplots(1, n_animals, figsize=(4 * n_animals, 4), sharey=True)
    if n_animals == 1:
        axes = [axes]

    for ax, (animal_id, day_metrics) in zip(axes, all_animal_metrics.items()):
        days = sorted(day_metrics.keys())
        early_days = days[:2]
        late_days  = days[-2:]

        for day_set, label, alpha in [(early_days, 'early', 1.0),
                                       (late_days,  'late',  0.5)]:
            profiles = [day_metrics[d]['speed_profile_vr']
                        for d in day_set if 'speed_profile_vr' in day_metrics[d]]
            if not profiles:
                continue
            mean_prof = np.nanmean(profiles, axis=0)
            bin_centers = day_metrics[days[0]]['bin_centers']
            ax.plot(bin_centers, mean_prof, label=label, alpha=alpha,
                    color=GROUP_COLORS.get(animal_id, 'gray'))

        for lm in LANDMARK_POSITIONS_CM:
            ax.axvline(lm, color='gray', lw=0.8, ls='--')
        ax.axvspan(REWARD_ZONE_START_CM, REWARD_ZONE_END_CM,
                   alpha=0.1, color='red', label='reward')
        ax.set_title(animal_id)
        ax.set_xlabel('Position (cm)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Mean speed (cm/s)')
    fig.suptitle('Speed profile by position (early vs late sessions)', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'speed_profiles_by_position.png'), dpi=150)
    plt.close(fig)
    print("  Saved: speed_profiles_by_position.png")


def plot_reward_decel(all_animal_metrics, output_dir):
    """Reward deceleration index per day per animal."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for animal_id, day_metrics in all_animal_metrics.items():
        days = sorted(day_metrics.keys())
        vals = [day_metrics[d].get('reward_decel_idx', np.nan) for d in days]
        color = GROUP_COLORS.get(animal_id, 'gray')
        ls    = '-' if animal_id in GROUP1 else '--'
        ax.plot(days, vals, marker='o', color=color, ls=ls,
                label=animal_id, lw=1.5, ms=5)

    ax.axhline(1.0, color='k', lw=0.8, ls=':', label='no decel (ratio=1)')
    ax.set_xlabel('Recording day')
    ax.set_ylabel('Reward decel index (reward zone / mean speed)')
    ax.set_title('Reward zone deceleration across days')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'reward_deceleration.png'), dpi=150)
    plt.close(fig)
    print("  Saved: reward_deceleration.png")


def plot_landmark_speed(all_animal_metrics, output_dir):
    """Relative speed at each landmark per animal (averaged across days)."""
    n_animals = len(all_animal_metrics)
    fig, axes = plt.subplots(1, n_animals, figsize=(3.5 * n_animals, 4), sharey=True)
    if n_animals == 1:
        axes = [axes]

    lm_labels = [f'LM{i+1}\n({p}cm)' for i, p in enumerate(LANDMARK_POSITIONS_CM)]

    for ax, (animal_id, day_metrics) in zip(axes, all_animal_metrics.items()):
        days = sorted(day_metrics.keys())
        rel_speeds = np.array([day_metrics[d].get('lm_speed_rel',
                                np.full(4, np.nan)) for d in days])
        mean_rel = np.nanmean(rel_speeds, axis=0)
        sem_rel  = np.nanstd(rel_speeds, axis=0) / np.sqrt(np.sum(~np.isnan(rel_speeds), axis=0))

        ax.bar(range(4), mean_rel, yerr=sem_rel,
               color=GROUP_COLORS.get(animal_id, 'gray'), alpha=0.7, capsize=4)
        ax.axhline(1.0, color='k', lw=0.8, ls='--', label='track mean')
        ax.set_xticks(range(4))
        ax.set_xticklabels(lm_labels, fontsize=8)
        ax.set_title(animal_id)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Speed relative to session mean')
    fig.suptitle('Speed at landmark positions (mean ± SD across days)', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'landmark_speed.png'), dpi=150)
    plt.close(fig)
    print("  Saved: landmark_speed.png")


def plot_within_session_trend(all_animal_metrics, output_dir):
    """Lap-to-lap speed slope: does the animal accelerate or slow across laps?"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for animal_id, day_metrics in all_animal_metrics.items():
        days  = sorted(day_metrics.keys())
        slopes = [day_metrics[d].get('lap_trend_slope', np.nan) for d in days]
        color  = GROUP_COLORS.get(animal_id, 'gray')
        ls     = '-' if animal_id in GROUP1 else '--'
        ax.plot(days, slopes, marker='o', color=color, ls=ls,
                label=animal_id, lw=1.5, ms=5)

    ax.axhline(0.0, color='k', lw=0.8, ls=':', label='no trend')
    ax.set_xlabel('Recording day')
    ax.set_ylabel('Speed slope (cm/s per lap)')
    ax.set_title('Within-session lap-to-lap speed trend')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'within_session_speed_trend.png'), dpi=150)
    plt.close(fig)
    print("  Saved: within_session_speed_trend.png")


def plot_group_comparison(all_animal_metrics, output_dir):
    """Day 1 metrics: Group 1 vs Group 2 summary."""
    metrics_to_compare = [
        ('mean_speed_vr',    'Mean speed (cm/s)'),
        ('reward_decel_idx', 'Reward decel index'),
        ('frac_running_vr',  'Fraction running'),
        ('cv_speed_vr',      'Speed CV'),
        ('lap_trend_slope',  'Lap trend slope (cm/s/lap)'),
    ]

    fig, axes = plt.subplots(1, len(metrics_to_compare),
                              figsize=(3.5 * len(metrics_to_compare), 4))

    for ax, (metric_key, ylabel) in zip(axes, metrics_to_compare):
        g1_vals = []
        g2_vals = []
        for animal_id, day_metrics in all_animal_metrics.items():
            # Use Day 1 if available, else first available day
            days = sorted(day_metrics.keys())
            day  = days[0]
            val  = day_metrics[day].get(metric_key, np.nan)
            if animal_id in GROUP1:
                g1_vals.append(val)
            elif animal_id in GROUP2:
                g2_vals.append(val)

        g1_vals = np.array([v for v in g1_vals if not np.isnan(v)])
        g2_vals = np.array([v for v in g2_vals if not np.isnan(v)])

        # Plot individual points + mean
        for i, (vals, label, color) in enumerate([(g1_vals, 'G1\n(044/051/055)', '#1f77b4'),
                                                    (g2_vals, 'G2\n(052/054)',     '#d62728')]):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, alpha=0.7, s=40, zorder=3)
            if len(vals) > 0:
                ax.plot([i - 0.2, i + 0.2], [np.mean(vals)] * 2,
                        color=color, lw=2.5, zorder=4)

        # Mann-Whitney U test
        if len(g1_vals) >= 2 and len(g2_vals) >= 2:
            _, p = mannwhitneyu(g1_vals, g2_vals, alternative='two-sided')
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.set_title(f'{sig}\np={p:.3f}', fontsize=9)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['G1', 'G2'], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Group comparison — Day 1 speed metrics', fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'group_comparison_day1.png'), dpi=150)
    plt.close(fig)
    print("  Saved: group_comparison_day1.png")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_animal_metrics = {}
    failed = []

    for animal_id, session_paths in SESSIONS.items():
        print(f"\n{'='*60}")
        print(f"Animal: {animal_id}")
        print(f"{'='*60}")

        animal_metrics = {}

        for day, h5_path in sorted(session_paths.items()):
            print(f"  Day {day}: {os.path.basename(h5_path)}")
            try:
                m = extract_session_metrics(h5_path)
                animal_metrics[day] = m
                has_tm = 'mean_speed_tm' in m
                print(f"    n_laps={m['n_laps']}  "
                      f"mean_vr={m['mean_speed_vr']:.1f} cm/s  "
                      f"{'mean_tm='+str(round(m['mean_speed_tm'],1))+' cm/s  ' if has_tm else ''}"
                      f"reward_decel={m['reward_decel_idx']:.2f}  "
                      f"lap_slope={m['lap_trend_slope']:+.2f} cm/s/lap  "
                      f"1st_half={m['first_half_speed']:.1f}  "
                      f"2nd_half={m['second_half_speed']:.1f} cm/s")
            except Exception as e:
                print(f"    FAILED: {e}")
                failed.append((animal_id, day, str(e)))

        if animal_metrics:
            save_animal_metrics(animal_id, animal_metrics, OUTPUT_DIR)
            all_animal_metrics[animal_id] = animal_metrics

    # ── Figures ────────────────────────────────────────────
    if all_animal_metrics:
        print(f"\nGenerating figures → {OUTPUT_DIR}")
        plot_mean_speed_trajectories(all_animal_metrics, OUTPUT_DIR)
        plot_speed_profiles(all_animal_metrics, OUTPUT_DIR)
        plot_reward_decel(all_animal_metrics, OUTPUT_DIR)
        plot_landmark_speed(all_animal_metrics, OUTPUT_DIR)
        plot_within_session_trend(all_animal_metrics, OUTPUT_DIR)
        plot_group_comparison(all_animal_metrics, OUTPUT_DIR)

    if failed:
        print(f"\nFailed sessions:")
        for animal_id, day, err in failed:
            print(f"  {animal_id} Day {day}: {err}")

    print("\nDone.")
