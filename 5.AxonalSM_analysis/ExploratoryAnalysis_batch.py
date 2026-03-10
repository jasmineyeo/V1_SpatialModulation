"""
ExploratoryAnalysis_batch.py
Batch version of Exploratory_Analysis.ipynb.

For each recording (preproc.h5), runs:
  1. Spatial Information Analysis (fast, vectorised)
  2. Reward-Proximity Ramping Analysis
  3. Bayesian Population Decoding
  ...and saves figures + an HDF5 results file to
  <data_filepath>/axon_spatial_analysis/

JSY, 2026
"""

import os
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for batch
import matplotlib.pyplot as plt
from scipy import stats as sp_stats
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")
from helper import files


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_FILEPATHS = [

            r"D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism\260225_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day1\TSeries-02252026-0903-001",
            r"D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism\260226_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day2\TSeries-02262026-0915-001",
            r"D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism\260227_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day3\TSeries-02262026-1253-001",
            r"D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism\260228_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day4\TSeries-02282026-0919-001",
            r"D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism\260301_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day5\TSeries-03012026-0914-002",
            r"D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism\260302_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day6\TSeries-03022026-1226-001",
            r"D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism\260303_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day7\TSeries-03032026-0817-001",


    # # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001",
    # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260203_JSY_JSY061_SpMod_AxonalImaging_Day2\TSeries-02032026-1751-002",
    # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260204_JSY_JSY061_SpMod_AxonalImaging_Day3\TSeries-02042026-2009-001",
    # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260205_JSY_JSY061_SpMod_AxonalImaging_Day4\TSeries-02052026-1833-002",
    # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260206_JSY_JSY061_SpMod_AxonalImaging_Day5\TSeries-02062026-1850-001",
    # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260207_JSY_JSY061_SpMod_AxonalImaging_Day6\TSeries-02072026-2023-001",
    # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260208_JSY_JSY061_SpMod_AxonalImaging_Day7\TSeries-02082026-1826-001",
]

# Analysis parameters
AXON_ACTIVITY_THRESHOLD = 0.01   # min fraction of frames with activity
N_SHUFFLES_SI           = 100    # spatial info shuffles (200 for publication)
ALPHA_SI                = 0.05
SMOOTH_SIGMA_SI         = 1.5
MIN_LAPS_RAMP           = 5
ALPHA_RAMP              = 0.05
N_DIST_BINS             = 30
THRESHOLD_SD_DECODE     = 2.0
SMOOTH_SIGMA_DECODE     = 2.0
DECODE_EVERY_N          = 3
N_SHUFFLES_DECODE       = 5      # increase to 20 for publication

# Set to True to skip sessions that already have results
SKIP_IF_DONE = False


# ============================================================================
# ── SPATIAL INFORMATION (fast, vectorised) ───────────────────────────────────
# ============================================================================

def compute_tuning_curves_batch(activity_2d, location, bin_edges, min_occupancy=2):
    n_cells, n_frames = activity_2d.shape
    n_bins = len(bin_edges) - 1
    bin_idx = np.clip(np.digitize(location, bin_edges) - 1, 0, n_bins - 1)
    occupancy = np.bincount(bin_idx, minlength=n_bins)
    tuning_curves = np.full((n_cells, n_bins), np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if np.sum(mask) >= min_occupancy:
            tuning_curves[:, b] = np.mean(activity_2d[:, mask], axis=1)
    return tuning_curves, occupancy


def skaggs_si_batch(tuning_curves, occupancy, epsilon=1e-10):
    n_cells, n_bins = tuning_curves.shape
    valid_bins = occupancy > 0
    p_i = np.zeros(n_bins)
    p_i[valid_bins] = occupancy[valid_bins] / np.sum(occupancy[valid_bins])
    si_bits_spike = np.full(n_cells, np.nan)
    mean_rates = np.full(n_cells, np.nan)
    for ci in range(n_cells):
        tc = tuning_curves[ci]
        valid = ~np.isnan(tc) & valid_bins
        if np.sum(valid) < 3:
            continue
        tc_v = tc[valid]
        p_v  = p_i[valid] / np.sum(p_i[valid])
        mr   = np.sum(p_v * tc_v)
        mean_rates[ci] = mr
        if mr <= epsilon:
            si_bits_spike[ci] = 0.0
            continue
        pos = tc_v > epsilon
        if not np.any(pos):
            si_bits_spike[ci] = 0.0
            continue
        ratio = tc_v[pos] / mr
        si_bits_spike[ci] = np.sum(p_v[pos] * ratio * np.log2(ratio))
    return si_bits_spike, mean_rates


def circular_shuffle_si_fast(activity_2d, location, bin_edges, occupancy,
                              n_shuffles=200, min_shift_frac=0.1):
    n_cells, n_frames = activity_2d.shape
    min_shift = int(n_frames * min_shift_frac)
    max_shift = n_frames - min_shift
    shifts = np.random.randint(min_shift, max_shift, size=n_shuffles)
    shuffle_si = np.zeros((n_cells, n_shuffles))
    for s_idx, shift in enumerate(shifts):
        shifted = np.roll(activity_2d, shift, axis=1)
        tc_shuf, _ = compute_tuning_curves_batch(shifted, location, bin_edges)
        si_shuf, _ = skaggs_si_batch(tc_shuf, occupancy)
        shuffle_si[:, s_idx] = si_shuf
    return shuffle_si


def run_spatial_information_analysis_fast(temporal_spikes, temporal_location,
                                          active_mask, bin_centers,
                                          n_shuffles=200, alpha=0.05,
                                          smooth_sigma=1.5, verbose=True):
    n_cells, n_frames = temporal_spikes.shape
    n_bins = len(bin_centers)
    half_sp = np.mean(np.diff(bin_centers)) / 2
    bin_edges = np.concatenate([[bin_centers[0] - half_sp],
                                 (bin_centers[:-1] + bin_centers[1:]) / 2,
                                 [bin_centers[-1] + half_sp]])
    active_idx   = np.where(active_mask)[0]
    n_active     = len(active_idx)
    active_spikes = temporal_spikes[active_idx]

    if verbose:
        print(f"  SI: {n_active} axons, {n_shuffles} shuffles...")

    # Tuning curves
    tuning_raw, occupancy = compute_tuning_curves_batch(active_spikes, temporal_location, bin_edges)
    tuning_smooth = np.zeros_like(tuning_raw)
    for i in range(n_active):
        tc = tuning_raw[i]
        tc_filled = np.where(np.isnan(tc), 0, tc)
        tc_s = gaussian_filter1d(tc_filled, sigma=smooth_sigma)
        tc_s[np.isnan(tc)] = np.nan
        tuning_smooth[i] = tc_s

    # Observed SI
    si_obs, mean_rates = skaggs_si_batch(tuning_smooth, occupancy)

    # Shuffle
    shuffle_si = circular_shuffle_si_fast(active_spikes, temporal_location, bin_edges,
                                          occupancy, n_shuffles=n_shuffles)

    p_values = np.zeros(n_active)
    si_corr  = np.full(n_active, np.nan)
    shuf_mean = np.full(n_active, np.nan)
    for i in range(n_active):
        obs  = si_obs[i]
        shuf = shuffle_si[i][~np.isnan(shuffle_si[i])]
        if np.isnan(obs) or len(shuf) == 0:
            p_values[i] = np.nan
            continue
        p_values[i] = np.mean(shuf >= obs)
        shuf_mean[i] = np.mean(shuf)
        si_corr[i]   = obs - shuf_mean[i]

    is_sig = p_values < alpha

    # Map back to full arrays
    tuning_full   = np.full((n_cells, n_bins), np.nan)
    si_full       = np.full(n_cells, np.nan)
    si_corr_full  = np.full(n_cells, np.nan)
    mr_full       = np.full(n_cells, np.nan)
    pv_full       = np.full(n_cells, np.nan)
    sig_full      = np.zeros(n_cells, dtype=bool)
    smean_full    = np.full(n_cells, np.nan)

    tuning_full[active_idx]  = tuning_smooth
    si_full[active_idx]      = si_obs
    si_corr_full[active_idx] = si_corr
    mr_full[active_idx]      = mean_rates
    pv_full[active_idx]      = p_values
    sig_full[active_idx]     = is_sig
    smean_full[active_idx]   = shuf_mean

    if verbose:
        n_sig = np.sum(is_sig)
        print(f"  → significant: {n_sig}/{n_active} ({100*n_sig/n_active:.1f}%)")

    return {
        'tuning_curves': tuning_full,
        'occupancy': occupancy,
        'si_bits_spike': si_full,
        'si_bits_spike_corrected': si_corr_full,
        'mean_rate': mr_full,
        'p_value': pv_full,
        'is_significant': sig_full,
        'shuffle_mean': smean_full,
        'bin_edges': bin_edges,
    }


def plot_spatial_info_summary(si_results, bin_centers, active_axons,
                               reliable_cells=None, max_examples=12):
    tc      = si_results['tuning_curves']
    sig_mask = si_results['is_significant']
    si_corr = si_results['si_bits_spike_corrected']
    cmap    = LinearSegmentedColormap.from_list('BlueBlack',
              [(1,1,1), (0.3,0.5,0.9), (0,0,0.5)])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Heatmap
    ax = axes[0, 0]
    sig_idx = np.where(sig_mask)[0]
    if len(sig_idx) > 0:
        tc_sig  = tc[sig_idx]
        tc_norm = np.zeros_like(tc_sig)
        for i in range(len(tc_sig)):
            row = tc_sig[i]
            valid = ~np.isnan(row)
            if np.any(valid):
                rmin, rmax = np.nanmin(row), np.nanmax(row)
                if rmax > rmin:
                    tc_norm[i] = (row - rmin) / (rmax - rmin)
        sort_order = np.argsort(np.nanargmax(tc_norm, axis=1))
        tc_disp = np.nan_to_num(tc_norm[sort_order])
        ax.imshow(tc_disp, aspect='auto', cmap=cmap, vmin=0, vmax=1,
                  extent=[bin_centers[0], bin_centers[-1], len(sig_idx), 0])
        ax.set_ylabel('Axon # (sorted by peak)')
    else:
        ax.text(0.5, 0.5, 'No significant axons', ha='center', va='center',
                transform=ax.transAxes)
    ax.set_xlabel('Position (cm)')
    ax.set_title(f'Significant spatial tuning (n={len(sig_idx)})', fontweight='bold')

    # (B) SI distribution
    ax = axes[0, 1]
    si_active = si_corr[active_axons & ~np.isnan(si_corr)]
    si_sig_v  = si_corr[sig_mask & active_axons & ~np.isnan(si_corr)]
    si_nsig_v = si_corr[~sig_mask & active_axons & ~np.isnan(si_corr)]
    bins_h = np.linspace(np.nanpercentile(si_active, 1) if len(si_active) else -0.05,
                         np.nanpercentile(si_active, 99) if len(si_active) else 0.5, 40)
    if len(si_nsig_v): ax.hist(si_nsig_v, bins=bins_h, alpha=0.5, color='gray',
                                label=f'Non-sig (n={len(si_nsig_v)})')
    if len(si_sig_v):  ax.hist(si_sig_v,  bins=bins_h, alpha=0.7, color='steelblue',
                                label=f'Significant (n={len(si_sig_v)})')
    ax.axvline(0, color='k', ls='--', alpha=0.5)
    ax.set_xlabel('Corrected SI (bits/spike)')
    ax.set_ylabel('Count')
    ax.set_title('Spatial Information Distribution', fontweight='bold')
    ax.legend(fontsize=10)

    # (C) Bar chart
    ax = axes[1, 0]
    n_act = int(np.sum(active_axons))
    n_sig_c = int(np.sum(sig_mask & active_axons))
    bars = ax.bar(['Non-significant', 'Significant'], [n_act - n_sig_c, n_sig_c],
                  color=['lightgray', 'steelblue'], edgecolor='black', lw=0.8)
    for bar, cnt in zip(bars, [n_act - n_sig_c, n_sig_c]):
        pct = 100 * cnt / n_act if n_act else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{cnt}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Number of axons')
    ax.set_title('Spatial Information Significance', fontweight='bold')

    # (D) Example curves
    ax = axes[1, 1]
    if len(sig_idx) > 0:
        top_order = sig_idx[np.argsort(si_corr[sig_idx])[::-1][:max_examples]]
        cmap_l = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_order)))
        for i, ci in enumerate(top_order):
            curve = tc[ci]
            valid = ~np.isnan(curve)
            if np.any(valid):
                cmin, cmax = np.nanmin(curve), np.nanmax(curve)
                cn = (curve - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(curve)
                ax.plot(bin_centers, cn + i * 0.15, color=cmap_l[i], lw=1.2, alpha=0.8)
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Norm. ΔF/F (stacked)')
    ax.set_title(f'Top {min(max_examples, len(sig_idx))} axons by SI', fontweight='bold')

    plt.suptitle('Axonal Spatial Information Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================================
# ── REWARD-PROXIMITY RAMPING ─────────────────────────────────────────────────
# ============================================================================

def compute_per_lap_ramping(activity_1d, distance_to_reward,
                            lap_starts, lap_ends, min_frames=20):
    slopes, r_values, lap_indices = [], [], []
    for lap_i, (s, e) in enumerate(zip(lap_starts, lap_ends)):
        if (e - s) < min_frames:
            continue
        act_lap  = activity_1d[s:e]
        dist_lap = distance_to_reward[s:e]
        if np.std(act_lap) < 1e-10 or np.std(dist_lap) < 1e-10:
            continue
        slope, _, r, _, _ = sp_stats.linregress(dist_lap, act_lap)
        slopes.append(slope)
        r_values.append(r)
        lap_indices.append(lap_i)
    return slopes, r_values, lap_indices


def classify_ramping(slopes, min_laps=5, alpha=0.05):
    slopes_clean = [s for s in slopes if not np.isnan(s)]
    if len(slopes_clean) < min_laps:
        return 'insufficient_data', np.nan, np.nan
    slopes_arr = np.array(slopes_clean)
    mean_slope  = np.mean(slopes_arr)
    try:
        _, p_value = sp_stats.wilcoxon(slopes_arr, alternative='two-sided')
    except ValueError:
        return 'non_ramping', 1.0, mean_slope
    if p_value < alpha:
        return ('ramp_up' if mean_slope < 0 else 'ramp_down'), p_value, mean_slope
    return 'non_ramping', p_value, mean_slope


def compute_reward_proximity_profile(activity_1d, distance_to_reward,
                                      n_dist_bins=30, max_dist_cm=None,
                                      smooth_sigma=1.0):
    if max_dist_cm is None:
        max_dist_cm = np.percentile(distance_to_reward, 99)
    dist_edges   = np.linspace(0, max_dist_cm, n_dist_bins + 1)
    dist_centers = (dist_edges[:-1] + dist_edges[1:]) / 2
    profile = np.full(n_dist_bins, np.nan)
    bin_idx = np.clip(np.digitize(distance_to_reward, dist_edges) - 1, 0, n_dist_bins - 1)
    for b in range(n_dist_bins):
        mask = bin_idx == b
        if np.sum(mask) >= 2:
            profile[b] = np.mean(activity_1d[mask])
    valid = ~np.isnan(profile)
    if np.sum(valid) > 3 and smooth_sigma > 0:
        profile = gaussian_filter1d(
            np.interp(np.arange(n_dist_bins), np.where(valid)[0], profile[valid]),
            sigma=smooth_sigma)
    return profile, dist_centers


def run_ramping_analysis(temporal_spikes, distance_to_reward,
                         lap_starts, lap_ends, active_mask,
                         min_laps=5, alpha=0.05, n_dist_bins=30, verbose=True):
    n_cells = temporal_spikes.shape[0]
    classifications = np.array(['unanalyzed'] * n_cells, dtype='U20')
    p_values   = np.full(n_cells, np.nan)
    mean_slopes = np.full(n_cells, np.nan)
    profiles   = np.full((n_cells, n_dist_bins), np.nan)
    dist_centers = None
    cell_indices = np.where(active_mask)[0]

    if verbose:
        print(f"  Ramping: {len(cell_indices)} axons...")

    for ci in cell_indices:
        trace  = temporal_spikes[ci]
        slopes, _, _ = compute_per_lap_ramping(trace, distance_to_reward,
                                                lap_starts, lap_ends)
        cls, pv, ms  = classify_ramping(slopes, min_laps=min_laps, alpha=alpha)
        classifications[ci] = cls
        p_values[ci]   = pv
        mean_slopes[ci] = ms
        prof, dc = compute_reward_proximity_profile(trace, distance_to_reward,
                                                     n_dist_bins=n_dist_bins)
        profiles[ci] = prof
        if dist_centers is None:
            dist_centers = dc

    analyzed = active_mask & ~np.isin(classifications, ['unanalyzed', 'insufficient_data'])
    n_an   = int(np.sum(analyzed))
    n_up   = int(np.sum(classifications == 'ramp_up'))
    n_down = int(np.sum(classifications == 'ramp_down'))
    if verbose:
        print(f"  → ramp-up: {n_up}, ramp-down: {n_down}, non-ramping: {n_an-n_up-n_down}")

    return {
        'classifications': classifications,
        'p_values': p_values,
        'mean_slopes': mean_slopes,
        'profiles': profiles,
        'dist_centers': dist_centers,
        'n_dist_bins': n_dist_bins,
    }


def plot_ramping_summary(ramp_results, active_axons):
    cls          = ramp_results['classifications']
    profiles     = ramp_results['profiles']
    dist_centers = ramp_results['dist_centers']
    mean_slopes  = ramp_results['mean_slopes']
    analyzed     = active_axons & ~np.isin(cls, ['unanalyzed', 'insufficient_data'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Proximity profiles
    ax = axes[0, 0]
    for label, color, ls in [('ramp_up', '#E53935', '-'),
                               ('ramp_down', '#1E88E5', '-'),
                               ('non_ramping', 'gray', '--')]:
        mask = (cls == label) & active_axons
        if not np.any(mask):
            continue
        mp  = np.nanmean(profiles[mask], axis=0)
        sem = np.nanstd(profiles[mask], axis=0) / np.sqrt(np.sum(mask))
        ax.plot(dist_centers, mp, color=color, ls=ls, lw=2,
                label=f'{label} (n={np.sum(mask)})')
        ax.fill_between(dist_centers, mp - sem, mp + sem, color=color, alpha=0.15)
    ax.set_xlabel('Distance to reward (cm)')
    ax.set_ylabel('Mean ΔF/F')
    ax.set_title('Reward Proximity Profiles', fontweight='bold')
    ax.legend(fontsize=10)
    ax.invert_xaxis()

    # (B) Pie
    ax = axes[0, 1]
    n_up   = int(np.sum(cls[analyzed] == 'ramp_up'))
    n_down = int(np.sum(cls[analyzed] == 'ramp_down'))
    n_non  = int(np.sum(cls[analyzed] == 'non_ramping'))
    sizes  = [n_up, n_down, n_non]
    labels_p = [f'Ramp-up\n(n={n_up})', f'Ramp-down\n(n={n_down})',
                f'Non-ramping\n(n={n_non})']
    colors_p = ['#E53935', '#1E88E5', 'lightgray']
    nz = [s > 0 for s in sizes]
    if any(nz):
        ax.pie([s for s, b in zip(sizes, nz) if b],
               labels=[l for l, b in zip(labels_p, nz) if b],
               colors=[c for c, b in zip(colors_p, nz) if b],
               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
    ax.set_title(f'Ramping Classification (n={np.sum(analyzed)})', fontweight='bold')

    # (C) Slope distribution
    ax = axes[1, 0]
    for label, color in [('ramp_up', '#E53935'), ('ramp_down', '#1E88E5'),
                          ('non_ramping', 'lightgray')]:
        mask = (cls == label) & analyzed
        s = mean_slopes[mask]
        s = s[~np.isnan(s)]
        if len(s):
            ax.hist(s, bins=30, alpha=0.6, color=color, label=label)
    ax.axvline(0, color='k', ls='--', lw=1)
    ax.set_xlabel('Mean slope (ΔF/F per cm)')
    ax.set_ylabel('Count')
    ax.legend(fontsize=9)
    ax.set_title('Distribution of Ramping Slopes', fontweight='bold')

    # (D) Examples
    ax = axes[1, 1]
    examples = []
    for label, color in [('ramp_up', '#E53935'), ('ramp_down', '#1E88E5')]:
        mask = (cls == label) & active_axons
        idx  = np.where(mask)[0]
        if len(idx):
            top = idx[np.argsort(np.abs(mean_slopes[idx]))[::-1][:3]]
            examples.extend([(ci, color, label) for ci in top])
    for i, (ci, color, label) in enumerate(examples[:6]):
        prof  = profiles[ci]
        valid = ~np.isnan(prof)
        if np.any(valid):
            pmin, pmax = np.nanmin(prof), np.nanmax(prof)
            pn = (prof - pmin) / (pmax - pmin) if pmax > pmin else np.zeros_like(prof)
            ax.plot(dist_centers, pn + i * 0.2, color=color, lw=1.5, alpha=0.8,
                    label=f'ROI {ci} ({label})')
    ax.set_xlabel('Distance to reward (cm)')
    ax.set_ylabel('Norm. ΔF/F (stacked)')
    ax.set_title('Example Ramping Axons', fontweight='bold')
    ax.invert_xaxis()
    ax.legend(fontsize=7)

    plt.suptitle('Axonal Reward-Proximity Ramping Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================================
# ── BAYESIAN DECODING ────────────────────────────────────────────────────────
# ============================================================================

def binarize_activity(temporal_spikes, method='threshold', threshold_sd=2.0):
    n_cells, n_frames = temporal_spikes.shape
    binary = np.zeros((n_cells, n_frames), dtype=np.int8)
    for ci in range(n_cells):
        trace = temporal_spikes[ci]
        mu, sd = np.mean(trace), np.std(trace)
        if sd < 1e-10:
            continue
        above = trace > (mu + threshold_sd * sd)
        if method == 'derivative':
            deriv = np.diff(trace, prepend=trace[0])
            binary[ci] = (above & (deriv > 0)).astype(np.int8)
        else:
            binary[ci] = above.astype(np.int8)
    return binary


def compute_decoder_tuning(binary_activity, location, bin_edges, cell_mask,
                            smooth_sigma=2.0, min_rate=1e-6):
    n_bins = len(bin_edges) - 1
    cell_indices = np.where(cell_mask)[0]
    n_used = len(cell_indices)
    bin_idx = np.clip(np.digitize(location, bin_edges) - 1, 0, n_bins - 1)
    tuning = np.zeros((n_used, n_bins))
    for i, ci in enumerate(cell_indices):
        for b in range(n_bins):
            mask = bin_idx == b
            if np.sum(mask) > 0:
                tuning[i, b] = np.sum(binary_activity[ci, mask]) / np.sum(mask)
        tuning[i] = np.clip(gaussian_filter1d(tuning[i], sigma=smooth_sigma),
                            min_rate, 1 - min_rate)
    return tuning, cell_indices


def bayesian_decode_frame(binary_frame, tuning, prior):
    log_like = np.zeros(len(prior))
    for i in range(len(binary_frame)):
        log_like += (np.log(tuning[i]) if binary_frame[i] == 1
                     else np.log(1 - tuning[i]))
    log_post = log_like + np.log(prior)
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= np.sum(post)
    return post, int(np.argmax(post))


def run_bayesian_decoding(temporal_spikes, temporal_location,
                          lap_starts, lap_ends, active_mask, bin_centers,
                          threshold_sd=2.0, binarize_method='threshold',
                          smooth_sigma=2.0, decode_every_n=3,
                          min_cells=10, verbose=True):
    n_cells = temporal_spikes.shape[0]
    n_bins  = len(bin_centers)
    n_laps  = len(lap_starts)
    half_sp = np.mean(np.diff(bin_centers)) / 2
    bin_edges = np.concatenate([[bin_centers[0] - half_sp],
                                 (bin_centers[:-1] + bin_centers[1:]) / 2,
                                 [bin_centers[-1] + half_sp]])

    if np.sum(active_mask) < min_cells:
        if verbose: print(f"  Only {np.sum(active_mask)} active cells — skipping decoding.")
        return None

    binary_all = binarize_activity(temporal_spikes, method=binarize_method,
                                    threshold_sd=threshold_sd)
    event_frac = np.mean(binary_all[active_mask], axis=1)
    cell_idx_all = np.where(active_mask)[0]
    usable = cell_idx_all[event_frac > 0.001]

    if len(usable) < min_cells:
        if verbose: print(f"  Only {len(usable)} cells with events — skipping decoding.")
        return None

    usable_mask = np.zeros(n_cells, dtype=bool)
    usable_mask[usable] = True
    prior = np.ones(n_bins) / n_bins

    all_actual, all_decoded, all_errors = [], [], []
    confusion = np.zeros((n_bins, n_bins), dtype=int)

    if verbose:
        print(f"  Decoding: {len(usable)} cells, {n_laps} laps leave-one-out...")

    for test_lap in range(n_laps):
        train_frames = []
        for i, (s, e) in enumerate(zip(lap_starts, lap_ends)):
            if i != test_lap:
                train_frames.extend(range(s, e))
        train_frames = np.array(train_frames)
        test_frames  = np.arange(lap_starts[test_lap], lap_ends[test_lap])
        if len(test_frames) < 5:
            continue

        tuning, used_idx = compute_decoder_tuning(
            binary_all[:, train_frames], temporal_location[train_frames],
            bin_edges, usable_mask, smooth_sigma=smooth_sigma)

        decode_frames = test_frames[::decode_every_n]
        actual_pos   = temporal_location[decode_frames]
        decoded_pos  = np.zeros(len(decode_frames))

        for fi, frame in enumerate(decode_frames):
            bf = binary_all[used_idx, frame]
            _, db = bayesian_decode_frame(bf, tuning, prior)
            decoded_pos[fi] = bin_centers[db]
            actual_bin = int(np.argmin(np.abs(bin_centers - actual_pos[fi])))
            confusion[actual_bin, db] += 1

        errors = np.abs(decoded_pos - actual_pos)
        all_actual.append(actual_pos)
        all_decoded.append(decoded_pos)
        all_errors.append(errors)

    if not all_errors:
        return None

    errors_flat   = np.concatenate(all_errors)
    chance_error  = (bin_centers[-1] - bin_centers[0]) / 3
    median_error  = float(np.median(errors_flat))
    mean_error    = float(np.mean(errors_flat))

    if verbose:
        print(f"  → median error: {median_error:.2f} cm  (chance ~{chance_error:.1f} cm, "
              f"{(1-median_error/chance_error)*100:.0f}% above chance)")

    return {
        'actual_positions': all_actual,
        'decoded_positions': all_decoded,
        'errors_cm': all_errors,
        'median_error': median_error,
        'mean_error': mean_error,
        'confusion_matrix': confusion,
        'chance_error': chance_error,
        'n_cells_used': len(usable),
        'bin_centers': bin_centers,
    }


def run_shuffle_decoding(temporal_spikes, temporal_location,
                         lap_starts, lap_ends, active_mask, bin_centers,
                         n_shuffles=5, **kwargs):
    n_cells, n_frames = temporal_spikes.shape
    shuffle_errors = []
    print(f"  Decoding shuffle controls ({n_shuffles})...")
    for s in range(n_shuffles):
        shifted = temporal_spikes.copy()
        for ci in np.where(active_mask)[0]:
            shift = np.random.randint(int(0.1 * n_frames), int(0.9 * n_frames))
            shifted[ci] = np.roll(shifted[ci], shift)
        res = run_bayesian_decoding(shifted, temporal_location, lap_starts, lap_ends,
                                    active_mask, bin_centers, verbose=False, **kwargs)
        if res:
            shuffle_errors.append(res['median_error'])
    shuffle_errors = np.array(shuffle_errors)
    if len(shuffle_errors):
        print(f"  → shuffle median: {np.median(shuffle_errors):.2f} ± "
              f"{np.std(shuffle_errors):.2f} cm")
    return shuffle_errors


def plot_decoding_summary(decode_results, shuffle_errors=None, n_example_laps=4):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (A) Confusion matrix
    ax = axes[0, 0]
    cm = decode_results['confusion_matrix'].astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0)
    bc = decode_results['bin_centers']
    ax.imshow(cm_norm, aspect='auto', origin='upper', cmap='hot',
              extent=[bc[0], bc[-1], bc[-1], bc[0]])
    ax.plot([bc[0], bc[-1]], [bc[0], bc[-1]], 'w--', lw=1, alpha=0.5)
    ax.set_xlabel('Decoded position (cm)')
    ax.set_ylabel('Actual position (cm)')
    ax.set_title('Confusion Matrix (normalized per row)', fontweight='bold')

    # (B) Error distribution
    ax = axes[0, 1]
    all_errors = np.concatenate(decode_results['errors_cm'])
    ax.hist(all_errors, bins=40, color='steelblue', edgecolor='white', lw=0.5)
    ax.axvline(decode_results['median_error'], color='k', ls='--',
               label=f"Median={decode_results['median_error']:.1f} cm")
    ax.axvline(decode_results['chance_error'], color='r', ls='--',
               label=f"Chance~{decode_results['chance_error']:.0f} cm")
    ax.set_xlabel('Absolute error (cm)')
    ax.set_ylabel('Frame count')
    ax.set_title('Decoding Error Distribution', fontweight='bold')
    ax.legend(fontsize=10)

    # (C) Shuffle comparison
    ax = axes[1, 0]
    bars = ax.bar(['Decoder', 'Shuffle'],
                  [decode_results['median_error'],
                   float(np.median(shuffle_errors)) if shuffle_errors is not None and len(shuffle_errors) else np.nan],
                  color=['steelblue', 'lightgray'], edgecolor='black', lw=0.8)
    ax.set_ylabel('Median error (cm)')
    ax.set_title('Decoder vs Shuffle Control', fontweight='bold')

    # (D) Example laps
    ax = axes[1, 1]
    cmap_l = plt.cm.tab10(np.linspace(0, 0.9, min(n_example_laps,
                           len(decode_results['actual_positions']))))
    for i, (act, dec) in enumerate(zip(decode_results['actual_positions'][:n_example_laps],
                                        decode_results['decoded_positions'][:n_example_laps])):
        t = np.arange(len(act))
        ax.plot(t, act, color=cmap_l[i], lw=1.5, alpha=0.9, label=f'Lap {i+1} actual')
        ax.plot(t, dec, color=cmap_l[i], lw=1, ls='--', alpha=0.6)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Position (cm)')
    ax.set_title('Actual (solid) vs Decoded (dashed) — example laps', fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    plt.suptitle('Axonal Population Decoding of Position', fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ============================================================================
# ── CORE PER-SESSION FUNCTION ────────────────────────────────────────────────
# ============================================================================

def run_session(data_filepath):
    print(f"\n{'='*80}")
    print(f"  {data_filepath}")
    print(f"{'='*80}")

    # ── Load preproc.h5 ────────────────────────────────────────────────────
    preproc_files = [f for f in os.listdir(data_filepath) if f.endswith('preproc.h5')]
    assert len(preproc_files) > 0, f"No preproc.h5 found in {data_filepath}"
    preproc = files.read_h5(os.path.join(data_filepath, preproc_files[0]))

    spatial_activity      = preproc['spatial_activity']
    bin_centers           = preproc['bin_centers']
    reliable_cells        = preproc['combined_reliable'].astype(bool)
    temporal_spikes       = preproc['smoothed_spks_temporal']
    temporal_location     = preproc['location_cm']
    lap_starts            = preproc['lap_starts'].astype(int)
    lap_ends              = preproc['lap_ends'].astype(int)

    n_cells, n_trials, n_bins = spatial_activity.shape
    n_frames = temporal_spikes.shape[1]
    n_laps   = len(lap_starts)
    print(f"Loaded: {n_cells} ROIs, {n_trials} trials, {n_bins} bins, "
          f"{n_frames} frames, {n_laps} laps")

    # ── Distance-to-reward ─────────────────────────────────────────────────
    lap_max_locs = np.array([np.max(temporal_location[s:e])
                              for s, e in zip(lap_starts, lap_ends)])
    distance_to_reward = np.zeros(n_frames)
    for i, (s, e) in enumerate(zip(lap_starts, lap_ends)):
        distance_to_reward[s:e] = lap_max_locs[i] - temporal_location[s:e]

    # ── Active axons ───────────────────────────────────────────────────────
    mean_act     = np.mean(temporal_spikes, axis=1)
    active_frac  = np.mean(temporal_spikes > 0, axis=1)
    active_axons = (mean_act > 0) & (active_frac > AXON_ACTIVITY_THRESHOLD)
    print(f"Active axons: {np.sum(active_axons)} / {n_cells}")

    save_dir = os.path.join(data_filepath, 'axon_spatial_analysis')
    os.makedirs(save_dir, exist_ok=True)

    # ── 1. Spatial information ─────────────────────────────────────────────
    print("\n[1] Spatial Information")
    si_results = run_spatial_information_analysis_fast(
        temporal_spikes, temporal_location, active_axons, bin_centers,
        n_shuffles=N_SHUFFLES_SI, alpha=ALPHA_SI, smooth_sigma=SMOOTH_SIGMA_SI)

    fig_si = plot_spatial_info_summary(si_results, bin_centers, active_axons,
                                        reliable_cells=reliable_cells)
    fig_si.savefig(os.path.join(save_dir, 'spatial_information_summary.png'),
                   dpi=300, bbox_inches='tight')
    plt.close(fig_si)

    # ── 2. Ramping ─────────────────────────────────────────────────────────
    print("\n[2] Reward-Proximity Ramping")
    ramp_results = run_ramping_analysis(
        temporal_spikes, distance_to_reward, lap_starts, lap_ends,
        active_axons, min_laps=MIN_LAPS_RAMP, alpha=ALPHA_RAMP,
        n_dist_bins=N_DIST_BINS)

    fig_ramp = plot_ramping_summary(ramp_results, active_axons)
    fig_ramp.savefig(os.path.join(save_dir, 'ramping_analysis_summary.png'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig_ramp)

    # ── 3. Decoding ────────────────────────────────────────────────────────
    print("\n[3] Bayesian Population Decoding")
    decode_results = run_bayesian_decoding(
        temporal_spikes, temporal_location, lap_starts, lap_ends,
        active_axons, bin_centers,
        threshold_sd=THRESHOLD_SD_DECODE, smooth_sigma=SMOOTH_SIGMA_DECODE,
        decode_every_n=DECODE_EVERY_N)

    shuffle_errors = None
    if decode_results:
        shuffle_errors = run_shuffle_decoding(
            temporal_spikes, temporal_location, lap_starts, lap_ends,
            active_axons, bin_centers, n_shuffles=N_SHUFFLES_DECODE,
            threshold_sd=THRESHOLD_SD_DECODE, smooth_sigma=SMOOTH_SIGMA_DECODE,
            decode_every_n=DECODE_EVERY_N)
        fig_dec = plot_decoding_summary(decode_results, shuffle_errors)
        fig_dec.savefig(os.path.join(save_dir, 'decoding_summary.png'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig_dec)

    # ── Save HDF5 results ──────────────────────────────────────────────────
    results_path = os.path.join(save_dir, 'axon_spatial_results.h5')
    with h5py.File(results_path, 'w') as f:
        # Spatial info
        si = f.create_group('spatial_information')
        si.create_dataset('tuning_curves',           data=np.nan_to_num(si_results['tuning_curves']))
        si.create_dataset('si_bits_spike',           data=np.nan_to_num(si_results['si_bits_spike']))
        si.create_dataset('si_bits_spike_corrected', data=np.nan_to_num(si_results['si_bits_spike_corrected']))
        si.create_dataset('p_value',                 data=np.nan_to_num(si_results['p_value']))
        si.create_dataset('is_significant',          data=si_results['is_significant'])
        si.create_dataset('occupancy',               data=si_results['occupancy'])
        si.attrs['n_significant'] = int(np.sum(si_results['is_significant']))
        si.attrs['n_active']      = int(np.sum(active_axons))

        # Ramping
        cls_map = {'unanalyzed': 0, 'insufficient_data': 1,
                   'non_ramping': 2, 'ramp_up': 3, 'ramp_down': 4}
        ramp = f.create_group('ramping')
        ramp.create_dataset('classification_int',
                            data=np.array([cls_map.get(c, 0)
                                           for c in ramp_results['classifications']]))
        ramp.create_dataset('p_values',     data=np.nan_to_num(ramp_results['p_values']))
        ramp.create_dataset('mean_slopes',  data=np.nan_to_num(ramp_results['mean_slopes']))
        ramp.create_dataset('profiles',     data=np.nan_to_num(ramp_results['profiles']))
        ramp.create_dataset('dist_centers', data=ramp_results['dist_centers'])
        ramp.attrs['class_mapping'] = str(cls_map)
        ramp.attrs['n_ramp_up']   = int(np.sum(ramp_results['classifications'] == 'ramp_up'))
        ramp.attrs['n_ramp_down'] = int(np.sum(ramp_results['classifications'] == 'ramp_down'))

        # Decoding
        if decode_results:
            dec = f.create_group('decoding')
            dec.create_dataset('confusion_matrix', data=decode_results['confusion_matrix'])
            dec.attrs['median_error']  = decode_results['median_error']
            dec.attrs['mean_error']    = decode_results['mean_error']
            dec.attrs['chance_error']  = decode_results['chance_error']
            dec.attrs['n_cells_used']  = decode_results['n_cells_used']
            if shuffle_errors is not None and len(shuffle_errors):
                dec.create_dataset('shuffle_errors', data=shuffle_errors)
                dec.attrs['shuffle_median'] = float(np.median(shuffle_errors))

        # Metadata
        f.attrs['data_filepath']   = str(data_filepath)
        f.attrs['n_cells_total']   = int(n_cells)
        f.attrs['n_active_axons']  = int(np.sum(active_axons))
        f.create_dataset('active_axons_mask', data=active_axons)
        f.create_dataset('bin_centers',       data=bin_centers)

    print(f"\n✓ Results → {results_path}")
    print(f"✓ Figures → {save_dir}")

    # ── Print summary ──────────────────────────────────────────────────────
    n_act   = int(np.sum(active_axons))
    n_si    = int(np.sum(si_results['is_significant'] & active_axons))
    n_up    = int(np.sum(ramp_results['classifications'] == 'ramp_up'))
    n_down  = int(np.sum(ramp_results['classifications'] == 'ramp_down'))

    return {
        'data_filepath': data_filepath,
        'n_cells': n_cells,
        'n_active': n_act,
        'n_si_sig': n_si,
        'n_ramp_up': n_up,
        'n_ramp_down': n_down,
        'decode_median_error': decode_results['median_error'] if decode_results else np.nan,
    }


# ============================================================================
# ── BATCH LOOP ───────────────────────────────────────────────────────────────
# ============================================================================

if __name__ == "__main__":

    successful, failed = [], []

    for data_filepath in DATA_FILEPATHS:
        results_path = os.path.join(data_filepath, 'axon_spatial_analysis',
                                    'axon_spatial_results.h5')
        if SKIP_IF_DONE and os.path.exists(results_path):
            print(f"[SKIP] Already done: {data_filepath}")
            continue

        preproc_files = [f for f in os.listdir(data_filepath)
                         if f.endswith('preproc.h5')] if os.path.isdir(data_filepath) else []
        if not preproc_files:
            print(f"[SKIP] No preproc.h5 found: {data_filepath}")
            continue

        try:
            r = run_session(data_filepath)
            successful.append(r)
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {data_filepath}\n{traceback.format_exc()}")
            failed.append((data_filepath, str(e)))

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"BATCH COMPLETE  —  {len(successful)} succeeded  /  {len(failed)} failed")
    print(f"{'='*80}")

    if successful:
        print(f"\n{'Session':<45} {'active':>7} {'SI-sig':>7} {'ramp↑':>6} {'ramp↓':>6} {'dec-err':>8}")
        print("-" * 80)
        for r in successful:
            label = os.path.basename(r['data_filepath'])
            de    = f"{r['decode_median_error']:.1f} cm" if not np.isnan(r['decode_median_error']) else "  N/A"
            print(f"{label:<45} {r['n_active']:>7} {r['n_si_sig']:>7} "
                  f"{r['n_ramp_up']:>6} {r['n_ramp_down']:>6} {de:>8}")

    if failed:
        print("\nFailed:")
        for path, err in failed:
            print(f"  {os.path.basename(path)}: {err}")
