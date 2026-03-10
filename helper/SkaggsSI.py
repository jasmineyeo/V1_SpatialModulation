"""
SkaggsSI.py
Skaggs spatial information analysis for axonal calcium imaging data.
Circular shuffle-based significance testing with bias correction.

JSY, 03/2026
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _compute_occupancy(location, bin_edges):
    counts, _ = np.histogram(location, bins=bin_edges)
    total = np.sum(counts)
    return counts / total if total > 0 else np.zeros(len(bin_edges) - 1)


def _skaggs_si_batch(tuning_curves, occupancy, epsilon=1e-10):
    """Skaggs SI (bits/spike) for all cells. Returns (si, mean_rates)."""
    n_cells = tuning_curves.shape[0]
    si = np.zeros(n_cells)
    mean_rates = np.zeros(n_cells)
    p_i = occupancy
    for ci in range(n_cells):
        tc = tuning_curves[ci]
        pos_mask = (p_i > 0) & (~np.isnan(tc)) & (tc > epsilon)
        if np.sum(pos_mask) < 2:
            si[ci] = np.nan
            mean_rates[ci] = np.nan
            continue
        mr = np.sum(p_i[pos_mask] * tc[pos_mask])
        mean_rates[ci] = mr
        if mr <= epsilon:
            continue
        ratio = tc[pos_mask] / mr
        si[ci] = np.sum(p_i[pos_mask] * ratio * np.log2(ratio + epsilon))
    return si, mean_rates


def _circular_shuffle_si_fast(activity_2d, location, bin_edges, occupancy,
                               n_shuffles=200, min_shift_frac=0.1):
    """Circular-shift location (not activity) to generate null SI distribution."""
    n_cells, n_frames = activity_2d.shape
    n_bins = len(bin_edges) - 1
    min_shift = max(1, int(n_frames * min_shift_frac))
    max_shift = n_frames - min_shift
    shifts = np.random.randint(min_shift, max_shift, size=n_shuffles)

    shuffle_si = np.zeros((n_cells, n_shuffles))
    for s_idx, shift in enumerate(shifts):
        loc_shuf = np.roll(location, shift)
        tc_shuf = np.zeros((n_cells, n_bins))
        for b in range(n_bins):
            mask = (loc_shuf >= bin_edges[b]) & (loc_shuf < bin_edges[b + 1])
            if np.sum(mask) > 0:
                tc_shuf[:, b] = np.mean(activity_2d[:, mask], axis=1)
        si_shuf, _ = _skaggs_si_batch(tc_shuf, occupancy)
        shuffle_si[:, s_idx] = si_shuf
    return shuffle_si


def run_spatial_information_analysis(temporal_spikes, temporal_location,
                                     active_mask, bin_centers,
                                     n_shuffles=200, alpha=0.05,
                                     smooth_sigma=1.5, verbose=True):
    """
    Skaggs SI analysis with circular shuffle significance testing.

    Parameters
    ----------
    temporal_spikes   : (n_cells, n_frames)
    temporal_location : (n_frames,)  — position in cm
    active_mask       : (n_cells,) bool — cells to analyze (e.g. active_axons)
    bin_centers       : (n_bins,) — spatial bin centers in cm
    n_shuffles        : int
    alpha             : float — one-tailed significance threshold
    smooth_sigma      : float — Gaussian smoothing sigma (bins) for tuning curves
    verbose           : bool

    Returns
    -------
    dict with keys (all length n_cells, NaN for inactive cells):
        si_bits_spike             — raw Skaggs SI (bits/spike)
        si_bits_spike_corrected   — bias-corrected SI (observed - mean shuffle)
        is_significant            — bool, p < alpha
        p_value                   — fraction of shuffles >= observed
        shuffle_mean              — mean of shuffle SI (bias estimate)
        mean_rate                 — mean firing rate per cell
    """
    n_cells = temporal_spikes.shape[0]
    n_bins = len(bin_centers)
    bin_spacing = np.mean(np.diff(bin_centers))
    bin_edges = np.append(bin_centers - bin_spacing / 2,
                          bin_centers[-1] + bin_spacing / 2)

    active_idx = np.where(active_mask)[0]
    n_active = len(active_idx)
    spikes_active = temporal_spikes[active_idx]

    if verbose:
        print(f"  Skaggs SI: {n_active} cells, {n_shuffles} shuffles, alpha={alpha}")

    occupancy = _compute_occupancy(temporal_location, bin_edges)

    # ── Tuning curves (smoothed) ──────────────────────────────────────────────
    if verbose:
        print("  Step 1/3: Tuning curves...")
    tuning_raw = np.zeros((n_active, n_bins))
    for b in range(n_bins):
        mask = (temporal_location >= bin_edges[b]) & (temporal_location < bin_edges[b + 1])
        if np.sum(mask) > 0:
            tuning_raw[:, b] = np.mean(spikes_active[:, mask], axis=1)

    tuning_smooth = np.zeros_like(tuning_raw)
    for i in range(n_active):
        tc = tuning_raw[i]
        nan_mask = np.isnan(tc)
        tc_filled = np.where(nan_mask, 0.0, tc)
        tc_s = gaussian_filter1d(tc_filled, sigma=smooth_sigma)
        tc_s[nan_mask] = np.nan
        tuning_smooth[i] = tc_s

    # ── Observed SI ───────────────────────────────────────────────────────────
    if verbose:
        print("  Step 2/3: Observed SI...")
    si_observed, mean_rates = _skaggs_si_batch(tuning_smooth, occupancy)

    # ── Circular shuffle ──────────────────────────────────────────────────────
    if verbose:
        print(f"  Step 3/3: {n_shuffles} circular shuffles...")
    shuffle_si = _circular_shuffle_si_fast(
        spikes_active, temporal_location, bin_edges, occupancy,
        n_shuffles=n_shuffles
    )

    # ── Significance + bias correction ───────────────────────────────────────
    si_corrected = np.zeros(n_active)
    shuffle_mean_arr = np.zeros(n_active)
    p_values = np.ones(n_active)

    for i in range(n_active):
        obs = si_observed[i]
        shuf = shuffle_si[i]
        shuf_clean = shuf[~np.isnan(shuf)]
        if len(shuf_clean) == 0 or np.isnan(obs):
            si_corrected[i] = np.nan
            shuffle_mean_arr[i] = np.nan
            continue
        p_values[i] = np.mean(shuf_clean >= obs)
        shuffle_mean_arr[i] = np.mean(shuf_clean)
        si_corrected[i] = obs - shuffle_mean_arr[i]

    is_significant = p_values < alpha

    if verbose:
        n_sig = np.sum(is_significant)
        print(f"  Significant: {n_sig}/{n_active} ({100*n_sig/n_active:.1f}%)")

    # ── Expand back to full n_cells arrays ────────────────────────────────────
    def _expand(arr, fill=np.nan):
        out = np.full(n_cells, fill)
        out[active_idx] = arr
        return out

    return {
        'si_bits_spike':           _expand(si_observed),
        'si_bits_spike_corrected': _expand(si_corrected),
        'is_significant':          _expand(is_significant, fill=False).astype(bool),
        'p_value':                 _expand(p_values, fill=1.0),
        'shuffle_mean':            _expand(shuffle_mean_arr),
        'mean_rate':               _expand(mean_rates),
    }
