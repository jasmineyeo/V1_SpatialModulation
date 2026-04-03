"""
SpatialCode_Evolution.py
Tracks how the STRUCTURE of the spatial code evolves for the same neurons.

Three analyses:
  1. Preferred position stability — does the place field stay in the same
                                    location, or remap across sessions?
  2. Landmark preference evolution — which landmark does each cell anchor to,
                                     and does that stabilize over learning?
  3. Field width evolution         — does spatial tuning sharpen over sessions?

Requires:
  - *_smi_results.h5    (preferred positions)
  - *_preproc*.h5       (norm_spatial_activity for landmark preference
                         and field width)

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from load_tracked import (
    load_tracking, filter_to_analysis_days, find_smi_files, find_preproc_files,
    find_files_from_tracking, assign_layers_from_smi, load_smi_session,
    load_preproc_session, build_matrix, parse_day_numbers, layer_mean_sem,
    animal_id_from_path, LAYER_ORDER, LAYER_COLORS, report_found_files,
)

# ============================================================
# CONFIGURATION
# ============================================================


ROI_TRACKING_FILE = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\TrackedROIs\roi_tracking_results.h5"
ANIMAL_DIR        = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"
REFERENCE_DAY     = "Day2"
ANALYSIS_DAYS     = ['Day2','Day3','Day4','Day5','Day6','Day7']    # e.g. ['Day2','Day3','Day4','Day5','Day6','Day7']
                             # None = use all tracked sessions

# Landmark positions in cm (VR corridor) — adjust to match your setup
LANDMARK_POSITIONS = [25, 55, 85, 115]
LANDMARK_COLORS    = ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3']

# Landmark window: ± cm around each landmark centre
LANDMARK_WINDOW_CM = 12.0

# Onset/reward exclusion for landmark preference (bins from each end)
EXCLUDE_FIRST_BINS = 5
EXCLUDE_LAST_BINS  = 5

# Gaussian smoothing sigma for field-width estimation (bins)
SMOOTH_SIGMA = 1.0

# check whether the fig_dir exists, if not create it
OUTPUT_DIR = os.path.join(ANIMAL_DIR, "TrackedROIs")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================


# ── Analysis 1: Preferred position stability ─────────────────

def _extract_pref_pos(smi_file, roi_indices):
    _, _, _, _, pref_pos, _ = load_smi_session(smi_file)
    out = np.full(len(roi_indices), np.nan)
    for i, roi in enumerate(roi_indices):
        if roi >= 0 and roi < len(pref_pos):
            out[i] = pref_pos[roi]
    return out


def compute_position_drift(pref_pos_matrix, cell_layers, reference_col=0):
    """
    For each tracked cell, measure |drift| = |pref_pos(session) - pref_pos(reference)|.

    Returns
    -------
    drift_matrix : (n_tracked, n_sessions) — NaN where no valid position
    layer_drift  : dict {layer: layer-slice of drift_matrix}
    """
    ref = pref_pos_matrix[:, reference_col:reference_col+1]  # keep shape
    drift = np.abs(pref_pos_matrix - ref)
    drift[:, reference_col] = 0.0   # drift from itself = 0

    layer_drift = {}
    for layer, rows in cell_layers.items():
        layer_drift[layer] = drift[rows, :]

    return drift, layer_drift


def plot_position_drift(layer_stats_drift, session_days, day_labels, animal_id,
                        output_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle(f"{animal_id} — Preferred Position Stability", fontweight='bold')

    x = np.array(session_days)

    # Left: mean |drift| per layer
    ax = axes[0]
    for layer in LAYER_ORDER:
        if layer not in layer_stats_drift:
            continue
        ls = layer_stats_drift[layer]
        c  = LAYER_COLORS[layer]
        ax.plot(x, ls['mean'], 'o-', color=c, label=layer, linewidth=2, markersize=5)
        ax.fill_between(x, ls['mean'] - ls['sem'], ls['mean'] + ls['sem'],
                        color=c, alpha=0.2)
    ax.set_xlabel('Recording Day')
    ax.set_ylabel('Mean |drift| from Day 1 (cm)')
    ax.set_title('Place Field Drift')
    ax.set_xticks(x)
    ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: % cells with drift < 10 cm (stable cells)
    ax2 = axes[1]
    # Re-compute from raw 'all' arrays
    for layer in LAYER_ORDER:
        if layer not in layer_stats_drift:
            continue
        raw = layer_stats_drift[layer]['all']   # (n_layer_cells, n_sessions)
        prop_stable = np.nanmean(raw < 10.0, axis=0) * 100
        ax2.plot(x, prop_stable, 's--', color=LAYER_COLORS[layer], label=layer,
                 linewidth=1.5, markersize=5)
    ax2.set_xlabel('Recording Day')
    ax2.set_ylabel('% cells with |drift| < 10 cm')
    ax2.set_title('Proportion with Stable Place Field')
    ax2.set_ylim(0, 105)
    ax2.set_xticks(x)
    ax2.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_pref_pos_heatmap(pref_pos_matrix, cell_layers, session_days, day_labels,
                           animal_id, output_path=None):
    """Heatmap: each row = tracked cell, colour = preferred position (cm).
       Sorted by reference-session preferred position within each layer."""
    sorted_rows, layer_bounds = [], {}
    current = 0
    for layer in LAYER_ORDER:
        if layer not in cell_layers:
            continue
        rows = cell_layers[layer]
        ref_pos = pref_pos_matrix[rows, 0]
        order = np.argsort(np.where(np.isnan(ref_pos), np.inf, ref_pos))
        sorted_rows.extend(rows[order].tolist())
        layer_bounds[layer] = (current, current + len(rows))
        current += len(rows)

    if not sorted_rows:
        return None

    data = pref_pos_matrix[sorted_rows, :]
    fig, ax = plt.subplots(figsize=(max(6, len(session_days) * 1.2), 8))
    im = ax.imshow(data, aspect='auto', cmap='viridis', interpolation='nearest')

    for layer, (start, end) in layer_bounds.items():
        if start > 0:
            ax.axhline(start - 0.5, color='white', linewidth=1.5)
        mid = (start + end) / 2
        ax.text(-0.6, mid, layer, ha='right', va='center', fontsize=9,
                fontweight='bold', color=LAYER_COLORS.get(layer, 'k'),
                transform=ax.get_yaxis_transform())

    ax.set_xticks(range(len(session_days)))
    ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=9)
    ax.set_title(f"{animal_id} — Preferred Position Heatmap", fontweight='bold')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04).set_label('Preferred position (cm)')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Analysis 2: Landmark preference per cell ─────────────────

def get_preferred_landmark_per_cell(norm_spatial_activity, bin_centers,
                                     landmark_positions, landmark_window_cm,
                                     exclude_first_bins=10, exclude_last_bins=10,
                                     smooth_sigma=1.0):
    """
    For each cell, determine which landmark it prefers based on where
    its mean spatial response peaks.

    Returns
    -------
    preferred_lm : (n_cells,) int  — landmark index (0-based), -1 = invalid
    valid        : (n_cells,) bool
    pref_strength: (n_cells,) float — response at preferred lm − mean of others
    """
    n_cells, n_trials, n_bins = norm_spatial_activity.shape
    n_lm = len(landmark_positions)

    bin_spacing = np.mean(np.diff(bin_centers))
    onset_thresh = bin_centers[0]  + exclude_first_bins * bin_spacing
    end_thresh   = bin_centers[-1] - exclude_last_bins  * bin_spacing

    # Windows [lm - window, lm + window]
    windows = [(lp - landmark_window_cm, lp + landmark_window_cm)
               for lp in landmark_positions]

    mean_profiles = np.mean(norm_spatial_activity, axis=1)  # (n_cells, n_bins)
    if smooth_sigma > 0:
        for c in range(n_cells):
            mean_profiles[c] = gaussian_filter1d(mean_profiles[c], sigma=smooth_sigma)

    preferred_lm  = np.full(n_cells, -1, dtype=int)
    valid         = np.zeros(n_cells, dtype=bool)
    pref_strength = np.zeros(n_cells, dtype=float)
    lm_responses  = np.zeros((n_cells, n_lm), dtype=float)

    for cell in range(n_cells):
        profile = mean_profiles[cell]
        peak_idx = np.argmax(profile)
        peak_pos = bin_centers[peak_idx]

        if profile[peak_idx] == 0:
            continue
        if peak_pos < onset_thresh or peak_pos > end_thresh:
            continue

        # Response in each landmark window
        for lm_idx, (wmin, wmax) in enumerate(windows):
            mask = (bin_centers >= wmin) & (bin_centers <= wmax)
            if mask.any():
                lm_responses[cell, lm_idx] = np.max(profile[mask])

        # Does the global peak fall inside any window?
        in_window = [(lm_idx, lm_responses[cell, lm_idx])
                     for lm_idx, (wmin, wmax) in enumerate(windows)
                     if wmin <= peak_pos <= wmax]
        if not in_window:
            continue

        best_lm, best_resp = max(in_window, key=lambda x: x[1])
        others = [lm_responses[cell, i] for i in range(n_lm) if i != best_lm]
        pref_strength[cell] = best_resp - (np.mean(others) if others else 0)
        preferred_lm[cell] = best_lm
        valid[cell] = True

    return preferred_lm, valid, pref_strength


def build_landmark_matrices(tracked_matrix, day_labels, preproc_files,
                             landmark_positions, landmark_window_cm,
                             exclude_first_bins, exclude_last_bins, smooth_sigma):
    """
    Returns
    -------
    lm_matrix  : (n_tracked, n_sessions) int   — preferred landmark index, -1=invalid
    valid_matrix: (n_tracked, n_sessions) bool
    strength_matrix: (n_tracked, n_sessions) float
    """
    n_cells, n_sessions = tracked_matrix.shape
    lm_matrix  = np.full((n_cells, n_sessions), -1, dtype=int)
    valid_mat  = np.zeros((n_cells, n_sessions), dtype=bool)
    str_matrix = np.zeros((n_cells, n_sessions), dtype=float)

    for col, day in enumerate(day_labels):
        if day not in preproc_files:
            continue
        print(f"  Computing landmark preference: {day}...", end=' ', flush=True)
        _, nsa, bc, _, _, _, _ = load_preproc_session(preproc_files[day])

        # Scale bin centres (matching preproc convention)
        shifted = bc - np.min(bc)
        scaled  = shifted * (len(bc) / np.max(shifted)) if np.max(shifted) > 0 else shifted

        pref_lm, valid, strength = get_preferred_landmark_per_cell(
            nsa, scaled, landmark_positions, landmark_window_cm,
            exclude_first_bins, exclude_last_bins, smooth_sigma)

        for row, roi in enumerate(tracked_matrix[:, col]):
            if roi >= 0 and roi < len(pref_lm):
                lm_matrix[row, col]  = pref_lm[roi]
                valid_mat[row, col]  = valid[roi]
                str_matrix[row, col] = strength[roi]
        print(f"done ({np.sum(valid_mat[:, col])} valid cells)")

    return lm_matrix, valid_mat, str_matrix


def compute_landmark_stability(lm_matrix, valid_matrix, cell_layers):
    """
    For each tracked cell with valid landmark preference in consecutive sessions,
    compute whether the preferred landmark is the SAME (stability = 1) or different (0).
    Returns stability (proportion same) per layer per session transition.
    """
    n_sessions = lm_matrix.shape[1]
    stability = {}

    for layer, rows in cell_layers.items():
        lm  = lm_matrix[rows, :]
        val = valid_matrix[rows, :]
        per_transition = []
        for s in range(n_sessions - 1):
            both_valid = val[:, s] & val[:, s + 1]
            if np.sum(both_valid) == 0:
                per_transition.append(np.nan)
                continue
            same = lm[both_valid, s] == lm[both_valid, s + 1]
            per_transition.append(np.mean(same) * 100)
        stability[layer] = np.array(per_transition)

    return stability


def plot_landmark_evolution(lm_matrix, valid_matrix, cell_layers,
                             session_days, day_labels, animal_id,
                             n_landmarks, output_path=None):
    """
    For each layer: stacked area/line plot of proportion of cells
    preferring each landmark across sessions.
    """
    layers = [l for l in LAYER_ORDER if l in cell_layers and len(cell_layers[l]) > 0]
    n = len(layers)
    if n == 0:
        return None

    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle(f"{animal_id} — Landmark Preference Evolution by Layer",
                 fontweight='bold', fontsize=11)

    x = np.array(session_days)
    lm_colors = LANDMARK_COLORS[:n_landmarks]

    for col, layer in enumerate(layers):
        rows = cell_layers[layer]
        lm   = lm_matrix[rows, :]
        val  = valid_matrix[rows, :]

        # Proportions per session
        props = np.zeros((n_landmarks, len(session_days)))
        for s in range(len(session_days)):
            v = val[:, s]
            if np.sum(v) == 0:
                continue
            for lm_i in range(n_landmarks):
                props[lm_i, s] = np.mean(lm[v, s] == lm_i) * 100

        # Top row: line plot per landmark
        ax0 = axes[0, col]
        for lm_i in range(n_landmarks):
            ax0.plot(x, props[lm_i], 'o-', color=lm_colors[lm_i],
                     label=f'LM{lm_i+1}', linewidth=2, markersize=5)
        ax0.set_title(layer, fontweight='bold', color=LAYER_COLORS[layer])
        ax0.set_ylabel('% cells' if col == 0 else '')
        ax0.set_ylim(0, 100)
        ax0.set_xticks(x); ax0.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=7)
        ax0.legend(fontsize=7); ax0.grid(True, alpha=0.2, axis='y')
        if col == 0:
            ax0.set_title(f"{layer}\nLandmark proportions", fontsize=9,
                          color=LAYER_COLORS[layer])

        # Bottom row: stability (% same landmark consecutive sessions)
        ax1 = axes[1, col]
        stab = []
        for s in range(len(session_days) - 1):
            both = val[:, s] & val[:, s + 1]
            if np.sum(both) == 0:
                stab.append(np.nan)
            else:
                stab.append(np.mean(lm[both, s] == lm[both, s + 1]) * 100)
        x_mid = (x[:-1] + x[1:]) / 2
        ax1.bar(x_mid, stab, width=np.diff(x) * 0.5, color=LAYER_COLORS[layer], alpha=0.7)
        ax1.set_ylabel('% same landmark' if col == 0 else '')
        ax1.set_ylim(0, 100)
        ax1.axhline(100 / n_landmarks, color='gray', linestyle=':', alpha=0.6,
                    label='Chance')
        ax1.set_xticks(x_mid)
        mid_labels = [f"{day_labels[i]}→{day_labels[i+1]}"
                      for i in range(len(day_labels)-1)]
        ax1.set_xticklabels(mid_labels, rotation=45, ha='right', fontsize=7)
        ax1.set_title('Stability\n(consecutive sessions)', fontsize=9)
        ax1.legend(fontsize=7); ax1.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Analysis 3: Field width evolution ────────────────────────

def compute_field_width(spatial_activity, bin_centers, smooth_sigma=1.0):
    """
    Estimate half-width at half-maximum (HWHM) of mean tuning curve per cell.

    Returns
    -------
    widths : (n_cells,) float — field width in cm, NaN if cannot be computed
    """
    n_cells = spatial_activity.shape[0]
    mean_profiles = np.mean(spatial_activity, axis=1)   # (n_cells, n_bins)
    bin_size = np.mean(np.diff(bin_centers))
    widths = np.full(n_cells, np.nan)

    for cell in range(n_cells):
        profile = gaussian_filter1d(mean_profiles[cell].astype(float), sigma=smooth_sigma)
        peak    = np.max(profile)
        if peak <= 0:
            continue
        half = peak / 2.0
        above = profile >= half
        if not above.any():
            continue
        # Find contiguous regions above half-max — use the largest one
        transitions = np.diff(above.astype(int))
        starts = np.where(transitions == 1)[0] + 1
        ends   = np.where(transitions == -1)[0] + 1
        # Handle edge cases
        if above[0]:
            starts = np.concatenate([[0], starts])
        if above[-1]:
            ends = np.concatenate([ends, [n_cells]])
        if len(starts) == 0 or len(ends) == 0:
            continue
        # Largest region
        lengths = ends[:len(starts)] - starts[:len(ends)]
        best = np.argmax(lengths)
        widths[cell] = lengths[best] * bin_size

    return widths


def build_field_width_matrix(tracked_matrix, day_labels, preproc_files,
                              smooth_sigma=1.0):
    n_cells, n_sessions = tracked_matrix.shape
    fw_matrix = np.full((n_cells, n_sessions), np.nan)

    for col, day in enumerate(day_labels):
        if day not in preproc_files:
            continue
        print(f"  Computing field widths: {day}...", end=' ', flush=True)
        sa, _, bc, rel, _, _, _ = load_preproc_session(preproc_files[day])
        widths = compute_field_width(sa, bc, smooth_sigma)

        for row, roi in enumerate(tracked_matrix[:, col]):
            if roi >= 0 and roi < len(widths):
                # Only include reliable cells
                fw_matrix[row, col] = widths[roi] if rel[roi] else np.nan
        print(f"done")

    return fw_matrix


def plot_field_width(layer_stats_fw, session_days, day_labels, animal_id,
                     output_path=None):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.array(session_days)

    for layer in LAYER_ORDER:
        if layer not in layer_stats_fw:
            continue
        ls = layer_stats_fw[layer]
        c  = LAYER_COLORS[layer]
        n  = int(np.nanmax(ls['n']))
        ax.plot(x, ls['mean'], 'o-', color=c, label=f"{layer} (n={n})",
                linewidth=2, markersize=5)
        ax.fill_between(x, ls['mean'] - ls['sem'], ls['mean'] + ls['sem'],
                        color=c, alpha=0.2)

    ax.set_xlabel('Recording Day')
    ax.set_ylabel('Mean field width — HWHM (cm)')
    ax.set_title(f"{animal_id} — Place Field Width Over Sessions", fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(day_labels, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("SPATIAL CODE EVOLUTION — TRACKED ROIs")
    print("=" * 65)

    animal_id = animal_id_from_path(ROI_TRACKING_FILE)
    out_dir   = OUTPUT_DIR or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"{animal_id}_SpatialCode")

    # Load
    print("\n[1] Loading tracking matrix...")
    tracked_matrix, day_labels, session_dirs = load_tracking(ROI_TRACKING_FILE)
    tracked_matrix, day_labels, session_dirs = filter_to_analysis_days(
        tracked_matrix, day_labels, session_dirs, ANALYSIS_DAYS)
    session_days = parse_day_numbers(day_labels)

    print("\n[2] Finding data files...")
    if session_dirs:
        smi_files     = find_files_from_tracking(session_dirs, "*_smi_results.h5")
        preproc_files = find_files_from_tracking(session_dirs, "*_preproc*.h5")
    else:
        smi_files     = find_smi_files(ANIMAL_DIR)
        preproc_files = find_preproc_files(ANIMAL_DIR)
    report_found_files("SMI files",     smi_files,     day_labels)
    report_found_files("Preproc files", preproc_files, day_labels)

    # Layer assignment
    print("\n[3] Assigning cells to layers...")
    cell_layers = assign_layers_from_smi(tracked_matrix, day_labels, smi_files,
                                          REFERENCE_DAY)
    for l in LAYER_ORDER:
        print(f"  {l}: {len(cell_layers.get(l, []))} tracked cells")

    # ── Analysis 1: Preferred position ──────────────────────
    print("\n[4] Preferred position stability...")
    pref_pos_matrix = build_matrix(tracked_matrix, day_labels, smi_files,
                                    _extract_pref_pos)
    drift_matrix, _ = compute_position_drift(pref_pos_matrix, cell_layers,
                                              reference_col=0)
    drift_stats = layer_mean_sem(drift_matrix, cell_layers)

    plot_position_drift(drift_stats, session_days, day_labels, animal_id,
                        output_path=f"{prefix}_position_drift.png")
    plot_pref_pos_heatmap(pref_pos_matrix, cell_layers, session_days, day_labels,
                           animal_id, output_path=f"{prefix}_position_heatmap.png")

    # ── Analysis 2: Landmark preference ─────────────────────
    print("\n[5] Landmark preference evolution...")
    lm_matrix, valid_mat, str_mat = build_landmark_matrices(
        tracked_matrix, day_labels, preproc_files,
        LANDMARK_POSITIONS, LANDMARK_WINDOW_CM,
        EXCLUDE_FIRST_BINS, EXCLUDE_LAST_BINS, SMOOTH_SIGMA)

    plot_landmark_evolution(lm_matrix, valid_mat, cell_layers,
                             session_days, day_labels, animal_id,
                             n_landmarks=len(LANDMARK_POSITIONS),
                             output_path=f"{prefix}_landmark_evolution.png")

    # ── Analysis 3: Field width ──────────────────────────────
    print("\n[6] Field width evolution...")
    fw_matrix  = build_field_width_matrix(tracked_matrix, day_labels,
                                           preproc_files, SMOOTH_SIGMA)
    fw_stats   = layer_mean_sem(fw_matrix, cell_layers)
    plot_field_width(fw_stats, session_days, day_labels, animal_id,
                     output_path=f"{prefix}_field_width.png")

    plt.show()
    print(f"\nDone. Outputs in: {out_dir}")

    return dict(tracked_matrix=tracked_matrix, pref_pos_matrix=pref_pos_matrix,
                drift_matrix=drift_matrix, lm_matrix=lm_matrix,
                valid_mat=valid_mat, fw_matrix=fw_matrix,
                cell_layers=cell_layers, day_labels=day_labels,
                session_days=session_days)


if __name__ == "__main__":
    main()
