"""
LandmarkPreference_OpenLoopVR_Analysis.py

OpenLoopVR-specific landmark preference analysis.

Key difference from LandmarkPrefernce_SingleSessionAnalysis.py:
  The standard pipeline rejects any cell whose *global* peak falls in the
  onset/reward bins.  In open-loop sessions cells with a genuine landmark
  response can also have a strong onset transient, so the standard approach
  drops them entirely.

  Here the onset/reward bins are *masked* before peak-finding.  The peak is
  searched only in the eligible (mid-corridor) region.  A cell is rejected
  only if it has *no* activity outside the onset/reward zone — i.e. it is
  truly onset-only or reward-only.

Everything else (layer analysis, dynamics, plotting, saving) is unchanged
and imported directly from the parent script.

JSY, 05/2026
"""

import sys
sys.path.insert(0, r'C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation')
sys.path.insert(0, r'C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation\3.LandmarkPreference')

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Import everything from the parent script so the rest of the pipeline is
# identical and the notebook only needs to change its import line.
from LandmarkPrefernce_SingleSessionAnalysis import (
    analyze_layer_landmark_preferences,
    analyze_within_session_dynamics,
    save_session_landmark_data,
    plot_layer_landmark_heatmap,
    plot_within_session_dynamics,
    plot_example_cells_by_landmark,
    plot_rejected_cells_response,
    plot_cells_by_landmark_assignment,
    plot_landmark_assignment_summary,
)


# ============================================================================
# OVERRIDDEN PHASE 1 — masked-onset peak finding
# ============================================================================

def identify_landmark_responses(normalized_spatial_activity, bin_centers,
                                landmark_positions,
                                landmark_windows_config=None,
                                landmark_window=10.0,
                                boundary_exclusion=(10, 10),
                                smoothing_sigma=1.0,
                                exclude_first_bins=10,
                                exclude_last_bins=10):
    """
    Identify landmark-preferring cells using masked peak-finding.

    Onset/reward bins are zeroed before searching for the preferred position.
    A cell is rejected only if it has no activity in the eligible
    (mid-corridor) region — not merely because its global peak happens to
    fall in the onset or reward zone.

    Parameters mirror LandmarkPrefernce_SingleSessionAnalysis.identify_landmark_responses
    exactly so the notebook call-site does not need to change.
    """
    n_cells, n_trials, n_bins = normalized_spatial_activity.shape
    n_landmarks = len(landmark_positions)

    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)

    start_exclude, end_exclude = boundary_exclusion
    min_allowed = min_pos + start_exclude
    max_allowed = max_pos - end_exclude

    bin_spacing = np.mean(np.diff(bin_centers))
    onset_threshold_cm = min_pos + (exclude_first_bins * bin_spacing)
    end_threshold_cm   = max_pos - (exclude_last_bins  * bin_spacing)

    print(f"\n=== LANDMARK PREFERENCE IDENTIFICATION (OpenLoopVR — masked onset) ===")
    print(f"Corridor: {min_pos:.1f} to {max_pos:.1f} cm  ({n_bins} bins, {bin_spacing:.2f} cm/bin)")
    print(f"Eligible region for peak: {onset_threshold_cm:.1f} – {end_threshold_cm:.1f} cm")
    print(f"  (onset bins masked: first {exclude_first_bins} bins < {onset_threshold_cm:.1f} cm)")
    print(f"  (reward bins masked: last  {exclude_last_bins} bins > {end_threshold_cm:.1f} cm)")
    print(f"Landmarks at: {landmark_positions} cm")

    # Build per-landmark windows
    landmark_windows = []
    if landmark_windows_config is not None:
        print(f"Using per-landmark window configuration:")
        for i, lm_pos in enumerate(landmark_positions):
            if i < len(landmark_windows_config):
                cfg = landmark_windows_config[i]
                lm_min = lm_pos - cfg['before']
                lm_max = lm_pos + cfg['after']
                print(f"  L{i+1} at {lm_pos} cm: [{lm_min:.1f}, {lm_max:.1f}] cm")
            else:
                lm_min = lm_pos - landmark_window
                lm_max = lm_pos + landmark_window
                print(f"  L{i+1} at {lm_pos} cm: [{lm_min:.1f}, {lm_max:.1f}] cm (symmetric fallback)")
            landmark_windows.append((lm_min, lm_max))
    else:
        print(f"Using symmetric windows: ±{landmark_window} cm")
        for i, lm_pos in enumerate(landmark_positions):
            landmark_windows.append((lm_pos - landmark_window, lm_pos + landmark_window))

    # Mean response across laps, optionally smoothed
    mean_profiles = np.mean(normalized_spatial_activity, axis=1)
    if smoothing_sigma > 0:
        for cell in range(n_cells):
            mean_profiles[cell] = gaussian_filter1d(mean_profiles[cell], sigma=smoothing_sigma)

    # Eligible-region mask (reused each cell)
    eligible_mask = (bin_centers >= onset_threshold_cm) & (bin_centers <= end_threshold_cm)

    # Outputs
    preferred_landmark  = np.full(n_cells, -1, dtype=int)
    landmark_responses  = np.zeros((n_cells, n_landmarks))
    preference_strength = np.zeros(n_cells)
    peak_positions      = np.zeros(n_cells)   # global peak (for reference)
    global_peak_bins    = np.zeros(n_cells, dtype=int)
    valid_cells         = np.zeros(n_cells, dtype=bool)

    rejected_onset_indices      = []
    rejected_reward_indices     = []
    rejected_no_landmark_indices = []
    rejected_zero_indices       = []

    for cell in range(n_cells):
        profile = mean_profiles[cell]

        # Store the true global peak position (informational only)
        global_peak_idx = np.argmax(profile)
        global_peak_pos = bin_centers[global_peak_idx]
        global_peak_bins[cell] = global_peak_idx
        peak_positions[cell]   = global_peak_pos

        if profile[global_peak_idx] == 0:
            rejected_zero_indices.append(cell)
            continue

        # --- Masked peak: search only in eligible region ---
        masked = profile.copy()
        masked[~eligible_mask] = 0.0

        eligible_peak_idx = np.argmax(masked)
        eligible_peak_val = masked[eligible_peak_idx]
        eligible_peak_pos = bin_centers[eligible_peak_idx]

        if eligible_peak_val == 0:
            # No activity in the eligible region → true onset-only / reward-only
            if global_peak_pos < onset_threshold_cm:
                rejected_onset_indices.append(cell)
            else:
                rejected_reward_indices.append(cell)
            continue

        # Check which landmark windows the eligible peak falls in
        landmark_peaks = []
        for lm_idx, (lm_min, lm_max) in enumerate(landmark_windows):
            lm_mask    = (bin_centers >= lm_min) & (bin_centers <= lm_max)
            lm_indices = np.where(lm_mask)[0]
            if len(lm_indices) > 0:
                lm_response = np.max(profile[lm_indices])
                landmark_responses[cell, lm_idx] = lm_response
                if lm_min <= eligible_peak_pos <= lm_max:
                    landmark_peaks.append((lm_idx, lm_response))
            else:
                landmark_responses[cell, lm_idx] = 0

        if len(landmark_peaks) == 0:
            rejected_no_landmark_indices.append(cell)
            continue

        preferred_lm_idx, preferred_response = max(landmark_peaks, key=lambda x: x[1])

        other_responses = [landmark_responses[cell, i]
                           for i in range(n_landmarks) if i != preferred_lm_idx]
        pref_strength = (preferred_response - np.mean(other_responses)
                         if other_responses else preferred_response)

        preferred_landmark[cell]  = preferred_lm_idx
        preference_strength[cell] = pref_strength
        valid_cells[cell]         = True

    # Summary
    n_valid = int(np.sum(valid_cells))
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total cells: {n_cells}")
    print(f"Valid cells with landmark preference: {n_valid} ({n_valid/n_cells*100:.1f}%)")
    print(f"\nRejection breakdown:")
    print(f"  Zero activity:                         {len(rejected_zero_indices)}")
    print(f"  Onset-only (no eligible-region peak):  {len(rejected_onset_indices)}")
    print(f"  Reward-only (no eligible-region peak): {len(rejected_reward_indices)}")
    print(f"  Peak outside landmark windows:         {len(rejected_no_landmark_indices)}")

    print(f"\nLandmark preference distribution:")
    for lm_idx in range(n_landmarks):
        n_pref = int(np.sum(preferred_landmark[valid_cells] == lm_idx))
        pct = n_pref / n_valid * 100 if n_valid > 0 else 0
        print(f"  L{lm_idx+1} ({landmark_positions[lm_idx]} cm): {n_pref} ({pct:.1f}%)")

    return {
        'preferred_landmark':  preferred_landmark,
        'landmark_responses':  landmark_responses,
        'preference_strength': preference_strength,
        'peak_positions':      peak_positions,
        'global_peak_bins':    global_peak_bins,
        'valid_cells':         valid_cells,
        'mean_profiles':       mean_profiles,
        'landmark_positions':  np.array(landmark_positions),
        'landmark_windows':    landmark_windows,
        'rejected_cells': {
            'onset':        np.array(rejected_onset_indices),
            'reward':       np.array(rejected_reward_indices),
            'no_landmark':  np.array(rejected_no_landmark_indices),
            'zero_activity':np.array(rejected_zero_indices),
        },
        'parameters': {
            'landmark_windows_config': landmark_windows_config,
            'landmark_window':         landmark_window,
            'exclude_first_bins':      exclude_first_bins,
            'exclude_last_bins':       exclude_last_bins,
            'onset_threshold_cm':      onset_threshold_cm,
            'end_threshold_cm':        end_threshold_cm,
            'boundary_exclusion':      boundary_exclusion,
            'min_allowed':             min_allowed,
            'max_allowed':             max_allowed,
            'n_cells':                 n_cells,
            'n_landmarks':             n_landmarks,
        },
    }


# ============================================================================
# run_landmark_analysis — identical to parent except calls the local
# identify_landmark_responses (masked-onset version above)
# ============================================================================

def run_landmark_analysis(normalized_spatial_activity, bin_centers, layer_cells,
                          reliable_valid_cells, landmark_positions=[30, 60, 90, 120],
                          landmark_windows_config=None,
                          landmark_window=10.0,
                          boundary_exclusion=(10, 10),
                          exclude_first_bins=5, exclude_last_bins=5,
                          trials_per_block=30, smoothing_sigma=1.0,
                          save_path=None, session_id=None, date_str=None):
    """
    Complete OpenLoopVR landmark preference workflow.

    Calls the masked-onset identify_landmark_responses defined above;
    all downstream phases use the parent-script functions unchanged.
    """
    print("\n" + "="*70)
    print("LANDMARK PREFERENCE ANALYSIS - OPENLOOPVR (masked onset)")
    print("="*70)

    # Phase 1 — masked-onset version
    print("\n" + "-"*70)
    print("PHASE 1: IDENTIFYING LANDMARK RESPONSES")
    print("-"*70)

    landmark_results = identify_landmark_responses(
        normalized_spatial_activity, bin_centers, landmark_positions,
        landmark_windows_config=landmark_windows_config,
        landmark_window=landmark_window,
        boundary_exclusion=boundary_exclusion,
        smoothing_sigma=smoothing_sigma,
        exclude_first_bins=exclude_first_bins,
        exclude_last_bins=exclude_last_bins,
    )

    # Phase 2
    print("\n" + "-"*70)
    print("PHASE 2: LAYER-SPECIFIC ANALYSIS (FULL SESSION)")
    print("-"*70)

    layer_results = analyze_layer_landmark_preferences(
        landmark_results, layer_cells, reliable_valid_cells
    )

    # Phase 3
    print("\n" + "-"*70)
    print("PHASE 3: WITHIN-SESSION TEMPORAL DYNAMICS")
    print("-"*70)

    dynamics_results = analyze_within_session_dynamics(
        normalized_spatial_activity, bin_centers, landmark_positions,
        layer_cells, reliable_valid_cells,
        landmark_window=landmark_window,
        boundary_exclusion=boundary_exclusion,
        trials_per_block=trials_per_block,
        smoothing_sigma=smoothing_sigma,
    )

    # Phase 4
    print("\n" + "-"*70)
    print("PHASE 4: CREATING VISUALIZATIONS")
    print("-"*70)

    if save_path is not None:
        h5_save_path = os.path.join(save_path, f"{session_id}_landmark_preferences.h5")
        save_dir = os.path.join(save_path, 'LandmarkPreference')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir     = None
        h5_save_path = None

    fig_heatmap = plot_layer_landmark_heatmap(
        layer_results, landmark_positions,
        title=f"Landmark Preferences by Layer - {session_id or 'Session'}",
        save_path=save_dir,
    )

    # fig_dynamics = None
    # if dynamics_results is not None:
    #     fig_dynamics = plot_within_session_dynamics(
    #         dynamics_results,
    #         title=f"Within-Session Dynamics - {session_id or 'Session'}",
    #         save_path=save_dir,
    #     )

    # fig_examples = plot_example_cells_by_landmark(
    #     normalized_spatial_activity, bin_centers,
    #     landmark_results, layer_results,
    #     landmark_positions, n_examples=2,
    #     save_path=save_dir,
    # )

    print("\n  Creating rejected cells visualization...")
    fig_rejected = plot_rejected_cells_response(
        landmark_results, bin_centers,
        landmark_positions=landmark_positions,
        trim_start_bins=exclude_first_bins,
        trim_end_bins=exclude_last_bins,
        save_path=save_dir,
    )

    print("\n  Creating landmark assignment visualization...")
    fig_by_landmark = plot_cells_by_landmark_assignment(
        landmark_results, bin_centers,
        landmark_positions=landmark_positions,
        trim_start_bins=exclude_first_bins,
        trim_end_bins=exclude_last_bins,
        save_path=save_dir,
    )

    # print("\n  Creating landmark assignment summary...")
    # fig_assignment_summary = plot_landmark_assignment_summary(
    #     landmark_results, bin_centers,
    #     landmark_positions=landmark_positions,
    #     save_path=save_dir,
    # )

    # Phase 5
    if h5_save_path is not None and session_id is not None:
        print("\n" + "-"*70)
        print("PHASE 5: SAVING HDF5 DATA")
        print("-"*70)
        save_session_landmark_data(
            layer_results, dynamics_results,
            h5_save_path, session_id, date_str,
        )

    return {
        'landmark_results':  landmark_results,
        'layer_results':     layer_results,
        'dynamics_results':  dynamics_results,
        'figures': {
            'heatmap':            fig_heatmap,
            # 'dynamics':           fig_dynamics,
            # 'examples':           fig_examples,
            'rejected':           fig_rejected,
            'by_landmark':        fig_by_landmark,
            # 'assignment_summary': fig_assignment_summary,
        },
    }
