"""
Preprocess_OpenLoopVR.py
Preprocessing for open-loop VR replay sessions (running replay, stationary replay).

Key differences from Preprocess.py:
  1. min_speed_cm_s = 0.0  — no speed filtering.
     In open-loop VR the position signal is externally driven (VR playback),
     so the VRlog-derived speed reflects playback, not the animal's movement.
     Using a speed threshold here would filter based on playback speed, not behavior.
  2. TMlog-derived speed is also computed and saved alongside VRlog-derived speed.
     The two signals can be compared in the analysis notebook (animal motion vs. VR playback).
  3. Session type ('moving' or 'stationary') is stored as metadata in the h5.
  4. Separate from Preprocess.py so existing analyses are completely unaffected.

Output: same preproc.h5 structure as Preprocess.py, with added keys:
  - speed_tmlog_cm_s  : TMlog encoder speed (cm/s) for the same lap-filtered frames
  - session_type      : string 'moving' or 'stationary'

JSY, 05/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import numpy as np
import datetime
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors

from helper import dataLoader, files
from helper import SpikeSmoothing, ReliabilityTesting as RT, SpatialDiscretization as SD
from helper import BehavioralDataFiltering as DF, ResponseVisualization as RV

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

# ── same Parula colormap as Preprocess.py ──────────────────────────────────
_PARULA_COLORS = [
    (0.2422, 0.1504, 0.6603), (0.2108, 0.3706, 0.9717),
    (0.0196, 0.5804, 0.8745), (0.0863, 0.6510, 0.7490),
    (0.1961, 0.6980, 0.6039), (0.3647, 0.7412, 0.5176),
    (0.6275, 0.7647, 0.3843), (0.8510, 0.7882, 0.1961),
    (0.9686, 0.8235, 0.0667), (0.9765, 0.9843, 0.0510),
]
PARULA = mcolors.LinearSegmentedColormap.from_list('parula', _PARULA_COLORS)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: detect session type from folder name
# ══════════════════════════════════════════════════════════════════════════════

def detect_session_type(twop_filepath):
    """
    Return 'moving' or 'stationary' based on the parent folder name.
    Falls back to 'unknown' if neither keyword is found.
    """
    parent = os.path.basename(os.path.dirname(twop_filepath)).lower()
    if 'stationary' in parent:
        return 'stationary'
    elif 'moving' in parent:
        return 'moving'
    else:
        print(f"  WARNING: could not detect session type from '{parent}', defaulting to 'unknown'")
        return 'unknown'


# ══════════════════════════════════════════════════════════════════════════════
# Helper: TMlog speed extraction with min_speed=0 (open-loop compatible)
# ══════════════════════════════════════════════════════════════════════════════

def extract_tmlog_speed_openloop(twop_filepath, vr_dict, smoothed, framerate,
                                  min_trial_duration_seconds, max_trial_duration_seconds):
    """
    Compute TMlog-derived speed for the same lap-filtered frames as the spike data.

    Uses min_speed_cm_s=0 so the lap selection matches Preprocess_OpenLoopVR exactly.
    Returns None if no TMlog file is found.
    """
    tmlog_files = sorted(glob.glob(os.path.join(twop_filepath, "TMlog*.txt")))
    if not tmlog_files:
        print("  No TMlog*.txt found — speed_tmlog_cm_s will not be saved.")
        return None

    tmlog_path = tmlog_files[0]
    print(f"  TMlog file: {os.path.basename(tmlog_path)}")

    try:
        from Calculate_Speed_From_Tmlog import align_tmlog_to_2p
        _, speed_tmlog_full, _, _, _ = align_tmlog_to_2p(twop_filepath, tmlog_path)
    except Exception as e:
        print(f"  WARNING: TMlog alignment failed ({e}) — skipping TMlog speed.")
        return None

    # Clip or pad to match the number of 2p frames in the spike array
    n_frames = smoothed.shape[1]
    if len(speed_tmlog_full) >= n_frames:
        speed_tmlog_clipped = speed_tmlog_full[:n_frames]
    else:
        pad = n_frames - len(speed_tmlog_full)
        speed_tmlog_clipped = np.pad(speed_tmlog_full, (0, pad), mode='edge')
        print(f"  WARNING: TMlog shorter than spike data by {pad} frames; padded with last value.")

    # Apply the same lap filtering as the spike preprocessing (min_speed_cm_s=0)
    # Pass the 1-D speed array as a single-row "spike" array so it goes through
    # the same lap detection and duration filter.
    filtered_tmlog_laps, _, _, n_valid = DF.process_data_with_speed_filtering(
        speed_tmlog_clipped[np.newaxis, :],   # shape (1, n_frames)
        vr_dict['interp_location'],
        min_trial_duration_seconds=min_trial_duration_seconds,
        max_trial_duration_seconds=max_trial_duration_seconds,
        framerate=framerate,
        min_speed_cm_s=0.0,                   # must match the spike preprocessing
        frames_to_keep=5
    )

    if n_valid == 0:
        print("  WARNING: no valid laps found during TMlog extraction.")
        return None

    # filtered_tmlog_laps[i] has shape (1, n_frames_in_lap_i)
    temporal_tmlog_speed = np.concatenate([lap[0] for lap in filtered_tmlog_laps])
    print(f"  TMlog speed extracted: {len(temporal_tmlog_speed)} frames across {n_valid} laps.")
    return temporal_tmlog_speed.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Helper: cross-validated Parula heatmap (same as Preprocess.py)
# ══════════════════════════════════════════════════════════════════════════════


def create_heatmap_responseplot(normalized_activity, cell_mask, bin_centers, title, save_path):
    """
    Sorted tuning-curve heatmap matching SMI_ResponsePlot_AllAnimals style.
    Mean over all laps, row-normalised 0–1, sorted by preferred position,
    with a preferred-position colour strip on the right.
    """
    indices = np.where(cell_mask)[0]
    n_cells = len(indices)
    if n_cells == 0:
        print(f"  No cells to plot for '{title}' — skipping heatmap.")
        return

    act = normalized_activity[indices]           # (n_cells, n_laps, n_bins)
    mean_tuning = np.nanmean(act, axis=1)        # (n_cells, n_bins)

    row_max = mean_tuning.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    tuning_norm = mean_tuning / row_max

    preferred_positions = bin_centers[np.argmax(tuning_norm, axis=1)]
    order = np.argsort(preferred_positions)
    tuning_sorted = tuning_norm[order]
    pp_sorted = preferred_positions[order]

    pos_min, pos_max = bin_centers[0], bin_centers[-1]

    fig, ax = plt.subplots(1, 1, figsize=(5, 8))
    ax.imshow(tuning_sorted, aspect='auto', cmap=PARULA, vmin=0, vmax=1,
              extent=[pos_min, pos_max, n_cells, 0], interpolation='nearest')

    for lm in [37, 65, 93, 120]:
        if pos_min <= lm <= pos_max:
            ax.axvline(lm, color='white', linewidth=1.2, linestyle='--', alpha=0.8)

    pp_norm = (pp_sorted - pos_min) / max(pos_max - pos_min, 1)
    ax_s = ax.inset_axes([1.01, 0, 0.04, 1], transform=ax.transAxes)
    ax_s.imshow(pp_norm[:, np.newaxis], aspect='auto', cmap='twilight', vmin=0, vmax=1,
                extent=[0, 1, n_cells, 0], interpolation='nearest')
    ax_s.set_xticks([])
    ax_s.set_yticks([])

    ax.set_xlim(pos_min, pos_max)
    ax.set_xticks([30, 60, 90, 120])
    ax.set_xticklabels(['30', '60', '90', '120'], fontsize=20)
    ax.set_yticks([0, n_cells])
    ax.set_yticklabels(['0', str(n_cells)], fontsize=20)
    ax.set_xlabel('VR position (cm)', fontsize=20)
    ax.set_ylabel('# cells', fontsize=20)
    ax.set_title(title, fontsize=25, fontweight='bold')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved heatmap: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main preprocessing function
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_openloopVR(twop_filepath):
    """
    Preprocess one open-loop VR session (running replay or stationary replay).

    Identical to preprocess_2pVR in Preprocess.py except:
      - min_speed_cm_s = 0.0  (no speed filtering)
      - TMlog speed computed and saved as 'speed_tmlog_cm_s'
      - 'session_type' metadata stored in h5

    Parameters
    ----------
    twop_filepath : str
        Path to the TSeries folder for the open-loop session.

    Returns
    -------
    preprocessed_dict : dict
    """
    session_type = detect_session_type(twop_filepath)
    print(f"\n  Session type detected: {session_type}")

    # ── Find VRlog (same search as Preprocess.py) ──────────────────────────
    search_dirs = [twop_filepath, os.path.dirname(twop_filepath)]
    vrlog_files = []
    for d in search_dirs:
        all_txt = glob.glob(os.path.join(d, "*.txt"))
        vrlog_files = [f for f in all_txt if 'vrlog' in os.path.basename(f).lower()]
        if vrlog_files:
            break

    if len(vrlog_files) == 1:
        vr_filepath = vrlog_files[0]
    elif len(vrlog_files) > 1:
        vr_filepath = sorted(vrlog_files)[0]
        print(f"  Warning: multiple VRlog files found, using: {os.path.basename(vr_filepath)}")
    else:
        all_txt = glob.glob(os.path.join(twop_filepath, "*.txt"))
        raise FileNotFoundError(
            f"No VRlog*.txt found in:\n  {twop_filepath}\n  {os.path.dirname(twop_filepath)}\n"
            f".txt files present: {[os.path.basename(f) for f in all_txt]}"
        )
    print(f"  VRlog: {os.path.basename(vr_filepath)}")

    # ── Step 1: Load and align ─────────────────────────────────────────────
    procData = dataLoader(twop_filepath, vr_filepath)
    animal_id, date, framerate = procData.load_data()
    twop_dict, vr_dict = procData.align_data()

    # ── Step 2a: Temporal offset ───────────────────────────────────────────
    optimal_offset, _, _ = SpikeSmoothing.run_offset_optimization(twop_filepath, vr_filepath, list(range(-10, 11)))
    offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], optimal_offset)

    # ── Step 2b: Gaussian smoothing ────────────────────────────────────────
    smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms=250)
    twop_dict['smoothed_spks'] = smoothed

    # ── Step 2c: Lap filtering — NO speed filter (key difference) ─────────
    print("\n" + "="*80)
    print("OPEN-LOOP VR: LAP FILTERING (min_speed_cm_s = 0)")
    print("="*80)

    min_trial_duration_seconds = 5
    max_trial_duration_seconds = 60

    filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps = \
        DF.process_data_with_speed_filtering(
            smoothed,
            vr_dict['interp_location'],
            min_trial_duration_seconds=min_trial_duration_seconds,
            max_trial_duration_seconds=max_trial_duration_seconds,
            framerate=framerate,
            min_speed_cm_s=0.0,    # open-loop: no speed filtering
            frames_to_keep=5
        )

    if n_valid_laps == 0:
        raise ValueError("No valid laps after filtering! Check VRlog lap detection.")

    print(f"  Valid laps: {n_valid_laps}")

    # ── Step 3: Spatial discretization ────────────────────────────────────
    single_revolution_VR        = 282.415
    single_revolution_treadmill = 27.8
    single_lap_VR               = 1320.645683
    single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR

    spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = \
        SD.spatial_assignment_with_physical_units(
            n_valid_laps, filtered_spks_laps, filtered_location_laps,
            physical_lap_length_cm=single_lap_treadmill
        )

    window_cm = 0.5
    smoothed_spatial_activity = SpikeSmoothing.spatial_smooth(spatial_activity, window_cm=window_cm)

    # ── Step 4: Reliability testing ───────────────────────────────────────
    normalized_spatial_activity = RT.normalize_spatial_activity(smoothed_spatial_activity)

    combined_reliable, reliable_cells, _, avg_cc, cohens_d, _, _, _ = \
        RT.combined_reliability_test_improved(
            smoothed_spatial_activity,
            n_shuffles=200,
            cc_percentile=90,
            cohen_threshold=0.8,
            min_cc_threshold=0.1,
            min_pattern_corr=0.3,
            peak_distance_threshold=5,
            use_activity_threshold=True,
            activity_method='absolute_percentile'
        )

    print(f"  Reliable cells: {np.sum(reliable_cells)} | Combined-reliable: {np.sum(combined_reliable)}")

    # ── Temporal arrays ────────────────────────────────────────────────────
    temporal_spikes   = np.concatenate(filtered_spks_laps, axis=1)
    temporal_location = np.concatenate(filtered_location_laps)
    temporal_speed    = np.concatenate(filtered_speed_laps)

    lap_starts, lap_ends = [], []
    cumsum = 0
    for lap_speed in filtered_speed_laps:
        lap_starts.append(cumsum)
        cumsum += len(lap_speed)
        lap_ends.append(cumsum)
    lap_starts = np.array(lap_starts)
    lap_ends   = np.array(lap_ends)

    # ── TMlog speed (open-loop compatible, min_speed=0) ───────────────────
    print("\nExtracting TMlog speed...")
    temporal_tmlog_speed = extract_tmlog_speed_openloop(
        twop_filepath, vr_dict, smoothed, framerate,
        min_trial_duration_seconds, max_trial_duration_seconds
    )

    # Shape verification
    assert temporal_spikes.shape[1] == len(temporal_speed),    "Spike/speed mismatch"
    assert len(temporal_speed) == len(temporal_location),       "Speed/location mismatch"
    if temporal_tmlog_speed is not None:
        assert len(temporal_tmlog_speed) == len(temporal_speed), \
            f"TMlog speed length ({len(temporal_tmlog_speed)}) ≠ spike frames ({len(temporal_speed)})"
    print("  All dimension checks passed.")

    # ── Step 5: Response plots ─────────────────────────────────────────────
    combinedreliable_dir = os.path.join(twop_filepath, 'combined_reliable_cell_plots')
    reliable_dir         = os.path.join(twop_filepath, 'reliable_cell_plots')
    os.makedirs(combinedreliable_dir, exist_ok=True)
    os.makedirs(reliable_dir,         exist_ok=True)


    RT.plot_individual_reliable_cells_to_pdf(
        spatial_activity=normalized_spatial_activity,
        reliable_cells=reliable_cells,
        save_directory=reliable_dir,
        avg_cc=avg_cc, cohen_d=cohens_d, bin_centers=bin_centers,
        normalize=True, dpi=150, cells_per_page=4
    )
    RT.plot_individual_reliable_cells_to_pdf(
        spatial_activity=normalized_spatial_activity,
        reliable_cells=combined_reliable,
        save_directory=combinedreliable_dir,
        avg_cc=avg_cc, cohen_d=cohens_d, bin_centers=bin_centers,
        normalize=True, dpi=150, cells_per_page=4
    )

    fig1, _ = RV.create_response_plot(normalized_spatial_activity, reliable_cells, clim=(0, 1))
    fig1.savefig(os.path.join(reliable_dir, 'reliable_cells.png'), dpi=150)
    plt.close(fig1)

    fig2, _ = RV.create_response_plot(normalized_spatial_activity, combined_reliable, clim=(0, 1))
    fig2.savefig(os.path.join(combinedreliable_dir, 'combined_reliable_cells.png'), dpi=150)
    plt.close(fig2)

    create_heatmap_responseplot(
        normalized_spatial_activity, reliable_cells, bin_centers,
        title=f'Reliable cells  (n={np.sum(reliable_cells)})  [{session_type}]',
        save_path=os.path.join(reliable_dir, 'reliable_cells_heatmap.png')
    )
    create_heatmap_responseplot(
        normalized_spatial_activity, combined_reliable, bin_centers,
        title=f'Combined-reliable cells  (n={np.sum(combined_reliable)})  [{session_type}]',
        save_path=os.path.join(combinedreliable_dir, 'combined_reliable_cells_heatmap.png')
    )

    # ── Build save dict ────────────────────────────────────────────────────
    med_coords = np.zeros((len(twop_dict['stat']), 2))
    for i, cell_stat in enumerate(twop_dict['stat']):
        med_coords[i, 0] = np.median(cell_stat['ypix'])
        med_coords[i, 1] = np.median(cell_stat['xpix'])

    preprocessed_dict = {
        # Spatial data
        'spatial_activity':      smoothed_spatial_activity,
        'norm_spatial_activity': normalized_spatial_activity,
        'reliable_cells':        reliable_cells.astype(bool),
        'combined_reliable':     combined_reliable.astype(bool),
        'avg_cc':                avg_cc.astype(np.float64),
        'cohen_d':               cohens_d.astype(np.float64),
        'bin_centers':           bin_centers.astype(np.float64),
        'med_coords':            med_coords.astype(np.float64),
        # Temporal data
        'speed_cm_s':            temporal_speed.astype(np.float64),      # VRlog-derived
        'smoothed_spks_temporal': temporal_spikes.astype(np.float64),
        'location_cm':           temporal_location.astype(np.float64),
        'lap_starts':            lap_starts.astype(np.int32),
        'lap_ends':              lap_ends.astype(np.int32),
        # Metadata
        'twop_filepath':         str(twop_filepath),
        'vr_filepath':           str(vr_filepath),
        'session_type':          str(session_type),
        'processing_timestamp':  datetime.datetime.now().isoformat(),
        'processing_params': {
            'framerate':                    float(framerate),
            'optimal_offset':               int(optimal_offset),
            'window_cm':                    float(window_cm),
            'min_speed_cm_s':               float(0.0),
            'min_trial_duration_seconds':   float(min_trial_duration_seconds),
            'max_trial_duration_seconds':   float(max_trial_duration_seconds),
            'single_lap_treadmill':         float(single_lap_treadmill),
        }
    }

    # Add TMlog speed if successfully extracted
    if temporal_tmlog_speed is not None:
        preprocessed_dict['speed_tmlog_cm_s'] = temporal_tmlog_speed

    # ── Save ───────────────────────────────────────────────────────────────
    save_dir  = os.path.dirname(twop_filepath) if os.path.isfile(twop_filepath) else twop_filepath
    savepath  = os.path.join(save_dir, f'{date}_{animal_id}_preproc.h5')
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    print(f'\nSaving preprocessed data → {savepath}')
    try:
        files.write_h5(savepath, preprocessed_dict)
        print('Successfully saved preprocessed data.')
    except Exception as e:
        print(f'Error saving to HDF5: {e}')
        raise

    return preprocessed_dict


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    twop_filepaths = [
        # ── JSY044 ────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\251122_JSY_JSY044_SpMod_OpenLoopVR_Moving\TSeries-11222025-1339-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\251123_JSY_JSY044_SpMod_OpenLoopVR_Stationary\TSeries-11232025-1222-001",

        # ── JSY051 ────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251106_JSY_JSY051_SpMO_OpenloopVR_moving\TSeries-11062025-1439-002",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251107_JSY_JSY051_SpMO_OpenloopVR_stationary\TSeries-11072025-1032-001",

        # ── JSY052 ────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251122_JSY_JSY052_SpMod_OpenLoopVR_Moving\TSeries-11222025-1339-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251123_JSY_JSY052_SpMod_OpenLoopVR_Stationary\TSeries-11232025-1222-002",

        # ── JSY054 ────────────────────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251106_JSY_JSY054_SpMO_OpenloopVR_moving\TSeries-11062025-1439-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251107_JSY_JSY054_SpMO_OpenloopVR_stationary\TSeries-11072025-1032-001",

        # ── JSY055 ────────────────────────────────────────────────────────
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251212_JSY_JSY055_SpatialModulation_OpenLoopVR_Moving\TSeries-12122025-1421-001",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251213_JSY_JSY055_SpatialModulation_OpenLoopVR_Stationary\TSeries-12132025-1711-001",
    ]

    n_total    = len(twop_filepaths)
    successful = []
    failed     = []

    for i, path in enumerate(twop_filepaths):
        print("\n" + "="*80)
        print(f"[{i+1}/{n_total}]  {os.path.basename(os.path.dirname(path))} / {os.path.basename(path)}")
        print("="*80)
        try:
            preprocess_openloopVR(path)
            successful.append(path)
        except Exception as e:
            import traceback
            failed.append((path, str(e)))
            print(f"FAILED: {e}")
            traceback.print_exc()

    print("\n" + "="*80)
    print(f"Done.  Successful: {len(successful)}/{n_total}  |  Failed: {len(failed)}/{n_total}")
    for p, err in failed:
        print(f"  FAILED: {os.path.basename(os.path.dirname(p))} — {err}")
