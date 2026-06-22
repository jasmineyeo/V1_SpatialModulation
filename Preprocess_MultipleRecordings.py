"""
Preprocess_MultipleRecordings.py

For sessions where the 2p recording was interrupted and restarted, yielding:
  - One combined suite2p output  (both recordings processed together in the GUI)
  - Two separate VRlog files     (one per recording, each time-locked to its 2p start)

Key assumptions:
  - VR sends a TTL start trigger to the 2p system, so both recordings have a
    synchronized START. Ends may differ (manually stopped) -- this is already
    handled by the existing align_data() logic which trims to the shorter signal.
  - Both recordings were run through suite2p together in the GUI, producing a
    single suite2p output folder (F.npy / spks.npy span both recordings).
  - suite2p stores per-recording frame counts in ops['frames_per_folder'].
  - The FOV is identical across recordings (same ROI set, no re-registration needed).

Strategy:
  1. Load the combined suite2p output ONCE from suite2p_path.
  2. Read ops['frames_per_folder'] to find the frame-index boundary between recordings.
  3. For each recording:
       a. Read its XML file to recover absolute frame timestamps.
       b. Load its VRlog and trim to the 's' (start) event.
       c. Align the VRlog to the neural segment using timestamp interpolation
          (same logic as dataLoader.align_data()).
  4. Run temporal offset optimisation on the FIRST recording segment only
     (the offset is a fixed hardware delay -- constant across recordings).
     Apply the same offset to all segments.
  5. Smooth spikes, filter by speed/duration, and collect laps per segment.
  6. Concatenate the lap-filtered data across all segments.
  7. Run the downstream pipeline (spatial discretisation, reliability testing,
     response plots, save) on the combined data -- identical to Preprocess.py.

New helper functions in this file (all existing pipeline functions are reused):
  - _get_frames_per_recording()     -- parse ops for per-recording frame counts
  - _read_xml_timestamps()          -- read XML and return absolute frame timestamps
  - _load_vr_data()                 -- load raw VRlog, trim to start event
  - _align_segment()                -- align one VRlog to one neural segment
  - _offset_optimization_on_arrays()-- run offset optimisation on pre-loaded arrays
  - preprocess_2pVR_multi()         -- main preprocessing function

JSY, 03/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import datetime
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

from helper import files
from helper import SpikeSmoothing, ReliabilityTesting as RT, SpatialDiscretization as SD
from helper import BehavioralDataFiltering as DF, ResponseVisualization as RV
from helper import read_xml
from helper.twop import TwoP

from Preprocess import convert_stat_to_serializable, convert_ops_to_serializable

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize']  = 20
rcParams['axes.titlesize']  = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20


# =============================================================================
# Helper: parse ops for per-recording frame counts
# =============================================================================

def _get_frames_per_recording(ops):
    """
    Extract the number of frames contributed by each input recording from the
    suite2p ops dictionary.

    When suite2p processes multiple folders together, it stores the frame count
    for each folder. Depending on the suite2p version the key may be named
    differently; this function tries several known key names.

    Parameters
    ----------
    ops : dict
        suite2p ops dictionary (loaded from ops.npy).

    Returns
    -------
    frames_per_recording : list of int
        Frame count for each recording, in the order they were fed to suite2p.

    Raises
    ------
    KeyError
        If none of the expected keys are found in ops.
    """
    candidate_keys = ['frames_per_folder', 'nframes_per_folder', 'frames_per_file']
    for key in candidate_keys:
        if key in ops:
            counts = ops[key]
            # ops may store this as a numpy array; convert to plain Python list of ints
            return [int(c) for c in counts]

    raise KeyError(
        "Could not find per-recording frame counts in ops. "
        f"Tried keys: {candidate_keys}. "
        "Keys present in ops: " + str(list(ops.keys()))
    )


# =============================================================================
# Helper: read XML timestamps for one recording
# =============================================================================

def _read_xml_timestamps(twop_path):
    """
    Read the PrairieView XML file for a recording and return absolute frame
    timestamps and the acquisition framerate.

    This replicates the XML-reading logic inside dataLoader.load_data().

    Parameters
    ----------
    twop_path : str
        Path to the TSeries folder (e.g. '.../TSeries-02032026-1751-002').
        The XML file must be named <basename>.xml inside this folder.

    Returns
    -------
    abs_timestamps : np.ndarray of datetime.datetime
        Absolute timestamp for each 2p frame.
    framerate : float
        Acquisition framerate in Hz.
    """
    twoP_filename = os.path.basename(twop_path)
    xml_path = os.path.join(twop_path, f"{twoP_filename}.xml")

    xml_dict = read_xml(xml_path)
    t0       = xml_dict["t0"]
    abs_time = xml_dict["abs_time"]
    rel_time = xml_dict["rel_time"]
    framerate = 1.0 / rel_time[1]

    # Build datetime array for each frame (same as dataLoader.load_data())
    abs_timestamps = np.zeros(np.size(abs_time, 0) - 1, dtype=datetime.datetime)
    for rep, t in enumerate(abs_time[:-1]):
        abs_timestamps[rep] = t0 + datetime.timedelta(seconds=t)

    return abs_timestamps, framerate


# =============================================================================
# Helper: load raw VRlog data
# =============================================================================

def _load_vr_data(vr_filepath):
    """
    Load and parse a VRlog text file, trimming everything before the start
    event ('s').

    This replicates the VRlog-loading logic inside dataLoader.load_data().

    Parameters
    ----------
    vr_filepath : str
        Full path to the VRlog .txt file.

    Returns
    -------
    vr_data : dict with keys:
        'absoluteT' : np.ndarray of str   -- wall-clock timestamps (HH.MM.SS.fff)
        'elapsedT'  : np.ndarray of float -- elapsed time in seconds
        'event'     : np.ndarray of str   -- event codes
        'location'  : np.ndarray of float -- VR position (clipped to >= 0)
    animal_id : str
    date : str
    """
    behav_filename = os.path.basename(vr_filepath)

    # Extract animal ID and date from filename
    match = re.match(r"VRlog_(JSY\d+)_(\d{8})_\d{2}-\d{2}-\d{2}\.txt", behav_filename)
    if match:
        animal_id = match.group(1)
        date      = match.group(2)
    else:
        print(f"Warning: VRlog filename '{behav_filename}' does not match expected pattern.")
        animal_id, date = "unknown", "unknown"

    rawVR_data = []
    with open(vr_filepath, "r") as f:
        lines = f.readlines()
        for line in lines[3:]:           # first 3 lines are header
            rawVR_data.append(line.strip().split("\t"))

    vr_data = {
        'absoluteT': np.array([row[0] for row in rawVR_data]),
        'elapsedT':  np.array([float(row[1]) for row in rawVR_data]),
        'event':     np.array([row[2] for row in rawVR_data]),
        'location':  np.array([float(row[3]) for row in rawVR_data]),
    }

    # Clip negative positions to zero
    vr_data['location'][vr_data['location'] < 0] = 0

    # Trim to the first start event ('s')
    start_idx = np.where(vr_data['event'] == 's')[0][0]
    for key in vr_data:
        vr_data[key] = vr_data[key][start_idx:]

    print(f"  VRlog first timestamp: {vr_data['absoluteT'][0]}")
    return vr_data, animal_id, date


# =============================================================================
# Helper: align one VRlog to one neural segment
# =============================================================================

def _align_segment(neural_spks_segment, neural_abs_timestamps, vr_data):
    """
    Align a VRlog to a neural data segment, producing an interpolated location
    trace at the 2p frame rate.

    This replicates dataLoader.align_data() but operates on pre-sliced arrays
    rather than full-session data, so it can be called independently for each
    recording segment.

    Parameters
    ----------
    neural_spks_segment : np.ndarray, shape (n_cells, n_frames)
        Spike (sps) data for this recording segment only.
    neural_abs_timestamps : np.ndarray of datetime.datetime, length n_frames
        Absolute timestamp for each frame in this segment.
    vr_data : dict
        Raw VRlog dict returned by _load_vr_data().

    Returns
    -------
    twop_dict_seg : dict
        Contains 'sps' (the spike segment) and 'RelativeT' (seconds from start).
    vr_dict_seg : dict
        Contains 'interp_location' (VR position interpolated to 2p frame times).
    """
    reference_date = neural_abs_timestamps[0].date()
    reference_hour = neural_abs_timestamps[0].hour

    # --- Parse VR wall-clock timestamps into datetime objects ---
    # (same AM/PM correction logic as dataLoader.align_data())
    vr_abs_t = []
    for t_str in vr_data['absoluteT']:
        time_obj = datetime.datetime.strptime(t_str, '%H.%M.%S.%f').time()
        dt = datetime.datetime.combine(reference_date, time_obj)

        if dt.hour < 12 and reference_hour >= 12:
            dt = dt + datetime.timedelta(hours=12)
        elif dt.hour >= 12 and reference_hour < 12:
            dt = dt - datetime.timedelta(hours=12)

        vr_abs_t.append(dt)
    vr_abs_t = np.array(vr_abs_t)

    # --- Build relative time axes (seconds from respective t=0) ---
    vr_rel_t = np.array([(t - vr_abs_t[0]).total_seconds() for t in vr_abs_t])

    # Anchor VR relative times to the 2p start so the two axes share the same
    # origin -- identical to dataLoader.align_data() approach
    vr_rel_t_td     = np.array([datetime.timedelta(seconds=t) for t in vr_rel_t])
    aligned_vr_abs  = neural_abs_timestamps[0] + vr_rel_t_td

    # Trim VR to the end of the neural recording (end-mismatch handled here)
    last_neural_t = neural_abs_timestamps[-1]
    values_after  = aligned_vr_abs[aligned_vr_abs > last_neural_t]
    if len(values_after) == 0:
        trim_idx = len(aligned_vr_abs)
    else:
        trim_idx = np.where(aligned_vr_abs == values_after[0])[0][0]

    trimmed_vr_abs      = aligned_vr_abs[:trim_idx]
    trimmed_vr_rel_t    = vr_rel_t[:trim_idx]
    trimmed_vr_location = vr_data['location'][:trim_idx]

    # --- Build the 2p relative time axis ---
    twop_rel_t = np.array(
        [(t - neural_abs_timestamps[0]).total_seconds() for t in neural_abs_timestamps]
    )

    # --- Interpolate VR location to 2p frame times ---
    interp_location = np.interp(twop_rel_t, trimmed_vr_rel_t, trimmed_vr_location)

    print(f"  Neural segment: {neural_spks_segment.shape[1]} frames")
    print(f"  Interpolated location: {interp_location.shape[0]} samples")

    twop_dict_seg = {
        'sps':       neural_spks_segment,
        'RelativeT': twop_rel_t,
    }
    vr_dict_seg = {
        'interp_location': interp_location,
    }
    return twop_dict_seg, vr_dict_seg


# =============================================================================
# Helper: offset optimisation on pre-loaded arrays
# =============================================================================

def _offset_optimization_on_arrays(twop_dict_seg, vr_dict_seg, framerate,
                                   save_dir=None):
    """
    Run temporal offset optimisation on pre-loaded, pre-aligned data arrays.

    This mirrors the logic of SpikeSmoothing.run_offset_optimization() but
    bypasses the internal dataLoader call, so it can be used with a data
    segment that has already been sliced and aligned.

    The offset accounts for a fixed hardware delay between the VR position
    signal and the calcium imaging frames. Because this is a hardware property
    it is constant across recordings from the same session, so we optimise on
    the first segment and apply the result to all segments.

    Parameters
    ----------
    twop_dict_seg : dict
        Segment dict with at least 'sps' (n_cells x n_frames).
    vr_dict_seg : dict
        Segment dict with at least 'interp_location' (n_frames,).
    framerate : float
        Acquisition framerate in Hz.
    save_dir : str or None
        If provided, the optimisation figure is saved here.

    Returns
    -------
    optimal_offset : int
        Recommended temporal offset in frames.
    results : dict
        Full results from find_optimal_temporal_offset.
    best_offsets : dict
        Best offset per individual sharpness metric.
    """
    results, best_offsets, optimal_offset = SpikeSmoothing.find_optimal_temporal_offset(
        twop_dict_seg,
        vr_dict_seg,
        framerate,
        offset_range=list(range(-10, 11)),
        twop_filepath=save_dir
    )
    return optimal_offset, results, best_offsets


# =============================================================================
# Main preprocessing function
# =============================================================================

def preprocess_2pVR_multi(suite2p_path, recording_pairs):
    """
    Preprocess a session consisting of multiple 2p recordings that were run
    together through suite2p (single combined output) but have separate VRlog
    files.

    Parameters
    ----------
    suite2p_path : str
        Path to the folder that CONTAINS the 'suite2p/' directory, i.e. the
        folder from which suite2p/plane0/F.npy etc. can be reached.
        In the GUI this is the 'save path' you selected; commonly this is the
        first recording's TSeries folder.

    recording_pairs : list of (twop_path, vr_path) tuples
        One tuple per recording, in the SAME ORDER that they were fed to
        suite2p. Each entry:
          twop_path : str  -- TSeries folder for that recording (needed for XML)
          vr_path   : str  -- Full path to the corresponding VRlog .txt file.
                             If None, the script will auto-detect a VRlog*.txt
                             file inside twop_path (same as Preprocess.py).

    Returns
    -------
    preprocessed_dict : dict
        Same structure as the output of preprocess_2pVR() in Preprocess.py,
        saved as an HDF5 file alongside the suite2p output.
    """
    n_recordings = len(recording_pairs)
    print(f"\n{'='*80}")
    print(f"MULTI-RECORDING PREPROCESSING  ({n_recordings} recordings)")
    print(f"suite2p path: {suite2p_path}")
    print(f"{'='*80}\n")

    # ------------------------------------------------------------------
    # 1. Load the combined suite2p output ONCE
    # ------------------------------------------------------------------
    print("Step 1: Loading combined suite2p output...")
    twoP_filename = os.path.basename(suite2p_path)
    raw_twop = TwoP(suite2p_path, twoP_filename)
    raw_twop.find_files()
    twop_combined = raw_twop.calc_dFF()   # dict: spikes_per_sec, stat, ops, ...

    sps_combined  = twop_combined['spikes_per_sec']  # shape: (n_cells, total_frames)
    stat          = twop_combined['stat']
    ops           = twop_combined['ops']
    n_cells, total_frames = sps_combined.shape
    print(f"  Combined neural data: {n_cells} cells x {total_frames} frames")

    # ------------------------------------------------------------------
    # 2. Determine frame boundaries between recordings
    # ------------------------------------------------------------------
    print("\nStep 2: Determining per-recording frame counts...")
    frames_per_rec = _get_frames_per_recording(ops)

    if len(frames_per_rec) != n_recordings:
        raise ValueError(
            f"ops['frames_per_folder'] has {len(frames_per_rec)} entries but "
            f"{n_recordings} recording pairs were provided. "
            "Ensure recording_pairs are in the same order as the suite2p input."
        )

    # Compute cumulative boundaries: segment i spans frames [starts[i], ends[i])
    starts = np.concatenate([[0], np.cumsum(frames_per_rec[:-1])]).astype(int)
    ends   = np.cumsum(frames_per_rec).astype(int)

    for i, (s, e, fp) in enumerate(zip(starts, ends, frames_per_rec)):
        print(f"  Recording {i+1}: frames {s} -- {e}  ({fp} frames)")

    # ------------------------------------------------------------------
    # 3. Auto-detect VRlog paths if not supplied, and load timestamps + VR
    # ------------------------------------------------------------------
    print("\nStep 3: Loading per-recording XML timestamps and VRlogs...")

    # Resolve any None vr_path entries using glob (mirrors Preprocess.py logic)
    resolved_pairs = []
    for twop_path, vr_path in recording_pairs:
        if vr_path is None:
            vrlog_files = sorted(glob.glob(os.path.join(twop_path, "VRlog*.txt")))
            if len(vrlog_files) == 0:
                raise FileNotFoundError(f"No VRlog*.txt found in {twop_path}")
            if len(vrlog_files) > 1:
                print(f"  Warning: multiple VRlogs in {twop_path}, using first: "
                      f"{os.path.basename(vrlog_files[0])}")
            vr_path = vrlog_files[0]
        resolved_pairs.append((twop_path, vr_path))

    # Load XML timestamps and VRlog for each recording
    all_timestamps = []   # list of np.ndarray of datetime
    all_vr_data    = []   # list of raw vr_data dicts
    framerates     = []
    animal_ids     = []
    dates          = []

    for i, (twop_path, vr_path) in enumerate(resolved_pairs):
        print(f"\n  Recording {i+1}: {os.path.basename(twop_path)}")
        abs_ts, fr = _read_xml_timestamps(twop_path)
        vr_data, animal_id, date = _load_vr_data(vr_path)
        print(f"  First 2p timestamp: {abs_ts[0]}")

        all_timestamps.append(abs_ts)
        all_vr_data.append(vr_data)
        framerates.append(fr)
        animal_ids.append(animal_id)
        dates.append(date)

    # Use the first recording's metadata for naming the output file
    animal_id = animal_ids[0]
    date      = dates[0]

    # Warn if framerates differ across recordings (should be the same FOV/settings)
    if len(set(round(fr, 2) for fr in framerates)) > 1:
        print(f"\nWarning: framerates differ across recordings: {framerates}")
        print("Using framerate from first recording for downstream processing.")
    framerate = framerates[0]

    # ------------------------------------------------------------------
    # 4. Align each VRlog to its neural segment
    # ------------------------------------------------------------------
    print(f"\nStep 4: Aligning VRlogs to neural segments...")

    aligned_twop_segs = []   # list of twop_dict_seg per recording
    aligned_vr_segs   = []   # list of vr_dict_seg per recording

    for i in range(n_recordings):
        print(f"\n  Aligning recording {i+1}...")

        # Slice neural data for this recording
        sps_seg = sps_combined[:, starts[i]:ends[i]]

        # Slice the XML timestamps to match this segment's frame count
        # (XML covers the full combined session, so slice by frame index)
        ts_seg = all_timestamps[i][:sps_seg.shape[1]]

        twop_seg, vr_seg = _align_segment(sps_seg, ts_seg, all_vr_data[i])
        aligned_twop_segs.append(twop_seg)
        aligned_vr_segs.append(vr_seg)

    # ------------------------------------------------------------------
    # 5. Temporal offset optimisation (on first segment only)
    # ------------------------------------------------------------------
    print(f"\nStep 5: Temporal offset optimisation (using recording 1)...")

    save_dir_opt = os.path.join(suite2p_path, 'offset_optimisation_rec1')
    os.makedirs(save_dir_opt, exist_ok=True)

    optimal_offset, _, _ = _offset_optimization_on_arrays(
        aligned_twop_segs[0],
        aligned_vr_segs[0],
        framerate,
        save_dir=save_dir_opt
    )
    print(f"  Optimal offset: {optimal_offset} frames "
          f"(applied to all {n_recordings} recordings)")

    # ------------------------------------------------------------------
    # 6. Smooth spikes and filter laps for each segment, then concatenate
    # ------------------------------------------------------------------
    print(f"\nStep 6: Smoothing, speed-filtering, and collecting laps...")

    min_trial_duration_seconds = 5
    max_trial_duration_seconds = 60

    all_filtered_spks     = []   # will concatenate across recordings
    all_filtered_location = []
    all_filtered_speed    = []

    for i in range(n_recordings):
        print(f"\n  Recording {i+1}:")
        sps_seg = aligned_twop_segs[i]['sps']

        # Apply temporal offset (same offset for all recordings)
        offset_sps = SpikeSmoothing.apply_temporal_offset(sps_seg, optimal_offset)

        # Smooth deconvolved traces with a 250 ms Gaussian window
        smoothed = SpikeSmoothing.smooth_spikes(offset_sps, framerate, window_ms=250)

        # Speed-filter and split into laps
        filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps = \
            DF.process_data_with_speed_filtering(
                smoothed,
                aligned_vr_segs[i]['interp_location'],
                min_trial_duration_seconds=min_trial_duration_seconds,
                max_trial_duration_seconds=max_trial_duration_seconds,
                framerate=framerate,
                min_speed_cm_s=2.0,
                frames_to_keep=5
            )

        if n_valid_laps == 0:
            print(f"  WARNING: recording {i+1} yielded 0 valid laps -- skipping.")
            continue

        print(f"  Valid laps: {n_valid_laps}")
        all_filtered_spks.extend(filtered_spks_laps)
        all_filtered_location.extend(filtered_location_laps)
        all_filtered_speed.extend(filtered_speed_laps)

    n_valid_laps_total = len(all_filtered_spks)
    if n_valid_laps_total == 0:
        raise ValueError("No valid laps found across all recordings!")

    print(f"\n  Total valid laps across all recordings: {n_valid_laps_total}")

    # ------------------------------------------------------------------
    # 7. Spatial discretisation on the combined laps
    # ------------------------------------------------------------------
    print(f"\nStep 7: Spatial discretisation...")

    single_revolution_VR       = 282.415
    single_revolution_treadmill = 27.8
    single_lap_VR              = 1320.645683
    single_lap_treadmill       = single_revolution_treadmill * single_lap_VR / single_revolution_VR

    spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = \
        SD.spatial_assignment_with_physical_units(
            n_valid_laps_total,
            all_filtered_spks,
            all_filtered_location,
            physical_lap_length_cm=single_lap_treadmill
        )

    window_cm = 0.5
    smoothed_spatial_activity = SpikeSmoothing.spatial_smooth(spatial_activity, window_cm=window_cm)

    # ------------------------------------------------------------------
    # 8. Reliability testing
    # ------------------------------------------------------------------
    print(f"\nStep 8: Reliability testing...")

    normalized_spatial_activity = RT.normalize_spatial_activity(smoothed_spatial_activity)

    combined_reliable, reliable_cells, _, avg_cc, cohens_d, _, _, _ = \
        RT.combined_reliability_test_improved(
            smoothed_spatial_activity,
            n_shuffles=200,
            cc_percentile=90,
            cohen_threshold=0.8,
            min_cc_threshold=0.2,
            min_pattern_corr=0.3,
            peak_distance_threshold=5,
            use_activity_threshold=True,
            activity_method='absolute_percentile'
        )

    print(f"  Reliable cells:          {np.sum(reliable_cells)} / {len(reliable_cells)}")
    print(f"  Combined reliable cells: {np.sum(combined_reliable)} / {len(combined_reliable)}")

    # ------------------------------------------------------------------
    # 9. Prepare continuous temporal data (for speed tuning analysis)
    # ------------------------------------------------------------------
    print(f"\nStep 9: Preparing temporal data...")

    temporal_spikes   = np.concatenate(all_filtered_spks, axis=1)
    temporal_location = np.concatenate(all_filtered_location)
    temporal_speed    = np.concatenate(all_filtered_speed)

    lap_starts = []
    lap_ends   = []
    cumsum     = 0
    for lap_speed in all_filtered_speed:
        lap_starts.append(cumsum)
        cumsum += len(lap_speed)
        lap_ends.append(cumsum)

    lap_starts = np.array(lap_starts)
    lap_ends   = np.array(lap_ends)

    assert temporal_spikes.shape[1] == len(temporal_speed),    "Spike/speed dimension mismatch!"
    assert len(temporal_speed) == len(temporal_location),       "Speed/location dimension mismatch!"
    assert lap_ends[-1] == len(temporal_speed),                 "Lap boundaries don't match data length!"

    print(f"  Temporal spikes:   {temporal_spikes.shape}  (cells x frames)")
    print(f"  Temporal speed:    {len(temporal_speed)} frames")
    print(f"  Total laps:        {len(lap_starts)}")

    # ------------------------------------------------------------------
    # 10. Response plots
    # ------------------------------------------------------------------
    print(f"\nStep 10: Generating response plots...")

    combined_save_dir = os.path.join(suite2p_path, 'combined_reliable_cell_plots')
    reliable_save_dir = os.path.join(suite2p_path, 'reliable_cell_plots')
    os.makedirs(combined_save_dir, exist_ok=True)
    os.makedirs(reliable_save_dir, exist_ok=True)

    pdf_path, stats = RT.plot_individual_reliable_cells_to_pdf(
        spatial_activity=normalized_spatial_activity,
        reliable_cells=combined_reliable,
        save_directory=combined_save_dir,
        avg_cc=avg_cc,
        cohen_d=cohens_d,
        bin_centers=bin_centers,
        normalize=True,
        dpi=150,
        cells_per_page=4
    )

    fig1, _ = RV.create_response_plot(normalized_spatial_activity, reliable_cells,   clim=(0, 1))
    fig1.savefig(os.path.join(reliable_save_dir, 'reliable_cells.png'), dpi=150)
    plt.close(fig1)

    fig2, _ = RV.create_response_plot(normalized_spatial_activity, combined_reliable, clim=(0, 1))
    fig2.savefig(os.path.join(combined_save_dir, 'combined_reliable_cells.png'), dpi=150)
    plt.close(fig2)

    # ------------------------------------------------------------------
    # 11. Build output dictionary and save to HDF5
    # ------------------------------------------------------------------
    print(f"\nStep 11: Saving preprocessed data...")

    # Compute median pixel coordinates for each cell (used in cross-session matching)
    med_coords = np.zeros((len(stat), 2))
    for i, cell_stat in enumerate(stat):
        med_coords[i, 0] = np.median(cell_stat['ypix'])   # y
        med_coords[i, 1] = np.median(cell_stat['xpix'])   # x

    serializable_stat = convert_stat_to_serializable(stat)
    serializable_ops  = convert_ops_to_serializable(ops)

    # Build a metadata string describing which recordings contributed
    recording_sources = [
        f"rec{i+1}: twop={os.path.basename(rp[0])}, vr={os.path.basename(rp[1])}"
        for i, rp in enumerate(resolved_pairs)
    ]

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
        'stat_serializable':     serializable_stat,
        'ops_serializable':      serializable_ops,

        # Temporal data (for speed tuning analysis)
        'speed_cm_s':             temporal_speed.astype(np.float64),
        'smoothed_spks_temporal': temporal_spikes.astype(np.float64),
        'location_cm':            temporal_location.astype(np.float64),
        'lap_starts':             lap_starts.astype(np.int32),
        'lap_ends':               lap_ends.astype(np.int32),

        # Metadata
        'suite2p_path':          str(suite2p_path),
        'n_recordings_combined': int(n_recordings),
        'recording_sources':     str(recording_sources),   # stored as string for HDF5 compat
        'processing_timestamp':  datetime.datetime.now().isoformat(),
        'processing_params': {
            'framerate':                   float(framerate),
            'optimal_offset':              int(optimal_offset),
            'window_cm':                   float(window_cm),
            'min_trial_duration_seconds':  float(min_trial_duration_seconds),
            'max_trial_duration_seconds':  float(max_trial_duration_seconds),
            'min_speed_cm_s':              float(2.0),
            'single_lap_treadmill':        float(single_lap_treadmill),
            'frames_per_recording':        np.array(frames_per_rec, dtype=np.int32),
        }
    }

    # Save
    _savepath = os.path.join(suite2p_path, f'{date}_{animal_id}_preproc_multi.h5')
    print(f"  Writing to {_savepath}")

    try:
        files.write_h5(_savepath, preprocessed_dict)
        print("  Successfully saved preprocessed data!")
    except Exception as e:
        print(f"  Error saving full data: {e}")
        print("  Retrying without stat/ops...")
        minimal_dict = {k: v for k, v in preprocessed_dict.items()
                        if k not in ['stat_serializable', 'ops_serializable']}
        try:
            files.write_h5(_savepath, minimal_dict)
            print("  Saved minimal data (without stat/ops).")
        except Exception as e2:
            print(f"  Failed to save even minimal data: {e2}")

    return preprocessed_dict


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Configure your session here.
    #
    # suite2p_path    : the folder that contains 'suite2p/plane0/' -- this is
    #                   the 'save path' you selected in the suite2p GUI.
    #                   Typically this is the first recording's TSeries folder.
    #
    # recording_pairs : list of (twop_path, vr_path) tuples, ONE PER RECORDING,
    #                   in the SAME ORDER you added them to suite2p.
    #                   Set vr_path to None to auto-detect VRlog*.txt inside
    #                   the corresponding twop_path.
    # --------------------------------------------------------------------------

    # Each entry = one day's session where two recordings need to be combined.
    # suite2p_path    : folder containing suite2p/plane0/ (the GUI save path,
    #                   typically the first TSeries folder for that day).
    # recording_pairs : (twop_path, vr_path) for each recording IN SUITE2P ORDER.
    #                   vr_path=None  →  auto-detect VRlog*.txt inside twop_path.
    #                   vr_path='...' →  explicit path (use when VRlog is in a
    #                                    separate folder, as with older sessions).

    BASE = r'F:\2P\unprocessed'

    sessions = [
        # --- Day 2 ---
        {
            'suite2p_path': rf'{BASE}\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001',
            'recording_pairs': [
                (rf'{BASE}\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001', None),
                (rf'{BASE}\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-002', None),
            ]
        },
        # Add further days following the same pattern above.
    ]

    n_total    = len(sessions)
    successful = []
    failed     = []

    for i, session in enumerate(sessions):
        print(f"\n{'='*80}")
        print(f"Session {i+1}/{n_total}: {os.path.basename(session['suite2p_path'])}")
        print(f"{'='*80}")

        try:
            preprocess_2pVR_multi(
                suite2p_path    = session['suite2p_path'],
                recording_pairs = session['recording_pairs']
            )
            successful.append(session['suite2p_path'])
        except Exception as e:
            failed.append((session['suite2p_path'], str(e)))
            print(f"FAILED: {e}")

    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE")
    print(f"  Successful: {len(successful)}/{n_total}")
    print(f"  Failed:     {len(failed)}/{n_total}")
    if failed:
        print("\nFailed sessions:")
        for path, err in failed:
            print(f"  {os.path.basename(path)}: {err}")
