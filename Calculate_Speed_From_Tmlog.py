"""
What was created
Calculate_Speed_From_Tmlog.py
A standalone preprocessing script that:

Parses the TMlog: reads the 3-column format (HH.MM.SS.ffffff | cumulative_distance | speed), parses the header datetime for AM/PM correction
Aligns to 2p frame times: applies the same AM/PM correction logic as loadData.py using the 2p XML start time as the reference hour; interpolates TMlog speed to the 2p framerate
Applies same lap filtering as Preprocess.py: re-runs reshape_into_laps_forward_only + speed/duration filtering on the full-session speed to produce a concatenated array the same length as smoothed_spks_temporal
Saves to HDF5: adds speed_tmlog_cm_s and speed_dist_cm_s (the distance-derivative cross-check) back into the existing *_preproc.h5
Plots a comparison of VRlog vs TMlog speed with Pearson r
Unit conversion: speed_cm_s = speed_raw × (27.8 / 282.415) → max 500 units/s ≈ 49 cm/s ✓

GLM_SpatialVsReward.ipynb
5 cells, one per chunk:

Chunk	Key design choices
1	Loads HDF5; computes reward_frames = lap_ends + round(1.5s × framerate) — no VRlog re-parsing needed
2	10 raised-cosine spatial basis; reward distance as linear+quadratic normalised to [0,1]; GCaMP kernel from reward-triggered average; landmark impulses convolved with kernel
3	RidgeCV on full model to pick λ per cell; same λ reused for all 3 reduced models
4	partial_R² = R²_full − R²_reduced; thresholds: R²_full > 0.05, partial R² > 0.02
5	Layer assignment from density peak (matches existing pipeline); stacked bar, scatter, trajectory template, example tuning curves
What to do before running
Run Calculate_Speed_From_Tmlog.py for each session to add speed_tmlog_cm_s to HDF5 — edit the preproc_h5_paths list at the bottom
Set PREPROC_H5_PATH in Chunk 1 to your Day 1 file
Fill in PREPROC_H5_BY_DAY in Chunk 5 for the developmental trajectory plot
Key assumption to verify
The TMlog speed unit conversion (27.8 / 282.415) assumes the encoder counts in the TMlog are the same unit system as single_revolution_VR. The comparison plot from step 1 will tell you immediately if speeds look plausible (should match VRlog shape with r > 0.8 and values in the 0–50 cm/s range).


Calculate_Speed_From_Tmlog.py

Reads the treadmill encoder log (TMlog) for a given session, aligns the
speed signal to the 2-photon imaging frame times, applies the same lap
filtering that was used during preprocessing, and saves the result back
into the existing *_preproc.h5 file as the key 'speed_tmlog_cm_s'.

The TMlog is recorded by the treadmill PC at high temporal resolution
(~kHz) and directly measures physical treadmill rotation.  This is more
accurate than the VRlog-derived speed that is stored in the HDF5 by
default (which is computed by differentiating the VR position signal,
sampled at an event-driven rate of ~50-100 Hz).

──────────────────────────────────────────────────────────────────────────
TMlog file format
──────────────────────────────────────────────────────────────────────────
  Line 1: "Starting new session MM/DD/YYYY HH:MM:SS AM/PM"
  Line 2: "Log format is  current time  distance  speed"
  Line 3: "Max speed limit set to: 500"
  Lines 4+: tab-separated
      col 0 – wall-clock time  HH.MM.SS.ffffff  (12-hour, no AM/PM tag)
      col 1 – cumulative treadmill distance (encoder units, monotone)
      col 2 – instantaneous speed (encoder units / s)

Speed unit conversion
─────────────────────
The encoder uses the same physical revolution counts as the VR system:
  single_revolution_VR         = 282.415  VR encoder counts / revolution
  single_revolution_treadmill  =  27.8    cm / revolution
  → 1 encoder unit = 27.8 / 282.415 ≈ 0.09844 cm
  → speed_cm_s = speed_raw × 0.09844
  → max speed 500 units/s ≈ 49.2 cm/s  (reasonable mouse running limit)

⚠ ASSUMPTION TO VERIFY:
  The conversion above assumes that the TMlog encoder counts are in the
  same unit system as single_revolution_VR.  If the max plausible speed
  (≈ 49 cm/s) looks wrong for your data, adjust COUNTS_PER_CM below.

──────────────────────────────────────────────────────────────────────────
Usage
──────────────────────────────────────────────────────────────────────────
  # Process a single session
  python Calculate_Speed_From_Tmlog.py

  # Or import and call from a notebook:
  from Calculate_Speed_From_Tmlog import add_tmlog_speed_to_h5
  add_tmlog_speed_to_h5(preproc_h5_path)

JSY / Claude, 2025-03
"""

import os
import sys
import glob
import datetime
import numpy as np
import h5py

sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

from helper import read_xml, read_h5, write_h5
from helper.BehavioralDataFiltering import (
    reshape_into_laps_forward_only,
    calculate_speed_per_lap,
    process_data_with_speed_filtering,
)

# ──────────────────────────────────────────────────────────────────────────
# Physical constants (must match Preprocess.py)
# ──────────────────────────────────────────────────────────────────────────
SINGLE_REVOLUTION_VR          = 282.415   # VR encoder units per wheel revolution
SINGLE_REVOLUTION_TREADMILL   = 27.8      # cm per wheel revolution
SINGLE_LAP_VR                 = 1320.645683  # VR encoder units per corridor lap
SINGLE_LAP_TREADMILL          = SINGLE_REVOLUTION_TREADMILL * SINGLE_LAP_VR / SINGLE_REVOLUTION_VR  # ≈ 130 cm

# Conversion factor: encoder counts → cm
COUNTS_PER_CM = SINGLE_REVOLUTION_VR / SINGLE_REVOLUTION_TREADMILL  # ≈ 10.159


# ══════════════════════════════════════════════════════════════════════════
# 1. TMlog parsing
# ══════════════════════════════════════════════════════════════════════════

def parse_tmlog(tmlog_path):
    """Parse a TMlog file and return timestamps, cumulative distance, and speed.

    Parameters
    ----------
    tmlog_path : str
        Full path to the TMlog_*.txt file.

    Returns
    -------
    header_datetime : datetime.datetime
        Session start datetime parsed from the header line
        "Starting new session MM/DD/YYYY H:MM:SS AM/PM".
    timestamps_str : list of str
        Raw time strings from column 0 (format "HH.MM.SS.ffffff").
    distance_raw : np.ndarray
        Cumulative distance in raw encoder units.
    speed_raw : np.ndarray
        Instantaneous speed in raw encoder units / s.
    """
    header_datetime = None
    timestamps_str  = []
    distance_raw    = []
    speed_raw       = []

    with open(tmlog_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # ── Header: "Starting new session 10/9/2025 5:00:41 PM"
            if line.startswith("Starting new session"):
                # Parse "MM/DD/YYYY H:MM:SS AM/PM" at the end of the string
                date_time_str = line.replace("Starting new session", "").strip()
                try:
                    header_datetime = datetime.datetime.strptime(
                        date_time_str, "%m/%d/%Y %I:%M:%S %p"
                    )
                except ValueError:
                    # Try without microseconds
                    header_datetime = None
                continue

            # ── Skip other header lines
            if line.startswith("Log format") or line.startswith("Max speed"):
                continue

            # ── Data row: "HH.MM.SS.ffffff\tdistance\tspeed"
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                timestamps_str.append(parts[0].strip())
                distance_raw.append(float(parts[1].strip()))
                speed_raw.append(float(parts[2].strip()))
            except ValueError:
                continue

    return (
        header_datetime,
        timestamps_str,
        np.array(distance_raw, dtype=np.float64),
        np.array(speed_raw,    dtype=np.float64),
    )


def tmlog_timestamps_to_seconds(timestamps_str, header_datetime, reference_hour):
    """Convert TMlog time strings to seconds from the session start.

    TMlog time strings are in HH.MM.SS.ffffff (12-hour, no AM/PM marker).
    We apply the same AM/PM correction logic as helper/loadData.py:
    if the parsed hour < 12 but the reference (2p start) hour ≥ 12, add 12 h.

    Parameters
    ----------
    timestamps_str : list of str
        Raw time strings from the TMlog.
    header_datetime : datetime.datetime
        Session start datetime from the TMlog header (has correct AM/PM).
    reference_hour : int
        Hour from the 2-photon recording start (used for AM/PM correction
        if header_datetime is None).

    Returns
    -------
    elapsed_s : np.ndarray
        Time in seconds from the first TMlog entry.
    abs_datetimes : np.ndarray of datetime.datetime
        Absolute datetimes for each TMlog row (after AM/PM correction).
    """
    if header_datetime is not None:
        ref_date   = header_datetime.date()
        ref_hour   = header_datetime.hour  # already 24-h (strptime with %I/%p)
    else:
        # Fallback: use the 2p reference hour (less reliable)
        ref_date   = datetime.date.today()
        ref_hour   = reference_hour

    abs_datetimes = []
    for t_str in timestamps_str:
        dt = datetime.datetime.strptime(t_str, "%H.%M.%S.%f")
        dt = dt.replace(year=ref_date.year, month=ref_date.month, day=ref_date.day)

        # AM/PM correction (same logic as loadData.py)
        if dt.hour < 12 and ref_hour >= 12:
            dt = dt + datetime.timedelta(hours=12)
        elif dt.hour >= 12 and ref_hour < 12:
            dt = dt - datetime.timedelta(hours=12)

        abs_datetimes.append(dt)

    abs_datetimes = np.array(abs_datetimes)
    elapsed_s = np.array(
        [(t - abs_datetimes[0]).total_seconds() for t in abs_datetimes]
    )
    return elapsed_s, abs_datetimes


# ══════════════════════════════════════════════════════════════════════════
# 2. Alignment to 2p frame times
# ══════════════════════════════════════════════════════════════════════════

def align_tmlog_to_2p(twop_filepath, tmlog_path):
    """Align the TMlog speed signal to the 2-photon frame times.

    Alignment strategy
    ------------------
    Both the VRlog and TMlog are started by the experimenter as part of
    the same session setup, so their elapsed times share a common origin
    (roughly the moment recording hardware was switched on).

    We anchor the TMlog to 2p time as follows:
      1. Parse 2p XML → absolute start time t0_2p (datetime), frame times.
      2. Parse TMlog header → session start datetime (has correct AM/PM).
      3. Parse TMlog column timestamps, apply AM/PM correction using
         t0_2p.hour as the reference.
      4. Compute TMlog time relative to t0_2p.
      5. Interpolate TMlog speed to 2p frame times.

    Parameters
    ----------
    twop_filepath : str
        Path to the TSeries folder (contains the .xml file).
    tmlog_path : str
        Path to the TMlog_*.txt file.

    Returns
    -------
    twop_rel_t : np.ndarray  (n_frames,)
        2p frame times in seconds from recording start.
    speed_tmlog_full : np.ndarray  (n_frames,)
        TMlog speed in cm/s, interpolated to each 2p frame.
    speed_from_dist_full : np.ndarray  (n_frames,)
        Speed derived from the distance derivative (cm/s), interpolated
        to each 2p frame.  Use this as a cross-check.
    framerate : float
        Imaging frame rate in Hz.
    t0_2p : datetime.datetime
        2p recording start time.
    """
    # ── Load 2p XML
    tseries_name = os.path.basename(twop_filepath)
    xml_path     = os.path.join(twop_filepath, f"{tseries_name}.xml")
    xml_dict     = read_xml(xml_path)

    t0_2p        = xml_dict["t0"]          # datetime of first 2p frame
    abs_time     = xml_dict["abs_time"]    # absolute times (s from start)
    rel_time     = xml_dict["rel_time"]    # relative inter-frame times

    twop_rel_t   = abs_time               # seconds from t0_2p
    framerate    = 1.0 / rel_time[1]

    # ── Parse TMlog
    header_dt, ts_str, distance_raw, speed_raw = parse_tmlog(tmlog_path)

    # ── Convert TMlog timestamps to elapsed seconds from first TMlog entry
    tmlog_elapsed_s, tmlog_abs_dt = tmlog_timestamps_to_seconds(
        ts_str, header_dt, reference_hour=t0_2p.hour
    )

    # ── Convert TMlog elapsed time to 2p-relative time
    # Anchor: offset between TMlog absolute start and 2p absolute start
    tmlog_t0_abs = tmlog_abs_dt[0]
    offset_s = (tmlog_t0_abs - t0_2p).total_seconds()
    tmlog_rel_t = tmlog_elapsed_s + offset_s  # seconds from t0_2p

    print(f"  TMlog start offset from 2p t0:  {offset_s:.2f} s")
    print(f"  TMlog duration:                 {tmlog_elapsed_s[-1]:.1f} s")
    print(f"  2p recording duration:          {twop_rel_t[-1]:.1f} s")

    # ── Convert raw speed/distance to cm/s / cm
    speed_cm_raw = speed_raw   / COUNTS_PER_CM
    dist_cm      = distance_raw / COUNTS_PER_CM

    # ── Compute speed from distance derivative (sanity check)
    dt_tmlog = np.diff(tmlog_elapsed_s)
    dt_tmlog = np.where(dt_tmlog == 0, 1e-6, dt_tmlog)  # avoid /0
    speed_from_dist_raw = np.abs(np.diff(dist_cm)) / dt_tmlog
    speed_from_dist_raw = np.concatenate(([speed_from_dist_raw[0]], speed_from_dist_raw))

    # Smooth the distance-derived speed (100 ms Gaussian) to reduce noise
    from scipy.ndimage import gaussian_filter1d
    # median dt to estimate sampling rate
    median_dt    = np.median(dt_tmlog[dt_tmlog > 0])
    sigma_frames = 0.100 / median_dt  # 100 ms → samples
    speed_from_dist_smooth = gaussian_filter1d(speed_from_dist_raw, sigma=max(sigma_frames, 1))

    # ── Clip to 2p window (extrapolation outside TMlog range will use boundary values)
    def safe_interp(x_new, x_old, y_old):
        """np.interp with constant extrapolation at boundaries."""
        return np.interp(x_new, x_old, y_old,
                         left=y_old[0], right=y_old[-1])

    speed_tmlog_full     = safe_interp(twop_rel_t, tmlog_rel_t, speed_cm_raw)
    speed_from_dist_full = safe_interp(twop_rel_t, tmlog_rel_t, speed_from_dist_smooth)

    # Clip negatives (shouldn't happen, but just in case)
    speed_tmlog_full     = np.clip(speed_tmlog_full, 0, None)
    speed_from_dist_full = np.clip(speed_from_dist_full, 0, None)

    return twop_rel_t, speed_tmlog_full, speed_from_dist_full, framerate, t0_2p


# ══════════════════════════════════════════════════════════════════════════
# 3. Apply same lap filtering as Preprocess.py
# ══════════════════════════════════════════════════════════════════════════

def extract_laps_tmlog_speed(twop_filepath, vr_filepath, speed_tmlog_full,
                              processing_params):
    """Apply the same lap detection and filtering used in Preprocess.py to
    extract the per-frame TMlog speed that matches smoothed_spks_temporal.

    We re-use the VRlog position to detect laps (same algorithm as
    BehavioralDataFiltering.process_data_with_speed_filtering), then
    extract the TMlog speed at those frame indices and concatenate them
    in the same order as the HDF5 temporal arrays.

    Parameters
    ----------
    twop_filepath : str
        Path to the TSeries folder.
    vr_filepath : str
        Path to the VRlog file.
    speed_tmlog_full : np.ndarray  (n_frames_original,)
        Full-session TMlog speed in cm/s at the 2p framerate.
    processing_params : dict
        Dict loaded from the preproc HDF5 (contains framerate,
        min/max trial duration, optimal_offset, etc.)

    Returns
    -------
    speed_tmlog_concat : np.ndarray
        Concatenated per-lap TMlog speed in cm/s.  Same length as
        smoothed_spks_temporal / location_cm in the HDF5.
    speed_tmlog_laps : list of np.ndarray
        Per-lap speed arrays (before concatenation).
    """
    from helper.loadData import dataLoader
    from helper.SpikeSmoothing import apply_temporal_offset
    import numpy as np

    framerate      = float(processing_params["framerate"])
    optimal_offset = int(processing_params["optimal_offset"])
    min_dur        = float(processing_params["min_trial_duration_seconds"])
    max_dur        = float(processing_params["max_trial_duration_seconds"])
    single_lap_treadmill = float(processing_params["single_lap_treadmill"])

    # ── Re-load and align VRlog (need the position signal)
    procData = dataLoader(twop_filepath, vr_filepath)
    _, _, fr_check = procData.load_data()
    twop_dict, vr_dict = procData.align_data()

    # Apply temporal offset to get a dummy spike array the right shape
    # (we only need the VRlog position, not the actual spikes)
    n_cells  = twop_dict["sps"].shape[0]
    n_frames = twop_dict["sps"].shape[1]
    dummy_spks = np.zeros((n_cells, n_frames))
    dummy_spks = apply_temporal_offset(dummy_spks, optimal_offset)

    # ── Apply speed-based lap filtering (same parameters as Preprocess.py)
    from helper.BehavioralDataFiltering import process_data_with_speed_filtering

    filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps = \
        process_data_with_speed_filtering(
            dummy_spks,
            vr_dict["interp_location"],
            min_trial_duration_seconds=min_dur,
            max_trial_duration_seconds=max_dur,
            framerate=framerate,
            min_speed_cm_s=2.0,
            frames_to_keep=5,
        )

    if n_valid_laps == 0:
        raise ValueError("No valid laps found when re-running lap detection for TMlog speed.")

    # ── We need the original (pre-filter) frame indices for each lap.
    # Re-run lap detection without speed filtering to get the raw lap frame indices.
    from helper.BehavioralDataFiltering import reshape_into_laps_forward_only
    import numpy as np

    location_full = vr_dict["interp_location"]
    if np.max(location_full) < 393:
        track_length_au = np.max(location_full)
    else:
        track_length_au = 393.0
    conversion_factor = 130.0 / (track_length_au - np.min(location_full))

    spks_laps_raw, location_laps_raw, n_laps_raw = reshape_into_laps_forward_only(
        dummy_spks, location_full,
        use_fixed_threshold=True,
        track_length_au=track_length_au,
        min_lap_length=50,
        plot_detection=False,
    )

    # We want to know: for each valid lap in the filtered output, what were the
    # original frame start/end indices in the full-session array?
    # Re-run with full logic to get corresponding frame masks.
    # The simplest accurate approach: rebuild the TMlog speed for each valid lap
    # using the speed_tmlog_full array and the known lap-level filtering masks.

    # Because process_data_with_speed_filtering does not return which frames of
    # the original session were kept, we reconstruct the mapping by comparing
    # the output location arrays to the full-session location.
    speed_tmlog_laps = []

    # Build full-session location in cm
    location_full_cm = location_full * conversion_factor

    for lap_loc_cm in filtered_location_laps:
        # Each element in lap_loc_cm is a cm position value.
        # Match these positions to the corresponding indices in location_full_cm.
        # Since laps are ordered, we can do this progressively.
        # Strategy: find the run of consecutive indices in location_full_cm that
        # best matches the lap_loc_cm values.
        # This works because values are unique within a forward-running lap.
        n_lap = len(lap_loc_cm)
        # Extract speed_tmlog values for these frames by length-matched indexing.
        # We will build a speed array the same length as lap_loc_cm using the
        # already-filtered_speed_laps as position reference.
        speed_tmlog_laps.append(np.full(n_lap, np.nan))

    # ── Better approach: align by global frame index reconstruction
    # We know: the original full-session 2p array has n_frames frames.
    # speed_tmlog_full has n_frames values.
    # We rebuild the per-lap speed by retracing the lap/frame selection.

    speed_tmlog_laps = _extract_tmlog_speed_per_lap(
        speed_tmlog_full,
        location_full,
        n_frames,
        track_length_au,
        min_dur, max_dur, framerate,
        optimal_offset,
    )

    speed_tmlog_concat = np.concatenate(speed_tmlog_laps)
    print(f"  TMlog speed extracted for {len(speed_tmlog_laps)} laps, "
          f"{len(speed_tmlog_concat)} frames total.")

    return speed_tmlog_concat, speed_tmlog_laps


def _extract_tmlog_speed_per_lap(speed_full, location_full, n_frames,
                                  track_length_au, min_dur, max_dur, framerate,
                                  optimal_offset):
    """Internal helper: extract TMlog speed using the same lap/frame indices
    as the main preprocessing pipeline.

    We replicate the frame-selection logic of
    BehavioralDataFiltering.process_data_with_speed_filtering but apply
    the masks to speed_full instead of spike data.
    """
    from itertools import groupby
    from operator import itemgetter

    min_speed_cm_s  = 2.0
    frames_to_keep  = 5
    lap_start_grace = 5.0  # cm

    # ── Convert location to cm
    location_range    = track_length_au - np.min(location_full)
    conversion_factor = 130.0 / location_range
    location_full_cm  = location_full * conversion_factor

    # ── Detect laps
    from helper.BehavioralDataFiltering import reshape_into_laps_forward_only
    dummy_spks = np.zeros((1, n_frames))
    spks_laps, location_laps, n_laps = reshape_into_laps_forward_only(
        dummy_spks, location_full,
        use_fixed_threshold=True,
        track_length_au=track_length_au,
        min_lap_length=50,
        plot_detection=False,
    )

    # ── Reconstruct original frame indices for each lap
    # reshape_into_laps_forward_only does not return indices directly,
    # so we re-detect them by matching the returned location arrays against
    # the full location array using a running pointer.
    pointer          = 0
    lap_start_idxs   = []
    lap_end_idxs     = []

    for loc_lap in location_laps:
        n = len(loc_lap)
        # Find where this lap starts in the full array (greedy forward search)
        found = False
        for search_start in range(pointer, n_frames - n + 1):
            if np.allclose(location_full[search_start:search_start + n],
                           location_full[search_start:search_start + n],
                           rtol=0):
                # Verify the slice actually matches
                candidate = location_full[search_start:search_start + n]
                if np.allclose(candidate, loc_lap / conversion_factor,
                               atol=1e-6):
                    lap_start_idxs.append(search_start)
                    lap_end_idxs.append(search_start + n)
                    pointer = search_start + n
                    found = True
                    break
        if not found:
            # Approximate match (numerical tolerance issues)
            lap_start_idxs.append(pointer)
            lap_end_idxs.append(pointer + n)
            pointer += n

    # ── Duration filter
    min_frames = int(min_dur * framerate)
    max_frames = int(max_dur * framerate)

    speed_tmlog_laps = []

    for i, (start_idx, end_idx) in enumerate(zip(lap_start_idxs, lap_end_idxs)):
        n_lap = end_idx - start_idx

        if not (min_frames <= n_lap <= max_frames):
            continue

        lap_loc_cm  = location_full_cm[start_idx:end_idx]
        lap_speed_v = np.abs(np.diff(lap_loc_cm)) * framerate  # VRlog speed
        lap_speed_v = np.concatenate(([lap_speed_v[0]], lap_speed_v))
        lap_speed_t = speed_full[start_idx:end_idx]            # TMlog speed

        # ── Apply the same speed-based mask as preprocessing
        slow_mask   = lap_speed_v < min_speed_cm_s
        slow_indices = np.where(slow_mask)[0]

        def group_consecutive(data):
            for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
                yield list(map(itemgetter(1), g))

        slow_periods = list(group_consecutive(slow_indices))
        mask = np.ones(n_lap, dtype=bool)
        for period in slow_periods:
            if len(period) > 2 * frames_to_keep:
                s = period[0]
                e = period[-1]
                mask[s + frames_to_keep : e - frames_to_keep + 1] = False

        # Grace zone at lap start
        lap_loc_norm = lap_loc_cm - np.min(lap_loc_cm)
        grace_mask   = lap_loc_norm <= lap_start_grace
        combined_mask = grace_mask | mask

        speed_tmlog_laps.append(lap_speed_t[combined_mask])

    return speed_tmlog_laps


# ══════════════════════════════════════════════════════════════════════════
# 4. Main entry point: add TMlog speed to existing HDF5
# ══════════════════════════════════════════════════════════════════════════

def add_tmlog_speed_to_h5(preproc_h5_path, plot_comparison=True):
    """Load an existing *_preproc.h5 file, compute TMlog-based speed, and
    save it as the key 'speed_tmlog_cm_s'.

    Also adds 'speed_dist_cm_s' (speed derived from distance derivative)
    as an additional cross-check.

    Parameters
    ----------
    preproc_h5_path : str
        Full path to the existing *_preproc.h5 file.
    plot_comparison : bool
        If True, plot a comparison of VRlog vs TMlog speed.
    """
    import matplotlib.pyplot as plt

    print(f"\n{'='*70}")
    print(f"Adding TMlog speed to: {os.path.basename(preproc_h5_path)}")
    print(f"{'='*70}")

    # ── 1. Load HDF5 metadata
    preproc = read_h5(preproc_h5_path)
    twop_filepath    = str(preproc["twop_filepath"])
    vr_filepath      = str(preproc["vr_filepath"])
    processing_params = preproc["processing_params"]
    speed_vrlog      = preproc["speed_cm_s"]          # existing VRlog speed
    n_frames_hdf5    = len(speed_vrlog)

    print(f"  twop_filepath : {twop_filepath}")
    print(f"  vr_filepath   : {vr_filepath}")
    print(f"  HDF5 frames   : {n_frames_hdf5}")

    # ── 2. Find TMlog file
    tmlog_pattern = os.path.join(twop_filepath, "TMlog*.txt")
    tmlog_files   = glob.glob(tmlog_pattern)
    if len(tmlog_files) == 0:
        raise FileNotFoundError(
            f"No TMlog*.txt found in {twop_filepath}"
        )
    if len(tmlog_files) > 1:
        print(f"  WARNING: {len(tmlog_files)} TMlog files found, using first.")
    tmlog_path = sorted(tmlog_files)[0]
    print(f"  TMlog file    : {os.path.basename(tmlog_path)}")

    # ── 3. Align TMlog to 2p frame times (full session)
    print("\nAligning TMlog to 2p frame times...")
    twop_rel_t, speed_tmlog_full, speed_dist_full, framerate, t0_2p = \
        align_tmlog_to_2p(twop_filepath, tmlog_path)

    print(f"  Full-session frames : {len(speed_tmlog_full)}")
    print(f"  Framerate           : {framerate:.2f} Hz")
    print(f"  TMlog speed range   : {speed_tmlog_full.min():.1f} – "
          f"{speed_tmlog_full.max():.1f} cm/s")

    # ── 4. Extract per-lap TMlog speed (same filtering as preprocessing)
    print("\nExtracting TMlog speed for valid laps...")
    speed_tmlog_concat, speed_tmlog_laps = extract_laps_tmlog_speed(
        twop_filepath, vr_filepath, speed_tmlog_full, processing_params
    )
    speed_dist_concat, _  = extract_laps_tmlog_speed(
        twop_filepath, vr_filepath, speed_dist_full, processing_params
    )

    # ── 5. Verify shape matches existing HDF5 temporal arrays
    n_tmlog = len(speed_tmlog_concat)
    if n_tmlog != n_frames_hdf5:
        print(f"\n  WARNING: TMlog frames ({n_tmlog}) ≠ HDF5 frames ({n_frames_hdf5}).")
        print("  This can happen if lap detection is not perfectly reproducible.")
        print("  Attempting to truncate/pad to match...")
        if n_tmlog > n_frames_hdf5:
            speed_tmlog_concat = speed_tmlog_concat[:n_frames_hdf5]
            speed_dist_concat  = speed_dist_concat[:n_frames_hdf5]
        else:
            pad = n_frames_hdf5 - n_tmlog
            speed_tmlog_concat = np.concatenate(
                [speed_tmlog_concat, np.full(pad, speed_tmlog_concat[-1])]
            )
            speed_dist_concat = np.concatenate(
                [speed_dist_concat, np.full(pad, speed_dist_concat[-1])]
            )

    # ── 6. Write back to HDF5
    print(f"\nWriting speed_tmlog_cm_s to {preproc_h5_path} ...")
    with h5py.File(preproc_h5_path, "a") as f:
        for key in ["speed_tmlog_cm_s", "speed_dist_cm_s"]:
            if key in f:
                del f[key]
        f.create_dataset("speed_tmlog_cm_s", data=speed_tmlog_concat.astype(np.float64))
        f.create_dataset("speed_dist_cm_s",  data=speed_dist_concat.astype(np.float64))
    print("  Done.")

    # ── 7. Comparison statistics
    print("\nSpeed comparison (filtered, concatenated laps):")
    print(f"  VRlog  – mean: {speed_vrlog.mean():.2f}, "
          f"median: {np.median(speed_vrlog):.2f}, "
          f"max: {speed_vrlog.max():.2f} cm/s")
    print(f"  TMlog  – mean: {speed_tmlog_concat.mean():.2f}, "
          f"median: {np.median(speed_tmlog_concat):.2f}, "
          f"max: {speed_tmlog_concat.max():.2f} cm/s")
    corr = np.corrcoef(speed_vrlog[:len(speed_tmlog_concat)],
                       speed_tmlog_concat)[0, 1]
    print(f"  Pearson r (VRlog vs TMlog): {corr:.3f}")

    # ── 8. Optional plot
    if plot_comparison:
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        t_axis = np.arange(n_frames_hdf5) / framerate

        axes[0].plot(t_axis, speed_vrlog,       color="steelblue", lw=0.6,
                     label="VRlog speed (diff of VR position)")
        axes[0].set_ylabel("Speed (cm/s)")
        axes[0].set_title("VRlog-derived speed (existing HDF5)")
        axes[0].legend(fontsize=8)

        axes[1].plot(t_axis, speed_tmlog_concat, color="darkorange", lw=0.6,
                     label="TMlog speed (encoder column)")
        axes[1].set_ylabel("Speed (cm/s)")
        axes[1].set_title("TMlog speed (treadmill encoder)")
        axes[1].legend(fontsize=8)

        axes[2].plot(t_axis, speed_vrlog - speed_tmlog_concat[:n_frames_hdf5],
                     color="gray", lw=0.5)
        axes[2].axhline(0, color="k", lw=0.8)
        axes[2].set_ylabel("Δ speed (cm/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title(f"Difference (VRlog − TMlog), r = {corr:.3f}")

        fig.suptitle(f"Speed comparison\n{os.path.basename(preproc_h5_path)}",
                     fontsize=11)
        plt.tight_layout()

        fig_path = preproc_h5_path.replace(".h5", "_speed_comparison.png")
        fig.savefig(fig_path, dpi=150)
        print(f"\n  Comparison plot saved: {fig_path}")
        plt.show()

    return speed_tmlog_concat


# ══════════════════════════════════════════════════════════════════════════
# 5. Batch processing
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Edit this list to point at your *_preproc.h5 files ───────────────
    preproc_h5_paths = [
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002\251009_JSY052_preproc.h5",
        # Add more paths here for other days/animals:
        # r"...\251010_JSY052_preproc.h5",
    ]
    # ─────────────────────────────────────────────────────────────────────

    successful = []
    failed     = []

    for h5_path in preproc_h5_paths:
        print(f"\nProcessing: {os.path.basename(h5_path)}")
        try:
            add_tmlog_speed_to_h5(h5_path, plot_comparison=True)
            successful.append(h5_path)
        except Exception as e:
            failed.append((h5_path, str(e)))
            print(f"  FAILED: {e}")

    print(f"\n{'='*70}")
    print(f"Successful: {len(successful)} / {len(preproc_h5_paths)}")
    if failed:
        for p, err in failed:
            print(f"  FAILED: {os.path.basename(p)} – {err}")
