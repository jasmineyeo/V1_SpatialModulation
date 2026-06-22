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
            # Try tab first, fall back to any whitespace
            parts = line.split("\t")
            if len(parts) < 3:
                parts = line.split()
            if len(parts) < 3:
                continue
            try:
                timestamps_str.append(parts[0].strip())
                distance_raw.append(float(parts[1].strip()))
                speed_raw.append(float(parts[2].strip()))
            except ValueError:
                continue

    if len(timestamps_str) == 0:
        # Print first lines for diagnosis
        with open(tmlog_path, "r") as _f:
            head = [next(_f, "") for _ in range(8)]
        print(f"  WARNING: parse_tmlog found no data rows in {os.path.basename(tmlog_path)}.")
        print(f"  First lines of file:\n" + "".join(f"    {l}" for l in head))

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

    _TS_FORMATS = ["%H.%M.%S.%f", "%H:%M:%S.%f", "%H.%M.%S", "%H:%M:%S"]

    def _parse_ts(t_str):
        for fmt in _TS_FORMATS:
            try:
                return datetime.datetime.strptime(t_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Cannot parse TMlog timestamp: {t_str!r}")

    abs_datetimes = []
    for t_str in timestamps_str:
        dt = _parse_ts(t_str)
        dt = dt.replace(year=ref_date.year, month=ref_date.month, day=ref_date.day)

        # AM/PM correction (same logic as loadData.py)
        if dt.hour < 12 and ref_hour >= 12:
            dt = dt + datetime.timedelta(hours=12)
        elif dt.hour >= 12 and ref_hour < 12:
            dt = dt - datetime.timedelta(hours=12)

        abs_datetimes.append(dt)

    # Fix noon-crossing: per-timestamp correction flips 12:xx PM → 00:xx AM
    for i in range(1, len(abs_datetimes)):
        delta = (abs_datetimes[i] - abs_datetimes[i-1]).total_seconds()
        if delta < -6 * 3600:
            for j in range(i, len(abs_datetimes)):
                abs_datetimes[j] = abs_datetimes[j] + datetime.timedelta(hours=12)
            break
        elif delta > 6 * 3600:
            for j in range(i, len(abs_datetimes)):
                abs_datetimes[j] = abs_datetimes[j] - datetime.timedelta(hours=12)
            break

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

    framerate    = 1.0 / rel_time[1]

    # Build the 2p frame time axis (seconds from t0_2p).
    # abs_time should be monotonically increasing to ~session_duration_s.
    # For some Bruker sessions nF in the XML counts non-Frame children too,
    # leaving trailing zeros — detect this and fall back to a uniform axis.
    n_frames_xml = len(abs_time)
    if abs_time[-1] < 1.0:
        # Find the last non-zero entry to get the true frame count
        nonzero = np.where(abs_time > 0)[0]
        if len(nonzero) > 0:
            n_frames_xml = nonzero[-1] + 1
            twop_rel_t   = abs_time[:n_frames_xml]
            print(f"  WARNING: abs_time has trailing zeros; trimmed to {n_frames_xml} frames.")
        else:
            # abs_time entirely zero — build uniform axis from framerate
            twop_rel_t = np.arange(n_frames_xml) / framerate
            print(f"  WARNING: abs_time entirely zero; using uniform axis at {framerate:.2f} Hz.")
    else:
        twop_rel_t = abs_time

    # ── Parse TMlog
    header_dt, ts_str, distance_raw, speed_raw = parse_tmlog(tmlog_path)

    # ── Convert TMlog timestamps to elapsed seconds from first TMlog entry
    tmlog_elapsed_s, tmlog_abs_dt = tmlog_timestamps_to_seconds(
        ts_str, header_dt, reference_hour=t0_2p.hour
    )

    if len(tmlog_abs_dt) == 0:
        raise ValueError(
            f"No valid timestamps parsed from TMlog. "
            f"Check file format (expected tab- or space-separated HH.MM.SS.ffffff columns)."
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

def _load_vr_location(twop_filepath, vr_filepath):
    """Load and align VRlog location to 2p frame times — WITHOUT loading suite2p.

    Replicates the relevant parts of loadData.dataLoader.load_data() +
    align_data() but skips TwoP/suite2p loading entirely.  Useful for
    recordings where the suite2p output is in a sibling folder (multi-recording
    sessions where all neural data were processed together in the first TSeries).

    Returns
    -------
    interp_location : np.ndarray  (n_frames,)
        VR position interpolated to each 2p frame (same as
        vr_dict["interp_location"] from dataLoader.align_data()).
    n_frames : int
        Number of 2p frames (length of interp_location).
    """
    import datetime

    # ── Read 2p XML for frame timestamps
    tseries_name = os.path.basename(twop_filepath)
    xml_path     = os.path.join(twop_filepath, f"{tseries_name}.xml")
    xml_dict     = read_xml(xml_path)

    t0_2p    = xml_dict["t0"]
    abs_time = xml_dict["abs_time"]

    # Handle trailing-zero artefact (same logic as align_tmlog_to_2p)
    if abs_time[-1] < 1.0:
        nonzero = np.where(abs_time > 0)[0]
        n_xml   = nonzero[-1] + 1 if len(nonzero) > 0 else len(abs_time)
        abs_time = abs_time[:n_xml]

    # Build absolute datetime for each frame (mirror of loadData: abs_time[:-1])
    twop_abs_t = np.array([
        t0_2p + datetime.timedelta(seconds=float(t))
        for t in abs_time[:-1]
    ])
    twop_rel_t = np.array([t.total_seconds() for t in (twop_abs_t - twop_abs_t[0])])
    n_frames   = len(twop_abs_t)

    # ── Parse VRlog (same as loadData.load_data)
    raw_rows = []
    with open(vr_filepath, "r") as fh:
        for line in fh.readlines()[3:]:
            parts = line.strip().split("\t")
            if len(parts) >= 4:
                raw_rows.append(parts)

    vr_abs_str = np.array([r[0] for r in raw_rows])
    vr_event   = np.array([r[2] for r in raw_rows])
    vr_loc_raw = np.array([float(r[3]) for r in raw_rows])
    vr_loc_raw[vr_loc_raw < 0] = 0.0

    start_idx  = np.where(vr_event == 's')[0][0]
    vr_abs_str = vr_abs_str[start_idx:]
    vr_loc_raw = vr_loc_raw[start_idx:]

    # ── Convert VR timestamps (same AM/PM correction as loadData.align_data)
    ref_date = twop_abs_t[0].date()
    ref_hour = twop_abs_t[0].hour
    vr_abs_dt = []
    for t_str in vr_abs_str:
        dt = datetime.datetime.strptime(t_str, '%H.%M.%S.%f')
        dt = dt.replace(year=ref_date.year, month=ref_date.month, day=ref_date.day)
        if dt.hour < 12 and ref_hour >= 12:
            dt = dt + datetime.timedelta(hours=12)
        elif dt.hour >= 12 and ref_hour < 12:
            dt = dt - datetime.timedelta(hours=12)
        vr_abs_dt.append(dt)

    # Fix noon-crossing: same correction as loadData.align_data
    for i in range(1, len(vr_abs_dt)):
        delta = (vr_abs_dt[i] - vr_abs_dt[i-1]).total_seconds()
        if delta < -6 * 3600:
            for j in range(i, len(vr_abs_dt)):
                vr_abs_dt[j] = vr_abs_dt[j] + datetime.timedelta(hours=12)
            break
        elif delta > 6 * 3600:
            for j in range(i, len(vr_abs_dt)):
                vr_abs_dt[j] = vr_abs_dt[j] - datetime.timedelta(hours=12)
            break

    vr_abs_dt  = np.array(vr_abs_dt)
    vr_rel_t   = np.array([(t - vr_abs_dt[0]).total_seconds() for t in vr_abs_dt])

    # Align VR relative time to 2p: shift by (t0_vr_abs - t0_2p)
    vr_aligned_rel_t = vr_rel_t + (vr_abs_dt[0] - twop_abs_t[0]).total_seconds()

    # Clip to recording duration
    clip_idx = np.searchsorted(vr_aligned_rel_t, twop_rel_t[-1], side='right')
    vr_aligned_rel_t = vr_aligned_rel_t[:clip_idx]
    vr_loc_clip      = vr_loc_raw[:clip_idx]

    # Interpolate location to 2p frame times
    interp_location = np.interp(twop_rel_t, vr_aligned_rel_t, vr_loc_clip)
    return interp_location, n_frames


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
    framerate      = float(processing_params["framerate"])
    optimal_offset = int(processing_params["optimal_offset"])
    min_dur        = float(processing_params["min_trial_duration_seconds"])
    max_dur        = float(processing_params["max_trial_duration_seconds"])

    # ── Align VRlog to 2p frame times without loading suite2p
    location_full, n_loc = _load_vr_location(twop_filepath, vr_filepath)

    # ── Quick lap-count check (re-uses the same filtering parameters)
    dummy_spks = np.zeros((1, n_loc))
    _, _, _, n_valid_laps = process_data_with_speed_filtering(
        dummy_spks,
        location_full,
        min_trial_duration_seconds=min_dur,
        max_trial_duration_seconds=max_dur,
        framerate=framerate,
        min_speed_cm_s=2.0,
        frames_to_keep=5,
    )

    if n_valid_laps == 0:
        raise ValueError("No valid laps found when re-running lap detection for TMlog speed.")

    # loadData.py builds AbsoluteT from abs_time[:-1], so location_full has
    # n_frames_xml-1 elements while speed_tmlog_full has n_frames_xml elements.
    # Clip speed to location length to prevent off-by-one boundary errors.
    speed_tmlog_full = speed_tmlog_full[:n_loc]

    if np.max(location_full) < 393:
        track_length_au = np.max(location_full)
    else:
        track_length_au = 393.0

    # ── Rebuild per-lap TMlog speed using the same frame-selection logic
    speed_tmlog_laps = _extract_tmlog_speed_per_lap(
        speed_tmlog_full,
        location_full,
        n_loc,
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
# 4a. Multi-recording helpers
# ══════════════════════════════════════════════════════════════════════════

def _collect_recording_folders(tseries_folder):
    """Return all sibling TSeries* subfolders of the parent day folder, sorted.

    For a multi-recording session the layout is:
        Day1_folder/
            TSeries-001/   ← tseries_folder (contains preproc_multi.h5)
                TMlog*.txt, VRlog*.txt
            TSeries-002/   ← second recording
                TMlog*.txt, VRlog*.txt
            ...

    Parameters
    ----------
    tseries_folder : str
        The TSeries folder that contains ``preproc_multi.h5``
        (i.e., the first recording folder).

    Returns
    -------
    list of str
        All TSeries* sibling directories under the parent, sorted
        alphabetically/numerically (recording order).
    """
    parent     = os.path.dirname(tseries_folder)
    candidates = sorted([
        c for c in glob.glob(os.path.join(parent, "TSeries*"))
        if os.path.isdir(c)
    ])
    # Fallback: if the glob found nothing, just return the folder itself
    return candidates if candidates else [tseries_folder]


def _process_multi_recording(tseries_folder, processing_params):
    """Extract and concatenate TMlog speed for all recordings in a multi-session.

    Discovers sibling TSeries folders, processes each independently (align
    TMlog → extract lap-filtered speed), then concatenates in recording order.

    Parameters
    ----------
    tseries_folder : str
        The first TSeries folder (same directory as ``preproc_multi.h5``).
    processing_params : dict
        Processing parameters from the ``preproc_multi.h5`` file.

    Returns
    -------
    speed_tmlog_concat : np.ndarray
        Concatenated lap-filtered TMlog speed (cm/s) across all recordings.
    speed_dist_concat : np.ndarray
        Same for the distance-derivative cross-check signal.
    """
    recording_folders = _collect_recording_folders(tseries_folder)
    print(f"  Multi-recording: {len(recording_folders)} TSeries folder(s) found:")
    for rf in recording_folders:
        print(f"    {os.path.basename(rf)}")

    all_speed_tmlog = []
    all_speed_dist  = []

    for rf in recording_folders:
        print(f"\n  --- Recording: {os.path.basename(rf)} ---")

        # ── Find TMlog for this recording
        tmlog_files = sorted(glob.glob(os.path.join(rf, "TMlog*.txt")))
        if len(tmlog_files) == 0:
            raise FileNotFoundError(f"No TMlog*.txt found in {rf}")
        if len(tmlog_files) > 1:
            print(f"    WARNING: {len(tmlog_files)} TMlog files found, using first.")
        tmlog_path = tmlog_files[0]
        print(f"    TMlog : {os.path.basename(tmlog_path)}")

        # ── Find VRlog for this recording
        vr_files = sorted(glob.glob(os.path.join(rf, "VRlog*.txt")))
        if len(vr_files) == 0:
            raise FileNotFoundError(f"No VRlog*.txt found in {rf}")
        if len(vr_files) > 1:
            print(f"    WARNING: {len(vr_files)} VRlog files found, using first.")
        vr_path = vr_files[0]
        print(f"    VRlog : {os.path.basename(vr_path)}")

        # ── Align TMlog to this recording's 2p frame times
        _, speed_tmlog_full, speed_dist_full, _, _ = align_tmlog_to_2p(rf, tmlog_path)

        # ── Extract lap-filtered speed for this recording segment
        speed_tm_rec, _ = extract_laps_tmlog_speed(
            rf, vr_path, speed_tmlog_full, processing_params
        )
        speed_di_rec, _ = extract_laps_tmlog_speed(
            rf, vr_path, speed_dist_full, processing_params
        )

        print(f"    Frames after lap filtering: {len(speed_tm_rec)}")
        all_speed_tmlog.append(speed_tm_rec)
        all_speed_dist.append(speed_di_rec)

    return np.concatenate(all_speed_tmlog), np.concatenate(all_speed_dist)


# ══════════════════════════════════════════════════════════════════════════
# 4b. Main entry point: add TMlog speed to existing HDF5
# ══════════════════════════════════════════════════════════════════════════

def add_tmlog_speed_to_h5(preproc_h5_path, plot_comparison=True,
                          twop_filepath_override=None, vr_filepath_override=None):
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
    twop_filepath_override : str or None
        If provided, use this path instead of the one stored in the h5.
        Useful when the data has moved drives since preprocessing.
    vr_filepath_override : str or None
        If provided, use this VRlog path instead of the one stored in the h5.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams['legend.fontsize'] = 20
    rcParams['axes.labelsize'] = 20
    rcParams['axes.titlesize'] = 25
    rcParams['xtick.labelsize'] = 20
    rcParams['ytick.labelsize'] = 20

    print(f"\n{'='*70}")
    print(f"Adding TMlog speed to: {os.path.basename(preproc_h5_path)}")
    print(f"{'='*70}")

    # ── 1. Load HDF5 metadata
    preproc = read_h5(preproc_h5_path)

    def _decode(val):
        """Return a clean str from either a str or bytes h5 value."""
        if isinstance(val, bytes):
            return val.decode('utf-8')
        return str(val)

    # ── Detect multi-recording FIRST so we can skip keys that don't exist
    is_multi = "multi" in os.path.basename(preproc_h5_path).lower()

    if twop_filepath_override:
        twop_filepath = twop_filepath_override
    elif "twop_filepath" in preproc:
        twop_filepath = _decode(preproc["twop_filepath"])
    else:
        raise KeyError("'twop_filepath' not found in h5 and no override provided")

    # vr_filepath is only needed for single-recording sessions;
    # for multi-recording each TSeries folder has its own VRlog.
    if not is_multi:
        if vr_filepath_override:
            vr_filepath = vr_filepath_override
        elif "vr_filepath" in preproc:
            vr_filepath = _decode(preproc["vr_filepath"])
        else:
            raise KeyError("'vr_filepath' not found in h5 and no override provided")
    else:
        vr_filepath = None  # resolved per-recording inside _process_multi_recording

    processing_params = preproc["processing_params"]
    speed_vrlog      = preproc["speed_cm_s"]          # existing VRlog speed
    n_frames_hdf5    = len(speed_vrlog)

    print(f"  twop_filepath : {twop_filepath}")
    if vr_filepath:
        print(f"  vr_filepath   : {vr_filepath}")
    print(f"  HDF5 frames   : {n_frames_hdf5}")

    if is_multi:
        # ── 3+4 (multi). Each sibling TSeries folder is processed independently;
        #                 results are concatenated in recording order.
        print("\nMulti-recording session detected — processing each TSeries folder separately...")
        speed_tmlog_concat, speed_dist_concat = _process_multi_recording(
            twop_filepath, processing_params
        )
        framerate        = float(processing_params["framerate"])
        speed_tmlog_full = None   # not available as single array for multi

    else:
        # ── 2. Find TMlog file (single-recording)
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

        # ── 2b. If vr_filepath is not overridden, check whether the stored path
        #        exists; if not, try to find a VRlog*.txt in the TSeries folder.
        if not vr_filepath_override and not os.path.isfile(vr_filepath):
            vr_candidates = glob.glob(os.path.join(twop_filepath, "VRlog*.txt"))
            if len(vr_candidates) == 0:
                raise FileNotFoundError(
                    f"VRlog not found at stored path ({vr_filepath}) "
                    f"and no VRlog*.txt found in {twop_filepath}"
                )
            if len(vr_candidates) > 1:
                print(f"  WARNING: {len(vr_candidates)} VRlog files found in TSeries folder, using first.")
            vr_filepath = sorted(vr_candidates)[0]
            print(f"  VRlog (auto)  : {os.path.basename(vr_filepath)}")

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

    # ── 7+8. Full-session VRlog vs TMlog comparison (single-recording only).
    #        Skipped for multi-recording sessions because there is no single
    #        continuous VRlog/TMlog pair to compare against.
    if is_multi:
        print("\nSkipping full-session speed comparison plot (multi-recording session).")
        return speed_tmlog_concat

    # Re-load VRlog and interpolate to 2P frame times (same n_frames as TMlog full).
    # This avoids the lap-filtering mismatch that makes concatenated signals
    # non-comparable frame-by-frame.
    from helper.loadData import dataLoader
    procData = dataLoader(twop_filepath, vr_filepath)
    procData.load_data()
    _, vr_dict_cmp = procData.align_data()

    loc_full    = np.array(vr_dict_cmp["interp_location"], dtype=float)
    n_full      = len(loc_full)

    # Clip to same length as TMlog full-session signal
    n_cmp = min(n_full, len(speed_tmlog_full))
    loc_cmp = loc_full[:n_cmp]

    # Convert to cm and differentiate to get full-session VRlog speed
    loc_range = np.max(loc_cmp) - np.min(loc_cmp)
    if loc_range > 0:
        conv = 130.0 / loc_range
    else:
        conv = 1.0
    loc_cm_full   = loc_cmp * conv
    vr_speed_full = np.abs(np.diff(loc_cm_full)) * framerate
    vr_speed_full = np.concatenate(([vr_speed_full[0]], vr_speed_full))
    # Clip outlier spikes (teleportation) to 60 cm/s for display
    vr_speed_disp = np.clip(vr_speed_full, 0, 60)
    tm_speed_disp = speed_tmlog_full[:n_cmp]

    corr_full = np.corrcoef(vr_speed_disp, tm_speed_disp)[0, 1]

    print("\nSpeed comparison (full session, before lap filtering):")
    print(f"  VRlog  – mean: {vr_speed_full.mean():.2f}, "
          f"median: {np.median(vr_speed_full):.2f}, "
          f"max: {vr_speed_full.max():.2f} cm/s")
    print(f"  TMlog  – mean: {tm_speed_disp.mean():.2f}, "
          f"median: {np.median(tm_speed_disp):.2f}, "
          f"max: {tm_speed_disp.max():.2f} cm/s")
    print(f"  Pearson r (VRlog vs TMlog, full session): {corr_full:.3f}")

    # ── 8. Optional plot (full-session comparison)
    if plot_comparison:
        t_axis_full = twop_rel_t[:n_cmp]

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        axes[0].plot(t_axis_full, vr_speed_disp, color="steelblue", lw=0.6,
                     label="VRlog speed (clipped to 60 cm/s)")
        axes[0].set_ylabel("Speed (cm/s)")
        axes[0].set_title("VRlog-derived speed (full session)")
        axes[0].legend(fontsize=8)

        axes[1].plot(t_axis_full, tm_speed_disp, color="darkorange", lw=0.6,
                     label="TMlog speed (encoder column)")
        axes[1].set_ylabel("Speed (cm/s)")
        axes[1].set_title("TMlog speed (full session)")
        axes[1].legend(fontsize=8)

        axes[2].plot(t_axis_full, vr_speed_disp - tm_speed_disp,
                     color="gray", lw=0.5)
        axes[2].axhline(0, color="k", lw=0.8)
        axes[2].set_ylabel("Δ speed (cm/s)")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title(f"Difference (VRlog − TMlog), full-session r = {corr_full:.3f}")

        fig.suptitle(f"Speed comparison (full session)\n{os.path.basename(preproc_h5_path)}",
                     fontsize=11)
        plt.tight_layout()

        fig_path = preproc_h5_path.replace(".h5", "_speed_comparison.png")
        
        fig.savefig(fig_path, dpi=150)
        print(f"\n  Comparison plot saved: {fig_path}")
        # plt.show()

    return speed_tmlog_concat


# ══════════════════════════════════════════════════════════════════════════
# 5. Batch processing
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Edit this list to point at your *_preproc.h5 files ───────────────
    # Each entry is a path to a *_preproc.h5 file.
    # twop_filepath is always overridden to the h5's own folder so that
    # TMlog and VRlog are found there regardless of what was stored at preprocessing.
    preproc_h5_paths = [
        # Day 1 — preproc_multi.h5 may lack twop/vr_filepath keys; tseries folder override handles both
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001\10302025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001\10312025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001\11012025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001\11022025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001\11032025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001\11042025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001\11052025_JSY038_preproc.h5",
        # # Add other animals below:
        # r"D:\...\JSY052_ChronicImaging\...\XXXXXX_JSY052_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251205_JSY_JSY055_SpatialModulation_Day1\TSeries-12052025-1740-001\12052025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251206_JSY_JSY055_SpatialModulation_Day2\TSeries-12062025-1810-001\12062025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251207_JSY_JSY055_SpatialModulation_Day3\TSeries-12072025-1825-001\12072025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251208_JSY_JSY055_SpatialModulation_Day4\TSeries-12082025-1633-001\12082025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251209_JSY_JSY055_SpatialModualtion_Day5\TSeries-12092025-2000-001\12092025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251210_JSY_JSY055_SpatialModulation_Day6\TSeries-12102025-1702-001\12102025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251211_JSY_JSY055_SpatialModulation_Day7\TSeries-12112025-1631-001\12112025_JSY038_preproc.h5",
        
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002\10092025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001\10102025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002\10112025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001\10122025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001\10132025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003\10142025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001\10152025_JSY038_preproc.h5",
        
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001\11012025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001\11022025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001\11032025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001\11042025_JSY038_preproc.h5",
        r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002\11052025_JSY038_preproc.h5",
        
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001\09062025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-001\09072025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3\TSeries-09082025-1540-001\09082025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001\09092025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001\09102025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001\09112025_JSY038_preproc_multi.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001\09122025_JSY038_preproc_multi.h5",
        
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001\06162025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001\06182025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001\06202025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001\06222025_JSY038_preproc.h5",
        
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001\06202025_JSY038_preproc.h5",
        # r"D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001\06222025_JSY038_preproc.h5",
    ]   
    # ─────────────────────────────────────────────────────────────────────

    successful = []
    failed     = []

    for entry in preproc_h5_paths:
        # Entry is either a plain path string or (h5_path, vr_filepath_override)
        if isinstance(entry, tuple):
            h5_path, vr_override = entry
        else:
            h5_path, vr_override = entry, None

        print(f"\nProcessing: {os.path.basename(h5_path)}")
        # Always use the folder containing the h5 as twop_filepath,
        # so the TMlog is found regardless of what drive was used at preprocessing.
        tseries_folder = os.path.dirname(h5_path)
        try:
            add_tmlog_speed_to_h5(h5_path, plot_comparison=True,
                                  twop_filepath_override=tseries_folder,
                                  vr_filepath_override=vr_override)
            successful.append(h5_path)
        except Exception as e:
            failed.append((h5_path, str(e)))
            print(f"  FAILED: {e}")

    print(f"\n{'='*70}")
    print(f"Successful: {len(successful)} / {len(preproc_h5_paths)}")
    if failed:
        for p, err in failed:
            print(f"  FAILED: {os.path.basename(p)} – {err}")
