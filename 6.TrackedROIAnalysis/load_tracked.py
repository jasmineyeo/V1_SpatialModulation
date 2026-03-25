"""
load_tracked.py
Shared data-loading utilities for tracked-ROI analyses.

All scripts in 6.TrackedROIAnalysis/ import from here.

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import h5py
from glob import glob


# ============================================================
# ROI tracking
# ============================================================

def load_tracking(roi_tracking_file):
    """
    Load the tracked ROI matrix from roi_tracking_*.h5.

    Handles two subtleties:
    - day_labels stored in the h5 may list ALL sessions (including the
      registration reference) while tracked_matrix only has columns for
      required_days (the subset that cells must be present in).
      We use required_days when it exists and its length matches the matrix.
    - plane0_path is .../TSeries-.../suite2p/plane0; going up 2 levels
      gives the TSeries directory where preproc/smi files are saved.

    Returns
    -------
    tracked_matrix : np.ndarray, shape (n_tracked_cells, n_sessions)
        ROI index in each session (-1 = absent).
    day_labels : list of str
        Labels for each column of tracked_matrix (the tracked sessions only).
    session_dirs : dict {day_label: str}
        Path to the TSeries folder for each tracked day.
        Empty dict if sessions group is absent from the h5.
    """
    with h5py.File(roi_tracking_file, 'r') as f:
        if 'tracked_matrix' in f:
            tracked_matrix = f['tracked_matrix'][:]
        elif 'verified_matrix' in f:
            print("  WARNING: tracked_matrix not found, using verified_matrix")
            tracked_matrix = f['verified_matrix'][:]
        else:
            raise KeyError("No tracked_matrix or verified_matrix in roi_tracking file.")

        n_cols = tracked_matrix.shape[1]

        def _decode_list(raw):
            return [d.decode() if isinstance(d, bytes) else str(d) for d in raw]

        # Prefer required_days (matches tracked_matrix columns) over day_labels
        # (which may include the reference session not in the matrix).
        req_raw = f.attrs.get('required_days', None)
        all_raw = f.attrs.get('day_labels',    None)

        if req_raw is not None and len(req_raw) == n_cols:
            day_labels = _decode_list(req_raw)
        elif all_raw is not None and len(all_raw) == n_cols:
            day_labels = _decode_list(all_raw)
        elif all_raw is not None and len(all_raw) > n_cols:
            # day_labels has more entries than matrix columns — drop entries
            # whose sessions group has no tracked data (typically Day1 reference).
            all_labels = _decode_list(all_raw)
            # Heuristic: keep the last n_cols labels (reference day is usually first)
            day_labels = all_labels[-n_cols:]
            print(f"  INFO: day_labels ({len(all_labels)}) > matrix columns ({n_cols}). "
                  f"Using last {n_cols}: {day_labels}")
        else:
            day_labels = [f"Day{i+1}" for i in range(n_cols)]
            print(f"  WARNING: day_labels not found or mismatched, inferred: {day_labels}")

        # Read per-session plane0 paths -> derive TSeries directory (2 levels up)
        session_dirs = {}
        if 'sessions' in f:
            for day_key in f['sessions'].keys():
                grp = f['sessions'][day_key]
                if 'plane0_path' in grp.attrs:
                    raw_path = grp.attrs['plane0_path']
                    plane0 = raw_path.decode() if isinstance(raw_path, bytes) else str(raw_path)
                    # plane0 = .../TSeries-.../suite2p/plane0
                    #   1 up  -> suite2p
                    #   2 up  -> TSeries-...   <- preproc/smi files live here
                    tseries_dir = str(os.path.dirname(os.path.dirname(plane0)))
                    session_dirs[day_key] = tseries_dir

    print(f"  Tracked cells: {tracked_matrix.shape[0]}, Sessions: {n_cols}")
    print(f"  Day labels   : {day_labels}")
    if session_dirs:
        missing_dirs = [d for d in day_labels if d not in session_dirs]
        if missing_dirs:
            print(f"  WARNING: no plane0_path recorded for: {missing_dirs}")
    else:
        print("  WARNING: no 'sessions' group in roi_tracking file — "
              "use find_smi_files(animal_dir) instead")

    return tracked_matrix, day_labels, session_dirs


# ============================================================
# File discovery
# ============================================================

def _find_files_by_pattern(animal_dir, pattern):
    """Return dict {day_label: filepath} found recursively."""
    files = glob(os.path.join(animal_dir, "**", pattern), recursive=True)
    day_map = {}
    for f in sorted(files):
        m = re.search(r'Day(\d+)', f, re.IGNORECASE)
        if m:
            key = f"Day{int(m.group(1))}"
            if key not in day_map:
                day_map[key] = f
    return day_map


def find_smi_files(animal_dir, pattern="*_smi_results.h5"):
    return _find_files_by_pattern(animal_dir, pattern)


def find_preproc_files(animal_dir, pattern="*_preproc*.h5"):
    return _find_files_by_pattern(animal_dir, pattern)


def find_landmark_files(animal_dir, pattern="*_landmark_preferences.h5"):
    return _find_files_by_pattern(animal_dir, pattern)


def find_files_from_tracking(session_dirs, pattern):
    """
    Preferred alternative to find_*_files() when session_dirs is available.

    Searches each session's exact TSeries directory rather than scanning
    the whole animal folder.  Handles animals where some sessions were
    excluded from tracking (session_dirs only contains the tracked days).

    Parameters
    ----------
    session_dirs : dict {day_label: tseries_dir_path}
        From load_tracking() — only includes days that were actually tracked.
    pattern : str
        Glob pattern, e.g. '*_smi_results.h5' or '*_preproc*.h5'.

    Returns
    -------
    file_map : dict {day_label: filepath}
    """
    file_map = {}
    for day, tseries_dir in session_dirs.items():
        # Search directly in TSeries dir first, then recursively
        matches = glob(os.path.join(tseries_dir, pattern))
        if not matches:
            matches = glob(os.path.join(tseries_dir, "**", pattern), recursive=True)
        if matches:
            file_map[day] = sorted(matches)[0]
        else:
            print(f"  WARNING: no '{pattern}' found for {day} in {tseries_dir}")
    return file_map


def report_found_files(label, file_map, day_labels):
    found = [d for d in day_labels if d in file_map]
    missing = [d for d in day_labels if d not in file_map]
    print(f"  {label}: {len(found)}/{len(day_labels)} sessions found"
          + (f" | MISSING: {missing}" if missing else ""))


# ============================================================
# Per-session loaders
# ============================================================

def load_smi_session(smi_file):
    """
    Returns
    -------
    smi_all : (n_cells,) float  — NaN for invalid cells
    valid_mask : (n_cells,) bool
    Rp  : (n_cells,) float
    Rn  : (n_cells,) float
    preferred_positions : (n_cells,) float
    layer_cells : dict {layer_name: array of cell indices}
    """
    with h5py.File(smi_file, 'r') as f:
        g = f['global_smi']
        smi_raw = g['SMI_all_cells'][:].astype(float)
        valid = g['valid_cells_mask'][:].astype(bool)
        Rp = g['Rp'][:].astype(float)
        Rn = g['Rn'][:].astype(float)
        pref_pos = g['preferred_positions'][:].astype(float)

        smi_all = smi_raw.copy()
        smi_all[~valid] = np.nan
        Rp[~valid] = np.nan
        Rn[~valid] = np.nan
        pref_pos[~valid] = np.nan

        layer_cells = {}
        if 'layer_smi' in f:
            for key in f['layer_smi'].keys():
                grp = f['layer_smi'][key]
                name = grp.attrs.get('original_name', key.replace('_', '/'))
                name = name if isinstance(name, str) else name.decode()
                layer_cells[name] = grp['cell_indices'][:]

    return smi_all, valid, Rp, Rn, pref_pos, layer_cells


def load_preproc_session(preproc_file):
    """
    Returns
    -------
    spatial_activity     : (n_cells, n_trials, n_bins)
    norm_spatial_activity: (n_cells, n_trials, n_bins)
    bin_centers          : (n_bins,)
    reliable_cells       : (n_cells,) bool
    avg_cc               : (n_cells,) float
    cohen_d              : (n_cells,) float
    med_coords           : (n_cells, 2)
    """
    with h5py.File(preproc_file, 'r') as f:
        sa   = f['spatial_activity'][:]
        nsa  = f['norm_spatial_activity'][:]
        bc   = f['bin_centers'][:]
        rel  = f['combined_reliable'][:].astype(bool)
        cc   = f['avg_cc'][:].astype(float)   if 'avg_cc'   in f else np.full(sa.shape[0], np.nan)
        cd   = f['cohen_d'][:].astype(float)  if 'cohen_d'  in f else np.full(sa.shape[0], np.nan)
        med  = f['med_coords'][:].astype(float) if 'med_coords' in f else np.zeros((sa.shape[0], 2))
    return sa, nsa, bc, rel, cc, cd, med


# ============================================================
# Layer assignment for tracked cells
# ============================================================

def assign_layers_from_smi(tracked_matrix, day_labels, smi_files, reference_day):
    """
    Assign each tracked cell to a layer using the reference session's
    layer_smi/*/cell_indices groups.

    Returns
    -------
    cell_layers : dict {layer_name: np.ndarray of row-indices into tracked_matrix}
    """
    if reference_day not in day_labels:
        raise ValueError(f"Reference day '{reference_day}' not in {day_labels}")
    if reference_day not in smi_files:
        raise ValueError(f"No SMI file for reference day '{reference_day}'")

    ref_col = day_labels.index(reference_day)
    ref_roi = tracked_matrix[:, ref_col]

    _, _, _, _, _, layer_cells_ref = load_smi_session(smi_files[reference_day])

    cell_layers = {}
    for layer_name, layer_indices in layer_cells_ref.items():
        layer_set = set(layer_indices.tolist())
        mask = np.array([roi in layer_set for roi in ref_roi])
        cell_layers[layer_name] = np.where(mask)[0]

    return cell_layers


# ============================================================
# Generic matrix builder
# ============================================================

def build_matrix(tracked_matrix, day_labels, file_map, extractor_fn,
                 fill=np.nan):
    """
    Build an (n_tracked, n_sessions) matrix using extractor_fn.

    extractor_fn(session_file, roi_indices) -> np.ndarray shape (n_tracked,)
        Called once per session; roi_indices is tracked_matrix[:, col].
        Should return fill value (NaN) for absent/invalid cells.
    """
    n_cells, n_sessions = tracked_matrix.shape
    matrix = np.full((n_cells, n_sessions), fill)

    for col, day in enumerate(day_labels):
        if day not in file_map:
            continue
        roi_indices = tracked_matrix[:, col]
        try:
            matrix[:, col] = extractor_fn(file_map[day], roi_indices)
        except Exception as e:
            print(f"  WARNING: failed to extract {day}: {e}")

    return matrix


# ============================================================
# Numeric day list
# ============================================================

def parse_day_numbers(day_labels):
    """['Day1', 'Day3', ...] -> [1, 3, ...]"""
    nums = []
    for d in day_labels:
        m = re.search(r'(\d+)', d)
        nums.append(int(m.group(1)) if m else len(nums) + 1)
    return nums


# ============================================================
# Shared plot helpers
# ============================================================

LAYER_ORDER  = ['L2/3', 'L4', 'L5', 'L6']
LAYER_COLORS = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}


def layer_mean_sem(matrix, cell_layers, layer_order=None):
    """
    Returns
    -------
    stats : dict {layer: {'mean', 'sem', 'n', 'all'}}
    """
    order = layer_order or LAYER_ORDER
    out = {}
    for layer in order:
        if layer not in cell_layers or len(cell_layers[layer]) == 0:
            continue
        rows = matrix[cell_layers[layer], :]
        n    = np.sum(~np.isnan(rows), axis=0)
        mean = np.nanmean(rows, axis=0)
        sem  = np.nanstd(rows, axis=0) / np.sqrt(n.clip(1))
        out[layer] = {'mean': mean, 'sem': sem, 'n': n, 'all': rows}
    return out


def animal_id_from_path(path):
    m = re.search(r'(JSY\d+)', path)
    return m.group(1) if m else ""
