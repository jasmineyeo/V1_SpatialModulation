"""
PCA_DataAggregation.py
Aggregates spatial response data across sessions for a single animal.
Creates a structured HDF5 file ready for PCA analysis.

Approach B: Exclude onset/reward responders, keep all other reliable cells
(including those with peaks between landmark windows)

JSY, 12/2025

Output HDF5 structure:


├── metadata/           # Animal ID, parameters, bin centers
├── cells/              # Per-cell labels
│   ├── session_labels     # Which session each cell came from
│   ├── layer_labels       # Cortical layer (L2/3, L4, etc.)
│   ├── preferred_landmark # Which landmark (0-3) or -1
│   ├── peak_positions     # Peak location in cm
│   └── cell_depths        # Imaging depth
├── features/           # The actual data for PCA
│   ├── spatial_profiles        # Mean profiles (cells × 115 bins)
│   └── spatial_profiles_zscore # Z-scored profiles
└── sessions/           # Info about each session
Summary: What Goes Into PCA
The rows of your PCA input matrix are individual neurons (pooled across all sessions). The columns are 115 spatial bins from 10-125 cm.

Each neuron included has:

Reliable spatial tuning (from preprocessing)
A peak NOT at trial onset or reward location
Z-scored activity profile (shape-normalized)
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import numpy as np
import h5py
from scipy.interpolate import interp1d
from datetime import datetime
from scipy.ndimage import gaussian_filter1d

from helper import files, TwoP
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer


# ============================================================================
# CONFIGURATION
# ============================================================================

# Animal to analyze
ANIMAL_ID = "JSY054"

# Base directory containing all session folders for this animal
BASE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"
# BASE_DIR = r"F:\2P\unprocessed\JSY040"

# Output directory for PCA data file
OUTPUT_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA"

# Landmark configuration (must match your landmark analysis)
# LANDMARK_POSITIONS = [25, 55, 85, 115]  # cm
LANDMARK_POSITIONS = [36, 64, 92, 120]  # cm

LANDMARK_WINDOWS_CONFIG = [
    # {'before': 10, 'after': 10},  # L1 at 25cm: [10, 35]
    # {'before': 20, 'after': 10},  # L2 at 55cm: [35, 65]
    # {'before': 20, 'after': 10},  # L3 at 85cm: [65, 95]
    # {'before': 20, 'after': 10},  # L4 at 115cm: [95, 125]
    {'before': 18, 'after': 0},  # L1 at 25cm: [10, 35]
    {'before': 18, 'after': 0},  # L2 at 55cm: [35, 65]
    {'before': 18, 'after': 0},  # L3 at 85cm: [65, 95]
    {'before': 18, 'after': 0},  # L4 at 115cm: [95, 125]
]

# Spatial trimming for PCA features (cm)
TRIM_START_CM = 10   # Start of analysis window (matches L1 window start)
TRIM_END_CM = 120    # End of analysis window (matches L4 window end)

# Filtering parameters
EXCLUDE_FIRST_BINS = 10  # Bins to exclude for onset filtering
EXCLUDE_LAST_BINS = 10   # Bins to exclude for reward filtering
SMOOTHING_SIGMA = 1.0   # Gaussian smoothing for profiles

# Cell selection mode:
#   'combined_reliable'    — lap-to-lap reliable cells from preproc (+ optional SMI_THRESHOLD)
#   'reliable_valid_cells' — cells that passed SMI geometry criteria (per-layer union from smi_results.h5)
#                            matches the population used in folder 3 landmark preference analyses
CELL_SELECTION = 'reliable_valid_cells'

# SMI threshold: only used when CELL_SELECTION = 'combined_reliable'.
# Cells with SMI <= this value are excluded. Set to None to disable.
SMI_THRESHOLD = 0.2

# Landmark alignment parameters
ALIGN_PROFILES = False   # Align each cell to its preferred landmark before saving

# Alignment method:
#   'type_aware'           — template-correlation alignment per cell type:
#                            single-peak ±15cm, multi-peak ±n_bins//2, onset+landmark
#                            post-onset only. (previous default)
#   'preferred_landmark'   — deterministic: shift each cell so its preferred landmark
#                            peak moves to CANONICAL_LANDMARK_CM.
#                            Adaptation-like (≥4 peaks) and onset+landmark cells are
#                            NOT shifted. Safer against landmark-jumping.
ALIGN_METHOD = 'preferred_landmark'

# Canonical reference position for preferred-landmark alignment (cm).
# Cells are shifted so their preferred landmark peak sits here.
# Recommended: mean of LANDMARK_POSITIONS so the corridor centre is the reference.
CANONICAL_LANDMARK_CM = 70.0   # mean of [25, 55, 85, 115]

# Peak detection thresholds used by preferred-landmark alignment
ADAPTATION_PEAK_THRESHOLD  = 4    # ≥ this many peaks → adaptation-like → no shift
PEAK_PROMINENCE_PL         = 0.3  # prominence for preferred-landmark peak detection
PEAK_MIN_DIST_CM_PL        = 20.0 # minimum separation between peaks (cm)
MAX_DIST_TO_LANDMARK_CM    = 15.0 # max allowed distance from peak to nearest landmark;
                                   # cells further than this are not shifted (ambiguous)

# Whether to re-z-score profiles after alignment.
#   False — magnitude preserved (matches MATLAB TreadmillResponseSorter)
#   True  — shape-only PCA
ZSCORE_AFTER_ALIGNMENT = False

# Gaussian width for each landmark peak in the 4-Gaussian template (cm)
TEMPLATE_SIGMA_CM = 8.0

# Onset-cell classification parameters (type 2 vs type 3)
POST_ONSET_START_CM = 35.0   # cm — start of post-onset window (just past L1)
ONSET_R_THRESHOLD   = 0.3    # min Pearson r (post-onset vs template) to be type 3
ONSET_MAX_SHIFT_CM  = 15.0   # ±cm search window when classifying / shifting type 3

# Cell type labels
CELL_TYPE_LANDMARK       = 'landmark'        # types 1 & 4: peak outside onset zone
CELL_TYPE_ONSET_ONLY     = 'onset_only'      # type 2: onset response, no landmark structure
CELL_TYPE_ONSET_LANDMARK = 'onset_landmark'  # type 3: onset + post-onset landmark structure


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def find_session_folders(base_dir, animal_id):
    """
    Find all session folders for an animal, sorted by day number.
    
    Returns list of tuples: (session_id, tseries_path)
    """
    sessions = []
    
    # Pattern: YYMMDD_JSY_JSYXXX_*_DayN
    pattern = os.path.join(base_dir, f"*{animal_id}*Day*")
    session_dirs = glob.glob(pattern)
    
    for session_dir in session_dirs:
        # Extract day number
        folder_name = os.path.basename(session_dir)
        day_match = re.search(r'Day(\d+)', folder_name)
        
        if day_match:
            day_num = int(day_match.group(1))
            session_id = f"Day{day_num}"
            
            # Find TSeries folder(s) within this session
            tseries_folders = glob.glob(os.path.join(session_dir, "TSeries-*"))
            
            for tseries_path in tseries_folders:
                # Check if preprocessed file exists
                preproc_files = glob.glob(os.path.join(tseries_path, "*preproc*.h5"))
                if preproc_files:
                    sessions.append((day_num, session_id, tseries_path))
    
    print(f"Found {len(sessions)} sessions for animal {animal_id}:")
    for day_num, session_id, tseries_path in sessions:
        print(f"  {session_id}: {tseries_path}")
        
    # Sort by day number
    sessions.sort(key=lambda x: x[0])
    
    return [(s[1], s[2]) for s in sessions]  # Return (session_id, path)


def identify_valid_cells_for_pca(normalized_spatial_activity, bin_centers,
                                  reliable_cells, exclude_first_bins=5,
                                  exclude_last_bins=5, smoothing_sigma=1.0,
                                  landmark_positions=None,
                                  post_onset_start_cm=POST_ONSET_START_CM,
                                  onset_r_threshold=ONSET_R_THRESHOLD,
                                  onset_max_shift_cm=ONSET_MAX_SHIFT_CM):
    """
    Identify cells valid for PCA analysis.

    Cell types
    ----------
    CELL_TYPE_LANDMARK        (types 1 & 4) — reliable, peak outside onset/reward zones
    CELL_TYPE_ONSET_ONLY      (type 2)      — peak in onset zone, no post-onset landmark
                                              structure; excluded from PCA
    CELL_TYPE_ONSET_LANDMARK  (type 3)      — peak in onset zone but with significant
                                              post-onset landmark structure; included

    If landmark_positions is None the function falls back to the original behaviour
    (all onset-zone cells are excluded).

    Returns
    -------
    valid_for_pca  : boolean array (n_cells,)
    peak_positions : array (n_cells,)
    mean_profiles  : array (n_cells, n_bins)
    rejection_info : dict — includes 'cell_type_labels' key
    """
    n_cells = normalized_spatial_activity.shape[0]

    # Compute mean profiles
    mean_profiles = np.mean(normalized_spatial_activity, axis=1)

    # Apply smoothing
    if smoothing_sigma > 0:
        for cell in range(n_cells):
            mean_profiles[cell] = gaussian_filter1d(mean_profiles[cell], sigma=smoothing_sigma)

    # Calculate thresholds
    min_pos     = np.min(bin_centers)
    max_pos     = np.max(bin_centers)
    bin_spacing = np.mean(np.diff(bin_centers))

    onset_threshold_cm = min_pos + (exclude_first_bins * bin_spacing)
    end_threshold_cm   = max_pos - (exclude_last_bins  * bin_spacing)

    # Initialise
    valid_for_pca    = np.zeros(n_cells, dtype=bool)
    peak_positions   = np.zeros(n_cells)
    cell_type_labels = np.array([CELL_TYPE_LANDMARK] * n_cells, dtype='U20')

    rejected_onset       = []   # type 2 — onset-only, excluded
    onset_landmark_cells = []   # type 3 — onset+landmark, included
    rejected_reward      = []
    rejected_zero        = []
    rejected_not_reliable = []

    for cell in range(n_cells):
        # Must be reliable first
        if not reliable_cells[cell]:
            rejected_not_reliable.append(cell)
            cell_type_labels[cell] = 'not_reliable'
            continue

        profile         = mean_profiles[cell]
        global_peak_idx = np.argmax(profile)
        global_peak_pos = bin_centers[global_peak_idx]
        peak_positions[cell] = global_peak_pos

        # Check for zero activity
        if profile[global_peak_idx] == 0:
            rejected_zero.append(cell)
            cell_type_labels[cell] = 'zero'
            continue

        # Check reward zone
        if global_peak_pos > end_threshold_cm:
            rejected_reward.append(cell)
            cell_type_labels[cell] = 'reward'
            continue

        # Check onset zone
        if global_peak_pos < onset_threshold_cm:
            if landmark_positions is not None:
                ctype, _, _ = classify_onset_cell(
                    profile, bin_centers, landmark_positions,
                    post_onset_start_cm=post_onset_start_cm,
                    max_shift_cm=onset_max_shift_cm,
                    r_threshold=onset_r_threshold,
                )
                cell_type_labels[cell] = ctype
                if ctype == CELL_TYPE_ONSET_LANDMARK:
                    valid_for_pca[cell] = True
                    onset_landmark_cells.append(cell)
                else:
                    rejected_onset.append(cell)
            else:
                # Legacy behaviour — exclude all onset-zone cells
                rejected_onset.append(cell)
                cell_type_labels[cell] = CELL_TYPE_ONSET_ONLY
            continue

        # Cell passes all filters — landmark type (1 or 4)
        valid_for_pca[cell] = True
        cell_type_labels[cell] = CELL_TYPE_LANDMARK

    rejection_info = {
        'onset':          np.array(rejected_onset),
        'onset_landmark': np.array(onset_landmark_cells),
        'reward':         np.array(rejected_reward),
        'zero':           np.array(rejected_zero),
        'not_reliable':   np.array(rejected_not_reliable),
        'onset_threshold_cm': onset_threshold_cm,
        'end_threshold_cm':   end_threshold_cm,
        'n_valid':  np.sum(valid_for_pca),
        'n_total':  n_cells,
        'cell_type_labels': cell_type_labels,
    }

    return valid_for_pca, peak_positions, mean_profiles, rejection_info


def assign_landmark_preference(peak_positions, landmark_positions, landmark_windows_config):
    """
    Assign landmark preference based on peak position.
    
    Returns:
    --------
    preferred_landmark : array of int
        Index of preferred landmark (0-3) or -1 if peak is between windows
    """
    n_cells = len(peak_positions)
    preferred_landmark = np.full(n_cells, -1, dtype=int)
    
    # Build window boundaries
    windows = []
    for i, lm_pos in enumerate(landmark_positions):
        config = landmark_windows_config[i]
        lm_min = lm_pos - config['before']
        lm_max = lm_pos + config['after']
        windows.append((lm_min, lm_max))
    
    for cell in range(n_cells):
        peak = peak_positions[cell]
        
        for lm_idx, (lm_min, lm_max) in enumerate(windows):
            if lm_min <= peak <= lm_max:
                preferred_landmark[cell] = lm_idx
                break
    
    return preferred_landmark


def assign_layer_labels_and_depths(med_coords, layer_cells):
    """
    Convert layer_cells dict to per-cell layer labels AND extract depths.
    
    Parameters:
    -----------
    med_coords : list of tuples
        Median coordinates for each cell (x, y) where y is depth
    layer_cells : dict
        Dictionary mapping layer names to cell indices
    
    Returns:
    --------
    layer_labels : array of strings
        Layer label for each cell
    cell_depths : array of floats
        Y-coordinate (depth) for each cell
    """
    n_cells = len(med_coords)
    layer_labels = np.array(['Unknown'] * n_cells, dtype='U10')
    
    # Extract Y-coordinates (depth) from med_coords
    # med_coords is a list of tuples like [(x1, y1), (x2, y2), ...]
    cell_depths = np.array([coord[1] for coord in med_coords], dtype=float)
    
    for layer_name, cell_indices in layer_cells.items():
        for idx in cell_indices:
            if idx < n_cells:
                layer_labels[idx] = layer_name
    
    return layer_labels, cell_depths


def extract_trimmed_profiles(mean_profiles, bin_centers, trim_start_cm, trim_end_cm, 
                            target_n_bins=115):
    """
    Extract profiles within the specified range and interpolate to common grid.
    
    Parameters:
    -----------
    mean_profiles : array (n_cells, n_bins)
    bin_centers : array
    trim_start_cm : float
    trim_end_cm : float
    target_n_bins : int
        Target number of bins after interpolation (default: 115)
    
    Returns:
    --------
    trimmed_profiles : array (n_cells, target_n_bins)
        Profiles interpolated to common grid
    common_bin_centers : array (target_n_bins,)
        Common bin centers for all sessions
    """
    from scipy.interpolate import interp1d
    
    # Find indices for trimming
    start_idx = np.searchsorted(bin_centers, trim_start_cm)
    end_idx = np.searchsorted(bin_centers, trim_end_cm)
    
    # Extract the spatial range
    trimmed_profiles_raw = mean_profiles[:, start_idx:end_idx]
    trimmed_bin_centers_raw = bin_centers[start_idx:end_idx]
    
    # Create common spatial grid
    common_bin_centers = np.linspace(trim_start_cm, trim_end_cm, target_n_bins)
    
    # Interpolate each cell's profile onto the common grid
    n_cells = mean_profiles.shape[0]
    trimmed_profiles = np.zeros((n_cells, target_n_bins))
    
    for i in range(n_cells):
        # Create interpolation function
        f = interp1d(trimmed_bin_centers_raw, trimmed_profiles_raw[i, :], 
                    kind='linear', bounds_error=False, fill_value='extrapolate')
        
        # Interpolate to common grid
        trimmed_profiles[i, :] = f(common_bin_centers)
    
    print(f"  Original: {len(trimmed_bin_centers_raw)} bins → Interpolated to {target_n_bins} bins")
    
    return trimmed_profiles, common_bin_centers

def zscore_profiles(profiles):
    """
    Z-score normalize each cell's profile (subtract mean, divide by std).
    This normalizes by shape, not magnitude.
    """
    zscored = np.zeros_like(profiles)
    
    for i in range(profiles.shape[0]):
        profile = profiles[i]
        mean_val = np.mean(profile)
        std_val = np.std(profile)
        
        if std_val > 0:
            zscored[i] = (profile - mean_val) / std_val
        else:
            zscored[i] = profile - mean_val
    
    return zscored


# ============================================================================
# LANDMARK ALIGNMENT FUNCTIONS
# ============================================================================

def _shift_profile_noncircular(profile, shift_bins):
    """Shift profile by shift_bins with zero-padding (no wrap-around)."""
    shifted = np.zeros_like(profile)
    if shift_bins > 0:
        shifted[shift_bins:] = profile[:-shift_bins]
    elif shift_bins < 0:
        shifted[:shift_bins] = profile[-shift_bins:]
    else:
        shifted = profile.copy()
    return shifted


def create_4gaussian_template(bin_centers, landmark_positions, sigma_cm=8.0):
    """Sum of 4 Gaussians centred at each landmark position."""
    template = np.zeros(len(bin_centers))
    for lm_pos in landmark_positions:
        template += np.exp(-0.5 * ((bin_centers - lm_pos) / sigma_cm) ** 2)
    return template


def align_profiles_template_correlation(profiles, bin_centers, landmark_positions,
                                         sigma_cm=8.0, zscore_after=False):
    """
    Align each cell's profile by shifting to maximise Pearson correlation
    with a 4-Gaussian template (one peak per landmark).
    Non-circular, zero-padded shifts only.

    Parameters
    ----------
    profiles           : (n_cells, n_bins) array
    bin_centers        : (n_bins,) array in cm
    landmark_positions : list of float
    sigma_cm           : float — Gaussian width for template peaks
    zscore_after       : bool  — if False, magnitude preserved (MATLAB-like)

    Returns
    -------
    aligned          : (n_cells, n_bins) array
    optimal_shifts   : (n_cells,) int array
    max_correlations : (n_cells,) float array
    """
    from scipy.stats import pearsonr

    template = create_4gaussian_template(bin_centers, landmark_positions, sigma_cm)
    n_cells, n_bins = profiles.shape
    bin_spacing = float(np.mean(np.diff(bin_centers)))

    print("\n  Aligning profiles via template-correlation (non-circular)...")
    print(f"    Template sigma: {sigma_cm} cm  |  zscore_after: {zscore_after}")

    aligned          = profiles.copy()
    optimal_shifts   = np.zeros(n_cells, dtype=int)
    max_correlations = np.zeros(n_cells)

    for cell_idx in range(n_cells):
        profile    = profiles[cell_idx]
        best_corr  = -np.inf
        best_shift = 0
        for shift in range(-n_bins // 2, n_bins // 2):
            candidate = _shift_profile_noncircular(profile, shift)
            corr, _   = pearsonr(candidate, template)
            if corr > best_corr:
                best_corr, best_shift = corr, shift
        aligned[cell_idx]          = _shift_profile_noncircular(profile, best_shift)
        optimal_shifts[cell_idx]   = best_shift
        max_correlations[cell_idx] = best_corr

    if zscore_after:
        aligned = zscore_profiles(aligned)

    shifts_cm = optimal_shifts * bin_spacing
    shifted   = optimal_shifts != 0
    print(f"    Cells shifted: {np.sum(shifted)}")
    if np.any(shifted):
        print(f"    Shift stats: mean={np.mean(shifts_cm[shifted]):.1f} cm, "
              f"std={np.std(shifts_cm[shifted]):.1f} cm")
    print(f"    Max-corr stats: mean={np.mean(max_correlations):.3f}, "
          f"min={np.min(max_correlations):.3f}")

    return aligned, optimal_shifts, max_correlations


def count_peaks(profile, min_prominence=0.5, min_distance_bins=5):
    """
    Count the number of local maxima in a profile.

    Parameters
    ----------
    profile           : 1D array (smoothed z-scored spatial profile)
    min_prominence    : float — peak must exceed neighbouring troughs by this much
    min_distance_bins : int   — minimum separation between peaks (bins)

    Returns
    -------
    n_peaks : int
    """
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(profile,
                              prominence=min_prominence,
                              distance=min_distance_bins)
    return len(peaks)


def classify_onset_cell(profile, bin_centers, landmark_positions,
                        post_onset_start_cm=POST_ONSET_START_CM,
                        max_shift_cm=ONSET_MAX_SHIFT_CM,
                        r_threshold=ONSET_R_THRESHOLD):
    """
    Classify an onset-zone cell as Type 2 (onset-only) or Type 3 (onset+landmark).

    Searches ±max_shift_cm shifts over the post-onset portion of the profile and
    records the maximum Pearson r against the corresponding 4-Gaussian template.
    Correlation is computed only on the valid (non-zero-padded) region at each
    shift to avoid penalising cells whose landmark peaks are offset.

    Parameters
    ----------
    profile             : (n_bins,) smoothed mean spatial profile
    bin_centers         : (n_bins,)
    landmark_positions  : list of float
    post_onset_start_cm : float — start of post-onset window (cm)
    max_shift_cm        : float — ±search window (cm)
    r_threshold         : float — minimum max-r to classify as onset+landmark

    Returns
    -------
    cell_type  : str   — CELL_TYPE_ONSET_ONLY or CELL_TYPE_ONSET_LANDMARK
    best_shift : int   — shift (bins) that gave max r (positive = rightward)
    max_r      : float — maximum Pearson r achieved
    """
    from scipy.stats import pearsonr

    bin_spacing    = float(np.mean(np.diff(bin_centers)))
    max_shift_bins = int(np.round(max_shift_cm / bin_spacing))

    template_full = create_4gaussian_template(bin_centers, landmark_positions,
                                              sigma_cm=TEMPLATE_SIGMA_CM)
    post_mask     = bin_centers >= post_onset_start_cm
    post_profile  = profile[post_mask]
    post_template = template_full[post_mask]
    n_post        = len(post_profile)

    best_r     = -np.inf
    best_shift = 0

    for shift in range(-max_shift_bins, max_shift_bins + 1):
        shifted = _shift_profile_noncircular(post_profile, shift)

        # Valid (non-zero-padded) region only
        if shift >= 0:
            valid = slice(shift, None)
        else:
            valid = slice(None, n_post + shift)

        vp = shifted[valid]
        vt = post_template[valid]

        if len(vp) < 10 or np.std(vp) == 0:
            continue

        r, _ = pearsonr(vp, vt)
        if r > best_r:
            best_r     = r
            best_shift = shift

    cell_type = (CELL_TYPE_ONSET_LANDMARK if best_r >= r_threshold
                 else CELL_TYPE_ONSET_ONLY)
    return cell_type, best_shift, float(best_r)


def align_profiles_type_aware(profiles, bin_centers, landmark_positions,
                               cell_types,
                               post_onset_start_cm=POST_ONSET_START_CM,
                               max_shift_cm=ONSET_MAX_SHIFT_CM,
                               sigma_cm=8.0, zscore_after=False):
    """
    Align profiles according to cell type:

      CELL_TYPE_LANDMARK       — full-profile template correlation (existing method,
                                 ±n_bins//2 search)
      CELL_TYPE_ONSET_LANDMARK — post-onset portion template correlation (±max_shift_cm,
                                 valid-region-only Pearson r)
      anything else            — no shift

    Parameters
    ----------
    profiles            : (n_cells, n_bins)
    bin_centers         : (n_bins,)
    landmark_positions  : list of float
    cell_types          : (n_cells,) str array — from identify_valid_cells_for_pca
    post_onset_start_cm : float
    max_shift_cm        : float — ±search window for onset+landmark cells (cm)
    sigma_cm            : float — Gaussian width for template
    zscore_after        : bool

    Returns
    -------
    aligned          : (n_cells, n_bins)
    optimal_shifts   : (n_cells,) int
    max_correlations : (n_cells,) float
    """
    from scipy.stats import pearsonr

    n_cells, n_bins = profiles.shape
    bin_spacing     = float(np.mean(np.diff(bin_centers)))
    max_shift_bins  = int(np.round(max_shift_cm / bin_spacing))

    template_full = create_4gaussian_template(bin_centers, landmark_positions, sigma_cm)
    post_mask     = bin_centers >= post_onset_start_cm
    post_template = template_full[post_mask]

    aligned          = profiles.copy()
    optimal_shifts   = np.zeros(n_cells, dtype=int)
    max_correlations = np.zeros(n_cells)

    n_landmark = n_onset_landmark = n_no_shift = 0

    # Peak detection parameters for single vs multi-peak classification
    MULTI_PEAK_THRESHOLD = 3      # ≥ this many peaks → multi-peak → ±n_bins//2
    PEAK_PROMINENCE      = 0.5    # z-score units
    PEAK_MIN_DIST_BINS   = 5      # minimum separation between peaks

    n_single_peak = n_multi_peak = 0

    print("\n  Type-aware alignment...")
    print(f"    Landmark single-peak : ±{max_shift_cm} cm (jitter correction only)")
    print(f"    Landmark multi-peak  : ±{n_bins // 2} bins (full search)")
    print(f"    Onset+landmark       : post-onset (≥{post_onset_start_cm} cm), ±{max_shift_cm} cm")

    for idx in range(n_cells):
        profile = profiles[idx]
        ctype   = cell_types[idx]

        if ctype == CELL_TYPE_LANDMARK:
            n_peaks = count_peaks(profile,
                                  min_prominence=PEAK_PROMINENCE,
                                  min_distance_bins=PEAK_MIN_DIST_BINS)
            if n_peaks >= MULTI_PEAK_THRESHOLD:
                # Multi-peak (adaptation-like): full range
                search_bins = range(-n_bins // 2, n_bins // 2)
                n_multi_peak += 1
            else:
                # Single-peak: constrain to ±15 cm to prevent landmark jumping
                search_bins = range(-max_shift_bins, max_shift_bins + 1)
                n_single_peak += 1

            best_corr, best_shift = -np.inf, 0
            for shift in search_bins:
                candidate = _shift_profile_noncircular(profile, shift)
                r, _      = pearsonr(candidate, template_full)
                if r > best_corr:
                    best_corr, best_shift = r, shift
            aligned[idx]          = _shift_profile_noncircular(profile, best_shift)
            optimal_shifts[idx]   = best_shift
            max_correlations[idx] = best_corr
            n_landmark += 1

        elif ctype == CELL_TYPE_ONSET_LANDMARK:
            # Post-onset portion, ±15 cm, valid-region-only Pearson r
            post_profile = profile[post_mask]
            n_post       = len(post_profile)
            best_corr, best_shift = -np.inf, 0

            for shift in range(-max_shift_bins, max_shift_bins + 1):
                shifted = _shift_profile_noncircular(post_profile, shift)
                if shift >= 0:
                    valid = slice(shift, None)
                else:
                    valid = slice(None, n_post + shift)
                vp = shifted[valid]
                vt = post_template[valid]
                if len(vp) < 10 or np.std(vp) == 0:
                    continue
                r, _ = pearsonr(vp, vt)
                if r > best_corr:
                    best_corr, best_shift = r, shift

            aligned[idx]          = _shift_profile_noncircular(profile, best_shift)
            optimal_shifts[idx]   = best_shift
            max_correlations[idx] = best_corr
            n_onset_landmark += 1

        else:
            aligned[idx]          = profile.copy()
            optimal_shifts[idx]   = 0
            max_correlations[idx] = 0.0
            n_no_shift += 1

    if zscore_after:
        aligned = zscore_profiles(aligned)

    shifts_cm = optimal_shifts * bin_spacing
    print("\n  Alignment summary:")
    print(f"    Landmark total:                    {n_landmark} cells")
    print(f"      Single-peak (±{max_shift_cm} cm):          {n_single_peak} cells")
    print(f"      Multi-peak  (±{n_bins // 2} bins):         {n_multi_peak} cells")
    print(f"    Onset+landmark (post-onset shift): {n_onset_landmark} cells")
    print(f"    Not shifted:                       {n_no_shift} cells")
    if n_landmark + n_onset_landmark > 0:
        shifted_mask = optimal_shifts != 0
        print(f"    Mean shift (shifted cells): "
              f"{np.mean(shifts_cm[shifted_mask]):.1f} cm, "
              f"std={np.std(shifts_cm[shifted_mask]):.1f} cm")

    return aligned, optimal_shifts, max_correlations


def align_profiles_preferred_landmark(profiles, bin_centers, landmark_positions,
                                       cell_types,
                                       canonical_cm=CANONICAL_LANDMARK_CM,
                                       adaptation_peak_threshold=ADAPTATION_PEAK_THRESHOLD,
                                       peak_prominence=PEAK_PROMINENCE_PL,
                                       peak_min_dist_cm=PEAK_MIN_DIST_CM_PL,
                                       max_dist_to_landmark_cm=MAX_DIST_TO_LANDMARK_CM,
                                       zscore_after=False):
    """
    Deterministic preferred-landmark alignment.

    For each cell:
      1. Detect peaks in the z-scored profile.
      2. If n_peaks >= adaptation_peak_threshold (adaptation-like) OR
         cell_type == CELL_TYPE_ONSET_LANDMARK → no shift.
      3. Otherwise, find which detected peak is closest to any landmark,
         identify that landmark, and shift the profile so that landmark
         moves to canonical_cm.

    This avoids the template-correlation problem of jumping across landmarks
    because the shift is derived directly from the cell's actual peak position
    relative to its nearest landmark, not from a correlation score.

    Parameters
    ----------
    profiles                  : (n_cells, n_bins)
    bin_centers               : (n_bins,)
    landmark_positions        : list of float
    cell_types                : (n_cells,) str — from identify_valid_cells_for_pca
    canonical_cm              : float — target position for preferred landmark
    adaptation_peak_threshold : int   — ≥ this many peaks → no shift
    peak_prominence           : float — min prominence for peak detection
    peak_min_dist_cm          : float — min distance between peaks (cm)
    zscore_after              : bool

    Returns
    -------
    aligned          : (n_cells, n_bins)
    optimal_shifts   : (n_cells,) int   (bins; positive = rightward)
    preferred_lm_idx : (n_cells,) int   (0-3, or -1 if no shift)
    """
    from scipy.signal import find_peaks

    n_cells, _ = profiles.shape
    bin_spacing = float(np.mean(np.diff(bin_centers)))
    min_dist_bins   = max(1, int(np.round(peak_min_dist_cm / bin_spacing)))

    aligned          = profiles.copy()
    optimal_shifts   = np.zeros(n_cells, dtype=int)
    preferred_lm_idx = np.full(n_cells, -1, dtype=int)

    n_shifted = n_no_shift_adapt = n_no_shift_onset = n_no_shift_nopeak = 0

    print("\n  Preferred-landmark alignment...")
    print(f"    Canonical reference: {canonical_cm} cm")
    print(f"    Adaptation threshold: ≥{adaptation_peak_threshold} peaks → no shift")

    for idx in range(n_cells):
        profile = profiles[idx]
        ctype   = cell_types[idx]

        # Onset+landmark cells: globally tuned, no shift
        if ctype == CELL_TYPE_ONSET_LANDMARK:
            n_no_shift_onset += 1
            continue

        # Detect peaks
        peaks, _ = find_peaks(profile,
                              prominence=peak_prominence,
                              distance=min_dist_bins)

        # Adaptation-like: too many peaks, no shift
        if len(peaks) >= adaptation_peak_threshold:
            n_no_shift_adapt += 1
            continue

        # No detectable peak: no shift
        if len(peaks) == 0:
            n_no_shift_nopeak += 1
            continue

        # Find the peak closest to any landmark
        peak_positions_cm = bin_centers[peaks]
        lm_arr = np.array(landmark_positions)

        # For each detected peak, find distance to nearest landmark
        best_lm_idx = None
        best_dist   = np.inf

        for peak_cm in peak_positions_cm:
            dists     = np.abs(lm_arr - peak_cm)
            near_lm   = int(np.argmin(dists))
            near_dist = dists[near_lm]
            if near_dist < best_dist:
                best_dist   = near_dist
                best_lm_idx = near_lm

        # If the closest peak is too far from any landmark, cell is ambiguous — no shift
        if best_dist > max_dist_to_landmark_cm:
            n_no_shift_nopeak += 1
            continue

        # Shift so preferred landmark moves to canonical_cm
        preferred_lm_cm = lm_arr[best_lm_idx]
        shift_cm        = canonical_cm - preferred_lm_cm
        shift_bins      = int(np.round(shift_cm / bin_spacing))

        aligned[idx]          = _shift_profile_noncircular(profile, shift_bins)
        optimal_shifts[idx]   = shift_bins
        preferred_lm_idx[idx] = best_lm_idx
        n_shifted += 1

    if zscore_after:
        aligned = zscore_profiles(aligned)

    shifts_cm = optimal_shifts * bin_spacing
    print(f"    Shifted:           {n_shifted} cells")
    print(f"    Not shifted (adaptation-like ≥{adaptation_peak_threshold} peaks): {n_no_shift_adapt}")
    print(f"    Not shifted (onset+landmark): {n_no_shift_onset}")
    print(f"    Not shifted (no detectable peak): {n_no_shift_nopeak}")
    shifted_mask = optimal_shifts != 0
    if np.any(shifted_mask):
        print(f"    Mean shift: {np.mean(shifts_cm[shifted_mask]):.1f} cm  "
              f"std={np.std(shifts_cm[shifted_mask]):.1f} cm")
    for li, lm in enumerate(landmark_positions):
        n_lm = int(np.sum(preferred_lm_idx == li))
        print(f"    Preferred L{li+1} ({lm} cm): {n_lm} cells")

    return aligned, optimal_shifts, preferred_lm_idx


# ============================================================================
# MAIN AGGREGATION FUNCTION
# ============================================================================

def aggregate_pca_data(animal_id, base_dir, output_dir,
                       landmark_positions, landmark_windows_config,
                       trim_start_cm, trim_end_cm,
                       exclude_first_bins, exclude_last_bins,
                       smoothing_sigma,
                       align_profiles=True,
                       align_method='template_correlation',
                       zscore_after_alignment=False,
                       template_sigma_cm=8.0):
    """
    Main function to aggregate data across sessions for PCA analysis.
    """
    
    print("=" * 80)
    print(f"PCA DATA AGGREGATION: {animal_id}")
    print("=" * 80)
    print(f"Base directory: {base_dir}")
    print(f"Analysis window: {trim_start_cm} - {trim_end_cm} cm")
    print(f"Onset exclusion: first {exclude_first_bins} bins")
    print(f"Reward exclusion: last {exclude_last_bins} bins")
    print("=" * 80)
    
    # Find all sessions
    sessions = find_session_folders(base_dir, animal_id)
    
    if len(sessions) == 0:
        raise ValueError(f"No sessions found for {animal_id} in {base_dir}")
    
    print(f"\nFound {len(sessions)} sessions:")
    for session_id, path in sessions:
        print(f"  {session_id}: {os.path.basename(path)}")
    
    # Storage for aggregated data
    all_profiles = []
    all_session_labels = []
    all_layer_labels = []
    all_landmark_prefs = []
    all_peak_positions = []
    all_original_indices = []
    all_cell_depths = []  # ← ADD THIS LINE
    all_cell_types = []   # type 1/4 → 'landmark', type 3 → 'onset_landmark'

    session_info = {}
    trimmed_bin_centers = None
    
    # Process each session
    for session_idx, (session_id, tseries_path) in enumerate(sessions):
        print(f"\n{'-'*60}")
        print(f"Processing {session_id} ({session_idx+1}/{len(sessions)})")
        print(f"Path: {tseries_path}")
        print(f"{'-'*60}")
        
        try:
            # Load preprocessed data
            preproc_files = glob.glob(os.path.join(tseries_path, "*preproc*.h5"))
            if not preproc_files:
                print(f"  WARNING: No preproc file found, skipping")
                continue
            
            preproc_data = files.read_h5(preproc_files[0])
            print(f"  Loaded: {os.path.basename(preproc_files[0])}")
            
            normalized_spatial_activity = preproc_data['norm_spatial_activity']
            bin_centers = preproc_data['bin_centers']

            n_cells, n_trials, n_bins = normalized_spatial_activity.shape
            print(f"  Data shape: {n_cells} cells, {n_trials} trials, {n_bins} bins")

            smi_files = glob.glob(os.path.join(tseries_path, '*_smi_results.h5'))

            if CELL_SELECTION == 'reliable_valid_cells':
                # Union of reliable_valid_cells across all layers from SMI h5
                if not smi_files:
                    print("  WARNING: No SMI results file found, skipping session")
                    continue
                reliable_cells = np.zeros(n_cells, dtype=bool)
                with h5py.File(smi_files[0], 'r') as sf:
                    for lk in sf['layer_smi'].keys():
                        rv = sf['layer_smi'][lk]['reliable_valid_cells'][:].astype(int)
                        reliable_cells[rv] = True
                print(f"  reliable_valid_cells (union across layers): {np.sum(reliable_cells)}")

            else:  # 'combined_reliable'
                reliable_cells = preproc_data['combined_reliable'].astype(bool)
                print(f"  combined_reliable: {np.sum(reliable_cells)}")
                if SMI_THRESHOLD is not None:
                    if smi_files:
                        with h5py.File(smi_files[0], 'r') as sf:
                            smi_all = sf['global_smi/SMI_all_cells'][:]
                        reliable_cells = reliable_cells & (smi_all > SMI_THRESHOLD)
                        print(f"  After SMI>{SMI_THRESHOLD} filter: {np.sum(reliable_cells)} cells")
                    else:
                        print("  WARNING: No SMI results file found, skipping SMI threshold filter")
            
            # Get layer information
            twoP_filename = os.path.basename(tseries_path)
            raw_twop_data = TwoP(tseries_path, twoP_filename)
            raw_twop_data.find_files()
            twop_dict = raw_twop_data.calc_dFF()
            
            med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
            layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)
            
            # Identify valid cells for PCA (Approach B)
            valid_for_pca, peak_positions, mean_profiles, rejection_info = \
                identify_valid_cells_for_pca(
                    normalized_spatial_activity, bin_centers, reliable_cells,
                    exclude_first_bins=exclude_first_bins,
                    exclude_last_bins=exclude_last_bins,
                    smoothing_sigma=smoothing_sigma,
                    landmark_positions=landmark_positions,
                    post_onset_start_cm=POST_ONSET_START_CM,
                    onset_r_threshold=ONSET_R_THRESHOLD,
                    onset_max_shift_cm=ONSET_MAX_SHIFT_CM,
                )
            cell_type_labels_session = rejection_info['cell_type_labels']

            print(f"  Valid for PCA: {rejection_info['n_valid']} / {n_cells}")
            print(f"    Landmark cells (type 1/4):      {np.sum(cell_type_labels_session == CELL_TYPE_LANDMARK)}")
            print(f"    Onset+landmark cells (type 3):  {len(rejection_info['onset_landmark'])}")
            print(f"    Rejected (onset-only, type 2):  {len(rejection_info['onset'])}")
            print(f"    Rejected (reward):              {len(rejection_info['reward'])}")
            print(f"    Rejected (zero):                {len(rejection_info['zero'])}")
            print(f"    Rejected (not reliable):        {len(rejection_info['not_reliable'])}")
            
            # Assign landmark preferences
            landmark_prefs = assign_landmark_preference(
                peak_positions, landmark_positions, landmark_windows_config
            )
            
            # Count landmark preferences for valid cells
            for lm_idx in range(len(landmark_positions)):
                n_pref = np.sum((landmark_prefs == lm_idx) & valid_for_pca)
                print(f"    L{lm_idx+1} preference: {n_pref}")
            n_between = np.sum((landmark_prefs == -1) & valid_for_pca)
            print(f"    Between landmarks: {n_between}")
            
            # Assign layer labels
            layer_labels, cell_depths = assign_layer_labels_and_depths(med_coords, layer_cells)
            
            # Extract trimmed profiles (interpolated to common grid)
            trimmed_profiles, session_bin_centers = extract_trimmed_profiles(
                mean_profiles, bin_centers, trim_start_cm, trim_end_cm,
                target_n_bins=115  # Use 115 since that's what most sessions have
            )
            
            # Store bin centers (should be same across sessions)
            if trimmed_bin_centers is None:
                trimmed_bin_centers = session_bin_centers
            
            # Get indices of valid cells
            valid_indices = np.where(valid_for_pca)[0]
            
            # Aggregate data for valid cells
            for local_idx, cell_idx in enumerate(valid_indices):
                all_profiles.append(trimmed_profiles[cell_idx])
                all_session_labels.append(session_id)
                all_layer_labels.append(layer_labels[cell_idx])
                all_landmark_prefs.append(landmark_prefs[cell_idx])
                all_peak_positions.append(peak_positions[cell_idx])
                all_original_indices.append(cell_idx)
                all_cell_depths.append(cell_depths[cell_idx])  # ← ADD THIS LINE
                all_cell_types.append(cell_type_labels_session[cell_idx])

            # Store session info
            session_info[session_id] = {
                'tseries_path': tseries_path,
                'n_cells_total': n_cells,
                'n_cells_reliable': int(np.sum(reliable_cells)),
                'n_cells_valid_pca': int(np.sum(valid_for_pca)),
                'n_trials': n_trials,
                'rejection_info': rejection_info,
                'layer_boundaries': layer_boundaries
            }
            
        except Exception as e:
            print(f"  ERROR processing session: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # DEBUG: Check profile shapes before converting to array
    print("\n" + "="*60)
    print("DEBUG: Checking profile shapes")
    print("="*60)
    profile_shapes = [p.shape for p in all_profiles]
    unique_shapes = set(profile_shapes)
    print(f"Unique profile shapes found: {unique_shapes}")
    
    if len(unique_shapes) > 1:
        print("\n⚠️ WARNING: Profiles have different shapes!")
        for shape in unique_shapes:
            count = profile_shapes.count(shape)
            print(f"  Shape {shape}: {count} cells")
    
    # Convert to arrays
    all_profiles = np.array(all_profiles)
    
    # Convert to arrays
    all_profiles = np.array(all_profiles)
    all_session_labels = np.array(all_session_labels, dtype='U10')
    all_layer_labels = np.array(all_layer_labels, dtype='U10')
    all_landmark_prefs = np.array(all_landmark_prefs, dtype=int)
    all_peak_positions = np.array(all_peak_positions)
    all_original_indices = np.array(all_original_indices, dtype=int)
    all_cell_depths = np.array(all_cell_depths)  # ← ADD THIS LINE
    all_cell_types  = np.array(all_cell_types,  dtype='U20')

    # Z-score normalize profiles
    all_profiles_zscore = zscore_profiles(all_profiles)

    # Landmark-aligned profiles (optional)
    all_profiles_aligned   = None
    all_optimal_shifts     = None
    all_max_correlations   = None
    if align_profiles:
        if align_method == 'preferred_landmark':
            all_profiles_aligned, all_optimal_shifts, all_max_correlations = \
                align_profiles_preferred_landmark(
                    all_profiles_zscore, trimmed_bin_centers, landmark_positions,
                    all_cell_types,
                    canonical_cm=CANONICAL_LANDMARK_CM,
                    adaptation_peak_threshold=ADAPTATION_PEAK_THRESHOLD,
                    peak_prominence=PEAK_PROMINENCE_PL,
                    peak_min_dist_cm=PEAK_MIN_DIST_CM_PL,
                    max_dist_to_landmark_cm=MAX_DIST_TO_LANDMARK_CM,
                    zscore_after=zscore_after_alignment,
                )
        else:  # 'type_aware' (original template-correlation method)
            all_profiles_aligned, all_optimal_shifts, all_max_correlations = \
                align_profiles_type_aware(
                    all_profiles_zscore, trimmed_bin_centers, landmark_positions,
                    all_cell_types,
                    post_onset_start_cm=POST_ONSET_START_CM,
                    max_shift_cm=ONSET_MAX_SHIFT_CM,
                    sigma_cm=template_sigma_cm,
                    zscore_after=zscore_after_alignment,
                )

    print(f"\n{'='*60}")
    print("AGGREGATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cells for PCA: {len(all_profiles)}")
    print(f"Sessions included: {len(session_info)}")
    print(f"Feature dimensions: {all_profiles.shape[1]} spatial bins")
    print(f"Spatial range: {trimmed_bin_centers[0]:.1f} - {trimmed_bin_centers[-1]:.1f} cm")
    
    # Count by session
    print(f"\nCells per session:")
    for session_id in sorted(session_info.keys(), key=lambda x: int(x.replace('Day', ''))):
        n_cells = np.sum(all_session_labels == session_id)
        print(f"  {session_id}: {n_cells}")
    
    # Count by layer
    print(f"\nCells per layer:")
    for layer in ['L2/3', 'L4', 'L5', 'L6']:
        n_cells = np.sum(all_layer_labels == layer)
        print(f"  {layer}: {n_cells}")
    
    # Count by landmark preference
    print(f"\nCells by landmark preference:")
    for lm_idx in range(len(landmark_positions)):
        n_cells = np.sum(all_landmark_prefs == lm_idx)
        print(f"  L{lm_idx+1} ({landmark_positions[lm_idx]}cm): {n_cells}")
    n_between = np.sum(all_landmark_prefs == -1)
    print(f"  Between landmarks: {n_between}")
    
    # Save to HDF5
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{animal_id}_pca_data.h5")
    
    print(f"\nSaving to: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['animal_id'] = animal_id
        meta.attrs['n_sessions'] = len(session_info)
        meta.attrs['n_cells_total'] = len(all_profiles)
        meta.attrs['creation_timestamp'] = datetime.now().isoformat()
        meta.attrs['trim_start_cm'] = trim_start_cm
        meta.attrs['trim_end_cm'] = trim_end_cm
        meta.attrs['exclude_first_bins'] = exclude_first_bins
        meta.attrs['exclude_last_bins'] = exclude_last_bins
        meta.attrs['smoothing_sigma'] = smoothing_sigma
        
        # Store session IDs as dataset (HDF5 handles string arrays better this way)
        session_ids = sorted(session_info.keys(), key=lambda x: int(x.replace('Day', '')))
        meta.create_dataset('session_ids', data=np.array(session_ids, dtype='S10'))
        
        meta.create_dataset('bin_centers_trimmed', data=trimmed_bin_centers)
        meta.create_dataset('landmark_positions', data=np.array(landmark_positions))
        
        # Store landmark windows config
        lm_config = meta.create_group('landmark_windows_config')
        for i, config in enumerate(landmark_windows_config):
            lm_config.attrs[f'L{i+1}_before'] = config['before']
            lm_config.attrs[f'L{i+1}_after'] = config['after']
        
        # Cell labels
        cells = f.create_group('cells')
        
        # Convert string arrays to bytes for HDF5 compatibility
        cells.create_dataset('session_labels', data=all_session_labels.astype('S10'))
        cells.create_dataset('layer_labels', data=all_layer_labels.astype('S10'))
        cells.create_dataset('preferred_landmark', data=all_landmark_prefs)
        cells.create_dataset('peak_positions', data=all_peak_positions)
        cells.create_dataset('original_cell_indices', data=all_original_indices)
        cells.create_dataset('cell_depths', data=all_cell_depths)  # ← ADD THIS LINE
        cells.create_dataset('cell_types', data=all_cell_types.astype('S20'))

        # Features
        features = f.create_group('features')
        features.create_dataset('spatial_profiles', data=all_profiles)
        features.create_dataset('spatial_profiles_zscore', data=all_profiles_zscore)
        if all_profiles_aligned is not None:
            features.create_dataset('spatial_profiles_aligned', data=all_profiles_aligned)
            align_grp = f.create_group('alignment')
            align_grp.create_dataset('optimal_shifts', data=all_optimal_shifts)
            align_grp.create_dataset('max_correlations', data=all_max_correlations)
            align_grp.attrs['method'] = align_method
            align_grp.attrs['zscore_after_alignment'] = zscore_after_alignment
            align_grp.attrs['post_onset_start_cm'] = POST_ONSET_START_CM
            align_grp.attrs['onset_r_threshold']   = ONSET_R_THRESHOLD
            align_grp.attrs['onset_max_shift_cm']  = ONSET_MAX_SHIFT_CM
        
        # Session info
        sessions_grp = f.create_group('sessions')
        for session_id, info in session_info.items():
            sess = sessions_grp.create_group(session_id)
            sess.attrs['tseries_path'] = info['tseries_path']
            sess.attrs['n_cells_total'] = info['n_cells_total']
            sess.attrs['n_cells_reliable'] = info['n_cells_reliable']
            sess.attrs['n_cells_valid_pca'] = info['n_cells_valid_pca']
            sess.attrs['n_trials'] = info['n_trials']
        
        # Placeholder for PCA results (will be filled by analysis script)
        f.create_group('pca_results')
    
    print(f"\n✓ Data saved successfully!")
    print(f"  File: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    return output_path


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    output_path = aggregate_pca_data(
        animal_id=ANIMAL_ID,
        base_dir=BASE_DIR,
        output_dir=OUTPUT_DIR,
        landmark_positions=LANDMARK_POSITIONS,
        landmark_windows_config=LANDMARK_WINDOWS_CONFIG,
        trim_start_cm=TRIM_START_CM,
        trim_end_cm=TRIM_END_CM,
        exclude_first_bins=EXCLUDE_FIRST_BINS,
        exclude_last_bins=EXCLUDE_LAST_BINS,
        smoothing_sigma=SMOOTHING_SIGMA,
        align_profiles=ALIGN_PROFILES,
        align_method=ALIGN_METHOD,
        zscore_after_alignment=ZSCORE_AFTER_ALIGNMENT,
        template_sigma_cm=TEMPLATE_SIGMA_CM,
    )