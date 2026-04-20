"""
PCA_TrackedCells.py
===================
Population-level PCA + k-means analysis, restricted to **tracked cells only**.

Mirrors 4.PCA/PCA_ComprehensiveAnalysis.py exactly, but instead of loading
pre-aggregated data from *_pca_data.h5, it builds the feature matrix on the
fly from the tracked-ROI file + per-session preproc HDF5 files.

Each row of the PCA matrix is one (tracked-cell × session) observation.
Cells must be valid (roi index ≥ 0, reliable, not onset/reward responders)
in a given session to be included for that session.

Output figures:
  *_tracked_scree.png
  *_tracked_k_selection.png
  *_tracked_pc_scatter.png
  *_tracked_mean_profiles.png
  *_tracked_global_proportions.png
  *_tracked_stacked_bars_by_layer.png
  *_tracked_trajectories_by_layer.png
  *_tracked_trajectories_by_type.png

Prerequisite: run TrackROIs.ipynb → roi_tracking_results.h5
              and ensure *_preproc*.h5 files exist per session.

JSY, 2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d as scipy_interp1d
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, linregress
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from load_tracked import (
    load_tracking, filter_to_analysis_days, find_files_from_tracking,
    find_preproc_files, assign_layers_from_smi, load_preproc_session,
    build_reliability_mask, parse_day_numbers, animal_id_from_path,
    LAYER_ORDER, LAYER_COLORS, report_found_files,
)


# ============================================================
# CONFIGURATION
# ============================================================
ROI_TRACKING_FILE = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\TrackedROIs\roi_tracking_results.h5"
ANIMAL_DIR        = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging"
REFERENCE_DAY     = "Day5"
ANALYSIS_DAYS     = ['Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']  # set to None to use all tracked sessions

# Profile preprocessing — must match PCA_DataAggregation.py
TRIM_START_CM     = 10.0
TRIM_END_CM       = 120.0
TARGET_N_BINS     = 115
SMOOTH_SIGMA      = 1.0
EXCLUDE_FIRST_BINS = 10
EXCLUDE_LAST_BINS  = 10

# Cell selection — must match PCA_DataAggregation.py
# 'reliable_cells'       — basic reliability from preproc
# 'combined_reliable'    — stricter: CC, Cohen's d, pattern correlation
# 'reliable_valid_cells' — combined_reliable + valid SMI geometry (needs SMI h5)
CELL_SELECTION = 'reliable_cells'

# Fixed-pool tracking: if True, reliability is evaluated only on REFERENCE_DAY
# and that fixed set of cells is followed across all sessions.
# If False, each session independently gates on its own reliability (population varies).
FIXED_POOL = False

# Alignment (same as PCA_DataAggregation.py)
USE_ALIGNED_PROFILES  = False   # True → type-aware alignment, False → z-scored only
POST_ONSET_START_CM   = 35.0
ONSET_R_THRESHOLD     = 0.3
ONSET_MAX_SHIFT_CM    = 15.0
CELL_TYPE_LANDMARK        = 'landmark'
CELL_TYPE_ONSET_ONLY      = 'onset_only'
CELL_TYPE_ONSET_LANDMARK  = 'onset_landmark'
TEMPLATE_SIGMA_CM         = 8.0

# PCA / clustering — must match 4.PCA/PCA_ComprehensiveAnalysis.py
N_PCA_COMPONENTS  = 10
N_CLUSTER_PCS     = 5
K_RANGE           = range(2, 8)
OVERRIDE_K        = 2   # set to int to force k, or None for auto

OUTPUT_DIR = os.path.join(ANIMAL_DIR, "TrackedROIs/PCA")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ============================================================


# ── Profile preprocessing helpers ────────────────────────────

def _zscore(profile):
    mu, sigma = np.mean(profile), np.std(profile)
    return (profile - mu) / sigma if sigma > 0 else profile - mu


def _preprocess_one(raw_profile, bc,
                    trim_start, trim_end, target_n_bins, smooth_sigma):
    """
    Trim → interpolate → z-score a single mean spatial profile.
    Returns (n_bins,) array on linspace(trim_start, trim_end, target_n_bins).
    """
    if smooth_sigma > 0:
        raw_profile = gaussian_filter1d(raw_profile.astype(float), sigma=smooth_sigma)

    start_idx = np.searchsorted(bc, trim_start)
    end_idx   = np.searchsorted(bc, trim_end)
    trimmed   = raw_profile[start_idx:end_idx]
    bc_trim   = bc[start_idx:end_idx]

    common_bc = np.linspace(trim_start, trim_end, target_n_bins)
    f = scipy_interp1d(bc_trim, trimmed, kind='linear',
                       bounds_error=False, fill_value='extrapolate')
    interpolated = f(common_bc)
    return _zscore(interpolated), common_bc


def _is_valid_cell(mean_profile, bc, roi_idx, reliable_cells,
                   exclude_first_bins, exclude_last_bins,
                   landmark_positions=None):
    """
    Return the cell type string ('landmark', 'onset_only', 'onset_landmark')
    if the cell should be included, or False if it should be excluded.

    Rules:
      - roi_idx ≥ 0 and in range
      - cell is reliable
      - zero-activity cells → excluded
      - reward zone peak → excluded
      - onset zone peak → classify via classify_onset_cell if landmark_positions
        provided; 'onset_landmark' cells are included, 'onset_only' excluded
      - all other cells → 'landmark'
    """
    if roi_idx < 0 or roi_idx >= len(reliable_cells):
        return False
    if not reliable_cells[roi_idx]:
        return False

    profile = mean_profile
    if np.max(profile) == 0:
        return False

    bin_spacing   = np.mean(np.diff(bc))
    onset_thresh  = bc[0]  + exclude_first_bins  * bin_spacing
    reward_thresh = bc[-1] - exclude_last_bins    * bin_spacing
    peak_pos      = bc[np.argmax(profile)]

    if peak_pos > reward_thresh:
        return False

    if peak_pos < onset_thresh:
        if landmark_positions is not None:
            ctype, _, _ = classify_onset_cell(
                profile, bc, landmark_positions,
                post_onset_start_cm=POST_ONSET_START_CM,
                max_shift_cm=ONSET_MAX_SHIFT_CM,
                r_threshold=ONSET_R_THRESHOLD,
            )
            if ctype == CELL_TYPE_ONSET_LANDMARK:
                return CELL_TYPE_ONSET_LANDMARK
            else:
                return False
        else:
            return False

    return CELL_TYPE_LANDMARK


# ── Alignment helpers (copied from PCA_DataAggregation.py) ───

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
    cell_types          : (n_cells,) str array — from _is_valid_cell
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
        for i in range(n_cells):
            p = aligned[i]
            mu, sigma = np.mean(p), np.std(p)
            aligned[i] = (p - mu) / sigma if sigma > 0 else p - mu

    shifts_cm = optimal_shifts * bin_spacing
    print("\n  Alignment summary:")
    print(f"    Landmark total:                    {n_landmark} cells")
    print(f"      Single-peak (±{max_shift_cm} cm):          {n_single_peak} cells")
    print(f"      Multi-peak  (±{n_bins // 2} bins):         {n_multi_peak} cells")
    print(f"    Onset+landmark (post-onset shift): {n_onset_landmark} cells")
    print(f"    Not shifted:                       {n_no_shift} cells")
    if n_landmark + n_onset_landmark > 0:
        shifted_mask = optimal_shifts != 0
        if np.any(shifted_mask):
            print(f"    Mean shift (shifted cells): "
                  f"{np.mean(shifts_cm[shifted_mask]):.1f} cm, "
                  f"std={np.std(shifts_cm[shifted_mask]):.1f} cm")

    return aligned, optimal_shifts, max_correlations


# ── Data aggregation from tracked ROIs ───────────────────────

def aggregate_tracked_profiles(tracked_matrix, day_labels, preproc_files,
                                cell_layers,
                                trim_start, trim_end, target_n_bins,
                                smooth_sigma, exclude_first_bins, exclude_last_bins,
                                landmark_positions=None, smi_files=None):
    """
    Build the PCA feature matrix from tracked cells.

    Each valid (tracked_cell × session) pair becomes one row.
    If USE_ALIGNED_PROFILES is True, type-aware alignment is applied after
    z-scoring. Onset+landmark cells (type 3) are included.

    Returns
    -------
    profiles       : (n_obs, target_n_bins) float — z-scored (+ optionally aligned)
    bin_centers    : (target_n_bins,) float
    session_labels : (n_obs,) str
    layer_labels   : (n_obs,) str
    tracked_ids    : (n_obs,) int — row index in tracked_matrix
    cell_types     : (n_obs,) str — 'landmark' or 'onset_landmark'
    """
    n_tracked, n_sessions = tracked_matrix.shape

    # Reverse-map each tracked cell to its layer
    cell_to_layer = {}
    for layer, rows in cell_layers.items():
        for r in rows:
            cell_to_layer[int(r)] = layer

    # Pre-compute reliability mask for all tracked cells × sessions
    ref_day = REFERENCE_DAY if FIXED_POOL else None
    rel_mask = build_reliability_mask(tracked_matrix, day_labels, preproc_files,
                                      cell_selection=CELL_SELECTION,
                                      smi_files=smi_files,
                                      reference_day=ref_day)

    # With FIXED_POOL, classify cell types once on the reference day so the
    # same cells (and same type labels) are used across all sessions.
    ref_cell_types = {}  # {tracked_row: cell_type_str} — populated when FIXED_POOL
    if FIXED_POOL and ref_day in day_labels:
        ref_col = day_labels.index(ref_day)
        if ref_day in preproc_files:
            _, nsa_ref, bc_ref, _, _, _, _ = load_preproc_session(preproc_files[ref_day])
            mean_ref = np.mean(nsa_ref, axis=1)
            ones_ref = np.ones(nsa_ref.shape[0], dtype=bool)
            for row in range(n_tracked):
                if not rel_mask[row, ref_col]:
                    continue
                roi_idx = int(tracked_matrix[row, ref_col])
                if not (0 <= roi_idx < len(mean_ref)):
                    continue
                ctype = _is_valid_cell(mean_ref[roi_idx], bc_ref, roi_idx,
                                       ones_ref, exclude_first_bins, exclude_last_bins,
                                       landmark_positions=landmark_positions)
                if ctype is not False:
                    ref_cell_types[row] = ctype

    profiles_list       = []
    session_labels_list = []
    layer_labels_list   = []
    tracked_ids_list    = []
    cell_types_list     = []
    common_bc           = None

    for col, day in enumerate(day_labels):
        if day not in preproc_files:
            print(f"  [WARNING] No preproc file for {day} — skipping")
            continue

        print(f"  Aggregating {day}...", end=' ', flush=True)
        _, nsa, bc, _, _, _, _ = load_preproc_session(preproc_files[day])

        mean_profiles_sess = np.mean(nsa, axis=1)   # (n_cells_sess, n_bins)
        n_valid_day = 0

        for row in range(n_tracked):
            if not rel_mask[row, col]:
                continue

            roi_idx = int(tracked_matrix[row, col])
            raw_profile = mean_profiles_sess[roi_idx] if (
                0 <= roi_idx < len(mean_profiles_sess)) else None

            if raw_profile is None:
                continue

            if FIXED_POOL:
                # Use reference-day classification; skip cells excluded on ref day
                cell_type = ref_cell_types.get(row, False)
            else:
                cell_type = _is_valid_cell(raw_profile, bc, roi_idx,
                                           np.ones(nsa.shape[0], dtype=bool),
                                           exclude_first_bins, exclude_last_bins,
                                           landmark_positions=landmark_positions)
            if cell_type is False:
                continue

            proc, cbc = _preprocess_one(raw_profile, bc,
                                        trim_start, trim_end,
                                        target_n_bins, smooth_sigma)
            if common_bc is None:
                common_bc = cbc

            profiles_list.append(proc)
            session_labels_list.append(day)
            layer_labels_list.append(cell_to_layer.get(row, 'Unknown'))
            tracked_ids_list.append(row)
            cell_types_list.append(cell_type)
            n_valid_day += 1

        print(f"{n_valid_day} valid observations")

    profiles       = np.array(profiles_list)
    session_labels = np.array(session_labels_list)
    layer_labels   = np.array(layer_labels_list)
    tracked_ids    = np.array(tracked_ids_list, dtype=int)
    cell_types     = np.array(cell_types_list, dtype='U20')

    print(f"\n  Total observations for PCA: {len(profiles)}")
    print(f"  Feature dims: {profiles.shape[1]} bins "
          f"({trim_start}–{trim_end} cm)")

    n_lm = int(np.sum(cell_types == CELL_TYPE_LANDMARK))
    n_ol = int(np.sum(cell_types == CELL_TYPE_ONSET_LANDMARK))
    print(f"  Cell types: landmark={n_lm}  onset_landmark={n_ol}")

    # Apply type-aware alignment if requested
    if USE_ALIGNED_PROFILES and len(profiles) > 0 and landmark_positions is not None:
        print("\n  Applying type-aware alignment...")
        profiles, _, _ = align_profiles_type_aware(
            profiles, common_bc, landmark_positions,
            cell_types,
            post_onset_start_cm=POST_ONSET_START_CM,
            max_shift_cm=ONSET_MAX_SHIFT_CM,
            sigma_cm=TEMPLATE_SIGMA_CM,
        )

    return profiles, common_bc, session_labels, layer_labels, tracked_ids, cell_types


# ── PCA + clustering (identical to PCA_ComprehensiveAnalysis) ─

def run_pca(profiles, n_components):
    pca    = PCA(n_components=n_components)
    scores = pca.fit_transform(profiles)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"  Variance explained: "
          + "  ".join([f"PC{i+1}={pca.explained_variance_ratio_[i]*100:.1f}%"
                       for i in range(min(5, n_components))])
          + f"  |  top-{n_components} cumulative: {cumvar[-1]:.1f}%")
    return pca, scores


def select_optimal_k(X, k_range, random_state=42):
    sil_scores, inertias = [], []
    for k in k_range:
        km  = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        lbs = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, lbs))
        inertias.append(km.inertia_)
    auto_k = list(k_range)[int(np.argmax(sil_scores))]
    return sil_scores, inertias, auto_k


def fit_kmeans(X, k, random_state=42):
    km     = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(X)
    return km, labels


def assign_semantic_labels(raw_labels, profiles, n_types,
                            bin_centers, landmark_positions):
    mean_profiles = np.array([
        np.mean(profiles[raw_labels == t], axis=0)
        if np.sum(raw_labels == t) > 0 else np.zeros(profiles.shape[1])
        for t in range(n_types)
    ])
    peak_bins = np.argmax(mean_profiles, axis=1)

    names, used = [None] * n_types, set()

    def _pick(order):
        for k in order:
            if k not in used:
                used.add(k)
                return k

    names[_pick(np.argsort(peak_bins))]       = 'Adaptation-like'
    names[_pick(np.argsort(peak_bins)[::-1])] = 'L4-preferring'
    # ranges = mean_profiles.max(axis=1) - mean_profiles.min(axis=1)
    # names[_pick(np.argsort(ranges))]           = 'Visually responsive'

    for k in range(n_types):
        if names[k] is None:
            peak_cm  = bin_centers[peak_bins[k]] if peak_bins[k] < len(bin_centers) else -1
            names[k] = f'Peak~{peak_cm:.0f}cm'

    base_colors = {
        'Adaptation-like':    '#C62828',  # deep red
        'L4-preferring':      '#00838F',  # teal
        'Visually responsive':'#F9A825',  # amber
    }
    extra = ['#6D4C41', '#AD1457', '#558B2F', '#00695C']
    ec, colors = 0, []
    for name in names:
        colors.append(base_colors.get(name, extra[ec % len(extra)]))
        if name not in base_colors:
            ec += 1

    return names, colors, mean_profiles


def build_prop_matrix(raw_labels, session_labels, layer_labels,
                       session_order, layer_list, n_types):
    n_sessions = len(session_order)
    n_layers   = len(layer_list)
    prop   = np.full((n_types, n_layers, n_sessions), np.nan)
    counts = np.zeros((n_layers, n_sessions), dtype=int)

    for si, sess in enumerate(session_order):
        sess_mask = session_labels == sess
        for li, layer in enumerate(layer_list):
            mask = sess_mask & (layer_labels == layer)
            n    = int(np.sum(mask))
            if n == 0:
                continue
            counts[li, si] = n
            for t in range(n_types):
                prop[t, li, si] = np.sum(raw_labels[mask] == t) / n

    return prop, counts


# ── Plotting (identical to PCA_ComprehensiveAnalysis) ─────────

def plot_scree(pca, n_cluster_pcs, animal_id, output_path=None):
    var    = pca.explained_variance_ratio_ * 100
    cumvar = np.cumsum(var)
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(f'{animal_id} (tracked) — PCA Scree', fontweight='bold')
    ax.bar(range(1, len(var) + 1), var, color='steelblue', alpha=0.8)
    ax.plot(range(1, len(var) + 1), cumvar, 'ko-', markersize=4)
    ax.axvline(n_cluster_pcs + 0.5, color='red', linestyle='--',
               label=f'Clustering uses top {n_cluster_pcs} PCs')
    ax.set_xlabel('PC'); ax.set_ylabel('Variance explained (%)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_k_selection(sil_scores, inertias, k_range, optimal_k,
                     animal_id, output_path=None):
    ks = list(k_range)
    fig, ax1 = plt.subplots(figsize=(5, 4))
    fig.suptitle(f'{animal_id} (tracked) — K selection', fontweight='bold')
    ax2 = ax1.twinx()
    ax1.plot(ks, sil_scores, 'b-o', label='Silhouette', linewidth=2)
    ax2.plot(ks, inertias,   'r-s', label='Inertia',    linewidth=2)
    ax1.axvline(optimal_k, color='green', linestyle='--', label=f'k={optimal_k}')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Silhouette score', color='blue')
    ax2.set_ylabel('Inertia',          color='red')
    ax1.set_xticks(ks)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors,
                    animal_id, output_path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.suptitle(f'{animal_id} (tracked) — PC scatter', fontweight='bold')
    for t, (name, col) in enumerate(zip(type_names, type_colors)):
        m = raw_labels == t
        ax.scatter(pc_scores[m, 0], pc_scores[m, 1],
                   c=col, alpha=0.5, s=15, label=f'{name} (n={m.sum()})')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(fontsize=7, markerscale=2); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                        type_names, type_colors, animal_id, output_path=None):
    n_types = len(type_names)
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 3.5), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Mean profiles per cluster',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        ax.plot(bin_centers, mean_profiles[t], color=type_colors[t], linewidth=2)
        for lp in landmark_positions:
            ax.axvline(lp, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(type_names[t], color=type_colors[t],
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Z-scored activity' if t == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_global_proportions(raw_labels, layer_labels, layer_list,
                             type_names, type_colors, animal_id, output_path=None):
    n_types  = len(type_names)
    n_layers = len(layer_list)
    props, ns = np.zeros((n_types, n_layers)), np.zeros(n_layers, dtype=int)
    for li, layer in enumerate(layer_list):
        mask = layer_labels == layer
        n    = int(np.sum(mask))
        ns[li] = n
        for t in range(n_types):
            props[t, li] = np.sum(raw_labels[mask] == t) / n if n > 0 else 0

    fig, ax = plt.subplots(figsize=(max(4, n_layers * 1.8), 4))
    fig.suptitle(f'{animal_id} (tracked) — Cell type proportions by layer (pooled)',
                 fontweight='bold')
    xpos    = np.arange(n_layers)
    bottoms = np.zeros(n_layers)
    for t in range(n_types):
        ax.bar(xpos, props[t] * 100, bottom=bottoms,
               color=type_colors[t], label=type_names[t], alpha=0.85, width=0.7)
        bottoms += props[t] * 100
    for xi, n in enumerate(ns):
        ax.text(xi, 103, f'n={n}', ha='center', va='bottom', fontsize=7, rotation=45)
    ax.set_xticks(xpos)
    ax.set_xticklabels(layer_list)
    ax.set_ylim(0, 120); ax.set_ylabel('% observations')
    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    ax.legend(handles=handles, fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_stacked_bars_by_layer(prop, counts, layer_list, session_order,
                                type_names, type_colors, animal_id,
                                output_path=None):
    n_layers, n_types = len(layer_list), len(type_names)
    fig, axes = plt.subplots(1, n_layers, figsize=(3.5 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Cell type proportions per layer × session',
                 fontweight='bold')
    if n_layers == 1:
        axes = [axes]
    xpos = np.arange(len(session_order))
    for li, (layer, ax) in enumerate(zip(layer_list, axes)):
        bottoms = np.zeros(len(session_order))
        for t in range(n_types):
            vals    = prop[t, li, :]
            heights = np.where(~np.isnan(vals), vals * 100, 0)
            ax.bar(xpos, heights, bottom=bottoms,
                   color=type_colors[t],
                   label=type_names[t] if li == 0 else '',
                   width=0.7, alpha=0.85)
            bottoms += heights
        for xi in range(len(session_order)):
            n = counts[li, xi]
            if n > 0:
                ax.text(xi, 103, f'n={n}', ha='center', va='bottom',
                        fontsize=6, rotation=45)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
        ax.set_xticks(xpos)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 120)
        ax.set_ylabel('% observations' if li == 0 else '')
        ax.grid(True, alpha=0.2, axis='y')
    handles = [mpatches.Patch(color=type_colors[t], label=type_names[t])
               for t in range(n_types)]
    axes[-1].legend(handles=handles, fontsize=7, loc='upper right')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def _session_days(session_order):
    return [int(s.replace('Day', '')) for s in session_order]


def plot_trajectories_by_layer(prop, layer_list, session_order,
                                type_names, type_colors, animal_id,
                                output_path=None):
    n_layers, n_types = len(layer_list), len(type_names)
    x = np.array(_session_days(session_order))
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Cell type trajectories per layer',
                 fontweight='bold')
    if n_layers == 1:
        axes = [axes]
    for li, (layer, ax) in enumerate(zip(layer_list, axes)):
        for t in range(n_types):
            y     = prop[t, li, :] * 100
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], y[valid], 'o-', color=type_colors[t],
                    label=type_names[t], linewidth=2, markersize=5)
        ax.set_title(layer, color=LAYER_COLORS.get(layer, 'black'), fontweight='bold')
        ax.set_xlabel('Recording day')
        ax.set_ylabel('% observations' if li == 0 else '')
        ax.set_ylim(0, 100); ax.set_xticks(x)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def plot_trajectories_by_type(prop, layer_list, session_order,
                               type_names, type_colors, animal_id,
                               output_path=None):
    n_types = len(type_names)
    x = np.array(_session_days(session_order))
    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4), sharey=True)
    fig.suptitle(f'{animal_id} (tracked) — Layer trajectories per cell type',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        for li, layer in enumerate(layer_list):
            y     = prop[t, li, :] * 100
            valid = ~np.isnan(y)
            if valid.sum() < 2:
                continue
            ax.plot(x[valid], y[valid], 'o-',
                    color=LAYER_COLORS.get(layer, 'gray'),
                    label=layer, linewidth=2, markersize=5)
        ax.set_title(type_names[t], color=type_colors[t], fontweight='bold')
        ax.set_xlabel('Recording day')
        ax.set_ylabel('% observations' if t == 0 else '')
        ax.set_ylim(0, 100); ax.set_xticks(x)
        ax.set_xticklabels(session_order, rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ── Statistical tests ─────────────────────────────────────────

def run_chi_square_tests(raw_labels, session_labels, layer_labels,
                          session_order, layer_list, n_types):
    print("\n── Chi-square: layer × cell type, per session ──")
    for sess in session_order:
        sess_mask = session_labels == sess
        table = np.zeros((len(layer_list), n_types), dtype=int)
        for li, layer in enumerate(layer_list):
            mask = sess_mask & (layer_labels == layer)
            for t in range(n_types):
                table[li, t] = int(np.sum(raw_labels[mask] == t))
        if table.sum() == 0 or table.min() < 1:
            print(f"  {sess}: sparse/no data — skipping")
            continue
        try:
            chi2, p, dof, _ = chi2_contingency(table)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"  {sess}: chi2={chi2:.2f}  df={dof}  p={p:.4f}  {sig}")
        except Exception as e:
            print(f"  {sess}: failed ({e})")


def run_kruskal_layer_tests(raw_labels, layer_labels, layer_list,
                             n_types, type_names):
    print("\n── Kruskal-Wallis: layer effect on cell type proportion ──")
    for t, tname in enumerate(type_names):
        groups = []
        for layer in layer_list:
            mask = layer_labels == layer
            if np.sum(mask) == 0:
                continue
            groups.append((raw_labels[mask] == t).astype(float))
        if len(groups) < 2:
            print(f"  {tname}: not enough layers")
            continue
        try:
            stat, p = kruskal(*groups)
            sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
            print(f"  {tname}: KW H={stat:.2f}  p={p:.4f}  {sig}")
            for layer in layer_list:
                mask = layer_labels == layer
                if np.sum(mask) == 0:
                    continue
                pct = np.mean(raw_labels[mask] == t) * 100
                print(f"    {layer}: {pct:.1f}%  (n={np.sum(mask)})")
        except Exception as e:
            print(f"  {tname}: failed ({e})")


# ── Statistical helpers ───────────────────────────────────────

def _sig_stars(p):
    return '***' if p < 0.001 else ('**' if p < 0.01
           else ('*' if p < 0.05 else 'ns'))


def _draw_sig_bracket(ax, x1, x2, y, h, text, fontsize=8):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c='black')
    ax.text((x1 + x2) / 2, y + h, text,
            ha='center', va='bottom', fontsize=fontsize)


def run_layer_posthoc(raw_labels, layer_labels, layer_list, n_types, type_names):
    """
    Per cell type: Kruskal-Wallis across layers (pooled sessions) +
    Bonferroni-corrected pairwise Mann-Whitney U post-hoc tests.
    Returns results[tname] = {'kw_p': float, 'posthoc': {(l1,l2): p_adj}}
    """
    print("\n── Layer effect: KW + post-hoc Mann-Whitney (Bonferroni) ──")
    results = {}
    for t, tname in enumerate(type_names):
        groups, grp_layers = [], []
        for layer in layer_list:
            mask = layer_labels == layer
            if np.sum(mask) == 0:
                continue
            groups.append((raw_labels[mask] == t).astype(float))
            grp_layers.append(layer)

        if len(groups) < 2:
            print(f"  {tname}: not enough layers")
            continue

        try:
            stat, kw_p = kruskal(*groups)
            sig = _sig_stars(kw_p)
            print(f"\n  {tname}: KW H={stat:.2f}  p={kw_p:.4f}  {sig}")
            for layer, grp in zip(grp_layers, groups):
                print(f"    {layer}: {np.mean(grp)*100:.1f}%  (n={len(grp)})")
        except Exception as e:
            print(f"  {tname}: KW failed ({e})")
            continue

        # Pairwise post-hoc
        pairs = [(i, j) for i in range(len(groups))
                         for j in range(i + 1, len(groups))]
        n_pairs = len(pairs)
        posthoc = {}
        for i, j in pairs:
            try:
                _, p_mw = mannwhitneyu(groups[i], groups[j],
                                       alternative='two-sided')
                p_adj = min(p_mw * n_pairs, 1.0)
                key   = (grp_layers[i], grp_layers[j])
                posthoc[key] = p_adj
                print(f"    {grp_layers[i]} vs {grp_layers[j]}: "
                      f"p_adj={p_adj:.4f}  {_sig_stars(p_adj)}")
            except Exception:
                pass

        results[tname] = {'kw_p': kw_p, 'posthoc': posthoc}
    return results


def plot_layer_posthoc(raw_labels, layer_labels, layer_list, n_types,
                        type_names, type_colors, animal_id, output_path=None):
    """
    Bar = proportion per layer (pooled sessions) per cell type,
    with Bonferroni-corrected Mann-Whitney U significance brackets.
    KW omnibus p shown in x-label.
    """
    fig, axes = plt.subplots(1, n_types, figsize=(4 * n_types, 5))
    fig.suptitle(f'{animal_id} (tracked) — Layer effect + post-hoc (Bonferroni)',
                 fontweight='bold')
    if n_types == 1:
        axes = [axes]

    xpos = np.arange(len(layer_list))
    for t, (tname, tcolor, ax) in enumerate(zip(type_names, type_colors, axes)):
        grand_mean, groups_here, grp_layers = [], [], []
        for layer in layer_list:
            mask = layer_labels == layer
            n    = int(np.sum(mask))
            grand_mean.append(np.mean(raw_labels[mask] == t) * 100 if n > 0 else 0)
            if n > 0:
                groups_here.append((raw_labels[mask] == t).astype(float))
                grp_layers.append(layer)

        ax.bar(xpos, grand_mean, color=tcolor, alpha=0.7, width=0.6, zorder=2)

        for xi, (layer, val) in enumerate(zip(layer_list, grand_mean)):
            n = int(np.sum(layer_labels == layer))
            ax.text(xi, val + 1, f'n={n}', ha='center', va='bottom',
                    fontsize=7, rotation=45)

        # Post-hoc brackets
        pairs   = [(i, j) for i in range(len(groups_here))
                           for j in range(i + 1, len(groups_here))]
        n_pairs = len(pairs)
        sig_pairs = []
        for i, j in pairs:
            try:
                _, p_mw = mannwhitneyu(groups_here[i], groups_here[j],
                                       alternative='two-sided')
                p_adj = min(p_mw * n_pairs, 1.0)
                if p_adj < 0.05:
                    xi = layer_list.index(grp_layers[i])
                    xj = layer_list.index(grp_layers[j])
                    sig_pairs.append((xi, xj, _sig_stars(p_adj)))
            except Exception:
                pass

        y_top = max(grand_mean) if grand_mean else 0
        step  = max(y_top * 0.13, 3)
        base  = y_top * 1.15
        placed = []
        for xi, xj, stars in sorted(sig_pairs, key=lambda x: abs(x[1] - x[0])):
            level = 0
            while any(not (xj < px or xi > pxj) and lv == level
                      for px, pxj, lv in placed):
                level += 1
            placed.append((xi, xj, level))
            _draw_sig_bracket(ax, xi, xj, base + level * step, step * 0.3, stars)

        ax.set_xticks(xpos)
        ax.set_xticklabels(layer_list, fontsize=9)
        ax.set_title(tname, color=tcolor, fontweight='bold')
        ax.set_ylabel('% observations' if t == 0 else '')
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.2, axis='y')

        try:
            _, kw_p = kruskal(*groups_here)
            ax.set_xlabel(f'KW p={kw_p:.4f} {_sig_stars(kw_p)}', fontsize=8)
        except Exception:
            pass

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def run_experience_effect(raw_labels, session_labels, layer_labels,
                           session_order, layer_list, n_types, type_names):
    """
    Per cell type × layer: OLS regression of proportion ~ recording day.
    Returns results[tname][layer] = {'slope': float, 'p': float, 'days': array,
                                      'props': array}
    """
    print("\n── Experience effect: proportion ~ day (OLS per type × layer) ──")
    days = np.array([int(s.replace('Day', '')) for s in session_order], dtype=float)
    results = {}
    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        results[tname] = {}
        for layer in layer_list:
            props, valid_days = [], []
            for si, sess in enumerate(session_order):
                mask = (session_labels == sess) & (layer_labels == layer)
                n    = int(np.sum(mask))
                if n == 0:
                    continue
                props.append(np.mean(raw_labels[mask] == t) * 100)
                valid_days.append(days[si])

            if len(valid_days) < 3:
                results[tname][layer] = None
                continue

            vd = np.array(valid_days)
            vp = np.array(props)
            try:
                slope, intercept, r, p, _ = linregress(vd, vp)
                print(f"    {layer}: slope={slope:+.2f}%/day  "
                      f"r={r:.3f}  p={p:.4f}  {_sig_stars(p)}")
                results[tname][layer] = {
                    'slope': slope, 'intercept': intercept,
                    'r': r, 'p': p,
                    'days': vd, 'props': vp,
                }
            except Exception as e:
                print(f"    {layer}: failed ({e})")
                results[tname][layer] = None
    return results


def plot_experience_regression(exp_results, layer_list, session_order,
                                type_names, type_colors, animal_id,
                                output_path=None):
    """
    Grid: rows = cell types, columns = layers.
    Each panel: scatter of proportion per session + OLS line.
    Solid line = p<0.05, dashed = ns.  Slope + p annotated.
    """
    n_types  = len(type_names)
    n_layers = len(layer_list)
    days_all = np.array([int(s.replace('Day', '')) for s in session_order],
                        dtype=float)

    fig, axes = plt.subplots(n_types, n_layers,
                             figsize=(4 * n_layers, 3.5 * n_types),
                             sharey='row', sharex='col')
    fig.suptitle(f'{animal_id} (tracked) — Experience effect: proportion ~ day',
                 fontweight='bold')

    # Ensure 2-D array
    if n_types == 1:
        axes = axes[np.newaxis, :]
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    for t, tname in enumerate(type_names):
        for li, layer in enumerate(layer_list):
            ax  = axes[t, li]
            res = exp_results[tname].get(layer)

            if t == 0:
                ax.set_title(layer,
                             color=LAYER_COLORS.get(layer, 'black'),
                             fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{tname}\n% observations',
                              color=type_colors[t], fontsize=8)

            if res is None:
                ax.text(0.5, 0.5, 'insufficient data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=7, color='gray')
                ax.set_xlabel('Day' if t == n_types - 1 else '')
                ax.grid(True, alpha=0.2)
                continue

            ax.scatter(res['days'], res['props'],
                       color=type_colors[t], s=40, zorder=3)

            x_line = np.array([res['days'].min(), res['days'].max()])
            y_line = res['intercept'] + res['slope'] * x_line
            ls = '-' if res['p'] < 0.05 else '--'
            ax.plot(x_line, y_line, color=type_colors[t],
                    linewidth=1.8, linestyle=ls, zorder=2)

            ax.annotate(f"slope={res['slope']:+.2f}%/day\n"
                        f"p={res['p']:.3f} {_sig_stars(res['p'])}",
                        xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=7, va='top',
                        color='black')

            ax.set_xticks(res['days'])
            ax.set_xticklabels(
                [s for s, d in zip(session_order, days_all)
                 if d in res['days']],
                rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('Day' if t == n_types - 1 else '')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


def run_session_trend_tests(raw_labels, session_labels, layer_labels,
                             session_order, layer_list, n_types, type_names,
                             n_early=3, n_late=3):
    """
    Per cell type × layer:
      1. Kruskal-Wallis across all sessions — omnibus "did anything change?"
      2. Early (first n_early sessions) vs late (last n_late sessions)
         Mann-Whitney U — net direction of change.

    Returns results[tname][layer] = {
        'kw_p': float, 'mw_p': float, 'direction': str,
        'early_mean': float, 'late_mean': float,
        'days': array, 'props': array }
    """
    print(f"\n── Session trend: KW (omnibus) + early({n_early}) vs "
          f"late({n_late}) MW-U ──")
    results = {}

    for t, tname in enumerate(type_names):
        print(f"\n  {tname}:")
        results[tname] = {}

        for layer in layer_list:
            # Collect per-session proportions
            props, valid_days = [], []
            days_all = np.array([int(s.replace('Day', ''))
                                 for s in session_order], dtype=float)
            for si, sess in enumerate(session_order):
                mask = (session_labels == sess) & (layer_labels == layer)
                n    = int(np.sum(mask))
                if n == 0:
                    continue
                props.append(np.mean(raw_labels[mask] == t) * 100)
                valid_days.append(days_all[si])

            if len(props) < 3:
                results[tname][layer] = None
                continue

            props      = np.array(props)
            valid_days = np.array(valid_days)

            # 1. KW across sessions (each session is one "group" = binary
            #    membership values for all cells in that session × layer)
            kw_groups = []
            for si, sess in enumerate(session_order):
                mask = (session_labels == sess) & (layer_labels == layer)
                if np.sum(mask) == 0:
                    continue
                kw_groups.append((raw_labels[mask] == t).astype(float))

            kw_p = np.nan
            if len(kw_groups) >= 2:
                try:
                    _, kw_p = kruskal(*kw_groups)
                except Exception:
                    pass

            # 2. Early vs late MW-U
            early_vals = props[:n_early]
            late_vals  = props[len(props) - n_late:]
            mw_p, direction = np.nan, 'n/a'
            if len(early_vals) >= 1 and len(late_vals) >= 1:
                try:
                    _, mw_p = mannwhitneyu(early_vals, late_vals,
                                           alternative='two-sided')
                    direction = ('increase' if np.mean(late_vals) > np.mean(early_vals)
                                 else 'decrease')
                except Exception:
                    pass

            print(f"    {layer}: KW p={kw_p:.4f} {_sig_stars(kw_p)}  |  "
                  f"early={np.mean(early_vals):.1f}% → late={np.mean(late_vals):.1f}%  "
                  f"MW p={mw_p:.4f} {_sig_stars(mw_p)}  [{direction}]")

            results[tname][layer] = {
                'kw_p':       kw_p,
                'mw_p':       mw_p,
                'direction':  direction,
                'early_mean': float(np.mean(early_vals)),
                'late_mean':  float(np.mean(late_vals)),
                'days':       valid_days,
                'props':      props,
            }

    return results


def plot_session_trend(trend_results, layer_list, session_order,
                        type_names, type_colors, animal_id,
                        n_early=2, n_late=2, output_path=None):
    """
    Grid: rows = cell types, columns = layers.
    Each panel: scatter of proportion per session with early/late shading,
    KW p + early-vs-late MW-U p annotated.
    """
    n_types  = len(type_names)
    n_layers = len(layer_list)

    fig, axes = plt.subplots(n_types, n_layers,
                             figsize=(4 * n_layers, 3.5 * n_types),
                             sharey='row', sharex='col')
    fig.suptitle(f'{animal_id} (tracked) — Session trend: KW omnibus + early vs late',
                 fontweight='bold')

    if n_types == 1:
        axes = axes[np.newaxis, :]
    if n_layers == 1:
        axes = axes[:, np.newaxis]

    for t, tname in enumerate(type_names):
        for li, layer in enumerate(layer_list):
            ax  = axes[t, li]
            res = trend_results[tname].get(layer)

            if t == 0:
                ax.set_title(layer,
                             color=LAYER_COLORS.get(layer, 'black'),
                             fontweight='bold')
            if li == 0:
                ax.set_ylabel(f'{tname}\n% observations',
                              color=type_colors[t], fontsize=8)

            if res is None:
                ax.text(0.5, 0.5, 'insufficient data',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=7, color='gray')
                ax.set_xlabel('Session' if t == n_types - 1 else '')
                ax.grid(True, alpha=0.2)
                continue

            days  = res['days']
            props = res['props']
            xpos  = np.arange(len(days))

            # Shade early / late
            if len(xpos) >= n_early:
                ax.axvspan(-0.5, n_early - 0.5, color='skyblue',
                           alpha=0.15, label='early')
            if len(xpos) >= n_late:
                ax.axvspan(len(xpos) - n_late - 0.5, len(xpos) - 0.5,
                           color='salmon', alpha=0.15, label='late')

            ax.scatter(xpos, props, color=type_colors[t], s=45, zorder=3)
            ax.plot(xpos, props, color=type_colors[t],
                    linewidth=1.2, alpha=0.6, zorder=2)

            # Early / late mean lines
            ax.hlines(res['early_mean'], -0.5, n_early - 0.5,
                      colors='steelblue', linewidths=1.5, linestyles='--')
            ax.hlines(res['late_mean'], len(xpos) - n_late - 0.5, len(xpos) - 0.5,
                      colors='firebrick', linewidths=1.5, linestyles='--')

            kw_str = f"KW p={res['kw_p']:.3f} {_sig_stars(res['kw_p'])}"
            mw_str = (f"early→late: {res['direction']}\n"
                      f"MW p={res['mw_p']:.3f} {_sig_stars(res['mw_p'])}")
            ax.annotate(f"{kw_str}\n{mw_str}",
                        xy=(0.03, 0.97), xycoords='axes fraction',
                        fontsize=6.5, va='top', color='black')

            ax.set_xticks(xpos)
            ax.set_xticklabels(session_order[:len(xpos)],
                               rotation=45, ha='right', fontsize=7)
            ax.set_xlabel('Session' if t == n_types - 1 else '')
            ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(output_path)}")
    return fig


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("PCA — TRACKED CELLS ONLY")
    print("=" * 65)

    animal_id = animal_id_from_path(ROI_TRACKING_FILE)
    prefix    = os.path.join(OUTPUT_DIR, f"{animal_id}_tracked")

    # ── 1. Load tracking ──────────────────────────────────────
    print("\n[1] Loading tracking matrix...")
    tracked_matrix, day_labels, session_dirs = load_tracking(ROI_TRACKING_FILE)
    tracked_matrix, day_labels, session_dirs = filter_to_analysis_days(
        tracked_matrix, day_labels, session_dirs, ANALYSIS_DAYS)

    print("\n[2] Finding preproc files...")
    if session_dirs:
        preproc_files = find_files_from_tracking(session_dirs, "*_preproc*.h5")
    else:
        preproc_files = find_preproc_files(ANIMAL_DIR)
    report_found_files("Preproc files", preproc_files, day_labels)

    # ── 2. Layer assignment ───────────────────────────────────
    print("\n[3] Assigning cells to layers (from SMI reference day)...")
    from load_tracked import find_files_from_tracking as _fft, find_smi_files
    if session_dirs:
        smi_files = _fft(session_dirs, "*_smi_results.h5")
    else:
        smi_files = find_smi_files(ANIMAL_DIR)
    cell_layers = assign_layers_from_smi(tracked_matrix, day_labels, smi_files,
                                          REFERENCE_DAY)
    for l in LAYER_ORDER:
        print(f"  {l}: {len(cell_layers.get(l, []))} tracked cells")

    # Landmark positions for profile plots and alignment
    landmark_positions = [25, 55, 85, 115]

    # ── 3. Aggregate profiles ─────────────────────────────────
    print("\n[4] Aggregating spatial profiles from tracked cells...")
    profiles, bin_centers, session_labels, layer_labels, tracked_ids, cell_types = \
        aggregate_tracked_profiles(
            tracked_matrix, day_labels, preproc_files, cell_layers,
            TRIM_START_CM, TRIM_END_CM, TARGET_N_BINS, SMOOTH_SIGMA,
            EXCLUDE_FIRST_BINS, EXCLUDE_LAST_BINS,
            landmark_positions=landmark_positions,
            smi_files=smi_files,
        )

    if len(profiles) == 0:
        print("No valid observations — check preproc files and tracking.")
        return

    session_order = [d for d in day_labels if d in session_labels]
    layer_list    = [l for l in LAYER_ORDER if np.any(layer_labels == l)]

    # Quick summary
    print(f"\n  Observations per session:")
    for sess in session_order:
        print(f"    {sess}: {np.sum(session_labels == sess)}")
    print(f"  Observations per layer:")
    for layer in layer_list:
        print(f"    {layer}: {np.sum(layer_labels == layer)}")

    # ── 4. PCA ────────────────────────────────────────────────
    print("\n[5] PCA...")
    pca, pc_scores = run_pca(profiles, N_PCA_COMPONENTS)

    # ── 5. K selection + clustering ───────────────────────────
    print("\n[6] K selection + clustering...")
    X = pc_scores[:, :N_CLUSTER_PCS]
    sil_scores, inertias, auto_k = select_optimal_k(X, K_RANGE)
    optimal_k = OVERRIDE_K if OVERRIDE_K is not None else auto_k
    print(f"  K: auto={auto_k}  used={optimal_k}")

    kmeans, raw_labels = fit_kmeans(X, optimal_k)
    n_types = optimal_k

    # ── 6. Semantic labels ────────────────────────────────────
    print("\n[7] Semantic labels...")
    type_names, type_colors, mean_profiles = assign_semantic_labels(
        raw_labels, profiles, n_types, bin_centers, landmark_positions)
    for t in range(n_types):
        n = int(np.sum(raw_labels == t))
        print(f"  [{t}] {type_names[t]:25s}  n={n} ({n/len(raw_labels)*100:.1f}%)")

    # ── 7. Figures ────────────────────────────────────────────
    print("\n[8] Generating figures...")

    plot_scree(pca, N_CLUSTER_PCS, animal_id,
               f"{prefix}_scree.png")

    plot_k_selection(sil_scores, inertias, K_RANGE, optimal_k, animal_id,
                     f"{prefix}_k_selection.png")

    plot_pc_scatter(pc_scores, raw_labels, type_names, type_colors, animal_id,
                    f"{prefix}_pc_scatter.png")

    plot_mean_profiles(mean_profiles, bin_centers, landmark_positions,
                       type_names, type_colors, animal_id,
                       f"{prefix}_mean_profiles.png")

    plot_global_proportions(raw_labels, layer_labels, layer_list,
                             type_names, type_colors, animal_id,
                             f"{prefix}_global_proportions.png")

    # ── 8. Layer × session proportion figures ─────────────────
    print("\n[9] Layer × session figures...")
    prop, counts = build_prop_matrix(
        raw_labels, session_labels, layer_labels,
        session_order, layer_list, n_types)

    plot_stacked_bars_by_layer(
        prop, counts, layer_list, session_order,
        type_names, type_colors, animal_id,
        f"{prefix}_stacked_bars_by_layer.png")

    plot_trajectories_by_layer(
        prop, layer_list, session_order,
        type_names, type_colors, animal_id,
        f"{prefix}_trajectories_by_layer.png")

    plot_trajectories_by_type(
        prop, layer_list, session_order,
        type_names, type_colors, animal_id,
        f"{prefix}_trajectories_by_type.png")

    # ── 9. Statistical tests ──────────────────────────────────
    print("\n[10] Statistical tests...")
    run_chi_square_tests(raw_labels, session_labels, layer_labels,
                          session_order, layer_list, n_types)
    run_kruskal_layer_tests(raw_labels, layer_labels, layer_list,
                             n_types, type_names)

    run_layer_posthoc(raw_labels, layer_labels, layer_list, n_types, type_names)
    exp_results = run_experience_effect(raw_labels, session_labels, layer_labels, session_order, layer_list, n_types, type_names)
    trend_results = run_session_trend_tests(raw_labels, session_labels, layer_labels, session_order, layer_list, n_types, type_names)

    plot_layer_posthoc(raw_labels, layer_labels, layer_list, n_types, type_names, type_colors, animal_id, f"{prefix}_layer_posthoc.png")
    plot_experience_regression(exp_results, layer_list, session_order, type_names, type_colors, animal_id, output_path=f"{prefix}_experience_regression.png")
    plot_session_trend(trend_results, layer_list, session_order, type_names, type_colors, animal_id, output_path=f"{prefix}_session_trend.png")

    plt.show()
    print(f"\nDone. Outputs in: {OUTPUT_DIR}")

    return dict(profiles=profiles, bin_centers=bin_centers,
                session_labels=session_labels, layer_labels=layer_labels,
                tracked_ids=tracked_ids, raw_labels=raw_labels,
                type_names=type_names, pca=pca, kmeans=kmeans,
                cell_types=cell_types)


if __name__ == "__main__":
    main()
