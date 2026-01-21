"""
PCA_DataAggregation.py
Aggregates spatial response data across sessions for a single animal.
Creates a structured HDF5 file ready for PCA analysis.

Approach B: Exclude onset/reward responders, keep all other reliable cells
(including those with peaks between landmark windows)

JSY, 12/2025
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
ANIMAL_ID = "JSY052"

# Base directory containing all session folders for this animal
BASE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging"

# Output directory for PCA data file
OUTPUT_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA"

# Landmark configuration (must match your landmark analysis)
LANDMARK_POSITIONS = [25, 55, 85, 115]  # cm
LANDMARK_WINDOWS_CONFIG = [
    {'before': 10, 'after': 10},  # L1 at 25cm: [10, 35]
    {'before': 20, 'after': 10},  # L2 at 55cm: [35, 65]
    {'before': 20, 'after': 10},  # L3 at 85cm: [65, 95]
    {'before': 20, 'after': 10},  # L4 at 115cm: [95, 125]
]

# Spatial trimming for PCA features (cm)
TRIM_START_CM = 10   # Start of analysis window (matches L1 window start)
TRIM_END_CM = 125    # End of analysis window (matches L4 window end)

# Filtering parameters
EXCLUDE_FIRST_BINS = 5  # Bins to exclude for onset filtering
EXCLUDE_LAST_BINS = 5   # Bins to exclude for reward filtering
SMOOTHING_SIGMA = 1.0   # Gaussian smoothing for profiles


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
                preproc_files = glob.glob(os.path.join(tseries_path, "*preproc.h5"))
                if preproc_files:
                    sessions.append((day_num, session_id, tseries_path))
    
    # Sort by day number
    sessions.sort(key=lambda x: x[0])
    
    return [(s[1], s[2]) for s in sessions]  # Return (session_id, path)


def identify_valid_cells_for_pca(normalized_spatial_activity, bin_centers,
                                  reliable_cells, exclude_first_bins=5, 
                                  exclude_last_bins=5, smoothing_sigma=1.0):
    """
    Identify cells valid for PCA analysis (Approach B):
    - Must be reliable (from preprocessing)
    - Must NOT have global peak in onset zone (first N bins)
    - Must NOT have global peak in reward zone (last N bins)
    - Cells peaking between landmarks ARE included
    
    Returns:
    --------
    valid_for_pca : boolean array
        Mask of cells valid for PCA
    peak_positions : array
        Peak position (cm) for each cell
    rejection_info : dict
        Information about rejected cells
    """
    n_cells = normalized_spatial_activity.shape[0]
    
    # Compute mean profiles
    mean_profiles = np.mean(normalized_spatial_activity, axis=1)
    
    # Apply smoothing
    if smoothing_sigma > 0:
        for cell in range(n_cells):
            mean_profiles[cell] = gaussian_filter1d(mean_profiles[cell], sigma=smoothing_sigma)
    
    # Calculate thresholds
    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)
    bin_spacing = np.mean(np.diff(bin_centers))
    
    onset_threshold_cm = min_pos + (exclude_first_bins * bin_spacing)
    end_threshold_cm = max_pos - (exclude_last_bins * bin_spacing)
    
    # Initialize
    valid_for_pca = np.zeros(n_cells, dtype=bool)
    peak_positions = np.zeros(n_cells)
    
    rejected_onset = []
    rejected_reward = []
    rejected_zero = []
    rejected_not_reliable = []
    
    for cell in range(n_cells):
        # Must be reliable first
        if not reliable_cells[cell]:
            rejected_not_reliable.append(cell)
            continue
        
        profile = mean_profiles[cell]
        global_peak_idx = np.argmax(profile)
        global_peak_pos = bin_centers[global_peak_idx]
        peak_positions[cell] = global_peak_pos
        
        # Check for zero activity
        if profile[global_peak_idx] == 0:
            rejected_zero.append(cell)
            continue
        
        # Check onset zone
        if global_peak_pos < onset_threshold_cm:
            rejected_onset.append(cell)
            continue
        
        # Check reward zone
        if global_peak_pos > end_threshold_cm:
            rejected_reward.append(cell)
            continue
        
        # Cell passes all filters
        valid_for_pca[cell] = True
    
    rejection_info = {
        'onset': np.array(rejected_onset),
        'reward': np.array(rejected_reward),
        'zero': np.array(rejected_zero),
        'not_reliable': np.array(rejected_not_reliable),
        'onset_threshold_cm': onset_threshold_cm,
        'end_threshold_cm': end_threshold_cm,
        'n_valid': np.sum(valid_for_pca),
        'n_total': n_cells
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
# MAIN AGGREGATION FUNCTION
# ============================================================================

def aggregate_pca_data(animal_id, base_dir, output_dir,
                       landmark_positions, landmark_windows_config,
                       trim_start_cm, trim_end_cm,
                       exclude_first_bins, exclude_last_bins,
                       smoothing_sigma):
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
            preproc_files = glob.glob(os.path.join(tseries_path, "*preproc.h5"))
            if not preproc_files:
                print(f"  WARNING: No preproc file found, skipping")
                continue
            
            preproc_data = files.read_h5(preproc_files[0])
            print(f"  Loaded: {os.path.basename(preproc_files[0])}")
            
            normalized_spatial_activity = preproc_data['norm_spatial_activity']
            bin_centers = preproc_data['bin_centers']
            reliable_cells = preproc_data['combined_reliable']
            
            n_cells, n_trials, n_bins = normalized_spatial_activity.shape
            print(f"  Data shape: {n_cells} cells, {n_trials} trials, {n_bins} bins")
            print(f"  Reliable cells: {np.sum(reliable_cells)}")
            
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
                    smoothing_sigma=smoothing_sigma
                )
            
            print(f"  Valid for PCA: {rejection_info['n_valid']} / {n_cells}")
            print(f"    Rejected (onset): {len(rejection_info['onset'])}")
            print(f"    Rejected (reward): {len(rejection_info['reward'])}")
            print(f"    Rejected (zero): {len(rejection_info['zero'])}")
            print(f"    Rejected (not reliable): {len(rejection_info['not_reliable'])}")
            
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

    # Z-score normalize profiles
    all_profiles_zscore = zscore_profiles(all_profiles)
    
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

        # Features
        features = f.create_group('features')
        features.create_dataset('spatial_profiles', data=all_profiles)
        features.create_dataset('spatial_profiles_zscore', data=all_profiles_zscore)
        
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
        smoothing_sigma=SMOOTHING_SIGMA
    )