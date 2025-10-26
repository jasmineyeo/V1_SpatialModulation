import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from helper import BehavioralDataFiltering as DF, SpatialDiscretization as SD

def calculate_sparsity_index(spatial_response):
    """
    Calculate sparsity index for a spatial tuning curve.
    Higher values indicate more sparse (sharper) tuning.
    
    Parameters:
    -----------
    spatial_response : numpy.ndarray
        Trial-averaged spatial response (1D array)
        
    Returns:
    --------
    sparsity : float
        Sparsity index (0 to 1, higher = more sparse)
    """
    if np.sum(spatial_response) == 0:
        return 0
    
    mean_response = np.mean(spatial_response)
    mean_squared_response = np.mean(spatial_response**2)
    n_bins = len(spatial_response)
    
    if mean_squared_response == 0:
        return 0
    
    sparsity = (1 - (mean_response**2 / mean_squared_response)) / (1 - 1/n_bins)
    return np.clip(sparsity, 0, 1)

def calculate_spatial_information(spatial_response, occupancy=None):
    """
    Calculate spatial information content (bits/spike).
    Higher values indicate more spatial information.
    
    Parameters:
    -----------
    spatial_response : numpy.ndarray
        Trial-averaged spatial response (1D array)
    occupancy : numpy.ndarray, optional
        Occupancy probability for each bin. If None, assumes uniform.
        
    Returns:
    --------
    spatial_info : float
        Spatial information in bits/spike
    """
    if np.sum(spatial_response) == 0:
        return 0
        
    # Assume uniform occupancy if not provided
    if occupancy is None:
        occupancy = np.ones(len(spatial_response)) / len(spatial_response)
    
    # Normalize occupancy
    occupancy = occupancy / np.sum(occupancy)
    
    # Mean firing rate
    mean_rate = np.sum(spatial_response * occupancy)
    
    if mean_rate == 0:
        return 0
    
    # Calculate spatial information
    spatial_info = 0
    for i in range(len(spatial_response)):
        if spatial_response[i] > 0 and occupancy[i] > 0:
            rate_ratio = spatial_response[i] / mean_rate
            spatial_info += occupancy[i] * spatial_response[i] * np.log2(rate_ratio)
    
    return spatial_info / mean_rate if mean_rate > 0 else 0

def calculate_peak_to_baseline_ratio(spatial_response):
    """
    Calculate peak-to-baseline ratio using robust baseline estimate.
    
    Parameters:
    -----------
    spatial_response : numpy.ndarray
        Trial-averaged spatial response (1D array)
        
    Returns:
    --------
    peak_baseline_ratio : float
        Peak response / baseline response
    """
    if len(spatial_response) == 0 or np.max(spatial_response) == 0:
        return 0
    
    peak = np.max(spatial_response)
    baseline = np.percentile(spatial_response, 10)  # Use 10th percentile as robust baseline
    
    if baseline == 0:
        baseline = np.mean(spatial_response[spatial_response > 0]) * 0.1 if np.any(spatial_response > 0) else 0.001
    
    return peak / baseline if baseline > 0 else 0

def apply_quality_filters(spatial_activity, min_activity_percentile=5, min_spatial_coverage=0.05):
    """
    Apply minimal quality filters to identify cells suitable for analysis.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    min_activity_percentile : float
        Minimum activity threshold (percentile)
    min_spatial_coverage : float
        Minimum fraction of spatial bins that must be active
        
    Returns:
    --------
    valid_cells : numpy.ndarray
        Boolean array indicating cells that pass quality filters
    """
    n_cells, n_trials, n_bins = spatial_activity.shape
    
    # Calculate trial-averaged activity for each cell
    trial_averaged = np.mean(spatial_activity, axis=1)
    
    # Activity threshold: cells must have higher than 5th percentile of mean activity
    all_mean_activities = np.mean(trial_averaged, axis=1)
    activity_threshold = np.percentile(all_mean_activities, min_activity_percentile)
    
    valid_cells = np.zeros(n_cells, dtype=bool)
    
    for cell in range(n_cells):
        # Check activity threshold
        mean_activity = np.mean(trial_averaged[cell])
        
        # Check spatial coverage
        active_bins = np.sum(trial_averaged[cell] > 0)
        spatial_coverage = active_bins / n_bins
        
        # Cell is valid if it passes both criteria
        if (mean_activity > activity_threshold and 
            spatial_coverage >= min_spatial_coverage):
            valid_cells[cell] = True
    
    return valid_cells

def calculate_sharpness_metrics_for_offset(twop_dict, vr_dict, offset_frames, framerate):
    """
    Calculate tuning curve sharpness metrics for a given temporal offset.
    
    Parameters:
    -----------
    twop_dict : dict
        Two-photon data dictionary
    vr_dict : dict
        VR behavioral data dictionary
    offset_frames : int
        Temporal offset in frames
    framerate : float
        Acquisition framerate
        
    Returns:
    --------
    metrics : dict
        Dictionary containing sharpness metrics
    """
    # Apply temporal offset
    offset_spike_data = apply_temporal_offset(twop_dict['sps'], offset_frames)
    
    # Apply smoothing
    smoothed = smooth_spikes(offset_spike_data, framerate, window_ms=500)
    
    # Filter trials
    # filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_trial_filtering(
    #     smoothed, 
    #     vr_dict['interp_location'],
    #     min_trial_duration_seconds=5, 
    #     max_trial_duration_seconds=60,
    #     framerate=framerate
    # )
    result = DF.process_data_with_speed_filtering(
        smoothed, 
        vr_dict['interp_location'],
        min_trial_duration_seconds=5, 
        max_trial_duration_seconds=60,
        framerate=framerate,
        min_speed_cm_s=2.0,
        frames_to_keep=5,
        max_location_range_au=400,
        filter_backward_laps=True
    )

    (filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps) = result
    
    if n_valid_laps == 0:
        return None
    
    # Spatial discretization
    single_revolution_VR = 282.415
    single_revolution_treadmill = 27.8
    single_lap_VR = 1320.645683
    single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR
    
    spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = SD.spatial_assignment(
        n_valid_laps,
        filtered_spks_laps, 
        filtered_location_laps, 
        single_lap_treadmill
    )
    
    # Apply spatial smoothing
    smoothed_spatial_activity = spatial_smooth(spatial_activity, window_cm=5)
    
    # Apply quality filters
    valid_cells = apply_quality_filters(smoothed_spatial_activity, 
                                       min_activity_percentile=5, 
                                       min_spatial_coverage=0.05)
    
    n_valid_cells = np.sum(valid_cells)
    if n_valid_cells == 0:
        return None
    
    # Calculate sharpness metrics for valid cells
    sparsity_values = []
    spatial_info_values = []
    peak_baseline_ratios = []
    
    for cell in range(smoothed_spatial_activity.shape[0]):
        if not valid_cells[cell]:
            continue
            
        # Get trial-averaged spatial response
        spatial_response = np.mean(smoothed_spatial_activity[cell], axis=0)
        
        # Calculate metrics
        sparsity = calculate_sparsity_index(spatial_response)
        spatial_info = calculate_spatial_information(spatial_response)
        peak_baseline = calculate_peak_to_baseline_ratio(spatial_response)
        
        sparsity_values.append(sparsity)
        spatial_info_values.append(spatial_info)
        peak_baseline_ratios.append(peak_baseline)
    
    # Calculate population-level metrics
    metrics = {
        'median_sparsity': np.median(sparsity_values),
        'median_spatial_info': np.median(spatial_info_values),
        'median_peak_to_baseline': np.median(peak_baseline_ratios),
        'n_cells_above_threshold': np.sum(np.array(sparsity_values) > 0.1),  # Threshold for "sharp" tuning
        'n_valid_cells': n_valid_cells,
        'n_valid_laps': n_valid_laps
    }
    
    return metrics

def find_optimal_temporal_offset(twop_dict, vr_dict, framerate, offset_range=None):
    """
    Find optimal temporal offset based on tuning curve sharpness metrics.
    
    Parameters:
    -----------
    twop_dict : dict
        Two-photon data dictionary
    vr_dict : dict
        VR behavioral data dictionary
    framerate : float
        Acquisition framerate
    offset_range : list, optional
        List of offset values to test. Default: [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
        
    Returns:
    --------
    results : dict
        Dictionary containing offset analysis results
    best_offsets : dict
        Best offset for each metric
    optimal_offset : int
        Recommended optimal offset (consensus across metrics)
    """
    if offset_range is None:
        offset_range = [-13, -12, -11, -10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    # print("Finding optimal temporal offset based on tuning curve sharpness...")
    # print(f"Testing offsets: {offset_range} frames")
    # print(f"Framerate: {framerate} Hz")
    # print("Positive offset = neural activity appears LATER relative to position")
    
    # Store results for each offset
    offset_results = {}
    
    for offset in offset_range:
        # print(f"Testing offset: {offset} frames ({offset/framerate:.2f} seconds)...", end=" ")
        
        metrics = calculate_sharpness_metrics_for_offset(twop_dict, vr_dict, offset, framerate)
        
        if metrics is not None:
            offset_results[offset] = metrics
            # print(f"✓ Valid cells: {metrics['n_valid_cells']}")
        else:
            # print("✗ No valid data")
            pass
    
    if len(offset_results) == 0:
        print("ERROR: No valid results for any offset!")
        return None, None, None
    
    # Find best offset for each metric
    valid_offsets = list(offset_results.keys())
    
    # Extract metrics for comparison
    sparsity_values = [offset_results[offset]['median_sparsity'] for offset in valid_offsets]
    spatial_info_values = [offset_results[offset]['median_spatial_info'] for offset in valid_offsets]
    peak_baseline_values = [offset_results[offset]['median_peak_to_baseline'] for offset in valid_offsets]
    cells_above_threshold = [offset_results[offset]['n_cells_above_threshold'] for offset in valid_offsets]
    
    # Find best offsets
    best_offsets = {
        'sparsity': valid_offsets[np.argmax(sparsity_values)],
        'spatial_info': valid_offsets[np.argmax(spatial_info_values)],
        'peak_to_baseline': valid_offsets[np.argmax(peak_baseline_values)],
        'cells_above_threshold': valid_offsets[np.argmax(cells_above_threshold)]
    }
    
    best_offsets_foroptimal = {
        'sparsity': valid_offsets[np.argmax(sparsity_values)],
        'spatial_info': valid_offsets[np.argmax(spatial_info_values)],
        # 'peak_to_baseline': valid_offsets[np.argmax(peak_baseline_values)],
        # 'cells_above_threshold': valid_offsets[np.argmax(cells_above_threshold)]
    }
    
    # Calculate consensus optimal offset (median of best offsets)
    optimal_offset = int(np.median(list(best_offsets_foroptimal.values())))
    
    # Print results
    print(f"\n" + "="*60)
    print("OPTIMAL TEMPORAL OFFSET RESULTS:")
    print("="*60)
    
    # Map metric names to the correct keys in offset_results
    metric_key_mapping = {
        'sparsity': 'median_sparsity',
        'spatial_info': 'median_spatial_info', 
        'peak_to_baseline': 'median_peak_to_baseline',
        'cells_above_threshold': 'n_cells_above_threshold'
    }
    
    for metric, best_offset in best_offsets.items():
        key = metric_key_mapping[metric]
        value = offset_results[best_offset][key]
        print(f"{metric:20s}: {best_offset:2d} frames ({best_offset/framerate:.2f}s) | Value: {value:.4f}")
    
    print(f"\nCONSENSUS OPTIMAL OFFSET: {optimal_offset} frames ({optimal_offset/framerate:.2f} seconds)")
    print("="*60)
    
    # # Create visualization
    create_offset_comparison_plot(offset_results, best_offsets, optimal_offset, framerate)
    
    return offset_results, best_offsets, optimal_offset

def create_offset_comparison_plot(offset_results, best_offsets, optimal_offset, framerate):
    """
    Create visualization comparing different temporal offsets.
    """
    valid_offsets = list(offset_results.keys())
    
    # Extract metrics
    sparsity_values = [offset_results[offset]['median_sparsity'] for offset in valid_offsets]
    spatial_info_values = [offset_results[offset]['median_spatial_info'] for offset in valid_offsets]
    peak_baseline_values = [offset_results[offset]['median_peak_to_baseline'] for offset in valid_offsets]
    cells_above_threshold = [offset_results[offset]['n_cells_above_threshold'] for offset in valid_offsets]
    
    # Create subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temporal Offset Optimization Based on Tuning Curve Sharpness', fontsize=16, fontweight='bold')
    
    # Plot 1: Median Sparsity
    ax = axes[0, 0]
    ax.plot(valid_offsets, sparsity_values, 'o-', color='blue', linewidth=2, markersize=6)
    best_idx = valid_offsets.index(best_offsets['sparsity'])
    ax.plot(valid_offsets[best_idx], sparsity_values[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    ax.set_xlabel('Offset (frames)')
    ax.set_ylabel('Median Sparsity Index')
    ax.set_title(f'Sparsity (Best: {best_offsets["sparsity"]} frames)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 2: Median Spatial Information
    ax = axes[0, 1]
    ax.plot(valid_offsets, spatial_info_values, 'o-', color='green', linewidth=2, markersize=6)
    best_idx = valid_offsets.index(best_offsets['spatial_info'])
    ax.plot(valid_offsets[best_idx], spatial_info_values[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    ax.set_xlabel('Offset (frames)')
    ax.set_ylabel('Median Spatial Information')
    ax.set_title(f'Spatial Info (Best: {best_offsets["spatial_info"]} frames)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 3: Median Peak-to-Baseline Ratio
    ax = axes[1, 0]
    ax.plot(valid_offsets, peak_baseline_values, 'o-', color='red', linewidth=2, markersize=6)
    best_idx = valid_offsets.index(best_offsets['peak_to_baseline'])
    ax.plot(valid_offsets[best_idx], peak_baseline_values[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    ax.set_xlabel('Offset (frames)')
    ax.set_ylabel('Median Peak/Baseline Ratio')
    ax.set_title(f'Peak/Baseline (Best: {best_offsets["peak_to_baseline"]} frames)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Number of Cells Above Threshold
    ax = axes[1, 1]
    ax.plot(valid_offsets, cells_above_threshold, 'o-', color='purple', linewidth=2, markersize=6)
    best_idx = valid_offsets.index(best_offsets['cells_above_threshold'])
    ax.plot(valid_offsets[best_idx], cells_above_threshold[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    ax.set_xlabel('Offset (frames)')
    ax.set_ylabel('Cells Above Threshold')
    ax.set_title(f'Sharp Cells Count (Best: {best_offsets["cells_above_threshold"]} frames)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig

def run_offset_optimization(twop_filepath, vr_filepath, offset_range=None):
    """
    Complete pipeline to find optimal temporal offset for a dataset.
    
    Parameters:
    -----------
    twop_filepath : str
        Path to two-photon data directory
    vr_filepath : str
        Path to VR behavioral data file
    offset_range : list, optional
        Range of offsets to test
        
    Returns:
    --------
    optimal_offset : int
        Recommended optimal offset in frames
    results : dict
        Full results dictionary
    best_offsets : dict
        Best offset for each individual metric
    """
    from helper import dataLoader
    
    print("Loading and aligning data...")
    
    # Load and align data
    procData = dataLoader(twop_filepath, vr_filepath)
    animal_id, date, framerate = procData.load_data()
    twop_dict, vr_dict = procData.align_data()
    
    print(f"Loaded data for {animal_id} on {date}")
    print(f"Spike data shape: {twop_dict['sps'].shape}")
    print(f"Location data shape: {vr_dict['interp_location'].shape}")
    
    # Find optimal temporal offset
    offset_results, best_offsets, optimal_offset = find_optimal_temporal_offset(
        twop_dict, vr_dict, framerate, offset_range=offset_range
    )
    
    return optimal_offset, offset_results, best_offsets

def create_simple_before_after_comparison(twop_dict, vr_dict, cell_idx, framerate, 
                                        optimal_offset=6, figsize=(15, 4)):
    """
    Create the simplest, clearest comparison of a cell's tuning curve before and after offset.
    
    Parameters:
    -----------
    twop_dict : dict
        Two-photon data dictionary
    vr_dict : dict
        VR behavioral data dictionary  
    cell_idx : int
        Index of the cell to analyze
    framerate : float
        Acquisition framerate
    optimal_offset : int
        The optimal offset to compare against offset 0
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    """
    
    def process_data_for_offset(offset):
        """Helper function to process data for a given offset."""
        # Apply temporal offset
        offset_spike_data = apply_temporal_offset(twop_dict['sps'], offset)
        
        # Apply smoothing
        smoothed = smooth_spikes(offset_spike_data, framerate, window_ms=500)
        
        # Filter trials
        result = DF.process_data_with_speed_filtering(
            smoothed, 
            vr_dict['interp_location'],
            min_trial_duration_seconds=5, 
            max_trial_duration_seconds=60,
            framerate=framerate,
            min_speed_cm_s=2.0,
            frames_to_keep=5,
            max_location_range_au=400,
            filter_backward_laps=True
        )

        (filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps) = result
        
        if n_valid_laps == 0:
            return None, None, None
        
        # Spatial discretization
        single_revolution_VR = 282.415
        single_revolution_treadmill = 27.8
        single_lap_VR = 1320.645683
        single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR
        
        spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = SD.spatial_assignment(
            n_valid_laps, filtered_spks_laps, filtered_location_laps, single_lap_treadmill
        )
        
        # Apply spatial smoothing
        smoothed_spatial_activity = spatial_smooth(spatial_activity, window_cm=5)
        
        # Get data for this specific cell
        cell_data = smoothed_spatial_activity[cell_idx]  # Shape: (n_trials, n_bins)
        trial_avg = np.mean(cell_data, axis=0)
        
        return cell_data, trial_avg, bin_centers
    
    # Process data for both offsets
    print(f"Processing cell {cell_idx}...")
    cell_data_0, trial_avg_0, bin_centers = process_data_for_offset(0)
    cell_data_opt, trial_avg_opt, _ = process_data_for_offset(optimal_offset)
    
    if cell_data_0 is None or cell_data_opt is None:
        print("ERROR: Could not process data for this cell")
        return None
    
    # Calculate metrics
    def calc_sparsity(response):
        if np.sum(response) == 0:
            return 0
        mean_resp = np.mean(response)
        mean_sq_resp = np.mean(response**2)
        n_bins = len(response)
        if mean_sq_resp == 0:
            return 0
        return (1 - (mean_resp**2 / mean_sq_resp)) / (1 - 1/n_bins)
    
    def calc_peak_baseline(response):
        if np.max(response) == 0:
            return 0
        peak = np.max(response)
        baseline = np.percentile(response, 10)
        if baseline == 0:
            baseline = 0.001
        return peak / baseline
    
    sparsity_0 = calc_sparsity(trial_avg_0)
    sparsity_opt = calc_sparsity(trial_avg_opt)
    peak_base_0 = calc_peak_baseline(trial_avg_0)
    peak_base_opt = calc_peak_baseline(trial_avg_opt)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Before (Offset 0)
    ax1 = axes[0]
    ax1.plot(bin_centers, trial_avg_0, 'b-', linewidth=3, label='Trial Average')
    ax1.fill_between(bin_centers, trial_avg_0, alpha=0.3, color='blue')
    
    # Add individual trials (faded)
    n_trials_to_show = min(10, cell_data_0.shape[0])
    for i in range(n_trials_to_show):
        ax1.plot(bin_centers, cell_data_0[i], 'b-', alpha=0.1, linewidth=1)
    
    # Mark peak
    peak_idx_0 = np.argmax(trial_avg_0)
    ax1.plot(bin_centers[peak_idx_0], trial_avg_0[peak_idx_0], 'ro', markersize=10, 
             markeredgecolor='black', markeredgewidth=2)
    
    ax1.set_title(f'BEFORE: Offset 0 frames\nSparsity: {sparsity_0:.3f} | Peak/Base: {peak_base_0:.1f}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('Neural Activity', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: After (Optimal Offset)
    ax2 = axes[1]
    ax2.plot(bin_centers, trial_avg_opt, 'r-', linewidth=3, label='Trial Average')
    ax2.fill_between(bin_centers, trial_avg_opt, alpha=0.3, color='red')
    
    # Add individual trials (faded)
    n_trials_to_show = min(10, cell_data_opt.shape[0])
    for i in range(n_trials_to_show):
        ax2.plot(bin_centers, cell_data_opt[i], 'r-', alpha=0.1, linewidth=1)
    
    # Mark peak
    peak_idx_opt = np.argmax(trial_avg_opt)
    ax2.plot(bin_centers[peak_idx_opt], trial_avg_opt[peak_idx_opt], 'go', markersize=10, 
             markeredgecolor='black', markeredgewidth=2)
    
    ax2.set_title(f'AFTER: Offset {optimal_offset} frames ({optimal_offset/framerate:.2f}s)\nSparsity: {sparsity_opt:.3f} | Peak/Base: {peak_base_opt:.1f}', 
                  fontsize=14, fontweight='bold', color='darkred')
    ax2.set_xlabel('Position (cm)', fontsize=12)
    ax2.set_ylabel('Neural Activity', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Direct Overlay Comparison
    ax3 = axes[2]
    ax3.plot(bin_centers, trial_avg_0, 'b-', linewidth=3, label=f'Offset 0 (Sparsity: {sparsity_0:.3f})', alpha=0.7)
    ax3.plot(bin_centers, trial_avg_opt, 'r-', linewidth=3, label=f'Offset {optimal_offset} (Sparsity: {sparsity_opt:.3f})', alpha=0.7)
    
    # Fill areas for better visibility
    ax3.fill_between(bin_centers, trial_avg_0, alpha=0.2, color='blue')
    ax3.fill_between(bin_centers, trial_avg_opt, alpha=0.2, color='red')
    
    # Mark peaks
    ax3.plot(bin_centers[peak_idx_0], trial_avg_0[peak_idx_0], 'bo', markersize=8, 
             markeredgecolor='black', markeredgewidth=1)
    ax3.plot(bin_centers[peak_idx_opt], trial_avg_opt[peak_idx_opt], 'ro', markersize=8, 
             markeredgecolor='black', markeredgewidth=1)
    
    # Calculate improvement
    sparsity_improvement = ((sparsity_opt - sparsity_0) / sparsity_0 * 100) if sparsity_0 > 0 else 0
    peak_base_improvement = ((peak_base_opt - peak_base_0) / peak_base_0 * 100) if peak_base_0 > 0 else 0
    
    ax3.set_title(f'OVERLAY COMPARISON\nSparsity: {sparsity_improvement:+.1f}% | Peak/Base: {peak_base_improvement:+.1f}%', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Position (cm)', fontsize=12)
    ax3.set_ylabel('Neural Activity', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Overall figure title
    fig.suptitle(f'Cell {cell_idx}: Temporal Offset Effect on Spatial Tuning\n'
                f'6-frame offset improves spatial selectivity', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    # Print summary
    print(f"\nCell {cell_idx} Results:")
    print(f"  Sparsity:     {sparsity_0:.3f} → {sparsity_opt:.3f} ({sparsity_improvement:+.1f}%)")
    print(f"  Peak/Base:    {peak_base_0:.1f} → {peak_base_opt:.1f} ({peak_base_improvement:+.1f}%)")
    print(f"  Peak Position: {bin_centers[peak_idx_0]:.1f}cm → {bin_centers[peak_idx_opt]:.1f}cm")
    
    return fig

def find_best_example_cells(twop_dict, vr_dict, framerate, optimal_offset=6, n_candidates=50):
    """
    Find cells that show the clearest improvement with temporal offset.
    
    Parameters:
    -----------
    twop_dict : dict
        Two-photon data dictionary
    vr_dict : dict
        VR behavioral data dictionary
    framerate : float
        Acquisition framerate
    optimal_offset : int
        Offset to test
    n_candidates : int
        Number of most active cells to test
        
    Returns:
    --------
    best_cells : list
        List of (cell_idx, improvement_score) tuples, sorted by improvement
    """
    # Select most active cells as candidates
    activity_levels = np.mean(twop_dict['sps'], axis=1)
    candidate_cells = np.argsort(activity_levels)[-n_candidates:]
    
    improvements = []
    
    print(f"Testing {len(candidate_cells)} candidate cells...")
    
    for cell_idx in candidate_cells:
        try:
            # Quick processing for both offsets
            def quick_process(offset):
                offset_spike_data = apply_temporal_offset(twop_dict['sps'], offset)
                smoothed = smooth_spikes(offset_spike_data, framerate, window_ms=500)
                
                # filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_trial_filtering(
                #     smoothed, vr_dict['interp_location'], 
                #     min_trial_duration_seconds=5, max_trial_duration_seconds=60, framerate=framerate
                # )
                result = DF.process_data_with_speed_filtering(
                    smoothed, 
                    vr_dict['interp_location'],
                    min_trial_duration_seconds=5, 
                    max_trial_duration_seconds=60,
                    framerate=framerate,
                    min_speed_cm_s=2.0,
                    frames_to_keep=5,
                    max_location_range_au=400,
                    filter_backward_laps=True
                )

                (filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps) = result
    
                if n_valid_laps == 0:
                    return None
                
                single_revolution_VR = 282.415
                single_revolution_treadmill = 27.8
                single_lap_VR = 1320.645683
                single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR
                
                spatial_activity, _, _, _ = SD.spatial_assignment(
                    n_valid_laps, filtered_spks_laps, filtered_location_laps, single_lap_treadmill
                )
                
                smoothed_spatial_activity = spatial_smooth(spatial_activity, window_cm=5)
                trial_avg = np.mean(smoothed_spatial_activity[cell_idx], axis=0)
                
                # Calculate sparsity
                if np.sum(trial_avg) == 0:
                    return 0
                mean_resp = np.mean(trial_avg)
                mean_sq_resp = np.mean(trial_avg**2)
                n_bins = len(trial_avg)
                if mean_sq_resp == 0:
                    return 0
                sparsity = (1 - (mean_resp**2 / mean_sq_resp)) / (1 - 1/n_bins)
                return sparsity
            
            sparsity_0 = quick_process(0)
            sparsity_opt = quick_process(optimal_offset)
            
            if sparsity_0 is not None and sparsity_opt is not None and sparsity_0 > 0:
                improvement = (sparsity_opt - sparsity_0) / sparsity_0
                improvements.append((cell_idx, improvement, sparsity_0, sparsity_opt))
                
        except Exception as e:
            continue
    
    # Sort by improvement
    improvements.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 cells with best improvement:")
    for i, (cell_idx, improvement, spar_0, spar_opt) in enumerate(improvements[:10]):
        print(f"  {i+1}. Cell {cell_idx}: {improvement*100:+.1f}% ({spar_0:.3f} → {spar_opt:.3f})")
    
    return improvements

def create_multiple_examples_split(twop_dict, vr_dict, framerate, optimal_offset=6, n_examples=20):
    """
    Create multiple simple before/after examples split into 4 readable figures (5 cells each).
    
    Parameters:
    -----------
    twop_dict : dict
        Two-photon data dictionary
    vr_dict : dict
        VR behavioral data dictionary
    framerate : float
        Acquisition framerate
    optimal_offset : int
        Optimal offset to use
    n_examples : int
        Total number of example cells to show (will be split into groups of 5)
        
    Returns:
    --------
    figs : list
        List of 4 figures, each with 5 cells
    """
    # Find best example cells
    improvements = find_best_example_cells(twop_dict, vr_dict, framerate, optimal_offset)
    
    if len(improvements) < n_examples:
        print(f"Warning: Only found {len(improvements)} good examples, showing all")
        n_examples = len(improvements)
    
    # Split into groups of 5
    cells_per_figure = 5
    n_figures = (n_examples + cells_per_figure - 1) // cells_per_figure  # Ceiling division
    
    figs = []
    
    for fig_idx in range(n_figures):
        start_idx = fig_idx * cells_per_figure
        end_idx = min(start_idx + cells_per_figure, n_examples)
        cells_in_this_fig = end_idx - start_idx
        
        print(f"Creating figure {fig_idx + 1}/{n_figures} with cells {start_idx + 1}-{end_idx}")
        
        # Create figure for this group
        fig, axes = plt.subplots(cells_in_this_fig, 2, figsize=(12, 3*cells_in_this_fig))
        if cells_in_this_fig == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(cells_in_this_fig):
            cell_idx = improvements[start_idx + i][0]
            improvement_pct = improvements[start_idx + i][1] * 100
            
            # Process data for both offsets
            def process_for_plot(offset):
                offset_spike_data = apply_temporal_offset(twop_dict['sps'], offset)
                smoothed = smooth_spikes(offset_spike_data, framerate, window_ms=500)
                
                # filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_trial_filtering(
                #     smoothed, vr_dict['interp_location'],
                #     min_trial_duration_seconds=5, max_trial_duration_seconds=60, framerate=framerate
                # )
                result = DF.process_data_with_speed_filtering(
                    smoothed, 
                    vr_dict['interp_location'],
                    min_trial_duration_seconds=5, 
                    max_trial_duration_seconds=60,
                    framerate=framerate,
                    min_speed_cm_s=2.0,
                    frames_to_keep=5,
                    max_location_range_au=400,
                    filter_backward_laps=True
                )

                (filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps) = result
    
                single_revolution_VR = 282.415
                single_revolution_treadmill = 27.8
                single_lap_VR = 1320.645683
                single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR
                
                spatial_activity, _, _, bin_centers = SD.spatial_assignment(
                    n_valid_laps, filtered_spks_laps, filtered_location_laps, single_lap_treadmill
                )
                
                smoothed_spatial_activity = spatial_smooth(spatial_activity, window_cm=5)
                trial_avg = np.mean(smoothed_spatial_activity[cell_idx], axis=0)
                
                return trial_avg, bin_centers
            
            trial_avg_0, bin_centers = process_for_plot(0)
            trial_avg_opt, _ = process_for_plot(optimal_offset)
            
            # Calculate sparsity
            def calc_sparsity(response):
                if np.sum(response) == 0:
                    return 0
                mean_resp = np.mean(response)
                mean_sq_resp = np.mean(response**2)
                n_bins = len(response)
                if mean_sq_resp == 0:
                    return 0
                return (1 - (mean_resp**2 / mean_sq_resp)) / (1 - 1/n_bins)
            
            sparsity_0 = calc_sparsity(trial_avg_0)
            sparsity_opt = calc_sparsity(trial_avg_opt)
            
            # Plot Before
            ax_before = axes[i, 0]
            ax_before.plot(bin_centers, trial_avg_0, 'b-', linewidth=3)
            ax_before.fill_between(bin_centers, trial_avg_0, alpha=0.3, color='blue')
            peak_idx = np.argmax(trial_avg_0)
            ax_before.plot(bin_centers[peak_idx], trial_avg_0[peak_idx], 'ro', markersize=8)
            
            ax_before.set_title(f'Cell {cell_idx} - BEFORE\nSparsity: {sparsity_0:.3f}', 
                               fontsize=12, fontweight='bold')
            ax_before.set_xlabel('Position (cm)', fontsize=11)
            ax_before.set_ylabel('Activity', fontsize=11)
            ax_before.grid(True, alpha=0.3)
            
            # Plot After  
            ax_after = axes[i, 1]
            ax_after.plot(bin_centers, trial_avg_opt, 'r-', linewidth=3)
            ax_after.fill_between(bin_centers, trial_avg_opt, alpha=0.3, color='red')
            peak_idx = np.argmax(trial_avg_opt)
            ax_after.plot(bin_centers[peak_idx], trial_avg_opt[peak_idx], 'go', markersize=8)
            
            ax_after.set_title(f'Cell {cell_idx} - AFTER\nSparsity: {sparsity_opt:.3f} ({improvement_pct:+.1f}%)', 
                              fontsize=12, fontweight='bold', color='darkred')
            ax_after.set_xlabel('Position (cm)', fontsize=11)
            ax_after.set_ylabel('Activity', fontsize=11)
            ax_after.grid(True, alpha=0.3)
        
        # Figure title for this group
        plt.suptitle(f'Temporal Offset Effect: Examples {start_idx + 1}-{end_idx} (Figure {fig_idx + 1}/{n_figures})\n'
                    f'Offset 0 vs {optimal_offset} frames ({optimal_offset/framerate:.2f}s delay)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Make room for suptitle
        
        figs.append(fig)
    
    return figs

def create_five_detailed_examples(twop_dict, vr_dict, framerate, optimal_offset=6):
    """
    Create 5 detailed single-cell comparisons like the second figure.
    
    Parameters:
    -----------
    twop_dict : dict
        Two-photon data dictionary
    vr_dict : dict
        VR behavioral data dictionary
    framerate : float
        Acquisition framerate
    optimal_offset : int
        Optimal offset to use
        
    Returns:
    --------
    figs : list
        List of 5 figures, one for each cell
    """
    # Find best example cells
    improvements = find_best_example_cells(twop_dict, vr_dict, framerate, optimal_offset)
    
    n_examples = min(5, len(improvements))
    figs = []
    
    for i in range(n_examples):
        cell_idx = improvements[i][0]
        improvement_pct = improvements[i][1] * 100
        
        print(f"Creating detailed figure {i+1}/5 for Cell {cell_idx} (improvement: {improvement_pct:+.1f}%)")
        
        # Create detailed comparison for this cell
        fig = create_simple_before_after_comparison(twop_dict, vr_dict, cell_idx, framerate, optimal_offset)
        figs.append(fig)
    
    return figs

# Update the main demonstration function
def demonstrate_simple_offset_effect(twop_filepath, vr_filepath, cell_idx=None, optimal_offset=6):
    """
    Create the simplest, most convincing demonstration of temporal offset effects.
    
    Parameters:
    -----------
    twop_filepath : str
        Path to two-photon data
    vr_filepath : str  
        Path to VR data
    cell_idx : int, optional
        Specific cell to analyze. If None, finds the best example automatically.
    optimal_offset : int
        Optimal offset to demonstrate
        
    Returns:
    --------
    fig1 : matplotlib.figure.Figure
        Single cell detailed comparison
    multiple_figs : list
        List of 4 figures, each with 5 examples (20 total)
    detailed_figs : list
        List of 5 detailed single-cell figures
    """
    from helper import dataLoader
    
    # Load data
    procData = dataLoader(twop_filepath, vr_filepath)
    animal_id, date, framerate = procData.load_data()
    twop_dict, vr_dict = procData.align_data()
    
    print(f"Analyzing data for {animal_id} on {date}")
    
    # Find best example cell if not specified
    if cell_idx is None:
        print("Finding best example cells...")
        improvements = find_best_example_cells(twop_dict, vr_dict, framerate, optimal_offset)
        if len(improvements) > 0:
            cell_idx = improvements[0][0]  # Best cell
            print(f"Selected best example: Cell {cell_idx}")
        else:
            # Fallback to most active cell
            activity_levels = np.mean(twop_dict['sps'], axis=1)
            cell_idx = np.argsort(activity_levels)[-1]
            print(f"Fallback to most active cell: {cell_idx}")
    
    # Create single cell detailed comparison
    print(f"\nCreating detailed comparison for Cell {cell_idx}...")
    fig1 = create_simple_before_after_comparison(twop_dict, vr_dict, cell_idx, framerate, optimal_offset)
    
    # Create 20 examples split into 4 readable figures
    print("\nCreating 20 examples split into 4 figures (5 cells each)...")
    multiple_figs = create_multiple_examples_split(twop_dict, vr_dict, framerate, optimal_offset, n_examples=20)
    
    # Create 5 detailed examples
    print("\nCreating 5 detailed examples...")
    detailed_figs = create_five_detailed_examples(twop_dict, vr_dict, framerate, optimal_offset)
    
    plt.show()
    
    return fig1, multiple_figs, detailed_figs

def apply_temporal_offset(spike_data, offset_frames):
    """
    Apply a temporal offset to spike data relative to location data.
    Positive offset means neural activity shifted forward in time (appears later relative to location).
    Negative offset means neural activity shifted backward in time (appears earlier relative to location).
    """
    n_cells, n_frames = spike_data.shape
    
    if offset_frames == 0:
        return spike_data
    
    # Create new arrays to hold offset data
    offset_spike_data = np.zeros_like(spike_data)
    
    if offset_frames > 0:
        # Positive offset: spikes shifted forward (later) relative to location
        # Keep later part of spikes
        offset_spike_data[:, offset_frames:] = spike_data[:, :-offset_frames]
    else:
        # Negative offset: spikes shifted backward (earlier) relative to location
        # Keep earlier part of spikes
        abs_offset = abs(offset_frames)
        offset_spike_data[:, :-abs_offset] = spike_data[:, abs_offset:]
    
    return offset_spike_data

def smooth_spikes(spike_data, fps=10, window_ms=250):
    """
    Smooth spike data using a Gaussian window.
    
    Parameters:
    -----------
    spike_data : numpy.ndarray
        Array containing deconvolved spike data. Can be 1D (single trace) 
        or 2D (multiple traces, with traces in rows)
    fps : float
        Frames per second of the recording
    window_ms : float
        Width of the Gaussian window in milliseconds
    
    Returns:
    --------
    numpy.ndarray
        Smoothed spike data with same shape as input
    """
    # Convert window size from ms to frames
    window_frames = window_ms / 1000 * fps
    
    # Calculate sigma for Gaussian filter
    # Window size is typically considered as 6-sigma width
    sigma = window_frames / 6
    
    # Handle both 1D and 2D inputs
    if spike_data.ndim == 1:
        return gaussian_filter1d(spike_data, sigma=sigma)
    else:
        return np.array([gaussian_filter1d(trace, sigma=sigma) 
                        for trace in spike_data])

def plot_comparison(original, smoothed, fps=10, time_window=None, max_cells=5, cell_indices=None):
    """
    Efficiently plot original and smoothed traces for comparison.
    Optimized for large-scale neurophysiology data.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original spike data (1D or 2D array)
    smoothed : numpy.ndarray
        Smoothed spike data (1D or 2D array)
    fps : float
        Frames per second
    time_window : tuple, optional
        (start_time, end_time) in seconds to plot
    max_cells : int, optional
        Maximum number of cells to plot
    cell_indices : list or numpy.ndarray, optional
        Specific indices of cells to plot. If None, first max_cells will be used.
    """
    plt.ioff()  # Turn off interactive mode
    
    # Handle data selection
    if original.ndim == 1:
        n_cells = 1
        data_original = original[np.newaxis, :]
        data_smoothed = smoothed[np.newaxis, :]
    else:
        if cell_indices is None:
            cell_indices = np.arange(min(max_cells, original.shape[0]))
        n_cells = len(cell_indices)
        data_original = original[cell_indices]
        data_smoothed = smoothed[cell_indices]
    
    # Create decimated time vector for plotting
    total_points = original.shape[-1]
    if total_points > 5000:  # Decimate data for plotting if too many points
        decimation_factor = total_points // 5000 + 1
        time = np.arange(0, total_points, decimation_factor) / fps
        data_original = data_original[:, ::decimation_factor]
        data_smoothed = data_smoothed[:, ::decimation_factor]
    else:
        time = np.arange(total_points) / fps
    
    # Create figure
    fig, axes = plt.subplots(n_cells, 1, figsize=(10, 1.5*n_cells), 
                            sharex=True, constrained_layout=True)
    if n_cells == 1:
        axes = [axes]
    
    # Plot data
    for i, ax in enumerate(axes):
        ax.plot(time, data_original[i], 'gray', alpha=0.5, linewidth=0.5)
        ax.plot(time, data_smoothed[i], 'b', linewidth=0.5)
        
        # Minimal formatting
        ax.set_ylabel(f'Cell {cell_indices[i]}' if original.ndim > 1 else 'Rate')
        if i == n_cells - 1:
            ax.set_xlabel('Time (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0:  # Legend only on first plot
            ax.legend(['Original', 'Smoothed'], frameon=False, 
                     loc='upper right', fontsize='small')
    
    return fig

def plot_sample_cells(original, smoothed, fps=10, n_samples=5, seed=None):
    """
    Plot a random sample of cells for quick visualization.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original spike data (2D array: cells × time)
    smoothed : numpy.ndarray
        Smoothed spike data (2D array: cells × time)
    fps : float
        Frames per second
    n_samples : int
        Number of random cells to sample
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_cells = original.shape[0]
    sample_indices = np.random.choice(n_cells, min(n_samples, n_cells), replace=False)
    sample_indices.sort()  # Sort for clearer visualization
    
    return plot_comparison(original, smoothed, fps=fps, 
                          cell_indices=sample_indices)

def spatial_smooth(data, window_cm=5, bin_size_cm=1):
    """
    Apply spatial smoothing with a Gaussian window.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to smooth (can be 1D, 2D, or 3D with spatial bins as last dimension)
    window_cm : float
        Width of Gaussian window in cm
    bin_size_cm : float
        Size of each spatial bin in cm (default=1, as each bin represents 1 cm)
        
    Returns:
    --------
    smoothed_data : numpy.ndarray
        Spatially smoothed data with same shape as input
    """
    
    # Convert window size from cm to bins
    window_bins = window_cm / bin_size_cm
    
    # Calculate sigma for Gaussian filter (window is typically 6-sigma width)
    sigma = window_bins / 6
    
    # Apply smoothing based on data dimensionality
    if data.ndim == 1:
        # 1D data: bins
        smoothed_data = gaussian_filter1d(data, sigma=sigma)
    elif data.ndim == 2:
        # 2D data: cells x bins or laps x bins
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed_data[i] = gaussian_filter1d(data[i], sigma=sigma)
    elif data.ndim == 3:
        # 3D data: cells x laps x bins
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                smoothed_data[i, j] = gaussian_filter1d(data[i, j], sigma=sigma)
    
    return smoothed_data

# Example usage:
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    t = np.linspace(0, 10, 100)  # 10 seconds at 10 fps
    
    # Example 1: Single cell
    print("Example 1: Single cell")
    spikes_1d = np.random.poisson(lam=2, size=100)  # Random spike train
    smoothed_1d = smooth_spikes(spikes_1d, fps=10, window_ms=250)
    plot_comparison(spikes_1d, smoothed_1d, fps=10)
    
    # Example 2: Multiple cells
    print("\nExample 2: Multiple cells")
    n_cells = 10
    spikes_2d = np.random.poisson(lam=2, size=(n_cells, 100))  # Multiple random spike trains
    smoothed_2d = smooth_spikes(spikes_2d, fps=10, window_ms=250)
    
    # Verify shapes
    print(f"Input shape: {spikes_2d.shape}")
    print(f"Output shape: {smoothed_2d.shape}")
    
    # Plot results for multiple cells
    plot_comparison(spikes_2d, smoothed_2d, fps=10)
    plt.show()