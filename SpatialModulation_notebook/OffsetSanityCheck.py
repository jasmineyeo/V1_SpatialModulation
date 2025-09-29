"""
OffsetSanityCheck.py
A script checking timing between 2p and VR
Using 2photon data with Unity rendering an illuminating sphere object
Input: average fluourescent data in a .mat format

JSY, 09/02/25
"""
        
import sys

parent_dir = r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation"
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
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
    offset_spike_data = apply_temporal_offset(twop_dict['fluorescence'], offset_frames)
    
    # # Apply smoothing
    # smoothed = smooth_spikes(offset_spike_data, framerate, window_ms=500)
    
    # Filter trials
    filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_trial_filtering(
        offset_spike_data, 
        vr_dict['interp_location'],
        min_trial_duration_seconds=1, 
        max_trial_duration_seconds=120,
        framerate=framerate
    )
    
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
    
    ## qulity filter disabled as twop_data['fluoresence'] is an average array
    # # Apply quality filters
    # valid_cells = apply_quality_filters(smoothed_spatial_activity, 
    #                                    min_activity_percentile=5, 
    #                                    min_spatial_coverage=0.05)
    
    # n_valid_cells = np.sum(valid_cells)
    # if n_valid_cells == 0:
    #     return None
    
    # Calculate sharpness metrics for valid cells
    sparsity_values = []
    spatial_info_values = []
    peak_baseline_ratios = []
    
    for cell in range(smoothed_spatial_activity.shape[0]):
        # if not valid_cells[cell]:
        #     continue
            
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
        # 'n_valid_cells': n_valid_cells,
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
        offset_range = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    
    print("Finding optimal temporal offset based on tuning curve sharpness...")
    print(f"Testing offsets: {offset_range} frames")
    print(f"Framerate: {framerate} Hz")
    print("Positive offset = neural activity appears LATER relative to position")
    
    # Store results for each offset
    offset_results = {}
    
    for offset in offset_range:
        print(f"Testing offset: {offset} frames ({offset/framerate:.2f} seconds)...", end=" ")
        
        metrics = calculate_sharpness_metrics_for_offset(twop_dict, vr_dict, offset, framerate)
        
        if metrics is not None:
            offset_results[offset] = metrics
            # print(f"✓ Valid cells: {metrics['n_valid_cells']}")
        else:
            print("✗ No valid data")
    
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
    
    # Calculate consensus optimal offset (median of best offsets)
    optimal_offset = int(np.median(list(best_offsets.values())))
    
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Temporal Offset Optimization Based on Tuning Curve Sharpness', fontsize=16, fontweight='bold')
    
    # Plot 1: Median Sparsity
    ax = axes[0]
    ax.plot(valid_offsets, sparsity_values, 'o-', color='blue', linewidth=2, markersize=6)
    best_idx = valid_offsets.index(best_offsets['sparsity'])
    ax.plot(valid_offsets[best_idx], sparsity_values[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    ax.set_xlabel('Offset (frames)')
    ax.set_ylabel('Median Sparsity Index')
    ax.set_title(f'Sparsity (Best: {best_offsets["sparsity"]} frames)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # # Plot 2: Median Spatial Information
    # ax = axes[0, 1]
    # ax.plot(valid_offsets, spatial_info_values, 'o-', color='green', linewidth=2, markersize=6)
    # best_idx = valid_offsets.index(best_offsets['spatial_info'])
    # ax.plot(valid_offsets[best_idx], spatial_info_values[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    # ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    # ax.set_xlabel('Offset (frames)')
    # ax.set_ylabel('Median Spatial Information')
    # ax.set_title(f'Spatial Info (Best: {best_offsets["spatial_info"]} frames)')
    # ax.grid(True, alpha=0.3)
    # ax.legend()
    
    # Plot 3: Median Peak-to-Baseline Ratio
    ax = axes[1]
    ax.plot(valid_offsets, peak_baseline_values, 'o-', color='red', linewidth=2, markersize=6)
    best_idx = valid_offsets.index(best_offsets['peak_to_baseline'])
    ax.plot(valid_offsets[best_idx], peak_baseline_values[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    ax.set_xlabel('Offset (frames)')
    ax.set_ylabel('Median Peak/Baseline Ratio')
    ax.set_title(f'Peak/Baseline (Best: {best_offsets["peak_to_baseline"]} frames)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # # Plot 4: Number of Cells Above Threshold
    # ax = axes[1, 1]
    # ax.plot(valid_offsets, cells_above_threshold, 'o-', color='purple', linewidth=2, markersize=6)
    # best_idx = valid_offsets.index(best_offsets['cells_above_threshold'])
    # ax.plot(valid_offsets[best_idx], cells_above_threshold[best_idx], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
    # ax.axvline(optimal_offset, color='red', linestyle='--', alpha=0.7, label=f'Consensus: {optimal_offset}')
    # ax.set_xlabel('Offset (frames)')
    # ax.set_ylabel('Cells Above Threshold')
    # ax.set_title(f'Sharp Cells Count (Best: {best_offsets["cells_above_threshold"]} frames)')
    # ax.grid(True, alpha=0.3)
    # ax.legend()
    
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

    # define a twoP_filename which is the variable after the very last \ from the twop_path
    twop_path = r"F:\2P\spmod\250829_JSY_VR_timingcheck\TSeries-08292025-1424.1.5gain-001"
    behav_path = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08292025_03-44-23.txt"

    twoP_filename = os.path.basename(twop_path)
    behav_filename = os.path.basename(behav_path)

    # Extract animal ID and date from the VR_log_filename
    match = re.match(r"VRlog_(JSY\d+)_(\d{8})_\d{2}-\d{2}-\d{2}\.txt", behav_filename)
    if match:
        animal_id = match.group(1)
        date = match.group(2)
    else:
        print("Filename format does not match the expected pattern.")

    # Initialize dictionaries to store raw data
    twoP_data = {}
    VR_data = {}

    # Load .mat file
    twoP_data['fluorescence'] = sio.loadmat(os.path.join(twop_path, "gain1.5_averageF.mat"))['frame_F']
    numFrames = np.size(twoP_data['fluorescence'], 1)
    numCells = len(twoP_data['fluorescence'])

    xml_path = os.path.join(twop_path, f"{twoP_filename}.xml")
    xml_dict = read_xml(xml_path)
    t0 = xml_dict["t0"]
    abs_time = xml_dict["abs_time"]
    rel_time = xml_dict["rel_time"]
    framerate = 1/rel_time[1]

    twopT = np.zeros(np.size(abs_time, 0) - 1, dtype=datetime.datetime)
    for rep, t in enumerate(abs_time[:-1]):
        twopT[rep] = t0 + datetime.timedelta(seconds=t)

    twopT_float = time2float(twopT)
    twoP_data['AbsoluteT'] = twopT


    # Load VRlog
    rawVR_data = []
    with open(behav_path, "r") as file:
        lines = file.readlines()
        for line in lines[3:]:
            rawVR_data.append(line.strip().split("\t"))

    # Extract VR data
    VR_data['absoluteT'] = np.array([line[0] for line in rawVR_data])
    VR_data['elapsedT'] = np.array([float(line[1]) for line in rawVR_data])
    VR_data['event'] = np.array([line[2] for line in rawVR_data])
    VR_data['location'] = np.array([float(line[3]) for line in rawVR_data])

    # for any VR_data['location'] that is less than 0, set it to 0
    VR_data['location'][VR_data['location'] > 385] = 385

    # Find the index of the first 's' in VR_data['event']
    start_index = np.where(VR_data['event'] == 's')[0][0]

    # Erase all elements before the start_index in all VR_data
    for key in VR_data.keys():
        VR_data[key] = VR_data[key][start_index:]


    # align data

    # Define absolute_t0 as the first element of VR_data['absoluteT'] -- with "s" for event type, which is the timestamp for 2p input trigger
    VR_absolute_t = np.array([datetime.datetime.strptime(t, '%H.%M.%S.%f') for t in VR_data['absoluteT'][0:]])

    # Calculate relative_t (time elapsed from absolute_t0)
    VR_relative_t = np.array([(t - VR_absolute_t[0]).total_seconds() for t in VR_absolute_t])

    # Add twoP_data['AbsoluteT'][0] to each timedelta object to get vrT
    VR_relative_t_timedelta = np.array([datetime.timedelta(seconds=t) for t in VR_relative_t])
    Aligned_Abs_vrT = twoP_data['AbsoluteT'][0] + VR_relative_t_timedelta

    # Find the closest value in Aligned_Abs_vrT that is greater than twoP_data['AbsoluteT'][-1]
    closest_value = Aligned_Abs_vrT[Aligned_Abs_vrT > twoP_data['AbsoluteT'][-1]][0]
    closest_index = np.where(Aligned_Abs_vrT == closest_value)[0][0]

    new_VR_data = {}
    new_VR_data['AbsoluteT'] = np.array(Aligned_Abs_vrT)[:closest_index]
    new_VR_data['RelativeT'] = VR_relative_t[:closest_index]
    new_VR_data['event'] = VR_data['event'][:closest_index]
    new_VR_data['location'] = VR_data['location'][:closest_index]

    # Calculate relative time points for VR_data and twoP_data
    twop_relativeT = twoP_data['AbsoluteT'] - twoP_data['AbsoluteT'][0]

    # Convert to seconds
    twop_relativeT = np.array([t.total_seconds() for t in twop_relativeT])
    twoP_data['RelativeT'] = twop_relativeT

    # Interpolate the location at twoP_data['RelativeT'] from new_VR_data['location'] at new_VR_data['RelativeT']
    interpolated_location = np.interp(twoP_data['RelativeT'], 
                                    new_VR_data['RelativeT'], 
                                    new_VR_data['location'])
    new_VR_data['interp_location'] = interpolated_location
    print(f"size of interpolated_location is {interpolated_location.shape}")
    print(f"size of new_VR_data['location'] is {new_VR_data['location'].shape}")

    # Find optimal temporal offset
    offset_results, best_offsets, optimal_offset = find_optimal_temporal_offset(
        twoP_data, new_VR_data, framerate, offset_range=offset_range
    )
    
    return optimal_offset, offset_results, best_offsets

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

import os
import re
import scipy.io as sio
import numpy as np
import datetime
from helper import TwoP, read_xml, time2float, SpikeSmoothing


# load data

# define a twoP_filename which is the variable after the very last \ from the twop_path
twop_path = r"F:\2P\spmod\250829_JSY_VR_timingcheck\TSeries-08292025-1424.1.5gain-001"
behav_path = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08292025_03-44-23.txt"

twoP_filename = os.path.basename(twop_path)
behav_filename = os.path.basename(behav_path)

# Extract animal ID and date from the VR_log_filename
match = re.match(r"VRlog_(JSY\d+)_(\d{8})_\d{2}-\d{2}-\d{2}\.txt", behav_filename)
if match:
    animal_id = match.group(1)
    date = match.group(2)
else:
    print("Filename format does not match the expected pattern.")

# Initialize dictionaries to store raw data
twoP_data = {}
VR_data = {}

# Load .mat file
twoP_data['fluorescence'] = sio.loadmat(os.path.join(twop_path, "gain1.5_averageF.mat"))['frame_F']
numFrames = np.size(twoP_data['fluorescence'], 1)
numCells = len(twoP_data['fluorescence'])

xml_path = os.path.join(twop_path, f"{twoP_filename}.xml")
xml_dict = read_xml(xml_path)
t0 = xml_dict["t0"]
abs_time = xml_dict["abs_time"]
rel_time = xml_dict["rel_time"]
framerate = 1/rel_time[1]

twopT = np.zeros(np.size(abs_time, 0) - 1, dtype=datetime.datetime)
for rep, t in enumerate(abs_time[:-1]):
    twopT[rep] = t0 + datetime.timedelta(seconds=t)

twopT_float = time2float(twopT)
twoP_data['AbsoluteT'] = twopT


# Load VRlog
rawVR_data = []
with open(behav_path, "r") as file:
    lines = file.readlines()
    for line in lines[3:]:
        rawVR_data.append(line.strip().split("\t"))

# Extract VR data
VR_data['absoluteT'] = np.array([line[0] for line in rawVR_data])
VR_data['elapsedT'] = np.array([float(line[1]) for line in rawVR_data])
VR_data['event'] = np.array([line[2] for line in rawVR_data])
VR_data['location'] = np.array([float(line[3]) for line in rawVR_data])

# for any VR_data['location'] that is less than 0, set it to 0
VR_data['location'][VR_data['location'] > 400] = 400

# Find the index of the first 's' in VR_data['event']
start_index = np.where(VR_data['event'] == 's')[0][0]

# Erase all elements before the start_index in all VR_data
for key in VR_data.keys():
    VR_data[key] = VR_data[key][start_index:]


# align data

# Define absolute_t0 as the first element of VR_data['absoluteT'] -- with "s" for event type, which is the timestamp for 2p input trigger
VR_absolute_t = np.array([datetime.datetime.strptime(t, '%H.%M.%S.%f') for t in VR_data['absoluteT'][0:]])

# Calculate relative_t (time elapsed from absolute_t0)
VR_relative_t = np.array([(t - VR_absolute_t[0]).total_seconds() for t in VR_absolute_t])

# Add twoP_data['AbsoluteT'][0] to each timedelta object to get vrT
VR_relative_t_timedelta = np.array([datetime.timedelta(seconds=t) for t in VR_relative_t])
Aligned_Abs_vrT = twoP_data['AbsoluteT'][0] + VR_relative_t_timedelta

# Find the closest value in Aligned_Abs_vrT that is greater than twoP_data['AbsoluteT'][-1]
closest_value = Aligned_Abs_vrT[Aligned_Abs_vrT > twoP_data['AbsoluteT'][-1]][0]
closest_index = np.where(Aligned_Abs_vrT == closest_value)[0][0]

new_VR_data = {}
new_VR_data['AbsoluteT'] = np.array(Aligned_Abs_vrT)[:closest_index]
new_VR_data['RelativeT'] = VR_relative_t[:closest_index]
new_VR_data['event'] = VR_data['event'][:closest_index]
new_VR_data['location'] = VR_data['location'][:closest_index]

# Calculate relative time points for VR_data and twoP_data
twop_relativeT = twoP_data['AbsoluteT'] - twoP_data['AbsoluteT'][0]

# Convert to seconds
twop_relativeT = np.array([t.total_seconds() for t in twop_relativeT])
twoP_data['RelativeT'] = twop_relativeT

# Interpolate the location at twoP_data['RelativeT'] from new_VR_data['location'] at new_VR_data['RelativeT']
interpolated_location = np.interp(twoP_data['RelativeT'], 
                                new_VR_data['RelativeT'], 
                                new_VR_data['location'])
new_VR_data['interp_location'] = interpolated_location
print(f"size of interpolated_location is {interpolated_location.shape}")
print(f"size of new_VR_data['location'] is {new_VR_data['location'].shape}")


optimal_offset, _, _ = run_offset_optimization(twop_path, behav_path)
