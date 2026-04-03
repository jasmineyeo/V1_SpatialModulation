"""
Preprocess.py
### This script is used to analyze the 2p data and corresponding treadmill behavior data
### 1. Preprocess 2p data and treadmill behavior data (load and align)
### 2a. Find temporal offset, which yields the best alignment between 2p and behavior data
### 2b. Smooth the deconvolved traces using a 250 ms Gaussian window
### 2c. Remove inactive data points
### 3. Spatial discretization (divide the VR corridor into ~110 bins, each representing 1cm and assign each data point to its corresponding spatial bin)
### 4. Test for reliability for individual cells (calculate Pearson CC or cohen's D)
### 5. Response Plot - plotting activity of all responsive cells (cross validation – split trials in half)

This script is used to preprocess the 2p data and corresponding treadmill behavior data and save the processed data -- before SMI calculation
JSY, 04/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import datetime
from helper import dataLoader, files
from helper import SpikeSmoothing, ReliabilityTesting as RT, SpatialDiscretization as SD, BehavioralDataFiltering as DF, ResponseVisualization as RV    
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['legend.fontsize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 20
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14

def convert_stat_to_serializable(stat_data):
    """
    Convert Suite2P stat data to HDF5-serializable format.
    
    Parameters:
    -----------
    stat_data : list or numpy.ndarray
        Suite2P stat data containing cell statistics
        
    Returns:
    --------
    serializable_stat : dict
        Dictionary with HDF5-compatible arrays
    """
    if stat_data is None:
        return None
    
    n_cells = len(stat_data)
    
    # Initialize arrays to store key statistics
    serializable_stat = {
        'n_cells': n_cells,
        'ypix_median': np.zeros(n_cells),
        'xpix_median': np.zeros(n_cells),
        'ypix_mean': np.zeros(n_cells),
        'xpix_mean': np.zeros(n_cells),
        'area': np.zeros(n_cells),
        'compact': np.zeros(n_cells, dtype=np.float32),
        'footprint': np.zeros(n_cells, dtype=np.float32),
        'mrs': np.zeros(n_cells, dtype=np.float32),
        'mrs0': np.zeros(n_cells, dtype=np.float32),
        'npix': np.zeros(n_cells, dtype=np.int32),
        'radius': np.zeros(n_cells, dtype=np.float32),
        'aspect_ratio': np.zeros(n_cells, dtype=np.float32),
    }
    
    # Extract key statistics from each cell
    for i, cell_stat in enumerate(stat_data):
        if isinstance(cell_stat, dict):
            # Calculate median coordinates (most important for layer assignment)
            if 'ypix' in cell_stat and len(cell_stat['ypix']) > 0:
                serializable_stat['ypix_median'][i] = np.median(cell_stat['ypix'])
                serializable_stat['ypix_mean'][i] = np.mean(cell_stat['ypix'])
            
            if 'xpix' in cell_stat and len(cell_stat['xpix']) > 0:
                serializable_stat['xpix_median'][i] = np.median(cell_stat['xpix'])
                serializable_stat['xpix_mean'][i] = np.mean(cell_stat['xpix'])
            
            # Extract other important statistics if they exist
            for key in ['area', 'compact', 'footprint', 'mrs', 'mrs0', 'npix', 'radius', 'aspect_ratio']:
                if key in cell_stat:
                    try:
                        value = cell_stat[key]
                        if np.isscalar(value):
                            serializable_stat[key][i] = value
                        elif hasattr(value, '__len__') and len(value) > 0:
                            serializable_stat[key][i] = value[0] if len(value) == 1 else np.mean(value)
                    except (ValueError, TypeError, IndexError):
                        # Set to 0 if conversion fails
                        serializable_stat[key][i] = 0
    
    return serializable_stat

def convert_ops_to_serializable(ops_data):
    """
    Convert Suite2P ops data to HDF5-serializable format.
    
    Parameters:
    -----------
    ops_data : dict
        Suite2P ops data
        
    Returns:
    --------
    serializable_ops : dict
        Dictionary with HDF5-compatible data
    """
    if ops_data is None:
        return None
    
    serializable_ops = {}
    
    # List of important ops parameters to preserve
    important_keys = [
        'Ly', 'Lx', 'nframes', 'nchannels', 'fs', 'tau', 'diameter',
        'spatial_scale', 'aspect', 'threshold_scaling', 'max_overlap',
        'high_pass', 'smooth_sigma', 'maxregshift', 'th_badframes',
        'yrange', 'xrange'
    ]
    
    for key in important_keys:
        if key in ops_data:
            value = ops_data[key]
            try:
                # Convert to numpy array if it's a list or ensure it's a simple type
                if isinstance(value, (list, tuple)):
                    serializable_ops[key] = np.array(value)
                elif isinstance(value, (int, float, bool, np.integer, np.floating)):
                    serializable_ops[key] = value
                elif isinstance(value, np.ndarray):
                    # Ensure the array is a simple numeric type
                    if value.dtype == object:
                        # Try to convert object array to numeric
                        try:
                            serializable_ops[key] = value.astype(float)
                        except (ValueError, TypeError):
                            # Skip if conversion fails
                            continue
                    else:
                        serializable_ops[key] = value
                elif isinstance(value, str):
                    serializable_ops[key] = value
            except (ValueError, TypeError):
                # Skip problematic values
                continue
    
    return serializable_ops

def preprocess_2pVR(twop_filepath):
    
    # Find a file that has "VRlog*.txt" in the twop_filepath directory
    import glob
    vrlog_pattern = os.path.join(twop_filepath, "VRlog*.txt")
    vrlog_files = glob.glob(vrlog_pattern)

    if len(vrlog_files) == 1:
        vr_filepath = vrlog_files[0]
        print(f"Found VRlog file: {vr_filepath}")
    elif len(vrlog_files) > 1:
        # If multiple files found, use the first one (or could sort by date)
        vr_filepath = sorted(vrlog_files)[0]
        print(f"Warning: Found {len(vrlog_files)} VRlog files, using: {vr_filepath}")

    # 1. Preprocess 2p data and treadmill behavior data (load and align)
    procData = dataLoader(twop_filepath, vr_filepath)
    animal_id, date, framerate = procData.load_data()
    twop_dict, vr_dict = procData.align_data()

    # 2a. Find temporal offset, which yields the best alignment between 2p and behavior data
    optimal_offset, _, _ = SpikeSmoothing.run_offset_optimization(twop_filepath, vr_filepath)
    # optimal_offset = 5
    # Use the optimal offset in your main preprocessing
    offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], optimal_offset)
    # offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], 5)

    # 2b. Smooth the deconvolved traces using a 250 ms Gaussian window
    smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms=250)
    twop_dict['smoothed_spks'] = smoothed
    
    # 2c. Remove inactive data points with CORRECTED speed calculation
    print("\n" + "="*80)
    print("FILTERING DATA WITH CORRECTED SPEED CALCULATION")
    print("="*80)
    
    min_trial_duration_seconds = 5
    max_trial_duration_seconds = 60

    # FIXED: Returns speed_laps now!
    filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps = \
        DF.process_data_with_speed_filtering(
            smoothed, 
            vr_dict['interp_location'],
            min_trial_duration_seconds=min_trial_duration_seconds, 
            max_trial_duration_seconds=max_trial_duration_seconds,
            framerate=framerate,
            min_speed_cm_s=2.0,
            frames_to_keep=5
        )
    
    if n_valid_laps == 0:
        raise ValueError("No valid laps after filtering!")
    
    # 3. Spatial discretization
    single_revolution_VR = 282.415
    single_revolution_treadmill = 27.8
    single_lap_VR = 1320.645683
    single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR

    spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = SD.spatial_assignment_with_physical_units(
        n_valid_laps,
        filtered_spks_laps, 
        filtered_location_laps,
        physical_lap_length_cm=single_lap_treadmill
    )
    
    window_cm = 0.5
    smoothed_spatial_activity = SpikeSmoothing.spatial_smooth(spatial_activity, window_cm=window_cm)

    # 4. Test for reliability for individual cells
    normalized_spatial_activity = RT.normalize_spatial_activity(smoothed_spatial_activity)

    combined_reliable, reliable_cells, _, avg_cc, cohens_d, _, _, _ = RT.combined_reliability_test_improved(
        smoothed_spatial_activity,
        n_shuffles = 200,
        cc_percentile=90,
        cohen_threshold=0.8,
        min_cc_threshold=0.2,
        min_pattern_corr=0.3,
        peak_distance_threshold=5,
        use_activity_threshold=True,
        activity_method='absolute_percentile'
    )

    print(f"Found {np.sum(reliable_cells)} reliable cells out of {len(reliable_cells)}")
    print(f"Found {np.sum(combined_reliable)} combined_reliable cells out of {len(combined_reliable)}")
    
    
    # # Option 2: Plot with both heatmap and trial-averaged activity
    # fig2 = RT.plot_reliable_cells_side_by_side(
    #     normalized_spatial_activity,
    #     reliable_cells,
    #     # combined_reliable,
    #     max_cells=30,              # Show up to 10 reliable cells
    #     # max_cells=np.sum(reliable_cells),                # Show up to 10 reliable cells
    #     avg_cc=avg_cc,               # Optional correlation coefficients
    #     cohen_d=cohens_d,            # Optional Cohen's D values
    #     normalize=False               # Apply normalization
    # )
    # # plt.suptitle('Spikes of reliable cells', fontsize=15)
    # # plt.tight_layout(rect=[0, 0, 1, 0.985])  # Adjust the rect parameter to add space at the top
    # plt.show()
    
    # if save_directory is not None:
    #     os.makedirs(save_directory, exist_ok=True)
    #     fig2.savefig(os.path.join(save_directory, 'reliable_cells_side_by_side.png'), dpi=150)
    #     print(f"\n✓ Reliable cells side-by-side plot saved to {save_directory}")
        
    # ===== NEW: Prepare temporal data for speed tuning analysis =====
    print("\n" + "="*80)
    print("PREPARING TEMPORAL DATA FOR SPEED TUNING ANALYSIS")
    print("="*80)
    
    # Concatenate all laps to get continuous temporal sequences
    temporal_spikes = np.concatenate(filtered_spks_laps, axis=1)
    temporal_location = np.concatenate(filtered_location_laps)
    temporal_speed = np.concatenate(filtered_speed_laps)
    
    # Create lap boundaries for the temporal data
    lap_starts = []
    lap_ends = []
    cumsum = 0
    
    for lap_speed in filtered_speed_laps:
        lap_starts.append(cumsum)
        cumsum += len(lap_speed)
        lap_ends.append(cumsum)
    
    lap_starts = np.array(lap_starts)
    lap_ends = np.array(lap_ends)
    
    print(f"\n✓ Temporal data prepared:")
    print(f"  Spikes: {temporal_spikes.shape} (cells × frames)")
    print(f"  Speed: {len(temporal_speed)} frames")
    print(f"  Location: {len(temporal_location)} frames")
    print(f"  Laps: {len(lap_starts)}")
    print(f"  Frame range: 0 to {lap_ends[-1]}")
    
    # Verify dimensions match
    assert temporal_spikes.shape[1] == len(temporal_speed), "Spike/speed dimension mismatch!"
    assert len(temporal_speed) == len(temporal_location), "Speed/location dimension mismatch!"
    assert lap_ends[-1] == len(temporal_speed), "Lap boundaries don't match data length!"
    
    print("\n✓ All dimensions verified!")
        
    # 5. Response Plot - plotting activity of all responsive cells
    combinedreliablecell_save_directory = os.path.join(twop_filepath, 'combined_reliable_cell_plots')
    reliablecell_save_directory = os.path.join(twop_filepath, 'reliable_cell_plots')
    os.makedirs(combinedreliablecell_save_directory, exist_ok=True)
    os.makedirs(reliablecell_save_directory, exist_ok=True)
    
    pdf_path, stats = RT.plot_individual_reliable_cells_to_pdf(
        spatial_activity=normalized_spatial_activity,
        reliable_cells=combined_reliable,
        save_directory=combinedreliablecell_save_directory,
        avg_cc=avg_cc,
        cohen_d=cohens_d,
        bin_centers=bin_centers,
        normalize=True,
        dpi=150,
        cells_per_page=4
    )

    fig1, _ = RV.create_response_plot(normalized_spatial_activity, reliable_cells, clim=(0, 1))
    fig1.savefig(os.path.join(reliablecell_save_directory, 'reliable_cells.png'), dpi=150)

    fig2, _ = RV.create_response_plot(normalized_spatial_activity, combined_reliable, clim=(0, 1))
    fig2.savefig(os.path.join(combinedreliablecell_save_directory, 'combined_reliable_cells.png'), dpi=150)
    
    # Calculate median coordinates for each cell (needed for combining datasets)
    med_coords = np.zeros((len(twop_dict['stat']), 2))
    for i, cell_stat in enumerate(twop_dict['stat']):
        med_coords[i, 0] = np.median(cell_stat['ypix'])  # y-coordinate
        med_coords[i, 1] = np.median(cell_stat['xpix'])  # x-coordinate

    # Convert problematic data structures to HDF5-compatible format
    serializable_stat = convert_stat_to_serializable(twop_dict['stat'])
    serializable_ops = convert_ops_to_serializable(twop_dict['ops'])

    # UPDATED: Include all necessary data for dataset combination with HDF5-compatible types
    preprocessed_dict = {
        # Spatial data (existing)
        'spatial_activity': smoothed_spatial_activity,
        'norm_spatial_activity': normalized_spatial_activity,
        'reliable_cells': reliable_cells.astype(bool),
        'combined_reliable': combined_reliable.astype(bool),
        'avg_cc': avg_cc.astype(np.float64),
        'cohen_d': cohens_d.astype(np.float64),
        'bin_centers': bin_centers.astype(np.float64),
        'med_coords': med_coords.astype(np.float64),
        'stat_serializable': serializable_stat,
        'ops_serializable': serializable_ops,
        
        # NEW: Temporal data for speed tuning analysis
        'speed_cm_s': temporal_speed.astype(np.float64),
        'smoothed_spks_temporal': temporal_spikes.astype(np.float64),
        'location_cm': temporal_location.astype(np.float64),
        'lap_starts': lap_starts.astype(np.int32),
        'lap_ends': lap_ends.astype(np.int32),
        
        # Metadata
        'twop_filepath': str(twop_filepath),
        'vr_filepath': str(vr_filepath),
        'processing_timestamp': datetime.datetime.now().isoformat(),
        'processing_params': {
            'framerate': float(framerate),
            'optimal_offset': int(optimal_offset),
            'window_cm': float(window_cm),
            'min_trial_duration_seconds': float(min_trial_duration_seconds),
            'max_trial_duration_seconds': float(max_trial_duration_seconds),
            'min_speed_cm_s': float(2.0),
            'single_lap_treadmill': float(single_lap_treadmill)
        }
    }
    
    # Additional data validation before saving
    print("Validating data types before saving...")
    
    # Check for any remaining object arrays
    for key, value in preprocessed_dict.items():
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                print(f"WARNING: {key} still has object dtype, attempting conversion...")
                try:
                    # Try to convert to float if possible
                    preprocessed_dict[key] = value.astype(float)
                    print(f"  Successfully converted {key} to float")
                except (ValueError, TypeError):
                    print(f"  ERROR: Could not convert {key}, removing from save dict")
                    # Remove problematic key rather than crash
                    del preprocessed_dict[key]
                    continue
            print(f"  {key}: {value.dtype} - OK")
        elif isinstance(value, dict):
            print(f"  {key}: dict with {len(value)} keys - OK")
        else:
            print(f"  {key}: {type(value)} - OK")
    
    # save preprocessed data 
    save_dir = os.path.dirname(twop_filepath) if os.path.isfile(twop_filepath) else twop_filepath
    _savepath = os.path.join(save_dir, f'{date}_{animal_id}_preproc.h5')
    os.makedirs(os.path.dirname(_savepath), exist_ok=True)
    
    print('Writing preprocessed data to {}'.format(_savepath))    
    
    try:
        files.write_h5(_savepath, preprocessed_dict)
        print('Successfully saved preprocessed data!')
    except Exception as e:
        print(f'Error saving to HDF5: {e}')
        print('Attempting to save without problematic stat and ops data...')
        
        # Create a minimal version without the problematic data
        minimal_dict = {key: value for key, value in preprocessed_dict.items() 
                       if key not in ['stat_serializable', 'ops_serializable']}
        
        try:
            files.write_h5(_savepath, minimal_dict)
            print('Successfully saved minimal preprocessed data (without full stat/ops)')
        except Exception as e2:
            print(f'Failed to save even minimal data: {e2}')
            print('Consider using pickle format instead of HDF5 for this dataset.')
            
    return preprocessed_dict

if __name__ == "__main__":
    
    twop_filepaths = [
        # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001"

        # r'D:\V1_SpatialModulation\2p\V1_window\JSY061_ChronicImaging_Axonal\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001',
        # r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003',
        # r'F:\2P\unprocessed\251123_JSY_JSY044_SpMod_OpenLoopVR_Stationary\TSeries-11232025-1222-001',
        # r'D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260205_JSY_JSY061_SpMod_AxonalImaging_Day4\TSeries-02052026-1833-002',
        # r'D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260206_JSY_JSY061_SpMod_AxonalImaging_Day5\TSeries-02062026-1850-001',
        # r'D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260207_JSY_JSY061_SpMod_AxonalImaging_Day6\TSeries-02072026-2023-001',
        # r"D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window\260208_JSY_JSY061_SpMod_AxonalImaging_Day7\TSeries-02082026-1826-001"
        ]

    
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001'
    
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001'
    
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002'
    
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-002'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1545-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1545-002'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001'
    
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001'
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001'

    # twop_filepaths = [
    # # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251205_JSY_JSY055_SpatialModulation_Day1\TSeries-12052025-1740-001'
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251206_JSY_JSY055_SpatialModulation_Day2\TSeries-12062025-1810-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251207_JSY_JSY055_SpatialModulation_Day3\TSeries-12072025-1825-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251208_JSY_JSY055_SpatialModulation_Day4\TSeries-12082025-1633-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251209_JSY_JSY055_SpatialModualtion_Day5\TSeries-12092025-2000-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251210_JSY_JSY055_SpatialModulation_Day6\TSeries-12102025-1702-001',
    # r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging\251211_JSY_JSY055_SpatialModulation_Day7\TSeries-12112025-1631-001'
    # ]

    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_raw_separateregistration\TSeries-09062025-1308-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09062025_01-50-45.txt"
    # twop_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_raw_separateregistration\TSeries-09072025-1257-001"
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09072025_01-18-32.txt"
    # twop_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_raw_separateregistration\TSeries-09082025-1540-001"
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09082025_04-02-31.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09092025_01-15-55.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09102025_02-14-21.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09112025_02-48-03.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09122025_01-57-03.txt"


    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_raw_separateregistration\TSeries-09062025-1308-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09062025_02-09-47.txt"
    # twop_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_raw_separateregistration\TSeries-09072025-1257-002"
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09072025_01-39-00.txt"
    # twop_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_raw_separateregistration\TSeries-09082025-1540-002"
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09082025_04-14-19.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09092025_01-29-34.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09102025_02-14-21.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09112025_03-10-04.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09122025_02-11-23.txt"

    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08112025_04-04-19.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250813_JSY_JSY044_SpatialModulation_Day3\TSeries-08132025-1456-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08132025_04-05-48.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250815_JSY_JSY044_SpatialModulation_Day5\TSeries-08152025-1527-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08152025_03-44-41.txt"
    # twop_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250815_JSY_JSY044_SpatialModulation_Day5\TSeries-08152025-1527-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08152025_05-30-39.txt"    
    
    

    # ==========================================================================
    # Process all files in a loop
    # ==========================================================================
    n_total = len(twop_filepaths)
    successful = []
    failed = []

    for i, twop_filepath in enumerate(twop_filepaths):
        print("\n" + "=" * 80)
        print(f"Processing {i+1}/{n_total}: {os.path.basename(twop_filepath)}")
        print("=" * 80)

        try:
            preprocess_2pVR(twop_filepath)
            successful.append(twop_filepath)
            print(f"Successfully processed: {os.path.basename(twop_filepath)}")
        except Exception as e:
            failed.append((twop_filepath, str(e)))
            print(f"FAILED: {os.path.basename(twop_filepath)}")
            print(f"Error: {e}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {len(successful)}/{n_total}")
    print(f"Failed: {len(failed)}/{n_total}")

    if failed:
        print("\nFailed files:")
        for filepath, error in failed:
            print(f"  - {os.path.basename(filepath)}: {error}")