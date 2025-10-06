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
import shutil
from helper import dataLoader, files
from helper import SpikeSmoothing, ReliabilityTesting as RT, SpatialDiscretization as SD, BehavioralDataFiltering as DF, ResponseVisualization as RV    
from matplotlib import rcParams

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

def preprocess_2pVR(twop_filepath, vr_filepath):

    if twop_filepath is None or vr_filepath is None:
        # file paths for twop and behavioral data
        twop_filepath = r'F:\2P\spmod\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001'
        vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08112025_04-04-19.txt"

    # 1. Preprocess 2p data and treadmill behavior data (load and align)
    procData = dataLoader(twop_filepath, vr_filepath)
    animal_id, date, framerate = procData.load_data()
    twop_dict, vr_dict = procData.align_data()

    # 2a. Find temporal offset, which yields the best alignment between 2p and behavior data
    optimal_offset, _, _ = SpikeSmoothing.run_offset_optimization(twop_filepath, vr_filepath)

    # Use the optimal offset in your main preprocessing
    offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], optimal_offset)
    # offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], 6)

    # 2b. Smooth the deconvolved traces using a 250 ms Gaussian window
    smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms=250)
    twop_dict['smoothed_spks'] = smoothed

    # 2c. Remove inactive data points
    min_trial_duration_seconds = 5
    max_trial_duration_seconds = 60

    filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_speed_filtering(
        smoothed, 
        vr_dict['interp_location'],
        min_trial_duration_seconds=min_trial_duration_seconds, 
        max_trial_duration_seconds=max_trial_duration_seconds,
        framerate=framerate,
        min_speed_cm_s=2.0,
        frames_to_keep=5
    )

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
    
    window_cm = 2
    smoothed_spatial_activity = SpikeSmoothing.spatial_smooth(spatial_activity, window_cm=window_cm)

    # 4. Test for reliability for individual cells
    normalized_spatial_activity = RT.normalize_spatial_activity(smoothed_spatial_activity)

    combined_reliable, reliable_cells, _, avg_cc, cohens_d, _, _, _ = RT.combined_reliability_test_improved(
        smoothed_spatial_activity,
        n_shuffles=500,
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
        
    # 5. Response Plot - plotting activity of all responsive cells
    combinedreliablecell_save_directory = os.path.join(twop_filepath, 'combined_reliable_cell_plots')
    reliablecell_save_directory = os.path.join(twop_filepath, 'reliable_cell_plots')
    
    saved_files, stats = RT.save_all_reliable_cell_plots(
        spatial_activity=normalized_spatial_activity,
        reliable_cells=reliable_cells,
        save_directory=reliablecell_save_directory,
        avg_cc=avg_cc,
        cohen_d=cohens_d,
        bin_centers=bin_centers,
        normalize=True,
        file_format='png',
        dpi=150
    )

    saved_files, stats = RT.save_all_reliable_cell_plots(
        spatial_activity=normalized_spatial_activity,
        reliable_cells=combined_reliable,
        save_directory=combinedreliablecell_save_directory,
        avg_cc=avg_cc,
        cohen_d=cohens_d,
        bin_centers=bin_centers,
        normalize=True,
        file_format='png',
        dpi=150
    )

    fig1, _ = RV.create_response_plot(normalized_spatial_activity, reliable_cells, clim=(0, 1))
    fig1.savefig(os.path.join(combinedreliablecell_save_directory, 'reliable_cells.png'), dpi=150)

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
        'spatial_activity': smoothed_spatial_activity,
        'norm_spatial_activity': normalized_spatial_activity,
        'reliable_cells': reliable_cells.astype(bool),  # Ensure boolean type
        'combined_reliable': combined_reliable.astype(bool),  # Ensure boolean type
        'avg_cc': avg_cc.astype(np.float64),  # Ensure float64
        'cohen_d': cohens_d.astype(np.float64),  # Ensure float64
        'bin_centers': bin_centers.astype(np.float64),  # Ensure float64
        'med_coords': med_coords.astype(np.float64),  # NEW: median coordinates for cell matching
        'stat_serializable': serializable_stat,  # NEW: serializable cell stat data
        'ops_serializable': serializable_ops,    # NEW: serializable ops data
        'twop_filepath': str(twop_filepath),
        'vr_filepath': str(vr_filepath),
        'processing_timestamp': datetime.datetime.now().isoformat(),
        'processing_params': {     # NEW: processing parameters for reference
            'framerate': float(framerate),
            'optimal_offset': int(6),
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
    # # day 1, 0906
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09062025_01-50-45.txt"
    
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1\TSeries-09062025-1308-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09062025_02-09-47.txt"

    # # day 2, 0907
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250907_JSY_JSY044_SpatialModulation_Day2\TSeries-09072025-1257-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09072025_01-18-32.txt"

    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2\TSeries-09072025-1257-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09072025_01-39-00.txt"
    
    # # day 3, 0908
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_togetherregistration\TSeries-09082025-1540-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09082025_04-02-31.txt"
    
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_togetherregistration\TSeries-09082025-1540-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09082025_04-14-19.txt"

    twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_togetherregistration\TSeries-09082025-1540-003'
    vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09082025_04-27-09.txt"
    
    # # day 4, 0909
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09092025_01-15-55.txt"
    
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09092025_01-29-34.txt"

    # # # day 5, 0910
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09102025_02-14-21.txt"
    
    
    # # day 6, 0911
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09112025_02-48-03.txt"
    
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09112025_03-10-04.txt"
    
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-003'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09112025_03-22-48.txt"
    
    # # # day 7, 0912
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09122025_01-57-03.txt"
    
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-002'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09122025_02-11-23.txt"
    
    # twop_filepath = r'F:\2P\spmod\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-003'
    # vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_09122025_02-27-27.txt"

    preprocess_2pVR(twop_filepath, vr_filepath)