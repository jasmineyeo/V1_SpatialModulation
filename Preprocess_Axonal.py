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
from helper.SkaggsSI import run_spatial_information_analysis
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20

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

def _infer_imaging_type(twop_filepath):
    """Infer imaging type from the session path."""
    p = twop_filepath.lower()
    if 'window' in p:
        return 'L1_axonal'
    if 'prism' in p:
        return 'L6_axonal'
    return 'unknown'


def preprocess_2pVR(twop_filepath):

    imaging_type = _infer_imaging_type(twop_filepath)
    print(f"Imaging type: {imaging_type}")

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
    optimal_offset, _, _ = SpikeSmoothing.run_offset_optimization(twop_filepath, vr_filepath, list(range(-10, 11)))
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

    # Apply same offset + lap filtering to dF/F so it is frame-for-frame aligned with temporal_spikes
    dFF_offset = SpikeSmoothing.apply_temporal_offset(twop_dict['dFF'], optimal_offset)
    filtered_dFF_laps, _, _, _ = DF.process_data_with_speed_filtering(
        dFF_offset,
        vr_dict['interp_location'],
        min_trial_duration_seconds=min_trial_duration_seconds,
        max_trial_duration_seconds=max_trial_duration_seconds,
        framerate=framerate,
        min_speed_cm_s=2.0,
        frames_to_keep=5
    )
    temporal_dFF = np.concatenate(filtered_dFF_laps, axis=1)

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

    # 4b. Skaggs SI — permissive activity filter then shuffle significance test
    print("\n" + "="*80)
    print("SKAGGS SPATIAL INFORMATION — ALTERNATIVE CELL SELECTION")
    print("="*80)

    mean_activity   = np.mean(temporal_spikes, axis=1)
    active_fraction = np.mean(temporal_spikes > 0, axis=1)
    active_axons    = (mean_activity > 0.05) & (active_fraction > 0.01)
    print(f"Active axons (permissive filter): {np.sum(active_axons)} / {len(active_axons)}")

    si_results = run_spatial_information_analysis(
        temporal_spikes, temporal_location,
        active_mask=active_axons,
        bin_centers=bin_centers,
        n_shuffles=200,
        alpha=0.05,
        smooth_sigma=1.5,
        verbose=True,
    )
    si_significant_cells = si_results['is_significant']
    print(f"SI-significant: {np.sum(si_significant_cells)} / {np.sum(active_axons)} active")

    # Compare the two selection methods
    print("\nComparison:")
    print(f"  combined_reliable only : {np.sum(combined_reliable & ~si_significant_cells)}")
    print(f"  si_significant only    : {np.sum(si_significant_cells & ~combined_reliable)}")
    print(f"  overlap (both)         : {np.sum(combined_reliable & si_significant_cells)}")

    combined_reliable_only = combined_reliable & ~si_significant_cells
    si_significant_only = si_significant_cells & ~combined_reliable
    
    # 5. Response Plot - plotting activity of all responsive cells
    combinedreliablecell_save_directory = os.path.join(twop_filepath, 'combined_reliable_cell_plots')
    reliablecell_save_directory = os.path.join(twop_filepath, 'reliable_cell_plots')
    si_significant_cell_save_directory = os.path.join(twop_filepath, 'si_significant_cell_plots')
    os.makedirs(combinedreliablecell_save_directory, exist_ok=True)
    os.makedirs(reliablecell_save_directory, exist_ok=True)
    os.makedirs(si_significant_cell_save_directory, exist_ok=True)
    from scipy.ndimage import gaussian_filter1d
    plot_activity = gaussian_filter1d(normalized_spatial_activity, sigma=1.5, axis=2)

    pdf_path, stats = RT.plot_individual_reliable_cells_to_pdf(
        spatial_activity=plot_activity,
        reliable_cells=combined_reliable,
        save_directory=combinedreliablecell_save_directory,
        avg_cc=avg_cc,
        cohen_d=cohens_d,
        bin_centers=bin_centers,
        normalize=True,
        dpi=150,
        cells_per_page=4
    )

    pdf_path, stats = RT.plot_individual_reliable_cells_to_pdf(
        spatial_activity=plot_activity,
        reliable_cells=si_significant_cells,
        save_directory=si_significant_cell_save_directory,
        avg_cc=avg_cc,
        cohen_d=cohens_d,
        bin_centers=bin_centers,
        normalize=True,
        dpi=150,
        cells_per_page=4
    )

    fig1, _ = RV.create_response_plot(plot_activity, reliable_cells, clim=(0, 1))
    fig1.savefig(os.path.join(reliablecell_save_directory, 'reliable_cells.png'), dpi=150)

    fig2, _ = RV.create_response_plot(plot_activity, combined_reliable, clim=(0, 1))
    fig2.savefig(os.path.join(combinedreliablecell_save_directory, 'combined_reliable_cells.png'), dpi=150)
    
    fig3, _ = RV.create_response_plot(plot_activity, si_significant_cells, clim=(0, 1))
    fig3.savefig(os.path.join(twop_filepath, 'si_significant_cells.png'), dpi=150)
    
    fig4, _ = RV.create_response_plot(plot_activity, combined_reliable_only, clim=(0, 1))
    fig4.savefig(os.path.join(twop_filepath, 'combined_reliable_only.png'), dpi=150)
    
    fig5, _ = RV.create_response_plot(plot_activity, si_significant_only, clim=(0, 1))
    fig5.savefig(os.path.join(twop_filepath, 'si_significant_only.png'), dpi=150)

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
        'si_significant_cells': si_significant_cells.astype(bool),
        'active_axons': active_axons.astype(bool),
        'avg_cc': avg_cc.astype(np.float64),
        'cohen_d': cohens_d.astype(np.float64),
        'bin_centers': bin_centers.astype(np.float64),
        'med_coords': med_coords.astype(np.float64),
        'stat_serializable': serializable_stat,
        'ops_serializable': serializable_ops,
        
        # NEW: Temporal data for speed tuning analysis
        'speed_cm_s': temporal_speed.astype(np.float64),
        'smoothed_spks_temporal': temporal_spikes.astype(np.float64),
        'dFF_temporal':           temporal_dFF.astype(np.float64),
        'location_cm': temporal_location.astype(np.float64),
        'lap_starts': lap_starts.astype(np.int32),
        'lap_ends': lap_ends.astype(np.int32),
        
        # Metadata
        'imaging_type': imaging_type,
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
    
    # Verify dFF_temporal is frame-for-frame aligned with temporal_spikes before saving
    assert temporal_dFF.shape == temporal_spikes.shape, \
        f'Shape mismatch: dFF {temporal_dFF.shape} vs spikes {temporal_spikes.shape}'
    print(f'✓ dFF_temporal shape verified: {temporal_dFF.shape}')

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

    from Preprocess_MultipleRecordings import preprocess_2pVR_multi

    # -----------------------------------------------------------------------
    # single recordings → preprocess_2pVR
    # two recordings combined in suite2p → preprocess_2pVR_multi
    # -----------------------------------------------------------------------

    # BASE = r'D:\V1_SpatialModulation\2p\V1_axonal\JSY061_ChronicImaging_window'

    # # Single-recording sessions
    # twop_filepaths = [
    #     rf'{BASE}\260202_JSY_JSY061_SpMod_AxonalImaging_Day1\TSeries-02022026-1804-001',
    #     # Day 2 handled separately below (multi-recording)
    #     rf'{BASE}\260204_JSY_JSY061_SpMod_AxonalImaging_Day3\TSeries-02042026-2009-001',
    #     rf'{BASE}\260205_JSY_JSY061_SpMod_AxonalImaging_Day4\TSeries-02052026-1833-002',
    #     rf'{BASE}\260206_JSY_JSY061_SpMod_AxonalImaging_Day5\TSeries-02062026-1850-001',
    #     rf'{BASE}\260207_JSY_JSY061_SpMod_AxonalImaging_Day6\TSeries-02072026-2023-001',
    #     rf'{BASE}\260208_JSY_JSY061_SpMod_AxonalImaging_Day7\TSeries-02082026-1826-001',
    # ]
    # twop_filepaths = []

    # # Day 2 — if two recordings merged in suite2p
    # day2_suite2p_path = rf'{BASE}\260203_JSY_JSY061_SpMod_AxonalImaging_Day2\TSeries-02032026-1751-001'
    # day2_recording_pairs = [
    #     (rf'{BASE}\260203_JSY_JSY061_SpMod_AxonalImaging_Day2\TSeries-02032026-1751-001', None),
    #     (rf'{BASE}\260203_JSY_JSY061_SpMod_AxonalImaging_Day2\TSeries-02032026-1751-002', None),
    # ]
    BASE = r'D:\V1_SpatialModulation\2p\V1_axonal\JSY060_ChronicImaging_prism'

    twop_filepaths = [
        # --- JSY061 ---
        # rf'{BASE}\260225_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day1\TSeries-02252026-0903-001',
        # rf'{BASE}\260226_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day2\TSeries-02262026-0915-001',  
        # rf'{BASE}\260227_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day3\TSeries-02262026-1253-001',
        # rf'{BASE}\260228_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day4\TSeries-02282026-0919-001',
        # rf'{BASE}\260301_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day5\TSeries-03012026-0914-002',
        # rf'{BASE}\260302_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day6\TSeries-03022026-1226-001',
        rf'{BASE}\260303_JSY_JSY060_LongitudinalImaging_Axonal_Prism_Day7\TSeries-03032026-0817-001',
    ]


    # -----------------------------------------------------------------------
    # Process single-recording sessions
    # -----------------------------------------------------------------------
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

    # # -----------------------------------------------------------------------
    # # Process Day 2 (multi-recording)
    # # -----------------------------------------------------------------------
    # print("\n" + "=" * 80)
    # print(f"Processing Day 2 (multi-recording): {os.path.basename(day2_suite2p_path)}")
    # print("=" * 80)

    # try:
    #     preprocess_2pVR_multi(day2_suite2p_path, day2_recording_pairs)
    #     successful.append(day2_suite2p_path)
    #     print("Successfully processed Day 2.")
    # except Exception as e:
    #     failed.append((day2_suite2p_path, str(e)))
    #     print(f"FAILED Day 2: {e}")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {len(successful)}/{n_total + 1}")
    print(f"Failed:     {len(failed)}/{n_total + 1}")

    if failed:
        print("\nFailed files:")
        for filepath, error in failed:
            print(f"  - {os.path.basename(filepath)}: {error}")



