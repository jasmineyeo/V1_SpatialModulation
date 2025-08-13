### This script is used to preprocess the 2p data and corresponding treadmill behavior data and save the processed data -- before SMI calculation
### JSY, 04/2025

### This script is used to analyze the 2p data and corresponding treadmill behavior data
### 1. Preprocess 2p data and treadmill behavior data (load and align)
### 2a. Find temporal offset, which yields the best alignment between 2p and behavior data
### 2b. Smooth the deconvolved traces using a 250 ms Gaussian window
### 2c. Remove inactive data points
### 3. Spatial discretization (divide the VR corridor into ~110 bins, each representing 1cm and assign each data point to its corresponding spatial bin)
### 4. Test for reliability for individual cells (calculate Pearson CC or cohen’s D)
### 5. Response Plot - plotting activity of all responsive cells (cross validation – split trials in half)

import sys
import os

parent_dir = r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation"
sys.path.insert(0, parent_dir)

import numpy as np
import matplotlib.pyplot as plt
from helper import dataLoader
from helper import files
from helper import SpikeSmoothing, ReliabilityTesting as RT, SpatialDiscretization as SD, BehavioralDataFiltering as DF, ResponseVisualization as RV    

from matplotlib import rcParams
rcParams['legend.fontsize'] = 14
rcParams['axes.labelsize'] = 14
rcParams['axes.titlesize'] = 20
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14


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
    # best_offset = SpikeSmoothing.find_temporal_offset(twop_dict, vr_dict, framerate)

    # 2b. Smooth the deconvolved traces using a 250 ms Gaussian window
    offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], 0)
    smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms=500)
    twop_dict['smoothed_spks'] = smoothed

    # 2c. Remove inactive data points
    min_trial_duration_seconds = 5
    max_trial_duration_seconds = 60

    filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_trial_filtering(
        twop_dict['smoothed_spks'], 
        vr_dict['interp_location'],
        min_trial_duration_seconds = min_trial_duration_seconds, 
        max_trial_duration_seconds = max_trial_duration_seconds,
        framerate=framerate
    )

    # 3. Spatial discretization (divide the VR corridor into ~110 bins, each representing 1cm and assign each data point to its corresponding spatial bin)
    # # when VR length was 300 at gain = 1.15 - 25/03/20 
    single_revolution_VR = 282.415
    single_revolution_treadmill = 27.8
    # single_lap_VR = 1726.99731 ### = 1146 when VR length was 125 at gain = 1.15 
    # single_lap_VR = 1320.645683 ### = 1146 when VR length was 125 at gain = 1.15 
    single_lap_VR = 1320.645683 ### = 1146 when VR length was 125 at gain = 1.15 
    single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR

    # Then perform spatial assignment on the filtered data
    spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = SD.spatial_assignment(
        n_valid_laps,
        filtered_spks_laps, 
        filtered_location_laps, 
        single_lap_treadmill
    )

    window_cm = 5
    smoothed_spatial_activity = SpikeSmoothing.spatial_smooth(spatial_activity, window_cm=window_cm)

    # 4. Test for reliability for individual cells (calculate Pearson CC or cohen’s D)
    normalized_spatial_activity = RT.normalize_spatial_activity(smoothed_spatial_activity)

    # Run the analysis
    combined_reliable, reliable_cells, _, avg_cc, cohens_d, _, _, _ = RT.combined_reliability_test(
        smoothed_spatial_activity,
        n_shuffles=100,           # Use 1000+ for final analysis
        cc_percentile=90,          # 90th percentile threshold for CC
        cohen_threshold=1,       # Medium-large effect size
        min_cc_threshold=0.2,      # Minimum correlation required
        min_activity_threshold=0.0, # Minimum activity level (relative to max)
        min_pattern_corr=0.3, 
        peak_distance_threshold=5
    )

    print(f"Found {np.sum(reliable_cells)} reliable cells out of {len(reliable_cells)}")
    print(f"Found {np.sum(combined_reliable)} combined_reliable -- reliable in both even and odd trials -- cells out of {len(combined_reliable)}")
    

    # 5. Response Plot - plotting activity of all responsive cells (cross validation – split trials in half)
    fig2 = RT.plot_reliable_cells_side_by_side(
        normalized_spatial_activity,
        # reliable_cells,
        combined_reliable,
        max_cells=50,              # Show up to 10 reliable cells
        # max_cells=np.sum(combined_reliable),                # Show up to 10 reliable cells
        avg_cc=avg_cc,               # Optional correlation coefficients
        cohen_d=cohens_d,            # Optional Cohen's D values
        normalize=False               # Apply normalization
    )
    plt.suptitle('Spikes of reliable cells', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.985])  # Adjust the rect parameter to add space at the top
    plt.show()

    fig1, sorted_indices = RV.create_response_plot(normalized_spatial_activity, combined_reliable, clim=(0, 1))  # Optional: manually set contrast limits for stronger effect
    plt.show()


    _savepath = os.path.join(twop_filepath, '{}_{}_preproc.h5'.format(date,animal_id))
    print('Writing preprocessed data to {}'.format(_savepath))
    # files.write_h5(_savepath, preprocessed_dict)


if __name__ == "__main__":
    # Replace 'your_config_file_path.yaml' with the actual path to your config file
    twop_filepath = r'F:\2P\spmod\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001'
    vr_filepath = r"D:\V1_SpatialModulation\V1_SpatialMod_VRLog\VRlog_JSY038_08112025_04-04-19.txt"
    preprocess_2pVR(twop_filepath, vr_filepath)      