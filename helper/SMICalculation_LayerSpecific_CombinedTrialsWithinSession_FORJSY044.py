"""
SMICalculation_LayerSpecific_CombinedTrialsWithinSession_FORJSY044.py
A script for calculating the Spatial Modulation Index (SMI) for specific layers in the mouse primary visual cortex
(for all sessions for JSY044, max number of trials for VR was 60, so there are multiple trials within a session)
Input: 2p (co-registered using suite2p, deconcat using CombiningMultipleTrialsWithinSession.py, preprocessed using preprocess.py
        and then concat using CombiningMultipleTrialsWithinSession.py) and VR data processed using preprocess.py
Output: SMI values for each layer

JSY, 10/04/25
"""
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from helper import files, TwoP
from helper import SpatialModulationIndex as SMI, ResponseVisualization as RV
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer

data_filepath = r"F:\2P\spmod\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_togetherregistration"

# Find files ending with preproc.h5
preproc_files = glob.glob(os.path.join(data_filepath, "*.h5"))

if preproc_files:
    # Use the first preproc.h5 file found
    preproc_file = preproc_files[0]
    print(f"Found preprocessed file: {os.path.basename(preproc_file)}")
    
    # Load it
    preproc_data = files.read_h5(preproc_file)
    print("Successfully loaded preprocessed data!")
    
else:
    print("No files ending with 'preproc.h5' found in the directory")

spatial_activity = preproc_data['spatial_activity']
normalized_spatial_activity = preproc_data['norm_spatial_activity']
bin_centers = preproc_data['bin_centers']
reliable_cells = preproc_data['session_reliable_cells']
combined_reliable_cells = preproc_data['session_combined_reliable']

# RV.create_response_plot(normalized_spatial_activity,combined_reliable_cells)
RV.create_response_plot(spatial_activity,reliable_cells)

# Step 1: Shift to start at 0
shifted_centers = bin_centers - np.min(bin_centers)

# Step 2: Scale to match the actual physical distance 
actual_corridor_length = np.size(bin_centers)  # cm
unity_corridor_length = np.max(shifted_centers)
scaled_bin_centers = shifted_centers * (actual_corridor_length / unity_corridor_length)

results = SMI.analyze_spatial_modulation_improved(
    spatial_activity=spatial_activity,
    # spatial_activity=normalized_spatial_activity,
    bin_centers=scaled_bin_centers,
    reliable_cells=reliable_cells,
    segment_distance=28,
    exclude_start_cm=15,  # 15cm from beginning
    exclude_end_cm=8,     # 7cm from end
    smoothing_sigma=1.0
)

# Extract the SMI values for valid cells
SMI_values = results['smi_results']['SMI']
reliable_valid_cells = results['smi_results']['reliable_valid_cells']
reliable_valid_SMI = SMI_values[reliable_valid_cells]

# Remove any NaN or Inf values if present
reliable_valid_SMI = reliable_valid_SMI[~np.isnan(reliable_valid_SMI) & ~np.isinf(reliable_valid_SMI)]
print("")
print(f"Number of total cells: " f"{len(SMI_values)}" " and number of reliable and valid cells: " f"{len(reliable_valid_SMI)}")

# Calculate summary statistics
median_SMI = np.median(reliable_valid_SMI)
mad_SMI = stats.median_abs_deviation(reliable_valid_SMI)
print(f"Median SMI ± MAD: {median_SMI:.2f} ± {mad_SMI:.2f}")

# Statistical test (Wilcoxon signed-rank test against 0)
stat, p_value = stats.wilcoxon(reliable_valid_SMI)
print(f"Wilcoxon test against SMI=0: p-value = {p_value:.2e}")

# Create cumulative distribution plot with full SMI range (including negative values)
plt.figure(figsize=(8, 6))
x_sorted = np.sort(reliable_valid_SMI)  # Keep original SMI values (including negatives)
# # remove all negative values from reliable_valid_SMI
# reliable_valid_SMI = reliable_valid_SMI[reliable_valid_SMI >= 0]
# x_sortefd = np.sort(reliable_valid_SMI)  # Keep original SMI values (exluding negatives)
y_cumulative = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

plt.plot(x_sorted, y_cumulative, 'k-', linewidth=2)

# Add reference lines
plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)

plt.xlabel('Spatial modulation index')
plt.ylabel('Cumul. probability')
plt.title('Cumulative Distribution of Spatial Modulation Index')
plt.xlim(-1, 1)
plt.ylim(0, 1)
plt.grid(False)
plt.tight_layout()
plt.show()

# Print proportion of cells with different modulation patterns
prop_positive = np.mean(reliable_valid_SMI > 0)
prop_negative = np.mean(reliable_valid_SMI < 0)
prop_strong_pos = np.mean(reliable_valid_SMI > 0.5)
print(f"Proportion of cells with positive modulation (SMI > 0): {prop_positive:.2f} ({prop_positive*100:.1f}%)")
print(f"Proportion of cells with negative modulation (SMI < 0): {prop_negative:.2f} ({prop_negative*100:.1f}%)")
print(f"Proportion of cells with strong positive modulation (SMI > 0.5): {prop_strong_pos:.2f} ({prop_strong_pos*100:.1f}%)")

print("entering the layer-specific analysis..")

# Layer-specific analysis
# filepath = r"F:\2P\spmod\250811_JSY_JSY044_SpatialModulation_Day1\TSeries-08112025-1505-001"
# twoP_filename should be a string after the last dash in data_filepath
twoP_filename = data_filepath.split('\\')[-1]

twoP_data = {}
raw_twop_data = TwoP(data_filepath, twoP_filename)
raw_twop_data.find_files()
twop_dict = raw_twop_data.calc_dFF()

twoP_data['stat'] = twop_dict['stat'].copy()
twoP_data['ops'] = twop_dict['ops'].copy()

numCells = len(twoP_data['stat'])

im = np.zeros((twoP_data['ops']['Ly'], twoP_data['ops']['Lx']))  # Create an empty image
for n in range(0, numCells):
    ypix = twoP_data['stat'][n]['ypix'][~twoP_data['stat'][n]['overlap']]
    xpix = twoP_data['stat'][n]['xpix'][~twoP_data['stat'][n]['overlap']]
    im[ypix, xpix] = xpix  # Assign xpix values to im for progressive color change along x-axis

# Extract the median coordinates of each cell
med_coords = np.array([cell['med'] for cell in twoP_data['stat']])
layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)
SMI_Layer.plot_layer_distribution(med_coords, layer_cells, reliable_cells,im)
plt.show()
print("extracted the median coordinates..")

layer_results, layer_cells = SMI_Layer.run_layer_SMI_analysis(
    smi_results=results['smi_results'],
    reliable_cells=reliable_valid_cells,
    med_coords=med_coords,
    layer_cells=layer_cells,
    normalized_spatial_activity=normalized_spatial_activity,
    bin_centers=scaled_bin_centers

)
print(bin_centers)
    
results = [None] * 2  # Initialize list with 4 None elements
numTrials = np.shape(normalized_spatial_activity)[1]
numTriQ = numTrials//2
for i in range(2):
    if i == 0:
        new_normalized_spatial_activity = normalized_spatial_activity[:, 0:numTriQ, :]
    else:
        new_normalized_spatial_activity = normalized_spatial_activity[:, (i)*numTriQ:(i+1)*numTriQ, :]

    RV.create_response_plot(new_normalized_spatial_activity,reliable_cells)
    plt.show()
    results[i] = SMI.analyze_spatial_modulation_improved(
    spatial_activity=new_normalized_spatial_activity,
    bin_centers=scaled_bin_centers,
    reliable_cells=reliable_cells,
    segment_distance=28,
    exclude_start_cm=20,  # 15cm from beginning
    exclude_end_cm=7,     # 7cm from end
    smoothing_sigma=1.0
)
    # results[i] = SMI.analyze_spatial_modulation_BBBB(new_normalized_spatial_activity, scaled_bin_centers, preproc_data['combined_reliable'], avg_cc=preproc_data['avg_cc'], cohens_d=preproc_data['cohen_d'],
    #                                 segment_distance=segment_distance, exclude_boundary_cm=exclude_boundary_cm)
    print(f"Quarter {i} completed")
    plt.close('all')
        
    # Extract the SMI values for valid cells
    SMI_values = results[i]['smi_results']['SMI']
    reliable_valid_cells = results[i]['smi_results']['reliable_valid_cells']
    reliable_valid_SMI = SMI_values[reliable_valid_cells]

    # Remove any NaN or Inf values if present
    reliable_valid_SMI = reliable_valid_SMI[~np.isnan(reliable_valid_SMI) & ~np.isinf(reliable_valid_SMI)]
    print("")
    print(f"Number of total cells: " f"{len(SMI_values)}" " and number of reliable and valid cells: " f"{len(reliable_valid_SMI)}")

    # Calculate summary statistics
    median_SMI = np.median(reliable_valid_SMI)
    mad_SMI = stats.median_abs_deviation(reliable_valid_SMI)
    print(f"Median SMI ± MAD: {median_SMI:.2f} ± {mad_SMI:.2f}")

    # Statistical test (Wilcoxon signed-rank test against 0)
    stat, p_value = stats.wilcoxon(reliable_valid_SMI)
    print(f"Wilcoxon test against SMI=0: p-value = {p_value:.2e}")

    # Create cumulative distribution plot with full SMI range (including negative values)
    plt.figure(figsize=(8, 6))
    x_sorted = np.sort(reliable_valid_SMI)  # Keep original SMI values (including negatives)
    # # remove all negative values from reliable_valid_SMI
    # reliable_valid_SMI = reliable_valid_SMI[reliable_valid_SMI >= 0]
    # x_sortefd = np.sort(reliable_valid_SMI)  # Keep original SMI values (exluding negatives)
    y_cumulative = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    plt.plot(x_sorted, y_cumulative, 'k-', linewidth=2)

    # Add reference lines
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.7)

    plt.xlabel('Spatial modulation index')
    plt.ylabel('Cumul. probability')
    plt.title('Cumulative Distribution of Spatial Modulation Index')
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Print proportion of cells with different modulation patterns
    prop_positive = np.mean(reliable_valid_SMI > 0)
    prop_negative = np.mean(reliable_valid_SMI < 0)
    prop_strong_pos = np.mean(reliable_valid_SMI > 0.5)
    print(f"Proportion of cells with positive modulation (SMI > 0): {prop_positive:.2f} ({prop_positive*100:.1f}%)")
    print(f"Proportion of cells with negative modulation (SMI < 0): {prop_negative:.2f} ({prop_negative*100:.1f}%)")
    print(f"Proportion of cells with strong positive modulation (SMI > 0.5): {prop_strong_pos:.2f} ({prop_strong_pos*100:.1f}%)")

    print("entering the layer-specific analysis..")

    # Layer-specific analysis
    twoP_filename = data_filepath.split('\\')[-1]
    twoP_data = {}
    raw_twop_data = TwoP(data_filepath, twoP_filename)
    raw_twop_data.find_files()
    twop_dict = raw_twop_data.calc_dFF()

    twoP_data['stat'] = twop_dict['stat'].copy()
    twoP_data['ops'] = twop_dict['ops'].copy()

    numCells = len(twoP_data['stat'])

    im = np.zeros((twoP_data['ops']['Ly'], twoP_data['ops']['Lx']))  # Create an empty image
    for n in range(0, numCells):
        ypix = twoP_data['stat'][n]['ypix'][~twoP_data['stat'][n]['overlap']]
        xpix = twoP_data['stat'][n]['xpix'][~twoP_data['stat'][n]['overlap']]
        im[ypix, xpix] = xpix  # Assign xpix values to im for progressive color change along x-axis

    # Extract the median coordinates of each cell
    med_coords = np.array([cell['med'] for cell in twoP_data['stat']])
    layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)
    SMI_Layer.plot_layer_distribution(med_coords, layer_cells, reliable_cells,im)
    plt.show()
    print("extracted the median coordinates..")

    layer_results, layer_cells = SMI_Layer.run_layer_SMI_analysis(
        smi_results=results[i]['smi_results'],
        reliable_cells=reliable_valid_cells,
        med_coords=med_coords,
        layer_cells=layer_cells,
        normalized_spatial_activity=normalized_spatial_activity,
        bin_centers=scaled_bin_centers

    )
    print(bin_centers)