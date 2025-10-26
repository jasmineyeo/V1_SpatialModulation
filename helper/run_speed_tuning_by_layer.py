"""
run_speed_tuning_by_layer.py

Example script showing how to run speed tuning analysis
after preprocessing and layer identification.

JSY, 2025
"""
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
from helper import files, TwoP
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
from helper.SpeedTuningAnalysis import SpeedTuningAnalysis

# =============================================================================
# CONFIGURATION
# =============================================================================
# Path to preprocessed session data
data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001"
preproc_file = os.path.join(data_filepath, "10102025_JSY038_preproc.h5")

# Output directory for speed analysis results
output_dir = os.path.join(data_filepath, "speed_tuning_analysis")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# =============================================================================
print("="*80)
print("LOADING PREPROCESSED DATA")
print("="*80)

preproc_data = files.read_h5(preproc_file)
print(f"Loaded: {preproc_file}")

# Check what's available
print("\nAvailable data fields:")
for key in preproc_data.keys():
    if isinstance(preproc_data[key], np.ndarray):
        print(f"  {key}: {preproc_data[key].shape}")
    else:
        print(f"  {key}: {type(preproc_data[key])}")

# =============================================================================
# STEP 2: EXTRACT REQUIRED DATA
# =============================================================================
print("\n" + "="*80)
print("EXTRACTING REQUIRED DATA FOR SPEED ANALYSIS")
print("="*80)

# Get smoothed spike data - this should be the temporal spike data, not spatial_activity
# We need to reconstruct it from spatial_activity or use a different approach

# CRITICAL: Load the TEMPORAL spike data, not spatial!
if 'smoothed_spks_temporal' in preproc_data:
    spike_data = preproc_data['smoothed_spks_temporal']
    print(f"\n✓ Using temporal smoothed spikes: {spike_data.shape}")
    
    # Verify it's 2D (cells × frames)
    if spike_data.ndim != 2:
        raise ValueError(f"temporal data should be 2D, got shape {spike_data.shape}")
        
elif 'smoothed_spks' in preproc_data and preproc_data['smoothed_spks'].ndim == 2:
    spike_data = preproc_data['smoothed_spks']
    print(f"\n✓ Using saved smoothed spikes: {spike_data.shape}")
    
else:
    raise ValueError(
        "No temporal spike data found!\n"
        "Available fields: " + ", ".join(preproc_data.keys()) + "\n"
        "You need to re-run Preprocess.py with the updated version that saves 'smoothed_spks_temporal'"
    )


# Get speed data
if 'speed_cm_s' not in preproc_data:
    raise ValueError("Missing speed_cm_s - re-run Preprocess.py")

speed_data = preproc_data['speed_cm_s']
print(f"✓ Speed data: {speed_data.shape}")

# Verify dimensions match
if spike_data.shape[1] != len(speed_data):
    raise ValueError(
        f"Dimension mismatch!\n"
        f"  Spike frames: {spike_data.shape[1]}\n"
        f"  Speed frames: {len(speed_data)}\n"
        f"  These must match! Re-run Preprocess.py to fix."
    )

# Get lap boundaries
if 'lap_starts' not in preproc_data or 'lap_ends' not in preproc_data:
    raise ValueError("Missing lap boundaries - re-run Preprocess.py")

lap_starts = preproc_data['lap_starts']
lap_ends = preproc_data['lap_ends']
print(f"✓ Lap boundaries: {len(lap_starts)} laps")

# Verify lap boundaries don't exceed data length
if lap_ends[-1] > len(speed_data):
    raise ValueError(
        f"Lap boundaries exceed data length!\n"
        f"  Last lap end: {lap_ends[-1]}\n"
        f"  Speed data length: {len(speed_data)}\n"
        f"  Re-run Preprocess.py to fix."
    )

print(f"\n✓ All data verified:")
print(f"  Spike data: {spike_data.shape[0]} cells × {spike_data.shape[1]} frames")
print(f"  Speed data: {len(speed_data)} frames")
print(f"  Lap boundaries: {len(lap_starts)} laps")
print(f"  Frame range: {lap_starts[0]} to {lap_ends[-1]}")

# Get reliable cells
reliable_cells = preproc_data['reliable_cells']
print(f"✓ Reliable cells: {np.sum(reliable_cells)}/{len(reliable_cells)}")

# Get framerate
framerate = preproc_data['processing_params']['framerate']
print(f"✓ Framerate: {framerate} Hz")

# Add this after loading data in run_speed_tuning_by_layer.py
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("DIAGNOSTIC: Checking speed-activity relationship")
print("="*80)

# Pick a few high-activity reliable cells
reliable_indices = np.where(reliable_cells)[0]
mean_activity = np.mean(spike_data[reliable_indices], axis=1)
top_cells = reliable_indices[np.argsort(mean_activity)[-5:]]  # Top 5 active cells

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i, cell_idx in enumerate(top_cells):
    if i >= 6:
        break
    
    ax = axes[i]
    
    # Scatter plot: speed vs activity
    # Sample 5000 random points for clarity
    n_sample = min(5000, len(speed_data))
    sample_idx = np.random.choice(len(speed_data), n_sample, replace=False)
    
    ax.scatter(speed_data[sample_idx], 
              spike_data[cell_idx, sample_idx],
              alpha=0.1, s=1)
    
    # Calculate correlation
    corr = np.corrcoef(speed_data, spike_data[cell_idx])[0, 1]
    
    ax.set_xlabel('Speed (cm/s)')
    ax.set_ylabel('Neural Activity')
    ax.set_title(f'Cell {cell_idx} (r={corr:.3f})')
    ax.set_xlim([0, 30])

plt.suptitle('Speed vs Neural Activity for Top 5 Active Cells')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostic_speed_activity.png'), dpi=150)
plt.show()

print(f"Diagnostic plot saved to: {output_dir}")

# Check speed distribution
print("\nSpeed distribution:")
print(f"  Mean: {np.mean(speed_data):.2f} cm/s")
print(f"  Median: {np.median(speed_data):.2f} cm/s")
print(f"  Range: {np.min(speed_data):.2f} to {np.max(speed_data):.2f} cm/s")
print(f"  % below 2 cm/s: {np.sum(speed_data < 2.0)/len(speed_data)*100:.1f}%")
print(f"  % above 10 cm/s: {np.sum(speed_data > 10.0)/len(speed_data)*100:.1f}%")

# Plot speed distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(speed_data, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Speed (cm/s)')
plt.ylabel('Count')
plt.title('Speed Distribution (Filtered Data)')
plt.axvline(2.0, color='r', linestyle='--', label='Min speed (preprocessing)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(speed_data[:1000], 'b-', alpha=0.7, linewidth=0.5)
plt.xlabel('Frame')
plt.ylabel('Speed (cm/s)')
plt.title('Speed Over Time (First 1000 frames)')
plt.axhline(2.0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'diagnostic_speed_distribution.png'), dpi=150)
plt.show()

# =============================================================================
# STEP 3: IDENTIFY LAYERS
# =============================================================================
print("\n" + "="*80)
print("IDENTIFYING CORTICAL LAYERS")
print("="*80)

# Load 2P data for layer identification
twoP_filename = os.path.basename(data_filepath)
raw_twop_data = TwoP(data_filepath, twoP_filename)
raw_twop_data.find_files()
twop_dict = raw_twop_data.calc_dFF()

# Extract median coordinates
med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
print(f"Cell coordinates extracted: {med_coords.shape}")

# Identify layers
layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)

print("\nLayer assignment:")
for layer_name, cell_indices in layer_cells.items():
    n_cells = len(cell_indices)
    n_reliable = np.sum(reliable_cells[cell_indices])
    print(f"  {layer_name}: {n_cells} total cells, {n_reliable} reliable")

# =============================================================================
# STEP 4: RUN SPEED TUNING ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("RUNNING SPEED TUNING ANALYSIS BY LAYER")
print("="*80)

speed_results = SpeedTuningAnalysis.analyze_speed_tuning_by_layer(
    spike_data=spike_data,
    speed_data=speed_data,
    lap_starts=lap_starts,
    lap_ends=lap_ends,
    layer_cells=layer_cells,
    reliable_cells=reliable_cells,
    framerate=framerate,
    min_speed=2.0,          # Stationary threshold (cm/s)
    max_speed=30.0,         # Maximum speed to consider (cm/s)
    n_bins=30,              # Number of speed bins (Saleem et al. 2013)
    Q_S_threshold=0.1       # Threshold for speed-tuned classification
)

# =============================================================================
# STEP 5: VISUALIZE RESULTS
# =============================================================================
print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

SpeedTuningAnalysis.plot_speed_tuning_results(
    speed_results, 
    save_dir=output_dir
)

# =============================================================================
# STEP 6: SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save results to file
results_file = os.path.join(output_dir, "speed_tuning_results.h5")
files.write_h5(results_file, speed_results)
print(f"✓ Results saved to: {results_file}")

# Create summary report
summary_file = os.path.join(output_dir, "speed_tuning_summary.txt")
with open(summary_file, 'w') as f:
    f.write("SPEED TUNING ANALYSIS SUMMARY\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"Session: {os.path.basename(data_filepath)}\n")
    f.write(f"Total cells: {spike_data.shape[0]}\n")
    f.write(f"Reliable cells: {np.sum(reliable_cells)}\n")
    f.write(f"Total laps: {len(lap_starts)}\n\n")
    
    f.write("LAYER-SPECIFIC RESULTS:\n")
    f.write("-"*60 + "\n")
    
    for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
        if layer_name in speed_results['layer_results']:
            lr = speed_results['layer_results'][layer_name]
            if lr is not None:
                f.write(f"\n{layer_name}:\n")
                f.write(f"  Total cells: {lr['n_cells']}\n")
                f.write(f"  Speed-tuned: {lr['n_tuned']} ({lr['prop_tuned']*100:.1f}%)\n")
                f.write(f"  Mean Q_S: {lr['mean_Q_S']:.3f}\n")
                f.write(f"  Median Q_S: {lr['median_Q_S']:.3f}\n")
                f.write(f"  Tuning types:\n")
                f.write(f"    Low-pass: {lr['n_low_pass']}\n")
                f.write(f"    Band-pass: {lr['n_band_pass']}\n")
                f.write(f"    High-pass: {lr['n_high_pass']}\n")

print(f"✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nCheck outputs in: {output_dir}")