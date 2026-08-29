"""
SpeedModulation_Run_Speed_Tuning_by_Layer_Bins.py

Run speed tuning analysis using n SPEED BINS with correlation-based modulation.
More fine-grained than the 3-bin approach.

JSY, 2025
"""
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from scipy import stats
from tqdm import tqdm
from helper import files, TwoP
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer

# =============================================================================
# CONFIGURATION
# =============================================================================
data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002"
preproc_file = os.path.join(data_filepath, "10092025_JSY038_preproc.h5")

output_dir = os.path.join(data_filepath, "speed_tuning_analysis_12bins")
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# ANALYSIS PARAMETERS
# =============================================================================
N_SPEED_BINS = 12
MIN_SPEED = 2.0      # cm/s - exclude stationary
MAX_SPEED = 40.0     # cm/s - upper limit
MIN_FRAMES_PER_BIN = 20  # Minimum frames required in each bin
P_THRESHOLD = 0.05
N_PERMUTATIONS = 300
CORRELATION_THRESHOLD = 0.15  # Minimum |correlation| to be considered modulated

# =============================================================================
# STEP 1: LOAD PREPROCESSED DATA
# =============================================================================
print("="*80)
print("LOADING PREPROCESSED DATA")
print("="*80)

preproc_data = files.read_h5(preproc_file)
print(f"Loaded: {preproc_file}")

# =============================================================================
# STEP 2: EXTRACT REQUIRED DATA
# =============================================================================
print("\n" + "="*80)
print("EXTRACTING REQUIRED DATA FOR SPEED ANALYSIS")
print("="*80)

spike_data = preproc_data['smoothed_spks_temporal']
speed_data = preproc_data['speed_cm_s']
location_data = preproc_data['location_cm']
lap_starts = preproc_data['lap_starts']
lap_ends = preproc_data['lap_ends']

if 'reliable_cells' not in preproc_data:
    reliable_cells = np.ones(spike_data.shape[0], dtype=bool)
else:
    reliable_cells = preproc_data['reliable_cells'].astype(bool)

if 'processing_params' in preproc_data and 'framerate' in preproc_data['processing_params']:
    framerate = float(preproc_data['processing_params']['framerate'])
else:
    framerate = 10.0

print(f"✓ Spike data: {spike_data.shape}")
print(f"✓ Speed data: {speed_data.shape}")
print(f"✓ Reliable cells: {np.sum(reliable_cells)}/{len(reliable_cells)}")
print(f"✓ Framerate: {framerate} Hz")
print(f"✓ Speed range: {np.min(speed_data):.2f} to {np.max(speed_data):.2f} cm/s")

# =============================================================================
# STEP 3: IDENTIFY LAYERS
# =============================================================================
print("\n" + "="*80)
print("IDENTIFYING CORTICAL LAYERS")
print("="*80)

twoP_filename = os.path.basename(data_filepath)
raw_twop_data = TwoP(data_filepath, twoP_filename)
raw_twop_data.find_files()
twop_dict = raw_twop_data.calc_dFF()

med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)

print("\nLayer assignment:")
for layer_name, cell_indices in layer_cells.items():
    n_cells = len(cell_indices)
    n_reliable = np.sum(reliable_cells[cell_indices])
    print(f"  {layer_name}: {n_cells} total cells, {n_reliable} reliable")

# =============================================================================
# STEP 4: SPEED BINNING
# =============================================================================
print("\n" + "="*80)
print(f"CREATING {N_SPEED_BINS} SPEED BINS")
print("="*80)

# Create speed bins
speed_bins = np.linspace(MIN_SPEED, MAX_SPEED, N_SPEED_BINS + 1)
speed_bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2

print(f"Speed bins (cm/s): {speed_bins}")
print(f"Bin centers: {speed_bin_centers}")

# Assign each frame to a speed bin
speed_bin_idx = np.digitize(speed_data, speed_bins) - 1
speed_bin_idx[speed_data < MIN_SPEED] = -1  # Exclude slow speeds
speed_bin_idx[speed_data > MAX_SPEED] = -1  # Exclude very fast speeds
speed_bin_idx[speed_bin_idx >= N_SPEED_BINS] = -1  # Cap at max bin

# Count frames per bin
frames_per_bin = np.array([np.sum(speed_bin_idx == i) for i in range(N_SPEED_BINS)])
print(f"\nFrames per bin:")
for i, (center, count) in enumerate(zip(speed_bin_centers, frames_per_bin)):
    print(f"  Bin {i+1} ({center:.1f} cm/s): {count} frames")

# Check if we have enough data
valid_bins = frames_per_bin >= MIN_FRAMES_PER_BIN
n_valid_bins = np.sum(valid_bins)
print(f"\n✓ {n_valid_bins}/{N_SPEED_BINS} bins have ≥{MIN_FRAMES_PER_BIN} frames")

if n_valid_bins < 3:
    print("⚠️  WARNING: Very few valid bins. Consider using 3-bin method instead.")

# =============================================================================
# STEP 5: CALCULATE SPEED TUNING FOR EACH CELL
# =============================================================================
print("\n" + "="*80)
print("CALCULATING SPEED TUNING CURVES")
print("="*80)

reliable_indices = np.where(reliable_cells)[0]
n_reliable = len(reliable_indices)

# Storage for results
speed_tuning_curves = np.full((n_reliable, N_SPEED_BINS), np.nan)
speed_correlations = np.zeros(n_reliable)
speed_correlation_pvals = np.ones(n_reliable)
is_modulated = np.zeros(n_reliable, dtype=bool)

print(f"Analyzing {n_reliable} reliable cells...")

for idx, cell_idx in enumerate(tqdm(reliable_indices, desc="Computing tuning curves")):
    cell_activity = spike_data[cell_idx, :]
    
    # Calculate mean activity in each speed bin
    for bin_i in range(N_SPEED_BINS):
        bin_mask = speed_bin_idx == bin_i
        if np.sum(bin_mask) >= MIN_FRAMES_PER_BIN:
            speed_tuning_curves[idx, bin_i] = np.mean(cell_activity[bin_mask])
    
    # Calculate correlation between speed and activity
    # Only use frames within our speed range
    valid_frames = speed_bin_idx >= 0
    
    if np.sum(valid_frames) > MIN_FRAMES_PER_BIN:
        corr, pval = stats.spearmanr(
            speed_data[valid_frames],
            cell_activity[valid_frames]
        )
        speed_correlations[idx] = corr
        speed_correlation_pvals[idx] = pval

# =============================================================================
# STEP 6: PERMUTATION TEST FOR SIGNIFICANCE
# =============================================================================
print("\n" + "="*80)
print("RUNNING PERMUTATION TESTS")
print("="*80)

print(f"Running {N_PERMUTATIONS} permutations per cell...")

for idx in tqdm(range(n_reliable), desc="Permutation tests"):
    cell_idx = reliable_indices[idx]
    cell_activity = spike_data[cell_idx, :]
    
    valid_frames = speed_bin_idx >= 0
    if np.sum(valid_frames) < MIN_FRAMES_PER_BIN:
        continue
    
    observed_corr = speed_correlations[idx]
    
    # Permutation test
    null_corrs = np.zeros(N_PERMUTATIONS)
    for perm_i in range(N_PERMUTATIONS):
        # Shuffle activity
        shuffled_activity = cell_activity[valid_frames].copy()
        np.random.shuffle(shuffled_activity)
        
        # Calculate correlation with shuffled data
        null_corrs[perm_i], _ = stats.spearmanr(
            speed_data[valid_frames],
            shuffled_activity
        )
    
    # Calculate p-value (two-tailed)
    p_value = np.mean(np.abs(null_corrs) >= np.abs(observed_corr))
    speed_correlation_pvals[idx] = p_value
    
    # Determine if modulated
    is_modulated[idx] = (p_value < P_THRESHOLD) and (np.abs(observed_corr) > CORRELATION_THRESHOLD)

# =============================================================================
# STEP 7: ORGANIZE RESULTS BY LAYER
# =============================================================================
print("\n" + "="*80)
print("ORGANIZING RESULTS BY LAYER")
print("="*80)

results = {
    'overall': {
        'n_total': n_reliable,
        'n_modulated': int(np.sum(is_modulated)),
        'prop_modulated': float(np.mean(is_modulated)),
        'n_positive': int(np.sum((is_modulated) & (speed_correlations > 0))),
        'n_negative': int(np.sum((is_modulated) & (speed_correlations < 0))),
        'mean_correlation': float(np.mean(speed_correlations[is_modulated])) if np.any(is_modulated) else 0,
        'median_correlation': float(np.median(speed_correlations[is_modulated])) if np.any(is_modulated) else 0,
        'reliable_indices': reliable_indices,
        'speed_correlations': speed_correlations,
        'p_values': speed_correlation_pvals,
        'is_modulated': is_modulated,
        'speed_tuning_curves': speed_tuning_curves,
    },
    'layer_results': {},
    'parameters': {
        'n_speed_bins': N_SPEED_BINS,
        'min_speed': MIN_SPEED,
        'max_speed': MAX_SPEED,
        'speed_bins': speed_bins,
        'speed_bin_centers': speed_bin_centers,
        'min_frames_per_bin': MIN_FRAMES_PER_BIN,
        'p_threshold': P_THRESHOLD,
        'correlation_threshold': CORRELATION_THRESHOLD,
        'n_permutations': N_PERMUTATIONS,
    }
}

# Layer-specific results
for layer_name, layer_cell_indices in layer_cells.items():
    # Find which reliable cells are in this layer
    layer_reliable_mask = np.isin(reliable_indices, layer_cell_indices)
    layer_reliable_indices = reliable_indices[layer_reliable_mask]
    
    n_layer_total = np.sum(layer_reliable_mask)
    layer_is_modulated = is_modulated[layer_reliable_mask]
    layer_correlations = speed_correlations[layer_reliable_mask]
    
    n_modulated = np.sum(layer_is_modulated)
    n_positive = np.sum((layer_is_modulated) & (layer_correlations > 0))
    n_negative = np.sum((layer_is_modulated) & (layer_correlations < 0))
    
    results['layer_results'][layer_name] = {
        'n_total': int(n_layer_total),
        'n_modulated': int(n_modulated),
        'prop_modulated': float(n_modulated / n_layer_total) if n_layer_total > 0 else 0,
        'n_positive': int(n_positive),
        'n_negative': int(n_negative),
        'mean_correlation': float(np.mean(layer_correlations[layer_is_modulated])) if n_modulated > 0 else 0,
        'cell_indices': layer_reliable_indices,
        'correlations': layer_correlations,
    }

# Chi-square test
layer_names = ['L2/3', 'L4', 'L5', 'L6']
observed = [results['layer_results'][ln]['n_modulated'] for ln in layer_names if ln in results['layer_results']]
total = [results['layer_results'][ln]['n_total'] for ln in layer_names if ln in results['layer_results']]

if len(observed) > 1 and all(t > 0 for t in total):
    chi2, p_value = stats.chi2_contingency([observed, np.array(total) - np.array(observed)])[:2]
    results['chi2'] = float(chi2)
    results['p_value'] = float(p_value)
else:
    results['chi2'] = None
    results['p_value'] = None

# Print results
print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"\nOverall:")
print(f"  Total reliable: {results['overall']['n_total']}")
print(f"  Speed-modulated: {results['overall']['n_modulated']} ({results['overall']['prop_modulated']*100:.1f}%)")
print(f"  Positive correlation: {results['overall']['n_positive']}")
print(f"  Negative correlation: {results['overall']['n_negative']}")
print(f"  Mean correlation: {results['overall']['mean_correlation']:.3f}")

print("\nLayer-specific:")
for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
    if layer_name in results['layer_results']:
        lr = results['layer_results'][layer_name]
        print(f"\n{layer_name}:")
        print(f"  Total: {lr['n_total']}")
        print(f"  Modulated: {lr['n_modulated']} ({lr['prop_modulated']*100:.1f}%)")
        print(f"  Positive: {lr['n_positive']}, Negative: {lr['n_negative']}")
        print(f"  Mean correlation: {lr['mean_correlation']:.3f}")

if results['chi2'] is not None:
    print(f"\nChi-square test: χ² = {results['chi2']:.3f}, p = {results['p_value']:.6f}")
    if results['p_value'] < 0.05:
        print("  *** Significant difference across layers")

# =============================================================================
# STEP 8: GENERATE FIGURES
# =============================================================================
print("\n" + "="*80)
print("GENERATING FIGURES")
print("="*80)

# Figure 1: Overall speed tuning curves
fig1, axes = plt.subplots(2, 2, figsize=(12, 10))
colors = {'L2/3': '#4472C4', 'L4': '#ED7D31', 'L5': '#70AD47', 'L6': '#C5504B'}

# Panel A: Example tuning curves
ax = axes[0, 0]
for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
    if layer_name in results['layer_results']:
        lr = results['layer_results'][layer_name]
        layer_idx = np.isin(reliable_indices, lr['cell_indices'])
        layer_modulated = is_modulated[layer_idx]
        
        if np.any(layer_modulated):
            layer_curves = speed_tuning_curves[layer_idx][layer_modulated]
            mean_curve = np.nanmean(layer_curves, axis=0)
            sem_curve = np.nanstd(layer_curves, axis=0) / np.sqrt(np.sum(layer_modulated))
            
            ax.plot(speed_bin_centers, mean_curve, 'o-', color=colors[layer_name], 
                   label=f"{layer_name} (n={np.sum(layer_modulated)})", linewidth=2)
            ax.fill_between(speed_bin_centers, mean_curve - sem_curve, mean_curve + sem_curve,
                           alpha=0.2, color=colors[layer_name])

ax.set_xlabel('Speed (cm/s)', fontweight='bold')
ax.set_ylabel('Normalized Activity', fontweight='bold')
ax.set_title('A. Mean Speed Tuning by Layer', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel B: Proportion modulated
ax = axes[0, 1]
layer_names_plot = []
props = []
for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
    if layer_name in results['layer_results']:
        layer_names_plot.append(layer_name)
        props.append(results['layer_results'][layer_name]['prop_modulated'] * 100)

ax.bar(range(len(layer_names_plot)), props, color=[colors[ln] for ln in layer_names_plot], alpha=0.7)
ax.set_xticks(range(len(layer_names_plot)))
ax.set_xticklabels(layer_names_plot)
ax.set_ylabel('% Speed-Modulated', fontweight='bold')
ax.set_title('B. Proportion Modulated by Layer', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Panel C: Correlation distribution
ax = axes[1, 0]
for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
    if layer_name in results['layer_results']:
        lr = results['layer_results'][layer_name]
        layer_idx = np.isin(reliable_indices, lr['cell_indices'])
        layer_modulated = is_modulated[layer_idx]
        layer_corrs = speed_correlations[layer_idx][layer_modulated]
        
        if len(layer_corrs) > 0:
            ax.hist(layer_corrs, bins=20, alpha=0.5, label=layer_name, color=colors[layer_name])

ax.axvline(0, color='black', linestyle='--', alpha=0.5)
ax.set_xlabel('Speed Correlation', fontweight='bold')
ax.set_ylabel('Number of Cells', fontweight='bold')
ax.set_title('C. Distribution of Speed Correlations', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Panel D: Positive vs Negative
ax = axes[1, 1]
x = np.arange(len(layer_names_plot))
width = 0.35

pos_counts = [results['layer_results'][ln]['n_positive'] for ln in layer_names_plot]
neg_counts = [results['layer_results'][ln]['n_negative'] for ln in layer_names_plot]

ax.bar(x - width/2, pos_counts, width, label='Positive', color='red', alpha=0.7)
ax.bar(x + width/2, neg_counts, width, label='Negative', color='blue', alpha=0.7)
ax.set_xlabel('Layer', fontweight='bold')
ax.set_ylabel('Number of Cells', fontweight='bold')
ax.set_title('D. Positive vs Negative Modulation', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(layer_names_plot)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Figure1_SpeedTuning_10bins.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Figure1_SpeedTuning_10bins.pdf'), bbox_inches='tight')
plt.close()

print("✓ Figure 1 saved")

# =============================================================================
# STEP 9: SAVE RESULTS
# =============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save detailed HDF5
detailed_file = os.path.join(output_dir, "speed_modulation_10bins_DETAILED.h5")

save_dict = {
    # Session metadata
    'session_date': os.path.basename(data_filepath),
    'session_path': data_filepath,
    'n_laps': len(lap_starts),
    'n_frames': len(speed_data),
    'n_total_cells': spike_data.shape[0],
    'n_reliable_cells': np.sum(reliable_cells),
    'framerate': framerate,
    
    # Speed statistics
    'speed_range_min': float(np.min(speed_data)),
    'speed_range_max': float(np.max(speed_data)),
    'n_speed_bins': N_SPEED_BINS,
    'speed_bins': np.array(speed_bins),
    'speed_bin_centers': np.array(speed_bin_centers),
    'frames_per_bin': np.array(frames_per_bin),
    
    # Overall results
    'overall_n_total': results['overall']['n_total'],
    'overall_n_modulated': results['overall']['n_modulated'],
    'overall_prop_modulated': results['overall']['prop_modulated'],
    'overall_n_positive': results['overall']['n_positive'],
    'overall_n_negative': results['overall']['n_negative'],
    'overall_mean_correlation': results['overall']['mean_correlation'],
    
    # Cell-by-cell data
    'cell_ids_reliable': np.array(reliable_indices),
    'cell_speed_correlations': np.array(speed_correlations),
    'cell_p_values': np.array(speed_correlation_pvals),
    'cell_is_modulated': np.array(is_modulated),
    'cell_speed_tuning_curves': np.array(speed_tuning_curves),
    'cell_coordinates': np.array(med_coords),
    'cell_coordinates_reliable': np.array(med_coords[reliable_indices]),
    
    # Layer boundaries
    'layer_boundaries': np.array([
        layer_boundaries['L2/3'][0], layer_boundaries['L2/3'][1],
        layer_boundaries['L4'][0], layer_boundaries['L4'][1],
        layer_boundaries['L5'][0], layer_boundaries['L5'][1],
        layer_boundaries['L6'][0], layer_boundaries['L6'][1]
    ]),
    
    # Statistical tests
    'chi2_statistic': results['chi2'] if results['chi2'] is not None else -1,
    'chi2_p_value': results['p_value'] if results['p_value'] is not None else -1,
    
    # Parameters
    'param_n_permutations': N_PERMUTATIONS,
    'param_p_threshold': P_THRESHOLD,
    'param_correlation_threshold': CORRELATION_THRESHOLD,
    'param_min_frames_per_bin': MIN_FRAMES_PER_BIN,
}

# Add cell layer assignments
cell_layer_assignments = np.array([''] * n_reliable, dtype='<U4')
for idx, cell_idx in enumerate(reliable_indices):
    for layer_name, layer_cell_indices in layer_cells.items():
        if cell_idx in layer_cell_indices:
            cell_layer_assignments[idx] = layer_name
            break
save_dict['cell_layer_assignments'] = cell_layer_assignments

# Add layer-specific data
for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
    if layer_name in results['layer_results']:
        lr = results['layer_results'][layer_name]
        prefix = layer_name.replace('/', '_')
        
        save_dict[f'{prefix}_n_total'] = lr['n_total']
        save_dict[f'{prefix}_n_modulated'] = lr['n_modulated']
        save_dict[f'{prefix}_prop_modulated'] = lr['prop_modulated']
        save_dict[f'{prefix}_n_positive'] = lr['n_positive']
        save_dict[f'{prefix}_n_negative'] = lr['n_negative']
        save_dict[f'{prefix}_mean_correlation'] = lr['mean_correlation']

files.write_h5(detailed_file, save_dict)
print(f"✓ Detailed results saved to: {detailed_file}")

# Save text summary
summary_file = os.path.join(output_dir, "speed_modulation_10bins_summary.txt")
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("SPEED MODULATION ANALYSIS - 10 BINS METHOD\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Session: {os.path.basename(data_filepath)}\n")
    f.write(f"Total cells: {spike_data.shape[0]}\n")
    f.write(f"Reliable cells: {np.sum(reliable_cells)}\n\n")
    
    f.write("PARAMETERS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Number of speed bins: {N_SPEED_BINS}\n")
    f.write(f"  Speed range: {MIN_SPEED} - {MAX_SPEED} cm/s\n")
    f.write(f"  Min frames per bin: {MIN_FRAMES_PER_BIN}\n")
    f.write(f"  Correlation threshold: {CORRELATION_THRESHOLD}\n")
    f.write(f"  P-value threshold: {P_THRESHOLD}\n")
    f.write(f"  Permutations: {N_PERMUTATIONS}\n\n")
    
    f.write("OVERALL RESULTS:\n")
    f.write("-"*80 + "\n")
    f.write(f"  Speed-modulated: {results['overall']['n_modulated']}/{results['overall']['n_total']} ")
    f.write(f"({results['overall']['prop_modulated']*100:.1f}%)\n")
    f.write(f"  Positive: {results['overall']['n_positive']}\n")
    f.write(f"  Negative: {results['overall']['n_negative']}\n")
    f.write(f"  Mean correlation: {results['overall']['mean_correlation']:.3f}\n\n")
    
    f.write("LAYER-SPECIFIC RESULTS:\n")
    f.write("-"*80 + "\n")
    for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
        if layer_name in results['layer_results']:
            lr = results['layer_results'][layer_name]
            f.write(f"\n{layer_name}:\n")
            f.write(f"  Total: {lr['n_total']}\n")
            f.write(f"  Modulated: {lr['n_modulated']} ({lr['prop_modulated']*100:.1f}%)\n")
            f.write(f"  Positive: {lr['n_positive']}, Negative: {lr['n_negative']}\n")
            f.write(f"  Mean correlation: {lr['mean_correlation']:.3f}\n")

print(f"✓ Summary saved to: {summary_file}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nOutputs saved to: {output_dir}")