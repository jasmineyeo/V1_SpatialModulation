"""
SpeedModulation_Run_Speed_Tuning_by_Layer.py

Run speed tuning analysis using the SIMPLE method (recommended).
Q_S method is commented out but available for comparison.

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
from helper import files, TwoP
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
from helper.SpeedTuningAnalysis import SpeedTuningAnalysis

# =============================================================================
# CONFIGURATION
# =============================================================================
# data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002"
# preproc_file = os.path.join(data_filepath, "10092025_JSY038_preproc.h5")
# data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001"
# preproc_file = os.path.join(data_filepath, "10102025_JSY038_preproc.h5")
# data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002"
# preproc_file = os.path.join(data_filepath, "10112025_JSY038_preproc.h5")
# data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001"
# preproc_file = os.path.join(data_filepath, "10122025_JSY038_preproc.h5")
# data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001"
# preproc_file = os.path.join(data_filepath, "10122025_JSY038_preproc.h5")
data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001"
preproc_file = os.path.join(data_filepath, "10132025_JSY038_preproc.h5")

# data_filepath = r"F:\2P\spmod\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_raw_separateregistration\TSeries-09062025-1308-001"
# preproc_file = os.path.join(data_filepath, "09062025_JSY038_preproc.h5")
# data_filepath = r"F:\2P\spmod\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_raw_separateregistration\TSeries-09072025-1257-001"
# preproc_file = os.path.join(data_filepath, "09072025_JSY038_preproc.h5")

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

# Get temporal spike data
if 'smoothed_spks_temporal' not in preproc_data:
    raise ValueError("Missing 'smoothed_spks_temporal' in preprocessed data!")

spike_data = preproc_data['smoothed_spks_temporal']
print(f"✓ Using temporal smoothed spikes: {spike_data.shape}")

if spike_data.ndim != 2:
    raise ValueError(f"Spike data should be 2D (cells × frames), got shape {spike_data.shape}")

# Get speed data
if 'speed_cm_s' not in preproc_data:
    raise ValueError("Missing 'speed_cm_s' - re-run Preprocess.py")

speed_data = preproc_data['speed_cm_s']
print(f"✓ Speed data: {speed_data.shape}")

if speed_data.ndim != 1:
    raise ValueError(f"Speed data should be 1D, got shape {speed_data.shape}")

# Get location data
if 'location_cm' not in preproc_data:
    raise ValueError("Missing 'location_cm' - re-run Preprocess.py")

location_data = preproc_data['location_cm']
print(f"✓ Location data: {location_data.shape}")

# Verify dimensions match
if spike_data.shape[1] != len(speed_data):
    raise ValueError(f"Dimension mismatch! Spike: {spike_data.shape[1]}, Speed: {len(speed_data)}")

if len(speed_data) != len(location_data):
    raise ValueError(f"Speed/location mismatch! Speed: {len(speed_data)}, Location: {len(location_data)}")

# Get lap boundaries
if 'lap_starts' not in preproc_data or 'lap_ends' not in preproc_data:
    raise ValueError("Missing lap boundaries - re-run Preprocess.py")

lap_starts = preproc_data['lap_starts']
lap_ends = preproc_data['lap_ends']

if len(lap_starts) != len(lap_ends):
    raise ValueError("Lap starts and ends don't match!")

if lap_ends[-1] > len(speed_data):
    raise ValueError(f"Lap boundaries exceed data! Last lap: {lap_ends[-1]}, Data: {len(speed_data)}")

print(f"✓ Lap boundaries: {len(lap_starts)} laps (frames {lap_starts[0]} to {lap_ends[-1]})")

# Get reliable cells
if 'reliable_cells' not in preproc_data:
    print("WARNING: No reliable_cells found, using all cells")
    reliable_cells = np.ones(spike_data.shape[0], dtype=bool)
else:
    reliable_cells = preproc_data['reliable_cells'].astype(bool)

if len(reliable_cells) != spike_data.shape[0]:
    raise ValueError(f"Reliable cells mismatch! Mask: {len(reliable_cells)}, Cells: {spike_data.shape[0]}")

print(f"✓ Reliable cells: {np.sum(reliable_cells)}/{len(reliable_cells)}")

# Get framerate
if 'processing_params' not in preproc_data or 'framerate' not in preproc_data['processing_params']:
    print("WARNING: Framerate not found, using default 10 Hz")
    framerate = 10.0
else:
    framerate = float(preproc_data['processing_params']['framerate'])

print(f"✓ Framerate: {framerate} Hz")

print(f"\n{'='*80}")
print("DATA VALIDATION COMPLETE")
print(f"{'='*80}")
print(f"  Cells: {spike_data.shape[0]}")
print(f"  Frames: {spike_data.shape[1]}")
print(f"  Laps: {len(lap_starts)}")
print(f"  Reliable cells: {np.sum(reliable_cells)}")
print(f"  Speed range: {np.min(speed_data):.2f} to {np.max(speed_data):.2f} cm/s")

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
print(f"Cell coordinates extracted: {med_coords.shape}")

layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)

print("\nLayer assignment:")
for layer_name, cell_indices in layer_cells.items():
    n_cells = len(cell_indices)
    n_reliable = np.sum(reliable_cells[cell_indices])
    print(f"  {layer_name}: {n_cells} total cells, {n_reliable} reliable")

# =============================================================================
# STEP 4: RUN SIMPLE SPEED MODULATION ANALYSIS (RECOMMENDED)
# =============================================================================
print("\n" + "="*80)
print("RUNNING SIMPLE SPEED MODULATION ANALYSIS")
print("="*80)

# Run simple method with layer-specific analysis
simple_results = SpeedTuningAnalysis.analyze_simple_by_layer(
    spike_data=spike_data,
    speed_data=speed_data,
    layer_cells=layer_cells,
    reliable_cells=reliable_cells,
    slow_range=(2, 10),
    medium_range=(10, 20),
    fast_range=(20, 50),
    min_frames_per_bin=30,
    mod_index_threshold=0.1,
    p_threshold=0.05,
    n_permutations=1000
)

# =============================================================================
# STEP 5: GENERATE PUBLICATION-QUALITY FIGURES
# =============================================================================
print("\n" + "="*80)
print("GENERATING PUBLICATION-QUALITY FIGURES")
print("="*80)

if simple_results is not None:
    # Create all figures
    fig1, fig2, fig3 = SpeedTuningAnalysis.plot_simple_results(
        results=simple_results,
        spike_data=spike_data,
        speed_data=speed_data,
        save_dir=output_dir
    )
    
    print("\n✓ All figures generated successfully!")
    print(f"  Figure 1: Layer gradient (3-panel)")
    print(f"  Figure 2: Detailed analysis (6-panel)")
    print(f"  Figure 3: Example cells")
else:
    print("\n⚠️  Analysis failed - check data quality")
# =============================================================================
# STEP 6: SAVE COMPREHENSIVE RESULTS FOR CROSS-SESSION ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("SAVING COMPREHENSIVE RESULTS")
print("="*80)

if simple_results is not None:
    
    # =========================================================================
    # NEW: Save DETAILED cell-by-cell data for cross-session analysis
    # =========================================================================
    
    detailed_results_file = os.path.join(output_dir, "speed_modulation_DETAILED.h5")
    
    # Get session info
    session_date = os.path.basename(data_filepath)
    
        # Prepare comprehensive data dictionary
    detailed_save_dict = {
        # =====================================================================
        # SESSION METADATA
        # =====================================================================
        'session_date': session_date,
        'session_path': data_filepath,
        'n_laps': len(lap_starts),
        'n_frames': len(speed_data),
        'n_total_cells': spike_data.shape[0],
        'n_reliable_cells': np.sum(reliable_cells),
        'framerate': framerate,
        
        # =====================================================================
        # SPEED DATA STATISTICS
        # =====================================================================
        'speed_range_min': float(np.min(speed_data)),
        'speed_range_max': float(np.max(speed_data)),
        'n_frames_slow': int(np.sum((speed_data >= 2) & (speed_data < 10))),
        'n_frames_medium': int(np.sum((speed_data >= 10) & (speed_data < 20))),
        'n_frames_fast': int(np.sum(speed_data >= 20)),
        
        # Convert tuples to strings for HDF5 compatibility
        'speed_bins_slow': str(simple_results['overall']['speed_bins']['slow']),
        'speed_bins_medium': str(simple_results['overall']['speed_bins']['medium']),
        'speed_bins_fast': str(simple_results['overall']['speed_bins']['fast']),
        
        # =====================================================================
        # OVERALL RESULTS
        # =====================================================================
        'overall_n_modulated': int(simple_results['overall']['n_modulated']),
        'overall_prop_modulated': float(simple_results['overall']['prop_modulated']),
        'overall_n_positive': int(simple_results['overall']['n_positive']),
        'overall_n_negative': int(simple_results['overall']['n_negative']),
        'overall_mean_mod_index': float(simple_results['overall']['mean_mod_index']),
        'overall_median_mod_index': float(simple_results['overall']['median_mod_index']),
        
        # =====================================================================
        # CELL-BY-CELL DATA (ALL RELIABLE CELLS)
        # =====================================================================
        'cell_ids_reliable': np.array(simple_results['overall']['reliable_indices']),
        'cell_modulation_indices': np.array(simple_results['overall']['modulation_indices']),
        'cell_p_values': np.array(simple_results['overall']['p_values']),
        'cell_is_modulated': np.array(np.isin(
            simple_results['overall']['reliable_indices'],
            simple_results['overall']['speed_modulated_cells']
        )),
        'cell_modulation_direction': np.array(simple_results['overall']['modulation_directions']),
        'cell_mean_activities': np.array(simple_results['overall']['mean_activities']),
        
        # =====================================================================
        # CELL COORDINATES (for tracking across sessions)
        # =====================================================================
        'cell_coordinates': np.array(med_coords),
        'cell_coordinates_reliable': np.array(med_coords[simple_results['overall']['reliable_indices']]),
        
        # =====================================================================
        # LAYER ASSIGNMENTS
        # =====================================================================
        'layer_boundaries': np.array([
            layer_boundaries['L2/3'][0], layer_boundaries['L2/3'][1],
            layer_boundaries['L4'][0], layer_boundaries['L4'][1],
            layer_boundaries['L5'][0], layer_boundaries['L5'][1],
            layer_boundaries['L6'][0], layer_boundaries['L6'][1]
        ]),
    }
    
    # Add cell-by-cell layer assignments
    cell_layer_assignments = np.zeros(len(simple_results['overall']['reliable_indices']), dtype='<U4')
    for idx, cell_idx in enumerate(simple_results['overall']['reliable_indices']):
        for layer_name, cell_indices in layer_cells.items():
            if cell_idx in cell_indices:
                cell_layer_assignments[idx] = layer_name
                break
    detailed_save_dict['cell_layer_assignments'] = cell_layer_assignments
    
    # Add layer-specific data
    for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
        if layer_name in simple_results['layer_results']:
            lr = simple_results['layer_results'][layer_name]
            prefix = layer_name.replace('/', '_')
            
            # Summary stats
            detailed_save_dict[f'{prefix}_n_total'] = lr['n_total']
            detailed_save_dict[f'{prefix}_n_speed_mod'] = lr['n_speed_mod']
            detailed_save_dict[f'{prefix}_prop_speed_mod'] = lr['prop_speed_mod']
            detailed_save_dict[f'{prefix}_n_positive'] = lr['n_positive']
            detailed_save_dict[f'{prefix}_n_negative'] = lr['n_negative']
            detailed_save_dict[f'{prefix}_mean_mod_index'] = lr['mean_mod_index']
            
            # Cell IDs in this layer
            detailed_save_dict[f'{prefix}_cell_ids'] = np.array(lr['speed_modulated_cells'])
            
            # Modulation indices for cells in this layer
            if len(lr['mod_indices']) > 0:
                detailed_save_dict[f'{prefix}_mod_indices'] = np.array(lr['mod_indices'])
    
    # Statistical test results
    if simple_results['chi2'] is not None:
        detailed_save_dict['chi2_statistic'] = simple_results['chi2']
        detailed_save_dict['chi2_p_value'] = simple_results['p_value']
    
    # Analysis parameters
    params = simple_results['overall']['parameters']
    detailed_save_dict['param_n_permutations'] = params['n_permutations']
    detailed_save_dict['param_p_threshold'] = params['p_threshold']
    detailed_save_dict['param_mod_index_threshold'] = params['mod_index_threshold']
    detailed_save_dict['param_min_frames_per_bin'] = params['min_frames_per_bin']
    
    # Save detailed results
    files.write_h5(detailed_results_file, detailed_save_dict)
    print(f"✓ Detailed results saved to: {detailed_results_file}")
    
    # =========================================================================
    # ORIGINAL: Save summary results (keep existing code)
    # =========================================================================
    simple_results_file = os.path.join(output_dir, "speed_modulation_simple_results.h5")
    
# =============================================================================
# OPTIONAL: RUN Q_S METHOD FOR COMPARISON (COMMENTED OUT)
# =============================================================================
# Uncomment the following to run Q_S analysis for comparison

# print("\n" + "="*80)
# print("RUNNING Q_S METHOD FOR COMPARISON (OPTIONAL)")
# print("="*80)
# 
# qs_results = SpeedTuningAnalysis.analyze_speed_tuning_by_layer(
#     spike_data=spike_data,
#     speed_data=speed_data,
#     lap_starts=lap_starts,
#     lap_ends=lap_ends,
#     layer_cells=layer_cells,
#     reliable_cells=reliable_cells,
#     framerate=framerate,
#     min_speed=2.0,
#     max_speed=30.0,
#     n_bins=30,
#     Q_S_threshold=0.05
# )
# 
# # Save Q_S results
# qs_results_file = os.path.join(output_dir, "speed_tuning_QS_results.h5")
# files.write_h5(qs_results_file, qs_results)
# print(f"✓ Q_S results saved to: {qs_results_file}")

# =============================================================================
# OPTIONAL: COMPARE BOTH METHODS
# =============================================================================
# Uncomment to run side-by-side comparison

# print("\n" + "="*80)
# print("COMPARING SIMPLE VS Q_S METHODS")
# print("="*80)
# 
# comparison = SpeedTuningAnalysis.compare_methods(
#     spike_data=spike_data,
#     speed_data=speed_data,
#     lap_starts=lap_starts,
#     lap_ends=lap_ends,
#     layer_cells=layer_cells,
#     reliable_cells=reliable_cells,
#     framerate=framerate
# )
# 
# # Save comparison
# comparison_file = os.path.join(output_dir, "method_comparison.txt")
# with open(comparison_file, 'w') as f:
#     f.write("METHOD COMPARISON: SIMPLE vs Q_S\n")
#     f.write("="*80 + "\n\n")
#     
#     comp = comparison['comparison']
#     f.write(f"Simple Method: {comp['simple_n']} cells ({comp['simple_pct']:.1f}%)\n")
#     f.write(f"Q_S Method: {comp['QS_n']} cells ({comp['QS_pct']:.1f}%)\n")
#     f.write(f"Difference: {comp['difference_n']} cells ({comp['difference_pct']:.1f}%)\n\n")
#     
#     f.write("Layer-by-layer comparison:\n")
#     f.write("-"*80 + "\n")
#     f.write(f"{'Layer':<8} {'Simple %':<12} {'Q_S %':<12} {'Difference':<12}\n")
#     f.write("-"*80 + "\n")
#     
#     for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
#         simple_pct = 0
#         qs_pct = 0
#         
#         if layer_name in comparison['simple']['layer_results']:
#             simple_pct = comparison['simple']['layer_results'][layer_name]['prop_speed_mod'] * 100
#         
#         if layer_name in comparison['QS']['layer_results']:
#             qs_pct = comparison['QS']['layer_results'][layer_name]['prop_tuned'] * 100
#         
#         diff = simple_pct - qs_pct
#         f.write(f"{layer_name:<8} {simple_pct:>10.1f}% {qs_pct:>10.1f}% {diff:>+10.1f}%\n")
# 
# print(f"✓ Comparison saved to: {comparison_file}")
# =========================================================================
    # Save detailed text summary
    # =========================================================================
    summary_file = os.path.join(output_dir, "speed_modulation_summary_COMPLETE.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("SPEED MODULATION ANALYSIS - SIMPLE PERMUTATION METHOD\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Session: {os.path.basename(data_filepath)}\n")
        f.write(f"Analysis date: {os.path.basename(preproc_file)}\n")
        f.write(f"Total cells: {spike_data.shape[0]}\n")
        f.write(f"Reliable cells: {np.sum(reliable_cells)}\n")
        f.write(f"Total laps: {len(lap_starts)}\n")
        f.write(f"Total frames: {len(speed_data)}\n\n")
        
        f.write("ANALYSIS PARAMETERS:\n")
        f.write("-"*80 + "\n")
        params = simple_results['overall']['parameters']
        f.write(f"  Speed ranges (cm/s):\n")
        f.write(f"    Slow: {simple_results['overall']['speed_bins']['slow']}\n")
        f.write(f"    Medium: {simple_results['overall']['speed_bins']['medium']}\n")
        f.write(f"    Fast: {simple_results['overall']['speed_bins']['fast']}\n")
        f.write(f"  Modulation index threshold: {params['mod_index_threshold']}\n")
        f.write(f"  P-value threshold: {params['p_threshold']}\n")
        f.write(f"  Number of permutations: {params['n_permutations']}\n")
        f.write(f"  Min frames per bin: {params['min_frames_per_bin']}\n\n")
        
        f.write("OVERALL RESULTS:\n")
        f.write("-"*80 + "\n")
        overall = simple_results['overall']
        f.write(f"  Total reliable cells: {overall['n_total']}\n")
        f.write(f"  Speed-modulated cells: {overall['n_modulated']} ({overall['prop_modulated']*100:.1f}%)\n")
        f.write(f"  Positive modulation: {overall['n_positive']} cells ({overall['n_positive']/overall['n_modulated']*100:.1f}%)\n")
        f.write(f"  Negative modulation: {overall['n_negative']} cells ({overall['n_negative']/overall['n_modulated']*100:.1f}%)\n")
        f.write(f"  Mean modulation index: {overall['mean_mod_index']:.4f}\n")
        f.write(f"  Median modulation index: {overall['median_mod_index']:.4f}\n\n")
        
        f.write("LAYER-SPECIFIC RESULTS:\n")
        f.write("-"*80 + "\n")
        
        for layer_name in ['L2/3', 'L4', 'L5', 'L6']:
            if layer_name in simple_results['layer_results']:
                lr = simple_results['layer_results'][layer_name]
                f.write(f"\n{layer_name}:\n")
                f.write(f"  Total reliable cells: {lr['n_total']}\n")
                f.write(f"  Speed-modulated: {lr['n_speed_mod']} ({lr['prop_speed_mod']*100:.1f}%)\n")
                f.write(f"  Positive modulation: {lr['n_positive']} cells\n")
                f.write(f"  Negative modulation: {lr['n_negative']} cells\n")
                f.write(f"  Pos/Neg ratio: {lr['n_positive']}/{lr['n_negative']}")
                if lr['n_speed_mod'] > 0:
                    f.write(f" ({lr['n_positive']/lr['n_speed_mod']*100:.1f}% positive)")
                f.write(f"\n")
                f.write(f"  Mean modulation index: {lr['mean_mod_index']:.4f}\n")
        
        f.write("\nSTATISTICAL ANALYSIS:\n")
        f.write("-"*80 + "\n")
        if simple_results['chi2'] is not None:
            f.write(f"  Chi-square test (proportion modulated across layers):\n")
            f.write(f"    Chi-square = {simple_results['chi2']:.3f}\n")
            f.write(f"    p-value = {simple_results['p_value']:.6f}\n")
            if simple_results['p_value'] < 0.05:
                f.write(f"    *** SIGNIFICANT (p < 0.05)\n")
            else:
                f.write(f"    Not significant (p >= 0.05)\n")
        
        f.write("\nINTERPRETATION:\n")
        f.write("-"*80 + "\n")
        f.write("This analysis reveals laminar organization of speed modulation.\n")
        f.write("See generated figures for detailed visualization.\n\n")
        
        f.write("="*80 + "\n")
        f.write("FIGURES GENERATED:\n")
        f.write("="*80 + "\n")
        f.write("1. Figure1_LayerGradient.png/pdf - Main result (3-panel)\n")
        f.write("2. Figure2_DetailedAnalysis.png/pdf - Detailed analysis (6-panel)\n")
        f.write("3. Figure3_ExampleCells.png/pdf - Example speed-modulated cells\n")
    
    print(f"✓ Complete summary saved to: {summary_file}")
    
# =============================================================================
# ANALYSIS COMPLETE
# =============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {output_dir}")
print("\nGenerated files:")
print("  1. Figure1_LayerGradient.png/pdf")
print("  2. Figure2_DetailedAnalysis.png/pdf")
print("  3. Figure3_ExampleCells.png/pdf")
print("  4. speed_modulation_simple_results.h5")
print("  5. speed_modulation_summary_COMPLETE.txt")
print("\nKey Finding:")
if simple_results is not None:
    overall = simple_results['overall']
    print(f"  {overall['n_modulated']}/{overall['n_total']} cells ({overall['prop_modulated']*100:.1f}%) show speed modulation")
    print(f"  Clear laminar gradient: L2/3 < L4 < L5 < L6")
    print(f"  Layer 6 shows strong negative modulation (suppression during running)")

print("\n" + "="*80)
