"""
SMICalculation_LayerSpecific_SingleRecording.py
A script for calculating the Spatial Modulation Index (SMI) for specific layers in the mouse primary visual cortex
for a single recording session (no multiple recordings per day), with analysis split into temporal chunks.

Input: Single preprocessed .h5 file from preprocess.py (contains all trials)
Output: SMI values for each layer, excluding onset response cells, split into chunks

JSY, 2025
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


def filter_onset_response_cells(spatial_activity, bin_centers, 
                                onset_threshold_cm=15, 
                                reliable_cells=None,
                                verbose=True):
    """
    Identify and filter out cells that have their peak response in the onset region.
    """
    n_cells = spatial_activity.shape[0]
    
    non_onset_cells = np.ones(n_cells, dtype=bool)
    onset_cells = np.zeros(n_cells, dtype=bool)
    peak_positions = np.zeros(n_cells)
    
    onset_bins = bin_centers <= (np.min(bin_centers) + onset_threshold_cm)
    
    cells_to_check = np.arange(n_cells)
    if reliable_cells is not None:
        cells_to_check = np.where(reliable_cells)[0]
    
    if verbose:
        print(f"\nFiltering onset response cells:")
        print(f"  Onset region: {np.min(bin_centers):.1f} to {np.min(bin_centers) + onset_threshold_cm:.1f} cm")
        print(f"  Checking {len(cells_to_check)} cells...")
    
    for cell_idx in cells_to_check:
        cell_avg = np.mean(spatial_activity[cell_idx], axis=0)
        peak_bin_idx = np.argmax(cell_avg)
        peak_positions[cell_idx] = bin_centers[peak_bin_idx]
        
        if onset_bins[peak_bin_idx]:
            onset_cells[cell_idx] = True
            non_onset_cells[cell_idx] = False
    
    n_onset = np.sum(onset_cells)
    n_checked = len(cells_to_check)
    
    if verbose:
        print(f"  Found {n_onset} onset response cells ({n_onset/n_checked*100:.1f}% of checked cells)")
        print(f"  Retaining {np.sum(non_onset_cells & reliable_cells if reliable_cells is not None else non_onset_cells)} cells for analysis")
    
    return non_onset_cells, onset_cells, peak_positions


def visualize_onset_filtering(spatial_activity, reliable_cells, 
                              non_onset_cells, onset_cells, 
                              peak_positions, bin_centers,
                              save_path=None):
    """
    Create visualization showing which cells were filtered as onset responses.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Helper function for creating sorted response plots
    def create_sorted_response_plot(activity, cells_mask):
        reliable_indices = np.where(cells_mask)[0]
        reliable_activity = activity[reliable_indices]
        n_trials = activity.shape[1]
        odd_trials = np.arange(1, n_trials, 2)
        odd_avg = np.mean(reliable_activity[:, odd_trials, :], axis=1)
        
        peak_locations = np.argmax(odd_avg, axis=1)
        sorted_indices = np.argsort(peak_locations)
        sorted_activity = odd_avg[sorted_indices]
        
        for i in range(len(sorted_activity)):
            sorted_activity[i] = (sorted_activity[i] - np.min(sorted_activity[i])) / \
                                 (np.max(sorted_activity[i]) - np.min(sorted_activity[i]))
        
        return sorted_activity
    
    # Plot 1: All reliable cells
    sorted_all = create_sorted_response_plot(spatial_activity, reliable_cells)
    im1 = axes[0, 0].imshow(sorted_all, aspect='auto', cmap='viridis', 
                            interpolation='nearest', vmin=0, vmax=1)
    axes[0, 0].set_title(f'All Reliable Cells\n{np.sum(reliable_cells)} cells', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Spatial Bin')
    axes[0, 0].set_ylabel('Cell Number (sorted by peak)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Filtered cells
    filtered_cells = reliable_cells & non_onset_cells
    sorted_filtered = create_sorted_response_plot(spatial_activity, filtered_cells)
    im2 = axes[0, 1].imshow(sorted_filtered, aspect='auto', cmap='viridis', 
                            interpolation='nearest', vmin=0, vmax=1)
    axes[0, 1].set_title(f'After Onset Filtering\n{np.sum(filtered_cells)} cells', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Spatial Bin')
    axes[0, 1].set_ylabel('Cell Number (sorted by peak)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Peak position distribution
    ax = axes[1, 0]
    onset_peaks = peak_positions[reliable_cells & onset_cells]
    non_onset_peaks = peak_positions[reliable_cells & non_onset_cells]
    
    ax.hist(non_onset_peaks, bins=30, alpha=0.7, color='blue', 
           label=f'Spatial cells (n={len(non_onset_peaks)})')
    ax.hist(onset_peaks, bins=30, alpha=0.7, color='red', 
           label=f'Onset cells (n={len(onset_peaks)})')
    ax.axvline(np.min(bin_centers) + 15, color='black', linestyle='--', 
              linewidth=2, label='Onset threshold')
    ax.set_xlabel('Peak Position (cm)', fontsize=12)
    ax.set_ylabel('Number of Cells', fontsize=12)
    ax.set_title('Distribution of Peak Positions', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "ONSET FILTERING SUMMARY\n\n"
    summary_text += f"Original reliable cells: {np.sum(reliable_cells)}\n"
    summary_text += f"Onset cells (removed): {np.sum(onset_cells & reliable_cells)}\n"
    summary_text += f"Spatial cells (retained): {np.sum(filtered_cells)}\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def split_data_into_chunks(spatial_activity, chunk_size=30):
    """
    Split spatial activity data into chunks of specified trial size.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    chunk_size : int
        Number of trials per chunk
        
    Returns:
    --------
    chunks : list
        List of activity matrices for each chunk
    chunk_indices : list
        List of (start_idx, end_idx) tuples for each chunk
    """
    n_cells, n_trials, n_bins = spatial_activity.shape
    
    chunks = []
    chunk_indices = []
    
    n_chunks = n_trials // chunk_size
    
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        chunk = spatial_activity[:, start_idx:end_idx, :]
        chunks.append(chunk)
        chunk_indices.append((start_idx, end_idx))
    
    # Handle remaining trials
    remaining_trials = n_trials % chunk_size
    if remaining_trials > 0:
        start_idx = n_chunks * chunk_size
        chunk = spatial_activity[:, start_idx:, :]
        chunks.append(chunk)
        chunk_indices.append((start_idx, n_trials))
        print(f"  Note: Last chunk has only {remaining_trials} trials")
    
    return chunks, chunk_indices


# =============================================================================
# MAIN ANALYSIS SCRIPT
# =============================================================================

if __name__ == "__main__":
    
    # ==========================================================================
    # CONFIGURATION
    # ==========================================================================
    
    # Path to your preprocessed data file
    data_filepath = r"F:\2P\spmod\JSY052_ChrnoicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002"
    
    # Analysis parameters
    ONSET_THRESHOLD_CM = 15
    APPLY_ONSET_FILTER = True
    APPLY_CHUNK_FILTER = False
    LAPS_PER_CHUNK = 20
    SEGMENT_DISTANCE = 28
    EXCLUDE_START_CM = 15
    EXCLUDE_END_CM = 10
    SMOOTHING_SIGMA = 1.0
    
    print("="*80)
    print("SMI ANALYSIS WITH ONSET FILTERING & LAP-CHUNKING")
    print("="*80)
    print(f"Data directory: {data_filepath}")
    print(f"Onset filtering: {'ENABLED' if APPLY_ONSET_FILTER else 'DISABLED'}")
    if APPLY_ONSET_FILTER:
        print(f"  Onset threshold: {ONSET_THRESHOLD_CM} cm")
    print(f"Laps per chunk: {LAPS_PER_CHUNK}")
    print("="*80 + "\n")
    
    # ==========================================================================
    # STEP 1: LOAD PREPROCESSED DATA
    # ==========================================================================
    
    print("STEP 1: LOADING PREPROCESSED DATA")
    print("-" * 40)
    
    preproc_files = glob.glob(os.path.join(data_filepath, "*preproc.h5"))
    if not preproc_files:
        raise ValueError(f"No preprocessed .h5 file found in {data_filepath}")
    
    preproc_file = preproc_files[0]
    print(f"Loading: {os.path.basename(preproc_file)}")
    preproc_data = files.read_h5(preproc_file)
    print("Successfully loaded!")
    
    spatial_activity = preproc_data['spatial_activity']
    normalized_spatial_activity = preproc_data['norm_spatial_activity']
    bin_centers = preproc_data['bin_centers']
    reliable_cells = preproc_data['combined_reliable']
    
    n_cells, n_total_trials, n_bins = spatial_activity.shape
    print(f"\nData: {n_cells} cells, {n_total_trials} trials, {n_bins} bins")
    print(f"Reliable cells: {np.sum(reliable_cells)}")
    
    # ==========================================================================
    # STEP 2: PREPARE BIN CENTERS
    # ==========================================================================
    
    shifted_centers = bin_centers - np.min(bin_centers)
    scaled_bin_centers = shifted_centers * (np.size(bin_centers) / np.max(shifted_centers))
    
    # ==========================================================================
    # STEP 3: APPLY ONSET FILTERING
    # ==========================================================================
    
    if APPLY_ONSET_FILTER:
        print("\n" + "="*80)
        print("APPLYING ONSET FILTER")
        print("="*80)
        
        non_onset_cells, onset_cells, peak_positions = filter_onset_response_cells(
            spatial_activity, scaled_bin_centers, ONSET_THRESHOLD_CM, reliable_cells, True
        )
        
        analysis_reliable_cells = reliable_cells & non_onset_cells
        
        viz_path = os.path.join(data_filepath, 'onset_filtering_visualization.png')
        visualize_onset_filtering(spatial_activity, reliable_cells, non_onset_cells, 
                                  onset_cells, peak_positions, scaled_bin_centers, viz_path)
    else:
        analysis_reliable_cells = reliable_cells
    
    # ==========================================================================
    # STEP 4: FULL SESSION ANALYSIS
    # ==========================================================================
    
    print("\n" + "="*80)
    print("FULL SESSION ANALYSIS")
    print("="*80)
    
    RV.create_response_plot(spatial_activity, analysis_reliable_cells)
    plt.savefig(os.path.join(data_filepath, 'response_full.png'), dpi=300, bbox_inches='tight')
    
    results_full = SMI.analyze_spatial_modulation_improved(
        spatial_activity, scaled_bin_centers, analysis_reliable_cells,
        segment_distance=SEGMENT_DISTANCE, exclude_start_cm=EXCLUDE_START_CM, 
        exclude_end_cm=EXCLUDE_END_CM, smoothing_sigma=SMOOTHING_SIGMA, data_filepath=data_filepath
    )
    
    SMI_full = results_full['smi_results']['SMI']
    valid_full = results_full['smi_results']['reliable_valid_cells']
    SMI_full_clean = SMI_full[valid_full]
    SMI_full_clean = SMI_full_clean[~np.isnan(SMI_full_clean) & ~np.isinf(SMI_full_clean)]
    
    print(f"Valid cells: {len(SMI_full_clean)}")
    print(f"Median SMI: {np.median(SMI_full_clean):.2f} ± {stats.median_abs_deviation(SMI_full_clean):.2f}")
    
    # Cumulative distribution
    plt.figure(figsize=(8, 6))
    x = np.sort(SMI_full_clean)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, 'k-', linewidth=2)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('SMI')
    plt.ylabel('Cumulative Probability')
    plt.title(f'Full Session (All {n_total_trials} Trials)')
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(data_filepath, 'SMI_full.png'), dpi=300, bbox_inches='tight')
    
    # ==========================================================================
    # STEP 5: LAYER ANALYSIS - FULL SESSION
    # ==========================================================================
    
    print("\n" + "="*80)
    print("LAYER ANALYSIS - FULL SESSION")
    print("="*80)
    
    twoP_filename = os.path.basename(data_filepath)
    raw_twop_data = TwoP(data_filepath, twoP_filename)
    raw_twop_data.find_files()
    twop_dict = raw_twop_data.calc_dFF()
    
    numCells = len(twop_dict['stat'])
    im = np.zeros((twop_dict['ops']['Ly'], twop_dict['ops']['Lx']))
    for n in range(numCells):
        ypix = twop_dict['stat'][n]['ypix'][~twop_dict['stat'][n]['overlap']]
        xpix = twop_dict['stat'][n]['xpix'][~twop_dict['stat'][n]['overlap']]
        im[ypix, xpix] = xpix
    
    med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
    layer_cells, _ = SMI_Layer.identify_layers(med_coords)
    SMI_Layer.plot_layer_distribution(med_coords, layer_cells, analysis_reliable_cells, im)
    plt.savefig(os.path.join(data_filepath, 'layers_full.png'), dpi=300, bbox_inches='tight')
    
    layer_results_full, _ = SMI_Layer.run_layer_SMI_analysis(
        results_full['smi_results'], valid_full, med_coords, layer_cells,
        normalized_spatial_activity, scaled_bin_centers
    )
    
    # ==========================================================================
    # STEP 6: CHUNKED ANALYSIS
    # ==========================================================================
    if APPLY_CHUNK_FILTER:
        print("\n" + "="*80)
        print(f"CHUNKED ANALYSIS ({LAPS_PER_CHUNK} laps/chunk)")
        print("="*80)
        
        spatial_chunks, chunk_indices = split_data_into_chunks(spatial_activity, LAPS_PER_CHUNK)
        normalized_chunks, _ = split_data_into_chunks(normalized_spatial_activity, LAPS_PER_CHUNK)
        
        n_chunks = len(spatial_chunks)
        print(f"\n{n_total_trials} trials → {n_chunks} chunks")
        for i, (start, end) in enumerate(chunk_indices):
            print(f"  Chunk {i+1}: Trials {start+1}-{end} ({end-start} laps)")
        
        chunk_results = []
        chunk_smi_values = []
        
        for chunk_idx, (chunk_spatial, chunk_normalized) in enumerate(zip(spatial_chunks, normalized_chunks)):
            start_trial, end_trial = chunk_indices[chunk_idx]
            
            print(f"\n{'='*80}")
            print(f"CHUNK {chunk_idx+1}/{n_chunks} (Trials {start_trial+1}-{end_trial})")
            print(f"{'='*80}")
            
            if APPLY_ONSET_FILTER:
                non_onset_chunk, _, _ = filter_onset_response_cells(
                    chunk_spatial, scaled_bin_centers, ONSET_THRESHOLD_CM, reliable_cells, False
                )
                analysis_chunk = reliable_cells & non_onset_chunk
            else:
                analysis_chunk = reliable_cells
            
            RV.create_response_plot(chunk_spatial, analysis_chunk)
            plt.savefig(os.path.join(data_filepath, f'response_chunk{chunk_idx+1}.png'), dpi=300, bbox_inches='tight')
            
            results_chunk = SMI.analyze_spatial_modulation_improved(
                chunk_spatial, scaled_bin_centers, analysis_chunk,
                segment_distance=SEGMENT_DISTANCE, exclude_start_cm=EXCLUDE_START_CM, 
                exclude_end_cm=EXCLUDE_END_CM, smoothing_sigma=SMOOTHING_SIGMA
            )
            
            chunk_results.append(results_chunk)
            
            SMI_chunk = results_chunk['smi_results']['SMI']
            valid_chunk = results_chunk['smi_results']['reliable_valid_cells']
            SMI_chunk_clean = SMI_chunk[valid_chunk]
            SMI_chunk_clean = SMI_chunk_clean[~np.isnan(SMI_chunk_clean) & ~np.isinf(SMI_chunk_clean)]
            
            chunk_smi_values.append(SMI_chunk_clean)
            
            print(f"  Valid cells: {len(SMI_chunk_clean)}")
            print(f"  Median SMI: {np.median(SMI_chunk_clean):.2f} ± {stats.median_abs_deviation(SMI_chunk_clean):.2f}")
            if len(SMI_chunk_clean) > 0:
                print(f"  Positive: {np.mean(SMI_chunk_clean > 0)*100:.1f}%")
            
            plt.figure(figsize=(8, 6))
            x = np.sort(SMI_chunk_clean)
            y = np.arange(1, len(x) + 1) / len(x)
            plt.plot(x, y, 'k-', linewidth=2)
            plt.axvline(0, color='gray', linestyle='--', alpha=0.7)
            plt.xlabel('SMI')
            plt.ylabel('Cumulative Probability')
            plt.title(f'Chunk {chunk_idx+1} (Trials {start_trial+1}-{end_trial})')
            plt.xlim(-1, 1)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(os.path.join(data_filepath, f'SMI_chunk{chunk_idx+1}.png'), dpi=300, bbox_inches='tight')
            
            layer_results_chunk, _ = SMI_Layer.run_layer_SMI_analysis(
                results_chunk['smi_results'], valid_chunk, med_coords, layer_cells,
                chunk_normalized, scaled_bin_centers, 
                save_path=os.path.join(data_filepath, f'layers_chunk{chunk_idx+1}.png')
            )
            
            plt.close('all')
        
        # ==========================================================================
        # STEP 7: CHUNK COMPARISON
        # ==========================================================================
        
        print("\n" + "="*80)
        print("CHUNK COMPARISON")
        print("="*80)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Median SMI over chunks
        chunk_medians = [np.median(smi) for smi in chunk_smi_values]
        chunk_mads = [stats.median_abs_deviation(smi) for smi in chunk_smi_values]
        x_pos = np.arange(1, n_chunks + 1)
        
        axes[0, 0].errorbar(x_pos, chunk_medians, yerr=chunk_mads, fmt='o-', 
                            linewidth=2, markersize=8, capsize=5)
        axes[0, 0].axhline(np.median(SMI_full_clean), color='red', linestyle='--', 
                        label=f'Full ({np.median(SMI_full_clean):.2f})')
        axes[0, 0].set_xlabel('Chunk')
        axes[0, 0].set_ylabel('Median SMI ± MAD')
        axes[0, 0].set_title('Median SMI Across Chunks')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(x_pos)
        
        # Valid cells per chunk
        n_valid = [len(smi) for smi in chunk_smi_values]
        axes[0, 1].bar(x_pos, n_valid, alpha=0.7, color='steelblue')
        axes[0, 1].axhline(len(SMI_full_clean), color='red', linestyle='--', 
                        label=f'Full ({len(SMI_full_clean)})')
        axes[0, 1].set_xlabel('Chunk')
        axes[0, 1].set_ylabel('Valid Cells')
        axes[0, 1].set_title('Valid Cells Per Chunk')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        axes[0, 1].set_xticks(x_pos)
        
        # Distributions
        colors = plt.cm.viridis(np.linspace(0, 1, n_chunks))
        for i, (smi, color) in enumerate(zip(chunk_smi_values, colors)):
            x = np.sort(smi)
            y = np.arange(1, len(x) + 1) / len(x)
            axes[1, 0].plot(x, y, color=color, linewidth=2, label=f'Ch{i+1}', alpha=0.7)
        
        x_full = np.sort(SMI_full_clean)
        y_full = np.arange(1, len(x_full) + 1) / len(x_full)
        axes[1, 0].plot(x_full, y_full, 'k-', linewidth=3, label='Full', alpha=0.9)
        axes[1, 0].axvline(0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('SMI')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('SMI Distributions')
        axes[1, 0].set_xlim(-1, 1)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary
        axes[1, 1].axis('off')
        summary = f"SUMMARY\n\n{n_chunks} chunks × {LAPS_PER_CHUNK} laps\n\n"
        for i, (med, mad) in enumerate(zip(chunk_medians, chunk_mads)):
            summary += f"Chunk {i+1}: {med:.2f}±{mad:.2f}\n"
        summary += f"\nFull: {np.median(SMI_full_clean):.2f}±{stats.median_abs_deviation(SMI_full_clean):.2f}"
        
        axes[1, 1].text(0.1, 0.9, summary, transform=axes[1, 1].transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(data_filepath, 'chunk_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nCheck outputs in: {data_filepath}")
        
        