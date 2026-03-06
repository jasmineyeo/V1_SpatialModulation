"""
SMICalculation_LayerSpecific_WithinSession_SingleRecording.py

Analyze how spatial modulation (SMI) develops within a single recording session.

Questions:
1. Does SMI increase over the course of a session?
2. Do deeper layers (L5/L6) stabilize faster than superficial layers (L2/3)?
3. How many laps are needed for SMI to plateau?

Analysis approaches:
- Fixed chunks: Non-overlapping blocks of 20 laps
- Cumulative: Progressive addition of laps (1-20, 1-40, 1-60, etc.)

Global onset filtering applied using all laps, then track SMI development
in identified spatial cells.

JSY, 2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import h5py
import pandas as pd
from collections import defaultdict

from helper import files, TwoP
from helper import SMI_Calculation as SMI
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def filter_onset_response_cells(spatial_activity, bin_centers, reliable_cells=None,
                                exclude_first_bins=5, exclude_last_bins=5, verbose=True):
    """
    Identify and filter out cells with peak response in onset/reward regions.
    (Same as in your original script)
    """
    n_cells = spatial_activity.shape[0]
    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)
    bin_spacing = np.mean(np.diff(bin_centers))
    
    onset_threshold = min_pos + (exclude_first_bins * bin_spacing)
    end_threshold = max_pos - (exclude_last_bins * bin_spacing)
    
    non_onset_cells = np.ones(n_cells, dtype=bool)
    rejected_onset, rejected_reward, rejected_zero = [], [], []
    
    mean_profiles = np.mean(spatial_activity, axis=1)
    cells_to_check = np.where(reliable_cells)[0] if reliable_cells is not None else np.arange(n_cells)
    peak_positions = np.zeros(n_cells)
    
    for cell_idx in cells_to_check:
        profile = mean_profiles[cell_idx]
        global_peak_idx = np.argmax(profile)
        global_peak_pos = bin_centers[global_peak_idx]
        peak_positions[cell_idx] = global_peak_pos
        
        if profile[global_peak_idx] == 0:
            rejected_zero.append(cell_idx)
            non_onset_cells[cell_idx] = False
        elif global_peak_pos < onset_threshold:
            rejected_onset.append(cell_idx)
            non_onset_cells[cell_idx] = False
        elif global_peak_pos > end_threshold:
            rejected_reward.append(cell_idx)
            non_onset_cells[cell_idx] = False
    
    rejected_info = {
        'onset': np.array(rejected_onset),
        'reward': np.array(rejected_reward),
        'zero_activity': np.array(rejected_zero),
        'peak_positions': peak_positions,
        'onset_threshold': onset_threshold,
        'end_threshold': end_threshold
    }
    
    if verbose:
        n_rejected = len(rejected_onset) + len(rejected_reward) + len(rejected_zero)
        print(f"\nOnset/Reward Filtering: Onset < {onset_threshold:.1f}cm, Reward > {end_threshold:.1f}cm")
        print(f"  Rejected: {n_rejected} (Onset: {len(rejected_onset)}, Reward: {len(rejected_reward)}, Zero: {len(rejected_zero)})")
    
    return non_onset_cells, rejected_info


# ============================================================================
# CHUNK ANALYSIS FUNCTIONS
# ============================================================================

def split_trials_into_chunks(n_trials, chunk_size=20, min_chunk_size=10):
    """
    Split trial indices into non-overlapping chunks.
    
    Parameters:
        n_trials (int): Total number of trials
        chunk_size (int): Target size for each chunk
        min_chunk_size (int): Minimum size for last chunk (default 10)
    
    Returns:
        chunks (list): List of (start_idx, end_idx) tuples
        chunk_labels (list): Labels for each chunk (e.g., "Laps 1-20")
    """
    chunks = []
    chunk_labels = []
    
    start_idx = 0
    
    while start_idx < n_trials:
        end_idx = min(start_idx + chunk_size, n_trials)
        
        # Check if this is the last chunk and if it's too small
        remaining = n_trials - end_idx
        if remaining > 0 and remaining < min_chunk_size:
            # Absorb remaining laps into current chunk
            end_idx = n_trials
        
        chunks.append((start_idx, end_idx))
        chunk_labels.append(f"Laps {start_idx+1}-{end_idx}")
        
        start_idx = end_idx
    
    return chunks, chunk_labels


def create_cumulative_chunks(n_trials, chunk_size=20):
    """
    Create cumulative chunks (1-20, 1-40, 1-60, etc.).
    
    Parameters:
        n_trials (int): Total number of trials
        chunk_size (int): Increment size
    
    Returns:
        chunks (list): List of (start_idx, end_idx) tuples
        chunk_labels (list): Labels for each chunk
    """
    chunks = []
    chunk_labels = []
    
    end_idx = chunk_size
    while end_idx <= n_trials:
        chunks.append((0, end_idx))
        chunk_labels.append(f"Laps 1-{end_idx}")
        end_idx += chunk_size
    
    # Add final chunk with all laps if not already included
    if chunks[-1][1] < n_trials:
        chunks.append((0, n_trials))
        chunk_labels.append(f"Laps 1-{n_trials}")
    
    return chunks, chunk_labels


def calculate_smi_for_chunk(spatial_activity_chunk, bin_centers, reliable_cells,
                            segment_distance=28, exclude_start_cm=15, 
                            exclude_end_cm=10, smoothing_sigma=1.0):
    """
    Calculate SMI for a specific chunk of trials.
    
    Parameters:
        spatial_activity_chunk (array): (n_cells, n_trials_in_chunk, n_bins)
        bin_centers (array): Spatial bin centers
        reliable_cells (array): Boolean mask of cells to analyze
        [other SMI parameters]
    
    Returns:
        smi_results (dict): SMI values and statistics
    """
    # Calculate SMI using existing function
    results = SMI.analyze_spatial_modulation_improved(
        spatial_activity_chunk, bin_centers, reliable_cells,
        segment_distance=segment_distance,
        exclude_start_cm=exclude_start_cm,
        exclude_end_cm=exclude_end_cm,
        smoothing_sigma=smoothing_sigma,
        data_filepath=None  # Don't save figures for chunks
    )
    
    return results['smi_results']


def analyze_chunks_by_layer(spatial_activity, bin_centers, analysis_reliable_cells, 
                            layer_cells, chunks, chunk_labels, analysis_type='fixed',
                            segment_distance=28, exclude_start_cm=15, 
                            exclude_end_cm=10, smoothing_sigma=1.0):
    """
    Calculate SMI for each chunk, organized by layer.
    
    Parameters:
        spatial_activity (array): Full spatial activity (n_cells, n_trials, n_bins)
        bin_centers (array): Spatial bin centers
        analysis_reliable_cells (array): Boolean mask (after onset filtering)
        layer_cells (dict): {layer_name: cell_indices}
        chunks (list): List of (start_idx, end_idx) tuples
        chunk_labels (list): Labels for chunks
        analysis_type (str): 'fixed' or 'cumulative'
    
    Returns:
        chunk_results (dict): Results organized by chunk and layer
    """
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    n_cells = spatial_activity.shape[0]
    
    print(f"\n{'='*70}")
    print(f"CHUNK ANALYSIS: {analysis_type.upper()}")
    print(f"{'='*70}")
    print(f"Total cells: {n_cells}")
    print(f"Cells for analysis: {np.sum(analysis_reliable_cells)}")
    print(f"Number of chunks: {len(chunks)}")
    
    # Results storage
    results = {
        'chunk_labels': chunk_labels,
        'n_laps': [end - start for start, end in chunks],
        'by_layer': {}
    }
    
    for layer_name in layer_order:
        if layer_name not in layer_cells or len(layer_cells[layer_name]) == 0:
            continue
        
        print(f"\n--- {layer_name} ---")
        
        # Get cells in this layer that passed filtering
        layer_indices = layer_cells[layer_name]
        layer_mask = np.zeros(n_cells, dtype=bool)
        layer_mask[layer_indices] = True
        layer_analysis_mask = layer_mask & analysis_reliable_cells
        
        n_layer_cells = np.sum(layer_analysis_mask)
        print(f"  Layer cells for analysis: {n_layer_cells}")
        
        if n_layer_cells == 0:
            continue
        
        chunk_medians = []
        chunk_means = []
        chunk_n_cells = []
        all_smi_values = []
        
        for i, (start_idx, end_idx) in enumerate(chunks):
            # Extract chunk
            chunk_data = spatial_activity[:, start_idx:end_idx, :]
            n_trials_chunk = end_idx - start_idx
            
            # Calculate SMI for this chunk
            try:
                smi_results = calculate_smi_for_chunk(
                    chunk_data, bin_centers, layer_analysis_mask,
                    segment_distance=segment_distance,
                    exclude_start_cm=exclude_start_cm,
                    exclude_end_cm=exclude_end_cm,
                    smoothing_sigma=smoothing_sigma
                )
                
                # Extract SMI values for cells in this layer
                smi_vals = smi_results['SMI'][layer_analysis_mask]
                smi_vals_clean = smi_vals[~np.isnan(smi_vals) & ~np.isinf(smi_vals)]
                
                if len(smi_vals_clean) > 0:
                    chunk_medians.append(np.median(smi_vals_clean))
                    chunk_means.append(np.mean(smi_vals_clean))
                    chunk_n_cells.append(len(smi_vals_clean))
                    all_smi_values.append(smi_vals_clean)
                    
                    print(f"  {chunk_labels[i]}: median={np.median(smi_vals_clean):.3f}, n={len(smi_vals_clean)}")
                else:
                    chunk_medians.append(np.nan)
                    chunk_means.append(np.nan)
                    chunk_n_cells.append(0)
                    all_smi_values.append(np.array([]))
                    print(f"  {chunk_labels[i]}: No valid cells")
                    
            except Exception as e:
                print(f"  {chunk_labels[i]}: ERROR - {e}")
                chunk_medians.append(np.nan)
                chunk_means.append(np.nan)
                chunk_n_cells.append(0)
                all_smi_values.append(np.array([]))
        
        results['by_layer'][layer_name] = {
            'chunk_medians': np.array(chunk_medians),
            'chunk_means': np.array(chunk_means),
            'chunk_n_cells': np.array(chunk_n_cells),
            'all_smi_values': all_smi_values
        }
    
    return results


def analyze_within_session_trends(chunk_results, analysis_type='fixed'):
    """
    Statistical analysis of within-session SMI trends.
    
    Tests:
    - Spearman correlation (chunk number vs median SMI)
    - Linear regression + permutation test
    - First vs Last chunk comparison
    
    Parameters:
        chunk_results (dict): Output from analyze_chunks_by_layer
        analysis_type (str): 'fixed' or 'cumulative'
    
    Returns:
        stats_results (dict): Statistical test results per layer
    """
    print(f"\n{'='*70}")
    print(f"STATISTICAL ANALYSIS: {analysis_type.upper()} CHUNKS")
    print(f"{'='*70}")
    
    stats_results = {}
    
    for layer_name, layer_data in chunk_results['by_layer'].items():
        print(f"\n--- {layer_name} ---")
        
        medians = layer_data['chunk_medians']
        valid_indices = ~np.isnan(medians)
        
        if np.sum(valid_indices) < 2:
            print("  Insufficient data for trend analysis")
            continue
        
        medians_valid = medians[valid_indices]
        chunk_numbers = np.arange(1, len(medians) + 1)[valid_indices]
        
        # Spearman correlation
        rho, p_spearman = stats.spearmanr(chunk_numbers, medians_valid)
        print(f"  Spearman: ρ={rho:.3f}, p={p_spearman:.4f}")
        
        # Linear regression
        slope, intercept, r_val, p_linreg, _ = stats.linregress(chunk_numbers, medians_valid)
        print(f"  Linear: slope={slope:.4f}, R²={r_val**2:.3f}, p={p_linreg:.4f}")
        
        # Permutation test for slope
        observed_slope = slope
        n_perm = 500
        perm_slopes = []
        
        for _ in range(n_perm):
            perm_medians = np.random.permutation(medians_valid)
            perm_slope, _, _, _, _ = stats.linregress(chunk_numbers, perm_medians)
            perm_slopes.append(perm_slope)
        
        p_perm = np.mean(np.abs(perm_slopes) >= np.abs(observed_slope))
        print(f"  Permutation test (slope≠0): p={p_perm:.4f}")
        
        # First vs Last chunk
        first_chunk_smi = chunk_results['by_layer'][layer_name]['all_smi_values'][0]
        last_chunk_idx = np.where(valid_indices)[0][-1]
        last_chunk_smi = chunk_results['by_layer'][layer_name]['all_smi_values'][last_chunk_idx]
        
        if len(first_chunk_smi) > 0 and len(last_chunk_smi) > 0:
            u_stat, p_mw = stats.mannwhitneyu(first_chunk_smi, last_chunk_smi, alternative='two-sided')
            delta_median = medians_valid[-1] - medians_valid[0]
            
            print(f"  First vs Last: Δ={delta_median:+.3f}, p={p_mw:.4f}")
            
            if p_mw < 0.05 and delta_median > 0:
                print(f"  ✓ Significant INCREASE")
            elif p_mw < 0.05 and delta_median < 0:
                print(f"  ✗ Significant DECREASE")
            else:
                print(f"  → No significant change")
        else:
            p_mw = np.nan
            delta_median = np.nan
        
        stats_results[layer_name] = {
            'spearman': {'rho': rho, 'p': p_spearman},
            'linreg': {'slope': slope, 'r2': r_val**2, 'p': p_linreg},
            'permutation': {'p': p_perm, 'observed_slope': observed_slope},
            'first_vs_last': {'delta': delta_median, 'p': p_mw}
        }
    
    return stats_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_within_session(fixed_results, cumulative_results, 
                             fixed_stats, cumulative_stats,
                             session_info, save_path=None):
    """
    Create comprehensive visualization of within-session SMI development.
    
    Layout (2×3):
    1. Fixed chunks - Line plot
    2. Cumulative - Line plot
    3. First vs Last chunk comparison
    4. Cell heatmap (fixed chunks)
    5. Slope comparison
    6. Statistics summary
    """
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    fig = plt.figure(figsize=(20, 12))
    
    # =========================================================================
    # Panel 1: Fixed chunks - Line plot
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    
    for layer in layer_order:
        if layer in fixed_results['by_layer']:
            medians = fixed_results['by_layer'][layer]['chunk_medians']
            chunk_nums = np.arange(1, len(medians) + 1)
            valid = ~np.isnan(medians)
            
            ax1.plot(chunk_nums[valid], medians[valid], 'o-', 
                    color=layer_colors[layer], linewidth=2.5, markersize=8, label=layer)
    
    ax1.set_xlabel('Chunk Number', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax1.set_title('Fixed Chunks (20 laps each)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # =========================================================================
    # Panel 2: Cumulative - Line plot
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    
    for layer in layer_order:
        if layer in cumulative_results['by_layer']:
            medians = cumulative_results['by_layer'][layer]['chunk_medians']
            n_laps = cumulative_results['n_laps']
            valid = ~np.isnan(medians)
            
            ax2.plot(np.array(n_laps)[valid], medians[valid], 'o-',
                    color=layer_colors[layer], linewidth=2.5, markersize=8, label=layer)
    
    ax2.set_xlabel('Number of Laps', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Median SMI', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Analysis', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # =========================================================================
    # Panel 3: First vs Last chunk comparison (violin plot)
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    
    plot_data_first = []
    plot_data_last = []
    plot_labels = []
    plot_colors = []
    positions = []
    
    for idx, layer in enumerate(layer_order):
        if layer in fixed_results['by_layer']:
            all_smi = fixed_results['by_layer'][layer]['all_smi_values']
            medians = fixed_results['by_layer'][layer]['chunk_medians']
            valid_indices = np.where(~np.isnan(medians))[0]
            
            if len(valid_indices) >= 2:
                first_smi = all_smi[valid_indices[0]]
                last_smi = all_smi[valid_indices[-1]]
                
                if len(first_smi) > 0 and len(last_smi) > 0:
                    plot_data_first.append(first_smi)
                    plot_data_last.append(last_smi)
                    plot_labels.append(layer)
                    plot_colors.append(layer_colors[layer])
                    positions.append(idx)
    
    if plot_data_first:
        width = 0.35
        x_pos = np.arange(len(plot_labels))
        
        parts1 = ax3.violinplot(plot_data_first, positions=x_pos - width/2, widths=width,
                                showmeans=False, showmedians=True)
        parts2 = ax3.violinplot(plot_data_last, positions=x_pos + width/2, widths=width,
                                showmeans=False, showmedians=True)
        
        for i, pc in enumerate(parts1['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.5)
        
        for i, pc in enumerate(parts2['bodies']):
            pc.set_facecolor(plot_colors[i])
            pc.set_alpha(0.8)
        
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(plot_labels)
        ax3.set_ylabel('SMI', fontsize=12, fontweight='bold')
        ax3.set_title('First (light) vs Last (dark) Chunk', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(bottom=0)
    
    # =========================================================================
    # Panel 4: Cell heatmap (fixed chunks) - Show a few example cells
    # =========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    
    # Combine all layers for heatmap
    all_layer_smi = []
    all_layer_labels = []
    
    for layer in layer_order:
        if layer in fixed_results['by_layer']:
            all_smi_values = fixed_results['by_layer'][layer]['all_smi_values']
            n_chunks = len(all_smi_values)
            
            # Get cells that appear in all chunks
            if n_chunks > 0 and len(all_smi_values[0]) > 0:
                # Create matrix: cells × chunks
                n_cells_layer = len(all_smi_values[0])
                smi_matrix = np.full((n_cells_layer, n_chunks), np.nan)
                
                for chunk_idx in range(n_chunks):
                    if len(all_smi_values[chunk_idx]) == n_cells_layer:
                        smi_matrix[:, chunk_idx] = all_smi_values[chunk_idx]
                
                # Keep only cells with data in all chunks
                valid_cells = ~np.any(np.isnan(smi_matrix), axis=1)
                smi_matrix_valid = smi_matrix[valid_cells]
                
                if len(smi_matrix_valid) > 0:
                    all_layer_smi.append(smi_matrix_valid)
                    all_layer_labels.extend([layer] * len(smi_matrix_valid))
    
    if all_layer_smi:
        combined_smi = np.vstack(all_layer_smi)
        
        # Sort by peak chunk
        peak_chunks = np.argmax(combined_smi, axis=1)
        sort_idx = np.argsort(peak_chunks)
        combined_smi_sorted = combined_smi[sort_idx]
        
        # Downsample if too many cells
        if len(combined_smi_sorted) > 100:
            sample_idx = np.linspace(0, len(combined_smi_sorted)-1, 100, dtype=int)
            combined_smi_sorted = combined_smi_sorted[sample_idx]
        
        im = ax4.imshow(combined_smi_sorted, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax4.set_xlabel('Chunk Number', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Cell #', fontsize=12, fontweight='bold')
        ax4.set_title('Cell SMI Across Chunks\n(sorted by peak)', fontsize=13, fontweight='bold')
        plt.colorbar(im, ax=ax4, label='SMI')
    
    # =========================================================================
    # Panel 5: Slope comparison (development rate)
    # =========================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    
    slopes = []
    slope_labels = []
    slope_colors = []
    p_values = []
    
    for layer in layer_order:
        if layer in fixed_stats:
            slope = fixed_stats[layer]['linreg']['slope']
            p_val = fixed_stats[layer]['permutation']['p']
            
            slopes.append(slope)
            slope_labels.append(layer)
            slope_colors.append(layer_colors[layer])
            p_values.append(p_val)
    
    if slopes:
        x_pos = np.arange(len(slopes))
        bars = ax5.bar(x_pos, slopes, color=slope_colors, alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        
        # Add significance stars
        for i, p_val in enumerate(p_values):
            if p_val < 0.001:
                sig_text = '***'
            elif p_val < 0.01:
                sig_text = '**'
            elif p_val < 0.05:
                sig_text = '*'
            else:
                sig_text = 'ns'
            
            y_pos = slopes[i] + 0.005 if slopes[i] > 0 else slopes[i] - 0.005
            ax5.text(i, y_pos, sig_text, ha='center', 
                    va='bottom' if slopes[i] > 0 else 'top',
                    fontsize=12, fontweight='bold')
        
        ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(slope_labels)
        ax5.set_ylabel('Slope (ΔSMI/chunk)', fontsize=12, fontweight='bold')
        ax5.set_title('Development Rate\n(* p<0.05, ** p<0.01, *** p<0.001)', 
                     fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel 6: Statistics summary
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    summary = f"WITHIN-SESSION SMI DEVELOPMENT\n{'='*50}\n\n"
    summary += f"Session: {session_info['animal']} - {session_info['day']}\n"
    summary += f"Total laps: {session_info['n_trials']}\n"
    summary += f"Cells analyzed: {session_info['n_cells_analyzed']}\n\n"
    
    summary += "FIXED CHUNKS (Spearman ρ):\n"
    for layer in layer_order:
        if layer in fixed_stats:
            rho = fixed_stats[layer]['spearman']['rho']
            p = fixed_stats[layer]['spearman']['p']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            summary += f"  {layer}: ρ={rho:+.3f}, p={p:.4f} {sig}\n"
    
    summary += "\nCUMULATIVE (Spearman ρ):\n"
    for layer in layer_order:
        if layer in cumulative_stats:
            rho = cumulative_stats[layer]['spearman']['rho']
            p = cumulative_stats[layer]['spearman']['p']
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
            summary += f"  {layer}: ρ={rho:+.3f}, p={p:.4f} {sig}\n"
    
    summary += "\nFIRST vs LAST CHUNK:\n"
    for layer in layer_order:
        if layer in fixed_stats and not np.isnan(fixed_stats[layer]['first_vs_last']['delta']):
            delta = fixed_stats[layer]['first_vs_last']['delta']
            p = fixed_stats[layer]['first_vs_last']['p']
            sig = '*' if p < 0.05 else ''
            summary += f"  {layer}: Δ={delta:+.3f}, p={p:.4f} {sig}\n"
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Main title
    title_text = f"Within-Session SMI Development: {session_info['animal']} - {session_info['day']}"
    fig.suptitle(title_text, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig_path = os.path.join(save_path, 'within_session_smi_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {fig_path}")
    
    return fig


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_within_session_results(save_path, session_info, fixed_results, cumulative_results,
                                fixed_stats, cumulative_stats):
    """Save within-session analysis results to HDF5."""
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")
    print(f"Output: {os.path.basename(save_path)}")
    
    with h5py.File(save_path, 'w') as f:
        # Metadata
        f.attrs['animal'] = session_info['animal']
        f.attrs['day'] = session_info['day']
        f.attrs['n_trials'] = session_info['n_trials']
        f.attrs['n_cells_analyzed'] = session_info['n_cells_analyzed']
        
        # Fixed chunks
        fixed_grp = f.create_group('fixed_chunks')
        fixed_grp.create_dataset('chunk_labels', data=np.array(fixed_results['chunk_labels'], dtype='S'))
        fixed_grp.create_dataset('n_laps', data=fixed_results['n_laps'])
        
        for layer_name, layer_data in fixed_results['by_layer'].items():
            layer_grp = fixed_grp.create_group(layer_name.replace('/', '_'))
            layer_grp.create_dataset('chunk_medians', data=layer_data['chunk_medians'])
            layer_grp.create_dataset('chunk_means', data=layer_data['chunk_means'])
            layer_grp.create_dataset('chunk_n_cells', data=layer_data['chunk_n_cells'])
            
            # Save SMI values for each chunk
            for i, smi_vals in enumerate(layer_data['all_smi_values']):
                if len(smi_vals) > 0:
                    layer_grp.create_dataset(f'chunk_{i}_smi', data=smi_vals)
        
        # Cumulative chunks
        cumul_grp = f.create_group('cumulative_chunks')
        cumul_grp.create_dataset('chunk_labels', data=np.array(cumulative_results['chunk_labels'], dtype='S'))
        cumul_grp.create_dataset('n_laps', data=cumulative_results['n_laps'])
        
        for layer_name, layer_data in cumulative_results['by_layer'].items():
            layer_grp = cumul_grp.create_group(layer_name.replace('/', '_'))
            layer_grp.create_dataset('chunk_medians', data=layer_data['chunk_medians'])
            layer_grp.create_dataset('chunk_means', data=layer_data['chunk_means'])
            layer_grp.create_dataset('chunk_n_cells', data=layer_data['chunk_n_cells'])
        
        # Statistics
        stats_grp = f.create_group('statistics')
        
        for layer_name, layer_stats in fixed_stats.items():
            layer_grp = stats_grp.create_group(f'fixed_{layer_name.replace("/", "_")}')
            layer_grp.attrs['spearman_rho'] = layer_stats['spearman']['rho']
            layer_grp.attrs['spearman_p'] = layer_stats['spearman']['p']
            layer_grp.attrs['slope'] = layer_stats['linreg']['slope']
            layer_grp.attrs['r2'] = layer_stats['linreg']['r2']
            layer_grp.attrs['slope_p_perm'] = layer_stats['permutation']['p']
            layer_grp.attrs['first_vs_last_delta'] = layer_stats['first_vs_last']['delta']
            layer_grp.attrs['first_vs_last_p'] = layer_stats['first_vs_last']['p']
        
        for layer_name, layer_stats in cumulative_stats.items():
            layer_grp = stats_grp.create_group(f'cumulative_{layer_name.replace("/", "_")}')
            layer_grp.attrs['spearman_rho'] = layer_stats['spearman']['rho']
            layer_grp.attrs['spearman_p'] = layer_stats['spearman']['p']
            layer_grp.attrs['slope'] = layer_stats['linreg']['slope']
            layer_grp.attrs['r2'] = layer_stats['linreg']['r2']
    
    print(f"✓ Saved: {os.path.basename(save_path)}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_within_session_analysis(data_filepath, chunk_size=20, min_chunk_size=10,
                                exclude_first_bins=5, exclude_last_bins=5,
                                segment_distance=28, exclude_start_cm=15,
                                exclude_end_cm=10, smoothing_sigma=1.0,
                                save_figures=True):
    """
    Complete workflow for within-session SMI analysis.
    
    Parameters:
        data_filepath (str): Path to TSeries folder
        chunk_size (int): Number of laps per chunk (default 20)
        min_chunk_size (int): Minimum laps for last chunk (default 10)
        [other SMI parameters...]
    
    Returns:
        results (dict): All analysis results
    """
    print(f"\n{'='*80}")
    print("WITHIN-SESSION SMI ANALYSIS")
    print(f"{'='*80}")
    print(f"Data: {data_filepath}")
    print(f"Chunk size: {chunk_size} laps, min: {min_chunk_size} laps")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("STEP 1: Loading preprocessed data...")
    
    preproc_files = glob.glob(os.path.join(data_filepath, "*preproc*.h5"))
    if not preproc_files:
        raise ValueError(f"No preprocessed .h5 file found in {data_filepath}")
    
    preproc_file = preproc_files[0]
    preproc_data = files.read_h5(preproc_file)
    
    spatial_activity = preproc_data['spatial_activity']
    bin_centers = preproc_data['bin_centers']
    reliable_cells = preproc_data['combined_reliable']
    
    n_cells, n_trials, n_bins = spatial_activity.shape
    print(f"  Data: {n_cells} cells, {n_trials} trials, {n_bins} bins")
    print(f"  Reliable cells: {np.sum(reliable_cells)}")
    
    # Prepare bin centers
    shifted_centers = bin_centers - np.min(bin_centers)
    scaled_bin_centers = shifted_centers * (np.size(bin_centers) / np.max(shifted_centers))
    
    # ========================================================================
    # STEP 2: GET LAYER INFORMATION
    # ========================================================================
    print("\nSTEP 2: Identifying cortical layers...")
    
    twoP_filename = os.path.basename(data_filepath)
    raw_twop_data = TwoP(data_filepath, twoP_filename)
    raw_twop_data.find_files()
    twop_dict = raw_twop_data.calc_dFF()
    
    med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
    layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)
    
    # ========================================================================
    # STEP 3: APPLY GLOBAL ONSET FILTERING
    # ========================================================================
    print("\nSTEP 3: Applying global onset/reward filtering...")
    
    non_onset_cells, rejected_info = filter_onset_response_cells(
        spatial_activity, scaled_bin_centers, reliable_cells,
        exclude_first_bins=exclude_first_bins,
        exclude_last_bins=exclude_last_bins
    )
    
    analysis_reliable_cells = reliable_cells & non_onset_cells
    print(f"  Final cells for analysis: {np.sum(analysis_reliable_cells)}")
    
    # ========================================================================
    # STEP 4: CREATE CHUNKS
    # ========================================================================
    print("\nSTEP 4: Creating trial chunks...")
    
    # Fixed chunks
    fixed_chunks, fixed_labels = split_trials_into_chunks(
        n_trials, chunk_size=chunk_size, min_chunk_size=min_chunk_size
    )
    print(f"  Fixed chunks: {len(fixed_chunks)}")
    for i, (start, end) in enumerate(fixed_chunks):
        print(f"    {fixed_labels[i]}: trials {start}-{end} ({end-start} laps)")
    
    # Cumulative chunks
    cumulative_chunks, cumulative_labels = create_cumulative_chunks(
        n_trials, chunk_size=chunk_size
    )
    print(f"\n  Cumulative chunks: {len(cumulative_chunks)}")
    for i, (start, end) in enumerate(cumulative_chunks):
        print(f"    {cumulative_labels[i]}: trials {start}-{end} ({end-start} laps)")
    
    # ========================================================================
    # STEP 5: ANALYZE FIXED CHUNKS
    # ========================================================================
    fixed_results = analyze_chunks_by_layer(
        spatial_activity, scaled_bin_centers, analysis_reliable_cells,
        layer_cells, fixed_chunks, fixed_labels, analysis_type='fixed',
        segment_distance=segment_distance, exclude_start_cm=exclude_start_cm,
        exclude_end_cm=exclude_end_cm, smoothing_sigma=smoothing_sigma
    )
    
    fixed_stats = analyze_within_session_trends(fixed_results, analysis_type='fixed')
    
    # ========================================================================
    # STEP 6: ANALYZE CUMULATIVE CHUNKS
    # ========================================================================
    cumulative_results = analyze_chunks_by_layer(
        spatial_activity, scaled_bin_centers, analysis_reliable_cells,
        layer_cells, cumulative_chunks, cumulative_labels, analysis_type='cumulative',
        segment_distance=segment_distance, exclude_start_cm=exclude_start_cm,
        exclude_end_cm=exclude_end_cm, smoothing_sigma=smoothing_sigma
    )
    
    cumulative_stats = analyze_within_session_trends(cumulative_results, analysis_type='cumulative')
    # cumulative_results = {}
    # cumulative_stats = {}
    # ========================================================================
    # STEP 7: EXTRACT SESSION INFO AND SAVE
    # ========================================================================
    print("\nSTEP 7: Saving results...")
    
    session_folder = os.path.basename(os.path.dirname(data_filepath))
    match = re.match(r'(\d{6})_.*_(Day\d+)', session_folder)
    if match:
        date_str = match.group(1)
        session_id = match.group(2)
    else:
        date_str = "unknown"
        session_id = "unknown"
    
    animal_match = re.search(r'(JSY\d+)', data_filepath)
    animal_id = animal_match.group(1) if animal_match else "unknown"
    
    session_info = {
        'animal': animal_id,
        'day': session_id,
        'n_trials': n_trials,
        'n_cells_analyzed': np.sum(analysis_reliable_cells)
    }
    
    # Save HDF5
    h5_save_path = os.path.join(data_filepath, f"{animal_id}_{session_id}_within_session_smi.h5")
    save_within_session_results(h5_save_path, session_info, fixed_results, cumulative_results,
                                fixed_stats, cumulative_stats)
    
    # ========================================================================
    # STEP 8: CREATE VISUALIZATION
    # ========================================================================
    if save_figures:
        print("\nSTEP 8: Creating visualization...")
        fig = visualize_within_session(fixed_results, cumulative_results,
                                       fixed_stats, cumulative_stats,
                                       session_info, save_path=data_filepath)
    
    print(f"\n{'='*80}")
    print("WITHIN-SESSION ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {data_filepath}")
    print(f"  - {animal_id}_{session_id}_within_session_smi.h5")
    print(f"  - within_session_smi_analysis.png")
    
    results = {
        'session_info': session_info,
        'fixed_results': fixed_results,
        'cumulative_results': cumulative_results,
        'fixed_stats': fixed_stats,
        'cumulative_stats': cumulative_stats,
        'figure': fig if save_figures else None
    }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Example usage
    data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002"
    
    results = run_within_session_analysis(data_filepath, chunk_size=20, min_chunk_size=10)
    
    # plt.show()
