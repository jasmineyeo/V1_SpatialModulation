"""
SMICalculation_WithinSession_AcrossRecordings.py

REVISED version with better visualization and statistics.

Changes from original:
- Focus on first vs last chunk comparison
- Add individual chunk trajectory plots
- Better handling of missing data
- Cleaner visualizations

JSY, 2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import h5py
from glob import glob
from collections import defaultdict
import pandas as pd


# ============================================================================
# [Keep all data loading functions from original Script 2]
# load_within_session_results()
# extract_day_number()
# load_animal_across_days()
# ============================================================================

def load_within_session_results(h5_path):
    """Load results from Script 1 output."""
    data = {
        'metadata': {},
        'fixed_chunks': {'by_layer': {}},
        'cumulative_chunks': {'by_layer': {}},
        'statistics': {}
    }
    
    with h5py.File(h5_path, 'r') as f:
        # Metadata
        data['metadata']['animal'] = f.attrs.get('animal', 'unknown')
        data['metadata']['day'] = f.attrs.get('day', 'unknown')
        data['metadata']['n_trials'] = f.attrs.get('n_trials', 0)
        data['metadata']['n_cells_analyzed'] = f.attrs.get('n_cells_analyzed', 0)
        
        # Fixed chunks
        if 'fixed_chunks' in f:
            fc = f['fixed_chunks']
            data['fixed_chunks']['chunk_labels'] = [label.decode() for label in fc['chunk_labels'][:]]
            data['fixed_chunks']['n_laps'] = fc['n_laps'][:]
            
            for layer_key in fc.keys():
                if layer_key not in ['chunk_labels', 'n_laps']:
                    layer_name = layer_key.replace('_', '/')
                    lg = fc[layer_key]
                    
                    # Load all chunk SMI values
                    all_smi_values = []
                    chunk_idx = 0
                    while f'chunk_{chunk_idx}_smi' in lg:
                        all_smi_values.append(lg[f'chunk_{chunk_idx}_smi'][:])
                        chunk_idx += 1
                    
                    data['fixed_chunks']['by_layer'][layer_name] = {
                        'chunk_medians': lg['chunk_medians'][:],
                        'chunk_means': lg['chunk_means'][:],
                        'chunk_n_cells': lg['chunk_n_cells'][:],
                        'all_smi_values': all_smi_values
                    }
        
        # Cumulative chunks
        if 'cumulative_chunks' in f:
            cc = f['cumulative_chunks']
            data['cumulative_chunks']['chunk_labels'] = [label.decode() for label in cc['chunk_labels'][:]]
            data['cumulative_chunks']['n_laps'] = cc['n_laps'][:]
            
            for layer_key in cc.keys():
                if layer_key not in ['chunk_labels', 'n_laps']:
                    layer_name = layer_key.replace('_', '/')
                    lg = cc[layer_key]
                    data['cumulative_chunks']['by_layer'][layer_name] = {
                        'chunk_medians': lg['chunk_medians'][:],
                        'chunk_means': lg['chunk_means'][:],
                        'chunk_n_cells': lg['chunk_n_cells'][:]
                    }
        
        # Statistics
        if 'statistics' in f:
            for stat_key in f['statistics'].keys():
                data['statistics'][stat_key] = {}
                for attr_name in f['statistics'][stat_key].attrs:
                    data['statistics'][stat_key][attr_name] = f['statistics'][stat_key].attrs[attr_name]
    
    return data


def extract_day_number(day_string):
    """Extract numeric day from 'Day1', 'Day3', etc."""
    match = re.search(r'Day(\d+)', day_string, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def load_animal_across_days(animal_dir, animal_id):
    """Load all within-session results for one animal."""
    print(f"\n{'='*70}")
    print(f"LOADING DATA FOR {animal_id}")
    print(f"{'='*70}")
    print(f"Directory: {animal_dir}\n")
    
    # Find all within-session result files
    pattern = os.path.join(animal_dir, "**", f"{animal_id}_*_within_session_smi.h5")
    result_files = glob(pattern, recursive=True)
    
    print(f"Found {len(result_files)} within-session result files\n")
    
    all_days_data = {}
    
    for file_path in sorted(result_files):
        filename = os.path.basename(file_path)
        print(f"  Loading: {filename}")
        
        try:
            data = load_within_session_results(file_path)
            day_str = data['metadata']['day']
            day_num = extract_day_number(day_str)
            
            if day_num is not None:
                all_days_data[day_num] = data
                print(f"    Day {day_num}: {data['metadata']['n_trials']} trials, "
                      f"{data['metadata']['n_cells_analyzed']} cells")
            else:
                print(f"    WARNING: Could not extract day number from '{day_str}'")
        
        except Exception as e:
            print(f"    ERROR loading {filename}: {e}")
    
    print(f"\nLoaded {len(all_days_data)} days: {sorted(all_days_data.keys())}")
    
    return all_days_data


# ============================================================================
# REVISED VISUALIZATION
# ============================================================================

def visualize_across_days_revised(all_days_data, animal_id, save_path=None, focus_days=[1, 2, 3]):
    """
    REVISED visualization: clearer, more focused.
    
    Layout (3×3):
    Row 1: Individual chunk trajectories (Days 1, 2, 3)
    Row 2: First vs Last chunk comparison
    Row 3: Summary statistics
    """
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # Filter to available focus days
    available_days = sorted([d for d in focus_days if d in all_days_data])
    n_days = len(available_days)
    
    if n_days == 0:
        print("No days available for visualization")
        return None
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    
    # =========================================================================
    # ROW 1: Chunk trajectories (one panel per day)
    # =========================================================================
    for plot_idx, day in enumerate(available_days[:3]):  # Max 3 days
        ax = fig.add_subplot(3, 3, plot_idx + 1)
        
        day_data = all_days_data[day]['fixed_chunks']
        
        for layer in layer_order:
            if layer in day_data['by_layer']:
                medians = day_data['by_layer'][layer]['chunk_medians']
                chunk_nums = np.arange(1, len(medians) + 1)
                valid = ~np.isnan(medians)
                
                ax.plot(chunk_nums[valid], medians[valid], 'o-',
                       color=layer_colors[layer], linewidth=2.5, markersize=8, label=layer)
        
        ax.set_xlabel('Chunk Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Median SMI', fontsize=11, fontweight='bold')
        ax.set_title(f'Day {day}\n({all_days_data[day]["metadata"]["n_trials"]} laps)',
                    fontsize=12, fontweight='bold')
        if plot_idx == 0:
            ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0, top=1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # ROW 2: First vs Last Chunk (Bar plots with individual points)
    # =========================================================================
    for plot_idx, day in enumerate(available_days[:3]):
        ax = fig.add_subplot(3, 3, 3 + plot_idx + 1)
        
        day_data = all_days_data[day]['fixed_chunks']
        
        first_vals = []
        last_vals = []
        labels = []
        colors = []
        
        for layer in layer_order:
            if layer in day_data['by_layer']:
                medians = day_data['by_layer'][layer]['chunk_medians']
                valid_indices = np.where(~np.isnan(medians))[0]
                
                if len(valid_indices) >= 2:
                    first_vals.append(medians[valid_indices[0]])
                    last_vals.append(medians[valid_indices[-1]])
                    labels.append(layer)
                    colors.append(layer_colors[layer])
        
        if first_vals:
            x_pos = np.arange(len(labels))
            width = 0.35
            
            bars1 = ax.bar(x_pos - width/2, first_vals, width, label='First Chunk',
                          color=colors, alpha=0.5, edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x_pos + width/2, last_vals, width, label='Last Chunk',
                          color=colors, alpha=0.9, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for i, (f, l) in enumerate(zip(first_vals, last_vals)):
                delta = l - f
                symbol = '↑' if delta > 0 else '↓' if delta < 0 else '='
                ax.text(i, max(f, l) + 0.05, f'{symbol}{abs(delta):.2f}',
                       ha='center', fontsize=9, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels)
            ax.set_ylabel('Median SMI', fontsize=11, fontweight='bold')
            ax.set_title(f'Day {day}: First vs Last Chunk', fontsize=12, fontweight='bold')
            if plot_idx == 0:
                ax.legend(fontsize=9)
            ax.set_ylim(bottom=0, top=1.0)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # ROW 3: Summary statistics
    # =========================================================================
    
    # Panel 7: Initial SMI across days
    ax7 = fig.add_subplot(3, 3, 7)
    
    for layer in layer_order:
        initial_smi_by_day = []
        days_with_data = []
        
        for day in available_days:
            if layer in all_days_data[day]['fixed_chunks']['by_layer']:
                medians = all_days_data[day]['fixed_chunks']['by_layer'][layer]['chunk_medians']
                if len(medians) > 0 and not np.isnan(medians[0]):
                    initial_smi_by_day.append(medians[0])
                    days_with_data.append(day)
        
        if len(days_with_data) >= 2:
            ax7.plot(days_with_data, initial_smi_by_day, 'o-',
                    color=layer_colors[layer], linewidth=2.5, markersize=10, label=layer)
    
    ax7.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Initial SMI (Chunk 1)', fontsize=11, fontweight='bold')
    ax7.set_title('Initial Spatial Tuning Across Days', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(bottom=0, top=1.0)
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)
    
    # Panel 8: Slopes across days
    ax8 = fig.add_subplot(3, 3, 8)
    
    for layer in layer_order:
        slopes_by_day = []
        days_with_data = []
        
        for day in available_days:
            stat_key = f"fixed_{layer.replace('/', '_')}"
            if stat_key in all_days_data[day]['statistics']:
                slope = all_days_data[day]['statistics'][stat_key].get('slope', np.nan)
                if not np.isnan(slope):
                    slopes_by_day.append(slope)
                    days_with_data.append(day)
        
        if len(days_with_data) >= 2:
            ax8.plot(days_with_data, slopes_by_day, 's-',
                    color=layer_colors[layer], linewidth=2.5, markersize=10, label=layer)
    
    ax8.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xlabel('Day', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Slope (ΔSMI/chunk)', fontsize=11, fontweight='bold')
    ax8.set_title('Development Rate Across Days', fontsize=12, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)
    
    # Panel 9: Statistics summary table
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    summary = f"{animal_id} - WITHIN-SESSION SMI SUMMARY\n{'='*50}\n\n"
    
    for day in available_days:
        summary += f"Day {day} ({all_days_data[day]['metadata']['n_trials']} laps):\n"
        
        for layer in layer_order:
            if layer in all_days_data[day]['fixed_chunks']['by_layer']:
                medians = all_days_data[day]['fixed_chunks']['by_layer'][layer]['chunk_medians']
                valid_indices = np.where(~np.isnan(medians))[0]
                
                if len(valid_indices) >= 2:
                    first = medians[valid_indices[0]]
                    last = medians[valid_indices[-1]]
                    delta = last - first
                    
                    stat_key = f"fixed_{layer.replace('/', '_')}"
                    if stat_key in all_days_data[day]['statistics']:
                        slope = all_days_data[day]['statistics'][stat_key].get('slope', np.nan)
                        p = all_days_data[day]['statistics'][stat_key].get('slope_p_perm', np.nan)
                        sig = '*' if p < 0.05 else ''
                        
                        summary += f"  {layer}: Δ={delta:+.3f}, slope={slope:+.4f} {sig}\n"
        
        summary += "\n"
    
    ax9.text(0.05, 0.95, summary, transform=ax9.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Main title
    fig.suptitle(f'{animal_id}: Within-Session SMI Development Across Days (REVISED)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig_path = os.path.join(save_path, f'{animal_id}_across_days_REVISED.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {fig_path}")
    
    return fig


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_across_days_analysis_revised(animal_dir, animal_id, save_path=None, focus_days=list(range(1, 8))):
    """REVISED across-days analysis with better visualization."""
    print(f"\n{'='*80}")
    print(f"ACROSS-DAYS ANALYSIS (REVISED): {animal_id}")
    print(f"{'='*80}")
    
    # Load data
    all_days_data = load_animal_across_days(animal_dir, animal_id)
    
    if len(all_days_data) == 0:
        print("No data loaded. Exiting.")
        return None
    
    # Visualization
    if save_path is None:
        save_path = animal_dir
    
    os.makedirs(save_path, exist_ok=True)
    
    fig = visualize_across_days_revised(all_days_data, animal_id, save_path, focus_days)
    
    print(f"\n{'='*80}")
    print("REVISED ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {save_path}")
    print(f"  - {animal_id}_across_days_REVISED.png")
    
    return {'animal_id': animal_id, 'all_days_data': all_days_data, 'figure': fig}


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    ANIMAL_DIRS = {
        'JSY040': r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging',
        'JSY041': r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging',
        'JSY044': r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging',
        'JSY051': r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging',
        'JSY052': r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging',
        'JSY054': r'D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging',
        'JSY055': r'D:\V1_SpatialModulation\2p\V1_prism\JSY055_ChronicImaging',
    }

    ANIMAL = 'JSY044'  # <-- change this to run a different animal

    animal_dir = ANIMAL_DIRS[ANIMAL]
    results = run_across_days_analysis_revised(animal_dir, ANIMAL)

    plt.show()
