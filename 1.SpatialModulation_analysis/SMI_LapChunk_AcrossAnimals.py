"""
SMICalculation_WithinSession_AcrossAnimals.py

REVISED population-level analysis with improved visualization and statistics.

Key improvements:
1. Handle missing data properly (different # chunks per animal)
2. Pool cells instead of animal medians (more power)
3. Focus on first vs last chunk (clearer signal)
4. Individual animal spaghetti plots
5. Better error visualization

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
# UTILITY FUNCTIONS
# ============================================================================

def fdr_correction(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR correction."""
    p_values = np.array(p_values)
    n = len(p_values)
    
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    
    reject = np.zeros(n, dtype=bool)
    for i, p in enumerate(sorted_p):
        if p <= (i + 1) / n * alpha:
            reject[sorted_indices[i]] = True
    
    p_adjusted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    p_adjusted = np.minimum.accumulate(p_adjusted[::-1])[::-1]
    
    original_order = np.argsort(sorted_indices)
    p_adjusted = p_adjusted[original_order]
    
    return reject, p_adjusted


# ============================================================================
# DATA LOADING
# ============================================================================

def load_within_session_results(h5_path):
    """Load results from Script 1 output."""
    data = {
        'metadata': {},
        'fixed_chunks': {'by_layer': {}},
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
        
        # Statistics
        if 'statistics' in f:
            for stat_key in f['statistics'].keys():
                data['statistics'][stat_key] = {}
                for attr_name in f['statistics'][stat_key].attrs:
                    data['statistics'][stat_key][attr_name] = f['statistics'][stat_key].attrs[attr_name]
    
    return data


def load_all_animals(parent_dir, focus_days=list(range(1, 8))):
    """
    Load Script 1 outputs from all animals for specified days.
    
    Returns:
        all_data (dict): {(animal_id, day): data}
    """
    print(f"\n{'='*70}")
    print("LOADING ALL ANIMALS - DAYS 1-3 WITHIN-SESSION DATA")
    print(f"{'='*70}")
    print(f"Parent directory: {parent_dir}\n")
    
    result_files = glob(os.path.join(parent_dir, "**", "*_within_session_smi.h5"), recursive=True)
    print(f"Found {len(result_files)} within-session result files\n")
    
    all_data = {}
    
    for file_path in sorted(result_files):
        filename = os.path.basename(file_path)
        
        try:
            data = load_within_session_results(file_path)
            animal_id = data['metadata']['animal']
            day_str = data['metadata']['day']
            
            # Extract day number
            day_match = re.search(r'Day(\d+)', day_str, re.IGNORECASE)
            if day_match:
                day_num = int(day_match.group(1))
            else:
                print(f"  WARNING: Could not extract day from {filename}")
                continue
            
            # Only load focus days
            if day_num in focus_days:
                all_data[(animal_id, day_num)] = data
                print(f"  Loaded: {animal_id} Day{day_num} - {data['metadata']['n_trials']} trials, "
                      f"{data['metadata']['n_cells_analyzed']} cells")
        
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")
    
    # Summary
    animals = sorted(set([animal_id for animal_id, _ in all_data.keys()]))
    print(f"\nLoaded {len(all_data)} recordings from {len(animals)} animals:")
    for animal in animals:
        days = sorted([d for aid, d in all_data.keys() if aid == animal])
        print(f"  {animal}: Days {days}")
    
    return all_data


# ============================================================================
# ANALYSIS: POOLED CELLS (OPTION 1)
# ============================================================================

def analyze_pooled_cells(all_data, focus_days=[1, 2, 3]):
    """
    Pool all cells from all animals for each day/layer/chunk.
    
    This gives more statistical power than pooling animal medians.
    """
    print(f"\n{'='*70}")
    print("ANALYSIS: POOLED CELLS ACROSS ANIMALS")
    print(f"{'='*70}\n")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    # Organize: pooled_data[day][layer][chunk_idx] = [all SMI values from all animals]
    pooled_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for (animal_id, day), recording_data in all_data.items():
        for layer in layer_order:
            if layer in recording_data['fixed_chunks']['by_layer']:
                all_smi_values = recording_data['fixed_chunks']['by_layer'][layer]['all_smi_values']
                
                for chunk_idx, smi_array in enumerate(all_smi_values):
                    if len(smi_array) > 0:
                        pooled_data[day][layer][chunk_idx].extend(smi_array)
    
    # Compute statistics
    results = {}
    
    for day in focus_days:
        print(f"--- Day {day} ---")
        results[day] = {}
        
        for layer in layer_order:
            if layer in pooled_data[day] and len(pooled_data[day][layer]) > 0:
                n_chunks = max(pooled_data[day][layer].keys()) + 1
                
                chunk_medians = []
                chunk_sems = []
                chunk_n_cells = []
                
                for chunk_idx in range(n_chunks):
                    if chunk_idx in pooled_data[day][layer]:
                        cells_smi = pooled_data[day][layer][chunk_idx]
                        chunk_medians.append(np.median(cells_smi))
                        chunk_sems.append(stats.sem(cells_smi))
                        chunk_n_cells.append(len(cells_smi))
                    else:
                        chunk_medians.append(np.nan)
                        chunk_sems.append(np.nan)
                        chunk_n_cells.append(0)
                
                results[day][layer] = {
                    'chunk_medians': np.array(chunk_medians),
                    'chunk_sems': np.array(chunk_sems),
                    'chunk_n_cells': np.array(chunk_n_cells),
                    'raw_data': pooled_data[day][layer]  # Keep for first vs last test
                }
                
                print(f"  {layer}: {n_chunks} chunks, {chunk_n_cells[0]} cells (chunk 1)")
        
        print()
    
    return results


def test_first_vs_last_pooled(pooled_results, focus_days=[1, 2, 3]):
    """
    Test first vs last chunk using pooled cells.
    
    Uses Mann-Whitney U (unpaired, since cells may differ between chunks).
    """
    print(f"\n{'='*70}")
    print("STATISTICAL TEST: FIRST VS LAST CHUNK (POOLED CELLS)")
    print(f"{'='*70}\n")
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    results = {}
    
    for day in focus_days:
        if day not in pooled_results:
            continue
        
        print(f"--- Day {day} ---")
        results[day] = {}
        
        for layer in layer_order:
            if layer not in pooled_results[day]:
                continue
            
            raw_data = pooled_results[day][layer]['raw_data']
            
            # Get first and last chunk indices
            chunk_indices = sorted(raw_data.keys())
            if len(chunk_indices) < 2:
                print(f"  {layer}: Insufficient chunks")
                continue
            
            first_idx = chunk_indices[0]
            last_idx = chunk_indices[-1]
            
            first_smi = raw_data[first_idx]
            last_smi = raw_data[last_idx]
            
            # Mann-Whitney U test
            u_stat, p_val = stats.mannwhitneyu(first_smi, last_smi, alternative='two-sided')
            
            delta_median = np.median(last_smi) - np.median(first_smi)
            
            print(f"  {layer}:")
            print(f"    First chunk: median={np.median(first_smi):.3f}, n={len(first_smi)}")
            print(f"    Last chunk:  median={np.median(last_smi):.3f}, n={len(last_smi)}")
            print(f"    Δ = {delta_median:+.3f}, p = {p_val:.4f}")
            
            if p_val < 0.05:
                if delta_median > 0:
                    print(f"    ✓ Significant INCREASE")
                else:
                    print(f"    ✗ Significant DECREASE")
            else:
                print(f"    → No significant change")
            
            results[day][layer] = {
                'first_median': np.median(first_smi),
                'last_median': np.median(last_smi),
                'delta': delta_median,
                'p_value': p_val,
                'n_first': len(first_smi),
                'n_last': len(last_smi)
            }
        
        print()
    
    return results


# ============================================================================
# ANALYSIS: ANIMAL-LEVEL (OPTION 2)
# ============================================================================

def organize_by_animal(all_data, focus_days=[1, 2, 3]):
    """Organize data by animal for spaghetti plots."""
    animals_data = defaultdict(dict)
    
    for (animal_id, day), recording_data in all_data.items():
        animals_data[animal_id][day] = recording_data
    
    return dict(animals_data)


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_population_revised(pooled_results, first_vs_last_results, 
                                 animals_data, save_path=None, focus_days=[1, 2, 3]):
    """
    REVISED population visualization.
    
    Layout (3×3):
    Row 1: Pooled cell trajectories (Days 1, 2, 3) - clean lines with SEM
    Row 2: First vs Last chunk (violin plots)
    Row 3: Individual animal trajectories (spaghetti plots)
    """
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    fig = plt.figure(figsize=(20, 15))
    
    # =========================================================================
    # ROW 1: Pooled cell trajectories (clean, with proper SEM)
    # =========================================================================
    for plot_idx, day in enumerate(focus_days[:3]):
        ax = fig.add_subplot(3, 3, plot_idx + 1)
        
        if day not in pooled_results:
            ax.text(0.5, 0.5, f'Day {day}\nNo data', ha='center', va='center',
                   fontsize=14, transform=ax.transAxes)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue
        
        for layer in layer_order:
            if layer in pooled_results[day]:
                medians = pooled_results[day][layer]['chunk_medians']
                sems = pooled_results[day][layer]['chunk_sems']
                n_cells = pooled_results[day][layer]['chunk_n_cells']
                
                chunk_nums = np.arange(1, len(medians) + 1)
                
                # Only plot where we have data
                valid = ~np.isnan(medians) & (n_cells >= 10)  # At least 10 cells
                
                if np.sum(valid) > 0:
                    ax.plot(chunk_nums[valid], medians[valid], 'o-',
                           color=layer_colors[layer], linewidth=2.5, markersize=8,
                           label=f'{layer} (n={n_cells[valid][0]} cells)', zorder=3)
                    ax.fill_between(chunk_nums[valid],
                                   medians[valid] - sems[valid],
                                   medians[valid] + sems[valid],
                                   color=layer_colors[layer], alpha=0.2, zorder=2)
        
        ax.set_xlabel('Chunk Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Median SMI', fontsize=11, fontweight='bold')
        ax.set_title(f'Day {day} - Population (Pooled Cells)', fontsize=12, fontweight='bold')
        if plot_idx == 0:
            ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # ROW 2: First vs Last Chunk (Violin plots with statistics)
    # =========================================================================
    for plot_idx, day in enumerate(focus_days[:3]):
        ax = fig.add_subplot(3, 3, 3 + plot_idx + 1)
        
        if day not in first_vs_last_results:
            continue
        
        plot_data_first = []
        plot_data_last = []
        plot_labels = []
        plot_colors = []
        p_values = []
        
        for layer in layer_order:
            if layer in first_vs_last_results[day]:
                lr = first_vs_last_results[day][layer]
                
                # Get raw data from pooled_results
                first_smi = pooled_results[day][layer]['raw_data'][0]
                last_idx = max(pooled_results[day][layer]['raw_data'].keys())
                last_smi = pooled_results[day][layer]['raw_data'][last_idx]
                
                # Subsample if too many cells (for plotting speed)
                if len(first_smi) > 500:
                    first_smi = np.random.choice(first_smi, 500, replace=False)
                if len(last_smi) > 500:
                    last_smi = np.random.choice(last_smi, 500, replace=False)
                
                plot_data_first.append(first_smi)
                plot_data_last.append(last_smi)
                plot_labels.append(layer)
                plot_colors.append(layer_colors[layer])
                p_values.append(lr['p_value'])
        
        if plot_data_first:
            x_pos = np.arange(len(plot_labels))
            width = 0.35
            
            # Violin plots
            parts1 = ax.violinplot(plot_data_first, positions=x_pos - width/2, widths=width,
                                  showmeans=False, showmedians=True)
            parts2 = ax.violinplot(plot_data_last, positions=x_pos + width/2, widths=width,
                                  showmeans=False, showmedians=True)
            
            for i, pc in enumerate(parts1['bodies']):
                pc.set_facecolor(plot_colors[i])
                pc.set_alpha(0.4)
            
            for i, pc in enumerate(parts2['bodies']):
                pc.set_facecolor(plot_colors[i])
                pc.set_alpha(0.7)
            
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
                
                y_pos = max(np.percentile(plot_data_first[i], 95), 
                           np.percentile(plot_data_last[i], 95)) + 0.05
                ax.text(i, y_pos, sig_text, ha='center', fontsize=12, fontweight='bold')
            
            ax.set_xticks(x_pos)
            ax.set_xticklabels(plot_labels)
            ax.set_ylabel('SMI', fontsize=11, fontweight='bold')
            ax.set_title(f'Day {day}: First (light) vs Last (dark) Chunk', 
                        fontsize=12, fontweight='bold')
            ax.set_ylim(0, 1.0)
            ax.grid(True, alpha=0.3, axis='y')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    # =========================================================================
    # ROW 3: Individual animal trajectories (spaghetti plots)
    # =========================================================================
    for plot_idx, day in enumerate(focus_days[:3]):
        ax = fig.add_subplot(3, 3, 6 + plot_idx + 1)
        
        # Plot individual animals as thin lines
        for animal_id, animal_days in animals_data.items():
            if day in animal_days:
                recording_data = animal_days[day]
                
                for layer in ['L2/3', 'L5']:  # Only L2/3 and L5 for clarity
                    if layer in recording_data['fixed_chunks']['by_layer']:
                        medians = recording_data['fixed_chunks']['by_layer'][layer]['chunk_medians']
                        chunk_nums = np.arange(1, len(medians) + 1)
                        valid = ~np.isnan(medians)
                        
                        ax.plot(chunk_nums[valid], medians[valid], '-',
                               color=layer_colors[layer], linewidth=1.5, alpha=0.4)
        
        # Overlay population means
        if day in pooled_results:
            for layer in ['L2/3', 'L5']:
                if layer in pooled_results[day]:
                    medians = pooled_results[day][layer]['chunk_medians']
                    chunk_nums = np.arange(1, len(medians) + 1)
                    valid = ~np.isnan(medians)
                    
                    if np.sum(valid) > 0:
                        ax.plot(chunk_nums[valid], medians[valid], 'o-',
                               color=layer_colors[layer], linewidth=3, markersize=10,
                               label=f'{layer} (population)', zorder=10)
        
        ax.set_xlabel('Chunk Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('Median SMI', fontsize=11, fontweight='bold')
        ax.set_title(f'Day {day}: Individual Animals (L2/3 & L5)', 
                    fontsize=12, fontweight='bold')
        if plot_idx == 0:
            ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Main title
    fig.suptitle('Population-Level Within-Session SMI Development (REVISED)\nDays 1-3, Pooled Cells',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        fig_path = os.path.join(save_path, 'across_animals_within_session_REVISED.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {fig_path}")
    
    return fig


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(pooled_results, first_vs_last_results, save_path=None):
    """Create comprehensive summary tables."""
    
    # Table 1: Pooled cell statistics
    pooled_rows = []
    for day, day_data in pooled_results.items():
        for layer, layer_data in day_data.items():
            medians = layer_data['chunk_medians']
            n_cells = layer_data['chunk_n_cells']
            
            for chunk_idx, (median, n) in enumerate(zip(medians, n_cells)):
                if not np.isnan(median):
                    pooled_rows.append({
                        'Day': day,
                        'Layer': layer,
                        'Chunk': chunk_idx + 1,
                        'Median_SMI': median,
                        'N_Cells': n
                    })
    
    df_pooled = pd.DataFrame(pooled_rows)
    
    # Table 2: First vs Last statistics
    fvl_rows = []
    for day, day_data in first_vs_last_results.items():
        for layer, layer_stats in day_data.items():
            fvl_rows.append({
                'Day': day,
                'Layer': layer,
                'First_Median': layer_stats['first_median'],
                'Last_Median': layer_stats['last_median'],
                'Delta': layer_stats['delta'],
                'P_Value': layer_stats['p_value'],
                'Significant': 'Yes' if layer_stats['p_value'] < 0.05 else 'No',
                'N_First': layer_stats['n_first'],
                'N_Last': layer_stats['n_last']
            })
    
    df_fvl = pd.DataFrame(fvl_rows)
    
    if save_path:
        pooled_path = os.path.join(save_path, 'pooled_cells_summary_REVISED.csv')
        fvl_path = os.path.join(save_path, 'first_vs_last_summary_REVISED.csv')
        
        df_pooled.to_csv(pooled_path, index=False)
        df_fvl.to_csv(fvl_path, index=False)
        
        print(f"\n✓ Saved: {os.path.basename(pooled_path)}")
        print(f"✓ Saved: {os.path.basename(fvl_path)}")
    
    return df_pooled, df_fvl


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_across_animals_analysis_revised(parent_dir, save_path=None, focus_days=list(range(1, 8))):
    """
    REVISED across-animals analysis workflow.
    
    Key changes:
    - Pool cells instead of animal medians
    - Handle missing data gracefully
    - Focus on first vs last chunk
    - Cleaner visualizations
    """
    print(f"\n{'='*80}")
    print("ACROSS-ANIMALS WITHIN-SESSION ANALYSIS (REVISED)")
    print(f"{'='*80}")
    
    # Load all animals
    all_data = load_all_animals(parent_dir, focus_days)
    
    if len(all_data) == 0:
        print("No data loaded. Exiting.")
        return None
    
    # Create save directory
    if save_path is None:
        save_path = os.path.join(parent_dir, 'across_animals_within_session_REVISED')
    os.makedirs(save_path, exist_ok=True)
    
    # Analysis 1: Pooled cells
    pooled_results = analyze_pooled_cells(all_data, focus_days)
    
    # Analysis 2: First vs last chunk
    first_vs_last_results = test_first_vs_last_pooled(pooled_results, focus_days)
    
    # Organize by animal for spaghetti plots
    animals_data = organize_by_animal(all_data, focus_days)
    
    # Visualization
    fig = visualize_population_revised(pooled_results, first_vs_last_results,
                                      animals_data, save_path, focus_days)
    
    # Summary tables
    df_pooled, df_fvl = create_summary_tables(pooled_results, first_vs_last_results, save_path)
    
    print(f"\n{'='*80}")
    print("REVISED ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {save_path}")
    print(f"  - across_animals_within_session_REVISED.png")
    print(f"  - pooled_cells_summary_REVISED.csv")
    print(f"  - first_vs_last_summary_REVISED.csv")
    
    # Print key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}\n")
    
    for day in focus_days:
        if day in first_vs_last_results:
            print(f"Day {day}:")
            for layer in ['L2/3', 'L4', 'L5', 'L6']:
                if layer in first_vs_last_results[day]:
                    lr = first_vs_last_results[day][layer]
                    sig = '***' if lr['p_value'] < 0.001 else '**' if lr['p_value'] < 0.01 else '*' if lr['p_value'] < 0.05 else 'ns'
                    direction = '↑' if lr['delta'] > 0 else '↓' if lr['delta'] < 0 else '='
                    print(f"  {layer}: {direction} Δ={lr['delta']:+.3f}, p={lr['p_value']:.4f} {sig}")
            print()
    
    results = {
        'pooled_results': pooled_results,
        'first_vs_last': first_vs_last_results,
        'animals_data': animals_data,
        'summary_tables': (df_pooled, df_fvl),
        'figure': fig
    }
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Configure paths
    parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
    save_dir = os.path.join(parent_dir, "across_animals_within_session_REVISED")
    
    results = run_across_animals_analysis_revised(parent_dir, save_dir)
    
    plt.show()
