"""
PCA_WithinLayer_SMI_Analysis.py

Test whether adaptation-like vs spatial-like cells differ in:
1. Initial spatial modulation (Day 1 SMI)
2. Learning trajectory (SMI change Day 1 → Day 7)

Analyzed separately for each layer to test:
- HYPOTHESIS A (Deep = Innate): Deep layers have high SMI from Day 1 regardless of cell type
- HYPOTHESIS B (Superficial = Learned): Superficial layers show larger SMI increases

JSY, 01/2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\figures\within_layer_smi"

EARLY_SLOPE_THRESHOLD = None  # Use median


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load PCA data."""
    print(f"Loading data from: {filepath}")
    
    data = {}
    with h5py.File(filepath, 'r') as f:
        data['bin_centers'] = f['metadata/bin_centers_trimmed'][:]
        data['landmark_positions'] = f['metadata/landmark_positions'][:]
        data['session_labels'] = f['cells/session_labels'][:].astype(str)
        data['layer_labels'] = f['cells/layer_labels'][:].astype(str)
        data['preferred_landmark'] = f['cells/preferred_landmark'][:]
        data['peak_positions'] = f['cells/peak_positions'][:]
        data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
        data['pc_scores'] = f['pca_results/pc_scores'][:]
    
    print(f"  Loaded {len(data['pc_scores'])} cells")
    
    return data


def compute_early_slope(profiles, bin_centers, onset_end_cm=15, slope_end_cm=30):
    """Compute early decay slope for each cell."""
    n_cells = profiles.shape[0]
    early_slopes = np.zeros(n_cells)
    
    onset_end_idx = np.searchsorted(bin_centers, onset_end_cm)
    slope_end_idx = np.searchsorted(bin_centers, slope_end_cm)
    
    for i in range(n_cells):
        early_region = profiles[i, onset_end_idx:slope_end_idx]
        if len(early_region) > 2:
            x = np.arange(len(early_region))
            slope, _, _, _, _ = stats.linregress(x, early_region)
            early_slopes[i] = slope
    
    return early_slopes


def compute_smi_like(profiles):
    """
    Compute SMI-like metric: peak / mean ratio.
    Higher values = more spatially modulated.
    """
    n_cells = profiles.shape[0]
    smi_like = np.zeros(n_cells)
    
    for i in range(n_cells):
        profile = profiles[i]
        peak = np.max(profile)
        mean_abs = np.mean(np.abs(profile))
        if mean_abs > 0:
            smi_like[i] = peak / mean_abs
        else:
            smi_like[i] = 0
    
    return smi_like


# ============================================================================
# CELL CLASSIFICATION
# ============================================================================

def classify_l1_cells(data, threshold=None):
    """
    Classify L1-preferring cells into adaptation-like vs spatial-like.
    """
    # Get L1 cells
    l1_mask = data['preferred_landmark'] == 0
    n_l1 = np.sum(l1_mask)
    
    print(f"\nClassifying {n_l1} L1-preferring cells...")
    
    # Compute early slopes
    early_slopes = compute_early_slope(
        data['spatial_profiles_zscore'], 
        data['bin_centers']
    )
    
    l1_slopes = early_slopes[l1_mask]
    
    # Use median if threshold not specified
    if threshold is None:
        threshold = np.median(l1_slopes)
    
    print(f"  Early slope threshold: {threshold:.4f}")
    
    # Classify: adaptation-like = BELOW threshold (negative/low slope)
    l1_indices = np.where(l1_mask)[0]
    adaptation_mask = l1_slopes < threshold
    spatial_mask = l1_slopes >= threshold
    
    results = {
        'l1_mask': l1_mask,
        'l1_indices': l1_indices,
        'early_slopes': l1_slopes,
        'threshold': threshold,
        'adaptation_mask': adaptation_mask,
        'spatial_mask': spatial_mask,
        'n_adaptation': np.sum(adaptation_mask),
        'n_spatial': np.sum(spatial_mask),
        'layer_labels': data['layer_labels'][l1_mask],
        'session_labels': data['session_labels'][l1_mask],
        'profiles': data['spatial_profiles_zscore'][l1_mask]
    }
    
    print(f"  Adaptation-like: {results['n_adaptation']}")
    print(f"  Spatial-like: {results['n_spatial']}")
    
    return results


# ============================================================================
# WITHIN-LAYER SMI ANALYSIS
# ============================================================================

def analyze_smi_by_cell_type_and_layer(data, classification_results):
    """
    For each layer, compare adaptation-like vs spatial-like cells on:
    - SMI for EVERY session (not just Day 1 and Day 7)
    - Overall learning trajectory
    """
    print("\n" + "=" * 70)
    print("WITHIN-LAYER SMI ANALYSIS")
    print("=" * 70)
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    session_order = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day7']
    
    # Get L1 cell data
    l1_mask = classification_results['l1_mask']
    adaptation_mask = classification_results['adaptation_mask']
    spatial_mask = classification_results['spatial_mask']
    layer_labels = classification_results['layer_labels']
    session_labels = classification_results['session_labels']
    profiles = classification_results['profiles']
    
    # Compute SMI for all L1 cells
    smi_values = compute_smi_like(profiles)
    
    results = {}
    
    for layer in layers:
        print(f"\n{'='*70}")
        print(f"{layer}")
        print(f"{'='*70}")
        
        layer_mask = layer_labels == layer
        
        # Get cells in this layer
        adapt_in_layer = adaptation_mask & layer_mask
        spatial_in_layer = spatial_mask & layer_mask
        
        n_adapt = np.sum(adapt_in_layer)
        n_spatial = np.sum(spatial_in_layer)
        
        print(f"  Total cells: Adaptation-like={n_adapt}, Spatial-like={n_spatial}")
        
        if n_adapt < 3 or n_spatial < 3:
            print(f"  ⚠ Insufficient cells for analysis")
            results[layer] = None
            continue
        
        # =====================================================================
        # ANALYZE EVERY SESSION
        # =====================================================================
        session_results = {}
        adapt_trajectory = []
        spatial_trajectory = []
        
        print(f"\n{'Session':<10} {'Adapt Mean':<12} {'Spatial Mean':<12} {'t-stat':<10} {'p-value':<10} {'Sig':<5}")
        print("-" * 70)
        
        for session in session_order:
            session_mask = session_labels == session
            
            # Get SMI for adaptation-like cells in this session
            adapt_session_mask = adapt_in_layer & session_mask
            smi_adapt_session = smi_values[adapt_session_mask]
            n_adapt_session = len(smi_adapt_session)
            
            # Get SMI for spatial-like cells in this session
            spatial_session_mask = spatial_in_layer & session_mask
            smi_spatial_session = smi_values[spatial_session_mask]
            n_spatial_session = len(smi_spatial_session)
            
            # Statistics
            if n_adapt_session >= 2 and n_spatial_session >= 2:
                mean_adapt = np.mean(smi_adapt_session)
                mean_spatial = np.mean(smi_spatial_session)
                
                t_stat, p_val = stats.ttest_ind(smi_adapt_session, smi_spatial_session)
                
                if p_val < 0.001:
                    sig = '***'
                elif p_val < 0.01:
                    sig = '**'
                elif p_val < 0.05:
                    sig = '*'
                else:
                    sig = 'ns'
                
                print(f"{session:<10} {mean_adapt:<12.3f} {mean_spatial:<12.3f} "
                      f"{t_stat:<10.3f} {p_val:<10.4f} {sig:<5}")
            else:
                mean_adapt = np.mean(smi_adapt_session) if n_adapt_session > 0 else np.nan
                mean_spatial = np.mean(smi_spatial_session) if n_spatial_session > 0 else np.nan
                t_stat, p_val = np.nan, np.nan
                sig = 'N/A'
                
                print(f"{session:<10} {mean_adapt:<12.3f} {mean_spatial:<12.3f} "
                      f"{'N/A':<10} {'N/A':<10} {sig:<5}")
            
            # Store for trajectory
            adapt_trajectory.append({
                'session': session,
                'mean': mean_adapt,
                'sem': np.std(smi_adapt_session) / np.sqrt(n_adapt_session) if n_adapt_session > 0 else np.nan,
                'n': n_adapt_session,
                'values': smi_adapt_session
            })
            
            spatial_trajectory.append({
                'session': session,
                'mean': mean_spatial,
                'sem': np.std(smi_spatial_session) / np.sqrt(n_spatial_session) if n_spatial_session > 0 else np.nan,
                'n': n_spatial_session,
                'values': smi_spatial_session
            })
            
            session_results[session] = {
                'smi_adapt': smi_adapt_session,
                'smi_spatial': smi_spatial_session,
                't_stat': t_stat,
                'p_val': p_val,
                'significance': sig
            }
        
        # =====================================================================
        # LEARNING TRAJECTORY ANALYSIS
        # =====================================================================
        print(f"\n--- Learning Trajectory Analysis ---")
        
        # Test if SMI increases over time for each cell type
        adapt_means = [d['mean'] for d in adapt_trajectory if not np.isnan(d['mean'])]
        spatial_means = [d['mean'] for d in spatial_trajectory if not np.isnan(d['mean'])]
        
        if len(adapt_means) >= 3:
            session_nums = np.arange(len(adapt_means))
            r_adapt, p_adapt = stats.pearsonr(session_nums, adapt_means)
            print(f"  Adaptation-like learning: r={r_adapt:.3f}, p={p_adapt:.4f}")
            
            if p_adapt < 0.05:
                trend = "increasing" if r_adapt > 0 else "decreasing"
                print(f"    ✓ Significant {trend} trend")
        else:
            r_adapt, p_adapt = np.nan, np.nan
        
        if len(spatial_means) >= 3:
            session_nums = np.arange(len(spatial_means))
            r_spatial, p_spatial = stats.pearsonr(session_nums, spatial_means)
            print(f"  Spatial-like learning: r={r_spatial:.3f}, p={p_spatial:.4f}")
            
            if p_spatial < 0.05:
                trend = "increasing" if r_spatial > 0 else "decreasing"
                print(f"    ✓ Significant {trend} trend")
        else:
            r_spatial, p_spatial = np.nan, np.nan
        
        # Compare Day 1 vs Day 7 explicitly
        if 'Day1' in session_results and 'Day7' in session_results:
            day1_adapt = session_results['Day1']['smi_adapt']
            day7_adapt = session_results['Day7']['smi_adapt']
            day1_spatial = session_results['Day1']['smi_spatial']
            day7_spatial = session_results['Day7']['smi_spatial']
            
            if len(day1_adapt) >= 2 and len(day7_adapt) >= 2:
                delta_adapt = np.mean(day7_adapt) - np.mean(day1_adapt)
                print(f"\n  Adaptation-like: Day1={np.mean(day1_adapt):.3f} → Day7={np.mean(day7_adapt):.3f}")
                print(f"                   ΔSMI = {delta_adapt:+.3f}")
            else:
                delta_adapt = np.nan
            
            if len(day1_spatial) >= 2 and len(day7_spatial) >= 2:
                delta_spatial = np.mean(day7_spatial) - np.mean(day1_spatial)
                print(f"  Spatial-like:    Day1={np.mean(day1_spatial):.3f} → Day7={np.mean(day7_spatial):.3f}")
                print(f"                   ΔSMI = {delta_spatial:+.3f}")
            else:
                delta_spatial = np.nan
            
            # Compare learning rates
            if not np.isnan(delta_adapt) and not np.isnan(delta_spatial):
                print(f"\n  Learning comparison:")
                if abs(delta_spatial) > abs(delta_adapt) * 1.2:
                    print(f"    → Spatial-like cells learn MORE (Δ={delta_spatial:.3f} vs {delta_adapt:.3f})")
                elif abs(delta_adapt) > abs(delta_spatial) * 1.2:
                    print(f"    → Adaptation-like cells learn MORE (Δ={delta_adapt:.3f} vs {delta_spatial:.3f})")
                else:
                    print(f"    → Similar learning rates")
        else:
            delta_adapt = np.nan
            delta_spatial = np.nan
        
        # Store results
        results[layer] = {
            'n_adapt': n_adapt,
            'n_spatial': n_spatial,
            'session_results': session_results,
            'adapt_trajectory': adapt_trajectory,
            'spatial_trajectory': spatial_trajectory,
            'learning_stats': {
                'r_adapt': r_adapt if 'r_adapt' in locals() else np.nan,
                'p_adapt': p_adapt if 'p_adapt' in locals() else np.nan,
                'r_spatial': r_spatial if 'r_spatial' in locals() else np.nan,
                'p_spatial': p_spatial if 'p_spatial' in locals() else np.nan,
                'delta_adapt': delta_adapt,
                'delta_spatial': delta_spatial
            }
        }
    
    return results

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_within_layer_smi_analysis(results, save_dir):
    """
    Comprehensive visualization of within-layer SMI analysis.
    Now showing ALL SESSIONS explicitly.
    """
    print("\nGenerating figures...")
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    adapt_color = '#E53935'
    spatial_color = '#1E88E5'
    session_order = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day7']
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    # =========================================================================
    # ROW 1: Day 1 SMI by layer (initial state)
    # =========================================================================
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[0, idx])
        
        if results[layer] is None:
            ax.text(0.5, 0.5, f'{layer}\nInsufficient data', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
            continue
        
        day1_results = results[layer]['session_results'].get('Day1', None)
        if day1_results is None:
            ax.axis('off')
            continue
        
        smi_adapt = day1_results['smi_adapt']
        smi_spatial = day1_results['smi_spatial']
        
        # Violin plot
        parts = ax.violinplot([smi_adapt, smi_spatial], 
                              positions=[1, 2],
                              widths=0.7,
                              showmeans=True,
                              showextrema=True)
        
        # Color the violins
        parts['bodies'][0].set_facecolor(adapt_color)
        parts['bodies'][0].set_alpha(0.7)
        parts['bodies'][1].set_facecolor(spatial_color)
        parts['bodies'][1].set_alpha(0.7)
        
        # Add individual points
        ax.scatter(np.ones(len(smi_adapt)) + np.random.normal(0, 0.05, len(smi_adapt)),
                  smi_adapt, alpha=0.3, s=20, c=adapt_color)
        ax.scatter(2*np.ones(len(smi_spatial)) + np.random.normal(0, 0.05, len(smi_spatial)),
                  smi_spatial, alpha=0.3, s=20, c=spatial_color)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Adapt\n' + f'(n={len(smi_adapt)})', 
                           'Spatial\n' + f'(n={len(smi_spatial)})'],
                          fontsize=9)
        ax.set_ylabel('Day 1 SMI', fontsize=10)
        ax.set_title(f'{layer}: Initial SMI', 
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add significance
        p = day1_results['p_val']
        if not np.isnan(p):
            y_max = max(np.max(smi_adapt), np.max(smi_spatial))
            sig_text = day1_results['significance']
            
            if sig_text != 'ns':
                ax.plot([1, 2], [y_max*1.1, y_max*1.1], 'k-', linewidth=1)
                ax.text(1.5, y_max*1.15, sig_text, ha='center', fontsize=12, fontweight='bold')
    
    # =========================================================================
    # ROW 2: SMI Trajectories (all sessions)
    # =========================================================================
    x = np.arange(len(session_order))
    
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[1, idx])
        
        if results[layer] is None:
            ax.axis('off')
            continue
        
        # Adaptation-like trajectory
        adapt_means = [d['mean'] for d in results[layer]['adapt_trajectory']]
        adapt_sems = [d['sem'] for d in results[layer]['adapt_trajectory']]
        adapt_ns = [d['n'] for d in results[layer]['adapt_trajectory']]
        
        # Spatial-like trajectory
        spatial_means = [d['mean'] for d in results[layer]['spatial_trajectory']]
        spatial_sems = [d['sem'] for d in results[layer]['spatial_trajectory']]
        spatial_ns = [d['n'] for d in results[layer]['spatial_trajectory']]
        
        ax.errorbar(x, adapt_means, yerr=adapt_sems, 
                   marker='o', linewidth=2.5, markersize=8, capsize=4,
                   color=adapt_color, label='Adaptation-like', alpha=0.8)
        ax.errorbar(x, spatial_means, yerr=spatial_sems,
                   marker='s', linewidth=2.5, markersize=8, capsize=4,
                   color=spatial_color, label='Spatial-like', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(session_order, fontsize=9)
        ax.set_xlabel('Session', fontsize=10)
        ax.set_ylabel('SMI', fontsize=10)
        ax.set_title(f'{layer}: SMI Development', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(alpha=0.3)
        
        # Add learning stats as text
        learning = results[layer]['learning_stats']
        r_adapt = learning['r_adapt']
        p_adapt = learning['p_adapt']
        r_spatial = learning['r_spatial']
        p_spatial = learning['p_spatial']
        
        stats_text = f"Adapt: r={r_adapt:.2f}"
        if not np.isnan(p_adapt) and p_adapt < 0.05:
            stats_text += "*"
        stats_text += f"\nSpatial: r={r_spatial:.2f}"
        if not np.isnan(p_spatial) and p_spatial < 0.05:
            stats_text += "*"
        
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
               fontsize=8, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # =========================================================================
    # ROW 3: Session-by-session significance heatmap
    # =========================================================================
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[2, idx])
        
        if results[layer] is None:
            ax.axis('off')
            continue
        
        # Create significance matrix
        sig_values = []
        for session in session_order:
            session_result = results[layer]['session_results'].get(session, None)
            if session_result is not None:
                p = session_result['p_val']
                if np.isnan(p):
                    sig_values.append(1.0)  # Gray for N/A
                else:
                    sig_values.append(p)
            else:
                sig_values.append(1.0)
        
        sig_matrix = np.array([sig_values])
        
        # Plot heatmap
        im = ax.imshow(sig_matrix, cmap='RdYlGn_r', aspect='auto', 
                      vmin=0, vmax=0.1, interpolation='nearest')
        
        ax.set_xticks(np.arange(len(session_order)))
        ax.set_xticklabels(session_order, fontsize=9)
        ax.set_yticks([])
        ax.set_title(f'{layer}: Adapt vs Spatial p-values', fontsize=11, fontweight='bold')
        
        # Add p-value text
        for i, (session, p) in enumerate(zip(session_order, sig_values)):
            if p < 1.0:
                text = f'{p:.3f}' if p >= 0.001 else '<.001'
                color = 'white' if p < 0.05 else 'black'
                ax.text(i, 0, text, ha='center', va='center', 
                       fontsize=8, color=color, fontweight='bold')
    
    # Add colorbar for heatmaps
    cbar_ax = fig.add_axes([0.92, 0.37, 0.01, 0.1])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('p-value', fontsize=10)
    
    # =========================================================================
    # ROW 4: Summary panels
    # =========================================================================
    
    # Panel 1: Day 1 comparison
    ax_summary1 = fig.add_subplot(gs[3, 0])
    
    x_layers = np.arange(len(layers))
    width = 0.35
    
    day1_adapt_means = []
    day1_spatial_means = []
    
    for layer in layers:
        if results[layer] is not None and 'Day1' in results[layer]['session_results']:
            day1 = results[layer]['session_results']['Day1']
            day1_adapt_means.append(np.mean(day1['smi_adapt']))
            day1_spatial_means.append(np.mean(day1['smi_spatial']))
        else:
            day1_adapt_means.append(0)
            day1_spatial_means.append(0)
    
    ax_summary1.bar(x_layers - width/2, day1_adapt_means, width, 
                    label='Adaptation-like', color=adapt_color, alpha=0.8)
    ax_summary1.bar(x_layers + width/2, day1_spatial_means, width,
                    label='Spatial-like', color=spatial_color, alpha=0.8)
    
    ax_summary1.set_xticks(x_layers)
    ax_summary1.set_xticklabels(layers)
    ax_summary1.set_xlabel('Layer', fontsize=11)
    ax_summary1.set_ylabel('Day 1 SMI', fontsize=11)
    ax_summary1.set_title('Initial SMI by Layer', fontsize=12, fontweight='bold')
    ax_summary1.legend(loc='upper left', fontsize=9)
    ax_summary1.grid(alpha=0.3, axis='y')
    
    # Panel 2: Learning effect (ΔSMI Day1→Day7)
    ax_summary2 = fig.add_subplot(gs[3, 1])
    
    delta_adapt = []
    delta_spatial = []
    
    for layer in layers:
        if results[layer] is not None:
            delta_adapt.append(results[layer]['learning_stats']['delta_adapt'])
            delta_spatial.append(results[layer]['learning_stats']['delta_spatial'])
        else:
            delta_adapt.append(0)
            delta_spatial.append(0)
    
    ax_summary2.bar(x_layers - width/2, delta_adapt, width,
                    label='Adaptation-like', color=adapt_color, alpha=0.8)
    ax_summary2.bar(x_layers + width/2, delta_spatial, width,
                    label='Spatial-like', color=spatial_color, alpha=0.8)
    
    ax_summary2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax_summary2.set_xticks(x_layers)
    ax_summary2.set_xticklabels(layers)
    ax_summary2.set_xlabel('Layer', fontsize=11)
    ax_summary2.set_ylabel('ΔSMI (Day7 - Day1)', fontsize=11)
    ax_summary2.set_title('Learning Effect', fontsize=12, fontweight='bold')
    ax_summary2.legend(loc='upper left', fontsize=9)
    ax_summary2.grid(alpha=0.3, axis='y')
    
    # Panel 3: Learning rates (correlation coefficients)
    ax_summary3 = fig.add_subplot(gs[3, 2])
    
    r_adapt_all = []
    r_spatial_all = []
    
    for layer in layers:
        if results[layer] is not None:
            r_adapt_all.append(results[layer]['learning_stats']['r_adapt'])
            r_spatial_all.append(results[layer]['learning_stats']['r_spatial'])
        else:
            r_adapt_all.append(0)
            r_spatial_all.append(0)
    
    ax_summary3.bar(x_layers - width/2, r_adapt_all, width,
                    label='Adaptation-like', color=adapt_color, alpha=0.8)
    ax_summary3.bar(x_layers + width/2, r_spatial_all, width,
                    label='Spatial-like', color=spatial_color, alpha=0.8)
    
    ax_summary3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax_summary3.set_xticks(x_layers)
    ax_summary3.set_xticklabels(layers)
    ax_summary3.set_xlabel('Layer', fontsize=11)
    ax_summary3.set_ylabel('Correlation (r)', fontsize=11)
    ax_summary3.set_title('Learning Rate\n(SMI × Session correlation)', fontsize=12, fontweight='bold')
    ax_summary3.legend(loc='upper left', fontsize=9)
    ax_summary3.set_ylim(-0.5, 1.0)
    ax_summary3.grid(alpha=0.3, axis='y')
    
    # Panel 4: Interpretation
    ax_summary4 = fig.add_subplot(gs[3, 3])
    ax_summary4.axis('off')
    
    # Determine key findings
    deep_higher_day1 = False
    superficial_learn_more = False
    
    # Check if deep layers start with higher SMI
    deep_adapt_day1 = np.nanmean([day1_adapt_means[2], day1_adapt_means[3]])  # L5, L6
    deep_spatial_day1 = np.nanmean([day1_spatial_means[2], day1_spatial_means[3]])
    sup_adapt_day1 = np.nanmean([day1_adapt_means[0], day1_adapt_means[1]])  # L2/3, L4
    sup_spatial_day1 = np.nanmean([day1_spatial_means[0], day1_spatial_means[1]])
    
    deep_day1 = (deep_adapt_day1 + deep_spatial_day1) / 2
    sup_day1 = (sup_adapt_day1 + sup_spatial_day1) / 2
    
    if deep_day1 > sup_day1 * 1.1:
        deep_higher_day1 = True
    
    # Check if superficial layers learn more
    deep_learn = np.nanmean([delta_adapt[2], delta_adapt[3], delta_spatial[2], delta_spatial[3]])
    sup_learn = np.nanmean([delta_adapt[0], delta_adapt[1], delta_spatial[0], delta_spatial[1]])
    
    if sup_learn > deep_learn * 1.2:
        superficial_learn_more = True
    
    interpretation_text = "KEY FINDINGS:\n\n"
    
    interpretation_text += "Day 1 (Initial State):\n"
    if deep_higher_day1:
        interpretation_text += "✓ Deep > Superficial\n"
        interpretation_text += "  → INNATE hypothesis\n\n"
    else:
        interpretation_text += "✗ No layer difference\n\n"
    
    interpretation_text += "Learning (Δ Day1→Day7):\n"
    if superficial_learn_more:
        interpretation_text += "✓ Superficial > Deep\n"
        interpretation_text += "  → LEARNED hypothesis\n\n"
    else:
        interpretation_text += "✗ No layer difference\n\n"
    
    interpretation_text += "───────────────────\n\n"
    interpretation_text += "Cell Type Effects:\n"
    interpretation_text += "• See heatmaps for\n"
    interpretation_text += "  session-by-session\n"
    interpretation_text += "  differences\n"
    interpretation_text += "• Check trajectories\n"
    interpretation_text += "  for learning patterns"
    
    ax_summary4.text(0.1, 0.9, interpretation_text,
                    transform=ax_summary4.transAxes,
                    fontsize=10, fontfamily='monospace',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # =========================================================================
    # Overall title
    # =========================================================================
    plt.suptitle('Within-Layer SMI Analysis: Adaptation-like vs Spatial-like Cells\n'
                 'All Sessions | Testing "Innate vs Learned" Hypothesis',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'within_layer_smi_all_sessions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure: {save_path}")
    
    return fig

# ============================================================================
# MAIN
# ============================================================================

def run_within_layer_smi_analysis(pca_data_path, figure_dir):
    """Main analysis workflow."""
    
    print("=" * 70)
    print("WITHIN-LAYER SMI ANALYSIS")
    print("Testing: Do adaptation-like vs spatial-like cells differ in")
    print("         initial SMI and learning trajectories?")
    print("=" * 70)
    
    # Load data
    data = load_data(pca_data_path)
    
    # Classify L1 cells
    classification_results = classify_l1_cells(data, threshold=EARLY_SLOPE_THRESHOLD)
    
    # Run within-layer SMI analysis
    results = analyze_smi_by_cell_type_and_layer(data, classification_results)
    
    # Visualize
    fig = plot_within_layer_smi_analysis(results, save_dir=figure_dir)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = run_within_layer_smi_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR
    )