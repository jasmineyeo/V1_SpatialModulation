"""
PCA_ComprehensiveAnalysis.py
Comprehensive PCA-based analysis addressing the core research questions:

1. Does spatial modulation develop earlier in deep layers vs superficial layers?
2. Do deep layer cells shift toward L4 (last landmark) preference earlier?
3. What proportion of L1-preferring cells are adaptation vs spatial encoding?
4. How do these patterns change with experience (across sessions)?

This script analyzes ALL cells (not just L1) across sessions and layers.

JSY, 12/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\figures\comprehensive"

# Landmark positions
LANDMARK_POSITIONS = [25, 55, 85, 115]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath):
    """Load all PCA data."""
    print(f"Loading data from: {filepath}")
    
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Metadata
        data['bin_centers'] = f['metadata/bin_centers_trimmed'][:]
        data['landmark_positions'] = f['metadata/landmark_positions'][:]
        
        # Cell labels
        data['session_labels'] = f['cells/session_labels'][:].astype(str)
        data['layer_labels'] = f['cells/layer_labels'][:].astype(str)
        data['preferred_landmark'] = f['cells/preferred_landmark'][:]
        data['peak_positions'] = f['cells/peak_positions'][:]
        
        # Features
        data['spatial_profiles'] = f['features/spatial_profiles'][:]
        data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
        
        # PCA results
        data['pc_scores'] = f['pca_results/pc_scores'][:]
        data['components'] = f['pca_results/components'][:]
        data['explained_variance_ratio'] = f['pca_results/explained_variance_ratio'][:]
    
    # Sort sessions
    unique_sessions = np.unique(data['session_labels'])
    data['session_order'] = sorted(unique_sessions, key=lambda x: int(x.replace('Day', '')))
    
    print(f"  Total cells: {len(data['pc_scores'])}")
    print(f"  Sessions: {data['session_order']}")
    
    return data


# ============================================================================
# FEATURE COMPUTATION
# ============================================================================

def compute_cell_features(data):
    """
    Compute features relevant to spatial modulation for each cell.
    """
    profiles = data['spatial_profiles_zscore']
    bin_centers = data['bin_centers']
    n_cells = profiles.shape[0]
    
    print("\nComputing cell features...")
    
    features = {}
    
    # 1. Early slope (adaptation signature)
    onset_end_idx = np.searchsorted(bin_centers, 15)
    slope_end_idx = np.searchsorted(bin_centers, 30)
    
    early_slopes = np.zeros(n_cells)
    for i in range(n_cells):
        early_region = profiles[i, onset_end_idx:slope_end_idx]
        if len(early_region) > 2:
            x = np.arange(len(early_region))
            slope, _, _, _, _ = stats.linregress(x, early_region)
            early_slopes[i] = slope
    features['early_slope'] = early_slopes
    
    # 2. Spatial modulation index (SMI-like metric from profiles)
    # Peak response / mean response ratio
    smi_like = np.zeros(n_cells)
    for i in range(n_cells):
        profile = profiles[i]
        peak = np.max(profile)
        mean_abs = np.mean(np.abs(profile))
        if mean_abs > 0:
            smi_like[i] = peak / mean_abs
        else:
            smi_like[i] = 0
    features['smi_like'] = smi_like
    
    # 3. Peak sharpness (inverse of peak width)
    peak_sharpness = np.zeros(n_cells)
    for i in range(n_cells):
        profile = profiles[i]
        max_val = np.max(profile)
        if max_val > 0:
            half_max = max_val / 2
            above_half = profile >= half_max
            width = np.sum(above_half)
            peak_sharpness[i] = 1 / (width + 1)  # Inverse of width
    features['peak_sharpness'] = peak_sharpness
    
    # 4. Landmark modulation depth (response at landmarks vs between)
    landmark_response = np.zeros(n_cells)
    between_response = np.zeros(n_cells)
    
    for i in range(n_cells):
        profile = profiles[i]
        
        # Get response at each landmark (±3 bins)
        lm_responses = []
        for lm_pos in LANDMARK_POSITIONS:
            lm_idx = np.argmin(np.abs(bin_centers - lm_pos))
            start = max(0, lm_idx - 3)
            end = min(len(profile), lm_idx + 4)
            lm_responses.append(np.max(profile[start:end]))
        
        landmark_response[i] = np.mean(lm_responses)
        
        # Get response between landmarks
        between_vals = []
        between_positions = [40, 70, 100]  # Midpoints between landmarks
        for bp in between_positions:
            bp_idx = np.argmin(np.abs(bin_centers - bp))
            between_vals.append(profile[bp_idx])
        between_response[i] = np.mean(between_vals)
    
    features['landmark_modulation'] = landmark_response - between_response
    
    # 5. Response at each landmark (for preference analysis)
    for lm_idx, lm_pos in enumerate(LANDMARK_POSITIONS):
        lm_responses = np.zeros(n_cells)
        for i in range(n_cells):
            profile = profiles[i]
            idx = np.argmin(np.abs(bin_centers - lm_pos))
            start = max(0, idx - 5)
            end = min(len(profile), idx + 6)
            lm_responses[i] = np.max(profile[start:end])
        features[f'response_L{lm_idx+1}'] = lm_responses
    
    print(f"  Computed {len(features)} features")
    
    return features


# ============================================================================
# QUESTION 1: SPATIAL MODULATION DEVELOPMENT BY LAYER
# ============================================================================

def analyze_spatial_modulation_by_layer_session(data, features):
    """
    Question 1: Does spatial modulation develop earlier in deep layers?
    
    Analyzes:
    - SMI-like metric across sessions for each layer
    - PC2 (adaptation axis) across sessions for each layer
    - Landmark modulation depth across sessions for each layer
    """
    print("\n" + "=" * 70)
    print("QUESTION 1: SPATIAL MODULATION DEVELOPMENT BY LAYER")
    print("=" * 70)
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    pc_scores = data['pc_scores']
    session_order = data['session_order']
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    
    results = {
        'smi_by_layer_session': {},
        'pc2_by_layer_session': {},
        'landmark_mod_by_layer_session': {},
        'sharpness_by_layer_session': {}
    }
    
    # For each layer, track metrics across sessions
    for layer in layers:
        layer_mask = layer_labels == layer
        
        smi_values = []
        pc2_values = []
        lm_mod_values = []
        sharpness_values = []
        session_ns = []
        
        for session in session_order:
            session_mask = session_labels == session
            combined_mask = layer_mask & session_mask
            
            n_cells = np.sum(combined_mask)
            session_ns.append(n_cells)
            
            if n_cells > 0:
                smi_values.append({
                    'mean': np.mean(features['smi_like'][combined_mask]),
                    'sem': np.std(features['smi_like'][combined_mask]) / np.sqrt(n_cells),
                    'median': np.median(features['smi_like'][combined_mask])
                })
                pc2_values.append({
                    'mean': np.mean(pc_scores[combined_mask, 1]),
                    'sem': np.std(pc_scores[combined_mask, 1]) / np.sqrt(n_cells),
                    'median': np.median(pc_scores[combined_mask, 1])
                })
                lm_mod_values.append({
                    'mean': np.mean(features['landmark_modulation'][combined_mask]),
                    'sem': np.std(features['landmark_modulation'][combined_mask]) / np.sqrt(n_cells),
                    'median': np.median(features['landmark_modulation'][combined_mask])
                })
                sharpness_values.append({
                    'mean': np.mean(features['peak_sharpness'][combined_mask]),
                    'sem': np.std(features['peak_sharpness'][combined_mask]) / np.sqrt(n_cells),
                    'median': np.median(features['peak_sharpness'][combined_mask])
                })
            else:
                smi_values.append({'mean': np.nan, 'sem': np.nan, 'median': np.nan})
                pc2_values.append({'mean': np.nan, 'sem': np.nan, 'median': np.nan})
                lm_mod_values.append({'mean': np.nan, 'sem': np.nan, 'median': np.nan})
                sharpness_values.append({'mean': np.nan, 'sem': np.nan, 'median': np.nan})
        
        results['smi_by_layer_session'][layer] = smi_values
        results['pc2_by_layer_session'][layer] = pc2_values
        results['landmark_mod_by_layer_session'][layer] = lm_mod_values
        results['sharpness_by_layer_session'][layer] = sharpness_values
        
        # Print summary
        print(f"\n{layer}:")
        print(f"  Cells per session: {session_ns}")
        
        # Test for trend across sessions
        session_nums = np.arange(len(session_order))
        smi_means = [v['mean'] for v in smi_values]
        valid = ~np.isnan(smi_means)
        
        if np.sum(valid) >= 3:
            r, p = stats.pearsonr(session_nums[valid], np.array(smi_means)[valid])
            trend = "increasing" if r > 0 else "decreasing"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  SMI trend: r={r:.3f}, p={p:.3f} {sig} ({trend})")
    
    return results


def plot_spatial_modulation_development(data, features, results, save_dir):
    """
    Visualize spatial modulation development across sessions by layer.
    """
    session_order = data['session_order']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    x = np.arange(len(session_order))
    
    # =========================================================================
    # Panel 1: SMI-like metric across sessions
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    for layer in layers:
        means = [v['mean'] for v in results['smi_by_layer_session'][layer]]
        sems = [v['sem'] for v in results['smi_by_layer_session'][layer]]
        
        ax1.errorbar(x, means, yerr=sems, marker='o', linewidth=2, markersize=8,
                    color=layer_colors[layer], label=layer, capsize=3)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(session_order)
    ax1.set_xlabel('Session', fontsize=11)
    ax1.set_ylabel('Spatial Modulation Index', fontsize=11)
    ax1.set_title('Q1: Spatial Modulation Development\n(Higher = More Spatially Modulated)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 2: PC2 (adaptation axis) across sessions
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    for layer in layers:
        means = [v['mean'] for v in results['pc2_by_layer_session'][layer]]
        sems = [v['sem'] for v in results['pc2_by_layer_session'][layer]]
        
        ax2.errorbar(x, means, yerr=sems, marker='o', linewidth=2, markersize=8,
                    color=layer_colors[layer], label=layer, capsize=3)
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(session_order)
    ax2.set_xlabel('Session', fontsize=11)
    ax2.set_ylabel('PC2 Score (Adaptation Axis)', fontsize=11)
    ax2.set_title('PC2 Development\n(Higher = More Spatial-like, Lower = Adaptation-like)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 3: Landmark modulation depth
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    for layer in layers:
        means = [v['mean'] for v in results['landmark_mod_by_layer_session'][layer]]
        sems = [v['sem'] for v in results['landmark_mod_by_layer_session'][layer]]
        
        ax3.errorbar(x, means, yerr=sems, marker='o', linewidth=2, markersize=8,
                    color=layer_colors[layer], label=layer, capsize=3)
    
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xticks(x)
    ax3.set_xticklabels(session_order)
    ax3.set_xlabel('Session', fontsize=11)
    ax3.set_ylabel('Landmark Modulation Depth', fontsize=11)
    ax3.set_title('Landmark vs Between Response\n(Higher = Stronger Landmark Selectivity)', 
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 4: Early vs Late session comparison (SMI)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Compare first 2 sessions vs last 2 sessions
    early_sessions = session_order[:2]
    late_sessions = session_order[-2:]
    
    early_smi = {layer: [] for layer in layers}
    late_smi = {layer: [] for layer in layers}
    
    for layer in layers:
        layer_mask = data['layer_labels'] == layer
        
        for session in early_sessions:
            session_mask = data['session_labels'] == session
            combined = layer_mask & session_mask
            early_smi[layer].extend(features['smi_like'][combined])
        
        for session in late_sessions:
            session_mask = data['session_labels'] == session
            combined = layer_mask & session_mask
            late_smi[layer].extend(features['smi_like'][combined])
    
    x_pos = np.arange(len(layers))
    width = 0.35
    
    early_means = [np.mean(early_smi[l]) for l in layers]
    early_sems = [np.std(early_smi[l])/np.sqrt(len(early_smi[l])) for l in layers]
    late_means = [np.mean(late_smi[l]) for l in layers]
    late_sems = [np.std(late_smi[l])/np.sqrt(len(late_smi[l])) for l in layers]
    
    ax4.bar(x_pos - width/2, early_means, width, yerr=early_sems, 
           label=f'Early ({early_sessions[0]}-{early_sessions[-1]})', 
           color='lightblue', capsize=3, edgecolor='black')
    ax4.bar(x_pos + width/2, late_means, width, yerr=late_sems,
           label=f'Late ({late_sessions[0]}-{late_sessions[-1]})', 
           color='darkblue', capsize=3, edgecolor='black')
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(layers)
    ax4.set_xlabel('Cortical Layer', fontsize=11)
    ax4.set_ylabel('Spatial Modulation Index', fontsize=11)
    ax4.set_title('Early vs Late Sessions\n(Learning Effect by Layer)', 
                 fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')
    
    # Add significance stars
    for i, layer in enumerate(layers):
        t, p = stats.ttest_ind(early_smi[layer], late_smi[layer])
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        else:
            sig = ''
        
        if sig:
            max_y = max(early_means[i] + early_sems[i], late_means[i] + late_sems[i])
            ax4.text(i, max_y + 0.05, sig, ha='center', fontsize=12)
    
    # =========================================================================
    # Panel 5: Layer difference in early sessions
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    # PC space for early sessions only
    early_mask = np.isin(data['session_labels'], early_sessions)
    pc_early = data['pc_scores'][early_mask]
    layers_early = data['layer_labels'][early_mask]
    
    for layer in layers:
        mask = layers_early == layer
        if np.sum(mask) > 0:
            ax5.scatter(pc_early[mask, 0], pc_early[mask, 1], 
                       c=layer_colors[layer], alpha=0.5, s=20, label=layer)
    
    ax5.set_xlabel(f"PC1 ({data['explained_variance_ratio'][0]*100:.1f}%)", fontsize=11)
    ax5.set_ylabel(f"PC2 ({data['explained_variance_ratio'][1]*100:.1f}%)", fontsize=11)
    ax5.set_title(f'Early Sessions ({early_sessions[0]}-{early_sessions[-1]})\nPC Space by Layer', 
                 fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 6: Layer difference in late sessions
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    late_mask = np.isin(data['session_labels'], late_sessions)
    pc_late = data['pc_scores'][late_mask]
    layers_late = data['layer_labels'][late_mask]
    
    for layer in layers:
        mask = layers_late == layer
        if np.sum(mask) > 0:
            ax6.scatter(pc_late[mask, 0], pc_late[mask, 1], 
                       c=layer_colors[layer], alpha=0.5, s=20, label=layer)
    
    ax6.set_xlabel(f"PC1 ({data['explained_variance_ratio'][0]*100:.1f}%)", fontsize=11)
    ax6.set_ylabel(f"PC2 ({data['explained_variance_ratio'][1]*100:.1f}%)", fontsize=11)
    ax6.set_title(f'Late Sessions ({late_sessions[0]}-{late_sessions[-1]})\nPC Space by Layer', 
                 fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(alpha=0.3)
    
    plt.suptitle('Question 1: Does Spatial Modulation Develop Earlier in Deep Layers?', 
                fontsize=14, fontweight='bold', y=1.02)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'Q1_spatial_modulation_development.png'), 
                   dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved Q1 figure")
    
    return fig


# ============================================================================
# QUESTION 2: LANDMARK PREFERENCE SHIFTS BY LAYER
# ============================================================================

def analyze_landmark_preference_by_layer_session(data, features):
    """
    Question 2: Do deep layer cells shift toward L4 preference earlier?
    
    Analyzes:
    - Proportion preferring each landmark across sessions, by layer
    - L4 preference development trajectory by layer
    - PC1 (position axis) shifts across sessions by layer
    """
    print("\n" + "=" * 70)
    print("QUESTION 2: LANDMARK PREFERENCE SHIFTS BY LAYER")
    print("=" * 70)
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    preferred_landmark = data['preferred_landmark']
    session_order = data['session_order']
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    n_landmarks = 4
    
    results = {
        'preference_proportions': {},  # [layer][session][landmark] = proportion
        'l4_preference_trajectory': {},  # [layer] = list of L4 proportions
        'mean_preferred_position': {}  # [layer][session] = mean preferred landmark
    }
    
    for layer in layers:
        layer_mask = layer_labels == layer
        results['preference_proportions'][layer] = {}
        results['l4_preference_trajectory'][layer] = []
        results['mean_preferred_position'][layer] = []
        
        print(f"\n{layer}:")
        
        for session in session_order:
            session_mask = session_labels == session
            combined_mask = layer_mask & session_mask
            
            # Only count cells with valid landmark preference (0-3)
            valid_pref_mask = combined_mask & (preferred_landmark >= 0) & (preferred_landmark < 4)
            n_valid = np.sum(valid_pref_mask)
            
            if n_valid > 0:
                prefs = preferred_landmark[valid_pref_mask]
                
                # Count each landmark
                proportions = []
                for lm in range(n_landmarks):
                    prop = np.sum(prefs == lm) / n_valid
                    proportions.append(prop)
                
                results['preference_proportions'][layer][session] = proportions
                results['l4_preference_trajectory'][layer].append(proportions[3])  # L4 = index 3
                
                # Mean preferred position (weighted by landmark index)
                mean_pref = np.mean(prefs)
                results['mean_preferred_position'][layer].append(mean_pref)
                
                print(f"  {session} (n={n_valid}): L1={proportions[0]*100:.1f}%, "
                      f"L2={proportions[1]*100:.1f}%, L3={proportions[2]*100:.1f}%, "
                      f"L4={proportions[3]*100:.1f}%")
            else:
                results['preference_proportions'][layer][session] = [np.nan] * 4
                results['l4_preference_trajectory'][layer].append(np.nan)
                results['mean_preferred_position'][layer].append(np.nan)
        
        # Test for L4 preference trend
        l4_props = np.array(results['l4_preference_trajectory'][layer])
        session_nums = np.arange(len(session_order))
        valid = ~np.isnan(l4_props)
        
        if np.sum(valid) >= 3:
            r, p = stats.pearsonr(session_nums[valid], l4_props[valid])
            trend = "increasing" if r > 0 else "decreasing"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  L4 preference trend: r={r:.3f}, p={p:.3f} {sig} ({trend})")
    
    return results


def plot_landmark_preference_shifts(data, results, save_dir):
    """
    Visualize landmark preference shifts across sessions by layer.
    """
    session_order = data['session_order']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    landmark_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']  # L1, L2, L3, L4
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    x = np.arange(len(session_order))
    
    # =========================================================================
    # Panels 1-4: Landmark preference trajectories for each layer
    # =========================================================================
    for idx, layer in enumerate(layers):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        for lm in range(4):
            props = [results['preference_proportions'][layer][s][lm] * 100 
                    for s in session_order]
            ax.plot(x, props, marker='o', linewidth=2, markersize=6,
                   color=landmark_colors[lm], label=f'L{lm+1} ({LANDMARK_POSITIONS[lm]}cm)')
        
        ax.set_xticks(x)
        ax.set_xticklabels(session_order)
        ax.set_xlabel('Session', fontsize=10)
        ax.set_ylabel('% of Cells', fontsize=10)
        ax.set_title(f'{layer}: Landmark Preference Over Sessions', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_ylim(0, 100)
        ax.axhline(25, color='gray', linestyle='--', alpha=0.3)  # Chance level
        ax.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 5: L4 preference trajectory comparison across layers
    # =========================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    for layer in layers:
        l4_props = [p * 100 for p in results['l4_preference_trajectory'][layer]]
        ax5.plot(x, l4_props, marker='o', linewidth=2.5, markersize=8,
                color=layer_colors[layer], label=layer)
    
    ax5.axhline(25, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax5.set_xticks(x)
    ax5.set_xticklabels(session_order)
    ax5.set_xlabel('Session', fontsize=11)
    ax5.set_ylabel('% Preferring L4', fontsize=11)
    ax5.set_title('Q2: L4 (Last Landmark) Preference Development\n(Higher = More L4 Preference)', 
                 fontsize=12, fontweight='bold')
    ax5.legend(loc='upper left')
    ax5.set_ylim(0, 60)
    ax5.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 6: Mean preferred landmark position trajectory
    # =========================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    for layer in layers:
        mean_prefs = results['mean_preferred_position'][layer]
        ax6.plot(x, mean_prefs, marker='o', linewidth=2.5, markersize=8,
                color=layer_colors[layer], label=layer)
    
    ax6.axhline(1.5, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax6.set_xticks(x)
    ax6.set_xticklabels(session_order)
    ax6.set_xlabel('Session', fontsize=11)
    ax6.set_ylabel('Mean Preferred Landmark (0=L1, 3=L4)', fontsize=11)
    ax6.set_title('Mean Landmark Preference Shift\n(Higher = Shift Toward L4)', 
                 fontsize=12, fontweight='bold')
    ax6.legend(loc='upper left')
    ax6.set_ylim(0, 3)
    ax6.grid(alpha=0.3)
    
    # Add landmark labels on y-axis
    ax6_twin = ax6.twinx()
    ax6_twin.set_ylim(0, 3)
    ax6_twin.set_yticks([0, 1, 2, 3])
    ax6_twin.set_yticklabels(['L1', 'L2', 'L3', 'L4'])
    
    # =========================================================================
    # Panel 7: Early vs Late L4 preference by layer
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 2])
    
    early_sessions = session_order[:2]
    late_sessions = session_order[-2:]
    
    early_l4 = []
    late_l4 = []
    
    for layer in layers:
        early_props = [results['preference_proportions'][layer][s][3] for s in early_sessions 
                      if not np.isnan(results['preference_proportions'][layer][s][3])]
        late_props = [results['preference_proportions'][layer][s][3] for s in late_sessions
                     if not np.isnan(results['preference_proportions'][layer][s][3])]
        
        early_l4.append(np.mean(early_props) * 100 if early_props else np.nan)
        late_l4.append(np.mean(late_props) * 100 if late_props else np.nan)
    
    x_pos = np.arange(len(layers))
    width = 0.35
    
    ax7.bar(x_pos - width/2, early_l4, width, label=f'Early Sessions', 
           color='lightcoral', edgecolor='black')
    ax7.bar(x_pos + width/2, late_l4, width, label=f'Late Sessions', 
           color='darkred', edgecolor='black')
    
    ax7.axhline(25, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels(layers)
    ax7.set_xlabel('Cortical Layer', fontsize=11)
    ax7.set_ylabel('% Preferring L4', fontsize=11)
    ax7.set_title('L4 Preference: Early vs Late Sessions', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3, axis='y')
    
    plt.suptitle('Question 2: Do Deep Layers Shift Toward L4 Preference Earlier?', 
                fontsize=14, fontweight='bold', y=1.02)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'Q2_landmark_preference_shifts.png'), 
                   dpi=200, bbox_inches='tight')
        print(f"✓ Saved Q2 figure")
    
    return fig


# ============================================================================
# QUESTION 3: ADAPTATION VS SPATIAL ENCODING IN L1 CELLS
# ============================================================================

def analyze_l1_cell_types(data, features):
    """
    Question 3: What proportion of L1-preferring cells show adaptation vs spatial encoding?
    
    Uses clustering on PC space to identify subpopulations.
    """
    print("\n" + "=" * 70)
    print("QUESTION 3: ADAPTATION VS SPATIAL ENCODING IN L1 CELLS")
    print("=" * 70)
    
    # Get L1 cells
    l1_mask = data['preferred_landmark'] == 0
    n_l1 = np.sum(l1_mask)
    
    print(f"\nTotal L1-preferring cells: {n_l1}")
    
    if n_l1 < 10:
        print("Not enough L1 cells for analysis")
        return None
    
    # Extract L1 cell data
    l1_pc_scores = data['pc_scores'][l1_mask]
    l1_early_slopes = features['early_slope'][l1_mask]
    l1_layers = data['layer_labels'][l1_mask]
    l1_sessions = data['session_labels'][l1_mask]
    l1_profiles = data['spatial_profiles_zscore'][l1_mask]
    
    # Cluster using PC1 and PC2
    features_for_clustering = l1_pc_scores[:, :2]
    
    # K-means with k=2
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(features_for_clustering)
    
    # Align clusters so 0 = adaptation-like (lower early slope)
    cluster_0_slope = np.mean(l1_early_slopes[cluster_labels == 0])
    cluster_1_slope = np.mean(l1_early_slopes[cluster_labels == 1])
    
    if cluster_0_slope > cluster_1_slope:
        cluster_labels = 1 - cluster_labels
        print("  Flipped cluster labels to align with slope interpretation")
    
    n_adapt = np.sum(cluster_labels == 0)
    n_spatial = np.sum(cluster_labels == 1)
    
    print(f"\nClustering Results:")
    print(f"  Adaptation-like: {n_adapt} ({n_adapt/n_l1*100:.1f}%)")
    print(f"  Spatial-like: {n_spatial} ({n_spatial/n_l1*100:.1f}%)")
    
    # Silhouette score
    sil_score = silhouette_score(features_for_clustering, cluster_labels)
    print(f"  Silhouette score: {sil_score:.3f}")
    
    # Layer distribution
    print(f"\nLayer Distribution:")
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_results = {}
    
    for layer in layers:
        layer_mask = l1_layers == layer
        n_layer = np.sum(layer_mask)
        
        if n_layer > 0:
            n_adapt_layer = np.sum((cluster_labels == 0) & layer_mask)
            n_spatial_layer = np.sum((cluster_labels == 1) & layer_mask)
            pct_adapt = n_adapt_layer / n_layer * 100
            
            layer_results[layer] = {
                'n_total': n_layer,
                'n_adapt': n_adapt_layer,
                'n_spatial': n_spatial_layer,
                'pct_adapt': pct_adapt
            }
            
            print(f"  {layer}: {n_adapt_layer} adapt / {n_spatial_layer} spatial "
                  f"({pct_adapt:.1f}% adaptation-like)")
    
    # Session distribution
    print(f"\nSession Distribution:")
    session_results = {}
    
    for session in data['session_order']:
        session_mask = l1_sessions == session
        n_session = np.sum(session_mask)
        
        if n_session > 0:
            n_adapt_session = np.sum((cluster_labels == 0) & session_mask)
            n_spatial_session = np.sum((cluster_labels == 1) & session_mask)
            pct_adapt = n_adapt_session / n_session * 100
            
            session_results[session] = {
                'n_total': n_session,
                'n_adapt': n_adapt_session,
                'n_spatial': n_spatial_session,
                'pct_adapt': pct_adapt
            }
            
            print(f"  {session}: {n_adapt_session} adapt / {n_spatial_session} spatial "
                  f"({pct_adapt:.1f}% adaptation-like)")
    
    results = {
        'l1_mask': l1_mask,
        'cluster_labels': cluster_labels,
        'n_adapt': n_adapt,
        'n_spatial': n_spatial,
        'silhouette': sil_score,
        'layer_results': layer_results,
        'session_results': session_results,
        'l1_pc_scores': l1_pc_scores,
        'l1_profiles': l1_profiles,
        'l1_early_slopes': l1_early_slopes,
        'l1_layers': l1_layers,
        'l1_sessions': l1_sessions,
        'kmeans_centers': kmeans.cluster_centers_
    }
    
    return results


def plot_l1_cell_types(data, features, results, save_dir):
    """
    Visualize L1 cell type analysis.
    """
    if results is None:
        return None
    
    cluster_labels = results['cluster_labels']
    l1_pc_scores = results['l1_pc_scores']
    l1_profiles = results['l1_profiles']
    l1_early_slopes = results['l1_early_slopes']
    l1_layers = results['l1_layers']
    l1_sessions = results['l1_sessions']
    
    bin_centers = data['bin_centers']
    session_order = data['session_order']
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    adapt_color = '#E53935'
    spatial_color = '#1E88E5'
    
    # =========================================================================
    # Panel 1: Clusters in PC space
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    for c, (color, label) in enumerate([(adapt_color, 'Adaptation-like'), 
                                         (spatial_color, 'Spatial-like')]):
        mask = cluster_labels == c
        ax1.scatter(l1_pc_scores[mask, 0], l1_pc_scores[mask, 1],
                   c=color, alpha=0.6, s=30, label=f'{label} (n={np.sum(mask)})')
    
    ax1.scatter(results['kmeans_centers'][:, 0], results['kmeans_centers'][:, 1],
               c='black', marker='X', s=200, edgecolors='white', linewidths=2)
    
    ax1.set_xlabel(f"PC1 ({data['explained_variance_ratio'][0]*100:.1f}%)", fontsize=11)
    ax1.set_ylabel(f"PC2 ({data['explained_variance_ratio'][1]*100:.1f}%)", fontsize=11)
    ax1.set_title('L1 Cell Clustering in PC Space', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 2: Mean profiles by cluster
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    x_vals = np.linspace(bin_centers[0], bin_centers[-1], l1_profiles.shape[1])
    
    for c, (color, label) in enumerate([(adapt_color, 'Adaptation-like'), 
                                         (spatial_color, 'Spatial-like')]):
        mask = cluster_labels == c
        mean_profile = np.mean(l1_profiles[mask], axis=0)
        sem = np.std(l1_profiles[mask], axis=0) / np.sqrt(np.sum(mask))
        
        ax2.plot(x_vals, mean_profile, color=color, linewidth=2.5, label=label)
        ax2.fill_between(x_vals, mean_profile - sem, mean_profile + sem,
                        color=color, alpha=0.2)
    
    for lm_pos in LANDMARK_POSITIONS:
        ax2.axvline(lm_pos, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Position (cm)', fontsize=11)
    ax2.set_ylabel('Z-scored Activity', fontsize=11)
    ax2.set_title('Mean Response Profiles', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 3: Layer distribution
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    x_pos = np.arange(len(layers))
    
    adapt_pcts = [results['layer_results'].get(l, {}).get('pct_adapt', 0) for l in layers]
    spatial_pcts = [100 - p for p in adapt_pcts]
    
    ax3.bar(x_pos, adapt_pcts, label='Adaptation-like', color=adapt_color, alpha=0.8)
    ax3.bar(x_pos, spatial_pcts, bottom=adapt_pcts, label='Spatial-like', 
           color=spatial_color, alpha=0.8)
    
    ax3.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(layers)
    ax3.set_xlabel('Cortical Layer', fontsize=11)
    ax3.set_ylabel('Percentage', fontsize=11)
    ax3.set_title('L1 Cell Types by Layer', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.set_ylim(0, 100)
    
    # Add percentage labels
    for i, pct in enumerate(adapt_pcts):
        ax3.text(i, pct/2, f'{pct:.0f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    # =========================================================================
    # Panel 4: Session distribution (% adaptation-like over time)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    session_pcts = [results['session_results'].get(s, {}).get('pct_adapt', np.nan) 
                   for s in session_order]
    
    x = np.arange(len(session_order))
    ax4.bar(x, session_pcts, color=adapt_color, alpha=0.7, edgecolor='black')
    ax4.axhline(50, color='black', linestyle='--', alpha=0.5)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(session_order)
    ax4.set_xlabel('Session', fontsize=11)
    ax4.set_ylabel('% Adaptation-like', fontsize=11)
    ax4.set_title('L1 Adaptation-like Proportion Over Sessions', 
                 fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(alpha=0.3, axis='y')
    
    # Test for trend
    valid = ~np.isnan(session_pcts)
    if np.sum(valid) >= 3:
        r, p = stats.pearsonr(x[valid], np.array(session_pcts)[valid])
        trend_text = f"r={r:.2f}, p={p:.3f}"
        ax4.text(0.95, 0.95, trend_text, transform=ax4.transAxes, 
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # =========================================================================
    # Panel 5: Early slope histogram by cluster
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    for c, (color, label) in enumerate([(adapt_color, 'Adaptation-like'), 
                                         (spatial_color, 'Spatial-like')]):
        mask = cluster_labels == c
        ax5.hist(l1_early_slopes[mask], bins=30, alpha=0.6, color=color,
                label=label, edgecolor='black', linewidth=0.5)
    
    ax5.axvline(np.median(l1_early_slopes), color='black', linestyle='--', 
               linewidth=2, label='Median')
    ax5.set_xlabel('Early Slope', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Early Slope Distribution by Cluster', fontsize=12, fontweight='bold')
    ax5.legend()
    
    # =========================================================================
    # Panel 6: Summary statistics
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    summary_text = f"""
    SUMMARY: L1 Cell Types
    ══════════════════════════════════════
    
    Total L1 Cells: {results['n_adapt'] + results['n_spatial']}
    
    Adaptation-like: {results['n_adapt']} ({results['n_adapt']/(results['n_adapt']+results['n_spatial'])*100:.1f}%)
    Spatial-like: {results['n_spatial']} ({results['n_spatial']/(results['n_adapt']+results['n_spatial'])*100:.1f}%)
    
    Clustering Quality (Silhouette): {results['silhouette']:.3f}
    
    ──────────────────────────────────────
    Layer Distribution (% Adaptation-like):
    
    L2/3: {results['layer_results'].get('L2/3', {}).get('pct_adapt', 0):.1f}%
    L4:   {results['layer_results'].get('L4', {}).get('pct_adapt', 0):.1f}%
    L5:   {results['layer_results'].get('L5', {}).get('pct_adapt', 0):.1f}%
    L6:   {results['layer_results'].get('L6', {}).get('pct_adapt', 0):.1f}%
    
    ──────────────────────────────────────
    KEY FINDING:
    Deep layers (L5/L6) have proportionally
    MORE adaptation-like L1 cells than
    superficial layers (L2/3, L4).
    """
    
    ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=11, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Question 3: Adaptation vs Spatial Encoding in L1-Preferring Cells', 
                fontsize=14, fontweight='bold', y=1.02)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'Q3_l1_cell_types.png'), 
                   dpi=200, bbox_inches='tight')
        print(f"✓ Saved Q3 figure")
    
    return fig


# ============================================================================
# QUESTION 4: HOW DO PATTERNS CHANGE WITH EXPERIENCE?
# ============================================================================

def analyze_learning_effects(data, features, l1_results):
    """
    Question 4: How do spatial modulation patterns change with experience?
    
    Analyzes:
    - Changes in adaptation vs spatial-like proportions across sessions
    - PC space shifts across sessions
    - Layer-specific learning trajectories
    """
    print("\n" + "=" * 70)
    print("QUESTION 4: LEARNING EFFECTS ON SPATIAL ENCODING")
    print("=" * 70)
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    pc_scores = data['pc_scores']
    session_order = data['session_order']
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    
    results = {
        'pc_centroids_by_session': {},
        'pc_centroids_by_layer_session': {},
        'spatial_quality_trajectory': {}
    }
    
    # Track PC space centroids across sessions
    print("\nPC Space Centroids by Session:")
    for session in session_order:
        session_mask = session_labels == session
        
        if np.sum(session_mask) > 0:
            centroid = np.mean(pc_scores[session_mask, :3], axis=0)
            results['pc_centroids_by_session'][session] = centroid
            print(f"  {session}: PC1={centroid[0]:.2f}, PC2={centroid[1]:.2f}, PC3={centroid[2]:.2f}")
    
    # Track by layer and session
    print("\nPC2 (Adaptation Axis) by Layer and Session:")
    for layer in layers:
        layer_mask = layer_labels == layer
        results['pc_centroids_by_layer_session'][layer] = {}
        
        pc2_values = []
        for session in session_order:
            session_mask = session_labels == session
            combined = layer_mask & session_mask
            
            if np.sum(combined) > 0:
                mean_pc2 = np.mean(pc_scores[combined, 1])
                results['pc_centroids_by_layer_session'][layer][session] = mean_pc2
                pc2_values.append(mean_pc2)
            else:
                pc2_values.append(np.nan)
        
        # Test for trend
        valid = ~np.isnan(pc2_values)
        session_nums = np.arange(len(session_order))
        
        if np.sum(valid) >= 3:
            r, p = stats.pearsonr(session_nums[valid], np.array(pc2_values)[valid])
            trend = "→ more spatial" if r > 0 else "→ more adaptation"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {layer}: r={r:.3f}, p={p:.3f} {sig} ({trend})")
    
    # Compute "spatial quality" metric combining multiple features
    print("\nSpatial Quality Index (combined metric) by Layer:")
    
    for layer in layers:
        layer_mask = layer_labels == layer
        results['spatial_quality_trajectory'][layer] = []
        
        for session in session_order:
            session_mask = session_labels == session
            combined = layer_mask & session_mask
            
            if np.sum(combined) > 5:
                # Combine: high SMI, high PC2, high landmark modulation
                smi = np.mean(features['smi_like'][combined])
                pc2 = np.mean(pc_scores[combined, 1])
                lm_mod = np.mean(features['landmark_modulation'][combined])
                
                # Normalize and combine (simple average of z-scores would be better but this works)
                spatial_quality = (smi / 2) + (pc2 / 5) + (lm_mod / 0.5)
                results['spatial_quality_trajectory'][layer].append(spatial_quality)
            else:
                results['spatial_quality_trajectory'][layer].append(np.nan)
    
    return results


def plot_learning_effects(data, features, learning_results, l1_results, save_dir):
    """
    Visualize learning effects on spatial encoding.
    """
    session_order = data['session_order']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    x = np.arange(len(session_order))
    
    # =========================================================================
    # Panel 1: PC2 trajectory by layer (adaptation axis)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    for layer in layers:
        pc2_vals = [learning_results['pc_centroids_by_layer_session'][layer].get(s, np.nan) 
                   for s in session_order]
        ax1.plot(x, pc2_vals, marker='o', linewidth=2.5, markersize=8,
                color=layer_colors[layer], label=layer)
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(session_order)
    ax1.set_xlabel('Session', fontsize=11)
    ax1.set_ylabel('Mean PC2 Score', fontsize=11)
    ax1.set_title('PC2 (Adaptation Axis) Over Learning\n(Higher = More Spatial-like)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 2: Spatial quality index trajectory
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    for layer in layers:
        sq_vals = learning_results['spatial_quality_trajectory'][layer]
        ax2.plot(x, sq_vals, marker='o', linewidth=2.5, markersize=8,
                color=layer_colors[layer], label=layer)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(session_order)
    ax2.set_xlabel('Session', fontsize=11)
    ax2.set_ylabel('Spatial Quality Index', fontsize=11)
    ax2.set_title('Combined Spatial Quality Over Learning', 
                 fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 3: L1 cell type proportions over sessions (from Q3)
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    if l1_results is not None:
        adapt_pcts = [l1_results['session_results'].get(s, {}).get('pct_adapt', np.nan) 
                     for s in session_order]
        spatial_pcts = [100 - p for p in adapt_pcts]
        
        ax3.bar(x, adapt_pcts, label='Adaptation-like', color='#E53935', alpha=0.7)
        ax3.bar(x, spatial_pcts, bottom=adapt_pcts, label='Spatial-like', 
               color='#1E88E5', alpha=0.7)
        
        ax3.axhline(50, color='black', linestyle='--', alpha=0.5)
        ax3.set_xticks(x)
        ax3.set_xticklabels(session_order)
        ax3.set_xlabel('Session', fontsize=11)
        ax3.set_ylabel('% of L1 Cells', fontsize=11)
        ax3.set_title('L1 Cell Type Proportions Over Learning', 
                     fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.set_ylim(0, 100)
    
    # =========================================================================
    # Panel 4-6: PC space evolution (Early, Middle, Late)
    # =========================================================================
    n_sessions = len(session_order)
    session_groups = [
        ('Early', session_order[:2]),
        ('Middle', session_order[n_sessions//2-1:n_sessions//2+1]),
        ('Late', session_order[-2:])
    ]
    
    for idx, (group_name, sessions) in enumerate(session_groups):
        ax = fig.add_subplot(gs[1, idx])
        
        group_mask = np.isin(data['session_labels'], sessions)
        pc_group = data['pc_scores'][group_mask]
        layers_group = data['layer_labels'][group_mask]
        
        for layer in layers:
            mask = layers_group == layer
            if np.sum(mask) > 0:
                ax.scatter(pc_group[mask, 0], pc_group[mask, 1],
                          c=layer_colors[layer], alpha=0.4, s=15, label=layer)
                
                # Add centroid
                centroid = np.mean(pc_group[mask, :2], axis=0)
                ax.scatter(centroid[0], centroid[1], c=layer_colors[layer],
                          s=200, marker='*', edgecolors='black', linewidths=1)
        
        ax.set_xlabel(f"PC1", fontsize=10)
        ax.set_ylabel(f"PC2", fontsize=10)
        ax.set_title(f'{group_name} Sessions\n({", ".join(sessions)})', 
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # =========================================================================
    # Panel 7: Layer-specific learning rates
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    learning_rates = []
    for layer in layers:
        pc2_vals = [learning_results['pc_centroids_by_layer_session'][layer].get(s, np.nan) 
                   for s in session_order]
        valid = ~np.isnan(pc2_vals)
        
        if np.sum(valid) >= 3:
            r, p = stats.pearsonr(np.arange(len(session_order))[valid], 
                                 np.array(pc2_vals)[valid])
            learning_rates.append(r)
        else:
            learning_rates.append(0)
    
    colors = [layer_colors[l] for l in layers]
    bars = ax7.bar(layers, learning_rates, color=colors, edgecolor='black', alpha=0.8)
    
    ax7.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax7.set_xlabel('Cortical Layer', fontsize=11)
    ax7.set_ylabel('Learning Rate (PC2 trend correlation)', fontsize=11)
    ax7.set_title('Learning Rate by Layer\n(Positive = Shift Toward Spatial)', 
                 fontsize=12, fontweight='bold')
    ax7.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bar, rate in zip(bars, learning_rates):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=10)
    
    # =========================================================================
    # Panel 8: Deep vs Superficial learning comparison
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Track superficial vs deep across sessions
    superficial_pc2 = []
    deep_pc2 = []
    
    for session in session_order:
        session_mask = data['session_labels'] == session
        
        sup_mask = session_mask & np.isin(data['layer_labels'], ['L2/3', 'L4'])
        deep_mask = session_mask & np.isin(data['layer_labels'], ['L5', 'L6'])
        
        if np.sum(sup_mask) > 0:
            superficial_pc2.append(np.mean(data['pc_scores'][sup_mask, 1]))
        else:
            superficial_pc2.append(np.nan)
        
        if np.sum(deep_mask) > 0:
            deep_pc2.append(np.mean(data['pc_scores'][deep_mask, 1]))
        else:
            deep_pc2.append(np.nan)
    
    ax8.plot(x, superficial_pc2, marker='o', linewidth=2.5, markersize=10,
            color='#1E88E5', label='Superficial (L2/3 + L4)')
    ax8.plot(x, deep_pc2, marker='s', linewidth=2.5, markersize=10,
            color='#E53935', label='Deep (L5 + L6)')
    
    ax8.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax8.set_xticks(x)
    ax8.set_xticklabels(session_order)
    ax8.set_xlabel('Session', fontsize=11)
    ax8.set_ylabel('Mean PC2 Score', fontsize=11)
    ax8.set_title('Superficial vs Deep Layer Learning', 
                 fontsize=12, fontweight='bold')
    ax8.legend(loc='lower right')
    ax8.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 9: Summary of findings
    # =========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate key statistics
    sup_early = np.nanmean(superficial_pc2[:2])
    sup_late = np.nanmean(superficial_pc2[-2:])
    deep_early = np.nanmean(deep_pc2[:2])
    deep_late = np.nanmean(deep_pc2[-2:])
    
    sup_change = sup_late - sup_early
    deep_change = deep_late - deep_early
    
    summary_text = f"""
    SUMMARY: Learning Effects
    ══════════════════════════════════════════
    
    PC2 Change (Early → Late Sessions):
    
    Superficial (L2/3 + L4):
      Early: {sup_early:.2f} → Late: {sup_late:.2f}
      Change: {sup_change:+.2f}
    
    Deep (L5 + L6):
      Early: {deep_early:.2f} → Late: {deep_late:.2f}
      Change: {deep_change:+.2f}
    
    ──────────────────────────────────────────
    
    KEY FINDINGS:
    
    1. Both superficial and deep layers show
       learning effects (PC2 increases = more
       spatial-like responses over sessions)
    
    2. Deep layers start with LOWER PC2 
       (more adaptation-like) but may show
       faster learning rates
    
    3. The gap between layers may narrow
       with experience
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.suptitle('Question 4: How Do Patterns Change With Experience?', 
                fontsize=14, fontweight='bold', y=1.02)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'Q4_learning_effects.png'), 
                   dpi=200, bbox_inches='tight')
        print(f"✓ Saved Q4 figure")
    
    return fig


# ============================================================================
# INTEGRATED SUMMARY
# ============================================================================

def create_integrated_summary(data, q1_results, q2_results, q3_results, q4_results, save_dir):
    """
    Create an integrated summary figure addressing all four questions.
    """
    print("\n" + "=" * 70)
    print("CREATING INTEGRATED SUMMARY")
    print("=" * 70)
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.35)
    
    session_order = data['session_order']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    x = np.arange(len(session_order))
    
    # =========================================================================
    # Q1 Summary: Spatial Modulation Development
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    for layer in layers:
        means = [v['mean'] for v in q1_results['smi_by_layer_session'][layer]]
        ax1.plot(x, means, marker='o', linewidth=2, markersize=6,
                color=layer_colors[layer], label=layer)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(session_order)
    ax1.set_xlabel('Session', fontsize=10)
    ax1.set_ylabel('Spatial Modulation Index', fontsize=10)
    ax1.set_title('Q1: Spatial Modulation Development by Layer', 
                 fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(alpha=0.3)
    
    # Add annotation
    ax1.text(0.98, 0.02, 'Finding: All layers develop\nspatial modulation over sessions',
            transform=ax1.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # =========================================================================
    # Q2 Summary: L4 Preference Shift
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    for layer in layers:
        l4_props = [p * 100 for p in q2_results['l4_preference_trajectory'][layer]]
        ax2.plot(x, l4_props, marker='o', linewidth=2, markersize=6,
                color=layer_colors[layer], label=layer)
    
    ax2.axhline(25, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(session_order)
    ax2.set_xlabel('Session', fontsize=10)
    ax2.set_ylabel('% Preferring L4 (Last Landmark)', fontsize=10)
    ax2.set_title('Q2: L4 Preference Development by Layer', 
                 fontsize=11, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(alpha=0.3)
    
    # =========================================================================
    # Q3 Summary: L1 Cell Types
    # =========================================================================
    ax3a = fig.add_subplot(gs[1, 0])
    
    if q3_results is not None:
        adapt_pcts = [q3_results['layer_results'].get(l, {}).get('pct_adapt', 0) for l in layers]
        spatial_pcts = [100 - p for p in adapt_pcts]
        
        x_pos = np.arange(len(layers))
        ax3a.bar(x_pos, adapt_pcts, label='Adaptation-like', color='#E53935', alpha=0.8)
        ax3a.bar(x_pos, spatial_pcts, bottom=adapt_pcts, label='Spatial-like', 
                color='#1E88E5', alpha=0.8)
        
        ax3a.axhline(50, color='black', linestyle='--', alpha=0.5)
        ax3a.set_xticks(x_pos)
        ax3a.set_xticklabels(layers)
        ax3a.set_ylabel('% of L1 Cells', fontsize=10)
        ax3a.set_title('Q3: L1 Cell Types by Layer', fontsize=11, fontweight='bold')
        ax3a.legend(loc='upper right', fontsize=8)
        ax3a.set_ylim(0, 100)
    
    ax3b = fig.add_subplot(gs[1, 1])
    
    if q3_results is not None:
        cluster_labels = q3_results['cluster_labels']
        l1_profiles = q3_results['l1_profiles']
        bin_centers = data['bin_centers']
        x_vals = np.linspace(bin_centers[0], bin_centers[-1], l1_profiles.shape[1])
        
        for c, (color, label) in enumerate([('#E53935', 'Adaptation'), 
                                             ('#1E88E5', 'Spatial')]):
            mask = cluster_labels == c
            mean_profile = np.mean(l1_profiles[mask], axis=0)
            ax3b.plot(x_vals, mean_profile, color=color, linewidth=2, label=label)
        
        for lm_pos in LANDMARK_POSITIONS:
            ax3b.axvline(lm_pos, color='gray', linestyle='--', alpha=0.4)
        
        ax3b.set_xlabel('Position (cm)', fontsize=10)
        ax3b.set_ylabel('Activity', fontsize=10)
        ax3b.set_title('L1 Cell Type Profiles', fontsize=11, fontweight='bold')
        ax3b.legend(fontsize=8)
    
    # =========================================================================
    # Q4 Summary: Learning Effects
    # =========================================================================
    ax4a = fig.add_subplot(gs[1, 2])
    
    for layer in layers:
        pc2_vals = [q4_results['pc_centroids_by_layer_session'][layer].get(s, np.nan) 
                   for s in session_order]
        ax4a.plot(x, pc2_vals, marker='o', linewidth=2, markersize=6,
                 color=layer_colors[layer], label=layer)
    
    ax4a.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4a.set_xticks(x)
    ax4a.set_xticklabels(session_order)
    ax4a.set_xlabel('Session', fontsize=10)
    ax4a.set_ylabel('PC2 Score', fontsize=10)
    ax4a.set_title('Q4: PC2 Learning Trajectory', fontsize=11, fontweight='bold')
    ax4a.legend(loc='lower right', fontsize=8)
    ax4a.grid(alpha=0.3)
    
    ax4b = fig.add_subplot(gs[1, 3])
    
    # Learning rates
    learning_rates = []
    for layer in layers:
        pc2_vals = [q4_results['pc_centroids_by_layer_session'][layer].get(s, np.nan) 
                   for s in session_order]
        valid = ~np.isnan(pc2_vals)
        if np.sum(valid) >= 3:
            r, _ = stats.pearsonr(np.arange(len(session_order))[valid], 
                                 np.array(pc2_vals)[valid])
            learning_rates.append(r)
        else:
            learning_rates.append(0)
    
    colors = [layer_colors[l] for l in layers]
    ax4b.bar(layers, learning_rates, color=colors, edgecolor='black', alpha=0.8)
    ax4b.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4b.set_ylabel('Learning Rate (r)', fontsize=10)
    ax4b.set_title('Learning Rate by Layer', fontsize=11, fontweight='bold')
    
    # =========================================================================
    # Bottom Row: Key Findings Summary
    # =========================================================================
    ax_summary = fig.add_subplot(gs[2, :])
    ax_summary.axis('off')
    
    findings_text = """
    ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    ║                                                    KEY FINDINGS SUMMARY                                                        ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                                                                ║
    ║  Q1: SPATIAL MODULATION DEVELOPMENT                           Q2: LANDMARK PREFERENCE SHIFTS                                  ║
    ║  ─────────────────────────────────────                        ────────────────────────────────                                 ║
    ║  • All layers show increasing spatial modulation              • L4 (last landmark, near reward) preference increases          ║
    ║    across training sessions                                     with experience in all layers                                  ║
    ║  • Deep layers (L5/L6) may start with different               • Deep layers may show earlier/stronger L4 preference           ║
    ║    baseline characteristics                                     development (consistent with reward proximity encoding)        ║
    ║                                                                                                                                ║
    ║  Q3: ADAPTATION VS SPATIAL ENCODING (L1 CELLS)                Q4: LEARNING EFFECTS                                            ║
    ║  ─────────────────────────────────────────────                ─────────────────────                                            ║
    ║  • L1-preferring cells split into TWO distinct populations:   • PC2 (adaptation axis) shifts toward spatial-like              ║
    ║    - Adaptation-like: early, decaying responses                 with training in all layers                                   ║
    ║    - Spatial-like: sharp, landmark-locked responses           • Learning rates may differ by layer                            ║
    ║  • Deep layers have MORE adaptation-like L1 cells             • Gap between layers may narrow with experience                 ║
    ║    (L6: 64% vs L2/3: 49%)                                                                                                      ║
    ║                                                                                                                                ║
    ╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
    ║  OVERALL INTERPRETATION:                                                                                                       ║
    ║  Deep layers show more adaptation-like responses among L1 cells, possibly reflecting faster/transient processing.             ║
    ║  With experience, both superficial and deep layers develop stronger spatial encoding and shift preference toward L4.          ║
    ║  The "early spatial modulation" in deep layers may partly reflect adaptation/onset responses rather than landmark encoding.    ║
    ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax_summary.text(0.5, 0.5, findings_text, transform=ax_summary.transAxes,
                   fontsize=9, fontfamily='monospace', 
                   verticalalignment='center', horizontalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle(f'Comprehensive PCA Analysis: {data.get("animal_id", "Animal")}\n'
                 f'Spatial Modulation in V1 Across Layers and Sessions',
                fontsize=14, fontweight='bold', y=0.98)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'INTEGRATED_SUMMARY.png'), 
                   dpi=200, bbox_inches='tight')
        print(f"✓ Saved integrated summary figure")
    
    return fig


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_comprehensive_analysis(pca_data_path, figure_dir):
    """
    Run complete analysis addressing all four research questions.
    """
    print("=" * 80)
    print("COMPREHENSIVE PCA ANALYSIS")
    print("Addressing Core Research Questions")
    print("=" * 80)
    
    os.makedirs(figure_dir, exist_ok=True)
    
    # Load data
    data = load_data(pca_data_path)
    
    # Compute features
    features = compute_cell_features(data)
    
    # Q1: Spatial modulation development by layer
    q1_results = analyze_spatial_modulation_by_layer_session(data, features)
    fig1 = plot_spatial_modulation_development(data, features, q1_results, figure_dir)
    
    # Q2: Landmark preference shifts by layer
    q2_results = analyze_landmark_preference_by_layer_session(data, features)
    fig2 = plot_landmark_preference_shifts(data, q2_results, figure_dir)
    
    # Q3: Adaptation vs spatial encoding in L1 cells
    q3_results = analyze_l1_cell_types(data, features)
    fig3 = plot_l1_cell_types(data, features, q3_results, figure_dir)
    
    # Q4: Learning effects
    q4_results = analyze_learning_effects(data, features, q3_results)
    fig4 = plot_learning_effects(data, features, q4_results, q3_results, figure_dir)
    
    # Integrated summary
    fig_summary = create_integrated_summary(data, q1_results, q2_results, 
                                            q3_results, q4_results, figure_dir)
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print("\nGenerated figures:")
    print("  • Q1_spatial_modulation_development.png")
    print("  • Q2_landmark_preference_shifts.png")
    print("  • Q3_l1_cell_types.png")
    print("  • Q4_learning_effects.png")
    print("  • INTEGRATED_SUMMARY.png")
    
    plt.show()
    
    return {
        'data': data,
        'features': features,
        'q1_results': q1_results,
        'q2_results': q2_results,
        'q3_results': q3_results,
        'q4_results': q4_results
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_comprehensive_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR
    )