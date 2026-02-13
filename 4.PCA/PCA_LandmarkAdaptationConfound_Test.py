"""
PCA_LandmarkAdaptationConfound_Test.py

Test whether "adaptation-like" classification is confounded by trial-onset effects.
If LD1-preferring cells show lower slopes simply because LD1 is first, this would
invalidate the adaptation vs spatial distinction.

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


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\figures\confound_test"


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


# ============================================================================
# CONFOUND TEST
# ============================================================================

def test_landmark_adaptation_confound(data):
    """
    Test if "adaptation-like" classification is confounded by 
    trial-onset effects (LD1 being first landmark).
    
    Key question: Do LD1-preferring cells have lower slopes simply 
    because LD1 is first in the trial?
    """
    
    print("\n" + "="*70)
    print("TESTING TRIAL-ONSET ADAPTATION CONFOUND")
    print("="*70)
    
    profiles = data['spatial_profiles_zscore']
    preferred_landmark = data['preferred_landmark']
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    layer_labels = data['layer_labels']
    
    # Compute early slopes for ALL cells (not just LD1)
    print("\nComputing early slopes for all cells...")
    early_slopes = compute_early_slope(profiles, bin_centers)
    
    # Group by landmark preference
    landmark_slopes = {}
    
    for lm_idx in range(4):
        lm_mask = preferred_landmark == lm_idx
        lm_slopes = early_slopes[lm_mask]
        
        if len(lm_slopes) > 10:
            landmark_slopes[f'LD{lm_idx+1}'] = {
                'mean': np.mean(lm_slopes),
                'median': np.median(lm_slopes),
                'std': np.std(lm_slopes),
                'n': len(lm_slopes),
                'slopes': lm_slopes,
                'mask': lm_mask
            }
    
    # Print comparison
    print("\n" + "="*70)
    print("EARLY SLOPE BY LANDMARK PREFERENCE")
    print("="*70)
    print(f"\n{'Landmark':<12} {'N':<8} {'Mean':<12} {'Median':<12} {'Std':<12}")
    print("-"*60)
    
    for lm in ['LD1', 'LD2', 'LD3', 'LD4']:
        if lm in landmark_slopes:
            stats_data = landmark_slopes[lm]
            print(f"{lm:<12} {stats_data['n']:<8} {stats_data['mean']:<12.4f} "
                  f"{stats_data['median']:<12.4f} {stats_data['std']:<12.4f}")
    
    # Statistical tests
    print("\n" + "="*70)
    print("STATISTICAL TESTS")
    print("="*70)
    
    # Test 1: LD1 vs LD4 (first vs last landmark)
    if 'LD1' in landmark_slopes and 'LD4' in landmark_slopes:
        ld1_slopes = landmark_slopes['LD1']['slopes']
        ld4_slopes = landmark_slopes['LD4']['slopes']
        
        t, p = stats.ttest_ind(ld1_slopes, ld4_slopes)
        
        print(f"\n--- Test 1: LD1 vs LD4 (First vs Last Landmark) ---")
        print(f"  LD1 mean: {np.mean(ld1_slopes):.4f}")
        print(f"  LD4 mean: {np.mean(ld4_slopes):.4f}")
        print(f"  Difference: {np.mean(ld1_slopes) - np.mean(ld4_slopes):+.4f}")
        print(f"  t-test: t={t:.3f}, p={p:.4f}")
        
        if p < 0.05:
            if np.mean(ld1_slopes) < np.mean(ld4_slopes):
                print("  ⚠ WARNING: LD1 cells have significantly LOWER slopes than LD4")
                print("     → This suggests TRIAL-ONSET ADAPTATION confound")
                print("     → Your 'adaptation-like' classification may be artifact")
            else:
                print("  ✓ LD1 cells have HIGHER slopes than LD4")
                print("    → Not consistent with trial-onset adaptation")
        else:
            print("  ✓ No significant difference between LD1 and LD4")
            print("    → Classification likely reflects true cell types")
    
    # Test 2: ANOVA across all landmarks
    print(f"\n--- Test 2: ANOVA Across All Landmarks ---")
    
    all_slopes = [landmark_slopes[lm]['slopes'] 
                  for lm in ['LD1', 'LD2', 'LD3', 'LD4'] 
                  if lm in landmark_slopes]
    
    if len(all_slopes) >= 3:
        f_stat, p_anova = stats.f_oneway(*all_slopes)
        print(f"  F-statistic: {f_stat:.3f}")
        print(f"  p-value: {p_anova:.4f}")
        
        if p_anova < 0.05:
            print("  ⚠ Significant differences across landmark preferences")
            print("    → Suggests trial position affects slope measurement")
        else:
            print("  ✓ No significant differences across landmarks")
            print("    → Slope reflects cell properties, not trial position")
    
    # Test 3: Check if layer effect holds across all landmarks
    print(f"\n--- Test 3: Layer Effect Across Landmark Preferences ---")
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    
    for lm in ['LD1', 'LD2', 'LD3', 'LD4']:
        if lm not in landmark_slopes:
            continue
        
        lm_mask = landmark_slopes[lm]['mask']
        lm_slopes_all = early_slopes[lm_mask]
        lm_layers = layer_labels[lm_mask]
        
        # Get median for this landmark preference
        threshold = np.median(lm_slopes_all)
        
        # Count adaptation-like by layer
        superficial_adapt = 0
        superficial_total = 0
        deep_adapt = 0
        deep_total = 0
        
        for layer in layers:
            layer_mask_within = lm_layers == layer
            n_layer = np.sum(layer_mask_within)
            
            if n_layer > 0:
                adapt_count = np.sum(lm_slopes_all[layer_mask_within] < threshold)
                
                if layer in ['L2/3', 'L4']:
                    superficial_adapt += adapt_count
                    superficial_total += n_layer
                else:
                    deep_adapt += adapt_count
                    deep_total += n_layer
        
        if superficial_total > 5 and deep_total > 5:
            sup_pct = 100 * superficial_adapt / superficial_total
            deep_pct = 100 * deep_adapt / deep_total
            
            print(f"\n  {lm}-preferring cells:")
            print(f"    Superficial: {sup_pct:.1f}% adaptation-like")
            print(f"    Deep: {deep_pct:.1f}% adaptation-like")
            print(f"    Difference: {deep_pct - sup_pct:+.1f} pp")
    
    return landmark_slopes


def plot_confound_analysis(data, landmark_slopes, save_dir):
    """
    Comprehensive visualization of confound test results.
    """
    
    print("\nGenerating figures...")
    
    profiles = data['spatial_profiles_zscore']
    preferred_landmark = data['preferred_landmark']
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    layer_labels = data['layer_labels']
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    landmark_names = ['LD1', 'LD2', 'LD3', 'LD4']
    
    # =========================================================================
    # ROW 1: Early slope distributions by landmark preference
    # =========================================================================
    
    # Panel 1: Violin plots
    ax1 = fig.add_subplot(gs[0, 0:2])
    
    slopes_to_plot = [landmark_slopes[lm]['slopes'] 
                     for lm in landmark_names 
                     if lm in landmark_slopes]
    labels = [lm for lm in landmark_names if lm in landmark_slopes]
    
    parts = ax1.violinplot(slopes_to_plot, positions=range(len(labels)), 
                          widths=0.7, showmeans=True, showextrema=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_xlabel('Preferred Landmark', fontsize=12)
    ax1.set_ylabel('Early Slope', fontsize=12)
    ax1.set_title('Early Slope Distribution by Landmark Preference\n'
                 '(Testing Trial-Onset Confound)', 
                 fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Add n counts
    for i, lm in enumerate(labels):
        n = landmark_slopes[lm]['n']
        ax1.text(i, ax1.get_ylim()[1]*0.95, f'n={n}', 
                ha='center', fontsize=9)
    
    # Panel 2: Mean ± SEM bar plot
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    x_pos = np.arange(len(labels))
    means = [landmark_slopes[lm]['mean'] for lm in labels]
    stds = [landmark_slopes[lm]['std'] for lm in labels]
    ns = [landmark_slopes[lm]['n'] for lm in labels]
    sems = [stds[i]/np.sqrt(ns[i]) for i in range(len(labels))]
    
    bars = ax2.bar(x_pos, means, yerr=sems, capsize=5, 
                   color=colors[:len(labels)], alpha=0.8, edgecolor='black')
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=11)
    ax2.set_xlabel('Preferred Landmark', fontsize=12)
    ax2.set_ylabel('Mean Early Slope', fontsize=12)
    ax2.set_title('Mean Early Slope by Landmark Preference', 
                 fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y')
    
    # Add statistical annotation
    if 'LD1' in landmark_slopes and 'LD4' in landmark_slopes:
        ld1_slopes = landmark_slopes['LD1']['slopes']
        ld4_slopes = landmark_slopes['LD4']['slopes']
        t, p = stats.ttest_ind(ld1_slopes, ld4_slopes)
        
        ld1_idx = labels.index('LD1')
        ld4_idx = labels.index('LD4')
        
        y_max = max(means[ld1_idx] + sems[ld1_idx], means[ld4_idx] + sems[ld4_idx])
        y_line = y_max * 1.2
        
        ax2.plot([ld1_idx, ld4_idx], [y_line, y_line], 'k-', linewidth=2)
        
        if p < 0.001:
            sig_text = '***'
        elif p < 0.01:
            sig_text = '**'
        elif p < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'
        
        ax2.text((ld1_idx + ld4_idx)/2, y_line*1.05, 
                f'{sig_text}\np={p:.4f}', 
                ha='center', fontsize=10, fontweight='bold')
    
    # =========================================================================
    # ROW 2: Mean response profiles by landmark preference
    # =========================================================================
    
    for lm_idx in range(4):
        ax = fig.add_subplot(gs[1, lm_idx])
        
        lm_name = f'LD{lm_idx+1}'
        
        if lm_name not in landmark_slopes:
            ax.axis('off')
            continue
        
        lm_mask = preferred_landmark == lm_idx
        
        # Get slopes for this landmark
        early_slopes = compute_early_slope(profiles, bin_centers)
        lm_slopes = early_slopes[lm_mask]
        threshold = np.median(lm_slopes)
        
        # Split into adaptation-like vs spatial-like
        adapt_mask = lm_slopes < threshold
        spatial_mask = lm_slopes >= threshold
        
        # Get profiles
        lm_profiles = profiles[lm_mask]
        adapt_profiles = lm_profiles[adapt_mask]
        spatial_profiles = lm_profiles[spatial_mask]
        
        x_vals = np.linspace(bin_centers[0], bin_centers[-1], adapt_profiles.shape[1])
        
        # Plot mean profiles
        if len(adapt_profiles) > 0:
            mean_adapt = np.mean(adapt_profiles, axis=0)
            sem_adapt = np.std(adapt_profiles, axis=0) / np.sqrt(len(adapt_profiles))
            ax.plot(x_vals, mean_adapt, color='#E53935', linewidth=2.5, 
                   label=f'Adaptation-like (n={len(adapt_profiles)})')
            ax.fill_between(x_vals, mean_adapt - sem_adapt, mean_adapt + sem_adapt,
                           color='#E53935', alpha=0.2)
        
        if len(spatial_profiles) > 0:
            mean_spatial = np.mean(spatial_profiles, axis=0)
            sem_spatial = np.std(spatial_profiles, axis=0) / np.sqrt(len(spatial_profiles))
            ax.plot(x_vals, mean_spatial, color='#1E88E5', linewidth=2.5,
                   label=f'Spatial-like (n={len(spatial_profiles)})')
            ax.fill_between(x_vals, mean_spatial - sem_spatial, mean_spatial + sem_spatial,
                           color='#1E88E5', alpha=0.2)
        
        # Mark landmarks
        for i, lm_pos in enumerate(landmark_positions):
            if i == lm_idx:
                ax.axvline(lm_pos, color=colors[lm_idx], linestyle='-', 
                          alpha=0.8, linewidth=2)
            else:
                ax.axvline(lm_pos, color='gray', linestyle='--', alpha=0.4)
        
        ax.set_xlabel('Position (cm)', fontsize=10)
        ax.set_ylabel('Z-scored Activity', fontsize=10)
        ax.set_title(f'{lm_name}-Preferring Cells', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)
    
    # =========================================================================
    # ROW 3: Layer effect across landmarks
    # =========================================================================
    
    ax_layer = fig.add_subplot(gs[2, 0:2])
    
    # For each landmark, calculate % adaptation-like in superficial vs deep
    landmark_layer_data = []
    
    for lm_idx in range(4):
        lm_name = f'LD{lm_idx+1}'
        if lm_name not in landmark_slopes:
            continue
        
        lm_mask = preferred_landmark == lm_idx
        early_slopes = compute_early_slope(profiles, bin_centers)
        lm_slopes = early_slopes[lm_mask]
        lm_layers = layer_labels[lm_mask]
        
        threshold = np.median(lm_slopes)
        
        # Count by layer group
        superficial_mask = np.isin(lm_layers, ['L2/3', 'L4'])
        deep_mask = np.isin(lm_layers, ['L5', 'L6'])
        
        sup_adapt_pct = 100 * np.mean(lm_slopes[superficial_mask] < threshold)
        deep_adapt_pct = 100 * np.mean(lm_slopes[deep_mask] < threshold)
        
        landmark_layer_data.append({
            'landmark': lm_name,
            'superficial': sup_adapt_pct,
            'deep': deep_adapt_pct,
            'difference': deep_adapt_pct - sup_adapt_pct
        })
    
    # Plot
    x_pos = np.arange(len(landmark_layer_data))
    width = 0.35
    
    sup_pcts = [d['superficial'] for d in landmark_layer_data]
    deep_pcts = [d['deep'] for d in landmark_layer_data]
    lm_labels = [d['landmark'] for d in landmark_layer_data]
    
    ax_layer.bar(x_pos - width/2, sup_pcts, width, 
                label='Superficial (L2/3+L4)', color='#1E88E5', alpha=0.8)
    ax_layer.bar(x_pos + width/2, deep_pcts, width,
                label='Deep (L5+L6)', color='#4CAF50', alpha=0.8)
    
    ax_layer.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax_layer.set_xticks(x_pos)
    ax_layer.set_xticklabels(lm_labels)
    ax_layer.set_xlabel('Landmark Preference', fontsize=12)
    ax_layer.set_ylabel('% Adaptation-like', fontsize=12)
    ax_layer.set_title('Layer Effect Across Landmark Preferences\n'
                      '(Does deep > superficial hold for all landmarks?)',
                      fontsize=13, fontweight='bold')
    ax_layer.legend(loc='upper left')
    ax_layer.set_ylim(0, 100)
    ax_layer.grid(alpha=0.3, axis='y')
    
    # Panel: Interpretation box
    ax_interp = fig.add_subplot(gs[2, 2:4])
    ax_interp.axis('off')
    
    # Determine interpretation
    if 'LD1' in landmark_slopes and 'LD4' in landmark_slopes:
        ld1_mean = landmark_slopes['LD1']['mean']
        ld4_mean = landmark_slopes['LD4']['mean']
        
        ld1_slopes = landmark_slopes['LD1']['slopes']
        ld4_slopes = landmark_slopes['LD4']['slopes']
        t, p = stats.ttest_ind(ld1_slopes, ld4_slopes)
        
        interp_text = "INTERPRETATION:\n\n"
        
        interp_text += f"LD1 vs LD4 Comparison:\n"
        interp_text += f"  LD1 mean slope: {ld1_mean:.4f}\n"
        interp_text += f"  LD4 mean slope: {ld4_mean:.4f}\n"
        interp_text += f"  Difference: {ld1_mean - ld4_mean:+.4f}\n"
        interp_text += f"  p-value: {p:.4f}\n\n"
        
        if p < 0.05 and ld1_mean < ld4_mean:
            interp_text += "⚠ CONFOUND DETECTED:\n"
            interp_text += "  LD1 cells have lower slopes\n"
            interp_text += "  than LD4 cells.\n\n"
            interp_text += "  This suggests trial-onset\n"
            interp_text += "  adaptation is contaminating\n"
            interp_text += "  your classification.\n\n"
            interp_text += "RECOMMENDATION:\n"
            interp_text += "  Re-analyze separately for\n"
            interp_text += "  each landmark preference.\n"
        elif p >= 0.05:
            interp_text += "✓ NO CONFOUND:\n"
            interp_text += "  No significant difference\n"
            interp_text += "  between LD1 and LD4.\n\n"
            interp_text += "  Classification appears to\n"
            interp_text += "  reflect true cell types,\n"
            interp_text += "  not trial position.\n\n"
            
            # Check if layer effect is consistent
            consistent_layer_effect = True
            for d in landmark_layer_data:
                if d['difference'] < 5:  # Less than 5pp difference
                    consistent_layer_effect = False
            
            if consistent_layer_effect:
                interp_text += "✓ Layer effect consistent\n"
                interp_text += "  across all landmarks.\n"
            else:
                interp_text += "⚠ Layer effect varies by\n"
                interp_text += "  landmark preference.\n"
        else:
            interp_text += "? UNCLEAR:\n"
            interp_text += "  LD1 has HIGHER slopes\n"
            interp_text += "  than LD4 (unexpected).\n\n"
            interp_text += "  Investigate further.\n"
    else:
        interp_text = "Insufficient data for\ninterpretation."
    
    ax_interp.text(0.1, 0.9, interp_text,
                  transform=ax_interp.transAxes,
                  fontsize=10, fontfamily='monospace',
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # =========================================================================
    # Overall title
    # =========================================================================
    plt.suptitle('Trial-Onset Adaptation Confound Test\n'
                 'Does LD1 have lower slopes simply because it\'s first?',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'landmark_adaptation_confound_test.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved figure: {save_path}")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def run_confound_test(pca_data_path, figure_dir):
    """Main analysis workflow."""
    
    print("=" * 70)
    print("TRIAL-ONSET ADAPTATION CONFOUND TEST")
    print("=" * 70)
    
    # Load data
    data = load_data(pca_data_path)
    
    # Test for confound
    landmark_slopes = test_landmark_adaptation_confound(data)
    
    # Visualize
    fig = plot_confound_analysis(data, landmark_slopes, save_dir=figure_dir)
    
    print("\n" + "=" * 70)
    print("CONFOUND TEST COMPLETE!")
    print("=" * 70)
    
    plt.show()
    
    return landmark_slopes


if __name__ == "__main__":
    landmark_slopes = run_confound_test(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR
    )