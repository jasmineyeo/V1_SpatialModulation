"""
PCA_LayerStatistics.py
Statistical tests and summary figures for the layer effect on L1 cell types.

Run this after PCA_Interpretation.py has identified adaptation-like vs spatial-like cells.

JSY, 12/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\figures\layer_analysis"

# Early slope threshold for splitting (use median from interpretation script)
# Set to None to use median automatically
EARLY_SLOPE_THRESHOLD = None  


# ============================================================================
# DATA LOADING AND PROCESSING
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


def classify_l1_cells(data, threshold=None):
    """
    Classify L1-preferring cells into adaptation-like vs spatial-like.
    
    Returns dict with classification results.
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
    
    # Classify
    l1_indices = np.where(l1_mask)[0]
    adaptation_mask = l1_slopes < threshold
    spatial_mask = l1_slopes >= threshold
    
    results = {
        'l1_mask': l1_mask,
        'l1_indices': l1_indices,
        'early_slopes': l1_slopes,
        'threshold': threshold,
        'adaptation_mask': adaptation_mask,  # Within L1 cells
        'spatial_mask': spatial_mask,        # Within L1 cells
        'n_adaptation': np.sum(adaptation_mask),
        'n_spatial': np.sum(spatial_mask),
        'layer_labels': data['layer_labels'][l1_mask],
        'session_labels': data['session_labels'][l1_mask],
        'pc_scores': data['pc_scores'][l1_mask],
        'profiles': data['spatial_profiles_zscore'][l1_mask]
    }
    
    print(f"  Adaptation-like: {results['n_adaptation']}")
    print(f"  Spatial-like: {results['n_spatial']}")
    
    return results


# ============================================================================
# STATISTICAL TESTS
# ============================================================================

def run_layer_statistics(classification_results):
    """
    Run comprehensive statistical tests on layer distribution.
    """
    layer_labels = classification_results['layer_labels']
    adaptation_mask = classification_results['adaptation_mask']
    spatial_mask = classification_results['spatial_mask']
    
    print("\n" + "=" * 70)
    print("LAYER STATISTICS FOR L1 CELL TYPES")
    print("=" * 70)
    
    # Count by layer
    layers = ['L2/3', 'L4', 'L5', 'L6']
    counts = {'adaptation': {}, 'spatial': {}, 'total': {}}
    
    for layer in layers:
        layer_mask = layer_labels == layer
        counts['adaptation'][layer] = np.sum(adaptation_mask & layer_mask)
        counts['spatial'][layer] = np.sum(spatial_mask & layer_mask)
        counts['total'][layer] = np.sum(layer_mask)
    
    # Print table
    print("\n" + "-" * 50)
    print(f"{'Layer':<10} {'Adapt':<10} {'Spatial':<10} {'Total':<10} {'% Adapt':<10}")
    print("-" * 50)
    
    for layer in layers:
        adapt = counts['adaptation'][layer]
        spatial = counts['spatial'][layer]
        total = counts['total'][layer]
        pct = 100 * adapt / total if total > 0 else 0
        print(f"{layer:<10} {adapt:<10} {spatial:<10} {total:<10} {pct:<10.1f}")
    
    print("-" * 50)
    
    # Test 1: Chi-square test for all layers
    print("\n--- Test 1: Chi-square test (all layers) ---")
    
    contingency_table = np.array([
        [counts['adaptation'][layer] for layer in layers],
        [counts['spatial'][layer] for layer in layers]
    ])
    
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-square statistic: {chi2:.3f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p:.4e}")
    
    if p < 0.001:
        print("Result: Highly significant (p < 0.001) ***")
    elif p < 0.01:
        print("Result: Very significant (p < 0.01) **")
    elif p < 0.05:
        print("Result: Significant (p < 0.05) *")
    else:
        print("Result: Not significant")
    
    # Test 2: Superficial vs Deep comparison
    print("\n--- Test 2: Superficial (L2/3 + L4) vs Deep (L5 + L6) ---")
    
    superficial_adapt = counts['adaptation']['L2/3'] + counts['adaptation']['L4']
    superficial_spatial = counts['spatial']['L2/3'] + counts['spatial']['L4']
    deep_adapt = counts['adaptation']['L5'] + counts['adaptation']['L6']
    deep_spatial = counts['spatial']['L5'] + counts['spatial']['L6']
    
    contingency_2x2 = np.array([
        [superficial_adapt, superficial_spatial],
        [deep_adapt, deep_spatial]
    ])
    
    print(f"\n{'Group':<15} {'Adapt':<10} {'Spatial':<10} {'% Adapt':<10}")
    print("-" * 45)
    sup_pct = 100 * superficial_adapt / (superficial_adapt + superficial_spatial)
    deep_pct = 100 * deep_adapt / (deep_adapt + deep_spatial)
    print(f"{'Superficial':<15} {superficial_adapt:<10} {superficial_spatial:<10} {sup_pct:<10.1f}")
    print(f"{'Deep':<15} {deep_adapt:<10} {deep_spatial:<10} {deep_pct:<10.1f}")
    
    # Fisher's exact test (more appropriate for 2x2)
    odds_ratio, fisher_p = stats.fisher_exact(contingency_2x2)
    
    print(f"\nFisher's exact test:")
    print(f"  Odds ratio: {odds_ratio:.3f}")
    print(f"  P-value: {fisher_p:.4e}")
    
    # Chi-square for 2x2
    chi2_2x2, p_2x2, _, _ = stats.chi2_contingency(contingency_2x2)
    print(f"\nChi-square test:")
    print(f"  Chi-square: {chi2_2x2:.3f}")
    print(f"  P-value: {p_2x2:.4e}")
    
    if fisher_p < 0.05:
        direction = "MORE" if deep_pct > sup_pct else "FEWER"
        print(f"\nConclusion: Deep layers have significantly {direction} adaptation-like cells")
    
    # Test 3: Trend test (Cochran-Armitage)
    print("\n--- Test 3: Trend across layers (linear) ---")
    
    # Assign numeric values to layers (1=superficial, 4=deep)
    layer_scores = {'L2/3': 1, 'L4': 2, 'L5': 3, 'L6': 4}
    
    # Calculate proportion adaptation-like for each layer
    proportions = []
    ns = []
    scores = []
    
    for layer in layers:
        n = counts['total'][layer]
        if n > 0:
            prop = counts['adaptation'][layer] / n
            proportions.append(prop)
            ns.append(n)
            scores.append(layer_scores[layer])
    
    # Pearson correlation between layer depth and adaptation proportion
    r, p_trend = stats.pearsonr(scores, proportions)
    
    print(f"Correlation (layer depth vs % adaptation): r = {r:.3f}, p = {p_trend:.4e}")
    
    if p_trend < 0.05:
        direction = "increases" if r > 0 else "decreases"
        print(f"Conclusion: Proportion of adaptation-like cells {direction} with layer depth")
    
    # Store results
    stats_results = {
        'counts': counts,
        'chi2_all': {'chi2': chi2, 'p': p, 'dof': dof},
        'superficial_vs_deep': {
            'contingency': contingency_2x2,
            'odds_ratio': odds_ratio,
            'fisher_p': fisher_p,
            'chi2': chi2_2x2,
            'p': p_2x2,
            'superficial_pct': sup_pct,
            'deep_pct': deep_pct
        },
        'trend': {'r': r, 'p': p_trend}
    }
    
    return stats_results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_layer_summary(classification_results, stats_results, save_path=None):
    """
    Create comprehensive summary figure for layer effect.
    """
    layer_labels = classification_results['layer_labels']
    adaptation_mask = classification_results['adaptation_mask']
    spatial_mask = classification_results['spatial_mask']
    pc_scores = classification_results['pc_scores']
    profiles = classification_results['profiles']
    
    counts = stats_results['counts']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    
    fig = plt.figure(figsize=(16, 12))
    
    # Color scheme
    adapt_color = '#E53935'  # Red
    spatial_color = '#1E88E5'  # Blue
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # =========================================================================
    # Panel 1: Stacked bar chart - counts
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    
    x = np.arange(len(layers))
    width = 0.6
    
    adapt_counts = [counts['adaptation'][layer] for layer in layers]
    spatial_counts = [counts['spatial'][layer] for layer in layers]
    
    ax1.bar(x, adapt_counts, width, label='Adaptation-like', color=adapt_color, alpha=0.8)
    ax1.bar(x, spatial_counts, width, bottom=adapt_counts, label='Spatial-like', 
            color=spatial_color, alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel('Cortical Layer', fontsize=11)
    ax1.set_ylabel('Number of L1 Cells', fontsize=11)
    ax1.set_title('L1 Cell Counts by Layer', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    
    # Add count labels
    for i, layer in enumerate(layers):
        total = counts['total'][layer]
        ax1.text(i, total + 5, str(total), ha='center', va='bottom', fontsize=10)
    
    # =========================================================================
    # Panel 2: Proportion bar chart
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    
    proportions_adapt = [counts['adaptation'][layer] / counts['total'][layer] * 100 
                         if counts['total'][layer] > 0 else 0 for layer in layers]
    proportions_spatial = [100 - p for p in proportions_adapt]
    
    ax2.bar(x, proportions_adapt, width, label='Adaptation-like', color=adapt_color, alpha=0.8)
    ax2.bar(x, proportions_spatial, width, bottom=proportions_adapt, label='Spatial-like', 
            color=spatial_color, alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel('Cortical Layer', fontsize=11)
    ax2.set_ylabel('Proportion (%)', fontsize=11)
    ax2.set_title('L1 Cell Type Proportions by Layer', fontsize=12, fontweight='bold')
    ax2.axhline(50, color='black', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_ylim(0, 100)
    
    # Add percentage labels
    for i, pct in enumerate(proportions_adapt):
        ax2.text(i, pct/2, f'{pct:.0f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    # =========================================================================
    # Panel 3: Superficial vs Deep comparison
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    
    sup_pct = stats_results['superficial_vs_deep']['superficial_pct']
    deep_pct = stats_results['superficial_vs_deep']['deep_pct']
    fisher_p = stats_results['superficial_vs_deep']['fisher_p']
    
    groups = ['Superficial\n(L2/3 + L4)', 'Deep\n(L5 + L6)']
    adapt_pcts = [sup_pct, deep_pct]
    spatial_pcts = [100 - sup_pct, 100 - deep_pct]
    
    x_groups = np.arange(2)
    ax3.bar(x_groups, adapt_pcts, width, label='Adaptation-like', color=adapt_color, alpha=0.8)
    ax3.bar(x_groups, spatial_pcts, width, bottom=adapt_pcts, label='Spatial-like', 
            color=spatial_color, alpha=0.8)
    
    ax3.set_xticks(x_groups)
    ax3.set_xticklabels(groups)
    ax3.set_ylabel('Proportion (%)', fontsize=11)
    ax3.set_title('Superficial vs Deep Layers', fontsize=12, fontweight='bold')
    ax3.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylim(0, 100)
    
    # Add significance annotation
    sig_str = f"Fisher's p = {fisher_p:.3e}"
    if fisher_p < 0.001:
        sig_str += " ***"
    elif fisher_p < 0.01:
        sig_str += " **"
    elif fisher_p < 0.05:
        sig_str += " *"
    
    ax3.text(0.5, 95, sig_str, ha='center', va='top', fontsize=10, 
            transform=ax3.get_xaxis_transform())
    
    # Add percentage labels
    for i, pct in enumerate(adapt_pcts):
        ax3.text(i, pct/2, f'{pct:.1f}%', ha='center', va='center', 
                fontsize=11, fontweight='bold', color='white')
    
    # =========================================================================
    # Panel 4: PC2 distribution by layer (PC2 = adaptation axis)
    # =========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    
    pc2_by_layer = []
    layer_labels_for_plot = []
    
    for layer in layers:
        layer_mask = layer_labels == layer
        if np.sum(layer_mask) > 0:
            pc2_by_layer.append(pc_scores[layer_mask, 1])
            layer_labels_for_plot.append(layer)
    
    bp = ax4.boxplot(pc2_by_layer, labels=layer_labels_for_plot, patch_artist=True)
    
    for patch, layer in zip(bp['boxes'], layer_labels_for_plot):
        patch.set_facecolor(layer_colors[layer])
        patch.set_alpha(0.6)
    
    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Cortical Layer', fontsize=11)
    ax4.set_ylabel('PC2 Score (Adaptation Axis)', fontsize=11)
    ax4.set_title('PC2 Distribution by Layer\n(Lower = More Adaptation-like)', 
                 fontsize=12, fontweight='bold')
    
    # =========================================================================
    # Panel 5: Mean profiles by layer for adaptation-like cells
    # =========================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    
    bin_centers = classification_results.get('bin_centers', np.arange(profiles.shape[1]))
    
    for layer in layers:
        layer_mask = (layer_labels == layer) & adaptation_mask
        if np.sum(layer_mask) > 5:
            mean_profile = np.mean(profiles[layer_mask], axis=0)
            sem = np.std(profiles[layer_mask], axis=0) / np.sqrt(np.sum(layer_mask))
            
            # Assume bin_centers might not be in classification_results
            x_vals = np.linspace(10, 125, len(mean_profile))
            ax5.plot(x_vals, mean_profile, color=layer_colors[layer], 
                    linewidth=2, label=f'{layer} (n={np.sum(layer_mask)})')
            ax5.fill_between(x_vals, mean_profile - sem, mean_profile + sem, 
                           color=layer_colors[layer], alpha=0.2)
    
    # Add landmark markers
    for lm_pos in [25, 55, 85, 115]:
        ax5.axvline(lm_pos, color='gray', linestyle='--', alpha=0.4)
    
    ax5.set_xlabel('Position (cm)', fontsize=11)
    ax5.set_ylabel('Z-scored Activity', fontsize=11)
    ax5.set_title('Adaptation-like L1 Cells by Layer', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 6: Mean profiles by layer for spatial-like cells
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    
    for layer in layers:
        layer_mask = (layer_labels == layer) & spatial_mask
        if np.sum(layer_mask) > 5:
            mean_profile = np.mean(profiles[layer_mask], axis=0)
            sem = np.std(profiles[layer_mask], axis=0) / np.sqrt(np.sum(layer_mask))
            
            x_vals = np.linspace(10, 125, len(mean_profile))
            ax6.plot(x_vals, mean_profile, color=layer_colors[layer], 
                    linewidth=2, label=f'{layer} (n={np.sum(layer_mask)})')
            ax6.fill_between(x_vals, mean_profile - sem, mean_profile + sem, 
                           color=layer_colors[layer], alpha=0.2)
    
    for lm_pos in [25, 55, 85, 115]:
        ax6.axvline(lm_pos, color='gray', linestyle='--', alpha=0.4)
    
    ax6.set_xlabel('Position (cm)', fontsize=11)
    ax6.set_ylabel('Z-scored Activity', fontsize=11)
    ax6.set_title('Spatial-like L1 Cells by Layer', fontsize=12, fontweight='bold')
    ax6.legend(loc='upper right', fontsize=9)
    ax6.grid(alpha=0.3)
    
    # =========================================================================
    # Overall title
    # =========================================================================
    plt.suptitle('Layer Distribution of L1 Cell Types\n(Adaptation-like vs Spatial-like)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved layer summary figure: {save_path}")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def run_layer_analysis(pca_data_path, figure_dir):
    """Main function to run layer statistics and generate figures."""
    
    print("=" * 70)
    print("LAYER ANALYSIS FOR L1 CELL TYPES")
    print("=" * 70)
    
    os.makedirs(figure_dir, exist_ok=True)
    
    # Load data
    data = load_data(pca_data_path)
    
    # Add bin_centers to be accessible later
    bin_centers = data['bin_centers']
    
    # Classify L1 cells
    classification_results = classify_l1_cells(data, threshold=EARLY_SLOPE_THRESHOLD)
    classification_results['bin_centers'] = bin_centers
    
    # Run statistics
    stats_results = run_layer_statistics(classification_results)
    
    # Generate figure
    fig = plot_layer_summary(
        classification_results, 
        stats_results,
        save_path=os.path.join(figure_dir, 'layer_effect_summary.png')
    )
    
    print("\n" + "=" * 70)
    print("LAYER ANALYSIS COMPLETE!")
    print("=" * 70)
    
    plt.show()
    
    return classification_results, stats_results


if __name__ == "__main__":
    classification_results, stats_results = run_layer_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR
    )