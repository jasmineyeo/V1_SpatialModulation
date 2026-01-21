"""
PCA_AllCells_LayerAnalysis.py

Test whether adaptation-like phenotype is a GENERAL property of deep layers
or specific to L1-preferring cells.

Extends PCA_LayerStatistics.py to analyze ALL cells, stratified by 
landmark preference.

JSY, 01/2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\figures\all_cells_analysis"

EARLY_SLOPE_THRESHOLD = None  # Use median


# ============================================================================
# LOAD DATA
# ============================================================================

def load_data(filepath):
    """Load all necessary data."""
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
    
    print(f"  Total cells: {len(data['session_labels'])}")
    
    return data


# ============================================================================
# COMPUTE EARLY SLOPE FOR ALL CELLS
# ============================================================================

def compute_early_slope(profiles, bin_centers, onset_end_cm=15, slope_end_cm=30):
    """Compute early decay slope for all cells."""
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


def classify_cells_by_slope(early_slopes, threshold=None):
    """
    Classify cells as adaptation-like vs spatial-like based on early slope.
    """
    if threshold is None:
        threshold = np.median(early_slopes)
    
    adaptation_mask = early_slopes < threshold
    
    return adaptation_mask, threshold


# ============================================================================
# STRATIFIED ANALYSIS BY LANDMARK PREFERENCE
# ============================================================================

def analyze_by_landmark_and_layer(data, adaptation_mask, early_slopes):
    """
    For each landmark preference group, test layer effect on adaptation proportion.
    """
    layer_labels = data['layer_labels']
    preferred_landmark = data['preferred_landmark']
    
    landmark_positions = data['landmark_positions']
    n_landmarks = len(landmark_positions)
    
    print("\n" + "=" * 70)
    print("LAYER EFFECT ANALYSIS FOR ALL LANDMARK PREFERENCES")
    print("=" * 70)
    
    results = {}
    
    # For each landmark (including "Between")
    landmark_groups = list(range(n_landmarks)) + [-1]  # 0,1,2,3,-1
    landmark_names = [f'L{i+1}' for i in range(n_landmarks)] + ['Between']
    
    for lm_idx, lm_name in zip(landmark_groups, landmark_names):
        
        # Get cells preferring this landmark
        if lm_idx == -1:
            lm_mask = preferred_landmark == -1
        else:
            lm_mask = preferred_landmark == lm_idx
        
        n_cells = np.sum(lm_mask)
        
        if n_cells < 50:  # Skip if too few cells
            print(f"\n{lm_name}: Only {n_cells} cells, skipping")
            continue
        
        print(f"\n{'='*70}")
        print(f"{lm_name}-PREFERRING CELLS (n={n_cells})")
        print(f"{'='*70}")
        
        # Get subsets
        lm_layers = layer_labels[lm_mask]
        lm_adaptation = adaptation_mask[lm_mask]
        lm_slopes = early_slopes[lm_mask]
        
        # Count by layer
        counts = {'adaptation': {}, 'spatial': {}, 'total': {}}
        
        for layer in ['L2/3', 'L4', 'L5', 'L6']:
            layer_mask = lm_layers == layer
            counts['adaptation'][layer] = np.sum(lm_adaptation & layer_mask)
            counts['spatial'][layer] = np.sum((~lm_adaptation) & layer_mask)
            counts['total'][layer] = np.sum(layer_mask)
        
        # Print table
        print(f"\n{'-'*50}")
        print(f"{'Layer':<10} {'Adapt':<10} {'Spatial':<10} {'Total':<10} {'% Adapt':<10}")
        print(f"{'-'*50}")
        
        for layer in ['L2/3', 'L4', 'L5', 'L6']:
            adapt = counts['adaptation'][layer]
            spatial = counts['spatial'][layer]
            total = counts['total'][layer]
            pct = 100 * adapt / total if total > 0 else 0
            print(f"{layer:<10} {adapt:<10} {spatial:<10} {total:<10} {pct:<10.1f}")
        
        print(f"{'-'*50}")
        
        # Superficial vs Deep comparison
        superficial_adapt = counts['adaptation']['L2/3'] + counts['adaptation']['L4']
        superficial_spatial = counts['spatial']['L2/3'] + counts['spatial']['L4']
        deep_adapt = counts['adaptation']['L5'] + counts['adaptation']['L6']
        deep_spatial = counts['spatial']['L5'] + counts['spatial']['L6']
        
        contingency_2x2 = np.array([
            [superficial_adapt, superficial_spatial],
            [deep_adapt, deep_spatial]
        ])
        
        sup_pct = 100 * superficial_adapt / (superficial_adapt + superficial_spatial)
        deep_pct = 100 * deep_adapt / (deep_adapt + deep_spatial)
        
        print(f"\nSuperficial vs Deep:")
        print(f"  Superficial: {sup_pct:.1f}% adaptation-like")
        print(f"  Deep: {deep_pct:.1f}% adaptation-like")
        print(f"  Difference: {deep_pct - sup_pct:+.1f} percentage points")
        
        # Fisher's exact test
        odds_ratio, fisher_p = stats.fisher_exact(contingency_2x2)
        
        print(f"\nFisher's exact test:")
        print(f"  Odds ratio: {odds_ratio:.3f}")
        print(f"  P-value: {fisher_p:.4e}")
        
        if fisher_p < 0.001:
            sig = "***"
        elif fisher_p < 0.01:
            sig = "**"
        elif fisher_p < 0.05:
            sig = "*"
        else:
            sig = "ns"
        
        print(f"  Significance: {sig}")
        
        # Store results
        results[lm_name] = {
            'n_cells': n_cells,
            'counts': counts,
            'superficial_pct': sup_pct,
            'deep_pct': deep_pct,
            'difference': deep_pct - sup_pct,
            'fisher_p': fisher_p,
            'odds_ratio': odds_ratio,
            'significance': sig
        }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comprehensive_comparison(results, save_path=None):
    """
    Create comprehensive figure comparing layer effects across landmark preferences.
    """
    # Filter to only groups with enough cells
    landmark_names = [k for k in results.keys() if results[k]['n_cells'] >= 50]
    
    if len(landmark_names) == 0:
        print("No landmark groups with sufficient cells!")
        return None
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Color scheme
    adapt_color = '#E53935'
    spatial_color = '#1E88E5'
    
    # =========================================================================
    # Panel 1: Superficial vs Deep by Landmark
    # =========================================================================
    ax1 = axes[0, 0]
    
    x = np.arange(len(landmark_names))
    width = 0.35
    
    sup_pcts = [results[lm]['superficial_pct'] for lm in landmark_names]
    deep_pcts = [results[lm]['deep_pct'] for lm in landmark_names]
    
    ax1.bar(x - width/2, sup_pcts, width, label='Superficial (L2/3+L4)', 
            color='#1E88E5', alpha=0.8)
    ax1.bar(x + width/2, deep_pcts, width, label='Deep (L5+L6)', 
            color='#4CAF50', alpha=0.8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(landmark_names)
    ax1.set_ylabel('% Adaptation-like', fontsize=11)
    ax1.set_xlabel('Landmark Preference', fontsize=11)
    ax1.set_title('Adaptation-like % by Layer Group', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add significance stars
    for i, lm in enumerate(landmark_names):
        if results[lm]['significance'] != 'ns':
            y_pos = max(sup_pcts[i], deep_pcts[i]) + 5
            ax1.text(i, y_pos, results[lm]['significance'], 
                    ha='center', fontsize=14, fontweight='bold')
    
    # =========================================================================
    # Panel 2: Effect size (difference) by landmark
    # =========================================================================
    ax2 = axes[0, 1]
    
    differences = [results[lm]['difference'] for lm in landmark_names]
    colors_by_sig = []
    
    for lm in landmark_names:
        if results[lm]['fisher_p'] < 0.05:
            colors_by_sig.append('#E53935')  # Red for significant
        else:
            colors_by_sig.append('#BDBDBD')  # Gray for ns
    
    bars = ax2.bar(x, differences, color=colors_by_sig, alpha=0.8, edgecolor='black')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(landmark_names)
    ax2.set_ylabel('Deep - Superficial (percentage points)', fontsize=11)
    ax2.set_xlabel('Landmark Preference', fontsize=11)
    ax2.set_title('Layer Effect Size', fontsize=12, fontweight='bold')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add p-values on bars
    for i, lm in enumerate(landmark_names):
        p_val = results[lm]['fisher_p']
        if p_val < 0.001:
            p_text = 'p<0.001'
        else:
            p_text = f'p={p_val:.3f}'
        
        y_pos = differences[i] + (2 if differences[i] > 0 else -4)
        ax2.text(i, y_pos, p_text, ha='center', fontsize=8)
    
    # =========================================================================
    # Panel 3: P-values by landmark
    # =========================================================================
    ax3 = axes[0, 2]
    
    p_values = [results[lm]['fisher_p'] for lm in landmark_names]
    
    ax3.bar(x, [-np.log10(p) for p in p_values], color=colors_by_sig, 
            alpha=0.8, edgecolor='black')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(landmark_names)
    ax3.set_ylabel('-log10(p-value)', fontsize=11)
    ax3.set_xlabel('Landmark Preference', fontsize=11)
    ax3.set_title('Statistical Significance', fontsize=12, fontweight='bold')
    ax3.axhline(-np.log10(0.05), color='red', linestyle='--', linewidth=2, 
               label='p=0.05 threshold')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # =========================================================================
    # Panel 4: All layers by landmark (stacked bars)
    # =========================================================================
    ax4 = axes[1, 0]
    
    # For each landmark, show all 4 layers
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 
                   'L5': '#4CAF50', 'L6': '#E53935'}
    
    # Pick one landmark to show detail (e.g., L1)
    if 'L1' in landmark_names:
        detail_lm = 'L1'
    else:
        detail_lm = landmark_names[0]
    
    counts = results[detail_lm]['counts']
    
    x_layers = np.arange(len(layer_order))
    width = 0.6
    
    adapt_counts = [counts['adaptation'][layer] for layer in layer_order]
    spatial_counts = [counts['spatial'][layer] for layer in layer_order]
    
    ax4.bar(x_layers, adapt_counts, width, label='Adaptation-like', 
           color=adapt_color, alpha=0.8)
    ax4.bar(x_layers, spatial_counts, width, bottom=adapt_counts, 
           label='Spatial-like', color=spatial_color, alpha=0.8)
    
    ax4.set_xticks(x_layers)
    ax4.set_xticklabels(layer_order)
    ax4.set_xlabel('Cortical Layer', fontsize=11)
    ax4.set_ylabel('Number of Cells', fontsize=11)
    ax4.set_title(f'{detail_lm}-Preferring Cells by Layer', 
                 fontsize=12, fontweight='bold')
    ax4.legend()
    
    # =========================================================================
    # Panel 5: Proportion view for detail landmark
    # =========================================================================
    ax5 = axes[1, 1]
    
    proportions_adapt = [100 * counts['adaptation'][layer] / counts['total'][layer] 
                         if counts['total'][layer] > 0 else 0 
                         for layer in layer_order]
    proportions_spatial = [100 - p for p in proportions_adapt]
    
    ax5.bar(x_layers, proportions_adapt, width, label='Adaptation-like', 
           color=adapt_color, alpha=0.8)
    ax5.bar(x_layers, proportions_spatial, width, bottom=proportions_adapt, 
           label='Spatial-like', color=spatial_color, alpha=0.8)
    
    ax5.set_xticks(x_layers)
    ax5.set_xticklabels(layer_order)
    ax5.set_xlabel('Cortical Layer', fontsize=11)
    ax5.set_ylabel('Proportion (%)', fontsize=11)
    ax5.set_title(f'{detail_lm} Cell Proportions by Layer', 
                 fontsize=12, fontweight='bold')
    ax5.axhline(50, color='black', linestyle='--', alpha=0.5)
    ax5.set_ylim(0, 100)
    
    # Add percentage labels
    for i, pct in enumerate(proportions_adapt):
        ax5.text(i, pct/2, f'{pct:.0f}%', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
    
    # =========================================================================
    # Panel 6: Summary statistics table
    # =========================================================================
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create summary table
    table_data = []
    for lm in landmark_names:
        n = results[lm]['n_cells']
        diff = results[lm]['difference']
        p = results[lm]['fisher_p']
        sig = results[lm]['significance']
        
        if p < 0.001:
            p_str = '<0.001'
        else:
            p_str = f'{p:.3f}'
        
        table_data.append([lm, n, f'{diff:+.1f}pp', p_str, sig])
    
    table = ax6.table(cellText=table_data,
                     colLabels=['Landmark', 'N cells', 'Δ (Deep-Sup)', 'p-value', 'Sig'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0.2, 1, 0.7])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color significant rows
    for i, lm in enumerate(landmark_names):
        if results[lm]['fisher_p'] < 0.05:
            for j in range(5):
                table[(i+1, j)].set_facecolor('#FFEBEE')
    
    ax6.text(0.5, 0.95, 'SUMMARY: Layer Effect by Landmark Preference',
            transform=ax6.transAxes, ha='center', fontsize=11, fontweight='bold')
    
    ax6.text(0.5, 0.05, 
            'Red background = significant (p<0.05)\npp = percentage points',
            transform=ax6.transAxes, ha='center', fontsize=8, style='italic')
    
    # =========================================================================
    # Overall title
    # =========================================================================
    plt.suptitle('Layer Effect on Adaptation-like Cells:\nAnalysis Across All Landmark Preferences',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved comprehensive figure: {save_path}")
    
    return fig


# ============================================================================
# SUMMARY INTERPRETATION
# ============================================================================

def print_interpretation(results):
    """
    Print interpretation of results.
    """
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    # Check which landmarks show significant effect
    significant_lms = [lm for lm in results.keys() 
                      if results[lm]['fisher_p'] < 0.05]
    
    nonsignificant_lms = [lm for lm in results.keys() 
                         if results[lm]['fisher_p'] >= 0.05]
    
    all_positive = all(results[lm]['difference'] > 0 for lm in results.keys())
    all_negative = all(results[lm]['difference'] < 0 for lm in results.keys())
    
    print(f"\nSignificant layer effects found for: {significant_lms}")
    print(f"Non-significant layer effects for: {nonsignificant_lms}")
    
    if all_positive:
        print("\n✓ ALL landmark preferences show deep > superficial adaptation")
        print("  → Adaptation-like phenotype is a GENERAL property of deep layers")
        print("  → Supports Hypothesis A: Layer-specific coding strategy")
    elif all_negative:
        print("\n✓ ALL landmark preferences show superficial > deep adaptation")
        print("  → This would be surprising and contradict L1 findings")
    else:
        print("\n✓ Mixed results across landmark preferences")
        print("  → Adaptation phenotype may depend on landmark identity")
        print("  → Supports Hypothesis B: Landmark-specific effects")
    
    # Check if L1 effect is special
    if 'L1' in results and 'L4' in results:
        l1_diff = results['L1']['difference']
        l4_diff = results['L4']['difference']
        
        print(f"\nL1 effect: {l1_diff:+.1f} pp (p={results['L1']['fisher_p']:.4f})")
        print(f"L4 effect: {l4_diff:+.1f} pp (p={results['L4']['fisher_p']:.4f})")
        
        if abs(l1_diff) > 2 * abs(l4_diff):
            print("  → L1 shows much stronger layer effect than L4")
            print("  → First landmark may be special")
    
    print("\n" + "=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def run_all_cells_analysis(pca_data_path, figure_dir):
    """Main function."""
    
    print("=" * 70)
    print("ALL CELLS LAYER ANALYSIS")
    print("Stratified by Landmark Preference")
    print("=" * 70)
    
    os.makedirs(figure_dir, exist_ok=True)
    
    # Load data
    data = load_data(pca_data_path)
    
    # Compute early slopes for ALL cells
    print("\nComputing early slopes for all cells...")
    early_slopes = compute_early_slope(
        data['spatial_profiles_zscore'],
        data['bin_centers']
    )
    
    # Classify as adaptation-like vs spatial-like
    adaptation_mask, threshold = classify_cells_by_slope(
        early_slopes, 
        threshold=EARLY_SLOPE_THRESHOLD
    )
    
    print(f"  Early slope threshold: {threshold:.4f}")
    print(f"  Adaptation-like: {np.sum(adaptation_mask)} ({100*np.mean(adaptation_mask):.1f}%)")
    print(f"  Spatial-like: {np.sum(~adaptation_mask)} ({100*np.mean(~adaptation_mask):.1f}%)")
    
    # Stratified analysis
    results = analyze_by_landmark_and_layer(data, adaptation_mask, early_slopes)
    
    # Visualization
    fig = plot_comprehensive_comparison(
        results,
        save_path=os.path.join(figure_dir, 'all_cells_layer_effect.png')
    )
    
    # Interpretation
    print_interpretation(results)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    
    plt.show()
    
    return results


if __name__ == "__main__":
    results = run_all_cells_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR
    )
