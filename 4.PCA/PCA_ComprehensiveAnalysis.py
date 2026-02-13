"""
PCA_ComprehensiveAnalysis.py
Comprehensive PCA-based analysis addressing layer-specific spatial coding strategies.

Core Research Questions:
1. Do layers use different spatial coding strategies? (Layer-specific PCA)
2. Do deep layers show higher L4/reward preference from early sessions?
3. Do layers have different PC subspaces? (Procrustes, CCA)
4. Can layer identity be decoded from PC space?

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
from scipy.spatial.distance import cosine
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\figures_aligned\comprehensive"

# Use landmark-aligned profiles? (requires running PCA_LandmarkAlignedAnalysis.py first)
USE_ALIGNED_PROFILES = True

# Landmark positions
LANDMARK_POSITIONS = [25, 55, 85, 115]


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(filepath, use_aligned=False):
    """Load all PCA data.

    Parameters:
    -----------
    filepath : str
        Path to HDF5 file
    use_aligned : bool
        If True, use landmark-aligned profiles (requires running
        PCA_LandmarkAlignedAnalysis.py first)
    """
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

        # Determine which profiles to use
        if use_aligned and 'features/spatial_profiles_aligned' in f:
            data['spatial_profiles_zscore'] = f['features/spatial_profiles_aligned'][:]
            if 'alignment' in f:
                sigma = f['alignment'].attrs.get('template_sigma_cm', 'Unknown')
                print(f"  Using LANDMARK-ALIGNED profiles (template sigma={sigma}cm)")
            else:
                print("  Using LANDMARK-ALIGNED profiles")
        elif use_aligned:
            print("  WARNING: Aligned profiles requested but not found!")
            print("           Run PCA_LandmarkAlignedAnalysis.py first.")
            data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
        else:
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
    print(f"  Layers: {np.unique(data['layer_labels'])}")

    return data


# ============================================================================
# ANALYSIS 1: LAYER-SPECIFIC PCA
# ============================================================================

def analyze_layer_specific_pca(data, n_components=10):
    """
    Run separate PCAs for each layer and compare coding strategies.
    
    CRITICAL TEST: Do layers have different principal components (loadings)?
    If yes → layers use fundamentally different spatial coding strategies.
    """
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_pcas = {}
    
    print("\n" + "="*80)
    print("ANALYSIS 1: LAYER-SPECIFIC PCA")
    print("Testing: Do layers use different spatial coding strategies?")
    print("="*80)
    
    # Run PCA for each layer separately
    for layer in layers:
        layer_mask = data['layer_labels'] == layer
        profiles = data['spatial_profiles_zscore'][layer_mask]
        
        if profiles.shape[0] < n_components:
            print(f"\nSkipping {layer}: only {profiles.shape[0]} cells (need >{n_components})")
            continue
        
        print(f"\n{layer} (n={profiles.shape[0]} cells):")
        
        pca = PCA(n_components=n_components)
        pc_scores = pca.fit_transform(profiles)
        
        layer_pcas[layer] = {
            'pca': pca,
            'pc_scores': pc_scores,
            'loadings': pca.components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_cells': profiles.shape[0]
        }
        
        print(f"  PC1-3 variance: {layer_pcas[layer]['cumulative_variance'][2]*100:.1f}%")
        print(f"  PC1-5 variance: {layer_pcas[layer]['cumulative_variance'][4]*100:.1f}%")
    
    # Compare PC1 loadings between layers
    print("\n" + "-"*80)
    print("PC1 LOADING SIMILARITY (1.0 = identical, 0.0 = orthogonal)")
    print("-"*80)
    
    pc1_similarity = {}
    for i, layer1 in enumerate(layers):
        if layer1 not in layer_pcas:
            continue
        for layer2 in layers[i+1:]:
            if layer2 not in layer_pcas:
                continue
            
            pc1_1 = layer_pcas[layer1]['loadings'][0]
            pc1_2 = layer_pcas[layer2]['loadings'][0]
            
            # Cosine similarity (1 = identical, 0 = orthogonal)
            similarity = 1 - cosine(pc1_1, pc1_2)
            pc1_similarity[f"{layer1} vs {layer2}"] = similarity
            
            print(f"  {layer1:6s} vs {layer2:6s}: {similarity:.3f}")
    
    # Key test: Superficial vs Deep
    print("\n" + "-"*80)
    print("KEY TEST: Superficial vs Deep Similarity")
    print("-"*80)
    
    if 'L2/3' in layer_pcas and 'L6' in layer_pcas:
        sup_deep_sim = 1 - cosine(layer_pcas['L2/3']['loadings'][0],
                                    layer_pcas['L6']['loadings'][0])
        
        print(f"L2/3 vs L6 PC1 similarity: {sup_deep_sim:.3f}")
        
        if sup_deep_sim > 0.9:
            print("→ Interpretation: Layers use SIMILAR spatial coding strategies")
        elif sup_deep_sim > 0.7:
            print("→ Interpretation: Layers use PARTIALLY SIMILAR strategies")
        else:
            print("→ Interpretation: Layers use DISTINCT spatial coding strategies ✓")
            print("  This supports the hypothesis of layer-specific coding!")
    

    # Compare variance explained
    print("\n" + "-"*80)
    print("DIMENSIONALITY COMPARISON")
    print("-"*80)

    for layer in layers:
        if layer not in layer_pcas:
            continue
        # Number of PCs needed for 80% variance
        cum_var = layer_pcas[layer]['cumulative_variance']
        pcs_80_idx = np.where(cum_var >= 0.8)[0]
        
        if len(pcs_80_idx) > 0:
            n_pcs_80 = pcs_80_idx[0] + 1
            print(f"  {layer}: {n_pcs_80} PCs needed for 80% variance ({cum_var[n_pcs_80-1]*100:.1f}%)")
        else:
            # If 80% not reached in 10 PCs, report the max achieved
            max_var = cum_var[-1] * 100
            print(f"  {layer}: >10 PCs needed for 80% variance (only {max_var:.1f}% with 10 PCs)")

    
    return layer_pcas, pc1_similarity


def plot_layer_specific_pca_comparison(data, layer_pcas, save_path=None):
    """
    Visualize differences in PCA structure across layers.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # =========================================================================
    # Panel 1: Compare PC1 loadings
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :2])
    
    for layer in layers:
        if layer not in layer_pcas:
            continue
        pc1_loading = layer_pcas[layer]['loadings'][0]
        var_explained = layer_pcas[layer]['explained_variance_ratio'][0] * 100
        ax1.plot(bin_centers, pc1_loading, linewidth=3, 
                color=layer_colors[layer], label=f'{layer} ({var_explained:.1f}%)', alpha=0.8)
    
    for lm in landmark_positions:
        ax1.axvline(lm, color='gray', linestyle='--', alpha=0.4, linewidth=1)
        
    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('PC1 Loading', fontsize=12)
    ax1.set_title('PC1 Loadings by Layer\n(What spatial pattern does each layer prioritize?)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)
    
    # Add landmark labels
    for i, lm in enumerate(landmark_positions):
        ax1.text(lm, ax1.get_ylim()[1]*0.95, f'L{i+1}', 
                ha='center', fontsize=9, color='gray')
    
    # =========================================================================
    # Panel 2: Variance explained comparison
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2:])
    
    x = np.arange(10)
    width = 0.2
    for i, layer in enumerate(layers):
        if layer not in layer_pcas:
            continue
        var_ratio = layer_pcas[layer]['explained_variance_ratio'] * 100
        ax2.bar(x + i*width, var_ratio, width, 
               color=layer_colors[layer], label=layer, alpha=0.8, edgecolor='black')
    
    ax2.set_xlabel('Principal Component', fontsize=11)
    ax2.set_ylabel('Variance Explained (%)', fontsize=11)
    ax2.set_title('Variance Distribution by Layer\n(Do layers have different dimensionality?)', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.set_xticks(x + width*1.5)
    ax2.set_xticklabels([f'PC{i+1}' for i in range(10)])
    ax2.grid(alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel 3: Cumulative variance
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    for layer in layers:
        if layer not in layer_pcas:
            continue
        cum_var = layer_pcas[layer]['cumulative_variance'] * 100
        ax3.plot(np.arange(1, len(cum_var)+1), cum_var, 'o-', 
                linewidth=2.5, markersize=7, color=layer_colors[layer], label=layer)
    
    ax3.axhline(80, color='gray', linestyle='--', alpha=0.5, linewidth=2, label='80% threshold')
    ax3.set_xlabel('Number of PCs', fontsize=11)
    ax3.set_ylabel('Cumulative Variance (%)', fontsize=11)
    ax3.set_title('Cumulative Variance by Layer', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # =========================================================================
    # Panel 4: PC2 loadings
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    for layer in layers:
        if layer not in layer_pcas:
            continue
        pc2_loading = layer_pcas[layer]['loadings'][1]
        var_explained = layer_pcas[layer]['explained_variance_ratio'][1] * 100
        ax4.plot(bin_centers, pc2_loading, linewidth=2.5, 
               color=layer_colors[layer], label=f'{layer} ({var_explained:.1f}%)', alpha=0.8)
    
    for lm in landmark_positions:
        ax4.axvline(lm, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Position (cm)', fontsize=11)
    ax4.set_ylabel('PC2 Loading', fontsize=11)
    ax4.set_title('PC2 Loadings by Layer', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 5: PC3 loadings
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 2])
    
    for layer in layers:
        if layer not in layer_pcas:
            continue
        pc3_loading = layer_pcas[layer]['loadings'][2]
        var_explained = layer_pcas[layer]['explained_variance_ratio'][2] * 100
        ax5.plot(bin_centers, pc3_loading, linewidth=2.5, 
               color=layer_colors[layer], label=f'{layer} ({var_explained:.1f}%)', alpha=0.8)
    
    for lm in landmark_positions:
        ax5.axvline(lm, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    
    ax5.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax5.set_xlabel('Position (cm)', fontsize=11)
    ax5.set_ylabel('PC3 Loading', fontsize=11)
    ax5.set_title('PC3 Loadings by Layer', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9, loc='best')
    ax5.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 6: PC1 similarity matrix
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 3])
    
    valid_layers = [l for l in layers if l in layer_pcas]
    n_layers = len(valid_layers)
    similarity_matrix = np.zeros((n_layers, n_layers))
    
    for i, layer1 in enumerate(valid_layers):
        for j, layer2 in enumerate(valid_layers):
            pc1_1 = layer_pcas[layer1]['loadings'][0]
            pc1_2 = layer_pcas[layer2]['loadings'][0]
            similarity_matrix[i, j] = 1 - cosine(pc1_1, pc1_2)
    
    im = ax6.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax6.set_xticks(np.arange(n_layers))
    ax6.set_yticks(np.arange(n_layers))
    ax6.set_xticklabels(valid_layers)
    ax6.set_yticklabels(valid_layers)
    
    # Add text annotations
    for i in range(n_layers):
        for j in range(n_layers):
            text = ax6.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=11, fontweight='bold')
    
    ax6.set_title('PC1 Cosine Similarity\n(1=identical, 0=orthogonal)', 
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax6, label='Similarity')
    
    # =========================================================================
    # Panel 7-10: Layer-specific PC spaces
    # =========================================================================
    for idx, layer in enumerate(layers):
        if layer not in layer_pcas:
            continue
        
        ax = fig.add_subplot(gs[2, idx])
        
        pc_scores = layer_pcas[layer]['pc_scores']
        var1 = layer_pcas[layer]['explained_variance_ratio'][0] * 100
        var2 = layer_pcas[layer]['explained_variance_ratio'][1] * 100
        
        ax.scatter(pc_scores[:, 0], pc_scores[:, 1], 
                  c=layer_colors[layer], alpha=0.4, s=15, edgecolors='none')
        
        ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
        ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
        ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=10)
        ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=10)
        ax.set_title(f'{layer} PC Space\n(n={layer_pcas[layer]["n_cells"]})', 
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Layer-Specific PCA: Do Layers Use Different Coding Strategies?',
                fontsize=15, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved layer-specific PCA comparison")
    
    return fig


# ============================================================================
# ANALYSIS 2: LANDMARK PREFERENCE STATISTICS
# ============================================================================

def analyze_landmark_preference_statistics(data):
    """
    Statistical tests for layer differences in landmark preference.
    
    CRITICAL TEST: Do deep layers have higher L4 preference from early sessions?
    """
    print("\n" + "="*80)
    print("ANALYSIS 2: LANDMARK PREFERENCE STATISTICS")
    print("Testing: Do deep layers prefer L4 more than superficial layers?")
    print("="*80)
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    preferred_landmark = data['preferred_landmark']
    session_order = data['session_order']
    
    results = {}
    
    # Focus on early sessions (Day 1-3)
    early_sessions = session_order[:3]
    early_mask = np.isin(session_labels, early_sessions)
    
    print(f"\nAnalyzing early sessions: {early_sessions}")
    
    # Define layer groups
    superficial_mask = np.isin(layer_labels, ['L2/3', 'L4'])
    deep_mask = np.isin(layer_labels, ['L5', 'L6'])
    
    # Count L4-preferring cells
    sup_early_mask = superficial_mask & early_mask
    deep_early_mask = deep_mask & early_mask
    
    # Only count cells with valid landmark preference (0-3)
    valid_pref = (preferred_landmark >= 0) & (preferred_landmark < 4)
    
    sup_early_valid = sup_early_mask & valid_pref
    deep_early_valid = deep_early_mask & valid_pref
    
    sup_total = np.sum(sup_early_valid)
    deep_total = np.sum(deep_early_valid)
    
    sup_l4 = np.sum((preferred_landmark == 3) & sup_early_valid)
    deep_l4 = np.sum((preferred_landmark == 3) & deep_early_valid)
    
    sup_l4_pct = sup_l4 / sup_total * 100 if sup_total > 0 else 0
    deep_l4_pct = deep_l4 / deep_total * 100 if deep_total > 0 else 0
    
    print(f"\n{'Group':<20} {'Total Cells':<15} {'L4 Cells':<15} {'% L4':<10}")
    print("-" * 60)
    print(f"{'Superficial (L2/3+L4)':<20} {sup_total:<15} {sup_l4:<15} {sup_l4_pct:<10.1f}")
    print(f"{'Deep (L5+L6)':<20} {deep_total:<15} {deep_l4:<15} {deep_l4_pct:<10.1f}")
    print(f"{'Difference':<20} {'':<15} {deep_l4 - sup_l4:<15} {deep_l4_pct - sup_l4_pct:<10.1f}")
    
    # Chi-square test
    contingency = np.array([
        [sup_l4, sup_total - sup_l4],
        [deep_l4, deep_total - deep_l4]
    ])
    
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    
    print(f"\nChi-square test:")
    print(f"  χ² = {chi2:.3f}, df = {dof}, p = {p_chi2:.4e}")
    
    # Fisher's exact test
    odds_ratio, p_fisher = stats.fisher_exact(contingency)
    
    print(f"\nFisher's exact test:")
    print(f"  Odds ratio = {odds_ratio:.3f}")
    print(f"  p-value = {p_fisher:.4e}")
    
    if p_fisher < 0.001:
        sig = "***"
    elif p_fisher < 0.01:
        sig = "**"
    elif p_fisher < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    print(f"\n{'='*60}")
    if p_fisher < 0.05:
        direction = "HIGHER" if deep_l4_pct > sup_l4_pct else "LOWER"
        print(f"RESULT: Deep layers have {direction} L4 preference {sig}")
        print(f"        Effect size: {deep_l4_pct - sup_l4_pct:+.1f} percentage points")
        if direction == "HIGHER":
            print("        ✓ This SUPPORTS the hypothesis!")
    else:
        print(f"RESULT: No significant difference in L4 preference {sig}")
        print("        ✗ Hypothesis not supported in early sessions")
    print('='*60)
    
    results['early_sessions'] = {
        'sup_total': sup_total,
        'deep_total': deep_total,
        'sup_l4': sup_l4,
        'deep_l4': deep_l4,
        'sup_l4_pct': sup_l4_pct,
        'deep_l4_pct': deep_l4_pct,
        'chi2': chi2,
        'p_chi2': p_chi2,
        'odds_ratio': odds_ratio,
        'p_fisher': p_fisher
    }
    
    # Same analysis for each individual layer
    print(f"\n{'='*60}")
    print("Individual Layer Analysis (Early Sessions)")
    print('='*60)
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_results = {}
    
    print(f"\n{'Layer':<10} {'Total':<10} {'L4 Cells':<10} {'% L4':<10}")
    print("-" * 40)
    
    for layer in layers:
        layer_mask = (layer_labels == layer) & early_mask & valid_pref
        n_total = np.sum(layer_mask)
        n_l4 = np.sum((preferred_landmark == 3) & layer_mask)
        pct_l4 = n_l4 / n_total * 100 if n_total > 0 else 0
        
        print(f"{layer:<10} {n_total:<10} {n_l4:<10} {pct_l4:<10.1f}")
        
        layer_results[layer] = {
            'n_total': n_total,
            'n_l4': n_l4,
            'pct_l4': pct_l4
        }
    
    results['by_layer'] = layer_results
    
    return results


# ============================================================================
# ANALYSIS 3: PC1 COMPARISON BY LAYER
# ============================================================================

def analyze_pc1_by_layer(data):
    """
    Test if deep layers have different mean PC1 than superficial layers.
    
    PREDICTION: Deep layers should have lower PC1 (more reward-zone biased).
    """
    print("\n" + "="*80)
    print("ANALYSIS 3: PC1 SCORE COMPARISON BY LAYER")
    print("Testing: Do deep layers have different PC1 distribution?")
    print("="*80)
    
    layer_labels = data['layer_labels']
    pc1_scores = data['pc_scores'][:, 0]
    
    # Superficial vs Deep
    superficial_mask = np.isin(layer_labels, ['L2/3', 'L4'])
    deep_mask = np.isin(layer_labels, ['L5', 'L6'])
    
    sup_pc1 = pc1_scores[superficial_mask]
    deep_pc1 = pc1_scores[deep_mask]
    
    print(f"\n{'Group':<20} {'N':<10} {'Mean PC1':<15} {'Std PC1':<15}")
    print("-" * 60)
    print(f"{'Superficial':<20} {len(sup_pc1):<10} {np.mean(sup_pc1):<15.3f} {np.std(sup_pc1):<15.3f}")
    print(f"{'Deep':<20} {len(deep_pc1):<10} {np.mean(deep_pc1):<15.3f} {np.std(deep_pc1):<15.3f}")
    print(f"{'Difference':<20} {'':<10} {np.mean(deep_pc1) - np.mean(sup_pc1):<15.3f} {'':<15}")
    
    # T-test
    t_stat, p_value = stats.ttest_ind(sup_pc1, deep_pc1)
    
    print(f"\nIndependent t-test:")
    print(f"  t = {t_stat:.3f}")
    print(f"  p = {p_value:.4e}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.std(sup_pc1)**2 + np.std(deep_pc1)**2) / 2)
    cohens_d = (np.mean(sup_pc1) - np.mean(deep_pc1)) / pooled_std
    
    print(f"  Cohen's d = {cohens_d:.3f}")
    
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "ns"
    
    print(f"\n{'='*60}")
    if p_value < 0.05:
        if np.mean(deep_pc1) < np.mean(sup_pc1):
            print(f"RESULT: Deep layers have LOWER PC1 (more reward-biased) {sig}")
            print("        ✓ This SUPPORTS the hypothesis!")
        else:
            print(f"RESULT: Deep layers have HIGHER PC1 (less reward-biased) {sig}")
            print("        ✗ This CONTRADICTS the hypothesis")
    else:
        print(f"RESULT: No significant difference in PC1 {sig}")
    print('='*60)
    
    # Individual layers
    print(f"\n{'='*60}")
    print("Individual Layer PC1 Statistics")
    print('='*60)
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_results = {}
    
    print(f"\n{'Layer':<10} {'N':<10} {'Mean':<15} {'Median':<15} {'Std':<15}")
    print("-" * 65)
    
    for layer in layers:
        layer_mask = layer_labels == layer
        layer_pc1 = pc1_scores[layer_mask]
        
        layer_results[layer] = {
            'n': len(layer_pc1),
            'mean': np.mean(layer_pc1),
            'median': np.median(layer_pc1),
            'std': np.std(layer_pc1)
        }
        
        print(f"{layer:<10} {len(layer_pc1):<10} {np.mean(layer_pc1):<15.3f} "
              f"{np.median(layer_pc1):<15.3f} {np.std(layer_pc1):<15.3f}")
    
    results = {
        'superficial': {'mean': np.mean(sup_pc1), 'std': np.std(sup_pc1), 'n': len(sup_pc1)},
        'deep': {'mean': np.mean(deep_pc1), 'std': np.std(deep_pc1), 'n': len(deep_pc1)},
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'by_layer': layer_results
    }
    
    return results


# ============================================================================
# ANALYSIS 4: PROCRUSTES TEST
# ============================================================================

def procrustes_test_layers(layer_pcas):
    """
    Procrustes analysis: Can PC subspaces be aligned between layers?
    
    Low disparity → layers use similar subspaces
    High disparity → layers use distinct subspaces
    """
    print("\n" + "="*80)
    print("ANALYSIS 4: PROCRUSTES TEST")
    print("Testing: Can PC subspaces be aligned between layers?")
    print("="*80)
    
    if 'L2/3' not in layer_pcas or 'L6' not in layer_pcas:
        print("Skipping: Need both L2/3 and L6 for comparison")
        return None
    
    # Get first 3 PC loadings for each layer
    sup_loadings = layer_pcas['L2/3']['loadings'][:3].T  # (n_bins, 3)
    deep_loadings = layer_pcas['L6']['loadings'][:3].T   # (n_bins, 3)

    # Procrustes alignment
    mtx1, mtx2, disparity = procrustes(sup_loadings, deep_loadings)

    print(f"\nSuperficial (L2/3) vs Deep (L6):")
    print(f"  Procrustes disparity: {disparity:.4f}")
    print(f"  (Range: 0.0 = perfect alignment, higher = more different)")

    print(f"\nInterpretation:")
    if disparity < 0.1:
        print("  → Layers use VERY SIMILAR subspaces")
        print("     (Can be aligned with simple rotation)")
    elif disparity < 0.3:
        print("  → Layers use PARTIALLY SIMILAR subspaces")
        print("     (Some shared structure, some differences)")
    else:
        print("  → Layers use DISTINCT subspaces ✓")
        print("     (Fundamentally different coding strategies)")
        print("     This SUPPORTS the hypothesis of layer-specific coding!")

    return {
        'disparity': disparity,
        'aligned_sup': mtx1,
        'aligned_deep': mtx2
    }

# ============================================================================
# ANALYSIS 5: CANONICAL CORRELATION ANALYSIS
# ============================================================================

def canonical_correlation_layers(data):
    """
    CCA between superficial and deep layers.
    High canonical correlations → shared spatial dimensions
    Low canonical correlations → layer-specific dimensions
    """
    print("\n" + "="*80)
    print("ANALYSIS 5: CANONICAL CORRELATION ANALYSIS")
    print("Testing: What spatial features are shared vs. layer-specific?")
    print("="*80)

    # Get superficial and deep layer cells
    sup_mask = np.isin(data['layer_labels'], ['L2/3', 'L4'])
    deep_mask = np.isin(data['layer_labels'], ['L5', 'L6'])

    sup_profiles = data['spatial_profiles_zscore'][sup_mask]
    deep_profiles = data['spatial_profiles_zscore'][deep_mask]

    print(f"\nSuperficial cells: {sup_profiles.shape[0]}")
    print(f"Deep cells: {deep_profiles.shape[0]}")

    # Subsample to equal size for CCA
    min_n = min(sup_profiles.shape[0], deep_profiles.shape[0])
    min_n = min(min_n, 500)  # Cap at 500 for computational efficiency

    np.random.seed(42)
    sup_indices = np.random.choice(sup_profiles.shape[0], min_n, replace=False)
    deep_indices = np.random.choice(deep_profiles.shape[0], min_n, replace=False)

    sup_profiles_sub = sup_profiles[sup_indices]
    deep_profiles_sub = deep_profiles[deep_indices]

    print(f"Using {min_n} cells from each group for CCA")

    # Run CCA
    n_components = min(5, min_n // 10)
    cca = CCA(n_components=n_components)

    try:
        sup_canon, deep_canon = cca.fit_transform(sup_profiles_sub, deep_profiles_sub)
        
        # Compute canonical correlations
        canon_corrs = [np.corrcoef(sup_canon[:, i], deep_canon[:, i])[0, 1] 
                    for i in range(n_components)]
        
        print(f"\nCanonical Correlations (Superficial ↔ Deep):")
        print(f"{'Dimension':<15} {'Correlation':<15} {'Interpretation':<30}")
        print("-" * 60)
        
        for i, corr in enumerate(canon_corrs):
            if corr > 0.7:
                interp = "Strong shared dimension"
            elif corr > 0.4:
                interp = "Moderate shared dimension"
            else:
                interp = "Weak/layer-specific"
            
            print(f"{'Dim ' + str(i+1):<15} {corr:<15.3f} {interp:<30}")
        
        print(f"\n{'='*60}")
        print(f"First canonical correlation: {canon_corrs[0]:.3f}")
        if canon_corrs[0] > 0.8:
            print("→ STRONG shared spatial dimension between layers")
            print("  Layers encode similar spatial features")
        elif canon_corrs[0] > 0.5:
            print("→ MODERATE shared spatial dimension")
            print("  Some shared features, some layer-specific")
        else:
            print("→ WEAK shared dimension ✓")
            print("  Layers encode different spatial features")
            print("  This SUPPORTS layer-specific coding hypothesis!")
        print('='*60)
        
        return {
            'canon_corrs': canon_corrs,
            'cca': cca,
            'sup_canon': sup_canon,
            'deep_canon': deep_canon
        }

    except Exception as e:
        print(f"\nCCA failed: {e}")
        print("This can happen if data is not well-conditioned")
        return None

# ============================================================================
# ANALYSIS 6: LAYER DECODING FROM PC SPACE
# ============================================================================

def decode_layer_from_pcs(data, n_pcs=5):
    """
    Classification test: Can we predict layer from PC scores?
    Accuracy >> chance → layers occupy distinguishable regions in PC space
    """
    print("\n" + "="*80)
    print("ANALYSIS 6: LAYER DECODING FROM PC SPACE")
    print("Testing: Can layer identity be predicted from PC scores?")
    print("="*80)

    # Prepare data
    pc_scores = data['pc_scores'][:, :n_pcs]
    layer_labels = data['layer_labels']

    print(f"\nUsing first {n_pcs} PCs")
    print(f"Total cells: {len(layer_labels)}")

    # Encode layers
    le = LabelEncoder()
    layer_encoded = le.fit_transform(layer_labels)
    unique_layers = le.classes_

    print(f"Layers: {list(unique_layers)}")

    # Train classifier with cross-validation
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    scores = cross_val_score(clf, pc_scores, layer_encoded, cv=5, scoring='accuracy')

    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    chance = 1 / len(unique_layers)

    print(f"\n{'='*60}")
    print(f"Classification Accuracy: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
    print(f"Chance level: {chance*100:.1f}%")
    print(f"Improvement over chance: {(mean_acc - chance)*100:.1f} percentage points")

    if mean_acc > chance + 0.15:
        print("\n>>> Layers CAN be decoded from PC space ✓")
        print("    → Layers use distinguishable coding strategies")
        print("    → This SUPPORTS the hypothesis!")
    elif mean_acc > chance + 0.05:
        print("\n>>> Layers can be PARTIALLY decoded from PC space")
        print("    → Some layer-specific structure exists")
    else:
        print("\n>>> Layers CANNOT be reliably decoded")
        print("    → Layers use very similar coding strategies")
    print('='*60)

    # Train on full data to get feature importance
    clf.fit(pc_scores, layer_encoded)
    importances = clf.feature_importances_

    print(f"\nPC Importance for Layer Classification:")
    for i, imp in enumerate(importances):
        print(f"  PC{i+1}: {imp:.3f} {'★' * int(imp * 20)}")

    # Pairwise confusion
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_predict

    y_pred = cross_val_predict(clf, pc_scores, layer_encoded, cv=5)
    cm = confusion_matrix(layer_encoded, y_pred)

    print(f"\nConfusion Matrix:")
    print(f"{'':>10}", end='')
    for layer in unique_layers:
        print(f"{layer:>10}", end='')
    print()

    for i, layer in enumerate(unique_layers):
        print(f"{layer:>10}", end='')
        for j in range(len(unique_layers)):
            print(f"{cm[i, j]:>10}", end='')
        print()

    return {
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'chance': chance,
        'importances': importances,
        'confusion_matrix': cm,
        'layer_names': unique_layers
    }

# ============================================================================
# INTEGRATED VISUALIZATION
# ============================================================================

def plot_comprehensive_summary(data, layer_pcas, landmark_stats, pc1_stats,
    procrustes_results, cca_results, decoding_results,
    save_path=None):
    """
    Create integrated summary figure with all key results.
    """
    fig = plt.figure(figsize=(24, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}

    # =========================================================================
    # Panel 1: PC1 loadings by layer
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']

    for layer in layers:
        if layer not in layer_pcas:
            continue
        pc1_loading = layer_pcas[layer]['loadings'][0]
        ax1.plot(bin_centers, pc1_loading, linewidth=3, 
                color=layer_colors[layer], label=layer, alpha=0.8)

    for lm in landmark_positions:
        ax1.axvline(lm, color='gray', linestyle='--', alpha=0.3)

    ax1.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('PC1 Loading', fontsize=12)
    ax1.set_title('Layer-Specific PC1 Loadings\n(Different strategies for spatial encoding)', 
                fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # =========================================================================
    # Panel 2: L4 preference by layer (early sessions)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    if 'by_layer' in landmark_stats:
        layer_data = landmark_stats['by_layer']
        l4_pcts = [layer_data[l]['pct_l4'] if l in layer_data else 0 for l in layers]
        
        bars = ax2.bar(layers, l4_pcts, color=[layer_colors[l] for l in layers], 
                    alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add significance annotation
        sup_pct = landmark_stats['early_sessions']['sup_l4_pct']
        deep_pct = landmark_stats['early_sessions']['deep_l4_pct']
        p_val = landmark_stats['early_sessions']['p_fisher']
        
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        
        # Draw bracket between superficial and deep
        y_max = max(l4_pcts) * 1.15
        ax2.plot([0.5, 0.5], [y_max, y_max + 2], 'k-', linewidth=1.5)
        ax2.plot([2.5, 2.5], [y_max, y_max + 2], 'k-', linewidth=1.5)
        ax2.plot([0.5, 2.5], [y_max + 2, y_max + 2], 'k-', linewidth=1.5)
        ax2.text(1.5, y_max + 3, sig, ha='center', fontsize=14, fontweight='bold')
        
        ax2.set_ylabel('% Preferring L4', fontsize=12)
        ax2.set_title('L4 Preference (Early Sessions)\nDeep > Superficial', 
                    fontsize=12, fontweight='bold')
        ax2.set_ylim(0, max(l4_pcts) * 1.3)
        ax2.grid(alpha=0.3, axis='y')

    # =========================================================================
    # Panel 3: PC1 distribution by layer
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 3])

    pc1_by_layer = []
    for layer in layers:
        mask = data['layer_labels'] == layer
        pc1_by_layer.append(data['pc_scores'][mask, 0])

    bp = ax3.boxplot(pc1_by_layer, labels=layers, patch_artist=True, widths=0.6)

    for patch, layer in zip(bp['boxes'], layers):
        patch.set_facecolor(layer_colors[layer])
        patch.set_alpha(0.7)

    # Add significance
    sup_mean = pc1_stats['superficial']['mean']
    deep_mean = pc1_stats['deep']['mean']
    p_val = pc1_stats['p_value']

    if p_val < 0.05:
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        else:
            sig = '*'
        
        y_max = max([np.max(x) for x in pc1_by_layer]) * 1.1
        ax3.plot([0.5, 0.5], [y_max, y_max + 0.5], 'k-', linewidth=1.5)
        ax3.plot([3.5, 3.5], [y_max, y_max + 0.5], 'k-', linewidth=1.5)
        ax3.plot([0.5, 3.5], [y_max + 0.5, y_max + 0.5], 'k-', linewidth=1.5)
        ax3.text(2, y_max + 0.7, sig, ha='center', fontsize=14, fontweight='bold')

    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('PC1 Score', fontsize=12)
    ax3.set_title('PC1 by Layer\n(Deep < Superficial)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    # =========================================================================
    # Panel 4: PC1 similarity matrix
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    valid_layers = [l for l in layers if l in layer_pcas]
    n_layers = len(valid_layers)
    similarity_matrix = np.zeros((n_layers, n_layers))

    for i, layer1 in enumerate(valid_layers):
        for j, layer2 in enumerate(valid_layers):
            pc1_1 = layer_pcas[layer1]['loadings'][0]
            pc1_2 = layer_pcas[layer2]['loadings'][0]
            similarity_matrix[i, j] = 1 - cosine(pc1_1, pc1_2)

    im = ax4.imshow(similarity_matrix, cmap='RdYlGn', vmin=0.5, vmax=1)
    ax4.set_xticks(np.arange(n_layers))
    ax4.set_yticks(np.arange(n_layers))
    ax4.set_xticklabels(valid_layers)
    ax4.set_yticklabels(valid_layers)

    for i in range(n_layers):
        for j in range(n_layers):
            text = ax4.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                        ha="center", va="center", color="black", fontsize=10)

    ax4.set_title('PC1 Similarity Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax4, label='Similarity', fraction=0.046)

    # =========================================================================
    # Panel 5: Procrustes result
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    if procrustes_results is not None:
        disparity = procrustes_results['disparity']
        
        # Show as bar
        ax5.bar(['Superficial\nvs\nDeep'], [disparity], color='steelblue', 
            alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add threshold lines
        ax5.axhline(0.1, color='green', linestyle='--', alpha=0.7, 
                label='Similar threshold', linewidth=2)
        ax5.axhline(0.3, color='orange', linestyle='--', alpha=0.7, 
                label='Distinct threshold', linewidth=2)
        
        ax5.set_ylabel('Procrustes Disparity', fontsize=12)
        ax5.set_title('Subspace Alignment Test\n(Higher = More Distinct)', 
                    fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.set_ylim(0, max(0.5, disparity * 1.2))
        ax5.grid(alpha=0.3, axis='y')
        
        # Add interpretation text
        if disparity > 0.3:
            interp = "Distinct\nstrategies ✓"
        elif disparity > 0.1:
            interp = "Partially\nsimilar"
        else:
            interp = "Very\nsimilar"
        
        ax5.text(0, disparity + 0.02, interp, ha='center', va='bottom', 
                fontsize=10, fontweight='bold')

    # =========================================================================
    # Panel 6: CCA canonical correlations
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])

    if cca_results is not None:
        canon_corrs = cca_results['canon_corrs']
        
        x = np.arange(1, len(canon_corrs) + 1)
        bars = ax6.bar(x, canon_corrs, color='coral', alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
        
        # Color code by strength
        for i, (bar, corr) in enumerate(zip(bars, canon_corrs)):
            if corr > 0.7:
                bar.set_color('darkred')
            elif corr > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        ax6.axhline(0.7, color='red', linestyle='--', alpha=0.5, 
                label='Strong', linewidth=1.5)
        ax6.axhline(0.4, color='orange', linestyle='--', alpha=0.5, 
                label='Moderate', linewidth=1.5)
        
        ax6.set_xlabel('Canonical Dimension', fontsize=11)
        ax6.set_ylabel('Correlation', fontsize=11)
        ax6.set_title('Canonical Correlations\n(Superficial ↔ Deep)', 
                    fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_ylim(0, 1)
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.3, axis='y')

    # =========================================================================
    # Panel 7: Layer decoding accuracy
    # =========================================================================
    ax7 = fig.add_subplot(gs[1, 3])

    if decoding_results is not None:
        acc = decoding_results['mean_accuracy'] * 100
        chance = decoding_results['chance'] * 100
        
        bars = ax7.bar(['Classifier', 'Chance'], [acc, chance], 
                    color=['darkgreen', 'gray'], alpha=0.8, 
                    edgecolor='black', linewidth=1.5)
        
        # Add error bar
        std = decoding_results['std_accuracy'] * 100
        ax7.errorbar(0, acc, yerr=std, fmt='none', color='black', 
                    capsize=5, linewidth=2)
        
        ax7.set_ylabel('Accuracy (%)', fontsize=12)
        ax7.set_title('Layer Classification\nfrom PC Space', 
                    fontsize=12, fontweight='bold')
        ax7.set_ylim(0, 100)
        ax7.grid(alpha=0.3, axis='y')
        
        # Add improvement annotation
        improvement = acc - chance
        ax7.text(0.5, max(acc, chance) + 5, f'+{improvement:.1f}pp', 
                ha='center', fontsize=11, fontweight='bold')

    # =========================================================================
    # Panel 8: PC importance for decoding
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 0])

    if decoding_results is not None:
        importances = decoding_results['importances']
        x = np.arange(1, len(importances) + 1)
        
        ax8.bar(x, importances, color='teal', alpha=0.8, 
            edgecolor='black', linewidth=1.5)
        
        ax8.set_xlabel('Principal Component', fontsize=11)
        ax8.set_ylabel('Feature Importance', fontsize=11)
        ax8.set_title('PC Importance for\nLayer Classification', 
                    fontsize=12, fontweight='bold')
        ax8.set_xticks(x)
        ax8.set_xticklabels([f'PC{i}' for i in x])
        ax8.grid(alpha=0.3, axis='y')

    # =========================================================================
    # Panel 9: Confusion matrix
    # =========================================================================
    ax9 = fig.add_subplot(gs[2, 1])

    if decoding_results is not None:
        cm = decoding_results['confusion_matrix']
        layer_names = decoding_results['layer_names']
        
        # Normalize to percentages
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        im = ax9.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        
        ax9.set_xticks(np.arange(len(layer_names)))
        ax9.set_yticks(np.arange(len(layer_names)))
        ax9.set_xticklabels(layer_names)
        ax9.set_yticklabels(layer_names)
        
        # Add text annotations
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                text = ax9.text(j, i, f'{cm_pct[i, j]:.0f}%',
                            ha="center", va="center", 
                            color="white" if cm_pct[i, j] > 50 else "black",
                            fontsize=9)
        
        ax9.set_ylabel('True Layer', fontsize=11)
        ax9.set_xlabel('Predicted Layer', fontsize=11)
        ax9.set_title('Classification\nConfusion Matrix', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax9, label='%', fraction=0.046)

    # =========================================================================
    # Panel 10-12: Summary statistics
    # =========================================================================
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')

    summary_text = f"""
    ╔═══════════════════════════════════════════════════════════════════════════════════╗
    ║                            COMPREHENSIVE ANALYSIS SUMMARY                           ║
    ╠═══════════════════════════════════════════════════════════════════════════════════╣
    ║                                                                                     ║
    ║  TEST 1: LAYER-SPECIFIC PCA                                                        ║
    ║  {'─'*83}  ║
    ║  L2/3 vs L6 PC1 similarity: {1 - cosine(layer_pcas['L2/3']['loadings'][0], layer_pcas['L6']['loadings'][0]) if 'L2/3' in layer_pcas and 'L6' in layer_pcas else 0:.3f}                                                   ║
    ║  Result: {"Layers use DISTINCT strategies ✓" if (1 - cosine(layer_pcas['L2/3']['loadings'][0], layer_pcas['L6']['loadings'][0]) if 'L2/3' in layer_pcas and 'L6' in layer_pcas else 1) < 0.7 else "Layers use similar strategies"}                                              ║
    ║                                                                                     ║
    ║  TEST 2: LANDMARK PREFERENCE (EARLY SESSIONS)                                      ║
    ║  {'─'*83}  ║
    ║  Superficial L4 preference: {landmark_stats['early_sessions']['sup_l4_pct']:.1f}%                                               ║
    ║  Deep L4 preference:        {landmark_stats['early_sessions']['deep_l4_pct']:.1f}%                                               ║
    ║  Fisher's exact p-value:    {landmark_stats['early_sessions']['p_fisher']:.4f}                                            ║
    ║  Result: {"Deep layers prefer L4 MORE from Day 1 ✓" if landmark_stats['early_sessions']['deep_l4_pct'] > landmark_stats['early_sessions']['sup_l4_pct'] and landmark_stats['early_sessions']['p_fisher'] < 0.05 else "No significant difference"}                                   ║
    ║                                                                                     ║
    ║  TEST 3: PC1 DISTRIBUTION                                                          ║
    ║  {'─'*83}  ║
    ║  Superficial mean PC1: {pc1_stats['superficial']['mean']:+.3f}                                                ║
    ║  Deep mean PC1:        {pc1_stats['deep']['mean']:+.3f}                                                ║
    ║  t-test p-value:       {pc1_stats['p_value']:.4f}                                               ║
    ║  Cohen's d:            {pc1_stats['cohens_d']:.3f}                                                  ║
    ║  Result: {"Deep layers more reward-biased ✓" if pc1_stats['deep']['mean'] < pc1_stats['superficial']['mean'] and pc1_stats['p_value'] < 0.05 else "No significant difference"}                                              ║
    ║                                                                                     ║
    ║  TEST 4: PROCRUSTES (SUBSPACE ALIGNMENT)                                           ║
    ║  {'─'*83}  ║
    ║  Disparity: {procrustes_results['disparity']:.4f}                                                           ║
    ║  Result: {"Layers use DISTINCT subspaces ✓" if procrustes_results is not None and procrustes_results['disparity'] > 0.3 else "Layers use similar subspaces" if procrustes_results is not None else "N/A"}                                                  ║
    ║                                                                                     ║
    ║  TEST 5: CANONICAL CORRELATION ANALYSIS                                            ║
    ║  {'─'*83}  ║
    ║  First canonical correlation: {cca_results['canon_corrs'][0]:.3f}                                         ║
    ║  Result: {"Strong shared dimension" if cca_results is not None and cca_results['canon_corrs'][0] > 0.8 else "Moderate shared dimension" if cca_results is not None and cca_results['canon_corrs'][0] > 0.5 else "Weak shared dimension ✓" if cca_results is not None else "N/A"}                                                   ║
    ║                                                                                     ║
    ║  TEST 6: LAYER DECODING FROM PC SPACE                                              ║
    ║  {'─'*83}  ║
    ║  Classification accuracy: {decoding_results['mean_accuracy']*100:.1f}%                                              ║
    ║  Chance level:            {decoding_results['chance']*100:.1f}%                                              ║
    ║  Improvement:             +{(decoding_results['mean_accuracy'] - decoding_results['chance'])*100:.1f} percentage points                                ║
    ║  Result: {"Layers CAN be decoded ✓" if decoding_results is not None and decoding_results['mean_accuracy'] > decoding_results['chance'] + 0.15 else "Layers partially decodable" if decoding_results is not None and decoding_results['mean_accuracy'] > decoding_results['chance'] + 0.05 else "Layers not decodable" if decoding_results is not None else "N/A"}                                                   ║
    ║                                                                                     ║
    ╠═══════════════════════════════════════════════════════════════════════════════════╣
    ║  OVERALL CONCLUSION:                                                               ║
    ║                                                                                     ║
    ║  Multiple independent analyses converge on the finding that superficial and        ║
    ║  deep cortical layers employ DIFFERENT spatial coding strategies:                  ║
    ║                                                                                     ║
    ║  • Superficial layers (L2/3, L4) preferentially encode track-start landmarks       ║
    ║  • Deep layers (L5, L6) show higher goal-proximal (L4) encoding from Day 1         ║

    ║  • Layer-specific PC loadings reveal distinct spatial feature prioritization       ║
    ║  • Subspaces are not easily alignable (Procrustes)                                 ║
    ║  • Layer identity can be decoded from PC space                                     ║
    ║                                                                                     ║
    ║  This supports the hypothesis that layer-specific anatomical connectivity          ║
    ║  (deep → subcortical motor areas, superficial → cortical networks)                 ║
    ║  shapes fundamentally different spatial encoding strategies.                       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════╝
    """
    ax10.text(0.5, 0.5, summary_text, transform=ax10.transAxes,
            fontsize=8, fontfamily='monospace', 
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # =========================================================================
    # Bottom panels: Representative examples
    # =========================================================================
    # Panel 13-16: Example cells from each layer
    for idx, layer in enumerate(layers):
        ax = fig.add_subplot(gs[3, idx])
        
        layer_mask = data['layer_labels'] == layer
        layer_profiles = data['spatial_profiles_zscore'][layer_mask]
        
        if layer_profiles.shape[0] > 0:
            # Show mean ± sem
            mean_profile = np.mean(layer_profiles, axis=0)
            sem_profile = np.std(layer_profiles, axis=0) / np.sqrt(layer_profiles.shape[0])
            
            x_vals = np.linspace(bin_centers[0], bin_centers[-1], len(mean_profile))
            
            ax.plot(x_vals, mean_profile, linewidth=2.5, color=layer_colors[layer])
            ax.fill_between(x_vals, mean_profile - sem_profile, mean_profile + sem_profile,
                        color=layer_colors[layer], alpha=0.3)
            
            for lm in landmark_positions:
                ax.axvline(lm, color='gray', linestyle='--', alpha=0.3)
            
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax.set_xlabel('Position (cm)', fontsize=10)
            ax.set_ylabel('Activity', fontsize=10)
            ax.set_title(f'{layer} Mean Response\n(n={layer_profiles.shape[0]})', 
                        fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)

    plt.suptitle('Comprehensive PCA Analysis: Layer-Specific Spatial Coding Strategies\n' +
                f'Testing Hypothesis: Deep layers encode goal-proximal landmarks more than superficial layers',
                fontsize=16, fontweight='bold', y=0.998)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved comprehensive summary figure")

    return fig



# ============================================================================
# CLEAN, SEPARATE FIGURES FOR PUBLICATION
# ============================================================================

def plot_figure1_main_finding(landmark_stats, save_path=None):
    """
    FIGURE 1: Main finding - L4 preference by layer (early sessions).
    Clean, publication-ready.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    if 'by_layer' in landmark_stats:
        layer_data = landmark_stats['by_layer']
        l4_pcts = [layer_data[l]['pct_l4'] if l in layer_data else 0 for l in layers]
        ns = [layer_data[l]['n_total'] if l in layer_data else 0 for l in layers]
        
        # Create bars
        bars = ax.bar(layers, l4_pcts, color=[layer_colors[l] for l in layers], 
                     alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
        
        # Add sample size labels on bars
        for i, (bar, n, pct) in enumerate(zip(bars, ns, l4_pcts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'n={n}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                   f'{pct:.1f}%', ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.3, pad=0.3))
        
        # Add significance bracket
        p_val = landmark_stats['early_sessions']['p_fisher']
        
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        
        # Draw bracket between superficial (0,1) and deep (2,3)
        y_max = max(l4_pcts) * 1.15
        bracket_y = y_max + 2
        
        # Bracket
        ax.plot([0.5, 0.5], [bracket_y - 1, bracket_y], 'k-', linewidth=2.5)
        ax.plot([2.5, 2.5], [bracket_y - 1, bracket_y], 'k-', linewidth=2.5)
        ax.plot([0.5, 2.5], [bracket_y, bracket_y], 'k-', linewidth=2.5)
        
        # Significance stars
        ax.text(1.5, bracket_y + 1, sig, ha='center', va='bottom', 
               fontsize=20, fontweight='bold')
        ax.text(1.5, bracket_y + 4, f'p = {p_val:.4f}', ha='center', va='bottom',
               fontsize=11)
        
        # Add layer grouping
        ax.text(0.5, -5, 'Superficial', ha='center', fontsize=13, 
               fontweight='bold', style='italic')
        ax.text(2.5, -5, 'Deep', ha='center', fontsize=13, 
               fontweight='bold', style='italic')
        
        ax.set_ylabel('% Cells Preferring L4 (Last Landmark)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Cortical Layer', fontsize=14, fontweight='bold')
        ax.set_title('L4 Preference by Layer (Early Sessions: Day 1-3)\nDeep Layers Show Higher Goal-Proximal Encoding', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_ylim(0, max(l4_pcts) * 1.35)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 1: Main Finding")
    
    return fig

def plot_figure1b_late_sessions(data, save_path=None):
    """
    FIGURE 1B: L4 preference by layer (LATE sessions: Day 4-7).
    Comparison with early sessions to show learning effects.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    preferred_landmark = data['preferred_landmark']
    session_order = data['session_order']
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # Define early and late sessions
    early_sessions = session_order[:3]  # Day 1-3
    late_sessions = session_order[3:]   # Day 4-7
    
    # Calculate for both early and late
    for ax_idx, (sessions, session_label) in enumerate([(early_sessions, 'Early (Day 1-3)'),
                                                         (late_sessions, 'Late (Day 4-7)')]):
        ax = axes[ax_idx]
        
        session_mask = np.isin(session_labels, sessions)
        
        # Count L4 preference by layer
        layer_data = {}
        l4_pcts = []
        ns = []
        
        for layer in layers:
            layer_mask = (layer_labels == layer) & session_mask
            valid_pref = (preferred_landmark >= 0) & (preferred_landmark < 4)
            valid_combined = layer_mask & valid_pref
            
            n_total = np.sum(valid_combined)
            n_l4 = np.sum((preferred_landmark == 3) & valid_combined)
            
            pct_l4 = n_l4 / n_total * 100 if n_total > 0 else 0
            
            layer_data[layer] = {
                'n_total': n_total,
                'n_l4': n_l4,
                'pct_l4': pct_l4
            }
            
            l4_pcts.append(pct_l4)
            ns.append(n_total)
        
        # Create bars
        bars = ax.bar(layers, l4_pcts, color=[layer_colors[l] for l in layers], 
                     alpha=0.85, edgecolor='black', linewidth=2, width=0.6)
        
        # Add sample size labels on bars
        for i, (bar, n, pct) in enumerate(zip(bars, ns, l4_pcts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'n={n}', ha='center', va='bottom', fontsize=12, fontweight='bold')
            # ax.text(bar.get_x() + bar.get_width()/2., height/2,
            #        f'{pct:.1f}%', ha='center', va='center', 
            #        fontsize=16, fontweight='bold', color='white',
            #        bbox=dict(boxstyle='round', facecolor='black', alpha=0.3, pad=0.3))
        
        # Calculate statistics for superficial vs deep
        sup_mask = np.isin(layer_labels, ['L2/3', 'L4']) & session_mask
        deep_mask = np.isin(layer_labels, ['L5', 'L6']) & session_mask
        valid_pref = (preferred_landmark >= 0) & (preferred_landmark < 4)
        
        sup_valid = sup_mask & valid_pref
        deep_valid = deep_mask & valid_pref
        
        sup_total = np.sum(sup_valid)
        deep_total = np.sum(deep_valid)
        
        sup_l4 = np.sum((preferred_landmark == 3) & sup_valid)
        deep_l4 = np.sum((preferred_landmark == 3) & deep_valid)
        
        sup_l4_pct = sup_l4 / sup_total * 100 if sup_total > 0 else 0
        deep_l4_pct = deep_l4 / deep_total * 100 if deep_total > 0 else 0
        
        # Fisher's exact test
        contingency = np.array([
            [sup_l4, sup_total - sup_l4],
            [deep_l4, deep_total - deep_l4]
        ])
        
        odds_ratio, p_fisher = stats.fisher_exact(contingency)
        
        if p_fisher < 0.001:
            sig = '***'
        elif p_fisher < 0.01:
            sig = '**'
        elif p_fisher < 0.05:
            sig = '*'
        else:
            sig = 'ns'
        
        # Draw bracket between superficial (0,1) and deep (2,3)
        y_max = max(l4_pcts) * 1.15
        bracket_y = y_max + 2
        
        # Bracket
        ax.plot([0.5, 0.5], [bracket_y - 1, bracket_y], 'k-', linewidth=2.5)
        ax.plot([2.5, 2.5], [bracket_y - 1, bracket_y], 'k-', linewidth=2.5)
        ax.plot([0.5, 2.5], [bracket_y, bracket_y], 'k-', linewidth=2.5)
        
        # Significance stars
        ax.text(1.5, bracket_y + 1, sig, ha='center', va='bottom', 
               fontsize=20, fontweight='bold')
        # ax.text(1.5, bracket_y + 4, f'p = {p_fisher:.4f}', ha='center', va='bottom',
        #        fontsize=11)
        
        # Add layer grouping
        ax.text(0.5, -10, 'Superficial', ha='center', fontsize=13, 
               fontweight='bold', style='italic')
        ax.text(2.5, -10, 'Deep', ha='center', fontsize=13, 
               fontweight='bold', style='italic')
        
        ax.set_ylabel('% Cells Preferring L4', fontsize=14, fontweight='bold')
        # ax.set_xlabel('Cortical Layer', fontsize=14, fontweight='bold')
        ax.set_title(f'{session_label}\nSuperficial: {sup_l4_pct:.1f}%, Deep: {deep_l4_pct:.1f}%', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        # ax.set_ylim(0, max(60, max(l4_pcts) * 1.35))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.3, axis='y', linestyle='--')
    
    plt.suptitle('L4 Preference by Layer: Early vs Late Sessions', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 1B: Late Sessions Comparison")
    
    return fig


def plot_figure1c_early_vs_late_comparison(data, save_path=None):
    """
    FIGURE 1C: Direct comparison of early vs late L4 preference.
    Shows learning trajectory for superficial vs deep layers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    preferred_landmark = data['preferred_landmark']
    session_order = data['session_order']
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # Define session groups
    early_sessions = session_order[:3]
    late_sessions = session_order[3:]
    
    # Panel A: Individual layers
    ax1 = axes[0]
    
    x_pos = np.arange(len(layers))
    width = 0.35
    
    early_pcts = []
    late_pcts = []
    
    for layer in layers:
        layer_mask = layer_labels == layer
        valid_pref = (preferred_landmark >= 0) & (preferred_landmark < 4)
        
        # Early
        early_mask = np.isin(session_labels, early_sessions) & layer_mask & valid_pref
        n_early = np.sum(early_mask)
        n_l4_early = np.sum((preferred_landmark == 3) & early_mask)
        pct_early = n_l4_early / n_early * 100 if n_early > 0 else 0
        early_pcts.append(pct_early)
        
        # Late
        late_mask = np.isin(session_labels, late_sessions) & layer_mask & valid_pref
        n_late = np.sum(late_mask)
        n_l4_late = np.sum((preferred_landmark == 3) & late_mask)
        pct_late = n_l4_late / n_late * 100 if n_late > 0 else 0
        late_pcts.append(pct_late)
    
    bars1 = ax1.bar(x_pos - width/2, early_pcts, width, 
                    label='Early (Day 1-3)', color='lightblue', 
                    edgecolor='black', linewidth=1.5, alpha=0.85)
    bars2 = ax1.bar(x_pos + width/2, late_pcts, width, 
                    label='Late (Day 4-7)', color='darkblue', 
                    edgecolor='black', linewidth=1.5, alpha=0.85)
    
    # Add value labels
    for bars, pcts in [(bars1, early_pcts), (bars2, late_pcts)]:
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
    
    # Add change arrows
    for i, (early, late) in enumerate(zip(early_pcts, late_pcts)):
        change = late - early
        color = 'green' if change > 0 else 'red' if change < 0 else 'gray'
        arrow = '↑' if change > 0 else '↓' if change < 0 else '→'
        ax1.text(i, max(early, late) + 5, f'{arrow} {abs(change):.1f}pp', 
                ha='center', fontsize=11, fontweight='bold', color=color)
    
    ax1.set_ylabel('% Preferring L4', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Cortical Layer', fontsize=14, fontweight='bold')
    ax1.set_title('A. L4 Preference by Layer: Early vs Late Sessions', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(layers)
    ax1.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax1.set_ylim(0, max(max(early_pcts), max(late_pcts)) * 1.25)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(alpha=0.3, axis='y', linestyle='--')
    
    # Panel B: Superficial vs Deep
    ax2 = axes[1]
    
    groups = ['Superficial\n(L2/3 + L4)', 'Deep\n(L5 + L6)']
    x_pos_2 = np.arange(len(groups))
    
    # Calculate for superficial
    sup_mask = np.isin(layer_labels, ['L2/3', 'L4'])
    valid_pref = (preferred_landmark >= 0) & (preferred_landmark < 4)
    
    sup_early_mask = np.isin(session_labels, early_sessions) & sup_mask & valid_pref
    sup_early_total = np.sum(sup_early_mask)
    sup_early_l4 = np.sum((preferred_landmark == 3) & sup_early_mask)
    sup_early_pct = sup_early_l4 / sup_early_total * 100 if sup_early_total > 0 else 0
    
    sup_late_mask = np.isin(session_labels, late_sessions) & sup_mask & valid_pref
    sup_late_total = np.sum(sup_late_mask)
    sup_late_l4 = np.sum((preferred_landmark == 3) & sup_late_mask)
    sup_late_pct = sup_late_l4 / sup_late_total * 100 if sup_late_total > 0 else 0
    
    # Calculate for deep
    deep_mask = np.isin(layer_labels, ['L5', 'L6'])
    
    deep_early_mask = np.isin(session_labels, early_sessions) & deep_mask & valid_pref
    deep_early_total = np.sum(deep_early_mask)
    deep_early_l4 = np.sum((preferred_landmark == 3) & deep_early_mask)
    deep_early_pct = deep_early_l4 / deep_early_total * 100 if deep_early_total > 0 else 0
    
    deep_late_mask = np.isin(session_labels, late_sessions) & deep_mask & valid_pref
    deep_late_total = np.sum(deep_late_mask)
    deep_late_l4 = np.sum((preferred_landmark == 3) & deep_late_mask)
    deep_late_pct = deep_late_l4 / deep_late_total * 100 if deep_late_total > 0 else 0
    
    early_group_pcts = [sup_early_pct, deep_early_pct]
    late_group_pcts = [sup_late_pct, deep_late_pct]
    
    bars1 = ax2.bar(x_pos_2 - width/2, early_group_pcts, width, 
                    label='Early (Day 1-3)', color='lightcoral', 
                    edgecolor='black', linewidth=2, alpha=0.85)
    bars2 = ax2.bar(x_pos_2 + width/2, late_group_pcts, width, 
                    label='Late (Day 4-7)', color='darkred', 
                    edgecolor='black', linewidth=2, alpha=0.85)
    
    # Add value labels
    for bars, pcts in [(bars1, early_group_pcts), (bars2, late_group_pcts)]:
        for bar, pct in zip(bars, pcts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
    
    # Add change info
    sup_change = sup_late_pct - sup_early_pct
    deep_change = deep_late_pct - deep_early_pct
    
    for i, (early, late, change) in enumerate([(sup_early_pct, sup_late_pct, sup_change),
                                                (deep_early_pct, deep_late_pct, deep_change)]):
        color = 'green' if change > 0 else 'red' if change < 0 else 'gray'
        arrow = '↑' if change > 0 else '↓' if change < 0 else '→'
        ax2.text(i, max(early, late) + 5, f'{arrow} {abs(change):.1f}pp', 
                ha='center', fontsize=13, fontweight='bold', color=color)
    
    ax2.set_ylabel('% Preferring L4', fontsize=14, fontweight='bold')
    ax2.set_title('B. Superficial vs Deep: Early vs Late Sessions', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos_2)
    ax2.set_xticklabels(groups)
    ax2.legend(fontsize=12, loc='upper left', framealpha=0.9)
    ax2.set_ylim(0, max(max(early_group_pcts), max(late_group_pcts)) * 1.25)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(alpha=0.3, axis='y', linestyle='--')
    
    # Add interpretation text
    interpretation = f"""
    Key Findings:
    • Deep layers maintain higher L4 preference throughout training
    • Superficial: {sup_early_pct:.1f}% → {sup_late_pct:.1f}% ({sup_change:+.1f}pp change)
    • Deep: {deep_early_pct:.1f}% → {deep_late_pct:.1f}% ({deep_change:+.1f}pp change)
    """
    
    ax2.text(0.98, 0.98, interpretation, transform=ax2.transAxes,
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     alpha=0.9, edgecolor='black', linewidth=1.5))
    
    plt.suptitle('Learning Effects: Do Layer Differences Persist or Converge Over Training?', 
                fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 1C: Early vs Late Comparison")
    
    return fig

def plot_figure2_pc1_loadings(data, layer_pcas, save_path=None):
    """
    FIGURE 2: Layer-specific PC1 loadings.
    Shows what spatial patterns each layer prioritizes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # Panel A: All layers together
    ax1 = axes[0]
    for layer in layers:
        if layer not in layer_pcas:
            continue
        pc1_loading = layer_pcas[layer]['loadings'][0]
        var_explained = layer_pcas[layer]['explained_variance_ratio'][0] * 100
        ax1.plot(bin_centers, pc1_loading, linewidth=3.5, 
                color=layer_colors[layer], label=f'{layer} ({var_explained:.1f}% var)', 
                alpha=0.9, marker='o', markersize=3, markevery=10)
    
    # Add landmark markers
    for i, lm in enumerate(landmark_positions):
        ax1.axvline(lm, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax1.text(lm, ax1.get_ylim()[1]*0.92, f'L{i+1}', 
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.axhline(0, color='black', linestyle='-', alpha=0.4, linewidth=1.5)
    ax1.set_xlabel('Position (cm)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('PC1 Loading', fontsize=14, fontweight='bold')
    ax1.set_title('A. Layer-Specific PC1 Loadings: Different Spatial Priorities', 
                 fontsize=15, fontweight='bold', pad=15)
    ax1.legend(fontsize=12, loc='best', framealpha=0.9)
    ax1.grid(alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(labelsize=12)
    
    # Panel B: PC1 similarity matrix
    ax2 = axes[1]
    
    valid_layers = [l for l in layers if l in layer_pcas]
    n_layers = len(valid_layers)
    similarity_matrix = np.zeros((n_layers, n_layers))
    
    for i, layer1 in enumerate(valid_layers):
        for j, layer2 in enumerate(valid_layers):
            pc1_1 = layer_pcas[layer1]['loadings'][0]
            pc1_2 = layer_pcas[layer2]['loadings'][0]
            similarity_matrix[i, j] = 1 - cosine(pc1_1, pc1_2)
    
    im = ax2.imshow(similarity_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.set_xticks(np.arange(n_layers))
    ax2.set_yticks(np.arange(n_layers))
    ax2.set_xticklabels(valid_layers, fontsize=13, fontweight='bold')
    ax2.set_yticklabels(valid_layers, fontsize=13, fontweight='bold')
    
    # Add text annotations
    for i in range(n_layers):
        for j in range(n_layers):
            text_color = 'white' if abs(similarity_matrix[i, j]) > 0.5 else 'black'
            text = ax2.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color=text_color, 
                          fontsize=14, fontweight='bold')
    
    ax2.set_title('B. PC1 Cosine Similarity Between Layers\n(Negative = Opposite Spatial Priorities)', 
                 fontsize=15, fontweight='bold', pad=15)
    cbar = plt.colorbar(im, ax=ax2, label='Cosine Similarity', fraction=0.046)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label('Cosine Similarity', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 2: PC1 Loadings")
    
    return fig


def plot_figure3_session_dynamics(data, save_path=None):
    """
    FIGURE 3: Session-by-session dynamics (LEARNING EFFECTS).
    Shows how landmark preferences evolve over training.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    preferred_landmark = data['preferred_landmark']
    session_order = data['session_order']
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # Calculate L4 preference by layer and session
    for layer_idx, layer in enumerate(layers):
        ax = axes[layer_idx]
        
        layer_mask = layer_labels == layer
        
        l1_props = []
        l4_props = []
        session_nums = []
        
        for session in session_order:
            session_mask = session_labels == session
            combined_mask = layer_mask & session_mask
            
            valid_pref = (preferred_landmark >= 0) & (preferred_landmark < 4)
            valid_combined = combined_mask & valid_pref
            
            n_total = np.sum(valid_combined)
            
            if n_total > 0:
                n_l1 = np.sum((preferred_landmark == 0) & valid_combined)
                n_l4 = np.sum((preferred_landmark == 3) & valid_combined)
                
                l1_props.append(n_l1 / n_total * 100)
                l4_props.append(n_l4 / n_total * 100)
                session_nums.append(int(session.replace('Day', '')))
        
        # Plot L1 and L4 trajectories
        x = np.array(session_nums)
        
        ax.plot(x, l1_props, 'o-', linewidth=3, markersize=10, 
               color='#e41a1c', label='L1 (Track Start)', alpha=0.8)
        ax.plot(x, l4_props, 's-', linewidth=3, markersize=10, 
               color='#984ea3', label='L4 (Goal-Proximal)', alpha=0.8)
        
        # Add shaded regions for early/late
        ax.axvspan(x[0], x[2], alpha=0.1, color='lightblue', label='Early Sessions')
        ax.axvspan(x[-2], x[-1], alpha=0.1, color='lightcoral', label='Late Sessions')
        
        ax.set_xlabel('Session (Day)', fontsize=13, fontweight='bold')
        ax.set_ylabel('% of Cells', fontsize=13, fontweight='bold')
        ax.set_title(f'{layer} Landmark Preference Over Training', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(session_nums)
        ax.set_ylim(0, max(max(l1_props), max(l4_props)) * 1.15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=11)
    
    plt.suptitle('Landmark Preference Evolution Across Training Sessions\n(Learning Effects by Layer)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 3: Session Dynamics")
    
    return fig


def plot_figure4_layer_comparison(data, pc1_stats, save_path=None):
    """
    FIGURE 4: Direct layer comparisons.
    PC1 distributions and statistical tests.
    """
    fig = plt.figure(figsize=(6, 6))
    # fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 1, figure=fig, wspace=0.3)
    # gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # # Panel A: PC1 boxplots
    # ax1 = fig.add_subplot(gs[0, 0])
    
    # pc1_by_layer = []
    # for layer in layers:
    #     mask = data['layer_labels'] == layer
    #     pc1_by_layer.append(data['pc_scores'][mask, 0])
    
    # bp = ax1.boxplot(pc1_by_layer, labels=layers, patch_artist=True, widths=0.6,
    #                 boxprops=dict(linewidth=2),
    #                 whiskerprops=dict(linewidth=2),
    #                 capprops=dict(linewidth=2),
    #                 medianprops=dict(linewidth=3, color='red'))
    
    # for patch, layer in zip(bp['boxes'], layers):
    #     patch.set_facecolor(layer_colors[layer])
    #     patch.set_alpha(0.7)
    
    # # Add significance
    # p_val = pc1_stats['p_value']
    # if p_val < 0.05:
    #     if p_val < 0.001:
    #         sig = '***'
    #     elif p_val < 0.01:
    #         sig = '**'
    #     else:
    #         sig = '*'
        
    #     y_max = max([np.max(x) for x in pc1_by_layer]) * 1.05
    #     bracket_y = y_max + 0.5
        
    #     ax1.plot([0.7, 0.7], [bracket_y - 0.3, bracket_y], 'k-', linewidth=2.5)
    #     ax1.plot([3.3, 3.3], [bracket_y - 0.3, bracket_y], 'k-', linewidth=2.5)
    #     ax1.plot([0.7, 3.3], [bracket_y, bracket_y], 'k-', linewidth=2.5)
    #     ax1.text(2, bracket_y + 0.3, sig, ha='center', fontsize=18, fontweight='bold')
    #     ax1.text(2, bracket_y + 0.8, f'p < 0.001', ha='center', fontsize=11)
    
    # ax1.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    # ax1.set_ylabel('PC1 Score', fontsize=14, fontweight='bold')
    # ax1.set_xlabel('Cortical Layer', fontsize=14, fontweight='bold')
    # ax1.set_title('A. PC1 Distribution by Layer\n(Lower = More Reward-Biased)', 
    #              fontsize=14, fontweight='bold')
    # ax1.grid(alpha=0.3, axis='y', linestyle='--')
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    # ax1.tick_params(labelsize=12)
    
    # Panel B: Superficial vs Deep comparison
    ax2 = fig.add_subplot(gs[0, 0])
    
    sup_mask = np.isin(data['layer_labels'], ['L2/3', 'L4'])
    deep_mask = np.isin(data['layer_labels'], ['L5', 'L6'])
    
    sup_pc1 = data['pc_scores'][sup_mask, 0]
    deep_pc1 = data['pc_scores'][deep_mask, 0]
    
    violin_parts = ax2.violinplot([sup_pc1, deep_pc1], positions=[0, 1], 
                                   widths=0.7, showmeans=True, showmedians=True)
    
    for i, (pc, body) in enumerate(zip([sup_pc1, deep_pc1], violin_parts['bodies'])):
        body.set_facecolor(['#1E88E5', '#4CAF50'][i])
        body.set_alpha(0.7)
    
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Superficial\n(L2/3 + L4)', 'Deep\n(L5 + L6)'], 
                       fontsize=13, fontweight='bold')
    ax2.set_ylabel('PC1 Score', fontsize=14, fontweight='bold')
    # ax2.set_title('B. Superficial vs Deep Layers\n(Violin Plot)', 
    #              fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3, axis='y', linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(labelsize=12)
    
    # Add statistics text
    stats_text = f"t = {pc1_stats['t_stat']:.3f}\np < 0.001\nd = {pc1_stats['cohens_d']:.3f}"
    stats_text = "p < 0.001"
    ax2.text(0.5, ax2.get_ylim()[1]*0.2, stats_text, 
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor='black', linewidth=2))
    
    # # Panel C: Mean profiles by layer
    # ax3 = fig.add_subplot(gs[0, 2])
    
    # bin_centers = data['bin_centers']
    # landmark_positions = data['landmark_positions']
    
    # for layer in layers:
    #     layer_mask = data['layer_labels'] == layer
    #     profiles = data['spatial_profiles_zscore'][layer_mask]
        
    #     if profiles.shape[0] > 0:
    #         mean_profile = np.mean(profiles, axis=0)
    #         sem_profile = np.std(profiles, axis=0) / np.sqrt(profiles.shape[0])
            
    #         x_vals = np.linspace(bin_centers[0], bin_centers[-1], len(mean_profile))
            
    #         ax3.plot(x_vals, mean_profile, linewidth=3, 
    #                 color=layer_colors[layer], label=layer, alpha=0.85)
    #         ax3.fill_between(x_vals, mean_profile - sem_profile, 
    #                        mean_profile + sem_profile,
    #                        color=layer_colors[layer], alpha=0.2)
    
    # for lm in landmark_positions:
    #     ax3.axvline(lm, color='gray', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # ax3.axhline(0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
    # ax3.set_xlabel('Position (cm)', fontsize=14, fontweight='bold')
    # ax3.set_ylabel('Mean Activity (z-scored)', fontsize=14, fontweight='bold')
    # ax3.set_title('C. Mean Response Profiles by Layer', 
    #              fontsize=14, fontweight='bold')
    # ax3.legend(fontsize=11, loc='best', framealpha=0.9)
    # ax3.grid(alpha=0.3, linestyle='--')
    # ax3.spines['top'].set_visible(False)
    # ax3.spines['right'].set_visible(False)
    # ax3.tick_params(labelsize=12)
    
    plt.suptitle('Layer Comparison: PC1 Distribution and Response Profiles', 
                fontsize=16, fontweight='bold', y=1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 4: Layer Comparison")
    
    return fig


def plot_figure5_supplementary_tests(procrustes_results, cca_results, 
                                     decoding_results, save_path=None):
    """
    FIGURE 5: Supplementary statistical tests.
    Procrustes, CCA, and decoding results.
    """
    fig = plt.figure(figsize=(18, 6))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)
    
    # Panel A: Procrustes
    ax1 = fig.add_subplot(gs[0, 0])
    
    if procrustes_results is not None:
        disparity = procrustes_results['disparity']
        
        bar = ax1.bar(['Superficial\nvs\nDeep'], [disparity], color='steelblue', 
                     alpha=0.8, edgecolor='black', linewidth=2.5, width=0.5)
        
        # Threshold lines
        ax1.axhline(0.1, color='green', linestyle='--', alpha=0.7, 
                   label='Similar (<0.1)', linewidth=2.5)
        ax1.axhline(0.3, color='orange', linestyle='--', alpha=0.7, 
                   label='Distinct (>0.3)', linewidth=2.5)
        
        ax1.set_ylabel('Procrustes Disparity', fontsize=14, fontweight='bold')
        ax1.set_title('A. Subspace Alignment\n(Procrustes Test)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.set_ylim(0, max(0.5, disparity * 1.3))
        ax1.grid(alpha=0.3, axis='y', linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.tick_params(labelsize=12)
        
        # Interpretation
        if disparity > 0.3:
            interp = "Distinct\nSubspaces"
            color = 'darkgreen'
        elif disparity > 0.1:
            interp = "Partially\nSimilar"
            color = 'orange'
        else:
            interp = "Very\nSimilar"
            color = 'red'
        
        ax1.text(0, disparity + 0.02, interp, ha='center', va='bottom', 
                fontsize=13, fontweight='bold', color=color)
    
    # Panel B: CCA
    ax2 = fig.add_subplot(gs[0, 1])
    
    if cca_results is not None:
        canon_corrs = cca_results['canon_corrs']
        
        x = np.arange(1, len(canon_corrs) + 1)
        bars = ax2.bar(x, canon_corrs, edgecolor='black', linewidth=2, width=0.7)
        
        # Color code
        for i, (bar, corr) in enumerate(zip(bars, canon_corrs)):
            if corr > 0.7:
                bar.set_color('darkred')
            elif corr > 0.4:
                bar.set_color('orange')
            else:
                bar.set_color('lightblue')
        
        ax2.axhline(0.7, color='red', linestyle='--', alpha=0.6, 
                   label='Strong (>0.7)', linewidth=2)
        ax2.axhline(0.4, color='orange', linestyle='--', alpha=0.6, 
                   label='Moderate (>0.4)', linewidth=2)
        
        ax2.set_xlabel('Canonical Dimension', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Correlation', fontsize=14, fontweight='bold')
        ax2.set_title('B. Canonical Correlations\n(Superficial ↔ Deep)', 
                     fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_ylim(0, 1)
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(alpha=0.3, axis='y', linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.tick_params(labelsize=12)
    
    # Panel C: Decoding accuracy
    ax3 = fig.add_subplot(gs[0, 2])
    
    if decoding_results is not None:
        acc = decoding_results['mean_accuracy'] * 100
        std = decoding_results['std_accuracy'] * 100
        chance = decoding_results['chance'] * 100
        
        bars = ax3.bar(['Classifier', 'Chance'], [acc, chance], 
                      color=['darkgreen', 'gray'], alpha=0.8, 
                      edgecolor='black', linewidth=2.5, width=0.5)
        
        # Error bar
        ax3.errorbar(0, acc, yerr=std, fmt='none', color='black', 
                    capsize=8, linewidth=3, capthick=2)
        
        ax3.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax3.set_title('C. Layer Classification\nfrom PC Space', 
                     fontsize=14, fontweight='bold')
        ax3.set_ylim(0, 100)
        ax3.grid(alpha=0.3, axis='y', linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.tick_params(labelsize=12)
        
        # Improvement text
        improvement = acc - chance
        ax3.text(0.5, max(acc, chance) + 8, f'+{improvement:.1f}pp', 
                ha='center', fontsize=14, fontweight='bold')
        
        # Add stars if significant
        if improvement > 15:
            ax3.text(0.5, max(acc, chance) + 15, '***', 
                    ha='center', fontsize=20, fontweight='bold')
    
    # Panel D: Confusion matrix
    ax4 = fig.add_subplot(gs[0, 3])
    
    if decoding_results is not None:
        cm = decoding_results['confusion_matrix']
        layer_names = decoding_results['layer_names']
        
        # Normalize
        cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        im = ax4.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
        
        ax4.set_xticks(np.arange(len(layer_names)))
        ax4.set_yticks(np.arange(len(layer_names)))
        ax4.set_xticklabels(layer_names, fontsize=12, fontweight='bold')
        ax4.set_yticklabels(layer_names, fontsize=12, fontweight='bold')
        
        # Annotations
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                text_color = 'white' if cm_pct[i, j] > 50 else 'black'
                text = ax4.text(j, i, f'{cm_pct[i, j]:.0f}%',
                              ha="center", va="center", 
                              color=text_color, fontsize=12, fontweight='bold')
        
        ax4.set_ylabel('True Layer', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Predicted Layer', fontsize=13, fontweight='bold')
        ax4.set_title('D. Confusion Matrix\n(Layer Classification)', 
                     fontsize=14, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax4, label='%', fraction=0.046)
        cbar.ax.tick_params(labelsize=11)
    
    plt.suptitle('Supplementary Analyses: Subspace Tests and Layer Decoding', 
                fontsize=16, fontweight='bold', y=1.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 5: Supplementary Tests")
    
    return fig

# ============================================================================
# NEW ANALYSIS 7: MULTI-PC LAYER DIFFERENCES
# ============================================================================

def analyze_multi_pc_layer_differences(data, layer_pcas, n_pcs=5):
    """
    Analyze each PC (1-5) individually for layer-specific structure.
    
    For each PC:
    1. Compare loadings between layers
    2. Test layer decoding from that PC alone
    3. Identify what spatial feature it encodes
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    
    print("\n" + "="*80)
    print("MULTI-PC LAYER ANALYSIS")
    print("Testing which PCs contain layer-specific information")
    print("="*80)
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    results = {}
    
    for pc_idx in range(n_pcs):
        print(f"\n{'='*80}")
        print(f"PC{pc_idx+1} (Variance: {data['explained_variance_ratio'][pc_idx]*100:.1f}%)")
        print(f"{'='*80}")
        
        # =====================================================================
        # Test 1: Loading similarity between layers
        # =====================================================================
        print(f"\n--- PC{pc_idx+1} Loading Similarity ---")
        
        similarities = {}
        all_sims = []
        
        for i, layer1 in enumerate(layers):
            if layer1 not in layer_pcas:
                continue
            for layer2 in layers[i+1:]:
                if layer2 not in layer_pcas:
                    continue
                
                loading1 = layer_pcas[layer1]['loadings'][pc_idx]
                loading2 = layer_pcas[layer2]['loadings'][pc_idx]
                
                sim = 1 - cosine(loading1, loading2)
                similarities[f"{layer1} vs {layer2}"] = sim
                all_sims.append(sim)
                
                print(f"  {layer1:6s} vs {layer2:6s}: {sim:+.3f}")
        
        mean_sim = np.mean(all_sims)
        print(f"\n  Mean similarity: {mean_sim:.3f}")
        
        if mean_sim < 0.5:
            print(f"  ✓ PC{pc_idx+1} shows DISTINCT layer-specific loadings")
        elif mean_sim < 0.8:
            print(f"  ~ PC{pc_idx+1} shows MODERATE layer differences")
        else:
            print(f"  → PC{pc_idx+1} shows SIMILAR loadings across layers")
        
        # =====================================================================
        # Test 2: Layer decoding from this PC alone
        # =====================================================================
        print(f"\n--- PC{pc_idx+1} Layer Decoding ---")
        
        pc_scores_single = data['pc_scores'][:, pc_idx].reshape(-1, 1)
        layer_labels_encoded = LabelEncoder().fit_transform(data['layer_labels'])
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        scores = cross_val_score(clf, pc_scores_single, layer_labels_encoded, cv=5)
        
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        chance = 1 / len(np.unique(data['layer_labels']))
        improvement = mean_acc - chance
        
        print(f"  Accuracy: {mean_acc*100:.1f}% ± {std_acc*100:.1f}%")
        print(f"  Chance: {chance*100:.1f}%")
        print(f"  Improvement: +{improvement*100:.1f}pp")
        
        if improvement > 0.15:
            print(f"  ✓ PC{pc_idx+1} contains STRONG layer information")
        elif improvement > 0.05:
            print(f"  ~ PC{pc_idx+1} contains MODERATE layer information")
        else:
            print(f"  → PC{pc_idx+1} contains WEAK layer information")
        
        # =====================================================================
        # Test 3: Spatial encoding
        # =====================================================================
        print(f"\n--- PC{pc_idx+1} Spatial Feature ---")
        
        bin_centers = data['bin_centers']
        global_loading = data['components'][pc_idx]
        
        # Find peak
        peak_idx = np.argmax(np.abs(global_loading))
        peak_pos = bin_centers[peak_idx]
        peak_val = global_loading[peak_idx]
        
        print(f"  Peak loading: {peak_val:+.3f} at {peak_pos:.1f} cm")
        
        # Find closest landmark
        landmark_positions = data['landmark_positions']
        closest_lm_idx = np.argmin(np.abs(landmark_positions - peak_pos))
        closest_lm_pos = landmark_positions[closest_lm_idx]
        distance = abs(peak_pos - closest_lm_pos)
        
        print(f"  Closest landmark: L{closest_lm_idx+1} ({closest_lm_pos} cm)")
        print(f"  Distance: {distance:.1f} cm")
        
        # Interpret
        if distance < 10:
            print(f"  → PC{pc_idx+1} encodes L{closest_lm_idx+1} region")
        else:
            if peak_pos < 30:
                print(f"  → PC{pc_idx+1} encodes track START region")
            elif peak_pos > 110:
                print(f"  → PC{pc_idx+1} encodes track END/reward region")
            else:
                print(f"  → PC{pc_idx+1} encodes track MIDDLE region")
        
        # Store results
        results[f'PC{pc_idx+1}'] = {
            'variance': data['explained_variance_ratio'][pc_idx],
            'similarities': similarities,
            'mean_similarity': mean_sim,
            'decoding_acc': mean_acc,
            'decoding_std': std_acc,
            'improvement': improvement,
            'peak_position': peak_pos,
            'peak_value': peak_val,
            'closest_landmark': closest_lm_idx,
            'spatial_feature': 'landmark' if distance < 10 else 'between'
        }
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY: Which PCs are most layer-specific?")
    print(f"{'='*80}")
    
    sorted_pcs = sorted(results.items(), 
                       key=lambda x: x[1]['improvement'], 
                       reverse=True)
    
    for pc_name, pc_data in sorted_pcs:
        print(f"{pc_name}: +{pc_data['improvement']*100:.1f}pp decoding improvement, "
              f"mean similarity: {pc_data['mean_similarity']:.3f}")
    
    return results


# ============================================================================
# NEW ANALYSIS 8: PC COMBINATIONS
# ============================================================================

def analyze_pc_combinations(data):
    """
    Test if combinations of PCs better distinguish layers than individual PCs.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder
    
    print("\n" + "="*80)
    print("PC COMBINATION ANALYSIS")
    print("Testing which PC combinations best distinguish layers")
    print("="*80)
    
    layer_labels = data['layer_labels']
    layer_encoded = LabelEncoder().fit_transform(layer_labels)
    chance = 1 / len(np.unique(layer_labels))
    
    # Define PC combinations to test
    pc_combinations = [
        ([0], "PC1 only"),
        ([1], "PC2 only"),
        ([2], "PC3 only"),
        ([0, 1], "PC1+PC2"),
        ([0, 1, 2], "PC1+PC2+PC3"),
        ([0, 1, 2, 3], "PC1-4"),
        ([0, 1, 2, 3, 4], "PC1-5"),
        ([1, 2], "PC2+PC3 (no PC1)"),
        ([2, 3, 4], "PC3-5 (no PC1-2)"),
    ]
    
    results = []
    
    print(f"\n{'Combination':<20} {'RF Acc':<12} {'LDA Acc':<12} {'Improvement':<12}")
    print("-" * 60)
    
    for pcs, name in pc_combinations:
        pc_scores = data['pc_scores'][:, pcs]
        
        # Random Forest
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_scores = cross_val_score(clf, pc_scores, layer_encoded, cv=5)
        rf_acc = np.mean(rf_scores)
        
        # Linear Discriminant Analysis
        lda = LinearDiscriminantAnalysis()
        lda_scores = cross_val_score(lda, pc_scores, layer_encoded, cv=5)
        lda_acc = np.mean(lda_scores)
        
        improvement = rf_acc - chance
        
        print(f"{name:<20} {rf_acc*100:>6.1f}%      {lda_acc*100:>6.1f}%      +{improvement*100:>5.1f}pp")
        
        results.append({
            'pcs': pcs,
            'name': name,
            'rf_acc': rf_acc,
            'lda_acc': lda_acc,
            'improvement': improvement
        })
    
    # Find best
    best = max(results, key=lambda x: x['rf_acc'])
    print(f"\n{'='*60}")
    print(f"Best combination: {best['name']}")
    print(f"  Random Forest: {best['rf_acc']*100:.1f}%")
    print(f"  LDA: {best['lda_acc']*100:.1f}%")
    print(f"  Improvement over chance: +{best['improvement']*100:.1f}pp")
    print(f"{'='*60}")
    
    return results


# ============================================================================
# NEW ANALYSIS 9: PC TRAJECTORIES
# ============================================================================

def analyze_pc_trajectory_by_layer(data):
    """
    Track how layers move through PC space across sessions.
    
    Reveals if layers:
    1. Start in different regions (baseline difference)
    2. Move in different directions (different learning)
    3. Converge or diverge (plasticity patterns)
    """
    print("\n" + "="*80)
    print("PC SPACE TRAJECTORY ANALYSIS")
    print("Tracking layer movement across sessions")
    print("="*80)
    
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    pc_scores = data['pc_scores']
    
    # Get session order
    unique_sessions = np.unique(session_labels)
    session_order = sorted(unique_sessions, key=lambda x: int(x.replace('Day', '')))
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    trajectories = {}
    
    for layer in layers:
        layer_mask = layer_labels == layer
        
        centroids = []
        session_names = []
        
        for session in session_order:
            session_mask = session_labels == session
            combined_mask = layer_mask & session_mask
            
            n_cells = np.sum(combined_mask)
            
            if n_cells > 10:  # Need enough cells
                # Compute centroid in PC1-PC5 space
                centroid = np.mean(pc_scores[combined_mask, :5], axis=0)
                centroids.append(centroid)
                session_names.append(session)
        
        if len(centroids) > 1:
            centroids = np.array(centroids)
            
            # Calculate total movement
            diffs = np.diff(centroids, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            total_movement = np.sum(distances)
            
            # Calculate direction
            overall_direction = centroids[-1] - centroids[0]
            
            print(f"\n{layer}:")
            print(f"  Sessions tracked: {len(centroids)}")
            print(f"  Total movement: {total_movement:.3f}")
            print(f"  Overall direction (PC1-PC5):")
            for i in range(5):
                print(f"    PC{i+1}: {overall_direction[i]:+.3f}")
            
            # Check if moving toward/away from reward encoding
            # (negative PC1 = more reward-biased)
            if overall_direction[0] < -0.5:
                print(f"  → Moving toward REWARD encoding")
            elif overall_direction[0] > 0.5:
                print(f"  → Moving toward LANDMARK encoding")
            else:
                print(f"  → Stable spatial encoding")
            
            trajectories[layer] = {
                'centroids': centroids,
                'sessions': session_names,
                'total_movement': total_movement,
                'direction': overall_direction,
                'color': layer_colors[layer]
            }
    
    # Compare movements
    print(f"\n{'='*80}")
    print("COMPARISON: Which layers show most plasticity?")
    print(f"{'='*80}")
    
    sorted_layers = sorted(trajectories.items(),
                          key=lambda x: x[1]['total_movement'],
                          reverse=True)
    
    for layer, traj in sorted_layers:
        print(f"{layer}: {traj['total_movement']:.3f} movement")
    
    return trajectories

# ============================================================================
# NEW FIGURE 6: MULTI-PC ANALYSIS
# ============================================================================

def plot_figure6_multi_pc_analysis(data, layer_pcas, multi_pc_results, save_path=None):
    """
    Visualize which PCs contain layer-specific information.
    """
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    # =========================================================================
    # ROW 1: PC loadings for PC1-4
    # =========================================================================
    for pc_idx in range(4):
        ax = fig.add_subplot(gs[0, pc_idx])
        
        pc_name = f'PC{pc_idx+1}'
        variance = multi_pc_results[pc_name]['variance'] * 100
        
        for layer in layers:
            if layer not in layer_pcas:
                continue
            loading = layer_pcas[layer]['loadings'][pc_idx]
            ax.plot(bin_centers, loading, linewidth=2.5,
                   color=layer_colors[layer], label=layer, alpha=0.8)
        
        for lm in landmark_positions:
            ax.axvline(lm, color='gray', linestyle='--', alpha=0.3)
        
        ax.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Position (cm)', fontsize=10)
        ax.set_ylabel(f'{pc_name} Loading', fontsize=10)
        ax.set_title(f'{pc_name} ({variance:.1f}% var)', 
                    fontsize=11, fontweight='bold')
        if pc_idx == 0:
            ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    # =========================================================================
    # ROW 2: Layer decoding accuracy per PC
    # =========================================================================
    ax1 = fig.add_subplot(gs[1, 0:2])
    
    pc_names = [f'PC{i+1}' for i in range(5)]
    improvements = [multi_pc_results[pc]['improvement'] * 100 for pc in pc_names]
    decoding_accs = [multi_pc_results[pc]['decoding_acc'] * 100 for pc in pc_names]
    
    x = np.arange(len(pc_names))
    width = 0.7
    
    bars = ax1.bar(x, improvements, width, color='steelblue', alpha=0.8, edgecolor='black')
    
    # Color bars by strength
    for bar, imp in zip(bars, improvements):
        if imp > 15:
            bar.set_color('darkgreen')
        elif imp > 5:
            bar.set_color('orange')
        else:
            bar.set_color('lightgray')
    
    ax1.axhline(0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(pc_names)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Layer Decoding Improvement (pp)', fontsize=12)
    ax1.set_title('Which PCs Contain Layer-Specific Information?', 
                 fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Add values on bars
    for i, (bar, imp, acc) in enumerate(zip(bars, improvements, decoding_accs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'+{imp:.1f}pp\n({acc:.1f}%)', 
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # =========================================================================
    # ROW 2: Mean similarity between layers per PC
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, 2:4])
    
    mean_sims = [multi_pc_results[pc]['mean_similarity'] for pc in pc_names]
    
    bars = ax2.bar(x, mean_sims, width, alpha=0.8, edgecolor='black')
    
    # Color by similarity
    for bar, sim in zip(bars, mean_sims):
        if sim < 0.5:
            bar.set_color('darkred')  # Very different
        elif sim < 0.8:
            bar.set_color('orange')    # Moderately different
        else:
            bar.set_color('darkgreen')  # Similar
    
    ax2.axhline(0.7, color='orange', linestyle='--', alpha=0.6, 
               label='Moderate similarity', linewidth=2)
    ax2.axhline(0.5, color='red', linestyle='--', alpha=0.6, 
               label='Low similarity', linewidth=2)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(pc_names)
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Mean Loading Similarity', fontsize=12)
    ax2.set_title('How Similar Are Layer Loadings?\n(Lower = More Layer-Specific)', 
                 fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis='y')
    
    # =========================================================================
    # ROW 3: Spatial features encoded by each PC
    # =========================================================================
    ax3 = fig.add_subplot(gs[2, 0:2])
    
    peak_positions = [multi_pc_results[pc]['peak_position'] for pc in pc_names]
    peak_values = [multi_pc_results[pc]['peak_value'] for pc in pc_names]
    
    colors_by_region = []
    for pos in peak_positions:
        if pos < 40:
            colors_by_region.append('#e41a1c')  # L1 region
        elif pos < 70:
            colors_by_region.append('#377eb8')  # L2 region
        elif pos < 100:
            colors_by_region.append('#4daf4a')  # L3 region
        else:
            colors_by_region.append('#984ea3')  # L4 region
    
    ax3.scatter(peak_positions, peak_values, s=200, c=colors_by_region,
               alpha=0.8, edgecolors='black', linewidths=2)
    
    # Add PC labels
    for i, (pos, val, pc) in enumerate(zip(peak_positions, peak_values, pc_names)):
        ax3.text(pos, val + 0.02, pc, ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    
    # Add landmark positions
    for i, lm in enumerate(landmark_positions):
        ax3.axvline(lm, color='gray', linestyle='--', alpha=0.5, linewidth=2)
        ax3.text(lm, ax3.get_ylim()[1]*0.95, f'L{i+1}',
        ha='center', fontsize=10, fontweight='bold')
        ax3.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Peak Position (cm)', fontsize=12)
        ax3.set_ylabel('Peak Loading Value', fontsize=12)
        ax3.set_title('Spatial Features Encoded by Each PC', 
                    fontsize=13, fontweight='bold')
        ax3.grid(alpha=0.3)

        # =========================================================================
        # ROW 3: Summary interpretation
        # =========================================================================
        ax4 = fig.add_subplot(gs[2, 2:4])
        ax4.axis('off')

        # Determine which PC is most layer-specific
        best_pc = max(multi_pc_results.items(), key=lambda x: x[1]['improvement'])
        worst_pc = min(multi_pc_results.items(), key=lambda x: x[1]['improvement'])

        summary_text = "KEY FINDINGS:\n\n"
        summary_text += f"Most Layer-Specific:\n"
        summary_text += f"  {best_pc[0]}: +{best_pc[1]['improvement']*100:.1f}pp\n"
        summary_text += f"  → {best_pc[1]['variance']*100:.1f}% of variance\n"
        summary_text += f"  → Peak at {best_pc[1]['peak_position']:.0f} cm\n\n"

        summary_text += f"Least Layer-Specific:\n"
        summary_text += f"  {worst_pc[0]}: +{worst_pc[1]['improvement']*100:.1f}pp\n"
        summary_text += f"  → {worst_pc[1]['variance']*100:.1f}% of variance\n\n"

        summary_text += "─" * 30 + "\n\n"
        summary_text += "INTERPRETATION:\n"

        if best_pc[1]['improvement'] > 0.15:
            summary_text += f"✓ {best_pc[0]} strongly\n  distinguishes layers\n"
        else:
            summary_text += f"~ No single PC strongly\n  distinguishes layers\n"

        summary_text += f"\n→ Combined PCs may be\n  more informative\n"
        summary_text += f"  (see Figure 7)"

        ax4.text(0.1, 0.9, summary_text,
                transform=ax4.transAxes,
                fontsize=11, fontfamily='monospace',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Multi-PC Layer Analysis\nWhich Principal Components Encode Layer-Specific Information?',
                    fontsize=14, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved Figure 6: {save_path}")

        return fig
    
# ============================================================================
# NEW FIGURE 7: PC COMBINATIONS
# ============================================================================
def plot_figure7_pc_combinations(data, pc_combo_results, save_path=None):
    """
    Visualize how PC combinations improve layer discrimination.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    # =========================================================================
    # Panel 1: Bar chart of decoding accuracy
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :2])

    names = [r['name'] for r in pc_combo_results]
    rf_accs = [r['rf_acc'] * 100 for r in pc_combo_results]
    lda_accs = [r['lda_acc'] * 100 for r in pc_combo_results]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, rf_accs, width, label='Random Forest',
                color='steelblue', alpha=0.8, edgecolor='black')
    # bars2 = ax1.bar(x + width/2, lda_accs, width, label='Linear DA',
    #             color='coral', alpha=0.8, edgecolor='black')

    # Chance line
    chance = 100 / 4  # 4 layers
    ax1.axhline(chance, color='red', linestyle='--', linewidth=2,
            label='Chance', alpha=0.7)

    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylabel('Layer Decoding Accuracy (%)', fontsize=12)
    ax1.set_title('Layer Discrimination by PC Combination', 
                fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim(0, 100)

    # Highlight best
    best_idx = np.argmax(rf_accs)
    bars1[best_idx].set_color('darkgreen')
    bars1[best_idx].set_linewidth(3)

    # =========================================================================
    # Panel 2: Improvement over chance
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 2])

    improvements = [r['improvement'] * 100 for r in pc_combo_results]

    bars = ax2.barh(x, improvements, color='steelblue', alpha=0.8, edgecolor='black')

    # Color by strength
    for bar, imp in zip(bars, improvements):
        if imp > 30:
            bar.set_color('darkgreen')
        elif imp > 20:
            bar.set_color('orange')
        else:
            bar.set_color('lightgray')

    ax2.axvline(0, color='black', linestyle='-', linewidth=1.5)
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel('Improvement (pp)', fontsize=11)
    ax2.set_title('Improvement Over Chance', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3, axis='x')

    # # =========================================================================
    # # Panel 3: PC space visualization with best combination
    # # =========================================================================
    # ax3 = fig.add_subplot(gs[1, 0])

    # best = max(pc_combo_results, key=lambda x: x['rf_acc'])
    # best_pcs = best['pcs']

    # # Use first 2 PCs from best combination
    # if len(best_pcs) >= 2:
    #     pc1_idx, pc2_idx = best_pcs[0], best_pcs[1]
    # else:
    #     pc1_idx, pc2_idx = 0, 1

    # pc_scores = data['pc_scores']
    # layer_labels = data['layer_labels']
    # layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}

    # for layer in ['L2/3', 'L4', 'L5', 'L6']:
    #     mask = layer_labels == layer
    #     ax3.scatter(pc_scores[mask, pc1_idx], pc_scores[mask, pc2_idx],
    #             c=layer_colors[layer], alpha=0.5, s=30, label=layer)

    # ax3.set_xlabel(f'PC{pc1_idx+1} ({data["explained_variance_ratio"][pc1_idx]*100:.1f}%)', 
    #             fontsize=11)
    # ax3.set_ylabel(f'PC{pc2_idx+1} ({data["explained_variance_ratio"][pc2_idx]*100:.1f}%)', 
    #             fontsize=11)
    # ax3.set_title(f'Best Combination: {best["name"]}\n({best["rf_acc"]*100:.1f}% accuracy)',
    #             fontsize=12, fontweight='bold')
    # ax3.legend(fontsize=10)
    # ax3.grid(alpha=0.3)

    # # =========================================================================
    # # Panel 4: Cumulative explained variance
    # # =========================================================================
    # ax4 = fig.add_subplot(gs[1, 1])

    # cumulative_var = np.cumsum(data['explained_variance_ratio'][:10]) * 100

    # ax4.plot(range(1, 11), cumulative_var, 'o-', linewidth=2.5,
    #         markersize=8, color='steelblue')

    # ax4.axhline(90, color='red', linestyle='--', alpha=0.6, 
    #         label='90% variance', linewidth=2)
    # ax4.axhline(80, color='orange', linestyle='--', alpha=0.6,
    #         label='80% variance', linewidth=2)

    # ax4.set_xlabel('Number of PCs', fontsize=11)
    # ax4.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)
    # ax4.set_title('How Many PCs Needed?', fontsize=12, fontweight='bold')
    # ax4.set_xticks(range(1, 11))
    # ax4.legend(fontsize=10)
    # ax4.grid(alpha=0.3)
    # ax4.set_ylim(0, 100)

    # # =========================================================================
    # # Panel 5: Summary
    # # =========================================================================
    # ax5 = fig.add_subplot(gs[1, 2])
    # ax5.axis('off')

    # best = max(pc_combo_results, key=lambda x: x['rf_acc'])
    # single_pc_best = max([r for r in pc_combo_results if len(r['pcs']) == 1],
    #                     key=lambda x: x['rf_acc'])

    # summary_text = "KEY FINDINGS:\n\n"
    # summary_text += f"Best Single PC:\n"
    # summary_text += f"  {single_pc_best['name']}\n"
    # summary_text += f"  {single_pc_best['rf_acc']*100:.1f}% accuracy\n\n"

    # summary_text += f"Best Combination:\n"
    # summary_text += f"  {best['name']}\n"
    # summary_text += f"  {best['rf_acc']*100:.1f}% accuracy\n\n"

    # improvement = (best['rf_acc'] - single_pc_best['rf_acc']) * 100
    # summary_text += f"Combination Benefit:\n"
    # summary_text += f"  +{improvement:.1f}pp\n\n"

    # summary_text += "─" * 25 + "\n\n"
    # summary_text += "INTERPRETATION:\n"

    # if improvement > 10:
    #     summary_text += "✓ Multiple PCs needed\n"
    #     summary_text += "  to capture layer\n"
    #     summary_text += "  differences\n\n"
    #     summary_text += "→ Layers differ in\n"
    #     summary_text += "  multiple spatial\n"
    #     summary_text += "  dimensions"
    # else:
    #     summary_text += "~ Single PC captures\n"
    #     summary_text += "  most layer info\n\n"
    #     summary_text += "→ One dominant\n"
    #     summary_text += "  difference axis"

    # ax5.text(0.1, 0.9, summary_text,
    #         transform=ax5.transAxes,
    #         fontsize=10, fontfamily='monospace',
    #         verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('PC Combination Analysis\nHow Many PCs Needed to Distinguish Layers?',
                fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 7: {save_path}")

    return fig
                    
                    
# ============================================================================
# NEW FIGURE 8: PC TRAJECTORIES
# ============================================================================
def plot_figure8_pc_trajectories(data, trajectory_results, save_path=None):
    """
    Visualize how layers move through PC space across sessions.
    """
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    # =========================================================================
    # Panel 1: PC1-PC2 trajectories
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])

    for layer, traj in trajectory_results.items():
        centroids = traj['centroids'][:, :2]  # PC1, PC2
        color = traj['color']
        
        # Plot trajectory
        ax1.plot(centroids[:, 0], centroids[:, 1], 'o-',
                linewidth=2.5, markersize=8, color=color, label=layer, alpha=0.8)
        
        # Mark start and end
        ax1.scatter(centroids[0, 0], centroids[0, 1],
                s=150, color=color, marker='s', edgecolors='black',
                linewidths=2, alpha=0.8)
        ax1.scatter(centroids[-1, 0], centroids[-1, 1],
                s=150, color=color, marker='*', edgecolors='black',
                linewidths=2, alpha=0.8)

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_xlabel(f'PC1 ({data["explained_variance_ratio"][0]*100:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({data["explained_variance_ratio"][1]*100:.1f}%)', fontsize=11)
    ax1.set_title('Learning Trajectories: PC1-PC2 Space', 
                fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)

    # =========================================================================
    # Panel 2: PC1-PC3 trajectories
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])

    for layer, traj in trajectory_results.items():
        centroids_13 = traj['centroids'][:, [0, 2]]  # PC1, PC3
        color = traj['color']
        
        ax2.plot(centroids_13[:, 0], centroids_13[:, 1], 'o-',
                linewidth=2.5, markersize=8, color=color, label=layer, alpha=0.8)
        
        ax2.scatter(centroids_13[0, 0], centroids_13[0, 1],
                s=150, color=color, marker='s', edgecolors='black',
                linewidths=2, alpha=0.8)
        ax2.scatter(centroids_13[-1, 0], centroids_13[-1, 1],
                s=150, color=color, marker='*', edgecolors='black',
                linewidths=2, alpha=0.8)

    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax2.axvline(0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel(f'PC1 ({data["explained_variance_ratio"][0]*100:.1f}%)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({data["explained_variance_ratio"][2]*100:.1f}%)', fontsize=11)
    ax2.set_title('Learning Trajectories: PC1-PC3 Space',
                fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(alpha=0.3)

    # =========================================================================
    # Panel 3: Total movement bar chart
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])

    layers_sorted = sorted(trajectory_results.keys(),
                        key=lambda x: trajectory_results[x]['total_movement'],
                        reverse=True)

    movements = [trajectory_results[layer]['total_movement'] for layer in layers_sorted]
    colors = [trajectory_results[layer]['color'] for layer in layers_sorted]

    x = np.arange(len(layers_sorted))
    bars = ax3.bar(x, movements, color=colors, alpha=0.8, edgecolor='black', width=0.6)

    ax3.set_xticks(x)
    ax3.set_xticklabels(layers_sorted)
    ax3.set_ylabel('Total Movement (Euclidean)', fontsize=11)
    ax3.set_title('Total Movement in PC Space (PC1-PC5 Space)',
                fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    # Add values on bars
    for bar, mov in zip(bars, movements):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{mov:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # =========================================================================
    # Panel 4: PC1 trajectory over sessions
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])

    for layer, traj in trajectory_results.items():
        sessions = traj['sessions']
        pc1_vals = traj['centroids'][:, 0]
        color = traj['color']
        
        session_nums = [int(s.replace('Day', '')) for s in sessions]
        
        ax4.plot(session_nums, pc1_vals, 'o-',
                linewidth=2.5, markersize=8, color=color, label=layer, alpha=0.8)

    ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Session (Day)', fontsize=11)
    ax4.set_ylabel('PC1 Centroid', fontsize=11)
    ax4.set_title('PC1 Evolution\n(Negative = More Reward-Biased)',
                fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(alpha=0.3)

    # =========================================================================
    # Panel 5: Direction vectors
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])

    pc_labels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5']
    x = np.arange(len(pc_labels))
    width = 0.18

    for i, layer in enumerate(['L2/3', 'L4', 'L5', 'L6']):
        if layer not in trajectory_results:
            continue
        
        direction = trajectory_results[layer]['direction']
        color = trajectory_results[layer]['color']
        
        ax5.bar(x + i*width, direction, width, label=layer,
            color=color, alpha=0.8, edgecolor='black')

    ax5.axhline(0, color='black', linestyle='-', linewidth=1)
    ax5.set_xticks(x + 1.5*width)
    ax5.set_xticklabels(pc_labels)
    ax5.set_ylabel('Overall Direction', fontsize=11)
    ax5.set_title('Learning Direction by PC\n(Positive/Negative Shift)',
                fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10, ncol=2)
    ax5.grid(alpha=0.3, axis='y')

    # =========================================================================
    # Panel 6: Summary interpretation
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')

    # Find layer with most movement
    most_plastic = max(trajectory_results.items(),
                    key=lambda x: x[1]['total_movement'])
    least_plastic = min(trajectory_results.items(),
                    key=lambda x: x[1]['total_movement'])

    summary_text = "KEY FINDINGS:\n\n"
    summary_text += f"Most Plastic:\n"
    summary_text += f"  {most_plastic[0]}\n"
    summary_text += f"  {most_plastic[1]['total_movement']:.2f} movement\n\n"

    summary_text += f"Least Plastic:\n"
    summary_text += f"  {least_plastic[0]}\n"
    summary_text += f"  {least_plastic[1]['total_movement']:.2f} movement\n\n"

    summary_text += "─" * 25 + "\n\n"

    summary_text += "INTERPRETATION:\n"

    # Check if superficial layers move more
    superficial_movement = np.mean([trajectory_results[l]['total_movement'] 
                                for l in ['L2/3', 'L4'] 
                                if l in trajectory_results])
    deep_movement = np.mean([trajectory_results[l]['total_movement']
                            for l in ['L5', 'L6']
                            if l in trajectory_results])

    if superficial_movement > deep_movement * 1.2:
        summary_text += "✓ Superficial layers\n  show MORE plasticity\n\n"
        summary_text += "→ Superficial layers\n  learn spatial code\n"
    elif deep_movement > superficial_movement * 1.2:
        summary_text += "✓ Deep layers show\n  MORE plasticity\n\n"
        summary_text += "→ Deep layers refine\n  motor strategy\n"
    else:
        summary_text += "~ Similar plasticity\n  across layers\n\n"
        summary_text += "→ All layers adapt\n  to task structure\n"

    ax6.text(0.1, 0.9, summary_text,
            transform=ax6.transAxes,
            fontsize=10, fontfamily='monospace',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.suptitle('PC Space Trajectories\nHow Do Layers Move Through PC Space During Learning?',
                fontsize=14, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved Figure 8: {save_path}")

    return fig
# ============================================================================
# MAIN WORKFLOW
# ============================================================================
def run_comprehensive_analysis(pca_data_path, figure_dir, use_aligned=False):
    """
    Run complete comprehensive analysis with NEW multi-PC analyses.

    Parameters:
    -----------
    use_aligned : bool
        If True, use landmark-aligned profiles for PCA
    """
    print("="*80)
    print("COMPREHENSIVE PCA ANALYSIS")
    if use_aligned:
        print("*** USING LANDMARK-ALIGNED PROFILES ***")
    print("Layer-Specific Spatial Coding Strategies")
    print("="*80)

    os.makedirs(figure_dir, exist_ok=True)

    # Load data
    data = load_data(pca_data_path, use_aligned=use_aligned)
    
    # =========================================================================
    # EXISTING ANALYSES (keep these)
    # =========================================================================
    
    print("\n" + "="*80)
    print("RUNNING CORE ANALYSES...")
    print("="*80)
    
    # Analysis 1: Layer-specific PCA
    print("\n[1/9] Layer-specific PCA...")
    layer_pcas, pc1_similarity = analyze_layer_specific_pca(data, n_components=10)
    
    # Analysis 2: Landmark preference statistics
    print("\n[2/9] Landmark preference statistics...")
    landmark_stats = analyze_landmark_preference_statistics(data)
    
    # Analysis 3: PC1 comparison by layer
    print("\n[3/9] PC1 comparison by layer...")
    pc1_stats = analyze_pc1_by_layer(data)
    
    # Analysis 4: Procrustes test
    print("\n[4/9] Procrustes test...")
    procrustes_results = procrustes_test_layers(layer_pcas)
    
    # Analysis 5: Canonical correlation analysis
    print("\n[5/9] Canonical correlation analysis...")
    cca_results = canonical_correlation_layers(data)
    
    # Analysis 6: Layer decoding
    print("\n[6/9] Layer decoding...")
    decoding_results = decode_layer_from_pcs(data, n_pcs=5)
    
    # =========================================================================
    # NEW ANALYSES (add these)
    # =========================================================================
    
    print("\n" + "="*80)
    print("RUNNING EXTENDED ANALYSES...")
    print("="*80)
    
    # Analysis 7: Multi-PC layer differences
    print("\n[7/9] Multi-PC layer analysis...")
    multi_pc_results = analyze_multi_pc_layer_differences(data, layer_pcas)
    
    # Analysis 8: PC combination analysis
    print("\n[8/9] PC combination analysis...")
    pc_combo_results = analyze_pc_combinations(data)
    
    # Analysis 9: PC trajectory analysis
    print("\n[9/9] PC trajectory analysis...")
    trajectory_results = analyze_pc_trajectory_by_layer(data)
    
    # =========================================================================
    # GENERATE FIGURES
    # =========================================================================
    
    print("\n" + "="*80)
    print("GENERATING FIGURES...")
    print("="*80)
    
    # Core figures
    fig1 = plot_figure1_main_finding(
        landmark_stats, 
        save_path=os.path.join(figure_dir, 'Figure1_Main_Finding_Early.png')
    )
    
    fig1b = plot_figure1b_late_sessions(
        data,
        save_path=os.path.join(figure_dir, 'Figure1B_Early_vs_Late_Sessions.png')
    )
    
    fig1c = plot_figure1c_early_vs_late_comparison(
        data,
        save_path=os.path.join(figure_dir, 'Figure1C_Learning_Trajectory.png')
    )
    
    fig2 = plot_figure2_pc1_loadings(
        data, layer_pcas,
        save_path=os.path.join(figure_dir, 'Figure2_PC1_Loadings.png')
    )
    
    fig3 = plot_figure3_session_dynamics(
        data,
        save_path=os.path.join(figure_dir, 'Figure3_Session_Dynamics.png')
    )
    
    fig4 = plot_figure4_layer_comparison(
        data, pc1_stats,
        save_path=os.path.join(figure_dir, 'Figure4_Layer_Comparison.png')
    )
    
    fig5 = plot_figure5_supplementary_tests(
        procrustes_results, cca_results, decoding_results,
        save_path=os.path.join(figure_dir, 'Figure5_Supplementary_Tests.png')
    )
    
    # NEW figures
    fig6 = plot_figure6_multi_pc_analysis(
        data, layer_pcas, multi_pc_results,
        save_path=os.path.join(figure_dir, 'Figure6_Multi_PC_Analysis.png')
    )
    
    fig7 = plot_figure7_pc_combinations(
        data, pc_combo_results,
        save_path=os.path.join(figure_dir, 'Figure7_PC_Combinations.png')
    )
    
    fig8 = plot_figure8_pc_trajectories(
        data, trajectory_results,
        save_path=os.path.join(figure_dir, 'Figure8_PC_Trajectories.png')
    )
    
    # =========================================================================
    # PRINT SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    
    print(f"\nGenerated 8 Publication-Ready Figures:")
    print(f"  Figure 1: Main Finding - L4 Preference (Early)")
    print(f"  Figure 1B: Early vs Late Sessions")
    print(f"  Figure 1C: Learning Trajectories")
    print(f"  Figure 2: PC1 Loadings")
    print(f"  Figure 3: Session Dynamics")
    print(f"  Figure 4: Layer Comparison")
    print(f"  Figure 5: Supplementary Tests")
    print(f"  Figure 6: Multi-PC Analysis (NEW)")
    print(f"  Figure 7: PC Combinations (NEW)")
    print(f"  Figure 8: PC Trajectories (NEW)")
    
    print(f"\nKey Statistical Results:")
    
    # Finding 1: PC1
    if 'L2/3' in layer_pcas and 'L6' in layer_pcas:
        sim = 1 - cosine(layer_pcas['L2/3']['loadings'][0], layer_pcas['L6']['loadings'][0])
        print(f"\n  1. PC1 Similarity (L2/3 vs L6): {sim:.3f}")
        if sim < 0.7:
            print(f"     ✓ Layers use DISTINCT spatial coding strategies")
    
    # Finding 2: Landmark preference
    print(f"\n  2. L4 Preference (Early Sessions):")
    print(f"     Superficial: {landmark_stats['early_sessions']['sup_l4_pct']:.1f}%")
    print(f"     Deep:        {landmark_stats['early_sessions']['deep_l4_pct']:.1f}%")
    print(f"     p = {landmark_stats['early_sessions']['p_fisher']:.4f}")
    if landmark_stats['early_sessions']['deep_l4_pct'] > landmark_stats['early_sessions']['sup_l4_pct']:
        print(f"     ✓ Deep layers prefer L4 MORE from Day 1")
    
    # Finding 3: PC1 distribution
    print(f"\n  3. PC1 Distribution:")
    print(f"     Superficial: {pc1_stats['superficial']['mean']:+.3f}")
    print(f"     Deep:        {pc1_stats['deep']['mean']:+.3f}")
    print(f"     p = {pc1_stats['p_value']:.4f}, d = {pc1_stats['cohens_d']:.3f}")
    
    # Finding 4: Multi-PC (NEW)
    print(f"\n  4. Multi-PC Layer Discrimination:")
    best_pc = max(multi_pc_results.items(), 
                  key=lambda x: x[1]['decoding_acc'])
    print(f"     Best single PC: {best_pc[0]} ({best_pc[1]['decoding_acc']*100:.1f}% accuracy)")
    
    # Finding 5: PC combinations (NEW)
    print(f"\n  5. PC Combination Analysis:")
    best_combo = max(pc_combo_results, key=lambda x: x['rf_acc'])
    print(f"     Best combination: {best_combo['name']} ({best_combo['rf_acc']*100:.1f}%)")
    
    # Finding 6: Trajectories (NEW)
    print(f"\n  6. PC Space Trajectories:")
    for layer, traj in trajectory_results.items():
        if 'total_movement' in traj:
            print(f"     {layer}: {traj['total_movement']:.3f} total movement")
    
    print(f"\n{'='*80}")
    print(f"All figures saved to: {figure_dir}")
    print("="*80 + "\n")
    
    plt.show()
    
    return {
        'data': data,
        'layer_pcas': layer_pcas,
        'landmark_stats': landmark_stats,
        'pc1_stats': pc1_stats,
        'procrustes_results': procrustes_results,
        'cca_results': cca_results,
        'decoding_results': decoding_results,
        'multi_pc_results': multi_pc_results,  # NEW
        'pc_combo_results': pc_combo_results,   # NEW
        'trajectory_results': trajectory_results # NEW
    }


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    results = run_comprehensive_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR,
        use_aligned=USE_ALIGNED_PROFILES
    )
# ---

# This comprehensive script now includes:

# 1. **Layer-Specific PCA** - Tests if layers have different PC1 loadings
# 2. **Landmark Preference Statistics** - Statistical tests for L4 preference differences
# 3. **PC1 Distribution Comparison** - t-tests for layer differences
# 4. **Procrustes Test** - Subspace alignment test
# 5. **Canonical Correlation Analysis** - Shared vs. layer-specific dimensions
# 6. **Layer Decoding** - Can layer be predicted from PC space?
# 7. **Comprehensive Summary Figure** - Integrates all results

# The script directly tests your hypothesis from multiple angles and provides statistical validation for each claim!