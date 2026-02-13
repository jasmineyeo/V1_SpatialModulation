"""
PCA_Interpretation.py
Interactive exploration and interpretation of PCA results.

Run this after PCA_Analysis.py to:
1. Quantify what each PC represents
2. Examine relationships between PCs and cell properties
3. Look for structure that might indicate meaningful subgroups
4. Generate additional targeted visualizations

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

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\figures\interpretation"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(filepath):
    """Load all data including PCA results."""
    
    print(f"Loading data from: {filepath}")
    
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Metadata
        data['animal_id'] = f['metadata'].attrs['animal_id']
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
        if 'pca_results' in f and 'pc_scores' in f['pca_results']:
            data['pc_scores'] = f['pca_results/pc_scores'][:]
            data['components'] = f['pca_results/components'][:]
            data['explained_variance_ratio'] = f['pca_results/explained_variance_ratio'][:]
        else:
            raise ValueError("PCA results not found! Run PCA_Analysis.py first.")
    
    print(f"  Loaded {len(data['pc_scores'])} cells with {data['pc_scores'].shape[1]} PCs")
    
    return data


# ============================================================================
# PC INTERPRETATION: CORRELATIONS WITH CELL PROPERTIES
# ============================================================================

def analyze_pc_correlations(data):
    """
    Analyze correlations between PC scores and cell properties.
    This helps interpret what each PC represents.
    """
    
    pc_scores = data['pc_scores']
    peak_positions = data['peak_positions']
    n_pcs = pc_scores.shape[1]
    
    print("\n" + "=" * 60)
    print("PC CORRELATIONS WITH CELL PROPERTIES")
    print("=" * 60)
    
    # Correlation with peak position
    print("\nCorrelation with Peak Position:")
    for pc_idx in range(min(5, n_pcs)):
        r, p = stats.pearsonr(pc_scores[:, pc_idx], peak_positions)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  PC{pc_idx+1}: r = {r:.3f}, p = {p:.2e} {sig}")
    
    # ANOVA by layer
    print("\nANOVA by Layer (does PC score differ by layer?):")
    layer_labels = data['layer_labels']
    unique_layers = ['L2/3', 'L4', 'L5', 'L6']
    
    for pc_idx in range(min(5, n_pcs)):
        groups = [pc_scores[layer_labels == layer, pc_idx] for layer in unique_layers]
        groups = [g for g in groups if len(g) > 0]  # Remove empty groups
        
        if len(groups) >= 2:
            f_stat, p = stats.f_oneway(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  PC{pc_idx+1}: F = {f_stat:.2f}, p = {p:.2e} {sig}")
    
    # ANOVA by landmark preference
    print("\nANOVA by Landmark Preference:")
    preferred_landmark = data['preferred_landmark']
    
    for pc_idx in range(min(5, n_pcs)):
        groups = [pc_scores[preferred_landmark == lm, pc_idx] for lm in range(4)]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            f_stat, p = stats.f_oneway(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  PC{pc_idx+1}: F = {f_stat:.2f}, p = {p:.2e} {sig}")
    
    # Kruskal-Wallis by session (non-parametric)
    print("\nKruskal-Wallis by Session:")
    session_labels = data['session_labels']
    unique_sessions = sorted(np.unique(session_labels), key=lambda x: int(x.replace('Day', '')))
    
    for pc_idx in range(min(5, n_pcs)):
        groups = [pc_scores[session_labels == sess, pc_idx] for sess in unique_sessions]
        groups = [g for g in groups if len(g) > 0]
        
        if len(groups) >= 2:
            h_stat, p = stats.kruskal(*groups)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  PC{pc_idx+1}: H = {h_stat:.2f}, p = {p:.2e} {sig}")
    
    return


def compute_profile_features(data):
    """
    Compute additional features from spatial profiles that might help
    interpret PCs and identify adaptation vs. true spatial encoding.
    """
    
    profiles = data['spatial_profiles_zscore']
    bin_centers = data['bin_centers']
    n_cells = profiles.shape[0]
    
    print("\n" + "=" * 60)
    print("COMPUTING ADDITIONAL PROFILE FEATURES")
    print("=" * 60)
    
    features = {}
    
    # Feature 1: Peak width (FWHM - Full Width at Half Maximum)
    print("  Computing peak widths...")
    peak_widths = np.zeros(n_cells)
    
    for i in range(n_cells):
        profile = profiles[i]
        max_val = np.max(profile)
        half_max = max_val / 2
        
        above_half = profile >= half_max
        if np.sum(above_half) > 0:
            indices = np.where(above_half)[0]
            width_bins = indices[-1] - indices[0]
            peak_widths[i] = width_bins * np.mean(np.diff(bin_centers))  # Convert to cm
        else:
            peak_widths[i] = np.nan
    
    features['peak_width'] = peak_widths
    
    # Feature 2: Response asymmetry around peak
    print("  Computing response asymmetry...")
    asymmetry = np.zeros(n_cells)
    
    for i in range(n_cells):
        profile = profiles[i]
        peak_idx = np.argmax(profile)
        
        # Get activity before and after peak (equal windows)
        window = 10  # bins
        before_start = max(0, peak_idx - window)
        after_end = min(len(profile), peak_idx + window)
        
        before_activity = np.sum(profile[before_start:peak_idx])
        after_activity = np.sum(profile[peak_idx:after_end])
        
        total = before_activity + after_activity
        if total > 0:
            asymmetry[i] = (before_activity - after_activity) / total
        else:
            asymmetry[i] = 0
    
    features['asymmetry'] = asymmetry  # Positive = more activity before peak
    
    # Feature 3: Early decay slope (potential adaptation signature)
    print("  Computing early decay slopes...")
    early_slopes = np.zeros(n_cells)
    
    # Find where onset zone ends (approximately 10-15 cm)
    onset_end_idx = np.searchsorted(bin_centers, 15)
    slope_end_idx = np.searchsorted(bin_centers, 30)  # Check first 15cm after onset zone
    
    for i in range(n_cells):
        profile = profiles[i]
        early_region = profile[onset_end_idx:slope_end_idx]
        
        if len(early_region) > 2:
            # Linear regression slope
            x = np.arange(len(early_region))
            slope, _, _, _, _ = stats.linregress(x, early_region)
            early_slopes[i] = slope
        else:
            early_slopes[i] = 0
    
    features['early_slope'] = early_slopes  # Negative = decaying (adaptation-like)
    
    # Feature 4: Baseline activity (mean activity in non-preferred regions)
    print("  Computing baseline activity...")
    baseline = np.zeros(n_cells)
    
    for i in range(n_cells):
        profile = profiles[i]
        peak_idx = np.argmax(profile)
        
        # Exclude ±15 bins around peak
        mask = np.ones(len(profile), dtype=bool)
        exclude_start = max(0, peak_idx - 15)
        exclude_end = min(len(profile), peak_idx + 15)
        mask[exclude_start:exclude_end] = False
        
        if np.sum(mask) > 0:
            baseline[i] = np.mean(profile[mask])
        else:
            baseline[i] = 0
    
    features['baseline'] = baseline
    
    # Feature 5: Response reliability (variance across the profile)
    print("  Computing profile variance...")
    features['profile_variance'] = np.var(profiles, axis=1)
    
    print(f"  Computed {len(features)} features")
    
    return features


def correlate_features_with_pcs(data, features):
    """
    Correlate computed profile features with PC scores.
    """
    
    pc_scores = data['pc_scores']
    n_pcs = min(5, pc_scores.shape[1])
    
    print("\n" + "=" * 60)
    print("FEATURE-PC CORRELATIONS")
    print("=" * 60)
    
    for feat_name, feat_values in features.items():
        print(f"\n{feat_name}:")
        
        # Remove NaN values for correlation
        valid = ~np.isnan(feat_values)
        
        for pc_idx in range(n_pcs):
            r, p = stats.pearsonr(feat_values[valid], pc_scores[valid, pc_idx])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  PC{pc_idx+1}: r = {r:.3f}, p = {p:.2e} {sig}")


# ============================================================================
# VISUALIZATION: PROFILE FEATURES
# ============================================================================

def plot_feature_distributions(data, features, save_path=None):
    """
    Plot distributions of computed features, split by landmark preference.
    """
    
    preferred_landmark = data['preferred_landmark']
    landmark_positions = data['landmark_positions']
    
    n_features = len(features)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    lm_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3', -1: 'gray'}
    
    for idx, (feat_name, feat_values) in enumerate(features.items()):
        if idx >= 6:
            break
            
        ax = axes[idx]
        
        # Violin plot by landmark
        data_by_lm = []
        labels = []
        colors = []
        
        for lm_idx in range(4):
            mask = preferred_landmark == lm_idx
            if np.sum(mask) > 0:
                vals = feat_values[mask]
                vals = vals[~np.isnan(vals)]
                if len(vals) > 0:
                    data_by_lm.append(vals)
                    labels.append(f'L{lm_idx+1}')
                    colors.append(lm_colors[lm_idx])
        
        if len(data_by_lm) > 0:
            parts = ax.violinplot(data_by_lm, positions=range(len(data_by_lm)), 
                                 showmeans=True, showmedians=True)
            
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
        
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel(feat_name)
        ax.set_title(feat_name, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
    
    # Hide unused axes
    for idx in range(len(features), 6):
        axes[idx].axis('off')
    
    plt.suptitle('Profile Features by Landmark Preference', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
    
    return fig


def plot_pc_vs_features(data, features, save_path=None):
    """
    Scatter plots of PC scores vs computed features.
    """
    
    pc_scores = data['pc_scores']
    preferred_landmark = data['preferred_landmark']
    
    lm_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a', 3: '#984ea3', -1: 'lightgray'}
    
    # Focus on key features for adaptation hypothesis
    key_features = ['early_slope', 'asymmetry', 'peak_width']
    key_features = [f for f in key_features if f in features]
    
    n_features = len(key_features)
    fig, axes = plt.subplots(n_features, 3, figsize=(14, 4*n_features))
    
    if n_features == 1:
        axes = axes.reshape(1, -1)
    
    for row, feat_name in enumerate(key_features):
        feat_values = features[feat_name]
        valid = ~np.isnan(feat_values)
        
        for col in range(3):
            ax = axes[row, col]
            
            # Plot by landmark preference
            for lm_idx in [-1, 0, 1, 2, 3]:  # Background first
                mask = (preferred_landmark == lm_idx) & valid
                if np.sum(mask) > 0:
                    alpha = 0.3 if lm_idx == -1 else 0.7
                    size = 20 if lm_idx == -1 else 40
                    ax.scatter(pc_scores[mask, col], feat_values[mask],
                              c=lm_colors[lm_idx], alpha=alpha, s=size,
                              label=f'L{lm_idx+1}' if lm_idx >= 0 else 'Between')
            
            # Add correlation
            r, p = stats.pearsonr(pc_scores[valid, col], feat_values[valid])
            ax.set_title(f'r={r:.2f}, p={p:.1e}', fontsize=10)
            
            ax.set_xlabel(f'PC{col+1}')
            if col == 0:
                ax.set_ylabel(feat_name)
            ax.grid(alpha=0.3)
            
            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle('PC Scores vs Profile Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
    
    return fig


# ============================================================================
# L1 CELLS: ADAPTATION ANALYSIS
# ============================================================================

def analyze_l1_adaptation(data, features, save_path=None):
    """
    Specifically analyze L1 cells for signatures of adaptation vs spatial encoding.
    """
    
    preferred_landmark = data['preferred_landmark']
    l1_mask = preferred_landmark == 0
    n_l1 = np.sum(l1_mask)
    
    print("\n" + "=" * 60)
    print(f"L1 CELL ADAPTATION ANALYSIS (n={n_l1})")
    print("=" * 60)
    
    if n_l1 < 10:
        print("  Not enough L1 cells for meaningful analysis")
        return None
    
    # Get L1 cell data
    l1_profiles = data['spatial_profiles_zscore'][l1_mask]
    l1_pc_scores = data['pc_scores'][l1_mask]
    l1_early_slopes = features['early_slope'][l1_mask]
    l1_asymmetry = features['asymmetry'][l1_mask]
    l1_layers = data['layer_labels'][l1_mask]
    l1_sessions = data['session_labels'][l1_mask]
    
    # Split L1 cells by early slope (potential adaptation marker)
    slope_median = np.median(l1_early_slopes)
    adaptation_like = l1_early_slopes < slope_median  # Negative slope = decaying
    spatial_like = l1_early_slopes >= slope_median
    
    print(f"\n  Split by early slope (median = {slope_median:.4f}):")
    print(f"    Adaptation-like (negative slope): {np.sum(adaptation_like)}")
    print(f"    Spatial-like (positive/zero slope): {np.sum(spatial_like)}")
    
    # Compare PC scores between groups
    print(f"\n  PC score comparison:")
    for pc_idx in range(min(3, l1_pc_scores.shape[1])):
        adapt_scores = l1_pc_scores[adaptation_like, pc_idx]
        spatial_scores = l1_pc_scores[spatial_like, pc_idx]
        
        if len(adapt_scores) > 1 and len(spatial_scores) > 1:
            t_stat, p = stats.ttest_ind(adapt_scores, spatial_scores)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"    PC{pc_idx+1}: adapt={np.mean(adapt_scores):.2f}±{np.std(adapt_scores):.2f}, "
                  f"spatial={np.mean(spatial_scores):.2f}±{np.std(spatial_scores):.2f}, "
                  f"t={t_stat:.2f}, p={p:.3f} {sig}")
    
    # Layer distribution
    print(f"\n  Layer distribution:")
    for layer in ['L2/3', 'L4', 'L5', 'L6']:
        n_adapt = np.sum((l1_layers == layer) & adaptation_like)
        n_spatial = np.sum((l1_layers == layer) & spatial_like)
        print(f"    {layer}: Adaptation-like={n_adapt}, Spatial-like={n_spatial}")
    
    # Session distribution
    print(f"\n  Session distribution:")
    unique_sessions = sorted(np.unique(l1_sessions), key=lambda x: int(x.replace('Day', '')))
    for session in unique_sessions:
        n_adapt = np.sum((l1_sessions == session) & adaptation_like)
        n_spatial = np.sum((l1_sessions == session) & spatial_like)
        print(f"    {session}: Adaptation-like={n_adapt}, Spatial-like={n_spatial}")
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    
    # Panel 1: Average profiles of the two groups
    ax1 = fig.add_subplot(2, 3, 1)
    
    mean_adapt = np.mean(l1_profiles[adaptation_like], axis=0)
    mean_spatial = np.mean(l1_profiles[spatial_like], axis=0)
    sem_adapt = np.std(l1_profiles[adaptation_like], axis=0) / np.sqrt(np.sum(adaptation_like))
    sem_spatial = np.std(l1_profiles[spatial_like], axis=0) / np.sqrt(np.sum(spatial_like))
    
    ax1.plot(bin_centers, mean_adapt, 'r-', linewidth=2, 
            label=f'Adaptation-like (n={np.sum(adaptation_like)})')
    ax1.fill_between(bin_centers, mean_adapt - sem_adapt, mean_adapt + sem_adapt, 
                    alpha=0.2, color='red')
    ax1.plot(bin_centers, mean_spatial, 'b-', linewidth=2, 
            label=f'Spatial-like (n={np.sum(spatial_like)})')
    ax1.fill_between(bin_centers, mean_spatial - sem_spatial, mean_spatial + sem_spatial, 
                    alpha=0.2, color='blue')
    
    for lm_pos in landmark_positions:
        ax1.axvline(lm_pos, color='green', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Z-scored Activity')
    ax1.set_title('L1 Cells: Adaptation-like vs Spatial-like', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Heatmap of adaptation-like cells
    ax2 = fig.add_subplot(2, 3, 2)
    sorted_idx = np.argsort(np.argmax(l1_profiles[adaptation_like], axis=1))
    im = ax2.imshow(l1_profiles[adaptation_like][sorted_idx], aspect='auto', cmap='viridis',
                   extent=[bin_centers[0], bin_centers[-1], 0, np.sum(adaptation_like)])
    for lm_pos in landmark_positions:
        ax2.axvline(lm_pos, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Position (cm)')
    ax2.set_ylabel('Cell #')
    ax2.set_title(f'Adaptation-like L1 cells (n={np.sum(adaptation_like)})', fontweight='bold')
    
    # Panel 3: Heatmap of spatial-like cells
    ax3 = fig.add_subplot(2, 3, 3)
    sorted_idx = np.argsort(np.argmax(l1_profiles[spatial_like], axis=1))
    im = ax3.imshow(l1_profiles[spatial_like][sorted_idx], aspect='auto', cmap='viridis',
                   extent=[bin_centers[0], bin_centers[-1], 0, np.sum(spatial_like)])
    for lm_pos in landmark_positions:
        ax3.axvline(lm_pos, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Position (cm)')
    ax3.set_ylabel('Cell #')
    ax3.set_title(f'Spatial-like L1 cells (n={np.sum(spatial_like)})', fontweight='bold')
    
    # Panel 4: PC1 vs PC2 with group coloring
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(l1_pc_scores[adaptation_like, 0], l1_pc_scores[adaptation_like, 1],
               c='red', alpha=0.7, s=50, label='Adaptation-like')
    ax4.scatter(l1_pc_scores[spatial_like, 0], l1_pc_scores[spatial_like, 1],
               c='blue', alpha=0.7, s=50, label='Spatial-like')
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('L1 Cells in PC Space', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Panel 5: Early slope distribution
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(l1_early_slopes[adaptation_like], bins=20, alpha=0.6, color='red', 
            label='Adaptation-like')
    ax5.hist(l1_early_slopes[spatial_like], bins=20, alpha=0.6, color='blue', 
            label='Spatial-like')
    ax5.axvline(slope_median, color='black', linestyle='--', label='Median')
    ax5.set_xlabel('Early Slope')
    ax5.set_ylabel('Count')
    ax5.set_title('Early Slope Distribution', fontweight='bold')
    ax5.legend()
    
    # Panel 6: Session distribution
    ax6 = fig.add_subplot(2, 3, 6)
    
    session_counts_adapt = [np.sum((l1_sessions == s) & adaptation_like) for s in unique_sessions]
    session_counts_spatial = [np.sum((l1_sessions == s) & spatial_like) for s in unique_sessions]
    
    x = np.arange(len(unique_sessions))
    width = 0.35
    ax6.bar(x - width/2, session_counts_adapt, width, label='Adaptation-like', color='red', alpha=0.7)
    ax6.bar(x + width/2, session_counts_spatial, width, label='Spatial-like', color='blue', alpha=0.7)
    ax6.set_xticks(x)
    ax6.set_xticklabels(unique_sessions, rotation=45)
    ax6.set_ylabel('Count')
    ax6.set_title('L1 Cell Types by Session', fontweight='bold')
    ax6.legend()
    
    plt.suptitle('L1-Preferring Cells: Adaptation vs Spatial Encoding Analysis', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(save_path)}")
    
    return fig


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_interpretation(pca_data_path, figure_dir):
    """
    Main function for PCA interpretation.
    """
    
    print("=" * 80)
    print("PCA INTERPRETATION")
    print("=" * 80)
    
    os.makedirs(figure_dir, exist_ok=True)
    
    # Load data
    data = load_all_data(pca_data_path)
    
    # Analyze PC correlations
    analyze_pc_correlations(data)
    
    # Compute additional features
    features = compute_profile_features(data)
    
    # Correlate features with PCs
    correlate_features_with_pcs(data, features)
    
    # Visualizations
    print("\nGenerating interpretation figures...")
    
    plot_feature_distributions(data, features,
                              save_path=os.path.join(figure_dir, 'feature_distributions.png'))
    
    plot_pc_vs_features(data, features,
                       save_path=os.path.join(figure_dir, 'pc_vs_features.png'))
    
    analyze_l1_adaptation(data, features,
                         save_path=os.path.join(figure_dir, 'l1_adaptation_analysis.png'))
    
    print("\n" + "=" * 80)
    print("INTERPRETATION COMPLETE!")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    
    plt.show()
    
    return data, features


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    data, features = run_interpretation(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR
    )