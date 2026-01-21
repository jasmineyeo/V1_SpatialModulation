"""
PCA_SessionCorrection.py

Test and apply session effect correction before PCA.
Addresses the concern that session-to-session variance might
dominate PCA and mask biological layer/learning effects.

JSY, 01/2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from scipy import stats
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\figures\session_correction"


# ============================================================================
# SESSION CORRECTION METHODS
# ============================================================================

def method1_global_centering(profiles, session_labels):
    """
    Method 1: Simple session-wise centering.
    Subtract session mean from each profile.
    """
    corrected = profiles.copy()
    unique_sessions = np.unique(session_labels)
    
    for session in unique_sessions:
        mask = session_labels == session
        session_mean = np.mean(profiles[mask], axis=0)
        corrected[mask] = profiles[mask] - session_mean
    
    return corrected


def method2_session_zscore(profiles, session_labels):
    """
    Method 2: Z-score within each session.
    Each session normalized to mean=0, std=1.
    """
    corrected = np.zeros_like(profiles)
    unique_sessions = np.unique(session_labels)
    
    for session in unique_sessions:
        mask = session_labels == session
        session_profiles = profiles[mask]
        
        # Z-score the entire session's data
        session_mean = np.mean(session_profiles)
        session_std = np.std(session_profiles)
        
        if session_std > 0:
            corrected[mask] = (session_profiles - session_mean) / session_std
        else:
            corrected[mask] = session_profiles - session_mean
    
    return corrected


def method3_residual_correction(profiles, session_labels):
    """
    Method 3: Regression residuals.
    Model: profile = session_effects + residuals
    Return residuals (session-corrected profiles).
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Encode sessions as numeric
    le = LabelEncoder()
    session_numeric = le.fit_transform(session_labels)
    
    # Create design matrix (one-hot encoding)
    n_sessions = len(np.unique(session_labels))
    X = np.zeros((len(session_labels), n_sessions))
    for i, sess in enumerate(session_numeric):
        X[i, sess] = 1
    
    # Fit linear model for each spatial bin
    n_cells, n_bins = profiles.shape
    residuals = np.zeros_like(profiles)
    
    for bin_idx in range(n_bins):
        y = profiles[:, bin_idx]
        
        # Compute session means
        session_means = np.dot(X.T, y) / np.sum(X, axis=0)
        
        # Predict using session means
        y_pred = np.dot(X, session_means)
        
        # Store residuals
        residuals[:, bin_idx] = y - y_pred
    
    return residuals


def method4_robust_centering(profiles, session_labels):
    """
    Method 4: Median-based robust centering.
    Uses median instead of mean (less sensitive to outliers).
    """
    corrected = profiles.copy()
    unique_sessions = np.unique(session_labels)
    
    for session in unique_sessions:
        mask = session_labels == session
        session_median = np.median(profiles[mask], axis=0)
        corrected[mask] = profiles[mask] - session_median
    
    return corrected


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_correction(profiles_original, profiles_corrected, 
                       session_labels, layer_labels):
    """
    Evaluate how well correction removed session effects while
    preserving biological (layer) effects.
    """
    results = {}
    
    # Run PCA on both
    pca_orig = PCA(n_components=2)
    pca_corr = PCA(n_components=2)
    
    pc_orig = pca_orig.fit_transform(profiles_original)
    pc_corr = pca_corr.fit_transform(profiles_corrected)
    
    # Metric 1: Session clustering (should decrease)
    # Use F-statistic from ANOVA: Does PC1 differ by session?
    unique_sessions = np.unique(session_labels)
    
    groups_orig_sess = [pc_orig[session_labels == s, 0] for s in unique_sessions]
    groups_corr_sess = [pc_corr[session_labels == s, 0] for s in unique_sessions]
    
    f_orig_sess, p_orig_sess = stats.f_oneway(*groups_orig_sess)
    f_corr_sess, p_corr_sess = stats.f_oneway(*groups_corr_sess)
    
    results['session_f_original'] = f_orig_sess
    results['session_p_original'] = p_orig_sess
    results['session_f_corrected'] = f_corr_sess
    results['session_p_corrected'] = p_corr_sess
    results['session_reduction'] = (f_orig_sess - f_corr_sess) / f_orig_sess
    
    # Metric 2: Layer separation (should be preserved or increase)
    unique_layers = ['L2/3', 'L4', 'L5', 'L6']
    layer_masks = [layer_labels == l for l in unique_layers]
    
    groups_orig_layer = [pc_orig[mask, 0] for mask in layer_masks if np.sum(mask) > 0]
    groups_corr_layer = [pc_corr[mask, 0] for mask in layer_masks if np.sum(mask) > 0]
    
    if len(groups_orig_layer) >= 2:
        f_orig_layer, p_orig_layer = stats.f_oneway(*groups_orig_layer)
        f_corr_layer, p_corr_layer = stats.f_oneway(*groups_corr_layer)
        
        results['layer_f_original'] = f_orig_layer
        results['layer_p_original'] = p_orig_layer
        results['layer_f_corrected'] = f_corr_layer
        results['layer_p_corrected'] = p_corr_layer
        results['layer_change'] = (f_corr_layer - f_orig_layer) / f_orig_layer
    
    # Metric 3: Variance retained
    var_orig = np.var(profiles_original)
    var_corr = np.var(profiles_corrected)
    results['variance_ratio'] = var_corr / var_orig
    
    # Metric 4: PC1 variance explained
    results['pc1_var_original'] = pca_orig.explained_variance_ratio_[0]
    results['pc1_var_corrected'] = pca_corr.explained_variance_ratio_[0]
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def compare_correction_methods(data, save_dir):
    """
    Compare all correction methods visually and quantitatively.
    """
    profiles = data['spatial_profiles_zscore']
    session_labels = data['session_labels']
    layer_labels = data['layer_labels']
    bin_centers = data['bin_centers']
    
    print("\nTesting session correction methods...")
    print("=" * 70)
    
    # Apply all methods
    methods = {
        'Original (no correction)': profiles,
        'Method 1: Session centering': method1_global_centering(profiles, session_labels),
        'Method 2: Session z-score': method2_session_zscore(profiles, session_labels),
        'Method 3: Regression residuals': method3_residual_correction(profiles, session_labels),
        'Method 4: Robust centering': method4_robust_centering(profiles, session_labels)
    }
    
    # Evaluate each method
    evaluations = {}
    for method_name, corrected_profiles in methods.items():
        if method_name != 'Original (no correction)':
            eval_results = evaluate_correction(
                profiles, corrected_profiles,
                session_labels, layer_labels
            )
            evaluations[method_name] = eval_results
            
            print(f"\n{method_name}:")
            print(f"  Session F: {eval_results['session_f_original']:.2f} → {eval_results['session_f_corrected']:.2f} ({eval_results['session_reduction']*100:.1f}% reduction)")
            if 'layer_f_original' in eval_results:
                print(f"  Layer F: {eval_results['layer_f_original']:.2f} → {eval_results['layer_f_corrected']:.2f} ({eval_results['layer_change']*100:+.1f}% change)")
            print(f"  Variance retained: {eval_results['variance_ratio']*100:.1f}%")
    
    # Visualization
    n_methods = len(methods)
    fig = plt.figure(figsize=(20, 4 * n_methods))
    gs = GridSpec(n_methods, 4, figure=fig, hspace=0.4, wspace=0.3)
    
    unique_sessions = sorted(np.unique(session_labels))
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    for idx, (method_name, corrected_profiles) in enumerate(methods.items()):
        # Panel 1: Session clustering in PC space
        ax1 = fig.add_subplot(gs[idx, 0])
        
        pca = PCA(n_components=2)
        pc_scores = pca.fit_transform(corrected_profiles)
        
        for si, session in enumerate(unique_sessions):
            mask = session_labels == session
            color = plt.cm.viridis(si / len(unique_sessions))
            ax1.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                       c=[color], alpha=0.4, s=10, label=session if idx == 0 else "")
            
            # Centroid
            centroid = np.mean(pc_scores[mask, :2], axis=0)
            ax1.scatter(centroid[0], centroid[1], c=[color],
                       s=200, marker='*', edgecolors='black', linewidths=1.5)
        
        ax1.set_xlabel('PC1', fontsize=9)
        ax1.set_ylabel('PC2', fontsize=9)
        ax1.set_title(f'{method_name}\nSession Clustering', fontsize=10)
        ax1.grid(alpha=0.3)
        if idx == 0:
            ax1.legend(fontsize=7, ncol=2)
        
        # Panel 2: Layer separation in PC space
        ax2 = fig.add_subplot(gs[idx, 1])
        
        for layer, color in layer_colors.items():
            mask = layer_labels == layer
            if np.sum(mask) > 0:
                ax2.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                           c=color, alpha=0.4, s=10, label=layer if idx == 0 else "")
        
        ax2.set_xlabel('PC1', fontsize=9)
        ax2.set_ylabel('PC2', fontsize=9)
        ax2.set_title('Layer Separation', fontsize=10)
        if idx == 0:
            ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
        
        # Panel 3: Mean profile per session
        ax3 = fig.add_subplot(gs[idx, 2])
        
        for si, session in enumerate(unique_sessions):
            mask = session_labels == session
            mean_profile = np.mean(corrected_profiles[mask], axis=0)
            color = plt.cm.viridis(si / len(unique_sessions))
            ax3.plot(bin_centers, mean_profile, color=color, 
                    alpha=0.7, linewidth=1.5, label=session if idx == 0 else "")
        
        ax3.set_xlabel('Position (cm)', fontsize=9)
        ax3.set_ylabel('Activity', fontsize=9)
        ax3.set_title('Mean Profiles by Session', fontsize=10)
        if idx == 0:
            ax3.legend(fontsize=7, ncol=2)
        
        # Panel 4: Evaluation metrics
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.axis('off')
        
        if method_name in evaluations:
            eval_res = evaluations[method_name]
            
            metrics_text = f"""{method_name}

Session Effect:
  F: {eval_res['session_f_original']:.1f} → {eval_res['session_f_corrected']:.1f}
  Reduction: {eval_res['session_reduction']*100:.1f}%

Layer Effect:
  F: {eval_res.get('layer_f_original', 0):.1f} → {eval_res.get('layer_f_corrected', 0):.1f}
  Change: {eval_res.get('layer_change', 0)*100:+.1f}%

Variance: {eval_res['variance_ratio']*100:.1f}%

GOAL: 
✓ Session F should ↓
✓ Layer F should ↑ or same
✓ Variance ~80-100%
"""
            
            ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes,
                    fontsize=8, fontfamily='monospace', verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Session Correction Method Comparison', 
                fontsize=14, fontweight='bold')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'session_correction_comparison.png'),
                   dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved comparison figure")
    
    return fig, methods, evaluations


def recommend_best_method(evaluations):
    """
    Recommend best correction method based on evaluation metrics.
    """
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    
    scores = {}
    
    for method_name, eval_res in evaluations.items():
        score = 0
        
        # Score 1: Session effect reduction (40 points max)
        session_reduction = eval_res['session_reduction']
        score += session_reduction * 40
        
        # Score 2: Layer effect preservation (40 points max)
        if 'layer_change' in eval_res:
            layer_change = eval_res['layer_change']
            if layer_change > 0:
                score += min(layer_change, 1.0) * 40
            else:
                score += max(0, (1 + layer_change) * 20)
        
        # Score 3: Variance retention (20 points max)
        var_score = min(eval_res['variance_ratio'], 1.0) * 20
        score += var_score
        
        scores[method_name] = score
    
    # Find best method
    best_method = max(scores, key=scores.get)
    
    print(f"\nScores (out of 100):")
    for method_name in sorted(scores, key=scores.get, reverse=True):
        print(f"  {method_name}: {scores[method_name]:.1f}")
    
    print(f"\n✓ RECOMMENDED: {best_method}")
    
    return best_method, scores


# ============================================================================
# APPLY AND SAVE
# ============================================================================

def apply_correction_and_save(data, method_name, pca_data_path):
    """
    Apply chosen correction method and save corrected profiles to HDF5.
    """
    profiles = data['spatial_profiles_zscore']
    session_labels = data['session_labels']
    
    print(f"\nApplying {method_name}...")
    
    if 'Method 1' in method_name:
        corrected = method1_global_centering(profiles, session_labels)
    elif 'Method 2' in method_name:
        corrected = method2_session_zscore(profiles, session_labels)
    elif 'Method 3' in method_name:
        corrected = method3_residual_correction(profiles, session_labels)
    elif 'Method 4' in method_name:
        corrected = method4_robust_centering(profiles, session_labels)
    else:
        print("Unknown method, using original profiles")
        corrected = profiles
    
    # Save to HDF5
    with h5py.File(pca_data_path, 'a') as f:
        if 'features/spatial_profiles_session_corrected' in f:
            del f['features/spatial_profiles_session_corrected']
        
        f['features'].create_dataset('spatial_profiles_session_corrected', 
                                     data=corrected)
        f['features'].attrs['correction_method'] = method_name
    
    print(f"✓ Saved session-corrected profiles to HDF5")
    print(f"  Dataset: features/spatial_profiles_session_corrected")
    
    return corrected


# ============================================================================
# MAIN
# ============================================================================

def run_session_correction_analysis(pca_data_path, figure_dir, 
                                    auto_apply_best=False):
    """
    Main function to test session correction methods.
    """
    print("=" * 70)
    print("SESSION EFFECT CORRECTION ANALYSIS")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = {}
    with h5py.File(pca_data_path, 'r') as f:
        data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
        data['session_labels'] = f['cells/session_labels'][:].astype(str)
        data['layer_labels'] = f['cells/layer_labels'][:].astype(str)
        data['bin_centers'] = f['metadata/bin_centers_trimmed'][:]
    
    print(f"  Loaded {len(data['spatial_profiles_zscore'])} cells")
    print(f"  Sessions: {np.unique(data['session_labels'])}")
    
    # Compare methods
    fig, methods, evaluations = compare_correction_methods(data, figure_dir)
    
    # Recommend best method
    best_method, scores = recommend_best_method(evaluations)
    
    # Optionally apply best method
    if auto_apply_best:
        apply_correction_and_save(data, best_method, pca_data_path)
    else:
        print("\n" + "=" * 70)
        print("To apply correction, rerun with auto_apply_best=True")
        print("=" * 70)
    
    plt.show()
    
    return methods, evaluations, best_method


if __name__ == "__main__":
    methods, evaluations, best_method = run_session_correction_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR,
        auto_apply_best=True  # Set to True to automatically apply correction
    )