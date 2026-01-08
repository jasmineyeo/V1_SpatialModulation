"""
PCA_Analysis.py
Performs PCA on aggregated spatial response data and generates diagnostic plots.

Run this after PCA_DataAggregation.py has created the _pca_data.h5 file.

JSY, 12/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to the aggregated PCA data file
PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\PCA\JSY051_pca_data.h5"

# Output directory for figures
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\PCA\figures"

# Number of PCs to compute and analyze
N_COMPONENTS = 10

# Figure settings
FIGSIZE_SCREE = (10, 4)
FIGSIZE_LOADINGS = (14, 8)
FIGSIZE_SCATTER = (12, 10)
DPI = 150


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pca_data(filepath):
    """
    Load aggregated PCA data from HDF5 file.
    """
    print(f"Loading data from: {filepath}")
    
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Metadata
        data['animal_id'] = f['metadata'].attrs['animal_id']
        data['n_sessions'] = f['metadata'].attrs['n_sessions']
        data['n_cells'] = f['metadata'].attrs['n_cells_total']
        data['bin_centers'] = f['metadata/bin_centers_trimmed'][:]
        data['landmark_positions'] = f['metadata/landmark_positions'][:]
        
        # Cell labels (decode bytes to strings)
        data['session_labels'] = f['cells/session_labels'][:].astype(str)
        data['layer_labels'] = f['cells/layer_labels'][:].astype(str)
        data['preferred_landmark'] = f['cells/preferred_landmark'][:]
        data['peak_positions'] = f['cells/peak_positions'][:]
        
        # Features
        data['spatial_profiles'] = f['features/spatial_profiles'][:]
        data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
    
    print(f"  Animal: {data['animal_id']}")
    print(f"  Cells: {data['n_cells']}")
    print(f"  Sessions: {data['n_sessions']}")
    print(f"  Features: {data['spatial_profiles'].shape[1]} spatial bins")
    
    return data


# ============================================================================
# PCA COMPUTATION
# ============================================================================

def run_pca(profiles_zscore, n_components=10):
    """
    Run PCA on z-scored spatial profiles.
    
    Returns:
    --------
    pca_results : dict
        Contains PCA object, scores, explained variance, etc.
    """
    print(f"\nRunning PCA with {n_components} components...")
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(profiles_zscore)
    
    pca_results = {
        'pca': pca,
        'pc_scores': pc_scores,
        'explained_variance': pca.explained_variance_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,  # Loadings (n_components x n_features)
        'n_components': n_components
    }
    
    print(f"  Variance explained by first 3 PCs: {pca_results['cumulative_variance_ratio'][2]*100:.1f}%")
    print(f"  Variance explained by first 5 PCs: {pca_results['cumulative_variance_ratio'][4]*100:.1f}%")
    
    return pca_results


# ============================================================================
# VISUALIZATION: SCREE PLOT
# ============================================================================

def plot_scree(pca_results, save_path=None):
    """
    Plot scree plot showing variance explained by each PC.
    """
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_SCREE)
    
    n_components = pca_results['n_components']
    x = np.arange(1, n_components + 1)
    
    # Left: Individual variance
    ax1 = axes[0]
    ax1.bar(x, pca_results['explained_variance_ratio'] * 100, 
            color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Variance Explained (%)', fontsize=11)
    ax1.set_title('Variance Explained by Each PC', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.grid(axis='y', alpha=0.3)
    
    # Right: Cumulative variance
    ax2 = axes[1]
    ax2.set_xlabel('Principal Component', fontsize=11)
    ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)
    ax2.set_title('Cumulative Variance Explained', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right')
    ax2.grid(alpha=0.3)
    
    ax2.plot(x, pca_results['cumulative_variance_ratio'] * 100, 'o-', 
         color='steelblue', linewidth=2, markersize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved scree plot: {os.path.basename(save_path)}")
    
    return fig


# ============================================================================
# VISUALIZATION: PC LOADINGS
# ============================================================================

def plot_pc_loadings(pca_results, bin_centers, landmark_positions, 
                     n_pcs_to_plot=4, save_path=None):
    """
    Plot loadings for the first N principal components.
    Shows what spatial patterns each PC captures.
    """
    n_pcs = min(n_pcs_to_plot, pca_results['n_components'])
    
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_LOADINGS)
    axes = axes.flatten()
    
    for pc_idx in range(n_pcs):
        ax = axes[pc_idx]
        loadings = pca_results['components'][pc_idx]
        var_explained = pca_results['explained_variance_ratio'][pc_idx] * 100
        
        # Plot loadings
        ax.plot(bin_centers, loadings, 'k-', linewidth=2)
        ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        # Fill positive and negative regions
        ax.fill_between(bin_centers, loadings, 0, 
                        where=(loadings > 0), alpha=0.3, color='blue', label='Positive')
        ax.fill_between(bin_centers, loadings, 0, 
                        where=(loadings < 0), alpha=0.3, color='red', label='Negative')
        
        # Mark landmark positions
        for i, lm_pos in enumerate(landmark_positions):
            ax.axvline(lm_pos, color='green', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.text(lm_pos, ax.get_ylim()[1], f'L{i+1}', ha='center', va='bottom', 
                   fontsize=9, color='green')
        
        ax.set_xlabel('Position (cm)', fontsize=10)
        ax.set_ylabel('Loading', fontsize=10)
        ax.set_title(f'PC{pc_idx+1} ({var_explained:.1f}% variance)', 
                    fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        
        if pc_idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Principal Component Loadings\n(What spatial patterns does each PC capture?)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved loadings plot: {os.path.basename(save_path)}")
    
    return fig


# ============================================================================
# VISUALIZATION: PC SCATTER PLOTS
# ============================================================================

def plot_pc_scatter_by_session(pca_results, session_labels, save_path=None):
    """
    Scatter plot of cells in PC1-PC2 space, colored by session.
    """
    pc_scores = pca_results['pc_scores']
    var1 = pca_results['explained_variance_ratio'][0] * 100
    var2 = pca_results['explained_variance_ratio'][1] * 100
    
    # Get unique sessions and sort by day number
    unique_sessions = np.unique(session_labels)
    unique_sessions = sorted(unique_sessions, key=lambda x: int(x.replace('Day', '')))
    
    # Create colormap
    n_sessions = len(unique_sessions)
    colors = plt.cm.viridis(np.linspace(0, 1, n_sessions))
    session_colors = {session: colors[i] for i, session in enumerate(unique_sessions)}
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: PC1 vs PC2
    ax1 = axes[0]
    for session in unique_sessions:
        mask = session_labels == session
        ax1.scatter(pc_scores[mask, 0], pc_scores[mask, 1], 
                   c=[session_colors[session]], label=session, 
                   alpha=0.6, s=30, edgecolors='none')
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
    ax1.set_title('PC1 vs PC2 by Session', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    # Right: PC1 vs PC3
    var3 = pca_results['explained_variance_ratio'][2] * 100
    ax2 = axes[1]
    for session in unique_sessions:
        mask = session_labels == session
        ax2.scatter(pc_scores[mask, 0], pc_scores[mask, 2], 
                   c=[session_colors[session]], label=session, 
                   alpha=0.6, s=30, edgecolors='none')
    
    ax2.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({var3:.1f}%)', fontsize=11)
    ax2.set_title('PC1 vs PC3 by Session', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved session scatter: {os.path.basename(save_path)}")
    
    return fig


def plot_pc_scatter_by_layer(pca_results, layer_labels, save_path=None):
    """
    Scatter plot of cells in PC1-PC2 space, colored by cortical layer.
    """
    pc_scores = pca_results['pc_scores']
    var1 = pca_results['explained_variance_ratio'][0] * 100
    var2 = pca_results['explained_variance_ratio'][1] * 100
    var3 = pca_results['explained_variance_ratio'][2] * 100
    
    # Layer colors (consistent with your other analyses)
    layer_colors = {
        'L2/3': '#1E88E5',  # Blue
        'L4': '#FF9800',    # Orange
        'L5': '#4CAF50',    # Green
        'L6': '#E53935',    # Red
        'Unknown': 'gray'
    }
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: PC1 vs PC2
    ax1 = axes[0]
    for layer in layer_order:
        mask = layer_labels == layer
        if np.sum(mask) > 0:
            ax1.scatter(pc_scores[mask, 0], pc_scores[mask, 1], 
                       c=layer_colors.get(layer, 'gray'), label=f'{layer} (n={np.sum(mask)})', 
                       alpha=0.6, s=30, edgecolors='none')
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
    ax1.set_title('PC1 vs PC2 by Layer', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    # Right: PC1 vs PC3
    ax2 = axes[1]
    for layer in layer_order:
        mask = layer_labels == layer
        if np.sum(mask) > 0:
            ax2.scatter(pc_scores[mask, 0], pc_scores[mask, 2], 
                       c=layer_colors.get(layer, 'gray'), label=f'{layer} (n={np.sum(mask)})', 
                       alpha=0.6, s=30, edgecolors='none')
    
    ax2.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({var3:.1f}%)', fontsize=11)
    ax2.set_title('PC1 vs PC3 by Layer', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved layer scatter: {os.path.basename(save_path)}")
    
    return fig


def plot_pc_scatter_by_landmark(pca_results, preferred_landmark, landmark_positions, 
                                save_path=None):
    """
    Scatter plot of cells in PC1-PC2 space, colored by landmark preference.
    """
    pc_scores = pca_results['pc_scores']
    var1 = pca_results['explained_variance_ratio'][0] * 100
    var2 = pca_results['explained_variance_ratio'][1] * 100
    var3 = pca_results['explained_variance_ratio'][2] * 100
    
    # Landmark colors
    lm_colors = {
        0: '#e41a1c',   # L1 - Red
        1: '#377eb8',   # L2 - Blue
        2: '#4daf4a',   # L3 - Green
        3: '#984ea3',   # L4 - Purple
        -1: 'lightgray' # Between landmarks
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: PC1 vs PC2
    ax1 = axes[0]
    
    # Plot "between" cells first (in background)
    mask = preferred_landmark == -1
    if np.sum(mask) > 0:
        ax1.scatter(pc_scores[mask, 0], pc_scores[mask, 1], 
                   c='lightgray', label=f'Between (n={np.sum(mask)})', 
                   alpha=0.3, s=20, edgecolors='none')
    
    # Plot landmark-preferring cells
    for lm_idx in range(len(landmark_positions)):
        mask = preferred_landmark == lm_idx
        if np.sum(mask) > 0:
            ax1.scatter(pc_scores[mask, 0], pc_scores[mask, 1], 
                       c=lm_colors[lm_idx], 
                       label=f'L{lm_idx+1} ({landmark_positions[lm_idx]}cm, n={np.sum(mask)})', 
                       alpha=0.7, s=40, edgecolors='white', linewidths=0.5)
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
    ax1.set_title('PC1 vs PC2 by Landmark Preference', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    # Right: PC1 vs PC3
    ax2 = axes[1]
    
    mask = preferred_landmark == -1
    if np.sum(mask) > 0:
        ax2.scatter(pc_scores[mask, 0], pc_scores[mask, 2], 
                   c='lightgray', label=f'Between (n={np.sum(mask)})', 
                   alpha=0.3, s=20, edgecolors='none')
    
    for lm_idx in range(len(landmark_positions)):
        mask = preferred_landmark == lm_idx
        if np.sum(mask) > 0:
            ax2.scatter(pc_scores[mask, 0], pc_scores[mask, 2], 
                       c=lm_colors[lm_idx], 
                       label=f'L{lm_idx+1} ({landmark_positions[lm_idx]}cm, n={np.sum(mask)})', 
                       alpha=0.7, s=40, edgecolors='white', linewidths=0.5)
    
    ax2.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({var3:.1f}%)', fontsize=11)
    ax2.set_title('PC1 vs PC3 by Landmark Preference', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax2.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved landmark scatter: {os.path.basename(save_path)}")
    
    return fig


def plot_pc_scatter_by_peak_position(pca_results, peak_positions, save_path=None):
    """
    Scatter plot of cells in PC1-PC2 space, colored by peak position (continuous).
    """
    pc_scores = pca_results['pc_scores']
    var1 = pca_results['explained_variance_ratio'][0] * 100
    var2 = pca_results['explained_variance_ratio'][1] * 100
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    scatter = ax.scatter(pc_scores[:, 0], pc_scores[:, 1], 
                        c=peak_positions, cmap='viridis',
                        alpha=0.6, s=30, edgecolors='none')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Peak Position (cm)', fontsize=11)
    
    ax.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
    ax.set_title('PC1 vs PC2 by Peak Position', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved peak position scatter: {os.path.basename(save_path)}")
    
    return fig


# ============================================================================
# VISUALIZATION: L1-SPECIFIC ANALYSIS
# ============================================================================

def plot_l1_cells_detailed(pca_results, preferred_landmark, session_labels,
                           layer_labels, spatial_profiles, bin_centers,
                           landmark_positions, save_path=None):
    """
    Detailed analysis of L1-preferring cells specifically.
    This is where we look for adaptation vs. true spatial encoding subgroups.
    """
    pc_scores = pca_results['pc_scores']
    var1 = pca_results['explained_variance_ratio'][0] * 100
    var2 = pca_results['explained_variance_ratio'][1] * 100
    var3 = pca_results['explained_variance_ratio'][2] * 100
    
    # Get L1 cells
    l1_mask = preferred_landmark == 0
    n_l1 = np.sum(l1_mask)
    
    if n_l1 == 0:
        print("  No L1 cells found!")
        return None
    
    print(f"\n  Analyzing {n_l1} L1-preferring cells...")
    
    l1_scores = pc_scores[l1_mask]
    l1_profiles = spatial_profiles[l1_mask]
    l1_sessions = session_labels[l1_mask]
    l1_layers = layer_labels[l1_mask]
    
    fig = plt.figure(figsize=(16, 12))
    
    # Panel 1: L1 cells in PC space, colored by session
    ax1 = fig.add_subplot(2, 3, 1)
    unique_sessions = sorted(np.unique(l1_sessions), key=lambda x: int(x.replace('Day', '')))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_sessions)))
    
    for i, session in enumerate(unique_sessions):
        mask = l1_sessions == session
        ax1.scatter(l1_scores[mask, 0], l1_scores[mask, 1], 
                   c=[colors[i]], label=session, alpha=0.7, s=50)
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}%)')
    ax1.set_ylabel(f'PC2 ({var2:.1f}%)')
    ax1.set_title('L1 Cells by Session', fontweight='bold')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)
    
    # Panel 2: L1 cells in PC space, colored by layer
    ax2 = fig.add_subplot(2, 3, 2)
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    
    for layer in ['L2/3', 'L4', 'L5', 'L6']:
        mask = l1_layers == layer
        if np.sum(mask) > 0:
            ax2.scatter(l1_scores[mask, 0], l1_scores[mask, 1], 
                       c=layer_colors.get(layer, 'gray'), 
                       label=f'{layer} (n={np.sum(mask)})', alpha=0.7, s=50)
    
    ax2.set_xlabel(f'PC1 ({var1:.1f}%)')
    ax2.set_ylabel(f'PC2 ({var2:.1f}%)')
    ax2.set_title('L1 Cells by Layer', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    
    # Panel 3: PC2 vs PC3 for L1 cells
    ax3 = fig.add_subplot(2, 3, 3)
    scatter = ax3.scatter(l1_scores[:, 1], l1_scores[:, 2], 
                         c=l1_scores[:, 0], cmap='coolwarm', alpha=0.7, s=50)
    plt.colorbar(scatter, ax=ax3, label='PC1 score')
    ax3.set_xlabel(f'PC2 ({var2:.1f}%)')
    ax3.set_ylabel(f'PC3 ({var3:.1f}%)')
    ax3.set_title('L1 Cells: PC2 vs PC3', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # Panel 4: Average profile of L1 cells with high vs low PC2
    ax4 = fig.add_subplot(2, 3, 4)
    
    pc2_median = np.median(l1_scores[:, 1])
    high_pc2 = l1_scores[:, 1] > pc2_median
    low_pc2 = l1_scores[:, 1] <= pc2_median
    
    mean_high = np.mean(l1_profiles[high_pc2], axis=0)
    mean_low = np.mean(l1_profiles[low_pc2], axis=0)
    sem_high = np.std(l1_profiles[high_pc2], axis=0) / np.sqrt(np.sum(high_pc2))
    sem_low = np.std(l1_profiles[low_pc2], axis=0) / np.sqrt(np.sum(low_pc2))
    
    ax4.plot(bin_centers, mean_high, 'b-', linewidth=2, label=f'High PC2 (n={np.sum(high_pc2)})')
    ax4.fill_between(bin_centers, mean_high - sem_high, mean_high + sem_high, alpha=0.2, color='blue')
    ax4.plot(bin_centers, mean_low, 'r-', linewidth=2, label=f'Low PC2 (n={np.sum(low_pc2)})')
    ax4.fill_between(bin_centers, mean_low - sem_low, mean_low + sem_low, alpha=0.2, color='red')
    
    for lm_pos in landmark_positions:
        ax4.axvline(lm_pos, color='green', linestyle='--', alpha=0.5)
    
    ax4.set_xlabel('Position (cm)')
    ax4.set_ylabel('Z-scored Activity')
    ax4.set_title('L1 Cells: High vs Low PC2', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Panel 5: Average profile of L1 cells with high vs low PC3
    ax5 = fig.add_subplot(2, 3, 5)
    
    pc3_median = np.median(l1_scores[:, 2])
    high_pc3 = l1_scores[:, 2] > pc3_median
    low_pc3 = l1_scores[:, 2] <= pc3_median
    
    mean_high = np.mean(l1_profiles[high_pc3], axis=0)
    mean_low = np.mean(l1_profiles[low_pc3], axis=0)
    sem_high = np.std(l1_profiles[high_pc3], axis=0) / np.sqrt(np.sum(high_pc3))
    sem_low = np.std(l1_profiles[low_pc3], axis=0) / np.sqrt(np.sum(low_pc3))
    
    ax5.plot(bin_centers, mean_high, 'b-', linewidth=2, label=f'High PC3 (n={np.sum(high_pc3)})')
    ax5.fill_between(bin_centers, mean_high - sem_high, mean_high + sem_high, alpha=0.2, color='blue')
    ax5.plot(bin_centers, mean_low, 'r-', linewidth=2, label=f'Low PC3 (n={np.sum(low_pc3)})')
    ax5.fill_between(bin_centers, mean_low - sem_low, mean_low + sem_low, alpha=0.2, color='red')
    
    for lm_pos in landmark_positions:
        ax5.axvline(lm_pos, color='green', linestyle='--', alpha=0.5)
    
    ax5.set_xlabel('Position (cm)')
    ax5.set_ylabel('Z-scored Activity')
    ax5.set_title('L1 Cells: High vs Low PC3', fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # Panel 6: Heatmap of all L1 cell profiles, sorted by PC2
    ax6 = fig.add_subplot(2, 3, 6)
    
    sort_idx = np.argsort(l1_scores[:, 1])  # Sort by PC2
    sorted_profiles = l1_profiles[sort_idx]
    
    im = ax6.imshow(sorted_profiles, aspect='auto', cmap='viridis',
                   extent=[bin_centers[0], bin_centers[-1], 0, n_l1])
    
    for lm_pos in landmark_positions:
        ax6.axvline(lm_pos, color='red', linestyle='--', alpha=0.7)
    
    ax6.set_xlabel('Position (cm)')
    ax6.set_ylabel('Cell # (sorted by PC2)')
    ax6.set_title('L1 Cell Profiles (sorted by PC2)', fontweight='bold')
    plt.colorbar(im, ax=ax6, label='Z-scored Activity')
    
    plt.suptitle(f'Detailed Analysis of L1-Preferring Cells (n={n_l1})', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved L1 analysis: {os.path.basename(save_path)}")
    
    return fig


# ============================================================================
# SAVE PCA RESULTS
# ============================================================================

def save_pca_results(pca_data_path, pca_results):
    """
    Save PCA results back to the HDF5 file.
    """
    print(f"\nSaving PCA results to: {pca_data_path}")
    
    with h5py.File(pca_data_path, 'a') as f:
        # Remove old results if they exist
        if 'pca_results' in f:
            del f['pca_results']
        
        pca_grp = f.create_group('pca_results')
        pca_grp.create_dataset('pc_scores', data=pca_results['pc_scores'])
        pca_grp.create_dataset('components', data=pca_results['components'])
        pca_grp.create_dataset('explained_variance', data=pca_results['explained_variance'])
        pca_grp.create_dataset('explained_variance_ratio', data=pca_results['explained_variance_ratio'])
        pca_grp.create_dataset('cumulative_variance_ratio', data=pca_results['cumulative_variance_ratio'])
        pca_grp.attrs['n_components'] = pca_results['n_components']
    
    print("  ✓ PCA results saved!")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_pca_analysis(pca_data_path, figure_dir, n_components=10):
    """
    Main function to run complete PCA analysis.
    """
    print("=" * 80)
    print("PCA ANALYSIS")
    print("=" * 80)
    
    # Create figure directory
    os.makedirs(figure_dir, exist_ok=True)
    
    # Load data
    data = load_pca_data(pca_data_path)
    
    # Run PCA
    pca_results = run_pca(data['spatial_profiles_zscore'], n_components=n_components)
    
    # Generate figures
    print("\nGenerating figures...")
    
    # Scree plot
    plot_scree(pca_results, 
              save_path=os.path.join(figure_dir, 'pca_scree_plot.png'))
    
    # PC Loadings
    plot_pc_loadings(pca_results, data['bin_centers'], data['landmark_positions'],
                    save_path=os.path.join(figure_dir, 'pca_loadings.png'))
    
    # Scatter by session
    plot_pc_scatter_by_session(pca_results, data['session_labels'],
                               save_path=os.path.join(figure_dir, 'pca_scatter_session.png'))
    
    # Scatter by layer
    plot_pc_scatter_by_layer(pca_results, data['layer_labels'],
                            save_path=os.path.join(figure_dir, 'pca_scatter_layer.png'))
    
    # Scatter by landmark
    plot_pc_scatter_by_landmark(pca_results, data['preferred_landmark'], 
                                data['landmark_positions'],
                                save_path=os.path.join(figure_dir, 'pca_scatter_landmark.png'))
    
    # Scatter by peak position
    plot_pc_scatter_by_peak_position(pca_results, data['peak_positions'],
                                     save_path=os.path.join(figure_dir, 'pca_scatter_peak.png'))
    
    # L1-specific analysis
    plot_l1_cells_detailed(pca_results, data['preferred_landmark'],
                           data['session_labels'], data['layer_labels'],
                           data['spatial_profiles_zscore'], data['bin_centers'],
                           data['landmark_positions'],
                           save_path=os.path.join(figure_dir, 'pca_l1_detailed.png'))
    
    # Save PCA results
    save_pca_results(pca_data_path, pca_results)
    
    print("\n" + "=" * 80)
    print("PCA ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print(f"PCA results saved to: {pca_data_path}")
    
    # Show all figures
    plt.show()
    
    return pca_results, data


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pca_results, data = run_pca_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR,
        n_components=N_COMPONENTS
    )