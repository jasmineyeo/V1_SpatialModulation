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
PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5"

# Output directory for figures
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\figures_aligned"

# Number of PCs to compute and analyze
N_COMPONENTS = 10

# Use landmark-aligned profiles? (requires running PCA_LandmarkAlignedAnalysis.py first)
USE_ALIGNED_PROFILES = True

# Figure settings
FIGSIZE_SCREE = (10, 4)
FIGSIZE_LOADINGS = (14, 8)
FIGSIZE_SCATTER = (12, 10)
DPI = 150


def load_pca_data(filepath, use_aligned=False):
    """
    Load aggregated PCA data from HDF5 file.

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

        # Determine which profiles to use for PCA
        if use_aligned and 'features/spatial_profiles_aligned' in f:
            data['spatial_profiles_zscore'] = f['features/spatial_profiles_aligned'][:]
            # Load alignment metadata
            if 'alignment' in f:
                data['alignment_shifts'] = f['alignment/optimal_shifts'][:]
                data['alignment_correlations'] = f['alignment/max_correlations'][:]
                sigma = f['alignment'].attrs.get('template_sigma_cm', 'Unknown')
                print(f"  Using LANDMARK-ALIGNED profiles (template sigma={sigma}cm)")
            else:
                print("  Using LANDMARK-ALIGNED profiles")
        elif use_aligned:
            print("  WARNING: Aligned profiles requested but not found!")
            print("           Run PCA_LandmarkAlignedAnalysis.py first.")
            print("           Falling back to session-corrected/z-scored profiles.")
            use_aligned = False

        if not use_aligned:
            if 'features/spatial_profiles_session_corrected' in f:
                data['spatial_profiles_zscore'] = f['features/spatial_profiles_session_corrected'][:]
                correction_method = f['features'].attrs.get('correction_method', 'Unknown')
                print(f"  Using session-corrected profiles ({correction_method})")
            else:
                data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
                print("  Using original z-scored profiles (no session correction)")

    print(f"  Animal: {data['animal_id']}")
    print(f"  Cells: {data['n_cells']}")
    print(f"  Sessions: {data['n_sessions']}")
    print(f"  Features: {data['spatial_profiles'].shape[1]} spatial bins")

    return data


def run_pca(profiles_zscore, n_components=10):
    """
    Run PCA on z-scored spatial profiles.
    
    Returns:
    --------
    pca_results : dict
        Contains PCA object, scores, explained variance, etc.

    Fits sklearn's PCA with 10 components
    Transforms data to get PC scores (each cell's coordinates in PC space)
    
    Output dictionary contains:
    pc_scores	(n_cells, 10)	Each cell's position in PC space
    components	(10, 115)	The "loadings" — what spatial pattern each PC represents
    explained_variance_ratio	(10,)	Fraction of variance explained by each PC

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

def plot_pc_scatter_by_layer(pca_results, layer_labels, subsample=False, save_path=None):
    """
    Scatter plot of cells in PC1-PC2 space, colored by cortical layer.
    
    Parameters:
    -----------
    subsample : bool
        If True, subsample each layer to equal size (smallest layer) for fair visualization
    """
    pc_scores = pca_results['pc_scores']
    var1 = pca_results['explained_variance_ratio'][0] * 100
    var2 = pca_results['explained_variance_ratio'][1] * 100
    var3 = pca_results['explained_variance_ratio'][2] * 100
    
    # Layer colors
    layer_colors = {
        'L2/3': '#1E88E5',
        'L4': '#FF9800',
        'L5': '#4CAF50',
        'L6': '#E53935',
        'Unknown': 'gray'
    }
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    # If subsampling, equalize layer sizes
    if subsample:
        # Find minimum layer size
        min_size = min([np.sum(layer_labels == layer) for layer in layer_order])
        print(f"  Subsampling layers to n={min_size} cells each for balanced visualization")
        
        # Subsample each layer
        subsampled_indices = []
        for layer in layer_order:
            layer_indices = np.where(layer_labels == layer)[0]
            if len(layer_indices) > 0:
                sampled = np.random.choice(layer_indices, 
                                          size=min(min_size, len(layer_indices)), 
                                          replace=False)
                subsampled_indices.extend(sampled)
        
        subsampled_indices = np.array(subsampled_indices)
        pc_scores_plot = pc_scores[subsampled_indices]
        layer_labels_plot = layer_labels[subsampled_indices]
        title_suffix = " (Equal Subsampling)"
    else:
        pc_scores_plot = pc_scores
        layer_labels_plot = layer_labels
        title_suffix = ""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: PC1 vs PC2
    ax1 = axes[0]
    for layer in layer_order:
        mask = layer_labels_plot == layer
        if np.sum(mask) > 0:
            ax1.scatter(pc_scores_plot[mask, 0], pc_scores_plot[mask, 1], 
                       c=layer_colors.get(layer, 'gray'), label=f'{layer} (n={np.sum(mask)})', 
                       alpha=0.6, s=30, edgecolors='none')
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
    ax1.set_title(f'PC1 vs PC2 by Layer{title_suffix}', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
    ax1.axvline(0, color='gray', linestyle='-', alpha=0.3)
    
    # Right: PC1 vs PC3
    ax2 = axes[1]
    for layer in layer_order:
        mask = layer_labels_plot == layer
        if np.sum(mask) > 0:
            ax2.scatter(pc_scores_plot[mask, 0], pc_scores_plot[mask, 2], 
                       c=layer_colors.get(layer, 'gray'), label=f'{layer} (n={np.sum(mask)})', 
                       alpha=0.6, s=30, edgecolors='none')
    
    ax2.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax2.set_ylabel(f'PC3 ({var3:.1f}%)', fontsize=11)
    ax2.set_title(f'PC1 vs PC3 by Layer{title_suffix}', fontsize=12, fontweight='bold')
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

def plot_layer_session_detailed_grid(pca_results, session_labels, layer_labels,
                                     landmark_positions, save_path=None):
    """
    Grid of PC1 histograms: rows = layers, columns = sessions.
    This directly visualizes layer-specific learning trajectories.
    
    Each subplot shows the distribution of PC1 scores for one layer in one session.
    Vertical lines mark expected landmark/reward positions in PC space.
    """
    pc1_scores = pca_results['pc_scores'][:, 0]
    
    # Get unique sessions and layers
    unique_sessions = sorted(np.unique(session_labels), 
                            key=lambda x: int(x.replace('Day', '')))
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    n_sessions = len(unique_sessions)
    n_layers = len(layer_order)
    
    # Create figure
    fig, axes = plt.subplots(n_layers, n_sessions, 
                             figsize=(3*n_sessions, 3*n_layers),
                             sharex=True, sharey=True)
    
    # Layer colors
    layer_colors = {
        'L2/3': '#1E88E5',
        'L4': '#FF9800',
        'L5': '#4CAF50',
        'L6': '#E53935'
    }
    
    # Common histogram settings
    bins = np.linspace(-10, 8, 40)
    
    # Plot each Layer × Session combination
    for layer_idx, layer in enumerate(layer_order):
        for session_idx, session in enumerate(unique_sessions):
            ax = axes[layer_idx, session_idx]
            
            # Get data for this layer and session
            mask = (layer_labels == layer) & (session_labels == session)
            pc1_subset = pc1_scores[mask]
            n_cells = len(pc1_subset)
            
            if n_cells > 0:
                # Plot histogram
                ax.hist(pc1_subset, bins=bins, color=layer_colors[layer], 
                       alpha=0.7, edgecolor='black', linewidth=0.5)
                
                # Mark mean
                mean_pc1 = np.mean(pc1_subset)
                ax.axvline(mean_pc1, color='red', linestyle='--', 
                          linewidth=2, label=f'Mean={mean_pc1:.2f}')
                
                # Mark reference zones
                ax.axvline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
                ax.axvspan(-10, -2, alpha=0.1, color='blue', label='Reward Zone')
                ax.axvspan(-2, 8, alpha=0.1, color='green', label='Track/Landmarks')
            
            # Titles and labels
            if layer_idx == 0:
                ax.set_title(f'{session}', fontsize=11, fontweight='bold')
            if session_idx == 0:
                ax.set_ylabel(f'{layer}\n(n={n_cells})', fontsize=10, fontweight='bold')
            if layer_idx == n_layers - 1:
                ax.set_xlabel('PC1 Score', fontsize=9)
            
            # Add cell count
            ax.text(0.95, 0.95, f'n={n_cells}', transform=ax.transAxes,
                   fontsize=9, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.grid(alpha=0.3, axis='y')
    
    plt.suptitle('PC1 Distribution by Layer Across Sessions\n' +
                '(Blue zone = Reward, Green zone = Track/Landmarks)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved layer-session grid: {os.path.basename(save_path)}")
    
    return fig

def plot_landmark_preference_evolution(pca_results, preferred_landmark, 
                                       session_labels, layer_labels,
                                       landmark_positions, save_path=None):
    """
    Line plot showing how landmark preferences evolve across sessions,
    comparing superficial vs. deep layers.
    
    This directly tests: Do superficial and deep layers show different 
    learning trajectories in their landmark preferences?
    """
    pc1_scores = pca_results['pc_scores'][:, 0]
    
    # Get unique sessions
    unique_sessions = sorted(np.unique(session_labels), 
                            key=lambda x: int(x.replace('Day', '')))
    
    # Define layer groups
    superficial_mask = (layer_labels == 'L2/3') | (layer_labels == 'L4')
    deep_mask = (layer_labels == 'L5') | (layer_labels == 'L6')
    
    # Define reward zone (PC1 < -2)
    reward_zone_threshold = -2.0
    
    # Calculate proportions for each session
    session_nums = np.arange(1, len(unique_sessions) + 1)
    
    # Storage for proportions
    data = {
        'Superficial': {'L1': [], 'L4': [], 'Reward': [], 'Other': []},
        'Deep': {'L1': [], 'L4': [], 'Reward': [], 'Other': []}
    }
    
    for session in unique_sessions:
        for layer_group, mask_group in [('Superficial', superficial_mask), 
                                        ('Deep', deep_mask)]:
            session_mask = (session_labels == session) & mask_group
            n_total = np.sum(session_mask)
            
            if n_total > 0:
                # Count each category
                n_l1 = np.sum((preferred_landmark == 0) & session_mask)
                n_l4 = np.sum((preferred_landmark == 3) & session_mask)
                n_reward = np.sum((pc1_scores < reward_zone_threshold) & session_mask)
                n_other = n_total - n_l1 - n_l4 - n_reward
                
                # Store proportions
                data[layer_group]['L1'].append(n_l1 / n_total * 100)
                data[layer_group]['L4'].append(n_l4 / n_total * 100)
                # data[layer_group]['Reward'].append(n_reward / n_total * 100)
                data[layer_group]['Other'].append(n_other / n_total * 100)
            else:
                # No cells in this group for this session
                data[layer_group]['L1'].append(0)
                data[layer_group]['L4'].append(0)
                # data[layer_group]['Reward'].append(0)
                data[layer_group]['Other'].append(0)
    
    # Convert to arrays
    for group in data:
        for category in data[group]:
            data[group][category] = np.array(data[group][category])
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot for Superficial layers
    ax1 = axes[0]
    ax1.plot(session_nums, data['Superficial']['L1'], 'o-', 
            color='#e41a1c', linewidth=2.5, markersize=8, label='L1 (25cm)')
    ax1.plot(session_nums, data['Superficial']['L4'], 's-', 
            color='#984ea3', linewidth=2.5, markersize=8, label='L4 (115cm)')
    # ax1.plot(session_nums, data['Superficial']['Reward'], '^-', 
    #         color='#377eb8', linewidth=2.5, markersize=8, label='Reward Zone')
    # ax1.plot(session_nums, data['Superficial']['Other'], 'd-', 
    #         color='gray', linewidth=2, markersize=7, label='Other', alpha=0.5)
    
    ax1.set_xlabel('Session (Day)', fontsize=12)
    ax1.set_ylabel('Proportion of Cells (%)', fontsize=12)
    ax1.set_title('Superficial Layers (L2/3 + L4)', fontsize=13, fontweight='bold')
    ax1.set_xticks(session_nums)
    ax1.set_xticklabels([s.replace('Day', '') for s in unique_sessions])
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)
    
    # Plot for Deep layers
    ax2 = axes[1]
    ax2.plot(session_nums, data['Deep']['L1'], 'o-', 
            color='#e41a1c', linewidth=2.5, markersize=8, label='L1 (25cm)')
    ax2.plot(session_nums, data['Deep']['L4'], 's-', 
            color='#984ea3', linewidth=2.5, markersize=8, label='L4 (115cm)')
    # ax2.plot(session_nums, data['Deep']['Reward'], '^-', 
    #         color='#377eb8', linewidth=2.5, markersize=8, label='Reward Zone')
    # ax2.plot(session_nums, data['Deep']['Other'], 'd-', 
    #         color='gray', linewidth=2, markersize=7, label='Other', alpha=0.5)
    
    ax2.set_xlabel('Session (Day)', fontsize=12)
    ax2.set_ylabel('Proportion of Cells (%)', fontsize=12)
    ax2.set_title('Deep Layers (L5 + L6)', fontsize=13, fontweight='bold')
    ax2.set_xticks(session_nums)
    ax2.set_xticklabels([s.replace('Day', '') for s in unique_sessions])
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(alpha=0.3)
    
    # Add hypothesis expectations as text box
    # textstr = ('Your Hypothesis:\n'
    #           'Superficial: L1 high early → shift to L4/Reward\n'
    #           'Deep: L4/Reward higher from start → faster stabilization')
    # fig.text(0.5, -0.05, textstr, ha='center', fontsize=10,
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Landmark Preference Evolution: Superficial vs. Deep Layers',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved landmark evolution: {os.path.basename(save_path)}")
    
    # Print summary
    print("\n  Landmark Preference Evolution Summary:")
    print("  " + "="*60)
    print(f"  SUPERFICIAL LAYERS:")
    print(f"    Day 1: L1={data['Superficial']['L1'][0]:.1f}%, " +
          f"L4={data['Superficial']['L4'][0]:.1f}%, ")
    print(f"    Day 7: L1={data['Superficial']['L1'][-1]:.1f}%, " +
          f"L4={data['Superficial']['L4'][-1]:.1f}%, " )
    
    print(f"  DEEP LAYERS:")
    print(f"    Day 1: L1={data['Deep']['L1'][0]:.1f}%, " +
          f"L4={data['Deep']['L4'][0]:.1f}%, " )
    print(f"    Day 7: L1={data['Deep']['L1'][-1]:.1f}%, " +
          f"L4={data['Deep']['L4'][-1]:.1f}%, ")
    
    return fig

def plot_pc1_trajectory_layer_session(pca_results, session_labels, layer_labels, 
                                      save_path=None):
    """
    Line plot showing mean PC1 score over sessions, separately for each layer.
    This directly tests the hypothesis of layer-specific learning trajectories.
    """
    pc_scores = pca_results['pc_scores']
    pc1_scores = pc_scores[:, 0]
    
    # Get unique sessions and layers
    unique_sessions = sorted(np.unique(session_labels), 
                            key=lambda x: int(x.replace('Day', '')))
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    # Layer colors
    layer_colors = {
        'L2/3': '#1E88E5',  # Blue
        'L4': '#FF9800',    # Orange
        'L5': '#4CAF50',    # Green
        'L6': '#E53935'     # Red
    }
    
    # Calculate mean PC1 and SEM for each Layer × Session
    layer_session_means = {}
    layer_session_sems = {}
    layer_session_ns = {}
    
    for layer in layer_order:
        means = []
        sems = []
        ns = []
        
        for session in unique_sessions:
            mask = (layer_labels == layer) & (session_labels == session)
            n = np.sum(mask)
            
            if n > 0:
                pc1_subset = pc1_scores[mask]
                means.append(np.mean(pc1_subset))
                sems.append(np.std(pc1_subset) / np.sqrt(n))
                ns.append(n)
            else:
                means.append(np.nan)
                sems.append(np.nan)
                ns.append(0)
        
        layer_session_means[layer] = np.array(means)
        layer_session_sems[layer] = np.array(sems)
        layer_session_ns[layer] = np.array(ns)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # X-axis: session numbers
    session_nums = np.arange(1, len(unique_sessions) + 1)
    
    # Plot each layer
    for layer in layer_order:
        means = layer_session_means[layer]
        sems = layer_session_sems[layer]
        
        # Plot line with error bars
        ax.plot(session_nums, means, 'o-', 
               color=layer_colors[layer], linewidth=2.5, 
               markersize=8, label=layer, alpha=0.8)
        ax.fill_between(session_nums, means - sems, means + sems, 
                       color=layer_colors[layer], alpha=0.2)
    
    # Formatting
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Session (Day)', fontsize=12)
    ax.set_ylabel('Mean PC1 Score', fontsize=12)
    ax.set_title('PC1 Trajectory by Layer Across Sessions\n' + 
                '(Test of Layer-Specific Learning)', 
                fontsize=13, fontweight='bold')
    ax.set_xticks(session_nums)
    ax.set_xticklabels([s.replace('Day', '') for s in unique_sessions])
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3)
    
    # Add interpretation guide as text
    textstr = 'Interpretation:\nPositive PC1 = Track/Landmark encoding\nNegative PC1 = Reward zone encoding'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved PC1 trajectory: {os.path.basename(save_path)}")
    
    # Print summary statistics
    print("\n  PC1 Trajectory Summary:")
    print("  " + "="*60)
    for layer in layer_order:
        day1_mean = layer_session_means[layer][0]
        day7_mean = layer_session_means[layer][-1]
        shift = day7_mean - day1_mean
        print(f"  {layer}: Day1={day1_mean:+.2f}, Day7={day7_mean:+.2f}, " +
              f"Shift={shift:+.2f}")
    
    return fig

def plot_landmark_proportions_layer_session(pca_results, preferred_landmark, 
                                           session_labels, layer_labels,
                                           landmark_positions, save_path=None):
    """
    Stacked bar chart showing proportion of cells preferring each landmark,
    by layer and session.
    """
    pc_scores = pca_results['pc_scores']
    
    # Get unique sessions and layers
    unique_sessions = sorted(np.unique(session_labels), 
                            key=lambda x: int(x.replace('Day', '')))
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    
    # Define categories
    # L1, L4, and "Reward Zone" (defined as cells with PC1 < -2)
    pc1_scores = pc_scores[:, 0]
    reward_zone_threshold = -2.0
    
    # Calculate proportions for each Layer × Session
    n_sessions = len(unique_sessions)
    n_layers = len(layer_order)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for layer_idx, layer in enumerate(layer_order):
        ax = axes[layer_idx]
        
        prop_l1 = []
        prop_l4 = []
        prop_reward = []
        prop_other = []
        
        for session in unique_sessions:
            mask = (layer_labels == layer) & (session_labels == session)
            n_total = np.sum(mask)
            
            if n_total > 0:
                # L1 cells
                n_l1 = np.sum((preferred_landmark == 0) & mask)
                # L4 cells
                n_l4 = np.sum((preferred_landmark == 3) & mask)
                # Reward zone cells (PC1 < threshold)
                n_reward = np.sum((pc1_scores < reward_zone_threshold) & mask)
                # Other (everything else)
                n_other = n_total - n_l1 - n_l4 - n_reward
                
                prop_l1.append(n_l1 / n_total * 100)
                prop_l4.append(n_l4 / n_total * 100)
                prop_reward.append(n_reward / n_total * 100)
                prop_other.append(n_other / n_total * 100)
            else:
                prop_l1.append(0)
                prop_l4.append(0)
                prop_reward.append(0)
                prop_other.append(0)
        
        # Create stacked bar chart
        session_nums = np.arange(len(unique_sessions))
        width = 0.6
        
        ax.bar(session_nums, prop_l1, width, label='L1 (25cm)', 
              color='#e41a1c', alpha=0.8)
        ax.bar(session_nums, prop_l4, width, bottom=prop_l1, 
              label='L4 (115cm)', color='#984ea3', alpha=0.8)
        ax.bar(session_nums, prop_reward, width, 
              bottom=np.array(prop_l1) + np.array(prop_l4),
              label='Reward Zone (PC1<-2)', color='#377eb8', alpha=0.8)
        ax.bar(session_nums, prop_other, width, 
              bottom=np.array(prop_l1) + np.array(prop_l4) + np.array(prop_reward),
              label='Other', color='lightgray', alpha=0.6)
        
        # Formatting
        ax.set_xlabel('Session (Day)', fontsize=11)
        ax.set_ylabel('Proportion of Cells (%)', fontsize=11)
        ax.set_title(f'{layer}', fontsize=12, fontweight='bold')
        ax.set_xticks(session_nums)
        ax.set_xticklabels([s.replace('Day', '') for s in unique_sessions])
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Landmark Preference Proportions by Layer Across Sessions',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved landmark proportions: {os.path.basename(save_path)}")
    
    return fig

def test_layer_session_interaction(pca_results, session_labels, layer_labels):
    """
    Statistical test for Layer × Session interaction on PC1 scores.
    Uses 2-way ANOVA.
    """
    try:
        from scipy import stats
        
        pc1_scores = pca_results['pc_scores'][:, 0]
        
        # Prepare data for ANOVA
        # Convert session labels to numeric
        unique_sessions = sorted(np.unique(session_labels), 
                                key=lambda x: int(x.replace('Day', '')))
        session_numeric = np.array([unique_sessions.index(s) + 1 
                                   for s in session_labels])
        
        # Create dataframe-like structure
        data_dict = {
            'PC1': pc1_scores,
            'Layer': layer_labels,
            'Session': session_numeric
        }
        
        print("\n" + "="*80)
        print("STATISTICAL TEST: Layer × Session Interaction")
        print("="*80)
        
        # Two-way ANOVA using scipy
        layer_order = ['L2/3', 'L4', 'L5', 'L6']
        
        # Test main effect of Layer
        layer_groups = [pc1_scores[layer_labels == layer] for layer in layer_order]
        f_layer, p_layer = stats.f_oneway(*layer_groups)
        
        print(f"\nMain Effect of Layer:")
        print(f"  F = {f_layer:.3f}, p = {p_layer:.4f}")
        if p_layer < 0.05:
            print(f"  ✓ Significant effect of Layer on PC1")
        else:
            print(f"  ✗ No significant effect of Layer on PC1")
        
        # Test main effect of Session
        session_groups = [pc1_scores[session_labels == s] for s in unique_sessions]
        f_session, p_session = stats.f_oneway(*session_groups)
        
        print(f"\nMain Effect of Session:")
        print(f"  F = {f_session:.3f}, p = {p_session:.4f}")
        if p_session < 0.05:
            print(f"  ✓ Significant effect of Session on PC1")
        else:
            print(f"  ✗ No significant effect of Session on PC1")
        
        # For interaction, we need a proper 2-way ANOVA
        # Simple approach: test if slope differs between layers
        print(f"\nTesting Layer × Session Interaction:")
        print(f"  (Comparing slopes of PC1 change across sessions)")
        
        slopes = {}
        for layer in layer_order:
            layer_means = []
            for session_num in range(len(unique_sessions)):
                mask = (layer_labels == layer) & (session_numeric == session_num + 1)
                if np.sum(mask) > 0:
                    layer_means.append(np.mean(pc1_scores[mask]))
                else:
                    layer_means.append(np.nan)
            
            # Linear fit to get slope
            x = np.arange(len(unique_sessions))
            valid = ~np.isnan(layer_means)
            if np.sum(valid) > 1:
                slope, _ = np.polyfit(x[valid], np.array(layer_means)[valid], 1)
                slopes[layer] = slope
                print(f"  {layer}: slope = {slope:+.3f} PC1 units/session")
        
        print("\nInterpretation:")
        print("  Negative slope = shift toward reward zone encoding")
        print("  Positive slope = shift toward landmark encoding")
        print("  Your hypothesis predicts:")
        print("    - Superficial (L2/3, L4): Negative slope (toward reward)")
        print("    - Deep (L5, L6): Flatter slope (already at reward)")
        
        print("="*80 + "\n")
        
        return slopes
        
    except ImportError:
        print("  ⚠ scipy not available, skipping statistical tests")
        return None
    
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


def run_pca_analysis(pca_data_path, figure_dir, n_components=10, use_aligned=False):
    """
    Main function to run complete PCA analysis.

    Parameters:
    -----------
    use_aligned : bool
        If True, use landmark-aligned profiles for PCA
    """
    print("=" * 80)
    print("PCA ANALYSIS")
    if use_aligned:
        print("*** USING LANDMARK-ALIGNED PROFILES ***")
    print("=" * 80)

    # Create figure directory
    os.makedirs(figure_dir, exist_ok=True)

    # Load data
    data = load_pca_data(pca_data_path, use_aligned=use_aligned)
    
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
    
    # Scatter by layer (subsampled for balanced visualization)
    plot_pc_scatter_by_layer(pca_results, data['layer_labels'], subsample=True,
                            save_path=os.path.join(figure_dir, 'pca_scatter_layer_balanced.png'))
    
    # Scatter by landmark
    plot_pc_scatter_by_landmark(pca_results, data['preferred_landmark'], 
                                data['landmark_positions'],
                                save_path=os.path.join(figure_dir, 'pca_scatter_landmark.png'))
    
    # Scatter by peak position
    plot_pc_scatter_by_peak_position(pca_results, data['peak_positions'],
                                     save_path=os.path.join(figure_dir, 'pca_scatter_peak.png'))
    

    # 1. PC1 trajectory by layer (line plot with error bars)
    plot_pc1_trajectory_layer_session(pca_results, data['session_labels'], 
                                      data['layer_labels'],
                                      save_path=os.path.join(figure_dir, 
                                                            'pca_pc1_trajectory_layer_session.png'))
    
    # 2. Grid of PC1 histograms (Layer × Session)
    plot_layer_session_detailed_grid(pca_results, data['session_labels'],
                                     data['layer_labels'], data['landmark_positions'],
                                     save_path=os.path.join(figure_dir,
                                                           'pca_layer_session_grid.png'))
    
    # 3. Landmark preference evolution (Superficial vs. Deep)
    plot_landmark_preference_evolution(pca_results, data['preferred_landmark'],
                                       data['session_labels'], data['layer_labels'],
                                       data['landmark_positions'],
                                       save_path=os.path.join(figure_dir,
                                                             'pca_landmark_evolution.png'))
    
    # 4. Landmark proportions by individual layers (stacked bars)
    plot_landmark_proportions_layer_session(pca_results, data['preferred_landmark'],
                                           data['session_labels'], data['layer_labels'],
                                           data['landmark_positions'],
                                           save_path=os.path.join(figure_dir,
                                                                 'pca_landmark_proportions_layer_session.png'))
    
    # 5. Statistical test for Layer × Session interaction
    test_layer_session_interaction(pca_results, data['session_labels'], 
                                   data['layer_labels'])
    
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
        n_components=N_COMPONENTS,
        use_aligned=USE_ALIGNED_PROFILES
    )