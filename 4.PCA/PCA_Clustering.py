"""
PCA_Clustering.py
Formal clustering of L1-preferring cells into adaptation-like vs spatial-like groups.

Uses multiple clustering approaches and validates against the early-slope based classification.

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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\JSY052_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\PCA\figures\clustering"

# Clustering parameters
N_CLUSTERS = 2  # Adaptation-like vs Spatial-like
PCS_FOR_CLUSTERING = [0, 1]  # PC1 and PC2 (0-indexed)


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
        data['spatial_profiles'] = f['features/spatial_profiles'][:]
        data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
        data['pc_scores'] = f['pca_results/pc_scores'][:]
        data['components'] = f['pca_results/components'][:]
        data['explained_variance_ratio'] = f['pca_results/explained_variance_ratio'][:]
    
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
# CLUSTERING METHODS
# ============================================================================

def run_kmeans(features, n_clusters=2, random_state=42):
    """Run K-means clustering."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features)
    
    return {
        'labels': labels,
        'centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'model': kmeans
    }


def run_gmm(features, n_clusters=2, random_state=42):
    """Run Gaussian Mixture Model clustering."""
    gmm = GaussianMixture(n_components=n_clusters, random_state=random_state, n_init=5)
    labels = gmm.fit_predict(features)
    probs = gmm.predict_proba(features)
    
    return {
        'labels': labels,
        'probabilities': probs,
        'means': gmm.means_,
        'covariances': gmm.covariances_,
        'bic': gmm.bic(features),
        'aic': gmm.aic(features),
        'model': gmm
    }


def run_hierarchical(features, n_clusters=2):
    """Run Hierarchical/Agglomerative clustering."""
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = hc.fit_predict(features)
    
    return {
        'labels': labels,
        'model': hc
    }


def find_optimal_clusters(features, max_k=6):
    """
    Find optimal number of clusters using multiple metrics.
    """
    print("\n--- Finding optimal number of clusters ---")
    
    k_range = range(2, max_k + 1)
    
    silhouette_scores = []
    inertias = []
    bics = []
    
    for k in k_range:
        # K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        km_labels = kmeans.fit_predict(features)
        inertias.append(kmeans.inertia_)
        
        # Silhouette
        if k > 1:
            sil = silhouette_score(features, km_labels)
            silhouette_scores.append(sil)
        else:
            silhouette_scores.append(0)
        
        # GMM BIC
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(features)
        bics.append(gmm.bic(features))
        
        print(f"  k={k}: Silhouette={silhouette_scores[-1]:.3f}, BIC={bics[-1]:.1f}")
    
    # Best k by silhouette
    best_k_sil = list(k_range)[np.argmax(silhouette_scores)]
    
    # Best k by BIC (lower is better)
    best_k_bic = list(k_range)[np.argmin(bics)]
    
    print(f"\n  Best k by Silhouette: {best_k_sil}")
    print(f"  Best k by BIC: {best_k_bic}")
    
    return {
        'k_range': list(k_range),
        'silhouette_scores': silhouette_scores,
        'inertias': inertias,
        'bics': bics,
        'best_k_silhouette': best_k_sil,
        'best_k_bic': best_k_bic
    }


# ============================================================================
# VALIDATION
# ============================================================================

def validate_clustering(cluster_labels, early_slopes, layer_labels, session_labels):
    """
    Validate clustering against early slope classification and analyze distribution.
    """
    print("\n" + "=" * 60)
    print("CLUSTERING VALIDATION")
    print("=" * 60)
    
    # Compare to early slope median split
    slope_median = np.median(early_slopes)
    slope_labels = (early_slopes >= slope_median).astype(int)
    
    # Ensure cluster labels align with slope labels
    # (Cluster 0 should be adaptation-like = low slope)
    cluster_0_mean_slope = np.mean(early_slopes[cluster_labels == 0])
    cluster_1_mean_slope = np.mean(early_slopes[cluster_labels == 1])
    
    if cluster_0_mean_slope > cluster_1_mean_slope:
        # Flip labels
        cluster_labels_aligned = 1 - cluster_labels
        print("  Note: Flipped cluster labels to align with slope interpretation")
    else:
        cluster_labels_aligned = cluster_labels.copy()
    
    # Agreement with slope-based classification
    agreement = np.mean(cluster_labels_aligned == (1 - slope_labels))  # 0=adapt, 1=spatial
    ari = adjusted_rand_score(slope_labels, cluster_labels_aligned)
    
    print(f"\n--- Agreement with Early Slope Classification ---")
    print(f"  Raw agreement: {agreement*100:.1f}%")
    print(f"  Adjusted Rand Index: {ari:.3f}")
    
    # Cluster characteristics
    print(f"\n--- Cluster Characteristics ---")
    
    for c in [0, 1]:
        mask = cluster_labels_aligned == c
        n_cells = np.sum(mask)
        
        cluster_type = "Adaptation-like" if c == 0 else "Spatial-like"
        
        mean_slope = np.mean(early_slopes[mask])
        std_slope = np.std(early_slopes[mask])
        
        print(f"\n  Cluster {c} ({cluster_type}): n = {n_cells}")
        print(f"    Early slope: {mean_slope:.4f} ± {std_slope:.4f}")
        
        # Layer distribution
        print(f"    Layer distribution:")
        for layer in ['L2/3', 'L4', 'L5', 'L6']:
            n_layer = np.sum((layer_labels == layer) & mask)
            pct = n_layer / n_cells * 100 if n_cells > 0 else 0
            print(f"      {layer}: {n_layer} ({pct:.1f}%)")
    
    return {
        'cluster_labels_aligned': cluster_labels_aligned,
        'slope_labels': slope_labels,
        'agreement': agreement,
        'ari': ari
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_clustering_results(data, l1_mask, cluster_labels, early_slopes,
                           kmeans_result, gmm_result, optimal_k_results,
                           save_dir=None):
    """
    Comprehensive visualization of clustering results.
    """
    
    pc_scores = data['pc_scores'][l1_mask]
    profiles = data['spatial_profiles_zscore'][l1_mask]
    layer_labels = data['layer_labels'][l1_mask]
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    
    var1 = data['explained_variance_ratio'][0] * 100
    var2 = data['explained_variance_ratio'][1] * 100
    
    fig = plt.figure(figsize=(18, 14))
    
    # Colors
    cluster_colors = {0: '#E53935', 1: '#1E88E5'}  # Red = adapt, Blue = spatial
    
    # =========================================================================
    # Panel 1: K-means clustering in PC space
    # =========================================================================
    ax1 = fig.add_subplot(2, 3, 1)
    
    for c in [0, 1]:
        mask = cluster_labels == c
        label = 'Adaptation-like' if c == 0 else 'Spatial-like'
        ax1.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                   c=cluster_colors[c], alpha=0.6, s=30, label=f'{label} (n={np.sum(mask)})')
    
    # Plot cluster centers
    centers = kmeans_result['centers']
    ax1.scatter(centers[:, 0], centers[:, 1], c='black', marker='X', s=200, 
               edgecolors='white', linewidths=2, label='Centroids')
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax1.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
    ax1.set_title('K-means Clustering (k=2)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 2: GMM clustering with probability contours
    # =========================================================================
    ax2 = fig.add_subplot(2, 3, 2)
    
    gmm_labels = gmm_result['labels']
    gmm_probs = gmm_result['probabilities']
    
    # Color by max probability (confidence)
    max_probs = np.max(gmm_probs, axis=1)
    
    for c in [0, 1]:
        mask = gmm_labels == c
        label = 'Cluster 0' if c == 0 else 'Cluster 1'
        scatter = ax2.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                             c=max_probs[mask], cmap='viridis', alpha=0.6, s=30,
                             vmin=0.5, vmax=1.0)
    
    plt.colorbar(scatter, ax=ax2, label='Classification Confidence')
    
    ax2.set_xlabel(f'PC1 ({var1:.1f}%)', fontsize=11)
    ax2.set_ylabel(f'PC2 ({var2:.1f}%)', fontsize=11)
    ax2.set_title('GMM Clustering (colored by confidence)', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 3: Silhouette and BIC scores
    # =========================================================================
    ax3 = fig.add_subplot(2, 3, 3)
    
    k_range = optimal_k_results['k_range']
    
    ax3_twin = ax3.twinx()
    
    line1 = ax3.plot(k_range, optimal_k_results['silhouette_scores'], 
                     'b-o', linewidth=2, markersize=8, label='Silhouette')
    ax3.set_ylabel('Silhouette Score', color='blue', fontsize=11)
    ax3.tick_params(axis='y', labelcolor='blue')
    
    line2 = ax3_twin.plot(k_range, optimal_k_results['bics'], 
                          'r-s', linewidth=2, markersize=8, label='BIC')
    ax3_twin.set_ylabel('BIC (lower is better)', color='red', fontsize=11)
    ax3_twin.tick_params(axis='y', labelcolor='red')
    
    ax3.axvline(2, color='green', linestyle='--', alpha=0.7, label='k=2')
    
    ax3.set_xlabel('Number of Clusters (k)', fontsize=11)
    ax3.set_title('Cluster Validation Metrics', fontsize=12, fontweight='bold')
    ax3.set_xticks(k_range)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='center right')
    
    # =========================================================================
    # Panel 4: Early slope distribution by cluster
    # =========================================================================
    ax4 = fig.add_subplot(2, 3, 4)
    
    for c in [0, 1]:
        mask = cluster_labels == c
        label = 'Adaptation-like' if c == 0 else 'Spatial-like'
        ax4.hist(early_slopes[mask], bins=30, alpha=0.6, color=cluster_colors[c],
                label=f'{label} (n={np.sum(mask)})', edgecolor='black', linewidth=0.5)
    
    ax4.axvline(np.median(early_slopes), color='black', linestyle='--', 
               linewidth=2, label='Median')
    
    ax4.set_xlabel('Early Slope', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Early Slope by Cluster', fontsize=12, fontweight='bold')
    ax4.legend()
    
    # =========================================================================
    # Panel 5: Mean profiles by cluster
    # =========================================================================
    ax5 = fig.add_subplot(2, 3, 5)
    
    x_vals = np.linspace(bin_centers[0], bin_centers[-1], profiles.shape[1])
    
    for c in [0, 1]:
        mask = cluster_labels == c
        label = 'Adaptation-like' if c == 0 else 'Spatial-like'
        
        mean_profile = np.mean(profiles[mask], axis=0)
        sem = np.std(profiles[mask], axis=0) / np.sqrt(np.sum(mask))
        
        ax5.plot(x_vals, mean_profile, color=cluster_colors[c], linewidth=2.5, 
                label=f'{label} (n={np.sum(mask)})')
        ax5.fill_between(x_vals, mean_profile - sem, mean_profile + sem,
                        color=cluster_colors[c], alpha=0.2)
    
    for lm_pos in landmark_positions:
        ax5.axvline(lm_pos, color='gray', linestyle='--', alpha=0.5)
    
    ax5.set_xlabel('Position (cm)', fontsize=11)
    ax5.set_ylabel('Z-scored Activity', fontsize=11)
    ax5.set_title('Mean Profiles by Cluster', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 6: Layer distribution by cluster
    # =========================================================================
    ax6 = fig.add_subplot(2, 3, 6)
    
    layers = ['L2/3', 'L4', 'L5', 'L6']
    x = np.arange(len(layers))
    width = 0.35
    
    adapt_counts = [np.sum((layer_labels == layer) & (cluster_labels == 0)) for layer in layers]
    spatial_counts = [np.sum((layer_labels == layer) & (cluster_labels == 1)) for layer in layers]
    
    # Calculate percentages
    adapt_pcts = []
    spatial_pcts = []
    for i, layer in enumerate(layers):
        total = adapt_counts[i] + spatial_counts[i]
        if total > 0:
            adapt_pcts.append(adapt_counts[i] / total * 100)
            spatial_pcts.append(spatial_counts[i] / total * 100)
        else:
            adapt_pcts.append(0)
            spatial_pcts.append(0)
    
    ax6.bar(x - width/2, adapt_pcts, width, label='Adaptation-like', 
           color=cluster_colors[0], alpha=0.8)
    ax6.bar(x + width/2, spatial_pcts, width, label='Spatial-like', 
           color=cluster_colors[1], alpha=0.8)
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(layers)
    ax6.set_xlabel('Cortical Layer', fontsize=11)
    ax6.set_ylabel('Percentage of L1 Cells', fontsize=11)
    ax6.set_title('Cluster Distribution by Layer', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.axhline(50, color='black', linestyle='--', alpha=0.5)
    
    # =========================================================================
    # Overall title
    # =========================================================================
    plt.suptitle('Clustering Analysis of L1-Preferring Cells', 
                fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'clustering_results.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"\n✓ Saved clustering figure: {save_path}")
    
    return fig


def plot_cluster_heatmaps(data, l1_mask, cluster_labels, save_dir=None):
    """
    Create heatmap visualizations for each cluster.
    """
    profiles = data['spatial_profiles_zscore'][l1_mask]
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    cluster_names = ['Adaptation-like', 'Spatial-like']
    cluster_colors_map = ['Reds', 'Blues']
    
    for c in [0, 1]:
        ax = axes[c]
        mask = cluster_labels == c
        cluster_profiles = profiles[mask]
        
        # Sort by peak position
        peak_positions = np.argmax(cluster_profiles, axis=1)
        sorted_idx = np.argsort(peak_positions)
        sorted_profiles = cluster_profiles[sorted_idx]
        
        im = ax.imshow(sorted_profiles, aspect='auto', cmap='viridis',
                      extent=[bin_centers[0], bin_centers[-1], 0, np.sum(mask)],
                      vmin=-2, vmax=3)
        
        for lm_pos in landmark_positions:
            ax.axvline(lm_pos, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Position (cm)', fontsize=11)
        ax.set_ylabel('Cell # (sorted by peak)', fontsize=11)
        ax.set_title(f'{cluster_names[c]} Cluster\n(n={np.sum(mask)})', 
                    fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Z-scored Activity')
    
    plt.suptitle('L1 Cell Response Profiles by Cluster', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'cluster_heatmaps.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved cluster heatmaps: {save_path}")
    
    return fig


# ============================================================================
# SAVE CLUSTERING RESULTS
# ============================================================================

def save_clustering_results(pca_data_path, l1_indices, cluster_labels, 
                           validation_results, kmeans_result, gmm_result):
    """
    Save clustering results back to the HDF5 file.
    """
    print(f"\nSaving clustering results to: {pca_data_path}")
    
    with h5py.File(pca_data_path, 'a') as f:
        # Create or replace clustering group
        if 'clustering' in f:
            del f['clustering']
        
        clust_grp = f.create_group('clustering')
        
        # L1 cell indices and labels
        clust_grp.create_dataset('l1_cell_indices', data=l1_indices)
        clust_grp.create_dataset('cluster_labels', data=cluster_labels)
        clust_grp.create_dataset('cluster_labels_aligned', 
                                data=validation_results['cluster_labels_aligned'])
        
        # Metadata
        clust_grp.attrs['n_clusters'] = 2
        clust_grp.attrs['method'] = 'kmeans'
        clust_grp.attrs['agreement_with_slope'] = validation_results['agreement']
        clust_grp.attrs['adjusted_rand_index'] = validation_results['ari']
        
        # K-means centers
        clust_grp.create_dataset('kmeans_centers', data=kmeans_result['centers'])
        
        # GMM parameters
        clust_grp.create_dataset('gmm_means', data=gmm_result['means'])
        clust_grp.create_dataset('gmm_probabilities', data=gmm_result['probabilities'])
        clust_grp.attrs['gmm_bic'] = gmm_result['bic']
        clust_grp.attrs['gmm_aic'] = gmm_result['aic']
    
    print("  ✓ Clustering results saved!")


# ============================================================================
# MAIN
# ============================================================================

def run_clustering_analysis(pca_data_path, figure_dir, n_clusters=2, 
                           pcs_for_clustering=[0, 1]):
    """
    Main function to run clustering analysis on L1 cells.
    """
    print("=" * 70)
    print("CLUSTERING ANALYSIS FOR L1 CELLS")
    print("=" * 70)
    
    os.makedirs(figure_dir, exist_ok=True)
    
    # Load data
    data = load_data(pca_data_path)
    
    # Get L1 cells
    l1_mask = data['preferred_landmark'] == 0
    l1_indices = np.where(l1_mask)[0]
    n_l1 = np.sum(l1_mask)
    
    print(f"\nAnalyzing {n_l1} L1-preferring cells")
    
    # Extract features for clustering (PC scores)
    pc_scores_l1 = data['pc_scores'][l1_mask][:, pcs_for_clustering]
    
    print(f"Using PCs: {[f'PC{i+1}' for i in pcs_for_clustering]}")
    print(f"Feature matrix shape: {pc_scores_l1.shape}")
    
    # Compute early slopes for validation
    early_slopes = compute_early_slope(
        data['spatial_profiles_zscore'][l1_mask],
        data['bin_centers']
    )
    
    # Find optimal number of clusters
    optimal_k_results = find_optimal_clusters(pc_scores_l1, max_k=6)
    
    # Run clustering methods
    print(f"\n--- Running clustering with k={n_clusters} ---")
    
    kmeans_result = run_kmeans(pc_scores_l1, n_clusters=n_clusters)
    gmm_result = run_gmm(pc_scores_l1, n_clusters=n_clusters)
    hc_result = run_hierarchical(pc_scores_l1, n_clusters=n_clusters)
    
    # Use K-means as primary
    cluster_labels = kmeans_result['labels']
    
    # Align labels so cluster 0 = adaptation-like (lower slope)
    cluster_0_mean_slope = np.mean(early_slopes[cluster_labels == 0])
    cluster_1_mean_slope = np.mean(early_slopes[cluster_labels == 1])
    
    if cluster_0_mean_slope > cluster_1_mean_slope:
        cluster_labels = 1 - cluster_labels
        print("  Flipped cluster labels to align with slope interpretation")
    
    # Compute silhouette score
    sil_score = silhouette_score(pc_scores_l1, cluster_labels)
    print(f"\n  Silhouette score: {sil_score:.3f}")
    
    # Validate clustering
    validation_results = validate_clustering(
        cluster_labels, early_slopes,
        data['layer_labels'][l1_mask],
        data['session_labels'][l1_mask]
    )
    
    # Compare clustering methods
    print("\n--- Clustering Method Agreement ---")
    print(f"  K-means vs GMM: {np.mean(cluster_labels == gmm_result['labels'])*100:.1f}%")
    
    hc_labels = hc_result['labels']
    hc_0_mean_slope = np.mean(early_slopes[hc_labels == 0])
    hc_1_mean_slope = np.mean(early_slopes[hc_labels == 1])
    if hc_0_mean_slope > hc_1_mean_slope:
        hc_labels = 1 - hc_labels
    print(f"  K-means vs Hierarchical: {np.mean(cluster_labels == hc_labels)*100:.1f}%")
    
    # Generate visualizations
    print("\nGenerating figures...")
    
    plot_clustering_results(
        data, l1_mask, cluster_labels, early_slopes,
        kmeans_result, gmm_result, optimal_k_results,
        save_dir=figure_dir
    )
    
    plot_cluster_heatmaps(data, l1_mask, cluster_labels, save_dir=figure_dir)
    
    # Save results
    save_clustering_results(
        pca_data_path, l1_indices, cluster_labels,
        validation_results, kmeans_result, gmm_result
    )
    
    print("\n" + "=" * 70)
    print("CLUSTERING ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"\nFigures saved to: {figure_dir}")
    
    plt.show()
    
    return {
        'cluster_labels': cluster_labels,
        'validation': validation_results,
        'kmeans': kmeans_result,
        'gmm': gmm_result,
        'optimal_k': optimal_k_results,
        'silhouette_score': sil_score
    }


if __name__ == "__main__":
    results = run_clustering_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR,
        n_clusters=N_CLUSTERS,
        pcs_for_clustering=PCS_FOR_CLUSTERING
    )