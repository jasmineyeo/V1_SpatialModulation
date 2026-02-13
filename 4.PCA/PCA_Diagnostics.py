"""
PCA_Diagnostics.py
Sanity checks and diagnostic plots for PCA analysis.

Run after PCA_DataAggregation.py to verify data quality.

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
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

ANIMAL_ID = "JSY054"
BASE_DIR = rf"D:\V1_SpatialModulation\2p\V1_prism\{ANIMAL_ID}_ChronicImaging"
PCA_DATA_PATH = rf"D:\V1_SpatialModulation\2p\V1_prism\{ANIMAL_ID}_ChronicImaging\PCA\{ANIMAL_ID}_pca_data.h5"
FIGURE_DIR = rf"D:\V1_SpatialModulation\2p\V1_prism\{ANIMAL_ID}_ChronicImaging\PCA\figures\diagnostics"

# Landmark positions for reference
LANDMARK_POSITIONS = [25, 55, 85, 115]


# ============================================================================
# DIAGNOSTIC 1: PEAK POSITION DISTRIBUTIONS
# ============================================================================

def plot_peak_position_sanity_check(pca_data_path, save_dir):
    """
    Sanity check: Where do included vs excluded cells peak?
    
    This helps verify that onset/reward exclusion is working as intended
    and that L1-preferring cells are truly landmark-responsive.
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC 1: Peak Position Distribution")
    print("="*60)
    
    with h5py.File(pca_data_path, 'r') as f:
# Load included cells data
        peak_positions = f['cells/peak_positions'][:]
        preferred_landmark = f['cells/preferred_landmark'][:]
        layer_labels = f['cells/layer_labels'][:].astype(str)
        session_labels = f['cells/session_labels'][:].astype(str)
        bin_centers = f['metadata/bin_centers_trimmed'][:]
        
        # Get exclusion thresholds from metadata
        exclude_first = f['metadata'].attrs.get('exclude_first_bins', 5)
        exclude_last = f['metadata'].attrs.get('exclude_last_bins', 5)
    
    n_cells = len(peak_positions)
    print(f"Total cells in PCA dataset: {n_cells}")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel 1: Overall peak position histogram
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    ax1.hist(peak_positions, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    
    # Mark landmarks
    for i, lm in enumerate(LANDMARK_POSITIONS):
        ax1.axvline(lm, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(lm, ax1.get_ylim()[1]*0.95, f'L{i+1}', ha='center', fontsize=10, color='red')
    
    # Mark exclusion zones
    min_pos = bin_centers[0]
    max_pos = bin_centers[-1]
    bin_spacing = np.mean(np.diff(bin_centers))
    onset_thresh = min_pos + exclude_first * bin_spacing
    reward_thresh = max_pos - exclude_last * bin_spacing
    
    ax1.axvspan(min_pos, onset_thresh, alpha=0.2, color='gray', label='Onset exclusion')
    ax1.axvspan(reward_thresh, max_pos, alpha=0.2, color='orange', label='Reward exclusion')
    
    ax1.set_xlabel('Peak Position (cm)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Peak Positions of Included Cells', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    
    # =========================================================================
    # Panel 2: Peak positions by landmark preference
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    lm_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 'gray']
    lm_labels = ['L1 (25cm)', 'L2 (55cm)', 'L3 (85cm)', 'L4 (115cm)', 'Between']
    
    for lm_idx in range(-1, 4):
        mask = preferred_landmark == lm_idx
        if np.sum(mask) > 0:
            color_idx = lm_idx if lm_idx >= 0 else 4
            ax2.hist(peak_positions[mask], bins=30, alpha=0.5, 
                    color=lm_colors[color_idx], label=f'{lm_labels[color_idx]} (n={np.sum(mask)})',
                    edgecolor='black', linewidth=0.5)
    
    for lm in LANDMARK_POSITIONS:
        ax2.axvline(lm, color='black', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Peak Position (cm)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Peak Positions by Landmark Preference', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    
    # =========================================================================
    # Panel 3: Peak positions by layer
    # =========================================================================
    ax3 = fig.add_subplot(gs[0, 2])
    
    layer_colors = {'L2/3': '#1E88E5', 'L4': '#FF9800', 'L5': '#4CAF50', 'L6': '#E53935'}
    layers = ['L2/3', 'L4', 'L5', 'L6']
    
    for layer in layers:
        mask = layer_labels == layer
        if np.sum(mask) > 0:
            ax3.hist(peak_positions[mask], bins=30, alpha=0.5,
                    color=layer_colors[layer], label=f'{layer} (n={np.sum(mask)})',
                    edgecolor='black', linewidth=0.5)
    
    for lm in LANDMARK_POSITIONS:
        ax3.axvline(lm, color='black', linestyle='--', alpha=0.5)
    
    ax3.set_xlabel('Peak Position (cm)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('Peak Positions by Layer', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    
    # =========================================================================
    # Panel 4: Landmark preference proportions by layer
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Calculate proportions
    layer_lm_counts = np.zeros((4, 5))  # 4 layers x 5 preference types (L1-L4 + between)
    
    for li, layer in enumerate(layers):
        layer_mask = layer_labels == layer
        n_layer = np.sum(layer_mask)
        
        if n_layer > 0:
            for lm_idx in range(-1, 4):
                count = np.sum((preferred_landmark == lm_idx) & layer_mask)
                col_idx = lm_idx + 1 if lm_idx >= 0 else 4  # Remap: L1=0, L2=1, L3=2, L4=3, Between=4
                if lm_idx == -1:
                    col_idx = 4
                else:
                    col_idx = lm_idx
                layer_lm_counts[li, col_idx] = count / n_layer * 100
    
    # Stacked bar
    x = np.arange(len(layers))
    bottom = np.zeros(4)
    
    for lm_idx in range(4):
        ax4.bar(x, layer_lm_counts[:, lm_idx], bottom=bottom, 
               color=lm_colors[lm_idx], label=f'L{lm_idx+1}', alpha=0.8)
        bottom += layer_lm_counts[:, lm_idx]
    
    ax4.bar(x, layer_lm_counts[:, 4], bottom=bottom, 
           color='gray', label='Between', alpha=0.8)
    
    ax4.set_xticks(x)
    ax4.set_xticklabels(layers)
    ax4.set_xlabel('Cortical Layer', fontsize=11)
    ax4.set_ylabel('Percentage', fontsize=11)
    ax4.set_title('Landmark Preference by Layer', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8, ncol=2)
    ax4.set_ylim(0, 100)
    
    # =========================================================================
    # Panel 5: Cells per session
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 1])
    
    unique_sessions = sorted(np.unique(session_labels), key=lambda x: int(x.replace('Day', '')))
    session_counts = [np.sum(session_labels == s) for s in unique_sessions]
    
    ax5.bar(unique_sessions, session_counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax5.set_xlabel('Session', fontsize=11)
    ax5.set_ylabel('Number of Cells', fontsize=11)
    ax5.set_title('Cells per Session', fontsize=12, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add mean line
    mean_count = np.mean(session_counts)
    ax5.axhline(mean_count, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_count:.0f}')
    ax5.legend()
    
    # =========================================================================
    # Panel 6: Cells per layer per session (heatmap)
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 2])
    
    layer_session_counts = np.zeros((len(layers), len(unique_sessions)))
    
    for li, layer in enumerate(layers):
        for si, session in enumerate(unique_sessions):
            layer_session_counts[li, si] = np.sum((layer_labels == layer) & 
                                                   (session_labels == session))
    
    im = ax6.imshow(layer_session_counts, aspect='auto', cmap='Blues')
    ax6.set_xticks(np.arange(len(unique_sessions)))
    ax6.set_xticklabels(unique_sessions, rotation=45, ha='right')
    ax6.set_yticks(np.arange(len(layers)))
    ax6.set_yticklabels(layers)
    ax6.set_xlabel('Session', fontsize=11)
    ax6.set_ylabel('Layer', fontsize=11)
    ax6.set_title('Cells per Layer × Session', fontsize=12, fontweight='bold')
    
    # Add text annotations
    for i in range(len(layers)):
        for j in range(len(unique_sessions)):
            ax6.text(j, i, f'{int(layer_session_counts[i, j])}', 
                    ha='center', va='center', fontsize=8)
    
    plt.colorbar(im, ax=ax6, label='Count')
    
    # =========================================================================
    # Panel 7: L1 cells specifically - peak position distribution
    # =========================================================================
    ax7 = fig.add_subplot(gs[2, 0])
    
    l1_mask = preferred_landmark == 0
    l1_peaks = peak_positions[l1_mask]
    
    ax7.hist(l1_peaks, bins=20, color='#e41a1c', edgecolor='black', alpha=0.7)
    ax7.axvline(25, color='black', linestyle='--', linewidth=2, label='L1 position (25cm)')
    ax7.axvline(np.mean(l1_peaks), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean: {np.mean(l1_peaks):.1f}cm')
    ax7.axvline(np.median(l1_peaks), color='green', linestyle='-', linewidth=2,
               label=f'Median: {np.median(l1_peaks):.1f}cm')
    
    ax7.set_xlabel('Peak Position (cm)', fontsize=11)
    ax7.set_ylabel('Count', fontsize=11)
    ax7.set_title(f'L1-Preferring Cells Peak Positions (n={np.sum(l1_mask)})', 
                 fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    
    # =========================================================================
    # Panel 8: Peak position vs landmark for L1 cells by layer
    # =========================================================================
    ax8 = fig.add_subplot(gs[2, 1])
    
    for layer in layers:
        mask = l1_mask & (layer_labels == layer)
        if np.sum(mask) > 0:
            peaks = peak_positions[mask]
            # Jitter for visualization
            jitter = np.random.normal(0, 0.1, len(peaks))
            layer_idx = layers.index(layer)
            ax8.scatter(peaks, layer_idx + jitter, c=layer_colors[layer], 
                       alpha=0.6, s=30, label=f'{layer} (n={np.sum(mask)})')
    
    ax8.axvline(25, color='black', linestyle='--', linewidth=2)
    ax8.set_xlabel('Peak Position (cm)', fontsize=11)
    ax8.set_yticks(range(len(layers)))
    ax8.set_yticklabels(layers)
    ax8.set_title('L1 Cells: Peak Position by Layer', fontsize=12, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=8)
    ax8.set_xlim(10, 40)
    
    # =========================================================================
    # Panel 9: Summary statistics
    # =========================================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Calculate stats
    n_l1 = np.sum(preferred_landmark == 0)
    n_l2 = np.sum(preferred_landmark == 1)
    n_l3 = np.sum(preferred_landmark == 2)
    n_l4 = np.sum(preferred_landmark == 3)
    n_between = np.sum(preferred_landmark == -1)
    
    l1_before = np.sum((preferred_landmark == 0) & (peak_positions < 25))
    l1_at = np.sum((preferred_landmark == 0) & (peak_positions >= 23) & (peak_positions <= 27))
    l1_after = np.sum((preferred_landmark == 0) & (peak_positions > 25))
    
    summary_text = f"""
    SANITY CHECK SUMMARY
    {'='*40}
    
    Total Cells in PCA: {n_cells}
    
    Landmark Preference:
      L1 (25cm):  {n_l1} ({n_l1/n_cells*100:.1f}%)
      L2 (55cm):  {n_l2} ({n_l2/n_cells*100:.1f}%)
      L3 (85cm):  {n_l3} ({n_l3/n_cells*100:.1f}%)
      L4 (115cm): {n_l4} ({n_l4/n_cells*100:.1f}%)
      Between:    {n_between} ({n_between/n_cells*100:.1f}%)
    
    L1 Cell Peak Distribution:
      Before L1 (<25cm): {l1_before} ({l1_before/n_l1*100:.1f}%)
      At L1 (23-27cm):   {l1_at} ({l1_at/n_l1*100:.1f}%)
      After L1 (>25cm):  {l1_after} ({l1_after/n_l1*100:.1f}%)
    
    Layer Distribution:
      L2/3: {np.sum(layer_labels=='L2/3')} ({np.sum(layer_labels=='L2/3')/n_cells*100:.1f}%)
      L4:   {np.sum(layer_labels=='L4')} ({np.sum(layer_labels=='L4')/n_cells*100:.1f}%)
      L5:   {np.sum(layer_labels=='L5')} ({np.sum(layer_labels=='L5')/n_cells*100:.1f}%)
      L6:   {np.sum(layer_labels=='L6')} ({np.sum(layer_labels=='L6')/n_cells*100:.1f}%)
    
    Sessions: {len(unique_sessions)}
    Cells/Session: {np.mean(session_counts):.0f} ± {np.std(session_counts):.0f}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes,
            fontsize=10, fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'PCA Data Sanity Check: {ANIMAL_ID}', fontsize=14, fontweight='bold', y=1.02)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'sanity_check_peak_positions.png'),
                   dpi=150, bbox_inches='tight')
        print(f"✓ Saved sanity check figure")
    
    return fig


# ============================================================================
# DIAGNOSTIC 2: SESSION EFFECTS
# ============================================================================

def plot_session_effects(pca_data_path, save_dir):
    """
    Check for session-specific effects that might confound PCA.
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: Session Effects")
    print("="*60)
    
    with h5py.File(pca_data_path, 'r') as f:
        session_labels = f['cells/session_labels'][:].astype(str)
        layer_labels = f['cells/layer_labels'][:].astype(str)
        spatial_profiles = f['features/spatial_profiles'][:]
        spatial_profiles_zscore = f['features/spatial_profiles_zscore'][:]
        pc_scores = f['pca_results/pc_scores'][:]
        bin_centers = f['metadata/bin_centers_trimmed'][:]
    
    unique_sessions = sorted(np.unique(session_labels), key=lambda x: int(x.replace('Day', '')))
    n_sessions = len(unique_sessions)
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # =========================================================================
    # Panel 1-4: Mean spatial profile per session
    # =========================================================================
    for si, session in enumerate(unique_sessions[:4]):  # First 4 sessions
        ax = fig.add_subplot(gs[0, si])
        
        mask = session_labels == session
        mean_profile = np.mean(spatial_profiles_zscore[mask], axis=0)
        sem = np.std(spatial_profiles_zscore[mask], axis=0) / np.sqrt(np.sum(mask))
        
        ax.plot(bin_centers, mean_profile, 'k-', linewidth=2)
        ax.fill_between(bin_centers, mean_profile - sem, mean_profile + sem, alpha=0.3)
        
        for lm in LANDMARK_POSITIONS:
            ax.axvline(lm, color='red', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Position (cm)', fontsize=9)
        ax.set_ylabel('Z-scored Activity', fontsize=9)
        ax.set_title(f'{session} (n={np.sum(mask)})', fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 5: PC1 mean by session
    # =========================================================================
    ax5 = fig.add_subplot(gs[1, 0])
    
    pc1_means = []
    pc1_sems = []
    for session in unique_sessions:
        mask = session_labels == session
        pc1_means.append(np.mean(pc_scores[mask, 0]))
        pc1_sems.append(np.std(pc_scores[mask, 0]) / np.sqrt(np.sum(mask)))
    
    x = np.arange(len(unique_sessions))
    ax5.errorbar(x, pc1_means, yerr=pc1_sems, marker='o', capsize=3, 
                linewidth=2, markersize=8, color='steelblue')
    ax5.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax5.set_xticks(x)
    ax5.set_xticklabels(unique_sessions, rotation=45)
    ax5.set_xlabel('Session', fontsize=11)
    ax5.set_ylabel('Mean PC1', fontsize=11)
    ax5.set_title('PC1 by Session', fontsize=12, fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 6: PC2 mean by session
    # =========================================================================
    ax6 = fig.add_subplot(gs[1, 1])
    
    pc2_means = []
    pc2_sems = []
    for session in unique_sessions:
        mask = session_labels == session
        pc2_means.append(np.mean(pc_scores[mask, 1]))
        pc2_sems.append(np.std(pc_scores[mask, 1]) / np.sqrt(np.sum(mask)))
    
    ax6.errorbar(x, pc2_means, yerr=pc2_sems, marker='o', capsize=3,
                linewidth=2, markersize=8, color='darkorange')
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xticks(x)
    ax6.set_xticklabels(unique_sessions, rotation=45)
    ax6.set_xlabel('Session', fontsize=11)
    ax6.set_ylabel('Mean PC2', fontsize=11)
    ax6.set_title('PC2 by Session', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 7: PC variance by session (are some sessions noisier?)
    # =========================================================================
    ax7 = fig.add_subplot(gs[1, 2])
    
    pc1_vars = []
    pc2_vars = []
    for session in unique_sessions:
        mask = session_labels == session
        pc1_vars.append(np.var(pc_scores[mask, 0]))
        pc2_vars.append(np.var(pc_scores[mask, 1]))
    
    width = 0.35
    ax7.bar(x - width/2, pc1_vars, width, label='PC1 var', color='steelblue', alpha=0.7)
    ax7.bar(x + width/2, pc2_vars, width, label='PC2 var', color='darkorange', alpha=0.7)
    ax7.set_xticks(x)
    ax7.set_xticklabels(unique_sessions, rotation=45)
    ax7.set_xlabel('Session', fontsize=11)
    ax7.set_ylabel('Variance', fontsize=11)
    ax7.set_title('PC Score Variance by Session', fontsize=12, fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3, axis='y')
    
    # =========================================================================
    # Panel 8: Mean raw activity level by session (before z-scoring)
    # =========================================================================
    ax8 = fig.add_subplot(gs[1, 3])
    
    mean_activity = []
    std_activity = []
    for session in unique_sessions:
        mask = session_labels == session
        session_profiles = spatial_profiles[mask]
        mean_activity.append(np.mean(session_profiles))
        std_activity.append(np.std(np.mean(session_profiles, axis=1)))
    
    ax8.errorbar(x, mean_activity, yerr=std_activity, marker='s', capsize=3,
                linewidth=2, markersize=8, color='green')
    ax8.set_xticks(x)
    ax8.set_xticklabels(unique_sessions, rotation=45)
    ax8.set_xlabel('Session', fontsize=11)
    ax8.set_ylabel('Mean Activity (pre-zscore)', fontsize=11)
    ax8.set_title('Raw Activity Level by Session', fontsize=12, fontweight='bold')
    ax8.grid(alpha=0.3)
    
    # =========================================================================
    # Panel 9-12: PC distributions per session (first 4)
    # =========================================================================
    for si, session in enumerate(unique_sessions[:4]):
        ax = fig.add_subplot(gs[2, si])
        
        mask = session_labels == session
        ax.scatter(pc_scores[mask, 0], pc_scores[mask, 1], alpha=0.5, s=15)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('PC1', fontsize=9)
        ax.set_ylabel('PC2', fontsize=9)
        ax.set_title(f'{session}', fontsize=10, fontweight='bold')
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
    
    plt.suptitle(f'Session Effects Diagnostic: {ANIMAL_ID}', fontsize=14, fontweight='bold', y=1.02)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'diagnostic_session_effects.png'),
                   dpi=150, bbox_inches='tight')
        print(f"✓ Saved session effects figure")
    
    # Print session statistics
    print("\nSession Statistics:")
    print(f"{'Session':<10} {'N cells':<10} {'PC1 mean':<12} {'PC2 mean':<12} {'Raw activity':<15}")
    print("-" * 60)
    for si, session in enumerate(unique_sessions):
        mask = session_labels == session
        print(f"{session:<10} {np.sum(mask):<10} {pc1_means[si]:<12.2f} {pc2_means[si]:<12.2f} {mean_activity[si]:<15.4f}")
    
    return fig


# ============================================================================
# DIAGNOSTIC 3: WITHIN-SESSION NORMALIZATION TEST
# ============================================================================

def compare_normalization_approaches(pca_data_path, save_dir):
    """
    Compare global z-scoring vs within-session z-scoring.
    This helps identify if session effects are dominating your PCA.
    """
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: Normalization Comparison")
    print("="*60)
    
    from sklearn.decomposition import PCA
    
    with h5py.File(pca_data_path, 'r') as f:
        session_labels = f['cells/session_labels'][:].astype(str)
        spatial_profiles = f['features/spatial_profiles'][:]
        bin_centers = f['metadata/bin_centers_trimmed'][:]
    
    unique_sessions = sorted(np.unique(session_labels), key=lambda x: int(x.replace('Day', '')))
    
    # Approach 1: Global z-scoring (current approach)
    global_zscore = np.zeros_like(spatial_profiles)
    for i in range(spatial_profiles.shape[0]):
        profile = spatial_profiles[i]
        global_zscore[i] = (profile - np.mean(profile)) / (np.std(profile) + 1e-10)
    
    # Approach 2: Within-session z-scoring
    session_zscore = np.zeros_like(spatial_profiles)
    for session in unique_sessions:
        mask = session_labels == session
        session_profiles = spatial_profiles[mask]
        
        # Z-score within session
        session_mean = np.mean(session_profiles)
        session_std = np.std(session_profiles)
        
        for i, idx in enumerate(np.where(mask)[0]):
            profile = spatial_profiles[idx]
            # Z-score by cell's own mean/std, then adjust by session
            cell_zscore = (profile - np.mean(profile)) / (np.std(profile) + 1e-10)
            session_zscore[idx] = cell_zscore
    
    # Run PCA on both
    pca_global = PCA(n_components=5)
    scores_global = pca_global.fit_transform(global_zscore)
    
    pca_session = PCA(n_components=5)
    scores_session = pca_session.fit_transform(session_zscore)
    
    # Compare
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Global normalization
    ax1 = axes[0, 0]
    for si, session in enumerate(unique_sessions):
        mask = session_labels == session
        color = plt.cm.viridis(si / len(unique_sessions))
        ax1.scatter(scores_global[mask, 0], scores_global[mask, 1], 
                   c=[color], alpha=0.5, s=15, label=session)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title('Global Z-score: PC1 vs PC2', fontweight='bold')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(alpha=0.3)
    
    # Row 2: Session normalization
    ax4 = axes[1, 0]
    for si, session in enumerate(unique_sessions):
        mask = session_labels == session
        color = plt.cm.viridis(si / len(unique_sessions))
        ax4.scatter(scores_session[mask, 0], scores_session[mask, 1],
                   c=[color], alpha=0.5, s=15, label=session)
    ax4.set_xlabel('PC1')
    ax4.set_ylabel('PC2')
    ax4.set_title('Session Z-score: PC1 vs PC2', fontweight='bold')
    ax4.legend(fontsize=7, ncol=2)
    ax4.grid(alpha=0.3)
    
    # Session centroids comparison
    ax2 = axes[0, 1]
    centroids_global = []
    for session in unique_sessions:
        mask = session_labels == session
        centroids_global.append(np.mean(scores_global[mask, :2], axis=0))
    centroids_global = np.array(centroids_global)
    
    for si, session in enumerate(unique_sessions):
        ax2.scatter(centroids_global[si, 0], centroids_global[si, 1], s=100, 
                   label=session, marker='o')
        ax2.annotate(session, (centroids_global[si, 0], centroids_global[si, 1]),
                    fontsize=8)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title('Global: Session Centroids', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    ax5 = axes[1, 1]
    centroids_session = []
    for session in unique_sessions:
        mask = session_labels == session
        centroids_session.append(np.mean(scores_session[mask, :2], axis=0))
    centroids_session = np.array(centroids_session)
    
    for si, session in enumerate(unique_sessions):
        ax5.scatter(centroids_session[si, 0], centroids_session[si, 1], s=100,
                   label=session, marker='o')
        ax5.annotate(session, (centroids_session[si, 0], centroids_session[si, 1]),
                    fontsize=8)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_title('Session: Session Centroids', fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # Variance explained comparison
    ax3 = axes[0, 2]
    x = np.arange(1, 6)
    ax3.bar(x - 0.2, pca_global.explained_variance_ratio_ * 100, 0.4, 
           label='Global', color='steelblue', alpha=0.7)
    ax3.bar(x + 0.2, pca_session.explained_variance_ratio_ * 100, 0.4,
           label='Session', color='darkorange', alpha=0.7)
    ax3.set_xlabel('PC')
    ax3.set_ylabel('Variance Explained (%)')
    ax3.set_title('Variance Explained', fontweight='bold')
    ax3.legend()
    ax3.set_xticks(x)
    
    # PC loadings comparison
    ax6 = axes[1, 2]
    ax6.plot(bin_centers, pca_global.components_[0], 'b-', linewidth=2, label='Global PC1')
    ax6.plot(bin_centers, pca_session.components_[0], 'r--', linewidth=2, label='Session PC1')
    for lm in LANDMARK_POSITIONS:
        ax6.axvline(lm, color='gray', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Position (cm)')
    ax6.set_ylabel('Loading')
    ax6.set_title('PC1 Loadings Comparison', fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)
    
    plt.suptitle(f'Normalization Comparison: {ANIMAL_ID}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'diagnostic_normalization.png'),
                   dpi=150, bbox_inches='tight')
        print(f"✓ Saved normalization comparison figure")
    
    # Quantify session clustering
    from scipy.spatial.distance import pdist
    
    global_spread = np.mean(pdist(centroids_global))
    session_spread = np.mean(pdist(centroids_session))
    
    print(f"\nSession centroid spread:")
    print(f"  Global normalization: {global_spread:.3f}")
    print(f"  Session normalization: {session_spread:.3f}")
    print(f"  Ratio: {global_spread/session_spread:.2f}x")
    
    if global_spread > session_spread * 1.5:
        print("  ⚠️ WARNING: Sessions cluster differently in PC space!")
        print("     Consider session normalization or adding session as covariate.")
    else:
        print("  ✓ Session effects appear minimal.")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def run_all_diagnostics():
    """Run all diagnostic checks."""
    print("="*70)
    print(f"RUNNING PCA DIAGNOSTICS: {ANIMAL_ID}")
    print("="*70)
    
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # Diagnostic 1: Peak positions
    fig1 = plot_peak_position_sanity_check(PCA_DATA_PATH, FIGURE_DIR)
    
    # Diagnostic 2: Session effects  
    fig2 = plot_session_effects(PCA_DATA_PATH, FIGURE_DIR)
    
    # Diagnostic 3: Normalization comparison
    fig3 = compare_normalization_approaches(PCA_DATA_PATH, FIGURE_DIR)
    
    print("\n" + "="*70)
    print("DIAGNOSTICS COMPLETE!")
    print("="*70)
    print(f"Figures saved to: {FIGURE_DIR}")
    
    plt.show()


if __name__ == "__main__":
    run_all_diagnostics()