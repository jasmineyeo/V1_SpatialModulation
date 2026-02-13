"""
PCA_DataVerification.py
Verify raw traces and landmark preferences are correct.

JSY, 12/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

ANIMAL_ID = "JSY054"  # Change this
BASE_DIR = rf"D:\V1_SpatialModulation\2p\V1_prism\{ANIMAL_ID}_ChronicImaging"
PCA_DATA_PATH = rf"D:\V1_SpatialModulation\2p\V1_prism\{ANIMAL_ID}_ChronicImaging\PCA\{ANIMAL_ID}_pca_data.h5"
FIGURE_DIR = rf"D:\V1_SpatialModulation\2p\V1_prism\{ANIMAL_ID}_ChronicImaging\PCA\figures\verification"

LANDMARK_POSITIONS = [25, 55, 85, 115]


# ============================================================================
# 1. VERIFY INDIVIDUAL CELL EXAMPLES
# ============================================================================

def plot_example_cells_by_landmark(pca_data_path, save_dir, n_examples=5):
    """
    Plot example cells for each landmark preference category.
    This lets you visually verify that L1 cells really peak at L1, etc.
    """
    print("\n" + "="*60)
    print("VERIFICATION 1: Example Cells by Landmark Preference")
    print("="*60)
    
    with h5py.File(pca_data_path, 'r') as f:
        spatial_profiles = f['features/spatial_profiles'][:]
        spatial_profiles_zscore = f['features/spatial_profiles_zscore'][:]
        preferred_landmark = f['cells/preferred_landmark'][:]
        peak_positions = f['cells/peak_positions'][:]
        session_labels = f['cells/session_labels'][:].astype(str)
        layer_labels = f['cells/layer_labels'][:].astype(str)
        bin_centers = f['metadata/bin_centers_trimmed'][:]
    
    fig, axes = plt.subplots(5, 5, figsize=(20, 16))
    
    categories = [
        (0, 'L1 (25cm)', '#e41a1c'),
        (1, 'L2 (55cm)', '#377eb8'),
        (2, 'L3 (85cm)', '#4daf4a'),
        (3, 'L4 (115cm)', '#984ea3'),
        (-1, 'Between', 'gray')
    ]
    
    for col, (lm_idx, label, color) in enumerate(categories):
        mask = preferred_landmark == lm_idx
        indices = np.where(mask)[0]
        
        if len(indices) == 0:
            continue
        
        # Sample random cells
        np.random.seed(42)
        sample_indices = np.random.choice(indices, min(n_examples, len(indices)), replace=False)
        
        for row, cell_idx in enumerate(sample_indices):
            ax = axes[row, col]
            
            # Plot raw profile
            ax.plot(bin_centers, spatial_profiles[cell_idx], 'k-', linewidth=1.5, alpha=0.7)
            
            # Mark peak
            peak_pos = peak_positions[cell_idx]
            peak_idx = np.argmin(np.abs(bin_centers - peak_pos))
            ax.axvline(peak_pos, color=color, linestyle='-', linewidth=2, alpha=0.8)
            
            # Mark landmarks
            for lm in LANDMARK_POSITIONS:
                ax.axvline(lm, color='gray', linestyle='--', alpha=0.3)
            
            # Labels
            session = session_labels[cell_idx]
            layer = layer_labels[cell_idx]
            ax.set_title(f'{session}, {layer}\nPeak: {peak_pos:.1f}cm', fontsize=9)
            
            if row == 0:
                ax.set_xlabel('')
            if row == n_examples - 1:
                ax.set_xlabel('Position (cm)', fontsize=9)
            if col == 0:
                ax.set_ylabel('Activity', fontsize=9)
        
        # Column header
        axes[0, col].text(0.5, 1.15, f'{label}\n(n={np.sum(mask)})', 
                         transform=axes[0, col].transAxes, ha='center', 
                         fontsize=11, fontweight='bold', color=color)
    
    plt.suptitle(f'Example Cells by Landmark Preference: {ANIMAL_ID}\n(Vertical colored line = detected peak)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'verify_example_cells.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved example cells figure")
    
    return fig


# ============================================================================
# 2. VERIFY "BETWEEN" CELLS - WHAT DO THEY LOOK LIKE?
# ============================================================================

def analyze_between_cells(pca_data_path, save_dir):
    """
    Analyze cells that fall "between" landmarks.
    Are they ramping cells? Noise? Multi-peaked?
    """
    print("\n" + "="*60)
    print("VERIFICATION 2: 'Between Landmarks' Cells Analysis")
    print("="*60)
    
    with h5py.File(pca_data_path, 'r') as f:
        spatial_profiles = f['features/spatial_profiles'][:]
        spatial_profiles_zscore = f['features/spatial_profiles_zscore'][:]
        preferred_landmark = f['cells/preferred_landmark'][:]
        peak_positions = f['cells/peak_positions'][:]
        bin_centers = f['metadata/bin_centers_trimmed'][:]
    
    between_mask = preferred_landmark == -1
    n_between = np.sum(between_mask)
    
    print(f"Total 'Between' cells: {n_between} ({n_between/len(preferred_landmark)*100:.1f}%)")
    
    if n_between == 0:
        print("No 'Between' cells found!")
        return None
    
    between_profiles = spatial_profiles_zscore[between_mask]
    between_peaks = peak_positions[between_mask]
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Panel 1: Peak position distribution for "between" cells
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(between_peaks, bins=30, color='gray', edgecolor='black', alpha=0.7)
    for lm in LANDMARK_POSITIONS:
        ax1.axvline(lm, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Peak Position (cm)', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'"Between" Cells Peak Positions (n={n_between})', fontsize=12, fontweight='bold')
    
    # Panel 2: Mean profile of "between" cells
    ax2 = fig.add_subplot(gs[0, 1])
    mean_profile = np.mean(between_profiles, axis=0)
    sem = np.std(between_profiles, axis=0) / np.sqrt(n_between)
    ax2.plot(bin_centers, mean_profile, 'k-', linewidth=2)
    ax2.fill_between(bin_centers, mean_profile - sem, mean_profile + sem, alpha=0.3)
    for lm in LANDMARK_POSITIONS:
        ax2.axvline(lm, color='red', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Position (cm)', fontsize=11)
    ax2.set_ylabel('Z-scored Activity', fontsize=11)
    ax2.set_title('Mean Profile of "Between" Cells', fontsize=12, fontweight='bold')
    
    # Panel 3: Classify "between" cells by profile shape
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Compute profile characteristics
    n_peaks_list = []
    is_ramping = []
    is_flat = []
    
    for profile in between_profiles:
        # Count peaks (simple threshold crossing)
        threshold = np.mean(profile) + 0.5 * np.std(profile)
        above_thresh = profile > threshold
        # Count transitions from below to above
        transitions = np.diff(above_thresh.astype(int))
        n_peaks = np.sum(transitions == 1)
        n_peaks_list.append(n_peaks)
        
        # Check if ramping (monotonic increase or decrease)
        slope = np.polyfit(np.arange(len(profile)), profile, 1)[0]
        is_ramping.append(abs(slope) > 0.01)
        
        # Check if flat (low variance)
        is_flat.append(np.std(profile) < 0.5)
    
    categories = ['0 peaks', '1 peak', '2+ peaks']
    counts = [np.sum(np.array(n_peaks_list) == 0),
              np.sum(np.array(n_peaks_list) == 1),
              np.sum(np.array(n_peaks_list) >= 2)]
    
    ax3.bar(categories, counts, color=['lightgray', 'steelblue', 'darkblue'], edgecolor='black')
    ax3.set_xlabel('Number of Peaks', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('"Between" Cells by Peak Count', fontsize=12, fontweight='bold')
    
    # Panels 4-6: Example "between" cells by type
    # Single-peaked "between" cells
    single_peak_mask = np.array(n_peaks_list) == 1
    if np.sum(single_peak_mask) > 0:
        ax4 = fig.add_subplot(gs[1, 0])
        single_peak_profiles = between_profiles[single_peak_mask]
        for i in range(min(10, len(single_peak_profiles))):
            ax4.plot(bin_centers, single_peak_profiles[i], alpha=0.5)
        ax4.plot(bin_centers, np.mean(single_peak_profiles, axis=0), 'k-', linewidth=3, label='Mean')
        for lm in LANDMARK_POSITIONS:
            ax4.axvline(lm, color='red', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Position (cm)')
        ax4.set_ylabel('Activity')
        ax4.set_title(f'Single-Peak "Between" (n={np.sum(single_peak_mask)})', fontweight='bold')
    
    # Multi-peaked "between" cells
    multi_peak_mask = np.array(n_peaks_list) >= 2
    if np.sum(multi_peak_mask) > 0:
        ax5 = fig.add_subplot(gs[1, 1])
        multi_peak_profiles = between_profiles[multi_peak_mask]
        for i in range(min(10, len(multi_peak_profiles))):
            ax5.plot(bin_centers, multi_peak_profiles[i], alpha=0.5)
        ax5.plot(bin_centers, np.mean(multi_peak_profiles, axis=0), 'k-', linewidth=3, label='Mean')
        for lm in LANDMARK_POSITIONS:
            ax5.axvline(lm, color='red', linestyle='--', alpha=0.3)
        ax5.set_xlabel('Position (cm)')
        ax5.set_ylabel('Activity')
        ax5.set_title(f'Multi-Peak "Between" (n={np.sum(multi_peak_mask)})', fontweight='bold')
    
    # No-peak "between" cells (flat or noisy)
    no_peak_mask = np.array(n_peaks_list) == 0
    if np.sum(no_peak_mask) > 0:
        ax6 = fig.add_subplot(gs[1, 2])
        no_peak_profiles = between_profiles[no_peak_mask]
        for i in range(min(10, len(no_peak_profiles))):
            ax6.plot(bin_centers, no_peak_profiles[i], alpha=0.5)
        ax6.plot(bin_centers, np.mean(no_peak_profiles, axis=0), 'k-', linewidth=3, label='Mean')
        for lm in LANDMARK_POSITIONS:
            ax6.axvline(lm, color='red', linestyle='--', alpha=0.3)
        ax6.set_xlabel('Position (cm)')
        ax6.set_ylabel('Activity')
        ax6.set_title(f'No-Peak "Between" (n={np.sum(no_peak_mask)})', fontweight='bold')
    
    # Panel 7: Heatmap of all "between" cells sorted by peak position
    ax7 = fig.add_subplot(gs[2, :2])
    sort_idx = np.argsort(between_peaks)
    sorted_profiles = between_profiles[sort_idx]
    
    im = ax7.imshow(sorted_profiles, aspect='auto', cmap='viridis',
                   extent=[bin_centers[0], bin_centers[-1], 0, n_between])
    for lm in LANDMARK_POSITIONS:
        ax7.axvline(lm, color='red', linestyle='--', alpha=0.7)
    ax7.set_xlabel('Position (cm)', fontsize=11)
    ax7.set_ylabel('Cell # (sorted by peak)', fontsize=11)
    ax7.set_title('"Between" Cells Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax7)
    
    # Panel 8: Summary
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')
    
    summary = f"""
    "BETWEEN" CELLS SUMMARY
    {'='*35}
    
    Total: {n_between} cells
    
    Peak count distribution:
      0 peaks (flat/noisy): {counts[0]} ({counts[0]/n_between*100:.1f}%)
      1 peak (off-landmark): {counts[1]} ({counts[1]/n_between*100:.1f}%)
      2+ peaks (multi-tuned): {counts[2]} ({counts[2]/n_between*100:.1f}%)
    
    Ramping cells: {np.sum(is_ramping)} ({np.sum(is_ramping)/n_between*100:.1f}%)
    Flat cells: {np.sum(is_flat)} ({np.sum(is_flat)/n_between*100:.1f}%)
    
    Peak position range:
      Min: {np.min(between_peaks):.1f} cm
      Max: {np.max(between_peaks):.1f} cm
      Mean: {np.mean(between_peaks):.1f} cm
    """
    
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes, fontsize=10,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle(f'Analysis of "Between Landmarks" Cells: {ANIMAL_ID}', 
                fontsize=14, fontweight='bold')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'verify_between_cells.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved 'between' cells analysis")
    
    return fig


# ============================================================================
# 3. VERIFY L1 CELLS - ADAPTATION VS SPATIAL
# ============================================================================

def verify_l1_cell_types(pca_data_path, save_dir):
    """
    Verify L1 cells: separate those peaking BEFORE vs AT/AFTER L1.
    Cells peaking before L1 are more likely adaptation responses.
    """
    print("\n" + "="*60)
    print("VERIFICATION 3: L1 Cell Types (Before vs After Landmark)")
    print("="*60)
    
    with h5py.File(pca_data_path, 'r') as f:
        spatial_profiles = f['features/spatial_profiles_zscore'][:]
        preferred_landmark = f['cells/preferred_landmark'][:]
        peak_positions = f['cells/peak_positions'][:]
        layer_labels = f['cells/layer_labels'][:].astype(str)
        bin_centers = f['metadata/bin_centers_trimmed'][:]
    
    l1_mask = preferred_landmark == 0
    l1_profiles = spatial_profiles[l1_mask]
    l1_peaks = peak_positions[l1_mask]
    l1_layers = layer_labels[l1_mask]
    
    n_l1 = np.sum(l1_mask)
    print(f"Total L1 cells: {n_l1}")
    
    # Split by peak position relative to L1 (25cm)
    before_l1 = l1_peaks < 25
    at_l1 = (l1_peaks >= 23) & (l1_peaks <= 27)  # ±2cm
    after_l1 = l1_peaks > 25
    
    print(f"  Before L1 (<25cm): {np.sum(before_l1)} ({np.sum(before_l1)/n_l1*100:.1f}%)")
    print(f"  At L1 (23-27cm): {np.sum(at_l1)} ({np.sum(at_l1)/n_l1*100:.1f}%)")
    print(f"  After L1 (>25cm): {np.sum(after_l1)} ({np.sum(after_l1)/n_l1*100:.1f}%)")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Mean profiles - Before vs After L1
    ax1 = axes[0, 0]
    
    if np.sum(before_l1) > 0:
        mean_before = np.mean(l1_profiles[before_l1], axis=0)
        sem_before = np.std(l1_profiles[before_l1], axis=0) / np.sqrt(np.sum(before_l1))
        ax1.plot(bin_centers, mean_before, 'r-', linewidth=2, label=f'Before L1 (n={np.sum(before_l1)})')
        ax1.fill_between(bin_centers, mean_before - sem_before, mean_before + sem_before, 
                        color='red', alpha=0.2)
    
    if np.sum(after_l1) > 0:
        mean_after = np.mean(l1_profiles[after_l1], axis=0)
        sem_after = np.std(l1_profiles[after_l1], axis=0) / np.sqrt(np.sum(after_l1))
        ax1.plot(bin_centers, mean_after, 'b-', linewidth=2, label=f'After L1 (n={np.sum(after_l1)})')
        ax1.fill_between(bin_centers, mean_after - sem_after, mean_after + sem_after,
                        color='blue', alpha=0.2)
    
    ax1.axvline(25, color='black', linestyle='--', linewidth=2, label='L1 (25cm)')
    for lm in LANDMARK_POSITIONS[1:]:
        ax1.axvline(lm, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Position (cm)')
    ax1.set_ylabel('Z-scored Activity')
    ax1.set_title('L1 Cells: Before vs After Landmark', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Panel 2: Layer distribution - Before vs After
    ax2 = axes[0, 1]
    layers = ['L2/3', 'L4', 'L5', 'L6']
    
    before_by_layer = [np.sum(before_l1 & (l1_layers == l)) for l in layers]
    after_by_layer = [np.sum(after_l1 & (l1_layers == l)) for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    ax2.bar(x - width/2, before_by_layer, width, label='Before L1', color='red', alpha=0.7)
    ax2.bar(x + width/2, after_by_layer, width, label='After L1', color='blue', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel('Cortical Layer')
    ax2.set_ylabel('Count')
    ax2.set_title('Before/After L1 by Layer', fontweight='bold')
    ax2.legend()
    
    # Panel 3: Proportion before L1 by layer
    ax3 = axes[0, 2]
    
    prop_before = []
    for layer in layers:
        layer_mask = l1_layers == layer
        n_layer = np.sum(layer_mask)
        if n_layer > 0:
            prop = np.sum(before_l1 & layer_mask) / n_layer * 100
        else:
            prop = 0
        prop_before.append(prop)
    
    colors = ['#1E88E5', '#FF9800', '#4CAF50', '#E53935']
    ax3.bar(layers, prop_before, color=colors, edgecolor='black', alpha=0.8)
    ax3.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Cortical Layer')
    ax3.set_ylabel('% L1 Cells Peaking BEFORE Landmark')
    ax3.set_title('Proportion Peaking Before L1 by Layer', fontweight='bold')
    ax3.set_ylim(0, 100)
    
    # Panel 4: Early slope comparison
    ax4 = axes[1, 0]
    
    # Calculate early slope (first 15 bins after track start)
    early_end = np.searchsorted(bin_centers, 25)
    
    slopes_before = []
    slopes_after = []
    
    for i, profile in enumerate(l1_profiles):
        early_region = profile[:early_end]
        if len(early_region) > 2:
            slope = np.polyfit(np.arange(len(early_region)), early_region, 1)[0]
            if before_l1[i]:
                slopes_before.append(slope)
            elif after_l1[i]:
                slopes_after.append(slope)
    
    ax4.hist(slopes_before, bins=20, alpha=0.6, color='red', label=f'Before L1', edgecolor='black')
    ax4.hist(slopes_after, bins=20, alpha=0.6, color='blue', label=f'After L1', edgecolor='black')
    ax4.axvline(0, color='black', linestyle='--')
    ax4.set_xlabel('Early Slope (onset → L1)')
    ax4.set_ylabel('Count')
    ax4.set_title('Early Slope Distribution', fontweight='bold')
    ax4.legend()
    
    # Panel 5: Heatmap of "before L1" cells
    ax5 = axes[1, 1]
    if np.sum(before_l1) > 0:
        before_profiles = l1_profiles[before_l1]
        sort_idx = np.argsort(l1_peaks[before_l1])
        im = ax5.imshow(before_profiles[sort_idx], aspect='auto', cmap='viridis',
                       extent=[bin_centers[0], bin_centers[-1], 0, np.sum(before_l1)])
        ax5.axvline(25, color='red', linestyle='--', linewidth=2)
        ax5.set_xlabel('Position (cm)')
        ax5.set_ylabel('Cell #')
        ax5.set_title(f'Cells Peaking BEFORE L1 (n={np.sum(before_l1)})', fontweight='bold')
        plt.colorbar(im, ax=ax5)
    
    # Panel 6: Heatmap of "after L1" cells
    ax6 = axes[1, 2]
    if np.sum(after_l1) > 0:
        after_profiles = l1_profiles[after_l1]
        sort_idx = np.argsort(l1_peaks[after_l1])
        im = ax6.imshow(after_profiles[sort_idx], aspect='auto', cmap='viridis',
                       extent=[bin_centers[0], bin_centers[-1], 0, np.sum(after_l1)])
        ax6.axvline(25, color='red', linestyle='--', linewidth=2)
        ax6.set_xlabel('Position (cm)')
        ax6.set_ylabel('Cell #')
        ax6.set_title(f'Cells Peaking AFTER L1 (n={np.sum(after_l1)})', fontweight='bold')
        plt.colorbar(im, ax=ax6)
    
    plt.suptitle(f'L1 Cell Verification: Before vs After Landmark: {ANIMAL_ID}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'verify_l1_before_after.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved L1 before/after analysis")
    
    return fig


# ============================================================================
# 4. SESSION-SPECIFIC RAW DATA CHECK
# ============================================================================

def check_session_raw_data(pca_data_path, save_dir):
    """
    Compare raw data characteristics across sessions.
    Look for sessions with unusual properties.
    """
    print("\n" + "="*60)
    print("VERIFICATION 4: Session-Specific Data Quality")
    print("="*60)
    
    with h5py.File(pca_data_path, 'r') as f:
        spatial_profiles = f['features/spatial_profiles'][:]
        spatial_profiles_zscore = f['features/spatial_profiles_zscore'][:]
        session_labels = f['cells/session_labels'][:].astype(str)
        preferred_landmark = f['cells/preferred_landmark'][:]
        bin_centers = f['metadata/bin_centers_trimmed'][:]
    
    unique_sessions = sorted(np.unique(session_labels), key=lambda x: int(x.replace('Day', '')))
    
    fig, axes = plt.subplots(3, len(unique_sessions), figsize=(3*len(unique_sessions), 10))
    
    session_stats = []
    
    for si, session in enumerate(unique_sessions):
        mask = session_labels == session
        session_profiles = spatial_profiles[mask]
        session_zscore = spatial_profiles_zscore[mask]
        session_lm = preferred_landmark[mask]
        
        n_cells = np.sum(mask)
        
        # Row 1: Raw profile distribution (percentiles)
        ax1 = axes[0, si]
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            profile_p = np.percentile(session_profiles, p, axis=0)
            ax1.plot(bin_centers, profile_p, label=f'{p}%', alpha=0.7)
        ax1.set_title(f'{session} (n={n_cells})', fontweight='bold')
        if si == 0:
            ax1.set_ylabel('Raw Activity')
        ax1.legend(fontsize=6)
        
        # Row 2: Landmark preference distribution
        ax2 = axes[1, si]
        lm_counts = [np.sum(session_lm == i) for i in range(4)]
        between_count = np.sum(session_lm == -1)
        
        colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', 'gray']
        ax2.bar(['L1', 'L2', 'L3', 'L4', 'Btw'], 
               lm_counts + [between_count], color=colors, edgecolor='black')
        if si == 0:
            ax2.set_ylabel('Count')
        
        # Row 3: Profile variance distribution
        ax3 = axes[2, si]
        profile_vars = np.var(session_profiles, axis=1)
        ax3.hist(profile_vars, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax3.axvline(np.median(profile_vars), color='red', linestyle='--', 
                   label=f'Median: {np.median(profile_vars):.4f}')
        if si == 0:
            ax3.set_ylabel('Count')
        ax3.set_xlabel('Profile Variance')
        ax3.legend(fontsize=7)
        
        # Collect stats
        session_stats.append({
            'session': session,
            'n_cells': n_cells,
            'mean_activity': np.mean(session_profiles),
            'median_variance': np.median(profile_vars),
            'pct_between': between_count / n_cells * 100 if n_cells > 0 else 0,
            'pct_l1': lm_counts[0] / n_cells * 100 if n_cells > 0 else 0
        })
    
    plt.suptitle(f'Session Data Quality Check: {ANIMAL_ID}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'verify_session_quality.png'), dpi=150, bbox_inches='tight')
        print(f"✓ Saved session quality check")
    
    # Print summary table
    print("\nSession Quality Summary:")
    print(f"{'Session':<10} {'N cells':<10} {'Mean Act':<12} {'Med Var':<12} {'% Between':<12} {'% L1':<10}")
    print("-" * 70)
    for stats in session_stats:
        print(f"{stats['session']:<10} {stats['n_cells']:<10} {stats['mean_activity']:<12.4f} "
              f"{stats['median_variance']:<12.4f} {stats['pct_between']:<12.1f} {stats['pct_l1']:<10.1f}")
    
    # Flag problematic sessions
    print("\nPotential Issues:")
    for stats in session_stats:
        issues = []
        if stats['pct_between'] > 30:
            issues.append(f"High 'between' rate ({stats['pct_between']:.1f}%)")
        if stats['n_cells'] < 100:
            issues.append(f"Low cell count ({stats['n_cells']})")
        if stats['median_variance'] < 0.001:
            issues.append(f"Low variance (possible bad imaging)")
        
        if issues:
            print(f"  {stats['session']}: {', '.join(issues)}")
    
    return fig


# ============================================================================
# MAIN
# ============================================================================

def run_all_verifications():
    """Run all verification checks."""
    print("="*70)
    print(f"DATA VERIFICATION: {ANIMAL_ID}")
    print("="*70)
    
    os.makedirs(FIGURE_DIR, exist_ok=True)
    
    # Verification 1: Example cells
    fig1 = plot_example_cells_by_landmark(PCA_DATA_PATH, FIGURE_DIR)
    
    # Verification 2: "Between" cells
    fig2 = analyze_between_cells(PCA_DATA_PATH, FIGURE_DIR)
    
    # Verification 3: L1 before vs after
    fig3 = verify_l1_cell_types(PCA_DATA_PATH, FIGURE_DIR)
    
    # Verification 4: Session quality
    fig4 = check_session_raw_data(PCA_DATA_PATH, FIGURE_DIR)
    
    print("\n" + "="*70)
    print("VERIFICATION COMPLETE!")
    print("="*70)
    print(f"Figures saved to: {FIGURE_DIR}")
    
    plt.show()


if __name__ == "__main__":
    run_all_verifications()