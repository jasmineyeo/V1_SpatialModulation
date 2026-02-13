"""
PCA_LandmarkAlignedAnalysis.py (FIXED VERSION - Non-circular shifts)
Performs PCA on landmark-aligned spatial profiles to test whether timing jitter
obscures the 4th landmark signal in the positional (PC1) axis.

KEY FIX: Uses NON-CIRCULAR shifts (no wrap-around)
- Shifts are padded with zeros at edges
- This preserves spatial meaning of positions

JSY, 01/2026
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.ndimage import gaussian_filter1d


# ============================================================================
# CONFIGURATION
# ============================================================================

PCA_DATA_PATH = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\JSY054_pca_data.h5"
FIGURE_DIR = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\PCA\figures_aligned"

LANDMARK_POSITIONS = [25, 55, 85, 115]  # cm
INTER_LANDMARK_DISTANCE = 30  # cm

# Alignment parameters
MAX_SHIFT_CM = 15  # Maximum shift allowed (±15 cm)
TEMPLATE_SIGMA_CM = 8  # Width of Gaussian peaks in template

N_COMPONENTS = 10
DPI = 150


# ============================================================================
# FIXED ALIGNMENT FUNCTIONS (NON-CIRCULAR)
# ============================================================================

def create_landmark_template(bin_centers, landmark_positions, sigma_cm=8):
    """Create ground truth template with Gaussian peaks at landmark locations."""
    template = np.zeros_like(bin_centers, dtype=float)
    
    for lm_pos in landmark_positions:
        gaussian_peak = np.exp(-0.5 * ((bin_centers - lm_pos) / sigma_cm) ** 2)
        template += gaussian_peak
    
    # Normalize
    template = (template - np.mean(template)) / np.std(template)
    
    return template


def shift_profile_noncircular(profile, shift_bins):
    """
    Shift profile WITHOUT circular wrapping.
    Positive shift = shift RIGHT (data moves to higher indices)
    Negative shift = shift LEFT (data moves to lower indices)
    Edges are filled with zeros.
    
    Parameters:
    -----------
    profile : array
        1D profile to shift
    shift_bins : int
        Number of bins to shift (positive=right, negative=left)
    
    Returns:
    --------
    shifted : array
        Shifted profile with zero-padding at edges
    """
    shifted = np.zeros_like(profile)
    
    if shift_bins > 0:
        # Shift RIGHT: move data to higher indices
        shifted[shift_bins:] = profile[:-shift_bins]
    elif shift_bins < 0:
        # Shift LEFT: move data to lower indices
        shifted[:shift_bins] = profile[-shift_bins:]
    else:
        # No shift
        shifted = profile.copy()
    
    return shifted


def find_optimal_shift_noncircular(profile, template, max_shift_bins):
    """
    Find the NON-CIRCULAR shift that maximizes correlation with template.
    
    Parameters:
    -----------
    profile : array
        Cell's spatial profile (1D)
    template : array
        Ground truth template
    max_shift_bins : int
        Maximum shift in bins (±max_shift_bins)
    
    Returns:
    --------
    optimal_shift : int
        Optimal shift in bins (positive = shift RIGHT)
    max_corr : float
        Maximum correlation achieved
    all_corrs : array
        Correlation at each shift
    """
    shift_range = np.arange(-max_shift_bins, max_shift_bins + 1)
    correlations = np.zeros(len(shift_range))
    
    for i, shift in enumerate(shift_range):
        # Non-circular shift
        shifted_profile = shift_profile_noncircular(profile, shift)
        
        # Compute correlation with template
        # Only compute on non-zero regions to avoid edge artifacts
        if shift >= 0:
            # Shifted right, use region [shift:]
            valid_region = slice(shift, None)
        else:
            # Shifted left, use region [:shift]
            valid_region = slice(None, len(profile) + shift)
        
        valid_profile = shifted_profile[valid_region]
        valid_template = template[valid_region]
        
        if len(valid_profile) > 10:  # Need enough points for correlation
            corr, _ = pearsonr(valid_profile, valid_template)
            correlations[i] = corr
        else:
            correlations[i] = -1  # Invalid
    
    # Find optimal shift
    best_idx = np.argmax(correlations)
    optimal_shift = shift_range[best_idx]
    max_corr = correlations[best_idx]
    
    return optimal_shift, max_corr, correlations


def align_profiles_to_landmarks(profiles, bin_centers, landmark_positions,
                                 max_shift_cm=15, template_sigma_cm=8,
                                 verbose=True):
    """
    Align all cell profiles to landmark template using NON-CIRCULAR shifts.
    
    Returns:
    --------
    aligned_profiles : array (n_cells, n_bins)
        Aligned spatial profiles (NO wrap-around)
    optimal_shifts : array (n_cells,)
        Optimal shift for each cell in bins (positive=right, negative=left)
    max_correlations : array (n_cells,)
        Maximum correlation achieved for each cell
    template : array
        The template used for alignment
    """
    n_cells, n_bins = profiles.shape
    
    # Calculate bin spacing and convert max_shift to bins
    bin_spacing_cm = np.mean(np.diff(bin_centers))
    max_shift_bins = int(np.round(max_shift_cm / bin_spacing_cm))
    
    if verbose:
        print(f"\nAligning profiles to landmark template (NON-CIRCULAR)...")
        print(f"  Template sigma: {template_sigma_cm} cm")
        print(f"  Max shift: ±{max_shift_cm} cm = ±{max_shift_bins} bins")
        print(f"  Bin spacing: {bin_spacing_cm:.2f} cm")
        print(f"  ⚠ Using NON-CIRCULAR shifts (no wrap-around)")
    
    # Create template
    template = create_landmark_template(bin_centers, landmark_positions,
                                        sigma_cm=template_sigma_cm)
    
    # Align each cell
    aligned_profiles = np.zeros_like(profiles)
    optimal_shifts = np.zeros(n_cells, dtype=int)
    max_correlations = np.zeros(n_cells)
    
    for cell_idx in range(n_cells):
        profile = profiles[cell_idx]
        
        # Find optimal NON-CIRCULAR shift
        opt_shift, max_corr, _ = find_optimal_shift_noncircular(
            profile, template, max_shift_bins
        )
        
        # Apply NON-CIRCULAR shift
        aligned_profiles[cell_idx] = shift_profile_noncircular(profile, opt_shift)
        optimal_shifts[cell_idx] = opt_shift
        max_correlations[cell_idx] = max_corr
    
    if verbose:
        # Summary statistics
        print(f"\nAlignment summary:")
        print(f"  Mean optimal shift: {np.mean(optimal_shifts):.1f} bins ({np.mean(optimal_shifts) * bin_spacing_cm:.1f} cm)")
        print(f"  Std of shifts: {np.std(optimal_shifts):.1f} bins")
        print(f"  Mean max correlation: {np.mean(max_correlations):.3f}")
        print(f"  Cells shifted left (<0): {np.sum(optimal_shifts < 0)}")
        print(f"  Cells not shifted (=0): {np.sum(optimal_shifts == 0)}")
        print(f"  Cells shifted right (>0): {np.sum(optimal_shifts > 0)}")
    
    return aligned_profiles, optimal_shifts, max_correlations, template


# ============================================================================
# DATA LOADING
# ============================================================================

def load_pca_data(filepath):
    """Load aggregated PCA data from HDF5 file."""
    print(f"Loading data from: {filepath}")
    
    data = {}
    
    with h5py.File(filepath, 'r') as f:
        # Metadata
        data['animal_id'] = f['metadata'].attrs['animal_id']
        data['n_sessions'] = f['metadata'].attrs['n_sessions']
        data['n_cells'] = f['metadata'].attrs['n_cells_total']
        data['bin_centers'] = f['metadata/bin_centers_trimmed'][:]
        data['landmark_positions'] = f['metadata/landmark_positions'][:]
        
        # Cell labels
        data['session_labels'] = f['cells/session_labels'][:].astype(str)
        data['layer_labels'] = f['cells/layer_labels'][:].astype(str)
        data['preferred_landmark'] = f['cells/preferred_landmark'][:]
        data['peak_positions'] = f['cells/peak_positions'][:]
        
        # Features - use session-corrected if available
        if 'features/spatial_profiles_session_corrected' in f:
            data['spatial_profiles_zscore'] = f['features/spatial_profiles_session_corrected'][:]
            print(f"  Using session-corrected profiles")
        else:
            data['spatial_profiles_zscore'] = f['features/spatial_profiles_zscore'][:]
            print(f"  Using original z-scored profiles")
        
        data['spatial_profiles'] = f['features/spatial_profiles'][:]
    
    print(f"  Animal: {data['animal_id']}")
    print(f"  Cells: {data['n_cells']}")
    print(f"  Features: {data['spatial_profiles'].shape[1]} spatial bins")
    
    return data


# ============================================================================
# PCA FUNCTIONS
# ============================================================================

def run_pca(profiles, n_components=10, label=""):
    """Run PCA on spatial profiles."""
    print(f"\nRunning PCA {label}...")
    
    pca = PCA(n_components=n_components)
    pc_scores = pca.fit_transform(profiles)
    
    pca_results = {
        'pca': pca,
        'pc_scores': pc_scores,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
        'components': pca.components_,
        'n_components': n_components
    }
    
    print(f"  Variance explained by PC1: {pca_results['explained_variance_ratio'][0]*100:.1f}%")
    print(f"  Variance explained by first 3 PCs: {pca_results['cumulative_variance_ratio'][2]*100:.1f}%")
    
    return pca_results


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_template_and_examples(template, bin_centers, landmark_positions,
                                profiles_original, profiles_aligned,
                                optimal_shifts, max_correlations,
                                n_examples=6, save_path=None):
    """Plot the template and example cells before/after alignment."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Top row: Template
    ax_template = axes[0, 1]
    ax_template.plot(bin_centers, template, 'k-', linewidth=2, label='Template')
    for i, lm_pos in enumerate(landmark_positions):
        ax_template.axvline(lm_pos, color='green', linestyle='--', alpha=0.6)
        ax_template.text(lm_pos, ax_template.get_ylim()[1] * 0.9, f'L{i+1}',
                        ha='center', fontsize=10, color='green')
    ax_template.set_xlabel('Position (cm)')
    ax_template.set_ylabel('Template (z-scored)')
    ax_template.set_title('Landmark Template (4 Gaussian peaks)', fontweight='bold')
    ax_template.grid(alpha=0.3)
    
    axes[0, 0].axis('off')
    axes[0, 2].axis('off')
    
    # Select example cells
    shift_percentiles = [10, 30, 50, 70, 90, 95]
    example_indices = []
    for p in shift_percentiles:
        target_shift = np.percentile(optimal_shifts, p)
        idx = np.argmin(np.abs(optimal_shifts - target_shift))
        if idx not in example_indices:
            example_indices.append(idx)
    
    while len(example_indices) < n_examples:
        idx = np.random.randint(0, len(profiles_original))
        if idx not in example_indices:
            example_indices.append(idx)
    
    example_indices = example_indices[:n_examples]
    
    # Plot examples
    for i, cell_idx in enumerate(example_indices):
        row = 1 + i // 3
        col = i % 3
        ax = axes[row, col]
        
        orig = profiles_original[cell_idx]
        aligned = profiles_aligned[cell_idx]
        shift = optimal_shifts[cell_idx]
        corr = max_correlations[cell_idx]
        
        ax.plot(bin_centers, orig, 'b-', alpha=0.5, linewidth=1.5, label='Original')
        ax.plot(bin_centers, aligned, 'r-', linewidth=2, label='Aligned')
        
        for lm_pos in landmark_positions:
            ax.axvline(lm_pos, color='green', linestyle='--', alpha=0.4)
        
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Activity (z-scored)')
        ax.set_title(f'Cell {cell_idx}: shift={shift:+d} bins, r={corr:.2f}', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(alpha=0.3)
    
    plt.suptitle('Landmark Alignment: Template and Example Cells\n(NON-CIRCULAR shifts - no wrap-around)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved template examples: {os.path.basename(save_path)}")
    
    return fig


def plot_shift_distribution(optimal_shifts, max_correlations, bin_spacing_cm,
                            save_path=None):
    """Plot distribution of optimal shifts and correlations."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Left: Shift histogram
    ax1 = axes[0]
    shifts_cm = optimal_shifts * bin_spacing_cm
    ax1.hist(shifts_cm, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='No shift')
    ax1.axvline(np.mean(shifts_cm), color='orange', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(shifts_cm):.1f}cm')
    ax1.set_xlabel('Optimal Shift (cm)\nPositive=Right, Negative=Left')
    ax1.set_ylabel('Number of Cells')
    ax1.set_title('Distribution of Optimal Shifts\n(NON-CIRCULAR)', fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Middle: Correlation histogram
    ax2 = axes[1]
    ax2.hist(max_correlations, bins=30, color='coral', edgecolor='black', alpha=0.7)
    ax2.axvline(np.mean(max_correlations), color='red', linestyle='-', linewidth=2,
                label=f'Mean={np.mean(max_correlations):.2f}')
    ax2.set_xlabel('Max Correlation with Template')
    ax2.set_ylabel('Number of Cells')
    ax2.set_title('Template Correlation After Alignment', fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Right: Shift vs Correlation scatter
    ax3 = axes[2]
    ax3.scatter(shifts_cm, max_correlations, alpha=0.5, s=20, c='steelblue')
    ax3.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Optimal Shift (cm)')
    ax3.set_ylabel('Max Correlation')
    ax3.set_title('Shift vs. Correlation', fontweight='bold')
    ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved shift distribution: {os.path.basename(save_path)}")
    
    return fig


def plot_pc_loadings_comparison(pca_original, pca_aligned, bin_centers,
                                 landmark_positions, save_path=None):
    """Compare PC loadings between original and aligned PCA."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for pc_idx in range(2):
        # Original loadings
        ax_orig = axes[pc_idx, 0]
        loadings_orig = pca_original['components'][pc_idx]
        var_orig = pca_original['explained_variance_ratio'][pc_idx] * 100
        
        ax_orig.plot(bin_centers, loadings_orig, 'b-', linewidth=2)
        ax_orig.fill_between(bin_centers, loadings_orig, 0,
                            where=(loadings_orig > 0), alpha=0.3, color='blue')
        ax_orig.fill_between(bin_centers, loadings_orig, 0,
                            where=(loadings_orig < 0), alpha=0.3, color='red')
        ax_orig.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        for i, lm_pos in enumerate(landmark_positions):
            ax_orig.axvline(lm_pos, color='green', linestyle='--', alpha=0.6)
            ax_orig.text(lm_pos, ax_orig.get_ylim()[1], f'L{i+1}',
                        ha='center', va='bottom', fontsize=9, color='green')
        
        ax_orig.set_xlabel('Position (cm)')
        ax_orig.set_ylabel('Loading')
        ax_orig.set_title(f'ORIGINAL: PC{pc_idx+1} ({var_orig:.1f}%)',
                         fontsize=11, fontweight='bold', color='blue')
        ax_orig.grid(alpha=0.3)
        
        # Aligned loadings
        ax_aligned = axes[pc_idx, 1]
        loadings_aligned = pca_aligned['components'][pc_idx]
        var_aligned = pca_aligned['explained_variance_ratio'][pc_idx] * 100
        
        ax_aligned.plot(bin_centers, loadings_aligned, 'r-', linewidth=2)
        ax_aligned.fill_between(bin_centers, loadings_aligned, 0,
                               where=(loadings_aligned > 0), alpha=0.3, color='blue')
        ax_aligned.fill_between(bin_centers, loadings_aligned, 0,
                               where=(loadings_aligned < 0), alpha=0.3, color='red')
        ax_aligned.axhline(0, color='gray', linestyle='-', alpha=0.5)
        
        for i, lm_pos in enumerate(landmark_positions):
            ax_aligned.axvline(lm_pos, color='green', linestyle='--', alpha=0.6)
            ax_aligned.text(lm_pos, ax_aligned.get_ylim()[1], f'L{i+1}',
                           ha='center', va='bottom', fontsize=9, color='green')
        
        ax_aligned.set_xlabel('Position (cm)')
        ax_aligned.set_ylabel('Loading')
        ax_aligned.set_title(f'ALIGNED: PC{pc_idx+1} ({var_aligned:.1f}%)',
                            fontsize=11, fontweight='bold', color='red')
        ax_aligned.grid(alpha=0.3)
    
    plt.suptitle('PC Loadings Comparison: Original vs. Landmark-Aligned\n' +
                '(NON-CIRCULAR shifts - Does alignment reveal 4 peaks instead of 3?)',
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved loadings comparison: {os.path.basename(save_path)}")
    
    return fig


def plot_grand_mean_comparison(profiles_original, profiles_aligned,
                                bin_centers, landmark_positions, save_path=None):
    """Compare grand mean profile (all cells) before/after alignment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    n_cells = profiles_original.shape[0]
    
    # Original
    ax1 = axes[0]
    mean_orig = np.mean(profiles_original, axis=0)
    sem_orig = np.std(profiles_original, axis=0) / np.sqrt(n_cells)
    
    ax1.plot(bin_centers, mean_orig, 'b-', linewidth=2.5, label='Mean')
    ax1.fill_between(bin_centers, mean_orig - sem_orig, mean_orig + sem_orig,
                    color='blue', alpha=0.3, label='±SEM')
    
    for i, lm_pos in enumerate(landmark_positions):
        ax1.axvline(lm_pos, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax1.text(lm_pos, ax1.get_ylim()[1] * 0.95, f'L{i+1}',
                ha='center', fontsize=11, color='green', fontweight='bold')
    
    ax1.set_xlabel('Position (cm)', fontsize=12)
    ax1.set_ylabel('Activity (z-scored)', fontsize=12)
    ax1.set_title(f'ORIGINAL Grand Mean (n={n_cells} cells)',
                 fontsize=13, fontweight='bold', color='blue')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    
    # Aligned
    ax2 = axes[1]
    mean_aligned = np.mean(profiles_aligned, axis=0)
    sem_aligned = np.std(profiles_aligned, axis=0) / np.sqrt(n_cells)
    
    ax2.plot(bin_centers, mean_aligned, 'r-', linewidth=2.5, label='Mean')
    ax2.fill_between(bin_centers, mean_aligned - sem_aligned, mean_aligned + sem_aligned,
                    color='red', alpha=0.3, label='±SEM')
    
    for i, lm_pos in enumerate(landmark_positions):
        ax2.axvline(lm_pos, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
        ax2.text(lm_pos, ax2.get_ylim()[1] * 0.95, f'L{i+1}',
                ha='center', fontsize=11, color='green', fontweight='bold')
    
    ax2.set_xlabel('Position (cm)', fontsize=12)
    ax2.set_ylabel('Activity (z-scored)', fontsize=12)
    ax2.set_title(f'ALIGNED Grand Mean (n={n_cells} cells)',
                 fontsize=13, fontweight='bold', color='red')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)
    
    plt.suptitle('Grand Mean Profile Comparison (NON-CIRCULAR alignment)\n' +
                '(Does alignment produce 4 equal peaks at landmark locations?)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        print(f"  Saved grand mean comparison: {os.path.basename(save_path)}")
    
    return fig


def save_aligned_profiles_to_h5(pca_data_path, profiles_aligned, optimal_shifts,
                                 max_correlations, template, max_shift_cm,
                                 template_sigma_cm):
    """Save aligned profiles to HDF5 file."""
    print(f"\nSaving aligned profiles to: {pca_data_path}")
    
    with h5py.File(pca_data_path, 'a') as f:
        # Save aligned profiles
        if 'features/spatial_profiles_aligned' in f:
            del f['features/spatial_profiles_aligned']
        f.create_dataset('features/spatial_profiles_aligned', data=profiles_aligned)
        
        # Create/update alignment group
        if 'alignment' in f:
            del f['alignment']
        align_grp = f.create_group('alignment')
        
        align_grp.create_dataset('optimal_shifts', data=optimal_shifts)
        align_grp.create_dataset('max_correlations', data=max_correlations)
        align_grp.create_dataset('template', data=template)
        
        # Store parameters
        align_grp.attrs['max_shift_cm'] = max_shift_cm
        align_grp.attrs['template_sigma_cm'] = template_sigma_cm
        align_grp.attrs['method'] = 'noncircular_shift_template_correlation'
        align_grp.attrs['shift_direction_convention'] = 'positive=right, negative=left'
    
    print("  Saved: features/spatial_profiles_aligned")
    print("  Saved: alignment/ (shifts, correlations, template, parameters)")
    print("  ⚠ Method: NON-CIRCULAR shifts (no wrap-around)")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_aligned_pca_analysis(pca_data_path, figure_dir,
                              max_shift_cm=15, template_sigma_cm=8,
                              n_components=10):
    """Main function to run landmark-aligned PCA analysis."""
    print("=" * 80)
    print("LANDMARK-ALIGNED PCA ANALYSIS (NON-CIRCULAR SHIFTS)")
    print("=" * 80)
    print(f"Purpose: Test if timing jitter obscures the 4th landmark in PC1")
    print(f"Method: NON-CIRCULAR shift each cell to maximize correlation with template")
    print(f"        (No wrap-around - edges are zero-padded)")
    print("=" * 80)
    
    # Create figure directory
    os.makedirs(figure_dir, exist_ok=True)
    
    # Load data
    data = load_pca_data(pca_data_path)
    profiles_original = data['spatial_profiles_zscore']
    bin_centers = data['bin_centers']
    landmark_positions = data['landmark_positions']
    
    bin_spacing_cm = np.mean(np.diff(bin_centers))
    
    # Align profiles (NON-CIRCULAR)
    profiles_aligned, optimal_shifts, max_correlations, template = \
        align_profiles_to_landmarks(
            profiles_original, bin_centers, landmark_positions,
            max_shift_cm=max_shift_cm, template_sigma_cm=template_sigma_cm
        )
    
    # Run PCA
    pca_original = run_pca(profiles_original, n_components=n_components,
                           label="(Original)")
    pca_aligned = run_pca(profiles_aligned, n_components=n_components,
                          label="(Aligned)")
    
    # Generate figures
    print("\nGenerating figures...")
    
    plot_template_and_examples(
        template, bin_centers, landmark_positions,
        profiles_original, profiles_aligned,
        optimal_shifts, max_correlations,
        save_path=os.path.join(figure_dir, 'alignment_template_examples_noncircular.png')
    )
    
    plot_shift_distribution(
        optimal_shifts, max_correlations, bin_spacing_cm,
        save_path=os.path.join(figure_dir, 'alignment_shift_distribution_noncircular.png')
    )
    
    plot_pc_loadings_comparison(
        pca_original, pca_aligned, bin_centers, landmark_positions,
        save_path=os.path.join(figure_dir, 'pca_loadings_comparison_noncircular.png')
    )
    
    plot_grand_mean_comparison(
        profiles_original, profiles_aligned,
        bin_centers, landmark_positions,
        save_path=os.path.join(figure_dir, 'grand_mean_comparison_noncircular.png')
    )
    
    # Save results
    save_aligned_profiles_to_h5(
        pca_data_path, profiles_aligned, optimal_shifts,
        max_correlations, template, max_shift_cm, template_sigma_cm
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print(f"\nKey change: NON-CIRCULAR shifts (no wrap-around)")
    print(f"  - Positive shifts move data RIGHT")
    print(f"  - Negative shifts move data LEFT")
    print(f"  - Edges are zero-padded (no wrap-around)")
    
    plt.show()
    
    return {
        'pca_original': pca_original,
        'pca_aligned': pca_aligned,
        'profiles_aligned': profiles_aligned,
        'optimal_shifts': optimal_shifts,
        'max_correlations': max_correlations,
        'template': template,
        'data': data
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    results = run_aligned_pca_analysis(
        pca_data_path=PCA_DATA_PATH,
        figure_dir=FIGURE_DIR,
        max_shift_cm=MAX_SHIFT_CM,
        template_sigma_cm=TEMPLATE_SIGMA_CM,
        n_components=N_COMPONENTS
    )