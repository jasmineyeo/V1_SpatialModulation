"""
LandmarkPrefernce_SingleSessionAnalysis.py
Standalone script for analyzing landmark preferences across cortical layers in a single session.
You have to run this and then LandmarkPrefernce_CompareSessionsWithinAnimal.py to compare across sessions within an animal,
and then LandmarkPrefernce_ComprehensiveAcrossAnimals.py for comprehensive analysis (across sessions, across animals)

JSY, 11/2025
"""

import re
import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
import seaborn as sns
from scipy import stats
from scipy.ndimage import gaussian_filter1d
import h5py
from helper import files
from helper import TwoP
from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
import glob
import matplotlib.colors as mcolors

# Parula colormap (MATLAB-compatible, perceptually uniform blue→yellow)
_PARULA_COLORS = [
    (0.2422, 0.1504, 0.6603),
    (0.2108, 0.3706, 0.9717),
    (0.0196, 0.5804, 0.8745),
    (0.0863, 0.6510, 0.7490),
    (0.1961, 0.6980, 0.6039),
    (0.3647, 0.7412, 0.5176),
    (0.6275, 0.7647, 0.3843),
    (0.8510, 0.7882, 0.1961),
    (0.9686, 0.8235, 0.0667),
    (0.9765, 0.9843, 0.0510),
]
PARULA = mcolors.LinearSegmentedColormap.from_list('parula', _PARULA_COLORS)

# ============================================================================
# PHASE 1: LANDMARK RESPONSE IDENTIFICATION
# ============================================================================
def identify_landmark_responses(normalized_spatial_activity, bin_centers, 
                                landmark_positions, 
                                landmark_windows_config=None,  # NEW: per-landmark config
                                landmark_window=10.0,  # fallback if config not provided
                                boundary_exclusion=(10, 10), smoothing_sigma=1.0,
                                exclude_first_bins=10, exclude_last_bins=10):
    """
    Identify which landmark each cell prefers based on peak responses.
    
    Parameters:
    -----------
    landmark_windows_config : list of dict, optional
        Per-landmark window configuration. Each dict should have:
        - 'before': cm to extend before landmark
        - 'after': cm to extend after landmark
        Example: [
            {'before': 12, 'after': 10},  # L1 - constrained by onset
            {'before': 15, 'after': 10},  # L2
            {'before': 15, 'after': 10},  # L3
            {'before': 15, 'after': 5},   # L4 - constrained by corridor end
        ]
        If None, uses symmetric ±landmark_window for all landmarks.
    landmark_window : float
        Fallback symmetric window if landmark_windows_config not provided
    exclude_first_bins : int
        Exclude cells whose GLOBAL peak is in the first N bins (onset response)
    exclude_last_bins : int
        Exclude cells whose GLOBAL peak is in the last N bins (reward/tunnel response)
    
    Returns:
    --------
    results : dict
        Now includes 'rejected_cells' with indices and reasons for rejection
    """
    
    n_cells, n_trials, n_bins = normalized_spatial_activity.shape
    n_landmarks = len(landmark_positions)
    
    # Calculate corridor boundaries
    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)
    
    # Define regions for display/analysis
    start_exclude, end_exclude = boundary_exclusion
    min_allowed = min_pos + start_exclude
    max_allowed = max_pos - end_exclude
    
    # Calculate bin thresholds for global peak exclusion
    bin_spacing = np.mean(np.diff(bin_centers))
    onset_threshold_cm = min_pos + (exclude_first_bins * bin_spacing)
    end_threshold_cm = max_pos - (exclude_last_bins * bin_spacing)
    
    print(f"\n=== LANDMARK PREFERENCE IDENTIFICATION ===")
    print(f"Corridor: {min_pos:.1f} to {max_pos:.1f} cm ({n_bins} bins)")
    print(f"Bin spacing: {bin_spacing:.2f} cm")
    print(f"Global peak exclusion:")
    print(f"  - First {exclude_first_bins} bins (< {onset_threshold_cm:.1f} cm): onset responses")
    print(f"  - Last {exclude_last_bins} bins (> {end_threshold_cm:.1f} cm): reward/tunnel responses")
    print(f"Landmarks at: {landmark_positions} cm")
    
    # Define landmark windows - PER-LANDMARK CONFIGURATION
    landmark_windows = []
    
    if landmark_windows_config is not None:
        print(f"Using per-landmark window configuration:")
        for i, lm_pos in enumerate(landmark_positions):
            if i < len(landmark_windows_config):
                config = landmark_windows_config[i]
                lm_min = lm_pos - config['before']
                lm_max = lm_pos + config['after']
                print(f"  Landmark {i+1} at {lm_pos} cm: [{lm_min:.1f}, {lm_max:.1f}] cm "
                      f"(-{config['before']}, +{config['after']})")
            else:
                # Fallback to symmetric if not enough configs
                lm_min = lm_pos - landmark_window
                lm_max = lm_pos + landmark_window
                print(f"  Landmark {i+1} at {lm_pos} cm: [{lm_min:.1f}, {lm_max:.1f}] cm (symmetric fallback)")
            landmark_windows.append((lm_min, lm_max))
    else:
        print(f"Using symmetric windows: ±{landmark_window} cm")
        for i, lm_pos in enumerate(landmark_positions):
            lm_min = lm_pos - landmark_window
            lm_max = lm_pos + landmark_window
            print(f"  Landmark {i+1} at {lm_pos} cm: [{lm_min:.1f}, {lm_max:.1f}] cm")
            landmark_windows.append((lm_min, lm_max))
    
    # Compute mean response profiles across trials
    mean_profiles = np.mean(normalized_spatial_activity, axis=1)
    
    # Apply Gaussian smoothing
    if smoothing_sigma > 0:
        for cell in range(n_cells):
            mean_profiles[cell] = gaussian_filter1d(mean_profiles[cell], sigma=smoothing_sigma)
    
    # Initialize results
    preferred_landmark = np.full(n_cells, -1, dtype=int)  # -1 = invalid
    landmark_responses = np.zeros((n_cells, n_landmarks))
    preference_strength = np.zeros(n_cells)
    peak_positions = np.zeros(n_cells)
    global_peak_bins = np.zeros(n_cells, dtype=int)
    valid_cells = np.zeros(n_cells, dtype=bool)
    
    # NEW: Track rejected cells by reason
    rejected_onset_indices = []
    rejected_reward_indices = []
    rejected_no_landmark_indices = []
    rejected_zero_indices = []
    
    for cell in range(n_cells):
        profile = mean_profiles[cell]
        
        # Find GLOBAL peak first (across ALL bins)
        global_peak_idx = np.argmax(profile)
        global_peak_pos = bin_centers[global_peak_idx]
        global_peak_bins[cell] = global_peak_idx
        peak_positions[cell] = global_peak_pos  # Store for all cells
        
        # Check for zero activity
        if profile[global_peak_idx] == 0:
            rejected_zero_indices.append(cell)
            continue
        
        # Reject if global peak is in onset region
        if global_peak_pos < onset_threshold_cm:
            rejected_onset_indices.append(cell)
            continue
        
        # Reject if global peak is in reward/tunnel region
        if global_peak_pos > end_threshold_cm:
            rejected_reward_indices.append(cell)
            continue
        
        # Cell passed global peak filter - now check landmark windows
        landmark_peaks = []
        for lm_idx, (lm_min, lm_max) in enumerate(landmark_windows):
            lm_mask = (bin_centers >= lm_min) & (bin_centers <= lm_max)
            lm_indices = np.where(lm_mask)[0]
            
            if len(lm_indices) > 0:
                lm_response = np.max(profile[lm_indices])
                landmark_responses[cell, lm_idx] = lm_response
                
                if lm_min <= global_peak_pos <= lm_max:
                    landmark_peaks.append((lm_idx, lm_response))
            else:
                landmark_responses[cell, lm_idx] = 0
        
        # Does global peak fall within any landmark window?
        if len(landmark_peaks) == 0:
            rejected_no_landmark_indices.append(cell)
            continue
        
        # Choose highest response if multiple windows
        preferred_lm_idx, preferred_response = max(landmark_peaks, key=lambda x: x[1])
        
        # Calculate preference strength
        other_responses = [landmark_responses[cell, i] for i in range(n_landmarks) if i != preferred_lm_idx]
        if len(other_responses) > 0:
            pref_strength = preferred_response - np.mean(other_responses)
        else:
            pref_strength = preferred_response
        
        # Store results - this cell is VALID
        preferred_landmark[cell] = preferred_lm_idx
        preference_strength[cell] = pref_strength
        valid_cells[cell] = True
    
    # Summary statistics
    n_valid = np.sum(valid_cells)
    print(f"\n=== VALIDATION SUMMARY ===")
    print(f"Total cells: {n_cells}")
    print(f"Valid cells with landmark preference: {n_valid} ({n_valid/n_cells*100:.1f}%)")
    print(f"\nRejection breakdown:")
    print(f"  - Zero activity: {len(rejected_zero_indices)}")
    print(f"  - Onset response (first {exclude_first_bins} bins): {len(rejected_onset_indices)}")
    print(f"  - Reward/tunnel response (last {exclude_last_bins} bins): {len(rejected_reward_indices)}")
    print(f"  - Peak outside landmark windows: {len(rejected_no_landmark_indices)}")
    
    # Count cells per landmark
    print(f"\nLandmark preference distribution:")
    for lm_idx in range(n_landmarks):
        n_pref = np.sum(preferred_landmark[valid_cells] == lm_idx)
        pct = n_pref/n_valid*100 if n_valid > 0 else 0
        print(f"  Landmark {lm_idx+1} ({landmark_positions[lm_idx]} cm): {n_pref} ({pct:.1f}%)")
    
    results = {
        'preferred_landmark': preferred_landmark,
        'landmark_responses': landmark_responses,
        'preference_strength': preference_strength,
        'peak_positions': peak_positions,
        'global_peak_bins': global_peak_bins,
        'valid_cells': valid_cells,
        'mean_profiles': mean_profiles,
        'landmark_positions': np.array(landmark_positions),
        'landmark_windows': landmark_windows,
        # NEW: Rejected cell tracking
        'rejected_cells': {
            'onset': np.array(rejected_onset_indices),
            'reward': np.array(rejected_reward_indices),
            'no_landmark': np.array(rejected_no_landmark_indices),
            'zero_activity': np.array(rejected_zero_indices)
        },
        'parameters': {
            'landmark_windows_config': landmark_windows_config,
            'landmark_window': landmark_window,
            'exclude_first_bins': exclude_first_bins,
            'exclude_last_bins': exclude_last_bins,
            'onset_threshold_cm': onset_threshold_cm,
            'end_threshold_cm': end_threshold_cm,
            'boundary_exclusion': boundary_exclusion,
            'min_allowed': min_allowed,
            'max_allowed': max_allowed,
            'n_cells': n_cells,
            'n_valid': n_valid,
            'n_landmarks': n_landmarks
        }
    }
    
    return results

def plot_cells_by_landmark_assignment(landmark_results, bin_centers,
                                       landmark_positions=[25, 55, 85, 115],
                                       trim_start_bins=5, trim_end_bins=5,
                                       save_path=None):
    """
    Create response plots for cells assigned to each landmark.
    Shows one panel per landmark with all cells preferring that landmark.
    
    This helps verify that landmark assignment is working correctly -
    cells assigned to L1 should have peaks near L1, etc.
    
    Parameters:
    -----------
    landmark_results : dict
        Results from identify_landmark_responses()
    bin_centers : numpy.ndarray
        Spatial bin centers
    landmark_positions : list
        Positions of landmarks in cm
    trim_start_bins : int
        Number of bins to trim from display start
    trim_end_bins : int
        Number of bins to trim from display end
    save_path : str, optional
        Directory to save figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    mean_profiles = landmark_results['mean_profiles']
    preferred_landmark = landmark_results['preferred_landmark']
    valid_cells = landmark_results['valid_cells']
    landmark_windows = landmark_results['landmark_windows']
    n_landmarks = len(landmark_positions)

    # Define trim region
    n_bins = len(bin_centers)
    valid_start = trim_start_bins
    valid_end = n_bins - trim_end_bins if trim_end_bins > 0 else n_bins
    trimmed_bin_centers = bin_centers[valid_start:valid_end]

    # Create figure - one column per landmark
    fig, axes = plt.subplots(1, n_landmarks, figsize=(5*n_landmarks, 8))

    if n_landmarks == 1:
        axes = [axes]

    for lm_idx in range(n_landmarks):
        ax = axes[lm_idx]

        # Get cells assigned to this landmark
        lm_cell_mask = valid_cells & (preferred_landmark == lm_idx)
        lm_cell_indices = np.where(lm_cell_mask)[0]
        
        if len(lm_cell_indices) == 0:
            ax.text(0.5, 0.5, f'No cells assigned to\nLandmark {lm_idx+1}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f'Landmark {lm_idx+1} ({landmark_positions[lm_idx]} cm)\n(n=0)')
            continue
        
        # Get activity for these cells
        cell_activity = mean_profiles[lm_cell_indices]
        
        # Trim for display
        trimmed_activity = cell_activity[:, valid_start:valid_end]
        
        # Find peak in trimmed region for sorting
        peak_locations = np.argmax(trimmed_activity, axis=1)
        sorted_indices = np.argsort(peak_locations)
        sorted_activity = trimmed_activity[sorted_indices]
        
        # Normalize each cell (0-1)
        sorted_activity_norm = np.zeros_like(sorted_activity)
        for i in range(len(sorted_activity)):
            cell_min = np.min(sorted_activity[i])
            cell_max = np.max(sorted_activity[i])
            if cell_max > cell_min:
                sorted_activity_norm[i] = (sorted_activity[i] - cell_min) / (cell_max - cell_min)
            else:
                sorted_activity_norm[i] = sorted_activity[i]
        
        # Plot
        im = ax.imshow(sorted_activity_norm, aspect='auto', cmap=PARULA,
                      interpolation='nearest', vmin=0, vmax=1)
        
        # Add ALL landmark lines (red dashed)
        for other_lm_pos in landmark_positions:
            lm_bin = np.argmin(np.abs(trimmed_bin_centers - other_lm_pos))
            ax.axvline(lm_bin, color='red', linestyle='--', alpha=0.4, linewidth=1)
        
        # Highlight THIS landmark's window (green shaded region)
        lm_min, lm_max = landmark_windows[lm_idx]
        # Convert to trimmed bin indices
        lm_min_bin = np.argmin(np.abs(trimmed_bin_centers - lm_min))
        lm_max_bin = np.argmin(np.abs(trimmed_bin_centers - lm_max))
        ax.axvspan(lm_min_bin, lm_max_bin, alpha=0.15, color='green', 
                  label=f'Window [{lm_min:.0f}, {lm_max:.0f}]')
        
        # Highlight THIS landmark position (green solid line)
        this_lm_bin = np.argmin(np.abs(trimmed_bin_centers - landmark_positions[lm_idx]))
        ax.axvline(this_lm_bin, color='green', linestyle='-', alpha=0.8, linewidth=2)
        
        # X-axis labels (position in cm)
        n_trimmed_bins = len(trimmed_bin_centers)
        tick_positions = np.linspace(0, n_trimmed_bins-1, 5).astype(int)
        tick_labels = [f'{trimmed_bin_centers[i]:.0f}' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        ax.set_xlabel('Position (cm)', fontsize=10)
        ax.set_ylabel('Cell # (sorted by peak)', fontsize=10)
        ax.set_title(f'Landmark {lm_idx+1} ({landmark_positions[lm_idx]} cm)\n'
                    f'(n={len(lm_cell_indices)}, window [{lm_min:.0f}, {lm_max:.0f}])', 
                    fontsize=11, fontweight='bold')
    
    plt.suptitle('Cells by Landmark Assignment\n(Green = assigned landmark window, Red dashed = all landmarks)', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    if save_path is not None:
        fig_path = os.path.join(save_path, 'cells_by_landmark_assignment.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved landmark assignment plot: {os.path.basename(fig_path)}")
    
    return fig


def plot_landmark_assignment_summary(landmark_results, bin_centers,
                                     landmark_positions=[25, 55, 85, 115],
                                     save_path=None):
    """
    Create a summary plot showing the distribution of peak positions
    for cells assigned to each landmark.
    
    This helps identify if cells are being assigned to the "wrong" landmark
    (e.g., cells with peaks near L3 being assigned to L2).
    
    Parameters:
    -----------
    landmark_results : dict
        Results from identify_landmark_responses()
    bin_centers : numpy.ndarray
        Spatial bin centers
    landmark_positions : list
        Positions of landmarks in cm
    save_path : str, optional
        Directory to save figure
    """
    
    peak_positions = landmark_results['peak_positions']
    preferred_landmark = landmark_results['preferred_landmark']
    valid_cells = landmark_results['valid_cells']
    landmark_windows = landmark_results['landmark_windows']
    n_landmarks = len(landmark_positions)
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Colors for each landmark
    colors = plt.cm.Set1(np.linspace(0, 1, n_landmarks))
    
    # =========================================================================
    # Panel 1: Histogram of peak positions, colored by landmark assignment
    # =========================================================================
    ax1 = axes[0]
    
    bin_edges = np.linspace(0, np.max(bin_centers), 50)
    
    for lm_idx in range(n_landmarks):
        lm_cell_mask = valid_cells & (preferred_landmark == lm_idx)
        lm_peaks = peak_positions[lm_cell_mask]
        
        if len(lm_peaks) > 0:
            ax1.hist(lm_peaks, bins=bin_edges, alpha=0.5, color=colors[lm_idx],
                    label=f'L{lm_idx+1} ({landmark_positions[lm_idx]}cm): n={len(lm_peaks)}',
                    edgecolor='black', linewidth=0.5)
    
    # Add landmark position markers
    for lm_idx, lm_pos in enumerate(landmark_positions):
        ax1.axvline(lm_pos, color=colors[lm_idx], linestyle='--', linewidth=2, alpha=0.8)
    
    # Add window boundaries (light gray)
    for lm_min, lm_max in landmark_windows:
        ax1.axvline(lm_min, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax1.axvline(lm_max, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Peak Position (cm)', fontsize=20)
    ax1.set_ylabel('Number of Cells', fontsize=20)
    ax1.set_title('Distribution of Peak Positions by Landmark Assignment', fontsize=25, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.tick_params(labelsize=18)
    ax1.set_xlim(0, np.max(bin_centers))

    # =========================================================================
    # Panel 2: Box plot of peak positions for each landmark
    # =========================================================================
    ax2 = axes[1]

    peak_data = []
    labels = []

    for lm_idx in range(n_landmarks):
        lm_cell_mask = valid_cells & (preferred_landmark == lm_idx)
        lm_peaks = peak_positions[lm_cell_mask]

        if len(lm_peaks) > 0:
            peak_data.append(lm_peaks)
            labels.append(f'L{lm_idx+1}\n({landmark_positions[lm_idx]}cm)')

    if len(peak_data) > 0:
        bp = ax2.boxplot(peak_data, labels=labels, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors[:len(peak_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        # Add landmark position reference lines
        for lm_idx, lm_pos in enumerate(landmark_positions):
            ax2.axhline(lm_pos, color=colors[lm_idx], linestyle='--', linewidth=1.5, alpha=0.6)

        # Add window boundaries
        for lm_idx, (lm_min, lm_max) in enumerate(landmark_windows):
            ax2.axhspan(lm_min, lm_max, alpha=0.1, color=colors[lm_idx])

    ax2.set_ylabel('Peak Position (cm)', fontsize=20)
    ax2.set_xlabel('Assigned Landmark', fontsize=20)
    ax2.tick_params(labelsize=18)
    ax2.set_title('Peak Position Distribution by Landmark (boxes show window boundaries)',
                  fontsize=25, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path is not None:
        fig_path = os.path.join(save_path, 'landmark_assignment_summary.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved assignment summary: {os.path.basename(fig_path)}")
    
    return fig

# ============================================================================
# NEW: VISUALIZATION OF REJECTED CELLS
# ============================================================================

def plot_rejected_cells_response(landmark_results, bin_centers, 
                                 landmark_positions=[25, 55, 85, 115],
                                 trim_start_bins=0, trim_end_bins=0,
                                 save_path=None):
    """
    Create response plots for cells that were rejected from landmark analysis.
    Shows separate panels for onset-responding, reward-responding, and no-landmark cells.
    
    Parameters:
    -----------
    landmark_results : dict
        Results from identify_landmark_responses()
    bin_centers : numpy.ndarray
        Spatial bin centers
    landmark_positions : list
        Positions of landmarks in cm
    trim_start_bins : int
        Number of bins to trim from display start
    trim_end_bins : int
        Number of bins to trim from display end
    save_path : str, optional
        Directory to save figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
    """
    mean_profiles = landmark_results['mean_profiles']
    rejected = landmark_results['rejected_cells']
    params = landmark_results['parameters']
    
    # Define trim region
    n_bins = len(bin_centers)
    valid_start = trim_start_bins
    valid_end = n_bins - trim_end_bins if trim_end_bins > 0 else n_bins
    trimmed_bin_centers = bin_centers[valid_start:valid_end]
    
    # Categories to plot
    categories = [
        ('onset', 'Onset-Responding Cells', rejected['onset']),
        ('reward', 'Reward/Tunnel-Responding Cells', rejected['reward']),
        ('no_landmark', 'No Landmark Preference (Between Landmarks)', rejected['no_landmark'])
    ]
    
    # Filter out empty categories
    categories = [(name, title, indices) for name, title, indices in categories if len(indices) > 0]
    
    if len(categories) == 0:
        print("No rejected cells to visualize")
        return None
    
    # Create figure
    n_categories = len(categories)
    fig, axes = plt.subplots(1, n_categories, figsize=(6*n_categories, 8))
    
    if n_categories == 1:
        axes = [axes]
    
    for ax_idx, (cat_name, cat_title, cell_indices) in enumerate(categories):
        ax = axes[ax_idx]
        
        if len(cell_indices) == 0:
            ax.text(0.5, 0.5, 'No cells', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(cat_title)
            continue
        
        # Get activity for these cells
        cell_activity = mean_profiles[cell_indices]
        
        # Trim for display
        trimmed_activity = cell_activity[:, valid_start:valid_end]
        
        # Find peak in trimmed region for sorting
        peak_locations = np.argmax(trimmed_activity, axis=1)
        sorted_indices = np.argsort(peak_locations)
        sorted_activity = trimmed_activity[sorted_indices]
        
        # Normalize each cell (0-1)
        sorted_activity_norm = np.zeros_like(sorted_activity)
        for i in range(len(sorted_activity)):
            cell_min = np.min(sorted_activity[i])
            cell_max = np.max(sorted_activity[i])
            if cell_max > cell_min:
                sorted_activity_norm[i] = (sorted_activity[i] - cell_min) / (cell_max - cell_min)
            else:
                sorted_activity_norm[i] = sorted_activity[i]
        
        # Plot
        im = ax.imshow(sorted_activity_norm, aspect='auto', cmap=PARULA,
                      interpolation='nearest', vmin=0, vmax=1)
        
        # Add landmark lines
        for lm_pos in landmark_positions:
            lm_bin = np.argmin(np.abs(trimmed_bin_centers - lm_pos))
            ax.axvline(lm_bin, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add exclusion zone markers
        if cat_name == 'onset':
            # Show onset threshold
            onset_cm = params['onset_threshold_cm']
            if onset_cm > trimmed_bin_centers[0]:
                onset_bin = np.argmin(np.abs(trimmed_bin_centers - onset_cm))
                ax.axvline(onset_bin, color='orange', linestyle='-', alpha=0.8, linewidth=2, 
                          label=f'Onset threshold ({onset_cm:.0f}cm)')
        elif cat_name == 'reward':
            # Show end threshold
            end_cm = params['end_threshold_cm']
            if end_cm < trimmed_bin_centers[-1]:
                end_bin = np.argmin(np.abs(trimmed_bin_centers - end_cm))
                ax.axvline(end_bin, color='orange', linestyle='-', alpha=0.8, linewidth=2,
                          label=f'End threshold ({end_cm:.0f}cm)')
        
        # X-axis labels (position in cm)
        n_trimmed_bins = len(trimmed_bin_centers)
        tick_positions = np.linspace(0, n_trimmed_bins-1, 5).astype(int)
        tick_labels = [f'{trimmed_bin_centers[i]:.0f}' for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        
        ax.set_xlabel('Position (cm)', fontsize=10)
        ax.set_ylabel('Cell # (sorted by peak)', fontsize=10)
        ax.set_title(f'{cat_title}\n(n={len(cell_indices)})', fontsize=11, fontweight='bold')
        
        if cat_name in ['onset', 'reward']:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Rejected Cells - Response Profiles', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path is not None:
        fig_path = os.path.join(save_path, 'rejected_cells_response_plots.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved rejected cells plot: {os.path.basename(fig_path)}")
    
    return fig

# ============================================================================
# PHASE 2: LAYER-SPECIFIC LANDMARK PREFERENCE ANALYSIS
# ============================================================================

def analyze_layer_landmark_preferences(landmark_results, layer_cells, reliable_valid_cells):
    """
    Analyze landmark preferences for each cortical layer.
    
    Parameters:
    -----------
    landmark_results : dict
        Results from identify_landmark_responses()
    layer_cells : dict
        Dictionary with indices of cells in each layer
    reliable_valid_cells : numpy.ndarray
        Boolean array indicating reliable and valid cells (from SMI analysis)
    
    Returns:
    --------
    layer_results : dict
        Dictionary with landmark preference results by layer
    """
    
    preferred_landmark = landmark_results['preferred_landmark']
    valid_cells = landmark_results['valid_cells']
    n_landmarks = landmark_results['parameters']['n_landmarks']
    landmark_positions = landmark_results['landmark_positions']
    
    # Combine with reliable cells from SMI analysis
    final_valid_cells = valid_cells & reliable_valid_cells
    
    print(f"\n=== LAYER-SPECIFIC LANDMARK PREFERENCES ===")
    print(f"Using {np.sum(final_valid_cells)} cells that are:")
    print(f"  - Valid for landmark analysis: {np.sum(valid_cells)}")
    print(f"  - Reliable from SMI analysis: {np.sum(reliable_valid_cells)}")
    print(f"  - Both: {np.sum(final_valid_cells)}")
    
    layer_results = {}
    
    for layer_name, layer_cell_indices in layer_cells.items():
        # Find cells in this layer that are valid
        layer_valid_cells = np.intersect1d(
            np.where(final_valid_cells)[0], 
            layer_cell_indices
        )
        
        if len(layer_valid_cells) == 0:
            print(f"\n{layer_name}: No valid cells")
            layer_results[layer_name] = None
            continue
        
        # Count preferences for each landmark
        layer_preferences = preferred_landmark[layer_valid_cells]
        landmark_counts = np.zeros(n_landmarks, dtype=int)
        
        for lm_idx in range(n_landmarks):
            landmark_counts[lm_idx] = np.sum(layer_preferences == lm_idx)
        
        # Calculate proportions
        landmark_proportions = landmark_counts / len(layer_valid_cells)
        
        # Chi-square test for non-uniform distribution
        expected_counts = np.full(n_landmarks, len(layer_valid_cells) / n_landmarks)
        if len(layer_valid_cells) >= n_landmarks * 5:  # Rule of thumb for chi-square
            chi2_stat, chi2_p = stats.chisquare(landmark_counts, expected_counts)
        else:
            chi2_stat, chi2_p = np.nan, np.nan
        
        # Store results
        layer_results[layer_name] = {
            'valid_cells': layer_valid_cells,
            'preferred_landmarks': layer_preferences,
            'landmark_counts': landmark_counts,
            'landmark_proportions': landmark_proportions,
            'stats': {
                'n_cells': len(layer_valid_cells),
                'chi2_stat': chi2_stat,
                'chi2_p': chi2_p
            }
        }
        
        # Print summary
        print(f"\n{layer_name}: {len(layer_valid_cells)} valid cells")
        for lm_idx in range(n_landmarks):
            print(f"  Landmark {lm_idx+1} ({landmark_positions[lm_idx]:.0f} cm): "
                  f"{landmark_counts[lm_idx]} cells ({landmark_proportions[lm_idx]*100:.1f}%)")
        
        if not np.isnan(chi2_p):
            print(f"  Chi-square test: χ² = {chi2_stat:.3f}, p = {chi2_p:.4f}", end="")
            if chi2_p < 0.05:
                print(" (Non-uniform preference *)")
            else:
                print(" (Uniform distribution)")
    
    # Between-layer comparison
    print(f"\n=== BETWEEN-LAYER COMPARISONS ===")
    valid_layers = [layer for layer, results in layer_results.items() if results is not None]
    
    if len(valid_layers) >= 2:
        for i, layer1 in enumerate(valid_layers):
            for layer2 in valid_layers[i+1:]:
                # Chi-square test for independence
                counts1 = layer_results[layer1]['landmark_counts']
                counts2 = layer_results[layer2]['landmark_counts']
                
                contingency_table = np.array([counts1, counts2])
                
                # Only test if sufficient sample size
                if np.all(contingency_table.sum(axis=0) >= 5):
                    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
                    print(f"{layer1} vs {layer2}: χ² = {chi2:.3f}, p = {p:.4f}", end="")
                    if p < 0.05:
                        print(" (Significantly different *)")
                    else:
                        print(" (Not different)")
    
    return layer_results


# ============================================================================
# PHASE 3: WITHIN-SESSION TEMPORAL DYNAMICS
# ============================================================================

def analyze_within_session_dynamics(normalized_spatial_activity, bin_centers,
                                    landmark_positions, layer_cells, reliable_valid_cells,
                                    landmark_window=7.0, boundary_exclusion=(10, 10),
                                    trials_per_block=30, smoothing_sigma=1.0):
    """
    Analyze how landmark preferences change across trial blocks within a session.
    
    Parameters:
    -----------
    normalized_spatial_activity : numpy.ndarray
        Activity matrix (cells x trials x spatial_bins)
    bin_centers : numpy.ndarray
        Centers of spatial bins (cm)
    landmark_positions : list or array
        Positions of landmarks (cm)
    layer_cells : dict
        Dictionary with indices of cells in each layer
    reliable_valid_cells : numpy.ndarray
        Boolean array indicating reliable and valid cells
    trials_per_block : int
        Number of trials per block (default: 30)
    
    Returns:
    --------
    dynamics_results : dict
        Dictionary with temporal dynamics results
    """
    
    n_cells, n_trials, n_bins = normalized_spatial_activity.shape
    n_landmarks = len(landmark_positions)
    
    # Calculate number of complete blocks
    n_blocks = n_trials // trials_per_block
    n_trials_used = n_blocks * trials_per_block
    
    print(f"\n=== WITHIN-SESSION TEMPORAL DYNAMICS ===")
    print(f"Total trials: {n_trials}")
    print(f"Trials per block: {trials_per_block}")
    print(f"Number of blocks: {n_blocks}")
    print(f"Trials used: {n_trials_used} (dropping {n_trials - n_trials_used} trials)")
    
    if n_blocks < 2:
        print("WARNING: Less than 2 blocks available. Need at least 60 trials for temporal analysis.")
        return None
    
    # Initialize storage for each block
    block_results = []
    
    for block_idx in range(n_blocks):
        start_trial = block_idx * trials_per_block
        end_trial = start_trial + trials_per_block
        
        print(f"\nAnalyzing Block {block_idx+1}/{n_blocks} (trials {start_trial}-{end_trial-1})...")
        
        # Extract this block's data
        block_activity = normalized_spatial_activity[:, start_trial:end_trial, :]
        
        # Identify landmark preferences for this block
        block_landmark_results = identify_landmark_responses(
            block_activity, bin_centers, landmark_positions,
            landmark_window=landmark_window,
            boundary_exclusion=boundary_exclusion,
            smoothing_sigma=smoothing_sigma
        )
        
        # Analyze by layer
        block_layer_results = analyze_layer_landmark_preferences(
            block_landmark_results, layer_cells, reliable_valid_cells
        )
        
        block_results.append({
            'block_idx': block_idx,
            'trial_range': (start_trial, end_trial),
            'landmark_results': block_landmark_results,
            'layer_results': block_layer_results
        })
    
    # Organize results by layer for easy visualization
    layer_names = [layer for layer in layer_cells.keys()]
    preference_by_block = {}
    
    for layer_name in layer_names:
        # Initialize array: (n_blocks, n_landmarks)
        layer_block_proportions = np.zeros((n_blocks, n_landmarks))
        
        for block_idx in range(n_blocks):
            layer_res = block_results[block_idx]['layer_results'].get(layer_name)
            if layer_res is not None:
                layer_block_proportions[block_idx, :] = layer_res['landmark_proportions']
            else:
                layer_block_proportions[block_idx, :] = np.nan
        
        preference_by_block[layer_name] = layer_block_proportions
    
    dynamics_results = {
        'n_blocks': n_blocks,
        'trials_per_block': trials_per_block,
        'block_results': block_results,
        'preference_by_block': preference_by_block,
        'landmark_positions': np.array(landmark_positions)
    }
    
    return dynamics_results


# ============================================================================
# PHASE 4: DATA EXPORT
# ============================================================================
def save_session_landmark_data(full_session_results, dynamics_results, 
                               save_path, session_id, date_str=None):
    """
    Save landmark preference analysis results to HDF5 file.
    CRITICAL FIX: Properly sanitize layer names to avoid HDF5 path issues
    """
    
    print(f"\n=== SAVING SESSION DATA ===")
    print(f"Session ID: {session_id}")
    print(f"Save path: {save_path}")
    
    # CRITICAL: Sanitize function MUST be defined BEFORE use
    def sanitize_name(name):
        """Replace characters that HDF5 interprets as path separators"""
        # Replace / and \ with underscore
        sanitized = str(name).replace('/', '_').replace('\\', '_')
        return sanitized
    
    with h5py.File(save_path, 'w') as f:
        # Metadata
        f.attrs['session_id'] = session_id
        if date_str:
            f.attrs['date'] = date_str
        
        print(f"  Creating 'full_session' group...")
        full_grp = f.create_group('full_session')
        
        # Save each layer
        for layer_name, layer_res in full_session_results.items():
            if layer_res is None:
                print(f"  Skipping {layer_name} (no data)")
                continue
            
            # CRITICAL: Sanitize the name FIRST
            safe_layer_name = sanitize_name(layer_name)
            print(f"  Saving layer: '{layer_name}' → '{safe_layer_name}'")
            
            # CRITICAL: Use the SANITIZED name when creating group
            layer_grp = full_grp.create_group(safe_layer_name)
            
            # Store original name so we can restore it when loading
            layer_grp.attrs['original_name'] = str(layer_name)
            layer_grp.attrs['n_cells'] = int(layer_res['stats']['n_cells'])
            
            # Save chi-square stats if valid
            chi2_stat = layer_res['stats']['chi2_stat']
            chi2_p = layer_res['stats']['chi2_p']
            if not (np.isnan(chi2_stat) or np.isinf(chi2_stat)):
                layer_grp.attrs['chi2_stat'] = float(chi2_stat)
                layer_grp.attrs['chi2_p'] = float(chi2_p)
            
            # Save datasets
            print(f"    Creating datasets...")
            
            # landmark_counts
            counts = np.array(layer_res['landmark_counts'], dtype=np.int32)
            layer_grp.create_dataset('landmark_counts', data=counts)
            print(f"      ✓ landmark_counts: {counts}")
            
            # landmark_proportions
            props = np.array(layer_res['landmark_proportions'], dtype=np.float64)
            layer_grp.create_dataset('landmark_proportions', data=props)
            print(f"      ✓ landmark_proportions: {props}")
            
            # preferred_landmarks
            prefs = np.array(layer_res['preferred_landmarks'], dtype=np.int32)
            layer_grp.create_dataset('preferred_landmarks', data=prefs)
            print(f"      ✓ preferred_landmarks: shape {prefs.shape}")
            
            # valid_cells
            valid = np.array(layer_res['valid_cells'], dtype=np.int64)
            layer_grp.create_dataset('valid_cells', data=valid)
            print(f"      ✓ valid_cells: shape {valid.shape}")
        
        # Dynamics results
        if dynamics_results is not None:
            print(f"  Creating 'dynamics' group...")
            dyn_grp = f.create_group('dynamics')
            dyn_grp.attrs['n_blocks'] = int(dynamics_results['n_blocks'])
            dyn_grp.attrs['trials_per_block'] = int(dynamics_results['trials_per_block'])
            
            landmark_pos = np.array(dynamics_results['landmark_positions'], dtype=np.float64)
            dyn_grp.create_dataset('landmark_positions', data=landmark_pos)
            print(f"    ✓ landmark_positions: {landmark_pos}")
            
            for layer_name, block_props in dynamics_results['preference_by_block'].items():
                # CRITICAL: Sanitize layer name here too
                safe_layer_name = sanitize_name(layer_name)
                dset_name = f'{safe_layer_name}_preference_by_block'
                
                block_props_array = np.array(block_props, dtype=np.float64)
                dyn_grp.create_dataset(dset_name, data=block_props_array)
                print(f"    ✓ {dset_name}: shape {block_props_array.shape}")
    
    print(f"\n✓ Data saved successfully: {os.path.basename(save_path)}")


# ============================================================================
# PHASE 5: VISUALIZATION
# ============================================================================
def plot_layer_landmark_heatmap(layer_results, landmark_positions, title="Landmark Preferences by Layer", save_path=None):
    """
    Create heatmap showing proportion of cells in each layer preferring each landmark.
    """
    
    # Extract valid layers
    valid_layers = [layer for layer, res in layer_results.items() if res is not None]
    n_landmarks = len(landmark_positions)
    
    if len(valid_layers) == 0:
        print("No valid layers for heatmap")
        return None
    
    # Build matrix: rows = layers, cols = landmarks
    heatmap_data = np.zeros((len(valid_layers), n_landmarks))
    count_data = np.zeros((len(valid_layers), n_landmarks), dtype=int)
    
    for i, layer_name in enumerate(valid_layers):
        heatmap_data[i, :] = layer_results[layer_name]['landmark_proportions']
        count_data[i, :] = layer_results[layer_name]['landmark_counts']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot heatmap
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_landmarks))
    ax.set_yticks(np.arange(len(valid_layers)))
    ax.set_xticklabels([f"L{i+1}\n({landmark_positions[i]:.0f}cm)" for i in range(n_landmarks)])
    ax.set_yticklabels(valid_layers)
    
    # Add text annotations with proportions AND counts
    for i in range(len(valid_layers)):
        for j in range(n_landmarks):
            text_str = f'{heatmap_data[i, j]:.2f}\n(n={count_data[i, j]})'
            text = ax.text(j, i, text_str,
                          ha="center", va="center", color="black", fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Landmark', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Proportion of Cells', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        fig_path = os.path.join(save_path, 'landmark_preference_heatmap.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved heatmap: {os.path.basename(fig_path)}")
    
    return fig


def plot_within_session_dynamics(dynamics_results, title="Within-Session Landmark Preference Dynamics", save_path=None):
    """
    Plot how landmark preferences change across trial blocks for each layer.
    """
    
    if dynamics_results is None:
        print("No dynamics results to plot")
        return None
    
    preference_by_block = dynamics_results['preference_by_block']
    landmark_positions = dynamics_results['landmark_positions']
    n_blocks = dynamics_results['n_blocks']
    n_landmarks = len(landmark_positions)
    
    # Get valid layers
    valid_layers = [layer for layer, data in preference_by_block.items() 
                   if not np.all(np.isnan(data))]
    
    if len(valid_layers) == 0:
        print("No valid layers for dynamics plot")
        return None
    
    # Create subplots: one per layer
    n_layers = len(valid_layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers), sharex=True)
    
    if n_layers == 1:
        axes = [axes]
    
    # Color scheme for landmarks
    colors = plt.cm.Set1(np.linspace(0, 1, n_landmarks))
    
    for ax_idx, layer_name in enumerate(valid_layers):
        ax = axes[ax_idx]
        layer_data = preference_by_block[layer_name]
        
        # Plot lines for each landmark
        for lm_idx in range(n_landmarks):
            ax.plot(range(1, n_blocks+1), layer_data[:, lm_idx], 
                   marker='o', linewidth=2, markersize=8,
                   color=colors[lm_idx], 
                   label=f"Landmark {lm_idx+1} ({landmark_positions[lm_idx]:.0f}cm)")
        
        ax.set_ylabel('Proportion of Cells', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Trial Block', fontsize=12)
    axes[-1].set_xticks(range(1, n_blocks+1))
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        fig_path = os.path.join(save_path, 'landmark_preference_dynamics.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved dynamics: {os.path.basename(fig_path)}")

    return fig


def plot_example_cells_by_landmark(normalized_spatial_activity, bin_centers,
                                  landmark_results, layer_results, 
                                  landmark_positions, n_examples=3, save_path=None):
    """
    Plot example cells preferring each landmark, organized by layer.
    """
    
    preferred_landmark = landmark_results['preferred_landmark']
    valid_cells = landmark_results['valid_cells']
    mean_profiles = landmark_results['mean_profiles']
    landmark_windows = landmark_results['landmark_windows']
    n_landmarks = len(landmark_positions)
    
    # Get valid layers
    valid_layers = [layer for layer, res in layer_results.items() if res is not None]
    
    if len(valid_layers) == 0:
        print("No valid layers for example plots")
        return None
    
    # Create figure
    fig, axes = plt.subplots(len(valid_layers), n_landmarks, 
                            figsize=(5*n_landmarks, 4*len(valid_layers)))
    
    if len(valid_layers) == 1:
        axes = axes.reshape(1, -1)
    
    for row_idx, layer_name in enumerate(valid_layers):
        layer_valid_cells = layer_results[layer_name]['valid_cells']
        layer_preferences = preferred_landmark[layer_valid_cells]
        
        for lm_idx in range(n_landmarks):
            ax = axes[row_idx, lm_idx]
            
            # Find cells in this layer preferring this landmark
            cells_pref_lm = layer_valid_cells[layer_preferences == lm_idx]
            
            if len(cells_pref_lm) == 0:
                ax.text(0.5, 0.5, 'No cells', ha='center', va='center',
                        transform=ax.transAxes, fontsize=20, color='gray')
                ax.set_title(f'{layer_name} - L{lm_idx+1}', fontsize=25, fontweight='bold')
                continue
            
            # Select top n_examples by response strength
            n_plot = min(n_examples, len(cells_pref_lm))
            example_cells = cells_pref_lm[:n_plot]
            
            # Plot each example
            for cell_idx in example_cells:
                profile = mean_profiles[cell_idx]
                ax.plot(bin_centers, profile, alpha=0.7, linewidth=1.5)
            
            # Highlight landmark window
            lm_min, lm_max = landmark_windows[lm_idx]
            ax.axvspan(lm_min, lm_max, alpha=0.2, color='red', label='Landmark window')
            ax.axvline(landmark_positions[lm_idx], color='red', linestyle='--', linewidth=2)
            
            ax.set_title(f'{layer_name} - Landmark {lm_idx+1}',
                         fontsize=20, fontweight='bold')
            ax.set_xlabel('Position (cm)', fontsize=20)
            ax.set_ylabel('Normalized Activity', fontsize=20)
            ax.tick_params(labelsize=18)
            ax.grid(True, alpha=0.3)

            if row_idx == 0 and lm_idx == 0:
                ax.legend(fontsize=16)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path is not None:
        fig_path = os.path.join(save_path, 'landmark_preference_examples.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved examples: {os.path.basename(fig_path)}")
    
    return fig


# ============================================================================
# MAIN WORKFLOW FUNCTION
# ============================================================================
# Update the run_landmark_analysis function signature and Phase 1 call:

def run_landmark_analysis(normalized_spatial_activity, bin_centers, layer_cells,
                         reliable_valid_cells, landmark_positions=[30, 60, 90, 120],
                         landmark_windows_config=None,  # NEW: per-landmark config
                         landmark_window=10.0,  # fallback
                         boundary_exclusion=(10, 10),
                         exclude_first_bins=5, exclude_last_bins=5,
                         trials_per_block=30, smoothing_sigma=1.0,
                         save_path=None, session_id=None, date_str=None):
    """
    Complete workflow for landmark preference analysis.
    
    NEW PARAMETER:
    --------------
    landmark_windows_config : list of dict, optional
        Per-landmark window configuration. Example:
        [
            {'before': 10, 'after': 10},  # L1
            {'before': 15, 'after': 10},  # L2
            {'before': 15, 'after': 10},  # L3
            {'before': 15, 'after': 5},   # L4
        ]
    """

    print("\n" + "="*70)
    print("LANDMARK PREFERENCE ANALYSIS - COMPLETE WORKFLOW")
    print("="*70)
    
    # Phase 1: Identify landmark responses
    print("\n" + "-"*70)
    print("PHASE 1: IDENTIFYING LANDMARK RESPONSES")
    print("-"*70)
    
    landmark_results = identify_landmark_responses(
        normalized_spatial_activity, bin_centers, landmark_positions,
        landmark_windows_config=landmark_windows_config,  # NEW
        landmark_window=landmark_window,
        boundary_exclusion=boundary_exclusion,
        smoothing_sigma=smoothing_sigma,
        exclude_first_bins=exclude_first_bins,
        exclude_last_bins=exclude_last_bins
    )
    
    # Phase 2: Layer-specific analysis (full session)
    print("\n" + "-"*70)
    print("PHASE 2: LAYER-SPECIFIC ANALYSIS (FULL SESSION)")
    print("-"*70)
    
    layer_results = analyze_layer_landmark_preferences(
        landmark_results, layer_cells, reliable_valid_cells
    )
    
    # Phase 3: Within-session dynamics
    print("\n" + "-"*70)
    print("PHASE 3: WITHIN-SESSION TEMPORAL DYNAMICS")
    print("-"*70)
    
    dynamics_results = analyze_within_session_dynamics(
        normalized_spatial_activity, bin_centers, landmark_positions,
        layer_cells, reliable_valid_cells,
        landmark_window=landmark_window,
        boundary_exclusion=boundary_exclusion,
        trials_per_block=trials_per_block,
        smoothing_sigma=smoothing_sigma
    )
    
    # Phase 4: Visualization
    print("\n" + "-"*70)
    print("PHASE 4: CREATING VISUALIZATIONS")
    print("-"*70)
    
    if save_path is not None:
        # H5 file stays at the session root (batch skip_existing checks here)
        h5_save_path = os.path.join(save_path, f"{session_id}_landmark_preferences.h5")
        # Figures go into a dedicated subfolder
        save_dir = os.path.join(save_path, 'LandmarkPreference')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None
        h5_save_path = None
    
    # Create visualizations
    fig_heatmap = plot_layer_landmark_heatmap(
        layer_results, landmark_positions,
        title=f"Landmark Preferences by Layer - {session_id if session_id else 'Session'}",
        save_path=save_dir
    )
    
    if dynamics_results is not None:
        fig_dynamics = plot_within_session_dynamics(
            dynamics_results,
            title=f"Within-Session Dynamics - {session_id if session_id else 'Session'}",
            save_path=save_dir
        )
    else:
        fig_dynamics = None
    
    fig_examples = plot_example_cells_by_landmark(
        normalized_spatial_activity, bin_centers,
        landmark_results, layer_results,
        landmark_positions, n_examples=2,
        save_path=save_dir
    )
    
    # NEW: Visualize rejected cells
    print("\n  Creating rejected cells visualization...")
    fig_rejected = plot_rejected_cells_response(
        landmark_results, bin_centers,
        landmark_positions=landmark_positions,
        trim_start_bins=exclude_first_bins,  # Match exclusion params
        trim_end_bins=exclude_last_bins,
        save_path=save_dir
    )
    # In Phase 4, after the existing visualizations:

    # NEW: Visualize cells by landmark assignment
    print("\n  Creating landmark assignment visualization...")
    fig_by_landmark = plot_cells_by_landmark_assignment(
        landmark_results, bin_centers,
        landmark_positions=landmark_positions,
        trim_start_bins=exclude_first_bins,
        trim_end_bins=exclude_last_bins,
        save_path=save_dir
    )

    # NEW: Summary statistics of landmark assignment
    print("\n  Creating landmark assignment summary...")
    fig_assignment_summary = plot_landmark_assignment_summary(
        landmark_results, bin_centers,
        landmark_positions=landmark_positions,
        save_path=save_dir
    )

    # plt.show()
    
    # Phase 5: Save HDF5 data
    if h5_save_path is not None and session_id is not None:
        print("\n" + "-"*70)
        print("PHASE 5: SAVING HDF5 DATA")
        print("-"*70)
        
        save_session_landmark_data(
            layer_results, dynamics_results,
            h5_save_path, session_id, date_str
        )
    
    # Compile all results
    results = {
        'landmark_results': landmark_results,
        'layer_results': layer_results,
        'dynamics_results': dynamics_results,
        'figures': {
            'heatmap': fig_heatmap,
            'dynamics': fig_dynamics,
            'examples': fig_examples,
            'rejected': fig_rejected,
            'by_landmark': fig_by_landmark,        # NEW
            'assignment_summary': fig_assignment_summary  # NEW
        }
    }
    
    print("\n" + "="*70)
    print("LANDMARK PREFERENCE ANALYSIS COMPLETE!")
    print("="*70)
    
    return results


# ============================================================================
# EXAMPLE USAGE WITH YOUR EXISTING PIPELINE
# ============================================================================

if __name__ == "__main__":
    """
    Example of how to use this script with your existing data.
    
    Replace these paths and variables with your actual data:
    - data_filepath: path to your preprocessed data
    - normalized_spatial_activity: from your preproc.h5 file
    - bin_centers: from your preproc.h5 file
    - layer_cells: from your layer identification
    - reliable_valid_cells: from your SMI analysis
    """
    
    # EXAMPLE: Load your data
    
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251030_JSY_JSY054_SpMod_Day1\TSeries-10302025-1512-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251031_JSY_JSY054_SpMod_Day2\TSeries-10312025-1751-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251101_JSY_JSY054_SpMod_Day3\TSeries-11012025-1725-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251102_JSY_JSY054_SpMod_Day4\TSeries-11022025-1642-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251103_JSY_JSY054_SpMod_Day5\TSeries-11032025-1715-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251104_JSY_JSY054_SpMod_Day6\TSeries-11042025-1418-001"
    data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY054_ChronicImaging\251105_JSY_JSY054_SpMod_Day7\TSeries-11052025-1512-001"
    
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251009_JSY_JSY052_SpatialModulation_Day1\TSeries-10092025-1542-002"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251010_JSY_JSY052_SpatialModulation_Day2\TSeries-10102025-0916-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251011_JSY_JSY052_SpatialModulation_Day3\TSeries-10112025-1441-002"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251012_JSY_JSY052_SpatialModulation_Day4\TSeries-10122025-1212-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251013_JSY_JSY052_SpatialModulation_Day5\TSeries-10132025-1236-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging\251014_JSY_JSY052_SpatialModulation_Day6\TSeries-10142025-1647-003"
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChronicImaging\251015_JSY_JSY052_SpatialModulation_Day7\TSeries-10152025-1103-001'

    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_togetherregistration\TSeries-09062025-1308-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250906_JSY_JSY044_SpatialModulation_Day1_togetherregistration\TSeries-09062025-1308-002"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_togetherregistration\TSeries-09072025-1257-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250907_JSY_JSY044_SpaitalModulation_Day2_togetherregistration\TSeries-09072025-1257-002"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_togetherregistration\TSeries-09082025-1540-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250908_JSY_JSY044_SpatialModulation_Day3_togetherregistration\TSeries-09082025-1540-002"
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250909_JSY_JSY044_SpatialModulation_Day4\TSeries-09092025-1256-002'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250910_JSY_JSY044_SpatialModulation_Day5\TSeries-09102025-1340-002'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250911_JSY_JSY044_SpatialModulation_Day6\TSeries-09112025-1414-002'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging\250912_JSY_JSY044_SpatialModulation_Day7\TSeries-09122025-1334-002'


    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251101_JSY_JSY051_SpMod_Day1\TSeries-11012025-1725-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251102_JSY_JSY051_SpMod_Day2\TSeries-11022025-1642-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251103_JSY_JSY051_SpMod_Day3\TSeries-11032025-1715-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251104_JSY_JSY051_SpMod_Day4\TSeries-11042025-1418-001"
    # data_filepath = r"D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251105_JSY_JSY051_SpMod_Day5\TSeries-11052025-1512-002"
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY051_ChronicImaging\251107_JSY_JSY051_SpMO_OpenloopVR_stationary\TSeries-11072025-1032-001'

    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250616_JSY_JSY041_SpatialModulation_Day1_V1Prism\TSeries-06162025-1521-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250618_JSY_JSY041_SpatialModulation_Day3_V1Prism\TSeries-06182025-1641-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250620_JSY_JSY041_SpatialModulation_Day5_V1Prism\TSeries-06202025-1515-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY041_ChronicImaging\250622_JSY_JSY041_SpatialModulation_Day7_V1Prism\TSeries-06222025-1550-001'
    
    
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250620_JSY_JSY040_SpatialModulation_Day1_V1Prism\TSeries-06202025-1515-001'
    # data_filepath = r'D:\V1_SpatialModulation\2p\V1_prism\JSY040_ChronicImaging\250622_JSY_JSY040_SpatialModulation_Day3_V1Prism\TSeries-06222025-1550-001'
    
    preproc_files = glob.glob(os.path.join(data_filepath, "*preproc.h5"))
    if not preproc_files:
        raise ValueError(f"No preprocessed .h5 file found in {data_filepath}")
    #
    preproc_file = preproc_files[0]
    print(f"Loading: {os.path.basename(preproc_file)}")
    preproc_data = files.read_h5(preproc_file)
    print("Successfully loaded!")
    
    normalized_spatial_activity = preproc_data['norm_spatial_activity']
    bin_centers = preproc_data['bin_centers']
    reliable_valid_cells = preproc_data['combined_reliable']
    
    twoP_filename = os.path.basename(data_filepath)
    raw_twop_data = TwoP(data_filepath, twoP_filename)
    raw_twop_data.find_files()
    twop_dict = raw_twop_data.calc_dFF()
    
    numCells = len(twop_dict['stat'])
    im = np.zeros((twop_dict['ops']['Ly'], twop_dict['ops']['Lx']))
    for n in range(numCells):
        ypix = twop_dict['stat'][n]['ypix'][~twop_dict['stat'][n]['overlap']]
        xpix = twop_dict['stat'][n]['xpix'][~twop_dict['stat'][n]['overlap']]
        im[ypix, xpix] = xpix
    
    med_coords = np.array([cell['med'] for cell in twop_dict['stat']])
    
    # Get layer information (from your SMI calculation script)
    layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)
    
    # find session_id in data_filepath (it is day2 in 'D:\\V1_SpatialModulation\\2p\\V1_prism\\JSY054_ChronicImaging\\251030_JSY_JSY054_SpMod_Day1\\TSeries-10302025-1512-001')
    session_folder = os.path.basename(os.path.dirname(data_filepath))
    
    # Extract date (6 digits at the start) and session ID (DayX), Pattern: YYMMDD_JSY_JSYXXX_SpMod_DayX
    match = re.match(r'(\d{6})_.*_(Day\d+)', session_folder)
    if match is None:
        match = re.match(r'(\d{6})_.*_(SpMO_OpenloopVR_(stationary|moving))', session_folder)
        
    date_str = match.group(1)
    session_id = match.group(2)

    # Define per-landmark window configuration
    # L1 (25cm): constrained before (close to onset zone)
    # L2 (55cm): full asymmetric
    # L3 (85cm): full asymmetric  
    # L4 (115cm): constrained after (close to corridor end)
    landmark_windows_config = [
    #     {'before': 25, 'after': 0},  # L1 at 25cm: [10, 35]
    #     {'before': 25, 'after': 0},  # L2 at 55cm: [35, 65]
    #     {'before': 25, 'after': 0},  # L3 at 85cm: [65, 95]
    #     {'before': 25, 'after': 0},  # L4 at 115cm: [95, 125]
        {'before': 15, 'after': 10},  # L1 at 25cm: [10, 35]
        {'before': 20, 'after': 10},  # L2 at 55cm: [35, 65]
        {'before': 20, 'after': 10},  # L3 at 85cm: [65, 95]
        {'before': 20, 'after': 10},  # L4 at 115cm: [95, 125]
    ]

    # Run landmark analysis
    results = run_landmark_analysis(
        normalized_spatial_activity=normalized_spatial_activity,
        bin_centers=bin_centers,
        layer_cells=layer_cells,
        reliable_valid_cells=reliable_valid_cells,
        # landmark_positions=[37, 65, 93, 120],

        landmark_positions=[25, 55, 85, 120],
        landmark_windows_config=landmark_windows_config,  # NEW
        landmark_window=10.0,  # fallback (not used if config provided)
        boundary_exclusion=(5, 5),
        exclude_first_bins=5,
        exclude_last_bins=5,
        trials_per_block=20,
        smoothing_sigma=1.0,
        save_path=data_filepath,
        session_id=session_id,
        date_str=date_str
    )
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")