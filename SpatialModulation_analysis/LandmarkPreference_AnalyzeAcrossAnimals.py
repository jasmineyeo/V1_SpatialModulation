"""
LandmarkPreference_AnalyzeAcrossAnimals.py
Script for comparing landmark preferences across multiple animals and sessions
Run this after running LandmarkPrefernce_SingleSessionAnalysis.py for each session for each animal

JSY, 11/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\V1_SpatialModulation")

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import h5py
from glob import glob
from collections import defaultdict
import pandas as pd

# ============================================================================
# STEP 1: LOAD ALL ANIMALS DATA
# ============================================================================

def extract_animal_id(path):
    """
    Extract animal ID from folder path.
    Example: 'JSY054_ChronicImaging' → 'JSY054'
    """
    # Look for pattern like JSY### or similar
    match = re.search(r'(JSY\d+)', path)
    if match:
        return match.group(1)
    
    # Fallback: use the folder name containing 'ChronicImaging'
    for part in path.split(os.sep):
        if 'ChronicImaging' in part or 'Chronic' in part:
            # Extract first part before underscore
            return part.split('_')[0]
    
    return 'Unknown'

def extract_session_day(path):
    """
    Extract day number from folder path.
    Example: '251030_JSY_JSY054_SpMod_Day1' → 'Day1'
    """
    # Look for 'Day#' pattern
    match = re.search(r'Day(\d+)', path, re.IGNORECASE)
    if match:
        return f"Day{match.group(1)}"
    
    return None
def verify_cell_counts(landmark_data, preproc_n_cells):
    """
    Check if cell counts match between landmark and preproc files.
    
    Returns:
    --------
    match : bool
        True if counts match
    total_landmark_cells : int
        Total cells in landmark file
    """
    # Count total cells from landmark file
    total_landmark_cells = 0
    for layer_name, layer_info in landmark_data['full_session'].items():
        total_landmark_cells += len(layer_info['valid_cells'])
    
    return total_landmark_cells == preproc_n_cells, total_landmark_cells


def load_spatial_with_layer_check(preproc_path, landmark_data, animal_id):
    """
    Load spatial data and organize by layer.
    If cell counts don't match, re-identify layers from preproc file.
    
    Parameters:
    -----------
    preproc_path : str
        Path to preproc.h5 file
    landmark_data : dict
        Landmark data containing layer assignments
    animal_id : str
        Animal identifier
    
    Returns:
    --------
    spatial_by_layer : dict
        Spatial activity organized by layer
    """
    
    print(f"    Loading spatial data from: {os.path.basename(preproc_path)}")
    
    with h5py.File(preproc_path, 'r') as f:
        # Load full spatial data
        full_spatial_activity = f['norm_spatial_activity'][:]
        full_raw_activity = f['spatial_activity'][:] if 'spatial_activity' in f else None
        bin_centers = f['bin_centers'][:]
        reliable_cells = f['combined_reliable'][:]
        med_coords = f['med_coords'][:]
        
        n_cells_preproc = full_spatial_activity.shape[0]
        print(f"      Preproc file has {n_cells_preproc} cells")
    
    # Check if cell counts match
    counts_match, n_cells_landmark = verify_cell_counts(landmark_data, n_cells_preproc)
    print(f"      Landmark file has {n_cells_landmark} cells assigned to layers")
    
    if counts_match:
        print(f"      ✓ Cell counts match! Using existing layer assignments.")
        
        # Use existing layer assignments from landmark file
        spatial_by_layer = {}
        
        for layer_name, layer_info in landmark_data['full_session'].items():
            layer_cell_indices = layer_info['valid_cells']
            
            with h5py.File(preproc_path, 'r') as f:
                full_spatial_activity = f['norm_spatial_activity'][:]
                bin_centers = f['bin_centers'][:]
                reliable_cells = f['combined_reliable'][:]
            
            spatial_by_layer[layer_name] = {
                'normalized_spatial_activity': full_spatial_activity[layer_cell_indices, :, :],
                'reliable_cells': reliable_cells[layer_cell_indices],
                'bin_centers': bin_centers,
                'cell_indices': layer_cell_indices,
                'n_cells': len(layer_cell_indices)
            }
            print(f"        {layer_name}: {len(layer_cell_indices)} cells")
    
    else:
        print(f"      ⚠ Cell counts DON'T match! Re-identifying layers from preproc file...")
        
        # Re-identify layers from med_coords in preproc file
        from helper.SpatialModulationIndexLayerSpecific import SpatialModulationIndexLayerSpecific as SMI_Layer
        
        layer_cells, layer_boundaries = SMI_Layer.identify_layers(med_coords)
        
        # Now organize spatial data by re-identified layers
        spatial_by_layer = {}
        
        with h5py.File(preproc_path, 'r') as f:
            full_spatial_activity = f['norm_spatial_activity'][:]
            bin_centers = f['bin_centers'][:]
            reliable_cells = f['combined_reliable'][:]
        
        for layer_name, layer_cell_indices in layer_cells.items():
            # Find intersection with landmark-identified cells
            # (only use cells that were also identified in landmark analysis)
            landmark_valid_cells = set()
            for lm_layer_name, lm_layer_info in landmark_data['full_session'].items():
                landmark_valid_cells.update(lm_layer_info['valid_cells'])
            
            # Keep only cells that are in both sets
            valid_indices = np.array([idx for idx in layer_cell_indices 
                                     if idx in landmark_valid_cells])
            
            if len(valid_indices) > 0:
                spatial_by_layer[layer_name] = {
                    'normalized_spatial_activity': full_spatial_activity[valid_indices, :, :],
                    'reliable_cells': reliable_cells[valid_indices],
                    'bin_centers': bin_centers,
                    'cell_indices': valid_indices,
                    'n_cells': len(valid_indices)
                }
                print(f"        {layer_name}: {len(valid_indices)} cells (re-identified)")
    
    return spatial_by_layer


def load_all_animals(parent_dir, landmark_pattern="*landmark_preferences.h5", 
                    preproc_pattern="*_preproc.h5"):
    """
    Load all animal data from hierarchical directory structure.
    Now handles spatial data from separate preproc files.
    
    Returns:
    --------
    all_animals_landmark : dict
        Nested dict: {animal_id: {session_id: landmark_data}}
    all_animals_spatial : dict
        Nested dict: {animal_id: {session_id: spatial_data_by_layer}}
    metadata : dict
        Information about animals and sessions
    """
    
    print("\n" + "="*70)
    print("LOADING ALL ANIMALS DATA")
    print("="*70)
    print(f"Scanning: {parent_dir}\n")
    
    # Find all landmark preference files
    landmark_files = glob(os.path.join(parent_dir, "**", landmark_pattern), recursive=True)
    
    print(f"Found {len(landmark_files)} landmark preference files\n")
    
    # Organize by animal and session
    all_animals_landmark = defaultdict(dict)
    all_animals_spatial = defaultdict(dict)
    file_mapping = defaultdict(lambda: defaultdict(dict))
    
    # Process each landmark file
    for landmark_path in landmark_files:
        animal_id = extract_animal_id(landmark_path)
        session_id = extract_session_day(landmark_path)
        
        if session_id is None:
            print(f"WARNING: Could not extract session from {landmark_path}")
            continue
        
        print(f"  Processing: {animal_id} - {session_id}")
        print(f"    Landmark file: {os.path.basename(landmark_path)}")
        
        try:
            # Load landmark preference data
            with h5py.File(landmark_path, 'r') as f:
                session_data = {
                    'session_id': f.attrs.get('session_id', session_id),
                    'date': f.attrs.get('date', 'Unknown'),
                    'full_session': {},
                    'dynamics': {}
                }
                
                # Load full session results
                if 'full_session' in f:
                    full_grp = f['full_session']
                    for h5_layer_name in full_grp.keys():
                        layer_grp = full_grp[h5_layer_name]
                        
                        # Restore original layer name (L2_3 → L2/3)
                        if 'original_name' in layer_grp.attrs:
                            layer_name = str(layer_grp.attrs['original_name'])
                        else:
                            if '_' in h5_layer_name and h5_layer_name.startswith('L'):
                                parts = h5_layer_name.split('_')
                                if len(parts) == 2 and parts[1].isdigit():
                                    layer_name = f"{parts[0]}/{parts[1]}"
                                else:
                                    layer_name = h5_layer_name
                            else:
                                layer_name = h5_layer_name
                        
                        session_data['full_session'][layer_name] = {
                            'landmark_counts': layer_grp['landmark_counts'][:],
                            'landmark_proportions': layer_grp['landmark_proportions'][:],
                            'n_cells': layer_grp.attrs['n_cells'],
                            'preferred_landmarks': layer_grp['preferred_landmarks'][:],
                            'valid_cells': layer_grp['valid_cells'][:]
                        }
                
                # Load dynamics if available
                if 'dynamics' in f:
                    dyn_grp = f['dynamics']
                    session_data['dynamics']['n_blocks'] = dyn_grp.attrs.get('n_blocks', 0)
                    session_data['dynamics']['trials_per_block'] = dyn_grp.attrs.get('trials_per_block', 30)
                    if 'landmark_positions' in dyn_grp:
                        session_data['dynamics']['landmark_positions'] = dyn_grp['landmark_positions'][:]
            
            all_animals_landmark[animal_id][session_id] = session_data
            file_mapping[animal_id][session_id]['landmark'] = landmark_path
            
            # NEW: Search for preproc file in SAME directory
            landmark_dir = os.path.dirname(landmark_path)
            preproc_files_in_dir = glob(os.path.join(landmark_dir, preproc_pattern))
            
            if len(preproc_files_in_dir) == 0:
                print(f"    ⚠ WARNING: No preproc file found in {landmark_dir}")
                continue
            
            if len(preproc_files_in_dir) > 1:
                print(f"    ⚠ WARNING: Multiple preproc files found, using first one")
            
            preproc_path = preproc_files_in_dir[0]
            
            # Load spatial data with layer verification
            spatial_data = load_spatial_with_layer_check(preproc_path, session_data, animal_id)
            all_animals_spatial[animal_id][session_id] = spatial_data
            file_mapping[animal_id][session_id]['preproc'] = preproc_path
            
        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create metadata summary
    metadata = {
        'animals': list(all_animals_landmark.keys()),
        'n_animals': len(all_animals_landmark),
        'sessions_per_animal': {},
        'all_sessions': set(),
        'file_mapping': dict(file_mapping)
    }
    
    for animal_id in all_animals_landmark.keys():
        sessions = sorted(all_animals_landmark[animal_id].keys())
        metadata['sessions_per_animal'][animal_id] = sessions
        metadata['all_sessions'].update(sessions)
    
    metadata['all_sessions'] = sorted(list(metadata['all_sessions']))
    
    # Print summary
    print("\n" + "="*70)
    print("DATA LOADING SUMMARY")
    print("="*70)
    print(f"\nAnimals loaded: {metadata['n_animals']}")
    for animal_id in metadata['animals']:
        sessions = metadata['sessions_per_animal'][animal_id]
        print(f"  {animal_id}: {len(sessions)} sessions - {', '.join(sessions)}")
    
    print(f"\nAll unique sessions: {', '.join(metadata['all_sessions'])}")
    print("="*70 + "\n")
    
    return dict(all_animals_landmark), dict(all_animals_spatial), metadata


# ============================================================================
# STEP 2: CREATE HEATMAP VISUALIZATIONS
# ============================================================================

def create_session_averaged_heatmaps(all_animals_landmark, metadata, 
                                    landmark_positions=[30, 60, 90, 120],
                                    save_path=None):
    """
    Create heatmaps showing landmark preferences by layer.
    
    Generates:
    1. Grand average (all animals, all sessions)
    2. Session-specific (averaged across animals for each session)
    
    Parameters:
    -----------
    all_animals_landmark : dict
        Nested dict with landmark data
    metadata : dict
        Metadata about animals and sessions
    landmark_positions : list
        Positions of landmarks in cm
    save_path : str, optional
        Directory to save figures
    """
    
    print("\n" + "="*70)
    print("CREATING HEATMAP VISUALIZATIONS")
    print("="*70)
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    n_landmarks = len(landmark_positions)
    all_sessions = metadata['all_sessions']
    
    # Collect all data for grand average
    grand_data = defaultdict(lambda: defaultdict(list))
    
    for animal_id, sessions in all_animals_landmark.items():
        for session_id, session_data in sessions.items():
            for layer_name, layer_data in session_data['full_session'].items():
                if layer_name in layer_order:
                    grand_data[layer_name]['proportions'].append(layer_data['landmark_proportions'])
                    grand_data[layer_name]['counts'].append(layer_data['landmark_counts'])
    
    # Calculate grand average
    grand_avg = np.zeros((len(layer_order), n_landmarks))
    grand_sem = np.zeros((len(layer_order), n_landmarks))
    grand_counts = np.zeros((len(layer_order), n_landmarks), dtype=int)
    
    for i, layer_name in enumerate(layer_order):
        if layer_name in grand_data and len(grand_data[layer_name]['proportions']) > 0:
            props = np.array(grand_data[layer_name]['proportions'])
            counts = np.array(grand_data[layer_name]['counts'])
            
            grand_avg[i, :] = np.mean(props, axis=0)
            grand_sem[i, :] = stats.sem(props, axis=0) if len(props) > 1 else np.zeros(n_landmarks)
            grand_counts[i, :] = np.sum(counts, axis=0)
    
    # Create figure with grand average + session-specific panels
    n_panels = len(all_sessions) + 1  # +1 for grand average
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
    axes = axes.flatten() if n_panels > 1 else [axes]
    
    # Plot grand average in first panel
    ax = axes[0]
    im = ax.imshow(grand_avg, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(n_landmarks))
    ax.set_yticks(np.arange(len(layer_order)))
    ax.set_xticklabels([f"L{i+1}\n({landmark_positions[i]:.0f}cm)" for i in range(n_landmarks)])
    ax.set_yticklabels(layer_order)
    
    # Add text annotations
    for i in range(len(layer_order)):
        for j in range(n_landmarks):
            text_str = f'{grand_avg[i, j]:.2f}\n±{grand_sem[i, j]:.2f}\n(n={grand_counts[i, j]})'
            ax.text(j, i, text_str, ha="center", va="center", color="black", fontsize=9)
    
    ax.set_xlabel('Landmark', fontsize=11)
    ax.set_ylabel('Layer', fontsize=11)
    ax.set_title('Grand Average\n(All Animals, All Sessions)', fontsize=12, fontweight='bold')
    
    # Plot session-specific heatmaps
    for panel_idx, session_id in enumerate(all_sessions, start=1):
        ax = axes[panel_idx]
        
        # Collect data for this session
        session_data_by_layer = defaultdict(lambda: {'proportions': [], 'counts': []})
        
        for animal_id, sessions in all_animals_landmark.items():
            if session_id in sessions:
                for layer_name, layer_data in sessions[session_id]['full_session'].items():
                    if layer_name in layer_order:
                        session_data_by_layer[layer_name]['proportions'].append(layer_data['landmark_proportions'])
                        session_data_by_layer[layer_name]['counts'].append(layer_data['landmark_counts'])
        
        # Calculate averages
        session_avg = np.zeros((len(layer_order), n_landmarks))
        session_sem = np.zeros((len(layer_order), n_landmarks))
        session_counts = np.zeros((len(layer_order), n_landmarks), dtype=int)
        
        for i, layer_name in enumerate(layer_order):
            if layer_name in session_data_by_layer and len(session_data_by_layer[layer_name]['proportions']) > 0:
                props = np.array(session_data_by_layer[layer_name]['proportions'])
                counts = np.array(session_data_by_layer[layer_name]['counts'])
                
                session_avg[i, :] = np.mean(props, axis=0)
                session_sem[i, :] = stats.sem(props, axis=0) if len(props) > 1 else np.zeros(n_landmarks)
                session_counts[i, :] = np.sum(counts, axis=0)
        
        # Plot
        im = ax.imshow(session_avg, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(n_landmarks))
        ax.set_yticks(np.arange(len(layer_order)))
        ax.set_xticklabels([f"L{i+1}\n({landmark_positions[i]:.0f}cm)" for i in range(n_landmarks)])
        ax.set_yticklabels(layer_order)
        
        # Add text annotations
        n_animals_this_session = len([a for a in all_animals_landmark.keys() if session_id in all_animals_landmark[a]])
        
        for i in range(len(layer_order)):
            for j in range(n_landmarks):
                if session_counts[i, j] > 0:
                    text_str = f'{session_avg[i, j]:.2f}\n±{session_sem[i, j]:.2f}\n(n={session_counts[i, j]})'
                else:
                    text_str = 'N/A'
                ax.text(j, i, text_str, ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xlabel('Landmark', fontsize=11)
        ax.set_ylabel('Layer', fontsize=11)
        ax.set_title(f'{session_id}\n({n_animals_this_session} animals)', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_panels, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Proportion of Cells')
    
    plt.tight_layout(rect=[0, 0, 0.90, 1])  # Leave space on right for colorbar
    
    if save_path:
        fig_path = os.path.join(save_path, 'heatmap_evolution_across_sessions.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(fig_path)}")
    
    # plt.show()
    plt.close()
    
    return fig


# ============================================================================
# STEP 3: CREATE VIOLIN PLOTS
# ============================================================================

def create_violin_plots(all_animals_landmark, metadata, landmark_positions=[30, 60, 90, 120],
                       save_path=None):
    """
    Create violin plots showing distribution of landmark preferences by layer.
    
    Parameters:
    -----------
    all_animals_landmark : dict
        Nested dict with landmark data
    metadata : dict
        Metadata about animals and sessions
    landmark_positions : list
        Positions of landmarks in cm
    save_path : str, optional
        Directory to save figure
    """
    
    print("\n" + "="*70)
    print("CREATING VIOLIN PLOTS")
    print("="*70)
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    n_landmarks = len(landmark_positions)
    
    # Collect all data points (each = one session from one animal)
    data_for_plotting = defaultdict(lambda: defaultdict(list))
    
    for animal_id, sessions in all_animals_landmark.items():
        for session_id, session_data in sessions.items():
            for layer_name, layer_data in session_data['full_session'].items():
                if layer_name in layer_order:
                    proportions = layer_data['landmark_proportions']
                    for lm_idx, prop in enumerate(proportions):
                        data_for_plotting[layer_name][lm_idx].append(prop)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for ax_idx, layer_name in enumerate(layer_order):
        ax = axes[ax_idx]
        
        if layer_name not in data_for_plotting:
            ax.text(0.5, 0.5, f'{layer_name}: No data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            continue
        
        # Prepare data for violin plot
        plot_data = []
        positions = []
        
        for lm_idx in range(n_landmarks):
            if lm_idx in data_for_plotting[layer_name]:
                plot_data.append(data_for_plotting[layer_name][lm_idx])
                positions.append(lm_idx)
        
        if len(plot_data) == 0:
            continue
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=positions, widths=0.7,
                             showmeans=True, showmedians=True)
        
        # Customize violin colors
        colors = plt.cm.Set1(np.linspace(0, 1, n_landmarks))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        # Overlay individual points
        for lm_idx, data_points in enumerate(plot_data):
            x = np.random.normal(positions[lm_idx], 0.04, size=len(data_points))
            ax.scatter(x, data_points, alpha=0.4, s=30, color='black', zorder=10)
        
        # Format
        ax.set_xticks(range(n_landmarks))
        ax.set_xticklabels([f"L{i+1}\n({landmark_positions[i]:.0f}cm)" for i in range(n_landmarks)])
        ax.set_ylabel('Proportion of Cells', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add sample size
        n_points = sum(len(d) for d in plot_data)
        ax.text(0.02, 0.98, f'n = {n_points} observations', transform=ax.transAxes,
               va='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    fig.suptitle('Landmark Preference Distributions by Layer\n(All Animals, All Sessions)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig_path = os.path.join(save_path, 'violin_plots_by_layer.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(fig_path)}")
    
    # plt.show()
    
    return fig


# ============================================================================
# STEP 4: EXPORT COMBINED SPATIAL DATA FOR NEURAL RESPONSE PLOTS
# ============================================================================

def create_combined_response_plot_single_layer_session(all_animals_spatial, layer_name, session_id, 
                                                       metadata, landmark_positions=[30, 60, 90, 120]):
    """
    Create a response plot for a single layer-session combination,
    combining all cells from all animals.
    
    Parameters:
    -----------
    all_animals_spatial : dict
        Spatial data organized by animal and session
    layer_name : str
        Layer to plot (e.g., 'L2/3')
    session_id : str
        Session to plot (e.g., 'Day1')
    metadata : dict
        Metadata about animals
    landmark_positions : list
        Positions of landmarks in cm
    
    Returns:
    --------
    combined_avg_activity : numpy.ndarray
        Combined averaged activity (n_cells, n_bins)
    combined_reliable : numpy.ndarray
        Combined reliable cells mask
    bin_centers : numpy.ndarray
        Spatial bin centers
    n_cells_total : int
        Total number of cells
    n_animals : int
        Number of animals contributed
    """
    
    all_cells_avg_activity = []
    all_cells_reliable = []
    bin_centers_ref = None
    n_animals_contributed = 0
    
    for animal_id in metadata['animals']:
        # Check if this animal has this session
        if session_id not in all_animals_spatial[animal_id]:
            continue
        
        # Get spatial data (already organized by layer)
        spatial_data_session = all_animals_spatial[animal_id][session_id]
        
        # Check if this layer exists
        if layer_name not in spatial_data_session:
            continue
        
        # Get data for this layer
        layer_spatial_data = spatial_data_session[layer_name]
        
        # Extract the spatial activity: (n_cells, n_trials, n_bins)
        cell_activity = layer_spatial_data['normalized_spatial_activity']
        cell_reliable = layer_spatial_data['reliable_cells']
        
        # CRITICAL: Average across trials for this animal
        # Shape: (n_cells, n_trials, n_bins) → (n_cells, n_bins)
        cell_activity_avg = np.mean(cell_activity, axis=1)
        
        all_cells_avg_activity.append(cell_activity_avg)
        all_cells_reliable.append(cell_reliable)
        n_animals_contributed += 1
        
        # Store bin_centers
        if bin_centers_ref is None:
            bin_centers_ref = layer_spatial_data['bin_centers']
    
    # Combine all cells from all animals
    if len(all_cells_avg_activity) > 0:
        combined_avg_activity = np.concatenate(all_cells_avg_activity, axis=0)
        combined_reliable = np.concatenate(all_cells_reliable, axis=0)
        
        return (combined_avg_activity, combined_reliable, bin_centers_ref, 
                combined_avg_activity.shape[0], n_animals_contributed)
    else:
        return None, None, None, 0, 0


def create_response_plot_for_averaged_data(combined_avg_activity, combined_reliable, 
                                           bin_centers, ax, title, 
                                           landmark_positions=[30, 60, 90, 120],
                                           show_landmarks=True):
    """
    Create a response plot for pre-averaged data (no trial splitting needed).
    Modified version of your create_response_plot function.
    
    Parameters:
    -----------
    combined_avg_activity : numpy.ndarray
        Already trial-averaged activity (n_cells, n_bins)
    combined_reliable : numpy.ndarray
        Reliable cells mask
    bin_centers : numpy.ndarray
        Spatial bin centers
    ax : matplotlib.axes.Axes
        Axes to plot on
    title : str
        Plot title
    landmark_positions : list
        Positions of landmarks in cm
    show_landmarks : bool
        Whether to show landmark lines
    
    Returns:
    --------
    sorted_indices : numpy.ndarray
        Indices of cells sorted by peak location
    """
    
    # Select only reliable cells
    reliable_indices = np.where(combined_reliable)[0]
    
    if len(reliable_indices) == 0:
        ax.text(0.5, 0.5, 'No reliable cells', ha='center', va='center',
               transform=ax.transAxes, fontsize=12)
        ax.set_title(title)
        return None
    
    reliable_activity = combined_avg_activity[reliable_indices]
    
    # Find peak location for each cell
    peak_locations = np.argmax(reliable_activity, axis=1)
    
    # Sort cells by peak location
    sorted_indices = np.argsort(peak_locations)
    sorted_activity = reliable_activity[sorted_indices]
    
    # Normalize each cell individually (0-1)
    sorted_activity_norm = np.zeros_like(sorted_activity)
    for i in range(len(sorted_activity)):
        cell_min = np.min(sorted_activity[i])
        cell_max = np.max(sorted_activity[i])
        if cell_max > cell_min:
            sorted_activity_norm[i] = (sorted_activity[i] - cell_min) / (cell_max - cell_min)
        else:
            sorted_activity_norm[i] = sorted_activity[i]
    
    # Create colormap
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('EnhancedBlues', 
                                           [(1,1,1), (0.8,0.8,1), (0.4,0.4,0.9), (0,0,0.8), (0,0,0.5)])
    
    # Plot
    im = ax.imshow(sorted_activity_norm, aspect='auto', cmap=cmap, 
                  interpolation='nearest', vmin=0, vmax=1)
    
    # Add landmark lines if requested
    if show_landmarks and bin_centers is not None:
        for lm_pos in landmark_positions:
            # Find closest bin to landmark position
            lm_bin = np.argmin(np.abs(bin_centers - lm_pos))
            ax.axvline(lm_bin, color='red', linestyle='--', alpha=0.5, linewidth=1)
    
    # Labels
    ax.set_xlabel('Spatial Bin', fontsize=9)
    ax.set_ylabel('Cell # (sorted)', fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    # Add cell count annotation
    ax.text(0.02, 0.98, f'n={len(reliable_indices)}', 
           transform=ax.transAxes, va='top', fontsize=8,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    return reliable_indices[sorted_indices]


def create_across_sessions_response_plots_all_layers(all_animals_spatial, metadata,
                                                     landmark_positions=[30, 60, 90, 120],
                                                     save_path=None):
    """
    Create ONE BIG FIGURE with response plots for all layers and all sessions.
    Layout: 4 rows (layers) × N columns (sessions)
    
    Parameters:
    -----------
    all_animals_spatial : dict
        Spatial data
    metadata : dict
        Metadata
    landmark_positions : list
        Landmark positions in cm
    save_path : str, optional
        Directory to save figure
    """
    
    print("\n" + "="*70)
    print("CREATING COMBINED RESPONSE PLOTS - ALL LAYERS")
    print("="*70)
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    all_sessions = metadata['all_sessions']
    
    n_layers = len(layer_order)
    n_sessions = len(all_sessions)
    
    # Create figure
    fig, axes = plt.subplots(n_layers, n_sessions, 
                            figsize=(4*n_sessions, 3.5*n_layers))
    
    # Handle single row/column cases
    if n_layers == 1 and n_sessions == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    elif n_sessions == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each layer-session combination
    for row_idx, layer_name in enumerate(layer_order):
        for col_idx, session_id in enumerate(all_sessions):
            ax = axes[row_idx, col_idx]
            
            print(f"  Processing {layer_name} - {session_id}...")
            
            # Get combined data for this layer-session
            result = create_combined_response_plot_single_layer_session(
                all_animals_spatial, layer_name, session_id, metadata, landmark_positions
            )
            
            if result[0] is not None:
                combined_avg, combined_reliable, bin_centers, n_cells, n_animals = result
                
                # Create response plot
                title = f'{layer_name} - {session_id}\n({n_animals} animals, {n_cells} cells)'
                create_response_plot_for_averaged_data(
                    combined_avg, combined_reliable, bin_centers, ax, title, 
                    landmark_positions, show_landmarks=True
                )
                
                print(f"    ✓ {n_cells} cells from {n_animals} animals")
            else:
                ax.text(0.5, 0.5, f'{layer_name} - {session_id}\nNo data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')
                print(f"    ✗ No data")
    
    # Add overall title
    fig.suptitle('Neural Response Evolution Across Sessions\n(All Animals Combined)', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        fig_path = os.path.join(save_path, 'response_plots_all_layers_all_sessions.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {os.path.basename(fig_path)}")
    
    # plt.show()
    
    return fig


def create_across_sessions_response_plots_per_layer(all_animals_spatial, metadata,
                                                    landmark_positions=[30, 60, 90, 120],
                                                    save_path=None):
    """
    Create SEPARATE FIGURES for each layer (4 figures total).
    Each figure shows all sessions for one layer.
    
    Parameters:
    -----------
    all_animals_spatial : dict
        Spatial data
    metadata : dict
        Metadata
    landmark_positions : list
        Landmark positions in cm
    save_path : str, optional
        Directory to save figures
    """
    
    print("\n" + "="*70)
    print("CREATING RESPONSE PLOTS - SEPARATE FIGURES PER LAYER")
    print("="*70)
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    all_sessions = metadata['all_sessions']
    n_sessions = len(all_sessions)
    
    figures = {}
    
    for layer_name in layer_order:
        print(f"\nCreating figure for {layer_name}...")
        
        # Determine grid layout
        n_cols = min(4, n_sessions)
        n_rows = int(np.ceil(n_sessions / n_cols))
        
        # Create figure for this layer
        fig, axes = plt.subplots(n_rows, n_cols, 
                                figsize=(5*n_cols, 4*n_rows))
        
        # Flatten axes array
        if n_sessions == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
        
        # Plot each session
        for idx, session_id in enumerate(all_sessions):
            ax = axes[idx]
            
            print(f"  Processing {session_id}...")
            
            # Get combined data
            result = create_combined_response_plot_single_layer_session(
                all_animals_spatial, layer_name, session_id, metadata, landmark_positions
            )
            
            if result[0] is not None:
                combined_avg, combined_reliable, bin_centers, n_cells, n_animals = result
                
                # Create response plot
                title = f'{session_id}\n({n_animals} animals, {n_cells} cells)'
                create_response_plot_for_averaged_data(
                    combined_avg, combined_reliable, bin_centers, ax, title, 
                    landmark_positions, show_landmarks=True
                )
                
                print(f"    ✓ {n_cells} cells from {n_animals} animals")
            else:
                ax.text(0.5, 0.5, f'{session_id}\nNo data', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.axis('off')
                print(f"    ✗ No data")
        
        # Hide unused subplots
        for idx in range(n_sessions, len(axes)):
            axes[idx].axis('off')
        
        # Add overall title
        fig.suptitle(f'{layer_name} - Neural Response Evolution Across Sessions\n(All Animals Combined)', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            # Sanitize layer name for filename
            safe_layer_name = layer_name.replace('/', '_')
            fig_path = os.path.join(save_path, f'response_plots_{safe_layer_name}_across_sessions.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {os.path.basename(fig_path)}")
        
        # plt.show()
        
        figures[layer_name] = fig
    
    return figures


# ============================================================================
# STEP 5: CREATE ANIMAL-SPECIFIC HEATMAPS (SUPPLEMENTARY)
# ============================================================================

def create_animal_specific_heatmaps(all_animals_landmark, metadata,
                                   landmark_positions=[30, 60, 90, 120],
                                   save_path=None):
    """
    Create one heatmap per animal (averaged across all sessions).
    
    Parameters:
    -----------
    all_animals_landmark : dict
        Landmark data
    metadata : dict
        Metadata
    landmark_positions : list
        Landmark positions
    save_path : str, optional
        Directory to save figures
    """
    
    print("\n" + "="*70)
    print("CREATING ANIMAL-SPECIFIC HEATMAPS")
    print("="*70)
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    n_landmarks = len(landmark_positions)
    n_animals = len(metadata['animals'])
    
    # Create figure
    n_cols = min(3, n_animals)
    n_rows = int(np.ceil(n_animals / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    axes = axes.flatten() if n_animals > 1 else [axes]
    
    for ax_idx, animal_id in enumerate(metadata['animals']):
        ax = axes[ax_idx]
        
        # Collect data for this animal (average across sessions)
        animal_data = defaultdict(lambda: {'proportions': [], 'counts': []})
        
        for session_id, session_data in all_animals_landmark[animal_id].items():
            for layer_name, layer_data in session_data['full_session'].items():
                if layer_name in layer_order:
                    animal_data[layer_name]['proportions'].append(layer_data['landmark_proportions'])
                    animal_data[layer_name]['counts'].append(layer_data['landmark_counts'])
        
        # Calculate averages
        animal_avg = np.zeros((len(layer_order), n_landmarks))
        animal_counts = np.zeros((len(layer_order), n_landmarks), dtype=int)
        
        for i, layer_name in enumerate(layer_order):
            if layer_name in animal_data and len(animal_data[layer_name]['proportions']) > 0:
                props = np.array(animal_data[layer_name]['proportions'])
                counts = np.array(animal_data[layer_name]['counts'])
                
                animal_avg[i, :] = np.mean(props, axis=0)
                animal_counts[i, :] = np.sum(counts, axis=0)
        
        # Plot
        im = ax.imshow(animal_avg, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        ax.set_xticks(np.arange(n_landmarks))
        ax.set_yticks(np.arange(len(layer_order)))
        ax.set_xticklabels([f"L{i+1}\n({landmark_positions[i]:.0f}cm)" for i in range(n_landmarks)])
        ax.set_yticklabels(layer_order)
        
        # Add text annotations
        for i in range(len(layer_order)):
            for j in range(n_landmarks):
                if animal_counts[i, j] > 0:
                    text_str = f'{animal_avg[i, j]:.2f}\n(n={animal_counts[i, j]})'
                else:
                    text_str = 'N/A'
                ax.text(j, i, text_str, ha="center", va="center", color="black", fontsize=9)
        
        n_sessions = len(all_animals_landmark[animal_id])
        ax.set_xlabel('Landmark', fontsize=11)
        ax.set_ylabel('Layer', fontsize=11)
        ax.set_title(f'{animal_id}\n({n_sessions} sessions)', fontsize=12, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_animals, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Proportion of Cells')  
      
    fig.suptitle('Animal-Specific Landmark Preferences\n(Averaged Across All Sessions)',
                fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.90, 1])  # Leave space on right for colorbar
    
    if save_path:
        fig_path = os.path.join(save_path, 'animal_specific_heatmaps.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(fig_path)}")
    
    # plt.show()
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_across_animals_analysis(parent_dir, save_path=None, 
                                landmark_positions=[30, 60, 90, 120]):
    """
    Complete workflow for across-animals analysis.
    """
    
    print("\n" + "="*80)
    print("ACROSS-ANIMALS LANDMARK PREFERENCE ANALYSIS")
    print("="*80)
    print(f"Parent directory: {parent_dir}")
    if save_path:
        print(f"Save directory: {save_path}")
    print("="*80)
    
    # Create save directory if needed
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}\n")
    
    # Step 1: Load all data
    all_animals_landmark, all_animals_spatial, metadata = load_all_animals(parent_dir)
    
    if metadata['n_animals'] == 0:
        print("ERROR: No animals found!")
        return None
    
    # Step 2: Create heatmaps
    fig_heatmaps = create_session_averaged_heatmaps(
        all_animals_landmark, metadata, landmark_positions, save_path
    )
    
    # Step 3: Create violin plots
    fig_violin = create_violin_plots(
        all_animals_landmark, metadata, landmark_positions, save_path
    )
    
    # Step 4A: Create combined response plots - ONE BIG FIGURE
    fig_response_all = create_across_sessions_response_plots_all_layers(
        all_animals_spatial, metadata, landmark_positions, save_path
    )
    
    # Step 4B: Create combined response plots - SEPARATE PER LAYER
    figs_response_per_layer = create_across_sessions_response_plots_per_layer(
        all_animals_spatial, metadata, landmark_positions, save_path
    )
    
    # Step 5: Create animal-specific heatmaps
    fig_animal_specific = create_animal_specific_heatmaps(
        all_animals_landmark, metadata, landmark_positions, save_path
    )
    
    # Compile results
    results = {
        'all_animals_landmark': all_animals_landmark,
        'all_animals_spatial': all_animals_spatial,
        'metadata': metadata,
        'figures': {
            'heatmaps': fig_heatmaps,
            'violin': fig_violin,
            'response_all_layers': fig_response_all,
            'response_per_layer': figs_response_per_layer,
            'animal_specific': fig_animal_specific
        }
    }
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated figures:")
    print(f"  1. Heatmap evolution (Grand + Session-specific)")
    print(f"  2. Violin plots (Population distributions)")
    print(f"  3. Response plots - All layers combined (ONE BIG FIGURE)")
    print(f"  4. Response plots - Separate per layer (4 figures)")
    print(f"  5. Animal-specific heatmaps (Supplementary)")
    if save_path:
        print(f"\n  All saved to: {save_path}")
    print("="*80 + "\n")
    
    return results

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    # Configure paths
    parent_dir = r"D:\V1_SpatialModulation\2p\V1_prism"
    save_dir = r"D:\V1_SpatialModulation\2p\V1_prism\across_animals_analysis"
    
    # Run analysis
    results = run_across_animals_analysis(
        parent_dir=parent_dir,
        save_path=save_dir,
        landmark_positions=[25, 55, 85, 115],
    )
    
    # # Access results
    # if results:
    #     print("\n" + "="*80)
    #     print("RESULTS SUMMARY")
    #     print("="*80)
        
    #     metadata = results['metadata']
    #     print(f"\nAnimals analyzed: {metadata['n_animals']}")
    #     for animal_id in metadata['animals']:
    #         sessions = metadata['sessions_per_animal'][animal_id]
    #         print(f"  {animal_id}: {sessions}")
        
    #     print(f"\nCombined spatial data available for sessions:")
    #     for session_id in results['combined_spatial'].keys():
    #         print(f"  {session_id}:")
    #         for layer_name, layer_data in results['combined_spatial'][session_id].items():
    #             print(f"    {layer_name}: {layer_data['n_cells']} cells from {layer_data['n_animals']} animals")
        
    #     print("\n" + "="*80)
    #     print("NEXT STEPS")
    #     print("="*80)
    #     print("\n1. Use your existing visualization script to create neural response plots")
    #     print("   Load from: combined_spatial_data_across_animals.h5")
    #     print("\n2. Example code to load combined data:")
    #     print("""
    # import h5py
    
    # with h5py.File('combined_spatial_data_across_animals.h5', 'r') as f:
    #     # Access Day1, L2/3 data
    #     day1_l23 = f['Day1']['L2_3']
    #     activity = day1_l23['normalized_spatial_activity'][:]
    #     bin_centers = day1_l23['bin_centers'][:]
    #     reliable_cells = day1_l23['reliable_cells'][:]
        
    #     # Use with your existing visualization script
    #     # create_response_plot(activity, reliable_cells)
    #     """)
    #     print("="*80)
