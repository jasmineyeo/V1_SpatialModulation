"""
LandmarkPrefernce_CompareSessionsWithinAnimal.py
Script for comparing landmark preferences across multiple recording sessions within the same animal
Run this after running LandmarkPrefernce_SingleSessionAnalysis.py for each session

JSY, 11/2025
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import h5py
from glob import glob

# ============================================================================
# LOADING FUNCTIONS
# ============================================================================
def load_session_data(h5_path):
    """
    Load landmark preference data from a single session HDF5 file.
    Handles sanitized layer names (L2_3) and restores them (L2/3).
    
    Parameters:
    -----------
    h5_path : str
        Path to HDF5 file
    
    Returns:
    --------
    session_data : dict
        Dictionary containing session data
    """
    
    session_data = {
        'full_session': {},
        'dynamics': {}
    }
    
    def restore_layer_name(h5_layer_name, layer_grp):
        """
        Restore original layer name from sanitized HDF5 name.
        Checks for 'original_name' attribute first, then converts back.
        """
        # First, check if original name was stored as attribute
        if 'original_name' in layer_grp.attrs:
            return str(layer_grp.attrs['original_name'])
        
        # Fallback: convert sanitized name back to original
        # L2_3 → L2/3
        # L4 → L4 (unchanged)
        if h5_layer_name.startswith('L') and '_' in h5_layer_name:
            # Check if it looks like a sanitized L2/3
            parts = h5_layer_name.split('_')
            if len(parts) == 2 and parts[0] in ['L2', 'L5', 'L6'] and parts[1].isdigit():
                return f"{parts[0]}/{parts[1]}"
        
        return h5_layer_name
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Load metadata
            session_data['session_id'] = f.attrs.get('session_id', 'Unknown')
            if 'date' in f.attrs:
                session_data['date'] = f.attrs['date']
            
            # Load full session results
            if 'full_session' not in f:
                raise KeyError(f"'full_session' group not found in {h5_path}")
            
            full_grp = f['full_session']
            
            # print(f"  Loading {os.path.basename(h5_path)}:")
            
            for h5_layer_name in full_grp.keys():
                layer_grp = full_grp[h5_layer_name]
                
                # Restore original layer name
                original_layer_name = restore_layer_name(h5_layer_name, layer_grp)
                # print(f"    Layer: '{h5_layer_name}' → '{original_layer_name}'")
                
                # Load data
                layer_data = {}
                
                # Load landmark_counts (REQUIRED)
                if 'landmark_counts' in layer_grp:
                    layer_data['landmark_counts'] = layer_grp['landmark_counts'][:]
                else:
                    raise KeyError(f"'landmark_counts' not found in {h5_layer_name}")
                
                # Load landmark_proportions (REQUIRED)
                if 'landmark_proportions' in layer_grp:
                    layer_data['landmark_proportions'] = layer_grp['landmark_proportions'][:]
                else:
                    # Calculate from counts if not present
                    counts = layer_data['landmark_counts']
                    total = np.sum(counts)
                    layer_data['landmark_proportions'] = counts / total if total > 0 else counts
                
                # Load n_cells
                if 'n_cells' in layer_grp.attrs:
                    layer_data['n_cells'] = layer_grp.attrs['n_cells']
                else:
                    # Calculate from counts if not present
                    layer_data['n_cells'] = int(np.sum(layer_data['landmark_counts']))
                
                # Optional: load additional data if present
                if 'preferred_landmarks' in layer_grp:
                    layer_data['preferred_landmarks'] = layer_grp['preferred_landmarks'][:]
                
                if 'valid_cells' in layer_grp:
                    layer_data['valid_cells'] = layer_grp['valid_cells'][:]
                
                # Store with ORIGINAL layer name as key
                session_data['full_session'][original_layer_name] = layer_data
            
            # Load dynamics if available (OPTIONAL)
            if 'dynamics' in f:
                dyn_grp = f['dynamics']
                session_data['dynamics']['n_blocks'] = dyn_grp.attrs.get('n_blocks', 0)
                session_data['dynamics']['trials_per_block'] = dyn_grp.attrs.get('trials_per_block', 30)
                
                if 'landmark_positions' in dyn_grp:
                    session_data['dynamics']['landmark_positions'] = dyn_grp['landmark_positions'][:]
                
                session_data['dynamics']['preference_by_block'] = {}
                for key in dyn_grp.keys():
                    if key != 'landmark_positions' and key.endswith('_preference_by_block'):
                        # Extract and restore layer name
                        h5_layer_name = key.replace('_preference_by_block', '')
                        
                        # Restore original name (L2_3 → L2/3)
                        if h5_layer_name.startswith('L') and '_' in h5_layer_name:
                            parts = h5_layer_name.split('_')
                            if len(parts) == 2 and parts[1].isdigit():
                                original_layer_name = f"{parts[0]}/{parts[1]}"
                            else:
                                original_layer_name = h5_layer_name
                        else:
                            original_layer_name = h5_layer_name
                        
                        session_data['dynamics']['preference_by_block'][original_layer_name] = dyn_grp[key][:]
        
        return session_data
        
    except Exception as e:
        raise Exception(f"Error loading {os.path.basename(h5_path)}: {str(e)}")


def load_multiple_sessions(data_dir, pattern="*landmark_preferences.h5", recursive=False):
    """
    Load landmark preference data from multiple sessions.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing session HDF5 files
    pattern : str
        Glob pattern to match session files
    recursive : bool
        If True, search recursively through subdirectories
    
    Returns:
    --------
    sessions : list of dict
        List of session data dictionaries
    """
    
    # Handle recursive search
    if '**' in pattern or recursive:
        search_pattern = os.path.join(data_dir, pattern)
        h5_files = sorted(glob(search_pattern, recursive=True))  # FIXED: Added recursive=True
    else:
        search_pattern = os.path.join(data_dir, pattern)
        h5_files = sorted(glob(search_pattern))
    
    if len(h5_files) == 0:
        # Try to provide helpful error message
        print(f"\nERROR: No files found matching pattern: {search_pattern}")
        print(f"\nSearching in: {data_dir}")
        print(f"Pattern: {pattern}")
        print(f"Recursive: {recursive or ('**' in pattern)}")
        
        # Check if directory exists
        if not os.path.exists(data_dir):
            print(f"\nDirectory does not exist: {data_dir}")
        else:
            # List what files ARE in the directory
            print(f"\nDirectory exists. Looking for .h5 files...")
            all_h5 = glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
            if len(all_h5) > 0:
                print(f"Found {len(all_h5)} .h5 files:")
                for f in all_h5[:10]:  # Show first 10
                    print(f"  - {f}")
                if len(all_h5) > 10:
                    print(f"  ... and {len(all_h5) - 10} more")
            else:
                print("No .h5 files found in this directory or subdirectories")
        
        raise ValueError(f"No files found matching pattern: {search_pattern}")
    
    # print(f"Found {len(h5_files)} session files:")
    # for f in h5_files:
    #     print(f"  - {os.path.basename(f)}")
    
    # Load each session
    sessions = []
    for h5_path in h5_files:
        try:
            session_data = load_session_data(h5_path)
            sessions.append(session_data)
            # print(f"Loaded: {session_data['session_id']}")
        except Exception as e:
            print(f"Error loading {h5_path}: {e}")
    
    return sessions


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================

def compare_sessions_by_layer(sessions, layer_name='L2/3'):
    """
    Compare landmark preferences across sessions for a specific layer.
    
    Parameters:
    -----------
    sessions : list of dict
        List of session data
    layer_name : str
        Layer to analyze
    
    Returns:
    --------
    comparison : dict
        Comparison results
    """
    
    n_sessions = len(sessions)
    
    # Check if layer exists in all sessions
    valid_sessions = []
    for session in sessions:
        if layer_name in session['full_session']:
            valid_sessions.append(session)
    
    if len(valid_sessions) == 0:
        print(f"No data for {layer_name} in any session")
        return None
    
    # print(f"\nComparing {layer_name} across {len(valid_sessions)} sessions:")
    
    # Extract proportions from each session
    n_landmarks = len(valid_sessions[0]['full_session'][layer_name]['landmark_proportions'])
    proportions_matrix = np.zeros((len(valid_sessions), n_landmarks))
    counts_matrix = np.zeros((len(valid_sessions), n_landmarks), dtype=int)
    session_ids = []
    
    for i, session in enumerate(valid_sessions):
        layer_data = session['full_session'][layer_name]
        proportions_matrix[i, :] = layer_data['landmark_proportions']
        counts_matrix[i, :] = layer_data['landmark_counts']
        session_ids.append(session['session_id'])
        
        # print(f"  {session['session_id']}: {layer_data['n_cells']} cells")
    
    # Calculate statistics
    mean_proportions = np.mean(proportions_matrix, axis=0)
    std_proportions = np.std(proportions_matrix, axis=0)
    sem_proportions = stats.sem(proportions_matrix, axis=0)
    
    comparison = {
        'layer_name': layer_name,
        'session_ids': session_ids,
        'proportions_matrix': proportions_matrix,
        'counts_matrix': counts_matrix,
        'mean_proportions': mean_proportions,
        'std_proportions': std_proportions,
        'sem_proportions': sem_proportions,
        'n_landmarks': n_landmarks
    }
    
    return comparison


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_across_sessions_comparison(sessions, landmark_positions=[30, 60, 90, 120],
                                   title="Landmark Preferences Across Sessions", save_path=None):
    """
    Create multi-panel plot comparing landmark preferences across sessions.
    Each panel shows one layer.
    """
    
    # Get all unique layers
    all_layers = set()
    for session in sessions:
        all_layers.update(session['full_session'].keys())
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    valid_layers = [layer for layer in layer_order if layer in all_layers]
    
    if len(valid_layers) == 0:
        print("No valid layers found")
        return None
    
    # Create figure
    n_layers = len(valid_layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers))
    
    if n_layers == 1:
        axes = [axes]
    
    # Color scheme
    n_landmarks = len(landmark_positions)
    colors = plt.cm.Set1(np.linspace(0, 1, n_landmarks))
    
    for ax_idx, layer_name in enumerate(valid_layers):
        ax = axes[ax_idx]
        
        # Get comparison data for this layer
        comparison = compare_sessions_by_layer(sessions, layer_name)
        
        if comparison is None:
            ax.text(0.5, 0.5, f'{layer_name}: No data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        session_ids = comparison['session_ids']
        proportions_matrix = comparison['proportions_matrix']
        n_sessions = len(session_ids)
        
        # Create stacked bar plot
        x_pos = np.arange(n_sessions)
        bar_width = 0.6
        
        bottom = np.zeros(n_sessions)
        for lm_idx in range(n_landmarks):
            ax.bar(x_pos, proportions_matrix[:, lm_idx], bar_width,
                  bottom=bottom, color=colors[lm_idx],
                  label=f"Landmark {lm_idx+1} ({landmark_positions[lm_idx]:.0f}cm)",
                  edgecolor='black', linewidth=1)
            bottom += proportions_matrix[:, lm_idx]
        
        # Formatting
        ax.set_ylabel('Proportion of Cells', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(session_ids, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Session', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'landmark_preferences_across_sessions.png'), dpi=300)
    plt.close(fig)
    return fig


def plot_landmark_trends_across_sessions(sessions, landmark_positions=[30, 60, 90, 120], save_path=None):
    """
    Plot how preference for each landmark changes across sessions.
    Separate line for each landmark, separate panel for each layer.
    """
    
    # Get all unique layers
    all_layers = set()
    for session in sessions:
        all_layers.update(session['full_session'].keys())
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    valid_layers = [layer for layer in layer_order if layer in all_layers]
    
    if len(valid_layers) == 0:
        print("No valid layers found")
        return None
    
    # Create figure
    n_layers = len(valid_layers)
    fig, axes = plt.subplots(n_layers, 1, figsize=(12, 4*n_layers), sharex=True)
    
    if n_layers == 1:
        axes = [axes]
    
    # Color scheme
    n_landmarks = len(landmark_positions)
    colors = plt.cm.Set1(np.linspace(0, 1, n_landmarks))
    
    for ax_idx, layer_name in enumerate(valid_layers):
        ax = axes[ax_idx]
        
        # Get comparison data
        comparison = compare_sessions_by_layer(sessions, layer_name)
        
        if comparison is None:
            ax.text(0.5, 0.5, f'{layer_name}: No data', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        session_ids = comparison['session_ids']
        proportions_matrix = comparison['proportions_matrix']
        n_sessions = len(session_ids)
        
        # Plot trend lines for each landmark
        x_pos = np.arange(n_sessions)
        for lm_idx in range(n_landmarks):
            ax.plot(x_pos, proportions_matrix[:, lm_idx],
                   marker='o', linewidth=2, markersize=8,
                   color=colors[lm_idx],
                   label=f"Landmark {lm_idx+1} ({landmark_positions[lm_idx]:.0f}cm)")
        
        # Formatting
        ax.set_ylabel('Proportion of Cells', fontsize=11)
        ax.set_title(f'{layer_name}', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Session', fontsize=12)
    axes[-1].set_xticks(np.arange(len(session_ids)))
    axes[-1].set_xticklabels(session_ids, rotation=45, ha='right')
    
    fig.suptitle('Landmark Preference Trends Across Sessions', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'landmark_preference_trends_across_sessions.png'), dpi=300)
    plt.close(fig)
    
    return fig


def plot_variability_across_sessions(sessions, landmark_positions=[30, 60, 90, 120], save_path=None):
    """
    Plot variability metrics: coefficient of variation for each landmark preference
    across sessions, by layer.
    """
    
    # Get all unique layers
    all_layers = set()
    for session in sessions:
        all_layers.update(session['full_session'].keys())
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    valid_layers = [layer for layer in layer_order if layer in all_layers]
    
    if len(valid_layers) == 0:
        print("No valid layers found")
        return None
    
    n_landmarks = len(landmark_positions)
    
    # Calculate CV for each layer and landmark
    cv_matrix = np.zeros((len(valid_layers), n_landmarks))
    
    for layer_idx, layer_name in enumerate(valid_layers):
        comparison = compare_sessions_by_layer(sessions, layer_name)
        
        if comparison is not None:
            proportions = comparison['proportions_matrix']
            mean_props = np.mean(proportions, axis=0)
            std_props = np.std(proportions, axis=0)
            
            # Coefficient of variation (handle division by zero)
            cv = np.where(mean_props > 0, std_props / mean_props, 0)
            cv_matrix[layer_idx, :] = cv
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_landmarks)
    bar_width = 0.2
    
    colors_layer = ['#1E88E5', '#FF9800', '#4CAF50', '#E53935']
    
    for layer_idx, layer_name in enumerate(valid_layers):
        offset = (layer_idx - len(valid_layers)/2 + 0.5) * bar_width
        ax.bar(x + offset, cv_matrix[layer_idx, :], bar_width,
              label=layer_name, color=colors_layer[layer_idx],
              edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Landmark', fontsize=12)
    ax.set_ylabel('Coefficient of Variation (CV)', fontsize=12)
    ax.set_title('Variability in Landmark Preferences Across Sessions', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{i+1}\n({landmark_positions[i]:.0f}cm)" 
                        for i in range(n_landmarks)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'landmark_preference_variability_across_sessions.png'), dpi=300)
    plt.close(fig)
    
    return fig

def plot_heatmap_across_sessions(sessions, landmark_positions=[25, 55, 85, 115],
                                 save_path=None, animal_id=None):
    """
    Create heatmap showing landmark preferences by layer across sessions.
    Similar to your across-animals heatmap but for single animal across days.
    
    Layout: One heatmap per session, showing layer x landmark
    """
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    n_landmarks = len(landmark_positions)
    n_sessions = len(sessions)
    
    # Sort sessions by day number
    def extract_day_number(session_id):
        import re
        match = re.search(r'Day(\d+)', session_id)
        return int(match.group(1)) if match else 0
    
    sessions_sorted = sorted(sessions, key=lambda s: extract_day_number(s['session_id']))
    
    # Create figure
    n_cols = min(4, n_sessions)
    n_rows = int(np.ceil(n_sessions / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for sess_idx, session in enumerate(sessions_sorted):
        ax = axes[sess_idx]
        session_id = session['session_id']
        
        # Build heatmap data for this session
        heatmap_data = np.full((len(layer_order), n_landmarks), np.nan)
        count_data = np.zeros((len(layer_order), n_landmarks), dtype=int)
        
        for layer_idx, layer_name in enumerate(layer_order):
            if layer_name in session['full_session']:
                layer_data = session['full_session'][layer_name]
                heatmap_data[layer_idx, :] = layer_data['landmark_proportions']
                count_data[layer_idx, :] = layer_data['landmark_counts']
        
        # Plot heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        
        # Add text annotations
        for i in range(len(layer_order)):
            for j in range(n_landmarks):
                if not np.isnan(heatmap_data[i, j]):
                    text_str = f'{heatmap_data[i, j]:.2f}\n(n={count_data[i, j]})'
                    ax.text(j, i, text_str, ha='center', va='center', 
                           fontsize=8, color='black')
                else:
                    ax.text(j, i, 'N/A', ha='center', va='center', fontsize=8)
        
        # Labels
        ax.set_xticks(np.arange(n_landmarks))
        ax.set_yticks(np.arange(len(layer_order)))
        ax.set_xticklabels([f'L{i+1}\n({landmark_positions[i]}cm)' for i in range(n_landmarks)], fontsize=9)
        ax.set_yticklabels(layer_order, fontsize=10)
        ax.set_xlabel('Landmark', fontsize=10)
        ax.set_ylabel('Layer', fontsize=10)
        ax.set_title(f'{session_id}', fontsize=11, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_sessions, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Proportion of Cells')
    
    title = f'Landmark Preferences Across Sessions'
    if animal_id:
        title = f'{animal_id}: {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    
    if save_path:
        fig_path = os.path.join(save_path, 'heatmap_across_sessions.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(fig_path)}")
    
    return fig


def plot_layer_specific_evolution(sessions, landmark_positions=[25, 55, 85, 115],
                                  save_path=None, animal_id=None):
    """
    For each layer, show how L1 vs L4 preference ratio changes across sessions.
    This highlights the key biological question: does L4 preference increase with experience?
    """
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    n_landmarks = len(landmark_positions)
    
    # Sort sessions by day
    def extract_day_number(session_id):
        import re
        match = re.search(r'Day(\d+)', session_id)
        return int(match.group(1)) if match else 0
    
    sessions_sorted = sorted(sessions, key=lambda s: extract_day_number(s['session_id']))
    session_ids = [s['session_id'] for s in sessions_sorted]
    day_numbers = [extract_day_number(s) for s in session_ids]
    n_sessions = len(sessions_sorted)
    
    # Create figure: 2 rows
    # Row 1: Line plots for each layer showing all landmark proportions
    # Row 2: L1/L4 ratio evolution for each layer
    fig, axes = plt.subplots(2, len(layer_order), figsize=(4*len(layer_order), 8))
    
    colors = plt.cm.Set1(np.linspace(0, 1, n_landmarks))
    
    for layer_idx, layer_name in enumerate(layer_order):
        ax_top = axes[0, layer_idx]
        ax_bottom = axes[1, layer_idx]
        
        # Collect data across sessions
        proportions = np.full((n_sessions, n_landmarks), np.nan)
        
        for sess_idx, session in enumerate(sessions_sorted):
            if layer_name in session['full_session']:
                proportions[sess_idx, :] = session['full_session'][layer_name]['landmark_proportions']
        
        # Top panel: All landmarks
        for lm_idx in range(n_landmarks):
            valid_mask = ~np.isnan(proportions[:, lm_idx])
            if np.any(valid_mask):
                ax_top.plot(np.array(day_numbers)[valid_mask], 
                           proportions[valid_mask, lm_idx],
                           marker='o', linewidth=2, markersize=8,
                           color=colors[lm_idx],
                           label=f'L{lm_idx+1} ({landmark_positions[lm_idx]}cm)')
        
        ax_top.set_xlabel('Day', fontsize=10)
        ax_top.set_ylabel('Proportion', fontsize=10)
        ax_top.set_title(f'{layer_name}', fontsize=11, fontweight='bold')
        ax_top.set_ylim(0, 1)
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(fontsize=7, loc='best')
        ax_top.set_xticks(day_numbers)
        
        # Bottom panel: L1/L4 ratio (or L1 - L4 difference)
        l1_props = proportions[:, 0]  # First landmark
        l4_props = proportions[:, -1]  # Last landmark
        
        # Calculate difference (positive = L1 dominant, negative = L4 dominant)
        diff = l1_props - l4_props
        
        valid_mask = ~np.isnan(diff)
        if np.any(valid_mask):
            ax_bottom.plot(np.array(day_numbers)[valid_mask], diff[valid_mask],
                          marker='s', linewidth=2, markersize=10, color='purple')
            ax_bottom.axhline(0, color='gray', linestyle='--', linewidth=1)
            
            # Fill regions
            ax_bottom.fill_between(np.array(day_numbers)[valid_mask], 
                                   0, diff[valid_mask],
                                   where=diff[valid_mask] > 0, 
                                   alpha=0.3, color='red', label='L1 dominant')
            ax_bottom.fill_between(np.array(day_numbers)[valid_mask], 
                                   0, diff[valid_mask],
                                   where=diff[valid_mask] < 0, 
                                   alpha=0.3, color='blue', label='L4 dominant')
        
        ax_bottom.set_xlabel('Day', fontsize=10)
        ax_bottom.set_ylabel('L1 - L4 Proportion', fontsize=10)
        ax_bottom.set_title(f'{layer_name}: L1 vs L4 Balance', fontsize=10)
        ax_bottom.set_ylim(-0.6, 0.6)
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.legend(fontsize=7, loc='best')
        ax_bottom.set_xticks(day_numbers)
    
    title = 'Landmark Preference Evolution'
    if animal_id:
        title = f'{animal_id}: {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig_path = os.path.join(save_path, 'layer_specific_evolution.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(fig_path)}")
    
    return fig


def plot_learning_summary(sessions, landmark_positions=[25, 55, 85, 115],
                          save_path=None, animal_id=None):
    """
    Summary plot showing:
    1. Early vs Late session comparison (first vs last session)
    2. Statistical test for change in L1/L4 balance
    """
    
    layer_order = ['L2/3', 'L4', 'L5', 'L6']
    n_landmarks = len(landmark_positions)
    
    # Sort sessions
    def extract_day_number(session_id):
        import re
        match = re.search(r'Day(\d+)', session_id)
        return int(match.group(1)) if match else 0
    
    sessions_sorted = sorted(sessions, key=lambda s: extract_day_number(s['session_id']))
    
    if len(sessions_sorted) < 2:
        print("Need at least 2 sessions for learning summary")
        return None
    
    early_session = sessions_sorted[0]
    late_session = sessions_sorted[-1]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Early session heatmap
    ax1 = axes[0]
    early_data = np.full((len(layer_order), n_landmarks), np.nan)
    for layer_idx, layer_name in enumerate(layer_order):
        if layer_name in early_session['full_session']:
            early_data[layer_idx, :] = early_session['full_session'][layer_name]['landmark_proportions']
    
    im1 = ax1.imshow(early_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(np.arange(n_landmarks))
    ax1.set_yticks(np.arange(len(layer_order)))
    ax1.set_xticklabels([f'L{i+1}' for i in range(n_landmarks)])
    ax1.set_yticklabels(layer_order)
    ax1.set_title(f"Early: {early_session['session_id']}", fontsize=12, fontweight='bold')
    ax1.set_xlabel('Landmark')
    ax1.set_ylabel('Layer')
    
    # Add values
    for i in range(len(layer_order)):
        for j in range(n_landmarks):
            if not np.isnan(early_data[i, j]):
                ax1.text(j, i, f'{early_data[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    # Panel 2: Late session heatmap
    ax2 = axes[1]
    late_data = np.full((len(layer_order), n_landmarks), np.nan)
    for layer_idx, layer_name in enumerate(layer_order):
        if layer_name in late_session['full_session']:
            late_data[layer_idx, :] = late_session['full_session'][layer_name]['landmark_proportions']
    
    im2 = ax2.imshow(late_data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(np.arange(n_landmarks))
    ax2.set_yticks(np.arange(len(layer_order)))
    ax2.set_xticklabels([f'L{i+1}' for i in range(n_landmarks)])
    ax2.set_yticklabels(layer_order)
    ax2.set_title(f"Late: {late_session['session_id']}", fontsize=12, fontweight='bold')
    ax2.set_xlabel('Landmark')
    ax2.set_ylabel('Layer')
    
    for i in range(len(layer_order)):
        for j in range(n_landmarks):
            if not np.isnan(late_data[i, j]):
                ax2.text(j, i, f'{late_data[i, j]:.2f}', ha='center', va='center', fontsize=9)
    
    # Panel 3: Change (Late - Early)
    ax3 = axes[2]
    change_data = late_data - early_data
    
    # Use diverging colormap for change
    max_abs = np.nanmax(np.abs(change_data))
    im3 = ax3.imshow(change_data, cmap='RdBu_r', aspect='auto', 
                     vmin=-max_abs, vmax=max_abs)
    ax3.set_xticks(np.arange(n_landmarks))
    ax3.set_yticks(np.arange(len(layer_order)))
    ax3.set_xticklabels([f'L{i+1}' for i in range(n_landmarks)])
    ax3.set_yticklabels(layer_order)
    ax3.set_title('Change (Late - Early)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Landmark')
    ax3.set_ylabel('Layer')
    
    for i in range(len(layer_order)):
        for j in range(n_landmarks):
            if not np.isnan(change_data[i, j]):
                color = 'white' if abs(change_data[i, j]) > max_abs * 0.5 else 'black'
                ax3.text(j, i, f'{change_data[i, j]:+.2f}', ha='center', va='center', 
                        fontsize=9, color=color)
    
    plt.colorbar(im3, ax=ax3, label='Δ Proportion')
    
    title = f'Learning Summary: {early_session["session_id"]} → {late_session["session_id"]}'
    if animal_id:
        title = f'{animal_id}: {title}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig_path = os.path.join(save_path, 'learning_summary.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {os.path.basename(fig_path)}")
    
    return fig

# ============================================================================
# MAIN COMPARISON WORKFLOW
# ============================================================================

def run_across_session_comparison(data_dir, pattern="**/*landmark_preferences.h5",
                                 landmark_positions=[25, 55, 85, 115], 
                                 recursive=True, save_path=None, animal_id=None):
    """
    Complete workflow for comparing landmark preferences across sessions.
    """
    
    print("\n" + "="*70)
    print("ACROSS-SESSION LANDMARK PREFERENCE COMPARISON")
    print("="*70)
    
    # Load all sessions
    sessions = load_multiple_sessions(data_dir, pattern, recursive=recursive)
    
    if len(sessions) < 2:
        print("ERROR: Need at least 2 sessions for comparison")
        return None
    
    print(f"\nLoaded {len(sessions)} sessions successfully")
    
    # Extract animal_id from path if not provided
    if animal_id is None:
        import re
        match = re.search(r'(JSY\d+)', data_dir)
        if match:
            animal_id = match.group(1)
    
    # Set save path
    if save_path is None:
        save_path = data_dir
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Original plots
    fig_stacked = plot_across_sessions_comparison(
        sessions, landmark_positions,
        title=f"{animal_id}: Landmark Preferences Across Sessions" if animal_id else "Landmark Preferences Across Sessions",
        save_path=save_path
    )
    
    fig_trends = plot_landmark_trends_across_sessions(
        sessions, landmark_positions, save_path=save_path
    )
    
    fig_variability = plot_variability_across_sessions(
        sessions, landmark_positions, save_path=save_path
    )
    
    # NEW: Improved plots
    fig_heatmaps = plot_heatmap_across_sessions(
        sessions, landmark_positions, save_path=save_path, animal_id=animal_id
    )
    
    fig_evolution = plot_layer_specific_evolution(
        sessions, landmark_positions, save_path=save_path, animal_id=animal_id
    )
    
    fig_learning = plot_learning_summary(
        sessions, landmark_positions, save_path=save_path, animal_id=animal_id
    )
    
    # plt.show()
    
    # Compile results
    results = {
        'sessions': sessions,
        'n_sessions': len(sessions),
        'animal_id': animal_id,
        'figures': {
            'stacked': fig_stacked,
            'trends': fig_trends,
            'variability': fig_variability,
            'heatmaps': fig_heatmaps,
            'evolution': fig_evolution,
            'learning': fig_learning
        }
    }
    
    print("\n" + "="*70)
    print("ACROSS-SESSION COMPARISON COMPLETE!")
    print("="*70)
    
    return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("ACROSS-SESSION LANDMARK PREFERENCE COMPARISON")
    print("="*70)
    
    # ========================================================================
    # CONFIGURE YOUR DATA PATH HERE
    # ========================================================================
    
    # Your data directory
    data_dir = r"D:\V1_SpatialModulation\2p\V1_prism\JSY044_ChronicImaging"
    
    # Pattern options:
    # Option 1: All files directly in data_dir
    # pattern = "*landmark_preferences.h5"
    # recursive = False
    
    # Option 2: Files in immediate subdirectories
    # pattern = "*/*landmark_preferences.h5"
    # recursive = False
    
    # Option 3: Files anywhere in directory tree (RECOMMENDED)
    pattern = "**/*landmark_preferences.h5"
    recursive = True
    
    # ========================================================================
    # CHECK IF PATH EXISTS
    # ========================================================================
    
    if not os.path.exists(data_dir):
        print(f"\nERROR: Directory does not exist: {data_dir}")
        print("\nPlease update the 'data_dir' variable with your actual data directory.")
        sys.exit(1)
    
    # ========================================================================
    # FIRST: Check what files exist
    # ========================================================================
    
    print(f"\nSearching for files in: {data_dir}")
    print(f"Pattern: {pattern}")
    print(f"Recursive: {recursive}")
    
    # Manual check first
    all_h5_files = glob(os.path.join(data_dir, "**/*.h5"), recursive=True)
    landmark_files = [f for f in all_h5_files if 'landmark_preferences' in os.path.basename(f)]
    
    print(f"\nFound {len(all_h5_files)} total .h5 files")
    print(f"Found {len(landmark_files)} files with 'landmark_preferences' in name:")
    for f in landmark_files:
        print(f"  - {f}")
    
    if len(landmark_files) == 0:
        print("\n" + "="*70)
        print("NO LANDMARK PREFERENCE FILES FOUND!")
        print("="*70)
        print("\nMake sure you have run the landmark analysis for each session first.")
        print("The analysis should create files named like: 'Day1_landmark_preferences.h5'")
        print("\nRun test_landmark_preference_analysis.py for each session first.")
        sys.exit(1)
    
    # ========================================================================
    # RUN COMPARISON
    # ========================================================================
    
    try:
        results = run_across_session_comparison(
            data_dir=data_dir,
            pattern=pattern,
            landmark_positions=[25, 55, 85, 115],
            recursive=recursive,
            save_path=data_dir
        )
        
        if results is not None:
            print("\n" + "="*70)
            print("SUCCESS! Comparison complete.")
            print(f"Compared {results['n_sessions']} sessions.")
            print("="*70)
        
    except Exception as e:
        print(f"\n" + "="*70)
        print(f"ERROR: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()