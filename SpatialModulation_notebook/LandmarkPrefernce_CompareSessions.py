"""
LandmarkPrefernce_CompareSessions.py
Script for comparing landmark preferences across multiple recording sessions

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
            
            print(f"  Loading {os.path.basename(h5_path)}:")
            
            for h5_layer_name in full_grp.keys():
                layer_grp = full_grp[h5_layer_name]
                
                # Restore original layer name
                original_layer_name = restore_layer_name(h5_layer_name, layer_grp)
                print(f"    Layer: '{h5_layer_name}' → '{original_layer_name}'")
                
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
    
    print(f"Found {len(h5_files)} session files:")
    for f in h5_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load each session
    sessions = []
    for h5_path in h5_files:
        try:
            session_data = load_session_data(h5_path)
            sessions.append(session_data)
            print(f"Loaded: {session_data['session_id']}")
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
    
    print(f"\nComparing {layer_name} across {len(valid_sessions)} sessions:")
    
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
        
        print(f"  {session['session_id']}: {layer_data['n_cells']} cells")
    
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


# ============================================================================
# MAIN COMPARISON WORKFLOW
# ============================================================================

def run_across_session_comparison(data_dir, pattern="*landmark_preferences.h5",
                                 landmark_positions=[30, 60, 90, 120], recursive=True, save_path = None):
    """
    Complete workflow for comparing landmark preferences across sessions.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing session HDF5 files (can contain wildcards or subdirs)
    pattern : str
        Glob pattern to match session files
    landmark_positions : list
        Positions of landmarks in cm
    
    Returns:
    --------
    results : dict
        Comparison results and figures
    """
    
    print("\n" + "="*70)
    print("ACROSS-SESSION LANDMARK PREFERENCE COMPARISON")
    print("="*70)
    
    # Load all sessions
    print("\nLoading session data...")
    sessions = load_multiple_sessions(data_dir, pattern, recursive=recursive)  # UPDATED
    
    if len(sessions) < 2:
        print("ERROR: Need at least 2 sessions for comparison")
        return None
    
    print(f"\nLoaded {len(sessions)} sessions successfully")
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    fig_stacked = plot_across_sessions_comparison(
        sessions, landmark_positions,
        title="Landmark Preferences Across Sessions (Stacked)", save_path=save_path
    )
    
    fig_trends = plot_landmark_trends_across_sessions(
        sessions, landmark_positions, save_path=save_path
    )
    
    fig_variability = plot_variability_across_sessions(
        sessions, landmark_positions, save_path=save_path
    )
    
    # plt.show()
    
    # Compile results
    results = {
        'sessions': sessions,
        'n_sessions': len(sessions),
        'figures': {
            'stacked': fig_stacked,
            'trends': fig_trends,
            'variability': fig_variability
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
    data_dir = r"D:\V1_SpatialModulation\2p\V1_prism\JSY052_ChrnoicImaging"
    
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
            landmark_positions=[30, 60, 90, 120],
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