import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_response_plot(normalized_spatial_activity, reliable_cells, clim=None):
    """
    Create an enhanced response plot with cells sorted by their peak locations,
    with improved contrast and visibility of the responses.
    
    Parameters:
    -----------
    normalized_spatial_activity : numpy.ndarray
        Already normalized activity matrix (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    clim : tuple or None
        Optional color limits (min, max) to enhance contrast. If None, auto-calculated.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with the enhanced sorted response plot
    sorted_reliable_indices : numpy.ndarray
        Indices of reliable cells sorted by peak location
    """
    # Get dimensions
    n_cells, n_trials, n_bins = normalized_spatial_activity.shape
    
    # Step 1: Select only reliable cells
    reliable_indices = np.where(reliable_cells)[0]
    reliable_activity = normalized_spatial_activity[reliable_indices]
    
    # Step 2: Split trials into even and odd
    even_trials = np.arange(0, n_trials, 2)
    odd_trials = np.arange(1, n_trials, 2)
    
    # Step 3: Calculate average responses for each set of trials
    even_avg = np.mean(reliable_activity[:, even_trials, :], axis=1)
    odd_avg = np.mean(reliable_activity[:, odd_trials, :], axis=1)
    
    # Step 4: Enhance contrast in the trial-averaged data
    # Apply a non-linear transformation to enhance small differences
    # (using power function, which enhances high values more than low ones)
    enhanced_even_avg = np.power(even_avg, 1.5)  # Adjust power as needed
    
    # Step 5: Find peak location for each cell in the enhanced even trials
    peak_locations = np.argmax(enhanced_even_avg, axis=1)
    
    # Step 6: Sort the cell indices by their peak locations
    sorted_indices = np.argsort(peak_locations)
    
    # Step 7: Apply sorting to odd trials for display
    # sorted_even_activity = even_avg[sorted_indices]
    sorted_odd_activity = odd_avg[sorted_indices]
    
    # Step 8: Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create a more vibrant colormap with stronger contrast
    cmap = LinearSegmentedColormap.from_list('EnhancedBlues', 
                                           [(1,1,1), (0.8,0.8,1), (0.4,0.4,0.9), (0,0,0.8), (0,0,0.5)])
    
    # Auto-calculate color limits if not provided
    if clim is None:
        # Use percentiles instead of min/max for better contrast
        vmin = np.percentile(sorted_odd_activity, 5)  # 5th percentile as minimum
        vmax = np.percentile(sorted_odd_activity, 95)  # 95th percentile as maximum
        # Ensure we have a reasonable range
        if vmax - vmin < 0.1:
            vmin = 0
            vmax = 1
    else:
        vmin, vmax = clim
    
    # Plot the sorted odd trials with enhanced contrast
    im = ax.imshow(sorted_odd_activity, aspect='auto', cmap=cmap, 
                  interpolation='nearest', vmin=vmin, vmax=vmax)    
    # im = ax.imshow(sorted_odd_activity, aspect='auto', cmap=cmap, 
    #               interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Activity')
    
    # Add labels and title
    ax.set_xlabel('Spatial Bin')
    ax.set_ylabel('Cell Number (sorted by peak location)')
    ax.set_title('Enhanced Spatial Responses - Sorted by Peak Location in Even Trials\n'
                f'(Displaying Odd Trials for {len(reliable_indices)} Reliable Cells)')
    
    # Return the figure and sorted indices of reliable cells
    sorted_reliable_indices = reliable_indices[sorted_indices]
    
    return fig, sorted_reliable_indices

def create_waterfall_plot(normalized_spatial_activity, reliable_cells, n_cells_to_show=None):
    """
    Create a waterfall plot of cell responses, sorted by peak location.
    This provides an alternative visualization that can make weak responses more visible.
    
    Parameters:
    -----------
    normalized_spatial_activity : numpy.ndarray
        Already normalized activity matrix (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    n_cells_to_show : int or None
        Number of cells to display. If None, all reliable cells are shown.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure with the waterfall plot
    """
    # Get reliable cells
    reliable_indices = np.where(reliable_cells)[0]
    
    # Limit number of cells if needed
    if n_cells_to_show is not None:
        n_cells_to_show = min(n_cells_to_show, len(reliable_indices))
        reliable_indices = reliable_indices[:n_cells_to_show]
    
    # Get average activity across all trials
    all_trials_avg = np.mean(normalized_spatial_activity, axis=1)
    reliable_avg = all_trials_avg[reliable_indices]
    
    # Find peak locations and sort
    peak_locations = np.argmax(reliable_avg, axis=1)
    sort_order = np.argsort(peak_locations)
    
    # Apply sorting
    sorted_reliable_avg = reliable_avg[sort_order]
    
    # Create figure for waterfall plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Determine spacing based on number of cells
    spacing = 3.0 / max(1, len(reliable_indices) / 20)  # Adaptive spacing
    
    # Plot each trace with vertical offset
    for i, activity in enumerate(sorted_reliable_avg):
        # Scale the activity to enhance visibility
        scaled_activity = activity * 0.9 * spacing
        ax.plot(scaled_activity + i * spacing, 'b-', linewidth=1.2)
        
        # Add shading under the curve for better visibility
        ax.fill_between(range(len(activity)), 
                       i * spacing, 
                       scaled_activity + i * spacing, 
                       alpha=0.2, color='blue')
    
    ax.set_xlabel('Spatial Bin')
    ax.set_ylabel('Cell Number (sorted by peak location)')
    ax.set_title('Waterfall Plot of Spatial Responses\n'
                f'(Showing {len(reliable_indices)} Reliable Cells)')
    ax.set_yticks([])
    
    return fig

