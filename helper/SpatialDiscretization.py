import numpy as np

def spatial_assignment(lapnum, spks_laps, location_laps, single_lap_treadmill):
    """
    Perform spatial discretization of neural activity data with variable-length laps.
    
    Parameters:
    -----------
    spks_laps : list
        List of numpy arrays containing spike data for each lap (each array is cells x lap_length)
    location_laps : list
        List of numpy arrays containing location data for each lap (each array is lap_length)
    single_lap_treadmill : float
        Length of a single lap
        
    Returns:
    --------
    spatial_activity : numpy.ndarray
        Activity in each spatial bin for each cell and lap (cells x laps x bins)
    spatial_bins : numpy.ndarray
        Edges of spatial bins
    trial_averaged_activity : numpy.ndarray
        Activity averaged across trials for each cell and spatial bin
    bin_centers : numpy.ndarray
        Centers of spatial bins
    """
    # Get dimensions
    n_cells = spks_laps[0].shape[0]
    n_laps = lapnum
    
    # Define number of bins
    n_bins = round(single_lap_treadmill)
    print(f"Using {n_bins} spatial bins")
    
    # Create spatial bins based on the range of all location data
    location_min = min(np.min(loc) for loc in location_laps)
    location_max = max(np.max(loc) for loc in location_laps)
    spatial_bins = np.linspace(location_min, location_max, n_bins + 1)
    bin_centers = (spatial_bins[:-1] + spatial_bins[1:]) / 2
    
    # Initialize arrays
    spatial_activity = np.zeros((n_cells, n_laps, n_bins))
    trial_averaged_activity = np.zeros((n_cells, n_bins))
    
    # Process each lap
    for lap_idx, (lap_spks, lap_location) in enumerate(zip(spks_laps, location_laps)):
        # Assign timepoints to spatial bins
        bin_indices = np.clip(np.digitize(lap_location, spatial_bins) - 1, 0, n_bins-1)
        
        # Calculate activity for each bin in this lap
        for bin_idx in range(n_bins):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                spatial_activity[:, lap_idx, bin_idx] = np.sum(lap_spks[:, bin_mask], axis=1)
                # Normalize by number of timepoints in this bin
                spatial_activity[:, lap_idx, bin_idx] /= np.sum(bin_mask)
    
    # Calculate trial average
    trial_averaged_activity = np.mean(spatial_activity, axis=1)
    
    print("Completed spatial discretization")
    return spatial_activity, spatial_bins, trial_averaged_activity, bin_centers

# Example usage and plotting:
"""
# First, reshape the data into laps
spks_laps, location_laps, n_laps = reshape_into_laps(
    new_spks, 
    new_location,
    high_percentile=90, 
    low_percentile=10
)

# Then perform spatial assignment
spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = spatial_assignment(
    spks_laps,
    location_laps,
    single_lap_treadmill
)

# Plot results for an example cell
cell_idx = 0  # Change this to plot different cells
plt.figure(figsize=(15, 5))

# Plot spatial activity for each lap
plt.subplot(121)
for lap in range(spatial_activity.shape[1]):
    plt.plot(bin_centers, spatial_activity[cell_idx, lap, :], alpha=0.5, label=f'Lap {lap+1}')
plt.title(f'Spatial Activity by Lap - Cell {cell_idx}')
plt.xlabel('Position')
plt.ylabel('Activity')
plt.legend()

# Plot trial-averaged activity
plt.subplot(122)
plt.plot(bin_centers, trial_averaged_activity[cell_idx])
plt.title(f'Trial-Averaged Activity - Cell {cell_idx}')
plt.xlabel('Position')
plt.ylabel('Average Activity')

plt.tight_layout()
plt.show()
"""