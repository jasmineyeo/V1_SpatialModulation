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
    # print(f"Using {n_bins} spatial bins")
    
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

def spatial_assignment_with_physical_units(lapnum, spks_laps, location_laps_cm, physical_lap_length_cm=130):
    """
    Perform spatial discretization of neural activity data with variable-length laps.
    Now uses physical distance units throughout with FIXED spatial bins.
    
    Parameters:
    -----------
    lapnum : int
        Number of laps
    spks_laps : list
        List of numpy arrays containing spike data for each lap (each array is cells x lap_length)
    location_laps_cm : list
        List of numpy arrays containing location data for each lap in cm (each array is lap_length)
    physical_lap_length_cm : float
        Expected physical length of a single lap in cm (default: 130)
        
    Returns:
    --------
    spatial_activity : numpy.ndarray
        Activity in each spatial bin for each cell and lap (cells x laps x bins)
    spatial_bins : numpy.ndarray
        Edges of spatial bins in cm
    trial_averaged_activity : numpy.ndarray
        Activity averaged across trials for each cell and spatial bin
    bin_centers : numpy.ndarray
        Centers of spatial bins in cm
    """
    # Get dimensions
    n_cells = spks_laps[0].shape[0]
    n_laps = lapnum
    
    # Define number of bins (1cm per bin, approximately)
    n_bins = round(physical_lap_length_cm)
    print(f"Using {n_bins} spatial bins (1 cm per bin)")
    
    # FIXED: Create spatial bins based on KNOWN track length, not actual data range
    location_min = 0.0  # Track always starts at 0
    location_max = physical_lap_length_cm  # Track always ends at physical_lap_length_cm
    
    # Check actual data range for diagnostics
    actual_min = min(np.min(loc) for loc in location_laps_cm)
    actual_max = max(np.max(loc) for loc in location_laps_cm)
    
    print(f"Fixed spatial bins: {location_min:.2f} to {location_max:.2f} cm")
    print(f"Actual data range: {actual_min:.2f} to {actual_max:.2f} cm")
    
    if actual_max < location_max * 0.95:
        print(f"⚠️  WARNING: Animals only reached {actual_max:.2f} cm ({actual_max/location_max*100:.1f}% of track)")
    
    # Create bins spanning the full track
    spatial_bins = np.linspace(location_min, location_max, n_bins + 1)
    bin_centers = (spatial_bins[:-1] + spatial_bins[1:]) / 2
    
    # Initialize arrays
    spatial_activity = np.zeros((n_cells, n_laps, n_bins))
    trial_averaged_activity = np.zeros((n_cells, n_bins))
    
    # Process each lap
    for lap_idx, (lap_spks, lap_location_cm) in enumerate(zip(spks_laps, location_laps_cm)):
        # Assign timepoints to spatial bins
        # FIXED: Clip to valid range before digitizing
        lap_location_cm_clipped = np.clip(lap_location_cm, location_min, location_max)
        bin_indices = np.clip(np.digitize(lap_location_cm_clipped, spatial_bins) - 1, 0, n_bins-1)
        
        # Calculate activity for each bin in this lap
        for bin_idx in range(n_bins):
            bin_mask = (bin_indices == bin_idx)
            if np.any(bin_mask):
                spatial_activity[:, lap_idx, bin_idx] = np.sum(lap_spks[:, bin_mask], axis=1)
                # Normalize by number of timepoints in this bin
                spatial_activity[:, lap_idx, bin_idx] /= np.sum(bin_mask)
    
    # Calculate trial average
    trial_averaged_activity = np.mean(spatial_activity, axis=1)
    
    print("Completed spatial discretization with physical units")
    print(f"Bin size: {(spatial_bins[1] - spatial_bins[0]):.2f} cm")
    
    return spatial_activity, spatial_bins, trial_averaged_activity, bin_centers