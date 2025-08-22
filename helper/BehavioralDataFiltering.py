import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter


def calculate_vr_speed_and_distance(location_data, framerate):
    """
    Calculate running speed in cm/s and convert location to physical distance.
    
    Parameters:
    -----------
    location_data : numpy.ndarray
        VR location data in arbitrary units
    framerate : float
        Recording framerate in Hz
        
    Returns:
    --------
    speed_cm_s : numpy.ndarray
        Running speed in cm/s
    location_cm : numpy.ndarray
        Location converted to physical distance in cm
    conversion_factor : float
        Conversion factor used (cm per AU)
    """
    # Calculate VR range for this specific dataset
    vr_range = np.max(location_data) - np.min(location_data)
    
    # Convert to physical distance: ~379 AU = 130 cm
    conversion_factor = 130.0 / vr_range  # cm per AU
    
    # Convert location to cm
    location_cm = location_data * conversion_factor
    
    # Calculate speed in cm/s
    speed_cm_s = np.abs(np.diff(location_cm)) * framerate
    
    # Pad to match original length (repeat first value)
    speed_cm_s = np.concatenate(([speed_cm_s[0]], speed_cm_s))
    
    return speed_cm_s, location_cm, conversion_factor

def reshape_into_laps(spks, location, high_percentile=90, low_percentile=10, plot_detection=True):
    """
    Reshape both spike data and location data into laps using percentile-based 
    thresholds for more robust lap detection. Preserves all data points in each lap.
    
    Parameters:
    -----------
    spks : numpy.ndarray
        Spike data (cells x time)
    location : numpy.ndarray
        Location data (time,)
    high_percentile : float, optional
        Percentile value to use for high threshold (default: 90)
    low_percentile : float, optional
        Percentile value to use for low threshold (default: 10)
    plot_detection : bool, optional
        Whether to plot the lap detection result (default: True)
        
    Returns:
    --------
    spks_laps : list
        List of numpy arrays containing spike data for each lap (each array is cells x lap_length)
    location_laps : list
        List of numpy arrays containing location data for each lap (each array is lap_length,)
    n_laps : int
        Number of laps detected
    """
    # Find thresholds based on percentiles instead of absolute values
    threshold_high = np.percentile(location, high_percentile)
    threshold_low = np.percentile(location, low_percentile)
    
    # print(f"Location range: {np.min(location):.2f} to {np.max(location):.2f}")
    # print(f"Using high threshold: {threshold_high:.2f} (at {high_percentile}th percentile)")
    # print(f"Using low threshold: {threshold_low:.2f} (at {low_percentile}th percentile)")
    
    # Find the first minimum in the data
    first_min_indices = np.where(location < threshold_low)[0]
    if len(first_min_indices) == 0:
        print("No location values below low threshold found!")
        return None, None, 0
    first_min_idx = first_min_indices[0]
    
    # Use a state machine approach to identify transitions
    lap_ends = []
    state = "low" if location[0] < threshold_low else "high"
    
    for i in range(1, len(location)):
        if state == "high" and location[i] < threshold_low:
            lap_ends.append(i)
            state = "low"
        elif state == "low" and location[i] > threshold_high:
            state = "high"
    
    lap_ends = np.array(lap_ends)
    
    if len(lap_ends) == 0:
        print("No lap transitions detected!")
        return None, None, 0
        
    # Add start and end indices
    lap_starts = np.concatenate(([first_min_idx], lap_ends))
    lap_ends = np.concatenate((lap_ends, [len(location)]))
    
    # Verify lap detection - ensure we have clear transitions
    valid_laps = []
    valid_starts = []
    valid_ends = []
    
    for i, (start, end) in enumerate(zip(lap_starts, lap_ends)):
        lap_loc = location[start:end]
        # Check if this lap has a clear high point
        if np.max(lap_loc) > threshold_high:
            valid_laps.append(i)
            valid_starts.append(start)
            valid_ends.append(end)
    
    if len(valid_laps) < len(lap_starts):
        print(f"Removed {len(lap_starts) - len(valid_laps)} incomplete laps")
        lap_starts = np.array(valid_starts)
        lap_ends = np.array(valid_ends)
    
    # Calculate lap lengths
    lap_lengths = lap_ends - lap_starts
    n_laps = len(lap_starts)
    
    # print(f"Found {n_laps} laps")
    # print(f"Lap lengths: {lap_lengths}")
    
    # Plot lap detection for verification
    if plot_detection:
        plt.figure(figsize=(15, 5))
        plt.plot(location, 'b-', alpha=0.5)
        
        # Plot lap transitions
        # For lap ends, use all except the last element (which is the end of data)
        if len(lap_ends) > 1:  # Make sure we have actual lap ends to plot
            plt.plot(lap_ends[:-1], location[lap_ends[:-1]], 'r.', markersize=10, label='Lap Ends')
        
        # For lap starts, use all elements
        plt.plot(lap_starts, location[lap_starts], 'g.', markersize=10, label='Lap Starts')
        
        # Add threshold lines
        plt.axhline(y=threshold_high, color='gray', linestyle='--', alpha=0.5, label='High Threshold')
        plt.axhline(y=threshold_low, color='gray', linestyle='--', alpha=0.5, label='Low Threshold')
        
        plt.title(f'Lap Detection (Found {n_laps} laps)')
        plt.xlabel('Time')
        plt.ylabel('Location')
        plt.legend()
        # plt.show()
    
    # Create lists to store data for each lap
    spks_laps = []
    location_laps = []
    
    # Fill in the data with full lap lengths
    for start, end in zip(lap_starts, lap_ends):
        location_laps.append(location[start:end])
        spks_laps.append(spks[:, start:end])
    
    return spks_laps, location_laps, n_laps

def process_data_with_speed_filtering(spks, location, 
                                    min_trial_duration_seconds=5, 
                                    max_trial_duration_seconds=30, 
                                    framerate=10, 
                                    min_speed_cm_s=2.0,
                                    frames_to_keep=5):
    """
    Process 2P and VR data with improved speed-based filtering:
    1. Split data into trials/laps
    2. Filter out trials that take too long or too short
    3. Remove periods with speed below threshold (stationary periods)
    
    Parameters:
    -----------
    spks : numpy.ndarray
        Spike data (cells x time)
    location : numpy.ndarray
        Location data (time,)
    min_trial_duration_seconds : float
        Minimum duration for a valid trial in seconds
    max_trial_duration_seconds : float
        Maximum duration for a valid trial in seconds
    framerate : float
        Frames per second of recording
    min_speed_cm_s : float
        Minimum speed threshold in cm/s (remove periods below this)
    frames_to_keep : int
        Number of frames to keep at the beginning and end of removed periods
        
    Returns:
    --------
    filtered_spks_laps : list
        List of filtered spike data for each valid lap
    filtered_location_laps : list
        List of filtered location data for each valid lap (in cm)
    n_valid_laps : int
        Number of valid laps after filtering
    """
    
    # Step 1: Calculate speed and convert to physical units
    print("Converting VR units to physical distance and calculating speed...")
    speed_cm_s, location_cm, conversion_factor = calculate_vr_speed_and_distance(location, framerate)
    
    print(f"VR conversion factor: {conversion_factor:.4f} cm per AU")
    print(f"Speed range: {np.min(speed_cm_s):.2f} to {np.max(speed_cm_s):.2f} cm/s")
    
    # Step 2: Reshape data into laps using original location (AU) for lap detection
    spks_laps, location_laps, n_laps = reshape_into_laps(spks, location, plot_detection=True)
    print("Number of detected laps:", n_laps)
    
    if n_laps == 0:
        print("No laps detected!")
        return None, None, 0
    
    # Step 3: Filter out trials by duration
    min_frames_per_trial = min_trial_duration_seconds * framerate
    max_frames_per_trial = max_trial_duration_seconds * framerate
    valid_trials = []
    
    print(f"\nFiltering trials by duration ({min_trial_duration_seconds}s to {max_trial_duration_seconds}s):")
    for i, (lap_spks, lap_loc) in enumerate(zip(spks_laps, location_laps)):
        trial_duration_frames = lap_spks.shape[1]
        trial_duration_seconds = trial_duration_frames / framerate
        
        if min_frames_per_trial <= trial_duration_frames <= max_frames_per_trial:
            valid_trials.append(i)
            print(f"  Trial {i+1}: {trial_duration_seconds:.2f} seconds - VALID")
        else:
            reason = "TOO SHORT" if trial_duration_frames < min_frames_per_trial else "TOO LONG"
            print(f"  Trial {i+1}: {trial_duration_seconds:.2f} seconds - {reason} (skipping)")
    
    # Step 4: For valid trials, remove low-speed periods
    filtered_spks_laps = []
    filtered_location_laps = []
    
    print(f"\nApplying speed filtering (min speed: {min_speed_cm_s} cm/s):")
    
    for i in valid_trials:
        lap_spks = spks_laps[i]
        lap_loc = location_laps[i]  # This is in AU
        
        # Convert this lap's location to cm
        lap_loc_cm = lap_loc * conversion_factor
        
        # Calculate speed for this lap
        lap_speed = np.abs(np.diff(lap_loc_cm)) * framerate
        lap_speed = np.concatenate(([lap_speed[0]], lap_speed))  # Pad to match length
        
        # Identify low-speed periods
        slow_mask = lap_speed < min_speed_cm_s
        slow_indices = np.where(slow_mask)[0]
        
        # Group consecutive slow indices
        def group_consecutive(data):
            for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
                yield list(map(itemgetter(1), g))
        
        slow_periods = list(group_consecutive(slow_indices))
        
        # Create mask for which frames to keep
        mask = np.ones(len(lap_loc), dtype=bool)
        
        removed_frames = 0
        for period in slow_periods:
            if len(period) > 2 * frames_to_keep:
                start_idx = period[0]
                end_idx = period[-1]
                
                # Keep beginning and end, remove middle
                mask[start_idx + frames_to_keep:end_idx - frames_to_keep + 1] = False
                removed_frames += (end_idx - frames_to_keep + 1) - (start_idx + frames_to_keep)
        
        # Apply mask
        filtered_spks_laps.append(lap_spks[:, mask])
        filtered_location_laps.append(lap_loc_cm[mask])  # Store in cm
        
        # Report filtering results
        original_frames = len(lap_loc)
        kept_frames = len(lap_loc_cm[mask])
        print(f"  Lap {i+1}: {kept_frames}/{original_frames} frames kept " +
              f"({kept_frames/original_frames*100:.1f}%), removed {removed_frames} slow frames")
    
    n_valid_laps = len(valid_trials)
    
    print(f"\nFinal result: {n_valid_laps} valid laps after duration and speed filtering")
    
    # Plot speed distribution for validation
    # plot_speed_distribution(speed_cm_s, min_speed_cm_s)
    
    return filtered_spks_laps, filtered_location_laps, n_valid_laps

def plot_speed_distribution(speed_cm_s, min_speed_threshold):
    """Plot speed distribution to visualize filtering threshold."""
    
    plt.figure(figsize=(12, 4))
    
    # Speed histogram
    plt.subplot(1, 2, 1)
    plt.hist(speed_cm_s, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(min_speed_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Min speed threshold: {min_speed_threshold} cm/s')
    plt.xlabel('Speed (cm/s)')
    plt.ylabel('Count')
    plt.title('Speed Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Speed over time (first 5000 points for visibility)
    plt.subplot(1, 2, 2)
    n_points = min(5000, len(speed_cm_s))
    time_axis = np.arange(n_points) / 10  # Assuming 10 Hz
    plt.plot(time_axis, speed_cm_s[:n_points], 'b-', alpha=0.7, linewidth=0.5)
    plt.axhline(min_speed_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Min speed threshold: {min_speed_threshold} cm/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (cm/s)')
    plt.title('Speed Over Time (first 500s)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    below_threshold = np.sum(speed_cm_s < min_speed_threshold)
    total_frames = len(speed_cm_s)
    print(f"\nSpeed filtering statistics:")
    print(f"  Frames below {min_speed_threshold} cm/s: {below_threshold}/{total_frames} ({below_threshold/total_frames*100:.1f}%)")
    print(f"  Mean speed: {np.mean(speed_cm_s):.2f} cm/s")
    print(f"  Median speed: {np.median(speed_cm_s):.2f} cm/s")
    print(f"  Speed range: {np.min(speed_cm_s):.2f} to {np.max(speed_cm_s):.2f} cm/s")

def process_data_with_trial_filtering(spks, location, 
                                      min_trial_duration_seconds=5, 
                                      max_trial_duration_seconds=30, 
                                      framerate=10, 
                                      inactivity_threshold=1e-2, 
                                      frames_to_keep=0):
    """
    Process 2P and VR data with improved filtering:
    1. Split data into trials/laps
    2. Filter out trials that take too long
    3. Remove inactive periods within valid trials
    
    Parameters:
    -----------
    spks : numpy.ndarray
        Spike data (cells x time)
    location : numpy.ndarray
        Location data (time,)
    max_trial_duration_seconds : float
        Maximum duration for a valid trial in seconds
    framerate : float
        Frames per second of recording
    inactivity_threshold : float
        Threshold for detecting inactivity in location data
    frames_to_keep : int
        Number of frames to keep at the beginning and end of inactive periods
        
    Returns:
    --------
    filtered_spks_laps : list
        List of filtered spike data for each valid lap
    filtered_location_laps : list
        List of filtered location data for each valid lap
    n_valid_laps : int
        Number of valid laps after filtering
    """
    
    # Step 1: Reshape data into laps
    spks_laps, location_laps, n_laps = reshape_into_laps(spks, location, plot_detection=False)
    print("number of detected laps:", n_laps)
    
    if n_laps == 0:
        print("No laps detected!")
        return None, None, 0
    
    # Step 2: Filter out trials that take too long
    
    min_frames_per_trial = min_trial_duration_seconds * framerate
    max_frames_per_trial = max_trial_duration_seconds * framerate
    valid_trials = []
    
    print("\nFiltering trials by duration:")
    for i, (lap_spks, lap_loc) in enumerate(zip(spks_laps, location_laps)):
        trial_duration_frames = lap_spks.shape[1]
        trial_duration_seconds = trial_duration_frames / framerate
        
        # if trial_duration_frames <= max_frames_per_trial:
        if min_frames_per_trial <= trial_duration_frames <= max_frames_per_trial:
            valid_trials.append(i)
            # print(f"  Trial {i+1}: {trial_duration_seconds:.2f} seconds - VALID")
        # else:
            # print(f"  Trial {i+1}: {trial_duration_seconds:.2f} seconds - TOO LONG (skipping)")
    
    # Step 3: For valid trials, remove inactive periods
    filtered_spks_laps = []
    filtered_location_laps = []
    
    for i in valid_trials:
        lap_spks = spks_laps[i]
        lap_loc = location_laps[i]
        
        # Detect inactivity within this lap
        location_diff = np.abs(np.diff(lap_loc, n=5))
        stationary_mask = location_diff < inactivity_threshold
        stationary_indices = np.where(stationary_mask)[0]
        
        # Group consecutive stationary indices
        from itertools import groupby
        from operator import itemgetter
        
        def group_consecutive(data):
            for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
                yield list(map(itemgetter(1), g))
        
        stationary_periods = list(group_consecutive(stationary_indices))
        
        # Create mask for which frames to keep
        mask = np.ones(len(lap_loc), dtype=bool)
        
        for period in stationary_periods:
            if len(period) > 2 * frames_to_keep:
                start_idx = period[0]
                end_idx = period[-1]
                
                # Keep beginning and end, remove middle
                mask[start_idx + frames_to_keep:end_idx - frames_to_keep + 1] = False
        
        # Apply mask
        filtered_spks_laps.append(lap_spks[:, mask])
        filtered_location_laps.append(lap_loc[mask])
    
    n_valid_laps = len(valid_trials)
    
    # print(f"\nRetained {n_valid_laps} valid laps after filtering")
    # for i, lap_idx in enumerate(valid_trials):
    #     original_frames = len(location_laps[lap_idx])
    #     filtered_frames = len(filtered_location_laps[i])
    #     # print(f"  Lap {lap_idx+1}: {filtered_frames}/{original_frames} frames kept" + 
    #     #       f" ({filtered_frames/original_frames*100:.1f}%)")
    
    return filtered_spks_laps, filtered_location_laps, n_valid_laps