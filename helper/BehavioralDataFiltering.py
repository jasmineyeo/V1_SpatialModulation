import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter


def calculate_speed_per_lap(location_laps_cm, framerate):
    """
    Calculate speed separately for each lap to avoid teleportation artifacts.
    
    Parameters:
    -----------
    location_laps_cm : list of numpy.ndarray
        Location data for each lap in cm
    framerate : float
        Recording framerate in Hz
        
    Returns:
    --------
    speed_laps : list of numpy.ndarray
        Speed for each lap in cm/s (non-negative)
    """
    speed_laps = []
    
    for lap_location_cm in location_laps_cm:
        # Calculate frame-to-frame differences
        location_diff = np.diff(lap_location_cm)
        
        # Calculate speed (use absolute value to avoid negative speeds)
        lap_speed = np.abs(location_diff) * framerate
        
        # Pad to match original length (repeat first value)
        lap_speed = np.concatenate(([lap_speed[0]], lap_speed))
        
        speed_laps.append(lap_speed)
    
    return speed_laps

def reshape_into_laps_forward_only(spks, location, 
                                   high_percentile=90,  # Keep for backward compatibility
                                   use_fixed_threshold=True,  # NEW
                                   track_length_au=None,  # NEW
                                   min_lap_length=50,
                                   plot_detection=False):
    """
    Detect laps based on reaching the HIGH point (end of corridor).
    Each lap = start → end. Teleportation periods are automatically excluded.
    """
    
    # Find the "end point" threshold
    if use_fixed_threshold and track_length_au is not None:
        # Use fixed threshold based on known track length
        threshold_high = track_length_au * 0.98  # 95% of track length
        threshold_low = track_length_au * 0.05   # 5% of track length
        print(f"\nUsing FIXED thresholds based on track length:")
    else:
        # Use percentile-based thresholds (original behavior)
        threshold_high = np.percentile(location, high_percentile)
        threshold_low = np.percentile(location, 10)
        print(f"\nUsing PERCENTILE thresholds:")
    
    print(f"  Start threshold: {threshold_low:.1f} AU")
    print(f"  End threshold: {threshold_high:.1f} AU")
    print(f"  Actual data range: {np.min(location):.1f} to {np.max(location):.1f} AU")
    
    # Find all crossings of the HIGH threshold (lap completions)
    above_threshold = location > threshold_high
    crossings = np.diff(above_threshold.astype(int))
    
    # crossings == 1 means we just crossed UP through threshold (reached end)
    lap_end_frames = np.where(crossings == 1)[0]
    
    print(f"  Detected {len(lap_end_frames)} lap completions")
    
    if len(lap_end_frames) == 0:
        print("  No lap completions detected!")
        return None, None, 0
    
    # FIXED: Find lap starts properly
    # Method: After each lap end, find the MINIMUM location point (track reset)
    # This captures the true start of the next lap
    lap_starts = []
    
    # First lap starts at the beginning
    lap_starts.append(0)
    
    # For subsequent laps, find minimum location after previous lap end
    for i in range(len(lap_end_frames) - 1):
        prev_lap_end = lap_end_frames[i]
        next_lap_end = lap_end_frames[i + 1]
        
        # Search for minimum location between lap completions
        # This is where the track "resets" and the new lap begins
        search_window = location[prev_lap_end:next_lap_end]
        min_location_idx = np.argmin(search_window)
        lap_start = prev_lap_end + min_location_idx
        
        lap_starts.append(lap_start)
    
    lap_starts = np.array(lap_starts)
    lap_ends = lap_end_frames
    
    # Verify we have matching starts and ends
    if len(lap_starts) != len(lap_ends):
        print(f"  WARNING: Mismatch between starts ({len(lap_starts)}) and ends ({len(lap_ends)})")
        # Trim to matching length
        min_len = min(len(lap_starts), len(lap_ends))
        lap_starts = lap_starts[:min_len]
        lap_ends = lap_ends[:min_len]
    
    # Filter valid laps
    print(f"\n  Lap verification:")
    valid_laps = []
    
    for i, (start, end) in enumerate(zip(lap_starts, lap_ends)):
        lap_length = end - start
        
        if lap_length < min_lap_length:
            if i < 10:
                print(f"    Lap {i+1}: INVALID (too short: {lap_length} frames)")
            continue
        
        lap_loc = location[start:end]
        lap_range = np.max(lap_loc) - np.min(lap_loc)
        
        if lap_range < 100:
            if i < 10:
                print(f"    Lap {i+1}: INVALID (range too small: {lap_range:.1f} AU)")
            continue
        
        # ADDED: Check that lap actually starts near the beginning
        lap_start_location = location[start]
        if lap_start_location > threshold_low * 2:  # Should start near bottom
            if i < 10:
                print(f"    Lap {i+1}: INVALID (starts too high: {lap_start_location:.1f} AU)")
            continue
        
        valid_laps.append(i)
        
        if i < 10:
            print(f"    Lap {i+1}: OK - frames {start}-{end} ({lap_length} frames), "
                  f"range {np.min(lap_loc):.1f}-{np.max(lap_loc):.1f} AU, "
                  f"start location: {lap_start_location:.1f} AU")
    
    lap_starts = lap_starts[valid_laps]
    lap_ends = lap_ends[valid_laps]
    n_laps = len(lap_starts)
    
    print(f"\n  Final: {n_laps} valid laps (forward motion only)")
    
    # Plot
    if plot_detection:
        plt.figure(figsize=(15, 5))
        plt.plot(location, 'b-', alpha=0.5, linewidth=0.5)
        
        plt.plot(lap_starts, location[lap_starts], 'go', 
                markersize=8, label='Lap Starts', alpha=0.7)
        plt.plot(lap_ends, location[lap_ends], 'ro', 
                markersize=8, label='Lap Ends', alpha=0.7)
        
        plt.axhline(threshold_high, color='r', linestyle='--', alpha=0.5, label='End threshold')
        plt.axhline(threshold_low, color='g', linestyle='--', alpha=0.5, label='Start threshold')
        
        plt.title(f'Forward-Only Lap Detection (Found {n_laps} laps)')
        plt.xlabel('Frame')
        plt.ylabel('Location (AU)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Create lap data (forward motion only, no teleportation)
    spks_laps = []
    location_laps = []
    
    for start, end in zip(lap_starts, lap_ends):
        location_laps.append(location[start:end+1])  # Include end frame
        spks_laps.append(spks[:, start:end+1])
    
    return spks_laps, location_laps, n_laps


def process_data_with_speed_filtering(spks, location, 
                                    min_trial_duration_seconds=5, 
                                    max_trial_duration_seconds=30, 
                                    framerate=10, 
                                    min_speed_cm_s=2.0,
                                    frames_to_keep=5,
                                    lap_start_grace_cm=5.0):
    """
    Process 2P and VR data with speed-based filtering.
    """
    
    # Step 1: Determine track length in VR units
    print("Analyzing VR track dimensions...")
    
    # CRITICAL: Use the MAXIMUM location as track length reference
    # Not the range, since location might start above 0
    if np.max(location) < 393:
        track_length_au = np.max(location)
    elif np.max(location) > 393:
        track_length_au = 393.0
        
    
    location_range = track_length_au - np.min(location)


    print(f"Track length (VR units): {track_length_au:.1f} AU")
    print(f"Location range: {location_range:.1f} AU")
    
    # Calculate conversion factor
    conversion_factor = 130.0 / location_range  # cm per AU
    print(f"VR conversion factor: {conversion_factor:.4f} cm per AU")
    
    # Step 2: Detect laps with FIXED thresholds
    spks_laps, location_laps, n_laps = reshape_into_laps_forward_only(
        spks, location,
        use_fixed_threshold=True,  # NEW
        track_length_au=track_length_au,  # NEW
        min_lap_length=50,
        plot_detection=False
    )
    
    print("Number of detected laps:", n_laps)
    
    if n_laps == 0:
        print("No laps detected!")
        return None, None, None, 0
    
    # Step 3: Convert to cm
    location_laps_cm = [lap_loc * conversion_factor for lap_loc in location_laps]
    
    # Step 4: Calculate speed per lap (NO TELEPORTATION FRAMES!)
    print("Calculating speed per lap...")
    speed_laps_list = calculate_speed_per_lap(location_laps_cm, framerate)
    
    # Print speed statistics
    print("\nSpeed statistics:")
    all_speeds = np.concatenate(speed_laps_list)
    print(f"  Mean speed: {np.mean(all_speeds):.2f} cm/s")
    print(f"  Median speed: {np.median(all_speeds):.2f} cm/s")
    print(f"  Max speed: {np.max(all_speeds):.2f} cm/s")
    
    # Step 5: Filter by duration
    min_frames_per_trial = min_trial_duration_seconds * framerate
    max_frames_per_trial = max_trial_duration_seconds * framerate
    valid_trials = []
    
    print(f"\nFiltering trials by duration ({min_trial_duration_seconds}s to {max_trial_duration_seconds}s):")
    for i, lap_spks in enumerate(spks_laps):
        trial_duration_frames = lap_spks.shape[1]
        duration_valid = min_frames_per_trial <= trial_duration_frames <= max_frames_per_trial
        
        if duration_valid:
            valid_trials.append(i)
    
    print(f"Valid trials after duration filtering: {len(valid_trials)}/{len(spks_laps)}")
    
    # Step 6: Remove low-speed periods
    filtered_spks_laps = []
    filtered_location_laps = []
    filtered_speed_laps = []
    
    print(f"\nApplying speed filtering (min speed: {min_speed_cm_s} cm/s):")
    
    for i in valid_trials:
        lap_spks = spks_laps[i]
        lap_loc_cm = location_laps_cm[i]
        lap_speed = speed_laps_list[i]
        
        # Identify low-speed periods
        slow_mask = lap_speed < min_speed_cm_s
        slow_indices = np.where(slow_mask)[0]
        
        # Group consecutive slow indices
        def group_consecutive(data):
            for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
                yield list(map(itemgetter(1), g))
        
        slow_periods = list(group_consecutive(slow_indices))
        
        # Create mask for which frames to keep
        mask = np.ones(len(lap_loc_cm), dtype=bool)
        
        removed_frames = 0
        for period in slow_periods:
            if len(period) > 2 * frames_to_keep:
                start_idx = period[0]
                end_idx = period[-1]
                mask[start_idx + frames_to_keep:end_idx - frames_to_keep + 1] = False
                removed_frames += (end_idx - frames_to_keep + 1) - (start_idx + frames_to_keep)
                
        lap_location_norm = lap_loc_cm - np.min(lap_loc_cm)
        grace_mask = lap_location_norm <= lap_start_grace_cm
        combined_mask = grace_mask | mask


        # Apply mask to ALL THREE arrays
        filtered_spks_laps.append(lap_spks[:, combined_mask])
        filtered_location_laps.append(lap_loc_cm[combined_mask])
        filtered_speed_laps.append(lap_speed[combined_mask])
        
        original_frames = len(lap_loc_cm)
        kept_frames = np.sum(combined_mask)
        print(f"  Lap {i+1}: {kept_frames}/{original_frames} frames kept " +
              f"({kept_frames/original_frames*100:.1f}%), removed {removed_frames} slow frames")
    
    n_valid_laps = len(valid_trials)
    
    print(f"\nFinal result: {n_valid_laps} valid laps after all filtering")
    
    # Plot speed distribution for final filtered data
    # all_lap_speeds = np.concatenate(filtered_speed_laps)
    # plot_speed_distribution(all_lap_speeds, min_speed_cm_s, framerate)

    return filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps


def plot_speed_distribution(speed_cm_s, min_speed_threshold, framerate=10):
    """
    Plot speed distribution to visualize filtering threshold.
    
    Parameters:
    -----------
    speed_cm_s : numpy.ndarray
        Speed data in cm/s (concatenated from all laps)
    min_speed_threshold : float
        Minimum speed threshold in cm/s
    framerate : float
        Framerate in Hz (default: 10)
    """
    
    plt.figure(figsize=(12, 4))
    
    # Speed histogram
    plt.subplot(1, 2, 1)
    plt.hist(speed_cm_s, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(min_speed_threshold, color='red', linestyle='--', linewidth=2, 
                label=f'Min speed: {min_speed_threshold} cm/s')
    plt.xlabel('Speed (cm/s)')
    plt.ylabel('Count')
    plt.title('Speed Distribution (All Valid Laps)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Speed over time (first 5000 points for visibility)
    plt.subplot(1, 2, 2)
    n_points = min(5000, len(speed_cm_s))
    time_axis = np.arange(n_points) / framerate
    plt.plot(time_axis, speed_cm_s[:n_points], 'b-', alpha=0.7, linewidth=0.5)
    plt.axhline(min_speed_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Min speed: {min_speed_threshold} cm/s')
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