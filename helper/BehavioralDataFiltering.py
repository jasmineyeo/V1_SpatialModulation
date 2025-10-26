import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from operator import itemgetter


def calculate_vr_speed_and_distance(location_data, framerate):
    """
    Calculate running speed in cm/s and convert location to physical distance.
    
    NOTE: This should be called PER LAP to avoid teleportation artifacts!
    
    Parameters:
    -----------
    location_data : numpy.ndarray
        VR location data in arbitrary units (for ONE lap)
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
    # Calculate VR range for this specific lap
    vr_range = np.max(location_data) - np.min(location_data)
    
    # Convert to physical distance: ~379 AU = 130 cm
    conversion_factor = 130.0 / vr_range  # cm per AU
    
    # Convert location to cm
    location_cm = location_data * conversion_factor
    
    # Calculate speed in cm/s (frame-to-frame difference)
    speed_cm_s = np.abs(np.diff(location_cm)) * framerate
    
    # Pad to match original length (repeat first value)
    speed_cm_s = np.concatenate(([speed_cm_s[0]], speed_cm_s))
    
    return speed_cm_s, location_cm, conversion_factor

def calculate_vr_speed_and_distance_per_lap(location_laps_list, framerate):
    """
    Calculate running speed in cm/s separately for each lap to avoid teleportation artifacts.
    
    Parameters:
    -----------
    location_laps_list : list of numpy.ndarray
        List of location data for each lap in cm
    framerate : float
        Recording framerate in Hz
        
    Returns:
    --------
    speed_laps_list : list of numpy.ndarray
        List of speed data for each lap in cm/s
    """
    speed_laps_list = []
    
    for lap_location_cm in location_laps_list:
        # Calculate speed within this lap only
        lap_speed = np.abs(np.diff(lap_location_cm)) * framerate
        
        # Pad to match original length (repeat first value)
        lap_speed = np.concatenate(([lap_speed[0]], lap_speed))
        
        speed_laps_list.append(lap_speed)
    
    return speed_laps_list

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
    
    # Plot lap detection for verification
    if plot_detection:
        plt.figure(figsize=(15, 5))
        plt.plot(location, 'b-', alpha=0.5)
        
        # Plot lap transitions
        if len(lap_ends) > 1:
            plt.plot(lap_ends[:-1], location[lap_ends[:-1]], 'r.', markersize=10, label='Lap Ends')
        
        plt.plot(lap_starts, location[lap_starts], 'g.', markersize=10, label='Lap Starts')
        
        # Add threshold lines
        plt.axhline(y=threshold_high, color='gray', linestyle='--', alpha=0.5, label='High Threshold')
        plt.axhline(y=threshold_low, color='gray', linestyle='--', alpha=0.5, label='Low Threshold')
        
        plt.title(f'Lap Detection (Found {n_laps} laps)')
        plt.xlabel('Time')
        plt.ylabel('Location')
        plt.legend()
    
    # Create lists to store data for each lap
    spks_laps = []
    location_laps = []
    
    # Fill in the data with full lap lengths
    for start, end in zip(lap_starts, lap_ends):
        location_laps.append(location[start:end])
        spks_laps.append(spks[:, start:end])
    
    return spks_laps, location_laps, n_laps

def reshape_into_laps_teleportation_aware(spks, location, min_backward_jump=100, plot_detection=True):
    """
    Reshape data into laps by detecting TELEPORTATION events (large backward jumps).
    
    This is more robust than threshold-based detection when laps reset via teleportation.
    
    Parameters:
    -----------
    spks : numpy.ndarray
        Spike data (cells x time)
    location : numpy.ndarray
        Location data (time,) in arbitrary units
    min_backward_jump : float
        Minimum backward jump (in AU) to consider as teleportation (default: 100)
    plot_detection : bool
        Whether to plot the lap detection result
        
    Returns:
    --------
    spks_laps : list
        List of spike data for each lap
    location_laps : list
        List of location data for each lap
    n_laps : int
        Number of laps detected
    """
    
    # Calculate frame-to-frame changes
    location_diff = np.diff(location)
    
    # Find teleportation events (large NEGATIVE jumps)
    # Teleportation: location drops by >100 AU (e.g., 390 → 1)
    teleportation_frames = np.where(location_diff < -min_backward_jump)[0]
    
    print(f"\nTeleportation-based lap detection:")
    print(f"  Minimum backward jump threshold: {min_backward_jump} AU")
    print(f"  Detected {len(teleportation_frames)} teleportation events")
    
    if len(teleportation_frames) == 0:
        print("  WARNING: No teleportation events detected!")
        print("  Falling back to threshold-based detection...")
        return reshape_into_laps(spks, location, plot_detection=plot_detection)
    
    # Lap boundaries: start of data, right after each teleportation, end of data
    # The teleportation frame belongs to the PREVIOUS lap (last frame before reset)
    lap_starts = np.concatenate(([0], teleportation_frames + 1))
    lap_ends = np.concatenate((teleportation_frames + 1, [len(location)]))
    
    n_laps = len(lap_starts)
    
    # Verify each lap
    print(f"\n  Lap verification:")
    valid_laps = []
    for i, (start, end) in enumerate(zip(lap_starts, lap_ends)):
        lap_length = end - start
        lap_loc = location[start:end]
        lap_range = np.max(lap_loc) - np.min(lap_loc)
        
        # Check if lap makes sense (has reasonable range, not too short)
        if lap_length > 50 and lap_range > 50:  # At least 50 frames and 50 AU range
            valid_laps.append(i)
            if i < 5:  # Print first 5
                print(f"    Lap {i+1}: frames {start}-{end} ({lap_length} frames), "
                      f"range {np.min(lap_loc):.1f}-{np.max(lap_loc):.1f} AU")
        else:
            print(f"    Lap {i+1}: INVALID (length={lap_length}, range={lap_range:.1f})")
    
    # Keep only valid laps
    lap_starts = lap_starts[valid_laps]
    lap_ends = lap_ends[valid_laps]
    n_laps = len(lap_starts)
    
    print(f"\n  Final: {n_laps} valid laps")
    
    # Plot detection
    if plot_detection:
        plt.figure(figsize=(15, 5))
        plt.plot(location, 'b-', alpha=0.5, linewidth=0.5)
        
        # Mark teleportation events
        plt.plot(teleportation_frames, location[teleportation_frames], 
                'rx', markersize=10, markeredgewidth=2, label='Teleportation')
        
        # Mark lap starts
        plt.plot(lap_starts, location[lap_starts], 'go', 
                markersize=8, label='Lap Starts', alpha=0.7)
        
        plt.title(f'Teleportation-Based Lap Detection (Found {n_laps} valid laps)')
        plt.xlabel('Frame')
        plt.ylabel('Location (AU)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Create lap data
    spks_laps = []
    location_laps = []
    
    for start, end in zip(lap_starts, lap_ends):
        location_laps.append(location[start:end])
        spks_laps.append(spks[:, start:end])
    
    return spks_laps, location_laps, n_laps
def diagnose_speed_artifacts(location_laps_cm, speed_laps_list, framerate, max_reasonable_speed=30):
    """
    Diagnose where speed artifacts are coming from within laps.
    
    Parameters:
    -----------
    location_laps_cm : list of numpy.ndarray
        Location data for each lap in cm
    speed_laps_list : list of numpy.ndarray
        Speed data for each lap in cm/s
    framerate : float
        Recording framerate in Hz
    max_reasonable_speed : float
        Maximum physiologically reasonable speed (cm/s)
    """
    
    print("\n" + "="*80)
    print("DIAGNOSING SPEED ARTIFACTS")
    print("="*80)
    
    total_artifacts = 0
    
    for lap_idx, (lap_loc, lap_speed) in enumerate(zip(location_laps_cm, speed_laps_list)):
        artifact_frames = np.where(lap_speed > max_reasonable_speed)[0]
        
        if len(artifact_frames) > 0:
            total_artifacts += len(artifact_frames)
            
            if lap_idx < 3:  # Show details for first 3 laps with artifacts
                print(f"\nLap {lap_idx + 1}: {len(artifact_frames)} artifact frames")
                print(f"  Lap location range: {np.min(lap_loc):.1f} to {np.max(lap_loc):.1f} cm")
                
                # Show first few artifacts
                for i, frame in enumerate(artifact_frames[:5]):
                    if frame > 0:
                        prev_loc = lap_loc[frame-1]
                        curr_loc = lap_loc[frame]
                        jump = curr_loc - prev_loc
                        speed = lap_speed[frame]
                        
                        print(f"    Frame {frame}: {prev_loc:.1f} → {curr_loc:.1f} cm "
                              f"(Δ={jump:.1f} cm, speed={speed:.1f} cm/s)")
                        
                        # Check if this looks like a backward jump (mini-teleport)
                        if jump < -10:
                            print(f"      ⚠️  BACKWARD JUMP detected!")
                        elif abs(jump) > 20:
                            print(f"      ⚠️  LARGE JUMP detected!")
    
    print(f"\n" + "="*80)
    print(f"Total artifacts across all laps: {total_artifacts}")
    print(f"Possible causes:")
    print(f"  1. Mini-teleportations within laps")
    print(f"  2. Tracking glitches in VR system")
    print(f"  3. Very rapid position changes")
    print("="*80)

def filter_backward_running_laps(spks_laps, location_laps_cm, speed_laps_list, 
                                 max_backward_threshold=-5.0, 
                                 max_backward_fraction=0.1):
    """
    Filter out laps where the animal runs backward significantly.
    
    Parameters:
    -----------
    spks_laps : list
        List of spike data for each lap
    location_laps_cm : list
        List of location data for each lap (in cm)
    speed_laps_list : list
        List of speed data for each lap (in cm/s)
    max_backward_threshold : float
        Threshold for detecting backward jumps (cm, negative value)
    max_backward_fraction : float
        Maximum allowed fraction of frames with backward movement
        
    Returns:
    --------
    filtered_spks : list
        Filtered spike data
    filtered_location : list
        Filtered location data
    filtered_speed : list
        Filtered speed data
    n_valid : int
        Number of valid laps
    backward_lap_indices : list
        Indices of laps that were filtered out
    """
    
    filtered_spks = []
    filtered_location = []
    filtered_speed = []
    backward_lap_indices = []
    
    print("\n" + "="*80)
    print("FILTERING LAPS WITH BACKWARD RUNNING")
    print("="*80)
    print(f"Criteria: >10% of frames with backward jumps < {max_backward_threshold} cm")
    
    for i, (spks, loc, speed) in enumerate(zip(spks_laps, location_laps_cm, speed_laps_list)):
        # Calculate frame-to-frame changes (with direction)
        loc_diff = np.diff(loc)
        
        # Count backward frames (large negative jumps)
        backward_frames = np.sum(loc_diff < max_backward_threshold)
        backward_fraction = backward_frames / len(loc_diff)
        
        # Also check for extreme speeds (>50 cm/s) as another indicator
        extreme_speed_frames = np.sum(speed > 50)
        
        if backward_fraction > max_backward_fraction or extreme_speed_frames > 5:
            backward_lap_indices.append(i)
            print(f"  ❌ Lap {i+1}: EXCLUDED - {backward_frames}/{len(loc_diff)} backward frames "
                  f"({backward_fraction*100:.1f}%), {extreme_speed_frames} extreme speeds")
        else:
            filtered_spks.append(spks)
            filtered_location.append(loc)
            filtered_speed.append(speed)
            if backward_frames > 0:
                print(f"  ✓ Lap {i+1}: OK - {backward_frames}/{len(loc_diff)} backward frames "
                      f"({backward_fraction*100:.1f}%)")
    
    n_valid = len(filtered_spks)
    n_excluded = len(backward_lap_indices)
    
    print(f"\n{'='*80}")
    print(f"BACKWARD RUNNING FILTER SUMMARY:")
    print(f"  Valid laps: {n_valid}")
    print(f"  Excluded laps: {n_excluded}")
    print(f"  Exclusion rate: {n_excluded/(n_valid+n_excluded)*100:.1f}%")
    print(f"{'='*80}\n")
    
    return filtered_spks, filtered_location, filtered_speed, n_valid, backward_lap_indices
def process_data_with_speed_filtering(spks, location, 
                                    min_trial_duration_seconds=5, 
                                    max_trial_duration_seconds=30, 
                                    framerate=10, 
                                    min_speed_cm_s=2.0,
                                    frames_to_keep=5,
                                    max_location_range_au=400,
                                    filter_backward_laps=True):
    """
    Process 2P and VR data with improved speed-based filtering:
    1. Split data into trials/laps
    2. Convert location to cm for each lap
    3. Calculate speed PER LAP (avoiding teleportation artifacts)
    4. Filter out laps with backward running (optional)
    5. Filter out trials that take too long or too short
    6. Remove periods with speed below threshold (stationary periods)
    
    Returns:
    --------
    filtered_spks_laps : list
        List of filtered spike data for each valid lap
    filtered_location_laps : list
        List of filtered location data for each valid lap (in cm)
    filtered_speed_laps : list
        List of filtered speed data for each valid lap (in cm/s)
    n_valid_laps : int
        Number of valid laps after filtering
    """
    
    # Step 1: Calculate conversion factor and convert location to cm
    print("Converting VR units to physical distance...")
    vr_range = np.max(location) - np.min(location)
    conversion_factor = 130.0 / vr_range  # cm per AU
    
    print(f"VR conversion factor: {conversion_factor:.4f} cm per AU")
    
    # Step 2: Reshape data into laps using original location (AU) for lap detection
    spks_laps, location_laps, n_laps = reshape_into_laps(spks, location, plot_detection=False)
    print("Number of detected laps:", n_laps)
    
    if n_laps == 0:
        print("No laps detected!")
        return None, None, None, 0
    
    # Step 3: Convert location to cm for each lap
    location_laps_cm = [lap_loc * conversion_factor for lap_loc in location_laps]
    
    # Step 4: Calculate speed PER LAP (this avoids teleportation artifacts!)
    print("Calculating speed per lap to avoid teleportation artifacts...")
    speed_laps_list = calculate_vr_speed_and_distance_per_lap(location_laps_cm, framerate)
    
    # Diagnose artifacts BEFORE filtering
    diagnose_speed_artifacts(location_laps_cm, speed_laps_list, framerate, max_reasonable_speed=30)
    
    # NEW: Filter out backward running laps BEFORE other filtering
    if filter_backward_laps:
        spks_laps, location_laps_cm, speed_laps_list, n_laps_after_backward, _ = \
            filter_backward_running_laps(
                spks_laps, location_laps_cm, speed_laps_list,
                max_backward_threshold=-5.0,
                max_backward_fraction=0.1
            )
        
        if n_laps_after_backward == 0:
            print("No valid laps after backward running filter!")
            return None, None, None, 0
    
    # Print speed statistics AFTER backward filtering
    print("\nSpeed statistics for remaining laps:")
    for i, lap_speed in enumerate(speed_laps_list[:10]):  # Show first 10
        print(f"  Lap {i+1}: Speed range {np.min(lap_speed):.2f} to {np.max(lap_speed):.2f} cm/s")
    if len(speed_laps_list) > 10:
        print(f"  ... and {len(speed_laps_list)-10} more laps")
    
    # Step 5: Filter by duration and location range
    min_frames_per_trial = min_trial_duration_seconds * framerate
    max_frames_per_trial = max_trial_duration_seconds * framerate
    valid_trials = []
    
    print(f"\nFiltering trials by duration ({min_trial_duration_seconds}s to {max_trial_duration_seconds}s):")
    for i, lap_spks in enumerate(spks_laps):
        trial_duration_frames = lap_spks.shape[1]
        trial_duration_seconds = trial_duration_frames / framerate
        
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
        
        # Apply mask to ALL THREE arrays
        filtered_spks_laps.append(lap_spks[:, mask])
        filtered_location_laps.append(lap_loc_cm[mask])
        filtered_speed_laps.append(lap_speed[mask])
        
        original_frames = len(lap_loc_cm)
        kept_frames = np.sum(mask)
        print(f"  Lap {i+1}: {kept_frames}/{original_frames} frames kept " +
              f"({kept_frames/original_frames*100:.1f}%), removed {removed_frames} slow frames")
    
    n_valid_laps = len(valid_trials)
    
    print(f"\nFinal result: {n_valid_laps} valid laps after all filtering")
    
    # Plot speed distribution for final filtered data
    all_lap_speeds = np.concatenate([filtered_speed_laps[i] for i in range(len(filtered_speed_laps))])
    plot_speed_distribution(all_lap_speeds, min_speed_cm_s, framerate)

    return filtered_spks_laps, filtered_location_laps, filtered_speed_laps, n_valid_laps

def debug_speed_calculation(location, framerate, conversion_factor):
    """
    Debug where speed artifacts come from.
    """
    print("\n" + "="*80)
    print("DEBUG: Speed Calculation")
    print("="*80)
    
    # Method 1: Global speed (WRONG - has teleportation)
    location_cm_global = location * conversion_factor
    speed_global = np.abs(np.diff(location_cm_global)) * framerate
    speed_global = np.concatenate(([speed_global[0]], speed_global))
    
    print(f"\nMethod 1 (GLOBAL - expect artifacts):")
    print(f"  Max speed: {np.max(speed_global):.1f} cm/s")
    print(f"  Speeds > 50 cm/s: {np.sum(speed_global > 50)} frames")
    
    # Find where artifacts occur
    artifact_frames = np.where(speed_global > 50)[0]
    if len(artifact_frames) > 0:
        print(f"  Artifact frames: {artifact_frames[:5]}...")
        for frame in artifact_frames[:3]:
            if frame > 0:
                print(f"    Frame {frame}: {location_cm_global[frame-1]:.1f} → "
                      f"{location_cm_global[frame]:.1f} cm "
                      f"(Δ = {location_cm_global[frame] - location_cm_global[frame-1]:.1f} cm)")
    
    # Method 2: Per-lap speed (CORRECT - no teleportation)
    from helper.BehavioralDataFiltering import reshape_into_laps_teleportation_aware
    
    # Create dummy spike data
    dummy_spks = np.zeros((1, len(location)))
    _, location_laps, n_laps = reshape_into_laps_teleportation_aware(
        dummy_spks, location, 
        min_backward_jump=100,
        plot_detection=False
    )
    
    speed_laps = []
    for lap_loc in location_laps:
        lap_loc_cm = lap_loc * conversion_factor
        lap_speed = np.abs(np.diff(lap_loc_cm)) * framerate
        lap_speed = np.concatenate(([lap_speed[0]], lap_speed))
        speed_laps.append(lap_speed)
    
    speed_per_lap = np.concatenate(speed_laps)
    
    print(f"\nMethod 2 (PER-LAP - should be clean):")
    print(f"  Max speed: {np.max(speed_per_lap):.1f} cm/s")
    print(f"  Speeds > 50 cm/s: {np.sum(speed_per_lap > 50)} frames")
    
    if np.max(speed_per_lap) > 50:
        print(f"  ⚠️  WARNING: Still have artifacts with per-lap calculation!")
        print(f"  This suggests the location data itself has issues.")
    else:
        print(f"  ✓ Per-lap calculation successfully removes artifacts")
    
    # Plot comparison
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(location_cm_global[:1000], 'b-', linewidth=0.5)
    plt.ylabel('Location (cm)')
    plt.title('Raw Location (First 1000 frames)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(speed_global[:1000], 'r-', linewidth=0.5)
    plt.ylabel('Speed (cm/s)')
    plt.title('Method 1: Global Speed Calculation (has artifacts)')
    plt.axhline(50, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(speed_per_lap[:1000], 'g-', linewidth=0.5)
    plt.ylabel('Speed (cm/s)')
    plt.xlabel('Frame')
    plt.title('Method 2: Per-Lap Speed Calculation (should be clean)')
    plt.axhline(50, color='k', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def plot_speed_distribution(speed_cm_s, min_speed_threshold, framerate=10):
    """
    Plot speed distribution to visualize filtering threshold.
    
    Parameters:
    -----------
    speed_cm_s : numpy.ndarray
        Speed data in cm/s (can be concatenated speeds from multiple laps)
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
    