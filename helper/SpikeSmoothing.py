import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def find_temporal_offset(twoP_data, new_VR_data, framerate):
    from helper import SpikeSmoothing, ReliabilityTesting as RT, spatial_discretization as SD, BehavioralDataFiltering as DF
    # Find the temporal offset between the twoP and VR data
            
    # Define offsets to test (in frames)
    offset_frames_list = [1, 2, 3, 4, 5, 7]

    # Dictionary to store results
    offset_results = {}

    # Loop through offsets
    for offset_frames in offset_frames_list:
        print(f"\n\nTesting offset: {offset_frames} frames ({offset_frames/framerate:.2f} seconds)")
        
        # Apply offset to original data
        offset_spike_data = SpikeSmoothing.apply_temporal_offset(twoP_data['sps'], offset_frames)
        
        # Apply smoothing
        smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms=500)
        
        # Process data with trial filtering
        filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_trial_filtering(
            smoothed, 
            new_VR_data['interp_location'],
            min_trial_duration_seconds=5, 
            max_trial_duration_seconds=120,
            framerate=framerate
        )
        
        if n_valid_laps == 0:
            print(f"No valid laps for offset {offset_frames}")
            continue
        
        # single_revolution_VR = 282.415
        # single_revolution_treadmill = 27.8
        # single_lap_VR = 1726.99731 ### = 1146 when VR length was 125 at gain = 1.15 
        
        single_revolution_VR = 282.415
        single_revolution_treadmill = 27.8
        single_lap_VR = 1320.645683 ### = 1146 when VR length was 125 at gain = 1.15 
        single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR

        # single_revolution_VR = 282.415
        # single_revolution_treadmill = 27.8
        # single_lap_VR = 1126.0667 ### = 1146 when VR length was 125 at gain = 1.15 
        
        single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR


        # Perform spatial assignment
        spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = SD.spatial_assignment(
            n_valid_laps,
            filtered_spks_laps, 
            filtered_location_laps, 
            single_lap_treadmill
        )
        
        # Apply spatial smoothing
        smoothed_spatial_activity = SpikeSmoothing.spatial_smooth(spatial_activity, window_cm=10)


        # Test for reliability
        reliable_cells, avg_cc, cohens_d, iter_cc, _ = RT.test_cell_reliability(
            smoothed_spatial_activity,
            n_shuffles=100,           
            cc_percentile=90,          
            cohen_threshold=0.8,       
            min_cc_threshold=0.2,      
            min_activity_threshold=0.0, 
        )

        # Store results
        offset_results[offset_frames] = {
            'reliable_cells': reliable_cells,
            'reliable_count': np.sum(reliable_cells),
            'avg_cc': avg_cc,
            'cohens_d': cohens_d,
            'spatial_activity': smoothed_spatial_activity,
            'n_valid_laps': n_valid_laps
        }

        # Print summary
        print(f"Offset {offset_frames}: Found {np.sum(reliable_cells)} reliable cells out of {len(reliable_cells)}")
        print(f"Mean correlation for reliable cells: {np.mean(avg_cc[reliable_cells]):.3f}")
        print(f"Mean Cohen's D for reliable cells: {np.mean(cohens_d[reliable_cells]):.3f}")
        
    # Extract metrics for visualization
    valid_offsets = list(offset_results.keys())
    reliable_counts = [offset_results[offset]['reliable_count'] for offset in valid_offsets]
    avg_cc_means = [np.mean(offset_results[offset]['avg_cc'][offset_results[offset]['reliable_cells']]) 
                if np.sum(offset_results[offset]['reliable_cells']) > 0 else 0
                for offset in valid_offsets]
    cohens_d_means = [np.mean(offset_results[offset]['cohens_d'][offset_results[offset]['reliable_cells']])
                    if np.sum(offset_results[offset]['reliable_cells']) > 0 else 0
                    for offset in valid_offsets]

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot reliable cell count
    axes[0].plot(valid_offsets, reliable_counts, 'o-', color='blue', linewidth=2)
    axes[0].set_ylabel('Number of Reliable Cells')
    axes[0].set_title('Effect of Temporal Offset on Cell Reliability Metrics')
    axes[0].grid(True, alpha=0.3)

    # Plot mean correlation coefficient
    axes[1].plot(valid_offsets, avg_cc_means, 'o-', color='green', linewidth=2)
    axes[1].set_ylabel('Mean Correlation Coefficient')
    axes[1].grid(True, alpha=0.3)

    # Plot mean Cohen's D
    axes[2].plot(valid_offsets, cohens_d_means, 'o-', color='red', linewidth=2)
    axes[2].set_xlabel('Offset (frames)')
    axes[2].set_ylabel("Mean Cohen's D")
    axes[2].grid(True, alpha=0.3)

    # Find optimal offset (if results exist)
    if len(valid_offsets) > 0:
        # Normalize metrics for weighted average
        norm_reliable = np.array(reliable_counts) / np.max(reliable_counts) if np.max(reliable_counts) > 0 else np.zeros_like(reliable_counts)
        norm_cc = np.array(avg_cc_means) / np.max(avg_cc_means) if np.max(avg_cc_means) > 0 else np.zeros_like(avg_cc_means)
        norm_d = np.array(cohens_d_means) / np.max(cohens_d_means) if np.max(cohens_d_means) > 0 else np.zeros_like(cohens_d_means)
        
        # Weighted sum of normalized metrics
        combined_metric = (0.2 * norm_reliable + 0.4 * norm_cc + 0.4 * norm_d)
        # combined_metric = (norm_reliable + norm_cc + norm_d) / 3
        best_idx = np.argmax(combined_metric)
        best_offset = valid_offsets[best_idx]
        
        # Add vertical line at optimal offset
        for ax in axes:
            ax.axvline(x=best_offset, color='black', linestyle='--', alpha=0.7)
            ax.text(best_offset, ax.get_ylim()[1]*0.95, f'Optimal: {best_offset}', 
                horizontalalignment='center', verticalalignment='top')
        
        print(f"\nBest offset: {best_offset} frames ({best_offset/framerate:.2f} seconds)")
        print(f"- Reliable cells: {offset_results[best_offset]['reliable_count']}")
        print(f"- Mean correlation: {np.mean(offset_results[best_offset]['avg_cc'][offset_results[best_offset]['reliable_cells']]):.3f}")
        print(f"- Mean Cohen's D: {np.mean(offset_results[best_offset]['cohens_d'][offset_results[best_offset]['reliable_cells']]):.3f}")

    plt.tight_layout()
    # plt.show()   

def apply_temporal_offset(spike_data, offset_frames):
    """
    Apply a temporal offset to spike data relative to location data.
    Positive offset means neural activity shifted forward in time (appears later relative to location).
    Negative offset means neural activity shifted backward in time (appears earlier relative to location).
    """
    n_cells, n_frames = spike_data.shape
    
    if offset_frames == 0:
        return spike_data
    
    # Create new arrays to hold offset data
    offset_spike_data = np.zeros_like(spike_data)
    
    if offset_frames > 0:
        # Positive offset: spikes shifted forward (later) relative to location
        # Keep later part of spikes
        offset_spike_data[:, offset_frames:] = spike_data[:, :-offset_frames]
    else:
        # Negative offset: spikes shifted backward (earlier) relative to location
        # Keep earlier part of spikes
        abs_offset = abs(offset_frames)
        offset_spike_data[:, :-abs_offset] = spike_data[:, abs_offset:]
    
    return offset_spike_data

def smooth_spikes(spike_data, fps=10, window_ms=250):
    """
    Smooth spike data using a Gaussian window.
    
    Parameters:
    -----------
    spike_data : numpy.ndarray
        Array containing deconvolved spike data. Can be 1D (single trace) 
        or 2D (multiple traces, with traces in rows)
    fps : float
        Frames per second of the recording
    window_ms : float
        Width of the Gaussian window in milliseconds
    
    Returns:
    --------
    numpy.ndarray
        Smoothed spike data with same shape as input
    """
    # Convert window size from ms to frames
    window_frames = window_ms / 1000 * fps
    
    # Calculate sigma for Gaussian filter
    # Window size is typically considered as 6-sigma width
    sigma = window_frames / 6
    
    # Handle both 1D and 2D inputs
    if spike_data.ndim == 1:
        return gaussian_filter1d(spike_data, sigma=sigma)
    else:
        return np.array([gaussian_filter1d(trace, sigma=sigma) 
                        for trace in spike_data])

def plot_comparison(original, smoothed, fps=10, time_window=None, max_cells=5, cell_indices=None):
    """
    Efficiently plot original and smoothed traces for comparison.
    Optimized for large-scale neurophysiology data.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original spike data (1D or 2D array)
    smoothed : numpy.ndarray
        Smoothed spike data (1D or 2D array)
    fps : float
        Frames per second
    time_window : tuple, optional
        (start_time, end_time) in seconds to plot
    max_cells : int, optional
        Maximum number of cells to plot
    cell_indices : list or numpy.ndarray, optional
        Specific indices of cells to plot. If None, first max_cells will be used.
    """
    plt.ioff()  # Turn off interactive mode
    
    # Handle data selection
    if original.ndim == 1:
        n_cells = 1
        data_original = original[np.newaxis, :]
        data_smoothed = smoothed[np.newaxis, :]
    else:
        if cell_indices is None:
            cell_indices = np.arange(min(max_cells, original.shape[0]))
        n_cells = len(cell_indices)
        data_original = original[cell_indices]
        data_smoothed = smoothed[cell_indices]
    
    # Create decimated time vector for plotting
    total_points = original.shape[-1]
    if total_points > 5000:  # Decimate data for plotting if too many points
        decimation_factor = total_points // 5000 + 1
        time = np.arange(0, total_points, decimation_factor) / fps
        data_original = data_original[:, ::decimation_factor]
        data_smoothed = data_smoothed[:, ::decimation_factor]
    else:
        time = np.arange(total_points) / fps
    
    # Create figure
    fig, axes = plt.subplots(n_cells, 1, figsize=(10, 1.5*n_cells), 
                            sharex=True, constrained_layout=True)
    if n_cells == 1:
        axes = [axes]
    
    # Plot data
    for i, ax in enumerate(axes):
        ax.plot(time, data_original[i], 'gray', alpha=0.5, linewidth=0.5)
        ax.plot(time, data_smoothed[i], 'b', linewidth=0.5)
        
        # Minimal formatting
        ax.set_ylabel(f'Cell {cell_indices[i]}' if original.ndim > 1 else 'Rate')
        if i == n_cells - 1:
            ax.set_xlabel('Time (s)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i == 0:  # Legend only on first plot
            ax.legend(['Original', 'Smoothed'], frameon=False, 
                     loc='upper right', fontsize='small')
    
    return fig

def plot_sample_cells(original, smoothed, fps=10, n_samples=5, seed=None):
    """
    Plot a random sample of cells for quick visualization.
    
    Parameters:
    -----------
    original : numpy.ndarray
        Original spike data (2D array: cells × time)
    smoothed : numpy.ndarray
        Smoothed spike data (2D array: cells × time)
    fps : float
        Frames per second
    n_samples : int
        Number of random cells to sample
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_cells = original.shape[0]
    sample_indices = np.random.choice(n_cells, min(n_samples, n_cells), replace=False)
    sample_indices.sort()  # Sort for clearer visualization
    
    return plot_comparison(original, smoothed, fps=fps, 
                          cell_indices=sample_indices)

def spatial_smooth(data, window_cm=5, bin_size_cm=1):
    """
    Apply spatial smoothing with a Gaussian window.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Data to smooth (can be 1D, 2D, or 3D with spatial bins as last dimension)
    window_cm : float
        Width of Gaussian window in cm
    bin_size_cm : float
        Size of each spatial bin in cm (default=1, as each bin represents 1 cm)
        
    Returns:
    --------
    smoothed_data : numpy.ndarray
        Spatially smoothed data with same shape as input
    """
    
    # Convert window size from cm to bins
    window_bins = window_cm / bin_size_cm
    
    # Calculate sigma for Gaussian filter (window is typically 6-sigma width)
    sigma = window_bins / 6
    
    # Apply smoothing based on data dimensionality
    if data.ndim == 1:
        # 1D data: bins
        smoothed_data = gaussian_filter1d(data, sigma=sigma)
    elif data.ndim == 2:
        # 2D data: cells x bins or laps x bins
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            smoothed_data[i] = gaussian_filter1d(data[i], sigma=sigma)
    elif data.ndim == 3:
        # 3D data: cells x laps x bins
        smoothed_data = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                smoothed_data[i, j] = gaussian_filter1d(data[i, j], sigma=sigma)
    
    return smoothed_data

# Example usage:
if __name__ == "__main__":
    # Generate example data
    np.random.seed(42)
    t = np.linspace(0, 10, 100)  # 10 seconds at 10 fps
    
    # Example 1: Single cell
    print("Example 1: Single cell")
    spikes_1d = np.random.poisson(lam=2, size=100)  # Random spike train
    smoothed_1d = smooth_spikes(spikes_1d, fps=10, window_ms=250)
    plot_comparison(spikes_1d, smoothed_1d, fps=10)
    
    # Example 2: Multiple cells
    print("\nExample 2: Multiple cells")
    n_cells = 10
    spikes_2d = np.random.poisson(lam=2, size=(n_cells, 100))  # Multiple random spike trains
    smoothed_2d = smooth_spikes(spikes_2d, fps=10, window_ms=250)
    
    # Verify shapes
    print(f"Input shape: {spikes_2d.shape}")
    print(f"Output shape: {smoothed_2d.shape}")
    
    # Plot results for multiple cells
    plot_comparison(spikes_2d, smoothed_2d, fps=10)
    plt.show()