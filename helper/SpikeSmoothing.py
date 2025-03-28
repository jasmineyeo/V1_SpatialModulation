import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

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