# ReliabilityTesting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import os
from matplotlib.backends.backend_pdf import PdfPages

def test_cell_reliability(spatial_activity, n_shuffles=1000, 
                         cc_percentile=95, cohen_threshold=0.5,
                         min_cc_threshold=0.2, min_activity_threshold=0.1):
    """
    Test reliability of neural responses with multiple criteria.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    n_shuffles : int
        Number of shuffles (recommended 1000+ for final analysis)
    cc_percentile : float
        Percentile threshold for correlation coefficient comparison
    cohen_threshold : float
        Threshold for Cohen's D statistic
    min_cc_threshold : float
        Minimum correlation coefficient required (regardless of shuffling)
    min_activity_threshold : float
        Minimum mean activity required (normalized to max activity)
    """
    n_cells = spatial_activity.shape[0]
    n_trials = spatial_activity.shape[1]
    n_bins = spatial_activity.shape[2]
    
    # Initialize output arrays
    reliable_cells = np.zeros(n_cells, dtype=bool)
    average_cc = np.zeros(n_cells)
    cohen_d = np.zeros(n_cells)
    iterated_cc = np.zeros((n_shuffles, n_cells))
    
    # Calculate activity levels for all cells
    cell_activity_levels = np.mean(spatial_activity, axis=(1,2))
    max_activity = np.max(cell_activity_levels)
    normalized_activity = cell_activity_levels / max_activity if max_activity > 0 else cell_activity_levels

    for cell in tqdm(range(n_cells), desc="Testing cell reliability"):
        # Skip cells with too little activity
        if normalized_activity[cell] < min_activity_threshold:
            continue
            
        # Arrays to store correlation coefficients
        bt_cc_data = np.zeros(n_shuffles)
        bt_cc_rand = np.zeros(n_shuffles)
        
        cell_activity = spatial_activity[cell]
        
        for shuffle in range(n_shuffles):
            # # Random split of trials
            # trial_indices = np.random.permutation(n_trials)
            # split_point = n_trials // 2
            # trials1 = trial_indices[:split_point]
            # trials2 = trial_indices[split_point:]
            
            # Instead of random trial splitting, do even-odd splitting
            trials1 = np.arange(0, n_trials, 2)
            trials2 = np.arange(1, n_trials, 2)
            # Calculate means for actual data
            first_half_mean = np.mean(cell_activity[trials1], axis=0)
            second_half_mean = np.mean(cell_activity[trials2], axis=0)
            
            # Calculate correlation for actual data
            cc = np.corrcoef(first_half_mean, second_half_mean)[0, 1]
            bt_cc_data[shuffle] = cc
            
            # Create shuffled version of the data
            activity_rand = np.zeros_like(cell_activity)
            for trial in range(n_trials):
                shift = np.random.randint(n_bins)
                activity_rand[trial] = np.roll(cell_activity[trial], shift)
            
            # Calculate means for shuffled data
            first_half_rand_mean = np.mean(activity_rand[trials1], axis=0)
            second_half_rand_mean = np.mean(activity_rand[trials2], axis=0)
            
            # Calculate correlation for shuffled data
            cc_rand = np.corrcoef(first_half_rand_mean, second_half_rand_mean)[0, 1]
            bt_cc_rand[shuffle] = cc_rand
        
        # Calculate average correlation coefficient
        average_cc[cell] = np.mean(bt_cc_data)
        
        # Calculate Cohen's D
        mean_diff = np.mean(bt_cc_data) - np.mean(bt_cc_rand)
        n1, n2 = len(bt_cc_data), len(bt_cc_rand)
        var1, var2 = np.var(bt_cc_data, ddof=1), np.var(bt_cc_rand, ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        cohen_d[cell] = mean_diff / pooled_sd if pooled_sd > 0 else 0
        
        # Multiple criteria for reliability
        avg_bt_cc = np.mean(bt_cc_data)
        shuffle_threshold = np.percentile(bt_cc_rand, cc_percentile)
        
        # Cell is considered reliable if:
        # 1. Average correlation is above minimum threshold
        # 2. Average correlation is above shuffle threshold
        # 3. Effect size (Cohen's D) is large enough
        # if (avg_bt_cc > min_cc_threshold):  # Large enough effect        
        if (avg_bt_cc > min_cc_threshold and  # Minimum correlation threshold
            avg_bt_cc > shuffle_threshold and  # Better than shuffled
            cohen_d[cell] > cohen_threshold):  # Large enough effect
            reliable_cells[cell] = True
            
    return reliable_cells, average_cc, cohen_d, iterated_cc, normalized_activity


def test_cell_reliability_with_edge_visualization(spatial_activity, n_shuffles=1000, 
                        cc_percentile=95, cohen_threshold=0.5,
                        min_cc_threshold=0.2, min_activity_threshold=0.1,
                        exclude_edge_bins=3, edge_activity_threshold=0.7):
    """
    Extended version of test_cell_reliability that also returns information
    about which cells were excluded due to edge activity, for visualization.
    
    Parameters:
    -----------
    Same as test_cell_reliability
        
    Returns:
    --------
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    average_cc : numpy.ndarray
        Average correlation coefficients
    cohen_d : numpy.ndarray
        Cohen's D values
    iterated_cc : numpy.ndarray
        Correlation coefficients for each shuffle
    normalized_activity : numpy.ndarray
        Normalized activity levels
    edge_cells : numpy.ndarray
        Boolean array indicating cells excluded due to edge activity
    edge_activity_start : numpy.ndarray
        Activity level in starting edge bins for each cell
    edge_activity_end : numpy.ndarray
        Activity level in ending edge bins for each cell
    """
    n_cells = spatial_activity.shape[0]
    n_trials = spatial_activity.shape[1]
    n_bins = spatial_activity.shape[2]
    
    # Initialize output arrays
    reliable_cells = np.zeros(n_cells, dtype=bool)
    average_cc = np.zeros(n_cells)
    cohen_d = np.zeros(n_cells)
    iterated_cc = np.zeros((n_shuffles, n_cells))
    
    # Arrays for edge detection
    edge_cells = np.zeros(n_cells, dtype=bool)
    edge_activity_start = np.zeros(n_cells)
    edge_activity_end = np.zeros(n_cells)
    
    # Calculate activity levels for all cells
    cell_activity_levels = np.mean(spatial_activity, axis=(1,2))
    max_activity = np.max(cell_activity_levels)
    normalized_activity = cell_activity_levels / max_activity if max_activity > 0 else cell_activity_levels
    
    for cell in range(n_cells):
        # Skip cells with too little activity
        if normalized_activity[cell] < min_activity_threshold:
            continue
            
        # Arrays to store correlation coefficients
        bt_cc_data = np.zeros(n_shuffles)
        bt_cc_rand = np.zeros(n_shuffles)
        
        cell_activity = spatial_activity[cell]
        
        # Check for high activity in edge bins
        # First calculate trial-averaged activity
        trial_avg_activity = np.mean(cell_activity, axis=0)
        
        # Normalize trial averaged activity
        norm_trial_avg = (trial_avg_activity - np.min(trial_avg_activity)) / (np.max(trial_avg_activity) - np.min(trial_avg_activity)) if np.max(trial_avg_activity) > np.min(trial_avg_activity) else np.zeros_like(trial_avg_activity)

        # Calculate average activity in first and last bins
        start_bins_avg = np.mean(norm_trial_avg[:exclude_edge_bins])
        end_bins_avg = np.mean(norm_trial_avg[-exclude_edge_bins:])
        
        # Store edge activity values for visualization
        edge_activity_start[cell] = start_bins_avg
        edge_activity_end[cell] = end_bins_avg
        
        # Skip cells with high activity in edge bins
        if start_bins_avg > edge_activity_threshold or end_bins_avg > edge_activity_threshold:
            edge_cells[cell] = True
            continue
        
        for shuffle in range(n_shuffles):
            # Random split of trials
            trial_indices = np.random.permutation(n_trials)
            split_point = n_trials // 2
            trials1 = trial_indices[:split_point]
            trials2 = trial_indices[split_point:]
            
            # Calculate means for actual data
            first_half_mean = np.mean(cell_activity[trials1], axis=0)
            second_half_mean = np.mean(cell_activity[trials2], axis=0)
            
            # Calculate correlation for actual data
            cc = np.corrcoef(first_half_mean, second_half_mean)[0, 1]
            bt_cc_data[shuffle] = cc
            
            # Create shuffled version of the data
            activity_rand = np.zeros_like(cell_activity)
            for trial in range(n_trials):
                shift = np.random.randint(n_bins)
                activity_rand[trial] = np.roll(cell_activity[trial], shift)
            
            # Calculate means for shuffled data
            first_half_rand_mean = np.mean(activity_rand[trials1], axis=0)
            second_half_rand_mean = np.mean(activity_rand[trials2], axis=0)
            
            # Calculate correlation for shuffled data
            cc_rand = np.corrcoef(first_half_rand_mean, second_half_rand_mean)[0, 1]
            bt_cc_rand[shuffle] = cc_rand
        
        # Calculate average correlation coefficient
        average_cc[cell] = np.mean(bt_cc_data)
        
        # Calculate Cohen's D
        mean_diff = np.mean(bt_cc_data) - np.mean(bt_cc_rand)
        n1, n2 = len(bt_cc_data), len(bt_cc_rand)
        var1, var2 = np.var(bt_cc_data, ddof=1), np.var(bt_cc_rand, ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        cohen_d[cell] = mean_diff / pooled_sd if pooled_sd > 0 else 0
        
        # Multiple criteria for reliability
        avg_bt_cc = np.mean(bt_cc_data)
        shuffle_threshold = np.percentile(bt_cc_rand, cc_percentile)
        
        # Cell is considered reliable if:
        # 1. Average correlation is above minimum threshold
        # 2. Average correlation is above shuffle threshold
        # 3. Effect size (Cohen's D) is large enough
        if (avg_bt_cc > min_cc_threshold and  # Minimum correlation threshold
            avg_bt_cc > shuffle_threshold and  # Better than shuffled
            cohen_d[cell] > cohen_threshold):  # Large enough effect
            reliable_cells[cell] = True
            
    return reliable_cells, average_cc, cohen_d, iterated_cc, normalized_activity, edge_cells, edge_activity_start, edge_activity_end


def plot_edge_activity_distributions(edge_activity_start, edge_activity_end, reliable_cells, edge_cells, edge_threshold=0.7):
    """
    Visualize the distribution of edge activity for cells, highlighting which were
    excluded due to high edge activity.
    
    Parameters:
    -----------
    edge_activity_start : numpy.ndarray
        Activity in starting edge bins
    edge_activity_end : numpy.ndarray
        Activity in ending edge bins
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells (before edge exclusion)
    edge_cells : numpy.ndarray
        Boolean array indicating cells excluded due to edge activity
    edge_threshold : float
        Threshold used for excluding edge cells
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Create categories for cells
    categories = np.zeros(len(reliable_cells), dtype=int)
    categories[reliable_cells & ~edge_cells] = 1  # Reliable, not edge
    categories[reliable_cells & edge_cells] = 2   # Would be reliable, but excluded due to edge
    categories[~reliable_cells & edge_cells] = 3  # Not reliable, high edge activity
    
    # Plot start edge activity
    for cat, label, color in zip(
        [1, 2, 3, 0], 
        ['Reliable', 'Would be reliable (excluded)', 'Not reliable, high edge', 'Not reliable, low edge'],
        ['green', 'red', 'orange', 'gray']
    ):
        mask = categories == cat
        if np.any(mask):
            ax1.scatter(np.where(mask)[0], edge_activity_start[mask], 
                      color=color, alpha=0.7, label=label)
    
    ax1.axhline(y=edge_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({edge_threshold})')
    ax1.set_xlabel('Cell Index')
    ax1.set_ylabel('Start Edge Activity')
    ax1.set_title('Activity in First Few Bins')
    ax1.legend()
    
    # Plot end edge activity
    for cat, label, color in zip(
        [1, 2, 3, 0], 
        ['Reliable', 'Would be reliable (excluded)', 'Not reliable, high edge', 'Not reliable, low edge'],
        ['green', 'red', 'orange', 'gray']
    ):
        mask = categories == cat
        if np.any(mask):
            ax2.scatter(np.where(mask)[0], edge_activity_end[mask], 
                      color=color, alpha=0.7, label=label)
    
    ax2.axhline(y=edge_threshold, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({edge_threshold})')
    ax2.set_xlabel('Cell Index')
    ax2.set_ylabel('End Edge Activity')
    ax2.set_title('Activity in Last Few Bins')
    ax2.legend()
    
    plt.tight_layout()
    return fig


def visualize_cell_edge_profiles(spatial_activity, cell_indices, exclude_edge_bins=3):
    """
    Visualize the trial-averaged activity profiles for specific cells, highlighting edge regions.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    cell_indices : list or numpy.ndarray
        Indices of cells to visualize
    exclude_edge_bins : int
        Number of bins at edges that are considered "edge regions"
    """
    n_cells = len(cell_indices)
    n_cols = min(3, n_cells)
    n_rows = (n_cells + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    
    for i, cell_idx in enumerate(cell_indices):
        if i >= n_rows * n_cols:
            break
            
        ax = axes[i] if n_cells > 1 else axes
        
        # Get cell activity and calculate trial average
        cell_activity = spatial_activity[cell_idx]
        trial_avg = np.mean(cell_activity, axis=0)
        
        # Normalize for better visualization
        normalized_avg = (trial_avg - np.min(trial_avg)) / (np.max(trial_avg) - np.min(trial_avg)) if np.max(trial_avg) > np.min(trial_avg) else np.zeros_like(trial_avg)
        
        # Plot the profile
        ax.plot(normalized_avg, 'b-', linewidth=2)
        
        # Highlight edge regions
        n_bins = len(normalized_avg)
        start_region = np.arange(exclude_edge_bins)
        end_region = np.arange(n_bins - exclude_edge_bins, n_bins)
        
        ax.fill_between(start_region, 0, 1, color='red', alpha=0.2)
        ax.fill_between(end_region, 0, 1, color='red', alpha=0.2)
        
        # Calculate edge activity
        start_avg = np.mean(normalized_avg[:exclude_edge_bins])
        end_avg = np.mean(normalized_avg[-exclude_edge_bins:])
        
        ax.set_title(f'Cell {cell_idx} (Start: {start_avg:.2f}, End: {end_avg:.2f})')
        ax.set_xlabel('Spatial Bin')
        ax.set_ylabel('Normalized Activity')
        ax.set_ylim(0, 1.05)
    
    # Remove empty subplots
    for i in range(n_cells, n_rows * n_cols):
        if n_cells > 1:
            fig.delaxes(axes[i])
    
    plt.tight_layout()
    return fig

def normalize_spatial_activity(spatial_activity):
    """
    Normalize spatial activity on a per-lap, per-cell basis.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
        
    Returns:
    --------
    normalized_data : numpy.ndarray
        Normalized activity matrix with same shape as input
    """
    n_cells, n_trials, n_bins = spatial_activity.shape
    normalized_data = np.zeros_like(spatial_activity)
    
    for cell in range(n_cells):
        for trial in range(n_trials):
            trial_data = spatial_activity[cell, trial, :]
            min_val = np.min(trial_data)
            max_val = np.max(trial_data)
            
            if max_val > min_val:  # Avoid division by zero
                normalized_data[cell, trial, :] = (trial_data - min_val) / (max_val - min_val)
    
    return normalized_data

def plot_reliable_cells_side_by_side(spatial_activity, reliable_cells, max_cells=20, 
                                    avg_cc=None, cohen_d=None, normalize=True):
    """
    Plot spatial activity and trial-averaged activity for reliable cells side by side.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    max_cells : int
        Maximum number of reliable cells to plot
    avg_cc : numpy.ndarray, optional
        Average correlation coefficients for each cell
    cohen_d : numpy.ndarray, optional
        Cohen's D values for each cell
    normalize : bool
        Whether to normalize activity for better visualization
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plots
    """
    # Get indices of reliable cells
    reliable_indices = np.where(reliable_cells)[0]
    n_reliable = len(reliable_indices)
    
    if n_reliable == 0:
        print("No reliable cells found!")
        return None
    
    # Limit to max_cells
    n_cells_to_plot = min(max_cells, n_reliable)
    cells_to_plot = reliable_indices[:n_cells_to_plot]
    
    # Set up the figure
    fig = plt.figure(figsize=(15, n_cells_to_plot * 2))
    gs = GridSpec(n_cells_to_plot, 2, width_ratios=[2, 1])
    
    for i, cell_idx in enumerate(cells_to_plot):
        cell_activity = spatial_activity[cell_idx].copy()
        
        # Normalize if requested
        if normalize:
            max_val = np.max(cell_activity)
            min_val = np.min(cell_activity)
            if max_val > min_val:
                cell_activity = (cell_activity - min_val) / (max_val - min_val)
        
        # Calculate trial-averaged activity
        trial_averaged = np.mean(cell_activity, axis=0)
        std_activity = np.std(cell_activity, axis=0)
        
        # Plot spatial activity (all trials)
        ax1 = fig.add_subplot(gs[i, 0])
        im = ax1.imshow(cell_activity, aspect='auto', cmap='viridis', interpolation='none')
        plt.colorbar(im, ax=ax1)
        
        # Add title with reliability info
        title = f'Cell {cell_idx}'
        if avg_cc is not None and cohen_d is not None:
            title += f' - CC: {avg_cc[cell_idx]:.2f}, d: {cohen_d[cell_idx]:.2f}'
        ax1.set_title(title)
        ax1.set_xlabel('Position (bins)')
        ax1.set_ylabel('Trial')
        
        # Plot trial-averaged activity
        ax2 = fig.add_subplot(gs[i, 1])
        ax2.plot(trial_averaged, 'b-', linewidth=2)
        ax2.fill_between(range(len(trial_averaged)), 
                        trial_averaged - std_activity,
                        trial_averaged + std_activity,
                        alpha=0.3, color='b')
        ax2.set_title(f'Trial-Averaged Activity')
        ax2.set_xlabel('Position (bins)')
        ax2.set_ylabel('Activity')
        
    plt.tight_layout()
    print(f"Displaying {n_cells_to_plot} reliable cells out of {n_reliable} total reliable cells")
    return fig

def plot_reliable_cells_grid(spatial_activity, reliable_cells, max_cells=20, 
                             avg_cc=None, cohen_d=None, normalize=True, n_rows=5, n_cols=4):
    """
    Plot spatial activity for reliable cells only in a grid layout.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    max_cells : int
        Maximum number of reliable cells to plot
    avg_cc : numpy.ndarray, optional
        Average correlation coefficients for each cell
    cohen_d : numpy.ndarray, optional
        Cohen's D values for each cell
    normalize : bool
        Whether to normalize activity for better visualization
    n_rows : int
        Number of rows in the grid
    n_cols : int
        Number of columns in the grid
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the plots
    """
    # Get indices of reliable cells
    reliable_indices = np.where(reliable_cells)[0]
    n_reliable = len(reliable_indices)
    
    if n_reliable == 0:
        print("No reliable cells found!")
        return None
    
    # Limit to max_cells
    n_cells_to_plot = min(max_cells, n_reliable)
    cells_to_plot = reliable_indices[:n_cells_to_plot]
    
    # Set up the figure
    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3))
    
    # Create a grid of subplots
    for i, cell_idx in enumerate(cells_to_plot):
        if i >= n_rows * n_cols:
            break  # Stop if we've filled the grid
            
        # Get cell activity
        cell_activity = spatial_activity[cell_idx].copy()
        
        # Normalize if requested
        if normalize:
            max_val = np.max(cell_activity)
            min_val = np.min(cell_activity)
            if max_val > min_val:
                cell_activity = (cell_activity - min_val) / (max_val - min_val)
        
        # Calculate trial-averaged activity
        trial_averaged = np.mean(cell_activity, axis=0)
        
        # Create subplot
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # Display heatmap
        im = ax.imshow(cell_activity, aspect='auto', cmap='viridis', interpolation='none')
        
        # Add title with cell info
        title = f'Cell {cell_idx}'
        if avg_cc is not None and cohen_d is not None:
            title += f'\nCC: {avg_cc[cell_idx]:.2f}, d: {cohen_d[cell_idx]:.2f}'
        ax.set_title(title)
        
        # Add minimal labels for cleaner look
        ax.set_xlabel('Position')
        ax.set_ylabel('Trial')
    
    plt.tight_layout()
    print(f"Displaying {n_cells_to_plot} reliable cells out of {n_reliable} total reliable cells")
    return fig

def plot_reliable_cells_waterfall(spatial_activity, reliable_cells, max_cells=10, spacing=1.5):
    """
    Create a waterfall plot of trial-averaged activity for reliable cells.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    max_cells : int
        Maximum number of cells to plot
    spacing : float
        Vertical spacing between traces
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure containing the waterfall plot
    """
    # Get indices of reliable cells
    reliable_indices = np.where(reliable_cells)[0]
    n_cells_to_plot = min(max_cells, len(reliable_indices))
    
    if n_cells_to_plot == 0:
        print("No reliable cells found!")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot each cell's trial-averaged activity
    for idx in range(n_cells_to_plot):
        cell_idx = reliable_indices[idx]
        cell_activity = spatial_activity[cell_idx]
        trial_averaged = np.mean(cell_activity, axis=0)
        
        # Normalize the trace
        normalized_trace = (trial_averaged - np.min(trial_averaged)) / \
                         (np.max(trial_averaged) - np.min(trial_averaged))
        
        # Plot with offset
        ax.plot(normalized_trace + idx * spacing, 'b-', linewidth=1.5)
        ax.text(-5, idx * spacing, f'Cell {cell_idx}', 
                verticalalignment='center', horizontalalignment='right')
    
    ax.set_title('Trial-Averaged Activity of Reliable Cells')
    ax.set_xlabel('Position (bins)')
    ax.set_ylabel('Cells')
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def evaluate_pattern_similarity(spatial_activity, min_pattern_corr=0.4, peak_distance_threshold=5):
    """
    Evaluates both correlation and pattern similarity between odd and even trials.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    min_pattern_corr : float
        Minimum correlation threshold for pattern similarity
    peak_distance_threshold : int
        Maximum allowed distance between peaks in odd and even trials (in bins)
        
    Returns:
    --------
    pattern_reliable : numpy.ndarray
        Boolean array of cells with consistent patterns
    odd_even_corr : numpy.ndarray
        Correlation between odd and even trials for each cell
    peak_distances : numpy.ndarray
        Distance between peaks in odd and even trials for each cell
    """
    n_cells, n_trials, n_bins = spatial_activity.shape
    
    # Initialize output arrays
    pattern_reliable = np.zeros(n_cells, dtype=bool)
    odd_even_corr = np.zeros(n_cells)
    peak_distances = np.zeros(n_cells)
    
    # Separate odd and even trials
    odd_trials = np.arange(0, n_trials, 2)
    even_trials = np.arange(1, n_trials, 2)
    
    for cell in range(n_cells):
        # Calculate mean activity for odd and even trials
        odd_mean = np.mean(spatial_activity[cell, odd_trials], axis=0)
        even_mean = np.mean(spatial_activity[cell, even_trials], axis=0)
        
        # Calculate correlation between odd and even mean activity patterns
        odd_even_corr[cell] = np.corrcoef(odd_mean, even_mean)[0, 1]
        
        # Find peaks in odd and even trials
        odd_peak = np.argmax(odd_mean)
        even_peak = np.argmax(even_mean)
        
        # Calculate shortest distance between peaks (accounting for circular track)
        raw_distance = abs(odd_peak - even_peak)
        circular_distance = min(raw_distance, n_bins - raw_distance)
        peak_distances[cell] = circular_distance
        
        # Check if patterns are similar (high correlation and consistent peak location)
        if (odd_even_corr[cell] >= min_pattern_corr and 
            peak_distances[cell] <= peak_distance_threshold):
            pattern_reliable[cell] = True
    
    return pattern_reliable, odd_even_corr, peak_distances

def combined_reliability_test(spatial_activity, n_shuffles=1000, 
                             cc_percentile=95, cohen_threshold=0.5,
                             min_cc_threshold=0.2, min_activity_threshold=0.1,
                             min_pattern_corr=0.4, peak_distance_threshold=5):
    """
    Combined reliability test checking both trial-to-trial reliability and odd-even pattern similarity
    """
    # Run original reliability test
    reliable_cells, avg_cc, cohens_d, iter_cc, norm_activity = test_cell_reliability(
        spatial_activity, n_shuffles, cc_percentile, cohen_threshold,
        min_cc_threshold, min_activity_threshold
    )
    
    # Run pattern similarity test
    pattern_reliable, odd_even_corr, peak_distances = evaluate_pattern_similarity(
        spatial_activity, min_pattern_corr, peak_distance_threshold
    )
    
    # Combine results (cells must pass both tests)
    combined_reliable = reliable_cells & pattern_reliable
    
    return (combined_reliable, reliable_cells, pattern_reliable, 
            avg_cc, cohens_d, odd_even_corr, peak_distances, norm_activity)
    
def improved_activity_threshold_check(spatial_activity, method='absolute_percentile', threshold_percentile=10):
    """
    Improved activity threshold that doesn't get dominated by highly active cells.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    method : str
        Method for threshold ('absolute_percentile', 'median_based', 'std_based')
    threshold_percentile : float
        Percentile of the activity distribution to use as threshold
        
    Returns:
    --------
    active_cells : numpy.ndarray
        Boolean array indicating cells that meet activity threshold
    threshold_value : float
        The threshold value used
    """
    
    # Calculate mean activity per cell
    cell_activity_levels = np.mean(spatial_activity, axis=(1, 2))
    
    if method == 'absolute_percentile':
        # Use percentile of all cell activities (not relative to max)
        threshold_value = np.percentile(cell_activity_levels, threshold_percentile)
        active_cells = cell_activity_levels >= threshold_value
        
    elif method == 'median_based':
        # Use fraction of median activity
        median_activity = np.median(cell_activity_levels)
        threshold_value = median_activity * 0.1  # 10% of median
        active_cells = cell_activity_levels >= threshold_value
        
    elif method == 'std_based':
        # Use mean - 2*std as threshold (excludes bottom ~2.5%)
        mean_activity = np.mean(cell_activity_levels)
        std_activity = np.std(cell_activity_levels)
        threshold_value = max(0, mean_activity - 2 * std_activity)
        active_cells = cell_activity_levels >= threshold_value
        
    print(f"Activity threshold ({method}): {threshold_value:.4f}")
    print(f"Cells passing activity threshold: {np.sum(active_cells)}/{len(active_cells)} ({np.sum(active_cells)/len(active_cells)*100:.1f}%)")
    
    return active_cells, threshold_value

def test_cell_reliability_improved(spatial_activity, n_shuffles=1000, 
                                 cc_percentile=95, cohen_threshold=0.8,
                                 min_cc_threshold=0.2, use_activity_threshold=True,
                                 activity_method='absolute_percentile'):
    """
    Improved reliability testing with better activity threshold handling.
    """
    n_cells = spatial_activity.shape[0]
    n_trials = spatial_activity.shape[1]
    n_bins = spatial_activity.shape[2]
    
    # Initialize output arrays
    reliable_cells = np.zeros(n_cells, dtype=bool)
    average_cc = np.zeros(n_cells)
    cohen_d = np.zeros(n_cells)
    
    # Apply activity threshold if requested
    if use_activity_threshold:
        active_cells, threshold_value = improved_activity_threshold_check(
            spatial_activity, method=activity_method
        )
    else:
        active_cells = np.ones(n_cells, dtype=bool)
        print("Skipping activity threshold - analyzing all cells")

    for cell in tqdm(range(n_cells), desc="Testing cell reliability"):
        # Skip cells that don't meet activity threshold
        if not active_cells[cell]:
            continue
            
        # Arrays to store correlation coefficients
        bt_cc_data = np.zeros(n_shuffles)
        bt_cc_rand = np.zeros(n_shuffles)
        
        cell_activity = spatial_activity[cell]
        
        for shuffle in range(n_shuffles):
            # Even-odd trial splitting (more robust than random)
            trials1 = np.arange(0, n_trials, 2)
            trials2 = np.arange(1, n_trials, 2)
            
            # Calculate means for actual data
            first_half_mean = np.mean(cell_activity[trials1], axis=0)
            second_half_mean = np.mean(cell_activity[trials2], axis=0)
            
            # Calculate correlation for actual data
            cc = np.corrcoef(first_half_mean, second_half_mean)[0, 1]
            bt_cc_data[shuffle] = cc if not np.isnan(cc) else 0
            
            # Create shuffled version of the data
            activity_rand = np.zeros_like(cell_activity)
            for trial in range(n_trials):
                shift = np.random.randint(n_bins)
                activity_rand[trial] = np.roll(cell_activity[trial], shift)
            
            # Calculate means for shuffled data
            first_half_rand_mean = np.mean(activity_rand[trials1], axis=0)
            second_half_rand_mean = np.mean(activity_rand[trials2], axis=0)
            
            # Calculate correlation for shuffled data
            cc_rand = np.corrcoef(first_half_rand_mean, second_half_rand_mean)[0, 1]
            bt_cc_rand[shuffle] = cc_rand if not np.isnan(cc_rand) else 0
        
        # Calculate average correlation coefficient
        average_cc[cell] = np.mean(bt_cc_data)
        
        # Calculate Cohen's D
        mean_diff = np.mean(bt_cc_data) - np.mean(bt_cc_rand)
        n1, n2 = len(bt_cc_data), len(bt_cc_rand)
        var1, var2 = np.var(bt_cc_data, ddof=1), np.var(bt_cc_rand, ddof=1)
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
        cohen_d[cell] = mean_diff / pooled_sd if pooled_sd > 0 else 0
        
        # Multiple criteria for reliability
        avg_bt_cc = np.mean(bt_cc_data)
        shuffle_threshold = np.percentile(bt_cc_rand, cc_percentile)
        
        # Cell is considered reliable if:
        if (avg_bt_cc > min_cc_threshold and  # Minimum correlation threshold
            avg_bt_cc > shuffle_threshold and  # Better than shuffled
            cohen_d[cell] > cohen_threshold):  # Large enough effect
            reliable_cells[cell] = True
            
    return reliable_cells, average_cc, cohen_d, active_cells

def combined_reliability_test_improved(spatial_activity, n_shuffles=1000, 
                                     cc_percentile=95, cohen_threshold=0.8,
                                     min_cc_threshold=0.2, min_pattern_corr=0.3, 
                                     peak_distance_threshold=5,
                                     use_activity_threshold=True,
                                     activity_method='absolute_percentile'):
    """
    Improved combined reliability test with better activity handling and 
    clearer pattern similarity evaluation.
    """
    
    # Run basic reliability test
    reliable_cells, avg_cc, cohens_d, active_cells = test_cell_reliability_improved(
        spatial_activity, n_shuffles, cc_percentile, cohen_threshold,
        min_cc_threshold, use_activity_threshold, activity_method
    )
    
    # Run pattern similarity test
    pattern_reliable, odd_even_corr, peak_distances = evaluate_pattern_similarity_improved(
        spatial_activity, min_pattern_corr, peak_distance_threshold
    )
    
    # Combine results (cells must pass both tests)
    combined_reliable = reliable_cells & pattern_reliable & active_cells
    
    print(f"\nReliability Test Results:")
    print(f"  Active cells: {np.sum(active_cells)}")
    print(f"  Reliable cells (correlation test): {np.sum(reliable_cells)}")
    print(f"  Pattern consistent cells: {np.sum(pattern_reliable)}")
    print(f"  Combined reliable cells: {np.sum(combined_reliable)}")
    
    return (combined_reliable, reliable_cells, pattern_reliable, 
            avg_cc, cohens_d, odd_even_corr, peak_distances, active_cells)

def evaluate_pattern_similarity_improved(spatial_activity, min_pattern_corr=0.3, 
                                       peak_distance_threshold=5):
    """
    Improved pattern similarity evaluation with better peak detection.
    
    This function is VERY IMPORTANT because it ensures that cells have:
    1. Consistent spatial patterns between odd and even trials
    2. Stable peak locations (not just good correlations)
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    min_pattern_corr : float
        Minimum correlation between odd/even trials
    peak_distance_threshold : int
        Maximum allowed distance between peaks in odd/even trials
        
    Returns:
    --------
    pattern_reliable : numpy.ndarray
        Boolean array of cells with consistent patterns
    odd_even_corr : numpy.ndarray
        Correlation between odd and even trials for each cell
    peak_distances : numpy.ndarray
        Distance between peaks in odd and even trials
    """
    n_cells, n_trials, n_bins = spatial_activity.shape
    
    # Initialize output arrays
    pattern_reliable = np.zeros(n_cells, dtype=bool)
    odd_even_corr = np.zeros(n_cells)
    peak_distances = np.zeros(n_cells)
    
    # Separate odd and even trials
    odd_trials = np.arange(0, n_trials, 2)
    even_trials = np.arange(1, n_trials, 2)
    
    for cell in range(n_cells):
        # Calculate mean activity for odd and even trials
        odd_mean = np.mean(spatial_activity[cell, odd_trials], axis=0)
        even_mean = np.mean(spatial_activity[cell, even_trials], axis=0)
        
        # Skip cells with no activity
        if np.max(odd_mean) == 0 or np.max(even_mean) == 0:
            continue
        
        # Calculate correlation between odd and even mean activity patterns
        correlation = np.corrcoef(odd_mean, even_mean)[0, 1]
        odd_even_corr[cell] = correlation if not np.isnan(correlation) else 0
        
        # Find peaks with improved detection
        odd_peak = find_robust_peak(odd_mean)
        even_peak = find_robust_peak(even_mean)
        
        # Calculate shortest distance between peaks (accounting for potential circular track)
        raw_distance = abs(odd_peak - even_peak)
        # For linear track, don't use circular distance
        peak_distances[cell] = raw_distance
        
        # Check if patterns are similar
        if (odd_even_corr[cell] >= min_pattern_corr and 
            peak_distances[cell] <= peak_distance_threshold):
            pattern_reliable[cell] = True
    
    print(f"\nPattern Similarity Results:")
    print(f"  Mean odd-even correlation: {np.mean(odd_even_corr[odd_even_corr > 0]):.3f}")
    print(f"  Mean peak distance: {np.mean(peak_distances):.1f} bins")
    print(f"  Cells with good correlation (>{min_pattern_corr}): {np.sum(odd_even_corr >= min_pattern_corr)}")
    print(f"  Cells with stable peaks (<{peak_distance_threshold} bins): {np.sum(peak_distances <= peak_distance_threshold)}")
    
    return pattern_reliable, odd_even_corr, peak_distances

def find_robust_peak(profile, smoothing_window=3):
    """
    Find peak location with optional smoothing for noise reduction.
    """
    if smoothing_window > 1:
        # Apply light smoothing to reduce noise in peak detection
        from scipy.ndimage import uniform_filter1d
        smoothed_profile = uniform_filter1d(profile, size=smoothing_window)
    else:
        smoothed_profile = profile
    
    return np.argmax(smoothed_profile)

# Usage example
def run_improved_reliability_test(normalized_spatial_activity):
    """
    Run the improved reliability test on your data.
    """
    
    print("=== IMPROVED RELIABILITY TESTING ===")
    
    # Test with activity threshold
    print("\nTesting WITH activity threshold:")
    results_with_threshold = combined_reliability_test_improved(
        normalized_spatial_activity,
        n_shuffles=100,
        cc_percentile=95,
        cohen_threshold=0.8,
        min_cc_threshold=0.3,
        min_pattern_corr=0.3,
        peak_distance_threshold=5,
        use_activity_threshold=True,
        activity_method='absolute_percentile'
    )
    
    # Test without activity threshold
    print("\nTesting WITHOUT activity threshold:")
    results_without_threshold = combined_reliability_test_improved(
        normalized_spatial_activity,
        n_shuffles=100,
        cc_percentile=95,
        cohen_threshold=0.8,
        min_cc_threshold=0.3,
        min_pattern_corr=0.3,
        peak_distance_threshold=5,
        use_activity_threshold=False
    )
    
    print(f"\nComparison:")
    print(f"  With activity threshold: {np.sum(results_with_threshold[0])} reliable cells")
    print(f"  Without activity threshold: {np.sum(results_without_threshold[0])} reliable cells")
    
    return results_with_threshold, results_without_threshold


def plot_individual_reliable_cells(spatial_activity, reliable_cells, save_directory, 
                                avg_cc=None, cohen_d=None, normalize=True,
                                bin_centers=None, figsize=(12, 8), dpi=150,
                                file_format='png'):
    """
    Create individual plots for each reliable cell and save them to a directory.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    save_directory : str
        Directory path where individual cell plots will be saved
    avg_cc : numpy.ndarray, optional
        Average correlation coefficients for each cell
    cohen_d : numpy.ndarray, optional
        Cohen's D values for each cell
    normalize : bool
        Whether to normalize activity for better visualization
    bin_centers : numpy.ndarray, optional
        Position bin centers for x-axis labeling (in cm)
    figsize : tuple
        Figure size for each individual plot
    dpi : int
        Resolution for saved images
    file_format : str
        File format for saved images ('png', 'pdf', 'svg')
        
    Returns:
    --------
    saved_files : list
        List of saved file paths
    summary_stats : dict
        Summary statistics about the plotting process
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    print(f"Saving individual cell plots to: {save_directory}")
    
    # Get indices of reliable cells
    reliable_indices = np.where(reliable_cells)[0]
    n_reliable = len(reliable_indices)
    
    if n_reliable == 0:
        print(" No reliable cells found!")
        return [], {}
    
    print(f"Creating individual plots for {n_reliable} reliable cells...")
    
    # Create bin centers if not provided
    if bin_centers is None:
        n_bins = spatial_activity.shape[2]
        bin_centers = np.arange(n_bins)
        x_label = 'Position (bins)'
    else:
        x_label = 'Position (cm)'
    
    saved_files = []
    plotting_stats = {
        'total_cells': n_reliable,
        'successfully_plotted': 0,
        'failed_plots': 0,
        'save_directory': save_directory
    }
    

    # Plot each reliable cell individually
    # Plot each reliable cell individually
    for i, cell_idx in enumerate(tqdm(reliable_indices, 
                                    desc="Plotting spatial profiles")):
        try:
            # Create figure for this cell
            fig = plt.figure(figsize=figsize, dpi=dpi)
            gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])
            
            # Get cell activity
            cell_activity = spatial_activity[cell_idx].copy()
            
            # Normalize if requested
            if normalize:
                max_val = np.max(cell_activity)
                min_val = np.min(cell_activity)
                if max_val > min_val:
                    cell_activity = (cell_activity - min_val) / (max_val - min_val)
            
            # Calculate statistics
            trial_averaged = np.mean(cell_activity, axis=0)
            std_activity = np.std(cell_activity, axis=0)
            sem_activity = std_activity / np.sqrt(cell_activity.shape[0])
            
            # Plot 1: All trials heatmap (top left)
            ax1 = fig.add_subplot(gs[0, 0])
            im = ax1.imshow(cell_activity, aspect='auto', cmap='viridis', 
                        interpolation='nearest', origin='lower')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label('Normalized Activity' if normalize else 'Activity')
            
            ax1.set_xlabel(x_label)
            ax1.set_ylabel('Trial')
            ax1.set_title(f'All Trials - Cell {cell_idx}')
            
            # Plot 2: Trial-averaged activity with error bars (top right)
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.plot(bin_centers, trial_averaged, 'b-', linewidth=2, label='Mean')
            ax2.fill_between(bin_centers, 
                        trial_averaged - sem_activity,
                        trial_averaged + sem_activity,
                        alpha=0.3, color='blue', label='±SEM')
            
            ax2.set_xlabel(x_label)
            ax2.set_ylabel('Activity')
            ax2.set_title('Trial-Averaged Activity')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: Individual trials overlay (bottom left)
            ax3 = fig.add_subplot(gs[1, 0])
            
            # Plot individual trials with transparency
            n_trials = cell_activity.shape[0]
            for trial in range(min(10, n_trials)):  # Show max 10 trials for clarity
                alpha = 0.3 if trial < 9 else 0.6  # Highlight last trial
                ax3.plot(bin_centers, cell_activity[trial], 
                        alpha=alpha, linewidth=1, color='gray')
            
            # Overlay the mean
            ax3.plot(bin_centers, trial_averaged, 'r-', linewidth=3, 
                    label=f'Mean (n={n_trials} trials)')
            
            ax3.set_xlabel(x_label)
            ax3.set_ylabel('Activity')
            ax3.set_title('Individual Trials + Mean')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Statistics and info (bottom right)
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.axis('off')
            
            # Prepare statistics text
            stats_text = f"Cell {cell_idx} Statistics\n\n"
            stats_text += f"Trials: {n_trials}\n"
            stats_text += f"Spatial bins: {len(bin_centers)}\n"
            stats_text += f"Peak activity: {np.max(trial_averaged):.3f}\n"
            stats_text += f"Mean activity: {np.mean(trial_averaged):.3f}\n"
            stats_text += f"Activity range: {np.max(trial_averaged) - np.min(trial_averaged):.3f}\n\n"
            
            # Add reliability metrics if available
            if avg_cc is not None:
                stats_text += f"Avg correlation: {avg_cc[cell_idx]:.3f}\n"
            if cohen_d is not None:
                stats_text += f"Cohen's D: {cohen_d[cell_idx]:.3f}\n"
            
            # Calculate additional metrics
            peak_location = bin_centers[np.argmax(trial_averaged)]
            if bin_centers is not None and len(bin_centers) > 1:
                stats_text += f"Peak location: {peak_location:.1f}\n"
            
            # Calculate spatial information
            if np.sum(trial_averaged) > 0:
                # Simple spatial information calculation
                mean_rate = np.mean(trial_averaged)
                if mean_rate > 0:
                    spatial_info = 0
                    for rate in trial_averaged:
                        if rate > 0:
                            spatial_info += (rate / len(trial_averaged)) * rate * np.log2(rate / mean_rate)
                    spatial_info = spatial_info / mean_rate
                    stats_text += f"Spatial info: {spatial_info:.3f} bits/spike\n"
            
            # Add the text
            ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
            
            # Overall figure title
            title_text = f'Cell {cell_idx} - Spatial Activity Profile'
            if avg_cc is not None and cohen_d is not None:
                title_text += f' (CC: {avg_cc[cell_idx]:.3f}, D: {cohen_d[cell_idx]:.3f})'
            
            fig.suptitle(title_text, fontsize=14, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)  # Make room for suptitle
            
            # Save the figure
            filename = f'cell_{cell_idx:04d}_spatial_profile.{file_format}'
            filepath = os.path.join(save_directory, filename)
            
            plt.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            plt.close(fig)  # Close to free memory
            
            saved_files.append(filepath)
            plotting_stats['successfully_plotted'] += 1
            
        except Exception as e:
            tqdm.write(f"Failed to plot cell {cell_idx}: {str(e)}")
            plotting_stats['failed_plots'] += 1
            continue
    
    # Create summary figure
    create_summary_figure(spatial_activity, reliable_cells, save_directory, 
                        avg_cc, cohen_d, bin_centers, file_format, dpi)
    
    # Print final summary
    print(f"\n Individual cell plotting complete!")
    print(f"   Successfully plotted: {plotting_stats['successfully_plotted']} cells")
    print(f"   Failed plots: {plotting_stats['failed_plots']} cells")
    print(f"   Files saved to: {save_directory}")
    
    return saved_files, plotting_stats

def plot_individual_reliable_cells_to_pdf(spatial_activity, reliable_cells, save_directory, 
                                        avg_cc=None, cohen_d=None, normalize=True,
                                        bin_centers=None, dpi=150, sigma=1,
                                        cells_per_page=4):
    """
    Create individual plots for each reliable cell and save them to a single PDF file.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    save_directory : str
        Directory path where PDF will be saved
    avg_cc : numpy.ndarray, optional
        Average correlation coefficients for each cell
    cohen_d : numpy.ndarray, optional
        Cohen's D values for each cell
    normalize : bool
        Whether to normalize activity for better visualization
    bin_centers : numpy.ndarray, optional
        Position bin centers for x-axis labeling (in cm)
    dpi : int
        Resolution for saved images
    sigma : float
        Standard deviation for gaussian smoothing (if needed)
    cells_per_page : int
        Number of cells to plot per PDF page
        
    Returns:
    --------
    pdf_filepath : str
        Path to the saved PDF file
    summary_stats : dict
        Summary statistics about the plotting process
    """
    
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)
    print(f"Saving reliable cell plots to PDF in: {save_directory}")
    
    # Get indices of reliable cells
    reliable_indices = np.where(reliable_cells)[0]
    n_reliable = len(reliable_indices)
    
    if n_reliable == 0:
        print("No reliable cells found!")
        return None, {}
    
    print(f"Creating PDF with {n_reliable} reliable cells...")
    
    # Create bin centers if not provided
    if bin_centers is None:
        n_bins = spatial_activity.shape[2]
        bin_centers = np.arange(n_bins)
        x_label = 'Position (bins)'
    else:
        x_label = 'Position (cm)'
    
    # Create PDF filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f'reliable_cells_spatial_profiles_{timestamp}.pdf'
    pdf_filepath = os.path.join(save_directory, pdf_filename)
    
    plotting_stats = {
        'total_cells': n_reliable,
        'successfully_plotted': 0,
        'failed_plots': 0,
        'pdf_filepath': pdf_filepath,
        'cells_per_page': cells_per_page
    }
    
    # Create PDF
    with PdfPages(pdf_filepath) as pdf:
        
        # Process cells in batches per page
        for batch_start in tqdm(range(0, n_reliable, cells_per_page), 
                               desc="Creating PDF pages"):
            batch_end = min(batch_start + cells_per_page, n_reliable)
            batch_indices = reliable_indices[batch_start:batch_end]
            
            # Create figure for this page
            fig = plt.figure(figsize=(16, 12), dpi=dpi)
            
            # Calculate subplot layout
            n_cells_this_page = len(batch_indices)
            if n_cells_this_page == 1:
                rows, cols = 1, 1
            elif n_cells_this_page == 2:
                rows, cols = 1, 2
            elif n_cells_this_page <= 4:
                rows, cols = 2, 2
            else:
                rows, cols = 3, 2  # For more than 4 cells
            
            try:
                # Plot each cell in this batch
                for i, cell_idx in enumerate(batch_indices):
                    
                    # Create subplot for this cell
                    ax = plt.subplot(rows, cols, i + 1)
                    
                    # Get cell activity
                    cell_activity = spatial_activity[cell_idx].copy()
                    
                    # Normalize if requested
                    if normalize:
                        max_val = np.max(cell_activity)
                        min_val = np.min(cell_activity)
                        if max_val > min_val:
                            cell_activity = (cell_activity - min_val) / (max_val - min_val)
                    
                    # Calculate trial-averaged activity
                    trial_averaged = np.mean(cell_activity, axis=0)
                    std_activity = np.std(cell_activity, axis=0)
                    sem_activity = std_activity / np.sqrt(cell_activity.shape[0])
                    
                    # Create a mini-subplot layout for this cell
                    gs_cell = GridSpec(2, 2, 
                                     height_ratios=[1, 1], 
                                     width_ratios=[2, 1],
                                     hspace=0.3, wspace=0.3)
                    
                    # Clear the current subplot and create mini subplots
                    ax.remove()
                    
                    # Calculate position for this cell's subplots
                    cell_row = i // cols
                    cell_col = i % cols
                    
                    # Define subplot positions
                    left = cell_col / cols
                    right = (cell_col + 1) / cols
                    bottom = 1 - (cell_row + 1) / rows
                    top = 1 - cell_row / rows
                    
                    # Create mini subplots
                    width = (right - left) * 0.48
                    height = (top - bottom) * 0.45
                    
                    # Heatmap (top left)
                    ax1 = fig.add_axes([left + 0.02, bottom + height + 0.05, width, height])
                    im = ax1.imshow(cell_activity, aspect='auto', cmap='viridis', 
                                  interpolation='nearest', origin='lower')
                    ax1.set_title(f'Cell {cell_idx} - All Trials', fontsize=10)
                    ax1.set_xlabel(x_label, fontsize=8)
                    ax1.set_ylabel('Trial', fontsize=8)
                    ax1.tick_params(labelsize=8)
                    
                    # Mean activity (top right)
                    ax2 = fig.add_axes([left + width + 0.04, bottom + height + 0.05, 
                                       width * 0.8, height])
                    ax2.plot(bin_centers, trial_averaged, 'b-', linewidth=2)
                    ax2.fill_between(bin_centers, 
                                   trial_averaged - sem_activity,
                                   trial_averaged + sem_activity,
                                   alpha=0.3, color='blue')
                    ax2.set_title('Mean ± SEM', fontsize=10)
                    ax2.set_xlabel(x_label, fontsize=8)
                    ax2.set_ylabel('Activity', fontsize=8)
                    ax2.tick_params(labelsize=8)
                    ax2.grid(True, alpha=0.3)
                    
                    # Individual trials (bottom left)
                    ax3 = fig.add_axes([left + 0.02, bottom + 0.02, width, height])
                    n_trials = cell_activity.shape[0]
                    for trial in range(min(8, n_trials)):
                        alpha = 0.4
                        ax3.plot(bin_centers, cell_activity[trial], 
                               alpha=alpha, linewidth=1, color='gray')
                    ax3.plot(bin_centers, trial_averaged, 'r-', linewidth=2, 
                           label=f'Mean (n={n_trials})')
                    ax3.set_title('Individual Trials', fontsize=10)
                    ax3.set_xlabel(x_label, fontsize=8)
                    ax3.set_ylabel('Activity', fontsize=8)
                    ax3.tick_params(labelsize=8)
                    ax3.grid(True, alpha=0.3)
                    
                    # Statistics (bottom right)
                    ax4 = fig.add_axes([left + width + 0.04, bottom + 0.02, 
                                       width * 0.8, height])
                    ax4.axis('off')
                    
                    # Prepare statistics text
                    stats_text = f"Cell {cell_idx}\n\n"
                    stats_text += f"Trials: {n_trials}\n"
                    stats_text += f"Peak: {np.max(trial_averaged):.3f}\n"
                    stats_text += f"Mean: {np.mean(trial_averaged):.3f}\n"
                    
                    if avg_cc is not None:
                        stats_text += f"Avg CC: {avg_cc[cell_idx]:.3f}\n"
                    if cohen_d is not None:
                        stats_text += f"Cohen's D: {cohen_d[cell_idx]:.3f}\n"
                    
                    peak_location = bin_centers[np.argmax(trial_averaged)]
                    stats_text += f"Peak loc: {peak_location:.1f}\n"
                    
                    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))
                    
                    plotting_stats['successfully_plotted'] += 1
                
                # Add page title
                page_title = f'Reliable Cells {reliable_indices[batch_start]}-{reliable_indices[batch_end-1]} '
                page_title += f'(Page {batch_start//cells_per_page + 1} of {(n_reliable-1)//cells_per_page + 1})'
                fig.suptitle(page_title, fontsize=16, fontweight='bold', y=0.98)
                
                # Save this page to PDF
                pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
                plt.close(fig)
                
            except Exception as e:
                print(f"Failed to create page for cells {batch_start}-{batch_end-1}: {str(e)}")
                plotting_stats['failed_plots'] += len(batch_indices)
                plt.close(fig)
                continue
    
    # Print final summary
    print(f"\nReliable cell PDF plotting complete!")
    print(f"   Successfully plotted: {plotting_stats['successfully_plotted']} cells")
    print(f"   Failed plots: {plotting_stats['failed_plots']} cells")
    print(f"   PDF saved to: {pdf_filepath}")
    
    return pdf_filepath, plotting_stats


def create_summary_figure(spatial_activity, reliable_cells, save_directory, 
                        avg_cc, cohen_d, bin_centers, file_format, dpi):
    """
    Create a summary figure showing overview of all reliable cells.
    """
    reliable_indices = np.where(reliable_cells)[0]
    n_reliable = len(reliable_indices)
    
    if n_reliable == 0:
        return
    
    # Create summary figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Distribution of reliability metrics
    ax = axes[0, 0]
    if avg_cc is not None:
        reliable_cc = avg_cc[reliable_cells]
        ax.hist(reliable_cc, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax.set_xlabel('Average Correlation Coefficient')
        ax.set_ylabel('Count')
        ax.set_title(f'Distribution of Correlation Coefficients\n(n={n_reliable} reliable cells)')
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of Cohen's D
    ax = axes[0, 1]
    if cohen_d is not None:
        reliable_cohen = cohen_d[reliable_cells]
        ax.hist(reliable_cohen, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel("Cohen's D")
        ax.set_ylabel('Count')
        ax.set_title(f"Distribution of Cohen's D\n(n={n_reliable} reliable cells)")
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Peak locations
    ax = axes[1, 0]
    peak_locations = []
    for cell_idx in reliable_indices:
        cell_avg = np.mean(spatial_activity[cell_idx], axis=0)
        peak_loc = np.argmax(cell_avg)
        if bin_centers is not None:
            peak_locations.append(bin_centers[peak_loc])
        else:
            peak_locations.append(peak_loc)
    
    ax.hist(peak_locations, bins=20, alpha=0.7, color='red', edgecolor='black')
    x_label = 'Peak Location (cm)' if bin_centers is not None else 'Peak Location (bins)'
    ax.set_xlabel(x_label)
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Peak Locations\n(n={n_reliable} reliable cells)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"RELIABLE CELLS SUMMARY\n\n"
    summary_text += f"Total reliable cells: {n_reliable}\n"
    summary_text += f"Total cells analyzed: {len(reliable_cells)}\n"
    summary_text += f"Reliability rate: {n_reliable/len(reliable_cells)*100:.1f}%\n\n"
    
    if avg_cc is not None:
        summary_text += f"Correlation Statistics:\n"
        summary_text += f"  Mean: {np.mean(avg_cc[reliable_cells]):.3f}\n"
        summary_text += f"  Median: {np.median(avg_cc[reliable_cells]):.3f}\n"
        summary_text += f"  Range: {np.min(avg_cc[reliable_cells]):.3f} - {np.max(avg_cc[reliable_cells]):.3f}\n\n"
    
    if cohen_d is not None:
        summary_text += f"Cohen's D Statistics:\n"
        summary_text += f"  Mean: {np.mean(cohen_d[reliable_cells]):.3f}\n"
        summary_text += f"  Median: {np.median(cohen_d[reliable_cells]):.3f}\n"
        summary_text += f"  Range: {np.min(cohen_d[reliable_cells]):.3f} - {np.max(cohen_d[reliable_cells]):.3f}\n\n"
    
    summary_text += f"Peak Location Statistics:\n"
    summary_text += f"  Mean: {np.mean(peak_locations):.1f}\n"
    summary_text += f"  Median: {np.median(peak_locations):.1f}\n"
    summary_text += f"  Range: {np.min(peak_locations):.1f} - {np.max(peak_locations):.1f}\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Reliable Cells Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save summary figure
    summary_filename = f'reliable_cells_summary.{file_format}'
    summary_filepath = os.path.join(save_directory, summary_filename)
    plt.savefig(summary_filepath, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    print(f"Summary figure saved: {summary_filename}")

def save_all_reliable_cell_plots(spatial_activity, reliable_cells, save_directory,
                                avg_cc=None, cohen_d=None, bin_centers=None,
                                normalize=True, file_format='png', dpi=150):
    """
    Simplified function to save all reliable cell plots.
    
    Usage example:
    --------------
    save_all_reliable_cell_plots(
        normalized_spatial_activity,  # or smoothed_spatial_activity
        combined_reliable,
        save_directory="./reliable_cell_plots",
        avg_cc=avg_cc,
        cohen_d=cohens_d,
        bin_centers=bin_centers
    )
    """
    
    saved_files, stats = plot_individual_reliable_cells(
        spatial_activity=spatial_activity,
        reliable_cells=reliable_cells,
        save_directory=save_directory,
        avg_cc=avg_cc,
        cohen_d=cohen_d,
        normalize=normalize,
        bin_centers=bin_centers,
        figsize=(12, 8),
        dpi=dpi,
        file_format=file_format
    )
    
    return saved_files, stats

