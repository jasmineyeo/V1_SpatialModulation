# ReliabilityTesting.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
    
    for cell in range(n_cells):
        # Skip cells with too little activity
        if normalized_activity[cell] < min_activity_threshold:
            continue
            
        # Arrays to store correlation coefficients
        bt_cc_data = np.zeros(n_shuffles)
        bt_cc_rand = np.zeros(n_shuffles)
        
        cell_activity = spatial_activity[cell]
        
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
        # if (avg_bt_cc > min_cc_threshold):  # Large enough effect        
        if (avg_bt_cc > min_cc_threshold and  # Minimum correlation threshold
            avg_bt_cc > shuffle_threshold and  # Better than shuffled
            cohen_d[cell] > cohen_threshold):  # Large enough effect
            reliable_cells[cell] = True
            
    return reliable_cells, average_cc, cohen_d, iterated_cc, normalized_activity

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

