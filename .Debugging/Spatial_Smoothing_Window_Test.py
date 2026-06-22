
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import seaborn as sns

def analyze_spatial_smoothing_on_existing_data(spatial_activity, bin_centers, 
                                            window_range_cm = [1.5, 2, 3, 4, 5, 6, 7, 8, 10], 
                                            reliable_cells=None, bin_size_cm=1):
    """
    Analyze spatial smoothing effects on your existing spatial_activity data.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Your spatial activity matrix (n_cells x n_trials x n_spatial_bins)
        Output from spatial_assignment_with_physical_units
    bin_centers : numpy.ndarray
        Centers of spatial bins in cm
    window_range_cm : list
        Range of smoothing window sizes to test (in cm)
    reliable_cells : numpy.ndarray, optional
        Boolean array indicating reliable cells (if available)
    bin_size_cm : float
        Size of each spatial bin in cm (default: 1)
        
    Returns:
    --------
    optimal_window : float
        Recommended optimal smoothing window size
    smoothed_spatial_activity : numpy.ndarray
        Optimally smoothed spatial activity data
    results : dict
        Analysis results for each window size
    fig : matplotlib.figure.Figure
        Comprehensive analysis plots
    """
    
    print("=== SPATIAL SMOOTHING ANALYSIS ===")
    print(f"Input data shape: {spatial_activity.shape}")
    print(f"Testing smoothing windows: {window_range_cm} cm")
    print(f"Bin size: {bin_size_cm} cm")
    
    results = {}
    
    # Test each smoothing window size
    for window_cm in window_range_cm:
        print(f"  Analyzing {window_cm} cm window...", end=" ")
        
        # Apply spatial smoothing
        smoothed_data = apply_spatial_smoothing(spatial_activity, window_cm=window_cm, bin_size_cm=bin_size_cm)
        
        # Calculate quality metrics
        metrics = calculate_smoothing_quality_metrics(smoothed_data, reliable_cells)
        results[window_cm] = metrics
        
        print(f"SNR: {metrics['signal_to_noise']:.2f}, Sharpness: {metrics['peak_sharpness']:.2f}")
    
    # Determine optimal window
    optimal_window = find_optimal_smoothing_window(results)
    
    # Apply optimal smoothing
    smoothed_spatial_activity = apply_spatial_smoothing(spatial_activity, window_cm=optimal_window, bin_size_cm=bin_size_cm)
    
    # Create visualization
    fig = create_smoothing_analysis_plots(spatial_activity, smoothed_spatial_activity, bin_centers, 
                                        window_range_cm, results, optimal_window, reliable_cells)
    
    print(f"\n✅ RECOMMENDED OPTIMAL SMOOTHING: {optimal_window} cm")
    print(f"   This provides the best balance of noise reduction and spatial precision")
    
    return optimal_window, smoothed_spatial_activity, results, fig

def apply_spatial_smoothing(spatial_activity, window_cm=5, bin_size_cm=1):
    """
    Apply Gaussian spatial smoothing to spatial activity data.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (n_cells x n_trials x n_spatial_bins)
    window_cm : float
        Smoothing window size in cm
    bin_size_cm : float
        Size of each spatial bin in cm
        
    Returns:
    --------
    smoothed_activity : numpy.ndarray
        Spatially smoothed activity with same shape as input
    """
    
    # Convert window size from cm to bins
    window_bins = window_cm / bin_size_cm
    
    # Calculate sigma for Gaussian filter (6-sigma rule)
    sigma = window_bins / 6
    
    n_cells, n_trials, n_bins = spatial_activity.shape
    smoothed_activity = np.zeros_like(spatial_activity)
    
    # Apply smoothing to each cell and trial
    for cell in range(n_cells):
        for trial in range(n_trials):
            smoothed_activity[cell, trial, :] = gaussian_filter1d(
                spatial_activity[cell, trial, :], sigma=sigma
            )
    
    return smoothed_activity

def calculate_smoothing_quality_metrics(smoothed_data, reliable_cells=None):
    """
    Calculate quality metrics for spatially smoothed data.
    
    Parameters:
    -----------
    smoothed_data : numpy.ndarray
        Smoothed spatial activity (n_cells x n_trials x n_spatial_bins)
    reliable_cells : numpy.ndarray, optional
        Boolean mask for reliable cells
        
    Returns:
    --------
    metrics : dict
        Quality metrics dictionary
    """
    
    # Select cells to analyze
    if reliable_cells is not None:
        data_to_analyze = smoothed_data[reliable_cells]
        n_cells = np.sum(reliable_cells)
    else:
        data_to_analyze = smoothed_data
        n_cells = smoothed_data.shape[0]
    
    if n_cells == 0:
        return {
            'signal_to_noise': 0,
            'peak_sharpness': 0,
            'spatial_information': 0,
            'trial_reliability': 0,
            'sparsity': 0,
            'n_cells_analyzed': 0
        }
    
    # Calculate trial-averaged activity
    trial_averaged = np.mean(data_to_analyze, axis=1)  # Shape: (n_cells, n_bins)
    
    # Metric 1: Signal-to-Noise Ratio
    snr_values = []
    for cell in range(n_cells):
        profile = trial_averaged[cell]
        signal = np.max(profile) - np.min(profile)  # Peak-to-trough
        noise = np.std(profile)
        if noise > 0:
            snr_values.append(signal / noise)
    
    # Metric 2: Peak Sharpness (1/FWHM)
    sharpness_values = []
    for cell in range(n_cells):
        profile = trial_averaged[cell]
        if np.max(profile) > np.min(profile):
            peak_val = np.max(profile)
            baseline = np.min(profile)
            half_max = baseline + (peak_val - baseline) / 2
            
            # Find FWHM
            above_half = profile >= half_max
            if np.any(above_half):
                indices = np.where(above_half)[0]
                fwhm = indices[-1] - indices[0] + 1
                sharpness_values.append(1.0 / fwhm if fwhm > 0 else 0)
    
    # Metric 3: Spatial Information Content
    spatial_info_values = []
    for cell in range(n_cells):
        profile = trial_averaged[cell]
        if np.sum(profile) > 0:
            # Normalized occupancy (uniform for simplicity)
            occupancy = np.ones(len(profile)) / len(profile)
            mean_rate = np.sum(profile * occupancy)
            
            if mean_rate > 0:
                info = 0
                for i, rate in enumerate(profile):
                    if rate > 0:
                        rate_ratio = rate / mean_rate
                        info += occupancy[i] * rate * np.log2(rate_ratio)
                spatial_info_values.append(info / mean_rate)
    
    # Metric 4: Trial-to-Trial Reliability
    reliability_values = []
    for cell in range(n_cells):
        cell_trials = data_to_analyze[cell]  # Shape: (n_trials, n_bins)
        n_trials = cell_trials.shape[0]
        
        if n_trials > 1:
            # Calculate pairwise correlations between trials
            correlations = []
            for i in range(n_trials):
                for j in range(i+1, n_trials):
                    corr = np.corrcoef(cell_trials[i], cell_trials[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            if correlations:
                reliability_values.append(np.mean(correlations))
    
    # Metric 5: Sparsity
    sparsity_values = []
    for cell in range(n_cells):
        profile = trial_averaged[cell]
        if np.sum(profile) > 0:
            mean_rate = np.mean(profile)
            mean_squared_rate = np.mean(profile**2)
            n_bins = len(profile)
            
            if mean_squared_rate > 0:
                sparsity = (1 - (mean_rate**2 / mean_squared_rate)) / (1 - 1/n_bins)
                sparsity_values.append(np.clip(sparsity, 0, 1))
    
    # Aggregate metrics
    metrics = {
        'signal_to_noise': np.mean(snr_values) if snr_values else 0,
        'peak_sharpness': np.mean(sharpness_values) if sharpness_values else 0,
        'spatial_information': np.mean(spatial_info_values) if spatial_info_values else 0,
        'trial_reliability': np.mean(reliability_values) if reliability_values else 0,
        'sparsity': np.mean(sparsity_values) if sparsity_values else 0,
        'n_cells_analyzed': n_cells
    }
    
    return metrics

def find_optimal_smoothing_window(results):
    """
    Find optimal smoothing window based on combined metrics.
    
    Parameters:
    -----------
    results : dict
        Results from different smoothing window sizes
        
    Returns:
    --------
    optimal_window : float
        Optimal smoothing window size
    """
    
    windows = list(results.keys())
    
    # Extract and normalize metrics
    snr = np.array([results[w]['signal_to_noise'] for w in windows])
    sharpness = np.array([results[w]['peak_sharpness'] for w in windows])
    information = np.array([results[w]['spatial_information'] for w in windows])
    reliability = np.array([results[w]['trial_reliability'] for w in windows])
    
    # Normalize to 0-1 range
    def normalize(arr):
        if np.max(arr) > np.min(arr):
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        return np.ones_like(arr)
    
    snr_norm = normalize(snr)
    sharpness_norm = normalize(sharpness)
    info_norm = normalize(information)
    reliability_norm = normalize(reliability)
    
    # Combined score with weights
    # Emphasize SNR and reliability for neural data
    combined_score = (0.3 * snr_norm + 
                    0.2 * sharpness_norm + 
                    0.2 * info_norm + 
                    0.3 * reliability_norm)
    
    # Find optimal
    optimal_idx = np.argmax(combined_score)
    optimal_window = windows[optimal_idx]
    
    return optimal_window

def create_smoothing_analysis_plots(original_data, smoothed_data, bin_centers, 
                                window_range_cm, results, optimal_window, reliable_cells=None):
    """
    Create comprehensive visualization of smoothing analysis.
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Metrics comparison
    ax = axes[0, 0]
    windows = list(results.keys())
    
    metrics = ['signal_to_noise', 'peak_sharpness', 'spatial_information', 'trial_reliability']
    colors = ['blue', 'red', 'green', 'orange']
    
    for metric, color in zip(metrics, colors):
        values = [results[w][metric] for w in windows]
        ax.plot(windows, values, 'o-', color=color, linewidth=2, markersize=6, 
                label=metric.replace('_', ' ').title())
    
    ax.axvline(optimal_window, color='black', linestyle='--', linewidth=2, alpha=0.8, 
            label=f'Optimal: {optimal_window}cm')
    ax.set_xlabel('Smoothing Window (cm)')
    ax.set_ylabel('Metric Value')
    ax.set_title('Quality Metrics vs Smoothing Window')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Example cell comparison
    ax = axes[0, 1]
    
    # Select a cell with good activity for demonstration
    if reliable_cells is not None and np.any(reliable_cells):
        example_cell = np.where(reliable_cells)[0][0]
    else:
        # Find cell with highest activity
        mean_activity = np.mean(original_data, axis=(1, 2))
        example_cell = np.argmax(mean_activity)
    
    # Show original vs different smoothing levels
    trial_avg_original = np.mean(original_data[example_cell], axis=0)
    trial_avg_smoothed = np.mean(smoothed_data[example_cell], axis=0)
    
    # Also show a heavily smoothed version for comparison
    heavy_smoothed = apply_spatial_smoothing(original_data, window_cm=10)
    trial_avg_heavy = np.mean(heavy_smoothed[example_cell], axis=0)
    
    ax.plot(bin_centers, trial_avg_original, 'k--', alpha=0.6, linewidth=1, label='Original')
    ax.plot(bin_centers, trial_avg_smoothed, 'b-', linewidth=3, label=f'Optimal ({optimal_window}cm)')
    ax.plot(bin_centers, trial_avg_heavy, 'r:', linewidth=2, alpha=0.7, label='Heavy (10cm)')
    
    ax.set_xlabel('Position (cm)')
    ax.set_ylabel('Activity')
    ax.set_title(f'Example Cell {example_cell}: Smoothing Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Combined score
    ax = axes[0, 2]
    
    # Calculate combined scores for visualization
    windows = list(results.keys())
    combined_scores = []
    
    for w in windows:
        # Simple combined score for visualization
        score = (results[w]['signal_to_noise'] + 
                results[w]['trial_reliability'] + 
                results[w]['spatial_information']) / 3
        combined_scores.append(score)
    
    bars = ax.bar(range(len(windows)), combined_scores, color='lightblue', edgecolor='black')
    
    # Highlight optimal
    optimal_idx = windows.index(optimal_window)
    bars[optimal_idx].set_color('red')
    bars[optimal_idx].set_alpha(0.8)
    
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f'{w}cm' for w in windows])
    ax.set_xlabel('Smoothing Window')
    ax.set_ylabel('Combined Score')
    ax.set_title('Combined Quality Score')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Signal-to-noise comparison
    ax = axes[1, 0]
    
    snr_values = [results[w]['signal_to_noise'] for w in windows]
    ax.plot(windows, snr_values, 'bo-', linewidth=2, markersize=8)
    ax.axvline(optimal_window, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Smoothing Window (cm)')
    ax.set_ylabel('Signal-to-Noise Ratio')
    ax.set_title('Signal-to-Noise vs Smoothing')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Peak sharpness
    ax = axes[1, 1]
    
    sharpness_values = [results[w]['peak_sharpness'] for w in windows]
    ax.plot(windows, sharpness_values, 'ro-', linewidth=2, markersize=8)
    ax.axvline(optimal_window, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Smoothing Window (cm)')
    ax.set_ylabel('Peak Sharpness (1/FWHM)')
    ax.set_title('Peak Sharpness vs Smoothing')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Trial reliability
    ax = axes[1, 2]
    
    reliability_values = [results[w]['trial_reliability'] for w in windows]
    ax.plot(windows, reliability_values, 'go-', linewidth=2, markersize=8)
    ax.axvline(optimal_window, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Smoothing Window (cm)')
    ax.set_ylabel('Trial-to-Trial Reliability')
    ax.set_title('Reliability vs Smoothing')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Print summary
    print(f"\n📊 SMOOTHING ANALYSIS SUMMARY:")
    print(f"   Optimal window: {optimal_window} cm")
    print(f"   Final metrics with optimal smoothing:")
    optimal_metrics = results[optimal_window]
    for metric, value in optimal_metrics.items():
        if metric != 'n_cells_analyzed':
            print(f"     {metric.replace('_', ' ').title()}: {value:.3f}")
    print(f"     Cells analyzed: {optimal_metrics['n_cells_analyzed']}")
    
    return fig

# Simple usage function for your workflow
def smooth_my_spatial_activity(spatial_activity, bin_centers, reliable_cells=None):
    """
    Simple function to optimally smooth your spatial_activity data.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Your spatial activity from spatial_assignment_with_physical_units
    bin_centers : numpy.ndarray  
        Your bin centers in cm
    reliable_cells : numpy.ndarray, optional
        Boolean array of reliable cells (if you have it)
        
    Returns:
    --------
    smoothed_spatial_activity : numpy.ndarray
        Optimally smoothed data ready for reliability testing
    """
    
    print("🔄 Running spatial smoothing optimization...")
    
    optimal_window, smoothed_data, results, fig = analyze_spatial_smoothing_on_existing_data(
        spatial_activity, bin_centers, reliable_cells=reliable_cells
    )
    
    plt.show()
    
    print(f"✅ Spatial smoothing complete!")
    print(f"   Using {optimal_window} cm Gaussian smoothing")
    print(f"   Ready for reliability testing and SMI calculation")
    
    return smoothed_data

# optimal_window, smoothed_spatial_activity, results, fig = analyze_spatial_smoothing_on_existing_data(
#     spatial_activity, 
#     bin_centers,
#     reliable_cells=None  # Add your reliable_cells array if you have it
# )
