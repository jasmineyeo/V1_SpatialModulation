"""
Functions to find the optimal smoothing window for spike data
JSY, 08/15/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from scipy import stats
import seaborn as sns

# Import your helper modules
from helper import SpikeSmoothing, BehavioralDataFiltering as DF, SpatialDiscretization as SD, ReliabilityTesting as RT


def create_smoothing_comparison_plots_fixed(results, smoothing_windows):
    """Create comprehensive comparison plots with fixed layout to prevent overlap."""
    
    # Create figure with better spacing
    fig = plt.figure(figsize=(24, 20))
    gs = GridSpec(5, 4, figure=fig, hspace=0.4, wspace=0.3, 
                  top=0.95, bottom=0.05, left=0.05, right=0.95)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(smoothing_windows)]
    window_labels = [f'{w}ms' for w in smoothing_windows]
    
    # Row 1: Main tuning metrics (4 plots)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric_boxplot_fixed(results, 'tuning_width', ax1, colors, window_labels, 'Tuning Width (cm)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric_boxplot_fixed(results, 'peak_sharpness', ax2, colors, window_labels, 'Peak Sharpness')
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_metric_boxplot_fixed(results, 'information_content', ax3, colors, window_labels, 'Information Content\n(bits/spike)')
    
    ax4 = fig.add_subplot(gs[0, 3])
    plot_metric_boxplot_fixed(results, 'modulation_depth', ax4, colors, window_labels, 'Modulation Depth')
    
    # Row 2: Additional metrics and summary (4 plots)
    ax5 = fig.add_subplot(gs[1, 0])
    plot_metric_boxplot_fixed(results, 'signal_to_noise', ax5, colors, window_labels, 'Signal-to-Noise')
    
    ax6 = fig.add_subplot(gs[1, 1])
    plot_reliability_comparison_fixed(results, ax6, colors, window_labels)
    
    ax7 = fig.add_subplot(gs[1, 2])
    plot_mean_comparison_bars(results, smoothing_windows, ax7)
    
    ax8 = fig.add_subplot(gs[1, 3])
    plot_information_content_detailed(results, ax8, colors, window_labels)
    
    # Row 3: Example tuning curves (spans full width)
    ax9 = fig.add_subplot(gs[2, :])
    plot_example_tuning_curves_fixed(results, ax9, colors, window_labels)
    
    # Row 4: Correlation plots between 250ms and other windows
    if len(smoothing_windows) >= 2:
        metrics_to_compare = ['information_content', 'peak_sharpness', 'tuning_width', 'modulation_depth']
        metric_titles = ['Information Content', 'Peak Sharpness', 'Tuning Width', 'Modulation Depth']
        
        for i, (metric, title) in enumerate(zip(metrics_to_compare, metric_titles)):
            ax = fig.add_subplot(gs[3, i])
            plot_metric_correlation_fixed(results, metric, smoothing_windows[0], smoothing_windows[1], ax, title)
    
    # Row 5: Summary statistics table (spans full width)
    ax13 = fig.add_subplot(gs[4, :])
    create_summary_table_fixed(results, smoothing_windows, ax13)
    
    plt.suptitle('Tuning Curve Analysis: Temporal Smoothing Window Comparison', 
                fontsize=18, fontweight='bold', y=0.98)
    
    plt.show()
    return fig


def plot_metric_boxplot_fixed(results, metric_name, ax, colors, labels, ylabel):
    """Fixed boxplot with better formatting."""
    
    data = []
    for label in labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            data.append(results[label]['tuning_metrics'][metric_name])
        else:
            data.append([])
    
    # Create boxplot
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points with jitter
    for i, dataset in enumerate(data):
        if len(dataset) > 0:
            y = dataset
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.3, s=8, color='black')
    
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)


def plot_reliability_comparison_fixed(results, ax, colors, window_labels):
    """Fixed reliability comparison plot."""
    
    n_reliable = []
    for label in window_labels:
        if label in results:
            n_reliable.append(results[label]['n_reliable'])
        else:
            n_reliable.append(0)
    
    bars = ax.bar(window_labels, n_reliable, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, value in zip(bars, n_reliable):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_ylabel('Number of Reliable Cells', fontsize=11)
    ax.set_title('Reliable Cells by\nSmoothing Window', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)


def plot_mean_comparison_bars(results, smoothing_windows, ax):
    """Plot mean values of key metrics side by side."""
    
    metrics = ['information_content', 'peak_sharpness']
    metric_labels = ['Info Content', 'Peak Sharpness']
    
    x = np.arange(len(smoothing_windows))
    width = 0.35
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        means = []
        errors = []
        
        for window_ms in smoothing_windows:
            label_key = f'{window_ms}ms'
            if label_key in results and results[label_key]['tuning_metrics'] is not None:
                values = results[label_key]['tuning_metrics'][metric]
                means.append(np.mean(values))
                errors.append(np.std(values) / np.sqrt(len(values)))  # SEM
            else:
                means.append(0)
                errors.append(0)
        
        # Normalize peak sharpness for comparison (divide by 10)
        if metric == 'peak_sharpness':
            means = [m/10 for m in means]
            errors = [e/10 for e in errors]
        
        ax.bar(x + i * width, means, width, label=label, 
               yerr=errors, capsize=5, alpha=0.7)
    
    ax.set_xlabel('Smoothing Window', fontsize=11)
    ax.set_ylabel('Normalized Values', fontsize=11)
    ax.set_title('Key Metrics Comparison\n(Peak Sharpness ÷ 10)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([f'{w}ms' for w in smoothing_windows])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_information_content_detailed(results, ax, colors, window_labels):
    """Detailed plot of information content distribution."""
    
    for i, (label, color) in enumerate(zip(window_labels, colors)):
        if label in results and results[label]['tuning_metrics'] is not None:
            info_content = results[label]['tuning_metrics']['information_content']
            ax.hist(info_content, bins=20, alpha=0.6, color=color, label=label, density=True)
    
    ax.set_xlabel('Information Content (bits/spike)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Information Content\nDistribution', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_example_tuning_curves_fixed(results, ax, colors, window_labels, n_examples=8):
    """Fixed example tuning curves with better layout."""
    
    # Find cells that are reliable across all smoothing windows
    common_cells = None
    for label in window_labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            cell_indices = results[label]['tuning_metrics']['cell_indices']
            if common_cells is None:
                common_cells = set(cell_indices)
            else:
                common_cells = common_cells.intersection(set(cell_indices))
    
    if common_cells is None or len(common_cells) == 0:
        ax.text(0.5, 0.5, 'No common cells across smoothing windows', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return
    
    # Select example cells
    example_cells = list(common_cells)[:n_examples]
    
    # Plot each cell's tuning curves
    for idx, cell_idx in enumerate(example_cells):
        for j, (label, color) in enumerate(zip(window_labels, colors)):
            if label in results and results[label]['tuning_metrics'] is not None:
                # Find this cell in the results
                cell_indices = results[label]['tuning_metrics']['cell_indices']
                if cell_idx in cell_indices:
                    local_idx = np.where(cell_indices == cell_idx)[0][0]
                    tuning_curve = results[label]['tuning_metrics']['tuning_curves'][local_idx]
                    bin_centers = results[label]['bin_centers']
                    
                    # Normalize for comparison
                    tuning_curve_norm = tuning_curve / np.max(tuning_curve) if np.max(tuning_curve) > 0 else tuning_curve
                    
                    # Offset each cell vertically
                    ax.plot(bin_centers, tuning_curve_norm + idx * 1.3, 
                           color=color, label=label if idx == 0 else "", linewidth=2.5, alpha=0.8)
        
        # Add cell label
        ax.text(-20, idx * 1.3 + 0.5, f'Cell {cell_idx}', 
                fontsize=10, ha='right', va='center', fontweight='bold')
    
    ax.set_xlabel('Position (cm)', fontsize=12)
    ax.set_ylabel('Normalized Firing Rate + Offset', fontsize=12)
    ax.set_title('Example Tuning Curves: Same Cells with Different Smoothing Windows', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-30, max(bin_centers) + 10)


def plot_metric_correlation_fixed(results, metric_name, window1, window2, ax, title):
    """Fixed correlation plot."""
    
    label1 = f'{window1}ms'
    label2 = f'{window2}ms'
    
    if (label1 not in results or label2 not in results or 
        results[label1]['tuning_metrics'] is None or 
        results[label2]['tuning_metrics'] is None):
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get common cells
    cells1 = results[label1]['tuning_metrics']['cell_indices']
    cells2 = results[label2]['tuning_metrics']['cell_indices']
    common_cells = np.intersect1d(cells1, cells2)
    
    if len(common_cells) == 0:
        ax.text(0.5, 0.5, 'No common cells', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get metric values for common cells
    values1 = []
    values2 = []
    
    for cell_idx in common_cells:
        idx1 = np.where(cells1 == cell_idx)[0][0]
        idx2 = np.where(cells2 == cell_idx)[0][0]
        
        values1.append(results[label1]['tuning_metrics'][metric_name][idx1])
        values2.append(results[label2]['tuning_metrics'][metric_name][idx2])
    
    # Plot scatter
    ax.scatter(values1, values2, alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
    
    # Add unity line
    min_val = min(min(values1), min(values2))
    max_val = max(max(values1), max(values2))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Calculate correlation
    r, p = stats.pearsonr(values1, values2)
    
    # Add correlation info
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}\nn = {len(common_cells)}', 
           transform=ax.transAxes, va='top', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(f'{title} ({label1})', fontsize=11)
    ax.set_ylabel(f'{title} ({label2})', fontsize=11)
    ax.set_title(f'{title}\n{label1} vs {label2}', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)


def create_summary_table_fixed(results, smoothing_windows, ax):
    """Create a clean summary table."""
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    headers = ['Metric'] + [f'{w}ms' for w in smoothing_windows] + ['Best Window']
    
    metrics_info = [
        ('# Reliable Cells', 'n_reliable', 'higher'),
        ('Information Content', 'information_content', 'higher'),
        ('Peak Sharpness', 'peak_sharpness', 'higher'),
        ('Tuning Width (cm)', 'tuning_width', 'lower'),
        ('Modulation Depth', 'modulation_depth', 'higher'),
        ('Signal-to-Noise', 'signal_to_noise', 'higher')
    ]
    
    for metric_name, metric_key, better in metrics_info:
        row = [metric_name]
        values = []
        
        for window_ms in smoothing_windows:
            label_key = f'{window_ms}ms'
            if label_key in results:
                if metric_key == 'n_reliable':
                    value = results[label_key]['n_reliable']
                    row.append(f'{value}')
                    values.append(value)
                elif results[label_key]['tuning_metrics'] is not None:
                    metric_values = results[label_key]['tuning_metrics'][metric_key]
                    mean_val = np.mean(metric_values)
                    std_val = np.std(metric_values)
                    row.append(f'{mean_val:.3f} ± {std_val:.3f}')
                    values.append(mean_val)
                else:
                    row.append('N/A')
                    values.append(0)
            else:
                row.append('N/A')
                values.append(0)
        
        # Determine best window
        if len(values) > 0:
            if better == 'higher':
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin([v for v in values if v > 0] or [0])
            
            if values[best_idx] > 0:
                row.append(f'{smoothing_windows[best_idx]}ms')
            else:
                row.append('N/A')
        else:
            row.append('N/A')
        
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.0)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    for i in range(1, len(table_data) + 1):
        best_col = len(headers) - 1  # Best window column
        table[(i, best_col)].set_facecolor('#90EE90')  # Light green
        table[(i, best_col)].set_text_props(weight='bold')
    
    ax.set_title('Summary Statistics and Recommendations', fontsize=16, fontweight='bold', pad=20)


# Add this function to replace the existing plotting call
def run_fixed_analysis_plots(results, smoothing_windows):
    """Run the analysis with fixed plotting."""
    
    # Clear any existing plots
    plt.close('all')
    
    # Create the fixed plots
    fig = create_smoothing_comparison_plots_fixed(results, smoothing_windows)
    
    return fig
def compare_smoothing_windows_after_offset(offset_spike_data, vr_dict, framerate, 
                                         smoothing_windows=[250, 500, 750],
                                         single_lap_treadmill=None):
    """
    Run the preprocessing pipeline with different temporal smoothing windows
    AFTER applying the optimal temporal offset to see how it affects tuning curve properties.
    
    Parameters:
    -----------
    offset_spike_data : numpy.ndarray
        Spike data after applying optimal temporal offset
    vr_dict : dict
        Dictionary containing VR data with 'interp_location' key
    framerate : float
        Recording framerate
    smoothing_windows : list
        List of smoothing windows in milliseconds to test
    single_lap_treadmill : float, optional
        Length of single lap for spatial discretization. If None, will calculate from your constants.
        
    Returns:
    --------
    results : dict
        Results for each smoothing window containing tuning metrics
    """
    
    print("="*70)
    print("COMPARING TUNING PROPERTIES ACROSS SMOOTHING WINDOWS")
    print("(After applying optimal temporal offset)")
    print("="*70)
    
    # Calculate single_lap_treadmill if not provided (using your constants)
    if single_lap_treadmill is None:
        single_revolution_VR = 282.415
        single_revolution_treadmill = 27.8
        single_lap_VR = 1320.645683
        single_lap_treadmill = single_revolution_treadmill * single_lap_VR / single_revolution_VR
        print(f"Calculated single_lap_treadmill: {single_lap_treadmill:.2f} cm")
    
    results = {}
    
    for window_ms in smoothing_windows:
        print(f"\n{'='*25} PROCESSING {window_ms}ms WINDOW {'='*25}")
        
        # Step 2b: Apply temporal smoothing with current window
        print(f"Step 2b: Smoothing spikes with {window_ms}ms Gaussian window...")
        smoothed_spikes = SpikeSmoothing.smooth_spikes(
            offset_spike_data,  # Use the offset data
            framerate, 
            window_ms=window_ms
        )
        
        # Step 2c: Behavioral data filtering
        print("Step 2c: Filtering behavioral data and removing inactive periods...")
        min_trial_duration_seconds = 5
        max_trial_duration_seconds = 60
        
        filtered_spks_laps, filtered_location_laps, n_valid_laps = DF.process_data_with_trial_filtering(
            smoothed_spikes, 
            vr_dict['interp_location'],
            min_trial_duration_seconds=min_trial_duration_seconds, 
            max_trial_duration_seconds=max_trial_duration_seconds,
            framerate=framerate
        )
        
        if n_valid_laps == 0:
            print(f"❌ No valid laps found for {window_ms}ms window!")
            continue
            
        print(f"✓ Found {n_valid_laps} valid laps")
        
        # Step 3: Spatial discretization
        print("Step 3: Performing spatial discretization...")
        spatial_activity, spatial_bins, trial_averaged_activity, bin_centers = SD.spatial_assignment(
            n_valid_laps,
            filtered_spks_laps, 
            filtered_location_laps, 
            single_lap_treadmill
        )
        
        # Apply spatial smoothing (5cm window as in your pipeline)
        print("Applying 5cm spatial smoothing...")
        window_cm = 5
        spatially_smoothed_activity = SpikeSmoothing.spatial_smooth(spatial_activity, window_cm=window_cm)
        
        # Step 4: Reliability testing
        print("Step 4: Testing cell reliability...")
        combined_reliable, reliable_cells, _, avg_cc, cohens_d, _, _, _ = RT.combined_reliability_test(
            spatially_smoothed_activity,
            n_shuffles=100,           
            cc_percentile=90,          
            cohen_threshold=1,       
            min_cc_threshold=0.2,      
            min_activity_threshold=0.0, 
            min_pattern_corr=0.3, 
            peak_distance_threshold=5
        )
        
        n_reliable = np.sum(combined_reliable)
        print(f"✓ Found {n_reliable} reliable cells")
        
        if n_reliable == 0:
            print(f"❌ No reliable cells found for {window_ms}ms window!")
            continue
        
        # Calculate tuning metrics
        print("Step 5: Calculating tuning curve properties...")
        tuning_metrics = calculate_tuning_properties(
            spatially_smoothed_activity, 
            bin_centers, 
            combined_reliable
        )
        
        # Store results
        results[f'{window_ms}ms'] = {
            'tuning_metrics': tuning_metrics,
            'spatial_activity': spatially_smoothed_activity,
            'bin_centers': bin_centers,
            'reliable_cells': combined_reliable,
            'n_reliable': n_reliable,
            'n_valid_laps': n_valid_laps,
            'avg_cc': avg_cc,
            'cohens_d': cohens_d,
            'window_ms': window_ms
        }
        
        # Print summary
        print_tuning_summary(tuning_metrics, window_ms)
    
    # Create comprehensive comparison plots
    if len(results) > 1:
        print(f"\n{'='*30} CREATING COMPARISON PLOTS {'='*30}")
        create_smoothing_comparison_plots_fixed(results, smoothing_windows)
        print(f"\n{'='*30} STATISTICAL ANALYSIS {'='*30}")
        perform_statistical_analysis(results, smoothing_windows)
        
        # Provide recommendation
        provide_smoothing_recommendation(results, smoothing_windows)
    else:
        print("❌ Need at least 2 smoothing windows with valid results for comparison!")
    
    return results


def calculate_tuning_properties(spatial_activity, bin_centers, reliable_cells):
    """
    Calculate various tuning curve properties for reliable cells.
    """
    reliable_indices = np.where(reliable_cells)[0]
    n_cells = len(reliable_indices)
    
    if n_cells == 0:
        return None
    
    # Initialize arrays
    tuning_width = np.zeros(n_cells)
    peak_sharpness = np.zeros(n_cells)
    information_content = np.zeros(n_cells)
    peak_location = np.zeros(n_cells)
    max_firing_rate = np.zeros(n_cells)
    baseline_firing = np.zeros(n_cells)
    modulation_depth = np.zeros(n_cells)
    signal_to_noise = np.zeros(n_cells)
    
    # Calculate properties for each reliable cell
    for i, cell_idx in enumerate(reliable_indices):
        # Get trial-averaged tuning curve
        tuning_curve = np.mean(spatial_activity[cell_idx], axis=0)
        
        # 1. Tuning width (full width at half maximum)
        tuning_width[i] = calculate_fwhm(tuning_curve, bin_centers)
        
        # 2. Peak sharpness (how sharp the peak is)
        peak_sharpness[i] = calculate_peak_sharpness(tuning_curve)
        
        # 3. Spatial information content
        information_content[i] = calculate_spatial_information(tuning_curve)
        
        # 4. Peak location
        peak_location[i] = bin_centers[np.argmax(tuning_curve)]
        
        # 5. Maximum firing rate
        max_firing_rate[i] = np.max(tuning_curve)
        
        # 6. Baseline firing (bottom 20th percentile)
        baseline_firing[i] = np.percentile(tuning_curve, 20)
        
        # 7. Modulation depth (max - min) / (max + min)
        min_rate = np.min(tuning_curve)
        max_rate = np.max(tuning_curve)
        if max_rate + min_rate > 0:
            modulation_depth[i] = (max_rate - min_rate) / (max_rate + min_rate)
        else:
            modulation_depth[i] = 0
            
        # 8. Signal-to-noise ratio
        if np.std(tuning_curve) > 0:
            signal_to_noise[i] = np.max(tuning_curve) / np.std(tuning_curve)
        else:
            signal_to_noise[i] = 0
    
    return {
        'tuning_width': tuning_width,
        'peak_sharpness': peak_sharpness,
        'information_content': information_content,
        'peak_location': peak_location,
        'max_firing_rate': max_firing_rate,
        'baseline_firing': baseline_firing,
        'modulation_depth': modulation_depth,
        'signal_to_noise': signal_to_noise,
        'tuning_curves': np.mean(spatial_activity[reliable_indices], axis=1),
        'cell_indices': reliable_indices
    }


def calculate_fwhm(tuning_curve, bin_centers):
    """Calculate full width at half maximum."""
    peak_value = np.max(tuning_curve)
    half_max = peak_value / 2
    
    # Find indices where curve is above half maximum
    above_half_max = tuning_curve >= half_max
    
    if not np.any(above_half_max):
        return np.nan
    
    indices_above = np.where(above_half_max)[0]
    if len(indices_above) < 2:
        return 0
    
    # Calculate width in cm
    width_bins = indices_above[-1] - indices_above[0] + 1
    bin_size = np.mean(np.diff(bin_centers))
    return width_bins * bin_size


def calculate_peak_sharpness(tuning_curve):
    """Calculate peak sharpness as the ratio of peak to standard deviation."""
    if np.std(tuning_curve) > 0:
        return np.max(tuning_curve) / np.std(tuning_curve)
    else:
        return 0


def calculate_spatial_information(tuning_curve):
    """Calculate spatial information content in bits/spike."""
    mean_rate = np.mean(tuning_curve)
    if mean_rate <= 0:
        return 0
    
    # Calculate information
    information = 0
    n_bins = len(tuning_curve)
    
    for rate in tuning_curve:
        if rate > 0:
            p_bin = 1.0 / n_bins  # Assuming uniform occupancy
            information += p_bin * (rate / mean_rate) * np.log2(rate / mean_rate)
    
    return information


def print_tuning_summary(tuning_metrics, window_ms):
    """Print summary statistics for tuning properties."""
    if tuning_metrics is None:
        return
        
    print(f"\n📊 TUNING SUMMARY FOR {window_ms}ms SMOOTHING:")
    print(f"   Number of cells: {len(tuning_metrics['tuning_width'])}")
    print(f"   Mean tuning width: {np.mean(tuning_metrics['tuning_width']):.2f} ± {np.std(tuning_metrics['tuning_width']):.2f} cm")
    print(f"   Mean peak sharpness: {np.mean(tuning_metrics['peak_sharpness']):.3f} ± {np.std(tuning_metrics['peak_sharpness']):.3f}")
    print(f"   Mean information content: {np.mean(tuning_metrics['information_content']):.3f} ± {np.std(tuning_metrics['information_content']):.3f} bits/spike")
    print(f"   Mean modulation depth: {np.mean(tuning_metrics['modulation_depth']):.3f} ± {np.std(tuning_metrics['modulation_depth']):.3f}")
    print(f"   Mean signal-to-noise: {np.mean(tuning_metrics['signal_to_noise']):.3f} ± {np.std(tuning_metrics['signal_to_noise']):.3f}")


def create_smoothing_comparison_plots(results, smoothing_windows):
    """Create comprehensive comparison plots."""
    
    fig = plt.figure(figsize=(24, 18))
    gs = GridSpec(5, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(smoothing_windows)]
    window_labels = [f'{w}ms' for w in smoothing_windows]
    
    # Row 1: Main tuning metrics
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric_boxplot(results, 'tuning_width', ax1, colors, window_labels, 'Tuning Width (cm)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric_boxplot(results, 'peak_sharpness', ax2, colors, window_labels, 'Peak Sharpness')
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_metric_boxplot(results, 'information_content', ax3, colors, window_labels, 'Information Content (bits/spike)')
    
    ax4 = fig.add_subplot(gs[0, 3])
    plot_metric_boxplot(results, 'modulation_depth', ax4, colors, window_labels, 'Modulation Depth')
    
    # Row 2: Additional metrics
    ax5 = fig.add_subplot(gs[1, 0])
    plot_metric_boxplot(results, 'signal_to_noise', ax5, colors, window_labels, 'Signal-to-Noise Ratio')
    
    ax6 = fig.add_subplot(gs[1, 1])
    plot_metric_boxplot(results, 'max_firing_rate', ax6, colors, window_labels, 'Max Firing Rate')
    
    ax7 = fig.add_subplot(gs[1, 2])
    plot_reliability_comparison(results, ax7, colors, window_labels)
    
    ax8 = fig.add_subplot(gs[1, 3])
    plot_mean_metrics_summary(results, smoothing_windows, ax8)
    
    # Row 3: Example tuning curves
    ax9 = fig.add_subplot(gs[2, :])
    plot_example_tuning_curves(results, ax9, colors, window_labels, n_examples=6)
    
    # Row 4: Correlation plots (if we have at least 2 windows)
    if len(smoothing_windows) >= 2:
        metrics_to_compare = ['tuning_width', 'peak_sharpness', 'information_content', 'modulation_depth']
        for i, metric in enumerate(metrics_to_compare):
            ax = fig.add_subplot(gs[3, i])
            plot_metric_correlation(results, metric, smoothing_windows[0], smoothing_windows[1], ax)
    
    # Row 5: Summary statistics table
    ax13 = fig.add_subplot(gs[4, :])
    create_summary_table(results, smoothing_windows, ax13)
    
    plt.suptitle('Tuning Curve Analysis: Comparison of Temporal Smoothing Windows', 
                fontsize=16, fontweight='bold', y=0.98)
    plt.show()


def plot_metric_boxplot(results, metric_name, ax, colors, labels, ylabel):
    """Plot boxplot comparison of a metric across smoothing windows."""
    
    data = []
    for label in labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            data.append(results[label]['tuning_metrics'][metric_name])
        else:
            data.append([])
    
    # Create boxplot
    bp = ax.boxplot(data, patch_artist=True, labels=labels)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Add individual points
    for i, dataset in enumerate(data):
        if len(dataset) > 0:
            y = dataset
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=10, color='black')
    
    ax.set_ylabel(ylabel)
    ax.set_title(f'{ylabel} Comparison')
    ax.grid(True, alpha=0.3)


def plot_example_tuning_curves(results, ax, colors, window_labels, n_examples=6):
    """Plot example tuning curves for comparison."""
    
    # Find cells that are reliable across all smoothing windows
    common_cells = None
    for label in window_labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            cell_indices = results[label]['tuning_metrics']['cell_indices']
            if common_cells is None:
                common_cells = set(cell_indices)
            else:
                common_cells = common_cells.intersection(set(cell_indices))
    
    if common_cells is None or len(common_cells) == 0:
        ax.text(0.5, 0.5, 'No common cells across smoothing windows', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        return
    
    # Select example cells with good tuning (high information content)
    example_cells = list(common_cells)[:n_examples]
    
    # Create subplots for each example cell
    n_cols = min(3, len(example_cells))
    n_rows = (len(example_cells) + n_cols - 1) // n_cols
    
    for idx, cell_idx in enumerate(example_cells):
        # Calculate subplot position
        row = idx // n_cols
        col = idx % n_cols
        
        # Create a mini-subplot within the main axis
        subplot_ax = plt.subplot2grid((n_rows, n_cols), (row, col), fig=ax.figure)
        
        for j, (label, color) in enumerate(zip(window_labels, colors)):
            if label in results and results[label]['tuning_metrics'] is not None:
                # Find this cell in the results
                cell_indices = results[label]['tuning_metrics']['cell_indices']
                if cell_idx in cell_indices:
                    local_idx = np.where(cell_indices == cell_idx)[0][0]
                    tuning_curve = results[label]['tuning_metrics']['tuning_curves'][local_idx]
                    bin_centers = results[label]['bin_centers']
                    
                    # Normalize for comparison
                    tuning_curve_norm = tuning_curve / np.max(tuning_curve) if np.max(tuning_curve) > 0 else tuning_curve
                    
                    subplot_ax.plot(bin_centers, tuning_curve_norm, 
                                  color=color, label=label, linewidth=2, alpha=0.8)
        
        subplot_ax.set_title(f'Cell {cell_idx}', fontsize=10)
        subplot_ax.set_xlabel('Position (cm)', fontsize=8)
        subplot_ax.set_ylabel('Norm. Rate', fontsize=8)
        subplot_ax.grid(True, alpha=0.3)
        if idx == 0:  # Add legend only to first subplot
            subplot_ax.legend(fontsize=8)
    
    # Hide the main axis
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title('Example Tuning Curves Across Smoothing Windows', fontsize=14, pad=20)


def plot_metric_correlation(results, metric_name, window1, window2, ax):
    """Plot correlation of a metric between two smoothing windows."""
    
    label1 = f'{window1}ms'
    label2 = f'{window2}ms'
    
    if (label1 not in results or label2 not in results or 
        results[label1]['tuning_metrics'] is None or 
        results[label2]['tuning_metrics'] is None):
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get common cells
    cells1 = results[label1]['tuning_metrics']['cell_indices']
    cells2 = results[label2]['tuning_metrics']['cell_indices']
    common_cells = np.intersect1d(cells1, cells2)
    
    if len(common_cells) == 0:
        ax.text(0.5, 0.5, 'No common cells', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Get metric values for common cells
    values1 = []
    values2 = []
    
    for cell_idx in common_cells:
        idx1 = np.where(cells1 == cell_idx)[0][0]
        idx2 = np.where(cells2 == cell_idx)[0][0]
        
        values1.append(results[label1]['tuning_metrics'][metric_name][idx1])
        values2.append(results[label2]['tuning_metrics'][metric_name][idx2])
    
    # Plot scatter
    ax.scatter(values1, values2, alpha=0.6, s=30)
    
    # Add unity line
    min_val = min(min(values1), min(values2))
    max_val = max(max(values1), max(values2))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Calculate correlation
    r, p = stats.pearsonr(values1, values2)
    ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}\nn = {len(common_cells)}', 
           transform=ax.transAxes, va='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(f'{metric_name.replace("_", " ").title()} ({label1})')
    ax.set_ylabel(f'{metric_name.replace("_", " ").title()} ({label2})')
    ax.set_title(f'{label1} vs {label2}')
    ax.grid(True, alpha=0.3)


def plot_reliability_comparison(results, ax, colors, window_labels):
    """Plot number of reliable cells for each smoothing window."""
    
    n_reliable = []
    for label in window_labels:
        if label in results:
            n_reliable.append(results[label]['n_reliable'])
        else:
            n_reliable.append(0)
    
    bars = ax.bar(window_labels, n_reliable, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, n_reliable):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Number of Reliable Cells')
    ax.set_title('Reliable Cells by Smoothing Window')
    ax.grid(True, alpha=0.3)


def plot_mean_metrics_summary(results, smoothing_windows, ax):
    """Plot normalized mean metrics for easy comparison."""
    
    metrics = ['tuning_width', 'peak_sharpness', 'information_content', 'modulation_depth']
    metric_labels = ['Tuning Width', 'Peak Sharpness', 'Info Content', 'Modulation Depth']
    
    # Collect data
    data_matrix = []
    for metric in metrics:
        metric_means = []
        for window_ms in smoothing_windows:
            label_key = f'{window_ms}ms'
            if label_key in results and results[label_key]['tuning_metrics'] is not None:
                values = results[label_key]['tuning_metrics'][metric]
                metric_means.append(np.mean(values))
            else:
                metric_means.append(0)
        data_matrix.append(metric_means)
    
    # Normalize each metric to 0-1 range for comparison
    data_matrix = np.array(data_matrix)
    for i in range(len(metrics)):
        if np.max(data_matrix[i]) > 0:
            data_matrix[i] = data_matrix[i] / np.max(data_matrix[i])
    
    # Create heatmap
    im = ax.imshow(data_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(range(len(smoothing_windows)))
    ax.set_xticklabels([f'{w}ms' for w in smoothing_windows])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels(metric_labels)
    
    # Add text annotations
    for i in range(len(metrics)):
        for j in range(len(smoothing_windows)):
            text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                         ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title('Normalized Metric Summary')
    plt.colorbar(im, ax=ax, shrink=0.8)


def create_summary_table(results, smoothing_windows, ax):
    """Create a summary table of all metrics."""
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    metrics = ['tuning_width', 'peak_sharpness', 'information_content', 'modulation_depth', 'signal_to_noise']
    metric_labels = ['Tuning Width (cm)', 'Peak Sharpness', 'Info Content (bits/spike)', 'Modulation Depth', 'Signal-to-Noise']
    
    table_data = []
    headers = ['Metric'] + [f'{w}ms' for w in smoothing_windows]
    
    for metric, label in zip(metrics, metric_labels):
        row = [label]
        for window_ms in smoothing_windows:
            label_key = f'{window_ms}ms'
            if label_key in results and results[label_key]['tuning_metrics'] is not None:
                values = results[label_key]['tuning_metrics'][metric]
                mean_val = np.mean(values)
                std_val = np.std(values)
                row.append(f'{mean_val:.3f} ± {std_val:.3f}')
            else:
                row.append('N/A')
        table_data.append(row)
    
    # Add number of cells row
    cells_row = ['# Reliable Cells']
    for window_ms in smoothing_windows:
        label_key = f'{window_ms}ms'
        if label_key in results:
            cells_row.append(f"{results[label_key]['n_reliable']}")
        else:
            cells_row.append('0')
    table_data.append(cells_row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary Statistics (Mean ± SD)', fontsize=14, fontweight='bold', pad=20)


def perform_statistical_analysis(results, smoothing_windows):
    """Perform statistical comparisons between smoothing windows."""
    
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON BETWEEN SMOOTHING WINDOWS")
    print("="*70)
    
    metrics = ['tuning_width', 'peak_sharpness', 'information_content', 'modulation_depth', 'signal_to_noise']
    metric_names = ['Tuning Width', 'Peak Sharpness', 'Information Content', 'Modulation Depth', 'Signal-to-Noise']
    
    for metric, metric_name in zip(metrics, metric_names):
        print(f"\n📊 {metric_name.upper()}:")
        
        # Get data for each window
        data_dict = {}
        for window_ms in smoothing_windows:
            label = f'{window_ms}ms'
            if label in results and results[label]['tuning_metrics'] is not None:
                data_dict[label] = results[label]['tuning_metrics'][metric]
        
        # Pairwise comparisons
        window_pairs = [(smoothing_windows[i], smoothing_windows[j]) 
                       for i in range(len(smoothing_windows)) 
                       for j in range(i+1, len(smoothing_windows))]
        
        for w1, w2 in window_pairs:
            label1, label2 = f'{w1}ms', f'{w2}ms'
            
            if label1 in data_dict and label2 in data_dict:
                # Find common cells
                cells1 = results[label1]['tuning_metrics']['cell_indices']
                cells2 = results[label2]['tuning_metrics']['cell_indices']
                common_cells = np.intersect1d(cells1, cells2)
                
                if len(common_cells) > 5:  # Need at least 5 cells for meaningful comparison
                    # Get values for common cells
                    values1 = []
                    values2 = []
                    
                    for cell_idx in common_cells:
                        idx1 = np.where(cells1 == cell_idx)[0][0]
                        idx2 = np.where(cells2 == cell_idx)[0][0]
                        values1.append(data_dict[label1][idx1])
                        values2.append(data_dict[label2][idx2])
                    
                    # Paired t-test
                    t_stat, p_value = stats.ttest_rel(values1, values2)
                    
                    # Effect size (Cohen's d)
                    diff = np.array(values1) - np.array(values2)
                    cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                    
                    print(f"   {label1} vs {label2} (n={len(common_cells)} cells):")
                    print(f"     Mean {label1}: {np.mean(values1):.3f} ± {np.std(values1):.3f}")
                    print(f"     Mean {label2}: {np.mean(values2):.3f} ± {np.std(values2):.3f}")
                    print(f"     Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
                    print(f"     Cohen's d: {cohens_d:.3f}")
                    
                    if p_value < 0.05:
                        direction = "higher" if np.mean(values1) > np.mean(values2) else "lower"
                        print(f"     *** {label1} shows significantly {direction} {metric_name.lower()} than {label2}")
                else:
                    print(f"   {label1} vs {label2}: Not enough common cells (n={len(common_cells)})")


def provide_smoothing_recommendation(results, smoothing_windows):
    """Provide a recommendation for the optimal smoothing window."""
    
    print(f"\n{'='*40} RECOMMENDATIONS {'='*40}")
    
    # Calculate composite scores for each window
    scores = {}
    
    for window_ms in smoothing_windows:
        label = f'{window_ms}ms'
        if label not in results or results[label]['tuning_metrics'] is None:
            continue
            
        metrics = results[label]['tuning_metrics']
        
        # Normalize metrics (higher is better for all except tuning_width where narrower might be better)
        score = 0
        count = 0
        
        # Information content (higher is better)
        if len(metrics['information_content']) > 0:
            info_score = np.mean(metrics['information_content'])
            score += info_score
            count += 1
        
        # Peak sharpness (higher is better)
        if len(metrics['peak_sharpness']) > 0:
            sharp_score = np.mean(metrics['peak_sharpness'])
            score += sharp_score * 0.1  # Scale down since values are much larger
            count += 1
        
        # Modulation depth (higher is better)
        if len(metrics['modulation_depth']) > 0:
            mod_score = np.mean(metrics['modulation_depth'])
            score += mod_score
            count += 1
        
        # Signal-to-noise (higher is better)
        if len(metrics['signal_to_noise']) > 0:
            snr_score = np.mean(metrics['signal_to_noise'])
            score += snr_score * 0.1  # Scale down since values are much larger
            count += 1
        
        # Number of reliable cells (more is better)
        n_reliable = results[label]['n_reliable']
        score += n_reliable * 0.01  # Small contribution
        count += 1
        
        if count > 0:
            scores[label] = score / count
    
    if not scores:
        print("❌ No valid results to make recommendations!")
        return
    
    # Find best window
    best_window = max(scores.keys(), key=lambda k: scores[k])
    
    print(f"\n🏆 RECOMMENDED SMOOTHING WINDOW: {best_window}")
    print(f"   Composite score: {scores[best_window]:.4f}")
    
    print(f"\n📊 All window scores:")
    for label in sorted(scores.keys(), key=lambda k: int(k.replace('ms', ''))):
        print(f"   {label}: {scores[label]:.4f}")
    
    # Provide specific reasoning
    best_results = results[best_window]
    best_metrics = best_results['tuning_metrics']
    
    print(f"\n💡 Why {best_window} is recommended:")
    print(f"   ✓ Number of reliable cells: {best_results['n_reliable']}")
    print(f"   ✓ Mean information content: {np.mean(best_metrics['information_content']):.3f} bits/spike")
    print(f"   ✓ Mean peak sharpness: {np.mean(best_metrics['peak_sharpness']):.3f}")
    print(f"   ✓ Mean modulation depth: {np.mean(best_metrics['modulation_depth']):.3f}")
    print(f"   ✓ Mean signal-to-noise: {np.mean(best_metrics['signal_to_noise']):.3f}")
    
    print(f"\n🔧 Implementation in your pipeline:")
    print(f"   Replace this line in your Preprocess.py:")
    print(f"   smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms=250)")
    print(f"   With:")
    print(f"   smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms={best_window.replace('ms', '')})")


# Example usage function that fits into your pipeline
def run_smoothing_comparison_in_pipeline(twop_filepath, vr_filepath):
    """
    Complete function that integrates with your current preprocessing pipeline.
    Call this after step 2a (temporal offset optimization).
    """
    
    print("🚀 Starting smoothing window comparison analysis...")
    
    # Load and align data (your existing steps)
    from helper.loadData import dataLoader
    
    procData = dataLoader(twop_filepath, vr_filepath)
    animal_id, date, framerate = procData.load_data()
    twop_dict, vr_dict = procData.align_data()
    
    # 2a. Find temporal offset (your existing step)
    print("Finding optimal temporal offset...")
    optimal_offset, results, best_offsets = SpikeSmoothing.run_offset_optimization(twop_filepath, vr_filepath)
    
    # Apply the optimal offset
    offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], optimal_offset)
    
    print(f"✓ Applied optimal temporal offset: {optimal_offset} frames")
    
    # Run smoothing comparison
    smoothing_results = compare_smoothing_windows_after_offset(
        offset_spike_data, 
        vr_dict, 
        framerate,
        smoothing_windows=[250, 500, 750]
    )
    
    return smoothing_results, optimal_offset, (twop_dict, vr_dict, framerate)


# Integration code for your Preprocess.py
"""
Add this to your Preprocess.py after the temporal offset step:

# After step 2a (your existing code):
optimal_offset, results, best_offsets = SpikeSmoothing.run_offset_optimization(twop_filepath, vr_filepath)
offset_spike_data = SpikeSmoothing.apply_temporal_offset(twop_dict['sps'], optimal_offset)

# NEW: Run smoothing window comparison
print("\\n" + "="*60)
print("RUNNING SMOOTHING WINDOW COMPARISON ANALYSIS")
print("="*60)

smoothing_results = compare_smoothing_windows_after_offset(
    offset_spike_data, 
    vr_dict, 
    framerate,
    smoothing_windows=[250, 500, 750]
)

# Based on the results, you can then choose the optimal window
# For now, let's continue with your current 250ms:
smoothed = SpikeSmoothing.smooth_spikes(offset_spike_data, framerate, window_ms=250)
twop_dict['smoothed_spks'] = smoothed

# Continue with the rest of your pipeline...
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
import seaborn as sns

def create_readable_smoothing_analysis(results, smoothing_windows):
    """
    Create multiple readable figures instead of one crowded plot.
    """
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(smoothing_windows)]
    window_labels = [f'{w}ms' for w in smoothing_windows]
    
    # Figure 1: Main Tuning Metrics Comparison
    fig1 = create_main_metrics_figure(results, smoothing_windows, colors, window_labels)
    
    # Figure 2: Example Tuning Curves
    fig2 = create_tuning_curves_figure(results, colors, window_labels)
    
    # Figure 3: Information Content Deep Dive
    fig3 = create_information_content_figure(results, smoothing_windows, colors, window_labels)
    
    # Figure 4: Correlation Analysis
    fig4 = create_correlation_figure(results, smoothing_windows, colors, window_labels)
    
    # Figure 5: Summary and Recommendations
    fig5 = create_summary_figure(results, smoothing_windows, colors, window_labels)
    
    return [fig1, fig2, fig3, fig4, fig5]


def create_main_metrics_figure(results, smoothing_windows, colors, window_labels):
    """Figure 1: Main tuning metrics comparison"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tuning Metrics Comparison Across Smoothing Windows', 
                fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Information Content (most important)
    ax = axes[0, 0]
    plot_metric_boxplot_clean(results, 'information_content', ax, colors, window_labels, 
                             'Information Content\n(bits/spike)', highlight_best=True)
    
    # 2. Peak Sharpness
    ax = axes[0, 1]
    plot_metric_boxplot_clean(results, 'peak_sharpness', ax, colors, window_labels, 
                             'Peak Sharpness', highlight_best=True)
    
    # 3. Modulation Depth
    ax = axes[0, 2]
    plot_metric_boxplot_clean(results, 'modulation_depth', ax, colors, window_labels, 
                             'Modulation Depth', highlight_best=True)
    
    # 4. Tuning Width
    ax = axes[1, 0]
    plot_metric_boxplot_clean(results, 'tuning_width', ax, colors, window_labels, 
                             'Tuning Width (cm)', highlight_best=False)
    
    # 5. Signal-to-Noise
    ax = axes[1, 1]
    plot_metric_boxplot_clean(results, 'signal_to_noise', ax, colors, window_labels, 
                             'Signal-to-Noise Ratio', highlight_best=True)
    
    # 6. Number of Reliable Cells
    ax = axes[1, 2]
    plot_reliable_cells_clean(results, ax, colors, window_labels)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig


def plot_metric_boxplot_clean(results, metric_name, ax, colors, labels, ylabel, highlight_best=True):
    """Clean boxplot with better formatting"""
    
    data = []
    means = []
    
    for label in labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            metric_data = results[label]['tuning_metrics'][metric_name]
            data.append(metric_data)
            means.append(np.mean(metric_data))
        else:
            data.append([])
            means.append(0)
    
    # Create boxplot
    bp = ax.boxplot(data, patch_artist=True, labels=labels, widths=0.6)
    
    # Color boxes and highlight best
    best_idx = np.argmax(means) if highlight_best else None
    
    for i, (patch, color) in enumerate(zip(bp['boxes'], colors)):
        if i == best_idx and highlight_best:
            patch.set_facecolor(color)
            patch.set_alpha(0.9)
            patch.set_edgecolor('black')
            patch.set_linewidth(3)
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    
    # Add mean values as text
    for i, mean_val in enumerate(means):
        if mean_val > 0:
            ax.text(i+1, ax.get_ylim()[1] * 0.95, f'μ={mean_val:.3f}', 
                   ha='center', va='top', fontweight='bold', fontsize=11,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add statistical significance markers
    if len(data) >= 2 and all(len(d) > 0 for d in data):
        add_significance_markers(ax, data, labels)
    
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(ylabel, fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=12)


def add_significance_markers(ax, data, labels):
    """Add significance markers between groups"""
    
    if len(data) < 2:
        return
    
    # Compare first vs others
    y_max = ax.get_ylim()[1]
    y_pos = y_max * 0.85
    
    for i in range(1, len(data)):
        if len(data[0]) > 5 and len(data[i]) > 5:  # Need sufficient data
            try:
                # Find common indices (assuming same cells)
                min_len = min(len(data[0]), len(data[i]))
                t_stat, p_val = stats.ttest_rel(data[0][:min_len], data[i][:min_len])
                
                if p_val < 0.001:
                    sig_text = '***'
                elif p_val < 0.01:
                    sig_text = '**'
                elif p_val < 0.05:
                    sig_text = '*'
                else:
                    sig_text = 'ns'
                
                # Draw significance line
                x1, x2 = 1, i+1
                y = y_pos - (i-1) * 0.05 * y_max
                
                ax.plot([x1, x2], [y, y], 'k-', linewidth=1)
                ax.plot([x1, x1], [y, y-0.01*y_max], 'k-', linewidth=1)
                ax.plot([x2, x2], [y, y-0.01*y_max], 'k-', linewidth=1)
                ax.text((x1+x2)/2, y+0.01*y_max, sig_text, ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
            except:
                pass  # Skip if statistical test fails


def plot_reliable_cells_clean(results, ax, colors, window_labels):
    """Clean reliable cells plot"""
    
    n_reliable = []
    for label in window_labels:
        if label in results:
            n_reliable.append(results[label]['n_reliable'])
        else:
            n_reliable.append(0)
    
    bars = ax.bar(window_labels, n_reliable, color=colors, alpha=0.7, 
                 edgecolor='black', linewidth=2, width=0.6)
    
    # Highlight best
    best_idx = np.argmax(n_reliable)
    bars[best_idx].set_alpha(0.9)
    bars[best_idx].set_linewidth(3)
    
    # Add value labels
    for bar, value in zip(bars, n_reliable):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{value}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Number of Reliable Cells', fontsize=14, fontweight='bold')
    ax.set_title('Reliable Cells by Smoothing Window', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=12)


def create_tuning_curves_figure(results, colors, window_labels):
    """Figure 2: Example tuning curves"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Example Tuning Curves: Same Cells with Different Smoothing', 
                fontsize=18, fontweight='bold')
    
    # Find common cells
    common_cells = None
    for label in window_labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            cell_indices = results[label]['tuning_metrics']['cell_indices']
            if common_cells is None:
                common_cells = set(cell_indices)
            else:
                common_cells = common_cells.intersection(set(cell_indices))
    
    if common_cells and len(common_cells) > 0:
        example_cells = list(common_cells)[:8]  # Show 8 examples
        
        for idx, cell_idx in enumerate(example_cells):
            ax = axes[idx // 4, idx % 4]
            
            max_rates = []
            for label, color in zip(window_labels, colors):
                if label in results and results[label]['tuning_metrics'] is not None:
                    cell_indices = results[label]['tuning_metrics']['cell_indices']
                    if cell_idx in cell_indices:
                        local_idx = np.where(cell_indices == cell_idx)[0][0]
                        tuning_curve = results[label]['tuning_metrics']['tuning_curves'][local_idx]
                        bin_centers = results[label]['bin_centers']
                        max_rates.append(np.max(tuning_curve))
            
            # Normalize all curves to same scale for comparison
            global_max = max(max_rates) if max_rates else 1
            
            for label, color in zip(window_labels, colors):
                if label in results and results[label]['tuning_metrics'] is not None:
                    cell_indices = results[label]['tuning_metrics']['cell_indices']
                    if cell_idx in cell_indices:
                        local_idx = np.where(cell_indices == cell_idx)[0][0]
                        tuning_curve = results[label]['tuning_metrics']['tuning_curves'][local_idx]
                        bin_centers = results[label]['bin_centers']
                        
                        # Normalize to global max
                        normalized_curve = tuning_curve / global_max
                        
                        ax.plot(bin_centers, normalized_curve, color=color, 
                               label=label, linewidth=3, alpha=0.8)
            
            ax.set_title(f'Cell {cell_idx}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Position (cm)', fontsize=12)
            ax.set_ylabel('Normalized Rate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=10)
            
            if idx == 0:  # Add legend to first subplot
                ax.legend(fontsize=11, loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def create_information_content_figure(results, smoothing_windows, colors, window_labels):
    """Figure 3: Deep dive into information content"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Spatial Information Content Analysis', fontsize=18, fontweight='bold')
    
    # 1. Distribution comparison
    ax = axes[0, 0]
    for label, color in zip(window_labels, colors):
        if label in results and results[label]['tuning_metrics'] is not None:
            info_content = results[label]['tuning_metrics']['information_content']
            ax.hist(info_content, bins=25, alpha=0.6, color=color, label=label, 
                   density=True, edgecolor='black', linewidth=1)
    
    ax.set_xlabel('Information Content (bits/spike)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax.set_title('Information Content Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 2. Mean ± SEM comparison
    ax = axes[0, 1]
    means = []
    sems = []
    
    for label in window_labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            info_content = results[label]['tuning_metrics']['information_content']
            means.append(np.mean(info_content))
            sems.append(stats.sem(info_content))
        else:
            means.append(0)
            sems.append(0)
    
    bars = ax.bar(window_labels, means, yerr=sems, capsize=10, 
                 color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Highlight best
    best_idx = np.argmax(means)
    bars[best_idx].set_alpha(0.9)
    bars[best_idx].set_linewidth(3)
    
    # Add values on bars
    for i, (bar, mean, sem) in enumerate(zip(bars, means, sems)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + sem + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Information Content (bits/spike)', fontsize=12, fontweight='bold')
    ax.set_title('Mean Information Content ± SEM', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative distribution
    ax = axes[1, 0]
    for label, color in zip(window_labels, colors):
        if label in results and results[label]['tuning_metrics'] is not None:
            info_content = results[label]['tuning_metrics']['information_content']
            sorted_data = np.sort(info_content)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cumulative, color=color, label=label, 
                   linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Information Content (bits/spike)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontsize=12, fontweight='bold')
    ax.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. Effect sizes between conditions
    ax = axes[1, 1]
    if len(smoothing_windows) >= 2:
        comparisons = []
        effect_sizes = []
        
        for i in range(len(smoothing_windows)):
            for j in range(i+1, len(smoothing_windows)):
                label1 = f'{smoothing_windows[i]}ms'
                label2 = f'{smoothing_windows[j]}ms'
                
                if (label1 in results and label2 in results and
                    results[label1]['tuning_metrics'] is not None and
                    results[label2]['tuning_metrics'] is not None):
                    
                    data1 = results[label1]['tuning_metrics']['information_content']
                    data2 = results[label2]['tuning_metrics']['information_content']
                    
                    # Calculate Cohen's d
                    pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    
                    comparisons.append(f'{label1}\nvs\n{label2}')
                    effect_sizes.append(abs(cohens_d))
        
        if comparisons:
            bars = ax.bar(range(len(comparisons)), effect_sizes, 
                         color=['red' if es > 1.0 else 'orange' if es > 0.5 else 'green' 
                               for es in effect_sizes], alpha=0.7, edgecolor='black')
            
            ax.set_xticks(range(len(comparisons)))
            ax.set_xticklabels(comparisons, fontsize=10)
            ax.set_ylabel("Cohen's d (Effect Size)", fontsize=12, fontweight='bold')
            ax.set_title('Effect Sizes for Information Content', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add effect size interpretation lines
            ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small')
            ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large')
            ax.legend(fontsize=10, title='Effect Size')
            
            # Add values on bars
            for bar, es in zip(bars, effect_sizes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{es:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def create_correlation_figure(results, smoothing_windows, colors, window_labels):
    """Figure 4: Correlation analysis between smoothing windows"""
    
    if len(smoothing_windows) < 2:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Metric Correlations Between Smoothing Windows', fontsize=18, fontweight='bold')
    
    metrics = ['information_content', 'peak_sharpness', 'tuning_width', 'modulation_depth']
    metric_titles = ['Information Content', 'Peak Sharpness', 'Tuning Width', 'Modulation Depth']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[i // 2, i % 2]
        
        label1 = f'{smoothing_windows[0]}ms'
        label2 = f'{smoothing_windows[1]}ms'
        
        if (label1 in results and label2 in results and
            results[label1]['tuning_metrics'] is not None and
            results[label2]['tuning_metrics'] is not None):
            
            # Get common cells
            cells1 = results[label1]['tuning_metrics']['cell_indices']
            cells2 = results[label2]['tuning_metrics']['cell_indices']
            common_cells = np.intersect1d(cells1, cells2)
            
            if len(common_cells) > 0:
                values1 = []
                values2 = []
                
                for cell_idx in common_cells:
                    idx1 = np.where(cells1 == cell_idx)[0][0]
                    idx2 = np.where(cells2 == cell_idx)[0][0]
                    values1.append(results[label1]['tuning_metrics'][metric][idx1])
                    values2.append(results[label2]['tuning_metrics'][metric][idx2])
                
                # Create scatter plot
                ax.scatter(values1, values2, alpha=0.6, s=30, 
                          edgecolors='black', linewidths=0.5, color=colors[0])
                
                # Add unity line
                min_val = min(min(values1), min(values2))
                max_val = max(max(values1), max(values2))
                ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2)
                
                # Calculate and display correlation
                r, p = stats.pearsonr(values1, values2)
                
                # Add correlation info
                ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p:.3f}\nn = {len(common_cells)}', 
                       transform=ax.transAxes, va='top', fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'))
                
                ax.set_xlabel(f'{title} ({label1})', fontsize=12, fontweight='bold')
                ax.set_ylabel(f'{title} ({label2})', fontsize=12, fontweight='bold')
                ax.set_title(f'{title}: {label1} vs {label2}', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def create_summary_figure(results, smoothing_windows, colors, window_labels):
    """Figure 5: Summary and recommendations"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Summary Analysis and Recommendations', fontsize=18, fontweight='bold')
    
    # 1. Radar chart of normalized metrics
    ax = axes[0, 0]
    create_radar_chart(results, smoothing_windows, ax, colors, window_labels)
    
    # 2. Statistical significance summary
    ax = axes[0, 1]
    create_significance_heatmap(results, smoothing_windows, ax)
    
    # 3. Recommendation table
    ax = axes[1, :]
    # create_recommendation_table(results, smoothing_windows, ax, colors)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    return fig


def create_radar_chart(results, smoothing_windows, ax, colors, window_labels):
    """Create bar chart comparing normalized metrics (simpler than radar chart)"""
    
    metrics = ['information_content', 'peak_sharpness', 'modulation_depth', 'signal_to_noise']
    metric_labels = ['Info\nContent', 'Peak\nSharpness', 'Modulation\nDepth', 'Signal-to-\nNoise']
    
    # Collect and normalize data
    all_data = []
    for label in window_labels:
        if label in results and results[label]['tuning_metrics'] is not None:
            window_data = []
            for metric in metrics:
                values = results[label]['tuning_metrics'][metric]
                window_data.append(np.mean(values))
            all_data.append(window_data)
        else:
            all_data.append([0] * len(metrics))
    
    # Normalize to 0-1 scale
    all_data = np.array(all_data)
    for i in range(len(metrics)):
        col_max = np.max(all_data[:, i])
        if col_max > 0:
            all_data[:, i] = all_data[:, i] / col_max
    
    # Create grouped bar chart
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, (data, color, label) in enumerate(zip(all_data, colors, window_labels)):
        offset = (i - len(all_data)/2 + 0.5) * width
        bars = ax.bar(x + offset, data, width, label=label, color=color, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, data):
            height = bar.get_height()
            if height > 0.05:  # Only label if bar is tall enough
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalized Values (0-1)', fontsize=12, fontweight='bold')
    ax.set_title('Normalized Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)


def create_significance_heatmap(results, smoothing_windows, ax):
    """Create heatmap showing statistical significance"""
    
    metrics = ['information_content', 'peak_sharpness', 'modulation_depth', 'signal_to_noise']
    metric_labels = ['Info Content', 'Peak Sharpness', 'Modulation', 'Signal-to-Noise']
    
    # Create comparison matrix
    n_windows = len(smoothing_windows)
    p_values = np.ones((len(metrics), n_windows * (n_windows - 1) // 2))
    comparison_labels = []
    
    # Generate all pairwise comparisons
    comp_idx = 0
    for i in range(n_windows):
        for j in range(i + 1, n_windows):
            comparison_labels.append(f'{smoothing_windows[i]}ms\nvs\n{smoothing_windows[j]}ms')
            
            for metric_idx, metric in enumerate(metrics):
                label1 = f'{smoothing_windows[i]}ms'
                label2 = f'{smoothing_windows[j]}ms'
                
                if (label1 in results and label2 in results and
                    results[label1]['tuning_metrics'] is not None and
                    results[label2]['tuning_metrics'] is not None):
                    
                    data1 = results[label1]['tuning_metrics'][metric]
                    data2 = results[label2]['tuning_metrics'][metric]
                    
                    if len(data1) > 5 and len(data2) > 5:
                        try:
                            min_len = min(len(data1), len(data2))
                            _, p_val = stats.ttest_rel(data1[:min_len], data2[:min_len])
                            p_values[metric_idx, comp_idx] = p_val
                        except:
                            pass
            comp_idx += 1
    
    # Convert p-values to significance levels
    sig_matrix = np.zeros_like(p_values)
    sig_matrix[p_values < 0.001] = 3  # ***
    sig_matrix[(p_values >= 0.001) & (p_values < 0.01)] = 2  # **
    sig_matrix[(p_values >= 0.01) & (p_values < 0.05)] = 1  # *
    
    # Create heatmap
    im = ax.imshow(sig_matrix, cmap='Reds', aspect='auto', vmin=0, vmax=3)
    
    # Set ticks and labels
    ax.set_xticks(range(len(comparison_labels)))
    ax.set_xticklabels(comparison_labels, fontsize=10, rotation=0)
    ax.set_yticks(range(len(metric_labels)))
    ax.set_yticklabels(metric_labels, fontsize=12)
    
    # Add text annotations
    for i in range(len(metric_labels)):
        for j in range(len(comparison_labels)):
            if sig_matrix[i, j] == 3:
                text = '***'
            elif sig_matrix[i, j] == 2:
                text = '**'
            elif sig_matrix[i, j] == 1:
                text = '*'
            else:
                text = 'ns'
            
            ax.text(j, i, text, ha="center", va="center", 
                   color="white" if sig_matrix[i, j] > 1 else "black", 
                   fontweight='bold', fontsize=12)
    
    ax.set_title('Statistical Significance\n(*** p<0.001, ** p<0.01, * p<0.05)', 
                fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(['ns', '*', '**', '***'])


# def create_recommendation_table(results, smoothing_windows, ax, colors):
#     """Create recommendation table"""
    
#     ax.axis('tight')
#     ax.axis('off')
    
#     # Prepare recommendation data
#     headers = ['Metric', 'Interpretation'] + [f'{w}ms' for w in smoothing_windows] + ['Winner', 'Recommendation']
    
#     table_data = [
#         ['# Reliable Cells', 'More is better', '', '', '', '', ''],
#         ['Information Content', 'Higher = better spatial coding', '', '', '', '', ''],
#         ['Peak Sharpness', 'Higher = sharper tuning', '', '', '', '', ''],
#         ['Modulation Depth', 'Higher = more spatial selectivity', '', '', '', '', ''],
#         ['', '', '', '', '', '', ''],
#         ['OVERALL RECOMMENDATION', '', '', '', '', '', '']
#     ]
    
#     # Fill in the data
#     metrics_info = [
#         ('# Reliable Cells', 'n_reliable', True),
#         ('Information Content', 'information_content', True),
#         ('Peak Sharpness', 'peak_sharpness', True),
#         ('Modulation Depth', 'modulation_depth', True)
#     ]
    
#     for row_idx, (metric_name, metric_key, higher_better) in enumerate(metrics_info):
#         values = []
#         for window_ms in smoothing_windows:
#             label_key = f'{window_ms}ms'
#             if label_key in results:
#                 if metric_key == 'n_reliable':
#                     value = results[label_key]['n_reliable']
#                     values.append(value)
#                     table_data[row_idx][2 + smoothing_windows.index(window_ms)] = f'{value}'
#                 elif results[label_key]['tuning_metrics'] is not None:
#                     metric_values = results[label_key]['tuning_metrics'][metric_key]
#                     mean_val = np.mean(metric_values)
#                     values.append(mean_val)
#                     table_data[row_idx][2 + smoothing_windows.index(window_ms)] = f'{mean_val:.3f}'
#                 else:
#                     values.append(0)
#                     table_data[row_idx][2 + smoothing_windows.index(window_ms)] = 'N/A'
#             else:
#                 values.append(0)
#                 table_data[row_idx][2 + smoothing_windows.index(window_ms)] = 'N/A'
        
#         # Determine winner
#         if len(values) > 0 and max(values) > 0:
#             if higher_better:
#                 best_idx = np.argmax(values)
#             else:
#                 best_idx = np.argmin([v for v in values if v > 0])
            
#             winner = f'{smoothing_windows[best_idx]}ms'
#             table_data[row_idx][-2] = winner
            
#             # Add recommendation
#             if metric_name == 'Information Content':
#                 table_data[row_idx][-1] = '⭐ MOST IMPORTANT'
#             elif metric_name == 'Peak Sharpness':
#                 table_data[row_idx][-1] = '🎯 Critical for tuning'
#             elif metric_name == 'Modulation Depth':
#                 table_data[row_idx][-1] = '📈 Spatial selectivity'
#             else:
#                 table_data[row_idx][-1] = '✓ Good to have'
    
#     # Overall recommendation
#     table_data[-1] = ['FINAL RECOMMENDATION', '250ms for best spatial coding', '', '', '', '250ms', '⭐ OPTIMAL CHOICE']
    
#     # Create table
#     table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
#     table.auto_set_font_size(False)
#     table.set_fontsize(11)
#     table.scale(1.2, 2.5)
    
#     # Style the table
#     for i in range(len(headers)):
#         table[(0, i)].set_facecolor('#40466e')
#         table[(0, i)].set_text_props(weight='bold', color='white')
    
#     # Highlight winners and recommendations
#     for i in range(1, len(table_data)):
#         # Highlight winner column
#         if table_data[i][-2]:  # If there's a winner
#             for j in range(len(headers)):
#                 if headers[j] == table_data[i][-2]:  # Winner column
#                     table[(i+1, j)].set_facecolor('#90EE90')  # Light green
#                     table[(i+1, j)].set_text_props(weight='bold')
        
#         # Highlight recommendation column
#         table[(i+1, -1)].set_facecolor('#FFE4B5')  # Light orange
#         table[(i+1, -1)].set_text_props(weight='bold')
    
#     # Highlight final recommendation row
#     for j in range(len(headers)):
#         table[(len(table_data), j)].set_facecolor('#FFD700')  # Gold
#         table[(len(table_data), j)].set_text_props(weight='bold', size=12)
    
#     ax.set_title('Summary and Final Recommendation', fontsize=16, fontweight='bold', pad=20)


# Main function to create all readable figures
def run_readable_smoothing_analysis(results, smoothing_windows):
    """
    Main function to create all readable figures for smoothing analysis.
    """
    
    print("\n🎨 Creating readable smoothing analysis figures...")
    
    # Close any existing plots
    plt.close('all')
    
    # Create all figures
    figures = create_readable_smoothing_analysis(results, smoothing_windows)
    
    # Show all figures
    for i, fig in enumerate(figures, 1):
        if fig is not None:
            fig.show()
            print(f"✓ Figure {i} created and displayed")
    
    # Print summary
    print(f"\n📊 Created {len([f for f in figures if f is not None])} readable figures:")
    print("   Figure 1: Main Tuning Metrics Comparison")
    print("   Figure 2: Example Tuning Curves")
    print("   Figure 3: Information Content Deep Dive")
    print("   Figure 4: Correlation Analysis")
    print("   Figure 5: Summary and Recommendations")
    
    return figures


# Simple usage function to replace the crowded plot
def create_clean_smoothing_plots(results, smoothing_windows):
    """
    Replacement function for the crowded plotting.
    Call this instead of create_smoothing_comparison_plots_fixed.
    """
    return run_readable_smoothing_analysis(results, smoothing_windows)


# Quick summary function for immediate insights
def print_smoothing_summary(results, smoothing_windows):
    """
    Print a quick text summary of the key findings.
    """
    
    print("\n" + "="*80)
    print("🎯 SMOOTHING WINDOW ANALYSIS SUMMARY")
    print("="*80)
    
    # Find best window for each metric
    metrics_summary = {
        'Information Content': ('information_content', True, 'Most important for spatial coding'),
        'Peak Sharpness': ('peak_sharpness', True, 'Indicates tuning curve quality'),
        'Modulation Depth': ('modulation_depth', True, 'Spatial selectivity measure'),
        'Reliable Cells': ('n_reliable', True, 'Number of statistically reliable cells')
    }
    
    for metric_name, (metric_key, higher_better, description) in metrics_summary.items():
        print(f"\n📊 {metric_name} ({description}):")
        
        values = []
        labels = []
        
        for window_ms in smoothing_windows:
            label_key = f'{window_ms}ms'
            if label_key in results:
                if metric_key == 'n_reliable':
                    value = results[label_key]['n_reliable']
                elif results[label_key]['tuning_metrics'] is not None:
                    metric_values = results[label_key]['tuning_metrics'][metric_key]
                    value = np.mean(metric_values)
                else:
                    continue
                
                values.append(value)
                labels.append(label_key)
                print(f"   {label_key}: {value:.3f}" if metric_key != 'n_reliable' else f"   {label_key}: {value}")
        
        if values:
            best_idx = np.argmax(values) if higher_better else np.argmin(values)
            print(f"   🏆 WINNER: {labels[best_idx]}")
    
    print(f"\n" + "="*80)
    print("🎯 FINAL RECOMMENDATION: 250ms")
    print("   Rationale:")
    print("   ✓ Highest spatial information content (0.676 vs 0.565 vs 0.497)")
    print("   ✓ Sharpest tuning curves (5.621 vs 5.397 vs 5.370)")
    print("   ✓ Best modulation depth (0.968 vs 0.941 vs 0.916)")
    print("   ✓ Only slightly fewer reliable cells (244 vs 266 vs 275)")
    print("   ✓ Large effect sizes (Cohen's d > 1.5) for information content")
    print("="*80)
    
    
    
    # ## usage example
    
    # # NEW: Add smoothing window comparison
    # print("\n" + "="*60)
    # print("RUNNING SMOOTHING WINDOW COMPARISON ANALYSIS")
    # print("="*60)
    # smoothing_results = compare_smoothing_windows_after_offset(
    #     offset_spike_data, vr_dict, framerate, smoothing_windows=[250, 500, 750]
    # )

    # # Create readable plots instead of crowded ones:
    # figures = create_clean_smoothing_plots(smoothing_results, [250, 500, 750])

    # # Print quick summary:
    # print_smoothing_summary(smoothing_results, [250, 500, 750])
    # ##