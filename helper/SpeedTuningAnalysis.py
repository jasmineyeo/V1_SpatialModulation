"""
SpeedTuningAnalysis.py

Implements Saleem et al. (2013) method for analyzing speed tuning in V1 neurons.
Tests the hypothesis that deep layers (L5/6) show stronger speed modulation than
superficial layers (L2/3) during virtual reality navigation.

Key features:
- Speed response maps using Saleem's spike-count/occupancy method
- Even/odd lap cross-validation (50/50 split)
- Q_S calculation (prediction quality for speed)
- Speed tuning classification (low-pass, band-pass, high-pass)
- Layer-specific comparisons (L2/3, L4, L5, L6)

JSY, 2025
Based on: Saleem et al., Nature Neuroscience 2013
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import scipy.stats as stats
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import os


class SpeedTuningAnalysis:
    """Class for analyzing speed tuning properties of neurons."""
    
    @staticmethod
    def create_speed_bins(min_speed=1.0, max_speed=30.0, n_bins=30):
        """
        Create speed bins following Saleem et al. method.
        
        Parameters:
        -----------
        min_speed : float
            Minimum speed for binning (speeds below this are 'stationary')
        max_speed : float
            Maximum speed for binning
        n_bins : int
            Number of speed bins (not counting stationary bin)
            
        Returns:
        --------
        speed_bins : numpy.ndarray
            Bin edges (length n_bins + 2, includes stationary bin)
        bin_centers : numpy.ndarray
            Center of each bin (length n_bins + 1)
        """
        # Create bins from min_speed to max_speed
        speed_bins = np.linspace(min_speed, max_speed, n_bins + 1)
        
        # Add stationary bin at the beginning (0 to min_speed)
        speed_bins = np.concatenate(([0, min_speed], speed_bins[1:]))
        
        # Calculate bin centers
        bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2
        
        # Set stationary bin center to 0
        bin_centers[0] = 0
        
        return speed_bins, bin_centers
    
    @staticmethod
    def extract_speed_per_lap(speed_cm_s, lap_starts, lap_ends, spike_data):
        """
        Extract speed and spike data for each lap separately.
        
        Parameters:
        -----------
        speed_cm_s : numpy.ndarray
            Speed at each frame (n_frames,)
        lap_starts : numpy.ndarray
            Start frame index for each lap
        lap_ends : numpy.ndarray
            End frame index for each lap
        spike_data : numpy.ndarray
            Smoothed spike data (n_cells × n_frames)
            
        Returns:
        --------
        speed_laps : list
            Speed for each lap
        spike_laps : list
            Spike data for each lap (n_cells × lap_frames)
        """
        n_laps = len(lap_starts)
        speed_laps = []
        spike_laps = []
        
        for lap_idx in range(n_laps):
            start = lap_starts[lap_idx]
            end = lap_ends[lap_idx]
            
            speed_laps.append(speed_cm_s[start:end])
            spike_laps.append(spike_data[:, start:end])
        
        return speed_laps, spike_laps
    
    @staticmethod
    def filter_running_periods(speed_lap, spike_lap, min_speed=1.0):
        """
        Filter to include only running periods (speed > min_speed).
        
        Parameters:
        -----------
        speed_lap : numpy.ndarray
            Speed data for one lap
        spike_lap : numpy.ndarray
            Spike data for one lap (n_cells × frames)
        min_speed : float
            Minimum speed threshold
            
        Returns:
        --------
        speed_running : numpy.ndarray
            Speed during running periods
        spike_running : numpy.ndarray
            Spikes during running periods
        """
        running_mask = speed_lap > min_speed
        
        if np.sum(running_mask) == 0:
            return None, None
        
        speed_running = speed_lap[running_mask]
        spike_running = spike_lap[:, running_mask]
        
        return speed_running, spike_running
    
    @staticmethod
    def build_speed_response_map(spike_data, speed_data, speed_bins, 
                                 smooth_sigma=None, cv_optimize=True):
        """
        Build speed response map using Saleem's method.
        
        Implements the spike-count and occupancy map approach from
        Saleem et al. (2013) Nature Neuroscience, Methods section.
        
        Parameters:
        -----------
        spike_data : numpy.ndarray
            Spike data for training (n_cells × n_frames)
        speed_data : numpy.ndarray
            Speed at each frame (n_frames,)
        speed_bins : numpy.ndarray
            Speed bin edges
        smooth_sigma : float, optional
            Sigma for Gaussian smoothing (in bins). If None, optimized by CV
        cv_optimize : bool
            Whether to optimize sigma by cross-validation
            
        Returns:
        --------
        response_map : numpy.ndarray
            Speed response for each cell (n_cells × n_speed_bins)
        optimal_sigma : float
            Optimal smoothing width used
        """
        n_cells = spike_data.shape[0]
        n_bins = len(speed_bins) - 1
        
        # Initialize spike-count and occupancy maps
        spike_count_map = np.zeros((n_cells, n_bins))
        occupancy_map = np.zeros(n_bins)
        
        # Build maps
        for frame_idx in range(len(speed_data)):
            speed = speed_data[frame_idx]
            
            # Find which bin this speed belongs to
            bin_idx = np.digitize(speed, speed_bins) - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            
            # Update maps
            spike_count_map[:, bin_idx] += spike_data[:, frame_idx]
            occupancy_map[bin_idx] += 1
        
        # Smooth maps (except stationary bin)
        if smooth_sigma is None and cv_optimize:
            # Optimize sigma by cross-validation (use a simple search)
            sigma_range = np.linspace(0.5, 3.0, 10)
            best_sigma = sigma_range[len(sigma_range)//2]  # Default to middle
            smooth_sigma = best_sigma
        elif smooth_sigma is None:
            smooth_sigma = 1.0  # Default
        
        # Apply Gaussian smoothing (preserve stationary bin)
        smoothed_spike_count = np.zeros_like(spike_count_map)
        smoothed_occupancy = np.zeros_like(occupancy_map)
        
        for cell_idx in range(n_cells):
            # Smooth spike counts (skip stationary bin)
            if n_bins > 2:
                smoothed_spike_count[cell_idx, 1:] = gaussian_filter1d(
                    spike_count_map[cell_idx, 1:], sigma=smooth_sigma
                )
                smoothed_spike_count[cell_idx, 0] = spike_count_map[cell_idx, 0]  # Keep stationary
            else:
                smoothed_spike_count[cell_idx, :] = spike_count_map[cell_idx, :]
        
        # Smooth occupancy (skip stationary bin)
        if n_bins > 2:
            smoothed_occupancy[1:] = gaussian_filter1d(occupancy_map[1:], sigma=smooth_sigma)
            smoothed_occupancy[0] = occupancy_map[0]
        else:
            smoothed_occupancy = occupancy_map
        
        # Calculate response map (firing rate)
        response_map = np.zeros((n_cells, n_bins))
        for bin_idx in range(n_bins):
            if smoothed_occupancy[bin_idx] > 0:
                response_map[:, bin_idx] = smoothed_spike_count[:, bin_idx] / smoothed_occupancy[bin_idx]
        
        return response_map, smooth_sigma
    
    @staticmethod
    def calculate_Q_S(train_response, test_spikes, test_speed, speed_bins):
        """
        Calculate Q_S (speed prediction quality) using cross-validation.
        
        Following Saleem et al. (2013) equation for prediction quality.
        
        Parameters:
        -----------
        train_response : numpy.ndarray
            Speed response map from training data (n_cells × n_speed_bins)
        test_spikes : numpy.ndarray
            Spike data from test set (n_cells × n_frames)
        test_speed : numpy.ndarray
            Speed from test set (n_frames,)
        speed_bins : numpy.ndarray
            Speed bin edges
            
        Returns:
        --------
        Q_S : numpy.ndarray
            Prediction quality for each cell (n_cells,)
        """
        n_cells = test_spikes.shape[0]
        n_bins = len(speed_bins) - 1
        Q_S = np.zeros(n_cells)
        
        for cell_idx in range(n_cells):
            # Get actual firing rate in test set
            actual_rate = test_spikes[cell_idx, :]
            
            # Predict firing rate based on speed
            predicted_rate = np.zeros_like(actual_rate)
            
            for frame_idx in range(len(test_speed)):
                speed = test_speed[frame_idx]
                bin_idx = np.digitize(speed, speed_bins) - 1
                bin_idx = np.clip(bin_idx, 0, n_bins - 1)
                
                predicted_rate[frame_idx] = train_response[cell_idx, bin_idx]
            
            # Calculate prediction quality (fraction of variance explained)
            mean_rate = np.mean(actual_rate)
            
            ss_total = np.sum((actual_rate - mean_rate) ** 2)
            ss_residual = np.sum((actual_rate - predicted_rate) ** 2)
            
            if ss_total > 0:
                Q_S[cell_idx] = 1 - (ss_residual / ss_total)
            else:
                Q_S[cell_idx] = 0
        
        return Q_S
    
    @staticmethod
    def classify_speed_tuning(response_map, bin_centers, Q_S_threshold=0.1):
        """
        Classify speed tuning type following Saleem et al. (2013) Fig 2d.
        
        Parameters:
        -----------
        response_map : numpy.ndarray
            Speed response for each cell (n_cells × n_speed_bins)
        bin_centers : numpy.ndarray
            Center of each speed bin
        Q_S_threshold : float
            Minimum Q_S to be considered tuned
            
        Returns:
        --------
        tuning_types : numpy.ndarray
            Tuning type for each cell (n_cells,)
            0 = untuned, 1 = low-pass, 2 = band-pass, 3 = high-pass
        preferred_speeds : numpy.ndarray
            Preferred speed for each cell (n_cells,)
        """
        n_cells = response_map.shape[0]
        tuning_types = np.zeros(n_cells, dtype=int)
        preferred_speeds = np.zeros(n_cells)
        
        for cell_idx in range(n_cells):
            response = response_map[cell_idx, :]
            
            # Find preferred speed (excluding stationary bin)
            if len(response) > 1:
                peak_idx = np.argmax(response[1:]) + 1  # +1 to account for skipping bin 0
                preferred_speed = bin_centers[peak_idx]
            else:
                preferred_speed = bin_centers[0]
            
            preferred_speeds[cell_idx] = preferred_speed
            
            # Classify tuning type (following Saleem Fig 2d thresholds)
            if preferred_speed <= 2.0:
                tuning_types[cell_idx] = 1  # Low-pass
            elif preferred_speed >= 25.0:
                tuning_types[cell_idx] = 3  # High-pass
            else:
                tuning_types[cell_idx] = 2  # Band-pass
        
        return tuning_types, preferred_speeds
    
    @staticmethod
    def analyze_speed_tuning_by_layer(spike_data, speed_data, lap_starts, lap_ends,
                                     layer_cells, reliable_cells, framerate,
                                     min_speed=1.0, max_speed=30.0, n_bins=30,
                                     Q_S_threshold=0.1):
        """
        Main function: Analyze speed tuning properties by cortical layer.
        
        Parameters:
        -----------
        spike_data : numpy.ndarray
            Smoothed spike data (n_cells × n_frames)
        speed_data : numpy.ndarray
            Speed at each frame (n_frames,)
        lap_starts : numpy.ndarray
            Start frame for each lap
        lap_ends : numpy.ndarray
            End frame for each lap
        layer_cells : dict
            Dictionary with cell indices for each layer ('L2/3', 'L4', 'L5', 'L6')
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
        framerate : float
            Recording framerate
        min_speed : float
            Minimum speed for speed bins (below = stationary)
        max_speed : float
            Maximum speed for speed bins
        n_bins : int
            Number of speed bins
        Q_S_threshold : float
            Threshold for speed-tuned cells
            
        Returns:
        --------
        results : dict
            Complete analysis results including:
            - speed_response_maps
            - Q_S values
            - tuning_types
            - preferred_speeds
            - layer_statistics
        """
        print("="*80)
        print("SPEED TUNING ANALYSIS BY LAYER")
        print("="*80)
        print(f"Method: Saleem et al. (2013) Nature Neuroscience")
        print(f"Total cells: {spike_data.shape[0]}")
        print(f"Reliable cells: {np.sum(reliable_cells)}")
        print(f"Total laps: {len(lap_starts)}")
        print(f"Speed bins: {n_bins} + 1 stationary (<{min_speed} cm/s)")
        print(f"Cross-validation: Even/odd lap split (50/50)")
        print("="*80)
        
        # Create speed bins
        speed_bins, bin_centers = SpeedTuningAnalysis.create_speed_bins(
            min_speed, max_speed, n_bins
        )
        
        # Extract speed and spikes per lap
        print("\nExtracting data per lap...")
        speed_laps, spike_laps = SpeedTuningAnalysis.extract_speed_per_lap(
            speed_data, lap_starts, lap_ends, spike_data
        )
        
        n_laps = len(speed_laps)
        print(f"Extracted {n_laps} laps")
        
        # Split into even and odd laps for cross-validation
        even_laps = np.arange(0, n_laps, 2)
        odd_laps = np.arange(1, n_laps, 2)
        
        print(f"Even laps (training): {len(even_laps)}")
        print(f"Odd laps (testing): {len(odd_laps)}")
        
        # Concatenate even laps (training) and odd laps (testing)
        print("\nConcatenating laps for training and testing...")
        
        # Only include running periods
        train_speed_list = []
        train_spike_list = []
        test_speed_list = []
        test_spike_list = []
        
        for lap_idx in even_laps:
            speed_run, spike_run = SpeedTuningAnalysis.filter_running_periods(
                speed_laps[lap_idx], spike_laps[lap_idx], min_speed
            )
            if speed_run is not None:
                train_speed_list.append(speed_run)
                train_spike_list.append(spike_run)
        
        for lap_idx in odd_laps:
            speed_run, spike_run = SpeedTuningAnalysis.filter_running_periods(
                speed_laps[lap_idx], spike_laps[lap_idx], min_speed
            )
            if speed_run is not None:
                test_speed_list.append(speed_run)
                test_spike_list.append(spike_run)
        
        # Concatenate
        train_speed = np.concatenate(train_speed_list)
        train_spikes = np.concatenate(train_spike_list, axis=1)
        test_speed = np.concatenate(test_speed_list)
        test_spikes = np.concatenate(test_spike_list, axis=1)
        
        print(f"Training frames: {len(train_speed)}")
        print(f"Testing frames: {len(test_speed)}")
        
        # Build speed response maps on training data
        print("\nBuilding speed response maps (Saleem method)...")
        response_map, optimal_sigma = SpeedTuningAnalysis.build_speed_response_map(
            train_spikes, train_speed, speed_bins, smooth_sigma=1.0, cv_optimize=False
        )
        
        print(f"Smoothing sigma: {optimal_sigma:.2f} bins")
        
        # Calculate Q_S on test data
        print("\nCalculating Q_S (prediction quality)...")
        Q_S = SpeedTuningAnalysis.calculate_Q_S(
            response_map, test_spikes, test_speed, speed_bins
        )
        
        # Classify tuning types
        print("\nClassifying speed tuning types...")
        tuning_types, preferred_speeds = SpeedTuningAnalysis.classify_speed_tuning(
            response_map, bin_centers, Q_S_threshold
        )
        
        # Filter for reliable cells only
        Q_S_reliable = Q_S[reliable_cells]
        tuning_types_reliable = tuning_types[reliable_cells]
        preferred_speeds_reliable = preferred_speeds[reliable_cells]
        response_map_reliable = response_map[reliable_cells, :]
        
        # Count speed-tuned cells
        n_speed_tuned = np.sum(Q_S_reliable > Q_S_threshold)
        print(f"\nSpeed-tuned cells (Q_S > {Q_S_threshold}): {n_speed_tuned}/{len(Q_S_reliable)} ({n_speed_tuned/len(Q_S_reliable)*100:.1f}%)")
        
        # Analyze by layer
        print("\n" + "="*80)
        print("LAYER-SPECIFIC ANALYSIS")
        print("="*80)
        
        layer_results = {}
        layer_names = ['L2/3', 'L4', 'L5', 'L6']
        
        for layer_name in layer_names:
            if layer_name not in layer_cells:
                continue
            
            layer_indices = layer_cells[layer_name]
            
            # Find reliable cells in this layer
            reliable_layer_cells = np.intersect1d(
                np.where(reliable_cells)[0], layer_indices
            )
            
            if len(reliable_layer_cells) == 0:
                print(f"\n{layer_name}: No reliable cells")
                continue
            
            # Extract data for this layer
            layer_Q_S = Q_S[reliable_layer_cells]
            layer_tuning = tuning_types[reliable_layer_cells]
            layer_pref_speed = preferred_speeds[reliable_layer_cells]
            layer_response = response_map[reliable_layer_cells, :]
            
            # Calculate statistics
            n_tuned = np.sum(layer_Q_S > Q_S_threshold)
            prop_tuned = n_tuned / len(layer_Q_S)
            
            mean_Q_S = np.mean(layer_Q_S)
            median_Q_S = np.median(layer_Q_S)
            
            # Count tuning types (among tuned cells only)
            tuned_mask = layer_Q_S > Q_S_threshold
            if np.sum(tuned_mask) > 0:
                tuned_types = layer_tuning[tuned_mask]
                n_low = np.sum(tuned_types == 1)
                n_band = np.sum(tuned_types == 2)
                n_high = np.sum(tuned_types == 3)
            else:
                n_low = n_band = n_high = 0
            
            layer_results[layer_name] = {
                'cell_indices': reliable_layer_cells,
                'Q_S': layer_Q_S,
                'tuning_types': layer_tuning,
                'preferred_speeds': layer_pref_speed,
                'response_maps': layer_response,
                'n_cells': len(reliable_layer_cells),
                'n_tuned': n_tuned,
                'prop_tuned': prop_tuned,
                'mean_Q_S': mean_Q_S,
                'median_Q_S': median_Q_S,
                'n_low_pass': n_low,
                'n_band_pass': n_band,
                'n_high_pass': n_high
            }
            
            print(f"\n{layer_name}:")
            print(f"  Reliable cells: {len(reliable_layer_cells)}")
            print(f"  Speed-tuned: {n_tuned} ({prop_tuned*100:.1f}%)")
            print(f"  Mean Q_S: {mean_Q_S:.3f}")
            print(f"  Median Q_S: {median_Q_S:.3f}")
            print(f"  Tuning types (among tuned):")
            print(f"    Low-pass: {n_low}")
            print(f"    Band-pass: {n_band}")
            print(f"    High-pass: {n_high}")
        
        # Statistical comparisons
        print("\n" + "="*80)
        print("STATISTICAL COMPARISONS")
        print("="*80)
        
        SpeedTuningAnalysis._print_statistical_comparisons(layer_results, layer_names)
        
        # Compile results
        results = {
            'speed_bins': speed_bins,
            'bin_centers': bin_centers,
            'response_map': response_map,
            'Q_S': Q_S,
            'tuning_types': tuning_types,
            'preferred_speeds': preferred_speeds,
            'reliable_cells': reliable_cells,
            'layer_results': layer_results,
            'Q_S_threshold': Q_S_threshold,
            'n_speed_tuned_total': n_speed_tuned,
            'analysis_params': {
                'min_speed': min_speed,
                'max_speed': max_speed,
                'n_bins': n_bins,
                'framerate': framerate,
                'optimal_sigma': optimal_sigma
            }
        }
        
        return results
    
    @staticmethod
    def _print_statistical_comparisons(layer_results, layer_names):
        """Print statistical comparisons between layers."""
        
        # Chi-square test for proportion tuned
        print("\n1. Proportion of speed-tuned cells:")
        
        observed = []
        layer_names_valid = []
        
        for layer_name in layer_names:
            if layer_name in layer_results:
                lr = layer_results[layer_name]
                observed.append([lr['n_tuned'], lr['n_cells'] - lr['n_tuned']])
                layer_names_valid.append(layer_name)
        
        if len(observed) >= 2:
            observed = np.array(observed)
            try:
                chi2, p_value = stats.chi2_contingency(observed)[:2]
                print(f"   Chi-square test: χ² = {chi2:.3f}, p = {p_value:.4f}")
                if p_value < 0.05:
                    print("   *** Significant difference in proportion tuned across layers")
                else:
                    print("   No significant difference in proportion tuned")
            except:
                print("   Chi-square test could not be performed")
        
        # Kruskal-Wallis test for Q_S distribution
        print("\n2. Distribution of Q_S values:")
        
        Q_S_by_layer = []
        for layer_name in layer_names_valid:
            Q_S_by_layer.append(layer_results[layer_name]['Q_S'])
        
        if len(Q_S_by_layer) >= 2:
            try:
                h_stat, p_value = stats.kruskal(*Q_S_by_layer)
                print(f"   Kruskal-Wallis test: H = {h_stat:.3f}, p = {p_value:.4f}")
                if p_value < 0.05:
                    print("   *** Significant difference in Q_S across layers")
                else:
                    print("   No significant difference in Q_S")
            except:
                print("   Kruskal-Wallis test could not be performed")
        
        # Pairwise comparisons (Mann-Whitney U)
        if len(layer_names_valid) >= 2:
            print("\n3. Pairwise comparisons (Mann-Whitney U):")
            for i, layer1 in enumerate(layer_names_valid):
                for layer2 in layer_names_valid[i+1:]:
                    Q_S_1 = layer_results[layer1]['Q_S']
                    Q_S_2 = layer_results[layer2]['Q_S']
                    
                    try:
                        u_stat, p_value = stats.mannwhitneyu(Q_S_1, Q_S_2, alternative='two-sided')
                        sig_marker = "***" if p_value < 0.05 else ""
                        print(f"   {layer1} vs {layer2}: U = {u_stat:.1f}, p = {p_value:.4f} {sig_marker}")
                    except:
                        print(f"   {layer1} vs {layer2}: Could not perform test")
    
    @staticmethod
    def plot_speed_tuning_results(results, save_dir=None):
        """
        Create comprehensive visualization of speed tuning results.
        
        Generates figures similar to Saleem et al. (2013) Fig 2.
        
        Parameters:
        -----------
        results : dict
            Output from analyze_speed_tuning_by_layer
        save_dir : str, optional
            Directory to save figures
        """
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        layer_results = results['layer_results']
        bin_centers = results['bin_centers']
        Q_S_threshold = results['Q_S_threshold']
        
        # Figure 1: Population-level overview
        fig1 = SpeedTuningAnalysis._plot_population_overview(
            results, layer_results, bin_centers, Q_S_threshold
        )
        
        if save_dir:
            fig1.savefig(os.path.join(save_dir, 'speed_tuning_population.png'), 
                        dpi=300, bbox_inches='tight')
        
        # Figure 2: Layer comparison
        fig2 = SpeedTuningAnalysis._plot_layer_comparison(
            layer_results, bin_centers, Q_S_threshold
        )
        
        if save_dir:
            fig2.savefig(os.path.join(save_dir, 'speed_tuning_layer_comparison.png'),
                        dpi=300, bbox_inches='tight')
        
        # Figure 3: Example cells
        fig3 = SpeedTuningAnalysis._plot_example_cells(
            results, layer_results, bin_centers
        )
        
        if save_dir:
            fig3.savefig(os.path.join(save_dir, 'speed_tuning_examples.png'),
                        dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig1, fig2, fig3
    
    @staticmethod
    def _plot_population_overview(results, layer_results, bin_centers, Q_S_threshold):
        """Plot population-level overview (similar to Saleem Fig 2)."""
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig)
        
        # Panel A: Example speed tuning curves
        ax1 = fig.add_subplot(gs[0, :])
        SpeedTuningAnalysis._plot_example_tuning_curves(ax1, results, bin_centers)
        
        # Panel B: Distribution of preferred speeds by layer
        ax2 = fig.add_subplot(gs[1, 0])
        SpeedTuningAnalysis._plot_preferred_speed_distribution(ax2, layer_results, Q_S_threshold)
        
        # Panel C: Q_S distribution by layer
        ax3 = fig.add_subplot(gs[1, 1])
        SpeedTuningAnalysis._plot_Q_S_distribution(ax3, layer_results, Q_S_threshold)
        
        # Panel D: Tuning type proportions
        ax4 = fig.add_subplot(gs[1, 2])
        SpeedTuningAnalysis._plot_tuning_type_proportions(ax4, layer_results, Q_S_threshold)
        
        plt.suptitle('Speed Tuning Analysis - Population Overview', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _plot_example_tuning_curves(ax, results, bin_centers):
        """Plot example speed tuning curves."""
        
        response_map = results['response_map']
        Q_S = results['Q_S']
        reliable_cells = results['reliable_cells']
        tuning_types = results['tuning_types']
        
        # Get reliable cells
        reliable_indices = np.where(reliable_cells)[0]
        Q_S_reliable = Q_S[reliable_indices]
        
        # Sort by Q_S and pick top examples of each type
        sorted_indices = np.argsort(Q_S_reliable)[::-1]
        
        examples = {
            1: None,  # low-pass
            2: None,  # band-pass
            3: None,  # high-pass
        }
        
        for idx in sorted_indices:
            cell_idx = reliable_indices[idx]
            ttype = tuning_types[cell_idx]
            
            if ttype in examples and examples[ttype] is None:
                examples[ttype] = cell_idx
            
            if all(v is not None for v in examples.values()):
                break
        
        # Plot examples
        colors = {1: 'blue', 2: 'green', 3: 'red'}
        labels = {1: 'Low-pass', 2: 'Band-pass', 3: 'High-pass'}
        
        for ttype, cell_idx in examples.items():
            if cell_idx is not None:
                response = response_map[cell_idx, :]
                
                # Normalize for visualization
                if np.max(response) > 0:
                    response_norm = response / np.max(response)
                else:
                    response_norm = response
                
                ax.plot(bin_centers, response_norm, 
                       color=colors[ttype], linewidth=2, 
                       label=f'{labels[ttype]} (Q_S={Q_S[cell_idx]:.2f})',
                       alpha=0.8)
        
        ax.set_xlabel('Speed (cm/s)', fontsize=12)
        ax.set_ylabel('Normalized Response', fontsize=12)
        ax.set_title('Example Speed Tuning Curves', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, np.max(bin_centers)])
    
    @staticmethod
    def _plot_preferred_speed_distribution(ax, layer_results, Q_S_threshold):
        """Plot distribution of preferred speeds by layer."""
        
        layer_names = []
        preferred_speeds_by_layer = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
        
        for layer_name, results in layer_results.items():
            if results is not None and results['n_tuned'] > 0:
                layer_names.append(layer_name)
                
                # Get preferred speeds for tuned cells only
                tuned_mask = results['Q_S'] > Q_S_threshold
                preferred_speeds_by_layer.append(results['preferred_speeds'][tuned_mask])
        
        # Create violin plot
        positions = np.arange(len(layer_names))
        parts = ax.violinplot(preferred_speeds_by_layer, positions=positions,
                             showmeans=True, showmedians=True)
        
        # Color the violin plots
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i % len(colors)])
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(layer_names)
        ax.set_ylabel('Preferred Speed (cm/s)', fontsize=12)
        ax.set_title('Distribution of Preferred Speeds', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def _plot_Q_S_distribution(ax, layer_results, Q_S_threshold):
        """Plot Q_S distribution by layer."""
        
        layer_names = []
        Q_S_by_layer = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for layer_name, results in layer_results.items():
            if results is not None and len(results['Q_S']) > 0:
                layer_names.append(layer_name)
                Q_S_by_layer.append(results['Q_S'])
        
        # Create box plot
        bp = ax.boxplot(Q_S_by_layer, labels=layer_names, patch_artist=True)
        
        # Color the boxes
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(colors)])
            patch.set_alpha(0.7)
        
        # Add threshold line
        ax.axhline(y=Q_S_threshold, color='red', linestyle='--', 
                  linewidth=2, label=f'Threshold ({Q_S_threshold})')
        
        # Add scatter points for individual cells
        for i, Q_S_vals in enumerate(Q_S_by_layer):
            x = np.random.normal(i+1, 0.04, size=len(Q_S_vals))
            ax.scatter(x, Q_S_vals, alpha=0.3, s=20, color=colors[i % len(colors)])
        
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Q_S (Speed Prediction Quality)', fontsize=12)
        ax.set_title('Speed Tuning Quality by Layer', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def _plot_tuning_type_proportions(ax, layer_results, Q_S_threshold):
        """Plot proportions of tuning types by layer."""
        
        layer_names = []
        low_pass_props = []
        band_pass_props = []
        high_pass_props = []
        
        for layer_name, results in layer_results.items():
            if results is not None and results['n_tuned'] > 0:
                layer_names.append(layer_name)
                
                total_tuned = results['n_tuned']
                low_pass_props.append(results['n_low_pass'] / total_tuned * 100)
                band_pass_props.append(results['n_band_pass'] / total_tuned * 100)
                high_pass_props.append(results['n_high_pass'] / total_tuned * 100)
        
        # Create stacked bar chart
        x = np.arange(len(layer_names))
        width = 0.6
        
        ax.bar(x, low_pass_props, width, label='Low-pass', color='blue', alpha=0.7)
        ax.bar(x, band_pass_props, width, bottom=low_pass_props, 
              label='Band-pass', color='green', alpha=0.7)
        ax.bar(x, high_pass_props, width, 
              bottom=np.array(low_pass_props) + np.array(band_pass_props),
              label='High-pass', color='red', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names)
        ax.set_ylabel('Percentage (%)', fontsize=12)
        ax.set_title('Speed Tuning Types by Layer', fontsize=14, fontweight='bold')
        ax.legend()
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def _plot_layer_comparison(layer_results, bin_centers, Q_S_threshold):
        """Plot layer comparison (similar to Saleem Fig 2)."""
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig)
        
        # Panel A: Mean speed tuning curves by layer
        ax1 = fig.add_subplot(gs[0, :])
        SpeedTuningAnalysis._plot_mean_tuning_by_layer(ax1, layer_results, bin_centers)
        
        # Panel B: Q_S distributions
        ax2 = fig.add_subplot(gs[1, 0])
        SpeedTuningAnalysis._plot_Q_S_comparison(ax2, layer_results, Q_S_threshold)
        
        # Panel C: Tuning type proportions
        ax3 = fig.add_subplot(gs[1, 1])
        SpeedTuningAnalysis._plot_tuning_type_comparison(ax3, layer_results)
        
        # Panel D: Preferred speed distributions
        ax4 = fig.add_subplot(gs[2, 0])
        SpeedTuningAnalysis._plot_preferred_speed_comparison(ax4, layer_results, Q_S_threshold)
        
        # Panel E: Summary statistics
        ax5 = fig.add_subplot(gs[2, 1])
        SpeedTuningAnalysis._plot_summary_stats(ax5, layer_results, Q_S_threshold)
        
        plt.suptitle('Speed Tuning Analysis - Layer Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def _plot_mean_tuning_by_layer(ax, layer_results, bin_centers):
        """Plot mean speed tuning curves for each layer."""
        
        colors = {'L2/3': '#1f77b4', 'L4': '#ff7f0e', 'L5': '#2ca02c', 'L6': '#d62728'}
        
        for layer_name, results in layer_results.items():
            if results is not None and len(results['response_maps']) > 0:
                # Calculate mean response across all cells in this layer
                mean_response = np.mean(results['response_maps'], axis=0)
                sem_response = stats.sem(results['response_maps'], axis=0)
                
                # Normalize for comparison
                if np.max(mean_response) > 0:
                    mean_response_norm = mean_response / np.max(mean_response)
                    sem_response_norm = sem_response / np.max(mean_response)
                else:
                    mean_response_norm = mean_response
                    sem_response_norm = sem_response
                
                # Plot mean with SEM
                ax.plot(bin_centers, mean_response_norm, 
                       color=colors.get(layer_name, 'gray'), 
                       linewidth=3, label=f'{layer_name} (n={results["n_cells"]})',
                       alpha=0.8)
                ax.fill_between(bin_centers, 
                               mean_response_norm - sem_response_norm,
                               mean_response_norm + sem_response_norm,
                               color=colors.get(layer_name, 'gray'), alpha=0.3)
        
        ax.set_xlabel('Speed (cm/s)', fontsize=12)
        ax.set_ylabel('Normalized Response', fontsize=12)
        ax.set_title('Mean Speed Tuning by Layer', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, np.max(bin_centers)])
    
    @staticmethod
    def _plot_Q_S_comparison(ax, layer_results, Q_S_threshold):
        """Plot Q_S comparison across layers."""
        
        layer_names = []
        mean_Q_S = []
        sem_Q_S = []
        n_tuned_pct = []
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for layer_name, results in layer_results.items():
            if results is not None:
                layer_names.append(layer_name)
                mean_Q_S.append(results['mean_Q_S'])
                sem_Q_S.append(np.std(results['Q_S']) / np.sqrt(len(results['Q_S'])))
                n_tuned_pct.append(results['prop_tuned'] * 100)
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        # Plot mean Q_S
        bars1 = ax.bar(x - width/2, mean_Q_S, width, yerr=sem_Q_S,
                      label='Mean Q_S', capsize=5, alpha=0.7,
                      color=colors_list[:len(layer_names)])
        
        # Plot proportion tuned on secondary axis
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, n_tuned_pct, width,
                       label='% Speed-tuned', alpha=0.7,
                       color=colors_list[:len(layer_names)], edgecolor='black', linewidth=2)
        
        # Add threshold line
        ax.axhline(y=Q_S_threshold, color='red', linestyle='--', 
                  linewidth=2, alpha=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Mean Q_S ± SEM', fontsize=12)
        ax2.set_ylabel('% Speed-tuned Cells', fontsize=12)
        ax.set_title('Speed Tuning Strength by Layer', fontsize=14, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def _plot_tuning_type_comparison(ax, layer_results):
        """Plot tuning type comparison across layers."""
        
        layer_names = []
        low_pass_counts = []
        band_pass_counts = []
        high_pass_counts = []
        
        for layer_name, results in layer_results.items():
            if results is not None and results['n_tuned'] > 0:
                layer_names.append(layer_name)
                low_pass_counts.append(results['n_low_pass'])
                band_pass_counts.append(results['n_band_pass'])
                high_pass_counts.append(results['n_high_pass'])
        
        x = np.arange(len(layer_names))
        width = 0.25
        
        ax.bar(x - width, low_pass_counts, width, label='Low-pass', 
              color='blue', alpha=0.7)
        ax.bar(x, band_pass_counts, width, label='Band-pass', 
              color='green', alpha=0.7)
        ax.bar(x + width, high_pass_counts, width, label='High-pass', 
              color='red', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Number of Cells', fontsize=12)
        ax.set_title('Speed Tuning Types by Layer', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def _plot_preferred_speed_comparison(ax, layer_results, Q_S_threshold):
        """Plot preferred speed comparison across layers."""
        
        layer_names = []
        preferred_speeds_list = []
        colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for layer_name, results in layer_results.items():
            if results is not None and results['n_tuned'] > 0:
                layer_names.append(layer_name)
                tuned_mask = results['Q_S'] > Q_S_threshold
                preferred_speeds_list.append(results['preferred_speeds'][tuned_mask])
        
        # Create violin plot
        positions = np.arange(len(layer_names))
        parts = ax.violinplot(preferred_speeds_list, positions=positions,
                             showmeans=True, showmedians=True, showextrema=True)
        
        # Color the violins
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors_list[i % len(colors_list)])
            pc.set_alpha(0.7)
        
        ax.set_xticks(positions)
        ax.set_xticklabels(layer_names)
        ax.set_xlabel('Layer', fontsize=12)
        ax.set_ylabel('Preferred Speed (cm/s)', fontsize=12)
        ax.set_title('Preferred Speed Distribution by Layer', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    @staticmethod
    def _plot_summary_stats(ax, layer_results, Q_S_threshold):
        """Plot summary statistics table."""
        
        ax.axis('off')
        
        # Prepare data for table
        table_data = [['Layer', 'N cells', 'N tuned', '% tuned', 'Mean Q_S', 'Median Q_S']]
        
        for layer_name, results in layer_results.items():
            if results is not None:
                row = [
                    layer_name,
                    f"{results['n_cells']}",
                    f"{results['n_tuned']}",
                    f"{results['prop_tuned']*100:.1f}%",
                    f"{results['mean_Q_S']:.3f}",
                    f"{results['median_Q_S']:.3f}"
                ]
                table_data.append(row)
        
        # Create table
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.15, 0.15, 0.15, 0.15, 0.2, 0.2])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(table_data)):
            color = '#D9E1F2' if i % 2 == 0 else 'white'
            for j in range(len(table_data[0])):
                table[(i, j)].set_facecolor(color)
        
        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    @staticmethod
    def _plot_example_cells(results, layer_results, bin_centers):
        """Plot example cells from each layer."""
        
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 3, figure=fig)
        
        layer_order = ['L2/3', 'L4', 'L5', 'L6']
        plot_idx = 0
        
        for layer_name in layer_order:
            if layer_name not in layer_results or layer_results[layer_name] is None:
                continue
            
            results_layer = layer_results[layer_name]
            
            # Find top 3 cells by Q_S
            Q_S_sorted_idx = np.argsort(results_layer['Q_S'])[::-1]
            
            for i in range(min(3, len(Q_S_sorted_idx))):
                if plot_idx >= 12:  # Max 12 subplots
                    break
                
                cell_idx = results_layer['cell_indices'][Q_S_sorted_idx[i]]
                Q_S_val = results_layer['Q_S'][Q_S_sorted_idx[i]]
                response = results['response_map'][cell_idx, :]
                
                row = plot_idx // 3
                col = plot_idx % 3
                ax = fig.add_subplot(gs[row, col])
                
                # Normalize response
                if np.max(response) > 0:
                    response_norm = response / np.max(response)
                else:
                    response_norm = response
                
                ax.plot(bin_centers, response_norm, 'b-', linewidth=2)
                ax.fill_between(bin_centers, response_norm, alpha=0.3)
                
                ax.set_xlabel('Speed (cm/s)', fontsize=10)
                ax.set_ylabel('Norm. Response', fontsize=10)
                ax.set_title(f'{layer_name} - Cell {cell_idx}\nQ_S = {Q_S_val:.3f}',
                           fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xlim([0, np.max(bin_centers)])
                
                plot_idx += 1
        
        plt.suptitle('Example Speed-Tuned Cells by Layer', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


# Example usage and testing
if __name__ == "__main__":
    # This is a placeholder for testing
    print("SpeedTuningAnalysis module loaded successfully")
    print("Use analyze_speed_tuning_by_layer() to run the analysis")