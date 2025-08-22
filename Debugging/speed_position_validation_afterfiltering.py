

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats
    import seaborn as sns

    def validate_speed_position_patterns(location_data, framerate, bin_size_cm=5, figsize=(16, 12)):
        """
        Create comprehensive validation plots for speed vs position patterns.
        
        Parameters:
        -----------
        location_data : numpy.ndarray
            VR location data in arbitrary units (before conversion)
        framerate : float
            Recording framerate in Hz
        bin_size_cm : float
            Size of position bins for analysis in cm
        figsize : tuple
            Figure size for the plots
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure containing validation plots
        validation_stats : dict
            Dictionary containing validation statistics
        """
        
        # Calculate speed and convert to physical units (same as in your pipeline)
        from helper.BehavioralDataFiltering import calculate_vr_speed_and_distance
        speed_cm_s, location_cm, conversion_factor = calculate_vr_speed_and_distance(location_data, framerate)
        
        print(f"Validation Analysis:")
        print(f"  Corridor length: {np.max(location_cm) - np.min(location_cm):.1f} cm")
        print(f"  Speed range: {np.min(speed_cm_s):.1f} to {np.max(speed_cm_s):.1f} cm/s")
        print(f"  Mean speed: {np.mean(speed_cm_s):.1f} cm/s")
        print(f"  Conversion factor: {conversion_factor:.4f} cm/AU")
        
        # Create position bins
        min_pos, max_pos = np.min(location_cm), np.max(location_cm)
        n_bins = int((max_pos - min_pos) / bin_size_cm)
        pos_bins = np.linspace(min_pos, max_pos, n_bins + 1)
        pos_centers = (pos_bins[:-1] + pos_bins[1:]) / 2
        
        # Assign each timepoint to a position bin
        bin_indices = np.digitize(location_cm, pos_bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Calculate statistics for each position bin
        mean_speed_by_pos = np.zeros(n_bins)
        std_speed_by_pos = np.zeros(n_bins)
        median_speed_by_pos = np.zeros(n_bins)
        occupancy_by_pos = np.zeros(n_bins)
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.any(bin_mask):
                speeds_in_bin = speed_cm_s[bin_mask]
                mean_speed_by_pos[i] = np.mean(speeds_in_bin)
                std_speed_by_pos[i] = np.std(speeds_in_bin)
                median_speed_by_pos[i] = np.median(speeds_in_bin)
                occupancy_by_pos[i] = np.sum(bin_mask)
        
        # Create the validation figure
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # Plot 1: Speed vs Position Scatter (sampled for visibility)
        ax = axes[0, 0]
        n_samples = min(10000, len(location_cm))  # Sample for better visualization
        sample_indices = np.random.choice(len(location_cm), n_samples, replace=False)
        
        scatter = ax.scatter(location_cm[sample_indices], speed_cm_s[sample_indices], 
                            alpha=0.3, s=1, c=speed_cm_s[sample_indices], 
                            cmap='viridis', vmin=0, vmax=50)
        
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Speed (cm/s)')
        ax.set_title(f'Speed vs Position\n(n={n_samples:,} sampled points)')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Speed (cm/s)')
        
        # Plot 2: Mean Speed by Position
        ax = axes[0, 1]
        ax.plot(pos_centers, mean_speed_by_pos, 'b-', linewidth=2, label='Mean')
        ax.fill_between(pos_centers, 
                        mean_speed_by_pos - std_speed_by_pos,
                        mean_speed_by_pos + std_speed_by_pos,
                        alpha=0.3, color='blue', label='±1 SD')
        ax.plot(pos_centers, median_speed_by_pos, 'r--', linewidth=2, label='Median')
        
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Speed (cm/s)')
        ax.set_title('Speed Statistics by Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Occupancy by Position
        ax = axes[0, 2]
        ax.bar(pos_centers, occupancy_by_pos, width=bin_size_cm*0.8, alpha=0.7, color='green')
        ax.set_xlabel('Position (cm)')
        ax.set_ylabel('Time Spent (frames)')
        ax.set_title('Spatial Occupancy')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Speed Distribution by Corridor Segment
        ax = axes[1, 0]
        
        # Divide corridor into segments (start, middle, end)
        corridor_length = max_pos - min_pos
        segment_bounds = [
            min_pos,
            min_pos + corridor_length * 0.33,
            min_pos + corridor_length * 0.67,
            max_pos
        ]
        
        segment_labels = ['Start\n(0-33%)', 'Middle\n(33-67%)', 'End\n(67-100%)']
        segment_colors = ['red', 'blue', 'green']
        
        for i, (start, end, label, color) in enumerate(zip(segment_bounds[:-1], segment_bounds[1:], 
                                                        segment_labels, segment_colors)):
            segment_mask = (location_cm >= start) & (location_cm < end)
            segment_speeds = speed_cm_s[segment_mask]
            
            if len(segment_speeds) > 0:
                ax.hist(segment_speeds, bins=30, alpha=0.6, label=label, 
                    color=color, density=True)
        
        ax.set_xlabel('Speed (cm/s)')
        ax.set_ylabel('Density')
        ax.set_title('Speed Distribution by Corridor Segment')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Movement Direction Analysis
        ax = axes[1, 1]
        
        # Calculate movement direction (positive = forward)
        position_diff = np.diff(location_cm)
        forward_mask = position_diff > 0
        backward_mask = position_diff < 0
        
        forward_speeds = speed_cm_s[1:][forward_mask]
        backward_speeds = speed_cm_s[1:][backward_mask]
        
        if len(forward_speeds) > 0 and len(backward_speeds) > 0:
            ax.hist(forward_speeds, bins=30, alpha=0.6, label=f'Forward (n={len(forward_speeds):,})', 
                color='blue', density=True)
            ax.hist(backward_speeds, bins=30, alpha=0.6, label=f'Backward (n={len(backward_speeds):,})', 
                color='red', density=True)
            
            ax.set_xlabel('Speed (cm/s)')
            ax.set_ylabel('Density')
            ax.set_title('Speed Distribution by Movement Direction')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 6: Speed Consistency Check
        ax = axes[1, 2]
        
        # Calculate running average speed over different window sizes
        window_sizes = [10, 50, 100, 500]  # frames
        colors_window = ['red', 'blue', 'green', 'orange']
        
        for window_size, color in zip(window_sizes, colors_window):
            if window_size < len(speed_cm_s):
                # Calculate running average
                kernel = np.ones(window_size) / window_size
                smoothed_speed = np.convolve(speed_cm_s, kernel, mode='valid')
                
                # Sample for plotting
                n_plot = min(5000, len(smoothed_speed))
                indices = np.linspace(0, len(smoothed_speed)-1, n_plot, dtype=int)
                time_axis = indices / framerate
                
                ax.plot(time_axis, smoothed_speed[indices], alpha=0.7, 
                    label=f'{window_size} frames ({window_size/framerate:.1f}s)', 
                    color=color, linewidth=1)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (cm/s)')
        ax.set_title('Speed Consistency Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Calculate validation statistics
        validation_stats = {
            'corridor_length_cm': corridor_length,
            'mean_speed_overall': np.mean(speed_cm_s),
            'median_speed_overall': np.median(speed_cm_s),
            'speed_range': (np.min(speed_cm_s), np.max(speed_cm_s)),
            'mean_speed_by_position': mean_speed_by_pos,
            'position_centers': pos_centers,
            'occupancy_uniformity': np.std(occupancy_by_pos) / np.mean(occupancy_by_pos),  # Lower is more uniform
            'forward_backward_ratio': len(forward_speeds) / len(backward_speeds) if len(backward_speeds) > 0 else np.inf,
            'conversion_factor': conversion_factor
        }
        
        # Print validation summary
        print(f"\nValidation Summary:")
        print(f"  Overall mean speed: {validation_stats['mean_speed_overall']:.1f} cm/s")
        print(f"  Overall median speed: {validation_stats['median_speed_overall']:.1f} cm/s")
        print(f"  Speed range: {validation_stats['speed_range'][0]:.1f} to {validation_stats['speed_range'][1]:.1f} cm/s")
        print(f"  Occupancy uniformity (lower=better): {validation_stats['occupancy_uniformity']:.2f}")
        print(f"  Forward/backward movement ratio: {validation_stats['forward_backward_ratio']:.1f}")
        
        # Check for potential issues
        print(f"\nPotential Issues Check:")
        
        # Check 1: Speed consistency across positions
        speed_cv_by_pos = std_speed_by_pos / (mean_speed_by_pos + 1e-6)  # Coefficient of variation
        high_variability_positions = np.sum(speed_cv_by_pos > 2)
        print(f"  Positions with high speed variability: {high_variability_positions}/{n_bins}")
        
        # Check 2: Reasonable speed ranges
        extreme_speeds = np.sum(speed_cm_s > 100)  # Very fast for a mouse
        print(f"  Extreme speed points (>100 cm/s): {extreme_speeds} ({extreme_speeds/len(speed_cm_s)*100:.2f}%)")
        
        # Check 3: Movement directionality
        if validation_stats['forward_backward_ratio'] > 10 or validation_stats['forward_backward_ratio'] < 0.1:
            print(f"  WARNING: Highly biased movement direction (ratio: {validation_stats['forward_backward_ratio']:.1f})")
        else:
            print(f"  Movement direction balance: Good")
        
        # Check 4: Spatial coverage
        empty_bins = np.sum(occupancy_by_pos == 0)
        if empty_bins > n_bins * 0.1:  # More than 10% empty
            print(f"  WARNING: {empty_bins}/{n_bins} position bins are empty")
        else:
            print(f"  Spatial coverage: Good ({empty_bins}/{n_bins} empty bins)")
        
        return fig, validation_stats

    # Example usage function
    def run_speed_position_validation(twop_filepath, vr_filepath):
        """
        Run the complete validation analysis on your data.
        
        Parameters:
        -----------
        twop_filepath : str
            Path to two-photon data
        vr_filepath : str
            Path to VR behavioral data
        """
        # Load and align data (using your existing pipeline)
        from helper import dataLoader
        
        procData = dataLoader(twop_filepath, vr_filepath)
        animal_id, date, framerate = procData.load_data()
        twop_dict, vr_dict = procData.align_data()
        
        print(f"Running validation for {animal_id} on {date}")
        print(f"Framerate: {framerate} Hz")
        
        # Run validation
        fig, stats = validate_speed_position_patterns(
            vr_dict['interp_location'], 
            framerate,
            bin_size_cm=5
        )
        
        plt.show()
        
        return fig, stats

    fig, stats = run_speed_position_validation(twop_filepath, vr_filepath)

