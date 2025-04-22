import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import warnings

class SpatialModulationIndexLayerSpecific:
    """Class for analyzing spatial modulation in neural data across cortical layers."""
    
    @staticmethod
    def identify_layers(med_coords, peak_density_method='auto'):
        """
        Identify cortical layers based on cell coordinates.
        
        Parameters:
        -----------
        med_coords : numpy.ndarray
            Median coordinates of cells (cells x dimensions)
        peak_density_method : str
            Method to identify peak density ('auto' or 'manual')
            
        Returns:
        --------
        layer_cells : dict
            Dictionary with indices of cells in each layer
        layer_boundaries : dict
            Dictionary with layer boundaries
        """
        # Calculate cell density along y-axis
        density, bins = np.histogram(med_coords[:, 0], bins=50)
        bin_centers_density = (bins[:-1] + bins[1:]) / 2
        
        # Smooth density for more reliable peak detection
        sigma = 1
        smooth_density = gaussian_filter1d(density, sigma)
        
        if peak_density_method == 'auto':
            # Find peak density (Layer 4)
            peak_idx = np.argmax(smooth_density)
            peak_density_y = bin_centers_density[peak_idx]
            print(f'Automatically detected peak density at y = {peak_density_y}')
        else:
            # Use middle of the distribution as an estimate
            peak_density_y = np.median(med_coords[:, 0])
            print(f'Using median y-value as peak density: y = {peak_density_y}')
        
        # Define layer boundaries based on distance from peak
        
        # for regular objective, each pixel corresponds to 1 bin and 1 bin = 0.947408849697405 um. Find bins +/- 70um away from peak_density
        # for cousa, each pixel corresponds to 1 bin and 1 bin = 1.5076603 um. Find bins +/- 70um away from peak_density
        um_per_bin = 0.947408849697405
        
        # Layer thicknesses based on mouse visual cortex anatomy
        layer4_half_width_um = 70  # μm from peak (±)
        layer4_half_width_bins = int(layer4_half_width_um / um_per_bin)
        
        # Calculate layer boundaries
        layer4_upper = peak_density_y - layer4_half_width_bins 
        layer4_lower = peak_density_y + layer4_half_width_bins 
        layer23_upper = np.min(med_coords[:, 0])
        layer23_lower = layer4_upper
        layer5_upper = layer4_lower
        layer5_lower = layer4_lower + int(150 / um_per_bin)   # 150μm below L4
        layer6_upper = layer5_lower
        layer6_lower = np.max(med_coords[:, 0])
        
        print(f'Layer 2/3 range: {layer23_upper} to {layer23_lower}')
        print(f'Layer 4 range: {layer4_upper} to {layer4_lower}')
        print(f'Layer 5 range: {layer5_upper} to {layer5_lower}')
        print(f'Layer 6 range: {layer6_upper} to {layer6_lower}')
        
        # Get indices of cells in each layer
        layer23_cells = np.where((med_coords[:, 0] >= layer23_upper) & (med_coords[:, 0] < layer23_lower))[0]
        layer4_cells = np.where((med_coords[:, 0] >= layer4_upper) & (med_coords[:, 0] < layer4_lower))[0]
        layer5_cells = np.where((med_coords[:, 0] >= layer5_upper) & (med_coords[:, 0] < layer5_lower))[0]
        layer6_cells = np.where(med_coords[:, 0] >= layer6_upper)[0]
        
        # Create the layer_cells dictionary
        layer_cells = {
            'L2/3': layer23_cells,
            'L4': layer4_cells,
            'L5': layer5_cells,
            'L6': layer6_cells
        }
        
        # Create dictionary of layer boundaries for plotting
        layer_boundaries = {
            'L2/3': (layer23_upper, layer23_lower),
            'L4': (layer4_upper, layer4_lower),
            'L5': (layer5_upper, layer5_lower),
            'L6': (layer6_upper, layer6_lower)
        }
        
        # Print number of cells in each layer
        for layer, cells in layer_cells.items():
            print(f"{layer}: {len(cells)} cells")
            
        return layer_cells, layer_boundaries
        
    @staticmethod
    def calculate_SMI(spatial_activity, bin_centers, reliable_cells, segment_distance=55, exclude_boundary_cm=10):
        """
        Calculate the Spatial Modulation Index (SMI) using cross-validation approach.
        - Odd trials to find preferred position
        - Even trials to measure responses
        - SMI = (Rp - Rn) / (Rp + Rn)
        
        Where Rp = response at preferred position, Rn = response at non-preferred position.
        
        Parameters:
        -----------
        spatial_activity : numpy.ndarray
            Activity matrix (n_cells x n_trials x n_spatial_bins)
        bin_centers : numpy.ndarray
            Centers of spatial bins
        segment_distance : float
            Distance between repeated segments in the VR environment
        exclude_boundary_cm : float
            Distance from corridor boundaries to exclude for preferred positions
            
        Returns:
        --------
        results : dict
            Dictionary with SMI calculation results
        """
        n_cells, n_trials, n_bins = spatial_activity.shape
        
        # Separate odd and even trials
        odd_indices = np.arange(0, n_trials, 2)
        even_indices = np.arange(1, n_trials, 2)
        
        # Calculate corridor boundaries
        min_pos = np.min(bin_centers)
        max_pos = np.max(bin_centers)
        corridor_length = max_pos - min_pos
        
        # Calculate boundary positions in the original coordinate system
        min_allowed = min_pos + exclude_boundary_cm
        max_allowed = max_pos - exclude_boundary_cm
        
        # print(f"  Corridor length: {corridor_length:.2f}")
        # print(f"  Valid position range: {min_allowed:.2f} to {max_allowed:.2f}")
        
        # Check if segment distance is compatible with the corridor length
        if segment_distance > corridor_length:
            print(f"WARNING: Segment distance ({segment_distance}) is larger than corridor length ({corridor_length:.2f})!")
            # Try to auto-correct by using half the corridor length
            segment_distance = corridor_length / 2
            print(f"  Auto-correcting segment distance to {segment_distance:.2f}")
        
        # Compute response profiles for odd and even trials
        odd_profiles = np.mean(spatial_activity[:, odd_indices, :], axis=1)
        even_profiles = np.mean(spatial_activity[:, even_indices, :], axis=1)
        
        # Initialize arrays to store results
        SMI_values = np.zeros(n_cells)
        preferred_positions = np.zeros(n_cells)
        non_preferred_positions = np.zeros(n_cells)
        Rp_values = np.zeros(n_cells)
        Rn_values = np.zeros(n_cells)
        valid_cells = np.zeros_like(reliable_cells, dtype=bool)
        
        # Count various rejection reasons
        outside_boundary_count = 0
        nonpref_outside_range_count = 0
        zero_response_count = 0
        valid_count = 0
        
        for cell in range(n_cells):
        # Find the preferred position from odd trials
            preferred_idx_odd = np.argmax(odd_profiles[cell])
            preferred_position_odd = bin_centers[preferred_idx_odd]

            # Check if the preferred position is within allowed boundaries
            if preferred_position_odd < min_allowed or preferred_position_odd > max_allowed:
                outside_boundary_count += 1
                if cell < len(valid_cells):
                    valid_cells[cell] = False
                continue

            # Find the maximum response in even trials within ±5 indices from preferred_idx_odd
            start_idx_pref = max(0, preferred_idx_odd - 5)
            end_idx_pref = min(n_bins, preferred_idx_odd + 5)
            window_profile_pref = even_profiles[cell, start_idx_pref:end_idx_pref]
            window_max_idx_pref = np.argmax(window_profile_pref)
            preferred_idx_even = start_idx_pref + window_max_idx_pref
            preferred_position_even = bin_centers[preferred_idx_even]

            # Calculate the non-preferred position (both possibilities)
            corridor_midpoint = min_pos + corridor_length / 2
            if preferred_position_even < corridor_midpoint:
                # If in first segment, the non-preferred position is in second segment
                non_preferred_position_approx = preferred_position_even + segment_distance
            else:
                # If in second segment, the non-preferred position is in first segment
                non_preferred_position_approx = preferred_position_even - segment_distance

            # Check if non-preferred position is within corridor bounds
            max_pos = np.max(bin_centers)
            if non_preferred_position_approx < min_pos or non_preferred_position_approx > max_pos:
                nonpref_outside_range_count += 1
                valid_cells[cell] = False
                continue

            # Find the closest bin to the non-preferred position
            non_preferred_idx_approx = np.argmin(np.abs(bin_centers - non_preferred_position_approx))

            # Find the maximum response within ±5 indices of the non-preferred position
            start_idx_nonpref = max(0, non_preferred_idx_approx - 5)
            end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 5)
            window_profile_nonpref = even_profiles[cell, start_idx_nonpref:end_idx_nonpref]
            window_max_idx_nonpref = np.argmax(window_profile_nonpref)
            non_preferred_idx_even = start_idx_nonpref + window_max_idx_nonpref
            non_preferred_position_even = bin_centers[non_preferred_idx_even]
            non_preferred_resp = window_profile_nonpref[window_max_idx_nonpref]

            # Get responses at the adjusted preferred and non-preferred positions from EVEN trials
            Rp = even_profiles[cell, preferred_idx_even]
            Rn = non_preferred_resp  

            # Calculate SMI
            if Rp + Rn > 0:  # Avoid division by zero
                SMI = (Rp - Rn) / (Rp + Rn)
                valid_count += 1
            else:
                SMI = 0
                zero_response_count += 1
                valid_cells[cell] = False
                continue
            
            # Store results - use the adjusted positions from even trials
            SMI_values[cell] = SMI
            preferred_positions[cell] = preferred_position_even
            non_preferred_positions[cell] = non_preferred_position_even
            Rp_values[cell] = Rp
            Rn_values[cell] = Rn
            valid_cells[cell] = True
            
        # find cells that are true for both reliable_cells and valid_cells
        if reliable_cells is not None:
            reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
        
        # Print summary statistics
        print(f"\nSMI calculation summary:")
        print(f"  Total cells: {n_cells}")
        print(f"  Rejected - preferred position outside boundary: {outside_boundary_count} ({outside_boundary_count/n_cells*100:.1f}%)")
        print(f"  Rejected - non-preferred position outside corridor: {nonpref_outside_range_count} ({nonpref_outside_range_count/n_cells*100:.1f}%)")
        print(f"  Rejected - zero response sum: {zero_response_count} ({zero_response_count/n_cells*100:.1f}%)")
        print(f"  Valid cells: {np.sum(valid_cells)} ({np.sum(valid_cells)/n_cells*100:.1f}%)")
        print(f"  Reliable cells: {np.sum(reliable_cells)} ({np.sum(reliable_cells)/n_cells*100:.1f}%)")
        print(f"  Reliable&Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
        print()
        
        # Create result dictionary
        results = {
            'SMI': SMI_values,
            'preferred_positions': preferred_positions,
            'non_preferred_positions': non_preferred_positions,
            'Rp': Rp_values,
            'Rn': Rn_values,
            'valid_cells': valid_cells,
            'reliable_valid_cells': reliable_valid_cells if reliable_cells is not None else None,
            'parameters': {
                    'segment_distance': segment_distance,
                    'exclude_boundary_cm': exclude_boundary_cm,
                    'n_cells': n_cells,
                    'n_trials': n_trials,
                    'n_bins': n_bins,
                    'corridor_length': corridor_length,
                    'min_pos': min_pos,
                    'max_pos': max_pos
                }
        }
        
        return results

    @staticmethod
    def analyze_layer_specific_SMI(layer_cells, all_smi_results, reliable_cells):
        """
        Analyze layer-specific spatial modulation.
        
        Parameters:
        -----------
        layer_cells : dict
            Dictionary with indices of cells in each layer
        all_smi_results : dict
            SMI calculation results from calculate_SMI
            
        Returns:
        --------
        layer_results : dict
            Dictionary with SMI results for each layer
        classified_cell_count : int
            Number of reliable valid cells that were classified into layers
        total_reliable_valid_cells : int
            Total number of reliable valid cells
        """
        # Initialize results dictionary for each layer
        layer_results = {}
        reliable_layer_cells_num = {}
        for layer, cells in layer_cells.items():
            reliable_layer_cells = np.intersect1d(np.where(reliable_cells)[0], cells)
            reliable_layer_cells_num[layer] = len(reliable_layer_cells)
        
        # Get all reliable valid cells
        reliable_valid_cells = np.where(all_smi_results['reliable_valid_cells'])[0]
        total_reliable_valid_cells = len(reliable_valid_cells)
        
        # Track how many cells we've classified into layers
        classified_cell_count = 0
        
        # Process each layer
        for layer_name, layer_cell_indices in layer_cells.items():
            # Find reliable valid cells within this layer
            layer_reliable_indices = np.intersect1d(reliable_valid_cells, layer_cell_indices)
            classified_cell_count += len(layer_reliable_indices)
            
            if len(layer_reliable_indices) == 0:
                print(f"No reliable valid cells found in {layer_name}")
                layer_results[layer_name] = None
                continue
            
            # Extract SMI values and other metrics directly for these cells
            valid_smi = all_smi_results['SMI'][layer_reliable_indices]
            valid_pref_pos = all_smi_results['preferred_positions'][layer_reliable_indices]
            valid_nonpref_pos = all_smi_results['non_preferred_positions'][layer_reliable_indices]
            valid_rp = all_smi_results['Rp'][layer_reliable_indices]
            valid_rn = all_smi_results['Rn'][layer_reliable_indices]
            
            # Calculate statistics
            mean_smi = np.mean(valid_smi)
            median_smi = np.median(valid_smi)
            std_smi = np.std(valid_smi)
            sem_smi = stats.sem(valid_smi) if len(valid_smi) > 1 else 0
            
            # Statistical test against 1 using Wilcoxon signed-rank test
            try:
                w_stat, p_value = stats.wilcoxon(valid_smi - 1) if len(valid_smi) > 1 else (np.nan, np.nan)
                t_stat = w_stat  # Store the W statistic for consistency
            except:
                t_stat, p_value = np.nan, np.nan
                warnings.warn(f"Wilcoxon test failed for layer {layer_name}. This can happen if all values are identical.")
            
            # Store results
            layer_results[layer_name] = {
                'reliable_valid_cells': layer_reliable_indices,
                'SMI': valid_smi,
                'preferred_positions': valid_pref_pos,
                'non_preferred_positions': valid_nonpref_pos,
                'Rp': valid_rp,
                'Rn': valid_rn,
                'stats': {
                    'mean': mean_smi,
                    'median': median_smi,
                    'std': std_smi,
                    'sem': sem_smi,
                    't_stat': t_stat,
                    'p_value': p_value
                }
            }
            
            # Print summary
            print(f"{layer_name}: {len(layer_reliable_indices)} / {reliable_layer_cells_num[layer_name]} valid cells")
            print(f"  Mean SMI = {mean_smi:.3f} ± {sem_smi:.3f} (SEM)")
            print(f"  Median SMI = {median_smi:.3f}")
            if not np.isnan(p_value):
                print(f"  Wilcoxon test against 1: W = {t_stat:.3f}, p = {p_value:.5f}")
            print()
        
        # Check if all reliable valid cells were assigned to a layer
        if classified_cell_count < total_reliable_valid_cells:
            print(f"Warning: {total_reliable_valid_cells - classified_cell_count} reliable valid cells were not assigned to any layer!")
        elif classified_cell_count > total_reliable_valid_cells:
            print(f"Warning: {classified_cell_count - total_reliable_valid_cells} cells were counted multiple times (assigned to multiple layers)!")
        else:
            print(f"All {total_reliable_valid_cells} reliable valid cells were successfully assigned to layers.")
        
        return layer_results

    @staticmethod
    def plot_layer_comparison(layer_results):
        """
        Create comparative plots for layer-specific SMI results.
        
        Parameters:
        -----------
        layer_results : dict
            Dictionary with SMI results for each layer
        """
        # Extract layers with valid results
        valid_layers = [layer for layer, results in layer_results.items() 
                        if results is not None and len(results['SMI']) > 0]
        
        if len(valid_layers) == 0:
            print("No valid layers for visualization")
            return
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Box plot of SMI by layer
        ax = axes[0, 0]
        layer_data = [layer_results[layer]['SMI'] for layer in valid_layers]
        layer_names = [f"{layer}\n(n={len(layer_results[layer]['SMI'])})" for layer in valid_layers]
        
        sns.boxplot(data=layer_data, ax=ax)
        ax.set_xticklabels(layer_names)
        # ax.axhline(y=0, color='r', linestyle='--')
        ax.axhline(y=1, color='r', linestyle='--')
        ax.set_ylabel('Spatial Modulation Index (SMI)')
        ax.set_title('Distribution of SMI by Cortical Layer')
        
        # Add individual points
        for i, data in enumerate(layer_data):
            ax.scatter([i] * len(data), data, alpha=0.5, s=20, edgecolor='k', linewidth=0.5)
        ax.set_ylim(-1, 1.5)
        
        means = [layer_results[layer]['stats']['mean'] for layer in valid_layers]
        sems = [layer_results[layer]['stats']['sem'] for layer in valid_layers]
        
        # Add significance markers
        for i, layer in enumerate(valid_layers):
            p_value = layer_results[layer]['stats']['p_value']
            if p_value < 0.001:
                text = '***'
            elif p_value < 0.01:
                text = '**'
            elif p_value < 0.05:
                text = '*'
            else:
                text = 'ns'
            
            ax.text(i, 1.2, text, ha='center')
            
        # 2. Mean and SEM of SMI by layer
        ax = axes[0, 1]

        # Create bar plot
        bars = ax.bar(range(len(valid_layers)), means, yerr=sems, capsize=10, alpha=0.7)
        ax.set_xticks(range(len(valid_layers)))
        ax.set_xticklabels(valid_layers)
        ax.set_ylim(0, 1)
        # ax.axhline(y=1, color='r', linestyle='--')
        ax.set_ylabel('Mean SMI ± SEM')
        ax.set_title('Mean Spatial Modulation by Layer')
        
        # Add statistical comparison bars between layers
        # Calculate the maximum height needed for bars
        max_height = max(means) + max(sems) * 3  # Start bars above the error bars
        
        # Only add bars if we have at least 2 layers to compare
        if len(valid_layers) >= 2:
            # For non-adjacent comparisons, we need higher bars
            # Calculate how many bars we'll need to draw
            total_comparisons = sum(range(len(valid_layers)))
            
            # Create a dictionary to track significant comparisons
            significant_comparisons = []
            
            # First, find which comparisons are significant
            for i, layer1 in enumerate(valid_layers):
                for j, layer2 in enumerate(valid_layers[i+1:], i+1):
                    smi1 = layer_results[layer1]['SMI']
                    smi2 = layer_results[layer2]['SMI']
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(smi1, smi2, equal_var=False)
                    
                    # Store significant comparisons
                    if p_value < 0.05:
                        significant_comparisons.append((i, j, p_value))
            
            # Sort in a logical order: first by the first layer (i), then by the distance between layers
            # This ensures L2/3 vs L4 comes before L2/3 vs L5, etc.
            significant_comparisons.sort(key=lambda x: (x[0], x[1] - x[0]))
            
            # Draw the significant comparison bars
            bar_spacing = 0.1  # Vertical spacing between bars
            
            # Group comparisons by first layer index
            comparison_groups = {}
            for comp in significant_comparisons:
                i, j, p_value = comp
                if i not in comparison_groups:
                    comparison_groups[i] = []
                comparison_groups[i].append(comp)
                
            # Draw bars group by group
            current_height = max_height
            for i in sorted(comparison_groups.keys()):
                comparisons = comparison_groups[i]
                
                # Sort comparisons within each group by distance
                comparisons.sort(key=lambda x: x[1] - x[0])
                
                for bar_idx, (i, j, p_value) in enumerate(comparisons):
                    # Increment height for each bar in this group
                    bar_y = current_height + bar_spacing * bar_idx
                    
                    # Draw the horizontal line connecting the bars
                    ax.plot([i, j], [bar_y, bar_y], 'k-', linewidth=1.5)
                    
                    # Draw the vertical lines down to the bars
                    ax.plot([i, i], [bar_y, bar_y-0.05], 'k-', linewidth=1.5)
                    ax.plot([j, j], [bar_y, bar_y-0.05], 'k-', linewidth=1.5)
                    
                    # Add significance marker
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        continue  # Skip non-significant comparisons
                    
                    ax.text((i+j)/2, bar_y+0.02, sig_text, ha='center', va='bottom', fontsize=12)
                
                # Update height for next group
                if comparisons:
                    current_height = bar_y + bar_spacing *1  # Add extra space between groups
                
            # If we have significant comparisons, adjust y-axis limit
            if significant_comparisons:
                ax.set_ylim(top=current_height)
            
            # If no significant comparisons, note this on the plot
            if not significant_comparisons:
                ax.text(len(valid_layers)/2 - 0.5, max_height, "No significant\ndifferences", 
                    ha='center', va='bottom', fontsize=10, fontstyle='italic')
        
        # 3. Preferred position distribution by layer
        ax = axes[1, 0]
        
        for i, layer in enumerate(valid_layers):
            positions = layer_results[layer]['preferred_positions']
            ax.hist(positions, bins=10, alpha=0.6, label=layer)
        
        ax.set_xlabel('Preferred Position')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Preferred Positions by Layer')
        ax.legend()
        
        # 4. Rp vs Rn scatterplot by layer
        ax = axes[1, 1]
        
        for layer in valid_layers:
            rp = layer_results[layer]['Rp']
            rn = layer_results[layer]['Rn']
            ax.scatter(rn, rp, alpha=0.6, label=layer)
        
        # Add unity line
        lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lims, lims, 'k--', alpha=0.5)
        
        ax.set_xlabel('Response at Non-preferred Position (Rn)')
        ax.set_ylabel('Response at Preferred Position (Rp)')
        ax.set_title('Preferred vs Non-preferred Responses')
        ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Create statistical comparison between layers
        print("\nStatistical comparison between layers:")
        if len(valid_layers) >= 2:
            # Perform pairwise comparisons
            for i, layer1 in enumerate(valid_layers):
                for layer2 in valid_layers[i+1:]:
                    smi1 = layer_results[layer1]['SMI']
                    smi2 = layer_results[layer2]['SMI']
                    
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(smi1, smi2, equal_var=False)
                    
                    print(f"{layer1} vs {layer2}: t = {t_stat:.3f}, p = {p_value:.5f}")
                    
                    # Interpret results
                    if p_value < 0.05:
                        print(f"  Significant difference in SMI between {layer1} and {layer2}")
                        if np.mean(smi1) > np.mean(smi2):
                            print(f"  {layer1} shows stronger spatial modulation than {layer2}")
                        else:
                            print(f"  {layer2} shows stronger spatial modulation than {layer1}")
                    else:
                        print(f"  No significant difference in SMI between {layer1} and {layer2}")
        
            # ANOVA if more than 2 groups
            if len(valid_layers) > 2:
                anova_groups = [layer_results[layer]['SMI'] for layer in valid_layers]
                f_stat, p_anova = stats.f_oneway(*anova_groups)
                print(f"\nANOVA across all layers: F = {f_stat:.3f}, p = {p_anova:.5f}")
                if p_anova < 0.05:
                    print("  Significant differences exist between layers")
                else:
                    print("  No significant differences between layers")

    @staticmethod
    def plot_layer_distribution(med_coords, layer_cells, reliable_cells, FOV):
        """
        Visualize layer distribution of cells with equal subplot sizes.
        
        Parameters:
        -----------
        med_coords : numpy.ndarray
            Median coordinates of cells (cells x dimensions)
        layer_cells : dict
            Dictionary with indices of cells in each layer
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
        FOV : numpy.ndarray
            Field of view image (for visualization)
        """
        # Calculate cell density along y-axis
        density, bins = np.histogram(med_coords[:, 0], bins=50)
        bin_centers_density = (bins[:-1] + bins[1:]) / 2
        
        # Smooth density
        sigma = 1
        smooth_density = gaussian_filter1d(density, sigma)
        
        # Close all existing figures to prevent multiple outputs
        plt.close('all')
        
        # Create figure with wider size to accommodate the legend
        fig = plt.figure(figsize=(15, 8))
        
        # Use GridSpec to ensure equal subplot heights
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Plot cell density on the first subplot
        ax1.plot(density, bin_centers_density, 'k-', alpha=0.5, label='Cell density')
        ax1.plot(smooth_density, bin_centers_density, 'b-', linewidth=2, label='Smoothed')
        ax1.invert_yaxis()  # Invert y-axis for first plot as specified

        # Extract layer boundaries for visualization
        layers = []
        for layer_name, cell_indices in layer_cells.items():
            if len(cell_indices) > 0:
                upper = np.min(med_coords[cell_indices, 0])
                lower = np.max(med_coords[cell_indices, 0])
                layers.append((layer_name, upper, lower))
        
        # Add layer boundaries to plot
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, (layer, upper, lower) in enumerate(reversed(layers)):
            ax1.axvspan(0, np.max(density), 
                ymin=(upper - np.min(bin_centers_density)) / (np.max(bin_centers_density) - np.min(bin_centers_density)),
                ymax=(lower - np.min(bin_centers_density)) / (np.max(bin_centers_density) - np.min(bin_centers_density)),
                alpha=0.2, color=colors[i % len(colors)])
            
        for i, (layer, upper, lower) in enumerate(layers):
            ax1.text(np.max(density)*0.8, np.mean([upper, lower]), layer, 
                ha='center', va='center', fontsize=12, rotation=90)
        
        ax1.set_xlabel('Cell count')
        ax1.set_ylabel('Y-coordinate (μm)')
        ax1.set_title('Cell density along Y-axis')
        ax1.legend()
        
        # Plot cell distribution on the second subplot
        ax2.imshow(FOV, cmap='gray', vmin=np.percentile(FOV, 2), vmax=np.percentile(FOV, 90))
        # DO NOT invert y-axis for the second plot as specified
        # ax2.invert_yaxis()  - commented out as per your requirements
        
        ax2.scatter(med_coords[:, 1], med_coords[:, 0], s=5, alpha=0.2, c='lightgray')
        
        # For each layer, plot reliable and unreliable cells
        legend_elements = []
        for i, (layer, cells) in enumerate(layer_cells.items()):
            if len(cells) > 0:
                # Find reliable cells in this layer
                reliable_layer_cells = np.intersect1d(np.where(reliable_cells)[0], cells)
                unreliable_layer_cells = np.setdiff1d(cells, reliable_layer_cells)
                
                color = colors[i % len(colors)]
                # Plot reliable cells with higher opacity
                ax2.scatter(med_coords[reliable_layer_cells, 1], med_coords[reliable_layer_cells, 0], 
                        s=12, alpha=0.9, c=color)
                
                # Plot unreliable cells with lower opacity
                if len(unreliable_layer_cells) > 0:
                    ax2.scatter(med_coords[unreliable_layer_cells, 1], med_coords[unreliable_layer_cells, 0], 
                        s=8, alpha=0.3, c=color)
                
                # Add to legend
                legend_elements.append(plt.Line2D([0], [0], marker='', linestyle='', color='w',
                                            label=f'{layer} ({len(cells)} cells)'))
                
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=8, alpha=0.9,
                                            label=f'   Reliable: {len(reliable_layer_cells)} cells'))
                
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markersize=6, markerfacecolor=color, alpha=0.3,
                                            label=f'   Unreliable: {len(unreliable_layer_cells)} cells'))
        
        ax2.set_xlabel('X-coordinate (μm)')
        ax2.set_ylabel('Y-coordinate (μm)')
        ax2.set_title('Cell positions by layer')
        
        # Place legend outside to the right of the second subplot
        ax2.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                title="Layer Cell Counts", title_fontsize=12)
        
        # Make sure the aspect ratio is the same (important for same vertical size)
        ax1.set_box_aspect(1.0)
        ax2.set_box_aspect(1.0)
        
        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Add space on the right for the legend
        
        return fig

    @staticmethod
    def run_layer_SMI_analysis(normalized_spatial_activity, reliable_cells, 
                              layer_cells, bin_centers, segment_distance=55, 
                              exclude_boundary_cm=10, plot_distribution=True,
                              med_coords=None):
        """
        Run complete layer-specific SMI analysis.
        
        Parameters:
        -----------
        normalized_spatial_activity : numpy.ndarray
            Normalized activity matrix (n_cells x n_trials x n_spatial_bins)
        reliable_cells : numpy.ndarray
            Boolean array indicating reliable cells
        layer_cells : dict
            Dictionary with indices of cells in each layer
        bin_centers : numpy.ndarray
            Centers of spatial bins
        segment_distance : float
            Distance between repeated segments in the VR environment
        exclude_boundary_cm : float
            Distance from corridor boundaries to exclude for preferred positions
        plot_distribution : bool
            Whether to plot the layer distribution
        med_coords : numpy.ndarray or None
            Median coordinates of cells (cells x dimensions), needed for plotting if plot_distribution=True
            
        Returns:
        --------
        layer_results : dict
            Dictionary with SMI results for each layer
        """
        # Print information about input data
        print(f"Input data shape: {normalized_spatial_activity.shape}")
        # print(f"Number of reliable cells: {np.sum(reliable_cells)}")
        # print(f"Bin centers shape: {bin_centers.shape}")
        # print(f"Bin centers range: {np.min(bin_centers):.2f} to {np.max(bin_centers):.2f}")
        
        # Print information about layer cells
        for layer, cells in layer_cells.items():
            reliable_layer_cells = np.intersect1d(np.where(reliable_cells)[0], cells)
            print(f"{layer}: {len(cells)} cells, {len(reliable_layer_cells)} reliable")
        
        # Calculate SMI for all reliable cells
        print(f"\nCalculating SMI with segment_distance={segment_distance}, exclude_boundary_cm={exclude_boundary_cm}")
        all_smi_results = SpatialModulationIndexLayerSpecific.calculate_SMI(
            normalized_spatial_activity, 
            bin_centers, 
            reliable_cells=reliable_cells,
            segment_distance=segment_distance,
            exclude_boundary_cm=exclude_boundary_cm
        )
        
        if all_smi_results is None:
            print("SMI calculation failed!")
            return None
        
        # If no valid cells, try reducing the boundary exclusion
        if np.sum(all_smi_results['valid_cells']) == 0:
            print("\nNo valid cells found. Trying with reduced boundary exclusion...")
            exclude_boundary_cm = 5
            all_smi_results = SpatialModulationIndexLayerSpecific.calculate_SMI(
                normalized_spatial_activity, 
                bin_centers, 
                reliable_cells=reliable_cells,
                segment_distance=segment_distance,
                exclude_boundary_cm=exclude_boundary_cm
            )
            
            # If still no valid cells, try again with even smaller boundary
            if np.sum(all_smi_results['valid_cells']) == 0:
                print("\nStill no valid cells. Trying with minimal boundary exclusion...")
                exclude_boundary_cm = 1
                all_smi_results = SpatialModulationIndexLayerSpecific.calculate_SMI(
                    normalized_spatial_activity, 
                    bin_centers, 
                    segment_distance=segment_distance,
                    exclude_boundary_cm=exclude_boundary_cm
                )
        
        reliable_valid_cells = np.where(all_smi_results['reliable_valid_cells'])[0]
        # print(f"reliable_valid_cells: {reliable_valid_cells}")
        
        # Now calculate SMI for each layer
        layer_results = SpatialModulationIndexLayerSpecific.analyze_layer_specific_SMI(layer_cells, all_smi_results, reliable_cells)

        # Plot comparison if there are valid results
        valid_layers = [layer for layer, results in layer_results.items() 
                       if results is not None and len(results['SMI']) > 0]
        
        if len(valid_layers) > 0:
            print("\nGenerating comparison plots...")
            SpatialModulationIndexLayerSpecific.plot_layer_comparison(layer_results)
        else:
            print("\nNo valid layers for visualization")
            
        # Plot cell distribution if requested
        if plot_distribution and med_coords is not None:
            print("\nPlotting layer distribution...")
            SpatialModulationIndexLayerSpecific.plot_layer_distribution(med_coords, layer_cells, reliable_cells)
        
        return layer_results, reliable_valid_cells
        
    @staticmethod
    def visualize_top_cells(normalized_spatial_activity, layer_results, layer_name, bin_centers, top_n=5):
        """
        Visualize the top spatially modulated cells from a specific layer.
        
        Parameters:
        -----------
        normalized_spatial_activity : numpy.ndarray
            Normalized activity matrix (n_cells x n_trials x n_spatial_bins)
        layer_results : dict
            Dictionary with SMI results for each layer
        layer_name : str
            Name of the layer to visualize
        bin_centers : numpy.ndarray
            Centers of spatial bins
        top_n : int
            Number of top cells to visualize
        """
        if layer_results[layer_name] is None or len(layer_results[layer_name]['SMI']) == 0:
            print(f"No valid cells in {layer_name}")
            return
            
        # Get cells sorted by SMI
        layer_smi = layer_results[layer_name]['SMI']
        layer_cells_sorted = layer_results[layer_name]['reliable_valid_cells'][np.argsort(-layer_smi)]
        
        # Limit to top_n cells
        n_cells = min(top_n, len(layer_cells_sorted))
        top_cells = layer_cells_sorted[:n_cells]
        
        # Get SMI values for these cells
        top_smi_values = []
        for cell in top_cells:
            idx = np.where(layer_results[layer_name]['reliable_valid_cells'] == cell)[0][0]
            top_smi_values.append(layer_results[layer_name]['SMI'][idx])
            
        # Create plot
        plt.figure(figsize=(15, 3*n_cells))
        
        for i, (cell_idx, smi) in enumerate(zip(top_cells, top_smi_values)):
            # Average across trials
            cell_activity = np.mean(normalized_spatial_activity[cell_idx], axis=0)
            
            plt.subplot(n_cells, 1, i+1)
            plt.plot(bin_centers, cell_activity, 'b-', linewidth=2)
            
            # Add preferred and non-preferred positions if available
            idx = np.where(layer_results[layer_name]['reliable_valid_cells'] == cell_idx)[0][0]
            pref_pos = layer_results[layer_name]['preferred_positions'][idx]
            nonpref_pos = layer_results[layer_name]['non_preferred_positions'][idx]
            
            # Find closest bin indices
            pref_idx = np.argmin(np.abs(bin_centers - pref_pos))
            nonpref_idx = np.argmin(np.abs(bin_centers - nonpref_pos))
            
            # Highlight preferred and non-preferred positions
            plt.axvline(x=pref_pos, color='g', linestyle='--', label='Preferred')
            plt.axvline(x=nonpref_pos, color='r', linestyle='--', label='Non-preferred')
            
            # Plot points at the exact positions
            plt.scatter([pref_pos], [cell_activity[pref_idx]], color='g', s=100, zorder=10)
            plt.scatter([nonpref_pos], [cell_activity[nonpref_idx]], color='r', s=100, zorder=10)
            
            plt.title(f"Cell {cell_idx}, SMI = {smi:.3f}")
            plt.xlabel('Position (cm)')
            plt.ylabel('Normalized Activity')
            
            if i == 0:
                plt.legend()
                
    @staticmethod
    def plot_layer_smi_comparison_final(med_coords, layer_cells, smi_values, reliable_cells, 
                                    save_path=None, fig_title="Layer-specific Spatial Modulation"):
        """
        Final version with better spacing for colorbars and no overlaps
        """
        from matplotlib.colors import Normalize, LinearSegmentedColormap, BoundaryNorm
        import matplotlib.pyplot as plt
        import numpy as np
        import matplotlib.gridspec as gridspec
        
        # Define vibrant layer colors 
        layer_colors = {
            'L2/3': '#1E88E5',  # Bright blue
            'L4': '#FF9800',    # Bright orange
            'L5': '#4CAF50',    # Vibrant green
            'L6': '#E53935'     # Bright red
        }
        
        # Create figure with proportioned layout
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        
        # Create main axis for the cell plot
        ax_main = fig.add_subplot(gs[0])
        ax_main.set_facecolor('#f0f0f0')  # Light gray background for better contrast
        
        # Create axis for colorbars
        ax_colorbars = fig.add_subplot(gs[1])
        ax_colorbars.axis('off')
        
        # Get active layers (those with reliable cells)
        active_layers = []
        layer_stats = {}
        
    # Gather statistics on SMI distributions - ONLY for valid cells
        for layer_name, cell_indices in layer_cells.items():
            reliable_layer_cells = np.intersect1d(np.where(reliable_cells)[0], cell_indices)
            
            # Only include cells with valid SMI values (not NaN)
            valid_reliable_cells = [idx for idx in reliable_layer_cells if not np.isnan(smi_values[idx])]
            
            if len(valid_reliable_cells) > 0:
                active_layers.append(layer_name)
                
                layer_smi = [smi_values[idx] for idx in valid_reliable_cells]
                
                layer_stats[layer_name] = {
                    'cells': valid_reliable_cells,  # Store only valid cells
                    'smi_values': layer_smi,
                    'mean': np.mean(layer_smi)
                }
        
        # Find global min and max SMI for reliable cells
        all_smi = []
        for layer_name in active_layers:
            all_smi.extend(layer_stats[layer_name]['smi_values'])
        
        # Focus on the 0.15-0.4 range for visualizing differences
        focused_min = 0.15
        focused_max = 0.4
        
        # Create category boundaries focused on the relevant range
        n_categories = 5
        
        # Create custom boundaries with more detail in the 0.15-0.4 range
        boundaries = np.concatenate([
            [0],                                      # Start at 0
            np.linspace(0.15, 0.4, n_categories),    # Focused range with n_categories
            [1]                                       # End at 1
        ])
        
            # Keep track of which cells should be included in the visualization
        valid_cell_mask = ~np.isnan(smi_values)

        # Modify the cell plotting section of plot_layer_smi_comparison_final:
        for layer_name in active_layers:
            # Get all reliable cells in this layer
            reliable_layer_cells = np.intersect1d(np.where(reliable_cells)[0], layer_cells[layer_name])
            
            # Filter for only those with valid SMI values
            valid_reliable_cells = reliable_layer_cells[~np.isnan(smi_values[reliable_layer_cells])]
            
            if len(valid_reliable_cells) == 0:
                continue
                
            base_color = layer_colors[layer_name]
            r, g, b = plt.cm.colors.to_rgb(base_color)
            
            # Check if cells are missing by printing counts
            print(f"{layer_name}: {len(valid_reliable_cells)} valid cells out of {len(reliable_layer_cells)} reliable cells")
            
            # Group cells by SMI category for more distinct visualization
            for cat_idx in range(len(boundaries)-1):
                low_bound = boundaries[cat_idx]
                high_bound = boundaries[cat_idx+1]
                
                # Find cells in this category - make sure to only check valid cells
                cat_cells = []
                for cell_idx in valid_reliable_cells:
                    smi = np.clip(smi_values[cell_idx], -1, 1)
                    if low_bound <= smi < high_bound or (cat_idx == len(boundaries)-2 and smi == high_bound):
                        cat_cells.append(cell_idx)
                
                if len(cat_cells) == 0:
                    continue
                
                # Calculate intensity for this category
                if cat_idx < 1:  # Below focused range
                    intensity = 0.2
                elif cat_idx >= len(boundaries)-2:  # Above focused range
                    intensity = 1.0
                else:
                    # Linear mapping within focused range
                    rel_position = (cat_idx - 1) / (n_categories - 1)  # 0 to 1 within focused range
                    intensity = 0.3 + 0.7 * rel_position
                
                # Create color
                color = (
                    np.clip(r * intensity, 0, 1),
                    np.clip(g * intensity, 0, 1),
                    np.clip(b * intensity, 0, 1)
                )
                
                # Size based on category
                size_factor = 0.5 + 0.5 * intensity
                sizes = 20 + 20 * size_factor
                
                # Plot cells in this category
                ax_main.scatter(
                    med_coords[cat_cells, 1], 
                    med_coords[cat_cells, 0],
                    s=sizes,
                    alpha=1.0,
                    color=color,
                    marker='o',
                    edgecolor='black' if intensity > 0.6 else 'white',
                    linewidth=0.8
                )
        
        # Add grid to help with spatial reference
        ax_main.grid(True, linestyle='--', alpha=0.3)
        
        # Invert y-axis to start from 0 at top
        ax_main.invert_yaxis()
        
        # Set y-axis limits to 0-760
        ax_main.set_ylim(760, 0)  # Inverted because y-axis is inverted
        
        # Create legend for SMI categories
        legend_elements = []
        
        # Add one example for each layer
        for layer_name in active_layers:
            base_color = layer_colors[layer_name]
            
            # Add examples for different SMI ranges
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=base_color, markersize=8,
                                        markeredgecolor='black', markeredgewidth=0.5,
                                        label=f"{layer_name} ({len(valid_reliable_cells)} cells)"))
        
        # Add size/color examples for SMI levels
        example_layer = active_layers[0]  # Just use the first layer for examples
        base_color = layer_colors[example_layer]
        r, g, b = plt.cm.colors.to_rgb(base_color)
        
        # Add examples for SMI ranges
        example_levels = [
            ("SMI < 0.15", 0.2, 6, 'white'),
            ("SMI 0.15-0.25", 0.4, 8, 'white'),
            ("SMI 0.25-0.35", 0.7, 10, 'black'),
            ("SMI > 0.35", 1.0, 12, 'black')
        ]
        
        for label, intensity, size, edge_color in example_levels:
            color = (
                np.clip(r * intensity, 0, 1),
                np.clip(g * intensity, 0, 1),
                np.clip(b * intensity, 0, 1)
            )
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=size,
                                            markeredgecolor=edge_color, markeredgewidth=0.8,
                                            label=label))
        
        # Add legend to main plot
        ax_main.legend(handles=legend_elements, loc="upper right", 
                    title="Layer & SMI Range", fontsize=9)
        
        # Set labels and title for main plot
        ax_main.set_xlabel('X-coordinate (μm)', fontsize=12)
        ax_main.set_ylabel('Y-coordinate (μm)', fontsize=12)
        ax_main.set_title(fig_title, fontsize=14, fontweight='bold')
        
        # Add explanatory text
        ax_main.text(0.02, 0.02, 
                    "SMI visualization enhanced to highlight 0.15-0.4 range\nDarker color & larger marker = higher SMI", 
                    transform=ax_main.transAxes, va='bottom', ha='left', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'),
                    fontsize=10)
        
        # Create colorbars with improved spacing
        # Add title for colorbar section at the very top
        ax_colorbars.text(0.5, 0.98, "SMI Scale by Layer", 
                        transform=ax_colorbars.transAxes, fontsize=14, fontweight='bold',
                        ha='center', va='top')
        
        # Calculate improved spacing with more room between elements
        n_layers = len(active_layers)
        
        # Each layer section has:
        # 1. Layer title and cell count (text)
        # 2. The colorbar itself
        # 3. Space for x ticks
        
        # Define heights for each component
        title_height = 0.05  # Height for layer title and cell count
        bar_height = 0.07    # Height for the colorbar
        tick_height = 0.03   # Height for x-axis ticks
        spacing = 0.07       # Space between layer sections
        
        # Calculate total height needed per layer
        layer_section_height = title_height + bar_height + tick_height
        
        # Calculate total height needed and available space
        total_section_height = n_layers * layer_section_height
        available_height = 0.95  # From 0.98 (title position) to ~0.03 (bottom margin)
        
        # Adjust spacing if needed to fit
        if total_section_height + (n_layers - 1) * spacing > available_height:
            spacing = (available_height - total_section_height) / (n_layers - 1)
            spacing = max(0.02, spacing)  # Ensure minimum spacing
        
        # Create colorbars
        for i, layer_name in enumerate(active_layers):
            # Calculate positions for this layer section
            section_top = 0.93 - i * (layer_section_height + spacing)
            
            # Layer title and cell count position
            title_pos = section_top - title_height/2
            
            # Colorbar position
            bar_top = section_top - title_height
            
            # Get statistics
            stats = layer_stats[layer_name]
            n_cells = len(stats['cells'])
            mean_smi = stats['mean']
            
            # Add layer title and cell count
            ax_colorbars.text(0.02, title_pos, f"{layer_name}: {n_cells} cells", 
                            va='center', ha='left', fontweight='bold', fontsize=10)
            
            # Add mean value
            ax_colorbars.text(0.98, title_pos, f"Mean: {mean_smi:.2f}", 
                            va='center', ha='right', fontsize=10)
            
            # Create a custom colormap for this layer
            base_color = layer_colors[layer_name]
            r, g, b = plt.cm.colors.to_rgb(base_color)
            
            # Create colors for each segment
            cmap_colors = []
            
            # Add colors for each boundary section
            for j in range(len(boundaries)-1):
                # Calculate intensity based on position relative to focused range
                if j < 1:  # Below focused range
                    intensity = 0.2
                elif j >= len(boundaries)-2:  # Above focused range
                    intensity = 1.0
                else:
                    # Linear mapping within focused range
                    rel_position = (j - 1) / (n_categories - 1)
                    intensity = 0.3 + 0.7 * rel_position
                
                color = (
                    np.clip(r * intensity, 0, 1),
                    np.clip(g * intensity, 0, 1),
                    np.clip(b * intensity, 0, 1)
                )
                
                # Calculate the width of this segment
                width = boundaries[j+1] - boundaries[j]
                # Add more points for wider segments to maintain proportional width
                n_points = max(1, int(width * 100))
                cmap_colors.extend([color] * n_points)
            
            layer_cmap = LinearSegmentedColormap.from_list(f"{layer_name}_cmap", cmap_colors)
            
            # Create colorbar axes - position carefully to avoid overlap
            cax = ax_colorbars.inset_axes([0.1, bar_top - bar_height, 0.8, bar_height])
            
            # Create colorbar
            sm = plt.cm.ScalarMappable(cmap=layer_cmap, norm=Normalize(vmin=0, vmax=1))
            sm.set_array([])
            
            cbar = plt.colorbar(sm, cax=cax, orientation='horizontal')
            
            # Add focused tick marks highlighting the 0.15-0.4 range
            cbar.set_ticks([0, 0.15, 0.25, 0.35, 0.4, 1.0])
            cbar.set_ticklabels(['0', '0.15', '0.25', '0.35', '0.4', '1.0'])
            cbar.ax.tick_params(labelsize=8)
            
            # Add white arrow for mean on the colorbar
            cax.plot(mean_smi, 0.5, 'v', color='white', markersize=9, 
                    markeredgecolor='black', markeredgewidth=1.0)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
