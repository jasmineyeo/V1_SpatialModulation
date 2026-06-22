import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['legend.fontsize'] = 20
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 25
rcParams['xtick.labelsize'] = 20
rcParams['ytick.labelsize'] = 20
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

def double_gaussian(x, A, mu, sigma1, sigma2):
    """
    A function that consists of two Gaussians that meet at the peak.
    A: amplitude (peak height)
    mu: peak position
    sigma1: left-side standard deviation
    sigma2: right-side standard deviation
    """
    result = np.zeros_like(x, dtype=float)
    left_mask = x <= mu
    right_mask = x > mu
    
    # Left Gaussian
    if np.any(left_mask):
        result[left_mask] = A * np.exp(-0.5 * ((x[left_mask] - mu) / sigma1) ** 2)
    
    # Right Gaussian
    if np.any(right_mask):
        result[right_mask] = A * np.exp(-0.5 * ((x[right_mask] - mu) / sigma2) ** 2)
    
    return result

def fit_response_profile(bin_centers, profile, initial_peak_idx, window_size=5):
    """
    Fit a double Gaussian to the response profile around the initial peak.
    """
    # Define the window around the peak
    start_idx = max(0, initial_peak_idx - window_size)
    end_idx = min(len(bin_centers), initial_peak_idx + window_size + 1)
    
    x_data = bin_centers[start_idx:end_idx]
    y_data = profile[start_idx:end_idx]
    
    # Skip fitting if not enough data points
    if len(x_data) < 4:
        return None, None, bin_centers[initial_peak_idx], profile[initial_peak_idx], False
    
    # Initial parameter guesses
    initial_peak_pos = bin_centers[initial_peak_idx]
    initial_peak_val = profile[initial_peak_idx]
    
    # Initial guesses for parameters
    p0 = [
        initial_peak_val,  # A - amplitude
        initial_peak_pos,  # mu - peak position
        3.0,  # sigma1 - left width
        3.0   # sigma2 - right width
    ]
    
    try:
        # Perform the curve fitting with bounds
        bounds = (
            [0, np.min(x_data), 0.1, 0.1],  # lower bounds
            [np.inf, np.max(x_data), 15.0, 15.0]  # upper bounds
        )
        popt, _ = curve_fit(double_gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=2000)
        
        # Generate the fitted curve across all bin positions
        fit_curve = double_gaussian(bin_centers, *popt)
        
        # Extract peak position and value from fitted parameters
        peak_position = popt[1]  # mu parameter
        peak_response = popt[0]  # A parameter
        
        return popt, fit_curve, peak_position, peak_response, True
        
    except (RuntimeError, ValueError):
        # Fall back to the original peak if fitting fails
        return None, None, bin_centers[initial_peak_idx], profile[initial_peak_idx], False

def calculate_SMI_improved(spatial_activity, bin_centers, reliable_cells, segment_distance=55, 
                          exclude_boundary_cm=15, exclude_start_cm=None, exclude_end_cm=None, 
                          smoothing_sigma=1.0):
    """
    Calculate the Spatial Modulation Index (SMI) with proper boundary exclusion and 
    multiple non-preferred positions at ±1, ±2, ±3 times segment_distance.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (cells x trials x spatial_bins)
    bin_centers : numpy.ndarray
        Centers of spatial bins along the corridor
    reliable_cells : numpy.ndarray
        Boolean array indicating reliable cells
    segment_distance : float
        Base distance for landmark positions (e.g., 55cm)
    exclude_boundary_cm : float
        Distance from corridor ends to exclude (e.g., 15cm) - used if start/end not specified
    exclude_start_cm : float, optional
        Distance to exclude from the start/beginning of corridor (e.g., 10cm)
    exclude_end_cm : float, optional  
        Distance to exclude from the end of corridor (e.g., 5cm)
    smoothing_sigma : float
        Sigma for Gaussian smoothing of activity profiles
    
    Returns:
    --------
    results : dict
        Dictionary containing SMI values and analysis results
    """
    
    n_cells, n_trials, n_bins = spatial_activity.shape
    
    # Split trials into odd and even
    training_indices = np.arange(1, n_trials, 2)  # odd trials
    testing_indices = np.arange(0, n_trials, 2)  # even trials
    
    # # Split trials randomly into two halves
    # all_indices = np.arange(n_trials)
    # np.random.shuffle(all_indices)
    # training_indices = all_indices[:n_trials//2]
    # testing_indices = all_indices[n_trials//2:]
    
    # Calculate corridor boundaries
    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)
    corridor_length = max_pos - min_pos
    
    # Handle asymmetric boundary exclusion
    if exclude_start_cm is not None or exclude_end_cm is not None:
        # Use separate start and end exclusions
        start_exclude = exclude_start_cm if exclude_start_cm is not None else exclude_boundary_cm
        end_exclude = exclude_end_cm if exclude_end_cm is not None else exclude_boundary_cm
    else:
        # Use symmetric exclusion
        start_exclude = exclude_boundary_cm
        end_exclude = exclude_boundary_cm
    
    # Calculate allowed region with asymmetric boundaries
    min_allowed = min_pos + start_exclude
    max_allowed = max_pos - end_exclude
    
    print(f"  Corridor: {min_pos:.1f} to {max_pos:.1f} cm (length: {corridor_length:.1f} cm)")
    print(f"  Boundary exclusions: START {start_exclude:.1f} cm, END {end_exclude:.1f} cm")
    print(f"  Allowed region: {min_allowed:.1f} to {max_allowed:.1f} cm")
    print(f"  Allowed length: {max_allowed - min_allowed:.1f} cm")
    
    # Check if allowed region is valid
    if min_allowed >= max_allowed:
        raise ValueError(f"Invalid allowed region: start exclusion ({start_exclude}) + end exclusion ({end_exclude}) >= corridor length ({corridor_length})")
    
    # Find bin indices for allowed region
    allowed_mask = (bin_centers >= min_allowed) & (bin_centers <= max_allowed)
    allowed_indices = np.where(allowed_mask)[0]
    
    if len(allowed_indices) == 0:
        raise ValueError("No bins in allowed region - reduce boundary exclusions")
    
    print(f"  Allowed bins: {len(allowed_indices)} out of {n_bins}")
    
    # Compute response profiles for training and testing trials with Gaussian smoothing
    training_profiles = np.mean(spatial_activity[:, training_indices, :], axis=1)
    testing_profiles = np.mean(spatial_activity[:, testing_indices, :], axis=1)
    
    # Apply Gaussian smoothing
    if smoothing_sigma > 0:
        for cell in range(n_cells):
            training_profiles[cell] = gaussian_filter1d(training_profiles[cell], sigma=smoothing_sigma)
            testing_profiles[cell] = gaussian_filter1d(testing_profiles[cell], sigma=smoothing_sigma)
    
    # Initialize result arrays
    SMI_values = np.zeros(n_cells)
    preferred_positions = np.zeros(n_cells)
    non_preferred_positions = np.zeros(n_cells)
    Rp_values = np.zeros(n_cells)
    Rn_values = np.zeros(n_cells)
    valid_cells = np.zeros(n_cells, dtype=bool)
    
    # Store fitted curves for visualization
    preferred_fitted_curves = np.zeros((n_cells, n_bins))
    non_preferred_fitted_curves = np.zeros((n_cells, n_bins))
    fitting_success = np.zeros((n_cells, 2), dtype=bool)
    
    # Store information about non-preferred position selection
    non_preferred_candidates = np.zeros((n_cells, 6, 2))  # [cell, candidate, [position, response]]
    chosen_candidate_idx = np.zeros(n_cells, dtype=int)
    
    # Counters for rejection reasons
    no_peak_in_allowed = 0
    no_valid_nonpref = 0
    zero_response_sum = 0
    fitting_failed = 0
    boundary_violations = 0
    
    # Define the multipliers for non-preferred positions
    distance_multipliers = [-3, -2, -1, 1, 2, 3]

    for cell in tqdm(range(n_cells)):
        # Step 1: Find preferred position in training trials (ONLY in allowed region)
        training_profile_allowed = training_profiles[cell, allowed_indices]
        
        if np.max(training_profile_allowed) == 0:
            no_peak_in_allowed += 1
            continue
        
        # Find peak ONLY within allowed region
        local_peak_idx = np.argmax(training_profile_allowed)
        global_peak_idx = allowed_indices[local_peak_idx]
        preferred_position_training = bin_centers[global_peak_idx]
        
        # Sanity check - this should always pass now
        if not (min_allowed <= preferred_position_training <= max_allowed):
            print(f"ERROR: Cell {cell} training position {preferred_position_training:.1f} outside allowed region!")
            no_peak_in_allowed += 1
            continue
        
        # Step 2: Fit Gaussian to refine preferred position in training data
        popt_training, _, preferred_pos_fitted_training, _, fit_success_training = fit_response_profile(
            bin_centers, training_profiles[cell], global_peak_idx, window_size=5
        )
        
        # Ensure fitted position stays in allowed region
        if (fit_success_training and 
            min_allowed <= preferred_pos_fitted_training <= max_allowed):
            preferred_position_final = preferred_pos_fitted_training
        else:
            preferred_position_final = preferred_position_training
        
        # Step 3: Find corresponding peak in testing trials
        # Look for peak near the preferred position from training
        preferred_bin_idx = np.argmin(np.abs(bin_centers - preferred_position_final))
        
        # Search in small window around expected position
        search_start = max(0, preferred_bin_idx - 2)
        search_end = min(n_bins, preferred_bin_idx + 3)
        testing_window = testing_profiles[cell, search_start:search_end]
        local_max_idx = np.argmax(testing_window)
        testing_peak_idx = search_start + local_max_idx
        
        # Step 4: Fit Gaussian to preferred position in testing data
        popt_testing, fit_curve_pref, preferred_pos_testing, Rp_raw, fit_success_pref = fit_response_profile(
            bin_centers, testing_profiles[cell], testing_peak_idx, window_size=5
        )
        
        # Store fitting results
        fitting_success[cell, 0] = fit_success_pref
        if fit_success_pref:
            preferred_fitted_curves[cell] = fit_curve_pref
            Rp = Rp_raw
            # CRITICAL FIX: Enforce boundary constraints on testing fit
            if min_allowed <= preferred_pos_testing <= max_allowed:
                preferred_position_final = preferred_pos_testing
            else:
                pass
                # If testing fit goes out of bounds, keep the training position
                # print(f"Cell {cell}: Testing fit outside bounds ({preferred_pos_testing:.1f}), keeping training position ({preferred_position_final:.1f})")
        else:
            Rp = testing_profiles[cell, testing_peak_idx]
            fitting_failed += 1
        
        # FINAL SAFETY CHECK: Ensure final position is in bounds
        if not (min_allowed <= preferred_position_final <= max_allowed):
            print(f"WARNING: Cell {cell} final preferred position {preferred_position_final:.1f} outside allowed region [{min_allowed:.1f}, {max_allowed:.1f}]")
            boundary_violations += 1
            continue
        
        # Step 5: Find non-preferred positions at ±1, ±2, ±3 times segment_distance
        candidate_positions = []
        candidate_responses = []
        
        for i, multiplier in enumerate(distance_multipliers):
            candidate_pos = preferred_position_final + (multiplier * segment_distance)
            
            # Initialize candidate storage
            non_preferred_candidates[cell, i, 0] = candidate_pos
            non_preferred_candidates[cell, i, 1] = -1  # Default to invalid
            
            # STRICT BOUNDARY CHECK: Only consider candidates in allowed region
            if not (min_allowed <= candidate_pos <= max_allowed):
                continue
            
            # ADDITIONAL CHECK: Ensure candidate is also within corridor bounds
            if not (min_pos <= candidate_pos <= max_pos):
                continue
            
            # Find closest bin
            candidate_bin_idx = np.argmin(np.abs(bin_centers - candidate_pos))
            
            # Get response at this position (with small search window)
            search_start = max(0, candidate_bin_idx - 1)
            search_end = min(n_bins, candidate_bin_idx + 2)
            window_profile = testing_profiles[cell, search_start:search_end]
            local_peak_idx = np.argmax(window_profile)
            final_bin_idx = search_start + local_peak_idx
            
            # Try to fit Gaussian for more precise measurement
            popt_nonpref, fit_curve_nonpref, fitted_pos, fitted_resp, fit_success_nonpref = fit_response_profile(
                bin_centers, testing_profiles[cell], final_bin_idx, window_size=3
            )
            
            # BOUNDARY CHECK ON FITTED NON-PREFERRED POSITION
            if (fit_success_nonpref and 
                abs(fitted_pos - candidate_pos) <= 10 and  # Within 10cm of expected
                min_allowed <= fitted_pos <= max_allowed):  # WITHIN ALLOWED REGION
                final_pos = fitted_pos
                final_resp = fitted_resp
                fit_curve_final = fit_curve_nonpref
            else:
                # Use raw response if fitting fails, drifts too far, or goes out of bounds
                final_pos = bin_centers[final_bin_idx]
                final_resp = testing_profiles[cell, final_bin_idx]
                
                # DOUBLE-CHECK: Ensure raw position is also in bounds
                if not (min_allowed <= final_pos <= max_allowed):
                    continue  # Skip this candidate entirely
            
            # If we get here, the candidate is valid
            candidate_positions.append(final_pos)
            candidate_responses.append(final_resp)
            
            # Store candidate info
            non_preferred_candidates[cell, i, 0] = final_pos
            non_preferred_candidates[cell, i, 1] = final_resp
        
        # Step 6: Select best non-preferred position (smallest non-zero response)
        if len(candidate_responses) == 0:
            no_valid_nonpref += 1
            continue
        
        # Find minimum non-zero response
        valid_responses = [(i, resp) for i, resp in enumerate(candidate_responses) if resp > 0.01]
        
        if len(valid_responses) == 0:
            # All responses are essentially zero, take the minimum
            min_idx = np.argmin(candidate_responses)
            Rn = candidate_responses[min_idx]
            non_preferred_position_final = candidate_positions[min_idx]
            chosen_candidate_idx[cell] = min_idx
        else:
            # Take the smallest non-zero response
            min_idx, Rn = min(valid_responses, key=lambda x: x[1])
            non_preferred_position_final = candidate_positions[min_idx]
            chosen_candidate_idx[cell] = min_idx
        
        # FINAL BOUNDARY CHECK on chosen non-preferred position
        if not (min_allowed <= non_preferred_position_final <= max_allowed):
            print(f"WARNING: Cell {cell} chosen non-preferred position {non_preferred_position_final:.1f} outside bounds!")
            boundary_violations += 1
            continue
        
        # Try to fit Gaussian to chosen non-preferred position for visualization
        nonpref_bin_idx = np.argmin(np.abs(bin_centers - non_preferred_position_final))
        _, fit_curve_nonpref_final, _, _, fit_success_nonpref_final = fit_response_profile(
            bin_centers, testing_profiles[cell], nonpref_bin_idx, window_size=3
        )
        
        fitting_success[cell, 1] = fit_success_nonpref_final
        if fit_success_nonpref_final:
            non_preferred_fitted_curves[cell] = fit_curve_nonpref_final
        
        # Step 7: Calculate SMI
        if Rp + Rn > 0:
            SMI = (Rp - Rn) / (Rp + Rn)
        else:
            zero_response_sum += 1
            continue
        
        # Store results
        SMI_values[cell] = SMI
        preferred_positions[cell] = preferred_position_final
        non_preferred_positions[cell] = non_preferred_position_final
        Rp_values[cell] = Rp
        Rn_values[cell] = Rn
        valid_cells[cell] = True
    
    # Combine with reliability information
    if reliable_cells is not None:
        reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
    else:
        reliable_valid_cells = valid_cells
    
    # Print summary statistics
    print(f"\nSMI calculation summary:")
    print(f"  Total cells: {n_cells}")
    print(f"  Valid cells: {np.sum(valid_cells)} ({np.sum(valid_cells)/n_cells*100:.1f}%)")
    print(f"  Reliable & Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
    print(f"  Rejected - no peak in allowed region: {no_peak_in_allowed}")
    print(f"  Rejected - no valid non-preferred positions: {no_valid_nonpref}")
    print(f"  Rejected - zero response sum: {zero_response_sum}")
    print(f"  Rejected - boundary violations: {boundary_violations}")
    print(f"  Fitting failed (used raw values): {fitting_failed}")
    
    # Create results dictionary
    results = {
        'SMI': SMI_values,
        'preferred_positions': preferred_positions,
        'non_preferred_positions': non_preferred_positions,
        'Rp': Rp_values,
        'Rn': Rn_values,
        'training_profiles': training_profiles,
        'testing_profiles': testing_profiles,
        'preferred_fitted_curves': preferred_fitted_curves,
        'non_preferred_fitted_curves': non_preferred_fitted_curves,
        'fitting_success': fitting_success,
        'non_preferred_candidates': non_preferred_candidates,
        'chosen_candidate_idx': chosen_candidate_idx,
        'min_allowed': min_allowed,
        'max_allowed': max_allowed,
        'valid_cells': valid_cells,
        'reliable_valid_cells': reliable_valid_cells,
        'parameters': {
            'segment_distance': segment_distance,
            'exclude_boundary_cm': exclude_boundary_cm,
            'exclude_start_cm': start_exclude,
            'exclude_end_cm': end_exclude,
            'smoothing_sigma': smoothing_sigma,
            'distance_multipliers': distance_multipliers,
            'n_cells': n_cells,
            'n_training_trials': len(training_indices),
            'n_testing_trials': len(testing_indices),
            'n_bins': n_bins,
            'corridor_length': corridor_length,
            'min_pos': min_pos,
            'max_pos': max_pos
        }
    }
    
    return results

def plot_SMI_results_improved(results, bin_centers, reliable_cells=None, avg_cc=None, 
                             cohens_d=None, max_examples=10, data_filepath=None):
    """
    Plot the results of improved SMI calculation with candidate positions.
    """
    SMI_values = results['SMI']
    preferred_positions = results['preferred_positions']
    non_preferred_positions = results['non_preferred_positions']
    valid_cells = results['valid_cells']
    reliable_valid_cells = results['reliable_valid_cells']
    training_profiles = results['training_profiles']
    testing_profiles = results['testing_profiles']
    preferred_fitted_curves = results['preferred_fitted_curves']
    non_preferred_fitted_curves = results['non_preferred_fitted_curves']
    fitting_success = results['fitting_success']
    min_allowed = results['min_allowed']
    max_allowed = results['max_allowed']
    non_preferred_candidates = results['non_preferred_candidates']
    chosen_candidate_idx = results['chosen_candidate_idx']
    distance_multipliers = results['parameters']['distance_multipliers']
    segment_distance = results['parameters']['segment_distance']
    
    # Use reliable & valid cells
    if reliable_valid_cells is not None:
        plot_indices = np.where(reliable_valid_cells)[0]
        valid_SMI = SMI_values[reliable_valid_cells]
    else:
        plot_indices = np.where(valid_cells)[0]
        valid_SMI = SMI_values[valid_cells]
    
    n_valid = len(plot_indices)
    
    if n_valid == 0:
        print("No valid cells for plotting.")
        return []
    
    print(f"Found {n_valid} cells for analysis.")
    print(f"Mean SMI: {np.mean(valid_SMI):.3f}")
    print(f"Median SMI: {np.median(valid_SMI):.3f}")
    print(f"SMI range: {np.min(valid_SMI):.3f} to {np.max(valid_SMI):.3f}")
    
    # Create histogram
    fig_hist = plt.figure(figsize=(10, 6))
    plt.hist(valid_SMI, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--', alpha=0.7, label='SMI = 0')
    plt.axvline(np.median(valid_SMI), color='orange', linestyle='-', 
                label=f'Median = {np.median(valid_SMI):.3f}')
    plt.xlabel('Spatial Modulation Index (SMI)')
    plt.ylabel('Count')
    plt.title(f'SMI Distribution (n={n_valid} cells)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if data_filepath is not None:
        os.makedirs(os.path.join(data_filepath, "SMI_Figures"), exist_ok=True)
        hist_filename = os.path.join(data_filepath, "SMI_Figures", "SMI_histogram.png")
        fig_hist.savefig(hist_filename)
        print(f"Saved histogram to {hist_filename}")
    
    # Categorize cells
    strong = np.sum(valid_SMI > 0.5)
    moderate = np.sum((valid_SMI > 0.2) & (valid_SMI <= 0.5))
    weak = np.sum((valid_SMI > 0) & (valid_SMI <= 0.2))
    non_mod = np.sum(np.abs(valid_SMI) <= 0.05)
    negative = np.sum(valid_SMI < -0.05)
    
    print(f"Strong modulation (>0.5): {strong} ({strong/n_valid*100:.1f}%)")
    print(f"Moderate modulation (0.2-0.5): {moderate} ({moderate/n_valid*100:.1f}%)")
    print(f"Weak modulation (0-0.2): {weak} ({weak/n_valid*100:.1f}%)")
    print(f"Non-modulated (±0.05): {non_mod} ({non_mod/n_valid*100:.1f}%)")
    print(f"Negative modulation (<-0.05): {negative} ({negative/n_valid*100:.1f}%)")
    
    # Plot all valid cells, sorted by |SMI| descending
    sorted_indices = plot_indices[np.argsort(np.abs(valid_SMI))[::-1]]
    fig_examples = []

    for i in range(n_valid):
        cell_idx = sorted_indices[i]
        smi = SMI_values[cell_idx]
        pref_pos = preferred_positions[cell_idx]
        nonpref_pos = non_preferred_positions[cell_idx]
        
        fig = plt.figure(figsize=(14, 6))

        # Subplot 1: Raw response
        plt.subplot(1, 2, 1)
        plt.plot(bin_centers, training_profiles[cell_idx], 'b-', alpha=0.7,
                label='Training Trials', linewidth=1.5)
        plt.plot(bin_centers, testing_profiles[cell_idx], 'r-', alpha=0.7,
                label='Testing Trials', linewidth=1.5)

        # Mark preferred and chosen non-preferred positions
        plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.8,
                   label=f'Preferred ({pref_pos:.1f}cm)')
        plt.axvline(nonpref_pos, color='purple', linestyle='-', alpha=0.8,
                   label=f'Non-Preferred ({nonpref_pos:.1f}cm)')

        # Show all candidate non-preferred positions
        cell_candidates = non_preferred_candidates[cell_idx]
        chosen_idx = chosen_candidate_idx[cell_idx]

        for j, (pos, resp) in enumerate(cell_candidates):
            if resp > 0:  # Valid candidate
                if j == chosen_idx:
                    continue  # Skip the chosen one (already marked above)
                plt.axvline(pos, color='purple', linestyle=':', alpha=0.5,
                           label=f'Alt. ({distance_multipliers[j]:+d}×{segment_distance:.0f}cm)' if j < 2 else '_nolegend_')

        # Highlight first and last 10cm as excluded regions
        plt.axvspan(bin_centers[0], bin_centers[0] + 10, color='red', alpha=0.2, label='Excluded (10cm)')
        plt.axvspan(bin_centers[-1] - 10, bin_centers[-1], color='red', alpha=0.2, label='_nolegend_')

        plt.xlabel('Position (cm)')
        plt.ylabel('Activity')
        plt.title(f'Cell {cell_idx} - SMI: {smi:.3f}')
        plt.grid(True, alpha=0.3)

        # Subplot 2: Response comparison
        plt.subplot(1, 2, 2)
        positions = ['Preferred', 'Non-Preferred']
        values = [results['Rp'][cell_idx], results['Rn'][cell_idx]]
        colors = ['green', 'purple']

        bars = plt.bar(positions, values, color=colors, alpha=0.7)
        plt.title('Response Comparison')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom')

        # Add additional info as text
        info_text = ''
        if avg_cc is not None:
            info_text += f'Avg CC: {avg_cc[cell_idx]:.3f}'
        if cohens_d is not None:
            info_text += f'\nCohen\'s d: {cohens_d[cell_idx]:.3f}'

        if info_text:
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes,
                    verticalalignment='top', fontsize=16,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if data_filepath is not None:
            example_filename = os.path.join(data_filepath, "SMI_Figures", f"SMI_example_{cell_idx}.png")
            fig.savefig(example_filename)
            print(f"Saved example figure for cell {cell_idx} to {example_filename}")
        fig_examples.append(fig)
    
    return [fig_hist] + fig_examples

def analyze_spatial_modulation_improved(spatial_activity, bin_centers, reliable_cells=None, 
                                       avg_cc=None, cohens_d=None, segment_distance=55, 
                                       exclude_boundary_cm=15, exclude_start_cm=None, exclude_end_cm=None,
                                       smoothing_sigma=1.0, data_filepath = None):
    """
    Improved spatial modulation analysis with asymmetric boundary exclusion and 
    multiple landmark distances.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (cells x trials x spatial_bins)
    bin_centers : numpy.ndarray
        Centers of spatial bins along the corridor
    reliable_cells : numpy.ndarray, optional
        Boolean array indicating reliable cells
    avg_cc : numpy.ndarray, optional
        Average correlation coefficients for each cell
    cohens_d : numpy.ndarray, optional
        Cohen's d values for each cell
    segment_distance : float
        Base distance between landmarks (cm)
    exclude_boundary_cm : float
        Distance to exclude from corridor boundaries (cm) - used if start/end not specified
    exclude_start_cm : float, optional
        Distance to exclude from the start/beginning of corridor (e.g., 10cm)
    exclude_end_cm : float, optional  
        Distance to exclude from the end of corridor (e.g., 5cm)
    smoothing_sigma : float
        Gaussian smoothing sigma for activity profiles
    
    Returns:
    --------
    analysis_results : dict
        Dictionary containing SMI results and figures
    """
    print("=== IMPROVED SPATIAL MODULATION ANALYSIS ===")
    print(f"Parameters:")
    print(f"  Segment distance: {segment_distance} cm")
    
    # Display boundary exclusion info
    if exclude_start_cm is not None or exclude_end_cm is not None:
        start_exclude = exclude_start_cm if exclude_start_cm is not None else exclude_boundary_cm
        end_exclude = exclude_end_cm if exclude_end_cm is not None else exclude_boundary_cm
        print(f"  Boundary exclusion: START {start_exclude} cm, END {end_exclude} cm")
    else:
        print(f"  Boundary exclusion: {exclude_boundary_cm} cm (symmetric)")
    
    print(f"  Smoothing sigma: {smoothing_sigma}")
    print(f"  Non-preferred positions: ±1, ±2, ±3 × {segment_distance} cm")
    
    # Calculate SMI
    print("\nCalculating SMI...")
    smi_results = calculate_SMI_improved(
        spatial_activity=spatial_activity,
        bin_centers=bin_centers,
        reliable_cells=reliable_cells,
        segment_distance=segment_distance,
        exclude_boundary_cm=exclude_boundary_cm,
        exclude_start_cm=exclude_start_cm,
        exclude_end_cm=exclude_end_cm,
        smoothing_sigma=smoothing_sigma
    )
    
    # Create plots
    print("\nGenerating plots...")
    figures = plot_SMI_results_improved(
        results=smi_results,
        bin_centers=bin_centers,
        reliable_cells=reliable_cells,
        avg_cc=avg_cc,
        cohens_d=cohens_d,
        max_examples=6,
        data_filepath=data_filepath
    )

    
    print("\nAnalysis complete!")
    
    return {
        'smi_results': smi_results,
        'figures': figures
    }
       
def calculate_SMI_improved_debug(spatial_activity, bin_centers, reliable_cells, segment_distance=55, 
                        exclude_boundary_cm=15, smoothing_sigma=1.0):
    """
    Debug version of SMI calculation with extensive boundary checking.
    """
    n_cells, n_trials, n_bins = spatial_activity.shape

    # Split trials randomly into two halves
    all_indices = np.arange(n_trials)
    np.random.shuffle(all_indices)
    training_indices = all_indices[:n_trials//2]
    testing_indices = all_indices[n_trials//2:]

    # Calculate corridor boundaries
    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)
    corridor_length = max_pos - min_pos

    # Calculate allowed region (excluding boundaries)
    min_allowed = min_pos + exclude_boundary_cm
    max_allowed = max_pos - exclude_boundary_cm

    print(f"DEBUG: Corridor: {min_pos:.1f} to {max_pos:.1f} cm")
    print(f"DEBUG: Exclude boundary: {exclude_boundary_cm:.1f} cm")
    print(f"DEBUG: Allowed region: {min_allowed:.1f} to {max_allowed:.1f} cm")

    # Find bin indices for allowed region
    allowed_mask = (bin_centers >= min_allowed) & (bin_centers <= max_allowed)
    allowed_indices = np.where(allowed_mask)[0]

    print(f"DEBUG: Allowed bin indices: {allowed_indices[0]} to {allowed_indices[-1]}")
    print(f"DEBUG: Allowed bin positions: {bin_centers[allowed_indices[0]]:.1f} to {bin_centers[allowed_indices[-1]]:.1f} cm")

    if len(allowed_indices) == 0:
        raise ValueError("No bins in allowed region - reduce exclude_boundary_cm")

    # Compute response profiles for training and testing trials with Gaussian smoothing
    training_profiles = np.mean(spatial_activity[:, training_indices, :], axis=1)
    testing_profiles = np.mean(spatial_activity[:, testing_indices, :], axis=1)

    # Apply Gaussian smoothing
    if smoothing_sigma > 0:
        for cell in range(n_cells):
            training_profiles[cell] = gaussian_filter1d(training_profiles[cell], sigma=smoothing_sigma)
            testing_profiles[cell] = gaussian_filter1d(testing_profiles[cell], sigma=smoothing_sigma)

    # Initialize result arrays
    SMI_values = np.zeros(n_cells)
    preferred_positions = np.zeros(n_cells)
    non_preferred_positions = np.zeros(n_cells)
    Rp_values = np.zeros(n_cells)
    Rn_values = np.zeros(n_cells)
    valid_cells = np.zeros(n_cells, dtype=bool)

    # Store fitted curves for visualization
    preferred_fitted_curves = np.zeros((n_cells, n_bins))
    non_preferred_fitted_curves = np.zeros((n_cells, n_bins))
    fitting_success = np.zeros((n_cells, 2), dtype=bool)

    # Store information about non-preferred position selection
    non_preferred_candidates = np.zeros((n_cells, 6, 2))  # [cell, candidate, [position, response]]
    chosen_candidate_idx = np.zeros(n_cells, dtype=int)

    # Counters for rejection reasons
    no_peak_in_allowed = 0
    no_valid_nonpref = 0
    zero_response_sum = 0
    fitting_failed = 0

    # Define the multipliers for non-preferred positions
    distance_multipliers = [-3, -2, -1, 1, 2, 3]

    # DEBUG: Check first few cells in detail
    debug_cells = [122] if 122 < n_cells else [0, 1, 2]

    for cell in range(n_cells):
        is_debug_cell = cell in debug_cells
        
        # Step 1: Find preferred position in training trials (ONLY in allowed region)
        
        # CRITICAL: Only look at activity in allowed region
        training_profile_allowed = training_profiles[cell, allowed_indices]
        
        if is_debug_cell:
            print(f"\nDEBUG Cell {cell}:")
            print(f"  Full profile max at bin {np.argmax(training_profiles[cell])} = {bin_centers[np.argmax(training_profiles[cell])]:.1f}cm")
            print(f"  Allowed region profile shape: {training_profile_allowed.shape}")
            print(f"  Allowed region max: {np.max(training_profile_allowed):.4f}")
        
        if np.max(training_profile_allowed) == 0:
            no_peak_in_allowed += 1
            if is_debug_cell:
                print(f"  REJECTED: No activity in allowed region")
            continue
        
        # Find peak ONLY within allowed region
        local_peak_idx = np.argmax(training_profile_allowed)
        global_peak_idx = allowed_indices[local_peak_idx]
        preferred_position_training = bin_centers[global_peak_idx]
        
        if is_debug_cell:
            print(f"  Local peak idx in allowed region: {local_peak_idx}")
            print(f"  Global peak idx: {global_peak_idx}")
            print(f"  Preferred position from training: {preferred_position_training:.1f}cm")
        
        # CRITICAL BOUNDARY CHECK
        if not (min_allowed <= preferred_position_training <= max_allowed):
            if is_debug_cell:
                print(f"  ERROR: Preferred position {preferred_position_training:.1f} is outside allowed region!")
            no_peak_in_allowed += 1
            continue
        
        # Step 2: Fit Gaussian to refine preferred position in training data
        popt_training, _, preferred_pos_fitted_training, _, fit_success_training = fit_response_profile(
            bin_centers, training_profiles[cell], global_peak_idx, window_size=5
        )
        
        # CRITICAL: Ensure fitted position stays in allowed region
        if (fit_success_training and 
            min_allowed <= preferred_pos_fitted_training <= max_allowed):
            preferred_position_final = preferred_pos_fitted_training
            if is_debug_cell:
                print(f"  Using fitted position: {preferred_position_final:.1f}cm")
        else:
            preferred_position_final = preferred_position_training
            if is_debug_cell:
                print(f"  Using raw position (fitting failed or out of bounds): {preferred_position_final:.1f}cm")
        
        # FINAL BOUNDARY CHECK
        if not (min_allowed <= preferred_position_final <= max_allowed):
            if is_debug_cell:
                print(f"  FINAL ERROR: Position {preferred_position_final:.1f} outside bounds!")
            no_peak_in_allowed += 1
            continue
        
        # Step 3: Find corresponding peak in testing trials
        preferred_bin_idx = np.argmin(np.abs(bin_centers - preferred_position_final))
        
        # Search in small window around expected position
        search_start = max(0, preferred_bin_idx - 2)
        search_end = min(n_bins, preferred_bin_idx + 3)
        testing_window = testing_profiles[cell, search_start:search_end]
        local_max_idx = np.argmax(testing_window)
        testing_peak_idx = search_start + local_max_idx
        
        # Step 4: Fit Gaussian to preferred position in testing data
        popt_testing, fit_curve_pref, preferred_pos_testing, Rp_raw, fit_success_pref = fit_response_profile(
            bin_centers, testing_profiles[cell], testing_peak_idx, window_size=5
        )
        
        # Store fitting results
        fitting_success[cell, 0] = fit_success_pref
        if fit_success_pref:
            preferred_fitted_curves[cell] = fit_curve_pref
            Rp = Rp_raw
            # ENSURE testing position is also in bounds
            if min_allowed <= preferred_pos_testing <= max_allowed:
                preferred_position_final = preferred_pos_testing
                if is_debug_cell:
                    print(f"  Final fitted position (testing): {preferred_position_final:.1f}cm")
            else:
                if is_debug_cell:
                    print(f"  Testing fit outside bounds, keeping training position")
        else:
            Rp = testing_profiles[cell, testing_peak_idx]
            fitting_failed += 1
            if is_debug_cell:
                print(f"  Testing fit failed, using raw response: {Rp:.4f}")
        
        # ABSOLUTE FINAL CHECK
        if not (min_allowed <= preferred_position_final <= max_allowed):
            if is_debug_cell:
                print(f"  ABSOLUTE FINAL ERROR: Position {preferred_position_final:.1f} outside bounds!")
            no_peak_in_allowed += 1
            continue
        
        if is_debug_cell:
            print(f"  FINAL preferred position: {preferred_position_final:.1f}cm (Rp={Rp:.4f})")
        
        # Step 5: Find non-preferred positions at ±1, ±2, ±3 times segment_distance
        candidate_positions = []
        candidate_responses = []
        
        for i, multiplier in enumerate(distance_multipliers):
            candidate_pos = preferred_position_final + (multiplier * segment_distance)
            
            # Check if candidate position is in allowed region and within corridor
            if (min_allowed <= candidate_pos <= max_allowed and 
                min_pos <= candidate_pos <= max_pos):
                
                # Find closest bin
                candidate_bin_idx = np.argmin(np.abs(bin_centers - candidate_pos))
                
                # Get response at this position (with small search window)
                search_start = max(0, candidate_bin_idx - 1)
                search_end = min(n_bins, candidate_bin_idx + 2)
                window_profile = testing_profiles[cell, search_start:search_end]
                local_peak_idx = np.argmax(window_profile)
                final_bin_idx = search_start + local_peak_idx
                
                # Try to fit Gaussian for more precise measurement
                popt_nonpref, fit_curve_nonpref, fitted_pos, fitted_resp, fit_success_nonpref = fit_response_profile(
                    bin_centers, testing_profiles[cell], final_bin_idx, window_size=3
                )
                
                if (fit_success_nonpref and 
                    abs(fitted_pos - candidate_pos) <= 10):  # Within 10cm of expected
                    final_pos = fitted_pos
                    final_resp = fitted_resp
                    fit_curve_final = fit_curve_nonpref
                else:
                    # Use raw response if fitting fails or drifts too far
                    final_pos = bin_centers[final_bin_idx]
                    final_resp = testing_profiles[cell, final_bin_idx]
                    fit_curve_final = None
                
                candidate_positions.append(final_pos)
                candidate_responses.append(final_resp)
                
                # Store candidate info
                non_preferred_candidates[cell, i, 0] = final_pos
                non_preferred_candidates[cell, i, 1] = final_resp
                
                if is_debug_cell:
                    print(f"  Candidate {multiplier:+d}×{segment_distance}: {final_pos:.1f}cm, resp={final_resp:.4f}")
            else:
                # Invalid candidate
                non_preferred_candidates[cell, i, 0] = candidate_pos  # Store attempted position
                non_preferred_candidates[cell, i, 1] = -1  # Mark as invalid
                if is_debug_cell:
                    print(f"  Candidate {multiplier:+d}×{segment_distance}: {candidate_pos:.1f}cm (OUTSIDE BOUNDS)")
        
        # Step 6: Select best non-preferred position (smallest non-zero response)
        if len(candidate_responses) == 0:
            no_valid_nonpref += 1
            if is_debug_cell:
                print(f"  REJECTED: No valid non-preferred positions")
            continue
        
        # Find minimum non-zero response
        valid_responses = [(i, resp) for i, resp in enumerate(candidate_responses) if resp > 0.01]
        
        if len(valid_responses) == 0:
            # All responses are essentially zero, take the minimum
            min_idx = np.argmin(candidate_responses)
            Rn = candidate_responses[min_idx]
            non_preferred_position_final = candidate_positions[min_idx]
            chosen_candidate_idx[cell] = min_idx
        else:
            # Take the smallest non-zero response
            min_idx, Rn = min(valid_responses, key=lambda x: x[1])
            non_preferred_position_final = candidate_positions[min_idx]
            chosen_candidate_idx[cell] = min_idx
        
        if is_debug_cell:
            print(f"  Chosen non-preferred: {non_preferred_position_final:.1f}cm, Rn={Rn:.4f}")
        
        # Try to fit Gaussian to chosen non-preferred position for visualization
        nonpref_bin_idx = np.argmin(np.abs(bin_centers - non_preferred_position_final))
        _, fit_curve_nonpref_final, _, _, fit_success_nonpref_final = fit_response_profile(
            bin_centers, testing_profiles[cell], nonpref_bin_idx, window_size=3
        )
        
        fitting_success[cell, 1] = fit_success_nonpref_final
        if fit_success_nonpref_final:
            non_preferred_fitted_curves[cell] = fit_curve_nonpref_final
        
        # Step 7: Calculate SMI
        if Rp + Rn > 0:
            SMI = (Rp - Rn) / (Rp + Rn)
        else:
            zero_response_sum += 1
            if is_debug_cell:
                print(f"  REJECTED: Zero response sum")
            continue
        
        if is_debug_cell:
            print(f"  SMI = ({Rp:.4f} - {Rn:.4f}) / ({Rp:.4f} + {Rn:.4f}) = {SMI:.4f}")
        
        # Store results
        SMI_values[cell] = SMI
        preferred_positions[cell] = preferred_position_final
        non_preferred_positions[cell] = non_preferred_position_final
        Rp_values[cell] = Rp
        Rn_values[cell] = Rn
        valid_cells[cell] = True

    # Combine with reliability information
    if reliable_cells is not None:
        reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
    else:
        reliable_valid_cells = valid_cells

    # Print summary statistics
    print(f"\nSMI calculation summary:")
    print(f"  Total cells: {n_cells}")
    print(f"  Valid cells: {np.sum(valid_cells)} ({np.sum(valid_cells)/n_cells*100:.1f}%)")
    print(f"  Reliable & Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
    print(f"  Rejected - no peak in allowed region: {no_peak_in_allowed}")
    print(f"  Rejected - no valid non-preferred positions: {no_valid_nonpref}")
    print(f"  Rejected - zero response sum: {zero_response_sum}")
    print(f"  Fitting failed (used raw values): {fitting_failed}")

    # Create results dictionary
    results = {
        'SMI': SMI_values,
        'preferred_positions': preferred_positions,
        'non_preferred_positions': non_preferred_positions,
        'Rp': Rp_values,
        'Rn': Rn_values,
        'training_profiles': training_profiles,
        'testing_profiles': testing_profiles,
        'preferred_fitted_curves': preferred_fitted_curves,
        'non_preferred_fitted_curves': non_preferred_fitted_curves,
        'fitting_success': fitting_success,
        'non_preferred_candidates': non_preferred_candidates,
        'chosen_candidate_idx': chosen_candidate_idx,
        'min_allowed': min_allowed,
        'max_allowed': max_allowed,
        'valid_cells': valid_cells,
        'reliable_valid_cells': reliable_valid_cells,
        'parameters': {
            'segment_distance': segment_distance,
            'exclude_boundary_cm': exclude_boundary_cm,
            'smoothing_sigma': smoothing_sigma,
            'distance_multipliers': distance_multipliers,
            'n_cells': n_cells,
            'n_training_trials': len(training_indices),
            'n_testing_trials': len(testing_indices),
            'n_bins': n_bins,
            'corridor_length': corridor_length,
            'min_pos': min_pos,
            'max_pos': max_pos
        }
    }

    return results