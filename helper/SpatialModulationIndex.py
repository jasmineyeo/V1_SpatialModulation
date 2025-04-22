import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    
    Parameters:
    -----------
    bin_centers : array
        The position values
    profile : array
        The response profile
    initial_peak_idx : int
        Initial guess for the peak index
    window_size : int
        Half-width of the window around the peak to use for fitting
    
    Returns:
    --------
    fit_params : tuple
        (A, mu, sigma1, sigma2) - fitted parameters
    fit_curve : array
        The fitted curve across all bin_centers
    peak_position : float
        The position of the fitted peak
    peak_response : float
        The response value at the fitted peak
    success : bool
        Whether fitting succeeded
    """
    # Define the window around the peak
    start_idx = max(0, initial_peak_idx - window_size)
    end_idx = min(len(bin_centers), initial_peak_idx + window_size + 1)
    
    x_data = bin_centers[start_idx:end_idx]
    y_data = profile[start_idx:end_idx]
    
    # Skip fitting if not enough data points
    if len(x_data) < 4:
        print(f"Not enough data points for fitting at index {initial_peak_idx}")
        return None, None, bin_centers[initial_peak_idx], profile[initial_peak_idx], False
    
    # Initial parameter guesses
    initial_peak_pos = bin_centers[initial_peak_idx]
    initial_peak_val = profile[initial_peak_idx]
    
    # Initial guesses for parameters
    p0 = [
        initial_peak_val,  # A - amplitude
        initial_peak_pos,  # mu - peak position
        5.0,  # sigma1 - left width
        5.0   # sigma2 - right width
    ]
    
    try:
        # Perform the curve fitting with bounds
        bounds = (
            [0, np.min(x_data), 0.1, 0.1],  # lower bounds
            [np.inf, np.max(x_data), 20.0, 20.0]  # upper bounds
        )
        popt, _ = curve_fit(double_gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=2000)
        
        # Generate the fitted curve across all bin positions
        fit_curve = double_gaussian(bin_centers, *popt)
        
        # Extract peak position and value from fitted parameters
        peak_position = popt[1]  # mu parameter
        peak_response = popt[0]  # A parameter
        
        return popt, fit_curve, peak_position, peak_response, True
        
    except (RuntimeError, ValueError) as e:
        print(f"Fitting failed at index {initial_peak_idx}: {str(e)}")
        # Fall back to the original peak if fitting fails
        return None, None, bin_centers[initial_peak_idx], profile[initial_peak_idx], False

def calculate_SMI(spatial_activity, bin_centers, reliable_cells, segment_distance=55, exclude_boundary_cm=15):
    """
    Calculate the Spatial Modulation Index (SMI) using cross-validation approach with Gaussian fitting:
    - Odd trials to find preferred position
    - Even trials to measure responses
    - SMI = (Rp - Rn) / (Rp + Rn)
    
    Where Rp = response at preferred position, Rn = response at non-preferred position.
    """
    n_cells, n_trials, n_bins = spatial_activity.shape
    
    # Separate odd and even trials
    odd_indices = np.arange(0, n_trials, 2)
    even_indices = np.arange(1, n_trials, 2)
    
    # Calculate corridor boundaries
    min_pos = np.min(bin_centers)
    max_pos = np.max(bin_centers)
    corridor_length = np.max(bin_centers) - min_pos
    
    # Calculate boundary positions in the original coordinate system
    min_allowed = min_pos + exclude_boundary_cm
    max_allowed = max_pos - exclude_boundary_cm
    print(f"  Corridor length: {corridor_length:.2f} and valid position range: {min_allowed:.2f} to {max_allowed:.2f}")
        
    # Compute response profiles for odd and even trials
    odd_profiles = np.mean(spatial_activity[:, odd_indices, :], axis=1)
    even_profiles = np.mean(spatial_activity[:, even_indices, :], axis=1)
    
    # Initialize arrays to store results
    SMI_values = np.zeros(n_cells)
    preferred_positions = np.zeros(n_cells)
    non_preferred_positions = np.zeros(n_cells)
    Rp_values = np.zeros(n_cells)
    Rn_values = np.zeros(n_cells)
    valid_cells = np.zeros(n_cells, dtype=bool)
    
    # Store fitted curves for visualization
    preferred_fitted_curves = np.zeros((n_cells, n_bins))
    non_preferred_fitted_curves = np.zeros((n_cells, n_bins))
    fitting_success = np.zeros((n_cells, 2), dtype=bool)  # [0] for preferred, [1] for non-preferred

    # Count various rejection reasons
    outside_boundary_count = 0
    nonpref_outside_range_count = 0
    zero_response_count = 0
    fitting_failed_count = 0
    valid_count = 0
    
    for cell in range(n_cells):
        # Find the initial preferred position from odd trials
        preferred_idx_odd = np.argmax(odd_profiles[cell])
        preferred_position_odd = bin_centers[preferred_idx_odd]

        # Check if the preferred position is within allowed boundaries
        if preferred_position_odd < min_allowed or preferred_position_odd > max_allowed:
            outside_boundary_count += 1
            valid_cells[cell] = False
            continue

        # Define a window around the peak in odd trials for more precise localization
        # Fit a double Gaussian to the odd trials profile to find a smoother peak
        popt_odd, fit_curve_odd, preferred_position_fitted_odd, _, fit_success_odd = fit_response_profile(
            bin_centers, odd_profiles[cell], preferred_idx_odd, window_size=5
        )
        
        # Find the closest bin to the fitted preferred position
        preferred_idx_odd_fitted = np.argmin(np.abs(bin_centers - preferred_position_fitted_odd))
        
        # Now find the corresponding peak in even trials within a window of the refined odd peak
        start_idx_pref = max(0, preferred_idx_odd_fitted - 3)
        end_idx_pref = min(n_bins, preferred_idx_odd_fitted + 3)
        window_profile_pref = even_profiles[cell, start_idx_pref:end_idx_pref]
        window_max_idx_pref = np.argmax(window_profile_pref)
        preferred_idx_even_initial = start_idx_pref + window_max_idx_pref
        
        # Fit a double Gaussian to the even trials around this peak for the final preferred position
        popt_even_pref, fit_curve_even_pref, preferred_position_fitted_even, peak_response_even, fit_success_even = fit_response_profile(
            bin_centers, even_profiles[cell], preferred_idx_even_initial, window_size=5
        )
        
        # Store the fitting success status
        fitting_success[cell, 0] = fit_success_even
        
        # If fitting failed, we can either use the raw peak or skip this cell
        if not fit_success_even:
            fitting_failed_count += 1
            preferred_position_even = bin_centers[preferred_idx_even_initial]
            Rp = even_profiles[cell, preferred_idx_even_initial]
        else:
            # Use the fitted peak position and height
            preferred_position_even = preferred_position_fitted_even
            Rp = peak_response_even
            # Store the fitted curve for visualization
            preferred_fitted_curves[cell] = fit_curve_even_pref

        # Calculate the non-preferred position (both possibilities)
        corridor_midpoint = min_pos + corridor_length / 2
        if preferred_position_even < corridor_midpoint:
            # If in first segment, the non-preferred position is in second segment
            non_preferred_position_approx = preferred_position_even + segment_distance
        else:
            # If in second segment, the non-preferred position is in first segment
            non_preferred_position_approx = preferred_position_even - segment_distance

        # Check if non-preferred position is within corridor bounds
        if non_preferred_position_approx < min_pos or non_preferred_position_approx > max_pos:
            nonpref_outside_range_count += 1
            valid_cells[cell] = False
            continue

        # Find the closest bin to the approximate non-preferred position
        non_preferred_idx_approx = np.argmin(np.abs(bin_centers - non_preferred_position_approx))

        # Find the maximum response within ±3 indices of the non-preferred position
        start_idx_nonpref = max(0, non_preferred_idx_approx - 3)
        end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 3)
        window_profile_nonpref = even_profiles[cell, start_idx_nonpref:end_idx_nonpref]
        window_max_idx_nonpref = np.argmax(window_profile_nonpref)
        non_preferred_idx_even_initial = start_idx_nonpref + window_max_idx_nonpref
        
        # Fit a double Gaussian to the non-preferred position response
        popt_even_nonpref, fit_curve_even_nonpref, non_preferred_position_fitted, non_preferred_resp_fitted, fit_success_nonpref = fit_response_profile(
            bin_centers, even_profiles[cell], non_preferred_idx_even_initial, window_size=5
        )
        
        # Store the fitting success status
        fitting_success[cell, 1] = fit_success_nonpref
        
        # If fitting failed, fall back to the raw peak
        if not fit_success_nonpref:
            non_preferred_position_even = bin_centers[non_preferred_idx_even_initial]
            Rn = window_profile_nonpref[window_max_idx_nonpref]
        else:
            # Use the fitted non-preferred position and response
            non_preferred_position_even = non_preferred_position_fitted
            Rn = non_preferred_resp_fitted
            # Store the fitted curve for visualization
            non_preferred_fitted_curves[cell] = fit_curve_even_nonpref

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
    
    print(f"Number of total cells: {n_cells} and number of valid cells: {np.sum(valid_cells)}")      

    # find cells that are true for both reliable_cells and valid_cells
    if reliable_cells is not None:
        reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
    else:
        reliable_valid_cells = valid_cells
    
    # Print summary statistics
    print(f"\nSMI calculation summary:")
    print(f"  Total cells: {n_cells}")
    print(f"  Reliable&Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
    print(f"  Rejected - preferred position outside boundary: {outside_boundary_count} ({outside_boundary_count/n_cells*100:.1f}%)")
    print(f"  Rejected - non-preferred position outside corridor: {nonpref_outside_range_count} ({nonpref_outside_range_count/n_cells*100:.1f}%)")
    print(f"  Rejected - zero response sum: {zero_response_count} ({zero_response_count/n_cells*100:.1f}%)")
    print(f"  Fitting failed (but used raw peak instead): {fitting_failed_count} ({fitting_failed_count/n_cells*100:.1f}%)")
        
    # Create result dictionary
    results = {
        'SMI': SMI_values,
        'preferred_positions': preferred_positions,
        'non_preferred_positions': non_preferred_positions,
        'Rp': Rp_values,
        'Rn': Rn_values,
        'odd_profiles': odd_profiles,
        'even_profiles': even_profiles,
        'preferred_fitted_curves': preferred_fitted_curves,
        'non_preferred_fitted_curves': non_preferred_fitted_curves,
        'fitting_success': fitting_success,
        'min_allowed': min_allowed,
        'max_allowed': max_allowed,
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

def plot_SMI_results(results, bin_centers, reliable_cells=None, avg_cc=None, cohens_d=None, max_examples=6, exclude_boundary_cm=5):
    """
    Plot the results of SMI calculation with example cells, including fitted curves.
    """
    SMI_values = results['SMI']
    preferred_positions = results['preferred_positions']
    non_preferred_positions = results['non_preferred_positions']
    valid_cells = results['valid_cells']
    reliable_valid_cells = results['reliable_valid_cells'] if reliable_cells is not None else None
    odd_profiles = results['odd_profiles']
    even_profiles = results['even_profiles']
    preferred_fitted_curves = results['preferred_fitted_curves']
    non_preferred_fitted_curves = results['non_preferred_fitted_curves']
    fitting_success = results['fitting_success']
    min_allowed = results['min_allowed']
    max_allowed = results['max_allowed']
    
    # Use reliable & valid cells
    valid_indices = np.where(reliable_valid_cells)[0] if reliable_valid_cells is not None else np.where(valid_cells)[0]
    n_valid = len(valid_indices)
    
    if n_valid == 0:
        print(" No valid cells for SMI analysis.")
        return []
    
    print(f"    Found {n_valid} valid cells for SMI analysis.")
    
    # Summary statistics
    valid_SMI = SMI_values[reliable_valid_cells] if reliable_cells is not None else SMI_values[valid_indices]
    print(f"    Mean SMI: {np.mean(valid_SMI):.3f}")
    print(f"    Median SMI: {np.median(valid_SMI):.3f}")
    print(f"    SMI range: {np.min(valid_SMI):.3f} to {np.max(valid_SMI):.3f}")
    
    # Create histogram of SMI values
    fig_hist = plt.figure(figsize=(10, 6))
    plt.hist(valid_SMI, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(0, color='r', linestyle='--', alpha=0.7)
    plt.xlabel('Spatial Modulation Index (SMI)')
    plt.ylabel('Count')
    plt.title('Distribution of Spatial Modulation Index (SMI)')
    
    # Categorize cells
    strongly_modulated = np.sum(valid_SMI > 0.5)
    moderately_modulated = np.sum((valid_SMI > 0.2) & (valid_SMI <= 0.5))
    weakly_modulated = np.sum((valid_SMI > 0) & (valid_SMI <= 0.2))
    non_modulated = np.sum(np.abs(valid_SMI) <= 0.05)
    inverted_modulated = np.sum(valid_SMI < 0)
    
    print(f"    Strongly modulated (SMI > 0.5): {strongly_modulated} cells ({strongly_modulated/n_valid*100:.1f}%)")
    print(f"    Moderately modulated (0.2 < SMI ≤ 0.5): {moderately_modulated} cells ({moderately_modulated/n_valid*100:.1f}%)")
    print(f"    Weakly modulated (0 < SMI ≤ 0.2): {weakly_modulated} cells ({weakly_modulated/n_valid*100:.1f}%)")
    print(f"    Non-modulated (|SMI| ≤ 0.05): {non_modulated} cells ({non_modulated/n_valid*100:.1f}%)")
    print(f"    Inverted modulation (SMI < 0): {inverted_modulated} cells ({inverted_modulated/n_valid*100:.1f}%)")
    
    # Plot example cells
    # Sort by absolute SMI value to find most modulated cells
    sorted_indices = valid_indices[np.argsort(np.abs(valid_SMI))[::-1]]
    
    # Plot top examples
    n_examples = min(max_examples, n_valid)
    fig_examples = []
    
    for i in range(n_examples):
        cell_idx = sorted_indices[i]
        avg_cc_value = avg_cc[cell_idx] if isinstance(avg_cc, np.ndarray) else avg_cc
        cohens_d_value = cohens_d[cell_idx] if isinstance(cohens_d, np.ndarray) else cohens_d
        smi = SMI_values[cell_idx]
        pref_pos = preferred_positions[cell_idx]
        non_pref_pos = non_preferred_positions[cell_idx]
        
        # Check if we have fitted curves
        has_pref_fit = fitting_success[cell_idx, 0]
        has_nonpref_fit = fitting_success[cell_idx, 1]
        
        # Create figure
        fig = plt.figure(figsize=(15, 5))
        
        # Plot odd and even trial profiles with fitted curves
        plt.subplot(1, 3, 1)
        plt.plot(bin_centers, odd_profiles[cell_idx], 'b-', alpha=0.6, label='Odd Trials (Training)')
        plt.plot(bin_centers, even_profiles[cell_idx], 'r-', alpha=0.6, label='Even Trials (Testing)')
        
        # Add fitted curves if available
        if has_pref_fit:
            plt.plot(bin_centers, preferred_fitted_curves[cell_idx], 'g--', linewidth=2, 
                     label='Preferred Fit (Even)')
        
        if has_nonpref_fit:
            plt.plot(bin_centers, non_preferred_fitted_curves[cell_idx], 'm--', linewidth=2, 
                     label='Non-Preferred Fit (Even)')
        
        # Mark preferred and non-preferred positions
        plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7, 
                   label=f'Preferred Position ({pref_pos:.1f}cm)')
        plt.axvline(non_pref_pos, color='purple', linestyle='-', alpha=0.7, 
                   label=f'Non-Preferred Position ({non_pref_pos:.1f}cm)')
        
        # Highlight the excluded boundary regions
        plt.axvspan(0, min_allowed, color='red', alpha=0.1)
        plt.axvspan(max_allowed, bin_centers[-1], color='red', alpha=0.1)
        
        plt.title(f'Cell {cell_idx} - SMI: {smi:.3f}, Avg CC: {avg_cc_value:.3f}, Cohen\'s d: {cohens_d_value:.3f}')
        plt.xlabel('Position (cm)')
        plt.ylabel('Activity')
        plt.legend(loc='upper right', fontsize='small')
        
        # Plot zoomed view of preferred position
        plt.subplot(1, 3, 2)
        # Determine zoom window
        pref_idx = np.argmin(np.abs(bin_centers - pref_pos))
        zoom_start = max(0, pref_idx - 10)
        zoom_end = min(len(bin_centers), pref_idx + 10)
        
        plt.plot(bin_centers[zoom_start:zoom_end], odd_profiles[cell_idx, zoom_start:zoom_end], 'b-', alpha=0.6, label='Odd')
        plt.plot(bin_centers[zoom_start:zoom_end], even_profiles[cell_idx, zoom_start:zoom_end], 'r-', alpha=0.6, label='Even')
        
        # Add fitted curve if available
        if has_pref_fit:
            plt.plot(bin_centers[zoom_start:zoom_end], preferred_fitted_curves[cell_idx, zoom_start:zoom_end], 
                     'g--', linewidth=2, label='Preferred Fit')
        
        plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7)
        plt.title('Zoomed Preferred Position')
        plt.xlabel('Position (cm)')
        plt.ylabel('Activity')
        plt.legend(loc='upper right', fontsize='small')
        
        # Plot response comparison at preferred and non-preferred positions
        plt.subplot(1, 3, 3)
        positions = ['Preferred', 'Non-Preferred']
        values = [results['Rp'][cell_idx], results['Rn'][cell_idx]]
        
        bars = plt.bar(positions, values, color=['green', 'purple'], alpha=0.6)
        
        plt.title('Response Comparison (Even Trials)')
        plt.ylabel('Response')
        
        # Add text with SMI value
        plt.text(0.5, max(values) * 1.1, f'SMI = {smi:.3f}', 
                horizontalalignment='center', fontsize=12)
        
        plt.tight_layout()
        fig_examples.append(fig)
    
    # Return all figures
    return [fig_hist] + fig_examples

def analyze_spatial_modulation(spatial_activity, bin_centers, reliable_cells=None, avg_cc=None, cohens_d=None, segment_distance=55, exclude_boundary_cm=3):
    """
    Main function to perform spatial modulation analysis with Gaussian smoothing.
    
    Parameters:
    -----------
    spatial_activity : numpy.ndarray
        Activity matrix (cells x trials x spatial_bins)
    bin_centers : numpy.ndarray
        Centers of spatial bins along the corridor
    reliable_cells : numpy.ndarray, optional
        Boolean array indicating reliable cells
    segment_distance : float
        Distance between visually identical positions in cm
    """
    print("=== SPATIAL MODULATION ANALYSIS WITH GAUSSIAN FITTING ===")
    
    # 1. Calculate SMI with Gaussian fitting
    print("Calculating Spatial Modulation Index (SMI) with Gaussian fitting...")
    smi_results = calculate_SMI(
        spatial_activity, 
        bin_centers, 
        reliable_cells=reliable_cells,
        segment_distance=segment_distance,
        exclude_boundary_cm=exclude_boundary_cm
    )
    
    # 2. Plot SMI results with fitted curves
    print("")
    print("Plotting SMI results...")
    smi_figures = plot_SMI_results(
        smi_results, bin_centers, reliable_cells, avg_cc, cohens_d, max_examples=10, exclude_boundary_cm=exclude_boundary_cm
    )
    
    plt.show()
    
    print("Analysis complete!")
    
    return {
        'smi_results': smi_results,
        'smi_figures': smi_figures
    }
