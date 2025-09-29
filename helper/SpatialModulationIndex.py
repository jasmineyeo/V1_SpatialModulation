import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm


# def double_gaussian(x, A, mu, sigma1, sigma2):
#     """
#     A function that consists of two Gaussians that meet at the peak.
#     A: amplitude (peak height)
#     mu: peak position
#     sigma1: left-side standard deviation
#     sigma2: right-side standard deviation
#     """
#     result = np.zeros_like(x, dtype=float)
#     left_mask = x <= mu
#     right_mask = x > mu
    
#     # Left Gaussian
#     if np.any(left_mask):
#         result[left_mask] = A * np.exp(-0.5 * ((x[left_mask] - mu) / sigma1) ** 2)
    
#     # Right Gaussian
#     if np.any(right_mask):
#         result[right_mask] = A * np.exp(-0.5 * ((x[right_mask] - mu) / sigma2) ** 2)
    
#     return result

# def fit_response_profile(bin_centers, profile, initial_peak_idx, window_size=5):
#     """
#     Fit a double Gaussian to the response profile around the initial peak.
    
#     Parameters:
#     -----------
#     bin_centers : array
#         The position values
#     profile : array
#         The response profile
#     initial_peak_idx : int
#         Initial guess for the peak index
#     window_size : int
#         Half-width of the window around the peak to use for fitting
    
#     Returns:
#     --------
#     fit_params : tuple
#         (A, mu, sigma1, sigma2) - fitted parameters
#     fit_curve : array
#         The fitted curve across all bin_centers
#     peak_position : float
#         The position of the fitted peak
#     peak_response : float
#         The response value at the fitted peak
#     success : bool
#         Whether fitting succeeded
#     """
#     # Define the window around the peak
#     start_idx = max(0, initial_peak_idx - window_size)
#     end_idx = min(len(bin_centers), initial_peak_idx + window_size + 1)
    
#     x_data = bin_centers[start_idx:end_idx]
#     y_data = profile[start_idx:end_idx]
    
#     # Skip fitting if not enough data points
#     if len(x_data) < 4:
#         print(f"Not enough data points for fitting at index {initial_peak_idx}")
#         return None, None, bin_centers[initial_peak_idx], profile[initial_peak_idx], False
    
#     # Initial parameter guesses
#     initial_peak_pos = bin_centers[initial_peak_idx]
#     initial_peak_val = profile[initial_peak_idx]
    
#     # Initial guesses for parameters
#     p0 = [
#         initial_peak_val,  # A - amplitude
#         initial_peak_pos,  # mu - peak position
#         5.0,  # sigma1 - left width
#         5.0   # sigma2 - right width
#     ]
    
#     try:
#         # Perform the curve fitting with bounds
#         bounds = (
#             [0, np.min(x_data), 0.1, 0.1],  # lower bounds
#             [np.inf, np.max(x_data), 20.0, 20.0]  # upper bounds
#         )
#         popt, _ = curve_fit(double_gaussian, x_data, y_data, p0=p0, bounds=bounds, maxfev=2000)
        
#         # Generate the fitted curve across all bin positions
#         fit_curve = double_gaussian(bin_centers, *popt)
        
#         # Extract peak position and value from fitted parameters
#         peak_position = popt[1]  # mu parameter
#         peak_response = popt[0]  # A parameter
        
#         return popt, fit_curve, peak_position, peak_response, True
        
#     except (RuntimeError, ValueError) as e:
#         print(f"Fitting failed at index {initial_peak_idx}: {str(e)}")
#         # Fall back to the original peak if fitting fails
#         return None, None, bin_centers[initial_peak_idx], profile[initial_peak_idx], False

# # def calculate_SMI(spatial_activity, bin_centers, reliable_cells, segment_distance=55, exclude_boundary_cm=15):
# #     """
# #     Calculate the Spatial Modulation Index (SMI) using cross-validation approach with Gaussian fitting:
# #     - Odd trials to find preferred position
# #     - Even trials to measure responses
# #     - SMI = (Rp - Rn) / (Rp + Rn)
    
# #     Where Rp = response at preferred position, Rn = response at non-preferred position.
# #     """
# #     n_cells, n_trials, n_bins = spatial_activity.shape
    
# #     # # Separate odd and even trials
# #     # odd_indices = np.arange(0, n_trials, 2)
# #     # even_indices = np.arange(1, n_trials, 2)
    
# #     # RANDOM HALVES
# #     odd_indices = np.random.choice(n_trials, n_trials // 2, replace=False)
# #     even_indices = np.setdiff1d(np.arange(n_trials), odd_indices)

    
# #     # Calculate corridor boundaries
# #     min_pos = np.min(bin_centers)
# #     max_pos = np.max(bin_centers)
# #     corridor_length = np.max(bin_centers) - min_pos
    
# #     # Calculate boundary positions in the original coordinate system
# #     min_allowed = min_pos + exclude_boundary_cm
# #     max_allowed = max_pos - exclude_boundary_cm
# #     print(f"  Corridor length: {corridor_length:.2f} and valid position range: {min_allowed:.2f} to {max_allowed:.2f}")
        
# #     # Compute response profiles for odd and even trials
# #     odd_profiles = np.mean(spatial_activity[:, odd_indices, :], axis=1)
# #     even_profiles = np.mean(spatial_activity[:, even_indices, :], axis=1)
    
# #     # Initialize arrays to store results
# #     SMI_values = np.zeros(n_cells)
# #     preferred_positions = np.zeros(n_cells)
# #     non_preferred_positions = np.zeros(n_cells)
# #     Rp_values = np.zeros(n_cells)
# #     Rn_values = np.zeros(n_cells)
# #     valid_cells = np.zeros(n_cells, dtype=bool)
    
# #     # Store fitted curves for visualization
# #     preferred_fitted_curves = np.zeros((n_cells, n_bins))
# #     non_preferred_fitted_curves = np.zeros((n_cells, n_bins))
# #     fitting_success = np.zeros((n_cells, 2), dtype=bool)  # [0] for preferred, [1] for non-preferred

# #     # Count various rejection reasons
# #     outside_boundary_count = 0
# #     nonpref_outside_range_count = 0
# #     zero_response_count = 0
# #     fitting_failed_count = 0
# #     valid_count = 0
    
# #     for cell in range(n_cells):
# #         # Find the initial preferred position from odd trials
# #         preferred_idx_odd = np.argmax(odd_profiles[cell])
# #         preferred_position_odd = bin_centers[preferred_idx_odd]

# #         # Check if the preferred position is within allowed boundaries
# #         if preferred_position_odd < min_allowed or preferred_position_odd > max_allowed:
# #             outside_boundary_count += 1
# #             valid_cells[cell] = False
# #             continue

# #         # Define a window around the peak in odd trials for more precise localization
# #         # Fit a double Gaussian to the odd trials profile to find a smoother peak
# #         popt_odd, fit_curve_odd, preferred_position_fitted_odd, _, fit_success_odd = fit_response_profile(
# #             bin_centers, odd_profiles[cell], preferred_idx_odd, window_size=5
# #         )
        
# #         # Find the closest bin to the fitted preferred position
# #         preferred_idx_odd_fitted = np.argmin(np.abs(bin_centers - preferred_position_fitted_odd))
        
# #         # Now find the corresponding peak in even trials within a window of the refined odd peak
# #         start_idx_pref = max(0, preferred_idx_odd_fitted - 1)
# #         end_idx_pref = min(n_bins, preferred_idx_odd_fitted + 1)
# #         window_profile_pref = even_profiles[cell, start_idx_pref:end_idx_pref]
# #         window_max_idx_pref = np.argmax(window_profile_pref)
# #         preferred_idx_even_initial = start_idx_pref + window_max_idx_pref
        
# #         # Fit a double Gaussian to the even trials around this peak for the final preferred position
# #         popt_even_pref, fit_curve_even_pref, preferred_position_fitted_even, peak_response_even, fit_success_even = fit_response_profile(
# #             bin_centers, even_profiles[cell], preferred_idx_even_initial, window_size=5
# #         )
        
# #         # Store the fitting success status
# #         fitting_success[cell, 0] = fit_success_even
        
# #         # If fitting failed, we can either use the raw peak or skip this cell
# #         if not fit_success_even:
# #             fitting_failed_count += 1
# #             preferred_position_even = bin_centers[preferred_idx_even_initial]
# #             Rp = even_profiles[cell, preferred_idx_even_initial]
# #         else:
# #             # Use the fitted peak position and height
# #             preferred_position_even = preferred_position_fitted_even
# #             Rp = peak_response_even
# #             # Store the fitted curve for visualization
# #             preferred_fitted_curves[cell] = fit_curve_even_pref

# #         # Calculate the non-preferred position (both possibilities)
# #         corridor_midpoint = min_pos + corridor_length / 2
# #         if preferred_position_even < corridor_midpoint:
# #             # If in first segment, the non-preferred position is in second segment
# #             non_preferred_position_approx = preferred_position_even + segment_distance
# #         else:
# #             # If in second segment, the non-preferred position is in first segment
# #             non_preferred_position_approx = preferred_position_even - segment_distance

# #         # Check if non-preferred position is within corridor bounds
# #         if non_preferred_position_approx < min_pos or non_preferred_position_approx > max_pos:
# #             nonpref_outside_range_count += 1
# #             valid_cells[cell] = False
# #             continue

# #         # Find the closest bin to the approximate non-preferred position
# #         non_preferred_idx_approx = np.argmin(np.abs(bin_centers - non_preferred_position_approx))

# #         # Find the maximum response within ±3 indices of the non-preferred position
# #         start_idx_nonpref = max(0, non_preferred_idx_approx - 1)
# #         end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 1)
# #         # start_idx_nonpref = max(0, non_preferred_idx_approx - 3)
# #         # end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 3)
# #         window_profile_nonpref = even_profiles[cell, start_idx_nonpref:end_idx_nonpref]
# #         window_max_idx_nonpref = np.argmax(window_profile_nonpref)
# #         non_preferred_idx_even_initial = start_idx_nonpref + window_max_idx_nonpref
        
# #         # Fit a double Gaussian to the non-preferred position response
# #         popt_even_nonpref, fit_curve_even_nonpref, non_preferred_position_fitted, non_preferred_resp_fitted, fit_success_nonpref = fit_response_profile(
# #             bin_centers, even_profiles[cell], non_preferred_idx_even_initial, window_size=5
# #         )
        
# #         # Store the fitting success status
# #         fitting_success[cell, 1] = fit_success_nonpref
        
# #         # If fitting failed, fall back to the raw peak
# #         if not fit_success_nonpref:
# #             non_preferred_position_even = bin_centers[non_preferred_idx_even_initial]
# #             Rn = window_profile_nonpref[window_max_idx_nonpref]
# #         else:
# #             # Use the fitted non-preferred position and response
# #             non_preferred_position_even = non_preferred_position_fitted
# #             Rn = non_preferred_resp_fitted
# #             # Store the fitted curve for visualization
# #             non_preferred_fitted_curves[cell] = fit_curve_even_nonpref

# #         # Calculate SMI
# #         if Rp + Rn > 0:  # Avoid division by zero
# #             SMI = (Rp - Rn) / (Rp + Rn)
# #             valid_count += 1

# #             # if Rp - Rn > 0:
# #             #     SMI = (Rp - Rn) / (Rp + Rn)
# #             #     valid_count += 1
# #             # else:
# #             #     SMI = 0
# #             #     zero_response_count += 1
# #             #     valid_cells[cell] = False
# #             #     continue
# #         else:
# #             SMI = 0
# #             zero_response_count += 1
# #             valid_cells[cell] = False
# #             continue
        
# #         # Store results - use the adjusted positions from even trials
# #         SMI_values[cell] = SMI
# #         preferred_positions[cell] = preferred_position_even
# #         non_preferred_positions[cell] = non_preferred_position_even
# #         Rp_values[cell] = Rp
# #         Rn_values[cell] = Rn
# #         valid_cells[cell] = True
    
# #     print(f"Number of total cells: {n_cells} and number of valid cells: {np.sum(valid_cells)}")      

# #     # find cells that are true for both reliable_cells and valid_cells
# #     if reliable_cells is not None:
# #         reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
# #     else:
# #         reliable_valid_cells = valid_cells
    
# #     # Print summary statistics
# #     print(f"\nSMI calculation summary:")
# #     print(f"  Total cells: {n_cells}")
# #     print(f"  Reliable&Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
# #     print(f"  Rejected - preferred position outside boundary: {outside_boundary_count} ({outside_boundary_count/n_cells*100:.1f}%)")
# #     print(f"  Rejected - non-preferred position outside corridor: {nonpref_outside_range_count} ({nonpref_outside_range_count/n_cells*100:.1f}%)")
# #     print(f"  Rejected - zero response sum: {zero_response_count} ({zero_response_count/n_cells*100:.1f}%)")
# #     print(f"  Fitting failed (but used raw peak instead): {fitting_failed_count} ({fitting_failed_count/n_cells*100:.1f}%)")
        
# #     # Create result dictionary
# #     results = {
# #         'SMI': SMI_values,
# #         'preferred_positions': preferred_positions,
# #         'non_preferred_positions': non_preferred_positions,
# #         'Rp': Rp_values,
# #         'Rn': Rn_values,
# #         'odd_profiles': odd_profiles,
# #         'even_profiles': even_profiles,
# #         'preferred_fitted_curves': preferred_fitted_curves,
# #         'non_preferred_fitted_curves': non_preferred_fitted_curves,
# #         'fitting_success': fitting_success,
# #         'min_allowed': min_allowed,
# #         'max_allowed': max_allowed,
# #         'valid_cells': valid_cells,
# #         'reliable_valid_cells': reliable_valid_cells if reliable_cells is not None else None,
# #         'parameters': {
# #                 'segment_distance': segment_distance,
# #                 'exclude_boundary_cm': exclude_boundary_cm,
# #                 'n_cells': n_cells,
# #                 'n_trials': n_trials,
# #                 'n_bins': n_bins,
# #                 'corridor_length': corridor_length,
# #                 'min_pos': min_pos,
# #                 'max_pos': max_pos
# #             }
# #     }
    
# #     return results

# def calculate_SMI(spatial_activity, bin_centers, reliable_cells, segment_distance=55, exclude_boundary_cm=15):
#     """
#     Calculate the Spatial Modulation Index (SMI) using cross-validation approach with Gaussian fitting:
#     - Odd trials to find preferred position (constrained within boundaries)
#     - Even trials to measure responses
#     - SMI = (Rp - Rn) / (Rp + Rn)
    
#     Where Rp = response at preferred position, Rn = response at non-preferred position.
#     """
#     n_cells, n_trials, n_bins = spatial_activity.shape
    
#     # RANDOM HALVES
#     odd_indices = np.random.choice(n_trials, n_trials // 2, replace=False)
#     even_indices = np.setdiff1d(np.arange(n_trials), odd_indices)

    
#     # Calculate corridor boundaries
#     min_pos = np.min(bin_centers)
#     max_pos = np.max(bin_centers)
#     corridor_length = np.max(bin_centers) - min_pos
    
#     # Calculate boundary positions in the original coordinate system
#     min_allowed = min_pos + exclude_boundary_cm
#     max_allowed = max_pos - exclude_boundary_cm
#     print(f"  Corridor length: {corridor_length:.2f} and valid position range: {min_allowed:.2f} to {max_allowed:.2f}")
    
#     # Find bin indices corresponding to the allowed boundaries
#     min_allowed_idx = np.argmin(np.abs(bin_centers - min_allowed))
#     max_allowed_idx = np.argmin(np.abs(bin_centers - max_allowed))
    
#     # Ensure we have a valid range
#     if min_allowed_idx >= max_allowed_idx:
#         print("Warning: Invalid boundary range - no valid bins available")
#         min_allowed_idx = 0
#         max_allowed_idx = n_bins - 1
    
#     print(f"  Boundary bins: min_allowed_idx={min_allowed_idx} (pos={bin_centers[min_allowed_idx]:.2f}), max_allowed_idx={max_allowed_idx} (pos={bin_centers[max_allowed_idx]:.2f})")
        
#     print(f"  Valid bin range: {min_allowed_idx} to {max_allowed_idx} (bins {bin_centers[min_allowed_idx]:.2f} to {bin_centers[max_allowed_idx]:.2f})")
        
#     # Compute response profiles for odd and even trials
#     odd_profiles = np.mean(spatial_activity[:, odd_indices, :], axis=1)
#     even_profiles = np.mean(spatial_activity[:, even_indices, :], axis=1)
    
#     # Initialize arrays to store results
#     SMI_values = np.zeros(n_cells)
#     preferred_positions = np.zeros(n_cells)
#     non_preferred_positions = np.zeros(n_cells)
#     Rp_values = np.zeros(n_cells)
#     Rn_values = np.zeros(n_cells)
#     valid_cells = np.zeros(n_cells, dtype=bool)
    
#     # Store fitted curves for visualization
#     preferred_fitted_curves = np.zeros((n_cells, n_bins))
#     non_preferred_fitted_curves = np.zeros((n_cells, n_bins))
#     fitting_success = np.zeros((n_cells, 2), dtype=bool)  # [0] for preferred, [1] for non-preferred

#     # Count various rejection reasons
#     nonpref_outside_range_count = 0
#     zero_response_count = 0
#     fitting_failed_count = 0
#     valid_count = 0
#     boundary_constrained_count = 0  # New counter for cells where we constrained the peak
    
#     for cell in range(n_cells):
#         # Find the preferred position from odd trials, but constrain it to valid boundaries
#         # First, find the global maximum
#         global_preferred_idx = np.argmax(odd_profiles[cell])
        
#         # Debug: print initial findings for problematic cells
#         if cell < 5:  # Debug first few cells
#             print(f"  Cell {cell}: Global max at bin {global_preferred_idx} (pos={bin_centers[global_preferred_idx]:.2f})")
#             print(f"    Boundary range: bins {min_allowed_idx}-{max_allowed_idx} (pos {bin_centers[min_allowed_idx]:.2f}-{bin_centers[max_allowed_idx]:.2f})")
        
#         # Check if global maximum is within boundaries
#         if min_allowed_idx <= global_preferred_idx <= max_allowed_idx:
#             # Global maximum is within boundaries, use it
#             preferred_idx_odd = global_preferred_idx
#             if cell < 5:
#                 print(f"    Using global max (within boundaries)")
#         else:
#             # Global maximum is outside boundaries, find the maximum within the valid range
#             valid_profile = odd_profiles[cell, min_allowed_idx:max_allowed_idx+1]
#             local_max_idx = np.argmax(valid_profile)
#             preferred_idx_odd = min_allowed_idx + local_max_idx
#             boundary_constrained_count += 1
#             if cell < 5:
#                 print(f"    Constraining peak from bin {global_preferred_idx} to bin {preferred_idx_odd} (pos {bin_centers[preferred_idx_odd]:.2f})")
        
#         preferred_position_odd = bin_centers[preferred_idx_odd]

#         # Define a window around the peak in odd trials for more precise localization
#         # Fit a double Gaussian to the odd trials profile to find a smoother peak
#         popt_odd, fit_curve_odd, preferred_position_fitted_odd, _, fit_success_odd = fit_response_profile(
#             bin_centers, odd_profiles[cell], preferred_idx_odd, window_size=5
#         )
        
#         # CRITICAL: Ensure the fitted position is also within boundaries
#         if preferred_position_fitted_odd < min_allowed or preferred_position_fitted_odd > max_allowed:
#             # If fitting pushed us outside boundaries, use the raw constrained position
#             preferred_position_fitted_odd = preferred_position_odd
#             if cell < 5:
#                 print(f"    Fitting pushed outside boundaries, using raw position: {preferred_position_fitted_odd:.2f}")
        
#         # Find the closest bin to the fitted preferred position
#         preferred_idx_odd_fitted = np.argmin(np.abs(bin_centers - preferred_position_fitted_odd))
        
#         # Now find the corresponding peak in even trials within a window of the refined odd peak
#         start_idx_pref = max(0, preferred_idx_odd_fitted - 1)
#         end_idx_pref = min(n_bins, preferred_idx_odd_fitted + 1)
#         window_profile_pref = even_profiles[cell, start_idx_pref:end_idx_pref]
#         window_max_idx_pref = np.argmax(window_profile_pref)
#         preferred_idx_even_initial = start_idx_pref + window_max_idx_pref
        
#         # Fit a double Gaussian to the even trials around this peak for the final preferred position
#         popt_even_pref, fit_curve_even_pref, preferred_position_fitted_even, peak_response_even, fit_success_even = fit_response_profile(
#             bin_centers, even_profiles[cell], preferred_idx_even_initial, window_size=5
#         )
        
#         # Store the fitting success status
#         fitting_success[cell, 0] = fit_success_even
        
#         # If fitting failed, we can either use the raw peak or skip this cell
#         if not fit_success_even:
#             fitting_failed_count += 1
#             preferred_position_even = bin_centers[preferred_idx_even_initial]
#             Rp = even_profiles[cell, preferred_idx_even_initial]
#         else:
#             # Use the fitted peak position and height
#             preferred_position_even = preferred_position_fitted_even
#             Rp = peak_response_even
#             # Store the fitted curve for visualization
#             preferred_fitted_curves[cell] = fit_curve_even_pref

#         # Calculate the non-preferred position (both possibilities)
#         corridor_midpoint = min_pos + corridor_length / 2
#         if preferred_position_even < corridor_midpoint:
#             # If in first segment, the non-preferred position is in second segment
#             non_preferred_position_approx = preferred_position_even + segment_distance
#         else:
#             # If in second segment, the non-preferred position is in first segment
#             non_preferred_position_approx = preferred_position_even - segment_distance

#         # Check if non-preferred position is within corridor bounds
#         if non_preferred_position_approx < min_pos or non_preferred_position_approx > max_pos:
#             nonpref_outside_range_count += 1
#             valid_cells[cell] = False
#             continue

#         # Find the closest bin to the approximate non-preferred position
#         non_preferred_idx_approx = np.argmin(np.abs(bin_centers - non_preferred_position_approx))

#         # Find the maximum response within ±1 indices of the non-preferred position
#         start_idx_nonpref = max(0, non_preferred_idx_approx - 1)
#         end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 1)
#         window_profile_nonpref = even_profiles[cell, start_idx_nonpref:end_idx_nonpref]
#         window_max_idx_nonpref = np.argmax(window_profile_nonpref)
#         non_preferred_idx_even_initial = start_idx_nonpref + window_max_idx_nonpref
        
#         # Fit a double Gaussian to the non-preferred position response
#         popt_even_nonpref, fit_curve_even_nonpref, non_preferred_position_fitted, non_preferred_resp_fitted, fit_success_nonpref = fit_response_profile(
#             bin_centers, even_profiles[cell], non_preferred_idx_even_initial, window_size=5
#         )
        
#         # Store the fitting success status
#         fitting_success[cell, 1] = fit_success_nonpref
        
#         # If fitting failed, fall back to the raw peak
#         if not fit_success_nonpref:
#             non_preferred_position_even = bin_centers[non_preferred_idx_even_initial]
#             Rn = window_profile_nonpref[window_max_idx_nonpref]
#         else:
#             # Use the fitted non-preferred position and response
#             non_preferred_position_even = non_preferred_position_fitted
#             Rn = non_preferred_resp_fitted
#             # Store the fitted curve for visualization
#             non_preferred_fitted_curves[cell] = fit_curve_even_nonpref

#         # Calculate SMI
#         if Rp + Rn > 0:  # Avoid division by zero
            
#             SMI = (Rp - Rn) / (Rp + Rn)
#             valid_count += 1
#             # if Rp - Rn > 0:
#             #         SMI = (Rp - Rn) / (Rp + Rn)
#             #         valid_count += 1
#             # else:
#             #     SMI = 0
#             #     zero_response_count += 1
#             #     valid_cells[cell] = False
#             #     continue
                
#         else:
#             SMI = 0
#             zero_response_count += 1
#             valid_cells[cell] = False
#             continue
        
#         # Store results - use the adjusted positions from even trials
#         SMI_values[cell] = SMI
#         preferred_positions[cell] = preferred_position_even
#         non_preferred_positions[cell] = non_preferred_position_even
#         Rp_values[cell] = Rp
#         Rn_values[cell] = Rn
#         valid_cells[cell] = True
    
#     print(f"Number of total cells: {n_cells} and number of valid cells: {np.sum(valid_cells)}")      

#     # find cells that are true for both reliable_cells and valid_cells
#     if reliable_cells is not None:
#         reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
#     else:
#         reliable_valid_cells = valid_cells
    
#     # Print summary statistics
#     print(f"\nSMI calculation summary:")
#     print(f"  Total cells: {n_cells}")
#     print(f"  Reliable&Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
#     print(f"  Cells with boundary-constrained peaks: {boundary_constrained_count} ({boundary_constrained_count/n_cells*100:.1f}%)")
#     print(f"  Rejected - non-preferred position outside corridor: {nonpref_outside_range_count} ({nonpref_outside_range_count/n_cells*100:.1f}%)")
#     print(f"  Rejected - zero response sum: {zero_response_count} ({zero_response_count/n_cells*100:.1f}%)")
#     print(f"  Fitting failed (but used raw peak instead): {fitting_failed_count} ({fitting_failed_count/n_cells*100:.1f}%)")
        
#     # Create result dictionary
#     results = {
#         'SMI': SMI_values,
#         'preferred_positions': preferred_positions,
#         'non_preferred_positions': non_preferred_positions,
#         'Rp': Rp_values,
#         'Rn': Rn_values,
#         'odd_profiles': odd_profiles,
#         'even_profiles': even_profiles,
#         'preferred_fitted_curves': preferred_fitted_curves,
#         'non_preferred_fitted_curves': non_preferred_fitted_curves,
#         'fitting_success': fitting_success,
#         'min_allowed': min_allowed,
#         'max_allowed': max_allowed,
#         'valid_cells': valid_cells,
#         'reliable_valid_cells': reliable_valid_cells if reliable_cells is not None else None,
#         'boundary_constrained_count': boundary_constrained_count,  # New field
#         'parameters': {
#                 'segment_distance': segment_distance,
#                 'exclude_boundary_cm': exclude_boundary_cm,
#                 'n_cells': n_cells,
#                 'n_trials': n_trials,
#                 'n_bins': n_bins,
#                 'corridor_length': corridor_length,
#                 'min_pos': min_pos,
#                 'max_pos': max_pos
#             }
#     }
    
#     return results


# def calculate_SMI_BBBB(spatial_activity, bin_centers, reliable_cells, segment_distances=[20, 40], exclude_boundary_cm=0):
#     """
#     Calculate the Spatial Modulation Index (SMI) using cross-validation approach with Gaussian fitting:
#     - Odd trials to find preferred position
#     - Even trials to measure responses at preferred and non-preferred positions
#     - SMI = (Rp - Rn) / (Rp + Rn)
    
#     With three landmarks, creating multiple possible non-preferred positions.
#     Where Rp = response at preferred position, Rn = response at non-preferred position.
#     """
#     n_cells, n_trials, n_bins = spatial_activity.shape
    
#     # RANDOM HALVES instead of odd/even
#     odd_indices = np.random.choice(n_trials, n_trials // 2, replace=False)
#     even_indices = np.setdiff1d(np.arange(n_trials), odd_indices)
    
#     # Calculate corridor boundaries
#     min_pos = np.min(bin_centers)
#     max_pos = np.max(bin_centers)
#     corridor_length = max_pos - min_pos
    
#     # Calculate boundary positions in the original coordinate system
#     min_allowed = min_pos + exclude_boundary_cm
#     max_allowed = max_pos - exclude_boundary_cm
#     print(f"  Corridor length: {corridor_length:.2f} and valid position range: {min_allowed:.2f} to {max_allowed:.2f}")
        
#     # Compute response profiles for odd and even trials
#     odd_profiles = np.mean(spatial_activity[:, odd_indices, :], axis=1)
#     even_profiles = np.mean(spatial_activity[:, even_indices, :], axis=1)
    
#     # Initialize arrays to store results
#     SMI_values = np.zeros(n_cells)
#     preferred_positions = np.zeros(n_cells)
#     non_preferred_positions = np.zeros(n_cells)
#     Rp_values = np.zeros(n_cells)
#     Rn_values = np.zeros(n_cells)
#     valid_cells = np.zeros(n_cells, dtype=bool)
    
#     # Store fitted curves for visualization
#     preferred_fitted_curves = np.zeros((n_cells, n_bins))
#     non_preferred_fitted_curves = np.zeros((n_cells, n_bins))
#     fitting_success = np.zeros((n_cells, 2), dtype=bool)  # [0] for preferred, [1] for non-preferred
    
#     # Track which landmark segment the preferred position falls near
#     landmark_segment = np.zeros(n_cells, dtype=int)  # 0=first, 1=second, 2=third
    
#     # Store all potential non-preferred positions and responses for visualization
#     all_potential_nonpref = np.zeros((n_cells, len(segment_distances) * 2, 3))  # [cell, option, [position, response, valid]]

#     # Count various rejection reasons
#     outside_boundary_count = 0
#     nonpref_outside_range_count = 0
#     zero_response_count = 0
#     fitting_failed_count = 0
#     valid_count = 0
    
#     # Define the landmarks (assuming they're evenly spaced)
#     corridor_fourth = corridor_length / 4
#     landmark_positions = [
#         min_pos + corridor_fourth,         # First landmark
#         min_pos + 2 * corridor_fourth,     # Second landmark
#         min_pos + 3 * corridor_fourth,     # Third landmark
#         max_pos                           # Fourth landmark
#     ]
    
#     for cell in range(n_cells):
#         # Find the initial preferred position from odd trials
#         preferred_idx_odd = np.argmax(odd_profiles[cell])
#         preferred_position_odd = bin_centers[preferred_idx_odd]

#         # Check if the preferred position is within allowed boundaries
#         if preferred_position_odd < min_allowed or preferred_position_odd > max_allowed:
#             outside_boundary_count += 1
#             valid_cells[cell] = False
#             continue

#         # Define a window around the peak in odd trials for more precise localization
#         # Fit a double Gaussian to the odd trials profile to find a smoother peak
#         popt_odd, fit_curve_odd, preferred_position_fitted_odd, _, fit_success_odd = fit_response_profile(
#             bin_centers, odd_profiles[cell], preferred_idx_odd, window_size=5
#         )
        
#         # Find the closest bin to the fitted preferred position
#         preferred_idx_odd_fitted = np.argmin(np.abs(bin_centers - preferred_position_fitted_odd))
        
#         # Now find the corresponding peak in even trials within a window of the refined odd peak
#         start_idx_pref = max(0, preferred_idx_odd_fitted - 1)
#         end_idx_pref = min(n_bins, preferred_idx_odd_fitted + 1)
#         window_profile_pref = even_profiles[cell, start_idx_pref:end_idx_pref]
#         window_max_idx_pref = np.argmax(window_profile_pref)
#         preferred_idx_even_initial = start_idx_pref + window_max_idx_pref
        
#         # Fit a double Gaussian to the even trials around this peak for the final preferred position
#         popt_even_pref, fit_curve_even_pref, preferred_position_fitted_even, peak_response_even, fit_success_even = fit_response_profile(
#             bin_centers, even_profiles[cell], preferred_idx_even_initial, window_size=5
#         )
        
#         # Store the fitting success status
#         fitting_success[cell, 0] = fit_success_even
        
#         # If fitting failed, we can either use the raw peak or skip this cell
#         if not fit_success_even:
#             fitting_failed_count += 1
#             preferred_position_even = bin_centers[preferred_idx_even_initial]
#             Rp = even_profiles[cell, preferred_idx_even_initial]
#         else:
#             # Use the fitted peak position and height
#             preferred_position_even = preferred_position_fitted_even
#             Rp = peak_response_even
#             # Store the fitted curve for visualization
#             preferred_fitted_curves[cell] = fit_curve_even_pref

#         # Determine which landmark segment the preferred position is closest to
#         distances_to_landmarks = [abs(preferred_position_even - pos) for pos in landmark_positions]
#         closest_landmark = np.argmin(distances_to_landmarks)
#         landmark_segment[cell] = closest_landmark
        
#         # Calculate potential non-preferred positions based on which landmark segment we're in
#         non_preferred_positions_list = []
        
#         if closest_landmark == 0:  # Near first landmark
#             # Non-preferred can be +20 and +40 from preferred
#             for dist in segment_distances:
#                 non_preferred_positions_list.append(preferred_position_even + dist)
#         elif closest_landmark == 1:  # Near second landmark
#             # Non-preferred can be -20 and +20 from preferred
#             non_preferred_positions_list.append(preferred_position_even - segment_distances[0])
#             non_preferred_positions_list.append(preferred_position_even + segment_distances[0])
#         else:  # Near third landmark
#             # Non-preferred can be -20 and -40 from preferred
#             for dist in segment_distances:
#                 non_preferred_positions_list.append(preferred_position_even - dist)
        
#         # Check if any non-preferred position is outside corridor bounds
#         all_outside = True
#         for pos in non_preferred_positions_list:
#             if min_pos <= pos <= max_pos:
#                 all_outside = False
#                 break
                
#         if all_outside:
#             nonpref_outside_range_count += 1
#             valid_cells[cell] = False
#             continue
        
#         # Process each potential non-preferred position
#         non_preferred_responses = []
#         valid_non_preferred_positions = []
#         fitted_non_preferred_curves = []
#         non_preferred_fit_success = []
        
#         # Initialize array to store all potential non-preferred positions for this cell
#         cell_potential_nonpref = np.zeros((len(non_preferred_positions_list), 3))
        
#         for i, non_pref_pos in enumerate(non_preferred_positions_list):
#             # Set default values for invalid positions
#             cell_potential_nonpref[i, 0] = non_pref_pos  # Position
#             cell_potential_nonpref[i, 1] = -1  # Response (invalid)
#             cell_potential_nonpref[i, 2] = 0   # Valid flag (0=invalid)
            
#             # Skip if outside corridor bounds
#             if non_pref_pos < min_pos or non_pref_pos > max_pos:
#                 continue
                
#             # Find the closest bin to this non-preferred position
#             non_preferred_idx_approx = np.argmin(np.abs(bin_centers - non_pref_pos))
            
#             # Get exact response at this position (no window, just the exact position)
#             exact_response = even_profiles[cell, non_preferred_idx_approx]
            
#             # Fit a double Gaussian to the non-preferred position response
#             # Use a window to allow finding the precise location
#             start_idx_nonpref = max(0, non_preferred_idx_approx - 5)
#             end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 5)
            
#             if start_idx_nonpref >= end_idx_nonpref:
#                 continue  # Skip if window is invalid
                
#             window_profile_nonpref = even_profiles[cell, start_idx_nonpref:end_idx_nonpref]
#             window_peak_idx = np.argmax(window_profile_nonpref) + start_idx_nonpref
            
#             popt_even_nonpref, fit_curve, non_preferred_position_fitted, non_preferred_resp_fitted, fit_success_nonpref = fit_response_profile(
#                 bin_centers, even_profiles[cell], window_peak_idx, window_size=5
#             )
            
#             # If fitting succeeded, use the fitted values
#             if fit_success_nonpref:
#                 # Calculate the distance from the estimated position to the fitted position
#                 distance_from_target = abs(non_pref_pos - non_preferred_position_fitted)
                
#                 # Only use the fitted position if it's close to our target position
#                 if distance_from_target <= 5:  # Within 5 units of target
#                     non_preferred_pos_final = non_preferred_position_fitted
#                     non_preferred_resp = non_preferred_resp_fitted
#                     fitted_curve_final = fit_curve
#                     fit_success = True
#                 else:
#                     # If fitted position is too far, use the exact response at the target
#                     non_preferred_pos_final = non_pref_pos
#                     non_preferred_resp = exact_response
#                     fitted_curve_final = np.zeros_like(bin_centers)
#                     fit_success = False
#             else:
#                 # If fitting failed, use the exact response at the target
#                 non_preferred_pos_final = non_pref_pos
#                 non_preferred_resp = exact_response
#                 fitted_curve_final = np.zeros_like(bin_centers)
#                 fit_success = False
            
#             # Store this position, its response, and its fitted curve
#             valid_non_preferred_positions.append(non_preferred_pos_final)
#             non_preferred_responses.append(non_preferred_resp)
#             fitted_non_preferred_curves.append(fitted_curve_final)
#             non_preferred_fit_success.append(fit_success)
            
#             # Store for visualization
#             cell_potential_nonpref[i, 0] = non_preferred_pos_final
#             cell_potential_nonpref[i, 1] = non_preferred_resp
#             cell_potential_nonpref[i, 2] = 1  # Valid flag
        
#         # Store all potential non-preferred positions for this cell
#         if cell < all_potential_nonpref.shape[0]:
#             all_potential_nonpref[cell, :len(cell_potential_nonpref)] = cell_potential_nonpref
        
#         # If no valid non-preferred positions, skip this cell
#         if len(non_preferred_responses) == 0:
#             nonpref_outside_range_count += 1
#             valid_cells[cell] = False
#             continue
        
#         # Find the non-preferred position with the SMALLEST response (for maximal contrast)
#         min_resp_idx = np.argmin(non_preferred_responses)
#         Rn = non_preferred_responses[min_resp_idx]

#         # Check if the minimum response is zero, if so use the second smallest value
#         if Rn < 0.05:
#             # Create a copy of the array and replace the minimum value with infinity
#             temp_responses = non_preferred_responses.copy()
#             temp_responses[min_resp_idx] = np.inf
            
#             # Find the second smallest value
#             second_min_idx = np.argmin(temp_responses)
#             Rn = non_preferred_responses[second_min_idx]
#             non_preferred_position_even = valid_non_preferred_positions[second_min_idx]
#         else:
#             non_preferred_position_even = valid_non_preferred_positions[min_resp_idx]
            
#         # Store the fitted curve for the chosen non-preferred position
#         if non_preferred_fit_success[min_resp_idx]:
#             non_preferred_fitted_curves[cell] = fitted_non_preferred_curves[min_resp_idx]
#             fitting_success[cell, 1] = True
#         else:
#             fitting_success[cell, 1] = False
        
#         # Calculate SMI - only allow positive SMI values (Rp must be > Rn)
#         if Rp + Rn > 0:  # Avoid division by zero
#                 SMI = (Rp - Rn) / (Rp + Rn)
#                 valid_count += 1

#                 # if Rp - Rn > 0:
#                 #     SMI = (Rp - Rn) / (Rp + Rn)
#                 #     valid_count += 1
#                 # else:
#                 #     SMI = 0
#                 #     zero_response_count += 1
#                 #     valid_cells[cell] = False
#                 #     continue
                
#         else:
#             # Skip cells where preferred response is not greater than non-preferred
#             SMI = 0
#             zero_response_count += 1
#             valid_cells[cell] = False
#             continue
        
#         # Store results - use the adjusted positions from even trials
#         SMI_values[cell] = SMI
#         preferred_positions[cell] = preferred_position_even
#         non_preferred_positions[cell] = non_preferred_position_even
#         Rp_values[cell] = Rp
#         Rn_values[cell] = Rn
#         valid_cells[cell] = True
    
#     print(f"Number of total cells: {n_cells} and number of valid cells: {np.sum(valid_cells)}")      

#     # find cells that are true for both reliable_cells and valid_cells
#     if reliable_cells is not None:
#         reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
#     else:
#         reliable_valid_cells = valid_cells
    
#     # Print summary statistics
#     print(f"\nSMI calculation summary:")
#     print(f"  Total cells: {n_cells}")
#     print(f"  Reliable&Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
#     print(f"  Rejected - preferred position outside boundary: {outside_boundary_count} ({outside_boundary_count/n_cells*100:.1f}%)")
#     print(f"  Rejected - non-preferred position outside corridor: {nonpref_outside_range_count} ({nonpref_outside_range_count/n_cells*100:.1f}%)")
#     print(f"  Rejected - zero response sum or Rp ≤ Rn: {zero_response_count} ({zero_response_count/n_cells*100:.1f}%)")
#     print(f"  Fitting failed (but used raw peak instead): {fitting_failed_count} ({fitting_failed_count/n_cells*100:.1f}%)")
    
#     # Calculate segment statistics
#     if np.sum(valid_cells) > 0:
#         segment_counts = np.zeros(4, dtype=int)
#         for i in range(4):
#             segment_counts[i] = np.sum(landmark_segment[valid_cells] == i)
        
#         print(f"\nPreferred position by landmark segment:")
#         for i in range(4):
#             print(f"  Landmark {i+1}: {segment_counts[i]} cells ({segment_counts[i]/np.sum(valid_cells)*100:.1f}%)")
        
#     # Create result dictionary
#     results = {
#         'SMI': SMI_values,
#         'preferred_positions': preferred_positions,
#         'non_preferred_positions': non_preferred_positions,
#         'Rp': Rp_values,
#         'Rn': Rn_values,
#         'odd_profiles': odd_profiles,
#         'even_profiles': even_profiles,
#         'preferred_fitted_curves': preferred_fitted_curves,
#         'non_preferred_fitted_curves': non_preferred_fitted_curves,
#         'fitting_success': fitting_success,
#         'landmark_segment': landmark_segment,
#         'min_allowed': min_allowed,
#         'max_allowed': max_allowed,
#         'valid_cells': valid_cells,
#         'reliable_valid_cells': reliable_valid_cells if reliable_cells is not None else None,
#         'all_potential_nonpref': all_potential_nonpref,
#         'parameters': {
#             'segment_distances': segment_distances,
#             'exclude_boundary_cm': exclude_boundary_cm,
#             'n_cells': n_cells,
#             'n_trials': n_trials,
#             'n_bins': n_bins,
#             'corridor_length': corridor_length,
#             'min_pos': min_pos,
#             'max_pos': max_pos,
#             'landmark_positions': landmark_positions
#         }
#     }
    
#     return results

# def plot_SMI_results(results, bin_centers, reliable_cells=None, avg_cc=None, cohens_d=None, max_examples=6, exclude_boundary_cm=5):
#     """
#     Plot the results of SMI calculation with example cells, including fitted curves.
#     """
#     SMI_values = results['SMI']
#     preferred_positions = results['preferred_positions']
#     non_preferred_positions = results['non_preferred_positions']
#     valid_cells = results['valid_cells']
#     reliable_valid_cells = results['reliable_valid_cells'] if reliable_cells is not None else None
#     odd_profiles = results['odd_profiles']
#     even_profiles = results['even_profiles']
#     preferred_fitted_curves = results['preferred_fitted_curves']
#     non_preferred_fitted_curves = results['non_preferred_fitted_curves']
#     fitting_success = results['fitting_success']
#     min_allowed = results['min_allowed']
#     max_allowed = results['max_allowed']
    
#     # Use reliable & valid cells
#     valid_indices = np.where(reliable_valid_cells)[0] if reliable_valid_cells is not None else np.where(valid_cells)[0]
#     n_valid = len(valid_indices)
    
#     if n_valid == 0:
#         print(" No valid cells for SMI analysis.")
#         return []
    
#     print(f"    Found {n_valid} valid cells for SMI analysis.")
    
#     # Summary statistics
#     valid_SMI = SMI_values[reliable_valid_cells] if reliable_cells is not None else SMI_values[valid_indices]
#     print(f"    Mean SMI: {np.mean(valid_SMI):.3f}")
#     print(f"    Median SMI: {np.median(valid_SMI):.3f}")
#     print(f"    SMI range: {np.min(valid_SMI):.3f} to {np.max(valid_SMI):.3f}")
    
#     # Create histogram of SMI values
#     fig_hist = plt.figure(figsize=(10, 6))
#     plt.hist(valid_SMI, bins=20, color='skyblue', edgecolor='black')
#     plt.axvline(0, color='r', linestyle='--', alpha=0.7)
#     plt.xlabel('Spatial Modulation Index (SMI)')
#     plt.ylabel('Count')
#     plt.title('Distribution of Spatial Modulation Index (SMI)')
    
#     # Categorize cells
#     strongly_modulated = np.sum(valid_SMI > 0.5)
#     moderately_modulated = np.sum((valid_SMI > 0.2) & (valid_SMI <= 0.5))
#     weakly_modulated = np.sum((valid_SMI > 0) & (valid_SMI <= 0.2))
#     non_modulated = np.sum(np.abs(valid_SMI) <= 0.05)
#     inverted_modulated = np.sum(valid_SMI < 0)
    
#     print(f"    Strongly modulated (SMI > 0.5): {strongly_modulated} cells ({strongly_modulated/n_valid*100:.1f}%)")
#     print(f"    Moderately modulated (0.2 < SMI ≤ 0.5): {moderately_modulated} cells ({moderately_modulated/n_valid*100:.1f}%)")
#     print(f"    Weakly modulated (0 < SMI ≤ 0.2): {weakly_modulated} cells ({weakly_modulated/n_valid*100:.1f}%)")
#     print(f"    Non-modulated (|SMI| ≤ 0.05): {non_modulated} cells ({non_modulated/n_valid*100:.1f}%)")
#     print(f"    Inverted modulation (SMI < 0): {inverted_modulated} cells ({inverted_modulated/n_valid*100:.1f}%)")
    
#     # Plot example cells
#     # Sort by absolute SMI value to find most modulated cells
#     sorted_indices = valid_indices[np.argsort(np.abs(valid_SMI))[::-1]]
    
#     # Plot top examples
#     n_examples = min(max_examples, n_valid)
#     fig_examples = []
    
#     for i in range(n_examples):
#         cell_idx = sorted_indices[i]
#         avg_cc_value = avg_cc[cell_idx] if isinstance(avg_cc, np.ndarray) else avg_cc
#         cohens_d_value = cohens_d[cell_idx] if isinstance(cohens_d, np.ndarray) else cohens_d
#         smi = SMI_values[cell_idx]
#         pref_pos = preferred_positions[cell_idx]
#         non_pref_pos = non_preferred_positions[cell_idx]
        
#         # Check if we have fitted curves
#         has_pref_fit = fitting_success[cell_idx, 0]
#         has_nonpref_fit = fitting_success[cell_idx, 1]
        
#         # Create figure
#         fig = plt.figure(figsize=(15, 5))
        
#         # Plot odd and even trial profiles with fitted curves
#         plt.subplot(1, 3, 1)
#         plt.plot(bin_centers, odd_profiles[cell_idx], 'b-', alpha=0.6, label='Odd Trials (Training)')
#         plt.plot(bin_centers, even_profiles[cell_idx], 'r-', alpha=0.6, label='Even Trials (Testing)')
        
#         # Add fitted curves if available
#         if has_pref_fit:
#             plt.plot(bin_centers, preferred_fitted_curves[cell_idx], 'g--', linewidth=2, 
#                      label='Preferred Fit (Even)')
        
#         if has_nonpref_fit:
#             plt.plot(bin_centers, non_preferred_fitted_curves[cell_idx], 'm--', linewidth=2, 
#                      label='Non-Preferred Fit (Even)')
        
#         # Mark preferred and non-preferred positions
#         plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7, 
#                    label=f'Preferred Position ({pref_pos:.1f}cm)')
#         plt.axvline(non_pref_pos, color='purple', linestyle='-', alpha=0.7, 
#                    label=f'Non-Preferred Position ({non_pref_pos:.1f}cm)')
        
#         # Highlight the excluded boundary regions
#         plt.axvspan(0, min_allowed, color='red', alpha=0.1)
#         plt.axvspan(max_allowed, bin_centers[-1], color='red', alpha=0.1)
        
#         # plt.title(f'Cell {cell_idx} - SMI: {smi:.3f}, Avg CC: {avg_cc_value:.3f}, Cohen\'s d: {cohens_d_value:.3f}')
#         plt.xlabel('Position (cm)')
#         plt.ylabel('Activity')
#         plt.legend(loc='upper right', fontsize='small')
        
#         # Plot zoomed view of preferred position
#         plt.subplot(1, 3, 2)
#         # Determine zoom window
#         pref_idx = np.argmin(np.abs(bin_centers - pref_pos))
#         zoom_start = max(0, pref_idx - 10)
#         zoom_end = min(len(bin_centers), pref_idx + 10)
        
#         plt.plot(bin_centers[zoom_start:zoom_end], odd_profiles[cell_idx, zoom_start:zoom_end], 'b-', alpha=0.6, label='Odd')
#         plt.plot(bin_centers[zoom_start:zoom_end], even_profiles[cell_idx, zoom_start:zoom_end], 'r-', alpha=0.6, label='Even')
        
#         # Add fitted curve if available
#         if has_pref_fit:
#             plt.plot(bin_centers[zoom_start:zoom_end], preferred_fitted_curves[cell_idx, zoom_start:zoom_end], 
#                      'g--', linewidth=2, label='Preferred Fit')
        
#         plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7)
#         plt.title('Zoomed Preferred Position')
#         plt.xlabel('Position (cm)')
#         plt.ylabel('Activity')
#         plt.legend(loc='upper right', fontsize='small')
        
#         # Plot response comparison at preferred and non-preferred positions
#         plt.subplot(1, 3, 3)
#         positions = ['Preferred', 'Non-Preferred']
#         values = [results['Rp'][cell_idx], results['Rn'][cell_idx]]
        
#         bars = plt.bar(positions, values, color=['green', 'purple'], alpha=0.6)
        
#         plt.title('Response Comparison (Even Trials)')
#         plt.ylabel('Response')
        
#         # Add text with SMI value
#         plt.text(0.5, max(values) * 1.1, f'SMI = {smi:.3f}', 
#                 horizontalalignment='center', fontsize=12)
        
#         plt.tight_layout()
#         fig_examples.append(fig)
    
#     # Return all figures
#     return [fig_hist] + fig_examples

#     """
#     Plot the results of SMI calculation with example cells, including fitted curves.
#     For three-landmark scenario with multiple potential non-preferred positions.
#     """
#     SMI_values = results['SMI']
#     preferred_positions = results['preferred_positions']
#     non_preferred_positions = results['non_preferred_positions']
#     valid_cells = results['valid_cells']
#     reliable_valid_cells = results['reliable_valid_cells'] if reliable_cells is not None else None
#     odd_profiles = results['odd_profiles']
#     even_profiles = results['even_profiles']
#     preferred_fitted_curves = results['preferred_fitted_curves']
#     non_preferred_fitted_curves = results['non_preferred_fitted_curves']
#     fitting_success = results['fitting_success']
#     min_allowed = results['min_allowed']
#     max_allowed = results['max_allowed']
    
#     # Get landmark information if available
#     landmark_segment = results.get('landmark_segment', None)
#     landmark_positions = results.get('parameters', {}).get('landmark_positions', None)
#     segment_distances = results.get('parameters', {}).get('segment_distances', [20, 40])
    
#     # Use reliable & valid cells
#     valid_indices = np.where(reliable_valid_cells)[0] if reliable_valid_cells is not None else np.where(valid_cells)[0]
#     n_valid = len(valid_indices)
    
#     if n_valid == 0:
#         print(" No valid cells for SMI analysis.")
#         return []
    
#     print(f"    Found {n_valid} valid cells for SMI analysis.")
    
#     # Summary statistics
#     valid_SMI = SMI_values[reliable_valid_cells] if reliable_cells is not None else SMI_values[valid_indices]
#     print(f"    Mean SMI: {np.mean(valid_SMI):.3f}")
#     print(f"    Median SMI: {np.median(valid_SMI):.3f}")
#     print(f"    SMI range: {np.min(valid_SMI):.3f} to {np.max(valid_SMI):.3f}")
    
#     # Create histogram of SMI values
#     fig_hist = plt.figure(figsize=(10, 6))
#     plt.hist(valid_SMI, bins=20, color='skyblue', edgecolor='black')
#     plt.axvline(0, color='r', linestyle='--', alpha=0.7)
#     plt.xlabel('Spatial Modulation Index (SMI)')
#     plt.ylabel('Count')
#     plt.title('Distribution of Spatial Modulation Index (SMI)')
    
#     # Categorize cells
#     strongly_modulated = np.sum(valid_SMI > 0.5)
#     moderately_modulated = np.sum((valid_SMI > 0.2) & (valid_SMI <= 0.5))
#     weakly_modulated = np.sum((valid_SMI > 0) & (valid_SMI <= 0.2))
#     non_modulated = np.sum(np.abs(valid_SMI) <= 0.05)
#     inverted_modulated = np.sum(valid_SMI < 0)
    
#     print(f"    Strongly modulated (SMI > 0.5): {strongly_modulated} cells ({strongly_modulated/n_valid*100:.1f}%)")
#     print(f"    Moderately modulated (0.2 < SMI ≤ 0.5): {moderately_modulated} cells ({moderately_modulated/n_valid*100:.1f}%)")
#     print(f"    Weakly modulated (0 < SMI ≤ 0.2): {weakly_modulated} cells ({weakly_modulated/n_valid*100:.1f}%)")
#     print(f"    Non-modulated (|SMI| ≤ 0.05): {non_modulated} cells ({non_modulated/n_valid*100:.1f}%)")
#     print(f"    Inverted modulation (SMI < 0): {inverted_modulated} cells ({inverted_modulated/n_valid*100:.1f}%)")
    
#     # Plot example cells
#     # Sort by absolute SMI value to find most modulated cells
#     sorted_indices = valid_indices[np.argsort(np.abs(valid_SMI))[::-1]]
    
#     # Plot top examples
#     n_examples = min(max_examples, n_valid)
#     fig_examples = []
    
#     for i in range(n_examples):
#         cell_idx = sorted_indices[i]
#         avg_cc_value = avg_cc[cell_idx] if isinstance(avg_cc, np.ndarray) else avg_cc
#         cohens_d_value = cohens_d[cell_idx] if isinstance(cohens_d, np.ndarray) else cohens_d
#         smi = SMI_values[cell_idx]
#         pref_pos = preferred_positions[cell_idx]
#         non_pref_pos = non_preferred_positions[cell_idx]
        
#         # Check if we have fitted curves
#         has_pref_fit = fitting_success[cell_idx, 0]
#         has_nonpref_fit = fitting_success[cell_idx, 1]
        
#         # Create figure
#         fig = plt.figure(figsize=(15, 5))
        
#         # Plot odd and even trial profiles with fitted curves
#         plt.subplot(1, 3, 1)
#         plt.plot(bin_centers, odd_profiles[cell_idx], 'b-', alpha=0.6, label='Odd Trials (Training)')
#         plt.plot(bin_centers, even_profiles[cell_idx], 'r-', alpha=0.6, label='Even Trials (Testing)')
        
#         # Add fitted curves if available
#         if has_pref_fit:
#             plt.plot(bin_centers, preferred_fitted_curves[cell_idx], 'g--', linewidth=2, 
#                      label='Preferred Fit (Even)')
        
#         if has_nonpref_fit:
#             plt.plot(bin_centers, non_preferred_fitted_curves[cell_idx], 'm--', linewidth=2, 
#                      label='Non-Preferred Fit (Even)')
        
#         # Mark preferred and non-preferred positions
#         plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7, 
#                    label=f'Preferred Position ({pref_pos:.1f}cm)')
#         plt.axvline(non_pref_pos, color='purple', linestyle='-', alpha=0.7, 
#                    label=f'Non-Preferred Position ({non_pref_pos:.1f}cm)')
        
#         # Highlight the excluded boundary regions
#         plt.axvspan(0, min_allowed, color='red', alpha=0.1)
#         plt.axvspan(max_allowed, bin_centers[-1], color='red', alpha=0.1)
        
#         # If we have landmark information, mark the three landmarks
#         if landmark_positions is not None:
#             for j, lpos in enumerate(landmark_positions):
#                 plt.axvline(lpos, color='gray', linestyle='--', alpha=0.5,
#                           label=f'Landmark {j+1}' if j == 0 else f'_Landmark {j+1}')
        
#         # Show alternative non-preferred positions based on the landmark segment
#         if landmark_segment is not None:
#             segment = landmark_segment[cell_idx]
#             # Calculate alternative non-preferred positions based on segment
#             alt_positions = []
            
#             if segment == 0:  # First landmark
#                 for dist in segment_distances:
#                     alt_positions.append(pref_pos + dist)
#             elif segment == 1:  # Second landmark
#                 alt_positions.append(pref_pos - segment_distances[0])
#                 alt_positions.append(pref_pos + segment_distances[0])
#             else:  # Third landmark
#                 for dist in segment_distances:
#                     alt_positions.append(pref_pos - dist)
            
#             # Mark alternative positions that differ from the chosen one
#             for alt_pos in alt_positions:
#                 # Skip if it's the chosen position or outside corridor bounds
#                 if (abs(alt_pos - non_pref_pos) < 1 or 
#                     alt_pos < np.min(bin_centers) or 
#                     alt_pos > np.max(bin_centers)):
#                     continue
                    
#                 plt.axvline(alt_pos, color='purple', linestyle=':', alpha=0.5,
#                           label=f'Alt. Non-Pref ({alt_pos:.1f}cm)')
        
#         title_text = f'Cell {cell_idx} - SMI: {smi:.3f}'
#         if avg_cc_value is not None:
#             title_text += f', Avg CC: {avg_cc_value:.3f}'
#         if cohens_d_value is not None:
#             title_text += f', Cohen\'s d: {cohens_d_value:.3f}'
        
#         plt.title(title_text)
#         plt.xlabel('Position (cm)')
#         plt.ylabel('Activity')
#         plt.legend(loc='upper right', fontsize='small')
        
#         # Plot zoomed view of preferred position
#         plt.subplot(1, 3, 2)
#         # Determine zoom window
#         pref_idx = np.argmin(np.abs(bin_centers - pref_pos))
#         zoom_start = max(0, pref_idx - 10)
#         zoom_end = min(len(bin_centers), pref_idx + 10)
        
#         plt.plot(bin_centers[zoom_start:zoom_end], odd_profiles[cell_idx, zoom_start:zoom_end], 'b-', alpha=0.6, label='Odd')
#         plt.plot(bin_centers[zoom_start:zoom_end], even_profiles[cell_idx, zoom_start:zoom_end], 'r-', alpha=0.6, label='Even')
        
#         # Add fitted curve if available
#         if has_pref_fit:
#             plt.plot(bin_centers[zoom_start:zoom_end], preferred_fitted_curves[cell_idx, zoom_start:zoom_end], 
#                      'g--', linewidth=2, label='Preferred Fit')
        
#         plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7)
#         plt.title('Zoomed Preferred Position')
#         plt.xlabel('Position (cm)')
#         plt.ylabel('Activity')
#         plt.legend(loc='upper right', fontsize='small')
        
#         # Plot response comparison at preferred and non-preferred positions
#         plt.subplot(1, 3, 3)
#         positions = ['Preferred', 'Non-Preferred']
#         values = [results['Rp'][cell_idx], results['Rn'][cell_idx]]
        
#         bars = plt.bar(positions, values, color=['green', 'purple'], alpha=0.6)
        
#         plt.title('Response Comparison (Even Trials)')
#         plt.ylabel('Response')
        
#         # Add text with SMI value and landmark segment info
#         text_y_pos = max(values) * 1.1
#         plt.text(0.5, text_y_pos, f'SMI = {smi:.3f}', 
#                 horizontalalignment='center', fontsize=12)
        
#         if landmark_segment is not None:
#             segment_names = ["First", "Second", "Third"]
#             segment_name = segment_names[landmark_segment[cell_idx]]
#             plt.text(0.5, text_y_pos * 0.9, f'Near {segment_name} Landmark', 
#                     horizontalalignment='center', fontsize=10)
        
#         plt.tight_layout()
#         fig_examples.append(fig)
    
#     # Return all figures
#     return [fig_hist] + fig_examples

# def plot_SMI_results_BBBB(results, bin_centers, reliable_cells=None, avg_cc=None, cohens_d=None, max_examples=6, exclude_boundary_cm=5):
#     """
#     Plot the results of SMI calculation with example cells, including fitted curves.
#     For three-landmark scenario with multiple potential non-preferred positions.
#     """
#     SMI_values = results['SMI']
#     preferred_positions = results['preferred_positions']
#     non_preferred_positions = results['non_preferred_positions']
#     valid_cells = results['valid_cells']
#     reliable_valid_cells = results['reliable_valid_cells'] if reliable_cells is not None else None
#     odd_profiles = results['odd_profiles']
#     even_profiles = results['even_profiles']
#     preferred_fitted_curves = results['preferred_fitted_curves']
#     non_preferred_fitted_curves = results['non_preferred_fitted_curves']
#     fitting_success = results['fitting_success']
#     min_allowed = results['min_allowed']
#     max_allowed = results['max_allowed']
    
#     # Get landmark information
#     landmark_segment = results.get('landmark_segment', None)
#     landmark_positions = results.get('parameters', {}).get('landmark_positions', None)
#     segment_distances = results.get('parameters', {}).get('segment_distances', [20, 40])
#     all_potential_nonpref = results.get('all_potential_nonpref', None)
    
#     # Use reliable & valid cells
#     valid_indices = np.where(reliable_valid_cells)[0] if reliable_valid_cells is not None else np.where(valid_cells)[0]
#     n_valid = len(valid_indices)
    
#     if n_valid == 0:
#         print(" No valid cells for SMI analysis.")
#         return []
    
#     print(f"    Found {n_valid} valid cells for SMI analysis.")
    
#     # Summary statistics
#     valid_SMI = SMI_values[reliable_valid_cells] if reliable_cells is not None else SMI_values[valid_indices]
#     print(f"    Mean SMI: {np.mean(valid_SMI):.3f}")
#     print(f"    Median SMI: {np.median(valid_SMI):.3f}")
#     print(f"    SMI range: {np.min(valid_SMI):.3f} to {np.max(valid_SMI):.3f}")
#     smi_median = np.median(valid_SMI)
    
#     # Create histogram of SMI values
#     fig_hist = plt.figure(figsize=(10, 6))
#     plt.hist(valid_SMI, bins=20, color='skyblue', edgecolor='black')
#     plt.axvline(0, color='r', linestyle='--', alpha=0.7)
#     plt.xlabel('Spatial Modulation Index (SMI)')
#     plt.ylabel('Count')
#     plt.title(f'Distribution of Spatial Modulation Index (SMI) - median SMI: {smi_median:.3f}')
    
#     # Categorize cells
#     strongly_modulated = np.sum(valid_SMI > 0.5)
#     moderately_modulated = np.sum((valid_SMI > 0.2) & (valid_SMI <= 0.5))
#     weakly_modulated = np.sum((valid_SMI > 0) & (valid_SMI <= 0.2))
#     non_modulated = np.sum(np.abs(valid_SMI) <= 0.05)
#     inverted_modulated = np.sum(valid_SMI < 0)
    
#     print(f"    Strongly modulated (SMI > 0.5): {strongly_modulated} cells ({strongly_modulated/n_valid*100:.1f}%)")
#     print(f"    Moderately modulated (0.2 < SMI ≤ 0.5): {moderately_modulated} cells ({moderately_modulated/n_valid*100:.1f}%)")
#     print(f"    Weakly modulated (0 < SMI ≤ 0.2): {weakly_modulated} cells ({weakly_modulated/n_valid*100:.1f}%)")
#     print(f"    Non-modulated (|SMI| ≤ 0.05): {non_modulated} cells ({non_modulated/n_valid*100:.1f}%)")
#     print(f"    Inverted modulation (SMI < 0): {inverted_modulated} cells ({inverted_modulated/n_valid*100:.1f}%)")
    
#     # Plot example cells
#     # Sort by absolute SMI value to find most modulated cells
#     sorted_indices = valid_indices[np.argsort(np.abs(valid_SMI))[::-1]]
    
#     # Plot top examples
#     n_examples = min(max_examples, n_valid)
#     fig_examples = []
    
#     for i in range(n_examples):
#         cell_idx = sorted_indices[i]
#         avg_cc_value = avg_cc[cell_idx] if isinstance(avg_cc, np.ndarray) else avg_cc
#         cohens_d_value = cohens_d[cell_idx] if isinstance(cohens_d, np.ndarray) else cohens_d
#         smi = SMI_values[cell_idx]
#         pref_pos = preferred_positions[cell_idx]
#         non_pref_pos = non_preferred_positions[cell_idx]
        
#         # Check if we have fitted curves
#         has_pref_fit = fitting_success[cell_idx, 0]
#         has_nonpref_fit = fitting_success[cell_idx, 1]
        
#         # Create figure
#         fig = plt.figure(figsize=(10, 5))
        
#         # Plot odd and even trial profiles with fitted curves
#         plt.subplot(1, 2, 1)
#         plt.plot(bin_centers, odd_profiles[cell_idx], 'b-', alpha=0.6, label='Odd Trials (Training)')
#         plt.plot(bin_centers, even_profiles[cell_idx], 'r-', alpha=0.6, label='Even Trials (Testing)')
        
#         # Add fitted curves if available
#         if has_pref_fit:
#             plt.plot(bin_centers, preferred_fitted_curves[cell_idx], 'g--', linewidth=2, 
#                      label='Preferred Fit (Even)')
        
#         if has_nonpref_fit:
#             plt.plot(bin_centers, non_preferred_fitted_curves[cell_idx], 'm--', linewidth=2, 
#                      label='Non-Preferred Fit (Even)')
        
#         # Mark preferred and non-preferred positions
#         plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7, 
#                    label=f'Preferred Position ({pref_pos:.1f}cm)')
#         plt.axvline(non_pref_pos, color='purple', linestyle='-', alpha=0.7, 
#                    label=f'Non-Preferred Position ({non_pref_pos:.1f}cm)')
        
#         # Highlight the excluded boundary regions
#         plt.axvspan(0, min_allowed, color='red', alpha=0.1)
#         plt.axvspan(max_allowed, bin_centers[-1], color='red', alpha=0.1)
        
#         # # If we have landmark information, mark the three landmarks
#         # if landmark_positions is not None:
#         #     for j, lpos in enumerate(landmark_positions):
#         #         plt.axvline(lpos, color='gray', linestyle='--', alpha=0.5,
#         #                   label=f'Landmark {j+1}' if j == 0 else f'_Landmark {j+1}')
        
#         # Show all potential non-preferred positions
#         if all_potential_nonpref is not None:
#             cell_potentials = all_potential_nonpref[cell_idx]
            
#             # Plot each valid potential non-preferred position
#             for j in range(cell_potentials.shape[0]):
#                 pos, resp, valid = cell_potentials[j]
#                 if valid > 0 and abs(pos - non_pref_pos) > 1:  # Skip the chosen one
#                     plt.axvline(pos, color='purple', linestyle=':', alpha=0.5,
#                                 label=f'Alt. Non-Pref ({pos:.1f}cm)')
        
#         # # Add segment info to title
#         # segment_info = ""
#         # if landmark_segment is not None:
#         #     segment_names = ["First", "Second", "Third", "Fourth"]
#         #     segment_info = f" - Near {segment_names[landmark_segment[cell_idx]]} Landmark"
        
 
#         plt.xlabel('Position (cm)')
#         plt.ylabel('Activity')
#         plt.legend(loc='upper right', fontsize='small')
        
#         # # Plot zoomed view of preferred position
#         # plt.subplot(1, 3, 2)
#         # # Determine zoom window
#         # pref_idx = np.argmin(np.abs(bin_centers - pref_pos))
#         # zoom_start = max(0, pref_idx - 10)
#         # zoom_end = min(len(bin_centers), pref_idx + 10)
        
#         # plt.plot(bin_centers[zoom_start:zoom_end], odd_profiles[cell_idx, zoom_start:zoom_end], 'b-', alpha=0.6, label='Odd')
#         # plt.plot(bin_centers[zoom_start:zoom_end], even_profiles[cell_idx, zoom_start:zoom_end], 'r-', alpha=0.6, label='Even')
        
#         # # Add fitted curve if available
#         # if has_pref_fit:
#         #     plt.plot(bin_centers[zoom_start:zoom_end], preferred_fitted_curves[cell_idx, zoom_start:zoom_end], 
#         #              'g--', linewidth=2, label='Preferred Fit')
        
#         # plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.7)
#         # plt.title('Zoomed Preferred Position')
#         # plt.xlabel('Position (cm)')
#         # plt.ylabel('Activity')
#         # plt.legend(loc='upper right', fontsize='small')
        
#         # Plot response comparison at preferred and non-preferred positions
#         plt.subplot(1, 2, 2)
#         positions = ['Preferred', 'Non-Preferred']
#         values = [results['Rp'][cell_idx], results['Rn'][cell_idx]]
        
#         bars = plt.bar(positions, values, color=['green', 'purple'], alpha=0.6)
#         plt.ylim(0, 1)
        
#         # Add segment info to bar chart
#         if landmark_segment is not None:
#             segment_names = ["First", "Second", "Third", "Fourth"]
#             segment_name = segment_names[landmark_segment[cell_idx]]
#             # plt.title(f'Response Comparison (Even Trials)\nNear {segment_name} Landmark')
#         else:
#             plt.title('Response Comparison (Even Trials)')
            
#         plt.ylabel('Response')
        
#         # Add text with SMI value
#         plt.title(f'SMI = {smi:.3f}', fontsize=12)
        
#         title_text = f'Cell {cell_idx} - SMI: {smi:.3f}'
#         if avg_cc_value is not None:
#             title_text += f', Avg CC: {avg_cc_value:.3f}'
#         if cohens_d_value is not None:
#             title_text += f', Cohen\'s d: {cohens_d_value:.3f}'
#         # title_text += segment_info
        
#         plt.suptitle(title_text, fontsize=20)
        
#         plt.tight_layout()
#         fig_examples.append(fig)

    
#     # Return all figures
#     return [fig_hist] + fig_examples

# def analyze_spatial_modulation(spatial_activity, bin_centers, reliable_cells=None, avg_cc=None, cohens_d=None, segment_distance=55, exclude_boundary_cm=3):
#     """
#     Main function to perform spatial modulation analysis with Gaussian smoothing.
    
#     Parameters:
#     -----------
#     spatial_activity : numpy.ndarray
#         Activity matrix (cells x trials x spatial_bins)
#     bin_centers : numpy.ndarray
#         Centers of spatial bins along the corridor
#     reliable_cells : numpy.ndarray, optional
#         Boolean array indicating reliable cells
#     segment_distance : float
#         Distance between visually identical positions in cm
#     """
#     print("=== SPATIAL MODULATION ANALYSIS WITH GAUSSIAN FITTING ===")
    
#     # 1. Calculate SMI with Gaussian fitting
#     print("Calculating Spatial Modulation Index (SMI) with Gaussian fitting...")
#     smi_results = calculate_SMI(
#         spatial_activity, 
#         bin_centers, 
#         reliable_cells=reliable_cells,
#         segment_distance=segment_distance,
#         exclude_boundary_cm=exclude_boundary_cm
#     )
    
    
#     # 2. Plot SMI results with fitted curves
#     print("")
#     print("Plotting SMI results...")
#     smi_figures = plot_SMI_results(
#         smi_results, bin_centers, reliable_cells, avg_cc, cohens_d, max_examples=50, exclude_boundary_cm=exclude_boundary_cm
#     )
    
#     plt.show()
    
#     print("Analysis complete!")
    
#     return {
#         'smi_results': smi_results,
#         'smi_figures': smi_figures
#     }
    
# def calculate_SMI_BBBB_modified(spatial_activity, bin_centers, reliable_cells, segment_distances=[30, 60, 90], exclude_boundary_cm=0, search_tolerance=3):
#     """
#     Calculate the Spatial Modulation Index (SMI) using cross-validation approach with Gaussian fitting:
#     - Odd trials to find preferred position
#     - Even trials to measure responses at preferred and non-preferred positions
#     - SMI = (Rp - Rn) / (Rp + Rn)
    
#     With four landmarks, searches for peaks around expected distances with tolerance.
#     Where Rp = response at preferred position, Rn = response at non-preferred position (smallest non-zero).
    
#     Parameters:
#     -----------
#     search_tolerance : int, default=3
#         Search window (±bins) around expected landmark distances to find actual peaks
#     """
#     n_cells, n_trials, n_bins = spatial_activity.shape
    
#     # Separate odd and even trials
#     odd_indices = np.arange(0, n_trials, 2)
#     even_indices = np.arange(1, n_trials, 2)

#     # Calculate corridor boundaries
#     min_pos = np.min(bin_centers)
#     max_pos = np.max(bin_centers)
#     corridor_length = max_pos - min_pos
    
#     # Calculate boundary positions in the original coordinate system
#     min_allowed = min_pos + exclude_boundary_cm
#     max_allowed = max_pos - exclude_boundary_cm
#     print(f"  Corridor length: {corridor_length:.2f} and valid position range: {min_allowed:.2f} to {max_allowed:.2f}")
        
#     # Compute response profiles for odd and even trials
#     odd_profiles = np.mean(spatial_activity[:, odd_indices, :], axis=1)
#     even_profiles = np.mean(spatial_activity[:, even_indices, :], axis=1)
    
#     # Initialize arrays to store results
#     SMI_values = np.zeros(n_cells)
#     preferred_positions = np.zeros(n_cells)
#     non_preferred_positions = np.zeros(n_cells)
#     Rp_values = np.zeros(n_cells)
#     Rn_values = np.zeros(n_cells)
#     valid_cells = np.zeros(n_cells, dtype=bool)
    
#     # Store fitted curves for visualization
#     preferred_fitted_curves = np.zeros((n_cells, n_bins))
#     non_preferred_fitted_curves = np.zeros((n_cells, n_bins))
#     fitting_success = np.zeros((n_cells, 2), dtype=bool)  # [0] for preferred, [1] for non-preferred
    
#     # Track which landmark segment the preferred position falls near
#     landmark_segment = np.zeros(n_cells, dtype=int)  # 0=first, 1=second, 2=third, 3=fourth
    
#     # Store all potential non-preferred positions and responses for visualization
#     all_potential_nonpref = np.zeros((n_cells, len(segment_distances) * 2, 3))  # [cell, option, [position, response, valid]]

#     # Count various rejection reasons
#     zero_response_count = 0
#     fitting_failed_count = 0
#     valid_count = 0
#     boundary_peak_count = 0
#     no_nonpref_count = 0
    
#     # Define the four landmarks spaced segment_distances bins apart
#     # Assuming landmarks are evenly distributed across the corridor
#     corridor_fifth = corridor_length / 5
#     landmark_positions = [
#         min_pos + corridor_fifth,         # First landmark
#         min_pos + 2 * corridor_fifth,     # Second landmark  
#         min_pos + 3 * corridor_fifth,     # Third landmark
#         min_pos + 4 * corridor_fifth      # Fourth landmark
#     ]
    
#     for cell in range(n_cells):
#         # Find the initial preferred position from odd trials
#         preferred_idx_odd = np.argmax(odd_profiles[cell])
#         preferred_position_odd = bin_centers[preferred_idx_odd]

#         # Check if the preferred position is within exclude boundary region
#         if preferred_position_odd < min_allowed or preferred_position_odd > max_allowed:
#             # Instead of rejecting, find a peak inside the exclude region
#             boundary_peak_count += 1
            
#             if preferred_position_odd < min_allowed:
#                 # Peak is in lower boundary, search in ALLOWED region
#                 boundary_mask = (bin_centers >= min_allowed) & (bin_centers <= max_allowed)
#             else:
#                 # Peak is in upper boundary, search in ALLOWED region  
#                 boundary_mask = (bin_centers >= min_allowed) & (bin_centers <= max_allowed)

#             boundary_profile = odd_profiles[cell].copy()
#             boundary_profile[~boundary_mask] = 0  # Zero out non-boundary regions
            
#             if np.max(boundary_profile) > 0:
#                 preferred_idx_odd = np.argmax(boundary_profile)
#                 preferred_position_odd = bin_centers[preferred_idx_odd]
#             # If no peak in boundary, continue with original peak

#         # Define a window around the peak in odd trials for more precise localization
#         # Fit a double Gaussian to the odd trials profile to find a smoother peak
#         popt_odd, fit_curve_odd, preferred_position_fitted_odd, _, fit_success_odd = fit_response_profile(
#             bin_centers, odd_profiles[cell], preferred_idx_odd, window_size=5
#         )
        
#         # Find the closest bin to the fitted preferred position
#         preferred_idx_odd_fitted = np.argmin(np.abs(bin_centers - preferred_position_fitted_odd))
        
#         # Now find the corresponding peak in even trials within a window of the refined odd peak
#         start_idx_pref = max(0, preferred_idx_odd_fitted - 1)
#         end_idx_pref = min(n_bins, preferred_idx_odd_fitted + 1)
#         window_profile_pref = even_profiles[cell, start_idx_pref:end_idx_pref]
#         window_max_idx_pref = np.argmax(window_profile_pref)
#         preferred_idx_even_initial = start_idx_pref + window_max_idx_pref
        
#         # Fit a double Gaussian to the even trials around this peak for the final preferred position
#         popt_even_pref, fit_curve_even_pref, preferred_position_fitted_even, peak_response_even, fit_success_even = fit_response_profile(
#             bin_centers, even_profiles[cell], preferred_idx_even_initial, window_size=5
#         )
        
#         # Store the fitting success status
#         fitting_success[cell, 0] = fit_success_even
        
#         # If fitting failed, we can either use the raw peak or skip this cell
#         if not fit_success_even:
#             fitting_failed_count += 1
#             preferred_position_even = bin_centers[preferred_idx_even_initial]
#             Rp = even_profiles[cell, preferred_idx_even_initial]
#         else:
#             # Use the fitted peak position and height
#             preferred_position_even = preferred_position_fitted_even
#             Rp = peak_response_even
#             # Store the fitted curve for visualization
#             preferred_fitted_curves[cell] = fit_curve_even_pref

#         # Determine which landmark segment the preferred position is closest to
#         distances_to_landmarks = [abs(preferred_position_even - pos) for pos in landmark_positions]
#         closest_landmark = np.argmin(distances_to_landmarks)
#         landmark_segment[cell] = closest_landmark
        
#         # Calculate potential non-preferred positions with flexible peak detection
#         # Search around expected landmark distances with tolerance
#         non_preferred_positions_list = []
        
#         # For each expected distance, search for actual peaks in the vicinity
#         for dist in segment_distances:
#             # Search forward direction
#             center_pos_forward = preferred_position_even + dist
#             if min_pos <= center_pos_forward <= max_pos:
#                 # Define search window around expected position
#                 search_start = max(min_pos, center_pos_forward - search_tolerance)
#                 search_end = min(max_pos, center_pos_forward + search_tolerance)
                
#                 # Apply boundary exclusion: ensure search window doesn't include boundary regions
#                 search_start = max(search_start, min_allowed)
#                 search_end = min(search_end, max_allowed)
                
#                 # Skip if search window is entirely in boundary region
#                 if search_start >= search_end:
#                     continue
                
#                 # Find bins within search window
#                 search_mask = (bin_centers >= search_start) & (bin_centers <= search_end)
#                 if np.any(search_mask):
#                     # Find the peak within this search window
#                     search_profile = even_profiles[cell].copy()
#                     search_profile[~search_mask] = 0  # Zero out regions outside search window
                    
#                     if np.max(search_profile) > 0:
#                         peak_idx = np.argmax(search_profile)
#                         peak_pos = bin_centers[peak_idx]
#                         # Double-check that the found peak is not in boundary region
#                         if min_allowed <= peak_pos <= max_allowed:
#                             non_preferred_positions_list.append(peak_pos)
            
#             # Search backward direction
#             center_pos_backward = preferred_position_even - dist
#             if min_pos <= center_pos_backward <= max_pos:
#                 # Define search window around expected position
#                 search_start = max(min_pos, center_pos_backward - search_tolerance)
#                 search_end = min(max_pos, center_pos_backward + search_tolerance)
                
#                 # Apply boundary exclusion: ensure search window doesn't include boundary regions
#                 search_start = max(search_start, min_allowed)
#                 search_end = min(search_end, max_allowed)
                
#                 # Skip if search window is entirely in boundary region
#                 if search_start >= search_end:
#                     continue
                
#                 # Find bins within search window
#                 search_mask = (bin_centers >= search_start) & (bin_centers <= search_end)
#                 if np.any(search_mask):
#                     # Find the peak within this search window
#                     search_profile = even_profiles[cell].copy()
#                     search_profile[~search_mask] = 0  # Zero out regions outside search window
                    
#                     if np.max(search_profile) > 0:
#                         peak_idx = np.argmax(search_profile)
#                         peak_pos = bin_centers[peak_idx]
#                         # Double-check that the found peak is not in boundary region
#                         if min_allowed <= peak_pos <= max_allowed:
#                             non_preferred_positions_list.append(peak_pos)
        
#         # Remove duplicate positions (in case search windows overlap)
#         non_preferred_positions_list = list(set(non_preferred_positions_list))
        
#         # If no valid non-preferred positions found, skip this cell
#         if len(non_preferred_positions_list) == 0:
#             no_nonpref_count += 1
#             valid_cells[cell] = False
#             continue
        
#         # Process each potential non-preferred position
#         non_preferred_responses = []
#         valid_non_preferred_positions = []
#         fitted_non_preferred_curves = []
#         non_preferred_fit_success = []
        
#         # Initialize array to store all potential non-preferred positions for this cell
#         cell_potential_nonpref = np.zeros((len(non_preferred_positions_list), 3))
        
#         for i, non_pref_pos in enumerate(non_preferred_positions_list):
#             # Set default values
#             cell_potential_nonpref[i, 0] = non_pref_pos  # Position
#             cell_potential_nonpref[i, 1] = -1  # Response (invalid)
#             cell_potential_nonpref[i, 2] = 0   # Valid flag (0=invalid)
                
#             # Find the closest bin to this non-preferred position
#             non_preferred_idx_approx = np.argmin(np.abs(bin_centers - non_pref_pos))
            
#             # Get exact response at this position
#             exact_response = even_profiles[cell, non_preferred_idx_approx]
            
#             # Fit a double Gaussian to the non-preferred position response
#             # Use a window to allow finding the precise location
#             start_idx_nonpref = max(0, non_preferred_idx_approx - 5)
#             end_idx_nonpref = min(n_bins, non_preferred_idx_approx + 5)
            
#             if start_idx_nonpref >= end_idx_nonpref:
#                 continue  # Skip if window is invalid
                
#             window_profile_nonpref = even_profiles[cell, start_idx_nonpref:end_idx_nonpref]
#             window_peak_idx = np.argmax(window_profile_nonpref) + start_idx_nonpref
            
#             popt_even_nonpref, fit_curve, non_preferred_position_fitted, non_preferred_resp_fitted, fit_success_nonpref = fit_response_profile(
#                 bin_centers, even_profiles[cell], window_peak_idx, window_size=5
#             )
            
#             # If fitting succeeded, use the fitted values
#             if fit_success_nonpref:
#                 # Calculate the distance from the estimated position to the fitted position
#                 distance_from_target = abs(non_pref_pos - non_preferred_position_fitted)
                
#                 # Only use the fitted position if it's close to our target position
#                 if distance_from_target <= 5:  # Within 5 units of target
#                     non_preferred_pos_final = non_preferred_position_fitted
#                     non_preferred_resp = non_preferred_resp_fitted
#                     fitted_curve_final = fit_curve
#                     fit_success = True
#                 else:
#                     # If fitted position is too far, use the exact response at the target
#                     non_preferred_pos_final = non_pref_pos
#                     non_preferred_resp = exact_response
#                     fitted_curve_final = np.zeros_like(bin_centers)
#                     fit_success = False
#             else:
#                 # If fitting failed, use the exact response at the target
#                 non_preferred_pos_final = non_pref_pos
#                 non_preferred_resp = exact_response
#                 fitted_curve_final = np.zeros_like(bin_centers)
#                 fit_success = False
            
#             # Store this position, its response, and its fitted curve
#             valid_non_preferred_positions.append(non_preferred_pos_final)
#             non_preferred_responses.append(non_preferred_resp)
#             fitted_non_preferred_curves.append(fitted_curve_final)
#             non_preferred_fit_success.append(fit_success)
            
#             # Store for visualization
#             cell_potential_nonpref[i, 0] = non_preferred_pos_final
#             cell_potential_nonpref[i, 1] = non_preferred_resp
#             cell_potential_nonpref[i, 2] = 1  # Valid flag
        
#         # Store all potential non-preferred positions for this cell
#         if cell < all_potential_nonpref.shape[0]:
#             all_potential_nonpref[cell, :len(cell_potential_nonpref)] = cell_potential_nonpref
        
#         # If no valid non-preferred positions, skip this cell
#         if len(non_preferred_responses) == 0:
#             valid_cells[cell] = False
#             continue
        
#         # Find the non-preferred position with the SMALLEST NON-ZERO response
#         non_zero_responses = [(i, resp) for i, resp in enumerate(non_preferred_responses) if resp > 0.05]
        
#         if len(non_zero_responses) == 0:
#             # If all responses are effectively zero, use the smallest positive response
#             min_resp_idx = np.argmin([abs(r) for r in non_preferred_responses])
#             Rn = non_preferred_responses[min_resp_idx]
#             non_preferred_position_even = valid_non_preferred_positions[min_resp_idx]
#         else:
#             # Find the smallest non-zero response
#             min_nonzero_idx, Rn = min(non_zero_responses, key=lambda x: x[1])
#             non_preferred_position_even = valid_non_preferred_positions[min_nonzero_idx]
            
#         # Store the fitted curve for the chosen non-preferred position
#         chosen_idx = min_resp_idx if len(non_zero_responses) == 0 else min_nonzero_idx
#         if chosen_idx < len(non_preferred_fit_success) and non_preferred_fit_success[chosen_idx]:
#             non_preferred_fitted_curves[cell] = fitted_non_preferred_curves[chosen_idx]
#             fitting_success[cell, 1] = True
#         else:
#             fitting_success[cell, 1] = False
        
#         # Calculate SMI
#         if Rp + Rn > 0:  # Avoid division by zero
#             SMI = (Rp - Rn) / (Rp + Rn)
#             valid_count += 1
#         else:
#             # Skip cells where sum is zero
#             SMI = 0
#             zero_response_count += 1
#             valid_cells[cell] = False
#             continue
        
#         # Store results - use the adjusted positions from even trials
#         SMI_values[cell] = SMI
#         preferred_positions[cell] = preferred_position_even
#         non_preferred_positions[cell] = non_preferred_position_even
#         Rp_values[cell] = Rp
#         Rn_values[cell] = Rn
#         valid_cells[cell] = True
    
#     print(f"Number of total cells: {n_cells} and number of valid cells: {np.sum(valid_cells)}")      

#     # find cells that are true for both reliable_cells and valid_cells
#     if reliable_cells is not None:
#         reliable_valid_cells = np.logical_and(valid_cells, reliable_cells)
#     else:
#         reliable_valid_cells = valid_cells
    
#     # Print summary statistics
#     print(f"\nSMI calculation summary:")
#     print(f"  Total cells: {n_cells}")
#     print(f"  Reliable&Valid cells: {np.sum(reliable_valid_cells)} ({np.sum(reliable_valid_cells)/n_cells*100:.1f}%)")
#     print(f"  Peaks found in boundary region: {boundary_peak_count} ({boundary_peak_count/n_cells*100:.1f}%)")
#     print(f"  Rejected - no valid non-preferred positions: {no_nonpref_count} ({no_nonpref_count/n_cells*100:.1f}%)")
#     print(f"  Rejected - zero response sum: {zero_response_count} ({zero_response_count/n_cells*100:.1f}%)")
#     print(f"  Fitting failed (but used raw peak instead): {fitting_failed_count} ({fitting_failed_count/n_cells*100:.1f}%)")
    
#     # Calculate segment statistics for four landmarks
#     if np.sum(valid_cells) > 0:
#         segment_counts = np.zeros(4, dtype=int)
#         for i in range(4):
#             segment_counts[i] = np.sum(landmark_segment[valid_cells] == i)
        
#         print(f"\nPreferred position by landmark segment (4 landmarks):")
#         for i in range(4):
#             print(f"  Landmark {i+1}: {segment_counts[i]} cells ({segment_counts[i]/np.sum(valid_cells)*100:.1f}%)")
        
#     # Create result dictionary
#     results = {
#         'SMI': SMI_values,
#         'preferred_positions': preferred_positions,
#         'non_preferred_positions': non_preferred_positions,
#         'Rp': Rp_values,
#         'Rn': Rn_values,
#         'odd_profiles': odd_profiles,
#         'even_profiles': even_profiles,
#         'preferred_fitted_curves': preferred_fitted_curves,
#         'non_preferred_fitted_curves': non_preferred_fitted_curves,
#         'fitting_success': fitting_success,
#         'landmark_segment': landmark_segment,
#         'min_allowed': min_allowed,
#         'max_allowed': max_allowed,
#         'valid_cells': valid_cells,
#         'reliable_valid_cells': reliable_valid_cells if reliable_cells is not None else None,
#         'all_potential_nonpref': all_potential_nonpref,
#         'parameters': {
#             'segment_distances': segment_distances,
#             'exclude_boundary_cm': exclude_boundary_cm,
#             'n_cells': n_cells,
#             'n_trials': n_trials,
#             'n_bins': n_bins,
#             'corridor_length': corridor_length,
#             'min_pos': min_pos,
#             'max_pos': max_pos,
#             'landmark_positions': landmark_positions
#         }
#     }
    
#     return results
    
# def analyze_spatial_modulation_BBBB(spatial_activity, bin_centers, reliable_cells=None, avg_cc=None, cohens_d=None, segment_distance=55, exclude_boundary_cm=3):
#     """
#     Main function to perform spatial modulation analysis with Gaussian smoothing.
    
#     Parameters:
#     -----------
#     spatial_activity : numpy.ndarray
#         Activity matrix (cells x trials x spatial_bins)
#     bin_centers : numpy.ndarray
#         Centers of spatial bins along the corridor
#     reliable_cells : numpy.ndarray, optional
#         Boolean array indicating reliable cells
#     segment_distance : float
#         Distance between visually identical positions in cm
#     """
#     print("=== SPATIAL MODULATION ANALYSIS WITH GAUSSIAN FITTING ===")
    
#     # 1. Calculate SMI with Gaussian fitting
#     print("Calculating Spatial Modulation Index (SMI) with Gaussian fitting...")
    
#     # smi_results = calculate_SMI_BBBB(
#     #     spatial_activity, 
#     #     bin_centers, 
#     #     reliable_cells=reliable_cells,
#     #     segment_distances=[segment_distance, segment_distance * 2],
#     #     exclude_boundary_cm=exclude_boundary_cm
#     # )
    
#     smi_results = calculate_SMI_BBBB_modified(
#         spatial_activity, 
#         bin_centers, 
#         reliable_cells=reliable_cells,
#         segment_distances=[segment_distance, segment_distance * 2, segment_distance * 3],
#         exclude_boundary_cm=exclude_boundary_cm,
#         search_tolerance=3
#     )
    
#     # 2. Plot SMI results with fitted curves
#     print("")
#     print("Plotting SMI results...")
#     smi_figures = plot_SMI_results_BBBB(
#         smi_results, bin_centers, reliable_cells, avg_cc, cohens_d, max_examples=20, exclude_boundary_cm=exclude_boundary_cm
#     )
    
#     plt.show()
    
#     print("Analysis complete!")
    
#     return {
#         'smi_results': smi_results,
#         'smi_figures': smi_figures
#     }

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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
                             cohens_d=None, max_examples=10):
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
    
    # Plot example cells
    sorted_indices = plot_indices[np.argsort(np.abs(valid_SMI))[::-1]]
    n_examples = min(max_examples, n_valid)
    fig_examples = []
    
    for i in range(n_examples):
        cell_idx = sorted_indices[i]
        smi = SMI_values[cell_idx]
        pref_pos = preferred_positions[cell_idx]
        nonpref_pos = non_preferred_positions[cell_idx]
        
        fig = plt.figure(figsize=(15, 5))
        
        # Main plot with all information
        plt.subplot(1, 3, 1)
        plt.plot(bin_centers, training_profiles[cell_idx], 'b-', alpha=0.7, 
                label='Training Trials', linewidth=1.5)
        plt.plot(bin_centers, testing_profiles[cell_idx], 'r-', alpha=0.7, 
                label='Testing Trials', linewidth=1.5)
        
        # Add fitted curves if available
        if fitting_success[cell_idx, 0]:
            plt.plot(bin_centers, preferred_fitted_curves[cell_idx], 'g--', 
                    linewidth=2, label='Preferred Fit')
        
        if fitting_success[cell_idx, 1]:
            plt.plot(bin_centers, non_preferred_fitted_curves[cell_idx], 'm--', 
                    linewidth=2, label='Non-Preferred Fit')
        
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
        
        # Highlight excluded boundary regions
        plt.axvspan(bin_centers[0], min_allowed, color='red', alpha=0.1, label='Excluded')
        plt.axvspan(max_allowed, bin_centers[-1], color='red', alpha=0.1, label='_nolegend_')
        
        plt.xlabel('Position (cm)')
        plt.ylabel('Activity')
        plt.title(f'Cell {cell_idx} - SMI: {smi:.3f}')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True, alpha=0.3)
        
        # Zoomed view of preferred position
        plt.subplot(1, 3, 2)
        pref_idx = np.argmin(np.abs(bin_centers - pref_pos))
        zoom_start = max(0, pref_idx - 15)
        zoom_end = min(len(bin_centers), pref_idx + 15)
        
        plt.plot(bin_centers[zoom_start:zoom_end], 
                training_profiles[cell_idx, zoom_start:zoom_end], 
                'b-', alpha=0.7, label='Training')
        plt.plot(bin_centers[zoom_start:zoom_end], 
                testing_profiles[cell_idx, zoom_start:zoom_end], 
                'r-', alpha=0.7, label='Testing')
        
        if fitting_success[cell_idx, 0]:
            plt.plot(bin_centers[zoom_start:zoom_end], 
                    preferred_fitted_curves[cell_idx, zoom_start:zoom_end], 
                    'g--', linewidth=2, label='Fit')
        
        plt.axvline(pref_pos, color='green', linestyle='-', alpha=0.8)
        plt.title('Preferred Position (Zoomed)')
        plt.xlabel('Position (cm)')
        plt.ylabel('Activity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Response comparison
        plt.subplot(1, 3, 3)
        positions = ['Preferred', 'Non-Preferred']
        values = [results['Rp'][cell_idx], results['Rn'][cell_idx]]
        colors = ['green', 'purple']
        
        bars = plt.bar(positions, values, color=colors, alpha=0.7)
        plt.title(f'Response Comparison\nSMI = {smi:.3f}')
        plt.ylabel('Response')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom')
        
        # Add additional info as text
        info_text = f'Distance: {abs(nonpref_pos - pref_pos):.1f}cm'
        if avg_cc is not None:
            info_text += f'\nAvg CC: {avg_cc[cell_idx]:.3f}'
        if cohens_d is not None:
            info_text += f'\nCohen\'s d: {cohens_d[cell_idx]:.3f}'
        
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        fig_examples.append(fig)
    
    return [fig_hist] + fig_examples


def analyze_spatial_modulation_improved(spatial_activity, bin_centers, reliable_cells=None, 
                                       avg_cc=None, cohens_d=None, segment_distance=55, 
                                       exclude_boundary_cm=15, exclude_start_cm=None, exclude_end_cm=None,
                                       smoothing_sigma=1.0):
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
        max_examples=6
    )
    
    plt.show()
    
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